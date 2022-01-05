import logging
import os
import numpy as np
from sklearn.metrics import pairwise
import functools
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib import metrics, slim
from tensorflow.contrib.metrics import streaming_mean

from . import nn
from . import weight_norm as wn
from .framework import assert_shape, HyperparamVariables
from . import string_utils


LOG = logging.getLogger('main')


class Model:
    DEFAULT_HYPERPARAMS = {
        # loss weight hyper-parameters
        # *** wd_coefficient is coeffs without multiplying lr
        'max_consistency_cost': 1.365,
        'wd_coefficient': .00005,

        # Consistency hyper-parameters
        'ema_consistency': True,
        'ema_decay_during_rampup': 0.99,
        'ema_decay_after_rampup': 0.999,

        # Optimizer hyper-parameters
        'max_learning_rate': 0.003,
        'adam_beta_1_before_rampdown': 0.9,
        'adam_beta_1_after_rampdown': 0.5,
        'adam_beta_2_during_rampup': 0.99,
        'adam_beta_2_after_rampup': 0.999,
        'adam_epsilon': 1e-8,

        # Architecture hyper-parameters
        'input_noise': 0.15,
        'student_dropout_probability': 0.5,
        'teacher_dropout_probability': 0.5,

        # Training schedule
        'rampup_length': 58606,
        'rampdown_length': 36629,
        'training_length': 219771,

        # Input augmentation
        'flip_horizontally': False,
        'translate': True,

        # Whether to scale each input image to mean=0 and std=1 per channel
        # Use False if input is already normalized in some other way
        'normalize_input': True,

        # Output schedule
        'print_span': 100,
        'evaluation_span': 2000,
    }

    # pylint: disable=too-many-instance-attributes
    def __init__(self, batch_size, context_size, n_labeled_per_batch, run_context=None):
        if run_context is not None:
            self.training_log = run_context.create_train_log('training')
            self.validation_log = run_context.create_train_log('validation')
            self.checkpoint_path = os.path.join(run_context.transient_dir, 'checkpoint')
            self.tensorboard_path = os.path.join(run_context.result_dir, 'tensorboard')

        with tf.name_scope("placeholders"):
            self.images = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='images')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.embedding_labels = tf.placeholder(dtype=tf.float32, shape=(None, context_size), name='embedding_labels')
            self.gamma = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='gamma')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        tf.add_to_collection("init_in_init", self.global_step)
        self.hyper = HyperparamVariables(self.DEFAULT_HYPERPARAMS)
        for var in self.hyper.variables.values():
            tf.add_to_collection("init_in_init", var)

        self.batch_size = batch_size
        self.context_size = context_size
        self.output_dim = 128
        self.r1 = 0.5
        self.r2 = 0.2
        self.sigma = 1.
        self.n_labeled_per_batch = n_labeled_per_batch
        # *** new hyper-parameters
        self.wd_coefficient = self.hyper['wd_coefficient']

        with tf.name_scope("ramps"):
            sigmoid_rampup_value = sigmoid_rampup(self.global_step, self.hyper['rampup_length'])
            sigmoid_rampdown_value = sigmoid_rampdown(self.global_step,
                                                      self.hyper['rampdown_length'],
                                                      self.hyper['training_length'])
            self.learning_rate = tf.multiply(sigmoid_rampup_value * sigmoid_rampdown_value,
                                             self.hyper['max_learning_rate'],
                                             name='learning_rate')
            self.adam_beta_1 = tf.add(sigmoid_rampdown_value * self.hyper['adam_beta_1_before_rampdown'],
                                      (1 - sigmoid_rampdown_value) * self.hyper['adam_beta_1_after_rampdown'],
                                      name='adam_beta_1')

            self.cons_coefficient = tf.multiply(sigmoid_rampup_value,
                                                self.hyper['max_consistency_cost'],
                                                name='consistency_coefficient')

            step_rampup_value = step_rampup(self.global_step, self.hyper['rampup_length'])
            self.adam_beta_2 = tf.add((1 - step_rampup_value) * self.hyper['adam_beta_2_during_rampup'],
                                      step_rampup_value * self.hyper['adam_beta_2_after_rampup'],
                                      name='adam_beta_2')
            self.ema_decay = tf.add((1 - step_rampup_value) * self.hyper['ema_decay_during_rampup'],
                                    step_rampup_value * self.hyper['ema_decay_after_rampup'],
                                    name='ema_decay')

        tower_args = dict(inputs=self.images,
                          is_training=self.is_training,
                          input_noise=self.hyper['input_noise'],
                          normalize_input=self.hyper['normalize_input'],
                          flip_horizontally=self.hyper['flip_horizontally'],
                          translate=self.hyper['translate'],
                          context_size=self.context_size,
                          output_dim=self.output_dim)

        self.f_logits_1, self.student_bx128, self.embedding_bxm = \
            tower(**tower_args, dropout_probability=self.hyper['student_dropout_probability'])
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Take only first call to update batch norm.

        # self.f_logits_2 = \
        #     tower(**tower_args, dropout_probability=self.hyper['student_dropout_probability'])

        ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
        ema_op = ema.apply(model_vars())
        ema_getter = functools.partial(getter_ema, ema)
        self.f_logits_ema, self.teacher_bx128, _ = \
            tower(**tower_args, dropout_probability=self.hyper['teacher_dropout_probability'], getter=ema_getter)
        self.f_logits_ema = tf.stop_gradient(self.f_logits_ema)
        self.f_logits_ema_norm = tf.nn.l2_normalize(self.f_logits_ema, axis=-1)
        # self.teacher_bx128 = tf.stop_gradient(self.teacher_bx128)
        self.teacher_bx128_norm = tf.nn.l2_normalize(self.teacher_bx128, axis=-1)
        self.teacher_bx128_norm = tf.stop_gradient(self.teacher_bx128_norm)

        # Weight Decay for Adam Optimizer
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - self.wd_coefficient)) for v in model_vars('classify') if 'kernel' in v.name])
        # Weight Decay for SGD Optimizer L2 regularization
        # self.wd_cost = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)

        with tf.name_scope("objectives"):
            # error of labeled samples
            _, self.errors_f_mt = errors(self.f_logits_ema, self.labels)
            self.mean_error_f, self.errors_f = errors(self.f_logits_1, self.labels)

            self.mean_class_cost_f, self.class_eval_costs_f = classification_costs_f(self.f_logits_1, self.labels)

            self.mean_cons_cost_mt_f = \
                consistency_costs_f(self.f_logits_1, self.f_logits_ema, self.cons_coefficient)

            initial_weight = tf.cond(tf.to_float(self.global_step) < 30000, lambda: 10e-8, lambda: 10e-7)
            self.embed_coefficient = tf.multiply(initial_weight, tf.to_float(self.global_step))
            self.embedding_logits = cal_embedding_logits(self.embedding_bxm, self.gamma)
            self.mean_embed_cost_f = embedding_costs_f(self.embedding_logits, self.embedding_labels, self.embed_coefficient)

            self.mean_total_cost_pi = self.mean_class_cost_f + self.mean_cons_cost_mt_f + self.mean_embed_cost_f
            self.mean_total_cost_mt = self.mean_class_cost_f + self.mean_cons_cost_mt_f + self.mean_embed_cost_f

            self.cost_to_be_minimized = tf.cond(self.hyper['ema_consistency'],
                                                lambda: self.mean_total_cost_mt,
                                                lambda: self.mean_total_cost_pi)

        with tf.name_scope("train_step"):
            # post_ops.append(ema_op)
            self.train_step_op = nn.adam_optimizer(self.cost_to_be_minimized,
                                                   self.global_step,
                                                   learning_rate=self.learning_rate,
                                                   beta1=self.adam_beta_1,
                                                   beta2=self.adam_beta_2,
                                                   epsilon=self.hyper['adam_epsilon'])
            with tf.control_dependencies([self.train_step_op]):
                self.train_step_op = tf.group(*post_ops)

        # return a dict
        self.training_control = training_control(self.global_step,
                                                 self.hyper['print_span'],
                                                 self.hyper['evaluation_span'],
                                                 self.hyper['training_length'])

        # *** need to be fixed
        self.training_metrics = {
            "learning_rate": self.learning_rate,
            "adam_beta_1": self.adam_beta_1,
            "adam_beta_2": self.adam_beta_2,
            "ema_decay": self.ema_decay,
            "cons_coefficient_f": self.cons_coefficient,
            "embed_coefficient_f": self.embed_coefficient,
            "train/error/f": self.mean_error_f,
            "train/class_cost/f": self.mean_class_cost_f,
            "train/cons_cost/f": self.mean_cons_cost_mt_f,
            # "train/total_cost/pi": self.mean_total_cost_pi,
            "train/embed_cost/f": self.mean_embed_cost_f,
            "train/total_cost/mt": self.mean_total_cost_mt,
        }

        # *** need to be fixed
        with tf.variable_scope("validation_metrics") as metrics_scope:
            self.metric_values, self.metric_update_ops = metrics.aggregate_metric_map({
                "eval/error/f_mt": streaming_mean(self.errors_f_mt),
                "eval/error/f": streaming_mean(self.errors_f),
                "eval/class_cost/f": streaming_mean(self.class_eval_costs_f),
            })
            metric_variables = slim.get_local_variables(scope=metrics_scope.name)
            self.metric_init_op = tf.variables_initializer(metric_variables)

        self.result_formatter = string_utils.DictFormatter(
            order=["error/f", "class_cost/f", "cons_cost/f", "embed_cost/f"],
            default_format='{name}: {value:>10.6f}',
            separator=",  ")
        self.result_formatter.add_format('error', '{name}: {value:>6.2%}')

        with tf.name_scope("initializers"):
            init_init_variables = tf.get_collection("init_in_init")
            train_init_variables = [
                var for var in tf.global_variables() if var not in init_init_variables
            ]
            self.init_init_op = tf.variables_initializer(init_init_variables)
            self.train_init_op = tf.variables_initializer(train_init_variables)

        self.saver = tf.train.Saver(max_to_keep=1)
        self.session = tf.Session()
        self.run(self.init_init_op)

        # add collection to restore model ***
        tf.add_to_collection('images', self.images)
        tf.add_to_collection('is_training', self.is_training)
        tf.add_to_collection('output_f', self.f_logits_1)
        tf.add_to_collection('output_f_mt', self.f_logits_ema)
        tf.add_to_collection('embedding', self.embedding_bxm)

    def __setitem__(self, key, value):
        self.hyper.assign(self.session, key, value)

    def __getitem__(self, key):
        return self.hyper.get(self.session, key)

    def train(self, training_batches, evaluation_batches_fn, context_training):
        self.run(self.train_init_op)
        LOG.info("Model variables initialized")
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()
        iter = 0
        for batch in training_batches:
            iter += 1
            teacher_bx128, teacher_bx10 = self.run([self.f_logits_ema_norm, self.f_logits_ema],
                                                   self.feed_dict(batch))
            teacher_mx128, teacher_mx10 = self.run([self.f_logits_ema_norm, self.f_logits_ema],
                                                   self.feed_dict(context_training))

            embedding_labels, gamma = cal_embedding_target(self.r1, self.r2, self.sigma,
                                                           teacher_bx128, teacher_bx10, teacher_mx128, teacher_mx10,
                                                           batch['z'], context_training['z'], iter)

            results, _ = self.run([self.training_metrics, self.train_step_op],
                                  self.feed_dict_train(batch, embedding_labels, gamma))
            step_control = self.get_training_control()
            self.training_log.record(step_control['step'], {**results, **step_control})
            if step_control['time_to_print']:
                LOG.info("step %5d:   %s", step_control['step'], self.result_formatter.format_dict(results))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                self.evaluate(evaluation_batches_fn)
                self.save_checkpoint()
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()

    def evaluate(self, evaluation_batches_fn):
        self.run(self.metric_init_op)
        for batch in evaluation_batches_fn():
            self.run(self.metric_update_ops,
                     feed_dict=self.feed_dict(batch, is_training=False))
        step = self.run(self.global_step)
        results = self.run(self.metric_values)
        self.validation_log.record(step, results)
        LOG.info("step %5d:   %s", step, self.result_formatter.format_dict(results))

    def get_training_control(self):
        return self.session.run(self.training_control)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def feed_dict(self, batch, is_training=False):
        return {
            self.images: batch['x'],
            self.labels: batch['y'],
            self.is_training: is_training
        }

    def feed_dict_train(self, batch, embedding_labels, gamma, is_training=True):
        return {
            self.images: batch['x'],
            self.labels: batch['y'],
            self.embedding_labels: embedding_labels,
            self.gamma: gamma,
            self.is_training: is_training
        }

    def save_checkpoint(self):
        path = self.saver.save(self.session, self.checkpoint_path, global_step=self.global_step)
        LOG.info("Saved checkpoint: %r", path)

    def save_tensorboard_graph(self):
        writer = tf.summary.FileWriter(self.tensorboard_path)
        writer.add_graph(self.session.graph)
        return writer.get_logdir()


Hyperparam = namedtuple("Hyperparam", ['tensor', 'getter', 'setter'])


def model_vars(scope=None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def getter_ema(ema, getter, name, *args, **kwargs):
    """Exponential moving average getter for variable scopes.
    Args:
        ema: ExponentialMovingAverage object, where to get variable moving averages.
        getter: default variable scope getter.
        name: variable name.
        *args: extra args passed to default getter.
        **kwargs: extra args passed to default getter.
    Returns:
        If found the moving average variable, otherwise the default variable.
    """
    var = getter(name, *args, **kwargs)
    ema_var = ema.average(var)
    return ema_var if ema_var else var


def training_control(global_step, print_span, evaluation_span, max_step, name=None):
    with tf.name_scope(name, "training_control"):
        return {
            "step": global_step,
            "time_to_print": tf.equal(tf.mod(global_step, print_span), 0),
            "time_to_evaluate": tf.equal(tf.mod(global_step, evaluation_span), 0),
            "time_to_stop": tf.greater_equal(global_step, max_step),
        }


def step_rampup(global_step, rampup_length):
    result = tf.cond(global_step < rampup_length,
                     lambda: tf.constant(0.0),
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="step_rampup")


def sigmoid_rampup(global_step, rampup_length):
    global_step = tf.to_float(global_step)
    rampup_length = tf.to_float(rampup_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, global_step) / rampup_length
        return tf.exp(-5.0 * phase * phase)

    result = tf.cond(global_step < rampup_length, ramp, lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampup")


def sigmoid_rampdown(global_step, rampdown_length, training_length):
    global_step = tf.to_float(global_step)
    rampdown_length = tf.to_float(rampdown_length)
    training_length = tf.to_float(training_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, training_length - global_step) / rampdown_length
        return tf.exp(-12.5 * phase * phase)

    result = tf.cond(global_step >= training_length - rampdown_length,
                     ramp,
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampdown")


def tower(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          flip_horizontally,
          translate,
          context_size,
          output_dim,
          getter=None):
    conv_args = dict(kernel_size=3, padding='same')
    bn_args = dict(training=is_training, momentum=0.999)
    with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
        net = inputs
        assert_shape(net, [None, 32, 32, 3])

        net = tf.cond(normalize_input,
                      lambda: slim.layer_norm(net,
                                              scale=False,
                                              center=False,
                                              scope='normalize_inputs'),
                      lambda: net)
        assert_shape(net, [None, 32, 32, 3])

        # Data Augmentation
        net = nn.flip_randomly(net,
                               horizontally=flip_horizontally,
                               vertically=False,
                               is_training=is_training,
                               name='random_flip')
        net = tf.cond(translate,
                      lambda: nn.random_translate(net, scale=2, is_training=is_training, name='random_translate'),
                      lambda: net)
        net = nn.gaussian_noise(net, scale=input_noise, is_training=is_training, name='gaussian_noise')

        net = tf.layers.conv2d(net, 128, **conv_args)
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        net = tf.layers.conv2d(net, 128, **conv_args)
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        net = tf.layers.conv2d(net, 128, **conv_args)
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.layers.dropout(net, dropout_probability, training=is_training)
        assert_shape(net, [None, 16, 16, 128])

        net = tf.layers.conv2d(net, 256, **conv_args)
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        net = tf.layers.conv2d(net, 256, **conv_args)
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        net = tf.layers.conv2d(net, 256, **conv_args)
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.layers.dropout(net, dropout_probability, training=is_training)
        assert_shape(net, [None, 8, 8, 256])

        net = tf.layers.conv2d(net, 512, kernel_size=3, padding='valid')
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        assert_shape(net, [None, 6, 6, 512])
        net = tf.layers.conv2d(net, 256, kernel_size=1, padding='same')
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        net = tf.layers.conv2d(net, 128, kernel_size=1, padding='same')
        assert_shape(net, [None, 6, 6, 128])

        # net = tf.layers.batch_normalization(net, **bn_args)
        # net = nn.lrelu(net)

        b_6x6x128 = tf.reshape(net, [tf.shape(net)[0], 6*6*128])
        assert_shape(b_6x6x128, [None, 6*6*128])

        net = tf.reduce_mean(net, [1, 2])  # (bs, 6, 6, 128) -> (bs, 128)
        bx128 = net
        assert_shape(net, [None, 128])

        net2 = tf.layers.dense(b_6x6x128, 1024)
        net2 = tf.layers.batch_normalization(net2, **bn_args)
        # net2 = nn.lrelu(net2)
        assert_shape(net2, [None, 1024])

        net2 = tf.layers.dense(net2, context_size)
        embedding_bxm = net2
        assert_shape(net2, [None, context_size])

        net2 = tf.layers.dense(net2, output_dim)
        to_concat_bxd = net2
        assert_shape(net2, [None, output_dim])

        # concatenation and prediction
        net = tf.concat([bx128, to_concat_bxd], axis=1)
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        assert_shape(net, [None, 128+output_dim])

        f_logits = tf.layers.dense(net, 10)
        assert_shape(f_logits, [None, 10])

        return f_logits, bx128, embedding_bxm


def errors(logits, labels, name=None):
    """Compute error mean and whether each labeled example is erroneous
    Assume unlabeled examples have label == -1.
    Compute the mean error over labeled examples.
    Mean error is NaN if there are no labeled examples.
    Note that unlabeled examples are treated differently in cost calculation.
    """
    with tf.name_scope(name, "errors") as scope:
        applicable = tf.not_equal(labels, -1)
        labels = tf.boolean_mask(labels, applicable)
        logits = tf.boolean_mask(logits, applicable)
        predictions = tf.argmax(logits, -1)
        labels = tf.cast(labels, tf.int64)
        per_sample = tf.to_float(tf.not_equal(predictions, labels))
        mean = tf.reduce_mean(per_sample, name=scope)
        return mean, per_sample


def classification_costs_f(logits, labels, name=None):
    """Compute classification cost mean and classification cost per sample
    Assume unlabeled examples have label == -1. For unlabeled examples, cost == 0.
    Compute the mean over all examples.
    Note that unlabeled examples are treated differently in error calculation.
    """
    with tf.name_scope(name, "classification_costs_f") as scope:
        applicable = tf.not_equal(labels, -1)

        # Change -1s to zeros to make cross-entropy computable
        labels = tf.where(applicable, labels, tf.zeros_like(labels))

        # This will now have incorrect values for unlabeled examples
        per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # Retain costs only for labeled
        per_sample = tf.where(applicable, per_sample, tf.zeros_like(per_sample))

        # Take mean over all examples, not just labeled examples.
        labeled_sum = tf.reduce_sum(per_sample)
        total_count = tf.to_float(tf.shape(per_sample)[0])
        mean = tf.div(labeled_sum, total_count, name=scope)

        return mean, per_sample


def consistency_costs_f(logits1, logits2, cons_coefficient, name=None):
    """Calculate the consistency loss between class prediction f.
    """
    with tf.name_scope(name, "consistency_costs_f") as scope:
        assert_shape(logits1, [None, 10])
        assert_shape(logits2, [None, 10])
        assert_shape(cons_coefficient, [])

        logits1 = tf.nn.softmax(logits1)
        logits2 = tf.nn.softmax(logits2)

        per_sample = tf.reduce_mean((logits1 - logits2) ** 2, axis=-1)
        mean_cost = tf.multiply(tf.reduce_mean(per_sample), cons_coefficient, name=scope)
        assert_shape(mean_cost, [])
        return mean_cost


def adj_matrix(bx128, mx128, sigma=1):
    gamma = 1/(2*sigma**2)
    dist = pairwise.rbf_kernel(bx128, mx128, gamma=gamma)
    return dist


def cal_embedding_target(r1, r2, sigma, teacher_bx128, teacher_bx10, teacher_mx128, teacher_mx10, true_batch_label, true_context_label, iter):
    """
    Calculate the embedding labels
    """
    bs = teacher_bx10.shape[0]
    context_size = teacher_mx10.shape[0]

    # p < r1, positive pair sampling,
    # else, negative pair sampling
    gamma = np.random.rand(bs) < r1
    # pos_pair_idxs = np.squeeze(np.argwhere(gamma))
    # neg_pair_idxs = np.squeeze(np.argwhere(~gamma))
    gamma_reshape = (gamma * 2 - 1.).reshape(-1, 1)

    # p < r2, sampling based on Graph A,
    # else, sampling based on pseudo labels
    beta = np.random.rand(bs) < r2
    graphA_idxs = np.squeeze(np.argwhere(beta))
    pseudo_label_idxs = np.squeeze(np.argwhere(~beta))

    # calculate bipartite Graph A: b x m
    A = adj_matrix(teacher_bx128, teacher_mx128, sigma)

    # if iter % 200 == 0:
    #     # testing
    #     with open("testing_res_norm_5NN.txt", "a") as f:
    #         f.write(str(teacher_bx128.shape) + '\n')
    #         f.write(str(teacher_mx128.shape) + '\n')
    #         f.write(str(teacher_bx128) + '\n')
    #         f.write(str(teacher_mx128) + '\n')
    #         f.write(str(np.sum((teacher_bx128[0] - teacher_mx128) ** 2, axis=1).shape) + '\n')
    #         f.write(str(np.sum((teacher_bx128[0] - teacher_mx128) ** 2, axis=1)) + '\n')
    #         f.write(str(A[0]) + '\n')
    #
    #         true_same_sum = 0
    #         true_diff_sum = 0
    #         for i in range(bs):
    #             A_max_val_idxs = np.argsort(-A[i])[:10]
    #             true_same_sum += np.sum(true_context_label[A_max_val_idxs] == true_batch_label[i])
    #             A_min_val_idxs = np.argsort(A[i])[:100]
    #             true_diff_sum += np.sum(true_context_label[A_min_val_idxs] != true_batch_label[i])
    #         f.write('true_same_sum: ' + str(true_same_sum) + '\n')
    #         f.write('true_diff_sum: ' + str(true_diff_sum) + '\n')

    embedding_labels = np.zeros(bs, dtype=int)
    for i, item_gamma in enumerate(gamma[graphA_idxs]):
        # random walk
        if item_gamma:
            # A[graphA_idxs][i]'s shape (m, )
            A_min_val_idxs = np.argsort(A[graphA_idxs[i]])[:495]
            A[graphA_idxs[i]][A_min_val_idxs] = 0.
            embedding_labels[graphA_idxs[i]] = np.random.choice(context_size, p=A[graphA_idxs[i]] / np.sum(A[graphA_idxs[i]]))
        # uniformly sampling
        else:
            A_min_val_idxs = np.argsort(A[graphA_idxs[i]])[:100]
            embedding_labels[graphA_idxs[i]] = np.random.choice(A_min_val_idxs)

    teacher_b_label = np.argmax(teacher_bx10, axis=1)
    teacher_m_label = np.argmax(teacher_mx10, axis=1)
    for i, item_gamma in enumerate(gamma[pseudo_label_idxs]):
        # same labels
        if item_gamma:
            same_label_idxs = np.squeeze(np.argwhere(teacher_m_label == teacher_b_label[pseudo_label_idxs[i]]))
            if same_label_idxs.size == 0:
                embedding_labels[pseudo_label_idxs[i]] = context_size
            else:
                embedding_labels[pseudo_label_idxs[i]] = np.random.choice(same_label_idxs)
        else:
            diff_label_idxs = np.squeeze(np.argwhere(teacher_m_label != teacher_b_label[pseudo_label_idxs[i]]))
            if diff_label_idxs.size == 0:
                embedding_labels[pseudo_label_idxs[i]] = context_size
            else:
                embedding_labels[pseudo_label_idxs[i]] = np.random.choice(diff_label_idxs)

    I = np.concatenate((np.eye(context_size), np.zeros((1, context_size))), axis=0)
    embedding_labels_reshape = I[embedding_labels]

    return embedding_labels_reshape, gamma_reshape


def cal_embedding_logits(embedding_bxm, gamma):
    """
    Calculate the embedding logits (after * gamma + sigmoid)
    """
    embedding_logits = tf.nn.sigmoid(embedding_bxm * gamma)

    return embedding_logits


def embedding_costs_f(embedding_logits, embedding_labels, embed_coefficient, name=None):
    """
    Calculate the embedding loss
    """
    with tf.name_scope(name, "embedding_costs_f") as scope:
        loss = -tf.reduce_sum(embedding_labels * tf.log(tf.add(embedding_logits, 1e-10)))
        total_count = tf.shape(embedding_logits)[0]  # count number of samples in both logits
        total_count = tf.to_float(total_count)
        mean_cost = tf.multiply(tf.div(loss, total_count), embed_coefficient, name=scope)
        assert_shape(mean_cost, [])

    return mean_cost
