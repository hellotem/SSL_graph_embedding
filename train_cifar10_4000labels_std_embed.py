import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
from datetime import datetime

from experiments.run_context import RunContext
from datasets.cifar10_stdnorm_embed import Cifar10
from method import std_norm
from method.model_cifar10_4000labels_embed import Model
from method import minibatching


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('main')


def run(data_seed=0):
    dataset = 'cifar10'
    n_labeled = 4000
    context_size = 500
    max_iter = 2000

    batch_size = 100
    n_labeled_per_batch = 'vary'

    model = Model(batch_size, context_size, n_labeled_per_batch, RunContext(__file__, 0))

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    cifar = Cifar10(data_seed, n_labeled, context_size, max_iter, test_phase=True)

    # cifar.training['x'] = std_norm.normalize_images(cifar.training['x'], dataset)
    # cifar.context_training['x'] = std_norm.normalize_images(cifar.context_training['x'], dataset)
    # cifar.evaluation['x'] = std_norm.normalize_images(cifar.evaluation['x'], dataset)

    training_batches = minibatching.training_batches(cifar.training, batch_size, n_labeled_per_batch)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(cifar.evaluation, batch_size=200)

    model.train(training_batches, evaluation_batches_fn, cifar.context_training)


if __name__ == "__main__":
    run()
