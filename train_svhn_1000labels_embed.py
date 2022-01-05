import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import logging
from datetime import datetime

from experiments.run_context import RunContext
from datasets.svhn_embed import SVHN
from method.model_svhn_1000labels_embed import Model
from method import minibatching


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('main')


def run(data_seed=0):
    n_labeled = 1000
    n_extra_unlabeled = 0
    context_size = 500
    max_iter = 2000

    batch_size = 100
    n_labeled_per_batch = 'vary'

    model = Model(batch_size, context_size, n_labeled_per_batch, RunContext(__file__, 0))

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    svhn = SVHN(data_seed, n_labeled, n_extra_unlabeled, context_size, max_iter, test_phase=True)
    training_batches = minibatching.training_batches(svhn.training, batch_size, n_labeled_per_batch)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(svhn.evaluation, batch_size=200)

    model.train(training_batches, evaluation_batches_fn, svhn.context_training)


if __name__ == "__main__":
    run()
