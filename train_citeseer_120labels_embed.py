import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import logging
from datetime import datetime

from experiments.run_context import RunContext
from datasets.citeseer_embed import Citeseer
from method.model_citeseer_120labels_embed_fixed import Model
from method import minibatching


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('main')


def run(data_seed=0):
    n_labeled = 120
    context_size = 200
    max_iter = 2000

    batch_size = 100
    n_labeled_per_batch = 20

    model = Model(batch_size, context_size, n_labeled_per_batch, RunContext(__file__, 0))

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    citeseer = Citeseer(data_seed, n_labeled, context_size, max_iter)

    training_batches = minibatching.training_batches(citeseer.training, batch_size, n_labeled_per_batch)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(citeseer.evaluation, batch_size=200)

    model.train(training_batches, evaluation_batches_fn, citeseer.context_training)


if __name__ == "__main__":
    run()
