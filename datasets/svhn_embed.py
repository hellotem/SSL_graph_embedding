import os

import numpy as np
import scipy.io

from .utils import random_balanced_partitions, random_partitions, create_context


class Datafile:
    def __init__(self, path, n_examples):
        self.path = path
        self.n_examples = n_examples
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._load()
        return self._data

    def _load(self):
        data = np.zeros(self.n_examples, dtype=[
            ('x', np.float32, (32, 32, 3)),
            ('y', np.int32, ()),  # We will be using -1 for unlabeled
            ('z', np.int32, ())
        ])
        dictionary = scipy.io.loadmat(self.path)
        data['x'] = np.transpose(dictionary['X'], [3, 0, 1, 2])
        data['y'] = dictionary['y'].reshape((-1))
        data['y'][data['y'] == 10] = 0  # Use label 0 for zeros
        data['z'] = data['y']
        self._data = data


class SVHN:
    DIR = os.path.join('data', 'images', 'svhn')
    FILES = {
        'train': Datafile(os.path.join(DIR, 'train_32x32.mat'), 73257),
        'extra': Datafile(os.path.join(DIR, 'extra_32x32.mat'), 531131),
        'test': Datafile(os.path.join(DIR, 'test_32x32.mat'), 26032),
    }
    VALIDATION_SET_SIZE = 5000
    UNLABELED = -1

    def __init__(self, data_seed=0, n_labeled='all', n_extra_unlabeled=0, context_size=500, max_iter=2000, test_phase=True):
        random = np.random.RandomState(seed=data_seed)

        if test_phase:
            self.evaluation, self.training = self._test_and_training()
        else:
            self.evaluation, self.training = self._validation_and_training(random)

        if n_labeled != 'all':
            self.training = self._unlabel(self.training, n_labeled, random)

        if n_extra_unlabeled > 0:
            self.training = self._add_extra_unlabeled(self.training, n_extra_unlabeled, random)

        # self.context_training = create_context(self.training, context_size, max_iter)
        self.context_training, _ = random_partitions(self.training, context_size, random)

    def _validation_and_training(self, random):
        return random_partitions(self.FILES['train'].data, self.VALIDATION_SET_SIZE, random)

    def _test_and_training(self):
        return self.FILES['test'].data, self.FILES['train'].data

    def _unlabel(self, data, n_labeled, random):
        labeled, unlabeled = random_balanced_partitions(
            data, n_labeled, labels=data['y'], random=random)
        unlabeled['y'] = self.UNLABELED
        return np.concatenate([labeled, unlabeled])

    def _add_extra_unlabeled(self, data, n_extra_unlabeled, random):
        extra_unlabeled, _ = random_partitions(self.FILES['extra'].data, n_extra_unlabeled, random)
        extra_unlabeled['y'] = self.UNLABELED
        return np.concatenate([data, extra_unlabeled])
