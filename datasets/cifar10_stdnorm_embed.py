import os

import numpy as np

from .utils import random_balanced_partitions, random_partitions, create_context


class Cifar10:
    DATA_PATH = os.path.join('data', 'images', 'cifar', 'cifar10', 'cifar10.npz')
    VALIDATION_SET_SIZE = 5000  # 10% of the training set
    UNLABELED = -1

    def __init__(self, data_seed=0, n_labeled='all', context_size=500, max_iter=2000, test_phase=True):
        random = np.random.RandomState(seed=data_seed)
        self._load()

        if test_phase:
            self.evaluation, self.training = self._test_and_training()
        else:
            self.evaluation, self.training = self._validation_and_training(random)

        if n_labeled != 'all':
            self.training = self._unlabel(self.training, n_labeled, random)

        # self.context_training = create_context(self.training, context_size, max_iter)
        self.context_training, _ = random_partitions(self.training, context_size, random)

    def _load(self):
        file_data = np.load(self.DATA_PATH)
        self._train_data = self._data_array(50000, file_data['train_x'], file_data['train_y'])
        self._test_data = self._data_array(10000, file_data['test_x'], file_data['test_y'])

    def _data_array(self, expected_n, x_data, y_data):
        array = np.zeros(expected_n, dtype=[
            ('x', np.float32, (32, 32, 3)),
            ('y', np.int32, ()),  # We will be using -1 for unlabeled
            ('z', np.int32, ())
        ])
        array['x'] = x_data
        array['y'] = y_data
        array['z'] = y_data
        return array

    def _validation_and_training(self, random):
        return random_partitions(self._train_data, self.VALIDATION_SET_SIZE, random)

    def _test_and_training(self):
        return self._test_data, self._train_data

    def _unlabel(self, data, n_labeled, random):
        labeled, unlabeled = random_balanced_partitions(
            data, n_labeled, labels=data['y'], random=random)
        unlabeled['y'] = self.UNLABELED
        return np.concatenate([labeled, unlabeled])
