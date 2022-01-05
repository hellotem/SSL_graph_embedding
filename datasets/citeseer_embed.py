"""
Only y and ty is numpy array, graph is default dict

x.shape             : (120, 3703)
y.shape             : (120, 6)
tx.shape            : (1000, 3703)
ty.shape            : (1000, 6)
allx.shape          : (2312, 3703)
ally.shape          : (2312, 6)
"""

import os
import pickle as cPickle
import numpy as np

from .utils import random_balanced_partitions, random_partitions, create_context


class Citeseer:
    DATASET = 'citeseer'
    UNLABELED = -1

    def __init__(self, data_seed=0, n_labeled='all', context_size=200, max_iter=2000):
        random = np.random.RandomState(seed=data_seed)
        self._load()

        self.evaluation, self.training = self._test_and_training()

        if n_labeled != 'all':
            self.training['y'][n_labeled:] = self.UNLABELED

        # self.context_training = create_context(self.training, context_size, max_iter)
        self.context_training, _ = random_partitions(self.training, context_size, random)

    def _load(self):
        NAMES = ['x', 'y', 'allx', 'ally', 'tx', 'ty', 'graph']
        OBJECTS = []
        for i in range(len(NAMES)):
            OBJECTS.append(cPickle.load(open("./data/text/ind.{}.{}".format(self.DATASET, NAMES[i]), 'rb'), encoding='latin1'))
        x, y, allx, ally, tx, ty, graph = tuple(OBJECTS)

        y = np.argmax(y, axis=1)
        ally = np.argmax(ally, axis=1)
        ty = np.argmax(ty, axis=1)
        trainx = np.concatenate((x.toarray(), allx.toarray()))
        trainy = np.concatenate((y, ally))

        self._train_data = self._data_array(2432, trainx, trainy)
        self._test_data = self._data_array(1000, tx.toarray(), ty)

    def _data_array(self, expected_n, x_data, y_data):
        array = np.zeros(expected_n, dtype=[
            ('x', np.float32, (3703)),
            ('y', np.int32, ()),  # We will be using -1 for unlabeled
            ('z', np.int32, ())
        ])
        array['x'] = x_data
        array['y'] = y_data
        array['z'] = y_data
        return array

    def _test_and_training(self):
        return self._test_data, self._train_data
