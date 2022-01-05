import os
import numpy as np
from scipy.io import loadmat


DIR = os.path.join('data', 'images', 'cifar', 'cifar10')


def assert_not_exists(path):
    assert not os.path.exists(path), ""


def cifar10_orig_train():
    return load_batch_files([os.path.join(DIR, "data_batch_{}.mat".format(i)) for i in range(1, 6)])


def cifar10_orig_test():
    return load_batch_file(os.path.join(DIR, "test_batch.mat"))


def load_batch_files(batch_files):
    data_batches, label_batches = zip(*[load_batch_file(batch_file) for batch_file in batch_files])
    x = np.concatenate(data_batches, axis=0)
    y = np.concatenate(label_batches, axis=0)
    return x, y


def load_batch_file(path):
    d = loadmat(path)
    x = d['data'].astype(np.uint8)
    y = d['labels'].astype(np.uint8).flatten()
    return x, y


def to_channel_rgb(x):
    return np.transpose(np.reshape(x, (x.shape[0], 3, 32, 32)), [0, 2, 3, 1])


def do():
    train_x_orig, train_y = cifar10_orig_train()
    test_x_orig, test_y = cifar10_orig_test()

    train_x = to_channel_rgb(train_x_orig)
    test_x = to_channel_rgb(test_x_orig)

    p = os.path.join(DIR, "cifar10.npz")
    assert_not_exists(p)
    np.savez(p, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


if __name__ == "__main__":
    do()
