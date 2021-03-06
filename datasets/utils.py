import numpy as np
from pyclust import KMedoids


def random_partitions(data, first_size, random):
    """Split data into two random partitions of sizes n and len(data) - n
    Args:
        data (ndarray): data to be split
        first_size (int): size of the first partition
        random (RandomState): source of randomness
    Return:
        tuple of two ndarrays
    """
    mask = np.zeros(len(data), dtype=bool)
    mask[:first_size] = True
    random.shuffle(mask)
    return data[mask], data[~mask]


def random_balanced_partitions(data, first_size, labels, random=np.random):
    """Split data into a balanced random partition and the rest
    Partition the `data` array into two random partitions, using
    the `labels` array (of equal size) to guide the choice of
    elements of the first returned array.
    Example:
        random_balanced_partition(['a', 'b', 'c'], 2, [3, 5, 5])
        # Both labels 3 and 5 need to be presented once, so
        # the result can be either (['a', 'b'], ['c']) or
        # (['a', 'c'], ['b']) but not (['b', 'c'], ['a']).
    Args:
        data (ndarray): data to be split
        first_size (int): size of the first partition
        balance (ndarray): according to which balancing is done
        random (RandomState): source of randomness
    Return:
        tuple of two ndarrays
    """
    assert len(data) == len(labels)

    classes, class_counts = np.unique(labels, return_counts=True)
    assert len(classes) <= 10000, "surprisingly many classes: {}".format(len(classes))
    assert first_size % len(classes) == 0, "not divisible: {}/{}".format(first_size, len(classes))
    assert np.all(class_counts >= first_size // len(classes)), "not enough examples of some class"

    idxs_per_class = [np.nonzero(labels == klass)[0] for klass in classes]
    chosen_idxs_per_class = [
        random.choice(idxs, first_size // len(classes), replace=False)
        for idxs in idxs_per_class
    ]
    first_idxs = np.concatenate(chosen_idxs_per_class)
    second_idxs = np.setdiff1d(np.arange(len(labels)), first_idxs)

    assert first_idxs.shape == (first_size,)
    assert second_idxs.shape == (len(data) - first_size,)
    return data[first_idxs], data[second_idxs]


def create_context(data, m, max_iter):
    """
    select m samples as context
    """
    KM = KMedoids(n_clusters=m, distance='euclidean', max_iter=max_iter)
    data_train_2d = data['x'].reshape((data['x'].shape[0], -1))
    n_features = data_train_2d.shape[1]

    print('start fitting')
    KM.fit(data_train_2d)
    print('end fitting')
    idxs = np.array([np.squeeze(np.argwhere(np.sum(data_train_2d == KM.centers_[i], axis=1) == n_features)) for i in range(m)])

    return data[idxs]
