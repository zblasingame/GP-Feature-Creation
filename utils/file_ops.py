"""Module for loading and parsing datasets.

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import csv
import numpy as np


def load_data(filename):
    """Returns the features of a dataset.

    Parameters
    ----------
    filename : str
        Location of dataset, csv formatted

    Returns
    -------
    X, Y : np.ndarray
        Tuple consisiting of the features, X, and the labels, Y.
    """

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = np.array(
            [row for row in reader if '#' not in row[0]]
        ).astype(np.float32)

    X = data[:, 1:]
    Y = data[:, 0]

    return X, Y


def load_shape(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = np.array(next(reader))

    header = data
    f1 = header[1]
    n_steps = np.sum(header == f1)
    n_features = (header.shape[0] - 1) / n_steps

    return int(n_steps), int(n_features)


def batcher(data, batch_size=100, halt=True):
    """Creates a generator to yield batches of `batch_size`.
    When batch is too large to fit remaining data the batch
    is clipped.

    Parameters
    ----------
    data : sequence of np.ndarrays
        List of datasets to be batched. The first dimension must
        be the number of samples and the same for allsets.
    batch_size : int, optional
        Size of the mini-batches, default is 100.
    halt : bool, optional
        Halts generator after yielding all samples once, default True.

    Yields
    ------
    batched_data : sequence of np.ndarrays
        The next mini-batch in the sequence of datasets.
    """

    cont = True

    while cont:
        cont = not halt

        batch_start = 0
        batch_end = batch_size

        while batch_end < data[0].shape[0]:
            yield [el[batch_start:batch_end] for el in data]

            batch_start = batch_end
            batch_end += batch_size

        yield [el[batch_start:] for el in data]


def random_batcher(data, batch_size=100):
    """Creates a generator to yield random mini-batches of batch_size.
    When batch is too large to fit the remaining data the batch
    is clipped. Will continously cycle through data.

    Parameters
    ----------
    data : sequence of np.ndarrays
        List of datasets to be batched. The first dimension must
        be the number of samples and the same for allsets.
    batch_size : int, optional
        Size of the mini-batches, default is 100.

    Yields
    ------
    batched_data : sequence of np.ndarrays
        The next mini-batch in the sequence of datasets.
    """

    while True:
        assert all(data[0].shape[0] == el.shape[0] for el in data), (
            'Not all data arrays have the same length'
        )

        rand = np.random.permutation(len(data[0]))
        data = [el[rand] for el in data]

        batch_start = 0
        batch_end = batch_size

        while batch_end < data[0].shape[0]:
            yield [el[batch_start:batch_end] for el in data]

            batch_start = batch_end
            batch_end += batch_size

        yield [el[batch_start:] for el in data]


def rescale(data, _min, _max, start=0.0, end=1.0):
    """Rescale features of a dataset.

    Parameters
    ----------
    data : np.array
        Dataset to be rescaled.
    _min : dtype of `data`
        Minimal features of dataset.
    _max : dtype of `data`
        Maximal features of dataset.
    start : float, optional
        Lowest value for normalized dataset, default is 0.
    end : float, optional
        Highest value for normalized dataset, default is 1.

    Returns
    -------
        normalized_data : np.array
        Normalized features, the same shape as `data`.
    """

    new_data = (data - _min) / (_max - _min)

    # check if feature is constant, will be nan in new_data
    np.place(new_data, np.isnan(new_data), (end-start)/2)

    new_data = (end - start) * new_data + start

    return new_data
