import numpy as np

# Developer: Alejandro Debus
# Email: aledebus@gmail.com

def partitions(number, k):
    '''
    Distribution of the folds
    Args:
        number: number of patients
        k: folds number
    '''
    n_partitions = np.ones(k) * int(number/k)
    n_partitions[0:(number % k)] += 1
    return n_partitions

def get_indices(n_splits, n_samples):
    '''
    Indices of the set test
    Args:
        n_splits: folds number
        n_samples: number of samples
    '''
    l = partitions(n_samples, n_splits)
    fold_sizes = l
    indices = np.arange(n_samples).astype(int)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop =  current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])

def k_folds(n_splits, n_samples):
    '''
    Generates folds for cross validation
    Args:
        n_splits: folds number
        n_samples: number of samples
    '''
    indices = np.arange(n_samples).astype(int)
    for test_idx in get_indices(n_splits, n_samples):
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx
