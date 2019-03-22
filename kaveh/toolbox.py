from scipy.signal import resample
import numpy as np
import os

def resample_to_freq(data, source_frequency, target_frequency):
    """
    resamples the input data from source_frequency to target_frequency
    """
    resample_ratio = float(target_frequency) / source_frequency
    if resample_ratio == 1:
        return data
    else:
        target_n_samples = int(np.size(data) * resample_ratio)
        resampled_data = resample(data, target_n_samples)
        return resampled_data


def closest_argmin(A, B):
    """
    Finds the indices of the nearest value in B to values in A
    Output is the same size as A
    Source: https://stackoverflow.com/a/45350318
    """
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]

def find_file(name, path):
    """
    Finds file name in path
    """
    not_found = True
    for root, dirs, files in os.walk(path):
        if name in files:
            not_found = False
            return os.path.join(root, name)
    if not_found:
        raise ValueError('could not find {} in {}'.format(name, path))


def rolling_window(a, window):
    '''
    Returns an ndarray with dimensions a.shape+window-1 by window
    '''
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def find_first(x, start=None, direction='forward', inclusive=True, pattern=[True]):
    """Finds the first true value in the sequence

    Returns the index of the first True value in the sequence (assuming a binary
    array, x.
    :param x: A binary (numpy) array
    :param start: The index to start looking [0, len(x) - 1]. Default is None
    :param direction: Forward or backwards (the direction of the search)
    :param inclusive: Include the start index in the search (defaults to True)
    :param pattern: The pattern to find, default to true (i.e. the first true)
    """
    if not hasattr(pattern, '__iter__'):
        pattern = [pattern]

    pattern = np.array(pattern)
    x = np.array(x)

    pattern_length = len(pattern)
    if direction == 'forward' or direction == 'forwards' or direction is True:
        if start is None:
            start = 0
        for i in range(start, len(x) - pattern_length + 1):
            if (x[i] == pattern[0]) and ((i == start and inclusive is True) or (i != start)) and np.all(x[i:i+pattern_length] == pattern):
                return i
    else:
        if start is None:
            start = len(x) - 1
        for i in range(start, -1 + pattern_length - 1, -1):
            if (x[i] == pattern[-1]) and ((i == start and inclusive is True) or (i != start)) and np.all(x[i-pattern_length+1:i+1] == pattern):
                return i
    return None  # Not found
