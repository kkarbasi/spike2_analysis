"""
Copyright (c) 2016 David Herzfeld

Written by David J. Herzfeld <herzfeldd@gmail.com>
"""

import numpy as np
import pickle
import h5py
import warnings
import scipy.stats

def sem(x, axis=0):
    """Returns the standard error about the mean for a particular axis

    :param x The input matrix or vector (numpy)
    :param axis The axis to use to compute the standard error (0 is default)
    :return The standard error along the particular dimension
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y = np.nanstd(x, axis) / np.sqrt(np.sum(~np.isnan(x), axis))
    return y

def ci(x, axis=0):
    """Returns the 95% confidence intervals about the mean for a particular axis

    :param x The input matrix or vector (numpy)
    :param axis The axis to use to compute the 95% confidence intervals (0 is default)
    :return The 95% confidence intervals along the particular dimension
    """

    return np.abs(scipy.stats.t.interval(0.95, np.array(x).shape[axis]-1, scale=sem(x, axis))[0])

def angle_mod(theta, phi):
    """Determines the minimum angular distance between theta and phi

    Determines the minimum angular distance between two angles, theta and
    phi. The angle is returned in the range [-pi, pi] radians.

    :param theta: First angle
    :param phi: Second angle
    :return: The difference between theta and phi in [-pi, pi] radians (numpy array)
    """
    theta = np.array(theta)
    phi = np.array(phi)

    x = phi - theta
    x = np.mod(x, 2 * np.pi) # Place the angle between [-2*pi, and 2*pi]

    # Place angle between -pi and pi
    x = np.array(x)
    x[x < -np.pi] = x[x < -np.pi] + 2 * np.pi
    x[x > np.pi] = x[x > np.pi] - 2 * np.pi
    return x

def pca(x):
    """Performs principle component analysis on the input matrix, x"""
    x = x - np.mean(x, axis=0)
    [latent, coeff] = np.linalg.eig(np.cov(x.T))
    # Convert to real
    latent = np.real(latent)
    coeff = np.real(coeff)

    score = np.dot(x, coeff) # projection of the data in the new space

    # Sort everything by the percent of the variance that was explained
    index = np.argsort(latent) # sort from low to high
    index = np.flipud(index) # Now sort from high to low
    latent = latent[index]
    score = score[:, index]
    coeff = coeff[:, index]
    # Determine the percent of the variance explained
    percent_explained = latent * 100 / np.sum(latent)
    return coeff, score, latent, percent_explained


def select(x, mask):
    """Returns a subset of the list, x, selected by the mask

    :param x: A list whose length matches mask
    :param mask: A boolean array or list used to select items in x
    """
    if isinstance(x, np.ndarray) and x.ndim == 1 and len(x) == len(mask):
        return x[np.where(mask)[0]]
    elif isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[0] == len(mask):
        return x[np.where(mask)[0], :]
    elif isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == len(mask):
        return x[:, np.where(mask)[0]]
    elif (isinstance(x, list) or isinstance(x, tuple)) and len(x) == len(mask):
        return [x[i] for i in range(len(x)) if mask[i]]
    else:
        raise RuntimeError('Types unknown or sizes are incommensurate')


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


def nargout():
    """Returns the number of output arguments are given function

    This function is equivalent to the nargout variable in Matlab (R).
    """
    import inspect
    import dis
    f = inspect.currentframe()
    f = f.f_back.f_back
    c = f.f_code
    i = f.f_lasti
    bytecode = c.co_code
    instruction = bytecode[i+3]
    if instruction == dis.opmap['UNPACK_SEQUENCE']:
        return bytecode[i+4]
    elif instruction == dis.opmap['POP_TOP']:
        return 0
    return 1

def write_csv(x, filename, format='short', id=None, measurement_axis=None):
    """Write the contents of a numpy array to a CSV file

    In the simplest case, data is an MxN data array containing N repeated measures
    for N subjects. The id for each subject are ordered from 1 to M as the first
    column in the output file.

    If `x` is a list of numpy arrays, we write each of the data arrays inside the
    cell array as a different group. Group numbering starts from 1 to the length of
    the list.

    By default, the format is `short' corresponding to measurement along the horizontal
    axis (one row in the output file per subjects). However specifying format='long'
    results in a single measurement per row. This is useful for analysis in R.

    By passing a list of strings for the measurement_axis, the column names are
    redefined.

    It is also possible to pass the ID for each subject (as a list).
    """

    # Ensure all inputs are the same size
    if isinstance(x, list) and isinstance(x[0], list):
        x = [np.array(x[i]) for i in range(0, len(x))]

    if id is None:
        if isinstance(x, np.ndarray):
            id = np.arange(0, x.shape[0])
        else:
            current_id = 0
            id = []
            for group_number in range(0, len(x)):
                id.append(np.arange(0, x[group_number].shape[0]) + current_id)
                current_id = current_id + x[group_number].shape[0]

    if isinstance(x, list) and not isinstance(id, list):
        raise RuntimeError('Expected a list of ids')

    # Ensure that the number of data elements is the same for all groups
    if isinstance(x, np.ndarray):
        num_measures = x.shape[1]
    else:
        num_measures = x[0].shape[1]
        for i in range(0, len(x)):
            if x[i].shape[1] != num_measures:
                raise RuntimeError('Number of measures is inconsistent across groups')

    fp = open(filename, "w")
    # Write out the header
    fp.write('id, ')
    if isinstance(x, list):
        fp.write('group, ')
    if 'short' in format:
        if measurement_axis is not None:
            fp.write(", ".join(measurement_axis))
        else:
            fp.write(", ".join(['measure_{:d}'.format(i) for i in range(0, num_measures)]))
    else:
        fp.write("measure_id, ")
        fp.write("dependent")
    fp.write("\n")

    if 'short' in format:
        if isinstance(x, list):
            for group_number in range(0, len(x)):
                for subject_number in range(0, x[group_number].shape[0]):
                    fp.write('{:s}, {:s}, '.format(str(id[group_number][subject_number]), str(group_number)))
                    fp.write(", ".join(['{:f}'.format(x[group_number][subject_number, i]) for i in range(0, num_measures)]))
                    fp.write("\n")
        else:
            for subject_number in range(0, x.shape[0]):
                fp.write('{:s},  '.format(str(id[subject_number])))
                fp.write(", ".join(['{:f}'.format(x[subject_number, i]) for i in range(0, num_measures)]))
                fp.write("\n")
    else:
        if isinstance(x, list):
            for group_number in range(0, len(x)):
                for subject_number in range(0, x[group_number].shape[0]):
                    for measure_number in range(0, num_measures):
                        fp.write('{:s}, {:s}, {:d}, {:f}\n'.format(str(id[group_number][subject_number]), str(group_number), measure_number, x[group_number][subject_number, measure_number]))
        else:
            for subject_number in range(0, x.shape[0]):
                    for measure_number in range(0, num_measures):
                        fp.write('{:s}, {:d}, {:f}\n'.format(str(id[subject_number]), measure_number, x[subject_number, measure_number]))
    fp.close()
