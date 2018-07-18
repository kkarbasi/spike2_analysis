"""
Copyright (c) 2016 David Herzfeld

Written by David J. Herzfeld <herzfeldd@gmail.com>
"""

import h5py as h5
import os
import sys
import numpy as np
import pickle

def hdf5_save_object(x, file, group=None, name=None, raise_error=True):
    """Save the contents of this hdf5 file to an hdf5 file

    Saves an object to a new HDF5 file. If the output file exists, the file is overwritten without
    warning. We save each of the attributes of the class (as defined by the built-in function, vars)
    into the HDF5 file. Sub-objects (children) are then saved in turn. Therefore, this function
    traverses the object tree recursively, saving each variable in turn.

    If an object defines a hook __hdf5_save__, then that function is called instead of this
    generic function. This allows a class to override the behavior of this function (and possibly
    omit or add extra data to the HDF5 file).

    :param x: is an instance of a class (something that inherits from object)
    :param file: A file object for an open HDF5 file. Otherwise, if this variable is a string, then
    we use this as the name HDF5 file.
    :param group: The current HDF5 group (or None if we should write to the root of the HDF5 file, '/')
    """

    # If this is a string, open a new HDF5 file
    close_file = False
    if isinstance(file, str):
        file = h5.File(file, 'w', libver='latest')
        close_file = True

    # If the group is not defined, use the root group ('/')
    if group is None:
        group = file.require_group("/")
        if name is None:
            raise RuntimeError("Requires a variable name")
    # Make sure that the passed object is an instance of object
    if not isinstance(x, object):
        raise TypeError

    if isinstance(x, bool) or isinstance(x, int) or isinstance(x, float) \
            or isinstance(x, list) or isinstance(x, tuple) or isinstance(x, np.ndarray) \
            or isinstance(x, np.floating) or isinstance(x, bytes) or isinstance(x, np.integer) \
            or isinstance(x, np.bool_):
        _hdf5_save_item(x, name, file=file, group=group)
    elif isinstance(x, dict):
        if name is not None:
            group = group.create_group(name)
        for key in x.keys():
            hdf5_save_object(x[key], name=key, file=file, group=group)
    else:  # This is a class, which we have to save
        if name is not None:
            group = group.create_group(name)  # New sub-group
        # Save the class name
        group.attrs['__class__'] = np.void(pickle.dumps(x.__class__))

        # Check if this object defines the hook __hdf5_save__
        if hasattr(x, '__hdf5_save__'):
            x.__hdf5_save__(file, group)
            return  # Nothing else to do

        # Get a list of all attributes of this class (that are not properties)
        try:
            attributes = vars(x)
        except:
            print(type(x))
            raise RuntimeError
        for attribute in attributes:
            # Skip saving any variables that are attributes of this class
            if isinstance(getattr(type(x), attribute, None), property):
                continue  # This is a class property, no need to save it
            hdf5_save_object(getattr(x, attribute), name=attribute, file=file, group=group)

    if close_file:
        file.close()  # Close the file, all done

def _hdf5_save_known_type(x, name, file, group):
    """Saves a basic type

    This function returns True if savings was successful, otherwise false
    :return True if saving was successful
    """
    if isinstance(x, int) or isinstance(x, float) or isinstance(x, bool) or isinstance(x, np.integer) \
            or isinstance(x, np.floating) or isinstance(x, np.bool_):
        group.attrs[name] = x
        return True
    elif isinstance(x, bytes):
        group.attrs[name] = str(x)
        return True
    # Save numpy arrays/matrices (no chunking as it slows access times)
    elif isinstance(x, np.ndarray) and x.dtype != np.dtype(object):
        # Save the nd-array
        group.create_dataset(name, data=x)
        return True
    elif x is None:  # Save none-types
        group.attrs[name] = np.empty((0,))  # h5.Empty("f")  # Empty
        return True
    else:
        # Try to coerce to ndarray
        try:
            x = np.array(x)
            if x.dtype != np.dtype(object):
                group.create_dataset(name, data=x)
                return True
        except:
            pass
    return False

def _hdf5_save_item(x, name, file, group):
    if _hdf5_save_known_type(x, name, file, group):
        return  # Saving successful
    # To save these, we create a list of object references. This is
    # the actual list. Then, the actual data is stored as /path/__name_[index]
    # During the load function, we skip any values that begin with '__'
    elif isinstance(x, list) or \
            isinstance(x, tuple) or \
            (isinstance(x, np.ndarray) and x.dtype == np.dtype('object')):
        # Save the list of objects - unable to coerce to numpy array
        dtype = h5.special_dtype(ref=h5.Reference)
        d = np.empty(len(x), dtype=dtype) #group.create_dataset(name, (len(x),), dtype=dtype)
        for i in range(len(x)):
            import saccades.neuron
            if isinstance(x[i], saccades.neuron.Neuron):
                print('saving neuron {:d}'.format(i))
            new_group = group.create_group('__' + name + '_{:d}'.format(i))
            hdf5_save_object(x[i], file, new_group)
            d[i] = new_group.ref
            if isinstance(x[i], saccades.neuron.Neuron):
                print(' -done saving neuron {:d}'.format(i))
        group.create_dataset(name, data=d)
    # Save all other objects
    elif isinstance(x, object):
        hdf5_save_object(x, file=file, name=name, group=group)
    return

def hdf5_load_object(file, group=None):
    """Load objects from a HDF5 data set

    Load HDF5 objects from a dataset, passing the name of the file as a parameter
    """
    # Open the file object
    close_file = False
    if isinstance(file, str):
        file = h5.File(file, "r", libver='latest')
        close_file = True

    if group is None:
        group = file['/']

    # Attempt to find the class name or data value (if class name not present)
    if not '__class__' in group.attrs and 'data' in group:
        # No class, just load the dataset under data
        return _hdf5_load_dataset(group['data'], file, group)
    elif '__class__' in group.attrs:
        c = pickle.loads(group.attrs['__class__'])
        x = c.__new__(c)
        if '__hdf5_load__' in dir(x):
            return x.__hdf5_load__(file, group)
        _hdf5_load_group(x, file, group)
        if '__hdf5_post_load__' in dir(x):
            x.__hdf5_post_load__()
    elif isinstance(group, h5.Group):
        # no class and no data, just load the top object
        loaded_group = type('h5class', (), {})()
        _hdf5_load_group(loaded_group, file, group)
        return loaded_group.__dict__  # NOTE: Added this 03/03/2017 (returns a dict instead of an opaque class)
    else:
        raise RuntimeError('Invalid HDF5 file')
    
    if close_file:
        file.close()

    return x  # Return the loaded array


def _hdf5_load_group(x, file, group):
    # Look at all data sets and load those
    for key in group:
        if len(key) >= 2 and key[:2] == '__':
            continue  # Skip hidden keys
        value = group[key]
        if isinstance(value, h5.Dataset):
            dataset = _hdf5_load_dataset(value, file, group)
            setattr(x, key, dataset)
        elif isinstance(value, h5.Group):
            value = hdf5_load_object(file, value) # Load the new group (value = group)
            setattr(x, key, value) #  Note that value is an object here
    for key in group.attrs:
        if len(key) >= 2 and key[:2] == '__':
            continue  # Skip hidden keys
        value = group.attrs[key]
        if isinstance(value, np.ndarray) and value.shape[0] == 0 and value.ndim == 1:
            setattr(x, key, None)
        elif isinstance(value, np.ndarray):
            setattr(x, key, np.squeeze(value))
        else:
            setattr(x, key, value)


def _hdf5_load_dataset(dataset, file, group):
    value = dataset.value
    if isinstance(value, np.ndarray):
        value = np.atleast_1d(np.squeeze(value))
        if len(value.shape) == 0:
            return value
        for i in range(len(value)):
            if isinstance(value[i], h5.Reference):  # Load any references
                new_group = group[value[i]]
                value[i] = hdf5_load_object(file, new_group)
            else:
                value = np.squeeze(value)
                break  # This is a standard array (no support for incomplete referencing)
    return value


def hdf5_load(filename):
    """ Loads an HDF5 into numpy elements
    :param filename: Name of the HDF5 file to open
    :return: A dictionary containing elements
    """

    if not os.path.isfile(filename):
        raise RuntimeError('Path to {:s} is invalid.'.format(filename))
    return hdf5_load_object(filename)

def hdf5_append_dataset(filename, name, data):
    if not os.path.isfile(filename):
        raise RuntimeError('Path to {:s} is invalid.'.format(filename))
    f = h5.File(filename, "r+")
    if name in f:
        del f[name]
    dset = f.create_dataset(name, data=data)
    f.close()
    
