import numpy as np
import os


def arc2deg(arc):
    return arc * 180 / np.pi


def deg2arc(deg):
    return deg * np.pi / 180


def int2str(num, length=None):
    """
    generate a string representation for a integer with a given length
    :param num: input number
    :param length: length of the string
    :return: a string representation of the integer
    """

    rawstr = str(int(num))
    if length is None or length == len(rawstr):
        return rawstr
    elif length < len(rawstr):
        raise(ValueError, 'Length of the number is longer then defined display length!')
    elif length > len(rawstr):
        return '0' * (length - len(rawstr)) + rawstr


def round_int(num):
    """
    :return: round integer of the given number
    """
    return int(round(num))


def array_nor(arr):
    """
    normalize a np.array to the scale [0, 1]
    """
    return (arr - np.amin(arr)) / (np.amax(arr) - np.amin(arr)).astype(np.float)


def array_nor_median(arr):
    '''
    normalize array by minus median, data type will be switch to np.float
    '''
    arr2=arr.astype(np.float)
    return arr2-np.median(arr2.flatten())


def array_nor_mean(arr):
    '''
    normalize array by minus mean, data type will be switch to np.float
    '''
    arr2=arr.astype(np.float)
    return arr2-np.mean(arr2.flatten())


def z_score(arr):
    """
    return Z score of an array.
    """
    if np.isnan(arr).any():
        return (arr-np.nanmean(arr.flatten()))/np.nanstd(arr.flatten())
    else:
        arr2 = arr.astype(np.float)
        return (arr2-np.mean(arr2.flatten()))/np.std(arr2.flatten())


def add_suffix(path, suffix):
    """
    add a suffix to file name of a given path

    :param path: original path
    :param suffix: str
    :return: new path
    """

    folder, file_name_full = os.path.split(path)
    file_name, file_ext = os.path.splitext(file_name_full)
    file_name_full_new = file_name + suffix + file_ext
    return os.path.join(folder, file_name_full_new)