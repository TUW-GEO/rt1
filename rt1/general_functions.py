# -*- coding: utf-8 -*-
"""
helper functions that are used both in rtfits and rtplots
"""

import numpy as np
from itertools import tee, islice

def rectangularize(array, return_mask=False, dim=None,
                   return_masked=False):
    '''
    return a rectangularized version of the input-array by repeating the
    last value to obtain the smallest possible rectangular shape.

        input:
            - array = [[1,2,3], [1], [1,2]]

        output:
            - return_masked=False: [[1,2,3], [1,1,1], [1,2,2]]
            - return_masked=True:  [[1,2,3], [1,--,--], [1,2,--]]

    Parameters:
    ------------
    array: list of lists
           the input-data that is intended to be rectangularized
    return_mask: bool (default = False)
              	 indicator if weights and mask should be evaluated or not
    dim: int (default = None)
         the dimension of the rectangularized array
         if None, the shortest length of all sub-lists will be used
    return_masked: bool (default=False)
                   indicator if a masked-array should be returned

    Returns:
    ----------
    new_array: array-like
               a rectangularized version of the input-array
    mask: array-like (only if 'weights_and_mask' is True)
          a mask indicating the added values

    '''
    if dim is None:
        # get longest dimension of sub-arrays
        dim  = len(max(array, key=len))

    if return_mask is True or return_masked is True:
        newarray, mask = [], []
        for s in array:
            adddim = dim - len(s)
            m = np.full_like(s, False)
            if adddim > 0:
                s = np.append(s, np.full(adddim, s[-1]))
                m = np.append(m, np.full(adddim, True))
            newarray += [s]
            mask     += [m]

        newarray = np.array(newarray)
        mask = np.array(mask, dtype=bool)

        if return_masked is True:
            return np.ma.masked_array(newarray, mask)
        else:
            return [newarray, mask]
    else:
        newarray = []
        for s in array:
            adddim = dim - len(s)
            if adddim > 0:
                s = np.append(s, np.full(adddim, s[-1]))
            newarray += [s]
        return np.array(newarray)


def meandatetime(datetimes):
    '''
    calculate the average date from a given list of datetime-objects
    (can be applied to a pandas-Series via Series.apply(meandatetime))

    Parameters:
    ------------
    datetimes: list
               a list of datetime-objects
    Returns:
    ---------
    meandate: Timestamp
              the center-date
    '''

    if len(datetimes) == 1:
        return datetimes[0]

    x = datetimes
    deltas = (x[0] - x[1:])/len(x)
    meandelta = sum(deltas)
    meandate = x[0] - meandelta
    return meandate


def dBsig0convert(val, inc,
                  dB, sig0,
                  fitdB, fitsig0):
    '''
    A convenience-function to convert an array of measurements (and it's
    associated incidence-angles).
        - between linear- and dB units   `( val_dB = 10 * log10(val_linear) )`
        - between sigma0 and intensity   `( sig0 = 4 * pi * cos(inc) * I )`

    Parameters:
    ----------------

    val: array-like
         the backscatter-values that should be converted
    inc: array-like
         the associated incidence-angle values (in radians)
    dB: bool
        indicator if the output-dataset should be in dB or not
    sig0: bool
          indicator if the output-values should be intensity or sigma_0
    fitdB: bool
           indicator if the input-values have been provided in linear-units
           or in dB
    fitsig0: bool
           indicator if the input-values are given as sigma0 or intensity

    Returns:
    ---------
    val : array-like
          the converted values

    '''

    if sig0 is not fitsig0:
        # if results are provided in dB convert them to linear units before
        # applying the sig0-intensity conversion
        if fitdB is True:
            val = 10**(val/10.)
        # convert sig0 to intensity
        if sig0 is False and fitsig0 is True:
            val = val/(4.*np.pi*np.cos(inc))
        # convert intensity to sig0
        if sig0 is True and fitsig0 is False:
            val = 4.*np.pi*np.cos(inc)*val
        # convert back to dB if required
        if dB is True:
            val = 10.*np.log10(val)
    elif dB is not fitdB:
        # if dB output is required, convert to dB
        if dB is True and fitdB is False:
            val = 10.*np.log10(val)
        # if linear output is required, convert to linear units
        if dB is False and fitdB is True:
            val = 10**(val/10.)


    return val


def pairwise(iterable):
    """
    a generator to return consecutive pairs from an iterable, e.g.:

        s -> (s0,s1), (s1,s2), (s2, s3), ...

    taken from https://docs.python.org/3.7/library/itertools.html
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def split_into(iterable, sizes):
    """
    a generator that splits the iterable into iterables with the given sizes

    see more_itertools split_into for details:
    https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.split_into
    """
    it = iter(iterable)
    for size in sizes:
        if size is None:
            yield list(it)
            return
        else:
            yield list(islice(it, size))
