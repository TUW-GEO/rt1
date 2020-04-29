# -*- coding: utf-8 -*-
"""
helper functions that are used both in rtfits and rtplots
"""
import sys
import numpy as np
from itertools import tee, islice
from collections import OrderedDict


def rectangularize(array, return_mask=False, dim=None,
                   return_masked=False, dtype=None):
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
    dtype: type (default = None)
           the dtype of the returned array. If None, the dtype of the first
           element will be used
    Returns:
    ----------
    new_array: array-like
               a rectangularized version of the input-array
    mask: array-like (only if 'weights_and_mask' is True)
          a mask indicating the added values

    '''
    # use this method to get the dtype of the first element since it works with
    # pandas-Series, lists, arrays, dict-value views, etc.
    if dtype is None:
        dtype = np.array(next(islice(array, 1))).dtype

    if dim is None:
        # get longest dimension of sub-arrays
        dim  = len(max(array, key=len))

    if return_mask is True or return_masked is True:
        newarray = np.empty((len(array), dim), dtype=dtype)
        mask = np.full((len(array), dim), False, dtype=bool)

        for i, s in enumerate(array):
            l = len(s)
            newarray[i, :l] = s
            newarray[i, l:] = s[-1]
            mask[i,l:] = True

        if return_masked is True:
            return np.ma.masked_array(newarray, mask)
        else:
            return [newarray, mask]
    else:
        newarray = np.empty((len(array), dim), dtype=dtype)
        for i, s in enumerate(array):
            l = len(s)
            newarray[i, :l] = s
            newarray[i, l:] = s[-1]
        return newarray



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


def pairwise(iterable, pairs=2):
    """
    a generator to return n consecutive values from an iterable, e.g.:

        pairs = 2
        s -> (s0,s1), (s1,s2), (s2, s3), ...

        pairs = 3
        s -> (s0, s1, s2), (s1, s2, s3), (s2, s3, s4), ...

    adapted from https://docs.python.org/3.7/library/itertools.html
    """
    x = tee(iterable, pairs)
    for n, n_iter in enumerate(x[1:]):
        [next(n_iter, None) for i in range(n + 1)]
    return zip(*x)

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


def scale(x, out_range=(0, 1),
          domainfuncs=(np.nanmin, np.nanmax)):
    '''
    scale an array between out_range = (min, max) where the range of the
    array is evaluated via the domainfuncs (min-function, max-funcion)

    useful domainfuncs are:

        >>> np.nanmin()
        >>> np.nanmax()

        >>> from itertools import partial
        >>> partial(np.percentile, q=95)

    Notice: using functions like np.percentile might result in values that
    exceed the specified `out_range`!  (e.g. if the out-range is (0,1),
    a min-function of np.percentile(q=5) might result in negative values!)
    '''
    domain = domainfuncs[0](x), domainfuncs[1](x)
    #domain = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2



def update_progress(progress, max_prog=100,
                    title="", finalmsg=" DONE\r\n",
                    progress2 = None):
    '''
    print a progress-bar

    adapted from: https://blender.stackexchange.com/a/30739
    '''

    length = 25 # the length of the progress bar
    block = int(round(length*progress/max_prog))
    if progress2 is not None:
        msg = (f'\r{title} {"#"*block + "-"*(length-block)}' +
              f' {progress} [{progress2}] / {max_prog}')
    else:
        msg = (f'\r{title} {"#"*block + "-"*(length-block)}' +
               f' {progress} / {max_prog}')


    if progress >= max_prog: msg = f'\r{finalmsg:<79}\n'
    sys.stdout.write(msg)
    sys.stdout.flush()


def dt_to_hms(td):
    '''
    convert a datetime.timedelta object into days, hours, minutes and seconds
    '''
    days, hours, minutes  = td.days, td.seconds // 3600, td.seconds %3600//60
    seconds = td.seconds - hours*3600 - minutes*60
    return days, hours, minutes, seconds


def groupby_unsorted(a, key=lambda x: x, sort=False, get=lambda x: x):
    '''
    group the elements of the input-array and return it as a dict with a list
    of the found values. optionally use a key- and a get- function.

    if sort is True, a OrderedDict with sorted keys will be returned

    roughly equivalent to:

        >>> # if only the input-array a is provided
        ... {unique value of a: [found copies of the unique value]}
        ... # if a and a key-function is provided
        ... {key(a) : [...values with the same key(a)...]}
        ... # if both a key- and a get-function is provided
        ... {key(a) : [get(x) for x in ...values with the same key(a)...]}


    '''
    # always use an OrderedDict to ensure sort-order for python < 3.6
    d = OrderedDict()
    for item in a:
        d.setdefault(key(item), []).append(get(item))
    if sort is True:
        return OrderedDict(sorted(d.items()))
    else:
        return d
