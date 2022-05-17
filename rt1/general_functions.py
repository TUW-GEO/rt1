# -*- coding: utf-8 -*-
"""helper functions that are used both in rtfits and rtplots"""
from itertools import tee, islice, count, chain
from collections import OrderedDict
import keyword

try:
    import numpy as np
except ModuleNotFoundError:
    pass


def isidentifier(ident: str) -> bool:
    """
    Determines if string is valid Python identifier.
    taken from: https://stackoverflow.com/a/29586366/9703451
    """

    if not isinstance(ident, str):
        raise TypeError("expected str, but got {!r}".format(type(ident)))

    if not ident.isidentifier():
        return False

    if keyword.iskeyword(ident):
        return False

    return True


def rectangularize(array, return_mask=False, dim=None, return_masked=False, dtype=None):
    """
    return a rectangularized version of the input-array by repeating the
    last value to obtain the smallest possible rectangular shape.

        input:
            - array = [[1,2,3], [1], [1,2]]

        output:
            - return_masked=False: [[1,2,3], [1,1,1], [1,2,2]]
            - return_masked=True:  [[1,2,3], [1,--,--], [1,2,--]]

    Parameters
    ----------
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

    Returns
    -------
    new_array: array-like
               a rectangularized version of the input-array
    mask: array-like (only if 'weights_and_mask' is True)
          a mask indicating the added values

    """
    # use this method to get the dtype of the first element since it works with
    # pandas-Series, lists, arrays, dict-value views, etc.
    if dtype is None:
        dtype = np.array(next(islice(array, 1))).dtype

    if dim is None:
        # get longest dimension of sub-arrays
        dim = len(max(array, key=len))

    if return_mask is True or return_masked is True:
        newarray = np.empty((len(array), dim), dtype=dtype)
        mask = np.full((len(array), dim), False, dtype=bool)

        for i, s in enumerate(array):
            le = len(s)
            newarray[i, :le] = s
            newarray[i, le:] = s[-1]
            mask[i, le:] = True

        if return_masked is True:
            return np.ma.masked_array(newarray, mask)
        else:
            return [newarray, mask]
    else:
        newarray = np.empty((len(array), dim), dtype=dtype)
        for i, s in enumerate(array):
            le = len(s)
            newarray[i, :le] = s
            newarray[i, le:] = s[-1]
        return newarray


def meandatetime(datetimes):
    """
    calculate the average date from a given list of datetime-objects
    (can be applied to a pandas-Series via Series.apply(meandatetime))

    Parameters
    ----------
    datetimes: list
               a list of datetime-objects
    Returns
    -------
    meandate: Timestamp

    """
    if len(datetimes) == 1:
        return datetimes[0]

    x = datetimes
    deltas = (x[0] - x[1:]) / len(x)
    meandelta = sum(deltas)
    meandate = x[0] - meandelta
    return meandate


def dBsig0convert(val, inc, dB, sig0, fitdB, fitsig0):
    """
    A convenience-function to convert an array of measurements (and it's
    associated incidence-angles).
        - between linear- and dB units   `( val_dB = 10 * log10(val_linear) )`
        - between sigma0 and intensity   `( sig0 = 4 * pi * cos(inc) * I )`

    Parameters
    ----------
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

    Returns
    -------
    val : array-like
          the converted values

    """

    if sig0 is not fitsig0:
        # if results are provided in dB convert them to linear units before
        # applying the sig0-intensity conversion
        if fitdB is True:
            val = 10 ** (val / 10.0)
        # convert sig0 to intensity
        if sig0 is False and fitsig0 is True:
            val = val / (4.0 * np.pi * np.cos(inc))
        # convert intensity to sig0
        if sig0 is True and fitsig0 is False:
            val = 4.0 * np.pi * np.cos(inc) * val
        # convert back to dB if required
        if dB is True:
            val = 10.0 * np.log10(val)
    elif dB is not fitdB:
        # if dB output is required, convert to dB
        if dB is True and fitdB is False:
            val = 10.0 * np.log10(val)
        # if linear output is required, convert to linear units
        if dB is False and fitdB is True:
            val = 10 ** (val / 10.0)

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


def scale(x, out_range=(0, 1), domainfuncs=None):
    """
    scale an array between out_range = (min, max) where the range of the
    array is evaluated via the domainfuncs (min-function, max-funcion)

    the default domainfuncs are:

        >>> np.nanmin()
        >>> np.nanmax()

        >>> from itertools import partial
        >>> partial(np.percentile, q=95)

    Notice: using functions like np.percentile might result in values that
    exceed the specified `out_range`!  (e.g. if the out-range is (0,1),
    a min-function of np.percentile(q=5) might result in negative values!)
    """
    if domainfuncs is None:
        domain = np.nanmin(x), np.nanmax(x)
    else:
        domain = domainfuncs[0](x), domainfuncs[1](x)

    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def update_progress(
    progress, max_prog=100, title="", finalmsg=" DONE\r\n", progress2=None
):
    """
    print a progress-bar

    adapted from: https://blender.stackexchange.com/a/30739
    """

    length = 20  # the length of the progress bar
    block = int(round(length * progress / max_prog))
    if progress2 is not None:
        msg = (
            f'\r{title} {"#"*block + "-"*(length-block)}'
            + f" {progress} [{progress2}] / {max_prog}"
        )
    else:
        msg = (
            f'\r{title} {"#"*block + "-"*(length-block)}' + f" {progress} / {max_prog}"
        )

    if progress >= max_prog:
        msg = f"\r{finalmsg:<79}\n"

    return msg


def dt_to_hms(td):
    """
    convert a datetime.timedelta object into days, hours,
    minutes and seconds
    """

    days, hours, minutes = td.days, td.seconds // 3600, td.seconds % 3600 // 60
    seconds = td.seconds - hours * 3600 - minutes * 60
    return days, hours, minutes, seconds


def groupby_unsorted(a, key=lambda x: x, sort=False, get=lambda x: x):
    """
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

    """
    # always use an OrderedDict to ensure sort-order for python < 3.6
    d = OrderedDict()
    for item in a:
        d.setdefault(key(item), []).append(get(item))
    if sort is True:
        return OrderedDict(sorted(d.items()))
    else:
        return d


def interpolate_to_index(data, index, data_index=None, **interp1d_kwargs):
    """
    A wrapper around scipy.interp1d to interpolate a dataset to a given index

    Parameters
    ----------
    data : list, array-like, pandas.Series or pandas.DataFrame
        The input-data as list, array, pandas.Series or pandas.DataFrame
        If the data is provided as pandas Series or DataFrame, the index
        must support a method .to_julian_date() to convert the timestamps
        into numerical values.
    index : array-like
        the index to which the dataset should be interpolated.
        It must support a method .to_julian_date()
    data_index : TYPE, optional
        DESCRIPTION. The default is None.
    **interp1d_kwargs :
        additional keyword-arguments passed to scipy.interpolate.interp1d
        the default is (fill_value=None, bounds_error=False)

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    from pandas import Series, DataFrame
    from scipy.interpolate import interp1d

    kwargs = dict(fill_value=None, bounds_error=False)
    kwargs.update(interp1d_kwargs)

    if isinstance(data, Series):
        # perform a linear interpolation to the auxiliary data timestamps
        f = interp1d(data.index.to_julian_date(), data.values, **kwargs)
        x = f(index.to_julian_date())
        return Series(x, index)
    elif isinstance(data, DataFrame):
        f = interp1d(data.index.to_julian_date(), data.values, axis=0, **kwargs)
        x = f(index.to_julian_date())
        return DataFrame(x, index, columns=data.columns)

    elif isinstance(data, (list, np.ndarray)):
        assert data_index is not None, (
            'you must provide "data_index"' + "if data is provided as list or array"
        )

        f = interp1d(data_index.to_julian_date(), data.values, **kwargs)
        x = f(index.to_julian_date())
        return Series(x, index)


def find_missing(lst=[], N=None):
    """
    return a generator that yields sorted unique integers that are NOT present
    in the provided "lst" list.

    returned numbers will "fill up" missing values in the sorted "lst",
    starting at 0:

        >>> list(find_missing([2, 5, 6, 34, 68], 5))
        >>> [0, 1, 3, 4, 7]

    if the max. of the provided list is smaller than "N" the list is
    extended to obtain N unique numbers, e.g.:

        >>> list(find_missing([0, 1, 2], 5))
        >>> [3, 4, 5, 6, 7]

    Parameters
    ----------
    lst : iterable, optional
        an iterable with numbers that should be excluded in the list.
        The default is [].
    N : int, optional
        If N is provided, the generator will be exhausted after N values have
        been returned. (otherwise it will continue yielding results forever!)
    Returns
    -------
    missingIDs : list
        a list of N unique integers that are not present in "lst".

    """
    start = 0
    # get the max. ID defined in "lst"
    maxID = max(lst) if len(lst) > 0 else 0

    if len(lst) > 0:
        # check for IDs that can be used to "fill-up" the provided list
        missingIDs = sorted(set(range(start, maxID)) - set(lst))
    else:
        missingIDs = []

    # init a counter that yields numbers larger than the max ID found in "lst"
    c = count(maxID + 1)

    if N is not None:
        return islice(chain(missingIDs, c), N)
    else:
        return chain(missingIDs, c)


def chunkit(arr, chunk_size=10000):
    """
    split a array-like object into chunks of a pre-defined size

    Parameters
    ----------
    arr : array-like
        the data to split.
    chunk_size : int, optional
        the chunk-size. The default is 10000.

    Yields
    ------
    array-like
        the chunks of the input-array.
    """
    num_chunks = len(arr) // chunk_size
    if len(arr) % chunk_size != 0:
        num_chunks += 1
    for i in range(num_chunks):
        yield arr[i * chunk_size : (i + 1) * chunk_size]
