# -*- coding: utf-8 -*-
"""
helper functions that are used both in rtfits and rtplots
"""

import numpy as np
import pandas as pd
import datetime

def rectangularize(array, weights_and_mask=False, dim=None,
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
    weights_and_mask: bool (default = False)
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
    weights: array-like (only if 'weights_and_mask' is True)
             a weighting-matrix whose entries are 1/sqrt(number of repetitions)
             (the square-root is used since this weighting will be applied to
             a sum of squares)
    mask: array-like (only if 'weights_and_mask' is True)
          a mask indicating the added values

    '''
    if dim is None:
        # get longest dimension of sub-arrays
        dim  = len(max(array, key=len))

    if weights_and_mask is True:
        newarray, weights, mask = [], [], []
        for s in array:
            adddim = dim - len(s)
            w = np.full_like(s, 1)
            m = np.full_like(s, False)
            if adddim > 0:
                s = np.append(s, np.full(adddim, s[-1]))
                w = np.append(w, np.full(adddim, 1/np.sqrt(adddim)))
                m = np.append(m, np.full(adddim, True))
            newarray += [s]
            weights  += [w]
            mask     += [m]

        newarray = np.array(newarray)
        weights = np.array(weights)
        mask = np.array(mask, dtype=bool)

        if return_masked is True:
            newarray = np.ma.masked_array(newarray, mask)

        return [newarray, weights, mask]

    elif return_masked is True:
        newarray, mask = [], []
        for s in array:
            adddim = dim - len(s)
            m = np.full_like(s, False)
            if adddim > 0:
                s = np.append(s, np.full(adddim, s[-1]))
                m = np.append(m, np.full(adddim, True))
            newarray += [s]
            mask     += [m]

        mask = np.array(mask, dtype=bool)
        newarray = np.array(newarray)

        return np.ma.masked_array(newarray, mask)

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

    if np.count_nonzero(datetimes) == 1:
        return datetimes[0]

    x = pd.to_datetime(datetimes)
    deltas = (x[0] - x[1:])/len(x)
    meandelta = sum(deltas, datetime.timedelta(0))
    meandate = x[0] - meandelta
    return meandate

