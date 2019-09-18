"""
Implementation of the rolling aggregations using njit methods.
This implementation mimics what we currently do in cython except the
calculation of window bounds is independent of the aggregation routine.
"""
from typing import Callable

import numba
import numpy as np


@numba.njit(nogil=True)
def rolling_mean(
    values: np.ndarray, begin: np.ndarray, end: np.ndarray, minimum_periods: int
) -> np.ndarray:
    """
    Compute a rolling mean over values.

    Parameters
    ----------
    values : ndarray[float64]
        values to roll over

    begin : ndarray[int64]
        starting indexers

    end : ndarray[int64]
        ending indexers

    minimum_periods : ndarray[float64]
        minimum

    Returns
    -------
    ndarray[float64]
    """
    result = np.empty(len(begin))
    previous_start = -1
    previous_end = -1
    count = 0
    total = 0
    for i, (start, stop) in enumerate(zip(begin, end)):
        if previous_start == -1 and previous_end == -1:
            # First aggregation over the values
            for value in values[start:stop]:
                if not np.isnan(value):
                    count += 1
                    total += value
        else:
            # Subsequent aggregations are calculated based on prior values
            for value in values[previous_start:start]:
                if not np.isnan(value):
                    count -= 1
                    total -= value
            for value in values[previous_end:stop]:
                if not np.isnan(value):
                    count += 1
                    total += value
        previous_start = start
        previous_end = stop
        val = np.nan
        if count and count >= minimum_periods:
            val = total / count
        result[i] = val
    return result


@numba.njit(nogil=True)
def rolling_apply(
    values: np.ndarray,
    begin: np.ndarray,
    end: np.ndarray,
    minimum_periods: int,
    numba_func: Callable,
    args,
) -> np.ndarray:
    result = np.empty(len(begin))
    for i, (start, stop) in enumerate(zip(begin, end)):
        window = values[start:stop]
        count_nan = np.sum(np.isnan(window))
        if len(window) - count_nan >= minimum_periods:
            result[i] = numba_func(window, *args)
        else:
            result[i] = np.nan
    return result
