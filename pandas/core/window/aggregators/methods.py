"""
Implementation of the rolling aggregations using njit methods.
This implementation mimics what we currently do in cython except the
calculation of window bounds is independent of the aggregation routine.
"""
import numba
import numpy as np


@numba.njit(nogil=True)
def rolling_mean_generic(
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
def rolling_mean_fixed_window(
    values: np.ndarray,
    begin: np.ndarray,
    end: np.ndarray,
    minimum_periods: int,
    window: int,
) -> np.ndarray:
    """
    Compute a rolling mean over values for a fixed window size.

    Parameters
    ----------
    values : ndarray[float64]
        values to roll over

    begin : ndarray[int64]
        starting indexers

    end : ndarray[int64]
        ending indexers

    minimum_periods : int
        minimum periods in the window to aggregate

    window : int
        window size

    Returns
    -------
    ndarray[float64]
    """
    result = np.empty(len(values))
    count = 0
    total = 0
    for i in range(minimum_periods - 1):
        value = values[i]
        if not np.isnan(value):
            count += 1
            total += value
        result[i] = np.nan
    for i in range(minimum_periods - 1, len(values)):
        value = values[i]
        if not np.isnan(value):
            count += 1
            total += value
        if i > window - 1:
            previous_value = values[i - window]
            if not np.isnan(previous_value):
                count -= 1
                total -= previous_value
        if count and count >= minimum_periods:
            result[i] = total / count
        else:
            result[i] = np.nan
    return result
