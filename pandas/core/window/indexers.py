"""
Note: In theory, we should be using jitclasses here,
but they are not as performant as njit.
https://github.com/numba/numba/issues/4522
"""
from typing import Optional, Sequence, Tuple, Union

import numba
import numpy as np

from pandas.tseries.offsets import DateOffset

BeginEnd = Tuple[np.ndarray, np.ndarray]


class BaseIndexer:
    """Base class for window bounds calculations"""

    def __init__(
        self,
        index=None,
        offset: Optional[Union[str, DateOffset]] = None,
        keys: Optional[Sequence[np.ndarray]] = None,
    ):
        """
        Parameters
        ----------
        index : ndarray[int64], default None
            pandas index to reference in the window bound calculation

        """
        self.index = index
        self.offset = offset
        self.keys = keys

    def get_window_bounds(
        self,
        num_values: int = 0,
        window_size: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        win_type: Optional[str] = None,
    ) -> BeginEnd:
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over

        window_size : int, default 0
            the number of rows in a window

        min_periods : int, default None
            min_periods passed from the top level rolling API

        center : bool, default None
            center passed from the top level rolling API

        closed : str, default None
            closed passed from the top level rolling API

        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        BeginEnd
            A tuple of ndarray[int64]s, indicating the boundaries of each
            window
        """
        raise NotImplementedError


class FixedWindowIndexer(BaseIndexer):
    """Calculate window boundaries that have a fixed window size"""

    @staticmethod
    @numba.njit(nogil=True)
    def get_window_bounds(
        num_values: int = 0,
        window_size: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        win_type: Optional[str] = None,
    ) -> BeginEnd:
        """
        Examples
        --------
        >>> FixedWindowIndexer().get_window_bounds(10, 2)
        (array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
         array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))

        >>> FixedWindowIndexer().get_window_bounds(5, 2)
        (array([0, 0, 1, 1, 2]), array([1, 2, 3, 4, 5]))
        """
        start_s = np.zeros(window_size, dtype=np.int64)
        start_e = np.arange(1, num_values - window_size + 1)
        start = np.concatenate((start_s, start_e))

        end = np.arange(1, num_values + 1)
        start = start[:num_values]
        end = end[:num_values]
        return start, end


class VariableWindowIndexer(BaseIndexer):
    """
    Calculate window boundaries with variable closed boundaries and index dependent
    """

    def get_window_bounds(
        self,
        num_values: int = 0,
        window_size: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        win_type: Optional[str] = None,
    ):
        return self._get_window_bounds(
            self.index, num_values, window_size, min_periods, center, closed, win_type
        )

    @staticmethod
    @numba.njit(nogil=True)
    def _get_window_bounds(
        index,
        num_values: int = 0,
        window_size: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        win_type: Optional[str] = None,
    ) -> BeginEnd:
        """
        Examples
        --------
        >>> variable = VariableWindowIndexer(np.arange(10))

        >>> variable.get_window_bounds(10, 2)
        (array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
         array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))

        >>> variable.get_window_bounds(10, 2, closed='left')
        (array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7]), array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

        >>> variable.get_window_bounds(10, 2, closed='both')
        (array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7]),
         array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))
        """
        left_closed = False
        right_closed = False

        # if windows is variable, default is 'right', otherwise default is 'both'
        if closed is None:
            closed = "right" if index is not None else "both"

        if closed == "both":
            left_closed = True
            right_closed = True

        elif closed == "right":
            right_closed = True

        elif closed == "left":
            left_closed = True

        start = np.empty(num_values, dtype=np.int64)
        start.fill(-1)

        end = np.empty(num_values, dtype=np.int64)
        end.fill(-1)

        start[0] = 0

        # right endpoint is closed
        if right_closed:
            end[0] = 1
        # right endpoint is open
        else:
            end[0] = 0

        # start is start of slice interval (including)
        # end is end of slice interval (not including)
        for i in range(1, num_values):
            end_bound = index[i]
            start_bound = index[i] - window_size

            # left endpoint is closed
            if left_closed:
                start_bound -= 1

            # advance the start bound until we are
            # within the constraint
            start[i] = i
            for j in range(start[i - 1], i):
                if index[j] > start_bound:
                    start[i] = j
                    break

            # end bound is previous end
            # or current index
            if index[end[i - 1]] <= end_bound:
                end[i] = i + 1
            else:
                end[i] = end[i - 1]

            # right endpoint is open
            if not right_closed:
                end[i] -= 1

        return start, end


class ExpandingIndexer(BaseIndexer):
    """Calculate expanding window bounds."""

    def get_window_bounds(
        self,
        num_values: int = 0,
        window_size: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        win_type: Optional[str] = None,
    ) -> BeginEnd:
        """
        Examples
        --------
        >>> ExpandingIndexer().get_window_bounds(10)

        (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
         array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))
        """
        return np.zeros(num_values, dtype=np.int64), np.arange(1, num_values + 1)
