import abc
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from pandas.tseries.offsets import DateOffset

BeginEnd = Tuple[np.ndarray, np.ndarray]

# TODO: Currently, when win_type is specified, it calls a special routine,
#  `roll_window`, while None win_type ops dispatch to specific methods.
#  Consider consolidating?

# TODO: Handle "minp" = "min_periods" validation


class BaseIndexer(abc.ABC):
    """Base class for window bounds calculations"""
    def __init__(self,
                 index=None,
                 offset: Optional[Union[str, DateOffset]] = None,
                 keys: Optional[Sequence[np.ndarray]] = None):
        """
        Parameters
        ----------
        index : , default None
            pandas index to reference in the window bound calculation

        offset: str or DateOffset, default None
            Offset used to calcuate the window boundary

        keys: np.ndarray, default None
            Additional columns needed to calculate the window bounds

        """
        self.index = index
        self.offset = offset
        self.keys = keys

    @classmethod
    @abc.abstractmethod
    def get_window_bounds(cls,
                          values: Optional[np.ndarray] = None,  # "self.N" = len(values) = len(index)
                          window_size: int = 0,  # "self.win"
                          min_periods: Optional[int] = None,
                          center: Optional[bool] = None,
                          closed: Optional[str] = None,
                          win_type: Optional[str] = None) -> BeginEnd:
        """
        Computes the bounds of a window.

        Parameters
        ----------
        # TODO: should users have access to _all_ the values or just the length (len(values))?
        values : np.ndarray, default None
            values that will have the rolling operation applied

        window_size : int, default 0
            min_periods passed from the top level rolling API

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


class FixedWindowIndexer(BaseIndexer):

    def get_window_bounds(self,
                          values: Optional[np.ndarray] = None,
                          window_size: int = 0,  # "self.win"
                          min_periods: Optional[int] = None,
                          center: Optional[bool] = None,
                          closed: Optional[str] = None,
                          win_type: Optional[str] = None):
        num_values = len(values) if values is not None else 0
        start_s = np.zeros(window_size, dtype=np.int64)
        start_e = np.arange(window_size, num_values, dtype=np.int64) - window_size + 1
        start = np.concatenate([start_s, start_e])

        end_s = np.arange(window_size, dtype=np.int64) + 1
        end_e = start_e + window_size
        end = np.concatenate([end_s, end_e])
        return start, end


class VariableWindowIndexer(BaseIndexer):

    def get_window_bounds(self,
                          values: Optional[np.ndarray] = None,
                          window_size: int = 0,
                          min_periods: Optional[int] = None,
                          center: Optional[bool] = None,
                          closed: Optional[str] = None,
                          win_type: Optional[str] = None):

        # TODO: Move this close validation upstream (it always applied to all indexers)
        left_closed = False
        right_closed = False

        # if windows is variable, default is 'right', otherwise default is 'both'
        if closed is None:
            closed = 'right' if self.index is not None else 'both'

        if closed in ['right', 'both']:
            right_closed = True

        if closed in ['left', 'both']:
            left_closed = True

        num_values = len(values) if values is not None else 0

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
            end_bound = self.index[i]
            start_bound = self.index[i] - window_size

            # left endpoint is closed
            if left_closed:
                start_bound -= 1

            # advance the start bound until we are
            # within the constraint
            start[i] = i
            for j in range(start[i - 1], i):
                if self.index[j] > start_bound:
                    start[i] = j
                    break

            # end bound is previous end
            # or current index
            if self.index[end[i - 1]] <= end_bound:
                end[i] = i + 1
            else:
                end[i] = end[i - 1]

            # right endpoint is open
            if not right_closed:
                end[i] -= 1

        return start, end


def _check_minp(win: int, minp: Optional[int], N: int, floor: Optional[int] = None):
    """
    Parameters
    ----------
    win: int
    minp: int or None
    N: len of window
    floor: int, optional
        default 1

    Returns
    -------
    minimum period
    """

    if minp is None:
        minp = 1
    if not isinstance(minp, int):
        raise ValueError("min_periods must be an integer")
    if minp > win:
        raise ValueError("min_periods (%d) must be <= "
                         "window (%d)" % (minp, win))
    elif minp > N:
        minp = N + 1
    elif minp < 0:
        raise ValueError('min_periods must be >= 0')
    if floor is None:
        floor = 1

    return max(minp, floor)


class MockFixedWindowIndexer:
    """

    We are just checking parameters of the indexer,
    and returning a consistent API with fixed/variable
    indexers.

    Parameters
    ----------
    values: ndarray
        values data array
    win: int64_t
        window size
    minp: int64_t
        min number of obs in a window to consider non-NaN
    index: object
        index of the values
    floor: optional
        unit for flooring
    left_closed: bint
        left endpoint closedness
    right_closed: bint
        right endpoint closedness

    """
    def __init__(self,
                 values: np.ndarray,
                 win: int,
                 minp: int,
                 left_closed: bool,
                 right_closed: bool,
                 index=None,
                 floor=None):

        assert index is None
        self.is_variable = 0
        self.N = len(values)
        self.start = np.empty(0, dtype='int64')
        self.end = np.empty(0, dtype='int64')
        self.win = win
