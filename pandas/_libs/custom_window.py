import abc
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from pandas._typing import Scalar
from pandas.tseries.offsets import DateOffset

BeginEnd = Tuple[np.ndarray, np.ndarray]

# TODO: Refactor MockFixedWindowIndexer, FixedWindowIndexer,
#  VariableWindowIndexer to also have `get_window_bounds` methods that
#  only calculates start & stop

# TODO: Currently, when win_type is specified, it calls a special routine,
#  `roll_window`, while None win_type ops dispatch to specific methods.
#  Consider consolidating?


class BaseIndexer(abc.ABC):
    """Base class to calculate custom rolling window bounds"""
    def __init__(self, index, offset, keys) -> None:
        # TODO: The alternative is for the `rolling` API to accept
        #  index, offset, and keys as keyword arguments
        self.index = index
        self.offset = offset  # type: Union[str, DateOffset]
        self.keys = keys  # type: Sequence[np.ndarray]

    @classmethod
    @abc.abstractmethod
    def get_window_bounds(
        cls,
        win_type: Optional[str] = None,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
    ) -> BeginEnd:
        """
        Compute the bounds of a window.

        Users should subclass this class to implement a custom method
        to calculate window bounds

        Parameters
        ----------
        win_type : str, default None
            win_type passed from the top level rolling API

        min_periods : int, default None
            min_periods passed from the top level rolling API

        center : bool, default None
            center passed from the top level rolling API

        closed : str, default None
            closed passed from the top level rolling API

        Returns
        -------
        BeginEnd
            A tuple of ndarray[int64]s, indicating the boundaries of each
            window

        """


def _check_minp(win, minp, N, floor=None):
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
    if isinstance(minp, int):
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


# TODO implement in terms of BaseIndexer
class WindowIndexer:

    def __init__(self):
        self.start = None
        self.end = None

    def get_window_bounds(self):
        return self.start, self.end


class MockFixedWindowIndexer(WindowIndexer):
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
        self.minp = _check_minp(win, minp, self.N, floor=floor)
        self.start = np.empty(0, dtype='int64')
        self.end = np.empty(0, dtype='int64')
        self.win = win


class FixedWindowIndexer(WindowIndexer):
    """
    create a fixed length window indexer object
    that has start & end, that point to offsets in
    the index object; these are defined based on the win
    arguments

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
        unit for flooring the unit
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
        self.minp = _check_minp(win, minp, self.N, floor=floor)

        start_s = np.zeros(win, dtype='int64')
        start_e = np.arange(win, self.N, dtype='int64') - win + 1
        self.start = np.concatenate([start_s, start_e])

        end_s = np.arange(win, dtype='int64') + 1
        end_e = start_e + win
        self.end = np.concatenate([end_s, end_e])
        self.win = win


class VariableWindowIndexer(WindowIndexer):
    """
    create a variable length window indexer object
    that has start & end, that point to offsets in
    the index object; these are defined based on the win
    arguments

    Parameters
    ----------
    values: ndarray
        values data array
    win: int64_t
        window size
    minp: int64_t
        min number of obs in a window to consider non-NaN
    index: ndarray
        index of the values
    left_closed: bint
        left endpoint closedness
        True if the left endpoint is closed, False if open
    right_closed: bint
        right endpoint closedness
        True if the right endpoint is closed, False if open
    floor: optional
        unit for flooring the unit
    """
    def __init__(self,
                 values: np.ndarray,
                 win: int,
                 minp: int,
                 left_closed: bool,
                 right_closed: bool,
                 index: np.ndarray,
                 floor=None):

        self.is_variable = 1
        self.N = len(index)
        self.minp = _check_minp(win, minp, self.N, floor=floor)

        self.start = np.empty(self.N, dtype='int64')
        self.start.fill(-1)

        self.end = np.empty(self.N, dtype='int64')
        self.end.fill(-1)

        self.build(index, win, left_closed, right_closed)

        # max window size
        self.win = (self.end - self.start).max()

    def build(self,
              index: np.ndarray,
              win: int,
              left_closed: bool,
              right_closed: bool):

        start = self.start
        end = self.end
        N = self.N

        start[0] = 0

        # right endpoint is closed
        if right_closed:
            end[0] = 1
        # right endpoint is open
        else:
            end[0] = 0

        # start is start of slice interval (including)
        # end is end of slice interval (not including)
        for i in range(1, N):
            end_bound = index[i]
            start_bound = index[i] - win

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


def get_window_indexer(values, win, minp, index, closed,
                       floor=None, use_mock=True, custom_indexer=None):
    """
    return the correct window indexer for the computation

    Parameters
    ----------
    values: 1d ndarray
    win: integer, window size
    minp: integer, minimum periods
    index: 1d ndarray, optional
        index to the values array
    closed: string, default None
        {'right', 'left', 'both', 'neither'}
        window endpoint closedness. Defaults to 'right' in
        VariableWindowIndexer and to 'both' in FixedWindowIndexer
    floor: optional
        unit for flooring the unit
    use_mock: boolean, default True
        if we are a fixed indexer, return a mock indexer
        instead of the FixedWindow Indexer. This is a type
        compat Indexer that allows us to use a standard
        code path with all of the indexers.
    custom_indexer: BaseIndexer, default None
        If not None, use the BaseIndexer (subclass) in order to calculate the window bounds
    TODO: add the kwargs for BaseIndexer.get_window_bounds


    Returns
    -------
    tuple of 1d int64 ndarrays of the offsets & data about the window

    """

    left_closed = False
    right_closed = False

    assert closed is None or closed in ['right', 'left', 'both', 'neither']

    # if windows is variable, default is 'right', otherwise default is 'both'
    if closed is None:
        closed = 'right' if index is not None else 'both'

    if closed in ['right', 'both']:
        right_closed = True

    if closed in ['left', 'both']:
        left_closed = True

    if custom_indexer is not None:
        return custom_indexer.get_window_bounds()
    if index is not None:
        indexer = VariableWindowIndexer(values, win, minp, left_closed,
                                        right_closed, index, floor)
    elif use_mock:
        indexer = MockFixedWindowIndexer(values, win, minp, left_closed,
                                         right_closed, index, floor)
    else:
        indexer = FixedWindowIndexer(values, win, minp, left_closed,
                                     right_closed, index, floor)
    return indexer.get_window_bounds()



class BaseAggregator(abc.ABC):
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    @abc.abstractmethod
    def query(
        self, start: int, stop: int, previous_start: int, previous_end: int
    ) -> Scalar:
        """
        Computes the result of an aggregation for values that are between
        the start and stop indices

        Parameters
        ----------
        start : int
            Current starting index

        stop : int
            Current stopping index

        previous_start : int
            Prior starting index

        previous_end : int
            Prior ending index

        Returns
        -------
        Scalar
            A scalar value that is the result of the aggregation.

        """

class AggKernel(abc.ABC):
    def __init__(self):
        self.count = 0

    @abc.abstractmethod
    def finalize(self):
        """Return the final value of the aggregation."""


class UnaryAggKernel(AggKernel):
    @abc.abstractmethod
    def step(self, value) -> None:
        """Update the state of the aggregation with `value`."""


class SubtractableAggregator(BaseAggregator):
    def __init__(self, values: np.ndarray, agg: AggKernel) -> None:
        super().__init__(values)
        self.agg = agg

    def query(
        self, start: int, stop: int, previous_start: int, previous_end: int
    ) -> Scalar:
        """Compute a value based on changes in bounds."""
        if previous_start == -1 and previous_end == -1:
            for value in self.values[start:stop]:
                self.agg.step(value)
        else:
            for value in self.values[previous_start:start]:
                self.agg.invert(value)
            for value in self.values[previous_end:stop]:
                self.agg.step(value)
        return self.agg.finalize()


class Sum(UnaryAggKernel):
    def __init__(self) -> None:
        super().__init__()
        self.total = 0

    def step(self, value) -> None:
        if value is not None:
            self.count += 1
            self.total += value

    def invert(self, value) -> None:
        """Used only in subtractable kernels."""
        if value is not None:
            self.count -= 1
            self.total -= value

    def finalize(self) -> Optional[int]:
        if not self.count:
            return None
        return self.total

    def combine(self, other) -> None:
        """Used only in segment tree aggregator."""
        self.total += other.total
        self.count += other.count

    @classmethod
    def make_aggregator(
        cls, values: np.ndarray
    ) -> BaseAggregator:
        aggregator = SubtractableAggregator(values, cls())
        return aggregator


class Mean(Sum):

    def finalize(self) -> Optional[float]:
        if not self.count:
            return None
        return self.total / self.count


def rolling_aggregation(
    values: np.ndarray,
    begin: np.ndarray,
    end: np.ndarray,
    kernel_class,
) -> np.ndarray:
    """Perform a generic rolling aggregation"""
    aggregator = kernel_class.make_aggregator(values)
    previous_start = previous_end = -1
    result = np.empty(len(begin))
    for i, (start, stop) in enumerate(zip(begin, end)):
        result[i] = aggregator.query(
            start, stop, previous_start, previous_end
        )
        previous_start = start
        previous_end = end
    return result
