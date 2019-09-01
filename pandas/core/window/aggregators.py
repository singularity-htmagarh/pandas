from functools import partial
from typing import Optional

import numba
import numpy as np

from pandas._typing import Scalar


class BaseAggregator:
    """
    Interface to return the current value of the rolling aggregation
    at the current step

    Attributes
    ----------
    values
    min_periods

    Methods
    -------
    query
    _meets_minimum_periods
    """

    def __init__(self, values: np.ndarray, min_periods: int) -> None:
        self.values = values
        self.min_periods = min_periods

    def query(self, start: int, stop: int) -> Optional[Scalar]:
        """
        Computes the result of an aggregation for values that are between
        the start and stop indices

        Parameters
        ----------
        start : int
            Current starting index

        stop : int
            Current stopping index

        Returns
        -------
        Scalar
            A scalar value that is the result of the aggregation.
        """
        raise NotImplementedError

    def _meets_minimum_periods(self, values: np.ndarray) -> bool:
        """
        Checks that the passed values contains more non-NaN values
        than min_periods.

        Parameters
        ----------
        values: ndarray
            values of the current window

        Returns
        -------
        Bool
        """
        not_null_counts = 0
        for value in values:
            if not np.isnan(value):
                not_null_counts += 1
            if not_null_counts >= self.min_periods:
                return True
        return False
        # Numba doesn't like count_nonzero
        # return np.count_nonzero(~np.isnan(values)) >= self.min_periods


class AggKernel:
    """
    Interface that computes the aggregation value

    Methods
    -------
    finalize
    make_aggregator
    """

    def __init__(self):
        pass

    def finalize(self):
        """Return the final value of the aggregation."""
        raise NotImplementedError

    def make_aggregator(
        self, values: np.ndarray, minimum_periods: int
    ) -> BaseAggregator:
        """Return an aggregator that performs the aggregation calculation"""
        raise NotImplementedError


class UnaryAggKernel(AggKernel):
    """Kernel to apply aggregations to singular inputs."""

    def step(self, value) -> None:
        """Update the state of the aggregation with `value`."""
        raise NotImplementedError

    def invert(self, value) -> None:
        """Undo the state of the aggregation with `value`."""
        raise NotImplementedError


agg_type = numba.deferred_type()


base_aggregator_spec = (
    ("values", numba.float64[:]),
    ("min_periods", numba.uint64),
    ("agg", agg_type),
    ("previous_start", numba.int64),
    ("previous_end", numba.int64),
)


@numba.jitclass(base_aggregator_spec)
class SubtractableAggregator(BaseAggregator):
    """
    Aggregator in which a current aggregated value
    is offset from a prior aggregated value.
    """

    def __init__(
        self, values: np.ndarray, min_periods: int, agg: UnaryAggKernel
    ) -> None:
        # Note: Numba doesn't like inheritance
        # super().__init__(values, min_periods)
        self.values = values
        self.min_periods = min_periods
        self.agg = agg
        self.previous_start = -1
        self.previous_end = -1

    def query(self, start: int, stop: int) -> Optional[Scalar]:
        """Compute a value based on changes in bounds."""
        if self.previous_start == -1 and self.previous_end == -1:
            # First aggregation over the values
            for value in self.values[start:stop]:
                self.agg.step(value)
        else:
            # Subsequent aggregations are calculated based on prior values
            for value in self.values[self.previous_start : start]:
                self.agg.invert(value)
            for value in self.values[self.previous_end : stop]:
                self.agg.step(value)
        self.previous_start = start
        self.previous_end = stop
        if self._meets_minimum_periods(self.values[start:stop]):
            return self.agg.finalize()
        # Numba wanted this to be None instead of None
        return np.nan


class Sum(UnaryAggKernel):
    def __init__(self) -> None:
        self.count = 0
        self.total = 0

    def step(self, value) -> None:
        if not np.isnan(value):
            self.count += 1
            self.total += value

    def invert(self, value) -> None:
        """Used only in subtractable kernels."""
        if not np.isnan(value):
            self.count -= 1
            self.total -= value

    def finalize(self) -> Optional[int]:
        if not self.count:
            return None
        return self.total

    def combine(self, other) -> None:
        """
        Combine the total and count from another kernel.
        Used only in segment tree aggregator.
        """
        self.total += other.total
        self.count += other.count

    def make_aggregator(self, values: np.ndarray, min_periods: int) -> BaseAggregator:
        aggregator = SubtractableAggregator(values, min_periods, self)
        return aggregator


sum_spec = (("count", numba.uint64), ("total", numba.float64))


@numba.jitclass(sum_spec)
class Mean(Sum):
    def finalize(self) -> Optional[float]:  # type: ignore
        if not self.count:
            return None
        return self.total / self.count


agg_type.define(Mean.class_type.instance_type)

aggregation_signature = (numba.float64[:], numba.int64[:], numba.int64[:], numba.int64)


@numba.njit(aggregation_signature, nogil=True, parallel=True)
def rolling_mean(
    values: np.ndarray,
    begin: np.ndarray,
    end: np.ndarray,
    minimum_periods: int,
    # kernel_class,  Don't think I can define this in the signature in nopython mode
) -> np.ndarray:
    """Perform a generic rolling aggregation"""
    aggregator = Mean().make_aggregator(values, minimum_periods)
    # aggregator = kernel_class().make_aggregator(values, minimum_periods)
    result = np.empty(len(begin))
    for i, (start, stop) in enumerate(zip(begin, end)):
        result[i] = aggregator.query(start, stop)
    return result


# rolling_mean = partial(rolling_aggregation, kernel_class=Mean)
