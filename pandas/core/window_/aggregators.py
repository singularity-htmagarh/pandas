import abc
from typing import Optional

import numpy as np

from pandas._typing import Scalar


class BaseAggregator(abc.ABC):
    """Interface to return the current value of the rolling aggregation at the current step"""
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
    """Interface that computes the aggregation value"""
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
    def make_aggregator(cls, values: np.ndarray) -> BaseAggregator:
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
