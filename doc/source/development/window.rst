Generalized Window Operations
=============================

Rationale
---------
The current window implementation in :file:`window.pyx` is difficult to
customize and couples the creation of window bounds, loop over the values to
aggregate, the logic for updating an aggregation. The aggregation algorithms
are also duplicated in :file:`groupby.pyx`, but unifying grouping aggregation
algorithms and windowing aggregation algorithms is out of scope for this
proposal

Requirements
------------
#. Pass all existing user-facing API tests
#. User-defined aggregations (UDAs) performance on par with builtin aggregations
#. Ideally no performance regressions
#. Performant support for custom offsets (e.g., custom business
   days)
#. New features are strict additions to the API

Soft Requirements
-----------------
#. Lower the maintenance burden of the existing implementation by using numba
   if possible
#. Reduce code duplication

Implementation
--------------
Here we describe details of the implementation that we think satisfy the
requirements.

Ideally, the implementation is able to use ``numba`` to implement the
performance sensitive parts of the proposal.

Aggregation
~~~~~~~~~~~
We propose to split aggregations into two parts: the algorithm used to compute
the value of an aggregation for a given row, and a class used for aggregation.

The aggregation algorithm is an interface that when given a start and stop
point representing indices into a :attr:`values` one-dimensional array, knows
how to efficiently compute the value of an aggregation in between start and
stop.

For example:

.. code-block:: python

   import abc
   from typing import Generic, Callable, TypeVar, Union

   Input1 = TypeVar("Input1")
   Input2 = TypeVar("Input2")
   Output = TypeVar("Output")

   class AggregationAlgorithm(abc.ABC):
       def __init__(
           self, values: ndarray, agg: Union[Callable[...], Kernel]
       ) -> None:
           self.values = values
           self.agg = agg

       @abc.abstractmethod
       def query(
           self, start: int, stop: int, previous_start: int, previous_end: int
       ) -> Output:
           ...

   class SegmentTreeAggregator(AggregationAlgorithm):
       def query(
           self, start: int, stop: int, previous_start: int, previous_end: int
       ) -> Output:
           """Compute a value by walking nodes of a segment tree."""

   class SubtractableAggregator(AggregationAlgorithm):
       def query(
           self, start: int, stop: int, previous_start: int, previous_end: int
       ) -> Output:
           """Compute a value based on changes in bounds."""

   class ApplyAggregator(AggregationAlgorithm):
       def query(
           self, start: int, stop: int, previous_start: int, previous_end: int
       ) -> Output:
           """Compute a value by applying a function over a range of values."""
           return self.agg(self.values[start:stop])


A *kernel* is an interface that implements methods to update its internal state
as well as a :meth:`finalize` method to return the current value of the
aggregation. Its constructor must take no arguments.

An example implementation of a :class:`Sum` kernel would look similar to the
following:

.. code-block:: python

   class AggKernel(Generic[Output]):
       def __init__(self):
           self.count = 0

       @abc.abstractmethod
       def finalize(self) -> Optional[Output]:
           """Return the final value of the aggregation."""


   class UnaryAggKernel(Generic[Input1, Output], AggKernel[Output]):
       @abc.abstractmethod
       def step(self, value: Optional[Input1]) -> None:
           """Update the state of the aggregation with `value`."""


   class Sum(UnaryAggKernel[Input1, Output]):
       def __init__(self):
           super().__init__()
           self.total: Output = 0

       def step(self, value: Optional[Input1]) -> None:
           if value is not None:
               self.count += 1
               self.total += value

       def invert(self, value: Optional[Input1]) -> None:
           """Used only in subtractable kernels."""
           if value is not None:
               self.count -= 1
               self.total -= value

       def finalize(self) -> Optional[Output]:
           if not self.count:
               return None
           return self.total

       def combine(self, other: Sum[Input1, Output]) -> None:
           """Used only in segment tree aggregator."""
           self.total += other.total
           self.count += other.count

       @classmethod
       def make_aggregator(
           cls, values: ndarray[Input]
       ) -> AggregationAlgorithm:
           SubtractableAggregator.check_agg(cls)
           aggregator = SubtractableAggregator(values, cls())
           return aggregator

   class BinaryAggKernel(Generic[Input1, Input2, Output], AggKernel[Output]):
       @abc.abstractmethod
       def step(
           self, value1: Optional[Input1], value2: Optional[Input2]
       ) -> None:
           ...

   class Covariance(BinaryAggKernel[Input1, Input2, float])
       def step(
           self, value1: Optional[Input1], value2: Optional[Input2]
       ) -> None:
           if value1 is not None and value2 is not None:
              ...

       def finalize(self) -> Optional[float]:
           ...


Customization of Window Spans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We propose a generic mechanism that allows power users and library authors to
customize the computation of window boundaries.

.. code-block:: python

   import numpy as np
   from typing import Sequence, Tuple

   BeginEnd = Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]
   Displacement = TypeVar("Displacement")

   class Indexer(abc.ABC):
       @abc.abstractmethod
       @classmethod
       def get_window_bounds(
           cls, index, offset, keys: Sequence[ndarray[Any]]
       ) -> BeginEnd:
           """Compute the bounds of a window.

           Parameters
           ----------
           index
               A pandas index to compute indices against
           offset
               An object that can be used to calculate the displacment for each
               element
           keys
               A possibly empty list of additional columns needed to compute
               window bounds

           Returns
           -------
           BeginEnd
               A tuple of ndarray[int64]s, indicating the boundaries of each
               window

           """

Example Aggregation Loop Implemention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here is an example implementation of a loop that would perform the aggregation
using the interfaces proposed above.

.. code-block:: python

   T = TypeVar("T")

   def do_agg(
       values: np.ndarray[T],
       index: np.ndarray[np.int64],
       offset: Displacement,
       indexer_class: Type[Indexer],
       kernel_class: Type[Kernel],
       keys: Sequence[ndarray[Any]],
   ) -> ndarray:
       result = np.empty(...)
       begin, end = indexer_class.get_window_bounds(index, offset, keys)
       aggregator = kernel_class.make_aggregator(values)
       previous_start = previous_end = -1
       for i, (start, stop) in enumerate(zip(begin, end)):
           result[i] = aggregator.query(
               start, stop, previous_start, previous_stop
           )
           previous_start = start
           previous_end = end
       return result
