Generalized Window Operations
=============================

Rationale
---------
The current window implementation in :file:`window.pyx` is difficult to
customize and couples the creation of window bounds, loop over the values to
aggregate, the logic for updating an aggregation.

Requirements
------------
#. Pass all existing user-facing API tests
#. User-defined aggregations performance on par with "builtins"
   #. UDAs can be implemented the same way as builtins
#. Ideally no performance regressions
#. Performant support for custom offsets (e.g., custom business
   days)
#. New features are strict additions to the API

Soft Requirements
-----------------
#. Lower the maintenance burden of the existing implementation
#. Reduce code duplication

Implementation
--------------
Here we describe details of the implementation that we think satisfy the
requirements.

Aggregation
~~~~~~~~~~~
We propose to split aggregations into two parts: the algorithm used to compute
the value of an aggregation for a given row, and a class used for aggregation.

The aggregation algorithm is an interface that when given a start and stop
point representing indices into a :attr:`values` ndarray, knows how to
efficiently compute the value of an aggregation in between start and stop.

For example:

.. code-block:: python

   import abc
   from typing import Union, Callable

   class AggregationAlgorithm(abc.ABC):
       def __init__(
           self, values: ndarray, agg: Union[Callable[...], Agg]
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
           ...

   class SubtractableAggregator(AggregationAlgorithm):
       def query(
           self, start: int, stop: int, previous_start: int, previous_end: int
       ) -> Output:
           ...

   class ApplyAggregator(AggregationAlgorithm):
       def query(
           self, start: int, stop: int, previous_start: int, previous_end: int
       ) -> Output:
           return self.agg(self.values[start:stop])


An aggregation is an interface that implements methods to update its internal
state and a :meth:`finalize` method to return the current value of the
aggregation based on the internal state.

An example implementation of a :class:`Sum` kernel would look similar to

.. code-block:: python

   from typing import Generic, TypeVar

   Input1 = TypeVar("Input1")
   Input2 = TypeVar("Input2")
   Output = TypeVar("Output")

   class UnaryAggregation(Generic[Input1, Output]):
       @abc.abstractmethod
       def step(self, value: Optional[Input1]) -> None:
           ...

   class Sum(UnaryAggregation[Input1, Output]):
       def __init__(self):
           self.count: int = 0
           self.total: Output = 0

       def step(self, value: Optional[Input1]) -> None:
           if value is not None:
               self.count += 1
               self.total += value

       def invert(self, value: Optional[Input1]) -> None:
           """Used only in subtractable aggregator."""
           if value is not None:
               self.count -= 1
               self.total -= value

       def finalize(self) -> Optional[Output]:
           if not self.count:
               return None
           return self.total

       def combine(self, other: Sum) -> None:
           """Used only in segment tree aggregator."""
           self.total += other.total
           self.count += other.count

       @classmethod
       def make_aggregator(cls, values: ndarray) -> AggregationAlgorithm:
           SubtractableAggregator.check_agg(cls)
           aggregator = SubtractableAggregator(values, cls())
           return aggregator

   class BinaryAggregation(Generic[Input1, Input2, Output]):
       @abc.abstractmethod
       def step(
           self, value1: Optional[Input1], value2: Optional[Input2]
       ) -> None:
           ...

   class Covariance(BinaryAggregation[Input1, Input2, float])
       ...


Customization of Window Spans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We propose a generic mechanism that allows power users and library authors to
customize the computation of window boundaries.

.. code-block:: python

   import abc
   from typing import Tuple

   BeginEnd = Tuple[ndarray, ndarray]

   class Indexer(abc.ABC):
       @abc.abstractmethod
       @classmethod
       def get_window_bounds(cls, index, offset) -> BeginEnd:
           ...

   class MyBusinessDayIndexer(Indexer):
       @classmethod
       def get_window_bounds(cls, index, offset) -> BeginEnd:
           ...

Example Aggregation Loop Implemention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here is an example implementation of a loop that would perform the aggregation
using the interfaces proposed above.

.. code-block:: python

   def do_agg(
       values: ndarray,
       index: ndarray,
       offset,
       indexer_class: Type[Indexer],
       agg_class: Type[Agg],
   ) -> ndarray:
       result = np.empty(...)
       begin, end = indexer_class.get_window_bounds(index, offset)
       aggregator = agg_class.make_aggregator(values)
       previous_start = previous_end = -1
       for i, (start, stop) in enumerate(zip(begin, end)):
           result[i] = aggregator.query(
               start, stop, previous_start, previous_stop
           )
           previous_start = start
           previous_end = end
       return result
