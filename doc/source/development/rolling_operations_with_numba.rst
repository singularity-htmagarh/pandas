Rolling Operations with Numba
=============================

Summary
-------

This proof of concept (POC) demonstrates using `Numba <http://numba.pydata.org/>`_ instead of ``Cython``
to compute ``rolling.mean`` and ``rolling.apply`` without introducing any user facing API changes.
The benefits of using ``Numba`` include:

#. Performance parity or improvement over ``Cython``
#. Eliminate shipping C-extensions
#. Ease of debugging and maintenance as pure Python code

Additionally this POC includes an new API that allows users to create custom window boundary calculations
when using ``Numba``.

Immediate Proposal
~~~~~~~~~~~~~~~~~~

In line with the `pandas roadmap <https://pandas.pydata.org/pandas-docs/stable/development/roadmap.html#numba-accelerated-operations>`_,
we would like to introduce ``Numba`` as a required dependency for pandas in the 1.0 release
and have the ``rolling.mean`` and ``rolling.apply`` operations use ``Numba`` instead of ``Cython`` in the release.

Implementation
--------------

``rolling.mean`` and ``rolling.apply`` utilizes ``njit`` functions to separately calculate:

* Window boundaries based on the window size and index
* The actual aggregation function (mean and apply in this POC)

The functional implementation mimics the current ``Cython`` implementation; however, now that
the calculation of the window boundaries is separate from the aggregation function, we are able to
expose a new API to users that allows them to specify how to calculate the rolling window boundaries.

New Custom Window Indexer API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, window bounds are calculated automatically based on the ``DataFrame`` or ``Series`` index
when a user passes an integer or offset in the rolling API (e.g. ``df.rolling(2)``). This POC also introduces
a ``BaseIndexer`` class is available from ``pandas/core/window/indexers.py`` for users to subclass
to create a custom routine to calculate window boundaries. Users will need to specify a
``get_window_bounds`` function to calculate window boundaries.

Below is an example of creating an ``ExpandingIndexer`` computes an expanding window similar to
the current ``expanding`` API:

.. code-block:: python

   from typing import Optional

   from pandas import Series
   from pandas.core.window.indexers import BaseIndexer

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
       ):
           """
           Examples
           --------
           >>> ExpandingIndexer().get_window_bounds(10)

           (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))
           """
           return np.zeros(num_values, dtype=np.int64), np.arange(1, num_values + 1)

   >>> s = Series(range(5))
   >>> s.rolling(ExpandingIndexer()).mean()
   0    0.0
   1    0.5
   2    1.0
   3    1.5
   4    2.0
   dtype: float64
   >>> s.expanding().mean()
   0    0.0
   1    0.5
   2    1.0
   3    1.5
   4    2.0
   dtype: float64

Performance
-----------

Below is table comparing the current performance difference between the Numba and Cython implementations
for 1 million data points. (Exact benchmark setup can be found in the Appendix)

+-------------------------+------------------+-----------------+
| Speed                   | Numba            | Cython          |
+=========================+==================+=================+
| mean (fixed window)     | 22.9 ms ± 2.31 ms| 21 ms ± 1.03 ms |
+-------------------------+------------------+-----------------+
| mean (offset window)    | 38.3 ms ± 1.52 ms| 28.2 ms ± 735 µs|
+-------------------------+------------------+-----------------+
| apply (fixed window)    | 579 ms ± 9.23 ms | 3.29 s ± 117 ms |
+-------------------------+------------------+-----------------+
| apply (offset window)   | 574 ms ± 5.11 ms | 3.54 s ± 98.6 ms|
+-------------------------+------------------+-----------------+

+-------------------------+------------------+-----------------+
| Peak Memory             | Numba            | Cython          |
+=========================+==================+=================+
| mean (fixed window)     | 232.32 MiB       | 161.13 MiB      |
+-------------------------+------------------+-----------------+
| mean (offset window)    | 243.85 MiB       | 177.08 MiB      |
+-------------------------+------------------+-----------------+
| apply (fixed window)    | 264.93 MiB       | 177.12 MiB      |
+-------------------------+------------------+-----------------+
| apply (offset window)   | 275.29 MiB       | 184.79 MiB      |
+-------------------------+------------------+-----------------+

Although peak memory has consistently increased, Numba has shown performance parity or improvement
over ``Cython``.

Future
------

Once ``Numba`` is a dependency in pandas, we believe it should be straightforward to implement the rest
of the ``rolling`` aggregations, as well as ``ewa`` and ``expanding`` by using ``njit`` functions.

Eventually, we aim to generalize data grouping APIs (e.g. ``rolling``, ``groupby``, ``resample``) and
the sharing of aggregation routines (``mean``, ``apply``, ``count``) through the use of ``jitclass``.
Currently this path is not fully explored or implemented due to performance reasons, but this issue
will be `actively developed by the Numba team <https://github.com/numba/numba/issues/4522#issuecomment-537872456>`_
The design document for this implementation can be found in :doc:`generalized_window`


Appendix
--------

Timings on master:

.. code-block:: ipython

   In [1]: %load_ext memory_profiler

   In [2]: pd.__version__
   Out[2]: '0.26.0.dev0+514.g24b1dd128'

   In [3]: n = 1_000_000

   In [4]: roll_fixed = pd.Series(range(n)).rolling(10)

   In [5]: roll_offset = pd.Series(range(n), index=pd.date_range('2019', freq='s', periods=n)).rolling('10s')

   In [6]: %timeit roll_fixed.mean()
   21 ms ± 1.03 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

   In [7]: %memit roll_fixed.mean()
   peak memory: 161.13 MiB, increment: -0.29 MiB

   In [8]: %timeit roll_offset.mean()
   28.2 ms ± 735 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

   In [9]: %memit roll_offset.mean()
   peak memory: 177.08 MiB, increment: -0.05 MiB

   In [10]: %timeit roll_fixed.apply(lambda x: np.sum(x) + 5, raw=True)
   3.29 s ± 117 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   In [11]: %memit roll_fixed.apply(lambda x: np.sum(x) + 5, raw=True)
   peak memory: 177.12 MiB, increment: 0.00 MiB

   In [12]: %timeit roll_offset.apply(lambda x: np.sum(x) + 5, raw=True)
   3.54 s ± 98.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   In [13]: %memit roll_offset.apply(lambda x: np.sum(x) + 5, raw=True)
   peak memory: 184.79 MiB, increment: 0.00 MiB

Timings on Numba branch:

.. code-block:: ipython

   In [1]: %load_ext memory_profiler

   In [2]: pd.__version__
   Out[2]: '0.26.0.dev0+762.ge1f569381'

   In [3]: n = 1_000_000

   In [4]: roll_fixed = pd.Series(range(n)).rolling(10)

   In [5]: roll_offset = pd.Series(range(n), index=pd.date_range('2019', freq='s', periods=n)).rolling('10s')

   In [6]: %timeit roll_fixed.mean()
   22.9 ms ± 2.31 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   In [7]: %memit roll_fixed.mean()
   peak memory: 232.32 MiB, increment: -0.00 MiB

   In [8]: %timeit roll_offset.mean()
   38.3 ms ± 1.52 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   In [9]: %memit roll_offset.mean()
   peak memory: 243.85 MiB, increment: -0.05 MiB

   In [10]: %timeit roll_fixed.apply(lambda x: np.sum(x) + 5, raw=True)
   579 ms ± 9.23 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   In [11]: %memit roll_fixed.apply(lambda x: np.sum(x) + 5, raw=True)
   peak memory: 264.93 MiB, increment: 2.12 MiB

   In [12]: %timeit roll_offset.apply(lambda x: np.sum(x) + 5, raw=True)
   574 ms ± 5.11 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   In [13]: %memit roll_offset.apply(lambda x: np.sum(x) + 5, raw=True)
   peak memory: 275.29 MiB, increment: 1.61 MiB