Rolling Operations with Numba
=============================

Summary
-------

This proof of concept (POC) demonstrates using `Numba <http://numba.pydata.org/>`_ instead of ``Cython``
in ``rolling`` applications. Namely, ``Numba`` was used to:

#. Compute ``rolling.mean``
#. Compute ``rolling.apply``
#. Calculate offset-based or numeric-based window boundaries.

The benefits of using ``Numba`` include:

#. Performance parity or improvement over ``Cython``
#. Eliminate shipping C-extensions
#. Ease of debugging

Additionally this POC includes an API that allows users to create custom window boundary calculations.

Performance
-----------

Below is table comparing the current performance difference between the Numba and Cython implementations
for 1 million data points. (Exact benchmark setup can be found in the Appendix)

+-------------------------+------------------+-----------------+
| Speed                   | Numba            | Cython          |
+=========================+==================+=================+
| mean (fixed window)     | 37.2 ms ± 1.66 ms| 21 ms ± 1.03 ms |
+-------------------------+------------------+-----------------+
| mean (offset window)    | 102 ms ± 2.73 ms | 28.2 ms ± 735 µs|
+-------------------------+------------------+-----------------+
| apply (fixed window)    | 577 ms ± 8.95 ms | 3.29 s ± 117 ms |
+-------------------------+------------------+-----------------+
| apply (offset window)   | 640 ms ± 5.51 ms | 3.54 s ± 98.6 ms|
+-------------------------+------------------+-----------------+

+-------------------------+------------------+-----------------+
| Peak Memory             | Numba            | Cython          |
+=========================+==================+=================+
| mean (fixed window)     | 240.22 MiB       | 161.13 MiB      |
+-------------------------+------------------+-----------------+
| mean (offset window)    | 245.81 MiB       | 177.08 MiB      |
+-------------------------+------------------+-----------------+
| apply (fixed window)    | 268.28 MiB       | 177.12 MiB      |
+-------------------------+------------------+-----------------+
| apply (offset window)   | 277.91 MiB       | 184.79 MiB      |
+-------------------------+------------------+-----------------+

Although peak memory has consistently increased and there is a slight performance decrease for rolling mean,
rolling apply benchmarks have greatly increased.

Window Indexer API
------------------

A ``BaseIndexer`` class is available from ``pandas/core/window/indexers.py`` for users to subclass
to create a custom routine to calculate window boundaries. Users will need to specify a
``get_window_bounds`` function to calculate window boundaries. Below is an example of creating an
``ExpandingIndexer`` computes an expanding window:

.. code-block:: python

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
       ) -> BeginEnd:
           """
           Examples
           --------
           >>> ExpandingIndexer().get_window_bounds(10)

           (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))
           """
           return np.zeros(num_values, dtype=np.int64), np.arange(1, num_values + 1)

   s = Series(range(10))
   s.rolling(ExpandingIndexer()).mean()


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
   Out[2]: '0.26.0.dev0+728.gfd7e4841e'

   In [3]: n = 1_000_000

   In [4]: roll_fixed = pd.Series(range(n)).rolling(10)

   In [5]: roll_offset = pd.Series(range(n), index=pd.date_range('2019', freq='s', periods=n)).rolling('10s')

   In [6]: %timeit roll_fixed.mean()
   37.2 ms ± 1.66 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   In [7]: %memit roll_fixed.mean()
   peak memory: 240.22 MiB, increment: 0.01 MiB

   In [8]: %timeit roll_offset.mean()
   102 ms ± 2.73 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   In [9]: %memit roll_offset.mean()
   peak memory: 245.81 MiB, increment: -0.05 MiB

   In [10]: %timeit roll_fixed.apply(lambda x: np.sum(x) + 5, raw=True)
   577 ms ± 8.95 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   In [11]: %memit roll_fixed.apply(lambda x: np.sum(x) + 5, raw=True)
   peak memory: 268.28 MiB, increment: 3.05 MiB

   In [12]: %timeit roll_offset.apply(lambda x: np.sum(x) + 5, raw=True)
   640 ms ± 5.51 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   In [13]: %memit roll_offset.apply(lambda x: np.sum(x) + 5, raw=True)
   peak memory: 277.91 MiB, increment: 1.57 MiB