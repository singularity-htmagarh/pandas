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

