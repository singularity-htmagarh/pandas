from __future__ import annotations

import datetime
from functools import partial
from textwrap import dedent
from typing import (
    Optional,
    Union,
)
import warnings

import numpy as np

from pandas._libs.tslibs import Timedelta
import pandas._libs.window.aggregations as window_aggregations
from pandas._typing import (
    Axis,
    FrameOrSeries,
    FrameOrSeriesUnion,
    TimedeltaConvertibleTypes,
)
from pandas.compat.numpy import function as nv
from pandas.util._decorators import doc

from pandas.core.dtypes.common import is_datetime64_ns_dtype
from pandas.core.dtypes.missing import isna

import pandas.core.common as common
from pandas.core.util.numba_ import maybe_use_numba
from pandas.core.window.common import zsqrt
from pandas.core.window.doc import (
    _shared_docs,
    args_compat,
    create_section_header,
    kwargs_compat,
    template_header,
    template_returns,
    template_see_also,
)
from pandas.core.window.indexers import (
    BaseIndexer,
    ExponentialMovingWindowIndexer,
    GroupbyIndexer,
)
from pandas.core.window.numba_ import generate_numba_groupby_ewma_func
from pandas.core.window.rolling import (
    BaseWindow,
    BaseWindowGroupby,
)
from pandas.core.window.aggregators import EWMean


def get_center_of_mass(
    comass: Optional[float],
    span: Optional[float],
    halflife: Optional[float],
    alpha: Optional[float],
) -> float:
    valid_count = common.count_not_none(comass, span, halflife, alpha)
    if valid_count > 1:
        raise ValueError("comass, span, halflife, and alpha are mutually exclusive")

    # Convert to center of mass; domain checks ensure 0 < alpha <= 1
    if comass is not None:
        if comass < 0:
            raise ValueError("comass must satisfy: comass >= 0")
    elif span is not None:
        if span < 1:
            raise ValueError("span must satisfy: span >= 1")
        comass = (span - 1) / 2
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError("halflife must satisfy: halflife > 0")
        decay = 1 - np.exp(np.log(0.5) / halflife)
        comass = 1 / decay - 1
    elif alpha is not None:
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must satisfy: 0 < alpha <= 1")
        comass = (1 - alpha) / alpha
    else:
        raise ValueError("Must pass one of comass, span, halflife, or alpha")

    return float(comass)


class ExponentialMovingWindow(BaseWindow):
    r"""
    Provide exponential weighted (EW) functions.

    Available EW functions: ``mean()``, ``var()``, ``std()``, ``corr()``, ``cov()``.

    Exactly one parameter: ``com``, ``span``, ``halflife``, or ``alpha`` must be
    provided.

    Parameters
    ----------
    com : float, optional
        Specify decay in terms of center of mass,
        :math:`\alpha = 1 / (1 + com)`, for :math:`com \geq 0`.
    span : float, optional
        Specify decay in terms of span,
        :math:`\alpha = 2 / (span + 1)`, for :math:`span \geq 1`.
    halflife : float, str, timedelta, optional
        Specify decay in terms of half-life,
        :math:`\alpha = 1 - \exp\left(-\ln(2) / halflife\right)`, for
        :math:`halflife > 0`.

        If ``times`` is specified, the time unit (str or timedelta) over which an
        observation decays to half its value. Only applicable to ``mean()``
        and halflife value will not apply to the other functions.

        .. versionadded:: 1.1.0

    alpha : float, optional
        Specify smoothing factor :math:`\alpha` directly,
        :math:`0 < \alpha \leq 1`.
    min_periods : int, default 0
        Minimum number of observations in window required to have a value
        (otherwise result is NA).
    adjust : bool, default True
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings (viewing EWMA as a moving average).

        - When ``adjust=True`` (default), the EW function is calculated using weights
          :math:`w_i = (1 - \alpha)^i`. For example, the EW moving average of the series
          [:math:`x_0, x_1, ..., x_t`] would be:

        .. math::
            y_t = \frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ... + (1 -
            \alpha)^t x_0}{1 + (1 - \alpha) + (1 - \alpha)^2 + ... + (1 - \alpha)^t}

        - When ``adjust=False``, the exponentially weighted function is calculated
          recursively:

        .. math::
            \begin{split}
                y_0 &= x_0\\
                y_t &= (1 - \alpha) y_{t-1} + \alpha x_t,
            \end{split}
    ignore_na : bool, default False
        Ignore missing values when calculating weights; specify ``True`` to reproduce
        pre-0.15.0 behavior.

        - When ``ignore_na=False`` (default), weights are based on absolute positions.
          For example, the weights of :math:`x_0` and :math:`x_2` used in calculating
          the final weighted average of [:math:`x_0`, None, :math:`x_2`] are
          :math:`(1-\alpha)^2` and :math:`1` if ``adjust=True``, and
          :math:`(1-\alpha)^2` and :math:`\alpha` if ``adjust=False``.

        - When ``ignore_na=True`` (reproducing pre-0.15.0 behavior), weights are based
          on relative positions. For example, the weights of :math:`x_0` and :math:`x_2`
          used in calculating the final weighted average of
          [:math:`x_0`, None, :math:`x_2`] are :math:`1-\alpha` and :math:`1` if
          ``adjust=True``, and :math:`1-\alpha` and :math:`\alpha` if ``adjust=False``.
    axis : {0, 1}, default 0
        The axis to use. The value 0 identifies the rows, and 1
        identifies the columns.
    times : str, np.ndarray, Series, default None

        .. versionadded:: 1.1.0

        Times corresponding to the observations. Must be monotonically increasing and
        ``datetime64[ns]`` dtype.

        If str, the name of the column in the DataFrame representing the times.

        If 1-D array like, a sequence with the same shape as the observations.

        Only applicable to ``mean()``.

    Returns
    -------
    DataFrame
        A Window sub-classed for the particular operation.

    See Also
    --------
    rolling : Provides rolling window calculations.
    expanding : Provides expanding transformations.

    Notes
    -----

    More details can be found at:
    :ref:`Exponentially weighted windows <window.exponentially_weighted>`.

    Examples
    --------
    >>> df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
    >>> df
         B
    0  0.0
    1  1.0
    2  2.0
    3  NaN
    4  4.0

    >>> df.ewm(com=0.5).mean()
              B
    0  0.000000
    1  0.750000
    2  1.615385
    3  1.615385
    4  3.670213

    Specifying ``times`` with a timedelta ``halflife`` when computing mean.

    >>> times = ['2020-01-01', '2020-01-03', '2020-01-10', '2020-01-15', '2020-01-17']
    >>> df.ewm(halflife='4 days', times=pd.DatetimeIndex(times)).mean()
              B
    0  0.000000
    1  0.585786
    2  1.523889
    3  1.523889
    4  3.233686
    """

    _attributes = [
        "com",
        "span",
        "halflife",
        "alpha",
        "min_periods",
        "adjust",
        "ignore_na",
        "axis",
        "times",
    ]

    def __init__(
        self,
        obj: FrameOrSeries,
        com: Optional[float] = None,
        span: Optional[float] = None,
        halflife: Optional[Union[float, TimedeltaConvertibleTypes]] = None,
        alpha: Optional[float] = None,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        axis: Axis = 0,
        times: Optional[Union[str, np.ndarray, FrameOrSeries]] = None,
    ):
        super().__init__(
            obj=obj,
            min_periods=max(int(min_periods), 1),
            on=None,
            center=False,
            closed=None,
            method="single",
            axis=axis,
        )
        self.com = com
        self.span = span
        self.halflife = halflife
        self.alpha = alpha
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.times = times
        if self.times is not None:
            if not self.adjust:
                raise NotImplementedError("times is not supported with adjust=False.")
            if isinstance(self.times, str):
                self.times = self._selected_obj[self.times]
            if not is_datetime64_ns_dtype(self.times):
                raise ValueError("times must be datetime64[ns] dtype.")
            # error: Argument 1 to "len" has incompatible type "Union[str, ndarray,
            # FrameOrSeries, None]"; expected "Sized"
            if len(self.times) != len(obj):  # type: ignore[arg-type]
                raise ValueError("times must be the same length as the object.")
            if not isinstance(self.halflife, (str, datetime.timedelta)):
                raise ValueError(
                    "halflife must be a string or datetime.timedelta object"
                )
            if isna(self.times).any():
                raise ValueError("Cannot convert NaT values to integer")
            # error: Item "str" of "Union[str, ndarray, FrameOrSeries, None]" has no
            # attribute "view"
            # error: Item "None" of "Union[str, ndarray, FrameOrSeries, None]" has no
            # attribute "view"
            _times = np.asarray(
                self.times.view(np.int64), dtype=np.float64  # type: ignore[union-attr]
            )
            _halflife = float(Timedelta(self.halflife).value)
            self._deltas = np.diff(_times) / _halflife
            # Halflife is no longer applicable when calculating COM
            # But allow COM to still be calculated if the user passes other decay args
            if common.count_not_none(self.com, self.span, self.alpha) > 0:
                self._com = get_center_of_mass(self.com, self.span, None, self.alpha)
            else:
                self._com = 1.0
        else:
            if self.halflife is not None and isinstance(
                self.halflife, (str, datetime.timedelta)
            ):
                raise ValueError(
                    "halflife can only be a timedelta convertible argument if "
                    "times is not None."
                )
            # Without times, points are equally spaced
            self._deltas = np.ones(max(len(self.obj) - 1, 0), dtype=np.float64)
            self._com = get_center_of_mass(
                # error: Argument 3 to "get_center_of_mass" has incompatible type
                # "Union[float, Any, None, timedelta64, signedinteger[_64Bit]]";
                # expected "Optional[float]"
                self.com,
                self.span,
                self.halflife,  # type: ignore[arg-type]
                self.alpha,
            )
        self._mean = EWMean(self._com, self.adjust, self.ignore_na)
        self._mean_result = type(self._selected_obj)()

    def _get_window_indexer(self) -> BaseIndexer:
        """
        Return an indexer class that will compute the window start and end bounds
        """
        return ExponentialMovingWindowIndexer()

    @doc(
        _shared_docs["aggregate"],
        see_also=dedent(
            """
        See Also
        --------
        pandas.DataFrame.rolling.aggregate
        """
        ),
        examples=dedent(
            """
        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.ewm(alpha=0.5).mean()
                  A         B         C
        0  1.000000  4.000000  7.000000
        1  1.666667  4.666667  7.666667
        2  2.428571  5.428571  8.428571
        """
        ),
        klass="Series/Dataframe",
        axis="",
    )
    def aggregate(self, func, *args, **kwargs):
        return super().aggregate(func, *args, **kwargs)

    agg = aggregate

    @doc(
        template_header,
        create_section_header("Parameters"),
        args_compat,
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="ewm",
        aggregation_description="(exponential weighted moment) mean",
        agg_method="mean",
    )
    def mean(self, *args, update=None, **kwargs):
        nv.validate_window_func("mean", args, kwargs)
        if update is not None and self._mean_result.empty:
            raise ValueError("Must call mean() first before passing `update`")
        if update is not None:
            obj = update
            new_result = [self._mean_result]
        elif self._mean_result.empty:
            obj = self.obj
            new_result = []
        else:
            return self._mean_result
        for i in range(len(obj)):
            self._mean.step(obj.iloc[i])
            new_result.append(self._mean.finalize())
            from pandas.core.reshape.concat import concat
            self._mean_result = concat(new_result, ignore_index=True)
        return self._mean_result
        # window_func = window_aggregations.ewma
        # window_func = partial(
        #     window_func,
        #     com=self._com,
        #     adjust=self.adjust,
        #     ignore_na=self.ignore_na,
        #     deltas=self._deltas,
        # )
        # return self._apply(window_func)

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        bias : bool, default False
            Use a standard estimation bias correction.
        """
        ).replace("\n", "", 1),
        args_compat,
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="ewm",
        aggregation_description="(exponential weighted moment) standard deviation",
        agg_method="std",
    )
    def std(self, bias: bool = False, *args, **kwargs):
        nv.validate_window_func("std", args, kwargs)
        return zsqrt(self.var(bias=bias, **kwargs))

    def vol(self, bias: bool = False, *args, **kwargs):
        warnings.warn(
            (
                "vol is deprecated will be removed in a future version. "
                "Use std instead."
            ),
            FutureWarning,
            stacklevel=2,
        )
        return self.std(bias, *args, **kwargs)

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        bias : bool, default False
            Use a standard estimation bias correction.
        """
        ).replace("\n", "", 1),
        args_compat,
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="ewm",
        aggregation_description="(exponential weighted moment) variance",
        agg_method="var",
    )
    def var(self, bias: bool = False, *args, **kwargs):
        nv.validate_window_func("var", args, kwargs)
        window_func = window_aggregations.ewmcov
        window_func = partial(
            window_func,
            com=self._com,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            bias=bias,
        )

        def var_func(values, begin, end, min_periods):
            return window_func(values, begin, end, min_periods, values)

        return self._apply(var_func)

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        other : Series or DataFrame , optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndex DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        bias : bool, default False
            Use a standard estimation bias correction.
        """
        ).replace("\n", "", 1),
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="ewm",
        aggregation_description="(exponential weighted moment) sample covariance",
        agg_method="cov",
    )
    def cov(
        self,
        other: Optional[FrameOrSeriesUnion] = None,
        pairwise: Optional[bool] = None,
        bias: bool = False,
        **kwargs,
    ):
        from pandas import Series

        def cov_func(x, y):
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = (
                self.min_periods
                if self.min_periods is not None
                else window_indexer.window_size
            )
            start, end = window_indexer.get_window_bounds(
                num_values=len(x_array),
                min_periods=min_periods,
                center=self.center,
                closed=self.closed,
            )
            result = window_aggregations.ewmcov(
                x_array,
                start,
                end,
                self.min_periods,
                y_array,
                self._com,
                self.adjust,
                self.ignore_na,
                bias,
            )
            return Series(result, index=x.index, name=x.name)

        return self._apply_pairwise(self._selected_obj, other, pairwise, cov_func)

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndex DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        """
        ).replace("\n", "", 1),
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="ewm",
        aggregation_description="(exponential weighted moment) sample correlation",
        agg_method="corr",
    )
    def corr(
        self,
        other: Optional[FrameOrSeriesUnion] = None,
        pairwise: Optional[bool] = None,
        **kwargs,
    ):
        from pandas import Series

        def cov_func(x, y):
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = (
                self.min_periods
                if self.min_periods is not None
                else window_indexer.window_size
            )
            start, end = window_indexer.get_window_bounds(
                num_values=len(x_array),
                min_periods=min_periods,
                center=self.center,
                closed=self.closed,
            )

            def _cov(X, Y):
                return window_aggregations.ewmcov(
                    X,
                    start,
                    end,
                    self.min_periods,
                    Y,
                    self._com,
                    self.adjust,
                    self.ignore_na,
                    1,
                )

            with np.errstate(all="ignore"):
                cov = _cov(x_array, y_array)
                x_var = _cov(x_array, x_array)
                y_var = _cov(y_array, y_array)
                result = cov / zsqrt(x_var * y_var)
            return Series(result, index=x.index, name=x.name)

        return self._apply_pairwise(self._selected_obj, other, pairwise, cov_func)


class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    """
    Provide an exponential moving window groupby implementation.
    """

    _attributes = ExponentialMovingWindow._attributes + BaseWindowGroupby._attributes

    def _get_window_indexer(self) -> GroupbyIndexer:
        """
        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
        """
        window_indexer = GroupbyIndexer(
            groupby_indicies=self._grouper.indices,
            window_indexer=ExponentialMovingWindowIndexer,
        )
        return window_indexer

    def mean(self, engine=None, engine_kwargs=None):
        """
        Parameters
        ----------
        engine : str, default None
            * ``'cython'`` : Runs mean through C-extensions from cython.
            * ``'numba'`` : Runs mean through JIT compiled code from numba.
              Only available when ``raw`` is set to ``True``.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

              .. versionadded:: 1.2.0

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``.

              .. versionadded:: 1.2.0

        Returns
        -------
        Series or DataFrame
            Return type is determined by the caller.
        """
        if maybe_use_numba(engine):
            groupby_ewma_func = generate_numba_groupby_ewma_func(
                engine_kwargs,
                self._com,
                self.adjust,
                self.ignore_na,
            )
            return self._apply(
                groupby_ewma_func,
                numba_cache_key=(lambda x: x, "groupby_ewma"),
            )
        elif engine in ("cython", None):
            if engine_kwargs is not None:
                raise ValueError("cython engine does not accept engine_kwargs")
            return super().mean()
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")
