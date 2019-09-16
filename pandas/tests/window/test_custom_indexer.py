from pandas import Series
from pandas.core.window.indexers import ExpandingIndexer
import pandas.util.testing as tm


def test_custom_indexer_validates(
    dummy_custom_indexer, win_types, closed, min_periods, center
):
    # Test passing a BaseIndexer subclass does not raise validation errors
    s = Series(range(10))
    s.rolling(
        dummy_custom_indexer,
        win_type=win_types,
        center=center,
        min_periods=min_periods,
        closed=closed,
    )


def test_expanding_indexer():
    s = Series(range(10))
    result = s.rolling(ExpandingIndexer()).mean()
    expected = s.expanding().mean()
    tm.assert_series_equal(result, expected)
