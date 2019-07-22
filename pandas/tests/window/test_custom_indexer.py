from pandas import Series


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
