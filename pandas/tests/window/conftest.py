import pytest

import pandas._libs.custom_window as libwindow_custom

from pandas import date_range, offsets


@pytest.fixture(params=[True, False])
def raw(request):
    return request.param


@pytest.fixture(
    params=[
        "triang",
        "blackman",
        "hamming",
        "bartlett",
        "bohman",
        "blackmanharris",
        "nuttall",
        "barthann",
    ]
)
def win_types(request):
    return request.param


@pytest.fixture(params=["kaiser", "gaussian", "general_gaussian", "exponential"])
def win_types_special(request):
    return request.param


@pytest.fixture(
    params=["sum", "mean", "median", "max", "min", "var", "std", "kurt", "skew"]
)
def arithmetic_win_operators(request):
    return request.param


@pytest.fixture(params=["right", "left", "both", "neither"])
def closed(request):
    return request.param


@pytest.fixture(params=[True, False])
def center(request):
    return request.param


@pytest.fixture(params=[None, 1])
def min_periods(request):
    return request.param


@pytest.fixture
def dummy_custom_indexer():
    class DummyIndexer(libwindow_custom.BaseIndexer):
        def __init__(self, index, offset, keys):
            super().__init__(index, offset, keys)

        def get_window_bounds(self, **kwargs):
            pass

    idx = date_range("2019", freq="D", periods=3)
    offset = offsets.BusinessDay(1)
    keys = ["A"]
    return DummyIndexer(index=idx, offset=offset, keys=keys)
