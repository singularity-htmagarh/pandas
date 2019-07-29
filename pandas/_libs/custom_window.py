import abc
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from pandas.tseries.offsets import DateOffset

BeginEnd = Tuple[np.ndarray, np.ndarray]

# TODO: Refactor MockFixedWindowIndexer, FixedWindowIndexer,
#  VariableWindowIndexer to also have `get_window_bounds` methods that
#  only calculates start & stop

# TODO: Currently, when win_type is specified, it calls a special routine,
#  `roll_window`, while None win_type ops dispatch to specific methods.
#  Consider consolidating?


class BaseIndexer(abc.ABC):
    """Base class to calculate custom rolling window bounds"""
    def __init__(self, index, offset, keys) -> None:
        # TODO: The alternative is for the `rolling` API to accept
        #  index, offset, and keys as keyword arguments
        self.index = index
        self.offset = offset  # type: Union[str, DateOffset]
        self.keys = keys  # type: Sequence[np.ndarray]

    @classmethod
    @abc.abstractmethod
    def get_window_bounds(
        cls,
        win_type: Optional[str] = None,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
    ) -> BeginEnd:
        """
        Compute the bounds of a window.

        Users should subclass this class to implement a custom method
        to calculate window bounds

        Parameters
        ----------
        win_type : str, default None
            win_type passed from the top level rolling API

        min_periods : int, default None
            min_periods passed from the top level rolling API

        center : bool, default None
            center passed from the top level rolling API

        closed : str, default None
            closed passed from the top level rolling API

        Returns
        -------
        BeginEnd
            A tuple of ndarray[int64]s, indicating the boundaries of each
            window

        """

