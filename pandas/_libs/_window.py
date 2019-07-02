import abc
from typing import Dict, Sequence, Tuple

import numpy as np

BeginEnd = Tuple[np.ndarray, np.ndarray]

# TODO: Refactor MockFixedWindowIndexer, FixedWindowIndexer,
#  VariableWindowIndexer to also have `get_window_bounds` methods that
#  only calculates start & stop

# TODO: Currently, when win_type is specified, it calls a special routine,
#  `roll_window`, while None win_type ops dispatch to specific methods.
#  Consider consolidating?


class BaseIndexer(abc.ABC):

    def __init__(self, index, offset, keys):
        # TODO: The alternative is for the `rolling` API to accept
        #  index, offset, and keys as keyword arguments
        self.index = index
        self.offset = offset
        self.keys = keys  # type: Sequence[np.ndarray]

    @classmethod
    @abc.abstractmethod
    def get_window_bounds(cls, **kwargs: Dict) -> BeginEnd:
        """
        Compute the bounds of a window.

        Users should subclass this class to implement a custom method
        to calculate window bounds

        Parameters
        ----------
        kwargs
            A dictionary of keyword arguments obtained from the top level
            rolling API e.g. min_periods, win_type

        Returns
        -------
        BeginEnd
            A tuple of ndarray[int64]s, indicating the boundaries of each
            window

        """
        return NotImplemented
