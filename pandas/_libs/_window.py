import abc
from typing import Any, Dict, Sequence, Tuple, TypeVar

import numpy as np

BeginEnd = Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]
Displacement = TypeVar("Displacement")

# TODO: Refactor MockFixedWindowIndexer, FixedWindowIndexer,
#  VariableWindowIndexer to also have `get_window_bounds` methods that
#  only calculates start & stop


class BaseIndexer(abc.ABC):

    def __init__(self, index, offset, keys):
        # TODO: Confirm: index, offset and keys are set in the __init__?
        #  Which patters will users have?
        #  1)
        #  indexer = Indexer(index=index, offset=offset, ...)
        #  df.rolling(indexer).mean()
        #  i.e. Indexer holds its values to pass to `get_window_bounds`
        #  2)
        #  df.rolling(Indexer, index=index, offset=offset, **kwargs).mean()
        #  **kwargs passed down to Indexer when get_window_bounds is called
        self.index = index
        self.offset = offset
        self.keys = keys

    @abc.abstractmethod
    @classmethod
    def get_window_bounds(
        cls, index, offset, keys: Sequence[np.ndarray[Any]], **kwargs: Dict
    ) -> BeginEnd:
        """
        Compute the bounds of a window.

        Users should subclass this class to implement a custom method
        to calculate window bounds

        Parameters
        ----------
        index
            A pandas index to compute indices against
        offset
            An object that can be used to calculate the displacment for each
            element
        keys
            A possibly empty list of additional columns needed to compute
            window bounds
        kwargs
            A dictionary of keyword arguments obtained from the top level
            rolling API e.g. min_periods, win_type

        Returns
        -------
        BeginEnd
            A tuple of ndarray[int64]s, indicating the boundaries of each
            window

        """
        pass


# TODO: Currently, when win_type is specified, it calls a special routine,
#  `roll_window`, while None win_type ops dispatch to specific methods.
#  Consider consolidating?

