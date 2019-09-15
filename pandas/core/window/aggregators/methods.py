import numpy as np


def rolling_mean(
    values: np.ndarray, begin: np.ndarray, end: np.ndarray, minimum_periods: int
) -> np.ndarray:
    result = np.empty(len(begin))
    previous_start = -1
    previous_end = -1
    count = 0
    total = 0
    for i, (start, stop) in enumerate(zip(begin, end)):
        if previous_start == -1 and previous_end == -1:
            # First aggregation over the values
            for value in values[start:stop]:
                if not np.isnan(value):
                    count += 1
                    total += value
        else:
            # Subsequent aggregations are calculated based on prior values
            for value in values[previous_start:start]:
                if not np.isnan(value):
                    count -= 1
                    total -= value
            for value in values[previous_end:stop]:
                if not np.isnan(value):
                    count += 1
                    total += value
        previous_start = start
        previous_end = stop
        val = np.nan
        if count and count >= minimum_periods:
            val = total / count
        result[i] = val
    return result
