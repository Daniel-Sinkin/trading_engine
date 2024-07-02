from typing import cast

import numpy as np
from scipy.signal import lfilter, lfiltic


def ewma(array, window: int):
    """From https://stackoverflow.com/a/59199643"""
    alpha = 2 / (window + 1)
    b = [alpha]
    a = [1, alpha - 1]
    zi = lfiltic(b, a, array[0:1], [0])
    return cast(np.ndarray[int, np.float32], lfilter(b, a, array, zi=zi)[0])
