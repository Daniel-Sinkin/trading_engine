import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import lfilter, lfiltic


def ewma(array, window: int):
    """From https://stackoverflow.com/a/59199643"""
    alpha = 2 / (window + 1)
    b = [alpha]
    a = [1, alpha - 1]
    zi = lfiltic(b, a, array[0:1], [0])
    return lfilter(b, a, array, zi=zi)[0]


def stochastic_oscillator_scipy(HLC: np.ndarray, period: int):
    assert HLC.ndim == 2
    assert HLC.shape[1] == 3

    highest_highs = maximum_filter1d(HLC[0], size=period, mode="nearest")
    lowest_lows = minimum_filter1d(HLC[1], size=period, mode="nearest")

    closes = HLC[2][period - 1 :]

    percent_k = (
        (closes - lowest_lows[period - 1 :])
        / (highest_highs[period - 1 :] - lowest_lows[period - 1 :])
    ) * 100

    return percent_k
