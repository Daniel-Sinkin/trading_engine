import datetime as dt
from datetime import datetime
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PERCENTAGE = 1e-2


def plot_candles(candles_1d, ticks_ts, mid_prices):
    for t, row in candles_1d.iterrows():
        t = cast(dt.datetime, t)
        t_right = t + dt.timedelta(days=1)
        t_mid = t + (t_right - t) / 2
        is_buy = row["close"] > row["open"]

        if is_buy:
            xy = (t, row["open"])
            color = "cyan"
            height = row["close"] - row["open"]
        else:
            xy = (t, row["close"])
            color = "pink"
            height = row["open"] - row["close"]
        width = t_right - t

        plt.gca().add_patch(
            plt.Rectangle(
                xy=xy, width=width, height=height, color=color, alpha=0.7, zorder=5
            )
        )
        plt.plot(
            [t, t_right], [row["high"], row["high"]], color="black", alpha=0.4, zorder=5
        )
        plt.plot(
            [t, t_right], [row["low"], row["low"]], color="black", alpha=0.4, zorder=5
        )
        plt.plot(
            [t_mid, t_mid],
            [row["low"], row.open if is_buy else row.close],
            color="black",
            alpha=0.4,
            zorder=1,
        )
        plt.plot(
            [t_mid, t_mid],
            [row["high"], row.close if is_buy else row.open],
            color="black",
            alpha=0.4,
            zorder=1,
        )
    plt.plot(ticks_ts, mid_prices, alpha=0.3)


def generate_synthetic_prices_naive_step(
    n=2520,
    p0=1.31,
    step_size_min=-1e-3,
    step_size_max=1e-3,
    seed: int = 0x2024_07_02,
) -> pd.DataFrame:
    """
    Generate synthetic forex data using a naive step model.
    """
    rng = np.random.default_rng(seed)
    S = np.zeros(n)
    S[0] = p0

    for t in range(1, n):
        S[t] = S[t - 1] + rng.uniform(step_size_min, step_size_max)

    return pd.DataFrame({"Price": S})


def generate_synthetic_prices_and_vola_heston(
    n=2520,
    p0=1.31,
    average_return_rate=2e-4,
    long_term_variance=1e-4,
    reversion_speed=1e-1,
    variance_volatility=1e-2,
    v0=1e-4,
    seed: int = 0x2024_07_02,
) -> pd.DataFrame:
    """
    As per
    ```markdown
    Steven L. Heston, A Closed-Form Solution for Options with Stochastic Volatility with Applications
    to Bond and Currency Options, The Review of Financial Studies, Volume 6, Issue 2, April 1993, Pages 327-343,
    https://doi.org/10.1093/rfs/6.2.327
    ```
    """
    rng = np.random.default_rng(seed)
    dt = 1 / n
    S = np.zeros(n)
    v = np.zeros(n)
    S[0] = p0
    v[0] = v0

    for t in range(1, n):
        z1 = rng.normal()
        z2 = rng.normal()
        v[t] = (
            v[t - 1]
            + reversion_speed * (long_term_variance - v[t - 1]) * dt
            + variance_volatility * np.sqrt(v[t - 1]) * np.sqrt(dt) * z1
        )
        v[t] = max(v[t], 0)  # variance must be non-negative
        S[t] = S[t - 1] * np.exp(
            (average_return_rate - 0.5 * v[t - 1]) * dt + np.sqrt(v[t - 1] * dt) * z2
        )

    return pd.DataFrame({"Price": S, "Variance": v})


def main() -> None:
    synthetic_data: pd.DataFrame = generate_synthetic_prices_and_vola_heston(n=252 * 10)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(synthetic_data["Price"], label="Price")
    ax[0].set_title("Synthetic Forex Price")
    ax[0].legend()

    ax[1].plot(synthetic_data["Variance"])
    ax[1].set_title("Synthetic Forex Variance")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
