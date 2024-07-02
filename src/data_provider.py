import datetime as dt
import time
from abc import ABC, abstractmethod
from typing import Optional, cast

import numpy as np
import pandas as pd
from pandas import DatetimeIndex

from .util import generate_synthetic_prices_naive_step

KILOBYTE = 1024
MEGABYTE = KILOBYTE * 1024
GIGABYTE = MEGABYTE * 1024


class DataProvider(ABC):
    def __init__(self) -> None:
        self.load_data()
        t0: float = time.perf_counter()
        self._candles_1m: pd.DataFrame = DataProvider.create_candles_from_ticks(
            self._data, "min"
        )
        self._candles_5m: pd.DataFrame = DataProvider.create_candles_from_ticks(
            self._data, "5min"
        )
        self._candles_1h: pd.DataFrame = DataProvider.create_candles_from_ticks(
            self._data, "h"
        )
        self._candles_1d: pd.DataFrame = DataProvider.create_candles_from_ticks(
            self._data, "d"
        )
        t1: float = time.perf_counter()
        print(
            f"Creating {self.n_candles_total_until_timestamp()} candles took {t1-t0:.2f} seconds ({self.n_candles_total_until_timestamp()/(t1-t0):.2f} cps)"
        )

    def __len__(self) -> int:
        return len(self._data)

    def n_candles_total_until_timestamp(self, ts: Optional[dt.datetime] = None) -> int:
        return (
            len(
                self._candles_1m.loc[: ts or self.max_ts + dt.timedelta(milliseconds=1)]
            )
            + len(
                self._candles_5m.loc[: ts or self.max_ts + dt.timedelta(milliseconds=1)]
            )
            + len(
                self._candles_1h.loc[: ts or self.max_ts + dt.timedelta(milliseconds=1)]
            )
            + len(
                self._candles_1d.loc[: ts or self.max_ts + dt.timedelta(milliseconds=1)]
            )
        )

    @abstractmethod
    def load_data(self) -> None: ...

    @property
    def max_ts(self) -> dt.datetime:
        return self._data.index[-1]

    @property
    def min_ts(self) -> dt.datetime:
        return self._data.index[0]

    def get_bid_ask_price_at_timestamp(
        self, timestamp: dt.datetime
    ) -> Optional[tuple[float, float]]:
        if timestamp < self.min_ts:
            raise ValueError(f"Timestamp {timestamp}>{self.min_ts=}")
        else:
            row = self._data.loc[: timestamp + dt.timedelta(milliseconds=1)].iloc[-1]
            return float(row["bid"]), float(row["ask"])

    def get_ticks_range(self, t0: dt.datetime, t1: dt.datetime) -> pd.DataFrame:
        return self._data.loc[t0:t1].copy()

    def get_candles_range(
        self,
        t0: Optional[dt.datetime] = None,
        t1: Optional[dt.datetime] = None,
        freq: str = "1m",
    ) -> pd.DataFrame:
        if t0 is None:
            t0 = self.min_ts
        if t1 is None:
            t1 = self.max_ts

        if freq == "1m":
            return self._candles_1m.loc[t0:t1].copy()
        elif freq == "5m":
            return self._candles_5m.loc[t0:t1].copy()
        elif freq == "1h":
            return self._candles_1h.loc[t0:t1].copy()
        elif freq == "1d":
            return self._candles_1d.loc[t0:t1].copy()
        else:
            raise ValueError("Unsupported frequency")

    def get_last_n_candles(self, ts: dt.datetime, n: int, freq: str) -> pd.DataFrame:
        if freq == "1m":
            return (
                self._candles_1m.loc[: ts or self.max_ts + dt.timedelta(milliseconds=1)]
                .iloc[-n:]
                .copy()
            )
        elif freq == "5m":
            return (
                self._candles_5m.loc[: ts or self.max_ts + dt.timedelta(milliseconds=1)]
                .iloc[-n:]
                .copy()
            )
        elif freq == "1h":
            return (
                self._candles_1h.loc[: ts or self.max_ts + dt.timedelta(milliseconds=1)]
                .iloc[-n:]
                .copy()
            )
        elif freq == "1d":
            return (
                self._candles_1d.loc[: ts or self.max_ts + dt.timedelta(milliseconds=1)]
                .iloc[-n:]
                .copy()
            )
        else:
            raise ValueError("Unsupported frequency")

    @staticmethod
    def create_candles_from_ticks(df, freq: str) -> pd.DataFrame:
        df["mid"] = (df["ask"] + df["bid"]) / 2.0
        return df.resample(freq).agg("mid").ohlc()


class FileDataProvider(DataProvider):
    def __init__(self, filepath: str, *args, **kwargs) -> None:
        self._filepath = filepath
        super().__init__(*args, **kwargs)

    def load_data(self) -> None:
        if not self._filepath.endswith(".pkl"):
            raise NotImplementedError("Currently only .pkl files are supported")
        self._filepath: str = self._filepath
        self._data: pd.DataFrame = cast(pd.DataFrame, pd.read_pickle(self._filepath))
        assert isinstance(self._data, pd.DataFrame)
        assert self._data.index.is_monotonic_increasing


class SyntheticDataProvider_NaiveStepRange(DataProvider):
    def __init__(
        self, t0: dt.datetime, t1: dt.datetime, n_ticks: int, seed=0x2024_07_02
    ):
        self._t0 = t0
        self._t1 = t1
        self._seed = seed
        self._n_ticks = n_ticks
        super().__init__()

    def load_data(self) -> None:
        # Calculate the time intervals
        total_duration = (self._t1 - self._t0).total_seconds()
        time_intervals = np.linspace(0, total_duration, self._n_ticks)

        # Generate the timestamps
        timestamps = [
            self._t0 + dt.timedelta(seconds=interval) for interval in time_intervals
        ]

        # Generate synthetic prices
        prices = generate_synthetic_prices_naive_step(
            self._n_ticks, 1.31, seed=self._seed
        )
        bid = prices - 0.001
        ask = prices + 0.001

        times_dtindex: DatetimeIndex = pd.DatetimeIndex(timestamps)

        ticks_data = pd.DataFrame(
            {
                "t": times_dtindex,
                "bid": bid.round(5).values.reshape(-1),
                "ask": ask.round(5).values.reshape(-1),
            }
        )
        ticks_data.set_index("t", inplace=True)
        self._data = ticks_data


class SyntheticDataProvider_NaiveStep(DataProvider):
    def __init__(self, seed: int, n_ticks: int, *args, **kwargs) -> None:
        self._seed: int = seed
        self._n_ticks: int = n_ticks
        super().__init__(*args, **kwargs)

    def load_data(self) -> None:
        # TODO: Make the timeseries generation more flexible
        _rng = np.random.default_rng(self._seed)

        t0: float = int(
            dt.datetime(2023, 10, 12, 9, 30, tzinfo=dt.timezone.utc).timestamp() * 1000
        )
        t_deltas = _rng.uniform(300, 1200, self._n_ticks).astype(np.int64)
        times = t0 + np.cumsum(t_deltas)

        prices = generate_synthetic_prices_naive_step(self._n_ticks, 1.31)
        bid = prices - 0.05
        ask = prices + 0.05

        times_dtindex: DatetimeIndex = cast(
            pd.DatetimeIndex, pd.to_datetime(times, unit="ms", utc=True)
        )

        ticks_data = pd.DataFrame(
            {
                "t": times_dtindex,
                "bid": bid.values.reshape(-1),
                "ask": ask.values.reshape(-1),
            }
        )
        ticks_data.set_index("t", inplace=True)
        self._data = ticks_data
