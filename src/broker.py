import datetime as dt
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from .data_provider import DataProvider


class BrokerManager(ABC):
    def __init__(self, data_provider: DataProvider):
        self._data_provider = data_provider

    @abstractmethod
    def send_market_order(self, symbol: str, volume: float, is_buy: float) -> int: ...

    @abstractmethod
    def send_limit_order(
        self, symbol: str, volume: float, is_buy: float, limit_price: float
    ) -> int: ...

    @abstractmethod
    def get_positions(self, symbol: str) -> pd.DataFrame: ...

    @abstractmethod
    def get_orders(self, symbol: str) -> pd.DataFrame: ...

    @abstractmethod
    def close_position(self, id_position: int) -> None: ...

    @abstractmethod
    def cancel_order(self, id_order: int) -> None: ...


class BacktestingBroker(BrokerManager):
    def __init__(
        self,
        starting_time: dt.datetime,
        stop_time: dt.datetime,
        data_provider: DataProvider,
        second_skip_per_iteration: float = 1.0,
        min_seconds_per_iteration: float = 0.1,
    ):
        super().__init__(data_provider)
        self._orders = pd.DataFrame(
            columns=["id", "symbol", "volume", "is_buy", "limit_price", "status"]
        )
        self._positions = pd.DataFrame(
            columns=["id", "symbol", "volume", "is_buy", "entry_price"]
        )
        self.time: dt.datetime = starting_time
        self.stop_time: dt.datetime = stop_time

        self.second_skip_per_iteration: float = second_skip_per_iteration
        self.min_seconds_per_iteration: float = min_seconds_per_iteration

        self.events: list[dict[str, any]] = []

    def get_positions(self, symbol: str) -> pd.DataFrame:
        return self._positions[self._positions["symbol"] == symbol]

    def get_orders(self, symbol: str) -> pd.DataFrame:
        return self._orders[self._orders["symbol"] == symbol]

    def send_market_order(self, symbol: str, volume: float, is_buy: float) -> int:
        order_id: str = "o_" + uuid.uuid4().hex[:16]
        self.events.append(
            {
                "type": "market_order",
                "symbol": symbol,
                "volume": volume,
                "is_buy": is_buy,
                "order_id": order_id,
            }
        )
        return order_id

    def send_limit_order(
        self, symbol: str, volume: float, is_buy: float, limit_price: float
    ) -> int: ...

    def cancel_order(self, id_order: int) -> None: ...

    def close_position(self, id_position: int) -> None: ...

    def _handle_events(self) -> None:
        for event in self.events:
            if event["type"] == "market_order":
                self._handle_market_order(event)
            elif event["type"] == "limit_order":
                self._handle_limit_order(event)
            elif event["type"] == "cancel_order":
                self._handle_cancel_order(event)
            elif event["type"] == "close_position":
                self._handle_close_position(event)

    def get_time(self) -> dt.datetime:
        return self.time

    def start_running(self) -> None:
        self.is_running = True
        while self.is_running:
            t0 = time.time()
            self._handle_events()
            self.time += dt.timedelta(seconds=self.second_skip_per_iteration)

            t1 = time.time()
            t_rem = self.second_skip_per_iteration - (t1 - t0)
            if t_rem > 0:
                time.sleep(t_rem)

            if self.time > self.stop_time:
                self.is_running = False
        print(f"Reached stop time of {self.stop_time}... Stopping broker.")

    def get_current_bid_ask(self) -> Optional[tuple[float, float]]:
        return self._data_provider.get_bid_ask_price_at_timestamp(self.time)

    def get_total_number_of_candles(self) -> int:
        return self._data_provider.n_candles_total_until_timestamp(self.time)

    def get_last_n_candles(self, n: int, freq: str = "1m") -> pd.DataFrame:
        return self._data_provider.get_last_n_candles(self.time, n, freq=freq)
