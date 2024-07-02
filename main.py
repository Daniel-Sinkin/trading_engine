import datetime as dt
import multiprocessing as mp
import time
from multiprocessing.managers import BaseManager

from src.broker import BacktestingBroker
from src.data_provider import SyntheticDataProvider_NaiveStepRange
from src.technical_analysis import ewma


def run_backtesting_broker(broker: BacktestingBroker):
    broker.start_running()


def print_broker_time(broker: BacktestingBroker):
    while True:
        candles = broker.get_last_n_candles(100, freq="1d")
        ema = ewma(candles["close"].values, 14)
        print(
            f"Broker time: {broker.get_time()}, current price {broker.get_current_bid_ask()}, total number of candles = {broker.get_total_number_of_candles()}, ema = {ema[-1]}"
        )
        time.sleep(0.1)


class MyManager(BaseManager):
    pass


if __name__ == "__main__":
    starting_time = dt.datetime(2023, 10, 12, 9, 30, tzinfo=dt.timezone.utc)
    stopping_time = starting_time + dt.timedelta(days=360)

    # Register the BacktestingBroker with the manager
    MyManager.register("BacktestingBroker", BacktestingBroker)

    print("Creating SyntheticDataProvider_NaiveStepRange...")
    t0 = time.perf_counter()
    data_provider = SyntheticDataProvider_NaiveStepRange(
        t0=starting_time, t1=stopping_time, n_ticks=1_000_000
    )
    t1 = time.perf_counter()
    print(f"Time to create SyntheticDataProvider_NaiveStepRange: {t1 - t0:.2f} seconds")

    with MyManager() as manager:
        shared_broker = manager.BacktestingBroker(
            starting_time=starting_time + dt.timedelta(days=105),
            data_provider=data_provider,
            stop_time=stopping_time,
            second_skip_per_iteration=10.0,
        )

        backtesting_process = mp.Process(
            target=run_backtesting_broker, args=(shared_broker,)
        )
        time_printing_process = mp.Process(
            target=print_broker_time, args=(shared_broker,)
        )

        backtesting_process.start()
        time_printing_process.start()

        backtesting_process.join()
        time_printing_process.join()
