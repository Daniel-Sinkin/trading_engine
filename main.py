import datetime as dt
import multiprocessing as mp
import time
from multiprocessing.managers import BaseManager

from src.broker import BacktestingBroker
from src.data_provider import SyntheticDataProvider_NaiveStepRange


def run_backtesting_broker(broker: BacktestingBroker, time_multiplier: float):
    broker.start_running(time_multiplier)


def print_broker_time(broker: BacktestingBroker):
    while True:
        print(
            f"Broker time: {broker.get_time()}, current price {broker.get_current_bid_ask()}, total number of candles = {broker.get_total_number_of_candles()}"
        )
        time.sleep(0.1)


class MyManager(BaseManager):
    pass


if __name__ == "__main__":
    starting_time = dt.datetime(2023, 10, 12, 9, 30, tzinfo=dt.timezone.utc)

    # Register the BacktestingBroker with the manager
    MyManager.register("BacktestingBroker", BacktestingBroker)

    print("Creating SyntheticDataProvider_NaiveStepRange...")
    t0 = time.perf_counter()
    data_provider = SyntheticDataProvider_NaiveStepRange(
        t0=starting_time, t1=starting_time + dt.timedelta(days=30), n_ticks=500_000
    )
    t1 = time.perf_counter()
    print(f"Time to create SyntheticDataProvider_NaiveStepRange: {t1 - t0:.2f} seconds")

    with MyManager() as manager:
        shared_broker = manager.BacktestingBroker(
            supported_symbols=["EURUSD", "GBPUSD"],
            starting_time=starting_time,
            data_provider=data_provider,
        )

        backtesting_process = mp.Process(
            target=run_backtesting_broker, args=(shared_broker, 60.0 * 12.0)
        )
        time_printing_process = mp.Process(
            target=print_broker_time, args=(shared_broker,)
        )

        backtesting_process.start()
        time_printing_process.start()

        backtesting_process.join()
        time_printing_process.join()
