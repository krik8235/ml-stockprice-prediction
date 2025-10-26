import argparse
import asyncio
import datetime
import json
import time
import yfinance as yf
import websockets
import random
import numpy as np
from collections import deque
from typing import Dict, Any, Set, List

from src import TICKER, PORT, HOST
from src._utils import main_logger


# global set to hold all active websocker connections (clients)
CLIENTS: Set[websockets.ClientConnection] = set()


# json encoder from np to json
class CustomJsonEncoder(json.JSONEncoder):
    """handles serialization for types not natively supported by json"""
    def default(self, obj):
        if isinstance(obj, np.float64): return float(obj)
        if isinstance(obj, datetime.datetime): return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


async def register(websocket: websockets.ClientConnection):
    CLIENTS.add(websocket)
    main_logger.info(f"... client connected. total active clients: {len(CLIENTS)} ...")


async def unregister(websocket: websockets.ClientConnection):
    CLIENTS.remove(websocket)
    main_logger.info(f"... client disconnected. total active clients: {len(CLIENTS)} ...")


def fetch_data(ticker: str) -> Dict[str, Any]:
    """fetch the latest data from yfinance. yfinance performs synchronous network i/o."""

    try:
        # ticker obj
        stock = yf.Ticker(ticker)

        # fetch the latest 1-minute historical data
        data = stock.history(period="1d", interval="1m")
        if data.empty: main_logger.info(f"... no data returned for {ticker}..."); return dict()

        # get the latest row
        latest_data = data.iloc[-1]

        return {
            'open': latest_data.get('Open', 0.0),
            'high': latest_data.get('High', 0.0),
            'low': latest_data.get('Low', 0.0),
            'close': latest_data.get('Close', 0.0),
            'volume': int(latest_data.get('Volume', 0.0)),
            'dt': datetime.datetime.now(),
            "timestamp_in_ms": int(time.time() * 1000)
        }

    except Exception as e:
        main_logger.error(f"... error fetching data for {ticker}: {e}")
        return dict()


def create_micro_batch(er_buffer: deque, batch_size: int) -> List[Dict[str, Any]]:
    """creates a micro batch for online learning with experience replay and recency-biased sampling"""

    # not enough samples to create a micro batch. return all available data
    if len(er_buffer) < batch_size: return list(er_buffer)

    # recency bias - include the latest sample
    latest_experience = er_buffer[-1]
    batch = [latest_experience]

    # sample remaining items from the past experiences in the er buffer. (using a simple random sample)
    num_samples_to_add = batch_size - 1
    past_experiences = list(er_buffer)[:-1]
    sampled_experiences = random.sample(past_experiences, num_samples_to_add)

    # add past experiences to the batch
    batch.extend(sampled_experiences)

    # sort samples in order of timestamp
    sorted_batch = sorted(batch, key=lambda d: d['timestamp_in_ms'])
    return sorted_batch


async def streaming(ticker: str, interval_seconds: int, er_buffer, batch_size: int):
    """core loop function to poll the data source and pushes updates to clients"""

    main_logger.info(f"... starting data streaming for {ticker}, polling every {interval_seconds} seconds ...")
    count = 0

    while True:
        count += 1
        # call the io layer to fetch the latest raw data asynchronously. using asyncio.to_thread() avoids blocking the main asynchronous event loop
        new_data = await asyncio.to_thread(fetch_data, ticker)

        if new_data:
            # store the new data in the er buffer
            er_buffer.append(new_data)
            main_logger.info(f"... # {count} appended new data to the er buffer (buffer size: {len(er_buffer)}) ...")

            if len(er_buffer) >= batch_size:
                # create a micro batch
                current_batch = create_micro_batch(er_buffer=er_buffer, batch_size=batch_size)
                message = json.dumps(current_batch, cls=CustomJsonEncoder)

                # broadcast the batch to the clients
                if CLIENTS:
                    try:
                        websockets.broadcast(CLIENTS, message)
                        main_logger.info(f"... successfully broadcasted the current micro batch ({len(current_batch)} items) to {len(CLIENTS)} clients.")
                    except Exception as e: main_logger.error(f"... error during broadcast: {e}")
                else:
                    main_logger.info("... current batch is ready, but no active clients to broadcast to ...")
            else:
                 main_logger.info(f"... buffer filling up ({len(er_buffer)}/{batch_size}) ...")

        # wait for the next polling interval
        await asyncio.sleep(interval_seconds)


async def handler(websocket: websockets.ClientConnection):
    """
    handles a single incoming client connection. register clients and keep the connection alive.
    """
    await register(websocket)
    try: await websocket.wait_closed()
    finally: await unregister(websocket)


async def main(ticker: str, interval: int, port: int, host: str, er_buffer, batch_size: int):
    """the main function to running the websocket server and streaming loop."""

    # start the core data fetching/pushing task
    streaming_task = asyncio.create_task(streaming(ticker=ticker, interval_seconds=interval, er_buffer=er_buffer, batch_size=batch_size))
    main_logger.info(f"... starting websocket server for {ticker} on ws://{host}:{port} ...")

    try:
        async with websockets.serve(handler, host, port): # type: ignore
            await asyncio.Future() # keeps the server running indefinitely
    except Exception as e:
        main_logger.error(f"... server failed to start: {e}")
    finally:
        streaming_task.cancel() # clean up the streaming task when the server stops


if __name__ == '__main__':
    # handle args
    POLLING_INTERVAL = 1
    ER_MAX_BUFFER_SIZE = 300  # for er
    MICRO_BATCH_SIZE = 4

    parser = argparse.ArgumentParser(description='creating micro batch for online learning')
    parser.add_argument('--ticker', type=str, default=TICKER, help=f"ticker. default = {TICKER}")
    parser.add_argument('--interval', type=int, default=POLLING_INTERVAL, help=f"polling interval. default = {POLLING_INTERVAL}")
    parser.add_argument('--port', type=int, default=PORT, help=f"port. default = {PORT}")
    parser.add_argument('--host', type=str, default=HOST, help=f"host. default = {HOST}")
    parser.add_argument('--er_max_buffer_size', type=int, default=ER_MAX_BUFFER_SIZE, help=f"max number of old samples to store. default = {ER_MAX_BUFFER_SIZE}")
    parser.add_argument('--batch_size', type=int, default=MICRO_BATCH_SIZE, help=f"micro batch size for online learning. default = {MICRO_BATCH_SIZE}")
    args = parser.parse_args()

    ER_BUFFER: deque[Dict[str, Any]] = deque(maxlen=args.er_max_buffer_size)

    # start running websocket
    try:
        asyncio.run(
            main(ticker=args.ticker, interval=args.interval, port=args.port, host=args.host, er_buffer=ER_BUFFER, batch_size=args.batch_size)
        )
    except KeyboardInterrupt: main_logger.info("\n--- server shutting down ---")
    except Exception as e: main_logger.error(f"... an unexpected error occurred: {e}")
