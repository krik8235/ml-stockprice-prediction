import json
import argparse
import datetime
import asyncio
import websockets

from src.model import ModelHandler
from src import PORT, HOST, TICKER
from src._utils import main_logger


async def client_handler(websocket_uri: str, model_handler: ModelHandler, max_retries: int = 5, delay: int = 1):
    """connects to the websocket server and handle online learning."""

    for attempt in range(max_retries):
        try:
            async with websockets.connect(websocket_uri, ping_interval=5) as websocket:
                main_logger.info(f"... successfully connected to the websocket server at {websocket_uri} ...")

                # loop to retrieve micro batches
                async for message in websocket:
                    try:
                        # deserialize the current batch
                        current_batch = json.loads(message)

                        if current_batch and isinstance(current_batch, list):
                            # cast dtypes
                            for record in current_batch:
                                if isinstance(record['dt'], str):
                                    record['dt'] = datetime.datetime.fromisoformat(record['dt'])

                            # start online learning
                            model_handler.online_learning(current_batch=current_batch)
                        else:
                            main_logger.info(f"... received empty or invalid data: {message} ...")

                    except json.JSONDecodeError: main_logger.error(f"... error decoding json message: {message}")
                    except Exception as e: main_logger.error(f"... an error occurred during training: {e}")

        except websockets.exceptions.ConnectionClosedOK:
            main_logger.info("... connection closed by server ...")
            break

        except ConnectionRefusedError:
            main_logger.error(f"... connection refused. retrying in {delay}s (attempt {attempt + 1}/{max_retries}) ...")

            # exponential backoff for connection attempts
            await asyncio.sleep(delay * (2 ** attempt))

        except Exception as e:
            main_logger.error(f"... unhandled connection error: {e}")
            break

    main_logger.info("... client shutting down ...")


def main(ticker: str, should_refresh: bool = False):
    try:
        websocket_uri = f"ws://{HOST}:{PORT}"
        asyncio.run(client_handler(websocket_uri=websocket_uri, model_handler=ModelHandler(ticker=ticker, should_refresh=should_refresh)))

    except KeyboardInterrupt:
        main_logger.info("\n--- client interrupted and shutting down ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creating micro batch for online learning')
    parser.add_argument('--ticker', type=str, default=TICKER, help=f"ticker. default = {TICKER}")
    parser.add_argument('--should_refresh', type=bool, default=False, help=f"whether to retrieve base df from the api. default = False")
    args = parser.parse_args()

    main(ticker=args.ticker, should_refresh=args.should_refresh)
