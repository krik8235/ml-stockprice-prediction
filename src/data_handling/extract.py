import os
import json
import requests
from dotenv import load_dotenv

from src._utils import main_logger


def extract_daily_stock_data(ticker: str = 'NVDA', function: str = 'TIME_SERIES_DAILY') -> dict:
    load_dotenv(override=True)

    API_URL = 'https://www.alphavantage.co/query'
    api_key = os.environ.get('API_KEY', None)
    if not api_key: main_logger.error('... missing api key ...'); raise

    params = {
        'function': function,
        'symbol': ticker,
        'outputsize': 'compact', # ol - extract only single latest day results
        'apikey': api_key
    }

    try:
        res = requests.get(API_URL, params=params)
        res.raise_for_status()
        data = res.json()
        stock_price_data = data['Time Series (Daily)']

        # get the most recent date (the first key in the dictionary)
        latest_date = next(iter(stock_price_data))

        # return only the data for the most recent day
        latest_day_data = {latest_date: stock_price_data[latest_date]}
        return latest_day_data

    except:
        main_logger.error(f'... failed to fetch. return an empty dict ...')
        return {}

        # try:
        #     local_file_path = None
        #     for dirpath, _, filenames in os.walk('data/raw'):
        #         for filename in filenames:
        #             if filename.startswith(ticker): local_file_path = os.path.join(dirpath, filename); break

        #     if local_file_path is not None:
        #         with open(local_file_path, 'r') as file:
        #             stock_price_data =  json.load(file)
        # except:
        #     raise

    # return stock_price_data
