import os
import json
import requests
from dotenv import load_dotenv

from src._utils import main_logger

def extract(ticker: str = 'NVDA', function: str = 'TIME_SERIES_DAILY') -> dict:
    load_dotenv(override=True)

    API_URL = 'https://www.alphavantage.co/query'
    api_key = os.environ.get('API_KEY')
    if not api_key: main_logger.error('... missing api key ...'); raise

    params = {
        'function': function,
        'symbol': ticker,
        'outputsize': 'full', # for batch learning
        'apikey': api_key
    }

    try:
        res = requests.get(API_URL, params=params)
        res.raise_for_status()
        data = res.json()
        stock_price_data = data['Time Series (Daily)']
        return stock_price_data

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
