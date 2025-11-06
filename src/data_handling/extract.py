import os
import boto3
import io
import requests
from dotenv import load_dotenv

from src import TICKER
from src._utils import main_logger


def extract(ticker: str = TICKER, function: str = 'TIME_SERIES_DAILY') -> dict:
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


def extract_from_s3(ticker: str = TICKER):
    load_dotenv(override=True)

    s3_client = boto3.client('s3', region_name=os.environ.get('AWS_REGION_NAME', 'us-east-1'))

    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-stockprice-pred')
    gold_s3_path = f's3a://{S3_BUCKET_NAME}/data/gold/{ticker}'
    obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=gold_s3_path)
    buffer = io.BytesIO(obj['Body'].read())
    return buffer
