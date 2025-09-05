import os
import datetime
import json
import boto3 # type: ignore
from dotenv import load_dotenv # type: ignore

from src._utils import main_logger


def load_to_s3(data, ticker: str = 'NVDA', verbose: bool = True):
    if not data: main_logger.error('missing data.'); return

    load_dotenv(override=True)

    # store in local
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-stockprice-pred')

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = f'{ticker}_{timestamp}.json'
    local_file_path = os.path.join('data', 'raw', file_name)
    with open(local_file_path, 'w') as f:
        json.dump(data, f, indent=4)
        if verbose: main_logger.info(f'... data stored as json file in local {local_file_path}')

    # store in s3
    s3_client = boto3.client('s3', region_name=os.environ.get('AWS_REGION_NAME', 'us-east-1'))
    s3_key_raw = os.path.join('data', 'raw', file_name)
    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key_raw, Body=json.dumps(data))

    main_logger.info(f'... data uploaded to s3://{S3_BUCKET_NAME}/{s3_key_raw}')
