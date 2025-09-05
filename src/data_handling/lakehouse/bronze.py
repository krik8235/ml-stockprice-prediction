import os
import pandas as pd
import json
import shutil
from deltalake import write_deltalake # type: ignore

from src._utils import main_logger


# read data from s3
def store_data_to_local(data, ticker: str = 'NVDA'):
    bronze_local_path = os.path.join('data', 'bronze', ticker)
    if os.path.exists(bronze_local_path): shutil.rmtree(bronze_local_path) # clean up previous run's data for a fresh start

    os.makedirs(bronze_local_path, exist_ok=True)

    # save the raw json data to a file in the bronze layer
    bronze_file_name = f'{ticker}_bronze.json'
    raw_data_file_path = os.path.join(bronze_local_path, bronze_file_name)

    with open(raw_data_file_path, 'w') as f:
        json.dump(data, f, indent=4)

    main_logger.info(f"... raw json data saved to {raw_data_file_path}.")


# convert original json data to pandas df
def create_df(data, ticker: str = 'NVDA', verbose: bool = False):
    rows = list()
    if data:
        for k, v in data.items():
            row = {
                'ticker': str(ticker),
                'dt': str(k),
                'open': float(v['1. open']),
                'high': float(v['2. high']),
                'low': float(v['3. low']),
                'close': float(v['4. close']),
                'volume': int(v['5. volume'])
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # sort data by dt
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
    df = df.sort_values(by='dt', ascending=True, ignore_index=True)
    if verbose: df.info()
    return df


def load_to_s3(df, ticker: str = 'NVDA') -> str:
    if df is None or df.empty: main_logger.error('missing df'); raise

    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-stockprice-pred')
    # 3. write delta table from df (automatically handles the file organization and metadata creation)
    # define the s3 path for the bronze table using the s3a:// scheme
    bronze_s3_path = f's3a://{S3_BUCKET_NAME}/data/bronze/{ticker}'
    write_deltalake(bronze_s3_path, df, mode='overwrite')

    main_logger.info(f'... delta table files (parquiet and _delta_log) loaded to s3 {bronze_s3_path}')

    return bronze_s3_path
