import os
import json
import shutil
import boto3

from src._utils import main_logger


# load the raw data to s3
def load(data, ticker: str = 'NVDA', should_local_save: bool = True) -> str:
    if not data: main_logger.error('missing data'); raise

    try:
        latest_date = next(iter(data.keys()))
    except:
        main_logger.error('... data dictionary is empty ...')
        return ""

    # define file path
    folder_path = os.path.join('data', 'bronze', ticker)
    if os.path.exists(folder_path): shutil.rmtree(folder_path) # clean up previous run's data for a fresh start

    folder_path = os.path.join('data', 'bronze', ticker, latest_date)
    file_name = f'{ticker}_{latest_date}_bronze.json'
    file_path = os.path.join(folder_path, file_name)

    # local load (optional for ol. for testing and debugging)
    if should_local_save:
        os.makedirs(folder_path, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)


    # s3 load using s3a schema
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-stockprice-pred')
    s3_key = os.path.join('data', ticker, 'bronze', f'dt={latest_date}', file_name)
    bronze_s3_path = f's3a://{S3_BUCKET_NAME}/{s3_key}'

    # convert to a json string
    json_string = json.dumps(data)

    AWS_ACCESS_KEY_ID = os.environ.get('AWS_IAM_USER_FOR_SPARK_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_IAM_USER_FOR_SPARK_SECRET_ACCESS_KEY')
    AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

    # initiate s3 boto3 client
    s3_client_spark = boto3.client(
        's3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION
    )
    try:
        s3_client_spark.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=json_string)
        main_logger.info(f'... raw json data loaded to s3 at {bronze_s3_path}')

    except Exception as e:
        main_logger.error(f'... failed to upload to s3: {e} ...')
        raise

    return bronze_s3_path
