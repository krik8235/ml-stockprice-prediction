import os
import shutil
import json
import pandas as pd
from pyspark.sql import SparkSession # type: ignore
from pyspark.sql.functions import col, to_timestamp, regexp_replace # type: ignore

from src._utils import main_logger


def process(delta_table) -> pd.DataFrame:
    # preprocess and transform bronze data and store in the silver table
    main_logger.info('... pre-processing and transforming data for the silver layer ...')

    # data clearning and type casting
    silver_df = delta_table.select(
        col('dt').cast('date').alias('dt'), # cast the timestamp to date type
        col('open').cast('float'), # cast all price columns to float
        col('high').cast('float'),
        col('low').cast('float'),
        col('close').cast('float'),
        col('volume').cast('integer') # cast to integer
    )
    main_logger.info(f'... transformed data schema in the silver layer: \n{silver_df.printSchema()}')

    return silver_df



def load(df, ticker: str = 'NVDA', should_local_save: bool = True) -> str:
    silver_local_path = os.path.join('data', 'silver', ticker)

    # clean up previous run's data for a fresh start
    if os.path.exists(silver_local_path):
        main_logger.info(f'... cleaning up existing Silver layer data at {silver_local_path} ...')
        shutil.rmtree(silver_local_path)

    # store as a parquet file in local
    if should_local_save:
        os.makedirs(silver_local_path, exist_ok=True)
        df.write.format('parquet').mode('overwrite').option('overwriteSchema', 'true').save(silver_local_path)

    # write the df in the silver layer as a new delta table and store it in s3
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-stockprice-pred')
    silver_s3_path = f's3a://{S3_BUCKET_NAME}/data/silver/{ticker}'

    df.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').save(silver_s3_path)
    main_logger.info(f'... data successfully written to the silver layer at {silver_s3_path}. terminate the spark session ...')

    return silver_s3_path
