import os
import argparse
import pandas as pd
from pyspark.sql import SparkSession

import src.data_handling as data_handling
from src import TICKER
from src._utils import main_logger



def run_lakehouse(ticker: str = TICKER, should_local_save: bool = False) -> tuple[pd.DataFrame, SparkSession]:
    df = None

    # extract
    stock_price_data = data_handling.extract(ticker=ticker)

    # bronze
    bronze_s3_path = data_handling.bronze.load(data=stock_price_data, ticker=ticker)

    # start the spark session
    spark = data_handling.config_and_start_spark_session()

    # silver
    bronze_delta_table = spark.read.json(bronze_s3_path, multiLine=True)
    silver_df = data_handling.silver.transform(delta_table=bronze_delta_table, spark=spark)
    silver_s3_path = data_handling.silver.load(df=silver_df, ticker=ticker)

    # gold
    silver_delta_table = data_handling.retrieve_delta_table(spark=spark, s3_path=silver_s3_path)
    gold_df = data_handling.gold.transform(delta_table=silver_delta_table, spark=spark)
    gold_s3_path = data_handling.gold.load(df=gold_df, ticker=ticker)

    df = gold_df.toPandas()

    if gold_df and should_local_save:
        # save gold_df as pandas df in parquet / csv file
        df['dt'] = pd.to_datetime(df['dt'])

        df_folder_path = os.path.join('data', 'df', ticker)
        os.makedirs(df_folder_path, exist_ok=True)

        df_parquet_file_name = 'df.parquet'
        df_csv_file_name = 'df.csv'

        df.to_parquet(os.path.join(df_folder_path, df_parquet_file_name), index=False)
        df.to_csv(os.path.join(df_folder_path, df_csv_file_name), index=False)

        main_logger.info(f'... pyspark df successfully convert to pandas df and saved to {os.path.join(df_folder_path, df_parquet_file_name)} and {os.path.join(df_folder_path, df_csv_file_name)}. returning pandas df for donwstream use ...')

    if not gold_s3_path: main_logger.error('... failed to load gold df. return none ...')

    return df, spark



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run batch learning")
    parser.add_argument('--ticker', type=str, default=TICKER, help="ticker")
    args = parser.parse_args()

    # transform data
    df, spark = run_lakehouse(ticker=args.ticker)

    # if no df created, retrieve df
    if df is None:
        df_local_file_path = os.path.join('data', 'df', args.ticker, 'df.parquet')
        df = pd.read_parquet(df_local_file_path)

    df.info()

    spark.stop()
