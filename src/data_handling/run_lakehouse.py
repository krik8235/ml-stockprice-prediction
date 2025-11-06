import argparse
import pandas as pd
from pyspark.sql import SparkSession

import src.data_handling as data_handling
from src import TICKER


def run_lakehouse(ticker: str = TICKER) -> tuple[pd.DataFrame, SparkSession, str]:
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
    return df, spark, gold_s3_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run batch learning")
    parser.add_argument('--ticker', type=str, default=TICKER, help="ticker")
    args = parser.parse_args()

    # transform data
    df, spark, gold_s3_path = run_lakehouse(ticker=args.ticker)

    df.info()
    spark.stop()
