import os
import shutil
from pyspark.sql.functions import col, avg, sum # type: ignore
from pyspark.sql.window import Window # type: ignore

from src._utils import main_logger


def process(delta_table):
    # perform feature engineering
    main_logger.info('... aggregating and transforming data for the gold layer ...')

    # add averages
    gold_df = delta_table.groupBy('dt').agg(
        avg(col('open')).alias('ave_open'),
        avg(col('high')).alias('ave_high'),
        avg(col('low')).alias('ave_low'),
        avg(col('close')).alias('ave_close'),
        sum(col('volume')).alias('total_volume')
    )

    # add moving average (ma)
    window_spec = Window.orderBy('dt').rowsBetween(-29, 0)
    gold_df = gold_df.withColumn("30_day_ma_close", avg(col('ave_close')).over(window_spec))

    main_logger.info(f'... transformed data schema in the gold layer: \n{gold_df.printSchema()}')

    return gold_df


def load(df, ticker: str = 'NVDA', should_local_save: bool = True) -> str:
    # define local/s3 path for the gold layer
    gold_local_path = os.path.join('data', 'gold', ticker)

    if os.path.exists(gold_local_path):
        main_logger.info(f'... cleaning up existing Silver layer data at {gold_local_path} ...')
        shutil.rmtree(gold_local_path)


    # store as a parquet file in local
    if should_local_save:
        os.makedirs(gold_local_path, exist_ok=True)
        df.write.format('parquet').mode('overwrite').option('overwriteSchema', 'true').save(gold_local_path)


    # store in s3
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-stockprice-pred')
    gold_s3_path = f's3a://{S3_BUCKET_NAME}/data/gold/{ticker}'

    df.write.format('delta').mode('overwrite').option("overwriteSchema", "true").save(gold_s3_path)
    main_logger.info(f'... data successfully written to the gold layer at {gold_s3_path}. terminate the spark session ...')

    return gold_s3_path
