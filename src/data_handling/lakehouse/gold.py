import os
import shutil
from pyspark.sql.functions import col, avg, sum, year, month, dayofmonth, log1p# type: ignore
from pyspark.sql.window import Window # type: ignore
from pyspark.sql.types import StructType, StructField, DateType, FloatType, IntegerType, StringType

from src._utils import main_logger


def process(delta_table, spark):
    # perform feature engineering
    main_logger.info('... aggregating and transforming data for the gold layer ...')

    # add averages
    _gold_df = delta_table.groupBy('dt').agg(
        avg(col('open')).alias('ave_open'),
        avg(col('high')).alias('ave_high'),
        avg(col('low')).alias('ave_low'),
        avg(col('close')).alias('ave_close'),
        sum(col('volume')).alias('total_volume')
    )
    # merge
    gold_df = delta_table.join(_gold_df, on='dt', how='inner')

    # add moving average
    window_spec = Window.orderBy('dt').rowsBetween(-29, 0)
    gold_df = gold_df.withColumn('30_day_ma_close', avg(col('ave_close')).over(window_spec))

    # add year, month, date cols
    gold_df = gold_df.withColumn('year', year(gold_df['dt']))
    gold_df = gold_df.withColumn('month', month(gold_df['dt']))
    gold_df = gold_df.withColumn('date', dayofmonth(gold_df['dt']))

    # log transform close
    gold_df = gold_df.withColumn('close', log1p(col('close')))

    # define schema
    schema = StructType([
        StructField('dt', DateType(), False), # explicitly set nullable = false
        StructField('open', FloatType(), False),
        StructField('high', FloatType(), False),
        StructField('low', FloatType(), False),
        StructField('close', FloatType(), False),
        StructField('volume', IntegerType(), False),
        StructField('ave_open', FloatType(), False),
        StructField('ave_high', FloatType(), False),
        StructField('ave_low', FloatType(), False),
        StructField('ave_close', FloatType(), False),
        StructField('total_volume', IntegerType(), False),
        StructField('30_day_ma_close', FloatType(), False),
        StructField('year', StringType(), False),
        StructField('month', StringType(), False),
        StructField('date', StringType(), False),
    ])

    # finalize df
    gold_df = spark.createDataFrame(gold_df.collect(), schema=schema)

    # sort by dt
    gold_df = gold_df.orderBy(col('dt').asc())

    main_logger.info(f'... transformed data schema in the gold layer: \n')
    gold_df.printSchema()
    return gold_df


def load(df, ticker: str = 'NVDA', should_local_save: bool = True) -> str:
    # define local/s3 path for the gold layer
    gold_local_path = os.path.join('data', 'gold', ticker)

    if os.path.exists(gold_local_path):
        main_logger.info(f'... cleaning up existing gold layer data at {gold_local_path} ...')
        shutil.rmtree(gold_local_path)


    # store as a parquet file in local
    if should_local_save:
        os.makedirs(gold_local_path, exist_ok=True)
        df.write.format('parquet').mode('overwrite').option('overwriteSchema', 'true').save(gold_local_path)


    # store in s3
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-stockprice-pred')
    gold_s3_path = f's3a://{S3_BUCKET_NAME}/data/gold/{ticker}'

    df.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').save(gold_s3_path)
    main_logger.info(f'... pyspark df successfully written to the gold layer at {gold_s3_path} ...')

    return gold_s3_path
