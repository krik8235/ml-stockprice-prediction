import os
import pandas as pd
from deltalake import DeltaTable, write_deltalake
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, DateType, FloatType, IntegerType, LongType
from dotenv import load_dotenv

from src._utils import main_logger


GOLD_SCHEMA = StructType([
    StructField('dt', DateType(), False),
    StructField('open', FloatType(), False),
    StructField('high', FloatType(), False),
    StructField('low', FloatType(), False),
    StructField('close', FloatType(), False),
    StructField('volume', IntegerType(), False),
    StructField('timestamp_in_ms', LongType(), False),
    StructField('ave_open', FloatType(), False),
    StructField('ave_high', FloatType(), False),
    StructField('ave_low', FloatType(), False),
    StructField('ave_close', FloatType(), False),
    StructField('total_volume', IntegerType(), False),
    StructField('30_day_ma_close', FloatType(), False),
    StructField('year', IntegerType(), False),
    StructField('month', IntegerType(), False),
    StructField('date', IntegerType(), False),
])


def transform(delta_table, spark, should_filter: bool = False):
    load_dotenv(override=True)

    # find latest data row
    latest_dt_df = delta_table.agg({ 'dt': 'max' })
    latest_dt_row = latest_dt_df.withColumnRenamed(latest_dt_df.columns[0], 'latest_dt').collect()

    if not latest_dt_row or latest_dt_row[0]['latest_dt'] is None:
        main_logger.error('... no data found in the silver delta table to process. return empty df ...')
        return spark.createDataFrame([], GOLD_SCHEMA) # return empty df with the correct schema

    latest_dt = latest_dt_row[0]['latest_dt']

    # moving ave. requires sorting by date and looking back 29 rows.
    window_spec = Window.orderBy('dt').rowsBetween(-29, 0)
    _gold_df = delta_table.withColumn('30_day_ma_close', F.avg(F.col('close')).over(window_spec))

    # add temporal features
    _gold_df = _gold_df.withColumn('year', F.year(F.col('dt')).cast(IntegerType()))
    _gold_df = _gold_df.withColumn('month', F.month(F.col('dt')).cast(IntegerType()))
    _gold_df = _gold_df.withColumn('date', F.dayofmonth(F.col('dt')).cast(IntegerType()))

    # log transform close
    _gold_df = _gold_df.withColumn('close', F.log1p(F.col('close')))

    # select final columns
    final_cols = [
        'dt', 'open', 'high', 'low', 'close', 'volume', 'timestamp_in_ms',
        F.col('open').alias('ave_open'), F.col('high').alias('ave_high'),
        F.col('low').alias('ave_low'), F.col('close').alias('ave_close'),
        F.col('volume').alias('total_volume'), '30_day_ma_close', 'year', 'month', 'date'
    ]
    gold_df = _gold_df.select(*final_cols)

    # filter result w/ latest_dt (python obj)
    if should_filter: gold_df = gold_df.filter(F.col('dt') == latest_dt)

    # finalize
    gold_df = gold_df.orderBy(F.col('dt').asc())

    main_logger.info(f'... calculated gold layer features for {latest_dt}. resulting data schema:\n')
    gold_df.printSchema()
    return gold_df


def load(df, ticker: str = 'NVDA', should_local_save: bool = True) -> str:
    # define local/s3 path for the gold layer

    gold_local_path = os.path.join('data', 'gold', ticker)
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-stockprice-pred')
    gold_s3_path = f's3a://{S3_BUCKET_NAME}/data/gold/{ticker}'

    # convert to pandas df
    pandas_df = df if isinstance(df, pd.DataFrame) else df.toPandas()

    # check if delta table exists
    try:
        # instantiate delta table using native deltalake
        gold_table = DeltaTable(gold_s3_path)

        # merge
        merger = gold_table.merge(
            source=pandas_df,
            predicate='target.dt = source.dt',
            source_alias='source',
            target_alias='target'
        )
        merger.when_matched_update_all().when_not_matched_insert_all().execute()

        main_logger.info(f'... pandas df successfully MERGED into the gold layer at {gold_s3_path} ...')

    # if not delta table exists
    except:
        main_logger.info(f'... data in the silver df dont exist. create one ...')
        write_deltalake(gold_s3_path, pandas_df, mode='overwrite')
        main_logger.info(f'... gold table created at {gold_s3_path} ...')

    # local saving (optional for ol)
    if should_local_save:
        os.makedirs(gold_local_path, exist_ok=True)
        df.write.format('parquet').mode('append').option('overwriteSchema', 'true').save(gold_local_path)
        main_logger.info(f'... single record appended to local gold data at {gold_local_path} ...')

    return gold_s3_path
