import os
import pandas as pd
from deltalake import DeltaTable, write_deltalake
from pyspark.sql.functions import col, expr, to_date, unix_millis
from pyspark.sql.types import StructType, StructField, DateType, FloatType, IntegerType, LongType

from src._utils import main_logger


SILVER_SCHEMA = StructType([
    StructField('dt', DateType(), False),
    StructField('open', FloatType(), False),
    StructField('high', FloatType(), False),
    StructField('low', FloatType(), False),
    StructField('close', FloatType(), False),
    StructField('volume', IntegerType(), False),
    StructField('timestamp_in_ms', LongType(), False)
])


def transform(delta_table, spark):
    main_logger.info('... pre-processing and transforming data for the silver layer ...')

    # get all the date-like column names
    date_columns = delta_table.columns

    # build the expression for the stack function
    stack_expr = f'stack({len(date_columns)}, '
    for date_col in date_columns:
        stack_expr += f'"{date_col}", `{date_col}`, '

    stack_expr = stack_expr.strip(', ') + ')'

    # use stack to unpivot the data from wide to tall format
    _silver_df = delta_table.select(expr(stack_expr).alias('dt_string', 'values'))

    # process the unpivoted df to cast types and rename columns
    silver_df = _silver_df.select(
        to_date(col('dt_string'), 'yyyy-MM-dd').alias('dt'),
        col('values').getItem('1. open').cast('float').alias('open'),
        col('values').getItem('2. high').cast('float').alias('high'),
        col('values').getItem('3. low').cast('float').alias('low'),
        col('values').getItem('4. close').cast('float').alias('close'),
        col('values').getItem('5. volume').cast('integer').alias('volume'),
        unix_millis(col('dt_string').cast('timestamp')).alias('timestamp_in_ms'),
    ).where(col('dt').isNotNull())

    # finalize df
    silver_df = spark.createDataFrame(silver_df.collect(), schema=SILVER_SCHEMA)

    main_logger.info(f'... transformed data schema in the silver layer:\n')
    silver_df.printSchema()
    return silver_df



def load(df, ticker: str = 'NVDA', should_local_save: bool = True) -> str:
    # silver s3 path
    silver_local_path = os.path.join('data', 'silver', ticker)
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-stockprice-pred')
    silver_s3_path = f's3a://{S3_BUCKET_NAME}/data/silver/{ticker}'

    # convert df to pandas df
    pandas_df = df if isinstance(df, pd.DataFrame) else df.toPandas()

    # check if delta table exists
    try:
        # load the existing Delta Table using the native deltalake api
        silver_table = DeltaTable(silver_s3_path)

        # merge w/ unique key 'dt'
        merger = silver_table.merge(
            source=pandas_df,
            predicate="target.dt = source.dt",
            source_alias='source',
            target_alias='target',
        )
        # update when matched, insert when not matched (new record)
        merger.when_matched_update_all().when_not_matched_insert_all().execute()
        main_logger.info(f'... successfully merged at {silver_s3_path} ...')

    # if delta table not exist
    except:
        main_logger.info(f'... silver table not found. creating ...')
        write_deltalake(
            silver_s3_path,
            pandas_df,
            mode='overwrite',
            storage_options={"AWS_REGION": os.environ.get('AWS_REGION', 'us-east-1')}
        )
        main_logger.info(f'... single record appended to local silver data at {silver_local_path} ...')


    # local saving (optional for ol consistency)
    if should_local_save:
        os.makedirs(silver_local_path, exist_ok=True)
        df.write.format('parquet').mode('append').option('overwriteSchema', 'true').save(silver_local_path)
        main_logger.info(f'... single record appended to local silver data at {silver_local_path} ...')

    return silver_s3_path
