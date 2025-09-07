import os
import shutil
from pyspark.sql.functions import col, expr, to_date # type: ignore
from pyspark.sql.types import StructType, StructField, DateType, FloatType, IntegerType

from src._utils import main_logger


def process(delta_table, spark):
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
        col('values').getItem('5. volume').cast('integer').alias('volume')
    ).where(col('dt').isNotNull())

    # explicitly define schema
    schema = StructType([
        StructField('dt', DateType(), False), # explicitly set nullable = false
        StructField('open', FloatType(), False),
        StructField('high', FloatType(), False),
        StructField('low', FloatType(), False),
        StructField('close', FloatType(), False),
        StructField('volume', IntegerType(), False)
    ])

    # finalize df
    silver_df = spark.createDataFrame(silver_df.collect(), schema=schema)

    main_logger.info(f'... transformed data schema in the silver layer:\n')
    silver_df.printSchema()
    return silver_df



def load(df, ticker: str = 'NVDA', should_local_save: bool = True) -> str:
    silver_local_path = os.path.join('data', 'silver', ticker)

    # clean up previous run's data for a fresh start
    if os.path.exists(silver_local_path):
        main_logger.info(f'... cleaning up existing silver layer data at {silver_local_path} ...')
        shutil.rmtree(silver_local_path)

    # store as a parquet file in local
    if should_local_save:
        os.makedirs(silver_local_path, exist_ok=True)
        df.write.format('parquet').mode('overwrite').option('overwriteSchema', 'true').save(silver_local_path)

    # write the df in the silver layer as a new delta table and store it in s3
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-stockprice-pred')
    silver_s3_path = f's3a://{S3_BUCKET_NAME}/data/silver/{ticker}'

    df.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').save(silver_s3_path)
    main_logger.info(f'... pyspark df successfully written to the silver layer at {silver_s3_path} ...')

    return silver_s3_path
