from src._utils import main_logger


def retrieve_delta_table(spark, s3_path):
    if not s3_path: main_logger.error('... missing s3 path to the bronze layer ...'); raise

    # read delta table stored in s3
    try:
        delta_table = spark.read.format('delta').load(s3_path)
        return delta_table

    except Exception as e:
        main_logger.error(f'... failed to retrieve delta table in the bronze layer: {e}. terminate spark session ...')
        spark.stop()
        exit(1)
