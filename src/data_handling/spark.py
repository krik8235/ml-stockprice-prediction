import os
from pyspark.sql import SparkSession # type: ignore


def config_and_start_spark_session() -> SparkSession:
    os.environ['SPARK_HOME'] = '/Library/spark-4.0.0-bin-hadoop3'
    os.environ['JAVA_HOME'] = '/opt/homebrew/opt/openjdk@17'
    os.environ['PYSPARK_PYTHON'] = 'python'


    # aws credentials
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

    # define the packages to connect delta lake, hadoop, and s3
    # compatible versions of delta lake v2.13.4.0.0 https://mvnrepository.com/artifact/io.delta/delta-spark_2.13/4.0.0
    spark_packages = ','.join([
        'io.delta:delta-spark_2.13:4.0.0', # delta lake package
        'org.apache.hadoop:hadoop-aws:3.4.0', # hadoop aws for s3. must match with the version defined in the $SPARK_HOME env
        'software.amazon.awssdk:bundle:2.23.19' # aws sdk for java. check the compatible version with hadoop from the official Marven site - https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-aws/3.4.0
    ])

    # config spark session including the spark pakages and aws credentials
    spark = SparkSession.builder.appName('Silver') \
        .config('spark.jars.packages', spark_packages) \
        .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension') \
        .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog') \
        .config('spark.driver.memory', '8g') \
        .config('spark.hadoop.fs.s3a.access.key', AWS_ACCESS_KEY_ID) \
        .config('spark.hadoop.fs.s3a.secret.key', AWS_SECRET_ACCESS_KEY) \
        .config('spark.hadoop.fs.s3a.endpoint', f's3.{AWS_REGION}.amazonaws.com') \
        .getOrCreate()

    return spark
