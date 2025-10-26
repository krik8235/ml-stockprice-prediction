import datetime
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from pydeequ.checks import Check, CheckLevel
from pydeequ.verification import VerificationSuite, VerificationResult

from src.model import BaselineModel
from src.data_handling.spark import config_and_start_spark_session

# define the airflow dag
default_args = {
    'owner': 'kuriko iwai',
    'start_date': datetime.datetime.now(),
    'retries': 1,
}


def check_data_quality():
    # initialize a sparksession to use deequ
    spark = config_and_start_spark_session(session_name='deequ_quality_check')

    # fetch delta table in the gold layer
    gold_data_path = "s3a://ml-stockprice-pred/data/gold"
    gold_delta_table = spark.read.format("delta").load(gold_data_path)

    # use deequ to run quality check
    check_result = (VerificationSuite(spark)
        .onData(gold_delta_table)
        .addCheck(
            Check(spark_session=spark, level=CheckLevel.Warning, description="Weekly Stock Data Check")
            .hasSize(lambda s: s > 0)  # check if the df is not empty
            .hasCompleteness("close", lambda s: s == 1.0)  # check for missing vals
            .hasCompleteness("volume", lambda s: s == 1.0) # check for missing vals
            .hasMin("open", lambda x: x > 0) # check if val is positive
            .hasMax("volume", lambda x: x < 10000000000) # check for max val
        )
        .run()
    )

    # convert the deequ result to a spark df for analysis and logging
    result_df = VerificationResult.checkResultsAsDataFrame(spark, check_result, pandas=False)
    result_df.show(truncate=False) # type: ignore

    # stop the Spark session
    spark.stop()



with DAG(
    dag_id='pyspark_data_handling',
    default_args=default_args,
    description='A monthly batch learning',
    schedule='@monthly',
    catchup=False,
) as dag:

    # data quality check
    data_quality_check_task = PythonOperator(
        task_id='data_quality_check',
        python_callable=check_data_quality,
        trigger_rule='all_success',
        depends_on_past=True, # ensures it only runs after the previous successful run
        schedule='@monthly',
    )

    # batch learning
    run_batch_learning = PythonOperator(
        task_id='elt_lakehouse',
        python_callable=BaselineModel().initial_batch_learning,
        schedule='@monthly',
    )

    # set task dependencies
    data_quality_check_task >> run_batch_learning  # type: ignore
