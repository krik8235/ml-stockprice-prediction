import datetime
from airflow import DAG # type: ignore
from airflow.operators.python import PythonOperator  # type: ignore

from src import main

# define the airflow dag
default_args = {
    'owner': 'kuriko iwai',
    'start_date': datetime.datetime.now(),
    'retries': 1,
}

with DAG(
    dag_id='pyspark_data_handling',
    default_args=default_args,
    description='A daily ETL job to process stock data into the gold layer.',
    schedule='@daily',
    catchup=False,
    tags=['pyspark', 'elt', 'lakehouse'],
) as dag:

    # define a single task using PythonOperator
    run_pyspark_task = PythonOperator(
        task_id='elt_lakehouse',
        python_callable=main,
    )
