import pytest
import datetime
import unittest
from unittest import mock
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator # type: ignore

from src._utils import main_logger


# mock func to called by the pythonoperator (avoid running actual pyspark task)
def _run_pyspark_job():
    main_logger.info('... [MOCK] this is a mock run of the PySpark job ...')


# create and return dag instance
def create_test_dag():
    default_args = {
        'owner': 'kuriko iwai',
        'start_date': datetime.datetime(2025, 10, 9), # use a fixed date for reproducible tests
        'retries': 1,
    }
    with DAG(
        dag_id='pyspark_pytest',
        default_args=default_args,
        description='mock',
        schedule='@daily',
        catchup=False,
        tags=['pyspark', 'etl', 'gold-layer'],
    ) as dag:

        # define a task
        run_pyspark_task = PythonOperator(
            task_id='process_data_to_gold',
            python_callable=_run_pyspark_job,
        )
    return dag


def test_dag_properties():
    """
    Tests that the DAG's basic properties are correctly defined.
    """
    dag = create_test_dag()
    assert dag is not None
    assert dag.dag_id == 'pyspark_pytest'
    assert dag.owner == 'kuriko iwai'
    assert dag.description == 'mock'
    assert dag.schedule == '@daily'
    assert 'pyspark' in dag.tags
    assert 'etl' in dag.tags
    assert 'gold-layer' in dag.tags


def test_dag_contains_expected_task():
    """
    Tests that the DAG contains the expected task.
    """
    dag = create_test_dag()

    # check if a task with the specific id exists in the dag
    assert 'process_data_to_gold' in dag.task_ids

    # get task instance from the dag and check if the task is correct type with correct properties, and correct callable
    task = dag.get_task('process_data_to_gold')
    assert isinstance(task, PythonOperator)
    assert task.task_id == 'process_data_to_gold'
    assert task.owner == 'kuriko iwai'
    assert task.python_callable.__name__ == '_run_pyspark_job'
