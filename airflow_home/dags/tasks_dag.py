from datetime import timedelta
from airflow import DAG
import os
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from airflow.models import Variable
from airflow.operators.bash_operator import BashOperator

# Set default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(year=2024, month=4, day=19, hour=22, minute=52),
    'schedule': "@weekly",
    'email': ['ebenco94@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'catchup': True,
    'max_active_runs': 1,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}   

# Define the DAG
dag = DAG(
    'complex_data_processing',
    default_args=default_args,
    description='A more complex DAG for data processing',
    schedule_interval=timedelta(days=1),
    tags=['tasks'],
)


run_script = BashOperator(
    task_id='databaseLoad',
    bash_command="bash {{ var.value.script_file_path }}",
    dag=dag,
)

# Set task dependencies
#t1 >> t2 >> t3 >> t4
