from datetime import timedelta
from airflow import DAG
import os
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from airflow.models import Variable
from airflow.operators.bash_operator import BashOperator


"""
Push migrations. Seeding data into the database after import and processing...
"""
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
    'machine_learning_task',
    default_args=default_args,
    description='Build machine learning models',
    schedule_interval=timedelta(days=1),
    tags=['tasks'],
)


run_script = BashOperator(
    task_id='machine_learning_id',
    bash_command="""
        export MPLCONFIGDIR=/tmp/matplotlib && mkdir -p $MPLCONFIGDIR && \
        cd /var/app && \
        python3 ./src/Jobs/MLJobs.py
    """,
    dag=dag,
)
