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
    'add_migration_and_questions',
    default_args=default_args,
    description='Migrate into database',
    schedule_interval=timedelta(days=1),
    tags=['tasks'],
)


run_script = BashOperator(
    task_id='quaestionTask',
    bash_command="""
        export MPLCONFIGDIR=/tmp/matplotlib && mkdir -p $MPLCONFIGDIR && \
        cd /var/app && \
        flask sync-question-with-database
    """,
    dag=dag,
)
