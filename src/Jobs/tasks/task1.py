from celery import shared_task
from src.Jobs.MLJobs import MLJob
# import os, sys
# sys.path.append(os.getcwd())
from flask import current_app
import subprocess

@shared_task(name="shared-task-machine-learning-job")
def machine_learning_job():
    from app import create_app  # Dynamically import app to access context
    app = create_app()
    with app.app_context():
        ml_job = MLJob()
        ml_job.fix_missing_data()\
            .variable_separation()\
            .feature_selection()\
            .dimensionality_reduction()\
            .plot_charts()\
            .semi_supervised_learning()\
            .supervised_learning()

    # Log the success
    print("Machine Learning Job completed successfully.")

@shared_task(name="shared-task-sync-protein-database")
def scheduled_sync_command():
    """
    Run the `sync-protein-database` Flask CLI command periodically.
    """
    from app import create_app  # Dynamically import app to access context
    app = create_app()
    with app.app_context():
        result = subprocess.run(
            ["flask", "sync-protein-database"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            app.logger.info(f"Command succeeded: {result.stdout}")
        else:
            app.logger.error(f"Command failed: {result.stderr}")
        
@shared_task(name="shared-task-sync-system_admin-with-database")
def system_admin_scheduled_sync_command():
    """
    Run the `sync-protein-database` Flask CLI command periodically.
    """
    from app import create_app  # Dynamically import app to access context
    app = create_app()
    with app.app_context():
        result = subprocess.run(
            ["flask", "sync-system_admin-with-database"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            app.logger.info(f"Command succeeded: {result.stdout}")
        else:
            app.logger.error(f"Command failed: {result.stderr}")
        
@shared_task(name="shared-task-sync-feedback-questions-with-database")
def feedback_scheduled_sync_command():
    """
    Run the `sync-protein-database` Flask CLI command periodically.
    """
    from app import create_app  # Dynamically import app to access context
    app = create_app()
    with app.app_context():
        result = subprocess.run(
            ["flask", "sync-feedback-questions-with-database"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            app.logger.info(f"Command succeeded: {result.stdout}")
        else:
            app.logger.error(f"Command failed: {result.stderr}")   

@shared_task(name="shared-task-sync-question-with-database")
def question_feedback_scheduled_sync_command():
    """
    Run the `sync-protein-database` Flask CLI command periodically.
    """
    from app import create_app  # Dynamically import app to access context
    app = create_app()
    with app.app_context():
        result = subprocess.run(
            ["flask", "sync-question-with-database"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            current_app.logger.info(f"Command succeeded: {result.stdout}")
        else:
            current_app.logger.error(f"Command failed: {result.stderr}")
