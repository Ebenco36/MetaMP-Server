from pathlib import Path

from celery import Celery
from celery.schedules import timedelta

celery = Celery("backend")


def configure_celery(app):
    broker_url = app.config.get("CELERY_BROKER_URL", "redis://redis:6379/0")
    result_backend = app.config.get(
        "CELERY_RESULT_BACKEND",
        broker_url,
    )
    beat_schedule_filename = app.config.get(
        "CELERY_BEAT_SCHEDULE_FILENAME",
        "./celery-beat/celerybeat-schedule",
    )
    Path(beat_schedule_filename).parent.mkdir(parents=True, exist_ok=True)

    celery.conf.update(
        broker_url=broker_url,
        result_backend=result_backend,
        timezone=app.config.get("CELERY_TIMEZONE", "UTC"),
        imports=("src.Jobs.tasks.task1",),
        beat_schedule_filename=beat_schedule_filename,
        task_track_started=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        broker_connection_retry_on_startup=True,
        result_expires=86400,
    )

    celery.conf.beat_schedule = {
        "monthly-production-maintenance": {
            "task": "shared-task-monthly-production-maintenance",
            "schedule": timedelta(days=30),
        }
    }

    class ContextTask(celery.Task):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
