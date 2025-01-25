from celery import Celery, shared_task
import os

celery = Celery(
    "backend",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
)

celery.config_from_object("celeryconfig")
