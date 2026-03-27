from app import app as flask_app
from src.core.celery_factory import celery, configure_celery


configure_celery(flask_app)
