import os

from redis import Redis


def get_redis_url(config=None):
    if config is not None:
        if config.get("CELERY_BROKER_URL", "").startswith("redis://"):
            return config.get("CELERY_BROKER_URL")
        if config.get("CELERY_RESULT_BACKEND", "").startswith("redis://"):
            return config.get("CELERY_RESULT_BACKEND")
        redis_host = config.get("REDIS_HOST")
        if redis_host:
            return f"redis://{redis_host}:6379/0"

    broker_url = os.getenv("CELERY_BROKER_URL", "")
    if broker_url.startswith("redis://"):
        return broker_url

    backend_url = os.getenv("CELERY_RESULT_BACKEND", "")
    if backend_url.startswith("redis://"):
        return backend_url

    redis_host = os.getenv("REDIS_HOST")
    if redis_host:
        return f"redis://{redis_host}:6379/0"

    return None


def get_redis_client(config=None):
    redis_url = get_redis_url(config=config)
    if not redis_url:
        return None
    return Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
