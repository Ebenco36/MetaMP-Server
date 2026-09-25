import json
import logging
from datetime import datetime, timezone

from redis import RedisError

from src.ingestion.redis_support import get_redis_client

logger = logging.getLogger(__name__)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class TaskStatusRecorder:
    def __init__(self, app_config=None):
        self.redis = get_redis_client(config=app_config)
        self.ttl_seconds = 7 * 24 * 60 * 60

    def record_started(self, task_name, task_id, extra=None):
        payload = {
            "task_name": task_name,
            "task_id": task_id,
            "status": "started",
            "started_at": utc_now(),
        }
        if extra:
            payload.update(extra)
        self._write(task_name, payload)

    def record_progress(self, task_name, task_id, message, stage_name=None, extra=None):
        payload = self._read_existing(task_name, task_id) or {
            "task_name": task_name,
            "task_id": task_id,
            "started_at": utc_now(),
        }
        payload.update(
            {
                "task_name": task_name,
                "task_id": task_id,
                "status": "running",
                "updated_at": utc_now(),
                "message": message,
            }
        )
        if stage_name:
            payload["current_stage"] = stage_name
        if extra:
            payload.update(extra)
        self._write(task_name, payload)

    def record_succeeded(self, task_name, task_id, extra=None):
        payload = self._read_existing(task_name, task_id) or {
            "task_name": task_name,
            "task_id": task_id,
        }
        payload.update({"status": "succeeded", "finished_at": utc_now()})
        if extra:
            payload.update(extra)
        self._write(task_name, payload)

    def record_failed(self, task_name, task_id, error_message, extra=None):
        payload = self._read_existing(task_name, task_id) or {
            "task_name": task_name,
            "task_id": task_id,
        }
        payload.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": error_message,
            }
        )
        if extra:
            payload.update(extra)
        self._write(task_name, payload)

    def record_skipped(self, task_name, task_id, reason, extra=None):
        payload = self._read_existing(task_name, task_id) or {
            "task_name": task_name,
            "task_id": task_id,
        }
        payload.update(
            {
                "status": "skipped",
                "finished_at": utc_now(),
                "reason": reason,
            }
        )
        if extra:
            payload.update(extra)
        self._write(task_name, payload)

    def _write(self, task_name, payload):
        if self.redis is None:
            return

        key = f"metamp:tasks:{task_name}:latest"
        try:
            self.redis.set(key, json.dumps(payload), ex=self.ttl_seconds)
        except RedisError as exc:
            logger.warning(
                "Redis unavailable while writing task status for %s: %s",
                task_name,
                exc,
            )

    def read_latest(self, task_name):
        if self.redis is None:
            return None

        key = f"metamp:tasks:{task_name}:latest"
        try:
            payload = self.redis.get(key)
        except RedisError as exc:
            logger.warning(
                "Redis unavailable while reading task status for %s: %s",
                task_name,
                exc,
            )
            return None
        if not payload:
            return None
        return json.loads(payload)

    def _read_existing(self, task_name, task_id):
        payload = self.read_latest(task_name)
        if not payload:
            return None
        if payload.get("task_id") != task_id:
            return None
        return payload
