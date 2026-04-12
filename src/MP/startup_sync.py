import logging
import os
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import current_app
from celery.result import AsyncResult
from sqlalchemy import func

from database.db import db
from src.MP.model_tmalphafold import TMAlphaFoldPrediction
from src.core.celery_factory import celery
from src.ingestion.redis_support import get_redis_client
from src.ingestion.task_status_recorder import TaskStatusRecorder


logger = logging.getLogger(__name__)

TASK_NAME = "shared-task-sync-tmalphafold-predictions"
LOCK_KEY = "metamp:locks:tmalphafold-startup-catchup"
LOCK_FILE_PATH = Path(".locks") / "tmalphafold_startup_catchup.lock"


def _coerce_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _coerce_int(value, default):
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _config_value(name, default=None):
    value = current_app.config.get(name)
    if value not in (None, ""):
        return value
    return os.getenv(name, default)


def _parse_timestamp(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _startup_catchup_enabled():
    return _coerce_bool(
        _config_value("TMALPHAFOLD_STARTUP_CATCHUP_ENABLED", "false"),
        default=False,
    )


def _startup_catchup_stale_days():
    return max(
        0,
        _coerce_int(
            _config_value("TMALPHAFOLD_STARTUP_CATCHUP_STALE_DAYS", 7),
            default=7,
        ),
    )


def _startup_catchup_lock_ttl_seconds():
    return max(
        60,
        _coerce_int(
            _config_value("TMALPHAFOLD_STARTUP_CATCHUP_LOCK_TTL_SECONDS", 1800),
            default=1800,
        ),
    )


def _startup_catchup_max_workers():
    return max(
        1,
        _coerce_int(
            _config_value("TMALPHAFOLD_STARTUP_CATCHUP_MAX_WORKERS", 4),
            default=4,
        ),
    )


def _startup_lock_token():
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4()}"


def _acquire_startup_lock():
    token = _startup_lock_token()
    ttl_seconds = _startup_catchup_lock_ttl_seconds()
    redis_client = get_redis_client(config=current_app.config)
    if redis_client is not None:
        acquired = redis_client.set(LOCK_KEY, token, nx=True, ex=ttl_seconds)
        return bool(acquired), ("redis", token, redis_client)

    LOCK_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    if LOCK_FILE_PATH.exists():
        age_seconds = now - LOCK_FILE_PATH.stat().st_mtime
        if age_seconds > ttl_seconds:
            try:
                LOCK_FILE_PATH.unlink()
            except FileNotFoundError:
                pass
    try:
        fd = os.open(str(LOCK_FILE_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False, ("file", token, None)

    try:
        os.write(fd, token.encode("utf-8"))
    finally:
        os.close(fd)
    return True, ("file", token, None)


def _release_startup_lock(lock_context):
    lock_type, token, redis_client = lock_context
    if lock_type == "redis" and redis_client is not None:
        try:
            current_token = redis_client.get(LOCK_KEY)
            if current_token == token:
                redis_client.delete(LOCK_KEY)
        except Exception:
            logger.debug("Unable to release TMAlphaFold startup Redis lock.", exc_info=True)
        return

    if lock_type == "file":
        try:
            if LOCK_FILE_PATH.exists():
                stored_token = LOCK_FILE_PATH.read_text(encoding="utf-8").strip()
                if stored_token == token:
                    LOCK_FILE_PATH.unlink()
        except Exception:
            logger.debug("Unable to release TMAlphaFold startup file lock.", exc_info=True)


def _latest_success_timestamp():
    latest_status = TaskStatusRecorder(current_app.config).read_latest(TASK_NAME) or {}
    if latest_status.get("status") in {"started", "running"}:
        running_at = (
            _parse_timestamp(latest_status.get("updated_at"))
            or _parse_timestamp(latest_status.get("started_at"))
        )
        if running_at is not None:
            return latest_status.get("status"), running_at
    if latest_status.get("status") == "succeeded":
        succeeded_at = (
            _parse_timestamp(latest_status.get("finished_at"))
            or _parse_timestamp(latest_status.get("updated_at"))
            or _parse_timestamp(latest_status.get("started_at"))
        )
        if succeeded_at is not None:
            return "succeeded", succeeded_at
    return None, None


def _find_live_tmalphafold_task():
    inspector = celery.control.inspect(timeout=2)
    payloads = []
    try:
        payloads.extend(
            [
                inspector.active() or {},
                inspector.reserved() or {},
                inspector.scheduled() or {},
            ]
        )
    except Exception:
        logger.debug("Unable to inspect live Celery TMAlphaFold tasks.", exc_info=True)

    for payload in payloads:
        for _, entries in (payload or {}).items():
            for entry in entries or []:
                task_name = str(entry.get("name") or "").strip()
                if task_name != TASK_NAME:
                    continue
                task_id = str(entry.get("id") or entry.get("request", {}).get("id") or "").strip()
                if task_id:
                    return {
                        "task_id": task_id,
                        "task_name": task_name,
                    }

    latest_status = TaskStatusRecorder(current_app.config).read_latest(TASK_NAME) or {}
    task_id = str(latest_status.get("task_id") or "").strip()
    if not task_id:
        return None
    try:
        state = str(AsyncResult(task_id, app=celery).status or "").strip().lower()
    except Exception:
        logger.debug("Unable to inspect AsyncResult for TMAlphaFold task %s.", task_id, exc_info=True)
        return None
    if state in {"started", "retry", "pending"}:
        return {
            "task_id": task_id,
            "task_name": TASK_NAME,
            "state": state,
        }
    return None


def _tmalphafold_store_snapshot():
    row_count, latest_updated_at = db.session.query(
        func.count(TMAlphaFoldPrediction.id),
        func.max(TMAlphaFoldPrediction.updated_at),
    ).filter(
        TMAlphaFoldPrediction.provider == "TMAlphaFold",
        TMAlphaFoldPrediction.status == "success",
    ).one()
    return int(row_count or 0), _parse_timestamp(latest_updated_at)


def _should_queue_startup_catchup():
    row_count, latest_updated_at = _tmalphafold_store_snapshot()
    reasons = []
    if row_count <= 0:
        reasons.append("no successful normalized TMAlphaFold rows exist yet")

    stale_days = _startup_catchup_stale_days()
    now = datetime.now(timezone.utc)
    status_state, latest_status_at = _latest_success_timestamp()

    if status_state in {"started", "running"}:
        return False, (
            "skipping startup TMAlphaFold catch-up because a TMAlphaFold sync is already "
            "running or was just started"
        )

    freshness_reference = latest_status_at or latest_updated_at
    if freshness_reference is None:
        if row_count > 0:
            reasons.append("no TMAlphaFold sync timestamp was found for the existing normalized rows")
    else:
        age_days = (now - freshness_reference).total_seconds() / 86400
        if age_days >= stale_days:
            reasons.append(
                f"latest TMAlphaFold normalized data is stale ({age_days:.1f} days old; threshold {stale_days} day(s))"
            )

    if not reasons:
        return False, (
            f"skipping startup TMAlphaFold catch-up because the normalized store already has "
            f"{row_count} successful row(s) and is fresh enough"
        )
    return True, "; ".join(reasons)


def maybe_queue_startup_tmalphafold_catchup():
    if not _startup_catchup_enabled():
        logger.info("TMAlphaFold startup catch-up is disabled.")
        return {
            "queued": False,
            "reason": "disabled",
        }

    live_task = _find_live_tmalphafold_task()
    if live_task is not None:
        logger.info(
            "Skipping startup TMAlphaFold catch-up because task %s is already active or queued.",
            live_task.get("task_id"),
        )
        return {
            "queued": False,
            "reason": "task-already-active",
            "task_id": live_task.get("task_id"),
        }

    try:
        should_queue, reason = _should_queue_startup_catchup()
    except Exception as exc:
        logger.warning(
            "Unable to evaluate TMAlphaFold startup catch-up eligibility: %s",
            exc,
        )
        return {
            "queued": False,
            "reason": f"eligibility-check-failed: {exc}",
        }

    if not should_queue:
        logger.info("%s.", reason)
        return {
            "queued": False,
            "reason": reason,
        }

    acquired, lock_context = _acquire_startup_lock()
    if not acquired:
        logger.info(
            "Skipping startup TMAlphaFold catch-up because another process already holds the startup queue lock."
        )
        return {
            "queued": False,
            "reason": "startup-lock-held",
        }

    try:
        from src.Jobs.tasks.task1 import sync_tmalphafold_predictions_task

        task = sync_tmalphafold_predictions_task.delay(
            methods=None,
            with_tmdet=False,
            refresh=False,
            retry_errors=False,
            max_workers=_startup_catchup_max_workers(),
            timeout=30,
            backfill_sequences=False,
            with_tmbed=False,
            tmbed_use_gpu=False,
            tmbed_batch_size=None,
            tmbed_max_workers=None,
            tmbed_refresh=False,
        )
        logger.info(
            "Queued startup TMAlphaFold catch-up task %s because %s.",
            task.id,
            reason,
        )
        return {
            "queued": True,
            "task_id": task.id,
            "reason": reason,
        }
    except Exception as exc:
        _release_startup_lock(lock_context)
        logger.warning("Failed to queue startup TMAlphaFold catch-up: %s", exc)
        return {
            "queued": False,
            "reason": f"queue-failed: {exc}",
        }
