import os
import socket
import uuid
from contextlib import contextmanager
from pathlib import Path

from src.ingestion.exceptions import DatasetRefreshAlreadyRunningError
from src.ingestion.redis_support import get_redis_client


class RefreshLockManager:
    def __init__(self, app_config=None):
        self.redis = get_redis_client(config=app_config)
        self.lock_key = "metamp:locks:protein-dataset-refresh"
        self.lock_ttl_seconds = 2 * 60 * 60 + 600
        self.file_lock_path = Path(".locks") / "protein_dataset_refresh.lock"

    @contextmanager
    def acquire(self):
        token = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4()}"
        if self.redis is not None:
            acquired = self.redis.set(
                self.lock_key,
                token,
                nx=True,
                ex=self.lock_ttl_seconds,
            )
            if not acquired:
                raise DatasetRefreshAlreadyRunningError(
                    "Protein dataset refresh is already running."
                )
            try:
                yield token
            finally:
                current_token = self.redis.get(self.lock_key)
                if current_token == token:
                    self.redis.delete(self.lock_key)
            return

        self.file_lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(
                str(self.file_lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise DatasetRefreshAlreadyRunningError(
                "Protein dataset refresh is already running."
            ) from exc

        try:
            os.write(fd, token.encode("utf-8"))
            os.close(fd)
            yield token
        finally:
            try:
                self.file_lock_path.unlink()
            except FileNotFoundError:
                pass
