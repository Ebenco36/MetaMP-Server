from flask import current_app
from time import monotonic

from src.ingestion.database_load_service import DatabaseLoadService
from src.ingestion import ProteinDatabaseIngestionPipeline
from src.ingestion.exceptions import DatasetRefreshAlreadyRunningError
from src.ingestion.refresh_lock import RefreshLockManager
from src.ingestion.settings import IngestionSettings
from src.ingestion.task_status_recorder import TaskStatusRecorder


class ProteinDatasetRefreshService:
    def __init__(self, app_config=None):
        self.app_config = app_config or current_app.config
        self.settings = IngestionSettings.from_environment()
        self.lock_manager = RefreshLockManager(app_config=self.app_config)
        self.status_recorder = TaskStatusRecorder(app_config=self.app_config)
        self.database_loader = DatabaseLoadService()

    def refresh_artifacts(self, stage_names=None, clean_valid=None, progress_callback=None):
        return ProteinDatabaseIngestionPipeline(settings=self.settings).run(
            clean_valid=clean_valid,
            stage_names=tuple(stage_names) if stage_names else None,
            progress_callback=progress_callback,
        )

    @staticmethod
    def _extract_stage_name(message):
        if message.startswith("Starting stage '") or message.startswith("Completed stage '"):
            parts = message.split("'")
            if len(parts) >= 2:
                return parts[1]

        if message.startswith("[") and "]" in message:
            return message[1 : message.index("]")]

        return None

    def _build_progress_callback(self, task_name=None, task_id=None, progress_callback=None):
        if not task_name or not task_id:
            return progress_callback

        run_started_at = monotonic()
        current_stage = {"name": None, "started_at": run_started_at}

        def callback(message):
            stage_name = self._extract_stage_name(message) or current_stage["name"]
            if stage_name and stage_name != current_stage["name"]:
                current_stage["name"] = stage_name
                current_stage["started_at"] = monotonic()

            duration_seconds = round(monotonic() - run_started_at, 1)
            stage_duration_seconds = round(
                monotonic() - current_stage["started_at"],
                1,
            )

            self.status_recorder.record_progress(
                task_name,
                task_id,
                message,
                stage_name=stage_name,
                extra={
                    "duration_seconds": duration_seconds,
                    "current_stage_duration_seconds": stage_duration_seconds,
                },
            )

            if progress_callback:
                progress_callback(message)

        return callback

    def sync_database_records(self, clear_db=False, seed_defaults=False):
        return self.database_loader.load_current_datasets(
            clear_db=clear_db,
            seed_defaults=seed_defaults,
        )

    def run(
        self,
        sync_database=True,
        clear_db=False,
        seed_defaults=False,
        task_name=None,
        task_id=None,
        stage_names=None,
        clean_valid=None,
        progress_callback=None,
    ):
        run_started_at = monotonic()
        try:
            with self.lock_manager.acquire():
                if task_name and task_id:
                    self.status_recorder.record_started(task_name, task_id)

                effective_progress_callback = self._build_progress_callback(
                    task_name=task_name,
                    task_id=task_id,
                    progress_callback=progress_callback,
                )

                context = self.refresh_artifacts(
                    stage_names=stage_names,
                    clean_valid=clean_valid,
                    progress_callback=effective_progress_callback,
                )
                if sync_database:
                    self.sync_database_records(
                        clear_db=clear_db,
                        seed_defaults=seed_defaults,
                    )

                if task_name and task_id:
                    self.status_recorder.record_succeeded(
                        task_name,
                        task_id,
                        extra={
                            "run_date": context.run_date,
                            "duration_seconds": round(monotonic() - run_started_at, 1),
                        },
                    )

                return context
        except DatasetRefreshAlreadyRunningError as error:
            if task_name and task_id:
                self.status_recorder.record_skipped(
                    task_name,
                    task_id,
                    str(error),
                    extra={"duration_seconds": round(monotonic() - run_started_at, 1)},
                )
            raise
        except Exception as error:
            if task_name and task_id:
                self.status_recorder.record_failed(
                    task_name,
                    task_id,
                    str(error),
                    extra={"duration_seconds": round(monotonic() - run_started_at, 1)},
                )
            raise
