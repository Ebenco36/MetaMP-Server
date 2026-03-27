import requests
from celery import shared_task
from celery.utils.log import get_task_logger
from src.Commands.populateData import addDefaultAdmin, addFeedbackQuestions, addQuestion
from src.Jobs.LoadProteinPredictions import (
    run_tm_prediction_backfill,
    run_tm_prediction_for_sequences,
)
from src.Jobs.TMAlphaFoldSync import (
    mirror_local_tm_prediction_rows,
    sync_tmalphafold_predictions,
)
from src.ingestion.exceptions import DatasetRefreshAlreadyRunningError
from src.ingestion.background_refresh_service import ProteinDatasetRefreshService
from src.ingestion.task_status_recorder import TaskStatusRecorder
from src.core.audit_log import MetaMPAuditLogService

logger = get_task_logger(__name__)


TASK_RETRY_OPTIONS = {
    "autoretry_for": (Exception,),
    "retry_backoff": True,
    "retry_jitter": True,
    "retry_kwargs": {"max_retries": 5},
}

DATASET_REFRESH_TASK_RETRY_OPTIONS = {
    "autoretry_for": (requests.exceptions.RequestException, TimeoutError),
    "retry_backoff": 60,
    "retry_backoff_max": 15 * 60,
    "retry_jitter": True,
    "retry_kwargs": {"max_retries": 4},
}


@shared_task(
    name="shared-task-machine-learning-job",
    bind=True,
    queue="ml",
    soft_time_limit=60 * 60,
    time_limit=65 * 60,
    **TASK_RETRY_OPTIONS,
)
def machine_learning_job(self):
    from src.Jobs.MLJobs import MLJob
    from src.Dashboard.services import DiscrepancyBenchmarkExportService

    ml_job = MLJob()
    ml_job.fix_missing_data()\
        .variable_separation()\
        .feature_selection()\
        .dimensionality_reduction()\
        .plot_charts()\
        .semi_supervised_learning()\
        .supervised_learning()\
        .benchmark_and_export_predictions()\
        .time_split_evaluation()
    try:
        DiscrepancyBenchmarkExportService.export_release(include_all=False)
    except Exception as exc:
        logger.warning(
            "Machine learning job completed, but discrepancy benchmark export failed: %s",
            exc,
        )
    MetaMPAuditLogService.record_event(
        "machine_learning_job_completed",
        {"task_id": self.request.id, "task_name": self.name},
    )
    logger.info("Machine learning job completed successfully.")


@shared_task(
    name="shared-task-sync-tmalphafold-predictions",
    bind=True,
    queue="tm",
    soft_time_limit=12 * 60 * 60,
    time_limit=12 * 60 * 60 + 300,
    **TASK_RETRY_OPTIONS,
)
def sync_tmalphafold_predictions_task(
    self,
    methods=None,
    with_tmdet=True,
    refresh=True,
    retry_errors=False,
    max_workers=8,
    timeout=30,
    backfill_sequences=True,
    with_tmbed=True,
    tmbed_use_gpu=False,
    tmbed_batch_size=None,
    tmbed_max_workers=None,
    tmbed_refresh=False,
):
    recorder = TaskStatusRecorder()

    def progress(message):
        recorder.record_progress(self.name, self.request.id, message)
        logger.info(message)

    recorder.record_started(
        self.name,
        self.request.id,
        extra={
            "methods": methods,
            "with_tmdet": with_tmdet,
            "refresh": refresh,
            "retry_errors": retry_errors,
            "max_workers": max_workers,
            "timeout": timeout,
            "backfill_sequences": backfill_sequences,
            "with_tmbed": with_tmbed,
            "tmbed_use_gpu": tmbed_use_gpu,
            "tmbed_batch_size": tmbed_batch_size,
            "tmbed_max_workers": tmbed_max_workers,
            "tmbed_refresh": tmbed_refresh,
        },
    )

    summary = {
        "tmalphafold": sync_tmalphafold_predictions(
            methods=methods,
            with_tmdet=with_tmdet,
            pdb_codes=None,
            limit=None,
            refresh=refresh,
            retry_errors=retry_errors,
            max_workers=max_workers,
            timeout=timeout,
            backfill_sequences=backfill_sequences,
            progress_callback=progress,
        ),
        "tmbed": None,
    }

    if with_tmbed:
        progress("Starting appended TMbed refresh...")
        summary["tmbed"] = run_tm_prediction_backfill(
            include_tmbed=True,
            include_deeptmhmm=False,
            use_gpu=tmbed_use_gpu,
            batch_size=tmbed_batch_size,
            max_workers=tmbed_max_workers,
            include_completed=tmbed_refresh,
            progress_callback=progress,
        )
        tmbed_records = (summary["tmbed"] or {}).get("records") or []
        if tmbed_records:
            summary["tmbed"]["normalized_store_verification"] = mirror_local_tm_prediction_rows(
                method="TMbed",
                records=tmbed_records,
                provider="MetaMP",
                prediction_kind="sequence_topology",
                progress_callback=progress,
            )
        else:
            progress(
                "Appended TMbed refresh completed without record previews; no extra TMbed normalized-store verification rows were written."
            )

    recorder.record_succeeded(self.name, self.request.id, extra=summary)
    MetaMPAuditLogService.record_event(
        "tmalphafold_sync_completed",
        {
            "task_id": self.request.id,
            "task_name": self.name,
            "summary": summary,
        },
    )
    logger.info("TMAlphaFold sync completed successfully.")
    return summary


@shared_task(
    name="shared-task-sync-tmbed-predictions",
    bind=True,
    queue="tm",
    soft_time_limit=12 * 60 * 60,
    time_limit=12 * 60 * 60 + 300,
    **TASK_RETRY_OPTIONS,
)
def sync_tmbed_predictions_task(
    self,
    pdb_codes=None,
    limit=None,
    use_gpu=False,
    batch_size=None,
    max_workers=None,
    refresh=False,
):
    recorder = TaskStatusRecorder()

    def progress(message):
        recorder.record_progress(self.name, self.request.id, message)
        logger.info(message)

    recorder.record_started(
        self.name,
        self.request.id,
        extra={
            "pdb_codes": pdb_codes,
            "limit": limit,
            "use_gpu": use_gpu,
            "batch_size": batch_size,
            "max_workers": max_workers,
            "refresh": refresh,
        },
    )

    summary = run_tm_prediction_backfill(
        include_tmbed=True,
        include_deeptmhmm=False,
        use_gpu=use_gpu,
        batch_size=batch_size,
        max_workers=max_workers,
        include_completed=refresh,
        pdb_codes=pdb_codes,
        limit=limit,
        progress_callback=progress,
    )
    tmbed_records = (summary or {}).get("records") or []
    if tmbed_records:
        summary["normalized_store_verification"] = mirror_local_tm_prediction_rows(
            method="TMbed",
            records=tmbed_records,
            provider="MetaMP",
            prediction_kind="sequence_topology",
            progress_callback=progress,
        )

    recorder.record_succeeded(self.name, self.request.id, extra=summary)
    MetaMPAuditLogService.record_event(
        "tmbed_sync_completed",
        {
            "task_id": self.request.id,
            "task_name": self.name,
            "summary": summary,
        },
    )
    logger.info("TMbed-only sync completed successfully.")
    return summary


@shared_task(
    name="shared-task-predict-tm-sequences",
    bind=True,
    queue="ml",
    soft_time_limit=30 * 60,
    time_limit=35 * 60,
    **TASK_RETRY_OPTIONS,
)
def predict_tm_sequences(
    self,
    records,
    include_deeptmhmm=None,
    use_gpu=None,
    max_workers=None,
):
    def progress(message):
        logger.info(message)

    result = run_tm_prediction_for_sequences(
        records,
        include_deeptmhmm=include_deeptmhmm,
        use_gpu=use_gpu,
        max_workers=max_workers,
        progress_callback=progress,
    )
    logger.info("TM sequence prediction completed successfully.")
    return result


@shared_task(
    name="background-refresh-protein-datasets",
    bind=True,
    soft_time_limit=2 * 60 * 60,
    time_limit=2 * 60 * 60 + 300,
    **DATASET_REFRESH_TASK_RETRY_OPTIONS,
)
def refresh_protein_datasets(self):
    try:
        ProteinDatasetRefreshService().run(
            sync_database=True,
            task_name=self.name,
            task_id=self.request.id,
        )
        logger.info("Protein dataset refresh completed successfully.")
    except DatasetRefreshAlreadyRunningError:
        logger.warning("Skipped refresh because another run is already active.")


@shared_task(
    name="shared-task-sync-protein-database",
    bind=True,
    soft_time_limit=2 * 60 * 60,
    time_limit=2 * 60 * 60 + 300,
    **DATASET_REFRESH_TASK_RETRY_OPTIONS,
)
def scheduled_sync_command(self):
    try:
        ProteinDatasetRefreshService().run(
            sync_database=True,
            task_name=self.name,
            task_id=self.request.id,
        )
        logger.info("Protein dataset refresh completed successfully.")
    except DatasetRefreshAlreadyRunningError:
        logger.warning("Skipped refresh because another run is already active.")


@shared_task(name="shared-task-sync-system_admin-with-database", bind=True, **TASK_RETRY_OPTIONS)
def system_admin_scheduled_sync_command(self):
    addDefaultAdmin()
    logger.info("System admin sync completed successfully.")


@shared_task(name="shared-task-sync-feedback-questions-with-database", bind=True, **TASK_RETRY_OPTIONS)
def feedback_scheduled_sync_command(self):
    addFeedbackQuestions()
    logger.info("Feedback questions sync completed successfully.")


@shared_task(name="shared-task-sync-question-with-database", bind=True, **TASK_RETRY_OPTIONS)
def question_feedback_scheduled_sync_command(self):
    addQuestion()
    logger.info("Question sync completed successfully.")


@shared_task(
    name="shared-task-monthly-production-maintenance",
    bind=True,
    soft_time_limit=3 * 60 * 60,
    time_limit=3 * 60 * 60 + 300,
    **DATASET_REFRESH_TASK_RETRY_OPTIONS,
)
def monthly_production_maintenance(
    self,
    use_gpu=None,
    batch_size=None,
    max_workers=None,
    export_benchmark=True,
    validate_regressions=True,
):
    recorder = TaskStatusRecorder()

    def progress(message, extra=None):
        recorder.record_progress(
            self.name,
            self.request.id,
            message,
            extra=extra,
        )
        logger.info(message)

    recorder.record_started(
        self.name,
        self.request.id,
        extra={
            "use_gpu": use_gpu,
            "batch_size": batch_size,
            "max_workers": max_workers,
            "export_benchmark": export_benchmark,
            "validate_regressions": validate_regressions,
        },
    )

    try:
        progress("Refreshing datasets and synchronizing the protein database.")
        ProteinDatasetRefreshService().run(
            sync_database=True,
            seed_defaults=True,
            task_name=self.name,
            task_id=self.request.id,
        )

        progress("Queueing machine learning training task.")
        ml_task = machine_learning_job.delay()

        progress("Queueing normalized TM annotation sync task.")
        tm_task = sync_tmalphafold_predictions_task.delay(
            methods=None,
            with_tmdet=True,
            refresh=True,
            max_workers=max_workers or 8,
            timeout=30,
            backfill_sequences=True,
            with_tmbed=True,
            tmbed_use_gpu=use_gpu,
            tmbed_batch_size=batch_size,
            tmbed_max_workers=max_workers,
            tmbed_refresh=True,
        )

        benchmark_summary = None
        validation_summary = None

        if export_benchmark:
            from src.Dashboard.services import DiscrepancyBenchmarkExportService

            progress("Exporting the discrepancy benchmark release.")
            benchmark_summary = DiscrepancyBenchmarkExportService.export_release(
                include_all=False
            )

        if validate_regressions:
            from src.Dashboard.services import DashboardRegressionValidationService

            progress("Running dashboard regression validation.")
            validation_summary = DashboardRegressionValidationService.run_checks()
            if not validation_summary.get("passed"):
                raise RuntimeError("Dashboard regression validation failed.")

        summary = {
            "ml_task_id": ml_task.id,
            "tmalphafold_task_id": tm_task.id,
            "tm_annotation_task_id": tm_task.id,
            "benchmark_exported": bool(benchmark_summary),
            "validation_passed": (
                validation_summary.get("passed")
                if validation_summary is not None
                else None
            ),
        }
        if benchmark_summary:
            summary["benchmark_metadata"] = benchmark_summary.get("metadata")
            summary["benchmark_csv_path"] = benchmark_summary.get("csv_path")
            summary["benchmark_json_path"] = benchmark_summary.get("json_path")

        recorder.record_succeeded(self.name, self.request.id, extra=summary)
        MetaMPAuditLogService.record_event(
            "monthly_production_maintenance_completed",
            {
                "task_id": self.request.id,
                "task_name": self.name,
                "summary": summary,
            },
        )
        logger.info("Monthly production maintenance completed successfully.")
        return summary
    except Exception as exc:
        recorder.record_failed(self.name, self.request.id, str(exc))
        raise
