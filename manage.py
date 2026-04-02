import click
import json
from pathlib import Path
from importlib.util import find_spec
import pandas as pd
from celery.result import AsyncResult
from flask.cli import FlaskGroup
from flask_migrate import Migrate
from sqlalchemy import text

from app import app, create_app
from database.db import db
from src.Commands.populateData import (
    addDefaultAdmin,
    addFeedbackQuestions,
    addQuestion,
)
from src.core.celery_factory import celery
from src.ingestion.background_refresh_service import ProteinDatasetRefreshService
from src.ingestion.dataset_layout import DatasetLayout
from src.ingestion.database_load_service import DatabaseLoadService
from src.ingestion.settings import DEFAULT_STAGE_ORDER
from src.ingestion.settings import IngestionSettings
from src.ingestion.schema_sync_service import SchemaSyncService
from src.ingestion.task_status_recorder import TaskStatusRecorder
from src.ingestion.validation_service import IngestionValidationService
from src.Dashboard.services import (
    DashboardRegressionValidationService,
    DiscrepancyBenchmarkExportService,
    DashboardAnnotationDatasetService,
)
from src.Jobs.tasks.task1 import (
    machine_learning_job,
    monthly_production_maintenance,
    scheduled_sync_command,
    sync_tmbed_predictions_task,
    sync_tmalphafold_predictions_task,
)
from src.Jobs.LoadProteinPredictions import (
    OPTIONAL_TM_PREDICTORS,
    VERIFIED_LOCAL_TM_FALLBACK_PREDICTORS,
    determine_tmalphafold_fallback_targets,
    export_optional_tm_prediction_inputs_bulk,
    export_optional_tm_prediction_inputs,
    get_optional_tm_runtime_status,
    import_optional_tm_prediction_results_bulk,
    import_optional_tm_prediction_results,
    normalize_optional_tm_predictor_names,
    normalize_verified_tm_fallback_predictor_names,
    run_optional_tm_prediction_backfill,
    run_verified_tm_fallback_pipeline,
    run_tm_prediction_backfill,
)
from src.Jobs.TMAlphaFoldSync import (
    TMALPHAFOLD_AUX_METHODS,
    TMALPHAFOLD_SEQUENCE_METHODS,
    mirror_local_tm_prediction_rows,
    sync_tmalphafold_predictions,
)

migrate = Migrate(app, db)
cli = FlaskGroup(create_app=create_app)
LEGACY_TM_PREDICTOR_COLUMNS = [
    "TMbed_tm_count",
    "DeepTMHMM_tm_count",
    "Phobius_tm_count",
    "TOPCONS_tm_count",
    "CCTOP_tm_count",
    "TMbed_tm_regions",
    "DeepTMHMM_tm_regions",
    "Phobius_tm_regions",
    "TOPCONS_tm_regions",
    "CCTOP_tm_regions",
]


def _ensure_ml_runtime(command_name, package_name):
    if find_spec(package_name) is not None:
        return

    raise click.ClickException(
        f"{command_name} requires the ML runtime package '{package_name}', which is not installed in this container. "
        "Run the command in the ML container instead:\n"
        f"docker compose --env-file .env.docker.deployment exec -T celery-worker-ml env FLASK_APP=manage.py flask {command_name}\n\n"
        "If you want native macOS MPS, Linux CUDA, or CPU execution outside Docker, use:\n"
        "bash scripts/metamp-native-tmbed.sh doctor\n"
        "bash scripts/metamp-native-tmbed.sh sync --all\n"
        "bash scripts/metamp-native-tmbed.sh fallback --all"
    )


def _normalize_cli_pdb_codes(pdb_codes):
    normalized = []
    for value in pdb_codes or ():
        text = str(value or "").strip().upper()
        if text:
            normalized.append(text)
    return list(dict.fromkeys(normalized))


@app.cli.command("refresh-protein-datasets")
@click.option(
    "--stages",
    default=",".join(DEFAULT_STAGE_ORDER),
    help="Comma-separated ingestion stages to run.",
)
@click.option(
    "--clean-valid/--skip-clean-valid",
    default=True,
    help="Write cleaned datasets into datasets/valid after ingestion.",
)
def refresh_protein_datasets(stages, clean_valid):
    click.echo("Refreshing protein dataset artifacts...")
    stage_names = tuple(
        stage.strip().lower() for stage in stages.split(",") if stage.strip()
    )
    ProteinDatasetRefreshService().refresh_artifacts(
        stage_names=stage_names,
        clean_valid=clean_valid,
        progress_callback=click.echo,
    )
    click.echo("Protein dataset artifacts refreshed.")


@app.cli.command("sync-protein-schema")
def sync_protein_schema():
    click.echo("Syncing database schema...")
    SchemaSyncService().sync()
    click.echo("Protein schema synced.")


@app.cli.command("export-tmalphafold-predictions-csv")
@click.option(
    "--output-path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="CSV path to write membrane_protein_tmalphafold_predictions into.",
)
@click.option(
    "--provider",
    "providers",
    multiple=True,
    help="Optional provider filter. Repeat the option to keep multiple providers.",
)
@click.option(
    "--method",
    "methods",
    multiple=True,
    help="Optional method filter. Repeat the option to keep multiple methods.",
)
def export_tmalphafold_predictions_csv(output_path, providers, methods):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query = """
        SELECT *
        FROM membrane_protein_tmalphafold_predictions
    """
    filters = []
    params = {}

    normalized_providers = [str(value).strip() for value in providers if str(value).strip()]
    normalized_methods = [str(value).strip() for value in methods if str(value).strip()]

    if normalized_providers:
        provider_placeholders = []
        for index, value in enumerate(normalized_providers):
            key = f"provider_{index}"
            params[key] = value
            provider_placeholders.append(f":{key}")
        filters.append("provider IN (" + ", ".join(provider_placeholders) + ")")

    if normalized_methods:
        method_placeholders = []
        for index, value in enumerate(normalized_methods):
            key = f"method_{index}"
            params[key] = value
            method_placeholders.append(f":{key}")
        filters.append("method IN (" + ", ".join(method_placeholders) + ")")

    if filters:
        query += " WHERE " + " AND ".join(filters)
    query += " ORDER BY provider, method, pdb_code, uniprot_id"

    frame = pd.read_sql_query(text(query), db.engine, params=params)
    frame.to_csv(output_path, index=False)
    click.echo(
        json.dumps(
            {
                "output_path": str(output_path),
                "row_count": int(len(frame)),
                "providers": normalized_providers,
                "methods": normalized_methods,
            },
            indent=2,
        )
    )


@app.cli.command("drop-legacy-tm-predictor-columns")
def drop_legacy_tm_predictor_columns():
    click.echo("Dropping legacy TM predictor columns from membrane_proteins...")
    with db.engine.begin() as connection:
        for column_name in LEGACY_TM_PREDICTOR_COLUMNS:
            connection.execute(
                text(f'ALTER TABLE membrane_proteins DROP COLUMN IF EXISTS "{column_name}"')
            )
    DashboardAnnotationDatasetService._record_payload_cache.clear()
    click.echo(
        json.dumps(
            {
                "table_name": "membrane_proteins",
                "dropped_columns": LEGACY_TM_PREDICTOR_COLUMNS,
            },
            indent=2,
        )
    )


@app.cli.command("load-protein-datasets")
@click.option("--clear-db", is_flag=True, help="Drop existing tables before loading data.")
@click.option(
    "--sync-schema/--skip-sync-schema",
    default=True,
    help="Sync database schema before loading dataset records.",
)
@click.option(
    "--seed-defaults/--no-seed-defaults",
    default=True,
    help="Seed default questions and admin records after loading.",
)
def load_protein_datasets(clear_db, sync_schema, seed_defaults):
    if sync_schema:
        click.echo("Syncing protein schema...")
        SchemaSyncService().sync()

    click.echo("Loading protein datasets into the database...")
    DatabaseLoadService().load_current_datasets(
        clear_db=clear_db,
        seed_defaults=seed_defaults,
    )
    click.echo("Protein datasets loaded into the database.")


@app.cli.command("sync-protein-database")
@click.option("--clear-db", default="n", help="Do you really want to clear the DB")
@click.option(
    "--skip-ingestion",
    is_flag=True,
    help="Skip dataset fetching and reuse the current CSV artifacts.",
)
@click.option(
    "--skip-schema",
    is_flag=True,
    help="Skip database migration.",
)
@click.option(
    "--stages",
    default=",".join(DEFAULT_STAGE_ORDER),
    help="Comma-separated ingestion stages to run when ingestion is enabled.",
)
@click.option(
    "--clean-valid/--skip-clean-valid",
    default=True,
    help="Write cleaned datasets into datasets/valid after ingestion.",
)
def sync_protein_database(clear_db, skip_ingestion, skip_schema, stages, clean_valid):
    if not skip_ingestion:
        click.echo("Refreshing dataset artifacts...")
        stage_names = tuple(
            stage.strip().lower() for stage in stages.split(",") if stage.strip()
        )
        ProteinDatasetRefreshService().refresh_artifacts(
            stage_names=stage_names,
            clean_valid=clean_valid,
            progress_callback=click.echo,
        )

    if not skip_schema:
        click.echo("Syncing schema...")
        SchemaSyncService().sync()

    click.echo("Loading datasets into the database...")
    DatabaseLoadService().load_current_datasets(
        clear_db=(clear_db == "y"),
        seed_defaults=True,
    )
    click.echo("Protein database sync completed.")


@app.cli.command("validate-protein-datasets")
def validate_protein_datasets():
    settings = IngestionSettings.from_environment()
    layout = DatasetLayout(base_dir=settings.dataset_base_dir)
    report = IngestionValidationService().build_report(layout)
    click.echo(json.dumps(report, indent=2))
    if not report.get("passed"):
        raise SystemExit(1)


@app.cli.command("protein-refresh-status")
def protein_refresh_status():
    status = TaskStatusRecorder(app.config).read_latest(
        "background-refresh-protein-datasets"
    )
    if status is None:
        click.echo("No background protein refresh status found.")
        return

    click.echo(json.dumps(status, indent=2))


@app.cli.command("sync-tmalphafold-predictions")
@click.option(
    "--all",
    "run_all",
    is_flag=True,
    help="Run across all eligible UniProt-backed records.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Restrict the sync to the first N eligible targets.",
)
@click.option(
    "--pdb-code",
    "pdb_codes",
    multiple=True,
    help="Restrict the sync to one or more specific PDB codes. Repeat the option to provide multiple codes.",
)
@click.option(
    "--methods",
    type=str,
    default=",".join(TMALPHAFOLD_SEQUENCE_METHODS + TMALPHAFOLD_AUX_METHODS),
    show_default=False,
    help="Comma-separated TMAlphaFold method list.",
)
@click.option(
    "--with-tmdet/--without-tmdet",
    default=True,
    help="Include TMDET structure-based membrane-plane predictions.",
)
@click.option(
    "--max-workers",
    type=int,
    default=8,
    show_default=True,
    help="Concurrent TMAlphaFold request count.",
)
@click.option(
    "--timeout",
    type=int,
    default=30,
    show_default=True,
    help="Per-request timeout in seconds.",
)
@click.option(
    "--refresh",
    is_flag=True,
    help="Refetch predictions even when successful TMAlphaFold records already exist.",
)
@click.option(
    "--retry-errors/--skip-errors",
    default=False,
    help="When not refreshing, retry previously stored error rows instead of skipping all existing rows.",
)
@click.option(
    "--backfill-sequences/--skip-backfill-sequences",
    default=True,
    help="Backfill blank sequence_sequence values from TMAlphaFold payloads when available.",
)
@click.option(
    "--with-tmbed/--without-tmbed",
    default=True,
    help="Also run the local TMbed backfill after TMAlphaFold sync completes.",
)
@click.option(
    "--tmbed-use-gpu",
    is_flag=True,
    help="Allow the appended TMbed run to use GPU if available.",
)
@click.option(
    "--tmbed-batch-size",
    type=int,
    default=None,
    help="Override batch size for the appended TMbed run.",
)
@click.option(
    "--tmbed-max-workers",
    type=int,
    default=None,
    help="Override predictor parallelism for the appended TMbed run.",
)
@click.option(
    "--tmbed-csv-out",
    type=str,
    default=None,
    help="Optional resumable CSV path for the appended TMbed run.",
)
@click.option(
    "--tmbed-refresh/--skip-tmbed-refresh",
    default=False,
    help="Rerun TMbed even for records that already have TMbed values.",
)
def run_tmalphafold_now(
    run_all,
    limit,
    pdb_codes,
    methods,
    with_tmdet,
    max_workers,
    timeout,
    refresh,
    retry_errors,
    backfill_sequences,
    with_tmbed,
    tmbed_use_gpu,
    tmbed_batch_size,
    tmbed_max_workers,
    tmbed_csv_out,
    tmbed_refresh,
):
    selected_codes = _normalize_cli_pdb_codes(pdb_codes)
    if not run_all and not selected_codes and limit is None:
        raise click.ClickException(
            "Provide --all, --limit, or at least one --pdb-code so the TMAlphaFold sync scope is explicit."
        )

    method_list = [item.strip() for item in str(methods or "").split(",") if item.strip()]
    summary = {
        "tmalphafold": sync_tmalphafold_predictions(
        methods=method_list,
        with_tmdet=with_tmdet,
        pdb_codes=selected_codes,
        limit=limit,
        refresh=refresh,
        retry_errors=retry_errors,
        max_workers=max_workers,
        timeout=timeout,
        backfill_sequences=backfill_sequences,
        progress_callback=click.echo,
        ),
        "tmbed": None,
    }
    if with_tmbed:
        _ensure_ml_runtime("sync-tmalphafold-predictions --with-tmbed", "tmbed")
        click.echo("Running appended TMbed backfill after TMAlphaFold sync...")
        summary["tmbed"] = run_tm_prediction_backfill(
            include_tmbed=True,
            include_deeptmhmm=False,
            use_gpu=tmbed_use_gpu,
            batch_size=tmbed_batch_size,
            max_workers=tmbed_max_workers,
            csv_out=tmbed_csv_out,
            include_completed=tmbed_refresh,
            pdb_codes=selected_codes,
            limit=limit,
            progress_callback=click.echo,
        )
        tmbed_records = (summary["tmbed"] or {}).get("records") or []
        if tmbed_records:
            summary["tmbed"]["normalized_store_verification"] = mirror_local_tm_prediction_rows(
                method="TMbed",
                records=tmbed_records,
                provider="MetaMP",
                prediction_kind="sequence_topology",
                progress_callback=click.echo,
            )
    DashboardAnnotationDatasetService._record_payload_cache.clear()
    click.echo(json.dumps(summary, indent=2))

@app.cli.command("tm-prediction-status")
def tm_prediction_status():
    status = TaskStatusRecorder(app.config).read_latest(
        "shared-task-sync-tmalphafold-predictions"
    )
    if status is None:
        click.echo("No TM annotation sync status found.")
        return

    click.echo(json.dumps(status, indent=2))


@app.cli.command("sync-tmbed-predictions")
@click.option(
    "--all",
    "run_all",
    is_flag=True,
    help="Run across all eligible protein records.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Restrict the sync to the first N eligible targets.",
)
@click.option(
    "--pdb-code",
    "pdb_codes",
    multiple=True,
    help="Restrict the sync to one or more specific PDB codes. Repeat the option to provide multiple codes.",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    help="Allow TMbed to use GPU if available.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Override batch size for TMbed.",
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="Override predictor parallelism for TMbed.",
)
@click.option(
    "--csv-out",
    type=str,
    default=None,
    help="Optional resumable CSV path for the TMbed run.",
)
@click.option(
    "--refresh/--skip-refresh",
    default=False,
    help="Rerun TMbed even for records that already have TMbed values.",
)
def sync_tmbed_predictions_cli(
    run_all,
    limit,
    pdb_codes,
    use_gpu,
    batch_size,
    max_workers,
    csv_out,
    refresh,
):
    selected_codes = _normalize_cli_pdb_codes(pdb_codes)
    if not run_all and not selected_codes and limit is None:
        raise click.ClickException(
            "Provide --all, --limit, or at least one --pdb-code so the TMbed sync scope is explicit."
        )

    _ensure_ml_runtime("sync-tmbed-predictions", "tmbed")
    summary = run_tm_prediction_backfill(
        include_tmbed=True,
        include_deeptmhmm=False,
        use_gpu=use_gpu,
        batch_size=batch_size,
        max_workers=max_workers,
        csv_out=csv_out,
        include_completed=refresh,
        pdb_codes=selected_codes,
        limit=limit,
        progress_callback=click.echo,
    )
    tmbed_records = (summary or {}).get("records") or []
    if tmbed_records:
        summary["normalized_store_verification"] = mirror_local_tm_prediction_rows(
            method="TMbed",
            records=tmbed_records,
            provider="MetaMP",
            prediction_kind="sequence_topology",
            progress_callback=click.echo,
        )
    DashboardAnnotationDatasetService._record_payload_cache.clear()
    click.echo(json.dumps(summary, indent=2))


@app.cli.command("export-optional-tm-prediction-inputs")
@click.option(
    "--predictor",
    required=True,
    type=str,
    multiple=True,
    help=(
        "Optional fallback predictor to prepare for import. "
        f"Supported values include: {', '.join(OPTIONAL_TM_PREDICTORS)}. "
        "Repeat the option to export multiple predictors, or use '--predictor all'."
    ),
)
@click.option(
    "--include-completed",
    is_flag=True,
    help="Also export records that already have MetaMP normalized rows for this predictor.",
)
@click.option(
    "--completion-provider",
    type=click.Choice(["MetaMP", "any", "TMAlphaFold"], case_sensitive=False),
    default="MetaMP",
    show_default=True,
    help=(
        "Which provider scope counts as already completed. "
        "Use MetaMP to prepare fallback work owned by MetaMP even when upstream TMAlphaFold rows already exist."
    ),
)
@click.option(
    "--pdb-code",
    "pdb_codes",
    multiple=True,
    help="Restrict the export to one or more specific PDB codes. Repeat the option to provide multiple codes.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Restrict the export to the first N eligible targets.",
)
@click.option(
    "--fasta-out",
    type=str,
    default=None,
    help="Optional FASTA output path. Defaults to the predictor-specific runtime directory.",
)
@click.option(
    "--csv-out",
    type=str,
    default=None,
    help="Optional CSV template output path. Defaults to the predictor-specific runtime directory.",
)
def export_optional_tm_prediction_inputs_cli(
    predictor,
    include_completed,
    completion_provider,
    pdb_codes,
    limit,
    fasta_out,
    csv_out,
):
    normalized_predictors = normalize_optional_tm_predictor_names(predictor)
    if len(normalized_predictors) == 1 and str(next(iter(predictor))).strip().lower() not in {"all", "*"}:
        summary = export_optional_tm_prediction_inputs(
            predictor_name=normalized_predictors[0],
            fasta_out=fasta_out,
            csv_out=csv_out,
            include_completed=include_completed,
            completion_provider=completion_provider,
            pdb_codes=_normalize_cli_pdb_codes(pdb_codes),
            limit=limit,
            progress_callback=click.echo,
        )
    else:
        if fasta_out or csv_out:
            raise click.ClickException(
                "--fasta-out and --csv-out are only supported for single-predictor export. "
                "For '--predictor all' or repeated --predictor values, MetaMP writes each predictor "
                "to its own runtime directory under /var/app/data/tm_predictions/external/<predictor>/."
            )
        summary = export_optional_tm_prediction_inputs_bulk(
            predictor_names=normalized_predictors,
            include_completed=include_completed,
            completion_provider=completion_provider,
            pdb_codes=_normalize_cli_pdb_codes(pdb_codes),
            limit=limit,
            progress_callback=click.echo,
        )
    click.echo(json.dumps(summary, indent=2))


@app.cli.command("run-optional-tm-predictions")
@click.option(
    "--predictor",
    required=True,
    type=str,
    multiple=True,
    help=(
        "Optional predictor(s) to execute. Only predictors with a local MetaMP runtime "
        "will be executed here. Repeat the option or use '--predictor all'."
    ),
)
@click.option(
    "--include-completed/--skip-completed",
    default=False,
    help="Rerun predictors even when MetaMP already stores successful rows.",
)
@click.option(
    "--pdb-code",
    "pdb_codes",
    multiple=True,
    help="Restrict execution to one or more specific PDB codes.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Restrict execution to the first N eligible targets.",
)
@click.option(
    "--use-gpu/--no-gpu",
    default=None,
    help="Override GPU usage for locally runnable predictors.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Optional batch size override for locally runnable predictors.",
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="Optional worker-count override for locally runnable predictors.",
)
def run_optional_tm_predictions_cli(
    predictor,
    include_completed,
    pdb_codes,
    limit,
    use_gpu,
    batch_size,
    max_workers,
):
    summary = run_optional_tm_prediction_backfill(
        predictor_names=normalize_optional_tm_predictor_names(predictor),
        include_completed=include_completed,
        pdb_codes=_normalize_cli_pdb_codes(pdb_codes),
        limit=limit,
        use_gpu=use_gpu,
        batch_size=batch_size,
        max_workers=max_workers,
        progress_callback=click.echo,
    )
    DashboardAnnotationDatasetService._record_payload_cache.clear()
    click.echo(json.dumps(summary, indent=2))


@app.cli.command("run-verified-tm-fallbacks")
@click.option(
    "--mode",
    "fallback_mode",
    type=click.Choice(["tmalphafold_first", "fallback_only"], case_sensitive=False),
    default="tmalphafold_first",
    show_default=True,
    help=(
        "Run TMAlphaFold first and only then local fallbacks for upstream misses, "
        "or skip TMAlphaFold and run the verified local fallbacks directly."
    ),
)
@click.option(
    "--all",
    "run_all",
    is_flag=True,
    help="Run across all eligible targets in the selected mode.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Restrict the run to the first N eligible targets.",
)
@click.option(
    "--pdb-code",
    "pdb_codes",
    multiple=True,
    help="Restrict the run to one or more specific PDB codes.",
)
@click.option(
    "--fallback-method",
    "fallback_methods",
    multiple=True,
    type=str,
    help=(
        "Verified local fallback method(s) to use. "
        "Repeat to restrict the run to specific local fallback methods. "
        f"When omitted, the full verified set is used: {', '.join(VERIFIED_LOCAL_TM_FALLBACK_PREDICTORS)}."
    ),
)
@click.option(
    "--all-verified-methods/--bulk-safe-only",
    "all_verified_methods",
    default=True,
    show_default=True,
    help=(
        "In bulk scope, run all verified local fallback methods or restrict the run to the "
        "bulk-safe subset (DeepTMHMM, TMHMM)."
    ),
)
@click.option(
    "--include-completed/--skip-completed",
    default=False,
    help="Rerun local fallback methods even when MetaMP already stores successful rows.",
)
@click.option(
    "--retry-errors/--skip-errors",
    default=False,
    help="Retry previously stored MetaMP fallback error rows without forcing successful rows to rerun.",
)
@click.option(
    "--methods",
    type=str,
    default=",".join(TMALPHAFOLD_SEQUENCE_METHODS + TMALPHAFOLD_AUX_METHODS),
    show_default=False,
    help="Comma-separated TMAlphaFold method list for tmalphafold_first mode.",
)
@click.option(
    "--with-tmdet/--without-tmdet",
    default=True,
    help="Include TMDET in the TMAlphaFold stage when using tmalphafold_first mode.",
)
@click.option(
    "--tmalphafold-max-workers",
    type=int,
    default=8,
    show_default=True,
    help="Concurrent TMAlphaFold request count in tmalphafold_first mode.",
)
@click.option(
    "--tmalphafold-timeout",
    type=int,
    default=30,
    show_default=True,
    help="Per-request timeout for TMAlphaFold in seconds.",
)
@click.option(
    "--tmalphafold-refresh",
    is_flag=True,
    help="Refetch TMAlphaFold predictions even when successful upstream rows already exist.",
)
@click.option(
    "--tmalphafold-retry-errors/--skip-tmalphafold-errors",
    default=False,
    help="Retry previously stored TMAlphaFold error rows instead of skipping all existing upstream rows.",
)
@click.option(
    "--backfill-sequences/--skip-backfill-sequences",
    default=True,
    help="Backfill missing sequence_sequence values from TMAlphaFold payloads when available.",
)
@click.option(
    "--use-gpu/--no-gpu",
    default=None,
    help="Override GPU usage for the local TMbed fallback stage.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Optional batch size override for the local fallback stage.",
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="Optional worker-count override for the local fallback stage.",
)
def run_verified_tm_fallbacks_cli(
    fallback_mode,
    run_all,
    limit,
    pdb_codes,
    fallback_methods,
    all_verified_methods,
    include_completed,
    retry_errors,
    methods,
    with_tmdet,
    tmalphafold_max_workers,
    tmalphafold_timeout,
    tmalphafold_refresh,
    tmalphafold_retry_errors,
    backfill_sequences,
    use_gpu,
    batch_size,
    max_workers,
):
    selected_codes = _normalize_cli_pdb_codes(pdb_codes)
    if not run_all and not selected_codes and limit is None:
        raise click.ClickException(
            "Provide --all, --limit, or at least one --pdb-code so the fallback scope is explicit."
        )

    _ensure_ml_runtime("run-verified-tm-fallbacks", "tmbed")
    fallback_predictors = (
        normalize_verified_tm_fallback_predictor_names(fallback_methods)
        if fallback_methods
        else None
    )
    method_list = [item.strip() for item in str(methods or "").split(",") if item.strip()]
    summary = run_verified_tm_fallback_pipeline(
        fallback_mode=str(fallback_mode or "").strip().lower(),
        fallback_predictor_names=fallback_predictors,
        bulk_safe_only=not all_verified_methods,
        pdb_codes=selected_codes,
        limit=limit,
        include_completed=include_completed,
        retry_errors=retry_errors,
        use_gpu=use_gpu,
        batch_size=batch_size,
        max_workers=max_workers,
        tmalphafold_methods=method_list,
        with_tmdet=with_tmdet,
        tmalphafold_max_workers=tmalphafold_max_workers,
        tmalphafold_timeout=tmalphafold_timeout,
        tmalphafold_refresh=tmalphafold_refresh,
        tmalphafold_retry_errors=tmalphafold_retry_errors,
        tmalphafold_backfill_sequences=backfill_sequences,
        progress_callback=click.echo,
    )
    DashboardAnnotationDatasetService._record_payload_cache.clear()
    click.echo(json.dumps(summary, indent=2))


@app.cli.command("optional-tm-runtime-status")
@click.option(
    "--predictor",
    type=str,
    multiple=True,
    help=(
        "Optional predictor(s) to inspect. Repeat the option or omit it to inspect all supported optional predictors."
    ),
)
def optional_tm_runtime_status_cli(predictor):
    summary = get_optional_tm_runtime_status(
        predictor_names=normalize_optional_tm_predictor_names(predictor) if predictor else None
    )
    click.echo(json.dumps(summary, indent=2))


@app.cli.command("import-optional-tm-prediction-results")
@click.option(
    "--predictor",
    required=True,
    type=str,
    multiple=True,
    help=(
        "Optional fallback predictor being imported into normalized MetaMP storage. "
        f"Supported values include: {', '.join(OPTIONAL_TM_PREDICTORS)}. "
        "Repeat the option to import multiple predictors, or use '--predictor all'."
    ),
)
@click.option(
    "--input-path",
    type=str,
    default=None,
    help="Optional CSV results path. Defaults to the predictor-specific runtime results file.",
)
@click.option(
    "--strict-missing-inputs/--skip-missing-inputs",
    default=False,
    help="Fail if any expected per-predictor results.csv file is missing during multi-predictor import.",
)
def import_optional_tm_prediction_results_cli(predictor, input_path, strict_missing_inputs):
    normalized_predictors = normalize_optional_tm_predictor_names(predictor)
    if len(normalized_predictors) == 1 and str(next(iter(predictor))).strip().lower() not in {"all", "*"}:
        summary = import_optional_tm_prediction_results(
            predictor_name=normalized_predictors[0],
            input_path=input_path,
            progress_callback=click.echo,
        )
    else:
        if input_path:
            raise click.ClickException(
                "--input-path is only supported for single-predictor import. "
                "For '--predictor all' or repeated --predictor values, MetaMP imports each predictor "
                "from its own runtime results file under /var/app/data/tm_predictions/external/<predictor>/results.csv."
            )
        summary = import_optional_tm_prediction_results_bulk(
            predictor_names=normalized_predictors,
            skip_missing_inputs=not strict_missing_inputs,
            progress_callback=click.echo,
        )
    DashboardAnnotationDatasetService._record_payload_cache.clear()
    click.echo(json.dumps(summary, indent=2))


@app.cli.command("export-discrepancy-benchmark")
@click.option(
    "--include-all",
    is_flag=True,
    help="Export the full discrepancy review queue instead of disagreement-only rows.",
)
def export_discrepancy_benchmark(include_all):
    result = DiscrepancyBenchmarkExportService.export_release(include_all=include_all)
    click.echo(json.dumps(result, indent=2))


@app.cli.command("discrepancy-benchmark-status")
def discrepancy_benchmark_status():
    status = DiscrepancyBenchmarkExportService.latest_export_metadata()
    if status is None:
        click.echo("No discrepancy benchmark export metadata found.")
        return
    click.echo(json.dumps(status, indent=2))


@app.cli.command("export-high-confidence-subset")
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["csv", "json"]),
    default="csv",
    show_default=True,
    help="Export format for the high-confidence subset.",
)
def export_high_confidence_subset(export_format):
    payload = DiscrepancyBenchmarkExportService.build_high_confidence_download_payload(
        export_format=export_format
    )
    click.echo(json.dumps(payload["metadata"], indent=2))


@app.cli.command("validate-dashboard-regressions")
def validate_dashboard_regressions():
    result = DashboardRegressionValidationService.run_checks()
    click.echo(json.dumps(result, indent=2))
    if not result.get("passed"):
        raise SystemExit(1)


@app.cli.command("queue-machine-learning-job")
def queue_machine_learning_job():
    result = machine_learning_job.delay()
    click.echo(f"Queued machine learning job: {result.id}")


@app.cli.command("queue-tmalphafold-sync")
@click.option(
    "--methods",
    type=str,
    default=",".join(TMALPHAFOLD_SEQUENCE_METHODS + TMALPHAFOLD_AUX_METHODS),
    show_default=False,
    help="Comma-separated TMAlphaFold method list.",
)
@click.option(
    "--with-tmdet/--without-tmdet",
    default=True,
    help="Include TMDET structure-based membrane-plane predictions.",
)
@click.option(
    "--refresh/--skip-refresh",
    default=False,
    help="Refetch predictions even when successful normalized rows already exist.",
)
@click.option(
    "--retry-errors/--skip-errors",
    default=False,
    help="When not refreshing, retry previously stored error rows instead of skipping all existing rows.",
)
@click.option(
    "--max-workers",
    type=int,
    default=8,
    show_default=True,
    help="Concurrent TMAlphaFold request count.",
)
@click.option(
    "--timeout",
    type=int,
    default=30,
    show_default=True,
    help="Per-request timeout in seconds.",
)
@click.option(
    "--backfill-sequences/--skip-backfill-sequences",
    default=True,
    help="Backfill blank sequence_sequence values from TMAlphaFold payloads when available.",
)
@click.option(
    "--with-tmbed/--without-tmbed",
    default=True,
    help="Also run the local TMbed refresh after the TMAlphaFold sync.",
)
@click.option(
    "--tmbed-use-gpu",
    is_flag=True,
    help="Allow the appended TMbed run to use GPU if available.",
)
@click.option(
    "--tmbed-batch-size",
    type=int,
    default=None,
    help="Override batch size for the appended TMbed run.",
)
@click.option(
    "--tmbed-max-workers",
    type=int,
    default=None,
    help="Override predictor parallelism for the appended TMbed run.",
)
@click.option(
    "--tmbed-refresh/--skip-tmbed-refresh",
    default=False,
    help="Rerun TMbed even for records that already have TMbed values.",
)
def queue_tmalphafold_sync(
    methods,
    with_tmdet,
    refresh,
    retry_errors,
    max_workers,
    timeout,
    backfill_sequences,
    with_tmbed,
    tmbed_use_gpu,
    tmbed_batch_size,
    tmbed_max_workers,
    tmbed_refresh,
):
    method_list = [item.strip() for item in str(methods or "").split(",") if item.strip()]
    result = sync_tmalphafold_predictions_task.delay(
        methods=method_list,
        with_tmdet=with_tmdet,
        refresh=refresh,
        retry_errors=retry_errors,
        max_workers=max_workers,
        timeout=timeout,
        backfill_sequences=backfill_sequences,
        with_tmbed=with_tmbed,
        tmbed_use_gpu=tmbed_use_gpu,
        tmbed_batch_size=tmbed_batch_size,
        tmbed_max_workers=tmbed_max_workers,
        tmbed_refresh=tmbed_refresh,
    )
    click.echo(f"Queued TMAlphaFold sync job: {result.id}")


@app.cli.command("queue-tmbed-sync")
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Restrict the sync to the first N eligible targets.",
)
@click.option(
    "--pdb-code",
    "pdb_codes",
    multiple=True,
    help="Restrict the sync to one or more specific PDB codes. Repeat the option to provide multiple codes.",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    help="Allow TMbed to use GPU if available.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Override batch size for TMbed.",
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="Override predictor parallelism for TMbed.",
)
@click.option(
    "--refresh/--skip-refresh",
    default=False,
    help="Rerun TMbed even for records that already have TMbed values.",
)
def queue_tmbed_sync(
    limit,
    pdb_codes,
    use_gpu,
    batch_size,
    max_workers,
    refresh,
):
    selected_codes = _normalize_cli_pdb_codes(pdb_codes)
    result = sync_tmbed_predictions_task.delay(
        pdb_codes=selected_codes,
        limit=limit,
        use_gpu=use_gpu,
        batch_size=batch_size,
        max_workers=max_workers,
        refresh=refresh,
    )
    click.echo(f"Queued TMbed sync job: {result.id}")


@app.cli.command("queue-protein-database-sync")
def queue_protein_database_sync():
    result = scheduled_sync_command.delay()
    click.echo(f"Queued protein database sync: {result.id}")


@app.cli.command("protein-database-sync-status")
def protein_database_sync_status():
    status = TaskStatusRecorder(app.config).read_latest(
        "shared-task-sync-protein-database"
    )
    if status is None:
        click.echo("No protein database sync status found.")
        return

    click.echo(json.dumps(status, indent=2))


@app.cli.command("queue-production-maintenance")
@click.option(
    "--use-gpu",
    is_flag=True,
    help="Allow TMbed to use GPU if available.",
)
@click.option("--batch-size", type=int, default=None, help="Override batch size.")
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="Override predictor parallelism inside each batch.",
)
@click.option(
    "--skip-benchmark-export",
    is_flag=True,
    help="Skip exporting the discrepancy benchmark during maintenance.",
)
@click.option(
    "--skip-validation",
    is_flag=True,
    help="Skip dashboard regression validation during maintenance.",
)
def queue_production_maintenance(
    use_gpu,
    batch_size,
    max_workers,
    skip_benchmark_export,
    skip_validation,
):
    result = monthly_production_maintenance.delay(
        use_gpu=use_gpu,
        batch_size=batch_size,
        max_workers=max_workers,
        export_benchmark=not skip_benchmark_export,
        validate_regressions=not skip_validation,
    )
    click.echo(f"Queued monthly production maintenance task: {result.id}")


@app.cli.command("production-maintenance-status")
def production_maintenance_status():
    status = TaskStatusRecorder(app.config).read_latest(
        "shared-task-monthly-production-maintenance"
    )
    if status is None:
        click.echo("No monthly production maintenance status found.")
        return

    click.echo(json.dumps(status, indent=2))


def _normalize_inspect_payload(payload):
    if not payload:
        return {}
    return {
        worker_name: entries or []
        for worker_name, entries in payload.items()
    }


@app.cli.command("celery-list-jobs")
def celery_list_jobs():
    inspector = celery.control.inspect(timeout=5)
    payload = {
        "active": _normalize_inspect_payload(inspector.active()),
        "reserved": _normalize_inspect_payload(inspector.reserved()),
        "scheduled": _normalize_inspect_payload(inspector.scheduled()),
    }
    payload["totals"] = {
        key: sum(len(entries) for entries in value.values())
        for key, value in payload.items()
        if isinstance(value, dict)
    }
    click.echo(json.dumps(payload, indent=2, default=str))


@app.cli.command("celery-task-status")
@click.option("--task-id", required=True, help="Celery task id to inspect.")
def celery_task_status(task_id):
    task = AsyncResult(task_id, app=celery)
    payload = {
        "task_id": task_id,
        "status": task.status.lower(),
    }
    if task.successful():
        payload["result"] = task.result
    elif task.failed():
        payload["error"] = str(task.result)

    click.echo(json.dumps(payload, indent=2, default=str))


@app.cli.command("celery-revoke-task")
@click.option("--task-id", required=True, help="Celery task id to revoke.")
@click.option(
    "--terminate",
    is_flag=True,
    help="Also terminate the worker process if the task is already running.",
)
@click.option(
    "--signal",
    default="SIGTERM",
    show_default=True,
    help="Signal to use when terminating a running task.",
)
def celery_revoke_task(task_id, terminate, signal):
    celery.control.revoke(task_id, terminate=terminate, signal=signal)
    payload = {
        "task_id": task_id,
        "status": "revoked",
        "terminate": terminate,
        "signal": signal,
    }
    click.echo(json.dumps(payload, indent=2, default=str))


@app.cli.command("celery-purge-queued")
@click.option(
    "--yes",
    is_flag=True,
    help="Confirm purging all waiting tasks from the broker queues.",
)
def celery_purge_queued(yes):
    if not yes:
        raise click.UsageError("Pass --yes to purge all queued Celery tasks.")
    purged = celery.control.purge()
    click.echo(
        json.dumps(
            {
                "status": "purged",
                "purged_task_count": purged,
            },
            indent=2,
            default=str,
        )
    )


@app.cli.command("sync-question-with-database")
def init_data_questions():
    addQuestion()


@app.cli.command("sync-system_admin-with-database")
def init_system_admin():
    addDefaultAdmin()


@app.cli.command("sync-feedback-questions-with-database")
def init_feedback_questions():
    addFeedbackQuestions()


if __name__ == "__main__":
    cli()
