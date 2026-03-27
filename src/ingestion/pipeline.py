from __future__ import annotations

import logging
from datetime import date
from time import monotonic
from typing import Callable, Optional

from src.ingestion.context import IngestionContext
from src.ingestion.dataset_layout import DatasetLayout
from src.ingestion.mpstruc_source import MPStrucDatasetSource
from src.ingestion.opm_source import OpmDatasetSource
from src.ingestion.pdb_source import PdbDatasetSource
from src.ingestion.quantitative_dataset_builder import QuantitativeDatasetBuilder
from src.ingestion.settings import IngestionSettings
from src.ingestion.stages import (
    ComponentStage,
    ValidationReportStage,
    ValidDatasetCleaningStage,
)
from src.ingestion.uniprot_source import UniProtDatasetSource

logger = logging.getLogger(__name__)


class ProteinDatabaseIngestionPipeline:
    def __init__(
        self,
        layout: Optional[DatasetLayout] = None,
        settings: Optional[IngestionSettings] = None,
    ):
        self.settings = settings or IngestionSettings.from_environment()
        self.layout = layout or DatasetLayout(base_dir=self.settings.dataset_base_dir)
        self.stages = self._build_stage_registry()

    def _build_stage_registry(self):
        return {
            "mpstruc": ComponentStage(
                name="mpstruc",
                description="Fetch and normalize MPStruc source data.",
                component=MPStrucDatasetSource(
                    allow_stale_fallback=self.settings.allow_stale_source_fallback,
                    freshness_days=self.settings.artifact_freshness_days,
                ),
            ),
            "pdb": ComponentStage(
                name="pdb",
                description="Fetch and normalize PDB source data.",
                component=PdbDatasetSource(
                    batch_size=self.settings.pdb_batch_size,
                    freshness_days=self.settings.artifact_freshness_days,
                ),
            ),
            "quantitative": ComponentStage(
                name="quantitative",
                description="Build the merged quantitative dataset.",
                component=QuantitativeDatasetBuilder(
                    freshness_days=self.settings.artifact_freshness_days
                ),
            ),
            "opm": ComponentStage(
                name="opm",
                description="Fetch and normalize OPM source data.",
                component=OpmDatasetSource(
                    batch_size=self.settings.opm_batch_size,
                    freshness_days=self.settings.artifact_freshness_days,
                ),
            ),
            "uniprot": ComponentStage(
                name="uniprot",
                description="Fetch and normalize UniProt source data.",
                component=UniProtDatasetSource(
                    batch_size=self.settings.uniprot_batch_size,
                    freshness_days=self.settings.artifact_freshness_days,
                ),
            ),
            "clean_valid": ValidDatasetCleaningStage(),
            "validate_release": ValidationReportStage(),
        }

    def get_available_stage_names(self):
        return tuple(self.stages.keys())

    def _resolve_stage_names(
        self,
        stage_names: tuple[str, ...] | None,
        clean_valid: bool,
    ) -> tuple[str, ...]:
        selected = stage_names or self.settings.stage_names
        normalized = []

        for stage_name in selected:
            normalized_name = stage_name.strip().lower()
            if normalized_name not in self.stages:
                available = ", ".join(self.get_available_stage_names())
                raise ValueError(
                    f"Unknown ingestion stage '{stage_name}'. Available stages: {available}"
                )
            normalized.append(normalized_name)

        if not clean_valid:
            normalized = [name for name in normalized if name != "clean_valid"]

        return tuple(normalized)

    def run(
        self,
        clean_valid: bool | None = None,
        stage_names: tuple[str, ...] | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        pipeline_started_at = monotonic()
        self.layout.ensure_directories()
        context = IngestionContext(
            run_date=date.today().strftime("%Y-%m-%d"),
            layout=self.layout,
            progress_callback=progress_callback,
        )
        should_clean_valid = (
            self.settings.clean_valid_datasets
            if clean_valid is None
            else clean_valid
        )
        active_stage_names = self._resolve_stage_names(
            stage_names=stage_names,
            clean_valid=should_clean_valid,
        )

        logger.info(
            "Running ingestion stages=%s for run_date=%s",
            ",".join(active_stage_names),
            context.run_date,
        )
        if progress_callback:
            progress_callback(
                f"Running ingestion stages: {', '.join(active_stage_names)}"
            )

        for stage_name in active_stage_names:
            stage = self.stages[stage_name]
            stage_started_at = monotonic()
            logger.info(
                "Starting ingestion stage '%s': %s",
                stage.name,
                stage.description,
            )
            if progress_callback:
                progress_callback(
                    f"Starting stage '{stage.name}': {stage.description}"
                )
            stage.run(context)
            stage_duration_seconds = monotonic() - stage_started_at
            logger.info("Completed ingestion stage '%s'", stage.name)
            if progress_callback:
                progress_callback(
                    f"Completed stage '{stage.name}' in {stage_duration_seconds:.1f}s"
                )

        pipeline_duration_seconds = monotonic() - pipeline_started_at
        logger.info(
            "Protein dataset ingestion pipeline completed for run_date=%s",
            context.run_date,
        )
        if progress_callback:
            progress_callback(
                "Protein dataset ingestion pipeline completed "
                f"for run date {context.run_date} in {pipeline_duration_seconds:.1f}s"
            )
        return context
