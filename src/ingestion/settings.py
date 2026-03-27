from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_STAGE_ORDER = (
    "mpstruc",
    "pdb",
    "quantitative",
    "opm",
    "uniprot",
    "clean_valid",
    "validate_release",
)


def _read_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_stage_names(default: tuple[str, ...]) -> tuple[str, ...]:
    configured = os.getenv("INGESTION_STAGES")
    if not configured:
        return default

    parsed = [stage.strip().lower() for stage in configured.split(",") if stage.strip()]
    return tuple(parsed) if parsed else default


@dataclass(frozen=True)
class IngestionSettings:
    dataset_base_dir: Path = field(
        default_factory=lambda: Path(os.getenv("INGESTION_DATASET_BASE_DIR", "datasets"))
    )
    stage_names: tuple[str, ...] = field(
        default_factory=lambda: _read_stage_names(DEFAULT_STAGE_ORDER)
    )
    clean_valid_datasets: bool = field(
        default_factory=lambda: _read_bool_env("INGESTION_CLEAN_VALID_DATASETS", True)
    )
    allow_stale_source_fallback: bool = field(
        default_factory=lambda: _read_bool_env(
            "INGESTION_ALLOW_STALE_SOURCE_FALLBACK",
            True,
        )
    )
    pdb_batch_size: int = field(
        default_factory=lambda: _read_int_env("INGESTION_PDB_BATCH_SIZE", 200)
    )
    opm_batch_size: int = field(
        default_factory=lambda: _read_int_env("INGESTION_OPM_BATCH_SIZE", 50)
    )
    uniprot_batch_size: int = field(
        default_factory=lambda: _read_int_env("INGESTION_UNIPROT_BATCH_SIZE", 200)
    )
    artifact_freshness_days: int = field(
        default_factory=lambda: _read_int_env("INGESTION_ARTIFACT_FRESHNESS_DAYS", 30)
    )

    @classmethod
    def from_environment(cls) -> "IngestionSettings":
        return cls()

    def with_overrides(
        self,
        *,
        stage_names: tuple[str, ...] | None = None,
        clean_valid_datasets: bool | None = None,
        dataset_base_dir: Path | None = None,
    ) -> "IngestionSettings":
        return IngestionSettings(
            dataset_base_dir=dataset_base_dir or self.dataset_base_dir,
            stage_names=stage_names or self.stage_names,
            clean_valid_datasets=(
                self.clean_valid_datasets
                if clean_valid_datasets is None
                else clean_valid_datasets
            ),
            allow_stale_source_fallback=self.allow_stale_source_fallback,
            pdb_batch_size=self.pdb_batch_size,
            opm_batch_size=self.opm_batch_size,
            uniprot_batch_size=self.uniprot_batch_size,
            artifact_freshness_days=self.artifact_freshness_days,
        )
