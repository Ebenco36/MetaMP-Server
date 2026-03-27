"""Production-oriented ingestion pipeline for MetaMP datasets."""

from src.ingestion.dataset_layout import DatasetLayout
from src.ingestion.pipeline import ProteinDatabaseIngestionPipeline

__all__ = ["DatasetLayout", "ProteinDatabaseIngestionPipeline"]
