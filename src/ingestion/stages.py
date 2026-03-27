from __future__ import annotations

from abc import ABC, abstractmethod

from src.ingestion.context import IngestionContext
from src.ingestion.valid_dataset_cleaner import write_valid_datasets
from src.ingestion.validation_service import IngestionValidationService


class IngestionStage(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, context: IngestionContext):
        raise NotImplementedError


class ComponentStage(IngestionStage):
    def __init__(self, name: str, description: str, component):
        self.name = name
        self.description = description
        self.component = component

    def run(self, context: IngestionContext):
        return self.component.run(context)


class ValidDatasetCleaningStage(IngestionStage):
    name = "clean_valid"
    description = "Clean and publish valid dataset artifacts."

    def run(self, context: IngestionContext):
        context.report("[clean_valid] Publishing cleaned datasets into datasets/valid")
        write_valid_datasets(context.layout, progress_callback=context.report)
        return None


class ValidationReportStage(IngestionStage):
    name = "validate_release"
    description = "Validate cleaned datasets and write a release validation report."

    def __init__(self):
        self.validator = IngestionValidationService()

    def run(self, context: IngestionContext):
        context.report("[validate_release] Validating cleaned dataset artifacts")
        return self.validator.run(context)
