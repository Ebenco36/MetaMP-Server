from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

from src.ingestion.dataset_layout import DatasetLayout


@dataclass(frozen=True)
class IngestionContext:
    run_date: str
    layout: DatasetLayout
    progress_callback: Optional[Callable[[str], None]] = None

    def report(self, message: str):
        if self.progress_callback:
            self.progress_callback(message)


class DatasetSource(ABC):
    @abstractmethod
    def run(self, context: IngestionContext):
        raise NotImplementedError
