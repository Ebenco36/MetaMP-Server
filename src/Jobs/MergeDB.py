from datetime import date

from src.ingestion.context import IngestionContext
from src.ingestion.dataset_layout import DatasetLayout
from src.ingestion.quantitative_dataset_builder import QuantitativeDatasetBuilder


def _build_context():
    layout = DatasetLayout()
    layout.ensure_directories()
    return IngestionContext(
        run_date=date.today().strftime("%Y-%m-%d"),
        layout=layout,
    )


class DataImport(QuantitativeDatasetBuilder):
    def loadFile(self):
        return self.run(_build_context())


if __name__ == "__main__":
    DataImport().loadFile()
