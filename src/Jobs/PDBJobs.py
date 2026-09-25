from datetime import date

from src.ingestion.context import IngestionContext
from src.ingestion.dataset_layout import DatasetLayout
from src.ingestion.pdb_source import PdbDatasetSource


def _build_context():
    layout = DatasetLayout()
    layout.ensure_directories()
    return IngestionContext(
        run_date=date.today().strftime("%Y-%m-%d"),
        layout=layout,
    )


class PDBJOBS(PdbDatasetSource):
    def __init__(self, batch_size=200):
        super().__init__(batch_size=batch_size)

    def fetch_data(self):
        return self.run(_build_context())


if __name__ == "__main__":
    PDBJOBS(batch_size=200).fetch_data()
