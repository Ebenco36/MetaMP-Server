from datetime import date

from src.ingestion.context import IngestionContext
from src.ingestion.dataset_layout import DatasetLayout
from src.ingestion.mpstruc_source import MPStrucDatasetSource


def _build_context():
    layout = DatasetLayout()
    layout.ensure_directories()
    return IngestionContext(
        run_date=date.today().strftime("%Y-%m-%d"),
        layout=layout,
    )


class MPSTRUCT(MPStrucDatasetSource):
    def load_data(self):
        return self.run(_build_context())

    def fetch_data(self):
        return self.load_data()


if __name__ == "__main__":
    MPSTRUCT().fetch_data()
