from datetime import date

from src.ingestion.context import IngestionContext
from src.ingestion.dataset_layout import DatasetLayout
from src.ingestion.opm_source import OpmDatasetSource


def _build_context():
    layout = DatasetLayout()
    layout.ensure_directories()
    return IngestionContext(
        run_date=date.today().strftime("%Y-%m-%d"),
        layout=layout,
    )


class NEWOPM(OpmDatasetSource):
    def fetch(self):
        return self.run(_build_context())


if __name__ == "__main__":
    NEWOPM().fetch()
