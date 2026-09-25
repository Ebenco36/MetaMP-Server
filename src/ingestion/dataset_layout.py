from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class DatasetLayout:
    base_dir: Path = field(default_factory=lambda: Path(".") / "datasets")

    @property
    def valid_dir(self) -> Path:
        return self.base_dir / "valid"

    @property
    def pdb_batch_dir(self) -> Path:
        return self.base_dir / "PDB"

    @property
    def opm_batch_dir(self) -> Path:
        return self.base_dir / "OPM"

    @property
    def uniprot_batch_dir(self) -> Path:
        return self.base_dir / "UniProt"

    def ensure_directories(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.valid_dir.mkdir(parents=True, exist_ok=True)
        self.pdb_batch_dir.mkdir(parents=True, exist_ok=True)
        self.opm_batch_dir.mkdir(parents=True, exist_ok=True)
        self.uniprot_batch_dir.mkdir(parents=True, exist_ok=True)

    def mpstruc_xml(self, run_date: str) -> Path:
        return self.base_dir / f"mpstrucTblXml_{run_date}.xml"

    @property
    def mpstruc_xml_current(self) -> Path:
        return self.base_dir / "mpstrucTblXml.xml"

    def mpstruc_dataset(self, run_date: str) -> Path:
        return self.base_dir / f"Mpstruct_dataset_{run_date}.csv"

    @property
    def mpstruc_dataset_current(self) -> Path:
        return self.base_dir / "Mpstruct_dataset.csv"

    @property
    def mpstruc_ids_current(self) -> Path:
        return self.base_dir / "mpstruct_ids.csv"

    def pdb_dataset(self, run_date: str) -> Path:
        return self.base_dir / f"PDB_data_{run_date}.csv"

    @property
    def pdb_dataset_current(self) -> Path:
        return self.base_dir / "PDB_data.csv"

    def pdb_transformed_dataset(self, run_date: str) -> Path:
        return self.base_dir / f"PDB_data_transformed_{run_date}.csv"

    @property
    def pdb_transformed_dataset_current(self) -> Path:
        return self.base_dir / "PDB_data_transformed.csv"

    @property
    def enriched_dataset_current(self) -> Path:
        return self.base_dir / "enriched_db.csv"

    def quantitative_dataset(self, run_date: str) -> Path:
        return self.base_dir / f"Quantitative_data_{run_date}.csv"

    @property
    def quantitative_dataset_current(self) -> Path:
        return self.base_dir / "Quantitative_data.csv"

    def opm_dataset(self, run_date: str) -> Path:
        return self.base_dir / f"NEWOPM_{run_date}.csv"

    @property
    def opm_dataset_current(self) -> Path:
        return self.base_dir / "NEWOPM.csv"

    def uniprot_dataset(self, run_date: str) -> Path:
        return self.base_dir / f"Uniprot_functions_{run_date}.csv"

    @property
    def uniprot_dataset_current(self) -> Path:
        return self.base_dir / "Uniprot_functions.csv"

    @staticmethod
    def _parse_run_date(path: Path, prefix: str, suffix: str) -> date | None:
        name = path.name
        expected_prefix = f"{prefix}_"
        if not (name.startswith(expected_prefix) and name.endswith(suffix)):
            return None

        raw_date = name[len(expected_prefix) : len(name) - len(suffix)]
        try:
            return datetime.strptime(raw_date, "%Y-%m-%d").date()
        except ValueError:
            return None

    def latest_recent_dated_path(
        self,
        prefix: str,
        suffix: str,
        freshness_days: int,
    ) -> tuple[str, Path] | None:
        freshest: tuple[date, Path] | None = None
        cutoff = date.today() - timedelta(days=max(freshness_days - 1, 0))

        for path in self.base_dir.glob(f"{prefix}_*{suffix}"):
            parsed_date = self._parse_run_date(path, prefix, suffix)
            if parsed_date is None or parsed_date < cutoff:
                continue
            if freshest is None or parsed_date > freshest[0]:
                freshest = (parsed_date, path)

        if freshest is None:
            return None

        return freshest[0].strftime("%Y-%m-%d"), freshest[1]
