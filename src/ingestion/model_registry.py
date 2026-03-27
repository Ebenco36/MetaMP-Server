from dataclasses import dataclass

from src.MP.model import MembraneProteinData
from src.MP.model_mpstruct import MPSTURC
from src.MP.model_opm import OPM
from src.MP.model_pdb import PDB
from src.MP.model_uniprot import Uniprot
from src.ingestion.dataset_layout import DatasetLayout
from src.ingestion.settings import IngestionSettings


@dataclass(frozen=True)
class DatasetModelBinding:
    name: str
    model_class: type
    csv_path: str


def get_dataset_model_bindings():
    settings = IngestionSettings.from_environment()
    layout = DatasetLayout(base_dir=settings.dataset_base_dir)
    return [
        DatasetModelBinding(
            name="mpstruct",
            model_class=MPSTURC,
            csv_path=str(layout.valid_dir / "Mpstruct_dataset.csv"),
        ),
        DatasetModelBinding(
            name="pdb",
            model_class=PDB,
            csv_path=str(layout.valid_dir / "PDB_data_transformed.csv"),
        ),
        DatasetModelBinding(
            name="opm",
            model_class=OPM,
            csv_path=str(layout.valid_dir / "NEWOPM.csv"),
        ),
        DatasetModelBinding(
            name="uniprot",
            model_class=Uniprot,
            csv_path=str(layout.valid_dir / "Uniprot_functions.csv"),
        ),
        DatasetModelBinding(
            name="quantitative",
            model_class=MembraneProteinData,
            csv_path=str(layout.valid_dir / "Quantitative_data.csv"),
        ),
    ]
