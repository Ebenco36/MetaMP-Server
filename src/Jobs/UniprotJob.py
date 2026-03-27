from datetime import date

import pandas as pd

from src.ingestion.context import IngestionContext
from src.ingestion.dataset_layout import DatasetLayout
from src.ingestion.uniprot_source import UniProtDatasetSource


def _build_context():
    layout = DatasetLayout()
    layout.ensure_directories()
    return IngestionContext(
        run_date=date.today().strftime("%Y-%m-%d"),
        layout=layout,
    )


class UniProtDataFetcher(UniProtDatasetSource):
    def fetch_and_save(self, df: pd.DataFrame, id_col: str):
        if id_col not in df.columns:
            raise KeyError(f"Missing expected column: {id_col}")

        compatible_df = df.copy()
        if "pdb_code" in compatible_df.columns and "Pdb Code" not in compatible_df.columns:
            compatible_df = compatible_df.rename(columns={"pdb_code": "Pdb Code"})

        required_columns = {"Pdb Code", id_col}
        missing_columns = required_columns - set(compatible_df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise KeyError(f"Missing required columns for UniProt sync: {missing}")

        if id_col != "uniprot_id":
            compatible_df = compatible_df.rename(columns={id_col: "uniprot_id"})

        context = _build_context()
        compatible_df.to_csv(context.layout.quantitative_dataset_current, index=False)
        return self.run(context)


if __name__ == "__main__":
    layout = DatasetLayout()
    df = pd.read_csv(layout.quantitative_dataset_current, low_memory=False)
    UniProtDataFetcher().fetch_and_save(df, "uniprot_id")
