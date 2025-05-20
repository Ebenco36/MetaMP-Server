import os, sys
sys.path.append(os.getcwd())

from database.db import db
from flask import current_app as app
import pandas as pd
from src.Dashboard.services import get_tables_as_dataframe, get_table_as_dataframe
from src.AI_Packages.TMProteinPredictor import DeepTMHMMPredictor, MultiModelAnalyzer, TMbedPredictor


import requests
from tqdm import tqdm 

def fetch_pdb_entry_sequence(pdb_id: str, timeout: int = 30) -> str:
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id.lower()}/download"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None              # 404, 500, etc. → skip
        fasta = resp.text
    except requests.RequestException:
        return None                  # connection problems → skip

    # strip header(s) and join lines
    return "".join(
        ln.strip() for ln in fasta.splitlines() if not ln.startswith(">")
    )

def fill_missing_sequences(
        df: pd.DataFrame,
        pdb_col: str = "pdb_code",
        seq_col: str = "sequence_sequence"
    ) -> None:
    if seq_col not in df.columns:
        df[seq_col] = pd.NA

    need_seq = df[seq_col].isna()
    if need_seq.any():
        for idx, pdb_id in tqdm(
            df.loc[need_seq, pdb_col].items(),
            total=need_seq.sum(),
            desc="Fetching PDB sequences",
        ):
            seq = fetch_pdb_entry_sequence(pdb_id)
            if seq is not None:
                df.at[idx, seq_col] = seq         # update in-place

    return df

def TMbedDeepTMHMM():
    with app.app_context():
        # 1) Load and merge your tables
        table_names = ['membrane_proteins', 'membrane_protein_opm']
        result_df = get_tables_as_dataframe(table_names, "pdb_code")
        result_df_uniprot = get_table_as_dataframe("membrane_protein_uniprot")

        common = (set(result_df.columns) - {"pdb_code"}) & set(result_df_uniprot.columns)
        right_pruned = result_df_uniprot.drop(columns=list(common))
        all_data = pd.merge(
            right=result_df,
            left=right_pruned,
            on="pdb_code",
            how="outer"
        )
        # all_data = all_data[all_data["sequence_sequence"].notna()]
        # all_data = all_data[all_data["sequence_sequence"] != ""]

        # 2) Define your DB connection parameters
        db_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mpvis_db',
            'user': 'mpvis_user',
            'password': 'mpvis_user'
        }


    required_cols = ["pdb_code", "sequence_sequence", "TMbed_tm_count", "DeepTMHMM_tm_count"]
    available = [c for c in required_cols if c in all_data.columns]
    all_data = all_data[required_cols]
    all_data = all_data.loc[
        all_data[["TMbed_tm_count", "DeepTMHMM_tm_count"]]
            .fillna("")
            .eq("")
            .all(axis=1)
    ]

    all_data = fill_missing_sequences(all_data, pdb_col="pdb_code", seq_col="sequence_sequence")

    all_data = all_data[all_data["sequence_sequence"].notna()]
    all_data = all_data[all_data["sequence_sequence"] != ""]


    # 3) Instantiate the analyzer
    #    - batch_size: number of sequences per DB update
    #    - max_workers: use >1 to parallelize predictor runs (optional)
    analyzer = MultiModelAnalyzer(
        db_params=db_params,
        table="membrane_proteins",
        batch_size=10,
        max_workers=2
    )

    analyzer.register(TMbedPredictor(format_code=0, use_gpu=False))
    analyzer.register(DeepTMHMMPredictor())

    result_df = analyzer.analyze(
        df=all_data,
        id_col="pdb_code",
        seq_col="sequence_sequence",
        csv_out="tm_summary.csv"
    )
# 