import subprocess
import sys
import re
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import pandas as pd
import psycopg2
from psycopg2 import sql
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor


def extract_tmr_counts_from_gff_text(gff_text: str) -> dict[str, int]:
    """
    Given GFF3 or DeepTMHMM markdown content, extract the number of predicted TMRs per sequence.
    """
    counts: dict[str, int] = {}
    for m in re.finditer(r"^#\s*(\S+).*Number of predicted TMRs:\s*(\d+)", gff_text, re.MULTILINE):
        seq, num = m.group(1), int(m.group(2))
        counts[seq] = num
    if counts:
        return counts
    # fallback: count TMhelix lines
    for line in gff_text.splitlines():
        parts = line.split("\t")
        if len(parts) > 2 and parts[2] == "TMhelix":
            seq = parts[0]
            counts[seq] = counts.get(seq, 0) + 1
    return counts


def run_tmbed_predict(fasta: str,
                     out_pred: str,
                     format_code: int = 0,
                     use_gpu: bool = True) -> bool:
    """
    Runs `python -m tmbed predict`:
      -f <fasta>
      -p <out_pred>
      --out-format <format_code>
    Returns True on success.
    """
    cmd = [
        sys.executable, "-m", "tmbed", "predict",
        "-f", fasta,
        "-p", out_pred,
        "--out-format", str(format_code)
    ]
    if use_gpu:
        cmd.append("--use-gpu")

    print("Running:", *cmd)
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(res.stdout, end="")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[TMbed] ERROR:\n{e.stdout}\n{e.stderr}", file=sys.stderr)
        return False


def count_tmbed_3line(pred_path: str) -> dict[str, int]:
    """
    Parse TMbed 3-line output to count TM runs per sequence.
    """
    counts: dict[str,int] = {}
    with open(pred_path) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    if len(lines) % 3 != 0:
        raise ValueError("TMbed 3-line output malformed.")
    for i in range(0, len(lines), 3):
        hdr, labels = lines[i], lines[i+2]
        seq = hdr.lstrip('>')
        runs = re.findall(r"[HhBb]+", labels)
        counts[seq] = len(runs)
    return counts


class BasePredictor(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(self, fasta_path: str) -> dict[str,int]:
        pass


class TMbedPredictor(BasePredictor):
    """
    TMbed wrapper using the standard 3-line implementation.
    """
    def __init__(self, format_code: int = 0, use_gpu: bool = True):
        super().__init__("TMbed")
        self.format_code = format_code
        self.use_gpu = use_gpu

    def predict(self, fasta_path: str) -> dict[str,int]:
        pred_path = fasta_path + ".pred"
        if not run_tmbed_predict(fasta_path, pred_path, self.format_code, self.use_gpu):
            return {}
        counts = count_tmbed_3line(pred_path)
        os.remove(pred_path)
        return counts


class DeepTMHMMPredictor(BasePredictor):
    """
    DeepTMHMM wrapper via biolib.
    """
    def __init__(self, spec: str = "DTU/DeepTMHMM:1.0.24"):
        super().__init__("DeepTMHMM")
        try:
            import biolib
        except ImportError:
            raise RuntimeError("Install biolib for DeepTMHMMPredictor")
        self.app = biolib.load(spec)

    def predict(self, fasta_path: str) -> dict[str,int]:
        job = self.app.run(fasta=fasta_path)
        fh = job.get_output_file('/deeptmhmm_results.md').get_file_handle()
        fh.seek(0)
        txt = fh.read().decode()
        return extract_tmr_counts_from_gff_text(txt)

class MultiModelAnalyzer:
    """
    Runs predictors on a DataFrame and returns the augmented DataFrame.
    If use_db=False, all database operations are skipped.
    If write_csv=True, writes out the full DataFrame to csv_out.
    """
    def __init__(self,
                 db_params: dict,
                 table: str,
                 batch_size: int = 10,
                 max_workers: int = 1,
                 max_sequences: int = 4,
                 use_db: bool = True,
                 write_csv: bool = False):
        self.predictors: list = []
        self.db_params = db_params
        self.table = table
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_sequences = max_sequences
        self.use_db = use_db
        self.write_csv = write_csv

    def register(self, predictor):
        """Add a BasePredictor subclass instance to the pipeline."""
        self.predictors.append(predictor)

    def analyze(self,
                df: pd.DataFrame,
                id_col: str,
                seq_col: str,
                csv_out: Optional[str] = None) -> pd.DataFrame:
        # 1) Reset index
        df = df.reset_index(drop=True)
        total = len(df)
        if total > self.max_sequences:
            raise ValueError(f"Too many sequences ({total}); max is {self.max_sequences}.")

        # 2) Add TM count columns
        for p in self.predictors:
            df[f"{p.name}_tm_count"] = pd.NA

        # 3) Optionally prepare DB
        if self.use_db:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            # ensure seq_col exists
            cur.execute(sql.SQL(
                "ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {seq_col} TEXT"
            ).format(
                tbl=sql.Identifier(self.table),
                seq_col=sql.Identifier(seq_col),
            ))
            # ensure TM count cols exist
            for p in self.predictors:
                col = f"{p.name}_tm_count"
                cur.execute(sql.SQL(
                    "ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} INTEGER"
                ).format(
                    tbl=sql.Identifier(self.table),
                    col=sql.Identifier(col),
                ))
            conn.commit()

        # 4) Process in batches
        batch_num = 0
        for start in range(0, total, self.batch_size):
            batch_num += 1
            end = min(start + self.batch_size, total)
            batch = df.iloc[start:end]

            # write batch FASTA
            with NamedTemporaryFile("w+", suffix=".fasta", delete=True) as fasta:
                for _, row in batch.iterrows():
                    fasta.write(f">{row[id_col]}\n{row[seq_col]}\n")
                fasta.flush()

                # run all predictors (possibly in parallel)
                def _run(p):
                    return p.name, p.predict(fasta.name)

                if self.max_workers > 1:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                        results = list(ex.map(_run, self.predictors))
                else:
                    results = [_run(p) for p in self.predictors]

            # fill results into df
            for name, counts in results:
                col = f"{name}_tm_count"
                for sid, ct in counts.items():
                    df.loc[df[id_col] == sid, col] = int(ct)

            # 5) Optionally update DB
            if self.use_db:
                updates = []
                for _, row in batch.iterrows():
                    vals = [int(row[f"{p.name}_tm_count"]) for p in self.predictors]
                    vals.append(row[id_col])
                    updates.append(tuple(vals))

                cols = [sql.Identifier(seq_col)] + [
                    sql.Identifier(f"{p.name}_tm_count") for p in self.predictors
                ]
                set_clause = sql.SQL(", ").join(
                    sql.SQL("{} = %s").format(c) for c in cols
                )
                query = sql.SQL(
                    "UPDATE {table} SET {sets} WHERE {id_col} = %s"
                ).format(
                    table=sql.Identifier(self.table),
                    sets=set_clause,
                    id_col=sql.Identifier(id_col)
                )
                cur.executemany(query.as_string(conn), updates)
                conn.commit()

        # 6) Cleanup DB
        if self.use_db:
            cur.close()
            conn.close()

        # 7) Optionally write CSV
        if self.write_csv and csv_out:
            df.to_csv(csv_out, index=False)

        return df
    

# class MultiModelAnalyzer:
#     """
#     Runs predictors on a DataFrame, updates Postgres in batches, and writes CSV.
#     """
#     def __init__(self,
#                  db_params: dict,
#                  table: str,
#                  batch_size: int = 10,
#                  max_workers: int = 1):
#         self.predictors: list[BasePredictor] = []
#         self.db_params = db_params
#         self.table = table
#         self.batch_size = batch_size
#         self.max_workers = max_workers

#     def register(self, p: BasePredictor):
#         self.predictors.append(p)

#     def analyze(self,
#                 df: pd.DataFrame,
#                 id_col: str,
#                 seq_col: str,
#                 csv_out: str) -> pd.DataFrame:

#         # 1) Reset index so .loc works predictably
#         df = df.reset_index(drop=True)
#         total = len(df)

#         # 2) Add any missing TM count columns
#         for p in self.predictors:
#             df[f"{p.name}_tm_count"] = pd.NA

#         # 3) Open one DB connection & create missing columns
#         conn = psycopg2.connect(**self.db_params)
#         cur = conn.cursor()

#         # sequence_sequence column (TEXT)
#         cur.execute(sql.SQL(
#             "ALTER TABLE {tbl} "
#             "ADD COLUMN IF NOT EXISTS {seq_col} TEXT"
#         ).format(
#             tbl=sql.Identifier(self.table),
#             seq_col=sql.Identifier(seq_col),
#         ))
        
#         for p in self.predictors:
#             col = f"{p.name}_tm_count"
#             cur.execute(sql.SQL(
#                 "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} INTEGER"
#             ).format(
#                 table=sql.Identifier(self.table),
#                 col=sql.Identifier(col),
#             ))
#         conn.commit()

#         # 4) Process in fixed-size batches
#         batch_num = 0
#         for start in range(0, total, self.batch_size):
#             batch_num += 1
#             end = min(start + self.batch_size, total)
#             batch = df.iloc[start:end]

#             # 4a) Write FASTA to a NamedTemporaryFile
#             with NamedTemporaryFile("w+", suffix=".fasta", delete=True) as fasta:
#                 for _, row in batch.iterrows():
#                     fasta.write(f">{row[id_col]}\n{row[seq_col]}\n")
#                 fasta.flush()

#                 # 4b) Run predictors (optionally in parallel)
#                 def _run(predictor):
#                     return predictor.name, predictor.predict(fasta.name)

#                 if self.max_workers > 1:
#                     with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
#                         results = list(ex.map(_run, self.predictors))
#                 else:
#                     results = [_run(p) for p in self.predictors]

#             # 4c) Populate counts back into df
#             for name, counts in results:
#                 col = f"{name}_tm_count"
#                 for sid, ct in counts.items():
#                     df.loc[df[id_col] == sid, col] = int(ct)

#             # 4d) Bulk-update Postgres for this batch
#             updates = []
#             for _, row in batch.iterrows():
#                 vals = [int(row[f"{p.name}_tm_count"]) for p in self.predictors]
#                 vals.append(row[id_col])
#                 updates.append(tuple(vals))

#             cols = [sql.Identifier(seq_col)] + [sql.Identifier(f"{p.name}_tm_count") for p in self.predictors]
#             set_clause = sql.SQL(", ").join(
#                 sql.SQL("{} = %s").format(c) for c in cols
#             )
#             query = sql.SQL(
#                 "UPDATE {table} SET {sets} WHERE {id_col} = %s"
#             ).format(
#                 table=sql.Identifier(self.table),
#                 sets=set_clause,
#                 id_col=sql.Identifier(id_col)
#             )
#             cur.executemany(query.as_string(conn), updates)
#             conn.commit()

#             print(f"Processed batch #{batch_num}: rows {start+1}-{end}")

#         # 5) Cleanup DB connection
#         cur.close()
#         conn.close()

#         # 6) Write out the full DataFrame (with TM counts) to CSV
#         df.to_csv(csv_out, index=False)
#         print(f"Saved CSV: {csv_out} ({len(df)} rows)")
#         return df
