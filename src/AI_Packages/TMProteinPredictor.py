import subprocess
import sys
import re
import os
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import pandas as pd
import psycopg2
from psycopg2 import sql
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor


def extract_tmr_counts_from_gff_text(gff_text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in re.finditer(r"^#\s*(\S+).*Number of predicted TMRs:\s*(\d+)", gff_text, re.MULTILINE):
        seq, num = m.group(1), int(m.group(2))
        counts[seq] = num
    if counts:
        return counts
    for line in gff_text.splitlines():
        parts = line.split("\t")
        if len(parts) > 2 and parts[2] == "TMhelix":
            seq = parts[0]
            counts[seq] = counts.get(seq, 0) + 1
    return counts


def run_tmbed_predict(fasta: str, out_pred: str, format_code: int = 0, use_gpu: bool = True) -> bool:
    cmd = [
        sys.executable, "-m", "tmbed", "predict",
        "-f", fasta,
        "-p", out_pred,
        "--out-format", str(format_code)
    ]
    if use_gpu:
        cmd.append("--use-gpu")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(res.stdout, end="")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[TMbed] ERROR:\n{e.stdout}\n{e.stderr}", file=sys.stderr)
        return False


def count_tmbed_3line(pred_path: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(pred_path) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    if len(lines) % 3 != 0:
        raise ValueError("TMbed 3-line output malformed.")
    for i in range(0, len(lines), 3):
        hdr, labels = lines[i], lines[i + 2]
        seq = hdr.lstrip('>')
        runs = re.findall(r"[HhBb]+", labels)
        counts[seq] = len(runs)
    return counts


class BasePredictor(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(self, fasta_path: str) -> dict[str, int]:
        pass


class TMbedPredictor(BasePredictor):
    def __init__(self, format_code: int = 0, use_gpu: bool = True):
        super().__init__("TMbed")
        self.format_code = format_code
        self.use_gpu = use_gpu

    def predict(self, fasta_path: str) -> dict[str, int]:
        pred_path = fasta_path + ".pred"
        if not run_tmbed_predict(fasta_path, pred_path, self.format_code, self.use_gpu):
            return {}
        counts = count_tmbed_3line(pred_path)
        os.remove(pred_path)
        return counts


class DeepTMHMMPredictor(BasePredictor):
    def __init__(self, spec: str = "DTU/DeepTMHMM:1.0.24"):
        super().__init__("DeepTMHMM")
        try:
            import biolib
        except ImportError:
            raise RuntimeError("Install biolib for DeepTMHMMPredictor")
        self.app = biolib.load(spec)

    def predict(self, fasta_path: str) -> dict[str, int]:
        job = self.app.run(fasta=fasta_path)

        print(f"[DeepTMHMM] Running job {job.id}, waiting for output...")

        # This blocks until the file is ready or the job fails internally
        fh = job.get_output_file('/deeptmhmm_results.md').get_file_handle()
        fh.seek(0)
        txt = fh.read().decode()

        print(f"[DeepTMHMM] Job {job.id} completed successfully.")

        return extract_tmr_counts_from_gff_text(txt)


class MultiModelAnalyzer:
    def __init__(self,
                 db_params: dict,
                 table: str,
                 batch_size: int = 10,
                 max_workers: int = 1,
                 max_sequences: int = 4,
                 use_db: bool = True,
                 write_csv: bool = False):
        self.predictors: list[BasePredictor] = []
        self.db_params = db_params
        self.table = table
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_sequences = max_sequences
        self.use_db = use_db
        self.write_csv = write_csv

    def register(self, predictor):
        self.predictors.append(predictor)

    def analyze(self,
        df: pd.DataFrame,
        id_col: str,
        seq_col: str,
        csv_out: Optional[str] = None,
        resume_from_csv: Optional[str] = None) -> pd.DataFrame:
        df = df.reset_index(drop=True)

        if resume_from_csv and os.path.exists(resume_from_csv):
            done_df = pd.read_csv(resume_from_csv)
            done_ids = set(done_df[id_col])
            df = df[~df[id_col].isin(done_ids)]
            print(f"[RESUME] Skipping {len(done_ids)} already processed entries.")
        else:
            done_df = pd.DataFrame()

        total = len(df)
        if total == 0:
            print("All sequences are already processed.")
            return done_df

        print(f"🚀 Starting analysis of {total} sequences in batches of {self.batch_size}...")

        for p in self.predictors:
            df[f"{p.name}_tm_count"] = pd.NA

        if self.use_db:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            cur.execute(sql.SQL(
                "ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {seq_col} TEXT"
            ).format(tbl=sql.Identifier(self.table), seq_col=sql.Identifier(seq_col)))

            for p in self.predictors:
                cur.execute(sql.SQL(
                    "ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} INTEGER"
                ).format(tbl=sql.Identifier(self.table), col=sql.Identifier(f"{p.name}_tm_count")))

            conn.commit()
        else:
            conn = cur = None

        results = []

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = df.iloc[start:end]
            print(f"\n🔄 Batch {start}-{end}...")

            try:
                # 1. Store sequences
                if self.use_db:
                    update_seqs = [(row[seq_col], row[id_col]) for _, row in batch.iterrows()]
                    cur.executemany(
                        sql.SQL("UPDATE {tbl} SET {seq_col} = %s WHERE {id_col} = %s").format(
                            tbl=sql.Identifier(self.table),
                            seq_col=sql.Identifier(seq_col),
                            id_col=sql.Identifier(id_col)
                        ).as_string(conn),
                        update_seqs
                    )
                    conn.commit()
                    print("Sequences committed to DB")

                # 2. Write FASTA
                with NamedTemporaryFile("w+", suffix=".fasta", delete=True) as fasta:
                    for _, row in batch.iterrows():
                        fasta.write(f">{row[id_col]}\n{row[seq_col]}\n")
                    fasta.flush()

                    def _run(p):
                        try:
                            print(f"🔍 Running predictor: {p.name}")
                            result = p.predict(fasta.name)
                            print(f"{p.name} completed.")
                            return p.name, result
                        except Exception as e:
                            traceback.print_exc()
                            print(f"{p.name} failed: {e}")
                            return p.name, {}

                    results_map = (
                        list(ThreadPoolExecutor(max_workers=self.max_workers).map(_run, self.predictors))
                        if self.max_workers > 1 else
                        [_run(p) for p in self.predictors]
                    )

                    for name, counts in results_map:
                        col = f"{name}_tm_count"
                        for sid, ct in counts.items():
                            batch.loc[batch[id_col] == sid, col] = int(ct)

                # 3. Store predictions
                if self.use_db:
                    updates = []
                    for _, row in batch.iterrows():
                        values = [int(row[f"{p.name}_tm_count"]) for p in self.predictors]
                        values.append(row[id_col])
                        updates.append(tuple(values))

                    set_clause = sql.SQL(", ").join(
                        sql.SQL("{} = %s").format(sql.Identifier(f"{p.name}_tm_count"))
                        for p in self.predictors
                    )
                    update_query = sql.SQL("UPDATE {tbl} SET {sets} WHERE {id_col} = %s").format(
                        tbl=sql.Identifier(self.table),
                        sets=set_clause,
                        id_col=sql.Identifier(id_col)
                    )
                    cur.executemany(update_query.as_string(conn), updates)
                    conn.commit()
                    print("📥 TM count updates committed to DB")

                results.append(batch)

                # 4. Save interim results
                combined = pd.concat([done_df] + results, ignore_index=True)
                if self.write_csv and csv_out:
                    combined.to_csv(csv_out, index=False)
                    print(f"💾 Batch {start}-{end} progress saved to {csv_out}")

            except Exception as e:
                traceback.print_exc()
                print(f"❌ Batch {start}-{end} failed: {e}")
                continue

        if self.use_db:
            cur.close()
            conn.close()

        final_df = pd.concat([done_df] + results, ignore_index=True)
        if self.write_csv and csv_out:
            final_df.to_csv(csv_out, index=False)

        print("Finished all batches.")
        return final_df

    

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
