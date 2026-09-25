from __future__ import annotations

import ast
import json
import re
import shutil

import pandas as pd
from bs4 import BeautifulSoup

from src.ingestion.context import DatasetSource, IngestionContext
from src.ingestion.http_client import build_retrying_session, default_timeout


def preprocess_str_data(str_value):
    if not isinstance(str_value, str):
        return {}

    try:
        parsed = ast.literal_eval(str_value)
    except (SyntaxError, ValueError):
        return {}

    if isinstance(parsed, list):
        if not parsed:
            return {}
        first_item = parsed[0]
        return first_item if isinstance(first_item, dict) else {}

    return parsed if isinstance(parsed, dict) else {}


def remove_html_tags(text):
    if text is None or pd.isna(text):
        return ""
    clean_text = re.sub(r"<.*?>", "", str(text))
    return clean_text.replace("\r", " ").replace("\n", " ")


class PdbDatasetSource(DatasetSource):
    def __init__(self, batch_size: int = 200, freshness_days: int = 30):
        self.batch_size = batch_size
        self.freshness_days = freshness_days
        self.session = build_retrying_session()

    def run(self, context: IngestionContext):
        existing_dataset = self._reuse_recent_artifacts_if_available(context)
        if existing_dataset is not None:
            return existing_dataset

        ids = self._load_ids(context)
        if not ids:
            raise FileNotFoundError("No MPStruc ids found for PDB ingestion.")

        context.report(f"[pdb] Loaded {len(ids)} MPStruc identifier(s)")
        batch_files = self._fetch_batches(context, ids)
        context.report(f"[pdb] Merging {len(batch_files)} batch file(s)")
        master_df = self._merge_batches(context, batch_files)
        context.report(
            f"[pdb] Writing raw merged dataset to {context.layout.pdb_dataset_current.name}"
        )
        self._write_current_copy(context, master_df, context.layout.pdb_dataset_current)

        context.report(
            f"[pdb] Transforming {len(master_df)} raw PDB record(s) into flattened dataset"
        )
        transformed_df = self._transform(master_df)
        transformed_path = context.layout.pdb_transformed_dataset(context.run_date)
        context.report(f"[pdb] Writing transformed dataset to {transformed_path.name}")
        transformed_df.to_csv(transformed_path, index=False)
        self._write_current_copy(
            context,
            transformed_df,
            context.layout.pdb_transformed_dataset_current,
        )
        return transformed_df

    def _reuse_recent_artifacts_if_available(self, context: IngestionContext):
        latest_raw = context.layout.latest_recent_dated_path(
            "PDB_data",
            ".csv",
            self.freshness_days,
        )
        latest_transformed = context.layout.latest_recent_dated_path(
            "PDB_data_transformed",
            ".csv",
            self.freshness_days,
        )
        current_raw_path = context.layout.pdb_dataset_current
        current_transformed_path = context.layout.pdb_transformed_dataset_current

        if latest_raw is None or latest_transformed is None:
            return None
        raw_run_date, dated_raw_path = latest_raw
        transformed_run_date, dated_transformed_path = latest_transformed
        if raw_run_date != transformed_run_date:
            return None

        if not context.layout.pdb_dataset(context.run_date).exists():
            shutil.copyfile(dated_raw_path, context.layout.pdb_dataset(context.run_date))
        if not context.layout.pdb_transformed_dataset(context.run_date).exists():
            shutil.copyfile(
                dated_transformed_path,
                context.layout.pdb_transformed_dataset(context.run_date),
            )

        shutil.copyfile(dated_raw_path, current_raw_path)
        shutil.copyfile(dated_transformed_path, current_transformed_path)

        context.report(
            f"[pdb] Reusing artifact set from {raw_run_date}; skipping source fetch. "
            f"Adjust INGESTION_ARTIFACT_FRESHNESS_DAYS or delete {dated_raw_path.name}, "
            f"{dated_transformed_path.name}, and dated batch files under datasets/PDB to rerun this stage."
        )
        return pd.read_csv(current_transformed_path, low_memory=False)

    def _load_ids(self, context: IngestionContext):
        ids_df = pd.read_csv(context.layout.mpstruc_ids_current)
        return ids_df["Pdb Code"].dropna().astype(str).tolist()

    def _fetch_batches(self, context: IngestionContext, ids):
        batch_files = []
        total_batches = max(1, (len(ids) + self.batch_size - 1) // self.batch_size)
        for batch_index, start in enumerate(
            range(0, len(ids), self.batch_size), start=1
        ):
            batch_ids = ids[start : start + self.batch_size]
            batch_path = (
                context.layout.pdb_batch_dir
                / f"PDB_data_{context.run_date}_batch{batch_index}.csv"
            )
            batch_files.append(batch_path)
            if batch_path.exists():
                context.report(
                    f"[pdb] Reusing batch {batch_index}/{total_batches} from {batch_path.name}"
                )
                continue

            context.report(
                f"[pdb] Fetching batch {batch_index}/{total_batches} with {len(batch_ids)} identifier(s)"
            )
            rows = []
            missing_count = 0
            replaced_count = 0
            for pdb_code in batch_ids:
                record, record_meta = self._read_entry(context, pdb_code)
                if record is not None:
                    rows.append(record)
                if record_meta["missing"]:
                    missing_count += 1
                if record_meta["replaced"]:
                    replaced_count += 1

            if rows:
                pd.concat(rows, ignore_index=True).to_csv(batch_path, index=False)
                context.report(
                    "[pdb] Completed batch "
                    f"{batch_index}/{total_batches} with {len(rows)} record(s), "
                    f"{replaced_count} replacement(s), {missing_count} missing record(s)"
                )
            else:
                context.report(
                    f"[pdb] Batch {batch_index}/{total_batches} produced no records"
                )

        return batch_files

    def _read_entry(self, context: IngestionContext, pdb_code):
        record = self._fetch_rcsb_record(pdb_code)
        replacement_code = ""
        is_replaced = "Not Replaced"
        metadata = {"missing": False, "replaced": False}

        if record is None:
            context.report(f"[pdb] No direct RCSB entry for {pdb_code}; checking replacement history")
            candidate = self._resolve_replacement_code(pdb_code)
            if candidate and candidate != pdb_code:
                replacement_code = candidate
                is_replaced = "Replaced"
                metadata["replaced"] = True
                context.report(f"[pdb] Using replacement entry {candidate} for removed code {pdb_code}")
                record = self._fetch_rcsb_record(candidate)

        if record is None:
            metadata["missing"] = True
            context.report(f"[pdb] No PDB record available for {pdb_code}")
            return None, metadata

        data_frame = pd.json_normalize(record, sep="_")
        canonical_pdb_code = replacement_code or pdb_code
        data_frame.insert(1, "Pdb Code", pdb_code)
        data_frame.insert(2, "Is Replaced", is_replaced)
        data_frame.insert(
            3,
            "PDB Code Changed",
            f"{pdb_code}->{replacement_code}" if replacement_code else "",
        )
        data_frame.insert(4, "replacement_pdb_code", replacement_code)
        data_frame.insert(5, "canonical_pdb_code", canonical_pdb_code)
        data_frame.insert(6, "uniprot_id", self._fetch_uniprot_id(pdb_code))
        return data_frame, metadata

    def _fetch_rcsb_record(self, pdb_code):
        response = self.session.get(
            f"https://data.rcsb.org/rest/v1/core/entry/{pdb_code}",
            timeout=default_timeout(),
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def _resolve_replacement_code(self, pdb_code):
        response = self.session.get(
            f"https://www.rcsb.org/structure/removed/{pdb_code}",
            timeout=default_timeout(),
        )
        if response.status_code == 404:
            return pdb_code
        response.raise_for_status()

        match = re.search(r"replaced(.{1,100})", response.text, re.IGNORECASE)
        if not match:
            return pdb_code

        replacement_html = match.group(1)
        soup = BeautifulSoup(replacement_html, "html.parser")
        link = soup.find("a", href=re.compile(r"/structure/(\w+)"))
        if link:
            return link["href"].split("/")[-1]

        fallback = re.search(r"\b\d[A-Za-z]{3}\b", replacement_html)
        return fallback.group(0) if fallback else pdb_code

    def _fetch_uniprot_id(self, pdb_code):
        response = self.session.get(
            f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_code}",
            timeout=default_timeout(),
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()

        try:
            data = response.json()
            return list(
                data.get(pdb_code.lower(), {"UniProt": {}})["UniProt"].keys()
            )[0]
        except (IndexError, KeyError, json.JSONDecodeError):
            return None

    def _merge_batches(self, context: IngestionContext, batch_files):
        data_frames = [
            pd.read_csv(batch_file, low_memory=False)
            for batch_file in batch_files
            if batch_file.exists()
        ]
        if not data_frames:
            raise FileNotFoundError("No PDB batch files were produced.")

        master_df = pd.concat(data_frames, ignore_index=True)
        context.report(f"[pdb] Writing dated raw dataset with {len(master_df)} row(s)")
        master_df.to_csv(
            context.layout.pdb_dataset(context.run_date), index=False
        )
        return master_df

    def _transform(self, data_frame):
        object_columns = data_frame.select_dtypes(include="object").columns
        data_frame[object_columns] = data_frame[object_columns].applymap(
            remove_html_tags
        )

        normalized = []
        for column in data_frame.columns:
            column_norm = data_frame[column].apply(preprocess_str_data)
            try:
                normalized_df = pd.json_normalize(column_norm, sep="_")
            except Exception:
                continue
            if normalized_df.empty:
                continue
            normalized_df.columns = [f"{column}_{name}" for name in normalized_df.columns]
            normalized.append(normalized_df)

        merged = pd.concat([data_frame] + normalized, axis=1)
        merged.columns = merged.columns.str.replace(r"\.", "_", regex=True)
        return merged

    def _write_current_copy(self, context: IngestionContext, data_frame, output_path):
        context.layout.ensure_directories()
        data_frame.to_csv(output_path, index=False)
