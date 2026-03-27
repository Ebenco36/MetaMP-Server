from __future__ import annotations

import shutil

import pandas as pd

from src.ingestion.context import DatasetSource, IngestionContext
from src.ingestion.http_client import build_retrying_session, default_timeout


class OpmDatasetSource(DatasetSource):
    def __init__(self, batch_size: int = 50, freshness_days: int = 30):
        self.batch_size = batch_size
        self.freshness_days = freshness_days
        self.session = build_retrying_session(
            headers={
                "Accept": "application/json, text/plain, */*",
                "Origin": "https://opm.phar.umich.edu",
                "Referer": "https://opm.phar.umich.edu/",
            }
        )
        self.host = "https://opm-back.cc.lehigh.edu/opm-backend"

    def run(self, context: IngestionContext):
        existing_dataset = self._reuse_recent_artifacts_if_available(context)
        if existing_dataset is not None:
            return existing_dataset

        codes = self._load_codes(context)
        context.report(f"[opm] Loaded {len(codes)} PDB code(s) from quantitative dataset")
        batch_files = self._fetch_batches(context, codes)
        context.report(f"[opm] Merging {len(batch_files)} batch file(s)")
        master_df = self._merge_batches(context, batch_files)
        context.report(f"[opm] Expanding taxonomy fields for {len(master_df)} row(s)")
        master_df = self._expand_taxonomy(master_df)
        context.report(
            f"[opm] Writing OPM dataset outputs with {len(master_df)} row(s)"
        )
        master_df.to_csv(context.layout.opm_dataset(context.run_date), index=False)
        master_df.to_csv(context.layout.opm_dataset_current, index=False)
        return master_df

    def _reuse_recent_artifacts_if_available(self, context: IngestionContext):
        latest_dataset = context.layout.latest_recent_dated_path(
            "NEWOPM",
            ".csv",
            self.freshness_days,
        )
        current_dataset_path = context.layout.opm_dataset_current

        if latest_dataset is None:
            return None
        latest_run_date, dated_dataset_path = latest_dataset

        if not context.layout.opm_dataset(context.run_date).exists():
            shutil.copyfile(dated_dataset_path, context.layout.opm_dataset(context.run_date))

        shutil.copyfile(dated_dataset_path, current_dataset_path)

        context.report(
            f"[opm] Reusing artifact set from {latest_run_date}; skipping source fetch. "
            f"Adjust INGESTION_ARTIFACT_FRESHNESS_DAYS or delete {dated_dataset_path.name} "
            "and dated batch files under datasets/OPM to rerun this stage."
        )
        return pd.read_csv(current_dataset_path, low_memory=False)

    def _load_codes(self, context: IngestionContext):
        # OPM coverage should be driven by the curated MPStruc anchor set, not by the
        # downstream quantitative merge output, which can be narrower if upstream
        # artifacts are stale or incomplete. Fall back only when the MPStruc identifier
        # snapshots are unavailable.
        if context.layout.mpstruc_ids_current.exists():
            ids = pd.read_csv(context.layout.mpstruc_ids_current, low_memory=False)
            if not ids.empty:
                first_column = ids.columns[0]
                return (
                    ids[first_column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .drop_duplicates()
                    .tolist()
                )

        if context.layout.mpstruc_dataset_current.exists():
            dataset = pd.read_csv(
                context.layout.mpstruc_dataset_current,
                usecols=["Pdb Code"],
                low_memory=False,
            )
            return (
                dataset["Pdb Code"]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .drop_duplicates()
                .tolist()
            )

        dataset = pd.read_csv(
            context.layout.quantitative_dataset_current,
            usecols=["Pdb Code"],
            low_memory=False,
        )
        return (
            dataset["Pdb Code"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .drop_duplicates()
            .tolist()
        )

    def _fetch_batches(self, context: IngestionContext, codes):
        batch_files = []
        total_batches = max(1, (len(codes) + self.batch_size - 1) // self.batch_size)
        for batch_index, start in enumerate(
            range(0, len(codes), self.batch_size), start=1
        ):
            batch_codes = codes[start : start + self.batch_size]
            batch_path = (
                context.layout.opm_batch_dir
                / f"NEWOPM_{context.run_date}_batch{batch_index}.csv"
            )
            batch_files.append(batch_path)
            if batch_path.exists():
                context.report(
                    f"[opm] Reusing batch {batch_index}/{total_batches} from {batch_path.name}"
                )
                continue

            context.report(
                f"[opm] Fetching batch {batch_index}/{total_batches} with {len(batch_codes)} code(s)"
            )
            rows = []
            missing_count = 0
            for pdb_code in batch_codes:
                record = self._fetch_record(pdb_code)
                if record is not None:
                    rows.append(record)
                else:
                    missing_count += 1

            if rows:
                pd.concat(rows, ignore_index=True).to_csv(batch_path, index=False)
                context.report(
                    "[opm] Completed batch "
                    f"{batch_index}/{total_batches} with {len(rows)} record(s) "
                    f"and {missing_count} missing code(s)"
                )
            else:
                context.report(
                    f"[opm] Batch {batch_index}/{total_batches} produced no records"
                )
        return batch_files

    def _fetch_record(self, pdb_code):
        index_response = self.session.get(
            f"{self.host}/primary_structures",
            params={"search": pdb_code, "pageSize": 1},
            timeout=default_timeout(),
        )
        if index_response.status_code == 404:
            return None
        index_response.raise_for_status()

        objects = index_response.json().get("objects") or []
        if not objects:
            return None

        record_id = objects[0].get("id")
        detail_response = self.session.get(
            f"{self.host}/primary_structures/{record_id}",
            timeout=default_timeout(),
        )
        if detail_response.status_code == 404:
            return None
        detail_response.raise_for_status()

        data_frame = pd.json_normalize(detail_response.json(), sep="_")
        data_frame["pdb_code"] = pdb_code
        return data_frame

    def _merge_batches(self, context: IngestionContext, batch_files):
        data_frames = [
            pd.read_csv(batch_file, low_memory=False)
            for batch_file in batch_files
            if batch_file.exists()
        ]
        if not data_frames:
            raise FileNotFoundError("No OPM batch files were produced.")
        merged = pd.concat(data_frames, ignore_index=True)
        context.report(f"[opm] Merged batch files into {len(merged)} total row(s)")
        return merged

    def _expand_taxonomy(self, data_frame):
        taxonomic_levels = [
            "Domain",
            "Kingdom",
            "Subkingdom",
            "Superphylum",
            "Phylum",
            "Subphylum",
            "Superclass",
            "Class",
            "Subclass",
            "Infraclass",
            "Superorder",
            "Order",
            "Suborder",
            "Infraorder",
            "Family",
            "Subfamily",
        ]

        if "species_description" not in data_frame.columns:
            return data_frame

        if any(column in data_frame.columns for column in taxonomic_levels):
            return data_frame

        split_df = data_frame["species_description"].str.split(
            ",", expand=True, n=len(taxonomic_levels) - 1
        )
        split_df.columns = taxonomic_levels
        return pd.concat([data_frame, split_df], axis=1)
