from __future__ import annotations

import re
import shutil

import pandas as pd

from src.ingestion.context import IngestionContext
from src.ingestion.pdb_source import preprocess_str_data, remove_html_tags


CAT_COLUMNS = [
    "Group",
    "Subgroup",
    "Species",
    "Taxonomic Domain",
    "Expressed in Species",
    "symmetry_space_group_name_hm",
    "rcsb_entry_info_structure_determination_methodology",
    "rcsb_entry_info_diffrn_resolution_high_provenance_source",
    "rcsb_entry_info_selected_polymer_entity_types",
    "rcsb_entry_info_experimental_method",
    "rcsb_entry_info_na_polymer_entity_types",
    "rcsb_entry_info_polymer_composition",
    "exptl_method",
    "exptl_crystal_grow_method",
]

SPELLING_CORRECTIONS = {
    "Methanocaldococcus jannaschi": "Methanocaldococcus jannaschii",
    "Rhodopeudomonas blastica": "Rhodopseudomonas blastica",
    "Shewanella oneidensi": "Shewanella oneidensis",
    "Synechocystis sp. pcc 6803": "Synechocystis sp. PCC6803",
    "Escherichia coli": "E. Coli",
    "Escherichia coli B": "E. Coli",
    "E. coli": "E. Coli",
    "e. coli": "E. Coli",
    "E.coli": "E. Coli",
    "HEK 293S cells": "HEK-293S cells",
    "HEK293S cells": "HEK-293S cells",
    "HEK293S": "HEK-293S cells",
    "HEK293": "HEK293 cells",
    "HEK 293": "HEK293 cells",
    "Homo sapiens": "Homo Sapiens",
    "homo sapiens": "Homo Sapiens",
    "Bacillus anthraciss": "Bacillus anthracis",
    "Bascillus subtilis": "Bacillus subtilis",
    "NMR": "NMR Structure",
    "NMR structure": "NMR Structure",
}

CRYSTAL_METHOD_CORRECTIONS = {
    "VAPOUR DIFFUSION": "VAPOR DIFFUSION",
    "LCP": "LIPIDIC CUBIC PHASE",
    "LIPID CUBIC PHASE": "LIPIDIC CUBIC PHASE",
    "LIPIDIC CUBIC PHASE (LCP)": "LIPIDIC CUBIC PHASE",
    "CUBIC LIPID PHASE": "LIPIDIC CUBIC PHASE",
    "BATCH": "BATCH MODE",
    "BATCH METHOD": "BATCH MODE",
    "MICRO-BATCH METHOD UNDER OIL": "MICROBATCH",
    "MICROBATCH-UNDER-OIL": "MICROBATCH",
}


def extract_year(bibliography):
    match = re.search(r"\['year', '(\d{4})'\]", str(bibliography))
    return match.group(1) if match else None


class QuantitativeDatasetBuilder:
    def __init__(self, freshness_days: int = 30):
        self.freshness_days = freshness_days

    def run(self, context: IngestionContext):
        self._context = context
        existing_dataset = self._reuse_recent_artifacts_if_available(context)
        if existing_dataset is not None:
            return existing_dataset

        context.report("[quantitative] Loading current MPStruc dataset")
        mpstruc = pd.read_csv(context.layout.mpstruc_dataset_current, low_memory=False)
        mpstruc = self._drop_duplicate_rows(
            mpstruc,
            ["Pdb Code", "Group", "Subgroup"],
            "MPStruc source",
        )
        context.report("[quantitative] Loading current PDB dataset")
        pdb_dataset = pd.read_csv(context.layout.pdb_dataset_current, low_memory=False)
        pdb_dataset = self._drop_duplicate_rows(
            pdb_dataset,
            ["Pdb Code", "pdb_code"],
            "PDB source",
        )

        context.report(
            f"[quantitative] Merging {len(mpstruc)} MPStruc row(s) with {len(pdb_dataset)} PDB row(s)"
        )
        enriched = pd.merge(mpstruc, pdb_dataset, on="Pdb Code")
        enriched = self._drop_duplicate_rows(
            enriched,
            ["Pdb Code", "Group", "Subgroup"],
            "enriched quantitative source",
        )
        context.report(
            f"[quantitative] Writing enriched dataset with {len(enriched)} row(s)"
        )
        enriched.to_csv(context.layout.enriched_dataset_current, index=False)

        context.report("[quantitative] Building normalized quantitative dataset")
        quantitative = self._build_quantitative_frame(enriched)
        quantitative = self._drop_duplicate_rows(
            quantitative,
            ["Pdb Code", "Group", "Subgroup"],
            "quantitative dataset",
        )
        context.report(
            f"[quantitative] Writing quantitative dataset with {len(quantitative)} row(s)"
        )
        quantitative.to_csv(
            context.layout.quantitative_dataset(context.run_date), index=False
        )
        quantitative.to_csv(
            context.layout.quantitative_dataset_current, index=False
        )
        return quantitative

    def _reuse_recent_artifacts_if_available(self, context: IngestionContext):
        latest_dataset = context.layout.latest_recent_dated_path(
            "Quantitative_data",
            ".csv",
            self.freshness_days,
        )
        current_quantitative_path = context.layout.quantitative_dataset_current

        if latest_dataset is None:
            return None
        latest_run_date, dated_quantitative_path = latest_dataset

        if not context.layout.quantitative_dataset(context.run_date).exists():
            shutil.copyfile(
                dated_quantitative_path,
                context.layout.quantitative_dataset(context.run_date),
            )

        shutil.copyfile(dated_quantitative_path, current_quantitative_path)

        context.report(
            f"[quantitative] Reusing artifact set from {latest_run_date}; skipping rebuild. "
            f"Adjust INGESTION_ARTIFACT_FRESHNESS_DAYS or delete {dated_quantitative_path.name} to rerun this stage."
        )
        reused = pd.read_csv(current_quantitative_path, low_memory=False)
        return self._drop_duplicate_rows(
            reused,
            ["Pdb Code", "Group", "Subgroup"],
            "reused quantitative artifact",
        )

    def _build_quantitative_frame(self, data_frame):
        original_column_count = len(data_frame.columns)
        object_columns = data_frame.select_dtypes(include="object").columns
        data_frame[object_columns] = data_frame[object_columns].applymap(
            remove_html_tags
        )

        normalized_data = []
        for column in data_frame.columns:
            column_data = data_frame[column].apply(preprocess_str_data)
            try:
                normalized_column = pd.json_normalize(column_data, sep="_")
            except Exception:
                continue
            if normalized_column.empty:
                continue
            normalized_column.columns = [
                f"{column}_{column_name}"
                for column_name in normalized_column.columns
            ]
            normalized_data.append(normalized_column)

        merged_df = pd.concat([data_frame] + normalized_data, axis=1)
        merged_df.columns = merged_df.columns.str.replace(".", "_", regex=False)
        added_columns = len(merged_df.columns) - original_column_count

        if "Bibliography" in merged_df.columns:
            merged_df["bibliography_year"] = merged_df["Bibliography"].apply(
                extract_year
            )

        if "Group" in merged_df.columns:
            merged_df.loc[
                merged_df["Group"] == "TRANSMEMBRANE PROTEINS: BETA-BARREL",
                "Group",
            ] = "TRANSMEMBRANE PROTEINS:BETA-BARREL"
            merged_df.loc[
                merged_df["Group"] == "TRANSMEMBRANE PROTEINS: ALPHA-HELICAL",
                "Group",
            ] = "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL"

        self._context.report(
            f"[quantitative] Expanded dataset with {added_columns} normalized column(s)"
        )
        return self._preprocess_for_analysis(merged_df)

    def _drop_duplicate_rows(self, data_frame, candidate_columns, label):
        key_fields = [column for column in candidate_columns if column in data_frame.columns]
        if not key_fields:
            return data_frame

        normalized = data_frame.copy()
        for column in key_fields:
            normalized[column] = normalized[column].fillna("").astype(str).str.strip()

        duplicate_count = int(normalized.duplicated(subset=key_fields).sum())
        if duplicate_count > 0:
            self._context.report(
                f"[quantitative] Removed {duplicate_count} duplicate row(s) from {label} using key {key_fields}"
            )
        keep_mask = ~normalized.duplicated(subset=key_fields, keep="first")
        return data_frame.loc[keep_mask].reset_index(drop=True)

    def _preprocess_for_analysis(self, data_frame):
        self._context.report("[quantitative] Applying preprocessing and value cleanup")
        existing_cat_columns = [
            column for column in CAT_COLUMNS if column in data_frame.columns
        ]
        if existing_cat_columns:
            data_frame[existing_cat_columns] = data_frame[existing_cat_columns].apply(
                lambda series: series.astype(str).str.strip()
            )

        if "exptl_crystal_grow_method" in data_frame.columns:
            data_frame["exptl_crystal_grow_method"] = (
                data_frame["exptl_crystal_grow_method"].astype(str).str.upper()
            )

        replace_map = {
            column: SPELLING_CORRECTIONS
            for column in ["Expressed in Species", "Species", "Resolution"]
            if column in data_frame.columns
        }
        if "exptl_crystal_grow_method" in data_frame.columns:
            replace_map["exptl_crystal_grow_method"] = CRYSTAL_METHOD_CORRECTIONS
        if replace_map:
            data_frame.replace(replace_map, inplace=True)

        if "exptl_crystal_grow_method" in data_frame.columns:
            split_columns = data_frame["exptl_crystal_grow_method"].str.split(
                ",", n=2, expand=True
            )
            split_columns = split_columns.reindex(columns=range(3))
            split_columns.columns = [
                "exptl_crystal_grow_method1",
                "exptl_crystal_grow_method2",
                "not_useful",
            ]
            data_frame = pd.concat([data_frame, split_columns], axis=1)
            data_frame = data_frame.apply(self._reorder_methods, axis=1)

        if "rcsb_entry_info_resolution_combined" in data_frame.columns:
            data_frame["processed_resolution"] = data_frame[
                "rcsb_entry_info_resolution_combined"
            ].astype(str).str.extract(r"\[(\d+\.\d+)\]", expand=False)

        return data_frame

    def _reorder_methods(self, row):
        second_method = row.get("exptl_crystal_grow_method2")
        first_method = row.get("exptl_crystal_grow_method1")
        if (
            pd.notna(second_method)
            and second_method in ["VAPOR DIFFUSION", "VAPOUR DIFFUSION"]
            and first_method not in ["VAPOR DIFFUSION", "VAPOUR DIFFUSION"]
        ):
            row["exptl_crystal_grow_method1"], row["exptl_crystal_grow_method2"] = (
                row["exptl_crystal_grow_method2"],
                row["exptl_crystal_grow_method1"],
            )
        return row
