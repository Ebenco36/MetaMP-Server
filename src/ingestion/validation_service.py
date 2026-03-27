from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.ingestion.dataset_layout import DatasetLayout


class IngestionValidationService:
    def __init__(self):
        self.report_basename = "validation_report"

    def run(self, context):
        layout = context.layout
        report = self.build_report(layout)
        report_path = self.write_report(layout, report, context.run_date)
        context.report(
            f"[validate_release] Wrote validation report to {report_path.name}"
        )
        if not report["passed"]:
            failure_names = ", ".join(
                check["name"] for check in report["checks"] if not check["passed"]
            )
            raise RuntimeError(
                f"Dataset validation failed for: {failure_names}"
            )
        return report

    def build_report(self, layout: DatasetLayout):
        checks = [
            self._check_mpstruct_duplicate_keys(layout),
            self._check_quantitative_duplicates(layout),
            self._check_replacement_resolution(layout),
            self._check_annotation_dataset(layout),
            self._check_uniprot_mapping(layout),
            self._check_known_gene_coverage(layout),
            self._check_row_count_drift(layout),
        ]
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "passed": all(check["passed"] for check in checks),
            "checks": checks,
        }

    def write_report(self, layout: DatasetLayout, report, run_date: str):
        reports_dir = layout.base_dir / "validation"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{self.report_basename}_{run_date}.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        latest_path = reports_dir / f"{self.report_basename}_latest.json"
        latest_path.write_text(json.dumps(report, indent=2, default=str))
        return report_path

    @staticmethod
    def _read_csv(path: Path):
        if path.exists():
            return pd.read_csv(path, low_memory=False)
        return pd.DataFrame()

    def _check_mpstruct_duplicate_keys(self, layout):
        mpstruct = self._read_csv(layout.valid_dir / "Mpstruct_dataset.csv")
        return self._check_duplicate_keys(
            dataframe=mpstruct,
            check_name="mpstruct_duplicate_keys",
            missing_message="MPstruc dataset is missing.",
        )

    def _check_quantitative_duplicates(self, layout):
        quantitative = self._read_csv(layout.valid_dir / "Quantitative_data.csv")
        return self._check_duplicate_keys(
            dataframe=quantitative,
            check_name="quantitative_duplicates",
            missing_message="Quantitative dataset is missing.",
        )

    def _check_duplicate_keys(self, dataframe, check_name, missing_message):
        if dataframe.empty:
            return self._failed(check_name, missing_message)

        key_column_map = {}
        for canonical_name, candidates in {
            "pdb_code": ["pdb_code", "Pdb Code", "PDB Code"],
            "group": ["group", "Group"],
            "subgroup": ["subgroup", "Subgroup"],
        }.items():
            column_name = self._find_column(dataframe, candidates)
            if column_name:
                key_column_map[canonical_name] = column_name

        if "pdb_code" not in key_column_map:
            return self._failed(
                check_name,
                f"{missing_message[:-1]} or has no pdb_code column.",
            )

        normalized = pd.DataFrame(
            {
                canonical_name: dataframe[column_name].fillna("").astype(str).str.strip()
                for canonical_name, column_name in key_column_map.items()
            }
        )

        natural_key_fields = [
            field_name
            for field_name in ["pdb_code", "group", "subgroup"]
            if field_name in normalized.columns
        ]
        duplicate_mask = normalized.duplicated(subset=natural_key_fields, keep=False)
        duplicate_rows = normalized[duplicate_mask]
        duplicate_row_count = int(len(duplicate_rows))
        duplicate_key_count = int(
            duplicate_rows.drop_duplicates(subset=natural_key_fields).shape[0]
        )

        if duplicate_row_count > 0:
            return self._failed(
                check_name,
                "Dataset contains duplicate rows for the natural key.",
                {
                    "row_count": int(len(dataframe)),
                    "natural_key_fields": natural_key_fields,
                    "duplicate_row_count": duplicate_row_count,
                    "duplicate_key_count": duplicate_key_count,
                    "sample_duplicates": duplicate_rows.head(5).to_dict(orient="records"),
                },
            )

        return self._passed(
            check_name,
            {
                "row_count": int(len(dataframe)),
                "natural_key_fields": natural_key_fields,
                "unique_key_count": int(normalized.drop_duplicates(subset=natural_key_fields).shape[0]),
            },
        )

    def _check_replacement_resolution(self, layout):
        pdb_df = self._read_csv(layout.valid_dir / "PDB_data_transformed.csv")
        if pdb_df.empty:
            return self._failed("replacement_resolution", "PDB transformed dataset is missing.")

        unresolved = pd.DataFrame()
        if "is_replaced" in pdb_df.columns:
            replaced_mask = (
                pdb_df["is_replaced"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "replaced", "yes", "y"})
            )
            unresolved = pdb_df[
                replaced_mask
                & (
                    pdb_df.get("replacement_pdb_code", pd.Series(index=pdb_df.index)).fillna("").astype(str).str.strip().eq("")
                    & pdb_df.get("canonical_pdb_code", pd.Series(index=pdb_df.index)).fillna("").astype(str).str.strip().eq("")
                )
            ]
        if not unresolved.empty:
            return self._failed(
                "replacement_resolution",
                "Replaced PDB entries are missing replacement/canonical codes.",
                {
                    "unresolved_count": int(len(unresolved)),
                    "sample_pdb_codes": unresolved["pdb_code"].dropna().astype(str).head(5).tolist()
                    if "pdb_code" in unresolved.columns
                    else [],
                },
            )
        return self._passed("replacement_resolution")

    @staticmethod
    def _find_column(dataframe, candidates):
        for column in candidates:
            if column in dataframe.columns:
                return column
        return None

    def _check_annotation_dataset(self, layout):
        annotation_path = self._resolve_annotation_dataset_path(layout)
        annotation_df = self._read_csv(annotation_path)
        if annotation_df.empty:
            return self._failed("annotation_dataset", "Expert annotation dataset is missing.")

        required_columns = ["PDB Code", "Group (Expert)", "TM (Expert)"]
        missing_columns = [column for column in required_columns if column not in annotation_df.columns]
        if missing_columns:
            return self._failed(
                "annotation_dataset",
                "Expert annotation dataset is missing required columns.",
                {"missing_columns": missing_columns},
            )

        missing_pdb_codes = int(annotation_df["PDB Code"].fillna("").astype(str).str.strip().eq("").sum())
        if missing_pdb_codes > 0:
            return self._failed(
                "annotation_dataset",
                "Expert annotation dataset contains rows without PDB Code.",
                {"missing_pdb_code_rows": missing_pdb_codes},
            )
        return self._passed("annotation_dataset", {"row_count": int(len(annotation_df))})

    @staticmethod
    def _resolve_annotation_dataset_path(layout):
        configured = os.getenv("DASHBOARD_ANNOTATION_DATASET_PATH") or os.getenv(
            "ANNOTATION_DATASET_PATH"
        )
        candidates = []
        if configured:
            candidates.append(Path(configured))
        candidates.extend(
            [
                layout.base_dir / "expert_annotation_predicted.csv",
                Path("/var/app/data/datasets/expert_annotation_predicted.csv"),
                layout.base_dir.parent / "datasets" / "expert_annotation_predicted.csv",
                Path.cwd() / "datasets" / "expert_annotation_predicted.csv",
                Path("/var/app/datasets/expert_annotation_predicted.csv"),
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _check_uniprot_mapping(self, layout):
        uniprot_df = self._read_csv(layout.valid_dir / "Uniprot_functions.csv")
        if uniprot_df.empty:
            return self._failed("uniprot_mapping", "UniProt dataset is missing.")

        if not {"pdb_code", "uniprot_id"}.issubset(uniprot_df.columns):
            return self._passed("uniprot_mapping", {"warning": "Columns not present for duplicate mapping validation."})

        duplicate_mappings = (
            uniprot_df.dropna(subset=["pdb_code", "uniprot_id"])
            .groupby("pdb_code")["uniprot_id"]
            .nunique()
        )
        ambiguous = duplicate_mappings[duplicate_mappings > 1]
        if not ambiguous.empty:
            return self._failed(
                "uniprot_mapping",
                "Some PDB entries map to multiple UniProt ids in the valid dataset.",
                {"ambiguous_pdb_code_count": int(len(ambiguous))},
            )
        return self._passed("uniprot_mapping")

    def _check_known_gene_coverage(self, layout):
        uniprot_df = self._read_csv(layout.valid_dir / "Uniprot_functions.csv")
        if uniprot_df.empty:
            return self._failed("known_gene_coverage", "UniProt dataset is missing.")

        haystack = ""
        for column in ("associated_genes", "gene_names", "protein_recommended_name", "comment_disease_name"):
            if column in uniprot_df.columns:
                haystack += " " + " ".join(uniprot_df[column].fillna("").astype(str).tolist())
        if "EGFR" not in haystack.upper():
            return self._failed(
                "known_gene_coverage",
                "Known reviewer-critical gene EGFR is not represented in the valid UniProt dataset.",
            )
        return self._passed("known_gene_coverage", {"required_gene": "EGFR"})

    def _check_row_count_drift(self, layout):
        reports_dir = layout.base_dir / "validation"
        latest_path = reports_dir / f"{self.report_basename}_latest.json"
        quantitative = self._read_csv(layout.valid_dir / "Quantitative_data.csv")
        current_row_count = int(len(quantitative))
        if not latest_path.exists() or current_row_count == 0:
            return self._passed("row_count_drift", {"current_row_count": current_row_count})

        try:
            previous_report = json.loads(latest_path.read_text())
        except json.JSONDecodeError:
            return self._passed("row_count_drift", {"current_row_count": current_row_count})

        previous_row_count = None
        for check in previous_report.get("checks", []):
            if check.get("name") == "quantitative_duplicates":
                previous_row_count = (check.get("details") or {}).get("row_count")
                break

        if not previous_row_count:
            return self._passed("row_count_drift", {"current_row_count": current_row_count})

        drift = abs(current_row_count - previous_row_count) / max(previous_row_count, 1)
        if drift > 0.5:
            return self._failed(
                "row_count_drift",
                "Quantitative dataset row count drifted by more than 50% from the previous validation run.",
                {
                    "current_row_count": current_row_count,
                    "previous_row_count": previous_row_count,
                    "drift_ratio": round(drift, 3),
                },
            )

        return self._passed(
            "row_count_drift",
            {
                "current_row_count": current_row_count,
                "previous_row_count": previous_row_count,
                "drift_ratio": round(drift, 3),
            },
        )

    @staticmethod
    def _passed(name, details=None):
        return {"name": name, "passed": True, "details": details or {}}

    @staticmethod
    def _failed(name, message, details=None):
        return {
            "name": name,
            "passed": False,
            "message": message,
            "details": details or {},
        }
