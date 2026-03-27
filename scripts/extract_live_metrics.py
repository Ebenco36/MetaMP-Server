#!/usr/bin/env python3
"""Create a local, git-ignored publication snapshot from the live MetaMP Docker app.

This script runs on the host machine, reads the currently running MetaMP Docker
containers, copies publication artifacts locally, computes manuscript tables, and
writes a structured snapshot under ``.metamp-publication/``.

The script intentionally avoids non-stdlib dependencies on the host. Whenever
dataframe-style processing is needed, it executes a small Python payload inside
the running application container, where the MetaMP runtime dependencies are
already available.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_ROOT = REPO_ROOT / ".metamp-publication"
RUNS_ROOT = PUBLICATION_ROOT / "runs"
LATEST_LINK = PUBLICATION_ROOT / "latest"
MODEL_REGISTRY_VISUALIZER = REPO_ROOT / "scripts" / "model_registry_visualizer.py"

DEFAULT_APP_CONTAINER = "testmpvis_app"
DEFAULT_DB_CONTAINER = "testmetaMPDB"
DEFAULT_DOCKER_BIN = "/Applications/Docker.app/Contents/Resources/bin/docker"


IN_CONTAINER_EXTRACTION = r'''
import csv
import io
import json
import os
import re
from pathlib import Path
from numbers import Number

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr
from sqlalchemy import create_engine, inspect, text
from src.services.data.columns.quantitative.quantitative import cell_columns, rcsb_entries
from src.services.data.columns.quantitative.quantitative_array import quantitative_array_column


def canonical_group(value):
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if not s:
        return None
    s = re.sub(r"\s+", " ", s)
    mappings = {
        "MONOTOPIC": "MONOTOPIC MEMBRANE PROTEINS",
        "MONOTOPIC MEMBRANE PROTEIN": "MONOTOPIC MEMBRANE PROTEINS",
        "MONOTOPIC MEMBRANE PROTEINS": "MONOTOPIC MEMBRANE PROTEINS",
        "BITOPIC": "BITOPIC PROTEINS",
        "BITOPIC PROTEIN": "BITOPIC PROTEINS",
        "BITOPIC PROTEINS": "BITOPIC PROTEINS",
        "TRANSMEMBRANE PROTEIN:ALPHA-HELICAL": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL",
        "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL",
        "ALPHA-HELICAL": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL",
        "TRANSMEMBRANE PROTEIN:BETA-BARREL": "TRANSMEMBRANE PROTEINS:BETA-BARREL",
        "TRANSMEMBRANE PROTEINS:BETA-BARREL": "TRANSMEMBRANE PROTEINS:BETA-BARREL",
        "BETA-BARREL": "TRANSMEMBRANE PROTEINS:BETA-BARREL",
        "TRANSMEMBRANE": "TRANSMEMBRANE PROTEINS",
        "TRANSMEMBRANE PROTEIN": "TRANSMEMBRANE PROTEINS",
        "TRANSMEMBRANE PROTEINS": "TRANSMEMBRANE PROTEINS",
        "NOT A MEMBRANE PROTEIN": "NOT A MEMBRANE PROTEIN",
        "NOT A MEMBRANE PROTEIN.": "NOT A MEMBRANE PROTEIN",
    }
    return mappings.get(s, s)


def collapse_for_expert(value):
    s = canonical_group(value)
    if s is None:
        return None
    if s == "MONOTOPIC MEMBRANE PROTEINS":
        return "MONOTOPIC MEMBRANE PROTEINS"
    if s in {
        "BITOPIC PROTEINS",
        "TRANSMEMBRANE",
        "TRANSMEMBRANE PROTEINS",
        "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL",
        "TRANSMEMBRANE PROTEINS:BETA-BARREL",
    }:
        return "TRANSMEMBRANE / BITOPIC BENCHMARK"
    return s


def parse_tm(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    try:
        return float(int(float(s)))
    except Exception:
        pass
    match = re.search(r"-?\d+", s)
    if match:
        return float(int(match.group(0)))
    return np.nan


def safe_corr(fn, predicted, truth):
    if len(predicted) < 2:
        return None
    try:
        value = fn(predicted, truth).statistic
    except Exception:
        return None
    if pd.isna(value):
        return None
    return float(value)


def summarize_segment(series, expert):
    mask = expert.notna() & series.notna()
    n = int(mask.sum())
    if n == 0:
        return None
    predicted = series.loc[mask].astype(float)
    truth = expert.loc[mask].astype(float)
    abs_diff = np.abs(predicted - truth)
    return {
        "n": n,
        "exact": int((predicted == truth).sum()),
        "pct": float((predicted == truth).mean()),
        "mae": float(abs_diff.mean()),
        "sd": float(abs_diff.std(ddof=1)) if n > 1 else 0.0,
        "spearman": safe_corr(spearmanr, predicted, truth),
        "pearson": safe_corr(pearsonr, predicted, truth),
    }


def query_api_json(path):
    response = requests.get(f"http://localhost:8081{path}", timeout=120)
    response.raise_for_status()
    return response.json()


def query_api_text(path):
    response = requests.get(f"http://localhost:8081{path}", timeout=120)
    response.raise_for_status()
    return response.text


INTERNAL_SUMMARY_COLUMNS = {"id", "created_at", "updated_at"}
QUANTITATIVE_SUMMARY_COLUMNS = set(
    list(cell_columns) + list(rcsb_entries) + list(quantitative_array_column)
)
SOURCE_SUMMARY_TABLES = [
    ("MPstruc", "membrane_protein_mpstruct"),
    ("PDB", "membrane_protein_pdb"),
    ("OPM", "membrane_protein_opm"),
    ("UniProt", "membrane_protein_uniprot"),
    ("MetaMP", "membrane_proteins"),
]


def is_quantitative_column(column):
    if column["name"] in QUANTITATIVE_SUMMARY_COLUMNS:
        return True
    try:
        python_type = column["type"].python_type
    except Exception:
        return False
    return issubclass(python_type, Number) and not issubclass(python_type, bool)


def compute_source_summary(engine):
    inspector = inspect(engine)
    summary_rows = []
    with engine.connect() as conn:
        for label, table_name in SOURCE_SUMMARY_TABLES:
            columns = [
                column
                for column in inspector.get_columns(table_name)
                if column["name"] not in INTERNAL_SUMMARY_COLUMNS
            ]
            quantitative_count = int(sum(1 for column in columns if is_quantitative_column(column)))
            attributes_count = int(len(columns))
            nominal_count = int(max(attributes_count - quantitative_count, 0))
            column_names = {column["name"] for column in inspector.get_columns(table_name)}
            row_count = int(
                conn.execute(text(f"select count(*) from {table_name}")).scalar() or 0
            )
            if "pdb_code" in column_names:
                observations = int(
                    conn.execute(
                        text(f"select count(distinct upper(pdb_code)) from {table_name}")
                    ).scalar()
                    or 0
                )
                count_basis = "distinct_pdb_code" if observations != row_count else "rows"
            else:
                observations = row_count
                count_basis = "rows"
            summary_rows.append(
                {
                    "database": label,
                    "table_name": table_name,
                    "observations": observations,
                    "source_rows": row_count,
                    "count_basis": count_basis,
                    "attributes": attributes_count,
                    "nominal": nominal_count,
                    "quantitative": quantitative_count,
                    "highlight": label == "MetaMP",
                }
            )
    return summary_rows


def compute_database_year_rows(engine):
    with engine.connect() as conn:
        pdb = pd.read_sql(
            text(
                """
                select bibliography_year, pdb_code, uniprot_id
                from membrane_proteins
                """
            ),
            conn,
        )
        mpstruc = pd.read_sql(text("select pdb_code from membrane_protein_mpstruct"), conn)
        opm_columns = {
            column["name"] for column in inspect(engine).get_columns("membrane_protein_opm")
        }
        opm_join_column = "pdbid" if "pdbid" in opm_columns else "pdb_code"
        opm = pd.read_sql(
            text(f"select {opm_join_column} from membrane_protein_opm"),
            conn,
        )
        uniprot = pd.read_sql(text("select pdb_code from membrane_protein_uniprot"), conn)

    pdb["Pdb Code"] = pdb["pdb_code"].astype(str).str.lower()
    mpstruc["Pdb Code"] = mpstruc["pdb_code"].astype(str).str.lower()
    opm["pdbid"] = opm.iloc[:, 0].astype(str).str.lower()
    uniprot["pdb_code"] = uniprot["pdb_code"].astype(str).str.lower()

    pdb_columns = ["bibliography_year", "Pdb Code", "uniprot_id"]
    unique_records = {
        "pdb": pdb.drop_duplicates(subset=["Pdb Code"]),
        "mpstruc": mpstruc.drop_duplicates(subset=["Pdb Code"]),
        "opm": opm.drop_duplicates(subset=["pdbid"]),
        "uniprot": uniprot.drop_duplicates(subset=["pdb_code"]),
    }

    merged_frames = [
        pd.merge(
            unique_records["pdb"][pdb_columns],
            unique_records["mpstruc"],
            on="Pdb Code",
            how="inner",
        ),
        pd.merge(
            unique_records["mpstruc"],
            unique_records["pdb"][pdb_columns],
            on="Pdb Code",
            how="inner",
        ),
        pd.merge(
            unique_records["opm"],
            unique_records["pdb"][pdb_columns],
            left_on="pdbid",
            right_on="Pdb Code",
            how="inner",
        ),
        pd.merge(
            unique_records["uniprot"],
            unique_records["pdb"][pdb_columns],
            left_on="pdb_code",
            right_on="Pdb Code",
            how="inner",
        ),
    ]

    grouped_frames = []
    for frame, label in zip(merged_frames, ["PDB", "MPstruc", "OPM", "UniProt"]):
        grouped = frame.groupby(["bibliography_year"]).size().reset_index(name="count")
        grouped["database"] = label
        grouped_frames.append(grouped)
    final_data = pd.concat(grouped_frames, axis=0, ignore_index=True)
    final_data["bibliography_year"] = pd.to_numeric(
        final_data["bibliography_year"], errors="coerce"
    ).astype("Int64")
    final_data = final_data[final_data["bibliography_year"].notna()].copy()
    final_data["bibliography_year"] = final_data["bibliography_year"].astype(int)

    crosstab = pd.crosstab(
        final_data["bibliography_year"],
        final_data["database"],
        values=final_data["count"],
        aggfunc="sum",
    ).sort_index().fillna(0).cumsum()
    crosstab = crosstab.reset_index()
    melted = crosstab.melt(
        id_vars="bibliography_year",
        var_name="database",
        value_name="count",
    )
    melted["percentage"] = melted.groupby("bibliography_year")["count"].transform(
        lambda values: values / values.sum() * 100.0 if values.sum() else 0.0
    )
    order_mapping = {"MPstruc": 0, "UniProt": 1, "PDB": 2, "OPM": 3}
    melted["order"] = melted["database"].map(order_mapping)
    melted = melted.sort_values(["bibliography_year", "order"]).reset_index(drop=True)
    return melted.to_dict(orient="records")


out = {}

expert = pd.read_csv("/var/app/data/datasets/expert_annotation_predicted.csv")
out["expert_rows"] = int(len(expert))
out["expert_columns"] = list(expert.columns)
expert = expert.rename(
    columns={
        "PDB Code": "pdb_code",
        "Group (OPM)": "group_opm_raw",
        "Group (MPstruc)": "group_mpstruc_raw",
        "Group (Predicted)": "group_predicted_raw",
        "Group (Expert)": "group_expert_raw",
        "TM (Expert)": "tm_expert_raw",
    }
)
expert["pdb_code"] = expert["pdb_code"].astype(str).str.strip().str.upper()
for col in [
    "group_opm_raw",
    "group_mpstruc_raw",
    "group_predicted_raw",
    "group_expert_raw",
]:
    expert[col.replace("_raw", "_canon")] = expert[col].map(canonical_group)
    expert[col.replace("_raw", "_benchmark")] = expert[col].map(collapse_for_expert)
expert["tm_expert"] = expert["tm_expert_raw"].map(parse_tm)
out["expert_tm_nonnull"] = int(expert["tm_expert"].notna().sum())

ml_workbench = query_api_json("/api/v1/ml-workbench")
discrepancy_status = query_api_json("/api/v1/discrepancy-benchmark/status")
discrepancy_export_text = query_api_text(
    "/api/v1/discrepancy-benchmark/export?export_format=json&include_all=true"
)
discrepancy_export_rows = list(csv.DictReader(io.StringIO(discrepancy_export_text)))

out["ml_workbench"] = ml_workbench
out["discrepancy_status"] = discrepancy_status
out["discrepancy_export_row_count"] = len(discrepancy_export_rows)

url = os.environ.get("DATABASE_URL") or os.environ.get("SQLALCHEMY_DATABASE_URI")
engine = create_engine(url)
codes = expert["pdb_code"].dropna().unique().tolist()

with engine.connect() as conn:
    membrane_counts = pd.read_sql(
        text(
            """
            select
                (select count(*) from membrane_proteins) as membrane_proteins_rows,
                (select count(distinct upper(pdb_code)) from membrane_proteins) as membrane_proteins_unique_pdb_codes,
                (select count(*) from membrane_protein_opm) as membrane_protein_opm_rows,
                (select count(*) from membrane_protein_uniprot) as membrane_protein_uniprot_rows
            """
        ),
        conn,
    )
    tmalphafold = pd.read_sql(
        text(
            """
            select pdb_code, provider, method, prediction_kind, tm_count
            from membrane_protein_tmalphafold_predictions
            where upper(pdb_code) = any(:codes)
            """
        ),
        conn,
        params={"codes": codes},
    )

out["live_table_counts"] = membrane_counts.to_dict(orient="records")[0]
out["source_summary"] = compute_source_summary(engine)
out["database_year_proportions"] = compute_database_year_rows(engine)
out["tmalphafold_predictors"] = (
    tmalphafold[["provider", "method", "prediction_kind"]]
    .drop_duplicates()
    .sort_values(["provider", "method", "prediction_kind"])
    .to_dict(orient="records")
)

registry = pd.read_csv("/var/app/data/models/production_ml/tables/model_bundle_registry.csv")
out["model_registry"] = registry.sort_values(
    ["training_mode", "expert_f1_weighted", "expert_accuracy"],
    ascending=[True, False, False],
).to_dict(orient="records")

selected_upload = registry.loc[registry["selected_for_upload"] == True].iloc[0].to_dict()
selected_mode_rows = registry.loc[registry["selected_for_mode"] == True].copy()
out["selected_upload_artifact_id"] = selected_upload["artifact_id"]
out["selected_mode_artifact_ids"] = selected_mode_rows["artifact_id"].tolist()

selected_upload_predictions_path = (
    f"/var/app/data/models/production_ml/predictions/{selected_upload['artifact_id']}_expert_annotation_predictions.csv"
)
selected_upload_predictions = pd.read_csv(selected_upload_predictions_path)
out["selected_upload_predictions_path"] = selected_upload_predictions_path

selected_upload_class_concordance = []
strict_n = int(len(selected_upload_predictions))
strict_matches = int(selected_upload_predictions["matches_expert_exact"].fillna(False).sum())
bench_matches = int(selected_upload_predictions["matches_expert"].fillna(False).sum())
selected_upload_class_concordance.append(
    {
        "source": "MetaMP selected upload model",
        "artifact_id": selected_upload["artifact_id"],
        "strict_matches": strict_matches,
        "strict_n": strict_n,
        "strict_pct": strict_matches / strict_n if strict_n else None,
        "benchmark_matches": bench_matches,
        "benchmark_n": strict_n,
        "benchmark_pct": bench_matches / strict_n if strict_n else None,
    }
)

for source_col, label in [("Group (OPM)", "OPM"), ("Group (MPstruc)", "MPstruc")]:
    source_canon = selected_upload_predictions[source_col].map(canonical_group)
    source_benchmark = selected_upload_predictions[source_col].map(collapse_for_expert)
    expert_canon = selected_upload_predictions["group_expert_standardized"]
    expert_benchmark = selected_upload_predictions["group_expert_benchmark"]
    strict_mask = source_canon.notna() & expert_canon.notna()
    strict_n = int(strict_mask.sum())
    strict_matches = int((source_canon.loc[strict_mask] == expert_canon.loc[strict_mask]).sum())
    bench_mask = source_benchmark.notna() & expert_benchmark.notna()
    bench_n = int(bench_mask.sum())
    bench_matches = int((source_benchmark.loc[bench_mask] == expert_benchmark.loc[bench_mask]).sum())
    selected_upload_class_concordance.append(
        {
            "source": label,
            "strict_matches": strict_matches,
            "strict_n": strict_n,
            "strict_pct": strict_matches / strict_n if strict_n else None,
            "benchmark_matches": bench_matches,
            "benchmark_n": bench_n,
            "benchmark_pct": bench_matches / bench_n if bench_n else None,
        }
    )
out["selected_upload_class_concordance"] = selected_upload_class_concordance

segment_frame = expert[["pdb_code", "tm_expert"]].copy()
segment_frame["MetaMP:TMbed:sequence_topology"] = pd.NA

for row in discrepancy_export_rows:
    pdb_code = str(row.get("pdb_code") or "").strip().upper()
    if not pdb_code:
        continue
    idx = segment_frame.index[segment_frame["pdb_code"] == pdb_code]
    if len(idx) != 1:
        continue
    segment_frame.loc[idx[0], "MetaMP:TMbed:sequence_topology"] = parse_tm(row.get("tm_tmbed"))
    segment_frame.loc[idx[0], "TMAlphaFold:DeepTMHMM:sequence_topology"] = parse_tm(row.get("tm_deeptmhmm"))
    segment_frame.loc[idx[0], "OPM:segment_count:derived"] = parse_tm(row.get("tm_opm"))

for (provider, method, prediction_kind), sub in tmalphafold.groupby(
    ["provider", "method", "prediction_kind"], dropna=False
):
    col_name = f"{provider}:{method}:{prediction_kind}"
    agg = (
        sub.assign(pdb_code=sub["pdb_code"].astype(str).str.strip().str.upper())
        .groupby("pdb_code", as_index=False)["tm_count"]
        .first()
        .rename(columns={"tm_count": col_name})
    )
    if col_name in segment_frame.columns:
        merged = segment_frame.merge(
            agg.rename(columns={col_name: f"{col_name}__dup"}),
            on="pdb_code",
            how="left",
        )
        merged[col_name] = merged[col_name].where(
            merged[col_name].notna(),
            merged[f"{col_name}__dup"],
        )
        segment_frame = merged.drop(columns=[f"{col_name}__dup"])
    else:
        segment_frame = segment_frame.merge(agg, on="pdb_code", how="left")

segment_metrics = []
for col in segment_frame.columns:
    if col in {"pdb_code", "tm_expert"}:
        continue
    summary = summarize_segment(segment_frame[col], segment_frame["tm_expert"])
    if summary is None:
        continue
    summary["predictor"] = col
    segment_metrics.append(summary)
segment_metrics = sorted(
    segment_metrics,
    key=lambda item: (-item["pct"], item["mae"], item["predictor"]),
)
out["segment_metrics"] = segment_metrics

with open("/var/app/data/models/production_ml/specs/explainability_manifest.json") as fh:
    explainability_manifest = json.load(fh)
out["explainability_manifest"] = explainability_manifest

shap_table_path = None
for raw_path in explainability_manifest.get("table_paths", []) or []:
    candidate = Path(str(raw_path))
    if candidate.exists():
        shap_table_path = candidate
        break

if shap_table_path is None:
    tables_dir = Path("/var/app/data/models/production_ml/tables")
    shap_candidates = sorted(tables_dir.glob("*_shap_feature_importance.csv"))
    if shap_candidates:
        shap_table_path = shap_candidates[0]

if shap_table_path is not None and shap_table_path.exists():
    shap_csv = pd.read_csv(shap_table_path)
    out["shap_feature_importance_path"] = str(shap_table_path)
    out["shap_feature_importance"] = shap_csv.to_dict(orient="records")
    out["shap_global_ranking"] = (
        shap_csv.groupby("display_feature", as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)
        .to_dict(orient="records")
    )
else:
    out["shap_feature_importance_path"] = None
    out["shap_feature_importance"] = []
    out["shap_global_ranking"] = []

print(json.dumps(out))
'''


IN_CONTAINER_PUBLICATION_FIGURES = r'''
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text


OUTPUT_DIR = Path("/tmp/metamp_publication_figures")
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_database_year_rows(engine):
    with engine.connect() as conn:
        pdb = pd.read_sql(
            text(
                """
                select bibliography_year, pdb_code, uniprot_id
                from membrane_proteins
                """
            ),
            conn,
        )
        mpstruc = pd.read_sql(text("select pdb_code from membrane_protein_mpstruct"), conn)
        opm_columns = {
            column["name"] for column in inspect(engine).get_columns("membrane_protein_opm")
        }
        opm_join_column = "pdbid" if "pdbid" in opm_columns else "pdb_code"
        opm = pd.read_sql(
            text(f"select {opm_join_column} from membrane_protein_opm"),
            conn,
        )
        uniprot = pd.read_sql(text("select pdb_code from membrane_protein_uniprot"), conn)

    pdb["Pdb Code"] = pdb["pdb_code"].astype(str).str.lower()
    mpstruc["Pdb Code"] = mpstruc["pdb_code"].astype(str).str.lower()
    opm["pdbid"] = opm.iloc[:, 0].astype(str).str.lower()
    uniprot["pdb_code"] = uniprot["pdb_code"].astype(str).str.lower()

    pdb_columns = ["bibliography_year", "Pdb Code", "uniprot_id"]
    unique_records = {
        "pdb": pdb.drop_duplicates(subset=["Pdb Code"]),
        "mpstruc": mpstruc.drop_duplicates(subset=["Pdb Code"]),
        "opm": opm.drop_duplicates(subset=["pdbid"]),
        "uniprot": uniprot.drop_duplicates(subset=["pdb_code"]),
    }

    merged_frames = [
        pd.merge(
            unique_records["pdb"][pdb_columns],
            unique_records["mpstruc"],
            on="Pdb Code",
            how="inner",
        ),
        pd.merge(
            unique_records["mpstruc"],
            unique_records["pdb"][pdb_columns],
            on="Pdb Code",
            how="inner",
        ),
        pd.merge(
            unique_records["opm"],
            unique_records["pdb"][pdb_columns],
            left_on="pdbid",
            right_on="Pdb Code",
            how="inner",
        ),
        pd.merge(
            unique_records["uniprot"],
            unique_records["pdb"][pdb_columns],
            left_on="pdb_code",
            right_on="Pdb Code",
            how="inner",
        ),
    ]

    grouped_frames = []
    for frame, label in zip(merged_frames, ["PDB", "MPstruc", "OPM", "UniProt"]):
        grouped = frame.groupby(["bibliography_year"]).size().reset_index(name="count")
        grouped["database"] = label
        grouped_frames.append(grouped)
    final_data = pd.concat(grouped_frames, axis=0, ignore_index=True)
    final_data["bibliography_year"] = pd.to_numeric(
        final_data["bibliography_year"], errors="coerce"
    ).astype("Int64")
    final_data = final_data[final_data["bibliography_year"].notna()].copy()
    final_data["bibliography_year"] = final_data["bibliography_year"].astype(int)

    crosstab = pd.crosstab(
        final_data["bibliography_year"],
        final_data["database"],
        values=final_data["count"],
        aggfunc="sum",
    ).sort_index().fillna(0).cumsum()
    crosstab = crosstab.reset_index()
    melted = crosstab.melt(
        id_vars="bibliography_year",
        var_name="database",
        value_name="count",
    )
    melted["percentage"] = melted.groupby("bibliography_year")["count"].transform(
        lambda values: values / values.sum() * 100.0 if values.sum() else 0.0
    )
    return melted


engine = create_engine(os.environ.get("DATABASE_URL") or os.environ.get("SQLALCHEMY_DATABASE_URI"))
frame = compute_database_year_rows(engine)
display_labels = {
    "MPstruc": "MPstruc (Membrane Protein Structures)",
    "UniProt": "UniProt (Universal Protein Resource)",
    "PDB": "PDB (Protein Data Bank)",
    "OPM": "OPM (Orientation of Proteins in Membranes)",
}
order = ["MPstruc", "UniProt", "PDB", "OPM"]
colors = {
    "MPstruc": "#ffffff",
    "UniProt": "#8da0cb",
    "PDB": "#d9d9d9",
    "OPM": "#66c2a5",
}

frame["database_display"] = frame["database"].map(display_labels)
frame.to_csv(OUTPUT_DIR / "database_year_proportional_representation.csv", index=False)

pivot = (
    frame.pivot_table(
        index="bibliography_year",
        columns="database",
        values="percentage",
        aggfunc="first",
        fill_value=0.0,
    )
    .reindex(columns=order, fill_value=0.0)
    .sort_index()
)

fig, ax = plt.subplots(figsize=(11.5, 5.8))
years = pivot.index.astype(str).tolist()
positions = np.arange(len(years))
bottom = np.zeros(len(pivot))
bar_width = 0.82
edge_color = "#7f7f7f"
for database in order:
    values = pivot[database].to_numpy(dtype=float)
    ax.bar(
        positions,
        values,
        width=bar_width,
        bottom=bottom,
        color=colors[database],
        edgecolor=edge_color,
        linewidth=0.45,
        label=display_labels[database],
    )
    bottom += values

ax.set_ylabel("Proportional Representation of Database Entries (%)")
ax.set_xlabel("Year")
ax.set_ylim(0, 100)
ax.set_xlim(-bar_width / 2 - 0.03, len(years) - 1 + bar_width / 2 + 0.03)
ax.set_xticks(positions)
ax.set_xticklabels(years)
ax.grid(axis="y", linewidth=0.3, alpha=0.35)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

fig.legend(
    title=None,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.045),
    frameon=False,
    handlelength=0.9,
    handletextpad=0.5,
    columnspacing=1.2,
)
plt.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUTPUT_DIR / "database_year_proportional_representation.pdf", bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "database_year_proportional_representation.png", dpi=300, bbox_inches="tight")
print(json.dumps({"out_dir": str(OUTPUT_DIR)}))
'''


def find_docker_bin() -> str:
    configured = os.environ.get("DOCKER_BIN")
    if configured:
        return configured
    if Path(DEFAULT_DOCKER_BIN).exists():
        return DEFAULT_DOCKER_BIN
    resolved = shutil.which("docker")
    if resolved:
        return resolved
    raise FileNotFoundError("Docker binary not found. Set DOCKER_BIN to the correct path.")


def run(cmd: list[str], *, input_text: str | None = None) -> str:
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout


def docker_exec(
    docker_bin: str,
    container: str,
    args: list[str],
    *,
    stdin_text: str | None = None,
) -> str:
    cmd = [docker_bin, "exec"]
    if stdin_text is not None:
        cmd.append("-i")
    cmd.extend([container, *args])
    return run(cmd, input_text=stdin_text)


def docker_cp(docker_bin: str, source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run([docker_bin, "cp", source, str(destination)])


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def latex_escape(text: object) -> str:
    value = "" if text is None else str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def format_float(value: object, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return latex_escape(value)
    return f"{numeric:.{digits}f}"


def format_pct(numerator: object, denominator: object, digits: int = 2) -> str:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return "n/a"
    if den == 0:
        return "n/a"
    return f"{int(num)}/{int(den)} ({100.0 * num / den:.{digits}f}\\%)"


def format_plain_pct(value: object, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return latex_escape(value)
    return f"{numeric:.{digits}f}"


def pretty_bundle_name(row: dict) -> str:
    reduction_labels = {
        "no_dr": "No-DR",
        "pca": "PCA",
        "tsne": "t-SNE",
        "umap": "UMAP",
    }
    reduction = reduction_labels.get(str(row.get("reduction_key") or ""), str(row.get("reduction_key") or ""))
    classifier = str(row.get("classifier_name") or "").replace("_", " ")
    return f"{reduction} {classifier}".strip()


def write_latex_table(path: Path, caption: str, label: str, columns: list[str], rows: list[list[str]]) -> None:
    ensure_parent(path)
    alignment = "l" * len(columns)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def build_snapshot_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = RUNS_ROOT / timestamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if LATEST_LINK.exists() or LATEST_LINK.is_symlink():
        if LATEST_LINK.is_symlink() or LATEST_LINK.is_file():
            LATEST_LINK.unlink()
        else:
            shutil.rmtree(LATEST_LINK)
    try:
        os.symlink(snapshot_dir.resolve(), LATEST_LINK, target_is_directory=True)
    except OSError:
        shutil.copytree(snapshot_dir, LATEST_LINK)
    return snapshot_dir


def copy_tree_from_container(docker_bin: str, container: str, source_dir: str, destination_dir: Path) -> None:
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    run([docker_bin, "cp", f"{container}:{source_dir}", str(destination_dir)])


def normalize_registry_rows(rows: list[dict]) -> list[dict]:
    ordered = []
    for row in rows:
        normalized = dict(row)
        for key in (
            "selected_for_upload",
            "selected_for_mode",
        ):
            if key in normalized:
                normalized[key] = "yes" if bool(normalized[key]) else "no"
        ordered.append(normalized)
    return ordered


def create_publication_exports(snapshot_dir: Path, payload: dict) -> None:
    derived_root = snapshot_dir / "derived"
    tables_root = derived_root / "tables"
    notes_root = derived_root / "notes"

    live_counts = payload["live_table_counts"]
    write_csv(
        tables_root / "live_table_counts.csv",
        [
            {
                "membrane_proteins_rows": live_counts["membrane_proteins_rows"],
                "membrane_proteins_unique_pdb_codes": live_counts["membrane_proteins_unique_pdb_codes"],
                "membrane_protein_opm_rows": live_counts["membrane_protein_opm_rows"],
                "membrane_protein_uniprot_rows": live_counts["membrane_protein_uniprot_rows"],
            }
        ],
        [
            "membrane_proteins_rows",
            "membrane_proteins_unique_pdb_codes",
            "membrane_protein_opm_rows",
            "membrane_protein_uniprot_rows",
        ],
    )

    source_rows = payload["source_summary"]
    write_csv(
        tables_root / "database_contribution_summary.csv",
        source_rows,
        [
            "database",
            "table_name",
            "observations",
            "source_rows",
            "count_basis",
            "attributes",
            "nominal",
            "quantitative",
            "highlight",
        ],
    )
    source_latex_rows = []
    for row in source_rows:
        display = rf"\textbf{{{latex_escape(row['database'])}}}" if row.get("highlight") else latex_escape(row["database"])
        observations = rf"\textbf{{{row['observations']}}}" if row.get("highlight") else latex_escape(row["observations"])
        attributes = rf"\textbf{{{row['attributes']}}}" if row.get("highlight") else latex_escape(row["attributes"])
        nominal = rf"\textbf{{{row['nominal']}}}" if row.get("highlight") else latex_escape(row["nominal"])
        quantitative = rf"\textbf{{{row['quantitative']}}}" if row.get("highlight") else latex_escape(row["quantitative"])
        source_latex_rows.append([display, observations, attributes, nominal, quantitative])
    write_latex_table(
        tables_root / "database_contribution_summary.tex",
        r"\textbf{Proportional contribution of each dedicated protein database to MetaMP.} The table reports the number of membrane-protein observations and the number of nominal and quantitative attributes contributed by each source. The broader attribute coverage in MetaMP reflects the benefit of multi-source harmonization for membrane-protein analysis.",
        "tab:databases",
        [
            r"\textbf{Database}",
            r"\textbf{Rows/Observations/MPs}",
            r"\textbf{Attributes/Features}",
            r"\textbf{Nominal}",
            r"\textbf{Quantitative}",
        ],
        source_latex_rows,
    )

    database_year_rows = payload["database_year_proportions"]
    write_csv(
        tables_root / "database_year_proportional_representation.csv",
        database_year_rows,
        ["bibliography_year", "database", "count", "percentage", "order"],
    )

    class_rows = payload["selected_upload_class_concordance"]
    write_csv(
        tables_root / "class_concordance_selected_upload.csv",
        class_rows,
        [
            "source",
            "artifact_id",
            "strict_matches",
            "strict_n",
            "strict_pct",
            "benchmark_matches",
            "benchmark_n",
            "benchmark_pct",
        ],
    )
    class_latex_rows = []
    for row in class_rows:
        class_latex_rows.append(
            [
                latex_escape(row["source"]),
                format_pct(row["strict_matches"], row["strict_n"]),
                format_pct(row["benchmark_matches"], row["benchmark_n"]),
            ]
        )
    write_latex_table(
        tables_root / "class_concordance_selected_upload.tex",
        r"\textbf{Cross-source broad-group concordance on the 121-row expert benchmark set.}",
        "tab:class_concordance_snapshot",
        [r"\textbf{Source}", r"\textbf{Strict agreement}", r"\textbf{Benchmark-aware agreement}"],
        class_latex_rows,
    )

    segment_rows = payload["segment_metrics"]
    write_csv(
        tables_root / "transmembrane_segment_benchmark.csv",
        segment_rows,
        ["predictor", "n", "exact", "pct", "mae", "sd", "spearman", "pearson"],
    )
    segment_latex_rows = []
    for row in segment_rows:
        segment_latex_rows.append(
            [
                latex_escape(row["predictor"]),
                latex_escape(row["n"]),
                format_pct(row["exact"], row["n"]),
                f"{format_float(row['mae'], 2)} $\\pm$ {format_float(row['sd'], 2)}",
                "n/a" if row["spearman"] is None else format_float(row["spearman"], 3),
                "n/a" if row["pearson"] is None else format_float(row["pearson"], 3),
            ]
        )
    write_latex_table(
        tables_root / "transmembrane_segment_benchmark.tex",
        r"\textbf{Transmembrane-segment benchmark summaries on the 121-row expert benchmark configuration.}",
        "tab:segment_benchmark_snapshot",
        [
            r"\textbf{Predictor}",
            r"\textbf{$n$}",
            r"\textbf{Exact matches}",
            r"\textbf{MAE $\pm$ SD}",
            r"\textbf{Spearman $\rho$}",
            r"\textbf{Pearson $r$}",
        ],
        segment_latex_rows,
    )

    registry_rows = normalize_registry_rows(payload["model_registry"])
    registry_fieldnames = list(registry_rows[0].keys())
    write_csv(tables_root / "model_bundle_registry_all.csv", registry_rows, registry_fieldnames)
    registry_latex_rows = []
    for row in registry_rows:
        registry_latex_rows.append(
            [
                latex_escape(row["training_mode"]),
                latex_escape(row["reduction_key"]),
                latex_escape(row["classifier_name"]),
                format_float(row["cv_mean_accuracy"]),
                format_float(row["cv_mean_precision"]),
                format_float(row["cv_mean_recall"]),
                format_float(row["cv_mean_f1"]),
                format_float(row["expert_accuracy"]),
                format_float(row["expert_precision_weighted"]),
                format_float(row["expert_recall_weighted"]),
                format_float(row["expert_f1_weighted"]),
                format_float(row["expert_accuracy_exact"]),
                format_float(row.get("expert_precision_weighted_exact")),
                format_float(row.get("expert_recall_weighted_exact")),
                format_float(row["expert_f1_weighted_exact"]),
                latex_escape(row["selected_for_upload"]),
                latex_escape(row["selected_for_mode"]),
            ]
        )
    write_latex_table(
        tables_root / "model_bundle_registry_all.tex",
        r"\textbf{Full live bundle registry for supervised and semi-supervised model combinations.}",
        "tab:model_bundle_registry_all",
        [
            r"\textbf{Mode}",
            r"\textbf{View}",
            r"\textbf{Classifier}",
            r"\textbf{CV acc.}",
            r"\textbf{CV prec.}",
            r"\textbf{CV rec.}",
            r"\textbf{CV F1}",
            r"\textbf{Expert acc.}",
            r"\textbf{Expert prec.}",
            r"\textbf{Expert rec.}",
            r"\textbf{Expert F1}",
            r"\textbf{Exact acc.}",
            r"\textbf{Exact prec.}",
            r"\textbf{Exact rec.}",
            r"\textbf{Exact F1}",
            r"\textbf{Upload}",
            r"\textbf{Mode sel.}",
        ],
        registry_latex_rows,
    )
    held_out_rows = []
    for row in registry_rows:
        held_out_rows.append(
            {
                "mode": row["training_mode"],
                "bundle": pretty_bundle_name(row),
                "cv_accuracy": row["cv_mean_accuracy"],
                "cv_precision": row["cv_mean_precision"],
                "cv_recall": row["cv_mean_recall"],
                "cv_f1": row["cv_mean_f1"],
                "expert_accuracy": row["expert_accuracy"],
                "expert_precision": row["expert_precision_weighted"],
                "expert_recall": row["expert_recall_weighted"],
                "expert_f1": row["expert_f1_weighted"],
                "exact_accuracy": row["expert_accuracy_exact"],
                "exact_precision": row.get("expert_precision_weighted_exact"),
                "exact_recall": row.get("expert_recall_weighted_exact"),
                "exact_f1": row["expert_f1_weighted_exact"],
                "artifact_id": row["artifact_id"],
            }
        )
    held_out_rows = sorted(
        held_out_rows,
        key=lambda item: (item["mode"], item["bundle"]),
    )
    write_csv(
        tables_root / "held_out_expert_benchmark_all_combinations.csv",
        held_out_rows,
        [
            "mode",
            "bundle",
            "cv_accuracy",
            "cv_precision",
            "cv_recall",
            "cv_f1",
            "expert_accuracy",
            "expert_precision",
            "expert_recall",
            "expert_f1",
            "exact_accuracy",
            "exact_precision",
            "exact_recall",
            "exact_f1",
            "artifact_id",
        ],
    )
    held_out_latex_rows = []
    for row in held_out_rows:
        held_out_latex_rows.append(
            [
                latex_escape(row["mode"]),
                latex_escape(row["bundle"]),
                format_plain_pct(row["cv_accuracy"]),
                format_plain_pct(row["cv_precision"]),
                format_plain_pct(row["cv_recall"]),
                format_plain_pct(row["cv_f1"]),
                format_plain_pct(row["expert_accuracy"]),
                format_plain_pct(row["expert_precision"]),
                format_plain_pct(row["expert_recall"]),
                format_plain_pct(row["expert_f1"]),
                format_plain_pct(row["exact_accuracy"]),
                format_plain_pct(row.get("exact_precision")),
                format_plain_pct(row.get("exact_recall")),
                format_plain_pct(row["exact_f1"]),
            ]
        )
    write_latex_table(
        tables_root / "held_out_expert_benchmark_all_combinations.tex",
        r"\textbf{Held-out expert benchmark and cross-validation metrics for all live supervised and semi-supervised model bundles.}",
        "tab:held_out_expert_benchmark_all_combinations",
        [
            r"\textbf{Mode}",
            r"\textbf{Bundle}",
            r"\textbf{CV acc.}",
            r"\textbf{CV prec.}",
            r"\textbf{CV rec.}",
            r"\textbf{CV F1}",
            r"\textbf{Expert acc.}",
            r"\textbf{Expert prec.}",
            r"\textbf{Expert rec.}",
            r"\textbf{Expert F1}",
            r"\textbf{Exact acc.}",
            r"\textbf{Exact prec.}",
            r"\textbf{Exact rec.}",
            r"\textbf{Exact F1}",
        ],
        held_out_latex_rows,
    )

    selected_rows = []
    selected_ids = set(payload["selected_mode_artifact_ids"]) | {payload["selected_upload_artifact_id"]}
    for row in registry_rows:
        if row["artifact_id"] in selected_ids:
            selected_rows.append(row)
    write_csv(tables_root / "selected_model_bundles.csv", selected_rows, registry_fieldnames)

    shap_rows = payload["shap_feature_importance"]
    shap_fieldnames = list(shap_rows[0].keys()) if shap_rows else [
        "feature",
        "display_feature",
        "class_name",
        "mean_abs_shap",
        "artifact_id",
        "classifier_name",
        "training_mode",
        "reduction_key",
        "source_scope",
    ]
    write_csv(tables_root / "shap_feature_importance.csv", shap_rows, shap_fieldnames)
    shap_global_rows = payload["shap_global_ranking"]
    write_csv(
        tables_root / "shap_global_ranking.csv",
        shap_global_rows,
        ["display_feature", "mean_abs_shap"],
    )

    methods_note = """\
Strict agreement uses standardized canonical labels without biological collapsing.
Benchmark-aware agreement uses the live MetaMP expert-benchmark collapsing rule:
- expert Bitopic counts as compatible with Bitopic, alpha-helical transmembrane, or beta-barrel transmembrane predictions
- expert Monotopic remains strictly monotopic

Segment benchmarking parses expert TM strings by extracting the leading integer.
Examples:
- 1** -> 1
- 1 * -> 1
- 1x8 -> 1
- 0** -> 0
Blank expert TM fields remain missing and are excluded from segment-level metrics.
"""
    ensure_parent(notes_root / "benchmark_notes.md")
    (notes_root / "benchmark_notes.md").write_text(methods_note)


def export_publication_figures(docker_bin: str, app_container: str, snapshot_dir: Path) -> None:
    docker_exec(
        docker_bin,
        app_container,
        ["python", "-"],
        stdin_text=IN_CONTAINER_PUBLICATION_FIGURES,
    )
    docker_cp(
        docker_bin,
        f"{app_container}:/tmp/metamp_publication_figures",
        snapshot_dir / "copied" / "publication_figures",
    )


def export_model_registry_figures(docker_bin: str, app_container: str, snapshot_dir: Path) -> None:
    visualizer_script = MODEL_REGISTRY_VISUALIZER.read_text()
    docker_exec(
        docker_bin,
        app_container,
        [
            "python",
            "-",
            "--csv",
            "/var/app/data/models/production_ml/tables/model_bundle_registry.csv",
            "--out",
            "/tmp/metamp_model_registry_figures",
            "--formats",
            "png",
            "pdf",
            "html",
            "json",
            "--no-summary",
        ],
        stdin_text=visualizer_script,
    )
    docker_cp(
        docker_bin,
        f"{app_container}:/tmp/metamp_model_registry_figures",
        snapshot_dir / "copied" / "publication_figures" / "model_registry",
    )
    model_registry_dir = snapshot_dir / "copied" / "publication_figures" / "model_registry"
    missing_pdf_companions = []
    for png_path in model_registry_dir.glob("*.png"):
        pdf_path = png_path.with_suffix(".pdf")
        if not pdf_path.exists():
            missing_pdf_companions.append(png_path.name)
    if missing_pdf_companions:
        raise RuntimeError(
            "Model registry figure export is incomplete; missing PDF companions for: "
            + ", ".join(sorted(missing_pdf_companions))
        )


def export_exploratory_dr_figures(docker_bin: str, app_container: str, snapshot_dir: Path) -> None:
    exploratory_dir = snapshot_dir / "copied" / "publication_figures" / "exploratory_dr"
    ensure_clean_dir(exploratory_dir)

    copied_paths: set[str] = set()
    for stem in ("pca", "tsne", "umap"):
        for suffix in (".png", ".pdf"):
            container_path = f"/var/app/data/models/{stem}{suffix}"
            destination_path = exploratory_dir / f"{stem}{suffix}"
            try:
                docker_cp(
                    docker_bin,
                    f"{app_container}:{container_path}",
                    destination_path,
                )
                copied_paths.add(f"{stem}{suffix}")
            except RuntimeError:
                continue

    required_pdfs = {f"{stem}.pdf" for stem in ("pca", "tsne", "umap")}
    missing_pdfs = sorted(required_pdfs - copied_paths)
    if missing_pdfs:
        raise RuntimeError(
            "Exploratory dimensionality-reduction PDF export is incomplete; missing: "
            + ", ".join(missing_pdfs)
        )


def list_relative_files(root: Path, suffix: str) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob(f"*{suffix}")
        if path.is_file()
    )


def main() -> None:
    docker_bin = find_docker_bin()
    app_container = os.environ.get("METAMP_APP_CONTAINER", DEFAULT_APP_CONTAINER)
    db_container = os.environ.get("METAMP_DB_CONTAINER", DEFAULT_DB_CONTAINER)

    snapshot_dir = build_snapshot_dir()
    copied_root = snapshot_dir / "copied"
    production_root = copied_root / "production_ml"

    payload_stdout = docker_exec(
        docker_bin,
        app_container,
        ["python", "-"],
        stdin_text=IN_CONTAINER_EXTRACTION,
    )
    payload = json.loads(payload_stdout)

    write_json(snapshot_dir / "metadata" / "live_metrics.json", payload)

    copy_tree_from_container(
        docker_bin,
        app_container,
        "/var/app/data/models/production_ml",
        production_root,
    )
    docker_cp(
        docker_bin,
        f"{app_container}:/var/app/data/datasets/expert_annotation_predicted.csv",
        copied_root / "datasets" / "expert_annotation_predicted.csv",
    )

    status_data = payload["discrepancy_status"]["data"]
    if status_data.get("csv_path"):
        docker_cp(
            docker_bin,
            f"{app_container}:{status_data['csv_path']}",
            copied_root / "benchmarks" / Path(status_data["csv_path"]).name,
        )
    if status_data.get("json_path"):
        docker_cp(
            docker_bin,
            f"{app_container}:{status_data['json_path']}",
            copied_root / "benchmarks" / Path(status_data["json_path"]).name,
        )

    full_export_csv = docker_exec(
        docker_bin,
        app_container,
        ["python", "-c", "import requests; print(requests.get('http://localhost:8081/api/v1/discrepancy-benchmark/export?export_format=json&include_all=true', timeout=120).text, end='')"],
    )
    ensure_parent(copied_root / "benchmarks" / "metamp_discrepancy_benchmark_include_all.csv")
    (copied_root / "benchmarks" / "metamp_discrepancy_benchmark_include_all.csv").write_text(full_export_csv)

    db_counts_stdout = docker_exec(
        docker_bin,
        db_container,
        [
            "psql",
            "-U",
            "mpvis_user",
            "-d",
            "mpvis_db",
            "-At",
            "-F",
            "\t",
            "-c",
            (
                "select 'membrane_proteins' as table_name, count(*) as rows, count(distinct upper(pdb_code)) as unique_pdb_codes from membrane_proteins "
                "union all select 'membrane_protein_opm', count(*), count(distinct upper(pdb_code)) from membrane_protein_opm "
                "union all select 'membrane_protein_uniprot', count(*), count(distinct upper(pdb_code)) from membrane_protein_uniprot "
                "order by table_name;"
            ),
        ],
    )
    ensure_parent(snapshot_dir / "metadata" / "live_db_counts.tsv")
    (snapshot_dir / "metadata" / "live_db_counts.tsv").write_text(db_counts_stdout)

    create_publication_exports(snapshot_dir, payload)
    export_publication_figures(docker_bin, app_container, snapshot_dir)
    export_model_registry_figures(docker_bin, app_container, snapshot_dir)
    export_exploratory_dr_figures(docker_bin, app_container, snapshot_dir)

    publication_figures_root = copied_root / "publication_figures"
    production_figures_root = copied_root / "production_ml" / "figures"

    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "app_container": app_container,
        "db_container": db_container,
        "docker_bin": docker_bin,
        "snapshot_dir": str(snapshot_dir),
        "latest_link": str(LATEST_LINK),
        "selected_upload_artifact_id": payload["selected_upload_artifact_id"],
        "selected_mode_artifact_ids": payload["selected_mode_artifact_ids"],
        "shap_bar_figure": str(
            snapshot_dir
            / "copied"
            / "production_ml"
            / "figures"
            / "semi_supervised_no_dr_decision_tree_shap_bar.pdf"
        ),
        "database_year_figure_pdf": str(
            snapshot_dir
            / "copied"
            / "publication_figures"
            / "database_year_proportional_representation.pdf"
        ),
        "database_year_figure_png": str(
            snapshot_dir
            / "copied"
            / "publication_figures"
            / "database_year_proportional_representation.png"
        ),
        "model_registry_figures_dir": str(
            snapshot_dir
            / "copied"
            / "publication_figures"
            / "model_registry"
        ),
        "model_registry_figure_manifest": str(
            snapshot_dir
            / "copied"
            / "publication_figures"
            / "model_registry"
            / "figure_manifest.json"
        ),
        "model_registry_cv_vs_expert_pdf": str(
            snapshot_dir
            / "copied"
            / "publication_figures"
            / "model_registry"
            / "fig1_cv_vs_expert_scatter.pdf"
        ),
        "model_registry_top_ranked_pdf": str(
            snapshot_dir
            / "copied"
            / "publication_figures"
            / "model_registry"
            / "fig2_top_n_ranked_bar.pdf"
        ),
        "model_registry_heatmap_pdf": str(
            snapshot_dir
            / "copied"
            / "publication_figures"
            / "model_registry"
            / "fig4_heatmap_classifier_dr.pdf"
        ),
        "exploratory_dr_figures_dir": str(
            snapshot_dir
            / "copied"
            / "publication_figures"
            / "exploratory_dr"
        ),
        "exploratory_pca_pdf": str(
            snapshot_dir
            / "copied"
            / "publication_figures"
            / "exploratory_dr"
            / "pca.pdf"
        ),
        "exploratory_tsne_pdf": str(
            snapshot_dir
            / "copied"
            / "publication_figures"
            / "exploratory_dr"
            / "tsne.pdf"
        ),
        "exploratory_umap_pdf": str(
            snapshot_dir
            / "copied"
            / "publication_figures"
            / "exploratory_dr"
            / "umap.pdf"
        ),
        "publication_figure_pdfs": list_relative_files(publication_figures_root, ".pdf"),
        "production_ml_figure_pdfs": list_relative_files(production_figures_root, ".pdf"),
        "publication_figure_pngs": list_relative_files(publication_figures_root, ".png"),
    }
    write_json(snapshot_dir / "metadata" / "publication_manifest.json", manifest)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - operational entry point
        print(f"[extract_live_metrics] {exc}", file=sys.stderr)
        sys.exit(1)
