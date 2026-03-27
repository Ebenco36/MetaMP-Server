import json
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
import requests
from flask import current_app, has_app_context
from psycopg2 import sql
from tqdm import tqdm
from database.db import db

sys.path.append(os.getcwd())

from src.AI_Packages.TMProteinPredictor import (  # noqa: E402
    DeepTMHMMPredictor,
    MultiModelAnalyzer,
    TMbedPredictor,
    parse_tm_regions_value,
    serialize_tm_regions,
)
from src.Dashboard.services import get_table_as_dataframe, get_tables_as_dataframe  # noqa: E402
from src.MP.model_tmalphafold import TMAlphaFoldPrediction  # noqa: E402


DEFAULT_TM_PREDICTION_OUTPUT_CSV = "/var/app/data/tm_predictions/tm_summary.csv"
DEFAULT_OPTIONAL_TM_PREDICTION_BASE_DIR = "/var/app/data/tm_predictions/external"
OPTIONAL_TM_PREDICTORS = ("Phobius", "TOPCONS", "CCTOP")
TM_PREDICTION_RECORD_COLUMNS = [
    "id",
    "sequence",
    "TMbed_tm_count",
    "TMbed_tm_regions",
    "DeepTMHMM_tm_count",
    "DeepTMHMM_tm_regions",
    "Phobius_tm_count",
    "Phobius_tm_regions",
    "TOPCONS_tm_count",
    "TOPCONS_tm_regions",
    "CCTOP_tm_count",
    "CCTOP_tm_regions",
]
NORMALIZED_STORE_METHODS = ("TMbed", "DeepTMHMM", "Phobius", "TOPCONS", "CCTOP")


def tm_predictor_column_names(predictor_name):
    normalized_name = str(predictor_name or "").strip()
    if not normalized_name:
        raise ValueError("Predictor name is required.")
    return (
        f"{normalized_name}_tm_count",
        f"{normalized_name}_tm_regions",
    )


def _load_normalized_tm_prediction_frame(
    predictor_names=None,
    provider="MetaMP",
    pdb_codes=None,
):
    selected_predictors = [
        str(name or "").strip()
        for name in (predictor_names or NORMALIZED_STORE_METHODS)
        if str(name or "").strip()
    ]
    if not selected_predictors:
        return pd.DataFrame(columns=["pdb_code"])

    query = TMAlphaFoldPrediction.query.filter(
        TMAlphaFoldPrediction.provider == provider,
        TMAlphaFoldPrediction.status == "success",
        TMAlphaFoldPrediction.method.in_(selected_predictors),
    )
    selected_codes = _normalize_pdb_code_selection(pdb_codes)
    if selected_codes:
        query = query.filter(
            db.func.upper(db.func.trim(TMAlphaFoldPrediction.pdb_code)).in_(selected_codes)
        )

    rows = query.all()
    grouped = {}
    for row in rows:
        pdb_code = str(row.pdb_code or "").strip().upper()
        method = str(row.method or "").strip()
        if not pdb_code or not method:
            continue
        grouped.setdefault(pdb_code, {}).setdefault(method, []).append(row)

    records = []
    for pdb_code, methods in grouped.items():
        item = {"pdb_code": pdb_code}
        for method in selected_predictors:
            method_rows = methods.get(method) or []
            count_col, region_col = tm_predictor_column_names(method)
            if not method_rows:
                item[count_col] = None
                item[region_col] = ""
                continue
            tm_count_values = {row.tm_count for row in method_rows if row.tm_count is not None}
            tm_region_values = {
                str(row.tm_regions_json or "[]")
                for row in method_rows
                if str(row.tm_regions_json or "").strip()
            }
            item[count_col] = next(iter(tm_count_values)) if len(tm_count_values) == 1 else None
            item[region_col] = next(iter(tm_region_values)) if len(tm_region_values) == 1 else ""
        records.append(item)

    return pd.DataFrame(records)


def normalize_optional_tm_predictor_name(predictor_name):
    normalized_name = str(predictor_name or "").strip().lower()
    for candidate in OPTIONAL_TM_PREDICTORS:
        if candidate.lower() == normalized_name:
            return candidate
    raise ValueError(
        f"Unsupported optional predictor '{predictor_name}'. "
        f"Expected one of: {', '.join(OPTIONAL_TM_PREDICTORS)}."
    )


def _resolve_runtime_directory_path(configured_path, local_relative_path):
    candidate = Path(configured_path)
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except (FileNotFoundError, PermissionError, OSError):
        if str(candidate).startswith("/var/app"):
            fallback = Path.cwd() / local_relative_path
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
        raise


def _resolve_runtime_file_path(configured_path, local_relative_path):
    candidate = Path(configured_path)
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate
    except (FileNotFoundError, PermissionError, OSError):
        if str(candidate).startswith("/var/app"):
            fallback = Path.cwd() / local_relative_path
            fallback.parent.mkdir(parents=True, exist_ok=True)
            return fallback
        raise


def get_optional_tm_prediction_base_dir(app_config=None):
    config = app_config or (current_app.config if has_app_context() else {})
    base_dir = (
        config.get("OPTIONAL_TM_PREDICTION_BASE_DIR")
        or os.getenv("OPTIONAL_TM_PREDICTION_BASE_DIR")
        or DEFAULT_OPTIONAL_TM_PREDICTION_BASE_DIR
    )
    return _resolve_runtime_directory_path(base_dir, "data/tm_predictions/external")


def get_optional_tm_prediction_paths(predictor_name, app_config=None):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    predictor_dir = get_optional_tm_prediction_base_dir(app_config=app_config) / normalized_name.lower()
    predictor_dir.mkdir(parents=True, exist_ok=True)
    return {
        "predictor": normalized_name,
        "predictor_dir": predictor_dir,
        "fasta_path": predictor_dir / "pending.fasta",
        "csv_template_path": predictor_dir / "template.csv",
        "results_path": predictor_dir / "results.csv",
        "export_manifest_path": predictor_dir / "export_manifest.json",
        "import_manifest_path": predictor_dir / "import_manifest.json",
    }


def write_optional_tm_prediction_manifest(manifest_path, payload):
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(payload, indent=2))
    return manifest_file


def _emit_progress(progress_callback, message):
    if progress_callback is not None:
        progress_callback(message)


def _selected_tm_predictor_columns(include_tmbed=True, include_deeptmhmm=True):
    columns = []
    if include_tmbed:
        columns.append("TMbed_tm_count")
    if include_deeptmhmm:
        columns.append("DeepTMHMM_tm_count")
    return columns


def _selected_tm_predictor_completion_columns(include_tmbed=True, include_deeptmhmm=True):
    columns = []
    if include_tmbed:
        columns.extend(["TMbed_tm_count", "TMbed_tm_regions"])
    if include_deeptmhmm:
        columns.extend(["DeepTMHMM_tm_count", "DeepTMHMM_tm_regions"])
    return columns


def _normalize_pdb_code_selection(pdb_codes=None):
    if not pdb_codes:
        return []
    normalized = []
    for value in pdb_codes:
        text = str(value or "").strip().upper()
        if text:
            normalized.append(text)
    # Preserve order while removing duplicates
    return list(dict.fromkeys(normalized))


def _ensure_tm_predictor_columns_exist(conn, table_name, predictor_names):
    cur = conn.cursor()
    try:
        for predictor_name in predictor_names:
            count_col, region_col = tm_predictor_column_names(predictor_name)
            cur.execute(
                sql.SQL("ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} INTEGER").format(
                    tbl=sql.Identifier(table_name),
                    col=sql.Identifier(count_col),
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} TEXT").format(
                    tbl=sql.Identifier(table_name),
                    col=sql.Identifier(region_col),
                )
            )
        conn.commit()
    finally:
        cur.close()


def _normalize_tm_prediction_update_rows(rows, predictor_name):
    count_col, region_col = tm_predictor_column_names(predictor_name)
    normalized_rows = []
    for row in rows or []:
        pdb_code = str((row or {}).get("pdb_code") or "").strip().upper()
        if not pdb_code:
            continue
        tm_regions = parse_tm_regions_value((row or {}).get("tm_regions"))
        tm_count = (row or {}).get("tm_count")
        if tm_count is None or (isinstance(tm_count, float) and pd.isna(tm_count)):
            tm_count = len(tm_regions) if tm_regions else None
        else:
            try:
                tm_count = int(tm_count)
            except (TypeError, ValueError):
                tm_count = len(tm_regions) if tm_regions else None
        normalized_rows.append(
            {
                "pdb_code": pdb_code,
                count_col: tm_count,
                region_col: serialize_tm_regions(tm_regions),
            }
        )
    return normalized_rows


def update_tm_prediction_records(
    predictor_name,
    rows,
    table_name="membrane_protein_tmalphafold_predictions",
    progress_callback=None,
):
    normalized_rows = _normalize_tm_prediction_update_rows(rows, predictor_name)
    normalized_name = str(predictor_name or "").strip()
    if not normalized_rows:
        summary = {
            "predictor": normalized_name,
            "updated_records": 0,
            "table_name": table_name,
        }
        _emit_progress(progress_callback, f"No {normalized_name} TM prediction rows to update.")
        return summary

    count_col, region_col = tm_predictor_column_names(normalized_name)

    summary = {
        "predictor": normalized_name,
        "updated_records": int(len(normalized_rows)),
        "table_name": table_name,
        "tm_count_column": count_col,
        "tm_regions_column": region_col,
    }
    from src.Jobs.TMAlphaFoldSync import mirror_local_tm_prediction_rows

    mirror_rows = [
        {
            "pdb_code": row["pdb_code"],
            count_col: row[count_col],
            region_col: row[region_col],
        }
        for row in normalized_rows
    ]
    summary["normalized_store"] = mirror_local_tm_prediction_rows(
        method=normalized_name,
        records=mirror_rows,
        provider="MetaMP",
        prediction_kind="sequence_topology",
        progress_callback=progress_callback,
    )
    _emit_progress(
        progress_callback,
        f"Updated {summary['updated_records']} {normalized_name} TM prediction record(s) in normalized storage.",
    )
    return summary


def normalize_external_tm_prediction_dataframe(dataframe, predictor_name):
    normalize_optional_tm_predictor_name(predictor_name)

    df = pd.DataFrame(dataframe).copy()
    if df.empty:
        return pd.DataFrame(columns=["pdb_code", "tm_count", "tm_regions"])

    lowercase_map = {str(column).strip().lower(): column for column in df.columns}
    pdb_column = lowercase_map.get("pdb_code") or lowercase_map.get("pdb")
    tm_count_column = lowercase_map.get("tm_count")
    tm_regions_column = (
        lowercase_map.get("tm_regions")
        or lowercase_map.get("tm_regions_json")
        or lowercase_map.get("regions")
    )

    if pdb_column is None:
        raise ValueError("Input must include a 'pdb_code' column.")
    if tm_count_column is None and tm_regions_column is None:
        raise ValueError("Input must include either 'tm_count' or 'tm_regions'.")

    normalized = pd.DataFrame()
    normalized["pdb_code"] = df[pdb_column].astype(str).str.strip().str.upper()
    normalized = normalized[normalized["pdb_code"] != ""]

    if tm_count_column is not None:
        normalized["tm_count"] = pd.to_numeric(df[tm_count_column], errors="coerce")
        normalized["tm_count"] = normalized["tm_count"].apply(
            lambda value: None if pd.isna(value) else int(value)
        )
    else:
        normalized["tm_count"] = None

    if tm_regions_column is not None:
        normalized["tm_regions"] = df[tm_regions_column].apply(parse_tm_regions_value)
    else:
        normalized["tm_regions"] = [[] for _ in range(len(normalized))]

    normalized["tm_regions"] = normalized["tm_regions"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    normalized["tm_count"] = normalized.apply(
        lambda row: row["tm_count"]
        if row["tm_count"] is not None and not pd.isna(row["tm_count"])
        else (len(row["tm_regions"]) if row["tm_regions"] else None),
        axis=1,
    )
    normalized = normalized.drop_duplicates(subset="pdb_code", keep="last")
    return normalized.reset_index(drop=True)


def import_optional_tm_prediction_results(
    predictor_name,
    input_path=None,
    table_name="membrane_proteins",
    progress_callback=None,
):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    path_config = get_optional_tm_prediction_paths(normalized_name)
    input_file = Path(input_path) if input_path else path_config["results_path"]
    if not input_file.exists():
        raise FileNotFoundError(f"Prediction input file not found: {input_file}")

    dataframe = pd.read_csv(input_file)
    normalized = normalize_external_tm_prediction_dataframe(dataframe, normalized_name)
    count_col, region_col = tm_predictor_column_names(normalized_name)
    update_tm_prediction_records(
        predictor_name=normalized_name,
        rows=normalized.to_dict(orient="records"),
        table_name=table_name,
        progress_callback=progress_callback,
    )

    summary = {
        "predictor": normalized_name,
        "input_path": str(input_file),
        "processed_records": int(len(normalized)),
        "tm_count_column": count_col,
        "tm_regions_column": region_col,
        "import_manifest_path": str(path_config["import_manifest_path"]),
    }
    write_optional_tm_prediction_manifest(path_config["import_manifest_path"], summary)
    _emit_progress(
        progress_callback,
        f"Imported {summary['processed_records']} {normalized_name} TM prediction record(s).",
    )
    return summary


def load_optional_tm_prediction_frame(
    predictor_name,
    include_completed=False,
    pdb_codes=None,
    limit=None,
    progress_callback=None,
):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)

    count_col, region_col = tm_predictor_column_names(normalized_name)
    table_names = ["membrane_proteins", "membrane_protein_opm"]
    result_df = get_tables_as_dataframe(table_names, "pdb_code")
    result_df_uniprot = get_table_as_dataframe("membrane_protein_uniprot")

    common = (set(result_df.columns) - {"pdb_code"}) & set(result_df_uniprot.columns)
    right_pruned = result_df_uniprot.drop(columns=list(common))
    all_data = pd.merge(
        right=result_df,
        left=right_pruned,
        on="pdb_code",
        how="outer",
    )
    normalized_prediction_df = _load_normalized_tm_prediction_frame(
        predictor_names=[normalized_name],
        provider="MetaMP",
        pdb_codes=pdb_codes,
    )
    if not normalized_prediction_df.empty:
        all_data = all_data.merge(
            normalized_prediction_df,
            on="pdb_code",
            how="left",
            suffixes=("", "_normalized"),
        )

    if count_col not in all_data.columns:
        all_data[count_col] = pd.NA
    if region_col not in all_data.columns:
        all_data[region_col] = ""
    if "sequence_sequence" not in all_data.columns:
        all_data["sequence_sequence"] = pd.NA

    required_cols = ["pdb_code", "sequence_sequence", count_col, region_col]
    all_data = all_data[required_cols].copy()

    selected_codes = _normalize_pdb_code_selection(pdb_codes)
    if selected_codes:
        all_data = all_data.loc[
            all_data["pdb_code"].astype(str).str.strip().str.upper().isin(selected_codes)
        ].copy()
        _emit_progress(
            progress_callback,
            f"Restricted {normalized_name} export to {len(all_data)} explicitly selected protein record(s).",
        )

    _emit_progress(
        progress_callback,
        f"Loaded {len(all_data)} merged protein record(s) for {normalized_name} export.",
    )

    if not include_completed:
        pending_mask = all_data[[count_col, region_col]].fillna("").astype(str).apply(
            lambda column: column.str.strip().eq("")
        ).all(axis=1)
        all_data = all_data.loc[pending_mask].copy()
        _emit_progress(
            progress_callback,
            f"Found {len(all_data)} protein record(s) still missing {normalized_name} results.",
        )

    sequence_missing_before = (
        all_data["sequence_sequence"].isna()
        | all_data["sequence_sequence"].astype(str).str.strip().eq("")
    )

    all_data = fill_missing_sequences(
        all_data,
        pdb_col="pdb_code",
        seq_col="sequence_sequence",
        progress_callback=progress_callback,
    )

    if not all_data.empty:
        fetched_sequences = all_data.loc[
            sequence_missing_before
            & all_data["sequence_sequence"].notna()
            & ~all_data["sequence_sequence"].astype(str).str.strip().eq("")
        ][["pdb_code", "sequence_sequence"]].drop_duplicates(subset="pdb_code")
        persist_sequences_to_db(
            fetched_sequences,
            pdb_col="pdb_code",
            seq_col="sequence_sequence",
            progress_callback=progress_callback,
        )

    all_data = all_data.loc[
        all_data["sequence_sequence"].notna()
        & ~all_data["sequence_sequence"].astype(str).str.strip().eq("")
    ].copy()
    if limit is not None:
        all_data = all_data.head(int(limit)).copy()
        _emit_progress(
            progress_callback,
            f"Restricted {normalized_name} export to the first {len(all_data)} protein record(s).",
        )
    _emit_progress(
        progress_callback,
        f"Prepared {len(all_data)} protein sequence(s) for {normalized_name} export.",
    )
    return all_data.reset_index(drop=True)


def export_optional_tm_prediction_inputs(
    predictor_name,
    fasta_out=None,
    csv_out=None,
    include_completed=False,
    pdb_codes=None,
    limit=None,
    progress_callback=None,
):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    path_config = get_optional_tm_prediction_paths(normalized_name)
    frame = load_optional_tm_prediction_frame(
        predictor_name=normalized_name,
        include_completed=include_completed,
        pdb_codes=pdb_codes,
        limit=limit,
        progress_callback=progress_callback,
    )

    fasta_path = Path(fasta_out) if fasta_out else path_config["fasta_path"]
    csv_path = Path(csv_out) if csv_out else path_config["csv_template_path"]
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with fasta_path.open("w") as fh:
        for _, row in frame.iterrows():
            fh.write(f">{row['pdb_code']}\n{row['sequence_sequence']}\n")

    template = pd.DataFrame(
        {
            "pdb_code": frame["pdb_code"].astype(str),
            "tm_count": [None] * len(frame),
            "tm_regions": [""] * len(frame),
        }
    )
    template.to_csv(csv_path, index=False)

    summary = {
        "predictor": normalized_name,
        "record_count": int(len(frame)),
        "fasta_path": str(fasta_path),
        "csv_template_path": str(csv_path),
        "results_input_path": str(path_config["results_path"]),
        "include_completed": bool(include_completed),
        "export_manifest_path": str(path_config["export_manifest_path"]),
    }
    write_optional_tm_prediction_manifest(path_config["export_manifest_path"], summary)
    _emit_progress(
        progress_callback,
        f"Exported {summary['record_count']} {normalized_name} input sequence(s) and CSV template.",
    )
    return summary


def fetch_pdb_entry_sequence(pdb_id: str, timeout: int = 30) -> str:
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id.lower()}/download"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
        fasta = resp.text
    except requests.RequestException:
        return None

    return "".join(
        line.strip() for line in fasta.splitlines() if not line.startswith(">")
    )


def fill_missing_sequences(
    df: pd.DataFrame,
    pdb_col: str = "pdb_code",
    seq_col: str = "sequence_sequence",
    progress_callback=None,
) -> pd.DataFrame:
    if seq_col not in df.columns:
        df[seq_col] = pd.NA

    need_seq = df[seq_col].isna() | df[seq_col].astype(str).str.strip().eq("")
    if need_seq.any():
        _emit_progress(
            progress_callback,
            f"Fetching missing sequences for {int(need_seq.sum())} protein(s) from RCSB.",
        )
        for idx, pdb_id in tqdm(
            df.loc[need_seq, pdb_col].items(),
            total=need_seq.sum(),
            desc="Fetching PDB sequences",
        ):
            seq = fetch_pdb_entry_sequence(pdb_id)
            if seq is not None:
                df.at[idx, seq_col] = seq

    return df


def load_resume_progress_ids(
    resume_from_csv,
    id_col,
    predictor_columns=None,
    progress_callback=None,
):
    if not resume_from_csv:
        return set()

    resume_path = Path(resume_from_csv)
    if not resume_path.exists():
        return set()

    try:
        done_df = pd.read_csv(resume_path)
    except Exception as exc:
        _emit_progress(
            progress_callback,
            f"Unable to read TM resume progress from {resume_path}: {exc}",
        )
        return set()

    if id_col not in done_df.columns:
        return set()

    predictor_columns = [
        column for column in (predictor_columns or []) if column in done_df.columns
    ]
    if predictor_columns:
        complete_mask = done_df[predictor_columns].fillna("").astype(str).apply(
            lambda column: column.str.strip().ne("")
        ).all(axis=1)
        done_df = done_df.loc[complete_mask]

    done_ids = {
        str(value).strip()
        for value in done_df[id_col].dropna().tolist()
        if str(value).strip()
    }
    if done_ids:
        _emit_progress(
            progress_callback,
            f"Loaded {len(done_ids)} processed TM prediction record(s) from {resume_path.name}.",
        )
    return done_ids


def build_db_params(app_config=None):
    config = app_config or (current_app.config if has_app_context() else {})
    return {
        "host": config.get("DB_HOST") or os.getenv("DB_HOST", "localhost"),
        "port": int(config.get("DB_PORT") or os.getenv("DB_PORT", 5432)),
        "dbname": config.get("DB_NAME") or os.getenv("DB_NAME", "mpvis_db"),
        "user": config.get("DB_USER") or os.getenv("DB_USER", "mpvis_user"),
        "password": config.get("DB_PASSWORD")
        or os.getenv("DB_PASSWORD", "mpvis_user"),
    }


def get_tm_prediction_output_csv(app_config=None):
    config = app_config or (current_app.config if has_app_context() else {})
    output_path = (
        config.get("TM_PREDICTION_OUTPUT_CSV")
        or os.getenv("TM_PREDICTION_OUTPUT_CSV")
        or DEFAULT_TM_PREDICTION_OUTPUT_CSV
    )
    return _resolve_runtime_file_path(
        output_path,
        "data/tm_predictions/tm_summary_local.csv",
    )


def get_tm_prediction_runtime_options(app_config=None):
    config = app_config or (current_app.config if has_app_context() else {})
    return {
        "batch_size": int(
            config.get("TM_PREDICTION_BATCH_SIZE")
            or os.getenv("TM_PREDICTION_BATCH_SIZE", 10)
        ),
        "max_workers": int(
            config.get("TM_PREDICTION_MAX_WORKERS")
            or os.getenv("TM_PREDICTION_MAX_WORKERS", 1)
        ),
        "use_gpu": str(
            config.get("TM_PREDICTION_USE_GPU")
            or os.getenv("TM_PREDICTION_USE_GPU", "false")
        ).lower()
        in {"1", "true", "t", "yes", "y"},
        "include_deeptmhmm": str(
            config.get("TM_PREDICTION_INCLUDE_DEEPTMHMM")
            or os.getenv("TM_PREDICTION_INCLUDE_DEEPTMHMM", "true")
        ).lower()
        in {"1", "true", "t", "yes", "y"},
    }


def persist_sequences_to_db(
    dataframe: pd.DataFrame,
    pdb_col: str = "pdb_code",
    seq_col: str = "sequence_sequence",
    progress_callback=None,
):
    if dataframe.empty or pdb_col not in dataframe.columns or seq_col not in dataframe.columns:
        return 0

    updates = [
        (str(row[seq_col]).strip(), str(row[pdb_col]).strip())
        for _, row in dataframe.iterrows()
        if pd.notna(row[pdb_col])
        and pd.notna(row[seq_col])
        and str(row[pdb_col]).strip()
        and str(row[seq_col]).strip()
    ]
    if not updates:
        return 0

    conn = psycopg2.connect(**build_db_params())
    try:
        cur = conn.cursor()
        query = sql.SQL(
            "UPDATE {tbl} SET {seq_col} = %s WHERE {id_col} = %s"
        ).format(
            tbl=sql.Identifier("membrane_proteins"),
            seq_col=sql.Identifier(seq_col),
            id_col=sql.Identifier(pdb_col),
        )
        cur.executemany(query.as_string(conn), updates)
        conn.commit()
    finally:
        conn.close()

    _emit_progress(
        progress_callback,
        f"Persisted {len(updates)} fetched protein sequence(s) into the database.",
    )
    return len(updates)


def load_pending_tm_prediction_frame(
    resume_from_csv=None,
    include_tmbed=True,
    include_deeptmhmm=True,
    include_completed=False,
    pdb_codes=None,
    limit=None,
    progress_callback=None,
):
    if not include_tmbed and not include_deeptmhmm:
        raise ValueError("At least one TM predictor must be selected.")

    table_names = ["membrane_proteins", "membrane_protein_opm"]
    result_df = get_tables_as_dataframe(table_names, "pdb_code")
    result_df_uniprot = get_table_as_dataframe("membrane_protein_uniprot")

    common = (set(result_df.columns) - {"pdb_code"}) & set(result_df_uniprot.columns)
    right_pruned = result_df_uniprot.drop(columns=list(common))
    all_data = pd.merge(
        right=result_df,
        left=right_pruned,
        on="pdb_code",
        how="outer",
    )
    normalized_prediction_df = _load_normalized_tm_prediction_frame(
        predictor_names=(
            (["TMbed"] if include_tmbed else [])
            + (["DeepTMHMM"] if include_deeptmhmm else [])
        ),
        provider="MetaMP",
        pdb_codes=pdb_codes,
    )
    if not normalized_prediction_df.empty:
        all_data = all_data.merge(
            normalized_prediction_df,
            on="pdb_code",
            how="left",
            suffixes=("", "_normalized"),
        )

    required_cols = ["pdb_code", "sequence_sequence"] + _selected_tm_predictor_completion_columns(
        include_tmbed=include_tmbed,
        include_deeptmhmm=include_deeptmhmm,
    )
    available = [column for column in required_cols if column in all_data.columns]
    all_data = all_data[available]
    if "pdb_code" in all_data.columns:
        before_dedup = len(all_data)
        all_data = all_data.drop_duplicates(subset="pdb_code", keep="first").copy()
        deduped = before_dedup - len(all_data)
        if deduped:
            _emit_progress(
                progress_callback,
                f"Collapsed {deduped} duplicate merged protein row(s) down to one row per pdb_code before TM prediction.",
            )

    selected_codes = _normalize_pdb_code_selection(pdb_codes)
    if selected_codes:
        all_data = all_data.loc[
            all_data["pdb_code"].astype(str).str.strip().str.upper().isin(selected_codes)
        ].copy()
        _emit_progress(
            progress_callback,
            f"Restricted TM prediction backfill to {len(all_data)} explicitly selected protein record(s).",
        )

    _emit_progress(progress_callback, f"Loaded {len(all_data)} merged protein record(s).")

    pending_columns = _selected_tm_predictor_completion_columns(
        include_tmbed=include_tmbed,
        include_deeptmhmm=include_deeptmhmm,
    )
    pending_columns = [column for column in pending_columns if column in all_data.columns]
    if pending_columns and not include_completed:
        all_data = all_data.loc[
            all_data[pending_columns].fillna("").astype(str).apply(
                lambda column: column.str.strip().eq("")
            ).any(axis=1)
        ]
    _emit_progress(
        progress_callback,
        (
            f"Found {len(all_data)} protein record(s) still missing TM predictor counts."
            if not include_completed
            else f"Selected {len(all_data)} protein record(s) for forced TM predictor rerun."
        ),
    )

    done_ids = load_resume_progress_ids(
        resume_from_csv,
        id_col="pdb_code",
        predictor_columns=pending_columns,
        progress_callback=progress_callback,
    )
    if done_ids:
        before_resume_filter = len(all_data)
        all_data = all_data.loc[
            ~all_data["pdb_code"].astype(str).str.strip().isin(done_ids)
        ].copy()
        skipped = before_resume_filter - len(all_data)
        _emit_progress(
            progress_callback,
            f"Skipped {skipped} already processed TM prediction record(s) before sequence fetching.",
        )

    sequence_missing_before = (
        all_data["sequence_sequence"].isna()
        | all_data["sequence_sequence"].astype(str).str.strip().eq("")
    ) if "sequence_sequence" in all_data.columns else pd.Series(False, index=all_data.index)

    all_data = fill_missing_sequences(
        all_data,
        pdb_col="pdb_code",
        seq_col="sequence_sequence",
        progress_callback=progress_callback,
    )

    if not all_data.empty and "sequence_sequence" in all_data.columns:
        fetched_sequences = all_data.loc[
            sequence_missing_before
            & all_data["sequence_sequence"].notna()
            & ~all_data["sequence_sequence"].astype(str).str.strip().eq("")
        ][["pdb_code", "sequence_sequence"]].drop_duplicates(subset="pdb_code")
        persist_sequences_to_db(
            fetched_sequences,
            pdb_col="pdb_code",
            seq_col="sequence_sequence",
            progress_callback=progress_callback,
        )

    all_data = all_data[all_data["sequence_sequence"].notna()]
    all_data = all_data[all_data["sequence_sequence"] != ""]
    if limit is not None:
        all_data = all_data.head(int(limit)).copy()
        _emit_progress(
            progress_callback,
            f"Restricted TM prediction run to the first {len(all_data)} protein record(s).",
        )
    _emit_progress(
        progress_callback,
        f"Prepared {len(all_data)} protein sequence(s) for TM prediction.",
    )
    return all_data


def run_tm_prediction_for_sequences(
    dataframe,
    include_tmbed=True,
    include_deeptmhmm=None,
    use_gpu=None,
    max_workers=None,
    progress_callback=None,
):
    dataframe = pd.DataFrame(dataframe).copy()
    runtime_options = get_tm_prediction_runtime_options()
    if not include_tmbed and include_deeptmhmm is False:
        raise ValueError("At least one TM predictor must be selected.")
    include_deeptmhmm = (
        runtime_options["include_deeptmhmm"]
        if include_deeptmhmm is None
        else include_deeptmhmm
    )
    use_gpu = runtime_options["use_gpu"] if use_gpu is None else use_gpu
    max_workers = (
        runtime_options["max_workers"] if max_workers is None else max_workers
    )

    analyzer = MultiModelAnalyzer(
        db_params={},
        table="unused",
        batch_size=max(1, len(dataframe)),
        max_workers=max_workers,
        max_sequences=4,
        use_db=False,
        write_csv=False,
    )
    if include_tmbed:
        analyzer.register(TMbedPredictor(format_code=0, use_gpu=use_gpu))
    if include_deeptmhmm:
        analyzer.register(DeepTMHMMPredictor())

    predictor_names = [predictor.name for predictor in analyzer.predictors]
    _emit_progress(
        progress_callback,
        "Running TM sequence predictors: " + ", ".join(predictor_names) + ".",
    )
    result_df = analyzer.analyze(
        dataframe.copy(),
        id_col="id",
        seq_col="sequence",
    )
    for column in ["TMbed_tm_regions", "DeepTMHMM_tm_regions"]:
        if column in result_df.columns:
            result_df[column] = result_df[column].apply(parse_tm_regions_value)
    return {
        "predictors": predictor_names,
        "processed_records": int(len(result_df)),
        "records": result_df[
            [column for column in TM_PREDICTION_RECORD_COLUMNS if column in result_df.columns]
        ].to_dict(orient="records"),
    }


def run_tm_prediction_backfill(
    include_tmbed=True,
    include_deeptmhmm=None,
    use_gpu=None,
    batch_size=None,
    max_workers=None,
    csv_out=None,
    resume_from_csv=None,
    include_completed=False,
    pdb_codes=None,
    limit=None,
    progress_callback=None,
):
    runtime_options = get_tm_prediction_runtime_options()
    if not include_tmbed and include_deeptmhmm is False:
        raise ValueError("At least one TM predictor must be selected.")
    include_deeptmhmm = (
        runtime_options["include_deeptmhmm"]
        if include_deeptmhmm is None
        else include_deeptmhmm
    )
    use_gpu = runtime_options["use_gpu"] if use_gpu is None else use_gpu
    batch_size = runtime_options["batch_size"] if batch_size is None else batch_size
    max_workers = (
        runtime_options["max_workers"] if max_workers is None else max_workers
    )

    csv_out_path = Path(csv_out) if csv_out else None
    if csv_out_path is not None:
        csv_out_path.parent.mkdir(parents=True, exist_ok=True)
    resume_from_csv = str(resume_from_csv) if resume_from_csv else None
    if resume_from_csv and csv_out_path is None:
        csv_out_path = Path(resume_from_csv)
        csv_out_path.parent.mkdir(parents=True, exist_ok=True)

    _emit_progress(progress_callback, "Loading protein records for TM prediction backfill.")
    all_data = load_pending_tm_prediction_frame(
        resume_from_csv=resume_from_csv,
        include_tmbed=include_tmbed,
        include_deeptmhmm=include_deeptmhmm,
        include_completed=include_completed,
        pdb_codes=pdb_codes,
        limit=limit,
        progress_callback=progress_callback,
    )
    if all_data.empty:
        summary = {
            "queued_records": 0,
            "processed_records": 0,
            "predictors": (
                (["TMbed"] if include_tmbed else [])
                + (["DeepTMHMM"] if include_deeptmhmm else [])
            ),
            "message": "No pending protein records require TM prediction backfill.",
            "include_completed": bool(include_completed),
        }
        if csv_out_path is not None:
            summary["csv_path"] = str(csv_out_path)
        if resume_from_csv:
            summary["resume_from_csv"] = resume_from_csv
        _emit_progress(progress_callback, summary["message"])
        return summary

    analyzer = MultiModelAnalyzer(
        db_params=build_db_params(),
        table="membrane_proteins",
        batch_size=batch_size,
        max_workers=max_workers,
        use_db=False,
        write_csv=bool(csv_out_path),
        force_predictors=(
            set((["TMbed"] if include_tmbed else []) + (["DeepTMHMM"] if include_deeptmhmm else []))
            if include_completed
            else None
        ),
    )
    if include_tmbed:
        analyzer.register(TMbedPredictor(format_code=4, use_gpu=use_gpu))
    if include_deeptmhmm:
        analyzer.register(DeepTMHMMPredictor())

    predictor_names = [predictor.name for predictor in analyzer.predictors]
    _emit_progress(
        progress_callback,
        "Running TM predictors: " + ", ".join(predictor_names) + ".",
    )
    from src.Jobs.TMAlphaFoldSync import mirror_local_tm_prediction_rows

    normalized_store = {
        predictor_name: {"stored_rows": 0, "provider": "MetaMP", "method": predictor_name}
        for predictor_name in predictor_names
    }

    def _persist_batch_to_normalized_store(batch_df, start=None, end=None, total=None):
        if batch_df is None or batch_df.empty:
            return
        preview_columns = [
            column
            for column in ["pdb_code"] + TM_PREDICTION_RECORD_COLUMNS
            if column in batch_df.columns
        ]
        batch_records = batch_df[preview_columns].to_dict(orient="records")
        for predictor_name in predictor_names:
            store_summary = mirror_local_tm_prediction_rows(
                method=predictor_name,
                records=batch_records,
                provider="MetaMP",
                prediction_kind="sequence_topology",
                progress_callback=progress_callback,
            )
            normalized_store[predictor_name]["stored_rows"] += int(
                store_summary.get("stored_rows") or 0
            )
        if start is not None and end is not None and total is not None:
            _emit_progress(
                progress_callback,
                f"Persisted TMbed/normalized predictor results for batch {start}-{end} of {total}.",
            )

    result_df = analyzer.analyze(
        df=all_data,
        id_col="pdb_code",
        seq_col="sequence_sequence",
        csv_out=str(csv_out_path) if csv_out_path is not None else None,
        resume_from_csv=resume_from_csv,
        on_batch_completed=_persist_batch_to_normalized_store,
    )
    processed_ids = {
        str(value).strip().upper()
        for value in result_df.get("pdb_code", pd.Series(dtype=str)).dropna().tolist()
        if str(value).strip()
    }
    queued_ids = {
        str(value).strip().upper()
        for value in all_data.get("pdb_code", pd.Series(dtype=str)).dropna().tolist()
        if str(value).strip()
    }
    processed_this_run = len(processed_ids & queued_ids)
    current_run_records = result_df.loc[
        result_df.get("pdb_code", pd.Series(dtype=str)).astype(str).str.strip().str.upper().isin(queued_ids)
    ].copy()
    if not current_run_records.empty:
        preview_columns = [
            column
            for column in ["pdb_code"] + TM_PREDICTION_RECORD_COLUMNS
            if column in current_run_records.columns
        ]
        current_run_records = current_run_records[preview_columns]

    summary = {
        "queued_records": int(len(all_data)),
        "processed_records": int(processed_this_run),
        "predictors": predictor_names,
        "record_columns": [
            column for column in TM_PREDICTION_RECORD_COLUMNS if column in result_df.columns
        ],
        "include_completed": bool(include_completed),
        "normalized_store": normalized_store,
    }
    if not current_run_records.empty:
        summary["records"] = current_run_records.to_dict(orient="records")
    if csv_out_path is not None:
        summary["csv_path"] = str(csv_out_path)
        summary["csv_total_records"] = int(len(result_df))
    if resume_from_csv:
        summary["resume_from_csv"] = resume_from_csv
    _emit_progress(
        progress_callback,
        f"TM prediction backfill completed for {summary['processed_records']} record(s).",
    )
    return summary


def TMbedDeepTMHMM():
    return run_tm_prediction_backfill()


if __name__ == "__main__":
    from app import create_app

    app = create_app()
    with app.app_context():
        TMbedDeepTMHMM()
