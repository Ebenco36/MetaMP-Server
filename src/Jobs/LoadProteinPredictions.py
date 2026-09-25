import json
import os
import shlex
import shutil
import subprocess
import sys
import math
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
    get_tmbed_runtime_status,
    parse_tm_regions_value,
    serialize_tm_regions,
)
from src.AI_Packages.TMAlphaFoldPredictorClient import (  # noqa: E402
    TMALPHAFOLD_AUX_METHODS,
    TMALPHAFOLD_SEQUENCE_METHODS,
    TMAlphaFoldPredictionResult,
)
from src.Dashboard.services import get_table_as_dataframe, get_tables_as_dataframe  # noqa: E402
from src.MP.model_tmalphafold import TMAlphaFoldPrediction  # noqa: E402
from src.MP.model_uniprot import Uniprot  # noqa: E402
from src.MP.replacement_resolution import (  # noqa: E402
    canonicalize_pdb_codes,
    canonicalize_pdb_frame,
)


DEFAULT_TM_PREDICTION_OUTPUT_CSV = "/var/app/data/tm_predictions/tm_summary.csv"
DEFAULT_OPTIONAL_TM_PREDICTION_BASE_DIR = "/var/app/data/tm_predictions/external"
DEFAULT_OPTIONAL_TM_TOOL_HOME = "/opt/metamp-optional-tools"
OPTIONAL_TM_PREDICTORS = ("TMbed",) + TMALPHAFOLD_SEQUENCE_METHODS + TMALPHAFOLD_AUX_METHODS + ("TMDET", "CCTOP")
VERIFIED_LOCAL_TM_FALLBACK_PREDICTORS = ("TMbed", "DeepTMHMM", "TMHMM", "TMDET")
BULK_SAFE_TM_FALLBACK_PREDICTORS = ("DeepTMHMM", "TMHMM")
VERIFIED_LOCAL_TM_FALLBACK_EXECUTION_ORDER = ("DeepTMHMM", "TMHMM", "TMDET", "TMbed")
OPTIONAL_TM_LOCAL_COMMAND_BATCH_SIZES = {
    "TMDET": 25,
    "TMHMM": 25,
}
OPTIONAL_TM_PREDICTOR_SPECS = {
    "TMbed": {"execution_mode": "local_sequence", "prediction_kind": "sequence_topology"},
    "DeepTMHMM": {"execution_mode": "local_sequence", "prediction_kind": "sequence_topology"},
    "Hmmtop": {"execution_mode": "external_sequence", "prediction_kind": "sequence_topology"},
    "Memsat": {"execution_mode": "external_sequence", "prediction_kind": "sequence_topology"},
    "Octopus": {"execution_mode": "external_sequence", "prediction_kind": "sequence_topology"},
    "Philius": {"execution_mode": "external_sequence", "prediction_kind": "sequence_topology"},
    "Phobius": {"execution_mode": "external_sequence", "prediction_kind": "sequence_topology"},
    "Pro": {"execution_mode": "external_sequence", "prediction_kind": "sequence_topology"},
    "Prodiv": {"execution_mode": "external_sequence", "prediction_kind": "sequence_topology"},
    "Scampi": {"execution_mode": "external_sequence", "prediction_kind": "sequence_topology"},
    "ScampiMsa": {"execution_mode": "external_sequence", "prediction_kind": "sequence_topology"},
    "TMHMM": {"execution_mode": "external_sequence", "prediction_kind": "sequence_topology"},
    "Topcons2": {"execution_mode": "external_sequence", "prediction_kind": "sequence_topology"},
    "SignalP": {"execution_mode": "external_sequence", "prediction_kind": "signal_peptide"},
    "TMDET": {"execution_mode": "external_structure", "prediction_kind": "structure_membrane_plane"},
    "CCTOP": {"execution_mode": "external_sequence", "prediction_kind": "sequence_topology"},
}
OPTIONAL_TM_PREDICTOR_ALIASES = {
    "tmbed": "TMbed",
    "deeptmhmm": "DeepTMHMM",
    "hmmtop": "Hmmtop",
    "memsat": "Memsat",
    "octopus": "Octopus",
    "philius": "Philius",
    "phobius": "Phobius",
    "pro": "Pro",
    "prodiv": "Prodiv",
    "scampi": "Scampi",
    "scampimsa": "ScampiMsa",
    "scampi-msa": "ScampiMsa",
    "signalp": "SignalP",
    "tmdet": "TMDET",
    "tmhmm": "TMHMM",
    "topcons2": "Topcons2",
    "topcons": "Topcons2",
    "cctop": "CCTOP",
}
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
NORMALIZED_STORE_METHODS = tuple(dict.fromkeys(("TMbed", "DeepTMHMM") + OPTIONAL_TM_PREDICTORS))


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
        TMAlphaFoldPrediction.status == "success",
        TMAlphaFoldPrediction.method.in_(selected_predictors),
    )
    if provider is not None:
        if isinstance(provider, (list, tuple, set)):
            selected_providers = [
                str(item or "").strip()
                for item in provider
                if str(item or "").strip()
            ]
            if selected_providers:
                query = query.filter(TMAlphaFoldPrediction.provider.in_(selected_providers))
        else:
            query = query.filter(TMAlphaFoldPrediction.provider == provider)
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
    provider_priority = {"MetaMP": 0, "TMAlphaFold": 1}
    for pdb_code, methods in grouped.items():
        item = {"pdb_code": pdb_code}
        for method in selected_predictors:
            method_rows = methods.get(method) or []
            count_col, region_col = tm_predictor_column_names(method)
            if not method_rows:
                item[count_col] = None
                item[region_col] = ""
                continue
            preferred_row = sorted(
                method_rows,
                key=lambda row: (
                    provider_priority.get(str(row.provider or "").strip(), 99),
                    -(int(row.id) if getattr(row, "id", None) is not None else 0),
                ),
            )[0]
            item[count_col] = preferred_row.tm_count
            item[region_col] = str(preferred_row.tm_regions_json or "")
        records.append(item)

    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = canonicalize_pdb_frame(frame, pdb_column="pdb_code")
    return frame


def normalize_optional_tm_predictor_name(predictor_name):
    normalized_name = str(predictor_name or "").strip().lower()
    if normalized_name in OPTIONAL_TM_PREDICTOR_ALIASES:
        return OPTIONAL_TM_PREDICTOR_ALIASES[normalized_name]
    raise ValueError(
        f"Unsupported optional predictor '{predictor_name}'. "
        f"Expected one of: {', '.join(OPTIONAL_TM_PREDICTORS)}."
    )


def normalize_optional_tm_predictor_names(predictor_names=None):
    normalized = []
    include_all = False
    for predictor_name in predictor_names or ():
        text = str(predictor_name or "").strip()
        if not text:
            continue
        if text.lower() in {"all", "*"}:
            include_all = True
            continue
        normalized.append(normalize_optional_tm_predictor_name(text))
    if include_all or not normalized:
        normalized = list(OPTIONAL_TM_PREDICTORS)
    return list(dict.fromkeys(normalized))


def normalize_optional_tm_completion_provider(provider_name=None):
    normalized_name = str(provider_name or "").strip().lower()
    if normalized_name in {"", "metamp", "local"}:
        return "MetaMP"
    if normalized_name in {"any", "either", "all"}:
        return ("MetaMP", "TMAlphaFold")
    if normalized_name in {"tmalphafold", "upstream"}:
        return "TMAlphaFold"
    raise ValueError(
        "Unsupported completion provider scope "
        f"'{provider_name}'. Expected one of: MetaMP, any, TMAlphaFold."
    )


def normalize_verified_tm_fallback_predictor_names(predictor_names=None):
    normalized = normalize_optional_tm_predictor_names(
        predictor_names or VERIFIED_LOCAL_TM_FALLBACK_PREDICTORS
    )
    unsupported = [
        predictor_name
        for predictor_name in normalized
        if predictor_name not in VERIFIED_LOCAL_TM_FALLBACK_PREDICTORS
    ]
    if unsupported:
        raise ValueError(
            "Verified local TM fallback predictors are limited to: "
            + ", ".join(VERIFIED_LOCAL_TM_FALLBACK_PREDICTORS)
            + ". Unsupported values: "
            + ", ".join(unsupported)
            + "."
        )
    return normalized


def order_verified_tm_fallback_predictor_names(predictor_names=None):
    normalized = normalize_verified_tm_fallback_predictor_names(predictor_names)
    return [
        predictor_name
        for predictor_name in VERIFIED_LOCAL_TM_FALLBACK_EXECUTION_ORDER
        if predictor_name in normalized
    ]


def _prediction_kind_for_optional_method(predictor_name):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    spec = OPTIONAL_TM_PREDICTOR_SPECS.get(normalized_name) or {}
    return str(spec.get("prediction_kind") or "sequence_topology")


def get_optional_tm_predictor_spec(predictor_name):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    spec = dict(OPTIONAL_TM_PREDICTOR_SPECS.get(normalized_name) or {})
    spec["predictor"] = normalized_name
    return spec


def get_optional_tm_local_command_map(app_config=None):
    config = app_config or (current_app.config if has_app_context() else {})
    raw_value = (
        config.get("OPTIONAL_TM_LOCAL_COMMANDS_JSON")
        or os.getenv("OPTIONAL_TM_LOCAL_COMMANDS_JSON")
        or ""
    )
    if not str(raw_value).strip():
        return {}
    try:
        parsed = json.loads(raw_value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    normalized = {}
    for predictor_name, command_spec in parsed.items():
        try:
            normalized_name = normalize_optional_tm_predictor_name(predictor_name)
        except ValueError:
            continue
        if isinstance(command_spec, str) and command_spec.strip():
            normalized[normalized_name] = command_spec.strip()
        elif isinstance(command_spec, list) and command_spec:
            normalized[normalized_name] = [str(item) for item in command_spec if str(item).strip()]
    return normalized


def get_optional_tm_tool_home(app_config=None):
    config = app_config or (current_app.config if has_app_context() else {})
    tool_home = (
        config.get("OPTIONAL_TM_TOOL_HOME")
        or os.getenv("OPTIONAL_TM_TOOL_HOME")
        or DEFAULT_OPTIONAL_TM_TOOL_HOME
    )
    return _resolve_runtime_directory_path(tool_home, "vendor/optional_tm_tools")


def get_optional_tm_tool_paths(app_config=None):
    configured_tool_home = get_optional_tm_tool_home(app_config=app_config)
    candidate_homes = []
    live_vendor_home = Path("/var/app/vendor/optional_tm_tools")
    cwd_vendor_home = Path.cwd() / "vendor/optional_tm_tools"
    for candidate in (live_vendor_home, cwd_vendor_home, configured_tool_home):
        if candidate not in candidate_homes:
            candidate_homes.append(candidate)
    existing_homes = [candidate for candidate in candidate_homes if candidate.exists()]
    tool_home = existing_homes[0] if existing_homes else configured_tool_home
    bin_dir = tool_home / "bin"
    wrappers_dir = tool_home / "wrappers"
    bin_dir.mkdir(parents=True, exist_ok=True)
    wrappers_dir.mkdir(parents=True, exist_ok=True)
    return {
        "tool_home": tool_home,
        "bin_dir": bin_dir,
        "wrappers_dir": wrappers_dir,
        "candidate_homes": candidate_homes,
    }


def _optional_tm_predictor_executable_candidates(predictor_name):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    slug = normalized_name.lower()
    condensed = slug.replace("-", "").replace("_", "")
    candidates = [
        f"metamp-run-{slug}",
        f"metamp-{slug}",
        slug,
        condensed,
    ]
    if normalized_name == "ScampiMsa":
        candidates.extend(["scampi_msa", "scampi-msa", "scampimsa"])
    if normalized_name == "Topcons2":
        candidates.extend(["topcons", "topcons2"])
    if normalized_name == "SignalP":
        candidates.extend(["signalp", "signalp6", "signalp5"])
    return list(dict.fromkeys(candidates))


def discover_optional_tm_local_command(predictor_name, app_config=None):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    tool_paths = get_optional_tm_tool_paths(app_config=app_config)
    for candidate in _optional_tm_predictor_executable_candidates(normalized_name):
        vendor_bin = tool_paths["bin_dir"] / candidate
        if vendor_bin.exists() and os.access(vendor_bin, os.X_OK):
            command = [
                str(vendor_bin),
                "--input",
                "{input_fasta}",
                "--output",
                "{output_csv}",
                "--reference",
                "{reference_csv}",
            ]
            if normalized_name == "TMDET":
                command.extend(["--structure-manifest", "{structure_manifest}", "--failures", "{failures_csv}"])
            return command
        vendor_wrapper = tool_paths["wrappers_dir"] / candidate
        if vendor_wrapper.exists() and os.access(vendor_wrapper, os.X_OK):
            command = [
                str(vendor_wrapper),
                "--input",
                "{input_fasta}",
                "--output",
                "{output_csv}",
                "--reference",
                "{reference_csv}",
            ]
            if normalized_name == "TMDET":
                command.extend(["--structure-manifest", "{structure_manifest}", "--failures", "{failures_csv}"])
            return command
        system_command = shutil.which(candidate)
        if system_command:
            command = [
                str(system_command),
                "--input",
                "{input_fasta}",
                "--output",
                "{output_csv}",
                "--reference",
                "{reference_csv}",
            ]
            if normalized_name == "TMDET":
                command.extend(["--structure-manifest", "{structure_manifest}", "--failures", "{failures_csv}"])
            return command
    return None


def get_optional_tm_local_command(predictor_name, app_config=None):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    configured = get_optional_tm_local_command_map(app_config=app_config).get(normalized_name)
    if configured is not None:
        return configured
    return discover_optional_tm_local_command(normalized_name, app_config=app_config)


def _optional_tm_command_executable(command_spec):
    if isinstance(command_spec, str):
        try:
            tokens = shlex.split(command_spec)
        except ValueError:
            return None
        return tokens[0] if tokens else None
    if isinstance(command_spec, (list, tuple)) and command_spec:
        return str(command_spec[0])
    return None


def _verify_optional_tm_local_command(
    predictor_name,
    command_spec,
    *,
    command_origin="discovered",
):
    executable = _optional_tm_command_executable(command_spec)
    if not executable:
        return {
            "verified": False,
            "reason": "missing_executable",
            "detail": "No executable could be resolved from the local command specification.",
        }

    executable_name = Path(executable).name
    if command_origin == "configured":
        return {
            "verified": False,
            "reason": "configured_command_unverified",
            "detail": "Configured local commands are treated as user-managed and are not self-tested automatically.",
            "executable": executable,
        }
    if not executable_name.startswith("metamp-run-"):
        return {
            "verified": False,
            "reason": "non_metamp_wrapper_unverified",
            "detail": "Only MetaMP vendor wrappers are self-tested automatically.",
            "executable": executable,
        }

    try:
        completed = subprocess.run(
            [executable, "--self-test"],
            shell=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return {
            "verified": False,
            "reason": "wrapper_not_found",
            "detail": f"Discovered wrapper for {predictor_name} no longer exists on disk.",
            "executable": executable,
        }
    except subprocess.TimeoutExpired:
        return {
            "verified": False,
            "reason": "self_test_timeout",
            "detail": f"Self-test for {predictor_name} timed out after 30 seconds.",
            "executable": executable,
        }
    return {
        "verified": completed.returncode == 0,
        "reason": "self_test_passed" if completed.returncode == 0 else "self_test_failed",
        "detail": str(completed.stdout or completed.stderr or "").strip(),
        "return_code": int(completed.returncode),
        "executable": executable,
    }


def _resolve_optional_tm_command_runtime(predictor_name, app_config=None):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    spec = get_optional_tm_predictor_spec(normalized_name)
    builtin_local = str(spec.get("execution_mode") or "").strip() == "local_sequence"
    configured_command = get_optional_tm_local_command_map(app_config=app_config).get(normalized_name)
    discovered_command = None
    verification = None

    if builtin_local:
        local_command = None
        readiness = "builtin_ready"
        enablement = "Available in the current MetaMP runtime without additional wrappers."
        locally_runnable = True
    elif configured_command is not None:
        local_command = configured_command
        verification = _verify_optional_tm_local_command(
            normalized_name,
            configured_command,
            command_origin="configured",
        )
        readiness = "command_configured_unverified"
        enablement = (
            "A configured local command is present. MetaMP will use it, but it is not self-tested automatically."
        )
        locally_runnable = True
    else:
        discovered_command = discover_optional_tm_local_command(normalized_name, app_config=app_config)
        local_command = discovered_command
        if discovered_command is not None:
            verification = _verify_optional_tm_local_command(
                normalized_name,
                discovered_command,
                command_origin="discovered",
            )
            if verification.get("verified"):
                readiness = "command_verified"
                enablement = "A discovered MetaMP wrapper passed runtime self-test and is ready for execution."
                locally_runnable = True
            else:
                readiness = "command_discovered_unverified"
                enablement = (
                    "A local command was discovered, but MetaMP could not verify it automatically. "
                    "Inspect runtime_verification for details."
                )
                locally_runnable = False
        else:
            readiness = "missing_tooling"
            enablement = (
                "Install or vendor a wrapper/binary under vendor/optional_tm_tools, "
                "or set OPTIONAL_TM_LOCAL_COMMANDS_JSON for this predictor."
            )
            locally_runnable = False

    return {
        "predictor": normalized_name,
        "execution_mode": spec.get("execution_mode"),
        "prediction_kind": spec.get("prediction_kind"),
        "builtin_local": bool(builtin_local),
        "configured_local_command": configured_command,
        "discovered_local_command": discovered_command,
        "effective_local_command": local_command,
        "command_available": bool(local_command is not None),
        "locally_runnable": bool(locally_runnable),
        "readiness": readiness,
        "how_to_enable": enablement,
        "runtime_verification": verification,
    }


def is_optional_tm_predictor_locally_runnable(predictor_name):
    runtime = _resolve_optional_tm_command_runtime(predictor_name)
    return bool(runtime.get("locally_runnable"))


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
        "failures_path": predictor_dir / "failures.csv",
        "reference_path": predictor_dir / "reference_from_tmalphafold.csv",
        "structure_manifest_path": predictor_dir / "structure_manifest.csv",
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


def _preview_tm_regions_value(value, limit=120):
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _build_optional_tm_export_debug_payload(reference_frame, results_frame, predictor_name):
    if results_frame is None or results_frame.empty:
        return {
            "predictor": predictor_name,
            "upstream_reference_rows": 0,
            "reference_rows_with_tm_count": 0,
            "reference_rows_with_tm_regions": 0,
            "blank_result_rows_requiring_completion": 0,
            "preview_rows": [],
        }

    preview_rows = []
    preview_source = reference_frame if reference_frame is not None and not reference_frame.empty else results_frame
    preview_frame = preview_source.head(3)
    for _, row in preview_frame.iterrows():
        preview_rows.append(
            {
                "pdb_code": str(row.get("pdb_code") or "").strip().upper(),
                "tm_count": None if pd.isna(row.get("tm_count")) else row.get("tm_count"),
                "tm_regions": _preview_tm_regions_value(row.get("tm_regions")),
            }
        )

    reference_tm_count_series = (
        reference_frame["tm_count"]
        if reference_frame is not None and "tm_count" in reference_frame.columns
        else pd.Series(dtype=object)
    )
    reference_tm_regions_series = (
        reference_frame["tm_regions"].fillna("").astype(str).str.strip()
        if reference_frame is not None and "tm_regions" in reference_frame.columns
        else pd.Series(dtype=str)
    )
    result_tm_count_series = results_frame["tm_count"] if "tm_count" in results_frame.columns else pd.Series(dtype=object)
    tm_regions_series = (
        results_frame["tm_regions"].fillna("").astype(str).str.strip()
        if "tm_regions" in results_frame.columns
        else pd.Series(dtype=str)
    )
    blank_mask = result_tm_count_series.isna() & tm_regions_series.eq("")
    return {
        "predictor": predictor_name,
        "upstream_reference_rows": int(len(reference_frame)) if reference_frame is not None else 0,
        "reference_rows_with_tm_count": int(reference_tm_count_series.notna().sum()),
        "reference_rows_with_tm_regions": int(reference_tm_regions_series.ne("").sum()),
        "blank_result_rows_requiring_completion": int(blank_mask.sum()),
        "preview_rows": preview_rows,
    }


def _emit_optional_tm_export_debug(progress_callback, predictor_name, debug_payload, results_path, reference_path):
    _emit_progress(
        progress_callback,
        f"{predictor_name}: wrote blank results target to {results_path} and TMAlphaFold reference to {reference_path}. "
        f"TMAlphaFold references={debug_payload['upstream_reference_rows']}, "
        f"reference_rows_with_tm_count={debug_payload['reference_rows_with_tm_count']}, "
        f"reference_rows_with_tm_regions={debug_payload['reference_rows_with_tm_regions']}, "
        f"blank_result_rows_requiring_completion={debug_payload['blank_result_rows_requiring_completion']}.",
    )
    preview_rows = debug_payload.get("preview_rows") or []
    if preview_rows:
        for row in preview_rows:
            _emit_progress(
                progress_callback,
                f"{predictor_name} preview: pdb_code={row['pdb_code']} tm_count={row['tm_count']} tm_regions={row['tm_regions']}",
            )


def _build_optional_tm_structure_manifest(frame):
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=["pdb_code", "structure_url", "structure_format", "structure_source"]
        )
    manifest = pd.DataFrame()
    manifest["pdb_code"] = frame["pdb_code"].astype(str).str.strip().str.upper()
    manifest = manifest[manifest["pdb_code"] != ""].copy()
    manifest["structure_url"] = manifest["pdb_code"].apply(
        lambda pdb_code: f"https://files.rcsb.org/download/{pdb_code}.cif.gz"
    )
    manifest["structure_format"] = "mmcif_gz"
    manifest["structure_source"] = "RCSB"
    return manifest.reset_index(drop=True)


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
    normalized = canonicalize_pdb_codes(pdb_codes)
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
        prediction_kind=_prediction_kind_for_optional_method(normalized_name),
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
    normalized = normalized.loc[
        normalized.apply(
            lambda row: (row["tm_count"] is not None and not pd.isna(row["tm_count"]))
            or bool(row["tm_regions"]),
            axis=1,
        )
    ].copy()
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


def import_optional_tm_prediction_results_bulk(
    predictor_names=None,
    table_name="membrane_proteins",
    skip_missing_inputs=True,
    progress_callback=None,
):
    normalized_names = normalize_optional_tm_predictor_names(predictor_names)
    imported_predictors = 0
    total_processed_records = 0
    per_predictor = {}

    for normalized_name in normalized_names:
        path_config = get_optional_tm_prediction_paths(normalized_name)
        input_file = path_config["results_path"]
        if not Path(input_file).exists():
            summary = {
                "predictor": normalized_name,
                "input_path": str(input_file),
                "processed_records": 0,
                "skipped": True,
                "skip_reason": "missing_input_file",
                "import_manifest_path": str(path_config["import_manifest_path"]),
            }
            if skip_missing_inputs:
                _emit_progress(
                    progress_callback,
                    f"Skipping {normalized_name} import because no results file exists at {input_file}.",
                )
                per_predictor[normalized_name] = summary
                continue
            raise FileNotFoundError(f"Prediction input file not found: {input_file}")

        summary = import_optional_tm_prediction_results(
            predictor_name=normalized_name,
            input_path=str(input_file),
            table_name=table_name,
            progress_callback=progress_callback,
        )
        per_predictor[normalized_name] = summary
        imported_predictors += 1
        total_processed_records += int(summary.get("processed_records") or 0)

    aggregate_summary = {
        "predictors": normalized_names,
        "imported_predictor_count": int(imported_predictors),
        "total_processed_records": int(total_processed_records),
        "skip_missing_inputs": bool(skip_missing_inputs),
        "per_predictor": per_predictor,
    }
    _emit_progress(
        progress_callback,
        "Completed optional TM import across "
        + str(len(normalized_names))
        + " predictor(s); "
        + str(total_processed_records)
        + " total record(s) processed.",
    )
    return aggregate_summary


def _format_optional_tm_local_command(
    command_spec,
    *,
    predictor_name,
    fasta_path,
    results_path,
    reference_path,
    failures_path,
    structure_manifest_path,
    work_dir,
):
    substitutions = {
        "predictor": predictor_name,
        "input_fasta": str(fasta_path),
        "output_csv": str(results_path),
        "results_csv": str(results_path),
        "failures_csv": str(failures_path),
        "reference_csv": str(reference_path),
        "work_dir": str(work_dir),
        "structure_manifest": str(structure_manifest_path),
    }
    if isinstance(command_spec, str):
        return command_spec.format(**substitutions)
    return [str(item).format(**substitutions) for item in command_spec]


def _get_optional_tm_local_command_batch_size(predictor_name, batch_size_override=None):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    if batch_size_override not in (None, ""):
        try:
            return max(int(batch_size_override), 1)
        except (TypeError, ValueError):
            pass
    configured_value = (
        current_app.config.get("OPTIONAL_TM_LOCAL_COMMAND_BATCH_SIZE")
        if has_app_context()
        else None
    ) or os.getenv("OPTIONAL_TM_LOCAL_COMMAND_BATCH_SIZE")
    if configured_value not in (None, ""):
        try:
            return max(int(configured_value), 1)
        except (TypeError, ValueError):
            pass
    return int(OPTIONAL_TM_LOCAL_COMMAND_BATCH_SIZES.get(normalized_name) or 0)


def _read_fasta_records(fasta_path):
    records = []
    current_header = None
    current_sequence_parts = []
    with Path(fasta_path).open() as handle:
        for raw_line in handle:
            line = str(raw_line or "").strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    records.append((current_header, "".join(current_sequence_parts)))
                current_header = line[1:].strip()
                current_sequence_parts = []
                continue
            current_sequence_parts.append(line)
    if current_header is not None:
        records.append((current_header, "".join(current_sequence_parts)))
    return records


def _write_fasta_records(fasta_path, records):
    fasta_path = Path(fasta_path)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def _append_csv_rows(target_path, source_path, fieldnames):
    target_path = Path(target_path)
    source_path = Path(source_path)
    if not source_path.exists():
        return 0
    try:
        source_rows = pd.read_csv(source_path)
    except pd.errors.EmptyDataError:
        source_rows = pd.DataFrame(columns=fieldnames)
    if source_rows.empty:
        return 0
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        try:
            target_rows = pd.read_csv(target_path)
        except pd.errors.EmptyDataError:
            target_rows = pd.DataFrame(columns=fieldnames)
    else:
        target_rows = pd.DataFrame(columns=fieldnames)
    combined = pd.concat([target_rows, source_rows], ignore_index=True)
    if "pdb_code" in combined.columns:
        combined["pdb_code"] = combined["pdb_code"].astype(str).str.strip().str.upper()
        combined = combined.drop_duplicates(subset="pdb_code", keep="last")
    combined.to_csv(target_path, index=False)
    return int(len(source_rows))


def _prepare_optional_tm_command_chunks(
    predictor_name,
    export_summary,
    *,
    chunk_size,
):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    path_config = get_optional_tm_prediction_paths(normalized_name)
    fasta_records = _read_fasta_records(export_summary["fasta_path"])
    fasta_by_code = {
        str(header or "").strip().upper(): (header, sequence)
        for header, sequence in fasta_records
        if str(header or "").strip()
    }
    results_df = pd.read_csv(export_summary["results_input_path"])
    reference_df = pd.read_csv(export_summary["reference_input_path"])
    structure_manifest_df = pd.read_csv(export_summary["structure_manifest_path"])
    if structure_manifest_df.empty:
        return []

    structure_manifest_df["pdb_code"] = (
        structure_manifest_df["pdb_code"].astype(str).str.strip().str.upper()
    )
    if not results_df.empty:
        results_df["pdb_code"] = results_df["pdb_code"].astype(str).str.strip().str.upper()
    if not reference_df.empty:
        reference_df["pdb_code"] = reference_df["pdb_code"].astype(str).str.strip().str.upper()

    chunk_root = path_config["predictor_dir"] / "chunks"
    if chunk_root.exists():
        shutil.rmtree(chunk_root)
    chunk_root.mkdir(parents=True, exist_ok=True)

    chunks = []
    total_rows = int(len(structure_manifest_df))
    total_chunks = int(math.ceil(total_rows / chunk_size))
    for chunk_index, start in enumerate(range(0, total_rows, chunk_size), start=1):
        end = min(start + chunk_size, total_rows)
        manifest_chunk = structure_manifest_df.iloc[start:end].copy()
        chunk_codes = manifest_chunk["pdb_code"].tolist()
        chunk_dir = chunk_root / f"chunk-{chunk_index:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_fasta_records = [
            fasta_by_code[pdb_code]
            for pdb_code in chunk_codes
            if pdb_code in fasta_by_code
        ]
        fasta_path = chunk_dir / "pending.fasta"
        results_path = chunk_dir / "results.csv"
        failures_path = chunk_dir / "failures.csv"
        reference_path = chunk_dir / "reference_from_tmalphafold.csv"
        structure_manifest_path = chunk_dir / "structure_manifest.csv"
        csv_template_path = chunk_dir / "template.csv"

        _write_fasta_records(fasta_path, chunk_fasta_records)
        chunk_results_df = (
            results_df.loc[results_df["pdb_code"].isin(chunk_codes)].copy()
            if not results_df.empty
            else pd.DataFrame(columns=["pdb_code", "tm_count", "tm_regions"])
        )
        if chunk_results_df.empty:
            chunk_results_df = pd.DataFrame(
                {
                    "pdb_code": chunk_codes,
                    "tm_count": [None] * len(chunk_codes),
                    "tm_regions": [""] * len(chunk_codes),
                }
            )
        chunk_results_df.to_csv(csv_template_path, index=False)
        chunk_results_df.to_csv(results_path, index=False)
        pd.DataFrame(columns=["pdb_code", "error_message"]).to_csv(failures_path, index=False)

        chunk_reference_df = (
            reference_df.loc[reference_df["pdb_code"].isin(chunk_codes)].copy()
            if not reference_df.empty
            else pd.DataFrame(columns=["pdb_code", "tm_count", "tm_regions"])
        )
        chunk_reference_df.to_csv(reference_path, index=False)
        manifest_chunk.to_csv(structure_manifest_path, index=False)

        chunks.append(
            {
                "index": chunk_index,
                "total_chunks": total_chunks,
                "record_count": int(len(chunk_codes)),
                "predictor_dir": chunk_dir,
                "fasta_path": fasta_path,
                "results_path": results_path,
                "failures_path": failures_path,
                "reference_path": reference_path,
                "structure_manifest_path": structure_manifest_path,
                "csv_template_path": csv_template_path,
            }
        )
    return chunks


def _execute_optional_tm_local_command_once(
    *,
    predictor_name,
    command_spec,
    input_paths,
    progress_callback=None,
):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    formatted_command = _format_optional_tm_local_command(
        command_spec,
        predictor_name=normalized_name,
        fasta_path=input_paths["fasta_path"],
        results_path=input_paths["results_path"],
        reference_path=input_paths["reference_path"],
        failures_path=input_paths["failures_path"],
        structure_manifest_path=input_paths["structure_manifest_path"],
        work_dir=input_paths["predictor_dir"],
    )
    _emit_progress(
        progress_callback,
        f"Running configured local command for {normalized_name} using {input_paths['fasta_path']}.",
    )
    completed = _run_optional_tm_command_with_live_output(
        formatted_command=formatted_command,
        cwd=input_paths["predictor_dir"],
        predictor_name=normalized_name,
        progress_callback=progress_callback,
    )
    executed_command = completed["executed_command"]
    command_summary = {
        "predictor": normalized_name,
        "command": executed_command,
        "return_code": int(completed["return_code"]),
        "stdout": str(completed["combined_output"] or "").strip(),
        "stderr": str(completed["combined_output"] or "").strip(),
        "failures_path": str(input_paths["failures_path"]),
    }
    if completed["return_code"] != 0:
        raise RuntimeError(
            f"Local command execution failed for {normalized_name} with exit code {completed['return_code']}.\n"
            f"Command: {executed_command}\n"
            f"STDERR: {command_summary['stderr']}"
        )

    failure_summary = import_optional_tm_prediction_failures(
        predictor_name=normalized_name,
        failures_path=str(input_paths["failures_path"]),
        provider="MetaMP",
        progress_callback=progress_callback,
    )
    import_summary = import_optional_tm_prediction_results(
        predictor_name=normalized_name,
        input_path=str(input_paths["results_path"]),
        progress_callback=progress_callback,
    )
    return command_summary, failure_summary, import_summary


def _run_optional_tm_command_with_live_output(
    *,
    formatted_command,
    cwd,
    predictor_name,
    progress_callback=None,
):
    if isinstance(formatted_command, str):
        process = subprocess.Popen(
            formatted_command,
            shell=True,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        executed_command = formatted_command
    else:
        process = subprocess.Popen(
            formatted_command,
            shell=False,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        executed_command = " ".join(shlex.quote(str(item)) for item in formatted_command)

    output_lines = []
    try:
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = str(raw_line or "").rstrip()
                if not line:
                    continue
                output_lines.append(line)
                _emit_progress(progress_callback, f"[{predictor_name}] {line}")
        return_code = int(process.wait())
    finally:
        if process.stdout is not None:
            process.stdout.close()

    combined_output = "\n".join(output_lines).strip()
    return {
        "executed_command": executed_command,
        "return_code": return_code,
        "combined_output": combined_output,
    }


def _load_optional_tm_completion_codes(
    predictor_name,
    provider="MetaMP",
    pdb_codes=None,
    statuses=("success", "error"),
):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    normalized_provider = normalize_optional_tm_completion_provider(provider)
    normalized_statuses = [
        str(status or "").strip().lower()
        for status in (statuses or ())
        if str(status or "").strip()
    ]
    if not normalized_statuses:
        return set()

    query = TMAlphaFoldPrediction.query.with_entities(
        TMAlphaFoldPrediction.pdb_code,
    ).filter(
        TMAlphaFoldPrediction.provider == normalized_provider,
        TMAlphaFoldPrediction.method == normalized_name,
        db.func.lower(db.func.trim(TMAlphaFoldPrediction.status)).in_(normalized_statuses),
    )
    selected_codes = _normalize_pdb_code_selection(pdb_codes)
    if selected_codes:
        query = query.filter(
            db.func.upper(db.func.trim(TMAlphaFoldPrediction.pdb_code)).in_(selected_codes)
        )

    completed_codes = {
        str(row.pdb_code or "").strip().upper()
        for row in query.all()
        if str(row.pdb_code or "").strip()
    }
    if "error" not in normalized_statuses and normalized_statuses == ["success"]:
        permanent_error_query = TMAlphaFoldPrediction.query.with_entities(
            TMAlphaFoldPrediction.pdb_code,
            TMAlphaFoldPrediction.error_message,
        ).filter(
            TMAlphaFoldPrediction.provider == normalized_provider,
            TMAlphaFoldPrediction.method == normalized_name,
            db.func.lower(db.func.trim(TMAlphaFoldPrediction.status)) == "error",
        )
        if selected_codes:
            permanent_error_query = permanent_error_query.filter(
                db.func.upper(db.func.trim(TMAlphaFoldPrediction.pdb_code)).in_(selected_codes)
            )
        completed_codes |= {
            str(row.pdb_code or "").strip().upper()
            for row in permanent_error_query.all()
            if _is_permanent_optional_tm_error(
                predictor_name=normalized_name,
                error_message=getattr(row, "error_message", None),
            )
            and str(row.pdb_code or "").strip()
        }

    return completed_codes


def _is_permanent_optional_tm_error(predictor_name, error_message):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    message = str(error_message or "").strip().lower()
    if not message:
        return False

    generic_permanent_markers = (
        "no usable sequence could be fetched",
        "sequence unavailable",
        "unsupported completion provider scope",
        "not locally runnable",
    )
    if any(marker in message for marker in generic_permanent_markers):
        return True

    if normalized_name == "TMDET":
        tmdet_permanent_markers = (
            "protein is too large",
            "number of residues:",
            "exiting.",
        )
        if all(marker in message for marker in tmdet_permanent_markers):
            return True

    return False


def _fallback_completion_statuses(retry_errors=False):
    return ("success",) if retry_errors else ("success", "error")


def _describe_optional_tm_runtime_error(predictor_name, row, fallback_message):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    message = str(fallback_message or "").strip()
    sequence = str((row or {}).get("sequence_sequence") or "").strip()
    sequence_length = len(sequence)

    if not message:
        if normalized_name == "TMbed" and sequence_length:
            return (
                f"TMbed finished without emitting a usable prediction for {sequence_length}-residue "
                "input on this runtime."
            )
        return "Local MetaMP fallback predictor finished without emitting a usable prediction."

    lowered = message.lower()
    if normalized_name == "TMbed":
        if "returncode=-9" in lowered or "killed" in lowered:
            return (
                f"TMbed embedding/prediction was killed while processing a {sequence_length}-residue "
                f"sequence (likely memory pressure on this runtime). Original error: {message}"
            )
        if "pathological full-length topology" in lowered:
            return f"TMbed produced a pathological full-length topology and the result was rejected. {message}"

    return message


def _persist_optional_tm_error_rows(
    predictor_name,
    error_rows,
    provider="MetaMP",
    progress_callback=None,
):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    cleaned_rows = []
    for row in error_rows or ():
        pdb_code = str((row or {}).get("pdb_code") or "").strip().upper()
        error_message = str((row or {}).get("error_message") or "").strip()
        if not pdb_code or not error_message:
            continue
        cleaned_rows.append({"pdb_code": pdb_code, "error_message": error_message})
    if not cleaned_rows:
        return {
            "predictor": normalized_name,
            "processed_records": 0,
            "stored_rows": 0,
            "inserted_rows": 0,
            "updated_rows": 0,
        }

    deduped_rows = {
        row["pdb_code"]: row
        for row in cleaned_rows
    }
    cleaned_rows = list(deduped_rows.values())
    pdb_codes = [row["pdb_code"] for row in cleaned_rows]
    uniprot_rows = (
        db.session.query(Uniprot.pdb_code, Uniprot.uniprot_id)
        .filter(
            Uniprot.pdb_code.isnot(None),
            Uniprot.uniprot_id.isnot(None),
            db.func.upper(db.func.trim(Uniprot.pdb_code)).in_(pdb_codes),
        )
        .all()
    )
    uniprot_mapping = {}
    for row in uniprot_rows:
        pdb_code = str(row.pdb_code or "").strip().upper()
        uniprot_id = str(row.uniprot_id or "").strip().upper()
        if not pdb_code or not uniprot_id:
            continue
        uniprot_mapping.setdefault(pdb_code, []).append(uniprot_id)

    prediction_kind = _prediction_kind_for_optional_method(normalized_name)
    payloads = []
    for row in cleaned_rows:
        pdb_code = row["pdb_code"]
        error_message = row["error_message"]
        uniprot_ids = list(dict.fromkeys(uniprot_mapping.get(pdb_code) or [f"PDB:{pdb_code}"]))
        raw_payload = json.dumps(
            {
                "provider": provider,
                "method": normalized_name,
                "pdb_code": pdb_code,
                "status": "error",
                "error_message": error_message,
                "uniprot_ids": uniprot_ids,
            }
        )
        for uniprot_id in uniprot_ids:
            payloads.append(
                TMAlphaFoldPredictionResult(
                    pdb_code=pdb_code,
                    uniprot_id=uniprot_id,
                    provider=provider,
                    method=normalized_name,
                    prediction_kind=prediction_kind,
                    tm_count=None,
                    tm_regions_json="[]",
                    raw_payload_json=raw_payload,
                    source_url="",
                    status="error",
                    sequence_sequence=None,
                    error_message=error_message,
                )
            )

    from src.Jobs.TMAlphaFoldSync import _ensure_tmalphafold_storage, _upsert_predictions

    _ensure_tmalphafold_storage(progress_callback=progress_callback)
    stored = _upsert_predictions(payloads)
    summary = {
        "predictor": normalized_name,
        "processed_records": int(len(cleaned_rows)),
        "stored_rows": int(stored.get("stored_rows") or 0),
        "inserted_rows": int(stored.get("inserted_rows") or 0),
        "updated_rows": int(stored.get("updated_rows") or 0),
    }
    _emit_progress(
        progress_callback,
        f"Recorded {summary['processed_records']} {normalized_name} error record(s) in normalized storage.",
    )
    return summary


def import_optional_tm_prediction_failures(
    predictor_name,
    failures_path=None,
    provider="MetaMP",
    progress_callback=None,
):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    input_file = Path(failures_path) if failures_path else get_optional_tm_prediction_paths(normalized_name)["failures_path"]
    if not input_file.exists():
        return {
            "predictor": normalized_name,
            "input_path": str(input_file),
            "processed_records": 0,
            "stored_rows": 0,
            "inserted_rows": 0,
            "updated_rows": 0,
            "skipped": True,
            "skip_reason": "missing_input_file",
        }

    dataframe = pd.read_csv(input_file)
    if dataframe.empty:
        return {
            "predictor": normalized_name,
            "input_path": str(input_file),
            "processed_records": 0,
            "stored_rows": 0,
            "inserted_rows": 0,
            "updated_rows": 0,
        }

    lowercase_map = {str(column).strip().lower(): column for column in dataframe.columns}
    pdb_column = lowercase_map.get("pdb_code") or lowercase_map.get("pdb")
    error_column = lowercase_map.get("error_message") or lowercase_map.get("error") or lowercase_map.get("message")
    if pdb_column is None or error_column is None:
        raise ValueError("Failure input must include 'pdb_code' and 'error_message' columns.")

    failure_df = pd.DataFrame()
    failure_df["pdb_code"] = dataframe[pdb_column].astype(str).str.strip().str.upper()
    failure_df["error_message"] = dataframe[error_column].fillna("").astype(str).str.strip()
    failure_df = failure_df.loc[
        failure_df["pdb_code"].ne("") & failure_df["error_message"].ne("")
    ].drop_duplicates(subset="pdb_code", keep="last")
    if failure_df.empty:
        return {
            "predictor": normalized_name,
            "input_path": str(input_file),
            "processed_records": 0,
            "stored_rows": 0,
            "inserted_rows": 0,
            "updated_rows": 0,
        }

    summary = _persist_optional_tm_error_rows(
        predictor_name=normalized_name,
        error_rows=failure_df.to_dict(orient="records"),
        provider=provider,
        progress_callback=progress_callback,
    )
    summary["input_path"] = str(input_file)
    return summary


def run_optional_tm_local_command(
    predictor_name,
    include_completed=False,
    retry_errors=False,
    pdb_codes=None,
    limit=None,
    completion_provider="MetaMP",
    batch_size=None,
    progress_callback=None,
):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    runtime = _resolve_optional_tm_command_runtime(normalized_name)
    command_spec = runtime.get("effective_local_command")
    if not command_spec:
        raise RuntimeError(
            f"No local command is configured for optional predictor '{normalized_name}'. "
            "Set OPTIONAL_TM_LOCAL_COMMANDS_JSON to enable local execution."
        )
    if not runtime.get("locally_runnable"):
        verification = runtime.get("runtime_verification") or {}
        detail = str(verification.get("detail") or "").strip()
        if not detail:
            detail = (
                f"MetaMP discovered a command for {normalized_name}, but it did not pass runtime validation."
            )
        raise RuntimeError(detail)

    export_summary = export_optional_tm_prediction_inputs(
        predictor_name=normalized_name,
        include_completed=include_completed,
        retry_errors=retry_errors,
        pdb_codes=pdb_codes,
        limit=limit,
        completion_provider=completion_provider,
        progress_callback=progress_callback,
    )
    missing_sequence_summary = None
    unrunnable_codes = list(
        dict.fromkeys(export_summary.get("unrunnable_missing_sequence_codes") or [])
    )
    execution_mode = get_optional_tm_predictor_spec(normalized_name).get("execution_mode")
    if unrunnable_codes and execution_mode != "external_structure" and not include_completed:
        missing_sequence_summary = _persist_optional_tm_error_rows(
            predictor_name=normalized_name,
            error_rows=[
                {
                    "pdb_code": pdb_code,
                    "error_message": "No usable sequence could be fetched from RCSB for local MetaMP fallback.",
                }
                for pdb_code in unrunnable_codes
            ],
            provider="MetaMP",
            progress_callback=progress_callback,
        )
    if int(export_summary.get("record_count") or 0) == 0:
        export_summary["command_skipped"] = True
        export_summary["skip_reason"] = "no_pending_records"
        return {
            "predictor": normalized_name,
            "command_configured": True,
            "export": export_summary,
            "missing_sequence_errors": missing_sequence_summary,
            "processed_records": int(missing_sequence_summary.get("processed_records") or 0)
            if missing_sequence_summary
            else 0,
            "message": "No pending records required local command execution.",
        }

    path_config = get_optional_tm_prediction_paths(normalized_name)
    batch_size = _get_optional_tm_local_command_batch_size(
        normalized_name,
        batch_size_override=batch_size,
    )
    execution_mode = get_optional_tm_predictor_spec(normalized_name).get("execution_mode")
    command_summary = None
    failure_summary = None
    import_summary = None
    processed_records = 0
    if batch_size and int(export_summary.get("record_count") or 0) > batch_size:
        chunks = _prepare_optional_tm_command_chunks(
            normalized_name,
            export_summary,
            chunk_size=batch_size,
        )
        _emit_progress(
            progress_callback,
            f"Running {normalized_name} in {len(chunks)} chunk(s) of up to {batch_size} record(s) with incremental imports and resumable persistence.",
        )
        command_summaries = []
        failure_summaries = []
        import_summaries = []
        for chunk in chunks:
            _emit_progress(
                progress_callback,
                f"Starting {normalized_name} chunk {chunk['index']}/{chunk['total_chunks']} with {chunk['record_count']} record(s).",
            )
            chunk_command_summary, chunk_failure_summary, chunk_import_summary = _execute_optional_tm_local_command_once(
                predictor_name=normalized_name,
                command_spec=command_spec,
                input_paths=chunk,
                progress_callback=progress_callback,
            )
            command_summaries.append(chunk_command_summary)
            failure_summaries.append(chunk_failure_summary)
            import_summaries.append(chunk_import_summary)
            _append_csv_rows(
                path_config["results_path"],
                chunk["results_path"],
                ["pdb_code", "tm_count", "tm_regions"],
            )
            _append_csv_rows(
                path_config["failures_path"],
                chunk["failures_path"],
                ["pdb_code", "error_message"],
            )
            chunk_processed = int(chunk_import_summary.get("processed_records") or 0) + int(
                chunk_failure_summary.get("processed_records") or 0
            )
            processed_records += chunk_processed
            _emit_progress(
                progress_callback,
                f"Persisted {normalized_name} chunk {chunk['index']}/{chunk['total_chunks']} to normalized storage ({chunk_processed} record(s) processed this chunk, {processed_records} total).",
            )
            _emit_progress(
                progress_callback,
                (
                    f"Database save confirmed for {normalized_name} chunk "
                    f"{chunk['index']}/{chunk['total_chunks']}. Continuing to the next chunk."
                ),
            )
        command_summary = {
            "predictor": normalized_name,
            "chunked": True,
            "batch_size": int(batch_size),
            "incremental_persistence": True,
            "chunk_count": int(len(command_summaries)),
            "chunks": command_summaries,
            "return_code": 0,
            "failures_path": str(path_config["failures_path"]),
        }
        failure_summary = {
            "predictor": normalized_name,
            "chunked": True,
            "batch_size": int(batch_size),
            "chunk_count": int(len(failure_summaries)),
            "processed_records": int(
                sum(int(item.get("processed_records") or 0) for item in failure_summaries)
            ),
            "stored_rows": int(
                sum(int(item.get("stored_rows") or 0) for item in failure_summaries)
            ),
            "inserted_rows": int(
                sum(int(item.get("inserted_rows") or 0) for item in failure_summaries)
            ),
            "updated_rows": int(
                sum(int(item.get("updated_rows") or 0) for item in failure_summaries)
            ),
            "chunks": failure_summaries,
        }
        import_summary = {
            "predictor": normalized_name,
            "chunked": True,
            "batch_size": int(batch_size),
            "chunk_count": int(len(import_summaries)),
            "processed_records": int(
                sum(int(item.get("processed_records") or 0) for item in import_summaries)
            ),
            "chunks": import_summaries,
            "input_path": str(path_config["results_path"]),
            "import_manifest_path": str(path_config["import_manifest_path"]),
        }
    else:
        command_summary, failure_summary, import_summary = _execute_optional_tm_local_command_once(
            predictor_name=normalized_name,
            command_spec=command_spec,
            input_paths=path_config,
            progress_callback=progress_callback,
        )
        processed_records = int(import_summary.get("processed_records") or 0) + int(
            failure_summary.get("processed_records") or 0
        )
        command_summary["batch_size"] = int(batch_size) if batch_size else None
        command_summary["incremental_persistence"] = False
    return {
        "predictor": normalized_name,
        "command_configured": True,
        "export": export_summary,
        "missing_sequence_errors": missing_sequence_summary,
        "command": command_summary,
        "failures": failure_summary,
        "import": import_summary,
        "processed_records": (
            int(processed_records)
            + (
                int(missing_sequence_summary.get("processed_records") or 0)
                if missing_sequence_summary
                else 0
            )
        ),
    }


def run_optional_tm_prediction_backfill(
    predictor_names=None,
    include_completed=False,
    retry_errors=False,
    pdb_codes=None,
    limit=None,
    use_gpu=None,
    batch_size=None,
    max_workers=None,
    progress_callback=None,
):
    normalized_names = normalize_optional_tm_predictor_names(predictor_names)
    runtime_map = {
        predictor_name: _resolve_optional_tm_command_runtime(predictor_name)
        for predictor_name in normalized_names
    }
    builtin_local_predictors = [
        predictor_name
        for predictor_name in normalized_names
        if bool(runtime_map[predictor_name].get("builtin_local"))
    ]
    command_local_predictors = [
        predictor_name
        for predictor_name in normalized_names
        if predictor_name not in builtin_local_predictors and bool(runtime_map[predictor_name].get("locally_runnable"))
    ]
    unsupported_predictors = [
        predictor_name
        for predictor_name in normalized_names
        if predictor_name not in builtin_local_predictors and predictor_name not in command_local_predictors
    ]

    if unsupported_predictors:
        for predictor_name in unsupported_predictors:
            spec = get_optional_tm_predictor_spec(predictor_name)
            _emit_progress(
                progress_callback,
                f"{predictor_name} is not locally runnable in this MetaMP runtime "
                f"(execution_mode={spec.get('execution_mode')}). Use its export/import workflow or upstream sync path.",
            )

    aggregate = {
        "predictors": normalized_names,
        "builtin_local_predictors": builtin_local_predictors,
        "command_local_predictors": command_local_predictors,
        "unsupported_predictors": unsupported_predictors,
        "processed_records": 0,
        "per_predictor": {},
    }

    if builtin_local_predictors:
        builtin_summary = run_tm_prediction_backfill(
            include_tmbed="TMbed" in builtin_local_predictors,
            include_deeptmhmm="DeepTMHMM" in builtin_local_predictors,
            use_gpu=use_gpu,
            batch_size=batch_size,
            max_workers=max_workers,
            include_completed=include_completed,
            retry_errors=retry_errors,
            pdb_codes=pdb_codes,
            limit=limit,
            progress_callback=progress_callback,
        )
        aggregate["builtin_summary"] = builtin_summary
        aggregate["processed_records"] += int(builtin_summary.get("processed_records") or 0)
        for predictor_name in builtin_summary.get("predictors") or []:
            aggregate["per_predictor"][predictor_name] = {
                "mode": "builtin_local",
                "summary": builtin_summary,
            }

    for predictor_name in command_local_predictors:
        command_summary = run_optional_tm_local_command(
            predictor_name=predictor_name,
            include_completed=include_completed,
            retry_errors=retry_errors,
            pdb_codes=pdb_codes,
            limit=limit,
            completion_provider="MetaMP",
            batch_size=batch_size,
            progress_callback=progress_callback,
        )
        aggregate["per_predictor"][predictor_name] = {
            "mode": "configured_local_command",
            "summary": command_summary,
        }
        aggregate["processed_records"] += int(command_summary.get("processed_records") or 0)

    if not builtin_local_predictors and not command_local_predictors:
        aggregate["message"] = "No selected optional predictors are locally runnable in this MetaMP runtime."

    return aggregate


def determine_tmalphafold_fallback_targets(
    methods=None,
    with_tmdet=True,
    pdb_codes=None,
    limit=None,
    progress_callback=None,
):
    from src.Jobs.TMAlphaFoldSync import load_tmalphafold_targets

    selected_methods = [
        str(method or "").strip()
        for method in (methods or TMALPHAFOLD_SEQUENCE_METHODS + TMALPHAFOLD_AUX_METHODS)
        if str(method or "").strip()
    ]
    expected_methods = list(
        dict.fromkeys(selected_methods + (["TMDET"] if with_tmdet else []))
    )
    targets = load_tmalphafold_targets(pdb_codes=pdb_codes, limit=limit)
    if not targets:
        summary = {
            "target_count": 0,
            "fallback_target_count": 0,
            "expected_methods": expected_methods,
            "pdb_codes": [],
            "missing_by_pdb": {},
        }
        _emit_progress(
            progress_callback,
            "No UniProt-backed targets were eligible for TMAlphaFold fallback evaluation.",
        )
        return summary

    normalized_codes = list(
        dict.fromkeys(str(item["pdb_code"]).strip().upper() for item in targets if str(item.get("pdb_code") or "").strip())
    )
    rows = (
        TMAlphaFoldPrediction.query.with_entities(
            TMAlphaFoldPrediction.pdb_code,
            TMAlphaFoldPrediction.method,
            TMAlphaFoldPrediction.status,
        )
        .filter(
            TMAlphaFoldPrediction.provider == "TMAlphaFold",
            TMAlphaFoldPrediction.method.in_(expected_methods),
            db.func.upper(db.func.trim(TMAlphaFoldPrediction.pdb_code)).in_(normalized_codes),
        )
        .all()
    )
    successful_pairs = {
        (
            str(row.pdb_code or "").strip().upper(),
            str(row.method or "").strip(),
        )
        for row in rows
        if str(row.status or "").strip().lower() == "success"
    }

    fallback_codes = []
    missing_by_pdb = {}
    for pdb_code in normalized_codes:
        missing_methods = [
            method
            for method in expected_methods
            if (pdb_code, method) not in successful_pairs
        ]
        if missing_methods:
            fallback_codes.append(pdb_code)
            missing_by_pdb[pdb_code] = missing_methods

    summary = {
        "target_count": int(len(normalized_codes)),
        "fallback_target_count": int(len(fallback_codes)),
        "expected_methods": expected_methods,
        "pdb_codes": fallback_codes,
        "missing_by_pdb": missing_by_pdb,
    }
    _emit_progress(
        progress_callback,
        f"TMAlphaFold fallback evaluation identified {len(fallback_codes)} target(s) still missing at least one upstream method.",
    )
    return summary


def determine_verified_tm_fallback_targets(
    predictor_names=None,
    pdb_codes=None,
    limit=None,
    completion_provider="MetaMP",
    retry_errors=False,
    progress_callback=None,
):
    normalized_predictors = normalize_verified_tm_fallback_predictor_names(predictor_names)
    target_codes = load_verified_tm_fallback_target_codes(
        pdb_codes=pdb_codes,
        limit=limit,
    )
    if not target_codes:
        return {
            "target_count": 0,
            "fallback_target_count": 0,
            "expected_methods": normalized_predictors,
            "pdb_codes": [],
            "missing_by_pdb": {},
            "missing_counts_by_method": {predictor: 0 for predictor in normalized_predictors},
            "completed_counts_by_method": {predictor: 0 for predictor in normalized_predictors},
        }

    completed_by_method = {}
    for predictor_name in normalized_predictors:
        completed_by_method[predictor_name] = _load_optional_tm_completion_codes(
            predictor_name,
            provider=completion_provider,
            pdb_codes=target_codes,
            statuses=_fallback_completion_statuses(retry_errors=retry_errors),
        )

    missing_by_pdb = {}
    missing_counts_by_method = {predictor: 0 for predictor in normalized_predictors}
    completed_counts_by_method = {
        predictor: int(len(completed_by_method.get(predictor) or set()))
        for predictor in normalized_predictors
    }

    fallback_codes = []
    for pdb_code in target_codes:
        missing_methods = [
            predictor_name
            for predictor_name in normalized_predictors
            if pdb_code not in (completed_by_method.get(predictor_name) or set())
        ]
        if missing_methods:
            fallback_codes.append(pdb_code)
            missing_by_pdb[pdb_code] = missing_methods
            for predictor_name in missing_methods:
                missing_counts_by_method[predictor_name] += 1

    summary = {
        "target_count": int(len(target_codes)),
        "fallback_target_count": int(len(fallback_codes)),
        "expected_methods": normalized_predictors,
        "pdb_codes": fallback_codes,
        "missing_by_pdb": missing_by_pdb,
        "missing_counts_by_method": missing_counts_by_method,
        "completed_counts_by_method": completed_counts_by_method,
    }
    _emit_progress(
        progress_callback,
        "Verified MetaMP fallback comparison found "
        + str(len(fallback_codes))
        + " target(s) still missing at least one selected fallback method.",
    )
    return summary


def load_verified_tm_fallback_target_codes(
    pdb_codes=None,
    limit=None,
):
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
    all_data = canonicalize_pdb_frame(all_data, pdb_column="pdb_code")
    if "pdb_code" in all_data.columns:
        all_data = all_data.drop_duplicates(subset="pdb_code", keep="first").copy()

    selected_codes = _normalize_pdb_code_selection(pdb_codes)
    if selected_codes:
        all_data = all_data.loc[
            all_data["pdb_code"].astype(str).str.strip().str.upper().isin(selected_codes)
        ].copy()
    if limit is not None:
        all_data = all_data.head(int(limit)).copy()

    target_codes = [
        str(value).strip().upper()
        for value in all_data.get("pdb_code", pd.Series(dtype=str)).dropna().tolist()
        if str(value).strip()
    ]
    return list(dict.fromkeys(target_codes))


def run_verified_tm_fallback_pipeline(
    fallback_mode="tmalphafold_first",
    fallback_predictor_names=None,
    bulk_safe_only=False,
    pdb_codes=None,
    limit=None,
    include_completed=False,
    retry_errors=False,
    use_gpu=None,
    batch_size=None,
    max_workers=None,
    tmalphafold_methods=None,
    with_tmdet=True,
    tmalphafold_max_workers=8,
    tmalphafold_timeout=30,
    tmalphafold_refresh=False,
    tmalphafold_retry_errors=False,
    tmalphafold_backfill_sequences=True,
    progress_callback=None,
):
    selected_codes = _normalize_pdb_code_selection(pdb_codes)
    bulk_scope = not selected_codes and limit is None
    if fallback_predictor_names is None:
        requested_fallback_predictors = (
            BULK_SAFE_TM_FALLBACK_PREDICTORS
            if bulk_scope and bulk_safe_only
            else VERIFIED_LOCAL_TM_FALLBACK_PREDICTORS
        )
    else:
        requested_fallback_predictors = fallback_predictor_names
    normalized_fallback_predictors = normalize_verified_tm_fallback_predictor_names(
        requested_fallback_predictors
    )
    execution_predictors = order_verified_tm_fallback_predictor_names(
        normalized_fallback_predictors
    )
    method_list = [
        str(method or "").strip()
        for method in (
            tmalphafold_methods or (TMALPHAFOLD_SEQUENCE_METHODS + TMALPHAFOLD_AUX_METHODS)
        )
        if str(method or "").strip()
    ]

    summary = {
        "mode": str(fallback_mode or "").strip(),
        "fallback_predictors": normalized_fallback_predictors,
        "execution_predictors": execution_predictors,
        "bulk_safe_defaults_applied": bool(
            fallback_predictor_names is None and bulk_scope and bulk_safe_only
        ),
        "all_verified_methods_applied": bool(
            fallback_predictor_names is None and (not bulk_scope or not bulk_safe_only)
        ),
        "omitted_verified_predictors": [
            predictor_name
            for predictor_name in VERIFIED_LOCAL_TM_FALLBACK_PREDICTORS
            if predictor_name not in normalized_fallback_predictors
        ],
        "selected_pdb_codes": selected_codes,
        "limit": limit,
        "retry_errors": bool(retry_errors),
        "tmalphafold": None,
        "fallback_scope": None,
        "omitted_fallback_scope": None,
        "fallback": None,
    }

    if summary["bulk_safe_defaults_applied"]:
        _emit_progress(
            progress_callback,
            "Bulk fallback scope detected; using bulk-safe default predictors: "
            + ", ".join(normalized_fallback_predictors)
            + ". Use --fallback-method to include TMbed or TMDET explicitly for targeted reruns or smaller limited batches.",
        )
    elif fallback_predictor_names is None and bulk_scope:
        _emit_progress(
            progress_callback,
            "Bulk fallback scope detected; running the full verified local fallback set in staged order: "
            + ", ".join(execution_predictors)
            + ". Use --bulk-safe-only to restrict bulk runs to "
            + ", ".join(BULK_SAFE_TM_FALLBACK_PREDICTORS)
            + ".",
        )

    if fallback_mode == "tmalphafold_first":
        from src.Jobs.TMAlphaFoldSync import sync_tmalphafold_predictions

        summary["tmalphafold"] = sync_tmalphafold_predictions(
            methods=method_list,
            with_tmdet=with_tmdet,
            pdb_codes=selected_codes,
            limit=limit,
            refresh=tmalphafold_refresh,
            retry_errors=tmalphafold_retry_errors,
            max_workers=tmalphafold_max_workers,
            timeout=tmalphafold_timeout,
            backfill_sequences=tmalphafold_backfill_sequences,
            progress_callback=progress_callback,
        )
        summary["fallback_scope"] = determine_tmalphafold_fallback_targets(
            methods=method_list,
            with_tmdet=with_tmdet,
            pdb_codes=selected_codes,
            limit=limit,
            progress_callback=progress_callback,
        )
        fallback_codes = summary["fallback_scope"]["pdb_codes"]
        if not fallback_codes:
            summary["fallback"] = {
                "predictors": normalized_fallback_predictors,
                "processed_records": 0,
                "message": "TMAlphaFold already has successful upstream coverage for the selected scope. No local fallback run was needed.",
            }
            return summary
        fallback_limit = None
    elif fallback_mode == "fallback_only":
        selected_target_codes = load_verified_tm_fallback_target_codes(
            pdb_codes=selected_codes,
            limit=limit,
        )
        summary["fallback_scope"] = determine_verified_tm_fallback_targets(
            predictor_names=normalized_fallback_predictors,
            pdb_codes=selected_codes,
            limit=limit,
            completion_provider="MetaMP",
            retry_errors=retry_errors,
            progress_callback=progress_callback,
        )
        if summary["omitted_verified_predictors"]:
            summary["omitted_fallback_scope"] = determine_verified_tm_fallback_targets(
                predictor_names=summary["omitted_verified_predictors"],
                pdb_codes=selected_codes,
                limit=limit,
                completion_provider="MetaMP",
                retry_errors=retry_errors,
                progress_callback=progress_callback,
            )
        fallback_codes = selected_target_codes if include_completed else summary["fallback_scope"]["pdb_codes"]
        fallback_limit = None
        if not fallback_codes:
            summary["fallback"] = {
                "predictors": normalized_fallback_predictors,
                "processed_records": 0,
                "message": "Selected verified fallback methods already have MetaMP coverage for the selected scope. No local fallback run was needed.",
            }
            return summary
    else:
        raise ValueError(
            "Unsupported fallback mode '"
            + str(fallback_mode)
            + "'. Expected one of: tmalphafold_first, fallback_only."
        )

    aggregate_fallback_summary = {
        "predictors": normalized_fallback_predictors,
        "execution_predictors": execution_predictors,
        "processed_records": 0,
        "per_predictor": {},
    }
    unsupported_missing_predictors = []
    target_selection_codes = fallback_codes or selected_codes
    for predictor_name in execution_predictors:
        predictor_scope = determine_verified_tm_fallback_targets(
            predictor_names=[predictor_name],
            pdb_codes=target_selection_codes,
            limit=fallback_limit,
            completion_provider="MetaMP",
            retry_errors=retry_errors,
            progress_callback=progress_callback,
        )
        predictor_codes = target_selection_codes if include_completed else predictor_scope["pdb_codes"]
        aggregate_fallback_summary["per_predictor"][predictor_name] = {
            "scope": predictor_scope,
        }
        if not predictor_codes:
            aggregate_fallback_summary["per_predictor"][predictor_name]["summary"] = {
                "predictor": predictor_name,
                "processed_records": 0,
                "message": (
                    f"{predictor_name} already has MetaMP coverage for the selected scope. "
                    "No local fallback run was needed."
                ),
            }
            continue
        _emit_progress(
            progress_callback,
            f"Starting verified local fallback predictor {predictor_name} for {len(predictor_codes)} protein(s).",
        )
        predictor_summary = run_optional_tm_prediction_backfill(
            predictor_names=[predictor_name],
            include_completed=include_completed,
            retry_errors=retry_errors,
            pdb_codes=predictor_codes,
            limit=None,
            use_gpu=use_gpu,
            batch_size=batch_size,
            max_workers=max_workers,
            progress_callback=progress_callback,
        )
        aggregate_fallback_summary["per_predictor"][predictor_name]["summary"] = predictor_summary
        aggregate_fallback_summary["processed_records"] += int(
            predictor_summary.get("processed_records") or 0
        )
        if (
            predictor_scope.get("pdb_codes")
            and predictor_name in (predictor_summary.get("unsupported_predictors") or [])
        ):
            unsupported_missing_predictors.append(
                {
                    "predictor": predictor_name,
                    "missing_count": len(predictor_scope.get("pdb_codes") or []),
                }
            )

    if aggregate_fallback_summary["processed_records"] == 0:
        if unsupported_missing_predictors:
            aggregate_fallback_summary["unsupported_missing_predictors"] = unsupported_missing_predictors
            unsupported_descriptions = ", ".join(
                f"{item['predictor']} ({item['missing_count']} missing)"
                for item in unsupported_missing_predictors
            )
            aggregate_fallback_summary["message"] = (
                "Some selected verified fallback methods are still missing MetaMP coverage but are not "
                "locally runnable in this MetaMP runtime: "
                + unsupported_descriptions
                + "."
            )
        else:
            aggregate_fallback_summary["message"] = (
                "Selected verified fallback methods already have MetaMP coverage for the selected scope. "
                "No local fallback run was needed."
            )
    summary["fallback"] = aggregate_fallback_summary
    return summary


def get_optional_tm_runtime_status(predictor_names=None, app_config=None):
    normalized_names = normalize_optional_tm_predictor_names(predictor_names)
    tool_paths = get_optional_tm_tool_paths(app_config=app_config)
    per_predictor = {}
    for predictor_name in normalized_names:
        per_predictor[predictor_name] = _resolve_optional_tm_command_runtime(
            predictor_name,
            app_config=app_config,
        )
    return {
        "tool_home": str(tool_paths["tool_home"]),
        "bin_dir": str(tool_paths["bin_dir"]),
        "wrappers_dir": str(tool_paths["wrappers_dir"]),
        "predictors": normalized_names,
        "per_predictor": per_predictor,
    }


def load_optional_tm_prediction_frame(
    predictor_name,
    include_completed=False,
    retry_errors=False,
    pdb_codes=None,
    limit=None,
    completion_provider="MetaMP",
    progress_callback=None,
):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    execution_mode = get_optional_tm_predictor_spec(normalized_name).get("execution_mode")
    structure_only = execution_mode == "external_structure"

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
    all_data = canonicalize_pdb_frame(all_data, pdb_column="pdb_code")
    normalized_prediction_df = _load_normalized_tm_prediction_frame(
        predictor_names=[normalized_name],
        provider=normalize_optional_tm_completion_provider(completion_provider),
        pdb_codes=pdb_codes,
    )
    normalized_prediction_df = canonicalize_pdb_frame(normalized_prediction_df, pdb_column="pdb_code")
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
        completed_codes = _load_optional_tm_completion_codes(
            normalized_name,
            provider=completion_provider,
            pdb_codes=all_data["pdb_code"].tolist(),
            statuses=_fallback_completion_statuses(retry_errors=retry_errors),
        )
        pending_mask = all_data[[count_col, region_col]].fillna("").astype(str).apply(
            lambda column: column.str.strip().eq("")
        ).all(axis=1)
        if completed_codes:
            pending_mask &= ~all_data["pdb_code"].astype(str).str.strip().str.upper().isin(completed_codes)
        all_data = all_data.loc[pending_mask].copy()
        _emit_progress(
            progress_callback,
            f"Found {len(all_data)} protein record(s) still missing {normalized_name} results "
            f"for completion provider scope {completion_provider}.",
        )

    unrunnable_codes = []
    if not structure_only:
        sequence_missing_before = (
            all_data["sequence_sequence"].isna()
            | all_data["sequence_sequence"].astype(str).str.strip().eq("")
        )
        pending_before_sequence = int(len(all_data))

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

        unresolved_mask = (
            all_data["sequence_sequence"].isna()
            | all_data["sequence_sequence"].astype(str).str.strip().eq("")
        )
        unrunnable_codes = (
            all_data.loc[unresolved_mask, "pdb_code"]
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )
        all_data = all_data.loc[~unresolved_mask].copy()
        unrunnable_due_to_missing_sequence = max(0, pending_before_sequence - int(len(all_data)))
        if unrunnable_due_to_missing_sequence:
            _emit_progress(
                progress_callback,
                f"Skipped {unrunnable_due_to_missing_sequence} {normalized_name} candidate record(s) because no usable sequence could be fetched from RCSB.",
            )
    else:
        all_data["sequence_sequence"] = all_data["sequence_sequence"].fillna("").astype(str)
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
    result = all_data.reset_index(drop=True)
    result.attrs["unrunnable_missing_sequence_codes"] = list(dict.fromkeys(unrunnable_codes))
    return result


def load_optional_tm_prediction_export_frame(
    predictor_names=None,
    pdb_codes=None,
    limit=None,
    completion_provider="MetaMP",
    progress_callback=None,
):
    normalized_names = normalize_optional_tm_predictor_names(predictor_names)
    execution_modes = {
        predictor_name: get_optional_tm_predictor_spec(predictor_name).get("execution_mode")
        for predictor_name in normalized_names
    }
    structure_only = bool(normalized_names) and all(
        execution_modes.get(predictor_name) == "external_structure"
        for predictor_name in normalized_names
    )
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
        predictor_names=normalized_names,
        provider=normalize_optional_tm_completion_provider(completion_provider),
        pdb_codes=pdb_codes,
    )
    if not normalized_prediction_df.empty:
        all_data = all_data.merge(
            normalized_prediction_df,
            on="pdb_code",
            how="left",
            suffixes=("", "_normalized"),
        )

    required_cols = ["pdb_code", "sequence_sequence"]
    for normalized_name in normalized_names:
        count_col, region_col = tm_predictor_column_names(normalized_name)
        if count_col not in all_data.columns:
            all_data[count_col] = pd.NA
        if region_col not in all_data.columns:
            all_data[region_col] = ""
        required_cols.extend([count_col, region_col])

    if "sequence_sequence" not in all_data.columns:
        all_data["sequence_sequence"] = pd.NA

    all_data = all_data[required_cols].copy()
    selected_codes = _normalize_pdb_code_selection(pdb_codes)
    if selected_codes:
        all_data = all_data.loc[
            all_data["pdb_code"].astype(str).str.strip().str.upper().isin(selected_codes)
        ].copy()
        _emit_progress(
            progress_callback,
            f"Restricted optional TM export to {len(all_data)} explicitly selected protein record(s).",
        )

    _emit_progress(
        progress_callback,
        "Loaded "
        + str(len(all_data))
        + " merged protein record(s) for optional TM export across predictors: "
        + ", ".join(normalized_names)
        + f". Completion provider scope: {completion_provider}.",
    )

    unrunnable_codes = []
    if not structure_only:
        sequence_missing_before = (
            all_data["sequence_sequence"].isna()
            | all_data["sequence_sequence"].astype(str).str.strip().eq("")
        )
        pending_before_sequence = int(len(all_data))

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

        unresolved_mask = (
            all_data["sequence_sequence"].isna()
            | all_data["sequence_sequence"].astype(str).str.strip().eq("")
        )
        unrunnable_codes = (
            all_data.loc[unresolved_mask, "pdb_code"]
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )
        all_data = all_data.loc[~unresolved_mask].copy()
        unrunnable_due_to_missing_sequence = max(0, pending_before_sequence - int(len(all_data)))
        if unrunnable_due_to_missing_sequence:
            _emit_progress(
                progress_callback,
                f"Skipped {unrunnable_due_to_missing_sequence} optional TM export candidate record(s) because no usable sequence could be fetched from RCSB.",
            )
    else:
        all_data["sequence_sequence"] = all_data["sequence_sequence"].fillna("").astype(str)

    if limit is not None:
        all_data = all_data.head(int(limit)).copy()
        _emit_progress(
            progress_callback,
            f"Restricted optional TM export to the first {len(all_data)} protein record(s).",
        )

    _emit_progress(
        progress_callback,
        f"Prepared {len(all_data)} protein sequence(s) for multi-predictor optional TM export.",
    )
    result = all_data.reset_index(drop=True)
    result.attrs["unrunnable_missing_sequence_codes"] = list(dict.fromkeys(unrunnable_codes))
    return result


def export_optional_tm_prediction_inputs(
    predictor_name,
    fasta_out=None,
    csv_out=None,
    include_completed=False,
    retry_errors=False,
    pdb_codes=None,
    limit=None,
    completion_provider="MetaMP",
    progress_callback=None,
):
    normalized_name = normalize_optional_tm_predictor_name(predictor_name)
    path_config = get_optional_tm_prediction_paths(normalized_name)
    frame = load_optional_tm_prediction_frame(
        predictor_name=normalized_name,
        include_completed=include_completed,
        retry_errors=retry_errors,
        pdb_codes=pdb_codes,
        limit=limit,
        completion_provider=completion_provider,
        progress_callback=progress_callback,
    )
    unrunnable_codes = list(dict.fromkeys(frame.attrs.get("unrunnable_missing_sequence_codes") or []))

    fasta_path = Path(fasta_out) if fasta_out else path_config["fasta_path"]
    csv_path = Path(csv_out) if csv_out else path_config["csv_template_path"]
    results_path = path_config["results_path"]
    failures_path = path_config["failures_path"]
    reference_path = path_config["reference_path"]
    structure_manifest_path = path_config["structure_manifest_path"]
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    structure_manifest_path.parent.mkdir(parents=True, exist_ok=True)

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
    template.to_csv(results_path, index=False)
    pd.DataFrame(columns=["pdb_code", "error_message"]).to_csv(failures_path, index=False)

    count_col, region_col = tm_predictor_column_names(normalized_name)
    if frame.empty:
        reference_frame = pd.DataFrame(columns=["pdb_code", "tm_count", "tm_regions"])
    else:
        upstream_seed_df = _load_normalized_tm_prediction_frame(
            predictor_names=[normalized_name],
            provider="TMAlphaFold",
            pdb_codes=frame["pdb_code"].tolist(),
        )
        if not upstream_seed_df.empty:
            reference_frame = upstream_seed_df.rename(
                columns={
                    count_col: "tm_count",
                    region_col: "tm_regions",
                }
            )[["pdb_code", "tm_count", "tm_regions"]]
        else:
            reference_frame = pd.DataFrame(columns=["pdb_code", "tm_count", "tm_regions"])
    reference_frame.to_csv(reference_path, index=False)
    structure_manifest = _build_optional_tm_structure_manifest(frame)
    structure_manifest.to_csv(structure_manifest_path, index=False)
    reference_row_count = int(len(reference_frame))
    debug_payload = _build_optional_tm_export_debug_payload(
        reference_frame=reference_frame,
        results_frame=template,
        predictor_name=normalized_name,
    )

    summary = {
        "predictor": normalized_name,
        "record_count": int(len(frame)),
        "unrunnable_missing_sequence_count": int(len(unrunnable_codes)),
        "unrunnable_missing_sequence_codes": unrunnable_codes,
        "fasta_path": str(fasta_path),
        "csv_template_path": str(csv_path),
        "results_input_path": str(results_path),
        "failures_input_path": str(failures_path),
        "reference_input_path": str(reference_path),
        "structure_manifest_path": str(structure_manifest_path),
        "seeded_from_upstream_count": reference_row_count,
        "reference_row_count": reference_row_count,
        "execution_mode": get_optional_tm_predictor_spec(normalized_name).get("execution_mode"),
        "include_completed": bool(include_completed),
        "completion_provider": completion_provider,
        "debug": debug_payload,
        "export_manifest_path": str(path_config["export_manifest_path"]),
    }
    write_optional_tm_prediction_manifest(path_config["export_manifest_path"], summary)
    _emit_progress(
        progress_callback,
        f"Exported {summary['record_count']} {normalized_name} input sequence(s) and CSV template.",
    )
    _emit_optional_tm_export_debug(
        progress_callback=progress_callback,
        predictor_name=normalized_name,
        debug_payload=debug_payload,
        results_path=results_path,
        reference_path=reference_path,
    )
    return summary


def export_optional_tm_prediction_inputs_bulk(
    predictor_names=None,
    include_completed=False,
    retry_errors=False,
    pdb_codes=None,
    limit=None,
    completion_provider="MetaMP",
    progress_callback=None,
):
    normalized_names = normalize_optional_tm_predictor_names(predictor_names)
    frame = load_optional_tm_prediction_export_frame(
        predictor_names=normalized_names,
        pdb_codes=pdb_codes,
        limit=limit,
        completion_provider=completion_provider,
        progress_callback=progress_callback,
    )
    shared_unrunnable_codes = list(
        dict.fromkeys(frame.attrs.get("unrunnable_missing_sequence_codes") or [])
    )

    per_predictor = {}
    total_export_records = 0
    predictors_with_pending = 0

    for normalized_name in normalized_names:
        count_col, region_col = tm_predictor_column_names(normalized_name)
        predictor_frame = frame.copy()
        if not include_completed:
            completed_codes = _load_optional_tm_completion_codes(
                normalized_name,
                provider=completion_provider,
                pdb_codes=predictor_frame["pdb_code"].tolist(),
                statuses=_fallback_completion_statuses(retry_errors=retry_errors),
            )
            pending_mask = predictor_frame[[count_col, region_col]].fillna("").astype(str).apply(
                lambda column: column.str.strip().eq("")
            ).all(axis=1)
            if completed_codes:
                pending_mask &= ~predictor_frame["pdb_code"].astype(str).str.strip().str.upper().isin(completed_codes)
            predictor_frame = predictor_frame.loc[pending_mask].copy()
            _emit_progress(
                progress_callback,
                f"Found {len(predictor_frame)} protein record(s) still missing {normalized_name} results "
                f"for completion provider scope {completion_provider}.",
            )
        else:
            _emit_progress(
                progress_callback,
                f"Including completed records for {normalized_name}; exporting {len(predictor_frame)} record(s).",
            )

        path_config = get_optional_tm_prediction_paths(normalized_name)
        fasta_path = path_config["fasta_path"]
        csv_path = path_config["csv_template_path"]
        results_path = path_config["results_path"]
        failures_path = path_config["failures_path"]
        reference_path = path_config["reference_path"]
        structure_manifest_path = path_config["structure_manifest_path"]
        fasta_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        structure_manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with fasta_path.open("w") as fh:
            for _, row in predictor_frame.iterrows():
                fh.write(f">{row['pdb_code']}\n{row['sequence_sequence']}\n")

        template = pd.DataFrame(
            {
                "pdb_code": predictor_frame["pdb_code"].astype(str),
                "tm_count": [None] * len(predictor_frame),
                "tm_regions": [""] * len(predictor_frame),
            }
        )
        template.to_csv(csv_path, index=False)
        template.to_csv(results_path, index=False)
        pd.DataFrame(columns=["pdb_code", "error_message"]).to_csv(failures_path, index=False)

        if predictor_frame.empty:
            reference_frame = pd.DataFrame(columns=["pdb_code", "tm_count", "tm_regions"])
        else:
            upstream_seed_df = _load_normalized_tm_prediction_frame(
                predictor_names=[normalized_name],
                provider="TMAlphaFold",
                pdb_codes=predictor_frame["pdb_code"].tolist(),
            )
            if not upstream_seed_df.empty:
                reference_frame = upstream_seed_df.rename(
                    columns={
                        count_col: "tm_count",
                        region_col: "tm_regions",
                    }
                )[["pdb_code", "tm_count", "tm_regions"]]
            else:
                reference_frame = pd.DataFrame(columns=["pdb_code", "tm_count", "tm_regions"])
        reference_frame.to_csv(reference_path, index=False)
        structure_manifest = _build_optional_tm_structure_manifest(predictor_frame)
        structure_manifest.to_csv(structure_manifest_path, index=False)
        reference_row_count = int(len(reference_frame))
        debug_payload = _build_optional_tm_export_debug_payload(
            reference_frame=reference_frame,
            results_frame=template,
            predictor_name=normalized_name,
        )

        summary = {
            "predictor": normalized_name,
            "record_count": int(len(predictor_frame)),
            "unrunnable_missing_sequence_count": int(len(shared_unrunnable_codes)),
            "unrunnable_missing_sequence_codes": shared_unrunnable_codes,
            "fasta_path": str(fasta_path),
            "csv_template_path": str(csv_path),
            "results_input_path": str(results_path),
            "failures_input_path": str(failures_path),
            "reference_input_path": str(reference_path),
            "structure_manifest_path": str(structure_manifest_path),
            "seeded_from_upstream_count": reference_row_count,
            "reference_row_count": reference_row_count,
            "execution_mode": get_optional_tm_predictor_spec(normalized_name).get("execution_mode"),
            "include_completed": bool(include_completed),
            "completion_provider": completion_provider,
            "debug": debug_payload,
            "export_manifest_path": str(path_config["export_manifest_path"]),
        }
        write_optional_tm_prediction_manifest(path_config["export_manifest_path"], summary)
        _emit_progress(
            progress_callback,
            f"Exported {summary['record_count']} {normalized_name} input sequence(s) and CSV template.",
        )
        _emit_optional_tm_export_debug(
            progress_callback=progress_callback,
            predictor_name=normalized_name,
            debug_payload=debug_payload,
            results_path=results_path,
            reference_path=reference_path,
        )
        per_predictor[normalized_name] = summary
        total_export_records += int(len(predictor_frame))
        if len(predictor_frame) > 0:
            predictors_with_pending += 1

    aggregate_summary = {
        "predictors": normalized_names,
        "selected_record_count": int(len(frame)),
        "predictors_with_pending_records": int(predictors_with_pending),
        "total_export_records": int(total_export_records),
        "include_completed": bool(include_completed),
        "completion_provider": completion_provider,
        "per_predictor": per_predictor,
    }
    _emit_progress(
        progress_callback,
        "Completed optional TM export across "
        + str(len(normalized_names))
        + " predictor(s); "
        + str(total_export_records)
        + " total record export(s) written.",
    )
    return aggregate_summary


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


def _resolve_requested_builtin_predictors(
    include_tmbed,
    include_deeptmhmm,
    use_gpu,
    progress_callback=None,
):
    resolved_include_tmbed = bool(include_tmbed)
    if resolved_include_tmbed:
        status = get_tmbed_runtime_status(use_gpu=bool(use_gpu))
        if not status.get("available"):
            _emit_progress(
                progress_callback,
                "Skipping TMbed because the runtime is not healthy: "
                + str(status.get("reason") or "unknown reason")
                + ".",
            )
            resolved_include_tmbed = False
    return resolved_include_tmbed, include_deeptmhmm


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
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND column_name = %s
              AND table_name IN ('membrane_proteins', 'membrane_protein_uniprot')
            """,
            (seq_col,),
        )
        available_tables = {row[0] for row in cur.fetchall()}

        updated_counts = {}
        if "membrane_proteins" in available_tables:
            protein_query = sql.SQL(
                """
                UPDATE {tbl}
                SET {seq_col} = %s
                WHERE {id_col} = %s
                  AND ({seq_col} IS NULL OR BTRIM({seq_col}) = '')
                """
            ).format(
                tbl=sql.Identifier("membrane_proteins"),
                seq_col=sql.Identifier(seq_col),
                id_col=sql.Identifier(pdb_col),
            )
            cur.executemany(protein_query.as_string(conn), updates)
            updated_counts["membrane_proteins"] = max(0, int(cur.rowcount or 0))

        if "membrane_protein_uniprot" in available_tables:
            uniprot_query = sql.SQL(
                """
                UPDATE {tbl}
                SET {seq_col} = %s
                WHERE {id_col} = %s
                  AND ({seq_col} IS NULL OR BTRIM({seq_col}) = '')
                """
            ).format(
                tbl=sql.Identifier("membrane_protein_uniprot"),
                seq_col=sql.Identifier(seq_col),
                id_col=sql.Identifier(pdb_col),
            )
            cur.executemany(uniprot_query.as_string(conn), updates)
            updated_counts["membrane_protein_uniprot"] = max(0, int(cur.rowcount or 0))

        conn.commit()
    finally:
        conn.close()

    _emit_progress(
        progress_callback,
        "Persisted fetched protein sequences into "
        + ", ".join(
            f"{table_name}={count}"
            for table_name, count in sorted(updated_counts.items())
        )
        if updated_counts
        else "Skipped sequence persistence because no target table exposes the requested sequence column.",
    )
    return int(sum(updated_counts.values()))


def load_pending_tm_prediction_frame(
    resume_from_csv=None,
    include_tmbed=True,
    include_deeptmhmm=True,
    include_completed=False,
    retry_errors=False,
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
    all_data = canonicalize_pdb_frame(all_data, pdb_column="pdb_code")
    normalized_prediction_df = _load_normalized_tm_prediction_frame(
        predictor_names=(
            (["TMbed"] if include_tmbed else [])
            + (["DeepTMHMM"] if include_deeptmhmm else [])
        ),
        provider="MetaMP",
        pdb_codes=pdb_codes,
    )
    normalized_prediction_df = canonicalize_pdb_frame(normalized_prediction_df, pdb_column="pdb_code")
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
        predictor_pending_masks = []
        selected_predictors = (
            (["TMbed"] if include_tmbed else [])
            + (["DeepTMHMM"] if include_deeptmhmm else [])
        )
        normalized_pdb_codes = (
            all_data["pdb_code"].astype(str).str.strip().str.upper()
            if "pdb_code" in all_data.columns
            else pd.Series(dtype=str)
        )
        for predictor_name in selected_predictors:
            count_col, region_col = tm_predictor_column_names(predictor_name)
            available_predictor_columns = [
                column for column in [count_col, region_col] if column in all_data.columns
            ]
            if not available_predictor_columns:
                continue
            predictor_missing_mask = all_data[available_predictor_columns].fillna("").astype(str).apply(
                lambda column: column.str.strip().eq("")
            ).all(axis=1)
            completed_codes = _load_optional_tm_completion_codes(
                predictor_name,
                provider="MetaMP",
                pdb_codes=all_data["pdb_code"].tolist(),
                statuses=_fallback_completion_statuses(retry_errors=retry_errors),
            )
            if completed_codes:
                predictor_missing_mask &= ~normalized_pdb_codes.isin(completed_codes)
            predictor_pending_masks.append(predictor_missing_mask)

        if predictor_pending_masks:
            pending_mask = predictor_pending_masks[0].copy()
            for predictor_mask in predictor_pending_masks[1:]:
                pending_mask |= predictor_mask
            all_data = all_data.loc[pending_mask]
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
    pending_before_sequence = int(len(all_data))

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

    unrunnable_codes = []
    if not all_data.empty and "sequence_sequence" in all_data.columns:
        unresolved_mask = sequence_missing_before.reindex(all_data.index, fill_value=False) & (
            all_data["sequence_sequence"].isna()
            | all_data["sequence_sequence"].astype(str).str.strip().eq("")
        )
        if unresolved_mask.any():
            unrunnable_codes = list(
                dict.fromkeys(
                    all_data.loc[unresolved_mask, "pdb_code"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .tolist()
                )
            )

    all_data = all_data[all_data["sequence_sequence"].notna()]
    all_data = all_data[all_data["sequence_sequence"] != ""]
    unrunnable_due_to_missing_sequence = max(0, pending_before_sequence - int(len(all_data)))
    if unrunnable_due_to_missing_sequence:
        _emit_progress(
            progress_callback,
            f"Skipped {unrunnable_due_to_missing_sequence} TM prediction candidate record(s) because no usable sequence could be fetched from RCSB.",
        )
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
    all_data.attrs["unrunnable_missing_sequence_codes"] = unrunnable_codes
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
    include_deeptmhmm = (
        runtime_options["include_deeptmhmm"]
        if include_deeptmhmm is None
        else include_deeptmhmm
    )
    use_gpu = runtime_options["use_gpu"] if use_gpu is None else use_gpu
    max_workers = (
        runtime_options["max_workers"] if max_workers is None else max_workers
    )
    include_tmbed, include_deeptmhmm = _resolve_requested_builtin_predictors(
        include_tmbed=include_tmbed,
        include_deeptmhmm=include_deeptmhmm,
        use_gpu=use_gpu,
        progress_callback=progress_callback,
    )
    if not include_tmbed and include_deeptmhmm is False:
        return {
            "predictors": [],
            "processed_records": 0,
            "records": [],
            "message": "No runnable built-in TM predictors are available on this runtime.",
        }

    analyzer = MultiModelAnalyzer(
        db_params={},
        table="unused",
        batch_size=max(1, len(dataframe)),
        max_workers=max_workers,
        max_sequences=4,
        use_db=False,
        write_csv=False,
    )
    tmbed_kwargs = {"format_code": 0, "use_gpu": use_gpu, "batch_size": None, "threads": None}
    if len(dataframe) <= 1:
        tmbed_kwargs.update({"batch_size": 1, "threads": 1})
    if include_tmbed:
        analyzer.register(TMbedPredictor(**tmbed_kwargs))
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
    retry_errors=False,
    pdb_codes=None,
    limit=None,
    progress_callback=None,
):
    runtime_options = get_tm_prediction_runtime_options()
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
    include_tmbed, include_deeptmhmm = _resolve_requested_builtin_predictors(
        include_tmbed=include_tmbed,
        include_deeptmhmm=include_deeptmhmm,
        use_gpu=use_gpu,
        progress_callback=progress_callback,
    )
    if not include_tmbed and include_deeptmhmm is False:
        summary = {
            "queued_records": 0,
            "processed_records": 0,
            "predictors": [],
            "message": "No runnable built-in TM predictors are available on this runtime.",
            "include_completed": bool(include_completed),
        }
        if csv_out:
            summary["csv_path"] = str(csv_out)
        _emit_progress(progress_callback, summary["message"])
        return summary

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
        retry_errors=retry_errors,
        pdb_codes=pdb_codes,
        limit=limit,
        progress_callback=progress_callback,
    )
    unrunnable_codes = list(
        dict.fromkeys(all_data.attrs.get("unrunnable_missing_sequence_codes") or [])
    )
    missing_sequence_error_summaries = {}
    if unrunnable_codes and not include_completed:
        builtin_predictors = (
            (["TMbed"] if include_tmbed else [])
            + (["DeepTMHMM"] if include_deeptmhmm else [])
        )
        for predictor_name in builtin_predictors:
            missing_sequence_error_summaries[predictor_name] = _persist_optional_tm_error_rows(
                predictor_name=predictor_name,
                error_rows=[
                    {
                        "pdb_code": pdb_code,
                        "error_message": "No usable sequence could be fetched from RCSB for local MetaMP fallback.",
                    }
                    for pdb_code in unrunnable_codes
                ],
                provider="MetaMP",
                progress_callback=progress_callback,
            )
        _emit_progress(
            progress_callback,
            "Recorded sequence-unavailable MetaMP fallback error rows for built-in predictors: "
            + ", ".join(builtin_predictors)
            + ".",
        )
    if all_data.empty:
        no_pending_message = "No pending protein records require TM prediction backfill."
        if missing_sequence_error_summaries:
            no_pending_message = (
                "No runnable protein records require TM prediction backfill after sequence resolution."
            )
        summary = {
            "queued_records": 0,
            "processed_records": int(len(unrunnable_codes)) if missing_sequence_error_summaries else 0,
            "predictors": (
                (["TMbed"] if include_tmbed else [])
                + (["DeepTMHMM"] if include_deeptmhmm else [])
            ),
            "message": no_pending_message,
            "include_completed": bool(include_completed),
            "missing_sequence_errors": missing_sequence_error_summaries,
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
    if include_completed:
        forced_predictor_names = [
            predictor_name
            for predictor_name in (["TMbed"] if include_tmbed else []) + (["DeepTMHMM"] if include_deeptmhmm else [])
        ]
        for predictor_name in forced_predictor_names:
            count_col, region_col = tm_predictor_column_names(predictor_name)
            if count_col in all_data.columns:
                all_data[count_col] = pd.NA
            if region_col in all_data.columns:
                all_data[region_col] = pd.NA
    tmbed_kwargs = {"format_code": 0, "use_gpu": use_gpu, "batch_size": None, "threads": None}
    if len(all_data) <= 1:
        tmbed_kwargs.update({"batch_size": 1, "threads": 1})
    if include_tmbed:
        analyzer.register(TMbedPredictor(**tmbed_kwargs))
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
    runtime_error_summaries = {}

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
            try:
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
            except Exception as exc:
                _emit_progress(
                    progress_callback,
                    f"Normalized-store persistence failed for {predictor_name} batch {start}-{end}: {exc}",
                )

            count_col, region_col = tm_predictor_column_names(predictor_name)
            error_col = f"{predictor_name}_error_message"
            unresolved_codes = []
            unresolved_error_rows = []
            for _, row in batch_df.iterrows():
                pdb_code = str(row.get("pdb_code") or "").strip().upper()
                if not pdb_code:
                    continue
                count_value = row.get(count_col) if count_col in batch_df.columns else None
                region_value = row.get(region_col) if region_col in batch_df.columns else None
                error_message = (
                    str(row.get(error_col) or "").strip()
                    if error_col in batch_df.columns
                    else ""
                )
                count_missing = pd.isna(count_value)
                region_missing = False
                if region_col in batch_df.columns:
                    if pd.isna(region_value):
                        region_missing = True
                    else:
                        region_missing = str(region_value).strip() == ""
                if count_missing and region_missing:
                    unresolved_codes.append(pdb_code)
                    unresolved_error_rows.append(
                        {
                            "pdb_code": pdb_code,
                            "error_message": _describe_optional_tm_runtime_error(
                                predictor_name,
                                row,
                                error_message
                                or "Local MetaMP fallback predictor finished without emitting a usable prediction.",
                            ),
                        }
                    )
            unresolved_codes = list(dict.fromkeys(unresolved_codes))
            if unresolved_codes and not include_completed:
                try:
                    error_summary = _persist_optional_tm_error_rows(
                        predictor_name=predictor_name,
                        error_rows=unresolved_error_rows,
                        provider="MetaMP",
                        progress_callback=progress_callback,
                    )
                    aggregate = runtime_error_summaries.setdefault(
                        predictor_name,
                        {
                            "predictor": predictor_name,
                            "processed_records": 0,
                            "stored_rows": 0,
                            "inserted_rows": 0,
                            "updated_rows": 0,
                        },
                    )
                    for key in ("processed_records", "stored_rows", "inserted_rows", "updated_rows"):
                        aggregate[key] += int(error_summary.get(key) or 0)
                except Exception as exc:
                    _emit_progress(
                        progress_callback,
                        f"Runtime-error persistence failed for {predictor_name} batch {start}-{end}: {exc}",
                    )
        if start is not None and end is not None and total is not None:
            _emit_progress(
                progress_callback,
                f"Persisted TMbed/normalized predictor results for batch {start}-{end} of {total}.",
            )
            _emit_progress(
                progress_callback,
                (
                    f"Database save confirmed for TMbed/DeepTMHMM batch {start}-{end} of {total}. "
                    "Continuing to the next batch."
                ),
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
        "processed_records": int(processed_this_run) + (
            int(len(unrunnable_codes)) if missing_sequence_error_summaries else 0
        ),
        "predictors": predictor_names,
        "record_columns": [
            column for column in TM_PREDICTION_RECORD_COLUMNS if column in result_df.columns
        ],
        "include_completed": bool(include_completed),
        "normalized_store": normalized_store,
        "missing_sequence_errors": missing_sequence_error_summaries,
        "runtime_error_rows": runtime_error_summaries,
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
