import argparse
import csv
import json
import re
import shutil
from pathlib import Path


CURRENT_DATASET_FILES = [
    "datasets/expert_annotation_predicted.csv",
    "datasets/Mpstruct_dataset.csv",
    "datasets/PDB_data.csv",
    "datasets/PDB_data_transformed.csv",
    "datasets/Quantitative_data.csv",
    "datasets/NEWOPM.csv",
    "datasets/Uniprot_functions.csv",
    "datasets/valid/Mpstruct_dataset.csv",
    "datasets/valid/PDB_data_transformed.csv",
    "datasets/valid/Quantitative_data.csv",
    "datasets/valid/NEWOPM.csv",
    "datasets/valid/Uniprot_functions.csv",
]

MODEL_SUPPORT_FILES = [
    "data/models/live_group_predictions.csv",
    "data/models/time_split_evaluation_metrics.json",
    "data/models/pca.png",
    "data/models/pca.pdf",
    "data/models/tsne.png",
    "data/models/tsne.pdf",
    "data/models/umap.png",
    "data/models/umap.pdf",
    "data/models/metrics_comparison_altair.png",
    "data/models/metrics_comparison_altair.pdf",
    "data/tm_predictions/tm_summary.csv",
]

PRODUCTION_SMALL_DIRS = [
    "data/models/production_ml/figures",
    "data/models/production_ml/specs",
    "data/models/production_ml/tables",
]

LEGACY_SEMI_SUPERVISED_DIR = "data/models/semi-supervised"

LEGACY_SEMI_SUPERVISED_REQUIRED_FILES = {
    "without_reduction_data.csv",
    "PCA_data.csv",
    "TSNE_data.csv",
    "UMAP_data.csv",
}

LEGACY_REVIEWER_ARTIFACT_SUFFIXES = {
    ".csv",
    ".json",
    ".png",
    ".pdf",
    ".svg",
    ".txt",
    ".tsv",
}

LEGACY_MODEL_NAME_PREFERENCE = [
    "Random Forest",
    "Gradient Boosting",
    "Logistic Regression",
    "SVM",
    "K-Nearest Neighbors",
    "Decision Tree",
]


def _copy_file_if_exists(source_root: Path, relative_path: str, output_root: Path) -> None:
    source_path = _resolve_source_path(source_root, relative_path)
    if source_path is None or not source_path.exists() or not source_path.is_file():
        return
    destination_path = output_root / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def _copy_tree_if_exists(source_root: Path, relative_path: str, output_root: Path) -> None:
    source_path = _resolve_source_path(source_root, relative_path)
    if source_path is None or not source_path.exists() or not source_path.is_dir():
        return
    destination_path = output_root / relative_path
    if destination_path.exists():
        shutil.rmtree(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_path, destination_path)


def _copy_latest_country_dataset(source_root: Path, output_root: Path):
    candidate_roots = [
        source_root / "data" / "datasets",
        source_root / "datasets",
        source_root / "src" / "datasets",
    ]
    latest_path = None
    latest_date = None

    for dataset_root in candidate_roots:
        if not dataset_root.exists() or not dataset_root.is_dir():
            continue
        for candidate in dataset_root.glob("country_data_*.csv"):
            match = re.search(r"country_data_(\d{4}-\d{2}-\d{2})\.csv$", candidate.name)
            if not match:
                continue
            try:
                candidate_date = match.group(1)
            except ValueError:
                continue
            if latest_date is None or candidate_date > latest_date:
                latest_date = candidate_date
                latest_path = candidate

    if latest_path is None:
        return None

    destination_path = output_root / "datasets" / latest_path.name
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest_path, destination_path)
    return str(destination_path.relative_to(output_root))


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _resolve_source_path(source_root: Path, relative_path: str):
    candidates = [source_root / relative_path]

    if relative_path.startswith("datasets/"):
        candidates.extend(
            [
                source_root / "data" / relative_path,
                source_root / "src" / relative_path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _load_registry_rows(registry_path: Path):
    if not registry_path.exists():
        return []
    with registry_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _select_artifact_ids(registry_rows, explainability_manifest: dict, top_models: int):
    ranked_rows = sorted(
        registry_rows,
        key=lambda row: (
            0 if _truthy(row.get("selected_for_upload")) else 1,
            -_to_float(row.get("expert_f1_weighted")),
            -_to_float(row.get("expert_accuracy")),
            -_to_float(row.get("cv_mean_f1")),
            str(row.get("artifact_id") or ""),
        ),
    )
    selected_ids = [
        str(row.get("artifact_id") or "").strip()
        for row in ranked_rows
        if str(row.get("artifact_id") or "").strip()
    ][: max(int(top_models or 5), 1)]

    shap_bundle = (
        (explainability_manifest or {}).get("selected_tree_bundle") or {}
    ).get("artifact_id")
    if shap_bundle and shap_bundle not in selected_ids:
        selected_ids.append(shap_bundle)

    return selected_ids


def _copy_selected_model_artifacts(
    source_root: Path,
    output_root: Path,
    registry_rows,
    selected_ids,
):
    models_dir = output_root / "data/models/production_ml/models"
    models_dir.mkdir(parents=True, exist_ok=True)
    rows_by_id = {
        str(row.get("artifact_id") or "").strip(): row
        for row in registry_rows
        if str(row.get("artifact_id") or "").strip()
    }
    copied_ids = []

    for artifact_id in selected_ids:
        row = rows_by_id.get(artifact_id)
        if row is None:
            continue
        preferred_path = source_root / f"data/models/production_ml/models/{artifact_id}.joblib"
        artifact_path = preferred_path
        if not artifact_path.exists():
            raw_artifact_path = str(row.get("artifact_path") or "").strip()
            if raw_artifact_path:
                candidate = Path(raw_artifact_path)
                if not candidate.is_absolute():
                    candidate = source_root / candidate
                artifact_path = candidate
        if not artifact_path.exists():
            continue

        shutil.copy2(artifact_path, models_dir / f"{artifact_id}.joblib")
        copied_ids.append(artifact_id)

    return copied_ids


def _write_filtered_registry(output_root: Path, registry_rows, selected_ids):
    output_path = output_root / "data/models/production_ml/tables/model_bundle_registry.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not registry_rows:
        output_path.write_text("")
        return []

    selected_lookup = set(selected_ids)
    filtered_rows = [
        row for row in registry_rows if str(row.get("artifact_id") or "").strip() in selected_lookup
    ]
    if not filtered_rows:
        return []

    fieldnames = list(filtered_rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    return filtered_rows


def _rewrite_manifest(output_root: Path, manifest: dict, selected_ids):
    output_path = output_root / "data/models/production_ml/specs/manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest:
        output_path.write_text("{}\n")
        return

    selected_lookup = set(selected_ids)
    filtered_manifest = dict(manifest)
    filtered_manifest["available_artifacts"] = [
        artifact
        for artifact in manifest.get("available_artifacts", [])
        if str((artifact or {}).get("artifact_id") or "").strip() in selected_lookup
    ]
    selected_upload_id = str(manifest.get("selected_upload_artifact_id") or "").strip()
    if selected_upload_id and selected_upload_id not in selected_lookup:
        filtered_manifest["selected_upload_artifact_id"] = selected_ids[0] if selected_ids else None
    output_path.write_text(json.dumps(filtered_manifest, indent=2))


def _legacy_model_sort_key(path: Path):
    stem = path.stem
    reduction_match = re.search(r"__semi_supervised_([A-Za-z0-9\-]+)_\d+$", stem)
    reduction = str(reduction_match.group(1)).lower() if reduction_match else "unknown"
    model_name = stem.split("__semi_supervised_", 1)[0]
    try:
        preference_index = LEGACY_MODEL_NAME_PREFERENCE.index(model_name)
    except ValueError:
        preference_index = len(LEGACY_MODEL_NAME_PREFERENCE)
    return (reduction, preference_index, model_name.lower(), path.name.lower())


def _select_legacy_joblibs(source_dir: Path):
    grouped = {}
    for path in sorted(source_dir.glob("*.joblib"), key=_legacy_model_sort_key):
        match = re.search(r"__semi_supervised_([A-Za-z0-9\-]+)_\d+$", path.stem)
        if not match:
            continue
        reduction = str(match.group(1)).lower()
        grouped.setdefault(reduction, path)
    return list(grouped.values())


def _copy_legacy_semi_supervised_artifacts(source_root: Path, output_root: Path):
    source_dir = _resolve_source_path(source_root, LEGACY_SEMI_SUPERVISED_DIR)
    if source_dir is None or not source_dir.exists() or not source_dir.is_dir():
        return {"reviewer_artifacts": [], "retained_model_files": []}

    destination_dir = output_root / LEGACY_SEMI_SUPERVISED_DIR
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied_reviewer_artifacts = []
    for path in sorted(source_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file():
            continue
        if path.name in LEGACY_SEMI_SUPERVISED_REQUIRED_FILES or path.suffix.lower() in LEGACY_REVIEWER_ARTIFACT_SUFFIXES:
            shutil.copy2(path, destination_dir / path.name)
            copied_reviewer_artifacts.append(path.name)

    retained_model_files = []
    for joblib_path in _select_legacy_joblibs(source_dir):
        shutil.copy2(joblib_path, destination_dir / joblib_path.name)
        retained_model_files.append(joblib_path.name)

    return {
        "reviewer_artifacts": copied_reviewer_artifacts,
        "retained_model_files": retained_model_files,
    }


def export_runtime_snapshot(source_root: Path, output_dir: Path, top_models: int):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for relative_path in CURRENT_DATASET_FILES:
        _copy_file_if_exists(source_root, relative_path, output_dir)

    copied_country_dataset = _copy_latest_country_dataset(source_root, output_dir)

    for relative_path in MODEL_SUPPORT_FILES:
        _copy_file_if_exists(source_root, relative_path, output_dir)

    for relative_path in PRODUCTION_SMALL_DIRS:
        _copy_tree_if_exists(source_root, relative_path, output_dir)

    legacy_semi_supervised_summary = _copy_legacy_semi_supervised_artifacts(
        source_root,
        output_dir,
    )

    registry_path = source_root / "data/models/production_ml/tables/model_bundle_registry.csv"
    manifest_path = source_root / "data/models/production_ml/specs/manifest.json"
    explainability_manifest_path = source_root / "data/models/production_ml/specs/explainability_manifest.json"

    registry_rows = _load_registry_rows(registry_path)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    explainability_manifest = (
        json.loads(explainability_manifest_path.read_text())
        if explainability_manifest_path.exists()
        else {}
    )
    selected_ids = _select_artifact_ids(registry_rows, explainability_manifest, top_models)
    copied_ids = _copy_selected_model_artifacts(source_root, output_dir, registry_rows, selected_ids)
    filtered_rows = _write_filtered_registry(output_dir, registry_rows, copied_ids)
    _rewrite_manifest(output_dir, manifest, copied_ids)

    snapshot_manifest = {
        "top_model_limit": int(top_models),
        "retained_artifact_ids": copied_ids,
        "retained_artifact_count": len(copied_ids),
        "registry_row_count": len(filtered_rows),
        "legacy_semi_supervised_reviewer_artifact_count": len(
            legacy_semi_supervised_summary.get("reviewer_artifacts", [])
        ),
        "legacy_semi_supervised_retained_model_files": legacy_semi_supervised_summary.get(
            "retained_model_files",
            [],
        ),
        "includes_database_dump": False,
        "notes": [
            "Runtime-required dataset CSVs were exported from the active application dataset locations.",
            "The latest country reference dataset was retained when available so dashboard geography views render in snapshot/reviewer builds.",
            "Reviewer-facing ML figures, dimensionality-reduction artifacts, and metric tables were retained.",
            "Only a trimmed set of inference model binaries was retained to reduce snapshot size.",
            "Use the shell snapshot workflow to add the PostgreSQL dump and restore it on another machine.",
        ],
    }
    if copied_country_dataset:
        snapshot_manifest["country_reference_dataset"] = copied_country_dataset
    (output_dir / "snapshot_manifest.json").write_text(json.dumps(snapshot_manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build a trimmed MetaMP runtime snapshot payload.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export datasets and trimmed ML assets.")
    export_parser.add_argument("--source-root", type=Path, default=Path("/var/app"))
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--top-models", type=int, default=1)

    args = parser.parse_args()
    if args.command == "export":
        export_runtime_snapshot(
            source_root=args.source_root.resolve(),
            output_dir=args.output_dir.resolve(),
            top_models=args.top_models,
        )


if __name__ == "__main__":
    main()
