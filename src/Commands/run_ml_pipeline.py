import argparse
import json
import logging
from pathlib import Path

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

from app import app
from src.Jobs.MLJobs import MLJob


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the MetaMP supervised and semi-supervised group-classification "
            "pipeline with production artifact export."
        )
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Train models without exporting expert/discrepancy prediction artifacts.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    with app.app_context():
        job = (
            MLJob()
            .fix_missing_data()
            .variable_separation()
            .feature_selection()
            .dimensionality_reduction()
            .plot_charts()
            .semi_supervised_learning()
            .supervised_learning()
        )
        if not args.skip_benchmark:
            job.benchmark_and_export_predictions()
        job.time_split_evaluation()

        manifest_path = Path(job.production_paths["specs"]) / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            selected = manifest.get("selected_upload_artifact_id")
            print(f"ML pipeline completed successfully. Selected upload artifact: {selected}")
            print(f"Manifest: {manifest_path}")
        else:
            print("ML pipeline completed successfully.")


if __name__ == "__main__":
    main()
