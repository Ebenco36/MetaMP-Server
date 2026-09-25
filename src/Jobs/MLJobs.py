import json
import os, sys
import random
import warnings
from pathlib import Path
from datetime import datetime, timezone
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    cross_validate, 
    StratifiedKFold,
    KFold
)
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
try:
    import shap
except ImportError:  # pragma: no cover - optional dependency for explainability plots
    shap = None
from sklearn.metrics import (
    make_scorer, 
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
# from app import app
from flask import current_app, has_app_context
import altair as alt
from database.db import db
from src.Dashboard.group_standardization import (
    collapse_group_label_for_expert_benchmark,
    standardize_group_label,
)
from src.Dashboard.services import get_tables_as_dataframe, get_table_as_dataframe
from src.Jobs.Utils import (
    ClassifierComparison,
    ClassifierComparisonSemiSupervised,
    evaluate_dimensionality_reduction,
)
from src.Jobs.transformData import report_and_clean_missing_values

def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)


def get_model_root():
    configured_dir = os.getenv("SEMI_SUPERVISED_MODEL_DIR")
    if configured_dir:
        return Path(configured_dir).resolve().parent
    return Path("./models").resolve()


def get_semi_supervised_model_dir():
    configured_dir = os.getenv("SEMI_SUPERVISED_MODEL_DIR")
    if configured_dir:
        return Path(configured_dir).resolve()
    return get_model_root() / "semi-supervised"


MODEL_ROOT = get_model_root()
SEMI_SUPERVISED_MODEL_DIR = get_semi_supervised_model_dir()
PRODUCTION_ML_DIR = MODEL_ROOT / "production_ml"

ensure_dir_exists(MODEL_ROOT)
ensure_dir_exists(SEMI_SUPERVISED_MODEL_DIR)
ensure_dir_exists(PRODUCTION_ML_DIR)

FEATURE_DISPLAY_NAMES = {
    "topology_subunit": "Topology Subunit Count",
    "thickness": "Membrane Thickness",
    "subunit_segments": "Subunit TM Segments",
    "tilt": "Helix Tilt",
    "gibbs": "Gibbs Free Energy",
    "membrane_topology_in": "Membrane Topology In",
    "membrane_topology_out": "Membrane Topology Out",
}

DR_PLOTLY_PALETTE = {
    "Monotopic Membrane Proteins": "#01696f",
    "Transmembrane Proteins: Alpha-helical": "#d19900",
    "Transmembrane Proteins: Beta-barrel": "#a13544",
}

class MLJob:
    TRAINING_NUMERIC_FEATURES = [
        "thickness",
        "subunit_segments",
        "tilt",
        "gibbs",
    ]
    TRAINING_CATEGORICAL_FEATURES = [
        "topology_subunit",
        "membrane_topology_in",
        "membrane_topology_out",
    ]

    def __init__(self):
        self.num_runs = 1  # Define the number of runs for averaging metrics
        self.random_state = 42  # For reproducibility
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.models = {}  # Initialize self.models as an empty dictionary
        # Data containers
        self.data = pd.DataFrame()
        self.numerical_data = pd.DataFrame()
        self.categorical_data = pd.DataFrame()
        self.complete_numerical_data = pd.DataFrame()
        self.data_combined_PCA = pd.DataFrame()
        self.data_combined_tsne = pd.DataFrame()
        self.data_combined_UMAP = pd.DataFrame()
        self.semi_supervised_metrics = pd.DataFrame()
        self.supervised_metrics = pd.DataFrame()
        self.label_encoder = LabelEncoder()
        self.over_sampling_data_selected_feature_data = pd.DataFrame()
        self.record_metadata = pd.DataFrame()
        self.inference_data = pd.DataFrame()
        self.inference_record_metadata = pd.DataFrame()
        self.inference_numerical_data = pd.DataFrame()
        self.inference_categorical_data = pd.DataFrame()
        self.topology_label_maps = {}
        self.selected_feature_columns = []
        self.discrepancy_exclusion_codes = []
        self.expert_annotation_exclusion_codes = []
        self.legacy_benchmark_exclusion_codes = []
        self.model_bundle_registry = []
        self.all_model_results = []
        self.training_scope_report = {}
        self.production_paths = {
            "root": PRODUCTION_ML_DIR,
            "models": PRODUCTION_ML_DIR / "models",
            "tables": PRODUCTION_ML_DIR / "tables",
            "figures": PRODUCTION_ML_DIR / "figures",
            "predictions": PRODUCTION_ML_DIR / "predictions",
            "specs": PRODUCTION_ML_DIR / "specs",
        }
        for path in self.production_paths.values():
            ensure_dir_exists(path)
        # Load data after initializing exclusion-tracking attributes so
        # the saved training scope reflects the actual held-out sets.
        self.load_data()

    @staticmethod
    def _log_info(message):
        if has_app_context():
            current_app.logger.info(message)
        else:
            print(message)

    @staticmethod
    def _log_warning(message):
        if has_app_context():
            current_app.logger.warning(message)
        else:
            print(message)
        
    def load_data(self):
        table_names = ['membrane_proteins', 'membrane_protein_opm']
        with current_app.app_context():
            # Load data from tables
            self.all_data = get_tables_as_dataframe(table_names, "pdb_code")
            self.result_df_db = get_table_as_dataframe("membrane_proteins")
            self.result_df_opm = get_table_as_dataframe("membrane_protein_opm")
            result_df_uniprot = get_table_as_dataframe("membrane_protein_uniprot")
        
        self.result_df = pd.merge(right=self.all_data, left=result_df_uniprot, on="pdb_code")
        # Reserve some data for test specifically for discrepancies
        legacy_benchmark_codes = [
            "1PFO", "1B12", "1GOS", "1MT5", "1KN9", "1OJA", "1O5W", "1UUM", "1T7D", "2BXR",
            "1YGM", "2GMH", "2OLV", "2OQO", "2QCU", "2PRM", "2Z5X", "2VQG", "3HYW", "3I65",
            "3VMA", "3NSJ", "3ML3", "3PRW", "3P1L", "3Q7M", "2YH3", "3LIM", "3VMT", "3Q54",
            "2YMK", "2LOU", "4LXJ", "4HSC", "4CDB", "4P6J", "4TSY", "5B49", "5IMW", "5IMY",
            "5JYN", "5LY6", "6BFG", "6MLU", "6DLW", "6H03", "6NYF", "6MTI", "7OFM", "7LQ6",
            "7RSL", "8A1D", "7QAM"
        ]
        discrepancy_exclusion_codes = self._load_discrepancy_exclusion_codes()
        self.discrepancy_exclusion_codes = sorted(discrepancy_exclusion_codes)
        expert_annotation_codes = {
            str(code).strip().upper()
            for code in self._load_expert_annotation_frame()["pdb_code"].tolist()
            if str(code).strip()
        }
        self.expert_annotation_exclusion_codes = sorted(expert_annotation_codes)
        self.legacy_benchmark_exclusion_codes = sorted(
            {
                str(code).strip().upper()
                for code in legacy_benchmark_codes
                if str(code).strip()
            }
        )
        discrepancy_only_codes = sorted(
            set(discrepancy_exclusion_codes).difference(expert_annotation_codes)
        )
        if discrepancy_only_codes:
            self._log_info(
                f"Excluding {len(discrepancy_only_codes)} live discrepancy benchmark record(s) not already covered by the expert holdout."
            )
        elif discrepancy_exclusion_codes:
            self._log_info(
                f"Live discrepancy benchmark contains {len(discrepancy_exclusion_codes)} record(s), all already covered by the expert holdout."
            )
        if expert_annotation_codes:
            self._log_info(
                f"Excluding {len(expert_annotation_codes)} expert-annotation benchmark record(s) from ML training."
            )
        exclude_pdb_codes = sorted(
            {
                str(code).strip().upper()
                for code in (
                    self.legacy_benchmark_exclusion_codes
                    + list(discrepancy_exclusion_codes)
                    + list(expert_annotation_codes)
                )
                if str(code).strip()
            }
        )
        
        self.inference_all_data = self.all_data.copy()
        self.all_data = self.all_data[
            ~self.all_data['pdb_code'].astype(str).str.strip().str.upper().isin(exclude_pdb_codes)
        ]

    def _load_discrepancy_exclusion_codes(self):
        try:
            from src.Dashboard.services import DiscrepancyBenchmarkExportService

            benchmark_df = DiscrepancyBenchmarkExportService.build_export_dataframe(
                include_all=False
            )
            if benchmark_df.empty:
                return set()

            exclusion_codes = set()
            for column in ("pdb_code", "canonical_pdb_code"):
                if column not in benchmark_df.columns:
                    continue
                exclusion_codes.update(
                    {
                        str(value).strip().upper()
                        for value in benchmark_df[column].dropna().tolist()
                        if str(value).strip()
                    }
                )
            return exclusion_codes
        except Exception as exc:
            self._log_warning(f"Unable to load live discrepancy benchmark exclusions: {exc}")
            return set()

    @staticmethod
    def _safe_slug(value):
        safe = "".join(
            character.lower() if str(character).isalnum() else "_"
            for character in str(value)
        )
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_") or "model"

    @staticmethod
    def _resolve_expert_annotation_dataset_path():
        configured = os.getenv("DASHBOARD_ANNOTATION_DATASET_PATH") or os.getenv(
            "ANNOTATION_DATASET_PATH"
        )
        candidates = []
        if configured:
            candidates.append(Path(configured))
        candidates.extend(
            [
                Path("/var/app/data/datasets/expert_annotation_predicted.csv"),
                Path.cwd() / "datasets" / "expert_annotation_predicted.csv",
                Path("/var/app/datasets/expert_annotation_predicted.csv"),
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @classmethod
    def _load_expert_annotation_frame(cls):
        annotation_path = cls._resolve_expert_annotation_dataset_path()
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Expert annotation dataset not found at {annotation_path}"
            )
        expert_frame = pd.read_csv(annotation_path)
        expert_frame["pdb_code"] = (
            expert_frame["PDB Code"].fillna("").astype(str).str.strip().str.upper()
        )
        expert_frame["group_expert_standardized"] = expert_frame["Group (Expert)"].apply(
            standardize_group_label
        )
        expert_frame["group_expert_benchmark"] = expert_frame["Group (Expert)"].apply(
            collapse_group_label_for_expert_benchmark
        )
        return expert_frame

    def _save_training_variables(self, raw_data):
        training_codes = sorted(
            {
                str(value).strip().upper()
                for value in self.record_metadata.get("pdb_code", pd.Series(dtype=str)).tolist()
                if str(value).strip()
            }
        )
        discrepancy_codes = sorted(self.discrepancy_exclusion_codes)
        expert_codes = sorted(self.expert_annotation_exclusion_codes)
        legacy_codes = sorted(self.legacy_benchmark_exclusion_codes)
        overlap = sorted(set(training_codes).intersection(discrepancy_codes))
        if overlap:
            raise ValueError(
                "Discrepancy benchmark records leaked into the training set: "
                + ", ".join(overlap[:10])
            )
        expert_overlap = sorted(set(training_codes).intersection(expert_codes))
        if expert_overlap:
            raise ValueError(
                "Expert annotation benchmark records leaked into the training set: "
                + ", ".join(expert_overlap[:10])
            )
        reserved_benchmark_codes = sorted(
            set(discrepancy_codes).union(expert_codes).union(legacy_codes)
        )
        reserved_overlap = sorted(set(training_codes).intersection(reserved_benchmark_codes))

        self.training_scope_report = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "random_state": int(self.random_state),
            "cross_validation_strategy": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "training_row_count": int(len(raw_data.index)),
            "training_unique_pdb_codes": int(len(training_codes)),
            "target_label_source": "Standardized MPstruc broad-group labels from membrane_proteins.group",
            "target_label_classes": sorted(
                pd.Series(raw_data["group"]).dropna().astype(str).unique().tolist()
            ),
            "expert_benchmark_metric_definition": {
                "expert_accuracy": "Benchmark-aware accuracy on the 121 held-out expert rows after biological equivalence collapsing. Expert Bitopic counts as correct for predicted Bitopic, alpha-helical transmembrane, or beta-barrel transmembrane; Monotopic remains strictly monotopic.",
                "expert_precision_weighted": "Benchmark-aware weighted precision on the same held-out expert rows using the same biological equivalence collapsing.",
                "expert_recall_weighted": "Benchmark-aware weighted recall on the same held-out expert rows using the same biological equivalence collapsing.",
                "expert_f1_weighted": "Benchmark-aware weighted F1 on the same held-out expert rows using the same biological equivalence collapsing.",
                "expert_accuracy_exact": "Strict exact-match accuracy between standardized expert labels and predicted labels before benchmark collapsing.",
                "expert_precision_weighted_exact": "Strict weighted precision between standardized expert labels and predicted labels before benchmark collapsing.",
                "expert_recall_weighted_exact": "Strict weighted recall between standardized expert labels and predicted labels before benchmark collapsing.",
                "expert_f1_weighted_exact": "Strict weighted F1 between standardized expert labels and predicted labels before benchmark collapsing.",
            },
            "selected_feature_columns": list(self.selected_feature_columns),
            "training_numeric_features": list(self.TRAINING_NUMERIC_FEATURES),
            "training_categorical_features": list(self.TRAINING_CATEGORICAL_FEATURES),
            "topology_label_maps": self.topology_label_maps,
            "discrepancy_exclusion_count": int(len(discrepancy_codes)),
            "legacy_benchmark_exclusion_count": int(len(legacy_codes)),
            "expert_annotation_count": int(len(expert_codes)),
            "discrepancy_training_overlap_count": int(len(overlap)),
            "reserved_benchmark_exclusion_count": int(len(reserved_benchmark_codes)),
            "reserved_benchmark_training_overlap_count": int(len(reserved_overlap)),
            "expert_training_overlap_count": int(len(expert_overlap)),
            "discrepancy_exclusion_codes": discrepancy_codes,
            "legacy_benchmark_exclusion_codes": legacy_codes,
            "expert_annotation_codes": expert_codes,
            "expert_training_overlap_codes": expert_overlap,
            "reserved_benchmark_codes": reserved_benchmark_codes,
            "reserved_benchmark_training_overlap_codes": reserved_overlap,
            "confirmed_discrepancy_codes_excluded_from_training": len(overlap) == 0,
            "confirmed_expert_annotation_codes_excluded_from_training": len(expert_overlap) == 0,
            "confirmed_all_reserved_benchmark_codes_excluded_from_training": len(reserved_overlap) == 0,
        }

        raw_data.to_csv(
            self.production_paths["tables"] / "training_dataset_used.csv",
            index=False,
        )
        pd.DataFrame(
            {
                "pdb_code": discrepancy_codes,
                "reason": "discrepancy_benchmark_exclusion",
            }
        ).to_csv(
            self.production_paths["tables"] / "training_excluded_discrepancy_codes.csv",
            index=False,
        )
        pd.DataFrame(
            {
                "pdb_code": expert_codes,
                "reason": "expert_annotation_benchmark_exclusion",
            }
        ).to_csv(
            self.production_paths["tables"] / "expert_annotation_codes.csv",
            index=False,
        )
        pd.DataFrame(
            {
                "pdb_code": legacy_codes,
                "reason": "legacy_manual_benchmark_exclusion",
            }
        ).to_csv(
            self.production_paths["tables"] / "training_excluded_legacy_benchmark_codes.csv",
            index=False,
        )
        (self.production_paths["specs"] / "training_variables.json").write_text(
            json.dumps(self.training_scope_report, indent=2, default=str)
        )
    
    def fix_missing_data(self):
        protected_columns = (
            ["pdb_code", "bibliography_year", "group"]
            + self.TRAINING_NUMERIC_FEATURES
            + self.TRAINING_CATEGORICAL_FEATURES
        )
        self.data = report_and_clean_missing_values(
            self.all_data,
            threshold=30,
            protected_columns=protected_columns,
        )
        self.data.columns = pd.Index([str(column) for column in self.data.columns], dtype="object")
        self.inference_data = report_and_clean_missing_values(
            self.inference_all_data.copy(),
            threshold=30,
            protected_columns=protected_columns,
        )
        self.inference_data.columns = pd.Index(
            [str(column) for column in self.inference_data.columns],
            dtype="object",
        )
        if "pdb_code" in self.data.columns:
            self.data["pdb_code"] = (
                self.data["pdb_code"].fillna("").astype(str).str.strip().str.upper()
            )
            self.data = self.data.sort_values("pdb_code").reset_index(drop=True)
        if "pdb_code" in self.inference_data.columns:
            self.inference_data["pdb_code"] = (
                self.inference_data["pdb_code"].fillna("").astype(str).str.strip().str.upper()
            )
            self.inference_data = self.inference_data.sort_values("pdb_code").reset_index(drop=True)
        metadata_columns = [
            column
            for column in ["pdb_code", "bibliography_year"]
            if column in self.data.columns
        ]
        self.record_metadata = self.data[metadata_columns].copy() if metadata_columns else pd.DataFrame(index=self.data.index)
        inference_metadata_columns = [
            column
            for column in ["pdb_code", "bibliography_year"]
            if column in self.inference_data.columns
        ]
        self.inference_record_metadata = (
            self.inference_data[inference_metadata_columns].copy()
            if inference_metadata_columns
            else pd.DataFrame(index=self.inference_data.index)
        )
        columns_to_drop = [col for col in self.data.columns if '_citation_' in col or '_count_' in col or col.startswith('count_') or col.endswith('_count') or col.startswith('revision_') or col.endswith('_revision') or col.startswith('id_') or col.endswith('_id') or col == "id"]
        removable_columns = columns_to_drop + ['pdbid', 'name_y', 'name_x', 'tilterror', 'description', 'family_name', 'species_name', 'exptl_method', 'thicknesserror', 'citation_country', 'family_name_cache', "bibliography_year", 'is_master_protein', 'species_name_cache', 'membrane_name_cache', 'species_description', 'membrane_short_name', "processed_resolution", 'family_superfamily_name', 'famsupclasstype_type_name', 'exptl_crystal_grow_method', 'exptl_crystal_grow_method1', 'family_superfamily_classtype_name', 'rcsentinfo_nonpolymer_molecular_weight_maximum', 'rcsentinfo_nonpolymer_molecular_weight_minimum', "rcsentinfo_polymer_molecular_weight_minimum", "rcsentinfo_molecular_weight", "rcsentinfo_polymer_molecular_weight_maximum"]
        self.data.drop(removable_columns, axis=1, inplace=True, errors="ignore")
        self.inference_data.drop(removable_columns, axis=1, inplace=True, errors="ignore")
        if "group" in self.data.columns:
            self.data["group"] = self.data["group"].apply(standardize_group_label)
        if "group" in self.inference_data.columns:
            self.inference_data["group"] = self.inference_data["group"].apply(standardize_group_label)
        return self

    def variable_separation(self):
        data = self.data.copy().reset_index(drop=True)
        data.columns = pd.Index([str(column) for column in data.columns], dtype="object")
        for column in self.TRAINING_NUMERIC_FEATURES:
            if column not in data.columns:
                data[column] = np.nan
        for column in self.TRAINING_CATEGORICAL_FEATURES:
            if column not in data.columns:
                data[column] = ""
        if "group" not in data.columns:
            raise ValueError("Missing required training target column 'group'.")
        if not self.record_metadata.empty:
            self.record_metadata = self.record_metadata.reset_index(drop=True)
        self.numerical_data = data[self.TRAINING_NUMERIC_FEATURES].copy()
        self.categorical_data = data[self.TRAINING_CATEGORICAL_FEATURES + ["group"]].copy()
        self.numerical_data.columns = pd.Index([str(column) for column in self.numerical_data.columns], dtype="object")
        self.categorical_data.columns = pd.Index([str(column) for column in self.categorical_data.columns], dtype="object")
        inference_frame = self.inference_data.copy()
        inference_frame.columns = pd.Index(
            [str(column) for column in inference_frame.columns],
            dtype="object",
        )
        for column in self.TRAINING_NUMERIC_FEATURES:
            if column not in inference_frame.columns:
                inference_frame[column] = np.nan
        for column in self.TRAINING_CATEGORICAL_FEATURES:
            if column not in inference_frame.columns:
                inference_frame[column] = ""
        self.inference_numerical_data = inference_frame[self.TRAINING_NUMERIC_FEATURES].copy()
        self.inference_categorical_data = inference_frame[self.TRAINING_CATEGORICAL_FEATURES].copy()
        return self
    
    def feature_selection(self):
        self.numerical_data.reset_index(drop=True, inplace=True)
        self.categorical_data.reset_index(drop=True, inplace=True)
        base_matrix = self._build_feature_base_matrix(
            self.numerical_data,
            self.categorical_data,
            fit_topology=True,
        ).reset_index(drop=True)
        labels = self.categorical_data["group"].apply(standardize_group_label).reset_index(drop=True)
        valid_mask = labels.notna() & labels.astype(str).str.strip().ne("")
        self.complete_numerical_data = base_matrix.loc[valid_mask].reset_index(drop=True)
        self.selected_feature_columns = self.complete_numerical_data.columns.tolist()
        filtered_labels = labels.loc[valid_mask].reset_index(drop=True)
        self.over_sampling_data_selected_feature_data = pd.concat(
            [self.complete_numerical_data, filtered_labels.rename("group")],
            axis=1,
        )
        if not self.record_metadata.empty:
            self.record_metadata = self.record_metadata.loc[valid_mask].reset_index(drop=True)

        raw_data = pd.concat(
            [
                self.record_metadata.reset_index(drop=True),
                self.numerical_data.loc[valid_mask].reset_index(drop=True),
                self.categorical_data.loc[valid_mask, self.TRAINING_CATEGORICAL_FEATURES].reset_index(drop=True),
                filtered_labels.rename("group"),
            ],
            axis=1,
        )
        raw_data.to_csv(SEMI_SUPERVISED_MODEL_DIR / "without_reduction_data.csv", index=False)
        self._save_training_variables(raw_data)
        return self
        
    def dimensionality_reduction(self):
        methods_params = {
            'PCA': {'n_components': 2},
            't-SNE': {'n_components': 2, 'perplexity': 30},
        }
        if os.getenv("ML_ENABLE_UMAP", "true").lower() != "false":
            methods_params['UMAP'] = {
                'n_components': 2,
                'n_neighbors': 15,
                'random_state': self.random_state,
                'n_jobs': 1,
            }
        self.complete_numerical_data = self.over_sampling_data_selected_feature_data.iloc[:, :-1]
        categorical_data = self.over_sampling_data_selected_feature_data["group"]

        prepared_for_dr = self._prepare_features_for_training(self.complete_numerical_data)
        reduced_data, plot_data = evaluate_dimensionality_reduction(prepared_for_dr, methods_params)
        if not plot_data:
            self.data_combined_PCA = pd.DataFrame()
            self.data_combined_tsne = pd.DataFrame()
            self.data_combined_UMAP = pd.DataFrame()
            return self

        combined_plot_data = pd.concat(plot_data)

        pca_plot_data = combined_plot_data[combined_plot_data["Method"] == "PCA"].reset_index(drop=True)
        if not pca_plot_data.empty:
            self.data_combined_PCA = pd.concat([pca_plot_data, categorical_data], axis=1)
            self.data_combined_PCA.to_csv(SEMI_SUPERVISED_MODEL_DIR / "PCA_data.csv", index=False)
        else:
            self.data_combined_PCA = pd.DataFrame()

        tsne_plot_data = combined_plot_data[combined_plot_data["Method"] == "t-SNE"].reset_index(drop=True)
        if not tsne_plot_data.empty:
            self.data_combined_tsne = pd.concat([tsne_plot_data, categorical_data], axis=1)
            self.data_combined_tsne.to_csv(SEMI_SUPERVISED_MODEL_DIR / "TSNE_data.csv", index=False)
        else:
            self.data_combined_tsne = pd.DataFrame()

        umap_plot_data = combined_plot_data[combined_plot_data["Method"] == "UMAP"].reset_index(drop=True)
        if not umap_plot_data.empty:
            self.data_combined_UMAP = pd.concat([umap_plot_data, categorical_data], axis=1)
            self.data_combined_UMAP.to_csv(SEMI_SUPERVISED_MODEL_DIR / "UMAP_data.csv", index=False)
        else:
            self.data_combined_UMAP = pd.DataFrame()
        
        return self

    def plot_charts(self):
        chart_list = {
            "pca": self.data_combined_PCA,
            "tsne": self.data_combined_tsne,
        }
        if not self.data_combined_UMAP.empty:
            chart_list["umap"] = self.data_combined_UMAP
        
        for key, obj in chart_list.items():
            if obj is None or obj.empty:
                continue
            title_map = {
                "pca": "Principal Component Analysis of MetaMP Training Records",
                "tsne": "t-SNE Projection of MetaMP Training Records",
                "umap": "UMAP Projection of MetaMP Training Records",
            }
            plotly_figure = self._build_plotly_dr_figure(
                obj,
                title=title_map.get(key, key.upper()),
                axis_label=key,
            )
            self._save_plotly_figure(plotly_figure, MODEL_ROOT / key)
        return self

    def run_classificationXXX(self, X, y, model_class, filename_prefix, X_unlabeled=None):
        """Run classification and save results."""
        metrics_list = []

        for run in range(self.num_runs):
            if X_unlabeled is not None:  # Semi-Supervised Case
                model = model_class(X, y, X_unlabeled, test_size=0.2, random_state=self.random_state + run)
            else:  # Supervised Case
                model = model_class(X, y, test_size=0.2, random_state=self.random_state + run)

            # Train and evaluate the models
            model.train_and_evaluate()

            # Collect metrics for aggregation
            metrics_list.append(model.results_df)

            # Save model and plot performance
            model.save_models(save_filename=f"{filename_prefix}_{run}")
            model.plot_performance_comparison(save_filename=f"{filename_prefix}_{run}")

        # Concatenate all metric results
        concatenated_metrics = pd.concat(metrics_list)

        # Select only numeric columns for aggregation
        numeric_columns = concatenated_metrics.select_dtypes(include=['number'])

        # Aggregate only numeric columns
        aggregated_metrics = numeric_columns.groupby(level=0).agg(['mean', 'std'])

        # Combine non-numeric data with aggregated numeric data (if needed)
        non_numeric_columns = concatenated_metrics.select_dtypes(exclude=['number']).drop_duplicates()
        if not non_numeric_columns.empty:
            aggregated_metrics = pd.concat([aggregated_metrics, non_numeric_columns], axis=1)

        # Save aggregated metrics
        aggregated_metrics.to_csv(f"./models/{filename_prefix}_metrics_mean.csv")
        print(f"Metrics saved to ./models/{filename_prefix}_metrics_mean.csv")      
        
    def run_classificationXXXXX(self, X, y, model_class, filename_prefix, X_unlabeled=None):
        """Run classification and save results."""
        metrics_list = []

        for run in range(self.num_runs):
            if X_unlabeled is not None:  # Semi-Supervised Case
                model = model_class(X, y, X_unlabeled, test_size=0.2, random_state=self.random_state + run)
            else:  # Supervised Case
                model = model_class(X, y, test_size=0.2, random_state=self.random_state + run)

            # Train and evaluate the models
            model.train_and_evaluate()

            # Perform cross-validation
            cv_scores = cross_val_score(model.model, X, y, cv=5)  # Adjust cv as needed
            print(f"Run {run} - Cross-validation scores: {cv_scores}")
            print(f"Run {run} - Mean cross-validation score: {cv_scores.mean()}")

            # Collect metrics for aggregation
            metrics_list.append(model.results_df)

            # Save model and plot performance
            model.save_models(save_filename=f"{filename_prefix}_{run}")
            model.plot_performance_comparison(save_filename=f"{filename_prefix}_{run}")

        # Concatenate all metric results
        concatenated_metrics = pd.concat(metrics_list)

        # Select only numeric columns for aggregation
        numeric_columns = concatenated_metrics.select_dtypes(include=['number'])

        # Aggregate only numeric columns
        aggregated_metrics = numeric_columns.groupby(level=0).agg(['mean', 'std'])

        # Combine non-numeric data with aggregated numeric data (if needed)
        non_numeric_columns = concatenated_metrics.select_dtypes(exclude=['number']).drop_duplicates()
        if not non_numeric_columns.empty:
            aggregated_metrics = pd.concat([aggregated_metrics, non_numeric_columns], axis=1)

        # Save aggregated metrics
        aggregated_metrics.to_csv(f"./models/{filename_prefix}_metrics_mean.csv")
        print(f"Metrics saved to ./models/{filename_prefix}_metrics_mean.csv")
    
    def plot_metrics_altair(self, metrics_data):
        # Convert metrics_data to a DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        
        # Melt the DataFrame to long format for Altair
        plot_data = metrics_df.melt(id_vars='Classifier', value_vars=['Mean Accuracy', 'Mean F1-Score', 'Mean Precision', 'Mean Recall'],
                                    var_name='Metric', value_name='Score')
        
        # Create a grouped bar chart with Altair
        chart = alt.Chart(plot_data).mark_bar().encode(
            x=alt.X('Classifier:N', title='Classifier'),
            y=alt.Y('Score:Q', title='Score'),
            color=alt.Color('Metric:N', title='Metric'),
            column=alt.Column('Metric:N', title='Metric')
        ).properties(
            title='Mean Performance Metrics for Each Classifier',
            width=200,
            height=300
        ).configure_axis(
            labelAngle=-45
        ).configure_view(
            stroke='transparent'
        )
        chart = self._style_altair_chart(chart, font_size=18)
        
        # Save the plot as a file
        self._save_altair_figure(chart, MODEL_ROOT / "metrics_comparison_altair", scale_factor=3.0)
        
    def run_classification(
        self,
        X,
        y,
        model_class,
        filename_prefix,
        X_unlabeled=None,
        X_test=None,
        y_test=None,
    ):
        """Run classification and save results."""
        metrics_list = []
        best_model = None
        best_clf_name = None
        best_runner = None
        best_score = -float('inf')  # Initialize with a very low value
        # Create an empty list to collect metrics for saving to CSV
        metrics_data = []
        X_prepared = self._prepare_features_for_training(X)
        X_unlabeled_prepared = self._prepare_features_for_training(X_unlabeled, fit_on=X) if X_unlabeled is not None else None
        X_test_prepared = self._prepare_features_for_training(X_test, fit_on=X) if X_test is not None else None
        
        for run in range(self.num_runs):
            if X_unlabeled_prepared is not None:  # Semi-Supervised Case
                model = model_class(X_prepared, y, X_unlabeled_prepared, test_size=0.2, random_state=self.random_state + run)
            elif X_test_prepared is not None and y_test is not None:
                model = model_class(
                    X_prepared,
                    y,
                    test_size=0.2,
                    random_state=self.random_state + run,
                    X_test=X_test_prepared,
                    y_test=y_test,
                )
            else:  # Supervised Case
                model = model_class(X_prepared, y, test_size=0.2, random_state=self.random_state + run)

            # Train and evaluate the models
            model.train_and_evaluate()
            for clf_name, clf in model.models.items():
                if X_unlabeled_prepared is not None:
                    row = model.results_df[model.results_df["Classifier"] == clf_name].iloc[0]
                    mean_accuracy = float(row["Accuracy"])
                    mean_f1 = float(row["F1-score"])
                    mean_precision = float(row["Precision"])
                    mean_recall = float(row["Recall"])
                    metrics_data.append({
                        'Run': run,
                        'Classifier': clf_name,
                        'Accuracy Scores': [mean_accuracy],
                        'Mean Accuracy': mean_accuracy,
                        'F1-Score': [mean_f1],
                        'Mean F1-Score': mean_f1,
                        'Precision': [mean_precision],
                        'Mean Precision': mean_precision,
                        'Recall': [mean_recall],
                        'Mean Recall': mean_recall
                    })
                else:
                    cv_splitter = StratifiedKFold(
                        n_splits=5,
                        shuffle=True,
                        random_state=self.random_state,
                    )
                    scoring = {
                        'accuracy': make_scorer(accuracy_score),
                        'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
                        'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
                        'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0)
                    }
                    cv_results = cross_validate(
                        clf,
                        X_prepared,
                        y,
                        cv=cv_splitter,
                        scoring=scoring,
                        return_train_score=False,
                    )
                    mean_accuracy = cv_results['test_accuracy'].mean()
                    mean_f1 = cv_results['test_f1_weighted'].mean()
                    mean_precision = cv_results['test_precision_weighted'].mean()
                    mean_recall = cv_results['test_recall_weighted'].mean()
                    metrics_data.append({
                        'Run': run,
                        'Classifier': clf_name,
                        'Accuracy Scores': cv_results['test_accuracy'],
                        'Mean Accuracy': mean_accuracy,
                        'F1-Score': cv_results['test_f1_weighted'],
                        'Mean F1-Score': mean_f1,
                        'Precision': cv_results['test_precision_weighted'],
                        'Mean Precision': mean_precision,
                        'Recall': cv_results['test_recall_weighted'],
                        'Mean Recall': mean_recall
                    })

                if mean_f1 > best_score:
                    best_score = mean_f1
                    best_model = clf
                    best_clf_name = clf_name
                    best_runner = model

                metrics_list.append(model.results_df)

            # Save models and plot performance only after all runs
            model.save_models(save_filename=f"{filename_prefix}_{run}")
            model.plot_performance_comparison(save_filename=f"{filename_prefix}_{run}")
            if hasattr(model, "save_diagnostics"):
                model.save_diagnostics(save_filename=f"{filename_prefix}_{run}")

        # Save the best model once after all runs
        if best_model is not None:
            # Use the last run number or a specific number if preferred
            self.save_best_model(best_model, best_clf_name, filename_prefix, self.num_runs - 1)

        # Convert metrics data to DataFrame
        metrics_df = pd.DataFrame(metrics_data)

        # Save metrics to CSV
        metrics_path = MODEL_ROOT / f"{filename_prefix}_cross_validation_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        self._log_info(f"Cross-validation metrics saved to {metrics_path}")

        # Concatenate all metric results
        concatenated_metrics = pd.concat(metrics_list)

        # Select only numeric columns for aggregation
        numeric_columns = concatenated_metrics.select_dtypes(include=['number'])

        # Aggregate only numeric columns
        aggregated_metrics = numeric_columns.groupby(level=0).agg(['mean', 'std'])

        # Combine non-numeric data with aggregated numeric data (if needed)
        non_numeric_columns = concatenated_metrics.select_dtypes(exclude=['number']).drop_duplicates()
        if not non_numeric_columns.empty:
            aggregated_metrics = pd.concat([aggregated_metrics, non_numeric_columns], axis=1)

        # Save aggregated metrics
        aggregated_metrics_path = MODEL_ROOT / f"{filename_prefix}_metrics_mean.csv"
        aggregated_metrics.to_csv(aggregated_metrics_path)
        self._log_info(f"Metrics saved to {aggregated_metrics_path}")
        
        self.plot_metrics_altair(metrics_data)
        return {
            "runner": best_runner,
            "best_classifier": best_clf_name,
            "best_model": best_model,
            "metrics_frame": metrics_df,
            "aggregated_metrics_path": str(aggregated_metrics_path),
            "cross_validation_metrics_path": str(metrics_path),
            "best_score": float(best_score) if best_score != -float("inf") else None,
        }

    @staticmethod
    def _prepare_features_for_training(X, fit_on=None):
        if X is None:
            return None

        X_frame = pd.DataFrame(X).copy()
        reference_frame = pd.DataFrame(fit_on).copy() if fit_on is not None else X_frame

        for column in X_frame.columns:
            reference_numeric = pd.to_numeric(reference_frame[column], errors="coerce")
            current_numeric = pd.to_numeric(X_frame[column], errors="coerce")
            if reference_numeric.notna().any():
                fallback = reference_numeric.median()
            elif current_numeric.notna().any():
                fallback = current_numeric.median()
            else:
                fallback = 0.0
            X_frame[column] = current_numeric.fillna(fallback)

        return X_frame
    
    def save_best_model(self, model, clf_name, filename_prefix, run):
        """Save the best model based on selected metric."""
        best_model_path = MODEL_ROOT / f"{filename_prefix}_best_{clf_name}_{run}.joblib"
        joblib.dump(model, best_model_path)
        self._log_info(f"Best model saved to {best_model_path}")

    def _build_topology_frame(self, categorical_data, fit=False):
        topology_frame = pd.DataFrame(index=categorical_data.index)
        for column in self.TRAINING_CATEGORICAL_FEATURES:
            if column not in categorical_data.columns:
                continue
            values = categorical_data[column].fillna("").astype(str).str.strip()
            if fit or column not in self.topology_label_maps:
                unique_values = sorted(values.unique().tolist())
                self.topology_label_maps[column] = {
                    value: index for index, value in enumerate(unique_values)
                }
            mapping = self.topology_label_maps.get(column, {})
            topology_frame[column] = values.map(mapping).fillna(-1).astype(int)
        return topology_frame

    def _build_feature_base_matrix(self, numerical_data, categorical_data, fit_topology=False):
        topology_frame = self._build_topology_frame(
            categorical_data,
            fit=fit_topology,
        )
        base_matrix = pd.concat(
            [numerical_data.reset_index(drop=True), topology_frame.reset_index(drop=True)],
            axis=1,
        )
        base_matrix.columns = pd.Index(
            [str(column) for column in base_matrix.columns],
            dtype="object",
        )
        return base_matrix

    def _build_semi_supervised_base_matrix(self):
        base_matrix = self._build_feature_base_matrix(
            self.numerical_data,
            self.categorical_data,
            fit_topology=True,
        )

        labels = self.categorical_data["group"].reset_index(drop=True)
        valid_mask = labels.notna() & labels.astype(str).str.strip().ne("")
        return (
            base_matrix.loc[valid_mask].reset_index(drop=True),
            labels.loc[valid_mask].reset_index(drop=True),
        )

    def _build_full_prediction_base_matrix(self):
        if self.inference_numerical_data.empty and self.inference_categorical_data.empty:
            return pd.DataFrame(), pd.DataFrame()

        metadata = self.inference_record_metadata.copy()
        if metadata.empty:
            metadata = pd.DataFrame(index=self.inference_data.index)

        if "pdb_code" in metadata.columns:
            valid_mask = metadata["pdb_code"].fillna("").astype(str).str.strip().ne("")
        else:
            valid_mask = pd.Series(True, index=metadata.index)

        base_matrix = self._build_feature_base_matrix(
            self.inference_numerical_data.loc[valid_mask].reset_index(drop=True),
            self.inference_categorical_data.loc[valid_mask].reset_index(drop=True),
            fit_topology=False,
        )
        metadata = metadata.loc[valid_mask].reset_index(drop=True)
        return base_matrix, metadata

    def _split_supervised_base_matrix(self):
        X_base, y_base = self._build_semi_supervised_base_matrix()
        return train_test_split(
            X_base,
            y_base,
            test_size=0.2,
            stratify=y_base,
            random_state=self.random_state,
        )

    def _select_training_features(self, X_labeled, y_labeled, num_features=30):
        if self.selected_feature_columns:
            return list(self.selected_feature_columns)
        return [str(column) for column in pd.DataFrame(X_labeled).columns.tolist()]

    def _build_reduced_training_view(self, X_labeled, X_unlabeled, method):
        X_labeled_prepared = self._prepare_features_for_training(X_labeled)
        X_unlabeled_prepared = self._prepare_features_for_training(
            X_unlabeled,
            fit_on=X_labeled,
        )

        if X_labeled_prepared is None or X_unlabeled_prepared is None:
            return None

        X_labeled_prepared = X_labeled_prepared.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_unlabeled_prepared = X_unlabeled_prepared.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        labeled_variance = X_labeled_prepared.var(axis=0, ddof=0)
        valid_columns = labeled_variance[labeled_variance > 0].index.tolist()
        if valid_columns:
            X_labeled_prepared = X_labeled_prepared[valid_columns]
            X_unlabeled_prepared = X_unlabeled_prepared.reindex(columns=valid_columns, fill_value=0.0)
        if X_labeled_prepared.empty or X_labeled_prepared.shape[1] == 0:
            return None

        if method == "pca":
            from sklearn.decomposition import PCA

            if X_labeled_prepared.shape[1] < 2 or len(X_labeled_prepared) < 2:
                return None
            reducer = PCA(n_components=2)
        elif method == "umap":
            from umap import UMAP

            reducer = UMAP(
                n_components=2,
                n_neighbors=15,
                random_state=self.random_state,
                n_jobs=1,
            )
        else:
            return None

        if method == "pca":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in divide",
                    category=RuntimeWarning,
                )
                labeled_array = reducer.fit_transform(X_labeled_prepared)
                unlabeled_array = reducer.transform(X_unlabeled_prepared)
        else:
            labeled_array = reducer.fit_transform(X_labeled_prepared)
            unlabeled_array = reducer.transform(X_unlabeled_prepared)

        labeled_view = pd.DataFrame(
            labeled_array,
            columns=["Component 1", "Component 2"],
            index=X_labeled.index,
        )
        unlabeled_view = pd.DataFrame(
            unlabeled_array,
            columns=["Component 1", "Component 2"],
            index=X_unlabeled.index,
        )
        return labeled_view, unlabeled_view, reducer

    def _build_leakage_safe_semi_supervised_views(self):
        X_base, y_base = self._build_semi_supervised_base_matrix()
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X_base,
            y_base,
            test_size=0.3,
            stratify=y_base,
            random_state=self.random_state,
        )

        top_features = self._select_training_features(X_labeled, y_labeled, num_features=30)
        X_labeled = X_labeled.reindex(columns=top_features).reset_index(drop=True)
        X_unlabeled = X_unlabeled.reindex(columns=top_features).reset_index(drop=True)
        y_labeled = y_labeled.reset_index(drop=True)

        views = {
            "no_dr": {
                "X_labeled": X_labeled,
                "X_unlabeled": X_unlabeled,
                "y_labeled": y_labeled,
                "feature_columns": top_features,
                "reducer": None,
            }
        }
        X_full, prediction_metadata = self._build_full_prediction_base_matrix()
        views["prediction_export"] = {
            "X_full": X_full.reindex(columns=top_features),
            "metadata": prediction_metadata,
            "feature_columns": top_features,
        }

        reduced_pca = self._build_reduced_training_view(X_labeled, X_unlabeled, method="pca")
        if reduced_pca is not None:
            views["pca"] = {
                "X_labeled": reduced_pca[0],
                "X_unlabeled": reduced_pca[1],
                "y_labeled": y_labeled,
                "feature_columns": top_features,
                "reducer": reduced_pca[2],
            }

        if os.getenv("ML_ENABLE_UMAP", "true").lower() != "false":
            try:
                reduced_umap = self._build_reduced_training_view(
                    X_labeled,
                    X_unlabeled,
                    method="umap",
                )
            except Exception as exc:
                self._log_warning(f"Skipping leakage-safe UMAP semi-supervised view: {exc}")
                reduced_umap = None
            if reduced_umap is not None:
                views["umap"] = {
                    "X_labeled": reduced_umap[0],
                    "X_unlabeled": reduced_umap[1],
                    "y_labeled": y_labeled,
                    "feature_columns": top_features,
                    "reducer": reduced_umap[2],
                }

        return views

    def _export_live_group_predictions(self, runner, classifier_name, X_full, metadata):
        if runner is None or not classifier_name or X_full is None or X_full.empty:
            return
        if not hasattr(runner, "models") or classifier_name not in runner.models:
            return

        prepared = self._prepare_features_for_training(X_full, fit_on=X_full)
        transformed = runner.imputer.transform(prepared)
        transformed = runner.scaler.transform(transformed)
        predictions = runner.models[classifier_name].predict(transformed)
        decoded_predictions = runner.label_encoder.inverse_transform(predictions)

        export_frame = metadata.copy()
        export_frame["predicted_group"] = [
            standardize_group_label(value) for value in decoded_predictions
        ]
        export_frame["model_name"] = classifier_name
        export_frame["generated_at"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        export_columns = [
            column
            for column in ["pdb_code", "predicted_group", "model_name", "generated_at"]
            if column in export_frame.columns
        ]
        if "pdb_code" not in export_columns:
            return

        export_path = MODEL_ROOT / "live_group_predictions.csv"
        export_frame[export_columns].drop_duplicates(subset=["pdb_code"]).to_csv(
            export_path,
            index=False,
        )
        self._log_info(f"Live group predictions saved to {export_path}")

    def _save_model_bundles(
        self,
        result,
        training_mode,
        reduction_key,
        feature_columns,
        reducer=None,
    ):
        runner = (result or {}).get("runner")
        metrics_frame = (result or {}).get("metrics_frame")
        if (
            runner is None
            or not hasattr(runner, "models")
            or not runner.models
        ):
            return []

        registry_entries = []
        for classifier_name, estimator in runner.models.items():
            artifact_id = f"{training_mode}_{reduction_key}_{self._safe_slug(classifier_name)}"
            artifact_path = self.production_paths["models"] / f"{artifact_id}.joblib"
            bundle = {
                "artifact_id": artifact_id,
                "training_mode": training_mode,
                "reduction_key": reduction_key,
                "classifier_name": classifier_name,
                "feature_columns": list(feature_columns or []),
                "model_input_columns": ["Component 1", "Component 2"] if reducer is not None else list(feature_columns or []),
                "numeric_feature_columns": list(self.TRAINING_NUMERIC_FEATURES),
                "categorical_feature_columns": list(self.TRAINING_CATEGORICAL_FEATURES),
                "topology_label_maps": self.topology_label_maps,
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "estimator": estimator,
                "imputer": getattr(runner, "imputer", None),
                "scaler": getattr(runner, "scaler", None),
                "label_encoder": getattr(runner, "label_encoder", None),
                "reducer": reducer,
            }
            joblib.dump(bundle, artifact_path)

            metric_row = {}
            if metrics_frame is not None and not metrics_frame.empty:
                classifier_rows = metrics_frame[metrics_frame["Classifier"] == classifier_name]
                if not classifier_rows.empty:
                    row = classifier_rows.iloc[0]
                    metric_row = {
                        "cv_mean_accuracy": float(row.get("Mean Accuracy", 0.0)),
                        "cv_mean_precision": float(row.get("Mean Precision", 0.0)),
                        "cv_mean_recall": float(row.get("Mean Recall", 0.0)),
                        "cv_mean_f1": float(row.get("Mean F1-Score", 0.0)),
                    }

            registry_entry = {
                "artifact_id": artifact_id,
                "training_mode": training_mode,
                "reduction_key": reduction_key,
                "classifier_name": classifier_name,
                "artifact_path": str(artifact_path),
                "feature_columns": json.dumps(list(feature_columns or [])),
                "selected_for_upload": False,
                "selected_for_mode": False,
                **metric_row,
            }
            self.model_bundle_registry.append(registry_entry)
            registry_entries.append(registry_entry)
        return registry_entries

    def _build_inference_feature_frame(self, dataframe, feature_columns):
        frame = pd.DataFrame(index=dataframe.index)
        for column in self.TRAINING_NUMERIC_FEATURES:
            if column in dataframe.columns:
                frame[column] = pd.to_numeric(dataframe[column], errors="coerce")
            else:
                frame[column] = np.nan
        topology_source = pd.DataFrame(index=dataframe.index)
        for column in self.TRAINING_CATEGORICAL_FEATURES:
            topology_source[column] = (
                dataframe[column]
                if column in dataframe.columns
                else pd.Series([""] * len(dataframe), index=dataframe.index)
            )
        topology_frame = self._build_topology_frame(topology_source, fit=False)
        frame = pd.concat([frame.reset_index(drop=True), topology_frame.reset_index(drop=True)], axis=1)
        frame = frame.reindex(columns=feature_columns)
        return frame

    def _predict_with_bundle(self, bundle, dataframe):
        feature_columns = bundle.get("feature_columns") or []
        feature_frame = self._build_inference_feature_frame(dataframe, feature_columns)
        transformed = feature_frame.copy()

        imputer = bundle.get("imputer")
        scaler = bundle.get("scaler")
        reducer = bundle.get("reducer")
        estimator = bundle.get("estimator")
        label_encoder = bundle.get("label_encoder")

        if reducer is not None:
            prepared_for_reducer = self._prepare_features_for_training(
                feature_frame,
                fit_on=feature_frame,
            )
            reduced_values = reducer.transform(prepared_for_reducer)
            transformed = pd.DataFrame(
                reduced_values,
                columns=bundle.get("model_input_columns") or ["Component 1", "Component 2"],
                index=feature_frame.index,
            )

        if imputer is not None:
            transformed = imputer.transform(transformed)
        if scaler is not None:
            transformed = scaler.transform(transformed)

        predictions = estimator.predict(transformed)
        if label_encoder is not None:
            predictions = label_encoder.inverse_transform(predictions)

        predicted_groups = [
            standardize_group_label(value) for value in pd.Series(predictions).tolist()
        ]
        return feature_frame, predicted_groups

    def _load_discrepancy_benchmark_frame(self):
        from src.Dashboard.services import DiscrepancyBenchmarkExportService

        discrepancy_frame = DiscrepancyBenchmarkExportService.build_export_dataframe(
            include_all=False
        )
        if discrepancy_frame.empty:
            return discrepancy_frame
        discrepancy_frame["pdb_code"] = (
            discrepancy_frame["pdb_code"].fillna("").astype(str).str.strip().str.upper()
        )
        return discrepancy_frame

    def _load_inference_source_frame(self):
        source_frame = self.inference_all_data.copy()
        source_frame["pdb_code"] = (
            source_frame["pdb_code"].fillna("").astype(str).str.strip().str.upper()
        )
        source_frame = source_frame.drop_duplicates(subset=["pdb_code"], keep="first")
        return source_frame

    def benchmark_and_export_predictions(self):
        if not self.model_bundle_registry:
            return self

        inference_source = self._load_inference_source_frame()
        expert_frame = self._load_expert_annotation_frame()
        discrepancy_frame = self._load_discrepancy_benchmark_frame()

        expert_input = expert_frame.merge(
            inference_source,
            on="pdb_code",
            how="left",
            suffixes=("_expert", ""),
        )
        discrepancy_input = discrepancy_frame.merge(
            inference_source,
            on="pdb_code",
            how="left",
            suffixes=("_benchmark", ""),
        )

        evaluation_rows = []
        for entry in self.model_bundle_registry:
            bundle = joblib.load(entry["artifact_path"])

            _, expert_predictions = self._predict_with_bundle(bundle, expert_input)
            expert_output = expert_frame.copy()
            expert_output["predicted_group"] = expert_predictions
            expert_output["predicted_group_benchmark"] = expert_output["predicted_group"].apply(
                collapse_group_label_for_expert_benchmark
            )
            expert_output["matches_expert_exact"] = (
                expert_output["group_expert_standardized"] == expert_output["predicted_group"]
            )
            expert_output["matches_expert"] = (
                expert_output["group_expert_benchmark"]
                == expert_output["predicted_group_benchmark"]
            )
            expert_output["training_mode"] = entry["training_mode"]
            expert_output["reduction_key"] = entry["reduction_key"]
            expert_output["classifier_name"] = entry["classifier_name"]
            expert_path = (
                self.production_paths["predictions"] / f"{entry['artifact_id']}_expert_annotation_predictions.csv"
            )
            expert_output.to_csv(expert_path, index=False)

            metrics = {
                "expert_row_count": int(len(expert_output.index)),
                "expert_accuracy": float(
                    accuracy_score(
                        expert_output["group_expert_benchmark"],
                        expert_output["predicted_group_benchmark"],
                    )
                ),
                "expert_precision_weighted": float(
                    precision_score(
                        expert_output["group_expert_benchmark"],
                        expert_output["predicted_group_benchmark"],
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "expert_recall_weighted": float(
                    recall_score(
                        expert_output["group_expert_benchmark"],
                        expert_output["predicted_group_benchmark"],
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "expert_f1_weighted": float(
                    f1_score(
                        expert_output["group_expert_benchmark"],
                        expert_output["predicted_group_benchmark"],
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "expert_accuracy_exact": float(
                    accuracy_score(
                        expert_output["group_expert_standardized"],
                        expert_output["predicted_group"],
                    )
                ),
                "expert_precision_weighted_exact": float(
                    precision_score(
                        expert_output["group_expert_standardized"],
                        expert_output["predicted_group"],
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "expert_recall_weighted_exact": float(
                    recall_score(
                        expert_output["group_expert_standardized"],
                        expert_output["predicted_group"],
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "expert_f1_weighted_exact": float(
                    f1_score(
                        expert_output["group_expert_standardized"],
                        expert_output["predicted_group"],
                        average="weighted",
                        zero_division=0,
                    )
                ),
            }

            _, discrepancy_predictions = self._predict_with_bundle(bundle, discrepancy_input)
            discrepancy_output = discrepancy_frame.copy()
            discrepancy_output["predicted_group"] = discrepancy_predictions
            discrepancy_output["training_mode"] = entry["training_mode"]
            discrepancy_output["reduction_key"] = entry["reduction_key"]
            discrepancy_output["classifier_name"] = entry["classifier_name"]
            discrepancy_path = (
                self.production_paths["predictions"] / f"{entry['artifact_id']}_discrepancy_predictions.csv"
            )
            discrepancy_output.to_csv(discrepancy_path, index=False)

            entry.update(
                {
                    **metrics,
                    "expert_predictions_path": str(expert_path),
                    "discrepancy_predictions_path": str(discrepancy_path),
                }
            )
            evaluation_rows.append(entry.copy())

        evaluation_df = pd.DataFrame(evaluation_rows)
        if evaluation_df.empty:
            return self

        evaluation_df["selection_priority"] = evaluation_df["reduction_key"].map(
            {"no_dr": 0, "pca": 1, "umap": 2}
        ).fillna(9)
        evaluation_df = evaluation_df.sort_values(
            by=[
                "expert_f1_weighted",
                "expert_accuracy",
                "cv_mean_f1",
                "selection_priority",
            ],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

        selected_artifact_id = evaluation_df.iloc[0]["artifact_id"]
        for index, row in evaluation_df.iterrows():
            is_selected = row["artifact_id"] == selected_artifact_id
            is_mode_selected = (
                row["artifact_id"]
                == evaluation_df[evaluation_df["training_mode"] == row["training_mode"]]
                .iloc[0]["artifact_id"]
            )
            evaluation_df.at[index, "selected_for_upload"] = bool(is_selected)
            evaluation_df.at[index, "selected_for_mode"] = bool(is_mode_selected)

        registry_path = self.production_paths["tables"] / "model_bundle_registry.csv"
        evaluation_df.drop(columns=["selection_priority"]).to_csv(registry_path, index=False)
        manifest = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "selected_upload_artifact_id": selected_artifact_id,
            "training_scope": self.training_scope_report,
            "registry_csv": str(registry_path),
            "prediction_template_columns": [
                "pdb_code",
                *self.TRAINING_NUMERIC_FEATURES,
                *self.TRAINING_CATEGORICAL_FEATURES,
            ],
            "available_artifacts": evaluation_df.drop(columns=["selection_priority"]).to_dict(orient="records"),
        }
        (self.production_paths["specs"] / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )
        self.model_bundle_registry = evaluation_df.drop(columns=["selection_priority"]).to_dict(orient="records")
        self._save_publication_outputs(evaluation_df.drop(columns=["selection_priority"]))
        return self

    def _save_publication_outputs(self, evaluation_df):
        comparison_table_path = self.production_paths["tables"] / "publication_model_comparison.csv"
        evaluation_df.to_csv(comparison_table_path, index=False)

        comparison_chart_df = evaluation_df.copy()
        reduction_order = {"no_dr": 0, "pca": 1, "umap": 2}
        mode_order = {"semi_supervised": 0, "supervised": 1}
        comparison_chart_df["model_label"] = (
            comparison_chart_df["training_mode"].str.replace("_", " ").str.title()
            + " / "
            + comparison_chart_df["reduction_key"].str.upper()
            + " / "
            + comparison_chart_df["classifier_name"]
        )
        comparison_chart_df["mode_rank"] = comparison_chart_df["training_mode"].map(mode_order).fillna(9)
        comparison_chart_df["reduction_rank"] = comparison_chart_df["reduction_key"].map(reduction_order).fillna(9)
        comparison_chart_df = comparison_chart_df.sort_values(
            by=["mode_rank", "reduction_rank", "classifier_name"]
        ).reset_index(drop=True)
        model_order = comparison_chart_df["model_label"].tolist()
        metric_rows = []
        for _, row in comparison_chart_df.iterrows():
            metric_rows.extend(
                [
                    {
                        "model_label": row["model_label"],
                        "training_mode": row["training_mode"],
                        "metric": "Expert F1",
                        "score": row["expert_f1_weighted"],
                    },
                    {
                        "model_label": row["model_label"],
                        "training_mode": row["training_mode"],
                        "metric": "CV F1",
                        "score": row["cv_mean_f1"],
                    },
                ]
            )
        metric_df = pd.DataFrame(metric_rows)
        metric_df["model_label"] = pd.Categorical(
            metric_df["model_label"],
            categories=model_order,
            ordered=True,
        )
        comparison_width = max(1080, len(model_order) * 95)

        grouped_chart = (
            alt.Chart(metric_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=22)
            .encode(
                x=alt.X("model_label:N", sort=model_order, title="Model bundle"),
                xOffset=alt.XOffset("metric:N", sort=["CV F1", "Expert F1"]),
                y=alt.Y("score:Q", title="Weighted score", scale=alt.Scale(domain=[0, 1.05])),
                color=alt.Color(
                    "metric:N",
                    title="Metric",
                    scale=alt.Scale(
                        domain=["CV F1", "Expert F1"],
                        range=["#1f5a91", "#d97706"],
                    ),
                ),
                tooltip=["model_label", "training_mode", "metric", alt.Tooltip("score:Q", format=".3f")],
            )
            .properties(
                width=comparison_width,
                height=420,
                title="Semi-supervised and supervised group-classification performance"
            )
            .configure_axis(labelAngle=-38, labelLimit=220, gridColor="#d7dde5", tickColor="#97a6b2")
            .configure_view(stroke=None)
        )
        grouped_chart = self._style_altair_chart(grouped_chart, font_size=20)
        self._save_altair_figure(
            grouped_chart,
            self.production_paths["figures"] / "semi_vs_supervised_performance",
            scale_factor=3.0,
        )
        (self.production_paths["specs"] / "semi_vs_supervised_performance.json").write_text(
            json.dumps(grouped_chart.to_dict(format="vega"), indent=2)
        )

        leaderboard_chart = (
            alt.Chart(comparison_chart_df)
            .mark_bar(cornerRadiusEnd=5)
            .encode(
                y=alt.Y("model_label:N", sort="-x", title="Model bundle"),
                x=alt.X(
                    "expert_f1_weighted:Q",
                    title="Expert-set weighted F1",
                    scale=alt.Scale(domain=[0, 1.0]),
                ),
                color=alt.Color(
                    "training_mode:N",
                    title="Training mode",
                    scale=alt.Scale(
                        domain=["semi_supervised", "supervised"],
                        range=["#0f766e", "#1f5a91"],
                    ),
                ),
                tooltip=[
                    "classifier_name",
                    "training_mode",
                    "reduction_key",
                    alt.Tooltip("expert_accuracy:Q", format=".3f"),
                    alt.Tooltip("expert_precision_weighted:Q", format=".3f"),
                    alt.Tooltip("expert_recall_weighted:Q", format=".3f"),
                    alt.Tooltip("expert_f1_weighted:Q", format=".3f"),
                    alt.Tooltip("expert_accuracy_exact:Q", format=".3f"),
                    alt.Tooltip("expert_precision_weighted_exact:Q", format=".3f"),
                    alt.Tooltip("expert_recall_weighted_exact:Q", format=".3f"),
                    alt.Tooltip("expert_f1_weighted_exact:Q", format=".3f"),
                    alt.Tooltip("cv_mean_accuracy:Q", format=".3f"),
                    alt.Tooltip("cv_mean_precision:Q", format=".3f"),
                    alt.Tooltip("cv_mean_recall:Q", format=".3f"),
                    alt.Tooltip("cv_mean_f1:Q", format=".3f"),
                ],
            )
            .properties(width=980, height=28 * max(len(comparison_chart_df.index), 4), title="Held-out expert benchmark leaderboard")
            .configure_axis(labelLimit=340, gridColor="#d7dde5", tickColor="#97a6b2")
            .configure_view(stroke=None)
        )
        leaderboard_chart = self._style_altair_chart(leaderboard_chart, font_size=20)
        self._save_altair_figure(
            leaderboard_chart,
            self.production_paths["figures"] / "expert_benchmark_leaderboard",
            scale_factor=3.0,
        )
        (self.production_paths["specs"] / "expert_benchmark_leaderboard.json").write_text(
            json.dumps(leaderboard_chart.to_dict(format="vega"), indent=2)
        )
        explainability_metadata = self._save_shap_publication_outputs(evaluation_df)
        (self.production_paths["specs"] / "explainability_manifest.json").write_text(
            json.dumps(explainability_metadata, indent=2, default=str)
        )

    def _save_shap_publication_outputs(self, evaluation_df):
        metadata = {
            "enabled": False,
            "reason": None,
            "selected_tree_bundle": None,
            "figure_paths": [],
            "table_paths": [],
        }
        if shap is None:
            metadata["reason"] = "SHAP is not installed in this environment."
            return metadata

        tree_classifiers = {
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting Classifier",
        }
        tree_candidates = evaluation_df[
            evaluation_df["classifier_name"].isin(tree_classifiers)
        ].copy()
        if tree_candidates.empty:
            metadata["reason"] = "No tree-based production bundles were available for SHAP."
            return metadata

        interpretable_candidates = tree_candidates[
            tree_candidates["reduction_key"] == "no_dr"
        ].copy()
        if interpretable_candidates.empty:
            metadata["reason"] = "No original-feature tree bundle was available for SHAP."
            return metadata

        interpretable_candidates = interpretable_candidates.sort_values(
            by=["expert_f1_weighted", "expert_accuracy", "cv_mean_f1"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        bundle_row = interpretable_candidates.iloc[0].to_dict()
        bundle = joblib.load(bundle_row["artifact_path"])
        sample_feature_frame, transformed_frame = self._build_shap_feature_frames(bundle)
        if transformed_frame is None or transformed_frame.empty:
            metadata["reason"] = "No eligible feature matrix was available for SHAP export."
            return metadata

        estimator = self._unwrap_shap_estimator(bundle.get("estimator"))
        if estimator is None:
            metadata["reason"] = "The selected tree bundle does not expose a SHAP-compatible estimator."
            return metadata
        label_encoder = bundle.get("label_encoder")
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed_frame)
        class_names = self._get_shap_class_names(label_encoder, shap_values)

        importance_df = self._summarize_shap_values(
            shap_values,
            transformed_frame.columns.tolist(),
            class_names,
        )
        importance_df["artifact_id"] = bundle_row["artifact_id"]
        importance_df["classifier_name"] = bundle_row["classifier_name"]
        importance_df["training_mode"] = bundle_row["training_mode"]
        importance_df["reduction_key"] = bundle_row["reduction_key"]
        importance_df["source_scope"] = "eligible_training_matrix_after_exclusions"
        importance_df["display_feature"] = importance_df["feature"].map(
            lambda value: FEATURE_DISPLAY_NAMES.get(value, str(value).replace("_", " ").title())
        )

        shap_table_path = self.production_paths["tables"] / f"{bundle_row['artifact_id']}_shap_feature_importance.csv"
        importance_df.to_csv(shap_table_path, index=False)

        overall_importance = (
            importance_df.groupby(["feature", "display_feature"], as_index=False)["mean_abs_shap"]
            .mean()
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        bar_path = self.production_paths["figures"] / f"{bundle_row['artifact_id']}_shap_bar.png"
        self._save_shap_bar_figure(
            overall_importance.head(15),
            bar_path,
            title=f"SHAP Feature Importance: {bundle_row['classifier_name']} ({bundle_row['training_mode']} / {bundle_row['reduction_key']})",
        )

        beeswarm_paths = self._save_shap_beeswarm_figures(
            shap_values=shap_values,
            feature_frame=transformed_frame.rename(
                columns={
                    column: FEATURE_DISPLAY_NAMES.get(column, str(column).replace("_", " ").title())
                    for column in transformed_frame.columns
                }
            ),
            class_names=class_names,
            artifact_id=bundle_row["artifact_id"],
        )

        metadata.update(
            {
                "enabled": True,
                "reason": None,
                "selected_tree_bundle": {
                    "artifact_id": bundle_row["artifact_id"],
                    "classifier_name": bundle_row["classifier_name"],
                    "training_mode": bundle_row["training_mode"],
                    "reduction_key": bundle_row["reduction_key"],
                    "expert_f1_weighted": float(bundle_row.get("expert_f1_weighted", 0.0)),
                    "expert_accuracy": float(bundle_row.get("expert_accuracy", 0.0)),
                    "cv_mean_f1": float(bundle_row.get("cv_mean_f1", 0.0)),
                    "feature_sample_count": int(len(transformed_frame.index)),
                },
                "figure_paths": self._with_pdf_companions([bar_path, *beeswarm_paths]),
                "table_paths": [str(shap_table_path)],
            }
        )
        return metadata

    @staticmethod
    def _with_pdf_companions(paths):
        ordered_paths = []
        seen = set()
        for raw_path in paths:
            path = Path(raw_path)
            for candidate in (path, path.with_suffix(".pdf")):
                candidate_str = str(candidate)
                if candidate_str in seen:
                    continue
                seen.add(candidate_str)
                ordered_paths.append(candidate_str)
        return ordered_paths

    def _build_shap_feature_frames(self, bundle, max_rows=500):
        feature_columns = bundle.get("feature_columns") or []
        if not feature_columns:
            return None, None

        feature_frame = self.complete_numerical_data.reindex(columns=feature_columns).copy()
        if feature_frame.empty:
            return None, None
        if len(feature_frame.index) > max_rows:
            feature_frame = feature_frame.sample(
                n=max_rows,
                random_state=self.random_state,
            ).reset_index(drop=True)
        else:
            feature_frame = feature_frame.reset_index(drop=True)

        transformed = feature_frame.copy()
        imputer = bundle.get("imputer")
        scaler = bundle.get("scaler")
        reducer = bundle.get("reducer")
        if imputer is not None:
            transformed = pd.DataFrame(
                imputer.transform(transformed),
                columns=feature_columns,
                index=feature_frame.index,
            )
        if reducer is not None:
            return None, None
        if scaler is not None:
            transformed = pd.DataFrame(
                scaler.transform(transformed.to_numpy()),
                columns=feature_columns,
                index=feature_frame.index,
            )
        return feature_frame, transformed

    @staticmethod
    def _unwrap_shap_estimator(estimator):
        current = estimator
        visited = set()
        supported_names = {
            "DecisionTreeClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
        }
        while current is not None:
            marker = id(current)
            if marker in visited:
                break
            visited.add(marker)

            if current.__class__.__name__ in supported_names:
                return current

            for attribute in ("estimator_", "base_estimator_", "estimator", "base_estimator"):
                nested = getattr(current, attribute, None)
                if nested is not None and nested is not current:
                    current = nested
                    break
            else:
                return None
        return None

    @staticmethod
    def _get_shap_class_names(label_encoder, shap_values):
        if isinstance(shap_values, list):
            class_count = len(shap_values)
        else:
            shap_array = np.asarray(shap_values)
            class_count = shap_array.shape[2] if shap_array.ndim == 3 else 1

        if label_encoder is not None and hasattr(label_encoder, "classes_"):
            classes = [
                standardize_group_label(value)
                for value in list(label_encoder.classes_)
            ]
            if len(classes) == class_count:
                return classes
        return [f"class_{index + 1}" for index in range(class_count)]

    @staticmethod
    def _summarize_shap_values(shap_values, feature_names, class_names):
        rows = []
        if isinstance(shap_values, list):
            for class_name, class_values in zip(class_names, shap_values):
                mean_abs = np.abs(np.asarray(class_values)).mean(axis=0)
                for feature_name, value in zip(feature_names, mean_abs):
                    rows.append(
                        {
                            "output_class": class_name,
                            "feature": feature_name,
                            "mean_abs_shap": float(value),
                        }
                    )
            return pd.DataFrame(rows)

        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            for class_index, class_name in enumerate(class_names):
                mean_abs = np.abs(shap_array[:, :, class_index]).mean(axis=0)
                for feature_name, value in zip(feature_names, mean_abs):
                    rows.append(
                        {
                            "output_class": class_name,
                            "feature": feature_name,
                            "mean_abs_shap": float(value),
                        }
                    )
            return pd.DataFrame(rows)

        mean_abs = np.abs(shap_array).mean(axis=0)
        return pd.DataFrame(
            [
                {
                    "output_class": class_names[0] if class_names else "overall",
                    "feature": feature_name,
                    "mean_abs_shap": float(value),
                }
                for feature_name, value in zip(feature_names, mean_abs)
            ]
        )

    @staticmethod
    def _style_shap_axes(ax):
        ax.set_title("")
        ax.set_facecolor("#ffffff")
        ax.figure.patch.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0)
        ax.grid(axis="x", color="#d7dde5", linewidth=0.8)

    @staticmethod
    def _save_shap_bar_figure(importance_df, output_path, title=None):
        plt.figure(figsize=(9, 5.8))
        ranked = importance_df.sort_values("mean_abs_shap", ascending=True)
        plt.barh(ranked["display_feature"], ranked["mean_abs_shap"], color="#1f5a91")
        plt.xlabel("Mean |SHAP value|")
        plt.ylabel("Feature")
        ax = plt.gca()
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        MLJob._style_shap_axes(ax)
        fig = plt.gcf()
        fig.tight_layout()
        MLJob._save_matplotlib_figure(fig, output_path)
        plt.close(fig)

    def _save_shap_beeswarm_figures(self, shap_values, feature_frame, class_names, artifact_id):
        output_paths = []
        max_display = min(10, len(feature_frame.columns))
        if isinstance(shap_values, list):
            for class_name, class_values in zip(class_names, shap_values):
                plt.figure(figsize=(9, 5.8))
                shap.summary_plot(
                    class_values,
                    feature_frame,
                    show=False,
                    max_display=max_display,
                )
                ax = plt.gca()
                ax.xaxis.label.set_fontweight("bold")
                ax.yaxis.label.set_fontweight("bold")
                plt.setp(ax.get_xticklabels(), fontweight="bold")
                plt.setp(ax.get_yticklabels(), fontweight="bold")
                self._style_shap_axes(ax)
                output_path = self.production_paths["figures"] / (
                    f"{artifact_id}_shap_beeswarm_{self._safe_slug(class_name)}.png"
                )
                fig = plt.gcf()
                fig.tight_layout()
                self._save_matplotlib_figure(fig, output_path)
                plt.close(fig)
                output_paths.append(output_path)
            return output_paths

        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            for class_index, class_name in enumerate(class_names):
                plt.figure(figsize=(9, 5.8))
                shap.summary_plot(
                    shap_array[:, :, class_index],
                    feature_frame,
                    show=False,
                    max_display=max_display,
                )
                ax = plt.gca()
                ax.xaxis.label.set_fontweight("bold")
                ax.yaxis.label.set_fontweight("bold")
                plt.setp(ax.get_xticklabels(), fontweight="bold")
                plt.setp(ax.get_yticklabels(), fontweight="bold")
                self._style_shap_axes(ax)
                output_path = self.production_paths["figures"] / (
                    f"{artifact_id}_shap_beeswarm_{self._safe_slug(class_name)}.png"
                )
                fig = plt.gcf()
                fig.tight_layout()
                self._save_matplotlib_figure(fig, output_path)
                plt.close(fig)
                output_paths.append(output_path)
            return output_paths

        plt.figure(figsize=(9, 5.8))
        shap.summary_plot(
            shap_array,
            feature_frame,
            show=False,
            max_display=max_display,
        )
        ax = plt.gca()
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        self._style_shap_axes(ax)
        output_path = self.production_paths["figures"] / f"{artifact_id}_shap_beeswarm.png"
        fig = plt.gcf()
        fig.tight_layout()
        self._save_matplotlib_figure(fig, output_path)
        plt.close(fig)
        output_paths.append(output_path)
        return output_paths

    @staticmethod
    def _save_altair_figure(chart, output_stem, scale_factor=3.0):
        output_stem = Path(output_stem)
        chart.save(str(output_stem.with_suffix(".png")), scale_factor=scale_factor)
        chart.save(str(output_stem.with_suffix(".pdf")))

    @staticmethod
    def _style_altair_chart(chart, font_size=18):
        return chart.configure_title(
            anchor="middle",
            fontSize=font_size,
            color="black",
        ).configure_legend(
            orient="bottom",
            direction="horizontal",
            titleAnchor="middle",
            labelLimit=220,
            symbolLimit=30,
        )

    @staticmethod
    def _save_matplotlib_figure(fig, output_path, dpi=400):
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

    @staticmethod
    def _build_plotly_dr_figure(dataframe, title, axis_label):
        df = dataframe.copy()
        hover_columns = [
            column
            for column in df.columns
            if column not in {"Component 1", "Component 2"}
        ]
        category_order = {}
        if "group" in df.columns:
            observed_groups = [
                value
                for value in DR_PLOTLY_PALETTE.keys()
                if value in set(df["group"].dropna().astype(str))
            ]
            if observed_groups:
                category_order["group"] = observed_groups

        fig = px.scatter(
            df,
            x="Component 1",
            y="Component 2",
            color="group" if "group" in df.columns else None,
            color_discrete_map=DR_PLOTLY_PALETTE if "group" in df.columns else None,
            hover_data=hover_columns,
            category_orders=category_order or None,
            title=title,
        )
        fig.update_traces(
            marker=dict(size=8, opacity=0.82, line=dict(width=0)),
            selector=dict(mode="markers"),
        )
        axis_prefix = axis_label.upper() if str(axis_label).lower() != "tsne" else "t-SNE"
        fig.update_layout(
            width=1120,
            height=760,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Arial Black, Arial, Helvetica, sans-serif", size=18, color="#28251d"),
            title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", font=dict(size=24)),
            legend=dict(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.18,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.94)",
                bordercolor="#d7dde5",
                borderwidth=1,
                title_text="<b>Group</b>" if "group" in df.columns else None,
                font=dict(size=16),
            ),
            margin=dict(l=85, r=55, t=95, b=105),
        )
        fig.update_xaxes(
            title=f"<b>{axis_prefix} 1</b>",
            showgrid=True,
            gridcolor="#d7dde5",
            zeroline=False,
            showline=False,
            tickfont=dict(size=17),
        )
        fig.update_yaxes(
            title=f"<b>{axis_prefix} 2</b>",
            showgrid=True,
            gridcolor="#d7dde5",
            zeroline=False,
            showline=False,
            tickfont=dict(size=17),
        )
        return fig

    @staticmethod
    def _save_plotly_figure(fig, output_stem):
        output_stem = Path(output_stem)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_stem.with_suffix(".png")), scale=3)
        fig.write_image(str(output_stem.with_suffix(".pdf")))
        html_path = output_stem.with_suffix(".html")
        fig.write_html(
            str(html_path),
            include_plotlyjs="cdn",
            include_mathjax=False,
            full_html=True,
        )
        html_text = html_path.read_text(encoding="utf-8")
        html_text = html_text.replace(
            "<script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>",
            "",
        )
        html_path.write_text(html_text, encoding="utf-8")
        fig.write_json(str(output_stem.with_suffix(".json")), pretty=True)

    def _build_leakage_safe_supervised_views(self):
        X_train, X_test, y_train, y_test = self._split_supervised_base_matrix()
        top_features = self._select_training_features(X_train, y_train, num_features=30)
        X_train = X_train.reindex(columns=top_features).reset_index(drop=True)
        X_test = X_test.reindex(columns=top_features).reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        views = {
            "no_dr": {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "feature_columns": top_features,
                "reducer": None,
            }
        }

        reduced_pca = self._build_reduced_training_view(X_train, X_test, method="pca")
        if reduced_pca is not None:
            views["pca"] = {
                "X_train": reduced_pca[0].reset_index(drop=True),
                "X_test": reduced_pca[1].reset_index(drop=True),
                "y_train": y_train,
                "y_test": y_test,
                "feature_columns": top_features,
                "reducer": reduced_pca[2],
            }

        if os.getenv("ML_ENABLE_UMAP", "true").lower() != "false":
            try:
                reduced_umap = self._build_reduced_training_view(
                    X_train,
                    X_test,
                    method="umap",
                )
            except Exception as exc:
                self._log_warning(f"Skipping leakage-safe UMAP supervised view: {exc}")
                reduced_umap = None
            if reduced_umap is not None:
                views["umap"] = {
                    "X_train": reduced_umap[0].reset_index(drop=True),
                    "X_test": reduced_umap[1].reset_index(drop=True),
                    "y_train": y_train,
                    "y_test": y_test,
                    "feature_columns": top_features,
                    "reducer": reduced_umap[2],
                }

        return views

    def semi_supervised_learning(self):
        """Run semi-supervised learning using leakage-safe training splits."""
        training_views = self._build_leakage_safe_semi_supervised_views()
        prediction_export = training_views.pop("prediction_export", None)
        for key, payload in training_views.items():
            result = self.run_classification(
                payload["X_labeled"],
                payload["y_labeled"],
                ClassifierComparisonSemiSupervised,
                f"semi_supervised_{key}",
                payload["X_unlabeled"],
            )
            self._save_model_bundles(
                result=result,
                training_mode="semi_supervised",
                reduction_key=key,
                feature_columns=payload.get("feature_columns"),
                reducer=payload.get("reducer"),
            )
            if key == "no_dr" and prediction_export:
                self._export_live_group_predictions(
                    result.get("runner") if result else None,
                    result.get("best_classifier") if result else None,
                    prediction_export.get("X_full"),
                    prediction_export.get("metadata"),
                )
        return self

    def supervised_learning(self):
        """Run leakage-safe supervised learning with publication-ready diagnostics."""
        training_views = self._build_leakage_safe_supervised_views()
        for key, payload in training_views.items():
            result = self.run_classification(
                payload["X_train"],
                payload["y_train"],
                ClassifierComparison,
                f"supervised_{key}",
                X_test=payload["X_test"],
                y_test=payload["y_test"],
            )
            self._save_model_bundles(
                result=result,
                training_mode="supervised",
                reduction_key=key,
                feature_columns=payload.get("feature_columns"),
                reducer=payload.get("reducer"),
            )
        return self

    def time_split_evaluation(self):
        if self.record_metadata.empty or "bibliography_year" not in self.record_metadata.columns:
            return self

        year_series = pd.to_numeric(
            self.record_metadata["bibliography_year"],
            errors="coerce",
        )
        valid_years = year_series.dropna().astype(int)
        if valid_years.empty:
            return self

        cutoff = os.getenv("ML_TIME_SPLIT_YEAR_CUTOFF")
        if cutoff:
            try:
                cutoff_year = int(cutoff)
            except ValueError:
                cutoff_year = None
        else:
            unique_years = sorted(valid_years.unique())
            cutoff_year = unique_years[-3] if len(unique_years) >= 3 else None

        if cutoff_year is None:
            return self

        metadata = self.record_metadata.copy()
        metadata["bibliography_year"] = year_series
        train_mask = metadata["bibliography_year"] <= cutoff_year
        test_mask = metadata["bibliography_year"] > cutoff_year

        if train_mask.sum() < 20 or test_mask.sum() < 5:
            return self

        X_train = self._prepare_features_for_training(self.complete_numerical_data.loc[train_mask])
        X_test = self._prepare_features_for_training(
            self.complete_numerical_data.loc[test_mask],
            fit_on=self.complete_numerical_data.loc[train_mask],
        )
        y_train = self.over_sampling_data_selected_feature_data.loc[train_mask, "group"]
        y_test = self.over_sampling_data_selected_feature_data.loc[test_mask, "group"]

        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            return self

        classifier = RandomForestClassifier(
            n_estimators=300,
            random_state=self.random_state,
            class_weight="balanced",
        )
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        metrics = {
            "cutoff_year": cutoff_year,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "precision_weighted": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        }

        serializable_metrics = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in metrics.items()
        }

        metrics_path = MODEL_ROOT / "time_split_evaluation_metrics.json"
        predictions_path = MODEL_ROOT / "time_split_evaluation_predictions.csv"
        report_path = MODEL_ROOT / "time_split_evaluation_report.txt"

        pd.DataFrame(
            [
                {
                    "pdb_code": metadata.loc[index, "pdb_code"] if "pdb_code" in metadata.columns else None,
                    "bibliography_year": metadata.loc[index, "bibliography_year"],
                    "actual_group": actual,
                    "predicted_group": predicted,
                }
                for index, actual, predicted in zip(
                    metadata.loc[test_mask].index,
                    y_test,
                    predictions,
                )
            ]
        ).to_csv(predictions_path, index=False)

        metrics_path.write_text(json.dumps(serializable_metrics, indent=2))
        report_path.write_text(
            classification_report(y_test, predictions, zero_division=0)
        )
        self._log_info(f"Time-split evaluation saved to {metrics_path}")
        return self
