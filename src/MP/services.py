import io
import json
import logging
import os
from pathlib import Path
import joblib
import requests
import altair as alt
import numpy as np
import plotly.express as px
import plotly.io as pio
from flask import current_app
from celery.result import AsyncResult
from src.Commands.Migration.classMigrate import Migration
from src.Dashboard.data import columns_to_retrieve, stats_data
from src.Dashboard.data import EM_columns, MM_columns, NMR_columns, X_ray_columns, reduce_value_length_version2
from src.Dashboard.services import (
    get_table_as_dataframe, 
    get_tables_as_dataframe,
    get_table_as_dataframe_download, 
    get_table_as_dataframe_exception,
    get_table_as_dataframe_with_specific_columns,
)
from src.Dashboard.group_standardization import standardize_group_label
import pandas as pd
from src.MP.data import cat_list
from src.MP.machine_learning_services_old import UnsupervisedPipeline
from src.MP.Helpers import get_joblib_files_and_splits
from src.services.graphs.helpers import Graph
from src.services.Helpers.fields_helper import (
    dimensionality_reduction_algorithms_helper_kit,
    machine_algorithms_helper_kit,
    missing_algorithms_helper_kit,
    normalization_algorithms_helper_kit,
    transform_data_dict_view,
    transform_data_view,
)
from src.Jobs.transformData import report_and_clean_missing_values
from src.core.celery_factory import celery
from src.ingestion.redis_support import get_redis_client
from src.ingestion.task_status_recorder import TaskStatusRecorder
from utils.package import (
    evaluate_dimensionality_reduction,
    onehot_encoder,
    select_features_using_decision_tree,
    separate_numerical_categorical,
)


def _coerce_optional_bool(value):
    if value is None or isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    return value


def _coerce_optional_int(value):
    if value in (None, ""):
        return None
    return int(value)


logger = logging.getLogger(__name__)

class DataService:
    _cache = {}

    @classmethod
    def _remember_cache_item(cls, key, value, max_entries=8):
        if key in cls._cache:
            cls._cache.pop(key)
        cls._cache[key] = value
        while len(cls._cache) > max_entries:
            oldest_key = next(iter(cls._cache))
            cls._cache.pop(oldest_key, None)

    @classmethod
    def invalidate_cache(cls):
        cls._cache.clear()

    @classmethod
    def _current_data_version(cls):
        try:
            from src.Dashboard.services import DashboardAnnotationDatasetService

            local_dataset = DashboardAnnotationDatasetService._load_local_merged_dataset()
            if not local_dataset.empty:
                annotation_path = (
                    DashboardAnnotationDatasetService.get_annotation_dataset_file()
                )
                annotation_mtime = 0
                if annotation_path and annotation_path.exists():
                    annotation_mtime = int(annotation_path.stat().st_mtime)
                return (
                    "local_merged_dataset",
                    tuple(local_dataset.columns.tolist()),
                    int(len(local_dataset.index)),
                    annotation_mtime,
                )
        except Exception:
            pass
        return ("database_tables",)

    @staticmethod
    def  get_data_by_column_search(table="membrane_proteins", column_name="rcsentinfo_experimental_method", value=None, page=1, per_page=10, distinct_column=None):
        data = get_table_as_dataframe_exception(table, column_name, value, page, per_page, distinct_column)
        return data
    
    @staticmethod
    def get_data_by_column_search_download(column_name="rcsentinfo_experimental_method", value=None):

        data = get_table_as_dataframe_download(
            table_name="membrane_proteins", columns=columns_to_retrieve(), filter_column=column_name, 
            filter_value=value
        )
        return data
    
    # Define a function to retrieve unique values for categorical columns
    def get_unique_values_for_categorical_columns():
        table_columns = stats_data()
        unique_values = set()
        for entry in table_columns:
            for item in entry['data']:
                unique_values.add(Migration.shorten_column_name(item['value'].split('*')[0]))

        unique_columns = list(unique_values)
        cache_key = (
            "categorical_values",
            DataService._current_data_version(),
            tuple(sorted(unique_columns)),
        )
        cached = DataService._cache.get(cache_key)
        if cached is not None:
            return cached

        df = get_table_as_dataframe_with_specific_columns("membrane_proteins", unique_columns)
        result = {}
        for column_name in unique_columns:
            if column_name not in df.columns:
                continue
            result[column_name] = df[column_name].dropna().unique()

        DataService._remember_cache_item(cache_key, result)
        return result
    
    @staticmethod
    def get_data_from_DB():
        cache_key = ("merged_database_payload", DataService._current_data_version())
        cached = DataService._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            from src.Dashboard.services import DashboardAnnotationDatasetService

            local_dataset = DashboardAnnotationDatasetService._load_local_merged_dataset()
        except Exception:
            local_dataset = pd.DataFrame()

        required_columns = {"pdb_code", "group", "subgroup"}
        if not local_dataset.empty and required_columns.issubset(local_dataset.columns):
            payload = (
                local_dataset.copy(deep=False),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                local_dataset.copy(deep=False),
            )
            DataService._remember_cache_item(cache_key, payload)
            return payload

        table_names = ['membrane_proteins', 'membrane_protein_opm']
        result_df = get_tables_as_dataframe(table_names, "pdb_code")
        result_df_db = get_table_as_dataframe("membrane_proteins")
        result_df_opm = get_table_as_dataframe("membrane_protein_opm")
        result_df_uniprot = get_table_as_dataframe("membrane_protein_uniprot")
        all_data = pd.merge(right=result_df, left=result_df_uniprot, on="pdb_code")
        payload = (result_df, result_df_db, result_df_opm, result_df_uniprot, all_data)
        DataService._remember_cache_item(cache_key, payload)
        return payload


class MPDatasetService:
    @staticmethod
    def get_records_by_experimental_method(experimental_method, page=1, per_page=10):
        return DataService.get_data_by_column_search(
            column_name="rcsentinfo_experimental_method",
            value=experimental_method,
            page=page,
            per_page=per_page,
        )

    @staticmethod
    def build_download_payload(experimental_method, download_format):
        data = DataService.get_data_by_column_search_download(
            "rcsentinfo_experimental_method",
            experimental_method,
        )
        dataframe = data["data"]
        if download_format == "csv":
            buffer = io.StringIO()
            dataframe.to_csv(buffer, index=False)
            return {
                "content": buffer.getvalue(),
                "content_type": "text/csv",
                "filename": "output_data.csv",
            }
        if download_format == "xlsx":
            buffer = io.BytesIO()
            dataframe.to_excel(buffer, index=False)
            return {
                "content": buffer.getvalue(),
                "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "filename": "output_data.xlsx",
            }
        raise ValueError(f"Unsupported download format: {download_format}")

    @staticmethod
    def get_categorical_values():
        return DataService.get_unique_values_for_categorical_columns()


class MPFilterService:
    DEFAULT_METHOD_OPTIONS = ["All", "EM", "Multiple methods", "NMR", "X-ray"]

    @classmethod
    def get_filter_options_payload(cls, method_type="All"):
        chart_types = transform_data_dict_view(
            [
                {"value": "bar_plot", "text": "bar_chart"},
                {"value": "line_plot", "text": "line_chart"},
                {"value": "scatter_plot", "text": "scatter_chart"},
            ],
            "Chart_options",
            "single",
            [],
            False,
        )
        categorical_column = transform_data_view(
            cat_list,
            "categorical",
            "single",
            [],
            False,
        )
        quantitative_columns = transform_data_view(
            cls._get_numeric_columns(method_type),
            "quantitative",
            "single",
            [],
            False,
        )
        experimental_method = transform_data_view(
            cls.DEFAULT_METHOD_OPTIONS,
            "experimental_method",
            "single",
            [],
            False,
        )
        return {
            "experimental_method": experimental_method,
            "quantitative": quantitative_columns,
            "categorical": categorical_column,
            "chart_types": chart_types,
        }

    @staticmethod
    def build_chart_payload(data):
        x_axis = data.get("x_axis", "rcsentinfo_molecular_weight") or ""
        y_axis = data.get("y_axis", "rcsentinfo_deposited_solvent_atom_count") or "resolution"
        categorical_axis = data.get("categorical_axis") or None
        experimental_method = data.get("experimental_method", "All")
        chart_type = data.get("chart_type", "bar_plot") or "point_plot"
        normalized_method = (
            None if experimental_method in {"All", ""} else experimental_method
        )
        data_frame = DataService.get_data_by_column_search_download(
            "rcsentinfo_experimental_method",
            normalized_method,
        )["data"]
        categorical_columns = reduce_value_length_version2(
            [
                value
                for value in [
                    categorical_axis,
                    "Group",
                    "Subgroup",
                    "Species",
                    "Taxonomic Domain",
                    "Expressed in Species",
                ]
                if value is not None
            ]
        )
        columns = categorical_columns + ([x_axis] if x_axis else []) + ([y_axis] if y_axis else [])
        data_frame = data_frame[list(set(columns))]

        if categorical_axis is not None and (not x_axis or not y_axis):
            data_frame = data_frame.groupby([data_frame[categorical_axis]]).size().reset_index(name="Value")
            plot = Graph(data_frame, axis=[categorical_axis, "Value"], labels=categorical_axis)
            plot = getattr(plot, str(chart_type).replace(" ", "_"))()
            plot.set_selection(type="single", groups=[categorical_axis]).encoding(
                tooltips=[categorical_axis, "Value"],
                encoding_tags=["norminal", "quantitative"],
                legend_columns=1,
            ).properties(
                width=0,
                title="Membrane Protein Structures categorized by "
                + categorical_axis.replace("rcsentinfo", " ").replace("_", " "),
            ).legend_config(orient="bottom").add_selection().interactive()
            return {"chart": plot.return_dict_obj()}

        label = "" if categorical_axis is None else categorical_axis
        plot = Graph(data_frame, axis=[x_axis, y_axis], labels=label)
        plot = getattr(plot, str(chart_type).replace(" ", "_"))()
        plot.set_selection(type="single", groups=[]).encoding(
            tooltips=columns,
            encoding_tags=["quantitative", "quantitative"],
            legend_columns=1,
            axis_label=[x_axis, y_axis],
        ).properties(
            width=0,
            title="Relationship between "
            + x_axis.replace("rcsentinfo", " ").replace("_", " ")
            + " and "
            + y_axis.replace("rcsentinfo", " ").replace("_", " "),
        ).legend_config(orient="bottom").add_selection().interactive()
        return {"chart": plot.return_dict_obj()}

    @classmethod
    def _get_numeric_columns(cls, method_type):
        if method_type == "X-ray":
            numeric_columns = list(set(X_ray_columns(include_general=False)))
            filtered = [
                item
                for item in numeric_columns
                if item not in ("group", "species")
            ] + ["resolution", "rcsentinfo_resolution_combined"]
        elif method_type == "NMR":
            numeric_columns = list(set(NMR_columns(include_general=False)))
            filtered = [
                item
                for item in numeric_columns
                if item not in ("group", "species")
            ]
        elif method_type == "Multiple methods":
            numeric_columns = list(set(MM_columns(include_general=False)))
            filtered = [
                item
                for item in numeric_columns
                if item not in ("group", "species")
            ] + ["resolution", "rcsentinfo_resolution_combined"]
        else:
            numeric_columns = list(
                set(X_ray_columns(include_general=False))
                & set(EM_columns(include_general=False))
                & set(NMR_columns(include_general=False))
            )
            filtered = [
                item
                for item in numeric_columns
                if item not in ("group", "species")
            ] + ["resolution", "rcsentinfo_resolution_combined"]

        if "rcsentinfo_resolution_combined" in filtered:
            filtered.remove("rcsentinfo_resolution_combined")
        return filtered


class MPModelingService:
    EXCLUDED_FIELD_GROUPS = [
        "reflns",
        "refine",
        "rcsb_",
        "rcs",
        "ref",
        "diffrn",
        "exptl",
        "cell_",
        "group_",
        "subgroup_",
        "species_",
        "expressed_in_species",
        "pdb",
        "taxonomic_domain",
        "symspa",
        "expcry",
        "em2cry",
        "software_",
    ]

    @classmethod
    def get_modeling_filter_payload(cls):
        return {
            "experimental_method_list": {
                "data": transform_data_view(
                    MPFilterService.DEFAULT_METHOD_OPTIONS,
                    "experimental_method",
                    "single",
                    [],
                    False,
                ),
                "help": """
                    The experimental method refers to the technique or approach
                    used to study membrane proteins in a laboratory setting
                    """,
            },
            "categorical_list": {
                "data": transform_data_view(
                    cat_list,
                    "categorical_columns",
                    "single",
                    [],
                    False,
                ),
                "help": """
                    
                    """,
            },
            "dimensionality_algorithms": {
                "data": dimensionality_reduction_algorithms_helper_kit(),
                "help": """
                    Dimensionality reduction refers to the process of simplifying
                    complex datasets while retaining essential information.
                    In the context of membrane protein research, techniques
                    like Principal Component Analysis (PCA) and t-Distributed
                    Stochastic Neighbor Embedding (t-SNE) are employed.
                    PCA identifies critical patterns, while t-SNE emphasizes
                    local similarities. These methods aid in visualizing and
                    interpreting high-dimensional data, enhancing the understanding
                    of intricate relationships within membrane protein datasets.
                    """,
                "child": {
                    "n_neighbors": transform_data_view(
                        [15, 20, 25, 30, 35, 40, 45, 50],
                        "n_neighbors",
                        "single",
                        ["umap_algorithm"],
                        False,
                    ),
                    "perplexity": transform_data_view(
                        [15, 20, 25, 30, 35, 40, 45, 50],
                        "perplexity",
                        "single",
                        ["tsne_algorithm"],
                        False,
                    ),
                    "min_dist": transform_data_view(
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                        "min_dist",
                        "single",
                        ["umap_algorithm"],
                        False,
                    ),
                },
            },
            "machine_algorithms": {
                "data": machine_algorithms_helper_kit(),
                "help": """
                    Unsupervised Machine Learning is a subset of machine learning
                    where the algorithm explores and identifies patterns in
                    data without explicit guidance. In membrane protein research,
                    unsupervised learning techniques, such as clustering and
                    dimensionality reduction, play a vital role. Clustering
                    algorithms group similar data points, revealing inherent
                    structures, while dimensionality reduction methods simplify
                    complex datasets. These unsupervised approaches contribute
                    to uncovering hidden relationships and structures within
                    membrane protein data, aiding researchers in gaining valuable
                    insights for further analysis and interpretation.
                """,
                "child": {
                    "distance_threshold": transform_data_view(
                        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        "distance_threshold",
                        "single",
                        ["agglomerative_clustering"],
                        False,
                    ),
                    "n_components": transform_data_view(
                        range(2, 10),
                        "n_components",
                        "single",
                        ["gaussian_clustering"],
                        False,
                    ),
                    "min_samples": transform_data_view(
                        [5, 10, 15, 20, 25, 30],
                        "min_samples",
                        "single",
                        ["optics_clustering", "dbscan_clustering"],
                        False,
                    ),
                    "n_clusters": transform_data_view(
                        range(2, 10),
                        "n_clusters",
                        "single",
                        ["agglomerative_clustering", "kMeans_clustering"],
                        False,
                    ),
                    "eps": transform_data_view(
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                        "eps",
                        "single",
                        ["dbscan_clustering"],
                        False,
                    ),
                },
            },
        }

    @classmethod
    def run_unsupervised_pipeline(cls, payload):
        from src.Training.services import get_quantification_data

        config = cls._normalize_unsupervised_payload(payload or {})
        if len(cls.EXCLUDED_FIELD_GROUPS) == len(config["excluded_fields"]):
            raise ValueError(
                "Kindly un-select one of the variables so that the ML can fit on something"
            )

        experimental_method = (
            None
            if config["experimental_method"] == "All"
            else config["experimental_method"]
        )
        dr_columns = cls._build_dr_columns(config["dimensionality_algorithms"])
        data_frame = DataService.get_data_by_column_search_download(
            "rcsentinfo_experimental_method",
            experimental_method,
        )["data"]
        data_frame = data_frame[get_quantification_data(experimental_method)]

        result, dataset, accuracy_metrics = (
            UnsupervisedPipeline(data_frame)
            .modify_dataframe(config["excluded_fields"])
            .select_numeric_columns()
            .apply_imputation(
                imputation_method=config["imputation_algorithms"],
                remove_by_percent=90,
            )
            .apply_normalization(
                normalization_method=config["normalization_algorithms"]
            )
            .apply_dimensionality_reduction(
                reduction_method=config["dimensionality_algorithms"],
                dr_columns=dr_columns,
                early_exaggeration=config["early_exaggeration"],
                DR_n_components=config["DR_n_components"],
                learning_rate=config["learning_rate"],
                perplexity=config["perplexity"],
                DR_metric=config["DR_metric"],
                n_iter=config["n_iter"],
                method=config["method"],
                angle=config["angle"],
                init=config["init"],
                n_epochs=config["n_epochs"],
                min_dist=config["min_dist"],
                n_neighbors=config["n_neighbors"],
            )
            .apply_clustering(
                distance_threshold=config["distance_threshold"],
                n_components=config["n_components"],
                min_samples=config["min_samples"],
                n_clusters=config["n_clusters"],
                method=config["machine_learning_algorithms"],
                linkage=config["linkage"],
                metric=config["metric"],
                eps=config["eps"],
            )
            .prepare_plot_DR(group_by=config["color_by"])
        )

        cluster_chart = cls._build_unsupervised_chart(
            result.copy(),
            dr_columns=dr_columns,
            label_column="classes",
            color_by=config["color_by"],
            title="Unsupervised Machine Learning (Clustering)",
            assign_cluster_labels=True,
        )
        reduction_chart = cls._build_unsupervised_chart(
            result.copy(),
            dr_columns=dr_columns,
            label_column=config["color_by"],
            color_by=config["color_by"],
            title="Dimensionality Reduction using "
            + config["dimensionality_algorithms"].replace("_", " ").title(),
            assign_cluster_labels=False,
        )

        return {
            "dataset": list(dataset.columns),
            "data": result.to_dict(orient="records"),
            "accuracy_metrics": accuracy_metrics,
            "DR_chart": reduction_chart,
            "chart": cluster_chart,
        }

    @staticmethod
    def _build_dr_columns(dimensionality_algorithm):
        get_column_tag = dimensionality_algorithm.upper().split("_")[0]
        return [f"{get_column_tag}{char}" for char in range(1, 3)]

    @staticmethod
    def _coerce_int(value, default, allow_none=False):
        if allow_none and value in (None, "", False):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _normalize_unsupervised_payload(cls, payload):
        color_by = payload.get("categorical_list", "species")
        if color_by == "exptl_method":
            color_by = "rcsentinfo_experimental_method"

        return {
            "machine_learning_algorithms": payload.get(
                "machine_algorithms",
                "kMeans_clustering",
            ),
            "imputation_algorithms": payload.get(
                "imputation_algorithms",
                "KNN_imputer_regressor",
            ),
            "normalization_algorithms": payload.get(
                "normalization_algorithms",
                "min_max_normalization",
            ),
            "dimensionality_algorithms": payload.get(
                "dimensionality_algorithms",
                "pca_algorithm",
            ),
            "experimental_method": payload.get(
                "experimental_method_list",
                "All",
            ),
            "color_by": color_by,
            "excluded_fields": payload.get("excluded_list", []),
            "distance_threshold": cls._coerce_int(
                payload.get("distance_threshold"),
                None,
                allow_none=True,
            ),
            "n_components": cls._coerce_int(payload.get("n_components"), 2),
            "min_samples": cls._coerce_int(payload.get("min_samples"), 15),
            "n_clusters": cls._coerce_int(payload.get("n_clusters"), 2),
            "linkage": payload.get("linkage", "ward") or "ward",
            "metric": payload.get("metric", "euclidean") or "euclidean",
            "eps": cls._coerce_float(payload.get("eps"), 0.3),
            "early_exaggeration": cls._coerce_int(
                payload.get("early_exaggeration"),
                12,
            ),
            "DR_n_components": cls._coerce_int(
                payload.get("DR_n_components"),
                2,
            ),
            "learning_rate": payload.get("learning_rate", "auto") or "auto",
            "perplexity": cls._coerce_int(payload.get("perplexity"), 30),
            "DR_metric": payload.get("DR_metric", "euclidean") or "euclidean",
            "n_iter": cls._coerce_int(payload.get("n_iter"), 1000),
            "method": payload.get("method", "barnes_hut") or "barnes_hut",
            "angle": cls._coerce_float(payload.get("angle"), 0.1),
            "init": payload.get("init", "pca") or "pca",
            "n_epochs": cls._coerce_int(payload.get("n_epochs"), 10),
            "min_dist": cls._coerce_float(payload.get("min_dist"), 0.1),
            "n_neighbors": cls._coerce_int(payload.get("n_neighbors"), 15),
        }

    @staticmethod
    def _build_unsupervised_chart(
        result,
        dr_columns,
        label_column,
        color_by,
        title,
        assign_cluster_labels,
    ):
        if assign_cluster_labels:
            result[label_column] = result["clustering"].apply(
                lambda value: f"{value}_Cluster"
            )
            groups = [label_column, color_by]
        else:
            groups = [label_column]

        scatter_plot = (
            Graph(result, axis=dr_columns, labels=label_column)
            .scatter_plot()
            .set_selection(type="single", groups=groups)
            .encoding(
                tooltips=result.columns,
                encoding_tags=["quantitative", "quantitative"],
                legend_columns=1,
            )
            .properties(width=0, title=title)
            .legend_config()
            .add_selection()
            .interactive()
        )
        return scatter_plot.return_dict_obj()


class MPArtifactService:
    DEFAULT_MODEL_DIR = Path("models") / "semi-supervised"

    @classmethod
    def list_models_and_reductions(cls):
        model_dir = cls._get_model_dir()
        if not model_dir.exists():
            return {"models": [], "dim_reductions": []}

        model_names, dim_reductions = get_joblib_files_and_splits(str(model_dir))
        return {
            "models": list(set(model_names)),
            "dim_reductions": list(set(dim_reductions)),
        }

    @classmethod
    def get_accuracy_records(cls, dim_reduction="pca"):
        accuracy_path = cls._get_model_dir() / f"semi_supervised_{dim_reduction}_0_main.csv"
        if not accuracy_path.exists():
            raise FileNotFoundError(f"Accuracy file not found for '{dim_reduction}'.")

        accuracy = pd.read_csv(accuracy_path)
        return {"data": accuracy.to_dict(orient="records")}

    @classmethod
    def get_dimensionality_reduction_charts(cls):
        model_dir = cls._get_model_dir()
        chart_files = {
            "pca_chart": model_dir / "PCA_data.csv",
            "t_sne_chart": model_dir / "TSNE_data.csv",
            "umap_chart": model_dir / "UMAP_data.csv",
        }
        charts = {}
        for key, path in chart_files.items():
            if not path.exists():
                continue
            charts[key] = cls._create_chart(pd.read_csv(path))
        if not charts:
            raise FileNotFoundError("No dimensionality reduction data files are available yet.")
        return charts

    @staticmethod
    def _create_chart(data):
        return (
            alt.Chart(data)
            .mark_circle()
            .encode(
                x="Component 1",
                y="Component 2",
                color="group",
                tooltip=["Method", "Parameter", "group"],
            )
            .properties(width="container")
            .interactive()
            .configure_legend(orient="bottom", direction="vertical", labelLimit=0)
            .to_dict(format="vega")
        )

    @classmethod
    def _get_model_dir(cls):
        configured_path = current_app.config.get("SEMI_SUPERVISED_MODEL_DIR")
        if configured_path:
            return Path(configured_path)
        return Path(current_app.root_path).parent / cls.DEFAULT_MODEL_DIR


class MPExternalSampleService:
    HEADERS = {
        "authority": "opm-back.cc.lehigh.edu:3000",
        "method": "GET",
        "scheme": "https",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Origin": "https://opm.phar.umich.edu",
        "Referer": "https://opm.phar.umich.edu/",
        "Sec-Ch-Ua": '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "cross-site",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    }
    HOST = "https://opm-back.cc.lehigh.edu/opm-backend/"
    EXPORTED_COLUMNS = [
        "id",
        "pdbid",
        "name",
        "resolution",
        "topology_subunit",
        "thickness",
        "subunit_segments",
        "tilt",
        "gibbs",
        "membrane_topology_in",
        "membrane_topology_out",
        "pdb_code",
    ]

    @classmethod
    def export_real_sample_csv(cls, pdb_codes):
        cleaned_codes = [code.strip() for code in pdb_codes if code and code.strip()]
        if len(cleaned_codes) > 20:
            raise ValueError("Too many PDB codes (maximum 20 allowed)")

        result_df = cls._fetch_records_for_proteins(cleaned_codes)
        if result_df is None or result_df.empty:
            raise LookupError("No data found for the provided PDB codes")

        output = io.BytesIO()
        result_df.to_csv(output, index=False)
        return output.getvalue()

    @classmethod
    def _fetch_records_for_proteins(cls, pdb_codes):
        all_dfs = []
        session = requests.Session()
        session.headers.update(cls.HEADERS)

        for pdb_code in pdb_codes:
            url = f"{cls.HOST}primary_structures?search={pdb_code}&sort=&pageSize=100"
            response = session.get(url, timeout=30)
            if response.status_code != 200:
                continue

            record = response.json().get("objects", [])
            if not record:
                continue

            filter_data = record[0]
            detail_url = f"{cls.HOST}primary_structures/{filter_data.get('id')}"
            response_filtered = session.get(detail_url, timeout=30)
            if response_filtered.status_code != 200:
                continue

            data_filtered = response_filtered.json()
            df_filtered = pd.json_normalize(data_filtered, sep="_")
            df_filtered["pdb_code"] = pdb_code
            all_dfs.append(df_filtered)

        if not all_dfs:
            return None

        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df[cls.EXPORTED_COLUMNS]


class MPSemiSupervisedPredictionService:
    EXPECTED_COLUMNS = {
        "pdb_code",
        "subunit_segments",
        "thickness",
        "tilt",
        "membrane_topology_in",
        "membrane_topology_out",
    }

    @classmethod
    def predict_uploaded_csv(cls, uploaded_frame, model_name, dim_reduction):
        if not cls.EXPECTED_COLUMNS.issubset(uploaded_frame.columns):
            raise ValueError("Missing or incorrect column names")

        baseline = cls._load_baseline_dataframe()
        data_ = pd.concat(
            [
                baseline,
                uploaded_frame[
                    [
                        "pdb_code",
                        "subunit_segments",
                        "thickness",
                        "tilt",
                        "membrane_topology_in",
                        "membrane_topology_out",
                    ]
                ],
            ],
            axis=0,
        )

        numerical_features = data_[["subunit_segments", "thickness", "tilt"]]
        encoded_data = onehot_encoder(
            data_[["membrane_topology_in", "membrane_topology_out"]]
        )
        complete_numerical_data = pd.concat([numerical_features, encoded_data], axis=1)

        reduction_key = cls._normalize_reduction_key(dim_reduction)
        methods_params = {
            "PCA": {"n_components": 2},
            "t-SNE": {"n_components": 2, "perplexity": 30},
            "UMAP": {
                "n_components": 2,
                "n_neighbors": 15,
                "random_state": 42,
                "n_jobs": 1,
            },
        }
        _, plot_data = evaluate_dimensionality_reduction(
            complete_numerical_data,
            {reduction_key: methods_params[reduction_key]},
        )

        combined_plot_data = pd.concat(plot_data)
        dm_data = combined_plot_data[
            combined_plot_data["Method"] == reduction_key
        ].reset_index(drop=True)
        label = data_["pdb_code"].reset_index(drop=True)
        complete_merge = pd.concat([dm_data, label], axis=1)
        complete_merge = complete_merge[
            complete_merge["pdb_code"].isin(uploaded_frame["pdb_code"].tolist())
        ]
        complete_merge = (
            complete_merge.drop_duplicates(subset="pdb_code").reset_index(drop=True)
        )

        model = cls._load_model(model_name, dim_reduction)
        predictions = model.predict(complete_merge[["Component 1", "Component 2"]])
        result_df = uploaded_frame.copy()
        result_df["predicted_class"] = predictions

        output = io.BytesIO()
        result_df.to_csv(output, index=False)
        return output.getvalue()

    @classmethod
    def _load_baseline_dataframe(cls):
        baseline_path = MPArtifactService._get_model_dir() / "without_reduction_data.csv"
        if not baseline_path.exists():
            raise FileNotFoundError("Missing semi-supervised baseline dataset.")
        return pd.read_csv(baseline_path)

    @classmethod
    def _load_model(cls, model_name, dim_reduction):
        model_dir = MPArtifactService._get_model_dir()
        requested_path = model_dir / f"{model_name}__semi_supervised_{dim_reduction}_0.joblib"
        fallback_path = model_dir / "Random Forest__semi_supervised_tsne_0.joblib"

        if requested_path.exists():
            return joblib.load(requested_path)
        if fallback_path.exists():
            return joblib.load(fallback_path)
        raise FileNotFoundError(f"Model {requested_path.name} doesn't exist.")

    @staticmethod
    def _normalize_reduction_key(dim_reduction):
        normalized = (dim_reduction or "PCA").upper()
        if normalized == "TSNE":
            return "t-SNE"
        if normalized == "PCA":
            return "PCA"
        if normalized == "UMAP":
            return "UMAP"
        raise ValueError(f"Unsupported dimensionality reduction '{dim_reduction}'.")


class MPMLWorkspaceService:
    DEFAULT_ARTIFACT_DIR = Path("data") / "models" / "production_ml"
    TEMPLATE_COLUMNS = [
        {
            "name": "pdb_code",
            "required": True,
            "type": "string",
            "description": "Unique row identifier or PDB-like code for the uploaded sample.",
        },
        {
            "name": "topology_subunit",
            "required": False,
            "type": "number",
            "description": "Topology subunit count or related OPM topology count when available.",
        },
        {
            "name": "thickness",
            "required": False,
            "type": "number",
            "description": "Membrane thickness value used in the MetaMP grouping model.",
        },
        {
            "name": "subunit_segments",
            "required": False,
            "type": "number",
            "description": "Number of subunit membrane segments or span count.",
        },
        {
            "name": "tilt",
            "required": False,
            "type": "number",
            "description": "Tilt angle associated with the membrane protein entry. ",
        },
        {
            "name": "gibbs",
            "required": False,
            "type": "number",
            "description": "Gibbs free-energy related value when available.",
        },
        {
            "name": "membrane_topology_in",
            "required": False,
            "type": "string",
            "description": "Topology label describing the inward-facing membrane side.",
        },
        {
            "name": "membrane_topology_out",
            "required": False,
            "type": "string",
            "description": "Topology label describing the outward-facing membrane side.",
        },
    ]
    TEMPLATE_ROWS = [
        {
            "pdb_code": "A123",
            "topology_subunit": 1,
            "thickness": 28.4,
            "subunit_segments": 1,
            "tilt": 8.6,
            "gibbs": -1.2,
            "membrane_topology_in": "cytoplasmic side",
            "membrane_topology_out": "periplasm",
        },
        {
            "pdb_code": "B456",
            "topology_subunit": 8,
            "thickness": 22.1,
            "subunit_segments": 16,
            "tilt": 31.4,
            "gibbs": -9.8,
            "membrane_topology_in": "matrix side",
            "membrane_topology_out": "intermembrane space",
        },
    ]

    @classmethod
    def build_summary(cls):
        manifest = cls._load_manifest()
        registry = cls._load_registry()
        charts = {
            "performance_comparison": cls._load_chart("semi_vs_supervised_performance.json"),
            "expert_leaderboard": cls._load_chart("expert_benchmark_leaderboard.json"),
            "dimensionality_reduction": cls._load_dimensionality_reduction_charts(),
        }

        selected_artifact = None
        selected_id = manifest.get("selected_upload_artifact_id")
        if selected_id and registry:
            selected_artifact = next(
                (item for item in registry if item.get("artifact_id") == selected_id),
                None,
            )

        return {
            "selected_upload_artifact_id": selected_id,
            "selected_artifact": selected_artifact,
            "artifacts": registry,
            "template": {
                "columns": cls.TEMPLATE_COLUMNS,
                "sample_rows": cls.TEMPLATE_ROWS,
            },
            "charts": charts,
            "training_scope": manifest.get("training_scope"),
            "explainability": cls._build_explainability_payload(),
            "metric_definitions": {
                "target_label_source": "Training target Y is the standardized MPstruc broad-group label from membrane_proteins.group.",
                "expert_accuracy": "Benchmark-aware accuracy on the 121 held-out expert rows. Expert Bitopic is treated as compatible with predicted Bitopic, alpha-helical transmembrane, or beta-barrel transmembrane. Expert Monotopic remains strictly monotopic.",
                "expert_precision_weighted": "Benchmark-aware weighted precision on the held-out expert rows using the same biological equivalence rule.",
                "expert_recall_weighted": "Benchmark-aware weighted recall on the held-out expert rows using the same biological equivalence rule.",
                "expert_f1_weighted": "Benchmark-aware weighted F1 on the same held-out expert rows using the same biological equivalence rule.",
                "expert_accuracy_exact": "Strict exact-match accuracy before the benchmark equivalence collapsing.",
                "expert_precision_weighted_exact": "Strict weighted precision before the benchmark equivalence collapsing.",
                "expert_recall_weighted_exact": "Strict weighted recall before the benchmark equivalence collapsing.",
                "expert_f1_weighted_exact": "Strict weighted F1 before the benchmark equivalence collapsing.",
                "cv_accuracy": "Cross-validated weighted accuracy measured on the model-training workflow, not on the held-out expert benchmark.",
                "cv_precision": "Cross-validated weighted precision measured on the model-training workflow, not on the held-out expert benchmark.",
                "cv_recall": "Cross-validated weighted recall measured on the model-training workflow, not on the held-out expert benchmark.",
                "cv_f1": "Cross-validated weighted F1 measured on the model-training workflow, not on the held-out expert benchmark.",
                "tsne_note": "t-SNE is displayed as an exploratory visualization only. It is not used as a production inference transform because it does not provide a stable held-out mapping for new samples.",
            },
            "commands": {
                "local": "python3 -m src.Commands.run_ml_pipeline",
                "docker": "docker compose exec flask-app python -m src.Commands.run_ml_pipeline",
            },
        }

    @classmethod
    def export_template_csv(cls):
        template_df = pd.DataFrame(cls.TEMPLATE_ROWS, columns=[item["name"] for item in cls.TEMPLATE_COLUMNS])
        output = io.BytesIO()
        template_df.to_csv(output, index=False)
        return output.getvalue()

    @classmethod
    def predict_uploaded_csv(cls, uploaded_frame, artifact_id=None):
        if uploaded_frame is None or uploaded_frame.empty:
            raise ValueError("Uploaded dataset is empty.")
        if "pdb_code" not in uploaded_frame.columns:
            raise ValueError("Missing required column 'pdb_code'.")

        manifest = cls._load_manifest()
        selected_artifact_id = artifact_id or manifest.get("selected_upload_artifact_id")
        if not selected_artifact_id:
            raise FileNotFoundError("No trained production ML artifact is available yet.")

        bundle = cls._load_bundle(selected_artifact_id)
        predicted_groups = cls._predict_groups(bundle, uploaded_frame.copy())

        result_df = uploaded_frame.copy()
        result_df["predicted_group"] = predicted_groups
        result_df["artifact_id"] = bundle.get("artifact_id")
        result_df["classifier_name"] = bundle.get("classifier_name")
        result_df["training_mode"] = bundle.get("training_mode")
        result_df["reduction_key"] = bundle.get("reduction_key")

        output = io.BytesIO()
        result_df.to_csv(output, index=False)
        return output.getvalue()

    @classmethod
    def _predict_groups(cls, bundle, uploaded_frame):
        feature_frame = cls._build_feature_frame(uploaded_frame, bundle)
        transformed = feature_frame

        if bundle.get("reducer") is not None:
            prepared = cls._prepare_features_for_reducer(feature_frame)
            reduced_values = bundle["reducer"].transform(prepared)
            transformed = pd.DataFrame(
                reduced_values,
                columns=bundle.get("model_input_columns") or ["Component 1", "Component 2"],
                index=feature_frame.index,
            )
        if bundle.get("imputer") is not None:
            transformed = bundle["imputer"].transform(transformed)
        if bundle.get("scaler") is not None:
            transformed = bundle["scaler"].transform(transformed)

        predictions = bundle["estimator"].predict(transformed)
        if bundle.get("label_encoder") is not None:
            predictions = bundle["label_encoder"].inverse_transform(predictions)

        return [
            standardize_group_label(value) for value in pd.Series(predictions).tolist()
        ]

    @staticmethod
    def _prepare_features_for_reducer(feature_frame):
        prepared = pd.DataFrame(feature_frame).copy()
        for column in prepared.columns:
            numeric_values = pd.to_numeric(prepared[column], errors="coerce")
            if numeric_values.notna().any():
                fallback = numeric_values.median()
            else:
                fallback = 0.0
            prepared[column] = numeric_values.fillna(fallback)
        return prepared

    @classmethod
    def _build_feature_frame(cls, uploaded_frame, bundle):
        feature_frame = pd.DataFrame(index=uploaded_frame.index)
        numeric_columns = bundle.get("numeric_feature_columns") or []
        categorical_columns = bundle.get("categorical_feature_columns") or []
        topology_maps = bundle.get("topology_label_maps") or {}

        for column in numeric_columns:
            if column in uploaded_frame.columns:
                feature_frame[column] = pd.to_numeric(uploaded_frame[column], errors="coerce")
            else:
                feature_frame[column] = np.nan

        for column in categorical_columns:
            values = (
                uploaded_frame[column]
                if column in uploaded_frame.columns
                else pd.Series([""] * len(uploaded_frame), index=uploaded_frame.index)
            )
            mapping = topology_maps.get(column, {})
            feature_frame[column] = (
                values.fillna("")
                .astype(str)
                .str.strip()
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )

        return feature_frame.reindex(columns=bundle.get("feature_columns") or [])

    @classmethod
    def _load_bundle(cls, artifact_id):
        bundle_path = cls._artifact_root() / "models" / f"{artifact_id}.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(f"ML artifact '{artifact_id}' was not found.")
        return joblib.load(bundle_path)

    @classmethod
    def _load_manifest(cls):
        manifest_path = cls._artifact_root() / "specs" / "manifest.json"
        if not manifest_path.exists():
            return {}
        return json.loads(manifest_path.read_text())

    @classmethod
    def _load_registry(cls):
        registry_path = cls._artifact_root() / "tables" / "model_bundle_registry.csv"
        if not registry_path.exists():
            return []
        registry_df = pd.read_csv(registry_path)
        return registry_df.to_dict(orient="records")

    @classmethod
    def _load_chart(cls, filename):
        chart_path = cls._artifact_root() / "specs" / filename
        if not chart_path.exists():
            return None
        return json.loads(chart_path.read_text())

    @classmethod
    def _load_dimensionality_reduction_charts(cls):
        try:
            return MPArtifactService.get_dimensionality_reduction_charts()
        except Exception:
            return {}

    @classmethod
    def _load_explainability_manifest(cls):
        explainability_path = cls._artifact_root() / "specs" / "explainability_manifest.json"
        if not explainability_path.exists():
            return {}
        return json.loads(explainability_path.read_text())

    @classmethod
    def _build_explainability_payload(cls):
        manifest = cls._load_explainability_manifest()
        if not manifest:
            return {}

        payload = dict(manifest)
        payload["figures"] = [
            cls._artifact_download_entry("figures", value)
            for value in manifest.get("figure_paths", [])
            if value
        ]
        payload["tables"] = [
            cls._artifact_download_entry("tables", value)
            for value in manifest.get("table_paths", [])
            if value
        ]
        return payload

    @staticmethod
    def _artifact_download_entry(category, raw_path):
        path = Path(str(raw_path))
        filename = path.name
        stem = path.stem.replace("_", " ")
        return {
            "filename": filename,
            "label": stem.title(),
            "endpoint": f"ml-workbench/{category}/{filename}",
        }

    @classmethod
    def get_artifact_file(cls, category, filename):
        safe_name = Path(filename).name
        if category not in {"figures", "tables"}:
            raise FileNotFoundError("Invalid artifact category.")

        artifact_path = cls._artifact_root() / category / safe_name
        if not artifact_path.exists() or not artifact_path.is_file():
            raise FileNotFoundError(f"Artifact '{safe_name}' was not found.")
        return artifact_path

    @classmethod
    def _artifact_root(cls):
        configured_path = (
            current_app.config.get("SEMI_SUPERVISED_MODEL_DIR")
            or os.getenv("SEMI_SUPERVISED_MODEL_DIR")
        )
        if configured_path:
            return Path(configured_path).resolve().parent / "production_ml"
        return Path(current_app.root_path).parent / cls.DEFAULT_ARTIFACT_DIR


class MPGroupSubgroupService:
    DROP_COLUMNS = [
        "bibliography_year",
        "processed_resolution",
        "gibbs",
        "thicknesserror",
        "tilterror",
        "citation_country",
        "is_master_protein",
        "description",
        "rcsentinfo_experimental_method",
        "exptl_crystal_grow_method",
        "exptl_crystal_grow_method1",
        "species_description",
        "famsupclasstype_type_name",
        "membrane_short_name",
        "species_name",
        "family_superfamily_name",
        "family_superfamily_classtype_name",
        "exptl_method",
        "pdbid",
        "name_y",
        "family_name_cache",
        "species_name_cache",
        "membrane_name_cache",
        "expressed_in_species",
        "rcsentinfo_nonpolymer_molecular_weight_maximum",
        "rcsentinfo_nonpolymer_molecular_weight_minimum",
    ]

    @classmethod
    def build_chart(cls, chart_type="sunburst", chart_width=800, chart_height=500):
        all_data, _, _, _, _ = DataService.get_data_from_DB()
        data = report_and_clean_missing_values(all_data, threshold=40)
        dynamic_columns_to_drop = [
            column
            for column in data.columns
            if "_citation_" in column
            or "_count_" in column
            or column.startswith("count_")
            or column.endswith("_count")
            or column.startswith("revision_")
            or column.endswith("_revision")
            or column.startswith("id_")
            or column.endswith("_id")
            or column == "id"
        ]
        data = data.drop(dynamic_columns_to_drop + cls.DROP_COLUMNS, axis=1, errors="ignore")
        data = data.dropna()

        numerical_cols, categorical_cols = separate_numerical_categorical(data)
        numerical_data = data[numerical_cols]
        categorical_data = data[categorical_cols]

        feature_frame = pd.concat([numerical_data, categorical_data["subgroup"]], axis=1)
        selected_features, _ = select_features_using_decision_tree(
            feature_frame,
            target_column="subgroup",
            num_features=10,
        )
        encoded_data = onehot_encoder(categorical_data[[]])
        complete_numerical_data = pd.concat([selected_features, encoded_data], axis=1)
        complete_data = pd.concat([complete_numerical_data, categorical_data], axis=1)
        group_subgroup_counts = (
            complete_data.groupby(["group", "subgroup"])
            .size()
            .reset_index(name="count")
        )
        new_frame = pd.merge(
            right=complete_data,
            left=group_subgroup_counts[["subgroup", "count"]],
            on="subgroup",
        )

        if chart_type == "sunburst":
            fig = px.sunburst(
                new_frame,
                path=["group", "subgroup", "count", "pdb_code"],
                values="count",
                color="count",
                color_continuous_scale="blues",
                width=chart_width,
                height=chart_height,
            )
            fig.update_layout(
                title_text="Sunburst Chart of Membrane Protein Sub-groups."
            )
            return pio.to_json(fig)

        fig = px.treemap(
            new_frame,
            path=["group", "subgroup", "count", "pdb_code"],
            values="count",
            color="group",
            hover_data=["group", "subgroup", "count", "pdb_code"],
            color_discrete_map={
                "MONOTOPIC MEMBRANE PROTEINS": "#2ca25f",
                "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL": "#fdc086",
                "TRANSMEMBRANE PROTEINS:BETA-BARREL": "#beaed4",
            },
            color_continuous_midpoint=np.average(
                new_frame["count"],
                weights=new_frame["count"],
            ),
            width=chart_width,
            height=chart_height,
        )
        fig.update_traces(root_color="lightgrey")
        fig.update_layout(
            title_text="Sunburst Chart of Membrane Protein Sub-groups.",
            margin=dict(t=50, l=25, r=25, b=25),
        )
        return pio.to_json(fig)


class MPSequenceTMService:
    @staticmethod
    def parse_uploaded_csv(uploaded_file):
        dataframe = pd.read_csv(uploaded_file)
        if "sequence" not in dataframe.columns:
            raise ValueError("Missing required column: 'sequence'")

        row_count = len(dataframe)
        if not (1 <= row_count <= 4):
            raise ValueError(f"Upload between 1 and 4 sequences; you sent {row_count}")

        if "id" not in dataframe.columns:
            dataframe = dataframe.reset_index().rename(columns={"index": "id"})

        return dataframe

    @staticmethod
    def analyze_sequences(dataframe):
        try:
            from src.Jobs.LoadProteinPredictions import (
                run_tm_prediction_for_sequences,
            )
        except Exception as error:
            raise RuntimeError(
                "TM prediction dependencies are not available in this service image."
            ) from error

        result = run_tm_prediction_for_sequences(
            dataframe.to_dict(orient="records"),
            include_deeptmhmm=True,
            use_gpu=False,
            max_workers=2,
        )
        return result["records"]

    @classmethod
    def submit_async(
        cls,
        dataframe,
        include_deeptmhmm=None,
        use_gpu=None,
        max_workers=None,
    ):
        from src.Jobs.tasks.task1 import predict_tm_sequences

        task = predict_tm_sequences.delay(
            records=dataframe.to_dict(orient="records"),
            include_deeptmhmm=_coerce_optional_bool(include_deeptmhmm),
            use_gpu=_coerce_optional_bool(use_gpu),
            max_workers=_coerce_optional_int(max_workers),
        )
        return {
            "task_id": task.id,
            "task_name": "shared-task-predict-tm-sequences",
            "status": "queued",
        }

    @staticmethod
    def get_async_status(task_id):
        task = AsyncResult(task_id, app=celery)
        payload = {
            "task_id": task_id,
            "status": task.status.lower(),
        }
        if task.successful():
            payload["result"] = task.result
        elif task.failed():
            payload["error"] = str(task.result)
        return payload


class MPTMBackfillService:
    TASK_NAME = "shared-task-sync-tmalphafold-predictions"

    @classmethod
    def queue_backfill(
        cls,
        use_gpu=None,
        batch_size=None,
        max_workers=None,
    ):
        from src.MP.startup_sync import _find_live_tmalphafold_task
        from src.Jobs.tasks.task1 import sync_tmalphafold_predictions_task

        live_task = _find_live_tmalphafold_task()
        if live_task is not None:
            return {
                "task_id": live_task.get("task_id"),
                "task_name": cls.TASK_NAME,
                "status": "already-active",
            }

        task = sync_tmalphafold_predictions_task.delay(
            methods=None,
            with_tmdet=True,
            refresh=False,
            max_workers=_coerce_optional_int(max_workers) or 8,
            timeout=30,
            backfill_sequences=True,
            with_tmbed=True,
            tmbed_use_gpu=_coerce_optional_bool(use_gpu),
            tmbed_batch_size=_coerce_optional_int(batch_size),
            tmbed_max_workers=_coerce_optional_int(max_workers),
            tmbed_refresh=False,
        )
        return {
            "task_id": task.id,
            "task_name": cls.TASK_NAME,
            "status": "queued",
        }

    @classmethod
    def latest_status(cls):
        return TaskStatusRecorder(current_app.config).read_latest(cls.TASK_NAME)


class MPTMRateLimitService:
    SEQUENCE_ACTION = "sequence"
    BACKFILL_ACTION = "backfill"

    @classmethod
    def enforce(cls, action, identifier):
        if not cls._is_enabled():
            return

        redis_client = get_redis_client(config=current_app.config)
        if redis_client is None:
            return

        limit = cls._get_limit(action)
        if limit is None or limit <= 0:
            return

        window_seconds = cls._get_window_seconds()
        window_bucket = cls._get_window_bucket(window_seconds)
        key = f"metamp:rate-limit:tm:{action}:{identifier}:{window_bucket}"

        try:
            current_count = redis_client.incr(key)
            if current_count == 1:
                redis_client.expire(key, window_seconds)
        except Exception as exc:
            logger.warning("TM rate limit check failed open: %s", exc)
            return

        if current_count > limit:
            raise ValueError(
                f"Too many TM prediction requests. Try again in about {window_seconds} seconds."
            )

    @staticmethod
    def _is_enabled():
        value = str(
            current_app.config.get("TM_PREDICTION_RATE_LIMIT_ENABLED", "true")
        ).lower()
        return value in {"1", "true", "t", "yes", "y"}

    @staticmethod
    def _get_window_seconds():
        return int(
            current_app.config.get("TM_PREDICTION_RATE_LIMIT_WINDOW_SECONDS", 60)
        )

    @classmethod
    def _get_limit(cls, action):
        if action == cls.SEQUENCE_ACTION:
            return int(
                current_app.config.get(
                    "TM_PREDICTION_RATE_LIMIT_SEQUENCE_REQUESTS",
                    10,
                )
            )
        if action == cls.BACKFILL_ACTION:
            return int(
                current_app.config.get(
                    "TM_PREDICTION_RATE_LIMIT_BACKFILL_REQUESTS",
                    2,
                )
            )
        return None

    @staticmethod
    def _get_window_bucket(window_seconds):
        import time

        return int(time.time() // max(window_seconds, 1))
