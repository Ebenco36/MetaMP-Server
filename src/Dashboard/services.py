import random
import json
import logging
import ast
import re
import io
import hashlib
from numbers import Number
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
from flask import abort, current_app
from database.db import db
from sqlalchemy import func
from datetime import datetime, timedelta
from sqlalchemy.sql import select
from sqlalchemy import or_, and_
from src.MP.model_opm import OPM
from collections import OrderedDict
from src.services.graphs.helpers import convert_chart
from utils.redisCache import RedisCache
from src.MP.model_uniprot import Uniprot
from sqlalchemy.orm import Query, aliased
from sqlalchemy import select, func, Table
from sqlalchemy.exc import SQLAlchemyError
from src.MP.model import MembraneProteinData
from src.Feedbacks.models import DiscrepancyReview
from src.Dashboard.scientific_assessment import build_scientific_assessment
from src.Dashboard.group_standardization import (
    collapse_group_label_for_disagreement,
    parse_tm_count,
    standardize_group_label,
)
from src.AI_Packages.TMAlphaFoldPredictorClient import TMALPHAFOLD_MEMBRANE_LABELS
from src.core.audit_log import MetaMPAuditLogService
from src.Jobs.TMAlphaFoldSync import (
    get_normalized_tm_prediction_summaries_for_pdb_codes,
    get_normalized_tm_prediction_summaries,
    get_tmalphafold_prediction_summaries,
)
from src.services.data.columns.quantitative.quantitative import cell_columns, rcsb_entries
from src.services.data.columns.quantitative.quantitative_array import quantitative_array_column
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
logger = logging.getLogger(__name__)


class DashboardConfigurationService:
    DEFAULT_ANNOTATION_DATASET_NAME = "expert_annotation_predicted.csv"
    DEFAULT_TM_PREDICTION_OUTPUT = Path("/var/app/data/tm_predictions/tm_summary.csv")
    DEFAULT_BENCHMARK_DIR = Path("/var/app/data/benchmarks")
    DEFAULT_LIVE_GROUP_PREDICTIONS = Path("/var/app/data/models/live_group_predictions.csv")

    @classmethod
    def get_annotation_dataset_path(cls):
        configured_path = current_app.config.get(
            "DASHBOARD_ANNOTATION_DATASET_PATH"
        ) or current_app.config.get("ANNOTATION_DATASET_PATH")
        if configured_path:
            configured = Path(configured_path)
            if configured.exists():
                return configured

        candidate_roots = []

        dataset_root = current_app.config.get("INGESTION_DATASET_ROOT")
        if dataset_root:
            candidate_roots.append(Path(dataset_root))

        dataset_base_dir = current_app.config.get("INGESTION_DATASET_BASE_DIR")
        if dataset_base_dir:
            candidate_roots.append(Path(dataset_base_dir))

        candidate_roots.extend(
            [
                Path(current_app.root_path) / "datasets",
                Path(current_app.root_path).parent / "datasets",
                Path(current_app.root_path).parent.parent / "datasets",
                Path.cwd() / "datasets",
                Path("/var/app/data/datasets"),
                Path("/var/app/datasets"),
            ]
        )

        seen = set()
        for root in candidate_roots:
            normalized_root = str(root)
            if normalized_root in seen:
                continue
            seen.add(normalized_root)

            candidate = root / cls.DEFAULT_ANNOTATION_DATASET_NAME
            if candidate.exists():
                return candidate

        return Path(current_app.root_path).parent / "datasets" / cls.DEFAULT_ANNOTATION_DATASET_NAME

    @classmethod
    def get_valid_dataset_path(cls, dataset_name):
        candidate_roots = []

        dataset_base_dir = current_app.config.get("INGESTION_DATASET_BASE_DIR")
        if dataset_base_dir:
            candidate_roots.append(Path(dataset_base_dir) / "valid")
            candidate_roots.append(Path(dataset_base_dir))

        candidate_roots.extend(
            [
                Path(current_app.root_path) / "datasets" / "valid",
                Path(current_app.root_path).parent / "datasets" / "valid",
                Path(current_app.root_path).parent.parent / "datasets" / "valid",
                Path.cwd() / "datasets" / "valid",
                Path("/var/app/data/datasets/valid"),
                Path("/var/app/data/datasets"),
                Path("/var/app/datasets/valid"),
                Path("/var/app/datasets"),
            ]
        )

        seen = set()
        for root in candidate_roots:
            normalized_root = str(root)
            if normalized_root in seen:
                continue
            seen.add(normalized_root)

            candidate = root / dataset_name
            if candidate.exists():
                return candidate

        return None
    @classmethod
    def get_tm_prediction_output_path(cls):
        configured_path = current_app.config.get("TM_PREDICTION_OUTPUT_CSV")
        if configured_path:
            return Path(configured_path)
        return cls.DEFAULT_TM_PREDICTION_OUTPUT

    @classmethod
    def get_live_group_predictions_path(cls):
        configured_path = current_app.config.get("LIVE_GROUP_PREDICTIONS_PATH")
        if configured_path:
            return Path(configured_path)
        return cls.DEFAULT_LIVE_GROUP_PREDICTIONS

    @classmethod
    def get_raw_opm_dataset_path(cls):
        candidate_roots = []

        dataset_base_dir = current_app.config.get("INGESTION_DATASET_BASE_DIR")
        if dataset_base_dir:
            candidate_roots.append(Path(dataset_base_dir))

        dataset_root = current_app.config.get("INGESTION_DATASET_ROOT")
        if dataset_root:
            candidate_roots.append(Path(dataset_root))

        candidate_roots.extend(
            [
                Path(current_app.root_path) / "datasets",
                Path(current_app.root_path).parent / "datasets",
                Path(current_app.root_path).parent.parent / "datasets",
                Path.cwd() / "datasets",
                Path("/var/app/data/datasets"),
                Path("/var/app/datasets"),
            ]
        )

        seen = set()
        for root in candidate_roots:
            normalized_root = str(root)
            if normalized_root in seen:
                continue
            seen.add(normalized_root)
            if not root.exists():
                continue

            for candidate in sorted(root.glob("NEWOPM*.csv"), reverse=True):
                if candidate.name == "NEWOPM.csv":
                    continue
                return candidate

        return None

    @classmethod
    def get_raw_pdb_dataset_path(cls):
        configured_dataset_root = current_app.config.get("INGESTION_DATASET_BASE_DIR")
        candidate_roots = []
        if configured_dataset_root:
            candidate_roots.append(Path(configured_dataset_root))

        dataset_root = current_app.config.get("INGESTION_DATASET_ROOT")
        if dataset_root:
            candidate_roots.append(Path(dataset_root))

        candidate_roots.extend(
            [
                Path(current_app.root_path) / "datasets",
                Path(current_app.root_path).parent / "datasets",
                Path(current_app.root_path).parent.parent / "datasets",
                Path.cwd() / "datasets",
                Path("/var/app/data/datasets"),
                Path("/var/app/datasets"),
            ]
        )

        seen = set()
        for root in candidate_roots:
            normalized_root = str(root)
            if normalized_root in seen:
                continue
            seen.add(normalized_root)
            if not root.exists():
                continue

            current_candidate = root / "PDB_data.csv"
            if current_candidate.exists():
                return current_candidate

            dated_candidates = sorted(root.glob("PDB_data_*.csv"), reverse=True)
            for candidate in dated_candidates:
                if candidate.name == "PDB_data.csv":
                    continue
                return candidate

        return None

    @classmethod
    def get_benchmark_dir(cls):
        configured = current_app.config.get("BENCHMARK_EXPORT_DIR")
        candidate_directories = []

        if configured:
            candidate_directories.append(Path(configured))
        else:
            candidate_directories.extend(
                [
                    cls.DEFAULT_BENCHMARK_DIR,
                    Path(current_app.root_path).parent / "data" / "benchmarks",
                    Path.cwd() / "data" / "benchmarks",
                    Path("/tmp/metamp-benchmarks"),
                ]
            )

        for benchmark_dir in candidate_directories:
            try:
                benchmark_dir.mkdir(parents=True, exist_ok=True)
                return benchmark_dir
            except OSError:
                continue

        raise OSError("Unable to create a writable benchmark export directory.")


@lru_cache(maxsize=32)
def _get_reflected_table(table_name):
    return Table(table_name, db.metadata, autoload_with=db.engine, extend_existing=True)


def _read_sql_dataframe(query):
    with db.engine.connect() as connection:
        return pd.read_sql_query(query, connection)


class DashboardFilterOptionsService:
    CACHE_TTL_SECONDS = int(timedelta(days=10).total_seconds())
    cache = RedisCache()

    @classmethod
    def get_filter_options_payload(cls):
        return {
            "group": cls.get_group_options(),
            "species": cls.get_species_options(),
            "subgroup": cls.get_subgroup_options(),
            "family_name": cls.get_family_name_options(),
            "super_family": cls.get_super_family_options(),
            "membrane_name": cls.get_membrane_name_options(),
            "taxonomic_domain": cls.get_taxonomic_domain_options(),
            "molecular_function": cls.get_molecular_function_options(),
            "cellular_component": cls.get_cellular_component_options(),
            "biological_process": cls.get_biological_process_options(),
            "experimental_methods": cls.get_experimental_methods_options(),
            "super_family_class_type": cls.get_super_family_class_type_options(),
        }

    @classmethod
    def get_group_options(cls):
        return cls._get_simple_options("unique_group", "membrane_proteins", "group")

    @classmethod
    def get_subgroup_options(cls):
        return cls._get_simple_options("unique_subgroup", "membrane_proteins", "subgroup")

    @classmethod
    def get_taxonomic_domain_options(cls):
        return cls._get_simple_options(
            "unique_taxonomic_domain",
            "membrane_proteins",
            "taxonomic_domain",
        )

    @classmethod
    def get_experimental_methods_options(cls):
        return cls._get_simple_options(
            "unique_experimental_methods",
            "membrane_proteins",
            "rcsentinfo_experimental_method",
        )

    @classmethod
    def get_family_name_options(cls):
        return cls._get_simple_options(
            "unique_family_names",
            "membrane_protein_opm",
            "family_name_cache",
        )

    @classmethod
    def get_species_options(cls):
        return cls._get_simple_options(
            "unique_species",
            "membrane_protein_opm",
            "species_name_cache",
        )

    @classmethod
    def get_membrane_name_options(cls):
        return cls._get_simple_options(
            "unique_membrane_name",
            "membrane_protein_opm",
            "membrane_name_cache",
        )

    @classmethod
    def get_super_family_options(cls):
        return cls._get_simple_options(
            "unique_super_family_names",
            "membrane_protein_opm",
            "family_superfamily_name",
        )

    @classmethod
    def get_super_family_class_type_options(cls):
        return cls._get_simple_options(
            "unique_family_superfamily_classtype_names",
            "membrane_protein_opm",
            "family_superfamily_classtype_name",
        )

    @classmethod
    def get_molecular_function_options(cls):
        return cls._get_split_options(
            "unique_molecular_function",
            "membrane_protein_uniprot",
            "molecular_function",
        )

    @classmethod
    def get_cellular_component_options(cls):
        return cls._get_split_options(
            "unique_cellular_component",
            "membrane_protein_uniprot",
            "cellular_component",
        )

    @classmethod
    def get_biological_process_options(cls):
        return cls._get_split_options(
            "unique_biological_process",
            "membrane_protein_uniprot",
            "biological_process",
        )

    @classmethod
    def _get_simple_options(cls, cache_key, table_name, column_name):
        cached_data = cls.cache.get_item(cache_key)
        if cached_data is not None:
            return cached_data

        option_lists_frame = get_table_as_dataframe_with_specific_columns(
            table_name, [column_name]
        )
        option_lists = cls._normalize_options(
            option_lists_frame[column_name].dropna().tolist()
        )
        cls.cache.set_item(cache_key, option_lists, ttl=cls.CACHE_TTL_SECONDS)
        return option_lists

    @classmethod
    def _get_split_options(cls, cache_key, table_name, column_name):
        cached_data = cls.cache.get_item(cache_key)
        if cached_data is not None:
            return cached_data

        option_lists_frame = get_table_as_dataframe_with_specific_columns(
            table_name, [column_name]
        )
        flattened_options = []
        for raw_value in option_lists_frame[column_name].dropna().tolist():
            flattened_options.extend(
                item.strip()
                for item in str(raw_value).split(";")
                if item and item.strip()
            )

        option_lists = cls._normalize_options(flattened_options)
        cls.cache.set_item(cache_key, option_lists, ttl=cls.CACHE_TTL_SECONDS)
        return option_lists

    @staticmethod
    def _normalize_options(values):
        unique_values = []
        seen = set()
        for value in values:
            normalized = str(value).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_values.append(normalized)
        return ["All"] + unique_values


class DashboardAnnotationDatasetService:
    LARGE_TOPOLOGY_REGION_THRESHOLD = 80
    NORMALIZED_METHOD_FIELD_MAP = {
        "TMbed": ("TMbed_tm_count", "TMbed_tm_regions"),
        "DeepTMHMM": ("DeepTMHMM_tm_count", "DeepTMHMM_tm_regions"),
        "Phobius": ("Phobius_tm_count", "Phobius_tm_regions"),
        "TOPCONS": ("TOPCONS_tm_count", "TOPCONS_tm_regions"),
        "CCTOP": ("CCTOP_tm_count", "CCTOP_tm_regions"),
    }
    CACHE_TTL_SECONDS = int(timedelta(hours=6).total_seconds())
    cache = RedisCache()
    _process_dataframe_cache = {}
    _process_payload_cache = {}
    _record_payload_cache = {}
    FIELD_SOURCE_MAP = {
        "PDB Code": ("expert_annotation_dataset", "expert_review"),
        "Group (OPM)": ("opm", "imported"),
        "Group (MPstruc)": ("mpstruc", "imported"),
        "Group (Predicted)": ("metamp_ml_classifier", "predicted"),
        "Group (Expert)": ("expert_annotation_dataset", "expert_review"),
        "TM (Expert)": ("expert_annotation_dataset", "expert_review"),
        "pdb_code": ("mpstruc", "imported"),
        "name": ("mpstruc", "imported"),
        "group": ("mpstruc", "imported"),
        "subgroup": ("mpstruc", "imported"),
        "species": ("mpstruc", "imported"),
        "taxonomic_domain": ("mpstruc", "imported"),
        "description": ("mpstruc", "imported"),
        "related_pdb_entries": ("mpstruc", "imported"),
        "is_replaced": ("pdb", "derived"),
        "legacy_pdb_code": ("pdb", "derived"),
        "replacement_pdb_code": ("pdb", "derived"),
        "canonical_pdb_code": ("pdb", "derived"),
        "uniprot_id": ("uniprot", "imported"),
        "struct_title": ("pdb", "imported"),
        "rcsentinfo_experimental_method": ("pdb", "imported"),
        "rcsaccinfo_initial_release_date": ("pdb", "imported"),
        "created_at": ("metamp", "system"),
        "updated_at": ("metamp", "system"),
        "protein_recommended_name": ("uniprot", "imported"),
        "protein_alternative_name": ("uniprot", "imported"),
        "associated_genes": ("uniprot", "imported"),
        "annotation_score": ("uniprot", "imported"),
        "comment_disease_name": ("uniprot", "imported"),
        "comment_disease": ("uniprot", "imported"),
        "molecular_function": ("uniprot", "imported"),
        "cellular_component": ("uniprot", "imported"),
        "biological_process": ("uniprot", "imported"),
        "family_name_cache": ("opm", "imported"),
        "family_superfamily_name": ("opm", "imported"),
        "membrane_name_cache": ("opm", "imported"),
        "membrane_topology_in": ("opm", "imported"),
        "membrane_topology_out": ("opm", "imported"),
        "subunit_segments": ("opm", "imported"),
        "thickness": ("opm", "imported"),
        "tilt": ("opm", "imported"),
        "opm_tm_regions": ("opm", "imported"),
        "TMbed_tm_count": ("tmbed", "predicted"),
        "TMbed_tm_regions": ("tmbed", "predicted"),
        "DeepTMHMM_tm_count": ("deeptmhmm", "predicted"),
        "DeepTMHMM_tm_regions": ("deeptmhmm", "predicted"),
        "Phobius_tm_count": ("phobius", "predicted"),
        "Phobius_tm_regions": ("phobius", "predicted"),
        "TOPCONS_tm_count": ("topcons", "predicted"),
        "TOPCONS_tm_regions": ("topcons", "predicted"),
        "CCTOP_tm_count": ("cctop", "predicted"),
        "CCTOP_tm_regions": ("cctop", "predicted"),
    }
    PROVENANCE_FIELD_GROUPS = {
        "identity": [
            "pdb_code",
            "legacy_pdb_code",
            "replacement_pdb_code",
            "canonical_pdb_code",
            "uniprot_id",
            "name",
            "protein_recommended_name",
            "protein_alternative_name",
            "associated_genes",
        ],
        "classification": [
            "group",
            "subgroup",
            "Group (OPM)",
            "Group (MPstruc)",
            "Group (Predicted)",
            "Group (Expert)",
        ],
        "topology": [
            "TM (Expert)",
            "opm_tm_regions",
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
            "thickness",
            "tilt",
            "subunit_segments",
            "membrane_topology_in",
            "membrane_topology_out",
        ],
        "biology": [
            "species",
            "taxonomic_domain",
            "family_name_cache",
            "family_superfamily_name",
            "membrane_name_cache",
            "comment_disease_name",
            "comment_disease",
            "molecular_function",
            "cellular_component",
            "biological_process",
        ],
    }

    @classmethod
    def get_records(cls):
        dataset_path = DashboardConfigurationService.get_annotation_dataset_path()
        cache_key = cls._cache_key("records_payload", dataset_path)
        cached = cls._process_payload_cache.get(cache_key)
        if cached is not None:
            return cached

        dataset = cls._load_enriched_dataset()
        records = [cls._normalize_record_metadata(record) for record in dataset.to_dict(orient="records")]
        records = cls._attach_normalized_tm_prediction_payloads(records)
        payload = cls._to_json_safe(records)
        cls._process_payload_cache = {cache_key: payload}
        return payload

    @classmethod
    def get_record(cls, pdb_code):
        normalized_code = cls._normalize_lookup_value(pdb_code)
        cached = cls._record_payload_cache.get(normalized_code)
        if cached is not None:
            return cached

        try:
            dataset = cls._load_enriched_dataset()
            annotation_record = cls._find_annotation_record(dataset, normalized_code)
        except Exception as exc:
            logger.warning("Unable to load enriched annotation dataset for record lookup %s: %s", normalized_code, exc)
            annotation_record = None

        try:
            merged_database_record = cls._find_merged_database_record(normalized_code)
        except Exception as exc:
            logger.warning("Unable to query merged database record for %s: %s", normalized_code, exc)
            merged_database_record = None

        try:
            live_database_record = cls._find_live_membrane_record(normalized_code)
        except Exception as exc:
            logger.warning("Unable to query live membrane record for %s: %s", normalized_code, exc)
            live_database_record = None

        local_dataset_record = cls._find_local_dataset_record(normalized_code)

        if (
            annotation_record is None
            and merged_database_record is None
            and live_database_record is None
            and local_dataset_record is None
        ):
            return None

        base_record = merged_database_record or local_dataset_record

        if annotation_record is None:
            record = dict(base_record)
        elif base_record is None:
            record = dict(annotation_record)
        else:
            record = dict(base_record)
            record.update(annotation_record)

        if live_database_record:
            record = cls._fill_missing_record_fields(record, live_database_record)
        if local_dataset_record:
            record = cls._fill_missing_record_fields(record, local_dataset_record)

        record = cls._normalize_record_metadata(record)
        record = cls._attach_normalized_tm_prediction_payload(record)
        record["provenance"] = cls._build_record_provenance(record)
        record["scientific_assessment"] = build_scientific_assessment(record)
        record["external_links"] = cls._build_external_links(record)
        record["annotation_lineage"] = cls._build_annotation_lineage(record)
        record["record_resolution"] = cls._build_record_resolution(record)
        record["discrepancy_review"] = DiscrepancyReviewService.get_review_payload_for_record(record)
        record["benchmark_decision"] = (
            DiscrepancyReviewService._build_candidate_payload(record).get("benchmark_decision")
        )
        record["ui_sections"] = cls._build_ui_sections(record)
        record["field_glossary_keys"] = DashboardFieldMetadataService.record_field_glossary_keys()
        payload = cls._to_json_safe(record)
        cls._remember_record_payload(normalized_code, payload)
        return payload

    @classmethod
    def _attach_normalized_tm_prediction_payloads(cls, records):
        prepared_records = [dict(record or {}) for record in records or []]
        pdb_codes = []
        for record in prepared_records:
            normalized_code = cls._resolve_record_lookup_code(record)
            if normalized_code:
                pdb_codes.append(normalized_code)
        summaries_by_code = get_normalized_tm_prediction_summaries_for_pdb_codes(pdb_codes)
        return [
            cls._attach_normalized_tm_prediction_payload(
                record,
                summaries_by_code=summaries_by_code,
            )
            for record in prepared_records
        ]

    @classmethod
    def _attach_normalized_tm_prediction_payload(cls, record, summaries_by_code=None):
        prepared_record = dict(record or {})
        normalized_code = cls._resolve_record_lookup_code(prepared_record)
        structure_context = prepared_record.get("structure_context") or {}
        if not structure_context:
            try:
                structure_context = cls._build_structure_context_from_row(prepared_record) or {}
            except Exception:
                structure_context = {}
        prepared_record["structure_context"] = structure_context or None

        if summaries_by_code is None:
            normalized_summaries = get_normalized_tm_prediction_summaries(normalized_code)
        else:
            normalized_summaries = summaries_by_code.get(normalized_code, [])
        prepared_record["normalized_tm_predictions"] = cls._enrich_normalized_tm_prediction_summaries(
            normalized_summaries,
            structure_context,
        )
        prepared_record["tmalphafold_predictions"] = [
            item
            for item in prepared_record["normalized_tm_predictions"]
            if item.get("provider") == "TMAlphaFold"
        ]
        prepared_record["tm_prediction_overview"] = cls._build_tm_prediction_overview(
            prepared_record["normalized_tm_predictions"]
        )
        prepared_record["tm_prediction_summary_card"] = cls._build_tm_prediction_summary_card(
            prepared_record["normalized_tm_predictions"]
        )
        return cls._apply_normalized_tm_prediction_fields(prepared_record)

    @classmethod
    def _apply_normalized_tm_prediction_fields(cls, record):
        normalized_predictions = record.get("normalized_tm_predictions") or []
        preferred_predictions_by_method = {}
        provider_priority = {"MetaMP": 0, "TMAlphaFold": 1}
        for item in normalized_predictions:
            method = item.get("method")
            if not method or item.get("ambiguous"):
                continue
            existing = preferred_predictions_by_method.get(method)
            if existing is None:
                preferred_predictions_by_method[method] = item
                continue
            current_priority = provider_priority.get(item.get("provider"), 99)
            existing_priority = provider_priority.get(existing.get("provider"), 99)
            if current_priority < existing_priority:
                preferred_predictions_by_method[method] = item
        summary_card = record.get("tm_prediction_summary_card") or cls._build_tm_prediction_summary_card(
            normalized_predictions
        )
        record["tm_prediction_summary_card"] = summary_card
        record["preferred_tm_prediction_provider"] = summary_card.get("preferred_provider")
        record["preferred_tm_prediction_method"] = summary_card.get("preferred_method")
        record["preferred_tm_prediction_kind"] = summary_card.get("preferred_prediction_kind")
        record["preferred_tm_prediction_count"] = summary_card.get("preferred_tm_count")
        record["preferred_tm_topology_label"] = summary_card.get("preferred_topology_label")
        record["preferred_tm_compact_label"] = summary_card.get("preferred_compact_label")
        record["preferred_tm_orientation_label"] = summary_card.get("preferred_orientation_label")
        record["preferred_tm_chain_summary"] = summary_card.get("preferred_chain_summary")
        record["tm_prediction_method_count"] = summary_card.get("method_count")
        record["tm_prediction_available_methods"] = summary_card.get("available_methods") or []
        record["tm_prediction_has_structure_evidence"] = bool(
            summary_card.get("has_structure_based_evidence")
        )
        record["tm_prediction_has_signal_peptide_evidence"] = bool(
            summary_card.get("has_signal_peptide_evidence")
        )
        record["tm_prediction_has_ambiguous_results"] = bool(
            summary_card.get("has_ambiguous_results")
        )
        for method, (count_field, regions_field) in cls.NORMALIZED_METHOD_FIELD_MAP.items():
            summary = preferred_predictions_by_method.get(method)
            if not summary:
                continue
            record[count_field] = summary.get("tm_count")
            record[regions_field] = summary.get("tm_regions_json")
        return record

    @classmethod
    def _enrich_normalized_tm_prediction_summaries(cls, summaries, structure_context=None):
        enriched = []
        for summary in summaries or []:
            enriched_summary = dict(summary or {})
            derived_topology = cls._build_tm_prediction_topology_summary(
                enriched_summary,
                structure_context=structure_context or {},
            )
            enriched_summary["derived_topology"] = derived_topology
            if (
                enriched_summary.get("tm_count") is None
                and derived_topology.get("available")
                and derived_topology.get("tm_count") is not None
            ):
                enriched_summary["tm_count"] = derived_topology.get("tm_count")
            enriched.append(enriched_summary)
        return enriched

    @classmethod
    def _build_tm_prediction_overview(cls, normalized_predictions):
        normalized_predictions = normalized_predictions or []
        available_methods = []
        topology_labels = []
        for item in normalized_predictions:
            method = str(item.get("method") or "").strip()
            if method:
                available_methods.append(method)
            derived = item.get("derived_topology") or {}
            topology_label = derived.get("topology_label")
            if topology_label:
                topology_labels.append(
                    {
                        "method": method,
                        "provider": item.get("provider"),
                        "topology_label": topology_label,
                        "compact_label": derived.get("compact_label"),
                    }
                )
        return {
            "available_method_count": len(available_methods),
            "available_methods": available_methods,
            "topology_labels": topology_labels,
        }

    @classmethod
    def _build_tm_prediction_summary_card(cls, normalized_predictions):
        normalized_predictions = normalized_predictions or []
        preferred_order = [
            ("TMAlphaFold", "DeepTMHMM"),
            ("MetaMP", "TMbed"),
            ("TMAlphaFold", "Topcons2"),
            ("TMAlphaFold", "Phobius"),
            ("TMAlphaFold", "TMHMM"),
            ("TMAlphaFold", "ScampiMsa"),
            ("TMAlphaFold", "Scampi"),
            ("TMAlphaFold", "Octopus"),
            ("TMAlphaFold", "Hmmtop"),
            ("TMAlphaFold", "Memsat"),
            ("TMAlphaFold", "Philius"),
            ("TMAlphaFold", "Pro"),
            ("TMAlphaFold", "Prodiv"),
            ("TMAlphaFold", "TMDET"),
            ("TMAlphaFold", "SignalP"),
        ]
        order_lookup = {
            key: index for index, key in enumerate(preferred_order)
        }

        def sort_key(item):
            return (
                order_lookup.get((item.get("provider"), item.get("method")), 999),
                str(item.get("provider") or ""),
                str(item.get("method") or ""),
            )

        sorted_predictions = sorted(normalized_predictions, key=sort_key)
        preferred = None
        for item in sorted_predictions:
            derived = item.get("derived_topology") or {}
            if derived.get("available"):
                preferred = item
                break
        if preferred is None and sorted_predictions:
            preferred = sorted_predictions[0]

        preferred_derived = (preferred or {}).get("derived_topology") or {}
        available_methods = [
            str(item.get("method") or "").strip()
            for item in sorted_predictions
            if str(item.get("method") or "").strip()
        ]
        return {
            "available": bool(sorted_predictions),
            "method_count": len(available_methods),
            "available_methods": available_methods,
            "preferred_provider": (preferred or {}).get("provider"),
            "preferred_method": (preferred or {}).get("method"),
            "preferred_prediction_kind": (preferred or {}).get("prediction_kind"),
            "preferred_tm_count": preferred_derived.get("tm_count", (preferred or {}).get("tm_count")),
            "preferred_topology_label": preferred_derived.get("topology_label"),
            "preferred_compact_label": preferred_derived.get("compact_label"),
            "preferred_topology_class": preferred_derived.get("topology_class"),
            "preferred_orientation_label": preferred_derived.get("orientation_label"),
            "preferred_chain_summary": preferred_derived.get("chain_summary"),
            "has_structure_based_evidence": any(
                item.get("prediction_kind") == "structure_membrane_plane"
                for item in sorted_predictions
            ),
            "has_signal_peptide_evidence": any(
                str(item.get("method") or "").strip() == "SignalP"
                for item in sorted_predictions
            ),
            "has_ambiguous_results": any(item.get("ambiguous") for item in sorted_predictions),
        }

    @classmethod
    def _fill_missing_record_fields(cls, record, fallback_record):
        merged_record = dict(record or {})
        fallback_record = fallback_record or {}
        for key, value in fallback_record.items():
            if key not in merged_record or not cls._has_value(merged_record.get(key)):
                if cls._has_value(value):
                    merged_record[key] = value
        return merged_record

    @classmethod
    def _remember_record_payload(cls, normalized_code, payload, max_entries=128):
        if normalized_code in cls._record_payload_cache:
            cls._record_payload_cache.pop(normalized_code)
        cls._record_payload_cache[normalized_code] = payload
        while len(cls._record_payload_cache) > max_entries:
            oldest_key = next(iter(cls._record_payload_cache))
            cls._record_payload_cache.pop(oldest_key, None)

    @classmethod
    def _normalize_record_metadata(cls, record):
        normalized = dict(record)
        mpstruc_group = standardize_group_label(
            normalized.get("Group (MPstruc)") or normalized.get("group")
        )
        if cls._has_value(mpstruc_group):
            normalized["Group (MPstruc)"] = mpstruc_group
            if not cls._has_value(normalized.get("group")):
                normalized["group"] = mpstruc_group

        opm_group = standardize_group_label(
            normalized.get("Group (OPM)")
            or normalized.get("famsupclasstype_type_name")
            or normalized.get("family_superfamily_classtype_type_name")
            or normalized.get("family_superfamily_classtype_name")
        )
        if cls._has_value(opm_group):
            normalized["Group (OPM)"] = opm_group

        normalized.update(_extract_replacement_metadata(normalized))
        normalized["replacement_status_label"] = _replacement_status_label(
            normalized.get("is_replaced")
        )
        return normalized

    @classmethod
    def _load_enriched_dataset(cls):
        dataset_path = DashboardConfigurationService.get_annotation_dataset_path()
        cache_key = cls._enriched_cache_key(dataset_path)
        cached = cls._get_process_cached_dataframe(cache_key)
        if cached is not None:
            return cached

        dataset = cls._load_dataset()
        if dataset.empty:
            return dataset

        try:
            extra_df = cls._load_live_membrane_metadata_frame(
                dataset["PDB Code"].tolist()
            )
        except Exception as exc:
            logger.warning(
                "Falling back to local dataset membrane metadata enrichment for annotation dataset: %s",
                exc,
            )
            local_dataset = cls._load_local_merged_dataset()
            if local_dataset.empty:
                extra_df = pd.DataFrame()
            else:
                extra_fields = [
                    "pdb_code",
                    "group",
                    "subgroup",
                    "species",
                    "taxonomic_domain",
                    "rcsentinfo_experimental_method",
                    "rcsb_primary_citation_country",
                    "bibliography_year",
                    "is_replaced",
                    "opm_tm_regions",
                ]
                available = [
                    field for field in extra_fields
                    if field in local_dataset.columns
                ]
                extra_df = local_dataset[available].copy()
        merged_df = dataset.merge(
            extra_df,
            how="left",
            left_on="PDB Code",
            right_on="pdb_code",
        )
        merged_df = cls._merge_live_membrane_metadata(merged_df)
        local_dataset = cls._load_local_merged_dataset()
        if not local_dataset.empty and "pdb_code" in local_dataset.columns:
            current_fields = [
                field
                for field in [
                    "pdb_code",
                    "bibliography_year",
                    "group",
                    "Group (MPstruc)",
                    "Group (OPM)",
                    "subunit_segments",
                    "opm_tm_regions",
                    "famsupclasstype_type_name",
                    "family_superfamily_classtype_type_name",
                    "family_superfamily_classtype_name",
                    "subgroup",
                    "species",
                    "taxonomic_domain",
                    "rcsentinfo_experimental_method",
                    "rcsb_primary_citation_country",
                    "uniprot_id",
                    "is_replaced",
                    "structure_context",
                ]
                if field in local_dataset.columns
            ]
            if current_fields:
                current_df = local_dataset[current_fields].drop_duplicates(
                    subset="pdb_code"
                )
                merged_df = merged_df.merge(
                    current_df,
                    how="left",
                    left_on="PDB Code",
                    right_on="pdb_code",
                    suffixes=("", "_current"),
                )
                merged_df = cls._merge_current_record_metadata(merged_df)
        merged_df = cls._merge_live_predictions(merged_df)
        merged_df = cls._finalize_annotation_dataset(merged_df)
        merged_df = merged_df.fillna("")
        return cls._set_process_cached_dataframe(cache_key, merged_df)

    @classmethod
    def _merge_live_predictions(cls, dataframe):
        live_predictions = cls._load_live_group_predictions()
        if live_predictions.empty:
            if "Group (Predicted)" not in dataframe.columns:
                dataframe["Group (Predicted)"] = None
            return dataframe

        merged_df = dataframe.merge(
            live_predictions,
            how="left",
            left_on="PDB Code",
            right_on="pdb_code",
            suffixes=("", "_live_prediction"),
        )
        if "predicted_group" in merged_df.columns:
            merged_df["Group (Predicted)"] = merged_df["predicted_group"]
        drop_columns = [
            column
            for column in ["pdb_code_live_prediction", "pdb_code", "predicted_group", "generated_at", "model_name"]
            if column in merged_df.columns
        ]
        return merged_df.drop(columns=drop_columns, errors="ignore")

    @classmethod
    def _merge_live_membrane_metadata(cls, dataframe):
        merged_df = dataframe.copy(deep=False)
        fill_pairs = [
            ("group", "group_y"),
            ("subgroup", "subgroup_y"),
            ("species", "species_y"),
            ("taxonomic_domain", "taxonomic_domain_y"),
            ("rcsentinfo_experimental_method", "rcsentinfo_experimental_method_y"),
            ("rcsb_primary_citation_country", "rcsb_primary_citation_country_y"),
            ("is_replaced", "is_replaced_y"),
        ]
        for target_column, source_column in fill_pairs:
            if source_column not in merged_df.columns:
                continue
            if target_column in merged_df.columns:
                merged_df[target_column] = merged_df[target_column].where(
                    merged_df[target_column].notna()
                    & (merged_df[target_column].astype(str).str.strip() != ""),
                    merged_df[source_column],
                )
            else:
                merged_df[target_column] = merged_df[source_column]

        if "bibliography_year_y" in merged_df.columns:
            merged_df["bibliography_year"] = pd.to_numeric(
                merged_df.get("bibliography_year_y"), errors="coerce"
            ).combine_first(
                pd.to_numeric(merged_df.get("bibliography_year"), errors="coerce")
            )
            if "Year" in merged_df.columns:
                merged_df["Year"] = merged_df["bibliography_year"].combine_first(
                    pd.to_numeric(merged_df["Year"], errors="coerce")
                )

        drop_columns = [
            column
            for column in [
                "pdb_code_y",
                "group_y",
                "subgroup_y",
                "species_y",
                "taxonomic_domain_y",
                "rcsentinfo_experimental_method_y",
                "rcsb_primary_citation_country_y",
                "bibliography_year_y",
                "is_replaced_y",
            ]
            if column in merged_df.columns
        ]
        return merged_df.drop(columns=drop_columns, errors="ignore")

    @classmethod
    def _finalize_annotation_dataset(cls, dataframe):
        merged_df = dataframe.copy(deep=False)

        if "group" in merged_df.columns:
            merged_df["Group (MPstruc)"] = merged_df["group"].apply(standardize_group_label)

        opm_source = None
        for candidate in [
            "famsupclasstype_type_name",
            "family_superfamily_classtype_type_name",
            "family_superfamily_classtype_name",
        ]:
            if candidate in merged_df.columns:
                opm_source = candidate
                break
        if opm_source:
            merged_df["Group (OPM)"] = merged_df[opm_source].apply(standardize_group_label)

        for field in ["group", "Group (MPstruc)", "Group (OPM)", "Group (Predicted)", "Group (Expert)"]:
            if field in merged_df.columns:
                merged_df[field] = merged_df[field].apply(standardize_group_label)

        if "TM (Expert)" in merged_df.columns:
            merged_df["TM (Expert)"] = merged_df["TM (Expert)"].apply(parse_tm_count)

        return merged_df

    @classmethod
    def _merge_current_record_metadata(cls, dataframe):
        merged_df = dataframe.copy(deep=False)
        fill_pairs = [
            ("group", "group_current"),
            ("subgroup", "subgroup_current"),
            ("species", "species_current"),
            ("taxonomic_domain", "taxonomic_domain_current"),
            (
                "rcsentinfo_experimental_method",
                "rcsentinfo_experimental_method_current",
            ),
            (
                "rcsb_primary_citation_country",
                "rcsb_primary_citation_country_current",
            ),
            ("uniprot_id", "uniprot_id_current"),
            ("is_replaced", "is_replaced_current"),
            ("structure_context", "structure_context_current"),
        ]
        for target_column, current_column in fill_pairs:
            if current_column not in merged_df.columns:
                continue
            if target_column in merged_df.columns:
                merged_df[target_column] = merged_df[target_column].where(
                    merged_df[target_column].notna()
                    & (merged_df[target_column].astype(str).str.strip() != ""),
                    merged_df[current_column],
                )
            else:
                merged_df[target_column] = merged_df[current_column]

        if "bibliography_year_current" in merged_df.columns:
            merged_df["bibliography_year"] = pd.to_numeric(
                merged_df["bibliography_year_current"], errors="coerce"
            ).combine_first(
                pd.to_numeric(merged_df.get("bibliography_year"), errors="coerce")
            )
            if "Year" in merged_df.columns:
                merged_df["Year"] = merged_df["bibliography_year"].combine_first(
                    pd.to_numeric(merged_df["Year"], errors="coerce")
                )
            else:
                merged_df["Year"] = merged_df["bibliography_year"]

        drop_columns = [
            column
            for column in merged_df.columns
            if column.endswith("_current") or column == "pdb_code_current"
        ]
        return merged_df.drop(columns=drop_columns, errors="ignore")

    @classmethod
    def _load_dataset(cls):
        dataset_path = DashboardConfigurationService.get_annotation_dataset_path()
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Annotation dataset file not found at {dataset_path}"
            )
        cache_key = cls._cache_key("raw", dataset_path)
        cached = cls._get_process_cached_dataframe(cache_key)
        if cached is not None:
            return cached

        dataset = pd.read_csv(dataset_path)
        required_columns = ["PDB Code", "Group (Expert)", "TM (Expert)"]
        available_columns = [column for column in required_columns if column in dataset.columns]
        dataset = dataset[available_columns].copy()
        if "PDB Code" in dataset.columns:
            dataset["PDB Code"] = (
                dataset["PDB Code"].fillna("").astype(str).str.strip().str.upper()
            )
        if "Group (Expert)" in dataset.columns:
            dataset["Group (Expert)"] = dataset["Group (Expert)"].apply(standardize_group_label)
        if "TM (Expert)" in dataset.columns:
            dataset["TM (Expert)"] = dataset["TM (Expert)"].apply(parse_tm_count)
        return cls._set_process_cached_dataframe(cache_key, dataset)

    @classmethod
    def _load_live_group_predictions(cls):
        path = DashboardConfigurationService.get_live_group_predictions_path()
        if not path.exists():
            return pd.DataFrame(columns=["pdb_code", "predicted_group"])
        cache_key = cls._cache_key("live_group_predictions", path)
        cached = cls._get_process_cached_dataframe(cache_key)
        if cached is not None:
            return cached

        frame = pd.read_csv(path)
        expected = [column for column in ["pdb_code", "predicted_group", "generated_at", "model_name"] if column in frame.columns]
        frame = frame[expected].copy()
        if "pdb_code" in frame.columns:
            frame["pdb_code"] = frame["pdb_code"].fillna("").astype(str).str.strip().str.upper()
        if "predicted_group" in frame.columns:
            frame["predicted_group"] = frame["predicted_group"].apply(standardize_group_label)
        return cls._set_process_cached_dataframe(cache_key, frame)

    @classmethod
    def _load_live_membrane_metadata_frame(cls, pdb_codes=None):
        try:
            membrane_table = _get_reflected_table("membrane_proteins")
        except Exception:
            logger.exception("Unable to reflect membrane_proteins for dashboard enrichment.")
            return pd.DataFrame()

        desired_columns = [
            "pdb_code",
            "group",
            "subgroup",
            "species",
            "taxonomic_domain",
            "rcsentinfo_experimental_method",
            "rcsb_primary_citation_country",
            "bibliography_year",
            "is_replaced",
            "uniprot_id",
            "name",
        ]
        available_columns = [
            membrane_table.c[column]
            for column in desired_columns
            if column in membrane_table.c
        ]
        if not available_columns:
            return pd.DataFrame()

        query = select(*available_columns)
        normalized_codes = [
            cls._normalize_lookup_value(code)
            for code in (pdb_codes or [])
            if cls._normalize_lookup_value(code)
        ]
        if normalized_codes and "pdb_code" in membrane_table.c:
            query = query.where(
                func.upper(func.trim(membrane_table.c.pdb_code)).in_(normalized_codes)
            )

        live_df = _read_sql_dataframe(query)
        if live_df.empty or "pdb_code" not in live_df.columns:
            return live_df

        live_df["pdb_code"] = (
            live_df["pdb_code"].fillna("").astype(str).str.strip().str.upper()
        )
        live_df = live_df[live_df["pdb_code"] != ""]
        if live_df.empty:
            return live_df

        if "bibliography_year" in live_df.columns:
            live_df["bibliography_year"] = pd.to_numeric(
                live_df["bibliography_year"], errors="coerce"
            )

        return live_df.drop_duplicates(subset="pdb_code", keep="first").replace(
            {np.nan: None}
        )

    @classmethod
    def _live_membrane_metadata_fingerprint(cls):
        try:
            membrane_table = _get_reflected_table("membrane_proteins")
            query_columns = [func.count().label("row_count")]
            if "id" in membrane_table.c:
                query_columns.append(func.max(membrane_table.c.id).label("max_id"))
            result = db.session.execute(select(*query_columns)).first()
            row_count = int(getattr(result, "row_count", 0) or 0)
            max_id = int(getattr(result, "max_id", 0) or 0)
            return f"membrane_proteins:{row_count}:{max_id}"
        except Exception as exc:
            logger.warning(
                "Unable to fingerprint membrane_proteins for dashboard cache invalidation: %s",
                exc,
            )
            return "membrane_proteins:unavailable"

    @staticmethod
    def _cache_key(prefix, dataset_path: Path):
        try:
            modified_at = int(dataset_path.stat().st_mtime)
        except FileNotFoundError:
            modified_at = 0
        return (
            "dashboard_annotation_dataset:"
            f"{prefix}:{dataset_path.resolve()}:{modified_at}"
        )

    @classmethod
    def _enriched_cache_key(cls, dataset_path: Path):
        parts = [cls._cache_key("annotation_source", dataset_path)]
        local_path_map = {
            "quantitative": DashboardConfigurationService.get_valid_dataset_path("Quantitative_data.csv"),
            "pdb": DashboardConfigurationService.get_valid_dataset_path("PDB_data_transformed.csv"),
            "pdb_raw": DashboardConfigurationService.get_raw_pdb_dataset_path(),
            "opm": DashboardConfigurationService.get_valid_dataset_path("NEWOPM.csv"),
            "opm_raw": DashboardConfigurationService.get_raw_opm_dataset_path(),
            "tm_summary": DashboardConfigurationService.get_tm_prediction_output_path(),
            "uniprot": DashboardConfigurationService.get_valid_dataset_path("Uniprot_functions.csv"),
        }
        parts.append(cls._local_dataset_cache_key(local_path_map))
        parts.append(cls._cache_key("live_group_predictions", DashboardConfigurationService.get_live_group_predictions_path()))
        parts.append(cls._live_membrane_metadata_fingerprint())
        return "dashboard_annotation_dataset:enriched:" + "|".join(parts)

    @staticmethod
    def _normalize_lookup_value(value):
        return str(value or "").strip().upper()

    @classmethod
    def _resolve_record_lookup_code(cls, record):
        return cls._normalize_lookup_value(
            (record or {}).get("canonical_pdb_code")
            or (record or {}).get("pdb_code")
            or (record or {}).get("PDB Code")
        )

    @classmethod
    def _find_annotation_record(cls, dataset, normalized_code):
        if dataset.empty or "PDB Code" not in dataset.columns:
            return None

        normalized_series = (
            dataset["PDB Code"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        record = dataset[normalized_series == normalized_code]
        if record.empty:
            return None
        return record.to_dict(orient="records")[0]

    @staticmethod
    def _find_merged_database_record(normalized_code):
        dataset = all_merged_databases()
        if dataset.empty:
            return None

        exact_match_columns = [
            column
            for column in [
                "pdb_code",
                "uniprot_id",
                "database2_database_code",
                "database2_database_code_pdb",
                "entry_id",
                "rcsb_id",
            ]
            if column in dataset.columns
        ]

        if exact_match_columns:
            mask = pd.Series(False, index=dataset.index)
            for column in exact_match_columns:
                mask = mask | (
                    dataset[column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    == normalized_code
                )
            matches = dataset[mask]
            if not matches.empty:
                return matches.iloc[0].replace({np.nan: None}).to_dict()

        records = search_merged_databases(normalized_code, limit=1)
        if not records:
            return None
        return records[0]

    @classmethod
    def _find_local_dataset_record(cls, normalized_code):
        dataset = cls._load_local_merged_dataset()
        if dataset.empty or "pdb_code" not in dataset.columns:
            return None

        pdb_matches = dataset["pdb_code"].fillna("").astype(str).str.strip().str.upper() == normalized_code
        if "uniprot_id" in dataset.columns:
            uniprot_matches = (
                dataset["uniprot_id"].fillna("").astype(str).str.strip().str.upper() == normalized_code
            )
            record = dataset[pdb_matches | uniprot_matches]
        else:
            record = dataset[pdb_matches]

        if record.empty:
            return None

        return record.iloc[0].replace({np.nan: None}).to_dict()

    @staticmethod
    def _find_live_membrane_record(normalized_code):
        row = (
            MembraneProteinData.query.filter(
                db.func.upper(db.func.trim(MembraneProteinData.pdb_code)) == normalized_code
            ).first()
        )
        if row is None:
            return None
        return {
            column.name: getattr(row, column.name)
            for column in MembraneProteinData.__table__.columns
        }

    @classmethod
    def _load_local_merged_dataset(cls):
        path_map = {
            "quantitative": DashboardConfigurationService.get_valid_dataset_path("Quantitative_data.csv"),
            "pdb": DashboardConfigurationService.get_valid_dataset_path("PDB_data_transformed.csv"),
            "pdb_raw": DashboardConfigurationService.get_raw_pdb_dataset_path(),
            "tm_summary": DashboardConfigurationService.get_tm_prediction_output_path(),
            "uniprot": DashboardConfigurationService.get_valid_dataset_path("Uniprot_functions.csv"),
        }

        quantitative_path = path_map["quantitative"]
        if quantitative_path is None or not quantitative_path.exists():
            return pd.DataFrame()

        cache_key = cls._local_dataset_cache_key(
            path_map,
            cls._local_dataset_database_fingerprint(),
        )
        cached = cls._get_process_cached_dataframe(cache_key)
        if cached is not None:
            return cached

        quantitative = pd.read_csv(quantitative_path)
        quantitative = quantitative.rename(columns={"Pdb Code": "pdb_code"})

        merged = quantitative

        pdb_path = path_map["pdb"]
        if pdb_path and pdb_path.exists():
            pdb_df = pd.read_csv(pdb_path).rename(columns={"Pdb Code": "pdb_code"})
            merged = merged.merge(pdb_df, on="pdb_code", how="left", suffixes=("", "_pdb"))

        raw_pdb_path = path_map["pdb_raw"]
        if raw_pdb_path and raw_pdb_path.exists():
            raw_pdb_df = pd.read_csv(raw_pdb_path, low_memory=False)
            if "Pdb Code" in raw_pdb_df.columns and "pdb_code" not in raw_pdb_df.columns:
                raw_pdb_df["pdb_code"] = raw_pdb_df["Pdb Code"]
            raw_pdb_fields = [
                field
                for field in [
                    "pdb_code",
                    "rcsb_entry_container_identifiers_assembly_ids",
                    "rcsb_entry_container_identifiers_entity_ids",
                    "rcsb_entry_container_identifiers_polymer_entity_ids",
                    "rcsb_entry_container_identifiers_non_polymer_entity_ids",
                    "rcsb_entry_container_identifiers_branched_entity_ids",
                    "rcsb_entry_container_identifiers_model_ids",
                    "em_entity_assembly",
                    "em_experiment_entity_assembly_id",
                ]
                if field in raw_pdb_df.columns
            ]
            if raw_pdb_fields:
                raw_pdb_df = raw_pdb_df[raw_pdb_fields].drop_duplicates(subset="pdb_code")
                merged = merged.merge(
                    raw_pdb_df,
                    on="pdb_code",
                    how="left",
                    suffixes=("", "_pdb_raw"),
                )

        opm_df = cls._load_opm_dataset_frame()
        if not opm_df.empty:
            if "pdbid" in opm_df.columns and "pdb_code" not in opm_df.columns:
                opm_df["pdb_code"] = opm_df["pdbid"]
            opm_df = opm_df.drop_duplicates(subset="pdb_code")
            merged = merged.merge(opm_df, on="pdb_code", how="left", suffixes=("", "_opm"))

        tm_summary_path = path_map["tm_summary"]
        if tm_summary_path and tm_summary_path.exists():
            tm_summary_df = pd.read_csv(tm_summary_path)
            if "id" in tm_summary_df.columns and "pdb_code" not in tm_summary_df.columns:
                tm_summary_df["pdb_code"] = tm_summary_df["id"]
            tm_summary_fields = [
                field
                for field in [
                    "pdb_code",
                    "TMbed_tm_count",
                    "TMbed_tm_regions",
                    "DeepTMHMM_tm_count",
                    "DeepTMHMM_tm_regions",
                ]
                if field in tm_summary_df.columns
            ]
            if tm_summary_fields:
                tm_summary_df = tm_summary_df[tm_summary_fields].drop_duplicates(subset="pdb_code")
                merged = merged.merge(
                    tm_summary_df,
                    on="pdb_code",
                    how="left",
                    suffixes=("", "_tm_summary"),
                )

        if "subunits" in merged.columns:
            merged["opm_tm_regions"] = merged["subunits"].apply(cls._parse_opm_tm_regions)
        else:
            merged["opm_tm_regions"] = None

        for field in ["TMbed_tm_regions", "DeepTMHMM_tm_regions", "opm_tm_regions"]:
            if field in merged.columns:
                merged[field] = merged[field].apply(cls._normalize_tm_region_value)

        merged["structure_context"] = merged.apply(cls._build_structure_context_from_row, axis=1)

        uniprot_path = path_map["uniprot"]
        if uniprot_path and uniprot_path.exists():
            uniprot_df = pd.read_csv(uniprot_path)
            merged = merged.merge(uniprot_df, on="pdb_code", how="left", suffixes=("", "_uniprot"))

        merged = merged.replace({np.nan: None})
        return cls._set_process_cached_dataframe(cache_key, merged)

    @staticmethod
    def _local_dataset_cache_key(path_map, database_fingerprint=None):
        parts = []
        for name in sorted(path_map):
            dataset_path = path_map[name]
            if dataset_path is None:
                parts.append(f"{name}:missing")
                continue
            try:
                modified_at = int(dataset_path.stat().st_mtime)
            except FileNotFoundError:
                modified_at = 0
            parts.append(f"{name}:{dataset_path.resolve()}:{modified_at}")
        if database_fingerprint:
            parts.append(f"db:{database_fingerprint}")
        return "dashboard_annotation_dataset:local:" + "|".join(parts)

    @classmethod
    def _local_dataset_database_fingerprint(cls):
        try:
            opm_table = _get_reflected_table("membrane_protein_opm")
            result = db.session.execute(
                select(
                    func.count().label("row_count"),
                    func.max(opm_table.c.id).label("max_id"),
                )
            ).first()
            row_count = int(getattr(result, "row_count", 0) or 0)
            max_id = int(getattr(result, "max_id", 0) or 0)
            return f"opm:{row_count}:{max_id}"
        except Exception:
            logger.exception("Unable to fingerprint membrane_protein_opm for dashboard cache invalidation.")
            return "opm:unavailable"

    @classmethod
    def _load_opm_dataset_frame(cls):
        try:
            opm_df = get_table_as_dataframe("membrane_protein_opm")
            if not opm_df.empty:
                return opm_df
        except Exception:
            logger.exception("Unable to load OPM annotations from the database; falling back to file datasets.")

        fallback_candidates = [
            DashboardConfigurationService.get_raw_opm_dataset_path(),
            DashboardConfigurationService.get_valid_dataset_path("NEWOPM.csv"),
        ]
        for candidate in fallback_candidates:
            if candidate and candidate.exists():
                try:
                    return pd.read_csv(candidate, low_memory=False)
                except Exception:
                    logger.exception("Unable to load fallback OPM dataset from %s", candidate)
        return pd.DataFrame()

    @classmethod
    def _get_process_cached_dataframe(cls, cache_key):
        cached = cls._process_dataframe_cache.get(cache_key)
        if cached is None:
            return None
        return cached.copy(deep=False)

    @classmethod
    def _set_process_cached_dataframe(cls, cache_key, dataframe):
        cls._process_dataframe_cache = {cache_key: dataframe}
        return dataframe.copy(deep=False)

    @classmethod
    def _build_record_provenance(cls, record):
        return {
            "sources_present": cls._get_sources_present(record),
            "field_sources": cls._build_field_sources(record),
            "confidence": cls._build_confidence_summary(record),
            "dataset_versions": cls._build_dataset_versions(),
            "timestamps": cls._build_timestamp_summary(record),
        }

    @classmethod
    def _get_sources_present(cls, record):
        sources = []
        if cls._has_any_value(record, ["group", "subgroup", "name", "species"]):
            sources.append("mpstruc")
        if cls._has_any_value(record, ["struct_title", "rcsentinfo_experimental_method", "is_replaced"]):
            sources.append("pdb")
        if cls._has_any_value(record, ["protein_recommended_name", "associated_genes", "annotation_score"]):
            sources.append("uniprot")
        if cls._has_any_value(record, ["family_name_cache", "thickness", "tilt", "membrane_topology_in", "opm_tm_regions"]):
            sources.append("opm")
        if cls._has_any_value(record, ["TMbed_tm_count", "TMbed_tm_regions"]):
            sources.append("tmbed")
        if cls._has_any_value(record, ["DeepTMHMM_tm_count", "DeepTMHMM_tm_regions"]):
            sources.append("deeptmhmm")
        if cls._has_any_value(record, ["Phobius_tm_count", "Phobius_tm_regions"]):
            sources.append("phobius")
        if cls._has_any_value(record, ["TOPCONS_tm_count", "TOPCONS_tm_regions"]):
            sources.append("topcons")
        if cls._has_any_value(record, ["CCTOP_tm_count", "CCTOP_tm_regions"]):
            sources.append("cctop")
        if record.get("tmalphafold_predictions"):
            sources.append("tmalphafold")
        if cls._has_any_value(record, ["Group (Predicted)"]):
            sources.append("metamp_ml_classifier")
        if cls._has_any_value(record, ["Group (Expert)", "TM (Expert)"]):
            sources.append("expert_annotation_dataset")
        return sources

    @classmethod
    def _build_field_sources(cls, record):
        field_sources = {}
        for group_name, fields in cls.PROVENANCE_FIELD_GROUPS.items():
            grouped = {}
            for field in fields:
                if not cls._has_value(record.get(field)):
                    continue
                source_name, value_kind = cls.FIELD_SOURCE_MAP.get(field, ("metamp", "derived"))
                grouped[field] = {
                    "source": source_name,
                    "value_kind": value_kind,
                }
            if grouped:
                field_sources[group_name] = grouped
        return field_sources

    @classmethod
    def _build_confidence_summary(cls, record):
        annotation_score = cls._safe_float(record.get("annotation_score"))
        tmbed_count = cls._safe_int(record.get("TMbed_tm_count"))
        deeptmhmm_count = cls._safe_int(record.get("DeepTMHMM_tm_count"))
        phobius_count = cls._safe_int(record.get("Phobius_tm_count"))
        topcons_count = cls._safe_int(record.get("TOPCONS_tm_count"))
        cctop_count = cls._safe_int(record.get("CCTOP_tm_count"))
        expert_group = cls._normalize_text(record.get("Group (Expert)"))
        predicted_group = cls._normalize_text(record.get("Group (Predicted)"))
        tm_region_sources_present = [
            source_name
            for source_name, field_name in [
                ("opm", "opm_tm_regions"),
                ("TMbed", "TMbed_tm_regions"),
                ("DeepTMHMM", "DeepTMHMM_tm_regions"),
                ("Phobius", "Phobius_tm_regions"),
                ("TOPCONS", "TOPCONS_tm_regions"),
                ("CCTOP", "CCTOP_tm_regions"),
            ]
            if cls._deserialize_tm_regions(record.get(field_name))
        ]

        available_predictor_counts = [
            value
            for value in [
                tmbed_count,
                deeptmhmm_count,
                phobius_count,
                topcons_count,
                cctop_count,
            ]
            if value is not None
        ]

        return {
            "annotation_score": annotation_score,
            "annotation_score_band": cls._annotation_score_band(annotation_score),
            "expert_annotation_available": bool(expert_group or cls._has_value(record.get("TM (Expert)"))),
            "group_prediction_available": bool(predicted_group),
            "expert_vs_prediction_agreement": (
                expert_group == predicted_group if expert_group and predicted_group else None
            ),
            "predictor_consensus": (
                len(set(available_predictor_counts)) == 1
                if len(available_predictor_counts) >= 2
                else None
            ),
            "predictor_counts": {
                "TMbed_tm_count": tmbed_count,
                "DeepTMHMM_tm_count": deeptmhmm_count,
                "Phobius_tm_count": phobius_count,
                "TOPCONS_tm_count": topcons_count,
                "CCTOP_tm_count": cctop_count,
            },
            "tm_region_sources_present": tm_region_sources_present,
            "predictor_region_count_agreement": None,
            "chain_aware_context_available": bool(record.get("structure_context")),
            "replacement_status": record.get("replacement_status_label") or "Unknown",
        }

    @classmethod
    def _build_dataset_versions(cls):
        dataset_paths = {
            "annotation_dataset": cls.get_annotation_dataset_file(),
            "quantitative_dataset": DashboardConfigurationService.get_valid_dataset_path("Quantitative_data.csv"),
            "pdb_dataset": DashboardConfigurationService.get_valid_dataset_path("PDB_data_transformed.csv"),
            "opm_dataset": DashboardConfigurationService.get_valid_dataset_path("NEWOPM.csv"),
            "opm_raw_dataset": DashboardConfigurationService.get_raw_opm_dataset_path(),
            "tm_prediction_summary": DashboardConfigurationService.get_tm_prediction_output_path(),
            "uniprot_dataset": DashboardConfigurationService.get_valid_dataset_path("Uniprot_functions.csv"),
        }
        versions = {}
        for source_name, path in dataset_paths.items():
            if path is None:
                continue
            versions[source_name] = cls._build_file_metadata(path)
        return versions

    @classmethod
    def get_annotation_dataset_file(cls):
        return DashboardConfigurationService.get_annotation_dataset_path()

    @staticmethod
    def _build_file_metadata(path):
        metadata = {
            "path": str(path),
            "exists": path.exists(),
        }
        if not path.exists():
            return metadata
        stats = path.stat()
        metadata["modified_at"] = datetime.fromtimestamp(stats.st_mtime).isoformat()
        metadata["size_bytes"] = stats.st_size
        return metadata

    @classmethod
    def _build_timestamp_summary(cls, record):
        return {
            "record_created_at": cls._normalize_timestamp(record.get("created_at")),
            "record_updated_at": cls._normalize_timestamp(record.get("updated_at")),
            "uniprot_info_created": cls._normalize_timestamp(record.get("info_created")),
            "uniprot_info_modified": cls._normalize_timestamp(record.get("info_modified")),
            "uniprot_sequence_updated": cls._normalize_timestamp(record.get("info_sequence_update")),
            "pdb_initial_release_date": cls._normalize_timestamp(record.get("rcsaccinfo_initial_release_date")),
            "pdb_revision_date": cls._normalize_timestamp(record.get("rcsaccinfo_revision_date")),
        }

    @classmethod
    def _build_annotation_lineage(cls, record):
        scientific_assessment = record.get("scientific_assessment") or build_scientific_assessment(record)
        review_payload = record.get("discrepancy_review") or DiscrepancyReviewService.get_review_payload_for_record(record)
        group_labels = {
            "mpstruc": record.get("group") or cls._normalize_text(record.get("Group (MPstruc)")),
            "opm": cls._normalize_text(record.get("Group (OPM)")),
            "predicted": cls._normalize_text(record.get("Group (Predicted)")),
            "expert": cls._normalize_text(record.get("Group (Expert)")),
            "adjudicated": cls._normalize_text(review_payload.get("reviewed_group")),
        }
        active_labels = {
            key: value for key, value in group_labels.items() if value
        }
        resolved_label = (
            active_labels.get("adjudicated")
            or active_labels.get("expert")
            or active_labels.get("predicted")
            or active_labels.get("opm")
            or active_labels.get("mpstruc")
        )
        disagreements = []
        active_values = [value for value in active_labels.values() if value]
        if active_values and len(set(active_values)) > 1:
            disagreements.append("group_label_disagreement")

        return {
            "pdb_code": cls._normalize_text(
                record.get("canonical_pdb_code") or record.get("pdb_code") or record.get("PDB Code")
            ),
            "legacy_pdb_code": cls._normalize_text(record.get("legacy_pdb_code")),
            "replacement_pdb_code": cls._normalize_text(record.get("replacement_pdb_code")),
            "resolved_group_label": resolved_label,
            "labels": active_labels,
            "review_status": review_payload.get("status"),
            "reviewed_group": review_payload.get("reviewed_group"),
            "reviewed_tm_count": review_payload.get("reviewed_tm_count"),
            "disagreements": disagreements,
            "scientific_assessment": scientific_assessment,
        }

    @classmethod
    def _build_external_links(cls, record):
        pdb_code = cls._normalize_text(
            record.get("canonical_pdb_code") or record.get("pdb_code") or record.get("PDB Code")
        )
        uniprot_id = cls._normalize_text(record.get("uniprot_id"))
        links = {}
        if pdb_code:
            links["pdb"] = f"https://www.rcsb.org/structure/{pdb_code}"
            links["emdb_search"] = f"https://www.ebi.ac.uk/emdb/search/{pdb_code}"
            links["opm_search"] = f"https://opm.phar.umich.edu/search?search={pdb_code}"
        if uniprot_id:
            links["uniprot"] = f"https://www.uniprot.org/uniprotkb/{uniprot_id}"
        return links

    @classmethod
    def _build_record_resolution(cls, record):
        lineage = record.get("annotation_lineage") or cls._build_annotation_lineage(record)
        review_status = (record.get("discrepancy_review") or {}).get("status") or "open"
        labels = lineage.get("labels") or {}

        if labels.get("adjudicated"):
            selected_group_label = labels["adjudicated"]
            selected_group_source = "expert_adjudication"
        elif labels.get("expert"):
            selected_group_label = labels["expert"]
            selected_group_source = "expert_annotation_dataset"
        elif labels.get("predicted"):
            selected_group_label = labels["predicted"]
            selected_group_source = "metamp_ml_classifier"
        elif labels.get("opm"):
            selected_group_label = labels["opm"]
            selected_group_source = "opm"
        else:
            selected_group_label = labels.get("mpstruc")
            selected_group_source = "mpstruc"

        tm_counts = {
            "expert": cls._safe_int(record.get("TM (Expert)")),
            "TMbed": cls._safe_int(record.get("TMbed_tm_count")),
            "DeepTMHMM": cls._safe_int(record.get("DeepTMHMM_tm_count")),
        }
        selected_tm_count = tm_counts.get("expert")
        selected_tm_source = "expert_annotation_dataset" if selected_tm_count is not None else None
        if selected_tm_count is None:
            for source_name in ("TMbed", "DeepTMHMM"):
                if tm_counts.get(source_name) is not None:
                    selected_tm_count = tm_counts[source_name]
                    selected_tm_source = source_name.lower()
                    break

        return {
            "selected_group_label": selected_group_label,
            "selected_group_source": selected_group_source,
            "selected_tm_count": selected_tm_count,
            "selected_tm_source": selected_tm_source,
            "review_status": review_status,
            "replacement_strategy": (
                "canonical_record" if _normalize_replacement_flag(record.get("is_replaced")) else "current_record"
            ),
            "disagreement_flags": lineage.get("disagreements") or [],
        }

    @classmethod
    def _build_ui_sections(cls, record):
        scientific_assessment = record.get("scientific_assessment") or build_scientific_assessment(record)
        discrepancy_review = record.get("discrepancy_review") or {}
        structure_context = record.get("structure_context") or {}
        record_resolution = record.get("record_resolution") or cls._build_record_resolution(record)

        return {
            "overview": {
                "pdb_code": cls._normalize_text(
                    record.get("canonical_pdb_code") or record.get("pdb_code") or record.get("PDB Code")
                ),
                "legacy_pdb_code": cls._normalize_text(record.get("legacy_pdb_code")),
                "replacement_pdb_code": cls._normalize_text(record.get("replacement_pdb_code")),
                "title": cls._normalize_text(
                    record.get("struct_title")
                    or record.get("protein_recommended_name")
                    or record.get("name")
                ),
                "resolved_group": record_resolution.get("selected_group_label"),
                "resolved_group_source": record_resolution.get("selected_group_source"),
                "resolved_tm_count": record_resolution.get("selected_tm_count"),
                "resolved_tm_source": record_resolution.get("selected_tm_source"),
                "replacement_status": record.get("replacement_status_label"),
                "review_status": discrepancy_review.get("status") or "open",
            },
            "comparison": cls._build_comparison_section(record, record_resolution),
            "tm_prediction_summary": record.get("tm_prediction_summary_card") or cls._build_tm_prediction_summary_card(
                record.get("normalized_tm_predictions") or []
            ),
            "annotation_sources": cls._build_annotation_source_cards(record),
            "scientific_flags": cls._build_scientific_flag_cards(scientific_assessment),
            "structure_context": cls._build_structure_context_display(structure_context),
            "tm_boundaries": {
                "rows": cls._build_tm_boundary_rows(record),
                "source_status": cls._build_tm_source_status(record),
                "interpretations": [
                    {
                        "source": f"{item.get('provider')} {item.get('method')}",
                        "provider": item.get("provider"),
                        "method": item.get("method"),
                        "prediction_kind": item.get("prediction_kind"),
                        "derived_topology": item.get("derived_topology"),
                    }
                    for item in (record.get("normalized_tm_predictions") or [])
                    if item.get("derived_topology", {}).get("available")
                ],
            },
            "benchmark_context": cls._build_benchmark_context(scientific_assessment),
            "live_status": cls._build_live_status(record),
        }

    @classmethod
    def _build_mpstruc_group_display(cls, record):
        record = record or {}
        grp = record.get("group") if record.get("group") else record.get("Group")
        group = cls._normalize_text(grp) or cls._normalize_text(
            record.get("Group (MPstruc)")
        )
        subgroup = cls._normalize_text(record.get("subgroup"))
        if group and subgroup and subgroup.lower() not in group.lower():
            return f"{group} / {subgroup}"
        return group or subgroup

    @classmethod
    def _build_opm_group_display(cls, record):
        record = record or {}
        primary = cls._normalize_text(record.get("Group (OPM)"))
        if primary:
            return primary
        class_name = cls._normalize_text(record.get("family_superfamily_classtype_name"))
        type_name = cls._normalize_text(
            record.get("famsupclasstype_type_name")
            or record.get("family_superfamily_classtype_type_name")
        )
        if class_name and type_name and type_name.lower() not in class_name.lower():
            return f"{class_name} ({type_name})"
        return class_name or type_name

    @classmethod
    def _build_comparison_section(cls, record, record_resolution=None):
        record_resolution = record_resolution or record.get("record_resolution") or cls._build_record_resolution(record)
        lineage = record.get("annotation_lineage") or cls._build_annotation_lineage(record)
        labels = lineage.get("labels") or {}
        opm_regions = cls._deserialize_tm_regions(record.get("opm_tm_regions"))
        # print(record)
        group_rows = [
            {
                "source": "Expert",
                "value": labels.get("expert"),
                "category": "expert_reference",
            },
            {
                "source": "Predicted",
                "value": labels.get("predicted"),
                "category": "live_prediction",
            },
            {
                "source": "OPM",
                "value": cls._build_opm_group_display(record) or labels.get("opm"),
                "category": "source_database",
            },
            {
                "source": "MPstruc",
                "value": cls._build_mpstruc_group_display(record) or labels.get("mpstruc"),
                "category": "source_database",
            },
        ]

        opm_tm_count = cls._safe_int(record.get("subunit_segments"))
        if opm_tm_count is None and opm_regions:
            opm_tm_count = len(opm_regions)

        segment_rows = [
            {
                "source": "Expert",
                "value": cls._safe_int(record.get("TM (Expert)")),
                "category": "expert_reference",
            },
            {
                "source": "OPM",
                "value": opm_tm_count,
                "category": "source_database",
            },
            {
                "source": "TMbed",
                "value": cls._safe_int(record.get("TMbed_tm_count")),
                "category": "live_predictor",
            },
            {
                "source": "DeepTMHMM",
                "value": cls._safe_int(record.get("DeepTMHMM_tm_count")),
                "category": "live_predictor",
            },
            {
                "source": "Phobius",
                "value": cls._safe_int(record.get("Phobius_tm_count")),
                "category": "live_predictor",
            },
            {
                "source": "TOPCONS",
                "value": cls._safe_int(record.get("TOPCONS_tm_count")),
                "category": "live_predictor",
            },
            {
                "source": "CCTOP",
                "value": cls._safe_int(record.get("CCTOP_tm_count")),
                "category": "live_predictor",
            },
        ]
        for summary in record.get("tmalphafold_predictions") or []:
            derived_topology = summary.get("derived_topology") or {}
            derived_tm_count = cls._safe_int(derived_topology.get("tm_count"))
            summary_tm_count = cls._safe_int(summary.get("tm_count"))
            display_tm_count = derived_tm_count if derived_tm_count is not None else summary_tm_count
            if not summary.get("ambiguous") and display_tm_count is None:
                continue
            segment_rows.append(
                {
                    "source": f"TMAlphaFold {summary.get('method')}",
                    "value": (
                        display_tm_count
                        if not summary.get("ambiguous")
                        else "Ambiguous across UniProt mappings"
                    ),
                    "category": "live_predictor",
                }
            )

        return {
            "group_rows": group_rows,
            "segment_rows": segment_rows,
            "segment_note": "Predicted broad groups do not generate TM segment counts.",
        }

    @classmethod
    def _build_annotation_source_cards(cls, record):
        provenance = record.get("provenance") or cls._build_record_provenance(record)
        dataset_versions = provenance.get("dataset_versions") or {}
        live_prediction_file = cls._build_file_metadata(
            DashboardConfigurationService.get_live_group_predictions_path()
        )
        source_definitions = DashboardFieldMetadataService.source_catalog()
        cards = []
        for source_key in [
            "mpstruc",
            "opm",
            "pdb",
            "uniprot",
            "expert_annotation_dataset",
            "metamp_ml_classifier",
            "tmbed",
            "deeptmhmm",
            "phobius",
            "topcons",
            "cctop",
            "tmalphafold",
        ]:
            definition = source_definitions.get(source_key, {})
            card = {
                "key": source_key,
                "label": definition.get("label", source_key),
                "category": definition.get("category"),
                "description": definition.get("description"),
                "is_static_reference": definition.get("is_static_reference", False),
                "is_live_generated": definition.get("is_live_generated", False),
                "present_for_record": source_key in (provenance.get("sources_present") or []),
                "last_updated": None,
            }
            if source_key == "expert_annotation_dataset":
                card["last_updated"] = dataset_versions.get("annotation_dataset", {}).get("modified_at")
            elif source_key == "metamp_ml_classifier":
                card["last_updated"] = live_prediction_file.get("modified_at")
            elif source_key in {"tmbed", "deeptmhmm", "phobius", "topcons", "cctop"}:
                card["last_updated"] = dataset_versions.get("tm_prediction_summary", {}).get("modified_at")
            elif source_key == "tmalphafold":
                card["last_updated"] = None
            elif source_key == "mpstruc":
                card["last_updated"] = dataset_versions.get("quantitative_dataset", {}).get("modified_at")
            elif source_key == "opm":
                card["last_updated"] = dataset_versions.get("opm_dataset", {}).get("modified_at")
            elif source_key == "pdb":
                card["last_updated"] = dataset_versions.get("pdb_dataset", {}).get("modified_at")
            elif source_key == "uniprot":
                card["last_updated"] = dataset_versions.get("uniprot_dataset", {}).get("modified_at")
            cards.append(card)
        return cards

    @classmethod
    def _build_scientific_flag_cards(cls, scientific_assessment):
        flags = (scientific_assessment or {}).get("flags") or {}
        notes = (scientific_assessment or {}).get("notes") or []
        glossary = DashboardFieldMetadataService.scientific_flag_glossary()
        items = []
        for key in [
            "recommended_for_sequence_topology_benchmark",
            "context_dependent_topology",
            "non_canonical_membrane_case",
            "multichain_context",
            "obsolete_or_replaced",
        ]:
            descriptor = glossary.get(key, {})
            value = (
                bool((scientific_assessment or {}).get("recommended_for_sequence_topology_benchmark"))
                if key == "recommended_for_sequence_topology_benchmark"
                else bool(flags.get(key))
            )
            items.append(
                {
                    "key": key,
                    "label": descriptor.get("label", key),
                    "value": value,
                    "description": descriptor.get("description"),
                    "interpretation": descriptor.get("interpretation"),
                }
            )
        return {"items": items, "notes": notes}

    @classmethod
    def _build_structure_context_display(cls, structure_context):
        structure_context = structure_context or {}
        chain_count = structure_context.get("chain_count") or 0
        entry_assembly_count = structure_context.get("entry_assembly_count")
        warnings = []
        if chain_count > 1:
            warnings.append(
                "This entry includes multiple chains, so chain-level or assembly-level interpretation may matter."
            )
        if entry_assembly_count and entry_assembly_count > 1:
            warnings.append(
                "Multiple assemblies are present, so topology interpretation may depend on the assembly context."
            )
        return {
            "chain_count": chain_count or None,
            "chain_ids": structure_context.get("chain_ids") or [],
            "assembly_count": entry_assembly_count,
            "assembly_ids": structure_context.get("assembly_ids") or [],
            "entity_ids": structure_context.get("entity_ids") or [],
            "polymer_entity_ids": structure_context.get("polymer_entity_ids") or [],
            "polymer_composition": structure_context.get("polymer_composition"),
            "selected_polymer_entity_types": structure_context.get("selected_polymer_entity_types"),
            "warnings": warnings,
        }

    @classmethod
    def _build_tm_boundary_rows(cls, record):
        def _normalize_sequence_context(summary):
            uniprot_ids = [item for item in (summary.get("uniprot_ids") or []) if item]
            if len(uniprot_ids) == 1 and not str(uniprot_ids[0]).startswith("PDB:"):
                return f"UniProt {uniprot_ids[0]}"
            if len(uniprot_ids) > 1:
                return f"{len(uniprot_ids)} mapped accessions"
            return "Sequence"

        def _selected_prediction_summaries():
            normalized_predictions = record.get("normalized_tm_predictions") or []
            summary_card = record.get("tm_prediction_summary_card") or {}
            preferred_provider = summary_card.get("preferred_provider")
            preferred_method = summary_card.get("preferred_method")

            selected = []
            seen = set()

            def add_summary(summary):
                if not summary or summary.get("ambiguous"):
                    return
                key = (
                    summary.get("provider"),
                    summary.get("method"),
                    summary.get("prediction_kind"),
                )
                if key in seen:
                    return
                seen.add(key)
                selected.append(summary)

            preferred_summary = next(
                (
                    item for item in normalized_predictions
                    if item.get("provider") == preferred_provider
                    and item.get("method") == preferred_method
                ),
                None,
            )
            add_summary(preferred_summary)

            add_summary(
                next(
                    (
                        item for item in normalized_predictions
                        if item.get("provider") == "MetaMP" and item.get("method") == "TMbed"
                    ),
                    None,
                )
            )

            add_summary(
                next(
                    (
                        item for item in normalized_predictions
                        if item.get("provider") == "TMAlphaFold" and item.get("method") == "TMDET"
                    ),
                    None,
                )
            )

            return selected

        rows = []
        for region in cls._deserialize_tm_regions(record.get("opm_tm_regions")):
            normalized_region = dict(region)
            normalized_region["source"] = "OPM reference"
            normalized_region["chain_label"] = (
                f"Chain {region.get('chain')}" if region.get("chain") else "Reference chain"
            )
            normalized_region["topology"] = f"TM{normalized_region.get('index') or '?'}"
            rows.append(normalized_region)

        for summary in _selected_prediction_summaries():
            regions = cls._get_tmalphafold_tm_regions(summary)
            sequence_context = _normalize_sequence_context(summary)
            source_label = (
                "Preferred predictor"
                if (
                    summary.get("provider") == (record.get("tm_prediction_summary_card") or {}).get("preferred_provider")
                    and summary.get("method") == (record.get("tm_prediction_summary_card") or {}).get("preferred_method")
                )
                else f"{summary.get('provider')} {summary.get('method')}"
            )
            for region in regions:
                normalized_region = dict(region)
                normalized_region["source"] = source_label
                normalized_region["chain_label"] = (
                    f"Chain {region.get('chain')}" if region.get("chain") else sequence_context
                )
                normalized_region["topology"] = (
                    f"TM{normalized_region.get('index') or '?'}"
                )
                normalized_region["source_method"] = summary.get("method")
                rows.append(normalized_region)
        rows.sort(
            key=lambda item: (
                str(item.get("source") or ""),
                int(item.get("index") or 0),
                int(item.get("start") or 0),
            )
        )
        return rows

    @classmethod
    def _build_tm_source_status(cls, record):
        items = []
        opm_regions = cls._deserialize_tm_regions(record.get("opm_tm_regions"))
        items.append(
            {
                "source": "OPM reference",
                "category": "reference",
                "tm_count": len(opm_regions) if opm_regions else None,
                "boundary_count": len(opm_regions),
                "available": bool(opm_regions),
            }
        )
        for summary in record.get("normalized_tm_predictions") or []:
            all_regions = cls._deserialize_tm_regions(summary.get("tm_regions_json"))
            tm_regions = cls._get_tmalphafold_tm_regions(summary)
            items.append(
                {
                    "source": f"{summary.get('provider')} {summary.get('method')}",
                    "category": (
                        "live_predictor"
                        if summary.get("prediction_kind") == "sequence_topology"
                        else "auxiliary_predictor"
                    ),
                    "tm_count": None if summary.get("ambiguous") else summary.get("tm_count"),
                    "boundary_count": len(tm_regions),
                    "available": bool(all_regions) or (summary.get("tm_count") is not None),
                }
            )
        return items

    @classmethod
    def _build_benchmark_context(cls, scientific_assessment):
        scientific_assessment = scientific_assessment or {}
        return {
            "recommended_for_sequence_topology_benchmark": scientific_assessment.get(
                "recommended_for_sequence_topology_benchmark"
            ),
            "benchmark_exclusion_reasons": scientific_assessment.get("benchmark_exclusion_reasons") or [],
            "cautionary_note": (
                "Sequence-based TM predictors are shown as assistive comparison signals and do not replace structural interpretation."
            ),
        }

    @classmethod
    def _build_live_status(cls, record):
        provenance = record.get("provenance") or cls._build_record_provenance(record)
        dataset_versions = provenance.get("dataset_versions") or {}
        live_prediction_file = cls._build_file_metadata(
            DashboardConfigurationService.get_live_group_predictions_path()
        )
        return {
            "annotation_dataset_modified_at": dataset_versions.get("annotation_dataset", {}).get("modified_at"),
            "quantitative_dataset_modified_at": dataset_versions.get("quantitative_dataset", {}).get("modified_at"),
            "pdb_dataset_modified_at": dataset_versions.get("pdb_dataset", {}).get("modified_at"),
            "opm_dataset_modified_at": dataset_versions.get("opm_dataset", {}).get("modified_at"),
            "uniprot_dataset_modified_at": dataset_versions.get("uniprot_dataset", {}).get("modified_at"),
            "tm_prediction_summary_modified_at": dataset_versions.get("tm_prediction_summary", {}).get("modified_at"),
            "live_group_predictions_modified_at": live_prediction_file.get("modified_at"),
        }

    @staticmethod
    def _normalize_timestamp(value):
        normalized = str(value or "").strip()
        return normalized or None

    @staticmethod
    def _has_value(value):
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, dict, tuple, set)):
            return len(value) > 0
        try:
            return not pd.isna(value)
        except TypeError:
            return True

    @classmethod
    def _has_any_value(cls, record, keys):
        return any(cls._has_value(record.get(key)) for key in keys)

    @staticmethod
    def _safe_float(value):
        try:
            if not DashboardAnnotationDatasetService._has_value(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value):
        if not DashboardAnnotationDatasetService._has_value(value):
            return None
        return parse_tm_count(value)

    @classmethod
    def _normalize_tm_region_value(cls, value):
        if not cls._has_value(value):
            return None
        if isinstance(value, list):
            return cls._serialize_tm_regions(value)
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            try:
                loaded = json.loads(normalized)
            except json.JSONDecodeError:
                return normalized
            if isinstance(loaded, list):
                return cls._serialize_tm_regions(loaded)
            return normalized
        return None

    @classmethod
    def _serialize_tm_regions(cls, regions):
        normalized = cls._normalize_tm_regions(regions)
        if not normalized:
            return None
        return json.dumps(normalized)

    @staticmethod
    def _parse_literal_payload(value, default=None):
        if value is None:
            return [] if default is None else default
        if isinstance(value, (list, dict)):
            return value
        normalized = str(value).strip()
        if not normalized:
            return [] if default is None else default
        try:
            return ast.literal_eval(normalized)
        except (ValueError, SyntaxError):
            return [] if default is None else default

    @classmethod
    def _parse_literal_string_list(cls, value):
        parsed = cls._parse_literal_payload(value, default=[])
        if not isinstance(parsed, list):
            return []
        result = []
        for item in parsed:
            normalized = cls._normalize_text(item)
            if normalized:
                result.append(normalized)
        return result

    @classmethod
    def _build_structure_context_from_row(cls, row):
        row_data = row.to_dict() if hasattr(row, "to_dict") else dict(row)

        chain_ids = []
        subunits = cls._parse_literal_payload(row_data.get("subunits"), default=[])
        if isinstance(subunits, list):
            for subunit in subunits:
                if not isinstance(subunit, dict):
                    continue
                chain_id = cls._normalize_text(subunit.get("protein_letter"))
                if chain_id:
                    chain_ids.append(chain_id)
        chain_ids = list(OrderedDict.fromkeys(chain_ids))

        entity_ids = cls._parse_literal_string_list(
            row_data.get("rcsb_entry_container_identifiers_entity_ids")
        )
        polymer_entity_ids = cls._parse_literal_string_list(
            row_data.get("rcsb_entry_container_identifiers_polymer_entity_ids")
        )
        non_polymer_entity_ids = cls._parse_literal_string_list(
            row_data.get("rcsb_entry_container_identifiers_non_polymer_entity_ids")
        )
        branched_entity_ids = cls._parse_literal_string_list(
            row_data.get("rcsb_entry_container_identifiers_branched_entity_ids")
        )
        assembly_ids = cls._parse_literal_string_list(
            row_data.get("rcsb_entry_container_identifiers_assembly_ids")
        )
        model_ids = cls._parse_literal_string_list(
            row_data.get("rcsb_entry_container_identifiers_model_ids")
        )

        em_entity_assemblies_raw = cls._parse_literal_payload(
            row_data.get("em_entity_assembly"),
            default=[],
        )
        entity_assemblies = []
        if isinstance(em_entity_assemblies_raw, list):
            for assembly in em_entity_assemblies_raw:
                if not isinstance(assembly, dict):
                    continue
                entity_assemblies.append(
                    {
                        "id": cls._normalize_text(assembly.get("id")),
                        "name": cls._normalize_text(assembly.get("name")),
                        "type": cls._normalize_text(assembly.get("type")),
                        "source": cls._normalize_text(assembly.get("source")),
                        "entity_id_list": cls._parse_literal_string_list(
                            assembly.get("entity_id_list")
                        ),
                    }
                )

        opm_uniprot_codes = cls._parse_literal_string_list(row_data.get("uniprotcodes"))

        structure_context = {
            "chain_ids": chain_ids,
            "chain_count": len(chain_ids) or None,
            "assembly_ids": assembly_ids,
            "entity_ids": entity_ids,
            "polymer_entity_ids": polymer_entity_ids,
            "non_polymer_entity_ids": non_polymer_entity_ids,
            "branched_entity_ids": branched_entity_ids,
            "model_ids": model_ids,
            "entity_assemblies": entity_assemblies,
            "opm_uniprot_codes": opm_uniprot_codes,
            "entry_assembly_count": cls._safe_int(
                row_data.get("rcsb_entry_info_assembly_count")
                or row_data.get("rcsb_entry_info_assembly_count_pdb")
            ),
            "entry_entity_count": cls._safe_int(
                row_data.get("rcsb_entry_info_entity_count")
                or row_data.get("rcsb_entry_info_entity_count_pdb")
            ),
            "polymer_entity_count": cls._safe_int(
                row_data.get("rcsb_entry_info_polymer_entity_count")
                or row_data.get("rcsb_entry_info_polymer_entity_count_pdb")
            ),
            "polymer_entity_count_protein": cls._safe_int(
                row_data.get("rcsb_entry_info_polymer_entity_count_protein")
                or row_data.get("rcsb_entry_info_polymer_entity_count_protein_pdb")
            ),
            "selected_polymer_entity_types": cls._normalize_text(
                row_data.get("rcsb_entry_info_selected_polymer_entity_types")
                or row_data.get("rcsb_entry_info_selected_polymer_entity_types_pdb")
            ),
            "polymer_composition": cls._normalize_text(
                row_data.get("rcsb_entry_info_polymer_composition")
                or row_data.get("rcsb_entry_info_polymer_composition_pdb")
            ),
            "point_symmetry": cls._normalize_text(
                row_data.get("em_single_particle_entity_point_symmetry")
                or row_data.get("em_single_particle_entity_point_symmetry_pdb")
            ),
        }

        has_any_value = any(
            value not in (None, "", [], {})
            for value in structure_context.values()
        )
        return structure_context if has_any_value else None

    @classmethod
    def _normalize_tm_regions(cls, regions):
        normalized = []
        for index, region in enumerate(regions or [], start=1):
            if not isinstance(region, dict):
                continue
            try:
                start = int(region.get("start"))
                end = int(region.get("end"))
            except (TypeError, ValueError):
                continue
            if end < start:
                start, end = end, start
            normalized_region = {
                "index": int(region.get("index") or index),
                "start": start,
                "end": end,
                "length": end - start + 1,
            }
            for optional_key in ("label", "chain", "attributes"):
                optional_value = region.get(optional_key)
                if optional_value not in (None, "", {}):
                    normalized_region[optional_key] = optional_value
            normalized.append(normalized_region)
        return normalized

    @classmethod
    def _deserialize_tm_regions(cls, value):
        if not cls._has_value(value):
            return []
        if isinstance(value, list):
            return cls._normalize_tm_regions(value)
        normalized = str(value).strip()
        if not normalized:
            return []
        try:
            loaded = json.loads(normalized)
        except json.JSONDecodeError:
            return []
        if not isinstance(loaded, list):
            return []
        return cls._normalize_tm_regions(loaded)

    @classmethod
    def _get_tmalphafold_tm_regions(cls, summary):
        regions = cls._deserialize_tm_regions((summary or {}).get("tm_regions_json"))
        if (summary or {}).get("prediction_kind") != "sequence_topology":
            return regions
        filtered = []
        for region in regions:
            if cls._is_membrane_region_label(region.get("label")):
                filtered.append(region)
        return filtered

    @classmethod
    def _classify_topology_region_label(cls, label):
        normalized = str(label or "").strip().lower()
        if cls._is_membrane_region_label(normalized):
            return "membrane"
        if cls._is_inside_region_label(normalized):
            return "inside"
        if cls._is_outside_region_label(normalized):
            return "outside"
        if cls._is_signal_region_label(normalized):
            return "signal"
        return "other"

    @classmethod
    def _is_membrane_region_label(cls, label):
        normalized = str(label or "").strip().lower()
        if not normalized:
            return False
        if normalized in TMALPHAFOLD_MEMBRANE_LABELS:
            return True
        if normalized in {"h", "b", "m", "membrane", "tm", "tm helix", "tm_helix", "tm barrel", "tm_barrel"}:
            return True
        if "membrane" in normalized:
            return True
        if normalized.startswith("tmdet membrane"):
            return True
        if normalized.startswith("tm") and "signal" not in normalized:
            return True
        return False

    @staticmethod
    def _is_inside_region_label(label):
        normalized = str(label or "").strip().lower()
        return normalized in {"inside", "i", "cytoplasmic", "intracellular"} or "inside" in normalized

    @staticmethod
    def _is_outside_region_label(label):
        normalized = str(label or "").strip().lower()
        return normalized in {"outside", "o", "extracellular", "periplasmic", "luminal", "lumen"} or "outside" in normalized

    @staticmethod
    def _is_signal_region_label(label):
        normalized = str(label or "").strip().lower()
        return normalized in {"signal", "signal peptide", "signal_peptide"} or "signal" in normalized

    @staticmethod
    def _orientation_side_label(side):
        if side == "inside":
            return "in"
        if side == "outside":
            return "out"
        return None

    @staticmethod
    def _orientation_phrase(side):
        if side == "inside":
            return "inside-facing"
        if side == "outside":
            return "outside-facing"
        return None

    @staticmethod
    def _loop_side_phrase(side):
        if side == "inside":
            return "intracellular"
        if side == "outside":
            return "extracellular"
        return None

    @classmethod
    def _build_tm_prediction_topology_summary(cls, summary, structure_context=None):
        summary = summary or {}
        structure_context = structure_context or {}

        if summary.get("ambiguous"):
            return {
                "available": False,
                "reason": summary.get("note") or "Multiple UniProt-backed results disagree for this entry.",
            }

        tm_regions = cls._get_tmalphafold_tm_regions(summary)
        all_regions = cls._deserialize_tm_regions(summary.get("tm_regions_json"))
        tm_count = cls._safe_int(summary.get("tm_count"))
        if tm_count is None:
            tm_count = len(tm_regions) if tm_regions else None

        if not all_regions and tm_count is None:
            return {
                "available": False,
                "reason": "No topology regions are available for this predictor.",
            }

        classified_regions = []
        tm_index = 0
        for region in all_regions:
            kind = cls._classify_topology_region_label(region.get("label"))
            normalized_region = dict(region)
            normalized_region["kind"] = kind
            if kind == "membrane":
                tm_index += 1
                normalized_region["tm_index"] = tm_index
                normalized_region["display_label"] = f"TM{tm_index}"
            elif kind == "inside":
                normalized_region["display_label"] = "Inside"
            elif kind == "outside":
                normalized_region["display_label"] = "Outside"
            elif kind == "signal":
                normalized_region["display_label"] = "Signal peptide"
            else:
                normalized_region["display_label"] = str(region.get("label") or "Region")
            classified_regions.append(normalized_region)

        first_non_membrane = next(
            (item for item in classified_regions if item.get("kind") in {"inside", "outside"}),
            None,
        )
        last_non_membrane = next(
            (
                item
                for item in reversed(classified_regions)
                if item.get("kind") in {"inside", "outside"}
            ),
            None,
        )
        n_terminal_side = first_non_membrane.get("kind") if first_non_membrane else None
        c_terminal_side = last_non_membrane.get("kind") if last_non_membrane else None

        n_orientation = cls._orientation_side_label(n_terminal_side)
        c_orientation = cls._orientation_side_label(c_terminal_side)
        orientation_label = None
        if n_orientation or c_orientation:
            orientation_label = f"N-{n_orientation or '?'} / C-{c_orientation or '?'}"

        topology_class = None
        broad_type = None
        compact_label = None
        chain_summary = None
        if tm_count and tm_count > 1:
            topology_class = "multipass alpha-helical"
            broad_type = "alpha-helical transmembrane protein"
            compact_label = f"{tm_count}TM alpha-helical membrane chain"
        elif tm_count == 1:
            topology_class = "single-pass alpha-helical"
            broad_type = "alpha-helical transmembrane protein"
            compact_label = "1TM alpha-helical membrane chain"
        elif tm_count == 0:
            topology_class = "no transmembrane segment predicted"
            broad_type = "non-membrane or peripheral candidate"
            compact_label = "0TM sequence topology prediction"

        if compact_label and orientation_label:
            compact_label = f"{compact_label}, {orientation_label}"
        if tm_count and tm_count > 0:
            chain_summary = (
                f"Single chain: {tm_count}TM alpha-helical integral membrane protein"
                + (f", {orientation_label}" if orientation_label else "")
            )

        topology_map = [
            {
                "label": item.get("display_label"),
                "kind": item.get("kind"),
                "start": item.get("start"),
                "end": item.get("end"),
                "length": item.get("length"),
            }
            for item in classified_regions
        ]

        topology_path = " -> ".join(
            item.get("display_label")
            for item in classified_regions
            if item.get("display_label")
        ) or None

        loop_annotations = []
        for item_index, item in enumerate(classified_regions):
            if item.get("kind") in {"membrane", "signal"}:
                continue
            if int(item.get("length") or 0) < cls.LARGE_TOPOLOGY_REGION_THRESHOLD:
                continue

            prev_tm = next(
                (
                    candidate.get("tm_index")
                    for candidate in reversed(classified_regions[:item_index])
                    if candidate.get("kind") == "membrane"
                ),
                None,
            )
            next_tm = next(
                (
                    candidate.get("tm_index")
                    for candidate in classified_regions[item_index + 1 :]
                    if candidate.get("kind") == "membrane"
                ),
                None,
            )
            side_phrase = cls._loop_side_phrase(item.get("kind"))
            descriptor = "region"
            if prev_tm is None and next_tm is not None:
                descriptor = f"N-terminal region before TM{next_tm}"
            elif prev_tm is not None and next_tm is not None:
                descriptor = f"loop/domain between TM{prev_tm} and TM{next_tm}"
            elif prev_tm is not None and next_tm is None:
                descriptor = f"C-terminal region after TM{prev_tm}"
            if side_phrase:
                descriptor = f"{side_phrase} {descriptor}"
            loop_annotations.append(
                {
                    "summary": f"Large {descriptor}",
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "length": item.get("length"),
                    "side": item.get("kind"),
                }
            )

        caveats = []
        chain_count = cls._safe_int((structure_context or {}).get("chain_count"))
        assembly_ids = (structure_context or {}).get("assembly_ids") or []
        if chain_count and chain_count > 1:
            caveats.append(
                "This topology summary applies to the predicted sequence or chain and does not by itself define the full multichain assembly."
            )
        if len(assembly_ids) > 1:
            caveats.append(
                "Multiple assemblies are present, so assembly-level interpretation may differ from the single-chain topology."
            )

        return {
            "available": True,
            "tm_count": tm_count,
            "n_terminal_side": n_terminal_side,
            "c_terminal_side": c_terminal_side,
            "n_terminal_orientation": cls._orientation_phrase(n_terminal_side),
            "c_terminal_orientation": cls._orientation_phrase(c_terminal_side),
            "orientation_label": orientation_label,
            "topology_class": topology_class,
            "broad_type": broad_type,
            "chain_type": "integral membrane chain" if tm_count and tm_count > 0 else None,
            "topology_label": (f"{tm_count}TM" + (f", {orientation_label}" if orientation_label else "")) if tm_count is not None else None,
            "compact_label": compact_label,
            "chain_summary": chain_summary,
            "topology_path": topology_path,
            "topology_map": topology_map,
            "membrane_segments": [
                {
                    "label": item.get("display_label"),
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "length": item.get("length"),
                }
                for item in classified_regions
                if item.get("kind") == "membrane"
            ],
            "loop_annotations": loop_annotations,
            "caveats": caveats,
        }

    @classmethod
    def _parse_opm_tm_regions(cls, value):
        if not cls._has_value(value):
            return None
        try:
            parsed = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return None
        if not isinstance(parsed, list):
            return None

        regions = []
        for subunit in parsed:
            if not isinstance(subunit, dict):
                continue
            segment_value = str(subunit.get("segment") or "").strip()
            if not segment_value:
                continue
            chain = subunit.get("protein_letter")
            for index, match in enumerate(
                re.finditer(r"(\d+)\((\d+)-(\d+)\)", segment_value),
                start=1,
            ):
                region_index, start, end = match.groups()
                regions.append(
                    {
                        "index": int(region_index or index),
                        "start": int(start),
                        "end": int(end),
                        "chain": chain,
                        "label": "OPM",
                    }
                )
        return cls._normalize_tm_regions(regions) or None

    @staticmethod
    def _normalize_text(value):
        normalized = str(value or "").strip()
        if normalized.lower() in {"", "nan", "none", "null", "undefined"}:
            return None
        return normalized

    @staticmethod
    def _annotation_score_band(annotation_score):
        if annotation_score is None:
            return None
        if annotation_score >= 4:
            return "high"
        if annotation_score >= 2:
            return "medium"
        return "low"

    @classmethod
    def _to_json_safe(cls, value):
        if isinstance(value, dict):
            return {key: cls._to_json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._to_json_safe(item) for item in value]
        if value is None:
            return None
        if pd.isna(value):
            return None
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        return value


class DiscrepancyReviewService:
    VALID_STATUSES = {"open", "reviewed", "accepted", "rejected", "uncertain"}
    _REVIEW_SENTINEL = object()
    DEFAULT_PAGE_SIZE = 25
    MAX_PAGE_SIZE = 200
    MAX_CANDIDATE_CACHE_ENTRIES = 8
    MAX_LIST_CACHE_ENTRIES = 24
    MAX_SUMMARY_CACHE_ENTRIES = 24
    _candidate_cache = {}
    _list_cache = {}
    _summary_cache = {}
    _shared_cache = RedisCache()
    SHARED_CACHE_TTL = timedelta(hours=6)

    @classmethod
    def list_candidates(
        cls,
        status=None,
        disagreement_only=False,
        search=None,
        page=None,
        per_page=None,
    ):
        cache_version = cls._candidate_cache_version()
        normalized_status = cls._normalize_text(status)
        normalized_search = cls._normalize_text(search)
        list_cache_key = (
            cache_version,
            normalized_status,
            bool(disagreement_only),
            normalized_search,
            cls._normalize_page(page),
            cls._normalize_per_page(per_page),
            page is None,
            per_page is None,
        )
        cached_list = cls._list_cache.get(list_cache_key)
        if cached_list is not None:
            return cached_list
        shared_cache_key = cls._build_shared_cache_key("list", *list_cache_key)
        shared_cached_list = cls._shared_cache.get_item(shared_cache_key)
        if shared_cached_list is not None:
            cls._remember_cache_item(
                cls._list_cache,
                list_cache_key,
                shared_cached_list,
                cls.MAX_LIST_CACHE_ENTRIES,
            )
            return shared_cached_list

        candidates = cls._get_cached_candidates(cache_version)
        if not candidates:
            if page is None and per_page is None:
                return []
            payload = cls._build_paginated_payload(
                [],
                page=page,
                per_page=per_page,
                status=normalized_status,
                disagreement_only=disagreement_only,
                search=normalized_search,
            )
            cls._remember_cache_item(cls._list_cache, list_cache_key, payload, cls.MAX_LIST_CACHE_ENTRIES)
            cls._shared_cache.set_item(shared_cache_key, payload, ttl=cls.SHARED_CACHE_TTL)
            return payload

        filtered_candidates = []
        for candidate in candidates:
            if disagreement_only and not candidate["discrepancy_summary"]["has_any_disagreement"]:
                continue
            if normalized_status and candidate["review"]["status"] != normalized_status:
                continue
            if normalized_search and not cls._candidate_matches_search(candidate, normalized_search):
                continue
            filtered_candidates.append(candidate)

        if page is not None or per_page is not None:
            payload = cls._build_paginated_payload(
                filtered_candidates,
                page=page,
                per_page=per_page,
                status=normalized_status,
                disagreement_only=disagreement_only,
                search=normalized_search,
            )
            cls._remember_cache_item(cls._list_cache, list_cache_key, payload, cls.MAX_LIST_CACHE_ENTRIES)
            cls._shared_cache.set_item(shared_cache_key, payload, ttl=cls.SHARED_CACHE_TTL)
            summary_cache_key = cls._build_shared_cache_key(
                "summary",
                cache_version,
                bool(disagreement_only),
                normalized_status,
                normalized_search,
            )
            summary = cls._build_summary_payload(filtered_candidates)
            cls._remember_cache_item(
                cls._summary_cache,
                (
                    cache_version,
                    bool(disagreement_only),
                    normalized_status,
                    normalized_search,
                ),
                summary,
                cls.MAX_SUMMARY_CACHE_ENTRIES,
            )
            cls._shared_cache.set_item(summary_cache_key, summary, ttl=cls.SHARED_CACHE_TTL)
            return payload
        cls._remember_cache_item(
            cls._list_cache,
            list_cache_key,
            filtered_candidates,
            cls.MAX_LIST_CACHE_ENTRIES,
        )
        return filtered_candidates

    @classmethod
    def summarize_candidates(cls, disagreement_only=False, status=None, search=None):
        cache_version = cls._candidate_cache_version()
        normalized_status = cls._normalize_text(status)
        normalized_search = cls._normalize_text(search)
        summary_cache_key = (
            cache_version,
            bool(disagreement_only),
            normalized_status,
            normalized_search,
        )
        cached_summary = cls._summary_cache.get(summary_cache_key)
        if cached_summary is not None:
            return cached_summary
        shared_summary_key = cls._build_shared_cache_key("summary", *summary_cache_key)
        shared_cached_summary = cls._shared_cache.get_item(shared_summary_key)
        if shared_cached_summary is not None:
            cls._remember_cache_item(
                cls._summary_cache,
                summary_cache_key,
                shared_cached_summary,
                cls.MAX_SUMMARY_CACHE_ENTRIES,
            )
            return shared_cached_summary

        candidates = cls.list_candidates(
            status=normalized_status,
            disagreement_only=disagreement_only,
            search=normalized_search,
        )
        summary = cls._build_summary_payload(candidates)
        cls._remember_cache_item(
            cls._summary_cache,
            summary_cache_key,
            summary,
            cls.MAX_SUMMARY_CACHE_ENTRIES,
        )
        cls._shared_cache.set_item(shared_summary_key, summary, ttl=cls.SHARED_CACHE_TTL)
        return summary

    @classmethod
    def _get_cached_candidates(cls, cache_version):
        cached = cls._candidate_cache.get(cache_version)
        if cached is not None:
            return cached

        dataset = DashboardAnnotationDatasetService._load_enriched_dataset()
        if dataset.empty:
            cls._remember_cache_item(
                cls._candidate_cache,
                cache_version,
                [],
                cls.MAX_CANDIDATE_CACHE_ENTRIES,
            )
            return []

        review_index = cls._load_review_index()
        raw_records = [
            DashboardAnnotationDatasetService._to_json_safe(_clean_search_record(record))
            for record in dataset.to_dict(orient="records")
        ]
        enriched_records = DashboardAnnotationDatasetService._attach_normalized_tm_prediction_payloads(
            raw_records
        )
        candidates = []
        for cleaned_record in enriched_records:
            candidate = cls._build_candidate_payload(
                cleaned_record,
                review=cls._find_review_for_record(
                    cleaned_record,
                    review_index=review_index,
                ),
                include_ui_sections=False,
            )
            candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                cls._status_sort_key(item["review"]["status"]),
                str(item["pdb_code"]),
            )
        )
        cls._remember_cache_item(
            cls._candidate_cache,
            cache_version,
            candidates,
            cls.MAX_CANDIDATE_CACHE_ENTRIES,
        )
        return candidates

    @classmethod
    def _candidate_cache_version(cls):
        dataset_path = DashboardConfigurationService.get_annotation_dataset_path()
        return (
            DashboardAnnotationDatasetService._enriched_cache_key(dataset_path),
            cls._review_state_key(),
        )

    @staticmethod
    def _build_shared_cache_key(namespace, *parts):
        serialized = json.dumps(parts, sort_keys=True, default=str)
        digest = hashlib.md5(serialized.encode("utf-8")).hexdigest()
        return f"discrepancy_review:{namespace}:{digest}"

    @classmethod
    def _build_summary_payload(cls, candidates):
        status_counts = {}
        benchmark_status_counts = {}
        scientific_flag_counts = {
            "benchmark_not_recommended": 0,
            "context_dependent_topology": 0,
            "non_canonical_membrane_case": 0,
            "multichain_context": 0,
            "obsolete_or_replaced": 0,
        }
        group_disagreement_count = 0
        tm_disagreement_count = 0
        tm_boundary_disagreement_count = 0
        benchmark_included_count = 0
        high_confidence_count = 0

        for candidate in candidates:
            review_status = ((candidate.get("review") or {}).get("status") or "open").strip().lower()
            status_counts[review_status] = status_counts.get(review_status, 0) + 1

            discrepancy = candidate.get("discrepancy_summary") or {}
            if discrepancy.get("has_group_disagreement"):
                group_disagreement_count += 1
            if discrepancy.get("has_tm_disagreement"):
                tm_disagreement_count += 1
            if discrepancy.get("has_tm_boundary_disagreement"):
                tm_boundary_disagreement_count += 1

            benchmark_decision = candidate.get("benchmark_decision") or cls._build_benchmark_decision(candidate)
            if benchmark_decision.get("include_in_benchmark"):
                benchmark_included_count += 1
            if benchmark_decision.get("high_confidence_subset"):
                high_confidence_count += 1
            benchmark_status = benchmark_decision.get("benchmark_status") or "excluded"
            benchmark_status_counts[benchmark_status] = benchmark_status_counts.get(benchmark_status, 0) + 1

            scientific_assessment = ((candidate.get("record") or {}).get("scientific_assessment") or {})
            flags = scientific_assessment.get("flags") or {}
            if scientific_assessment.get("recommended_for_sequence_topology_benchmark") is False:
                scientific_flag_counts["benchmark_not_recommended"] += 1
            for key in ("context_dependent_topology", "non_canonical_membrane_case", "multichain_context", "obsolete_or_replaced"):
                if flags.get(key):
                    scientific_flag_counts[key] += 1

        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "row_count": len(candidates),
            "status_counts": status_counts,
            "disagreement_counts": {
                "group": group_disagreement_count,
                "tm_count": tm_disagreement_count,
                "tm_boundary": tm_boundary_disagreement_count,
            },
            "benchmark_counts": {
                "included": benchmark_included_count,
                "high_confidence": high_confidence_count,
            },
            "benchmark_status_counts": benchmark_status_counts,
            "scientific_flag_counts": scientific_flag_counts,
        }

    @staticmethod
    def _remember_cache_item(cache_store, key, value, max_entries):
        if key in cache_store:
            cache_store.pop(key)
        cache_store[key] = value
        while len(cache_store) > max_entries:
            oldest_key = next(iter(cache_store))
            cache_store.pop(oldest_key, None)

    @staticmethod
    def _review_state_key():
        try:
            review_count, latest_reviewed_at = db.session.query(
                func.count(DiscrepancyReview.id),
                func.max(DiscrepancyReview.reviewed_at),
            ).one()
        except Exception as exc:
            logger.warning("Unable to build discrepancy review cache fingerprint: %s", exc)
            return ("unavailable",)

        return (
            int(review_count or 0),
            latest_reviewed_at.isoformat() if latest_reviewed_at else None,
        )

    @classmethod
    def _build_paginated_payload(
        cls,
        candidates,
        page=None,
        per_page=None,
        status=None,
        disagreement_only=False,
        search=None,
    ):
        normalized_page = cls._normalize_page(page)
        normalized_per_page = cls._normalize_per_page(per_page)
        total_items = len(candidates)
        total_pages = max(1, int(np.ceil(total_items / normalized_per_page))) if total_items else 1
        normalized_page = min(normalized_page, total_pages)
        start_index = (normalized_page - 1) * normalized_per_page
        end_index = start_index + normalized_per_page
        page_items = candidates[start_index:end_index]

        return {
            "items": page_items,
            "pagination": {
                "page": normalized_page,
                "per_page": normalized_per_page,
                "total_items": total_items,
                "total_pages": total_pages,
                "returned_items": len(page_items),
                "has_prev": normalized_page > 1,
                "has_next": normalized_page < total_pages,
            },
            "filters": {
                "status": cls._normalize_text(status),
                "disagreement_only": bool(disagreement_only),
                "search": cls._normalize_text(search),
            },
            "export_formats": ["json", "csv", "xlsx", "tsv"],
        }

    @classmethod
    def _candidate_matches_search(cls, candidate, search):
        normalized_search = cls._normalize_text(search)
        if not normalized_search:
            return True

        discrepancy = candidate.get("discrepancy_summary") or {}
        review = candidate.get("review") or {}
        record = candidate.get("record") or {}
        group_labels = discrepancy.get("group_labels") or {}
        tm_counts = discrepancy.get("tm_counts") or {}
        searchable_values = [
            candidate.get("pdb_code"),
            candidate.get("canonical_pdb_code"),
            candidate.get("bibliography_year"),
            review.get("status"),
            record.get("PDB Code"),
            record.get("legacy_pdb_code"),
            record.get("replacement_pdb_code"),
            record.get("canonical_pdb_code"),
            record.get("uniprot_id"),
            record.get("group"),
            record.get("Group (MPstruc)"),
            record.get("Group (OPM)"),
            record.get("Group (Predicted)"),
            record.get("Group (Expert)"),
            record.get("species"),
            record.get("subgroup"),
            record.get("name"),
            record.get("struct_title"),
            group_labels.get("mpstruc"),
            group_labels.get("opm"),
            group_labels.get("predicted"),
            group_labels.get("expert"),
        ]
        searchable_values.extend(
            f"{method}:{value}"
            for method, value in (tm_counts or {}).items()
            if value not in (None, "")
        )
        search_blob = " ".join(
            str(value).strip().lower()
            for value in searchable_values
            if str(value or "").strip()
        )
        return normalized_search.lower() in search_blob

    @classmethod
    def _normalize_page(cls, page):
        try:
            return max(int(page or 1), 1)
        except (TypeError, ValueError):
            return 1

    @classmethod
    def _normalize_per_page(cls, per_page):
        try:
            requested = int(per_page or cls.DEFAULT_PAGE_SIZE)
        except (TypeError, ValueError):
            requested = cls.DEFAULT_PAGE_SIZE
        return max(1, min(requested, cls.MAX_PAGE_SIZE))

    @classmethod
    def get_candidate(cls, pdb_code):
        record = DashboardAnnotationDatasetService.get_record(pdb_code)
        if record is None:
            return None
        return cls._build_candidate_payload(record, include_ui_sections=True)

    @classmethod
    def upsert_review(cls, pdb_code, payload, current_user):
        status = str(payload.get("status", "reviewed")).strip().lower()
        if status not in cls.VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Expected one of: {', '.join(sorted(cls.VALID_STATUSES))}"
            )

        record = DashboardAnnotationDatasetService.get_record(pdb_code)
        if record is None:
            raise LookupError(f"Annotated discrepancy record '{pdb_code}' was not found")

        review_key = cls._review_key(record)
        review = cls._find_review(review_key)
        if review is None:
            review = DiscrepancyReview(pdb_code=review_key)
            db.session.add(review)

        review.canonical_pdb_code = (
            record.get("canonical_pdb_code")
            or record.get("pdb_code")
            or record.get("PDB Code")
        )
        review.status = status
        review.rationale = payload.get("rationale")
        review.reviewer_note = payload.get("reviewer_note")
        review.reviewed_group = standardize_group_label(payload.get("reviewed_group"))

        reviewed_tm_count = payload.get("reviewed_tm_count")
        if reviewed_tm_count in ("", None):
            review.reviewed_tm_count = None
        else:
            review.reviewed_tm_count = int(reviewed_tm_count)

        review.reviewed_by_user_id = getattr(current_user, "id", None)
        review.reviewed_at = datetime.utcnow()
        review.source_snapshot = cls._build_source_snapshot(record)
        db.session.commit()
        MetaMPAuditLogService.record_event(
            "discrepancy_review_updated",
            {
                "pdb_code": review.pdb_code,
                "canonical_pdb_code": review.canonical_pdb_code,
                "status": review.status,
                "reviewed_group": review.reviewed_group,
                "reviewed_tm_count": review.reviewed_tm_count,
                "reviewed_by_user_id": review.reviewed_by_user_id,
            },
            app_config=current_app.config,
        )

        return cls._build_candidate_payload(record, include_ui_sections=True)

    @classmethod
    def get_review_payload_for_record(cls, record, review=_REVIEW_SENTINEL):
        if review is cls._REVIEW_SENTINEL:
            review = cls._find_review_for_record(record)
        if review is None:
            return cls._default_review_payload(record)
        return cls._serialize_review(review)

    @classmethod
    def _build_candidate_payload(cls, record, review=_REVIEW_SENTINEL, include_ui_sections=False):
        normalized_record = DashboardAnnotationDatasetService._normalize_record_metadata(
            dict(record)
        )
        if "provenance" not in normalized_record:
            normalized_record["provenance"] = DashboardAnnotationDatasetService._build_record_provenance(
                normalized_record
            )
        bibliography_year = cls._resolve_record_year(normalized_record)
        if bibliography_year is not None:
            normalized_record["bibliography_year"] = bibliography_year
            normalized_record["Year"] = bibliography_year
            normalized_record["year"] = bibliography_year

        candidate = {
            "pdb_code": cls._review_key(normalized_record),
            "canonical_pdb_code": (
                normalized_record.get("canonical_pdb_code")
                or normalized_record.get("pdb_code")
                or normalized_record.get("PDB Code")
            ),
            "bibliography_year": bibliography_year,
            "Year": bibliography_year,
            "year": bibliography_year,
            "discrepancy_summary": cls._build_discrepancy_summary(normalized_record),
            "review": cls.get_review_payload_for_record(normalized_record, review=review),
            "record": normalized_record,
        }
        if include_ui_sections:
            candidate["ui_sections"] = DashboardAnnotationDatasetService._build_ui_sections(normalized_record)
            normalized_record["ui_sections"] = candidate["ui_sections"]
        candidate["benchmark_decision"] = cls._build_benchmark_decision(candidate)
        if include_ui_sections:
            candidate["source_freshness"] = candidate["ui_sections"].get("live_status") or {}
        return candidate

    @staticmethod
    def _resolve_record_year(record):
        for key in ("bibliography_year", "Year", "year", "citation_year", "rcsb_primary_citation_year"):
            value = record.get(key)
            if value in (None, ""):
                continue
            try:
                return int(float(value))
            except (TypeError, ValueError):
                continue
        return None

    @classmethod
    def _build_discrepancy_summary(cls, record):
        group_labels = {
            "opm": standardize_group_label(
                record.get("Group (OPM)")
                or record.get("famsupclasstype_type_name")
                or record.get("family_superfamily_classtype_type_name")
                or record.get("family_superfamily_classtype_name")
            ),
            "mpstruc": standardize_group_label(record.get("Group (MPstruc)") or record.get("group")),
            "predicted": standardize_group_label(record.get("Group (Predicted)")),
            "expert": standardize_group_label(record.get("Group (Expert)")),
        }
        comparison_group_labels = {
            key: collapse_group_label_for_disagreement(value)
            for key, value in group_labels.items()
        }
        unique_group_labels = sorted({value for value in comparison_group_labels.values() if value})
        tm_regions = {
            "opm": DashboardAnnotationDatasetService._deserialize_tm_regions(record.get("opm_tm_regions")),
        }
        opm_tm_count = DashboardAnnotationDatasetService._safe_int(
            record.get("subunit_segments")
        )
        tm_counts = {
            "expert": DashboardAnnotationDatasetService._safe_int(record.get("TM (Expert)")),
            "opm": opm_tm_count if opm_tm_count is not None else (len(tm_regions["opm"]) if tm_regions["opm"] else None),
        }

        normalized_predictions = record.get("normalized_tm_predictions") or []
        for summary in normalized_predictions:
            method = cls._normalize_text(summary.get("method"))
            if not method or summary.get("ambiguous"):
                continue
            method_regions = DashboardAnnotationDatasetService._get_tmalphafold_tm_regions(summary)
            tm_regions[method] = method_regions
            method_count = DashboardAnnotationDatasetService._safe_int(summary.get("tm_count"))
            if method_count is None and method_regions:
                method_count = len(method_regions)
            tm_counts[method] = method_count

        for legacy_method in ("TMbed", "DeepTMHMM", "Phobius", "TOPCONS", "CCTOP"):
            tm_regions.setdefault(
                legacy_method,
                DashboardAnnotationDatasetService._deserialize_tm_regions(
                    record.get(f"{legacy_method}_tm_regions")
                ),
            )
            tm_counts.setdefault(
                legacy_method,
                DashboardAnnotationDatasetService._safe_int(record.get(f"{legacy_method}_tm_count")),
            )

        available_tm_counts = [value for value in tm_counts.values() if value is not None]
        available_region_counts = {
            key: len(value)
            for key, value in tm_regions.items()
            if value
        }
        structure_context = record.get("structure_context") or {}
        scientific_assessment = record.get("scientific_assessment") or build_scientific_assessment(record)

        return {
            "group_labels": group_labels,
            "group_labels_for_disagreement": comparison_group_labels,
            "tm_counts": tm_counts,
            "tm_regions": tm_regions,
            "structure_context": structure_context,
            "scientific_assessment": scientific_assessment,
            "has_group_disagreement": len(unique_group_labels) > 1,
            "has_tm_disagreement": len(set(available_tm_counts)) > 1 if available_tm_counts else False,
            "has_tm_boundary_disagreement": (
                len(set(available_region_counts.values())) > 1
                if available_region_counts
                else False
            ),
            "has_multichain_context": bool(structure_context.get("chain_count") and structure_context.get("chain_count") > 1),
            "has_any_disagreement": (len(unique_group_labels) > 1) or (
                len(set(available_tm_counts)) > 1 if available_tm_counts else False
            ) or (
                len(set(available_region_counts.values())) > 1
                if available_region_counts
                else False
            ),
            "expert_vs_prediction_agreement": (
                comparison_group_labels["expert"] == comparison_group_labels["predicted"]
                if comparison_group_labels["expert"] and comparison_group_labels["predicted"]
                else None
            ),
        }

    @classmethod
    def _build_benchmark_decision(cls, candidate):
        record = candidate.get("record", {}) or {}
        discrepancy = candidate.get("discrepancy_summary", {}) or {}
        review = candidate.get("review", {}) or {}
        review_status = review.get("status") or "open"
        tm_counts = discrepancy.get("tm_counts") or {}
        group_labels = discrepancy.get("group_labels") or {}
        scientific_assessment = record.get("scientific_assessment") or build_scientific_assessment(record)
        scientific_flags = scientific_assessment.get("flags") or {}

        inclusion_reasons = []
        exclusion_reasons = []

        if discrepancy.get("has_any_disagreement"):
            inclusion_reasons.append("has disagreement")
        if review_status in {"accepted", "reviewed", "uncertain"}:
            inclusion_reasons.append(f"review_status:{review_status}")
        if not DashboardAnnotationDatasetService._has_value(record.get("Group (Expert)")):
            exclusion_reasons.append("missing_expert_group")
        if not DashboardAnnotationDatasetService._has_value(record.get("TM (Expert)")):
            exclusion_reasons.append("missing_expert_tm_count")
        if tm_counts.get("TMbed") is None and tm_counts.get("DeepTMHMM") is None:
            exclusion_reasons.append("missing_predictor_tm_counts")
        if review_status == "rejected":
            exclusion_reasons.append("review_rejected")
        exclusion_reasons.extend(scientific_assessment.get("benchmark_exclusion_reasons") or [])

        include_in_benchmark = bool(inclusion_reasons) and not (
            "missing_expert_group" in exclusion_reasons
            and "missing_expert_tm_count" in exclusion_reasons
        ) and review_status != "rejected"

        high_confidence_subset = (
            review_status != "rejected"
            and not _normalize_replacement_flag(record.get("is_replaced"))
            and not scientific_flags.get("context_dependent_topology")
            and not scientific_flags.get("non_canonical_membrane_case")
            and discrepancy.get("has_tm_boundary_disagreement") is not True
            and (
                discrepancy.get("expert_vs_prediction_agreement") is True
                or (
                    review_status in {"accepted", "reviewed"}
                    and DashboardAnnotationDatasetService._has_value(review.get("reviewed_group"))
                    and cls._normalize_text(review.get("reviewed_group"))
                    == cls._normalize_text(group_labels.get("predicted"))
                )
            )
            and (
                tm_counts.get("expert") is None
                or (
                    (tm_counts.get("TMbed") is None or tm_counts.get("TMbed") == tm_counts.get("expert"))
                    and (
                        tm_counts.get("DeepTMHMM") is None
                        or tm_counts.get("DeepTMHMM") == tm_counts.get("expert")
                    )
                )
            )
        )

        benchmark_status = cls._derive_benchmark_status(
            include_in_benchmark=include_in_benchmark,
            high_confidence_subset=high_confidence_subset,
            benchmark_recommended=scientific_assessment.get(
                "recommended_for_sequence_topology_benchmark"
            ),
            review_status=review_status,
        )
        benchmark_reason = cls._derive_benchmark_reason(
            benchmark_status=benchmark_status,
            inclusion_reasons=inclusion_reasons,
            exclusion_reasons=exclusion_reasons,
            scientific_assessment=scientific_assessment,
            review_status=review_status,
        )

        return {
            "include_in_benchmark": include_in_benchmark,
            "high_confidence_subset": high_confidence_subset,
            "inclusion_reasons": list(dict.fromkeys(inclusion_reasons)),
            "exclusion_reasons": list(dict.fromkeys(exclusion_reasons)),
            "benchmark_status": benchmark_status,
            "benchmark_reason": benchmark_reason,
        }

    @staticmethod
    def _derive_benchmark_status(
        include_in_benchmark,
        high_confidence_subset,
        benchmark_recommended,
        review_status,
    ):
        if review_status == "rejected":
            return "excluded"
        if high_confidence_subset:
            return "high_confidence_subset"
        if include_in_benchmark:
            return "included_with_caution"
        if benchmark_recommended is False:
            return "not_recommended"
        return "excluded"

    @classmethod
    def _derive_benchmark_reason(
        cls,
        benchmark_status,
        inclusion_reasons,
        exclusion_reasons,
        scientific_assessment,
        review_status,
    ):
        inclusion_reasons = list(dict.fromkeys(inclusion_reasons or []))
        exclusion_reasons = list(dict.fromkeys(exclusion_reasons or []))
        notes = list(dict.fromkeys((scientific_assessment or {}).get("notes") or []))

        reason_labels = {
            "has disagreement": "the record contains cross-source disagreement",
            "review_status:accepted": "the record was accepted in expert review",
            "review_status:reviewed": "the record was reviewed by an expert",
            "review_status:uncertain": "the record was reviewed but remains uncertain",
            "missing_expert_group": "the expert broad-group label is missing",
            "missing_expert_tm_count": "the expert TM-count label is missing",
            "missing_predictor_tm_counts": "predictor TM counts are missing",
            "review_rejected": "the review marked the record as rejected",
            "context_dependent_topology": "topology interpretation depends on biological state or transition",
            "non_canonical_membrane_case": "the record appears to be a non-canonical membrane case",
            "replaced_entry": "the entry is obsolete or replaced",
            "sequence_topology_not_recommended": "sequence-only topology comparison is not recommended",
        }

        def _phrase(value):
            return reason_labels.get(value, str(value).replace("_", " "))

        if benchmark_status == "high_confidence_subset":
            return (
                "Included in the high-confidence subset because the record passed the benchmark inclusion rules "
                "without the major caution criteria used to exclude context-dependent, non-canonical, replaced, "
                "or TM-boundary-disagreeing cases."
            )

        if benchmark_status == "included_with_caution":
            positives = ", ".join(_phrase(value) for value in inclusion_reasons) or "benchmark inclusion criteria"
            cautions = ", ".join(_phrase(value) for value in exclusion_reasons) or "additional caution criteria"
            return f"Included in the benchmark because {positives}, but retained with caution because {cautions}."

        if benchmark_status == "not_recommended":
            details = ", ".join(_phrase(value) for value in exclusion_reasons) or "benchmark caution criteria"
            if notes:
                return f"Not recommended for straightforward sequence-topology benchmarking because {details}. Notes: {notes[0]}"
            return f"Not recommended for straightforward sequence-topology benchmarking because {details}."

        if review_status == "rejected":
            return "Excluded from the benchmark because the discrepancy review status is rejected."

        details = ", ".join(_phrase(value) for value in exclusion_reasons) or "the record did not satisfy the benchmark inclusion rules"
        return f"Excluded from the benchmark because {details}."

    @classmethod
    def _find_review_for_record(cls, record, review_index=None):
        review_key = cls._review_key(record)
        review = cls._find_review(review_key, review_index=review_index)
        if review is not None:
            return review

        canonical_code = cls._normalize_text(
            record.get("canonical_pdb_code") or record.get("pdb_code")
        )
        if canonical_code and canonical_code != review_key:
            return cls._find_review(canonical_code, review_index=review_index)
        return None

    @staticmethod
    def _find_review(review_key, review_index=None):
        if not review_key:
            return None
        if review_index is not None:
            return review_index.get(review_key)
        try:
            return (
                DiscrepancyReview.query.filter_by(pdb_code=review_key).first()
                or DiscrepancyReview.query.filter_by(canonical_pdb_code=review_key).first()
            )
        except Exception as exc:
            logger.warning(
                "Discrepancy review lookup unavailable for %s: %s",
                review_key,
                exc,
            )
            return None

    @staticmethod
    def _load_review_index():
        try:
            reviews = DiscrepancyReview.query.all()
        except Exception as exc:
            logger.warning("Discrepancy review index unavailable: %s", exc)
            return {}

        review_index = {}
        for review in reviews:
            if review.pdb_code:
                review_index[str(review.pdb_code).strip().upper()] = review
            if review.canonical_pdb_code:
                review_index[str(review.canonical_pdb_code).strip().upper()] = review
        return review_index

    @classmethod
    def _review_key(cls, record):
        return cls._normalize_text(record.get("PDB Code") or record.get("pdb_code"))

    @staticmethod
    def _default_review_payload(record):
        return {
            "status": "open",
            "rationale": None,
            "reviewer_note": None,
            "reviewed_group": None,
            "reviewed_tm_count": None,
            "reviewed_by": None,
            "reviewed_at": None,
            "review_key": DiscrepancyReviewService._review_key(record),
            "canonical_pdb_code": (
                record.get("canonical_pdb_code")
                or record.get("pdb_code")
                or record.get("PDB Code")
            ),
        }

    @staticmethod
    def _serialize_review(review):
        reviewer = getattr(review, "reviewer", None)
        reviewer_payload = None
        if reviewer is not None:
            reviewer_payload = {
                "id": reviewer.id,
                "username": getattr(reviewer, "username", None),
                "email": getattr(reviewer, "email", None),
                "name": getattr(reviewer, "name", None),
            }

        return {
            "status": review.status,
            "rationale": review.rationale,
            "reviewer_note": review.reviewer_note,
            "reviewed_group": standardize_group_label(review.reviewed_group),
            "reviewed_tm_count": review.reviewed_tm_count,
            "reviewed_by": reviewer_payload,
            "reviewed_at": review.reviewed_at.isoformat() if review.reviewed_at else None,
            "review_key": review.pdb_code,
            "canonical_pdb_code": review.canonical_pdb_code,
        }

    @staticmethod
    def _build_source_snapshot(record):
        return {
            "group_opm": standardize_group_label(record.get("Group (OPM)")),
            "group_mpstruc": standardize_group_label(record.get("Group (MPstruc)") or record.get("group")),
            "group_predicted": standardize_group_label(record.get("Group (Predicted)")),
            "group_expert": standardize_group_label(record.get("Group (Expert)")),
            "tm_expert": parse_tm_count(record.get("TM (Expert)")),
            "opm_tm_regions": record.get("opm_tm_regions"),
            "tmbed_tm_count": record.get("TMbed_tm_count"),
            "tmbed_tm_regions": record.get("TMbed_tm_regions"),
            "deeptmhmm_tm_count": record.get("DeepTMHMM_tm_count"),
            "deeptmhmm_tm_regions": record.get("DeepTMHMM_tm_regions"),
            "structure_context": record.get("structure_context"),
            "scientific_assessment": record.get("scientific_assessment") or build_scientific_assessment(record),
            "canonical_pdb_code": record.get("canonical_pdb_code"),
        }

    @staticmethod
    def _normalize_text(value):
        normalized = str(value or "").strip()
        if normalized.lower() in {"", "nan", "none", "null", "undefined"}:
            return None
        return normalized

    @staticmethod
    def _status_sort_key(status):
        ordering = {
            "open": 0,
            "uncertain": 1,
            "reviewed": 2,
            "accepted": 3,
            "rejected": 4,
        }
        return ordering.get(status, 99)


class DiscrepancyReviewExportService:
    EXPORT_BASENAME = "metamp_discrepancy_review_queue"

    @classmethod
    def build_export_rows(cls, status=None, disagreement_only=False, search=None):
        candidates = DiscrepancyReviewService.list_candidates(
            status=status,
            disagreement_only=disagreement_only,
            search=search,
        )
        return [
            DiscrepancyBenchmarkExportService._build_export_row(candidate)
            for candidate in candidates
        ]

    @classmethod
    def build_export_dataframe(cls, status=None, disagreement_only=False, search=None):
        return pd.DataFrame(
            cls.build_export_rows(
                status=status,
                disagreement_only=disagreement_only,
                search=search,
            )
        )

    @classmethod
    def build_download_payload(
        cls,
        export_format="csv",
        status=None,
        disagreement_only=False,
        search=None,
    ):
        dataframe = cls.build_export_dataframe(
            status=status,
            disagreement_only=disagreement_only,
            search=search,
        )
        metadata = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "row_count": int(len(dataframe.index)),
            "filters": {
                "status": DiscrepancyReviewService._normalize_text(status),
                "disagreement_only": bool(disagreement_only),
                "search": DiscrepancyReviewService._normalize_text(search),
            },
            "export_scope": "discrepancy_review_queue",
        }
        timestamp = metadata["generated_at"].replace(":", "").replace("-", "")
        filename_prefix = f"{cls.EXPORT_BASENAME}_{timestamp}"
        return DiscrepancyBenchmarkExportService._build_table_download_payload(
            dataframe=dataframe,
            metadata=metadata,
            export_format=export_format,
            filename_prefix=filename_prefix,
            json_wrapper=True,
        )


class DiscrepancyBenchmarkExportService:
    EXPORT_BASENAME = "metamp_discrepancy_benchmark"
    HIGH_CONFIDENCE_BASENAME = "metamp_high_confidence_subset"

    @classmethod
    def build_export_rows(cls, include_all=False):
        candidates = DiscrepancyReviewService.list_candidates(
            disagreement_only=not include_all,
        )
        rows = []
        for candidate in candidates:
            rows.append(cls._build_export_row(candidate))
        return rows

    @classmethod
    def build_export_dataframe(cls, include_all=False):
        rows = cls.build_export_rows(include_all=include_all)
        return pd.DataFrame(rows)

    @classmethod
    def export_release(cls, include_all=False):
        dataframe = cls.build_export_dataframe(include_all=include_all)
        benchmark_dir = DashboardConfigurationService.get_benchmark_dir()
        previous_metadata = cls.latest_export_metadata()
        metadata = cls._build_release_metadata(
            dataframe,
            include_all=include_all,
            previous_metadata=previous_metadata,
        )
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        export_prefix = benchmark_dir / f"{cls.EXPORT_BASENAME}_{timestamp}"
        csv_path = export_prefix.with_suffix(".csv")
        json_path = export_prefix.with_suffix(".json")

        dataframe.to_csv(csv_path, index=False)
        json_path.write_text(
            json.dumps(
                {
                    "metadata": metadata,
                    "records": dataframe.to_dict(orient="records"),
                },
                indent=2,
                default=str,
            )
        )

        latest_metadata_path = benchmark_dir / f"{cls.EXPORT_BASENAME}_latest.json"
        latest_metadata_path.write_text(
            json.dumps(
                {
                    **metadata,
                    "csv_path": str(csv_path),
                    "json_path": str(json_path),
                },
                indent=2,
                default=str,
            )
        )
        cls._append_manifest_entry(benchmark_dir, {**metadata, "csv_path": str(csv_path), "json_path": str(json_path)})
        MetaMPAuditLogService.record_event(
            "benchmark_exported",
            {
                "release_id": metadata["release_id"],
                "row_count": metadata["row_count"],
                "included_row_count": metadata["included_row_count"],
                "csv_path": str(csv_path),
                "json_path": str(json_path),
            },
            app_config=current_app.config,
        )

        return {
            "metadata": metadata,
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "latest_metadata_path": str(latest_metadata_path),
        }

    @classmethod
    def ensure_fresh_export_metadata(cls, include_all=False):
        metadata = cls.latest_export_metadata()
        if metadata and cls._metadata_is_current(metadata, include_all=include_all):
            return metadata
        return cls.export_release(include_all=include_all).get("metadata")

    @classmethod
    def latest_export_metadata(cls):
        benchmark_dir = DashboardConfigurationService.get_benchmark_dir()
        metadata_path = benchmark_dir / f"{cls.EXPORT_BASENAME}_latest.json"
        if not metadata_path.exists():
            return None
        return json.loads(metadata_path.read_text())

    @classmethod
    def build_download_payload(cls, export_format="csv", include_all=False):
        dataframe = cls.build_export_dataframe(include_all=include_all)
        metadata = cls._build_release_metadata(
            dataframe,
            include_all=include_all,
            previous_metadata=cls.latest_export_metadata(),
        )
        filename_suffix = "all" if include_all else "disagreements"
        timestamp = metadata["generated_at"].replace(":", "").replace("-", "")
        return cls._build_table_download_payload(
            dataframe=dataframe,
            metadata=metadata,
            export_format=export_format,
            filename_prefix=f"{cls.EXPORT_BASENAME}_{filename_suffix}_{timestamp}",
            json_wrapper=True,
        )

    @classmethod
    def build_high_confidence_dataframe(cls):
        dataframe = cls.build_export_dataframe(include_all=True)
        if dataframe.empty or "high_confidence_subset" not in dataframe.columns:
            return dataframe.iloc[0:0]
        return dataframe[dataframe["high_confidence_subset"] == True].reset_index(drop=True)

    @classmethod
    def build_high_confidence_download_payload(cls, export_format="csv"):
        dataframe = cls.build_high_confidence_dataframe()
        metadata = cls._build_release_metadata(
            dataframe,
            include_all=True,
            previous_metadata=cls.latest_export_metadata(),
        )
        metadata["subset_type"] = "high_confidence"
        timestamp = metadata["generated_at"].replace(":", "").replace("-", "")
        return cls._build_table_download_payload(
            dataframe=dataframe,
            metadata=metadata,
            export_format=export_format,
            filename_prefix=f"{cls.HIGH_CONFIDENCE_BASENAME}_{timestamp}",
            json_wrapper=True,
        )

    @classmethod
    def _build_table_download_payload(
        cls,
        dataframe,
        metadata,
        export_format,
        filename_prefix,
        json_wrapper=False,
    ):
        normalized_format = str(export_format or "csv").strip().lower()
        if normalized_format == "csv":
            buffer = io.StringIO()
            dataframe.to_csv(buffer, index=False)
            return {
                "content": buffer.getvalue(),
                "content_type": "text/csv",
                "filename": f"{filename_prefix}.csv",
                "metadata": metadata,
            }
        if normalized_format == "tsv":
            buffer = io.StringIO()
            dataframe.to_csv(buffer, index=False, sep="\t")
            return {
                "content": buffer.getvalue(),
                "content_type": "text/tab-separated-values",
                "filename": f"{filename_prefix}.tsv",
                "metadata": metadata,
            }
        if normalized_format == "xlsx":
            buffer = io.BytesIO()
            dataframe.to_excel(buffer, index=False, sheet_name="discrepancy_reviews")
            return {
                "content": buffer.getvalue(),
                "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "filename": f"{filename_prefix}.xlsx",
                "metadata": metadata,
            }
        if normalized_format == "json":
            payload = (
                json.dumps(
                    {"metadata": metadata, "records": dataframe.to_dict(orient="records")},
                    indent=2,
                    default=str,
                )
                if json_wrapper
                else json.dumps(dataframe.to_dict(orient="records"), indent=2, default=str)
            )
            return {
                "content": payload,
                "content_type": "application/json",
                "filename": f"{filename_prefix}.json",
                "metadata": metadata,
            }
        raise ValueError(f"Unsupported benchmark export format '{export_format}'.")

    @classmethod
    def _build_release_metadata(cls, dataframe, include_all=False, previous_metadata=None):
        records = dataframe.to_dict(orient="records")
        included_rows = [row for row in records if row.get("include_in_benchmark")]
        high_confidence_rows = [row for row in records if row.get("high_confidence_subset")]
        available_years = []
        for row in records:
            value = row.get("bibliography_year")
            if value in (None, ""):
                continue
            try:
                available_years.append(int(float(value)))
            except (TypeError, ValueError):
                continue
        release_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        previous_release_id = (previous_metadata or {}).get("release_id")
        return {
            "release_id": release_id,
            "previous_release_id": previous_release_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "include_all": include_all,
            "row_count": int(len(records)),
            "included_row_count": int(len(included_rows)),
            "disagreement_row_count": int(
                sum(1 for row in records if row.get("has_any_disagreement"))
            ),
            "high_confidence_row_count": int(len(high_confidence_rows)),
            "accepted_review_count": int(
                sum(1 for row in records if row.get("review_status") == "accepted")
            ),
            "rejected_review_count": int(
                sum(1 for row in records if row.get("review_status") == "rejected")
            ),
            "annotation_dataset": str(
                DashboardConfigurationService.get_annotation_dataset_path()
            ),
            "annotation_dataset_modified_at": cls._file_modified_at(
                DashboardConfigurationService.get_annotation_dataset_path()
            ),
            "max_bibliography_year": max(available_years) if available_years else None,
            "min_bibliography_year": min(available_years) if available_years else None,
            "raw_opm_dataset": str(
                DashboardConfigurationService.get_raw_opm_dataset_path()
                or ""
            ),
            "raw_pdb_dataset": str(
                DashboardConfigurationService.get_raw_pdb_dataset_path()
                or ""
            ),
            "source_state": cls._build_source_state(),
            "changes_since_previous_release": cls._build_release_delta(
                previous_metadata,
                {
                    "row_count": int(len(records)),
                    "included_row_count": int(len(included_rows)),
                    "high_confidence_row_count": int(len(high_confidence_rows)),
                    "accepted_review_count": int(
                        sum(1 for row in records if row.get("review_status") == "accepted")
                    ),
                    "rejected_review_count": int(
                        sum(1 for row in records if row.get("review_status") == "rejected")
                    ),
                },
            ),
        }

    @classmethod
    def _build_source_state(cls):
        review_count = 0
        latest_reviewed_at = None
        try:
            review_count = DiscrepancyReview.query.count()
            latest_review = (
                DiscrepancyReview.query.order_by(DiscrepancyReview.reviewed_at.desc())
                .first()
            )
            if latest_review and latest_review.reviewed_at:
                latest_reviewed_at = latest_review.reviewed_at.isoformat()
        except Exception as exc:
            logger.warning("Unable to build discrepancy review source state: %s", exc)

        return {
            "annotation_dataset_modified_at": cls._file_modified_at(
                DashboardConfigurationService.get_annotation_dataset_path()
            ),
            "quantitative_dataset_modified_at": cls._file_modified_at(
                DashboardConfigurationService.get_valid_dataset_path("Quantitative_data.csv")
            ),
            "pdb_dataset_modified_at": cls._file_modified_at(
                DashboardConfigurationService.get_valid_dataset_path("PDB_data_transformed.csv")
            ),
            "opm_dataset_modified_at": cls._file_modified_at(
                DashboardConfigurationService.get_valid_dataset_path("NEWOPM.csv")
            ),
            "uniprot_dataset_modified_at": cls._file_modified_at(
                DashboardConfigurationService.get_valid_dataset_path("Uniprot_functions.csv")
            ),
            "tm_prediction_summary_modified_at": cls._file_modified_at(
                DashboardConfigurationService.get_tm_prediction_output_path()
            ),
            "review_count": int(review_count),
            "latest_reviewed_at": latest_reviewed_at,
        }

    @classmethod
    def _metadata_is_current(cls, metadata, include_all=False):
        if not metadata:
            return False
        if bool(metadata.get("include_all")) != bool(include_all):
            return False
        return (metadata.get("source_state") or {}) == cls._build_source_state()

    @staticmethod
    def _build_release_delta(previous_metadata, current_counts):
        if not previous_metadata:
            return None
        return {
            key: current_counts[key] - int(previous_metadata.get(key, 0) or 0)
            for key in current_counts
        }

    @classmethod
    def _append_manifest_entry(cls, benchmark_dir, metadata):
        manifest_path = benchmark_dir / f"{cls.EXPORT_BASENAME}_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                manifest = []
        else:
            manifest = []
        manifest.append(metadata)
        manifest = manifest[-25:]
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    @staticmethod
    def _file_modified_at(path):
        if path is None or not Path(path).exists():
            return None
        return datetime.fromtimestamp(Path(path).stat().st_mtime).isoformat()

    @classmethod
    def _build_export_row(cls, candidate):
        record = candidate.get("record", {}) or {}
        discrepancy = candidate.get("discrepancy_summary", {}) or {}
        review = candidate.get("review", {}) or {}
        structure_context = discrepancy.get("structure_context") or record.get("structure_context") or {}
        review_status = review.get("status") or "open"
        group_labels = discrepancy.get("group_labels") or {}
        tm_counts = discrepancy.get("tm_counts") or {}
        tm_regions = discrepancy.get("tm_regions") or {}
        scientific_assessment = record.get("scientific_assessment") or build_scientific_assessment(record)
        scientific_flags = scientific_assessment.get("flags") or {}
        benchmark_decision = candidate.get("benchmark_decision") or DiscrepancyReviewService._build_benchmark_decision(candidate)
        include_in_benchmark = benchmark_decision.get("include_in_benchmark")
        high_confidence_subset = benchmark_decision.get("high_confidence_subset")
        inclusion_reasons = benchmark_decision.get("inclusion_reasons") or []
        exclusion_reasons = benchmark_decision.get("exclusion_reasons") or []
        benchmark_status = benchmark_decision.get("benchmark_status")
        benchmark_reason = benchmark_decision.get("benchmark_reason")

        return {
            "pdb_code": candidate.get("pdb_code"),
            "canonical_pdb_code": candidate.get("canonical_pdb_code"),
            "bibliography_year": candidate.get("bibliography_year"),
            "legacy_pdb_code": record.get("legacy_pdb_code"),
            "replacement_pdb_code": record.get("replacement_pdb_code"),
            "is_replaced": record.get("is_replaced"),
            "group_opm": group_labels.get("opm"),
            "group_mpstruc": group_labels.get("mpstruc"),
            "group_predicted": group_labels.get("predicted"),
            "group_expert": group_labels.get("expert"),
            "tm_expert": tm_counts.get("expert"),
            "tm_opm": tm_counts.get("opm"),
            "tm_tmbed": tm_counts.get("TMbed"),
            "tm_deeptmhmm": tm_counts.get("DeepTMHMM"),
            "opm_tm_regions": json.dumps(tm_regions.get("opm") or []),
            "tmbed_tm_regions": json.dumps(tm_regions.get("TMbed") or []),
            "deeptmhmm_tm_regions": json.dumps(tm_regions.get("DeepTMHMM") or []),
            "has_group_disagreement": bool(discrepancy.get("has_group_disagreement")),
            "has_tm_disagreement": bool(discrepancy.get("has_tm_disagreement")),
            "has_tm_boundary_disagreement": bool(discrepancy.get("has_tm_boundary_disagreement")),
            "has_any_disagreement": bool(discrepancy.get("has_any_disagreement")),
            "expert_vs_prediction_agreement": discrepancy.get("expert_vs_prediction_agreement"),
            "benchmark_recommended": scientific_assessment.get(
                "recommended_for_sequence_topology_benchmark"
            ),
            "context_dependent_topology": scientific_flags.get("context_dependent_topology"),
            "non_canonical_membrane_case": scientific_flags.get("non_canonical_membrane_case"),
            "multichain_context": scientific_flags.get("multichain_context"),
            "obsolete_or_replaced": scientific_flags.get("obsolete_or_replaced"),
            "scientific_notes": json.dumps(scientific_assessment.get("notes") or []),
            "review_status": review_status,
            "benchmark_status": benchmark_status,
            "benchmark_reason": benchmark_reason,
            "reviewed_group": review.get("reviewed_group"),
            "reviewed_tm_count": review.get("reviewed_tm_count"),
            "review_rationale": review.get("rationale"),
            "reviewer_note": review.get("reviewer_note"),
            "reviewed_at": review.get("reviewed_at"),
            "structure_chain_ids": json.dumps(structure_context.get("chain_ids") or []),
            "structure_chain_count": structure_context.get("chain_count"),
            "structure_entity_ids": json.dumps(structure_context.get("entity_ids") or []),
            "structure_polymer_entity_ids": json.dumps(structure_context.get("polymer_entity_ids") or []),
            "structure_assembly_ids": json.dumps(structure_context.get("assembly_ids") or []),
            "structure_entity_assemblies": json.dumps(structure_context.get("entity_assemblies") or []),
            "structure_polymer_composition": structure_context.get("polymer_composition"),
            "structure_selected_polymer_entity_types": structure_context.get("selected_polymer_entity_types"),
            "include_in_benchmark": include_in_benchmark,
            "high_confidence_subset": high_confidence_subset,
            "inclusion_reasons": json.dumps(inclusion_reasons),
            "exclusion_reasons": json.dumps(list(dict.fromkeys(exclusion_reasons))),
        }

    @staticmethod
    def _has_value(value):
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, dict, tuple, set)):
            return len(value) > 0
        try:
            return not pd.isna(value)
        except TypeError:
            return True


class DashboardFieldMetadataService:
    @staticmethod
    def source_catalog():
        return {
            "mpstruc": {
                "label": "MPstruc",
                "category": "source_database",
                "description": "Curated membrane-protein grouping and subgroup information.",
            },
            "opm": {
                "label": "OPM",
                "category": "source_database",
                "description": "Membrane orientation and topology-related structural annotations.",
            },
            "pdb": {
                "label": "RCSB PDB",
                "category": "source_database",
                "description": "Structure-level experimental, bibliographic, and replacement metadata.",
            },
            "uniprot": {
                "label": "UniProt",
                "category": "source_database",
                "description": "Sequence-linked functional, taxonomic, and disease-linked metadata.",
            },
            "expert_annotation_dataset": {
                "label": "Expert Annotation Reference",
                "category": "static_reference",
                "description": "Static expert-reviewed reference fields used only for Group (Expert) and TM (Expert).",
                "is_static_reference": True,
            },
            "metamp_ml_classifier": {
                "label": "MetaMP Semi-supervised Classifier",
                "category": "live_generated",
                "description": "Live broad-group predictions generated by the current MetaMP semi-supervised model.",
                "is_live_generated": True,
            },
            "tmbed": {
                "label": "TMbed",
                "category": "live_generated",
                "description": "Sequence-based transmembrane predictor used as an assistive comparison signal.",
                "is_live_generated": True,
            },
            "deeptmhmm": {
                "label": "DeepTMHMM",
                "category": "live_generated",
                "description": "Sequence-based topology predictor used as an assistive comparison signal.",
                "is_live_generated": True,
            },
            "phobius": {
                "label": "Phobius",
                "category": "live_generated",
                "description": "Sequence-based predictor that jointly considers signal peptides and transmembrane helices.",
                "is_live_generated": True,
            },
            "topcons": {
                "label": "TOPCONS",
                "category": "live_generated",
                "description": "Consensus membrane-topology predictor used as an additional sequence-only comparison signal.",
                "is_live_generated": True,
            },
            "cctop": {
                "label": "CCTOP",
                "category": "live_generated",
                "description": "Consensus transmembrane-topology predictor used as an additional sequence-only comparison signal.",
                "is_live_generated": True,
            },
            "tmalphafold": {
                "label": "TMAlphaFold",
                "category": "live_generated",
                "description": "Remote UniProt-based multi-method prediction layer used for additional sequence-only and structure-aware comparison signals.",
                "is_live_generated": True,
            },
        }

    @staticmethod
    def scientific_flag_glossary():
        return {
            "recommended_for_sequence_topology_benchmark": {
                "label": "Benchmark Recommended",
                "description": "Whether the record appears suitable for straightforward comparison between sequence-based topology predictors and reference annotations.",
                "interpretation": "A false value means the record carries scientific or curation caveats that warrant extra caution.",
                "computation": "True unless the scientific assessment flags the record as context-dependent, non-canonical, or obsolete/replaced, or an entry-specific override marks sequence-only benchmarking as unsuitable.",
            },
            "context_dependent_topology": {
                "label": "Context-dependent Topology",
                "description": "Whether topology interpretation may depend on biological state, conformational transition, or related context.",
                "interpretation": "A true value means sequence-only topology comparison may not reflect the biologically relevant state.",
                "computation": "Derived from curated entry overrides and keyword rules that flag state-dependent proteins or soluble-to-membrane transitions.",
            },
            "non_canonical_membrane_case": {
                "label": "Non-canonical Membrane Case",
                "description": "Whether the record appears unlikely to represent a canonical membrane-protein target.",
                "interpretation": "A true value often reflects helper constructs, fusion components, or other non-canonical cases.",
                "computation": "Derived from curated entry overrides and keyword rules that identify helper constructs, fusion partners, or records treated as non-canonical membrane cases.",
            },
            "multichain_context": {
                "label": "Multichain Context",
                "description": "Whether multichain or assembly-level context could complicate simple per-sequence interpretation.",
                "interpretation": "A true value means chain-level or assembly-level review may be required.",
                "computation": "True when the structure-context summary indicates more than one chain in the resolved entry.",
            },
            "obsolete_or_replaced": {
                "label": "Obsolete or Replaced",
                "description": "Whether the record carries replacement or obsolescence metadata.",
                "interpretation": "A true value means the canonical replacement entry should be preferred during interpretation.",
                "computation": "True when PDB replacement metadata marks the entry as replaced or obsolete.",
            },
        }

    @classmethod
    def field_glossary(cls):
        glossary = {
            "benchmark_status": {
                "label": "Benchmark Decision",
                "description": "Backend summary of whether the record is excluded, included with caution, or retained in the high-confidence subset for benchmarking workflows.",
                "source": "benchmark_decision",
                "allowed_values": {
                    "excluded": "The record is not included in the discrepancy benchmark export.",
                    "included_with_caution": "The record is included in the broader discrepancy benchmark export, but at least one caution criterion still applies.",
                    "high_confidence_subset": "The record satisfies the stricter rules used for the high-confidence subset.",
                    "not_recommended": "The record is not considered suitable for straightforward sequence-only topology benchmarking, even if it remains visible for review.",
                },
            },
            "benchmark_reason": {
                "label": "Decision Notes",
                "description": "Human-readable explanation derived from benchmark inclusion rules, exclusion rules, and scientific assessment notes.",
                "source": "benchmark_decision",
                "interpretation": "Use together with Benchmark Decision and Benchmark Recommended to understand whether a record is benchmark-eligible, cautionary, or scientifically atypical.",
            },
            "Group (Expert)": {
                "label": "Group (Expert)",
                "description": "Broad structural group from the static expert annotation reference file.",
                "source": "expert_annotation_dataset",
            },
            "TM (Expert)": {
                "label": "TM (Expert)",
                "description": "Expert-reviewed transmembrane-segment count from the static expert annotation reference file.",
                "source": "expert_annotation_dataset",
            },
            "Group (Predicted)": {
                "label": "Group (Predicted)",
                "description": "Live broad-group prediction produced by the current MetaMP semi-supervised classifier.",
                "source": "metamp_ml_classifier",
            },
            "Group (MPstruc)": {
                "label": "Group (MPstruc)",
                "description": "Broad structural group currently available from MPstruc-linked MetaMP data after standardization.",
                "source": "mpstruc",
            },
            "Group (OPM)": {
                "label": "Group (OPM)",
                "description": "Broad structural group currently available from OPM-linked MetaMP data after standardization.",
                "source": "opm",
            },
            "TMbed TM Count": {
                "label": "TMbed TM Count",
                "description": "Live TM-segment count predicted by TMbed.",
                "source": "tmbed",
            },
            "DeepTMHMM TM Count": {
                "label": "DeepTMHMM TM Count",
                "description": "Live TM-segment count predicted by DeepTMHMM.",
                "source": "deeptmhmm",
            },
            "Phobius TM Count": {
                "label": "Phobius TM Count",
                "description": "Imported or computed TM-segment count predicted by Phobius.",
                "source": "phobius",
            },
            "TOPCONS TM Count": {
                "label": "TOPCONS TM Count",
                "description": "Imported or computed TM-segment count predicted by TOPCONS.",
                "source": "topcons",
            },
            "CCTOP TM Count": {
                "label": "CCTOP TM Count",
                "description": "Imported or computed TM-segment count predicted by CCTOP.",
                "source": "cctop",
            },
            "normalized_tm_predictions": {
                "label": "Normalized TM Predictions",
                "description": "Per-protein predictor summaries joined from membrane_protein_tmalphafold_predictions, including TMAlphaFold methods, TMDET, and MetaMP-local TMbed rows.",
                "source": "metamp",
            },
            "tm_prediction_overview": {
                "label": "TM Prediction Overview",
                "description": "Compact overview of available normalized TM predictors and their derived topology labels for quick frontend comparison.",
                "source": "metamp",
            },
            "tm_prediction_summary_card": {
                "label": "TM Prediction Summary Card",
                "description": "Frontend-ready preferred topology summary assembled from normalized TM predictor rows, including available methods and the preferred compact topology label.",
                "source": "metamp",
            },
            "preferred_tm_prediction_method": {
                "label": "Preferred TM Prediction Method",
                "description": "Preferred membrane-topology method selected from the normalized TM predictor store for compact frontend displays.",
                "source": "metamp",
            },
            "preferred_tm_prediction_count": {
                "label": "Preferred TM Prediction Count",
                "description": "Preferred transmembrane-segment count selected from the normalized TM predictor store for compact frontend displays.",
                "source": "metamp",
            },
            "preferred_tm_compact_label": {
                "label": "Preferred TM Compact Label",
                "description": "Short paper-facing topology label such as '6TM alpha-helical membrane chain, N-in / C-in' derived from the preferred normalized TM prediction.",
                "source": "metamp",
            },
            "preferred_tm_orientation_label": {
                "label": "Preferred TM Orientation Label",
                "description": "Orientation summary such as 'N-in / C-in' derived from the preferred normalized TM prediction.",
                "source": "metamp",
            },
            "Resolved Group": {
                "label": "Resolved Group",
                "description": "Priority-based group label selected from adjudicated, expert, predicted, OPM, then MPstruc values.",
                "source": "metamp",
            },
            "Group Disagreement": {
                "label": "Group Disagreement",
                "description": "Whether available broad group labels disagree across sources.",
                "source": "metamp",
            },
            "TM Disagreement": {
                "label": "TM Disagreement",
                "description": "Whether available TM counts disagree across expert and predictor sources.",
                "source": "metamp",
            },
            "TM Boundary Disagreement": {
                "label": "TM Boundary Disagreement",
                "description": "Whether available TM-region counts disagree across OPM and predictor sources.",
                "source": "metamp",
            },
            "structure_context": {
                "label": "Structure Context",
                "description": "Chain, entity, and assembly context used to qualify topology interpretation.",
                "source": "pdb",
            },
            "provenance": {
                "label": "Provenance",
                "description": "Source-trace information showing which values come from which resource and when the underlying files were last updated.",
                "source": "metamp",
            },
        }
        for key, value in cls.scientific_flag_glossary().items():
            glossary[key] = {
                **value,
                "label": value.get("label", key),
                "description": value.get("description"),
                "interpretation": value.get("interpretation"),
                "source": value.get("source", "metamp"),
            }
        return glossary

    @classmethod
    def record_field_glossary_keys(cls):
        return list(cls.field_glossary().keys()) + [
            "recommended_for_sequence_topology_benchmark",
            "context_dependent_topology",
            "non_canonical_membrane_case",
            "multichain_context",
            "obsolete_or_replaced",
        ]

    @classmethod
    def build_metadata_payload(cls):
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source_catalog": cls.source_catalog(),
            "scientific_flags": cls.scientific_flag_glossary(),
            "field_glossary": cls.field_glossary(),
            "review_status_options": sorted(DiscrepancyReviewService.VALID_STATUSES),
            "ui_sections": {
                "overview": "Record-level canonical identifiers, resolved labels, and review status.",
                "annotation_sources": "Source-by-source cards showing whether a source is static, live-generated, and present for the current record.",
                "scientific_flags": "Heuristic scientific caution flags that support interpretation rather than replace expert judgment.",
                "structure_context": "Chain and assembly context used to qualify record interpretation.",
                "tm_boundaries": "Source-specific TM-boundary rows with start, end, length, chain, and source name.",
                "benchmark_context": "Cautionary notes describing how predictor outputs should be interpreted.",
                "live_status": "Latest known modification times for the files behind live and reference annotations.",
            },
        }


class DashboardRegressionValidationService:
    @classmethod
    def run_checks(cls):
        checks = [
            cls._validate_egfr_search(),
            cls._validate_replacement_resolution(),
            cls._validate_record_enrichment(),
            cls._validate_structure_context(),
            cls._validate_tm_boundary_enrichment(),
            cls._validate_benchmark_export(),
        ]
        passed = all(check["passed"] for check in checks)
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "passed": passed,
            "check_count": len(checks),
            "failed_check_count": sum(1 for check in checks if not check["passed"]),
            "checks": checks,
        }

    @classmethod
    def _validate_egfr_search(cls):
        try:
            results = search_merged_databases("EGFR", limit=5)
        except Exception as exc:
            return cls._failed_check("egfr_search", f"Search raised an exception: {exc}")

        if not results:
            return cls._failed_check(
                "egfr_search",
                "Canonical search returned no EGFR-related results.",
            )

        if not any(
            "gene_contains_exact" in (record.get("match_reasons") or [])
            or "EGFR" in json.dumps(record, default=str).upper()
            for record in results
        ):
            return cls._failed_check(
                "egfr_search",
                "Search returned results, but none clearly matched EGFR.",
                details={"result_count": len(results)},
            )

        return {
            "name": "egfr_search",
            "passed": True,
            "details": {
                "result_count": len(results),
                "top_result": results[0].get("pdb_code"),
                "top_match_reasons": results[0].get("match_reasons"),
            },
        }

    @classmethod
    def _validate_replacement_resolution(cls):
        replaced_record = DashboardAnnotationDatasetService.get_record("5TSI")
        canonical_record = DashboardAnnotationDatasetService.get_record("5UAR")

        if not replaced_record or replaced_record.get("canonical_pdb_code") != "5UAR":
            return cls._failed_check(
                "replacement_resolution",
                "Legacy replaced entry 5TSI did not resolve to canonical code 5UAR.",
                details={"record_found": bool(replaced_record)},
            )

        if not canonical_record or canonical_record.get("canonical_pdb_code") != "5UAR":
            return cls._failed_check(
                "replacement_resolution",
                "Canonical entry 5UAR did not resolve cleanly.",
                details={"record_found": bool(canonical_record)},
            )

        return {
            "name": "replacement_resolution",
            "passed": True,
            "details": {
                "legacy_pdb_code": replaced_record.get("legacy_pdb_code"),
                "replacement_pdb_code": replaced_record.get("replacement_pdb_code"),
                "canonical_pdb_code": replaced_record.get("canonical_pdb_code"),
            },
        }

    @classmethod
    def _validate_record_enrichment(cls):
        record = DashboardAnnotationDatasetService.get_record("6A69")
        if not record:
            return cls._failed_check(
                "record_enrichment",
                "Expected enriched record 6A69 was not found.",
            )

        provenance = record.get("provenance") or {}
        sources_present = provenance.get("sources_present") or []
        structure_context = record.get("structure_context") or {}

        if len(sources_present) < 3 or not structure_context:
            return cls._failed_check(
                "record_enrichment",
                "6A69 did not include the expected provenance and structural enrichment.",
                details={
                    "sources_present": sources_present,
                    "has_structure_context": bool(structure_context),
                },
            )

        return {
            "name": "record_enrichment",
            "passed": True,
            "details": {
                "sources_present": sources_present,
                "chain_count": structure_context.get("chain_count"),
            },
        }

    @classmethod
    def _validate_structure_context(cls):
        record = DashboardAnnotationDatasetService.get_record("6CB2")
        structure_context = (record or {}).get("structure_context") or {}
        chain_count = structure_context.get("chain_count") or 0

        if chain_count < 2:
            return cls._failed_check(
                "structure_context",
                "6CB2 should expose multichain structure context.",
                details={"chain_count": chain_count},
            )

        return {
            "name": "structure_context",
            "passed": True,
            "details": {
                "chain_ids": structure_context.get("chain_ids"),
                "chain_count": chain_count,
                "entity_ids": structure_context.get("entity_ids"),
            },
        }

    @classmethod
    def _validate_tm_boundary_enrichment(cls):
        record = DashboardAnnotationDatasetService.get_record("1O5W")
        tm_regions = record.get("opm_tm_regions") if record else None
        discrepancy_review = record.get("discrepancy_review") if record else None

        if not record or not tm_regions:
            return cls._failed_check(
                "tm_boundary_enrichment",
                "1O5W did not expose OPM TM boundary data.",
                details={"record_found": bool(record)},
            )

        return {
            "name": "tm_boundary_enrichment",
            "passed": True,
            "details": {
                "has_opm_tm_regions": bool(tm_regions),
                "review_status": (discrepancy_review or {}).get("status"),
            },
        }

    @classmethod
    def _validate_benchmark_export(cls):
        try:
            payload = DiscrepancyBenchmarkExportService.build_download_payload(
                export_format="csv",
                include_all=False,
            )
        except Exception as exc:
            return cls._failed_check(
                "benchmark_export",
                f"Benchmark export generation failed: {exc}",
            )

        metadata = payload.get("metadata") or {}
        row_count = metadata.get("row_count") or 0
        csv_content = payload.get("content") or ""

        if row_count <= 0 or "pdb_code" not in csv_content:
            return cls._failed_check(
                "benchmark_export",
                "Benchmark export payload was generated but did not contain expected data.",
                details={"row_count": row_count},
            )

        return {
            "name": "benchmark_export",
            "passed": True,
            "details": {
                "row_count": row_count,
                "included_row_count": metadata.get("included_row_count"),
                "filename": payload.get("filename"),
            },
        }

    @staticmethod
    def _failed_check(name, message, details=None):
        return {
            "name": name,
            "passed": False,
            "message": message,
            "details": details or {},
        }


class DashboardPageService:
    INTERNAL_SUMMARY_COLUMNS = {"id", "created_at", "updated_at"}
    QUANTITATIVE_SUMMARY_COLUMNS = set(
        list(cell_columns) + list(rcsb_entries) + list(quantitative_array_column)
    )
    MAX_PAYLOAD_CACHE_ENTRIES = 12
    MAX_DATAFRAME_CACHE_ENTRIES = 12
    _payload_cache = {}
    _dataframe_cache = {}
    _empty_dashboard_data = {
        "items": [],
        "page": 1,
        "per_page": 0,
        "total_items": 0,
        "total_pages": 0,
        "total_columns": 0,
        "total_rows": 0,
    }

    @staticmethod
    def _data_service():
        from src.MP.services import DataService

        return DataService

    @staticmethod
    def build_welcome_page_payload():
        from src.services.basic_plots import data_flow

        all_data = DashboardPageService._get_database_snapshot()
        unique_data = get_table_as_dataframe_download(
            table_name="membrane_proteins",
            columns=["group", "bibliography_year"],
            filter_column="is_master_protein",
            filter_value="MasterProtein",
        ).get("data", [{}])

        return {
            "all_data": all_data["all_data"],
            "latest_date": DashboardPageService._extract_latest_update(
                all_data["all_data"]
            ),
            "all_data_uniprot": all_data["all_data_uniprot"],
            "all_data_opm": all_data["all_data_opm"],
            "all_data_mpstruc": all_data["all_data_mpstruc"],
            "all_data_pdb": all_data["all_data_pdb"],
            "unique_trend": data_flow(
                unique_data, "Unique membrane proteins from MPstruc"
            ),
            "dashboard_metadata": DashboardFieldMetadataService.build_metadata_payload(),
        }

    @staticmethod
    def build_about_payload():
        from src.services.pages import Pages

        pages = Pages(pd.DataFrame())
        all_data = DashboardPageService._get_database_snapshot()
        data_service = DashboardPageService._data_service()

        return {
            "trends_by_database_year": pages.view_trends_by_database_year_default(),
            "get_master_proteins": data_service.get_data_by_column_search(
                column_name="is_master_protein",
                value="MasterProtein",
                page=1,
                per_page=1,
            ),
            "all_data_uniprot": all_data["all_data_uniprot"],
            "all_data_mpstruc": all_data["all_data_mpstruc"],
            "all_data_opm": all_data["all_data_opm"],
            "all_data_pdb": all_data["all_data_pdb"],
            "all_data": all_data["all_data"],
            "dashboard_metadata": DashboardFieldMetadataService.build_metadata_payload(),
            "metamp_added_value": DashboardPageService.build_added_value_payload(),
            "case_studies": DashboardPageService.build_case_studies_payload(),
        }

    @staticmethod
    def build_about_summary_payload():
        all_data = DashboardPageService._get_database_snapshot()
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "rows": [
                DashboardPageService._build_about_summary_row(
                    "MPstruc",
                    "membrane_protein_mpstruct",
                    all_data["all_data_mpstruc"].get("total_rows", 0),
                ),
                DashboardPageService._build_about_summary_row(
                    "PDB",
                    "membrane_protein_pdb",
                    all_data["all_data_pdb"].get("total_rows", 0),
                ),
                DashboardPageService._build_about_summary_row(
                    "OPM",
                    "membrane_protein_opm",
                    all_data["all_data_opm"].get("total_rows", 0),
                ),
                DashboardPageService._build_about_summary_row(
                    "UniProtKB",
                    "membrane_protein_uniprot",
                    all_data["all_data_uniprot"].get("total_rows", 0),
                ),
                DashboardPageService._build_about_summary_row(
                    "MetaMP",
                    "membrane_proteins",
                    all_data["all_data"].get("total_rows", 0),
                    highlight=True,
                ),
            ],
        }

    @staticmethod
    def build_added_value_payload():
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "items": [
                {
                    "key": "provenance_preserving_harmonization",
                    "title": "Provenance-preserving harmonization",
                    "why_it_matters": "MetaMP keeps source-specific annotations visible instead of silently collapsing them.",
                    "evidence": [
                        "/api/v1/records/<pdb_code>",
                        "/api/v1/dashboard-metadata",
                    ],
                },
                {
                    "key": "discrepancy_review",
                    "title": "Cross-source discrepancy review",
                    "why_it_matters": "Conflicting group and topology annotations can be surfaced, reviewed, and exported.",
                    "evidence": [
                        "/api/v1/discrepancy-reviews",
                        "/api/v1/discrepancy-benchmark/export",
                    ],
                },
                {
                    "key": "live_predictor_overlay",
                    "title": "Live predictor overlay",
                    "why_it_matters": "Predicted group labels and TM counts are refreshed dynamically instead of being fixed in a static annotation file.",
                    "evidence": [
                        "/api/v1/records/<pdb_code>",
                        "/api/v1/discrepancy-reviews/<pdb_code>",
                    ],
                },
                {
                    "key": "scientific_caveats",
                    "title": "Scientific caveat flags",
                    "why_it_matters": "State dependence, soluble-to-membrane transitions, and multichain ambiguity are exposed explicitly to avoid overinterpretation.",
                    "evidence": [
                        "/api/v1/dashboard-metadata",
                        "/api/v1/records/<pdb_code>",
                    ],
                },
                {
                    "key": "replacement_aware_resolution",
                    "title": "Replacement-aware resolution",
                    "why_it_matters": "Legacy or replaced entries are linked to canonical records so users can inspect both history and current interpretation.",
                    "evidence": [
                        "/api/v1/records/5TSI",
                        "/api/v1/search-merged-db?q=5TSI",
                    ],
                },
            ],
        }

    @staticmethod
    def build_case_studies_payload():
        cases = []
        case_builders = [
            DashboardPageService._build_egfr_case_study,
            DashboardPageService._build_replacement_case_study,
            DashboardPageService._build_state_dependent_case_study,
            DashboardPageService._build_tm_boundary_case_study,
            DashboardPageService._build_multichain_case_study,
        ]
        for builder in case_builders:
            case = builder()
            if case:
                cases.append(case)
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "items": cases,
        }

    @staticmethod
    def _build_egfr_case_study():
        results = search_merged_databases("EGFR", limit=3)
        if not results:
            return None
        top_result = results[0]
        return {
            "key": "egfr_search",
            "title": "EGFR retrieval",
            "summary": "Search should return EGFR-related records directly rather than unrelated near matches.",
            "evidence": {
                "query": "EGFR",
                "top_result_pdb_code": top_result.get("pdb_code"),
                "match_reasons": top_result.get("match_reasons"),
            },
            "endpoint": "/api/v1/search-merged-db?q=EGFR",
        }

    @staticmethod
    def _build_replacement_case_study():
        legacy_record = DashboardAnnotationDatasetService.get_record("5TSI")
        canonical_record = DashboardAnnotationDatasetService.get_record("5UAR")
        if not legacy_record or not canonical_record:
            return None
        return {
            "key": "replacement_resolution",
            "title": "Replacement-aware record resolution",
            "summary": "Legacy entries remain searchable while still resolving to the canonical replacement record.",
            "evidence": {
                "legacy_pdb_code": legacy_record.get("legacy_pdb_code"),
                "canonical_pdb_code": legacy_record.get("canonical_pdb_code"),
                "replacement_pdb_code": legacy_record.get("replacement_pdb_code"),
            },
            "endpoint": "/api/v1/records/5TSI",
        }

    @staticmethod
    def _build_state_dependent_case_study():
        record = DashboardAnnotationDatasetService.get_record("1PFO")
        if not record:
            return None
        flags = ((record.get("scientific_assessment") or {}).get("flags") or {})
        return {
            "key": "state_dependent_caveat",
            "title": "State-dependent benchmark caveat",
            "summary": "Sequence-only predictor outputs are qualified for pore-forming or soluble-to-membrane transition proteins.",
            "evidence": {
                "pdb_code": record.get("canonical_pdb_code"),
                "context_dependent_topology": flags.get("context_dependent_topology"),
                "context_reasons": (
                    ((record.get("scientific_assessment") or {}).get("details") or {}).get("context_reasons")
                    or []
                ),
                "recommended_for_sequence_topology_benchmark": (
                    (record.get("scientific_assessment") or {}).get(
                        "recommended_for_sequence_topology_benchmark"
                    )
                ),
            },
            "endpoint": "/api/v1/records/1PFO",
        }

    @staticmethod
    def _build_tm_boundary_case_study():
        record = DashboardAnnotationDatasetService.get_record("1O5W")
        if not record:
            return None
        tm_rows = (((record.get("ui_sections") or {}).get("tm_boundaries") or {}).get("rows") or [])
        return {
            "key": "tm_boundary_display",
            "title": "TM boundary display",
            "summary": "The record view now exposes explicit TM boundary rows instead of only summary counts.",
            "evidence": {
                "pdb_code": record.get("canonical_pdb_code"),
                "boundary_row_count": len(tm_rows),
                "sources_present": sorted({row.get("source") for row in tm_rows if row.get("source")}),
            },
            "endpoint": "/api/v1/records/1O5W",
        }

    @staticmethod
    def _build_multichain_case_study():
        record = DashboardAnnotationDatasetService.get_record("6CB2")
        if not record:
            return None
        structure_context = ((record.get("ui_sections") or {}).get("structure_context") or {})
        return {
            "key": "multichain_context",
            "title": "Chain and assembly context",
            "summary": "Records with multichain context expose chain and assembly summaries so users can interpret topology more cautiously.",
            "evidence": {
                "pdb_code": record.get("canonical_pdb_code"),
                "chain_count": structure_context.get("chain_count"),
                "chain_ids": structure_context.get("chain_ids"),
                "assembly_ids": structure_context.get("assembly_ids"),
            },
            "endpoint": "/api/v1/records/6CB2",
        }

    @staticmethod
    def build_dashboard_payload(get_header, conf, request_for_group, first_leveled_width):
        from src.services.basic_plots import (
            create_combined_chart_cumulative_growth,
            group_data_by_methods,
        )
        from src.services.pages import Pages

        if get_header != "none":
            return {"data": get_items()}

        default_display = "taxonomic_domain"
        request_for_group_list = request_for_group or default_display
        unique_group_list = convert_to_list_of_dicts(request_for_group_list)
        requested_columns = {
            "group",
            "taxonomic_domain",
            "bibliography_year",
            "rcsentinfo_experimental_method",
            "rcsentinfo_software_programs_combined",
        }
        requested_columns.update(graph["name"] for graph in unique_group_list)
        table_df = DashboardPageService._get_membrane_dataframe(
            sorted(requested_columns)
        )

        cache_key = DashboardPageService._build_payload_cache_key(
            "dashboard",
            DashboardPageService._get_membrane_table_version(),
            {
                "conf": conf,
                "group_key": request_for_group_list,
                "width": int(first_leveled_width),
            },
        )
        cached = DashboardPageService._payload_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        pages = Pages(table_df.copy(deep=False))
        parsed_conf = json.loads(conf)
        trend = create_combined_chart_cumulative_growth(
            table_df.copy(deep=True), int(first_leveled_width)
        )
        trend_by_method = group_data_by_methods(
            table_df[["bibliography_year", "rcsentinfo_experimental_method"]].copy(
                deep=True
            )
        )
        group_graph_array = []

        for key, graph in enumerate(unique_group_list):
            graph_conf = dict(parsed_conf)
            graph_conf["x-angle"] = graph["x-angle"]
            group_graph, _ = pages.view_dashboard(
                get_query_params=graph["name"],
                conf=graph_conf,
                ranges_={},
            )
            group_graph_array.append(
                {
                    "chart_obj": group_graph,
                    "id": f"graph{key}",
                    "name": f"graph {key}",
                    "groups": graph,
                }
            )

        payload = {
            "data": dict(DashboardPageService._empty_dashboard_data),
            "trend": trend,
            "trend_by_method": trend_by_method,
            "group_graph_array": group_graph_array,
            "membrane_group_chart": create_grouped_bar_chart(
                table_df[["group"]].copy(deep=True)
            ),
        }
        DashboardPageService._remember_cache_item(
            DashboardPageService._payload_cache,
            cache_key,
            payload,
            DashboardPageService.MAX_PAYLOAD_CACHE_ENTRIES,
        )
        return payload

    @staticmethod
    def build_inconsistencies_payload(chart_width):
        from src.Training.services import (
            aggregate_inconsistencies,
            create_visualization,
            transform_dataframe,
        )

        all_data, _, _, _, _ = DashboardPageService._data_service().get_data_from_DB()
        selected_columns = [
            "pdb_code",
            "famsupclasstype_type_name",
            "family_superfamily_classtype_name",
            "group",
            "bibliography_year",
            "rcsentinfo_experimental_method",
        ]
        combined = all_data[selected_columns].copy()
        complete_years = pd.to_numeric(
            all_data.get("bibliography_year"),
            errors="coerce",
        ).dropna().astype(int).tolist()
        combined.dropna(inplace=True)
        transformed_data = transform_dataframe(
            aggregate_inconsistencies(combined, complete_years=complete_years)
        )
        chart_with_table = create_visualization(transformed_data, chart_width)
        return {"inconsistencies": convert_chart(chart_with_table)}

    @staticmethod
    def build_dashboard_others_payload():
        from src.services.pages import Pages

        pages = Pages(pd.DataFrame())
        table_df = DashboardPageService._get_membrane_dataframe(
            [
                "bibliography_year",
                "processed_resolution",
                "rcsentinfo_experimental_method",
            ]
        )
        return {
            "trends_by_database_year": pages.view_trends_by_database_year(),
            "mean_resolution_by_year": pages.average_resolution_over_years(table_df),
        }

    @staticmethod
    def build_dashboard_map_payload():
        from src.services.Helpers.helper import get_data_by_countries
        from src.services.pages import Pages

        cache_key = DashboardPageService._build_payload_cache_key(
            "dashboard_map",
            DashboardPageService._get_membrane_table_version(),
            {},
        )
        cached = DashboardPageService._payload_cache.get(cache_key)
        if cached is not None:
            return cached

        table_df = DashboardPageService._get_membrane_dataframe(
            ["id", "bibliography_year", "rcsb_primary_citation_country"]
        )
        pages = Pages(table_df.copy(deep=False))
        map_data, map_chart = pages.getMap()
        country_data = get_data_by_countries(table_df.copy(deep=False)).to_dict(
            orient="records"
        )
        payload = {
            "map": map_chart,
            "europe_map": {},
            "map_data": map_data,
            "get_country_data": country_data,
            "release_by_country": pages.releasedStructuresByCountries(map_data),
        }
        DashboardPageService._remember_cache_item(
            DashboardPageService._payload_cache,
            cache_key,
            payload,
            DashboardPageService.MAX_PAYLOAD_CACHE_ENTRIES,
        )
        return payload

    @staticmethod
    def _get_membrane_dataframe(columns):
        normalized_columns = tuple(sorted(set(columns)))
        version = DashboardPageService._get_membrane_table_version()
        cache_key = ("membrane_df", version, normalized_columns)
        cached = DashboardPageService._dataframe_cache.get(cache_key)
        if cached is not None:
            return cached.copy(deep=False)

        dataframe = get_table_as_dataframe_with_specific_columns(
            "membrane_proteins",
            list(normalized_columns),
        )
        DashboardPageService._remember_cache_item(
            DashboardPageService._dataframe_cache,
            cache_key,
            dataframe,
            DashboardPageService.MAX_DATAFRAME_CACHE_ENTRIES,
        )
        return dataframe.copy(deep=False)

    @staticmethod
    def _remember_cache_item(cache_store, key, value, max_entries):
        if key in cache_store:
            cache_store.pop(key)
        cache_store[key] = value
        while len(cache_store) > max_entries:
            oldest_key = next(iter(cache_store))
            cache_store.pop(oldest_key, None)

    @staticmethod
    def _get_membrane_table_version():
        table = _get_reflected_table("membrane_proteins")
        query = select(
            func.count().label("row_count"),
            func.max(table.c.updated_at).label("max_updated_at"),
        )
        with db.engine.connect() as connection:
            row = connection.execute(query).mappings().first()

        row_count = int((row or {}).get("row_count") or 0)
        max_updated_at = (row or {}).get("max_updated_at")
        return f"{row_count}:{max_updated_at}"

    @staticmethod
    def _build_payload_cache_key(name, version, params):
        encoded = json.dumps(params, sort_keys=True, default=str)
        return f"{name}:{version}:{encoded}"

    @staticmethod
    def _build_about_summary_row(label, table_name, observations, highlight=False):
        columns = DashboardPageService._get_summary_columns(table_name)
        quantitative_count = sum(
            1 for column in columns if DashboardPageService._is_quantitative_summary_column(column)
        )
        attributes_count = len(columns)
        nominal_count = max(attributes_count - quantitative_count, 0)
        unique_observations = DashboardPageService._get_about_observation_count(table_name)
        row_count = int(observations or 0)
        return {
            "database": label,
            "table_name": table_name,
            "observations": unique_observations,
            "source_rows": row_count,
            "count_basis": "distinct_pdb_code" if unique_observations != row_count else "rows",
            "attributes": attributes_count,
            "nominal": nominal_count,
            "quantitative": quantitative_count,
            "highlight": highlight,
        }

    @staticmethod
    def _get_summary_columns(table_name):
        table = _get_reflected_table(table_name)
        return [
            column
            for column in table.columns
            if column.name not in DashboardPageService.INTERNAL_SUMMARY_COLUMNS
        ]

    @staticmethod
    def _is_quantitative_summary_column(column):
        if column.name in DashboardPageService.QUANTITATIVE_SUMMARY_COLUMNS:
            return True

        try:
            python_type = column.type.python_type
        except (NotImplementedError, AttributeError):
            return False

        return issubclass(python_type, Number) and not issubclass(python_type, bool)

    @staticmethod
    def _get_about_observation_count(table_name):
        table = _get_reflected_table(table_name)
        if hasattr(table.c, "pdb_code"):
            query = select(func.count(func.distinct(table.c.pdb_code)))
        else:
            query = select(func.count()).select_from(table)

        with db.engine.connect() as connection:
            return int(connection.execute(query).scalar() or 0)

    @staticmethod
    def _get_database_snapshot():
        data_service = DashboardPageService._data_service()
        return {
            "all_data": data_service.get_data_by_column_search(
                column_name=None,
                value=None,
                page=1,
                per_page=1,
            ),
            "all_data_mpstruc": data_service.get_data_by_column_search(
                table="membrane_protein_mpstruct",
                column_name=None,
                value=None,
                page=1,
                per_page=1,
            ),
            "all_data_pdb": data_service.get_data_by_column_search(
                table="membrane_protein_pdb",
                column_name=None,
                value=None,
                page=1,
                per_page=1,
            ),
            "all_data_opm": data_service.get_data_by_column_search(
                table="membrane_protein_opm",
                column_name=None,
                value=None,
                page=1,
                per_page=1,
            ),
            "all_data_uniprot": data_service.get_data_by_column_search(
                table="membrane_protein_uniprot",
                column_name=None,
                value=None,
                page=1,
                per_page=1,
                distinct_column="pdb_code",
            ),
        }

    @staticmethod
    def _extract_latest_update(all_data_payload):
        rows = all_data_payload.get("data", [])
        timestamps = [
            datetime.strptime(row["updated_at_readable"], "%Y-%m-%d %H:%M:%S")
            for row in rows
            if row.get("updated_at_readable")
        ]
        if not timestamps:
            return None
        return max(timestamps).strftime("%Y-%m-%d %H:%M:%S")


class DashboardQueryRepository:
    @staticmethod
    def build_base_query():
        membrane_protein = MembraneProteinData
        opm = OPM
        uniprot = Uniprot
        query = (
            db.session.query(membrane_protein, opm, uniprot)
            .select_from(membrane_protein)
            .outerjoin(opm, opm.pdb_code == membrane_protein.pdb_code)
            .outerjoin(uniprot, uniprot.pdb_code == membrane_protein.pdb_code)
        )
        return query, membrane_protein, opm, uniprot


class DashboardRecordPresenter:
    EXCLUDED_COLUMNS = {"created_at", "updated_at", "id"}

    @classmethod
    def build_export_dataframe(cls, items):
        if not items:
            return pd.DataFrame()

        membrane_protein_columns = cls._get_column_names(MembraneProteinData)
        opm_columns = cls._get_column_names(OPM)
        uniprot_columns = cls._get_column_names(Uniprot)
        all_columns = (
            [(f"membrane_{col}", col) for col in membrane_protein_columns]
            + [(f"opm_{col}", col) for col in opm_columns]
            + [(f"uniprot_{col}", col) for col in uniprot_columns]
        )

        records = []
        for mp_data, op_data, up_data in items:
            record = {}
            for prefixed_col, col in all_columns:
                if col in membrane_protein_columns:
                    record[prefixed_col] = getattr(mp_data, col, None)
                elif col in opm_columns:
                    record[prefixed_col] = getattr(op_data, col, None) if op_data else None
                elif col in uniprot_columns:
                    record[prefixed_col] = getattr(up_data, col, None) if up_data else None
            records.append(record)

        return pd.DataFrame(records)

    @classmethod
    def build_paginated_result(cls, paginated_items, total_item_count, membrane_protein, opm, uniprot):
        total_columns = cls._count_visible_columns(membrane_protein, opm, uniprot)
        items_list = [
            OrderedDict(
                [
                    ("id", mp_data.id),
                    ("name", mp_data.name),
                    ("group", mp_data.group),
                    ("species", mp_data.species),
                    ("subgroup", mp_data.subgroup),
                    ("pdb_code", mp_data.pdb_code),
                    ("uniprot_id", up_data.uniprot_id if up_data else None),
                    ("resolution", mp_data.resolution),
                    ("exptl_method", mp_data.exptl_method),
                    ("taxonomic_domain", mp_data.taxonomic_domain),
                    ("expressed_in_species", mp_data.expressed_in_species),
                    ("rcsentinfo_experimental_method", mp_data.rcsentinfo_experimental_method),
                    ("comment_disease_name", up_data.comment_disease_name if up_data else None),
                ]
            )
            for mp_data, op_data, up_data in paginated_items.items
        ]
        return {
            "items": items_list,
            "page": paginated_items.page,
            "per_page": paginated_items.per_page,
            "total_items": paginated_items.total,
            "total_pages": paginated_items.pages,
            "total_columns": total_columns,
            "total_rows": total_item_count,
        }

    @classmethod
    def _count_visible_columns(cls, membrane_protein, opm, uniprot):
        return sum(
            len(
                [
                    column
                    for column in model.__table__.columns
                    if column.name not in cls.EXCLUDED_COLUMNS
                ]
            )
            for model in (membrane_protein, opm, uniprot)
        )

    @staticmethod
    def _get_column_names(model):
        return [column.name for column in model.__table__.columns]

def get_all_items():
    return MembraneProteinData.query


def _normalize_filter_text(value):
    normalized = str(value or "").strip()
    if normalized.lower() in {"", "all", "none", "null", "undefined", "nan"}:
        return ""
    return normalized


def _collect_search_columns(model, column_names):
    return [
        getattr(model, column_name)
        for column_name in column_names
        if hasattr(model, column_name)
    ]

def apply_search_and_filter(query, search_terms, MP, OP, UP):
    data = search_terms.get("search_terms", {})
    
    # Get all values
    search_term = _normalize_filter_text(data.get('search_term', ''))
    group = _normalize_filter_text(data.get('group', ''))
    subgroup = _normalize_filter_text(data.get('subgroup', ''))
    taxonomic_domain = _normalize_filter_text(data.get('taxonomic_domain', ''))
    experimental_method = _normalize_filter_text(data.get('experimental_methods', ''))
    molecular_function = _normalize_filter_text(data.get('molecular_function', ''))
    cellular_component = _normalize_filter_text(data.get('cellular_component', ''))
    biological_process = _normalize_filter_text(data.get('biological_process', ''))
    family_name = _normalize_filter_text(data.get('family_name', ''))
    species = _normalize_filter_text(data.get('species', ''))
    membrane_name = _normalize_filter_text(data.get('membrane_name', ''))
    super_family = _normalize_filter_text(data.get('super_family', ''))
    superfamily_classtype_name = _normalize_filter_text(data.get('super_family_class_type', ''))

    conditions = []

    # Global search term
    if search_term:
        search_columns = (
            _collect_search_columns(MP, [
                "name",
                "pdb_code",
                "group",
                "subgroup",
                "species",
                "taxonomic_domain",
                "description",
                "related_pdb_entries",
                "struct_title",
                "rcsentinfo_experimental_method",
                "expressed_in_species",
            ])
            + _collect_search_columns(OP, [
                "family_name_cache",
                "species_name_cache",
                "membrane_name_cache",
                "family_superfamily_name",
                "family_superfamily_classtype_name",
                "famsupclasstype_type_name",
            ])
            + _collect_search_columns(UP, [
                "uniprot_id",
                "comment_disease",
                "comment_disease_name",
                "protein_recommended_name",
                "protein_alternative_name",
                "associated_genes",
                "molecular_function",
                "cellular_component",
                "biological_process",
                "keywords",
                "comment_function",
            ])
        )
        token_conditions = []
        for token in re.split(r"\s+", search_term):
            token = token.strip()
            if not token:
                continue
            token_conditions.append(
                or_(*[column.ilike(f"%{token}%") for column in search_columns])
            )
        if token_conditions:
            conditions.append(and_(*token_conditions))

    # MembraneProteinData filters
    if group:
        conditions.append(MP.group.ilike(f"%{group}%"))
    if subgroup:
        conditions.append(MP.subgroup.ilike(f"%{subgroup}%"))
    if taxonomic_domain:
        conditions.append(MP.taxonomic_domain.ilike(f"%{taxonomic_domain}%"))
    if experimental_method:
        conditions.append(MP.rcsentinfo_experimental_method.ilike(f"%{experimental_method}%"))

    # OPM filters
    if family_name:
        conditions.append(OP.family_name_cache.ilike(f"%{family_name}%"))
    if species:
        conditions.append(OP.species_name_cache.ilike(f"%{species}%"))
    if membrane_name:
        conditions.append(OP.membrane_name_cache.ilike(f"%{membrane_name}%"))
    if super_family:
        conditions.append(OP.family_superfamily_name.ilike(f"%{super_family}%"))
    if superfamily_classtype_name:
        conditions.append(OP.family_superfamily_classtype_name.ilike(f"%{superfamily_classtype_name}%"))

    # Uniprot filters
    if molecular_function:
        conditions.append(UP.molecular_function.ilike(f"%{molecular_function}%"))
    if cellular_component:
        conditions.append(UP.cellular_component.ilike(f"%{cellular_component}%"))
    if biological_process:
        conditions.append(UP.biological_process.ilike(f"%{biological_process}%"))

    if conditions:
        query = query.filter(and_(*conditions))

    # Print out the final SQL query
    # print(str(query.statement.compile(db.engine, compile_kwargs={"literal_binds": True})))

    return query


def apply_sorting(query, sort_by, sort_order, MP, OP, UP):
    if sort_by:
        # Determine which alias the sort column belongs to
        if hasattr(MP, sort_by):
            column = getattr(MP, sort_by)
        elif hasattr(OP, sort_by):
            column = getattr(OP, sort_by)
        elif hasattr(UP, sort_by):
            column = getattr(UP, sort_by)
        else:
            raise ValueError(f"Unknown sort_by column: {sort_by}")
        
        query = query.order_by(column.asc() if sort_order.lower() == 'asc' else column.desc())
    # print(query.all())
    return query

def get_columns_excluding(model, exclude_columns):
    return [column for column in model.__table__.columns if column.name not in exclude_columns]


def get_items(request=None):
    request = request or {}
    page = int(request.get("page", 1))
    per_page = int(request.get("per_page", 10))
    sort_by = request.get("sort_by", "id")
    sort_order = request.get("sort_order", "asc")
    download = request.get("download", "")
    query, MP, OP, UP = DashboardQueryRepository.build_base_query()

    query = apply_search_and_filter(query, request, MP, OP, UP)
    items = apply_sorting(query, sort_by, sort_order, MP, OP, UP)
    if download in ["csv", "xlsx"]:
        return DashboardRecordPresenter.build_export_dataframe(items.all())

    paginated_items = items.paginate(page=page, per_page=per_page, error_out=False)
    return extract_items_and_metadata(paginated_items, paginated_items.total, MP, OP, UP)
    
    
def extract_items_and_metadata(paginated_items, total_item_count = 0, MP=None, OP=None, UP=None):
    return DashboardRecordPresenter.build_paginated_result(
        paginated_items,
        total_item_count,
        MP,
        OP,
        UP,
    )

def get_table_as_dataframe(table_name):
    table = _get_reflected_table(table_name)
    return _read_sql_dataframe(select(table))

def get_table_as_dataframe_with_specific_columns(table_name, columns=None):
    table = _get_reflected_table(table_name)

    if columns:
        columns = [getattr(table.c, column) for column in columns if hasattr(table.c, column)]
    else:
        columns = [table]

    return _read_sql_dataframe(select(*columns))

def get_tables_as_dataframe(table_names, common_id_field):
    tables = [_get_reflected_table(table_name) for table_name in table_names]
    dataframes = [_read_sql_dataframe(select(table)) for table in tables]

    # Merge DataFrames based on the common ID field
    merged_df = pd.merge(dataframes[0], dataframes[1], on=common_id_field, how='outer')
    for df in dataframes[2:]:
        merged_df = pd.merge(merged_df, df, on=common_id_field, how='outer')

    return merged_df

class PaginatedQuery(Query):
    def paginate(self, page, per_page=10, error_out=True):
        if error_out and page < 1:
            abort(404)

        items = self.limit(per_page).offset((page - 1) * per_page).all()
        total = self.order_by(None).count()

        return {'items': items, 'total': total, 'page': page, 'per_page': per_page}

# Apply the PaginatedQuery to the SQLAlchemy session
db.session.query_class = PaginatedQuery

def get_table_as_dataframe_exception(table_name, filter_column=None, filter_value=None, page=1, per_page=10, distinct_column=None):
    page = int(page)
    try:
        per_page = int(per_page)
        table = _get_reflected_table(table_name)
        query = select(table)

        if filter_column and filter_value is not None:
            query = query.where(getattr(table.c, filter_column) == filter_value)

        paginated_query = query.limit(per_page).offset((page - 1) * per_page)
        df = _read_sql_dataframe(paginated_query)
        
        if 'updated_at' in df.columns:
            df['updated_at_readable'] = pd.to_datetime(df['updated_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'created_at' in df.columns:
            df['created_at_readable'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        date_columns = df.select_dtypes(include=['datetime64[ns]']).columns

        df = df.drop(columns=date_columns)
        df = df.where(pd.notnull(df), "")

        total_rows_query = select(func.count()).select_from(table)
        if filter_column and filter_value is not None:
            total_rows_query = total_rows_query.where(
                getattr(table.c, filter_column) == filter_value
            )

        if table_name == 'membrane_protein_uniprot':
            total_rows_query = total_rows_query.where(
                ~getattr(table.c, 'uniprot_id').like('%uniParcId:%')
            )
        
        if distinct_column:
            total_rows_query = total_rows_query.with_only_columns(
                func.count(func.distinct(getattr(table.c, distinct_column)))
            )

        total_rows = db.session.execute(total_rows_query).scalar()

        return {'data': df.to_dict(orient='records'), 'total_rows': total_rows, 'page': page, 'per_page': per_page}
    
    except SQLAlchemyError:
        logger.exception("Failed to fetch table dataframe for %s", table_name)
        return {'data': [], 'total_rows': 0, 'page': 10, 'per_page': 10}
    except ValueError:
        logger.exception("Invalid request while fetching table dataframe for %s", table_name)
        return {'data': [], 'total_rows': 0, 'page': 10, 'per_page': 10}
    except Exception:
        logger.exception("Unexpected error while fetching table dataframe for %s", table_name)
        return {'data': [], 'total_rows': 0, 'page': 10, 'per_page': 10}
    

def get_table_as_dataframe_download(table_name, columns=None, filter_column=None, filter_value=None):
    columns = columns or []
    table = _get_reflected_table(table_name)

    if columns:
        select_columns = []
        for column in columns:
            if column.endswith('.*'):
                sub_table_name = column.split('.')[0]
                sub_table = _get_reflected_table(sub_table_name)
                select_columns.extend(sub_table.columns)
            elif ' as ' in column:
                column_name, alias = column.split(' as ')
                column_obj = getattr(table.columns, column_name).label(alias)
                select_columns.append(column_obj)
            elif '.' in column:
                table_name, column_name = column.split('.')
                column_obj = getattr(_get_reflected_table(table_name).columns, column_name)
                select_columns.append(column_obj)
            else:
                if hasattr(table.columns, column):
                    select_columns.append(getattr(table.columns, column))
        query = select(*select_columns)
    else:
        query = select(table)

    # Add a filter condition if provided
    if filter_column and filter_value:
        if isinstance(filter_value, (list, tuple, set,)):
            query = query.where(getattr(table.columns, filter_column).in_(filter_value))
        else:
            query = query.where(getattr(table.columns, filter_column) == filter_value)

    df = _read_sql_dataframe(query)

    # Calculate the total_rows separately
    total_rows = None
    if not filter_column and not filter_value:
        # Execute a count query without any filter condition
        total_rows = db.session.execute(select([func.count()]).select_from(table)).scalar()
    elif filter_column:
        total_rows = db.session.execute(select([func.count()]).select_from(table).where(getattr(table.columns, filter_column) == filter_value)).scalar()

    return {'data': df, 'total_rows': total_rows}

def getMPstructDB():
    return [column.name for column in _get_reflected_table("membrane_protein_mpstruct").columns]
    
def getPDBDB():
    return [column.name for column in _get_reflected_table("membrane_protein_pdb").columns]

def getOPMDB():
    return [column.name for column in _get_reflected_table("membrane_protein_opm").columns]
    
def getUniprotDB():
    return [column.name for column in _get_reflected_table("membrane_protein_uniprot").columns]

def preprocessVariables(variables=None):
    variables = variables or []
    formatted_strings = []

    for string in variables:
        # Split the string by underscores
        words = string.split('_')

        # Remove the first word if the number of words is greater than 4
        if len(words) > 4:
            words = words[1:]

        # Capitalize each remaining word and join them back into a string
        formatted_string = ' '.join(word.capitalize() for word in words)
        formatted_strings.append(formatted_string)

    return formatted_strings

def all_merged_databases():
    local_dataset = DashboardAnnotationDatasetService._load_local_merged_dataset()
    if not local_dataset.empty:
        return local_dataset.copy(deep=False)

    table_names = ['membrane_proteins', 'membrane_protein_opm']
    result_df = get_tables_as_dataframe(table_names, "pdb_code")
    result_df_uniprot = get_table_as_dataframe("membrane_protein_uniprot")

    common = set(result_df.columns) - {"pdb_code"} & set(result_df_uniprot.columns)
    right_pruned = result_df_uniprot.drop(columns=list(common))

    return pd.merge(right=result_df, left=right_pruned, on="pdb_code", how="outer")


def _normalize_search_value(value):
    return str(value or "").strip().upper()


def _normalize_replacement_flag(value):
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"replaced", "true", "1", "yes", "y"}:
        return True
    if normalized in {"not replaced", "false", "0", "no", "n", ""}:
        return False
    return False


def _replacement_status_label(value):
    return "Replaced" if _normalize_replacement_flag(value) else "Not Replaced"


def _clean_search_record(record):
    cleaned = {}
    for key, value in record.items():
        if pd.isna(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def _get_record_value(record, *keys):
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return ""


def _parse_replacement_mapping(record):
    replacement_mapping = _get_record_value(record, "PDB Code Changed", "pdb_code_changed")
    if not replacement_mapping or "->" not in replacement_mapping:
        return "", ""

    legacy_code, replacement_code = replacement_mapping.split("->", 1)
    return legacy_code.strip().upper(), replacement_code.strip().upper()


def _extract_replacement_metadata(record):
    legacy_code = _normalize_search_value(
        _get_record_value(record, "pdb_code", "PDB Code")
    )
    mapped_legacy_code, replacement_code = _parse_replacement_mapping(record)

    if mapped_legacy_code:
        legacy_code = mapped_legacy_code

    if not replacement_code:
        candidate_code = _normalize_search_value(
            _get_record_value(
                record,
                "database2_database_code",
                "database2_database_code_pdb",
                "entry_id",
                "rcsb_id",
            )
        )
        if candidate_code and candidate_code != legacy_code:
            replacement_code = candidate_code

    is_replaced = _normalize_replacement_flag(
        _get_record_value(record, "is_replaced", "Is Replaced")
    )
    canonical_pdb_code = replacement_code or legacy_code

    return {
        "legacy_pdb_code": legacy_code,
        "replacement_pdb_code": replacement_code or None,
        "canonical_pdb_code": canonical_pdb_code or None,
        "is_replaced": is_replaced,
        "replacement_status_label": _replacement_status_label(is_replaced),
    }


def _rank_search_record(record, normalized_query):
    if not normalized_query:
        return 0, []

    query_lower = normalized_query.lower()
    score = 0
    reasons = []

    replacement_metadata = _extract_replacement_metadata(record)
    exact_fields = [
        ("pdb_code_exact", _normalize_search_value(record.get("pdb_code")), 120),
        ("uniprot_id_exact", _normalize_search_value(record.get("uniprot_id")), 115),
        ("legacy_pdb_code_exact", replacement_metadata["legacy_pdb_code"] or "", 112),
        ("canonical_pdb_code_exact", replacement_metadata["canonical_pdb_code"] or "", 110),
        ("replacement_pdb_code_exact", replacement_metadata["replacement_pdb_code"] or "", 108),
    ]

    for reason, candidate, points in exact_fields:
        if candidate and candidate == normalized_query:
            score += points
            reasons.append(reason)

    searchable_fields = [
        ("name_contains", ["name", "struct_title"], 36),
        ("protein_name_contains", ["protein_recommended_name", "protein_alternative_name"], 34),
        ("gene_contains", ["associated_genes"], 30),
        ("group_contains", ["group", "subgroup"], 24),
        ("family_contains", ["family_name_cache", "family_superfamily_name"], 20),
        ("species_contains", ["species", "organism_scientific_name", "organism_common_name"], 18),
        ("disease_contains", ["comment_disease_name", "comment_disease"], 16),
        ("keywords_contains", ["keywords", "description", "comment_function"], 12),
    ]

    for reason, keys, points in searchable_fields:
        values = [
            _get_record_value(record, key)
            for key in keys
        ]
        values = [value for value in values if value]
        if not values:
            continue

        if any(value.upper() == normalized_query for value in values):
            score += points + 12
            reasons.append(f"{reason}_exact")
            continue

        if any(query_lower in value.lower() for value in values):
            score += points
            reasons.append(reason)

    return score, reasons


def _enrich_search_frame_with_local_metadata(dataframe):
    try:
        local_dataset = DashboardAnnotationDatasetService._load_local_merged_dataset()
    except Exception as exc:
        logger.debug("Unable to enrich merged search frame with local metadata: %s", exc)
        return dataframe

    if local_dataset.empty or "pdb_code" not in local_dataset.columns:
        return dataframe

    extra_columns = [
        "pdb_code",
        "PDB Code Changed",
        "Is Replaced",
        "struct_title",
    ]
    available_columns = [
        column for column in extra_columns
        if column in local_dataset.columns
    ]

    if "pdb_code" not in available_columns or len(available_columns) == 1:
        return dataframe

    local_subset = local_dataset[available_columns].drop_duplicates(subset=["pdb_code"], keep="first")
    return dataframe.merge(local_subset, on="pdb_code", how="left", suffixes=("", "_local"))


def _normalize_search_frame_columns(dataframe):
    if dataframe.empty:
        return dataframe

    rename_map = {
        "Pdb Code": "pdb_code",
        "Group": "group",
        "Subgroup": "subgroup",
        "Name": "name",
        "Species": "species",
        "Taxonomic Domain": "taxonomic_domain",
        "Description": "description",
        "Related Pdb Entries": "related_pdb_entries",
    }
    applicable = {
        source: target
        for source, target in rename_map.items()
        if source in dataframe.columns and target not in dataframe.columns
    }
    if not applicable:
        return dataframe
    return dataframe.rename(columns=applicable)


def search_merged_databases(search_term, limit=25):
    normalized_query = _normalize_search_value(search_term)
    if not normalized_query:
        return []

    try:
        all_data = _normalize_search_frame_columns(all_merged_databases())
    except Exception as exc:
        logger.warning(
            "Falling back to local merged dataset search for %s: %s",
            normalized_query,
            exc,
        )
        all_data = _normalize_search_frame_columns(
            DashboardAnnotationDatasetService._load_local_merged_dataset()
        )

    if all_data.empty:
        return []

    candidate_columns = [
        column
        for column in [
            "pdb_code",
            "uniprot_id",
            "name",
            "struct_title",
            "protein_recommended_name",
            "protein_alternative_name",
            "associated_genes",
            "group",
            "subgroup",
            "family_name_cache",
            "family_superfamily_name",
            "species",
            "organism_scientific_name",
            "organism_common_name",
            "comment_disease_name",
            "comment_disease",
            "keywords",
            "description",
            "comment_function",
            "PDB Code Changed",
        ]
        if column in all_data.columns
    ]

    if candidate_columns:
        normalized_lower = normalized_query.lower()
        candidate_mask = pd.Series(False, index=all_data.index)
        for column in candidate_columns:
            series = all_data[column].fillna("").astype(str)
            candidate_mask = candidate_mask | series.str.lower().str.contains(
                normalized_lower,
                regex=False,
            )
        all_data = all_data[candidate_mask]

    if all_data.empty:
        return []

    records = []

    for record in all_data.to_dict(orient="records"):
        cleaned_record = _clean_search_record(record)
        match_score, match_reasons = _rank_search_record(cleaned_record, normalized_query)
        if match_score <= 0:
            continue

        replacement_metadata = _extract_replacement_metadata(cleaned_record)
        cleaned_record.update(replacement_metadata)
        cleaned_record["match_score"] = match_score
        cleaned_record["match_reasons"] = match_reasons
        scientific_assessment = build_scientific_assessment(cleaned_record)
        cleaned_record["scientific_assessment"] = scientific_assessment
        cleaned_record["external_links"] = DashboardAnnotationDatasetService._build_external_links(cleaned_record)
        cleaned_record["record_resolution"] = DashboardAnnotationDatasetService._build_record_resolution(cleaned_record)
        cleaned_record["provenance_summary"] = {
            "sources_present": DashboardAnnotationDatasetService._get_sources_present(cleaned_record),
            "confidence": DashboardAnnotationDatasetService._build_confidence_summary(cleaned_record),
        }
        cleaned_record["match_summary"] = {
            "primary_reason": (match_reasons or [None])[0],
            "matched_on": match_reasons,
            "resolved_group": (cleaned_record.get("record_resolution") or {}).get("selected_group_label"),
            "has_benchmark_caution": not scientific_assessment.get("recommended_for_sequence_topology_benchmark", True),
        }
        records.append(cleaned_record)

    records.sort(
        key=lambda item: (
            -item.get("match_score", 0),
            str(item.get("canonical_pdb_code") or ""),
            str(item.get("pdb_code") or ""),
        )
    )

    records = records[:limit]
    records = DashboardAnnotationDatasetService._attach_normalized_tm_prediction_payloads(records)
    for record in records:
        record["match_summary"] = {
            **(record.get("match_summary") or {}),
            "available_tm_prediction_methods": (
                (record.get("tm_prediction_overview") or {}).get("available_methods") or []
            ),
            "preferred_tm_prediction_method": record.get("preferred_tm_prediction_method"),
            "preferred_tm_prediction_count": record.get("preferred_tm_prediction_count"),
            "preferred_tm_compact_label": record.get("preferred_tm_compact_label"),
            "preferred_tm_orientation_label": record.get("preferred_tm_orientation_label"),
        }
    return records


def get_columns_by_pdb_codes(pdb_codes, columns):
    """
    Given a list of pdb_codes and a list of column names, 
    returns those columns (plus pdb_code) for the matching rows.
    """
    df = all_merged_databases()

    upc = [code.upper() for code in pdb_codes]
    df = df[df["pdb_code"].fillna("").str.upper().isin(upc)]

    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"Columns not found in merged DB: {missing}")

    cols = list(dict.fromkeys(["pdb_code"] + columns))
    return df[cols].fillna("").to_dict(orient="records")




######################################## LIST OF OPTION FOR FILTERS ################################
#                                                                                                  #
#                                                                                                  #
####################################################################################################
def group_filter_options():
    return DashboardFilterOptionsService.get_group_options()

def subgroup_filter_options():
    return DashboardFilterOptionsService.get_subgroup_options()

def taxonomic_domain_filter_options():
    return DashboardFilterOptionsService.get_taxonomic_domain_options()

def experimental_methods_filter_options():
    return DashboardFilterOptionsService.get_experimental_methods_options()

def molecular_function_filter_options():
    return DashboardFilterOptionsService.get_molecular_function_options()

def cellular_component_filter_options():
    return DashboardFilterOptionsService.get_cellular_component_options()

def biological_process_filter_options():
    return DashboardFilterOptionsService.get_biological_process_options()

def family_name_filter_options():
    return DashboardFilterOptionsService.get_family_name_options()

def species_filter_options():
    return DashboardFilterOptionsService.get_species_options()

def membrane_name_filter_options():
    return DashboardFilterOptionsService.get_membrane_name_options()

def super_family_filter_options():
    return DashboardFilterOptionsService.get_super_family_options()

def super_family_class_type_filter_options():
    return DashboardFilterOptionsService.get_super_family_class_type_options()


def convert_to_list_of_dicts(input_str):
    # Split the input string by comma and strip any extra spaces
    keys = [key.strip() for key in input_str.split(",")]
    
    # Create a set to store unique dictionaries
    unique_dicts = set()
    
    for key in keys:
        # Create a dictionary entry
        entry = {
            "key": key,
            "name": key,
            "x-angle": 5 if key == "group" else 0,
        }
        
        # Convert the dictionary to a tuple of tuples to store in the set
        entry_tuple = tuple(entry.items())
        
        # Add the tuple to the set to ensure uniqueness
        unique_dicts.add(entry_tuple)
    
    # Convert the set of tuples back to a list of dictionaries
    result = [dict(t) for t in unique_dicts]
    
    return result


import matplotlib.colors as mcolors
def generate_color_palette(start_color, end_color, num_colors):
    # Convert hex colors to RGB
    start_rgb = mcolors.hex2color(start_color)
    end_rgb = mcolors.hex2color(end_color)

    # Create a list of RGB colors in the gradient
    colors = []
    for i in range(num_colors):
        r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * (i / (num_colors - 1))
        g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * (i / (num_colors - 1))
        b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * (i / (num_colors - 1))
        colors.append((r, g, b))

    # Convert RGB colors back to hex
    hex_colors = [mcolors.rgb2hex(color) for color in colors]

    return hex_colors


def create_grouped_bar_chart(table_df):
    
    # Identify date columns
    date_columns = table_df.select_dtypes(include=['datetime64[ns]']).columns
    
    # Drop date columns
    table_df = table_df.drop(columns=date_columns)
    
    # Group by 'group' and count the occurrences
    grouped_data = table_df.groupby("group").size().reset_index(name='CumulativeCount')
    
    # Sort by count
    grouped_data = grouped_data.sort_values(by='CumulativeCount', ascending=True)
    grouped_data['group'] = grouped_data['group'].replace({
        'MONOTOPIC MEMBRANE PROTEINS': "Group 1",
        'TRANSMEMBRANE PROTEINS:BETA-BARREL': "Group 2",
        'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL': "Group 3",
    })
    
    # Define group labels and their meanings
    group_labels = {
        "Group 1": 'Group 1 (MONOTOPIC MEMBRANE PROTEINS)',
        "Group 2": 'Group 2 (TRANSMEMBRANE PROTEINS:BETA-BARREL)',
        "Group 3": 'Group 3 (TRANSMEMBRANE PROTEINS:ALPHA-HELICAL)'
    }
    
    # Add a 'label' column for detailed legend information
    grouped_data['label'] = grouped_data['group'].map(group_labels)
    
    # Define color list
    color_list = ['#D9DE84', '#93C4F6', '#005EB8', '#636B05']
    
    # Create the grouped bar chart
    chart = alt.Chart(grouped_data).mark_bar().encode(
        x=alt.X(
            'group:N', title='Group', 
            sort=None, axis=alt.Axis(
                labelAngle=0,
                labelLimit=0
            )
        ),
        y=alt.Y('CumulativeCount:Q', title='Cumulative MP Structures'),
        color=alt.Color(
            'label:N', scale=alt.Scale(domain=list(group_labels.values()), range=color_list),
            legend=alt.Legend(title="Group", orient="bottom", labelLimit=0, direction="vertical")
        ),
        tooltip=["group", "CumulativeCount"]
    ).properties(
        title='Cumulative sum of resolved Membrane Protein (MP) Structures categorized by group',
        width="container"
    ).configure_legend(
        symbolType='square'
    ).configure_axisX(
        labelAngle=0  # Ensure labels are horizontal
    )

    return convert_chart(chart)


def extract_widths(chart_dict, chart_width=800):
    widths = []
    
    # Calculate the available width by subtracting the padding
    padding = 200 
    available_width = chart_width - padding

    chart_width_1 = 0.5*available_width
    chart_width_2 = 0.5*chart_width_1
    new_widths = [chart_width_2, chart_width_2, chart_width_1]
    # Check hconcat items
    if 'hconcat' in chart_dict:
        for item in chart_dict['hconcat']:
            width = item.get('width')
            if width:
                widths.append(width)
            if 'layer' in item:
                for layer in item['layer']:
                    width = layer.get('width')
                    if width:
                        widths.append(width)

    return widths == new_widths
