import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from sqlalchemy import Column, Integer, MetaData, Table, Text
from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect

from app import app as flask_app
from src.Dashboard.services import (
    DashboardAnnotationDatasetService,
    DashboardPageService,
    DashboardRegressionValidationService,
    DiscrepancyBenchmarkExportService,
    DiscrepancyReviewService,
    _build_dashboard_global_search_clause,
    _normalize_dashboard_filter_value,
    search_merged_databases,
)
from src.Dashboard.scientific_assessment import build_scientific_assessment
from src.MP.model import MembraneProteinData
from src.MP.model_opm import OPM
from src.MP.model_uniprot import Uniprot
from src.AI_Packages.TMProteinPredictor import (
    MultiModelAnalyzer,
    build_tm_prediction_payload,
    parse_tmbed_output,
    serialize_tm_regions,
)
from src.AI_Packages.TMAlphaFoldPredictorClient import (
    _count_transmembrane_regions,
    _extract_sequence_from_payload,
    _parse_regions_from_payload,
    _parse_tmdet_regions_from_payload,
)
from src.Jobs.LoadProteinPredictions import (
    export_optional_tm_prediction_inputs,
    get_optional_tm_prediction_paths,
    normalize_external_tm_prediction_dataframe,
    run_tm_prediction_backfill,
)
from src.Jobs.TMAlphaFoldSync import (
    _collect_sequence_backfills,
    _collect_sequence_backfills_from_stored_rows,
    mirror_local_tm_prediction_rows,
)
from src.ingestion.schema_sync_service import SchemaSyncService
from src.services.Helpers.helper import getPercentage


class DashboardRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp_dir = tempfile.TemporaryDirectory()
        flask_app.config.update(
            TESTING=True,
            BENCHMARK_EXPORT_DIR=cls._temp_dir.name,
            OPTIONAL_TM_PREDICTION_BASE_DIR=cls._temp_dir.name,
        )
        cls.app_context = flask_app.app_context()
        cls.app_context.push()
        cls.client = flask_app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.app_context.pop()
        cls._temp_dir.cleanup()

    def test_canonical_search_returns_egfr_results(self):
        results = search_merged_databases("EGFR", limit=5)

        self.assertTrue(results)
        self.assertEqual(results[0]["pdb_code"], "5LV6")
        self.assertIn("gene_contains_exact", results[0].get("match_reasons") or [])

    def test_replaced_entry_resolves_to_canonical_record(self):
        legacy_record = DashboardAnnotationDatasetService.get_record("5TSI")
        canonical_record = DashboardAnnotationDatasetService.get_record("5UAR")

        self.assertIsNotNone(legacy_record)
        self.assertEqual(legacy_record["canonical_pdb_code"], "5UAR")
        self.assertTrue(legacy_record["is_replaced"])

        self.assertIsNotNone(canonical_record)
        self.assertEqual(canonical_record["canonical_pdb_code"], "5UAR")

    def test_record_payload_exposes_provenance_and_structure_context(self):
        record = DashboardAnnotationDatasetService.get_record("6A69")

        self.assertIsNotNone(record)
        self.assertEqual(record["canonical_pdb_code"], "6A69")
        self.assertGreaterEqual(record["structure_context"]["chain_count"], 2)
        self.assertTrue(
            {"mpstruc", "pdb", "uniprot"}.issubset(
                set(record["provenance"]["sources_present"])
            )
        )
        self.assertEqual(
            record["provenance"]["confidence"]["annotation_score_band"],
            "high",
        )
        self.assertIn("ui_sections", record)
        self.assertIn("comparison", record["ui_sections"])
        self.assertIn("tm_boundaries", record["ui_sections"])
        self.assertIn("scientific_flags", record["ui_sections"])
        self.assertIn("field_glossary_keys", record)

    def test_tm_boundary_enrichment_is_available(self):
        record = DashboardAnnotationDatasetService.get_record("1O5W")

        self.assertIsNotNone(record)
        self.assertTrue(record["opm_tm_regions"])
        self.assertTrue(record["ui_sections"]["tm_boundaries"]["rows"])
        self.assertEqual(
            record["provenance"]["field_sources"]["topology"]["opm_tm_regions"]["source"],
            "opm",
        )
        self.assertEqual(record["discrepancy_review"]["status"], "open")
        comparison_groups = record["ui_sections"]["comparison"]["group_rows"]
        comparison_sources = {item["source"] for item in comparison_groups}
        self.assertIn("Expert", comparison_sources)
        self.assertIn("Predicted", comparison_sources)
        self.assertIn("OPM", comparison_sources)
        self.assertIn("MPstruc", comparison_sources)

    def test_optional_tm_predictor_import_normalization(self):
        dataframe = pd.DataFrame(
            [
                {
                    "pdb_code": "1abc",
                    "tm_regions": '[{"start": 10, "end": 30}, {"start": 45, "end": 67}]',
                },
                {
                    "pdb_code": "2def",
                    "tm_count": 3,
                },
            ]
        )

        normalized = normalize_external_tm_prediction_dataframe(dataframe, "Phobius")

        self.assertEqual(list(normalized["pdb_code"]), ["1ABC", "2DEF"])
        self.assertEqual(normalized.iloc[0]["tm_count"], 2)
        self.assertEqual(normalized.iloc[1]["tm_count"], 3)
        self.assertEqual(len(normalized.iloc[0]["tm_regions"]), 2)

    def test_tm_prediction_payload_keeps_zero_tm_region_lists(self):
        payload = build_tm_prediction_payload(
            {"SEQ_ZERO": 0, "SEQ_ONE": 1},
            {"SEQ_ONE": [{"start": 5, "end": 15}]},
        )

        self.assertIn("SEQ_ZERO", payload["regions"])
        self.assertEqual(payload["regions"]["SEQ_ZERO"], [])
        self.assertEqual(len(payload["regions"]["SEQ_ONE"]), 1)

    def test_serialize_tm_regions_handles_region_lists_without_pd_isna_failures(self):
        serialized = serialize_tm_regions(
            [{"start": 168, "end": 187, "label": "H"}]
        )
        empty_serialized = serialize_tm_regions([])

        self.assertEqual(
            json.loads(serialized),
            [{"start": 168, "end": 187, "label": "H"}],
        )
        self.assertEqual(json.loads(empty_serialized), [])

    def test_parse_tmbed_output_extracts_tm_regions_from_three_line_output(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".pred", delete=False) as handle:
            handle.write(">test_protein\n")
            handle.write("MAAAAAAA\n")
            handle.write("iiHHHooo\n")
            pred_path = handle.name

        try:
            parsed = parse_tmbed_output(pred_path, out_format=4)
        finally:
            Path(pred_path).unlink(missing_ok=True)

        self.assertIn("test_protein", parsed)
        self.assertEqual(parsed["test_protein"]["count"], 1)
        self.assertEqual(parsed["test_protein"]["regions"][0]["start"], 3)
        self.assertEqual(parsed["test_protein"]["regions"][0]["end"], 5)

    def test_sequence_topology_interpretation_is_derived_from_full_regions(self):
        summary = {
            "provider": "TMAlphaFold",
            "method": "DeepTMHMM",
            "prediction_kind": "sequence_topology",
            "tm_count": 6,
            "tm_regions_json": json.dumps(
                [
                    {"start": 1, "end": 166, "label": "Inside"},
                    {"start": 167, "end": 187, "label": "Membrane"},
                    {"start": 188, "end": 204, "label": "Outside"},
                    {"start": 205, "end": 228, "label": "Membrane"},
                    {"start": 229, "end": 248, "label": "Inside"},
                    {"start": 249, "end": 266, "label": "Membrane"},
                    {"start": 267, "end": 282, "label": "Outside"},
                    {"start": 283, "end": 303, "label": "Membrane"},
                    {"start": 304, "end": 511, "label": "Inside"},
                    {"start": 512, "end": 532, "label": "Membrane"},
                    {"start": 533, "end": 538, "label": "Outside"},
                    {"start": 539, "end": 558, "label": "Membrane"},
                    {"start": 559, "end": 733, "label": "Inside"},
                ]
            ),
        }

        derived = DashboardAnnotationDatasetService._build_tm_prediction_topology_summary(
            summary,
            structure_context={"chain_count": 1, "assembly_ids": ["1"]},
        )

        self.assertTrue(derived["available"])
        self.assertEqual(derived["tm_count"], 6)
        self.assertEqual(derived["orientation_label"], "N-in / C-in")
        self.assertEqual(derived["topology_class"], "multipass alpha-helical")
        self.assertEqual(derived["chain_type"], "integral membrane chain")
        self.assertEqual(derived["membrane_segments"][0]["start"], 167)
        self.assertEqual(derived["membrane_segments"][-1]["end"], 558)
        self.assertEqual(
            [item["summary"] for item in derived["loop_annotations"]],
            [
                "Large intracellular N-terminal region before TM1",
                "Large intracellular loop/domain between TM4 and TM5",
                "Large intracellular C-terminal region after TM6",
            ],
        )

    def test_tmdet_membrane_chain_labels_are_classified_as_membrane(self):
        summary = {
            "provider": "TMAlphaFold",
            "method": "TMDET",
            "prediction_kind": "sequence_topology",
            "tm_regions_json": json.dumps(
                [
                    {"start": 16, "end": 34, "label": "TMDET membrane chain A"},
                    {"start": 60, "end": 78, "label": "TMDET membrane chain A"},
                ]
            ),
        }

        regions = DashboardAnnotationDatasetService._get_tmalphafold_tm_regions(summary)

        self.assertEqual(len(regions), 2)
        self.assertEqual(
            DashboardAnnotationDatasetService._classify_topology_region_label("TMDET membrane chain A"),
            "membrane",
        )

    def test_discrepancy_group_equivalence_treats_bitopic_and_alpha_helical_as_same_bucket(self):
        record = {
            "Group (OPM)": "Bitopic Proteins",
            "Group (MPstruc)": "Transmembrane Proteins:Alpha-Helical",
            "Group (Predicted)": "Bitopic",
            "Group (Expert)": "Bitopic",
            "opm_tm_regions": json.dumps([{"start": 10, "end": 30, "label": "OPM"}]),
            "normalized_tm_predictions": [],
        }

        summary = DiscrepancyReviewService._build_discrepancy_summary(record)

        self.assertFalse(summary["has_group_disagreement"])
        self.assertEqual(
            summary["group_labels_for_disagreement"]["opm"],
            "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL / BITOPIC",
        )

    def test_discrepancy_summary_uses_normalized_tm_prediction_counts(self):
        record = {
            "Group (Expert)": "Bitopic",
            "normalized_tm_predictions": [
                {
                    "provider": "TMAlphaFold",
                    "method": "DeepTMHMM",
                    "tm_count": 2,
                    "tm_regions_json": json.dumps(
                        [
                            {"start": 10, "end": 30, "label": "Membrane"},
                            {"start": 40, "end": 60, "label": "Membrane"},
                        ]
                    ),
                }
            ],
        }

        summary = DiscrepancyReviewService._build_discrepancy_summary(record)

        self.assertEqual(summary["tm_counts"]["DeepTMHMM"], 2)
        self.assertEqual(len(summary["tm_regions"]["DeepTMHMM"]), 2)

    def test_discrepancy_summary_prefers_opm_subunit_segments_for_tm_count(self):
        record = {
            "Group (OPM)": "Transmembrane",
            "subunit_segments": 3,
            "opm_tm_regions": json.dumps(
                [{"start": 10, "end": 30}, {"start": 40, "end": 60}]
            ),
            "normalized_tm_predictions": [],
        }

        summary = DiscrepancyReviewService._build_discrepancy_summary(record)

        self.assertEqual(summary["tm_counts"]["opm"], 3)

    def test_build_candidate_payload_normalizes_group_mpstruc_from_group_column(self):
        candidate = DiscrepancyReviewService._build_candidate_payload(
            {
                "PDB Code": "1B12",
                "group": "Monotopic Membrane Proteins",
                "TM (Expert)": 1,
            }
        )

        self.assertEqual(
            candidate["discrepancy_summary"]["group_labels"]["mpstruc"],
            "MONOTOPIC MEMBRANE PROTEINS",
        )
        self.assertEqual(
            candidate["record"]["Group (MPstruc)"],
            "MONOTOPIC MEMBRANE PROTEINS",
        )

    def test_normalize_record_metadata_promotes_group_to_group_mpstruc(self):
        normalized = DashboardAnnotationDatasetService._normalize_record_metadata(
            {
                "group": "Monotopic Membrane Proteins",
                "famsupclasstype_type_name": "Transmembrane",
            }
        )

        self.assertEqual(
            normalized["Group (MPstruc)"],
            "MONOTOPIC MEMBRANE PROTEINS",
        )
        self.assertEqual(
            normalized["Group (OPM)"],
            "TRANSMEMBRANE",
        )

    @patch.object(
        DashboardAnnotationDatasetService,
        "_load_live_group_predictions",
        return_value=pd.DataFrame(columns=["pdb_code", "predicted_group"]),
    )
    @patch.object(
        DashboardAnnotationDatasetService,
        "_load_local_merged_dataset",
        return_value=pd.DataFrame(
            [
                {
                    "pdb_code": "1AFO",
                    "subunit_segments": 2,
                    "opm_tm_regions": '[{"start": 72, "end": 93, "label": "OPM"}]',
                }
            ]
        ),
    )
    @patch.object(
        DashboardAnnotationDatasetService,
        "_load_live_membrane_metadata_frame",
        return_value=pd.DataFrame(
            [
                {
                    "pdb_code": "1AFO",
                    "group": "Bitopic Proteins",
                    "subgroup": "Bitopic proteins",
                    "species": "Homo sapiens",
                    "bibliography_year": 1997,
                    "is_replaced": False,
                }
            ]
        ),
    )
    @patch.object(
        DashboardAnnotationDatasetService,
        "_load_dataset",
        return_value=pd.DataFrame(
            [{"PDB Code": "1AFO", "Group (Expert)": "Bitopic Proteins", "TM (Expert)": 1}]
        ),
    )
    def test_load_enriched_dataset_merges_live_membrane_group(
        self,
        _mock_dataset,
        _mock_live_metadata,
        _mock_local,
        _mock_live_predictions,
    ):
        DashboardAnnotationDatasetService._process_dataframe_cache = {}

        enriched = DashboardAnnotationDatasetService._load_enriched_dataset()

        self.assertEqual(len(enriched.index), 1)
        self.assertEqual(
            enriched.iloc[0]["Group (MPstruc)"],
            "BITOPIC PROTEINS",
        )
        self.assertEqual(
            enriched.iloc[0]["group"],
            "BITOPIC PROTEINS",
        )

    @patch(
        "src.Dashboard.services.get_normalized_tm_prediction_summaries_for_pdb_codes"
    )
    def test_attach_normalized_tm_predictions_uses_pdb_code_header_when_present(
        self,
        mock_get_summaries,
    ):
        mock_get_summaries.return_value = {
            "1B12": [
                {
                    "provider": "TMAlphaFold",
                    "method": "DeepTMHMM",
                    "tm_count": 1,
                    "tm_regions_json": '[{"start": 12, "end": 34, "label": "Membrane"}]',
                }
            ]
        }

        enriched = DashboardAnnotationDatasetService._attach_normalized_tm_prediction_payloads(
            [{"PDB Code": "1B12"}]
        )

        self.assertEqual(len(enriched[0]["normalized_tm_predictions"]), 1)
        self.assertEqual(
            enriched[0]["normalized_tm_predictions"][0]["method"],
            "DeepTMHMM",
        )

    @patch.object(
        DiscrepancyReviewService,
        "_build_candidate_payload",
        side_effect=lambda record, review=DiscrepancyReviewService._REVIEW_SENTINEL: {
            "pdb_code": record["PDB Code"],
            "canonical_pdb_code": record["PDB Code"],
            "bibliography_year": record.get("bibliography_year"),
            "review": {"status": "open"},
            "record": record,
            "discrepancy_summary": {"has_any_disagreement": True, "tm_counts": {}, "group_labels": {}},
            "benchmark_decision": {"include_in_benchmark": True, "high_confidence_subset": False},
        },
    )
    @patch.object(
        DiscrepancyReviewService,
        "_find_review_for_record",
        return_value=None,
    )
    @patch.object(
        DiscrepancyReviewService,
        "_load_review_index",
        return_value={},
    )
    @patch.object(
        DashboardAnnotationDatasetService,
        "_attach_normalized_tm_prediction_payloads",
        side_effect=lambda records: records,
    )
    @patch.object(
        DashboardAnnotationDatasetService,
        "_load_enriched_dataset",
        return_value=pd.DataFrame(
            [
                {"PDB Code": "1AFO", "group": "Bitopic Proteins", "bibliography_year": 1997},
                {"PDB Code": "2BBB", "group": "Monotopic Membrane Proteins", "bibliography_year": 2002},
                {"PDB Code": "3CCC", "group": "Transmembrane Proteins:Alpha-Helical", "bibliography_year": 2010},
            ]
        ),
    )
    def test_list_candidates_returns_paginated_payload_when_requested(
        self,
        _mock_dataset,
        _mock_attach_predictions,
        _mock_review_index,
        _mock_find_review,
        _mock_build_candidate,
    ):
        payload = DiscrepancyReviewService.list_candidates(
            disagreement_only=True,
            search="2BBB",
            page=1,
            per_page=10,
        )

        self.assertEqual(payload["pagination"]["total_items"], 1)
        self.assertEqual(payload["pagination"]["total_pages"], 1)
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["pdb_code"], "2BBB")
        self.assertEqual(payload["filters"]["search"], "2BBB")

    @patch.object(
        DiscrepancyReviewService,
        "_build_candidate_payload",
        side_effect=lambda record, review=DiscrepancyReviewService._REVIEW_SENTINEL: {
            "pdb_code": record["PDB Code"],
            "canonical_pdb_code": record["PDB Code"],
            "bibliography_year": record.get("bibliography_year"),
            "review": {"status": "open"},
            "record": record,
            "discrepancy_summary": {"has_any_disagreement": True, "tm_counts": {}, "group_labels": {}},
            "benchmark_decision": {"include_in_benchmark": True, "high_confidence_subset": False},
        },
    )
    @patch.object(
        DiscrepancyReviewService,
        "_find_review_for_record",
        return_value=None,
    )
    @patch.object(
        DiscrepancyReviewService,
        "_load_review_index",
        return_value={},
    )
    @patch.object(
        DashboardAnnotationDatasetService,
        "_attach_normalized_tm_prediction_payloads",
        side_effect=lambda records: records,
    )
    @patch.object(
        DashboardAnnotationDatasetService,
        "_load_enriched_dataset",
        return_value=pd.DataFrame(
            [
                {"PDB Code": "1AFO", "group": "Bitopic Proteins", "bibliography_year": 1997},
                {"PDB Code": "2BBB", "group": "Monotopic Membrane Proteins", "bibliography_year": 2002},
                {"PDB Code": "3CCC", "group": "Transmembrane Proteins:Alpha-Helical", "bibliography_year": 2010},
            ]
        ),
    )
    def test_list_candidates_ignores_undefined_search_placeholder(
        self,
        _mock_dataset,
        _mock_attach_predictions,
        _mock_review_index,
        _mock_find_review,
        _mock_build_candidate,
    ):
        payload = DiscrepancyReviewService.list_candidates(
            disagreement_only=True,
            search="undefined",
            page=1,
            per_page=25,
        )

        self.assertEqual(payload["pagination"]["total_items"], 3)
        self.assertEqual(len(payload["items"]), 3)
        self.assertIsNone(payload["filters"]["search"])

    def test_discrepancy_review_export_supports_xlsx_and_tsv(self):
        dataframe = pd.DataFrame([{"pdb_code": "1AFO", "group_mpstruc": "BITOPIC PROTEINS"}])
        metadata = {"generated_at": "2026-03-24T10:00:00Z"}

        tsv_payload = DiscrepancyBenchmarkExportService._build_table_download_payload(
            dataframe=dataframe,
            metadata=metadata,
            export_format="tsv",
            filename_prefix="test_queue",
            json_wrapper=True,
        )
        xlsx_payload = DiscrepancyBenchmarkExportService._build_table_download_payload(
            dataframe=dataframe,
            metadata=metadata,
            export_format="xlsx",
            filename_prefix="test_queue",
            json_wrapper=True,
        )

        self.assertEqual(tsv_payload["content_type"], "text/tab-separated-values")
        self.assertTrue(tsv_payload["filename"].endswith(".tsv"))
        self.assertIn("pdb_code", tsv_payload["content"])
        self.assertEqual(
            xlsx_payload["content_type"],
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        self.assertTrue(xlsx_payload["filename"].endswith(".xlsx"))
        self.assertGreater(len(xlsx_payload["content"]), 0)

    def test_dashboard_filter_normalization_ignores_placeholder_values(self):
        self.assertIsNone(_normalize_dashboard_filter_value(""))
        self.assertIsNone(_normalize_dashboard_filter_value("All"))
        self.assertIsNone(_normalize_dashboard_filter_value("undefined"))
        self.assertEqual(
            _normalize_dashboard_filter_value(["All", "alpha-helical", "", "null"]),
            ["alpha-helical"],
        )

    def test_dashboard_global_search_clause_covers_subgroup_and_joined_tables(self):
        clause = _build_dashboard_global_search_clause(
            "EGFR bitopic",
            MembraneProteinData,
            OPM,
            Uniprot,
        )

        compiled = str(
            clause.compile(
                dialect=sqlite_dialect(),
                compile_kwargs={"literal_binds": True},
            )
        )

        self.assertIn("subgroup", compiled)
        self.assertIn("associated_genes", compiled)
        self.assertIn("family_name_cache", compiled)
        self.assertIn("comment_disease", compiled)
        self.assertIn(" AND ", compiled)

    def test_completed_predictor_mask_requires_regions_and_counts(self):
        dataframe = pd.DataFrame(
            [
                {"TMbed_tm_count": 1, "TMbed_tm_regions": '[{"start": 10, "end": 30}]'},
                {"TMbed_tm_count": 1, "TMbed_tm_regions": ""},
                {"TMbed_tm_count": None, "TMbed_tm_regions": '[{"start": 10, "end": 30}]'},
            ]
        )

        completion_mask = MultiModelAnalyzer._completed_predictor_mask(dataframe, "TMbed")

        self.assertEqual(completion_mask.tolist(), [True, False, False])

    def test_comparison_section_surfaces_optional_predictors(self):
        record = {
            "TM (Expert)": 2,
            "subunit_segments": 1,
            "TMbed_tm_count": 2,
            "DeepTMHMM_tm_count": 2,
            "Phobius_tm_count": 2,
            "TOPCONS_tm_count": 1,
            "CCTOP_tm_count": 2,
            "annotation_lineage": {
                "labels": {
                    "expert": "MONOTOPIC MEMBRANE PROTEINS",
                    "predicted": "MONOTOPIC MEMBRANE PROTEINS",
                    "opm": "All alpha monotopic/peripheral",
                    "mpstruc": "MONOTOPIC MEMBRANE PROTEINS",
                }
            },
        }

        comparison = DashboardAnnotationDatasetService._build_comparison_section(record)
        segment_sources = {item["source"] for item in comparison["segment_rows"]}

        self.assertIn("Phobius", segment_sources)
        self.assertIn("TOPCONS", segment_sources)
        self.assertIn("CCTOP", segment_sources)

    def test_tmalphafold_payload_parser_preserves_all_topology_regions(self):
        payload = {
            "Method": "Scampi",
            "Regions": {
                "Region": [
                    {"_attributes": {"Start": 1, "End": 36, "Loc": "Inside"}},
                    {"_attributes": {"Start": 37, "End": 57, "Loc": "Membrane"}},
                    {"_attributes": {"Start": 58, "End": 73, "Loc": "Outside"}},
                    {"_attributes": {"Start": 74, "End": 94, "Loc": "Membrane"}},
                ]
            },
        }

        regions = _parse_regions_from_payload(payload, "sequence_topology")

        self.assertEqual(len(regions), 4)
        self.assertEqual(regions[0]["label"], "Inside")
        self.assertEqual(regions[1]["label"], "Membrane")
        self.assertEqual(regions[-1]["end"], 94)
        self.assertEqual(_count_transmembrane_regions(regions), 2)

    def test_tmalphafold_payload_parser_handles_array_like_loc_values(self):
        payload = {
            "Method": "Prodiv",
            "Regions": {
                "Region": [
                    {"_attributes": {"Start": 448, "End": 468, "Loc": ["Membrane"]}},
                    {"_attributes": {"Start": 469, "End": 511, "Loc": ["Outside"]}},
                    {"_attributes": {"Start": 512, "End": 532, "Loc": ["Membrane"]}},
                ]
            },
        }

        regions = _parse_regions_from_payload(payload, "sequence_topology")

        self.assertEqual(len(regions), 3)
        self.assertEqual(regions[0]["start"], 448)
        self.assertEqual(regions[1]["label"], "Outside")
        self.assertEqual(regions[2]["end"], 532)
        self.assertEqual(_count_transmembrane_regions(regions), 2)

    def test_tmalphafold_payload_parser_extracts_sequence(self):
        payload = {
            "Sequence": {
                "_attributes": {"Length": 5},
                "Seq": "MKTLL",
            }
        }

        sequence = _extract_sequence_from_payload(payload)

        self.assertEqual(sequence, "MKTLL")

    def test_tmalphafold_tmdet_parser_extracts_membrane_chain_regions(self):
        payload = {
            "CHAIN": [
                {
                    "_attributes": {"CHAINID": "A", "NUM_TM": 1},
                    "SEQ": "MKTLLVVVV",
                    "REGION": [
                        {"_attributes": {"seq_beg": 1, "seq_end": 27, "type": "G"}},
                        {"_attributes": {"seq_beg": 136, "seq_end": 156, "type": "M"}},
                        {"_attributes": {"seq_beg": 157, "seq_end": 200, "type": "G"}},
                    ],
                }
            ]
        }

        regions = _parse_tmdet_regions_from_payload(payload)
        sequence = _extract_sequence_from_payload(payload)

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["start"], 136)
        self.assertEqual(regions[0]["end"], 156)
        self.assertEqual(sequence, "MKTLLVVVV")

    def test_tmalphafold_sequence_backfill_collection_deduplicates_targets(self):
        predictions = [
            SimpleNamespace(
                status="success",
                pdb_code="1abc",
                uniprot_id="p12345",
                sequence_sequence="MKTLL",
            ),
            SimpleNamespace(
                status="success",
                pdb_code="1ABC",
                uniprot_id="P12345",
                sequence_sequence="MKTLL",
            ),
            SimpleNamespace(
                status="success",
                pdb_code="2DEF",
                uniprot_id="Q99999",
                sequence_sequence="VVVVG",
            ),
            SimpleNamespace(
                status="error",
                pdb_code="3XYZ",
                uniprot_id="A00001",
                sequence_sequence="SHOULD_SKIP",
            ),
        ]

        collected = _collect_sequence_backfills(predictions)

        self.assertEqual(len(collected["rows"]), 2)
        self.assertEqual(collected["conflicting_targets"], [])
        self.assertEqual(collected["rows"][0]["pdb_code"], "1ABC")

    def test_tmalphafold_sequence_backfill_collection_reads_stored_payloads(self):
        stored_rows = [
            SimpleNamespace(
                pdb_code="1ABC",
                uniprot_id="P12345",
                raw_payload_json=json.dumps({"Sequence": {"Seq": "MKTLL"}}),
            ),
            SimpleNamespace(
                pdb_code="2DEF",
                uniprot_id="Q99999",
                raw_payload_json="",
            ),
        ]

        collected = _collect_sequence_backfills_from_stored_rows(stored_rows)

        self.assertEqual(len(collected["rows"]), 1)
        self.assertEqual(collected["rows"][0]["sequence_sequence"], "MKTLL")

    @patch("src.Jobs.TMAlphaFoldSync._upsert_predictions")
    @patch("src.Jobs.TMAlphaFoldSync._lookup_uniprot_ids_for_pdb_codes")
    @patch("src.Jobs.TMAlphaFoldSync._ensure_tmalphafold_storage")
    def test_mirror_local_tm_prediction_rows_stores_tmbed_in_normalized_table(
        self,
        mock_ensure_storage,
        mock_lookup_uniprots,
        mock_upsert_predictions,
    ):
        mock_lookup_uniprots.return_value = {"8TDJ": ["Q99999"]}
        mock_upsert_predictions.side_effect = lambda rows: len(rows)

        summary = mirror_local_tm_prediction_rows(
            method="TMbed",
            records=[
                {
                    "pdb_code": "8TDJ",
                    "TMbed_tm_count": 6,
                    "TMbed_tm_regions": '[{"start": 168, "end": 187, "label": "H"}]',
                }
            ],
            provider="MetaMP",
        )

        stored_rows = mock_upsert_predictions.call_args.args[0]
        self.assertEqual(summary["stored_rows"], 1)
        self.assertEqual(len(stored_rows), 1)
        self.assertEqual(stored_rows[0].provider, "MetaMP")
        self.assertEqual(stored_rows[0].method, "TMbed")
        self.assertEqual(stored_rows[0].pdb_code, "8TDJ")
        self.assertEqual(stored_rows[0].uniprot_id, "Q99999")

    @patch("src.Jobs.LoadProteinPredictions.MultiModelAnalyzer")
    @patch("src.Jobs.LoadProteinPredictions.DeepTMHMMPredictor")
    @patch("src.Jobs.LoadProteinPredictions.TMbedPredictor")
    @patch("src.Jobs.LoadProteinPredictions.load_pending_tm_prediction_frame")
    @patch("src.Jobs.TMAlphaFoldSync.mirror_local_tm_prediction_rows")
    def test_tm_prediction_backfill_summary_counts_only_current_run(
        self,
        mock_mirror_rows,
        mock_frame_loader,
        mock_tmbed_predictor,
        mock_deeptmhmm_predictor,
        mock_analyzer_cls,
    ):
        mock_frame_loader.return_value = pd.DataFrame(
            [
                {"pdb_code": "9DI6", "sequence_sequence": "MKTLL"},
            ]
        )
        mock_tmbed_predictor.return_value = SimpleNamespace(name="TMbed")
        mock_deeptmhmm_predictor.return_value = SimpleNamespace(name="DeepTMHMM")
        mock_mirror_rows.return_value = {"stored_rows": 1}
        analyzer = mock_analyzer_cls.return_value
        analyzer.predictors = [SimpleNamespace(name="TMbed"), SimpleNamespace(name="DeepTMHMM")]
        analyzer.analyze.return_value = pd.DataFrame(
            [
                {"pdb_code": "OLD1", "TMbed_tm_count": 1},
                {"pdb_code": "9DI6", "TMbed_tm_count": 2, "DeepTMHMM_tm_count": 2},
            ]
        )

        summary = run_tm_prediction_backfill(
            include_tmbed=True,
            include_deeptmhmm=True,
            progress_callback=None,
        )

        self.assertEqual(summary["queued_records"], 1)
        self.assertEqual(summary["processed_records"], 1)
        self.assertNotIn("csv_total_records", summary)
        self.assertNotIn("csv_path", summary)

    @patch("src.Jobs.LoadProteinPredictions.MultiModelAnalyzer")
    @patch("src.Jobs.LoadProteinPredictions.TMbedPredictor")
    @patch("src.Jobs.LoadProteinPredictions.load_pending_tm_prediction_frame")
    @patch("src.Jobs.TMAlphaFoldSync.mirror_local_tm_prediction_rows")
    def test_tm_prediction_backfill_persists_each_batch_to_normalized_store(
        self,
        mock_mirror_rows,
        mock_frame_loader,
        mock_tmbed_predictor,
        mock_analyzer_cls,
    ):
        mock_frame_loader.return_value = pd.DataFrame(
            [{"pdb_code": "8TDJ", "sequence_sequence": "MKTLL"}]
        )
        mock_tmbed_predictor.return_value = SimpleNamespace(name="TMbed")
        mock_mirror_rows.return_value = {"stored_rows": 1}
        analyzer = mock_analyzer_cls.return_value
        analyzer.predictors = [SimpleNamespace(name="TMbed")]

        def _analyze_side_effect(*args, **kwargs):
            batch_callback = kwargs.get("on_batch_completed")
            batch_df = pd.DataFrame(
                [
                    {
                        "pdb_code": "8TDJ",
                        "TMbed_tm_count": 2,
                        "TMbed_tm_regions": '[{"start": 10, "end": 30}]',
                    }
                ]
            )
            batch_callback(batch_df, start=0, end=1, total=1)
            return batch_df

        analyzer.analyze.side_effect = _analyze_side_effect

        summary = run_tm_prediction_backfill(
            include_tmbed=True,
            include_deeptmhmm=False,
            progress_callback=None,
        )

        self.assertEqual(summary["normalized_store"]["TMbed"]["stored_rows"], 1)
        self.assertTrue(mock_mirror_rows.called)

    def test_comparison_and_tm_boundaries_include_tmalphafold_methods(self):
        record = {
            "tmalphafold_predictions": [
                {
                    "method": "Scampi",
                    "prediction_kind": "sequence_topology",
                    "tm_count": 2,
                    "tm_regions_json": (
                        '[{"start": 1, "end": 36, "label": "Inside"}, '
                        '{"start": 37, "end": 57, "label": "Membrane"}, '
                        '{"start": 58, "end": 73, "label": "Outside"}, '
                        '{"start": 74, "end": 94, "label": "Membrane"}]'
                    ),
                    "ambiguous": False,
                },
                {
                    "method": "TMDET",
                    "prediction_kind": "structure_membrane_plane",
                    "tm_count": 2,
                    "tm_regions_json": '[{"start": 40, "end": 60}]',
                    "ambiguous": False,
                },
            ]
        }

        comparison = DashboardAnnotationDatasetService._build_comparison_section(record)
        segment_sources = {item["source"] for item in comparison["segment_rows"]}
        boundary_sources = {
            item["source"] for item in DashboardAnnotationDatasetService._build_tm_boundary_rows(record)
        }
        boundary_rows = DashboardAnnotationDatasetService._build_tm_boundary_rows(record)
        scampi_rows = [
            item for item in boundary_rows if item["source"] == "TMAlphaFold Scampi"
        ]

        self.assertIn("TMAlphaFold Scampi", segment_sources)
        self.assertIn("TMAlphaFold Scampi", boundary_sources)
        self.assertIn("TMAlphaFold TMDET", boundary_sources)
        self.assertEqual(len(scampi_rows), 2)

    def test_normalized_tm_prediction_fields_hydrate_legacy_payload_keys(self):
        record = {
            "normalized_tm_predictions": [
                {
                    "provider": "MetaMP",
                    "method": "TMbed",
                    "tm_count": 6,
                    "tm_regions_json": '[{"start": 168, "end": 187, "label": "H"}]',
                    "ambiguous": False,
                },
                {
                    "provider": "MetaMP",
                    "method": "DeepTMHMM",
                    "tm_count": 5,
                    "tm_regions_json": '[{"start": 170, "end": 190, "label": "TM"}]',
                    "ambiguous": False,
                },
            ]
        }

        hydrated = DashboardAnnotationDatasetService._apply_normalized_tm_prediction_fields(record)

        self.assertEqual(hydrated["TMbed_tm_count"], 6)
        self.assertEqual(
            hydrated["TMbed_tm_regions"],
            '[{"start": 168, "end": 187, "label": "H"}]',
        )
        self.assertEqual(hydrated["DeepTMHMM_tm_count"], 5)

    @patch("src.Dashboard.services.get_normalized_tm_prediction_summaries_for_pdb_codes")
    def test_attach_normalized_tm_predictions_bulk_enriches_records(self, mocked_fetch):
        mocked_fetch.return_value = {
            "1PRH": [
                {
                    "provider": "TMAlphaFold",
                    "method": "DeepTMHMM",
                    "prediction_kind": "sequence_topology",
                    "pdb_code": "1PRH",
                    "uniprot_ids": ["P12345"],
                    "accession_count": 1,
                    "tm_count": 6,
                    "tm_regions_json": json.dumps(
                        [
                            {"start": 1, "end": 10, "label": "Inside"},
                            {"start": 11, "end": 30, "label": "Membrane"},
                            {"start": 31, "end": 40, "label": "Outside"},
                        ]
                    ),
                    "source_urls": ["https://example.test/deeptmhmm"],
                    "consensus": True,
                    "ambiguous": False,
                    "note": None,
                }
            ]
        }

        enriched = DashboardAnnotationDatasetService._attach_normalized_tm_prediction_payloads(
            [{"pdb_code": "1PRH", "name": "Example protein"}]
        )

        self.assertEqual(len(enriched), 1)
        self.assertEqual(
            enriched[0]["normalized_tm_predictions"][0]["method"],
            "DeepTMHMM",
        )
        self.assertEqual(
            enriched[0]["tm_prediction_overview"]["available_methods"],
            ["DeepTMHMM"],
        )
        self.assertEqual(
            enriched[0]["normalized_tm_predictions"][0]["derived_topology"]["tm_count"],
            6,
        )
        self.assertEqual(
            enriched[0]["tm_prediction_summary_card"]["preferred_method"],
            "DeepTMHMM",
        )
        self.assertEqual(
            enriched[0]["tm_prediction_summary_card"]["preferred_tm_count"],
            6,
        )
        self.assertEqual(
            enriched[0]["preferred_tm_prediction_method"],
            "DeepTMHMM",
        )
        self.assertEqual(
            enriched[0]["preferred_tm_prediction_count"],
            6,
        )
        self.assertEqual(
            enriched[0]["tm_prediction_available_methods"],
            ["DeepTMHMM"],
        )

    @patch("src.Dashboard.services.get_normalized_tm_prediction_summaries_for_pdb_codes")
    def test_attach_normalized_tm_predictions_backfills_summary_tm_count_from_derived_topology(
        self, mocked_fetch
    ):
        mocked_fetch.return_value = {
            "2YMK": [
                {
                    "provider": "TMAlphaFold",
                    "method": "SignalP",
                    "prediction_kind": "signal_peptide",
                    "pdb_code": "2YMK",
                    "uniprot_ids": ["P81605"],
                    "accession_count": 1,
                    "tm_count": None,
                    "tm_regions_json": json.dumps(
                        [
                            {
                                "index": 1,
                                "start": 1,
                                "end": 19,
                                "length": 19,
                                "label": "Signal",
                            }
                        ]
                    ),
                    "source_urls": ["https://example.test/signalp"],
                    "consensus": True,
                    "ambiguous": False,
                    "note": None,
                }
            ]
        }

        enriched = DashboardAnnotationDatasetService._attach_normalized_tm_prediction_payloads(
            [{"pdb_code": "2YMK", "name": "Signal peptide example"}]
        )

        self.assertEqual(len(enriched), 1)
        self.assertEqual(
            enriched[0]["normalized_tm_predictions"][0]["method"],
            "SignalP",
        )
        self.assertEqual(
            enriched[0]["normalized_tm_predictions"][0]["derived_topology"]["tm_count"],
            1,
        )
        self.assertEqual(
            enriched[0]["normalized_tm_predictions"][0]["tm_count"],
            1,
        )

    def test_comparison_section_includes_signalp_tm_count_rows(self):
        record = {
            "group": "Bitopic proteins",
            "subgroup": "Example subgroup",
            "TM (Expert)": 1,
            "subunit_segments": 1,
            "normalized_tm_predictions": [
                {
                    "provider": "TMAlphaFold",
                    "method": "SignalP",
                    "prediction_kind": "signal_peptide",
                    "tm_count": 1,
                    "ambiguous": False,
                    "derived_topology": {
                        "available": True,
                        "tm_count": 1,
                    },
                }
            ],
        }

        comparison = DashboardAnnotationDatasetService._build_comparison_section(record)
        signalp_row = next(
            item for item in comparison["segment_rows"] if item["source"] == "TMAlphaFold SignalP"
        )

        self.assertEqual(signalp_row["value"], 1)

    def test_scientific_assessment_uses_simplified_flag_shape(self):
        assessment = build_scientific_assessment(
            {
                "pdb_code": "1PFO",
                "name": "Perfringolysin O toxin",
                "structure_context": {"chain_count": 2},
                "is_replaced": False,
            }
        )

        self.assertFalse(assessment["recommended_for_sequence_topology_benchmark"])
        self.assertTrue(assessment["flags"]["context_dependent_topology"])
        self.assertTrue(assessment["flags"]["multichain_context"])
        self.assertIn(
            "soluble_to_membrane_transition",
            (assessment.get("details") or {}).get("context_reasons") or [],
        )

    @patch("src.Jobs.LoadProteinPredictions.load_optional_tm_prediction_frame")
    def test_optional_tm_predictor_export_writes_fasta_and_template(self, mock_frame_loader):
        mock_frame_loader.return_value = pd.DataFrame(
            [
                {"pdb_code": "1ABC", "sequence_sequence": "MKTLLA"},
                {"pdb_code": "2DEF", "sequence_sequence": "VVVVGG"},
            ]
        )
        fasta_path = f"{self._temp_dir.name}/phobius_pending.fasta"
        csv_path = f"{self._temp_dir.name}/phobius_template.csv"

        summary = export_optional_tm_prediction_inputs(
            predictor_name="Phobius",
            fasta_out=fasta_path,
            csv_out=csv_path,
        )

        self.assertEqual(summary["record_count"], 2)
        self.assertTrue(Path(fasta_path).exists())
        self.assertTrue(Path(csv_path).exists())
        fasta_text = Path(fasta_path).read_text()
        self.assertIn(">1ABC", fasta_text)
        self.assertIn("VVVVGG", fasta_text)
        template = pd.read_csv(csv_path)
        self.assertEqual(list(template.columns), ["pdb_code", "tm_count", "tm_regions"])
        self.assertEqual(list(template["pdb_code"]), ["1ABC", "2DEF"])

    def test_optional_tm_predictor_paths_are_predictor_specific(self):
        with patch.dict(
            flask_app.config,
            {"OPTIONAL_TM_PREDICTION_BASE_DIR": self._temp_dir.name},
            clear=False,
        ):
            paths = get_optional_tm_prediction_paths("Phobius")

        self.assertTrue(str(paths["predictor_dir"]).endswith("/phobius"))
        self.assertTrue(str(paths["fasta_path"]).endswith("/phobius/pending.fasta"))
        self.assertTrue(str(paths["csv_template_path"]).endswith("/phobius/template.csv"))
        self.assertTrue(str(paths["results_path"]).endswith("/phobius/results.csv"))

    def test_schema_sync_service_adds_missing_columns(self):
        metadata = MetaData()
        table = Table(
            "membrane_proteins",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("existing_col", Text),
            Column("Phobius_tm_count", Integer),
        )

        executed = []

        class DummyConnection:
            dialect = sqlite_dialect()

            def execute(self, statement):
                executed.append(str(statement))

        with patch(
            "src.ingestion.schema_sync_service.inspect"
        ) as inspect_mock:
            inspect_mock.return_value.get_columns.return_value = [
                {"name": "id"},
                {"name": "existing_col"},
            ]
            added = SchemaSyncService._add_missing_columns(DummyConnection(), table)

        self.assertEqual(added, ["Phobius_tm_count"])
        self.assertEqual(len(executed), 1)
        self.assertIn('ALTER TABLE membrane_proteins ADD COLUMN "Phobius_tm_count"', executed[0])

    def test_schema_sync_service_skips_existing_truncated_postgres_columns(self):
        long_name = "pdbvrpsummary_em_author_provided_fsc_resolution_by_cutoff_halfbit"
        metadata = MetaData()
        table = Table(
            "membrane_protein_pdb",
            metadata,
            Column("id", Integer, primary_key=True),
            Column(long_name, Text),
        )

        executed = []

        class DummyConnection:
            dialect = sqlite_dialect()
            dialect.max_identifier_length = 63

            def execute(self, statement):
                executed.append(str(statement))

        truncated_name = long_name[:63]
        with patch(
            "src.ingestion.schema_sync_service.inspect"
        ) as inspect_mock:
            inspect_mock.return_value.get_columns.return_value = [
                {"name": "id"},
                {"name": truncated_name},
            ]
            added = SchemaSyncService._add_missing_columns(DummyConnection(), table)

        self.assertEqual(added, [])
        self.assertEqual(executed, [])

    def test_record_fill_missing_fields_keeps_live_tm_counts(self):
        record = {"pdb_code": "7S5T", "TMbed_tm_count": None, "structure_context": None}
        fallback = {
            "pdb_code": "7S5T",
            "TMbed_tm_count": 1,
            "DeepTMHMM_tm_count": 2,
            "TMbed_tm_regions": '[{"start": 10, "end": 30}]',
            "structure_context": {"chain_count": 2},
        }

        merged = DashboardAnnotationDatasetService._fill_missing_record_fields(record, fallback)

        self.assertEqual(merged["TMbed_tm_count"], 1)
        self.assertEqual(merged["DeepTMHMM_tm_count"], 2)
        self.assertEqual(merged["TMbed_tm_regions"], '[{"start": 10, "end": 30}]')
        self.assertEqual(merged["structure_context"], {"chain_count": 2})

    def test_search_results_include_transparency_metadata(self):
        results = search_merged_databases("EGFR", limit=3)

        self.assertTrue(results)
        top_result = results[0]
        self.assertIn("match_summary", top_result)
        self.assertIn("external_links", top_result)
        self.assertIn("record_resolution", top_result)
        self.assertEqual(top_result["match_summary"]["primary_reason"], "gene_contains_exact")

    def test_dashboard_metadata_endpoint_exposes_glossary(self):
        response = self.client.get("/api/v1/dashboard-metadata")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        data = payload["data"]
        self.assertIn("field_glossary", data)
        self.assertIn("scientific_flags", data)
        self.assertIn("Benchmark Recommended", json.dumps(data["scientific_flags"]))
        self.assertIn("context_dependent_topology", data["scientific_flags"])

    def test_about_summary_payload_builds_dynamic_table_rows(self):
        summary_columns = [
            SimpleNamespace(name="group", type=SimpleNamespace(python_type=str)),
            SimpleNamespace(name="processed_resolution", type=SimpleNamespace(python_type=float)),
            SimpleNamespace(name="bibliography_year", type=SimpleNamespace(python_type=int)),
        ]

        with patch.object(
            DashboardPageService,
            "_get_database_snapshot",
            return_value={
                "all_data": {"total_rows": 4402},
                "all_data_mpstruc": {"total_rows": 3801},
                "all_data_pdb": {"total_rows": 4386},
                "all_data_opm": {"total_rows": 3002},
                "all_data_uniprot": {"total_rows": 3691},
            },
        ), patch.object(
            DashboardPageService,
            "_get_summary_columns",
            return_value=summary_columns,
        ), patch.object(
            DashboardPageService,
            "_get_about_observation_count",
            return_value=3793,
        ):
            payload = DashboardPageService.build_about_summary_payload()

        self.assertEqual(len(payload["rows"]), 5)
        metamp_row = next(row for row in payload["rows"] if row["database"] == "MetaMP")
        self.assertEqual(metamp_row["observations"], 3793)
        self.assertEqual(metamp_row["source_rows"], 4402)
        self.assertEqual(metamp_row["count_basis"], "distinct_pdb_code")
        self.assertEqual(metamp_row["attributes"], 3)
        self.assertEqual(metamp_row["quantitative"], 2)
        self.assertEqual(metamp_row["nominal"], 1)
        self.assertTrue(metamp_row["highlight"])

    def test_discrepancy_summary_endpoint_exposes_benchmark_counts(self):
        response = self.client.get("/api/v1/discrepancy-reviews/summary")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        data = payload["data"]
        self.assertIn("row_count", data)
        self.assertIn("benchmark_counts", data)
        self.assertIn("status_counts", data)

    def test_case_studies_and_added_value_endpoints(self):
        case_response = self.client.get("/api/v1/dashboard-case-studies")
        added_value_response = self.client.get("/api/v1/dashboard-added-value")

        self.assertEqual(case_response.status_code, 200)
        self.assertEqual(added_value_response.status_code, 200)

        case_payload = case_response.get_json()["data"]
        added_value_payload = added_value_response.get_json()["data"]

        self.assertTrue(case_payload["items"])
        self.assertTrue(any(item["key"] == "replacement_resolution" for item in case_payload["items"]))
        self.assertTrue(added_value_payload["items"])
        self.assertTrue(any(item["key"] == "discrepancy_review" for item in added_value_payload["items"]))

    def test_get_percentage_handles_missing_unique_field(self):
        dataframe = pd.DataFrame(
            {
                "rcsentinfo_selected_polymer_entity_types": [
                    "Protein (only)",
                    "Protein (only)",
                    "Protein/Oligosaccharide",
                ]
            }
        )

        percentages = getPercentage(
            dataframe,
            column="rcsentinfo_selected_polymer_entity_types",
        )

        self.assertIn("Protein (only)", percentages)
        self.assertIn("Protein/Oligosaccharide", percentages)
        self.assertAlmostEqual(percentages["Protein (only)"], 66.67, places=2)

    def test_benchmark_export_payload_and_release_metadata(self):
        payload = DiscrepancyBenchmarkExportService.build_download_payload(
            export_format="csv",
            include_all=False,
        )
        export_result = DiscrepancyBenchmarkExportService.export_release(include_all=False)
        latest = DiscrepancyBenchmarkExportService.latest_export_metadata()

        self.assertEqual(payload["content_type"], "text/csv")
        self.assertIn("pdb_code", payload["content"])
        self.assertGreater(payload["metadata"]["row_count"], 0)
        self.assertGreater(payload["metadata"]["included_row_count"], 0)
        self.assertTrue(export_result["csv_path"].endswith(".csv"))
        self.assertTrue(export_result["json_path"].endswith(".json"))
        self.assertIsNotNone(latest)
        self.assertGreater(latest["row_count"], 0)
        self.assertGreater(latest["included_row_count"], 0)

    def test_dashboard_regression_validation_service_passes(self):
        result = DashboardRegressionValidationService.run_checks()

        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_check_count"], 0)
        self.assertGreaterEqual(result["check_count"], 6)


if __name__ == "__main__":
    unittest.main()
