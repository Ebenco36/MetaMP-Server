from __future__ import annotations

import glob
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from src.ingestion.dataset_layout import DatasetLayout


def get_latest_file(directory: str, basename_pattern: str) -> str:
    matches = glob.glob(os.path.join(directory, basename_pattern))
    dated_files = []

    for path in matches:
        name = os.path.basename(path)
        match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
        if match:
            try:
                file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
            except ValueError:
                continue
        else:
            file_date = datetime.min.date()
        dated_files.append((file_date, path))

    if not dated_files:
        raise FileNotFoundError(
            f"No files match {basename_pattern} in {directory}"
        )

    return max(dated_files, key=lambda item: item[0])[1]


def remove_columns_with_listlike_contents(df, exempt_columns=None):
    if exempt_columns is None:
        exempt_columns = []

    exempt = {col.lower() for col in exempt_columns}
    to_drop = []

    for col in df.columns:
        if col.lower() in exempt:
            continue
        if df[col].apply(
            lambda value: isinstance(value, str)
            and value.startswith("[")
            and value.endswith("]")
        ).any():
            to_drop.append(col)

    return df.drop(columns=to_drop)


def report_and_clean_missing_values(df, threshold=20, protected_columns=None):
    protected = {str(column).lower() for column in (protected_columns or [])}
    exempt = [
        "rcsb_entry_info_software_programs_combined",
        "audit_author",
        "citations",
        "citation",
        "rcsb_entry_info_resolution_combined",
    ]
    df = remove_columns_with_listlike_contents(df, exempt_columns=exempt)
    df = df.replace("NaN", np.nan)

    missing_pct = df.isna().mean() * 100
    keep_cols = missing_pct[missing_pct <= threshold].index
    keep_cols = [
        col
        for col in keep_cols
        if (("_id" not in col.lower() and "id_" not in col.lower()))
        or "uniprot" in col.lower()
    ]
    for column in df.columns:
        if column.lower() in protected and column not in keep_cols:
            keep_cols.append(column)
    df = df[keep_cols]

    extra_drops = {
        "ordering",
        "secondary_representations_count",
        "structure_subunits_count",
        "citations_count",
        "uniprotcodes",
        "subunits",
        "secondary_representations",
        "citations",
        "family_tcdb",
        "family_pfam",
        "family_primary_structures_count",
        "family_superfamily_pfam",
        "family_superfamily_families_count",
        "famsupclasstype_superfamilies_count",
        "famsupclasstype_type_classtypes_count",
        "species_primary_structures_count",
        "membrane_primary_structures_count",
        "description_y",
        "resolution_y",
        "bibliography",
        "audit_author",
        "is_replaced",
        "citation",
        "exptl",
        "pdbx_audit_revision_details",
        "pdbx_audit_revision_group",
        "pdbx_audit_revision_history",
        "pdbdatstatus_pdb_format_compatible",
        "pdbdatstatus_recvd_initial_deposition_date",
        "pdbdatstatus_status_code",
        "rcsaccinfo_deposit_date",
        "rcsaccinfo_has_released_experimental_data",
        "rcsaccinfo_initial_release_date",
        "rcsaccinfo_major_revision",
        "rcsaccinfo_minor_revision",
        "rcsaccinfo_revision_date",
        "rcsaccinfo_status_code",
        "rcspricitation_journal_abbrev",
        "rcspricitation_journal_volume",
        "rcspricitation_page_first",
        "rcspricitation_page_last",
        "rcspricitation_rcsb_authors",
        "rcspricitation_rcsb_journal_abbrev",
        "rcsb_primary_citation_title",
        "rcsb_primary_citation_year",
        "struct_title",
        "struct_keywords_pdbx_keywords",
        "struct_keywords_text",
        "pdbx_audit_revision_category",
        "pdbx_audit_revision_item",
        "pdbdatstatus_process_site",
        "pdbdatstatus_deposit_site",
        "pdbx_database_related",
        "rcspricitation_rcsb_orcididentifiers",
        "pdbx_database_status_sgentry",
        "struct_pdbx_caspflag",
        "citation_journal_abbrev",
        "citation_journal_volume",
        "citation_page_first",
        "citation_page_last",
        "citation_rcsb_authors",
        "citation_rcsb_is_primary",
        "citation_rcsb_journal_abbrev",
        "citation_title",
        "citation_year",
        "pdbaudrevision_details_data_content_type",
        "pdbaudrevision_details_ordinal",
        "pdbaudrevision_details_provider",
        "pdbaudrevision_details_revision_ordinal",
        "pdbaudrevision_details_type",
        "created_at",
        "updated_at",
        "topology_subunit",
        "topology_show_in",
        "resolution_x",
        "rcsentinfo_structure_determination_methodology",
        "rcsentinfo_structure_determination_methodology_priority",
        "id",
        "family_interpro",
        "refine_ls_restr",
        "rcsentinfo_resolution_combined",
        "pdbdatrelated_content_type",
        "pdbdatrelated_db_name",
        "pdbx_database_related_details",
        "pdbinirefinement_model_source_name",
        "pdbinirefinement_model_type",
        "em3d_reconstruction_symmetry_type",
        "em_ctf_correction_type",
        "em_entity_assembly_name",
        "em_entity_assembly_source",
        "em_entity_assembly_type",
        "emimarecording_avg_electron_dose_per_image",
        "emimarecording_film_or_detector_model",
        "rcsb_external_references_type",
        "rcsb_external_references_link",
    }

    safe_to_drop = {
        column
        for column in (set(df.columns) & extra_drops)
        if column.lower() not in protected
    }
    return df.drop(columns=safe_to_drop)


def deduplicate_valid_dataset(df, out_name: str):
    key_candidates = {
        "Quantitative_data.csv": [["Pdb Code", "Group", "Subgroup"], ["pdb_code", "group", "subgroup"]],
        "Mpstruct_dataset.csv": [["Pdb Code", "Group", "Subgroup"], ["pdb_code", "group", "subgroup"]],
        "PDB_data_transformed.csv": [["Pdb Code"], ["pdb_code"]],
        "NEWOPM.csv": [["pdbid"], ["pdb_code"]],
        "Uniprot_functions.csv": [["pdb_code", "uniprot_id"], ["Pdb Code", "uniprot_id"]],
    }

    for candidate_fields in key_candidates.get(out_name, []):
        if all(field in df.columns for field in candidate_fields):
            normalized = df.copy()
            for field in candidate_fields:
                normalized[field] = normalized[field].fillna("").astype(str).str.strip()
            keep_mask = ~normalized.duplicated(subset=candidate_fields, keep="first")
            return df.loc[keep_mask].reset_index(drop=True)

    return df


def write_valid_datasets(layout: DatasetLayout, progress_callback=None) -> None:
    layout.ensure_directories()
    jobs = [
        ("Quantitative_data_*.csv", "Quantitative_data.csv"),
        ("Uniprot_functions_*.csv", "Uniprot_functions.csv"),
        ("PDB_data_transformed_*.csv", "PDB_data_transformed.csv"),
        ("Mpstruct_dataset_*.csv", "Mpstruct_dataset.csv"),
        ("NEWOPM_*.csv", "NEWOPM.csv"),
    ]

    def report(message: str):
        if progress_callback:
            progress_callback(message)

    for pattern, out_name in jobs:
        canonical = layout.base_dir / out_name
        if canonical.exists():
            src = str(canonical)
            report(f"[clean_valid] Preparing {out_name} from current artifact {canonical.name}")
        else:
            report(f"[clean_valid] Preparing {out_name} from pattern {pattern}")
            try:
                src = get_latest_file(str(layout.base_dir), pattern)
            except FileNotFoundError:
                report(f"[clean_valid] Skipping {out_name}; no source artifact found")
                continue

        df = pd.read_csv(src, low_memory=False, encoding="utf-8")
        clean = report_and_clean_missing_values(df, threshold=90)
        deduplicated = deduplicate_valid_dataset(clean, out_name)
        removed_count = len(clean) - len(deduplicated)
        if removed_count > 0:
            report(f"[clean_valid] Removed {removed_count} duplicate row(s) from {out_name}")
        clean = deduplicated
        clean.to_csv(layout.valid_dir / out_name, index=False)
        report(
            f"[clean_valid] Wrote {out_name} with {len(clean)} row(s) and {len(clean.columns)} column(s)"
        )
