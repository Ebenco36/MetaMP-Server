import os
import glob
import re
import pandas as pd
import numpy as np
from datetime import datetime

# ——————————————————————————————————————————————————————————————
# Helpers
# ——————————————————————————————————————————————————————————————

def get_latest_file(directory: str, basename_pattern: str) -> str:
    """
    In `directory`, find files matching `basename_pattern` (a glob, e.g. "Quantitative_data_*.csv"),
    extract dates of the form YYYY-MM-DD from their names, and return the file with the newest date.
    """
    matches = glob.glob(os.path.join(directory, basename_pattern))
    dated_files = []
    for path in matches:
        name = os.path.basename(path)
        m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
        if m:
            try:
                d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            except ValueError:
                continue
        else:
            # fallback: treat undated files as epoch
            d = datetime.min.date()
        dated_files.append((d, path))
    if not dated_files:
        raise FileNotFoundError(f"No files match {basename_pattern} in {directory}")
    # pick the one with max date
    return max(dated_files, key=lambda x: x[0])[1]


def remove_columns_with_listlike_contents(df, exempt_columns=None):
    if exempt_columns is None:
        exempt_columns = []
    exempt = {col.lower() for col in exempt_columns}
    to_drop = []
    for col in df.columns:
        if col.lower() in exempt:
            continue
        if df[col].apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')).any():
            to_drop.append(col)
    return df.drop(columns=to_drop)


def report_and_clean_missing_values(df, threshold=20):
    exempt = [
        "rcsb_entry_info_software_programs_combined",
        "audit_author",
        "citations",
        "citation",
        "rcsb_entry_info_resolution_combined"
    ]
    df = remove_columns_with_listlike_contents(df, exempt_columns=exempt)
    df = df.replace('NaN', np.nan)

    missing_pct = df.isna().mean() * 100
    keep_cols = missing_pct[missing_pct <= threshold].index
    # drop any col containing '_id' or 'id_' unless it mentions 'uniprot'
    keep_cols = [
        c for c in keep_cols
        if (('_id' not in c.lower() and 'id_' not in c.lower()) or 'uniprot' in c.lower())
    ]
    df = df[keep_cols]

    # some extra manual drops
    extra_drops = {
        "ordering", "secondary_representations_count", "structure_subunits_count",
        "citations_count", "uniprotcodes", "subunits", "secondary_representations",
        "citations", "family_tcdb", "family_pfam", "family_primary_structures_count",
        "family_superfamily_pfam", "family_superfamily_families_count",
        "famsupclasstype_superfamilies_count", "famsupclasstype_type_classtypes_count",
        "species_primary_structures_count", "membrane_primary_structures_count",
        "description_y", "resolution_y", "bibliography", "audit_author", "is_replaced",
        "citation", "exptl", "pdbx_audit_revision_details", "pdbx_audit_revision_group",
        "pdbx_audit_revision_history", "pdbdatstatus_pdb_format_compatible",
        "pdbdatstatus_recvd_initial_deposition_date", "pdbdatstatus_status_code",
        "rcsaccinfo_deposit_date", "rcsaccinfo_has_released_experimental_data",
        "rcsaccinfo_initial_release_date", "rcsaccinfo_major_revision",
        "rcsaccinfo_minor_revision", "rcsaccinfo_revision_date", "rcsaccinfo_status_code",
        "rcspricitation_journal_abbrev", "rcspricitation_journal_volume",
        "rcspricitation_page_first", "rcspricitation_page_last",
        "rcspricitation_rcsb_authors", "rcspricitation_rcsb_journal_abbrev",
        "rcsb_primary_citation_title", "rcsb_primary_citation_year", "struct_title",
        "struct_keywords_pdbx_keywords", "struct_keywords_text",
        "pdbx_audit_revision_category", "pdbx_audit_revision_item",
        "pdbdatstatus_process_site", "pdbdatstatus_deposit_site",
        "pdbx_database_related", "rcspricitation_rcsb_orcididentifiers",
        "pdbx_database_status_sgentry", "struct_pdbx_caspflag",
        "citation_journal_abbrev", "citation_journal_volume",
        "citation_page_first", "citation_page_last", "citation_rcsb_authors",
        "citation_rcsb_is_primary", "citation_rcsb_journal_abbrev",
        "citation_title", "citation_year", "pdbaudrevision_details_data_content_type",
        "pdbaudrevision_details_ordinal", "pdbaudrevision_details_provider",
        "pdbaudrevision_details_revision_ordinal", "pdbaudrevision_details_type",
        "created_at", "updated_at", "topology_subunit", "topology_show_in",
        "resolution_x", "rcsentinfo_structure_determination_methodology",
        "rcsentinfo_structure_determination_methodology_priority", "id",
        "family_interpro", "refine_ls_restr", "rcsentinfo_resolution_combined",
        "pdbdatrelated_content_type", "pdbdatrelated_db_name",
        "pdbx_database_related_details", "pdbinirefinement_model_source_name",
        "pdbinirefinement_model_type", "em3d_reconstruction_symmetry_type",
        "em_ctf_correction_type", "em_entity_assembly_name",
        "em_entity_assembly_source", "em_entity_assembly_type",
        "emimarecording_avg_electron_dose_per_image",
        "emimarecording_film_or_detector_model",
        "rcsb_external_references_type", "rcsb_external_references_link"
    }
    safe_to_drop = set(df.columns) & extra_drops
    return df.drop(columns=safe_to_drop)


# ——————————————————————————————————————————————————————————————
# Main cleaning pipeline
# ——————————————————————————————————————————————————————————————

if __name__ == "__main__":
    base = os.path.join(modified_path := ".", "datasets")
    valid_dir = os.path.join(base, "valid")
    os.makedirs(valid_dir, exist_ok=True)

    # Define each dataset’s glob pattern and output name
    jobs = [
        ("Quantitative_data_*.csv",        "Quantitative_data.csv"),
        ("Uniprot_functions_*.csv",        "Uniprot_functions.csv"),
        ("PDB_data_transformed_*.csv",     "PDB_data_transformed.csv"),
        ("Mpstruct_dataset_*.csv",         "Mpstruct_dataset.csv"),
    ]

    for pattern, out_name in jobs:
        try:
            src = get_latest_file(base, pattern)
        except FileNotFoundError as e:
            print(e)
            continue

        print(f"Cleaning {os.path.basename(src)} → {out_name}")
        df = pd.read_csv(src, low_memory=False, encoding='utf-8')
        clean = report_and_clean_missing_values(df, threshold=90)
        dst = os.path.join(valid_dir, out_name)
        clean.to_csv(dst, index=False)
        print(f"  ✔ Wrote cleaned data to valid/{out_name}")
