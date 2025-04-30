import os
import pandas as pd
import numpy as np

modified_path = "."

def remove_columns_with_listlike_contents(df, exempt_columns=None):
    if exempt_columns is None:
        exempt_columns = []

    # Normalize exempt_columns to lower case for case-insensitive comparison
    exempt_columns = [col.lower() for col in exempt_columns]

    # Initialize an empty list to hold the names of columns to drop
    columns_to_drop = []

    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Skip columns that are in the exempt list
        if column.lower() in exempt_columns:
            continue
        # Check if any cell in the column contains a string starting with '[' and ending with ']'
        if df[column].apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')).any():
            columns_to_drop.append(column)

    # Drop the identified columns from the DataFrame
    df_cleaned = df.drop(columns=columns_to_drop)

    return df_cleaned


def report_and_clean_missing_values(df, threshold=20):
    exempt = [
        "rcsb_entry_info_software_programs_combined", 
        "audit_author", 
        "citations", 
        "citation", 
        "rcsb_entry_info_resolution_combined"
    ]
    df = remove_columns_with_listlike_contents(df, exempt_columns=exempt)

    # Replace 'NaN' strings with np.nan to unify missing value representations
    df.replace('NaN', np.nan, inplace=True)

    # Calculate the number of missing values in each column
    missing_count = df.isna().sum()
    
    # Calculate the percentage of missing values in each column
    missing_percentage = (missing_count / len(df)) * 100
    
    # Create a DataFrame to hold the results
    missing_report = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage': missing_percentage
    })

    # Determine columns to keep by applying the threshold filter
    columns_to_keep = missing_percentage[missing_percentage <= threshold].index

    # Further filter out any columns whose names contain '_id' or 'id_'
    columns_to_keep = [
        col for col in columns_to_keep
        if ('_id' not in col.lower() and 'id_' not in col.lower()) or 'uniprot' in col.lower()
    ]

    # Select the columns to keep in the cleaned DataFrame
    df_cleaned = df[columns_to_keep]
    columns_to_drop = [
        "ordering", "secondary_representations_count", "structure_subunits_count", 
        "citations_count", "uniprotcodes", "subunits", "secondary_representations",
        "citations", "family_tcdb", "family_pfam", "family_primary_structures_count",
        "family_superfamily_pfam", "family_superfamily_families_count",
        "famsupclasstype_superfamilies_count", "famsupclasstype_type_classtypes_count",
        "species_primary_structures_count", "membrane_primary_structures_count",
        "description_y", "resolution_y", "bibliography", "audit_author", "is_replaced", "citation", "exptl",
        "pdbx_audit_revision_details", "pdbx_audit_revision_group", "pdbx_audit_revision_history",
        "pdbdatstatus_pdb_format_compatible", "pdbdatstatus_recvd_initial_deposition_date",
        "pdbdatstatus_status_code", "rcsaccinfo_deposit_date", "rcsaccinfo_has_released_experimental_data",
        "rcsaccinfo_initial_release_date", "rcsaccinfo_major_revision", "rcsaccinfo_minor_revision",
        "rcsaccinfo_revision_date", "rcsaccinfo_status_code", 
        "rcspricitation_journal_abbrev", "rcspricitation_journal_volume", "rcspricitation_page_first",
        "rcspricitation_page_last", "rcspricitation_rcsb_authors", "rcspricitation_rcsb_journal_abbrev",
        "rcsb_primary_citation_title", "rcsb_primary_citation_year", "struct_title", "struct_keywords_pdbx_keywords",
        "struct_keywords_text", "pdbx_audit_revision_category", "pdbx_audit_revision_item", "pdbdatstatus_process_site",
        "pdbdatstatus_deposit_site", "pdbx_database_related", "rcspricitation_rcsb_orcididentifiers",
        "pdbx_database_status_sgentry", "struct_pdbx_caspflag",  "citation_journal_abbrev",
        "citation_journal_volume", "citation_page_first", "citation_page_last", "citation_rcsb_authors",
        "citation_rcsb_is_primary", "citation_rcsb_journal_abbrev", "citation_title", "citation_year",
        "pdbaudrevision_details_data_content_type", "pdbaudrevision_details_ordinal", "pdbaudrevision_details_provider",
        "pdbaudrevision_details_revision_ordinal", "pdbaudrevision_details_type", "created_at", "updated_at", 
        "topology_subunit", "topology_show_in", "resolution_x", "rcsentinfo_structure_determination_methodology",
        "rcsentinfo_structure_determination_methodology_priority", "id", "family_interpro", "refine_ls_restr",
        "rcsentinfo_resolution_combined",
        "pdbdatrelated_content_type", "pdbdatrelated_db_name", "pdbx_database_related_details", 
        "pdbinirefinement_model_source_name", "pdbinirefinement_model_type", "em3d_reconstruction_symmetry_type"
        "em_ctf_correction_type", "em_entity_assembly_name", "em_entity_assembly_source", "em_entity_assembly_type",
        "emimarecording_avg_electron_dose_per_image", "emimarecording_film_or_detector_model", "rcsb_external_references_type", "rcsb_external_references_link"
    ]

    # Use set intersection to safely filter columns that exist in the DataFrame
    safe_columns_to_drop = set(df_cleaned.columns) & set(columns_to_drop)

    # Drop the safe columns
    df_cleaned = df_cleaned.drop(columns=safe_columns_to_drop, inplace=False)


    return df_cleaned


def identify_categorical_columns(df, unique_threshold=2):
    df = report_and_clean_missing_values(df)
    
    # Filter out columns starting with 'id_' or ending with '_id'
    filtered_columns = [col for col in df.columns if not col.startswith('id_') and not col.endswith('_id')]
    df = df[filtered_columns]

    categorical_cols = []

    for col in df.columns:
        # Check if the column data type is object or category
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
        else:
            # Check the number of unique values relative to the total number of values
            num_unique_values = df[col].nunique()
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.05 and num_unique_values <= unique_threshold:
                categorical_cols.append(col)
    
    return categorical_cols


merged_df = pd.read_csv(modified_path + '/datasets/Quantitative_data.csv',low_memory=False, encoding='utf-8')
valid_dir = os.path.join(modified_path, 'datasets', 'valid')
os.makedirs(valid_dir, exist_ok=True)
valid_data_merged_df = report_and_clean_missing_values(merged_df, threshold=90)
valid_data_merged_df.to_csv(modified_path + '/datasets/valid/Quantitative_data.csv', index=False)



opm = pd.read_csv(modified_path + '/datasets/NEWOPM.csv', low_memory=False, encoding='utf-8')

valid_data_opm = report_and_clean_missing_values(opm, threshold=90)
valid_data_opm.to_csv(modified_path + '/datasets/valid/NEWOPM.csv', index=False)



pdb = pd.read_csv(modified_path + '/datasets/PDB_data_transformed.csv', low_memory=False, encoding='utf-8')

valid_data_pdb = report_and_clean_missing_values(pdb, threshold=90)
valid_data_pdb.to_csv(modified_path + '/datasets/valid/PDB_data_transformed.csv', index=False)



mpstruc = pd.read_csv(modified_path + '/datasets/Mpstruct_dataset.csv', low_memory=False, encoding='utf-8')

valid_data_mpstruc = report_and_clean_missing_values(mpstruc, threshold=90)
valid_data_mpstruc.to_csv(modified_path + '/datasets/valid/Mpstruct_dataset.csv', index=False)