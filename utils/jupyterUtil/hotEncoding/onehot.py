import pandas as pd
from sklearn.preprocessing import OneHotEncoder

cat_list = [
'Group',
 'Subgroup',
 'Species',
 'Taxonomic Domain',
 'Expressed in Species',
 'symmetry_space_group_name_hm',
 'rcsb_entry_info_structure_determination_methodology',
 'rcsb_entry_info_diffrn_resolution_high_provenance_source',
 'rcsb_entry_info_selected_polymer_entity_types',
 'rcsb_entry_info_experimental_method',
 'rcsb_entry_info_na_polymer_entity_types',
 'rcsb_entry_info_polymer_composition',
 'exptl_method',
 'em2d_crystal_entity_space_group_name_hm'
]
"""
 
df = pd.read_csv("./Quantitative_data.csv", low_memory=False)
df_categorical_features = df.select_dtypes(include='object')
encoder = OneHotEncoder(sparse_output=False, drop='if_binary')  # 'sparse=False' returns a dense matrix, 'drop' handles multicollinearity
encoder.fit(df[cat_list])
one_hot_encoded = encoder.transform(df[cat_list])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(cat_list))
ds = pd.concat([df, one_hot_df], axis=1)
ds.to_csv("New_dataframe.csv")
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Read the CSV file
df = pd.read_csv("./Quantitative_data.csv", low_memory=False)


# List of columns to be removed
not_needed_columns = [
    'Unnamed: 0', 'Unnamed: 0_x', 'Unnamed: 0.1', 'Unnamed: 0_y', 'Unnamed: 0_1', 'Secondary Bibliogrpahies', 
    'Related Pdb Entries', 'rcsb_primary_citation_pdbx_database_id_pub_med', 'citation_pdbx_database_id_pub_med', 
    'rcsb_entry_container_identifiers_pubmed_id', "rcsb_accession_info_major_revision", "rcsb_accession_info_minor_revision",
    "rcsb_primary_citation_journal_id_csd", "rcsb_primary_citation_journal_volume", "rcsb_primary_citation_year", 
    "symmetry_int_tables_number", "pdbx_nmr_representative_conformer_id", "citation_journal_id_csd", "citation_journal_volume", 
    "citation_year", "diffrn_crystal_id", "diffrn_id", "diffrn_radiation_diffrn_id", "diffrn_radiation_wavelength_id", 
    "exptl_crystal_id", "bibliography_year", "em_experiment_entity_assembly_id", "em_experiment_id", "diffrn_detector_diffrn_id", 
    "diffrn_source_diffrn_id", "exptl_crystal_grow_crystal_id", "pdbx_reflns_twin_crystal_id", "pdbx_reflns_twin_diffrn_id",
    "pdbx_reflns_twin_domain_id", "pdbx_sgproject_id", "em3d_fitting_id", "em3d_reconstruction_id", "em3d_reconstruction_image_processing_id",
    "em_ctf_correction_em_image_processing_id", "em_ctf_correction_id", "em_entity_assembly_id", "em_entity_assembly_parent_id",
    "em_image_recording_id", "em_image_recording_imaging_id", "em_imaging_id", "em_imaging_specimen_id", "em_particle_selection_id", 
    "em_particle_selection_image_processing_id", "em_single_particle_entity_id", "em_single_particle_entity_image_processing_id",
    "em_software_id", "em_software_image_processing_id", "em_specimen_experiment_id", "em_specimen_id", "em_vitrification_id", "em_vitrification_specimen_id",
    "pdbx_nmr_exptl_conditions_id", "pdbx_nmr_exptl_experiment_id", "pdbx_nmr_exptl_solution_id", "pdbx_nmr_exptl_spectrometer_id", 
    "pdbx_nmr_exptl_sample_conditions_conditions_id", "pdbx_nmr_sample_details_solution_id", "pdbx_nmr_spectrometer_spectrometer_id",
    "em3d_fitting_list_id", "em3d_fitting_list_3d_fitting_id", "em_helical_entity_id", "em_helical_entity_image_processing_id",
    "pdbx_initial_refinement_model_id", "em3d_crystal_entity_id", "em3d_crystal_entity_image_processing_id", "em_diffraction_id", 
    "em_diffraction_imaging_id", "em_diffraction_shell_em_diffraction_stats_id", "em_diffraction_shell_id", "em_diffraction_stats_id",
    "em_diffraction_stats_image_processing_id", "em_embedding_id", "em_embedding_specimen_id", "pdbx_serial_crystallography_sample_delivery_diffrn_id",
    "pdbx_serial_crystallography_sample_delivery_injection_diffrn_id", "pdbx_serial_crystallography_sample_delivery_fixed_target_diffrn_id",
    "pdbx_serial_crystallography_data_reduction_diffrn_id", "pdbx_serial_crystallography_measurement_diffrn_id", "em_staining_id", "em_staining_specimen_id",
    "em2d_crystal_entity_id", "em2d_crystal_entity_image_processing_id", "rcsb_entry_info_structure_determination_methodology_priority",
    "audit_author_pdbx_ordinal", "pdbx_audit_revision_details_ordinal", "pdbx_audit_revision_details_revision_ordinal", 
    "pdbx_audit_revision_group_ordinal", "pdbx_audit_revision_group_revision_ordinal", "pdbx_audit_revision_history_major_revision",
    "pdbx_audit_revision_history_minor_revision", "pdbx_audit_revision_history_ordinal", "pdbx_audit_revision_category_ordinal",
    "pdbx_audit_revision_category_revision_ordinal", "pdbx_audit_revision_item_ordinal", "pdbx_audit_revision_item_revision_ordinal", 
    "reflns_pdbx_ordinal", "reflns_shell_pdbx_ordinal", "struct_keywords_pdbx_keywords", 
    "rcsb_entry_info_software_programs_combined", "rcsb_entry_info_nonpolymer_bound_components", "rcsb_entry_info_experimental_method_count",
    "refine_pdbx_rfree_selection_details", "refine_pdbx_ls_cross_valid_method", "refine_pdbx_method_to_determine_struct",
    "diffrn_detector_detector", "diffrn_detector_type", "diffrn_source_source", "diffrn_source_type", "exptl_crystal_grow_method"
]

# Drop unnecessary columns
df = df.drop(not_needed_columns, inplace=False, axis=1)

df[cat_list] = df[cat_list].apply(lambda x: x.str.strip())
# Define a dictionary for spelling corrections before other things
spelling_corrections = {
    'Methanocaldococcus jannaschi': 'Methanocaldococcus jannaschii',
    'Rhodopeudomonas blastica': 'Rhodopseudomonas blastica', 
    'Shewanella oneidensi': 'Shewanella oneidensis',
    'Synechocystis sp. pcc 6803': 'Synechocystis sp. PCC6803',
    'E. Coli': 'E. Coli',
    'E. Colli': 'E. Coli',
    'E.coli': 'E. Coli',
    'e. Coli': 'E. Coli',
    'Escherichia coli': 'E. Coli',
    'HEK 293S cells': 'HEK-293S cells',
    'HEK-293S cells': 'HEK-293S cells',
    'HEK293S cells': 'HEK-293S cells',
    'HEK293s cells': 'HEK-293S cells',
    'S. Crevisiae': 'S. Cerevisiae',
    'S. frugiperda': 'S. Frugiperda',
    'Sf9 cells': 'Sf9 cells',
    'sf9 cells': 'Sf9 cells',
    'Trichoplusia ni': 'Trichoplusia ni',
    'Trichoplusia ni': 'Trichoplusia ni',
    'Trichoplusia ni)': 'Trichoplusia ni'
}

# Apply spelling corrections to the specified columns
df.replace({'Expressed in Species': spelling_corrections, 'Species': spelling_corrections}, inplace=True)


# Apply transformations to DataFrame columns by removing spaces and converting to a lower case
df[cat_list] = df[cat_list].apply(lambda x: x.str.lower().str.replace(' ', '_'))


# Select categorical features
df_categorical_features = df.select_dtypes(include='object')

# Define the categorical column to group by
column_to_group_by = 'rcsb_entry_info_experimental_method'  # Replace with your actual column name

# Get unique values in the specified column
unique_values = df[column_to_group_by].unique()

# One-hot encode each subset separately and save as individual CSV files
for value in unique_values:
    # Filter the DataFrame based on the unique value in the specified column
    subset_df = df[df[column_to_group_by] == value]

    # One-hot encode the subset
    encoder = OneHotEncoder(sparse_output=False, drop='if_binary')
    encoder.fit(subset_df[cat_list])
    one_hot_encoded = encoder.transform(subset_df[cat_list])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(cat_list))
    
    # Reset index if needed
    one_hot_df.reset_index(drop=True, inplace=True)
    subset_df.reset_index(drop=True, inplace=True)

    # Concatenate the one-hot encoded subset with the original subset
    subset_result_df = pd.concat([subset_df, one_hot_df], axis=1)

    # Save the subset DataFrame to a CSV file
    subset_result_df.to_csv(f"New_dataframe_{value}.csv", index=False)
    subset_df.to_csv(f"NoOneHotNew_dataframe_{value}.csv", index=False)
