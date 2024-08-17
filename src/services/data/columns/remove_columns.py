
"""
Remove columns that are not needed
"""
not_needed_columns = [
    'Unnamed: 0', 'Unnamed: 0_x', 
    'Unnamed: 0.1', 'Unnamed: 0_y', 'Unnamed: 0_1',
    'Secondary Bibliogrpahies', 'Related Pdb Entries', 'rcsb_primary_citation_pdbx_database_id_pub_med', 
    'citation_pdbx_database_id_pub_med', 'rcsb_entry_container_identifiers_pubmed_id'
]

not_needed_columns2 = [
    'Unnamed: 0', 'Unnamed: 0_x', 
    'Unnamed: 0.1', 'Unnamed: 0_y', 'Unnamed: 0_1',
    'Secondary Bibliogrpahies', 'Related Pdb Entries', 'rcsb_primary_citation_pdbx_database_id_pub_med', 
    'citation_pdbx_database_id_pub_med', 'rcsb_entry_container_identifiers_pubmed_id',
    "rcsb_accession_info_major_revision", "rcsb_accession_info_minor_revision",
    "rcsb_primary_citation_journal_id_csd", "rcsb_primary_citation_journal_volume",
    "rcsb_primary_citation_year", "symmetry_int_tables_number", "pdbx_nmr_representative_conformer_id",
    "citation_journal_id_csd", "citation_journal_volume", "citation_year", "diffrn_crystal_id",
    "diffrn_id", "diffrn_radiation_diffrn_id", "diffrn_radiation_wavelength_id", "exptl_crystal_id",
    "bibliography_year", "em_experiment_entity_assembly_id", "em_experiment_id", "diffrn_detector_diffrn_id", 
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
    "em2d_crystal_entity_id", "em2d_crystal_entity_image_processing_id"
]