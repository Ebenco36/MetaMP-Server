"""
Descriptors for analysis
"""

descriptors = [
    'Group', 'Subgroup', 'Pdb Code', 'Is Master Protein?', 'Name', 'Species', 'Taxonomic Domain', 
    'Expressed in Species', 'Description', 'Bibliography', 'Member Proteins', 'audit_author', 
    'citation', 'software', 'rcsb_id', 'symmetry.space_group_name_hm', 
    'diffrn_detector', 'symmetry.cell_setting' 
]


em_descriptors = [
    'em3d_fitting_list', 'em_helical_entity', 
    'em_diffraction', 'em_diffraction_shell', 'em_diffraction_stats', 
    'em_embedding', 'em_staining', 'em2d_crystal_entity', 
    'em_experiment.aggregation_state', 'em_experiment.entity_assembly_id', 
    'em_experiment.id', 'em_experiment.reconstruction_method', 'em3d_fitting', 
    'em3d_reconstruction', 'em_ctf_correction', 'em_entity_assembly', 'em_image_recording', 
    'em_imaging', 'em_particle_selection', 'em_single_particle_entity', 
    'em_software', 'em_specimen', 'em_vitrification'
]

struct_descriptors = [
    'struct.pdbx_model_details', 'struct.pdbx_model_type_details', 'struct.pdbx_caspflag', 
    'struct.pdbx_descriptor', 'struct.title', 'struct_keywords.pdbx_keywords', 'struct_keywords.text'
]

pdbx_descriptors = [
    'pdbx_related_exp_data_set', 'pdbx_reflns_twin', 'pdbx_audit_support', 'pdbx_database_related',
    'pdbx_database_pdbobs_spr', 'pdbx_sgproject', 'pdbx_initial_refinement_model', 
    'pdbx_molecule_features', 'pdbx_audit_revision_details', 'pdbx_audit_revision_group', 
    'pdbx_audit_revision_history', 'pdbx_audit_revision_category', 'pdbx_audit_revision_item'
]

pdbx_database_descriptors = [
    'pdbx_database_status.status_code_mr', 'pdbx_database_status.methods_development_category',
    'pdbx_database_status.status_code_cs', 'pdbx_database_status.pdb_format_compatible', 
    'pdbx_database_status.recvd_initial_deposition_date', 'pdbx_database_status.status_code', 
    'pdbx_database_status.methods_development_category', 'pdbx_database_status.process_site',
    'pdbx_database_status.status_code_sf', 'pdbx_database_status.sgentry', 
    'pdbx_database_status.deposit_site'
]

rcsb_descriptors = [
    'rcsb_accession_info.has_released_experimental_data', 'rcsb_accession_info.major_revision', 
    'rcsb_accession_info.minor_revision', 'rcsb_accession_info.status_code', 
    'rcsb_entry_container_identifiers.assembly_ids', 'rcsb_entry_container_identifiers.entity_ids', 
    'rcsb_entry_container_identifiers.model_ids', 'rcsb_entry_container_identifiers.non_polymer_entity_ids', 
    'rcsb_entry_container_identifiers.polymer_entity_ids', 'rcsb_entry_container_identifiers.rcsb_id', 
    'rcsb_entry_container_identifiers.pubmed_id', 'rcsb_entry_info.experimental_method', 
    'rcsb_entry_info.experimental_method_count', 'rcsb_entry_info.na_polymer_entity_types', 
    'rcsb_entry_info.nonpolymer_bound_components', 'rcsb_entry_info.polymer_composition', 
    'rcsb_entry_info.selected_polymer_entity_types', 'rcsb_entry_info.software_programs_combined', 
    'rcsb_entry_info.structure_determination_methodology', 'rcsb_external_references',
    'rcsb_entry_info.diffrn_resolution_high.provenance_source', 'rcsb_primary_citation.country', 
    'rcsb_primary_citation.id', 'rcsb_primary_citation.journal_abbrev', 'rcsb_binding_affinity',
    'rcsb_primary_citation.pdbx_database_id_doi', 'rcsb_primary_citation.rcsb_authors', 
    'rcsb_primary_citation.rcsb_journal_abbrev', 'rcsb_primary_citation.title',
    'rcsb_entry_info.ndb_struct_conf_na_feature_combined', 'rcsb_external_references', 
    'rcsb_entry_container_identifiers.emdb_ids', 'rcsb_entry_container_identifiers.related_emdb_ids',
    'rcsb_entry_container_identifiers.branched_entity_ids', 'rcsb_primary_citation.rcsb_orcididentifiers'
]

pdbx_nmr_descriptors = [
    'pdbx_nmr_exptl', 'pdbx_nmr_exptl_sample_conditions', 
    'pdbx_nmr_refine', 'pdbx_nmr_sample_details', 'pdbx_nmr_software', 
    'pdbx_nmr_spectrometer', 'pdbx_nmr_ensemble.conformer_selection_criteria',
    'pdbx_nmr_ensemble.average_torsion_angle_constraint_violation', 
    'pdbx_nmr_ensemble.maximum_torsion_angle_constraint_violation', 
    'pdbx_nmr_ensemble.torsion_angle_constraint_violation_method',
    'pdbx_nmr_ensemble.maximum_lower_distance_constraint_violation', 
    'pdbx_nmr_ensemble.maximum_upper_distance_constraint_violation',
    'pdbx_nmr_ensemble.conformers_calculated_total_number', 
    'pdbx_nmr_representative.conformer_id', 'pdbx_nmr_representative.selection_criteria', 
    'pdbx_nmr_details.text', 'pdbx_nmr_ensemble.representative_conformer'
]


pdbx_serial_descriptors = [
    'pdbx_serial_crystallography_sample_delivery',
    'pdbx_serial_crystallography_sample_delivery_injection', 
    'pdbx_serial_crystallography_sample_delivery_fixed_target', 
    'pdbx_serial_crystallography_data_reduction', 
    'pdbx_serial_crystallography_measurement'
]


all_descriptors = descriptors + em_descriptors + rcsb_descriptors + struct_descriptors + pdbx_descriptors + pdbx_nmr_descriptors\
    + pdbx_serial_descriptors + pdbx_database_descriptors