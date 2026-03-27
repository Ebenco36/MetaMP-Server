"""
    This method provides data filter for our basic summary statistics
    @return list
    Param None
"""
from src.Commands.Migration.classMigrate import Migration


def stats_data(for_processing=True):
    data = []
    # trying to group these sections accordingly.
    section1 = [
        {
            "value" : "rcsb_entry_info_selected_polymer_entity_types", 
            "name"  : "Polymer Entity Types"
        },
        {
            "value" : "species", 
            "name"  : "Engineered Source Organism"
        },
        {
            "value" : "expressed in species", 
            "name"  : "Expression System Organism"
        },
        {
            "value" : "processed_resolution", 
            "name"  :  "Resolution"
        },
        {
            "value" : "rcsb_entry_info_software_programs_combined", 
            "name"  : "Software"
        },
        {
            "value" : "symmetry_space_group_name_hm", 
            "name"  : "Space Group"
        },
        {
            "value" : "rcsb_entry_info_molecular_weight", 
            "name"  :  "Molecular Weight (Structure)"
        },
        {
            "value" : "rcsb_entry_info_deposited_atom_count", 
            "name"  :  "Atom Count"
        },
        {
            "value" : "rcsb_entry_info_experimental_method", 
            "name"  : "Experimental Method Overview"
        },
        {
            "value" : "Group", 
            "name"  : "Groups"
        },
        {
            "value" : "rcspricitation_rcsb_journal_abbrev", 
            "name"  :  "Journal"
        },
        {
            "value" : "exptl_crystal_grow_method1",
            "name" : "Growth Method"
        },
        # {
        #    "value" : "pdbx_sgproject_full_name_of_center", 
        #    "name"  :  "by Structural Genomics Centers"
        # },
    ]
    sub_section1 = [
        
    ]
    section2 = [
        
        {
            "value" : "rcsb_entry_info_experimental_method*X-ray", 
            "name"  : "by X-ray Crystallography (X-ray)",
        },
        {
            "value" : "rcsb_entry_info_experimental_method*NMR", 
            "name"  : "by Nuclear magnetic resonance (NMR)",
        },
        {
            "value" : "rcsb_entry_info_experimental_method*EM", 
            "name"  : "Electron Microscopy (EM)",
        },
        {
            "value" : "rcsb_entry_info_experimental_method*Multiple methods", 
            "name"  : "Multi-method",
        },
        {
            "value" : "rcsb_entry_info_selected_polymer_entity_types*Protein (only)", 
            "name"  : "Protein-only",
        },
        
        {
            "value" : "Expressed in Species*E. Coli", 
            "name"  : "Expressed in Species (E. Coli)",
        },
        {
            "value" : "Expressed in Species*HEK293 cells", 
            "name"  : "Expressed in Species (HEK293 cells)",
        },
        {
            "value" : "Expressed in Species*S. Frugiperda", 
            "name"  : "Expressed in Species (S. Frugiperda)",
        },
        {
            "value" : "Group*MONOTOPIC MEMBRANE PROTEINS", 
            "name"  : "Monotopic Membrane Proteins",
        },
        {
            "value" : "Group*TRANSMEMBRANE PROTEINS:ALPHA-HELICAL", 
            "name"  : "Transmembrane Proteins:Alpha-Helical",
        },
        {
            "value" : "Group*TRANSMEMBRANE PROTEINS:BETA-BARREL", 
            "name"  : "Transmembrane Proteins:Beta-Barrel",
        },
        {
            "value" : "Taxonomic Domain", 
            "name"  : "Taxonomic Domain",
        },
        {
            "value" : "Taxonomic Domain*Bacteria", 
            "name"  : "Taxonomic Domain (Bacteria)",
        },
        {
            "value" : "Taxonomic Domain*Eukaryota", 
            "name"  : "Taxonomic Domain (Eukaryota)",
        },
        {
            "value" : "Taxonomic Domain*Viruses", 
            "name"  : "Taxonomic Domain (Viruses)",
        },
        {
            "value" : "Taxonomic Domain*Unclassified", 
            "name"  : "Taxonomic Domain (Unclassified)",
        }
        #{
        #    "value" : "rcsb_entry_info_selected_polymer_entity_types*ProteinNA", 
        #    "name"  : "by Protein-Nucleic Acid Complexes",
        #},
        # {
        #     "value" : "rcsb_entry_info_experimental_method*AS", 
        #     "name"  : "by Assembly Symmetry",
        # },
        #{
        #    "value" : "rcsb_entry_info_experimental_method", 
        #    "name"  : "By Methods"
        #}
    ]
    
    section3 = [
        
        
    ]
    
    if(not for_processing):
        section1 = reduce_value_length(section1)
        sub_section1 = reduce_value_length(sub_section1, False)
        section2 = reduce_value_length(section2, False)
        section3 = reduce_value_length(section3, False)
    
    data.append({
        "section": 'data_distribution', 
        "data": section1[:2] + sub_section1 + section1[2:]
    })

    data.append({
        "section": 'released_structure_per_year', 
        "data": section2
    })
    
    # data.append({
    #     "section": 'Other Variables', 
    #     "data": section3
    # })


    return data


def array_string_type ():
    array_list = ['rcsentinfo_software_programs_combined']

    return array_list


def reduce_value_length(data, strip_special_character=True):
    for item in data:
        if 'value' in item: 
            item['value'] = Migration.shorten_column_name(item['value'], strip_special_character)
    return data


def reduce_value_length_version2(data):
    res = []
    for item in data: 
        res.append(Migration.shorten_column_name(item))
    return res

def general_columns():
    return [
        "group", "species", "pdb_code", "name", "taxonomic_domain", "resolution", "rcsentinfo_experimental_method"
    ]
    
def EM_columns(include_general=True):
    return reduce_value_length_version2([
        "rcsb_entry_info_assembly_count",
        "rcsb_entry_info_branched_entity_count",
        "rcsb_entry_info_cis_peptide_count",
        "rcsb_entry_info_deposited_atom_count",
        "rcsb_entry_info_deposited_hydrogen_atom_count",
        "rcsb_entry_info_deposited_model_count",
        "rcsb_entry_info_deposited_modeled_polymer_monomer_count",
        "rcsb_entry_info_deposited_nonpolymer_entity_instance_count",
        "rcsb_entry_info_deposited_polymer_entity_instance_count",
        "rcsb_entry_info_deposited_polymer_monomer_count",
        "rcsb_entry_info_deposited_solvent_atom_count",
        "rcsb_entry_info_deposited_unmodeled_polymer_monomer_count",
        "rcsb_entry_info_disulfide_bond_count",
        "rcsb_entry_info_entity_count",
        "rcsb_entry_info_inter_mol_covalent_bond_count",
        "rcsb_entry_info_inter_mol_metalic_bond_count",
        "rcsb_entry_info_molecular_weight",
        "rcsb_entry_info_nonpolymer_entity_count",
        "rcsb_entry_info_polymer_entity_count",
        "rcsb_entry_info_polymer_entity_count_dna",
        "rcsb_entry_info_polymer_entity_count_rna",
        "rcsb_entry_info_polymer_entity_count_nucleic_acid",
        "rcsb_entry_info_polymer_entity_count_protein",
        "rcsb_entry_info_polymer_entity_taxonomy_count",
        "rcsb_entry_info_polymer_molecular_weight_maximum",
        "rcsb_entry_info_polymer_molecular_weight_minimum",
        "rcsb_entry_info_polymer_monomer_count_maximum",
        "rcsb_entry_info_polymer_monomer_count_minimum",
        "rcsb_entry_info_solvent_entity_count",
        "em3d_reconstruction_num_particles",
        "em3d_reconstruction_resolution",
        "em_image_recording_avg_electron_dose_per_image",
        "em_imaging_accelerating_voltage"

    ] + (general_columns() if include_general else []))
    
def X_ray_columns(include_general=True):
    return reduce_value_length_version2([
        "cell_angle_alpha",
        "cell_angle_beta",
        "cell_angle_gamma","cell_length_a","cell_length_b","cell_length_c","cell_zpdb","rcsb_entry_info_assembly_count","rcsb_entry_info_branched_entity_count","rcsb_entry_info_cis_peptide_count","rcsb_entry_info_deposited_atom_count","rcsb_entry_info_deposited_hydrogen_atom_count","rcsb_entry_info_deposited_model_count","rcsb_entry_info_deposited_modeled_polymer_monomer_count","rcsb_entry_info_deposited_nonpolymer_entity_instance_count","rcsb_entry_info_deposited_polymer_entity_instance_count","rcsb_entry_info_deposited_polymer_monomer_count","rcsb_entry_info_deposited_solvent_atom_count","rcsb_entry_info_deposited_unmodeled_polymer_monomer_count","rcsb_entry_info_disulfide_bond_count","rcsb_entry_info_entity_count","rcsb_entry_info_inter_mol_covalent_bond_count","rcsb_entry_info_inter_mol_metalic_bond_count","rcsb_entry_info_molecular_weight","rcsb_entry_info_nonpolymer_entity_count","rcsb_entry_info_nonpolymer_molecular_weight_maximum","rcsb_entry_info_nonpolymer_molecular_weight_minimum","rcsb_entry_info_polymer_entity_count","rcsb_entry_info_polymer_entity_count_dna","rcsb_entry_info_polymer_entity_count_rna","rcsb_entry_info_polymer_entity_count_nucleic_acid","rcsb_entry_info_polymer_entity_count_protein","rcsb_entry_info_polymer_entity_taxonomy_count","rcsb_entry_info_polymer_molecular_weight_maximum","rcsb_entry_info_polymer_molecular_weight_minimum","rcsb_entry_info_polymer_monomer_count_maximum","rcsb_entry_info_polymer_monomer_count_minimum","rcsb_entry_info_solvent_entity_count","rcsb_entry_info_diffrn_resolution_high_value","rcsb_entry_info_diffrn_radiation_wavelength_maximum","rcsb_entry_info_diffrn_radiation_wavelength_minimum","diffrn_ambient_temp","exptl_crystals_number","exptl_crystal_density_matthews","exptl_crystal_density_percent_sol","refine_ls_rfactor_rfree","refine_ls_rfactor_rwork","refine_ls_rfactor_obs","refine_ls_dres_high","refine_ls_dres_low","refine_ls_number_reflns_obs","refine_pdbx_ls_sigma_f","refine_ls_percent_reflns_rfree","refine_ls_percent_reflns_obs","refine_biso_mean","refine_ls_number_reflns_rfree","refine_overall_suml","refine_pdbx_solvent_shrinkage_radii","refine_pdbx_solvent_vdw_probe_radii","refine_hist_d_res_high","refine_hist_d_res_low","refine_hist_number_atoms_solvent","refine_hist_number_atoms_total","refine_hist_pdbx_number_atoms_ligand","refine_hist_pdbx_number_atoms_nucleic_acid","refine_hist_pdbx_number_atoms_protein","reflns_d_resolution_high","reflns_d_resolution_low","reflns_number_obs","reflns_pdbx_ordinal","reflns_pdbx_redundancy","reflns_percent_possible_obs","reflns_pdbx_net_iover_sigma_i","exptl_crystal_grow_p_h","exptl_crystal_grow_temp","reflns_shell_d_res_high","reflns_shell_d_res_low"
    ] + (general_columns() if include_general else []))
    

def NMR_columns(include_general=True):
    return reduce_value_length_version2([
        "rcsb_entry_info_assembly_count","rcsb_entry_info_branched_entity_count","rcsb_entry_info_cis_peptide_count","rcsb_entry_info_deposited_atom_count","rcsb_entry_info_deposited_hydrogen_atom_count","rcsb_entry_info_deposited_model_count","rcsb_entry_info_deposited_modeled_polymer_monomer_count","rcsb_entry_info_deposited_nonpolymer_entity_instance_count","rcsb_entry_info_deposited_polymer_entity_instance_count","rcsb_entry_info_deposited_polymer_monomer_count","rcsb_entry_info_deposited_solvent_atom_count","rcsb_entry_info_deposited_unmodeled_polymer_monomer_count","rcsb_entry_info_disulfide_bond_count","rcsb_entry_info_entity_count","rcsb_entry_info_inter_mol_covalent_bond_count","rcsb_entry_info_inter_mol_metalic_bond_count","rcsb_entry_info_molecular_weight","rcsb_entry_info_nonpolymer_entity_count","rcsb_entry_info_polymer_entity_count","rcsb_entry_info_polymer_entity_count_dna","rcsb_entry_info_polymer_entity_count_rna","rcsb_entry_info_polymer_entity_count_nucleic_acid","rcsb_entry_info_polymer_entity_count_protein","rcsb_entry_info_polymer_entity_taxonomy_count","rcsb_entry_info_polymer_molecular_weight_maximum","rcsb_entry_info_polymer_molecular_weight_minimum","rcsb_entry_info_polymer_monomer_count_maximum","rcsb_entry_info_polymer_monomer_count_minimum","rcsb_entry_info_solvent_entity_count", "pdbx_nmr_exptl_sample_conditions_p_h","pdbx_nmr_exptl_sample_conditions_temperature"

    ]  + (general_columns() if include_general else []))
    
def MM_columns(include_general=True):
    return reduce_value_length_version2([
        "rcsb_entry_info_assembly_count","rcsb_entry_info_branched_entity_count","rcsb_entry_info_cis_peptide_count","rcsb_entry_info_deposited_atom_count","rcsb_entry_info_deposited_hydrogen_atom_count","rcsb_entry_info_deposited_model_count","rcsb_entry_info_deposited_modeled_polymer_monomer_count","rcsb_entry_info_deposited_nonpolymer_entity_instance_count","rcsb_entry_info_deposited_polymer_entity_instance_count","rcsb_entry_info_deposited_polymer_monomer_count","rcsb_entry_info_deposited_solvent_atom_count","rcsb_entry_info_deposited_unmodeled_polymer_monomer_count","rcsb_entry_info_disulfide_bond_count","rcsb_entry_info_entity_count","rcsb_entry_info_inter_mol_covalent_bond_count","rcsb_entry_info_inter_mol_metalic_bond_count","rcsb_entry_info_molecular_weight","rcsb_entry_info_nonpolymer_entity_count","rcsb_entry_info_polymer_entity_count","rcsb_entry_info_polymer_entity_count_dna","rcsb_entry_info_polymer_entity_count_rna","rcsb_entry_info_polymer_entity_count_nucleic_acid","rcsb_entry_info_polymer_entity_count_protein","rcsb_entry_info_polymer_entity_taxonomy_count","rcsb_entry_info_polymer_molecular_weight_maximum","rcsb_entry_info_polymer_molecular_weight_minimum","rcsb_entry_info_polymer_monomer_count_maximum","rcsb_entry_info_polymer_monomer_count_minimum","rcsb_entry_info_solvent_entity_count","pdbx_nmr_ensemble_conformers_calculated_total_number","pdbx_nmr_ensemble_conformers_submitted_total_number"

    ] + (general_columns() if include_general else []))
   
def columns_to_retrieve():
    membrane_proteins = reduce_value_length_version2([
        'group', 'species', 'resolution', 'name', 'pdb_code',
        'expressed_in_species', 'subgroup', 'rcsentinfo_resolution_combined', 
        'rcsentinfo_experimental_method', 'bibliography_year',
        'taxonomic_domain', 'refine_hist_pdbx_number_atoms_protein', 
        'rcsb_entry_info_diffrn_radiation_wavelength_minimum', 
        'rcsb_entry_info_nonpolymer_molecular_weight_minimum', 
        'em_imaging_accelerating_voltage', 'rcsb_entry_info_polymer_entity_count_protein', 
        'rcsb_entry_info_diffrn_resolution_high_value', 'reflns_d_resolution_high', 
        'refine_hist_d_res_low', 'rcsb_entry_info_cis_peptide_count', 
        'rcsb_entry_info_deposited_solvent_atom_count', 'refine_ls_rfactor_rwork', 
        'pdbx_nmr_ensemble_conformers_calculated_total_number', 
        'cell_angle_gamma', 'cell_length_b', 'refine_ls_number_reflns_rfree', 
        'rcsb_entry_info_polymer_monomer_count_minimum', 
        'cell_length_c', 'exptl_crystals_number', 'rcsb_entry_info_entity_count', 
        'refine_ls_dres_high', 'rcsb_entry_info_nonpolymer_entity_count', 
        'rcsb_entry_info_polymer_entity_taxonomy_count', 
        'rcsb_entry_info_assembly_count', 'rcsb_entry_info_molecular_weight', 
        'exptl_crystal_density_percent_sol', 'refine_ls_rfactor_obs', 
        'rcsb_entry_info_deposited_polymer_entity_instance_count', 
        'refine_pdbx_solvent_shrinkage_radii',  
        'rcsb_entry_info_nonpolymer_molecular_weight_maximum', 
        'rcsb_entry_info_polymer_molecular_weight_maximum', 
        'rcsb_entry_info_polymer_molecular_weight_minimum', 
        'rcsb_entry_info_deposited_unmodeled_polymer_monomer_count', 
        'refine_ls_dres_low', 'reflns_d_resolution_low', 
        'rcsb_entry_info_inter_mol_metalic_bond_count', 
        'pdbx_nmr_ensemble_conformers_submitted_total_number', 
        'refine_overall_suml', 'diffrn_ambient_temp', 'reflns_shell_d_res_high', 
        'refine_hist_pdbx_number_atoms_ligand', 'pdbx_nmr_exptl_sample_conditions_temperature', 
        'rcsb_entry_info_polymer_monomer_count_maximum', 
        'exptl_crystal_grow_p_h', 'reflns_pdbx_ordinal', 
        'refine_ls_number_reflns_obs', 'refine_hist_number_atoms_total', 
        'rcsb_entry_info_deposited_atom_count', 'cell_angle_beta', 
        'exptl_crystal_density_matthews', 'reflns_pdbx_net_iover_sigma_i', 
        'refine_hist_number_atoms_solvent', 'refine_ls_percent_reflns_obs', 
        'rcsb_entry_info_deposited_modeled_polymer_monomer_count', 
        'rcsb_entry_info_polymer_entity_count_rna', 
        'rcsb_entry_info_polymer_entity_count_nucleic_acid', 
        'exptl_crystal_grow_temp', 'rcsb_entry_info_polymer_entity_count', 
        'em_image_recording_avg_electron_dose_per_image', 'cell_zpdb', 
        'rcsb_entry_info_polymer_entity_count_dna', 
        'rcsb_entry_info_solvent_entity_count', 'reflns_number_obs', 
        'rcsb_entry_info_branched_entity_count', 'rcsb_entry_info_diffrn_radiation_wavelength_maximum', 
        'cell_angle_alpha', 'refine_biso_mean', 'rcsb_entry_info_disulfide_bond_count', 
        'rcsb_entry_info_inter_mol_covalent_bond_count', 'refine_pdbx_solvent_vdw_probe_radii', 
        'reflns_pdbx_redundancy', 'refine_pdbx_ls_sigma_f', 
        'rcsb_entry_info_deposited_hydrogen_atom_count', 
        'refine_ls_percent_reflns_rfree', 'refine_ls_rfactor_rfree', 
        'rcsb_entry_info_deposited_polymer_monomer_count', 'reflns_shell_d_res_low', 
        'reflns_percent_possible_obs', 'cell_length_a', 
        'rcsb_entry_info_deposited_nonpolymer_entity_instance_count', 
        'em3d_reconstruction_resolution', 'refine_hist_pdbx_number_atoms_nucleic_acid', 
        'refine_hist_d_res_high', 'rcsb_entry_info_deposited_model_count', 
        'pdbx_nmr_exptl_sample_conditions_p_h', 'em3d_reconstruction_num_particles', 'uniprot_id'
    ])
    return membrane_proteins