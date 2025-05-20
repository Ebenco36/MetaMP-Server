import datetime
import pandas as pd
import os, re, ast, sys
sys.path.append(os.getcwd())

# from src.services.Helpers.helper import add_uniprot_id, get_published_date

def does_file_exist(file_path:str):
    return os.path.exists(file_path)


def extract_year(bibliography:str):
    match = re.search(r"\['year', '(\d{4})'\]", bibliography)
    if match:
        return match.group(1)
    else:
        return None
    
# Function to check if a string of list dictionaries is not empty
def preprocess_str_data(str_value:str):
    try:
        # Parse the string into a list of dictionaries using ast.literal_eval
        value_list = ast.literal_eval(str_value)
        if(isinstance(value_list, list) and len(value_list) > 1):
            # then take the first on the list 
            new_str = ast.literal_eval([value_list[0]])
            return new_str
        else:
            return ast.literal_eval(str_value.strip('[]'))
    except (SyntaxError, ValueError):
        return {}

# Function to remove all HTML tags from a string
def remove_html_tags(text:str):
    try:
        if not text is None and pd.notna(text):
            clean_text = re.sub(r'<.*?>', '', text)
            # Replace '\r' and '\n' with a space
            clean_text = clean_text.replace('\r', ' ').replace('\n', ' ')
            return clean_text
        else:
            return ''
    
    except (Exception, TypeError) as e:
        print(str(e))


"""
Date columns from the sheet.
"""
dates_columns = [
    'rcsb_accession_info.deposit_date', 
    'rcsb_accession_info.initial_release_date', 
    'rcsb_accession_info.revision_date'
]

column_year = [
    'rcsb_primary_citation.year'
]

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
    'pdbx_serial_crystallography_measurement',
    'pdbx_serial_crystallography_data_reduction', 
    'pdbx_serial_crystallography_sample_delivery',
    'pdbx_serial_crystallography_sample_delivery_injection', 
    'pdbx_serial_crystallography_sample_delivery_fixed_target', 
]

all_descriptors = descriptors + em_descriptors + rcsb_descriptors + struct_descriptors + pdbx_descriptors + pdbx_nmr_descriptors\
    + pdbx_serial_descriptors + pdbx_database_descriptors
    
"""
Quantitative but came out as an array
"""

quantitative_array_column = [
    # 'rcsb_entry_info.resolution_combined'
]
"""
Quantitative data
"""

cell_columns = [
    'cell.zpdb',
    'cell.length_a', 
    'cell.length_b', 
    'cell.length_c', 
    'cell.angle_beta', 
    'cell.angle_alpha', 
    'cell.angle_gamma', 
]

rcsb_entries = [  
    'rcsb_entry_info.entity_count',
    'rcsb_entry_info.assembly_count',  
    'rcsb_entry_info.molecular_weight',  
    'rcsb_entry_info.cis_peptide_count',  
    'rcsb_entry_info.polymer_entity_count', 
    'rcsb_entry_info.disulfide_bond_count',  
    'rcsb_entry_info.solvent_entity_count', 
    'rcsb_entry_info.branched_entity_count', 
    'rcsb_entry_info.deposited_model_count',
    'rcsb_entry_info.nonpolymer_entity_count',
    'rcsb_entry_info.deposited_solvent_atom_count',
    'rcsb_entry_info.polymer_entity_count_protein', 
    'rcsb_entry_info.inter_mol_metalic_bond_count', 
    'rcsb_entry_info.inter_mol_covalent_bond_count', 
    'rcsb_entry_info.polymer_entity_taxonomy_count',
    'rcsb_entry_info.polymer_monomer_count_maximum', 
    'rcsb_entry_info.polymer_monomer_count_minimum',  
    'rcsb_entry_info.deposited_hydrogen_atom_count', 
    'rcsb_entry_info.deposited_polymer_monomer_count', 
    'rcsb_entry_info.polymer_molecular_weight_maximum', 
    'rcsb_entry_info.polymer_molecular_weight_minimum', 
    'rcsb_entry_info.branched_molecular_weight_maximum', 
    'rcsb_entry_info.branched_molecular_weight_minimum', 
    'rcsb_entry_info.diffrn_radiation_wavelength_maximum', 
    'rcsb_entry_info.diffrn_radiation_wavelength_minimum',
    'rcsb_entry_info.nonpolymer_molecular_weight_maximum', 
    'rcsb_entry_info.nonpolymer_molecular_weight_minimum',
    'rcsb_entry_info.deposited_modeled_polymer_monomer_count',
    'rcsb_entry_info.deposited_polymer_entity_instance_count', 
    'rcsb_entry_info.deposited_unmodeled_polymer_monomer_count',
    'rcsb_entry_info.deposited_nonpolymer_entity_instance_count', 
    'rcsb_entry_info.structure_determination_methodology_priority', 
]


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
    'exptl_method', 'exptl_crystal_grow_method'
]

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

# Define a dictionary for spelling corrections before other things
spelling_corrections = {
    'Methanocaldococcus jannaschi': 'Methanocaldococcus jannaschii',
    'Rhodopeudomonas blastica': 'Rhodopseudomonas blastica', 
    'Shewanella oneidensi': 'Shewanella oneidensis',
    'Synechocystis sp. pcc 6803': 'Synechocystis sp. PCC6803',
    'E. Coli': 'E. Coli',
    'E. Colli': 'E. Coli',
    'E. colli': 'E. Coli',
    'E.coli': 'E. Coli',
    'E. coli ': 'E. Coli',
    'E. coli': 'E. Coli',
    'e. Coli': 'E. Coli',
    'Escherichia coli': 'E. Coli',
    'e. coli': 'E. Coli',
    'Escherichia coli B': 'E. Coli',
    'E. coli-based cell-free expression system': 'E. Coli',
    'E coli': 'E. Coli',
    'E. coli + semi-synthesis': 'E. Coli',
    'HEK 293S cells': 'HEK-293S cells',
    'HEK-293S cells': 'HEK-293S cells',
    'HEK293S cells': 'HEK-293S cells',
    'HEK293s cells': 'HEK-293S cells',
    'HEK293S': 'HEK-293S cells',
    'HEK 293 cells': 'HEK293 cells',
    'HEK293': 'HEK293 cells',
    'HEK-293 cells': 'HEK293 cells',
    'HEK 293': 'HEK293 cells',
    'expressed in HEK293 cells': 'HEK293 cells',
    'HEK293S GnTI- cells': 'HEK293S GnTI- cells',
    'HEK293S GnTI-': 'HEK293S GnTI- cells',
    'HEK2935-GnTl- cells': 'HEK293S GnTI- cells',
    'HEK-293S-GnTI-': 'HEK293S GnTI- cells',
    'HEK293F cells': 'HEK293F cells',
    'HEK293F': 'HEK293F cells',
    'HEK 293F cells': 'HEK293F cells',
    'HEK293F': 'HEK293F cells',
    'S. Crevisiae': 'S. Cerevisiae',
    'S. crevisiae': 'S. Cerevisiae',
    'S. frugiperda': 'S. Frugiperda',
    'S. frugiperda cells': 'S. Frugiperda',
    'S. frugiperda & Trichoplusia ni': 'S. frugiperda & Trichoplusia ni',
    'S. frugiperda &  Trichoplusia ni': 'S. frugiperda & Trichoplusia ni',
    'Sf9 cells': 'SF9 cells',
    'sf9 cells': 'SF9 cells',
    'SF9 cells': 'SF9 cells',
    'sf-9 cells': 'SF9 cells',
    'Trichoplusia ni': 'Trichoplusia ni',
    'Trichoplusia ni': 'Trichoplusia ni',
    'Trichoplusia ni)': 'Trichoplusia ni',
    'NMR Structure': 'NMR Structure',
    'NMR structure': 'NMR Structure',
    'NMR strucuture': 'NMR Structure',
    'NMR Structure (DPC micelles)': 'NMR Structure',
    'NMR/Molecular Dynamics Model': 'NMR Structure',
    'solid-state NMR structure': 'NMR Structure',
    'NMR (DHPC micelles)': 'NMR Structure',
    'NMR (DPC micelles, with H-bond constraints)': 'NMR Structure',
    '1.77 Å': '1.77',
    '3.4/3.7&nbsp;&Aring;': '3.4-3.7',
    '1.6&nbsp;&Aring; (native structure)': '1.6',
    '1.8&nbsp;&Aring; (P63 crystal form)': '1.8',
    'Synthesized': 'Synthesized',
    'synthesized': 'Synthesized',
    'NMR': 'NMR Structure',
    'Homo Sapiens': 'Homo Sapiens',
    'Homo sapiens': 'Homo Sapiens', 
    'homo sapiens': 'Homo Sapiens',
    'Homo sapien': 'Homo Sapiens',
    'Homo spiens': 'Homo Sapiens',
    'Hom sapiens': 'Homo Sapiens', 
    'Homo sapiens': 'Homo Sapiens',
    'Homo sapien': 'Homo Sapiens',
    'Homo Sapiens': 'Homo Sapiens',
    'Bacillus anthraciss': 'Bacillus anthracis',
    'Bacillus sp. PS3': 'Bacillus sp. (strain PS3)', 
    'Bascillus subtilis': 'Bacillus subtilis',
    'De novo designed protein': 'De novo designed',
    'Escherichia coi': 'E. Coli',
    'Escherichia coli (cell-free expression)': 'E. Coli',
}

correct_spelling_exptl_crystal = {
    "VAPOUR DIFFUSION": "VAPOR DIFFUSION",
    "LCP SANDWICH PLATES": "LIPIDIC CUBIC PHASE", 
    "LIPDIC CUBIC PHASE": "LIPIDIC CUBIC PHASE", 
    "LIPIC CUBIC PHASE": "LIPIDIC CUBIC PHASE", 
    "LIPID CUBIC PHASE": "LIPIDIC CUBIC PHASE", 
    "LIPID CUBIC PHASE (LCP)": "LIPIDIC CUBIC PHASE", 
    "LIPID MESOPHASE": "LIPIDIC CUBIC PHASE", 
    "LIPIDIC CUBIC PHASE (LCP)": "LIPIDIC CUBIC PHASE", 
    "LIPIDIC CUBIC PHASE MONOOLEIN": "LIPIDIC CUBIC PHASE", 
    "LIPIDIC MESOPHASE": "LIPIDIC CUBIC PHASE", 
    "LIPIDIC QUBIC PHASE": "LIPIDIC CUBIC PHASE", 
    "LIPIDIC SUBIC PHASE": "LIPIDIC CUBIC PHASE", 
    "LUPUC CUBIC PHASE": "LIPIDIC CUBIC PHASE", 
    "CUBIC LIPID PHASE": "LIPIDIC CUBIC PHASE", 
    "CUBIC PHASE CRYSTALLIZATION": "LIPIDIC CUBIC PHASE", 
    "LCP": "LIPIDIC CUBIC PHASE",
    "BATCH": "BATCH MODE", 
    "BATCH METHOD": "BATCH MODE",
    "DIALYSIS WITH CONTINUOUS FLOW DIALYSIS MACHINE": "DIALYSIS", 
    "DIALYSYS": "DIALYSIS", 
    "MICRODIALYSIS": "DIALYSIS",
    "MICRO-BATCH METHOD UNDER OIL": "MICROBATCH", 
    "DIALYSYS": "MICROBATCH", 
    "MICRODIALYSIS": "MICROBATCH", 
    "MICROBATCH (PARAFFIN OIL)": "MICROBATCH", 
    "MICROBATCH TECHNIQUE UNDER OIL": "MICROBATCH", 
    "MICROBATCH-UNDER-OIL": "MICROBATCH", 
    "OIL-MICROBATCH METHOD": "MICROBATCH"
}

modified_path = "."
current_date = datetime.date.today().strftime('%Y-%m-%d')
#Get the two prepared tables

import glob
# 1. List all matching files
pattern = modified_path + "/datasets/PDB_data_*.csv"
files = glob.glob(pattern)

# 2. Extract dates and find the latest
date_file_map = {}
for f in files:
    m = re.search(r"PDB_data_(\d{4}-\d{2}-\d{2})\.csv$", f)
    if m:
        dt = datetime.datetime.strptime(m.group(1), "%Y-%m-%d").date()
        date_file_map[dt] = f

if not date_file_map:
    raise FileNotFoundError(f"No files matching {pattern}")

latest_date = max(date_file_map)
latest_file = date_file_map[latest_date]


PDB = pd.read_csv(latest_file, low_memory=False)
Mpstruc = pd.read_csv(modified_path + "/datasets/Mpstruct_dataset.csv")

#merge them
merged_db = pd.merge(Mpstruc, PDB, on = "Pdb Code")
merged_db.to_csv(modified_path + "/datasets/enriched_db.csv", index=False)


pd.options.mode.chained_assignment = None  # default='warn' 

class DataImport:
    def __init__(self, needed_columns:list = []) -> None:
        # setting class properties here.
        self.needed_columns = needed_columns if (len(needed_columns) > 0) else all_descriptors\
        + dates_columns + cell_columns + rcsb_entries + quantitative_array_column

    def dataPrePreprocessing(self, df):
        df[cat_list] = df[cat_list].apply(lambda x: x.str.strip())
        
        df['exptl_crystal_grow_method'] = df['exptl_crystal_grow_method'].astype(str).str.upper()
        # Apply spelling corrections to the specified columns
        df.replace({
            'Expressed in Species': spelling_corrections, 
            'Species': spelling_corrections, 
            'Resolution': spelling_corrections,
            'exptl_crystal_grow_method': correct_spelling_exptl_crystal,
            }, 
            inplace=True
        )
        # process exptl_crystal_grow_method for easy usage
        df[['exptl_crystal_grow_method1', 'exptl_crystal_grow_method2', 'not_useful']] = df['exptl_crystal_grow_method'].str.split(',', expand=True)
        
        # reorder content for example 
        def reorder_methods(row):
            # Ensure both methods are not None, then check conditions
            if pd.notna(row['exptl_crystal_grow_method2']) and row['exptl_crystal_grow_method2'] in ["VAPOR DIFFUSION", "VAPOUR DIFFUSION"] and  row['exptl_crystal_grow_method1'] not in ["VAPOR DIFFUSION", "VAPOUR DIFFUSION"]:
                # Swap the methods
                row['exptl_crystal_grow_method1'], row['exptl_crystal_grow_method2'] = row['exptl_crystal_grow_method2'], row['exptl_crystal_grow_method1']
            return row  
        # Apply the reordering function
        df = df.apply(reorder_methods, axis=1)
        
        df['processed_resolution'] = df['rcsb_entry_info_resolution_combined'].str.extract(r'\[(\d+\.\d+)\]', expand=False).astype(float)
        # Apply transformations to DataFrame columns by removing spaces and converting to a lower case
        # df[cat_list] = df[cat_list].apply(lambda x: x.str.lower().str.replace(' ', '_'))
        # Get unique values in the specified column
        return df
    
    
    def loadFile(self):
        check_quant_file = does_file_exist(modified_path + "/datasets/Quantitative_data.csv")
        if(not check_quant_file):
            current_date = datetime.date.today().strftime('%Y-%m-%d')
            file_path = modified_path + '/datasets/enriched_db.csv'
            data = pd.read_csv(file_path, low_memory=False)

            # data vis
            print("Number of total Pdb Entries:", len(set(data["Pdb Code"])))
            print("Number of Subgroups:", len(set(data["Subgroup"])))
            print("Number of different species:", len(set(data["Species"])))

            # Filter out columns with string data type for the removal of special characters
            transform_data = data.select_dtypes(include='object')

            data[transform_data.columns] = transform_data[transform_data.columns].applymap(remove_html_tags)

            # data  = remove_bad_columns(data)

            # Apply the conversion function to each column and append parent column name
            normalized_data = []
            for one_column in data.columns:
                col_data  = data[one_column].apply(lambda x: preprocess_str_data(x))
                try:
                    normalized_col = pd.json_normalize(col_data, sep="_")
                except (AttributeError):
                    print(one_column)
                if not normalized_col.empty:
                    col = one_column
                    normalized_col.columns = [f"{col}_{col_name}" for col_name in normalized_col.columns]
                    normalized_data.append(normalized_col)

            # Merge the normalized data with the original DataFrame
            merged_df_ = pd.concat([data] + normalized_data, axis=1)
            print("Number of total Pdb Entries:", len(set(merged_df_["Pdb Code"])))
            merged_df_.index = merged_df_[['Pdb Code']]
            # extract bibiography column
            merged_df = merged_df_.copy()
            merged_df['bibliography_year'] = merged_df['Bibliography'].apply(extract_year)
            # merged_df['published_date'] = merged_df['rcsb_primary_citation_pdbx_database_id_doi'].apply(get_published_date)
            # merged_df['published_date'] = pd.to_datetime(merged_df['published_date'])
            # merged_df['published_year'] = merged_df['rcsb_accession_info_deposit_date'].dt.year
            merged_df.loc[merged_df['Group'] == 'TRANSMEMBRANE PROTEINS: BETA-BARREL', 'Group'] = 'TRANSMEMBRANE PROTEINS:BETA-BARREL'
            merged_df.loc[merged_df['Group'] == 'TRANSMEMBRANE PROTEINS: ALPHA-HELICAL', 'Group'] = 'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL'
            
            # merged_df['uniprot_id'] = merged_df['Pdb Code'].apply(lambda x: add_uniprot_id(x))
            
            # Replace dots with underscores in column names
            merged_df.columns = merged_df.columns.str.replace('.', '_')
            merged_df = self.dataPrePreprocessing(merged_df)
            merged_df.to_csv(modified_path + '/datasets/Quantitative_data.csv', index=False)
        else:
            merged_df = pd.read_csv(modified_path + "/datasets/Quantitative_data.csv", low_memory=False)
            merged_df.loc[merged_df['Group'] == 'TRANSMEMBRANE PROTEINS: BETA-BARREL', 'Group'] = 'TRANSMEMBRANE PROTEINS:BETA-BARREL'
            merged_df.loc[merged_df['Group'] == 'TRANSMEMBRANE PROTEINS: ALPHA-HELICAL', 'Group'] = 'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL'
            
            # merged_df['uniprot_id'] = merged_df['Pdb Code'].apply(lambda x: add_uniprot_id(x))
            
            #merged_df.loc[merged_df['rcsb_entry_info_experimental_method'] == 'Multiple methods', 'rcsb_entry_info_experimental_method'] = 'Multi-Method'
            merged_df = self.dataPrePreprocessing(merged_df)
            print("Number of total Pdb Entries:", len(set(merged_df["Pdb Code"])))
            merged_df.to_csv(modified_path + '/datasets/Quantitative_data.csv', index=False)
        return merged_df
    

data_merge = DataImport()
data_merge.loadFile()