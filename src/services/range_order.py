
from src.Commands.Migration.classMigrate import Migration


columns_range_limit = {
    'Resolution': 2,
    'processed_resolution': 2,
    'rcsb_entry_info_deposited_atom_count': 5000,
    'refine_hist_pdbx_number_residues_total': 100,
    "rcsb_entry_info_molecular_weight": 500,
    'citation_year': 5,
}

def reduce_key_length_in_dict(data_dict):
    new_dict = {}
    for key, value in data_dict.items():
        new_key = Migration.shorten_column_name(key) if isinstance(key, str)  else key
        new_dict[new_key] = value
    return new_dict


columns_range_limit = reduce_key_length_in_dict(columns_range_limit)