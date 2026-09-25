import os, sys
from Bio.PDB import PDBParser
sys.path.append(os.getcwd())
import requests
from io import StringIO
import numpy as np
import pandas as pd
from tqdm import tqdm  # Optional: For progress tracking
from app import app
from database.db import db
from src.Dashboard.services import get_tables_as_dataframe, get_table_as_dataframe
table_names = ['membrane_proteins', 'membrane_protein_opm']
with app.app_context():
    result_df = get_tables_as_dataframe(table_names, "pdb_code")
    result_df_db = get_table_as_dataframe("membrane_proteins")
    result_df_opm = get_table_as_dataframe("membrane_protein_opm")
    result_df_uniprot = get_table_as_dataframe("membrane_protein_uniprot")
    
all_data = pd.merge(right=result_df, left=result_df_uniprot, on="pdb_code")


# Function to fetch PDB data from RCSB PDB
def fetch_pdb_data(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch PDB data for {pdb_id}")

# Function to extract features from a structure
def extract_features(pdb_id, structure):
    features = {}
    num_chains = 0
    num_residues = 0
    num_atoms = 0
    b_factors = []
    coords = []
    sequence_length = []

    for model in structure:
        for chain in model:
            num_chains += 1
            chain_length = 0
            for residue in chain:
                num_residues += 1
                chain_length += 1
                for atom in residue:
                    num_atoms += 1
                    b_factors.append(atom.bfactor)
                    coords.append(atom.coord)
            sequence_length.append(chain_length)

    b_factors = np.array(b_factors)
    coords = np.array(coords)
    sequence_length = np.array(sequence_length)

    features['pdb_id'] = pdb_id
    features['num_chains'] = num_chains
    features['num_residues'] = num_residues
    features['num_atoms'] = num_atoms
    features['avg_b_factor'] = np.mean(b_factors) if b_factors.size > 0 else 0.0
    features['std_b_factor'] = np.std(b_factors) if b_factors.size > 0 else 0.0
    features['median_b_factor'] = np.median(b_factors) if b_factors.size > 0 else 0.0
    features['min_b_factor'] = np.min(b_factors) if b_factors.size > 0 else 0.0
    features['max_b_factor'] = np.max(b_factors) if b_factors.size > 0 else 0.0
    features['range_b_factor'] = features['max_b_factor'] - features['min_b_factor']
    features['percentile_25_b_factor'] = np.percentile(b_factors, 25) if b_factors.size > 0 else 0.0
    features['percentile_75_b_factor'] = np.percentile(b_factors, 75) if b_factors.size > 0 else 0.0
    features['skewness_b_factor'] = pd.Series(b_factors).skew() if b_factors.size > 0 else 0.0
    features['kurtosis_b_factor'] = pd.Series(b_factors).kurtosis() if b_factors.size > 0 else 0.0
    features['centroid_x'] = np.mean(coords[:, 0]) if coords.size > 0 else 0.0
    features['centroid_y'] = np.mean(coords[:, 1]) if coords.size > 0 else 0.0
    features['centroid_z'] = np.mean(coords[:, 2]) if coords.size > 0 else 0.0
    features['max_sequence_length'] = np.max(sequence_length) if sequence_length.size > 0 else 0
    features['min_sequence_length'] = np.min(sequence_length) if sequence_length.size > 0 else 0
    features['mean_sequence_length'] = np.mean(sequence_length) if sequence_length.size > 0 else 0.0
    features['std_sequence_length'] = np.std(sequence_length) if sequence_length.size > 0 else 0.0
    
    return features

# List of PDB IDs for the proteins you want to analyze
pdb_ids = all_data["pdb_code"].to_list()  # Add more PDB IDs as needed

# Initialize an empty list to store the extracted features
all_features = []

# Iterate over each PDB ID and extract features
for pdb_id in tqdm(pdb_ids, desc="Processing proteins", unit="protein"):
    try:
        # Fetch PDB data
        pdb_data = fetch_pdb_data(pdb_id)

        # Parse PDB data
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, StringIO(pdb_data))

        # Extract features
        features = extract_features(pdb_id, structure)

        # Append features to the list
        all_features.append(features)
    except Exception as e:
        print(f"Failed to process PDB ID {pdb_id}: {e}")

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(all_features)

# Display the DataFrame
df.to_csv("protein_structural_data.csv")
