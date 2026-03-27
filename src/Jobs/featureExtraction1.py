import os, sys
sys.path.append(os.getcwd())
import requests
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from io import StringIO
import numpy as np
from app import app
import pandas as pd
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

# Function to compute the radius of gyration of a protein structure
def compute_radius_of_gyration(structure):
    coordinates = np.array([atom.get_coord() for atom in structure.get_atoms()])
    center_of_mass = np.mean(coordinates, axis=0)
    distances_sq = np.sum((coordinates - center_of_mass) ** 2, axis=1)
    radius_of_gyration = np.sqrt(np.mean(distances_sq))
    return radius_of_gyration

# Function to compute the diameter of a protein structure
def compute_diameter(structure):
    coordinates = np.array([atom.get_coord() for atom in structure.get_atoms()])
    differences = coordinates[:, None, :] - coordinates
    distances = np.sqrt(np.sum(differences ** 2, axis=-1))
    return np.max(distances)

# Function to compute the volume of a protein structure
def compute_volume(structure):
    return len(list(structure.get_atoms()))

# Function to compute the hydrophobicity of a protein structure
def compute_hydrophobicity(structure):
    kd_hydrophobicity = {
        'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
        'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
        'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
        'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
    }
    total_hydrophobicity = 0
    num_residues = 0
    for residue in structure.get_residues():
        resname = residue.get_resname()
        if resname in kd_hydrophobicity:
            total_hydrophobicity += kd_hydrophobicity[resname]
            num_residues += 1
    return total_hydrophobicity / num_residues if num_residues > 0 else None

# Function to compute the polarity of a protein structure
def compute_polarity(structure):
    amino_acid_polarity = {
        "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
        "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
        "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
        "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2
    }
    total_polarity = 0.0
    for residue in structure.get_residues():
        resname = residue.get_resname()
        if resname in amino_acid_polarity:
            total_polarity += amino_acid_polarity[resname]
    return total_polarity

# Function to compute the number of salt bridges in a protein structure
def compute_salt_bridges(structure):
    charged_residues = {"LYS", "ARG", "HIS", "ASP", "GLU"}
    salt_bridges_count = 0
    for residue1 in structure.get_residues():
        if residue1.get_resname() in charged_residues:
            for residue2 in structure.get_residues():
                if residue2.get_resname() in charged_residues:
                    distance = residue1["CA"] - residue2["CA"]
                    if 4 <= distance <= 7:
                        if (residue1.get_resname() in {"LYS", "ARG"} and residue2.get_resname() in {"ASP", "GLU"}) or \
                           (residue1.get_resname() in {"ASP", "GLU"} and residue2.get_resname() in {"LYS", "ARG"}):
                            salt_bridges_count += 1
    return salt_bridges_count


# Function to extract features from a protein structure
def extract_features(pdb_code):
    pdb_data = fetch_pdb_data(pdb_code)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_code, StringIO(pdb_data))
    atoms = [atom.get_coord() for atom in structure.get_atoms()]
    centroid = np.mean(atoms, axis=0)
    distances = [np.linalg.norm(atom1 - atom2) for atom1 in atoms for atom2 in atoms]
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    avg_distance = np.mean(distances)
    radius_of_gyration = compute_radius_of_gyration(structure)
    diameter = compute_diameter(structure)
    volume = compute_volume(structure)
    hydrophobicity = compute_hydrophobicity(structure)
    polarity = compute_polarity(structure)
    salt_bridges = compute_salt_bridges(structure)
    features = {
        "pdb_code": pdb_code,
        "atom_centroid_x": centroid[0],
        "atom_centroid_y": centroid[1],
        "atom_centroid_z": centroid[2],
        "min_distance": min_distance,
        "max_distance": max_distance,
        "avg_distance": avg_distance,
        "radius_of_gyration": radius_of_gyration,
        "diameter": diameter,
        "volume": volume,
        "hydrophobicity": hydrophobicity,
        "polarity": polarity,
        "salt_bridges": salt_bridges,
    }
    return features

# Example usage with multiple PDB codes and creating a DataFrame
pdb_codes = all_data["pdb_code"].to_list()  
data = []
from tqdm import tqdm 
for pdb_code in tqdm(pdb_codes, desc="Processing proteins", unit="protein"):
    try:
        features = extract_features(pdb_code)
        features["pdb_code"] = pdb_code
        data.append(features)
    except Exception as e:
        print(f"Failed to process PDB ID {pdb_code}: {e}")

df = pd.DataFrame(data)
# Display the DataFrame
df.to_csv("protein_structural_data2.csv")
