import os
import pandas as pd
from Bio import PDB
from Bio.PDB import PDBList, PDBParser

def download_pdb_sequences(pdb_ids, file_format="pdb", output_dir="PDBs/"):
    sequences = {}

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdb_list = PDBList()

    for pdb_id in pdb_ids:
        try:
            # Download the PDB file
            pdb_filename = pdb_list.retrieve_pdb_file(pdb_id, file_format=file_format, pdir=output_dir)

            # Check if the downloaded file exists
            if os.path.exists(pdb_filename):
                # Manually rename the file to have the desired extension
                new_filename = os.path.join(output_dir, f"{pdb_id}.{file_format}")
                os.rename(pdb_filename, new_filename)

                # Parse the PDB file
                pdb_parser = PDBParser(QUIET=True)
                structure = pdb_parser.get_structure(pdb_id, new_filename)

                # Extract the sequence
                sequence = ""
                for model in structure:
                    for chain in model:
                        sequence += "".join(residue.get_resname() for residue in chain.get_residues())

                sequences[pdb_id] = sequence
            else:
                print(f"File not found for PDB {pdb_id}")
                continue
        except Exception as e:
            print(f"Error processing PDB {pdb_id}: {e}")
            continue

    return sequences


def update_csv_with_sequences(input_csv, output_csv, pdb_ids_column="Pdb Code"):
    try:
        # Read the existing CSV file
        df = pd.read_csv(input_csv)

        # Download sequences
        pdb_ids = df[pdb_ids_column].tolist()
        sequences = download_pdb_sequences(pdb_ids)

        # Update the DataFrame with sequences
        df["Sequence"] = df[pdb_ids_column].map(sequences)

        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_csv, index=False)
    except Exception as e:
        print(f"Error updating CSV: {e}")
        
    
    
def create_dir(directory_path):
    import os
    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")
       


def convert_ent_to_pdb(ent_file, pdb_file):
    # Create a parser for the .ent file
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", ent_file)

    # Write the structure to a PDB file
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(pdb_file) 
        
        
def getDictFromDirectory(directory_path):
    # List all files in the directory
    file_names = os.listdir(directory_path)
    
    # Create a dictionary with file names as keys and full paths as values
    file_paths_dict = {file_name: os.path.join(directory_path, file_name) for file_name in file_names}
    return file_paths_dict