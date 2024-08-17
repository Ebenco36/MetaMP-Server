import os
import requests
import pandas as pd

data_path = os.environ.get("AIRFLOW_HOME")
if (data_path):
    modified_path = data_path.replace("/airflow_home", "")
else:
    modified_path = "."

class NEWOPM:
    def __init__(self):
        self.headers = {
            "authority": "opm-back.cc.lehigh.edu:3000",
            "method": "GET",
            "scheme": "https",
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Origin": "https://opm.phar.umich.edu",
            "Referer": "https://opm.phar.umich.edu/",
            "Sec-Ch-Ua": '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "cross-site",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        }
        self.host = "https://opm-back.cc.lehigh.edu/opm-backend/"
 
    def fetch(self):
        # Load PDB codes from a CSV file
        csv_file_path = modified_path + "/datasets/newOPMData/opm_data_prediction_data.csv"
        df = pd.read_csv(csv_file_path, low_memory=False)
        df = df.head(1000)  # Limit to 2 rows for testing
        
        all_dfs = []  # List to store dataframes for each PDB code
        
        file_path = modified_path + "/datasets/newOPMData/FullNEWOPM.csv"
        
        for pdb_code in df['pdbid']:
            # Check if the PDB code already exists in the output CSV file
            if os.path.isfile(file_path):
                existing_df = pd.read_csv(file_path)
                if pdb_code in existing_df['pdbid'].values:
                    print(f"PDB code {pdb_code} already exists. Skipping...")
                    continue
            
            # Fetch record for each PDB code
            url = self.host + "/primary_structures?search=" + pdb_code + "&sort=&pageSize=100"
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                record = data["objects"]
                if record and len(record) > 0:
                    filter_data = record[0]
                    # Fetch all related information for the protein entry
                    url2 = self.host + "/primary_structures/" + str(filter_data.get("id"))
                    response_filtered = requests.get(url2, headers=self.headers)

                    if response_filtered.status_code == 200:
                        data_filtered = response_filtered.json()
                        # Convert data to DataFrame
                        df_filtered = pd.json_normalize(data_filtered, sep="_")
                        df_filtered['pdb_code'] = pdb_code  # Add a new column for PDB code
                        all_dfs.append(df_filtered)  # Append dataframe to list

        # Concatenate all dataframes into a single dataframe
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combine_data = combined_df[[
                "id","ordering","pdbid","name","resolution",
                "topology_subunit","topology_show_in","thickness","thicknesserror",
                "subunit_segments","tilt","tilterror","gibbs","tau",
                "family_name_cache","species_name_cache","membrane_name_cache","membrane_id",
                "species_id","family_id","superfamily_id","classtype_id","type_id","uniprotcodes",
                "family_name","family_pfam","family_interpro","family_tcdb","family_superfamily_id",
                "family_superfamily_name","family_superfamily_pfam","family_superfamily_tcdb",
                "family_superfamily_classtype_id","family_superfamily_classtype_name",
                "family_superfamily_classtype_type_id","family_superfamily_classtype_type_name",
                "species_name","membrane_name","membrane_short_name","membrane_abbrevation",
                "membrane_topology_in","membrane_topology_out",
                "membrane_lipid_pubmed","pdb_code"
            ]]
            # Save the combined dataframe to a CSV file
            combine_data.to_csv(file_path, index=False)
            print(f"All data saved to {file_path}")
        else:
            print("No new data to save.")
            
        print("we need to expand the columns...")

# Instantiate the class and call the function
opm = NEWOPM()
opm.fetch()
