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

    def process_dataframe(self, existing_df, modified_path):
        # Define the columns representing taxonomic levels
        taxonomic_levels = [
            'Domain', 'Kingdom', 'Subkingdom', 'Superphylum', 'Phylum',
            'Subphylum', 'Superclass', 'Class', 'Subclass', 'Infraclass',
            'Superorder', 'Order', 'Suborder', 'Infraorder', 'Family',
            'Subfamily'
        ]

        # Check if any of the taxonomic level columns already exist in the DataFrame
        existing_columns = [col for col in taxonomic_levels if col in existing_df.columns]

        # If none of the taxonomic level columns exist, proceed with expansion
        if not existing_columns:
            # Split the species_description column by commas
            split_df = existing_df['species_description'].str.split(',', expand=True, n=len(taxonomic_levels) -1)

            print(taxonomic_levels)
            print(existing_df['species_description'].str.split(', ', expand=True))
            
            # Rename the columns based on taxonomic hierarchy
            split_df.columns = taxonomic_levels
            # Display the expanded DataFrame
            existing_df = pd.concat([existing_df, split_df], axis=1)
            existing_df.to_csv(modified_path + "/datasets/NEWOPM.csv")
        else:
            print("Taxonomic level columns already exist in the DataFrame.")

 
    def fetch(self):
        # Load PDB codes from a CSV file
        csv_file_path = modified_path + "/datasets/Quantitative_data.csv"
        df = pd.read_csv(csv_file_path, low_memory=False)
        #df = df.head(2)  # Limit to 2 rows for testing
        
        all_dfs = []  # List to store dataframes for each PDB code
        
        file_path = modified_path + "/datasets/NEWOPM.csv"
        
        if os.path.isfile(file_path):
            existing_df = pd.read_csv(file_path)
            self.process_dataframe(existing_df, modified_path)
            print("We are done processing and expanding.")
        
        for pdb_code in df['Pdb Code']:
            # Check if the PDB code already exists in the output CSV file
            if os.path.isfile(file_path):
                existing_df = pd.read_csv(file_path)
                if pdb_code in existing_df['pdb_code'].values:
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
            # Save the combined dataframe to a CSV file
            combined_df.to_csv(file_path, index=False)
            print(f"All data saved to {file_path}")
        else:
            print("No new data to save.")
            
        print("we need to expand the columns...")

# Instantiate the class and call the function
opm = NEWOPM()
opm.fetch()
