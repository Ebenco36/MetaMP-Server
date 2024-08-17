import os
import requests
import pandas as pd

data_path = os.environ.get("AIRFLOW_HOME")
if data_path:
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
            "Sec-Ch-Ua": '"Not A;Brand";v="99", "Google Chrome";v="91", "Chromium";v="91"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "cross-site",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }
        self.host = "https://opm-back.cc.lehigh.edu/opm-backend/"

    def fetch(self):
        # Load PDB codes from a CSV file
        csv_file_path = modified_path + "/opm_data_prediction_data.csv"
        df = pd.read_csv(csv_file_path, low_memory=False)
        # df = df.head(2)  # Limit to 2 rows for testing
        
        file_path_base = modified_path + "/FullNEWOPM_"  # Base file path for saving batches

        for pdb_code in df['pdbid']:
            all_dfs = []  # List to store dataframes for each PDB code
            total_records_fetched = 0
            page_num = 1

            while True:
                url = self.host + f"/primary_structures?search={pdb_code}&sort=&pageSize=1000&pageNum={page_num}"
                response = requests.get(url, headers=self.headers)

                if response.status_code == 200:
                    data = response.json()
                    records = data["objects"]

                    if not records:
                        break  # No more records to fetch

                    for record in records:
                        url2 = self.host + f"/primary_structures/{record.get('id')}"
                        response_filtered = requests.get(url2, headers=self.headers)

                        if response_filtered.status_code == 200:
                            data_filtered = response_filtered.json()
                            df_filtered = pd.json_normalize(data_filtered, sep="_")
                            df_filtered['pdbid'] = pdb_code  # Add a new column for PDB code
                            all_dfs.append(df_filtered)  # Append dataframe to list
                        else:
                            print(f"Failed to fetch detailed data for record ID {record.get('id')}")
                            continue

                    total_records_fetched += len(records)
                    
                    # Save the batch of data for each page if it has records
                    if len(all_dfs) >= 1000 or not records:
                        combined_df = pd.concat(all_dfs[:1000], ignore_index=True)
                        file_path = file_path_base + f"{pdb_code}_batch_{page_num}.csv"
                        combined_df.to_csv(file_path, index=False)
                        print(f"Saved {len(combined_df)} records for {pdb_code}, batch {page_num}")
                        all_dfs = all_dfs[1000:]  # Remove saved records from the list

                    if len(records) < 1000:
                        break  # Less than 1000 records fetched means no more pages

                    page_num += 1
                else:
                    print(f"Failed to fetch data for PDB code {pdb_code}. Status code: {response.status_code}")
                    break

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
            split_df = existing_df['species_description'].str.split(',', expand=True, n=len(taxonomic_levels) - 1)

            # Rename the columns based on taxonomic hierarchy
            split_df.columns = taxonomic_levels
            # Display the expanded DataFrame
            existing_df = pd.concat([existing_df, split_df], axis=1)
            existing_df.to_csv(modified_path + "/FullNEWOPM.csv")
        else:
            print("Taxonomic level columns already exist in the DataFrame.")

# Instantiate the class and call the function
opm = NEWOPM()
opm.fetch()
