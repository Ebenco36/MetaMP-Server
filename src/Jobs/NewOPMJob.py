import os
import requests
import pandas as pd
import datetime
from math import ceil

# Base path for your datasets folder
DATASETS_PATH = "./datasets"
OPM_BATCH_PATH = os.path.join(DATASETS_PATH, "OPM")

class NEWOPM:
    def __init__(self):
        self.headers = {
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://opm.phar.umich.edu",
            "Referer": "https://opm.phar.umich.edu/",
            "User-Agent": "Mozilla/5.0"
        }
        self.host = "https://opm-back.cc.lehigh.edu/opm-backend"

    def create_directories(self):
        """Ensure the necessary directories exist."""
        os.makedirs(DATASETS_PATH, exist_ok=True)
        os.makedirs(OPM_BATCH_PATH, exist_ok=True)

    def process_dataframe(self, df: pd.DataFrame, out_path: str):
        """Expand taxonomy columns on the final combined file."""
        taxonomic_levels = [
            'Domain', 'Kingdom', 'Subkingdom', 'Superphylum', 'Phylum',
            'Subphylum', 'Superclass', 'Class', 'Subclass', 'Infraclass',
            'Superorder', 'Order', 'Suborder', 'Infraorder', 'Family',
            'Subfamily'
        ]
        if not any(col in df.columns for col in taxonomic_levels):
            split_df = (
                df['species_description']
                  .str.split(',', expand=True, n=len(taxonomic_levels)-1)
            )
            split_df.columns = taxonomic_levels
            df = pd.concat([df, split_df], axis=1)
            df.to_csv(out_path, index=False)
            print(f"▶ Expanded taxonomy and saved to {out_path}")
        else:
            print("☑ Taxonomy columns already exist—no expansion needed.")

    def fetch(self):
        """Fetch in batches of 50 into datasets/OPM/, then combine."""
        self.create_directories()

        today     = datetime.date.today().strftime('%Y-%m-%d')
        master_fn = f"NEWOPM_{today}.csv"
        master_fp = os.path.join(DATASETS_PATH, master_fn)

        # Load PDB codes
        codes_csv = os.path.join(DATASETS_PATH, "Quantitative_data.csv")
        if not os.path.isfile(codes_csv):
            raise FileNotFoundError(f"Missing PDB code list: {codes_csv}")
        codes = pd.read_csv(codes_csv, usecols=['Pdb Code'])['Pdb Code'].astype(str)
        total   = len(codes)
        batches = ceil(total / 50)

        print(f"Total codes: {total}, splitting into {batches} batches of ≤50 each.")

        # 1) Fetch each batch into datasets/OPM/
        for b in range(batches):
            start = b * 50
            end   = min(start + 50, total)
            batch_codes = codes.iloc[start:end].tolist()

            batch_fn = f"NEWOPM_{today}_batch{b+1}.csv"
            batch_fp = os.path.join(OPM_BATCH_PATH, batch_fn)

            if os.path.exists(batch_fp):
                print(f"✓ Batch {b+1} ({start+1}-{end}) exists, skipping.")
                continue

            print(f"→ Processing batch {b+1}: codes {start+1}-{end}")
            rows = []
            for pdb_code in batch_codes:
                # index query
                idx_url = f"{self.host}/primary_structures?search={pdb_code}&pageSize=1"
                r1 = requests.get(idx_url, headers=self.headers)
                if r1.status_code != 200:
                    print(f"  ✗ Index failed for {pdb_code}")
                    continue
                objs = r1.json().get("objects") or []
                if not objs:
                    print(f"  ⚠ No record for {pdb_code}")
                    continue

                # detail query
                rec_id = objs[0].get("id")
                detail_url = f"{self.host}/primary_structures/{rec_id}"
                r2 = requests.get(detail_url, headers=self.headers)
                if r2.status_code != 200:
                    print(f"  ✗ Detail failed for {pdb_code}")
                    continue

                df = pd.json_normalize(r2.json(), sep="_")
                df['pdb_code'] = pdb_code
                rows.append(df)

                print(f"  ✔ Fetched {pdb_code}")

            # write out this batch
            if rows:
                pd.concat(rows, ignore_index=True).to_csv(batch_fp, index=False)
                print(f"★ Saved batch file {os.path.join('OPM', batch_fn)}")
            else:
                print(f"⚠ No data in batch {b+1}, no file written.")

        # 2) Combine all batch files
        part_files = sorted(
            f for f in os.listdir(OPM_BATCH_PATH)
            if f.startswith(f"NEWOPM_{today}_batch") and f.endswith(".csv")
        )
        if not part_files:
            print("⚠ No batch files found, aborting combine.")
            return

        combined_rows = []
        for fn in part_files:
            fp = os.path.join(OPM_BATCH_PATH, fn)
            combined_rows.append(pd.read_csv(fp, low_memory=False))

        master_df = pd.concat(combined_rows, ignore_index=True)
        master_df.to_csv(master_fp, index=False)
        print(f"✅ Combined all into {master_fn}")

        # 3) Expand taxonomy on the master file
        self.process_dataframe(master_df, master_fp)


if __name__ == "__main__":
    NEWOPM().fetch()
