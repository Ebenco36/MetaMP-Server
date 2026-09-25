import os
import glob
import re
import datetime
import pandas as pd
import pycountry
from geopy.geocoders import Nominatim
from math import ceil

# Base directories
DATASETS_PATH = os.path.join('.', 'datasets')
BATCH_PATH = os.path.join(DATASETS_PATH, 'Countries')

class CountriesFetcher:
    def __init__(self, user_agent='country-fetcher'):
        # Geocoder setup
        self.geolocator = Nominatim(user_agent=user_agent)

    def create_directories(self):
        """Ensure datasets/ and datasets/Countries/ exist."""
        os.makedirs(DATASETS_PATH, exist_ok=True)
        os.makedirs(BATCH_PATH, exist_ok=True)

    def fetch(self, batch_size: int = 20) -> pd.DataFrame:
        """
        Batch-geocode ISO countries in chunks of `batch_size`, resuming from the
        latest date-stamped master file if present. Saves batches to
        datasets/Countries/ and merges into datasets/country_data_YYYY-MM-DD.csv.
        """
        self.create_directories()

        # 1) Check for existing master file
        pattern = os.path.join(DATASETS_PATH, 'country_data_*.csv')
        files = glob.glob(pattern)
        latest_date = datetime.date.min
        latest_file = None
        for f in files:
            m = re.search(r'country_data_(\d{4}-\d{2}-\d{2})\.csv$', f)
            if m:
                try:
                    d = datetime.datetime.strptime(m.group(1), '%Y-%m-%d').date()
                except ValueError:
                    continue
                if d > latest_date:
                    latest_date = d
                    latest_file = f
        if latest_file:
            print(f"Loading existing master file: {os.path.basename(latest_file)}")
            return pd.read_csv(latest_file)

        # 2) No master found: start new for today
        today = datetime.date.today().strftime('%Y-%m-%d')
        countries = list(pycountry.countries)
        total = len(countries)
        batch_count = ceil(total / batch_size)
        print(f"No existing master. Processing {total} countries in {batch_count} batches of {batch_size}.")

        # 3) Batch-process
        for batch_idx in range(batch_count):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch_countries = countries[start:end]
            batch_name = f"country_data_{today}_batch{batch_idx+1}.csv"
            batch_fp = os.path.join(BATCH_PATH, batch_name)
            if os.path.exists(batch_fp):
                print(f"✓ Batch {batch_idx+1} exists; skipping.")
                continue

            print(f"→ Processing batch {batch_idx+1}: indices {start+1}-{end}")
            rows = []
            for country in batch_countries:
                name = country.name
                flag = country.flag
                iso2 = country.alpha_2
                iso3 = country.alpha_3
                num = country.numeric
                wflag = f"{flag} {name}"
                try:
                    loc = self.geolocator.geocode(name)
                    lat, lon = (loc.latitude, loc.longitude) if loc else (None, None)
                except Exception as e:
                    print(f"  ✗ Geocode failed for {name}: {e}")
                    lat = lon = None

                rows.append({
                    'location': '',
                    'country': name,
                    'flag': flag,
                    'country_number': num,
                    'latitude': lat,
                    'longitude': lon,
                    'iso_code_2': iso2,
                    'iso_code_3': iso3,
                    'country_with_flag': wflag
                })
                print(f"  ✔ {name}")

            pd.DataFrame(rows).to_csv(batch_fp, index=False)
            print(f"★ Saved batch file: Countries/{batch_name}")

        # 4) Merge all batches into master
        part_files = sorted(
            f for f in os.listdir(BATCH_PATH)
            if f.startswith(f"country_data_{today}_batch") and f.endswith('.csv')
        )
        if not part_files:
            print("No batch files found to merge.")
            return pd.DataFrame()

        combined = pd.concat(
            [pd.read_csv(os.path.join(BATCH_PATH, f)) for f in part_files],
            ignore_index=True
        )
        master_fp = os.path.join(DATASETS_PATH, f"country_data_{today}.csv")
        combined.to_csv(master_fp, index=False)
        print(f"Combined into {os.path.basename(master_fp)}")
        return combined

if __name__ == '__main__':
    fetcher = CountriesFetcher()
    df = fetcher.fetch(batch_size=20)