import os
import pandas as pd
import datetime
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
        """Ensure datasets and Countries subdirs exist."""
        os.makedirs(DATASETS_PATH, exist_ok=True)
        os.makedirs(BATCH_PATH, exist_ok=True)

    def fetch(self, batch_size: int = 200) -> pd.DataFrame:
        """
        Batch-geocode all ISO countries in groups of `batch_size`.
        Saves each batch to datasets/Countries/country_data_YYYY-MM-DD_batchN.csv,
        then merges into datasets/country_data_YYYY-MM-DD.csv.
        Returns the combined DataFrame.
        """
        self.create_directories()
        # Today stamp
        today = datetime.date.today().strftime('%Y-%m-%d')
        master_name = f"country_data_{today}.csv"
        master_fp = os.path.join(DATASETS_PATH, master_name)

        # List of countries
        countries = list(pycountry.countries)
        total = len(countries)
        batch_count = ceil(total / batch_size)
        print(f"Total countries: {total}; batching into {batch_count} of {batch_size} each.")

        # Process each batch
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
                alpha2 = country.alpha_2
                alpha3 = country.alpha_3
                numeric = country.numeric
                # Geocode
                try:
                    loc = self.geolocator.geocode(name)
                    lat, lon = (loc.latitude, loc.longitude) if loc else (None, None)
                except Exception as e:
                    print(f"  ✗ Geocode failed for {name}: {e}")
                    lat, lon = None, None
                rows.append({
                    'country': name,
                    'iso_code_2': alpha2,
                    'iso_code_3': alpha3,
                    'country_number': numeric,
                    'latitude': lat,
                    'longitude': lon
                })
                print(f"  ✔ {name}")

            # Save batch file
            batch_df = pd.DataFrame(rows)
            batch_df.to_csv(batch_fp, index=False)
            print(f"★ Saved batch file Countries/{batch_name}")

        # Merge all batches
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
        combined.to_csv(master_fp, index=False)
        print(f"✅ Combined all into {master_name}")
        return combined

# Example usage
if __name__ == '__main__':
    fetcher = CountriesFetcher()
    df = fetcher.fetch(batch_size=200)
