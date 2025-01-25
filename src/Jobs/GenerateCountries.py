import os, sys
import pandas as pd
import pycountry
from geopy.geocoders import Nominatim

sys.path.append(os.getcwd())


def countriesD():
    # File to store the country data
    modified_path = "."
    data_file = modified_path + '/datasets/country_data.csv'
    check_file = modified_path + "/datasets/country_data.csv"
    if os.path.exists(check_file):
        print(f"Error: File {check_file} already downloaded. You can delete to download new one.")
        return

    # Check if the data file exists
    if os.path.isfile(data_file):
        # If file exists, load the data from the file
        country_df = pd.read_csv(data_file)
        # Find the last country index processed
        last_country_index = country_df.index[-1] + 1
    else:
        # If file doesn't exist, create an empty DataFrame
        country_df = pd.DataFrame(columns=['location', 'country', 'flag', 'country_number', 'latitude', 'longitude'])
        # Start from the first country
        last_country_index = 0

    # Set the batch size
    batch_size = 20
    
    # Iterate over all countries in batches
    for i in range(last_country_index, len(pycountry.countries), batch_size):
        batch_countries = list(pycountry.countries)[i:i+batch_size]
        # Iterate over countries in the current batch
        for country in batch_countries:
            # Get country name, ISO country code (alpha-3), and numeric code
            country_name = country.flag + ' ' + country.name
            flag = country.flag
            iso_code_3 = country.alpha_3
            iso_code_2 = country.alpha_2
            country_number = country.numeric

            # Geocode country to get latitude and longitude
            try:
                geolocator = Nominatim(user_agent="APIH")
                location = geolocator.geocode(country_name)
            except Exception as e:
                print(f"Geocoding failed for {country_name}: {e}")
                continue

            # Append country information to the DataFrame
            data = {
                'country': country_name, 
                'flag': flag, 
                'iso_code_2': iso_code_2, 
                'iso_code_3': iso_code_3, 
                'country_number': country_number,
                'latitude': location.latitude if location else None, 
                'longitude': location.longitude if location else None
            }
            country_df = pd.concat([country_df, pd.DataFrame([data])], ignore_index=True)
            print("Processed:", country_name)
        
        # Save the DataFrame to the data file after processing each batch
        country_df.to_csv(data_file, index=False)
        print("Country data saved to:", data_file)

    return country_df


def generateCountries():
    return countriesD()

generateCountries()