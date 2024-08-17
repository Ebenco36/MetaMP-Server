import os
import re
import ast
import html
import json
import math
import inspect
import requests
import pycountry
import numpy as np
import pandas as pd
from flask import jsonify
import matplotlib.colors as mcolors
from geopy.geocoders import Nominatim
from src.Dashboard.services import get_table_as_dataframe
from src.services.data.columns.remove_columns import not_needed_columns

def convert_to_type(string_array):
    try:
        if string_array and not pd.isna(string_array):
            # Convert string array to numeric array
            value = ast.literal_eval(string_array)
        else:
            value = []
    except (Exception, ValueError, TypeError) as ex:
        value = []
    return value
    


def get_mean_value(value):
    try:
        data = convert_to_type(value)
        if (isinstance(value, (int, float))):
            average = 0
        else:
            average = sum(data) / len(data)
    except(Exception, ValueError, TypeError) as ex:
        average = 0
    return average

def extract_year(value):
    return pd.to_datetime(value, format='%b %d %Y').year
    # usage df["Year"] = df["date"].apply(extract_year)


def extract_function_names(file_path):
    function_names = []
    
    with open(file_path, 'r') as file:
        content = file.read()
        function_name_pattern = r'def\s+([\w_]+)\('
        matches = re.findall(function_name_pattern, content)
        # Remove __init__ if it exists in the list
        matches_list = [item for item in matches if item != '__init__']
        function_names.extend(matches_list)
    
    return function_names


def get_class_functions(class_instance):
    return [func for func, _ in inspect.getmembers(class_instance, inspect.ismethod)]


# Function to decode HTML entities
def decode_html_entities(text):
    return html.unescape(text)

def get_number(text):
    # Extract the number using regular expressions
    try:
        if(isinstance(text, (str))):
            number = re.findall(r'\d+(\.\d+)?', text)[0]
        else:
            number = text
        return number
    except (Exception, ValueError, TypeError) as ex:
        print(str(ex), text)


def group_by_year():
    pass


# Inject custom CSS to modify the color theme
def set_custom_theme():
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #FF0000;  /* Replace with your desired color */
            --background-color: #F5F5F5;
        }
        .block-container{
            padding-top: 0px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def remove_default_side_menu():
    no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
    """
    st.markdown(no_sidebar_style, unsafe_allow_html=True)


def does_file_exist(file_path):
    return os.path.exists(file_path)


def parser_change_dot_to_underscore(column_list:list = []):
    return [str(s).replace('.', '_') for s in column_list]


def is_date_valid_format(string):
    try:
        pd.to_datetime(string)
        return True
    except ValueError as ex:
        return False
    

def extract_year(bibliography):
    match = re.search(r"\['year', '(\d{4})'\]", bibliography)
    if match:
        return match.group(1)
    else:
        return None



#Convert month from 3-char string to respective number
def convert_month(mon):
    if (mon == "Jan"):
        return 1
    if (mon == "Feb"):
        return 2
    if (mon == "Mar"):
        return 3
    if (mon == "Apr"):
        return 4
    if (mon == "May"):
        return 5
    if (mon == "Jun"):
        return 6
    if (mon == "Jul"):
        return 7
    if (mon == "Aug"):
        return 8
    if (mon == "Sep"):
        return 9
    if (mon == "Oct"):
        return 10
    if (mon == "Nov"):
        return 11
    if (mon == "Dec"):
        return 12
    



#Strip the json strings and fix strings that would later become problematic (-> scottish names)
def prepare_column(df, column_name):
    table = df[column_name]
    table = table.replace(np.nan,"nan")
    ph = []
    for entry in table:
        entry = str(entry)
        ph_2 = entry.replace("[","")
        ph_3 = ph_2.replace("]","")
        ph_4 = ph_3.replace("\'", "\"")
        ph_41 = ph_4.replace("O\"C","OC")
        ph_42 = ph_41.replace("O\"D","OD")
        ph_5 = ph_42.split("},")

        ph_ls = []
        for one_ph in ph_5:
            if ((one_ph) != "nan"):
                ph5_len = len(ph_5)-1
                if (ph_5.index(one_ph) != ph5_len):
                    ph_ls.append(one_ph + "}")
                else:
                    ph_ls.append(one_ph)
                    
        ph.append(ph_ls)
        
    return ph


#Strip the specific entries
def strip_entry(column_entry):
    column_entry = str(column_entry)
    column_entry = column_entry.replace("[","")
    column_entry = column_entry.replace("]","")
    column_entry = column_entry.replace("\'", "")
    column_entry = column_entry.replace("{", "")
    column_entry = column_entry.replace("}", "")
    column_entry = column_entry.replace("\"", "")
    column_entry = column_entry.replace(",", "")
    column_entry = html.unescape(column_entry)
    
    column_entry = column_entry.split()
    
    return column_entry


#Find the key-value pairs within the strings and save them as paired lists
def find_kv_pairs(stripped_list):
    
    paired_list = []
    
    keys = []
    for one_entry in stripped_list:
        if (one_entry.find(":") != -1):
            keys.append(one_entry)
            
    for i in range(len(keys)):
        one_pair = []
        j = stripped_list.index(keys[i])
        if (i < len(keys)-1):
            while (j < stripped_list.index(keys[i+1])):
                one_pair.append(stripped_list[j])
                j += 1
        else:
            while (j <= len(stripped_list)-1):
                one_pair.append(stripped_list[j])
                j += 1            
        paired_list.append(one_pair)    
            
        
    return paired_list



#KIND OF OPTIONAL: save the paired lists as a string, separating elements with semicolon
def finalize_entry(paired_list):
    
    output_string = ""
    
    for one_entry in paired_list:
        for one_element in one_entry:
            output_string += one_element + " "
        if (paired_list.index(one_entry) != len(paired_list)-1):
            output_string += "; "
            
    return output_string 




#Apply all above functions to a given column based on the number of elements in each entry
def work_column(prepared_column):
    
    finished_column = []
    
    if (len(prepared_column) == 1):
        for one_entry in prepared_column:
            finished_entry = finalize_entry(find_kv_pairs(strip_entry(one_entry)))
            finished_column.append(finished_entry)
    elif (len(prepared_column) > 1):
        for one_entry_list in prepared_column:
            finished_entries = []
            for one_entry in one_entry_list:
                finished_entry = finalize_entry(find_kv_pairs(strip_entry(one_entry)))
                finished_entries.append(finished_entry)
            finished_column.append(finished_entries)
    
    return finished_column




# Function to check if a value is empty
def can_be_processed(value):
    if isinstance(value, str):
        return value.strip() != ''  # Check if string is not empty
    elif isinstance(value, (list, np.ndarray)):
        return len(value) > 0  # Check if list/array is not empty
    else:
        return False  # Value is not empty if it's neither a string nor a list/array



# Function to check if a value is empty
def preprocess_data(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, (list, np.ndarray)):
        if (len(value) > 1):
           return value
        else:
            # strip off []
            return value.strip('[]')
        return len(value) > 0  # Check if list/array is not empty
    else:
        return value


# Function to check if a string of list dictionaries is not empty
def preprocess_str_data(str_value):
    try:
        # Parse the string into a list of dictionaries using ast.literal_eval
        value_list = ast.literal_eval(str_value)
        if(isinstance(value_list, list) and len(value_list) > 1):
            # then take the first on the list 
            new_str = ast.literal_eval([value_list[0]])
            return new_str
        else:
            return ast.literal_eval(str_value.strip('[]'))
    except (SyntaxError, ValueError):
        return {}
    
def str_can_be_processed(str_value):
    try:
        # Parse the string into a list of dictionaries using ast.literal_eval
        value_list = ast.literal_eval(str_value)
        if(isinstance(value_list, list) and len(value_list) >= 1):
            return True
    except (SyntaxError, ValueError):
        return False
    

def NAPercent(df):
    NA = pd.DataFrame(data=[df.isna().sum().tolist(), [i \
           for i in (df.isna().sum()/df.shape[0]*100).tolist()]], 
           columns=df.columns, index=['NA Count', 'NA Percent']).transpose()

    return NA 



def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  



def remove_bad_columns(df):
    # Remove unnamed columns from the list.
    columns = not_needed_columns
    df = df.drop(columns, inplace=False, axis=1)
    return df


def round_to_2dp(value):
    return math.ceil(value * 100) / 100

def generate_range_bins_old(range_resolution, max_range):
    num_bins = int(max_range / range_resolution)
    range_bins = [(round_to_2dp(i * range_resolution), round_to_2dp((i + 1) * range_resolution)) for i in range(num_bins)]
    range_bins_str = [f"{item[0]}-{item[1]}" for item in range_bins]
    return range_bins_str

"""
THis is needed so that we will be able to cover for all the bins
while displaying bins 
"""
def generate_max_upper_for_bin(column, value, range_resolution_meters):
    if(column in ["resolution", "processed_resolution"]):
        value = value + range_resolution_meters
    elif(column in ["rcsentinfo_molecular_weight"]):
        value = value + range_resolution_meters
    elif(column in ["rcsentinfo_deposited_atom_count"]):
        value = value + range_resolution_meters
        
    return value

  
def generate_range_bins(range_resolution, max_range):
    num_bins = int(max_range / range_resolution) + 1
    range_bins = [(round_to_2dp(i * range_resolution), round_to_2dp((i + 1) * range_resolution)) for i in range(num_bins - 1)]
    # Include the upper limit in the last bin
    range_bins.append((round_to_2dp((num_bins - 1) * range_resolution), max_range))
    range_bins_str = [f"{format(item[0], '.2f')}-{format(item[1], '.2f')}" for item in range_bins]

    return range_bins_str


def generate_list_with_difference(num_elements, difference):
    result_list = [round_to_2dp(i * difference) for i in range(num_elements)]
    return result_list


# Custom function to convert string numbers to numeric values
def convert_to_numeric_or_str(value):
    if(value and value != " "):
        try:
            return float(value)
        except (ValueError, KeyError):
            try:
                return float(value)
            except (ValueError, KeyError):
                return value
        


# Function to remove all HTML tags from a string
def remove_html_tags(text):
    try:
        if not text is None and pd.notna(text):
            clean_text = re.sub(r'<.*?>', '', text)
            # Replace '\r' and '\n' with a space
            clean_text = clean_text.replace('\r', ' ').replace('\n', ' ')
            return clean_text
        else:
            return ''
    
    except (Exception, TypeError) as e:
        print(str(e))


def remove_underscore_change_toupper(original_string):
    return original_string.replace("_", " ")


def replace_and_separate(text):
    # Define the regex patterns for matching the substrings to replace
    patterns = [
        r'^pdbx_serial_',
        r'^pdbx_nmr_'
        r'^pdbx_database_status_',
        r'^pdbx_nmr_ensemble_',
        r'^rcsb_primary_citation_rcsb_',
        r'^rcsb_primary_citation_rcsb_'
    ]

    # Replace the substrings with 'pdbx' or 'rcsb'
    try:
        for pattern in patterns:
            text = re.sub(pattern, 'pdbx_' if pattern.startswith('^pdbx') else 'rcsb_', text)
    except (Exception, TypeError, ValueError) as e:
        print(e)

    return text
    

def filter_list(item_list, ends_with):
    result = list(filter(lambda item: item.endswith(ends_with), item_list))
    return result

def format_string_caps(input_string):
    # Replace underscores with spaces
    formatted_string = input_string.replace('_', ' ')
    
    # Capitalize the first character
    formatted_string = formatted_string.capitalize()

    return formatted_string


"""
    Converter for the basic statistics selection
    This method helps us to match the text selected
    to the list item or if doesn't exist as text then we 
    want to assume the key was used.
"""
from src.Dashboard.data import stats_data
def summaryStatisticsConverter(search_key):
    # get content of data in each object
    data = []
    for d in stats_data(for_processing=False):
        data += d['data']

    found_key = ""

    # Check if the name exists directly in the values
    for entry in data:
        if search_key == entry['value']:
            content_value = entry['value']
            found_key = content_value
            break
    else:
        # If not found, check the name in the name attributes
        for entry in data:
            if search_key == entry['name']:
                content_value = entry['value']
                found_key = content_value
                break
        else:
            found_key = ""
    
    return stats_data(for_processing=False), data, found_key

"""
    return a simple list of summary statistics options rather
    than complex dataset for filter.
"""

def summaryStatisticsFilterOptions():
    merged_list = [data for d in stats_data() for data in d["data"].values()]

    return merged_list


def removeUnderscoreIDFromList(_list):
    cleaned_list = [item for item in _list if not item.endswith("_id") and not item.endswith("_id_pub_med")]
    return cleaned_list


def tableHeader(header_list:list = []):
    list_of_objects = [
        {'id': i, 'text': format_string_caps(item.replace("rcsentinfo_", "")), 'value': item.replace("rcsentinfo_", ""), "sortable": True} for i, item in enumerate(header_list, start=1)
    ]

    return list_of_objects



def create_json_response(httpResponse=False, data=None, status=None, status_code=200, message=None, error_message=None, extras=None):
    response_data = {'status_code': status_code}
    
    if data is not None:
        response_data['data'] = data
    if message is not None:
        response_data['message'] = message
    if status is not None:
        response_data['status'] = status
    if error_message is not None:
        response_data['error_message'] = error_message
    if extras is not None:
        response_data.update(extras)
    
    if(httpResponse):
        response = jsonify(response_data)
        response.status_code = status_code
    else:
        response = response_data

    return response

def convert_json_to_dict(json_response):
    return json.loads(json_response)


def find_dict_with_value_in_nested_data(array_of_dicts, search_value):
    for data_dict in array_of_dicts:
        for inner_dict in data_dict["data"]:
            if inner_dict["value"] == search_value:
                return data_dict
    return None


def find_dicts_with_value_not_equal(array_of_dicts, search_value):
    result_list = []

    for data_dict in array_of_dicts:
        inner_data = data_dict.get("data", [])
        if not any(inner_dict.get("value") == search_value for inner_dict in inner_data):
            result_list.append(data_dict)

    return result_list or None



def generate_color_palette(start_color, end_color, num_colors):
    # Convert hex colors to RGB
    start_rgb = mcolors.hex2color(start_color)
    end_rgb = mcolors.hex2color(end_color)

    # Create a list of RGB colors in the gradient
    colors = []
    for i in range(num_colors):
        r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * (i / (num_colors - 1))
        g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * (i / (num_colors - 1))
        b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * (i / (num_colors - 1))
        colors.append((r, g, b))

    # Convert RGB colors back to hex
    hex_colors = [mcolors.rgb2hex(color) for color in colors]

    return hex_colors


    # USAGE: # Define the start and end hex colors
    # start_color = '#005EB8'  # Red
    # end_color = '#B87200'    # Green

    # # Generate a color palette with 10 colors
    # num_colors = 5
    # palette = generate_color_palette(start_color, end_color, num_colors)
    # print(palette)
    # # Display the color palette
    # fig, ax = plt.subplots(figsize=(8, 2))
    # cmap = mcolors.ListedColormap(palette)
    # cax = ax.matshow([[i] for i in range(num_colors)], cmap=cmap)
    # plt.xticks([])  # Hide x-axis labels
    # plt.yticks([])  # Hide y-axis labels
    # plt.show()

    # # Print the hex colors in the palette
    # print(palette)


import ast
def get_class_names(file_path):
    class_names = []
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_names.append(node.name)

    return class_names


def get_country_name(country_code):
    try:
        if country_code == "UK":
            country_code = "GB" 
        elif country_code == "KO":
            country_code = "KR" 
        country_name = pycountry.countries.get(alpha_2=country_code).name
        return country_name
    except AttributeError:
        return "Unknown Country"
    
def get_lat_long(country_code):
    # Initialize the geocoder
    geolocator = Nominatim(user_agent="APIH")

    # Perform geocoding for the specified country code
    country_ = get_country_name(country_code)
    location = geolocator.geocode(country_)

    # Check if location was found
    if location:
        return location.latitude, location.longitude, country_
    else:
        return None, None, None
    
     
def add_uniprot_id(pdb_id):
    # Construct the URL to query the PDBe API mappings
    url = f'https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}'

    # Send a GET request to the API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Extract the UniProt ID from the response JSON
        data = response.json()
        try:
            uniprot_id = list(data.get(pdb_id.lower(), {"UniProt": ""})["UniProt"].keys())[0]
            return uniprot_id
        except KeyError:
            print(f"UniProt ID not found for {pdb_id}")
            return None
    else:
        print(f"Error: Unable to fetch UniProt ID for {pdb_id}")
        return None
    
    
    
def fetch_and_process_chloropleth_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.content.decode('utf-8')
        data = json.loads(data)
        data_ = data.get("objects", {"europe": {}}).get("europe", {})
        GeometryCollection = data_.get("geometries", [])
        selected_data = []
        for dat in GeometryCollection:
            location = dat.get("properties", {}).get("NAME")
            country = dat.get("id")
            # Append extracted data to the list
            selected_data.append({'location': location, 'country': country, 'count': 0})
        # Create DataFrame from the list of dictionaries
        df = pd.DataFrame(selected_data)
        return df
    else:
        print("Failed to fetch data. Status code:", response.status_code)
        return None
    
def countriesDOld():
    # List to store country information
    country_data = []

    # Iterate over all countries
    for country in pycountry.countries:
        # Get country name and ISO country code (alpha-3)
        country_name = country.name
        iso_code_3 = country.alpha_3
        iso_code_2 = country.alpha_2
        country_number = country.numeric
        
        
        geolocator = Nominatim(user_agent="APIH")

        
        location = geolocator.geocode(country_name)
        # Append country information to the list
        data = {
            'country': country_name, 
            'iso_code_2': iso_code_2, 
            'iso_code_3': iso_code_3, 
            'country_number': country_number,
            'latitude': location.latitude if location else '', 
            'longitude': location.longitude  if location else ''
        }
        print(data)
        country_data.append(data)
        

    # Create DataFrame from the list of country data
    country_df = pd.DataFrame(country_data)

    return country_df

def countriesD():
    # File to store the country data
    data_file = './datasets/country_data.csv'
    
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

def operateD(df):
    # Assuming 'rcsb_primary_citation_country' is the column containing country names
    grouped_df = df.groupby('rcsb_primary_citation_country')

    # Now you can perform operations on each group, for example, count the number of occurrences of each country
    country_counts = grouped_df.size().reset_index()

    # Rename the column to 'count'
    country_counts = country_counts.rename(columns={0: 'count'})

    # Reset the index
    country_counts = country_counts.reset_index(drop=True)
    # Display the updated DataFrame
    country_counts.loc[country_counts["rcsb_primary_citation_country"]  == "UK", "rcsb_primary_citation_country"] = "GB"
    country_counts.loc[country_counts["rcsb_primary_citation_country"]  == "KO", "rcsb_primary_citation_country"] = "KR"
    country_counts = country_counts.rename(columns={'rcsb_primary_citation_country': 'iso_code_2'})
    return country_counts

def getPercentage(df, unique_field="pdb_code", column="group"):
    # Drop duplicates based on the unique field
    unique_df = df.drop_duplicates(subset=[unique_field])
    
    # Convert lists in the 'group' column to strings
    unique_df[column] = unique_df[column].astype(str)
    
    # Group by the specified column and count the occurrences
    grouped_count = unique_df[column].value_counts()
    
    # Calculate the total sum
    total_sum = grouped_count.sum()

    # Calculate the percentage of each group against the total
    percentage = ((grouped_count / total_sum) * 100).round(2)
    
    # Convert the percentage series to a DataFrame and then to a dictionary
    percentage_df = percentage.to_frame().reset_index()
    percentage_df.columns = [column, "percentage"]
    percentage_dict = percentage_df.set_index(column).to_dict()["percentage"]
    
    return percentage_dict

from datetime import date

def get_published_date(doi):
    """
    Retrieve the publication date for a given DOI from the CrossRef API and convert it to a date object.

    Args:
    doi (str): The DOI of the publication.

    Returns:
    datetime.date: The publication date of the document as a date object.
    """
    # Base URL for the CrossRef API
    base_url = "https://api.crossref.org/works/"
    
    # Construct the full URL with the DOI
    full_url = f"{base_url}{doi}"
    
    try:
        # Send a GET request to the CrossRef API
        response = requests.get(full_url)
        
        # Raise an exception if the response was unsuccessful
        response.raise_for_status()
        
        # Convert the response to JSON
        data = response.json()
        
        # Extract the date parts for the published date
        date_parts = data['message']['published']['date-parts'][0]
        
        # Handle cases where not all date parts (year, month, day) are provided
        if len(date_parts) == 1:
            # Only year is provided
            return date(date_parts[0], 1, 1)
        elif len(date_parts) == 2:
            # Year and month are provided
            return date(date_parts[0], date_parts[1], 1)
        elif len(date_parts) == 3:
            # Year, month, and day are provided
            return date(date_parts[0], date_parts[1], date_parts[2])
    except requests.RequestException as e:
        print(f"Request failed: {e}")
    except KeyError as e:
        print(f"Data parsing failed, key not found: {e}")
    except ValueError as e:
        print(f"Date conversion failed: {e}")


# START group entries by year and for different databases
def load_and_prepare_data(data_list:list = [
        "membrane_proteins", 
        "membrane_protein_mpstruct", 
        "membrane_protein_opm", 
        "membrane_protein_uniprot"
    ]
):
    
    pdb = get_table_as_dataframe(data_list[0])
    mpstruc = get_table_as_dataframe(data_list[1])
    opm = get_table_as_dataframe(data_list[2])
    uniprot = get_table_as_dataframe(data_list[3])

    # Convert date columns and extract year-month
    date_columns = [
        'rcsaccinfo_initial_release_date',
        'pdbdatstatus_recvd_initial_deposition_date',
        'rcsaccinfo_deposit_date'
    ]
    for date_col in date_columns:
        pdb = convert_date(pdb, date_col)

    pdb['Pdb Code'] = pdb['pdb_code'].astype(str).str.lower()
    mpstruc['Pdb Code'] = mpstruc['pdb_code'].astype(str).str.lower()
    opm['pdbid'] = opm['pdbid'].astype(str).str.lower()
    uniprot['uniprot_id'] = uniprot['uniprot_id'].astype(str).str.lower()
    pdb['uniprot_id'] = pdb['uniprot_id'].astype(str).str.lower()

    # Deduplicate and merge data
    merged_data = merge_datasets(pdb, mpstruc, opm, uniprot)

    return merged_data

def convert_date(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name])
    df[f"{column_name}_year_month"] = df[column_name].dt.strftime('%Y-%b')
    return df

def merge_datasets(pdb, mpstruc, opm, uniprot):
    pdb_columns = ["bibliography_year", "rcsaccinfo_initial_release_date_year_month",
                   "pdbdatstatus_recvd_initial_deposition_date_year_month",
                   "rcsaccinfo_deposit_date_year_month", "Pdb Code", "uniprot_id"]

    unique_records = {
        'pdb': pdb.drop_duplicates(subset=['Pdb Code']),
        'mpstruc': mpstruc.drop_duplicates(subset=['Pdb Code']),
        'opm': opm.drop_duplicates(subset=['pdbid']),
        'uniprot': uniprot.drop_duplicates(subset=['uniprot_id'])
    }

    merged_data_pdb = pd.merge(unique_records['pdb'][pdb_columns], unique_records['mpstruc'], on='Pdb Code', how='inner')
    merged_data_opm = pd.merge(unique_records['opm'], unique_records['pdb'][pdb_columns], left_on='pdbid', right_on='Pdb Code', how='inner')
    merged_data_uniprot = pd.merge(unique_records['uniprot'], unique_records['pdb'][pdb_columns], left_on='uniprot_id', right_on='uniprot_id', how='inner')
    # unique_records['pdb'][pdb_columns].to_csv("ssmsmsm.csv")
    # unique_records['uniprot'].to_csv("skkkaaa.csv")
    # Group by bibliography year and count entries
    return concatenate_and_group([merged_data_pdb, merged_data_opm, merged_data_uniprot])

def concatenate_and_group(dataframes):
    result = []
    for df, name in zip(dataframes, ["PDB", "OPM", "UniProt"]):
        grouped = df.groupby(['bibliography_year']).size().reset_index(name='count')
        grouped['database'] = name
        result.append(grouped)
    return pd.concat(result, axis=0)


# END group entries by year and for different databases


def get_data_by_countries(table_df):
    # Step 1: Identify and remove date columns
    date_columns = table_df.select_dtypes(include=['datetime64[ns]']).columns
    table_df = table_df.drop(columns=date_columns)
    
    # Step 2: Group by 'rcsb_primary_citation_country' and 'bibliography_year', and count 'id'
    grouped_df = table_df[["rcsb_primary_citation_country", "bibliography_year", "id"]].groupby(
        ["rcsb_primary_citation_country", "bibliography_year"]
    ).count().reset_index()
    
    # Step 3: Rename columns
    grouped_df = grouped_df.rename(
        columns={
            "rcsb_primary_citation_country": "iso_code_2",
            "bibliography_year": "year",
            "id": "count"
        }
    )
    
    # Step 4: Update specific country codes
    grouped_df.loc[grouped_df["iso_code_2"] == "UK", "iso_code_2"] = "GB"
    grouped_df.loc[grouped_df["iso_code_2"] == "KO", "iso_code_2"] = "KR"
    
    # Step 5: Merge with countries DataFrame
    all_country_count = pd.merge(grouped_df, countriesD(), on="iso_code_2", how="outer")
    
    # Step 6: Drop unnecessary columns and filter counts greater than 0
    all_country_count = all_country_count.drop(columns=["location", "iso_code_3"])
    all_country_count = all_country_count[all_country_count["count"] > 0]
    
    # Step 7: Handle NaN values
    all_country_count['iso_code_2'] = all_country_count['iso_code_2'].fillna('Unknown Country')
    all_country_count['country'] = all_country_count['country'].fillna('Unknown Country')
    all_country_count['country_number'] = all_country_count['country_number'].fillna('Unknown Country')
    all_country_count = all_country_count.where(pd.notnull(all_country_count), "")
    
    return all_country_count