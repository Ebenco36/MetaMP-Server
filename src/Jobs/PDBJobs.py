import os, ast
import re
import json
import urllib
import requests
import datetime
import pandas as pd
from bs4 import BeautifulSoup


data_path = os.environ.get("AIRFLOW_HOME")
if (data_path):
    modified_path = data_path.replace("/airflow_home", "")
else:
    modified_path = "."

def does_file_exist(file_path):
    return os.path.exists(file_path)

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
    

def extract_year(bibliography):
    match = re.search(r"\['year', '(\d{4})'\]", bibliography)
    if match:
        return match.group(1)
    else:
        return None
    
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


"""
    We are doing this for PDB code that has been replaced with something else.
    If PDB assertion number is not found while enrichment is going on then 
    try to check if it was replaced with something else. As of now, we discovered that 
    some PDB codes were replace thereby causing some issues.
"""

def extract_pdb_code(html_content):
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the link containing the replacement PDB code
    replacement_link = soup.find('a', href=re.compile(r'/structure/(\w+)'))

    if replacement_link:
        # Extract the PDB code from the href attribute
        replacement_pdb_code = replacement_link['href'].split('/')[-1]
        return replacement_pdb_code
    else:
        pattern = r'\b\d[A-Za-z]{3}\b'
        # Search the text for the pattern
        match = re.search(pattern, html_content)
        if match:
            pdb_code = match.group(0)
            return pdb_code
        else:
            return None
    
    
def check_pdb_replacement(pdb_code):
    # PDB website URL for a specific entry
    pdb_url = f"https://www.rcsb.org/structure/removed/{pdb_code}"

    # Send a GET request to the PDB website
    response = requests.get(pdb_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Check if the page contains a specific message indicating replacement
        content = response.text
        # Use regular expression to find the next 100 characters after "replaced"
        match = re.search(r'replaced(.{1,100})', content, re.IGNORECASE)
        
        if match:
            next_100_chars = match.group(1).strip()
            res = extract_pdb_code(next_100_chars)
            return res
        else:
            return pdb_code
    else:
        return pdb_code
    
class PDBJOBS:
    
    def create_directory(self, directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except FileExistsError:
            print(f"Directory '{directory_path}' already exists.")
            
    def load_data(self):
        # create directory
        self.create_directory("datasets")
        #Read the ids from the mpstruc
        current_date = datetime.date.today().strftime('%Y-%m-%d')
        self.ids = pd.read_csv(modified_path + "/datasets/mpstruct_ids.csv")
        protein_entries = []

        return self

    #Fetch the information on the ids 
    def read_in(self, ident):
        pdb_id = ident 
        try:
            # https://search.rcsb.org/structure-search-attributes.html
            # Fetch data for the original PDB ID
            req = urllib.request.urlopen("https://data.rcsb.org/rest/v1/core/entry/" + pdb_id)
            data = json.load(req)
            data = pd.json_normalize(data, sep="_")
            data.insert(1, "Pdb Code", pdb_id)
            data.insert(2, "Is Replaced", "Not Replaced")
            data.insert(3, "PDB Code Changed", "")
            data.insert(4, 'uniprot_id', add_uniprot_id(pdb_id))
            return data
        except urllib.error.HTTPError as e:
            # If there's an HTTP error, check for replacement PDB ID
            check_phase_2 = check_pdb_replacement(pdb_id)

            # Fetch data for the replacement PDB ID
            try:
                req = urllib.request.urlopen("https://data.rcsb.org/rest/v1/core/entry/" + str(check_phase_2))
                data = json.load(req)
                data = pd.json_normalize(data, sep="_")
                data.insert(1, "Pdb Code", pdb_id)
                data.insert(2, "Is Replaced", "Replaced")
                data.insert(3, "PDB Code Changed", pdb_id + " was replaced by " + check_phase_2)
                data.insert(4, 'uniprot_id', add_uniprot_id(pdb_id))
                return data
            except urllib.error.HTTPError as e:
                # If there's still an HTTP error, print an error message
                print("There is an issue with : https://data.rcsb.org/rest/v1/core/entry/" + pdb_id)
                
    def parse_data(self):
        check_quant_file = does_file_exist(modified_path + "/datasets/PDB_data.csv")
        if(not check_quant_file):
            # create directory
            self.create_directory("datasets")
            ids = self.ids["Pdb Code"]
            data = pd.DataFrame()

            #Read in all the information about the ids and display at which id you are at the moment
            i = 0
            for one_id in ids:
                entry = self.read_in(str(one_id))
                # Not available any more
                # data = data.append(entry)
                # Append the new DataFrame to the existing DataFrame

                # New method
                data = pd.concat([data, entry], ignore_index=True)
                i += 1
                if (i%10 == 0):
                    print ("Currently at:", i)

            #Save the resulting data in a .csv-table
            data.to_csv(modified_path + "/datasets/PDB_data.csv", index=False)
            return self
        else:
            print("We are done downloading...")
            return self
        
    def fetch_data(self):
        return self.load_data().parse_data().loadFile()
    
    def convert_month(self, mon):
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
    
    def loadFile(self):
        check_quant_file = does_file_exist(modified_path + "/datasets/PDB_data_transformed.csv")
        if(not check_quant_file):
            current_date = datetime.date.today().strftime('%Y-%m-%d')
            file_path = modified_path + '/datasets/PDB_data.csv'
            data = pd.read_csv(file_path, low_memory=False)

            # Filter out columns with string data type for the removal of special characters
            transform_data = data.select_dtypes(include='object')

            data[transform_data.columns] = transform_data[transform_data.columns].applymap(remove_html_tags)

            # data  = remove_bad_columns(data)

            # Apply the conversion function to each column and append parent column name
            normalized_data = []
            for one_column in data.columns:
                col_data  = data[one_column].apply(lambda x: preprocess_str_data(x))
                try:
                    normalized_col = pd.json_normalize(col_data, sep="_")
                except (AttributeError):
                    print(one_column)
                if not normalized_col.empty:
                    col = one_column
                    normalized_col.columns = [f"{col}_{col_name}" for col_name in normalized_col.columns]
                    normalized_data.append(normalized_col)

            # Merge the normalized data with the original DataFrame
            merged_df_ = pd.concat([data] + normalized_data, axis=1)
            merged_df_.index = merged_df_[['Pdb Code']]
            # extract bibiography column
            merged_df = merged_df_.copy()
            # merged_df['bibliography_year'] = merged_df['Bibliography'].apply(extract_year)
            # Replace dots with underscores in column names
            merged_df.columns = merged_df.columns.str.replace('.', '_')
            # merged_df = self.dataPrePreprocessing(merged_df)
            merged_df.to_csv(modified_path + '/datasets/PDB_data_transformed.csv', index=False)
        else:
            merged_df = pd.read_csv(modified_path + "/datasets/PDB_data_transformed.csv", low_memory=False)
            # merged_df = self.dataPrePreprocessing(merged_df)
            merged_df.to_csv(modified_path + '/datasets/PDB_data_transformed.csv', index=False)
        return merged_df
    
            
# Instantiate the class and call the function
pdb_obj = PDBJOBS()
pdb_obj.fetch_data()