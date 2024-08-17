from src.Commands.Migration.classMigrate import Migration
from src.Dashboard.data import columns_to_retrieve, stats_data
from src.Dashboard.services import (
    get_table_as_dataframe, 
    get_tables_as_dataframe,
    get_table_as_dataframe_download, 
    get_table_as_dataframe_exception
)
import pandas as pd

class DataService:
    @staticmethod
    def get_data_by_column_search(table="membrane_proteins", column_name="rcsentinfo_experimental_method", value=None, page=1, per_page=10, distinct_column=None):
        data = get_table_as_dataframe_exception(table, column_name, value, page, per_page, distinct_column)
        return data
    
    @staticmethod
    def get_data_by_column_search_download(column_name="rcsentinfo_experimental_method", value=None):

        data = get_table_as_dataframe_download(
            table_name="membrane_proteins", columns=columns_to_retrieve(), filter_column=column_name, 
            filter_value=value
        )
        return data
    
    # Define a function to retrieve unique values for categorical columns
    def get_unique_values_for_categorical_columns():
        table_columns = stats_data()
        # Collect unique values using a set
        unique_values = set()
        for entry in table_columns:
            for item in entry['data']:
                unique_values.add(Migration.shorten_column_name(item['value'].split('*')[0]))

        # Convert the set to a list if needed
        unique_columns = list(unique_values)
        unique_values = {}

        df = get_table_as_dataframe("membrane_proteins")
        # Retrieve unique values for each categorical column
        for column_name in unique_columns:
            unique_values[column_name] = df[column_name].unique()

        return unique_values
    
    @staticmethod
    def get_data_from_DB():
        table_names = ['membrane_proteins', 'membrane_protein_opm']
        result_df = get_tables_as_dataframe(table_names, "pdb_code")
        result_df_db = get_table_as_dataframe("membrane_proteins")
        result_df_opm = get_table_as_dataframe("membrane_protein_opm")
        result_df_uniprot = get_table_as_dataframe("membrane_protein_uniprot")
            
        all_data = pd.merge(right=result_df, left=result_df_uniprot, on="uniprot_id")
        
        return result_df, result_df_db, result_df_opm, result_df_uniprot, all_data
