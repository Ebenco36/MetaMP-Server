# reset variable back to normal 
# %reset -sf
import os
import logging
import pandas as pd
from src.services.data.columns.dates import dates_columns
from src.services.data.columns.norminal import all_descriptors
from src.services.data.columns.quantitative.quantitative import cell_columns, rcsb_entries
from src.services.data.columns.quantitative.quantitative_array import quantitative_array_column
from src.services.Helpers.helper import (
    does_file_exist, extract_year, 
    preprocess_str_data, remove_html_tags
)

pd.options.mode.chained_assignment = None  # default='warn' 
logger = logging.getLogger(__name__)

class DataImport:
    def __init__(self, needed_columns=None) -> None:
        # setting class properties here.
        self.needed_columns = needed_columns if needed_columns else all_descriptors\
        + dates_columns + cell_columns + rcsb_entries + quantitative_array_column


    def loadFile(self):
        check_quant_file = does_file_exist("Quantitative_data.csv")
        if(not check_quant_file):
            directory = os.path.join(os.getcwd(), 'dist', 'data_folder')
            file_path = os.path.join(directory, 'enriched_db.csv')
            data = pd.read_csv(file_path, low_memory=False)

            # Filter out columns with string data type for the removal of special characters
            transform_data = data.select_dtypes(include='object')

            data[transform_data.columns] = transform_data[transform_data.columns].applymap(remove_html_tags)

            # data  = remove_bad_columns(data)

            # Apply the conversion function to each column and append parent column name
            normalized_data = []
            for one_column in data.columns:
                col_data  = data[one_column].apply(lambda x: preprocess_str_data(x))
                normalized_col = pd.DataFrame()
                try:
                    normalized_col = pd.json_normalize(col_data, sep="_")
                except (AttributeError):
                    logger.debug("Skipping normalization for column %s", one_column)
                if not normalized_col.empty:
                    col = one_column
                    normalized_col.columns = [f"{col}_{col_name}" for col_name in normalized_col.columns]
                    normalized_data.append(normalized_col)

            # Merge the normalized data with the original DataFrame
            merged_df_ = pd.concat([data] + normalized_data, axis=1)


            merged_df_.index = merged_df_[['Pdb Code']]
            # extract bibiography column
            merged_df = merged_df_.copy()
            merged_df['bibliography_year'] = merged_df['Bibliography'].apply(extract_year)
            # Replace dots with underscores in column names
            merged_df.columns = merged_df.columns.str.replace('.', '_')
        else:
            merged_df = pd.read_csv("Quantitative_data.csv", low_memory=False)
        return merged_df
