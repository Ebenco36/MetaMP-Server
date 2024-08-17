import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from src.services.Helpers.helper import create_json_response

class Regressors:

    def __init__(self, data = [], remove_by_percent = 30):
        self.data = data
        self.remove_by_percent = remove_by_percent
       

    def run_regressor_algorithm(self, algorithm, **kwargs):
        self.data, self.needed_columns, self.removed_columns = self.remove_empty_by_percent(float(self.remove_by_percent))
        regressor = algorithm(**kwargs)
        # Create a copy of the dataset to store the imputed values
        imputed_data = self.data.copy()
        if (imputed_data.empty):
            response = create_json_response(
                httpResponse=False, 
                data=[], 
                status=False, 
                status_code=200, 
                message="Filtering option removes all records. Data is empty", 
                error_message="Filtering option removes all records. Data is empty. Kindly adjust the filter to a value more than what was selected earlier."
            )
            return response
        # columns = not_needed_columns
        # imputed_data = imputed_data.drop(columns, inplace=False, axis=1)
        imputed_data = imputed_data[imputed_data.select_dtypes(include=['float', 'int', 'float64', 'int64']).columns]

        # fill empty for us to train a model that fills in None /NaN spaces
        imputed_data_ = regressor.fit_transform(imputed_data)
        imputed_data_ = pd.DataFrame(imputed_data_, columns=self.data.columns)

        
        data = imputed_data_
        return data

    # Random Forest
    def KNN_imputer_regressor(self):
        predicted_missing = self.run_regressor_algorithm(KNNImputer, n_neighbors=3)
        return predicted_missing

    def simple_regressor(self):
        predicted_missing = self.run_regressor_algorithm(SimpleImputer, strategy='mean', missing_values=np.NaN, keep_empty_features=True)
        return predicted_missing
    
    def remove_empty_by_percent(self, remove_by_percent = 90):
        # Create a copy of the dataset to store the imputed values
        imputed_data = self.data.copy()  
        # Calculate the percentage of None values in each column
        none_percentage = (imputed_data.isna().sum() / len(imputed_data)) * 100

        # Determine columns with over 50% missing values
        needed_columns = none_percentage[none_percentage <= remove_by_percent]

        # get removed columns
        # Get the columns that are not in the excluded_columns list
        removed_columns = [col for col in imputed_data.columns if col not in needed_columns.index]

        # Drop the columns from the DataFrame
        filtered_data = imputed_data[needed_columns.index]

        return filtered_data, needed_columns.index, removed_columns


