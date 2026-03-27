import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

import pandas as pd
import numpy as np

def remove_empty_by_percent(data, remove_by_percent = 90):
    # Create a copy of the dataset to store the imputed values
    imputed_data = data.copy()  
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


# Read the CSV file containing "New_dataframe_value.csv" files
file_list = [
  "New_dataframe.csv",
  "New_dataframe_X-ray.csv", 
  "New_dataframe_EM.csv", 
  "New_dataframe_NMR.csv", 
  "New_dataframe_Multiple methods.csv",
  "NoOneHotNew_dataframe_X-ray.csv", 
  "NoOneHotNew_dataframe_EM.csv", 
  "NoOneHotNew_dataframe_NMR.csv", 
  "NoOneHotNew_dataframe_Multiple methods.csv"
]  # Add the actual file names

for file_name in file_list:
    # Read the CSV file
    df = pd.read_csv(f"./{file_name}", low_memory=False)

    # Select numeric columns
    numeric_columns = df.select_dtypes(include=['number', 'int', 'float']).columns.tolist()
    df_numeric = df[numeric_columns]
    # Remove empty columns by percentage
    processed_df, _, _ = remove_empty_by_percent(df_numeric, 30)

    # Save the processed DataFrame to a new CSV file
    processed_df.to_csv(f"./dataSpace/{file_name}_Processed.csv", index=False)
