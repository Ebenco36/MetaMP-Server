import pandas as pd
from src.services.exceptions.DataFrameNotFound import DataFrameNotFound
from src.services.Helpers.helper import is_date_valid_format

class BasicPlot:
    def __init__(self):
        pass

    def set_data_frame(self, data_frame):
        self.data_frame = data_frame
        return self

    def check_data_frame(self):
        if(self.data_frame.empty):
            raise DataFrameNotFound("Data frame has not been set or data frame is empty")
        
    """Not needed though. Once done with optimization, this will be removed."""
    def group_by_years_extra(self, date_string:str, quant:list, aggregate_type:str):
        self.check_data_frame()
        # New column name
        col_year = date_string+'_year'
        self.data_frame[col_year] = self.data_frame[date_string]
        # Group the data by 'Year' and calculate the sum of 'Value' for each year
        grouped_data = self.dynamic_aggregate_selection (col_year, quant, aggregate_type)
        # return the grouped data
        return grouped_data, col_year
    

    def group_by_years(self, date_string:str, quant:list, aggregate_type:str):
        self.check_data_frame()
        # New column name
        col_year = date_string+'_year'
        # Convert the 'Date' column to datetime format
        self.data_frame[date_string] = pd.to_datetime(self.data_frame[date_string])
        # Extract the year from the 'Date' column and create a new column 'Year'
        self.data_frame[col_year] = self.data_frame[date_string].dt.year
        # Group the data by 'Year' and calculate the sum of 'Value' for each year
        grouped_data = self.dynamic_aggregate_selection (col_year, quant, aggregate_type)
        # return the grouped data
        return grouped_data, col_year


    def group_by_months(self, date_string:str, quant:list, aggregate_type:str):
        self.check_data_frame()
        # New column name
        col_month = date_string+'_month'
        # Convert the 'Date' column to datetime format
        self.data_frame[date_string] = pd.to_datetime(self.data_frame[date_string])
        # Extract the year from the 'Date' column and create a new column 'Year'
        self.data_frame[col_month] = self.data_frame[date_string].dt.month
        # Group the data by 'Year' and calculate the sum of 'Value' for each year
        grouped_data = self.dynamic_aggregate_selection (col_month, quant, aggregate_type)
        # return the grouped data
        return grouped_data, col_month


    def group_by_others(self, _string:str, quant:list, aggregate_type:str):
        self.check_data_frame()
        # New column name
        col_ = _string
        # Group the data by 'Year' and calculate the sum of 'Value' for each year
        grouped_data = self.dynamic_aggregate_selection (col_, quant, aggregate_type)
        # return the grouped data
        return grouped_data, col_
    
    

    """
        Below are the different aggregate methods we can leverage on rather than just using sum
        we have others such as mean, median, mode, variance, standard deviation and others.
    """

    def dynamic_aggregate_selection(self, col_, quant, aggregate_type):
        data = self.data_frame.groupby([col_])[quant]
        # Call the method using getattr()
        grouped_data = getattr(data, aggregate_type)().reset_index()

        return grouped_data