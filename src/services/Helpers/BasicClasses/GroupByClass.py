import pandas as pd
from src.services.graphs.BasicPlot import BasicPlot

class GroupBy:

    def __init__(self, data_frame:pd.DataFrame):
        self.basic_plot = BasicPlot()
        self.data_frame = data_frame

    """Not needed though. Once done with optimization, this will be removed."""
    def group_by_years_extra(self, group_by_column:str, quant_columns:list = [], aggregate_type:str = "sum"):
        frame, pivot_col = self.basic_plot.set_data_frame(data_frame = self.data_frame)\
            .group_by_years_extra(group_by_column, quant_columns, aggregate_type)
        return frame, pivot_col
        
    def group_by_year(self, group_by_column:str, quant_columns:list = [], aggregate_type:str = "sum"):
        frame, pivot_col = self.basic_plot.set_data_frame(data_frame = self.data_frame)\
            .group_by_years(group_by_column, quant_columns, aggregate_type)
        
        return frame, pivot_col
    
    def group_by_month(self, group_by_column:str, quant_columns:list = [], aggregate_type:str = "sum"):
        frame, pivot_col = self.basic_plot.set_data_frame(self.data_frame)\
            .group_by_months(group_by_column, quant_columns, aggregate_type)
        
        return frame, pivot_col
    
    def group_by_other(self, group_by_column:str, quant_columns:list = [], aggregate_type:str = "sum"):
        frame, pivot_col = self.basic_plot.set_data_frame(self.data_frame)\
            .group_by_others(group_by_column, quant_columns, aggregate_type)
        return frame, pivot_col