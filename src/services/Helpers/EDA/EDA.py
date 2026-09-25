import pandas as pd
import matplotlib.pyplot as plt
from src.services.graphs.helpers import Graph

class EDA:
    def __init__(self, data):
        self.data = data
    
    def summary_statistics(self):
        summary_stats = self.data.describe()
        return summary_stats
    
    def missing_values(self):
        missing_values = self.data.isnull().sum()
        return missing_values
    
    def correlation_matrix(self):
        correlation_matrix = self.data.corr()
        return correlation_matrix
    
    def plot_correlation_matrix(self, correlation_matrix):
        graph = Graph(correlation_matrix)
        disp = graph.correlation_matrix(['variable2:O', 'variable:O'], "correlation:Q" , "correlation_label")
        return disp
    
    def get_data(self):
        return self.data
    
   

