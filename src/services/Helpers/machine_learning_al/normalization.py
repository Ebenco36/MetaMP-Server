from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
import pandas as pd

class Normalization:

    def __init__(self, data_frame = []):
        self.data_frame = data_frame

    def select_option(self):
        # not processing any information here just sending it back.
        return self.data_frame
    
    def min_max_normalization(self):
        scaler = MinMaxScaler()
        normalized_data = pd.DataFrame(scaler.fit_transform(self.data_frame), columns=self.data_frame.columns)
        return normalized_data
    
    def robust_scaler_normalization(self):
        scaler = RobustScaler()
        normalized_data = pd.DataFrame(scaler.fit_transform(self.data_frame), columns=self.data_frame.columns)
        return normalized_data

    def max_abs_scaler_normalization(self):
        scaler = MaxAbsScaler()
        normalized_data = pd.DataFrame(scaler.fit_transform(self.data_frame), columns=self.data_frame.columns)
        return normalized_data
    

    def standard_scaler_normalization(self):
        scaler = StandardScaler()
        normalized_data = pd.DataFrame(scaler.fit_transform(self.data_frame), columns=self.data_frame.columns)
        return normalized_data