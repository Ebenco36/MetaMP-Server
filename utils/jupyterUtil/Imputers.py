# Drop rows with missing values
import pandas as pd
import numpy as np
from sklearn.svm import SVR
# from fancyimpute import SoftImpute
from sklearn.impute import SimpleImputer, KNNImputer

def KNN_AL(df):
    data = df.copy()  
    # Create a KNNImputer instance
    imputer = KNNImputer(n_neighbors=2)  # You can adjust the number of neighbors as needed
    
    # Impute missing values
    imputed_data = imputer.fit_transform(data)
    imputed_data_ = pd.DataFrame(imputed_data, columns=data.columns)
    return imputed_data_


def soft_imputer_regressor(df):
    data = df.copy()  
    
    # imputer = SoftImpute() 
    # imputed_data = imputer.fit_transform(data)
    # imputed_data_ = pd.DataFrame(imputed_data, columns=data.columns)
    return data

def simple_regressor(df):
    data = df.copy()  
    imputer = SimpleImputer(strategy='mean', missing_values=np.NaN, keep_empty_features=True)  # You can adjust the number of neighbors as needed
    
    # Impute missing values
    imputed_data = imputer.fit_transform(data)
    imputed_data_ = pd.DataFrame(imputed_data, columns=data.columns)
    return imputed_data_