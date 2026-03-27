from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, \
    FunctionTransformer, QuantileTransformer, Normalizer, PowerTransformer
from scipy.stats import boxcox
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

def standardScaler(df):
    data = df.copy()  
    # Create a StandardScaler object
    scaler = StandardScaler()
    
    # Fit the scaler to your data and transform it
    X_train_scaled = scaler.fit_transform(data)
    imputed_data_ = pd.DataFrame(X_train_scaled, columns=data.columns)
    return imputed_data_


def minmax(df):
    data = df.copy()  
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()
    
    # Fit the scaler to your data and transform it
    X_train_scaled = scaler.fit_transform(data)
    imputed_data_ = pd.DataFrame(X_train_scaled, columns=data.columns)
    return imputed_data_

def robustScaler(df):
    data = df.copy()  
    # Create a MinMaxScaler object
    scaler = RobustScaler()
    
    # Fit the scaler to your data and transform it
    X_train_scaled = scaler.fit_transform(data)
    imputed_data_ = pd.DataFrame(X_train_scaled, columns=data.columns)
    return imputed_data_

def maxAbs(df):
    data = df.copy()  
    # Create a MinMaxScaler object
    scaler = MaxAbsScaler()
    
    # Fit the scaler to your data and transform it
    X_train_scaled = scaler.fit_transform(data)
    imputed_data_ = pd.DataFrame(X_train_scaled, columns=data.columns)
    return imputed_data_

def log_transform(df):
    data = df.copy()  
    # Log Transformation
    scaler = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)
    X_train_scaled = scaler.transform(data)
    imputed_data_ = pd.DataFrame(X_train_scaled, columns=data.columns)
    return imputed_data_

def box_cox(df):
    data = df.copy()  
    # Box-Cox Transformation
    X_train_scaled, _ = boxcox(data)
    imputed_data_ = pd.DataFrame(X_train_scaled, columns=data.columns)
    return imputed_data_

def yeo_johnson(df):
    data = df.copy()  
    # Yeo-Johnson Transformation
    scaler = PowerTransformer(method='yeo-johnson', standardize=False)
    X_train_scaled = scaler.fit_transform(data)
    imputed_data_ = pd.DataFrame(X_train_scaled, columns=data.columns)
    return imputed_data_
    
def quantile_transform(df):
    data = df.copy()  
    # Quantile Transformation (QuantileTransformer)
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=0)
    X_train_scaled = scaler.fit_transform(data)
    imputed_data_ = pd.DataFrame(X_train_scaled, columns=data.columns)
    return imputed_data_

def unit_vector(df):
    data = df.copy()  
    # Unit Vector Transformation (L2 normalization)
    scaler = Normalizer(norm='l2')
    X_train_scaled = scaler.transform(data)
    imputed_data_ = pd.DataFrame(X_train_scaled, columns=data.columns)
    return imputed_data_



def PCA_AL(df):
    data = df.copy()  
    # Create a PCA instance with the desired number of components (e.g., 2)
    pca = PCA(n_components=2)
    
    # Fit and transform the data (no labels required)
    X_pca = pca.fit_transform(data)
    imputed_data_ = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    return imputed_data_

def TSNE_AL(df):
    data = df.copy()  
    # Create a t-SNE instance with the desired number of dimensions (e.g., 2)
    tsne = TSNE(n_components=2)
    
    # Fit and transform the data (no labels required)
    X_tsne = tsne.fit_transform(data)
    imputed_data_ = pd.DataFrame(X_tsne, columns=["PCA1", "PCA2"])
    return imputed_data_
