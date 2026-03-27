import os

from src.services.graphs.helpers import convert_chart
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, NMF, FactorAnalysis
from sklearn.manifold import TSNE
# import umap
from sklearn.manifold import Isomap
import pandas as pd
import numpy as np
import json
import altair as alt


class DimensionalityReduction:
    def __init__(self, X = [], n_features=2, pca_columns:list=[]):
        self.X = X
        self.n_features = n_features if n_features > 1 else 2
        self.dr_columns = pca_columns

    def pca_algorithm(self):
        model = PCA(self.n_features).fit(self.X)
        model_data = model.transform(self.X)
        data = pd.DataFrame(model_data, columns=self.dr_columns)
        explainable = self.explainable(model, self.X.columns)

        return data, explainable
    """
    def TruncatedSVD_algorithm(self):
        model = TruncatedSVD(self.n_features).fit(self.X)
        model_data = model.transform(self.X)
        data = pd.DataFrame(model_data, columns=self.dr_columns)
        explainable = self.explainable(model, self.X.columns)

        return data, explainable
    
    def NMF_algorithm(self):
        model = NMF(self.n_features).fit(self.X)
        model_data = model.transform(self.X)
        data = pd.DataFrame(model_data, columns=self.dr_columns)
        explainable = self.explainable(model, self.X.columns)

        return data, explainable

    def FactorAnalysis_algorithm(self):
        model = FactorAnalysis(self.n_features).fit(self.X)
        model_data = model.transform(self.X)
        data = pd.DataFrame(model_data, columns=self.dr_columns)
        explainable = self.explainable(model, self.X.columns)

        return data, explainable

    
    def ica_algorithm(self):
        model = FastICA(n_components=self.n_features).fit(self.X)
        model_data = model.transform(self.X)
        data = pd.DataFrame(model_data, columns=self.dr_columns)
        explainable = self.explainable(model, self.X.columns)

        return data, explainable
    """
    def gaussian_random_proj_algorithm(self):
        model = GaussianRandomProjection(n_components=self.n_features).fit(self.X)
        model_data = model.transform(self.X)
        data = pd.DataFrame(model_data, columns=self.dr_columns)
        explainable = self.explainable(model, self.X.columns)

        return data, explainable
    
    def tsne_algorithm(self, **kwargs):
        model = TSNE(n_components=self.n_features, **kwargs)
        model_data = model.fit_transform(self.X)
        data = pd.DataFrame(model_data, columns=self.dr_columns)
        explainable = self.explainable(model, self.X.columns)

        return data, explainable
    
    """
    def isomap_algorithm(self):
        model = Isomap(n_components=self.n_features)
        model_data = model.fit_transform(self.X)
        data = pd.DataFrame(model_data, columns=self.dr_columns)
        explainable = self.explainable(model, self.X.columns)

        return data, explainable
    """
    
    # def umap_algorithm(
    #     self, 
    #     n_epochs=10,
    #     min_dist=None,
    #     n_neighbors=None, 
    #     metric="euclidean"
    # ):
    #     model = umap.UMAP(
    #         metric=metric, 
    #         min_dist=min_dist, 
    #         n_neighbors=n_neighbors, 
    #         n_components=self.n_features
    #     )
    #     model_data = model.fit_transform(self.X)
    #     data = pd.DataFrame(model_data, columns=self.dr_columns)
    #     explainable = ''
        
    #     return data, explainable

    def pca_contribution(self, data, data_  , n_components):
        # Access the principal components
        components = data_.components_

        # Get the variables contributing the most to each component
        contributing_variables = []

        # Iterate over the components
        for component in components:
            # Get the indices of the variables with the highest absolute values in the component
            contributing_indices = np.argsort(np.abs(component))[::-1][:n_components]
            contributing_variables.append([data.columns[i] for i in contributing_indices])

        return contributing_variables


    def tsne_contribution(self, data, data_):
        # Get the variables in relation to the t-SNE output
        variable_contributions = []

        # Iterate over the components (only 2 in this case)
        for i in range(data_.shape[1]):
            # Get the variables' values for the given component
            component_values = data_[:, i]

            # Sort the variables based on their values for the component
            sorted_indices = np.argsort(component_values)

            # Get the names of the variables corresponding to the sorted indices
            sorted_variables = [data.columns[j] for j in sorted_indices]

            # Append the sorted variables to the variable_contributions list
            variable_contributions.append(sorted_variables)

        return variable_contributions
    """
    def ile_algorithm(self):
        model = LocallyLinearEmbedding(n_components=self.n_features)
        model_data = model.fit_transform(self.X)
        data = pd.DataFrame(model_data, columns=self.dr_columns)
        explainable = self.explainable(model, self.X.columns)

        return data, explainable

    """
    def explainable(self, model, features:list = []):

        n_components = None
        explained_variance_ratio = None

        if hasattr(model, 'components_'):
            n_components = model.components_

        if hasattr(model, 'explained_variance_ratio_'):
            explained_variance_ratio = model.explained_variance_ratio_

        # Create a list to store individual charts
        charts = []
        graph_data = []

        if (not n_components is None):
            # Given NDArray
            data_array = n_components

            # Convert the array to a Pandas DataFrame
            df = pd.DataFrame(data_array, columns=features)
            data_transformed = df.T
            data_transformed.columns = self.dr_columns
            
            # Loop through the columns and create charts
            for col in self.dr_columns:
                chart = alt.Chart(data_transformed.reset_index()).mark_bar().encode(
                    x='index',
                    y=col,
                    color=alt.value('blue')
                ).properties(
                    title=f'Plot of {col} against x (Attributes)'
                )
                charts.append(chart)

            # Concatenate the charts vertically
            combined_charts = alt.hconcat(*charts)

            graph_data = convert_chart(combined_charts)

        return {
            "explained_variance_ratio": explained_variance_ratio.tolist() if (not explained_variance_ratio is None) else [],
            "n_components": n_components.tolist() if (not n_components is None) else [],
            "graph_data": graph_data,
        }