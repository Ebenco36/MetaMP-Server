import os

import umap
import pandas as pd
import altair as alt
from datetime import timedelta
from sklearn.manifold import TSNE
from src.Dashboard.services import (
    get_items, get_table_as_dataframe
)
from src.MP.machine_learning_services import (
    UnsupervisedPipeline, plotCharts
)
from src.MP.services import DataService
from src.Training.services import (  
    preprocess_data, 
    group_annotation, 
    getChartForQuestion, 
    transform_dataframe,
    create_visualization,
    aggregate_inconsistencies,
    outlier_detection_implementation, 
)
from src.services.basic_plots import group_data_by_methods
from sklearn.decomposition import PCA
os.environ["NUMBA_CACHE_DIR"] = "/tmp"
from utils.redisCache import RedisCache
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering
)
from sklearn.datasets import (
    load_breast_cancer, load_diabetes
)
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score
)

class MachineLearningService:
    def __init__(self):
        self.normalized_data = None
        self.imputed_data = None
        self.dataset = None
        self.data = None
        self.data_ = None
        self.dr = None
        self.silhouette = None
        self.calinski_harabasz = None
        
    def setDataset(self, dataset):
        if dataset == 'cancer':
            self.data_ = load_breast_cancer()
        elif dataset == 'diabetes':
            # Load your diabetes dataset
            self.data_ = load_diabetes()
        return self
            
        
    def dataview(self):
        data = self.data_['data']
        self.data = pd.DataFrame(data, columns=self.data_['feature_names'])
        # self.data['target'] = self.data_['target']
        # Append the new column without mutating the original DataFrame
        df_with_new_column = self.data.assign(target=self.data_['target'])
        return df_with_new_column
        
    def impute_data(self):
        self.dataview()
        # Implement your imputation logic here
        imputer = SimpleImputer(strategy='mean')
        imputed = imputer.fit_transform(self.data)
        self.imputed_data = pd.DataFrame(imputed, columns=self.data.columns)
        return self.imputed_data

    def normalize_data(self):
        self.impute_data()
        # Implement your normalization logic here
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(self.imputed_data)
        self.normalized_data = pd.DataFrame(normalized_data, columns=self.imputed_data.columns)
        return self.normalized_data

    def apply_pca(self, n_components=2):
        self.impute_data()
        self.normalize_data()
        # Implement your PCA logic here
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(self.normalized_data)
        self.dr = pd.DataFrame(pca_result, columns=['Feature1', 'Feature2'])
        self.dr['target'] = self.data_['target']
        return self.dr

    def apply_tsne(self, n_components=2, learning_rate='auto', metric='euclidean', perplexity=15, early_exaggeration=12):
        self.impute_data()
        self.normalize_data()
        # Implement your t-SNE logic here
        tsne = TSNE(
            n_components=n_components, learning_rate=learning_rate, 
            metric=metric, perplexity=perplexity, early_exaggeration=early_exaggeration
        )
        tsne_result = tsne.fit_transform(self.normalized_data)
        self.dr = pd.DataFrame(tsne_result, columns=['Feature1', 'Feature2'])
        self.dr['target'] = self.data_['target']
        return self.dr
    
    def apply_umap(self, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', method="barnes_hut"):
        self.impute_data()
        self.normalize_data()
        # Implement your t-SNE logic here
        umap_ = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, 
            n_components=n_components, metric=metric, method=method
        )
        umap_result = umap_.fit_transform(self.normalized_data)
        self.dr = pd.DataFrame(umap_result, columns=['Feature1', 'Feature2'])
        self.dr['target'] = self.data_['target']
        return self.dr

    def apply_kmeans(
        self, n_clusters=2, max_iter=300, n_init="auto", dimension_method="tsne",
        n_neighbors=15, min_dist=0.1, n_components=2, method="barnes_hut",
        learning_rate='auto',  perplexity=15, early_exaggeration=12, metric=None
    ):
        if(dimension_method == "tsne"):
            self.apply_tsne(
                n_components=n_components, learning_rate=learning_rate, 
                metric=metric, perplexity=perplexity, early_exaggeration=early_exaggeration)
        elif(dimension_method == "umap"):
            self.apply_umap(
                n_neighbors=n_neighbors, min_dist=min_dist, 
                n_components=n_components, metric=metric, method=method
            )
        else:
            self.apply_pca(n_components=n_components)
        # Implement your KMeans clustering logic here
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init)
        kmeans_result = kmeans.fit_predict(self.dr)
        self.dr["labels"] = kmeans_result
        self.silhouette = self.silhouette_evaluate_cluster_quality(kmeans_result, self.dr)
        self.calinski_harabasz = self.calinski_evaluate_cluster_quality(kmeans_result, self.dr)
        scores = {
            "silhouette": self.silhouette, 
            "calinski_harabasz": self.calinski_harabasz
        }
        return self.dr, scores

    def apply_agglomerative(
        self, n_clusters=2, distance_threshold=0, linkage="ward", dimension_method="tsne",
        n_neighbors=15, min_dist=0.1, n_components=2, method="barnes_hut",
        learning_rate='auto',  perplexity=15, early_exaggeration=12, metric=None
    ):
        if(dimension_method == "tsne"):
            self.apply_tsne(
                n_components=n_components, learning_rate=learning_rate, 
                metric=metric, perplexity=perplexity, early_exaggeration=early_exaggeration)
        elif(dimension_method == "umap"):
            self.apply_umap(
                n_neighbors=n_neighbors, min_dist=min_dist, 
                n_components=n_components, metric=metric, method=method
            )
        else:
            self.apply_pca(n_components=n_components)
        # Implement your DBSCAN clustering logic here
        if (n_clusters and  n_clusters > 1):
            distance_threshold = None
        else:
            n_clusters = None
            
        agglomerative = AgglomerativeClustering(
            n_clusters=n_clusters, distance_threshold=distance_threshold, linkage=linkage
        )
        agglomerative_result = agglomerative.fit_predict(self.dr)
        self.dr["labels"] = agglomerative_result
        
        self.silhouette = self.silhouette_evaluate_cluster_quality(agglomerative_result, self.dr)
        self.calinski_harabasz = self.calinski_evaluate_cluster_quality(agglomerative_result, self.dr)
        scores = {
            "silhouette": self.silhouette, 
            "calinski_harabasz": self.calinski_harabasz
        }
        return self.dr, scores
    
    def apply_dbscan(
        self, eps=0.5, min_samples=5, metric=None, 
        algorithm=None, dimension_method="tsne",
        n_neighbors=15, min_dist=0.1, n_components=2, method="barnes_hut",
        learning_rate='auto',  perplexity=15, early_exaggeration=12
    ):
        if(dimension_method == "tsne"):
            self.apply_tsne(
                n_components=n_components, learning_rate=learning_rate, 
                metric=metric, perplexity=perplexity, early_exaggeration=early_exaggeration)
        elif(dimension_method == "umap"):
            self.apply_umap(
                n_neighbors=n_neighbors, min_dist=min_dist, 
                n_components=n_components, metric=metric, method=method
            )
        else:
            self.apply_pca(n_components=n_components)
        # Implement your DBSCAN clustering logic here
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm)
        dbscan_result = dbscan.fit_predict(self.dr)
        self.dr["labels"] = dbscan_result
        
        self.silhouette = self.silhouette_evaluate_cluster_quality(dbscan_result, self.dr)
        self.calinski_harabasz = self.calinski_evaluate_cluster_quality(dbscan_result, self.dr)
        scores = {
            "silhouette": self.silhouette, 
            "calinski_harabasz": self.calinski_harabasz
        }
        return self.dr, scores



    # Evaluate cluster quality using Silhouette Score
    def silhouette_evaluate_cluster_quality(self, labels, data_scaled):
        if (len(set(labels)) > 1):
            self.silhouette = silhouette_score(data_scaled, labels)
            return self.silhouette
        else:
            return 0
        return -1
        

    # Evaluate cluster quality using Silhouette Score
    def calinski_evaluate_cluster_quality(self, labels, data_scaled):
        if (len(set(labels)) > 1):
            self.calinski_harabasz = calinski_harabasz_score(data_scaled, labels)
            
            return self.calinski_harabasz
        else:
            return 0
        return -1
    
    
    

def get_use_cases(data:dict={}):
    """
    Processes data based on the specified use case and generates corresponding visualizations or outputs.

    This function handles various use cases, such as generating summary statistics, detecting outliers, identifying discrepancies, or performing group classification. Depending on the use case provided in the input dictionary, the function retrieves and processes data, caches results for efficiency, and returns the appropriate visual output or data structure.

    Parameters:
    -----------
    data : dict, optional
        A dictionary containing relevant keys for processing the use cases. 
        Expected keys and values include:
        - `groupby`: Column name for grouping data.
        - `features`: List of features for analysis.
        - `use_case`: Specifies the operation to perform (e.g., "summary_statistics", "outlier_detection", "discrepancies").
        - `chart_type`: The type of chart to be generated (if applicable).
        - `chart_width`: The desired width of the chart.
        - `outlier_detection_by_method`: Specifies the method for outlier detection (e.g., "EM", "x-ray").

    Returns:
    --------
    dict or list or None
        Depending on the use case:
        - For "summary_statistics": Returns a dictionary containing a bar chart visualization.
        - For "outlier_detection": Returns a dictionary containing a visualization for outlier detection.
        - For "discrepancies": Returns a dictionary containing a visualization for data discrepancies.
        - For other cases: May return `None` or a different structure as needed.

    Use Cases:
    ----------
    1. **Summary Statistics**:
       - Generates a bar chart representing the cumulative count of structures based on the specified groupby variable.
       - Caches the result for future requests to optimize performance.

    2. **Outlier Detection**:
       - Performs outlier detection for either "EM" or "x-ray" methods.
       - Generates visualizations indicating potential outliers in the data.
       - Caches the processed results for future use.

    3. **Discrepancies**:
       - Identifies discrepancies in the data and generates a corresponding visualization.
       - Caches the result for future requests.

    4. **Group Classification**:
       - Placeholder for future group classification logic.

    Caching:
    --------
    - Uses a Redis cache to store results for each use case.
    - Cache keys are managed based on the use case and relevant parameters.
    - Cached items are stored for 10 days (configurable).

    Example:
    --------
    >>> data = {
    ...     "use_case": "summary_statistics",
    ...     "groupby": "resolution",
    ...     "chart_type": "bar",
    ...     "chart_width": 800
    ... }
    >>> result = get_use_cases(data)
    >>> # Result is a bar chart in dictionary format.

    Notes:
    ------
    - The function is designed to handle multiple use cases, and additional cases can be integrated as needed.
    - Caching is used extensively to optimize performance, especially for computationally intensive operations.
    - The output format may vary depending on the use case.
    """
    
    cache = RedisCache()
    groupby = data.get("groupby", "group")
    features = data.get("features", [
        'emt_molecular_weight', 
        'reconstruction_num_particles',
        'processed_resolution'
    ])
    use_case = data.get("use_case", "summary_statistics")
    chart_type = data.get("chart_type", "bar")
    chart_width = data.get("chart_width", 800)
    chart_trend = data.get("chart_trend", "No")
    outlier_detection_by_method = data.get("outlier_detection_by_method", "x-ray")

    if use_case == "summary_statistics":
        if chart_trend == "No":
            # Creating a bar chart using Altair
            variable = groupby
            """
                Cache Keys Management
            """
            cache_key = "use_case_" + use_case + "_section_" + str(variable) + "_trend_" + chart_trend
            # Set expiration time for cache if used
            ttl_in_seconds = timedelta(days=10).total_seconds()
            
            cached_result = cache.get_item(cache_key)
            
            if cached_result:
                variable_counts = pd.DataFrame(cached_result)
            else:
                dataset = getChartForQuestion(column=variable, filter="")
                view_data = [
                    'resolution', 'bibliography_year', 
                    'group', 'rcsentinfo_molecular_weight', 
                    "pdb_code", "refine_ls_rfactor_rfree", 
                    "rcsentinfo_experimental_method", "taxonomic_domain"
                ]
                
                chart_data = dataset[view_data]
                # Grouping data by 'exptl_method' and counting occurrences
                variable_counts = chart_data[variable].value_counts().reset_index()
                variable_counts.columns = [variable, 'Cumulative MP Structures']

                # Store the result in the cache
                cache.set_item(cache_key, variable_counts.to_dict(), ttl=ttl_in_seconds)  # Cache for 10 days
            
            if("rcsentinfo_experimental_method" in variable_counts):
                variable_counts['rcsentinfo_experimental_method'] = variable_counts['rcsentinfo_experimental_method'].replace({
                    'EM': 'Cryo-Electron Microscopy (Cryo-EM)',
                    'X-ray': 'X-Ray Crystallography',
                    'NMR': 'Nuclear Magnetic Resonance (NMR)',
                    'Multiple methods': 'Multi-methods',
                })    
            if chart_type == "bar":
                chart_obj = alt.Chart(variable_counts).mark_bar()
            else:
                chart_obj = alt.Chart(variable_counts).mark_line()
            
            selected_feature = ""
            if "rcsentinfo_" in variable:
                selected_feature = "experimental method"
            elif variable == "taxonomic_domain":
                selected_feature = "taxonomic domain"
            else:
                selected_feature = variable
                
            chart = chart_obj.encode(
                x=alt.X(
                    variable + ':N', 
                    axis=alt.Axis(
                        labelAngle=10, 
                        title=variable.replace("rcsentinfo", "").replace("_", " ").title(),
                        labelLimit=0
                    )
                ),
                y=alt.Y(
                    'Cumulative MP Structures:Q', 
                    axis=alt.Axis(
                        title='Cumulative MP Structures'
                    )
                ),
                color=alt.Color(variable + ':N', legend=alt.Legend(title=variable.replace("rcsentinfo", " ").replace("_", " ").title())),
                tooltip=[variable + ':N', 'Cumulative MP Structures:Q']
            ).properties(
                width="container",
                title="Cumulative sum of Membrane Protein Structures by " + selected_feature
            ).interactive().configure_legend(
                orient='bottom', 
                direction = 'vertical', 
                labelLimit=0
            )
            return chart.to_dict()
        elif chart_trend == "Yes":
            variable = groupby
            """
                Cache Keys Management
            """
            cache_key = "use_case_" + use_case + "_section_" + str(variable) + "_trend_" + chart_trend + "_chart_type_" + chart_type
            # Set expiration time for cache if used
            ttl_in_seconds = timedelta(days=10).total_seconds()
            
            cached_result = cache.get_item(cache_key)
            
            if cached_result:
                chart = cached_result
            else:
                mark_type = chart_type
                bin_value = None

                dataset = getChartForQuestion(column=variable, filter="")
                view_data = [
                    'resolution', 'bibliography_year', 
                    'group', 'rcsentinfo_molecular_weight', 
                    "pdb_code", "refine_ls_rfactor_rfree", 
                    "rcsentinfo_experimental_method", "taxonomic_domain"
                ]
                
                chart_data = dataset[view_data]

                # Grouping data by 'rcsb_entry_info_experimental_method' and 'Group' and counting occurrences
                group_method_year_counts = chart_data.groupby([variable, 'bibliography_year']).size().reset_index(name='count')

                # Choose mark type dynamically
                if mark_type == "line":
                    chart = alt.Chart(group_method_year_counts).mark_line()
                else:
                    chart = alt.Chart(group_method_year_counts).mark_bar()
                
                if("rcsentinfo_experimental_method" in group_method_year_counts):
                    group_method_year_counts['rcsentinfo_experimental_method'] = group_method_year_counts['rcsentinfo_experimental_method'].replace({
                        'EM': 'Cryo-Electron Microscopy (Cryo-EM)',
                        'X-ray': 'X-Ray Crystallography',
                        'NMR': 'Nuclear Magnetic Resonance (NMR)',
                        'Multiple methods': 'Multi-methods',
                    }) 
                    group_method_year_counts = group_method_year_counts.rename(columns={
                        'rcsentinfo_experimental_method': 'Experimental Method',
                    })
                    
                chart = alt.Chart.from_dict(
                    group_data_by_methods(
                        chart_data, 
                        columns = [
                            'bibliography_year', 
                            variable
                        ], 
                        col_color=variable, 
                        chart_type=mark_type, 
                        bin_value=bin_value, 
                        interactive=True,
                        arange_legend="vertical"
                    )
                ).to_dict()
                # Store the result in the cache
                cache.set_item(cache_key, chart, ttl=ttl_in_seconds)
                
            return chart
    
    elif use_case == "outlier_detection":
        if(outlier_detection_by_method == "EM"):
            
            """
                Cache Keys Management
            """
            cache_key = "use_case_" + use_case + "_section_" + str(outlier_detection_by_method) + ','.join(str(x) for x in features) + "_width_" + str(chart_width)
            # Set expiration time for cache if used
            ttl_in_seconds = timedelta(days=10).total_seconds()
            
            cached_result = cache.get_item(cache_key)
            
            if cached_result:
                response = cached_result
            else:
                chart_width = data.get("chart_width", 800)
                variable = features
                variable = ['molecular_weight' if var == 'emt_molecular_weight' else var for var in variable]
                _, _, _, _, all_data = DataService.get_data_from_DB()
                numerical_data, categorical_data = preprocess_data(all_data, "EM")
                width_chart_single = (chart_width / len(variable)) - 70
                response = outlier_detection_implementation(
                    variable, numerical_data, 
                    categorical_data, 
                    training_attrs=['Component 1', 'Component 2'], 
                    plot_attrs=['Component 1', 'Component 2'],
                    width_chart_single=width_chart_single,
                    width_chart_single2=(chart_width - 70),
                    create_pairwise_plot_bool=True
                ).to_dict(format="vega")
                # Store the result in the cache
                cache.set_item(cache_key, response, ttl=ttl_in_seconds)  # Cache for 10 days
            
            return response
            
        elif(outlier_detection_by_method == "X-ray"):
            """
                Cache Keys Management
            """
            cache_key = "use_case_" + use_case + "_section_" + str(outlier_detection_by_method) + ','.join(str(x) for x in features) + "_width_" + str(chart_width)
            # Set expiration time for cache if used
            ttl_in_seconds = timedelta(days=10).total_seconds()
            
            cached_result = cache.get_item(cache_key)
            
            if cached_result:
                response = cached_result
            else:
                chart_width = data.get("chart_width", 800)
                variable = features
                _, _, _, _, all_data = DataService.get_data_from_DB()
                numerical_data, categorical_data = preprocess_data(all_data, "X-ray")
                width_chart_single = (chart_width / len(variable)) - 70
                response = outlier_detection_implementation(
                    variable, numerical_data, 
                    categorical_data, 
                    training_attrs=['Component 1', 'Component 2'], 
                    plot_attrs=['Component 1', 'Component 2'],
                    width_chart_single=width_chart_single,
                    width_chart_single2=(chart_width - 70),
                    create_pairwise_plot_bool=True
                ).to_dict(format="vega")
                # Store the result in the cache
                cache.set_item(cache_key, response, ttl=ttl_in_seconds)  # Cache for 10 days
                
            return response
        
    elif use_case == "discrepancies":
        """
            Cache Keys Management
        """
        cache_key = "use_case_" + use_case + "_width_" + str(chart_width)
        # Set expiration time for cache if used
        ttl_in_seconds = timedelta(days=10).total_seconds()
        
        cached_result = cache.get_item(cache_key)
        
        if cached_result:
            response = cached_result
        else:
            _, _, _, _, all_data = DataService.get_data_from_DB()
            # Define a dictionary to map keywords in famsupclasstype_type_name to expected group values
            expected_groups = {
                'Monotopic': 'MONOTOPIC MEMBRANE PROTEINS',
                'Transmembrane': 'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL',
                'Transmembrane': 'TRANSMEMBRANE PROTEINS:BETA-BARREL'
                # Add more mappings as needed
            }

            # Assume all_data is already defined and loaded with appropriate data
            dtd = all_data
            df_combined = dtd[[
                "pdb_code", "famsupclasstype_type_name", 
                "family_superfamily_classtype_name", 
                "group", "bibliography_year", 
                "rcsentinfo_experimental_method"
            ]].copy()
            df_combined.dropna(inplace=True)

            # Aggregate inconsistencies
            inconsistencies_by_year = aggregate_inconsistencies(df_combined, expected_groups)

            # Transform the aggregated data
            transformed_data = transform_dataframe(inconsistencies_by_year)
        
            response = create_visualization(data=transformed_data, chart_width=chart_width).to_dict(format="vega")
            # Store the result in the cache
            cache.set_item(cache_key, response, ttl=ttl_in_seconds)  # Cache for 10 days
        return response
    
    elif use_case == "group_classification":
        pass
