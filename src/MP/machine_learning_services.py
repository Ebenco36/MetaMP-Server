import os, sys
import numpy as np
from sklearn.calibration import LabelEncoder
sys.path.append(os.getcwd())
import pandas as pd
import altair as alt
from src.Jobs.Utils import (
    onehot_encoder,
    select_features_using_decision_tree, 
    separate_numerical_categorical, 
    evaluate_dimensionality_reduction
)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import AgglomerativeClustering
from src.Jobs.transformData import report_and_clean_missing_values

class UnsupervisedPipeline:
    def __init__(self, data_frame):
        self.all_data = data_frame
        self.data = pd.DataFrame()
        self.numerical_data = pd.DataFrame()
        self.categorical_data = pd.DataFrame()
        self.complete_numerical_data = pd.DataFrame()
        self.data_combined = pd.DataFrame()
        self.dimentionality_data = pd.DataFrame()
        self.semi_supervised_metrics = pd.DataFrame()
        self.supervised_metrics = pd.DataFrame()
        
        self.clustering_data = pd.DataFrame()
        self.clustering_number = 0 
        
        
    def fix_missing_data(self, method_type = "X-ray"):
        self.data = report_and_clean_missing_values(self.all_data, threshold=40)
        columns_to_drop = [col for col in self.data.columns if '_citation_' in col or '_count_' in col or col.startswith('count_') or col.endswith('_count') or col.startswith('revision_') or col.endswith('_revision') or col.startswith('id_') or col.endswith('_id') or col == "id"]
        
        pdb_code_remove = [
            "1PFO", "1B12", "1GOS", "1MT5", "1KN9", "1OJA", "1O5W", "1T7D", "1UUM", "2BXR",
            "1YGM", "2GMH", "2OLV", "2OQO", "2Z5X", "2QCU", "2PRM", "2VQG", "3HYW", "3VMA",
            "3I65", "3NSJ", "3PRW", "3P1L", "3Q7M", "2YH3", "3LIM", "3ML3", "3VMT", "2YMK",
            "2LOU", "4LXJ", "4HSC", "4CDB", "4TSY", "5B49", "5IMW", "5IMY", "5JYN", "5LY6",
            "6BFG", "6H03", "6DLW", "6MLU", "6NYF", "6MTI", "7LQ6", "7OFM", "7RSL", "8A1D",
            "7QAM"
        ]
        # Remove rows with PDB codes in pdb_code_remove
        self.data = self.data[~self.data['pdb_code'].isin(pdb_code_remove)]
        
        column_drop_first = [
            'pdbid', 
            'name',
            'tilterror', 
            'description',
            'family_name', 
            'species_name', 
            'exptl_method', 
            'thicknesserror', 
            'citation_country', 
            'family_name_cache',
            "bibliography_year", 
            'is_master_protein', 
            "biological_process", 
            "cellular_component", 
            "molecular_function",  
            'species_name_cache',
            'membrane_name_cache', 
            'species_description',
            'membrane_short_name', 
            'expressed_in_species', 
            "processed_resolution",
            'family_superfamily_name',
            'famsupclasstype_type_name',
            'exptl_crystal_grow_method', 
            'exptl_crystal_grow_method1',
            'family_superfamily_classtype_name', 
            'rcsentinfo_nonpolymer_molecular_weight_maximum',
            'rcsentinfo_nonpolymer_molecular_weight_minimum',
            #'gibbs',
            #"subunit_segments",
            #"thickness",
            #"tilt",
            #"rcsentinfo_molecular_weight"
        ]
        
        # Combine dynamic and additional columns to drop
        drop_columns_first = [col for col in columns_to_drop if col in self.data.columns] + column_drop_first
        # Drop columns if they exist
        self.data = self.data.drop(columns=drop_columns_first, axis=1, errors='ignore')
            
        
        if (method_type == "X-ray"):
            to_be_removed = [
                'refine_biso_mean', 
                'reflsnumber_reflns_rfree', 
                'refine_overall_suml', 
                'refpdbsolvent_shrinkage_radii', 
                'refpdbsolvent_vdw_probe_radii', 
                'refhisd_res_high',
                'symspagroup_name_hm',
                'rcsentinfo_diffrn_resolution_high_provenance_source',
                'rcsentinfo_polymer_composition',
                'rcsentinfo_na_polymer_entity_types',
                'pdbdatstatus_status_code_sf',
                'difradpdbx_monochromatic_or_laue_ml',
                'difradpdbx_diffrn_protocol',
                'refpdbrfree_selection_details', 
                'refpdbmethod_to_determine_struct', 'refine_solvent_model_details',
                'refpdbstereochemistry_target_values',
                'refine_pdbx_starting_model', 
                'difdetpdbx_collection_date', 
                'diffrn_detector_detector', 'diffrn_detector_type', 
                'diffrn_source_source', 'diffrn_source_type', 'difsoupdbx_synchrotron_beamline',
                'difsoupdbx_synchrotron_site', 'difsoupdbx_wavelength_list', 'expcrygrow_pdbx_details',
                'reflns_pdbx_ordinal', 'reflns_pdbx_redundancy', 'reflns_pdbx_rmerge_iobs'
            ]
            
            # Combine dynamic and additional columns to drop
            drop_columns = [col for col in columns_to_drop if col in self.data.columns] + to_be_removed
            # Drop columns if they exist
            self.data = self.data.drop(columns=drop_columns, axis=1, errors='ignore')
        
        return self
         
    def variable_separation(self):
        data = self.data.dropna()
        numerical_cols, categorical_cols = separate_numerical_categorical(data)
        self.numerical_data = data[numerical_cols]
        self.categorical_data = data[categorical_cols]
        return self

    def feature_selection(self, target="group"):
        # Encode string labels to integers
        self.categorical_data.reset_index(drop=True, inplace=True)
        encode_data = self.categorical_data[[
            #"species", 
            #"subgroup", 
            # "taxonomic_domain", 
            "membrane_topology_in", 
            # "membrane_topology_out",
            #"rcsentinfo_polymer_composition", 
            # "rcsentinfo_experimental_method", 
            #"rcsentinfo_na_polymer_entity_types", 
        ]]
        encoded_data = onehot_encoder(encode_data)
        encoded_data.reset_index(drop=True, inplace=True)
        self.numerical_data.reset_index(drop=True, inplace=True)
        self.complete_numerical_data = pd.concat([self.numerical_data, encoded_data], axis=1)
        self.complete_numerical_data.reset_index(drop=True, inplace=True)
        
        label_encoder = LabelEncoder()
        y = self.categorical_data[target]
        y_encoded = label_encoder.fit_transform(y)
        y_data_frame = pd.DataFrame(y_encoded, columns=[target])
        
        y_data_frame.reset_index(drop=True, inplace=True)
        new_d = pd.concat([self.complete_numerical_data, y_data_frame], axis=1)
        new_d = new_d.dropna()
        self.complete_numerical_data, top_features = select_features_using_decision_tree(new_d, target_column=target, num_features=50)
        self.complete_numerical_data.reset_index(drop=True, inplace=True)
        print(top_features)
        return self
         
    def dimensionality_reduction(
        self, reduction_method='pca_algorithm', 
        dr_columns=[], early_exaggeration=12, 
        DR_n_components=2, learning_rate="auto", 
        perplexity=30, DR_metric="euclidean", 
        n_iter=1000, method="barnes_hut", angle=0.1, 
        init="pca", n_epochs=10, min_dist=0.1, 
        n_neighbors=15, data_type="", 
        with_clustering=False
    ):
        #######################################
        ### Set Parameters for DM Reduction ###
        #######################################
        if reduction_method == "tsne_algorithm":
            methods_params = {
                't-SNE': {
                    'n_components': DR_n_components, 
                    'perplexity': perplexity
                }
            }
        elif reduction_method == "umap_algorithm":
            methods_params = {
                'UMAP': {
                    'n_components': DR_n_components, 
                    'n_neighbors': n_neighbors
                }
            }
        else:
            methods_params = {
                'PCA': {
                    'n_components': DR_n_components
                }
            }
        
        # Generate 2 D data #
        _, plot_data = evaluate_dimensionality_reduction(
            self.complete_numerical_data, 
            methods_params
        )

        combined_plot_data = pd.concat(plot_data)
        categorical_data = self.categorical_data.reset_index(drop=True)
        
        if reduction_method == "tsne_algorithm":
            self.dimentionality_data = combined_plot_data[
                combined_plot_data["Method"] == "t-SNE"
            ].reset_index(drop=True)
            self.data_combined = pd.concat([self.dimentionality_data, categorical_data], axis=1)
        
        elif reduction_method == "umap_algorithm":
            self.dimentionality_data = combined_plot_data[
                combined_plot_data["Method"] == "UMAP"
            ].reset_index(drop=True)
            self.data_combined = pd.concat([self.dimentionality_data, categorical_data], axis=1)
        
        else:
            self.dimentionality_data = combined_plot_data[
                combined_plot_data["Method"] == "PCA"
            ].reset_index(drop=True)
            self.data_combined = pd.concat([self.dimentionality_data, categorical_data], axis=1)
        
        return self
    
    def dm_data(self):
        return self
    
    def cluster_data(self, method='dbscan'):
        # self.data_combined.to_csv("welcomehome.csv")
        df = self.data_combined[["Component 1", "Component 2"]]
        # Step 2: Preprocess Data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        if method == 'dbscan':
            # Apply DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(X_scaled)
            self.clustering_number = len(set(clusters)) - (1 if -1 in clusters else 0)
            silhouette_avg = silhouette_score(X_scaled, clusters)
            print(f'DBSCAN Silhouette Score: {silhouette_avg}')
            df['cluster'] = clusters

        elif method == 'kmeans':
            # Determine the optimal number of clusters using the elbow method
            inertia = []
            silhouette_scores = []
            k_range = range(2, 10)

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

            # Select the best k (highest silhouette score)
            optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
            print(f'Optimal number of clusters: {optimal_k}')

            # Apply K-Means with the optimal number of clusters
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            self.clustering_number = optimal_k
            silhouette_avg = max(silhouette_scores)
            df['cluster'] = clusters

        elif method == 'agglomerative':
            # Determine the optimal number of clusters using the silhouette score
            silhouette_scores = []
            k_range = range(2, 10)

            for k in k_range:
                agg = AgglomerativeClustering(n_clusters=k)
                clusters = agg.fit_predict(X_scaled)
                silhouette_scores.append(silhouette_score(X_scaled, clusters))

            # Select the best k (highest silhouette score)
            optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
            print(f'Optimal number of clusters: {optimal_k}')

            # Apply Agglomerative Clustering with the optimal number of clusters
            agg = AgglomerativeClustering(n_clusters=optimal_k)
            clusters = agg.fit_predict(X_scaled)
            self.clustering_number = optimal_k
            silhouette_avg = max(silhouette_scores)
            df['cluster'] = clusters

        else:
            raise ValueError("Method not supported. Use 'dbscan', 'kmeans' or 'agglomerative'.")

        # Create an Altair scatter plot
        chart = alt.Chart(df).mark_circle(size=60).encode(
            x='feature_1',
            y='feature_2',
            color='cluster:N',
            tooltip=['feature_1', 'feature_2', 'cluster']
        ).properties(
            title=f'Clustering with {method.upper()}'
        ).interactive()
        self.clustering_data = chart.to_dict()
        
        return self

class OutlierDetection:
    def __init__(self, training_attrs=['Component 1', 'Component 2'], plot_attrs=['Component 1', 'Component 2']):
        self.training_attrs = training_attrs
        self.plot_attrs = plot_attrs
        
        self.models = {
            'IsolationForest': IsolationForest(contamination=0.07, random_state=42),
            'LocalOutlierFactor': LocalOutlierFactor(n_neighbors=30, contamination=0.07, novelty=True),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
        }
    
    def fit_predict(self, data, algorithm):
        if algorithm not in self.models:
            raise ValueError(f"Algorithm {algorithm} is not supported.")
        
        model = self.models[algorithm]
        data_copy = data.copy()
        
        if algorithm == 'LocalOutlierFactor':
            model.fit(data_copy[self.training_attrs])
            data_copy['Outlier'] = model.predict(data_copy[self.training_attrs])
        elif algorithm == 'DBSCAN':
            model.fit(data_copy[self.training_attrs])
            data_copy['Outlier'] = np.where(model.labels_ == -1, 1, 0)
        else:  # IsolationForest
            model.fit(data_copy[self.training_attrs])
            data_copy['Outlier'] = model.predict(data_copy[self.training_attrs])
        
        if algorithm != 'DBSCAN':
            data_copy['Outlier'] = data_copy['Outlier'].map({1: 0, -1: 1})
        
        self.data = data_copy
        self.data['Outlier'] = self.data['Outlier'].map({0: 'Inlier', 1: 'Outlier'})
        return self.data
    
    def visualize(self, title, numerical_data_columns, categorical_columns=['pdb_code', 'subgroup','group'], axis_name="PCA", width_chart_single=None):
        chart = alt.Chart(self.data).mark_circle().encode(
            x=alt.X(self.plot_attrs[0], title=axis_name + " 1"),
            y=alt.X(self.plot_attrs[1], title=axis_name + " 2"),
            color=alt.Color('Outlier', legend=alt.Legend(
                    orient="bottom",
                    columns=5, 
                    columnPadding=20, 
                    labelLimit=0, 
                    direction = 'vertical'
                )
            ),
            tooltip=categorical_columns + numerical_data_columns
        ).properties(
            title=title,
            width=width_chart_single
        ).interactive()
        
        return chart
    
    
def plotCharts(data, class_group=None, variables:list=["PCA 1", "PCA 2"]):
    # Plot using Altair
    color = (alt.Color(class_group + ':N', legend=alt.Legend(
            title=class_group, columns=5, 
            columnPadding=20, labelLimit=0, 
            direction = 'vertical'
        )) if not class_group is None else None)
    data = data.rename(columns={
        'Component 1': variables[0], 
        'Component 2': variables[1]
    })
    chart = alt.Chart(data).mark_circle().encode(
        x=variables[0],
        y=variables[1],
        color = color,
        tooltip=[
            'pdb_code', variables[0], variables[0],
            'Method', 'Parameter', 
            'group', 'subgroup'
        ]
    ).properties(
        width=800,
        height=500
    ).configure_legend(
        orient='bottom',
        titleLimit=0
    ).interactive()
    return chart.to_dict()