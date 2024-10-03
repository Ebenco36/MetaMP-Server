import os, sys
sys.path.append(os.getcwd())
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np

from app import app
import altair as alt
from database.db import db
from src.Dashboard.services import get_tables_as_dataframe, get_table_as_dataframe
from src.Jobs.Utils import (
    ClassifierComparison, onehot_encoder, ClassifierComparisonSemiSupervised,
    select_features_using_decision_tree, separate_numerical_categorical,
    evaluate_dimensionality_reduction
)
from src.Jobs.transformData import report_and_clean_missing_values

class MLJob:
    def __init__(self):
        self.num_runs = 1  # Define the number of runs for averaging metrics
        self.random_state = 42  # For reproducibility
        np.random.seed(self.random_state)
        
        # Load data
        self.load_data()
        
        # Data containers
        self.data = pd.DataFrame()
        self.numerical_data = pd.DataFrame()
        self.categorical_data = pd.DataFrame()
        self.complete_numerical_data = pd.DataFrame()
        self.data_combined_PCA = pd.DataFrame()
        self.data_combined_tsne = pd.DataFrame()
        self.data_combined_UMAP = pd.DataFrame()
        self.semi_supervised_metrics = pd.DataFrame()
        self.supervised_metrics = pd.DataFrame()
        self.over_sampling_data_selected_feature_data = pd.DataFrame()
        
    def load_data(self):
        table_names = ['membrane_proteins', 'membrane_protein_opm']
        with app.app_context():
            # Load data from tables
            self.all_data = get_tables_as_dataframe(table_names, "pdb_code")
            self.result_df_db = get_table_as_dataframe("membrane_proteins")
            self.result_df_opm = get_table_as_dataframe("membrane_protein_opm")
            result_df_uniprot = get_table_as_dataframe("membrane_protein_uniprot")
        
        self.result_df = pd.merge(right=self.all_data, left=result_df_uniprot, on="pdb_code")
    
    def fix_missing_data(self):
        self.data = report_and_clean_missing_values(self.all_data, threshold=30)
        columns_to_drop = [col for col in self.data.columns if '_citation_' in col or '_count_' in col or col.startswith('count_') or col.endswith('_count') or col.startswith('revision_') or col.endswith('_revision') or col.startswith('id_') or col.endswith('_id') or col == "id"]
        self.data.drop(columns_to_drop + ['pdbid', 'name_y', 'name_x', 'tilterror', 'description', 'family_name', 'species_name', 'exptl_method', 'thicknesserror', 'citation_country', 'family_name_cache', "bibliography_year", 'is_master_protein', 'species_name_cache', 'membrane_name_cache', 'species_description', 'membrane_short_name', "processed_resolution", 'family_superfamily_name', 'famsupclasstype_type_name', 'exptl_crystal_grow_method', 'exptl_crystal_grow_method1', 'family_superfamily_classtype_name', 'rcsentinfo_nonpolymer_molecular_weight_maximum', 'rcsentinfo_nonpolymer_molecular_weight_minimum', "rcsentinfo_polymer_molecular_weight_minimum", "rcsentinfo_molecular_weight", "rcsentinfo_polymer_molecular_weight_maximum", 'gibbs'], axis=1, inplace=True)
        return self

    def variable_separation(self):
        data = self.data.dropna()
        numerical_cols, categorical_cols = separate_numerical_categorical(data)
        self.numerical_data = data[numerical_cols]
        self.categorical_data = data[categorical_cols]
        return self
    
    def feature_selection(self):
        self.categorical_data.reset_index(drop=True, inplace=True)
        encode_data = self.categorical_data[["membrane_topology_in", "membrane_topology_out"]]
        encoded_data = onehot_encoder(encode_data)
        encoded_data.reset_index(drop=True, inplace=True)
        self.numerical_data.reset_index(drop=True, inplace=True)
        self.complete_numerical_data = pd.concat([self.numerical_data, encoded_data], axis=1).reset_index(drop=True)
        
        label_encoder = LabelEncoder()
        y = self.categorical_data["group"]
        y_encoded = label_encoder.fit_transform(y)
        y_data_frame = pd.DataFrame(y_encoded, columns=['group']).reset_index(drop=True)
        
        combined_data = pd.concat([self.complete_numerical_data, y_data_frame], axis=1).dropna()
        self.complete_numerical_data, top_features = select_features_using_decision_tree(combined_data, target_column='group', num_features=30)
        print(top_features)
        
        self.over_sampling_data_selected_feature_data = pd.concat([self.complete_numerical_data, y], axis=1)
        
        raw_data = pd.concat([
            self.numerical_data, 
            self.categorical_data[[
                    "pdb_code", "membrane_topology_in", 
                    "membrane_topology_out"
                ]
            ]], axis=1)
        raw_data.reset_index(drop=True, inplace=True)
        raw_data.to_csv("./models/semi-supervised/without_reduction_data.csv", index=False)
        return self
        
    def dimensionality_reduction(self):
        methods_params = {
            'PCA': {'n_components': 2},
            't-SNE': {'n_components': 2, 'perplexity': 30},
            'UMAP': {'n_components': 2, 'n_neighbors': 15}
        }
        self.complete_numerical_data = self.over_sampling_data_selected_feature_data.iloc[:, :-1]
        categorical_data = self.over_sampling_data_selected_feature_data["group"]
        
        reduced_data, plot_data = evaluate_dimensionality_reduction(self.complete_numerical_data, methods_params)
        combined_plot_data = pd.concat(plot_data)
        
        self.data_combined_PCA = pd.concat([combined_plot_data[combined_plot_data["Method"] == "PCA"].reset_index(drop=True), categorical_data], axis=1)
        self.data_combined_PCA.to_csv("./models/semi-supervised/PCA_data.csv", index=False)
        
        self.data_combined_tsne = pd.concat([combined_plot_data[combined_plot_data["Method"] == "t-SNE"].reset_index(drop=True), categorical_data], axis=1)
        self.data_combined_tsne.to_csv("./models/semi-supervised/TSNE_data.csv", index=False)
        
        self.data_combined_UMAP = pd.concat([combined_plot_data[combined_plot_data["Method"] == "UMAP"].reset_index(drop=True), categorical_data], axis=1)
        self.data_combined_UMAP.to_csv("./models/semi-supervised/UMAP_data.csv", index=False)
        
        return self

    def plot_charts(self):
        chart_list = {
            "pca": self.data_combined_PCA,
            "tsne": self.data_combined_tsne,
            "umap": self.data_combined_UMAP
        }
        
        for key, obj in chart_list.items():
            chart = alt.Chart(obj).mark_circle().encode(
                x='Component 1',
                y='Component 2',
                color='group',
                tooltip=["group"]
            ).properties(width=800, height=500)
            chart.save('models/' + key + '.png', scale_factor=2.0)
        return self

    def run_classification(self, X, y, model_class, filename_prefix, X_unlabeled=None):
        """Run classification and save results."""
        metrics_list = []

        for run in range(self.num_runs):
            if X_unlabeled is not None:  # Semi-Supervised Case
                model = model_class(X, y, X_unlabeled, test_size=0.2, random_state=self.random_state + run)
            else:  # Supervised Case
                model = model_class(X, y, test_size=0.2, random_state=self.random_state + run)

            # Train and evaluate the models
            model.train_and_evaluate()

            # Collect metrics for aggregation
            metrics_list.append(model.results_df)

            # Save model and plot performance
            model.save_models(save_filename=f"{filename_prefix}_{run}")
            model.plot_performance_comparison(save_filename=f"{filename_prefix}_{run}")

        # Concatenate all metric results
        concatenated_metrics = pd.concat(metrics_list)

        # Select only numeric columns for aggregation
        numeric_columns = concatenated_metrics.select_dtypes(include=['number'])

        # Aggregate only numeric columns
        aggregated_metrics = numeric_columns.groupby(level=0).agg(['mean', 'std'])

        # Combine non-numeric data with aggregated numeric data (if needed)
        non_numeric_columns = concatenated_metrics.select_dtypes(exclude=['number']).drop_duplicates()
        if not non_numeric_columns.empty:
            aggregated_metrics = pd.concat([aggregated_metrics, non_numeric_columns], axis=1)

        # Save aggregated metrics
        aggregated_metrics.to_csv(f"./models/{filename_prefix}_metrics_mean.csv")
        print(f"Metrics saved to ./models/{filename_prefix}_metrics_mean.csv")
    
    def semi_supervised_learning(self):
        """Run semi-supervised learning on different dimensionality-reduced datasets."""
        data_list = {
            "pca": self.data_combined_PCA,
            "tsne": self.data_combined_tsne,
            "umap": self.data_combined_UMAP
        }
        
        for key, data in data_list.items():
            X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
                data[["Component 1", "Component 2"]],
                data["group"], test_size=0.66,
                stratify=data["group"].to_list(),
                random_state=self.random_state
            )
            
            # Semi-supervised learning requires both labeled and unlabeled data
            self.run_classification(X_labeled, y_labeled, ClassifierComparisonSemiSupervised, f"semi_supervised_{key}", X_unlabeled)

        # Without Dimensionality Reduction
        categorical_data = self.over_sampling_data_selected_feature_data["group"]
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            self.complete_numerical_data,
            categorical_data, test_size=0.66,
            stratify=categorical_data.to_list(),
            random_state=self.random_state
        )
        
        self.run_classification(X_labeled, y_labeled, ClassifierComparisonSemiSupervised, "semi_supervised_no_dr", X_unlabeled)
        
        return self

    def supervised_learning(self):
        """Run supervised learning on different dimensionality-reduced datasets."""
        data_list = {
            "pca": self.data_combined_PCA,
            "tsne": self.data_combined_tsne,
            "umap": self.data_combined_UMAP
        }
        
        for key, data in data_list.items():
            X = data[["Component 1", "Component 2"]]
            y = data["group"]
            
            self.run_classification(X, y, ClassifierComparison, f"supervised_{key}")
        
        # Without Dimensionality Reduction
        X = self.complete_numerical_data
        y = self.over_sampling_data_selected_feature_data["group"]
        self.run_classification(X, y, ClassifierComparison, "supervised_no_dr")
        
        return self


ml_job = MLJob()
    
# Run the steps sequentially
ml_job.fix_missing_data()\
        .variable_separation()\
        .feature_selection()\
        .dimensionality_reduction()\
        .plot_charts()\
        .semi_supervised_learning()\
        .supervised_learning()

print("Machine Learning Job completed successfully.")