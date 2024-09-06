import os, sys

from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
sys.path.append(os.getcwd())
# from src.services.Hel
import pandas as pd
from app import app
import altair as alt
from database.db import db
from src.Dashboard.services import (
    get_tables_as_dataframe, 
    get_table_as_dataframe
)
from src.Jobs.Utils import (
    ClassifierComparison, onehot_encoder, ClassifierComparisonSemiSupervised, select_features_using_decision_tree, 
    separate_numerical_categorical, 
    evaluate_dimensionality_reduction
)
from src.Jobs.transformData import report_and_clean_missing_values

class MLJob:
    def __init__(self):
        table_names = ['membrane_proteins', 'membrane_protein_opm']
        with app.app_context():
            # changing to all since uniprot has multiple records
            self.all_data = get_tables_as_dataframe(table_names, "pdb_code")
            self.result_df_db = get_table_as_dataframe("membrane_proteins")
            self.result_df_opm = get_table_as_dataframe("membrane_protein_opm")
            result_df_uniprot = get_table_as_dataframe("membrane_protein_uniprot")
            
        self.result_df = pd.merge(right=self.all_data, left=result_df_uniprot, on="pdb_code")
        self.data = pd.DataFrame()
        self.numerical_data = pd.DataFrame()
        self.categorical_data = pd.DataFrame()
        self.complete_numerical_data = pd.DataFrame()
        self.data_combined_PCA = pd.DataFrame()
        self.data_combined_tsne = pd.DataFrame()
        self.data_combined_UMAP = pd.DataFrame()
        
        #test
        self.data_combined_PCA_test = pd.DataFrame()
        self.data_combined_tsne_test = pd.DataFrame()
        self.data_combined_UMAP_test = pd.DataFrame()
        
        self.semi_supervised_metrics = pd.DataFrame()
        self.supervised_metrics = pd.DataFrame()
        self.over_sampling_data_selected_feature_data = pd.DataFrame()
        
        
    def fix_missing_data(self):
        self.data = report_and_clean_missing_values(self.all_data, threshold=30)
        columns_to_drop = [col for col in self.data.columns if '_citation_' in col or '_count_' in col or col.startswith('count_') or col.endswith('_count') or col.startswith('revision_') or col.endswith('_revision') or col.startswith('id_') or col.endswith('_id') or col == "id"]
        # pdb_code_remove = [
        #     "1PFO", "1B12", "1GOS", "1MT5", "1KN9", "1OJA", "1O5W", "1T7D", "1UUM", "2BXR",
        #     "1YGM", "2GMH", "2OLV", "2OQO", "2Z5X", "2QCU", "2PRM", "2VQG", "3HYW", "3VMA",
        #     "3I65", "3NSJ", "3PRW", "3P1L", "3Q7M", "2YH3", "3LIM", "3ML3", "3VMT", "2YMK",
        #     "2LOU", "4LXJ", "4HSC", "4CDB", "4TSY", "5B49", "5IMW", "5IMY", "5JYN", "5LY6",
        #     "6BFG", "6H03", "6DLW", "6MLU", "6NYF", "6MTI", "7LQ6", "7OFM", "7RSL", "8A1D",
        #     "7QAM"
        # ]
        # # Remove rows with PDB codes in pdb_code_remove
        # self.data = self.data[~self.data['pdb_code'].isin(pdb_code_remove)]

        self.data.drop(columns_to_drop + [
            'pdbid', 
            'name_y', 
            'name_x',
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
            # "biological_process", 
            # "cellular_component", 
            # "molecular_function",  
            'species_name_cache',
            'membrane_name_cache', 
            'species_description',
            'membrane_short_name', 
            # 'expressed_in_species', 
            "processed_resolution",
            'family_superfamily_name',
            'famsupclasstype_type_name',
            'exptl_crystal_grow_method', 
            'exptl_crystal_grow_method1',
            'family_superfamily_classtype_name', 
            'rcsentinfo_nonpolymer_molecular_weight_maximum',
            'rcsentinfo_nonpolymer_molecular_weight_minimum',
            "rcsentinfo_polymer_molecular_weight_minimum",
            "rcsentinfo_molecular_weight",
            "rcsentinfo_polymer_molecular_weight_maximum", 
            'gibbs',
            #"subunit_segments",
            #"thickness",
            #"tilt",
        ], axis=1, inplace=True)
        return self
        
        
    def variable_separation(self):
        data = self.data.dropna()
        numerical_cols, categorical_cols = separate_numerical_categorical(data)
        self.numerical_data = data[numerical_cols]
        self.categorical_data = data[categorical_cols]
        return self
    
    def feature_selection(self):
        # Encode string labels to integers
        self.categorical_data.reset_index(drop=True, inplace=True)
        encode_data = self.categorical_data[[
            #"species", 
            #"subgroup", 
            # "taxonomic_domain", 
            "membrane_topology_in", 
            "membrane_topology_out",
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
        y = self.categorical_data["group"]
        y_encoded = label_encoder.fit_transform(y)
        y_data_frame = pd.DataFrame(y_encoded, columns=['group'])
        
        y_data_frame.reset_index(drop=True, inplace=True)
        new_d = pd.concat([self.complete_numerical_data, y_data_frame], axis=1)
        new_d = new_d.dropna()
        self.complete_numerical_data, top_features = select_features_using_decision_tree(new_d, target_column='group', num_features=30)
        self.complete_numerical_data.reset_index(drop=True, inplace=True)
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
        #######################################
        ### Set Parameters for DM Reduction ###
        #######################################
        methods_params = {
            'PCA': {
                'n_components': 2
            },
            't-SNE': {
                'n_components': 2, 
                'perplexity': 30
            },
            'UMAP': {
                'n_components': 2, 
                'n_neighbors': 15
            }
        }
        self.complete_numerical_data = self.over_sampling_data_selected_feature_data.iloc[:, :-1]
        categorical_data = self.over_sampling_data_selected_feature_data["group"]
        # Generate 2 D data #
        reduced_data, plot_data = evaluate_dimensionality_reduction(
            self.complete_numerical_data, 
            methods_params
        )

        combined_plot_data = pd.concat(plot_data)
        pca = combined_plot_data[
            combined_plot_data["Method"] == "PCA"
        ].reset_index(drop=True)
        
        t_sne = combined_plot_data[
            combined_plot_data["Method"] == "t-SNE"
        ].reset_index(drop=True)
        
        umap = combined_plot_data[
            combined_plot_data["Method"] == "UMAP"
        ].reset_index(drop=True)
        
        self.data_combined_PCA = pd.concat([pca, categorical_data], axis=1)
        self.data_combined_PCA.to_csv("./models/semi-supervised/PCA_data.csv", index=False)
        self.data_combined_tsne = pd.concat([t_sne, categorical_data], axis=1)
        self.data_combined_tsne.to_csv("./models/semi-supervised/TSNE_data.csv", index=False)
        self.data_combined_UMAP = pd.concat([umap, categorical_data], axis=1)
        self.data_combined_UMAP.to_csv("./models/semi-supervised/UMAP_data.csv", index=False)
        
        return self
    
    def plotCharts(self):
        
        chart_list = {
            "pca": self.data_combined_PCA,
            "tsne": self.data_combined_tsne,
            "umap": self.data_combined_UMAP
        }
        
        for key, obj in chart_list.items():
            # Plot using Altair
            chart = alt.Chart(obj).mark_circle().encode(
                x='Component 1',
                y='Component 2',
                color='group',
                tooltip=["group"]
                # [
                #     'pdb_code',
                #     'Method', 'Parameter', 
                #     'group', 'subgroup'
                # ]
            ).properties(
                width=800,
                height=500
            )
            chart.save('models/' + key + '.png', scale_factor=2.0)
        return self
        
    def semi_supervised_learning(self):
        
        data_list = {
            "pca": self.data_combined_PCA,
            "tsne": self.data_combined_tsne,
            "umap": self.data_combined_UMAP
        }
        """
        data_list_test = {
            "pca_test": self.data_combined_PCA_test,
            "tsne_test": self.data_combined_tsne_test,
            "umap_test": self.data_combined_UMAP_test
        }
        """
        for key, data in data_list.items():
            X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
                data[["Component 1", "Component 2"]], 
                data["group"], test_size=0.66, 
                stratify=data["group"].to_list(), 
                random_state=42
            )
            
            cc_semi_supervised = ClassifierComparisonSemiSupervised(
                X_labeled, y_labeled, X_unlabeled, test_size=0.2
            )
            # Train and evaluate the models
            cc_semi_supervised.train_and_evaluate()
            # Plot the performance comparison
            cc_semi_supervised.plot_performance_comparison(save_filename=key)
            # Save the trained models
            cc_semi_supervised.save_models(save_filename=key)
            cc_semi_supervised.results_df.to_csv("./models/semi-supervised/metrics" + key + ".csv")
            
        """
            Without Dimensionality Reduction
        """
        categorical_data = self.over_sampling_data_selected_feature_data["group"]
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            self.complete_numerical_data, 
            categorical_data, test_size=0.4, 
            stratify=categorical_data.to_list(), 
            random_state=42
        )
        semi_supervised = ClassifierComparisonSemiSupervised(X_labeled, y_labeled, X_unlabeled)
        semi_supervised.train_and_evaluate()
        semi_supervised.plot_performance_comparison(save_filename="withoutDimensionalityReduction")
        semi_supervised.save_models(save_filename="withoutDimensionalityReduction")
        semi_supervised.results_df.to_csv("./models/semi-supervised/metrics_withoutDimensionalityReduction.csv")
            
        return self
        
    
    def supervised_learning(self):
        
        data_list = {
            "pca": self.data_combined_PCA,
            "tsne": self.data_combined_tsne,
            "umap": self.data_combined_UMAP
        }
        for key, data in data_list.items():
            cc = ClassifierComparison(
                data[[
                    "Component 1", 
                    "Component 2"
                ]], 
                data["group"], test_size=0.2
            )
            cc.train_and_evaluate()
            cc.save_models(save_filename=key)
            cc.plot_performance_comparison()
            self.supervised_metrics = cc.results_df
            cc.results_df.to_csv("./models/metrics" + key + ".csv")
        categorical_data = self.over_sampling_data_selected_feature_data["group"] 
        cc_ = ClassifierComparison(
            self.complete_numerical_data, 
            categorical_data
        )
        cc_.train_and_evaluate()
        cc_.save_models(save_filename="noDM")
        cc_.plot_performance_comparison()
        self.supervised_metrics = cc_.results_df
        cc_.results_df.to_csv("./models/metrics_withoutDimensionalityReduction.csv")
        return self
    
    
ML_job = MLJob()
(ML_job.fix_missing_data() 
    .variable_separation()
    .feature_selection()
    .dimensionality_reduction()
    .semi_supervised_learning()
    .supervised_learning()
    .plotCharts())