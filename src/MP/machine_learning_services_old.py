from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.services.Helpers.Regressors.index import Regressors
from src.services.Helpers.helper import does_file_exist
from src.services.Helpers.machine_learning_al.UnsupervisedMachineLearning import MachineLearning
from src.services.Helpers.machine_learning_al.dimensionality_reduction import DimensionalityReduction

from src.services.Helpers.machine_learning_al.normalization import Normalization

class UnsupervisedPipeline:

    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.filter_frame = None
        self.numeric_data = None
        self.normalized_data = None
        self.reduced_data = None
        self.imputed_data = None
        self.clustered_data = None
        self.dendogram = None
        self.silhouette = None
        self.calinski_harabasz = None
        
    def dataPrePreprocessing(self, column_to_group_by = 'rcsentinfo_experimental_method', method = None):
        from src.MP.data import not_needed_columns, cat_list, spelling_corrections
        list2 = [
            'unnamed_0', 'unnamed_0_x', 'unnamed_01', 'unnamed_0_y', 
            'unnamed_0_1', 'rcspricitation_pdbx_database_id_pub_med', 
            'citpdbdatabase_id_pub_med', 'rcsentcontainer_identifiers_pubmed_id', 
            'rcspricitation_journal_id_csd', 'pdbnmrrepresentative_conformer_id', 
            'citation_journal_id_csd', 'diffrn_crystal_id', 'diffrn_id', 
            'diffrn_radiation_diffrn_id', 'diffrn_radiation_wavelength_id', 
            'exptl_crystal_id', 'emexpentity_assembly_id', 'em_experiment_id', 
            'diffrn_detector_diffrn_id', 'diffrn_source_diffrn_id', 
            'expcrygrow_crystal_id', 'pdbreftwin_crystal_id', 'pdbreftwin_diffrn_id', 
            'pdbreftwin_domain_id', 'pdbx_sgproject_id', 'em3d_fitting_id', 
            'em3d_reconstruction_id', 'em3recimage_processing_id', 
            'emctfcorrection_em_image_processing_id', 'em_ctf_correction_id', 
            'em_entity_assembly_id', 'ementassembly_parent_id', 
            'em_image_recording_id', 'emimarecording_imaging_id', 
            'em_imaging_id', 'em_imaging_specimen_id', 'em_particle_selection_id', 
            'emparselection_image_processing_id', 'emsinparticle_entity_id', 
            'emsinparticle_entity_image_processing_id', 'em_software_id', 
            'emsofimage_processing_id', 'em_specimen_experiment_id', 'em_specimen_id', 
            'em_vitrification_id', 'em_vitrification_specimen_id', 
            'pdbnmrexptl_conditions_id', 'pdbnmrexptl_experiment_id', 
            'pdbnmrexptl_solution_id', 'pdbnmrexptl_spectrometer_id', 
            'pdbnmrexptl_sample_conditions_conditions_id', 'pdbnmrsample_details_solution_id', 
            'pdbnmrspectrometer_spectrometer_id', 'em3d_fitting_list_id', 
            'em3fitlist_3d_fitting_id', 'em_helical_entity_id', 
            'emhelentity_image_processing_id', 'pdbinirefinement_model_id', 
            'em3d_crystal_entity_id', 'em3cryentity_image_processing_id', 
            'em_diffraction_id', 'em_diffraction_imaging_id', 
            'emdifshell_em_diffraction_stats_id', 'em_diffraction_shell_id', 
            'em_diffraction_stats_id', 'emdifstats_image_processing_id', 
            'em_embedding_id', 'em_embedding_specimen_id', 'pdbsercrystallography_sample_delivery_diffrn_id', 
            'pdbsercrystallography_sample_delivery_injection_diffrn_id', 
            'pdbsercrystallography_sample_delivery_fixed_target_diffrn_id', 
            'pdbsercrystallography_data_reduction_diffrn_id', 
            'pdbsercrystallography_measurement_diffrn_id', 'em_staining_id', 
            'em_staining_specimen_id', 'em2d_crystal_entity_id', 'em2cryentity_image_processing_id', 'id'
        ]
        new_filters = [item for item in not_needed_columns if item not in list2]
        for col in new_filters:
            if col in self.data_frame.columns:
                self.data_frame = self.data_frame.drop(col, inplace=False, axis=1)
        
    
        """
        self.data_frame [cat_list] = self.data_frame [cat_list].apply(lambda x: x.str.strip())
        # Apply spelling corrections to the specified columns
        self.data_frame .replace({'Expressed in Species': spelling_corrections, 'Species': spelling_corrections}, inplace=True)
        # Apply transformations to DataFrame columns by removing spaces and converting to a lower case
        self.data_frame [cat_list] = self.data_frame [cat_list].apply(lambda x: x.str.lower().str.replace(' ', '_'))
        # Get unique values in the specified column
        #unique_values = df[column_to_group_by].unique()
        """
        
        # Filter the DataFrame based on the unique value in the specified column
        if(method):
            subset_df = self.data_frame[self.data_frame[column_to_group_by] == method]
        else:
            subset_df = self.data_frame
            
        # One-hot encode the subset
        encoder = OneHotEncoder(sparse_output=False, drop='if_binary')
        encoder.fit(subset_df[cat_list])
        one_hot_encoded = encoder.transform(subset_df[cat_list])
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(cat_list))
        
        # Reset index if needed
        one_hot_df.reset_index(drop=True, inplace=True)
        subset_df.reset_index(drop=True, inplace=True)

        # Concatenate the one-hot encoded subset with the original subset
        subset_result_df = pd.concat([subset_df, one_hot_df], axis=1)
        self.data_frame = subset_result_df
        # Save the subset DataFrame to a CSV file
        return self
    
    def preprocessingWithHotEncoding(self):
        from src.MP.data import not_needed_columns
        list2 = [
            'unnamed_0', 'unnamed_0_x', 'unnamed_01', 'unnamed_0_y', 
            'unnamed_0_1', 'rcspricitation_pdbx_database_id_pub_med', 
            'citpdbdatabase_id_pub_med', 'rcsentcontainer_identifiers_pubmed_id', 
            'rcspricitation_journal_id_csd', 'pdbnmrrepresentative_conformer_id', 
            'citation_journal_id_csd', 'diffrn_crystal_id', 'diffrn_id', 
            'diffrn_radiation_diffrn_id', 'diffrn_radiation_wavelength_id', 
            'exptl_crystal_id', 'emexpentity_assembly_id', 'em_experiment_id', 
            'diffrn_detector_diffrn_id', 'diffrn_source_diffrn_id', 
            'expcrygrow_crystal_id', 'pdbreftwin_crystal_id', 'pdbreftwin_diffrn_id', 
            'pdbreftwin_domain_id', 'pdbx_sgproject_id', 'em3d_fitting_id', 
            'em3d_reconstruction_id', 'em3recimage_processing_id', 
            'emctfcorrection_em_image_processing_id', 'em_ctf_correction_id', 
            'em_entity_assembly_id', 'ementassembly_parent_id', 
            'em_image_recording_id', 'emimarecording_imaging_id', 
            'em_imaging_id', 'em_imaging_specimen_id', 'em_particle_selection_id', 
            'emparselection_image_processing_id', 'emsinparticle_entity_id', 
            'emsinparticle_entity_image_processing_id', 'em_software_id', 
            'emsofimage_processing_id', 'em_specimen_experiment_id', 'em_specimen_id', 
            'em_vitrification_id', 'em_vitrification_specimen_id', 
            'pdbnmrexptl_conditions_id', 'pdbnmrexptl_experiment_id', 
            'pdbnmrexptl_solution_id', 'pdbnmrexptl_spectrometer_id', 
            'pdbnmrexptl_sample_conditions_conditions_id', 'pdbnmrsample_details_solution_id', 
            'pdbnmrspectrometer_spectrometer_id', 'em3d_fitting_list_id', 
            'em3fitlist_3d_fitting_id', 'em_helical_entity_id', 
            'emhelentity_image_processing_id', 'pdbinirefinement_model_id', 
            'em3d_crystal_entity_id', 'em3cryentity_image_processing_id', 
            'em_diffraction_id', 'em_diffraction_imaging_id', 
            'emdifshell_em_diffraction_stats_id', 'em_diffraction_shell_id', 
            'em_diffraction_stats_id', 'emdifstats_image_processing_id', 
            'em_embedding_id', 'em_embedding_specimen_id', 'pdbsercrystallography_sample_delivery_diffrn_id', 
            'pdbsercrystallography_sample_delivery_injection_diffrn_id', 
            'pdbsercrystallography_sample_delivery_fixed_target_diffrn_id', 
            'pdbsercrystallography_data_reduction_diffrn_id', 
            'pdbsercrystallography_measurement_diffrn_id', 'em_staining_id', 
            'em_staining_specimen_id', 'em2d_crystal_entity_id', 'em2cryentity_image_processing_id', 'id'
        ]
        new_filters = [item for item in not_needed_columns if item not in list2]
        for col in new_filters:
            if col in self.data_frame.columns:
                self.data_frame = self.data_frame.drop(col, inplace=False, axis=1)
        self.data_frame.to_csv("Frame1.csv")
        return self
    
    def modify_dataframe(self, excluded_fields:list=[]):
        self.filter_frame = self.data_frame
        for field in excluded_fields:
            if field == "reflns":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('reflns')]
            if field == "refine":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('refine')]
            if field == "rcsb_":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('rcsb_')]
            if field == "rcs":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('rcs')]
            if field == "ref":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('ref')]
                
            if field == "expressed_in_species":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('expressed_in_species')]
            if field == "pdb":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('pdb')]
            if field == "taxonomic_domain":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('taxonomic_domain')]
            if field == "symspa":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('symspa')]
                
            if field == "expcry":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('expcry')]
            if field == "em2cry":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('em2cry')]
            if field == "software_":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('software_')]
        
            if field == "diffrn":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('diffrn')]
            if field == "exptl":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('exptl')]
            if field == "cell_":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('cell_')]
            if field == "group_":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('group_')]
            if field == "subgroup_":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('subgroup_')]
            if field == "species_":
                self.filter_frame = self.filter_frame.loc[:, ~self.filter_frame.columns.str.startswith('species_')]
        
        self.filter_frame.to_csv("Frame2.csv")
        return self

    
    def select_numeric_columns(self):
        self.numeric_data = self.filter_frame.select_dtypes(include=['float', 'int', 'float64', 'int64'])
        self.numeric_data.to_csv("Frame3.csv")
        missing_percentage = (self.numeric_data.isnull().sum() / len(self.numeric_data)) * 100
        # print(missing_percentage)
        return self

    def apply_normalization(self, normalization_method='min_max_normalization'):
        if self.imputed_data is None:
            raise ValueError("Numeric data has not been selected. Call apply_imputation() first.")
        
        normalization = Normalization(self.imputed_data)

        if normalization_method == 'min_max_normalization':
            self.normalized_data = normalization.min_max_normalization()
        elif normalization_method == 'robust_scaler_normalization':
            self.normalized_data = normalization.robust_scaler_normalization()
        elif normalization_method == 'max_abs_scaler_normalization':
            self.normalized_data = normalization.max_abs_scaler_normalization()
        elif normalization_method == 'standard_scaler_normalization':
            self.normalized_data = normalization.standard_scaler_normalization()
        else:
            raise ValueError("Invalid normalization method. Choose from 'min_max', 'robust_scaler', 'max_abs_scaler', 'standard_scaler'.")
        
        return self

    def apply_dimensionality_reduction(
        self, reduction_method='pca_algorithm', dr_columns=[], early_exaggeration=12, DR_n_components=2,
        learning_rate="auto", perplexity=30, DR_metric="euclidean", n_iter=1000, 
        method="barnes_hut", angle=0.1, init="pca", n_epochs=10, min_dist=0.1, 
        n_neighbors=15, data_type="", with_clustering=False
        
    ):
        # check if reduced file already exist
        if(with_clustering and does_file_exist("./datasets/dimensionalityReduction_" + reduction_method + data_type + ".csv")):
            self.reduced_data = pd.read_csv("./datasets/dimensionalityReduction_" + reduction_method + data_type + ".csv")
            # Make sure to reset the index before using positional indexing
            self.reduced_data.reset_index(drop=True, inplace=True)
            self.reduced_data = self.reduced_data.iloc[:, :2] # We believe the first 2 columns are numeric
            
        else:
            if self.normalized_data is None:
                raise ValueError("Normalized data has not been generated. Call apply_normalization() first.")

            dimensionality_reduction = DimensionalityReduction(self.normalized_data, n_features=DR_n_components, pca_columns=dr_columns)

            if reduction_method == 'pca_algorithm':
                self.reduced_data, _ = dimensionality_reduction.pca_algorithm()
            elif reduction_method == 'TruncatedSVD_algorithm':
                self.reduced_data, _ = dimensionality_reduction.TruncatedSVD_algorithm()
            elif reduction_method == 'NMF_algorithm':
                self.reduced_data, _ = dimensionality_reduction.NMF_algorithm()
            elif reduction_method == 'FactorAnalysis_algorithm':
                self.reduced_data, _ = dimensionality_reduction.FactorAnalysis_algorithm()
            elif reduction_method == 'ica_algorithm':
                self.reduced_data, _ = dimensionality_reduction.ica_algorithm()
            elif reduction_method == 'gaussian_random_proj_algorithm':
                self.reduced_data, _ = dimensionality_reduction.gaussian_random_proj_algorithm()
            elif reduction_method == 'tsne_algorithm':
                self.reduced_data, _ = dimensionality_reduction.tsne_algorithm(
                    early_exaggeration = early_exaggeration,
                    learning_rate =  learning_rate,
                    perplexity = perplexity, metric = DR_metric,
                    n_iter = n_iter, method = method, 
                    angle = angle, init = init
                )
            elif reduction_method == 'umap_algorithm':
                self.reduced_data, _ = dimensionality_reduction.umap_algorithm(
                    metric=DR_metric, 
                    n_epochs=n_epochs,
                    min_dist=min_dist, 
                    n_neighbors=n_neighbors,
                )
            elif reduction_method == 'isomap_algorithm':
                self.reduced_data, _ = dimensionality_reduction.isomap_algorithm()
            elif reduction_method == 'ile_algorithm':
                self.reduced_data, _ = dimensionality_reduction.ile_algorithm()
            else:
                raise ValueError("Invalid dimensionality reduction method. Choose from 'pca', 'truncated_svd', 'nmf', 'factor_analysis', 'ica', 'gaussian_random_proj', 'tsne', 'isomap', 'ile'.")
            
        return self

    def apply_imputation(self, imputation_method='KNN_imputer_regressor', remove_by_percent=30):
        if self.numeric_data is None:
            raise ValueError("numeric_data data has not been generated. Call select_numeric_columns() first.")

        numeric_data_for_imputation = self.numeric_data  # Use original numeric data for imputation
        if imputation_method == 'KNN_imputer_regressor':
            imputer = Regressors(numeric_data_for_imputation, remove_by_percent)
            self.imputed_data = imputer.KNN_imputer_regressor()
        elif imputation_method == 'simple_regressor':
            imputer = Regressors(numeric_data_for_imputation, remove_by_percent)
            self.imputed_data = imputer.simple_regressor()
        else:
            raise ValueError("Invalid imputation method. Choose from 'knn', 'simple'.")
        
        return self

    def apply_clustering(
        self, method='agglomerative_clustering', 
        n_clusters=3, 
        n_components = 2, min_samples=15,
        linkage = "ward", metric = "euclidean",
        eps = 0.3, distance_threshold = None
    ):
        if self.reduced_data is None:
            raise ValueError("Imputed data has not been generated. Call apply_dimensionality_reduction() first.")
        
        clustering = MachineLearning(
            self.reduced_data, eps=eps, 
            distance_threshold = distance_threshold,
            min_samples=min_samples, n_clusters=n_clusters, 
            n_components=n_components, linkage = linkage, 
            metric = metric
        )
        
        if method == 'dbscan_clustering':
            self.clustered_data, _, _, self.dendogram, self.silhouette, self.calinski_harabasz = clustering.dbscan_clustering()
        elif method == 'mean_shift_clustering':
            self.clustered_data, _, _, self.dendogram, self.silhouette, self.calinski_harabasz = clustering.mean_shift_clustering()
        elif method == 'agglomerative_clustering':
            self.clustered_data, _, _, self.dendogram, self.silhouette, self.calinski_harabasz = clustering.agglomerative_clustering()
        elif method == 'kMeans_clustering':
            self.clustered_data, _, _, self.dendogram, self.silhouette, self.calinski_harabasz = clustering.kMeans_clustering()
        elif method == 'gaussian_clustering':
            self.clustered_data, _, _, self.dendogram, self.silhouette, self.calinski_harabasz = clustering.gaussian_clustering()
        else:
            raise ValueError("Invalid clustering method. Choose from 'dbscan', 'mean_shift', 'agglomerative', 'kmeans', 'gaussian'.")
        
        return self
    
    def prepare_plot_DR(self, group_by="species"):
        
        columns_to_check = ["pdb_code", "name", "taxonomic_domain", "resolution"]

        missing_columns = [column for column in columns_to_check if column not in self.data_frame.columns]

        if not missing_columns:
            group_data = self.data_frame[[group_by] + columns_to_check]
        else:
            group_data = self.data_frame[[group_by]]
            
        # group_data = self.data_frame[[group_by, "pdb_code", "name", "taxonomic_domain", "resolution"]]
        
        self.clustered_data = pd.concat([self.clustered_data, group_data], axis=1)
        
        accuracy_metrics = {
            "silhouette": float(self.silhouette), 
            "calinski_harabasz": float(self.calinski_harabasz)
        }
        
        return self.clustered_data, self.numeric_data, accuracy_metrics
    
    def prepare_plot_DR_(self, group_by="species"):
        
        columns_to_check = ["pdb_code", "name", "taxonomic_domain", "resolution"]

        missing_columns = [column for column in columns_to_check if column not in self.data_frame.columns]

        if not missing_columns:
            group_data = self.data_frame[[group_by] + columns_to_check]
        else:
            group_data = self.data_frame[[group_by]]
            
        # group_data = self.data_frame[[group_by, "pdb_code", "name", "taxonomic_domain", "resolution"]]
        
        self.reduced_data = pd.concat([self.reduced_data, group_data], axis=1)
        
        return self.reduced_data, self.numeric_data
    
    def send_result(self):
        return self.reduced_data