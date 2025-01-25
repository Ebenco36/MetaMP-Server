import io
import joblib
import numpy as np
import pandas as pd
import altair as alt
from io import StringIO
import plotly.io as pio
import plotly.express as px
from src.MP.data import cat_list
from flask.views import MethodView
from src.MP.services import DataService
from src.utils.response import ApiResponse
from werkzeug.utils import secure_filename
from flask_restful import Resource, reqparse
from src.services.graphs.helpers import Graph
from flask import jsonify, request, send_file
from src.MP.Helpers import get_joblib_files_and_splits
from utils.package import separate_numerical_categorical
from src.Training.services import get_quantification_data
from src.Jobs.Utils import (
    evaluate_dimensionality_reduction, 
    onehot_encoder
)
from utils.package import select_features_using_decision_tree
from src.Jobs.transformData import report_and_clean_missing_values
from src.Dashboard.services import export_to_csv, export_to_excel
from src.MP.machine_learning_services_old import UnsupervisedPipeline
from src.Dashboard.data import (
    EM_columns, MM_columns, 
    NMR_columns, X_ray_columns, 
    reduce_value_length_version2
)
from src.services.Helpers.fields_helper import (
    dimensionality_reduction_algorithms_helper_kit, 
    machine_algorithms_helper_kit, missing_algorithms_helper_kit, 
    normalization_algorithms_helper_kit, transform_data_view,
    transform_data_dict_view
)
from src.middlewares.auth_middleware import token_required


class DataResource(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('experimental_method', type=str, help='select resolution method')
        self.parser.add_argument('download', type=str, help='download data as well')
        
    def get(self):
        # Access query parameters from the URL
        experimental_method = request.args.get('experimental_method', None)
        page = request.args.get('page', 1)
        per_page = request.args.get('per_page', 10)
        
        # download format
        download = request.args.get('download', None)
        if(download):
            data = DataService.get_data_by_column_search_download("rcsentinfo_experimental_method", experimental_method)
            if(download == "csv"):
                filename = 'output_data.csv'
                export_to_csv(data['data'], filename)
                return send_file(filename, as_attachment=True)
            elif(download == "xlsx"):
                filename = 'output_data.xlsx'
                export_to_excel(data['data'], filename)
                return send_file(filename, as_attachment=True)
        else:
            # page default
            data = DataService.get_data_by_column_search(
                column_name="rcsentinfo_experimental_method", 
                value=experimental_method, 
                page=page, 
                per_page=per_page
            )
            if(data):
                return ApiResponse.success(data, "Fetch records successfully.")
            else: 
                return ApiResponse.error("Not found!", 404)
        
class CategoricalDataResource(Resource):
    def get(self):
        data = DataService.get_unique_values_for_categorical_columns()
        if(data):
            return data
        else: 
            return ApiResponse.error("Not found!", 404)
        
class getFilterBasedOnMethodResource(Resource):
    def get(self):
        method_type = request.args.get('method_type', "All")
        method_type = None if method_type == "All" else method_type
        chart_types = transform_data_dict_view(
            [
                {"value": "bar_plot", "text": "bar_chart" },
                {"value": "line_plot", "text": "line_chart" },
                {"value": "scatter_plot", "text":  "scatter_chart"},
            ], 
            'Chart_options', 'single', [], False
        ) # graph_types_kit()
        categorical_column = transform_data_view(
            cat_list, 'categorical', 'single', [], False
        )
        
        if(method_type == "X-ray"):
            # Convert columns to numeric (if possible)
            numeric_columns = list(set(X_ray_columns(include_general=False)))
            numeric_columns_filtered_list = [item for item in numeric_columns if item not in ("group", "species")] + ["resolution", "rcsentinfo_resolution_combined"]
        elif(method_type == "NMR"):
            # Convert columns to numeric (if possible)
            numeric_columns = list(set(NMR_columns(include_general=False)))
            numeric_columns_filtered_list = [item for item in numeric_columns if item not in ("group", "species")]
        elif(method_type == "Multiple methods"):
            # Convert columns to numeric (if possible)
            numeric_columns = list(set(MM_columns(include_general=False)))
            numeric_columns_filtered_list = [item for item in numeric_columns if item not in ("group", "species")] + ["resolution", "rcsentinfo_resolution_combined"]
        else:
            # Convert columns to numeric (if possible)
            numeric_columns = list(set(X_ray_columns(include_general=False))  & set(EM_columns(include_general=False)) & set(NMR_columns(include_general=False)))
            
            numeric_columns_filtered_list = [item for item in numeric_columns if item not in ("group", "species")] + ["resolution", "rcsentinfo_resolution_combined"]
        numeric_columns_filtered_list.remove("rcsentinfo_resolution_combined")  
        
        quantitative_columns = transform_data_view(
            numeric_columns_filtered_list, 'quantitative', 
            'single', [], False
        )
        methods = [
            "All", "EM", 
            "Multiple methods", 
            "NMR", "X-ray"
        ]
        experimental_method = transform_data_view(methods, 'experimental_method', 'single', [], False)
        
            
        filter_list = {
            "experimental_method": experimental_method,
            "quantitative": quantitative_columns,
            "categorical": categorical_column,
            "chart_types": chart_types
        }
        return ApiResponse.success(filter_list, "Fetch filter list successfully.")
    
    def post(self):
        data = request.get_json()
        x_axis = data.get('x_axis', "rcsentinfo_molecular_weight")
        x_axis = "" if (x_axis is None or len(x_axis) == 0) else x_axis
        y_axis = data.get('y_axis', "rcsentinfo_deposited_solvent_atom_count")
        y_axis = "resolution" if (y_axis is None or len(y_axis) == 0) else y_axis
        
        categorical_axis = data.get('categorical_axis', None)
        categorical_axis = None if (categorical_axis is None or len(categorical_axis) == 0) else categorical_axis
        experimental_method = data.get('experimental_method', "All")
        chart_type = data.get('chart_type', "bar_plot")
        chart_type = "point_plot" if (len(chart_type) == 0) else chart_type
        experimental_method = None if (experimental_method == "All" or experimental_method == "") else experimental_method
        data_frame = DataService.get_data_by_column_search_download("rcsentinfo_experimental_method", experimental_method)['data']
        cat_columns = reduce_value_length_version2([x for x in [
            categorical_axis, 'Group',
            'Subgroup',
            'Species',
            'Taxonomic Domain',
            'Expressed in Species' 
        ] if x is not None])
        columns = cat_columns + ([x_axis] if x_axis != "" else []) + ([y_axis] if y_axis != "" else []) 
        data_frame = data_frame[list(set(columns))]
        
        if categorical_axis is not None and ((x_axis is None or x_axis == "") or (y_axis is None or y_axis == "")):
            data_frame = data_frame.groupby([
                data_frame[categorical_axis]
            ]).size().reset_index(name='Value')
            _plot = Graph(data_frame, axis = [categorical_axis, "Value"], labels=categorical_axis)
            _plot = getattr(_plot, str(chart_type).replace(' ', '_'))()
            _plot.set_selection(type='single', groups=[categorical_axis])\
                .encoding(
                    tooltips = [categorical_axis, "Value"], 
                    encoding_tags = ["norminal", "quantitative"],
                    legend_columns=1
                )\
                .properties(width=0, title="Membrane Protein Structures categorized by " + categorical_axis.replace("rcsentinfo", " ").replace("_", " "))\
                .legend_config(orient='bottom')\
                .add_selection()\
                .interactive()
        
            # Convert the Altair chart to a dictionary
            chart_dict = _plot.return_dict_obj()
            
        else:
            label = data.get('categorical_axis', "")
            label = "" if (categorical_axis is None or len(categorical_axis) == 0) else categorical_axis
            
            _plot = Graph(data_frame, axis = [x_axis, y_axis], labels=label)
            _plot = getattr(_plot, str(chart_type).replace(' ', '_'))()
            _plot.set_selection(type='single', groups=[])\
                .encoding(
                    tooltips = columns, 
                    encoding_tags = ["quantitative", "quantitative"],
                    legend_columns=1,
                    axis_label = [x_axis, y_axis]
                )\
                .properties(width=0, title="Relationship between " + x_axis.replace("rcsentinfo", " ").replace("_", " ") + " and " + y_axis.replace("rcsentinfo", " ").replace("_", " "))\
                .legend_config(orient='bottom')\
                .add_selection()\
                .interactive()
        
            # Convert the Altair chart to a dictionary
            chart_dict = _plot.return_dict_obj()
            
        resp = {
            'chart': chart_dict
        }
        
        return ApiResponse.success(resp, "Fetch filter list successfully.")
               
class DataFilterResource(Resource):
    def __init__(self):
        pass
        
    def get(self):
        dimensionality_algorithms = dimensionality_reduction_algorithms_helper_kit()
        normalization_algorithms = normalization_algorithms_helper_kit()
        machine_learning_algorithms = machine_algorithms_helper_kit()
        imputation_algorithms = missing_algorithms_helper_kit()
        categorical_column = transform_data_view(
            cat_list, 'categorical_columns', 'single', [], False
        )
        methods = [
            "All", "EM", 
            "Multiple methods", 
            "NMR", "X-ray"
        ]
        experimental_method = transform_data_view(methods, 'experimental_method', 'single', [], False)
        excluded_fields = [
            "reflns", "refine", "rcsb_", "rcs", "ref", "diffrn", 
            "exptl", 
            "cell_", 
            "group_", "subgroup_", "species_", 
            "expressed_in_species", "pdb", "taxonomic_domain", 
            "symspa", "expcry", "em2cry", "software_"
        ]
        excluded_list = transform_data_view(excluded_fields, 'variable_groups', 'multiple', [], False)
        
        # child options
        
        eps = transform_data_view(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 
            'eps', 'single', ["dbscan_clustering"], False
        )
        min_samples = transform_data_view(
            [5, 10, 15, 20, 25, 30], 'min_samples', 
            'single', ["optics_clustering", "dbscan_clustering"], False
        )
        n_clusters = transform_data_view(
            range(2, 10), 'n_clusters', 
            'single', ["agglomerative_clustering", "kMeans_clustering"], False
        )
        n_components = transform_data_view(
            range(2, 10), 'n_components', 
            'single', ["gaussian_clustering"], False
        )
        # for agglomerative
        linkage = transform_data_view(
            ['ward', 'complete', 'average', 'single'], 
            'linkage', 'single', ["agglomerative_clustering"], False
        )
        metric = transform_data_view(
            ['euclidean'], 'metric', 
            'single', ["agglomerative_clustering"], False
        )
        distance_threshold = transform_data_view([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'distance_threshold', 
            'single', ["agglomerative_clustering"], False
        )
        
        #Dimensionality reduction methods change
        DR_n_components = transform_data_view(range(2, 3), 'n_components', 
            'single', ["tsne_algorithm", "umap_algorithm"], False
        )
        perplexity = transform_data_view([15, 20, 25, 30, 35, 40, 45, 50], 'perplexity', 
            'single', ["tsne_algorithm"], False
        )
        early_exaggeration = transform_data_view([10, 12, 20, 22, 30, 32, 40, 42], 'early_exaggeration', 
            'single', ["tsne_algorithm"], False
        )
        learning_rate = transform_data_view(['auto'], 'learning_rate', 
            'single', ["tsne_algorithm"], False
        )
        n_iter = transform_data_view([1000, 1500, 2000, 2500, 3000], 'n_iter', 
            'single', ["tsne_algorithm"], False
        )
        DR_metric = transform_data_view(['euclidean'], 'metric', 
            'single', ["tsne_algorithm", "umap_algorithm"], False
        )
        init = transform_data_view(['random', 'pca'], "init", 
            'single', ["tsne_algorithm"], False
        )
        method = transform_data_view(['barnes_hut', 'exact'], "method", 
            'single', ["tsne_algorithm"], False
        )
        angle = transform_data_view([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 'angle', 
            'single', ["tsne_algorithm"], False
        )
        # UMAP Param
        n_epochs = transform_data_view([10, 20, 30, 40, 50], 'n_epochs', 
            'single', ["umap_algorithm"], False
        )
        min_dist = transform_data_view([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 'min_dist', 
            'single', ["umap_algorithm"], False
        )
        n_neighbors = transform_data_view([15, 20, 25, 30, 35, 40, 45, 50], 'n_neighbors', 
            'single', ["umap_algorithm"], False
        )
            
        filter_list = {
            "experimental_method_list": {
                "data": experimental_method,
                "help": """
                    The experimental method refers to the technique or approach 
                    used to study membrane proteins in a laboratory setting
                    """
            },
            "categorical_list": {
                "data": categorical_column,
                "help": """
                    
                    """
            },
            "dimensionality_algorithms": {
                "data": dimensionality_algorithms,
                "help": """
                    Dimensionality reduction refers to the process of simplifying 
                    complex datasets while retaining essential information. 
                    In the context of membrane protein research, techniques 
                    like Principal Component Analysis (PCA) and t-Distributed 
                    Stochastic Neighbor Embedding (t-SNE) are employed. 
                    PCA identifies critical patterns, while t-SNE emphasizes 
                    local similarities. These methods aid in visualizing and 
                    interpreting high-dimensional data, enhancing the understanding
                    of intricate relationships within membrane protein datasets.
                    """,
                "child": {
                    # "early_exaggeration": early_exaggeration,
                    # "DR_n_components": DR_n_components,
                    # "learning_rate": learning_rate,
                    "n_neighbors": n_neighbors,
                    "perplexity": perplexity,
                    # "DR_metric": DR_metric,
                    "min_dist": min_dist,
                    # "n_epochs": n_epochs,
                    # "n_iter": n_iter,
                    # "method": method,
                    # "angle": angle,
                    # "init": init,
                    
                }
            },
            
            #"normalization_list": {
            #    "data": normalization_algorithms
            #},
            #"imputation_algorithms": {
            #    "data": imputation_algorithms
            #},
            #"excluded_list": {
            #    "data": excluded_list
            #},
            
            "machine_algorithms": {
                "data": machine_learning_algorithms,
                "help": """
                    Unsupervised Machine Learning is a subset of machine learning 
                    where the algorithm explores and identifies patterns in 
                    data without explicit guidance. In membrane protein research, 
                    unsupervised learning techniques, such as clustering and 
                    dimensionality reduction, play a vital role. Clustering 
                    algorithms group similar data points, revealing inherent 
                    structures, while dimensionality reduction methods simplify 
                    complex datasets. These unsupervised approaches contribute 
                    to uncovering hidden relationships and structures within 
                    membrane protein data, aiding researchers in gaining valuable 
                    insights for further analysis and interpretation.
                """,
                "child": {
                    "distance_threshold": distance_threshold,
                    "n_components": n_components,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    # "linkage": linkage,
                    # "metric": metric,
                    "eps": eps,
                }
            },
        }
        return ApiResponse.success(filter_list, "Fetch filter list successfully.")
                   
class UsupervisedResource(Resource):
    def post(self):
        data = request.get_json()
        if data and data != "":
            machine_learning_algorithms = data.get('machine_algorithms', "kMeans_clustering")
            imputation_algorithms = data.get('imputation_algorithms', "KNN_imputer_regressor")
            normalization_algorithms = data.get('normalization_algorithms', "min_max_normalization")
            dimensionality_algorithms = data.get('dimensionality_algorithms', "pca_algorithm")
            experimental_method = data.get('experimental_method_list', "All")
            color_by = data.get('categorical_list', "species")
            color_by = "rcsentinfo_experimental_method" if color_by == "exptl_method" else color_by
            excluded_fields = data.get('excluded_list', [])
            #ML Option
            distance_threshold = data.get('distance_threshold', None)
            if not distance_threshold or not isinstance(distance_threshold, int):
                distance_threshold = None
            n_components = data.get('n_components', 2)
            if not n_components or not isinstance(n_components, int):
                n_components = 2
            min_samples = data.get('min_samples', 15)
            if not min_samples or not isinstance(min_samples, int):
                method = 15
            n_clusters = data.get('n_clusters', 2)
            if not n_clusters or not isinstance(n_clusters, int):
                n_clusters = 2
            linkage = data.get('linkage', "ward")
            if not linkage or not linkage.strip():
                linkage = "ward"
            metric = data.get('metric', "euclidean")
            if not metric or not metric.strip():
                metric = "euclidean"
            eps = data.get('eps', 0.3)
            if not eps or not isinstance(eps, int):
                eps = 0.3
            #Dimensionality Reduction
            early_exaggeration = data.get('early_exaggeration', 12)
            if not early_exaggeration or not isinstance(early_exaggeration, int):
                early_exaggeration = 12
            DR_n_components = data.get('DR_n_components', 2)
            if not DR_n_components or not isinstance(DR_n_components, int):
                DR_n_components = 2
            learning_rate = data.get('learning_rate', "auto")
            if not learning_rate or not learning_rate.strip():
                learning_rate = "auto"
            perplexity = data.get('perplexity', 30)
            if not perplexity or not isinstance(perplexity, int):
                perplexity = 30
            DR_metric = data.get('DR_metric', "euclidean")
            if not DR_metric or not isinstance(DR_metric, int):
                DR_metric = "euclidean"
            n_iter = data.get('n_iter', 1000)
            if not n_iter or not isinstance(n_iter, int):
                n_iter = 1000
            method = data.get('method', "barnes_hut")
            if not method or not method.strip():
                method = "barnes_hut"
            angle = data.get('angle', 0.1)
            if not angle or not isinstance(angle, int):
                angle = 0.1
            init = data.get('init', "pca")
            if not init or not init.strip():
                init = "pca"
            #UMAP
            n_epochs=data.get('n_epochs', 10)
            if not n_epochs or not isinstance(n_epochs, int):
                n_epochs = 10
            min_dist=data.get('min_dist', 0.1)
            if not min_dist or not isinstance(min_dist, int):
                min_dist = 0.1
            n_neighbors=data.get('n_neighbors', 15)
            if not n_neighbors or not isinstance(n_neighbors, int):
                n_neighbors = 15
        else:
            machine_learning_algorithms = "kMeans_clustering"
            imputation_algorithms = "KNN_imputer_regressor"
            normalization_algorithms = "min_max_normalization"
            dimensionality_algorithms = "pca_algorithm"
            experimental_method = "All"
            excluded_fields = []
            color_by = "species"
            n_components = 2
            min_samples = 10
            distance_threshold = None
            n_clusters = 2
            linkage = "ward"
            metric = "euclidean"
            eps = 0.3
            
            #Dimensionality Reduction
            early_exaggeration =12
            DR_n_components = 2
            learning_rate =  "auto"
            perplexity = 50
            DR_metric = "euclidean"
            n_iter = 1000
            method = "barnes_hut"
            angle = 0.1
            init = "pca"
            
            # UMAP
            
            n_epochs=10,
            min_dist=0.1, 
            n_neighbors=15
        
        excluded_fields_list = [
            "reflns", "refine", "rcsb_", "rcs", "ref", "diffrn", 
            "exptl", "cell_", "group_", "subgroup_", "species_", 
            "expressed_in_species", "pdb", "taxonomic_domain", 
            "symspa", "expcry", "em2cry", "software_"
        ]
        if(len(excluded_fields_list) == len(excluded_fields)):
            return ApiResponse.error(
                message="Kindly un-select one of the variables so that the ML can fit on something", 
                status_code=400
            )
        
        experimental_method = None if experimental_method == "All" else experimental_method
        
        
        """
            We are adding this to the filter. Either to use categorical data or not.
            
        """
        # dimensionality reduction columns
        get_column_tag = dimensionality_algorithms.upper().split("_")[0]
        dr_columns = [ get_column_tag + str(char) for char in range(1, 3)]
        
        # Replace 'your_data_frame' with the actual variable holding your DataFrame
        data_frame = DataService.get_data_by_column_search_download("rcsentinfo_experimental_method", experimental_method)['data']
        data_frame = data_frame[get_quantification_data(experimental_method)]
        # print(get_quantification_data(experimental_method))
        # return list(data_frame.columns)
        # data_frame.to_csv("MergedDB.csv")
        result, dataset, accuracy_metrics = (
            UnsupervisedPipeline(data_frame)
                .modify_dataframe(excluded_fields)
                .select_numeric_columns()
                .apply_imputation(imputation_method=imputation_algorithms, remove_by_percent=90)
                .apply_normalization(normalization_method=normalization_algorithms)
                .apply_dimensionality_reduction(
                    reduction_method=dimensionality_algorithms, dr_columns=dr_columns,
                    early_exaggeration = early_exaggeration,
                    DR_n_components = DR_n_components,
                    learning_rate =  learning_rate,
                    perplexity = perplexity, DR_metric = DR_metric,
                    n_iter = n_iter, method = method, angle = angle, 
                    init = init, n_epochs=n_epochs, min_dist=min_dist, 
                    n_neighbors=n_neighbors
                )
                .apply_clustering(
                    distance_threshold = distance_threshold,
                    n_components = n_components,
                    min_samples = min_samples,
                    n_clusters = n_clusters, 
                    method = machine_learning_algorithms, 
                    linkage = linkage,
                    metric = metric,
                    eps = eps
                )
                .prepare_plot_DR(group_by = color_by)
        )
        
        # ML Plot 
        label="classes"
        result[label] = result['clustering'].apply(lambda x:  str(x) + "_Cluster")
        scatter_plot = Graph(result, axis = dr_columns, labels=label)\
            .scatter_plot()\
            .set_selection(type='single', groups=[label, color_by])\
            .encoding(
                tooltips = result.columns, 
                encoding_tags = ["quantitative", "quantitative"],
                legend_columns=1
            )\
            .properties(width=0, title="Unsupervised Machine Learning (Clustering)")\
            .legend_config()\
            .add_selection()\
            .interactive()
        
        # Convert the Altair chart to a dictionary
        chart_dict = scatter_plot.return_dict_obj()
        
        scatter_plot_DR = Graph(result, axis = dr_columns, labels=color_by)\
            .scatter_plot()\
            .set_selection(type='single', groups=[color_by])\
            .encoding(
                tooltips = result.columns, 
                encoding_tags = ["quantitative", "quantitative"],
                legend_columns=1
            )\
            .properties(width=0, title="Dimensionality Reduction using " + dimensionality_algorithms.replace("_", " ").title())\
            .legend_config()\
            .add_selection()\
            .interactive()
            
        # Convert the Altair chart to a dictionary
        chart_dict_DR = scatter_plot_DR.return_dict_obj()
        

        resp = {
            'dataset': list(dataset.columns),
            'data': result.to_dict(orient='records'), 
            'accuracy_metrics': accuracy_metrics,
            'DR_chart': chart_dict_DR,
            'chart': chart_dict
        }
        return ApiResponse.success(resp, "Fetch records successfully.")
      
class GroupSubGroupResource(Resource):
    def __init__(self) -> None:
        super().__init__()
        
    def get(self):
        all_data, _, _, _, _ = DataService.get_data_from_DB()
        data = report_and_clean_missing_values(all_data, threshold=40)

        columns_to_drop = [col for col in data.columns if '_citation_' in col or '_count_' in col or col.startswith('count_') or col.endswith('_count') or col.startswith('revision_') or col.endswith('_revision') or col.startswith('id_') or col.endswith('_id') or col == "id"]

        data.drop(columns_to_drop + [
            "bibliography_year", 
            "processed_resolution",
            "gibbs", "thicknesserror",
            "tilterror", "citation_country",
            "is_master_protein", "description",
            "rcsentinfo_experimental_method",
            "exptl_crystal_grow_method", "exptl_crystal_grow_method1",
            'species_description', 'famsupclasstype_type_name',
            'membrane_short_name', 'famsupclasstype_type_name',
            'species_name', 'family_superfamily_name',
            'family_superfamily_classtype_name', 'exptl_method',
            'pdbid', 'name_y', 'family_name_cache',
            'family_superfamily_name',
            'family_superfamily_classtype_name',
            'species_name_cache', 'membrane_name_cache',
            'expressed_in_species', 'rcsentinfo_nonpolymer_molecular_weight_maximum',
            'rcsentinfo_nonpolymer_molecular_weight_minimum',
        ], axis=1, inplace=True)
        data.columns.to_list()
        data = data.dropna()
        numerical_cols, categorical_cols = separate_numerical_categorical(data)
        numerical_data = data[numerical_cols]
        categorical_data = data[categorical_cols]

        new_d = pd.concat([numerical_data, categorical_data["subgroup"]], axis=1)
        numerical_data, top_features = select_features_using_decision_tree(new_d, target_column='subgroup', num_features=10)
        
        encode_data = categorical_data[[
            #"subgroup", 
            #"species", 
            #"taxonomic_domain", 
            # "rcsentinfo_experimental_method", 
            #"rcsentinfo_na_polymer_entity_types", 
            #"rcsentinfo_polymer_composition", 
            #"membrane_topology_in", "membrane_topology_out"
        ]]
        encoded_data = onehot_encoder(encode_data)
        complete_numerical_data = pd.concat([numerical_data, encoded_data], axis=1)
        complete_data = pd.concat([complete_numerical_data, categorical_data], axis=1)
        # Calculate the counts of occurrences of each group-subgroup pair
        group_subgroup_counts = complete_data.groupby(['group', 'subgroup']).size().reset_index(name='count')

        group_subgroup_counts_df = pd.DataFrame(group_subgroup_counts).sort_values(by='count', ascending=False)

        chart_type = request.args.get('chart_type', "subburst")
        chart_width = int(request.args.get('chart_width', 800))
        chart_height = int(request.args.get('chart_height', 500))

        new_frame = pd.merge(right=complete_data, left=group_subgroup_counts[['subgroup', 'count']], on="subgroup")
        
        if chart_type == "sunburst":
            # Create a sunburst chart
            fig = px.sunburst(new_frame, path=[
                'group', 'subgroup', 
                'count', 'pdb_code'
            ], values='count', 
            color='count', color_continuous_scale='blues', 
            width=chart_width, height=chart_height
            )
            fig.update_layout(title_text='Sunburst Chart of Membrane Protein Sub-groups.')
            fig_json = pio.to_json(fig)
        else:
            fig = px.treemap(new_frame, path=[
                'group', 'subgroup', 
                'count', 'pdb_code'
                ], values='count',
                color='group', hover_data=[
                    'group', 'subgroup', 
                    'count', 'pdb_code'
                ],
                color_discrete_map={
                    'MONOTOPIC MEMBRANE PROTEINS':'#2ca25f', 
                    'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL':'#fdc086', 
                    'TRANSMEMBRANE PROTEINS:BETA-BARREL':'#beaed4'
                },
                color_continuous_midpoint=np.average(
                    new_frame['count'], 
                    weights=new_frame['count']
                ),
                width=chart_width, height=chart_height
                )
            fig.update_traces(root_color="lightgrey", )
            fig.update_layout(
                title_text='Sunburst Chart of Membrane Protein Sub-groups.', 
                margin = dict(t=50, l=25, r=25, b=25)
            )
            fig_json = pio.to_json(fig)
            
        return ApiResponse.success(fig_json, "Fetch records successfully.") 
      
class MLPrediction(Resource):
    def __init__(self):
        pass
    
    def get(self):
        # select from model list 
        directory_path = './models/semi-supervised'
        model_names, dim_reductions = get_joblib_files_and_splits(directory_path)
        data = {
            "models": list(set(model_names)),
            "dim_reductions": list(set(dim_reductions))
        }
        return ApiResponse.success(data, "Fetch records successfully.")

class MLPredictionAccuracy(Resource):
    def __init__(self):
        pass
    
    def get(self):
        dim_reduction = request.args.get('dim_reduction', "pca")
        # Convert DataFrame to CSV
        accuracy = pd.read_csv("./models/semi-supervised/metrics" + dim_reduction + ".csv")
        data = {
            "data": accuracy.to_dict(orient="records")
        }

        return ApiResponse.success(
            data, 
            "Fetch records successfully."
        )
         
class MLDimensionalityReductionCharts(Resource):
    def __init__(self):
        pass
    
    def create_chart(self, data):
        return alt.Chart(data).mark_circle().encode(
            x='Component 1',
            y='Component 2',
            color='group',
            tooltip=['Method', 'Parameter', "group"]
        ).properties(
            width="container",
            height=600
        ).interactive().configure_legend(orient='bottom', direction = 'vertical', labelLimit=0).to_dict(format="vega")
    
    def get(self):
        pca = pd.read_csv("./models/semi-supervised/PCA_data.csv")
        t_sne = pd.read_csv("./models/semi-supervised/TSNE_data.csv")
        umap = pd.read_csv("./models/semi-supervised/UMAP_data.csv")
        
        pca_chart = self.create_chart(pca)
        t_sne_chart = self.create_chart(t_sne)
        umap_chart = self.create_chart(umap)
        # plot charts for each and color them based on groups
        
        data = {
            "pca_chart": pca_chart,
            "t_sne_chart": t_sne_chart,
            "umap_chart": umap_chart
        }

        return ApiResponse.success(
            data, 
            "Fetch records successfully."
        )

import requests        
class GenerateRealSampleDataTest(Resource):

    def __init__(self):
        self.headers = {
            "authority": "opm-back.cc.lehigh.edu:3000",
            "method": "GET",
            "scheme": "https",
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Origin": "https://opm.phar.umich.edu",
            "Referer": "https://opm.phar.umich.edu/",
            "Sec-Ch-Ua": '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "cross-site",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        }
        self.host = "https://opm-back.cc.lehigh.edu/opm-backend/"
    
    def fetch_records_for_proteins(self, pdb_codes):
        all_dfs = []  # List to store dataframes for each PDB code

        for pdb_code in pdb_codes:
            url = self.host + "primary_structures?search=" + pdb_code + "&sort=&pageSize=100"
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                record = data.get("objects", [])
                if record:
                    filter_data = record[0]
                    url2 = self.host + "primary_structures/" + str(filter_data.get("id"))
                    response_filtered = requests.get(url2, headers=self.headers)

                    if response_filtered.status_code == 200:
                        data_filtered = response_filtered.json()
                        df_filtered = pd.json_normalize(data_filtered, sep="_")
                        df_filtered['pdb_code'] = pdb_code  # Add a new column for PDB code
                        all_dfs.append(df_filtered)  # Append dataframe to list

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combine_data = combined_df[[
                "id", "pdbid", "name", "resolution",
                "topology_subunit", "thickness",
                "subunit_segments", "tilt", "gibbs",
                "membrane_topology_in", "membrane_topology_out", 
                "pdb_code"
            ]]
            return combine_data
        else:
            return None
                 
    def post(self):
        try:
            data = request.form
            
        except Exception as e:
            return ApiResponse.error("Validation Error", 400, "Invalid JSON data" + str(e))
        
        pdb_codes_unclean = data.get('pdb_codes', '')  # Replace with your list of PDB codes
        pdb_codes = [code.strip() for code in pdb_codes_unclean.split(",") if code.strip()]  # Remove empty strings and strip whitespace

        if len(pdb_codes) > 20:
            return ApiResponse.error("Validation Error", 400, "Too many PDB codes (maximum 20 allowed)")
        
        result_df = self.fetch_records_for_proteins(pdb_codes)
        if result_df is not None:
            output = io.BytesIO()
            result_df.to_csv(output, index=False)
            output.seek(0)
            return send_file(output, mimetype='text/csv', download_name='real_examples.csv', as_attachment=True)
        else:
            return ApiResponse.error("Validation Error", 404, "No data found for the provided PDB codes")
            
            
import os

class MLPredictionPost(Resource):    
    def post(self):
        if 'data_file' not in request.files:
            return ApiResponse.error("Validation Error", 400, "No file part")

        file = request.files['data_file']

        if file.filename == '':
            return ApiResponse.error("Validation Error", 400, "No selected file")

        if file and file.filename.endswith('.csv'):
            # Read the CSV file into a pandas DataFrame
            stream = StringIO(file.stream.read().decode("UTF8"), newline=None)
            filename_data = pd.read_csv(stream)
            
            expected_columns = [
                'pdb_code', 'subunit_segments', 'thickness', 
                'tilt', 'membrane_topology_in', 'membrane_topology_out'
            ]  
            
            if not set(expected_columns).issubset(filename_data.columns):
                return ApiResponse.error("Validation Error", 400, "Missing or incorrect column names")
            
            # import the original old data
            read_path = pd.read_csv("./models/semi-supervised/without_reduction_data.csv")
            data_ = pd.concat([read_path, filename_data[[
                'pdb_code', 'subunit_segments', 'thickness', 
                'tilt', 'membrane_topology_in', 'membrane_topology_out'
            ]]], axis=0)
            # Numerical columns
            filename_data_ = data_[[ 'subunit_segments', 'thickness', 'tilt']]
            
            """
                Onehot encoding
            """
            encode_data = data_[[
                "membrane_topology_in", 
                "membrane_topology_out"
            ]]
            encoded_data = onehot_encoder(encode_data)
            complete_numerical_data = pd.concat([filename_data_, encoded_data], axis=1)
            
            # Implementation for dimensionality reduction 
            perplexity_count = 50 if len(filename_data) > 50 else len(filename_data) - 1
            methods_params = {
                'PCA': {'n_components': 2},
                't-SNE': {'n_components': 2, 'perplexity': 30},
                'UMAP': {'n_components': 2, 'n_neighbors': 15}
            }
            
            if request.form.get("dim_reduction", "PCA").upper() == "TSNE":
                key = "t-SNE"
            else:
                key = request.form.get("dim_reduction", "PCA").upper()

            params = {
                key: methods_params.get(key)
            }
            
            _, plot_data = evaluate_dimensionality_reduction(complete_numerical_data, params)
            
            combined_plot_data = pd.concat(plot_data)
            dm_data = combined_plot_data[combined_plot_data["Method"] == key].reset_index(drop=True)
            
            label = data_["pdb_code"].reset_index(drop=True)
            complete_merge = pd.concat([dm_data, label], axis=1)
            complete_merge = complete_merge[complete_merge["pdb_code"].isin(filename_data["pdb_code"].to_list())]
            complete_merge.reset_index()
            
            # Load the saved model
            model_name = request.form.get("model")
            dim_reduction = request.form.get("dim_reduction")
            model_path = f'./models/semi-supervised/{model_name}__{dim_reduction}.joblib'
            first_model_search = './models/semi-supervised/Random Forest__semi_supervised_tsne_0.joblib'
            # Check if the file exists
            if os.path.exists(first_model_search):
                model = joblib.load(first_model_search)
                print(f"Loaded model from {first_model_search}")
            else:
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    print(f"Loaded alternative model from {model_path}")
                else:
                    return ApiResponse.error("File Not Found", 404, f"Neither {model_path} nor {first_model_search} exists.")
            
            # Make predictions using the imputed DataFrame
            predictions = model.predict(complete_merge[["Component 1", "Component 2"]])
            filename_data["predicted_class"] = predictions
            
            # Convert DataFrame to CSV
            output = io.BytesIO()
            filename_data.to_csv(output, index=False)
            output.seek(0)

            return send_file(output, mimetype='text/csv', download_name='predictions.csv', as_attachment=True)
        else:
            return ApiResponse.error("Validation Error", 400, "File must be a CSV file")