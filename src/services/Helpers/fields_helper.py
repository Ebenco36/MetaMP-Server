import numpy as np
from src.services.Helpers.helper import (
    extract_function_names, 
    get_class_functions, 
    replace_and_separate,
    filter_list, 
    format_string_caps
)
from src.services.graphs.helpers import Graph
from src.services.data.columns.dates import dates_columns
from src.services.Helpers.Regressors.index import Regressors
from src.services.data.columns.norminal import all_descriptors
from src.services.Helpers.machine_learning_al.normalization import Normalization
from src.services.data.columns.quantitative.quantitative import cell_columns, rcsb_entries
from src.services.data.columns.quantitative.quantitative_array import quantitative_array_column
from src.services.Helpers.machine_learning_al.UnsupervisedMachineLearning import MachineLearning
from src.services.Helpers.machine_learning_al.dimensionality_reduction import DimensionalityReduction

def graph_options(kit_id:int = 0):
    # default selections for graphs
    selection_avenue_default = ["click", "drag"]
    selection_type_default = ["multiple", "single", "interval"]

    selection_avenue_default = transform_data_view(selection_avenue_default, 'selection_avenue', 'single', [], False)
    selection_type_default = transform_data_view(selection_type_default, 'selection_type', 'single', [], False)

    return selection_avenue_default, selection_type_default


def graph_types_kit(kit_id:int=0):
    # Graph Dimension
    graph_dimensions = filter_list(get_class_functions(Graph()), "_plot")
    graph_types = transform_data_view(graph_dimensions, 'Chart_options', 'single')
    return graph_types

def graph_combined_types_kit(kit_id:int=0):
    # Graph Dimension
    file_path_graph = 'src\services\graphs\helpers.py'
    graph_dimensions = filter_list(extract_function_names(file_path_graph), "_plot")

    graph_types = transform_data_view(graph_dimensions, 'Combine_chart_options', 'single', [], False)

    return graph_types


def machine_algorithms_helper_kit(kit_id:int=0):
    # ML algorithms
    ml_algorithms = filter_list(get_class_functions(MachineLearning()), "_clustering")

    ML_types = transform_data_view(ml_algorithms, 'Machine_learning_algorithms', 'single')

    return ML_types


def missing_algorithms_helper_kit(kit_id:int=0):
    # ML algorithms
    missing_algorithms = filter_list(get_class_functions(Regressors()), "_regressor")

    Missing_types = transform_data_view(missing_algorithms, 'Data_augmentation', 'single', [], False)

    return Missing_types


def perc_of_missing_value_kit(kit_id:int = 0):
    # allowed percentage of missing values
    perc = [str(i) for i in range(10, 100, 10)]

    perc_missing_value = transform_data_view(perc, 'maintain_missing_values', 'single', [], False)

    return perc_missing_value


def dimensionality_reduction_algorithms_helper_kit(kit_id:int=0):
    # DR algorithms
    dr_algorithms = filter_list(get_class_functions(DimensionalityReduction()), "_algorithm")
    
    DR_types = transform_data_view(dr_algorithms, 'Dimensionality_reduction_techniques', 'single', [], False)

    return DR_types


def normalization_algorithms_helper_kit(kit_id:int=0):
    # Normalization algorithms
    normal_algorithms = filter_list(get_class_functions(Normalization()), "_normalization")

    normal_types = transform_data_view(normal_algorithms, 'Normalization_algorithms', 'single', [], False)

    return normal_types



def graph_selection_categories_UI_kit(kit_id:int=0):
    options=["Date Fields", "Descriptions"]

    graph_selection_categories = transform_data_view(options, 'Chart_selection_category_options', 'single', [], False)

    return graph_selection_categories



def graph_group_by_date(kit_id:int = 0):
    dates_data = dates_columns
    # replace dot with underscore
    dates_data = tuple([str(s).replace('.', '_') for s in dates_data])

    dates_types = transform_data_view(dates_data, 'Date_options', 'single', [], False)

    return dates_types


def graph_group_by_others(kit_id:int = 0):
    desc = all_descriptors
    # replace dot with underscore
    desc = tuple([str(s).replace('.', '_') for s in desc])

    graph_group_by = transform_data_view(desc, 'graph_group_by', 'single', [], False)

    return graph_group_by


def date_grouping_methods(kit_id:int = 0):
    classes_options=["By Year", "By Month"]

    date_group_by = transform_data_view(classes_options, 'Date_category_options', 'single', [], False)
    return date_group_by

def grouping_aggregation_methods(kit_id:int = 0):
    classes_options={ 
        "mean": "Mean", 
        "median": "Median", 
        "sum": "Sum", 
        "min": "Minimum", 
        "max": "Maximum ", 
        "count": "Count",
        "std": "Standard Deviation",
        "var": "Variance"
    }
    
    aggregate_options = {
        "options": [{"value": key, "name": format_string_caps(classes_options[key])} for key in classes_options],
        "model_name": [],
        "safe_name" : "name",
        "field_name": format_string_caps("aggregate_options"),
        "multiple": False,
        "parents": [], 
        "show_option": False
    }
    return aggregate_options


def multi_select_kit(kit_id:int=0):
    select_options = ["single select", "multiple select"]
    select_options_ = transform_data_view(select_options, 'multiple_select_options', 'single', [], False)

    return select_options_

def quantification_fields_kit(kit_id:int=0, selection_type:str = "single select"):
    # Quantifications
    quantitative_data = cell_columns+rcsb_entries+quantitative_array_column

    quantitative_attributes = transform_data_view(quantitative_data, 'attribute_options', 'multiple', [], False)

    return quantitative_attributes

def merge_graph_into_one_kit(kit_id:int=0):
    merge_graph_options = ["no", "yes"]
    merge_graph_options = transform_data_view(merge_graph_options, 'merge_graph_option', 'single', [], False)
    return merge_graph_options


def test_or_train_kit(kit_id:int=0):
    options = ["training", "testing"]
    options = transform_data_view(options, 'train_or_test', 'single', [], False)
    return options

def ml_slider_selector():
    """
        Select these option based on the ML selected
    """
    option_list = [str(i) for i in range(1, 11)]

    eps_slider = transform_data_view([str(round(i, 2)) for i in np.arange(0.1, 10.0, 0.1)], "eps (epsilon)", "single", [
        "dbscan_clustering"
    ], False)

    pca_features_slider = transform_data_view(option_list, "pca_options", "single", [], False)

    min_samples_slider = transform_data_view(option_list, "min_samples_options", "single", [
        "dbscan_clustering", "optics_clustering"
    ], False)

    n_clusters_slider = transform_data_view(option_list, "number_of_clusters", "single", [
        'agglomerative_clustering', 'kMeans_clustering'
    ], False)

    n_components_slider = transform_data_view(option_list, "number_of_components", "single", [
        'gaussian_clustering'
    ], False)

    # date_group_by = transform_data_view(classes_options, 'date_group_by')

    return eps_slider, pca_features_slider, min_samples_slider, n_clusters_slider, n_components_slider



def dataSplitPercOption():
    start = 0.1
    end = 0.5
    step = 0.1  # Adjust the step size based on your preference
    numbers = [str(round(start + i * step, 1)) for i in range(int((end - start) / step) + 1)]
    data = transform_data_view(numbers, 'train_and_test_split_options', 'single', [], False)

    return data

def PCAComponentsOption(n_features = 2):
    data = [str(i) for i in range(2, int(n_features)+1)]
    data = transform_data_view(data, 'number_of_PCA_components', 'single', [], False)

    return data


def transform_data_view(data, unique_field_name, multiple_selection, parents:list = [], status = True):
    data_object = {
        "options": [{
            "value": value, 
            "name": (
                    format_string_caps(
                        replace_and_separate(value)
                    ).replace("Rcsentinfo", " ").replace("rcsentinfo", " ").replace("_", " ").title()
                ) if (isinstance(value, str)) else value
            } for value in data
        ],
        "model_name": [],
        "safe_name" : "name",
        "field_name": format_string_caps(unique_field_name),
        "multiple": True if multiple_selection == "multiple" else False,
        "tooltip": "",
        "parents": parents,
        "show_option": status
    }

    return data_object



def transform_data_dict_view(data, unique_field_name, multiple_selection, parents: list = [], status: bool = True):
    """
    Transforms data view from a list of dictionaries into the expected format.
    
    Args:
    - data (list): List of dictionaries from which a value will be extracted.
    - unique_field_name (str): Field name for formatting.
    - multiple_selection (str): Whether the field is multiple selection.
    - parents (list, optional): Parent options.
    - status (bool, optional): Whether to show the option.
    - value_key (str, optional): The key to extract the value from each dict.

    Returns:
    - dict: Transformed data object.
    """
    data_object = {
        "options": [{
            "value": item.get("value"), 
            "name": (
                format_string_caps(
                    replace_and_separate(item.get("text"))
                ).replace("Rcsentinfo", " ").replace("rcsentinfo", " ").replace("_", " ").title()
            ) if isinstance(item.get("text"), str) else item.get("value")
        } for item in data],
        "model_name": [],
        "safe_name": "name",
        "field_name": format_string_caps(unique_field_name),
        "multiple": multiple_selection == "multiple",
        "tooltip": "",
        "parents": parents,
        "show_option": status
    }

    return data_object