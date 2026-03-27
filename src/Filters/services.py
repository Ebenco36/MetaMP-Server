from src.services.Helpers.fields_helper import (
    PCAComponentsOption,
    dataSplitPercOption,
    date_grouping_methods,
    dimensionality_reduction_algorithms_helper_kit,
    graph_combined_types_kit,
    graph_group_by_date,
    graph_group_by_others,
    graph_options,
    graph_selection_categories_UI_kit,
    graph_types_kit,
    grouping_aggregation_methods,
    machine_algorithms_helper_kit,
    merge_graph_into_one_kit,
    missing_algorithms_helper_kit,
    ml_slider_selector,
    multi_select_kit,
    normalization_algorithms_helper_kit,
    perc_of_missing_value_kit,
    quantification_fields_kit,
    test_or_train_kit,
)


class FilterKitService:
    @staticmethod
    def get_filters_payload():
        selection_avenue_default, selection_type_default = graph_options()
        (
            eps_slider,
            pca_features_slider,
            min_samples_slider,
            n_clusters_slider,
            n_components_slider,
        ) = ml_slider_selector()

        return {
            "selection_avenue_default": selection_avenue_default,
            "selection_type_default": selection_type_default,
            "graph_types_kit": graph_types_kit(),
            "machine_algorithms_helper_kit": machine_algorithms_helper_kit(),
            "dimensionality_reduction_algorithms_helper_kit": (
                dimensionality_reduction_algorithms_helper_kit()
            ),
            "normalization_algorithms_helper_kit": normalization_algorithms_helper_kit(),
            "graph_selection_categories_UI_kit": graph_selection_categories_UI_kit(),
            "graph_group_by_date": graph_group_by_date(),
            "graph_group_by_others": graph_group_by_others(),
            "date_grouping_methods": date_grouping_methods(),
            "grouping_aggregation_methods": grouping_aggregation_methods(),
            "multi_select_kit": multi_select_kit(),
            "quantification_fields_kit": quantification_fields_kit(),
            "merge_graph_into_one_kit": merge_graph_into_one_kit(),
            "eps_kit": eps_slider,
            "pca_feature_kit": pca_features_slider,
            "minimum_samples_kit": min_samples_slider,
            "clusters_kit": n_clusters_slider,
            "components_kit": n_components_slider,
        }

    @staticmethod
    def get_missing_filter_options():
        return missing_algorithms_helper_kit()

    @staticmethod
    def get_missing_percentage_options():
        return perc_of_missing_value_kit()

    @staticmethod
    def get_normalization_options():
        return normalization_algorithms_helper_kit()

    @staticmethod
    def get_dimensionality_reduction_options():
        return dimensionality_reduction_algorithms_helper_kit()

    @staticmethod
    def get_data_split_options():
        return dataSplitPercOption()

    @staticmethod
    def get_pca_component_options(n_features):
        return PCAComponentsOption(n_features)

    @staticmethod
    def get_machine_learning_options():
        return machine_algorithms_helper_kit()

    @staticmethod
    def get_graph_options():
        return graph_combined_types_kit()

    @staticmethod
    def get_train_and_test_split_options():
        return test_or_train_kit()


class FilterQueryService:
    @staticmethod
    def parse_feature_count(raw_value):
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = 2

        return max(value, 1)
