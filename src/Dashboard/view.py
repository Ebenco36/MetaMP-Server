import io
import json
import pandas as pd
import altair as alt
from scipy import stats
from http import HTTPStatus
from datetime import timedelta
from flask_restful import Resource
from src.MP.services import DataService
from src.Training.services import (
    aggregate_inconsistencies, 
    create_visualization, 
    getChartForQuestion,
    group_annotation, 
    transform_dataframe
)
from src.services.graphs.helpers import convert_chart
from src.services.pages import Pages
from src.Dashboard.data import stats_data
from src.utils.response import ApiResponse
from src.utils.common import generate_response
from flask import jsonify, request, current_app
from src.services.Helpers.helper import get_data_by_countries, tableHeader
from src.middlewares.auth_middleware import token_required
from src.services.basic_plots import create_combined_chart_cumulative_growth, group_data_by_methods, data_flow
from src.services.Helpers.helper import find_dict_with_value_in_nested_data
from src.services.Helpers.helper import (
    find_dicts_with_value_not_equal,  
    summaryStatisticsConverter,
    getPercentage,
)
from src.Dashboard.services import (
    biological_process_filter_options, 
    cellular_component_filter_options,
    convert_to_list_of_dicts,
    create_grouped_bar_chart, 
    experimental_methods_filter_options,
    extract_widths,
    family_name_filter_options, get_items, 
    get_table_as_dataframe,
    get_table_as_dataframe_download, getMPstructDB, 
    getOPMDB, getPDBDB, getUniprotDB, 
    group_filter_options, membrane_name_filter_options, 
    molecular_function_filter_options, preprocessVariables, 
    search_merged_databases, species_filter_options, 
    subgroup_filter_options, super_family_class_type_filter_options, 
    super_family_filter_options, taxonomic_domain_filter_options,
    get_columns_by_pdb_codes
)
from utils.redisCache import RedisCache


class WelcomePage(Resource):
    def __init__(self):
        self.cache = RedisCache()
    # @staticmethod
    # @token_required
    def get(self):
        # Create a unique cache key based on request parameters
        cache_key = f"get:welcome-page"

        # Check if the result is in the cache
        cached_result = self.cache.get_item(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        dataset = getChartForQuestion(column="group", filter="")
        view_data = [
            'resolution', 'bibliography_year', 
            'group', 'rcsentinfo_molecular_weight', 
            "pdb_code", "refine_ls_rfactor_rfree", 
            "rcsentinfo_experimental_method", "taxonomic_domain"
        ]
        
        # chart_data = dataset[view_data]
        
        #########################Chart 1########################
        # Grouping data by 'exptl_method' and counting occurrences
        # variable_counts = chart_data["group"].value_counts().reset_index()
        # variable_counts.columns = ["group", 'Cumulative MP Structures']

        # chart = alt.Chart(variable_counts).mark_bar().encode(
        #     x=alt.X(
        #         "group:N", 
        #         axis=alt.Axis(
        #             labelAngle=30, 
        #             title="group".title(),
        #             labelLimit=0
        #         )
        #     ),
        #     y=alt.Y(
        #         'Cumulative MP Structures:Q', 
        #         axis=alt.Axis(
        #             title='Cumulative MP Structures'
        #         )
        #     ),
        #     color=alt.Color("group:N", legend=alt.Legend(title="group".title())),
        #     tooltip=["group:N", 'Cumulative MP Structures:Q']
        # ).properties(
        #     width="container",
        #     title="Membrane Protein Structures by Group"
        # ).interactive().configure_legend(
        #     orient='bottom', 
        #     direction = 'vertical', 
        #     labelLimit=0
        # )
        
        #########################Chart 2########################
        
        # Grouping data by 'rcsb_entry_info_experimental_method' and 'Group' and counting occurrences
        # group_method_year_counts = chart_data.groupby(["rcsentinfo_experimental_method", 'bibliography_year']).size().reset_index(name='count')
        # if("rcsentinfo_experimental_method" in group_method_year_counts):
        #     group_method_year_counts['rcsentinfo_experimental_method'] = group_method_year_counts['rcsentinfo_experimental_method'].replace({
        #         'EM': 'Cryo-Electron Microscopy (Cryo-EM)',
        #         'X-ray': 'X-Ray Crystallography',
        #         'NMR': 'Nuclear Magnetic Resonance (NMR)',
        #         'Multiple methods': 'Multi-methods',
        #     }) 
        #     group_method_year_counts = group_method_year_counts.rename(columns={
        #         'rcsentinfo_experimental_method': 'Experimental Method',
        #     })
            
        # chart_method = alt.Chart.from_dict(
        #     group_data_by_methods(
        #         chart_data, 
        #         columns = [
        #             'bibliography_year', 
        #             'rcsentinfo_experimental_method'
        #         ], 
        #         col_color='rcsentinfo_experimental_method', 
        #         chart_type="line",
        #         bin_value=None,
        #         interactive=True,
        #         arange_legend="vertical"
        #     )
        # )
        
        #########################Chart 3########################
        # if not dataset.empty:
        #     # Convert 'Resolution' column to numeric, and filter out non-numeric values
        #     dataset.loc[:, 'resolution'] = pd.to_numeric(dataset['resolution'], errors='coerce')
        #     dataset = dataset.dropna(subset=['resolution'])

        #     # Handle non-positive values before applying logarithmic scale
        #     dataset['resolution'] = dataset['resolution'].apply(lambda x: max(x, 1e-10))

        #     # Calculate group-specific median of 'Resolution'
        #     group_median_resolution = dataset.groupby('group')['resolution'].median().reset_index()
        #     group_median_resolution.columns = ['group', 'Group_Median_Resolution']

        #     # Merge the median values back into the main dataframe
        #     dataset = pd.merge(dataset, group_median_resolution, on='group')

        #     # Calculate group-specific z-scores
        #     dataset['Resolution_Z'] = dataset.groupby('group')['resolution'].transform(lambda x: stats.zscore(x))

        #     # Identify and filter potential outliers based on a z-score threshold and group-specific median resolution
        #     outliers = dataset[(abs(dataset['Resolution_Z']) > dataset['Group_Median_Resolution']) & (dataset['resolution'] > dataset['Group_Median_Resolution'])]
            
        #     # Altair boxplot with logarithmic scale
        #     dataset['group'] = dataset['group'].replace({
        #         'MONOTOPIC MEMBRANE PROTEINS': 1,
        #         'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL': 2,
        #         'TRANSMEMBRANE PROTEINS:BETA-BARREL': 3,
        #     })
        #     chart_obj = alt.Chart(dataset).mark_boxplot().encode(
        #         x=alt.X('group:N', title="Group", axis=alt.Axis(labelAngle=360)),
        #         y=alt.Y('resolution:Q', title="Resolution (Angstrom (Å))", scale=alt.Scale(type='log')),
        #         color=alt.value("#005EB8"),
        #         tooltip=['group:N', 'resolution:Q', 'pdb_code:N']
        #     ).properties(
        #         width="container",
        #         title='Boxplot of Resolution for all within Each Group (Log Scale)'
        #     )
            
        #     chart_outlier = group_annotation(chart_obj)
        # else:
        #     chart_outlier = {}
        
        #########################Chart 4########################
        # If not in cache, proceed with data retrieval and processing
        table_df = get_table_as_dataframe("membrane_proteins")
        # get_master_proteins = DataService.get_data_by_column_search(
        #     column_name="is_master_protein", 
        #     value="MasterProtein", 
        #     page=1, per_page=1
        # )
        
        all_data = DataService.get_data_by_column_search(
            column_name=None, 
            value=None, 
            page=1, 
            per_page=1
        )
        
        all_data_mpstruc = DataService.get_data_by_column_search(
            table="membrane_protein_mpstruct",
            column_name=None, 
            value=None, 
            page=1, 
            per_page=1
        )
        all_data_pdb = DataService.get_data_by_column_search(
            table="membrane_protein_pdb",
            column_name=None, 
            value=None, 
            page=1, 
            per_page=1
        )
        all_data_opm = DataService.get_data_by_column_search(
            table="membrane_protein_opm",
            column_name=None, 
            value=None, 
            page=1, 
            per_page=1
        )
        all_data_uniprot = DataService.get_data_by_column_search(
            table="membrane_protein_uniprot",
            column_name=None, 
            value=None, 
            page=1, 
            per_page=1,
            distinct_column="pdb_code"
        )
        
        
        # Get the DataFrame directly
        # pages = Pages(table_df)
        # _, map = pages.getMap()
        # trend = data_flow(table_df)
        
        unique_data = get_table_as_dataframe_download(
            table_name="membrane_proteins",
            filter_column="is_master_protein", 
            filter_value="MasterProtein"
        ).get("data", [{}])
        unique_trend = data_flow(unique_data, "Unique membrane proteins from MPstruc")
        from datetime import datetime
        latest_datetime = max(
            datetime.strptime(item['updated_at_readable'], "%Y-%m-%d %H:%M:%S")
            for item in all_data.get("data", {}) if 'updated_at_readable' in item
        )
        # Convert to string
        latest_date_str = latest_datetime.strftime("%Y-%m-%d %H:%M:%S")
        result = {
            # "trend": trend,
            # "map_chart": map,
            "all_data": all_data,
            "latest_date": latest_date_str,
            # "group_chart": convert_chart(chart),
            # "method_chart": convert_chart(chart_method),
            # "outlier_chart": convert_chart(chart_outlier),
            # 'get_master_proteins': get_master_proteins,
            "all_data_uniprot": all_data_uniprot,
            "all_data_opm": all_data_opm,
            "all_data_mpstruc": all_data_mpstruc,
            "all_data_pdb": all_data_pdb,
            "unique_trend": unique_trend
        }

        # Store the result in the cache
        ttl_in_seconds = timedelta(days=10).total_seconds()
        self.cache.set_item(cache_key, result, ttl=ttl_in_seconds)  # Cache for 10 days

        return jsonify(result)

class AboutMetaMP(Resource):
    def __init__(self):
        self.cache = RedisCache()

    # @staticmethod
    @token_required
    def get(self):
        # Create a unique cache key based on request parameters
        cache_key = f"get:aboutMetaMP"

        # Check if the result is in the cache
        cached_result = self.cache.get_item(cache_key)
        ########################Check width if they are the same ###########################
        
        if cached_result:
            return jsonify(cached_result)

        # If not in cache, proceed with data retrieval and processing
        table_df = get_table_as_dataframe("membrane_proteins")
        # Get the DataFrame directly
        pages = Pages(table_df)
        trends_by_database_year = pages.view_trends_by_database_year_default()
        
        get_master_proteins = DataService.get_data_by_column_search(
            column_name="is_master_protein", 
            value="MasterProtein", 
            page=1, per_page=1
        )
        
        all_data = DataService.get_data_by_column_search(
            column_name=None, 
            value=None, 
            page=1, 
            per_page=1
        )
        
        all_data_mpstruc = DataService.get_data_by_column_search(
            table="membrane_protein_mpstruct",
            column_name=None, 
            value=None, 
            page=1, 
            per_page=1
        )
        all_data_pdb = DataService.get_data_by_column_search(
            table="membrane_protein_pdb",
            column_name=None, 
            value=None, 
            page=1, 
            per_page=1
        )
        all_data_opm = DataService.get_data_by_column_search(
            table="membrane_protein_opm",
            column_name=None, 
            value=None, 
            page=1, 
            per_page=1
        )
        all_data_uniprot = DataService.get_data_by_column_search(
            table="membrane_protein_uniprot",
            column_name=None, 
            value=None, 
            page=1, 
            per_page=1,
            distinct_column="pdb_code"
        )
        
        result = {
            'trends_by_database_year': trends_by_database_year,
            'get_master_proteins': get_master_proteins,
            "all_data_uniprot": all_data_uniprot,
            "all_data_mpstruc": all_data_mpstruc,
            "all_data_opm": all_data_opm,
            "all_data_pdb": all_data_pdb,
            "all_data": all_data,
        }
        # Store the result in the cache
        ttl_in_seconds = timedelta(days=10).total_seconds()
        self.cache.set_item(cache_key, result, ttl=ttl_in_seconds)  # Cache for 10 days

        return jsonify(result)
   
class Dashboard(Resource):
    def __init__(self):
        self.cache = RedisCache()

    # @staticmethod
    @token_required
    def get(self):
        get_header = request.args.get('get_header', default="none", type=str)
        conf = request.args.get('chart_conf', '{"color": "#005EB8", "opacity": 0.9}')
        request_for_group = request.args.get('group_key', 'taxonomic_domain')
        first_leveled_width = request.args.get('first_leveled_width', 800)
        # Create a unique cache key based on request parameters
        cache_key = f"get:{get_header}:{conf}:{request_for_group}"

        # Check if the result is in the cache
        cached_result = self.cache.get_item(cache_key)
        ########################Check width if they are the same ###########################
        
        if cached_result:
            # print(cached_result.get("trend", {}))
            if extract_widths(cached_result.get("trend", {}), int(first_leveled_width)):
                return jsonify(cached_result)

        # If not in cache, proceed with data retrieval and processing
        table_df = get_table_as_dataframe("membrane_proteins")
        data = get_items()

        if get_header == "none":
            pages = Pages(table_df)
            conf = json.loads(conf)
            # trend = data_flow(table_df)
            trend = create_combined_chart_cumulative_growth(table_df, int(first_leveled_width))
            trend_by_method = group_data_by_methods(table_df)
            default_display = "taxonomic_domain"
            request_for_group_list = request_for_group
            group_item_str = default_display # Add later (+ request_for_group_list)
            unique_group_list = convert_to_list_of_dicts(group_item_str)
            
            group_graph_array = []

            for key, graph in enumerate(unique_group_list):
                conf["x-angle"] = graph["x-angle"]
                
                group_graph, _ = pages.view_dashboard(
                    get_query_params=graph["name"], 
                    conf=conf, ranges_={}
                )
                obj = {
                    "chart_obj": group_graph,
                    "id": "graph" + str(key),
                    "name": "graph " + str(key),
                    "groups": graph,
                }
                group_graph_array.append(obj)
            
            membrane_group_chart = create_grouped_bar_chart(table_df)
            
            # membrane_group_chart = group_annotation(
            #     membrane_group_chart_obj, 
            #     ["Group 1", "Group 2", "Group 3"]
            # ).to_dict()

            result = {
                'data': data,
                'trend': trend,
                'trend_by_method': trend_by_method,
                'group_graph_array': group_graph_array,
                'membrane_group_chart': membrane_group_chart
            }
        else:
            result = {'data': data}

        # Store the result in the cache
        ttl_in_seconds = timedelta(days=10).total_seconds()
        self.cache.set_item(cache_key, result, ttl=ttl_in_seconds)  # Cache for 10 days

        return jsonify(result)

class DashboardInconsistencies(Resource):
    def __init__(self):
        self.cache = RedisCache()

    # @staticmethod
    @token_required
    def get(self):
        chart_width = request.args.get('width', 800)
        # Define a static cache key for this method
        cache_key = "get:inconsistencies"

        # Check if the result is in the cache
        cached_result = self.cache.get_item(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        all_data, _, _, _, _ = DataService.get_data_from_DB()
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
        inconsistencies_by_year = aggregate_inconsistencies(df_combined)

        # Transform the aggregated data
        transformed_data = transform_dataframe(inconsistencies_by_year)
        # transformed_data.to_csv("discrepancies.csv")
        # Create and display the visualization
        chart_with_table = create_visualization(transformed_data, chart_width)
        result = {
            'inconsistencies': convert_chart(chart_with_table),
        }

        # Store the result in the cache
        ttl_in_seconds = timedelta(days=10).total_seconds()
        self.cache.set_item(cache_key, result, ttl=ttl_in_seconds)  # Cache for 1 hour

        return jsonify(result)

class DashboardOthers(Resource):
    def __init__(self):
        self.cache = RedisCache()

    # @staticmethod
    @token_required
    def get(self):
        # Define a static cache key for this method
        cache_key = "get:trends_and_mean_resolution"

        # Check if the result is in the cache
        cached_result = self.cache.get_item(cache_key)
        if cached_result:
            return jsonify(cached_result)

        # If not in cache, proceed with data retrieval and processing
        table_df = get_table_as_dataframe("membrane_proteins")

        # Get the DataFrame directly
        pages = Pages(table_df)
        trends_by_database_year = pages.view_trends_by_database_year()
        mean_resolution_by_year = pages.average_resolution_over_years(table_df)

        result = {
            'trends_by_database_year': trends_by_database_year,
            'mean_resolution_by_year': mean_resolution_by_year
        }

        # Store the result in the cache
        ttl_in_seconds = timedelta(days=10).total_seconds()
        self.cache.set_item(cache_key, result, ttl=ttl_in_seconds)  # Cache for 1 hour

        return jsonify(result)
    
class DashboardMap(Resource):
    def __init__(self):
        self.cache = RedisCache()

    # @staticmethod
    @token_required
    def get(self):
        # Define a static cache key for this method
        cache_key = "get:dashboard_map"

        # Check if the result is in the cache
        cached_result = self.cache.get_item(cache_key)
        if cached_result:
            return jsonify(cached_result)

        # If not in cache, proceed with data retrieval and processing
        table_df = get_table_as_dataframe("membrane_proteins")

        # Get the DataFrame directly
        pages = Pages(table_df)
        map_data, map = pages.getMap()
        release_by_country = pages.releasedStructuresByCountries(map)
        get_country_data = get_data_by_countries(table_df).to_dict(orient="records")
        
        result = {
            "map": map,
            "europe_map": {},
            "map_data": map_data,
            "get_country_data": get_country_data,
            "release_by_country": release_by_country
        }

        # Store the result in the cache
        ttl_in_seconds = timedelta(days=10).total_seconds()
        self.cache.set_item(cache_key, result, ttl=ttl_in_seconds)  # Cache for 1 hour

        return jsonify(result)

class MembraneProteinList(Resource):
    # @staticmethod
    @token_required
    def post(self):
        data = request.json
        download = data.get("download", "none") # request.args.get('download', default='none', type=str)
        data = get_items(data)
        if(download in ["csv", "xlsx"]):
            if(download == "csv"):
                filename = 'output_data.csv'
                # Convert DataFrame to CSV
                csv_data = data.to_csv(index=False)
                # Create a file-like buffer
                buffer = io.StringIO()
                # Write the CSV data to the buffer
                buffer.write(csv_data)
                # Set up response headers
                response = current_app.make_response(buffer.getvalue())
                response.headers['Content-Type'] = 'text/csv'
                response.headers['Content-Disposition'] = 'attachment; filename=' + filename

                return response
            elif(download == "xlsx"):
                filename = 'output_data.xlsx'
                # Create a file-like buffer
                buffer = io.BytesIO()
                # Convert DataFrame to XLSX
                data.to_excel(buffer, index=False)
                # Set up response headers
                response = current_app.make_response(buffer.getvalue())
                response.headers['Content-Type'] = 'text/xlsx'
                response.headers['Content-Disposition'] = 'attachment; filename=' + filename

                return response
        result = {
            'data': data,
        }

        return jsonify(result)
      
class SummaryStatistics(Resource):
    @token_required
    def get(self):
        # Get the DataFrame directly
        table_df = get_table_as_dataframe("membrane_proteins")
        pages = Pages(table_df)
        
        check_which_page = request.args.get('stats-data', 'no_where')
        group_field_selection = request.args.get('field_selection', 'species')
        from_range = request.args.get('from', 0)
        to_range = request.args.get('to', 50)
        if(not "*" in group_field_selection):
            parent_data, summary_search_filter_options, group_field_selection = summaryStatisticsConverter(group_field_selection)
        else:
            parent_data, summary_search_filter_options, _ = summaryStatisticsConverter(group_field_selection)

        group_dict = find_dict_with_value_in_nested_data(stats_data(for_processing=False), group_field_selection)
        other_group_dict = find_dicts_with_value_not_equal(stats_data(for_processing=False), group_field_selection)

        if (check_which_page == "stats-categories"):
            data = parent_data
        else:
            conf = request.args.get('chart_conf', '{"color": "#005EB8", "opacity": 0.9}')
            conf = json.loads(conf)
            ranges_ = {
                "from": from_range,
                "to": to_range
            }
            group_graph, dataframe = pages.view_dashboard(group_field_selection, conf, ranges_)
            merged_list = summary_search_filter_options
            sorted_frame = dataframe.sort_values(by='Cumulative MP Structures', ascending=False).to_dict('records')
            if("rcsentinfo_" in group_field_selection):
                sorted_frame = pd.DataFrame(sorted_frame).rename(
                    columns={
                        group_field_selection.split("*")[0]: group_field_selection.split("*")[0].replace("rcsentinfo_", "")
                    }
                ).to_dict(orient='records')
            dataPerc = {}
            if(not '*' in group_field_selection):
                dataPerc = getPercentage(df=table_df, column=group_field_selection)
                
            data = {
                "status"        : 'success',
                "dataPerc"      : dataPerc,
                "group_dict"    : group_dict,
                "search_object" : merged_list,
                "data"          : group_graph,
                "dataframe"     : sorted_frame,
                "other_group_dict": other_group_dict,
                "search_key"    : group_field_selection,
                "headers"       : tableHeader(dataframe.columns),
            }
        return jsonify(data)

class SummaryStatisticsLines(Resource):

    def __init__(self):
        pass

    def get(self):
        check_which_page = request.args.get('stats-data', 'no_where')

        group_field_selection = request.args.get('field_selection', 'species')

        parent_data, summary_search_filter_options, group_field_selection = summaryStatisticsConverter(group_field_selection)
        group_dict = find_dict_with_value_in_nested_data(stats_data(), group_field_selection)
        if (check_which_page == "stats-categories"):
            data = parent_data
        else:
            conf = request.args.get('chart_conf', '{"color": "#005EB8", "opacity": 0.9}')
            conf = json.loads(conf)
            group_graph, dataframe = self.pages.view_dashboard(group_field_selection, conf)
            merged_list = summary_search_filter_options
            sorted_frame = dataframe.sort_values(by='Values', ascending=False).to_dict('records')
            data = {
                "group_dict"    : group_dict,\
                "search_object" : merged_list,
                "data"          : group_graph,
                "headers"       : tableHeader(dataframe.columns),
                "dataframe"     : sorted_frame,
                "search_key"    : group_field_selection,
                "status"        : 'success',
            }
        return jsonify(data)

class UseCases(Resource):
    def __init__(self):
        pass

    def get(self):
        cases = [
            {"value": "case_1", "name": "case 1", "desc": """
                Use Case 1: K-Means Clustering for Structural Similarity Analysis
                Objective: Perform clustering on enriched Mpstruct and PDB data to identify structurally similar protein conformations.
                <div>   
                    Steps:
                    <ul>
                        <li>Data Collection: Retrieve protein structural data from Mpstruct and PDB databases, including attributes like secondary structure elements, ligand binding sites, and torsion angles.</li>
                        <li>Feature Engineering: Preprocess and transform the data to create relevant features for clustering, such as combining torsion angles and secondary structure information.</li>
                        <li>K-Means Clustering: Apply K-Means clustering algorithm to group protein structures based on their structural features. Determine the optimal number of clusters using techniques like the elbow method.</li>
                        <li>Visualization: Visualize the clusters in a reduced dimension space using techniques like Principal Component Analysis (PCA).</li>
                        <li>Interpretation: Analyze the clusters to identify proteins with similar structural characteristics, potentially revealing insights into functional relationships.</li>
                    </ul>
                </div>
            """,
            "target": "Resolution"
            },
            {"value": "case_2", "name": "case 2", "desc": """
                Use Case 2: K-Means Clustering for Structural Similarity Analysis
                Objective: Perform clustering on enriched Mpstruct and PDB data to identify structurally similar protein conformations.
                <div>   
                    Steps:
                    <ul>
                        <li>Data Collection: Retrieve protein structural data from Mpstruct and PDB databases, including attributes like secondary structure elements, ligand binding sites, and torsion angles.</li>
                        <li>Feature Engineering: Preprocess and transform the data to create relevant features for clustering, such as combining torsion angles and secondary structure information.</li>
                        <li>K-Means Clustering: Apply K-Means clustering algorithm to group protein structures based on their structural features. Determine the optimal number of clusters using techniques like the elbow method.</li>
                        <li>Visualization: Visualize the clusters in a reduced dimension space using techniques like Principal Component Analysis (PCA).</li>
                        <li>Interpretation: Analyze the clusters to identify proteins with similar structural characteristics, potentially revealing insights into functional relationships.</li>
                    </ul>
                </div>
            """,
            "target": "Resolution"
            },
            {"value": "case_3", "name": "case 3", "desc": """
                Use Case 3: K-Means Clustering for Structural Similarity Analysis
                Objective: Perform clustering on enriched Mpstruct and PDB data to identify structurally similar protein conformations.
                <div>   
                    Steps:
                    <ul>
                        <li>Data Collection: Retrieve protein structural data from Mpstruct and PDB databases, including attributes like secondary structure elements, ligand binding sites, and torsion angles.</li>
                        <li>Feature Engineering: Preprocess and transform the data to create relevant features for clustering, such as combining torsion angles and secondary structure information.</li>
                        <li>K-Means Clustering: Apply K-Means clustering algorithm to group protein structures based on their structural features. Determine the optimal number of clusters using techniques like the elbow method.</li>
                        <li>Visualization: Visualize the clusters in a reduced dimension space using techniques like Principal Component Analysis (PCA).</li>
                        <li>Interpretation: Analyze the clusters to identify proteins with similar structural characteristics, potentially revealing insights into functional relationships.</li>
                    </ul>
                </div>
            """,
            "target": "Resolution"
            },
        ]

        return jsonify(cases)

class AttributeVisualization(Resource):
    def __init__(self):
        self.cache = RedisCache()

    @token_required
    def get(self):
        # Define a static cache key for this method
        cache_key = "get:data_attributes"

        # Check if the result is in the cache
        cached_result = self.cache.get_item(cache_key)
        if cached_result:
            return ApiResponse.success(cached_result, "Fetched variables successfully", 200)

        # If not in cache, proceed with data retrieval and processing
        column_PDB = preprocessVariables(getPDBDB())
        column_MPstruct = preprocessVariables(getMPstructDB())
        
        column_OPM = preprocessVariables(getOPMDB())
        column_Uniprot = preprocessVariables(getUniprotDB())
        
        # Common attribute for OPM and MPstruc and PDB
        common_attributes_opm = set(column_PDB) & set(column_MPstruct) & set(column_OPM)
        common_attributes_opm.discard('Id')
        common_column_opm = preprocessVariables(list(common_attributes_opm))
        
        common_attributes_uniprot = set(column_PDB) & set(column_Uniprot)
        common_attributes_uniprot.discard('Id')
        common_column_uniprot = preprocessVariables(list(common_attributes_uniprot))
        
        common_attributes = set(column_PDB) & set(column_MPstruct)
        common_attributes.discard('Id')
        updated_text_list = [text.replace("Uniprot id", "Uniprot Id") for text in list(common_attributes)]
        common_columns = preprocessVariables(updated_text_list)
        
        data = [
            {
                "name": "OPM",
                "columns": column_OPM,
                "column_count": len(column_OPM),
                "route": "/attribute-opm"
            },
            {
                "name": "Uniprot",
                "columns": column_Uniprot,
                "column_count": len(column_Uniprot),
                "route": "/attribute-uniprot"
            },
            {
                "name": "PDB",
                "columns": column_PDB,
                "column_count": len(column_PDB),
                "route": "/attribute-pdb"
            },
            {
                "name": "MPstruc",
                "columns": column_MPstruct,
                "column_count": len(column_MPstruct),
                "route": "/attribute-mpstruc"
            },
            {
                "name": "MPstruc, PDB",
                "columns": common_columns,
                "column_count": len(common_columns),
                "route": "/attributes-mpstruc-pdb"
            },
            {
                "name": "MPstruc, OPM",
                "columns": common_column_opm,
                "column_count": len(common_column_opm),
                "route": "/attributes-mpstruc-opm"
            },
            {
                "name": "PDB, OPM",
                "columns": common_column_opm,
                "column_count": len(common_column_opm),
                "route": "/attributes-pdb-opm"
            },
            {
                "name": "PDB, Uniprot",
                "columns": common_column_uniprot,
                "column_count": len(common_column_uniprot),
                "route": "/attributes-pdb-uniprot"
            }
        ]
        
        # Store the result in the cache
        ttl_in_seconds = timedelta(days=10).total_seconds()
        self.cache.set_item(cache_key, data, ttl=ttl_in_seconds)  # Cache for 10 days

        return ApiResponse.success(data, "Fetched variables successfully", 200)
          
class SearchMergedDatabases(Resource):
    # @staticmethod
    @token_required
    def get(self):
        pdb_code = request.args.get('pdb_code', default='1PTH', type=str)
        data = search_merged_databases(pdb_code)
        result = {
            'data': data,
        }

        return jsonify(result)
    
class OptionFilters(Resource):
    
    def get(self):
        
        response = {
            "group": group_filter_options(),
            "species": species_filter_options(),
            "subgroup": subgroup_filter_options(),
            "family_name": family_name_filter_options(),
            "super_family": super_family_filter_options(),
            "membrane_name": membrane_name_filter_options(),
            "taxonomic_domain": taxonomic_domain_filter_options(),
            "molecular_function": molecular_function_filter_options(),
            "cellular_component": cellular_component_filter_options(),
            "biological_process": biological_process_filter_options(),
            "experimental_methods": experimental_methods_filter_options(),
            "super_family_class_type": super_family_class_type_filter_options()
        }
        
        return generate_response(
            data=response, message="fetched filter options successfully", status=HTTPStatus.CREATED
        )

# Load CSV once at startup
DF = pd.read_csv('./datasets/expert_annotation_predicted.csv') 
class RecordsListAnnotated(Resource):
    def get(self):
        df = DF
        extra = get_columns_by_pdb_codes(pdb_codes=df['PDB Code'].tolist(), columns=["pdb_code", "TMbed_tm_count", "DeepTMHMM_tm_count"])
        extra_df = pd.DataFrame(extra)
        merged_df = df.merge(
            extra_df,
            how='left',
            left_on='PDB Code',
            right_on='pdb_code'
        )
        # Apply any query-param filters that match column names
        # for col in DF.columns:
        #     val = request.args.get(col)
        #     if val is not None:
        #         df = df[df[col] == val]
        # Replace NaN values with an empty string
        df = merged_df.fillna("")
        # merged_df.to_csv("expert_annotation_predicted.csv", index=False)
        records = df.to_dict(orient='records')
        return ApiResponse.success(data=records)


class RecordAnnotated(Resource):
    def get(self, pdb_code: str):
        rec = DF[DF['PDB Code'] == pdb_code]
        if rec.empty:
            
            return ApiResponse.error(
                message=f'PDB Code {pdb_code} not found',
                status_code=404
            )
        # Replace NaN values with an empty string
        rec = rec.fillna("")
        record = rec.to_dict(orient='records')[0]
        return ApiResponse.success(data=record)