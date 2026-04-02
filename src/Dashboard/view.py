import io
import json
import pandas as pd
from http import HTTPStatus
from datetime import timedelta
from flask_restful import Resource
from src.services.pages import Pages
from src.Dashboard.data import stats_data
from src.utils.response import ApiResponse
from src.utils.common import generate_response
from flask import jsonify, request, current_app, g
from src.services.Helpers.helper import tableHeader
from src.middlewares.auth_middleware import token_required
from src.services.Helpers.helper import find_dict_with_value_in_nested_data
from src.services.Helpers.helper import (
    find_dicts_with_value_not_equal,  
    summaryStatisticsConverter,
    getPercentage,
)
from src.Dashboard.services import (
    DashboardAnnotationDatasetService,
    DashboardConfigurationService,
    DashboardFieldMetadataService,
    DiscrepancyBenchmarkExportService,
    DiscrepancyReviewExportService,
    DiscrepancyReviewService,
    DashboardFilterOptionsService,
    DashboardPageService,
    get_items, 
    get_table_as_dataframe,
    get_table_as_dataframe_with_specific_columns,
    get_table_as_dataframe_download, getMPstructDB, 
    getOPMDB, getPDBDB, getUniprotDB, 
    preprocessVariables, 
    search_merged_databases
)
from utils.redisCache import RedisCache


class WelcomePage(Resource):
    def __init__(self):
        self.cache = RedisCache()
    # @staticmethod
    # @token_required
    def get(self):
        # Create a unique cache key based on request parameters
        cache_key = "get:welcome-page:" + DashboardAnnotationDatasetService._enriched_cache_key(
            DashboardConfigurationService.get_annotation_dataset_path()
        )

        # Check if the result is in the cache
        cached_result = self.cache.get_item(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        result = DashboardPageService.build_welcome_page_payload()

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
        result = DashboardPageService.build_about_payload()
        # Store the result in the cache
        ttl_in_seconds = timedelta(days=10).total_seconds()
        self.cache.set_item(cache_key, result, ttl=ttl_in_seconds)  # Cache for 10 days

        return jsonify(result)


class AboutMetaMPSummary(Resource):
    # @staticmethod
    @token_required
    def get(self):
        return jsonify(DashboardPageService.build_about_summary_payload())
   
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

        result = DashboardPageService.build_dashboard_payload(
            get_header=get_header,
            conf=conf,
            request_for_group=request_for_group,
            first_leveled_width=first_leveled_width,
        )

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
        result = DashboardPageService.build_inconsistencies_payload(chart_width)

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
        result = DashboardPageService.build_dashboard_others_payload()

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
        result = DashboardPageService.build_dashboard_map_payload()

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
            selected_column = group_field_selection.split("*")[0]
            table_columns = [
                "pdb_code",
                "bibliography_year",
                selected_column,
                "rcsentinfo_software_programs_combined",
            ]
            table_df = get_table_as_dataframe_with_specific_columns(
                "membrane_proteins",
                table_columns,
            )
            pages = Pages(table_df)
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
        query = request.args.get('q', type=str)
        if query is None:
            query = request.args.get('pdb_code', default='1PTH', type=str)

        data = search_merged_databases(query)
        result = {
            'query': query,
            'data': data,
        }

        return jsonify(result)
    
class OptionFilters(Resource):
    
    def get(self):
        response = DashboardFilterOptionsService.get_filter_options_payload()
        return generate_response(
            data=response, message="fetched filter options successfully", status=HTTPStatus.CREATED
        )

class RecordsListAnnotated(Resource):
    def get(self):
        try:
            records = DashboardAnnotationDatasetService.get_records()
        except FileNotFoundError as exc:
            return ApiResponse.error(message=str(exc), status_code=404)

        return ApiResponse.success(data=records)


class RecordAnnotated(Resource):
    def get(self, pdb_code: str):
        try:
            record = DashboardAnnotationDatasetService.get_record(pdb_code)
        except FileNotFoundError as exc:
            return ApiResponse.error(message=str(exc), status_code=404)

        if record is None:
            return ApiResponse.error(
                message=f'PDB Code {pdb_code} not found',
                status_code=404
            )

        return ApiResponse.success(data=record)


class RecordLineageResource(Resource):
    def get(self, pdb_code: str):
        try:
            record = DashboardAnnotationDatasetService.get_record(pdb_code)
        except FileNotFoundError as exc:
            return ApiResponse.error(message=str(exc), status_code=404)

        if record is None:
            return ApiResponse.error(
                message=f"PDB Code {pdb_code} not found",
                status_code=404,
            )

        return ApiResponse.success(
            data=record.get("annotation_lineage") or {},
            message="Fetched annotation lineage successfully",
        )


class DashboardMetadataResource(Resource):
    def get(self):
        return ApiResponse.success(
            data=DashboardFieldMetadataService.build_metadata_payload(),
            message="Fetched dashboard metadata successfully",
        )


class DiscrepancyReviewSummaryResource(Resource):
    def get(self):
        status = request.args.get("status", type=str)
        search = request.args.get("search", type=str)
        disagreement_only = request.args.get("disagreement_only", default="true", type=str)
        disagreement_only = str(disagreement_only).strip().lower() not in {"false", "0", "no"}
        return ApiResponse.success(
            data=DiscrepancyReviewService.summarize_candidates(
                disagreement_only=disagreement_only,
                status=status,
                search=search,
            ),
            message="Fetched discrepancy review summary successfully",
        )


class DashboardCaseStudiesResource(Resource):
    def get(self):
        return ApiResponse.success(
            data=DashboardPageService.build_case_studies_payload(),
            message="Fetched dashboard case studies successfully",
        )


class DashboardAddedValueResource(Resource):
    def get(self):
        return ApiResponse.success(
            data=DashboardPageService.build_added_value_payload(),
            message="Fetched MetaMP added-value summary successfully",
        )


class DiscrepancyReviewListResource(Resource):
    def get(self):
        status = request.args.get("status", type=str)
        search = request.args.get("search", type=str)
        page = request.args.get("page", default=1, type=int)
        per_page = request.args.get("per_page", default=DiscrepancyReviewService.DEFAULT_PAGE_SIZE, type=int)
        disagreement_only = request.args.get("disagreement_only", default="true", type=str)
        disagreement_only = str(disagreement_only).strip().lower() not in {"false", "0", "no"}

        try:
            data = DiscrepancyReviewService.list_candidates(
                status=status,
                disagreement_only=disagreement_only,
                search=search,
                page=page,
                per_page=per_page,
            )
        except FileNotFoundError as exc:
            return ApiResponse.error(message=str(exc), status_code=404)

        return ApiResponse.success(
            data=data,
            message="Discrepancy review candidates fetched successfully",
        )


class DiscrepancyReviewExportResource(Resource):
    def get(self):
        status = request.args.get("status", type=str)
        search = request.args.get("search", type=str)
        export_format = request.args.get("format", default="csv", type=str).lower()
        disagreement_only = request.args.get("disagreement_only", default="true", type=str)
        disagreement_only = str(disagreement_only).strip().lower() not in {"false", "0", "no"}

        try:
            payload = DiscrepancyReviewExportService.build_download_payload(
                export_format=export_format,
                status=status,
                disagreement_only=disagreement_only,
                search=search,
            )
        except ValueError as exc:
            return ApiResponse.error(message=str(exc), status_code=400)

        response = current_app.make_response(payload["content"])
        response.headers["Content-Type"] = payload["content_type"]
        response.headers["Content-Disposition"] = (
            "attachment; filename=" + payload["filename"]
        )
        response.headers["X-MetaMP-Generated-At"] = payload["metadata"]["generated_at"]
        return response


class DiscrepancyReviewResource(Resource):
    def get(self, pdb_code: str):
        try:
            candidate = DiscrepancyReviewService.get_candidate(pdb_code)
        except FileNotFoundError as exc:
            return ApiResponse.error(message=str(exc), status_code=404)

        if candidate is None:
            return ApiResponse.error(
                message=f"Discrepancy review candidate '{pdb_code}' not found",
                status_code=404,
            )

        return ApiResponse.success(
            data=candidate,
            message="Discrepancy review candidate fetched successfully",
        )

    @token_required
    def put(self, pdb_code: str):
        payload = request.get_json() or {}
        current_user = g.current_user

        try:
            candidate = DiscrepancyReviewService.upsert_review(
                pdb_code,
                payload,
                current_user,
            )
        except FileNotFoundError as exc:
            return ApiResponse.error(message=str(exc), status_code=404)
        except LookupError as exc:
            return ApiResponse.error(message=str(exc), status_code=404)
        except ValueError as exc:
            return ApiResponse.error(message=str(exc), status_code=400)

        return ApiResponse.success(
            data=candidate,
            message="Discrepancy review updated successfully",
        )


class DiscrepancyBenchmarkStatusResource(Resource):
    def get(self):
        try:
            metadata = DiscrepancyBenchmarkExportService.ensure_fresh_export_metadata(
                include_all=False
            )
        except FileNotFoundError as exc:
            return ApiResponse.error(message=str(exc), status_code=404)
        return ApiResponse.success(
            data=metadata,
            message="Fetched discrepancy benchmark export metadata successfully",
        )


class DiscrepancyBenchmarkExportResource(Resource):
    def get(self):
        export_format = request.args.get("format", default="csv", type=str).lower()
        include_all = request.args.get("include_all", default="false", type=str)
        include_all = str(include_all).strip().lower() in {"true", "1", "yes"}
        try:
            payload = DiscrepancyBenchmarkExportService.build_download_payload(
                export_format=export_format,
                include_all=include_all,
            )
        except ValueError as exc:
            return ApiResponse.error(message=str(exc), status_code=400)

        response = current_app.make_response(payload["content"])
        response.headers["Content-Type"] = payload["content_type"]
        response.headers["Content-Disposition"] = (
            "attachment; filename=" + payload["filename"]
        )
        response.headers["X-MetaMP-Benchmark-Generated-At"] = payload["metadata"][
            "generated_at"
        ]
        return response


class HighConfidenceSubsetExportResource(Resource):
    def get(self):
        export_format = request.args.get("format", default="csv", type=str).lower()
        try:
            payload = DiscrepancyBenchmarkExportService.build_high_confidence_download_payload(
                export_format=export_format
            )
        except ValueError as exc:
            return ApiResponse.error(message=str(exc), status_code=400)

        response = current_app.make_response(payload["content"])
        response.headers["Content-Type"] = payload["content_type"]
        response.headers["Content-Disposition"] = (
            "attachment; filename=" + payload["filename"]
        )
        response.headers["X-MetaMP-Benchmark-Generated-At"] = payload["metadata"][
            "generated_at"
        ]
        return response
