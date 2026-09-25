import io
from io import StringIO
from src.utils.response import ApiResponse
from werkzeug.utils import secure_filename
from flask_restful import Resource, reqparse
from flask import jsonify, request, send_file, current_app
from src.middlewares.auth_middleware import token_required


def _mp_services():
    from src.MP.services import (
        MPArtifactService,
        MPDatasetService,
        MPExternalSampleService,
        MPFilterService,
        MPGroupSubgroupService,
        MPMLWorkspaceService,
        MPModelingService,
        MPSemiSupervisedPredictionService,
        MPSequenceTMService,
        MPTMBackfillService,
        MPTMRateLimitService,
    )

    return {
        "MPArtifactService": MPArtifactService,
        "MPDatasetService": MPDatasetService,
        "MPExternalSampleService": MPExternalSampleService,
        "MPFilterService": MPFilterService,
        "MPGroupSubgroupService": MPGroupSubgroupService,
        "MPMLWorkspaceService": MPMLWorkspaceService,
        "MPModelingService": MPModelingService,
        "MPSemiSupervisedPredictionService": MPSemiSupervisedPredictionService,
        "MPSequenceTMService": MPSequenceTMService,
        "MPTMBackfillService": MPTMBackfillService,
        "MPTMRateLimitService": MPTMRateLimitService,
    }


def _tm_request_identifier():
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or "unknown"


class DataResource(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('experimental_method', type=str, help='select resolution method')
        self.parser.add_argument('download', type=str, help='download data as well')
        
    def get(self):
        services = _mp_services()

        # Access query parameters from the URL
        experimental_method = request.args.get('experimental_method', None)
        page = request.args.get('page', 1)
        per_page = request.args.get('per_page', 10)
        
        # download format
        download = request.args.get('download', None)
        if(download):
            try:
                download_payload = services["MPDatasetService"].build_download_payload(
                    experimental_method,
                    download,
                )
            except ValueError as exc:
                return ApiResponse.error(str(exc), 400)

            response = current_app.make_response(download_payload["content"])
            response.headers["Content-Type"] = download_payload["content_type"]
            response.headers["Content-Disposition"] = (
                "attachment; filename=" + download_payload["filename"]
            )
            return response
        else:
            # page default
            data = services["MPDatasetService"].get_records_by_experimental_method(
                experimental_method,
                page,
                per_page,
            )
            if(data):
                return ApiResponse.success(data, "Fetch records successfully.")
            else: 
                return ApiResponse.error("Not found!", 404)
        
class CategoricalDataResource(Resource):
    def get(self):
        data = _mp_services()["MPDatasetService"].get_categorical_values()
        if(data):
            return data
        else: 
            return ApiResponse.error("Not found!", 404)
        
class getFilterBasedOnMethodResource(Resource):
    def get(self):
        method_type = request.args.get('method_type', "All")
        filter_list = _mp_services()["MPFilterService"].get_filter_options_payload(method_type)
        return ApiResponse.success(filter_list, "Fetch filter list successfully.")
    
    def post(self):
        resp = _mp_services()["MPFilterService"].build_chart_payload(request.get_json() or {})
        return ApiResponse.success(resp, "Fetch filter list successfully.")
               
class DataFilterResource(Resource):
    def __init__(self):
        pass
        
    def get(self):
        filter_list = _mp_services()["MPModelingService"].get_modeling_filter_payload()
        return ApiResponse.success(filter_list, "Fetch filter list successfully.")
                   
class UsupervisedResource(Resource):
    def post(self):
        try:
            resp = _mp_services()["MPModelingService"].run_unsupervised_pipeline(
                request.get_json() or {}
            )
        except ValueError as exc:
            return ApiResponse.error(message=str(exc), status_code=400)
        return ApiResponse.success(resp, "Fetch records successfully.")
      
class GroupSubGroupResource(Resource):
    def __init__(self) -> None:
        super().__init__()
        
    def get(self):
        chart_type = request.args.get('chart_type', "subburst")
        chart_width = int(request.args.get('chart_width', 800))
        chart_height = int(request.args.get('chart_height', 500))
        try:
            fig_json = _mp_services()["MPGroupSubgroupService"].build_chart(
                chart_type=chart_type,
                chart_width=chart_width,
                chart_height=chart_height,
            )
        except Exception as exc:
            return ApiResponse.error(str(exc), 500)

        return ApiResponse.success(fig_json, "Fetch records successfully.") 
      
class MLPrediction(Resource):
    def __init__(self):
        pass
    
    def get(self):
        data = _mp_services()["MPArtifactService"].list_models_and_reductions()
        return ApiResponse.success(data, "Fetch records successfully.")

class MLPredictionAccuracy(Resource):
    def __init__(self):
        pass
    
    def get(self):
        dim_reduction = request.args.get('dim_reduction', "pca")
        try:
            data = _mp_services()["MPArtifactService"].get_accuracy_records(dim_reduction)
        except FileNotFoundError as exc:
            return ApiResponse.error(str(exc), 404)

        return ApiResponse.success(
            data, 
            "Fetch records successfully."
        )
         
class MLDimensionalityReductionCharts(Resource):
    def __init__(self):
        pass
    
    def get(self):
        try:
            data = _mp_services()["MPArtifactService"].get_dimensionality_reduction_charts()
        except FileNotFoundError as exc:
            return ApiResponse.error(str(exc), 404)

        return ApiResponse.success(
            data, 
            "Fetch records successfully."
        )

class GenerateRealSampleDataTest(Resource):
    def post(self):
        try:
            data = request.form
            
        except Exception as e:
            return ApiResponse.error("Validation Error", 400, "Invalid JSON data" + str(e))
        
        pdb_codes_unclean = data.get('pdb_codes', '')  # Replace with your list of PDB codes
        pdb_codes = [code.strip() for code in pdb_codes_unclean.split(",") if code.strip()]  # Remove empty strings and strip whitespace
        try:
            csv_bytes = _mp_services()["MPExternalSampleService"].export_real_sample_csv(pdb_codes)
        except ValueError as exc:
            return ApiResponse.error("Validation Error", 400, str(exc))
        except LookupError as exc:
            return ApiResponse.error("Validation Error", 404, str(exc))
        except Exception as exc:
            return ApiResponse.error("Validation Error", 500, str(exc))

        output = io.BytesIO(csv_bytes)
        output.seek(0)
        return send_file(output, mimetype='text/csv', download_name='real_examples.csv', as_attachment=True)

class MLPredictionPost(Resource):    
    def post(self):
        if 'data_file' not in request.files:
            return ApiResponse.error("Validation Error", 400, "No file part")

        file = request.files['data_file']

        if file.filename == '':
            return ApiResponse.error("Validation Error", 400, "No selected file")

        if file and file.filename.endswith('.csv'):
            import pandas as pd

            # Read the CSV file into a pandas DataFrame
            stream = StringIO(file.stream.read().decode("UTF8"), newline=None)
            filename_data = pd.read_csv(stream)
            try:
                csv_bytes = _mp_services()["MPMLWorkspaceService"].predict_uploaded_csv(
                    uploaded_frame=filename_data,
                    artifact_id=request.form.get("artifact_id"),
                )
            except ValueError as exc:
                return ApiResponse.error("Validation Error", 400, str(exc))
            except FileNotFoundError as exc:
                return ApiResponse.error("File Not Found", 404, str(exc))
            except Exception as exc:
                return ApiResponse.error("Validation Error", 500, str(exc))

            output = io.BytesIO(csv_bytes)
            output.seek(0)
            return send_file(output, mimetype='text/csv', download_name='predictions.csv', as_attachment=True)
        else:
            return ApiResponse.error("Validation Error", 400, "File must be a CSV file")


class MLWorkbenchSummaryResource(Resource):
    def get(self):
        data = _mp_services()["MPMLWorkspaceService"].build_summary()
        return ApiResponse.success(data, "Fetched ML workspace successfully.")


class MLWorkbenchTemplateResource(Resource):
    def get(self):
        csv_bytes = _mp_services()["MPMLWorkspaceService"].export_template_csv()
        output = io.BytesIO(csv_bytes)
        output.seek(0)
        return send_file(
            output,
            mimetype="text/csv",
            download_name="MetaMP_ML_Template.csv",
            as_attachment=True,
        )


class MLWorkbenchArtifactFigureResource(Resource):
    def get(self, filename):
        try:
            artifact_path = _mp_services()["MPMLWorkspaceService"].get_artifact_file("figures", filename)
        except FileNotFoundError as exc:
            return ApiResponse.error("File Not Found", 404, str(exc))

        return send_file(
            artifact_path,
            download_name=artifact_path.name,
            as_attachment=False,
            conditional=True,
        )


class MLWorkbenchArtifactTableResource(Resource):
    def get(self, filename):
        try:
            artifact_path = _mp_services()["MPMLWorkspaceService"].get_artifact_file("tables", filename)
        except FileNotFoundError as exc:
            return ApiResponse.error("File Not Found", 404, str(exc))

        return send_file(
            artifact_path,
            download_name=artifact_path.name,
            as_attachment=True,
            conditional=True,
        )


class MLWorkbenchPredictResource(Resource):
    def post(self):
        if "data_file" not in request.files:
            return ApiResponse.error("Validation Error", 400, "No file part")

        file = request.files["data_file"]
        if file.filename == "":
            return ApiResponse.error("Validation Error", 400, "No selected file")
        if not file.filename.lower().endswith(".csv"):
            return ApiResponse.error("Validation Error", 400, "File must be a CSV file")

        import pandas as pd

        stream = StringIO(file.stream.read().decode("UTF8"), newline=None)
        uploaded_frame = pd.read_csv(stream)
        try:
            csv_bytes = _mp_services()["MPMLWorkspaceService"].predict_uploaded_csv(
                uploaded_frame=uploaded_frame,
                artifact_id=request.form.get("artifact_id"),
            )
        except ValueError as exc:
            return ApiResponse.error("Validation Error", 400, str(exc))
        except FileNotFoundError as exc:
            return ApiResponse.error("File Not Found", 404, str(exc))
        except Exception as exc:
            return ApiResponse.error("Validation Error", 500, str(exc))

        output = io.BytesIO(csv_bytes)
        output.seek(0)
        return send_file(
            output,
            mimetype="text/csv",
            download_name="MetaMP_ML_Predictions.csv",
            as_attachment=True,
        )
        
class SequenceTMResource(Resource):
    def post(self):
        services = _mp_services()

        if 'file' not in request.files:
            return ApiResponse.error("No file part in request", 400)
        up = request.files['file']
        filename = secure_filename(up.filename)
        if not filename:
            return ApiResponse.error("No selected file", 400)
        if not filename.lower().endswith('.csv'):
            return ApiResponse.error("File must be a CSV", 400)

        try:
            df = services["MPSequenceTMService"].parse_uploaded_csv(up)
            requested_mode = request.args.get("mode")
            mode = (
                requested_mode
                or current_app.config.get("TM_PREDICTION_REQUEST_MODE", "async")
            ).lower()

            if mode == "async":
                services["MPTMRateLimitService"].enforce(
                    services["MPTMRateLimitService"].SEQUENCE_ACTION,
                    _tm_request_identifier(),
                )
                task_payload = services["MPSequenceTMService"].submit_async(
                    df,
                    include_deeptmhmm=request.form.get("include_deeptmhmm"),
                    use_gpu=request.form.get("use_gpu"),
                    max_workers=request.form.get("max_workers"),
                )
                return ApiResponse.success(
                    task_payload,
                    "TM sequence prediction queued successfully.",
                    202,
                )

            records = services["MPSequenceTMService"].analyze_sequences(df)
        except ValueError as exc:
            if "Too many TM prediction requests" in str(exc):
                return ApiResponse.error(str(exc), 429)
            return ApiResponse.error(str(exc), 400)
        except RuntimeError as exc:
            return ApiResponse.error(
                "TM prediction dependencies are not available in this service image.",
                503,
                str(exc.__cause__ or exc),
            )
        except Exception as exc:
            return ApiResponse.error(str(exc), 500)

        return ApiResponse.success(records, "TM segment counts computed successfully.")


class SequenceTMTaskStatusResource(Resource):
    def get(self, task_id):
        data = _mp_services()["MPSequenceTMService"].get_async_status(task_id)
        return ApiResponse.success(data, "Fetched TM sequence prediction status successfully.")


class TMBackfillResource(Resource):
    def post(self):
        payload = request.get_json(silent=True) or {}
        services = _mp_services()
        try:
            services["MPTMRateLimitService"].enforce(
                services["MPTMRateLimitService"].BACKFILL_ACTION,
                _tm_request_identifier(),
            )
            data = services["MPTMBackfillService"].queue_backfill(
                use_gpu=payload.get("use_gpu"),
                batch_size=payload.get("batch_size"),
                max_workers=payload.get("max_workers"),
            )
        except ValueError as exc:
            return ApiResponse.error(str(exc), 429)
        return ApiResponse.success(data, "TM annotation sync queued successfully.", 202)


class TMBackfillStatusResource(Resource):
    def get(self):
        status = _mp_services()["MPTMBackfillService"].latest_status()
        if status is None:
            return ApiResponse.error("No TM annotation sync status found.", 404)
        return ApiResponse.success(status, "Fetched TM annotation sync status successfully.")
