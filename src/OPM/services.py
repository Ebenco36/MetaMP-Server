from src.MP.services import DataService


class OPMDatasetService:
    @staticmethod
    def get_records():
        result = DataService.get_data_by_column_search_download(
            column_name=None,
            value=None,
        )
        return result.get("data", {}).to_dict(orient="records")


class OPMStatusService:
    @staticmethod
    def get_merge_status():
        return {
            "status": "success",
            "message": "OPM data access is available through the shared MP dataset service.",
        }
