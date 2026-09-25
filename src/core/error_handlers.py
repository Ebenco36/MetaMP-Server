import logging

from flask import Flask

from src.utils.response import ApiResponse
from utils.errors import BadRequestException
from utils.http import bad_request, not_allowed, not_found


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(BadRequestException)
    def handle_bad_request_exception(error: BadRequestException):
        return bad_request(error)

    @app.errorhandler(404)
    def handle_not_found(_error):
        return not_found("route")

    @app.errorhandler(405)
    def handle_method_not_allowed(_error):
        return not_allowed()

    @app.errorhandler(Exception)
    def handle_unexpected_exception(error: Exception):
        app.logger.exception("Unhandled application error", exc_info=error)
        return ApiResponse.error("Internal server error", 500)


def configure_logging(app: Flask) -> None:
    if app.logger.handlers:
        return

    logging.basicConfig(level=logging.INFO)
