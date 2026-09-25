import os

from flask import Flask, current_app, g
from flask_restful import Api
from sqlalchemy import text

from src.core.celery_factory import configure_celery
from src.core.config import configure_runtime_environment
from src.core.error_handlers import configure_logging, register_error_handlers
from src.core.extensions import admin, babel, cors, db, mail
from src.Feedbacks.realtime import configure_socketio, register_structure_note_socket_handlers
from src.middlewares.auth_middleware import token_required
from src.routes import RouteInitialization
from src.utils.response import ApiResponse


configure_runtime_environment()

# Import ORM models that must be registered with SQLAlchemy metadata at app startup.
from src.MP import model_tmalphafold  # noqa: E402,F401


def _resolve_config_object() -> str:
    return os.getenv("APP_SETTINGS", "config.config.DevelopmentConfig")


def get_routes():
    app = current_app._get_current_object()
    routes = []

    for rule in app.url_map.iter_rules():
        routes.append(
            {
                "endpoint": rule.endpoint,
                "methods": sorted(rule.methods),
                "path": rule.rule,
            }
        )

    return routes


def create_app() -> Flask:
    app = Flask(__name__)
    Api(app)
    app.config.from_object(_resolve_config_object())
    app.url_map.strict_slashes = False

    db.init_app(app)
    mail.init_app(app)
    cors.init_app(app)
    babel.init_app(app)
    admin.init_app(app)
    configure_celery(app)
    configure_socketio(app)
    register_structure_note_socket_handlers()

    configure_logging(app)
    register_error_handlers(app)
    RouteInitialization().init_app(app)
    register_health_routes(app)

    return app


def register_health_routes(app: Flask) -> None:
    @app.route("/api/v1/health/live")
    def live_health():
        return ApiResponse.success(
            {
                "status": "ok",
                "service": "metamp-server",
                "check": "live",
            }
        )

    @app.route("/api/v1/health/ready")
    def ready_health():
        checks = {"database": "ok"}
        overall_status = "ok"

        try:
            db.session.execute(text("SELECT 1"))
        except Exception:
            checks["database"] = "error"
            overall_status = "error"

        status_code = 200 if overall_status == "ok" else 503
        return ApiResponse.success(
            {
                "status": overall_status,
                "service": "metamp-server",
                "check": "ready",
                "checks": checks,
            },
            status_code=status_code,
        )

    @app.route("/api/v1/protected_route")
    @token_required
    def protected_route():
        current_user = g.current_user
        return (
            f"This route is protected. Current user: {current_user.username}"
        )

    @app.route("/api/v1/route_list")
    def route_lists():
        return ApiResponse.success(get_routes())
