import os
import time
import logging
from flask import Flask, g
from database.db import db
from flask_mail import Mail
from flask_cors import CORS
from flask_restful import Api
from flask_admin import Admin
from dotenv import load_dotenv
from src.utils.response import ApiResponse
try:
    from src.routes import RouteInitialization
    is_route_ready = True
except ImportError as e:
    print(e)
    is_route_ready = False

from utils.errors import BadRequestException
from logging.handlers import RotatingFileHandler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from utils.http import bad_request, not_found, not_allowed, internal_error
from src.middlewares.auth_middleware import token_required

# load_dotenv()  # load env files
# Determine which .env file to load
env = os.environ.get('FLASK_ENV', 'development')
if env == 'production':
    load_dotenv('.env.production')
else:
    load_dotenv('.env.development')
    
def reload_dotenv():
    # Clear the current environment variables
    for key in os.environ.keys():
        if key in original_keys:
            continue
        del os.environ[key]

    # Reload .env file
    load_dotenv(override=True)

original_keys = set(os.environ.keys())
reload_dotenv()

def get_routes():
    routes = []

    # Routes registered using @app.route(...)
    for rule in app.url_map.iter_rules():
        route = {
            'endpoint': rule.endpoint,
            'methods': sorted(rule.methods),
            'path': rule.rule
        }
        routes.append(route)

    # Routes registered using api.add_resource(...)
    for endpoint, rule in app.url_map._rules_by_endpoint.items():
        for rule_item in rule:
            route = {
                'endpoint': endpoint,
                'methods': sorted(rule_item.methods),
                'path': rule_item.rule
            }
            routes.append(route)

    return routes

def create_app():
    app = Flask(__name__)
    api = Api(app)
    # os.environ['REQUESTS_CA_BUNDLE'] = './ca-bundle.crt'
    app.config.from_object(os.getenv('APP_SETTINGS'))
    app.url_map.strict_slashes = False
    db.init_app(app)
    CORS(app)
    # Mail(app)
    admin = Admin(app)
    
    # Configure logging to write to a file
    log_handler = RotatingFileHandler('error.log', maxBytes=1024 * 1024, backupCount=10)
    log_handler.setLevel(logging.ERROR)
    app.logger.addHandler(log_handler)

    @app.errorhandler(500)
    def internal_server_error(e):
        # Log the error to the configured file
        app.logger.error('An internal server error occurred', exc_info=e)
        return 'Internal Server Error', 500

    """
        Route Implementation. Well structured
    """
    if(is_route_ready):
        init_route = RouteInitialization()
        init_route.init_app(app)
    
    @app.route('/api/v1/protected_route')
    @token_required
    def protected_route():
        current_user = g.current_user
        return f'This route is protected. Current user: {current_user.username}'
    
    @app.route('/api/v1/route_list')
    def route_lists():
        with app.test_request_context():
            routes = get_routes()
            return ApiResponse.success(routes)

    @app.errorhandler(BadRequestException)
    def bad_request_exception(e):
        return bad_request(e)

    @app.errorhandler(404)
    def route_not_found(e):
        return not_found('route')

    @app.errorhandler(405)
    def method_not_allowed(e):
        return not_allowed()

    @app.errorhandler(Exception)
    def internal_server_error(e):
        # Log the error to the configured file
        app.logger.error('An internal server error occurred', exc_info=e)
        return internal_error()

    return app


app = create_app()
