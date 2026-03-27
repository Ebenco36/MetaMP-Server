from dataclasses import dataclass
from importlib import import_module

from flask import Blueprint, Flask
from flask_restful import Api


@dataclass(frozen=True)
class RouteModule:
    name: str
    register_callback_path: str
    url_prefix: str = "/api/v1"

    def load_callback(self):
        module_path, callback_name = self.register_callback_path.rsplit(".", 1)
        return getattr(import_module(module_path), callback_name)


class RouteInitialization:
    def __init__(self):
        self.route_modules = [
            RouteModule(name="auth", register_callback_path="src.User.routes.create_authentication_routes"),
            RouteModule(name="api", register_callback_path="src.Dashboard.routes.routes"),
            RouteModule(name="uot", register_callback_path="src.UOT.routes.UOT_routes"),
            RouteModule(name="text_to_db", register_callback_path="src.sql2db.routes.text_to_db"),
            RouteModule(name="training", register_callback_path="src.Training.routes.training_routes"),
            RouteModule(name="click_management", register_callback_path="src.ClickManagement.routes.click_routes"),
            RouteModule(name="feedbacks", register_callback_path="src.Feedbacks.routes.feedback_routes"),
            RouteModule(name="mp", register_callback_path="src.MP.routes.MP_routes"),
            RouteModule(name="opm", register_callback_path="src.OPM.routes.OPM_routes"),
            RouteModule(name="contact", register_callback_path="src.Contact.routes.Contact_routes"),
            RouteModule(name="submission", register_callback_path="src.Submission.routes.submission_routes"),
        ]

    def init_app(self, flask_app: Flask):
        for route_module in self.route_modules:
            blueprint = Blueprint(route_module.name, __name__, static_url_path="assets")
            api = Api(blueprint)
            route_module.load_callback()(api)
            flask_app.register_blueprint(blueprint, url_prefix=route_module.url_prefix)
