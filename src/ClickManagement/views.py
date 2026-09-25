import uuid
from flask_restful import Resource
from flask import request, g
from src.ClickManagement.requests import UserClickRequest
from src.ClickManagement.services import ClickTrackingService

from src.middlewares.auth_middleware import token_required


class ClickResource(Resource):
    def __init__(self):
        self.click_tracking_service = ClickTrackingService()

    @token_required
    def post(self):
        arg = UserClickRequest.parse_args()
        element_id = arg["event"]
        session_id = request.headers.get('X-Session-ID') or str(uuid.uuid4())
        current_user = g.current_user

        self.click_tracking_service.record_click(
            user_id=current_user.id,
            session_id=session_id,
            element_id=element_id,
            page_url=request.url,
            data=arg.get("data"),
        )

        return {
            'message': 'Click recorded successfully',
            'element_id': element_id,
            'session_id': session_id,
        }
