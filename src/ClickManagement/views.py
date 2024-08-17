import uuid
from flask_restful import Resource, reqparse
from src.ClickManagement.models import db, Click
from flask import request, g
from src.ClickManagement.requests import UserClickRequest

from src.middlewares.auth_middleware import token_required

class ClickResource(Resource):
    @token_required
    def post(self):
        arg = UserClickRequest.parse_args()
        element_id = arg['event']
        # Generate a session ID if not already present
        session_id = request.headers.get('X-Session-ID') or str(uuid.uuid4())
        current_user = g.current_user
        # Record the user click
        new_click = Click(
            user_id=current_user.id,
            session_id=session_id,
            element_id=element_id,
            page_url=request.url  # Assuming you want to store the URL where the click occurred
        )
        db.session.add(new_click)
        db.session.commit()

        return {'message': 'Click recorded successfully', 'element_id': element_id, 'session_id': session_id}
