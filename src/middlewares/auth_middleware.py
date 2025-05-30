import jwt
from functools import wraps
from flask import current_app
from flask import request, abort, g
from src.User.model import UserModel

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if "Authorization" in request.headers:
            token_auth = request.headers["Authorization"].split(" ")
            if (len(token_auth) > 1):
                token = token_auth[1]
            elif (len(token_auth) == 1):
                token = token_auth[0]
            else:
                token = None

        if not token:
            return {
                "message": "Authentication Token is missing!",
                "data": None,
                "error": "Unauthorized"
            }, 401

        try:
            data = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])
            current_user = UserModel.query.filter_by(id=data["id"]).first()

            if current_user is None:
                return {
                    "message": "Invalid Authentication token!",
                    "data": None,
                    "error": "Unauthorized"
                }, 401

            if not current_user.status:
                abort(403)

            # Store the current user in the Flask 'g' object
            g.current_user = current_user

        except jwt.ExpiredSignatureError:
            return {'message': 'Token has expired'}, 401
        except jwt.InvalidTokenError:
            return {'message': 'Invalid token'}, 401
        except Exception as e:
            return {
                "message": "Something went wrong",
                "data": None,
                "error": str(e)
            }, 500

        return f(*args, **kwargs)

    return decorated