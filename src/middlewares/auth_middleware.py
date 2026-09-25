import jwt
from functools import wraps
from flask import current_app
from flask import request, abort, g
from src.User.model import UserModel


class AuthenticationError(Exception):
    pass


def extract_auth_token(req=None):
    req = req or request
    if "Authorization" not in req.headers:
        return None

    token_auth = req.headers["Authorization"].split(" ")
    if len(token_auth) > 1:
        return token_auth[1]
    if len(token_auth) == 1:
        return token_auth[0]
    return None


def get_authenticated_user_from_token(token):
    if not token:
        raise AuthenticationError("Authentication Token is missing!")

    try:
        data = jwt.decode(
            token,
            current_app.config["SECRET_KEY"],
            algorithms=["HS256"],
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Invalid token") from exc

    current_user = UserModel.query.filter_by(id=data["id"]).first()
    if current_user is None:
        raise AuthenticationError("Invalid Authentication token!")

    if not current_user.status:
        abort(403)

    return current_user

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            token = extract_auth_token(request)
            current_user = get_authenticated_user_from_token(token)
            g.current_user = current_user
        except AuthenticationError as exc:
            return {
                "message": str(exc),
                "data": None,
                "error": "Unauthorized"
            }, 401
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
