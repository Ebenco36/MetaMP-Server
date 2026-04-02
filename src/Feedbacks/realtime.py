from __future__ import annotations

import logging

from flask import current_app, request
from flask_socketio import ConnectionRefusedError, join_room, leave_room

from src.core.extensions import socketio
from src.ingestion.redis_support import get_redis_url
from src.middlewares.auth_middleware import AuthenticationError, get_authenticated_user_from_token


logger = logging.getLogger(__name__)

STRUCTURE_NOTES_NAMESPACE = "/structure-notes"
_socket_handlers_registered = False
_authenticated_socket_users = {}


def _normalize_code(value):
    text = str(value or "").strip().upper()
    return text or None


def _room_name(pdb_code):
    normalized = _normalize_code(pdb_code)
    return f"structure-notes:{normalized}" if normalized else None


def _parse_cors_origins(value):
    text = str(value or "").strip()
    if not text or text == "*":
        return "*"
    origins = [item.strip() for item in text.split(",") if item.strip()]
    return origins or "*"


def configure_socketio(app):
    socketio.init_app(
        app,
        async_mode="gevent",
        cors_allowed_origins=_parse_cors_origins(
            app.config.get("SOCKETIO_CORS_ALLOWED_ORIGINS", "*")
        ),
        message_queue=get_redis_url(config=app.config),
        ping_interval=int(app.config.get("SOCKETIO_PING_INTERVAL", "25")),
        ping_timeout=int(app.config.get("SOCKETIO_PING_TIMEOUT", "60")),
        logger=False,
        engineio_logger=False,
    )


def register_structure_note_socket_handlers():
    global _socket_handlers_registered
    if _socket_handlers_registered:
        return

    @socketio.on("connect", namespace=STRUCTURE_NOTES_NAMESPACE)
    def handle_connect(auth=None):
        auth = auth or {}
        token = auth.get("token")
        if not token and "Authorization" in request.headers:
            token = request.headers.get("Authorization", "").split(" ")[-1].strip()
        try:
            user = get_authenticated_user_from_token(token)
        except AuthenticationError as exc:
            raise ConnectionRefusedError(str(exc)) from exc

        _authenticated_socket_users[request.sid] = getattr(user, "id", None)
        logger.info(
            "Structure notes socket connected: sid=%s user_id=%s",
            request.sid,
            getattr(user, "id", None),
        )
        return True

    @socketio.on("disconnect", namespace=STRUCTURE_NOTES_NAMESPACE)
    def handle_disconnect():
        _authenticated_socket_users.pop(request.sid, None)

    @socketio.on("subscribe_structure_notes", namespace=STRUCTURE_NOTES_NAMESPACE)
    def handle_subscribe_structure_notes(payload=None):
        if request.sid not in _authenticated_socket_users:
            raise ConnectionRefusedError("Unauthorized")

        payload = payload or {}
        raw_codes = payload.get("pdb_codes")
        if raw_codes is None:
            raw_codes = [payload.get("pdb_code")]

        normalized_codes = []
        for value in raw_codes or []:
            normalized = _normalize_code(value)
            if normalized and normalized not in normalized_codes:
                normalized_codes.append(normalized)
                room_name = _room_name(normalized)
                if room_name:
                    join_room(room_name)

        return {
            "status": "ok",
            "pdb_codes": normalized_codes,
        }

    @socketio.on("unsubscribe_structure_notes", namespace=STRUCTURE_NOTES_NAMESPACE)
    def handle_unsubscribe_structure_notes(payload=None):
        payload = payload or {}
        raw_codes = payload.get("pdb_codes")
        if raw_codes is None:
            raw_codes = [payload.get("pdb_code")]

        normalized_codes = []
        for value in raw_codes or []:
            normalized = _normalize_code(value)
            if normalized and normalized not in normalized_codes:
                normalized_codes.append(normalized)
                room_name = _room_name(normalized)
                if room_name:
                    leave_room(room_name)

        return {
            "status": "ok",
            "pdb_codes": normalized_codes,
        }

    _socket_handlers_registered = True


def broadcast_structure_note_update(payload):
    if not payload:
        return

    normalized_codes = []
    for value in (
        payload.get("pdb_code"),
        payload.get("canonical_pdb_code"),
    ):
        normalized = _normalize_code(value)
        if normalized and normalized not in normalized_codes:
            normalized_codes.append(normalized)

    if not normalized_codes:
        return

    for pdb_code in normalized_codes:
        room_name = _room_name(pdb_code)
        if not room_name:
            continue
        socketio.emit(
            "structure_note_updated",
            payload,
            namespace=STRUCTURE_NOTES_NAMESPACE,
            room=room_name,
        )

    logger.info(
        "Broadcast structure note update for %s to %s room(s)",
        payload.get("pdb_code") or payload.get("canonical_pdb_code"),
        len(normalized_codes),
    )
