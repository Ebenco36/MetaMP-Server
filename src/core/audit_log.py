from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class MetaMPAuditLogService:
    DEFAULT_AUDIT_DIR = Path("/var/app/data/audit")

    @classmethod
    def get_audit_dir(cls, app_config=None):
        configured = None
        if app_config is not None:
            configured = app_config.get("AUDIT_LOG_DIR")
        configured = configured or os.getenv("AUDIT_LOG_DIR")

        candidates = []
        if configured:
            candidates.append(Path(configured))
        else:
            candidates.extend(
                [
                    cls.DEFAULT_AUDIT_DIR,
                    Path.cwd() / "data" / "audit",
                    Path("/tmp/metamp-audit"),
                ]
            )

        for candidate in candidates:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            except OSError:
                continue

        raise OSError("Unable to create a writable audit log directory.")

    @classmethod
    def record_event(cls, event_type, payload, app_config=None):
        audit_dir = cls.get_audit_dir(app_config=app_config)
        timestamp = datetime.now(timezone.utc)
        event = {
            "event_type": event_type,
            "timestamp": timestamp.isoformat(),
            "payload": payload,
        }
        audit_path = audit_dir / f"metamp-audit-{timestamp.date().isoformat()}.jsonl"
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")
        return audit_path
