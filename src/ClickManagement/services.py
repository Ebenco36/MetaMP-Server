from __future__ import annotations

from src.ClickManagement.models import Click
from database.db import db


class ClickRepository:
    def create(self, click: Click) -> Click:
        db.session.add(click)
        db.session.commit()
        return click


class ClickTrackingService:
    def __init__(self, repository: ClickRepository | None = None):
        self.repository = repository or ClickRepository()

    def record_click(
        self,
        *,
        user_id: int,
        session_id: str,
        element_id: str,
        page_url: str,
        data: str | None = None,
    ) -> Click:
        click = Click(
            user_id=user_id,
            session_id=session_id,
            element_id=element_id,
            page_url=page_url,
            data=data,
        )
        return self.repository.create(click)
