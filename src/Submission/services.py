from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional

from database.db import db
from src.Submission.models import Submission
from src.utils.emailService import EmailService


class SubmissionNotFoundError(Exception):
    """Raised when a submission cannot be found."""


@dataclass(frozen=True)
class SubmissionNotificationPayload:
    subject: str
    sender: str
    recipients: list[str]
    body: str
    html: str


class SubmissionRepository(ABC):
    @abstractmethod
    def list_all(self) -> Iterable[Submission]:
        raise NotImplementedError

    @abstractmethod
    def get_by_id(self, submission_id: int) -> Optional[Submission]:
        raise NotImplementedError

    @abstractmethod
    def create(self, payload: dict) -> Submission:
        raise NotImplementedError

    @abstractmethod
    def update(self, submission: Submission, payload: dict) -> Submission:
        raise NotImplementedError

    @abstractmethod
    def delete(self, submission: Submission) -> None:
        raise NotImplementedError


class SubmissionNotifier(ABC):
    @abstractmethod
    def notify(self, payload: SubmissionNotificationPayload) -> None:
        raise NotImplementedError


class SQLAlchemySubmissionRepository(SubmissionRepository):
    def list_all(self) -> Iterable[Submission]:
        return Submission.query.all()

    def get_by_id(self, submission_id: int) -> Optional[Submission]:
        return db.session.get(Submission, submission_id)

    def create(self, payload: dict) -> Submission:
        submission = Submission(**payload)
        db.session.add(submission)
        db.session.commit()
        return submission

    def update(self, submission: Submission, payload: dict) -> Submission:
        for key, value in payload.items():
            setattr(submission, key, value)

        db.session.commit()
        return submission

    def delete(self, submission: Submission) -> None:
        db.session.delete(submission)
        db.session.commit()


class EmailSubmissionNotifier(SubmissionNotifier):
    def __init__(self, email_service: EmailService):
        self._email_service = email_service

    def notify(self, payload: SubmissionNotificationPayload) -> None:
        self._email_service.send_email_from_data(
            {
                "subject": payload.subject,
                "sender": payload.sender,
                "recipients": payload.recipients,
                "body": payload.body,
                "html": payload.html,
            }
        )


class SubmissionService:
    def __init__(
        self,
        repository: SubmissionRepository,
        notifier: Optional[SubmissionNotifier] = None,
    ):
        self._repository = repository
        self._notifier = notifier

    def list_submissions(self) -> Iterable[Submission]:
        return self._repository.list_all()

    def get_submission(self, submission_id: int) -> Submission:
        submission = self._repository.get_by_id(submission_id)
        if submission is None:
            raise SubmissionNotFoundError("Submission not found")
        return submission

    def create_submission(self, payload: dict) -> Submission:
        submission = self._repository.create(payload)

        if self._notifier is not None:
            self._notifier.notify(
                SubmissionNotificationPayload(
                    subject=(
                        f"Message from: {submission.name} "
                        f"with email: {submission.email}"
                    ),
                    sender=submission.email,
                    recipients=["ola@gmail.com", "ebenco94@gmail.com"],
                    body=f"Message from: {submission.name}",
                    html=submission.description,
                )
            )

        return submission

    def update_submission(self, submission_id: int, payload: dict) -> Submission:
        submission = self.get_submission(submission_id)
        return self._repository.update(submission, payload)

    def delete_submission(self, submission_id: int) -> Submission:
        submission = self.get_submission(submission_id)
        self._repository.delete(submission)
        return submission
