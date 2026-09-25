from __future__ import annotations

import logging
from datetime import datetime
from collections import defaultdict

from marshmallow import ValidationError
from sqlalchemy import or_

from database.db import db
from src.Feedbacks.models import (
    Feedback,
    FeedbackOption,
    FeedbackQuestion,
    StructureExpertNote,
)
from src.Feedbacks.serializers import (
    FeedbackQuestionSchema,
    FeedbackSchema,
    StructureExpertNoteCreateSchema,
    StructureExpertNoteSchema,
)


logger = logging.getLogger(__name__)


class FeedbackQuestionRepository:
    def list_all(self) -> list[FeedbackQuestion]:
        return FeedbackQuestion.query.order_by(FeedbackQuestion.id.asc()).all()

    def get_by_id(self, question_id: int) -> FeedbackQuestion | None:
        return db.session.get(FeedbackQuestion, question_id)

    def get_by_text(self, question_text: str) -> FeedbackQuestion | None:
        return FeedbackQuestion.query.filter_by(question_text=question_text).first()

    def create(self, question: FeedbackQuestion) -> FeedbackQuestion:
        db.session.add(question)
        db.session.commit()
        return question

    def delete(self, question: FeedbackQuestion) -> None:
        db.session.delete(question)
        db.session.commit()

    def save(self) -> None:
        db.session.commit()


class FeedbackRepository:
    def list_all(self) -> list[Feedback]:
        return Feedback.query.order_by(Feedback.created_at.desc()).all()

    def list_by_user_id(self, user_id: int) -> list[Feedback]:
        return Feedback.query.filter_by(user_id=user_id).order_by(Feedback.created_at.desc()).all()

    def create(self, feedback: Feedback) -> Feedback:
        db.session.add(feedback)
        db.session.commit()
        return feedback


class FeedbackQuestionService:
    def __init__(
        self,
        repository: FeedbackQuestionRepository | None = None,
        schema: FeedbackQuestionSchema | None = None,
    ):
        self.repository = repository or FeedbackQuestionRepository()
        self.schema = schema or FeedbackQuestionSchema()

    def list_questions(self) -> list[FeedbackQuestion]:
        return self.repository.list_all()

    def get_question(self, question_id: int) -> FeedbackQuestion | None:
        return self.repository.get_by_id(question_id)

    def create_question(self, payload: dict) -> FeedbackQuestion:
        validated_data = self.schema.load(payload)
        question = FeedbackQuestion(question_text=validated_data["question_text"])
        question.options = self._build_options(validated_data["options"])
        return self.repository.create(question)

    def replace_question_options(
        self,
        question_id: int,
        payload: dict,
    ) -> FeedbackQuestion | None:
        question = self.repository.get_by_id(question_id)
        if question is None:
            return None

        validated_data = self.schema.load(
            {
                "question_text": question.question_text,
                "options": payload.get("options", []),
            }
        )
        self._replace_options(question, validated_data["options"])
        return question

    def delete_question(self, question_id: int) -> bool:
        question = self.repository.get_by_id(question_id)
        if question is None:
            return False
        self.repository.delete(question)
        return True

    def update_or_create_questions(self, data: list[dict]) -> list[dict]:
        results = []

        for item in data:
            question_text = item.get("question_text")
            if not question_text:
                results.append({"error": "Invalid data format"})
                continue

            try:
                validated_data = self.schema.load(item)
            except ValidationError as error:
                results.append(
                    {
                        "error": (
                            f"Validation error for question '{question_text}': "
                            f"{error.messages}"
                        )
                    }
                )
                continue

            existing_question = self.repository.get_by_text(question_text)
            if existing_question is None:
                question = FeedbackQuestion(
                    question_text=validated_data["question_text"]
                )
                question.options = self._build_options(validated_data["options"])
                self.repository.create(question)
                results.append(
                    {"message": f"Question '{question_text}' created successfully"}
                )
                continue

            if self._options_match(existing_question, validated_data["options"]):
                results.append(
                    {"message": f"Question '{question_text}' already up to date"}
                )
                continue

            self._replace_options(existing_question, validated_data["options"])
            results.append(
                {"message": f"Question '{question_text}' updated successfully"}
            )

        return results

    @staticmethod
    def _build_options(options: list[dict]) -> list[FeedbackOption]:
        return [
            FeedbackOption(
                text=option.get("text", "") or "",
                value=option["value"],
            )
            for option in options
        ]

    def _replace_options(
        self,
        question: FeedbackQuestion,
        options: list[dict],
    ) -> None:
        FeedbackOption.query.filter_by(question_id=question.id).delete(
            synchronize_session=False
        )
        db.session.flush()

        for option in self._build_options(options):
            option.question_id = question.id
            db.session.add(option)

        db.session.commit()
        db.session.refresh(question)

    @staticmethod
    def _options_match(
        question: FeedbackQuestion,
        options: list[dict],
    ) -> bool:
        current_options = sorted(
            [
                {"text": option.text or "", "value": option.value}
                for option in question.options
            ],
            key=lambda item: (item["value"], item["text"]),
        )
        incoming_options = sorted(
            [
                {"text": option.get("text", "") or "", "value": option["value"]}
                for option in options
            ],
            key=lambda item: (item["value"], item["text"]),
        )
        return current_options == incoming_options


class UserFeedbackService:
    def __init__(
        self,
        repository: FeedbackRepository | None = None,
        schema: FeedbackSchema | None = None,
    ):
        self.repository = repository or FeedbackRepository()
        self.schema = schema or FeedbackSchema()

    def get_user_feedbacks(self, user_id: int) -> list[dict]:
        feedbacks = self.repository.list_by_user_id(user_id)
        return self.schema.dump(feedbacks, many=True)

    def get_all_feedbacks(self) -> list[dict]:
        feedbacks = self.repository.list_all()
        return self.schema.dump(feedbacks, many=True)

    def store_user_feedback(self, args: dict) -> dict:
        feedback = Feedback(
            user_id=args['user_id'],
            comment=args['comment'],
            responses=args['responses'],
            name=args['name'],
            gender=args['gender'],
            domain=args['domain'],
            is_student=args['is_student'],
            years_of_experience=args['years_of_experience'],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        created_feedback = self.repository.create(feedback)
        return self.schema.dump(created_feedback)


class FeedbackService:
    _question_service = FeedbackQuestionService()

    @staticmethod
    def update_or_create_questions(data):
        return FeedbackService._question_service.update_or_create_questions(data)


class StructureExpertNoteService:
    DEFAULT_LIST_LIMIT = 100
    DEFAULT_RECENT_LIMIT = 3
    MAX_COMMENT_EXCERPT = 160
    _table_ensured = False

    def __init__(
        self,
        schema: StructureExpertNoteCreateSchema | None = None,
        response_schema: StructureExpertNoteSchema | None = None,
    ):
        self.schema = schema or StructureExpertNoteCreateSchema()
        self.response_schema = response_schema or StructureExpertNoteSchema()

    @classmethod
    def _ensure_table_exists(cls):
        if cls._table_ensured:
            return
        StructureExpertNote.__table__.create(bind=db.engine, checkfirst=True)
        cls._table_ensured = True

    @staticmethod
    def _normalize_code(value):
        text = str(value or "").strip().upper()
        return text or None

    @classmethod
    def _resolve_record(cls, pdb_code):
        from src.Dashboard.services import DashboardAnnotationDatasetService

        normalized_code = cls._normalize_code(pdb_code)
        record = DashboardAnnotationDatasetService.get_record(normalized_code)
        if record is None:
            raise LookupError(f"Structure '{pdb_code}' was not found")
        return record

    @classmethod
    def _record_codes(cls, record):
        codes = []
        for value in (
            record.get("PDB Code"),
            record.get("pdb_code"),
            record.get("canonical_pdb_code"),
        ):
            normalized = cls._normalize_code(value)
            if normalized and normalized not in codes:
                codes.append(normalized)
        return codes

    @classmethod
    def _query_for_codes(cls, codes):
        cls._ensure_table_exists()
        normalized_codes = [cls._normalize_code(value) for value in codes if cls._normalize_code(value)]
        if not normalized_codes:
            return []

        return (
            StructureExpertNote.query.filter(
                or_(
                    StructureExpertNote.pdb_code.in_(normalized_codes),
                    StructureExpertNote.canonical_pdb_code.in_(normalized_codes),
                )
            )
            .order_by(StructureExpertNote.created_at.desc(), StructureExpertNote.id.desc())
            .all()
        )

    def _serialize_note(self, note):
        payload = self.response_schema.dump(note)
        payload["comment_excerpt"] = self._comment_excerpt(payload.get("comment"))
        return payload

    def _build_record_payload(self, record, limit=None):
        normalized_pdb_code = self._normalize_code(
            record.get("PDB Code") or record.get("pdb_code") or record.get("canonical_pdb_code")
        )
        canonical_code = self._normalize_code(record.get("canonical_pdb_code")) or normalized_pdb_code
        notes = self._query_for_codes(self._record_codes(record))
        serialized_all = [self._serialize_note(note) for note in notes]
        safe_limit = max(int(limit or self.DEFAULT_LIST_LIMIT), 1)
        serialized = serialized_all[:safe_limit]
        return {
            "pdb_code": normalized_pdb_code,
            "canonical_pdb_code": canonical_code,
            "summary": self._summarize_serialized_notes(serialized_all),
            "items": serialized,
        }

    @classmethod
    def _comment_excerpt(cls, value):
        text = str(value or "").strip()
        if not text:
            return None
        if len(text) <= cls.MAX_COMMENT_EXCERPT:
            return text
        return text[: cls.MAX_COMMENT_EXCERPT - 1].rstrip() + "..."

    @classmethod
    def _default_summary(cls):
        return {
            "note_count": 0,
            "open_note_count": 0,
            "latest_note_at": None,
            "latest_note_excerpt": None,
            "recent_notes": [],
        }

    @classmethod
    def _summarize_serialized_notes(cls, serialized_notes, recent_limit=None):
        notes = list(serialized_notes or [])
        recent_limit = recent_limit if recent_limit is not None else cls.DEFAULT_RECENT_LIMIT
        latest_note = notes[0] if notes else None
        open_note_count = sum(
            1 for item in notes if str(item.get("status") or "").strip().lower() == "open"
        )
        summary = cls._default_summary()
        summary.update(
            {
                "note_count": len(notes),
                "open_note_count": open_note_count,
                "latest_note_at": latest_note.get("created_at") if latest_note else None,
                "latest_note_excerpt": latest_note.get("comment_excerpt") if latest_note else None,
                "recent_notes": notes[: max(int(recent_limit or 0), 0)],
            }
        )
        return summary

    @classmethod
    def build_note_index_for_records(cls, records, recent_limit=None):
        cls._ensure_table_exists()
        recent_limit = recent_limit if recent_limit is not None else cls.DEFAULT_RECENT_LIMIT
        record_keys = {}
        all_codes = set()

        for record in records or []:
            primary_code = cls._normalize_code(
                record.get("PDB Code") or record.get("pdb_code") or record.get("canonical_pdb_code")
            )
            if not primary_code:
                continue
            codes = cls._record_codes(record)
            record_keys[primary_code] = codes
            all_codes.update(codes)

        if not record_keys:
            return {}

        service = cls()
        notes = cls._query_for_codes(all_codes)
        notes_by_code = defaultdict(list)
        for note in notes:
            payload = service._serialize_note(note)
            matched_codes = {
                key
                for key in {
                    cls._normalize_code(payload.get("pdb_code")),
                    cls._normalize_code(payload.get("canonical_pdb_code")),
                }
                if key in all_codes
            }
            for code in matched_codes:
                notes_by_code[code].append(payload)

        summaries = {}
        for primary_code, codes in record_keys.items():
            seen_ids = set()
            combined = []
            for code in codes:
                for payload in notes_by_code.get(code, []):
                    note_id = payload.get("id")
                    if note_id in seen_ids:
                        continue
                    seen_ids.add(note_id)
                    combined.append(payload)
            combined.sort(
                key=lambda item: (item.get("created_at") or "", item.get("id") or 0),
                reverse=True,
            )
            summaries[primary_code] = cls._summarize_serialized_notes(
                combined,
                recent_limit=recent_limit,
            )

        return summaries

    def list_notes_for_record(self, pdb_code, limit=None):
        record = self._resolve_record(pdb_code)
        return self._build_record_payload(record, limit=limit)

    def create_note_for_record(self, pdb_code, payload, current_user):
        record = self._resolve_record(pdb_code)
        validated = self.schema.load(payload or {})
        normalized_pdb_code = self._normalize_code(
            record.get("PDB Code") or record.get("pdb_code") or pdb_code
        )
        canonical_code = self._normalize_code(record.get("canonical_pdb_code") or normalized_pdb_code)
        note = StructureExpertNote(
            pdb_code=normalized_pdb_code,
            canonical_pdb_code=canonical_code,
            title=(validated.get("title") or None),
            comment=validated["comment"].strip(),
            category=validated.get("category") or "annotation",
            status=validated.get("status") or "open",
            suggested_group=(validated.get("suggested_group") or None),
            suggested_tm_count=validated.get("suggested_tm_count"),
            source_context=validated.get("source_context"),
            created_by_user_id=getattr(current_user, "id", None),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self._ensure_table_exists()
        db.session.add(note)
        db.session.commit()
        self.invalidate_related_caches()
        record_payload = self._build_record_payload(record, limit=self.DEFAULT_LIST_LIMIT)
        latest_note = self._serialize_note(note)
        response_payload = {
            **record_payload,
            "item": latest_note,
        }
        try:
            from src.Feedbacks.realtime import broadcast_structure_note_update

            broadcast_structure_note_update(response_payload)
        except Exception as exc:
            logger.warning("Unable to broadcast structure note update: %s", exc)
        return response_payload

    @classmethod
    def invalidate_related_caches(cls):
        try:
            from src.Dashboard.services import (
                DashboardAnnotationDatasetService,
                DiscrepancyReviewService,
            )

            DashboardAnnotationDatasetService._record_payload_cache.clear()
            DiscrepancyReviewService._candidate_cache.clear()
            DiscrepancyReviewService._list_cache.clear()
            DiscrepancyReviewService._summary_cache.clear()
        except Exception as exc:
            logger.warning("Unable to clear dashboard caches after expert note update: %s", exc)
