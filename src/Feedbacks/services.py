from __future__ import annotations

from datetime import datetime

from marshmallow import ValidationError

from database.db import db
from src.Feedbacks.models import Feedback, FeedbackOption, FeedbackQuestion
from src.Feedbacks.serializers import FeedbackQuestionSchema, FeedbackSchema


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
