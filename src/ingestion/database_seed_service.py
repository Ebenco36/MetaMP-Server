from src.Commands.populateData import addDefaultAdmin, addFeedbackQuestions, addQuestion


class DatabaseSeedService:
    def seed_defaults(self):
        addQuestion()
        addDefaultAdmin()
        addFeedbackQuestions()
