import traceback
from src.Feedbacks.services import FeedbackService
from src.Training.models import FilterTool, FilterToolOption, Question, Category, Option
from database.db import db
from src.User.service import UserService
from src.Training.data import questions_data


def addDefaultAdmin():
    check_if_user_exist = UserService.get_user_by_username("Admin1").first()
    if not check_if_user_exist:
        UserService.create_user(
            "Admin admin",
            "08299938839",
            "Admin1",
            "admin01@gmail.com",
            "password",
            True,
            True,
        )
        print("Admin User has been created.")
    else:
        print("Admin has been created already.")


def addFeedbackQuestions():
    data = [
        {
            "question_text": "This tool would be useful for classifying membrane proteins into groups, and sub-groups.",
            "options": [
                {"text": "Strongly Agree", "value": 5},
                {"text": "Agree", "value": 4},
                {"text": "Neutral", "value": 3},
                {"text": "Disagree", "value": 2},
                {"text": "Strongly Disagree", "value": 1},
            ],
        }, 
        {
            "question_text": "Do you agree that this system accurately detects outliers based on the analysis of scatter plot matrix (SPLOM) ?",
            "options": [
                {"text": "Strongly Agree", "value": 5},
                {"text": "Agree", "value": 4},
                {"text": "Neutral", "value": 3},
                {"text": "Disagree", "value": 2},
                {"text": "Strongly Disagree", "value": 1},
            ],
        },
        {
            "question_text": "Do you agree that this system accurately detects outliers based on the analysis of the boxplot ?",
            "options": [
                {"text": "Strongly Agree", "value": 5},
                {"text": "Agree", "value": 4},
                {"text": "Neutral", "value": 3},
                {"text": "Disagree", "value": 2},
                {"text": "Strongly Disagree", "value": 1},
            ],
        },
        {
            "question_text": "Do you agree that this system accurately detects outliers based on the analysis of the scatter plot (outlier detection plot) using DBSCAN ?",
            "options": [
                {"text": "Strongly Agree", "value": 5},
                {"text": "Agree", "value": 4},
                {"text": "Neutral", "value": 3},
                {"text": "Disagree", "value": 2},
                {"text": "Strongly Disagree", "value": 1},
            ],
        },
        {
            "question_text": "This tool would be useful for the study of membrane proteins.",
            "options": [
                {"text": "Strongly Agree", "value": 5},
                {"text": "Agree", "value": 4},
                {"text": "Neutral", "value": 3},
                {"text": "Disagree", "value": 2},
                {"text": "Strongly Disagree", "value": 1},
            ],
        },
        {
            "question_text": "I think I would like to use this system frequently and most people would learn to use it very quickly.",
            "options": [
                {"text": "Strongly Agree", "value": 5},
                {"text": "Agree", "value": 4},
                {"text": "Neutral", "value": 3},
                {"text": "Disagree", "value": 2},
                {"text": "Strongly Disagree", "value": 1},
            ],
        },
        {
            "question_text": "I found the system unnecessarily complex and very cumbersome to use.",
            "options": [
                {"text": "Strongly Agree", "value": 5},
                {"text": "Agree", "value": 4},
                {"text": "Neutral", "value": 3},
                {"text": "Disagree", "value": 2},
                {"text": "Strongly Disagree", "value": 1},
            ],
        },
        {
            "question_text": "The system is intuitive, facilitating easy navigation to locate desired information.",
            "options": [
                {"text": "Strongly Agree", "value": 5},
                {"text": "Agree", "value": 4},
                {"text": "Neutral", "value": 3},
                {"text": "Disagree", "value": 2},
                {"text": "Strongly Disagree", "value": 1},
            ],
        },
        {
            "question_text": "I would need the support of a technical person to use this system.",
            "options": [
                {"text": "Strongly Agree", "value": 5},
                {"text": "Agree", "value": 4},
                {"text": "Neutral", "value": 3},
                {"text": "Disagree", "value": 2},
                {"text": "Strongly Disagree", "value": 1},
            ],
        },
        {
            "question_text": "I thought there was too much inconsistency in this system.",
            "options": [
                {"text": "Strongly Agree", "value": 5},
                {"text": "Agree", "value": 4},
                {"text": "Neutral", "value": 3},
                {"text": "Disagree", "value": 2},
                {"text": "Strongly Disagree", "value": 1},
            ],
        },
        {
            "question_text": "I felt confident using the system and am very satisfied with the overall user experience.",
            "options": [
                {"text": "Strongly Agree", "value": 5},
                {"text": "Agree", "value": 4},
                {"text": "Neutral", "value": 3},
                {"text": "Disagree", "value": 2},
                {"text": "Strongly Disagree", "value": 1},
            ],
        },
        {
            "question_text": "The system loads quickly.",
            "options": [
                {"text": "Strongly Agree", "value": 5},
                {"text": "Agree", "value": 4},
                {"text": "Neutral", "value": 3},
                {"text": "Disagree", "value": 2},
                {"text": "Strongly Disagree", "value": 1},
            ],
        },
        {
            "question_text": "I found it easy to interact with the charts, identify outliers using the box-plot, and use the interactive elements.",
            "options": [
                {"text": "Strongly Agree", "value": 5},
                {"text": "Agree", "value": 4},
                {"text": "Neutral", "value": 3},
                {"text": "Disagree", "value": 2},
                {"text": "Strongly Disagree", "value": 1},
            ],
        },
        {
            "question_text": "The layout and organization of graphical elements are intuitive.",
            "options": [
                {"text": "Strongly Agree", "value": 5},
                {"text": "Agree", "value": 4},
                {"text": "Neutral", "value": 3},
                {"text": "Disagree", "value": 2},
                {"text": "Strongly Disagree", "value": 1},
            ],
        },
        {
            "question_text": "What specific suggestions do you have for improving the UI/UX?",
            "options": [],  # Open-ended question, no predefined options
        },
    ]

    results = FeedbackService.update_or_create_questions(data)
    return results


def addQuestion():
    try:
        # Begin a transaction
        with db.session.begin(subtransactions=True):
            # Add questions to the database
            j = 1
            for category_data in questions_data:
                category_name = category_data["name"]
                category_desc = category_data["description"]

                # Check if the category already exists
                existing_category = Category.query.filter_by(name=category_name).first()
                if existing_category:
                    print(
                        f"Category '{category_name}' already exists. Skipping insertion."
                    )
                    category = existing_category
                else:
                    category = Category(name=category_name, description=category_desc)
                    db.session.add(category)
                    db.session.flush()  # Ensure that the category gets an ID before associating questions
                questions = category_data["questions"]
                i = 1
                for question in questions:
                    text = question["text"]
                    question_type = question["type"]
                    instruction = question.get("instruction", "")
                    hints = question.get("hints", "")
                    item_order = question.get("item_order", "")

                    # Check if the question text already exists
                    existing_question = Question.query.filter_by(
                        category=category, text=text, question_type=question_type
                    ).first()
                    if existing_question:
                        print(
                            f"Question with text '{text}' already exists. Skipping insertion."
                        )
                        continue

                    # Add the main question to the database and associate it with the category
                    main_question = Question(
                        category=category,
                        text=text,
                        item_order=item_order,
                        question_type=question_type,
                        instruction=instruction,
                        hints=hints,
                    )
                    db.session.add(main_question)
                    db.session.flush()  # Ensure that the main question gets an ID before associating options

                    # Add options to the database and associate them with the main question
                    for option_data in question.get("options", []):
                        option_is_correct = option_data["is_correct"]
                        option_text = option_data["text"]
                        option = Option(
                            question=main_question,
                            text=option_text,
                            is_correct=option_is_correct,
                        )
                        db.session.add(option)
                        db.session.flush()

                    # Add filter tools
                    filter_tools = question.get("filter_tool", [])
                    for filter_tool_data in filter_tools:
                        title = filter_tool_data.get("title", "")
                        name = filter_tool_data.get("name", "")
                        parent = filter_tool_data.get("parent", "")
                        selected_option = filter_tool_data.get("selectedOption", "")

                        filter_tool = FilterTool(
                            question=main_question,
                            title=title,
                            name=name,
                            parent=parent,
                            selected_option=selected_option,
                        )
                        db.session.add(filter_tool)
                        db.session.flush()

                        # Add options to the database and associate them with the filter_tool
                        filter_tool_options = filter_tool_data.get("options", [])
                        for filter_tool_option_data in filter_tool_options:
                            filter_tool_option_text = filter_tool_option_data["text"]
                            filter_tool_option_value = filter_tool_option_data["value"]
                            filter_tool_option = FilterToolOption(
                                filter_tool=filter_tool,
                                text=filter_tool_option_text,
                                value=filter_tool_option_value,
                            )
                            db.session.add(filter_tool_option)
                            db.session.flush()

                    i = i + 1
            j = j + 1
        # Commit the transaction
        db.session.commit()
        print("Transaction committed successfully")
        # Close the database connection
        db.session.close()
    except Exception as e:
        db.session.rollback()
        print(e)
        print(f"An unexpected error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        db.session.close()
