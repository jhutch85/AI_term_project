import os
import json
import uuid
import random
import re
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, session, redirect, url_for

# Optional: Suppress TensorFlow warnings (adjust as needed)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO and WARNING messages

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to your TensorFlow models
MODEL_PATH_ANN = "./models/ann_model.keras"
MODEL_PATH_CURRENT = "./models/current_model.keras"


# Load TensorFlow models
def load_model_if_exists(model_path):
    if Path(model_path).is_file():
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None
    else:
        logger.warning(f"Model file {model_path} does not exist.")
        return None


loaded_model_ann = load_model_if_exists(MODEL_PATH_ANN)
loaded_model_current = load_model_if_exists(MODEL_PATH_CURRENT)

# Decide which model to use
if loaded_model_ann is not None:
    active_model = loaded_model_ann
    model_used = "ann_model.keras"
elif loaded_model_current is not None:
    active_model = loaded_model_current
    model_used = "current_model.keras"
else:
    logger.critical("No models could be loaded. Exiting the application.")
    exit(1)

logger.info(f"Using model: {model_used}")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY", os.urandom(24))

DATASET_PATH = "./questions_corrected.json"

questions = []


def load_questions():
    global questions
    if Path(DATASET_PATH).is_file():
        with open(DATASET_PATH, "r") as f:
            try:
                questions_data = json.load(f)
                for idx, q in enumerate(questions_data):
                    question_text = q.get("question")
                    correct_answer = q.get("answer").strip(".")
                    difficulty = q.get("difficulty", "easy")
                    questions.append(
                        {
                            "id": idx,
                            "question": question_text,
                            "answer": correct_answer,
                            "difficulty": difficulty,
                        }
                    )
                logger.info(f"Loaded {len(questions)} questions successfully.")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON: {e}")
    else:
        logger.warning(
            f"Dataset file not found at {DATASET_PATH}, creating sample dataset."
        )
        questions_data = [
            {"question": "What is 2 + 2?", "answer": "4", "difficulty": "easy"},
            {
                "question": "Solve for x: x^2 - 4 = 0",
                "answer": "2",
                "difficulty": "medium",
            },
            {
                "question": "Integrate: ∫ x dx",
                "answer": "0.5 x^2 + C",
                "difficulty": "hard",
            },
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "difficulty": "easy",
            },
            {
                "question": "What is the derivative of sin(x)?",
                "answer": "cos(x)",
                "difficulty": "medium",
            },
            {
                "question": "Solve the integral ∫ e^x dx",
                "answer": "e^x + C",
                "difficulty": "hard",
            },
            {
                "question": "What is the chemical symbol for Gold?",
                "answer": "Au",
                "difficulty": "easy",
            },
            {
                "question": "What is the value of Pi up to two decimal places?",
                "answer": "3.14",
                "difficulty": "medium",
            },
            {
                "question": "What is the powerhouse of the cell?",
                "answer": "Mitochondria",
                "difficulty": "hard",
            },
            {
                "question": "What year did World War II end?",
                "answer": "1945",
                "difficulty": "easy",
            },
        ]
        with open(DATASET_PATH, "w") as f:
            json.dump(questions_data, f)
        # Load the sample questions
        for idx, q in enumerate(questions_data):
            question_text = q.get("question")
            correct_answer = q.get("answer").strip(".")
            difficulty = q.get("difficulty", "easy")
            questions.append(
                {
                    "id": idx,
                    "question": question_text,
                    "answer": correct_answer,
                    "difficulty": difficulty,
                }
            )
        logger.info(f"Loaded {len(questions)} sample questions successfully.")


# Load questions at startup
load_questions()


# Function to generate multiple-choice questions (MCQs)
def generate_mcq(correct_answer, question_text, num_options=4):
    """
    Generate multiple-choice options including the correct answer.
    Ensures the correct answer is included and distractors are plausible.

    Parameters:
    - correct_answer (str): The correct answer to the question.
    - question_text (str): The text of the question, used to generate context-specific distractors.
    - num_options (int): The total number of options to generate.

    Returns:
    - List[str]: A list of answer options shuffled randomly.
    """
    options = [correct_answer]
    distractors = set()

    # Helper function to modify numeric values in the answer
    def modify_numbers_in_expression(expression, num_variations=3):
        matches = re.findall(r"-?\d+(?:\.\d+)?", expression)
        if not matches:
            return [expression]  # If no numeric component, return the original

        modified_expressions = set()
        while len(modified_expressions) < num_variations:
            modified_expression = expression
            for match in matches:
                num = float(match)
                variation = num + random.uniform(-0.3 * num, 0.3 * num)
                variation_str = (
                    f"{variation:.2f}" if "." in match else str(int(variation))
                )
                modified_expression = re.sub(
                    re.escape(match), variation_str, modified_expression, count=1
                )
            if modified_expression != expression:
                modified_expressions.add(modified_expression)
        return list(modified_expressions)

    # Generate distractors based on correct answer
    if correct_answer.replace(".", "", 1).replace("-", "", 1).isdigit():
        correct_num = float(correct_answer)
        percentage = 0.2
        while len(distractors) < (num_options - 1):
            variation = correct_num + random.uniform(
                -percentage * correct_num, percentage * correct_num
            )
            distractors.add(
                str(int(variation)) if "." not in correct_answer else f"{variation:.2f}"
            )
    else:
        distractors = set(modify_numbers_in_expression(correct_answer, num_options - 1))

    options.extend(distractors)
    random.shuffle(options)
    return options


def provide_hint(difficulty):
    hints = {
        "easy": "Simplify the equation step by step.",
        "medium": "Try isolating the variable and solving for it.",
        "hard": "You may need to apply more complex algebraic rules.",
    }
    return hints.get(difficulty, "No hint available.")


user_data = {}


def track_user_answer(
    user_id, question_id, selected_answer, correct_answer, used_hint, response_time
):
    if user_id not in user_data:
        user_data[user_id] = {
            "attempts": 0,
            "correct": 0,
            "hint_count": 0,
            "total_response_time": 0,
        }
    user_data[user_id]["attempts"] += 1
    if selected_answer == correct_answer:
        user_data[user_id]["correct"] += 1
    if used_hint:
        user_data[user_id]["hint_count"] += 1
    user_data[user_id]["total_response_time"] += response_time


def classify_learning_style(success_rate, hint_usage_ratio):
    if success_rate > 0.8 and hint_usage_ratio < 0.2:
        return "Independent Learner"
    elif success_rate < 0.5 and hint_usage_ratio > 0.5:
        return "Hint-Dependent"
    else:
        return "Balanced Learner"


def predict_learning_style_helper(features):
    if len(features) != active_model.input_shape[1]:
        logger.error(
            f"Incorrect number of features. Expected {active_model.input_shape[1]}, got {len(features)}"
        )
        return None  # Handle incorrect feature dimensions

    try:
        input_data = np.array([features], dtype=np.float32)  # Shape: (1, 3)

        prediction = active_model.predict(input_data)
        if prediction.shape[-1] == 1:
            learning_style = (
                "Self-guided" if prediction[0][0] > 0.5 else "Tutor-assisted"
            )
        else:
            predicted_class = np.argmax(prediction, axis=1)[0]
            class_labels = [
                "Self-guided",
                "Tutor-assisted",
                "Other",
            ]
            learning_style = (
                class_labels[predicted_class]
                if predicted_class < len(class_labels)
                else "Unknown"
            )

        logger.info(f"Prediction result: {learning_style}")
        return {"learning_style": learning_style}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None


@app.route("/")
def home():
    return redirect(url_for("start_quiz"))


# Route: Start Quiz
@app.route("/start_quiz", methods=["GET"])
def start_quiz():
    session["user_id"] = str(uuid.uuid4())
    session["current_question"] = 0
    session["answers"] = []

    if len(questions) < 10:
        return "Not enough questions available to start the quiz."

    session["quiz_questions"] = random.sample(questions, 10)
    return redirect(url_for("question"))


@app.route("/question", methods=["GET"])
def question():
    if "current_question" not in session or "quiz_questions" not in session:
        return redirect(url_for("start_quiz"))

    current = session["current_question"]
    if current >= 10:
        return redirect(url_for("results"))

    question = session["quiz_questions"][current]
    mcq_options = generate_mcq(question["answer"], question["question"])
    hint = provide_hint(question["difficulty"])

    return render_template(
        "question.html",
        question_number=current + 1,
        question=question,
        options=mcq_options,
        hint=hint,
    )


@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    if not request.is_json and not request.form:
        return (
            jsonify(
                {
                    "error": "Unsupported Media Type. Use 'Content-Type: application/json' or form data"
                }
            ),
            415,
        )

    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "User session not found."}), 400

    if request.form:
        data = request.form
        used_hint = data.get("used_hint", "false").lower() == "true"
        try:
            response_time = float(data.get("response_time", 0))
        except ValueError:
            response_time = 0.0
    else:
        data = request.get_json()
        used_hint = data.get("used_hint", False)
        try:
            response_time = float(data.get("response_time", 0))
        except ValueError:
            response_time = 0.0

    required_fields = ["question_id", "selected_answer", "response_time"]
    missing_fields = [
        field for field in required_fields if data.get(field) in [None, ""]
    ]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    question_id = int(data["question_id"])
    selected_answer = data["selected_answer"]

    # Retrieve the current question
    current = session.get("current_question", 0)
    if current >= 10:
        return redirect(url_for("results"))

    question = session["quiz_questions"][current]
    correct_answer = question["answer"]

    session["answers"].append(
        {
            "question_id": question_id,
            "selected_answer": selected_answer,
            "correct_answer": correct_answer,
            "used_hint": used_hint,
            "response_time": response_time,
        }
    )

    track_user_answer(
        user_id, question_id, selected_answer, correct_answer, used_hint, response_time
    )

    session["current_question"] = current + 1

    if session["current_question"] >= 10:
        return redirect(url_for("results"))
    else:
        return redirect(url_for("question"))


@app.route("/results", methods=["GET"])
def results():
    if "answers" not in session or len(session["answers"]) < 10:
        return redirect(url_for("question"))

    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "User session not found."}), 400

    user_stats = user_data.get(
        user_id,
        {"attempts": 0, "correct": 0, "hint_count": 0, "total_response_time": 0},
    )
    success_rate = (
        user_stats["correct"] / user_stats["attempts"]
        if user_stats["attempts"] > 0
        else 0
    )
    hint_usage_ratio = (
        user_stats["hint_count"] / user_stats["attempts"]
        if user_stats["attempts"] > 0
        else 0
    )
    average_response_time = (
        user_stats["total_response_time"] / user_stats["attempts"]
        if user_stats["attempts"] > 0
        else 0
    )

    features = [success_rate, hint_usage_ratio, average_response_time]

    prediction_result = predict_learning_style_helper(features)
    if not prediction_result:
        return jsonify({"error": "Prediction model failed"}), 500

    session.pop("current_question", None)
    session.pop("answers", None)
    session.pop("quiz_questions", None)
    session.pop("user_id", None)

    return render_template(
        "results.html", learning_style=prediction_result["learning_style"]
    )


@app.route("/predict_learning_style_api", methods=["POST"])
def predict_learning_style_api():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400

    features = data.get("features", [])
    if len(features) != active_model.input_shape[1]:
        return jsonify({"error": "Incorrect feature dimensions"}), 400

    prediction_result = predict_learning_style_helper(features)
    if not prediction_result:
        return jsonify({"error": "Prediction model failed"}), 500

    return jsonify({"learning_style": prediction_result["learning_style"]})


if __name__ == "__main__":
    app.run(debug=True)
