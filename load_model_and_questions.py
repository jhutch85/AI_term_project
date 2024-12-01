import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import mod
import random
import re
from flask import Flask, request, jsonify, render_template
import os
import json
import threading

model = "./ann_model.keras"
loaded_model = tf.keras.models.load_model(model)


# Initialize Flask app
app = Flask(__name__)

dataset_path = "./questions_corrected.json"

questions = []

# Load questions from dataset
if os.path.isfile(dataset_path):
    with open(dataset_path, "r") as f:
        try:
            questions_data = json.load(f)
            for idx, q in enumerate(questions_data):
                question_text = q.get("question")
                correct_answer = q.get("answer").strip(
                    "."
                )  # Remove trailing periods if present
                difficulty = q.get(
                    "difficulty", "easy"
                )  # Default to 'easy' if not specified
                questions.append(
                    {
                        "id": idx,  # Assign a unique ID to each question
                        "question": question_text,
                        "answer": correct_answer,
                        "difficulty": difficulty,
                    }
                )
            print(f"Loaded {len(questions)} questions successfully.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
else:
    # Create sample questions if dataset does not exist
    print(f"Dataset file not found at {dataset_path}, creating sample dataset.")
    questions_data = [
        {"question": "What is 2 + 2?", "answer": "4", "difficulty": "easy"},
        {"question": "Solve for x: x^2 - 4 = 0", "answer": "2", "difficulty": "medium"},
        {
            "question": "Integrate: ∫ x dx",
            "answer": "0.5 x^2 + C",
            "difficulty": "hard",
        },
    ]
    with open(dataset_path, "w") as f:
        json.dump(questions_data, f)
    # Load the sample questions
    questions = []
    for idx, q in enumerate(questions_data):
        question_text = q.get("question")
        correct_answer = q.get("answer").strip(
            "."
        )  # Remove trailing periods if present
        difficulty = q.get("difficulty", "easy")  # Default to 'easy' if not specified
        questions.append(
            {
                "id": idx,  # Assign a unique ID to each question
                "question": question_text,
                "answer": correct_answer,
                "difficulty": difficulty,
            }
        )
    print(f"Loaded {len(questions)} sample questions successfully.")


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
    options = [correct_answer]  # Include the correct answer
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
                variation = num + random.uniform(
                    -0.3 * num, 0.3 * num
                )  # Add random variation
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
        # If the answer is purely numeric, generate numeric distractors
        correct_num = float(correct_answer)
        percentage = 0.2  # Allow a 20% range for variations
        while len(distractors) < (num_options - 1):
            variation = correct_num + random.uniform(
                -percentage * correct_num, percentage * correct_num
            )
            distractors.add(
                str(int(variation)) if "." not in correct_answer else f"{variation:.2f}"
            )
    else:
        # If the answer is an expression, modify numeric parts while keeping symbolic parts intact
        distractors = set(modify_numbers_in_expression(correct_answer, num_options - 1))

    # Combine the correct answer with distractors and shuffle
    options.extend(distractors)
    random.shuffle(options)
    return options


# Function to provide hints based on difficulty
def provide_hint(difficulty):
    hints = {
        "easy": "Hint: Simplify the equation step by step.",
        "medium": "Hint: Try isolating the variable and solving for it.",
        "hard": "Hint: You may need to apply more complex algebraic rules.",
    }
    return hints.get(difficulty, "Hint: No hint available.")


# Initialize user data
user_data = {}


# Function to track user answers
def track_user_answer(user_id, question_id, selected_answer, correct_answer, used_hint):
    if user_id not in user_data:
        user_data[user_id] = {"attempts": 0, "correct": 0, "hint_count": 0}
    user_data[user_id]["attempts"] += 1
    if selected_answer == correct_answer:
        user_data[user_id]["correct"] += 1
    if used_hint:
        user_data[user_id]["hint_count"] += 1


# Function to classify learning style based on user interaction
def classify_learning_style(success_rate, hint_usage_ratio):
    if success_rate > 0.8 and hint_usage_ratio < 0.2:
        return "Independent Learner"
    elif success_rate < 0.5 and hint_usage_ratio > 0.5:
        return "Hint-Dependent"
    else:
        return "Balanced Learner"

@app.route("/")
def index():
    if not questions:
        return "No questions available."
    question = random.choice(questions)
    mcq_options = generate_mcq(question["answer"], question["question"])
    return render_template("index.html", question=question, options=mcq_options)


@app.route("/predict_learning_style", methods=["POST"])
def predict_learning_style():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid data"}), 400

    # Extract features for prediction
    features = data.get("features", [])
    if len(features) != loaded_model.input_shape[1]:
        return jsonify({"error": "Incorrect feature dimensions"}), 400

    prediction = loaded_model.predict([features])
    learning_style = "Self-guided" if prediction[0][0] > 0.5 else "Tutor-assisted"
    return jsonify({"learning_style": learning_style})


# Endpoint to get a new question
@app.route("/get_new_question", methods=["GET"])
def get_new_question():
    if not questions:
        return jsonify({"error": "No questions available"}), 500
    question = random.choice(questions)
    question_id = question["id"]
    question_text = question["question"]
    correct_answer = question["answer"]  # Fetch the correct answer
    mcq_options = generate_mcq(correct_answer, question_text)  # Updated call
    difficulty = question.get("difficulty", "easy")  # Default to 'easy' if not provided

    return jsonify(
        {
            "question_id": question_id,  # Include question ID for tracking
            "question": question_text,
            "difficulty": difficulty,
            "mcq_options": mcq_options,  # Options include the correct answer
        }
    )


@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    if request.content_type != 'application/json':
        return jsonify({"error": "Unsupported Media Type. Use 'Content-Type: application/json'"}), 415

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400

    required_fields = ["user_id", "question_id", "selected_answer"]
    missing_fields = [field for field in required_fields if not data.get(field)]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    user_id = data["user_id"]
    question_id = data["question_id"]
    selected_answer = data["selected_answer"]
    used_hint = data.get("used_hint", False)

    question = next((q for q in questions if q["id"] == question_id), None)
    if not question:
        return jsonify({"error": f"Invalid question_id: {question_id}"}), 400

    correct_answer = question["answer"]

    track_user_answer(user_id, question_id, selected_answer, correct_answer, used_hint)

    user_stats = user_data[user_id]
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

    learning_style = classify_learning_style(success_rate, hint_usage_ratio)

    return jsonify(
        {
            "message": "Answer submitted successfully",
            "user_data": user_stats,
            "learning_style": learning_style,
        }
    )


if __name__ == "__main__":
    app.run()
