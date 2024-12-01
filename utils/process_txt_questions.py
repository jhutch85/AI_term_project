import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--txt_file")
parser.add_argument("-j", "--json_file")
parser.add_argument("-d", "--difficulty")


def process_and_add_to_json(filepath, json_path, difficulty):
    """
    Processes a text file containing questions and answers and appends them to a JSON file.

    Args:
        filepath (str): Path to the text file.
        json_path (str): Path to the JSON file.
        difficulty (str): Difficulty level to add to each entry.
    """
    data = []

    try:
        # Read the text file
        with open(filepath, "r") as file:
            lines = file.readlines()

        # Process questions and answers
        for i in range(0, len(lines), 2):
            question = lines[i].strip()
            answer = lines[i + 1].strip() if i + 1 < len(lines) else None
            if question and answer:  # Ensure both are valid
                entry = {
                    "question": question,
                    "answer": answer,
                    "difficulty": difficulty,
                }
                data.append(entry)

        # Load existing JSON data
        try:
            with open(json_path, "r") as json_file:
                existing_data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        # Append new data to the existing data
        existing_data.extend(data)

        # Write back to the JSON file
        with open(json_path, "w") as json_file:
            json.dump(existing_data, json_file, indent=4)

        print(f"Successfully processed and appended data to {json_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")


args = parser.parse_args()

process_and_add_to_json(args.txt_file, args.json_file, args.difficulty)
