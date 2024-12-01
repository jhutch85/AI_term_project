import json
import random
import argparse
import os
import sys

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Randomly sample a subset of questions from a JSON file.")
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input JSON file containing all questions."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to the output JSON file to save the sampled questions."
    )
    parser.add_argument(
        "-f", "--fraction",
        type=float,
        default=1/3,
        help="Fraction of questions to sample (default is 1/3). Must be between 0 and 1."
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)."
    )
    return parser.parse_args()

def load_questions(input_path):
    """
    Loads questions from the input JSON file.
    
    Parameters:
        input_path (str): Path to the input JSON file.
    
    Returns:
        list: A list of question dictionaries.
    """
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)
    
    try:
        with open(input_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            if not isinstance(data, list):
                print(f"Error: JSON structure is not a list of questions.")
                sys.exit(1)
            return data
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading '{input_path}'. {e}")
        sys.exit(1)

def sample_questions(questions, fraction, seed=None):
    """
    Samples a subset of questions based on the specified fraction.
    
    Parameters:
        questions (list): List of question dictionaries.
        fraction (float): Fraction of questions to sample (between 0 and 1).
        seed (int, optional): Seed for the random number generator.
    
    Returns:
        list: A list of sampled question dictionaries.
    """
    if not (0 < fraction <= 1):
        print("Error: Fraction must be between 0 (exclusive) and 1 (inclusive).")
        sys.exit(1)
    
    total_questions = len(questions)
    sample_size = int(total_questions * fraction)
    
    # Handle edge cases
    if sample_size == 0:
        print("Warning: Fraction too small. At least one question will be selected.")
        sample_size = 1
    elif sample_size > total_questions:
        print(f"Warning: Fraction too large. Selecting all {total_questions} questions.")
        sample_size = total_questions
    
    if seed is not None:
        random.seed(seed)
    
    sampled = random.sample(questions, sample_size)
    return sampled

def save_sampled_questions(sampled_questions, output_path):
    """
    Saves the sampled questions to the output JSON file.
    
    Parameters:
        sampled_questions (list): List of sampled question dictionaries.
        output_path (str): Path to the output JSON file.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(sampled_questions, file, indent=4, ensure_ascii=False)
        print(f"Successfully saved {len(sampled_questions)} questions to '{output_path}'.")
    except Exception as e:
        print(f"Error: Failed to write to '{output_path}'. {e}")
        sys.exit(1)

def main():
    args = parse_arguments()
    
    # Load all questions
    all_questions = load_questions(args.input)
    total = len(all_questions)
    print(f"Total questions loaded: {total}")
    
    # Sample questions
    sampled = sample_questions(all_questions, args.fraction, args.seed)
    print(f"Number of questions sampled: {len(sampled)}")
    
    # Save sampled questions
    save_sampled_questions(sampled, args.output)

if __name__ == "__main__":
    main()
