import tensorflow as tf
from pathlib import Path
import logging
import sys
import numpy as np  # Import NumPy for array manipulation

# Optional: Suppress TensorFlow CUDA warnings if not using GPU
# Uncomment the following lines if you encounter CUDA warnings and are not using GPU
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define model paths
model1_path = "./ann_model.keras"
model2_path = "./current_model.keras"


# Function to load a model if it exists
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


# Load models
loaded_model1 = load_model_if_exists(model1_path)
loaded_model2 = load_model_if_exists(model2_path)

# Decide which model to use for testing
if loaded_model1 is not None:
    active_model = loaded_model1
    model_used = "ann_model.keras"
elif loaded_model2 is not None:
    active_model = loaded_model2
    model_used = "current_model.keras"
else:
    logger.critical("No models could be loaded. Exiting the test.")
    sys.exit(1)

logger.info(f"Using model: {model_used}")


# Function to make a prediction
def predict_learning_style(model, features):
    """
    Predicts the learning style based on input features.

    Parameters:
    - model: The loaded TensorFlow model.
    - features (list of float): Input features for prediction.

    Returns:
    - str: Predicted learning style.
    """
    try:
        # Convert features to a NumPy array with appropriate shape
        input_data = np.array([features])  # Shape: (1, 3)
        prediction = model.predict(input_data)

        # Check the model's output shape and activation
        # Assuming binary classification with sigmoid activation
        if prediction.shape[-1] == 1:
            learning_style = (
                "Self-guided" if prediction[0][0] > 0.5 else "Tutor-assisted"
            )
        else:
            # If model outputs probabilities for multiple classes
            predicted_class = np.argmax(prediction, axis=1)[0]
            learning_style = "Self-guided" if predicted_class == 0 else "Tutor-assisted"

        return learning_style
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None


# Define sample input data with three features
# Assuming the third feature is 'average_response_time'
sample_inputs = [
    {
        "success_rate": 0.9,
        "hint_usage_ratio": 0.1,
        "average_response_time": 30,
    },  # Expected: Self-guided
    {
        "success_rate": 0.4,
        "hint_usage_ratio": 0.6,
        "average_response_time": 60,
    },  # Expected: Tutor-assisted
    {
        "success_rate": 0.6,
        "hint_usage_ratio": 0.3,
        "average_response_time": 45,
    },  # Expected: Self-guided or Balanced based on model
]

# Perform predictions
for idx, input_data in enumerate(sample_inputs, start=1):
    features = [
        input_data["success_rate"],
        input_data["hint_usage_ratio"],
        input_data["average_response_time"],
    ]
    predicted_style = predict_learning_style(active_model, features)
    if predicted_style:
        logger.info(
            f"Test Case {idx}: Features = {features} => Predicted Learning Style: {predicted_style}"
        )
    else:
        logger.error(f"Test Case {idx}: Prediction failed for features {features}")
