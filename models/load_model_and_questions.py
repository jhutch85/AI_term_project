import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path

model = "./ann_model.keras"
loaded_model = tf.keras.models.load_model(model)
skill_build_data = "./skill_builder_data_corrected.csv"
model2 = "./current_model.keras"
loaded_model2 = tf.keras.models.load_model(model2)


if not Path(model).exists() and not Path(model2).exists():
    df = pd.read_csv(skill_build_data, encoding="latin-1")
    df["success_rate"] = df["correct"] / df["attempt_count"]
    df["hint_usage_ratio"] = df["hint_count"] / df["attempt_count"]
    df["tutor_mode_usage"] = df["tutor_mode"].apply(lambda x: 1 if x == "tutor" else 0)

    df_cleaned = df[df["attempt_count"] > 0]

    user_features = (
        df_cleaned.groupby("user_id")
        .agg(
            {
                "success_rate": "mean",
                "hint_usage_ratio": "mean",
                "tutor_mode_usage": "mean",
            }
        )
        .reset_index()
    )

    scaler = StandardScaler()
    user_features_scaled = scaler.fit_transform(
        user_features[["success_rate", "hint_usage_ratio", "tutor_mode_usage"]]
    )

    kmeans = KMeans(n_clusters=3, random_state=42)
    user_features["learning_style"] = kmeans.fit_predict(user_features_scaled)

    X = user_features[["success_rate", "hint_usage_ratio", "tutor_mode_usage"]]
    y = user_features["learning_style"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    model.save("current_model.keras")
else:
    if Path(model).exists():
        print(loaded_model.summary())
    if Path(model2).exists():
        print(loaded_model2.summary())
