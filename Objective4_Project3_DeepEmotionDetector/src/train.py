import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


def main():
    df = pd.read_csv("../data/emotion_sample.csv")

    X = df["text"].astype(str).values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=3000,
        output_mode="int",
        output_sequence_length=20
    )
    vectorizer.adapt(X_train)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorizer,
        tf.keras.layers.Embedding(input_dim=3000, output_dim=32),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=4,
        validation_data=(X_test, y_test),
        verbose=2
    )

    preds = model.predict(X_test, verbose=0).argmax(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_weighted": float(f1_score(y_test, preds, average="weighted")),
        "classes": encoder.classes_.tolist(),
    }

    print("Metrics:", metrics)

    model.save("../models/deep_emotion_detector.keras")
    print("Saved model to ../models/deep_emotion_detector.keras")

    with open("../models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save label mapping for predict.py
    with open("../models/labels.json", "w", encoding="utf-8") as f:
        json.dump(encoder.classes_.tolist(), f)

    print("Saved metrics and labels")


if __name__ == "__main__":
    main()
