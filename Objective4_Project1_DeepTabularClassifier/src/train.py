import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def main():
    try:
        df = pd.read_csv("../data/student_success.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Run make_data.py first to create ../data/student_success.csv")

    y = df["success_next"].astype(int).values
    X = df.drop(columns=["success_next"]).astype("float32").values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Best practice: scale numeric inputs for neural nets
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train_s.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )

    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_test_s, y_test),
        epochs=12,
        batch_size=64,
        verbose=2
    )

    probs = model.predict(X_test_s, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "model": "DeepTabularClassifier",
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": ["study_minutes", "sleep_hours", "past_accuracy", "streak", "days_since_last"],
    }

    print("Metrics:", metrics)

    # Save model + scaler
    model.save("../models/deep_tabular_classifier.keras")
    print("Saved model to ../models/deep_tabular_classifier.keras")

    import joblib
    joblib.dump(scaler, "../models/scaler.joblib")
    print("Saved scaler to ../models/scaler.joblib")

    with open("../models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics to ../models/metrics.json")


if __name__ == "__main__":
    main()
