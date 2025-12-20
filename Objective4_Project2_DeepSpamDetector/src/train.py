import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def main():
    df = pd.read_csv("../data/spam_sample.csv")
    df["y"] = (df["label"].str.lower() == "spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].astype(str).values, df["y"].values,
        test_size=0.33, random_state=42, stratify=df["y"].values
    )

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=2000,
        output_mode="int",
        output_sequence_length=30
    )
    vectorizer.adapt(X_train)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorizer,
        tf.keras.layers.Embedding(2000, 32),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=25, batch_size=4, validation_data=(X_test, y_test), verbose=2)

    probs = model.predict(X_test).reshape(-1)
    preds = (probs >= 0.5).astype(int)

    print("accuracy:", float(accuracy_score(y_test, preds)))
    print("f1:", float(f1_score(y_test, preds)))

    model.save("../models/deep_spam_detector.keras")
    print("Saved model to ../models/deep_spam_detector.keras")

if __name__ == "__main__":
    main()
