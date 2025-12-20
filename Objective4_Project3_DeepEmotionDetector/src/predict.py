import json
import tensorflow as tf


def main():
    print("Loading model...")
    model = tf.keras.models.load_model("../models/deep_emotion_detector.keras")
    print("Model loaded.")

    with open("../models/labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)

    texts = tf.constant([
        "I am so excited for this project",
        "This is really upsetting and sad",
        "I am angry about the delay",
        "It is just a normal day"
    ], dtype=tf.string)

    print("Running inference...")
    probs = model(texts, training=False).numpy()
    preds = probs.argmax(axis=1)

    for text, idx in zip(texts.numpy(), preds):
        print(f"{labels[idx]:7s} | {text.decode('utf-8')}")


if __name__ == "__main__":
    main()
