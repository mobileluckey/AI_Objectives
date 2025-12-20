import tensorflow as tf

def main():
    print("Loading model...")
    model = tf.keras.models.load_model("../models/deep_spam_detector.keras")
    print("Model loaded.")

    messages = tf.constant([
        "Free winner! claim your reward now",
        "Are you coming to the meeting tomorrow?",
        "URGENT: verify your account now to unlock access",
        "Can you pick up milk on the way home?"
    ], dtype=tf.string)
    print("Messages ready.")

    print("Running inference...")
    probs = model(messages, training=False).numpy().reshape(-1)  # direct call avoids predict() adapter weirdness
    print("Inference done.")

    for msg, p in zip(messages.numpy(), probs):
        msg_str = msg.decode("utf-8")
        label = "spam" if p >= 0.5 else "ham"
        print(f"{label:4s} (p={float(p):.3f}) | {msg_str}")

if __name__ == "__main__":
    main()
