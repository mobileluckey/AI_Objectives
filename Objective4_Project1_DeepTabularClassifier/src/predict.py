import numpy as np
import tensorflow as tf
import joblib

def main():
    model = tf.keras.models.load_model("../models/deep_tabular_classifier.keras")
    scaler = joblib.load("../models/scaler.joblib")

    # Two sample cases for screenshots (edit these for "run 2")
    samples = np.array([
        [90, 7.5, 0.72, 4, 2],   # likely success
        [10, 5.5, 0.35, 0, 18],  # likely lower success
    ], dtype="float32")

    samples_s = scaler.transform(samples)

    probs = model.predict(samples_s, verbose=0).reshape(-1)
    for i, p in enumerate(probs):
        label = "success" if p >= 0.5 else "not_success"
        print(f"Sample {i+1}: {label} (p={float(p):.3f}) | raw={samples[i].tolist()}")

if __name__ == "__main__":
    main()
