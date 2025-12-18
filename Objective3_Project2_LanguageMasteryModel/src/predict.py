import joblib
import pandas as pd

def main():
    pipe = joblib.load("../models/language_mastery_pipeline.joblib")

    # Example “next word attempt” cases
    samples = pd.DataFrame([
        {"user_id": "u1", "word": "engine", "attempts_total": 6, "correct_rate": 0.66, "days_since_last_seen": 2, "current_streak": 3},
        {"user_id": "u1", "word": "however", "attempts_total": 5, "correct_rate": 0.40, "days_since_last_seen": 12, "current_streak": 0},
    ])

    probs = pipe.predict_proba(samples)[:, 1]
    for i, p in enumerate(probs):
        print(f"Sample {i+1} probability correct next time: {p:.3f}")

if __name__ == "__main__":
    main()
