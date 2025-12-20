import numpy as np
import pandas as pd

def main():
    rng = np.random.default_rng(42)

    rows = []
    # 2000 simulated students / sessions
    for _ in range(2000):
        study_minutes = int(rng.integers(0, 180))
        sleep_hours = float(np.clip(rng.normal(7.0, 1.2), 3.5, 10.0))
        past_accuracy = float(np.clip(rng.normal(0.65, 0.18), 0.05, 0.98))
        streak = int(np.clip(rng.normal(2.5, 2.0), 0, 12))
        days_since_last = int(np.clip(rng.normal(4.0, 4.0), 0, 30))

        # Probability of passing/doing well next lesson (synthetic but realistic-ish)
        score = (
            0.015 * study_minutes +
            0.25 * sleep_hours +
            2.5 * past_accuracy +
            0.12 * streak -
            0.04 * days_since_last +
            rng.normal(0.0, 0.6)
        )
        p = 1 / (1 + np.exp(-(score - 3.0)))  # logistic squish
        success = int(rng.random() < p)

        rows.append([study_minutes, sleep_hours, past_accuracy, streak, days_since_last, success])

    df = pd.DataFrame(rows, columns=[
        "study_minutes", "sleep_hours", "past_accuracy", "streak", "days_since_last", "success_next"
    ])

    df.to_csv("../data/student_success.csv", index=False)
    print("Wrote ../data/student_success.csv")

if __name__ == "__main__":
    main()
