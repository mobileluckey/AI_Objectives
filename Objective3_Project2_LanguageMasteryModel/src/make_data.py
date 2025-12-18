import numpy as np
import pandas as pd

def main():
    rng = np.random.default_rng(13)
    users = [f"u{i}" for i in range(1, 21)]
    words = ["apple","river","engine","quiet","build","however","practice","teacher","signal","future",
             "system","lesson","memory","network","sensor","predict","vocab","command","meaning","device"]

    rows = []
    for _ in range(600):
        user = rng.choice(users)
        word = rng.choice(words)

        attempts = int(rng.integers(1, 15))
        correct_rate = float(np.clip(rng.normal(0.65, 0.2), 0.0, 1.0))
        days_since = int(np.clip(rng.normal(4, 4), 0, 30))
        streak = int(np.clip(rng.normal(2, 2), 0, 10))

        # Probability of being correct next time:
        # better correct_rate + lower days_since + higher streak
        p_correct = 0.15 + (0.65 * correct_rate) + (0.03 * streak) - (0.02 * days_since)
        p_correct = float(np.clip(p_correct, 0.02, 0.98))
        is_correct_next = int(rng.random() < p_correct)

        rows.append([user, word, attempts, round(correct_rate, 2), days_since, streak, is_correct_next])

    df = pd.DataFrame(rows, columns=[
        "user_id","word","attempts_total","correct_rate","days_since_last_seen","current_streak","is_correct_next"
    ])
    df.to_csv("../data/mastery_generated.csv", index=False)
    print("Wrote ../data/mastery_generated.csv")

if __name__ == "__main__":
    main()
