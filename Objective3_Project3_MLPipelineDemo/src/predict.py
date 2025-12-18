import joblib
import pandas as pd

def main():
    pipeline = joblib.load("../models/pipeline_demo_model.joblib")

    new_rows = pd.DataFrame([
        {"project_type": "furniture", "hours": 10, "tools_count": 5, "wood_type": "oak", "used_finish": "yes"},
        {"project_type": "repair", "hours": 14, "tools_count": 6, "wood_type": "pine", "used_finish": "no"},
    ])

    probs = pipeline.predict_proba(new_rows)[:, 1]
    preds = pipeline.predict(new_rows)

    for i in range(len(new_rows)):
        print(f"Row {i+1} predicted on_time={preds[i]} (prob={probs[i]:.3f})")

if __name__ == "__main__":
    main()
