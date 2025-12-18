import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def main():
    try:
        df = pd.read_csv("../data/mastery_generated.csv")
    except FileNotFoundError:
        df = pd.read_csv("../data/mastery_sample.csv")

    target = "is_correct_next"
    X = df.drop(columns=[target])
    y = df[target]

    cat_cols = ["user_id", "word"]
    num_cols = ["attempts_total", "correct_rate", "days_since_last_seen", "current_streak"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    model = LogisticRegression(max_iter=2000)

    pipe = Pipeline([
        ("pre", pre),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "model": "LogisticRegression (Pipeline with preprocessing)",
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": {"categorical": cat_cols, "numeric": num_cols}
    }

    joblib.dump(pipe, "../models/language_mastery_pipeline.joblib")
    with open("../models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved pipeline to ../models/language_mastery_pipeline.joblib")
    print("Saved metrics to ../models/metrics.json")
    print(metrics)

if __name__ == "__main__":
    main()
