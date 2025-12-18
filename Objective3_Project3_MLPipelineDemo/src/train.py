import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

def main():
    df = pd.read_csv("../data/pipeline_demo_sample.csv")

    target = "on_time"
    X = df.drop(columns=[target])
    y = df[target]

    cat_cols = ["project_type", "wood_type", "used_finish"]
    num_cols = ["hours", "tools_count"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

    model = GradientBoostingClassifier(random_state=42)

    pipeline = Pipeline([
        ("pre", pre),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "model": "GradientBoostingClassifier (full Pipeline)",
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "categorical_features": cat_cols,
        "numeric_features": num_cols
    }

    joblib.dump(pipeline, "../models/pipeline_demo_model.joblib")
    with open("../models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved pipeline to ../models/pipeline_demo_model.joblib")
    print("Saved metrics to ../models/metrics.json")
    print(metrics)

if __name__ == "__main__":
    main()
