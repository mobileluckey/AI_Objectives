"""
Objective 5 - Project 3: Explainable AI Demo

Goal:
- Demonstrate deep understanding of *how models make decisions* by explaining predictions.
- Train a model (RandomForest) and compute *Permutation Feature Importance* to show
  which inputs drive performance.

Why this fits Objective 5:
- Neural networks and ML systems are often "black boxes".
- Explainable AI makes them more transparent, trustworthy, and debuggable.
- Feature importance is one industry-friendly way to communicate model reasoning.

Outputs (saved to /models):
- feature_importance_top10.png  (bar chart proof)
- feature_importance_top10.csv  (table proof)
- metrics.json                  (accuracy + dataset size proof)
"""

import os
import json

import numpy as np
import pandas as pd

# Use a non-interactive backend so scripts don't hang in VS Code / terminals.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score


def ensure_models_dir() -> str:
    """
    Create (if needed) and return the absolute path to the project's /models folder.
    This keeps outputs inside the repo so you can commit screenshots/plots easily.
    """
    here = os.path.dirname(__file__)  # .../src
    models_dir = os.path.abspath(os.path.join(here, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def main():
    print("=" * 60)
    print("Objective 5 - Project 3: Explainable AI Demo")
    print("Model: RandomForest + Permutation Feature Importance")
    print("=" * 60)

    models_dir = ensure_models_dir()

    # 1) Load a real, built-in dataset (no CSV file headaches).
    #    Breast cancer dataset: numeric features, binary label.
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="label")

    # 2) Train/test split with stratify to keep class balance similar in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # 3) Train a strong baseline model.
    #    RandomForest is not a neural net, but it is a classic "intelligent system"
    #    and is frequently used in real industry settings.
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 4) Evaluate accuracy (simple, understandable metric).
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Dataset rows: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Test Accuracy: {acc:.4f}")

    # 5) Explainability: Permutation Feature Importance
    #    Idea: shuffle one feature column at a time.
    #    If accuracy drops a lot, that feature was important to the model.
    print("\nComputing permutation importance (this may take a moment)...")
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=15,        # more repeats = more stable importance estimates
        random_state=42,
        n_jobs=-1
    )

    importances = pd.Series(
        result.importances_mean,
        index=X.columns
    ).sort_values(ascending=False)

    top10 = importances.head(10)

    print("\nTop 10 important features (higher = more influence on decisions):")
    for name, score in top10.items():
        print(f" - {name}: {score:.6f}")

    # 6) Save artifacts to /models for GitHub + website proof.
    csv_path = os.path.join(models_dir, "feature_importance_top10.csv")
    png_path = os.path.join(models_dir, "feature_importance_top10.png")
    metrics_path = os.path.join(models_dir, "metrics.json")

    top10.to_csv(csv_path, header=["importance_mean"])

    plt.figure(figsize=(10, 5))
    top10.sort_values().plot(kind="barh")  # horizontal bar chart reads better for long names
    plt.title("Permutation Feature Importance (Top 10)")
    plt.xlabel("Importance (mean decrease in score)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    metrics = {
        "model": "RandomForestClassifier",
        "test_accuracy": float(acc),
        "rows_total": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "explainability_method": "permutation_importance",
        "saved_files": {
            "plot_png": "models/feature_importance_top10.png",
            "top10_csv": "models/feature_importance_top10.csv"
        }
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved outputs:")
    print(f" - {png_path}")
    print(f" - {csv_path}")
    print(f" - {metrics_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
