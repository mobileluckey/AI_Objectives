import joblib
import pandas as pd
from train import make_lag_features

def main():
    model = joblib.load("../models/smart_temp_model.joblib")

    # Use generated data if available, otherwise sample
    try:
        df = pd.read_csv("../data/smart_temp_generated.csv")
    except FileNotFoundError:
        df = pd.read_csv("../data/smart_temp_sample.csv")

    feat = make_lag_features(df)
    feature_cols = [c for c in feat.columns if c not in ("timestamp", "temp_c")]

    last_row = feat.iloc[-1]
    X = last_row[feature_cols].to_frame().T
    pred = model.predict(X)[0]

    print("Last timestamp:", last_row["timestamp"])
    print("Predicted next temp (approx):", round(float(pred), 2), "C")

if __name__ == "__main__":
    main()
