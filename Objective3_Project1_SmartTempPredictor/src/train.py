import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def make_lag_features(df: pd.DataFrame, lags=(1,2,3,6,12), rolling=(3,6,12)) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    # Lag features (5-min steps)
    for lag in lags:
        df[f"lag_{lag}"] = df["temp_c"].shift(lag)

    # Rolling stats
    for w in rolling:
        df[f"roll_mean_{w}"] = df["temp_c"].rolling(window=w).mean()
        df[f"roll_std_{w}"] = df["temp_c"].rolling(window=w).std()

    df = df.dropna().reset_index(drop=True)
    return df

def time_split(df: pd.DataFrame, train_frac=0.8):
    split_idx = int(len(df) * train_frac)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test

def main():
    data_path = "../data/smart_temp_generated.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        df = pd.read_csv("../data/smart_temp_sample.csv")

    feat = make_lag_features(df)
    train, test = time_split(feat)

    feature_cols = [c for c in feat.columns if c not in ("timestamp", "temp_c")]
    X_train, y_train = train[feature_cols], train["temp_c"]
    X_test, y_test = test[feature_cols], test["temp_c"]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    metrics = {
        "model": "RandomForestRegressor",
        "mae": mae,
        "rmse": rmse,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": feature_cols
    }

    joblib.dump(model, "../models/smart_temp_model.joblib")
    with open("../models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to ../models/smart_temp_model.joblib")
    print("Saved metrics to ../models/metrics.json")
    print(metrics)

if __name__ == "__main__":
    main()
