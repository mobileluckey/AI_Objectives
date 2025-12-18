import numpy as np
import pandas as pd

def main():
    # Synthetic time series: daily cycle + noise + small drift
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2025-12-01 00:00:00")
    periods = 12 * 24 * 7  # 7 days at 5-minute intervals
    freq = "5min"

    ts = pd.date_range(start=start, periods=periods, freq=freq)
    minutes = np.arange(periods) * 5

    daily_cycle = 2.5 * np.sin(2 * np.pi * (minutes / (60 * 24)))
    drift = 0.02 * (minutes / (60 * 24))
    noise = rng.normal(0, 0.25, size=periods)

    base = 21.0
    temp_c = base + daily_cycle + drift + noise

    df = pd.DataFrame({"timestamp": ts, "temp_c": np.round(temp_c, 2)})
    df.to_csv("../data/smart_temp_generated.csv", index=False)
    print("Wrote ../data/smart_temp_generated.csv")

if __name__ == "__main__":
    main()
