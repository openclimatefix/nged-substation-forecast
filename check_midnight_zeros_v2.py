import pandas as pd
import pathlib

# Paths
base_path = pathlib.Path("data/NGED/delta/raw_power_time_series")
ids_to_check = [1, 8]

for ts_id in ids_to_check:
    print(f"\nChecking time_series_id={ts_id}")
    # Read the parquet files directly
    df = pd.read_parquet(base_path / f"time_series_id={ts_id}")

    # Check for midnight
    # 'period_end_time' is likely a datetime column.
    if "period_end_time" in df.columns:
        df["hour"] = pd.to_datetime(df["period_end_time"]).dt.hour
        midnight_data = df[df["hour"] == 0]

        print(f"Total samples at midnight: {len(midnight_data)}")

        # Check for zero power
        if "power" in df.columns:
            zeros = midnight_data[midnight_data["power"] == 0]
            print(f"Zero values found at midnight: {len(zeros)}")
            if len(zeros) > 0:
                print("Sample of zero values at midnight:")
                print(zeros.head())
        else:
            print("Column 'power' not found. Available columns:", df.columns)
    else:
        print("Column 'period_end_time' not found. Available columns:", df.columns)
