import pandas as pd
import pathlib

# Paths
base_path = pathlib.Path("data/NGED/delta/raw_power_time_series")
ids_to_check = [1, 8]

for ts_id in ids_to_check:
    print(f"\nChecking time_series_id={ts_id}")
    file_path = base_path / f"time_series_id={ts_id}"

    # Read all parquet files in the partition
    df = pd.read_parquet(file_path)

    # Check if 'timestamp' and 'power_mw' columns exist (adjusting based on typical schemas)
    # I should verify column names first. Let's just list the columns first.
    if "timestamp" in df.columns:
        # Filter for midnight: hour == 0
        # Assuming timestamp is datetime64
        df["hour"] = df["timestamp"].dt.hour
        midnight_data = df[df["hour"] == 0]

        print(f"Total samples at midnight: {len(midnight_data)}")

        # Check for zero power_mw
        if "power_mw" in df.columns:
            zeros = midnight_data[midnight_data["power_mw"] == 0]
            print(f"Zero values found at midnight: {len(zeros)}")
            if len(zeros) > 0:
                print("Sample of zero values:")
                print(zeros.head())
        else:
            print("Column 'power_mw' not found. Available columns:", df.columns)
    else:
        print("Column 'timestamp' not found. Available columns:", df.columns)
