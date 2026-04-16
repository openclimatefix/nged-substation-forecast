import json
import os


def trim_json(input_path, output_path, num_entries=1000):
    with open(input_path, "r") as f:
        data = json.load(f)

    if "data" in data:
        data["data"] = data["data"][:num_entries]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


input_dir = "data/NGED/from_sharepoint/OneDrive_1_4-8-2026/1451606400000_1774512000000/"
output_dir = "packages/nged_data/tests/data/"

files_to_trim = [
    "TimeSeries_10_20160101T003000Z_20260326T083000Z.json",
    "TimeSeries_11_20160101T003000Z_20260326T083000Z.json",
]

for file_name in files_to_trim:
    trim_json(os.path.join(input_dir, file_name), os.path.join(output_dir, file_name))
