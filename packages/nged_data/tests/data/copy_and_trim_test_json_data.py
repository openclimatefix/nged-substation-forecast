import json
from pathlib import Path

"""Run this from the uv workspace root."""


def trim_json(input_path: Path, output_path: Path, num_entries: int = 1000) -> None:
    with open(input_path, "r") as f:
        data = json.load(f)

    if "data" in data:
        data["data"] = data["data"][-num_entries:]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    input_dir = Path("data/NGED/from_sharepoint/OneDrive_1_4-8-2026/1451606400000_1774512000000/")
    output_dir = Path("packages/nged_data/tests/data/")

    files_to_trim = [
        "TimeSeries_10_20160101T003000Z_20260326T083000Z.json",
        "TimeSeries_11_20160101T003000Z_20260326T083000Z.json",
    ]

    for file_name in files_to_trim:
        output_path = output_dir / file_name.replace("_20160101T003000Z_20260326T083000Z", "")
        trim_json(input_dir / file_name, output_path)


if __name__ == "__main__":
    main()
