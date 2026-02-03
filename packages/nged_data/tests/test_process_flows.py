from pathlib import Path

import pytest
from nged_data.process_flows import process_live_primary_substation_flows

EXAMPLE_DATA_DIR = Path(__file__).parent.parent / "example_csv_data"


@pytest.mark.parametrize(
    "csv_filename",
    [
        "aberaeron-primary-transformer-flows.csv",
        "abington-primary-transformer-flows.csv",
        "albrighton-11kv-primary-transformer-flows.csv",
        "filton-dc-primary-transformer-flows.csv",
        "regent-street.csv",
        "milford-haven-grid.csv",
    ],
)
def test_primary_substation_csv_to_dataframe(csv_filename: str):
    csv_path = EXAMPLE_DATA_DIR / csv_filename

    df = process_live_primary_substation_flows(csv_path)

    assert df.height > 0
    assert "timestamp" in df.columns
    assert "MW" in df.columns or "MVA" in df.columns
