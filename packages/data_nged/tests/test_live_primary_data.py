from pathlib import Path

import polars as pl
import pytest
from data_nged.live_primary_data import SubstationLocations, __read_primary_substation_csv

EXAMPLE_DATA_DIR = Path(__file__).parent.parent / "example_csv_data"


@pytest.mark.parametrize(
    "csv_filename",
    [
        "aberaeron-primary-transformer-flows.csv",
        "abington-primary-transformer-flows.csv",
        "albrighton-11kv-primary-transformer-flows.csv",
        "filton-dc-primary-transformer-flows.csv",
    ],
)
def test_primary_substation_csv_to_dataframe(csv_filename: str):
    csv_path = EXAMPLE_DATA_DIR / csv_filename
    substation_name = csv_filename.split("-")[0].capitalize()

    df = __read_primary_substation_csv(csv_path, substation_name=substation_name)

    assert df.height > 0
    assert "substation_name" in df.columns
    assert "timestamp" in df.columns
    assert (df["substation_name"] == substation_name).all()

    # Check that it has the expected columns from SubstationFlows
    assert "MW" in df.columns or "MVA" in df.columns


def test_substation_locations_csv_validation():
    csv_path = EXAMPLE_DATA_DIR / "primary_substation_locations.csv"
    df = pl.read_csv(csv_path)

    # Manually apply renames as download_substation_locations would
    df = df.rename(
        {
            "Substation Name": "substation_name",
            "Latitude": "latitude",
            "Longitude": "longitude",
        }
    )
    df = df.cast(
        {
            "latitude": pl.Float32,
            "longitude": pl.Float32,
        }
    )

    validated_df = SubstationLocations.validate(df, drop_superfluous_columns=True)
    validated_df = validated_df.select(SubstationLocations.columns)

    assert validated_df.height > 0
    assert set(validated_df.columns) == set(SubstationLocations.columns)
    assert validated_df.schema["latitude"] == pl.Float32
    assert validated_df.schema["longitude"] == pl.Float32
