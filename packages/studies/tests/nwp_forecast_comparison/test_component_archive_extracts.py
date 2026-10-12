"""Native component extracts preserve polar consumers and reject mismatched snapshots."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import fetch_ens_day4_supplement as supplement
import fetch_ens_forecast_horizons as extract
import polars as pl
import pytest
from polars.testing import assert_frame_equal


def test_matching_extract_and_supplement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "components", tmp_path / "extract"
    start = datetime(2025, 1, 15, tzinfo=UTC)
    rows = [
        {
            "nwp_model_id": extract.NWP_MODEL_ID,
            "h3_index": 1,
            "init_time": start,
            "valid_time": start + timedelta(hours=lead),
            "ensemble_member": member,
            "temperature_2m": 10.0,
            extract.RADIATION: 100.0,
            "wind_u_10m": 0.0,
            "wind_v_10m": -5.0,
            "wind_u_100m": 3.0,
            "wind_v_100m": 4.0,
        }
        for lead in range(90, 127, 3)
        for member in range(51)
    ]
    pl.DataFrame(rows).write_delta(str(source))
    sites = pl.DataFrame({"site": ["synthetic"], "latitude": [0.0], "longitude": [0.0]})
    empty = sites.head(0)
    lookup = pl.DataFrame({"site": ["synthetic"], "h3_index": [1]})
    for script in (extract, supplement):
        monkeypatch.setattr(script, "pv_sites", lambda: sites)
        monkeypatch.setattr(script, "wind_sites", lambda: empty)
        monkeypatch.setattr(script, "_cells", lambda **kwargs: lookup)
    # Study survival logging is unrelated to the extraction/equality contract under test.
    monkeypatch.setattr(supplement, "log_surviving_runs", lambda **kwargs: None)
    assert extract.main(source=source, source_version=0, output_directory=output) == 0
    native = pl.read_parquet(output / "ens_members.parquet")
    assert native["speed_10m"].unique().to_list() == [5.0]
    assert native["direction_10m"].unique().to_list() == [0.0]
    assert native["speed_100m"].unique().to_list() == [5.0]
    assert native["direction_100m"].unique().to_list() == pytest.approx([216.8699], abs=1e-4)
    assert supplement.main(source=source, source_version=0, output_directory=output) == 0
    extra = pl.read_parquet(output / "ens_members_day4.parquet")
    combined = pl.concat([native, extra]).sort("valid_time", "ensemble_member")
    expected = extract._members(
        source=source, source_version=0, cells=[1], leads=list(range(90, 127))
    )
    expected = lookup.join(expected, on="h3_index").drop("h3_index").select(combined.columns)
    assert_frame_equal(combined, expected.sort("valid_time", "ensemble_member"), check_exact=True)


@pytest.mark.parametrize(
    ("version", "representation"), [(0, "polar"), (99, "components-derived-polar-native-v1")]
)
def test_supplement_rejects_mismatched_source(
    tmp_path: Path, version: int, representation: str
) -> None:
    source, output = tmp_path / "source", tmp_path / "extract"
    pl.DataFrame({"wind_u_10m": [0.0]}).write_delta(str(source))
    output.mkdir()
    metadata = extract.source_metadata(source=source, source_version=0)
    metadata.update(version=version, wind_representation=representation)
    (output / "ens_members.source.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="same component source snapshot"):
        supplement.main(source=source, source_version=0, output_directory=output)
    assert not (output / "ens_members_day4.parquet").exists()
