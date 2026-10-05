import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from studies.trial_area import TrialAreaBox, load_trial_area_box, write_trial_area_box_from_roster

from studies import trial_area


def test_the_grid_covers_the_box_with_one_point_per_spacing_step():
    box = TrialAreaBox(lat_min=52.0, lat_max=52.2, lon_min=-1.0, lon_max=-0.8)

    points = box.grid_points(spacing_deg=0.1)

    assert points.columns == ["point_id", "latitude", "longitude"]
    assert points.height == 9
    assert points["latitude"].unique().sort().to_list() == pytest.approx([52.0, 52.1, 52.2])
    assert points["longitude"].unique().sort().to_list() == pytest.approx([-1.0, -0.9, -0.8])
    assert points["point_id"].to_list() == list(range(9))


def test_the_box_written_from_a_roster_is_widened_by_the_margin_and_read_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    metadata_path = tmp_path / "metadata.parquet"
    pl.DataFrame({"latitude": [52.0, 52.5], "longitude": [-1.0, -0.5]}).write_parquet(metadata_path)
    monkeypatch.setattr(trial_area, "WEATHER_DATA_DIR", tmp_path / "weather")
    monkeypatch.setattr(trial_area, "TRIAL_AREA_BOX_PATH", tmp_path / "weather" / "box.json")
    monkeypatch.setattr(
        "contracts.settings.get_settings", lambda: SimpleNamespace(metadata_path=metadata_path)
    )

    write_trial_area_box_from_roster(margin_deg=0.1)
    box = load_trial_area_box()

    assert json.loads((tmp_path / "weather" / "box.json").read_text()) == pytest.approx(
        {"lat_min": 51.9, "lat_max": 52.6, "lon_min": -1.1, "lon_max": -0.4}
    )
    assert (box.lat_min, box.lat_max, box.lon_min, box.lon_max) == pytest.approx(
        (51.9, 52.6, -1.1, -0.4)
    )
