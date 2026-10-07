"""Tests for `studies/weather_downloads/fetch_cams_public_points.py`. No test touches the network.

Each test is built to fail on the bug it exists for: two BMUs sharing a coordinate requested twice,
a multi-site BMU leaking into the point list, a period-start timestamp where the period end belongs,
and a finished or truncated download judged wrongly on resume.
"""

import importlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Final

import polars as pl
import pytest

CSV_TEXT: Final[str] = (
    "# Latitude (positive North): 51.338767\n"
    "# Longitude (positive East): 0.913885\n"
    "# Observation period;TOA;Clear sky GHI;Clear sky BHI;Clear sky DHI;Clear sky BNI;GHI;BHI;DHI;"
    "BNI;Reliability\n"
    "2025-01-01T00:00:00.0/2025-01-01T01:00:00.0;0;0;0;0;0;0;0;0;0;1\n"
    "2025-01-01T11:00:00.0/2025-01-01T12:00:00.0;300;250.5;150;100.5;400;120.25;40;80.25;100;0.75\n"
)


def _load() -> ModuleType:
    """Import the study script by name, from the study folder pytest puts on `sys.path`."""
    return importlib.import_module("fetch_cams_public_points")


fetch = _load()


def _census() -> pl.DataFrame:
    """Return a small census: a shared-coordinate pair, a lone BMU, and a multi-site BMU."""
    return pl.DataFrame(
        {
            "elexon_bmu_id": ["T_CLVHS-2", "T_CLVHS-1", "T_BURWS-1", "T_MULTI-1"],
            "scope": ["single-site", "single-site", "single-site", "multi-site"],
            "latitude": [51.5, 51.5, 52.25, 53.0],
            "longitude": [0.5, 0.5, 0.25, 1.0],
        }
    )


def test_shared_coordinate_is_one_point_labelled_with_both_bmus() -> None:
    points = fetch.bmu_points(census=_census())

    assert [point.point_id for point in points] == ["bmu_T_BURWS-1", "bmu_T_CLVHS-1"]
    shared = points[1]
    assert shared.bmu_ids == ("T_CLVHS-1", "T_CLVHS-2")
    assert (shared.latitude, shared.longitude) == (51.5, 0.5)


def test_multi_site_bmu_is_not_a_point() -> None:
    points = fetch.bmu_points(census=_census())

    assert all("T_MULTI-1" not in (point.bmu_ids or ()) for point in points)


def test_grid_points_are_numbered_gb_01_to_gb_18() -> None:
    points = fetch.grid_points()

    assert [point.point_id for point in points] == [f"gb_{n:02d}" for n in range(1, 19)]
    assert (points[0].latitude, points[0].longitude) == (57.5, -4.0)
    assert all(point.bmu_ids is None for point in points)


def test_read_one_keeps_the_end_of_each_period_and_casts_to_float32(tmp_path: Path) -> None:
    path = tmp_path / "cams.csv"
    path.write_text(CSV_TEXT)
    point = fetch.Point(
        point_id="bmu_T_CLVHS-1",
        bmu_ids=("T_CLVHS-1", "T_CLVHS-2"),
        region=None,
        latitude=51.338767,
        longitude=0.913885,
    )

    frame = fetch.read_one(path=path, point=point)

    assert frame["time"].to_list() == [
        datetime(2025, 1, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 12, tzinfo=UTC),
    ]
    assert frame["ghi_w_m2"].to_list() == [0.0, 120.25]
    assert frame["clear_sky_ghi_w_m2"].to_list() == [0.0, 250.5]
    assert frame["reliability"].to_list() == [1.0, 0.75]
    assert frame.schema["ghi_w_m2"] == pl.Float32
    assert frame["bmu_ids"].to_list() == [["T_CLVHS-1", "T_CLVHS-2"]] * 2
    assert frame["latitude"].to_list() == [51.338767] * 2


def test_read_one_gives_null_bmu_ids_for_a_grid_point(tmp_path: Path) -> None:
    path = tmp_path / "cams.csv"
    path.write_text(CSV_TEXT)

    frame = fetch.read_one(path=path, point=fetch.grid_points()[0])

    assert frame["bmu_ids"].to_list() == [None, None]
    assert frame.schema["bmu_ids"] == pl.List(pl.String)


def test_is_downloaded_skips_a_finished_file_but_not_an_empty_or_missing_one(
    tmp_path: Path,
) -> None:
    finished = tmp_path / "finished.csv"
    finished.write_text(CSV_TEXT)
    empty = tmp_path / "empty.csv"
    empty.write_text("")

    assert fetch.is_downloaded(path=finished)
    assert not fetch.is_downloaded(path=empty)
    assert not fetch.is_downloaded(path=tmp_path / "missing.csv")


def test_jobs_cover_every_calendar_year_of_every_point() -> None:
    points = fetch.grid_points()[:2]

    jobs = fetch.build_jobs(points=points)

    years = range(fetch.FIRST_YEAR, fetch.LAST_YEAR + 1)
    assert [(job.point.point_id, job.year) for job in jobs] == [
        (point.point_id, year) for point in points for year in years
    ]
    assert fetch.FIRST_YEAR == 2025


def test_drop_bad_rows_removes_a_nan_row_that_drop_nulls_would_keep(tmp_path: Path) -> None:
    path = tmp_path / "cams.csv"
    path.write_text(
        CSV_TEXT + "2025-01-01T12:00:00.0/2025-01-01T13:00:00.0;0;nan;0;0;0;nan;0;0;0;1\n"
    )
    frame = fetch.read_one(path=path, point=fetch.grid_points()[0])

    counts = fetch.count_bad_values(frame=frame)
    kept = fetch.drop_bad_rows(frame=frame)

    assert counts["ghi_w_m2_nan"] == 1
    assert counts["ghi_w_m2_null"] == 0
    assert frame.height == 3
    assert kept.height == 2


def test_find_gaps_reports_a_missing_hour_and_only_that_one() -> None:
    point = fetch.grid_points()[0]
    jobs = [fetch.PointYear(point=point, year=2025)]
    hours = fetch.expected_hours(year=2025)
    times = pl.datetime_range(
        datetime(2025, 1, 1, 1),
        datetime(2025, 1, 1, 1) + timedelta(hours=hours - 1),
        interval="1h",
        time_zone="UTC",
        eager=True,
    )
    full = pl.DataFrame({"point_id": point.point_id, "time": times})

    assert fetch.find_gaps(tidy=full, jobs=jobs) == {}
    assert fetch.find_gaps(tidy=full.slice(1), jobs=jobs) == {f"{point.point_id}/2025": -1}


def test_date_range_and_hours_follow_the_requested_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "LAST_DATE", "2026-09-10")
    monkeypatch.setattr(fetch, "LAST_YEAR", 2026)

    assert fetch._date_range_of(year=2025) == "2025-01-01/2025-12-31"
    assert fetch._date_range_of(year=2026) == "2026-01-01/2026-09-10"
    assert fetch.expected_hours(year=2025) == 8760
    assert fetch.expected_hours(year=2026) == 6072


def test_final_year_csv_name_carries_the_end_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "LAST_DATE", "2026-09-10")
    monkeypatch.setattr(fetch, "LAST_YEAR", 2026)
    raw = Path("raw")

    assert fetch.csv_path(raw_dir=raw, point_id="gb_01", year=2025).name == "cams_gb_01_2025.csv"
    assert (
        fetch.csv_path(raw_dir=raw, point_id="gb_01", year=2026).name
        == "cams_gb_01_2026_to_2026-09-10.csv"
    )
