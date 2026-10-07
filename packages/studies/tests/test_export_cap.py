from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from studies import export_cap

START = datetime(2024, 1, 1, tzinfo=UTC)
LIMIT_MW = 5.0


def _write_cap(*, directory: Path, time_series_id: int, caps: list[tuple[float, float]]) -> None:
    """Write one generator's half-hourly export cap, as `anm_setpoints.py` does.

    Args:
        directory: The directory the cap files live in.
        time_series_id: The generator's identifier, which ends the filename.
        caps: One `(cap_mw, lowest_cap_mw)` pair per half-hour from `START`.
    """
    pl.DataFrame(
        {
            "time": [START + timedelta(minutes=30 * step) for step in range(1, len(caps) + 1)],
            "cap_mw": [cap for cap, _ in caps],
            "lowest_cap_mw": [lowest for _, lowest in caps],
        }
    ).write_parquet(directory / f"{export_cap.CAP_FILE_PREFIX}{time_series_id}.parquet")


@pytest.fixture
def dataset() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": ["A", "A", "A", "A", "B"],
            "time": [START + timedelta(hours=hour) for hour in (1, 2, 3, 4, 1)],
        }
    )


@pytest.fixture
def one_site_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Serve a site list where generator 7 is site A, and return the cap directory."""
    monkeypatch.setattr(export_cap, "ANM_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        export_cap, "pv_sites", lambda: pl.DataFrame({"time_series_id": [7], "site": ["A"]})
    )
    return tmp_path


def test_an_hour_is_constrained_when_any_of_its_half_hours_is_below_the_limit(
    one_site_list: Path, dataset: pl.DataFrame
):
    # Hour ending 01:00: both half-hours at the limit. Hour ending 02:00: one half-hour at the limit
    # and one at 2 MW. Hour ending 03:00: 4 MW then 2 MW. Hour ending 04:00: a cap a hair below the
    # limit, which the tolerance treats as a rounding artefact.
    caps = [
        (LIMIT_MW, LIMIT_MW),
        (LIMIT_MW, LIMIT_MW),
        (LIMIT_MW, LIMIT_MW),
        (2.0, 2.0),
        (4.0, 4.0),
        (2.0, 2.0),
        (LIMIT_MW - 0.0005, LIMIT_MW - 0.0005),
        (LIMIT_MW, LIMIT_MW),
    ]
    _write_cap(directory=one_site_list, time_series_id=7, caps=caps)

    result = export_cap.with_export_cap(dataset=dataset).sort("site", "time")

    site_a = result.filter(site="A")
    assert site_a["constrained"].to_list() == [False, True, True, False]
    assert site_a["cap_mw"].to_list() == pytest.approx(
        [LIMIT_MW, (LIMIT_MW + 2.0) / 2, 3.0, LIMIT_MW - 0.00025]
    )


def test_readings_before_the_scheme_went_live_are_discarded(
    one_site_list: Path, dataset: pl.DataFrame
):
    # The first two half-hours sit below the limit, so the scheme is not live yet; they must not
    # flag the hour ending at 01:00 as constrained.
    caps = [(1.0, 1.0), (1.0, 1.0), (LIMIT_MW, LIMIT_MW), (LIMIT_MW, LIMIT_MW)]
    _write_cap(directory=one_site_list, time_series_id=7, caps=caps)

    result = export_cap.with_export_cap(dataset=dataset).sort("site", "time")

    assert result.filter(site="A")["cap_mw"].to_list() == [None, LIMIT_MW, None, None]
    assert result["constrained"].to_list() == [False] * dataset.height


def test_a_cap_file_for_a_generator_outside_the_site_list_is_skipped(
    one_site_list: Path, dataset: pl.DataFrame
):
    _write_cap(directory=one_site_list, time_series_id=99, caps=[(LIMIT_MW, LIMIT_MW)] * 2)

    result = export_cap.with_export_cap(dataset=dataset)

    assert result["cap_mw"].null_count() == dataset.height
    assert result["constrained"].to_list() == [False] * dataset.height


def test_with_no_cap_files_every_row_reads_as_unconstrained(
    one_site_list: Path, dataset: pl.DataFrame
):
    result = export_cap.with_export_cap(dataset=dataset)

    assert result["cap_mw"].null_count() == dataset.height
    assert result["constrained"].to_list() == [False] * dataset.height
