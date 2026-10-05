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
            "site": ["A", "A", "A", "B"],
            "time": [START + timedelta(hours=hour) for hour in (1, 2, 3, 1)],
        }
    )


def test_hours_are_constrained_only_after_the_cap_moves_off_the_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dataset: pl.DataFrame
):
    # Half-hours 1-2 are the hour ending at 01:00 (cap at the limit); half-hours 3-4 are the hour
    # ending at 02:00 (the operator pulls the cap to 2 MW for the second half-hour).
    caps = [(LIMIT_MW, LIMIT_MW), (LIMIT_MW, LIMIT_MW), (4.0, 4.0), (2.0, 2.0)]
    _write_cap(directory=tmp_path, time_series_id=7, caps=caps)
    monkeypatch.setattr(export_cap, "ANM_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        export_cap, "pv_sites", lambda: pl.DataFrame({"time_series_id": [7], "site": ["A"]})
    )

    result = export_cap.with_export_cap(dataset=dataset).sort("site", "time")

    assert result["constrained"].to_list() == [False, True, False, False]
    assert result["cap_mw"].to_list() == [LIMIT_MW, 3.0, None, None]


def test_readings_before_the_scheme_went_live_are_discarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dataset: pl.DataFrame
):
    # The first two half-hours sit below the limit, so the scheme is not live yet; they must not
    # flag the hour ending at 01:00 as constrained.
    caps = [(1.0, 1.0), (1.0, 1.0), (LIMIT_MW, LIMIT_MW), (LIMIT_MW, LIMIT_MW)]
    _write_cap(directory=tmp_path, time_series_id=7, caps=caps)
    monkeypatch.setattr(export_cap, "ANM_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        export_cap, "pv_sites", lambda: pl.DataFrame({"time_series_id": [7], "site": ["A"]})
    )

    result = export_cap.with_export_cap(dataset=dataset).sort("site", "time")

    assert result.filter(site="A")["cap_mw"].to_list() == [None, LIMIT_MW, None]
    assert result["constrained"].to_list() == [False, False, False, False]


def test_with_no_cap_files_every_row_reads_as_unconstrained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dataset: pl.DataFrame
):
    monkeypatch.setattr(export_cap, "ANM_DATA_DIR", tmp_path)

    result = export_cap.with_export_cap(dataset=dataset)

    assert result["cap_mw"].null_count() == dataset.height
    assert result["constrained"].to_list() == [False] * dataset.height
