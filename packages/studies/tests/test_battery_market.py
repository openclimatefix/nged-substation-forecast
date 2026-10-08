from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from studies.battery_market import FOLD_MONTHS, battery_output, p99_output_mw

from studies import battery_market


def test_battery_output_moves_the_stamp_from_the_period_end_to_the_period_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    end = datetime(2025, 10, 1, 0, 30, tzinfo=UTC)
    pl.DataFrame({"half_hour_end_time": [end], "output_mwh": [1.5]}).write_parquet(
        tmp_path / f"BMU-1{battery_market.B1610_SUFFIX}"
    )
    monkeypatch.setattr(battery_market, "B1610_DIR", tmp_path)

    result = battery_output(bmu_id="BMU-1")

    assert result["time"].to_list() == [datetime(2025, 10, 1, 0, 0, tzinfo=UTC)]
    assert result["output_mw"].to_list() == [3.0]  # half-hour energy doubled to power


def test_p99_is_of_the_absolute_output_so_a_big_import_counts() -> None:
    frame = pl.DataFrame({"output_mw": [-100.0] + [1.0] * 99 + [2.0] * 900})

    assert p99_output_mw(frame=frame) == pytest.approx(2.0)
    assert p99_output_mw(frame=frame.with_columns(output_mw=pl.col("output_mw") * -1)) == (
        pytest.approx(2.0)
    )
    assert p99_output_mw(frame=pl.DataFrame({"output_mw": [-5.0] * 50 + [1.0] * 50})) == 5.0


def test_every_calendar_month_is_in_exactly_one_of_four_three_month_folds() -> None:
    assert sorted(FOLD_MONTHS) == list(range(1, 13))
    for fold in range(4):
        assert sum(1 for f in FOLD_MONTHS.values() if f == fold) == 3
