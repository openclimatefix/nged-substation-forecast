import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

from fit_aifs import (  # noqa: E402
    WN3_DAYS,
    WN3_SPLITS,
    shuffled_prefix,
    wn3_arms,
    wn3_dropped_rows,
    wn3_sensitivity_arms,
    wn3_split,
    wn3_stage_lines,
)
from nwp_forecast_charts import load_row_set_marks  # noqa: E402
from nwp_forecast_comparison import METRIC  # noqa: E402

MONTHS = [2, 4, 6, 7, 8, 9]


def _rows(*, arm: str, values: dict[datetime, float], setting: str = "primary") -> pl.DataFrame:
    """Two seeds per valid time at one site."""
    return pl.DataFrame(
        [
            {
                "arm": arm,
                "site": "A",
                "time": time,
                "month": f"{time:%Y-%m}",
                "seed": seed,
                "setting": setting,
                METRIC: value + 0.001 * seed,
                "device": "cuda",
            }
            for time, value in values.items()
            for seed in (0, 1)
        ],
        schema_overrides={"time": pl.Datetime("us", "UTC")},
    )


def _monthly(*, value: float = 1.0) -> dict[datetime, float]:
    """One valid time on the 10th of each month, well clear of the training end."""
    return {datetime(2026, month, 10, 12, tzinfo=UTC): value for month in MONTHS}


def test_june_is_in_sample_and_july_is_out_of_sample():
    losses = _rows(arm="x", values=_monthly())

    in_sample = wn3_split(losses=losses, split="in-sample", domain="wind", day=1)
    out_of_sample = wn3_split(losses=losses, split="out-of-sample", domain="wind", day=1)

    assert sorted(in_sample["month"].unique().to_list()) == ["2026-02", "2026-04", "2026-06"]
    assert sorted(out_of_sample["month"].unique().to_list()) == [
        "2026-07",
        "2026-08",
        "2026-09",
    ]


def test_a_day_14_july_row_from_a_june_run_is_in_neither_group():
    june_run = datetime(2026, 7, 5, 12, tzinfo=UTC)  # day 14 reads the 00 UTC run of 21 June
    july_run = datetime(2026, 7, 20, 12, tzinfo=UTC)  # day 14 reads the run of 6 July
    losses = _rows(arm="x", values={**_monthly(), june_run: 1.0, july_run: 1.0})

    out_of_sample = wn3_split(losses=losses, split="out-of-sample", domain="wind", day=14)
    in_sample = wn3_split(losses=losses, split="in-sample", domain="wind", day=14)

    assert june_run not in out_of_sample["time"].to_list()
    assert june_run not in in_sample["time"].to_list()
    assert july_run in out_of_sample["time"].to_list()
    # The 10 July row of `_monthly()` reads the run of 26 June, so it is dropped too.
    assert wn3_dropped_rows(losses=losses, domain="wind", day=14) == 2


def test_a_solar_hour_ending_at_midnight_reads_the_previous_days_run():
    """Solar `time` labels the hour's end, so 00:00 on 1 July at day 1 reads the run of 29 June."""
    time = datetime(2026, 7, 1, 0, tzinfo=UTC)
    losses = _rows(arm="x", values={**_monthly(), time: 1.0})

    assert wn3_dropped_rows(losses=losses, domain="solar", day=1) == 1
    assert wn3_dropped_rows(losses=losses, domain="wind", day=1) == 1


def test_an_unknown_split_raises():
    losses = _rows(arm="x", values=_monthly())

    with pytest.raises(ValueError, match="unknown WN3 split"):
        wn3_split(losses=losses, split="all", domain="wind", day=1)


def test_a_group_with_no_scored_row_raises():
    losses = _rows(arm="x", values={datetime(2026, 2, 10, 12, tzinfo=UTC): 1.0})

    with pytest.raises(ValueError, match="out-of-sample group holds no scored row"):
        wn3_split(losses=losses, split="out-of-sample", domain="wind", day=1)


def test_the_two_named_splits_are_the_only_ones():
    assert WN3_SPLITS == ("in-sample", "out-of-sample")


def test_the_chart_marks_use_only_out_of_sample_months(tmp_path: Path):
    for day in WN3_DAYS:
        pl.concat(
            [
                _rows(arm=f"wn3_mean_day{day}", values=_monthly(value=2.0)),
                _rows(arm=f"ens_mean_day{day}", values=_monthly(value=3.0)),
            ]
        ).write_parquet(tmp_path / f"solar_wn3_day{day}_losses.parquet")

    marks = load_row_set_marks(blends_dir=None, wn3_dir=tmp_path, domain="solar")

    assert len(marks) == 1
    assert sorted(marks[0].losses["month"].unique().to_list()) == [
        "2026-07",
        "2026-08",
        "2026-09",
    ]


def _stage_losses(*, out_of_sample_gap: dict[int, float]) -> pl.DataFrame:
    """Every solar day-1 arm's losses: far from each other in the in-sample months.

    Args:
        out_of_sample_gap: The negative control's minus WN3's loss in each out-of-sample month.

    Returns:
        WN3 is 1 point below its shuffled copy and 2 below ENS in every in-sample month, and
        `out_of_sample_gap` below its shuffled copy in the out-of-sample months.
    """
    permuted = shuffled_prefix(source="wn3_mean_day1")
    permuted_b = shuffled_prefix(source="wn3_mean_day1", variant="_b")
    arms = list(wn3_arms(domain="solar", day=1))
    wn3, ens = {}, {}
    control, control_b = {}, {}
    for month in MONTHS:
        time = datetime(2026, month, 10, 12, tzinfo=UTC)
        wn3[time] = 3.0
        ens[time] = 5.0
        gap = out_of_sample_gap.get(month, 1.0)
        control[time] = 3.0 + gap
        control_b[time] = 3.0 + gap + 1.0
    values = {
        "wn3_mean_day1": wn3,
        "ens_mean_day1": ens,
        permuted: control,
        permuted_b: control_b,
    }
    assert set(values) == set(arms)
    return pl.concat([_rows(arm=arm, values=values[arm]) for arm in arms])


def test_a_pair_near_the_line_only_in_the_out_of_sample_months_is_refitted():
    permuted = shuffled_prefix(source="wn3_mean_day1")
    losses = _stage_losses(out_of_sample_gap={7: 0.0, 8: 0.0, 9: 0.1})

    chosen = wn3_sensitivity_arms(losses=losses, domain="solar", day=1)

    assert {"wn3_mean_day1", "ens_mean_day1", permuted} <= set(chosen)


def test_a_pair_far_from_the_line_in_both_groups_is_not_refitted():
    permuted = shuffled_prefix(source="wn3_mean_day1")
    losses = _stage_losses(out_of_sample_gap={})

    chosen = wn3_sensitivity_arms(losses=losses, domain="solar", day=1)

    assert permuted not in chosen


def test_the_report_has_a_table_for_each_group_and_the_fair_comparison_warning():
    losses = _stage_losses(out_of_sample_gap={7: 0.0, 8: 0.0, 9: 0.1})
    frame = pl.DataFrame({"month": [f"2026-{m:02d}" for m in MONTHS], "site": ["A"] * len(MONTHS)})

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("fit_aifs.coverage_table", lambda *, frame: pl.DataFrame({"covered": [True]}))
        patch.setattr("fit_aifs.arm_features", lambda *, arm, domain: ("ghi",))
        text = "\n".join(wn3_stage_lines(domain="solar", day=1, frame=frame, losses=losses))

    assert "#### In-sample months (February to June 2026, 3 months)" in text
    assert "#### Out-of-sample months (July to September 2026, 3 months)" in text
    assert "may not be a fair comparison here" in text
    assert "0 (site, hour) rows fall in neither group" in text
    assert text.index("In-sample months") < text.index("Out-of-sample months")
