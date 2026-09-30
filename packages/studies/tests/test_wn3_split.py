import sys
from pathlib import Path

import polars as pl
import pytest

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

from fit_aifs import (  # noqa: E402
    WN3_DAYS,
    WN3_SPLITS,
    wn3_arms,
    wn3_sensitivity_arms,
    wn3_split,
    wn3_stage_lines,
)
from nwp_forecast_charts import load_row_set_marks  # noqa: E402
from nwp_forecast_comparison import METRIC  # noqa: E402

MONTHS = ["2026-02", "2026-04", "2026-06", "2026-07", "2026-09"]


def _losses(*, arm: str, month_values: dict[str, float], site: str = "A") -> pl.DataFrame:
    """Two seeds per month at one site, both settings' rows kept apart by `setting`."""
    rows = [
        {
            "arm": arm,
            "site": site,
            "time": f"{month}-01T00:00:00",
            "month": month,
            "seed": seed,
            "setting": "primary",
            METRIC: value + 0.001 * seed,
            "device": "cuda",
        }
        for month, value in month_values.items()
        for seed in (0, 1)
    ]
    return pl.DataFrame(rows)


def test_june_is_in_sample_and_july_is_out_of_sample():
    losses = _losses(arm="x", month_values=dict.fromkeys(MONTHS, 1.0))

    in_sample = wn3_split(losses=losses, split="in-sample")
    out_of_sample = wn3_split(losses=losses, split="out-of-sample")

    assert sorted(in_sample["month"].unique().to_list()) == ["2026-02", "2026-04", "2026-06"]
    assert sorted(out_of_sample["month"].unique().to_list()) == ["2026-07", "2026-09"]
    assert in_sample.height + out_of_sample.height == losses.height


def test_the_two_groups_do_not_share_a_row():
    losses = _losses(arm="x", month_values=dict.fromkeys(MONTHS, 1.0))

    in_sample = wn3_split(losses=losses, split="in-sample")
    out_of_sample = wn3_split(losses=losses, split="out-of-sample")

    assert not set(in_sample["month"]) & set(out_of_sample["month"])


def test_an_unknown_split_raises():
    losses = _losses(arm="x", month_values={"2026-07": 1.0})

    with pytest.raises(ValueError, match="unknown WN3 split"):
        wn3_split(losses=losses, split="all")


def test_a_group_with_no_scored_row_raises():
    losses = _losses(arm="x", month_values={"2026-02": 1.0})

    with pytest.raises(ValueError, match="out-of-sample group holds no scored row"):
        wn3_split(losses=losses, split="out-of-sample")


def test_the_two_named_splits_are_the_only_ones():
    assert WN3_SPLITS == ("in-sample", "out-of-sample")


def test_the_chart_marks_use_only_out_of_sample_months(tmp_path: Path):
    for day in WN3_DAYS:
        pl.concat(
            [
                _losses(arm=f"wn3_mean_day{day}", month_values=dict.fromkeys(MONTHS, 2.0)),
                _losses(arm=f"ens_mean_day{day}", month_values=dict.fromkeys(MONTHS, 3.0)),
            ]
        ).write_parquet(tmp_path / f"solar_wn3_day{day}_losses.parquet")

    marks = load_row_set_marks(blends_dir=None, wn3_dir=tmp_path, domain="solar")

    assert len(marks) == 1
    assert sorted(marks[0].losses["month"].unique().to_list()) == ["2026-07", "2026-09"]


def _stage_losses(*, arms: list[str]) -> pl.DataFrame:
    """Per-arm losses over all five months, the WN3 arm 1 point below every other arm."""
    return pl.concat(
        [
            _losses(
                arm=arm, month_values=dict.fromkeys(MONTHS, 2.0 if arm.startswith("wn3") else 3.0)
            )
            for arm in arms
        ]
    )


def test_every_sensitivity_pair_is_chosen_from_both_groups():
    arms = list(wn3_arms(domain="solar", day=1))
    losses = _stage_losses(arms=arms)

    chosen = wn3_sensitivity_arms(losses=losses, domain="solar", day=1)

    assert {"wn3_mean_day1", "ens_mean_day1"} <= set(chosen)


def test_the_report_has_a_table_for_each_group_and_the_fair_comparison_warning():
    from fit_aifs import wn3_arms

    arms = list(wn3_arms(domain="solar", day=1))
    losses = _stage_losses(arms=arms)
    frame = pl.DataFrame({"month": MONTHS, "site": ["A"] * len(MONTHS)})

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("fit_aifs.coverage_table", lambda *, frame: pl.DataFrame({"covered": [True]}))
        patch.setattr("fit_aifs.arm_features", lambda *, arm, domain: ("ghi",))
        text = "\n".join(wn3_stage_lines(domain="solar", day=1, frame=frame, losses=losses))

    assert "#### In-sample months (February to June 2026, 3 months)" in text
    assert "#### Out-of-sample months (July to September 2026, 2 months)" in text
    assert "not a fair comparison here" in text
    assert text.index("In-sample months") < text.index("Out-of-sample months")
