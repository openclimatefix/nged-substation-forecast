"""Tests for the post hoc follow-up scripts under `studies/lag_features/`.

Each test is written to fail on the defect it names.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import fit_followups
import followup_frames
import polars as pl
import pytest
import report_followups


def _hourly(*, sites: tuple[str, ...], days: int) -> pl.DataFrame:
    start = datetime(2024, 6, 1, tzinfo=UTC)
    rows = [
        {"site": site, "time": start + timedelta(hours=hour)}
        for site in sites
        for hour in range(24 * days)
    ]
    return pl.DataFrame(rows)


def test_plant_steps_are_seeded_bounded_and_one_set_per_plant():
    hourly = _hourly(sites=("A", "B"), days=800)

    first = followup_frames.plant_steps(hourly=hourly)
    second = followup_frames.plant_steps(hourly=hourly)

    assert first.equals(second)
    assert first.group_by("site").len()["len"].to_list() == [followup_frames.STEPS_PER_PLANT] * 2
    summary = first.select(
        shortest=((pl.col("end") - pl.col("start")).dt.total_days() / 7).min(),
        longest=((pl.col("end") - pl.col("start")).dt.total_days() / 7).max(),
        earliest=pl.col("start").min(),
    ).row(0, named=True)
    assert summary["shortest"] >= followup_frames.STEP_WEEKS[0]
    assert summary["longest"] <= followup_frames.STEP_WEEKS[1]
    assert summary["earliest"] >= followup_frames.STEP_RANGE_START


def test_the_step_factor_shifts_only_the_named_plant_inside_the_step():
    steps = pl.DataFrame(
        {
            "site": ["A"],
            "start": [datetime(2025, 1, 10, tzinfo=UTC)],
            "end": [datetime(2025, 1, 20, tzinfo=UTC)],
        }
    )
    frame = pl.DataFrame(
        {
            "site": ["A", "A", "A", "B"],
            "time": [
                datetime(2025, 1, 9, 23, tzinfo=UTC),
                datetime(2025, 1, 10, tzinfo=UTC),
                datetime(2025, 1, 20, tzinfo=UTC),
                datetime(2025, 1, 15, tzinfo=UTC),
            ],
        }
    )

    factor = frame.select(factor=followup_frames.step_factor(steps=steps, shift=0.1))["factor"]

    assert factor.to_list() == pytest.approx([1.0, 0.9, 1.0, 1.0])


def test_the_month_factor_uses_the_hours_midpoint_month():
    shifted = min(followup_frames.shifted_months())
    year, month = (int(part) for part in shifted.split("-"))
    first_hour_end = datetime(year, month, 1, 0, 0, tzinfo=UTC)
    frame = pl.DataFrame({"time": [first_hour_end, first_hour_end + timedelta(hours=1)]})

    factor = frame.select(factor=followup_frames.month_factor(shift=0.05))["factor"]

    assert factor[1] == pytest.approx(0.95)


def test_the_climatology_column_withholds_the_scored_fold_from_training_rows():
    rows = [
        {
            "site": "A",
            "time": datetime(2025, 3, 1 + day + 3 * fold, 12, tzinfo=UTC),
            "fold": fold,
            "constrained": False,
            "power_mw": 100.0 * fold,
        }
        for fold in range(5)
        for day in range(3)
    ]

    frame = followup_frames.add_climatology_columns(frame=pl.DataFrame(rows))

    own_fold_zero = frame.filter(pl.col("fold") == 0)
    assert own_fold_zero["climatology_fold0"].unique().to_list() == [250.0]
    assert own_fold_zero["climatology_fold1"].unique().to_list() == [300.0]
    own_fold_three = frame.filter(pl.col("fold") == 3)
    assert own_fold_three["climatology_fold1"].unique().to_list() == [200.0]


def test_a_pooled_arm_is_15_fits_and_a_per_plant_arm_90():
    fits = fit_followups.followup_fits()

    pooled = [fit for fit in fits if fit.pooled]
    per_plant = [fit for fit in fits if not fit.pooled]

    assert {fit_followups.fits_in(fit=fit) for fit in pooled} == {15}
    assert {fit_followups.fits_in(fit=fit) for fit in per_plant} == {90}
    assert sum(fit_followups.fits_in(fit=fit) for fit in fits) == 4800


def test_no_followup_fit_names_an_arm_without_columns():
    for fit in fit_followups.followup_fits():
        assert fit.arm in {
            *followup_frames.MONTH_CONTROL_ARMS,
            *followup_frames.STEP_CONTROL_ARMS,
            *fit_followups.LONG_LEAD_ARMS,
            *fit_followups.LONG_LEAD_SENSITIVITY_ARMS,
            "PC2",
            "G-ID+TF",
            "G-FPnoCK",
        }


def test_the_first_report_parser_counts_only_exploratory_difference_intervals(tmp_path: Path):
    report = tmp_path / "report.md"
    lines = [
        "## Phase 2: the planned contrasts",
        "",
        "| Contrast | Difference (pp) | 99% interval (pp) |",
        "|---|---|---|",
        "| P1 | -0.2 | [-0.3, -0.1] |",
        "",
        "## Exploratory contrasts at lead-day 1",
        "",
        "| Contrast (exploratory) | Difference (pp) | 95% interval (pp) |",
        "|---|---|---|",
        "| A − B | -0.2 | [-0.300, -0.100] |",
        "| C − D | +0.1 | [-0.050, +0.250] |",
        "",
        "## Monthly differences",
        "",
        "| Month | Difference (pp) |",
        "|---|---|",
        "| 2025-01 | +0.1 |",
    ]
    report.write_text("\n".join(lines))

    records = report_followups.first_report_intervals(path=report)

    assert [(r["lower"], r["upper"]) for r in records] == [(-0.3, -0.1), (-0.05, 0.25)]


def test_the_interval_counts_split_wins_losses_and_zero_crossings():
    ledger = report_followups.Ledger(
        records=[
            {"lower": -0.3, "upper": -0.1},
            {"lower": 0.1, "upper": 0.3},
            {"lower": -0.1, "upper": 0.2},
        ]
    )

    _, table = report_followups.count_lines(first=[], ledger=ledger)

    row = table.filter(pl.col("report") == "follow-ups").row(0, named=True)
    assert (row["intervals"], row["win_side"], row["loss_side"], row["include_zero"]) == (
        3,
        1,
        1,
        1,
    )


def test_the_month_level_control_fits_l1_and_n2_is_refitted_at_long_leads():
    assert "L1" in followup_frames.MONTH_CONTROL_ARMS
    assert "N2" in fit_followups.LONG_LEAD_ARMS


def test_the_pairing_guard_raises_when_follow_up_rows_differ_from_the_first_runs():
    first = pl.DataFrame({"site": ["A", "A"], "time": [1, 2]})
    other = pl.DataFrame({"site": ["A"], "time": [1]})

    with pytest.raises(ValueError, match="differ from the first run's"):
        report_followups.require_same_rows(reference=first, other=other, name="test")
    report_followups.require_same_rows(reference=first, other=first, name="test")


def test_the_manifest_digests_include_the_first_runs_losses(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="missing input"):
        fit_followups.require_first_run(root=tmp_path, smoke=False)
    losses = fit_followups.first_run_losses_path(root=tmp_path, smoke=False)

    assert losses in fit_followups.input_files(root=tmp_path, smoke=False)
    assert losses.name.endswith("_smoke.parquet") is False
    assert fit_followups.first_run_losses_path(root=tmp_path, smoke=True).name.endswith(
        "_smoke.parquet"
    )
