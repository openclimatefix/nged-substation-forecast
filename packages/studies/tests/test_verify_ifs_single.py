import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

from build_forecast_inputs import IFS_SINGLE_DAYS, ifs_single_arm  # noqa: E402
from verify_ifs_single import (  # noqa: E402
    expected_served,
    gap_table,
    gap_verdict,
    hour_ending_verdict,
    lookup_verdict,
    mirror_asymmetry,
    native_step_verdict,
    second_difference_by_step,
    served_table,
    served_verdict,
    source_facts,
    source_verdict,
    stable_step_widths,
)


def _daylight(*, shift: float) -> dict[int, float]:
    """The mean radiation by label hour of a day whose noon mean sits at midday, shifted by `shift`.

    A curve `sin(pi * (h - 5.5 - shift) / 13)` peaks at 12 UTC when `shift` is 0.
    """
    return {
        hour: max(0.0, 700.0 * math.sin(math.pi * (hour - 5.5 - shift) / 13)) for hour in range(24)
    }


def test_the_served_rule_agrees_between_the_two_expressions_and_day_ten_is_unservable():
    table = served_table(first_day=datetime(2025, 3, 10, tzinfo=UTC))

    assert served_verdict(table=table) == []
    assert max(table.filter(pl.col("day") == 10)["lead"].to_list()) > 240


def test_a_wrong_lead_is_flagged_by_the_served_runs_check():
    table = served_table(first_day=datetime(2025, 3, 10, tzinfo=UTC)).with_columns(
        lead=pl.when((pl.col("technology") == "wind") & (pl.col("day") == 3))
        .then(1)
        .otherwise(pl.col("lead")),
        agrees=pl.lit(True),
    )

    assert served_verdict(table=table)


def test_a_day_ten_whose_leads_fit_in_the_run_is_flagged():
    table = served_table(first_day=datetime(2025, 3, 10, tzinfo=UTC)).with_columns(
        lead=pl.when(pl.col("day") == 10).then(pl.col("lead") - 100).otherwise(pl.col("lead")),
        agrees=pl.lit(True),
    )

    assert any("day 10" in line for line in served_verdict(table=table))


def test_the_plain_python_rule_matches_the_worked_examples():
    solar = expected_served(time=datetime(2025, 3, 10, 13, tzinfo=UTC), day=2, solar=True)
    wind = expected_served(time=datetime(2025, 3, 10, 13, tzinfo=UTC), day=0, solar=False)

    assert solar == (datetime(2025, 3, 8, tzinfo=UTC), 61)
    assert wind == (datetime(2025, 3, 10, tzinfo=UTC), 13)


def test_a_curve_of_hour_ending_means_mirrors_better_under_the_hour_ending_pairing():
    # Means over the hour ending at each label have their midpoint half an hour earlier, so the
    # curve of labels peaks half an hour after the curve of midpoints.
    ending_curve = _daylight(shift=0.5)

    failures, ending, starting = hour_ending_verdict(mean_by_label_hour=ending_curve)

    assert failures == []
    assert ending < starting


def test_a_curve_of_hour_starting_means_fails_the_hour_ending_check():
    starting_curve = _daylight(shift=-0.5)

    failures, _, _ = hour_ending_verdict(mean_by_label_hour=starting_curve)

    assert failures


def test_a_perfectly_mirrored_curve_has_no_asymmetry():
    curve = {hour: float(min(hour, 25 - hour)) for hour in range(24)}
    curve[0] = 0.0

    assert mirror_asymmetry(mean_by_label_hour=curve, pair_sum=25) == 0.0


def test_the_stable_step_widths_leave_out_the_leads_beside_a_change_of_width():
    widths = dict(zip(*stable_step_widths().to_dict(as_series=False).values(), strict=True))

    assert {89, 92, 143, 146} <= set(widths)
    assert not {90, 91, 144, 145} & set(widths)
    assert widths[50] == 1
    assert widths[100] == 3
    assert widths[200] == 6


def _smooth_interpolation(*, x: np.ndarray, knots: list[int], values: np.ndarray) -> np.ndarray:
    """Interpolate between knots along a smoothstep, which is flat at every knot."""
    upper = np.clip(np.searchsorted(knots, x, side="left"), 1, len(knots) - 1)
    x0, x1 = np.array(knots)[upper - 1], np.array(knots)[upper]
    t = (x - x0) / (x1 - x0)
    weight = t * t * (3 - 2 * t)
    return values[upper - 1] + (values[upper] - values[upper - 1]) * weight


def _archive_with_interpolated_wind(*, path: Path) -> None:
    """Runs whose wind speed is random at each native step and interpolated smoothly between."""
    rng = np.random.default_rng(3)
    native = [*range(91), *range(93, 145, 3), *range(150, 241, 6)]
    records = []
    for day in range(6):
        values = rng.normal(20.0, 5.0, size=len(native))
        interpolated = _smooth_interpolation(x=np.arange(241), knots=native, values=values)
        records.extend(
            {
                "site": "A",
                "init_time": datetime(2025, 3, 1 + day),
                "lead_hours": lead,
                "wind_speed_100m": float(interpolated[lead]),
            }
            for lead in range(241)
        )
    pl.DataFrame(records, schema_overrides={"lead_hours": pl.Int32}).write_parquet(path)


def test_wind_interpolated_beyond_lead_90_has_a_smaller_second_difference_there(tmp_path: Path):
    path = tmp_path / "archive.parquet"
    _archive_with_interpolated_wind(path=path)

    table = second_difference_by_step(path=path)

    size = dict(
        zip(table["step_hours"].to_list(), table["second_difference"].to_list(), strict=True)
    )
    assert size[3] < 0.6 * size[1]
    assert size[6] < 0.3 * size[1]
    assert native_step_verdict(table=table) == []


def test_wind_that_is_equally_jagged_at_every_lead_fails_the_native_step_check():
    table = pl.DataFrame({"step_hours": [1, 3, 6], "second_difference": [2.4, 2.3, 2.5]})

    failures = native_step_verdict(table=table)

    assert len(failures) == 2


def _source_rows(*, runs: list[int], leads: range = range(241)) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "site": "A",
                "init_time": datetime(2025, 3, 1 + run),
                "lead_hours": lead,
                "shortwave_radiation": None if lead == 0 else (-1.0 if lead == 80 else 5.0),
            }
            for run in runs
            for lead in leads
        ],
        schema_overrides={"lead_hours": pl.Int32, "shortwave_radiation": pl.Float64},
    )


def test_source_facts_counts_absent_runs_the_negative_radiation_and_passes_a_complete_archive(
    tmp_path: Path,
):
    path = tmp_path / "archive.parquet"
    _source_rows(runs=[0, 1, 3]).write_parquet(path)

    facts = source_facts(path=path)

    assert (facts.runs_present, facts.runs_expected) == (3, 4)
    assert facts.radiation_below_zero == 3
    assert facts.radiation_minimum == -1.0
    assert (facts.radiation_below_zero_first_lead, facts.radiation_below_zero_last_lead) == (80, 80)
    assert source_verdict(facts=facts) == []


def test_an_incomplete_run_or_radiation_at_lead_zero_fails_the_source_check(tmp_path: Path):
    path = tmp_path / "archive.parquet"
    _source_rows(runs=[0, 1], leads=range(240)).write_parquet(path)

    facts = source_facts(path=path)

    assert facts.incomplete_runs == 2
    assert any("lack some lead" in line for line in source_verdict(facts=facts))


def _built(
    *, times: list[datetime], domain: str, present_runs: set[datetime] | None
) -> pl.DataFrame:
    """A built frame: 1.0 everywhere, or null where the arm's serving run is not in `present_runs`.

    With `present_runs` of None nothing is null, whatever runs exist.
    """
    columns: dict[str, list[float | None]] = {}
    fields = (
        ("ghi", "temp")
        if domain == "solar"
        else ("speed_100m", "sin_100m", "cos_100m", "speed_10m")
    )
    for day in IFS_SINGLE_DAYS:
        served = [
            (time - timedelta(hours=1 if domain == "solar" else 0)).replace(hour=0)
            - timedelta(days=day)
            for time in times
        ]
        for field in fields:
            columns[f"{ifs_single_arm(day=day)}_{field}"] = [
                None if present_runs is not None and run not in present_runs else 1.0
                for run in served
            ]
    return pl.DataFrame({"site": ["A"] * len(times), "time": times, **columns}).with_columns(
        pl.col("time").dt.replace_time_zone("UTC")
    )


_TIMES = [datetime(2025, 3, 1, 12), datetime(2025, 3, 2, 12), datetime(2025, 3, 3, 12)]
"""Target hours on the three days of a two-run archive (runs on 1 and 2 March)."""


def _write_built(*, directory: Path, present_runs: set[datetime] | None) -> None:
    directory.mkdir()
    for domain in ("solar", "wind"):
        _built(times=_TIMES, domain=domain, present_runs=present_runs).write_parquet(
            directory / f"{domain}_extra_lead_inputs.parquet"
        )


def test_a_gap_filled_without_its_run_fails_the_gap_check(tmp_path: Path):
    archive = tmp_path / "archive.parquet"
    _source_rows(runs=[0, 1]).write_parquet(archive)
    _write_built(directory=tmp_path / "filled", present_runs=None)

    table = gap_table(built_dir=tmp_path / "filled", archive_path=archive)

    # Day 0 of 3 March needs the 3 March run, which the archive lacks, yet the row is filled.
    assert table.filter(pl.col("day") == 0)["filled_but_run_absent"].min() == 1
    assert gap_verdict(table=table)


def test_a_null_row_whose_run_is_present_fails_the_gap_check(tmp_path: Path):
    archive = tmp_path / "archive.parquet"
    _source_rows(runs=[0, 1]).write_parquet(archive)
    every_day_missing = set()
    _write_built(directory=tmp_path / "empty", present_runs=every_day_missing)

    table = gap_table(built_dir=tmp_path / "empty", archive_path=archive)

    assert table.filter(pl.col("day") == 0)["null_but_run_present"].min() == 2
    assert gap_verdict(table=table)


def test_gap_rows_are_counted_and_pass_when_they_are_exactly_the_absent_runs(tmp_path: Path):
    archive = tmp_path / "archive.parquet"
    _source_rows(runs=[0, 1]).write_parquet(archive)
    _write_built(
        directory=tmp_path / "honest",
        present_runs={datetime(2025, 3, 1), datetime(2025, 3, 2)},
    )

    table = gap_table(built_dir=tmp_path / "honest", archive_path=archive)

    assert gap_verdict(table=table) == []
    day_zero_wind = table.filter((pl.col("technology") == "wind") & (pl.col("day") == 0))
    assert day_zero_wind["null_rows"].to_list() == [1]
    assert day_zero_wind["gap_days"].to_list() == [1]


def test_a_column_with_a_differing_or_no_compared_value_fails_the_lookup_check():
    table = pl.DataFrame(
        {
            "technology": ["solar", "solar", "wind"],
            "day": [1, 1, 2],
            "column": ["a", "b", "c"],
            "compared": [10, 10, 0],
            "differing": [0, 2, 0],
            "skipped": [0, 0, 10],
        }
    )

    assert len(lookup_verdict(table=table)) == 2
