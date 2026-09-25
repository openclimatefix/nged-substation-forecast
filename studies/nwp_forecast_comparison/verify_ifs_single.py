"""Check the IFS HRES (9 km, Open-Meteo) arms' source rows, run mapping and built columns.

One-off throwaway script for the IFS HRES (9 km, Open-Meteo) arms of
<https://github.com/openclimatefix/nged-substation-forecast/issues/912>. It reads and never fits,
and writes nothing under `data/` except its own Markdown files in `<output-dir>/verification/`. Run
it after `build_forecast_inputs.py --extra-leads --batch fourth` and before `fit_extra_leads.py
--batch fourth`. It writes three Markdown files and exits non-zero if any verdict fails:

1. `ifs_single_source.md`: the archive's rows. Every run present holds every site at every lead 0
   to 240 exactly once. The runs absent are counted, never listed. Radiation is null at lead 0 and
   nowhere else. The count of radiation values below zero, and their most negative value and lead
   range, are printed, and the clipped minimum must be zero. The radiation's hour-ending label is
   checked by symmetry: the annual mean of the solar sites' radiation by label hour must mirror
   about midday better when a label ends its hour (label hours `h` and `25 - h` pair up) than when
   it starts its hour (`h` and `23 - h`). The native steps are checked by smoothness: the mean
   absolute second difference of the hourly 100 m wind speed must fall to under 0.6 of its hourly
   value from lead 91 to 144 hours and under 0.3 beyond, because the archive interpolates the
   coarser native steps to hourly.
2. `ifs_single_served_runs.md`: the run and lead each target hour reads at each lead day, for solar
   and wind, worked out twice (`studies.ifs_single_runs`'s expressions and plain Python) and
   required to agree with the rule stated in `build_forecast_inputs._ifs_single_frame`. Day 10
   must be unservable, and day 9 the last servable day.
3. `ifs_single_built_columns.md`: with `--built-dir`, the gap rows and a sample of the built
   columns. Every null row of an arm must be a target hour whose serving run the archive lacks,
   every such row must be null (a gap is never filled from another run), and only the count of
   those rows and of the days they fall on is printed. A sample of rows of each built column is
   recomputed straight from the archive by plain Python and must match, and no built radiation
   may be below zero.

Run it with `uv run python studies/nwp_forecast_comparison/verify_ifs_single.py --output-dir DIR
[--built-dir DIR]`.
"""

import argparse
import logging
import math
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "beam_diffuse_split"))
import ens_forecast_horizons as efh
from build_forecast_inputs import (
    IFS_SINGLE_DAYS,
    IFS_SINGLE_DIR_NAME,
    IFS_SINGLE_FILE_NAME,
    KMH_TO_MS,
    _repo_data_dir,
    ifs_single_arm,
)
from studies.ifs_single_runs import (
    LAST_LEAD_HOURS,
    last_servable_day,
    native_step_hours,
    served_init_time,
    served_lead_hours,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

LEADS_PER_RUN: Final[int] = LAST_LEAD_HOURS + 1
"""How many hourly leads a complete run holds, 0 to 240."""

SAMPLE_ROWS: Final[int] = 400
"""How many rows of each built column the lookup check recomputes, per technology and day."""

LOOKUP_TOLERANCE: Final[float] = 1e-6
"""How close a recomputed value must be to the built one, in its own unit."""

MAX_SECOND_DIFFERENCE_SHARE: Final[dict[int, float]] = {3: 0.6, 6: 0.3}
"""The most that the mean second difference of the hourly wind speed may be, as a share of its value
at the hourly leads, at leads on 3-hourly and 6-hourly native steps. The archive's values measured
0.42 and 0.19 of it."""


# --- The run and lead rule -------------------------------------------------------------------


def expected_served(*, time: datetime, day: int, solar: bool) -> tuple[datetime, int]:
    """Work out the run and lead one target hour reads at a lead day, in plain Python.

    Args:
        time: The target hour's label, timezone-aware.
        day: The lead day.
        solar: Whether the hour is a solar hour, labelled by its end.

    Returns:
        The run's start (00 UTC of the hour's own day, `day` days earlier) and the lead in hours.
    """
    instant = time - timedelta(hours=1) if solar else time
    init = instant.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=day)
    return init, int((time - init).total_seconds() // 3600)


def served_table(
    *, first_day: datetime, days: tuple[int, ...] = (*IFS_SINGLE_DAYS, 10)
) -> pl.DataFrame:
    """Tabulate the run and lead of every hour of one date at every lead day, both ways.

    Args:
        first_day: The date (UTC midnight, timezone-aware) whose 24 target labels are tabulated:
            01:00 to 00:00 the next day for solar and 00:00 to 23:00 for wind.
        days: The lead days to tabulate.

    Returns:
        One row per (technology, day, label), with the run's start and lead from
        `studies.ifs_single_runs`, the same from `expected_served`, and `agrees`.
    """
    records = []
    for domain in ("solar", "wind"):
        offset = 1 if domain == "solar" else 0
        frame = pl.DataFrame({"time": [first_day + timedelta(hours=k + offset) for k in range(24)]})
        for day in days:
            served = frame.select(
                "time",
                init=served_init_time(time=pl.col("time"), day=day, domain=domain),
                lead=served_lead_hours(time=pl.col("time"), day=day, domain=domain),
            )
            for time, init, lead in served.iter_rows():
                plain_init, plain_lead = expected_served(
                    time=time, day=day, solar=domain == "solar"
                )
                records.append(
                    {
                        "technology": domain,
                        "day": day,
                        "label": time,
                        "init": init,
                        "lead": lead,
                        "plain_init": plain_init,
                        "plain_lead": plain_lead,
                        "agrees": init == plain_init and lead == plain_lead,
                    }
                )
    return pl.DataFrame(records)


def served_verdict(*, table: pl.DataFrame) -> list[str]:
    """List the failures of the run and lead rule in `served_table`'s result.

    Args:
        table: `served_table`'s result.

    Returns:
        One line per failure. Two expressions of the rule must agree on every row. A day at most
        `last_servable_day` must read one 00 UTC run at 24 consecutive leads starting at `24 * day`
        (wind) or `24 * day + 1` (solar), all within the run's 240 hours. A later day must reach a
        lead beyond 240, and the last servable day must be 9.
    """
    failures = [
        f"{row['technology']} day {row['day']} {row['label']}: rules disagree"
        for row in table.filter(~pl.col("agrees")).iter_rows(named=True)
    ]
    for (technology, day), group in table.group_by("technology", "day"):
        leads = sorted(group["lead"].to_list())
        first = (24 * day + 1) if technology == "solar" else 24 * day
        if leads != list(range(first, first + 24)):
            failures.append(f"{technology} day {day}: leads {leads} are not 24 from {first}")
        if group["init"].n_unique() != 1 or group["init"][0].hour != 0:
            failures.append(f"{technology} day {day} does not read one 00 UTC run")
        servable = day <= last_servable_day(domain=technology)
        if servable != (max(leads) <= LAST_LEAD_HOURS):
            failures.append(f"{technology} day {day}: servable is {servable} but leads disagree")
    failures.extend(
        f"{technology}: the last servable day is not 9"
        for technology in ("solar", "wind")
        if last_servable_day(domain=technology) != 9
    )
    return failures


# --- The archive's rows ------------------------------------------------------------------------


class SourceFacts(NamedTuple):
    """What the archive holds, by count: runs, completeness, radiation and the clip."""

    runs_present: int
    runs_expected: int
    incomplete_runs: int
    lead_zero_radiation_not_null: int
    radiation_null_beyond_lead_zero: int
    radiation_below_zero: int
    radiation_minimum: float
    radiation_below_zero_first_lead: int | None
    radiation_below_zero_last_lead: int | None


def source_facts(*, path: Path) -> SourceFacts:
    """Count the archive's runs, its incomplete (site, run) pairs, and its radiation oddities.

    Args:
        path: The archive's combined parquet.

    Returns:
        The counts. A run is present if any row carries its `init_time`; a (site, run) pair is
        incomplete unless it holds each lead 0 to 240 exactly once.
    """
    lazy = pl.scan_parquet(path)
    bounds = lazy.select(first=pl.col("init_time").min(), last=pl.col("init_time").max()).collect()
    first, last = bounds.row(0)
    runs_expected = (last - first).days + 1
    per_run = (
        lazy.group_by("site", "init_time")
        .agg(n=pl.len(), distinct=pl.col("lead_hours").n_unique())
        .collect()
    )
    incomplete = per_run.filter(
        (pl.col("n") != LEADS_PER_RUN) | (pl.col("distinct") != LEADS_PER_RUN)
    )
    radiation = lazy.select("lead_hours", "shortwave_radiation")
    counts = radiation.select(
        zero_not_null=(
            (pl.col("lead_hours") == 0) & pl.col("shortwave_radiation").is_not_null()
        ).sum(),
        null_beyond=((pl.col("lead_hours") > 0) & pl.col("shortwave_radiation").is_null()).sum(),
        below=(pl.col("shortwave_radiation") < 0).sum(),
        minimum=pl.col("shortwave_radiation").min(),
    ).collect()
    negative_leads = (
        radiation.filter(pl.col("shortwave_radiation") < 0)
        .select(first=pl.col("lead_hours").min(), last=pl.col("lead_hours").max())
        .collect()
    )
    return SourceFacts(
        runs_present=per_run["init_time"].n_unique(),
        runs_expected=runs_expected,
        incomplete_runs=incomplete.height,
        lead_zero_radiation_not_null=int(counts["zero_not_null"][0]),
        radiation_null_beyond_lead_zero=int(counts["null_beyond"][0]),
        radiation_below_zero=int(counts["below"][0]),
        radiation_minimum=float(counts["minimum"][0]),
        radiation_below_zero_first_lead=negative_leads["first"][0],
        radiation_below_zero_last_lead=negative_leads["last"][0],
    )


def source_verdict(*, facts: SourceFacts) -> list[str]:
    """List the failures of the archive's row checks.

    Args:
        facts: `source_facts`'s result.

    Returns:
        One line per failure: an incomplete (site, run) pair, radiation present at lead 0 or absent
        beyond it. Absent runs are expected, so they are not a failure.
    """
    failures = []
    if facts.incomplete_runs:
        failures.append(f"{facts.incomplete_runs} (site, run) pairs lack some lead or repeat one")
    if facts.lead_zero_radiation_not_null:
        failures.append("radiation is present at lead 0")
    if facts.radiation_null_beyond_lead_zero:
        failures.append(f"{facts.radiation_null_beyond_lead_zero} radiation values are null")
    return failures


def mirror_asymmetry(*, mean_by_label_hour: dict[int, float], pair_sum: int) -> float:
    """Measure how far a daily radiation curve is from mirroring about midday, for one labelling.

    Args:
        mean_by_label_hour: The mean radiation at each label hour of day (UTC), 0 to 23.
        pair_sum: Which label hours pair up: `h` with `pair_sum - h`.

    Returns:
        The summed absolute difference of the paired means over label hours 4 to 20, as a share of
        the summed means over the same hours.
    """
    hours = range(4, 21)
    total = sum(mean_by_label_hour[hour] for hour in hours)
    gap = sum(
        abs(mean_by_label_hour[hour] - mean_by_label_hour[(pair_sum - hour) % 24]) for hour in hours
    )
    return gap / total


def label_hour_means(*, path: Path) -> dict[int, float]:
    """Return the archive's annual mean radiation by label hour of day, over the solar sites."""
    table = (
        pl.scan_parquet(path)
        .filter(~pl.col("site").str.starts_with("W"), pl.col("lead_hours") > 0)
        .group_by(hour=pl.col("valid_time").dt.hour())
        .agg(mean=pl.col("shortwave_radiation").mean())
        .collect()
    )
    return dict(zip(table["hour"].to_list(), table["mean"].to_list(), strict=True))


def hour_ending_verdict(*, mean_by_label_hour: dict[int, float]) -> tuple[list[str], float, float]:
    """Check that a label ends the hour its radiation averages over.

    Args:
        mean_by_label_hour: `label_hour_means`'s result.

    Returns:
        The failures (none if the hour-ending pairing mirrors better), and the asymmetry under the
        hour-ending pairing (`h + h' = 25`) and under the hour-starting pairing (`h + h' = 23`).
    """
    ending = mirror_asymmetry(mean_by_label_hour=mean_by_label_hour, pair_sum=25)
    starting = mirror_asymmetry(mean_by_label_hour=mean_by_label_hour, pair_sum=23)
    failures = [] if ending < starting else ["the daily curve mirrors better if labels start hours"]
    return failures, ending, starting


def stable_step_widths() -> pl.DataFrame:
    """Return each lead from 1 to 239 whose native step width equals its two neighbours' widths.

    Returns:
        `lead_hours` and `step_hours` (`native_step_hours`), leaving out the leads beside a change
        of width (90, 91, 144, and 145), where a triple mixes two widths.
    """
    width = {lead: native_step_hours(lead_hours=lead) for lead in range(LAST_LEAD_HOURS + 1)}
    stable = [
        lead
        for lead in range(1, LAST_LEAD_HOURS)
        if width[lead - 1] == width[lead] == width[lead + 1]
    ]
    return pl.DataFrame(
        {"lead_hours": stable, "step_hours": [width[lead] for lead in stable]},
        schema={"lead_hours": pl.Int32, "step_hours": pl.Int32},
    )


def second_difference_by_step(*, path: Path) -> pl.DataFrame:
    """Return, per native step width, the mean size of the hourly wind speed's second difference.

    The second difference of three consecutive hourly values measures how jagged the hourly series
    is. Weather does not get smoother with lead, so a series that is interpolated from coarser
    native steps has a smaller second difference at the leads where the steps coarsen.

    Args:
        path: The archive's combined parquet.

    Returns:
        `step_hours` (1, 3, or 6, of the middle lead's native step) and `second_difference`, the
        mean absolute second difference of the 100 m wind speed in km/h, pooled over every site and
        run.
    """
    speed = pl.col("wind_speed_100m")
    second = (speed.shift(1) - 2 * speed + speed.shift(-1)).over("site", "init_time").abs()
    return (
        pl.scan_parquet(path)
        .sort("site", "init_time", "lead_hours")
        .with_columns(second_difference=second)
        .join(stable_step_widths().lazy(), on="lead_hours", how="inner")
        .group_by("step_hours")
        .agg(pl.col("second_difference").mean())
        .sort("step_hours")
        .collect()
    )


def native_step_verdict(*, table: pl.DataFrame) -> list[str]:
    """List the failures of the native-step reading.

    Args:
        table: `second_difference_by_step`'s result.

    Returns:
        One line per width (3 and 6 hours) whose second difference is not below its maximum share
        of the hourly width's (`MAX_SECOND_DIFFERENCE_SHARE`).
    """
    size = dict(
        zip(table["step_hours"].to_list(), table["second_difference"].to_list(), strict=True)
    )
    return [
        f"{width}-hourly leads: second difference {size[width]:.3f} is not below "
        f"{limit:.2f} of the hourly leads' {size[1]:.3f}"
        for width, limit in MAX_SECOND_DIFFERENCE_SHARE.items()
        if size[width] >= limit * size[1]
    ]


# --- The built columns -------------------------------------------------------------------------


def _minimum(*, column: pl.Series) -> float:
    """Return a column's minimum, or NaN if every value is null."""
    minimum = column.drop_nulls().to_list()
    return min(minimum) if minimum else float("nan")


def gap_table(*, built_dir: Path, archive_path: Path) -> pl.DataFrame:
    """Count each arm's null rows, and check they are the rows whose serving run is absent.

    Args:
        built_dir: The folder holding `<technology>_extra_lead_inputs.parquet`.
        archive_path: The archive's combined parquet.

    Returns:
        One row per (technology, day) with `rows`, `null_rows`, the number of distinct `gap_days`
        (target dates with a null row), `null_but_run_present` (null rows whose run the archive
        holds), and `filled_but_run_absent` (non-null rows whose run it lacks), and `min_ghi`.
    """
    runs = (
        pl.scan_parquet(archive_path)
        .select(init_time=pl.col("init_time").dt.replace_time_zone("UTC"))
        .unique()
        .collect()["init_time"]
    )
    records = []
    for domain in ("solar", "wind"):
        built = pl.read_parquet(built_dir / f"{domain}_extra_lead_inputs.parquet")
        run_type = built.schema["time"]
        present = runs.cast(run_type)
        for day in IFS_SINGLE_DAYS:
            columns = efh.ens_columns(arm=ifs_single_arm(day=day), domain=domain)
            frame = built.select(
                "time",
                null=pl.any_horizontal(pl.col(c).is_null() for c in columns),
                run_present=served_init_time(time=pl.col("time"), day=day, domain=domain).is_in(
                    present.implode()
                ),
            )
            records.append(
                {
                    "technology": domain,
                    "day": day,
                    "rows": frame.height,
                    "null_rows": int(frame["null"].sum()),
                    "gap_days": frame.filter(pl.col("null"))["time"].dt.date().n_unique(),
                    "null_but_run_present": frame.filter(
                        pl.col("null") & pl.col("run_present")
                    ).height,
                    "filled_but_run_absent": frame.filter(
                        ~pl.col("null") & ~pl.col("run_present")
                    ).height,
                    "min_ghi": (
                        _minimum(column=built[columns[0]]) if domain == "solar" else float("nan")
                    ),
                }
            )
    return pl.DataFrame(records)


def gap_verdict(*, table: pl.DataFrame) -> list[str]:
    """List the failures of the gap check.

    Args:
        table: `gap_table`'s result.

    Returns:
        One line per arm with a null row whose run is present, a non-null row whose run is absent,
        or a radiation below zero.
    """
    failures = []
    for row in table.iter_rows(named=True):
        label = f"{row['technology']} day {row['day']}"
        if row["null_but_run_present"]:
            failures.append(f"{label}: {row['null_but_run_present']} null rows have a run")
        if row["filled_but_run_absent"]:
            failures.append(f"{label}: {row['filled_but_run_absent']} rows filled without a run")
        if row["min_ghi"] < 0:
            failures.append(f"{label}: radiation below zero")
    return failures


class Sample(NamedTuple):
    """One built value to recompute: its column, the archive's site, run and lead, and the value."""

    technology: str
    column: str
    field: str
    day: int
    site: str
    init: datetime
    lead: int
    value: float | None


def _sample_rows(*, built_dir: Path) -> list[Sample]:
    """Pick the built values to recompute: an even stride of rows, every arm column of each."""
    samples: list[Sample] = []
    for domain in ("solar", "wind"):
        built = pl.read_parquet(built_dir / f"{domain}_extra_lead_inputs.parquet")
        picked = built.sort("site", "time").gather_every(max(1, built.height // SAMPLE_ROWS))
        for day in IFS_SINGLE_DAYS:
            for column in efh.ens_columns(arm=ifs_single_arm(day=day), domain=domain):
                field = column.removeprefix(f"{ifs_single_arm(day=day)}_")
                for site, time, value in picked.select("site", "time", column).iter_rows():
                    init, lead = expected_served(time=time, day=day, solar=domain == "solar")
                    samples.append(Sample(domain, column, field, day, site, init, lead, value))
    return samples


def _archive_values(*, path: Path, samples: list[Sample]) -> dict[tuple, dict[str, float]]:
    """Read only the archive's rows the samples need, keyed by (site, run, lead)."""
    keys = pl.DataFrame(
        [
            {"site": s.site, "init_time": s.init.replace(tzinfo=None), "lead_hours": s.lead}
            for s in samples
        ],
        schema={"site": pl.String, "init_time": pl.Datetime("us"), "lead_hours": pl.Int32},
    ).unique()
    archive = (
        pl.scan_parquet(path).join(keys.lazy(), on=["site", "init_time", "lead_hours"], how="semi")
    ).collect()
    return {
        (row["site"], row["init_time"].replace(tzinfo=UTC), row["lead_hours"]): row
        for row in archive.iter_rows(named=True)
    }


def _expected_value(*, values: dict[tuple, dict[str, float]], sample: Sample) -> float | None:
    """Recompute one built value from the archive, or None if the archive lacks the row."""
    row = values.get((sample.site, sample.init, sample.lead))
    if row is None:
        return None
    recompute: dict[str, Callable[[], float]] = {
        "ghi": lambda: max(row["shortwave_radiation"], 0.0),
        "temp": lambda: row["temperature_2m"],
        "speed_100m": lambda: row["wind_speed_100m"] * KMH_TO_MS,
        "speed_10m": lambda: row["wind_speed_10m"] * KMH_TO_MS,
        "sin_100m": lambda: math.sin(math.radians(row["wind_direction_100m"])),
        "cos_100m": lambda: math.cos(math.radians(row["wind_direction_100m"])),
    }
    return recompute[sample.field]()


def lookup_table(*, built_dir: Path, archive_path: Path) -> pl.DataFrame:
    """Recompute a sample of the built columns from the archive in plain Python and compare.

    Args:
        built_dir: The folder holding the fourth batch's `<technology>_extra_lead_inputs.parquet`.
        archive_path: The archive's combined parquet.

    Returns:
        One row per (technology, day, column), with the values compared, those that differ by more
        than `LOOKUP_TOLERANCE`, and those skipped because the archive or built column lacks one.
    """
    samples = _sample_rows(built_dir=built_dir)
    values = _archive_values(path=archive_path, samples=samples)
    tallies: dict[tuple[str, int, str], list[int]] = {}
    for sample in samples:
        tally = tallies.setdefault((sample.technology, sample.day, sample.column), [0, 0, 0])
        expected = _expected_value(values=values, sample=sample)
        if expected is None or sample.value is None:
            tally[2] += 1
            continue
        tally[0] += 1
        tally[1] += abs(expected - sample.value) > LOOKUP_TOLERANCE
    return pl.DataFrame(
        [
            {
                "technology": technology,
                "day": day,
                "column": column,
                "compared": compared,
                "differing": differing,
                "skipped": skipped,
            }
            for (technology, day, column), (compared, differing, skipped) in tallies.items()
        ]
    )


def lookup_verdict(*, table: pl.DataFrame) -> list[str]:
    """List the built columns that differ from a plain-Python recomputation, or were never compared.

    Args:
        table: `lookup_table`'s result.

    Returns:
        One line per column with any differing row or with no compared row.
    """
    return [
        f"{row['technology']} day {row['day']} {row['column']}: {row['differing']} of "
        f"{row['compared']} differ"
        for row in table.iter_rows(named=True)
        if row["differing"] or not row["compared"]
    ]


def _markdown(*, frame: pl.DataFrame) -> list[str]:
    """Render a frame as a Markdown table."""
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = [
        "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for row in frame.iter_rows()
    ]
    return [header, rule, *body]


def _verdict_line(*, failures: list[str], ok: str) -> str:
    """Return the verdict line of a file: `ok` if no failure, else the failures."""
    return "**Verdict:** " + (ok if not failures else "FAILS: " + "; ".join(failures))


def main() -> int:
    """Run the checks and write their Markdown files; exit non-zero if any verdict fails."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--built-dir", type=Path, default=None)
    parser.add_argument("--ifs-single-dir", type=Path, default=None)
    args = parser.parse_args()
    directory = (
        args.ifs_single_dir or _repo_data_dir() / "studies" / "weather" / IFS_SINGLE_DIR_NAME
    )
    archive_path = directory / IFS_SINGLE_FILE_NAME
    verification = args.output_dir / "verification"
    verification.mkdir(parents=True, exist_ok=True)

    facts = source_facts(path=archive_path)
    means = label_hour_means(path=archive_path)
    hour_failures, ending, starting = hour_ending_verdict(mean_by_label_hour=means)
    steps = second_difference_by_step(path=archive_path)
    failures = [
        *source_verdict(facts=facts),
        *hour_failures,
        *native_step_verdict(table=steps),
    ]
    (verification / "ifs_single_source.md").write_text(
        "\n".join(
            [
                "# IFS HRES (9 km, Open-Meteo): the archive's rows",
                "",
                (
                    f"- Runs present: {facts.runs_present} of {facts.runs_expected} run days in "
                    f"its span; {facts.runs_expected - facts.runs_present} run days are absent "
                    "(gaps)."
                ),
                (
                    "- (Site, run) pairs without every lead 0 to 240 exactly once: "
                    f"{facts.incomplete_runs}."
                ),
                (
                    f"- Radiation present at lead 0: {facts.lead_zero_radiation_not_null}; null "
                    f"beyond lead 0: {facts.radiation_null_beyond_lead_zero}."
                ),
                (
                    f"- Radiation below zero: {facts.radiation_below_zero} values, the lowest "
                    f"{facts.radiation_minimum}, at leads "
                    f"{facts.radiation_below_zero_first_lead} to "
                    f"{facts.radiation_below_zero_last_lead}. The build clips them at zero."
                ),
                (
                    "- Hour-ending label: asymmetry of the daily radiation curve is "
                    f"{ending:.4f} when labels end their hour and {starting:.4f} when they "
                    "start it."
                ),
                "",
                (
                    "Mean absolute second difference of the hourly 100 m wind speed (km/h), by "
                    "the native step width of the middle lead. A smaller value at the coarser "
                    "widths shows the archive interpolates them to hourly:"
                ),
                "",
                *_markdown(frame=steps),
                "",
                _verdict_line(
                    failures=failures,
                    ok="the archive is complete, radiation follows the hour-ending convention, "
                    "and hourly values beyond lead 90 are interpolated.",
                ),
            ]
        )
        + "\n"
    )

    served = served_table(first_day=datetime(2025, 3, 10, tzinfo=UTC))
    served_failures = served_verdict(table=served)
    failures += served_failures
    (verification / "ifs_single_served_runs.md").write_text(
        "\n".join(
            [
                "# IFS HRES (9 km, Open-Meteo): the run and lead each target hour reads",
                "",
                (
                    "Each technology at each lead day, for the 24 target labels of one date. "
                    "Day 10 is tabulated to show that its leads pass 240."
                ),
                "",
                *_markdown(frame=served.drop("agrees")),
                "",
                _verdict_line(
                    failures=served_failures,
                    ok="both expressions of the rule agree, every servable day reads its stated "
                    "leads, and day 10 is not servable.",
                ),
            ]
        )
        + "\n"
    )

    if args.built_dir is not None:
        gaps = gap_table(built_dir=args.built_dir, archive_path=archive_path)
        lookups = lookup_table(built_dir=args.built_dir, archive_path=archive_path)
        built_failures = [*gap_verdict(table=gaps), *lookup_verdict(table=lookups)]
        failures += built_failures
        (verification / "ifs_single_built_columns.md").write_text(
            "\n".join(
                [
                    "# IFS HRES (9 km, Open-Meteo): gap rows and built columns",
                    "",
                    (
                        "Gap rows, by count only: `null_rows` are target hours whose serving "
                        "run the archive lacks."
                    ),
                    "",
                    *_markdown(frame=gaps),
                    "",
                    "Built columns against a plain-Python recomputation:",
                    "",
                    *_markdown(frame=lookups),
                    "",
                    _verdict_line(
                        failures=built_failures,
                        ok="every null row is a gap, no gap is filled, and every compared value "
                        "matches.",
                    ),
                ]
            )
            + "\n"
        )
    _LOG.info("wrote %s", verification)
    if failures:
        _LOG.error("IFS HRES (9 km, Open-Meteo) verification failed: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
