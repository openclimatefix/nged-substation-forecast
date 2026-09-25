"""Check the native GFS arms' radiation windows and run mapping from the data on disk.

One-off throwaway script for the native GFS arms of
<https://github.com/openclimatefix/nged-substation-forecast/issues/912>. It reads and never fits,
and writes nothing under `data/` except its own Markdown files in `<output-dir>/verification/`. Run
it after `build_forecast_inputs.py --extra-leads --batch third` and before `fit_extra_leads.py
--batch third`. It writes three Markdown files and exits non-zero if any verdict fails:

1. `gfs_radiation_window.md`: whether the store's radiation is a mean since the last 6-hourly
   reset, with the lead labelling the window's end, as `studies.gfs_native.step_means` assumes.
   Every number pools every grid cell of the crop, every run at 00, 06, 12, and 18 UTC, and every
   month, so it prints no site and no coordinate. It compares, at each valid hour of day, the mean
   radiation at the hourly leads (1 to 120 h) with the mean at the 3-hourly leads (123 h onward),
   which must agree if both zones use the reset rule. At a valid hour where the reset rule's window
   is 6 hours, the mean of the last 3 hours (worked out from the hourly leads) differs from the
   window mean by at least 15%, so a 3-hourly value that were a plain 3-hour mean would sit closer
   to the 3-hour figure, and the check fails. A second table gives, for each window length of 1 to
   6 hours (hours where the window is shorter than 6 hours included), the share of recovered step
   means below -3 W/m2 before clipping, which must be small if the windows are right, and shows
   that lead 0 holds no radiation while it holds temperature and wind.
2. `gfs_served_runs.md`: the run and lead each target hour reads at each lead day, for solar and
   wind, worked out twice (`studies.gfs_native`'s expressions and plain Python) and required to
   agree with the rule stated in `build_forecast_inputs._gfs_native_frame`.
3. `gfs_built_columns.md`: with `--built-dir`, a sample of rows of the built columns recomputed
   straight from the store by plain Python (its own run and lead arithmetic, and its own hand
   inversion of a single window), which must match: solar radiation and temperature at days 0 to 4,
   and wind at every day whose target hour falls on a step of the run (all of days 0 to 4, and one
   hour in three at days 5 and above, where the arm's upsampling passes through the steps).

Run it with `uv run python studies/nwp_forecast_comparison/verify_gfs_native.py --output-dir DIR
[--built-dir DIR]`.
"""

import argparse
import logging
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "beam_diffuse_split"))
import ens_forecast_horizons as efh
from build_forecast_inputs import (
    GFS_NATIVE_DAYS,
    GFS_NATIVE_DIR_NAME,
    _gefs_cell_selection,
    _repo_data_dir,
    gfs_native_arm,
)
from studies.gfs_native import (
    HOURLY_SERVED_LAST_DAY,
    LAST_HOURLY_LEAD_HOURS,
    gfs_leads,
    served_init_time,
    served_lead_hours,
    step_means,
    window_hours,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

RADIATION: Final[str] = "downward_short_wave_radiation_flux_surface"
"""The store's radiation column."""

MIN_WINDOW_DIFFERENCE: Final[float] = 0.15
"""A valid hour tells a 6-hour window from a 3-hour window only if their mean radiation differs by
at least this share of the 6-hour window's."""

MIN_MEAN_W_M2: Final[float] = 20.0
"""Below this mean radiation a valid hour is night-time and carries no evidence either way."""

MAX_RATIO_ERROR: Final[float] = 0.10
"""How far the 3-hourly zone's mean radiation may sit from the hourly zone's, as a share of it."""

NIGHT_HOURS: Final[tuple[int, ...]] = (22, 23, 0, 1, 2)
"""Valid hours (UTC) at which the sun is below the horizon on every day of the year in the trial
area, so the recovered step mean must be near zero."""

MAX_NIGHT_W_M2: Final[float] = 5.0
"""The largest mean recovered step radiation at a night hour."""

NEGATIVE_W_M2: Final[float] = -3.0
"""A recovered step mean below this before clipping counts as negative. The store's radiation is
quantised to about 1 W/m2, and inverting a 6-hour window multiplies that error by up to 11, so a
dark hour at the end of a sunlit window comes back as low as -8 W/m2 (winter, 17 to 18 UTC). On the
full store 0.13% of 6-hour steps fall below -3 W/m2; a window offset by 3 hours puts 10.6% below
-8 W/m2."""

MAX_NEGATIVE_SHARE: Final[float] = 0.01
"""The largest share of recovered step means below `NEGATIVE_W_M2` at one window length."""

SOLAR_NOON_HOURS: Final[tuple[int, int]] = (12, 13)
"""The valid hours (UTC, the hour ending at the label) at which the mean step radiation may peak."""

SAMPLE_ROWS: Final[int] = 400
"""How many rows of each built column the lookup check recomputes, per technology and day."""

LOOKUP_TOLERANCE: Final[float] = 1e-3
"""How close a recomputed value must be to the built one, in its own unit (the store's 13-bit
rounding is applied before both, so a match is exact up to float arithmetic)."""


def window_length(*, valid_hour: int) -> int:
    """Return the reset window's length at a valid hour of day, from the 6-hourly reset rule.

    Args:
        valid_hour: The hour of day (UTC) at which the window ends, 0 to 23.

    Returns:
        1 to 6: 1 an hour after a reset (hours 1, 7, 13, and 19) and 6 at the reset itself.
    """
    return (valid_hour + 23) % 6 + 1


def recovered_step_means(*, window_means: dict[int, float]) -> dict[int, float]:
    """Recover the mean over each hour of the day from the mean of each reset window.

    Args:
        window_means: The mean radiation of the window ending at each valid hour of day, all 24.

    Returns:
        The mean over the hour ending at each valid hour: the window's own mean where it is 1 hour,
        else the window's total less the total of the window as it stood an hour earlier.
    """
    recovered = {}
    for hour in range(24):
        length = window_length(valid_hour=hour)
        earlier = window_means[(hour - 1) % 24]
        recovered[hour] = (
            window_means[hour]
            if length == 1
            else length * window_means[hour] - (length - 1) * earlier
        )
    return recovered


def window_table(*, path: Path) -> pl.DataFrame:
    """Return the mean radiation at each valid hour, in the hourly and the 3-hourly leads.

    Args:
        path: The store's `GFS.parquet`.

    Returns:
        One row per valid hour 0 to 23, with `window_hours` (the reset rule's), `hourly_zone` (mean
        radiation at leads 1 to 120 h), `coarse_zone` (mean at leads beyond 120 h, null at the hours
        a 3-hourly lead never lands on), `step_mean` (the hourly zone's mean over the hour ending
        there), `last_three_hours` (its mean over the 3 hours ending there), and the row counts.
    """
    lead_hours = (pl.col("lead_time").dt.total_minutes() / 60).cast(pl.Int32)
    pooled = (
        pl.scan_parquet(path)
        .select(
            lead_hours=lead_hours,
            valid_hour=(pl.col("init_time").dt.hour() + lead_hours) % 24,
            value=pl.col(RADIATION),
        )
        .filter(pl.col("value").is_not_nan(), pl.col("lead_hours") > 0)
        .group_by("valid_hour", (pl.col("lead_hours") > LAST_HOURLY_LEAD_HOURS).alias("coarse"))
        .agg(mean=pl.col("value").mean(), n=pl.len())
        .collect()
    )
    hourly = pooled.filter(~pl.col("coarse"))
    coarse = pooled.filter(pl.col("coarse"))
    hourly_mean = dict(zip(hourly["valid_hour"], hourly["mean"], strict=True))
    coarse_mean = dict(zip(coarse["valid_hour"], coarse["mean"], strict=True))
    step = recovered_step_means(window_means=hourly_mean)
    return pl.DataFrame(
        [
            {
                "valid_hour": hour,
                "window_hours": window_length(valid_hour=hour),
                "hourly_zone": hourly_mean[hour],
                "coarse_zone": coarse_mean.get(hour),
                "step_mean": step[hour],
                "last_three_hours": float(np.mean([step[(hour - back) % 24] for back in range(3)])),
                "n_hourly": dict(zip(hourly["valid_hour"], hourly["n"], strict=True))[hour],
            }
            for hour in range(24)
        ]
    )


def window_verdict(*, table: pl.DataFrame) -> list[str]:
    """List the failures of the reset-window reading in `window_table`'s result.

    Args:
        table: `window_table`'s result.

    Returns:
        One line per failure; empty when the reading holds. A valid hour with a 3-hourly lead and
        an hourly-zone mean of at least `MIN_MEAN_W_M2` fails if its 3-hourly mean is more than
        `MAX_RATIO_ERROR` from the hourly zone's, or, where the reset window's mean and the mean
        of the last 3 hours differ by `MIN_WINDOW_DIFFERENCE`, closer to the last 3 hours. The
        mean step radiation must also fall below `MAX_NIGHT_W_M2` at every `NIGHT_HOURS` and peak
        at a `SOLAR_NOON_HOURS` hour.
    """
    failures: list[str] = []
    for row in table.iter_rows(named=True):
        coarse, window, three = row["coarse_zone"], row["hourly_zone"], row["last_three_hours"]
        if coarse is None or window < MIN_MEAN_W_M2:
            continue
        if abs(coarse - window) > MAX_RATIO_ERROR * window:
            failures.append(
                f"valid hour {row['valid_hour']:02d} UTC: 3-hourly mean {coarse:.1f} W/m2 is more "
                f"than {MAX_RATIO_ERROR:.0%} from the hourly zone's {window:.1f}"
            )
        elif abs(three - window) >= MIN_WINDOW_DIFFERENCE * window and abs(coarse - three) < abs(
            coarse - window
        ):
            failures.append(
                f"valid hour {row['valid_hour']:02d} UTC: 3-hourly mean {coarse:.1f} W/m2 is "
                f"closer to a plain 3-hour mean {three:.1f} than to the reset window's {window:.1f}"
            )
    step = dict(zip(table["valid_hour"], table["step_mean"], strict=True))
    failures.extend(
        f"night hour {hour:02d} UTC: mean step radiation {step[hour]:.2f} W/m2 exceeds "
        f"{MAX_NIGHT_W_M2} W/m2"
        for hour in NIGHT_HOURS
        if step[hour] > MAX_NIGHT_W_M2
    )
    peak = max(step, key=lambda hour: step[hour])
    if peak not in SOLAR_NOON_HOURS:
        failures.append(f"mean step radiation peaks at valid hour {peak:02d} UTC, not near noon")
    return failures


def negative_share_table(*, path: Path) -> pl.DataFrame:
    """Return, by window length, how often a recovered step mean is negative before clipping.

    This check cannot detect two plausible defects on its own: values that are already step
    means, or resets every 3 hours. Both produce no negatives. The night check catches them.

    Args:
        path: The store's `GFS.parquet`.

    Returns:
        One row per window length 1 to 6 with `share_negative` (recovered means below
        `NEGATIVE_W_M2`, over the hourly leads only, daylight or not), `share_missing`, and `rows`,
        plus a row `lead_zero` holding the share of lead-0 radiation, temperature, and wind values
        that are present (the `window_hours` column is 0 there).
    """
    lead_hours = (pl.col("lead_time").dt.total_minutes() / 60).cast(pl.Int32)
    frame = pl.read_parquet(
        path,
        columns=[
            "init_time",
            "lead_time",
            "lat_index",
            "lon_index",
            RADIATION,
            "temperature_2m",
            "wind_u_10m",
            "wind_u_100m",
        ],
    ).with_columns(lead_hours=lead_hours)
    radiation = frame.filter(pl.col("lead_hours") > 0).select(
        "lat_index",
        "lon_index",
        "init_time",
        "lead_hours",
        ghi_raw=pl.col(RADIATION).cast(pl.Float64).fill_nan(None),
    )
    leads = gfs_leads()
    wide = radiation.pivot(
        on="lead_hours", index=["lat_index", "lon_index", "init_time"], values="ghi_raw"
    )
    values = np.full((wide.height, len(leads)), np.nan)
    for position, lead in enumerate(leads):
        if (name := str(int(lead))) in wide.columns:
            values[:, position] = wide[name].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    recovered = step_means(values=values, leads=leads, negative_floor=-np.inf)
    window = window_hours(leads=leads)
    hourly = leads <= LAST_HOURLY_LEAD_HOURS
    records = []
    for length in range(1, 7):
        block = recovered[:, hourly & (window == length)]
        records.append(
            {
                "window_hours": length,
                "share_negative": float(np.mean(block[~np.isnan(block)] < NEGATIVE_W_M2)),
                "share_missing": float(np.mean(np.isnan(block))),
                "rows": int(block.size),
            }
        )
    zero = frame.filter(pl.col("lead_hours") == 0)
    present = {
        column: float(zero[column].is_not_nan().mean())  # ty: ignore[invalid-argument-type]
        for column in (RADIATION, "temperature_2m", "wind_u_10m", "wind_u_100m")
    }
    records.append(
        {
            "window_hours": 0,
            "share_negative": present[RADIATION],
            "share_missing": present["temperature_2m"],
            "rows": zero.height,
        }
    )
    return pl.DataFrame(records)


def negative_share_verdict(*, table: pl.DataFrame) -> list[str]:
    """List the window lengths whose recovered step means are too often negative, and lead-0 faults.

    Args:
        table: `negative_share_table`'s result.

    Returns:
        One line per failure: a window length with more than `MAX_NEGATIVE_SHARE` of its recovered
        means negative, lead-0 radiation that is present, or lead-0 temperature that is missing.
    """
    failures = []
    for row in table.iter_rows(named=True):
        if row["window_hours"] == 0:
            if row["share_negative"] > 0.0:
                failures.append(f"lead 0 holds radiation in {row['share_negative']:.2%} of rows")
            if row["share_missing"] < 0.99:
                failures.append(f"lead 0 holds temperature in only {row['share_missing']:.2%}")
        elif row["share_negative"] > MAX_NEGATIVE_SHARE:
            failures.append(
                f"window {row['window_hours']} h: {row['share_negative']:.2%} of recovered step "
                f"means are below {NEGATIVE_W_M2} W/m2"
            )
    return failures


def expected_served(*, time: datetime, day: int, solar: bool) -> tuple[datetime, int]:
    """Work out, in plain Python, the run and lead a target hour reads at a lead day.

    Args:
        time: The target hour's label (UTC).
        day: The lead day.
        solar: Whether the hour is a solar hour, labelled by its end.

    Returns:
        The run's start and the lead in hours.
    """
    instant = time - timedelta(hours=1) if solar else time
    if day == 0:
        init = instant.replace(hour=instant.hour // 6 * 6, minute=0, second=0, microsecond=0)
    else:
        init = instant.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=day)
    return init, int((time - init).total_seconds() // 3600)


def served_table(*, first_day: datetime) -> pl.DataFrame:
    """Tabulate the run and lead of every hour of one date at every lead day, both ways.

    Args:
        first_day: The date (UTC midnight, timezone-aware) whose 24 target labels are tabulated:
            01:00 to 00:00 the next day for solar and 00:00 to 23:00 for wind.

    Returns:
        One row per (technology, day, label), with the run's start and lead from
        `studies.gfs_native`, the same from `expected_served`, and `agrees`.
    """
    records = []
    for domain in ("solar", "wind"):
        offset = 1 if domain == "solar" else 0
        frame = pl.DataFrame({"time": [first_day + timedelta(hours=k + offset) for k in range(24)]})
        for day in GFS_NATIVE_DAYS:
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
        One line per failure. Two expressions of the rule must agree on every row. Day 0's leads
        must be 1 to 6 (solar) and 0 to 5 (wind) from a 00, 06, 12, or 18 UTC run. A day of 1 or
        more must read one 00 UTC run at 24 consecutive leads starting at `24 * day` (wind) or
        `24 * day + 1` (solar).
    """
    failures = [
        f"{row['technology']} day {row['day']} {row['label']}: rules disagree"
        for row in table.filter(~pl.col("agrees")).iter_rows(named=True)
    ]
    for (technology, day), group in table.group_by("technology", "day"):
        leads = sorted(group["lead"].to_list())
        first = (24 * day + 1) if technology == "solar" else 24 * day
        if day == 0:
            lead_range = list(range(1, 7)) if technology == "solar" else list(range(6))
            expected = sorted(lead_range * 4)
            if any(init.hour % 6 for init in group["init"]):
                failures.append(f"{technology} day 0 reads a run not at 00, 06, 12, or 18 UTC")
        else:
            expected = list(range(first, first + 24))
            if group["init"].n_unique() != 1 or group["init"][0].hour != 0:
                failures.append(f"{technology} day {day} does not read one 00 UTC run")
        if leads != expected:
            failures.append(f"{technology} day {day}: leads {leads} are not {expected}")
    return failures


class Sample(NamedTuple):
    """One built value to recompute: which column, and the store's cell, run and lead behind it."""

    technology: str
    column: str
    field: str
    day: int
    cell: int
    init: datetime
    lead: int
    value: float | None


def _sample_rows(*, built_dir: Path, gfs_dir: Path) -> list[Sample]:
    """Pick the built values to recompute: an even stride of rows, every arm column of each.

    Args:
        built_dir: The folder holding `<technology>_extra_lead_inputs.parquet`.
        gfs_dir: The native GFS store's folder, for `_grid_cells.parquet`.

    Returns:
        One entry per (row, column) compared: solar at days 0 to `HOURLY_SERVED_LAST_DAY` only, and
        wind rows whose lead is a multiple of 3 hours (a step of the run, where the arm passes
        through the store's values).
    """
    grid_cells = pl.read_parquet(gfs_dir / "_grid_cells.parquet")
    samples: list[Sample] = []
    for domain in ("solar", "wind"):
        built = pl.read_parquet(built_dir / f"{domain}_extra_lead_inputs.parquet")
        cell_by_site = _gefs_cell_selection(
            grid_cells=grid_cells, domain=domain, sites=sorted(built["site"].unique().to_list())
        )
        picked = built.sort("site", "time").gather_every(max(1, built.height // SAMPLE_ROWS))
        for day in GFS_NATIVE_DAYS:
            if domain == "solar" and day > HOURLY_SERVED_LAST_DAY:
                continue
            for column in efh.ens_columns(arm=gfs_native_arm(day=day), domain=domain):
                field = column.removeprefix(f"{gfs_native_arm(day=day)}_")
                for site, time, value in picked.select("site", "time", column).iter_rows():
                    init, lead = expected_served(time=time, day=day, solar=domain == "solar")
                    if domain == "wind" and lead % 3:
                        continue
                    samples.append(
                        Sample(domain, column, field, day, cell_by_site[site], init, lead, value)
                    )
    return samples


def _store_values(*, gfs_dir: Path, samples: list[Sample]) -> dict[tuple, dict[str, float]]:
    """Read only the store's rows the samples need, keyed by (cell, run, lead).

    Args:
        gfs_dir: The native GFS store's folder.
        samples: `_sample_rows`'s result.

    Returns:
        Each needed (cell, run, lead) to its columns' values (NaN as None). A solar sample also
        needs the lead an hour earlier, for temperature and for a window's earlier mean.
    """
    wanted = {(s.cell, s.init, s.lead - back) for s in samples for back in (0, 1)}
    keys = pl.DataFrame(
        [{"cell": c, "init_time": i, "lead_hours": lead} for c, i, lead in wanted],
        schema={
            "cell": pl.Int32,
            "init_time": pl.Datetime("ns", "UTC"),
            "lead_hours": pl.Int32,
        },
    )
    lead_hours = (pl.col("lead_time").dt.total_minutes() / 60).cast(pl.Int32)
    store = (
        pl.scan_parquet(gfs_dir / "GFS.parquet")
        .with_columns(
            lead_hours=lead_hours,
            cell=(pl.col("lat_index") * 10 + pl.col("lon_index")).cast(pl.Int32),
            init_time=pl.col("init_time").dt.replace_time_zone("UTC"),
        )
        .join(keys.lazy(), on=["cell", "init_time", "lead_hours"], how="semi")
        .collect()
        .with_columns(pl.col(pl.Float32).cast(pl.Float64).fill_nan(None))
    )
    return {
        (row["cell"], row["init_time"], row["lead_hours"]): row
        for row in store.iter_rows(named=True)
    }


def _hand_step_mean(*, values: dict[tuple, dict[str, float]], sample: Sample) -> float | None:
    """Recover one hourly step mean of the store's radiation by hand, at a lead up to 120."""
    window = (sample.lead - 1) % 6 + 1
    current = values.get((sample.cell, sample.init, sample.lead), {}).get(RADIATION)
    if window == 1 or current is None:
        return current
    earlier = values.get((sample.cell, sample.init, sample.lead - 1), {}).get(RADIATION)
    return None if earlier is None else max(window * current - (window - 1) * earlier, 0.0)


def _solar_value(*, values: dict[tuple, dict[str, float]], sample: Sample) -> float | None:
    """Recompute one solar column: the hand-inverted radiation, or the midpoint temperature."""
    if sample.field == "ghi":
        return _hand_step_mean(values=values, sample=sample)
    here = values.get((sample.cell, sample.init, sample.lead), {}).get("temperature_2m")
    before = values.get((sample.cell, sample.init, sample.lead - 1), {}).get("temperature_2m")
    return None if here is None or before is None else (here + before) / 2.0


def _wind_value(*, values: dict[tuple, dict[str, float]], sample: Sample) -> float | None:
    """Recompute one wind column from the store's components at the sample's own lead."""
    here = values.get((sample.cell, sample.init, sample.lead), {})
    u100, v100 = here.get("wind_u_100m"), here.get("wind_v_100m")
    u10, v10 = here.get("wind_u_10m"), here.get("wind_v_10m")
    if u100 is None or v100 is None or u10 is None or v10 is None:
        return None
    speed_100m = math.hypot(u100, v100)
    result: float | None = None
    if sample.field == "speed_100m":
        result = speed_100m
    elif sample.field == "speed_10m":
        result = math.hypot(u10, v10)
    elif speed_100m != 0.0:
        # The wind blows from the bearing whose sine is -u / speed and whose cosine is -v / speed.
        result = -u100 / speed_100m if sample.field == "sin_100m" else -v100 / speed_100m
    return result


def _expected_value(*, values: dict[tuple, dict[str, float]], sample: Sample) -> float | None:
    """Recompute one built value from the store, or None if the store lacks an input."""
    if sample.technology == "solar":
        return _solar_value(values=values, sample=sample)
    return _wind_value(values=values, sample=sample)


def lookup_table(*, built_dir: Path, gfs_dir: Path) -> pl.DataFrame:
    """Recompute a sample of the built columns from the store in plain Python and compare.

    Args:
        built_dir: The folder holding `<technology>_extra_lead_inputs.parquet` from the third batch.
        gfs_dir: The native GFS store's folder.

    Returns:
        One row per (technology, day, column), with the values compared, those that differ by more
        than `LOOKUP_TOLERANCE`, and those skipped because the store or the built column lacks one.
    """
    samples = _sample_rows(built_dir=built_dir, gfs_dir=gfs_dir)
    values = _store_values(gfs_dir=gfs_dir, samples=samples)
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


def main() -> int:
    """Run the checks and write their Markdown files; exit non-zero if any verdict fails."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--built-dir", type=Path, default=None)
    parser.add_argument("--gfs-dir", type=Path, default=None)
    args = parser.parse_args()
    gfs_dir = _repo_data_dir() / "studies" / "weather" / GFS_NATIVE_DIR_NAME
    gfs_dir = args.gfs_dir or gfs_dir
    verification = args.output_dir / "verification"
    verification.mkdir(parents=True, exist_ok=True)

    windows = window_table(path=gfs_dir / "GFS.parquet")
    negatives = negative_share_table(path=gfs_dir / "GFS.parquet")
    failures = [
        *window_verdict(table=windows),
        *negative_share_verdict(table=negatives),
    ]
    (verification / "gfs_radiation_window.md").write_text(
        "\n".join(
            [
                "# GFS native radiation: reset windows",
                "",
                "Mean radiation (W/m2) pooled over every grid cell, run, and month, by valid hour.",
                "",
                *_markdown(frame=windows),
                "",
                (
                    "By window length: `share_negative` is the share of recovered step means "
                    f"below {NEGATIVE_W_M2} W/m2 before clipping. The row with window 0 is lead "
                    "0: `share_negative` holds the share of radiation values present, and "
                    "`share_missing` the share of temperature values present."
                ),
                "",
                *_markdown(frame=negatives),
                "",
                "**Verdict:** "
                + (
                    "the store's radiation is a mean since the last 6-hourly reset, labelled at "
                    "the window's end."
                    if not failures
                    else "the reset-window reading FAILS: " + "; ".join(failures)
                ),
            ]
        )
        + "\n"
    )

    served = served_table(first_day=datetime(2025, 3, 10, tzinfo=UTC))
    served_failures = served_verdict(table=served)
    failures += served_failures
    (verification / "gfs_served_runs.md").write_text(
        "\n".join(
            [
                "# Native GFS: the run and lead each target hour reads",
                "",
                "Each technology at each lead day, for the 24 target labels of one date.",
                "",
                *_markdown(frame=served.drop("agrees")),
                "",
                "**Verdict:** "
                + (
                    "both expressions of the rule agree and every day reads its stated leads."
                    if not served_failures
                    else "FAILS: " + "; ".join(served_failures)
                ),
            ]
        )
        + "\n"
    )

    if args.built_dir is not None:
        lookups = lookup_table(built_dir=args.built_dir, gfs_dir=gfs_dir)
        lookup_failures = lookup_verdict(table=lookups)
        failures += lookup_failures
        (verification / "gfs_built_columns.md").write_text(
            "\n".join(
                [
                    "# Native GFS: built columns against a plain-Python recomputation",
                    "",
                    *_markdown(frame=lookups),
                    "",
                    "**Verdict:** "
                    + (
                        "every compared value matches."
                        if not lookup_failures
                        else "FAILS: " + "; ".join(lookup_failures)
                    ),
                ]
            )
            + "\n"
        )
    _LOG.info("wrote %s", verification)
    if failures:
        _LOG.error("native GFS verification failed: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
