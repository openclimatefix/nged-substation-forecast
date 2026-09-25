"""Validate the combined IFS Single Runs parquet that `fetch_open_meteo_single_runs.py` wrote.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>. It reads
`data/studies/weather/ECMWF-IFS-SINGLE-RUNS/ECMWF-IFS-SINGLE-RUNS.parquet` and runs the checks in
`CHECK_NAMES`: run spacing (every run day missing from the first day the archive serves listed,
and any gap in neither ledger a failure), 241 leads per run per site, all seven variables, nulls
only in radiation at lead 0, physical ranges, a diurnal check on radiation, and nine site labels in
every month file.

**The script prints one PASS or FAIL line per check, then every gap.** Gaps are listed rather than
hidden. It prints no coordinate. It exits non-zero when any check fails.

Run it with `uv run python studies/weather_downloads/validate_ifs_single_runs.py`.
"""

import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl
from fetch_open_meteo_single_runs import (
    BASE_VARIABLES,
    COMBINED_FILENAME,
    FIRST_RUN_DATE,
    INCOMPLETE_FILENAME,
    LEADS_PER_RUN,
    PRODUCT_DIR,
    RADIATION_VARIABLES,
    TRAILING_DAYS_MAY_BE_INCOMPLETE,
    UNAVAILABLE_FILENAME,
)
from studies.anonymise import SITE_LABELS, WIND_SITE_LABELS

EXPECTED_SITES: Final[frozenset[str]] = frozenset(SITE_LABELS + WIND_SITE_LABELS)

MAX_RADIATION_W_M2: Final[float] = 1400.0
TEMPERATURE_RANGE_C: Final[tuple[float, float]] = (-40.0, 50.0)
MAX_WIND_SPEED_KM_H: Final[float] = 200.0
MAX_DIRECTION_DEG: Final[float] = 360.0
"""Physical bounds. A value outside them is a fault, not an extreme."""

NIGHT_HOURS_UTC: Final[tuple[int, ...]] = (0, 1, 2, 22, 23)
NIGHT_MAX_MEAN_RADIATION_W_M2: Final[float] = 1.0
"""Around midnight UTC the sun is below the horizon in Great Britain all year, so a mean above 1
W/m^2 means the timestamps are shifted. The ceiling holds for a subset of months as well as for the
whole archive, because the sun is below the horizon at 22 to 02 UTC even at the June solstice."""

NOON_HOURS_UTC: Final[tuple[int, ...]] = (11, 12, 13)
NOON_MIN_MEAN_RADIATION_W_M2: Final[float] = 20.0
"""The mean over all runs and sites around solar noon. A December-only subset still averages well
above this (short days, low sun, cloud), so a mean below it means night values at midday."""

CHECK_NAMES: Final[tuple[str, ...]] = (
    "run_spacing",
    "leads_per_run_per_site",
    "columns_present",
    "no_unexpected_nulls",
    "lead_zero_radiation_null",
    "value_ranges",
    "diurnal_radiation",
    "nine_sites_every_month",
    "no_duplicate_leads",
    "valid_time_matches_lead",
    "sites_not_constant",
)
"""Every check, in report order."""


def _run_days(*, frame: pl.DataFrame) -> list[date]:
    """Return the distinct run days in the frame, sorted."""
    return sorted(frame["init_time"].dt.date().unique().to_list())


def _gaps(*, days: list[date]) -> list[date]:
    """Return every run day missing from `FIRST_RUN_DATE` to `TRAILING_DAYS_MAY_BE_INCOMPLETE` ago.

    The span starts at the first day the archive serves, not at the first run fetched, so days
    missing before the first run and after the last run count as gaps too.
    """
    present = set(days)
    end = datetime.now(UTC).date() - timedelta(days=TRAILING_DAYS_MAY_BE_INCOMPLETE)
    span = (end - FIRST_RUN_DATE).days
    return [
        day
        for day in (FIRST_RUN_DATE + timedelta(days=n) for n in range(span + 1))
        if day not in present
    ]


def _read_ledger(*, filename: str) -> dict[date, str]:
    """Return the fetch script's ledger of run days to reasons, or an empty one."""
    path = PRODUCT_DIR / filename
    if not path.exists():
        return {}
    return {date.fromisoformat(day): reason for day, reason in json.loads(path.read_text()).items()}


def _check_run_spacing(*, frame: pl.DataFrame, gaps: list[date]) -> str | None:
    """Fail on a non-00-UTC init or a gap that is not a documented unavailable run."""
    off_hour = frame.filter(pl.col("init_time").dt.hour() != 0)
    if not off_hour.is_empty():
        return "some init_time is not 00 UTC"
    documented = (
        _read_ledger(filename=UNAVAILABLE_FILENAME).keys()
        | _read_ledger(filename=INCOMPLETE_FILENAME).keys()
    )
    undocumented = sorted(set(gaps) - documented)
    if undocumented:
        return f"{len(undocumented)} run days missing without a documented reason"
    return None


def _check_leads(*, frame: pl.DataFrame) -> str | None:
    """Fail unless every (site, init_time) has exactly leads 0 to 240."""
    per_group = frame.group_by("site", "init_time").agg(
        pl.len().alias("n"),
        pl.col("lead_hours").n_unique().alias("n_unique"),
        pl.col("lead_hours").min().alias("lo"),
        pl.col("lead_hours").max().alias("hi"),
    )
    wrong = per_group.filter(
        (pl.col("n") != LEADS_PER_RUN)
        | (pl.col("n_unique") != LEADS_PER_RUN)
        | (pl.col("lo") != 0)
        | (pl.col("hi") != LEADS_PER_RUN - 1)
    )
    if wrong.is_empty():
        return None
    return f"{wrong.height} (site, run) groups do not hold leads 0..{LEADS_PER_RUN - 1}"


def _check_columns(*, frame: pl.DataFrame) -> str | None:
    """Fail unless every expected column is present."""
    expected = {"site", "init_time", "valid_time", "lead_hours", *BASE_VARIABLES}
    missing = expected - set(frame.columns)
    return f"missing columns {sorted(missing)}" if missing else None


def _check_nulls(*, frame: pl.DataFrame) -> str | None:
    """Fail on any null outside radiation at lead 0."""
    offenders = []
    for name in BASE_VARIABLES:
        column = (
            frame if name not in RADIATION_VARIABLES else frame.filter(pl.col("lead_hours") > 0)
        )
        n_null = column[name].null_count()
        if n_null:
            offenders.append(f"{name}: {n_null}")
    return f"nulls found ({', '.join(offenders)})" if offenders else None


def _check_lead_zero_radiation(*, frame: pl.DataFrame) -> str | None:
    """Fail unless radiation at lead 0 is null everywhere, the documented convention."""
    lead_zero = frame.filter(pl.col("lead_hours") == 0)
    filled = [
        name for name in RADIATION_VARIABLES if lead_zero[name].null_count() != lead_zero.height
    ]
    return f"radiation at lead 0 is populated for {filled}" if filled else None


def _check_ranges(*, frame: pl.DataFrame) -> str | None:
    """Fail on any value outside its physical range."""
    bounds: dict[str, tuple[float, float]] = dict.fromkeys(
        RADIATION_VARIABLES, (0.0, MAX_RADIATION_W_M2)
    )
    bounds["temperature_2m"] = TEMPERATURE_RANGE_C
    bounds["wind_speed_10m"] = (0.0, MAX_WIND_SPEED_KM_H)
    bounds["wind_speed_100m"] = (0.0, MAX_WIND_SPEED_KM_H)
    bounds["wind_direction_10m"] = (0.0, MAX_DIRECTION_DEG)
    bounds["wind_direction_100m"] = (0.0, MAX_DIRECTION_DEG)
    offenders = []
    for name, (low, high) in bounds.items():
        n_out = frame.filter((pl.col(name) < low) | (pl.col(name) > high)).height
        if n_out:
            offenders.append(f"{name}: {n_out}")
    return f"values out of range ({', '.join(offenders)})" if offenders else None


def _check_diurnal(*, frame: pl.DataFrame) -> str | None:
    """Fail unless radiation is near zero around midnight UTC and clearly positive around noon."""
    by_hour = (
        frame.filter(pl.col("lead_hours") > 0)
        .group_by(pl.col("valid_time").dt.hour().alias("hour"))
        .agg(pl.col("shortwave_radiation").mean().alias("mean"))
    )
    night_raw = by_hour.filter(pl.col("hour").is_in(NIGHT_HOURS_UTC))["mean"].max()
    noon_raw = by_hour.filter(pl.col("hour").is_in(NOON_HOURS_UTC))["mean"].min()
    if night_raw is None or noon_raw is None:
        return "no night or noon hours in the data"
    night, noon = float(night_raw), float(noon_raw)  # ty: ignore[invalid-argument-type]
    if night > NIGHT_MAX_MEAN_RADIATION_W_M2:
        return f"mean night radiation {night:.2f} W/m^2 exceeds {NIGHT_MAX_MEAN_RADIATION_W_M2}"
    if noon < NOON_MIN_MEAN_RADIATION_W_M2:
        return f"mean noon radiation {noon:.1f} W/m^2 is below {NOON_MIN_MEAN_RADIATION_W_M2}"
    return None


def _check_sites_every_month(*, frame: pl.DataFrame) -> str | None:
    """Fail unless every month holds exactly the nine site labels."""
    by_month = frame.group_by(pl.col("init_time").dt.strftime("%Y-%m").alias("month")).agg(
        pl.col("site").unique().alias("sites")
    )
    wrong = [month for month, sites in by_month.iter_rows() if frozenset(sites) != EXPECTED_SITES]
    return f"months without exactly the nine sites: {sorted(wrong)}" if wrong else None


def _check_duplicate_leads(*, frame: pl.DataFrame) -> str | None:
    """Fail on any (site, init_time, lead_hours) key that appears more than once."""
    n_duplicated = frame.height - frame.select("site", "init_time", "lead_hours").n_unique()
    return f"{n_duplicated} duplicated (site, run, lead) rows" if n_duplicated else None


def _check_valid_time(*, frame: pl.DataFrame) -> str | None:
    """Fail unless every row has `valid_time == init_time + lead_hours`."""
    wrong = frame.filter(
        pl.col("valid_time") != pl.col("init_time") + pl.duration(hours=pl.col("lead_hours"))
    )
    return (
        f"{wrong.height} rows where valid_time != init_time + lead_hours" if wrong.height else None
    )


def _site_series(*, frame: pl.DataFrame) -> dict[str, tuple[tuple[float | None, ...], ...]]:
    """Return each site's series of every variable, in run then lead order."""
    per_site = (
        frame.sort("init_time", "lead_hours")
        .group_by("site")
        .agg(pl.col(name) for name in BASE_VARIABLES)
    )
    return {
        site: tuple(tuple(row[name]) for name in BASE_VARIABLES)
        for site, row in zip(per_site["site"], per_site.iter_rows(named=True), strict=True)
    }


def _check_not_constant(*, frame: pl.DataFrame) -> str | None:
    """Fail if any site's temperature or wind speed at 100 m never changes."""
    constant = [
        site
        for site, series in _site_series(frame=frame).items()
        if len(set(series[BASE_VARIABLES.index("temperature_2m")])) <= 1
        or len(set(series[BASE_VARIABLES.index("wind_speed_100m")])) <= 1
    ]
    return f"constant series for sites {sorted(constant)}" if constant else None


def _warn_identical_series(*, frame: pl.DataFrame) -> str | None:
    """Warn, without failing, about site labels that carry an identical series in every variable.

    Two sites whose nearest 9 km grid cell is the same carry identical series legitimately, so this
    lists the label pairs for a manual look and is never a failure.
    """
    series = _site_series(frame=frame)
    sites = sorted(series)
    pairs = [
        f"{first}/{second}"
        for index, first in enumerate(sites)
        for second in sites[index + 1 :]
        if series[first] == series[second]
    ]
    return f"identical series for {pairs}" if pairs else None


def main() -> int:
    """Run every check, print one line each, then list every gap. Return non-zero on a failure."""
    path: Path = PRODUCT_DIR / COMBINED_FILENAME
    frame = pl.read_parquet(path)
    days = _run_days(frame=frame)
    gaps = _gaps(days=days)
    failures = {
        "run_spacing": _check_run_spacing(frame=frame, gaps=gaps),
        "leads_per_run_per_site": _check_leads(frame=frame),
        "columns_present": _check_columns(frame=frame),
        "no_unexpected_nulls": _check_nulls(frame=frame),
        "lead_zero_radiation_null": _check_lead_zero_radiation(frame=frame),
        "value_ranges": _check_ranges(frame=frame),
        "diurnal_radiation": _check_diurnal(frame=frame),
        "nine_sites_every_month": _check_sites_every_month(frame=frame),
        "no_duplicate_leads": _check_duplicate_leads(frame=frame),
        "valid_time_matches_lead": _check_valid_time(frame=frame),
        "sites_not_constant": _check_not_constant(frame=frame),
    }
    for name in CHECK_NAMES:
        reason = failures[name]
        print(f"PASS {name}" if reason is None else f"FAIL {name}: {reason}")
    warning = _warn_identical_series(frame=frame)
    print(
        "PASS identical_series_sites"
        if warning is None
        else f"WARN identical_series_sites: {warning}"
    )
    print(f"runs: {len(days)}, first {days[0]}, last {days[-1]}")
    unavailable = _read_ledger(filename=UNAVAILABLE_FILENAME)
    incomplete = _read_ledger(filename=INCOMPLETE_FILENAME)
    print(f"gaps ({len(gaps)}) from {FIRST_RUN_DATE} to the newest expected run:")
    for day in gaps:
        if day in unavailable:
            kind = f"refused by the API ({unavailable[day]})"
        elif day in incomplete:
            kind = f"incomplete ({incomplete[day]})"
        else:
            kind = "UNDOCUMENTED"
        print(f"  {day}: {kind}")
    return 1 if any(reason is not None for reason in failures.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
