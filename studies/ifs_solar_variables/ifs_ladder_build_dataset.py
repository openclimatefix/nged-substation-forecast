"""Build the frames the IFS variable ladder is fitted on: one per lead day, and one for blend rows.

One-off throwaway script for the study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1137>. It takes the rows, targets,
capacities, export caps, solar geometry, and CAMS columns of the ERA5 study's built frame, and joins
onto each row the IFS forecast issued `L` days before the valid day, for each lead day `L`
(`studies.ifs_lead_days`). It writes one parquet per lead day, one parquet for the blend rows, and
one markdown file of checks, under `data/studies/per_study/ifs_solar_variables/inputs/`.

**The rows are set by the ERA5 study's rows, the clock, and IFS's presence, and never by an IFS
value's size.** An hour is kept at a lead day when all of these hold:

- the ERA5 study kept the hour (daylight, both targets, no outage run, no commissioning ramp), and
  the hour falls on or after the day after the archive's first run;
- the hour is not in a month that straddles a change of IFS cycle (2024-11 and 2026-05), nor in a
  month after the second change (`studies.ifs_lead_days.with_ifs_eras`);
- the run issued `L` days before the valid day is in the archive (a missing run day drops that valid
  day at that lead day only);
- every IFS variable of the full set is present, except convective inhibition, which Open-Meteo
  leaves missing wherever it is undefined.

**The folds are assigned once, on all of the ERA5 study's rows inside the archive's span, before any
lead-day join.** The
build asserts that every lead day gives the same (farm, hour) the same fold, and that no calendar
month is absent from the training rows of the fold that holds it out.

**The blend rows are the rows of lead days 1 to 3 that AIFS Single also covers**, from its runs on
or after 2025-03-01 (`studies.ifs_lead_days.AIFS_FIRST_OPERATIONAL_INIT`). Their folds are cut again
inside that one era, from the months alone, so each lead day gives a month the same fold. The
partner's columns carry no lead day in their names, and a `lead_day` column says which lead day a
row belongs to.

**Radiation of exactly -1 W m⁻² is clipped to 0**, as the product's README warns.

Anonymised: farms carry only the letters A to F, and no coordinate is read.

Run it with `uv run python studies/ifs_solar_variables/ifs_ladder_build_dataset.py`.
"""

import logging
import sys
from datetime import timedelta
from typing import Final

import polars as pl
from ifs_ladder_arms import (
    BLEND_LEAD_DAYS,
    ERA5_COMPARISON_LEAD_DAY,
    ERA5_DATASET_NAME,
    INPUTS_DIR,
    blend_dataset_path,
    checks_path,
    lead_day_dataset_path,
)
from studies.cross_validation import calendar_month_coverage, raise_on_uncovered_months
from studies.era5_ladder import SHARED_FEATURES as ERA5_SHARED_FEATURES
from studies.era5_ladder import RungType as Era5RungType
from studies.era5_ladder import rung_variables as era5_rung_variables
from studies.guards import refuse_to_overwrite
from studies.ifs_ladder import (
    ERA5_PREFIX,
    MISSING_BY_DESIGN,
    RUNGS,
    rung_variables,
)
from studies.ifs_lead_days import (
    DROPPED_MONTHS,
    HOURS_PER_DAY,
    LEAD_DAYS,
    PARTNER_COLUMNS,
    blend_month_folds,
    find_fold_offsets,
    join_aifs_partner,
    join_forecast_at_lead_day,
    raise_unless_same_folds,
    with_blend_folds,
    with_ifs_eras,
)
from studies.ifs_single_runs import clip_radiation
from studies.sources import (
    ECMWF_IFS_SINGLE_RUNS_SOLAR_PRODUCT_DIR,
    ERA5_LADDER_INPUTS_DIR,
    NFC_AIFS_BLENDS_DIR,
    NFC_AIFS_EXTRA_DAYS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("ifs_ladder_build_dataset")

IFS_FILE_NAME: Final[str] = "ECMWF-IFS-SINGLE-RUNS-SOLAR.parquet"
"""The combined parquet of the solar-variable IFS runs, in the product's folder."""

PARTNER_FILE_NAME: Final[str] = "solar_aifs_inputs.parquet"
"""The AIFS Single site-level columns, in each batch folder of the forecast comparison."""

RADIATION_COLUMNS: Final[tuple[str, ...]] = ("shortwave_radiation", "direct_radiation")
"""The IFS variables that are mean radiation fluxes, clipped at zero."""

ERA5_REGIME_COLUMN: Final[str] = f"{ERA5_PREFIX}tcc"
"""ERA5's total cloud cover, which defines the cloud regimes of the splits."""

ROW_COLUMNS: Final[tuple[str, ...]] = (
    "site",
    "time",
    "month",
    "hour_of_day",
    "day_of_year",
    "power_mw",
    "effective_capacity_mw",
    "cap_mw",
    "constrained",
    "cams_clearness_index",
    "cams_clear_sky_index",
    "cams_ghi_w_m2",
    *ERA5_SHARED_FEATURES,
)
"""The columns taken from the ERA5 study's frame: keys, targets, capacity, geometry, and CAMS."""


def ifs_variables() -> tuple[str, ...]:
    """Return the 20 IFS variables of the full set."""
    return rung_variables(rung=RUNGS[-1])


def era5_columns() -> dict[str, str]:
    """Return each ERA5 variable's name in the ERA5 study's frame and its prefixed name here."""
    last: Era5RungType = "g9"
    return {name: f"{ERA5_PREFIX}{name}" for name in era5_rung_variables(rung=last)}


def read_base_rows() -> pl.DataFrame:
    """Read the ERA5 study's rows with the columns this study keeps, ERA5 variables prefixed.

    Returns:
        One row per (farm, hour), sorted, with `ROW_COLUMNS` and the prefixed ERA5 variables.
    """
    names = era5_columns()
    frame = pl.read_parquet(ERA5_LADDER_INPUTS_DIR / ERA5_DATASET_NAME)
    selected = list(dict.fromkeys([*ROW_COLUMNS, *names]))
    return (
        frame.select(selected)
        .rename(names)
        .with_columns(
            hour_of_day=pl.col("time").dt.hour().cast(pl.Int8),
            day_of_year=pl.col("time").dt.ordinal_day().cast(pl.Int16),
        )
        .sort("site", "time")
    )


def read_forecasts() -> pl.DataFrame:
    """Read the IFS runs with radiation clipped at zero.

    Returns:
        One row per (farm, run, lead), carrying the 20 variables.
    """
    forecasts = pl.read_parquet(ECMWF_IFS_SINGLE_RUNS_SOLAR_PRODUCT_DIR / IFS_FILE_NAME)
    return forecasts.with_columns(
        clip_radiation(radiation=pl.col(name)) for name in RADIATION_COLUMNS
    )


def restrict_to_archive_span(*, rows: pl.DataFrame, forecasts: pl.DataFrame) -> pl.DataFrame:
    """Keep the rows from the first valid day that the archive's first run can serve at lead day 1.

    The folds are cut on these rows, so that the first era holds only the months with IFS runs.
    Cutting them on the ERA5 study's whole span would put every first-era IFS month in one fold.

    Args:
        rows: The ERA5 study's rows.
        forecasts: The IFS runs.

    Returns:
        The rows whose valid time is on or after the day after the earliest run.
    """
    first_valid = forecasts.select(pl.col("init_time").min() + timedelta(days=1)).item()
    return rows.filter(pl.col("time").dt.replace_time_zone(None) >= first_valid)


def keep_complete_hours(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the hours where every IFS variable is present, except those missing by design.

    Args:
        frame: Rows with the IFS variables joined on.

    Returns:
        The rows with no missing value (null or not-a-number) in the required variables.
    """
    required = [name for name in ifs_variables() if name not in MISSING_BY_DESIGN]
    return frame.filter(
        pl.all_horizontal(
            pl.col(name).is_not_null() & ~pl.col(name).is_nan().fill_null(value=False)
            for name in required
        )
    )


def lead_day_frame(
    *, base: pl.DataFrame, forecasts: pl.DataFrame, lead_day: int
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Join the forecast at one lead day onto the base rows and keep the complete hours.

    Args:
        base: The base rows with eras and folds.
        forecasts: The IFS runs.
        lead_day: The lead day.

    Returns:
        The kept rows, and the counts of rows at each stage.
    """
    joined = join_forecast_at_lead_day(rows=base, forecasts=forecasts, lead_day=lead_day)
    complete = keep_complete_hours(frame=joined)
    kept_columns = [
        *ROW_COLUMNS,
        "era_code",
        "era",
        "fold",
        "lead_day",
        "lead_hours",
        "init_time",
        *ifs_variables(),
        ERA5_REGIME_COLUMN,
        *(
            era5_columns().values()
            if lead_day == ERA5_COMPARISON_LEAD_DAY
            else [ERA5_REGIME_COLUMN]
        ),
    ]
    frame = complete.select(list(dict.fromkeys(kept_columns))).sort("site", "time")
    return frame, {
        "base": base.height,
        "with_run": joined.height,
        "complete": complete.height,
    }


def read_partner(*, lead_day: int) -> pl.DataFrame:
    """Read the AIFS Single columns that hold a lead day."""
    folder = NFC_AIFS_BLENDS_DIR if lead_day <= 2 else NFC_AIFS_EXTRA_DAYS_DIR
    return pl.read_parquet(folder / PARTNER_FILE_NAME)


def blend_frame(*, frames: dict[int, pl.DataFrame]) -> pl.DataFrame:
    """Build the blend rows of lead days 1 to 3, with folds cut again inside the one AIFS era.

    Args:
        frames: Each lead day's kept rows.

    Returns:
        The stacked blend rows, with `lead_day`, the two `PARTNER_COLUMNS`, and a new `fold`.

    Raises:
        ValueError: If a calendar month is absent from the training rows of the fold that holds
            it out.
    """
    joined = {
        lead_day: join_aifs_partner(
            rows=frames[lead_day].drop(
                name
                for name in frames[lead_day].columns
                if name.startswith(ERA5_PREFIX) and name != ERA5_REGIME_COLUMN
            ),
            partner=read_partner(lead_day=lead_day),
            lead_day=lead_day,
        )
        for lead_day in BLEND_LEAD_DAYS
    }
    months = [month for frame in joined.values() for month in frame["month"].unique().to_list()]
    month_folds = blend_month_folds(months=months)
    folded = pl.concat(
        with_blend_folds(rows=frame.drop("fold"), month_folds=month_folds)
        for frame in joined.values()
    ).sort("lead_day", "site", "time")
    for lead_day in BLEND_LEAD_DAYS:
        raise_on_uncovered_months(
            coverage=calendar_month_coverage(frame=folded.filter(pl.col("lead_day") == lead_day))
        )
    return folded


def missing_shares(*, frame: pl.DataFrame) -> str:
    """Return a markdown table of the share of missing values in each IFS variable."""
    lines = ["| variable | share missing |", "|---|---|"]
    for name in ifs_variables():
        share = frame.select(
            (pl.col(name).is_null() | pl.col(name).is_nan().fill_null(value=False)).mean()
        ).item()
        lines.append(f"| `{name}` | {share:.4f} |")
    return "\n".join(lines)


def build_checks(
    *,
    frames: dict[int, pl.DataFrame],
    counts: dict[int, dict[str, int]],
    blend: pl.DataFrame,
    base: pl.DataFrame,
    straddling_rows: int,
) -> str:
    """Write the build's checks as markdown: rows, missing values, eras, and folds.

    Args:
        frames: Each lead day's kept rows.
        counts: Each lead day's row counts at each stage.
        blend: The blend rows.
        base: The base rows with eras and folds.
        straddling_rows: How many of the ERA5 study's rows fall in a dropped month.

    Returns:
        The markdown text.
    """
    lines = [
        "# Build checks",
        "",
        (
            "- Rows inside the archive's span dropped for the straddling months "
            f"{sorted(DROPPED_MONTHS)} and for the months after the second change: "
            f"{straddling_rows:,}."
        ),
        f"- Base rows kept: {base.height:,}, from {base['time'].min()} to {base['time'].max()}.",
        "- Every lead day gives each (farm, hour) the same fold: asserted.",
        (
            "- No calendar month is absent from the training rows of the fold that holds it out: "
            "asserted for every lead day and for the blend rows."
        ),
        "",
        "## Rows by lead day",
        "",
        (
            "| lead day | base rows | with a run | complete | dropped for a missing run | "
            "dropped as incomplete |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for lead_day, stage in counts.items():
        lines.append(
            f"| {lead_day} | {stage['base']:,} | {stage['with_run']:,} | {stage['complete']:,} | "
            f"{stage['base'] - stage['with_run']:,} | {stage['with_run'] - stage['complete']:,} |"
        )
    lines += ["", "## Rows per farm and lead day", ""]
    farms = sorted(base["site"].unique().to_list())
    lines += [
        "| lead day | " + " | ".join(farms) + " |",
        "|---|" + "---|" * len(farms),
    ]
    for lead_day, frame in frames.items():
        per_farm = dict(frame.group_by("site").len().iter_rows())
        lines.append(
            f"| {lead_day} | " + " | ".join(f"{per_farm.get(f, 0):,}" for f in farms) + " |"
        )
    lines += ["", "## Blend rows per farm and lead day", ""]
    lines += ["| lead day | " + " | ".join(farms) + " |", "|---|" + "---|" * len(farms)]
    for lead_day in BLEND_LEAD_DAYS:
        per_farm = dict(
            blend.filter(pl.col("lead_day") == lead_day).group_by("site").len().iter_rows()
        )
        lines.append(
            f"| {lead_day} | " + " | ".join(f"{per_farm.get(f, 0):,}" for f in farms) + " |"
        )
    lines += [
        "",
        (
            f"Blend rows start with the runs of {blend['init_time'].min()} and cover "
            f"{blend['month'].min()} to {blend['month'].max()}."
        ),
        "",
        "## Missing values at lead day 1 (convective inhibition is missing by design)",
        "",
        missing_shares(frame=frames[LEAD_DAYS[0]]),
        "",
        "## Eras and folds at lead day 1",
        "",
        "| era | fold | rows | first month | last month |",
        "|---|---|---|---|---|",
    ]
    table = (
        frames[LEAD_DAYS[0]]
        .group_by("era", "fold")
        .agg(rows=pl.len(), first=pl.col("month").min(), last=pl.col("month").max())
        .sort("era", "fold")
    )
    lines += [
        f"| {row['era']} | {row['fold']} | {row['rows']:,} | {row['first']} | {row['last']} |"
        for row in table.iter_rows(named=True)
    ]
    lines += [
        "",
        "## Folds of the blend rows",
        "",
        "| fold | rows (all lead days) | first month | last month |",
        "|---|---|---|---|",
    ]
    blend_table = (
        blend.group_by("fold")
        .agg(rows=pl.len(), first=pl.col("month").min(), last=pl.col("month").max())
        .sort("fold")
    )
    lines += [
        f"| {row['fold']} | {row['rows']:,} | {row['first']} | {row['last']} |"
        for row in blend_table.iter_rows(named=True)
    ]
    lines += [
        "",
        (
            f"Lead of the rows at lead day 1: {HOURS_PER_DAY} to "
            f"{frames[LEAD_DAYS[0]]['lead_hours'].max()} hours."
        ),
        f"Partner columns: {', '.join(PARTNER_COLUMNS)}.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build every frame and write it with its checks."""
    outputs = [
        *(lead_day_dataset_path(lead_day=day) for day in LEAD_DAYS),
        blend_dataset_path(),
        checks_path(),
    ]
    refuse_to_overwrite(paths=outputs)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    forecasts = read_forecasts()
    full = restrict_to_archive_span(rows=read_base_rows(), forecasts=forecasts)
    offsets = find_fold_offsets(rows=full)
    base = with_ifs_eras(rows=full, fold_offsets=offsets)
    _LOG.info("fold offsets by era: %s", dict(offsets))

    frames: dict[int, pl.DataFrame] = {}
    counts: dict[int, dict[str, int]] = {}
    for lead_day in LEAD_DAYS:
        frames[lead_day], counts[lead_day] = lead_day_frame(
            base=base, forecasts=forecasts, lead_day=lead_day
        )
        raise_on_uncovered_months(coverage=calendar_month_coverage(frame=frames[lead_day]))
        _LOG.info("lead day %d: %d rows", lead_day, frames[lead_day].height)
    raise_unless_same_folds(frames=list(frames.values()))
    blend = blend_frame(frames=frames)

    for lead_day, frame in frames.items():
        frame.write_parquet(lead_day_dataset_path(lead_day=lead_day))
    blend.write_parquet(blend_dataset_path())
    checks_path().write_text(
        build_checks(
            frames=frames,
            counts=counts,
            blend=blend,
            base=base,
            straddling_rows=full.height - base.height,
        )
    )
    _LOG.info("wrote %d lead-day frames, the blend frame, and the checks", len(frames))
    return 0


if __name__ == "__main__":
    sys.exit(main())
