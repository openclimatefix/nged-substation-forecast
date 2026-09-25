"""Fit the AIFS arms and their references on the GPU, and write their losses and report.

One-off throwaway script for the AIFS arms of
<https://github.com/openclimatefix/nged-substation-forecast/issues/923>. It reads the published
shared rows (`nwp_forecast_comparison.rows` on the published inputs), joins the columns
`build_forecast_inputs.py --aifs` wrote onto them, keeps the hours whose 00 UTC run lies inside one
AIFS version era, recuts the folds inside those eras, and fits every arm out of fold on the GPU.
It writes `<domain>_<row_set>_losses.parquet`, `<domain>_<row_set>_predictions.parquet` and
`report.md` to the `--output-dir` the AIFS inputs are in, and never writes to the published folder.
`report.md` refuses to be overwritten. A set whose losses file exists is not refitted, so a rerun
after a crash resumes at the first set without losses.

Two nested row sets are fitted, each on its own rows: `single` (AIFS Single and every reference) and
`ens` (those and AIFS ENS, whose store starts later). One contrast is named deciding before any fit,
`aifs_single_day1 - ens_control6_day1` on `single`, and the AIFS ENS contrasts are descriptive only:
about 85% of scored `ens` cells have no training row of their calendar month. Every other number is
exploratory and post hoc. Every arm is fitted at the primary setting; the deciding pair, and every
other pair with an interval bound within 20% of its width from zero, is also fitted at the
sensitivity setting in the same invocation. The deciding pair is also refitted with `day_of_year`
removed from both arms.

`--check` fits one arm at one site twice on the GPU, stops unless the two fingerprints agree, and
prints an estimate of the full run's time. Run it before the full fit.

`--blends` fits AIFS at days 1, 2, 7 and 14 and blends of ENS's mean with an AIFS forecast, each
day on a frame of its own (an hour whose day-N run lies outside its AIFS version era is dropped
row by row, its month kept), from the inputs `build_forecast_inputs.py --aifs --aifs-days 1 2 7 14`
wrote to `--output-dir`. It reads the extra-lead folders beside the published one and never writes
to them. Four contrasts per technology are deciding, named before any fit: H7 and H14 (AIFS Single
against the ENS control member) and B7 and B14 (the blend against ENS's mean alone and against its
control, which shuffles AIFS within site, year-month and hour of day). The day-14 reading rule, the
smoothing reading and the weather-spread table go in the report. Every stage (a row set at one
day) writes its own losses, predictions and stamp, and a stage whose losses exist is not refitted.

`--p4-controls` refits the published P4 blends, their first control and a second-seed control, and
ENS's day-1 mean on the GPU, on the published run's own rows and folds, into a new `--output-dir`.
If the two controls disagree on the guard's verdict, the blend claim is unresolved.

Every output carries only the anonymised `site` label.

Run it with `uv run python studies/nwp_forecast_comparison/fit_aifs.py --published-dir PUBLISHED
--output-dir DIR`, after `build_forecast_inputs.py --aifs --output-dir DIR` has written the AIFS
inputs there. Add `--blends` or `--p4-controls`, and `--check` first, for the other two fits.
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import re
import subprocess
import sys
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, Literal, NamedTuple

import numpy as np
import polars as pl
from fit_extra_leads import error_text, interval_text
from nwp_forecast_comparison import (
    BLEND_ARMS,
    METRIC,
    SETTINGS,
    TARGET,
    DomainType,
    add_blend_guard_columns,
    arm_columns,
    coverage_table,
    difference,
    fingerprint,
    leaderboard,
    predictions_from_losses,
    rows,
)
from studies.blending import climatology_permutation
from studies.bootstrap import (
    NO_DETECTABLE_DIFFERENCE,
    BlendVerdict,
    BootstrapInterval,
    blend_verdict,
    bootstrap_difference_at_level,
    combine_setting_verdicts,
    paired_differences,
)
from studies.cross_validation import DeviceType, cut_eras, out_of_fold_losses, search_fold_offsets
from studies.guards import check_no_missing, refuse_to_overwrite

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

DOMAINS: Final[tuple[DomainType, DomainType]] = ("solar", "wind")

DEVICE: Final[DeviceType] = "cuda"
"""The XGBoost device every fit here uses."""

EXISTING_AIFS_DIR_NAME: Final[str] = "nwp_forecast_comparison_aifs"
"""Under `data/studies/`, the folder of the day-1 and day-2 fit, which the blends fit never writes
to."""

PRIMARY: Final[str] = "primary"
SENSITIVITY: Final[str] = "sensitivity"

AIFS_DROPPED_MONTHS: Final[tuple[str, ...]] = ("2025-08", "2026-05")
"""Months holding a version switch: AIFS Single's reverted v1.1 attempt and v1.0 to v1.1 switch in
2025-08, and AIFS v2 with IFS Cycle 50r1 in 2026-05. `nwp_forecast_comparison.DROPPED_MONTHS` drops
2026-01 (the UKV upgrade) as well."""

RUNS_OPEN_END: Final[date] = date(2999, 12, 31)
"""The last run date of the final era, which has no end yet."""

PERMUTED: Final[str] = "_permuted"
"""Appended to a prefix to name its shuffled copy's prefix."""

SHUFFLE_SEEDS: Final[dict[str, int]] = {"": 0, "_b": 1000}
"""Each shuffled copy's variant suffix and its seed. The unsuffixed copy is the climatology
reference (and the blend control's shuffled columns); `_b` is its null repeat."""


def shuffled_prefix(*, source: str, variant: str = "") -> str:
    """Return the prefix of `source`'s shuffled copy: `<source>_permuted` and the variant."""
    return f"{source}{PERMUTED}{variant}"


PERMUTED_PREFIX: Final[str] = shuffled_prefix(source="aifs_single_day1")
PERMUTED_B_PREFIX: Final[str] = shuffled_prefix(source="aifs_single_day1", variant="_b")
PERMUTED_SEEDS: Final[dict[str, int]] = {PERMUTED_PREFIX: 0, PERMUTED_B_PREFIX: 1000}
"""The climatology reference and its null repeat, each a shuffle of AIFS Single's day-1 columns
among hours sharing a site, a year-month and an hour of day, under its own seed."""

OLD_SHUFFLES: Final[dict[str, tuple[str, ...]]] = {"aifs_single_day1": ("", "_b")}
"""The day-1 and day-2 fit's shuffles: AIFS Single's day-1 columns, under both seeds."""

NO_DOY_SUFFIX: Final[str] = "_no_doy"
"""Appended to an arm's name for its refit with `day_of_year` removed from its columns."""

NEAR_LINE_SHARE: Final[float] = 0.2
"""An interval is near the 5% line when a bound lies within this share of its width from zero."""

MIN_ERA_MONTHS_FOR_INTERVAL: Final[int] = 6
"""An era's contrast prints an interval only when the era holds at least this many months."""

BONFERRONI_LEVEL: Final[float] = 97.5
"""The coverage of the deciding contrast's wider interval: 95% corrected across two technologies."""

N_SITES: Final[dict[DomainType, int]] = {"solar": 6, "wind": 3}
"""How many generators each technology has, for `--check`'s time estimate."""

Job = tuple[str, str]
"""One fit: the arm's name and the setting's name."""


@dataclass(frozen=True)
class RowSet:
    """One nested row set: which months it keeps, its eras, its folds, and its arms."""

    first_month: str
    """The first valid month kept (`%Y-%m`)."""
    era_start_months: tuple[str, ...]
    """The first month of every era after the first."""
    fold_offsets: dict[int, int]
    """Each era's fold rotation. `search_fold_offsets` must return this design on the rows."""
    era_runs: dict[int, tuple[date, date]]
    """Each era to the first and last 00 UTC run date that every AIFS arm on the set may read."""
    arms: tuple[str, ...]
    """Every arm fitted on the set, by weather-column prefix."""
    deciding: tuple[str, str] | None
    """The (treatment, reference) named deciding before any fit, or `None`."""


REFERENCE_ARMS: Final[tuple[str, ...]] = (
    "ens_mean6_day1",
    "ens_mean6_day2",
    "ens_control6_day1",
    "ens_control6_day2",
    "ens_mean_day1",
    "ifs025_day1",
    PERMUTED_PREFIX,
    PERMUTED_B_PREFIX,
)
SINGLE_ARMS: Final[tuple[str, ...]] = (
    "aifs_single_day1",
    "aifs_single_day2",
    "aifs_single_nearest_day1",
    *REFERENCE_ARMS,
)

ROW_SETS: Final[dict[str, RowSet]] = {
    "single": RowSet(
        first_month="2025-03",
        era_start_months=("2025-09", "2026-06"),
        fold_offsets={0: 0, 1: 0, 2: 0},
        era_runs={
            0: (date(2025, 2, 26), date(2025, 7, 31)),
            1: (date(2025, 8, 28), date(2026, 5, 11)),
            2: (date(2026, 5, 13), RUNS_OPEN_END),
        },
        arms=SINGLE_ARMS,
        deciding=("aifs_single_day1", "ens_control6_day1"),
    ),
    "ens": RowSet(
        first_month="2025-09",
        era_start_months=("2026-06",),
        fold_offsets={0: 0, 1: 0},
        era_runs={
            0: (date(2025, 8, 28), date(2026, 5, 11)),
            1: (date(2026, 5, 13), RUNS_OPEN_END),
        },
        arms=(*SINGLE_ARMS, "aifs_ens_mean_day1", "aifs_ens_mean_day2"),
        deciding=None,
    ),
}
"""The two row sets. Era 0 of `single` is AIFS Single v1.0 (its 2025-07-31 06 UTC to 2025-08-01
18 UTC window is excluded), era 1 is v1.1 and AIFS ENS v1, and the last era is v2, which changed
with IFS Cycle 50r1. The dates are from ECMWF's release pages as recorded in
`docs/roadmap/data-sources.md`; the data does not verify them. Only 00 UTC runs are read, so a
06 UTC boundary is a run date that the 00 UTC run of the day after starts."""


class Contrast(NamedTuple):
    """One paired contrast, treatment minus reference, and how the page may use it."""

    treatment: str
    reference: str
    label: str


def contrasts(*, row_set: str) -> list[Contrast]:
    """Return every listed contrast of one row set, in report order.

    Args:
        row_set: `single` or `ens`.

    Returns:
        The contrasts. On `single` one is `deciding`; the AIFS ENS contrasts on `ens` are
        `descriptive`, and every other contrast is `exploratory, post hoc`.
    """
    exploratory = "exploratory, post hoc"
    steps = Contrast("ens_mean6_day1", "ens_mean_day1", "exploratory, post hoc (positive control)")
    if row_set == "ens":
        descriptive = "descriptive, no deciding label"
        return [
            Contrast("aifs_ens_mean_day1", "ens_mean6_day1", descriptive),
            Contrast("aifs_ens_mean_day2", "ens_mean6_day2", descriptive),
            Contrast("aifs_ens_mean_day1", "aifs_single_day1", descriptive),
            Contrast("aifs_ens_mean_day2", "aifs_single_day2", descriptive),
            steps,
        ]
    return [
        Contrast("aifs_single_day1", "ens_control6_day1", "deciding"),
        Contrast("aifs_single_day2", "ens_control6_day2", exploratory),
        Contrast("aifs_single_day1", "ens_mean6_day1", exploratory),
        Contrast("aifs_single_day2", "ens_mean6_day2", exploratory),
        Contrast("aifs_single_day1", "ifs025_day1", "exploratory, post hoc (one-sided reference)"),
        Contrast("aifs_single_day1", "aifs_single_nearest_day1", "exploratory (spatial read)"),
        steps,
        Contrast("aifs_single_day1", PERMUTED_PREFIX, "exploratory (climatology)"),
        Contrast(PERMUTED_PREFIX, PERMUTED_B_PREFIX, "exploratory (null)"),
    ]


BLEND_PREFIX: Final[str] = "blend_"
"""Every blend arm's name starts with this; `arm_prefixes` reads the rest."""

BLEND_AIFS_PREFIXES: Final[dict[str, str]] = {
    "aifs_single": "aifs_single_day{day}",
    "aifs_ens": "aifs_ens_mean_day{day}",
}
"""Each blend's AIFS product, to the weather-column prefix of that product at one day."""

BlendRoleType = Literal["", "_control", "_mirror"]
"""A blend arm's role: the blend itself, its control (AIFS shuffled), or its mirror control (ENS
shuffled)."""

_P4_ARM: Final[re.Pattern[str]] = re.compile(r"blend_(p4a|p4b)(_control|_control_b)?")
"""The published P4 blends and their two controls: the first shuffle and the second seed's."""

_BLEND_ARM: Final[re.Pattern[str]] = re.compile(
    rf"{BLEND_PREFIX}({'|'.join(BLEND_AIFS_PREFIXES)})_day(\d+)(_control|_mirror)?"
)
_DAY_OF_PREFIX: Final[re.Pattern[str]] = re.compile(r".*_day(\d+)")


def blend_arm_name(*, product: str, day: int, role: BlendRoleType = "") -> str:
    """Return a blend arm's name, such as `blend_aifs_single_day7_control`."""
    return f"{BLEND_PREFIX}{product}_day{day}{role}"


def arm_prefixes(*, arm: str) -> tuple[str, ...]:
    """Return the weather-column prefixes one arm is shown.

    Args:
        arm: An arm's name; `NO_DOY_SUFFIX` marks the refit without `day_of_year`. A blend arm is
            `blend_<aifs_single or aifs_ens>_day<N>` with an optional `_control` or `_mirror`.
            Any other arm is its own prefix.

    Returns:
        A single product's prefix, or, for a blend, ENS's mean first and then the AIFS product's,
        with the shuffled side carrying `PERMUTED` in its prefix for a control or mirror control.
    """
    name = arm.removesuffix(NO_DOY_SUFFIX)
    published = _P4_ARM.fullmatch(name)
    if published is not None:
        ens, *others = BLEND_ARMS[f"blend_{published[1]}"]
        variant = {None: None, "_control": "", "_control_b": "_b"}[published[2]]
        if variant is not None:
            others = [shuffled_prefix(source=other, variant=variant) for other in others]
        return (ens, *others)
    match = _BLEND_ARM.fullmatch(name)
    if match is None:
        return (name,)
    product, day, role = match.groups()
    aifs = BLEND_AIFS_PREFIXES[product].format(day=day)
    ens = f"ens_mean_day{day}"
    if role == "_control":
        aifs = shuffled_prefix(source=aifs)
    elif role == "_mirror":
        ens = shuffled_prefix(source=ens)
    return (ens, aifs)


def arm_features(*, arm: str, domain: DomainType) -> tuple[str, ...]:
    """Return one arm's feature columns.

    Args:
        arm: An arm's name; `NO_DOY_SUFFIX` marks the refit without `day_of_year`.
        domain: `solar` or `wind`.

    Returns:
        The arm's fixed-length column tuple, with `day_of_year` removed for the refit without it.
    """
    columns = arm_columns(domain=domain, prefixes=arm_prefixes(arm=arm))
    return (
        tuple(c for c in columns if c != "day_of_year") if arm.endswith(NO_DOY_SUFFIX) else columns
    )


def expected_column_count(*, arm: str, domain: DomainType) -> int:
    """Return how many columns an arm's kind promises, counted without `arm_columns`.

    Args:
        arm: An arm's name.
        domain: `solar` or `wind`.

    Returns:
        The calendar columns (3 for wind, 5 for solar, which adds the two sun-position columns)
        plus each product's weather columns (4 for wind, 2 for solar). A single product is 7
        columns, a blend or its control is 9 for solar and 11 for wind, a published P4 blend (ENS
        and two other products) or its control is 11 for solar and 15 for wind, and the refit
        without `day_of_year` has one column fewer.
    """
    products = 1
    if _P4_ARM.fullmatch(arm.removesuffix(NO_DOY_SUFFIX)):
        products = 3
    elif arm.startswith(BLEND_PREFIX):
        products = 2
    calendar, weather = (3, 4) if domain == "wind" else (5, 2)
    return calendar + products * weather - int(arm.endswith(NO_DOY_SUFFIX))


def source_columns(
    *, arms: Sequence[str], domain: DomainType, nullable: Sequence[str] = ()
) -> list[str]:
    """Return the columns the input rows must hold for `arms`, leaving out shuffled copies.

    Args:
        arms: The arms to be fitted.
        domain: `solar` or `wind`.
        nullable: Prefixes whose columns may hold nulls on purpose, which the missing-value check
            skips.

    Returns:
        The calendar and weather columns of every arm's non-shuffled prefixes, without repeats.
    """
    prefixes = dict.fromkeys(
        prefix
        for arm in arms
        for prefix in arm_prefixes(arm=arm)
        if PERMUTED not in prefix and prefix not in nullable
    )
    return list(
        dict.fromkeys(
            column
            for prefix in prefixes
            for column in arm_columns(domain=domain, prefixes=(prefix,))
        )
    )


def add_shuffled_columns(
    *, frame: pl.DataFrame, domain: DomainType, shuffles: Mapping[str, Sequence[str]]
) -> pl.DataFrame:
    """Add shuffled copies of each source product's weather columns.

    Every copy moves a source's values only among the hours that share a site, a year-month and an
    hour of day, under the seed of its variant (`SHUFFLE_SEEDS`), so a shuffle is deterministic and
    never crosses a site, a year-month or an hour of day. A product's direction sine and cosine
    (wind) are shuffled together.

    Args:
        frame: Rows carrying `site`, `month`, `hour_of_day` and each source's columns.
        domain: `solar` or `wind`.
        shuffles: Each prefix to shuffle, such as `aifs_single_day7`, to the variant suffixes to
            build for it (keys of `SHUFFLE_SEEDS`).

    Returns:
        `frame` with `<source>_permuted<variant>_<field>` for every source, variant and field.
    """
    output = frame
    for source, variants in shuffles.items():
        fields = tuple(
            column
            for column in arm_columns(domain=domain, prefixes=(source,))
            if column.startswith(f"{source}_")
        )
        groups = (
            [(fields[0],), (fields[1], fields[2]), (fields[3],)]
            if domain == "wind"
            else [(fields[0],), (fields[1],)]
        )
        for variant in variants:
            prefix = shuffled_prefix(source=source, variant=variant)
            output = climatology_permutation(
                frame=output,
                column_groups=groups,
                by=("site", "month", "hour_of_day"),
                seed=SHUFFLE_SEEDS[variant],
                suffix=f"_{prefix}",
            ).rename(
                {
                    f"{column}_{prefix}": f"{prefix}_{column.removeprefix(f'{source}_')}"
                    for group in groups
                    for column in group
                }
            )
    return output


def run_date(*, domain: DomainType, day: int) -> pl.Expr:
    """Return the date of the 00 UTC run that day `day` reads, for each row's `time`.

    A solar hour is labelled by its end, so its day is the day of `time` minus 30 minutes.
    """
    time_of_day = pl.col("time") - pl.duration(minutes=30) if domain == "solar" else pl.col("time")
    return (time_of_day.dt.truncate("1d") - pl.duration(days=day)).dt.date()


def era_run_bounds(*, spec: RowSet, era_code: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Return the first and last run date every AIFS arm may read, for each row's era."""
    first = era_code.replace_strict(
        {era: runs[0] for era, runs in spec.era_runs.items()}, return_dtype=pl.Date
    )
    last = era_code.replace_strict(
        {era: runs[1] for era, runs in spec.era_runs.items()}, return_dtype=pl.Date
    )
    return first, last


def check_runs(
    *, frame: pl.DataFrame, domain: DomainType, row_set: str, arms: Sequence[str]
) -> None:
    """Raise unless every stamped arm's run on every row is the right 00 UTC run inside its era.

    The prefixes checked come from `arms`: every non-shuffled prefix that starts `aifs_` must carry
    a `<prefix>_init_time` column, and any other prefix (ENS's) is checked when its column is
    present. Each prefix's day is read from its name.

    Args:
        frame: The set's rows, carrying `time`, `era_code` and each stamped arm's `_init_time`.
        domain: `solar` or `wind`.
        row_set: `single` or `ens`.
        arms: The arms fitted on the frame.

    Raises:
        ValueError: If an AIFS prefix has no run column or no `_day<N>` ending, or naming each arm
            whose run is not a 00 UTC run, is not the run `day` days before the row's day, or
            falls outside its era's run dates.
    """
    first, last = era_run_bounds(spec=ROW_SETS[row_set], era_code=pl.col("era_code"))
    problems: dict[str, dict[str, int]] = {}
    prefixes = dict.fromkeys(
        prefix for arm in arms for prefix in arm_prefixes(arm=arm) if PERMUTED not in prefix
    )
    for prefix in prefixes:
        stamped = f"{prefix}_init_time" in frame.columns
        if not stamped and prefix.startswith("aifs_"):
            msg = (
                f"{domain}/{row_set}: {prefix}_init_time is missing, so its runs cannot be checked"
            )
            raise ValueError(msg)
        if not stamped:
            continue
        match = _DAY_OF_PREFIX.fullmatch(prefix)
        if match is None:
            msg = f"{domain}/{row_set}: {prefix} does not end in _day<N>, so its day is unknown"
            raise ValueError(msg)
        init = pl.col(f"{prefix}_init_time")
        counts = frame.select(
            not_midnight=(init.dt.hour() != 0).sum(),
            wrong_day=(init.dt.date() != run_date(domain=domain, day=int(match[1]))).sum(),
            outside_era=((init.dt.date() < first) | (init.dt.date() > last)).sum(),
        ).row(0, named=True)
        if any(counts.values()):
            problems[prefix] = counts
    if problems:
        msg = f"{domain}/{row_set}: rows whose run is wrong: {problems}"
        raise ValueError(msg)


def drop_runs_outside_era(
    *, frame: pl.DataFrame, domain: DomainType, row_set: str, day: int
) -> pl.DataFrame:
    """Drop, row by row, the hours whose day-`day` run lies outside their AIFS version era.

    A month stays in the row set; only the hours whose run belongs to the previous era go. No hour
    is ever filled from another run.

    Args:
        frame: Rows carrying `month` and `time`, before `cut_eras` labels the eras.
        domain: `solar` or `wind`.
        row_set: `single` or `ens`.
        day: The lead day whose run each hour reads.

    Returns:
        The rows whose run date lies inside the era window of the hour's month.
    """
    spec = ROW_SETS[row_set]
    era_code = pl.lit(0, dtype=pl.Int8) + sum(
        (pl.col("month") >= month).cast(pl.Int8) for month in spec.era_start_months
    )
    first, last = era_run_bounds(spec=spec, era_code=era_code)
    run = run_date(domain=domain, day=day)
    return frame.filter(run >= first, run <= last)


def aifs_rows(
    *,
    published_dir: Path,
    aifs: pl.DataFrame,
    domain: DomainType,
    row_set: str,
    arms: Sequence[str],
    day: int | None,
    shuffles: Mapping[str, Sequence[str]],
    nullable: Sequence[str] = (),
) -> pl.DataFrame:
    """Return one set's rows: the published shared rows inside the set's eras, with AIFS joined on.

    Args:
        published_dir: The folder holding the published inputs.
        aifs: The AIFS input columns, keyed by `site` and `time`.
        domain: `solar` or `wind`.
        row_set: `single` or `ens`.
        arms: The arms to be fitted on the rows, whose columns must be present.
        day: The lead day, or `None` for the day-1 and day-2 fit, which keeps every hour of its
            months. With a day, hours whose run of that day lies outside their era are dropped.
        shuffles: Each prefix to shuffle to the variant suffixes to build for it.
        nullable: Prefixes whose columns may hold nulls on purpose.

    Returns:
        The rows with `month`, `era_code`, `era` and `fold` recut inside the AIFS eras, and the
        shuffled copies.

    Raises:
        ValueError: If a shared row has no row in the AIFS inputs, an arm's column is missing or
            has a null or a not-a-number on a kept row, the fold design is no longer among those
            covering every calendar month, or a row's run is wrong.
    """
    spec = ROW_SETS[row_set]
    shared = rows(input_dir=published_dir, domain=domain).drop("era_code", "era", "fold")
    keys = ["site", "time"]
    if shared.select(keys).join(aifs.select(keys), on=keys, how="anti").height:
        msg = f"{domain}: shared rows are missing from the AIFS inputs"
        raise ValueError(msg)
    kept = (
        shared.join(aifs, on=keys, how="left")
        .filter(pl.col("month") >= spec.first_month, ~pl.col("month").is_in(AIFS_DROPPED_MONTHS))
        .sort("site", "time")
    )
    if day is not None:
        kept = drop_runs_outside_era(frame=kept, domain=domain, row_set=row_set, day=day)
    found = search_fold_offsets(frame=kept, first_months=spec.era_start_months)
    if spec.fold_offsets not in [dict(offsets) for offsets in found]:
        msg = f"{domain}/{row_set}: {spec.fold_offsets} no longer covers every calendar month"
        raise ValueError(msg)
    cut = cut_eras(frame=kept, first_months=spec.era_start_months, fold_offsets=spec.fold_offsets)
    check_no_missing(frame=cut, columns=source_columns(arms=arms, domain=domain, nullable=nullable))
    coverage_table(frame=cut)
    check_runs(frame=cut, domain=domain, row_set=row_set, arms=arms)
    return add_shuffled_columns(frame=cut, domain=domain, shuffles=shuffles)


def fit_jobs(
    *, frame: pl.DataFrame, domain: DomainType, jobs: list[Job], workers: int
) -> pl.DataFrame:
    """Fit every job at every site, out of fold, on the GPU, and stack the losses.

    Args:
        frame: One set's rows from `aifs_rows`.
        domain: `solar` or `wind`.
        jobs: The (arm, setting) fits to run.
        workers: How many (job, site) fits run at once.

    Returns:
        Every fit's per-row losses, labelled with `arm` and `setting`.

    Raises:
        ValueError: If an arm's columns are absent from `frame`, or an arm does not hold the
            column count its name promises.
    """
    sites = sorted(frame["site"].unique().to_list())
    outputs: list[pl.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for arm, setting in jobs:
            features = arm_features(arm=arm, domain=domain)
            absent = [name for name in features if name not in frame.columns]
            if absent:
                msg = f"{domain}: {arm} has no columns {absent} in the rows"
                raise ValueError(msg)
            expected = expected_column_count(arm=arm, domain=domain)
            if len(features) != expected:
                msg = f"{domain}: {arm} has {len(features)} columns, its kind promises {expected}"
                raise ValueError(msg)
            for site in sites:
                future = pool.submit(
                    out_of_fold_losses,
                    site_rows=frame.filter(pl.col("site") == site),
                    features=list(features),
                    target=TARGET,
                    hyper_parameters=SETTINGS[setting],
                    with_quantiles=False,
                    device=DEVICE,
                )
                futures[future] = (arm, setting, site)
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            arm, setting, site = futures[future]
            outputs.append(future.result().with_columns(arm=pl.lit(arm), setting=pl.lit(setting)))
            _LOG.info(
                "%s: %d/%d done: %s %s site %s", domain, done, len(futures), arm, setting, site
            )
    return pl.concat(outputs)


def near_line(*, interval: BootstrapInterval) -> bool:
    """Return whether a bound of a 95% interval is within `NEAR_LINE_SHARE` of its width from 0."""
    width = interval["upper_95"] - interval["lower_95"]
    return min(abs(interval["lower_95"]), abs(interval["upper_95"])) <= NEAR_LINE_SHARE * width


def sensitivity_arms(*, losses: pl.DataFrame, row_set: str) -> list[str]:
    """Return every arm to refit at the sensitivity setting: the deciding pair and near-line pairs.

    Args:
        losses: The primary setting's per-row losses on the set.
        row_set: `single` or `ens`.

    Returns:
        The arm names, deciding pair first, without repeats.
    """
    deciding = ROW_SETS[row_set].deciding
    arms = list(deciding or ())
    for contrast in contrasts(row_set=row_set):
        interval = difference(
            losses=losses, treatment=contrast.treatment, reference=contrast.reference
        )
        if near_line(interval=interval):
            arms += [contrast.treatment, contrast.reference]
    return list(dict.fromkeys(arms))


def fit_row_set(
    *, frame: pl.DataFrame, domain: DomainType, row_set: str, workers: int
) -> pl.DataFrame:
    """Fit every arm of one set at the primary setting, then the extra second-setting fits.

    Args:
        frame: One set's rows from `aifs_rows`.
        domain: `solar` or `wind`.
        row_set: `single` or `ens`.
        workers: How many fits run at once.

    Returns:
        Every fit's per-row losses.
    """
    spec = ROW_SETS[row_set]
    no_doy = [f"{arm}{NO_DOY_SUFFIX}" for arm in spec.deciding or ()]
    primary = fit_jobs(
        frame=frame,
        domain=domain,
        jobs=[(arm, PRIMARY) for arm in (*spec.arms, *no_doy)],
        workers=workers,
    )
    second_arms = sensitivity_arms(
        losses=primary.filter(pl.col("setting") == PRIMARY), row_set=row_set
    )
    if not second_arms:
        return primary
    second = fit_jobs(
        frame=frame,
        domain=domain,
        jobs=[(arm, SENSITIVITY) for arm in second_arms],
        workers=workers,
    )
    return pl.concat([primary, second])


def deciding_verdict(
    *,
    primary: BootstrapInterval,
    sensitivity: BootstrapInterval,
    every_drop_same_sign: bool,
    no_doy_point: float,
) -> str:
    """Return whether the deciding contrast may be claimed, and which forecast has the lower error.

    The contrast is treatment minus reference, so a negative difference is a lower treatment error.

    Args:
        primary: The contrast at the primary setting.
        sensitivity: The contrast at the sensitivity setting.
        every_drop_same_sign: Whether the point estimate keeps its sign with each month dropped.
        no_doy_point: The point estimate refitted without `day_of_year`.

    Returns:
        A sentence for the report.
    """
    excludes_zero = all(r["lower_95"] > 0 or r["upper_95"] < 0 for r in (primary, sensitivity))
    signs = {np.sign(r["difference"]) for r in (primary, sensitivity)} | {np.sign(no_doy_point)}
    if excludes_zero and every_drop_same_sign and len(signs) == 1:
        lower = "lower" if primary["difference"] < 0 else "higher"
        return f"claimable: the treatment has the {lower} error at both settings"
    return "not claimable: the interval, the second setting, a month drop, or the refit disagrees"


def leave_one_month_out(
    *, losses: pl.DataFrame, treatment: str, reference: str
) -> tuple[float, float, float, bool]:
    """Return a contrast's point estimate with each month dropped in turn.

    Args:
        losses: Per-row losses at one setting.
        treatment: The arm whose error is compared.
        reference: The arm it is compared against.

    Returns:
        The point estimate over all months, the lowest and highest with one month dropped, and
        whether every drop keeps the all-months sign.
    """
    differences, months = paired_differences(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    full = float(differences.mean())
    without = [float(differences[:, months != month].mean()) for month in np.unique(months)]
    return full, min(without), max(without), all(value * full > 0.0 for value in without)


def era_line(*, losses: pl.DataFrame, frame: pl.DataFrame, contrast: Contrast, era: int) -> str:
    """Format one contrast inside one era: point estimate, rows, months, and month agreement.

    Args:
        losses: Per-row losses at the primary setting.
        frame: The set's rows, carrying `era_code`.
        contrast: The contrast to score.
        era: The era's code.

    Returns:
        A table row. An interval is printed only for an era with `MIN_ERA_MONTHS_FOR_INTERVAL`
        months or more.
    """
    subset = losses.join(
        frame.filter(pl.col("era_code") == era).select("site", "time"), on=["site", "time"]
    )
    differences, months = paired_differences(
        losses=subset, treatment=contrast.treatment, reference=contrast.reference, metric=METRIC
    )
    point = float(differences.mean())
    per_month = [float(differences[:, months == m].mean()) for m in np.unique(months)]
    agree = sum(value * point > 0.0 for value in per_month)
    text = f"{point * 100:+.3f}"
    if len(per_month) >= MIN_ERA_MONTHS_FOR_INTERVAL:
        interval = difference(
            losses=subset, treatment=contrast.treatment, reference=contrast.reference
        )
        text = interval_text(point=point, lower=interval["lower_95"], upper=interval["upper_95"])
    return (
        f"| era {era}: {contrast.treatment} − {contrast.reference} | {text} "
        f"| {differences.shape[1]} | {len(per_month)} | {agree} of {len(per_month)} |"
    )


def contrast_row(*, losses: pl.DataFrame, contrast: Contrast, label: str | None = None) -> str:
    """Format one contrast as a table row with its label (or `label`), rows and months."""
    result = difference(losses=losses, treatment=contrast.treatment, reference=contrast.reference)
    text = interval_text(
        point=result["difference"], lower=result["lower_95"], upper=result["upper_95"]
    )
    return (
        f"| {contrast.treatment} − {contrast.reference} | {text} | {result['n_rows']} "
        f"| {result['n_months']} | {contrast.label if label is None else label} |"
    )


CONTRAST_HEADER: Final[list[str]] = [
    "",
    "| Contrast (points) | Difference [95% interval] | Rows | Months | Label |",
    "|---|---|---|---|---|",
]


def errors_sentence(*, losses: pl.DataFrame, arms: Sequence[str]) -> str:
    """Name each arm's own absolute error, in percent of capacity, for a line beside a contrast.

    Args:
        losses: Per-row losses at one setting, holding every arm of `arms`.
        arms: The arms to name.

    Returns:
        A sentence listing each arm and its error.
    """
    board = leaderboard(losses=losses, arms=list(arms))
    parts = ", ".join(f"{r['arm']} {r['value'] * 100:.3f}" for r in board.iter_rows(named=True))
    return f"Absolute error at the primary setting (percent of capacity): {parts}."


def deciding_lines(
    *,
    losses: pl.DataFrame,
    treatment: str,
    reference: str,
    domain: DomainType,
    bonferroni_level: float,
    across: str,
) -> list[str]:
    """Report one deciding contrast at both settings, wider, month-dropped and no-`day_of_year`.

    Args:
        losses: Every fit's per-row losses on the set.
        treatment: The deciding contrast's treatment arm.
        reference: The deciding contrast's reference arm.
        domain: `solar` or `wind`.
        bonferroni_level: The coverage, in percent, of the wider interval.
        across: What the wider interval is corrected across, for its row label.

    Returns:
        Markdown lines.
    """
    primary_losses = losses.filter(pl.col("setting") == PRIMARY)
    second_losses = losses.filter(pl.col("setting") == SENSITIVITY)
    primary = difference(losses=primary_losses, treatment=treatment, reference=reference)
    second = difference(losses=second_losses, treatment=treatment, reference=reference)
    wide = bootstrap_difference_at_level(
        losses=primary_losses,
        treatment=treatment,
        reference=reference,
        metric=METRIC,
        level=bonferroni_level,
    )
    full, lowest, highest, same_sign = leave_one_month_out(
        losses=primary_losses, treatment=treatment, reference=reference
    )
    no_doy = difference(
        losses=primary_losses,
        treatment=f"{treatment}{NO_DOY_SUFFIX}",
        reference=f"{reference}{NO_DOY_SUFFIX}",
    )
    verdict = deciding_verdict(
        primary=primary,
        sensitivity=second,
        every_drop_same_sign=same_sign,
        no_doy_point=no_doy["difference"],
    )
    return [
        f"### Deciding contrast, {domain}: {treatment} − {reference}",
        "",
        "| Check | Difference [interval] (points) | Rows | Months |",
        "|---|---|---|---|",
        *(
            f"| {name} | "
            + interval_text(point=r["difference"], lower=r["lower_95"], upper=r["upper_95"])
            + f" | {r['n_rows']} | {r['n_months']} |"
            for name, r in (
                ("primary setting, 95%", primary),
                ("sensitivity setting, 95%", second),
                ("primary, without day_of_year in either arm, 95%", no_doy),
            )
        ),
        (
            f"| primary, {bonferroni_level}% ({across}) | "
            f"{full * 100:+.3f} [{wide[0] * 100:+.3f}, {wide[1] * 100:+.3f}] "
            f"| {primary['n_rows']} | {primary['n_months']} |"
        ),
        "",
        errors_sentence(losses=primary_losses, arms=(treatment, reference)),
        "",
        (
            f"With one month dropped in turn the point estimate ranges from {lowest * 100:+.3f} to "
            f"{highest * 100:+.3f} points; every drop keeps its sign: "
            f"{'yes' if same_sign else 'no'}."
        ),
        "",
        f"Verdict: {verdict}.",
        "",
    ]


def report_set(
    *, losses: pl.DataFrame, frame: pl.DataFrame, row_set: str, domain: DomainType
) -> list[str]:
    """Write one (technology, row set) report section.

    Args:
        losses: Every fit's per-row losses.
        frame: The set's rows.
        row_set: `single` or `ens`.
        domain: `solar` or `wind`.

    Returns:
        The section's Markdown lines.
    """
    spec = ROW_SETS[row_set]
    primary = losses.filter(pl.col("setting") == PRIMARY)
    coverage = coverage_table(frame=frame)
    per_era = (
        frame.group_by("era_code")
        .agg(rows=pl.len(), months=pl.col("month").n_unique())
        .sort("era_code")
    )
    lines = [
        f"## {domain.capitalize()}, row set `{row_set}`",
        "",
        (
            f"{frame.height} rows over {frame['month'].n_unique()} months at "
            f"{frame['site'].n_unique()} sites. "
            f"Calendar-month coverage: {coverage.filter(~pl.col('covered')).height} of "
            f"{coverage.height} (site, fold, calendar month) cells have no training row of their "
            "calendar month."
        ),
        "",
        "| Era | Rows | Months |",
        "|---|---|---|",
        *(
            f"| {r['era_code']} | {r['rows']} | {r['months']} |"
            for r in per_era.iter_rows(named=True)
        ),
        "",
        "### Feature columns of every arm",
        "",
        *(f"- {arm}: {', '.join(arm_features(arm=arm, domain=domain))}" for arm in spec.arms),
        "",
        "### Absolute error of every arm (GPU, primary setting)",
        "",
        "| Arm | Error (% of capacity) | 95% interval | Rows | Months |",
        "|---|---|---|---|---|",
    ]
    board = leaderboard(
        losses=primary,
        arms=[*spec.arms, *(f"{arm}{NO_DOY_SUFFIX}" for arm in spec.deciding or ())],
    )
    for row in board.iter_rows(named=True):
        text = error_text(value=row["value"], lower=row["lower_95"], upper=row["upper_95"])
        lines.append(
            f"| {row['arm']} | {text.split(' [')[0]} | [{text.split(' [')[1]} "
            f"| {row['n_rows']} | {row['n_months']} |"
        )
    if spec.deciding is not None:
        lines += [
            "",
            *deciding_lines(
                losses=losses,
                treatment=spec.deciding[0],
                reference=spec.deciding[1],
                domain=domain,
                bonferroni_level=BONFERRONI_LEVEL,
                across="Bonferroni across solar and wind",
            ),
        ]
    lines += ["### Listed contrasts (primary setting)", *CONTRAST_HEADER]
    lines += [contrast_row(losses=primary, contrast=c) for c in contrasts(row_set=row_set)]
    second = losses.filter(pl.col("setting") == SENSITIVITY)
    second_arms = set(second["arm"].unique().to_list())
    steps = difference(losses=primary, treatment="ens_mean6_day1", reference="ens_mean_day1")
    lines += [
        "",
        (
            "Positive control, ens_mean6_day1 − ens_mean_day1 (expected positive: 6-hourly steps "
            f"raise ENS's error): {steps['difference'] * 100:+.3f} points, "
            + (
                "passes."
                if steps["difference"] > 0
                else "FAILS: stop the write-up until explained."
            )
        ),
    ]
    lines += [
        "",
        "### Pairs also fitted at the sensitivity setting",
        *CONTRAST_HEADER,
    ]
    for contrast in contrasts(row_set=row_set):
        if not {contrast.treatment, contrast.reference} <= second_arms:
            continue
        if contrast.label == "deciding":
            why = "deciding"
        elif near_line(
            interval=difference(
                losses=primary, treatment=contrast.treatment, reference=contrast.reference
            )
        ):
            why = "near the 5% line"
        else:
            why = "both arms fitted at the second setting for other pairs"
        lines.append(contrast_row(losses=second, contrast=contrast, label=why))
    era_pair = "aifs_single_day1" if row_set == "single" else "aifs_ens_mean_day1"
    lines += [
        "",
        f"### {era_pair} − ens_mean6_day1 inside each version era (primary setting)",
        "",
        (
            "| Era contrast (points) | Point estimate [interval if 6 months or more] | Rows "
            "| Months | Months with the era's sign |"
        ),
        "|---|---|---|---|---|",
        *(
            era_line(
                losses=primary,
                frame=frame,
                contrast=Contrast(era_pair, "ens_mean6_day1", "era"),
                era=era,
            )
            for era in sorted(spec.era_runs)
        ),
        "",
    ]
    return lines


def path_for(*, output_dir: Path, domain: DomainType, row_set: str, kind: str) -> Path:
    """Return where one (technology, row set)'s saved `losses` or `predictions` live."""
    return output_dir / f"{domain}_{row_set}_{kind}.parquet"


def time_two_fits(*, frame: pl.DataFrame, arm: str, domain: DomainType) -> tuple[bool, float]:
    """Fit one arm at the first site of `frame` twice on the GPU and compare the fingerprints.

    Args:
        frame: A set's rows from `aifs_rows`.
        arm: The arm to fit at the primary setting.
        domain: `solar` or `wind`.

    Returns:
        Whether the two fingerprints agree, and the shorter of the two fit times in seconds.
    """
    site = min(frame["site"].unique().to_list())
    one_site = frame.filter(pl.col("site") == site)
    fingerprints = []
    seconds = []
    for _ in range(2):
        start = time.monotonic()
        losses = fit_jobs(frame=one_site, domain=domain, jobs=[(arm, PRIMARY)], workers=1)
        seconds.append(time.monotonic() - start)
        fingerprints.append(fingerprint(frame=losses))
    return fingerprints[0] == fingerprints[1], min(seconds)


def check_determinism(*, published_dir: Path, aifs_dir: Path) -> bool:
    """Fit one arm at one site twice on the GPU, compare fingerprints, and print a time estimate.

    Args:
        published_dir: The folder holding the published inputs.
        aifs_dir: The folder holding the AIFS inputs.

    Returns:
        Whether the two fingerprints agree.
    """
    spec = ROW_SETS["single"]
    frame = aifs_rows(
        published_dir=published_dir,
        aifs=pl.read_parquet(aifs_dir / "wind_aifs_inputs.parquet"),
        domain="wind",
        row_set="single",
        arms=spec.arms,
        day=None,
        shuffles=OLD_SHUFFLES,
    )
    agree, seconds = time_two_fits(frame=frame, arm="aifs_single_day1", domain="wind")
    n_fits = sum(len(spec.arms) for spec in ROW_SETS.values()) * sum(N_SITES.values())
    sys.stdout.write(
        f"one arm at one site: {seconds:.0f} s; {n_fits} primary (arm, site) fits are about "
        f"{n_fits * seconds / 3600:.1f} h on one worker (sensitivity, near-line and "
        "no-day_of_year refits add about a third)\n"
    )
    return agree


def sha256_of(*, path: Path) -> str:
    """Return a file's SHA-256 as a hex string."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_stamp(
    *,
    published_dir: Path,
    aifs_dir: Path,
    domain: DomainType,
    arms: Sequence[str],
    extra_dirs: Mapping[str, Path] | None = None,
) -> dict[str, str]:
    """Return what a saved losses file must match.

    Args:
        published_dir: The folder holding the published `<domain>_forecast_inputs.parquet`.
        aifs_dir: The folder holding `<domain>_aifs_inputs.parquet`.
        domain: `solar` or `wind`.
        arms: Every arm the saved losses may hold.
        extra_dirs: The extra-lead folders a blends fit reads, by name, or `None`.

    Returns:
        Every input file's SHA-256, the device, both hyper-parameter settings, the shuffle seeds
        (for a blends fit), and every arm's feature columns.
    """
    stamp = {
        "inputs_sha256": sha256_of(path=aifs_dir / f"{domain}_aifs_inputs.parquet"),
        "published_sha256": sha256_of(path=published_dir / f"{domain}_forecast_inputs.parquet"),
        "device": DEVICE,
        "settings": json.dumps(SETTINGS, sort_keys=True),
        "columns": json.dumps({arm: arm_features(arm=arm, domain=domain) for arm in sorted(arms)}),
    }
    if extra_dirs is not None:
        stamp["extra_sha256"] = json.dumps(
            {
                name: sha256_of(path=folder / f"{domain}_extra_lead_inputs.parquet")
                for name, folder in sorted(extra_dirs.items())
            }
        )
        stamp["seeds"] = json.dumps(SHUFFLE_SEEDS, sort_keys=True)
    return stamp


def check_gpu_visible() -> None:
    """Raise if `DEVICE` is `cuda` but `nvidia-smi` sees no GPU.

    XGBoost falls back to the CPU with only a logged warning when no GPU is visible, and the stamp
    would then name a device that fitted nothing.

    Raises:
        RuntimeError: If `nvidia-smi` exits non-zero.
    """
    if (
        DEVICE == "cuda"
        and subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=False).returncode
    ):
        msg = "DEVICE is cuda but nvidia-smi sees no GPU, so XGBoost would fit on the CPU"
        raise RuntimeError(msg)


def check_saved_losses(
    *,
    losses: pl.DataFrame,
    frame: pl.DataFrame,
    stage: str,
    arms: Collection[str],
    stamp_file: Path,
    stamp: dict[str, str],
) -> None:
    """Raise unless saved losses come from this build, this device, and exactly these rows.

    Args:
        losses: The saved losses.
        frame: `aifs_rows`'s frame for the stage.
        stage: The stage's name, for messages.
        arms: Every arm the primary losses must hold, and no others.
        stamp_file: The stamp written beside the losses file.
        stamp: `build_stamp`'s result for the current inputs and device.

    Raises:
        ValueError: If the stamp is missing or differs, or the primary losses hold other arms or
            cover other (site, time, fold) rows than the stage.
    """
    if not stamp_file.exists() or json.loads(stamp_file.read_text()) != stamp:
        msg = f"{stamp_file} is missing or names another build or device"
        raise ValueError(msg)
    primary = losses.filter(pl.col("setting") == PRIMARY)
    if set(primary["arm"].unique().to_list()) != set(arms):
        msg = f"{stage}: the saved losses hold other arms than the stage's"
        raise ValueError(msg)
    on = ["site", "time", "fold"]
    keys = frame.select(on)
    saved = primary.select(on).unique()
    if saved.join(keys, on=on, how="anti").height or keys.join(saved, on=on, how="anti").height:
        msg = f"{stage}: the saved losses cover other rows or folds than the stage's"
        raise ValueError(msg)


def old_mode_arms(*, row_set: str) -> list[str]:
    """Return every arm the day-1 and day-2 fit puts on one row set, refits included."""
    spec = ROW_SETS[row_set]
    return [*spec.arms, *(f"{arm}{NO_DOY_SUFFIX}" for arm in spec.deciding or ())]


# --- The blends fit: AIFS at days 1, 2, 7 and 14 ------------------------------------------------

BLEND_DAYS: Final[tuple[int, ...]] = (1, 2, 7, 14)
"""The lead days the blends fit scores, each on a frame of its own."""

LONG_DAYS: Final[tuple[int, ...]] = (7, 14)
"""The lead days beyond 144 hours, where ENS has 6-hourly steps natively and the deciding contrasts
sit."""

N_DECIDING: Final[int] = 8
"""Deciding contrasts across both technologies: H7, H14, B7 and B14 for each."""

BLEND_BONFERRONI_LEVEL: Final[float] = 100.0 - 5.0 / N_DECIDING
"""The coverage, in percent, of the wider interval: 95% corrected across the deciding contrasts."""

NULLABLE_PREFIXES: Final[tuple[str, ...]] = ("ifs_single_day7",)
"""Prefixes whose nulls stay in the rows: IFS HRES's missing target days are left missing, and its
contrasts run on the rows it has."""

NO_SKILL: Final[str] = "no skill to compare at day 14"
SKILL: Final[str] = "skill to compare at day 14"

EXTRA_FOLDERS: Final[dict[str, str]] = {
    "leads_day10": "nwp_forecast_comparison_leads_day10",
    "leads_day10b": "nwp_forecast_comparison_leads_day10b",
    "leads_day10d": "nwp_forecast_comparison_leads_day10d",
}
"""The extra-lead folders the blends fit reads, by short name, under `data/studies/`. The fit never
writes to them."""

EQUAL_TO_EXTRA: Final[dict[str, tuple[str, ...]]] = {
    "leads_day10b": ("ens_mean_day7", "ens_control_day7", "ens_control_day14"),
    "leads_day10": ("ens_mean_day14",),
}
"""The ENS prefixes the AIFS build reads natively at days 7 and 14, each with the extra-lead folder
whose earlier build wrote it. The two must agree on every row."""

JOINED_FROM_EXTRA: Final[dict[str, tuple[str, ...]]] = {
    "leads_day10": ("ifs025_day7",),
    "leads_day10d": ("ifs_single_day7",),
}
"""The prefixes joined from an extra-lead folder."""

ENS_STAMPED: Final[frozenset[str]] = frozenset(
    {f"ens_control6_day{day}" for day in BLEND_DAYS if day not in LONG_DAYS}
    | {f"ens_{way}_day{day}" for day in LONG_DAYS for way in ("mean", "control")}
)
"""The ENS prefixes that must carry a run stamp, so `check_runs` checks their run dates."""

EQUAL_TOLERANCE: Final[float] = 1e-6
"""The relative and absolute difference between the AIFS build's ENS columns and the extra-lead
folders' that Float32 storage allows."""


def ens_control_prefix(*, day: int) -> str:
    """Return the ENS control member's prefix at a day: 6-hourly at days 1 and 2, native beyond."""
    return f"ens_control_day{day}" if day in LONG_DAYS else f"ens_control6_day{day}"


def blend_arms(*, row_set: str, day: int) -> tuple[str, ...]:
    """Return every arm fitted on one (row set, day) frame.

    Args:
        row_set: `single` or `ens`.
        day: A day of `BLEND_DAYS`.

    Returns:
        On `single`: AIFS Single, ENS's mean and control member, the AIFS Single blend with its
        control and mirror control, at days 7 and 14 the two climatology arms, and at day 7 the two
        IFS references. On `ens`: the AIFS ENS mean, ENS's mean, and the AIFS ENS blend with its
        control.
    """
    mean = f"ens_mean_day{day}"
    if row_set == "ens":
        return (
            f"aifs_ens_mean_day{day}",
            mean,
            blend_arm_name(product="aifs_ens", day=day),
            blend_arm_name(product="aifs_ens", day=day, role="_control"),
        )
    single = f"aifs_single_day{day}"
    arms = [
        single,
        mean,
        ens_control_prefix(day=day),
        blend_arm_name(product="aifs_single", day=day),
        blend_arm_name(product="aifs_single", day=day, role="_control"),
        blend_arm_name(product="aifs_single", day=day, role="_mirror"),
    ]
    if day in LONG_DAYS:
        arms += [shuffled_prefix(source=single), shuffled_prefix(source=single, variant="_b")]
    if day == LONG_DAYS[0]:
        arms += [*JOINED_FROM_EXTRA["leads_day10"], *JOINED_FROM_EXTRA["leads_day10d"]]
    return tuple(arms)


def blend_shuffles(*, row_set: str, day: int) -> dict[str, tuple[str, ...]]:
    """Return the prefixes shuffled on one (row set, day) frame, with their variants.

    Args:
        row_set: `single` or `ens`.
        day: A day of `BLEND_DAYS`.

    Returns:
        AIFS's prefix (both seeds at days 7 and 14, where the null repeat is fitted, seed 0 alone at
        days 1 and 2), and on `single` ENS's mean too, for the mirror control (seed 0).
    """
    if row_set == "ens":
        return {f"aifs_ens_mean_day{day}": ("",)}
    return {
        f"aifs_single_day{day}": ("", "_b") if day in LONG_DAYS else ("",),
        f"ens_mean_day{day}": ("",),
    }


def stage_deciding_arms(*, row_set: str, day: int) -> tuple[str, ...]:
    """Return the arms of the deciding contrasts on one frame, which get the second setting.

    Args:
        row_set: `single` or `ens`.
        day: A day of `BLEND_DAYS`.

    Returns:
        AIFS Single, the ENS control member, ENS's mean, the blend and its control at day 7 and 14
        on `single`; nothing elsewhere.
    """
    if row_set != "single" or day not in LONG_DAYS:
        return ()
    return (
        f"aifs_single_day{day}",
        ens_control_prefix(day=day),
        f"ens_mean_day{day}",
        blend_arm_name(product="aifs_single", day=day),
        blend_arm_name(product="aifs_single", day=day, role="_control"),
    )


def stage_no_day_of_year_arms(*, row_set: str, day: int) -> tuple[str, ...]:
    """Return the arms refitted without `day_of_year`: H7's and H14's pair, on `single`."""
    if row_set != "single" or day not in LONG_DAYS:
        return ()
    return tuple(
        f"{arm}{NO_DOY_SUFFIX}" for arm in (f"aifs_single_day{day}", ens_control_prefix(day=day))
    )


def stage_arms_fitted(*, row_set: str, day: int) -> list[str]:
    """Return every arm one stage holds at the primary setting, refits included."""
    return [
        *blend_arms(row_set=row_set, day=day),
        *stage_no_day_of_year_arms(row_set=row_set, day=day),
    ]


def blend_contrasts(*, row_set: str, day: int) -> list[Contrast]:
    """Return every listed contrast of one (row set, day) frame, in report order.

    The deciding contrasts are H7, H14, B7 and B14 (the blend against ENS's mean, and against its
    control). The two-lead contrasts, the by-era contrasts and the gap between days 14 and 7 are
    built by the report from several stages.

    Args:
        row_set: `single` or `ens`.
        day: A day of `BLEND_DAYS`.

    Returns:
        The contrasts, each labelled deciding, exploratory or descriptive.
    """
    mean = f"ens_mean_day{day}"
    if row_set == "ens":
        descriptive = "descriptive, no deciding label"
        blend = blend_arm_name(product="aifs_ens", day=day)
        contrasts_ = [
            Contrast(blend, mean, descriptive),
            Contrast(blend, f"{blend}_control", f"{descriptive} (guard)"),
        ]
        if day in LONG_DAYS:
            contrasts_.insert(0, Contrast(f"aifs_ens_mean_day{day}", mean, descriptive))
        return contrasts_
    single = f"aifs_single_day{day}"
    control = ens_control_prefix(day=day)
    blend = blend_arm_name(product="aifs_single", day=day)
    mirror = blend_arm_name(product="aifs_single", day=day, role="_mirror")
    long = day in LONG_DAYS
    kind = f"deciding (B{day})" if long else "exploratory, post hoc"
    guard_kind = f"deciding (B{day} guard)" if long else "exploratory, post hoc (guard)"
    contrasts_ = []
    if long:
        contrasts_ += [
            Contrast(single, control, f"deciding (H{day})"),
            Contrast(single, mean, f"exploratory, beside H{day} (smoothing)"),
        ]
    contrasts_ += [
        Contrast(blend, mean, kind),
        Contrast(blend, f"{blend}_control", guard_kind),
        Contrast(blend, mirror, "exploratory, post hoc (column-matched with AIFS alone)"),
    ]
    if long:
        permuted = shuffled_prefix(source=single)
        permuted_b = shuffled_prefix(source=single, variant="_b")
        contrasts_ += [
            Contrast(single, permuted, "exploratory (climatology)"),
            Contrast(single, permuted_b, "exploratory (climatology, second seed)"),
            Contrast(permuted, permuted_b, "exploratory (null)"),
        ]
    if day == LONG_DAYS[0]:
        one_sided = "exploratory, one-sided reference"
        contrasts_ += [
            Contrast(single, "ifs025_day7", one_sided),
            Contrast(single, "ifs_single_day7", f"{one_sided}, rows IFS HRES has"),
            Contrast(control, mean, "exploratory, positive control (expected positive)"),
        ]
    if day == LONG_DAYS[1]:
        contrasts_ += [
            Contrast(control, shuffled_prefix(source=single), "exploratory (day-14 reading rule)"),
            Contrast(
                control,
                shuffled_prefix(source=single, variant="_b"),
                "exploratory (day-14 reading rule)",
            ),
        ]
    return contrasts_


def contrast_losses(
    *, losses: pl.DataFrame, frame: pl.DataFrame, domain: DomainType, contrast: Contrast
) -> pl.DataFrame:
    """Return `losses` restricted to the rows a contrast can be scored on.

    A contrast that names an arm in `NULLABLE_PREFIXES` runs on the rows where that arm's own
    columns hold values. Every other contrast keeps every row.

    Args:
        losses: Per-row losses of the stage.
        frame: The stage's rows.
        domain: `solar` or `wind`.
        contrast: The contrast to score.

    Returns:
        The losses on the rows both arms score.
    """
    nullable = [p for p in NULLABLE_PREFIXES if p in (contrast.treatment, contrast.reference)]
    if not nullable:
        return losses
    columns = [
        column
        for prefix in nullable
        for column in arm_columns(domain=domain, prefixes=(prefix,))
        if column.startswith(f"{prefix}_")
    ]
    present = frame.filter(pl.all_horizontal(pl.col(c).is_not_null() for c in columns))
    return losses.join(present.select("site", "time"), on=["site", "time"])


def blend_sensitivity_arms(
    *, losses: pl.DataFrame, frame: pl.DataFrame, domain: DomainType, row_set: str, day: int
) -> list[str]:
    """Return every arm to refit at the sensitivity setting on one frame.

    Args:
        losses: The stage's primary-setting per-row losses.
        frame: The stage's rows.
        domain: `solar` or `wind`.
        row_set: `single` or `ens`.
        day: A day of `BLEND_DAYS`.

    Returns:
        The deciding arms first, then both arms of every listed contrast that is near the 5% line,
        without repeats.
    """
    arms = list(stage_deciding_arms(row_set=row_set, day=day))
    for contrast in blend_contrasts(row_set=row_set, day=day):
        interval = difference(
            losses=contrast_losses(losses=losses, frame=frame, domain=domain, contrast=contrast),
            treatment=contrast.treatment,
            reference=contrast.reference,
        )
        if near_line(interval=interval):
            arms += [contrast.treatment, contrast.reference]
    return list(dict.fromkeys(arms))


def fit_blend_stage(
    *, frame: pl.DataFrame, domain: DomainType, row_set: str, day: int, workers: int
) -> pl.DataFrame:
    """Fit every arm of one (row set, day) frame, then the second-setting fits.

    Args:
        frame: The stage's rows from `blend_rows`.
        domain: `solar` or `wind`.
        row_set: `single` or `ens`.
        day: A day of `BLEND_DAYS`.
        workers: How many fits run at once.

    Returns:
        Every fit's per-row losses, stamped with the device.
    """
    primary = fit_jobs(
        frame=frame,
        domain=domain,
        jobs=[(arm, PRIMARY) for arm in stage_arms_fitted(row_set=row_set, day=day)],
        workers=workers,
    )
    second_arms = blend_sensitivity_arms(
        losses=primary, frame=frame, domain=domain, row_set=row_set, day=day
    )
    second = fit_jobs(
        frame=frame,
        domain=domain,
        jobs=[(arm, SENSITIVITY) for arm in second_arms],
        workers=workers,
    )
    return pl.concat([primary, second]).with_columns(device=pl.lit(DEVICE))


def check_columns_equal(
    *, built: pl.DataFrame, reference: pl.DataFrame, columns: Sequence[str], label: str
) -> None:
    """Raise unless `built` and `reference` hold the same values in `columns` on every row.

    Args:
        built: Rows keyed by `site` and `time`.
        reference: Rows keyed the same way, from another build.
        columns: The columns both frames hold.
        label: What `reference` is, for the message.

    Raises:
        ValueError: If a key of `built` is absent from `reference`, or a column's values differ by
            more than `EQUAL_TOLERANCE`, or are null in one frame and not in the other.
    """
    keys = ["site", "time"]
    if built.select(keys).join(reference.select(keys), on=keys, how="anti").height:
        msg = f"{label}: rows of the AIFS build are missing from it"
        raise ValueError(msg)
    joined = built.select(*keys, *columns).join(
        reference.select(*keys, *columns), on=keys, how="left", suffix="_reference"
    )
    unequal = {}
    for column in columns:
        ours = pl.col(column).cast(pl.Float64)
        theirs = pl.col(f"{column}_reference").cast(pl.Float64)
        bad = joined.select(
            (
                (ours.is_null() != theirs.is_null())
                | (
                    ours.is_not_null()
                    & theirs.is_not_null()
                    & ((ours - theirs).abs() > EQUAL_TOLERANCE + EQUAL_TOLERANCE * theirs.abs())
                )
            ).sum()
        ).item()
        if bad:
            unequal[column] = int(bad)
    if unequal:
        msg = f"{label}: columns differ from the AIFS build's on some rows: {unequal}"
        raise ValueError(msg)


def blend_inputs(
    *, aifs_dir: Path, extra_dirs: Mapping[str, Path], domain: DomainType
) -> pl.DataFrame:
    """Return the AIFS build's columns with the extra-lead columns the blends fit needs.

    The build's ENS columns at days 7 and 14 are renamed to their native names
    (`ens_mean6_day7_*` to `ens_mean_day7_*`, and likewise for the control member and the run
    stamps), checked against the extra-lead folders that hold the same columns, and the IFS
    references are joined on.

    Args:
        aifs_dir: The folder holding `<domain>_aifs_inputs.parquet`.
        extra_dirs: The extra-lead folders by `EXTRA_FOLDERS` short name.
        domain: `solar` or `wind`.

    Returns:
        The columns, keyed by `site` and `time`.

    Raises:
        ValueError: If an extra-lead folder lacks a row of the build or disagrees with it, or an
            ENS prefix that must carry a run stamp has none.
    """
    aifs = pl.read_parquet(aifs_dir / f"{domain}_aifs_inputs.parquet")
    renames = {
        column: column.replace(f"ens_{way}6_day{day}_", f"ens_{way}_day{day}_", 1)
        for column in aifs.columns
        for day in LONG_DAYS
        for way in ("mean", "control")
        if column.startswith(f"ens_{way}6_day{day}_")
    }
    aifs = aifs.rename(renames)
    unstamped = sorted(p for p in ENS_STAMPED if f"{p}_init_time" not in aifs.columns)
    if unstamped:
        msg = f"{domain}: the AIFS inputs carry no run stamp for {unstamped}; rebuild them"
        raise ValueError(msg)
    for name, prefixes in EQUAL_TO_EXTRA.items():
        extra = pl.read_parquet(extra_dirs[name] / f"{domain}_extra_lead_inputs.parquet")
        columns = [
            column
            for prefix in prefixes
            for column in aifs.columns
            if column.startswith(f"{prefix}_") and not column.endswith("_init_time")
        ]
        check_columns_equal(built=aifs, reference=extra, columns=columns, label=f"{domain}/{name}")
    keys = ["site", "time"]
    for name, prefixes in JOINED_FROM_EXTRA.items():
        extra = pl.read_parquet(extra_dirs[name] / f"{domain}_extra_lead_inputs.parquet")
        columns = [
            column
            for column in extra.columns
            if column.startswith(tuple(f"{p}_" for p in prefixes))
        ]
        if aifs.select(keys).join(extra.select(keys), on=keys, how="anti").height:
            msg = f"{domain}/{name}: rows of the AIFS inputs are missing from it"
            raise ValueError(msg)
        aifs = aifs.join(extra.select(*keys, *columns), on=keys, how="left")
    return aifs


def blend_rows(
    *,
    published_dir: Path,
    inputs: pl.DataFrame,
    domain: DomainType,
    row_set: str,
    day: int,
) -> pl.DataFrame:
    """Return one (row set, day) frame: the shared rows with each hour's day-`day` run in its era.

    Args:
        published_dir: The folder holding the published inputs.
        inputs: `blend_inputs`'s result.
        domain: `solar` or `wind`.
        row_set: `single` or `ens`.
        day: A day of `BLEND_DAYS`.

    Returns:
        `aifs_rows`'s frame for the stage's arms, with hours whose run lies outside their era
        dropped one row at a time.
    """
    return aifs_rows(
        published_dir=published_dir,
        aifs=inputs,
        domain=domain,
        row_set=row_set,
        arms=blend_arms(row_set=row_set, day=day),
        day=day,
        shuffles=blend_shuffles(row_set=row_set, day=day),
        nullable=NULLABLE_PREFIXES,
    )


def lead_verdict(
    *, day: int, versus_ens: BootstrapInterval, versus_control: BootstrapInterval
) -> BlendVerdict:
    """Return the verdict on a blend at one lead.

    The blend lowers the error only if the blend minus ENS alone and the blend minus its control
    (the same columns with AIFS shuffled) both have an upper 95% bound below zero. The published
    blend verdict is not reused, because its label names the day-ahead lead.

    Args:
        day: The lead day, for the label.
        versus_ens: The blend's error minus ENS alone's.
        versus_control: The blend's error minus its control's.

    Returns:
        `lowers the error at day N`, or `no detectable difference` with the largest gain the
        blend-minus-ENS lower bound leaves open (a positive number in the metric's unit).
    """
    if versus_ens["upper_95"] < 0.0 and versus_control["upper_95"] < 0.0:
        return {"verdict": f"lowers the error at day {day}", "largest_gain_not_excluded": None}
    return {
        "verdict": NO_DETECTABLE_DIFFERENCE,
        "largest_gain_not_excluded": max(0.0, -versus_ens["lower_95"]),
    }


def day14_reading(
    *, aifs_single: Sequence[BootstrapInterval], ens_control: Sequence[BootstrapInterval]
) -> str:
    """Return whether the day-14 frame holds skill to compare, by the reading rule.

    The rule needs AIFS Single or the ENS control member to have an error significantly below both
    shuffled-AIFS arms (two seeds), so one noisy shuffle cannot decide it.

    Args:
        aifs_single: AIFS Single minus each shuffled-AIFS arm.
        ens_control: The ENS control member minus each shuffled-AIFS arm.

    Returns:
        `SKILL` or `NO_SKILL`.
    """
    below = [
        all(interval["upper_95"] < 0.0 for interval in intervals)
        for intervals in (aifs_single, ens_control)
    ]
    return SKILL if any(below) else NO_SKILL


def smoothing_reading(*, versus_control: BootstrapInterval, versus_mean: BootstrapInterval) -> str:
    """Say what AIFS Single's error against the control member and against ENS's mean allows.

    Args:
        versus_control: AIFS Single's error minus the ENS control member's.
        versus_mean: AIFS Single's error minus ENS's mean's.

    Returns:
        A sentence for the report. AIFS Single has a lower error than the control member but not
        than ENS's mean is read as consistent with smoothing, not as better weather.
    """
    lower_than_control = versus_control["upper_95"] < 0.0
    lower_than_mean = versus_mean["upper_95"] < 0.0
    if lower_than_control and lower_than_mean:
        return "AIFS Single has a lower error than both the control member and the ENS mean."
    if lower_than_control:
        return (
            "AIFS Single has a lower error than the control member but not than the ENS mean: "
            "consistent with smoothing, so not described as the better weather forecast."
        )
    return "AIFS Single does not have a significantly lower error than the control member."


def summed_arm(*, losses: pl.DataFrame, arms: Sequence[str], name: str) -> pl.DataFrame:
    """Return one arm whose per-row loss is the sum of `arms`' losses on the rows they share.

    Args:
        losses: Per-row losses at one setting, holding every arm of `arms`.
        arms: The arms to add.
        name: The new arm's name.

    Returns:
        `arm`, `site`, `time`, `seed`, `month` and the metric column, one row per shared row.
    """
    keys = ["site", "time", "seed"]
    total: pl.DataFrame | None = None
    for arm in arms:
        part = losses.filter(pl.col("arm") == arm).select(*keys, "month", loss=pl.col(METRIC))
        total = (
            part
            if total is None
            else total.join(part, on=keys, how="inner", suffix="_next")
            .with_columns(loss=pl.col("loss") + pl.col("loss_next"))
            .drop("loss_next", "month_next")
        )
    if total is None:
        msg = "summed_arm needs at least one arm"
        raise ValueError(msg)
    return total.select(*keys, "month", arm=pl.lit(name), **{METRIC: pl.col("loss")})


def gap_lines(
    *, domain: DomainType, losses_by_day: Mapping[int, pl.DataFrame], frame_14: pl.DataFrame
) -> list[str]:
    """Report whether AIFS Single's gap to the ENS control member changes between days 7 and 14.

    The gap at day `N` is `aifs_single_dayN − ens_control_dayN`, so the change is
    `(aifs14 + control7) − (control14 + aifs7)`, built from summed per-row losses and read through
    `difference`. Day 7's losses are restricted to the day-14 rows.

    Args:
        domain: `solar` or `wind`.
        losses_by_day: Each day's stage losses on `single`.
        frame_14: The day-14 rows.

    Returns:
        Markdown lines.
    """
    rows_14 = frame_14.select("site", "time")
    seven = losses_by_day[7].filter(pl.col("setting") == PRIMARY).join(rows_14, on=["site", "time"])
    fourteen = losses_by_day[14].filter(pl.col("setting") == PRIMARY)
    both = pl.concat([fourteen, seven], how="diagonal_relaxed")
    up = summed_arm(losses=both, arms=("aifs_single_day14", "ens_control_day7"), name="gap_up")
    down = summed_arm(losses=both, arms=("ens_control_day14", "aifs_single_day7"), name="gap_down")
    result = difference(losses=pl.concat([up, down]), treatment="gap_up", reference="gap_down")
    text = interval_text(
        point=result["difference"], lower=result["lower_95"], upper=result["upper_95"]
    )
    return [
        f"### Gap at day 14 minus gap at day 7, {domain}, `single`, on the day-14 rows",
        "",
        (
            "The gap is aifs_single − ens_control at each day, so a negative change means AIFS "
            "Single's advantage over the control member widens with lead. Exploratory, post hoc."
        ),
        "",
        *CONTRAST_HEADER,
        (
            f"| (aifs_single_day14 − ens_control_day14) − (aifs_single_day7 − ens_control_day7) "
            f"| {text} | {result['n_rows']} | {result['n_months']} | exploratory, post hoc |"
        ),
        "",
        errors_sentence(
            losses=both,
            arms=("aifs_single_day7", "ens_control_day7", "aifs_single_day14", "ens_control_day14"),
        ),
        "",
    ]


def weather_spread(
    *, frame: pl.DataFrame, domain: DomainType, prefixes: Sequence[str]
) -> pl.DataFrame:
    """Return the standard deviation of each product's weather columns over the scored rows.

    Args:
        frame: The stage's rows.
        domain: `solar` or `wind`.
        prefixes: The products' column prefixes.

    Returns:
        `prefix`, `column`, `std` and `n_rows`, one row per weather column.
    """
    return pl.DataFrame(
        [
            {
                "prefix": prefix,
                "column": column,
                "std": frame[column].cast(pl.Float64).std(),
                "n_rows": frame.height,
            }
            for prefix in prefixes
            for column in arm_columns(domain=domain, prefixes=(prefix,))
            if column.startswith(f"{prefix}_")
        ]
    )


def deciding_blend_lines(*, losses: pl.DataFrame, domain: DomainType, day: int) -> list[str]:
    """Report one blend deciding contrast (B7 or B14): both bounds, both settings, and the verdict.

    Args:
        losses: The stage's per-row losses, both settings.
        domain: `solar` or `wind`.
        day: 7 or 14.

    Returns:
        Markdown lines.
    """
    blend = blend_arm_name(product="aifs_single", day=day)
    control = f"{blend}_control"
    mean = f"ens_mean_day{day}"
    by_setting = {
        setting: losses.filter(pl.col("setting") == setting) for setting in (PRIMARY, SENSITIVITY)
    }
    intervals = {
        setting: (
            difference(losses=frame, treatment=blend, reference=mean),
            difference(losses=frame, treatment=blend, reference=control),
            difference(losses=frame, treatment=control, reference=mean),
        )
        for setting, frame in by_setting.items()
    }
    verdicts = {
        setting: lead_verdict(day=day, versus_ens=pair[0], versus_control=pair[1])
        for setting, pair in intervals.items()
    }
    combined = combine_setting_verdicts(
        primary=verdicts[PRIMARY]["verdict"],
        sensitivity=verdicts[SENSITIVITY]["verdict"],
        unresolved=NO_DETECTABLE_DIFFERENCE,
    )
    wide = bootstrap_difference_at_level(
        losses=by_setting[PRIMARY],
        treatment=blend,
        reference=mean,
        metric=METRIC,
        level=BLEND_BONFERRONI_LEVEL,
    )
    full, lowest, highest, same_sign = leave_one_month_out(
        losses=by_setting[PRIMARY], treatment=blend, reference=mean
    )
    rows_ = []
    for setting, (versus_ens, versus_control, control_vs_ens) in intervals.items():
        for name, result in (
            (f"{blend} − {mean}", versus_ens),
            (f"{blend} − {control} (guard)", versus_control),
            (f"{control} − {mean}", control_vs_ens),
        ):
            text = interval_text(
                point=result["difference"], lower=result["lower_95"], upper=result["upper_95"]
            )
            rows_.append(
                f"| {name} | {setting}, 95% | {text} | {result['n_rows']} | {result['n_months']} |"
            )
    uninformative = any(pair[2]["lower_95"] > 0.0 for pair in intervals.values())
    gain = verdicts[PRIMARY]["largest_gain_not_excluded"]
    return [
        f"### Deciding contrast B{day}, {domain}: {blend} against {mean} and against its control",
        "",
        "| Contrast (points) | Setting | Difference [interval] | Rows | Months |",
        "|---|---|---|---|---|",
        *rows_,
        (
            f"| {blend} − {mean} | primary, {BLEND_BONFERRONI_LEVEL}% (Bonferroni across the "
            f"{N_DECIDING} deciding contrasts) | {full * 100:+.3f} "
            f"[{wide[0] * 100:+.3f}, {wide[1] * 100:+.3f}] | {intervals[PRIMARY][0]['n_rows']} "
            f"| {intervals[PRIMARY][0]['n_months']} |"
        ),
        "",
        (
            f"With one month dropped in turn the blend-minus-ENS point estimate ranges from "
            f"{lowest * 100:+.3f} to {highest * 100:+.3f} points; every drop keeps its sign: "
            f"{'yes' if same_sign else 'no'}."
        ),
        "",
        errors_sentence(losses=by_setting[PRIMARY], arms=(blend, control, mean)),
        "",
        (
            "The control is itself significantly worse than ENS alone at a setting, so the guard "
            "is uninformative there."
            if uninformative
            else "The control is not significantly worse than ENS alone at either setting."
        ),
        "",
        f"Verdict: {combined}"
        + (
            f"; the largest gain the primary lower bound leaves open is {gain * 100:.3f} points"
            if gain is not None and combined == NO_DETECTABLE_DIFFERENCE
            else ""
        )
        + ".",
        "",
    ]


def blend_stage_lines(
    *,
    domain: DomainType,
    row_set: str,
    day: int,
    frame: pl.DataFrame,
    losses: pl.DataFrame,
) -> list[str]:
    """Write one (technology, row set, day) report section.

    Args:
        domain: `solar` or `wind`.
        row_set: `single` or `ens`.
        day: A day of `BLEND_DAYS`.
        frame: The stage's rows.
        losses: The stage's per-row losses.

    Returns:
        The section's Markdown lines: the arms' columns, every arm's absolute error, the deciding
        contrasts, the listed contrasts at both settings, and the by-era lines.
    """
    arms = blend_arms(row_set=row_set, day=day)
    primary = losses.filter(pl.col("setting") == PRIMARY)
    second = losses.filter(pl.col("setting") == SENSITIVITY)
    lines = [
        f"### {domain.capitalize()}, `{row_set}`, day {day}",
        "",
        (
            f"{frame.height} rows over {frame['month'].n_unique()} months at "
            f"{frame['site'].n_unique()} sites."
        ),
        "",
        "#### Feature columns of every arm",
        "",
        *(
            f"- {arm}: {', '.join(arm_features(arm=arm, domain=domain))}"
            for arm in stage_arms_fitted(row_set=row_set, day=day)
        ),
        "",
        "#### Absolute error of every arm (GPU, primary setting)",
        "",
        "| Arm | Error (% of capacity) | 95% interval | Rows | Months |",
        "|---|---|---|---|---|",
    ]
    board = leaderboard(losses=primary, arms=stage_arms_fitted(row_set=row_set, day=day))
    for row in board.iter_rows(named=True):
        text = error_text(value=row["value"], lower=row["lower_95"], upper=row["upper_95"])
        lines.append(
            f"| {row['arm']} | {text.split(' [')[0]} | [{text.split(' [')[1]} "
            f"| {row['n_rows']} | {row['n_months']} |"
        )
    for prefix in NULLABLE_PREFIXES:
        if prefix in arms:
            restricted = contrast_losses(
                losses=primary,
                frame=frame,
                domain=domain,
                contrast=Contrast(prefix, prefix, "rows the arm has"),
            )
            only = leaderboard(losses=restricted, arms=[prefix]).row(0, named=True)
            text = error_text(value=only["value"], lower=only["lower_95"], upper=only["upper_95"])
            lines += [
                "",
                (
                    f"{prefix} on the {only['n_rows']} rows where it has values: {text} percent "
                    "of capacity."
                ),
            ]
    lines += [""]
    if row_set == "single" and day in LONG_DAYS:
        lines += deciding_lines(
            losses=losses,
            treatment=f"aifs_single_day{day}",
            reference=ens_control_prefix(day=day),
            domain=domain,
            bonferroni_level=BLEND_BONFERRONI_LEVEL,
            across=f"Bonferroni across the {N_DECIDING} deciding contrasts",
        )
        lines += deciding_blend_lines(losses=losses, domain=domain, day=day)
    contrast_list = blend_contrasts(row_set=row_set, day=day)
    lines += ["#### Listed contrasts (primary setting)", *ERROR_CONTRAST_HEADER]
    lines += [
        contrast_row_with_errors(
            losses=contrast_losses(losses=primary, frame=frame, domain=domain, contrast=c),
            contrast=c,
        )
        for c in contrast_list
    ]
    second_arms = set(second["arm"].unique().to_list())
    lines += ["", "#### Pairs also fitted at the sensitivity setting", *ERROR_CONTRAST_HEADER]
    for contrast in contrast_list:
        if not {contrast.treatment, contrast.reference} <= second_arms:
            continue
        interval = difference(
            losses=contrast_losses(losses=primary, frame=frame, domain=domain, contrast=contrast),
            treatment=contrast.treatment,
            reference=contrast.reference,
        )
        if contrast.label.startswith("deciding"):
            why = contrast.label
        elif near_line(interval=interval):
            why = "near the 5% line"
        else:
            why = "both arms fitted at the second setting for other pairs"
        lines.append(
            contrast_row_with_errors(
                losses=contrast_losses(
                    losses=second, frame=frame, domain=domain, contrast=contrast
                ),
                contrast=contrast,
                label=why,
            )
        )
    if row_set == "single" and day in LONG_DAYS:
        pair = Contrast(f"aifs_single_day{day}", ens_control_prefix(day=day), "era")
        lines += [
            "",
            f"#### {pair.treatment} − {pair.reference} inside each version era (primary setting)",
            "",
            (
                "| Era contrast (points) | Point estimate [interval if 6 months or more] | Rows "
                "| Months | Months with the era's sign |"
            ),
            "|---|---|---|---|---|",
            *(
                era_line(losses=primary, frame=frame, contrast=pair, era=era)
                for era in sorted(frame["era_code"].unique().to_list())
            ),
        ]
    lines += [""]
    return lines


def count_listed_intervals() -> int:
    """Return how many listed contrasts the blends report prints at the primary setting.

    Counts every stage's listed contrasts for both technologies, and the gap between days 14 and 7.
    The sensitivity-setting, Bonferroni, month-drop, and by-era lines are not counted.
    """
    per_technology = sum(
        len(blend_contrasts(row_set=row_set, day=day)) for row_set in ROW_SETS for day in BLEND_DAYS
    )
    return len(DOMAINS) * (per_technology + 1)


def blend_set_lines(
    *,
    domain: DomainType,
    row_set: str,
    frames: Mapping[int, pl.DataFrame],
    losses: Mapping[int, pl.DataFrame],
) -> list[str]:
    """Write one (technology, row set) report section, all four days.

    Args:
        domain: `solar` or `wind`.
        row_set: `single` or `ens`.
        frames: Each day's rows.
        losses: Each day's per-row losses.

    Returns:
        The section's Markdown lines: the row counts, each day's section, and on `single` the
        day-14 reading rule, the smoothing reading, the gap between days, and the spread of each
        product's weather columns.
    """
    lines = [
        f"## {domain.capitalize()}, row set `{row_set}`",
        "",
        "| Day | Rows | Months | Hours dropped: their run lies outside their AIFS version era |",
        "|---|---|---|---|",
        *(
            f"| {day} | {frames[day].height} | {frames[day]['month'].n_unique()} "
            f"| {frames[1].height - frames[day].height} |"
            for day in BLEND_DAYS
        ),
        "",
    ]
    for day in BLEND_DAYS:
        lines += blend_stage_lines(
            domain=domain, row_set=row_set, day=day, frame=frames[day], losses=losses[day]
        )
    if row_set != "single":
        return lines
    primary_14 = losses[14].filter(pl.col("setting") == PRIMARY)
    permuted = shuffled_prefix(source="aifs_single_day14")
    shuffled = [permuted, shuffled_prefix(source="aifs_single_day14", variant="_b")]
    reading = day14_reading(
        aifs_single=[
            difference(losses=primary_14, treatment="aifs_single_day14", reference=name)
            for name in shuffled
        ],
        ens_control=[
            difference(losses=primary_14, treatment="ens_control_day14", reference=name)
            for name in shuffled
        ],
    )
    lines += [
        f"### Day-14 reading rule, {domain}, `single`",
        "",
        (
            "Neither AIFS Single nor the ENS control member has an error significantly below both "
            "shuffled-AIFS arms (two seeds, primary setting, 95%) unless the rule says otherwise. "
            f"Reading: **{reading}**. H14's interval is printed above either way."
        ),
        "",
        "### Smoothing reading beside H7 and H14",
        "",
    ]
    for day in LONG_DAYS:
        primary = losses[day].filter(pl.col("setting") == PRIMARY)
        lines += [
            f"- Day {day}: "
            + smoothing_reading(
                versus_control=difference(
                    losses=primary,
                    treatment=f"aifs_single_day{day}",
                    reference=ens_control_prefix(day=day),
                ),
                versus_mean=difference(
                    losses=primary,
                    treatment=f"aifs_single_day{day}",
                    reference=f"ens_mean_day{day}",
                ),
            )
        ]
    positive = difference(
        losses=losses[7].filter(pl.col("setting") == PRIMARY),
        treatment="ens_control_day7",
        reference="ens_mean_day7",
    )
    lines += [
        "",
        (
            "Positive control, ens_control_day7 − ens_mean_day7 (expected positive: the mean "
            f"smooths the control member): {positive['difference'] * 100:+.3f} points, "
            + ("passes." if positive["difference"] > 0 else "FAILS: stop the write-up.")
        ),
        "",
        *gap_lines(domain=domain, losses_by_day=losses, frame_14=frames[14]),
        "### Spread of each forecast's weather columns over the scored rows",
        "",
        (
            "A smoother forecast has a smaller standard deviation. The rows are each day's scored "
            "rows."
        ),
        "",
        "| Day | Forecast | Column | Standard deviation | Rows |",
        "|---|---|---|---|---|",
    ]
    for day in BLEND_DAYS:
        spread = weather_spread(
            frame=frames[day],
            domain=domain,
            prefixes=(f"aifs_single_day{day}", ens_control_prefix(day=day), f"ens_mean_day{day}"),
        )
        lines += [
            f"| {day} | {r['prefix']} | {r['column']} | {r['std']:.4f} | {r['n_rows']} |"
            for r in spread.iter_rows(named=True)
        ]
    lines += [""]
    return lines


def check_blends(*, published_dir: Path, aifs_dir: Path, extra_dirs: Mapping[str, Path]) -> bool:
    """Fit `aifs_single_day7` at one wind site twice on the GPU, and print a time estimate.

    Args:
        published_dir: The folder holding the published inputs.
        aifs_dir: The blends folder holding the AIFS inputs.
        extra_dirs: The extra-lead folders.

    Returns:
        Whether the two fingerprints agree.
    """
    inputs = blend_inputs(aifs_dir=aifs_dir, extra_dirs=extra_dirs, domain="wind")
    frame = blend_rows(
        published_dir=published_dir, inputs=inputs, domain="wind", row_set="single", day=7
    )
    agree, seconds = time_two_fits(frame=frame, arm="aifs_single_day7", domain="wind")
    n_arms = sum(
        len(stage_arms_fitted(row_set=row_set, day=day))
        for row_set in ROW_SETS
        for day in BLEND_DAYS
    )
    n_fits = n_arms * sum(N_SITES.values())
    sys.stdout.write(
        f"one arm at one site: {seconds:.0f} s; {n_fits} primary (arm, site) fits are about "
        f"{n_fits * seconds / 3600:.1f} h on one worker (the sensitivity and near-line refits add "
        "about a third)\n"
    )
    return agree


def refuse_read_only_folders(*, output_dir: Path, read_only: Collection[Path]) -> None:
    """Raise if `output_dir` is any folder the blends fit must not write to.

    Args:
        output_dir: Where the fit would write.
        read_only: The published folder, the day-1 and day-2 AIFS folder, and the extra-lead
            folders.

    Raises:
        ValueError: If `output_dir` resolves to any of them.
    """
    if output_dir.resolve() in {folder.resolve() for folder in read_only}:
        msg = f"{output_dir} is a folder the blends fit reads and never writes to"
        raise ValueError(msg)


def run_blends(
    *, published_dir: Path, output_dir: Path, extra_dirs: Mapping[str, Path], workers: int
) -> int:
    """Fit every stage of the blends fit and write its outputs once.

    Args:
        published_dir: The folder holding the published inputs.
        output_dir: The new folder holding `<domain>_aifs_inputs.parquet`, built with
            `--aifs-days 1 2 7 14`, which receives every output.
        extra_dirs: The extra-lead folders by `EXTRA_FOLDERS` short name.
        workers: How many (arm, site) fits run at once.

    Returns:
        0.
    """
    report_path = output_dir / "report.md"
    refuse_to_overwrite(paths=[report_path])
    sections: list[str] = []
    for domain in DOMAINS:
        inputs = blend_inputs(aifs_dir=output_dir, extra_dirs=extra_dirs, domain=domain)
        for row_set in ROW_SETS:
            frames: dict[int, pl.DataFrame] = {}
            stage_losses: dict[int, pl.DataFrame] = {}
            for day in BLEND_DAYS:
                frame = blend_rows(
                    published_dir=published_dir,
                    inputs=inputs,
                    domain=domain,
                    row_set=row_set,
                    day=day,
                )
                stage = f"{row_set}_day{day}"
                losses_file = path_for(
                    output_dir=output_dir, domain=domain, row_set=stage, kind="losses"
                )
                predictions_file = path_for(
                    output_dir=output_dir, domain=domain, row_set=stage, kind="predictions"
                )
                stamp_file = losses_file.with_suffix(".json")
                arms = stage_arms_fitted(row_set=row_set, day=day)
                stamp = build_stamp(
                    published_dir=published_dir,
                    aifs_dir=output_dir,
                    domain=domain,
                    arms=arms,
                    extra_dirs=extra_dirs,
                )
                if losses_file.exists():
                    losses = pl.read_parquet(losses_file)
                    check_saved_losses(
                        losses=losses,
                        frame=frame,
                        stage=f"{domain}/{stage}",
                        arms=arms,
                        stamp_file=stamp_file,
                        stamp=stamp,
                    )
                else:
                    refuse_to_overwrite(paths=[predictions_file, stamp_file])
                    losses = fit_blend_stage(
                        frame=frame, domain=domain, row_set=row_set, day=day, workers=workers
                    )
                    losses.write_parquet(losses_file)
                    stamp_file.write_text(json.dumps(stamp))
                if not predictions_file.exists():
                    predictions_from_losses(losses=losses, frame=frame).write_parquet(
                        predictions_file
                    )
                frames[day] = frame
                stage_losses[day] = losses
            sections += blend_set_lines(
                domain=domain, row_set=row_set, frames=frames, losses=stage_losses
            )
    n_listed = count_listed_intervals()
    header = [
        "# AIFS Single and AIFS ENS at days 1, 2, 7 and 14, and blends with ENS's mean: report",
        "",
        (
            "Every fit is on the GPU. Differences are first arm minus second, in percentage points "
            "of capacity, so a negative difference means the first arm has the lower error. Four "
            "contrasts per technology are deciding, named before any fit of this report: H7 and "
            "H14 (AIFS Single against the ENS control member, `single`) and B7 and B14 (the blend "
            "of ENS's mean and AIFS Single against ENS's mean alone and against its control, "
            "`single`). They are not planned contrasts in the published page's sense: the day-1 "
            "and day-2 AIFS results and the ENS results at days 7 and 14 were known when they "
            "were named. Every other contrast is exploratory or, on `ens`, descriptive. The report "
            f"prints {n_listed} listed intervals at the primary setting across both technologies, "
            f"so about {n_listed / 20:.1f} would reach statistical significance at the 5% level by "
            "chance."
        ),
        "",
        (
            "AIFS Single may have the lower error because it is smoother, not because its weather "
            "is better: H7 and H14 are read beside aifs_single − ens_mean, and the spread of each "
            "forecast's weather columns is printed."
        ),
        "",
    ]
    report_path.write_text("\n".join([*header, *sections]))
    return 0


# --- The P4 second-seed control refit -----------------------------------------------------------

P4_SECOND_SEED: Final[int] = 1000
"""The second shuffle seed of each published P4 control. A product's shuffle uses this plus the
product's index in the blend (1 for the first other product, 2 for the second), as the published
guard's `20260920` plus that index."""

P4_REFERENCE: Final[str] = "ens_mean_day1"
"""The arm each P4 blend is contrasted with."""

P4_BLENDS: Final[tuple[str, ...]] = ("blend_p4a", "blend_p4b")
"""The two published blends."""

UNRESOLVED_ACROSS_SEEDS: Final[str] = "unresolved: the two shuffle seeds disagree on the guard"

P4_DIR_NAME: Final[str] = "nwp_forecast_comparison_p4_seeds"
"""Under `data/studies/`, the folder the P4 refit writes to, once."""

BLENDS_DIR_NAME: Final[str] = "nwp_forecast_comparison_aifs_blends"
"""Under `data/studies/`, the folder of the blends fit, which the P4 refit never writes to."""


def p4_arms() -> tuple[str, ...]:
    """Return every arm of the P4 refit: ENS's day-1 mean, and each blend with both controls."""
    return (
        P4_REFERENCE,
        *(arm for blend in P4_BLENDS for arm in (blend, f"{blend}_control", f"{blend}_control_b")),
    )


def add_second_seed_guard_columns(*, frame: pl.DataFrame, domain: DomainType) -> pl.DataFrame:
    """Add each P4 blend's second-seed guard columns: the non-ENS products' weather, shuffled again.

    The shuffle is `add_blend_guard_columns`'s, with the seed `P4_SECOND_SEED` plus the product's
    index in place of the published `20260920` plus that index, so a value moves only among the
    hours sharing a site, a year-month and an hour of day, and a product's direction sine and
    cosine (wind) move together.

    Args:
        frame: The published rows with the published guard columns.
        domain: `solar` or `wind`.

    Returns:
        `frame` with `<prefix>_permuted_b_<field>` for every non-ENS blend product's weather field.
    """
    output = frame
    for blend in P4_BLENDS:
        for index, prefix in enumerate(BLEND_ARMS[blend][1:], start=1):
            fields = tuple(
                column
                for column in arm_columns(domain=domain, prefixes=(prefix,))
                if column.startswith(f"{prefix}_")
            )
            groups = (
                [(fields[0],), (fields[1], fields[2]), (fields[3],)]
                if domain == "wind"
                else [(fields[0],), (fields[1],)]
            )
            shuffled = shuffled_prefix(source=prefix, variant="_b")
            output = climatology_permutation(
                frame=output,
                column_groups=groups,
                by=("site", "month", "hour_of_day"),
                seed=P4_SECOND_SEED + index,
                suffix=f"_{shuffled}",
            ).rename(
                {
                    f"{column}_{shuffled}": f"{shuffled}_{column.removeprefix(f'{prefix}_')}"
                    for group in groups
                    for column in group
                }
            )
    return output


def p4_frame(*, published_dir: Path, domain: DomainType) -> pl.DataFrame:
    """Return the published rows and folds with both guards' columns, for the P4 refit.

    The rows, the folds and the first guard are the published run's own (`rows` and
    `add_blend_guard_columns`), in the published row order.

    Args:
        published_dir: The folder holding the published inputs.
        domain: `solar` or `wind`.

    Returns:
        The frame.

    Raises:
        ValueError: If an arm's source column has a null, or the folds leave a calendar month
            uncovered.
    """
    frame = add_second_seed_guard_columns(
        frame=add_blend_guard_columns(
            frame=rows(input_dir=published_dir, domain=domain), domain=domain
        ),
        domain=domain,
    )
    coverage_table(frame=frame)
    check_no_missing(frame=frame, columns=source_columns(arms=p4_arms(), domain=domain))
    return frame


def fit_p4(*, frame: pl.DataFrame, domain: DomainType, workers: int) -> pl.DataFrame:
    """Fit every P4 arm at both settings, out of fold, on the GPU.

    Args:
        frame: `p4_frame`'s result.
        domain: `solar` or `wind`.
        workers: How many (arm, site) fits run at once.

    Returns:
        Every fit's per-row losses, stamped with the device.
    """
    losses = fit_jobs(
        frame=frame,
        domain=domain,
        jobs=[(arm, setting) for arm in p4_arms() for setting in SETTINGS],
        workers=workers,
    )
    return losses.with_columns(device=pl.lit(DEVICE))


def seed_agreement_verdict(*, first: str, second: str) -> str:
    """Return the blend verdict both shuffle seeds give, or `UNRESOLVED_ACROSS_SEEDS`.

    Args:
        first: The verdict with the first control (the published shuffle) as the guard.
        second: The verdict with the second-seed control as the guard.

    Returns:
        `first` where the two agree, otherwise `UNRESOLVED_ACROSS_SEEDS`.
    """
    return first if first == second else UNRESOLVED_ACROSS_SEEDS


def p4_contrasts() -> list[Contrast]:
    """Return every listed P4 contrast, in report order, each with both controls.

    Returns:
        For each blend: the blend minus ENS's day-1 mean, the blend minus each control (the
        guards), each control minus ENS's day-1 mean, and the seed-to-seed gap (the first control
        minus the second).
    """
    contrasts_ = []
    for blend in P4_BLENDS:
        name = blend.removeprefix("blend_").replace("p4", "P4")
        control, second = f"{blend}_control", f"{blend}_control_b"
        contrasts_ += [
            Contrast(blend, P4_REFERENCE, f"{name}: blend minus ENS mean day 1"),
            Contrast(blend, control, f"{name} guard, first control (published shuffle)"),
            Contrast(blend, second, f"{name} guard, second control (seed {P4_SECOND_SEED})"),
            Contrast(control, P4_REFERENCE, f"{name}: first control minus ENS mean day 1"),
            Contrast(second, P4_REFERENCE, f"{name}: second control minus ENS mean day 1"),
            Contrast(control, second, f"{name} seed-to-seed gap: first control minus second"),
        ]
    return contrasts_


def p4_verdicts(*, losses: pl.DataFrame) -> dict[str, str]:
    """Return the published blend verdict at one setting, with each control as the guard.

    Args:
        losses: The P4 losses at one setting.

    Returns:
        `first` and `second`: `studies.bootstrap.blend_verdict` with P4a's and P4b's guard being
        the first control and the second control, and `agreed`, the verdict where the two seeds
        agree and `UNRESOLVED_ACROSS_SEEDS` where they do not.
    """
    verdicts = {}
    for name, suffix in (("first", "_control"), ("second", "_control_b")):
        intervals = {
            blend: (
                difference(losses=losses, treatment=blend, reference=P4_REFERENCE),
                difference(losses=losses, treatment=blend, reference=f"{blend}{suffix}"),
            )
            for blend in P4_BLENDS
        }
        verdicts[name] = blend_verdict(
            p4a=intervals["blend_p4a"][0],
            p4a_guard=intervals["blend_p4a"][1],
            p4b=intervals["blend_p4b"][0],
            p4b_guard=intervals["blend_p4b"][1],
        )["verdict"]
    verdicts["agreed"] = seed_agreement_verdict(first=verdicts["first"], second=verdicts["second"])
    return verdicts


ERROR_CONTRAST_HEADER: Final[list[str]] = [
    "",
    (
        "| Contrast (points) | Difference [95% interval] | Error of the first arm (%) "
        "| Error of the second arm (%) | Rows | Months | Label |"
    ),
    "|---|---|---|---|---|---|---|",
]


def contrast_row_with_errors(
    *, losses: pl.DataFrame, contrast: Contrast, label: str | None = None
) -> str:
    """Format one contrast as a table row, with both arms' own absolute errors beside it.

    Args:
        losses: Per-row losses at one setting, holding both arms.
        contrast: The contrast.
        label: The row's label, or the contrast's own.

    Returns:
        `| first − second | difference [interval] | first's error | second's error | rows | months
        | label |`, the errors in percent of capacity.
    """
    result = difference(losses=losses, treatment=contrast.treatment, reference=contrast.reference)
    errors = leaderboard(losses=losses, arms=[contrast.treatment, contrast.reference])
    by_arm = {row["arm"]: row["value"] * 100 for row in errors.iter_rows(named=True)}
    text = interval_text(
        point=result["difference"], lower=result["lower_95"], upper=result["upper_95"]
    )
    return (
        f"| {contrast.treatment} − {contrast.reference} | {text} "
        f"| {by_arm[contrast.treatment]:.3f} | {by_arm[contrast.reference]:.3f} "
        f"| {result['n_rows']} | {result['n_months']} | "
        f"{contrast.label if label is None else label} |"
    )


def p4_lines(*, domain: DomainType, frame: pl.DataFrame, losses: pl.DataFrame) -> list[str]:
    """Write one technology's P4 refit section.

    Args:
        domain: `solar` or `wind`.
        frame: `p4_frame`'s result.
        losses: The P4 per-row losses, both settings.

    Returns:
        Markdown lines: the arms' columns, every arm's absolute error, every contrast with both
        arms' errors beside it at each setting, the verdict with each control as the guard, and
        the seed-agreement rule.
    """
    by_setting_ = {name: losses.filter(pl.col("setting") == name) for name in SETTINGS}
    lines = [
        f"## {domain.capitalize()}: the published P4 blends refitted with two controls",
        "",
        (
            f"{frame.height} rows over {frame['month'].n_unique()} months at "
            f"{frame['site'].n_unique()} sites, the published run's rows and folds."
        ),
        "",
        "### Feature columns of every arm",
        "",
        *(f"- {arm}: {', '.join(arm_features(arm=arm, domain=domain))}" for arm in p4_arms()),
        "",
        "### Absolute error of every arm (GPU)",
        "",
        "| Arm | Setting | Error (% of capacity) | 95% interval | Rows | Months |",
        "|---|---|---|---|---|---|",
    ]
    for setting, setting_losses in by_setting_.items():
        for row in leaderboard(losses=setting_losses, arms=list(p4_arms())).iter_rows(named=True):
            text = error_text(value=row["value"], lower=row["lower_95"], upper=row["upper_95"])
            lines.append(
                f"| {row['arm']} | {setting} | {text.split(' [')[0]} | [{text.split(' [')[1]} "
                f"| {row['n_rows']} | {row['n_months']} |"
            )
    for setting, setting_losses in by_setting_.items():
        lines += [
            "",
            f"### Contrasts at the {setting} setting",
            *ERROR_CONTRAST_HEADER,
            *(
                contrast_row_with_errors(losses=setting_losses, contrast=contrast)
                for contrast in p4_contrasts()
            ),
        ]
    verdicts = {
        setting: p4_verdicts(losses=setting_losses)
        for setting, setting_losses in by_setting_.items()
    }
    lines += ["", "### Blend verdict with each control as the guard", ""]
    lines += [
        f"- {setting} setting: first control {v['first']}; second control {v['second']}; "
        f"seeds agree: {'yes' if v['first'] == v['second'] else 'no'}."
        for setting, v in verdicts.items()
    ]
    combined = {
        key: combine_setting_verdicts(
            primary=verdicts[PRIMARY][key],
            sensitivity=verdicts[SENSITIVITY][key],
            unresolved=NO_DETECTABLE_DIFFERENCE,
        )
        for key in ("first", "second")
    }
    final = seed_agreement_verdict(first=combined["first"], second=combined["second"])
    lines += [
        "",
        (
            f"Combined across the two settings: first control {combined['first']}; second "
            f"control {combined['second']}. **Verdict: {final}.**"
        ),
        "",
        (
            "If the two seeds disagree on the guard's verdict, the page must call the blend claim "
            "unresolved. A control significantly worse than ENS alone makes its guard "
            "uninformative."
        ),
        "",
    ]
    return lines


def check_p4(*, published_dir: Path) -> bool:
    """Fit `blend_p4b` at one wind site twice on the GPU, and print the refit's time estimate.

    Args:
        published_dir: The folder holding the published inputs.

    Returns:
        Whether the two fingerprints agree.
    """
    frame = p4_frame(published_dir=published_dir, domain="wind")
    agree, seconds = time_two_fits(frame=frame, arm="blend_p4b", domain="wind")
    n_fits = len(p4_arms()) * sum(N_SITES.values())
    slower = SETTINGS[SENSITIVITY]["num_boost_round"] / SETTINGS[PRIMARY]["num_boost_round"]
    sys.stdout.write(
        f"one P4 arm at one site: {seconds:.0f} s; the P4 refit is {n_fits} fits at each of the "
        f"two settings, about {n_fits * seconds / 60:.0f} min at the primary setting plus at most "
        f"{n_fits * seconds * slower / 60:.0f} min at the sensitivity setting (its rounds are "
        f"{slower:.1f} times as many), on one worker\n"
    )
    return agree


def p4_stamp(*, published_dir: Path, domain: DomainType) -> dict[str, str]:
    """Return what the saved P4 losses must match: the published inputs, the device, the seeds."""
    return {
        "published_sha256": sha256_of(path=published_dir / f"{domain}_forecast_inputs.parquet"),
        "device": DEVICE,
        "settings": json.dumps(SETTINGS, sort_keys=True),
        "seeds": json.dumps({"first": "add_blend_guard_columns", "second": P4_SECOND_SEED}),
        "columns": json.dumps({arm: arm_features(arm=arm, domain=domain) for arm in p4_arms()}),
    }


def run_p4(*, published_dir: Path, output_dir: Path, workers: int) -> int:
    """Refit the P4 arms and their two controls, and write the outputs once.

    Args:
        published_dir: The folder holding the published inputs, which is only read.
        output_dir: The new folder that receives every output.
        workers: How many (arm, site) fits run at once.

    Returns:
        0.
    """
    report_path = output_dir / "report.md"
    refuse_to_overwrite(paths=[report_path])
    output_dir.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    for domain in DOMAINS:
        frame = p4_frame(published_dir=published_dir, domain=domain)
        losses_file = path_for(output_dir=output_dir, domain=domain, row_set="p4", kind="losses")
        predictions_file = path_for(
            output_dir=output_dir, domain=domain, row_set="p4", kind="predictions"
        )
        stamp_file = losses_file.with_suffix(".json")
        stamp = p4_stamp(published_dir=published_dir, domain=domain)
        if losses_file.exists():
            losses = pl.read_parquet(losses_file)
            check_saved_losses(
                losses=losses,
                frame=frame,
                stage=f"{domain}/p4",
                arms=p4_arms(),
                stamp_file=stamp_file,
                stamp=stamp,
            )
        else:
            refuse_to_overwrite(paths=[predictions_file, stamp_file])
            losses = fit_p4(frame=frame, domain=domain, workers=workers)
            losses.write_parquet(losses_file)
            stamp_file.write_text(json.dumps(stamp))
        if not predictions_file.exists():
            predictions_from_losses(losses=losses, frame=frame).write_parquet(predictions_file)
        sections += p4_lines(domain=domain, frame=frame, losses=losses)
    header = [
        "# The published P4 blends refitted on the GPU with two shuffle seeds: report",
        "",
        (
            "Every fit is on the GPU, on the published run's rows and folds. The published page's "
            "fits are on the CPU, so nothing here is compared with the published numbers. "
            "Differences are first arm minus second, in percentage points of capacity, so a "
            "negative difference means the first arm has the lower error. Each P4 blend is "
            f"contrasted with {P4_REFERENCE} and with two controls, the published shuffle and a "
            f"second shuffle under seed {P4_SECOND_SEED} plus the product's index, both within "
            "site, year-month and hour of day. Every contrast is exploratory. If the two seeds "
            "disagree on the guard's verdict, the page must call the blend claim unresolved."
        ),
        "",
    ]
    report_path.write_text("\n".join([*header, *sections]))
    return 0


def main() -> int:
    """Fit every arm on both row sets and both technologies, and write the outputs once."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--published-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="(arm, site) fits run at once.")
    parser.add_argument("--check", action="store_true", help="Compare two GPU runs of one arm.")
    parser.add_argument(
        "--blends",
        action="store_true",
        help="Fit AIFS at days 1, 2, 7 and 14 and the blends with ENS's mean, from the inputs "
        "`build_forecast_inputs.py --aifs --aifs-days 1 2 7 14` wrote to --output-dir.",
    )
    parser.add_argument(
        "--p4-controls",
        action="store_true",
        help="Refit the published P4 blends on the GPU with their first control and a "
        "second-seed control, into a new --output-dir (no AIFS inputs are read).",
    )
    args = parser.parse_args()
    check_gpu_visible()
    if args.output_dir.resolve() == args.published_dir.resolve():
        msg = "the output folder must not be the published folder"
        raise ValueError(msg)
    if args.p4_controls:
        studies_dir = args.published_dir.resolve().parent
        refuse_read_only_folders(
            output_dir=args.output_dir,
            read_only=[
                args.published_dir,
                studies_dir / EXISTING_AIFS_DIR_NAME,
                studies_dir / BLENDS_DIR_NAME,
                *(studies_dir / folder for folder in EXTRA_FOLDERS.values()),
            ],
        )
        if args.check:
            agree = check_p4(published_dir=args.published_dir)
            sys.stdout.write(f"two GPU runs agree: {agree}\n")
            return 0 if agree else 1
        return run_p4(
            published_dir=args.published_dir, output_dir=args.output_dir, workers=args.workers
        )
    if args.blends:
        studies_dir = args.published_dir.resolve().parent
        extra_dirs = {name: studies_dir / folder for name, folder in EXTRA_FOLDERS.items()}
        refuse_read_only_folders(
            output_dir=args.output_dir,
            read_only=[
                args.published_dir,
                studies_dir / EXISTING_AIFS_DIR_NAME,
                *extra_dirs.values(),
            ],
        )
        if args.check:
            agree = check_blends(
                published_dir=args.published_dir, aifs_dir=args.output_dir, extra_dirs=extra_dirs
            )
            sys.stdout.write(f"two GPU runs agree: {agree}\n")
            return 0 if agree else 1
        return run_blends(
            published_dir=args.published_dir,
            output_dir=args.output_dir,
            extra_dirs=extra_dirs,
            workers=args.workers,
        )
    if args.check:
        agree = check_determinism(published_dir=args.published_dir, aifs_dir=args.output_dir)
        sys.stdout.write(f"two GPU runs agree: {agree}\n")
        return 0 if agree else 1
    report_path = args.output_dir / "report.md"
    refuse_to_overwrite(paths=[report_path])
    report = [
        "# AIFS Single and AIFS ENS at matched leads, GPU fits: report",
        "",
        (
            "Every fit is on the GPU. Differences are first arm minus second, in percentage points "
            "of capacity, so a negative difference means the first arm has the lower error. One "
            "contrast is named deciding before any fit: aifs_single_day1 − ens_control6_day1 on "
            "`single`, for solar and for wind. The AIFS ENS contrasts on `ens` are descriptive "
            "only, because most scored `ens` cells have no training row of their calendar month. "
            "Every other contrast is exploratory and post hoc; among that many intervals, about "
            "1 in 20 reaches significance at the 5% level by chance."
        ),
        "",
    ]
    for domain in DOMAINS:
        for row_set in ROW_SETS:
            frame = aifs_rows(
                published_dir=args.published_dir,
                aifs=pl.read_parquet(args.output_dir / f"{domain}_aifs_inputs.parquet"),
                domain=domain,
                row_set=row_set,
                arms=ROW_SETS[row_set].arms,
                day=None,
                shuffles=OLD_SHUFFLES,
            )
            losses_file = path_for(
                output_dir=args.output_dir, domain=domain, row_set=row_set, kind="losses"
            )
            predictions_file = path_for(
                output_dir=args.output_dir, domain=domain, row_set=row_set, kind="predictions"
            )
            stamp = build_stamp(
                published_dir=args.published_dir,
                aifs_dir=args.output_dir,
                domain=domain,
                arms=sorted({arm for name in ROW_SETS for arm in old_mode_arms(row_set=name)}),
            )
            stamp_file = losses_file.with_suffix(".json")
            if losses_file.exists():
                losses = pl.read_parquet(losses_file)
                check_saved_losses(
                    losses=losses,
                    frame=frame,
                    stage=row_set,
                    arms=old_mode_arms(row_set=row_set),
                    stamp_file=stamp_file,
                    stamp=stamp,
                )
            else:
                refuse_to_overwrite(paths=[predictions_file, stamp_file])
                losses = fit_row_set(
                    frame=frame, domain=domain, row_set=row_set, workers=args.workers
                )
                losses.write_parquet(losses_file)
                stamp_file.write_text(json.dumps(stamp))
            if not predictions_file.exists():
                predictions_from_losses(losses=losses, frame=frame).write_parquet(predictions_file)
            report += report_set(losses=losses, frame=frame, row_set=row_set, domain=domain)
    report_path.write_text("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
