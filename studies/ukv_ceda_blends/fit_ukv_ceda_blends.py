"""Fit ENS's mean with and without UKV-CEDA at lead days 1 to 4, and write the losses and report.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1016>. It answers whether the
03 UTC run of UKV-CEDA lowers the power-forecast error of the ECMWF ENS mean at lead days 1 to 4.
For solar and for wind separately, at each lead day `N`, it fits four arms out of fold on the GPU,
each an XGBoost model per generator:

- `blend_ukv_ceda_dayN_pad`: ENS's mean padded to the blend's column count with exact copies of its
  own columns, the reference.
- `blend_ukv_ceda_dayN`: ENS's mean plus UKV-CEDA's columns.
- `blend_ukv_ceda_dayN_control` and `blend_ukv_ceda_dayN_control_b`: ENS's mean plus UKV-CEDA's
  columns shuffled within generator, year-month, and hour of day, under two shuffle seeds.

Two contrasts are planned before any fit. **P1** is the blend minus the padded ENS arm. **P2** is
the blend minus each control. The blend lowers the error at a technology and lead day only if the
upper 95% bound of P1 and of both P2 contrasts is below zero at both hyperparameter settings. The
rows, eras, folds, settings, seeds, metric, and paired month-resampled intervals are those of
`nwp_forecast_comparison` and `fit_aifs`, which this script imports and does not change. Each lead
day has its own rows, so contrasts are made within a lead day only.

**Rows.** `nwp_forecast_comparison.candidate_rows` (2024-12 onwards, 2026-01 dropped) joined with
ENS's day-4 mean and the UKV-CEDA inputs `build_ukv_ceda_inputs.py` wrote, kept where the target,
ENS's columns at that day, and UKV-CEDA's columns at that day are all present. The folds are the
shared design (`assign_folds_with_eras`); a stage whose rows leave a calendar month uncovered takes
the first rotation `search_fold_offsets` returns, before any fit.

**Outputs**, all under `--output-dir` (`data/studies/ukv_ceda_blends`) and written once:
`<domain>_day<N>_planned_losses.parquet`, `_predictions.parquet`, and a `.json` stamp for each
technology and lead day, `wind_day1_cpu_losses.parquet` (the GPU-CPU noise floor), `report.md`, and
`intervals.parquet`. Every output carries only the anonymised `site` label.

**Modes.** `--dry-run` builds every frame, checks every arm's columns, and lists the fits, writing
nothing. `--check` fits one arm at one wind site twice on the GPU, stops unless the two fingerprints
agree, prints a time estimate, and compares ENS's mean alone with its padded copy on the wind day-1
rows. A stage whose losses exist is not refitted. `--only-missing` fits the (arm, setting) pairs no
saved file holds into a new `_added_<k>` file. `--post-hoc-stale` fits the post hoc stale blend
(below) into a new `_added_<k>` file per stage, after every planned pair is saved, and writes a
report under `--report-name`. `--report-only` writes the report from the saved losses to a new
`--report-name`, fitting nothing. Run `uptime` and `nvidia-smi` before a fit, and
start only below a load average of about 24.

**Post hoc stale blend.** After the first science review, `--post-hoc-stale` adds, for lead days 1
to 3, the arm `blend_ukv_ceda_stale_dayN`: ENS day `N` plus UKV-CEDA day `N + 1`, the 03 UTC run one
day before ENS's run, which is 21 hours staler than ENS's where the planned blend's is 3 hours
fresher. It has the planned arms' column counts, and its controls are the shuffles of those
columns under both shuffle seeds. Its padded ENS reference is refitted on
the stale rows (`blend_ukv_ceda_stale_dayN_pad`), so the blend and the reference train on the same
rows, and the report prints the planned reference minus the stale-rows reference as the measured
tilt from the planned reference's extra training rows. Every contrast scores the rows where
UKV-CEDA's day `N + 1` is present. These arms are exploratory and post hoc.

**Post hoc permutation test (solar).** After the second science review, `--post-hoc-permutation`
fits, for each solar stage (lead days 1 to 4), 15 further shuffled controls
(`blend_ukv_ceda_dayN_control_s<seed>`) at the primary setting only. They use the same shuffle
groups as the planned control (within generator, year-month, and hour of day) under seeds that no
planned shuffle uses. The report prints the planned blend's P1 against the distribution of
(control minus padded ENS) over all 17 shuffled controls (the planned two and the 15 extra), with
the rank of P1 and a permutation p-value, per stage. The test is exploratory and post hoc.

**Post hoc older run.** `--post-hoc-older-run` adds, for lead days 1 to 3 and both technologies,
the arm `blend_ukv_ceda_run15_dayN`: ENS day `N` plus UKV-CEDA columns built from the 15 UTC run of
the day before ENS's run (`build.OLDER_RUN`), read from `--older-dir`. That run starts 9 hours
before ENS's 00 UTC run and leads 12 hours longer than the planned blend's run. It is fitted at both
settings with a padded ENS reference refitted on its rows and one shuffled control (seed 0). The
arm tests how the gain falls as the UKV-CEDA run gets older. It cannot separate the effect of the
run's lead from the effect of its timing against ENS's run, because the two move together.

Run it with `uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --dry-run`.
"""

import argparse
import json
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Final, Literal, NamedTuple, TypedDict

import build_ukv_ceda_inputs as build
import fit_aifs
import numpy as np
import polars as pl
from fit_extra_leads import error_text, interval_text
from nwp_forecast_comparison import (
    METRIC,
    NWP_ERA_FOLD_OFFSETS,
    NWP_ERA_START_MONTHS,
    PERCENTAGE_POINTS,
    SETTINGS,
    TARGET,
    DomainType,
    assign_folds_with_eras,
    coverage_table,
    difference,
    leaderboard,
    predictions_from_losses,
)
from studies.bootstrap import (
    MIN_MONTHS_FOR_INTERVAL,
    NO_DETECTABLE_DIFFERENCE,
    BootstrapInterval,
    bootstrap_difference_at_level,
    combine_setting_verdicts,
    paired_differences,
)
from studies.cross_validation import (
    calendar_month_coverage,
    cut_eras,
    out_of_fold_losses,
    search_fold_offsets,
    uncovered_months,
)
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.sources import REPO_DATA_DIR

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

PRIMARY: Final[str] = fit_aifs.PRIMARY
SENSITIVITY: Final[str] = fit_aifs.SENSITIVITY

PRODUCT: Final[str] = "ukv_ceda"
"""The product's key in `fit_aifs.BLEND_AIFS_PREFIXES`."""

DAYS: Final[tuple[int, ...]] = build.LEAD_DAYS

PLANNED_GROUP: Final[str] = "planned"
CPU_GROUP: Final[str] = "cpu"
CPU_SITE: Final[str] = "W1"
"""The wind site whose day-1 blend is refitted on the CPU to measure the GPU-CPU difference."""

MAX_WORKERS: Final[int] = 2
"""The load rule: at most two (arm, site) fits at once, each on `THREADS_PER_FIT` threads."""

N_P1_INTERVALS: Final[int] = 2 * len(DAYS)
"""The planned P1 intervals per setting: four lead days at two technologies."""

BONFERRONI_LEVEL: Final[float] = 100.0 - 5.0 / N_P1_INTERVALS
"""The coverage, in percent, of the wider P1 interval: 95% corrected across `N_P1_INTERVALS`."""

UNRESOLVED_LOWER: Final[str] = "unresolved: lower than padded ENS, control test not passed"
"""The reading where P1 is below zero at both settings but some P2 bound is not: the blend's error
is statistically significantly lower than padded ENS's, and the planned control test is not met."""

STALE_PRODUCT: Final[str] = "ukv_ceda_stale"
"""The key in `fit_aifs.BLEND_AIFS_PREFIXES` of the post hoc blend that reads UKV-CEDA's run one
lead day older than ENS's, which makes the UKV-CEDA run 21 hours staler than ENS's."""

STALE_DAYS: Final[tuple[int, ...]] = tuple(day for day in DAYS if day + 1 in DAYS)
"""The ENS lead days of the stale blend: UKV-CEDA's day `N + 1` must be built."""

STALE_ROLES: Final[tuple[fit_aifs.BlendRoleType, ...]] = ("_pad", "", "_control", "_control_b")
"""The roles fitted for the stale blend: the padded ENS reference, the blend, and the controls of
both shuffle seeds, all trained on the stale rows."""

STALE_SCOPE: Final[str] = "post hoc stale"
"""The `scope` of every stale-blend interval in `intervals.parquet`."""

PostHocKind = Literal["stale", "permutation", "older run"]
"""Which post hoc analysis a fit adds."""

PERMUTATION_SEEDS: Final[tuple[int, ...]] = tuple(range(2010, 2160, 10))
"""The shuffle seeds of the post hoc permutation test's 15 extra controls. A shuffle of two columns
(solar) uses its seed and the next one (`studies.blending.climatology_permutation`), so no two draws
share a seed, and none shares one with the planned shuffles (0, 1, 1000, and 1001)."""

PERMUTATION_SCOPE: Final[str] = "post hoc permutation"
"""The `scope` of every permutation-test row in `intervals.parquet`."""

PERMUTATION_DRAWS: Final[int] = len(PERMUTATION_SEEDS) + len(fit_aifs.SHUFFLE_SEEDS)
"""The shuffled controls the permutation test compares with: the planned two and the extra 15."""

OLDER_PRODUCT: Final[str] = "ukv_ceda_run15"
"""The key in `fit_aifs.BLEND_AIFS_PREFIXES` of the post hoc blend that reads UKV-CEDA's 15 UTC run
of the day before ENS's run."""

OLDER_DAYS: Final[tuple[int, ...]] = build.OLDER_RUN.lead_days
"""The ENS lead days of the older-run blend."""

OLDER_ROLES: Final[tuple[fit_aifs.BlendRoleType, ...]] = ("_pad", "", "_control")
"""The roles fitted for the older-run blend: the padded ENS reference, the blend, and one control
(shuffle seed 0), all trained on the rows where the older run is present."""

OLDER_SCOPE: Final[str] = "post hoc older run"
"""The `scope` of every older-run interval in `intervals.parquet`."""

HOURLY_LEAD_LIMIT_HOURS: Final[int] = 48
"""The store holds UKV-CEDA hourly to this lead and every 3 hours after it, so a row read beyond
this lead holds values rebuilt from 3-hourly steps."""

OLDER_SPLIT_SCOPES: Final[dict[bool, str]] = {
    True: f"{OLDER_SCOPE}, lead 48 hours or less",
    False: f"{OLDER_SCOPE}, lead beyond 48 hours",
}
"""The scope of a day-1 older-run contrast on the rows at or before the hourly limit (`True`) and
beyond it (`False`). Neither equals `OLDER_SCOPE`, so no chart reads these rows as the whole."""

TIME_RESOLUTION_CAVEAT: Final[str] = (
    "The store holds UKV-CEDA hourly only to lead 48 hours, so at lead days 1 and 2 more of this "
    "run's hours are rebuilt from 3-hourly steps than the planned run's. The contrast therefore "
    "also changes the time resolution of the UKV-CEDA columns."
)
"""The sentence the post hoc stale and older-run sections carry on the confound they share."""

PADDING_CHECK_NAME: Final[str] = "padding_check.json"
"""Where `--check` writes whether ENS's mean alone and its padded copy score identically."""

SECOND_SEED_VARIANT: Final[str] = "_b"
BEFORE_UPGRADE_ERAS: Final[tuple[int, ...]] = (0, 1)
AFTER_UPGRADE_ERAS: Final[tuple[int, ...]] = (2,)

ROW_KEYS: Final[tuple[str, ...]] = ("site", "time", "fold")

SPECS: Final[Mapping[str, str]] = {
    "p1": "blend minus padded ENS",
    "p2": "blend minus control",
    "p2b": "blend minus second-seed control",
}


class Stage(NamedTuple):
    """One frame to build and fit: a technology and a lead day."""

    domain: DomainType
    day: int


class IntervalRecord(TypedDict):
    """One interval the report prints, for `intervals.parquet`."""

    domain: str
    day: int
    setting: str
    contrast: str
    scope: str
    level: float
    difference: float
    lower: float
    upper: float
    n_rows: int
    n_months: int


# --- Arms and jobs --------------------------------------------------------------------------------


def arm_name(*, day: int, role: fit_aifs.BlendRoleType) -> str:
    """Return one of a lead day's blend arms, such as `blend_ukv_ceda_day2_control`."""
    return fit_aifs.blend_arm_name(product=PRODUCT, day=day, role=role)


def stage_arms(*, day: int) -> tuple[str, str, str, str]:
    """Return a lead day's arms: the padded ENS reference, the blend, and its two controls."""
    return (
        arm_name(day=day, role="_pad"),
        arm_name(day=day, role=""),
        arm_name(day=day, role="_control"),
        arm_name(day=day, role="_control_b"),
    )


def stale_arm_name(*, day: int, role: fit_aifs.BlendRoleType) -> str:
    """Return one of the post hoc stale blend's arms, such as `blend_ukv_ceda_stale_day2`."""
    return fit_aifs.blend_arm_name(product=STALE_PRODUCT, day=day, role=role)


def stale_arms(*, day: int) -> tuple[str, ...]:
    """Return the stale arms of lead day `day`: padded reference, blend, and both controls."""
    return tuple(stale_arm_name(day=day, role=role) for role in STALE_ROLES)


def is_stale_arm(*, arm: str) -> bool:
    """Return whether an arm is one of the post hoc stale blend's."""
    return arm.startswith(f"{fit_aifs.BLEND_PREFIX}{STALE_PRODUCT}_day")


def stale_jobs(*, day: int) -> list[fit_aifs.Job]:
    """Return the post hoc stale fits of a lead day: its arms at both settings, none at day 4."""
    if day not in STALE_DAYS:
        return []
    return [(arm, setting) for arm in stale_arms(day=day) for setting in SETTINGS]


def permutation_arm_name(*, day: int, seed: int) -> str:
    """Return one extra shuffled control, such as `blend_ukv_ceda_day2_control_s2010`."""
    return f"{arm_name(day=day, role='_control')}_s{seed}"


def permutation_arms(*, day: int) -> tuple[str, ...]:
    """Return a lead day's 15 extra shuffled controls, one per `PERMUTATION_SEEDS`."""
    return tuple(permutation_arm_name(day=day, seed=seed) for seed in PERMUTATION_SEEDS)


def is_permutation_arm(*, arm: str) -> bool:
    """Return whether an arm is one of the post hoc permutation test's extra controls."""
    pattern = rf"{fit_aifs.BLEND_PREFIX}{PRODUCT}_day\d+_control_s\d+"
    return re.fullmatch(pattern, arm) is not None


def permutation_jobs(*, stage: Stage) -> list[fit_aifs.Job]:
    """Return the permutation test's fits of a stage: each extra control at the primary setting.

    Only solar stages are tested, so a wind stage has none.
    """
    if stage.domain != "solar":
        return []
    return [(arm, PRIMARY) for arm in permutation_arms(day=stage.day)]


def older_arm_name(*, day: int, role: fit_aifs.BlendRoleType) -> str:
    """Return one of the post hoc older-run blend's arms, such as `blend_ukv_ceda_run15_day2`."""
    return fit_aifs.blend_arm_name(product=OLDER_PRODUCT, day=day, role=role)


def older_arms(*, day: int) -> tuple[str, ...]:
    """Return the older-run arms of lead day `day`: padded reference, blend, and one control."""
    return tuple(older_arm_name(day=day, role=role) for role in OLDER_ROLES)


def is_older_arm(*, arm: str) -> bool:
    """Return whether an arm is one of the post hoc older-run blend's."""
    return arm.startswith(f"{fit_aifs.BLEND_PREFIX}{OLDER_PRODUCT}_day")


def older_jobs(*, day: int) -> list[fit_aifs.Job]:
    """Return the older-run fits of a lead day: its arms at both settings, none at day 4."""
    if day not in OLDER_DAYS:
        return []
    return [(arm, setting) for arm in older_arms(day=day) for setting in SETTINGS]


def planned_jobs(*, day: int) -> list[fit_aifs.Job]:
    """Return every (arm, setting) fit of a lead day: each arm at both settings."""
    return [(arm, setting) for arm in stage_arms(day=day) for setting in SETTINGS]


def stages() -> list[Stage]:
    """Return every stage, solar before wind, each lead day in order."""
    return [Stage(domain=domain, day=day) for domain in fit_aifs.DOMAINS for day in DAYS]


def stem(*, stage: Stage, group: str) -> str:
    """Return the file stem of a stage's group, such as `wind_day1_planned`."""
    return f"{stage.domain}_day{stage.day}_{group}"


def group_files(*, output_dir: Path, stage: Stage) -> list[Path]:
    """Return every saved losses file of a stage's fits at the two settings, oldest group first.

    Args:
        output_dir: The write-once folder.
        stage: The stage.

    Returns:
        The `planned` file and every `added_<k>` file that exists. The CPU refit is not included.
    """
    prefix = f"{stage.domain}_day{stage.day}_"
    found = [
        path
        for path in output_dir.glob(f"{prefix}*_losses.parquet")
        if re.fullmatch(
            rf"{PLANNED_GROUP}|added_\d+",
            path.name.removeprefix(prefix).removesuffix("_losses.parquet"),
        )
    ]
    return sorted(
        found, key=lambda path: (path.name != f"{prefix}{PLANNED_GROUP}_losses.parquet", path.name)
    )


def saved_pairs(*, output_dir: Path, stage: Stage) -> set[fit_aifs.Job]:
    """Return every (arm, setting) pair that a saved losses file of the stage holds."""
    pairs: set[fit_aifs.Job] = set()
    for path in group_files(output_dir=output_dir, stage=stage):
        pairs |= set(pl.read_parquet(path).select("arm", "setting").unique().iter_rows())
    return pairs


def next_group(*, output_dir: Path, stage: Stage) -> str:
    """Return the group the next fit of a stage writes: `planned`, then `added_1`, `added_2`."""
    existing = group_files(output_dir=output_dir, stage=stage)
    return PLANNED_GROUP if not existing else f"added_{len(existing)}"


def saved_losses(*, output_dir: Path, stage: Stage) -> pl.DataFrame:
    """Return every saved loss of a stage at both settings, stacked."""
    return pl.concat(
        [pl.read_parquet(path) for path in group_files(output_dir=output_dir, stage=stage)],
        how="diagonal",
    )


# --- Rows -----------------------------------------------------------------------------------------


def present(*, column: str) -> pl.Expr:
    """Return whether a column holds a value: neither null nor not-a-number."""
    return pl.col(column).is_not_null() & ~pl.col(column).is_nan().fill_null(value=False)


def ukv_columns(
    *, domain: DomainType, day: int, spec: build.RunSpec = build.PLANNED_RUN
) -> list[str]:
    """Return UKV-CEDA's weather columns at one lead day, from the planned or the older run."""
    return [f"{spec.column_prefix}_day{day}_{field}" for field in build.WEATHER_FIELDS[domain]]


def check_init_times(
    *,
    frame: pl.DataFrame,
    domain: DomainType,
    day: int,
    spec: build.RunSpec = build.PLANNED_RUN,
) -> None:
    """Raise unless every row's stamped run is the run `spec` says its hour reads.

    The expected run is recomputed here from the plan's rule and not from the function the build
    stamps with: the run that starts at `spec.run_hour` UTC on the day `day + spec.extra_days` days
    before the hour's own day. A solar label names the hour ending at it, so its day is the day of
    the label minus one hour. A wind label is an instant, so its day is its own.

    Args:
        frame: Rows carrying `time` and `<prefix>_day<N>_init_time`.
        domain: `solar` or `wind`.
        day: The lead day.
        spec: Which run is read.

    Raises:
        ValueError: Naming how many rows read another run than the one the plan names.
    """
    instant = pl.col("time") - pl.duration(hours=1) if domain == "solar" else pl.col("time")
    expected = (
        instant.dt.truncate("1d")
        - pl.duration(days=day + spec.extra_days)
        + pl.duration(hours=spec.run_hour)
    )
    wrong = frame.filter(pl.col(f"{spec.column_prefix}_day{day}_init_time") != expected).height
    if wrong:
        msg = (
            f"{domain} day {day}: {wrong} rows read a run other than the {spec.run_hour:02d} UTC "
            f"run of day D-{day + spec.extra_days}"
        )
        raise ValueError(msg)


def cut_folds(*, frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[int, int]]:
    """Assign the shared design's folds, or the first covering rotation if the design leaves a gap.

    Args:
        frame: Rows carrying `site`, `time`, and `month`.

    Returns:
        The rows with `era_code`, `era`, and `fold`, and the fold offsets of each era used.

    Raises:
        ValueError: If no rotation covers every calendar month.
    """
    shared = assign_folds_with_eras(frame=frame)
    if uncovered_months(coverage=calendar_month_coverage(frame=shared)).is_empty():
        return shared, dict(NWP_ERA_FOLD_OFFSETS)
    found = search_fold_offsets(frame=frame, first_months=NWP_ERA_START_MONTHS)
    if not found:
        msg = "no fold rotation leaves a training row for every calendar month of every site"
        raise ValueError(msg)
    offsets = dict(found[0])
    return (
        cut_eras(frame=frame, first_months=NWP_ERA_START_MONTHS, fold_offsets=offsets),
        offsets,
    )


def add_copy_columns(*, frame: pl.DataFrame, domain: DomainType, day: int) -> pl.DataFrame:
    """Add exact copies of ENS's mean columns, which pad the reference to the blend's column count.

    Args:
        frame: Rows carrying ENS's mean columns at the lead day.
        domain: `solar` or `wind`.
        day: The lead day.

    Returns:
        `frame` with `ens_mean_day<N>_copy_<field>` equal to `ens_mean_day<N>_<field>`.
    """
    source = f"ens_mean_day{day}"
    return frame.with_columns(
        pl.col(column).alias(f"{source}{fit_aifs.COPY_SUFFIX}_{column.removeprefix(f'{source}_')}")
        for column in build.ens_columns(domain=domain, day=day)
    )


def stage_frame(
    *,
    stage: Stage,
    candidates: pl.DataFrame,
    inputs: pl.DataFrame,
    permutation: bool = False,
) -> tuple[pl.DataFrame, dict[int, int]]:
    """Return the rows every arm of one stage is trained and scored on.

    Args:
        stage: The technology and lead day.
        candidates: `build_ukv_ceda_inputs.published_rows`' result for the technology.
        inputs: The technology's `<domain>_ukv_ceda_inputs.parquet`.
        permutation: Whether to add the permutation test's 15 extra shuffles as well. They are
            further columns on the same rows, so no planned arm's rows or columns change.

    Returns:
        The rows with folds, ENS's copies, and the shuffled UKV-CEDA copies, and the fold offsets.

    Raises:
        ValueError: If a candidate row has no row in the inputs, a row reads the wrong run, an arm's
            column holds a missing value, or a calendar month is left uncovered.
    """
    domain, day = stage
    keys = ["site", "time"]
    if candidates.select(keys).join(inputs.select(keys), on=keys, how="anti").height:
        msg = f"{domain}: candidate rows are missing from the UKV-CEDA inputs"
        raise ValueError(msg)
    stamp = f"{PRODUCT}_day{day}_init_time"
    joined = candidates.join(
        inputs.select(*keys, stamp, *ukv_columns(domain=domain, day=day)), on=keys, how="left"
    )
    required = [
        TARGET,
        *build.ens_columns(domain=domain, day=day),
        *ukv_columns(domain=domain, day=day),
    ]
    kept = joined.filter(pl.all_horizontal(present(column=column) for column in required)).sort(
        keys
    )
    check_init_times(frame=kept, domain=domain, day=day)
    cut, offsets = cut_folds(frame=kept)
    padded = add_copy_columns(frame=cut, domain=domain, day=day)
    arms = stage_arms(day=day)
    shuffled_arms = (*arms, *permutation_arms(day=day)) if permutation else arms
    frame = fit_aifs.add_shuffled_columns(
        frame=padded, domain=domain, shuffles=fit_aifs.control_shuffles(arms=shuffled_arms)
    )
    check_no_missing(frame=frame, columns=fit_aifs.source_columns(arms=arms, domain=domain))
    coverage_table(frame=frame)
    return frame, offsets


def stale_stage_frame(*, stage: Stage, frame: pl.DataFrame, inputs: pl.DataFrame) -> pl.DataFrame:
    """Return the rows of the post hoc stale blend: the stage's rows that hold UKV-CEDA's next day.

    The stale blend reads UKV-CEDA's day `N + 1`, the 03 UTC run one day before ENS's run, so it
    needs those columns on top of everything the planned arms need. The folds are the stage's, so
    the planned arms score the same folds.

    Args:
        stage: The technology and ENS lead day `N`, which must be in `STALE_DAYS`.
        frame: The stage's rows from `stage_frame`.
        inputs: The technology's `<domain>_ukv_ceda_inputs.parquet`.

    Returns:
        The rows of `frame` where UKV-CEDA's day `N + 1` columns are present, with the shuffles of
        those columns under both seeds.

    Raises:
        ValueError: If a row reads the wrong run, an arm's column holds a missing value, or no row
            holds UKV-CEDA's day `N + 1`.
    """
    domain, day = stage
    keys = ["site", "time"]
    next_day = day + 1
    columns = ukv_columns(domain=domain, day=next_day)
    stamp = f"{PRODUCT}_day{next_day}_init_time"
    kept = frame.join(inputs.select(*keys, stamp, *columns), on=keys, how="left").filter(
        pl.all_horizontal(present(column=column) for column in columns)
    )
    if kept.is_empty():
        msg = f"{domain} day {day}: no row holds UKV-CEDA's day {next_day} columns"
        raise ValueError(msg)
    check_init_times(frame=kept, domain=domain, day=next_day)
    arms = stale_arms(day=day)
    shuffled = fit_aifs.add_shuffled_columns(
        frame=kept, domain=domain, shuffles=fit_aifs.control_shuffles(arms=arms)
    )
    check_no_missing(frame=shuffled, columns=fit_aifs.source_columns(arms=arms, domain=domain))
    return shuffled


def older_stage_frame(*, stage: Stage, frame: pl.DataFrame, inputs: pl.DataFrame) -> pl.DataFrame:
    """Return the rows of the post hoc older-run blend: the stage's rows that hold the older run.

    The older run is the 15 UTC run of the day before ENS's run (`build.OLDER_RUN`), so a row needs
    those columns on top of everything the planned arms need. The folds are the stage's, so the
    planned arms score the same folds.

    Args:
        stage: The technology and ENS lead day `N`, which must be in `OLDER_DAYS`.
        frame: The stage's rows from `stage_frame`.
        inputs: The technology's `<domain>_ukv_ceda_run15_inputs.parquet`.

    Returns:
        The rows of `frame` where the older run's day `N` columns are present, with the shuffle of
        those columns under the first seed.

    Raises:
        ValueError: If a candidate row has no row in the inputs, a row reads the wrong run, an
            arm's column holds a missing value, or no row holds the older run's columns.
    """
    domain, day = stage
    spec = build.OLDER_RUN
    keys = ["site", "time"]
    if frame.select(keys).join(inputs.select(keys), on=keys, how="anti").height:
        msg = f"{domain}: stage rows are missing from the older-run inputs"
        raise ValueError(msg)
    columns = ukv_columns(domain=domain, day=day, spec=spec)
    stamp = f"{spec.column_prefix}_day{day}_init_time"
    kept = frame.join(inputs.select(*keys, stamp, *columns), on=keys, how="left").filter(
        pl.all_horizontal(present(column=column) for column in columns)
    )
    if kept.is_empty():
        msg = f"{domain} day {day}: no row holds the older run's columns"
        raise ValueError(msg)
    check_init_times(frame=kept, domain=domain, day=day, spec=spec)
    arms = older_arms(day=day)
    shuffled = fit_aifs.add_shuffled_columns(
        frame=kept, domain=domain, shuffles=fit_aifs.control_shuffles(arms=arms)
    )
    check_no_missing(frame=shuffled, columns=fit_aifs.source_columns(arms=arms, domain=domain))
    return shuffled


def check_arm_columns(*, frame: pl.DataFrame, domain: DomainType, arms: Sequence[str]) -> None:
    """Raise unless every arm's columns are in the frame and hold the count the arm's kind promises.

    Args:
        frame: A stage's rows.
        domain: `solar` or `wind`.
        arms: The arms to be fitted.

    Raises:
        ValueError: Naming the arm and the absent columns or the two counts.
    """
    for arm in arms:
        features = fit_aifs.arm_features(arm=arm, domain=domain)
        absent = [name for name in features if name not in frame.columns]
        if absent:
            msg = f"{domain}: {arm} has no columns {absent} in the rows"
            raise ValueError(msg)
        expected = fit_aifs.expected_column_count(arm=arm, domain=domain)
        if len(features) != expected:
            msg = f"{domain}: {arm} has {len(features)} columns, its kind promises {expected}"
            raise ValueError(msg)


# --- The build gate and the stamp -----------------------------------------------------------------


def read_build_stamp(
    *, output_dir: Path, spec: build.RunSpec = build.PLANNED_RUN
) -> dict[str, object]:
    """Return `build.json`, after checking the build passed its coverage guard and is unchanged.

    Args:
        output_dir: The folder holding the build's outputs.
        spec: Which run the build read, which names its inputs files.

    Returns:
        The stamp.

    Raises:
        ValueError: If the build did not pass its coverage guard, or an inputs file's SHA-256 is not
            the one the build recorded.
    """
    stamp = json.loads((output_dir / build.STAMP_NAME).read_text())
    if stamp.get("coverage_guard_passed") is not True:
        msg = "the build did not pass its coverage guard, so no stage may run"
        raise ValueError(msg)
    if stamp.get("run_hour") != spec.run_hour:
        msg = f"the build read the {stamp.get('run_hour')} UTC run, not the {spec.run_hour} UTC run"
        raise ValueError(msg)
    for domain in fit_aifs.DOMAINS:
        actual = fit_aifs.sha256_of(
            path=output_dir / f"{domain}_{spec.column_prefix}_inputs.parquet"
        )
        if stamp["inputs_sha256"][domain] != actual:
            msg = f"{domain}: the inputs file is not the one build.json recorded"
            raise ValueError(msg)
    return stamp


def check_verified(*, output_dir: Path, stamp: Mapping[str, object]) -> None:
    """Raise unless `verify_ukv_ceda_inputs.py` passed on the inputs the build recorded.

    Args:
        output_dir: The folder holding the build's outputs and `verify.json`.
        stamp: The build's stamp, from `read_build_stamp`.

    Raises:
        ValueError: If `verify.json` is absent, did not pass, or names other inputs than the build.
    """
    path = output_dir / build.VERIFY_STAMP_NAME
    if not path.exists():
        msg = f"{path.name} is absent: run verify_ukv_ceda_inputs.py first"
        raise ValueError(msg)
    verified = json.loads(path.read_text())
    if verified.get("passed") is not True:
        msg = "verify_ukv_ceda_inputs.py did not pass, so no stage may run"
        raise ValueError(msg)
    if verified.get("inputs_sha256") != stamp["inputs_sha256"]:
        msg = "verify_ukv_ceda_inputs.py passed on other inputs than the build recorded"
        raise ValueError(msg)


def stage_stamp(
    *,
    published_dir: Path,
    day4_dir: Path,
    output_dir: Path,
    stage: Stage,
    arms: Sequence[str],
    offsets: Mapping[int, int],
    snapshot_id: object,
) -> dict[str, str]:
    """Return what a stage's saved losses must match.

    Args:
        published_dir: The folder holding the published inputs.
        day4_dir: The folder holding ENS's day-4 mean.
        output_dir: The folder holding the UKV-CEDA inputs.
        stage: The stage.
        arms: The arms the stamp's `columns` entry lists.
        offsets: The fold offsets of each era.
        snapshot_id: The Icechunk snapshot the inputs were built from.

    Returns:
        `fit_aifs.build_stamp`'s entries, plus the fold offsets and the snapshot.
    """
    return {
        **fit_aifs.build_stamp(
            published_dir=published_dir,
            aifs_dir=output_dir,
            domain=stage.domain,
            arms=arms,
            extra_dirs={"day4_shared": day4_dir},
            inputs_name=PRODUCT,
        ),
        "fold_offsets": json.dumps(dict(sorted(offsets.items()))),
        "icechunk_snapshot": str(snapshot_id),
    }


def check_saved_file(
    *,
    losses: pl.DataFrame,
    frame: pl.DataFrame,
    label: str,
    stamp_file: Path,
    stamp: dict[str, str],
    stale_frame: pl.DataFrame | None = None,
    older_frame: pl.DataFrame | None = None,
) -> None:
    """Raise unless a saved losses file comes from this build, this device, and exactly these rows.

    Args:
        losses: The saved losses.
        frame: The stage's rows.
        label: The stage, for messages.
        stamp_file: The stamp written beside the losses.
        stamp: The current build's stamp.
        stale_frame: The rows of the post hoc stale blend, which a stale arm must score instead of
            the stage's.
        older_frame: The rows of the post hoc older-run blend, which an older-run arm must score
            instead of the stage's.

    Raises:
        ValueError: If the stamp is missing or differs, an (arm, setting) of the losses scores
            other (site, time, fold) rows than its rows, or a stale or older-run arm is saved for
            a stage with no such rows.
    """
    if not stamp_file.exists() or json.loads(stamp_file.read_text()) != stamp:
        msg = f"{stamp_file} is missing or names another build or device"
        raise ValueError(msg)
    for (arm, setting), group in losses.group_by(["arm", "setting"]):
        rows, kind = frame, "stage"
        if is_stale_arm(arm=str(arm)):
            rows, kind = stale_frame, "stale"
        elif is_older_arm(arm=str(arm)):
            rows, kind = older_frame, "older-run"
        if rows is None:
            msg = f"{label}: {arm} is a {kind} arm, but the stage has no {kind} rows"
            raise ValueError(msg)
        want = rows.select(ROW_KEYS).unique().sort(ROW_KEYS)
        if not group.select(ROW_KEYS).unique().sort(ROW_KEYS).equals(want):
            msg = f"{label}: {arm} at {setting} scores other rows or folds than its rows"
            raise ValueError(msg)


def write_atomically(*, path: Path, frame: pl.DataFrame) -> None:
    """Write a parquet file to a temporary name and rename it, so a crash leaves no partial file."""
    temporary = path.with_name(path.name + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


def predictions_table(*, losses: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """Return each fit's capped out-of-fold prediction beside the measured power and the fold.

    Args:
        losses: A stage's per-row losses.
        frame: The stage's rows, carrying the target.

    Returns:
        `arm`, `setting`, `site`, `time`, `seed`, `fold`, `actual_mw`, and `prediction_capped_mw`.
    """
    return (
        predictions_from_losses(losses=losses, frame=frame)
        .join(
            losses.select("arm", "setting", "site", "time", "seed", "fold"),
            on=["arm", "setting", "site", "time", "seed"],
        )
        .join(
            frame.select("site", "time", actual_mw=pl.col(TARGET).cast(pl.Float64)),
            on=["site", "time"],
        )
        .select(
            "arm", "setting", "site", "time", "seed", "fold", "actual_mw", "prediction_capped_mw"
        )
    )


# --- Fitting --------------------------------------------------------------------------------------


class PlannedStage(NamedTuple):
    """A stage ready to fit: its rows, fold offsets, stamp, and its post hoc blends' rows.

    `stale_frame` is `None` at a lead day with no stale blend, and `older_frame` is `None` unless
    the older-run inputs were read and the lead day has an older-run blend. `older_stamp` names the
    older-run inputs file's SHA-256 and the Icechunk snapshot it was built from, and is `None`
    where `older_frame` is.
    """

    stage: Stage
    frame: pl.DataFrame
    offsets: dict[int, int]
    stamp: dict[str, str]
    stale_frame: pl.DataFrame | None = None
    older_frame: pl.DataFrame | None = None
    older_stamp: dict[str, str] | None = None


def plan_stages(
    *,
    published_dir: Path,
    day4_dir: Path,
    output_dir: Path,
    older_dir: Path | None = None,
    permutation: bool = False,
) -> list[PlannedStage]:
    """Build every stage's rows and check its arms, after the build gate.

    Args:
        published_dir: The folder holding the published inputs.
        day4_dir: The folder holding ENS's day-4 mean.
        output_dir: The folder holding the UKV-CEDA inputs, and where results are written.
        older_dir: The older-run build's folder, or `None` to leave the older-run blend out.
        permutation: Whether to add the permutation test's extra shuffles to the solar stages.

    Returns:
        The stages, solar before wind.
    """
    build_stamp = read_build_stamp(output_dir=output_dir)
    older_build_stamp = (
        None if older_dir is None else read_build_stamp(output_dir=older_dir, spec=build.OLDER_RUN)
    )
    planned: list[PlannedStage] = []
    for domain in fit_aifs.DOMAINS:
        candidates = build.published_rows(
            published_dir=published_dir, day4_dir=day4_dir, domain=domain
        )
        inputs = pl.read_parquet(output_dir / f"{domain}_ukv_ceda_inputs.parquet")
        older_inputs = (
            None
            if older_dir is None
            else pl.read_parquet(
                older_dir / f"{domain}_{build.OLDER_RUN.column_prefix}_inputs.parquet"
            )
        )
        for day in DAYS:
            stage = Stage(domain=domain, day=day)
            frame, offsets = stage_frame(
                stage=stage,
                candidates=candidates,
                inputs=inputs,
                permutation=permutation and domain == "solar",
            )
            check_arm_columns(frame=frame, domain=domain, arms=stage_arms(day=day))
            stamp = stage_stamp(
                published_dir=published_dir,
                day4_dir=day4_dir,
                output_dir=output_dir,
                stage=stage,
                arms=stage_arms(day=day),
                offsets=offsets,
                snapshot_id=build_stamp["snapshot_id"],
            )
            stale_frame = None
            if day in STALE_DAYS:
                stale_frame = stale_stage_frame(stage=stage, frame=frame, inputs=inputs)
                check_arm_columns(frame=stale_frame, domain=domain, arms=stale_arms(day=day))
            older_frame = None
            older_stamp = None
            if (
                older_inputs is not None
                and older_dir is not None
                and older_build_stamp is not None
                and day in OLDER_DAYS
            ):
                older_frame = older_stage_frame(stage=stage, frame=frame, inputs=older_inputs)
                check_arm_columns(frame=older_frame, domain=domain, arms=older_arms(day=day))
                older_stamp = {
                    "older_run_inputs_sha256": fit_aifs.sha256_of(
                        path=older_dir / f"{domain}_{build.OLDER_RUN.column_prefix}_inputs.parquet"
                    ),
                    "older_run_snapshot": str(older_build_stamp["snapshot_id"]),
                }
            planned.append(
                PlannedStage(
                    stage=stage,
                    frame=frame,
                    offsets=offsets,
                    stamp=stamp,
                    stale_frame=stale_frame,
                    older_frame=older_frame,
                    older_stamp=older_stamp,
                )
            )
    return planned


def post_hoc_jobs(*, stage: Stage, kind: PostHocKind) -> list[fit_aifs.Job]:
    """Return the (arm, setting) pairs of one post hoc analysis at a stage, saved or not."""
    if kind == "stale":
        return stale_jobs(day=stage.day)
    if kind == "permutation":
        return permutation_jobs(stage=stage)
    return older_jobs(day=stage.day)


def post_hoc_frame(*, planned: PlannedStage, kind: PostHocKind) -> pl.DataFrame | None:
    """Return the rows a post hoc analysis's arms are trained and scored on, or `None` if none."""
    if kind == "stale":
        return planned.stale_frame
    if kind == "permutation":
        return planned.frame
    return planned.older_frame


def jobs_to_fit(
    *, output_dir: Path, stage: Stage, only_missing: bool, post_hoc: PostHocKind | None = None
) -> tuple[str, list[fit_aifs.Job]]:
    """Return the group to write and the (arm, setting) pairs of a stage to fit.

    Args:
        output_dir: The write-once folder.
        stage: The stage.
        only_missing: Whether a stage that holds some but not all of its pairs may fit the rest.
        post_hoc: Which post hoc analysis's pairs to list instead of the planned pairs, which need
            the planned pairs saved first.

    Returns:
        The group name and the pairs: all of them for a stage with no saved file, the missing ones
        under `--only-missing`, and none for a complete stage. Under `post_hoc`, the pairs of that
        analysis no saved file holds.

    Raises:
        ValueError: If a stage holds some pairs but not all and `only_missing` is false, or if
            `post_hoc` is set before the stage's planned pairs are all saved.
    """
    saved = saved_pairs(output_dir=output_dir, stage=stage)
    planned_missing = [job for job in planned_jobs(day=stage.day) if job not in saved]
    if post_hoc is not None:
        if planned_missing:
            msg = (
                f"{stem(stage=stage, group='*')}: fit the planned pairs before the {post_hoc} ones"
            )
            raise ValueError(msg)
        return next_group(output_dir=output_dir, stage=stage), [
            job for job in post_hoc_jobs(stage=stage, kind=post_hoc) if job not in saved
        ]
    if saved and planned_missing and not only_missing:
        msg = (
            f"{stem(stage=stage, group='*')}: saved fits lack {planned_missing}; "
            "pass --only-missing"
        )
        raise ValueError(msg)
    return next_group(output_dir=output_dir, stage=stage), planned_missing


def file_stamp(*, planned: PlannedStage, arms: Sequence[str]) -> dict[str, str]:
    """Return what a saved losses file of these arms must match.

    The stage's stamp and the arms' columns, plus the older-run inputs' SHA-256 and snapshot if any
    arm is an older-run arm, plus the permutation seeds if any arm is a permutation control. A file
    of planned or stale arms therefore keeps the stamp it was written with.

    Args:
        planned: The stage.
        arms: The arms the file holds.

    Returns:
        The stamp.

    Raises:
        ValueError: If an older-run arm is listed for a stage with no older-run inputs.
    """
    stamp = {
        **planned.stamp,
        "columns": json.dumps(
            {
                arm: fit_aifs.arm_features(arm=arm, domain=planned.stage.domain)
                for arm in sorted(arms)
            }
        ),
    }
    if any(is_older_arm(arm=arm) for arm in arms):
        if planned.older_stamp is None:
            msg = f"{stem(stage=planned.stage, group='*')}: no older-run inputs were read"
            raise ValueError(msg)
        stamp.update(planned.older_stamp)
    if any(is_permutation_arm(arm=arm) for arm in arms):
        stamp["permutation_seeds"] = json.dumps(PERMUTATION_SEEDS)
    return stamp


def fit_stage(
    *,
    planned: PlannedStage,
    output_dir: Path,
    workers: int,
    only_missing: bool,
    post_hoc: PostHocKind | None = None,
) -> pl.DataFrame:
    """Fit a stage's missing pairs and write them once, then return all its saved losses.

    Args:
        planned: The stage.
        output_dir: The write-once folder.
        workers: How many (arm, site) fits run at once.
        only_missing: Whether to fit the pairs a partly saved stage lacks.
        post_hoc: Which post hoc analysis's missing pairs to fit, on that analysis's rows.

    Returns:
        Every saved loss of the stage at both settings.
    """
    stage = planned.stage
    group, jobs = jobs_to_fit(
        output_dir=output_dir, stage=stage, only_missing=only_missing, post_hoc=post_hoc
    )
    if jobs:
        fit_frame = (
            planned.frame if post_hoc is None else post_hoc_frame(planned=planned, kind=post_hoc)
        )
        if fit_frame is None:
            msg = f"{stem(stage=stage, group='*')}: the stage has no {post_hoc} rows to fit"
            raise ValueError(msg)
        file = output_dir / f"{stem(stage=stage, group=group)}_losses.parquet"
        predictions = file.with_name(file.name.replace("_losses", "_predictions"))
        refuse_to_overwrite(paths=[file, predictions])
        arms = list(dict.fromkeys(arm for arm, _ in jobs))
        stamp = file_stamp(planned=planned, arms=arms)
        file.with_suffix(".json").write_text(json.dumps(stamp))
        losses = fit_aifs.fit_jobs(
            frame=fit_frame, domain=stage.domain, jobs=jobs, workers=workers
        ).with_columns(device=pl.lit(fit_aifs.DEVICE))
        write_atomically(path=file, frame=losses)
        write_atomically(
            path=predictions, frame=predictions_table(losses=losses, frame=planned.frame)
        )
    return verified_losses(planned=planned, output_dir=output_dir)


def verified_losses(*, planned: PlannedStage, output_dir: Path) -> pl.DataFrame:
    """Return a stage's saved losses after checking each file's stamp and rows.

    Args:
        planned: The stage.
        output_dir: The folder holding the saved files.

    Returns:
        Every saved loss of the stage at both settings.
    """
    stage = planned.stage
    for path in group_files(output_dir=output_dir, stage=stage):
        losses = pl.read_parquet(path)
        arms = sorted(set(losses["arm"].unique().to_list()))
        check_saved_file(
            losses=losses,
            frame=planned.frame,
            label=path.name,
            stamp_file=path.with_suffix(".json"),
            stamp=file_stamp(planned=planned, arms=arms),
            stale_frame=planned.stale_frame,
            older_frame=planned.older_frame,
        )
    return saved_losses(output_dir=output_dir, stage=stage)


def cpu_noise_floor(*, planned: PlannedStage, output_dir: Path, gpu: pl.DataFrame) -> pl.DataFrame:
    """Refit the wind day-1 blend at one site on the CPU, write it once, and return it.

    Args:
        planned: The wind day-1 stage.
        output_dir: The write-once folder.
        gpu: The stage's saved losses, holding the GPU fit.

    Returns:
        The CPU losses, labelled with the arm, setting, and `cpu` device. A rerun reads the saved
        file after checking its stamp.
    """
    arm = arm_name(day=1, role="")
    file = output_dir / f"{stem(stage=planned.stage, group=CPU_GROUP)}_losses.parquet"
    stamp = {
        **planned.stamp,
        "device": "cpu",
        "columns": json.dumps({arm: fit_aifs.arm_features(arm=arm, domain="wind")}),
    }
    if file.exists():
        if json.loads(file.with_suffix(".json").read_text()) != stamp:
            msg = f"{file.with_suffix('.json')} names another build"
            raise ValueError(msg)
        return pl.read_parquet(file)
    refuse_to_overwrite(paths=[file])
    file.with_suffix(".json").write_text(json.dumps(stamp))
    losses = out_of_fold_losses(
        site_rows=planned.frame.filter(pl.col("site") == CPU_SITE),
        features=list(fit_aifs.arm_features(arm=arm, domain="wind")),
        target=TARGET,
        hyper_parameters=SETTINGS[PRIMARY],
        with_quantiles=False,
        device="cpu",
    ).with_columns(arm=pl.lit(arm), setting=pl.lit(PRIMARY), device=pl.lit("cpu"))
    write_atomically(path=file, frame=losses)
    return losses


def noise_floor_line(*, cpu: pl.DataFrame, gpu: pl.DataFrame) -> str:
    """Describe how far the CPU refit's per-row error is from the GPU fit's, in points of capacity.

    Args:
        cpu: `cpu_noise_floor`'s result.
        gpu: The stage's saved losses.

    Returns:
        A sentence giving the mean absolute and the largest per-row difference, and both fits' mean
        error.
    """
    arm = cpu["arm"][0]
    reference = gpu.filter(
        pl.col("arm") == arm, pl.col("setting") == PRIMARY, pl.col("site") == CPU_SITE
    )
    joined = cpu.join(reference, on=["site", "time", "seed"], suffix="_gpu").with_columns(
        gap=(pl.col(METRIC) - pl.col(f"{METRIC}_gpu"))
    )
    gap = np.abs(joined["gap"].to_numpy()) * PERCENTAGE_POINTS
    cpu_error = joined[METRIC].to_numpy().mean() * PERCENTAGE_POINTS
    gpu_error = joined[f"{METRIC}_gpu"].to_numpy().mean() * PERCENTAGE_POINTS
    by_seed = reference.pivot(on="seed", index=["site", "time"], values=METRIC).drop("site", "time")
    seeds = by_seed.to_numpy() * PERCENTAGE_POINTS
    seed_gaps = np.concatenate(
        [
            np.abs(seeds[:, first] - seeds[:, second])
            for first, second in combinations(range(seeds.shape[1]), 2)
        ]
    )
    seed_means = seeds.mean(axis=0)
    return (
        f"The CPU refit of `{arm}` at wind site {CPU_SITE} (primary setting) differs from the GPU "
        f"fit by {gap.mean():.4f} points of capacity per row on average and {gap.max():.4f} at "
        f"most; the two fits' mean errors are {cpu_error:.4f} (CPU) and {gpu_error:.4f} (GPU). "
        f"For comparison, two fitting seeds of the same GPU blend at that site differ by "
        f"{seed_gaps.mean():.4f} points of capacity per row on average and {seed_gaps.max():.4f} "
        f"at most, and the {seeds.shape[1]} seeds' mean errors span "
        f"{np.ptp(seed_means):.4f} points ({seed_means.min():.4f} to {seed_means.max():.4f})."
    )


# --- Reading rule ---------------------------------------------------------------------------------


def setting_verdict(
    *, day: int, p1: BootstrapInterval, p2: Sequence[BootstrapInterval]
) -> dict[str, object]:
    """Return the verdict at one setting.

    The blend lowers the error only if P1 and every P2 contrast (both shuffle seeds) have an upper
    95% bound below zero, so one noisy shuffle cannot decide it. Where P1 alone is below zero, the
    verdict is `UNRESOLVED_LOWER`, because the blend's error is statistically significantly lower
    than padded ENS's and the control test is not passed.

    Args:
        day: The lead day, for the label.
        p1: The blend minus the padded ENS arm.
        p2: The blend minus each control, one per shuffle seed.

    Returns:
        `verdict`: `lowers the error at day N`, `UNRESOLVED_LOWER`, `raises the error at day N`
        where P1's lower bound is above zero, or `no detectable difference`; and
        `largest_gain_not_excluded`, the gain P1's lower bound leaves open, or `None` where the
        verdict is not `no detectable difference`.
    """
    if p1["upper_95"] < 0.0:
        if all(interval["upper_95"] < 0.0 for interval in p2):
            return {"verdict": f"lowers the error at day {day}", "largest_gain_not_excluded": None}
        return {"verdict": UNRESOLVED_LOWER, "largest_gain_not_excluded": None}
    if p1["lower_95"] > 0.0:
        return {"verdict": f"raises the error at day {day}", "largest_gain_not_excluded": None}
    return {
        "verdict": NO_DETECTABLE_DIFFERENCE,
        "largest_gain_not_excluded": max(0.0, -p1["lower_95"]),
    }


def reading(
    *,
    day: int,
    p1: Mapping[str, BootstrapInterval],
    p2: Mapping[str, Sequence[BootstrapInterval]],
) -> str:
    """Return the verdict that both settings give, or the weaker reading if they differ.

    Args:
        day: The lead day.
        p1: P1 at each setting.
        p2: The P2 contrasts of both shuffle seeds at each setting.

    Returns:
        The verdict, from `studies.bootstrap.combine_setting_verdicts`. Where the settings give
        different verdicts, `UNRESOLVED_LOWER` if P1 is below zero at both, else `no detectable
        difference`.
    """
    verdicts = {
        setting: str(setting_verdict(day=day, p1=p1[setting], p2=p2[setting])["verdict"])
        for setting in (PRIMARY, SENSITIVITY)
    }
    p1_below_at_both = all(p1[setting]["upper_95"] < 0.0 for setting in (PRIMARY, SENSITIVITY))
    return combine_setting_verdicts(
        primary=verdicts[PRIMARY],
        sensitivity=verdicts[SENSITIVITY],
        unresolved=UNRESOLVED_LOWER if p1_below_at_both else NO_DETECTABLE_DIFFERENCE,
    )


def survives_bonferroni(*, wide: Mapping[str, tuple[float, float]]) -> bool:
    """Return whether P1's Bonferroni-corrected interval is below zero at both settings.

    Args:
        wide: P1's lower and upper bound at the corrected level, at each setting.

    Returns:
        True only if the upper bound is below zero at the primary and the sensitivity setting.
    """
    return all(wide[setting][1] < 0.0 for setting in (PRIMARY, SENSITIVITY))


def left_open_text(*, p1: Mapping[str, BootstrapInterval]) -> str:
    """State, at each setting, the largest gain P1's interval does not exclude.

    Args:
        p1: P1 at each setting.

    Returns:
        `primary X, sensitivity Y` in points of capacity, with `below zero` for a setting where
        P1's whole interval is below zero.
    """
    parts = []
    for setting in (PRIMARY, SENSITIVITY):
        interval = p1[setting]
        if interval["upper_95"] < 0.0:
            parts.append(f"{setting} below zero")
        else:
            parts.append(f"{setting} {max(0.0, -interval['lower_95']) * PERCENTAGE_POINTS:.3f}")
    return ", ".join(parts)


def null_reading(*, interval: BootstrapInterval) -> str:
    """Say what a contrast's interval leaves open, in points of capacity.

    Args:
        interval: A contrast, treatment minus reference, so a negative difference is a gain.

    Returns:
        A sentence naming the largest gain the interval does not exclude, or saying that the
        interval is entirely on one side of zero.
    """
    scale = PERCENTAGE_POINTS
    if interval["upper_95"] < 0.0:
        return "the interval is below zero: the blend's error is lower"
    if interval["lower_95"] > 0.0:
        return "the interval is above zero: the blend's error is higher"
    gain = max(0.0, -interval["lower_95"]) * scale
    return (
        f"an effect as large as {gain:.3f} points of capacity is not excluded, and an effect "
        f"larger than {gain:.3f} points is excluded"
    )


# --- Report ---------------------------------------------------------------------------------------


def record(
    *,
    stage: Stage,
    setting: str,
    contrast: str,
    scope: str,
    interval: BootstrapInterval,
    level: float = 95.0,
) -> IntervalRecord:
    """Return one printed interval as a row of `intervals.parquet`."""
    return {
        "domain": stage.domain,
        "day": stage.day,
        "setting": setting,
        "contrast": contrast,
        "scope": scope,
        "level": level,
        "difference": interval["difference"],
        "lower": interval["lower_95"],
        "upper": interval["upper_95"],
        "n_rows": interval["n_rows"],
        "n_months": interval["n_months"],
    }


def interval_cell(*, interval: BootstrapInterval) -> str:
    """Format an interval as `difference [lower, upper]` in points of capacity."""
    return interval_text(
        point=interval["difference"], lower=interval["lower_95"], upper=interval["upper_95"]
    )


def at_setting(*, losses: pl.DataFrame, setting: str) -> pl.DataFrame:
    """Return the losses of one hyperparameter setting."""
    return losses.filter(pl.col("setting") == setting)


def era_losses(*, losses: pl.DataFrame, frame: pl.DataFrame, eras: Sequence[int]) -> pl.DataFrame:
    """Return the losses on the rows whose era is one of `eras`."""
    rows_in_eras = frame.filter(pl.col("era_code").is_in(list(eras))).select("site", "time")
    return losses.join(rows_in_eras, on=["site", "time"])


def scoped_line(
    *,
    losses: pl.DataFrame,
    treatment: str,
    reference: str,
    label: str,
    stage: Stage,
    records: list[IntervalRecord],
) -> str:
    """Format P1 on a subset of rows as a table row, with an interval only for six months or more.

    Args:
        losses: The subset's losses at the primary setting.
        treatment: The blend.
        reference: The padded ENS arm.
        label: The subset's name.
        stage: The stage.
        records: Where an interval is appended for `intervals.parquet`.

    Returns:
        `| label | difference [interval] | rows | months |`.
    """
    if losses.is_empty():
        return f"| {label} | no rows | 0 | 0 |"
    differences, months = paired_differences(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    n_months = len(np.unique(months))
    if n_months >= MIN_MONTHS_FOR_INTERVAL:
        interval = difference(losses=losses, treatment=treatment, reference=reference)
        records.append(
            record(stage=stage, setting=PRIMARY, contrast="p1", scope=label, interval=interval)
        )
        text = interval_cell(interval=interval)
    else:
        text = (
            f"{float(differences.mean()) * PERCENTAGE_POINTS:+.3f} (no interval: {n_months} months)"
        )
    return f"| {label} | {text} | {differences.shape[1]} | {n_months} |"


GENERATOR_ERROR_SCOPE: Final[str] = "generator error"
"""The scope prefix of a per-generator absolute error in `intervals.parquet`, followed by
`<site>: <arm>`. The row carries the mean error, with no interval."""


def generator_error_lines(
    *,
    losses: pl.DataFrame,
    arms: Sequence[str],
    stage: Stage,
    records: list[IntervalRecord],
) -> list[str]:
    """Format each arm's mean absolute error at each generator alone, with no interval.

    Args:
        losses: Per-row losses at one setting, holding every arm of `arms`.
        arms: The arms to show, as columns.
        stage: The stage, whose technology and lead day label each record.
        records: Where each mean error is appended for `intervals.parquet`, at the primary setting.

    Returns:
        A Markdown table of percent of capacity, one row per generator in label order.
    """
    means = (
        losses.filter(pl.col("arm").is_in(list(arms)))
        .group_by("site", "arm")
        .agg(
            error=pl.col(METRIC).mean() * PERCENTAGE_POINTS,
            n_rows=pl.col("time").n_unique(),
            n_months=pl.col("time").dt.strftime("%Y-%m").n_unique(),
        )
    )
    records.extend(
        {
            "domain": stage.domain,
            "day": stage.day,
            "setting": PRIMARY,
            "contrast": "generator_error",
            "scope": f"{GENERATOR_ERROR_SCOPE} {row['site']}: {row['arm']}",
            "level": float("nan"),
            "difference": row["error"] / PERCENTAGE_POINTS,
            "lower": float("nan"),
            "upper": float("nan"),
            "n_rows": row["n_rows"],
            "n_months": row["n_months"],
        }
        for row in means.sort("site", "arm").iter_rows(named=True)
    )
    table = means.pivot(on="arm", index="site", values="error").sort("site")
    lines = [
        "Mean absolute error at each generator alone (primary setting, % of capacity):",
        "",
        "| Generator | " + " | ".join(f"`{arm}`" for arm in arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    lines += [
        f"| {row['site']} | " + " | ".join(f"{row[arm]:.3f}" for arm in arms) + " |"
        for row in table.iter_rows(named=True)
    ]
    return lines


class StageReading(NamedTuple):
    """What a stage's section concludes, for the readings table."""

    reading: str
    survives_bonferroni: bool
    left_open: str


class MonthDrop(NamedTuple):
    """A contrast recomputed with one calendar month left out."""

    month: str
    interval: BootstrapInterval


def month_drops(*, losses: pl.DataFrame, treatment: str, reference: str) -> list[MonthDrop]:
    """Return a contrast's interval with each calendar month dropped in turn.

    Args:
        losses: Per-row losses at one setting, carrying `month` and both arms.
        treatment: The arm whose error is compared.
        reference: The arm it is compared against.

    Returns:
        One `MonthDrop` per month, in month order. Nothing is refitted: the saved losses of the
        remaining months are resampled again.
    """
    return [
        MonthDrop(
            month=str(month),
            interval=difference(
                losses=losses.filter(pl.col("month") != month),
                treatment=treatment,
                reference=reference,
            ),
        )
        for month in sorted(losses["month"].unique().to_list())
    ]


def leave_one_month_out_lines(
    *,
    per_setting: Mapping[str, pl.DataFrame],
    full: Mapping[str, BootstrapInterval],
    treatment: str,
    reference: str,
    stage: Stage,
    records: list[IntervalRecord],
) -> list[str]:
    """Format P1 with each month dropped in turn, at both settings, as an exploratory table.

    Args:
        per_setting: The stage's losses at each setting.
        full: P1 over all months at each setting.
        treatment: The blend.
        reference: The padded ENS arm.
        stage: The stage.
        records: Where the most influential drop's interval is appended for `intervals.parquet`.

    Returns:
        A Markdown table, one row per setting, giving the lowest and highest point estimate over
        the drops with the month dropped, the drop that moves the estimate most, and the highest
        95% upper bound over the drops.
    """
    lines = [
        (
            "Post hoc, exploratory: P1 with each calendar month dropped in turn (nothing is "
            "refitted). A drop whose upper bound reaches zero shows that one month carries the "
            "result."
        ),
        "",
        (
            "| Setting | All months | Lowest drop (month dropped) | Highest drop (month dropped) "
            "| Month that moves it most | That drop's difference [interval] "
            "| Highest upper bound over the drops (month dropped) |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for setting in SETTINGS:
        drops = month_drops(losses=per_setting[setting], treatment=treatment, reference=reference)
        point = full[setting]["difference"]
        lowest = min(drops, key=lambda drop: drop.interval["difference"])
        highest = max(drops, key=lambda drop: drop.interval["difference"])
        moves_most = max(drops, key=lambda drop: abs(drop.interval["difference"] - point))
        loosest = max(drops, key=lambda drop: drop.interval["upper_95"])
        records.append(
            record(
                stage=stage,
                setting=setting,
                contrast="p1",
                scope="E6 most influential month dropped",
                interval=moves_most.interval,
            )
        )
        lines.append(
            f"| {setting} | {point * PERCENTAGE_POINTS:+.3f} "
            f"| {lowest.interval['difference'] * PERCENTAGE_POINTS:+.3f} ({lowest.month}) "
            f"| {highest.interval['difference'] * PERCENTAGE_POINTS:+.3f} ({highest.month}) "
            f"| {moves_most.month} | {interval_cell(interval=moves_most.interval)} "
            f"| {loosest.interval['upper_95'] * PERCENTAGE_POINTS:+.3f} ({loosest.month}) |"
        )
    lines.append("")
    return lines


def control_gap_lines(
    *,
    per_setting: Mapping[str, pl.DataFrame],
    control: str,
    control_b: str,
    stage: Stage,
    records: list[IntervalRecord],
) -> list[str]:
    """Format the gap between the two shuffled controls at both settings.

    The two controls carry the same (no) information and differ only in their shuffle seed, so a
    gap between them is the size of difference the pipeline produces from nothing.

    Args:
        per_setting: The stage's losses at each setting.
        control: The first-seed control.
        control_b: The second-seed control.
        stage: The stage.
        records: Where each interval is appended for `intervals.parquet`.

    Returns:
        A Markdown table, one row per setting.
    """
    lines = [
        (
            "Exploratory: the gap between the two shuffled controls, seed 0 minus seed 1000, "
            "which differ only in their shuffle seed."
        ),
        "",
        "| Setting | Difference [interval] (points) | Statistically significant at the 5% level |",
        "|---|---|---|",
    ]
    for setting in SETTINGS:
        interval = difference(losses=per_setting[setting], treatment=control, reference=control_b)
        records.append(
            record(
                stage=stage,
                setting=setting,
                contrast="control_gap",
                scope="control gap",
                interval=interval,
            )
        )
        significant = interval["upper_95"] < 0.0 or interval["lower_95"] > 0.0
        lines.append(
            f"| {setting} | {interval_cell(interval=interval)} | {'yes' if significant else 'no'} |"
        )
    lines.append("")
    return lines


def settings_text(*, settings: Sequence[str]) -> str:
    """Name one or both hyperparameter settings: `the primary setting` or `both settings`."""
    return "both settings" if len(settings) == len(SETTINGS) else f"the {settings[0]} setting"


def stage_lines(
    *, planned: PlannedStage, losses: pl.DataFrame, records: list[IntervalRecord]
) -> tuple[list[str], StageReading]:
    """Format one stage's section of the report.

    Args:
        planned: The stage and its rows.
        losses: The stage's saved losses at both settings.
        records: Where every printed interval is appended for `intervals.parquet`.

    Returns:
        The section's Markdown lines and what the stage concludes.
    """
    stage, frame = planned.stage, planned.frame
    pad, blend, control, control_b = stage_arms(day=stage.day)
    per_setting = {setting: at_setting(losses=losses, setting=setting) for setting in SETTINGS}
    p1 = {s: difference(losses=per_setting[s], treatment=blend, reference=pad) for s in SETTINGS}
    p2 = {
        s: [
            difference(losses=per_setting[s], treatment=blend, reference=c)
            for c in (control, control_b)
        ]
        for s in SETTINGS
    }
    wide = {
        s: bootstrap_difference_at_level(
            losses=per_setting[s],
            treatment=blend,
            reference=pad,
            metric=METRIC,
            level=BONFERRONI_LEVEL,
        )
        for s in SETTINGS
    }
    for setting in SETTINGS:
        records.append(
            record(
                stage=stage, setting=setting, contrast="p1", scope="all rows", interval=p1[setting]
            )
        )
        records.append(
            record(
                stage=stage,
                setting=setting,
                contrast="p1",
                scope="Bonferroni",
                interval={
                    **p1[setting],
                    "lower_95": wide[setting][0],
                    "upper_95": wide[setting][1],
                },
                level=BONFERRONI_LEVEL,
            )
        )
        for label, interval in zip(("p2", "p2b"), p2[setting], strict=True):
            records.append(
                record(
                    stage=stage,
                    setting=setting,
                    contrast=label,
                    scope="all rows",
                    interval=interval,
                )
            )
    verdict = reading(day=stage.day, p1=p1, p2=p2)
    corrected = survives_bonferroni(wide=wide)
    level = f"Bonferroni {BONFERRONI_LEVEL}%"
    lines = [
        f"### {stage.domain.capitalize()}, lead day {stage.day}",
        "",
        (
            f"{frame.height} rows over {frame['month'].n_unique()} months at "
            f"{frame['site'].n_unique()} generators; fold offsets by era "
            f"{json.dumps(dict(sorted(planned.offsets.items())))}."
        ),
        "",
        (
            f"| Contrast (points) | Primary | Sensitivity | Primary, {level} "
            f"| Sensitivity, {level} |"
        ),
        "|---|---|---|---|---|",
        (
            f"| P1: {SPECS['p1']} | {interval_cell(interval=p1[PRIMARY])} "
            f"| {interval_cell(interval=p1[SENSITIVITY])} "
            f"| {wide[PRIMARY][0] * PERCENTAGE_POINTS:+.3f}, "
            f"{wide[PRIMARY][1] * PERCENTAGE_POINTS:+.3f} "
            f"| {wide[SENSITIVITY][0] * PERCENTAGE_POINTS:+.3f}, "
            f"{wide[SENSITIVITY][1] * PERCENTAGE_POINTS:+.3f} |"
        ),
        (
            f"| P2: {SPECS['p2']} | {interval_cell(interval=p2[PRIMARY][0])} "
            f"| {interval_cell(interval=p2[SENSITIVITY][0])} | not adjusted | not adjusted |"
        ),
        (
            f"| P2: {SPECS['p2b']} | {interval_cell(interval=p2[PRIMARY][1])} "
            f"| {interval_cell(interval=p2[SENSITIVITY][1])} | not adjusted | not adjusted |"
        ),
        "",
        f"Reading: {verdict}.",
        "",
        f"P1 survives the Bonferroni correction at both settings: {'yes' if corrected else 'no'}.",
        "",
        (
            f"P1 at the primary setting: {null_reading(interval=p1[PRIMARY])}. "
            f"At the sensitivity setting: {null_reading(interval=p1[SENSITIVITY])}."
        ),
        "",
    ]
    near_by_contrast = {
        "P1": [fit_aifs.near_line(interval=p1[s]) for s in SETTINGS],
        "P2 (first-seed control)": [fit_aifs.near_line(interval=p2[s][0]) for s in SETTINGS],
        "P2 (second-seed control)": [fit_aifs.near_line(interval=p2[s][1]) for s in SETTINGS],
    }
    for name, flags in near_by_contrast.items():
        near = [s for s, flag in zip(SETTINGS, flags, strict=True) if flag]
        if near:
            lines += [f"{name} is near the 5% line at {settings_text(settings=near)}.", ""]
    signs = {s: np.sign(p1[s]["difference"]) for s in SETTINGS}
    if signs[PRIMARY] != signs[SENSITIVITY]:
        lines += ["P1 changes sign between the two settings.", ""]
    lines += [
        "| Arm | Setting | Error (% of capacity) | 95% interval | Rows | Months |",
        "|---|---|---|---|---|---|",
    ]
    for setting in SETTINGS:
        board = leaderboard(losses=per_setting[setting], arms=list(stage_arms(day=stage.day)))
        for row in board.iter_rows(named=True):
            records.append(
                {
                    "domain": stage.domain,
                    "day": stage.day,
                    "setting": setting,
                    "contrast": "error",
                    "scope": row["arm"],
                    "level": 95.0,
                    "difference": row["value"],
                    "lower": row["lower_95"],
                    "upper": row["upper_95"],
                    "n_rows": row["n_rows"],
                    "n_months": row["n_months"],
                }
            )
            text = error_text(value=row["value"], lower=row["lower_95"], upper=row["upper_95"])
            value, interval_part = text.split(" [")
            lines.append(
                f"| `{row['arm']}` | {setting} | {value} | [{interval_part} | {row['n_rows']} "
                f"| {row['n_months']} |"
            )
    lines += [
        "",
        *generator_error_lines(
            losses=per_setting[PRIMARY],
            arms=stage_arms(day=stage.day),
            stage=stage,
            records=records,
        ),
        "",
        *control_gap_lines(
            per_setting=per_setting,
            control=control,
            control_b=control_b,
            stage=stage,
            records=records,
        ),
        *leave_one_month_out_lines(
            per_setting=per_setting,
            full=p1,
            treatment=blend,
            reference=pad,
            stage=stage,
            records=records,
        ),
    ]
    lines += [
        (
            "Exploratory: P1 by era (primary setting). Era 0 is before 2025-10, era 1 is 2025-10 "
            "to 2025-12, and era 2 starts in 2026-02, after the UKV upgrade settled."
        ),
        "",
        "| Subset | Difference [interval] (points) | Rows | Months |",
        "|---|---|---|---|",
    ]
    primary = per_setting[PRIMARY]
    eras: dict[str, Sequence[int]] = {
        "era 0": (0,),
        "era 1": (1,),
        "era 2": (2,),
        "before the upgrade (eras 0 and 1)": BEFORE_UPGRADE_ERAS,
        "after the upgrade (era 2)": AFTER_UPGRADE_ERAS,
    }
    lines += [
        scoped_line(
            losses=era_losses(losses=primary, frame=frame, eras=members),
            treatment=blend,
            reference=pad,
            label=f"E4 {name}",
            stage=stage,
            records=records,
        )
        for name, members in eras.items()
    ]
    lines += [
        "",
        "Exploratory: P1 by generator (primary setting).",
        "",
        "| Subset | Difference [interval] (points) | Rows | Months |",
        "|---|---|---|---|",
    ]
    lines += [
        scoped_line(
            losses=primary.filter(pl.col("site") == site),
            treatment=blend,
            reference=pad,
            label=f"E5 generator {site}",
            stage=stage,
            records=records,
        )
        for site in sorted(frame["site"].unique().to_list())
    ]
    lines.append("")
    return lines, StageReading(
        reading=verdict, survives_bonferroni=corrected, left_open=left_open_text(p1=p1)
    )


STALE_CONTRASTS: Final[dict[str, str]] = {
    "stale_p1": "Stale blend minus padded ENS trained on the stale rows (P1, stale)",
    "stale_p2": "Stale blend minus its shuffled control (P2, stale; first seed)",
    "stale_p2b": "Stale blend minus its second-seed shuffled control (P2b, stale)",
    "stale_vs_fresh": "Stale blend minus the planned blend (UKV-CEDA run 21 hours staler)",
    "fresh_p1_same_rows": "Planned blend minus padded ENS, on the stale blend's rows",
    "training_rows": (
        "Tilt: padded ENS trained on all rows minus padded ENS trained on the stale rows"
    ),
}
"""The post hoc stale section's contrasts, by their code in `intervals.parquet`."""


def stale_lines(
    *, planned: PlannedStage, losses: pl.DataFrame, records: list[IntervalRecord]
) -> tuple[list[str], str | None]:
    """Format the post hoc stale-blend section of one stage, if its fits are saved.

    The stale blend is ENS day N plus UKV-CEDA day N + 1, so UKV-CEDA's run is 21 hours staler than
    ENS's, where the planned blend's is 3 hours fresher. If it still lowers the error, the planned
    gain is not explained by UKV-CEDA's later run alone.

    Args:
        planned: The stage and its rows.
        losses: The stage's saved losses at both settings.
        records: Where every printed interval is appended for `intervals.parquet`.

    Returns:
        The section's Markdown lines, and the reading of the one-control rule. Both are empty or
        `None` where the stage has no complete saved stale fit.
    """
    stage, rows = planned.stage, planned.stale_frame
    saved = set(losses.select("arm", "setting").unique().iter_rows())
    if rows is None or not set(stale_jobs(day=stage.day)) <= saved:
        return [], None
    pad, fresh, _, _ = stage_arms(day=stage.day)
    stale_pad, stale, stale_control, stale_control_b = stale_arms(day=stage.day)
    on_rows = {
        setting: at_setting(losses=losses, setting=setting).join(
            rows.select("site", "time"), on=["site", "time"]
        )
        for setting in SETTINGS
    }
    pairs = {
        "stale_p1": (stale, stale_pad),
        "stale_p2": (stale, stale_control),
        "stale_p2b": (stale, stale_control_b),
        "stale_vs_fresh": (stale, fresh),
        "fresh_p1_same_rows": (fresh, pad),
        "training_rows": (pad, stale_pad),
    }
    intervals = {
        code: {
            setting: difference(losses=on_rows[setting], treatment=treatment, reference=reference)
            for setting in SETTINGS
        }
        for code, (treatment, reference) in pairs.items()
    }
    for code, by_setting in intervals.items():
        records.extend(
            record(
                stage=stage, setting=setting, contrast=code, scope=STALE_SCOPE, interval=interval
            )
            for setting, interval in by_setting.items()
        )
    verdict = reading(
        day=stage.day,
        p1=intervals["stale_p1"],
        p2={
            setting: [intervals["stale_p2"][setting], intervals["stale_p2b"][setting]]
            for setting in SETTINGS
        },
    )
    lines = [
        f"#### Post hoc: ENS day {stage.day} plus UKV-CEDA day {stage.day + 1}, {stage.domain}",
        "",
        (
            f"Post hoc and exploratory, added after the first science review. UKV-CEDA's run is "
            f"21 hours staler than ENS's, so the 3-hour advantage of the planned blend is "
            f"reversed. {rows.height} of the stage's {planned.frame.height} rows hold UKV-CEDA's "
            f"day {stage.day + 1}. The stale blend, its controls, and the stale-rows padded ENS "
            f"reference were fitted on those {rows.height} rows. The planned blend and the "
            f"planned padded ENS reference were fitted on all {planned.frame.height} rows, and "
            f"every contrast below scores the same {rows.height} rows. The last contrast is the "
            f"measured effect of the planned reference's extra training rows, and bounds the tilt "
            f"in the stale-versus-planned contrast. {TIME_RESOLUTION_CAVEAT}"
        ),
        "",
        "| Contrast (points) | Primary | Sensitivity |",
        "|---|---|---|",
        *(
            f"| {label} | {interval_cell(interval=intervals[code][PRIMARY])} "
            f"| {interval_cell(interval=intervals[code][SENSITIVITY])} |"
            for code, label in STALE_CONTRASTS.items()
        ),
        "",
        (
            f"Post hoc reading, both control seeds: {verdict}. P1 (stale) at the primary setting: "
            f"{null_reading(interval=intervals['stale_p1'][PRIMARY])}. At the sensitivity "
            f"setting: {null_reading(interval=intervals['stale_p1'][SENSITIVITY])}."
        ),
        "",
        "| Arm | Setting | Error (% of capacity) | 95% interval | Rows | Months |",
        "|---|---|---|---|---|---|",
    ]
    for setting in SETTINGS:
        board = leaderboard(
            losses=on_rows[setting],
            arms=[pad, fresh, stale_pad, stale, stale_control, stale_control_b],
        )
        for row in board.iter_rows(named=True):
            records.append(
                {
                    "domain": stage.domain,
                    "day": stage.day,
                    "setting": setting,
                    "contrast": "error",
                    "scope": f"{STALE_SCOPE} rows: {row['arm']}",
                    "level": 95.0,
                    "difference": row["value"],
                    "lower": row["lower_95"],
                    "upper": row["upper_95"],
                    "n_rows": row["n_rows"],
                    "n_months": row["n_months"],
                }
            )
            text = error_text(value=row["value"], lower=row["lower_95"], upper=row["upper_95"])
            value, interval_part = text.split(" [")
            lines.append(
                f"| `{row['arm']}` | {setting} | {value} | [{interval_part} | {row['n_rows']} "
                f"| {row['n_months']} |"
            )
    lines.append("")
    return lines, verdict


class PermutationSummary(NamedTuple):
    """Where the planned blend's P1 falls among the shuffled controls' differences from padded ENS.

    Every value is in the loss column's own units (a share of capacity), and a negative value is a
    lower error than padded ENS's.
    """

    planned: float
    draws: tuple[float, ...]
    rank: int
    p_value: float


def permutation_summary(*, planned: float, draws: Sequence[float]) -> PermutationSummary:
    """Place the planned blend's P1 among the shuffled controls' differences from padded ENS.

    The null hypothesis is that UKV-CEDA's columns carry no information the shuffles remove, so the
    planned blend's P1 is one more draw from the controls' distribution. A lower (more negative)
    value is a larger gain, so the one-sided permutation p-value counts the planned value and the
    draws at or below it.

    Args:
        planned: The planned blend minus padded ENS (P1).
        draws: Each shuffled control minus padded ENS.

    Returns:
        The inputs, the rank of `planned` among all of them (1 for the lowest, with a tie ranked
        above the draws it equals), and the p-value `(1 + draws at or below planned) / (1 + draws)`.
    """
    at_or_below = sum(draw <= planned for draw in draws)
    return PermutationSummary(
        planned=planned,
        draws=tuple(draws),
        rank=1 + sum(draw < planned for draw in draws),
        p_value=(1 + at_or_below) / (1 + len(draws)),
    )


def mean_difference(
    *, losses: pl.DataFrame, treatment: str, reference: str
) -> tuple[float, int, int]:
    """Return the mean paired difference of two arms, with the rows and months it rests on.

    Args:
        losses: Per-row losses at one setting, carrying both arms.
        treatment: The arm whose error is compared.
        reference: The arm it is compared against.

    Returns:
        The treatment-minus-reference mean over fitting seeds and rows, the rows per seed, and the
        months.
    """
    differences, months = paired_differences(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    return float(differences.mean()), differences.shape[1], len(np.unique(months))


def permutation_record(
    *, stage: Stage, contrast: str, scope: str, value: float, n_rows: int, n_months: int
) -> IntervalRecord:
    """Return one permutation-test value as a row of `intervals.parquet`, with no interval."""
    return {
        "domain": stage.domain,
        "day": stage.day,
        "setting": PRIMARY,
        "contrast": contrast,
        "scope": scope,
        "level": 0.0,
        "difference": value,
        "lower": float("nan"),
        "upper": float("nan"),
        "n_rows": n_rows,
        "n_months": n_months,
    }


def permutation_lines(
    *, planned: PlannedStage, losses: pl.DataFrame, records: list[IntervalRecord]
) -> tuple[list[str], PermutationSummary | None]:
    """Format the post hoc permutation test of one solar stage, if its fits are saved.

    Args:
        planned: The stage and its rows.
        losses: The stage's saved losses at both settings.
        records: Where every printed value is appended for `intervals.parquet`.

    Returns:
        The section's Markdown lines and the summary. Both are empty or `None` where the stage is
        not solar or its extra controls are not all saved.
    """
    stage = planned.stage
    saved = set(losses.select("arm", "setting").unique().iter_rows())
    if not permutation_jobs(stage=stage) or not set(permutation_jobs(stage=stage)) <= saved:
        return [], None
    pad, blend, control, control_b = stage_arms(day=stage.day)
    primary = at_setting(losses=losses, setting=PRIMARY)
    seeded = {
        0: control,
        1000: control_b,
        **dict(zip(PERMUTATION_SEEDS, permutation_arms(day=stage.day), strict=True)),
    }
    draws: dict[int, float] = {}
    for seed, arm in seeded.items():
        value, n_rows, n_months = mean_difference(losses=primary, treatment=arm, reference=pad)
        draws[seed] = value
        records.append(
            permutation_record(
                stage=stage,
                contrast="permutation_draw",
                scope=f"{PERMUTATION_SCOPE}: seed {seed}",
                value=value,
                n_rows=n_rows,
                n_months=n_months,
            )
        )
    p1, n_rows, n_months = mean_difference(losses=primary, treatment=blend, reference=pad)
    summary = permutation_summary(planned=p1, draws=list(draws.values()))
    for contrast, value in (
        ("permutation_p1", p1),
        ("permutation_rank", float(summary.rank)),
        ("permutation_p", summary.p_value),
    ):
        records.append(
            permutation_record(
                stage=stage,
                contrast=contrast,
                scope=PERMUTATION_SCOPE,
                value=value,
                n_rows=n_rows,
                n_months=n_months,
            )
        )
    scale = PERCENTAGE_POINTS
    lines = [
        f"#### Post hoc: permutation test, {stage.domain} lead day {stage.day}",
        "",
        (
            f"Post hoc and exploratory, added after the second science review. Each of the "
            f"{len(draws)} shuffled controls (the planned two and {len(PERMUTATION_SEEDS)} "
            "extra, fitted at the primary setting only) shuffles UKV-CEDA's columns within "
            "generator, year-month, and hour of day under its own seed. The table gives each "
            "control's mean error minus padded ENS's, in points of capacity on the stage's rows."
        ),
        "",
        "| Shuffle seed | Control minus padded ENS (points) |",
        "|---|---|",
        *(f"| {seed} | {value * scale:+.3f} |" for seed, value in sorted(draws.items())),
        f"| planned blend (P1) | {p1 * scale:+.3f} |",
        "",
        (
            f"The planned blend's P1 is {p1 * scale:+.3f} points. The lowest control is "
            f"{min(draws.values()) * scale:+.3f} and the median is "
            f"{float(np.median(list(draws.values()))) * scale:+.3f}. Among the {len(draws) + 1} "
            f"values (the {len(draws)} controls and P1), P1 ranks {summary.rank} from the lowest. "
            f"The one-sided permutation p-value is {summary.p_value:.3f}; with {len(draws)} "
            f"controls the smallest value it can take is {1 / (len(draws) + 1):.3f}."
        ),
        "",
    ]
    return lines, summary


def permutation_summary_lines(
    *, summaries: Mapping[tuple[str, int], PermutationSummary]
) -> list[str]:
    """Format every solar stage's permutation test as one table, empty if none is saved."""
    if not summaries:
        return []
    scale = PERCENTAGE_POINTS
    lines = [
        (
            "| Technology | Lead day | Planned P1 (points) | Lowest control (points) "
            "| Median control (points) | Highest control (points) | Rank of P1 (1 is lowest) "
            "| Permutation p-value |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {domain} | {day} | {item.planned * scale:+.3f} | {min(item.draws) * scale:+.3f} "
        f"| {float(np.median(item.draws)) * scale:+.3f} | {max(item.draws) * scale:+.3f} "
        f"| {item.rank} of {len(item.draws) + 1} | {item.p_value:.3f} |"
        for (domain, day), item in summaries.items()
    ]
    return lines


OLDER_CONTRASTS: Final[dict[str, str]] = {
    "older_p1": "Older-run blend minus padded ENS trained on the older-run rows (P1, older run)",
    "older_p2": "Older-run blend minus its shuffled control (P2, older run; first seed)",
    "older_vs_fresh": "Older-run blend minus the planned blend (UKV-CEDA run 12 hours older)",
    "fresh_p1_same_rows": "Planned blend minus padded ENS, on the older-run rows",
    "training_rows": (
        "Tilt: padded ENS trained on all rows minus padded ENS trained on the older-run rows"
    ),
}
"""The post hoc older-run section's contrasts, by their code in `intervals.parquet`."""


class OlderReading(NamedTuple):
    """What a stage's older-run section concludes, for the older-run readings table."""

    reading: str
    p1: dict[str, BootstrapInterval]
    p2: dict[str, BootstrapInterval]
    fresh_p1: dict[str, BootstrapInterval]
    vs_fresh: dict[str, BootstrapInterval]


def older_lead_hours(*, day: int) -> pl.Expr:
    """Return the lead, in hours, at which the older run is read for each row of a stage.

    The lead is `24 * (day + extra_days) - run_hour + h`, where `h` is the hour of the row's `time`
    label: the build's rule, which for a solar label (the hour ending at it) is the instant's hour
    plus 1.

    Args:
        day: The ENS lead day.

    Returns:
        An expression over `time`.
    """
    spec = build.OLDER_RUN
    return pl.col("time").dt.hour() + (24 * (day + spec.extra_days) - spec.run_hour)


def older_split_lines(
    *,
    planned: PlannedStage,
    on_rows: Mapping[str, pl.DataFrame],
    records: list[IntervalRecord],
) -> list[str]:
    """Split a day-1 older-run stage at the store's hourly limit, from the saved losses.

    The older run is read at a lead of up to 56 hours at day 1, past the store's 48 hours of hourly
    steps, where the planned run is hourly throughout. Scoring the rows at or before lead 48 hours
    keeps both runs hourly. The rows beyond it are the late hours of the day, so the split also
    separates hours of the day.

    Args:
        planned: The stage and its rows, whose `older_frame` is set.
        on_rows: The stage's losses at each setting, on the older-run rows.
        records: Where every printed interval is appended for `intervals.parquet`.

    Returns:
        The section's Markdown lines, or an empty list at a lead day other than 1.
    """
    stage, rows = planned.stage, planned.older_frame
    if rows is None or stage.day != 1:
        return []
    pad, fresh, _, _ = stage_arms(day=stage.day)
    older_pad, older, _ = older_arms(day=stage.day)
    pairs = {
        "older_p1": (older, older_pad),
        "fresh_p1_same_rows": (fresh, pad),
        "older_vs_fresh": (older, fresh),
    }
    hourly = rows.select(
        "site", "time", hourly=older_lead_hours(day=stage.day) <= HOURLY_LEAD_LIMIT_HOURS
    )
    lines = [
        (
            f"Post hoc split of day {stage.day} at the store's hourly limit, scored from the "
            f"saved losses: the older run is read at lead {HOURLY_LEAD_LIMIT_HOURS} hours or less "
            "(hourly steps) or beyond (3-hourly steps rebuilt hourly). The planned run is hourly "
            "throughout. The rows beyond the limit are the late hours of the day."
        ),
        "",
        "| Rows | Share of rows | Contrast (points) | Primary | Sensitivity |",
        "|---|---|---|---|---|",
    ]
    for is_hourly, scope in OLDER_SPLIT_SCOPES.items():
        keys = hourly.filter(pl.col("hourly") == is_hourly).select("site", "time")
        share = keys.height / rows.height
        label = (
            f"lead {HOURLY_LEAD_LIMIT_HOURS} hours or less"
            if is_hourly
            else f"lead beyond {HOURLY_LEAD_LIMIT_HOURS} hours"
        )
        for code, (treatment, reference) in pairs.items():
            by_setting = {
                setting: difference(
                    losses=on_rows[setting].join(keys, on=["site", "time"]),
                    treatment=treatment,
                    reference=reference,
                )
                for setting in SETTINGS
            }
            records.extend(
                record(stage=stage, setting=setting, contrast=code, scope=scope, interval=interval)
                for setting, interval in by_setting.items()
            )
            lines.append(
                f"| {label} ({keys.height} rows) | {share:.1%} | {OLDER_CONTRASTS[code]} "
                f"| {interval_cell(interval=by_setting[PRIMARY])} "
                f"| {interval_cell(interval=by_setting[SENSITIVITY])} |"
            )
    lines.append("")
    return lines


def older_lines(
    *, planned: PlannedStage, losses: pl.DataFrame, records: list[IntervalRecord]
) -> tuple[list[str], OlderReading | None]:
    """Format the post hoc older-run section of one stage, if its fits are saved.

    The older-run blend reads UKV-CEDA's 15 UTC run of the day before ENS's run, which starts 9
    hours before ENS's 00 UTC run and leads 12 hours longer than the planned blend's run.

    Args:
        planned: The stage and its rows.
        losses: The stage's saved losses at both settings.
        records: Where every printed interval is appended for `intervals.parquet`.

    Returns:
        The section's Markdown lines, and what the stage concludes. Both are empty or `None` where
        the stage has no complete saved older-run fit.
    """
    stage, rows = planned.stage, planned.older_frame
    saved = set(losses.select("arm", "setting").unique().iter_rows())
    if rows is None or not set(older_jobs(day=stage.day)) <= saved:
        return [], None
    pad, fresh, _, _ = stage_arms(day=stage.day)
    older_pad, older, older_control = older_arms(day=stage.day)
    on_rows = {
        setting: at_setting(losses=losses, setting=setting).join(
            rows.select("site", "time"), on=["site", "time"]
        )
        for setting in SETTINGS
    }
    pairs = {
        "older_p1": (older, older_pad),
        "older_p2": (older, older_control),
        "older_vs_fresh": (older, fresh),
        "fresh_p1_same_rows": (fresh, pad),
        "training_rows": (pad, older_pad),
    }
    intervals = {
        code: {
            setting: difference(losses=on_rows[setting], treatment=treatment, reference=reference)
            for setting in SETTINGS
        }
        for code, (treatment, reference) in pairs.items()
    }
    for code, by_setting in intervals.items():
        records.extend(
            record(
                stage=stage, setting=setting, contrast=code, scope=OLDER_SCOPE, interval=interval
            )
            for setting, interval in by_setting.items()
        )
    verdict = reading(
        day=stage.day,
        p1=intervals["older_p1"],
        p2={setting: [intervals["older_p2"][setting]] for setting in SETTINGS},
    )
    lines = [
        f"#### Post hoc: ENS day {stage.day} plus UKV-CEDA's older run, {stage.domain}",
        "",
        (
            "Post hoc and exploratory, added after the second science review. UKV-CEDA's columns "
            "come from the 15 UTC run of the day before ENS's run. That run starts 9 hours before "
            "ENS's 00 UTC run, where the planned blend's run starts 3 hours after it, and its lead "
            "is 12 hours longer. The two changes move together, so this contrast cannot separate "
            f"the effect of the longer lead from the effect of the earlier start. {rows.height} of "
            f"the stage's {planned.frame.height} rows hold the older run. The older-run blend, its "
            "control, and its padded ENS reference were fitted on those rows. The planned blend "
            f"and the planned padded ENS reference were fitted on all {planned.frame.height} "
            f"rows, and every contrast below scores the same {rows.height} rows. The last "
            "contrast is the measured effect of the planned reference's extra training rows. "
            f"{TIME_RESOLUTION_CAVEAT}"
        ),
        "",
        "| Contrast (points) | Primary | Sensitivity |",
        "|---|---|---|",
        *(
            f"| {label} | {interval_cell(interval=intervals[code][PRIMARY])} "
            f"| {interval_cell(interval=intervals[code][SENSITIVITY])} |"
            for code, label in OLDER_CONTRASTS.items()
        ),
        "",
        (
            f"Post hoc reading, one control seed: {verdict}. P1 (older run) at the primary "
            f"setting: {null_reading(interval=intervals['older_p1'][PRIMARY])}. At the "
            f"sensitivity setting: {null_reading(interval=intervals['older_p1'][SENSITIVITY])}."
        ),
        "",
        "| Arm | Setting | Error (% of capacity) | 95% interval | Rows | Months |",
        "|---|---|---|---|---|---|",
    ]
    for setting in SETTINGS:
        board = leaderboard(
            losses=on_rows[setting], arms=[pad, fresh, older_pad, older, older_control]
        )
        for row in board.iter_rows(named=True):
            records.append(
                {
                    "domain": stage.domain,
                    "day": stage.day,
                    "setting": setting,
                    "contrast": "error",
                    "scope": f"{OLDER_SCOPE} rows: {row['arm']}",
                    "level": 95.0,
                    "difference": row["value"],
                    "lower": row["lower_95"],
                    "upper": row["upper_95"],
                    "n_rows": row["n_rows"],
                    "n_months": row["n_months"],
                }
            )
            text = error_text(value=row["value"], lower=row["lower_95"], upper=row["upper_95"])
            value, interval_part = text.split(" [")
            lines.append(
                f"| `{row['arm']}` | {setting} | {value} | [{interval_part} | {row['n_rows']} "
                f"| {row['n_months']} |"
            )
    lines.append("")
    lines += older_split_lines(planned=planned, on_rows=on_rows, records=records)
    return lines, OlderReading(
        reading=verdict,
        p1=intervals["older_p1"],
        p2=intervals["older_p2"],
        fresh_p1=intervals["fresh_p1_same_rows"],
        vs_fresh=intervals["older_vs_fresh"],
    )


OLDER_READING_COLUMN: Final[str] = "Post hoc reading (one control seed, no Bonferroni correction)"
"""The older-run readings column's heading: it rests on P1 and one control at the unadjusted 95%
level, where a planned reading rests on both controls and the corrected intervals."""


def older_summary_lines(*, readings: Mapping[tuple[str, int], OlderReading]) -> list[str]:
    """Format the older-run readings as one table, empty if none are saved."""
    if not readings:
        return []
    scale = PERCENTAGE_POINTS
    lines = [
        (
            f"| Technology | Lead day | {OLDER_READING_COLUMN} "
            "| Planned P1, same rows, primary / sensitivity "
            "| Older-run P1, primary / sensitivity "
            "| Older-run P2 (control), primary / sensitivity "
            "| Older-run blend minus planned blend, primary / sensitivity |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for (domain, day), item in readings.items():
        cells = [
            " / ".join(
                f"{interval[setting]['difference'] * scale:+.3f} "
                f"[{interval[setting]['lower_95'] * scale:+.3f}, "
                f"{interval[setting]['upper_95'] * scale:+.3f}]"
                for setting in SETTINGS
            )
            for interval in (item.fresh_p1, item.p1, item.p2, item.vs_fresh)
        ]
        lines.append(f"| {domain} | {day} | {item.reading} | " + " | ".join(cells) + " |")
    return lines


def kept_gain_lines(*, records: Sequence[IntervalRecord]) -> list[str]:
    """Format the share of the planned gain that each post hoc blend keeps, empty if none is saved.

    The share is the post hoc blend's P1 point estimate over the planned blend's P1 on the same
    rows. It is a ratio of two estimates, so it carries no interval, and it is read only where both
    estimates are well below zero.

    Args:
        records: Every printed interval.

    Returns:
        A Markdown table with one row per technology, lead day, and post hoc blend.
    """
    point = {
        (r["domain"], r["day"], r["setting"], r["contrast"], r["scope"]): r["difference"]
        for r in records
    }
    lines: list[str] = []
    for blend, scope, contrast in (
        ("stale", STALE_SCOPE, "stale_p1"),
        ("older run", OLDER_SCOPE, "older_p1"),
    ):
        for domain in fit_aifs.DOMAINS:
            for day in DAYS:
                shares = []
                for setting in SETTINGS:
                    planned = point.get((domain, day, setting, "fresh_p1_same_rows", scope))
                    post_hoc = point.get((domain, day, setting, contrast, scope))
                    if planned is None or post_hoc is None:
                        break
                    shares.append(f"{post_hoc / planned:.0%}")
                else:
                    lines.append(f"| {blend} | {domain} | {day} | {' / '.join(shares)} |")
    if not lines:
        return []
    return [
        (
            "| Post hoc blend | Technology | Lead day | Share of the planned gain kept, "
            "primary / sensitivity |"
        ),
        "|---|---|---|---|",
        *lines,
    ]


def columns_lines() -> list[str]:
    """Print every arm's feature columns at lead day 1 for each technology, for a reviewer."""
    lines = ["## Columns of every arm at lead day 1", ""]
    for domain in fit_aifs.DOMAINS:
        lines += [f"### {domain}", ""]
        lines += [
            f"- `{arm}` ({len(fit_aifs.arm_features(arm=arm, domain=domain))} columns): "
            + ", ".join(fit_aifs.arm_features(arm=arm, domain=domain))
            for arm in stage_arms(day=1)
        ]
        lines.append("")
    return lines


def report_text(
    *,
    sections: Sequence[tuple[str, list[str]]],
    summary: Sequence[str],
    stale_summary: Sequence[str],
    cpu_line: str,
    padding_line: str,
    permutation_summary: Sequence[str] = (),
    older_summary: Sequence[str] = (),
    kept_summary: Sequence[str] = (),
) -> str:
    """Return `report.md`: the design, the readings, the columns, and every stage.

    Args:
        sections: Each technology's name and its stages' lines.
        summary: The readings table's lines.
        stale_summary: The post hoc stale blend's readings table, empty if none is saved.
        cpu_line: The GPU-CPU noise-floor sentence.
        padding_line: The sentence on whether padded and unpadded ENS score identically.
        permutation_summary: The post hoc permutation test's table, empty if none is saved.
        older_summary: The post hoc older-run readings table, empty if none is saved.
        kept_summary: The share of the planned gain each post hoc blend keeps, empty if none.

    Returns:
        The report.
    """
    lines = [
        "# UKV-CEDA blends: report",
        "",
        (
            "Every fit is on the GPU. Differences are first arm minus second, in percentage points "
            "of capacity, so a negative difference means the first arm has the lower error. P1 is "
            "the blend minus ENS's mean padded to the blend's column count, and P2 is the blend "
            "minus each control, whose UKV-CEDA columns are shuffled within generator, year-month, "
            "and hour of day under two seeds. The blend lowers the error at a technology and lead "
            "day only if the upper 95% bound of P1 and of both P2 contrasts is below zero at both "
            "settings. Where P1 is below zero at both settings and a P2 bound is not, the reading "
            f"is `{UNRESOLVED_LOWER}`: the blend's error is statistically significantly lower than "
            "padded ENS's, and the planned control test is not passed. The Bonferroni interval "
            f"corrects P1's 95% level across {N_P1_INTERVALS} P1 intervals per setting "
            f"({BONFERRONI_LEVEL}%), is printed at both settings, and P2 is not adjusted. "
            "UKV-CEDA's lead is 3 hours fresher than ENS's at every hour, which favours the "
            "blend, so the planned contrasts cannot separate UKV-CEDA's weather from its later "
            "run; the post hoc stale blend, where UKV-CEDA's run is 21 hours staler than ENS's, "
            "tests that, and it also changes UKV-CEDA's lead and, at days 1 and 2, its time "
            "resolution. UKV-CEDA's wind columns are its native 10 m and 925 hPa winds, where "
            "ENS's are 10 m and 100 m, so part of a wind gain may come from the 925 hPa level. "
            "Rows lost to each cause are in the folder's `README.md`."
        ),
        "",
        "## Readings",
        "",
        (
            "Where P1's interval includes zero, the last column gives the largest gain the "
            "interval does not exclude, in points of capacity, at each setting. A reading of "
            "`no detectable difference` never means no gain."
        ),
        "",
        *summary,
        "",
        *(
            [
                (
                    "Post hoc: ENS day N plus UKV-CEDA day N + 1 (UKV-CEDA 21 hours staler than "
                    "ENS), a post hoc reading with both control seeds."
                ),
                "",
                *stale_summary,
                "",
            ]
            if stale_summary
            else []
        ),
        *(
            [
                (
                    "Post hoc: permutation test of the solar blend. The planned blend's P1 is "
                    "placed among the differences from padded ENS of the planned two shuffled "
                    "controls and 15 extra, all at the primary setting."
                ),
                "",
                *permutation_summary,
                "",
            ]
            if permutation_summary
            else []
        ),
        *(
            [
                (
                    "Post hoc: ENS day N plus UKV-CEDA's 15 UTC run of the day before ENS's run. "
                    "This tests how the gain falls as the UKV-CEDA run gets older: the run "
                    "starts 9 hours before ENS's 00 UTC run, where the planned blend's run starts "
                    "3 hours after it, and the lead is 12 hours longer. It cannot separate the "
                    "effect of the longer lead from the effect of the earlier start, because "
                    "the two change together, and at days 1 and 2 more of its hours are "
                    "rebuilt from 3-hourly steps. The planned P1 on the same rows is the "
                    "reference."
                ),
                "",
                *older_summary,
                "",
            ]
            if older_summary
            else []
        ),
        *(
            [
                (
                    "Post hoc: the share of the planned gain that each post hoc blend keeps, the "
                    "post hoc blend's P1 over the planned blend's P1 on the same rows (point "
                    "estimates, no interval)."
                ),
                "",
                *kept_summary,
                "",
            ]
            if kept_summary
            else []
        ),
        cpu_line,
        "",
        padding_line,
        "",
        *columns_lines(),
    ]
    for name, section in sections:
        lines += [f"## {name}", "", *section]
    return "\n".join(lines)


def summary_lines(*, readings: Mapping[tuple[str, int], StageReading]) -> list[str]:
    """Format the readings of every stage as one table."""
    lines = [
        (
            "| Technology | Lead day | Reading | P1 survives the Bonferroni correction at both "
            "settings | Largest gain P1 leaves open (points of capacity) |"
        ),
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {domain} | {day} | {item.reading} | {'yes' if item.survives_bonferroni else 'no'} "
        f"| {item.left_open} |"
        for (domain, day), item in readings.items()
    ]
    return lines


def stale_summary_lines(*, readings: Mapping[tuple[str, int], str]) -> list[str]:
    """Format the post hoc stale blend's readings as one table, empty if none are saved."""
    if not readings:
        return []
    lines = ["| Technology | Lead day | Reading |", "|---|---|---|"]
    lines += [f"| {domain} | {day} | {text} |" for (domain, day), text in readings.items()]
    return lines


def padding_check_line(*, output_dir: Path) -> str:
    """Return the sentence on the saved padding check, or say that none is saved."""
    path = output_dir / PADDING_CHECK_NAME
    if not path.exists():
        return "No padding check is saved."
    checks: dict[str, dict[str, object]] = json.loads(path.read_text())
    parts = [
        f"{name.replace('_', ' ')}: {'identical' if check['identical'] else 'not identical'} "
        f"(largest per-row gap {float(str(check['largest_gap_mw'])):.3g} MW)"
        for name, check in checks.items()
    ]
    return (
        "Padding check, ENS's mean alone against its padded copy, per-row losses at the primary "
        "setting: " + "; ".join(parts) + "."
    )


def report_paths(*, output_dir: Path, name: str) -> tuple[Path, Path]:
    """Return the report and intervals paths: `report.md` and `intervals.parquet` for `report`."""
    if name == "report":
        return output_dir / "report.md", output_dir / "intervals.parquet"
    return output_dir / f"{name}.md", output_dir / f"{name}_intervals.parquet"


def build_report(*, planned: Sequence[PlannedStage], output_dir: Path) -> tuple[str, pl.DataFrame]:
    """Build the report and the interval table from the saved losses, fitting nothing.

    Args:
        planned: Every stage.
        output_dir: The folder holding the saved losses.

    Returns:
        The report text and one row per printed interval.
    """
    records: list[IntervalRecord] = []
    readings: dict[tuple[str, int], StageReading] = {}
    stale_readings: dict[tuple[str, int], str] = {}
    permutations: dict[tuple[str, int], PermutationSummary] = {}
    older_readings: dict[tuple[str, int], OlderReading] = {}
    sections: list[tuple[str, list[str]]] = []
    gpu_wind_day1: pl.DataFrame | None = None
    for domain in fit_aifs.DOMAINS:
        lines: list[str] = []
        for item in (p for p in planned if p.stage.domain == domain):
            losses = verified_losses(planned=item, output_dir=output_dir)
            if item.stage == Stage("wind", 1):
                gpu_wind_day1 = losses
            stage_text, stage_reading = stage_lines(planned=item, losses=losses, records=records)
            lines += stage_text
            readings[(domain, item.stage.day)] = stage_reading
            stale_text, stale_reading = stale_lines(planned=item, losses=losses, records=records)
            lines += stale_text
            if stale_reading is not None:
                stale_readings[(domain, item.stage.day)] = stale_reading
            permutation_text, permutation = permutation_lines(
                planned=item, losses=losses, records=records
            )
            lines += permutation_text
            if permutation is not None:
                permutations[(domain, item.stage.day)] = permutation
            older_text, older_reading = older_lines(planned=item, losses=losses, records=records)
            lines += older_text
            if older_reading is not None:
                older_readings[(domain, item.stage.day)] = older_reading
        sections.append((domain.capitalize(), lines))
    cpu_file = output_dir / f"{stem(stage=Stage('wind', 1), group=CPU_GROUP)}_losses.parquet"
    cpu_line = "No CPU refit is saved."
    if cpu_file.exists() and gpu_wind_day1 is not None:
        cpu_line = noise_floor_line(cpu=pl.read_parquet(cpu_file), gpu=gpu_wind_day1)
    return (
        report_text(
            sections=sections,
            summary=summary_lines(readings=readings),
            stale_summary=stale_summary_lines(readings=stale_readings),
            cpu_line=cpu_line,
            padding_line=padding_check_line(output_dir=output_dir),
            permutation_summary=permutation_summary_lines(summaries=permutations),
            older_summary=older_summary_lines(readings=older_readings),
            kept_summary=kept_gain_lines(records=records),
        ),
        pl.DataFrame(records),
    )


# --- Modes ----------------------------------------------------------------------------------------


def print_plan(
    *,
    planned: Sequence[PlannedStage],
    output_dir: Path,
    only_missing: bool,
    post_hoc: PostHocKind | None = None,
) -> None:
    """Print every stage's rows and fits, and the number of (arm, site) fits, without fitting.

    Under `post_hoc` the listed fits are that analysis's, on its rows.
    """
    total = 0
    for item in planned:
        group, jobs = jobs_to_fit(
            output_dir=output_dir, stage=item.stage, only_missing=only_missing, post_hoc=post_hoc
        )
        rows = item.frame if post_hoc is None else post_hoc_frame(planned=item, kind=post_hoc)
        if rows is None or (
            post_hoc is not None and not post_hoc_jobs(stage=item.stage, kind=post_hoc)
        ):
            sys.stdout.write(f"{item.stage.domain} day {item.stage.day}: no {post_hoc} fits\n")
            continue
        sites = rows["site"].n_unique()
        total += len(jobs) * sites
        missing = "the next day" if post_hoc == "stale" else "the older run"
        lacking = f" ({item.frame.height - rows.height} stage rows lack {missing})"
        sys.stdout.write(
            f"{item.stage.domain} day {item.stage.day}: {rows.height} rows"
            f"{lacking if post_hoc in ('stale', 'older run') else ''}, {sites} sites, "
            f"fold offsets {item.offsets}, group {group}: {len(jobs)} (arm, setting) fits\n"
        )
    sys.stdout.write(f"{total} (arm, site) fits in all\n")


class PaddingCheck(NamedTuple):
    """Whether ENS's mean alone and its padded copy score identically at one stage."""

    identical: bool
    largest_gap_mw: float


def padded_matches_unpadded(*, frame: pl.DataFrame, domain: DomainType, day: int) -> PaddingCheck:
    """Fit ENS's mean alone and its padded copy at the primary setting and compare per-row losses.

    Args:
        frame: A stage's rows.
        domain: `solar` or `wind`.
        day: The lead day.

    Returns:
        Whether every row's absolute error is identical, and the largest per-row gap in megawatts.
    """
    alone = f"ens_mean_day{day}"
    padded = arm_name(day=day, role="_pad")
    losses = fit_aifs.fit_jobs(
        frame=frame, domain=domain, jobs=[(alone, PRIMARY), (padded, PRIMARY)], workers=1
    )
    keys = ["site", "time", "seed"]
    columns = [*keys, "absolute_error_mw"]
    first = losses.filter(pl.col("arm") == alone).select(columns).sort(keys)
    second = losses.filter(pl.col("arm") == padded).select(columns).sort(keys)
    gap = float(
        np.abs(first["absolute_error_mw"].to_numpy() - second["absolute_error_mw"].to_numpy()).max()
    )
    return PaddingCheck(identical=first.equals(second), largest_gap_mw=gap)


def run_check(*, planned: Sequence[PlannedStage], output_dir: Path) -> int:
    """Time one arm twice, print the run's estimate, and compare padded with unpadded ENS.

    The padding comparison runs at day 1 for wind and for solar, and its result is written once to
    `PADDING_CHECK_NAME` in `output_dir`, which the report prints.

    Args:
        planned: Every stage.
        output_dir: The folder holding the saved fits, which sets the number still to fit.

    Returns:
        0 if the two GPU runs agree, else 1.
    """
    check_path = output_dir / PADDING_CHECK_NAME
    refuse_to_overwrite(paths=[check_path])
    first = next(item for item in planned if item.stage == Stage("wind", 1))
    agree, seconds = fit_aifs.time_two_fits(
        frame=first.frame, arm=arm_name(day=1, role=""), domain="wind"
    )
    n_fits = sum(
        len(jobs_to_fit(output_dir=output_dir, stage=item.stage, only_missing=True)[1])
        * item.frame["site"].n_unique()
        for item in planned
    )
    checks = {
        f"{stage.domain}_day{stage.day}": padded_matches_unpadded(
            frame=item.frame, domain=stage.domain, day=stage.day
        )
        for stage in (Stage("wind", 1), Stage("solar", 1))
        for item in planned
        if item.stage == stage
    }
    check_path.write_text(
        json.dumps(
            {
                name: {
                    "identical": check.identical,
                    "largest_gap_mw": check.largest_gap_mw,
                    "setting": PRIMARY,
                }
                for name, check in checks.items()
            }
        )
    )
    sys.stdout.write(
        f"one arm at one site: {seconds:.0f} s; {n_fits} (arm, site) fits are about "
        f"{n_fits * seconds / 3600:.1f} h on one worker\n"
        f"two GPU runs agree: {agree}\n"
    )
    for name, check in checks.items():
        sys.stdout.write(
            f"{name}: ENS's mean alone and padded to the blend's column count have identical "
            f"per-row losses: {check.identical} (largest gap {check.largest_gap_mw:.3g} MW)\n"
        )
    sys.stdout.write(f"CHECK {'PASS' if agree else 'FAIL'}\n")
    return 0 if agree else 1


def run_fits(
    *,
    planned: Sequence[PlannedStage],
    output_dir: Path,
    workers: int,
    only_missing: bool,
    post_hoc: PostHocKind | None = None,
    report_name: str = "report",
) -> int:
    """Fit every stage, then the CPU refit, and write the report and intervals once.

    Args:
        planned: Every stage.
        output_dir: The write-once folder.
        workers: How many (arm, site) fits run at once.
        only_missing: Whether to fit the pairs a partly saved stage lacks.
        post_hoc: Which post hoc analysis to fit instead of the planned arms. The CPU refit is
            then skipped.
        report_name: The report's name, `report` or a new name for a rerun.

    Returns:
        0.
    """
    report_path, intervals_path = report_paths(output_dir=output_dir, name=report_name)
    refuse_to_overwrite(paths=[report_path, intervals_path])
    for item in planned:
        losses = fit_stage(
            planned=item,
            output_dir=output_dir,
            workers=workers,
            only_missing=only_missing,
            post_hoc=post_hoc,
        )
        if item.stage == Stage("wind", 1) and post_hoc is None:
            cpu_noise_floor(planned=item, output_dir=output_dir, gpu=losses)
        _LOG.info("%s day %d: fitted", item.stage.domain, item.stage.day)
    text, intervals = build_report(planned=planned, output_dir=output_dir)
    report_path.write_text(text)
    intervals.write_parquet(intervals_path)
    return 0


def check_output_dir(*, output_dir: Path, read_only: Sequence[Path]) -> None:
    """Raise unless `output_dir` is the one folder this script writes to and no input folder.

    Args:
        output_dir: Where the fits write.
        read_only: The folders the script only reads.

    Raises:
        ValueError: If `output_dir` is a read-only folder or is not named like the build's.
    """
    build.check_output_dir(output_dir=output_dir, read_only=read_only)


def workers_argument(text: str) -> int:
    """Parse `--workers`, refusing a count above `MAX_WORKERS`."""
    workers = fit_aifs.workers_argument(text)
    if workers > MAX_WORKERS:
        msg = f"--workers is at most {MAX_WORKERS} under the load rule, not {workers}"
        raise argparse.ArgumentTypeError(msg)
    return workers


def main() -> int:
    """Fit the blends, or list (`--dry-run`), time (`--check`), or report (`--report-only`)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    studies_dir = REPO_DATA_DIR / "studies"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--published-dir", type=Path, default=studies_dir / build.PUBLISHED_DIR_NAME
    )
    parser.add_argument("--day4-dir", type=Path, default=studies_dir / build.DAY4_DIR_NAME)
    parser.add_argument("--output-dir", type=Path, default=studies_dir / build.OUTPUT_DIR_NAME)
    parser.add_argument(
        "--older-dir",
        type=Path,
        default=None,
        help="The older-run build's folder; read to fit or to report the older-run blend.",
    )
    parser.add_argument("--workers", type=workers_argument, default=MAX_WORKERS)
    parser.add_argument("--dry-run", action="store_true", help="List the fits; fit nothing.")
    parser.add_argument("--check", action="store_true", help="Time one fit twice; compare padding.")
    parser.add_argument("--only-missing", action="store_true", help="Fit the pairs no file holds.")
    parser.add_argument("--report-only", action="store_true", help="Write the report; fit nothing.")
    post_hoc = parser.add_mutually_exclusive_group()
    post_hoc.add_argument(
        "--post-hoc-stale",
        action="store_const",
        const="stale",
        dest="post_hoc",
        help="Fit the post hoc stale UKV-CEDA blend (ENS day N plus UKV-CEDA day N + 1).",
    )
    post_hoc.add_argument(
        "--post-hoc-permutation",
        action="store_const",
        const="permutation",
        dest="post_hoc",
        help="Fit 15 extra shuffled controls per solar stage at the primary setting.",
    )
    post_hoc.add_argument(
        "--post-hoc-older-run",
        action="store_const",
        const="older run",
        dest="post_hoc",
        help="Fit the post hoc older-run blend (UKV-CEDA's 15 UTC run of the day before ENS's).",
    )
    parser.add_argument(
        "--report-name", default="report", help="The report's name, `report` or new."
    )
    args = parser.parse_args()
    kind: PostHocKind | None = args.post_hoc
    older_dir: Path | None = args.older_dir
    if kind == "older run" and older_dir is None:
        older_dir = studies_dir / build.OLDER_RUN.output_dir_name
    check_output_dir(output_dir=args.output_dir, read_only=[args.published_dir, args.day4_dir])
    planned = plan_stages(
        published_dir=args.published_dir,
        day4_dir=args.day4_dir,
        output_dir=args.output_dir,
        older_dir=older_dir,
        permutation=kind == "permutation" and not args.report_only,
    )
    if args.dry_run:
        print_plan(
            planned=planned,
            output_dir=args.output_dir,
            only_missing=args.only_missing,
            post_hoc=kind,
        )
        return 0
    if args.report_only:
        report_path, intervals_path = report_paths(
            output_dir=args.output_dir, name=args.report_name
        )
        refuse_to_overwrite(paths=[report_path, intervals_path])
        text, intervals = build_report(planned=planned, output_dir=args.output_dir)
        report_path.write_text(text)
        intervals.write_parquet(intervals_path)
        return 0
    fit_aifs.check_gpu_visible()
    if args.check:
        return run_check(planned=planned, output_dir=args.output_dir)
    check_verified(output_dir=args.output_dir, stamp=read_build_stamp(output_dir=args.output_dir))
    if older_dir is not None:
        check_verified(
            output_dir=older_dir,
            stamp=read_build_stamp(output_dir=older_dir, spec=build.OLDER_RUN),
        )
    return run_fits(
        planned=planned,
        output_dir=args.output_dir,
        workers=args.workers,
        only_missing=args.only_missing,
        post_hoc=kind,
        report_name=args.report_name,
    )


if __name__ == "__main__":
    sys.exit(main())
