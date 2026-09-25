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

Every output carries only the anonymised `site` label.

Run it with `uv run python studies/nwp_forecast_comparison/fit_aifs.py --published-dir PUBLISHED
--output-dir DIR`, after `build_forecast_inputs.py --aifs --output-dir DIR` has written the AIFS
inputs there.
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import polars as pl
from fit_extra_leads import error_text, interval_text
from nwp_forecast_comparison import (
    METRIC,
    SETTINGS,
    TARGET,
    DomainType,
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
    BootstrapInterval,
    bootstrap_difference_at_level,
    paired_differences,
)
from studies.cross_validation import DeviceType, cut_eras, out_of_fold_losses, search_fold_offsets
from studies.guards import check_no_missing, refuse_to_overwrite

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

DOMAINS: Final[tuple[DomainType, DomainType]] = ("solar", "wind")

DEVICE: Final[DeviceType] = "cuda"
"""The XGBoost device every fit here uses."""

PRIMARY: Final[str] = "primary"
SENSITIVITY: Final[str] = "sensitivity"

AIFS_DROPPED_MONTHS: Final[tuple[str, ...]] = ("2025-08", "2026-05")
"""Months holding a version switch: AIFS Single's reverted v1.1 attempt and v1.0 to v1.1 switch in
2025-08, and AIFS v2 with IFS Cycle 50r1 in 2026-05. `nwp_forecast_comparison.DROPPED_MONTHS` drops
2026-01 (the UKV upgrade) as well."""

RUNS_OPEN_END: Final[date] = date(2999, 12, 31)
"""The last run date of the final era, which has no end yet."""

PERMUTED_PREFIX: Final[str] = "aifs_single_day1_permuted"
PERMUTED_B_PREFIX: Final[str] = "aifs_single_day1_permuted_b"
PERMUTED_SEEDS: Final[dict[str, int]] = {PERMUTED_PREFIX: 0, PERMUTED_B_PREFIX: 1000}
"""The climatology reference and its null repeat, each a shuffle of AIFS Single's day-1 columns
among hours sharing a site, a year-month and an hour of day, under its own seed."""

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

AIFS_ARM_PREFIXES: Final[tuple[str, ...]] = (
    "aifs_single_day1",
    "aifs_single_day2",
    "aifs_single_nearest_day1",
    "aifs_ens_mean_day1",
    "aifs_ens_mean_day2",
)
"""The arms that carry a `<prefix>_init_time` column."""


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


def arm_features(*, arm: str, domain: DomainType) -> tuple[str, ...]:
    """Return one arm's feature columns.

    Args:
        arm: An arm's name; `NO_DOY_SUFFIX` marks the refit without `day_of_year`.
        domain: `solar` or `wind`.

    Returns:
        The arm's fixed-length column tuple: seven columns, or six without `day_of_year`.
    """
    columns = arm_columns(domain=domain, prefixes=(arm.removesuffix(NO_DOY_SUFFIX),))
    return (
        tuple(c for c in columns if c != "day_of_year") if arm.endswith(NO_DOY_SUFFIX) else columns
    )


def add_permuted_columns(*, frame: pl.DataFrame, domain: DomainType) -> pl.DataFrame:
    """Add the two permuted copies of AIFS Single's day-1 columns.

    Args:
        frame: Rows carrying `site`, `month`, `hour_of_day` and `aifs_single_day1`'s columns.
        domain: `solar` or `wind`.

    Returns:
        `frame` with `<prefix>_<field>` for both permuted prefixes.
    """
    fields = tuple(
        column
        for column in arm_features(arm="aifs_single_day1", domain=domain)
        if column.startswith("aifs_single_day1_")
    )
    groups = (
        [(fields[0],), (fields[1], fields[2]), (fields[3],)]
        if domain == "wind"
        else [(fields[0],), (fields[1],)]
    )
    output = frame
    for prefix, seed in PERMUTED_SEEDS.items():
        output = climatology_permutation(
            frame=output,
            column_groups=groups,
            by=("site", "month", "hour_of_day"),
            seed=seed,
            suffix=f"_{prefix}",
        ).rename(
            {
                f"{column}_{prefix}": f"{prefix}_{column.removeprefix('aifs_single_day1_')}"
                for group in groups
                for column in group
            }
        )
    return output


def check_runs(*, frame: pl.DataFrame, domain: DomainType, row_set: str) -> None:
    """Raise unless every AIFS arm's run on every row is the right 00 UTC run inside the row's era.

    Args:
        frame: The set's rows, carrying `time`, `era_code` and each AIFS arm's `_init_time`.
        domain: `solar` or `wind`.
        row_set: `single` or `ens`.

    Raises:
        ValueError: Naming each arm whose run is not a 00 UTC run, is not the run `day` days before
            the row's day, or falls outside its era's run dates.
    """
    spec = ROW_SETS[row_set]
    first = pl.col("era_code").replace_strict(
        {era: runs[0] for era, runs in spec.era_runs.items()}, return_dtype=pl.Date
    )
    last = pl.col("era_code").replace_strict(
        {era: runs[1] for era, runs in spec.era_runs.items()}, return_dtype=pl.Date
    )
    time_of_day = pl.col("time") - pl.duration(minutes=30) if domain == "solar" else pl.col("time")
    problems: dict[str, dict[str, int]] = {}
    for prefix in AIFS_ARM_PREFIXES:
        if prefix not in spec.arms:
            continue
        if f"{prefix}_init_time" not in frame.columns:
            msg = (
                f"{domain}/{row_set}: {prefix}_init_time is missing, so its runs cannot be checked"
            )
            raise ValueError(msg)
        init = pl.col(f"{prefix}_init_time")
        day = int(prefix[-1])
        expected_date = (time_of_day.dt.truncate("1d") - pl.duration(days=day)).dt.date()
        counts = frame.select(
            not_midnight=(init.dt.hour() != 0).sum(),
            wrong_day=(init.dt.date() != expected_date).sum(),
            outside_era=((init.dt.date() < first) | (init.dt.date() > last)).sum(),
        ).row(0, named=True)
        if any(counts.values()):
            problems[prefix] = counts
    if problems:
        msg = f"{domain}/{row_set}: rows whose AIFS run is wrong: {problems}"
        raise ValueError(msg)


def aifs_rows(
    *, published_dir: Path, aifs_dir: Path, domain: DomainType, row_set: str
) -> pl.DataFrame:
    """Return one set's rows: the published shared rows inside the set's eras, with AIFS joined on.

    Args:
        published_dir: The folder holding the published inputs.
        aifs_dir: The folder holding `<domain>_aifs_inputs.parquet`.
        domain: `solar` or `wind`.
        row_set: `single` or `ens`.

    Returns:
        The rows with `month`, `era_code`, `era` and `fold` recut inside the AIFS eras, and both
        permuted copies.

    Raises:
        ValueError: If a shared row has no row in the AIFS inputs, an arm's column is missing or
            has a null or a not-a-number on a kept row, the fold design is no longer among those
            covering every calendar month, or a row's AIFS run is wrong.
    """
    spec = ROW_SETS[row_set]
    shared = rows(input_dir=published_dir, domain=domain).drop("era_code", "era", "fold")
    aifs = pl.read_parquet(aifs_dir / f"{domain}_aifs_inputs.parquet")
    keys = ["site", "time"]
    if shared.select(keys).join(aifs.select(keys), on=keys, how="anti").height:
        msg = f"{domain}: shared rows are missing from the AIFS inputs"
        raise ValueError(msg)
    kept = (
        shared.join(aifs, on=keys, how="left")
        .filter(pl.col("month") >= spec.first_month, ~pl.col("month").is_in(AIFS_DROPPED_MONTHS))
        .sort("site", "time")
    )
    columns = [
        column
        for arm in spec.arms
        if arm not in PERMUTED_SEEDS
        for column in arm_features(arm=arm, domain=domain)
    ]
    found = search_fold_offsets(frame=kept, first_months=spec.era_start_months)
    if spec.fold_offsets not in [dict(offsets) for offsets in found]:
        msg = f"{domain}/{row_set}: {spec.fold_offsets} no longer covers every calendar month"
        raise ValueError(msg)
    cut = cut_eras(frame=kept, first_months=spec.era_start_months, fold_offsets=spec.fold_offsets)
    check_no_missing(frame=cut, columns=columns)
    coverage_table(frame=cut)
    check_runs(frame=cut, domain=domain, row_set=row_set)
    return add_permuted_columns(frame=cut, domain=domain)


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
            if len(features) != (6 if arm.endswith(NO_DOY_SUFFIX) else 7):
                msg = f"{domain}: {arm} has {len(features)} columns"
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


def deciding_lines(*, losses: pl.DataFrame, row_set: str, domain: DomainType) -> list[str]:
    """Report the deciding contrast at both settings, wider, month-dropped and no-`day_of_year`.

    Args:
        losses: Every fit's per-row losses on the set.
        row_set: `single` or `ens`.
        domain: `solar` or `wind`.

    Returns:
        Markdown lines, empty for a set with no deciding contrast.
    """
    deciding = ROW_SETS[row_set].deciding
    if deciding is None:
        return []
    treatment, reference = deciding
    primary_losses = losses.filter(pl.col("setting") == PRIMARY)
    second_losses = losses.filter(pl.col("setting") == SENSITIVITY)
    primary = difference(losses=primary_losses, treatment=treatment, reference=reference)
    second = difference(losses=second_losses, treatment=treatment, reference=reference)
    wide = bootstrap_difference_at_level(
        losses=primary_losses,
        treatment=treatment,
        reference=reference,
        metric=METRIC,
        level=BONFERRONI_LEVEL,
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
            f"| primary, {BONFERRONI_LEVEL}% (Bonferroni across solar and wind) | "
            f"{full * 100:+.3f} [{wide[0] * 100:+.3f}, {wide[1] * 100:+.3f}] "
            f"| {primary['n_rows']} | {primary['n_months']} |"
        ),
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
    lines += ["", *deciding_lines(losses=losses, row_set=row_set, domain=domain)]
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


def check_determinism(*, published_dir: Path, aifs_dir: Path) -> bool:
    """Fit one arm at one site twice on the GPU, compare fingerprints, and print a time estimate.

    Args:
        published_dir: The folder holding the published inputs.
        aifs_dir: The folder holding the AIFS inputs.

    Returns:
        Whether the two fingerprints agree.
    """
    frame = aifs_rows(
        published_dir=published_dir, aifs_dir=aifs_dir, domain="wind", row_set="single"
    )
    site = min(frame["site"].unique().to_list())
    one_site = frame.filter(pl.col("site") == site)
    fingerprints = []
    seconds = []
    for _ in range(2):
        start = time.monotonic()
        losses = fit_jobs(
            frame=one_site, domain="wind", jobs=[("aifs_single_day1", PRIMARY)], workers=1
        )
        seconds.append(time.monotonic() - start)
        fingerprints.append(fingerprint(frame=losses))
    n_fits = sum(len(spec.arms) for spec in ROW_SETS.values()) * sum(N_SITES.values())
    sys.stdout.write(
        f"one arm at one site: {min(seconds):.0f} s; {n_fits} primary (arm, site) fits are about "
        f"{n_fits * min(seconds) / 3600:.1f} h on one worker (sensitivity, near-line and "
        "no-day_of_year refits add about a third)\n"
    )
    return fingerprints[0] == fingerprints[1]


def build_stamp(*, aifs_dir: Path, domain: DomainType) -> dict[str, str]:
    """Return the AIFS inputs' SHA-256 and the device, which a saved losses file must match.

    Args:
        aifs_dir: The folder holding `<domain>_aifs_inputs.parquet`.
        domain: `solar` or `wind`.

    Returns:
        `inputs_sha256` and `device`.
    """
    digest = hashlib.sha256((aifs_dir / f"{domain}_aifs_inputs.parquet").read_bytes()).hexdigest()
    return {"inputs_sha256": digest, "device": DEVICE}


def check_saved_losses(
    *,
    losses: pl.DataFrame,
    frame: pl.DataFrame,
    row_set: str,
    stamp_file: Path,
    stamp: dict[str, str],
) -> None:
    """Raise unless saved losses come from this build, this device, and exactly these rows.

    Args:
        losses: The saved losses.
        frame: `aifs_rows`'s frame for the row set.
        row_set: The row set's name.
        stamp_file: The stamp written beside the losses file.
        stamp: `build_stamp`'s result for the current inputs and device.

    Raises:
        ValueError: If the stamp is missing or differs, or the primary losses hold other arms or
            cover other (site, time) rows than the row set.
    """
    if not stamp_file.exists() or json.loads(stamp_file.read_text()) != stamp:
        msg = f"{stamp_file} is missing or names another build or device"
        raise ValueError(msg)
    spec = ROW_SETS[row_set]
    arms = {*spec.arms, *(f"{arm}{NO_DOY_SUFFIX}" for arm in spec.deciding or ())}
    primary = losses.filter(pl.col("setting") == PRIMARY)
    if set(primary["arm"].unique().to_list()) != arms:
        msg = f"{row_set}: the saved losses hold other arms than the set's"
        raise ValueError(msg)
    keys = frame.select("site", "time")
    saved = primary.select("site", "time").unique()
    if (
        saved.join(keys, on=["site", "time"], how="anti").height
        or keys.join(saved, on=["site", "time"], how="anti").height
    ):
        msg = f"{row_set}: the saved losses cover other rows than the set's"
        raise ValueError(msg)


def main() -> int:
    """Fit every arm on both row sets and both technologies, and write the outputs once."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--published-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="(arm, site) fits run at once.")
    parser.add_argument("--check", action="store_true", help="Compare two GPU runs of one arm.")
    args = parser.parse_args()
    if args.output_dir.resolve() == args.published_dir.resolve():
        msg = "the output folder must not be the published folder"
        raise ValueError(msg)
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
                aifs_dir=args.output_dir,
                domain=domain,
                row_set=row_set,
            )
            losses_file = path_for(
                output_dir=args.output_dir, domain=domain, row_set=row_set, kind="losses"
            )
            predictions_file = path_for(
                output_dir=args.output_dir, domain=domain, row_set=row_set, kind="predictions"
            )
            stamp = build_stamp(aifs_dir=args.output_dir, domain=domain)
            stamp_file = losses_file.with_suffix(".json")
            if losses_file.exists():
                losses = pl.read_parquet(losses_file)
                check_saved_losses(
                    losses=losses,
                    frame=frame,
                    row_set=row_set,
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
