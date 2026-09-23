"""Measure whether blending CAMS and SARAH-3 beats CAMS alone, on the two-satellite weather study.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/860>. It reuses the past-solar
study's `record` panel (`weather_products.PANELS["record"]`): the common rows of ERA5, CAMS, SARAH-3
and ICON-DREAM-EU, 2021 to August 2026, with the same folds, eras, seeds, export cap and
`run_experiment.SHARED_FEATURES` the panel already uses. It builds no row set of its own.

**Pre-registered design, fixed before any result exists.**

- **Reproduction guard.** The `cams` and `sarah3` arms here are refitted exactly as the record
  panel's `cams_global` and `sarah3_global` arms: the same feature columns, on the same rows, at the
  same hyperparameters. Before any blend is fitted, each is checked row for row against the record
  panel's saved losses at
  `data/studies/beam_diffuse_split/past_weather_v2/solar_record/losses.parquet` — the same
  `(site, time, fold, seed)` keys and a bit-identical `signed_error_capped_mw`. The run stops if
  either differs.
- **Arms.** SARAH-3 is scored on global irradiance only (its own split is a separation model's
  output, not a retrieval, so `weather_products.py` never fits a SARAH-3 split arm), so every arm
  here reads CAMS's and SARAH-3's global irradiance and nothing else of either product:
    - `cams`, `sarah3`: the single products, refitted for the guard.
    - `cams_sarah3_xgb`: both products' global columns.
    - `cams_sarah3_control`: CAMS's real column plus SARAH-3's column permuted within site, month
      and hour of day (`studies.blending.climatology_permutation`), so the control keeps SARAH-3's
      climatology but loses its weather.
    - `cams_sarah3_mean`: the mean of the two products' global irradiance, as one column.
    - `cams_sarah3_stack`, `cams_sarah3_equal`: the cross-fitted linear stack and the equal-weight
      mean of the two single-product models' out-of-fold predictions
      (`studies.blending.stacked_errors`), derived from the fitted losses with no refit.
    - `cams_cams_noise_xgb`: a negative control. CAMS's real column plus a second copy of it with
      small Gaussian noise added, so the arm has two columns but effectively one product's
      information. It checks that a second, redundant column does not by itself make a blend look
      better, and is reported against `cams` alone.
- **Planned contrasts**, named before any run: `cams_sarah3_xgb` − `cams` (does the blend beat the
  best single product?) and `cams_sarah3_xgb` − `cams_sarah3_control` (does the gain come from
  SARAH-3's information, or only from having a second column?). Both are refitted at
  `studies.cross_validation.SENSITIVITY_HYPER_PARAMETERS` too, so the report can say whether an
  ordering belongs to the features or to the settings. Every other contrast in the report is
  exploratory: the mean, stack and equal blends against `cams`; the two planned contrasts broken
  down by generator, by `weather_products.CLEARNESS_BANDS`, and by calendar year
  (`studies.bootstrap.bootstrap_difference_by_year`, restricted to `weather_products.
  ERA5_BY_YEAR_MONTHS` so a partial final year compares on the same months as a complete one); and
  the negative control against `cams`.
- **Intervals.** Every contrast is bootstrapped with `studies.bootstrap.bootstrap_difference`,
  resampling whole months and a fitting seed, 2,000 times. The report prints every figure as a
  percentage of capacity to 2 decimal places.

Run it with `uv run python studies/beam_diffuse_split/blend_satellites.py`, after
`weather_products.py` has written the `record` panel (`--panel record`). `--resume` reuses the
per-arm fits a previous run left in `fits/`. `--report-only` rebuilds `report.md` from
`losses.parquet` and `reproduction.md` already on disk, fitting nothing; move the current outputs to
a `superseded/` subfolder first, since neither mode overwrites a file.
"""

import argparse
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import polars as pl
import weather_products
from export_cap import with_export_cap
from run_experiment import SHARED_FEATURES as RUN_SHARED_FEATURES
from run_experiment import Job, _add_time_features, run_all
from sources import STUDY_DATA_DIR
from studies.blending import climatology_permutation, stacked_errors
from studies.bootstrap import (
    YearInterval,
    bootstrap_difference,
    bootstrap_difference_by_year,
    per_fold_differences,
)
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS, SENSITIVITY_HYPER_PARAMETERS
from studies.guards import check_no_missing, refuse_to_overwrite

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

OUTPUT_DIR: Final[Path] = STUDY_DATA_DIR / "satellite_blend"
"""Where every output of this study is written."""

FITS_DIR_NAME: Final[str] = "fits"
"""The per-arm fits under `OUTPUT_DIR`, kept until the report is written so `--resume` can reuse
them after a crash, and deleted once every output is on disk."""

RECORD_PANEL_LOSSES: Final[Path] = weather_products.PANELS["record"].output_dir / "losses.parquet"
"""The past-solar study's `record` panel losses, which the reproduction guard checks against."""

SHARED: Final[tuple[str, ...]] = (*RUN_SHARED_FEATURES, "era_code")
"""The features every arm gets: the past-solar study's shared features, plus the era."""

PERMUTATION_GROUPS: Final[tuple[str, ...]] = ("site", "month", "hour_of_day")
"""The rows the control's permuted SARAH-3 column may move between: one site, month, hour of day."""

PERMUTATION_SEED: Final[int] = 20260927
"""Seeds `cams_sarah3_control`'s permutation of SARAH-3's global irradiance."""

MEAN_COLUMN: Final[str] = "ghi_mean_cams_sarah3"
NOISE_COLUMN: Final[str] = "ghi_cams_noise"

NEGATIVE_CONTROL_NOISE_SEED: Final[int] = 20260928
"""Seeds the negative control's noise, drawn once per row in the frame's fixed row order."""

NEGATIVE_CONTROL_NOISE_STD_W_M2: Final[float] = 5.0
"""The negative control's noise standard deviation.

CAMS's global irradiance runs from 0 to about 1,000 W/m2 at midday, so 5 W/m2 keeps the noised copy
almost identical to CAMS's own column: the arm carries two columns but very nearly one product's
information, which is the point of a redundant-column negative control.
"""

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss every table reports: each row's clamped error over its own generator's capacity."""

PERCENTAGE_POINTS: Final[float] = 100.0

KEY_COLUMNS: Final[tuple[str, ...]] = (
    "site",
    "time",
    "month",
    "fold",
    "seed",
    "effective_capacity_mw",
    "constrained",
)
"""The columns identifying a scored row, shared by every arm."""

LOSS_COLUMNS: Final[tuple[str, ...]] = (
    *KEY_COLUMNS,
    "signed_error_capped_mw",
    METRIC,
    "arm",
    "setting",
)
"""The columns `losses.parquet` keeps for every arm, fitted or derived."""

SINGLE_ARMS: Final[tuple[str, str]] = ("cams", "sarah3")

PLANNED_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("cams_sarah3_xgb", "cams"),
    ("cams_sarah3_xgb", "cams_sarah3_control"),
)
"""The only contrasts a recommendation may rest on, named before any result exists."""

SENSITIVITY_ARMS: Final[tuple[str, ...]] = ("cams", "cams_sarah3_xgb", "cams_sarah3_control")
"""The arms refitted at `SENSITIVITY_HYPER_PARAMETERS`, so both planned contrasts can be checked."""

NEGATIVE_CONTROL: Final[tuple[str, str]] = ("cams_cams_noise_xgb", "cams")

DERIVED_ARMS: Final[tuple[str, str]] = ("cams_sarah3_stack", "cams_sarah3_equal")
"""Arms derived from the fitted single-product losses, with no XGBoost fit of their own."""

CONTRAST_HEADER: Final[tuple[str, str]] = (
    (
        "| Scope | Contrast | ΔMAE (pp of capacity) | 95% interval | Excludes zero? "
        "| Folds agreeing | Rows |"
    ),
    "|---|---|---|---|---|---|---|",
)
"""Matches `studies.charts.CONTRAST_COLUMNS`, so the chart script can parse the report directly."""


def _arm_columns() -> dict[str, tuple[str, ...]]:
    """Return every fitted arm's feature columns, in the order the model sees them.

    Returns:
        Arm name to feature columns. The two derived arms (`DERIVED_ARMS`) are not here: they are
        combined from `cams` and `sarah3`'s fitted losses, not fitted themselves.
    """
    return {
        "cams": (*SHARED, "ghi_cams"),
        "sarah3": (*SHARED, "ghi_sarah3"),
        "cams_sarah3_xgb": (*SHARED, "ghi_cams", "ghi_sarah3"),
        "cams_sarah3_control": (*SHARED, "ghi_cams", "ghi_sarah3_shuffled"),
        "cams_sarah3_mean": (*SHARED, MEAN_COLUMN),
        "cams_cams_noise_xgb": (*SHARED, "ghi_cams", NOISE_COLUMN),
    }


def _primary_jobs() -> list[Job]:
    """Return every arm's fit at the primary hyperparameters.

    Returns:
        The jobs `run_experiment.run_all` takes.
    """
    return [
        (arm, "pooled", "power_mw", columns, PRIMARY_HYPER_PARAMETERS, False)
        for arm, columns in _arm_columns().items()
    ]


def _sensitivity_jobs() -> list[Job]:
    """Return the planned contrasts' arms, refitted at the second hyperparameter setting.

    Returns:
        The jobs `run_experiment.run_all` takes.
    """
    columns = _arm_columns()
    return [
        (arm, "sensitivity", "power_mw", columns[arm], SENSITIVITY_HYPER_PARAMETERS, False)
        for arm in SENSITIVITY_ARMS
    ]


def build_rows() -> pl.DataFrame:
    """Build the record panel's common rows, exactly as `weather_products.run_panel` builds them.

    Adds the columns this study's arms need beyond the record panel's own: SARAH-3's
    climatology-permuted copy, the noised copy of CAMS, the mean of CAMS and SARAH-3, and the
    clearness index the exploratory breakdown bins on.

    Returns:
        One row per common site-hour, carrying every column `_arm_columns` names.
    """
    panel = weather_products.PANELS["record"]
    rows = weather_products.common_rows(frame=weather_products.joined(products=panel.products))
    frame = with_export_cap(
        dataset=weather_products.with_eras(frame=_add_time_features(dataset=rows))
    )
    frame = climatology_permutation(
        frame=frame,
        column_groups=[("ghi_sarah3",)],
        by=PERMUTATION_GROUPS,
        seed=PERMUTATION_SEED,
    )
    noise = np.random.default_rng(NEGATIVE_CONTROL_NOISE_SEED).normal(
        scale=NEGATIVE_CONTROL_NOISE_STD_W_M2, size=frame.height
    )
    frame = frame.with_columns(
        (pl.col("ghi_cams") + pl.Series(noise)).alias(NOISE_COLUMN),
        pl.mean_horizontal("ghi_cams", "ghi_sarah3").alias(MEAN_COLUMN),
    ).with_columns(
        kt=pl.when(pl.col("extraterrestrial_horizontal_w_m2") > 0)
        .then(pl.col(MEAN_COLUMN) / pl.col("extraterrestrial_horizontal_w_m2"))
        .otherwise(None)
    )
    check_no_missing(
        frame=frame, columns=[column for columns in _arm_columns().values() for column in columns]
    )
    return frame


def _fit_path(*, arm: str, setting: str) -> Path:
    """Return where one arm's fit is kept until the report is written.

    Args:
        arm: The arm.
        setting: The setting.

    Returns:
        The parquet path.
    """
    return OUTPUT_DIR / FITS_DIR_NAME / f"{setting}__{arm}.parquet"


def _fitted(*, frame: pl.DataFrame, jobs: list[Job], resume: bool) -> pl.DataFrame:
    """Fit every job not already on disk, keep each arm's losses, and return them all.

    Args:
        frame: The record panel's common rows.
        jobs: The fits wanted.
        resume: Whether to reuse a fit a previous run left on disk.

    Returns:
        Every job's losses in `LOSS_COLUMNS`, one row per (site, time, seed, arm, setting).
    """
    missing = [
        job for job in jobs if not (resume and _fit_path(arm=job[0], setting=job[1]).exists())
    ]
    _LOG.info("%d of %d fits to run", len(missing), len(jobs))
    if missing:
        fresh = run_all(dataset=frame, jobs=missing)
        for arm, setting, *_ in missing:
            path = _fit_path(arm=arm, setting=setting)
            path.parent.mkdir(parents=True, exist_ok=True)
            fresh.filter(pl.col("arm") == arm, pl.col("setting") == setting).sort(
                "site", "time", "seed"
            ).write_parquet(path)
    return pl.concat(
        pl.read_parquet(_fit_path(arm=arm, setting=setting)).select(LOSS_COLUMNS)
        for arm, setting, *_ in jobs
    )


class ReproductionRow(TypedDict):
    """One refitted single-product arm's comparison against the record panel's saved losses."""

    arm: str
    published_arm: str
    rows: int
    published_rows: int
    keys_equal: bool
    bit_identical: bool
    max_abs_difference_mw: float


def check_reproduction(*, fitted: pl.DataFrame) -> tuple[list[ReproductionRow], list[str]]:
    """Compare the refitted `cams` and `sarah3` arms with the record panel's saved losses.

    Args:
        fitted: This study's fitted losses, holding `cams` and `sarah3` at setting `pooled`.

    Returns:
        One row per single-product arm, and the same rows rendered as a markdown table.
    """
    published = pl.read_parquet(RECORD_PANEL_LOSSES)
    keys = ["site", "time", "fold", "seed"]
    rows: list[ReproductionRow] = []
    for arm in SINGLE_ARMS:
        published_arm = f"{arm}_global"
        ours, theirs = (
            losses.filter(pl.col("arm") == name, pl.col("setting") == "pooled")
            .sort(keys)
            .select(*keys, "signed_error_capped_mw")
            for losses, name in ((fitted, arm), (published, published_arm))
        )
        keys_equal = ours.height == theirs.height and ours.select(keys).equals(theirs.select(keys))
        identical = keys_equal and ours["signed_error_capped_mw"].equals(
            theirs["signed_error_capped_mw"]
        )
        difference = (
            float(
                np.max(
                    np.abs(
                        ours["signed_error_capped_mw"].to_numpy()
                        - theirs["signed_error_capped_mw"].to_numpy()
                    )
                )
            )
            if keys_equal
            else float("nan")
        )
        rows.append(
            {
                "arm": arm,
                "published_arm": published_arm,
                "rows": ours.height,
                "published_rows": theirs.height,
                "keys_equal": keys_equal,
                "bit_identical": identical,
                "max_abs_difference_mw": difference,
            }
        )
    lines = [
        (
            "| Refitted arm | Published arm | Rows | Published rows | Keys equal "
            "| Bit-identical `signed_error_capped_mw` | Largest difference (MW) |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['arm']} | {row['published_arm']} | {row['rows']:,} | {row['published_rows']:,} "
        f"| {'yes' if row['keys_equal'] else '**no**'} "
        f"| {'yes' if row['bit_identical'] else '**no**'} | {row['max_abs_difference_mw']:.3g} |"
        for row in rows
    ]
    return rows, lines


def _derived_losses(
    *, keys: pl.DataFrame, errors: np.ndarray, arm: str, setting: str
) -> pl.DataFrame:
    """Score a derived arm's signed errors in the loss columns every arm shares.

    Args:
        keys: The key columns, one row per error.
        errors: The derived arm's capped signed error on each row, in MW.
        arm: The derived arm's name.
        setting: The setting of the fits it was derived from.

    Returns:
        One row per scored row, in `LOSS_COLUMNS`.
    """
    return keys.select(KEY_COLUMNS).with_columns(
        signed_error_capped_mw=pl.Series(errors, dtype=pl.Float64),
        **{METRIC: pl.Series(np.abs(errors), dtype=pl.Float64) / pl.col("effective_capacity_mw")},
        arm=pl.lit(arm),
        setting=pl.lit(setting),
    )


def stack_and_equal(*, fitted: pl.DataFrame) -> pl.DataFrame:
    """Combine `cams` and `sarah3`'s fitted losses into the cross-fitted stack and the equal mean.

    Args:
        fitted: The fitted losses, holding `cams` and `sarah3` at setting `pooled`.

    Returns:
        The `cams_sarah3_stack` and `cams_sarah3_equal` arms' losses.
    """
    wide = (
        fitted.filter(pl.col("arm").is_in(SINGLE_ARMS), pl.col("setting") == "pooled")
        .pivot(on="arm", index=list(KEY_COLUMNS), values="signed_error_capped_mw")
        .sort("site", "time", "seed")
    )
    errors = wide.select(*SINGLE_ARMS).to_numpy()
    result = stacked_errors(
        errors=errors,
        sites=wide["site"].to_numpy(),
        folds=wide["fold"].to_numpy(),
        seeds=wide["seed"].to_numpy(),
        fit_rows=~wide["constrained"].to_numpy(),
    )
    stack_arm, equal_arm = DERIVED_ARMS
    return pl.concat(
        [
            _derived_losses(keys=wide, errors=result.errors, arm=stack_arm, setting="pooled"),
            _derived_losses(keys=wide, errors=errors.mean(axis=1), arm=equal_arm, setting="pooled"),
        ]
    )


def _mae(*, losses: pl.DataFrame, arm: str) -> float:
    """Return one arm's mean error, in percentage points of capacity.

    Args:
        losses: Per-row losses, already restricted to one setting.
        arm: The arm.

    Returns:
        The mean, or NaN if the arm is absent.
    """
    rows = losses.filter(pl.col("arm") == arm)
    if rows.is_empty():
        return float("nan")
    return float(rows.select(pl.col(METRIC).mean()).item()) * PERCENTAGE_POINTS


def _contrast_line(*, losses: pl.DataFrame, treatment: str, reference: str, label: str) -> str:
    """Return one markdown row: the paired difference, its interval, and the folds agreeing.

    Args:
        losses: Per-row losses holding both arms, restricted to the scope wanted.
        treatment: The arm whose error is being compared.
        reference: The arm it is compared against.
        label: The scope label for the first column.

    Returns:
        The table row, matching `CONTRAST_HEADER`.
    """
    interval = bootstrap_difference(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    folds = per_fold_differences(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    same_sign = sum(np.sign(value) == np.sign(interval["difference"]) for value in folds)
    difference, lower, upper = (
        interval[key] * PERCENTAGE_POINTS for key in ("difference", "lower_95", "upper_95")
    )
    excludes = interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0
    return (
        f"| {label} | {treatment} − {reference} | {difference:+.2f} "
        f"| [{lower:+.2f}, {upper:+.2f}] | {'**yes**' if excludes else 'no'} "
        f"| {same_sign} of {len(folds)} | {interval['n_rows']:,} |"
    )


def _arms_table(*, losses: pl.DataFrame) -> list[str]:
    """Render every arm's mean error, primary setting and second setting where fitted.

    Args:
        losses: Every arm's losses, both settings.

    Returns:
        Markdown lines.
    """
    pooled = losses.filter(pl.col("setting") == "pooled")
    sensitivity = losses.filter(pl.col("setting") == "sensitivity")
    lines = [
        "#### Every arm's mean absolute error",
        "",
        "| Arm | MAE, primary setting | MAE, second setting |",
        "|---|---|---|",
    ]
    arms = (*_arm_columns(), *DERIVED_ARMS)
    for arm in arms:
        second = _mae(losses=sensitivity, arm=arm)
        lines.append(
            f"| {arm} | {_mae(losses=pooled, arm=arm):.2f} "
            f"| {'–' if np.isnan(second) else f'{second:.2f}'} |"
        )
    return lines


def _planned_lines(*, losses: pl.DataFrame) -> list[str]:
    """Render the two planned contrasts at both hyperparameter settings.

    Args:
        losses: Every arm's losses, both settings.

    Returns:
        Markdown lines.
    """
    lines = ["#### Planned contrasts", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(
            losses=losses.filter(pl.col("setting") == "pooled"),
            treatment=treatment,
            reference=reference,
            label="all",
        )
        for treatment, reference in PLANNED_CONTRASTS
    ]
    lines += [
        "",
        "#### Planned contrasts at the second hyperparameter setting",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(
            losses=losses.filter(pl.col("setting") == "sensitivity"),
            treatment=treatment,
            reference=reference,
            label="all",
        )
        for treatment, reference in PLANNED_CONTRASTS
    ]
    return lines


def _negative_control_lines(*, losses: pl.DataFrame) -> list[str]:
    """Render the negative control against `cams`.

    Args:
        losses: Every arm's losses, primary setting.

    Returns:
        Markdown lines.
    """
    treatment, reference = NEGATIVE_CONTROL
    return [
        "#### Negative control: CAMS plus a noised copy of itself, against CAMS alone",
        "",
        *CONTRAST_HEADER,
        _contrast_line(losses=losses, treatment=treatment, reference=reference, label="all"),
    ]


def _method_lines(*, losses: pl.DataFrame) -> list[str]:
    """Render the mean, stack and equal blends against `cams` (exploratory).

    Args:
        losses: Every arm's losses, primary setting.

    Returns:
        Markdown lines.
    """
    lines = [
        "#### Exploratory: the mean, stack and equal blends against CAMS",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(losses=losses, treatment=arm, reference="cams", label="all")
        for arm in ("cams_sarah3_mean", *DERIVED_ARMS)
    ]
    return lines


def _by_generator_lines(*, losses: pl.DataFrame) -> list[str]:
    """Break the two planned contrasts down by generator (exploratory).

    Args:
        losses: Every arm's losses, primary setting.

    Returns:
        Markdown lines.
    """
    lines = ["#### Exploratory: the planned contrasts by generator", "", *CONTRAST_HEADER]
    for site in sorted(losses["site"].unique().to_list()):
        rows = losses.filter(pl.col("site") == site)
        lines += [
            _contrast_line(
                losses=rows, treatment=treatment, reference=reference, label=f"site {site}"
            )
            for treatment, reference in PLANNED_CONTRASTS
        ]
    return lines


def _by_clearness_lines(*, losses: pl.DataFrame, frame: pl.DataFrame) -> list[str]:
    """Break the two planned contrasts down by clearness band (exploratory).

    Args:
        losses: Every arm's losses, primary setting.
        frame: The common rows, holding `kt`.

    Returns:
        Markdown lines.
    """
    keyed = losses.join(frame.select("site", "time", "kt"), on=["site", "time"], how="left")
    lines = [
        "#### Exploratory: the planned contrasts by clearness band (CAMS and SARAH-3's mean `kt`)",
        "",
        *CONTRAST_HEADER,
    ]
    for name, low, high in weather_products.CLEARNESS_BANDS:
        band = keyed.filter(pl.col("kt").is_between(low, high, closed="left"))
        lines += [
            _contrast_line(losses=band, treatment=treatment, reference=reference, label=name)
            for treatment, reference in PLANNED_CONTRASTS
        ]
    return lines


def _year_line(*, interval: YearInterval, treatment: str) -> str:
    """Render one year's paired difference, flagging a year with too few months.

    Args:
        interval: The interval, from `bootstrap_difference_by_year`.
        treatment: The treatment arm, for the row's label.

    Returns:
        The table row.
    """
    difference, lower, upper = (
        interval[key] * PERCENTAGE_POINTS for key in ("difference", "lower_95", "upper_95")
    )
    excludes = interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0
    months_note = "" if interval["enough_months"] else " (too few months)"
    return (
        f"| {interval['year']}{months_note} | {treatment} − {interval['reference']} "
        f"| {difference:+.2f} | [{lower:+.2f}, {upper:+.2f}] | {'**yes**' if excludes else 'no'} "
        f"| {interval['n_months']} |"
    )


def _by_year_lines(*, losses: pl.DataFrame) -> list[str]:
    """Break the two planned contrasts down by calendar year (exploratory).

    Restricted to `weather_products.ERA5_BY_YEAR_MONTHS` (January to August), so 2026's partial
    year compares against the same months of every complete year.

    Args:
        losses: Every arm's losses, primary setting.

    Returns:
        Markdown lines.
    """
    lines = [
        "#### Exploratory: the planned contrasts by calendar year (January to August only)",
        "",
        "| Year | Contrast | ΔMAE (pp of capacity) | 95% interval | Excludes zero? | Months |",
        "|---|---|---|---|---|---|",
    ]
    for treatment, reference in PLANNED_CONTRASTS:
        intervals = bootstrap_difference_by_year(
            losses=losses,
            treatment=treatment,
            references=(reference,),
            metric=METRIC,
            months=weather_products.ERA5_BY_YEAR_MONTHS,
        )
        lines += [_year_line(interval=interval, treatment=treatment) for interval in intervals]
    return lines


def _feature_lines() -> list[str]:
    """Render every fitted arm's feature columns.

    Returns:
        Markdown lines.
    """
    lines = [
        "#### Every fitted arm's feature columns",
        "",
        "| Arm | Columns | Feature columns |",
        "|---|---|---|",
    ]
    for arm, columns in _arm_columns().items():
        lines.append(f"| {arm} | {len(columns)} | " + ", ".join(f"`{c}`" for c in columns) + " |")
    lines += [
        f"| {arm} | – | derived from `{', '.join(SINGLE_ARMS)}`'s out-of-fold predictions |"
        for arm in DERIVED_ARMS
    ]
    return lines


def build_report(
    *, frame: pl.DataFrame, losses: pl.DataFrame, reproduction_lines: list[str]
) -> str:
    """Assemble the markdown report.

    Args:
        frame: The record panel's common rows.
        losses: Every arm's losses, primary and second settings.
        reproduction_lines: The reproduction check's rendered lines.

    Returns:
        The report.
    """
    pooled = losses.filter(pl.col("setting") == "pooled")
    lines = [
        "### Does blending CAMS and SARAH-3 beat CAMS alone?",
        "",
        (
            f"- {frame.height:,} common site-hours, {frame['site'].n_unique()} generators, "
            f"{frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d}, the past-solar "
            "study's `record` panel rows unchanged."
        ),
        (
            "- Mean absolute error as a percentage of each generator's own P99 output (its "
            "`effective_capacity_mw`), to 2 decimal places. A negative ΔMAE means the treatment's "
            "error is lower. Every 95% interval resamples whole months and one of the three "
            "fitting seeds, 2,000 times, and covers month-to-month weather and the fitting seed "
            "only."
        ),
        "",
        "#### Reproduction check: refitted `cams` and `sarah3` against the record panel's losses",
        "",
        *reproduction_lines,
        "",
        *_arms_table(losses=losses),
        "",
        *_planned_lines(losses=losses),
        "",
        *_negative_control_lines(losses=pooled),
        "",
        *_method_lines(losses=pooled),
        "",
        *_by_generator_lines(losses=pooled),
        "",
        *_by_clearness_lines(losses=pooled, frame=frame),
        "",
        *_by_year_lines(losses=pooled),
        "",
        *_feature_lines(),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Refit `cams` and `sarah3`, check them, fit and derive every blend, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", action="store_true", help="Reuse per-arm fits a previous run left on disk."
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Rebuild report.md from losses.parquet and reproduction.md already on disk, with no "
            "refit. Move the current outputs aside first."
        ),
    )
    arguments = parser.parse_args()
    started = datetime.now(tz=UTC)

    frame = build_rows()
    _LOG.info("record panel common rows: %d", frame.height)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loss_path = OUTPUT_DIR / "losses.parquet"
    reproduction_path = OUTPUT_DIR / "reproduction.md"
    report_path = OUTPUT_DIR / "report.md"

    if arguments.report_only:
        refuse_to_overwrite(paths=[report_path])
        losses = pl.read_parquet(loss_path)
        reproduction_lines = reproduction_path.read_text().splitlines()
    else:
        refuse_to_overwrite(paths=[loss_path, reproduction_path, report_path])
        jobs = _primary_jobs() + _sensitivity_jobs()
        fitted = _fitted(frame=frame, jobs=jobs, resume=arguments.resume)
        reproduction_rows, reproduction_lines = check_reproduction(fitted=fitted)
        gate = "\n".join(reproduction_lines) + "\n"
        reproduction_path.write_text(gate)
        sys.stdout.write(gate)
        if not all(row["bit_identical"] for row in reproduction_rows):
            _LOG.error("cams and sarah3 do not reproduce the record panel's losses; stopping")
            return 1
        derived = stack_and_equal(fitted=fitted)
        losses = pl.concat([fitted, derived], how="vertical_relaxed")
        losses.write_parquet(loss_path)
        shutil.rmtree(OUTPUT_DIR / FITS_DIR_NAME, ignore_errors=True)

    report = build_report(frame=frame, losses=losses, reproduction_lines=reproduction_lines)
    report_path.write_text(report)
    sys.stdout.write(report)
    _LOG.info("finished in %s", datetime.now(tz=UTC) - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
