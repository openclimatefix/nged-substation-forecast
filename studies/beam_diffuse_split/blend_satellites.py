"""Measure whether blending CAMS and SARAH-3 beats CAMS's own split.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/860>. It reuses the past-solar
study's `record` panel (`weather_products.PANELS["record"]`): the common rows of ERA5, CAMS, SARAH-3
and ICON-DREAM-EU, 2021 to August 2026, with the same folds, eras, seeds, export cap and
`run_experiment.SHARED_FEATURES` the panel already uses. It builds no row set of its own.

**Pre-registered design, fixed before any result exists.**

- **Reproduction guard, run before any blend is fitted.** The `cams`, `sarah3` and `cams_split`
  arms here are refitted exactly as the record panel's `cams_global`, `sarah3_global` and
  `cams_split` arms: the same feature columns, on the same rows, at the same hyperparameters. Each
  is checked row for row against the record panel's saved losses at
  `data/studies/beam_diffuse_split/past_weather_v2/solar_record/losses.parquet` — the same
  `(site, time, fold, seed)` keys and a bit-identical `signed_error_capped_mw`. The run stops before
  fitting any blend if any of the three differs.
- **Arms.** SARAH-3 is scored on global irradiance only (its own split is a separation model's
  output, not a retrieval, so `weather_products.py` never fits a SARAH-3 split arm), so every blend
  here reads CAMS's split (or global) columns plus SARAH-3's global irradiance, and nothing else of
  either product:
    - `cams`, `sarah3`, `cams_split`: the single-product arms, refitted for the guard. `cams_split`
      carries CAMS's own beam and diffuse columns alongside its global irradiance, exactly as the
      record panel's `cams_split` arm does.
    - `cams_split_sarah3_xgb`: CAMS's split columns plus SARAH-3's global irradiance.
    - `cams_split_sarah3_control`: the same columns, with SARAH-3's column permuted within site,
      month and hour of day (`studies.blending.climatology_permutation`), so the control keeps
      SARAH-3's climatology but loses its weather.
    - `cams_sarah3_xgb`, `cams_sarah3_control`: the same blend and control, built on CAMS's global
      irradiance instead of its split, so the two products contribute the same kind of column.
      Exploratory: they give the two products a matched comparison, but do not answer whether a
      blend beats CAMS's own split.
    - `cams_sarah3_mean`, `cams_sarah3_stack`, `cams_sarah3_equal`: the mean of the two products'
      global irradiance, the cross-fitted linear stack, and the equal-weight mean of the two
      single-product models' out-of-fold predictions (`studies.blending.stacked_errors`), derived
      from the fitted global losses with no refit. Built on the global columns only, since SARAH-3
      has no split arm to combine with CAMS's own. Exploratory.
    - `cams_cams_noise_xgb`: the primary negative control. CAMS's real column plus a second copy of
      it with Gaussian noise added, its standard deviation the RMS difference between CAMS's and
      SARAH-3's global irradiance on these rows, measured at run time and printed in the report, so
      the control's two columns differ from each other about as much as CAMS and SARAH-3 genuinely
      do. It checks that a second, comparably-different column does not by itself make a blend look
      better, and is reported against `cams` alone.
    - `cams_cams_noise5_xgb`: a second, cheaper negative control, noised at a fixed 5 W/m2, close to
      a duplicate column. Exploratory.
- **Planned contrasts**, named before any run: `cams_split_sarah3_xgb` − `cams_split` (does the
  blend beat CAMS's best single-product arm?) and `cams_split_sarah3_xgb` −
  `cams_split_sarah3_control` (does the gain come from SARAH-3's information, or only from having a
  second column?). Both are refitted at `studies.cross_validation.SENSITIVITY_HYPER_PARAMETERS`
  too, so the report can say whether an ordering belongs to the features or to the settings. Every
  other contrast in the report is exploratory: the all-global blend and its control against `cams`;
  the mean, stack and equal blends against `cams`; the two planned contrasts broken down by
  generator, by `weather_products.CLEARNESS_BANDS`, and by calendar year
  (`studies.bootstrap.bootstrap_difference_by_year`, restricted to `weather_products.
  ERA5_BY_YEAR_MONTHS` so a partial final year compares on the same months as a complete one); and
  both negative controls against `cams`.
- **Intervals.** Every contrast is bootstrapped with `studies.bootstrap.bootstrap_difference`,
  resampling whole months and a fitting seed, 2,000 times. Every arm's own absolute error is
  bootstrapped the same way with `studies.bootstrap.bootstrap_absolute`. The report prints every
  figure as a percentage of capacity to 2 decimal places.

Run it with `uv run python studies/beam_diffuse_split/blend_satellites.py`, after
`weather_products.py` has written the `record` panel (`--panel record`). `--resume` reuses the
per-arm fits a previous run left in `fits/`, refusing to reuse one whose rows, feature-column
values, fold assignment, target or hyperparameters have since changed (checked against a
fingerprint saved beside each fit). `--report-only` rebuilds `report.md` from `losses.parquet` and
`reproduction.md` already on disk, fitting nothing, and stops if the rebuilt rows do not match the
keys `losses.parquet` was fitted on; move the current outputs to a `superseded/` subfolder first,
since neither mode overwrites a file.
"""

import argparse
import hashlib
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
    bootstrap_absolute,
    bootstrap_difference,
    bootstrap_difference_by_year,
    per_fold_differences,
)
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SEEDS,
    SENSITIVITY_HYPER_PARAMETERS,
    HyperParameters,
)
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
"""Seeds `cams_sarah3_control`'s and `cams_split_sarah3_control`'s permutation of SARAH-3's global
irradiance."""

PERMUTED_UNCHANGED_FRACTION_LIMIT: Final[float] = 0.9
"""`_check_control_permutation` raises if the permuted column matches the real one on at least this
fraction of rows, which would mean the permutation barely moved anything."""

MEAN_COLUMN: Final[str] = "ghi_mean_cams_sarah3"
NOISE_COLUMN: Final[str] = "ghi_cams_noise_rms"
NOISE_COLUMN_NEAR_DUPLICATE: Final[str] = "ghi_cams_noise5"

NEGATIVE_CONTROL_NOISE_SEED: Final[int] = 20260928
"""Seeds both negative controls' noise, drawn once per row in the frame's fixed row order."""

NEAR_DUPLICATE_NOISE_STD_W_M2: Final[float] = 5.0
"""`cams_cams_noise5_xgb`'s fixed noise standard deviation.

CAMS's global irradiance runs from 0 to about 1,000 W/m2 at midday, so 5 W/m2 keeps the noised copy
almost identical to CAMS's own column: the arm carries two columns but very nearly one product's
information. `cams_cams_noise_xgb`'s noise, by contrast, is set from the measured CAMS/SARAH-3 RMS
difference (`_cams_sarah3_rms_difference`), so its two columns differ from each other about as much
as the two real products do."""

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

IDENTITY_KEYS: Final[tuple[str, str, str, str]] = ("site", "time", "fold", "seed")
"""The keys that must match, arm for arm, before a contrast's inner join can trust its rows."""

LOSS_COLUMNS: Final[tuple[str, ...]] = (
    *KEY_COLUMNS,
    "signed_error_capped_mw",
    METRIC,
    "arm",
    "setting",
)
"""The columns `losses.parquet` keeps for every arm, fitted or derived."""

SINGLE_ARMS: Final[tuple[str, str]] = ("cams", "sarah3")
"""The single-product global arms the mean, stack and equal blends are derived from."""

GUARD_ARMS: Final[tuple[str, str, str]] = ("cams", "sarah3", "cams_split")
"""The single-product arms fit and checked against the record panel before any blend is fitted."""

PUBLISHED_ARM_NAMES: Final[dict[str, str]] = {
    "cams": "cams_global",
    "sarah3": "sarah3_global",
    "cams_split": "cams_split",
}
"""Each guard arm's name in the record panel's own saved losses."""

PLANNED_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("cams_split_sarah3_xgb", "cams_split"),
    ("cams_split_sarah3_xgb", "cams_split_sarah3_control"),
)
"""The only contrasts a recommendation may rest on, named before any result exists."""

SENSITIVITY_ARMS: Final[tuple[str, str, str]] = (
    "cams_split",
    "cams_split_sarah3_xgb",
    "cams_split_sarah3_control",
)
"""The arms refitted at `SENSITIVITY_HYPER_PARAMETERS`, so both planned contrasts can be checked."""

NEGATIVE_CONTROLS: Final[tuple[tuple[str, str], ...]] = (
    ("cams_cams_noise_xgb", "cams"),
    ("cams_cams_noise5_xgb", "cams"),
)
"""Both negative controls, against `cams`: the primary one noised at the measured CAMS/SARAH-3 RMS
difference, and the cheap second one noised at a fixed 5 W/m2."""

DERIVED_ARMS: Final[tuple[str, str]] = ("cams_sarah3_stack", "cams_sarah3_equal")
"""Arms derived from the fitted single-product global losses, with no XGBoost fit of their own."""

EXPLORATORY_METHOD_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("cams_sarah3_xgb", "cams"),
    ("cams_sarah3_xgb", "cams_sarah3_control"),
    ("cams_sarah3_mean", "cams"),
    (DERIVED_ARMS[0], "cams"),
    (DERIVED_ARMS[1], "cams"),
)
"""The all-global blend and its control, and the mean, stack and equal blends, all against `cams`.
Exploratory: none of these decides whether a blend beats CAMS's own split."""

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
        "cams_split": (*SHARED, "ghi_cams", "bhi_cams", "dhi_cams"),
        "cams_split_sarah3_xgb": (*SHARED, "ghi_cams", "bhi_cams", "dhi_cams", "ghi_sarah3"),
        "cams_split_sarah3_control": (
            *SHARED,
            "ghi_cams",
            "bhi_cams",
            "dhi_cams",
            "ghi_sarah3_shuffled",
        ),
        "cams_sarah3_xgb": (*SHARED, "ghi_cams", "ghi_sarah3"),
        "cams_sarah3_control": (*SHARED, "ghi_cams", "ghi_sarah3_shuffled"),
        "cams_sarah3_mean": (*SHARED, MEAN_COLUMN),
        "cams_cams_noise_xgb": (*SHARED, "ghi_cams", NOISE_COLUMN),
        "cams_cams_noise5_xgb": (*SHARED, "ghi_cams", NOISE_COLUMN_NEAR_DUPLICATE),
    }


def _guard_jobs() -> list[Job]:
    """Return the single-product arms' fits, at the primary hyperparameters.

    Returns:
        The jobs `run_experiment.run_all` takes.
    """
    columns = _arm_columns()
    return [
        (arm, "pooled", "power_mw", columns[arm], PRIMARY_HYPER_PARAMETERS, False)
        for arm in GUARD_ARMS
    ]


def _blend_jobs() -> list[Job]:
    """Return every non-guard arm's fit at the primary hyperparameters.

    Returns:
        The jobs `run_experiment.run_all` takes.
    """
    columns = _arm_columns()
    return [
        (arm, "pooled", "power_mw", columns[arm], PRIMARY_HYPER_PARAMETERS, False)
        for arm in columns
        if arm not in GUARD_ARMS
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


def _cams_sarah3_rms_difference(*, frame: pl.DataFrame) -> float:
    """Return the RMS difference between CAMS's and SARAH-3's global irradiance on these rows.

    Sets `cams_cams_noise_xgb`'s noise standard deviation, so its two columns differ from each
    other about as much as CAMS and SARAH-3 genuinely do, rather than by a value picked by hand.

    Args:
        frame: The common rows, carrying `ghi_cams` and `ghi_sarah3`.

    Returns:
        The RMS difference, in W/m2.
    """
    return float(
        frame.select(((pl.col("ghi_cams") - pl.col("ghi_sarah3")) ** 2).mean().sqrt()).item()
    )


def _check_control_permutation(*, frame: pl.DataFrame) -> None:
    """Raise unless SARAH-3's permuted column is a genuine within-group permutation of the real one.

    Guards the control arms' whole purpose: if the permutation crossed a site, month or hour-of-day
    boundary, or barely moved any row, a control would keep some of SARAH-3's real weather and
    understate the size of gain a merely-redundant column can produce.

    Args:
        frame: The common rows, carrying `ghi_sarah3`, `ghi_sarah3_shuffled`, `site`, `month` and
            `hour_of_day`.

    Raises:
        ValueError: If any (site, month, hour of day) group's permuted values are not a permutation
            of its real ones, or if the permuted column matches the real one on
            `PERMUTED_UNCHANGED_FRACTION_LIMIT` or more of the rows.
    """
    groups = frame.group_by("site", "month", "hour_of_day").agg(
        is_permutation=(pl.col("ghi_sarah3").sort() == pl.col("ghi_sarah3_shuffled").sort()).all()
    )
    if not bool(groups["is_permutation"].all()):
        msg = "ghi_sarah3_shuffled is not a within-(site, month, hour) permutation of ghi_sarah3"
        raise ValueError(msg)
    unchanged_fraction = float(
        frame.select((pl.col("ghi_sarah3_shuffled") == pl.col("ghi_sarah3")).mean()).item()
    )
    if unchanged_fraction >= PERMUTED_UNCHANGED_FRACTION_LIMIT:
        msg = (
            f"ghi_sarah3_shuffled matches ghi_sarah3 on {unchanged_fraction:.0%} of rows; the "
            "permutation barely moved any row"
        )
        raise ValueError(msg)


def _check_control_arms_read_the_right_column() -> None:
    """Raise unless every `_control` arm reads the permuted SARAH-3 column and no other arm does.

    A blend arm accidentally reading `ghi_sarah3_shuffled`, or a control arm accidentally reading
    the real `ghi_sarah3`, would swap a planned contrast's treatment and control without either
    arm's name changing.

    Raises:
        ValueError: Naming the arm, if a `_control` arm's columns hold the real `ghi_sarah3`, or a
            non-control arm's columns hold the permuted `ghi_sarah3_shuffled`.
    """
    for arm, columns in _arm_columns().items():
        is_control = arm.endswith("_control")
        if is_control and "ghi_sarah3" in columns:
            msg = f"control arm {arm!r} reads the real ghi_sarah3 column"
            raise ValueError(msg)
        if not is_control and "ghi_sarah3_shuffled" in columns:
            msg = f"non-control arm {arm!r} reads the permuted ghi_sarah3_shuffled column"
            raise ValueError(msg)


def build_rows() -> tuple[pl.DataFrame, float]:
    """Build the record panel's common rows, exactly as `weather_products.run_panel` builds them.

    Adds the columns this study's arms need beyond the record panel's own: SARAH-3's
    climatology-permuted copy, two noised copies of CAMS, the mean of CAMS and SARAH-3, and the
    clearness index the exploratory breakdown bins on.

    Returns:
        One row per common site-hour, carrying every column `_arm_columns` names, and the measured
        RMS difference between CAMS's and SARAH-3's global irradiance on these rows.
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
    _check_control_permutation(frame=frame)
    _check_control_arms_read_the_right_column()
    rms_difference = _cams_sarah3_rms_difference(frame=frame)
    generator = np.random.default_rng(NEGATIVE_CONTROL_NOISE_SEED)
    noise = generator.normal(scale=rms_difference, size=frame.height)
    noise_near_duplicate = generator.normal(scale=NEAR_DUPLICATE_NOISE_STD_W_M2, size=frame.height)
    frame = frame.with_columns(
        (pl.col("ghi_cams") + pl.Series(noise)).alias(NOISE_COLUMN),
        (pl.col("ghi_cams") + pl.Series(noise_near_duplicate)).alias(NOISE_COLUMN_NEAR_DUPLICATE),
        pl.mean_horizontal("ghi_cams", "ghi_sarah3").alias(MEAN_COLUMN),
    ).with_columns(
        kt=pl.when(pl.col("extraterrestrial_horizontal_w_m2") > 0)
        .then(pl.col(MEAN_COLUMN) / pl.col("extraterrestrial_horizontal_w_m2"))
        .otherwise(None)
    )
    check_no_missing(
        frame=frame, columns=[column for columns in _arm_columns().values() for column in columns]
    )
    return frame, rms_difference


def _fit_path(*, arm: str, setting: str) -> Path:
    """Return where one arm's fit is kept until the report is written.

    Args:
        arm: The arm.
        setting: The setting.

    Returns:
        The parquet path.
    """
    return OUTPUT_DIR / FITS_DIR_NAME / f"{setting}__{arm}.parquet"


def _fingerprint_path(*, arm: str, setting: str) -> Path:
    """Return where one arm's fingerprint is kept, beside its fit.

    Args:
        arm: The arm.
        setting: The setting.

    Returns:
        The path.
    """
    return _fit_path(arm=arm, setting=setting).with_suffix(".fingerprint")


def _fingerprint(
    *, frame: pl.DataFrame, columns: tuple[str, ...], target: str, hyperparameters: HyperParameters
) -> str:
    """Return a hash of a job's rows, feature-column values, fold, target and hyperparameters.

    `--resume` compares this against the fingerprint saved beside a previous run's fit, so a run
    whose rows, an arm's feature-column values, fold assignment, target or hyperparameters have
    since changed refuses to reuse the stale fit rather than silently mixing it into a report the
    code no longer matches.

    Args:
        frame: The rows the job is fitted on.
        columns: The job's feature columns, in fit order.
        target: The column the job predicts.
        hyperparameters: The hyperparameters the job fits at.

    Returns:
        A hex digest, independent of the frame's row order.
    """
    row_hashes = (
        frame.select("site", "time", "fold", *columns).hash_rows(seed=0).sort().to_numpy().tobytes()
    )
    digest = hashlib.sha256(row_hashes)
    digest.update("|".join(columns).encode())
    digest.update(str(SEEDS).encode())
    digest.update(target.encode())
    digest.update(str(hyperparameters).encode())
    return digest.hexdigest()


def _fitted(*, frame: pl.DataFrame, jobs: list[Job], resume: bool) -> pl.DataFrame:
    """Fit every job not already on disk with a matching fingerprint, and return every job's losses.

    Args:
        frame: The record panel's common rows.
        jobs: The fits wanted.
        resume: Whether to reuse a fit a previous run left on disk.

    Returns:
        Every job's losses in `LOSS_COLUMNS`, one row per (site, time, seed, arm, setting).

    Raises:
        ValueError: If `--resume` finds a fit on disk whose saved fingerprint does not match this
            run's rows, feature-column values, fold assignment, target or hyperparameters.
    """
    missing: list[Job] = []
    for job in jobs:
        arm, setting, target, columns, hyperparameters, *_ = job
        path = _fit_path(arm=arm, setting=setting)
        if resume and path.exists():
            fingerprint_path = _fingerprint_path(arm=arm, setting=setting)
            saved = fingerprint_path.read_text().strip() if fingerprint_path.exists() else None
            fingerprint = _fingerprint(
                frame=frame, columns=columns, target=target, hyperparameters=hyperparameters
            )
            if saved != fingerprint:
                msg = (
                    f"--resume: {path} was fitted on different rows, columns, fold assignment, "
                    "target or hyperparameters than this run would use; move it to a superseded/ "
                    "subfolder or re-run without --resume"
                )
                raise ValueError(msg)
        else:
            missing.append(job)
    _LOG.info("%d of %d fits to run", len(missing), len(jobs))
    if missing:
        fresh = run_all(dataset=frame, jobs=missing)
        for arm, setting, target, columns, hyperparameters, *_ in missing:
            path = _fit_path(arm=arm, setting=setting)
            path.parent.mkdir(parents=True, exist_ok=True)
            fresh.filter(pl.col("arm") == arm, pl.col("setting") == setting).sort(
                "site", "time", "seed"
            ).write_parquet(path)
            _fingerprint_path(arm=arm, setting=setting).write_text(
                _fingerprint(
                    frame=frame, columns=columns, target=target, hyperparameters=hyperparameters
                )
            )
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
    """Compare the refitted guard arms with the record panel's saved losses.

    Args:
        fitted: This study's fitted losses, holding `GUARD_ARMS` at setting `pooled`.

    Returns:
        One row per guard arm, and the same rows rendered as a markdown table.
    """
    published = pl.read_parquet(RECORD_PANEL_LOSSES)
    keys = ["site", "time", "fold", "seed"]
    rows: list[ReproductionRow] = []
    for arm in GUARD_ARMS:
        published_arm = PUBLISHED_ARM_NAMES[arm]
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


def _assert_arms_share_keys(*, losses: pl.DataFrame) -> None:
    """Raise unless every arm holds exactly the reference arm's (site, time, fold, seed) keys.

    A contrast's paired difference joins two arms on these keys with an inner join
    (`studies.bootstrap.paired_differences`), which drops any row one side lacks with no warning.
    This is the check that keeps that join from ever doing so: `cams` is the reference for the
    pooled setting, and `cams_split` is the reference for the sensitivity setting, since `cams` is
    not refit there.

    Args:
        losses: Every arm's fitted and derived losses, both settings.

    Raises:
        ValueError: Naming the arm, the setting, and both row counts, when an arm's keys differ
            from its setting's reference.
    """
    references = {"pooled": "cams", "sensitivity": "cams_split"}
    for setting, reference_arm in references.items():
        setting_losses = losses.filter(pl.col("setting") == setting)
        if setting_losses.is_empty():
            continue
        reference_keys = (
            setting_losses.filter(pl.col("arm") == reference_arm)
            .select(*IDENTITY_KEYS)
            .sort(*IDENTITY_KEYS)
        )
        for arm in setting_losses["arm"].unique().to_list():
            if arm == reference_arm:
                continue
            arm_keys = (
                setting_losses.filter(pl.col("arm") == arm)
                .select(*IDENTITY_KEYS)
                .sort(*IDENTITY_KEYS)
            )
            if not arm_keys.equals(reference_keys):
                msg = (
                    f"arm {arm!r} at setting {setting!r} holds {arm_keys.height:,} "
                    f"(site, time, fold, seed) keys; {reference_arm!r} holds "
                    f"{reference_keys.height:,}; a contrast's inner join would silently drop rows"
                )
                raise ValueError(msg)


def _assert_rows_match_losses(
    *, frame: pl.DataFrame, losses: pl.DataFrame, loss_path: Path
) -> None:
    """Raise unless the rebuilt rows hold exactly the (site, time) keys `losses` was fitted on.

    `--report-only` trusts `losses.parquet` to still match the current code's rows; this is the
    check that trust rests on, rather than assuming a saved file still matches a script that may
    have since changed.

    Args:
        frame: The rebuilt common rows.
        losses: The saved losses `--report-only` is building a report from.
        loss_path: Where `losses` was read from, named in the error.

    Raises:
        ValueError: Naming both row counts, when the rebuilt rows differ from `cams`'s saved keys.
    """
    saved_keys = (
        losses.filter(pl.col("arm") == "cams", pl.col("setting") == "pooled")
        .select("site", "time")
        .unique()
        .sort("site", "time")
    )
    rebuilt_keys = frame.select("site", "time").unique().sort("site", "time")
    if not saved_keys.equals(rebuilt_keys):
        msg = (
            f"--report-only: the rebuilt rows ({rebuilt_keys.height:,}) do not match the keys "
            f"{loss_path} was fitted on ({saved_keys.height:,}); the saved losses are stale, "
            "rebuild them with a full run"
        )
        raise ValueError(msg)


def _absolute_cell(*, losses: pl.DataFrame, arm: str) -> str:
    """Render one arm's own absolute error and its 95% interval, in points of capacity.

    Args:
        losses: Per-row losses, already restricted to one setting.
        arm: The arm.

    Returns:
        `"MAE [lower, upper]"`, bootstrapped with `studies.bootstrap.bootstrap_absolute`, or `"–"`
        if the arm was not fitted at this setting.
    """
    if losses.filter(pl.col("arm") == arm).is_empty():
        return "–"
    interval = bootstrap_absolute(losses=losses, arm=arm, metric=METRIC)
    value, lower, upper = (
        interval[key] * PERCENTAGE_POINTS for key in ("value", "lower_95", "upper_95")
    )
    return f"{value:.2f} [{lower:.2f}, {upper:.2f}]"


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
        (
            "Each arm's own absolute error, with its own 95% interval from resampling whole "
            "months and a seed (`studies.bootstrap.bootstrap_absolute`). The interval is wide "
            "mainly because every arm's error rises and falls together from month to month; the "
            "paired contrasts below resample the same months for both arms, which cancels that "
            "shared swing."
        ),
        "",
        "| Arm | MAE, primary setting (95% interval) | MAE, second setting (95% interval) |",
        "|---|---|---|",
    ]
    arms = (*_arm_columns(), *DERIVED_ARMS)
    lines += [
        f"| {arm} | {_absolute_cell(losses=pooled, arm=arm)} "
        f"| {_absolute_cell(losses=sensitivity, arm=arm)} |"
        for arm in arms
    ]
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


def _negative_control_lines(*, losses: pl.DataFrame, rms_difference: float) -> list[str]:
    """Render both negative controls against `cams`.

    Args:
        losses: Every arm's losses, primary setting.
        rms_difference: The measured RMS difference between CAMS's and SARAH-3's global
            irradiance, which sets the primary control's noise standard deviation.

    Returns:
        Markdown lines.
    """
    lines = [
        "#### Negative controls: CAMS plus a noised copy of itself, against CAMS alone",
        "",
        (
            f"- Primary control (`cams_cams_noise_xgb`): noise standard deviation "
            f"{rms_difference:.1f} W/m2, the measured RMS difference between CAMS's and SARAH-3's "
            "global irradiance on these rows."
        ),
        (
            f"- Second control (`cams_cams_noise5_xgb`): noise standard deviation "
            f"{NEAR_DUPLICATE_NOISE_STD_W_M2:.1f} W/m2, close to a duplicate column."
        ),
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(losses=losses, treatment=treatment, reference=reference, label="all")
        for treatment, reference in NEGATIVE_CONTROLS
    ]
    return lines


def _method_lines(*, losses: pl.DataFrame) -> list[str]:
    """Render the all-global blend and its control, and the mean, stack and equal blends.

    All exploratory, all against `cams`. The mean, stack and equal blends are built on the global
    columns only, since SARAH-3 has no split.

    Args:
        losses: Every arm's losses, primary setting.

    Returns:
        Markdown lines.
    """
    lines = [
        (
            "#### Exploratory: the all-global blend, and the mean, stack and equal blends, "
            "against CAMS"
        ),
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(losses=losses, treatment=treatment, reference=reference, label="all")
        for treatment, reference in EXPLORATORY_METHOD_CONTRASTS
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
    *,
    frame: pl.DataFrame,
    losses: pl.DataFrame,
    reproduction_lines: list[str],
    rms_difference: float,
) -> str:
    """Assemble the markdown report.

    Args:
        frame: The record panel's common rows.
        losses: Every arm's losses, primary and second settings.
        reproduction_lines: The reproduction check's rendered lines.
        rms_difference: The measured RMS difference between CAMS's and SARAH-3's global
            irradiance, printed alongside the negative controls.

    Returns:
        The report.
    """
    pooled = losses.filter(pl.col("setting") == "pooled")
    lines = [
        "### Does blending CAMS and SARAH-3 beat CAMS's own split?",
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
        (
            f"- CAMS's and SARAH-3's global irradiance differ by {rms_difference:.1f} W/m2 RMS on "
            "these rows; that value sets the primary negative control's noise (below)."
        ),
        "",
        (
            "#### Reproduction check: refitted `" + "`, `".join(GUARD_ARMS) + "` against the "
            "record panel's losses"
        ),
        "",
        *reproduction_lines,
        "",
        *_arms_table(losses=losses),
        "",
        *_planned_lines(losses=losses),
        "",
        *_negative_control_lines(losses=pooled, rms_difference=rms_difference),
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
    """Fit and guard the single-product arms, fit and derive every blend, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse per-arm fits a previous run left on disk, refusing one whose saved fingerprint "
            "no longer matches this run's rows, columns, fold assignment, target or "
            "hyperparameters."
        ),
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

    frame, rms_difference = build_rows()
    _LOG.info("record panel common rows: %d", frame.height)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loss_path = OUTPUT_DIR / "losses.parquet"
    reproduction_path = OUTPUT_DIR / "reproduction.md"
    report_path = OUTPUT_DIR / "report.md"

    if arguments.report_only:
        refuse_to_overwrite(paths=[report_path])
        losses = pl.read_parquet(loss_path)
        _assert_rows_match_losses(frame=frame, losses=losses, loss_path=loss_path)
        _assert_arms_share_keys(losses=losses)
        reproduction_lines = reproduction_path.read_text().splitlines()
    else:
        refuse_to_overwrite(paths=[loss_path, reproduction_path, report_path])
        guard_fitted = _fitted(frame=frame, jobs=_guard_jobs(), resume=arguments.resume)
        reproduction_rows, reproduction_lines = check_reproduction(fitted=guard_fitted)
        gate = "\n".join(reproduction_lines) + "\n"
        reproduction_path.write_text(gate)
        sys.stdout.write(gate)
        if not all(row["bit_identical"] for row in reproduction_rows):
            _LOG.error(
                "%s do not reproduce the record panel's losses; stopping", ", ".join(GUARD_ARMS)
            )
            return 1
        rest_fitted = _fitted(
            frame=frame, jobs=_blend_jobs() + _sensitivity_jobs(), resume=arguments.resume
        )
        fitted = pl.concat([guard_fitted, rest_fitted])
        derived = stack_and_equal(fitted=fitted)
        losses = pl.concat([fitted, derived], how="vertical_relaxed")
        _assert_arms_share_keys(losses=losses)
        losses.write_parquet(loss_path)
        shutil.rmtree(OUTPUT_DIR / FITS_DIR_NAME, ignore_errors=True)

    report = build_report(
        frame=frame,
        losses=losses,
        reproduction_lines=reproduction_lines,
        rms_difference=rms_difference,
    )
    report_path.write_text(report)
    sys.stdout.write(report)
    _LOG.info("finished in %s", datetime.now(tz=UTC) - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
