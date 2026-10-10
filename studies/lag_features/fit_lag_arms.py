"""Fit every arm of the lagged-power-features study, out of fold, and save the per-row losses.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/1138>. It reads the frames
`build_lag_frame.py` wrote and fits them in the order below. **Nothing is fitted twice**: phase 1
(the sweep) and phase 2 (the rigorous comparison) are two views of one table of per-row losses.

**Stages, for `--weather-product ens_mean`:**

1. Stage 1. One XGBoost model per plant and per pair of withheld folds predicts power from the
   baseline arm B0's seven columns, so that each prediction withholds both the scored fold and the
   predicted hour's own fold (an hour outside the shared rows withholds the scored fold alone). The
   predictions give the two-step arms S2 and S3, and the ratio R1 uses, their `{fold}` columns.
2. The sweep: every arm of `SWEEP_ARMS` at lead-day 1, per plant, at the primary setting, three
   seeds, five folds. B0, L1, and N2 are fitted with their quantile models at once.
3. The shortlist rule picks X from the sweep's losses on the screening months.
4. Phase 2: the sensitivity setting for `PHASE2_POINT_ARMS`, the quantile models for the arms the
   plan names, the global model for `GLOBAL_ARMS`, and the positive controls.
5. The longer leads: `LONGER_LEAD_ARMS` at the other lead-days, point model, primary setting.

For `--weather-product ifs_single`, only B0 and L1 at lead-days 1 and 2 (an exploratory replicate).

**The decision rules, written into the plan before any fit.** The plan's wording is kept in the
constants' docstrings below, and nothing here changes it.

**Checkpoints.** Each (scope, setting, arm) is saved to `<output>/checkpoints/` when it finishes,
and a rerun skips the ones already saved, so a reboot costs one arm-setting. The final
`losses_<product>.parquet` is written once, when everything is done, and the script refuses to
overwrite it.

**Device.** `--device cuda` (the default) or `cpu`; `--max-workers` (default 4) sets how many plants
fit at once, each on `THREADS_PER_FIT` cores. A run writes `checkpoints/run_manifest.json` with its
device, and a resume on another device raises, because one device must serve each planned contrast.

**Smoke test.** `--smoke` runs the real path (the sweep with batching, the shortlist, phase 2, the
controls, the longer leads, the intervals, the importance refit and the global and
leave-one-plant-out fits) on 20 seeded random rows of each plant's month, with 20 boosting rounds,
to check the plumbing in a few minutes. It writes `checkpoints_smoke/` and
`losses_<product>_smoke.parquet`, and is not a result. `report_lag_features.py --smoke` and
`lag_features_charts.py --smoke` read its output.

Run it with `uv run python studies/lag_features/fit_lag_arms.py`.
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import polars as pl
import xgboost as xgb
from build_lag_frame import (
    B0_COLUMNS,
    CLOCK_RATIO_TEMPLATE,
    ENS_MEAN_LEAD_DAYS,
    FULL_SWEEP_LEAD_DAY,
    GLOBAL_ONLY_ARMS,
    IFS_LEAD_DAYS,
    LAG_FEATURES_DIR,
    LONGER_LEAD_ARMS,
    NWP_ERA_FOLD_OFFSETS,
    NWP_ERA_START_MONTHS,
    POSITIVE_CONTROL_SHIFTS,
    POWER_COLUMN_PREFIXES,
    RATIO_TEMPLATE,
    REPLICATE_ARMS,
    RESIDUAL_LAG_TEMPLATE,
    RESIDUAL_MEAN_TEMPLATES,
    STAGE1_LAG_TEMPLATE,
    STAGE1_TARGET_TEMPLATE,
    SWEEP_ARMS,
    TARGET,
    WEATHER_PRODUCTS,
    WeatherProduct,
    assert_before_cutoff,
    extra_columns_of,
    final_test_cutoff,
    output_paths,
    run_suffix,
    shared_rows,
    smoke_subsample,
    target_date,
    weather_table,
    write_parquet_atomic,
)
from studies.arm_runner import Job, run_all
from studies.baselines import issue_time, same_clock_hour_window
from studies.cross_validation import (
    N_FOLDS,
    PRIMARY_HYPER_PARAMETERS,
    QUANTILE_LEVELS,
    SEEDS,
    SENSITIVITY_HYPER_PARAMETERS,
    DeviceType,
    HyperParameters,
    booster_parameters,
    calendar_month_coverage,
    clamp_to_cap,
    cut_eras,
    fit_one_fold,
    out_of_fold_losses,
    raise_on_uncovered_months,
    score_prediction,
)
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fit_lag_arms")

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss the shortlist rule ranks arms on: the clamped error over the plant's own capacity."""

SCREENING_MONTHS: Final[tuple[str, ...]] = (
    "2024-12",
    *(f"2025-{month:02d}" for month in range(1, 10)),
)
"""Phase 1 reads the first 10 calendar months, 2024-12 to 2025-09, and nothing else."""

SHORTLIST_CANDIDATES: Final[tuple[str, ...]] = (
    "IM",
    "CTX7",
    "W7",
    "Q30",
    "CK",
    "TF",
    "AN",
    "PC",
    "S3",
    "RP",
    "KS",
)
"""**The shortlist rule, fixed before any fit:** X is the candidate with the lowest phase-1 mean
absolute error over `SCREENING_MONTHS`. The excluded arms are B0, L1, L2, S2, N1 and N2 (phase 2
carries those regardless), T1 (an interpolation bound), the references and post-model corrections
(none is fitted), and the global-only arms. Phase 2 carries X."""

REPRODUCTION_MAE_PERCENT: Final[float] = 8.771
"""The published mean absolute error of the ENS-mean B0 at lead-day 1 on the 35,263 shared rows, in
percentage points of capacity, from a GPU fit."""

REPRODUCTION_TOLERANCE_PERCENT: Final[float] = 0.03
"""How far the refit's mean may differ from the published figure. Device differences are smaller."""

WIDE_X_COLUMNS: Final[int] = 3
"""If X adds more columns than this, N2-k is fitted with as many random lags as X adds."""

REPRODUCTION_CPU_MAE_PERCENT: Final[float] = 8.766
"""The mean absolute error of the matched-lead study's saved CPU losses for the ENS-mean B0 at
lead-day 1 on the 35,263 shared rows, in percentage points of capacity."""

REPRODUCTION_GPU_MAE_PERCENT: Final[float] = 8.771
"""The published GPU refit's mean absolute error for the same arm. The check reports a GPU leg's
difference from it and does not assert it, because a GPU fit is not bit-identical."""

REPRODUCTION_TOLERANCE_PERCENT: Final[float] = 0.0005
"""How far a refit's mean may differ from the published figure, which is quoted to 3 decimals."""

PHASE2_POINT_ARMS: Final[tuple[str, ...]] = ("B0", "L1", "L2", "S2", "X", "N1", "N2")
"""The arms fitted at both hyperparameter settings; `X` stands for the shortlist rule's arm."""

SWEEP_QUANTILE_ARMS: Final[tuple[str, ...]] = ("B0", "L1", "N2")
"""Arms fitted with their quantile models in the sweep itself, so the point fit is not repeated."""

QUANTILE_PRIMARY_ARMS: Final[tuple[str, ...]] = ("B0", "L1", "X", "N2")
"""Arms whose quantile models are fitted at the primary setting."""

QUANTILE_SENSITIVITY_ARMS: Final[tuple[str, ...]] = ("B0", "L1")
"""Arms whose quantile models are fitted at the sensitivity setting."""

GLOBAL_ARMS: Final[tuple[str, ...]] = ("B0", "L1", "X")
"""Arms fitted as one XGBoost model across the six plants, at both settings."""

GLOBAL_QUANTILE_ARMS: Final[tuple[str, ...]] = ("B0", "L1")
"""Arms whose global quantile models are fitted, at both settings."""

LEAVE_ONE_PLANT_OUT_ARMS: Final[tuple[str, ...]] = ("B0", "L1", "G-FP")
"""Arms fitted leaving each plant out in turn (exploratory). In the global scope, `B0` and `L1` are
the plan's G-B0 and G-L1, and `G-ID` and `G-FP` add the plant code and the fingerprint pack."""

CONTROL_ARMS: Final[tuple[str, ...]] = ("B0", "L1")
"""The positive control's arms, at the primary setting."""

STAGE1_SEED: Final[int] = 0
"""Stage 1 is fitted once, at the primary setting, with this seed, and reused by stage 2."""

MIN_RESIDUAL_HOURS: Final[int] = 3
"""A day's mean stage-1 residual needs at least this many hours with both power and a prediction."""

RESIDUAL_WINDOWS: Final[tuple[tuple[int, int], ...]] = ((1, 1), (7, 5), (30, 20))
"""S3's windows: the number of whole days, and the fewest of them that must hold a residual."""

CLOCK_RATIO_DAYS: Final[int] = 30
CLOCK_RATIO_MIN_DAYS: Final[int] = 15
"""R2's per-clock-hour ratio of observed to predicted power over 30 days, at least 15 present."""

RATIO_WINDOW: Final[tuple[int, int]] = (7, 5)
"""R1's energy ratio is taken over 7 whole days, at least 5 of which hold both energies."""

SMOKE_ROUNDS: Final[int] = 20
"""The boosting rounds a smoke test uses in place of the settings' own."""


class Context(NamedTuple):
    """What every fit needs besides its data."""

    checkpoint_dir: Path
    device: DeviceType
    max_workers: int
    smoke: bool


def settings_for(*, context: Context) -> dict[str, HyperParameters]:
    """Return the two hyperparameter settings, shortened for a smoke test.

    Args:
        context: The run's context.

    Returns:
        The primary and sensitivity settings by name.
    """
    settings = {"primary": PRIMARY_HYPER_PARAMETERS, "sensitivity": SENSITIVITY_HYPER_PARAMETERS}
    if not context.smoke:
        return settings
    return {name: {**params, "num_boost_round": SMOKE_ROUNDS} for name, params in settings.items()}


def features_of(*, arm: str) -> tuple[str, ...]:
    """Return an arm's feature columns, `{fold}` placeholders unfilled.

    Args:
        arm: The arm.

    Returns:
        B0's columns followed by the arm's own.
    """
    return (*B0_COLUMNS, *extra_columns_of(arm=arm))


# --- Stage 1 ------------------------------------------------------------------------------------


def stage1_predictions(
    *, hours: pl.DataFrame, hyper_parameters: HyperParameters, device: DeviceType
) -> pl.DataFrame:
    """Predict power at every hour, once per scored fold, withholding two folds from each model.

    The prediction for scored fold `k` at an hour in fold `f` comes from a model trained on the
    plant's shared rows outside folds `k` and `f`. An hour outside the shared rows has no fold, so
    only fold `k` is withheld.

    Args:
        hours: `build_lag_frame.stage1_hours`'s result.
        hyper_parameters: The setting to fit at.
        device: XGBoost's device.

    Returns:
        One row per hour with `site`, `time` and `stage1_pred_fold<k>` for each scored fold `k`.
    """
    parts = []
    for _site, rows in hours.group_by("site", maintain_order=True):
        trainable = rows.filter(
            pl.col("fold").is_not_null()
            & ~pl.col("constrained")
            & pl.col(TARGET).is_not_null()
            & (pl.col("time") < final_test_cutoff())
        )
        assert_before_cutoff(frame=trainable, name="the stage-1 training rows")
        predictions: dict[frozenset[int], np.ndarray] = {}
        for first in range(N_FOLDS):
            for second in range(first, N_FOLDS):
                withheld = frozenset({first, second})
                point, _ = fit_one_fold(
                    train=trainable.filter(~pl.col("fold").is_in(list(withheld))),
                    test=rows,
                    features=list(B0_COLUMNS),
                    target=TARGET,
                    hyper_parameters=hyper_parameters,
                    seed=STAGE1_SEED,
                    with_quantiles=False,
                    device=device,
                )
                predictions[withheld] = point
        folds = rows["fold"].to_numpy()
        columns: dict[str, np.ndarray] = {}
        for scored in range(N_FOLDS):
            own = np.where(np.isnan(folds), scored, folds).astype(int)
            column = np.full(rows.height, np.nan)
            for fold in range(N_FOLDS):
                mask = own == fold
                column[mask] = predictions[frozenset({scored, fold})][mask]
            columns[f"stage1_pred_fold{scored}"] = column
        parts.append(rows.select("site", "time").with_columns(**columns))
        _LOG.info("stage 1 done for one plant (%d fits)", len(predictions))
    return pl.concat(parts)


def _daily_residual_features(*, residual: pl.DataFrame) -> pl.DataFrame:
    """Roll each plant's daily stage-1 residuals and energies into S3's and R1's windows.

    Args:
        residual: One row per hour with `site`, `date`, `observed_mw` and `predicted_mw`, for the
            hours with both.

    Returns:
        One row per `(site, date)` with `resid_mean_<n>d` for S3's windows and `energy_ratio_7d`.
    """
    daily = (
        residual.group_by("site", "date")
        .agg(
            n_hours=pl.len(),
            resid=(
                pl.col("observed_mw").cast(pl.Float64) - pl.col("predicted_mw").cast(pl.Float64)
            ).mean(),
            observed_energy=pl.col("observed_mw").cast(pl.Float64).sum(),
            predicted_energy=pl.col("predicted_mw").cast(pl.Float64).sum(),
        )
        .with_columns(
            resid=pl.when(pl.col("n_hours") >= MIN_RESIDUAL_HOURS).then(pl.col("resid")),
            observed_energy=pl.when(pl.col("n_hours") >= MIN_RESIDUAL_HOURS).then(
                pl.col("observed_energy")
            ),
            predicted_energy=pl.when(pl.col("n_hours") >= MIN_RESIDUAL_HOURS).then(
                pl.col("predicted_energy")
            ),
        )
    )
    last = daily["date"].max()
    grid = (
        daily.group_by("site")
        .agg(first=pl.col("date").min())
        .select("site", date=pl.date_ranges(pl.col("first"), pl.lit(last), interval="1d"))
        .explode("date")
    )
    full = grid.join(daily, on=["site", "date"], how="left").sort("site", "date")
    window, minimum = RATIO_WINDOW
    return full.select(
        "site",
        "date",
        *(
            pl.col("resid")
            .rolling_mean(window_size=days, min_samples=least)
            .over("site")
            .alias(f"resid_mean_{days}d")
            for days, least in RESIDUAL_WINDOWS
        ),
        energy_ratio_7d=(
            pl.col("observed_energy")
            .rolling_sum(window_size=window, min_samples=minimum)
            .over("site")
            / pl.col("predicted_energy")
            .rolling_sum(window_size=window, min_samples=minimum)
            .over("site")
        ),
    )


def stage1_columns(
    *, frame: pl.DataFrame, hours: pl.DataFrame, predictions: pl.DataFrame
) -> pl.DataFrame:
    """Derive the two-step arms' `{fold}` columns and R1's energy ratio.

    Args:
        frame: The lead-day 1 frame, with `lag_d1`.
        hours: `build_lag_frame.stage1_hours`'s result.
        predictions: `stage1_predictions`'s result.

    Returns:
        One row per row of `frame`, with `site`, `time` and, for each scored fold, the columns of
        `STAGE1_TARGET_TEMPLATE`, `STAGE1_LAG_TEMPLATE`, `RESIDUAL_LAG_TEMPLATE`,
        `RESIDUAL_MEAN_TEMPLATES` and `RATIO_TEMPLATE`.
    """
    lead_day = FULL_SWEEP_LEAD_DAY
    observed = hours.select("site", "time", "observed_mw")
    keys = frame.select(
        "site", "time", "lag_d1", lag_time=pl.col("time") - pl.duration(days=lead_day + 1)
    ).with_columns(asof_date=target_date(lead_day=lead_day))
    out = keys.select("site", "time")
    for fold in range(N_FOLDS):
        column = f"stage1_pred_fold{fold}"
        own = predictions.select("site", "time", pl.col(column).alias("predicted"))
        target_prediction = keys.join(own, on=["site", "time"], how="left", maintain_order="left")
        lag_prediction = keys.join(
            own.rename({"time": "lag_time"}),
            on=["site", "lag_time"],
            how="left",
            maintain_order="left",
        )
        residual = (
            own.join(observed, on=["site", "time"])
            .drop_nulls()
            .rename({"predicted": "predicted_mw"})
            .with_columns(date=(pl.col("time") - pl.duration(minutes=30)).dt.date())
        )
        clock_ratio = same_clock_hour_window(
            keys=keys,
            hourly=residual.select("site", "time", power_mw=pl.col("observed_mw")),
            day=lead_day,
            first_days_back=1,
            last_days_back=CLOCK_RATIO_DAYS,
            statistic="mean",
            min_count=CLOCK_RATIO_MIN_DAYS,
        ) / same_clock_hour_window(
            keys=keys,
            hourly=residual.select("site", "time", power_mw=pl.col("predicted_mw")),
            day=lead_day,
            first_days_back=1,
            last_days_back=CLOCK_RATIO_DAYS,
            statistic="mean",
            min_count=CLOCK_RATIO_MIN_DAYS,
        )
        rolled = _daily_residual_features(residual=residual).rename({"date": "asof_date"})
        joined = keys.join(rolled, on=["site", "asof_date"], how="left", maintain_order="left")
        derived = {
            STAGE1_TARGET_TEMPLATE.format(fold=fold): target_prediction["predicted"],
            STAGE1_LAG_TEMPLATE.format(fold=fold): lag_prediction["predicted"],
            RESIDUAL_LAG_TEMPLATE.format(fold=fold): keys["lag_d1"] - lag_prediction["predicted"],
            RATIO_TEMPLATE.format(fold=fold): joined["energy_ratio_7d"],
            CLOCK_RATIO_TEMPLATE.format(fold=fold): clock_ratio,
            **{
                template.format(fold=fold): joined[f"resid_mean_{days}d"]
                for template, (days, _) in zip(
                    RESIDUAL_MEAN_TEMPLATES, RESIDUAL_WINDOWS, strict=True
                )
            },
        }
        out = out.with_columns(**derived)
    return out


STAGE1_PROBE_DAYS: Final[int] = 10
STAGE1_PROBE_ROWS_PER_DAY: Final[int] = 20
STAGE1_PROBE_SEED: Final[int] = 1138
STAGE1_PROBE_RELATIVE_TOLERANCE: Final[float] = 1e-9
"""The stage-1 anchor probe rebuilds 20 rows on each of 10 seeded target days (about 200 rows)."""


def stage1_anchor_probe(
    *,
    frame: pl.DataFrame,
    hours: pl.DataFrame,
    predictions: pl.DataFrame,
    derived: pl.DataFrame,
) -> list[str]:
    """Rebuild sampled rows' stage-1 columns from inputs cut at the issue time, and compare.

    The columns checked are S2's lag prediction and residual, S3's residual means, and the ratios
    R1s and R2 read. The prediction at the target hour itself is not checked, because the forecast
    of the target hour is available at the issue time by design.

    Args:
        frame: The lead-day 1 frame, with `lag_d1`.
        hours: The stage-1 hours.
        predictions: The stage-1 predictions.
        derived: `stage1_columns`' result for every row of `frame`.

    Returns:
        A report line giving the rows and columns compared.

    Raises:
        ValueError: If a column changes when every hour after the issue time is removed.
    """
    keyed = frame.select(
        "site",
        "time",
        "lag_d1",
        issue=issue_time(
            day_start=(pl.col("time") - pl.duration(minutes=30)).dt.truncate("1d"),
            day=FULL_SWEEP_LEAD_DAY,
        ),
    )
    columns = [c for c in derived.columns if c not in {"site", "time"} and "stage1_target" not in c]
    issues = keyed["issue"].unique().sort().sample(n=STAGE1_PROBE_DAYS, seed=STAGE1_PROBE_SEED)
    compared = 0
    for issue in issues:
        day_rows = keyed.filter(pl.col("issue") == issue)
        sample = day_rows.sample(
            n=min(STAGE1_PROBE_ROWS_PER_DAY, day_rows.height), seed=STAGE1_PROBE_SEED
        ).sort("site", "time")
        rebuilt = stage1_columns(
            frame=sample,
            hours=hours.filter(pl.col("time") <= issue),
            predictions=predictions.filter(pl.col("time") <= issue),
        )
        full = sample.select("site", "time").join(derived, on=["site", "time"], how="left")
        for column in columns:
            both = full.select("site", "time", full=pl.col(column)).join(
                rebuilt.select("site", "time", cut=pl.col(column)), on=["site", "time"]
            )
            different = both.filter(
                (pl.col("full").is_null() != pl.col("cut").is_null())
                | (
                    (pl.col("full") - pl.col("cut")).abs()
                    > STAGE1_PROBE_RELATIVE_TOLERANCE * (1 + pl.col("full").abs())
                )
            ).height
            if different:
                msg = (
                    f"{column} changes on {different} rows when every stage-1 hour after {issue} "
                    "is removed, so it reads the future"
                )
                raise ValueError(msg)
        compared += sample.height
    return [
        (
            f"- stage-1 anchor probe: {compared} seeded random rows on {STAGE1_PROBE_DAYS} target "
            f"days; all {len(columns)} stage-1 columns equal the full build when every stage-1 "
            "hour after the issue time is removed, nulls included."
        )
    ]


# --- Fitting and checkpoints -----------------------------------------------------------------


def checkpoint_path(*, context: Context, scope: str, setting: str, arm: str) -> Path:
    """Return the checkpoint file of one (scope, setting, arm).

    Args:
        context: The run's context.
        scope: The scope, such as `lead1` or `global`.
        setting: The hyperparameter setting's name.
        arm: The arm.

    Returns:
        The path of the parquet file.
    """
    return context.checkpoint_dir / f"{scope}__{setting}__{arm}.parquet"


def _annotated(
    *, losses: pl.DataFrame, dataset: pl.DataFrame, scope: str, lead_day: int, quantiles: bool
) -> pl.DataFrame:
    """Add the actual power, the prediction, the scope and the capacity-normalised CRPS.

    Args:
        losses: Per-row losses of one arm and setting, with `arm`, `setting` and `signed_error_mw`.
        dataset: The frame the losses were fitted on, for the actual power.
        scope: The scope label.
        lead_day: The lead-day.
        quantiles: Whether the losses carry quantile scores.

    Returns:
        The losses with `actual`, `prediction`, `scope`, `lead_day`, `with_quantiles` and
        `crps_capped_fraction_of_capacity`.
    """
    return (
        losses.join(
            dataset.select("site", "time", actual=pl.col(TARGET)), on=["site", "time"], how="left"
        )
        .with_columns(
            prediction=pl.col("actual") + pl.col("signed_error_mw"),
            scope=pl.lit(scope),
            lead_day=pl.lit(lead_day, dtype=pl.Int32),
            with_quantiles=pl.lit(quantiles),
            crps_capped_fraction_of_capacity=pl.col("crps_capped_mw")
            / pl.col("effective_capacity_mw"),
        )
        .cast({"actual": pl.Float64})
    )


def fit_arms(
    *,
    context: Context,
    dataset: pl.DataFrame,
    scope: str,
    lead_day: int,
    arms: Sequence[str],
    setting: str,
    quantiles: bool,
    pooled: bool = False,
    lopo: bool = False,
) -> list[pl.DataFrame]:
    """Fit several arms at one setting in one batch, or read back any a checkpoint holds.

    The per-plant jobs of every arm still to fit share one `run_all` call, so the workers stay busy
    across arms. Each arm is checkpointed on its own.

    Args:
        context: The run's context.
        dataset: The site-sorted frame, with `fold`, `month`, `cap_mw`, `constrained` and capacity.
        scope: A label for the frame: `lead<N>`, `global`, `lopo` or `control_s<percent>`.
        lead_day: The lead-day of the frame.
        arms: The arms.
        setting: `primary` or `sensitivity`.
        quantiles: Whether to fit the quantile models too.
        pooled: Whether `dataset` is one pooled model's frame (fitted as a single `site_rows`).
        lopo: Whether to fit the pooled frame leaving each plant out in turn (exploratory).

    Returns:
        One frame per arm, in the order of `arms`: its per-row losses with `arm`, `setting`,
        `scope`, `lead_day`, `with_quantiles`, `actual` and `prediction` (in the target's own
        units) and `crps_capped_fraction_of_capacity`.
    """
    assert_before_cutoff(frame=dataset, name=f"the {scope} frame's rows")
    suffix = ".quantile.parquet" if quantiles else ".parquet"
    paths = {
        arm: checkpoint_path(context=context, scope=scope, setting=setting, arm=arm).with_suffix(
            suffix
        )
        for arm in arms
    }
    missing = [arm for arm in arms if not paths[arm].exists()]
    hyper_parameters = settings_for(context=context)[setting]
    fitted: dict[str, pl.DataFrame] = {}
    if missing and not (pooled or lopo):
        jobs: list[Job] = [
            (arm, setting, TARGET, features_of(arm=arm), hyper_parameters, quantiles)
            for arm in missing
        ]
        stacked = run_all(
            dataset=dataset, jobs=jobs, max_workers=context.max_workers, device=context.device
        )
        fitted = {arm: stacked.filter(pl.col("arm") == arm) for arm in missing}
    for arm in missing if (pooled or lopo) else []:
        if lopo:
            fitted[arm] = leave_one_plant_out(
                context=context, dataset=dataset, arm=arm, hyper_parameters=hyper_parameters
            ).with_columns(arm=pl.lit(arm), setting=pl.lit(setting))
        else:
            fitted[arm] = out_of_fold_losses(
                site_rows=dataset,
                features=features_of(arm=arm),
                target=TARGET,
                hyper_parameters=hyper_parameters,
                with_quantiles=quantiles,
                device=context.device,
            ).with_columns(arm=pl.lit(arm), setting=pl.lit(setting), target=pl.lit(TARGET))
    for arm, losses in fitted.items():
        if lopo:
            annotated = losses.with_columns(
                scope=pl.lit(scope),
                lead_day=pl.lit(lead_day, dtype=pl.Int32),
                with_quantiles=pl.lit(value=False),
            )
        else:
            annotated = _annotated(
                losses=losses, dataset=dataset, scope=scope, lead_day=lead_day, quantiles=quantiles
            )
        write_parquet_atomic(frame=annotated, path=paths[arm])
        _LOG.info("saved %s", paths[arm].name)
    return [pl.read_parquet(paths[arm]) for arm in arms]


def leave_one_plant_out(
    *, context: Context, dataset: pl.DataFrame, arm: str, hyper_parameters: HyperParameters
) -> pl.DataFrame:
    """Predict each plant from a model that never saw the plant, withholding the scored months.

    For plant `p` and fold `k`, the model trains on the other plants' rows outside fold `k` (the
    fleet-wide fold, so the scored months are withheld at every plant), and predicts plant `p`'s
    rows in fold `k`.

    Args:
        context: The run's context.
        dataset: The pooled, capacity-normalised frame from `pooled_frame`.
        arm: The arm.
        hyper_parameters: The setting to fit at.

    Returns:
        One row per (plant, time, seed) with the losses `score_prediction` gives, `actual` and
        `prediction`, in fractions of capacity.
    """
    features = features_of(arm=arm)
    sites = sorted(dataset["site"].unique().to_list())

    def one_plant(site: str) -> pl.DataFrame:
        parts = []
        for fold in sorted(dataset["fold"].unique().to_list()):
            train = dataset.filter(
                (pl.col("site") != site) & (pl.col("fold") != fold) & ~pl.col("constrained")
            )
            test = dataset.filter((pl.col("site") == site) & (pl.col("fold") == fold))
            if test.is_empty() or train.is_empty():
                continue
            fold_features = [name.format(fold=fold) for name in features]
            for seed in SEEDS:
                point, _ = fit_one_fold(
                    train=train,
                    test=test,
                    features=fold_features,
                    target=TARGET,
                    hyper_parameters=hyper_parameters,
                    seed=seed,
                    with_quantiles=False,
                    device=context.device,
                )
                prediction = test.select("site", "time").with_columns(
                    seed=pl.lit(seed, dtype=pl.Int32), prediction=pl.Series(point, dtype=pl.Float64)
                )
                scored = score_prediction(rows=test, prediction=prediction, target=TARGET)
                parts.append(
                    scored.join(prediction, on=["site", "time", "seed"]).join(
                        test.select("site", "time", actual=pl.col(TARGET).cast(pl.Float64)),
                        on=["site", "time"],
                    )
                )
        return pl.concat(parts)

    with concurrent.futures.ThreadPoolExecutor(max_workers=context.max_workers) as pool:
        return pl.concat(list(pool.map(one_plant, sites)))


INTERVAL_ARMS: Final[tuple[str, ...]] = ("B0", "L1")
"""The arms whose nine quantile predictions are saved, to measure coverage and width (and, for
B0, to score R4 and R5 on the continuous ranked probability score)."""

INTERVAL_SEED: Final[int] = 0
"""The one seed the saved quantile predictions come from; coverage and width are descriptive."""

QUANTILE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"q{index}" for index in range(len(QUANTILE_LEVELS))
)
"""The saved columns, one per level of `QUANTILE_LEVELS`, sorted within each row."""


def _plant_intervals(
    *, rows: pl.DataFrame, arm: str, hyper_parameters: HyperParameters, device: DeviceType
) -> pl.DataFrame:
    """Predict one plant's nine quantiles out of fold, with `INTERVAL_SEED`.

    Args:
        rows: One plant's rows.
        arm: The arm.
        hyper_parameters: The setting to fit at.
        device: XGBoost's device.

    Returns:
        `site`, `time`, `month`, `fold`, the forecast irradiance and clear-sky irradiance that
        classify the sky, the actual power, CK's ceiling (null where the frame lacks it), and the
        nine quantiles sorted within each row and held to the export cap, all as fractions of the
        plant's capacity.
    """
    parts = []
    for fold in sorted(rows["fold"].unique().to_list()):
        test = rows.filter(pl.col("fold") == fold)
        train = rows.filter((pl.col("fold") != fold) & ~pl.col("constrained"))
        _, quantiles = fit_one_fold(
            train=train,
            test=test,
            features=[name.format(fold=fold) for name in features_of(arm=arm)],
            target=TARGET,
            hyper_parameters=hyper_parameters,
            seed=INTERVAL_SEED,
            with_quantiles=True,
            device=device,
        )
        if quantiles is None:
            msg = "quantile models were requested but not returned"
            raise ValueError(msg)
        capped = np.sort(
            clamp_to_cap(prediction=np.sort(quantiles, axis=1), cap_mw=test["cap_mw"]), axis=1
        )
        capacity = test["effective_capacity_mw"].cast(pl.Float64)
        ceiling = (
            test["ck_expanding_p995"].cast(pl.Float64) / capacity
            if "ck_expanding_p995" in test.columns
            else pl.Series([None] * test.height, dtype=pl.Float64)
        )
        parts.append(
            test.select("site", "time", "month", "fold", "nwp_ghi", "clear_sky_w_m2").with_columns(
                actual=test[TARGET].cast(pl.Float64) / capacity,
                ceiling=ceiling,
                **{
                    name: pl.Series(capped[:, index]) / capacity
                    for index, name in enumerate(QUANTILE_COLUMNS)
                },
            )
        )
    return pl.concat(parts)


def fit_intervals(*, context: Context, dataset: pl.DataFrame, arm: str) -> pl.DataFrame:
    """Return an arm's out-of-fold quantile predictions, or read back a checkpoint.

    Args:
        context: The run's context.
        dataset: The lead-day 1 frame.
        arm: The arm.

    Returns:
        `_plant_intervals`' result for every plant, labelled with the arm.
    """
    path = context.checkpoint_dir / f"intervals__{arm}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    hyper_parameters = settings_for(context=context)["primary"]
    sites = sorted(dataset["site"].unique().to_list())
    with concurrent.futures.ThreadPoolExecutor(max_workers=context.max_workers) as pool:
        parts = list(
            pool.map(
                lambda site: _plant_intervals(
                    rows=dataset.filter(pl.col("site") == site),
                    arm=arm,
                    hyper_parameters=hyper_parameters,
                    device=context.device,
                ),
                sites,
            )
        )
    result = pl.concat(parts).with_columns(arm=pl.lit(arm))
    write_parquet_atomic(frame=result, path=path)
    return result


IMPORTANCE_ARMS: Final[tuple[str, ...]] = ("B0", "L1", "L2", "S2", "X")
"""The arms whose boosters' gains are saved; `X` stands for the shortlist rule's arm."""

IMPORTANCE_SEED: Final[int] = 0
"""The one seed of the importance-only refit."""

FEATURE_GROUPS: Final[tuple[tuple[str, str], ...]] = (
    ("hour_of_day", "calendar"),
    ("day_of_year", "calendar"),
    ("era_code", "calendar"),
    ("solar_", "sun position"),
    ("nwp_", "forecast weather"),
    ("lag_nwp", "lag-hour forecast weather"),
    ("lag_ghi", "lag-hour forecast weather"),
    ("lag_d", "lagged power"),
    ("null_lag", "random lags (controls)"),
    ("stage1_", "stage-1 predictions and residuals"),
    ("resid_", "stage-1 predictions and residuals"),
    ("week_", "windowed power statistics"),
    ("q30_", "windowed power statistics"),
    ("im_", "issue morning"),
    ("tf_", "transfer function"),
    ("ck_", "clipping ceiling"),
    ("an_", "analogue ensemble"),
    ("pc_", "satellite ratios"),
    ("rp_", "relative plant energy"),
    ("dt_", "diurnal shape"),
    ("days_since", "date"),
    ("plant_code", "plant code"),
)
"""Name prefixes mapped to feature groups, tried in order; the importance figure sums by group."""


def feature_group(*, column: str) -> str:
    """Return the feature group of a column.

    Args:
        column: A feature column name, with any `{fold}` already filled.

    Returns:
        The group of the first matching prefix.

    Raises:
        ValueError: If no prefix matches, so a new column cannot silently fall outside every group.
    """
    for prefix, group in FEATURE_GROUPS:
        if column.startswith(prefix):
            return group
    msg = f"column {column!r} belongs to no feature group"
    raise ValueError(msg)


def _plant_importance(
    *, rows: pl.DataFrame, arm: str, hyper_parameters: HyperParameters, device: DeviceType
) -> pl.DataFrame:
    """Refit one plant's point model per fold and return each column's share of total gain.

    Args:
        rows: One plant's rows.
        arm: The arm.
        hyper_parameters: The setting to fit at.
        device: XGBoost's device.

    Returns:
        `site`, `fold`, `column`, `group` and `share`: the shares sum to 1 within a (site, fold)
        model, and a column the booster never split on has share 0.
    """
    records = []
    for fold in sorted(rows["fold"].unique().to_list()):
        train = rows.filter((pl.col("fold") != fold) & ~pl.col("constrained"))
        features = [name.format(fold=fold) for name in features_of(arm=arm)]
        matrix = xgb.DMatrix(train.select(features).to_numpy(), label=train[TARGET].to_numpy())
        booster = xgb.train(
            {
                **booster_parameters(
                    hyper_parameters=hyper_parameters, seed=IMPORTANCE_SEED, device=device
                ),
                "objective": "reg:absoluteerror",
            },
            matrix,
            num_boost_round=hyper_parameters["num_boost_round"],
        )
        gains = booster.get_score(importance_type="total_gain")
        values = np.asarray([gains.get(f"f{index}", 0.0) for index in range(len(features))])
        total = values.sum()
        shares = values / total if total > 0 else values
        records += [
            {
                "site": rows["site"][0],
                "fold": fold,
                "column": column,
                "group": feature_group(column=column),
                "share": float(share),
            }
            for column, share in zip(features, shares, strict=True)
        ]
    return pl.DataFrame(records)


def fit_importance(*, context: Context, dataset: pl.DataFrame, arm: str) -> pl.DataFrame:
    """Return an arm's per-model gain shares from a separate refit, or read back a checkpoint.

    Importance is descriptive: gain is measured on the training rows and splits credit between
    correlated columns by whichever the greedy search picks first. The refit adds no column to any
    arm and enters no planned contrast.

    Args:
        context: The run's context.
        dataset: The lead-day 1 frame.
        arm: The arm.

    Returns:
        `_plant_importance`' result for every plant, labelled with the arm.
    """
    path = context.checkpoint_dir / f"importance__{arm}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    hyper_parameters = settings_for(context=context)["primary"]
    sites = sorted(dataset["site"].unique().to_list())
    with concurrent.futures.ThreadPoolExecutor(max_workers=context.max_workers) as pool:
        parts = list(
            pool.map(
                lambda site: _plant_importance(
                    rows=dataset.filter(pl.col("site") == site),
                    arm=arm,
                    hyper_parameters=hyper_parameters,
                    device=context.device,
                ),
                sites,
            )
        )
    result = pl.concat(parts).with_columns(arm=pl.lit(arm))
    write_parquet_atomic(frame=result, path=path)
    return result


def phase1_table(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Rank the sweep's arms by mean absolute error on the screening months.

    Args:
        losses: The lead-day 1 sweep's primary-setting losses.

    Returns:
        One row per arm with `mean_error` (a fraction of capacity), best first.
    """
    return (
        losses.filter(pl.col("month").is_in(SCREENING_MONTHS))
        .group_by("arm")
        .agg(mean_error=pl.col(METRIC).mean(), n_rows=pl.len())
        .sort("mean_error")
    )


def shortlist(*, losses: pl.DataFrame, global_scope: bool = False) -> str:
    """Apply the shortlist rule.

    Args:
        losses: The lead-day 1 sweep's primary-setting losses.
        global_scope: Whether to choose the arm for the global model, which skips every arm whose
            columns carry a `{fold}` placeholder (S3 and KS). A stage-1 column is built from
            models that withheld the scored fold at one plant, and pooling the plants would let
            another plant's rows in the scored months train the model.

    Returns:
        X: the arm among `SHORTLIST_CANDIDATES` with the lowest phase-1 mean absolute error.
    """
    eligible = [
        arm
        for arm in SHORTLIST_CANDIDATES
        if not (global_scope and any("{fold}" in c for c in extra_columns_of(arm=arm)))
    ]
    candidates = phase1_table(losses=losses).filter(pl.col("arm").is_in(eligible))
    return str(candidates["arm"][0])


# --- The global frame ---------------------------------------------------------------------------


def pooled_frame(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Pool the plants into one capacity-normalised frame with one fleet-wide fold assignment.

    Every power column is divided by the plant's effective capacity, which is then set to 1.0, so
    the loss columns stay in fractions of capacity and the model cannot read a plant's size off a
    lag in megawatts. Folds are cut over the whole fleet's months, so a scored month is withheld at
    every plant at once.

    Args:
        frame: The per-plant frames stacked, site-sorted.

    Returns:
        The pooled frame, site-sorted, with `fold` from `cut_eras` over a single fleet group.

    Raises:
        ValueError: If a month is held out of every fold's training at some calendar month.
    """
    scaled = [
        (pl.col(c) / pl.col("effective_capacity_mw")).alias(c)
        for c in frame.columns
        if c.startswith(POWER_COLUMN_PREFIXES) and frame[c].dtype.is_float()
    ]
    fleet = cut_eras(
        frame=frame.with_columns(plant=pl.col("site"), site=pl.lit("fleet")).drop("fold"),
        first_months=NWP_ERA_START_MONTHS,
        fold_offsets=NWP_ERA_FOLD_OFFSETS,
    )
    raise_on_uncovered_months(coverage=calendar_month_coverage(frame=fleet))
    pooled = (
        frame.drop("fold")
        .with_columns(fold=fleet["fold"])
        .with_columns(scaled)
        .with_columns(effective_capacity_mw=pl.lit(1.0, dtype=pl.Float64))
    )
    return pooled.sort("site", "time")


# --- The plan -----------------------------------------------------------------------------------


class Fit(NamedTuple):
    """One arm at one setting on one frame."""

    frame_key: str
    scope: str
    lead_day: int
    arm: str
    setting: str
    quantiles: bool
    pooled: bool = False
    lopo: bool = False


BATCH_ARMS: Final[int] = 4
"""How many arms share one `run_all` call: 4 arms is 24 plant fits on 4 workers, and a reboot loses
at most one batch."""


def _batch_key(*, fit: Fit) -> tuple[object, ...]:
    """Return everything about a fit except its arm, which decides what can share a batch."""
    return (*fit[:3], *fit[4:])


def fit_all(
    *, context: Context, frames: dict[str, pl.DataFrame], fits: list[Fit]
) -> list[pl.DataFrame]:
    """Run the fits in order, batching neighbouring per-plant fits of up to `BATCH_ARMS` arms.

    Args:
        context: The run's context.
        frames: The frames the fits name, by key.
        fits: The fits to run.

    Returns:
        The results, in the order of `fits`.
    """
    results: list[pl.DataFrame] = []
    index = 0
    while index < len(fits):
        first = fits[index]
        batch = [first]
        batchable = not (first.pooled or first.lopo)
        while (
            batchable
            and index + len(batch) < len(fits)
            and len(batch) < BATCH_ARMS
            and _batch_key(fit=fits[index + len(batch)]) == _batch_key(fit=first)
        ):
            batch.append(fits[index + len(batch)])
        results += fit_arms(
            context=context,
            dataset=frames[first.frame_key],
            scope=first.scope,
            lead_day=first.lead_day,
            arms=[fit.arm for fit in batch],
            setting=first.setting,
            quantiles=first.quantiles,
            pooled=first.pooled,
            lopo=first.lopo,
        )
        index += len(batch)
    return results


def sweep_fits(*, context: Context) -> list[Fit]:
    """Return the lead-day 1 sweep: every arm at the primary setting.

    Args:
        context: The run's context.

    Returns:
        The fits. B0, L1 and N2 carry their quantile models.
    """
    fits = [
        Fit(
            "lead1",
            f"lead{FULL_SWEEP_LEAD_DAY}",
            FULL_SWEEP_LEAD_DAY,
            arm,
            "primary",
            arm in SWEEP_QUANTILE_ARMS,
        )
        for arm in SWEEP_ARMS
    ]
    return sorted(fits, key=lambda fit: fit.quantiles)


def phase2_fits(*, chosen: str, chosen_global: str) -> list[Fit]:
    """Return phase 2's fits at lead-day 1: sensitivity, quantile, global and control fits.

    Args:
        chosen: The shortlist rule's X.
        chosen_global: X for the global scope, which skips arms with `{fold}` columns.

    Returns:
        The fits, with `X` replaced by `chosen` wherever an arm list names it, and in the global
        scope by `chosen_global`.
    """
    lead, scope = FULL_SWEEP_LEAD_DAY, f"lead{FULL_SWEEP_LEAD_DAY}"

    def named(arms: tuple[str, ...]) -> list[str]:
        return [chosen if arm == "X" else arm for arm in arms]

    fits = [
        Fit("lead1", scope, lead, arm, "primary", quantiles=True)
        for arm in named(QUANTILE_PRIMARY_ARMS)
        if arm not in SWEEP_QUANTILE_ARMS
    ]
    fits += sorted(
        (
            Fit("lead1", scope, lead, arm, "sensitivity", arm in QUANTILE_SENSITIVITY_ARMS)
            for arm in named(PHASE2_POINT_ARMS)
        ),
        key=lambda fit: fit.quantiles,
    )
    width = len(extra_columns_of(arm=chosen))
    if width > WIDE_X_COLUMNS:
        fits += [
            Fit("lead1", scope, lead, f"N2-{width}", setting, quantiles=False)
            for setting in ("primary", "sensitivity")
        ]
    fits += [
        Fit("global", "global", lead, arm, setting, arm in GLOBAL_QUANTILE_ARMS, pooled=True)
        for setting in ("primary", "sensitivity")
        for arm in [chosen_global if arm == "X" else arm for arm in GLOBAL_ARMS]
    ]
    fits += [
        Fit("global", "global", lead, arm, "primary", quantiles=False, pooled=True)
        for arm in GLOBAL_ONLY_ARMS
    ]
    fits += [
        Fit("global", "lopo", lead, arm, "primary", quantiles=False, lopo=True)
        for arm in LEAVE_ONE_PLANT_OUT_ARMS
    ]
    for shift in POSITIVE_CONTROL_SHIFTS:
        percent = f"{round(shift * 100):02d}"
        fits += [
            Fit(f"control{percent}", f"control_s{percent}", lead, arm, "primary", quantiles=False)
            for arm in CONTROL_ARMS
        ]
    return fits


def longer_lead_fits(*, lead_days: tuple[int, ...], arms: tuple[str, ...]) -> list[Fit]:
    """Return point fits at the primary setting for each arm at each lead-day.

    Args:
        lead_days: The lead-days.
        arms: The arms.

    Returns:
        The fits, whose frames are keyed `lead<N>`.
    """
    return [
        Fit(f"lead{lead}", f"lead{lead}", lead, arm, "primary", quantiles=False)
        for lead in lead_days
        for arm in arms
    ]


def read_lead_frames(
    *, root: Path, product: WeatherProduct, lead_days: tuple[int, ...], smoke: bool
) -> dict[str, pl.DataFrame]:
    """Read the built frame of each lead-day.

    Args:
        root: The output root.
        product: The weather product.
        lead_days: The lead-days.
        smoke: Whether to subsample each frame as a smoke run does.

    Returns:
        The frames, keyed `lead<N>`.
    """
    paths = output_paths(root=root, product=product, lead_days=lead_days)
    frames = {f"lead{lead}": pl.read_parquet(paths[f"day{lead}"]) for lead in lead_days}
    return {key: smoke_subsample(frame=frame) if smoke else frame for key, frame in frames.items()}


def lead1_dataset(*, context: Context, root: Path, product: WeatherProduct) -> pl.DataFrame:
    """Return the lead-day 1 frame with the stage-1 columns, fitting stage 1 if need be.

    Args:
        context: The run's context.
        root: The output root.
        product: The weather product.

    Returns:
        The frame.
    """
    paths = output_paths(root=root, product=product, lead_days=(FULL_SWEEP_LEAD_DAY,))
    frame = pl.read_parquet(paths[f"day{FULL_SWEEP_LEAD_DAY}"])
    derived_path = context.checkpoint_dir / "stage1_columns.parquet"
    if not derived_path.exists():
        hours = pl.read_parquet(paths["stage1"])
        predictions = stage1_predictions(
            hours=hours,
            hyper_parameters=settings_for(context=context)["primary"],
            device=context.device,
        )
        write_parquet_atomic(
            frame=predictions, path=context.checkpoint_dir / "stage1_predictions.parquet"
        )
        built = stage1_columns(frame=frame, hours=hours, predictions=predictions)
        probe = stage1_anchor_probe(
            frame=frame, hours=hours, predictions=predictions, derived=built
        )
        (context.checkpoint_dir / "stage1_probe.md").write_text("\n".join(probe) + "\n")
        write_parquet_atomic(frame=built, path=derived_path)
    dataset = frame.join(
        pl.read_parquet(derived_path), on=["site", "time"], how="left", maintain_order="left"
    )
    if context.smoke:
        dataset = smoke_subsample(frame=dataset)
    raise_on_uncovered_months(coverage=calendar_month_coverage(frame=dataset))
    return dataset


def run_ens_mean(*, context: Context, root: Path, product: WeatherProduct) -> list[pl.DataFrame]:
    """Run every stage of the plan for the ENS mean.

    Args:
        context: The run's context.
        root: The output root.
        product: The weather product (`ens_mean`).

    Returns:
        Every (scope, setting, arm) result, in the order fitted.
    """
    dataset = lead1_dataset(context=context, root=root, product=product)
    frames = {"lead1": dataset}
    results = fit_all(context=context, frames=frames, fits=sweep_fits(context=context))
    sweep = pl.concat(results, how="vertical_relaxed")
    chosen = shortlist(losses=sweep)
    chosen_global = shortlist(losses=sweep, global_scope=True)
    _write_shortlist(
        directory=context.checkpoint_dir, sweep=sweep, chosen=chosen, chosen_global=chosen_global
    )
    for arm in INTERVAL_ARMS:
        fit_intervals(context=context, dataset=dataset, arm=arm)

    other_leads = tuple(lead for lead in ENS_MEAN_LEAD_DAYS if lead != FULL_SWEEP_LEAD_DAY)
    frames |= read_lead_frames(
        root=root, product=product, lead_days=other_leads, smoke=context.smoke
    )
    frames["global"] = pooled_frame(frame=dataset)
    paths = output_paths(root=root, product=product, lead_days=())
    for shift in POSITIVE_CONTROL_SHIFTS:
        percent = f"{round(shift * 100):02d}"
        control = pl.read_parquet(paths[f"control{percent}"])
        frames[f"control{percent}"] = smoke_subsample(frame=control) if context.smoke else control
    fits = phase2_fits(chosen=chosen, chosen_global=chosen_global) + longer_lead_fits(
        lead_days=other_leads, arms=LONGER_LEAD_ARMS
    )
    results += fit_all(context=context, frames=frames, fits=fits)
    for arm in IMPORTANCE_ARMS:
        fit_importance(context=context, dataset=dataset, arm=chosen if arm == "X" else arm)
    return results


def run_ifs_single(*, context: Context, root: Path, product: WeatherProduct) -> list[pl.DataFrame]:
    """Fit B0 and L1 with IFS HRES at lead-days 1 and 2, point model, primary setting.

    Args:
        context: The run's context.
        root: The output root.
        product: The weather product (`ifs_single`).

    Returns:
        The results, in the order fitted.
    """
    frames = read_lead_frames(
        root=root, product=product, lead_days=IFS_LEAD_DAYS, smoke=context.smoke
    )
    fits = longer_lead_fits(lead_days=IFS_LEAD_DAYS, arms=REPLICATE_ARMS)
    return fit_all(context=context, frames=frames, fits=fits)


def _write_shortlist(
    *, directory: Path, sweep: pl.DataFrame, chosen: str, chosen_global: str
) -> None:
    """Print the phase-1 ranking and the shortlist rule's X, and save them beside the checkpoints.

    Args:
        directory: The checkpoint directory.
        sweep: The sweep's primary-setting losses.
        chosen: X.
        chosen_global: X for the global scope, which skips arms with `{fold}` columns.
    """
    table = phase1_table(losses=sweep)
    (directory / "shortlist.json").write_text(
        json.dumps({"x": chosen, "x_global": chosen_global, "phase1": table.to_dicts()}, indent=2)
    )
    _LOG.info("phase 1 ranking:\n%s", table)
    _LOG.info("the shortlist rule's X is %s (%s for the global scope)", chosen, chosen_global)


def combine(*, results: list[pl.DataFrame]) -> pl.DataFrame:
    """Stack every result, keeping the quantile run where an arm was fitted both ways.

    Args:
        results: Every (scope, setting, arm) result.

    Returns:
        One frame; for a (scope, setting, arm) fitted with and without quantiles, only the run with
        quantiles, whose point model is the same fit.
    """
    stacked = pl.concat(results, how="diagonal_relaxed")
    quantile_keys = (
        stacked.filter(pl.col("with_quantiles")).select("scope", "setting", "arm").unique()
    )
    return pl.concat(
        [
            stacked.filter(pl.col("with_quantiles")),
            stacked.filter(~pl.col("with_quantiles")).join(
                quantile_keys, on=["scope", "setting", "arm"], how="anti"
            ),
        ],
        how="diagonal_relaxed",
    ).sort("scope", "setting", "arm", "site", "time", "seed")


def reproduction_check(*, root: Path, device: DeviceType, max_workers: int) -> int:
    """Refit the ENS-mean B0 at lead-day 1 on the shared rows and compare it with the published run.

    The refit uses all 35,263 shared rows, including the 5,144 from `final_test_start` on, because
    the published number includes them; B0 reads no power, and nothing else at or after
    `final_test_start` reaches a fit.

    Args:
        root: The output root, where `reproduction_check.md` is written.
        device: The device to refit on, the study's own.
        max_workers: How many plants fit at once.

    Returns:
        0 on success.

    Raises:
        ValueError: If the refit's mean differs from `REPRODUCTION_MAE_PERCENT` by more than
            `REPRODUCTION_TOLERANCE_PERCENT`.
    """
    report = root / "ens_mean" / "reproduction_check.md"
    refuse_to_overwrite(paths=[report])
    frame = (
        shared_rows()
        .join(
            weather_table(product="ens_mean", lead_day=FULL_SWEEP_LEAD_DAY),
            on=["site", "time"],
            how="left",
        )
        .sort("site", "time")
    )
    job: Job = ("B0", "primary", TARGET, features_of(arm="B0"), PRIMARY_HYPER_PARAMETERS, False)
    losses = run_all(dataset=frame, jobs=[job], max_workers=max_workers, device=device)
    mean = float(losses.select(pl.col(METRIC).mean()).item()) * 100
    line = (
        f"- {device}: mean absolute error {mean:.3f}% of capacity on "
        f"{losses.select('site', 'time').n_unique()} rows; published {REPRODUCTION_MAE_PERCENT}%, "
        f"difference {mean - REPRODUCTION_MAE_PERCENT:+.3f} points "
        f"(tolerance {REPRODUCTION_TOLERANCE_PERCENT})."
    )
    heading = "# Reproduction check: ENS-mean B0 at lead-day 1 on the shared rows"
    text = f"{heading}\n\n{line}"
    report.write_text(text + "\n")
    sys.stdout.write(text + "\n")
    if abs(mean - REPRODUCTION_MAE_PERCENT) > REPRODUCTION_TOLERANCE_PERCENT:
        msg = f"the refit gives {mean:.4f}%, not the published {REPRODUCTION_MAE_PERCENT}%"
        raise ValueError(msg)
    return 0


def input_fingerprint(*, root: Path, product: WeatherProduct) -> dict[str, str]:
    """Return the SHA-256 of each built frame a run fits on.

    Args:
        root: The output root.
        product: The weather product.

    Returns:
        The digest of each lead-day-1 (or, for IFS HRES, lead-day 1 and 2) frame, and for the ENS
        mean the stage-1 hours and the positive-control frames, by file name.
    """
    if product == "ens_mean":
        paths = output_paths(root=root, product=product, lead_days=(FULL_SWEEP_LEAD_DAY,))
        files = [paths[key] for key in paths if key != "report"]
    else:
        paths = output_paths(root=root, product=product, lead_days=IFS_LEAD_DAYS)
        files = [paths[f"day{day}"] for day in IFS_LEAD_DAYS]
    return {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(files)}


def check_manifest(*, directory: Path, device: str, smoke: bool, inputs: dict[str, str]) -> None:
    """Write the run manifest, or raise if a resumed run's device, mode or inputs differ from it.

    Args:
        directory: The checkpoint directory.
        device: The run's device.
        smoke: Whether the run is a smoke test.
        inputs: `input_fingerprint`'s result.

    Raises:
        ValueError: If the checkpoints came from another device, mode or set of built frames.
    """
    path = directory / "run_manifest.json"
    current = {"device": device, "smoke": smoke, "inputs": inputs}
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != current:
            msg = (
                f"{directory} holds a run with another device, mode or set of input frames "
                f"({saved}), not {current}: use a new --output-root"
            )
            raise ValueError(msg)
        return
    path.write_text(json.dumps(current))


def main() -> int:
    """Run the plan for one weather product and write the combined losses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-product", choices=WEATHER_PRODUCTS, default="ens_mean")
    parser.add_argument("--output-root", type=Path, default=LAG_FEATURES_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--reproduction-check",
        action="store_true",
        help="Refit the ENS-mean B0 at lead-day 1 on --device, check it, and stop.",
    )
    arguments = parser.parse_args()
    product: WeatherProduct = arguments.weather_product
    if arguments.reproduction_check:
        return reproduction_check(
            root=arguments.output_root,
            device=arguments.device,
            max_workers=arguments.max_workers,
        )
    directory = arguments.output_root / product
    suffix = run_suffix(smoke=arguments.smoke)
    final = directory / f"losses_{product}{suffix}.parquet"
    refuse_to_overwrite(paths=[final])
    checkpoints = directory / f"checkpoints{suffix}"
    checkpoints.mkdir(parents=True, exist_ok=True)
    check_manifest(
        directory=checkpoints,
        device=arguments.device,
        smoke=arguments.smoke,
        inputs=input_fingerprint(root=arguments.output_root, product=product),
    )
    context = Context(
        checkpoint_dir=checkpoints,
        device=arguments.device,
        max_workers=arguments.max_workers,
        smoke=arguments.smoke,
    )
    runner = run_ens_mean if product == "ens_mean" else run_ifs_single
    losses = combine(results=runner(context=context, root=arguments.output_root, product=product))
    write_parquet_atomic(frame=losses, path=final)
    _LOG.info("wrote %s (%d rows)", final, losses.height)
    return 0


if __name__ == "__main__":
    sys.exit(main())
