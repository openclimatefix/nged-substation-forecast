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

**Device.** `--device cpu` (the default) or `cuda`; `--max-workers` sets how many plants fit at
once, each on `THREADS_PER_FIT` cores.

**Smoke test.** `--smoke` fits B0 and S2 on 150 rows per plant from folds 0 and 1, with 20 boosting
rounds, to check the plumbing. It is not a result.

Run it with `uv run python studies/lag_features/fit_lag_arms.py`.
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import polars as pl
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
    extra_columns_of,
    output_paths,
    shared_rows,
    target_date,
    weather_table,
)
from studies.arm_runner import Job, run_all
from studies.baselines import same_clock_hour_window
from studies.cross_validation import (
    N_FOLDS,
    PRIMARY_HYPER_PARAMETERS,
    SEEDS,
    SENSITIVITY_HYPER_PARAMETERS,
    DeviceType,
    HyperParameters,
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
    *(f"2025-{month:02d}" for month in range(1, 13)),
)
"""Phase 1 reads the first 13 calendar months, 2024-12 to 2025-12, and nothing else."""

SHORTLIST_CANDIDATES: Final[tuple[str, ...]] = (
    "IM",
    "CTX7",
    "W7",
    "Q30",
    "CK",
    "TF",
    "AN",
    "S3",
    "RP",
    "KS",
)
"""**The shortlist rule, fixed before any fit:** X is the candidate with the lowest phase-1 mean
absolute error over `SCREENING_MONTHS`. The excluded arms are B0, L1, L2, S2, N1 and N2 (phase 2
carries those regardless), T1 (an interpolation bound), PC (study-only, so a win could not ship),
the references and post-model corrections (none is fitted), and the global-only arms. Phase 2
carries X."""

WIDE_X_COLUMNS: Final[int] = 3
"""If X adds more columns than this, N2-k is fitted with as many random lags as X adds."""

REPRODUCTION_MAE_PERCENT: Final[float] = 8.771
"""The published mean absolute error of the ENS-mean B0 at lead-day 1 on the 35,263 shared rows, in
percentage points of capacity."""

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

SMOKE_ROWS_PER_FOLD: Final[int] = 150
"""The rows per plant and fold a smoke test keeps."""

SMOKE_ARMS: Final[tuple[str, ...]] = ("B0", "S2")
"""The arms a smoke test fits; S2 exercises the `{fold}` columns and the quantile path."""


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
            pl.col("fold").is_not_null() & ~pl.col("constrained") & pl.col(TARGET).is_not_null()
        )
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
            resid=(pl.col("observed_mw") - pl.col("predicted_mw")).mean(),
            observed_energy=pl.col("observed_mw").sum(),
            predicted_energy=pl.col("predicted_mw").sum(),
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
    grid = (
        daily.group_by("site")
        .agg(date=pl.date_ranges(pl.col("date").min(), pl.col("date").max(), interval="1d"))
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


def fit_arm(
    *,
    context: Context,
    dataset: pl.DataFrame,
    scope: str,
    lead_day: int,
    arm: str,
    setting: str,
    quantiles: bool,
    pooled: bool = False,
    lopo: bool = False,
) -> pl.DataFrame:
    """Fit one arm at one setting, or read it back if a checkpoint holds it.

    Args:
        context: The run's context.
        dataset: The site-sorted frame, with `fold`, `month`, `cap_mw`, `constrained` and capacity.
        scope: A label for the frame: `lead<N>`, `global`, or `control_s<percent>`.
        lead_day: The lead-day of the frame.
        arm: The arm.
        setting: `primary` or `sensitivity`.
        quantiles: Whether to fit the quantile models too.
        pooled: Whether `dataset` is one pooled model's frame (fitted as a single `site_rows`).
        lopo: Whether to fit the pooled frame leaving each plant out in turn (exploratory).

    Returns:
        The arm's per-row losses with `arm`, `setting`, `scope`, `lead_day`, `with_quantiles`,
        `actual` and `prediction` (in the target's own units) and
        `crps_capped_fraction_of_capacity`.
    """
    path = checkpoint_path(context=context, scope=scope, setting=setting, arm=arm)
    kind_path = path.with_suffix(".quantile.parquet" if quantiles else ".parquet")
    if kind_path.exists():
        return pl.read_parquet(kind_path)
    hyper_parameters = settings_for(context=context)[setting]
    features = features_of(arm=arm)
    if lopo:
        annotated = leave_one_plant_out(
            context=context, dataset=dataset, arm=arm, hyper_parameters=hyper_parameters
        ).with_columns(
            arm=pl.lit(arm),
            setting=pl.lit(setting),
            scope=pl.lit(scope),
            lead_day=pl.lit(lead_day, dtype=pl.Int32),
            with_quantiles=pl.lit(value=False),
        )
        partial = kind_path.with_suffix(".partial")
        annotated.write_parquet(partial)
        partial.replace(kind_path)
        return annotated
    if pooled:
        losses = out_of_fold_losses(
            site_rows=dataset,
            features=features,
            target=TARGET,
            hyper_parameters=hyper_parameters,
            with_quantiles=quantiles,
            device=context.device,
        ).with_columns(arm=pl.lit(arm), setting=pl.lit(setting), target=pl.lit(TARGET))
    else:
        job: Job = (arm, setting, TARGET, features, hyper_parameters, quantiles)
        losses = run_all(
            dataset=dataset, jobs=[job], max_workers=context.max_workers, device=context.device
        )
    annotated = (
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
    partial = kind_path.with_suffix(".partial")
    annotated.write_parquet(partial)
    partial.replace(kind_path)
    _LOG.info("saved %s", kind_path.name)
    return annotated


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
"""The arms whose 10% and 90% quantile predictions are saved, to measure coverage and width."""

INTERVAL_SEED: Final[int] = 0
"""The one seed the saved quantile predictions come from; coverage and width are descriptive."""

LOWER_LEVEL_INDEX: Final[int] = 0
UPPER_LEVEL_INDEX: Final[int] = 8
"""The positions of the 0.1 and 0.9 levels in `studies.cross_validation.QUANTILE_LEVELS`."""


def _plant_intervals(
    *, rows: pl.DataFrame, arm: str, hyper_parameters: HyperParameters, device: DeviceType
) -> pl.DataFrame:
    """Predict one plant's 10% and 90% quantiles out of fold, with `INTERVAL_SEED`.

    Args:
        rows: One plant's rows.
        arm: The arm.
        hyper_parameters: The setting to fit at.
        device: XGBoost's device.

    Returns:
        `site`, `time`, `month`, the actual power, the two quantiles held to the export cap, and
        the columns that classify the forecast sky, all as fractions of the plant's capacity.
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
        capped = clamp_to_cap(prediction=quantiles, cap_mw=test["cap_mw"])
        capacity = test["effective_capacity_mw"].cast(pl.Float64)
        parts.append(
            test.select("site", "time", "month", "nwp_ghi", "clear_sky_w_m2").with_columns(
                actual=test[TARGET].cast(pl.Float64) / capacity,
                lower=pl.Series(capped[:, LOWER_LEVEL_INDEX]) / capacity,
                upper=pl.Series(capped[:, UPPER_LEVEL_INDEX]) / capacity,
            )
        )
    return pl.concat(parts)


def fit_intervals(*, context: Context, dataset: pl.DataFrame, arm: str) -> pl.DataFrame:
    """Return an arm's out-of-fold 10% and 90% quantile predictions, or read back a checkpoint.

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
    result.write_parquet(path)
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


def shortlist(*, losses: pl.DataFrame) -> str:
    """Apply the shortlist rule.

    Args:
        losses: The lead-day 1 sweep's primary-setting losses.

    Returns:
        X: the arm with the lowest phase-1 mean absolute error, other than `SHORTLIST_CANDIDATES`.
    """
    candidates = phase1_table(losses=losses).filter(pl.col("arm").is_in(SHORTLIST_CANDIDATES))
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


def fit_all(
    *, context: Context, frames: dict[str, pl.DataFrame], fits: list[Fit]
) -> list[pl.DataFrame]:
    """Run each fit in order, reading back any a checkpoint holds.

    Args:
        context: The run's context.
        frames: The frames the fits name, by key.
        fits: The fits to run.

    Returns:
        The results, in the order of `fits`.
    """
    return [
        fit_arm(
            context=context,
            dataset=frames[fit.frame_key],
            scope=fit.scope,
            lead_day=fit.lead_day,
            arm=fit.arm,
            setting=fit.setting,
            quantiles=fit.quantiles,
            pooled=fit.pooled,
            lopo=fit.lopo,
        )
        for fit in fits
    ]


def sweep_fits(*, context: Context) -> list[Fit]:
    """Return the lead-day 1 sweep: every arm at the primary setting.

    Args:
        context: The run's context.

    Returns:
        The fits. B0, L1 and N2 carry their quantile models (S2 in a smoke test).
    """
    quantile_arms = ("S2",) if context.smoke else SWEEP_QUANTILE_ARMS
    arms = SMOKE_ARMS if context.smoke else SWEEP_ARMS
    return [
        Fit(
            "lead1",
            f"lead{FULL_SWEEP_LEAD_DAY}",
            FULL_SWEEP_LEAD_DAY,
            arm,
            "primary",
            arm in quantile_arms,
        )
        for arm in arms
    ]


def phase2_fits(*, chosen: str) -> list[Fit]:
    """Return phase 2's fits at lead-day 1: sensitivity, quantile, global and control fits.

    Args:
        chosen: The shortlist rule's X.

    Returns:
        The fits, with `X` replaced by `chosen` wherever an arm list names it.
    """
    lead, scope = FULL_SWEEP_LEAD_DAY, f"lead{FULL_SWEEP_LEAD_DAY}"

    def named(arms: tuple[str, ...]) -> list[str]:
        return [chosen if arm == "X" else arm for arm in arms]

    fits = [
        Fit("lead1", scope, lead, arm, "primary", quantiles=True)
        for arm in named(QUANTILE_PRIMARY_ARMS)
        if arm not in SWEEP_QUANTILE_ARMS
    ]
    fits += [
        Fit("lead1", scope, lead, arm, "sensitivity", arm in QUANTILE_SENSITIVITY_ARMS)
        for arm in named(PHASE2_POINT_ARMS)
    ]
    width = len(extra_columns_of(arm=chosen))
    if width > WIDE_X_COLUMNS:
        fits += [
            Fit("lead1", scope, lead, f"N2-{width}", setting, quantiles=False)
            for setting in ("primary", "sensitivity")
        ]
    fits += [
        Fit("global", "global", lead, arm, setting, arm in GLOBAL_QUANTILE_ARMS, pooled=True)
        for setting in ("primary", "sensitivity")
        for arm in named(GLOBAL_ARMS)
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
    *, root: Path, product: WeatherProduct, lead_days: tuple[int, ...]
) -> dict[str, pl.DataFrame]:
    """Read the built frame of each lead-day.

    Args:
        root: The output root.
        product: The weather product.
        lead_days: The lead-days.

    Returns:
        The frames, keyed `lead<N>`.
    """
    paths = output_paths(root=root, product=product, lead_days=lead_days)
    return {f"lead{lead}": pl.read_parquet(paths[f"day{lead}"]) for lead in lead_days}


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
        predictions.write_parquet(context.checkpoint_dir / "stage1_predictions.parquet")
        stage1_columns(frame=frame, hours=hours, predictions=predictions).write_parquet(
            derived_path
        )
    dataset = frame.join(
        pl.read_parquet(derived_path), on=["site", "time"], how="left", maintain_order="left"
    )
    raise_on_uncovered_months(coverage=calendar_month_coverage(frame=dataset))
    return dataset


def smoke_subsample(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Keep `SMOKE_ROWS_PER_FOLD` rows per plant from each of folds 0 and 1.

    Args:
        dataset: The lead-day 1 frame.

    Returns:
        The site-sorted subsample.
    """
    return (
        dataset.filter(pl.col("fold") < 2)
        .sort("site", "fold", "time")
        .group_by("site", "fold", maintain_order=True)
        .head(SMOKE_ROWS_PER_FOLD)
        .sort("site", "time")
    )


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
    if context.smoke:
        frames = {"lead1": smoke_subsample(dataset=dataset), "global": pooled_frame(frame=dataset)}
        lead = FULL_SWEEP_LEAD_DAY
        smoke_global = [
            Fit("global", "global", lead, "B0", "primary", False, True),
            Fit("global", "global", lead, "G-FP", "primary", False, True),
            Fit("global", "lopo", lead, "G-FP", "primary", False, lopo=True),
        ]
        fit_intervals(context=context, dataset=frames["lead1"], arm="B0")
        return fit_all(
            context=context, frames=frames, fits=sweep_fits(context=context) + smoke_global
        )
    frames = {"lead1": dataset}
    results = fit_all(context=context, frames=frames, fits=sweep_fits(context=context))
    sweep = pl.concat(results, how="vertical_relaxed")
    chosen = shortlist(losses=sweep)
    _write_shortlist(directory=context.checkpoint_dir, sweep=sweep, chosen=chosen)

    other_leads = tuple(lead for lead in ENS_MEAN_LEAD_DAYS if lead != FULL_SWEEP_LEAD_DAY)
    frames |= read_lead_frames(root=root, product=product, lead_days=other_leads)
    frames["global"] = pooled_frame(frame=dataset)
    paths = output_paths(root=root, product=product, lead_days=())
    for shift in POSITIVE_CONTROL_SHIFTS:
        percent = f"{round(shift * 100):02d}"
        frames[f"control{percent}"] = pl.read_parquet(paths[f"control{percent}"])
    fits = phase2_fits(chosen=chosen) + longer_lead_fits(
        lead_days=other_leads, arms=LONGER_LEAD_ARMS
    )
    return results + fit_all(context=context, frames=frames, fits=fits)


def run_ifs_single(*, context: Context, root: Path, product: WeatherProduct) -> list[pl.DataFrame]:
    """Fit B0 and L1 with IFS HRES at lead-days 1 and 2, point model, primary setting.

    Args:
        context: The run's context.
        root: The output root.
        product: The weather product (`ifs_single`).

    Returns:
        The results, in the order fitted.
    """
    frames = read_lead_frames(root=root, product=product, lead_days=IFS_LEAD_DAYS)
    fits = longer_lead_fits(lead_days=IFS_LEAD_DAYS, arms=REPLICATE_ARMS)
    return fit_all(context=context, frames=frames, fits=fits)


def _write_shortlist(*, directory: Path, sweep: pl.DataFrame, chosen: str) -> None:
    """Print the phase-1 ranking and the shortlist rule's X, and save them beside the checkpoints.

    Args:
        directory: The checkpoint directory.
        sweep: The sweep's primary-setting losses.
        chosen: X.
    """
    table = phase1_table(losses=sweep)
    (directory / "shortlist.json").write_text(
        json.dumps({"x": chosen, "phase1": table.to_dicts()}, indent=2)
    )
    _LOG.info("phase 1 ranking:\n%s", table)
    _LOG.info("the shortlist rule's X is %s", chosen)


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


def loss_checksum(*, losses: pl.DataFrame) -> str:
    """Return a checksum of the per-row losses, in a fixed row order.

    Args:
        losses: Per-row losses with `site`, `time`, `seed` and `METRIC`.

    Returns:
        The SHA-256 hex digest of the sorted per-row loss values' bytes.
    """
    ordered = losses.sort("site", "time", "seed")
    return hashlib.sha256(ordered[METRIC].cast(pl.Float64).to_numpy().tobytes()).hexdigest()


def reproduction_check(
    *, root: Path, device: DeviceType, max_workers: int, expected_checksum: str | None
) -> int:
    """Refit the ENS-mean B0 at lead-day 1 on the shared rows and compare it with the published run.

    The CPU refit must give `REPRODUCTION_MAE_PERCENT` (and `expected_checksum`, if given). If
    `device` is `cuda`, a GPU refit is reported beside it, so the device difference can be set
    against the published solar device range.

    Args:
        root: The output root, where `reproduction_check.md` is written.
        device: The device of the second refit; `cpu` runs the CPU refit alone.
        max_workers: How many plants fit at once.
        expected_checksum: The published per-row loss checksum, or `None` to print it only.

    Returns:
        0 on success.

    Raises:
        ValueError: If the CPU refit's mean or checksum differs from the published one.
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
    lines = ["# Reproduction check: ENS-mean B0 at lead-day 1 on the shared rows", ""]
    means = {}
    for fitted_on in dict.fromkeys(("cpu", device)):
        losses = run_all(dataset=frame, jobs=[job], max_workers=max_workers, device=fitted_on)
        means[fitted_on] = float(losses.select(pl.col(METRIC).mean()).item()) * 100
        checksum = loss_checksum(losses=losses)
        lines.append(
            f"- {fitted_on}: mean absolute error {means[fitted_on]:.3f}% of capacity on "
            f"{losses.select('site', 'time').n_unique()} rows; per-row loss checksum `{checksum}`."
        )
        if fitted_on == "cpu":
            if abs(means["cpu"] - REPRODUCTION_MAE_PERCENT) > REPRODUCTION_TOLERANCE_PERCENT:
                msg = (
                    f"CPU refit gives {means['cpu']:.4f}%, "
                    f"not the published {REPRODUCTION_MAE_PERCENT}%"
                )
                raise ValueError(msg)
            if expected_checksum is not None and checksum != expected_checksum:
                msg = f"CPU refit's per-row loss checksum {checksum} is not {expected_checksum}"
                raise ValueError(msg)
    if "cuda" in means:
        lines.append(f"- GPU minus CPU: {means['cuda'] - means['cpu']:+.3f} points.")
    report.write_text("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


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
        help="Refit the ENS-mean B0 at lead-day 1 on the CPU (and on --device) and stop.",
    )
    parser.add_argument("--expected-checksum", default=None, help="The published checksum.")
    arguments = parser.parse_args()
    product: WeatherProduct = arguments.weather_product
    if arguments.reproduction_check:
        return reproduction_check(
            root=arguments.output_root,
            device=arguments.device,
            max_workers=arguments.max_workers,
            expected_checksum=arguments.expected_checksum,
        )
    directory = arguments.output_root / product
    final = directory / f"losses_{product}.parquet"
    refuse_to_overwrite(paths=[final])
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    context = Context(
        checkpoint_dir=checkpoints,
        device=arguments.device,
        max_workers=arguments.max_workers,
        smoke=arguments.smoke,
    )
    runner = run_ens_mean if product == "ens_mean" else run_ifs_single
    losses = combine(results=runner(context=context, root=arguments.output_root, product=product))
    losses.write_parquet(final)
    _LOG.info("wrote %s (%d rows)", final, losses.height)
    return 0


if __name__ == "__main__":
    sys.exit(main())
