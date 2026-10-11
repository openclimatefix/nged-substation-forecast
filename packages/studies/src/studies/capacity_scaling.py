"""Divide a forecast's target by a capacity scale and score it in megawatts, for issue #1143.

**Every arm is scored on its megawatt error over the plant's fixed 99th-percentile power, whatever
scale it divided by.** Dividing `power_mw`, `cap_mw`, and `effective_capacity_mw` by the row's scale
`s` before `studies.cross_validation.out_of_fold_losses` gives exactly that score with no change to
the fit loop. The training target becomes `y / s`. The fraction error becomes
`|y/s - yhat| / (P99/s) = |y - s * yhat| / P99`. Clamping `yhat` at `cap/s` and multiplying by `s`
equals clamping `s * yhat` at the cap, because `s > 0`. The continuous ranked probability score is
positively homogeneous of degree 1, so its fraction reduces the same way. An estimator therefore
cannot shrink its errors by inflating its capacity on its worst days.

The module also holds the controls: the negative control that deranges whole months of the scale
within each plant, the oracle scale of the positive control, the rescaled-forecast arms that fit
nothing, and the checks that the scoring and the refit reproduce the lag study's baseline.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from studies.cross_validation import (
    HyperParameters,
    DeviceType,
    out_of_fold_losses,
)

SCALED_COLUMNS: tuple[str, ...] = ("power_mw", "cap_mw", "effective_capacity_mw")
"""The three columns of the lag frame that carry megawatts and are divided by the scale."""

LOSS_COLUMNS_IN_MEGAWATTS: tuple[str, ...] = (
    "absolute_error_mw",
    "signed_error_mw",
    "crps_mw",
    "absolute_error_capped_mw",
    "signed_error_capped_mw",
    "crps_capped_mw",
)
"""The loss columns in the scaled unit, which mean nothing across arms and so are dropped."""

SCORED_COLUMN: str = "absolute_error_capped_fraction_of_capacity"
"""The column every contrast scores."""


def scaled_frame(*, frame: pl.DataFrame, scale_mw: np.ndarray) -> pl.DataFrame:
    """Divide the lag frame's three megawatt columns by a per-row scale.

    Args:
        frame: The lag frame, with `power_mw`, `cap_mw`, and `effective_capacity_mw`.
        scale_mw: One positive scale per row, in the frame's row order.

    Returns:
        The frame with the three columns divided in Float64, plus `capacity_scale_mw`.

    Raises:
        ValueError: If a scale is not positive and finite, or the length differs from the frame's.
    """
    if len(scale_mw) != frame.height:
        msg = f"{len(scale_mw)} scales for {frame.height} rows"
        raise ValueError(msg)
    if not np.all(np.isfinite(scale_mw) & (scale_mw > 0)):
        msg = "every scale must be positive and finite"
        raise ValueError(msg)
    scale = pl.Series("capacity_scale_mw", scale_mw, dtype=pl.Float64)
    return frame.with_columns(
        capacity_scale_mw=scale,
        **{name: pl.col(name).cast(pl.Float64) / scale for name in SCALED_COLUMNS},
    )


def drop_scaled_loss_columns(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Drop the loss columns that are in the scaled unit.

    Args:
        losses: Per-row losses from a scaled frame.

    Returns:
        The losses without `LOSS_COLUMNS_IN_MEGAWATTS`. The fraction columns remain.
    """
    return losses.drop(*(name for name in LOSS_COLUMNS_IN_MEGAWATTS if name in losses.columns))


def derange_months(
    *,
    months_by_plant: Mapping[int, Sequence[str]],
    low_months: Mapping[int, Sequence[str]],
    seed: int,
) -> dict[int, dict[str, str]]:
    """Choose, within each plant, a donor month for every month so that no month keeps its own.

    A low month never takes another low month as donor, so a real low month is never given another
    real low month's scale.

    Args:
        months_by_plant: Each plant index's scored months.
        low_months: Each plant's real low months, a subset of its scored months.
        seed: The generator's seed.

    Returns:
        For each plant, a map from each month to the month whose scale it takes.

    Raises:
        ValueError: If a plant has fewer than two months or no valid derangement turns up in 1,000
            tries.
    """
    rng = np.random.default_rng(seed)
    result: dict[int, dict[str, str]] = {}
    for plant, months in months_by_plant.items():
        if len(months) < 2:
            msg = f"plant index {plant} has fewer than two months to derange"
            raise ValueError(msg)
        low = set(low_months.get(plant, ()))
        for _ in range(1000):
            donors = list(rng.permutation(list(months)))
            if all(
                donor != month and not (month in low and donor in low)
                for month, donor in zip(months, donors, strict=True)
            ):
                result[plant] = dict(zip(months, donors, strict=True))
                break
        else:
            msg = f"no derangement of plant index {plant}'s months turned up"
            raise ValueError(msg)
    return result


def deranged_scale(
    *,
    scale_mw: np.ndarray,
    dates: np.ndarray,
    donors: Mapping[int, Mapping[str, str]],
) -> np.ndarray:
    """Replace each scored day's scale by the same day-of-month's scale in the donor month.

    A donor month shorter than the day's month repeats its last day.

    Args:
        scale_mw: The scale of shape (plants, days).
        dates: One `datetime64[D]` per day.
        donors: `derange_months`' result.

    Returns:
        The scale with every day in a deranged month replaced; other days are unchanged.
    """
    months = dates.astype("datetime64[M]").astype(str)
    first_of_month = {month: int(np.flatnonzero(months == month)[0]) for month in set(months)}
    length = {month: int((months == month).sum()) for month in set(months)}
    out = scale_mw.copy()
    for plant, mapping in donors.items():
        for month, donor in mapping.items():
            if month not in first_of_month or donor not in first_of_month:
                continue
            for offset in range(length[month]):
                source = first_of_month[donor] + min(offset, length[donor] - 1)
                out[plant, first_of_month[month] + offset] = scale_mw[plant, source]
    return out


def multiplied_forecast_scores(
    *,
    baseline_losses: pl.DataFrame,
    frame: pl.DataFrame,
    factor: np.ndarray,
) -> pl.DataFrame:
    """Score a baseline forecast multiplied by a per-row factor, normalised on the training rows.

    Each row's factor is divided by the mean factor over the plant's training rows for the fold
    that scores the row (every other fold's rows that are not constrained), so a factor with no
    information leaves the forecast unchanged on average. No model is fitted.

    Args:
        baseline_losses: The baseline's per-row losses at one setting, with `site`, `time`,
            `seed`, `fold`, `actual`, and `prediction`.
        frame: The lag frame, with `site`, `time`, `fold`, `constrained`, `cap_mw`, and
            `effective_capacity_mw`.
        factor: One positive factor per frame row, in the frame's row order.

    Returns:
        Per-row losses with `site`, `time`, `month`, `fold`, `seed`, and `SCORED_COLUMN`, in the
        baseline's rows.
    """
    keyed = frame.select("site", "time", "fold", "constrained", "cap_mw", "effective_capacity_mw").with_columns(
        factor=pl.Series(factor, dtype=pl.Float64)
    )
    folds = sorted(keyed["fold"].unique().to_list())
    training_means = pl.concat(
        [
            keyed.filter((pl.col("fold") != fold) & ~pl.col("constrained"))
            .group_by("site")
            .agg(training_mean=pl.col("factor").mean())
            .with_columns(fold=pl.lit(fold, dtype=pl.Int32))
            for fold in folds
        ]
    )
    joined = (
        baseline_losses.select("site", "time", "month", "fold", "seed", "actual", "prediction")
        .join(keyed.drop("fold"), on=["site", "time"], how="left", maintain_order="left")
        .join(training_means, on=["site", "fold"], how="left", maintain_order="left")
    )
    forecast = pl.col("prediction") * pl.col("factor") / pl.col("training_mean")
    capped = pl.min_horizontal(forecast, pl.col("cap_mw").fill_null(float("inf")))
    return joined.select(
        "site",
        "time",
        "month",
        "fold",
        "seed",
        ((capped - pl.col("actual")).abs() / pl.col("effective_capacity_mw").cast(pl.Float64)).alias(
            SCORED_COLUMN
        ),
    )


def assert_constant_scale_reproduces_baseline(
    *,
    site_rows: pl.DataFrame,
    features: Sequence[str],
    hyper_parameters: HyperParameters,
    scale: float,
    tolerance_points: float,
    device: DeviceType,
) -> float:
    """Raise unless dividing by one constant leaves the baseline's score unchanged.

    **This check exists because under `reg:absoluteerror` a constant divisor only rescales the
    leaves, so the scaled arm must reproduce the baseline to float noise.** A larger difference
    means the scaled scoring is wrong. The check compares both fraction columns, the capped and the
    uncapped, on a plant that has an export cap and constrained rows.

    Args:
        site_rows: One plant's lag-frame rows with `fold`, `month`, `constrained`, `cap_mw`,
            `power_mw`, `effective_capacity_mw`, and the features.
        features: The baseline's feature columns.
        hyper_parameters: The setting to fit at.
        scale: The constant divisor.
        tolerance_points: The largest allowed difference in the mean, in points of capacity.
        device: XGBoost's device.

    Returns:
        The largest difference found, in points of capacity.

    Raises:
        ValueError: If the plant has no constrained row, or a difference exceeds the tolerance.
    """
    if not site_rows["constrained"].any():
        msg = "the constant-scale check needs a plant with constrained rows"
        raise ValueError(msg)
    scaled = scaled_frame(frame=site_rows, scale_mw=np.full(site_rows.height, scale))
    plain = out_of_fold_losses(
        site_rows=site_rows,
        features=features,
        target="power_mw",
        hyper_parameters=hyper_parameters,
        with_quantiles=False,
        device=device,
    )
    divided = out_of_fold_losses(
        site_rows=scaled,
        features=features,
        target="power_mw",
        hyper_parameters=hyper_parameters,
        with_quantiles=False,
        device=device,
    )
    worst = 0.0
    for column in ("absolute_error_fraction_of_capacity", SCORED_COLUMN):
        difference = abs(float(plain[column].mean()) - float(divided[column].mean())) * 100.0
        worst = max(worst, difference)
        if difference > tolerance_points:
            msg = (
                f"dividing by the constant {scale} moved the mean of {column} by {difference:.6f} "
                f"points, above the tolerance {tolerance_points}"
            )
            raise ValueError(msg)
    return worst


def assert_refit_reproduces(*, refit: pl.DataFrame, published: pl.DataFrame) -> None:
    """Raise unless a refit's per-row scores equal the published ones bit for bit.

    Args:
        refit: Per-row losses from this study's refit, with `site`, `time`, `seed`, and
            `SCORED_COLUMN`.
        published: The lag study's per-row losses for the same arm and setting.

    Raises:
        ValueError: If the rows differ or any score differs at all.
    """
    keys = ["site", "time", "seed"]
    ours = refit.select(*keys, SCORED_COLUMN).sort(keys)
    theirs = published.select(*keys, SCORED_COLUMN).sort(keys)
    if ours.height != theirs.height or not ours.select(keys).equals(theirs.select(keys)):
        msg = f"the refit scores {ours.height} rows and the published run {theirs.height}, or their keys differ"
        raise ValueError(msg)
    if not ours[SCORED_COLUMN].equals(theirs[SCORED_COLUMN]):
        worst = float((ours[SCORED_COLUMN] - theirs[SCORED_COLUMN]).abs().max())
        msg = f"the refit differs from the published run (largest difference {worst:.3e})"
        raise ValueError(msg)


def oracle_scale(*, static_p99_mw: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Return the oracle scale of the positive control: the true factor times the static power.

    Args:
        static_p99_mw: Each plant's static power, shape (plants,).
        factor: The planted share of power left, shape (plants, days).

    Returns:
        An array of shape (plants, days).
    """
    return factor * static_p99_mw[:, None]
