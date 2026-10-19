"""Score the arms of experiment 99 and print every number the write-up quotes.

Scores the arms (raw, and with the spread factor 1.15 about the median as a labelled
sensitivity) and the stored references of experiment 87 on the shared rows of
`fold_a_daily_gap_v1` at the 16 Primary substations. Prints the absolute quantile score with a 95%
interval, the planned contrasts, the tail-calibration tables, the scores by lead band and month,
and the threshold-weighted CRPS. It writes the tables behind the figures to `results/`.

    uv run python ml-experiments/00099-*/e00099_report.py
"""

from collections.abc import Mapping
from typing import Final

import e00099_common as common
import numpy as np
import polars as pl
from studies.experiment_store import read_predictions
from studies.tail_calibration import (
    TAIL_LEVELS,
    align_arms,
    tail_calibration_markdown,
    tail_calibration_table,
)

POINTS: Final[float] = 100.0
"""A quantile score is a fraction of capacity; the page quotes points of capacity."""
RESULTS_DIR = common.WORK_DIR.parent / "results"
ARMS: Final[tuple[str, ...]] = ("p1", "p2", "p3", "p3m", "c1")
REFERENCES: Final[Mapping[str, str]] = {
    "bg2_weekly": "p_s4_weekly",
    "manual_heuristic": "ref_manual_heuristic",
}
SPREAD_FACTOR: Final[float] = 1.15
VERDICT_LEVEL: Final[float] = 95.0
N_RESAMPLES: Final[int] = 20_000
BOOTSTRAP_SEED: Final[int] = 0
VERDICT_THRESHOLD: Final[float] = 0.05
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("<12 h", 0.0, 12.0),
    ("12-36 h", 12.0, 36.0),
    ("36-96 h", 36.0, 96.0),
    (">=96 h", 96.0, 10_000.0),
)
DECISION_BAND: Final[tuple[float, float]] = (12.0, 36.0)
METRICS: Final[tuple[str, ...]] = ("qs", "mae", "tw_p99", "tw_p95", "tw_p05")


def widen(*, frame: pl.DataFrame, factor: float) -> pl.DataFrame:
    """Widen the quantile columns about the median by `factor`, as experiment 87 does."""
    median = pl.col("q50")
    return frame.with_columns(
        [
            (median + factor * (pl.col(c) - median)).cast(pl.Float32).alias(c)
            for c in common.LEVEL_COLUMNS
            if c != "q50"
        ]
    )


def thresholds() -> pl.DataFrame:
    """Return each series' 99th, 95th, and 5th percentile of power over capacity before 2025-07."""
    series = common.primary_series()
    arrays = common.load_power_arrays(series=series)
    rows = []
    for row in series.iter_rows(named=True):
        power = arrays[row["site"]].power
        values = power[: int(common.grid_index(times=common.LAST_TRAINING_VALID)) + 1]
        values = values[np.isfinite(values)]
        rows.append(
            {
                "time_series_id": row["time_series_id"],
                "p99": float(np.quantile(values, 0.99)),
                "p95": float(np.quantile(values, 0.95)),
                "p05": float(np.quantile(values, 0.05)),
            }
        )
    return pl.DataFrame(rows).cast({"time_series_id": pl.Int32})


def with_losses(*, frame: pl.DataFrame, limits: pl.DataFrame) -> pl.DataFrame:
    """Add the per-row metrics (`METRICS`), in points of capacity, to a forecast frame."""
    joined = frame.join(limits, on="time_series_id", how="left", validate="m:1")
    quantiles = joined.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = joined["actual"].to_numpy().astype(np.float64)
    qs = common.pinball_quantile_score(quantiles=quantiles, actual=actual) * POINTS
    out = {"qs": qs, "mae": np.abs(quantiles[:, common.MEDIAN] - actual) * POINTS}
    for name in ("p99", "p95"):
        floor = joined[name].to_numpy().astype(np.float64)
        out[f"tw_{name}"] = (
            common.pinball_quantile_score(
                quantiles=np.maximum(quantiles, floor[:, None]), actual=np.maximum(actual, floor)
            )
            * POINTS
        )
    cap = joined["p05"].to_numpy().astype(np.float64)
    out["tw_p05"] = (
        common.pinball_quantile_score(
            quantiles=np.minimum(quantiles, cap[:, None]), actual=np.minimum(actual, cap)
        )
        * POINTS
    )
    return frame.with_columns([pl.Series(k, v) for k, v in out.items()])


def read_all() -> dict[str, pl.DataFrame]:
    """Read the arms, their widened variants, and the references on their shared rows."""
    series_ids = common.primary_series()["time_series_id"].to_list()
    arms: dict[str, pl.DataFrame] = {
        name: read_predictions(
            experiment_id=common.EXPERIMENT_ID, name=name, row_set_id=common.ROW_SET_ID
        )
        for name in ARMS
    }
    for name in ARMS:
        arms[f"{name}_x115"] = widen(frame=arms[name], factor=SPREAD_FACTOR)
    for label, stored in REFERENCES.items():
        frame = read_predictions(
            experiment_id=87, name=stored, row_set_id=common.ROW_SET_ID
        ).filter(pl.col("time_series_id").is_in(series_ids))
        arms[label] = frame
    aligned = align_arms(arms=arms)
    limits = thresholds()
    return align_rows(
        arms={name: with_losses(frame=frame, limits=limits) for name, frame in aligned.items()}
    )


def scope(*, arms: Mapping[str, pl.DataFrame], low: float, high: float) -> dict[str, pl.DataFrame]:
    """Restrict every arm to the rows whose lead is in [low, high)."""
    return {
        name: frame.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
        for name, frame in arms.items()
    }


def month_sums(*, values: np.ndarray, months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the sum of `values` and the row count in each origin month (month index 0..n-1)."""
    n = int(months.max()) + 1
    return np.bincount(months, weights=values, minlength=n), np.bincount(months, minlength=n)


def month_interval(
    *, sums: np.ndarray, counts: np.ndarray, level: float = 95.0
) -> tuple[float, float, float]:
    """Return a mean and its interval from resampling whole months with replacement.

    The mean of a resample is the sum of the drawn months' sums over the sum of their row counts,
    which equals the mean of the drawn months' rows.

    Args:
        sums: Each month's sum of the metric.
        counts: Each month's row count.
        level: The interval's coverage in percent.

    Returns:
        The mean over all rows, and the lower and upper bounds.
    """
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.integers(0, len(sums), size=(N_RESAMPLES, len(sums)))
    drawn = np.zeros((N_RESAMPLES, len(sums)))
    for column in range(len(sums)):
        drawn[:, column] = (draws == column).sum(axis=1)
    resampled = (drawn @ sums) / (drawn @ counts)
    tail = (100.0 - level) / 2.0
    lower, upper = np.percentile(resampled, [tail, 100.0 - tail])
    return float(sums.sum() / counts.sum()), float(lower), float(upper)


def months_of(*, frame: pl.DataFrame) -> np.ndarray:
    """Return each row's origin month as an index from 0 (2025-07)."""
    origin = frame["power_fcst_init_time"]
    return ((origin.dt.year() - 2025) * 12 + origin.dt.month() - 7).to_numpy().astype(np.int64)


def align_rows(*, arms: Mapping[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Sort every arm by the row keys, so that row `i` is the same forecast row in each arm.

    Args:
        arms: Each arm's forecast rows.

    Returns:
        Each arm's rows, sorted by `common.KEYS`.

    Raises:
        ValueError: If the arms do not hold the same keys.
    """
    sorted_arms = {name: frame.sort(*common.KEYS) for name, frame in arms.items()}
    first = next(iter(sorted_arms.values())).select(*common.KEYS)
    for name, frame in sorted_arms.items():
        if not frame.select(*common.KEYS).equals(first):
            msg = f"arm {name} does not hold the same keys as the others"
            raise ValueError(msg)
    return sorted_arms


def absolute_table(*, arms: Mapping[str, pl.DataFrame], metric: str) -> pl.DataFrame:
    """Return each arm's mean metric with its 95% interval."""
    rows = []
    for name, frame in arms.items():
        sums, counts = month_sums(values=frame[metric].to_numpy(), months=months_of(frame=frame))
        value, lower, upper = month_interval(sums=sums, counts=counts)
        rows.append({"arm": name, "value": value, "lower_95": lower, "upper_95": upper})
    return pl.DataFrame(rows)


def difference_interval(
    *, arms: Mapping[str, pl.DataFrame], treatment: str, reference: str, metric: str, level: float
) -> tuple[float, float, float]:
    """Return the mean paired difference (treatment minus reference) and its interval."""
    frame = arms[treatment]
    difference = frame[metric].to_numpy() - arms[reference][metric].to_numpy()
    sums, counts = month_sums(values=difference, months=months_of(frame=frame))
    return month_interval(sums=sums, counts=counts, level=level)


def contrast_table(
    *, arms: Mapping[str, pl.DataFrame], pairs: list[tuple[str, str, str]]
) -> pl.DataFrame:
    """Return the paired contrasts (treatment minus reference) with 95% intervals, per metric."""
    rows = []
    for label, treatment, reference in pairs:
        for metric in METRICS:
            difference, lower, upper = difference_interval(
                arms=arms,
                treatment=treatment,
                reference=reference,
                metric=metric,
                level=VERDICT_LEVEL,
            )
            rows.append(
                {
                    "contrast": label,
                    "metric": metric,
                    "difference": difference,
                    "lower_95": lower,
                    "upper_95": upper,
                }
            )
    return pl.DataFrame(rows)


def verdict(*, difference: float, lower: float, upper: float) -> str:
    """Apply the pre-registered decision rule to the interval of contrast C1."""
    if upper < -VERDICT_THRESHOLD:
        return "supported"
    if upper < 0.0:
        return "small gain"
    # From here the upper end is at or above zero: no gain is established.
    if lower > -VERDICT_THRESHOLD:
        return "refuted"
    return "inconclusive"


def main() -> None:
    """Print the scores, contrasts, and tail tables, and write the tables behind the figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arms = read_all()
    names = list(arms)
    print(f"arms: {names}")
    print(
        f"rows per arm: {arms['p1'].height:,}; series {arms['p1']['time_series_id'].n_unique()}; "
        f"runs {arms['p1']['power_fcst_init_time'].n_unique()}"
    )
    pairs = [
        ("C1 c1 - p3 (loss)", "c1", "p3"),
        ("C5 c1 - p3m (loss, same pooling)", "c1", "p3m"),
        ("C2 p2 - p1 (6 members)", "p2", "p1"),
        ("C3 p3 - p2 (optimiser)", "p3", "p2"),
        ("C4 c1 - p1 (whole change)", "c1", "p1"),
        ("C1 with spread 1.15", "c1_x115", "p3_x115"),
        ("c1 - bg2_weekly (reference)", "c1", "bg2_weekly"),
        ("p1 - bg2_weekly (reference)", "p1", "bg2_weekly"),
        ("c1 - manual_heuristic", "c1", "manual_heuristic"),
    ]
    for metric in METRICS:
        table = absolute_table(arms=arms, metric=metric)
        table.write_csv(RESULTS_DIR / f"absolute_{metric}.csv")
        print(f"\nAbsolute {metric} (points of capacity, all leads)\n{table}")
    contrasts = contrast_table(arms=arms, pairs=pairs)
    contrasts.write_csv(RESULTS_DIR / "contrasts.csv")
    print(f"\nContrasts, all leads\n{contrasts}")
    difference, low, high = difference_interval(
        arms=arms, treatment="c1", reference="p3", metric="qs", level=VERDICT_LEVEL
    )
    word = verdict(difference=difference, lower=low, upper=high)
    print(
        f"\nVERDICT contrast C1 (qs): {difference:+.4f} ({low:+.4f} to {high:+.4f}) at "
        f"{VERDICT_LEVEL}% with {N_RESAMPLES} resamples -> {word}"
    )
    # Score by lead band.
    band_rows = []
    for label, band_low, band_high in LEAD_BANDS:
        banded = scope(arms=arms, low=band_low, high=band_high)
        table = absolute_table(arms=banded, metric="qs").with_columns(band=pl.lit(label))
        band_rows.append(table)
        print(f"\nQuantile score, lead {label}\n{table}")
    pl.concat(band_rows).write_csv(RESULTS_DIR / "by_lead_band.csv")
    # Tail calibration: all leads, and the decision band.
    for title, (band_low, band_high) in (
        ("all leads", (0.0, 10_000.0)),
        ("12-36 h", DECISION_BAND),
    ):
        banded = scope(arms=arms, low=band_low, high=band_high)
        tables = {name: tail_calibration_table(frame=frame) for name, frame in banded.items()}
        print(f"\nTail calibration, {title}")
        print(tail_calibration_markdown(tables=tables, unit="points of capacity"))
        pl.concat(
            [t.with_columns(arm=pl.lit(n), scope=pl.lit(title)) for n, t in tables.items()]
        ).write_csv(RESULTS_DIR / f"tails_{title.replace(' ', '_')}.csv")
    # Reliability: coverage of all 11 levels.
    rows = []
    for name, frame in arms.items():
        for level, column in zip(common.LEVELS, common.LEVEL_COLUMNS, strict=True):
            rows.append(
                {
                    "arm": name,
                    "level": float(level),
                    "coverage": float(
                        np.mean(frame["actual"].to_numpy() <= frame[column].to_numpy())
                    ),
                }
            )
    pl.DataFrame(rows).write_csv(RESULTS_DIR / "reliability.csv")
    print("\nReliability (share of actuals at or below the quantile)")
    print(pl.DataFrame(rows).pivot(on="arm", index="level", values="coverage"))
    # By month.
    monthly = []
    for name, frame in arms.items():
        monthly.append(
            frame.group_by(pl.col("power_fcst_init_time").dt.strftime("%Y-%m").alias("month"))
            .agg(pl.col("qs").mean())
            .with_columns(arm=pl.lit(name))
        )
    pl.concat(monthly).write_csv(RESULTS_DIR / "by_month.csv")
    print("\nTAIL LEVELS", TAIL_LEVELS)


if __name__ == "__main__":
    main()
