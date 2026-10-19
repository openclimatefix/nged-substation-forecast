"""Score the arms of experiment 99 and print every number the write-up quotes.

Scores the four arms (raw, and with the spread factor 1.15 about the median as a labelled
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
from studies.bootstrap import (
    bootstrap_absolute,
    bootstrap_difference,
    bootstrap_difference_at_level,
)
from studies.experiment_store import losses_for_bootstrap, read_predictions
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
    "climatology": "ref_climatology",
}
SPREAD_FACTOR: Final[float] = 1.15
VERDICT_LEVEL: Final[float] = 95.0
N_RESAMPLES: Final[int] = 20_000
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
    return {name: with_losses(frame=frame, limits=limits) for name, frame in aligned.items()}


def scope(*, arms: Mapping[str, pl.DataFrame], low: float, high: float) -> dict[str, pl.DataFrame]:
    """Restrict every arm to the rows whose lead is in [low, high)."""
    return {
        name: frame.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
        for name, frame in arms.items()
    }


def losses_of(*, arms: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Return the loss frame `studies.bootstrap` reads, one row per arm and forecast row."""
    return losses_for_bootstrap(
        arm_losses={
            name: frame.select(
                "time_series_id", "power_fcst_init_time", "valid_time", "seed", *METRICS
            )
            for name, frame in arms.items()
        }
    )


def absolute_table(*, arms: Mapping[str, pl.DataFrame], metric: str) -> pl.DataFrame:
    """Return each arm's mean metric with its 95% interval."""
    losses = losses_of(arms=arms)
    rows = []
    for name in arms:
        interval = bootstrap_absolute(losses=losses, arm=name, metric=metric)
        rows.append(
            {
                "arm": name,
                "value": interval["value"],
                "lower_95": interval["lower_95"],
                "upper_95": interval["upper_95"],
            }
        )
    return pl.DataFrame(rows)


def contrast_table(
    *, arms: Mapping[str, pl.DataFrame], pairs: list[tuple[str, str, str]]
) -> pl.DataFrame:
    """Return the paired contrasts (treatment minus reference) with 95% intervals, per metric."""
    losses = losses_of(arms=arms)
    rows = []
    for label, treatment, reference in pairs:
        for metric in ("qs", "mae", "tw_p99", "tw_p95", "tw_p05"):
            interval = bootstrap_difference(
                losses=losses, treatment=treatment, reference=reference, metric=metric
            )
            rows.append(
                {
                    "contrast": label,
                    "metric": metric,
                    "difference": interval["difference"],
                    "lower_95": interval["lower_95"],
                    "upper_95": interval["upper_95"],
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
    losses = losses_of(arms=arms)
    low, high = bootstrap_difference_at_level(
        losses=losses,
        treatment="c1",
        reference="p3",
        metric="qs",
        level=VERDICT_LEVEL,
        n_resamples=N_RESAMPLES,
    )
    c1 = contrasts.filter(
        (pl.col("contrast").str.starts_with("C1 c1")) & (pl.col("metric") == "qs")
    )
    difference = float(c1["difference"][0])
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
                    "coverage": float((frame["actual"] <= frame[column]).mean()),
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
