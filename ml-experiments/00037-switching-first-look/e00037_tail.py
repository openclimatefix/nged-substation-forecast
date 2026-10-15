"""Q2: how much of the stack's score falls in half-hours of logged switching?

Scores the stored predictions of experiment 00019's stack (three seeds, nine quantiles) per case,
with the quantile score and the threshold-weighted quantile score at each series' historical 99th
percentile (experiment 00033's `thresholds.parquet`). The scoring functions are copied from
experiment 00033's `e00033_common.py`, because an experiment may not import another experiment.
Cases are split by whether the valid time lies inside a confirmed transfer, inside another logged
interval, or neither, on that series.
"""

from typing import Final

import numpy as np
import polars as pl
from e00037_common import ADJACENCY, WORK_DIR  # noqa: F401
from e00037_grid import GRID_LENGTH, GRID_START
from studies.sources import ML_EXPERIMENTS_DATA_DIR

LEVELS: Final[np.ndarray] = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
)
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
"""Probability mass of each level: the interval between the mid-points to its neighbours."""
STACK_PATH: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00019-primary-demand-stack-vs-manual-heuristic"
    / "predictions"
    / "stack"
    / "predictions.parquet"
)
THRESHOLDS_PATH: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00033-best-guess-architecture-vs-production"
    / "work"
    / "thresholds.parquet"
)
FITS_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00017-training-window-and-retraining-rule-drift-demand" / "fits"
)


def quantile_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Twice the mass-weighted mean pinball loss of the 9 levels, per case."""
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)


def threshold_weighted_score(
    *, quantiles: np.ndarray, actual: np.ndarray, threshold: np.ndarray
) -> np.ndarray:
    """The quantile score after the forecast and the observation are floored at the threshold."""
    return quantile_score(
        quantiles=np.maximum(quantiles, threshold[:, None]), actual=np.maximum(actual, threshold)
    )


def case_classes(*, oracle: pl.DataFrame, cases: pl.DataFrame) -> pl.Series:
    """Label each case confirmed (two-sided or one-sided), logged but unconfirmed, or other."""
    label = np.full(cases.height, "other", dtype=object)
    series = cases["time_series_id"].to_numpy()
    stamp = cases["stamp"].to_numpy()
    oracle = oracle.with_columns(
        label=pl.when(pl.col("kind") == "two_sided")
        .then(pl.lit("confirmed_two_sided"))
        .otherwise(pl.lit("confirmed_one_sided"))
    )
    for row in oracle.sort("confirmed").iter_rows(named=True):  # confirmed written last, wins
        name = row["label"] if row["confirmed"] else "logged_unconfirmed"
        for sid in (row["series_a"], row["series_b"]):
            if sid is None:
                continue
            inside = (
                (series == sid)
                & (stamp >= row["first_stamp"])
                & (stamp < row["last_stamp_exclusive"])
            )
            label[inside] = name
    return pl.Series("class", label.tolist())


def main() -> None:
    """Score the stack by class and print and save the shares."""
    oracle = pl.read_parquet(WORK_DIR / "oracle_real.parquet").filter("primary")
    thresholds = pl.read_parquet(THRESHOLDS_PATH).select(
        pl.col("time_series_id").cast(pl.Int64), "p99"
    )
    actual = pl.concat(
        [
            pl.read_parquet(FITS_DIR / f"{p.stem}.parquet").select(
                "time_series_id", "valid_time", "target"
            )
            for p in FITS_DIR.glob("*.parquet")
        ]
    ).select(
        pl.col("time_series_id").cast(pl.Int64), "valid_time", pl.col("target").cast(pl.Float64)
    )
    stack = pl.read_parquet(STACK_PATH).with_columns(pl.col("time_series_id").cast(pl.Int64))
    cases = (
        stack.join(actual, on=["time_series_id", "valid_time"])
        .join(thresholds, on="time_series_id")
        .with_columns(
            stamp=(pl.col("valid_time").dt.epoch("s") - int(GRID_START.timestamp())) // 1800
        )
        .filter(
            pl.col("stamp")
            < (
                pl.datetime(2026, 3, 26, time_zone="UTC").dt.epoch("s")
                - int(GRID_START.timestamp())
            )
            // 1800
        )
        .filter(pl.col("stamp").is_between(0, GRID_LENGTH - 1))
    )
    quantiles = np.sort(cases.select(LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1)
    target = cases["target"].to_numpy()
    threshold = cases["p99"].to_numpy()
    cases = cases.with_columns(
        qs=pl.Series(quantile_score(quantiles=quantiles, actual=target) * 100.0),
        tw_p99=pl.Series(
            threshold_weighted_score(quantiles=quantiles, actual=target, threshold=threshold)
            * 100.0
        ),
    )
    cases = cases.with_columns(case_classes(oracle=oracle, cases=cases))
    total = cases.select(n=pl.len(), qs=pl.col("qs").sum(), tw=pl.col("tw_p99").sum())
    by_class = (
        cases.group_by("class")
        .agg(
            n=pl.len(),
            qs=pl.col("qs").sum(),
            tw=pl.col("tw_p99").sum(),
            qs_mean=pl.col("qs").mean(),
            tw_mean=pl.col("tw_p99").mean(),
        )
        .with_columns(
            share_cases=pl.col("n") / total["n"][0],
            share_qs=pl.col("qs") / total["qs"][0],
            share_tw=pl.col("tw") / total["tw"][0],
        )
        .sort("class")
    )
    by_class.write_parquet(WORK_DIR / "tail_shares.parquet")
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    print(by_class)
    conserved = cases.filter(pl.col("class").str.starts_with("confirmed"))
    per_series = (
        conserved.group_by("time_series_id")
        .agg(n=pl.len(), tw=pl.col("tw_p99").sum())
        .with_columns(share_of_conserved_tw=pl.col("tw") / conserved["tw_p99"].sum())
        .sort("tw", descending=True)
    )
    per_series.write_parquet(WORK_DIR / "tail_conserved_by_series.parquet")
    print(per_series)
    confirmed = cases.filter(pl.col("class").str.starts_with("confirmed"))
    other = cases.filter(~pl.col("class").str.starts_with("confirmed"))
    unconfirmed = cases.filter(pl.col("class") == "logged_unconfirmed")
    print(
        f"confirmed (both kinds): share of cases {confirmed.height / cases.height:.4f}, "
        f"quantile score {confirmed['qs'].sum() / cases['qs'].sum():.3f}, "  # ty: ignore[unsupported-operator]
        f"tw_p99 {confirmed['tw_p99'].sum() / cases['tw_p99'].sum():.3f}"  # ty: ignore[unsupported-operator]
    )
    print(
        f"tw_p99 per case, confirmed / logged-unconfirmed = "
        f"{confirmed['tw_p99'].mean() / unconfirmed['tw_p99'].mean():.2f}; "  # ty: ignore[unsupported-operator]
        f"confirmed / other = {confirmed['tw_p99'].mean() / other['tw_p99'].mean():.2f}"  # ty: ignore[unsupported-operator]
    )
    rest = confirmed.filter(~pl.col("time_series_id").is_in([17, 18]))
    print(
        f"confirmed without series 17 and 18: {rest.height} cases, tw_p99 "
        f"{rest['tw_p99'].sum() / cases['tw_p99'].sum():.4f} of the total"  # ty: ignore[unsupported-operator]
    )


if __name__ == "__main__":
    main()
