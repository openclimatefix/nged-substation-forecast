# ruff: noqa: E501
"""Score experiment 191's arms and print every number the write-up quotes.

Quantile score is twice the mean pinball loss over the 11 quantiles, in points of effective
capacity. Mean absolute error is of the mean forecast (the corrected reading). Rows are the
scored year's runs (`fold_a_daily_gap_v1`) at leads 0.5 to 168 hours, outside the hand-filtered
switching events. Intervals resample the 12 calendar months of the scored year (20,000 draws).

    uv run --no-sync python ml-experiments/00191-*/e00191_score.py [--variant gated]
"""

import argparse
import json

import e00191_common as common
import numpy as np
import polars as pl
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS

ARMS: tuple[str, ...] = ("a0", "h", "m", "p", "t", "jb")
LEVELS = np.array(QUANTILE_LEVELS)
N_RESAMPLES = 20_000
BAND_EDGES = ((0.0, 24.0, "0-24 h"), (24.0, 72.0, "24-72 h"), (72.0, 168.0, "72-168 h"))
TAILS = (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99))


def load(*, variant: str, site: str, period: str = "scored") -> pl.DataFrame:
    """Return an arm-stacked prediction frame restricted to the scored rows."""
    frame = pl.read_parquet(common.PREDS_DIR / variant / f"{period}_{site}.parquet")
    return frame.filter(
        (pl.col("lead_hours") > 0.0)
        & (pl.col("lead_hours") <= 168.0)
        & ~pl.col("switching_mask")
        & pl.col("actual").is_not_null()
    ).with_columns(month=pl.col("valid_time").dt.month() + 100 * pl.col("valid_time").dt.year())


def losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add per-row quantile score, absolute error and tail pinball losses."""
    quantiles = frame.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    error = actual[:, None] - quantiles
    pinball = np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error)
    columns = {
        "qs": 2.0 * pinball.mean(axis=1),
        "ae": np.abs(actual - frame["mean"].to_numpy().astype(np.float64)),
        "in90": ((actual >= quantiles[:, 1]) & (actual <= quantiles[:, 9])).astype(np.float64),
    }
    for name, level in TAILS:
        column = QUANTILE_COLUMNS.index(name)
        e = actual - quantiles[:, column]
        columns[f"pin_{name}"] = np.maximum(level * e, (level - 1.0) * e)
        columns[f"below_{name}"] = (actual < quantiles[:, column]).astype(np.float64)
    return frame.with_columns([pl.Series(k, v) for k, v in columns.items()])


def interval(
    *, a: pl.DataFrame, b: pl.DataFrame, column: str, level: float
) -> tuple[float, float, float]:
    """Return the mean of `b - a` and its month-resampled interval at `level` (two-sided)."""
    joined = a.select("run", "valid_time", "month", pl.col(column).alias("a")).join(
        b.select("run", "valid_time", pl.col(column).alias("b")),
        on=["run", "valid_time"],
        how="inner",
    )
    if joined.height == 0:
        return float("nan"), float("nan"), float("nan")
    by_month = joined.group_by("month").agg(d=(pl.col("b") - pl.col("a")).sum(), n=pl.len())
    d = by_month["d"].to_numpy().astype(np.float64)
    n = by_month["n"].to_numpy().astype(np.float64)
    rng = np.random.default_rng(191)
    draws = rng.integers(0, len(d), size=(N_RESAMPLES, len(d)))
    stats = d[draws].sum(axis=1) / n[draws].sum(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(stats, [tail, 1.0 - tail])
    return float(d.sum() / n.sum()), float(low), float(high)


def subset(*, frame: pl.DataFrame, name: str) -> pl.DataFrame:
    """Return the rows of a named cut."""
    if name == "all":
        return frame
    if name == "high_spread":
        return frame.filter(pl.col("spread_positive"))
    if name == "low_spread":
        return frame.filter(~pl.col("spread_positive"))
    low, high, _ = next(b for b in BAND_EDGES if b[2] == name)
    return frame.filter((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))


def main() -> None:
    """Print the absolute scores and contrasts for each site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="gated")
    parser.add_argument("--sites", nargs="*", default=list(common.ALL_SITES))
    args = parser.parse_args()
    report: dict = {}
    cuts = ["all", "high_spread", "low_spread", *[b[2] for b in BAND_EDGES]]
    for site in args.sites:
        frame = losses(frame=load(variant=args.variant, site=site))
        arms = {arm: frame.filter(pl.col("arm") == arm) for arm in ARMS}
        report[site] = {}
        print(
            f"\n=== {site} ({args.variant}) rows per arm: {arms['a0'].height:,}, "
            f"gate-on share {arms['a0']['gate'].mean():.2f}, "
            f"spread-positive share {arms['a0']['spread_positive'].mean():.2f}"
        )
        for cut in cuts:
            print(f"-- cut {cut}")
            row = {}
            for arm in ARMS:
                part = subset(frame=arms[arm], name=cut)
                row[arm] = {
                    "n": part.height,
                    "qs": float(part["qs"].to_numpy().mean()),
                    "ae": float(part["ae"].to_numpy().mean()),
                    "cov90": float(part["in90"].to_numpy().mean()),
                    **{
                        k: float(part[k].to_numpy().mean())
                        for k in part.columns
                        if k.startswith(("pin_", "below_"))
                    },
                }
                print(
                    f"{arm:3s} n={part.height:7d} qs={row[arm]['qs']:.4f} mae={row[arm]['ae']:.4f} "
                    f"cov90={row[arm]['cov90']:.3f}"
                )
            for arm in ARMS[1:]:
                a, b = subset(frame=arms["a0"], name=cut), subset(frame=arms[arm], name=cut)
                for column in ("qs", "ae"):
                    mean, low, high = interval(a=a, b=b, column=column, level=0.95)
                    relative = mean / row["a0"][column] * 100.0
                    row[f"{arm}_minus_a0_{column}"] = {
                        "mean": mean,
                        "low": low,
                        "high": high,
                        "pct": relative,
                    }
                    print(
                        f"   {arm}-a0 {column}: {mean:+.4f} [{low:+.4f}, {high:+.4f}] = {relative:+.2f}%"
                    )
            report[site][cut] = row
    out = common.WORK_DIR / f"score_{args.variant}.json"
    out.write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
