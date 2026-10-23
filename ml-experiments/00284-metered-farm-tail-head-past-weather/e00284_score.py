# ruff: noqa: E501
"""Score experiment 284: tail pinball loss and coverage, farm-equal means, month-block intervals.

Rows are experiment 275's scored rows (experiment 224's band-0 scored year, daylight half-hours at 12
to 36 h leads that every arm holds). The reference arms l2 (H5) and x2 (XGBoost) are read from
experiment 275's stored predictions. `t3_full` is t3's 0.01, 0.05, 0.95, and 0.99 quantiles with
l2's other seven quantiles of the same seed (sorted per row), so it has the 11 levels the quantile
score needs. The mean forecast is the integral of the quantile function (trapezoid rule, end values
held flat to 0 and 1).

    uv run --no-sync python ml-experiments/00284-*/e00284_score.py
"""

from pathlib import Path
from typing import Final

import e00284_common as common
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

DRAWS: Final[int] = 20_000
E275_DIR: Final[Path] = common.E224_DIR.parent / "00275-metered-farm-long-history-neural-head"
LEVELS11: Final[np.ndarray] = np.array(common.MIXTURE_LEVELS)
Q11: Final[list[str]] = [level_column(level=lv) for lv in LEVELS11]
TAIL_ALL: Final[dict[str, float]] = {
    "01": 0.01,
    "02": 0.02,
    "05": 0.05,
    "95": 0.95,
    "98": 0.98,
    "99": 0.99,
}
SWAP: Final[list[str]] = ["q01", "q05", "q95", "q99"]
CONFIDENCE: Final[float] = 98.75
"""Bonferroni over the four planned contrasts."""
METRICS: Final[list[str]] = [
    "score",
    "ae_mean",
    "ae_median",
    "median_minus_mean",
    *[f"pin{k}" for k in TAIL_ALL],
    *[f"cover{k}" for k in TAIL_ALL],
]


def reference_keys() -> pl.DataFrame:
    """Return experiment 224's common-row keys (farm and valid time) that every arm holds."""
    frame = pl.read_parquet(common.E224_DIR / "predictions" / "scored_year.parquet").filter(
        pl.col("band") == 0
    )
    ok = frame.filter(pl.all_horizontal([pl.col(c).is_finite() for c in Q11]))
    return (
        ok.group_by("time_series_id", "valid_time")
        .agg(n=pl.col("arm").n_unique())
        .filter(pl.col("n") == frame["arm"].n_unique())
        .select("time_series_id", time="valid_time")
    )


def losses(*, frame: pl.DataFrame, levels: dict[str, float], full: bool) -> pl.DataFrame:
    """Return per-row losses (points of capacity) for the given tail levels, and the 11-level ones when `full`."""
    actual = frame["actual"].to_numpy()
    columns: dict[str, np.ndarray] = {}
    for key, level in levels.items():
        q = frame[level_column(level=level)].to_numpy()
        error = actual - q
        columns[f"pin{key}"] = 100.0 * np.maximum(level * error, (level - 1.0) * error)
        columns[f"cover{key}"] = (actual <= q).astype(float)
    if full:
        q = np.sort(frame.select(Q11).to_numpy(), axis=1)
        error = actual[:, None] - q
        pin = np.maximum(LEVELS11 * error, (LEVELS11 - 1.0) * error)
        edges = np.concatenate([[0.0], LEVELS11, [1.0]])
        padded = np.concatenate([q[:, :1], q, q[:, -1:]], axis=1)
        mean_forecast = np.sum(0.5 * (padded[:, 1:] + padded[:, :-1]) * np.diff(edges), axis=1)
        columns["score"] = 200.0 * pin.mean(axis=1)
        columns["ae_mean"] = 100.0 * np.abs(actual - mean_forecast)
        columns["ae_median"] = 100.0 * np.abs(error[:, 5])
        columns["median_minus_mean"] = 100.0 * (q[:, 5] - mean_forecast)
    return frame.select("time_series_id", "time", "arm", "seed").with_columns(
        **{k: pl.Series(v) for k, v in columns.items()}
    )


def read_dir(*, directory: Path, pattern: str) -> pl.DataFrame:
    """Read every file in a folder matching the pattern, with float quantile columns."""
    files = sorted(directory.glob(pattern))
    return pl.concat(
        [pl.read_parquet(f).with_columns(pl.col(pl.Float32).cast(pl.Float64)) for f in files],
        how="diagonal",
    )


def per_row_losses() -> pl.DataFrame:
    """Return seed-averaged per-row losses of every arm on the common rows."""
    keys = reference_keys()
    own = read_dir(directory=common.PRED_DIR, pattern="*.parquet").join(
        keys, on=["time_series_id", "time"]
    )
    refs = read_dir(
        directory=E275_DIR / "predictions" / "own_history", pattern="[lx]2_s*_*.parquet"
    )
    refs = refs.join(keys, on=["time_series_id", "time"])
    l2 = refs.filter(pl.col("arm") == "l2")
    x2 = refs.filter(pl.col("arm") == "x2")
    t3 = own.filter(pl.col("arm") == "t3")
    # t3_full: t3's four swapped quantiles inside l2's 11 quantiles (same seed), sorted per row.
    full = l2.drop(SWAP).join(
        t3.select("time_series_id", "time", "seed", *SWAP),
        on=["time_series_id", "time", "seed"],
    )
    full = full.with_columns(arm=pl.lit("t3_full"))
    pieces = [losses(frame=own, levels=TAIL_ALL, full=False)]
    eleven = {k: v for k, v in TAIL_ALL.items() if k in ("01", "05", "95", "99")}
    pieces.extend(losses(frame=frame, levels=eleven, full=True) for frame in (l2, x2, full))
    all_rows = pl.concat(pieces, how="diagonal")
    values = [c for c in all_rows.columns if c not in ("time_series_id", "time", "arm", "seed")]
    averaged = all_rows.group_by("arm", "time_series_id", "time").agg(
        [pl.col(c).mean() for c in values]
    )
    return averaged.with_columns(month=pl.col("time").dt.strftime("%Y-%m"))


def farm_sums(*, rows: pl.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return per-(arm, farm, month) sums and per-(farm, month) counts for the bootstrap."""
    arms = sorted(rows.filter(pl.col(value).is_not_null())["arm"].unique().to_list())
    months = sorted(rows["month"].unique().to_list())
    sums = np.zeros((len(arms), len(common.FARMS), len(months)))
    counts = np.zeros((len(common.FARMS), len(months)))
    grouped = (
        rows.filter(pl.col(value).is_not_null() & pl.col(value).is_not_nan())
        .group_by("arm", "time_series_id", "month")
        .agg(total=pl.col(value).sum(), n=pl.len())
    )
    for r in grouped.iter_rows(named=True):
        a, f, m = (
            arms.index(r["arm"]),
            common.FARMS.index(r["time_series_id"]),
            months.index(r["month"]),
        )
        sums[a, f, m], counts[f, m] = r["total"], r["n"]
    return sums, counts, arms


def boot(
    *, sums: np.ndarray, counts: np.ndarray, farms: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the point estimate and bootstrap draws (arms, draws) of the mean over the chosen farms."""
    rng = np.random.default_rng(284)
    n_months = counts.shape[1]
    sums, counts = sums[:, farms, :], counts[farms, :]
    farm_n = counts.sum(axis=1)
    point = np.nanmean(sums.sum(axis=2) / np.where(farm_n > 0, farm_n, np.nan)[None, :], axis=1)
    weights = rng.multinomial(n_months, np.ones(n_months) / n_months, size=DRAWS).astype(float)
    denominator = weights @ counts.T
    draws = np.empty((sums.shape[0], DRAWS))
    with np.errstate(invalid="ignore", divide="ignore"):
        for a in range(sums.shape[0]):
            draws[a] = np.nanmean(
                (weights @ sums[a].T) / np.where(denominator > 0, denominator, np.nan), axis=1
            )
    return point, draws


def interval(*, values: np.ndarray, level: float) -> tuple[float, float]:
    """Return the central interval of the draws at the given confidence level (percent)."""
    tail = (100.0 - level) / 2.0
    lo, hi = np.nanpercentile(values, [tail, 100.0 - tail])
    return float(lo), float(hi)


CONTRASTS: Final[list[tuple[str, str, str, str]]] = [
    ("C1", "pin99", "t3", "x2"),
    ("C2", "pin01", "t3", "x2"),
    ("C3", "pin99", "t3", "t5"),
    ("C4", "pin99", "t3", "t4b"),
    ("t3_vs_l2_99", "pin99", "t3", "l2"),
    ("t3_vs_l2_01", "pin01", "t3", "l2"),
    ("t3_vs_p2_99", "pin99", "t3", "p2"),
    ("t3_vs_p2_01", "pin01", "t3", "p2"),
    ("t3_vs_t0_99", "pin99", "t3", "t0"),
    ("t3_vs_t0_01", "pin01", "t3", "t0"),
    ("t3_vs_t4a_99", "pin99", "t3", "t4a"),
    ("t3_vs_t4a_01", "pin01", "t3", "t4a"),
    ("t3_vs_x2_95", "pin95", "t3", "x2"),
    ("t3_vs_x2_05", "pin05", "t3", "x2"),
    ("t3full_vs_l2_score", "score", "t3_full", "l2"),
    ("t3full_vs_x2_score", "score", "t3_full", "x2"),
    ("t3full_vs_l2_ae_mean", "ae_mean", "t3_full", "l2"),
    ("t3full_vs_x2_ae_mean", "ae_mean", "t3_full", "x2"),
]


def main() -> None:  # noqa: PLR0915
    """Print and write the score tables."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    rows = per_row_losses()
    counts = rows.group_by("arm", "time_series_id").len()
    assert counts.group_by("time_series_id").agg(pl.col("len").n_unique())["len"].max() == 1
    assert counts.group_by("arm").len()["len"].n_unique() == 1
    assert counts.group_by("arm").len()["len"][0] == len(common.FARMS)
    rows.write_parquet(common.DATA_DIR / "row_losses.parquet")
    all_farms = list(range(len(common.FARMS)))
    scores, contrasts, per_farm_contrasts, leave_out = [], [], [], []
    drawn: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for metric in METRICS:
        sums, counts, arms = farm_sums(rows=rows, value=metric)
        drawn[metric] = (sums, counts, arms)
        point, draws = boot(sums=sums, counts=counts, farms=all_farms)
        for i, arm in enumerate(arms):
            lo, hi = interval(values=draws[i], level=95.0)
            scores.append({"metric": metric, "arm": arm, "value": point[i], "lo": lo, "hi": hi})
    for name, metric, arm, ref in CONTRASTS:
        sums, counts, arms = drawn[metric]
        if arm not in arms or ref not in arms:
            continue
        a, r = arms.index(arm), arms.index(ref)
        point, draws = boot(sums=sums, counts=counts, farms=all_farms)
        change = 100.0 * (draws[a] - draws[r]) / draws[r]
        row = {
            "contrast": name,
            "metric": metric,
            "arm": arm,
            "reference": ref,
            "percent_change": 100.0 * (point[a] - point[r]) / point[r],
        }
        for level in (95.0, CONFIDENCE):
            row[f"lo{level}"], row[f"hi{level}"] = interval(values=change, level=level)
        contrasts.append(row)
        if name in ("C1", "C2"):
            for farm_index, site in enumerate(common.FARMS):
                p, d = boot(sums=sums, counts=counts, farms=[farm_index])
                c = 100.0 * (d[a] - d[r]) / d[r]
                lo, hi = interval(values=c, level=CONFIDENCE)
                per_farm_contrasts.append(
                    {
                        "contrast": name,
                        "farm": site,
                        "percent_change": 100.0 * (p[a] - p[r]) / p[r],
                        "lo": lo,
                        "hi": hi,
                    }
                )
                kept = [i for i in all_farms if i != farm_index]
                p, d = boot(sums=sums, counts=counts, farms=kept)
                c = 100.0 * (d[a] - d[r]) / d[r]
                lo, hi = interval(values=c, level=CONFIDENCE)
                leave_out.append(
                    {
                        "contrast": name,
                        "left_out": site,
                        "percent_change": 100.0 * (p[a] - p[r]) / p[r],
                        "lo": lo,
                        "hi": hi,
                    }
                )
    farm_table = rows.group_by("arm", "time_series_id").agg(
        *[pl.col(m).mean() for m in METRICS], n=pl.len()
    )
    exceed = rows.group_by("arm", "time_series_id").agg(
        below_q01=pl.col("cover01").sum(),
        above_q99=(1.0 - pl.col("cover99")).sum(),
    )
    pl.DataFrame(scores).write_csv(common.DATA_DIR / "scores.csv")
    pl.DataFrame(contrasts).write_csv(common.DATA_DIR / "contrasts.csv")
    pl.DataFrame(per_farm_contrasts).write_csv(common.DATA_DIR / "per_farm_contrasts.csv")
    pl.DataFrame(leave_out).write_csv(common.DATA_DIR / "leave_one_farm_out.csv")
    farm_table.write_csv(common.DATA_DIR / "per_farm.csv")
    exceed.write_csv(common.DATA_DIR / "exceedances.csv")
    scores_df = pl.DataFrame(scores)
    for metric in ("pin01", "pin05", "pin95", "pin99", "cover01", "cover99", "score", "ae_mean"):
        print(metric)
        print(scores_df.filter(pl.col("metric") == metric).sort("value"))
    print(pl.DataFrame(contrasts))
    print(pl.DataFrame(per_farm_contrasts))
    print(pl.DataFrame(leave_out))
    print(
        farm_table.select("arm", "time_series_id", "cover01", "cover99").sort(
            "arm", "time_series_id"
        )
    )
    print(exceed.sort("arm", "time_series_id"))


if __name__ == "__main__":
    main()
