# ruff: noqa: E501
"""Score experiment 275: seed-averaged per-row losses, farm-equal means, month-block intervals.

Rows are experiment 243's scored rows (experiment 224's band-0 scored year, daylight half-hours at
12 to 36 h leads that every arm holds). The quantile score is twice the mean pinball loss over the
11 levels, in points of capacity. The mean forecast is the integral of the quantile function
(trapezoid rule between the 0.01 and 0.99 levels, with the end values held flat to 0 and 1).

    uv run --no-sync python ml-experiments/00275-*/e00275_score.py
"""

from pathlib import Path
from typing import Final

import e00275_common as common
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

DRAWS: Final[int] = 20_000
QCOLS: Final[list[str]] = [level_column(level=lv) for lv in common.LEVELS]
E243_DIR: Final[Path] = common.E224_DIR.parent / "00243-metered-farm-architecture-comparison"
REFERENCE_ARMS: Final[dict[str, str]] = {"h5_quantile": "e243_h5", "xgboost": "e243_xgboost"}
TAIL_INDEX: Final[dict[str, int]] = {"01": 0, "05": 1, "95": 9, "99": 10}
CONFIDENCE: Final[float] = 98.3
"""Bonferroni over the three planned contrasts."""


def reference_keys() -> pl.DataFrame:
    """Return experiment 224's common-row keys (farm and valid time) that every arm holds."""
    frame = pl.read_parquet(common.E224_DIR / "predictions" / "scored_year.parquet").filter(
        pl.col("band") == 0
    )
    ok = frame.filter(pl.all_horizontal([pl.col(c).is_finite() for c in QCOLS]))
    return (
        ok.group_by("time_series_id", "valid_time")
        .agg(n=pl.col("arm").n_unique())
        .filter(pl.col("n") == frame["arm"].n_unique())
        .select("time_series_id", time="valid_time")
    )


def row_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return per-row losses (points of capacity), tail pinball and coverage, and the mean forecast."""
    q = frame.select(QCOLS).to_numpy()
    actual = frame["actual"].to_numpy()
    error = actual[:, None] - q
    pin = np.maximum(common.LEVELS * error, (common.LEVELS - 1.0) * error)
    edges = np.concatenate([[0.0], common.LEVELS, [1.0]])
    padded = np.concatenate([q[:, :1], q, q[:, -1:]], axis=1)
    mean_forecast = np.sum(0.5 * (padded[:, 1:] + padded[:, :-1]) * np.diff(edges), axis=1)
    columns = {
        "score": 200.0 * pin.mean(axis=1),
        "ae_median": 100.0 * np.abs(error[:, 5]),
        "ae_mean": 100.0 * np.abs(actual - mean_forecast),
        "median_minus_mean": 100.0 * (q[:, 5] - mean_forecast),
    }
    for name, i in TAIL_INDEX.items():
        columns[f"pin{name}"] = 100.0 * pin[:, i]
        columns[f"cover{name}"] = (actual <= q[:, i]).astype(float)
    crossed = (
        frame["crossed"].cast(pl.Float64).to_numpy()
        if "crossed" in frame.columns
        else np.zeros(len(q))
    )
    columns["crossed"] = crossed
    return frame.select("time_series_id", "time", "arm", "seed").with_columns(
        **{k: pl.Series(v) for k, v in columns.items()}
    )


def load_arms(*, setting: str) -> pl.DataFrame:
    """Return the seed-averaged per-row losses of every arm on the common rows."""
    keys = reference_keys()
    frames = [
        pl.read_parquet(f).with_columns(pl.col([*QCOLS, "actual"]).cast(pl.Float64))
        for f in sorted((common.PRED_DIR / setting).glob("*.parquet"))
    ]
    own = pl.concat(frames, how="diagonal").join(keys, on=["time_series_id", "time"])
    pieces = [row_losses(frame=own)]
    if setting == "own_history":
        for stored, label in REFERENCE_ARMS.items():
            files = sorted((E243_DIR / "predictions" / setting).glob(f"{stored}_s*_*.parquet"))
            ref = pl.concat(
                [
                    pl.read_parquet(f).with_columns(pl.col([*QCOLS, "actual"]).cast(pl.Float64))
                    for f in files
                ]
            ).join(keys, on=["time_series_id", "time"])
            pieces.append(row_losses(frame=ref.with_columns(arm=pl.lit(label))))
    losses = pl.concat(pieces)
    value_columns = [
        c for c in losses.columns if c not in ("time_series_id", "time", "arm", "seed")
    ]
    averaged = losses.group_by("arm", "time_series_id", "time").agg(
        [pl.col(c).mean() for c in value_columns]
    )
    return averaged.with_columns(month=pl.col("time").dt.strftime("%Y-%m"))


def farm_sums(*, rows: pl.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return per-(arm, farm, month) sums and per-(farm, month) counts for the bootstrap."""
    arms = sorted(rows["arm"].unique().to_list())
    months = sorted(rows["month"].unique().to_list())
    sums = np.zeros((len(arms), len(common.FARMS), len(months)))
    counts = np.zeros((len(common.FARMS), len(months)))
    grouped = rows.group_by("arm", "time_series_id", "month").agg(
        total=pl.col(value).sum(), n=pl.len()
    )
    for r in grouped.iter_rows(named=True):
        a, f, m = (
            arms.index(r["arm"]),
            common.FARMS.index(r["time_series_id"]),
            months.index(r["month"]),
        )
        sums[a, f, m], counts[f, m] = r["total"], r["n"]
    return sums, counts, arms


def boot(*, sums: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the point estimate and bootstrap draws (arms, draws) of the farm-averaged mean."""
    rng = np.random.default_rng(275)
    n_months = counts.shape[1]
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


CONTRASTS: Final[list[tuple[str, str, str]]] = [
    ("C1", "l2", "l0"),
    ("C3", "l2", "control"),
    ("l1_vs_l0", "l1", "l0"),
    ("l2_vs_l1", "l2", "l1"),
    ("l3_vs_l0", "l3", "l0"),
    ("l3_vs_l2", "l3", "l2"),
    ("x2_vs_x0", "x2", "x0"),
    ("l0_vs_e243_h5", "l0", "e243_h5"),
    ("l0_vs_x0", "l0", "x0"),
    ("l2_vs_x2", "l2", "x2"),
    ("posthoc_ens_offsets_vs_l0", "l2_ens_offsets", "l0"),
    ("posthoc_pre_ens_vs_l0", "l2_pre_ens", "l0"),
]


def main() -> None:
    """Print and write the score tables."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    out_scores, out_contrasts, out_farms = [], [], []
    for setting in common.SETTINGS:
        if not any((common.PRED_DIR / setting).glob("*.parquet")):
            continue
        rows = load_arms(setting=setting)
        rows.write_parquet(common.DATA_DIR / f"row_losses_{setting}.parquet")
        metrics = ["score", "ae_median", "ae_mean", "median_minus_mean", "crossed"] + [
            f"{kind}{name}" for name in TAIL_INDEX for kind in ("pin", "cover")
        ]
        for metric in metrics:
            sums, counts, arms = farm_sums(rows=rows, value=metric)
            point, draws = boot(sums=sums, counts=counts)
            idx = {a: i for i, a in enumerate(arms)}
            for a in arms:
                lo, hi = interval(values=draws[idx[a]], level=95.0)
                out_scores.append(
                    {
                        "setting": setting,
                        "metric": metric,
                        "arm": a,
                        "value": point[idx[a]],
                        "lo": lo,
                        "hi": hi,
                    }
                )
            if metric not in ("score", "ae_median", "ae_mean", "pin01", "pin99"):
                continue
            for name, arm, ref in CONTRASTS:
                if arm not in idx or ref not in idx:
                    continue
                change = 100.0 * (draws[idx[arm]] - draws[idx[ref]]) / draws[idx[ref]]
                row = {
                    "setting": setting,
                    "metric": metric,
                    "contrast": name,
                    "arm": arm,
                    "reference": ref,
                    "percent_change": 100.0 * (point[idx[arm]] - point[idx[ref]]) / point[idx[ref]],
                }
                for level in (95.0, CONFIDENCE):
                    row[f"lo{level}"], row[f"hi{level}"] = interval(values=change, level=level)
                out_contrasts.append(row)
            if metric == "score" and {"l2", "l0", "x2", "x0"} <= idx.keys():
                gain_l = 100.0 * (draws[idx["l2"]] - draws[idx["l0"]]) / draws[idx["l0"]]
                gain_x = 100.0 * (draws[idx["x2"]] - draws[idx["x0"]]) / draws[idx["x0"]]
                diff = gain_l - gain_x
                row = {
                    "setting": setting,
                    "metric": metric,
                    "contrast": "C2",
                    "arm": "l2_gain_minus_x2_gain",
                    "reference": "points of percent",
                    "percent_change": (
                        100.0 * (point[idx["l2"]] - point[idx["l0"]]) / point[idx["l0"]]
                        - 100.0 * (point[idx["x2"]] - point[idx["x0"]]) / point[idx["x0"]]
                    ),
                }
                for level in (95.0, CONFIDENCE):
                    row[f"lo{level}"], row[f"hi{level}"] = interval(values=diff, level=level)
                out_contrasts.append(row)
        farm_table = (
            rows.group_by("arm", "time_series_id")
            .agg(
                pl.col("score").mean(),
                pl.col("ae_median").mean(),
                pl.col("ae_mean").mean(),
                pl.col("median_minus_mean").mean(),
                pl.col("crossed").mean(),
                pl.col("pin01").mean(),
                pl.col("pin99").mean(),
                n=pl.len(),
            )
            .with_columns(setting=pl.lit(setting))
        )
        out_farms.append(farm_table)
    scores = pl.DataFrame(out_scores)
    contrasts = pl.DataFrame(out_contrasts)
    farms = pl.concat(out_farms)
    scores.write_csv(common.DATA_DIR / "scores.csv")
    contrasts.write_csv(common.DATA_DIR / "contrasts.csv")
    farms.write_csv(common.DATA_DIR / "per_farm.csv")
    print(scores.filter(pl.col("metric") == "score").sort("setting", "value"))
    print(contrasts.filter(pl.col("metric") == "score").sort("setting", "percent_change"))
    gains = (
        farms.filter(pl.col("setting") == "own_history")
        .pivot(on="arm", index="time_series_id", values="score")
        .with_columns(l2_gain_percent=100.0 * (pl.col("l2") - pl.col("l0")) / pl.col("l0"))
        .select("time_series_id", "l0", "l2", "l2_gain_percent")
        if "l2" in farms["arm"].to_list()
        else None
    )
    print(gains)


if __name__ == "__main__":
    main()
