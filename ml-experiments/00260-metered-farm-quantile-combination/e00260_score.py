# ruff: noqa: E501
"""Score experiment 260: per-row losses (seed-averaged), farm-equal means, month-block intervals.

Rows are experiment 224's scored-year, 12 to 36 h rows that every arm holds, including its floors.
The quantile score is twice the mean pinball loss over the 11 levels, in points of capacity.

    uv run --no-sync python ml-experiments/00260-*/e00260_score.py
"""

from typing import Final

import e00260_common as common
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

DRAWS: Final[int] = 20_000
QCOLS: Final[list[str]] = [level_column(level=lv) for lv in common.LEVELS]
CLOUDY_CSI: Final[float] = 0.5
FLOORS: Final[tuple[str, ...]] = ("floor_heuristic", "floor_climatology")
E243_ARMS: Final[tuple[str, ...]] = ("h5_quantile", "nn_pure", "xgboost", "plant")
"""Experiment 243's stored arms that this experiment compares against."""
CONTRASTS: Final[dict[str, tuple[str, str, str]]] = {
    "C1": ("c2", "h5_quantile", "own_history"),
    "C2": ("c2", "nn_pure", "own_history"),
    "C3": ("c3", "c2", "own_history"),
    "C4": ("c2", "h5_quantile", "zero_history"),
    "X1": ("c1", "h5_quantile", "own_history"),
    "X2": ("c2_lomo", "h5_quantile", "own_history"),
    "X3": ("c2", "xgboost", "own_history"),
}
"""Planned contrasts C1, C3, C4 (arm, reference, setting); C2 is read on `pin99`, and X1 to X3 are exploratory."""
NEAR_ZERO: Final[float] = 0.03
"""Actual output under this share of capacity counts as near zero."""


def reference_rows() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return experiment 224's common-row keys and its two floors' quantiles at those rows."""
    frame = pl.read_parquet(common.E224_DIR / "predictions" / "scored_year.parquet").filter(
        pl.col("band") == 0
    )
    ok = frame.filter(pl.all_horizontal([pl.col(c).is_finite() for c in QCOLS]))
    keys = (
        ok.group_by("time_series_id", "valid_time")
        .agg(n=pl.col("arm").n_unique())
        .filter(pl.col("n") == frame["arm"].n_unique())
        .select("time_series_id", "valid_time")
    )
    floors = (
        frame.filter(pl.col("arm").is_in(FLOORS))
        .join(keys, on=["time_series_id", "valid_time"])
        .rename({"valid_time": "time"})
    )
    return keys.rename({"valid_time": "time"}), floors


def row_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return per-row quantile score, median absolute error, and pinball at 0.01 and 0.99 (points of capacity)."""
    q = frame.select(QCOLS).to_numpy()
    actual = frame["actual"].to_numpy()
    error = actual[:, None] - q
    pin = np.maximum(common.LEVELS * error, (common.LEVELS - 1.0) * error)
    return frame.select("time_series_id", "time", "arm", "setting").with_columns(
        score=pl.Series(200.0 * pin.mean(axis=1)),
        ae_median=pl.Series(100.0 * np.abs(error[:, 5])),
        pin01=pl.Series(100.0 * pin[:, 0]),
        pin99=pl.Series(100.0 * pin[:, 10]),
        cover01=pl.Series((actual <= q[:, 0]).astype(float)),
        cover99=pl.Series((actual <= q[:, 10]).astype(float)),
        miss01_near_zero=pl.Series(((actual <= q[:, 0]) & (actual < NEAR_ZERO)).astype(float)),
        miss01=pl.Series((actual <= q[:, 0]).astype(float)),
    )


def load_arms() -> pl.DataFrame:
    """Return the seed-averaged per-row losses of every arm and setting on the common rows."""
    keys, floors = reference_rows()
    pieces = []
    for setting in common.SETTINGS:
        files = sorted((common.PRED_DIR / setting).glob("*.parquet"))
        files += [
            f
            for a in E243_ARMS
            for f in sorted((common.E243_DIR / "predictions" / setting).glob(f"{a}_s*_*.parquet"))
        ]
        frame = pl.concat(
            [
                pl.read_parquet(f)
                .select(
                    "time",
                    "init_time",
                    "lead_hours",
                    "time_series_id",
                    "arm",
                    "seed",
                    "actual",
                    *QCOLS,
                )
                .with_columns(
                    pl.col(QCOLS).cast(pl.Float64),
                    pl.col("actual").cast(pl.Float64),
                    pl.col("lead_hours").cast(pl.Float64),
                )
                for f in files
            ]
        )
        frame = frame.join(keys, on=["time_series_id", "time"], how="inner")
        losses = row_losses(frame=frame.with_columns(setting=pl.lit(setting)))
        # Average over seeds per (farm, time, arm).
        by_seed = frame.select("time_series_id", "time", "arm", "seed")
        losses = losses.with_columns(seed=by_seed["seed"])
        pieces.append(losses)
        f = floors.with_columns(setting=pl.lit(setting))
        pieces.append(row_losses(frame=f).with_columns(seed=pl.lit(0)))
    allrows = pl.concat(pieces)
    averaged = allrows.group_by("setting", "arm", "time_series_id", "time").agg(
        pl.col("score").mean(),
        pl.col("ae_median").mean(),
        pl.col("pin01").mean(),
        pl.col("pin99").mean(),
        pl.col("cover01").mean(),
        pl.col("cover99").mean(),
        pl.col("miss01").mean(),
        pl.col("miss01_near_zero").mean(),
    )
    csi = pl.concat(
        [
            common.load_farm(s).frame.select("time", "csi", time_series_id=pl.lit(s))
            for s in common.FARMS
        ]
    )
    return averaged.join(csi, on=["time_series_id", "time"], how="left").with_columns(
        month=pl.col("time").dt.strftime("%Y-%m")
    )


def farm_means(
    *, rows: pl.DataFrame, value: str
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return per-(arm, farm, month) sums and per-(farm, month) counts for a bootstrap."""
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
    return sums, counts, arms, months


def boot(*, sums: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the point estimate and bootstrap draws (arms, draws) of the farm-averaged mean."""
    rng = np.random.default_rng(260)
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


def main() -> None:
    """Print and write the score tables."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    rows = load_arms()
    rows.write_parquet(common.DATA_DIR / "row_losses.parquet")
    out_scores, out_contrasts, out_farms = [], [], []
    for setting in common.SETTINGS:
        sub = rows.filter(pl.col("setting") == setting)
        for slice_name, slice_rows in (
            ("all", sub),
            ("cloudy", sub.filter(pl.col("csi") < CLOUDY_CSI)),
            ("clear", sub.filter(pl.col("csi") >= CLOUDY_CSI)),
        ):
            for metric in ("score", "ae_median", "pin01", "pin99"):
                sums, counts, arms, _ = farm_means(rows=slice_rows, value=metric)
                point, draws = boot(sums=sums, counts=counts)
                idx = {a: i for i, a in enumerate(arms)}
                for a in arms:
                    lo, hi = np.nanpercentile(draws[idx[a]], [2.5, 97.5])
                    out_scores.append(
                        {
                            "setting": setting,
                            "slice": slice_name,
                            "metric": metric,
                            "arm": a,
                            "value": point[idx[a]],
                            "lo": lo,
                            "hi": hi,
                        }
                    )
                for name, (arm, ref, contrast_setting) in CONTRASTS.items():
                    if contrast_setting != setting:
                        continue
                    for metric_name in (metric,):
                        change = 100.0 * (draws[idx[arm]] - draws[idx[ref]]) / draws[idx[ref]]
                        lo, hi = np.nanpercentile(change, [2.5, 97.5])
                        lo99, hi99 = np.nanpercentile(change, [0.5, 99.5])
                        out_contrasts.append(
                            {
                                "contrast": name,
                                "setting": setting,
                                "slice": slice_name,
                                "metric": metric_name,
                                "arm": arm,
                                "reference": ref,
                                "percent_change": 100.0
                                * (point[idx[arm]] - point[idx[ref]])
                                / point[idx[ref]],
                                "lo": lo,
                                "hi": hi,
                                "lo99": lo99,
                                "hi99": hi99,
                            }
                        )
        farm_table = (
            sub.group_by("arm", "time_series_id")
            .agg(
                pl.col("score").mean(),
                pl.col("ae_median").mean(),
                pl.col("pin01").mean(),
                pl.col("pin99").mean(),
                pl.col("cover01").mean(),
                pl.col("cover99").mean(),
                pl.col("miss01").mean(),
                pl.col("miss01_near_zero").mean(),
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
    print(
        scores.filter((pl.col("slice") == "all") & (pl.col("metric") == "score")).sort(
            "setting", "value"
        )
    )
    print(contrasts.filter(pl.col("slice") == "all").sort("contrast", "metric"))


if __name__ == "__main__":
    main()
