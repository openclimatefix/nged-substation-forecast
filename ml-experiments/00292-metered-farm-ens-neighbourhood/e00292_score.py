# ruff: noqa: E501
"""Score experiment 292: per-row losses (seed-averaged), farm-equal means, month-block intervals.

Rows are experiment 243's scored rows (experiment 224's scored-year, 12 to 36 h rows that every arm
holds). The quantile score is twice the mean pinball loss over the 11 levels, in points of capacity.
The planned contrasts use 98.75% intervals (Bonferroni over four); other intervals are 95%.

    uv run --no-sync python ml-experiments/00292-*/e00292_score.py
"""

from typing import Final

import e00292_common as common
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

DRAWS: Final[int] = 20_000
QCOLS: Final[list[str]] = [level_column(level=lv) for lv in common.LEVELS]
CLOUDY_CSI: Final[float] = 0.5
SUNNY_CSI: Final[float] = 0.8
FLOORS: Final[tuple[str, ...]] = ("floor_heuristic", "floor_climatology")
PLANNED: Final[tuple[tuple[str, str, str], ...]] = (
    ("C1", "n1", "n0"),
    ("C2", "n1", "placebo"),
    ("C3", "n1", "x1"),
    ("C4", "n2", "n0"),
)
EXTRA: Final[tuple[tuple[str, str], ...]] = (
    ("n1", "x0"),
    ("n0", "x0"),
    ("x1", "x0"),
    ("n2", "x1"),
    ("n0", "h5_243"),
    ("placebo", "n0"),
)
LEVEL_INDEX: Final[dict[str, int]] = {"01": 0, "05": 1, "95": 9, "99": 10}
QUANTILE_WEIGHTS: Final[np.ndarray] = np.gradient(common.LEVELS) / np.gradient(common.LEVELS).sum()


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
    """Return per-row losses in points of capacity, coverage, and squared errors of the median."""
    q = frame.select(QCOLS).to_numpy()
    actual = frame["actual"].to_numpy()
    error = actual[:, None] - q
    pin = np.maximum(common.LEVELS * error, (common.LEVELS - 1.0) * error)
    mean_forecast = q @ QUANTILE_WEIGHTS
    columns = {
        "score": 200.0 * pin.mean(axis=1),
        "ae_median": 100.0 * np.abs(error[:, 5]),
        "ae_mean": 100.0 * np.abs(actual - mean_forecast),
        "median_minus_mean": 100.0 * (q[:, 5] - mean_forecast),
        "sse_median": error[:, 5] ** 2,
        "sq_actual": actual**2,
    }
    for name, index in LEVEL_INDEX.items():
        columns[f"pin{name}"] = 100.0 * pin[:, index]
        columns[f"cover{name}"] = (actual <= q[:, index]).astype(float)
    return frame.select("time_series_id", "time", "arm").with_columns(
        **{k: pl.Series(v) for k, v in columns.items()}
    )


def load_arms() -> pl.DataFrame:
    """Return the seed-averaged per-row losses of every arm on the common rows."""
    keys, floors = reference_rows()
    mine = pl.concat(
        [
            pl.read_parquet(f).with_columns(
                pl.col(QCOLS).cast(pl.Float64), pl.col("actual").cast(pl.Float64)
            )
            for f in sorted((common.PRED_DIR / "own_history").glob("*.parquet"))
        ]
    ).select("time_series_id", "time", "arm", "seed", "actual", *QCOLS)
    old = (
        pl.concat(
            [
                pl.read_parquet(f).with_columns(
                    pl.col(QCOLS).cast(pl.Float64), pl.col("actual").cast(pl.Float64)
                )
                for f in sorted(
                    (common.E243_DIR / "predictions" / "own_history").glob("h5_quantile_s*.parquet")
                )
            ]
        )
        .select("time_series_id", "time", "arm", "seed", "actual", *QCOLS)
        .with_columns(arm=pl.lit("h5_243"))
    )
    frame = pl.concat([mine, old]).join(keys, on=["time_series_id", "time"], how="inner")
    losses = row_losses(frame=frame).with_columns(seed=frame["seed"])
    floor_losses = row_losses(frame=floors).with_columns(seed=pl.lit(0))
    allrows = pl.concat([losses, floor_losses])
    value_columns = [
        c for c in allrows.columns if c not in ("time_series_id", "time", "arm", "seed")
    ]
    averaged = allrows.group_by("arm", "time_series_id", "time").agg(
        *[pl.col(c).mean() for c in value_columns]
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


def boot(
    *, sums: np.ndarray, counts: np.ndarray, farms: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the point estimate and bootstrap draws (arms, draws) of the farm-averaged mean."""
    keep = farms if farms is not None else list(range(counts.shape[0]))
    sums, counts = sums[:, keep, :], counts[keep]
    rng = np.random.default_rng(292)
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


def interval(*, draws: np.ndarray, level: float) -> tuple[float, float]:
    """Return the central interval of the draws at the level (a percentage)."""
    tail = (100.0 - level) / 2.0
    lo, hi = np.nanpercentile(draws, [tail, 100.0 - tail])
    return float(lo), float(hi)


def main() -> None:
    """Print and write the score tables."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    rows = load_arms()
    rows.write_parquet(common.DATA_DIR / "row_losses.parquet")
    print("rows per arm:", rows.group_by("arm").len().sort("arm").to_dicts())
    out_scores, out_contrasts = [], []
    slices = (
        ("all", rows),
        ("cloudy", rows.filter(pl.col("csi") < CLOUDY_CSI)),
        ("clear", rows.filter(pl.col("csi") >= CLOUDY_CSI)),
    )
    metrics = (
        "score",
        "ae_median",
        "ae_mean",
        "median_minus_mean",
        "pin01",
        "pin05",
        "pin95",
        "pin99",
        "cover01",
        "cover05",
        "cover95",
        "cover99",
    )
    for slice_name, slice_rows in slices:
        for metric in metrics:
            sums, counts, arms, _ = farm_means(rows=slice_rows, value=metric)
            point, draws = boot(sums=sums, counts=counts)
            idx = {a: i for i, a in enumerate(arms)}
            for a in arms:
                lo, hi = interval(draws=draws[idx[a]], level=95.0)
                out_scores.append(
                    {
                        "slice": slice_name,
                        "metric": metric,
                        "arm": a,
                        "value": point[idx[a]],
                        "lo": lo,
                        "hi": hi,
                    }
                )
            if metric in ("cover01", "cover05", "cover95", "cover99", "median_minus_mean"):
                continue
            planned = [(name, a, r, 98.75) for name, a, r in PLANNED]
            extra = [("extra", a, r, 95.0) for a, r in EXTRA]
            for name, arm, ref, level in (*planned, *extra):
                if arm not in idx or ref not in idx:
                    continue
                change = 100.0 * (draws[idx[arm]] - draws[idx[ref]]) / draws[idx[ref]]
                lo, hi = interval(draws=change, level=level)
                out_contrasts.append(
                    {
                        "slice": slice_name,
                        "metric": metric,
                        "contrast": name,
                        "arm": arm,
                        "reference": ref,
                        "level": level,
                        "percent_change": 100.0
                        * (point[idx[arm]] - point[idx[ref]])
                        / point[idx[ref]],
                        "lo": lo,
                        "hi": hi,
                    }
                )
    scores, contrasts = pl.DataFrame(out_scores), pl.DataFrame(out_contrasts)
    scores.write_csv(common.DATA_DIR / "scores.csv")
    contrasts.write_csv(common.DATA_DIR / "contrasts.csv")
    # Per-farm table and the stop rule: the gain of n1 over n0 at each farm.
    per_farm = rows.group_by("arm", "time_series_id").agg(
        *[pl.col(m).mean() for m in metrics], n=pl.len()
    )
    per_farm.write_csv(common.DATA_DIR / "per_farm.csv")
    wide = per_farm.filter(pl.col("arm").is_in(["n0", "n1", "n2", "placebo", "x0", "x1"])).pivot(
        on="arm", index="time_series_id", values="score"
    )
    gain = wide.with_columns(
        n1_vs_n0=100.0 * (pl.col("n1") - pl.col("n0")) / pl.col("n0"),
        placebo_vs_n0=100.0 * (pl.col("placebo") - pl.col("n0")) / pl.col("n0"),
        **(
            {"n2_vs_n0": 100.0 * (pl.col("n2") - pl.col("n0")) / pl.col("n0")}
            if "n2" in wide.columns
            else {}
        ),
    ).sort("time_series_id")
    gain.write_csv(common.DATA_DIR / "per_farm_gain.csv")
    print(gain)
    # Leave-one-farm-out for the planned contrasts on the quantile score.
    sums, counts, arms, _ = farm_means(rows=rows, value="score")
    idx = {a: i for i, a in enumerate(arms)}
    lofo = []
    for name, arm, ref in PLANNED:
        if arm not in idx or ref not in idx:
            continue
        for left_out in range(len(common.FARMS)):
            keep = [i for i in range(len(common.FARMS)) if i != left_out]
            point, draws = boot(sums=sums, counts=counts, farms=keep)
            change = 100.0 * (draws[idx[arm]] - draws[idx[ref]]) / draws[idx[ref]]
            lo, hi = interval(draws=change, level=98.75)
            lofo.append(
                {
                    "contrast": name,
                    "left_out": common.FARMS[left_out],
                    "percent_change": 100.0 * (point[idx[arm]] - point[idx[ref]]) / point[idx[ref]],
                    "lo": lo,
                    "hi": hi,
                }
            )
    pl.DataFrame(lofo).write_csv(common.DATA_DIR / "leave_one_farm_out.csv")
    # Fraction of output fitted by the median forecast (1 - SSE / sum of squared output), median over farms.
    fraction = rows.group_by("arm", "time_series_id").agg(
        overall=1.0 - pl.col("sse_median").sum() / pl.col("sq_actual").sum(),
    )
    sunny = (
        rows.filter(pl.col("csi") >= SUNNY_CSI)
        .group_by("arm", "time_series_id")
        .agg(sunny=1.0 - pl.col("sse_median").sum() / pl.col("sq_actual").sum())
    )
    fraction = (
        fraction.join(sunny, on=["arm", "time_series_id"])
        .group_by("arm")
        .agg(pl.col("overall").median(), pl.col("sunny").median())
    )
    fraction.write_csv(common.DATA_DIR / "fraction_fitted.csv")
    print(fraction.sort("arm"))
    print(scores.filter((pl.col("slice") == "all") & (pl.col("metric") == "score")).sort("value"))
    print(contrasts.filter((pl.col("slice") == "all") & (pl.col("metric") == "score")))


if __name__ == "__main__":
    main()
