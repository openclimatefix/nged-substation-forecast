"""Arm S: a convex surface fitted to the network's median correction, scored on held-out months.

The target is the median correction of experiment 201's arm 2 (its `q50` minus arm D's `q50`,
averaged over the three seeds) on the scored year's rows outside switching events, six
substations pooled. The surface is piecewise linear in the ensemble-mean temperature (knots every
2 degrees Celsius from -6 to 26) for each hour band and day type, with a second-difference penalty
along temperature and a small ridge. Each distinct (substation, valid time) counts once. Explained
variance is `1 - SSE / SST` with those weights, by leave-one-month-out folds on the month of the
run's origin. Variants add one input at a time.

    uv run --no-sync python ml-experiments/00216-*/e00216_surface.py
"""

import json
from typing import Final

import e00216_common as common
import e00216_score as score
import numpy as np
import polars as pl
from studies.dp_calendar import bank_holiday_names
from studies.sources import ML_EXPERIMENTS_DATA_DIR

FEATURES: Final = ML_EXPERIMENTS_DATA_DIR / "00201-dp-gam-residual-mlp" / "work" / "features"
KNOTS: Final[np.ndarray] = np.arange(-6.0, 26.1, 2.0)
SMOOTH_WEIGHTS: Final[tuple[float, ...]] = (0.1, 1.0, 10.0)
RIDGE: Final[float] = 1e-3
LONDON: Final[str] = "Europe/London"


def correction_rows() -> pl.DataFrame:
    """Return the network's seed-averaged median correction with the inputs of the surface."""
    stored = score.read_arm(arm="stored", period="scored").select(
        "site", "power_fcst_init_time", "valid_time", "lead_hours", "q50", "hour_band"
    )
    seeds = [
        pl.read_parquet(score.E201_PREDS / f"arm2_seed{seed}.parquet").select(
            "site", "power_fcst_init_time", "valid_time", q50_n="q50"
        )
        for seed in score.N_SEEDS
    ]
    network = (
        pl.concat(seeds)
        .group_by("site", "power_fcst_init_time", "valid_time")
        .agg(pl.col("q50_n").mean())
    )
    features = pl.concat(
        [
            pl.read_parquet(FEATURES / f"{site}.parquet")
            .filter(pl.col("period") == "scored")
            .select("power_fcst_init_time", "valid_time", "drv_temp_c", "level", "anomaly")
            .with_columns(site=pl.lit(site))
            for site in common.SCORED_SITES
        ]
    )
    frame = (
        stored.join(network, on=list(score.KEYS), how="inner")
        .join(features, on=list(score.KEYS), how="inner")
        .with_columns(correction=pl.col("q50_n") - pl.col("q50"))
    )
    holidays = pl.Series(sorted(bank_holiday_names())).cast(pl.Date)
    local = pl.col("valid_time").dt.convert_time_zone(LONDON)
    return frame.with_columns(
        day_type=pl.when(local.dt.date().is_in(holidays))
        .then(3)
        .when(local.dt.weekday() == 7)
        .then(2)
        .when(local.dt.weekday() == 6)
        .then(1)
        .otherwise(0),
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        day_of_year=local.dt.ordinal_day(),
    ).filter(pl.col("drv_temp_c").is_finite())


def hat_basis(*, temperature: np.ndarray) -> np.ndarray:
    """Return piecewise-linear hat functions on `KNOTS`; temperatures are clipped to the range."""
    clipped = np.clip(temperature, KNOTS[0], KNOTS[-1])
    position = np.interp(clipped, KNOTS, np.arange(len(KNOTS)))
    lower = np.floor(position).astype(int).clip(0, len(KNOTS) - 2)
    weight = position - lower
    basis = np.zeros((len(temperature), len(KNOTS)))
    basis[np.arange(len(temperature)), lower] = 1.0 - weight
    basis[np.arange(len(temperature)), lower + 1] = weight
    return basis


def design(*, frame: pl.DataFrame, extras: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return the design matrix and the penalty matrix (smoothness 1) of the surface."""
    basis = hat_basis(temperature=frame["drv_temp_c"].to_numpy())
    band = frame["hour_band"].to_numpy().astype(int)
    day = frame["day_type"].to_numpy().astype(int)
    cell = band * 4 + day
    n_cells = 16
    blocks = np.zeros((len(frame), n_cells * len(KNOTS)))
    for c in range(n_cells):
        rows = cell == c
        blocks[rows, c * len(KNOTS) : (c + 1) * len(KNOTS)] = basis[rows]
    columns = [blocks]
    if "filter" in extras:
        for name in ("level", "anomaly"):
            value = frame[name].to_numpy().astype(float)[:, None]
            columns.append(value * (band[:, None] == np.arange(4)[None, :]))
    if "season" in extras:
        angle = 2 * np.pi * frame["day_of_year"].to_numpy() / 365.25
        columns.append(
            np.column_stack([np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle)])
        )
    x = np.hstack(columns)
    second = np.diff(np.eye(len(KNOTS)), n=2, axis=0)
    smooth = second.T @ second
    penalty = np.zeros((x.shape[1], x.shape[1]))
    for c in range(n_cells):
        penalty[c * len(KNOTS) : (c + 1) * len(KNOTS), c * len(KNOTS) : (c + 1) * len(KNOTS)] = (
            smooth
        )
    return x, penalty


def fit_predict(
    *,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    penalty: np.ndarray,
    smoothness: float,
    train: np.ndarray,
) -> np.ndarray:
    """Fit the weighted penalised least squares on the `train` rows; return the fit on all rows."""
    xt, yt, wt = x[train], y[train], w[train]
    gram = xt.T @ (xt * wt[:, None]) + smoothness * penalty * wt.sum() / 1000.0
    gram += RIDGE * wt.sum() / 1000.0 * np.eye(x.shape[1])
    theta = np.linalg.solve(gram, xt.T @ (wt * yt))
    return x @ theta


def explained(*, y: np.ndarray, fit: np.ndarray, w: np.ndarray, mask: np.ndarray) -> float:
    """Return `1 - SSE / SST` over the masked rows, with weights `w` and the weighted mean."""
    wm = w[mask]
    mean = (wm * y[mask]).sum() / wm.sum()
    sse = (wm * (y[mask] - fit[mask]) ** 2).sum()
    sst = (wm * (y[mask] - mean) ** 2).sum()
    return float(1.0 - sse / sst)


def main() -> None:
    """Fit S and its variants, and write `work/surface.json`."""
    frame = correction_rows()
    counts = frame.group_by("site", "valid_time").len().rename({"len": "n_runs"})
    frame = frame.join(counts, on=["site", "valid_time"]).with_columns(
        weight=1.0 / pl.col("n_runs")
    )
    y = frame["correction"].to_numpy().astype(float)
    w = frame["weight"].to_numpy().astype(float)
    months = frame["month"].to_numpy()
    temperature = frame["drv_temp_c"].to_numpy()
    out: dict = {
        "rows": frame.height,
        "variance_of_correction": float(np.average((y - np.average(y, weights=w)) ** 2, weights=w)),
    }
    for name, extras in (
        ("S", ()),
        ("S_filter", ("filter",)),
        ("S_season", ("season",)),
        ("S_both", ("filter", "season")),
    ):
        x, penalty = design(frame=frame, extras=extras)
        for smoothness in SMOOTH_WEIGHTS:
            in_sample = fit_predict(
                x=x, y=y, w=w, penalty=penalty, smoothness=smoothness, train=np.ones(len(y), bool)
            )
            held = np.zeros(len(y))
            in_range = np.zeros(len(y), bool)
            for month in np.unique(months):
                test = months == month
                held[test] = fit_predict(
                    x=x, y=y, w=w, penalty=penalty, smoothness=smoothness, train=~test
                )[test]
                lo, hi = temperature[~test].min(), temperature[~test].max()
                in_range[test] = (temperature[test] >= lo) & (temperature[test] <= hi)
            all_rows = np.ones(len(y), bool)
            result = {
                "in_sample": explained(y=y, fit=in_sample, w=w, mask=all_rows),
                "leave_one_month_out": explained(y=y, fit=held, w=w, mask=all_rows),
                "leave_one_month_out_in_range": explained(y=y, fit=held, w=w, mask=in_range),
                "share_in_range": float(w[in_range].sum() / w.sum()),
            }
            out[f"{name}_smooth{smoothness:g}"] = result
            if name == "S" and smoothness == 1.0:
                frame = frame.with_columns(s_held=pl.Series(held), s_in=pl.Series(in_sample))
    path = common.WORK_DIR / "surface_rows.parquet"
    frame.select(
        "site",
        "power_fcst_init_time",
        "valid_time",
        "drv_temp_c",
        "hour_band",
        "day_type",
        "correction",
        "s_held",
        "weight",
        "month",
    ).write_parquet(path)
    common.write_json_atomic(path=common.WORK_DIR / "surface.json", content=out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
