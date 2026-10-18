"""Run experiment 00092's drift diagnostics D1 to D6 on stored predictions and print every number.

Nothing here fits a model. The residual is r = target - F in points of capacity, where F is
experiment 00017's model frozen at its first origin (ERA5 weather), so r carries all drift since
2022-01. Every table is written to `data/ml-experiments/00092-.../work/results/` for the charts.

    cd ml-experiments/00092-lv0-drift-diagnostics-on-stored-predictions-demand
    uv run python e00092_diagnostics.py
"""

import itertools
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
RESULTS: Final[Path] = (
    DATA / "00092-lv0-drift-diagnostics-on-stored-predictions-demand/work/results"
)
GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
N_RESAMPLES: Final[int] = 2000
SEED: Final[int] = 20261018
BANDS: Final[dict[str, tuple[int, int]]] = {
    "night (00-06)": (0, 6),
    "midday (10-15)": (10, 15),
    "evening (16-20)": (16, 20),
}
BAND_NAMES: Final[list[str]] = [*BANDS, "rest", "all hours"]
MIN_MONTHS: Final[int] = 24
"""A series needs this many months with data before a trend or step is read from it."""
STEP_POINTS: Final[float] = 2.0
MIN_SEGMENT_WEEKS: Final[int] = 8
QUANTILES: Final[list[float]] = [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95]


def _first(pattern: str) -> str:
    return str(next(DATA.glob(pattern)))


def load_residual() -> pl.DataFrame:
    """Return experiment 17's scored rows with local hour, month index, and r in points."""
    rows = pl.read_parquet(_first("00017-*/results/scored_rows.parquet"))
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    hour = local.dt.hour()
    band = (
        pl.when(hour < 6)
        .then(pl.lit("night (00-06)"))
        .when(hour.is_between(10, 14))
        .then(pl.lit("midday (10-15)"))
        .when(hour.is_between(16, 19))
        .then(pl.lit("evening (16-20)"))
        .otherwise(pl.lit("rest"))
    )
    rows = rows.with_columns(
        r=(pl.col("target") - pl.col("F")).cast(pl.Float64) * 100.0,
        band=band,
        summer=local.dt.month().is_between(4, 9),
        month_idx=(pl.col("valid_time").dt.year() - 2022) * 12
        + pl.col("valid_time").dt.month()
        - 1,
        week_idx=((pl.col("valid_time") - GRID_START).dt.total_days() // 7),
    )
    months = rows.group_by("time_series_id").agg(n_months=pl.col("month_idx").n_unique())
    eligible = months.filter(pl.col("n_months") >= MIN_MONTHS)["time_series_id"]
    print(
        "series dropped for under",
        MIN_MONTHS,
        "months of data:",
        months.filter(pl.col("n_months") < MIN_MONTHS).to_dicts(),
    )
    return rows.filter(pl.col("time_series_id").is_in(eligible))


def ols_slopes(*, values: np.ndarray, weights: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return the weighted OLS slope of `values[..., month]` on `t`; NaN months get zero weight."""
    w = weights * np.isfinite(values)
    v = np.nan_to_num(values)
    sw = w.sum(-1)
    mt = (w * t).sum(-1) / np.maximum(sw, 1e-9)
    mv = (w * v).sum(-1) / np.maximum(sw, 1e-9)
    cov = (w * (t - mt[..., None]) * (v - mv[..., None])).sum(-1)
    var = (w * (t - mt[..., None]) ** 2).sum(-1)
    out = cov / np.where(var > 1e-9, var, np.nan)
    out[sw < 6] = np.nan  # a band-and-season cell holds half the months, so 6 is a floor
    return out


def monthly_cube(*, residual: pl.DataFrame) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Return mean r as [series, 8 groups, month] plus the series ids and the months in years."""
    series = sorted(residual["time_series_id"].unique().to_list())
    n_months = int(residual.select(pl.col("month_idx").max()).item()) + 1
    groups = [*BANDS, "rest"]
    cube = np.full((len(series), 2 * len(groups) + 1, n_months), np.nan)
    agg = residual.group_by("time_series_id", "band", "summer", "month_idx").agg(pl.col("r").mean())
    for s, b, summer, m, v in agg.iter_rows():
        cube[series.index(s), groups.index(b) + (0 if summer else len(groups)), m] = v
    allhrs = residual.group_by("time_series_id", "month_idx").agg(pl.col("r").mean())
    for s, m, v in allhrs.iter_rows():
        cube[series.index(s), -1, m] = v
    return cube, series, (np.arange(n_months) + 0.5) / 12.0


def d1(*, residual: pl.DataFrame) -> dict:
    """D1: slope in points of capacity a year per hour band and season, with a month bootstrap."""
    cube, series, t = monthly_cube(residual=residual)
    groups = [*BANDS, "rest"]
    labels = [f"{g}, Apr-Sep" for g in groups] + [f"{g}, Oct-Mar" for g in groups] + ["all hours"]
    ones = np.ones(cube.shape[-1])
    slopes = ols_slopes(values=cube, weights=ones, t=t)
    rng = np.random.default_rng(SEED)
    boot = np.empty((N_RESAMPLES, len(labels)))
    boot_share = np.empty((N_RESAMPLES, len(series)))
    for i in range(N_RESAMPLES):
        w = np.bincount(rng.integers(0, cube.shape[-1], cube.shape[-1]), minlength=cube.shape[-1])
        sl = ols_slopes(values=cube, weights=w.astype(float), t=t)
        boot[i] = np.nanmedian(sl, axis=0)
        boot_share[i] = trend_share(slopes=sl)
    share = trend_share(slopes=slopes)
    rows = []
    for k, label in enumerate(labels):
        lo, hi = np.nanpercentile(boot[:, k], [2.5, 97.5])
        rows.append(
            {
                "group": label,
                "median_slope": float(np.nanmedian(slopes[:, k])),
                "lo": float(lo),
                "hi": float(hi),
            }
        )
    n_series = (boot_share > 0.5).sum(axis=1)
    out = pl.DataFrame(rows)
    per_series = pl.DataFrame(
        {"time_series_id": series, "slope_all_hours": slopes[:, -1], "trend_share_common": share}
    )
    # Rolling 12-month version of the common share, to include step-like drift (point estimate).
    roll_share = rolling_share(cube=cube)
    per_series = per_series.with_columns(rolling_share_common=pl.Series(roll_share))
    summary = {
        "n_series_over_half_trend": int((share > 0.5).sum()),
        "n_series_over_half_trend_ci": [float(x) for x in np.percentile(n_series, [2.5, 97.5])],
        "n_series_over_half_rolling": int((roll_share > 0.5).sum()),
        "n_series_slope_ge_1_abs": int((np.abs(slopes[:, -1]) >= 1.0).sum()),
        "n_series_any_band_ge_1_abs": int((np.nanmax(np.abs(slopes[:, :8]), axis=1) >= 1.0).sum()),
        "median_abs_all_hours_slope": float(np.nanmedian(np.abs(slopes[:, -1]))),
        "pooled_all_hours_slope": [
            float(np.nanmedian(slopes[:, -1])),
            *[float(x) for x in np.nanpercentile(boot[:, -1], [2.5, 97.5])],
        ],
    }
    out.write_parquet(RESULTS / "d1_band_slopes.parquet")
    per_series.write_parquet(RESULTS / "d1_per_series.parquet")
    pl.DataFrame(
        {"time_series_id": series, **{lab: slopes[:, k] for k, lab in enumerate(labels)}}
    ).write_parquet(RESULTS / "d1_slopes_by_series.parquet")
    print(
        "\nD1 pooled (median over series) slope, points of capacity per year, 95% month-bootstrap"
    )
    print(out)
    print(per_series)
    print(summary)
    allh = cube[:, -1, :]
    level_change = np.array(
        [
            abs(np.nanmean(row[np.isfinite(row)][-12:]) - np.nanmean(row[np.isfinite(row)][:12]))
            for row in allh
        ]
    )
    return {
        "summary": summary,
        "cube": cube,
        "series": series,
        "slopes_all": slopes[:, -1],
        "level_change": level_change,
    }


def trend_share(*, slopes: np.ndarray) -> np.ndarray:
    """Share of the trend variance carried by the cross-band mean slope (all-hours component)."""
    bands = slopes[:, [0, 1, 2, 3]] / 2 + slopes[:, [4, 5, 6, 7]] / 2
    common = bands.mean(axis=1)
    shape = ((bands - common[:, None]) ** 2).mean(axis=1)
    return common**2 / (common**2 + shape)


def rolling_share(*, cube: np.ndarray) -> np.ndarray:
    """Share of the variance of the 12-month rolling mean that is common to the four bands."""
    out = np.full(cube.shape[0], np.nan)
    for s in range(cube.shape[0]):
        bands = []
        for b in range(4):
            both = np.nanmean(np.stack([cube[s, b], cube[s, 4 + b]]), axis=0)
            roll = np.array(
                [
                    np.nanmean(both[i : i + 12])
                    if np.isfinite(both[i : i + 12]).sum() >= 9
                    else np.nan
                    for i in range(len(both) - 11)
                ]
            )
            bands.append(roll)
        bands_arr = np.array(bands)
        ok = np.isfinite(bands_arr).all(axis=0)
        if ok.sum() < 6:
            continue
        b = bands_arr[:, ok]
        common = b.mean(axis=0)
        shape = (b - common).var(axis=1).mean()
        out[s] = common.var() / (common.var() + shape)
    return out


def pelt(*, y: np.ndarray, penalty: float, min_size: int) -> list[int]:
    """Return the change-point indices of the exact optimal partition with an L2 cost.

    This is the optimum that the pruned exact linear time method (PELT) also finds. The series are
    short (about 230 weeks), so the quadratic search without pruning is fast enough and simpler.
    """
    n = len(y)
    cs = np.concatenate([[0.0], np.cumsum(y)])
    cs2 = np.concatenate([[0.0], np.cumsum(y**2)])
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    for t in range(min_size, n + 1):
        starts = np.arange(0, t - min_size + 1)
        starts = starts[np.isfinite(best[starts])]
        seg = cs[t] - cs[starts]
        cost = cs2[t] - cs2[starts] - seg**2 / (t - starts)
        total = best[starts] + cost + penalty
        k = int(np.argmin(total))
        best[t], last[t] = total[k], starts[k]
    cps = []
    t = n
    while t > 0 and last[t] > 0:
        cps.append(int(last[t]))
        t = int(last[t])
    return sorted(cps)


def weekly_series(
    *, residual: pl.DataFrame, night_only: bool
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return weekly mean r per series as (week index, value); weeks under 50% covered drop."""
    frame = residual.filter(pl.col("band") == "night (00-06)") if night_only else residual
    per_week = 336 * (6 / 24 if night_only else 1.0)
    agg = frame.group_by("time_series_id", "week_idx").agg(pl.col("r").mean(), n=pl.len())
    out = {}
    for s in sorted(agg["time_series_id"].unique().to_list()):
        a = agg.filter((pl.col("time_series_id") == s) & (pl.col("n") >= 0.5 * per_week)).sort(
            "week_idx"
        )
        out[s] = (a["week_idx"].to_numpy(), a["r"].to_numpy())
    return out


def calibrate_penalty(*, weekly: dict[int, tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """Return the smallest penalty multiplier whose placebo gives under 1 step per 5 years."""
    rng = np.random.default_rng(SEED)
    for mult in [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16]:
        counts, years = [], []
        for _, y in weekly.values():
            sigma2 = float(np.var(np.diff(y)) / 2)
            for _ in range(5):
                shuffled = rng.permutation(y)
                counts.append(
                    len(
                        pelt(
                            y=shuffled,
                            penalty=mult * sigma2 * np.log(len(y)),
                            min_size=MIN_SEGMENT_WEEKS,
                        )
                    )
                )
                years.append(len(y) / 52.18)
        rate = float(np.sum(counts) / np.sum(years) * 5)
        if rate < 1.0:
            return float(mult), rate
    return float("inf"), float("nan")


def d2(*, residual: pl.DataFrame) -> pl.DataFrame:
    """D2: census of persistent steps in the weekly mean of r, for all hours and for night only."""
    alarms = pl.read_parquet(_first("00082-*/alarms.parquet")).filter(pl.col("setting") == "XL")
    alarm_week = {
        s: (g["stamp"].to_numpy() / 336.0)
        for s, g in alarms.group_by("series", maintain_order=True)
    }
    alarm_week = {int(k[0]) if isinstance(k, tuple) else int(k): v for k, v in alarm_week.items()}
    census = []
    for night in (False, True):
        weekly = weekly_series(residual=residual, night_only=night)
        mult, placebo = calibrate_penalty(weekly=weekly)
        print(f"\nD2 night_only={night}: penalty {mult}x; placebo steps per 5y {placebo:.2f}")
        steps = []
        for s, (wk, y) in weekly.items():
            sigma2 = float(np.var(np.diff(y)) / 2)
            cps = pelt(y=y, penalty=mult * sigma2 * np.log(len(y)), min_size=MIN_SEGMENT_WEEKS)
            edges = [0, *cps, len(y)]
            means = [y[a:b].mean() for a, b in itertools.pairwise(edges)]
            for k, cp in enumerate(cps):
                size = float(means[k + 1] - means[k])
                if abs(size) >= STEP_POINTS:
                    steps.append(
                        {
                            "night_only": night,
                            "time_series_id": s,
                            "week": float(wk[cp]),
                            "size": size,
                        }
                    )
        frame = pl.DataFrame(
            steps,
            schema={
                "night_only": pl.Boolean,
                "time_series_id": pl.Int64,
                "week": pl.Float64,
                "size": pl.Float64,
            },
        )
        tags = []
        for row in frame.iter_rows(named=True):
            same_event = np.any(
                np.abs(alarm_week.get(row["time_series_id"], np.array([1e9])) - row["week"]) <= 2.0
            )
            near = frame.filter(
                (pl.col("time_series_id") != row["time_series_id"])
                & ((pl.col("week") - row["week"]).abs() <= 4)
            )
            common = near.filter(pl.col("size") * row["size"] > 0)["time_series_id"].n_unique() >= 7
            transfer = near.filter(pl.col("size") * row["size"] < 0).height > 0
            tags.append(
                "common to most series"
                if common
                else "opposite step at a neighbour"
                if transfer
                else "matches an RO-NC alarm"
                if same_event
                else "none"
            )
        frame = frame.with_columns(
            tag=pl.Series(tags, dtype=pl.String),
            penalty_multiplier=pl.lit(mult),
            placebo_rate=pl.lit(placebo),
        )
        census.append(frame)
        print(frame.group_by("tag").len().sort("tag"))
        print(
            f"steps: {frame.height}; series with a step: "
            f"{frame['time_series_id'].n_unique()} of {len(weekly)}"
        )
    out = pl.concat(census)
    out.write_parquet(RESULTS / "d2_steps.parquet")
    return out


def d3(*, cube: np.ndarray) -> dict:
    """D3: share of the monthly level's variance in the first principal component."""
    x = cube[:, -1, :]
    x = x - np.nanmean(x, axis=1, keepdims=True)

    def stats(m: np.ndarray) -> tuple[float, float]:
        filled = np.nan_to_num(m)
        sv = np.linalg.svd(filled, compute_uv=False)
        corr = np.corrcoef(filled)
        return float(sv[0] ** 2 / (sv**2).sum()), float(
            np.nanmedian(corr[np.triu_indices_from(corr, 1)])
        )

    point = stats(x)
    rng = np.random.default_rng(SEED)
    boot = np.array(
        [stats(x[:, rng.integers(0, x.shape[1], x.shape[1])]) for _ in range(N_RESAMPLES)]
    )
    out = {
        "pc1_share": point[0],
        "pc1_share_ci": [float(v) for v in np.percentile(boot[:, 0], [2.5, 97.5])],
        "median_pairwise_correlation": point[1],
        "median_pairwise_correlation_ci": [
            float(v) for v in np.percentile(boot[:, 1], [2.5, 97.5])
        ],
    }
    print("\nD3", out)
    filled = np.nan_to_num(x)
    _, s, vt = np.linalg.svd(filled, full_matrices=False)
    pl.DataFrame({"month": np.arange(x.shape[1]), "pc1": vt[0] * s[0]}).write_parquet(
        RESULTS / "d3_pc1.parquet"
    )
    return out


def _mae_by(*, frame: pl.DataFrame, column: str) -> pl.DataFrame:
    return frame.group_by("time_series_id", "month_idx").agg(
        (pl.col(column) - pl.col("target")).abs().mean().mul(100).alias(column),
        pl.len().alias(f"n_{column}"),
    )


def d4(*, residual: pl.DataFrame, level_change: np.ndarray, series: list[int]) -> dict:
    """D4: extra-years loss against each series' total level change, in experiments 17 and 89."""
    drift = pl.DataFrame({"time_series_id": series, "level_change": level_change})
    mae = _mae_by(frame=residual, column="A").join(
        _mae_by(frame=residual, column="W12"), on=["time_series_id", "month_idx"]
    )
    per_series = (
        mae.with_columns(loss=pl.col("A") - pl.col("W12"))
        .group_by("time_series_id")
        .agg(loss=pl.col("loss").mean())
        .join(drift, on="time_series_id")
    )
    out: dict[str, Any] = {"exp17_A_minus_W12": corr_with_ci(per_series, "level_change", "loss")}
    out["exp17_mean_loss_points"] = float(per_series.select(pl.col("loss").mean()).item())
    # Experiment 89: L minus R, scored on experiment 17's target (same series and valid time).
    target = residual.select("time_series_id", "valid_time", "target")
    arms = {}
    for arm in ("r", "l"):
        files = sorted(DATA.glob(f"00089-*/work/anchor_preds/{arm}/*.parquet"))
        frame = pl.concat([pl.read_parquet(f) for f in files]).join(
            target, on=["time_series_id", "valid_time"]
        )
        arms[arm] = frame.group_by("time_series_id").agg(
            ((pl.col("q50") - pl.col("target")).abs().mean() * 100).alias(arm)
        )
    e89 = (
        arms["l"]
        .join(arms["r"], on="time_series_id")
        .with_columns(loss=pl.col("l") - pl.col("r"))
        .join(drift, on="time_series_id")
    )
    out["exp89_L_minus_R"] = corr_with_ci(e89, "level_change", "loss")
    for key, frame in (("exp17_A_minus_W12", per_series), ("exp89_L_minus_R", e89)):
        trimmed = frame.sort("level_change").head(frame.height - 1)
        out[key]["without_largest_level_change"] = {
            "pearson": float(
                np.corrcoef(trimmed["level_change"].to_numpy(), trimmed["loss"].to_numpy())[0, 1]
            ),
            "spearman": corr_with_ci(trimmed, "level_change", "loss")["spearman"],
        }
    out["exp89_mean_loss_points"] = float(e89.select(pl.col("loss").mean()).item())
    per_series.write_parquet(RESULTS / "d4_exp17.parquet")
    e89.write_parquet(RESULTS / "d4_exp89.parquet")
    print("\nD4", out)
    return out


def corr_with_ci(frame: pl.DataFrame, x: str, y: str) -> dict:
    """Return Pearson and Spearman correlations of two columns with a series bootstrap interval."""
    a, b = frame[x].to_numpy(), frame[y].to_numpy()
    rng = np.random.default_rng(SEED)

    def pearson(i: np.ndarray) -> float:
        return float(np.corrcoef(a[i], b[i])[0, 1])

    boot = [pearson(rng.integers(0, len(a), len(a))) for _ in range(N_RESAMPLES)]
    ranks = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    return {
        "n": len(a),
        "pearson": pearson(np.arange(len(a))),
        "pearson_ci": [float(v) for v in np.nanpercentile(boot, [2.5, 97.5])],
        "spearman": float(np.corrcoef(ranks(a), ranks(b))[0, 1]),
    }


def d5(*, residual: pl.DataFrame) -> dict:
    """D5: pooling gap and pooled monthly bias against each series' level change (experiment 24)."""
    target = residual.select("time_series_id", "valid_time", "target")
    level = residual.group_by("time_series_id", "month_idx").agg(level=pl.col("r").mean())
    arms = {}
    for name in ("own", "all_id", "type_id", "own_stack", "all_id_stack", "type_id_stack"):
        q = pl.read_parquet(_first(f"00024-*/predictions/{name}/predictions.parquet"))
        q = q.filter(pl.col("seed") == 0).join(target, on=["time_series_id", "valid_time"])
        arms[name] = q.with_columns(
            err=(pl.col("q50") - pl.col("target")) * 100,
            month_idx=(pl.col("valid_time").dt.year() - 2022) * 12
            + pl.col("valid_time").dt.month()
            - 1,
        )
    out = {}
    for pooled, own, label in (
        ("all_id", "own", "uncorrected"),
        ("all_id_stack", "own_stack", "corrected"),
        ("type_id", "own", "type uncorrected"),
    ):
        key = ["time_series_id", "power_fcst_init_time", "valid_time"]
        j = (
            arms[pooled]
            .select(*key, "month_idx", e_p="err")
            .join(arms[own].select(*key, e_o="err"), on=key)
        )
        gap = j.group_by("time_series_id").agg(
            gap=(pl.col("e_p").abs() - pl.col("e_o").abs()).mean()
        )
        bias = (
            j.group_by("time_series_id", "month_idx")
            .agg(bias_diff=(pl.col("e_p") - pl.col("e_o")).mean())
            .join(level, on=["time_series_id", "month_idx"])
        )
        # Remove each series' own mean so the correlation is about change over time, not level.
        bias = bias.with_columns(
            bias_diff=pl.col("bias_diff") - pl.col("bias_diff").mean().over("time_series_id"),
            level=pl.col("level") - pl.col("level").mean().over("time_series_id"),
        )
        a, b = bias["level"].to_numpy(), bias["bias_diff"].to_numpy()
        months = bias["month_idx"].to_numpy()
        rng = np.random.default_rng(SEED)
        uniq = np.unique(months)
        boot = []
        for _ in range(N_RESAMPLES):
            pick = rng.choice(uniq, len(uniq))
            idx = np.concatenate([np.where(months == m)[0] for m in pick])
            boot.append(np.corrcoef(a[idx], b[idx])[0, 1])
        out[label] = {
            "mean_pooled_minus_own_points": float(gap.select(pl.col("gap").mean()).item()),
            "bias_level_correlation": float(np.corrcoef(a, b)[0, 1]),
            "bias_level_correlation_ci": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
        }
        if label == "uncorrected":
            bias.write_parquet(RESULTS / "d5_bias_vs_level.parquet")
            gap.write_parquet(RESULTS / "d5_gap_by_series.parquet")
    print("\nD5", out)
    return out


def pinball(*, frame: pl.DataFrame) -> pl.Expr:
    """Return the mean pinball loss over the nine quantile levels, in the units of `actual`."""
    terms = []
    for q, name in zip(
        QUANTILES, ["q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95"], strict=True
    ):
        d = pl.col("actual") - pl.col(name)
        terms.append(pl.max_horizontal(q * d, (q - 1) * d))
    return pl.sum_horizontal(terms) / len(terms)


def _flag_difference(*, gap: np.ndarray, mask: np.ndarray) -> float:
    """Return the mean gap where the flag is true minus the mean gap where it is false."""
    if not mask.any() or mask.all():
        return float("nan")
    return float(gap[mask].mean() - gap[~mask].mean())


def d6(
    *, residual: pl.DataFrame, steps: pl.DataFrame, slopes_all: np.ndarray, series: list[int]
) -> dict:
    """D6: weekly-minus-monthly refit gap (experiment 48) split by drift flags."""
    frames = {}
    for name in ("weekly", "monthly"):
        q = pl.read_parquet(_first(f"00048-*/predictions/{name}/predictions.parquet")).filter(
            pl.col("group") == "Primary"
        )
        q = q.filter(pl.col("actual").is_not_null() & pl.col("actual").is_not_nan())
        frames[name] = q.with_columns(
            score=pinball(frame=q),
            month_idx=(pl.col("valid_time").dt.year() - 2022) * 12
            + pl.col("valid_time").dt.month()
            - 1,
        )
    key = ["time_series_id", "power_fcst_init_time", "valid_time"]
    j = (
        frames["monthly"]
        .select(*key, "month_idx", "actual", m="score")
        .join(frames["weekly"].select(*key, w="score"), on=key)
    )
    # Actual power is already a fraction of capacity, so times 100 gives points of capacity.
    per = j.group_by("time_series_id", "month_idx").agg(
        gap_rel=(pl.col("m") - pl.col("w")).mean() * 100
    )
    ids = set(series)
    per = per.filter(pl.col("time_series_id").is_in(ids))
    s_steps = steps.filter(~pl.col("night_only"))
    step_dates = [GRID_START + timedelta(weeks=float(w)) for w in s_steps["week"]]
    step_months = {
        (series_id, (d.year - 2022) * 12 + d.month - 1)
        for series_id, d in zip(s_steps["time_series_id"], step_dates, strict=True)
    }
    top = set(
        np.array(series)[np.argsort(-np.abs(slopes_all))[: max(1, len(series) // 4)]].tolist()
    )
    alarms = pl.read_parquet(_first("00082-*/alarms.parquet")).filter(pl.col("setting") == "XL")
    alarm_months = set()
    for s, stamp in zip(alarms["series"].to_list(), alarms["stamp"].to_list(), strict=True):
        d = GRID_START + timedelta(minutes=30 * stamp)
        alarm_months.add((s, (d.year - 2022) * 12 + d.month - 1))
    cap = pl.read_parquet(_first("00078-*/work/capacity_vintages.parquet"))
    site = (
        pl.read_parquet(_first("00048-*/predictions/weekly/predictions.parquet"))
        .select("time_series_id", "site")
        .unique()
    )
    growth = (
        cap.sort("run_date")
        .group_by("site")
        .agg(g=(pl.col("weekly_mw").last() / pl.col("weekly_mw").first() - 1))
        .join(site, on="site")
    )
    grown = set(growth.filter(pl.col("g") > 0.05)["time_series_id"].to_list())
    per = per.with_columns(
        step=pl.struct("time_series_id", "month_idx").map_elements(
            lambda r: (r["time_series_id"], r["month_idx"]) in step_months, return_dtype=pl.Boolean
        ),
        steep=pl.col("time_series_id").is_in(top),
        grown=pl.col("time_series_id").is_in(grown),
        event=pl.struct("time_series_id", "month_idx").map_elements(
            lambda r: (r["time_series_id"], r["month_idx"]) in alarm_months, return_dtype=pl.Boolean
        ),
    ).with_columns(any_flag=pl.col("step") | pl.col("steep") | pl.col("grown") | pl.col("event"))
    per.write_parquet(RESULTS / "d6_gap.parquet")
    tails = {}
    for name, frame in frames.items():
        f = frame.filter(pl.col("time_series_id").is_in(ids))
        row = {}
        for level, col in ((0.05, "q05"), (0.95, "q95")):
            d = pl.col("actual") - pl.col(col)
            loss = pl.max_horizontal(level * d, (level - 1) * d)
            r = f.select(cov=(pl.col("actual") <= pl.col(col)).mean(), pin=loss.mean() * 100)
            row[col] = {"coverage": float(r["cov"][0]), "pinball_points": float(r["pin"][0])}
        tails[name] = row
    total = per["gap_rel"].sum()
    rng = np.random.default_rng(SEED)
    months = per["month_idx"].to_numpy()
    uniq = np.unique(months)
    out = {
        "n_series": int(per["time_series_id"].n_unique()),
        "n_series_months": per.height,
        "mean_gap_points": float(per.select(pl.col("gap_rel").mean()).item()),
    }
    for flag in ("step", "steep", "grown", "event", "any_flag"):
        mask = per[flag].to_numpy()
        gap = per["gap_rel"].to_numpy()

        boot, boot_diff = [], []
        for _ in range(N_RESAMPLES):
            pick = rng.choice(uniq, len(uniq))
            idx = np.concatenate([np.where(months == m)[0] for m in pick])
            boot.append(float(gap[idx][mask[idx]].sum() / gap[idx].sum()))
            boot_diff.append(_flag_difference(gap=gap[idx], mask=mask[idx]))
        out[flag] = {
            "mean_gap_flagged_minus_unflagged": _flag_difference(gap=gap, mask=mask),
            "mean_gap_diff_ci": [float(v) for v in np.nanpercentile(boot_diff, [2.5, 97.5])],
            "share_of_gap": float(gap[mask].sum() / total),
            "share_ci": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
            "share_of_series_months": float(mask.mean()),
        }
    out["tails_primary"] = tails
    print("\nD6", json.dumps(out, indent=1))
    return out


def main() -> None:
    """Run every diagnostic and write the summary."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    residual = load_residual()
    print("rows", residual.height, "series", residual["time_series_id"].n_unique())
    print(residual.select(mean_r=pl.col("r").mean(), std_r=pl.col("r").std()))
    one = d1(residual=residual)
    steps = d2(residual=residual)
    three = d3(cube=one["cube"])
    four = d4(residual=residual, level_change=one["level_change"], series=one["series"])
    five = d5(residual=residual)
    six = d6(residual=residual, steps=steps, slopes_all=one["slopes_all"], series=one["series"])
    (RESULTS / "summary.json").write_text(
        json.dumps({"d1": one["summary"], "d3": three, "d4": four, "d5": five, "d6": six}, indent=1)
    )


if __name__ == "__main__":
    main()
