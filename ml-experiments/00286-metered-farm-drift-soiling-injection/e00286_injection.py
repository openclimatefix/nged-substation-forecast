# ruff: noqa: E501
"""Inject known drift and soiling into the six metered solar farms and see what two indices recover.

Writes private tables under `data/ml-experiments/00286-*/`. Farms appear only as labels A to F.

Run: `uv run --no-sync python ml-experiments/00286-metered-farm-drift-soiling-injection/e00286_injection.py`
"""

from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.pv_dataset import pv_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00286-metered-farm-drift-soiling-injection"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
FARMS_FILE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00179-metered-solar-farm-exploration" / "cleaned_solar_farms.parquet"
)
TP_FILE: Final[Path] = Path(
    "/home/jack/dev/nged-substation-forecast/data/studies/weather/ERA5/stage1_box/tp.parquet"
)
DRIFT_RATES: Final[tuple[float, ...]] = (0.5, 1.0, 2.0)
"""Injected linear losses, percent of output per year."""
N_STARTS: Final[int] = 8
SOILING_PER_DAY: Final[float] = 0.001
RAIN_RESET_MM: Final[float] = 5.0
N_BOOT: Final[int] = 500
CS_MIN: Final[float] = 300.0
CS_MAX_B: Final[float] = 650.0
"""Method B stays under the flat top that experiment 179 found above about 650 W/m²."""
NEIGHBOUR_BAND: Final[tuple[float, float]] = (0.8, 1.2)
MIN_NEIGHBOURS: Final[int] = 2
MIN_MONTH_ROWS: Final[int] = 40
MIN_MONTH_ROWS_B: Final[int] = 20
MIN_DAY_ROWS_A: Final[int] = 6
MIN_DAY_ROWS_B: Final[int] = 4
REL_RANGE: Final[tuple[float, float]] = (0.75, 1.33)
RECOVERY_RANGE: Final[tuple[float, float]] = (0.7, 1.3)
ERA5_LAST_DAY: Final[date] = date(2026, 7, 31)
"""Last day of the private ERA5 rainfall copy; later days are left out of the soiling readout."""


def year_fraction(month_start: pl.Expr) -> pl.Expr:
    """Return the decimal year of a month's first day."""
    return month_start.dt.year() + (month_start.dt.month() - 1) / 12


def fit_trend(*, t: np.ndarray, rel: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Return the straight-line slope in percent per year with a 90% month-resampled interval."""
    slope = 100 * np.polyfit(t, rel, 1)[0]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(t), len(t))
        if np.ptp(t[i]) == 0:
            continue
        draws.append(100 * np.polyfit(t[i], rel[i], 1)[0])
    return float(slope), float(np.percentile(draws, 5)), float(np.percentile(draws, 95))


def load_farms() -> tuple[pl.DataFrame, dict[str, dict[str, float]]]:
    """Return the wide per-farm power-over-capacity and clear-sky frames, and each farm's site row."""
    sites = pv_sites().sort("site")
    info = {
        r["site"]: {
            "tsid": r["time_series_id"],
            "cap": float(r["effective_capacity_mw"]),
            "lat": float(r["latitude"]),
            "lon": float(r["longitude"]),
        }
        for r in sites.iter_rows(named=True)
    }
    tsid_to_site = {v["tsid"]: k for k, v in info.items()}
    raw = pl.read_parquet(FARMS_FILE).filter(pl.col("drop_reason").is_null())
    parts = []
    for tsid, site in tsid_to_site.items():
        parts.append(
            raw.filter(pl.col("time_series_id") == tsid).select(
                "time",
                (pl.col("power") / info[site]["cap"]).alias(f"pu_{site}"),
                pl.col("clear_sky_ghi_w_m2").alias(f"cs_{site}"),
            )
        )
    wide = parts[0]
    for part in parts[1:]:
        wide = wide.join(part, on="time", how="full", coalesce=True)
    return wide.sort("time"), info


def daily_rain_reset(*, lat: float, lon: float, days: pl.Series) -> np.ndarray:
    """Return a boolean per day: whether ERA5 rainfall at the nearest grid cell is at least 5 mm."""
    cells = pl.scan_parquet(TP_FILE).select("latitude", "longitude").unique().collect()
    nearest = cells.with_columns(
        d=((pl.col("latitude") - lat) ** 2 + (pl.col("longitude") - lon) ** 2)
    ).sort("d")[0]
    cell_lat, cell_lon = nearest["latitude"][0], nearest["longitude"][0]
    rain = (
        pl.scan_parquet(TP_FILE)
        .filter((pl.col("latitude") == cell_lat) & (pl.col("longitude") == cell_lon))
        .select(
            (pl.col("time") - pl.duration(hours=1)).dt.date().alias("day"),
            (pl.col("value") * 1000).alias("mm"),
        )
        .group_by("day")
        .agg(pl.col("mm").sum())
        .collect()
    )
    joined = pl.DataFrame({"day": days}).join(rain, on="day", how="left")
    return (joined["mm"].fill_null(0.0) >= RAIN_RESET_MM).to_numpy()


def build_target_frame(*, wide: pl.DataFrame, target: str, sites: list[str]) -> pl.DataFrame:
    """Return one row per half-hour with the un-injected inputs of both methods for one farm.

    `ya` is method A's yield (power over capacity, per kW/m² of clear-sky irradiance).
    `r_<j>` is the ratio of the target's power to farm j's, null unless method B's mask holds.
    """
    others = [s for s in sites if s != target]
    t = wide["time"]
    pu = {s: wide[f"pu_{s}"].to_numpy() for s in sites}
    cs = {s: wide[f"cs_{s}"].to_numpy() for s in sites}
    yld = {s: pu[s] / (cs[s] / 1000.0) for s in sites}
    cal_month = t.dt.month().to_numpy()
    ref = {}
    for s in sites:
        ok = np.isfinite(yld[s]) & (cs[s] >= CS_MIN) & (cs[s] <= CS_MAX_B)
        by_month = np.full(13, np.nan)
        for month in range(1, 13):
            in_month = ok & (cal_month == month)
            if in_month.sum() >= MIN_MONTH_ROWS:
                by_month[month] = np.quantile(yld[s][in_month], 0.9)
        ref[s] = by_month[cal_month]
    avail = np.stack([np.isfinite(pu[s]) & np.isfinite(cs[s]) & (cs[s] > CS_MIN) for s in others])
    frac = np.stack([yld[s] / ref[s] for s in others])
    in_band = (frac >= NEIGHBOUR_BAND[0]) & (frac <= NEIGHBOUR_BAND[1])
    with np.errstate(invalid="ignore"):
        all_clear = np.where(avail, in_band, True).all(axis=0)
    mask = (
        all_clear
        & (avail.sum(axis=0) >= MIN_NEIGHBOURS)
        & np.isfinite(pu[target])
        & (cs[target] >= CS_MIN)
        & (cs[target] <= CS_MAX_B)
    )
    cols: dict[str, np.ndarray | pl.Series] = {"time": t}
    cols["ya"] = np.where(np.isfinite(yld[target]) & (cs[target] > CS_MIN), yld[target], np.nan)
    for k, s in enumerate(others):
        ratio = pu[target] / pu[s]
        cols[f"r_{k}"] = np.where(mask & avail[k], ratio, np.nan)
    frame = pl.DataFrame(cols).with_columns(pl.col("ya", "r_0").fill_nan(None))
    for k in range(1, len(others)):
        frame = frame.with_columns(pl.col(f"r_{k}").fill_nan(None))
    return frame.with_columns(
        month=pl.col("time").dt.truncate("1mo"),
        day=pl.col("time").dt.date(),
    )


def monthly_rel(frame: pl.DataFrame, *, factor: np.ndarray, method: str) -> pl.DataFrame:
    """Return one row per month with the relative yield index (`t`, `rel`) for one method."""
    f = pl.Series("f", factor)
    r_cols = [c for c in frame.columns if c.startswith("r_")]
    f_frame = frame.with_columns(f)
    if method == "A":
        monthly = (
            f_frame.with_columns(y=pl.col("ya") * pl.col("f"))
            .group_by("month")
            .agg(pl.col("y").quantile(0.9), pl.col("y").count().alias("n"), pl.col("f").mean())
            .filter(pl.col("n") >= MIN_MONTH_ROWS)
            .sort("month")
            .with_columns(cal=pl.col("month").dt.month())
            .with_columns(rel=pl.col("y") / pl.col("y").mean().over("cal"))
        )
    else:
        agg = f_frame.group_by("month").agg(
            [(pl.col(c) * pl.col("f")).median().alias(f"m_{c}") for c in r_cols]
            + [pl.col(c).count().alias(f"n_{c}") for c in r_cols]
            + [pl.col("f").mean()]
        )
        rels = []
        for c in r_cols:
            sub = (
                agg.filter(pl.col(f"n_{c}") >= MIN_MONTH_ROWS_B)
                .select("month", pl.col(f"m_{c}").alias("m"))
                .with_columns(cal=pl.col("month").dt.month())
                .with_columns(rel=pl.col("m") / pl.col("m").mean().over("cal"))
                .select("month", "rel")
            )
            rels.append(sub)
        monthly = (
            pl.concat(rels)
            .group_by("month")
            .agg(pl.col("rel").mean())
            .join(agg.select("month", "f"), on="month")
            .sort("month")
        )
    monthly = monthly.filter(pl.col("rel").is_between(*REL_RANGE))
    return monthly.with_columns(t=year_fraction(pl.col("month")))


def drift_estimate(frame: pl.DataFrame, *, factor: np.ndarray, method: str) -> dict[str, float]:
    """Return the fitted trend with its interval, and the true slope of the injected multiplier."""
    m = monthly_rel(frame, factor=factor, method=method)
    t, rel = m["t"].to_numpy(), m["rel"].to_numpy()
    slope, lo, hi = fit_trend(t=t, rel=rel)
    true = 100 * np.polyfit(t, m["f"].to_numpy(), 1)[0]
    return {"trend": slope, "lo": lo, "hi": hi, "true": float(true), "months": float(len(t))}


def daily_frame(frame: pl.DataFrame, *, factor: np.ndarray, method: str) -> pl.DataFrame:
    """Return per-day log index rows: `group`, `monthyear`, `y` (log index) and `logf`."""
    f_frame = frame.with_columns(pl.Series("f", factor)).with_columns(
        monthyear=pl.col("day").dt.truncate("1mo")
    )
    if method == "A":
        daily = (
            f_frame.with_columns(y=pl.col("ya") * pl.col("f"))
            .group_by("day", "monthyear")
            .agg(pl.col("y").quantile(0.9), pl.col("y").count().alias("n"), pl.col("f").mean())
            .filter(pl.col("n") >= MIN_DAY_ROWS_A)
            .with_columns(group=pl.lit("A"))
        )
        return daily.select(
            "day", "monthyear", "group", pl.col("y").log(), pl.col("f").log().alias("logf")
        ).filter(pl.col("y").is_finite())
    r_cols = [c for c in frame.columns if c.startswith("r_")]
    parts = [
        f_frame.with_columns(y=pl.col(c) * pl.col("f"))
        .group_by("day", "monthyear")
        .agg(pl.col("y").median(), pl.col("y").count().alias("n"), pl.col("f").mean())
        .filter(pl.col("n") >= MIN_DAY_ROWS_B)
        .select(
            "day",
            "monthyear",
            pl.lit(c).alias("group"),
            pl.col("y").log(),
            pl.col("f").log().alias("logf"),
        )
        for c in r_cols
    ]
    return pl.concat(parts).filter(pl.col("y").is_finite())


def soiling_beta(
    daily: pl.DataFrame, *, s_by_day: pl.DataFrame, column: str
) -> tuple[float, float, float]:
    """Return the within-month slope of `column` on days since reset, as percent per day, with a 90% interval.

    Each pair and calendar month has its own level. The interval resamples calendar months.
    """
    d = (
        daily.filter(pl.col("day") <= ERA5_LAST_DAY)
        .join(s_by_day, on="day")
        .with_columns(
            yd=pl.col(column) - pl.col(column).mean().over("group", "monthyear"),
            sd=pl.col("s") - pl.col("s").mean().over("group", "monthyear"),
        )
    )
    per_month = (
        d.group_by("monthyear")
        .agg(
            (pl.col("yd") * pl.col("sd")).sum().alias("sxy"), (pl.col("sd") ** 2).sum().alias("sxx")
        )
        .sort("monthyear")
    )
    sxy, sxx = per_month["sxy"].to_numpy(), per_month["sxx"].to_numpy()
    beta = 100 * sxy.sum() / sxx.sum()
    rng = np.random.default_rng(1)
    draws = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(sxy), len(sxy))
        draws.append(100 * sxy[i].sum() / sxx[i].sum())
    return float(beta), float(np.percentile(draws, 5)), float(np.percentile(draws, 95))


def main() -> None:  # noqa: D103
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    wide, info = load_farms()
    sites = sorted(info)
    time = wide["time"]
    day_all = time.dt.date()
    days_unique = day_all.unique().sort()
    drift_rows: list[dict] = []
    soil_rows: list[dict] = []
    baseline_rows: list[dict] = []
    for target in sites:
        print("farm", target, flush=True)
        frame = build_target_frame(wide=wide, target=target, sites=sites)
        keep = np.isfinite(wide[f"pu_{target}"].to_numpy())
        first, last = time.filter(pl.Series(keep)).min(), time.filter(pl.Series(keep)).max()
        span_days = (last - first).total_seconds() / 86400  # ty: ignore[unsupported-operator]
        years = span_days / 365.25
        one = np.ones(frame.height)
        reset = daily_rain_reset(lat=info[target]["lat"], lon=info[target]["lon"], days=days_unique)
        rng = np.random.default_rng(1000 + ord(target))
        starts = first + (last - first) * rng.uniform(0.0, 0.25, N_STARTS)  # ty: ignore[unsupported-operator]
        t_years = ((time - first).dt.total_seconds() / (365.25 * 86400)).to_numpy()
        base = {m: drift_estimate(frame, factor=one, method=m) for m in "AB"}
        for m, b in base.items():
            baseline_rows.append(
                {
                    "farm": target,
                    "method": m,
                    "years": round(years, 2),
                    "months": int(b["months"]),
                    "trend": b["trend"],
                    "lo": b["lo"],
                    "hi": b["hi"],
                }
            )
        for start_i, start in enumerate(starts):
            after = (time >= start).to_numpy()
            since = np.where(
                after, t_years - ((start - first).total_seconds() / (365.25 * 86400)), 0.0
            )
            for rate in DRIFT_RATES:
                factor = 1.0 - rate / 100.0 * since
                for m in "AB":
                    est = drift_estimate(frame, factor=factor, method=m)
                    drift_rows.append(
                        {
                            "farm": target,
                            "method": m,
                            "rate": rate,
                            "start": start_i,
                            "recovery": (est["trend"] - base[m]["trend"]) / est["true"],
                            "ci_excludes_zero": bool(est["lo"] > 0 or est["hi"] < 0),
                            "trend": est["trend"],
                            "lo": est["lo"],
                            "hi": est["hi"],
                            "true": est["true"],
                            "base_trend": base[m]["trend"],
                            "base_lo": base[m]["lo"],
                            "base_hi": base[m]["hi"],
                        }
                    )
            # Soiling: days since reset, from the start day, over the unique days.
            s = np.zeros(len(days_unique))
            start_day = start.date()
            current = 0
            for k, day in enumerate(days_unique.to_list()):
                if day < start_day:
                    s[k] = 0
                    continue
                current = 0 if reset[k] else current + 1
                s[k] = current
            s_by_day = pl.DataFrame({"day": days_unique, "s": s})
            s_map = dict(zip(days_unique.to_list(), s, strict=True))
            f_half = 1.0 - SOILING_PER_DAY * np.array([s_map[d] for d in day_all.to_list()])
            for m in "AB":
                d0 = daily_frame(frame, factor=one, method=m)
                d1 = daily_frame(frame, factor=f_half, method=m)
                b0 = soiling_beta(d0, s_by_day=s_by_day, column="y")
                b1 = soiling_beta(d1, s_by_day=s_by_day, column="y")
                truth = soiling_beta(d1, s_by_day=s_by_day, column="logf")
                soil_rows.append(
                    {
                        "farm": target,
                        "method": m,
                        "start": start_i,
                        "recovery": (b1[0] - b0[0]) / truth[0],
                        "ci_excludes_zero": bool(b1[1] > 0 or b1[2] < 0),
                        "beta": b1[0],
                        "lo": b1[1],
                        "hi": b1[2],
                        "truth": truth[0],
                        "base_beta": b0[0],
                        "base_lo": b0[1],
                        "base_hi": b0[2],
                    }
                )
    pl.DataFrame(baseline_rows).write_csv(DATA_DIR / "baseline_trends.csv")
    pl.DataFrame(drift_rows).write_csv(DATA_DIR / "drift_injection.csv")
    pl.DataFrame(soil_rows).write_csv(DATA_DIR / "soiling_injection.csv")
    print(pl.DataFrame(baseline_rows))


if __name__ == "__main__":
    main()
