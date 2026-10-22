"""Explore the 6 metered solar farms: validation checks, clipping, steps, drift, cleaned series.

Writes private tables and the cleaned series under `data/ml-experiments/00179-*/`, and anonymised
figures and tables under `docs/ml-experiments/00179-*/`. Farms appear only as labels A to F.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pvlib
from studies.power import scan_power
from studies.pv_dataset import pv_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00179-metered-solar-farm-exploration"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / SLUG
STUCK_MIN_RUN: Final[int] = 6
"""Consecutive identical positive daylight readings that count as a stuck meter."""
DAYLIGHT_CS_W_M2: Final[float] = 20.0
HIGH_CS_W_M2: Final[float] = 500.0
CLIP_RATIO: Final[float] = 1.02
CLIP_SLOPE_FRACTION: Final[float] = 0.10
POST_HOC_SLOPE_FRACTION: Final[float] = 0.25
COMMISSIONED_FROM: Final[dict[str, datetime]] = {
    "E": datetime(2024, 10, 6, tzinfo=UTC),  # `studies.commissioning.SETTLED_OUTPUT_FROM`
    "D": datetime(2019, 10, 1, tzinfo=UTC),  # first partial month reads 0.19 of later yield
}


def clear_sky(*, stamps: pl.Series, latitude: float, longitude: float) -> np.ndarray:
    """Return clear-sky GHI (Ineichen) at the mid-point of each period-ending stamp."""
    mid = pd_index(stamps - timedelta(minutes=15))
    location = pvlib.location.Location(latitude, longitude)
    return location.get_clearsky(mid, model="ineichen")["ghi"].to_numpy()


def pd_index(stamps: pl.Series):  # noqa: ANN201, D103
    import pandas as pd  # noqa: TID251  (pvlib's clear-sky API takes a pandas index)

    return pd.DatetimeIndex(stamps.dt.convert_time_zone("UTC").to_numpy(), tz="UTC")


def main() -> None:  # noqa: D103, PLR0915
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    sites = pv_sites().sort("site")
    power_all = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .collect()
        .sort("time_series_id", "time")
    )
    drift_rows, cleaned_parts, checks, clip_rows, bin_rows, month_rows, steps = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for row in sites.iter_rows(named=True):
        site, tsid, cap = row["site"], row["time_series_id"], float(row["effective_capacity_mw"])
        p = power_all.filter(pl.col("time_series_id") == tsid).select("time", "power")
        full = pl.datetime_range(  # ty: ignore[no-matching-overload]
            p["time"].min(), p["time"].max(), interval="30m", time_zone="UTC", eager=True
        )
        n_dup = p.height - p["time"].n_unique()
        p = p.unique("time", keep="first").sort("time")
        n_missing = full.len() - p.height
        p = p.with_columns(
            ghi_cs=pl.Series(
                clear_sky(stamps=p["time"], latitude=row["latitude"], longitude=row["longitude"])
            ),
        ).with_columns(ratio=pl.col("power") / cap)
        # Timestamp convention: lag of best correlation between power and clear-sky GHI.
        corr = {}
        for lag in range(-3, 4):
            s = p.with_columns(sh=pl.col("ghi_cs").shift(lag)).drop_nulls("sh")
            corr[lag] = float(np.corrcoef(s["power"].to_numpy(), s["sh"].to_numpy())[0, 1])
        best_lag = max(corr, key=corr.get)  # ty: ignore[no-matching-overload]
        # Stuck runs: identical consecutive positive daylight values.
        p = p.with_columns(
            run_id=(pl.col("power") != pl.col("power").shift()).cum_sum()
        ).with_columns(run_len=pl.len().over("run_id"))
        stuck = (
            (pl.col("run_len") >= STUCK_MIN_RUN)
            & (pl.col("power") > 0.01 * cap)
            & (pl.col("ghi_cs") > DAYLIGHT_CS_W_M2)
        )
        zero_in_sun = (pl.col("power") <= 0) & (pl.col("ghi_cs") > HIGH_CS_W_M2)
        negative = pl.col("power") < -0.01 * cap
        night_pos = (pl.col("ghi_cs") <= 0) & (pl.col("power") > 0.02 * cap)
        over_cap = pl.col("power") > 1.5 * cap
        p = p.with_columns(
            flag=pl.when(
                pl.col("time") < COMMISSIONED_FROM.get(site, datetime(2000, 1, 1, tzinfo=UTC))
            )
            .then(pl.lit("commissioning"))
            .when(zero_in_sun)
            .then(pl.lit("zero_in_clear_sky"))
            .when(stuck)
            .then(pl.lit("stuck"))
            .when(negative)
            .then(pl.lit("negative"))
            .when(night_pos)
            .then(pl.lit("night_positive"))
            .when(over_cap)
            .then(pl.lit("over_capacity"))
            .otherwise(pl.lit(None, dtype=pl.String)),
            zero_in_clear_sun=zero_in_sun,
        )
        n_zero = int(p["zero_in_clear_sun"].sum())
        checks.append(
            {
                "site": site,
                "capacity_mw": cap,
                "rows": p.height,
                "first": p["time"].min().date(),  # ty: ignore[unresolved-attribute]
                "last": p["time"].max().date(),  # ty: ignore[unresolved-attribute]
                "missing_half_hours": n_missing,
                "missing_pct": round(100 * n_missing / full.len(), 2),
                "duplicate_timestamps": n_dup,
                "stuck_rows": int((p["flag"] == "stuck").sum()),
                "negative_rows": int((p["flag"] == "negative").sum()),
                "night_positive_rows": int((p["flag"] == "night_positive").sum()),
                "over_1p5_capacity_rows": int((p["flag"] == "over_capacity").sum()),
                "zero_when_clear_sky_over_500": n_zero,
                "best_lag_half_hours": best_lag,
                "max_over_capacity": round(float(p["ratio"].max()), 3),  # ty: ignore[invalid-argument-type]
            }
        )
        # Clipping: upper envelope of power/capacity against clear-sky GHI.
        day = p.filter((pl.col("ghi_cs") > DAYLIGHT_CS_W_M2) & pl.col("flag").is_null())
        edges = np.linspace(0, float(day["ghi_cs"].max()) + 1e-6, 21)  # ty: ignore[invalid-argument-type]
        day = day.with_columns(b=pl.Series(np.digitize(day["ghi_cs"].to_numpy(), edges[1:-1])))
        env = (
            day.group_by("b")
            .agg(
                pl.col("ghi_cs").mean().alias("ghi_cs"),
                pl.col("ratio").median().alias("p50"),
                pl.col("ratio").quantile(0.95).alias("p95"),
                pl.col("ratio").quantile(0.99).alias("p99"),
                pl.len().alias("n"),
            )
            .filter(pl.col("n") >= 30)
            .sort("b")
            .with_columns(site=pl.lit(site))
        )
        bin_rows.append(env)
        e = env.sort("ghi_cs")
        top = e.tail(3)
        slope = lambda f: float(np.polyfit(f["ghi_cs"].to_numpy(), f["p95"].to_numpy(), 1)[0])  # noqa: E731
        top_slope, mid_slope = slope(top), slope(e.slice(e.height // 4, 4))
        top_ratio = float(top["p99"].mean() / top["p95"].mean())  # ty: ignore[unsupported-operator]
        clipping = top_ratio <= CLIP_RATIO and top_slope <= CLIP_SLOPE_FRACTION * mid_slope
        clip_rows.append(
            {
                "site": site,
                "p99_over_p95_top_bins": round(top_ratio, 3),
                "p95_slope_top_over_mid": round(top_slope / mid_slope, 3),
                "p99_peak_over_capacity": round(float(e["p99"].max()), 3),  # ty: ignore[invalid-argument-type]
                "p95_peak_over_capacity": round(float(e["p95"].max()), 3),  # ty: ignore[invalid-argument-type]
                "clipping": clipping,
                "clipping_post_hoc": top_slope <= POST_HOC_SLOPE_FRACTION * mid_slope,
            }
        )
        # Monthly yield index: p90 of power / clear-sky GHI where clear-sky GHI is high.
        hi = day.filter(pl.col("ghi_cs") > 300).with_columns(
            yield_idx=pl.col("power") / pl.col("ghi_cs") * 1000 / cap
        )
        monthly = (
            hi.with_columns(month=pl.col("time").dt.truncate("1mo"))
            .group_by("month")
            .agg(pl.col("yield_idx").quantile(0.90).alias("yield_p90"), pl.len().alias("n"))
            .filter(pl.col("n") >= 40)
            .sort("month")
            .with_columns(site=pl.lit(site))
        )
        month_rows.append(monthly)
        mm = monthly.with_columns(
            m=pl.col("month").dt.month(),
            t=pl.col("month").dt.year() + (pl.col("month").dt.month() - 1) / 12,
        )
        mm = mm.with_columns(rel=pl.col("yield_p90") / pl.col("yield_p90").mean().over("m"))
        drift_rows.append(
            {
                "site": site,
                "months": mm.height,
                "drift_pct_per_year": round(
                    100 * float(np.polyfit(mm["t"].to_numpy(), mm["rel"].to_numpy(), 1)[0]), 2
                ),
            }
        )
        # Candidate outage runs: days whose peak power is under 5% of capacity in months whose
        # median-day peak exceeds 30% of capacity, in runs of 3 or more days.
        daily = (
            p.filter(
                pl.col("flag").is_null() | pl.col("flag").is_in(["stuck", "zero_in_clear_sky"])
            )
            .group_by(pl.col("time").dt.truncate("1d").alias("day"))
            .agg(pl.col("power").max().alias("peak"), pl.len().alias("n"))
            .sort("day")
            .with_columns(
                low=(pl.col("peak") < 0.05 * cap)
                & (pl.col("n") >= 40)
                & (pl.col("day").dt.month().is_between(4, 9))
            )
            .with_columns(g=(pl.col("low") != pl.col("low").shift()).cum_sum())
        )
        for _, run in daily.filter(pl.col("low")).group_by("g"):
            if run.height >= 3:
                steps.append({"site": site, "start": run["day"].min().date(), "days": run.height})  # ty: ignore[unresolved-attribute]
        # Cleaned series (private): flagged rows keep their value and the reason.
        cleaned_parts.append(
            p.select(
                pl.lit(tsid).cast(pl.Int32).alias("time_series_id"),
                "time",
                "power",
                pl.col("ghi_cs").alias("clear_sky_ghi_w_m2"),
                pl.col("flag").alias("drop_reason"),
            )
        )
    cleaned = pl.concat(cleaned_parts)
    cleaned.write_parquet(DATA_DIR / "cleaned_solar_farms.parquet")
    checks_df, clip_df = pl.DataFrame(checks), pl.DataFrame(clip_rows)
    steps_df = (
        pl.DataFrame(steps)
        if steps
        else pl.DataFrame(schema={"site": pl.String, "start": pl.Date, "days": pl.Int64})
    )
    pl.concat(bin_rows).write_csv(DOCS_DIR / "envelope_bins.csv")
    pl.concat(month_rows).write_csv(DOCS_DIR / "monthly_yield.csv")
    checks_df.write_csv(DOCS_DIR / "validation_checks.csv")
    clip_df.write_csv(DOCS_DIR / "clipping_tests.csv")
    steps_df.write_csv(DOCS_DIR / "candidate_outages.csv")
    pl.Config.set_tbl_rows(50)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    print(checks_df)
    print(clip_df)
    print("farms with clipping:", int(clip_df["clipping"].sum()), "of", clip_df.height)
    print(steps_df)
    drift_df = pl.DataFrame(drift_rows)
    drift_df.write_csv(DOCS_DIR / "drift.csv")
    print(drift_df)
    print(
        cleaned.group_by("time_series_id", "drop_reason")
        .len()
        .sort("time_series_id", "drop_reason")
    )
    print(
        pl.concat(month_rows)
        .group_by("site")
        .agg(
            pl.col("yield_p90").first().alias("first"),
            pl.col("yield_p90").last().alias("last"),
            pl.col("yield_p90").min().alias("min"),
            pl.col("yield_p90").max().alias("max"),
        )
        .sort("site")
    )


if __name__ == "__main__":
    main()
