"""Test whether the flat top of metered solar farms C to F is an export cap or panel geometry.

Reads the cleaned series that experiment 00179 wrote, computes each farm's flat level (90th
percentile of power over capacity at high clear-sky irradiance) by month, by morning or afternoon,
and by half of the years, and writes private tables under `data/ml-experiments/00183-*/`. Farms
appear only as labels A to F.
"""

from datetime import timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pvlib
from studies.pv_dataset import pv_sites
from studies.pv_physics import SunAndSky, plane_of_array_w_m2
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00183-metered-farm-flat-top-cause"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
CLEANED: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00179-metered-solar-farm-exploration" / "cleaned_solar_farms.parquet"
)
FARMS: Final[tuple[str, ...]] = ("C", "D", "E", "F")
FLEET_TILT_DEG: Final[float] = 35.0
FLEET_AZIMUTH_DEG: Final[float] = 180.0
"""The generic south-facing plant of `studies.dp_features.FLEET`."""
HIGH_GHI_W_M2: Final[float] = 650.0
"""Pre-registered: the clear-sky horizontal irradiance above which the flat top is reached."""
SEASON_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
MIN_HALF_HOURS: Final[int] = 30
CONSTANT_TOLERANCE: Final[float] = 0.03
LEVEL_QUANTILE: Final[float] = 0.90


def clear_sky_poa(*, stamps: pl.Series, latitude: float, longitude: float) -> np.ndarray:
    """Return clear-sky plane-of-array irradiance of the fleet plant at period-ending stamps."""
    mid = stamps - timedelta(minutes=15)
    location = pvlib.location.Location(latitude, longitude)
    index = mid.dt.convert_time_zone("UTC").to_numpy()
    import pandas as pd  # noqa: TID251  (pvlib's API takes a pandas index)

    times = pd.DatetimeIndex(index, tz="UTC")
    sky = location.get_clearsky(times, model="ineichen")
    position = location.get_solarposition(times)
    sun = SunAndSky(
        half_hour_end_time=stamps.to_numpy(),
        ghi_w_m2=sky["ghi"].to_numpy(),
        dni_w_m2=sky["dni"].to_numpy(),
        dhi_w_m2=sky["dhi"].to_numpy(),
        zenith_deg=position["apparent_zenith"].to_numpy(),
        sun_azimuth_deg=position["azimuth"].to_numpy(),
    )
    return plane_of_array_w_m2(
        sky=sun, tilt_deg=FLEET_TILT_DEG, azimuth_deg=FLEET_AZIMUTH_DEG, tracker=False
    )


def spread(levels: pl.Series) -> float:
    """Return the range of the levels over their mean."""
    return float((levels.max() - levels.min()) / levels.mean())  # ty: ignore[unsupported-operator]


def main() -> None:  # noqa: D103
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sites = pv_sites().filter(pl.col("site").is_in(FARMS))
    cleaned = pl.read_parquet(CLEANED).filter(pl.col("drop_reason").is_null())
    parts = []
    for row in sites.sort("site").iter_rows(named=True):
        p = cleaned.filter(pl.col("time_series_id") == row["time_series_id"]).sort("time")
        poa = clear_sky_poa(stamps=p["time"], latitude=row["latitude"], longitude=row["longitude"])
        position = pvlib.solarposition.get_solarposition(
            time=(p["time"] - timedelta(minutes=15)).to_numpy(),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        parts.append(
            p.with_columns(
                site=pl.lit(row["site"]),
                ratio=pl.col("power") / row["effective_capacity_mw"],
                poa_cs=pl.Series(poa),
                sun_az=pl.Series(position["azimuth"].to_numpy()),
            )
        )
    d = (
        pl.concat(parts)
        .with_columns(
            month=pl.col("time").dt.month(),
            year=pl.col("time").dt.year(),
            half=pl.when(pl.col("time").dt.year() % 2 == 0)
            .then(pl.lit("even years"))
            .otherwise(pl.lit("odd years")),
            side=pl.when(pl.col("sun_az") < 180)
            .then(pl.lit("morning"))
            .otherwise(pl.lit("afternoon")),
        )
        .filter(pl.col("month").is_in(SEASON_MONTHS))
    )
    hi = d.filter(pl.col("clear_sky_ghi_w_m2") > HIGH_GHI_W_M2)

    def level_table(frame: pl.DataFrame, *by: str) -> pl.DataFrame:
        return (
            frame.group_by("site", *by)
            .agg(
                pl.col("ratio").quantile(LEVEL_QUANTILE).alias("level"),
                pl.col("ratio").quantile(0.99).alias("p99"),
                pl.len().alias("n"),
            )
            .filter(pl.col("n") >= MIN_HALF_HOURS)
            .sort("site", *by)
        )

    monthly = level_table(hi, "month")
    monthly_half = level_table(hi, "half", "month")
    by_side = level_table(hi, "side")
    by_year = level_table(hi, "year", "month")
    monthly.write_csv(DATA_DIR / "monthly_levels.csv")
    monthly_half.write_csv(DATA_DIR / "monthly_levels_by_half.csv")
    by_side.write_csv(DATA_DIR / "levels_by_side.csv")
    by_year.write_csv(DATA_DIR / "levels_by_year_month.csv")

    sensitivity = []
    long_levels = []
    for quantile in (0.90, 0.95, 0.99):
        levels = (
            hi.group_by("site", "month")
            .agg(pl.col("ratio").quantile(quantile).alias("level"), pl.len().alias("n"))
            .filter(pl.col("n") >= MIN_HALF_HOURS)
        )
        long_levels.append(levels.with_columns(quantile=pl.lit(quantile)))
        for site in FARMS:
            m = levels.filter(pl.col("site") == site)
            sensitivity.append(
                {"site": site, "quantile": quantile, "spread": round(spread(m["level"]), 4)}
            )
    pl.concat(long_levels).sort("site", "quantile", "month").write_csv(
        DATA_DIR / "monthly_levels_by_quantile.csv"
    )
    pl.DataFrame(sensitivity).write_csv(DATA_DIR / "quantile_sensitivity.csv")
    print(pl.DataFrame(sensitivity).pivot(on="quantile", index="site", values="spread"))
    print(
        hi.group_by("site", "month")
        .agg(pl.col("ratio").quantile(0.99).alias("p99"))
        .pivot(on="month", index="site", values="p99")
        .sort("site")
    )

    rows = []
    for site in FARMS:
        m = monthly.filter(pl.col("site") == site)
        entry = {
            "site": site,
            "months": ",".join(map(str, m["month"].to_list())),
            "mean_level": round(float(m["level"].mean()), 3),  # ty: ignore[invalid-argument-type]
            "spread_all": round(spread(m["level"]), 4),
        }
        for half in ("even years", "odd years"):
            h = monthly_half.filter((pl.col("site") == site) & (pl.col("half") == half))
            entry[f"spread_{half}"] = round(spread(h["level"]), 4) if h.height >= 3 else None
            entry[f"months_{half}"] = ",".join(map(str, h["month"].to_list()))
        s = by_side.filter(pl.col("site") == site)
        if s.height == 2:
            entry["morning_over_afternoon"] = round(
                float(s.filter(pl.col("side") == "morning")["level"][0])
                / float(s.filter(pl.col("side") == "afternoon")["level"][0]),
                4,
            )
        rows.append(entry)
    summary = pl.DataFrame(rows, strict=False)
    summary.write_csv(DATA_DIR / "summary.csv")
    with pl.Config(tbl_rows=50, tbl_cols=20, fmt_table_cell_list_len=8, tbl_width_chars=250):
        print(monthly.pivot(on="month", index="site", values="level").sort("site"))
        print(monthly.pivot(on="month", index="site", values="n").sort("site"))
        print(summary)
        print(by_side)


if __name__ == "__main__":
    main()
