"""Beam and diffuse sensitivity row of experiment 129.

The unit PV shape of every other script splits CAMS global irradiance into beam and diffuse light
with the Erbs correlation. The raw CAMS files that the demand-point download wrote hold CAMS' own
horizontal beam (`bhi`) and diffuse (`dhi`) irradiance at the five decision Primaries, so no new
download is needed. This script rebuilds the shape from those two columns and compares, for arm 2,
the matched-farm recovery and the latest-year capacity with the Erbs shape. Hours are spread over
half-hours by the cosine of the zenith, as `half_hour_sun_and_sky` does for global irradiance.

    E00129_THROUGH_JULY=1 uv run --no-sync python ml-experiments/00129-*/e00129_beam.py
"""

import argparse
from dataclasses import replace
from typing import Final

import e00129_arms as arms
import e00129_common as common
import numpy as np
import polars as pl
from studies.dp_features import FLEET, FeatureCache
from studies.pv_physics import half_hour_sun_and_sky, plant_power_mw
from studies.sources import CAMS_PRODUCT_DIR

RAW_DIR: Final = CAMS_PRODUCT_DIR / "demand_points" / "raw"
COLUMNS: Final[tuple[str, ...]] = (
    "observation_period",
    "toa",
    "clear_ghi",
    "clear_bhi",
    "clear_dhi",
    "clear_bni",
    "ghi",
    "bhi",
    "dhi",
    "bni",
    "reliability",
)
MIN_COS_ZENITH: Final[float] = 0.05
"""Below this the beam on a horizontal plane is too small to divide into a normal-incidence beam."""
MAX_DNI_W_M2: Final[float] = 1400.0
"""A guard against a division by a small cosine; above the solar constant."""
LATEST_COMPLETE_YEAR: Final[int] = common.COMPLETE_SEASON_YEARS[-1]


def hourly_beam_diffuse(*, site: str) -> pl.DataFrame:
    """Return the site's hourly `time` (end of the hour), `ghi_w_m2`, `bhi_w_m2`, and `dhi_w_m2`."""
    frames = []
    for path in sorted(RAW_DIR.glob(f"cams_{site}_*.csv")):
        rows = [
            line.rstrip("\n").split(";")
            for line in path.read_text().splitlines()
            if line and not line.startswith("#")
        ]
        frames.append(pl.DataFrame(rows, schema=list(COLUMNS), orient="row"))
    return (
        pl.concat(frames)
        .select(
            time=pl.col("observation_period")
            .str.split("/")
            .list.get(1)
            .str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", time_zone="UTC"),
            ghi_w_m2=pl.col("ghi").cast(pl.Float64),
            bhi_w_m2=pl.col("bhi").cast(pl.Float64),
            dhi_w_m2=pl.col("dhi").cast(pl.Float64),
        )
        .unique("time")
        .sort("time")
    )


def beam_diffuse_cache(*, cache: FeatureCache, site: str) -> FeatureCache:
    """Return the cache with the site's PV shape built from CAMS beam and diffuse irradiance."""
    table = cache.sites[site].table
    hourly = hourly_beam_diffuse(site=site)
    sky = half_hour_sun_and_sky(hourly=hourly, latitude=table.latitude, longitude=table.longitude)
    cos_zenith = np.clip(np.cos(np.radians(sky.zenith_deg)), 0.0, None)
    hour_mean_cos = np.repeat(cos_zenith.reshape(-1, 2).mean(axis=1), 2)
    share = np.divide(
        cos_zenith, hour_mean_cos, out=np.zeros_like(cos_zenith), where=hour_mean_cos > 0
    )
    beam_horizontal = np.repeat(hourly["bhi_w_m2"].to_numpy(), 2) * share
    diffuse = np.repeat(hourly["dhi_w_m2"].to_numpy(), 2) * share
    dni = np.divide(
        beam_horizontal,
        cos_zenith,
        out=np.zeros_like(cos_zenith),
        where=cos_zenith > MIN_COS_ZENITH,
    )
    new_sky = replace(
        sky,
        dni_w_m2=np.clip(dni, 0.0, MAX_DNI_W_M2),
        dhi_w_m2=np.clip(diffuse, 0.0, None),
    )
    shape = pl.DataFrame(
        {
            "time": pl.Series(new_sky.half_hour_end_time).dt.replace_time_zone("UTC"),
            "pv_unit_shape": plant_power_mw(sky=new_sky, parameters=FLEET),
        }
    )
    joined = pl.DataFrame({"time": cache.times_series()}).join(
        shape, on="time", how="left", validate="1:1", maintain_order="left"
    )
    features = replace(
        cache.sites[site], pv_unit_shape=joined["pv_unit_shape"].to_numpy().astype(np.float64)
    )
    return replace(cache, sites={**cache.sites, site: features})


def matched_recovery_and_capacity(*, cache: FeatureCache, site: str, device: str) -> dict:
    """Return arm 2's matched-farm recovery in the latest complete year and its latest capacity."""
    pinned = common.pinned_points(cache=cache, site=site)
    base = arms.fitted_capacity(arm="arm2", cache=cache, site=site, device=device)
    recoveries = []
    for farm in common.FARMS:
        injected = common.inject(
            cache=cache, site=site, output=pinned * common.farm_norm(cache=cache, farm=farm)
        )
        capacity = arms.fitted_capacity(arm="arm2", cache=injected, site=site, device=device)
        recoveries.append((capacity[LATEST_COMPLETE_YEAR] - base[LATEST_COMPLETE_YEAR]) / pinned)
    recovery = float(np.median(recoveries))
    scale = cache.sites[site].table.effective_capacity_mw / 100.0
    latest = common.SEASON_YEARS[-1]
    return {
        "recovery": recovery,
        "raw_latest_mw": base[latest] * scale,
        "estimate_latest_mw": base[latest] / recovery * scale,
    }


def main() -> None:
    """Write `work/beam_diffuse.parquet`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    cache = common.load_cache()
    rows = []
    for site in common.DECISION_SITES:
        for shape, source in (
            ("erbs_split", cache),
            ("cams_beam_diffuse", beam_diffuse_cache(cache=cache, site=site)),
        ):
            rows.append(
                {
                    "site": site,
                    "shape": shape,
                    **matched_recovery_and_capacity(
                        cache=source, site=site, device=arguments.device
                    ),
                }
            )
        print(site, "done", flush=True)
    table = pl.DataFrame(rows)
    table.write_parquet(common.WORK_DIR / "beam_diffuse_to_july.parquet")
    with pl.Config(tbl_rows=20, float_precision=3):
        print(table)


if __name__ == "__main__":
    main()
