"""Repeat experiment 218's fold-1 plants with the ECMWF 51-member ensemble mean at 12 to 36 h leads.

The ensemble store holds one run per day (00:00 UTC) with 3-hourly stamps. Downward shortwave is a
mean over the 3 hours that end at the stamp, so it is spread over the six half-hours by the cosine
of the zenith, then split into beam and diffuse with the Erbs correlation. Air temperature is an
instant, interpolated linearly between stamps. A half-hour ending `T` takes the run whose 12 to 36
hour window contains `T`, so every half-hour has exactly one run.

The ensemble mean exists at forecast time, unlike CAMS irradiance (about 2 days late) and ERA5
temperature (about 5 days late) that the main fit used.
"""

from datetime import timedelta
from pathlib import Path
from typing import Final

import e00218_common as common
import e00218_fit as fit
import h3
import numpy as np
import pandas as pd  # noqa: TID251  (pvlib's API takes a pandas index)
import polars as pl
import pvlib
from studies.pv_dataset import pv_sites
from studies.sources import REPO_DATA_DIR

NWP_URI: Final[str] = str(REPO_DATA_DIR / "NWP")
H3_RESOLUTION: Final[int] = 5
PERIOD_HOURS: Final[int] = 3
MIN_LEAD_HOURS: Final[int] = 12
MAX_LEAD_HOURS: Final[int] = 36
ENS_ROWS_PATH: Final[Path] = common.DATA_DIR / "ens_rows_fold1.parquet"
ENS_SCORES_PATH: Final[Path] = common.DATA_DIR / "ens_scores.csv"
HALF_HOURS_PER_PERIOD: Final[int] = 6
CAMS_ARMS: Final[tuple[str, ...]] = ("arm1_fleet_shape", "arm2_per_farm", "arm4_heat_off")


def nearest_cell(*, latitude: float, longitude: float, cells: list[int]) -> int:
    """Return the store's H3 cell whose centre is nearest the farm."""
    centres = np.array([h3.cell_to_latlng(h3.int_to_str(c)) for c in cells])
    distance = (centres[:, 0] - latitude) ** 2 + (
        (centres[:, 1] - longitude) * np.cos(np.radians(latitude))
    ) ** 2
    return int(cells[int(np.argmin(distance))])


def run_init_for_period_end(*, period_end: pl.Expr) -> pl.Expr:
    """Return the 00:00 UTC init of the run whose 12 to 36 h window holds each 3-hour period end."""
    shifted = period_end - timedelta(hours=MIN_LEAD_HOURS)
    return (shifted - timedelta(microseconds=1)).dt.truncate("1d")


def ensemble_means(*, cell: int, first: pl.Series) -> pl.DataFrame:
    """Return the ensemble-mean shortwave and temperature per (init, valid_time) for one cell."""
    lazy = pl.scan_delta(NWP_URI).filter(
        (pl.col("h3_index") == cell) & pl.col("init_time").is_in(first.to_list())
    )
    return (
        lazy.group_by("init_time", "valid_time")
        .agg(
            ghi_3h=pl.col("downward_short_wave_radiation_flux_surface").mean(),
            temperature_c=pl.col("temperature_2m").mean(),
            members=pl.col("ensemble_member").n_unique(),
        )
        .collect()
    )


def farm_ens_rows(*, row: dict[str, float | str], scored_times: pl.Series) -> pl.DataFrame:
    """Build one farm's half-hourly sun and sky from the ensemble means at its scored times."""
    periods = (
        pl.DataFrame({"time": scored_times})
        .with_columns(
            period_end=(pl.col("time") - timedelta(microseconds=1)).dt.truncate("3h")
            + timedelta(hours=PERIOD_HOURS)
        )
        .with_columns(init_time=run_init_for_period_end(period_end=pl.col("period_end")))
    )
    period_ends = periods["period_end"].unique().sort()
    inits = periods["init_time"].unique()
    store_cells = (
        pl.scan_delta(NWP_URI)
        .filter(pl.col("init_time") == inits[0])
        .select("h3_index")
        .unique()
        .collect()["h3_index"]
        .to_list()
    )
    cell = nearest_cell(
        latitude=float(row["latitude"]), longitude=float(row["longitude"]), cells=store_cells
    )
    means = ensemble_means(cell=cell, first=inits)
    # Six half-hours per period, for the cosine weights.
    offsets = [timedelta(minutes=30 * k) for k in range(HALF_HOURS_PER_PERIOD)]
    grid = pl.concat(
        [
            pl.DataFrame({"period_end": period_ends}).with_columns(
                time=pl.col("period_end") - o, k=pl.lit(k)
            )
            for k, o in enumerate(offsets)
        ]
    ).sort("time")
    index = pd.DatetimeIndex((grid["time"] - timedelta(minutes=15)).to_numpy(), tz="UTC")
    location = pvlib.location.Location(row["latitude"], row["longitude"])
    position = location.get_solarposition(index)
    zenith = position["apparent_zenith"].to_numpy()
    grid = grid.with_columns(
        zenith_deg=zenith,
        sun_azimuth_deg=position["azimuth"].to_numpy(),
        cosine=np.clip(np.cos(np.radians(zenith)), 0.0, None),
    ).with_columns(mean_cosine=pl.col("cosine").mean().over("period_end"))
    grid = grid.with_columns(init_time=run_init_for_period_end(period_end=pl.col("period_end")))
    stamped = means.rename({"valid_time": "period_end", "ghi_3h": "ghi_period_mean"})
    previous = means.select(
        init_time="init_time",
        period_end=pl.col("valid_time") + timedelta(hours=PERIOD_HOURS),
        temperature_previous="temperature_c",
    )
    joined = (
        grid.join(
            stamped.select("init_time", "period_end", "ghi_period_mean", "temperature_c"),
            on=["init_time", "period_end"],
            how="inner",
        )
        .join(previous, on=["init_time", "period_end"], how="left")
        .with_columns(
            ghi_w_m2=pl.when(pl.col("mean_cosine") > 0)
            .then(pl.col("ghi_period_mean") * pl.col("cosine") / pl.col("mean_cosine"))
            .otherwise(0.0),
            weight=(pl.col("k").cast(pl.Float64) + 0.5) / HALF_HOURS_PER_PERIOD,
        )
        .with_columns(
            temperature_c=pl.col("temperature_c") * (1 - pl.col("weight"))
            + pl.col("temperature_previous").fill_null(pl.col("temperature_c")) * pl.col("weight")
        )
    )
    # `k` counts back from the period end (k = 0 is the half-hour ending at the stamp), so the
    # weight on the earlier stamp grows with k.
    split = pvlib.irradiance.erbs(
        ghi=joined["ghi_w_m2"].to_numpy(),
        zenith=joined["zenith_deg"].to_numpy(),
        datetime_or_doy=joined["time"].dt.ordinal_day().to_numpy(),
    )
    return joined.with_columns(
        dni_w_m2=pl.Series(np.nan_to_num(np.asarray(split["dni"]))),
        dhi_w_m2=pl.Series(np.nan_to_num(np.asarray(split["dhi"]))),
        site=pl.lit(row["site"]),
    ).select(
        "site",
        "time",
        "ghi_w_m2",
        "dni_w_m2",
        "dhi_w_m2",
        "zenith_deg",
        "sun_azimuth_deg",
        "temperature_c",
    )


def score(*, ens: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
    """Score fold-1 plants driven by the ensemble mean and by CAMS on the same half-hours."""
    parameters = pl.read_parquet(fit.PARAMETERS_PATH).filter(pl.col("fold") == 1)
    rows = []
    for site in common.FARMS:
        observed = predictions.filter(pl.col("site") == site)
        frame = observed.select("time", "output", *CAMS_ARMS).join(
            ens.filter(pl.col("site") == site).drop_nulls(), on="time", how="inner"
        )
        if frame.is_empty():
            continue
        data = fit.FarmData(
            frame=frame,
            tuning=np.zeros(frame.height, dtype=bool),
            scored=np.ones(frame.height, dtype=bool),
        )
        output = frame["output"].to_numpy().astype(np.float64)
        for arm in CAMS_ARMS:
            point = parameters.filter((pl.col("site") == site) & (pl.col("arm") == arm)).row(
                0, named=True
            )
            if arm == "arm1_fleet_shape":
                point = {**point, "scale_plain": point["scale"]}
                predicted = fit._predict_plain(data=data, point=point)
            else:
                point = {**point, "scale_clip": point["scale"]}
                predicted = fit._predict_clip(data=data, point=point)
            for source, values in (("ens_mean", predicted), ("cams", frame[arm].to_numpy())):
                rows.append(
                    {
                        "site": site,
                        "arm": arm,
                        "weather": source,
                        "n": frame.height,
                        "fraction": 1
                        - float(np.sum((output - values) ** 2)) / float(np.sum(output**2)),
                        "mae_capacity_points": 100 * float(np.mean(np.abs(output - values))),
                    }
                )
    return pl.DataFrame(rows)


def main() -> None:
    """Build the ensemble-driven sky for the fold-1 scored rows and score the plants on it."""
    predictions = pl.read_parquet(fit.PREDICTIONS_PATH).filter(
        pl.col("scored") & (pl.col("fold") == 1)
    )
    sites = pv_sites()
    if not ENS_ROWS_PATH.exists():
        parts = []
        for row in sites.sort("site").iter_rows(named=True):
            times = predictions.filter(pl.col("site") == row["site"])["time"]
            parts.append(farm_ens_rows(row=row, scored_times=times))
        pl.concat(parts).write_parquet(ENS_ROWS_PATH)
    ens = pl.read_parquet(ENS_ROWS_PATH)
    scores = score(ens=ens, predictions=predictions)
    scores.write_csv(ENS_SCORES_PATH)
    print(ens.group_by("site").len().sort("site"))
    print(scores.pivot(on="weather", index=["arm", "site"], values="fraction").sort("arm", "site"))
    print(
        scores.group_by("arm", "weather")
        .agg(
            median_fraction=pl.col("fraction").median(), mae=pl.col("mae_capacity_points").median()
        )
        .sort("arm", "weather")
    )


if __name__ == "__main__":
    main()
