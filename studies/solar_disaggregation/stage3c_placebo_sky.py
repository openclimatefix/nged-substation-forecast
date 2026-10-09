"""Stage 3c: does the cloud increment survive when the cloud information is destroyed?

The cloud increment is the share of the variance of four-hour output changes that a plant fitted to
CAMS's all-sky irradiance explains, minus the share a plant fitted to CAMS's clear-sky irradiance
explains. A positive increment could come from real cloud, or from anything else that the all-sky
series carries and the clear-sky series lacks. The placebo removes the real cloud and keeps the
rest: within each calendar month the script permutes whole days of the all-sky hourly irradiance, so
every day keeps its hour-of-day shape and the month keeps its seasonal level, but the day's weather
belongs to another day of the same month. The plant is refitted to the permuted sky with the same
starting capacity and orientation as stage 3 or stage 3b, and the placebo increment is the permuted
sky's share minus the real clear-sky share.

Each BMU uses the sky its own stage used: the mean of the three CAMS points nearest to its GSP
group's centroid for the aggregate BMUs (stage 3), and the mean of all 18 grid points at the centre
of Great Britain for the single-site solar BMUs (stage 3b).

The script also splits the real increment by season: the pairs of half-hours that the fit uses are
divided into April to September and October to March, and each subset gets its own share from the
same fitted plants.

Run: `uv run python studies/solar_disaggregation/stage3c_placebo_sky.py`. Needs the outputs of
stages 3 and 3b.
"""

from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from typing import Final

import numpy as np
import polars as pl
from inputs import (
    OUTPUT_DIR,
    REFERENCE_LATITUDE,
    REFERENCE_LONGITUDE,
    VALIDATION_BMUS,
    bmu_register,
    grid_index,
    grid_point_ids,
    hourly_cams,
    nearest_grid_points,
    output_on_grid,
    sky_at,
)
from stage3_real_aggregates import NEAREST_POINTS, START_SHARE, gsp_group_centroids
from studies.pv_fit import FitResult, fit_plant_to_changes, lagged_pairs
from studies.pv_physics import SunAndSky, plant_power_mw
from studies.pv_separation import DIFFERENCE_LAG_HALF_HOURS

AGGREGATE_BMUS: Final[tuple[str, ...]] = (
    "2__ATGPL000",
    "2__HTGPL000",
    "2__BTGPL000",
    "2__LTGPL000",
    "2__KTGPL000",
    "2__DRWED000",
    "V__HFLEX002",
    "2__BECOT003",
    "2__JAXPO000",
)
"""The aggregate BMUs whose cloud increment the placebo tests."""
PERMUTATIONS: Final[int] = 20
SEED: Final[int] = 20261009
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
"""The months of the summer half of the year; the other six months are the winter half."""
MAX_WORKERS: Final[int] = 4


def day_permutation(*, days: list[date], rng: np.random.Generator) -> dict[date, date]:
    """Return a map from each day to the day of the same calendar month whose weather it takes.

    Args:
        days: The days to permute.
        rng: The random generator.

    Returns:
        For each day, the source day: a random day of the same calendar month, each day used once.
    """
    mapping: dict[date, date] = {}
    for key in sorted({(d.year, d.month) for d in days}):
        month_days = [d for d in days if (d.year, d.month) == key]
        order = rng.permutation(len(month_days))
        mapping |= {day: month_days[i] for day, i in zip(month_days, order.tolist(), strict=True)}
    return mapping


def permute_hourly(*, hourly: pl.DataFrame, mapping: dict[date, date]) -> pl.DataFrame:
    """Give every hour the irradiance of the same hour of its source day.

    Args:
        hourly: Hourly irradiance, as `hourly_cams` returns it.
        mapping: `day_permutation`'s map from each day to its source day.

    Returns:
        The hourly irradiance with the original timestamps. An hour whose source hour is absent
        from `hourly` is dropped.
    """
    stamped = hourly.with_columns(
        day=(pl.col("time") - pl.duration(hours=1)).dt.date(),
        hour=(pl.col("time") - pl.duration(hours=1)).dt.hour(),
    )
    source = pl.DataFrame(
        {"day": list(mapping), "source_day": list(mapping.values())},
        schema={"day": pl.Date, "source_day": pl.Date},
    )
    values = stamped.select("day", "hour", "ghi_w_m2")
    return (
        stamped.select("time", "day", "hour")
        .join(source, on="day")
        .join(values, left_on=["source_day", "hour"], right_on=["day", "hour"])
        .select("time", "ghi_w_m2")
        .sort("time")
    )


def _fit_and_share(
    *, sky: SunAndSky, output: np.ndarray, ac_guess_mw: float
) -> tuple[FitResult, np.ndarray] | None:
    """Fit a plant to a sky and return the fit with the plant's power at every half-hour."""
    on_sky = output[grid_index(half_hour_end_time=sky.half_hour_end_time)]
    fit = fit_plant_to_changes(
        sky=sky,
        output_mw=on_sky,
        orientation="free",
        ac_guess_mw=ac_guess_mw,
        lag_half_hours=DIFFERENCE_LAG_HALF_HOURS,
    )
    if fit is None:
        return None
    return fit, plant_power_mw(sky=sky, parameters=fit.parameters)


def _shares(*, sky: SunAndSky, output: np.ndarray, power: np.ndarray) -> tuple[float, float, float]:
    """Return the share of change variance explained over all pairs, summer pairs, and winter pairs.

    A pair belongs to the season of its later half-hour, in UTC.
    """
    lag = DIFFERENCE_LAG_HALF_HOURS
    on_sky = output[grid_index(half_hour_end_time=sky.half_hour_end_time)]
    pairs = lagged_pairs(sky=sky, output_mw=on_sky, lag_half_hours=lag)
    change = on_sky[pairs] - on_sky[pairs - lag]
    residual = change - (power[pairs] - power[pairs - lag])
    months = sky.half_hour_end_time[pairs].astype("datetime64[M]").astype(int) % 12 + 1
    summer = np.isin(months, SUMMER_MONTHS)

    def share(mask: np.ndarray) -> float:
        return 1.0 - float((residual[mask] ** 2).sum() / (change[mask] ** 2).sum())

    everything = np.ones_like(summer)
    return share(everything), share(summer), share(~summer)


def analyse(task: tuple[str, str]) -> dict | None:
    """Run the placebo on one BMU: its identifier and its group.

    Args:
        task: The BMU's identifier and its group, `aggregate` or `solar`.

    Returns:
        The BMU's real increment, the placebo increments, and the real increment by season, or None
        when a plant cannot be fitted.
    """
    bmu_id, group = task
    output = output_on_grid(bmu_id=bmu_id)
    if group == "aggregate":
        register = bmu_register().filter(pl.col("elexonBmUnit") == bmu_id)
        latitude, longitude = gsp_group_centroids()[register["gspGroupId"][0]]
        point_ids = nearest_grid_points(
            latitude=latitude, longitude=longitude, count=NEAREST_POINTS
        )
        previous = pl.read_parquet(OUTPUT_DIR / "stage3_separations.parquet").filter(
            pl.col("bmu") == bmu_id
        )
        capacity = float(previous["regional_difference_capacity_mw"][0])
    else:
        latitude, longitude = REFERENCE_LATITUDE, REFERENCE_LONGITUDE
        point_ids = grid_point_ids()
        previous = pl.read_parquet(OUTPUT_DIR / "stage3b_real_controls.parquet").filter(
            (pl.col("bmu") == bmu_id) & (pl.col("group") == "positive control")
        )
        capacity = float(previous["separation_capacity_mw"][0])
    guess = max(capacity, START_SHARE * float(np.nanquantile(np.abs(output), 0.99)))

    hourly = hourly_cams(point_ids=point_ids)
    all_sky = sky_at(hourly=hourly, latitude=latitude, longitude=longitude)
    clear_sky = sky_at(
        hourly=hourly_cams(point_ids=point_ids, column="clear_sky_ghi_w_m2"),
        latitude=latitude,
        longitude=longitude,
    )
    real = _fit_and_share(sky=all_sky, output=output, ac_guess_mw=guess)
    clear = _fit_and_share(sky=clear_sky, output=output, ac_guess_mw=guess)
    if real is None or clear is None:
        return None
    real_all, real_summer, real_winter = _shares(sky=all_sky, output=output, power=real[1])
    clear_all, clear_summer, clear_winter = _shares(sky=clear_sky, output=output, power=clear[1])

    days = sorted(set((hourly["time"] - timedelta(hours=1)).dt.date().to_list()))
    placebo = []
    for index in range(PERMUTATIONS):
        mapping = day_permutation(days=days, rng=np.random.default_rng([SEED, index]))
        sky = sky_at(
            hourly=permute_hourly(hourly=hourly, mapping=mapping),
            latitude=latitude,
            longitude=longitude,
        )
        fitted = _fit_and_share(sky=sky, output=output, ac_guess_mw=guess)
        if fitted is not None:
            placebo.append(_shares(sky=sky, output=output, power=fitted[1])[0] - clear_all)
    return {
        "bmu": bmu_id,
        "group": group,
        "real_increment": real_all - clear_all,
        "placebo_increment_mean": float(np.mean(placebo)),
        "placebo_increment_p95": float(np.quantile(placebo, 0.95)),
        "placebo_increment_max": float(np.max(placebo)),
        "summer_increment": real_summer - clear_summer,
        "winter_increment": real_winter - clear_winter,
    }


def main() -> None:
    """Run every BMU and write `stage3c_placebo.parquet`."""
    tasks = [(b, "aggregate") for b in AGGREGATE_BMUS] + [(b, "solar") for b in VALIDATION_BMUS]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = [r for r in pool.map(analyse, tasks) if r is not None]
    frame = pl.DataFrame(results, infer_schema_length=None)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUTPUT_DIR / "stage3c_placebo.parquet")
    print(frame)


if __name__ == "__main__":
    main()
