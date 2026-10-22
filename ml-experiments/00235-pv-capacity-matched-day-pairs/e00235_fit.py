"""Estimate the solar capacity from matched sunny-minus-cloudy day pairs, with an injected farm.

A pair is two days of the same day type and clock offset, within 21 days of each other, with mean
daylight temperatures within 1 K, where the sunny day's daylight clear-sky index exceeds the cloudy
day's by at least 0.4. The estimator is the least-squares scale, through the origin, of the pair's
difference in net power on the pair's difference in the derated unit shape, over the daylight
half-hours where both days have a training row. The estimator is linear in the power, so the
recovery of an injected farm is the same regression applied to the farm's own output.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00235-*/e00235_fit.py --sites D10
"""

import argparse
from dataclasses import dataclass, replace
from typing import Final

import e00235_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, derated_unit_shape, prepare_fit
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

HALF_HOURS_PER_DAY: Final[int] = 48
TOY_CAPACITY: Final[float] = 7.0
TOY_TOLERANCE: Final[float] = 1e-6


@dataclass(frozen=True)
class DayTable:
    """The training half-hours of every day on a `(days, 48)` grid, `NaN` where there is no row.

    Attributes:
        days: Each day as days since 1970-01-01 (UTC date of the half-hour's start).
        shape: The derated unit shape.
        net: The base net power in points.
        output: The injected farm's output in points.
        load: The cloud-driven load in points.
        clear_index: The day's daylight clear-sky index, `NaN` for a day with too few daylight rows.
        temperature: The day's mean daylight temperature.
        day_type: The index into the day types.
        offset: The clock offset of the day, in half-hours.
        holiday: True on a day in a holiday period.
        season_year: The season-year of the day (July to June, named by the July's year).
    """

    days: np.ndarray
    shape: np.ndarray
    net: np.ndarray
    output: np.ndarray
    load: np.ndarray
    clear_index: np.ndarray
    temperature: np.ndarray
    day_type: np.ndarray
    offset: np.ndarray
    holiday: np.ndarray
    season_year: np.ndarray


def day_table(
    *,
    min_daylight_rows: int,
    cache: FeatureCache,
    site: str,
    index: np.ndarray,
    output: np.ndarray,
    load: np.ndarray,
) -> DayTable:
    """Lay the training rows `index` out by day; a day needs `min_daylight_rows` daylight rows."""
    features = cache.sites[site]
    start = cache.end_times[index] - np.timedelta64(30, "m")
    day_of_row = start.astype("datetime64[D]").astype(np.int64)
    slot = ((start - start.astype("datetime64[D]")) // np.timedelta64(30, "m")).astype(np.int64)
    days, row_day = np.unique(day_of_row, return_inverse=True)
    n_days = len(days)

    def grid(values: np.ndarray) -> np.ndarray:
        table = np.full((n_days, HALF_HOURS_PER_DAY), np.nan)
        table[row_day, slot] = values
        return table

    shape = derated_unit_shape(features=features, index=index)
    ghi, clear_sky = common.clear_sky_index(cache=cache, site=site)
    daylight = clear_sky[index] > common.DAYLIGHT_CLEAR_SKY_W_M2
    valid = daylight & np.isfinite(ghi[index])
    counts = np.bincount(row_day[valid], minlength=n_days)
    clear_index = np.full(n_days, np.nan)
    temperature = np.full(n_days, np.nan)
    ghi_sum = np.bincount(row_day[valid], weights=ghi[index][valid], minlength=n_days)
    clear_sum = np.bincount(row_day[valid], weights=clear_sky[index][valid], minlength=n_days)
    temperature_sum = np.bincount(
        row_day[valid], weights=features.temperature_c[index][valid], minlength=n_days
    )
    enough = counts >= min_daylight_rows
    clear_index[enough] = ghi_sum[enough] / clear_sum[enough]
    temperature[enough] = temperature_sum[enough] / counts[enough]
    first = np.zeros(n_days, dtype=np.int64)
    first[row_day[::-1]] = np.arange(len(index))[::-1]
    local = cache.calendar.half_hour[index][first].astype(np.int64)
    utc = slot[first]
    holiday = np.zeros(n_days, dtype=bool)
    holiday[row_day[cache.calendar.holiday_period[index] >= 0]] = True
    return DayTable(
        days=days,
        shape=grid(shape),
        net=grid(features.y[index]),
        output=grid(output[index]),
        load=grid(load[index]),
        clear_index=clear_index,
        temperature=temperature,
        day_type=cache.calendar.day_type[index][first].astype(np.int64),
        offset=(local - utc) % HALF_HOURS_PER_DAY,
        holiday=holiday,
        season_year=cache.calendar.season_year[index][first].astype(np.int64),
    )


def find_pairs(*, table: DayTable, month_filter: tuple[int, ...] | None = None) -> np.ndarray:
    """Return the pairs as an array of (sunny day, cloudy day) positions, shape (P, 2).

    Args:
        table: The days.
        month_filter: Calendar months the sunny day must fall in, or `None` for any.

    Returns:
        The pairs; each sunny day takes at most `PARTNERS_PER_DAY` partners.
    """
    pairs = []
    ok = np.isfinite(table.clear_index) & ~table.holiday
    month = table.days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12 + 1
    for sunny in np.flatnonzero(ok):
        if month_filter is not None and month[sunny] not in month_filter:
            continue
        near = np.abs(table.days - table.days[sunny]) <= common.PAIR_MAX_DAY_GAP
        candidates = (
            ok
            & near
            & (table.day_type == table.day_type[sunny])
            & (table.offset == table.offset[sunny])
            & (table.clear_index <= table.clear_index[sunny] - common.PAIR_MIN_CLEAR_INDEX_GAP)
            & (
                np.abs(table.temperature - table.temperature[sunny])
                <= common.PAIR_MAX_TEMPERATURE_GAP_K
            )
        )
        chosen = np.flatnonzero(candidates)
        gaps = np.abs(table.temperature[chosen] - table.temperature[sunny])
        pairs.extend(
            (sunny, cloudy) for cloudy in chosen[np.argsort(gaps)[: common.PARTNERS_PER_DAY]]
        )
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def pair_sums(*, table: DayTable, pairs: np.ndarray) -> pl.DataFrame:
    """Return per-pair sums over the pair's usable half-hours, with the month of the sunny day.

    The columns are `xx` (sum of the squared shape difference), `xy_net` (shape difference times
    the net-power difference), `xy_output` and `xy_load` (the same for the farm output and the
    load), and `month` (months since 1970-01).
    """
    sunny, cloudy = pairs[:, 0], pairs[:, 1]
    dx = table.shape[sunny] - table.shape[cloudy]
    usable = (
        np.isfinite(dx)
        & np.isfinite(table.net[sunny])
        & np.isfinite(table.net[cloudy])
        & (
            (np.nan_to_num(table.shape[sunny]) > common.DAYLIGHT_UNIT_SHAPE)
            | (np.nan_to_num(table.shape[cloudy]) > common.DAYLIGHT_UNIT_SHAPE)
        )
    )
    dx = np.where(usable, dx, 0.0)

    def difference(values: np.ndarray) -> np.ndarray:
        return np.where(usable, np.nan_to_num(values[sunny]) - np.nan_to_num(values[cloudy]), 0.0)

    month = table.days[sunny].astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
    return pl.DataFrame(
        {
            "month": month,
            "xx": (dx**2).sum(axis=1),
            "xy_net": (dx * difference(table.net)).sum(axis=1),
            "xy_output": (dx * difference(table.output)).sum(axis=1),
            "xy_load": (dx * difference(table.load)).sum(axis=1),
            "n_half_hours": usable.sum(axis=1),
        }
    )


def month_sums(*, sums: pl.DataFrame) -> pl.DataFrame:
    """Return the pair sums added up by the sunny day's month."""
    return sums.group_by("month").sum().sort("month")


def bootstrap_weights(*, months: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Return month weights (n_resamples + 1, M); row 0 is ones. Strata are calendar quarters."""
    stratum = ((months + 6) // 12) * 4 + (months % 12) // 3
    generator = np.random.default_rng(seed)
    counts = np.ones((n_resamples + 1, len(months)))
    for resample in range(1, n_resamples + 1):
        counts[resample] = 0.0
        for value in np.unique(stratum):
            members = np.flatnonzero(stratum == value)
            drawn = generator.choice(members, size=len(members), replace=True)
            counts[resample] += np.bincount(drawn, minlength=len(months))
    return counts


def estimates(
    *, by_month: pl.DataFrame, n_resamples: int, seed: int, injected_points: float
) -> dict[str, np.ndarray]:
    """Return the capacity and the recoveries for the point fit and every resample."""
    weights = bootstrap_weights(
        months=by_month["month"].to_numpy(), n_resamples=n_resamples, seed=seed
    )
    xx = weights @ by_month["xx"].to_numpy()
    xy_net = weights @ by_month["xy_net"].to_numpy()
    xy_output = weights @ by_month["xy_output"].to_numpy()
    xy_load = weights @ by_month["xy_load"].to_numpy()
    return {
        "capacity": -xy_net / xx,
        "recovery": xy_output / xx / injected_points,
        "recovery_with_load": (xy_output - xy_load) / xx / injected_points,
    }


def toy_check() -> None:
    """Recover a known capacity from a synthetic series with no demand and a farm of known size."""
    generator = np.random.default_rng(0)
    n_days = 120
    shape = generator.uniform(0.2, 1.0, (n_days, HALF_HOURS_PER_DAY))
    table = DayTable(
        days=np.arange(n_days, dtype=np.int64) + 19000,
        shape=shape,
        net=-TOY_CAPACITY * shape,
        output=0.5 * TOY_CAPACITY * shape,
        load=np.zeros_like(shape),
        clear_index=np.tile([0.9, 0.3], n_days // 2),
        temperature=np.full(n_days, 10.0),
        day_type=np.zeros(n_days, dtype=np.int64),
        offset=np.zeros(n_days, dtype=np.int64),
        holiday=np.zeros(n_days, dtype=bool),
        season_year=np.full(n_days, common.READ_OUT_SEASON_YEAR),
    )
    pairs = find_pairs(table=table)
    by_month = month_sums(sums=pair_sums(table=table, pairs=pairs))
    result = estimates(by_month=by_month, n_resamples=0, seed=0, injected_points=TOY_CAPACITY)
    if abs(result["capacity"][0] - TOY_CAPACITY) > TOY_TOLERANCE:
        msg = f"toy capacity {result['capacity'][0]} differs from {TOY_CAPACITY}"
        raise ValueError(msg)
    if abs(result["recovery"][0] - 0.5) > TOY_TOLERANCE:
        msg = f"toy recovery {result['recovery'][0]} differs from 0.5"
        raise ValueError(msg)
    print(f"toy check passed: {len(pairs)} pairs, capacity {result['capacity'][0]:.6f}")


def run_farm(
    *,
    cache: FeatureCache,
    site: str,
    farm: str,
    n_resamples: int,
    injected_points: float,
    min_daylight_rows: int,
) -> list[dict]:
    """Return the result rows of one Primary and farm."""
    features = cache.sites[site]
    norm = common.farm_norm(cache=cache, farm=farm)
    output = injected_points * norm
    load = common.cloud_load(cache=cache, site=site, capacity_points=injected_points)
    masked = replace(cache, sites={**cache.sites, site: replace(features, y=features.y - output)})
    _, index = prepare_fit(
        cache=masked,
        site=site,
        spec=GamSpec(ridge=common.RIDGE, capacity="fitted"),
        cut=common.CUT,
        window=common.TRAINING_WINDOW,
        season_years=common.SEASON_YEARS,
    )
    table = day_table(
        min_daylight_rows=min_daylight_rows,
        cache=cache,
        site=site,
        index=index,
        output=output,
        load=load,
    )
    seed = 1000 + common.ALL_SITES.index(site)
    rows: list[dict] = []
    read_out = table.season_year == common.READ_OUT_SEASON_YEAR
    for name, day_mask, months_filter in (
        ("main", read_out, None),
        ("placebo", np.ones(len(table.days), dtype=bool), common.PLACEBO_MONTHS),
    ):
        subset = replace(
            table,
            clear_index=np.where(day_mask, table.clear_index, np.nan),
        )
        pairs = find_pairs(table=subset, month_filter=months_filter)
        if len(pairs) == 0:
            rows.append(
                {"site": site, "farm": farm, "set": name, "stat": "n_pairs", "b": 0, "value": 0.0}
            )
            continue
        sums = pair_sums(table=table, pairs=pairs)
        result = estimates(
            by_month=month_sums(sums=sums),
            n_resamples=n_resamples,
            seed=seed,
            injected_points=injected_points,
        )
        for stat, values in result.items():
            rows.extend(
                {"site": site, "farm": farm, "set": name, "stat": stat, "b": b, "value": float(v)}
                for b, v in enumerate(values)
            )
        rows += [
            {
                "site": site,
                "farm": farm,
                "set": name,
                "stat": "n_pairs",
                "b": 0,
                "value": float(len(pairs)),
            },
            {
                "site": site,
                "farm": farm,
                "set": name,
                "stat": "n_half_hours",
                "b": 0,
                "value": float(sums["n_half_hours"].sum()),
            },
        ]
    return rows


def main() -> None:
    """Write `work/pairs_<site>.parquet` for each Primary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", default=list(common.ALL_SITES))
    parser.add_argument("--farms", nargs="+", default=list(common.FARMS))
    parser.add_argument("--resamples", type=int, default=common.N_RESAMPLES)
    parser.add_argument("--min-daylight-rows", type=int, default=common.MIN_DAYLIGHT_ROWS)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--toy", action="store_true", help="only run the toy check")
    arguments = parser.parse_args()
    toy_check()
    if arguments.toy:
        return
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = common.load_cache()
    for site in arguments.sites:
        pinned = pinned_capacity(
            vintages=read_vintages(), table=cache.sites[site].table, cut=common.CUT
        ).held_points
        injected_points = common.INJECTED_SHARE * pinned
        table: list[dict] = []
        for farm in arguments.farms:
            table += run_farm(
                cache=cache,
                site=site,
                farm=farm,
                n_resamples=arguments.resamples,
                injected_points=injected_points,
                min_daylight_rows=arguments.min_daylight_rows,
            )
            print(f"{site} farm {farm}: done", flush=True)
        pl.DataFrame(table).write_parquet(
            common.WORK_DIR / f"pairs_{site}{arguments.suffix}.parquet"
        )


if __name__ == "__main__":
    main()
