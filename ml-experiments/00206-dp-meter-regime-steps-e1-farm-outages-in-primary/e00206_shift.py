"""Test whether each farm outage shifts the power of the Primaries near it, and run the placebo.

Needs `work/outages.parquet` and `work/farm_days_<farm>.parquet` from `e00206_outages.py`. Writes
`work/shifts.parquet`, `work/placebo.parquet`, and `work/map.parquet`.

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00206-*/e00206_shift.py
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

import e00206_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, default_input_paths
from studies.power import scan_power

POINTS_PER_CAPACITY: Final[float] = 100.0
"""Points of effective capacity in one effective capacity (`y` is in percent)."""


@dataclass(frozen=True)
class PrimaryDays:
    """One Primary's sunny-daylight power on a day-by-half-hour grid.

    Attributes:
        site: The Primary's label.
        first_day: The date of row 0.
        power: Power in MW, shape (days, 48), `NaN` where not sunny daylight or not valid.
        total: `power` with `NaN` set to 0, for resampled means.
        count: 1.0 where `power` is finite.
    """

    site: str
    first_day: date
    power: np.ndarray
    total: np.ndarray
    count: np.ndarray


def great_circle_km(*, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres."""
    p = np.pi / 180
    a = (
        np.sin((lat2 - lat1) * p / 2) ** 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    )
    return float(12742 * np.arcsin(np.sqrt(a)))


def primary_table() -> pl.DataFrame:
    """Return every Primary with units, location, and bulk supply point (null when unknown)."""
    series = pl.read_parquet(default_input_paths().series_table).filter(
        pl.col("site").is_not_null()
    )
    monitor = pl.read_csv(common.HIERARCHY_DIR / "Monitor Direction.csv")
    hierarchy = pl.read_csv(common.HIERARCHY_DIR / "Substations.csv")
    return (
        series.join(
            monitor.select(
                pl.col("TimeSeriesID").alias("time_series_id"), "SubstationNumber", "SubstationType"
            ),
            on="time_series_id",
        )
        .join(
            hierarchy.select(pl.col("substation_number").alias("SubstationNumber"), "bsp_name"),
            on="SubstationNumber",
            how="left",
        )
        .filter(pl.col("SubstationType") == "Primary")
        .select("site", "units", "latitude", "longitude", "bsp_name", "effective_capacity_mw")
    )


def farm_bsp() -> dict[str, str | None]:
    """Return each farm's bulk supply point from NGED's hierarchy file, or `None` when absent."""
    farms = common.farm_table()
    monitor = pl.read_csv(common.HIERARCHY_DIR / "Monitor Direction.csv")
    hierarchy = pl.read_csv(common.HIERARCHY_DIR / "Substations.csv")
    joined = farms.join(
        monitor.select(pl.col("TimeSeriesID").alias("time_series_id"), "SubstationNumber"),
        on="time_series_id",
    ).join(
        hierarchy.select(pl.col("substation_number").alias("SubstationNumber"), "bsp_name"),
        on="SubstationNumber",
        how="left",
    )
    return dict(zip(joined["site"].to_list(), joined["bsp_name"].to_list(), strict=True))


PROXY_SITE: Final[str] = "D10"
"""The cached Primary whose CAMS irradiance stands in for the bulk supply point's."""
BSP_SITE: Final[str] = "D13"
"""The Skegness bulk supply point, which reports in MW. It is tested for farms whose bulk supply
point is Skegness or unknown, as a post hoc check on whether a farm sits above the Primaries."""


def build_map() -> pl.DataFrame:
    """Return the farm-to-Primary map: every MW Primary within `NEAR_KM`, and which is assigned."""
    primaries = primary_table()
    bsp = farm_bsp()
    rows = []
    for farm in common.farm_table().iter_rows(named=True):
        near = []
        for p in primaries.iter_rows(named=True):
            km = great_circle_km(
                lat1=farm["latitude"],
                lon1=farm["longitude"],
                lat2=p["latitude"],
                lon2=p["longitude"],
            )
            near.append({**p, "km": km})
        mw = sorted((n for n in near if n["units"] == "MW"), key=lambda n: n["km"])
        same = [n for n in mw if bsp[farm["site"]] is None or n["bsp_name"] == bsp[farm["site"]]]
        assigned = same[0]["site"]
        rows += [
            {
                "farm": farm["site"],
                "primary": n["site"],
                "km": round(n["km"], 1),
                "assigned": n["site"] == assigned,
                "bsp_known": bsp[farm["site"]] is not None,
                "units": n["units"],
            }
            for n in sorted(near, key=lambda n: n["km"])
            if n["km"] <= common.NEAR_KM or n["site"] == assigned
        ]
    series = pl.read_parquet(common.default_input_paths().series_table)
    bsp_row = series.filter(pl.col("site") == BSP_SITE).row(0, named=True)
    for farm in common.farm_table().iter_rows(named=True):
        if bsp[farm["site"]] in (None, "SKEGNESS 33kV S STN"):
            rows.append(
                {
                    "farm": farm["site"],
                    "primary": BSP_SITE,
                    "km": round(
                        great_circle_km(
                            lat1=farm["latitude"],
                            lon1=farm["longitude"],
                            lat2=bsp_row["latitude"],
                            lon2=bsp_row["longitude"],
                        ),
                        1,
                    ),
                    "assigned": False,
                    "bsp_known": True,
                    "units": "MW",
                }
            )
    return pl.DataFrame(rows)


def primary_days(*, cache: FeatureCache, site: str) -> PrimaryDays:
    """Return a Primary's sunny-daylight power by day and half-hour (UTC).

    The bulk supply point is not in the cache, so its power is read from the cleaned power table,
    and its sunny mask uses the CAMS irradiance of the nearest cached Primary.
    """
    if site in cache.sites:
        features = cache.sites[site]
        table = features.table
        power = features.y * table.effective_capacity_mw / POINTS_PER_CAPACITY
        valid = features.present & ~features.switching_mask & np.isfinite(power)
        cams = features.ghi_w_m2
        latitude, longitude = table.latitude, table.longitude
    else:
        series = pl.read_parquet(common.default_input_paths().series_table).filter(
            pl.col("site") == site
        )
        row = series.row(0, named=True)
        latitude, longitude = row["latitude"], row["longitude"]
        raw = (
            scan_power()
            .filter(pl.col("time_series_id") == row["time_series_id"])
            .select("time", "power")
            .collect()
            .unique("time")
        )
        grid = pl.DataFrame({"time": cache.times_series()}).join(
            raw, on="time", how="left", maintain_order="left"
        )
        power = grid["power"].to_numpy().astype(np.float64)
        valid = np.isfinite(power)
        cams = cache.sites[PROXY_SITE].ghi_w_m2
    clear = common.clear_sky_ghi(end_times=cache.end_times, latitude=latitude, longitude=longitude)
    sunny = (
        (clear > common.SUNNY_CLEAR_SKY_W_M2)
        & np.isfinite(cams)
        & (cams >= common.SUNNY_CAMS_SHARE * clear)
        & valid
    )
    middle = cache.end_times - np.timedelta64(15, "m")
    days = middle.astype("datetime64[D]")
    first = days.min()
    day_index = (days - first).astype(int)
    half_hour = ((middle - days.astype("datetime64[us]")) / np.timedelta64(30, "m")).astype(int)
    grid = np.full((int(day_index.max()) + 1, 48), np.nan)
    grid[day_index[sunny], half_hour[sunny]] = power[sunny]
    finite = np.isfinite(grid)
    return PrimaryDays(
        site=site,
        first_day=first.astype(object),
        power=grid,
        total=np.where(finite, grid, 0.0),
        count=finite.astype(float),
    )


def window_shift(
    *, days: PrimaryDays, inside: np.ndarray, reference: np.ndarray, rng: np.random.Generator
) -> dict:
    """Return the shift (inside minus reference, MW) with a day-resampling 95% interval.

    The shift is averaged over the half-hours of the day that have at least
    `MIN_SAMPLES_PER_HALF_HOUR` sunny samples in both windows.

    Args:
        days: The Primary's sunny power.
        inside: Day indices inside the outage.
        reference: Day indices of the reference windows.
        rng: The random generator for the resampling.

    Returns:
        `shift`, `lo`, `hi`, `n_in`, `n_ref`, and `tested`.
    """
    n_in = int(days.count[inside].sum())
    n_ref = int(days.count[reference].sum())
    out = {
        "shift": np.nan,
        "lo": np.nan,
        "hi": np.nan,
        "n_in": n_in,
        "n_ref": n_ref,
        "tested": False,
    }
    if n_in < common.MIN_SUNNY_IN_OUTAGE or n_ref < common.MIN_SUNNY_IN_REFERENCE:
        return out
    total_in, count_in = days.total[inside], days.count[inside]
    total_ref, count_ref = days.total[reference], days.count[reference]

    def shift_of(w_in: np.ndarray, w_ref: np.ndarray) -> np.ndarray:
        c_in, c_ref = w_in @ count_in, w_ref @ count_ref
        ok = (c_in >= common.MIN_SAMPLES_PER_HALF_HOUR) & (
            c_ref >= common.MIN_SAMPLES_PER_HALF_HOUR
        )
        diff = (w_in @ total_in) / np.maximum(c_in, 1) - (w_ref @ total_ref) / np.maximum(c_ref, 1)
        return np.where(ok, diff, np.nan)

    point = np.nanmean(shift_of(np.ones((1, len(inside))), np.ones((1, len(reference)))), axis=1)
    w_in = rng.multinomial(
        len(inside), np.full(len(inside), 1 / len(inside)), size=common.N_RESAMPLES
    )
    w_ref = rng.multinomial(
        len(reference), np.full(len(reference), 1 / len(reference)), size=common.N_RESAMPLES
    )
    resampled = np.nanmean(shift_of(w_in.astype(float), w_ref.astype(float)), axis=1)
    lo, hi = np.nanpercentile(resampled, [2.5, 97.5])
    out.update(shift=float(point[0]), lo=float(lo), hi=float(hi), tested=True)
    return out


def reference_days(*, start: int, end: int, bad: np.ndarray) -> np.ndarray:
    """Return the day indices within `REFERENCE_DAYS` either side of a window, bad days excluded."""
    before = np.arange(start - common.REFERENCE_DAYS, start)
    after = np.arange(end + 1, end + 1 + common.REFERENCE_DAYS)
    both = np.concatenate([before, after])
    both = both[(both >= 0) & (both < len(bad))]
    return both[~bad[both]]


def farm_p99(*, cleaned: pl.DataFrame, farm_id: int) -> float:
    """Return the farm's 99th-percentile AC output over its daylight half-hours, in MW."""
    own = cleaned.filter(
        pl.col("time_series_id") == farm_id,
        pl.col("drop_reason").is_null(),
        pl.col("clear_sky_ghi_w_m2") > 20.0,
    )
    return float(np.percentile(own["power"].to_numpy(), 99))


def classify(*, ratio: float, lo: float, hi: float) -> dict:
    """Return whether a ratio (positive = more solar in the Primary) shows the shift, either way.

    `shows_either` ignores the sign, which an MVA series does not carry.
    """
    more = common.SHIFT_RATIO_LOW <= ratio <= common.SHIFT_RATIO_HIGH and lo > 0.0
    less = -common.SHIFT_RATIO_HIGH <= ratio <= -common.SHIFT_RATIO_LOW and hi < 0.0
    either = common.SHIFT_RATIO_LOW <= abs(ratio) <= common.SHIFT_RATIO_HIGH and (
        lo > 0.0 or hi < 0.0
    )
    return {"shows_more_pv": bool(more), "shows_less_pv": bool(less), "shows_either": bool(either)}


def main() -> None:
    """Write the shift, placebo, and map tables."""
    cache = common.load_cache()
    cleaned = pl.read_parquet(common.CLEANED_FARMS_PATH)
    farms = common.farm_table()
    mapping = build_map()
    mapping.write_parquet(common.WORK_DIR / "map.parquet")
    print(mapping)
    outages = pl.read_parquet(common.WORK_DIR / "outages.parquet")
    needed = sorted(set(mapping["primary"].to_list()))
    cache_days = {s: primary_days(cache=cache, site=s) for s in needed}
    sample = cache_days[needed[0]]
    day0 = sample.first_day
    n_days = sample.power.shape[0]
    rng = np.random.default_rng(common.SEED)
    shift_rows, placebo_rows = [], []
    for farm in farms.iter_rows(named=True):
        label = farm["site"]
        p99 = farm_p99(cleaned=cleaned, farm_id=farm["time_series_id"])
        days_table = pl.read_parquet(common.WORK_DIR / f"farm_days_{label}.parquet")
        bad = np.zeros(n_days, dtype=bool)
        eligible = np.zeros(n_days, dtype=bool)
        for r in days_table.iter_rows(named=True):
            if r["n_clear"] >= common.MIN_CLEAR_HALF_HOURS_PER_DAY:
                index = (r["day"] - day0).days
                eligible[index] = True
                bad[index] = (r["n_zero"] + r["n_missing"]) >= common.BAD_DAY_SHARE * r["n_clear"]
        targets = mapping.filter(pl.col("farm") == label)
        own = outages.filter(pl.col("farm") == label)
        for o in own.iter_rows(named=True):
            start, end = (o["start"] - day0).days, (o["end"] - day0).days
            inside = np.arange(start, end + 1)
            reference = reference_days(start=start, end=end, bad=bad)
            for t in targets.iter_rows(named=True):
                res = window_shift(
                    days=cache_days[t["primary"]], inside=inside, reference=reference, rng=rng
                )
                ratio, lo_ratio, hi_ratio = -res["shift"] / p99, -res["hi"] / p99, -res["lo"] / p99
                shift_rows.append(
                    {
                        "farm": label,
                        "kind": o["kind"],
                        "start": o["start"],
                        "end": o["end"],
                        "primary": t["primary"],
                        "assigned": t["assigned"],
                        "units": t["units"],
                        "km": t["km"],
                        "farm_p99_mw": p99,
                        "shift_mw": res["shift"],
                        "ratio_more_pv": ratio,
                        "ratio_lo": lo_ratio,
                        "ratio_hi": hi_ratio,
                        "n_sunny_in": res["n_in"],
                        "n_sunny_ref": res["n_ref"],
                        "tested": res["tested"],
                        **(
                            classify(ratio=ratio, lo=lo_ratio, hi=hi_ratio)
                            if res["tested"]
                            else {
                                "shows_more_pv": False,
                                "shows_less_pv": False,
                                "shows_either": False,
                            }
                        ),
                    }
                )
        length = int(np.median(own["calendar_days"].to_numpy())) if own.height else 4
        assigned = targets.filter(pl.col("assigned"))["primary"][0]
        months = np.array([(day0 + timedelta(days=int(i))).month for i in range(n_days)])
        usable = [
            i
            for i in range(common.REFERENCE_DAYS, n_days - length - common.REFERENCE_DAYS)
            if 3 <= months[i] <= 9
            and not bad[i : i + length].any()
            and eligible[i : i + length].sum() >= common.MIN_OUTAGE_DAYS
            and eligible[i]
            and days_table["day"].min() <= day0 + timedelta(days=i)  # ty: ignore[unsupported-operator]
            and day0 + timedelta(days=i + length) <= days_table["day"].max()  # ty: ignore[unsupported-operator]
        ]
        picks = rng.choice(usable, size=min(common.N_PLACEBO, len(usable)), replace=False)
        for i in picks:
            inside = np.arange(i, i + length)
            reference = reference_days(start=int(i), end=int(i + length - 1), bad=bad)
            res = window_shift(
                days=cache_days[assigned], inside=inside, reference=reference, rng=rng
            )
            ratio, lo_ratio, hi_ratio = -res["shift"] / p99, -res["hi"] / p99, -res["lo"] / p99
            placebo_rows.append(
                {
                    "farm": label,
                    "primary": assigned,
                    "start": day0 + timedelta(days=int(i)),
                    "length": length,
                    "ratio_more_pv": ratio,
                    "tested": res["tested"],
                    **(
                        classify(ratio=ratio, lo=lo_ratio, hi=hi_ratio)
                        if res["tested"]
                        else {"shows_more_pv": False, "shows_less_pv": False, "shows_either": False}
                    ),
                }
            )
        print(label, "done", flush=True)
    shifts = pl.DataFrame(shift_rows)
    placebo = pl.DataFrame(placebo_rows)
    shifts.write_parquet(common.WORK_DIR / "shifts.parquet")
    placebo.write_parquet(common.WORK_DIR / "placebo.parquet")
    print(shifts.filter(pl.col("tested")).sort("farm", "start", "primary"))
    print(
        placebo.filter(pl.col("tested"))
        .group_by("farm")
        .agg(
            n=pl.len(),
            more=pl.col("shows_more_pv").mean(),
            less=pl.col("shows_less_pv").mean(),
            mean_ratio=pl.col("ratio_more_pv").mean(),
            sd_ratio=pl.col("ratio_more_pv").std(),
        )
        .sort("farm")
    )


if __name__ == "__main__":
    main()
