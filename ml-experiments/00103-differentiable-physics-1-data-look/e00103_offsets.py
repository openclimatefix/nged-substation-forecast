"""Estimate the offset of the power stamps from the sun's ramps at sunrise and sunset.

For each series, the half-hour-to-half-hour change of net power on clear-sky days, while the sun is
between 3 degrees below and 25 degrees above the horizon, is regressed on the change of a
sun-height predictor (the cosine of the solar zenith angle, clipped at zero, averaged over the
half-hour that ends `offset` minutes before the stamp). The offset with the smallest residual sum
of squares is the estimate, refined by a parabola through the three offsets around it. A positive
estimate means the stored stamps run late: the power stamped `T` matches the sun over the half-hour
ending at `T - offset`. Intervals come from resampling whole clear-sky days.

The stored stamps already carry the ingest correction (30 minutes earlier before 2026-03-26 08:30
UTC), so a correct correction gives an estimate of zero on both sides of that instant, and the
uncorrected feed would read 30 minutes more before it.

Two kinds of series are measured: the 8 MW Primaries (net power, demand minus embedded PV), and
the metered PV series, which are measured internally and reported only as an aggregate, with no
names, identifiers, or dates.

    POLARS_MAX_THREADS=4 uv run python ml-experiments/00103-*/e00103_offsets.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
import pvlib
from e00103_common import (
    METADATA_PATH,
    MW_PRIMARIES,
    PANEL_END,
    STAMP_FIX_UTC,
    WORK_DIR,
    load_series,
)
from e00103_stats import load_panel
from studies.power import scan_power

OFFSETS_MINUTES: Final[np.ndarray] = np.arange(-120, 125, 5)
"""The offsets tested, in minutes; the refinement interpolates between them."""
STEP_MINUTES: Final[int] = 5
"""The spacing of the sun-height grid, so a half-hour holds 6 samples."""
FIX_INSTANT: Final[np.datetime64] = np.datetime64(STAMP_FIX_UTC.replace(tzinfo=None), "us")
"""The instant NGED began stamping correctly (naive UTC, to match the stamps)."""
RAMP_ELEVATION_RANGE: Final[tuple[float, float]] = (-3.0, 25.0)
"""The solar elevations, in degrees, at the middle of the earlier half-hour that enter the fit."""
CLEAR_DAY_MEAN_INDEX: Final[float] = 0.85
CLEAR_DAY_MIN_INDEX: Final[float] = 0.6
"""A day is clear when CAMS irradiance over clear-sky irradiance averages above the first value
and never falls below the second, over the hours with clear-sky irradiance above 100 W/m2."""
N_RESAMPLES: Final[int] = 1000
MIN_DAYS: Final[int] = 25


def sun_grid(
    *, latitude: float, longitude: float, start: datetime, end: datetime
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The clipped cosine of the zenith, the elevation, and the azimuth every 5 minutes."""
    stamps = pl.datetime_range(start, end, interval=f"{STEP_MINUTES}m", time_zone="UTC", eager=True)
    position = pvlib.solarposition.get_solarposition(
        time=stamps.to_numpy(), latitude=latitude, longitude=longitude
    )
    zenith = position["apparent_zenith"].to_numpy().astype(float)
    return (
        np.clip(np.cos(np.radians(zenith)), 0.0, None),
        90.0 - zenith,
        position["azimuth"].to_numpy().astype(float),
    )


def clear_days(*, cams: pl.DataFrame) -> set:
    """The UTC dates on which the CAMS irradiance stays close to its clear-sky value."""
    index = (
        cams.filter(pl.col("clear_sky_ghi_w_m2") > 100)
        .with_columns(
            day=pl.col("time").dt.date(), ratio=pl.col("ghi_w_m2") / pl.col("clear_sky_ghi_w_m2")
        )
        .group_by("day")
        .agg(mean=pl.col("ratio").mean(), low=pl.col("ratio").min(), n=pl.len())
        .filter(
            pl.col("n") >= 6,
            pl.col("mean") > CLEAR_DAY_MEAN_INDEX,
            pl.col("low") > CLEAR_DAY_MIN_INDEX,
        )
    )
    return set(index["day"].to_list())


def day_sums(
    *,
    times: np.ndarray,
    values: np.ndarray,
    grid_start: np.datetime64,
    cos_grid: np.ndarray,
    elevation: np.ndarray,
    azimuth: np.ndarray,
    days: set,
) -> dict:
    """Per-day cross-products of the power change and the predictor change, by offset and ramp.

    Args:
        times: Stamps (`datetime64[us]`, UTC) of the readings, ascending.
        values: Power at each stamp, in percent of effective capacity.
        grid_start: The first instant of the 5-minute sun grid (`datetime64[us]`).
        cos_grid: The clipped cosine of the zenith on that grid.
        elevation: The solar elevation on that grid, degrees.
        azimuth: The solar azimuth on that grid, degrees.
        days: The clear UTC dates.

    Returns:
        For each ramp (`morning`, `evening`), a dict with the day of each row's date, the
        before-fix flag, and arrays of `xy`, `xx`, and `yy` per row and offset.
    """
    step = np.timedelta64(STEP_MINUTES, "m")
    index = ((times - grid_start) // step).astype(np.int64)
    # Keep readings whose predecessor half-hour is also present.
    previous = np.searchsorted(times, times - np.timedelta64(30, "m"))
    has_previous = (previous < len(times)) & (
        times[np.minimum(previous, len(times) - 1)] == times - np.timedelta64(30, "m")
    )
    cumulative = np.concatenate([[0.0], np.cumsum(cos_grid)])
    delta_p = values - values[np.minimum(previous, len(values) - 1)]
    mid_index = index - 6  # the middle of the earlier half-hour, 30 minutes before the stamp
    mid_elevation = elevation[np.clip(mid_index - 3, 0, len(elevation) - 1)]
    morning = azimuth[np.clip(mid_index - 3, 0, len(azimuth) - 1)] < 180.0
    day = times.astype("datetime64[D]")
    in_window = (
        has_previous
        & (mid_elevation > RAMP_ELEVATION_RANGE[0])
        & (mid_elevation < RAMP_ELEVATION_RANGE[1])
        & np.isin(day, np.array(sorted(days), dtype="datetime64[D]"))
    )
    xy = np.zeros((len(times), len(OFFSETS_MINUTES)))
    xx = np.zeros_like(xy)
    for j, offset in enumerate(OFFSETS_MINUTES):
        shift = int(offset // STEP_MINUTES)
        end = np.clip(index - shift + 1, 6, len(cumulative) - 1)
        later = (cumulative[end] - cumulative[end - 6]) / 6.0
        end_previous = np.clip(index - 6 - shift + 1, 6, len(cumulative) - 1)
        earlier = (cumulative[end_previous] - cumulative[end_previous - 6]) / 6.0
        d_s = later - earlier
        xy[:, j] = delta_p * d_s
        xx[:, j] = d_s * d_s
    yy = delta_p * delta_p
    out = {}
    for name, ramp in (("morning", morning), ("evening", ~morning)):
        keep = in_window & ramp
        out[name] = {
            "day": day[keep],
            "before": times[keep] < FIX_INSTANT,
            "month": times[keep].astype("datetime64[M]").astype(int) % 12 + 1,
            "xy": xy[keep],
            "xx": xx[keep],
            "yy": yy[keep],
        }
    return out


def refine(*, sse: np.ndarray) -> float:
    """The offset of the smallest residual sum of squares, refined by a parabola."""
    best = int(np.argmin(sse))
    if best in (0, len(sse) - 1):
        return float(OFFSETS_MINUTES[best])
    below, centre, above = sse[best - 1], sse[best], sse[best + 1]
    curvature = below - 2 * centre + above
    if curvature <= 0:
        return float(OFFSETS_MINUTES[best])
    return float(OFFSETS_MINUTES[best] + STEP_MINUTES * 0.5 * (below - above) / curvature)


def _per_day(
    *, unique: np.ndarray, days: np.ndarray, arrays: tuple[np.ndarray, ...]
) -> list[np.ndarray]:
    """Sum each array over the rows of each day in `unique` (zeros for a day with no rows)."""
    position = np.searchsorted(unique, days)
    out = []
    for array in arrays:
        total = np.zeros((unique.size, *array.shape[1:]))
        np.add.at(total, position, array)
        out.append(total)
    return out


def _offset_from(
    *, total_xy: np.ndarray, total_xx: np.ndarray, total_yy: float, sign: int
) -> tuple[float, float]:
    """The offset of the best fit and the share of variance it explains.

    Only a slope of the expected sign counts: demand ramps can correlate with the sun at the wrong
    sign, and a fit that wins that way says nothing about timing.
    """
    explained = np.maximum(sign * total_xy, 0.0) ** 2 / np.maximum(total_xx, 1e-12)
    sse = total_yy - explained
    return refine(sse=sse), float(1.0 - sse.min() / total_yy)


def estimate(*, ramps: dict, sign: int, seed: int = 0) -> dict:
    """Offsets for the morning ramp, the evening ramp, and their mean, with 95% intervals.

    A fixed-tilt panel facing south sees a sun at low height that is not the cosine of the zenith:
    the morning ramp reads late and the evening ramp early by about the same amount. A stamp offset
    moves both ramps the same way. The mean of the two ramps' offsets is therefore the stamp offset,
    and half their difference is the panel-orientation effect. Intervals resample whole clear-sky
    days, the same days for both ramps.

    Args:
        ramps: For `morning` and `evening`, the day, `xy`, `xx`, and `yy` of each row.
        sign: The sign of the slope that counts as a sun-driven response.
        seed: The resampling seed.

    Returns:
        The estimates and intervals, the number of days, and the share of variance explained.
    """
    all_days = np.unique(np.concatenate([r["day"] for r in ramps.values()]))
    if all_days.size < MIN_DAYS:
        return {"days": int(all_days.size)}
    sums = {
        name: _per_day(unique=all_days, days=r["day"], arrays=(r["xy"], r["xx"], r["yy"]))
        for name, r in ramps.items()
    }
    point = {
        name: _offset_from(
            total_xy=s[0].sum(0), total_xx=s[1].sum(0), total_yy=float(s[2].sum()), sign=sign
        )
        for name, s in sums.items()
    }
    rng = np.random.default_rng(seed)
    draws = {"morning": [], "evening": []}
    for _ in range(N_RESAMPLES):
        counts = rng.multinomial(all_days.size, np.full(all_days.size, 1.0 / all_days.size))
        for name, s in sums.items():
            draws[name].append(
                _offset_from(
                    total_xy=counts @ s[0],
                    total_xx=counts @ s[1],
                    total_yy=float(counts @ s[2]),
                    sign=sign,
                )[0]
            )
    morning, evening = np.array(draws["morning"]), np.array(draws["evening"])
    out: dict = {"days": int(all_days.size)}
    for name, estimate_, draw in (
        ("morning", point["morning"][0], morning),
        ("evening", point["evening"][0], evening),
        ("mean", (point["morning"][0] + point["evening"][0]) / 2, (morning + evening) / 2),
        (
            "half_difference",
            (point["morning"][0] - point["evening"][0]) / 2,
            (morning - evening) / 2,
        ),
    ):
        low, high = np.percentile(draw, [2.5, 97.5])
        out[name] = estimate_
        out[f"{name}_low"] = float(low)
        out[f"{name}_high"] = float(high)
    out["r_squared"] = float(np.mean([point["morning"][1], point["evening"][1]]))
    return out


def measure(
    *,
    label: str,
    kind: str,
    times: np.ndarray,
    values: np.ndarray,
    latitude: float,
    longitude: float,
    days: set,
    sign: int,
) -> list[dict]:
    """The offset estimates for one series: each side of the fix, each ramp, and both ramps."""
    grid_start = times.min().astype("datetime64[D]") - np.timedelta64(2, "D")
    grid_end = times.max().astype("datetime64[D]") + np.timedelta64(2, "D")
    cos_grid, elevation, azimuth = sun_grid(
        latitude=latitude,
        longitude=longitude,
        start=grid_start.astype("datetime64[us]").item().replace(tzinfo=UTC),
        end=grid_end.astype("datetime64[us]").item().replace(tzinfo=UTC),
    )
    sums = day_sums(
        times=times,
        values=values,
        grid_start=grid_start.astype("datetime64[us]"),
        cos_grid=cos_grid,
        elevation=elevation,
        azimuth=azimuth,
        days=days,
    )
    rows = []
    sides = (
        ("before the fix", lambda before, month: before),
        (
            "before the fix, April to September",
            lambda before, month: before & (month >= 4) & (month <= 9),
        ),
        ("after the fix", lambda before, month: ~before),
    )
    for side, selector in sides:
        ramps = {}
        for name in ("morning", "evening"):
            keep = selector(sums[name]["before"], sums[name]["month"])
            ramps[name] = {key: sums[name][key][keep] for key in ("day", "xy", "xx", "yy")}
        rows.append(
            {"series": label, "kind": kind, "side": side, **estimate(ramps=ramps, sign=sign)}
        )
    return rows


def main() -> None:
    """Measure every series and write `tables/timestamp_offsets.parquet`."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(200)
    series = load_series()
    ids = dict(series.select("site", "time_series_id").iter_rows())
    panel = load_panel(extended=True)
    cams = pl.read_parquet(
        WORK_DIR.parent.parent
        / "studies/downloads/reanalysis/CAMS/demand_points/cams_demand_points.parquet",
        columns=["site", "time", "ghi_w_m2", "clear_sky_ghi_w_m2"],
    )
    days_by_site = {
        site: clear_days(cams=cams.filter(pl.col("site") == site)) for site in cams["site"].unique()
    }
    sites_table = series.select("site", "latitude", "longitude")
    rows: list[dict] = []
    for site in MW_PRIMARIES:
        sid = ids[site]
        own = panel.filter(
            pl.col("time_series_id") == sid, pl.col("clean"), ~pl.col("switching_mask")
        ).sort("time")
        latitude, longitude = sites_table.filter(pl.col("site") == site).row(0)[1:]
        rows += measure(
            label=site,
            kind="MW Primary",
            times=own["time"].dt.replace_time_zone(None).to_numpy(),
            values=own["points"].to_numpy().astype(float),
            latitude=float(latitude),
            longitude=float(longitude),
            days=days_by_site[site],
            sign=-1,
        )
        print("measured", site)
    # Metered PV series: internal only, reported as an aggregate.
    metadata = pl.read_parquet(METADATA_PATH).filter(pl.col("time_series_type") == "PV")
    power = scan_power().filter(pl.col("time") <= PANEL_END).collect()
    for k, row in enumerate(metadata.iter_rows(named=True)):
        nearest = (
            sites_table.with_columns(
                distance=(pl.col("latitude") - row["latitude"]) ** 2
                + (pl.col("longitude") - row["longitude"]) ** 2
            )
            .sort("distance")
            .row(0)[0]
        )
        own = power.filter(pl.col("time_series_id") == row["time_series_id"]).sort("time")
        capacity = float(own["power"].quantile(0.99) or 1.0)
        rows += measure(
            label=f"PV-{k + 1}",
            kind="metered PV",
            times=own["time"].dt.replace_time_zone(None).to_numpy(),
            values=100.0 * own["power"].to_numpy().astype(float) / capacity,
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            days=days_by_site[nearest],
            sign=1,
        )
        print("measured a metered PV series")
    table = pl.DataFrame(rows)
    table.write_parquet(WORK_DIR / "tables" / "timestamp_offsets.parquet")
    numbers = [c for c in table.columns if c not in ("series", "kind", "side", "days")]
    columns = ["series", "side", "days", "mean", "mean_low", "mean_high", "morning", "evening"]
    print("\n== offsets in minutes the stored stamps run late: MW Primaries ==")
    print(
        table.filter(pl.col("kind") == "MW Primary")
        .with_columns(pl.col(numbers).round(1))
        .select(*columns, "half_difference", "r_squared")
    )
    pv = table.filter(pl.col("kind") == "metered PV", pl.col("days") > 0)
    print("\n== metered PV series, aggregate over the series (no names, identifiers, or dates) ==")
    print(
        pv.group_by("side")
        .agg(
            series=pl.len(),
            median_mean_offset=pl.col("mean").median(),
            lowest_mean_offset=pl.col("mean").min(),
            highest_mean_offset=pl.col("mean").max(),
            lowest_interval_end=pl.col("mean_low").min(),
            highest_interval_end=pl.col("mean_high").max(),
            median_half_difference=pl.col("half_difference").median(),
        )
        .with_columns(pl.col(pl.Float64).round(1))
        .sort("side")
    )


if __name__ == "__main__":
    main()
