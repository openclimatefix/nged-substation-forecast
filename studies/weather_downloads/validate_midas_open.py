"""Validate the tidy MIDAS Open parquets that `fetch_midas_open.py` wrote.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>, following the
`data-validation` skill's checklist. It reads the two tidy parquets under
`data/studies/weather/MIDAS-OPEN/`, runs each check, prints one report, and writes the same report
as `validation.json` and `validation_report.md` next to the parquets. A run that raises is not
evidence the data is right, and neither is a run that prints: a reader compares each printed
number with what it should be.

**Station coordinates are read from `_station_metadata/` in memory, for solar geometry and for the
nearest-ERA5-cell lookup, and are never printed or written.** The report carries station ids only.

Run it with `uv run python studies/weather_downloads/validate_midas_open.py`.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from paths import WEATHER_DOWNLOADS_DIR
from studies.solar import cos_zenith, extraterrestrial_horizontal, zenith

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("validate_midas_open")

PRODUCT_DIR: Final[Path] = WEATHER_DOWNLOADS_DIR / "MIDAS-OPEN"
ERA5_GRID_PATH: Final[Path] = WEATHER_DOWNLOADS_DIR / "ERA5" / "beam_diffuse_open_meteo.parquet"
"""The ERA5 irradiance and temperature on the study grid, hour-ending for irradiance."""

ERA5_HALF_CELL_DEG: Final[float] = 0.125
"""Half the ERA5 grid spacing: a station is compared only where an ERA5 cell contains it."""

KJ_PER_M2_PER_W_HOUR: Final[float] = 3.6
"""One hour at 1 W/m2 delivers 3.6 kJ/m2."""

DAYTIME_COS_ZENITH: Final[float] = 0.2
"""Hours whose mid-hour sun is higher than about 78 degrees from the vertical count as daytime."""

NIGHT_COS_ZENITH: Final[float] = -0.1
"""Mid-hour sun more than about 6 degrees below the horizon counts as night. A hour label nearer the
horizon can hold twilight, or a few minutes of sun, so it is not tested."""


def _read_station_locations(*, dataset: str) -> dict[str, tuple[float, float]]:
    """Return `{src_id: (latitude, longitude)}` from the private station-metadata CSV.

    Args:
        dataset: `uk-radiation-obs` or `uk-hourly-weather-obs`.

    Returns:
        The coordinates, held in memory only.
    """
    path = next((PRODUCT_DIR / "_station_metadata").glob(f"*{dataset}*station-metadata.csv"))
    lines = path.read_text().split("\ndata\n", maxsplit=1)[1].split("\nend data")[0].splitlines()
    rows = csv.DictReader(lines)
    return {
        row["src_id"]: (float(row["station_latitude"]), float(row["station_longitude"]))
        for row in rows
    }


def _solar_columns(
    *, frame: pl.DataFrame, locations: dict[str, tuple[float, float]]
) -> pl.DataFrame:
    """Add mid-hour `cos_zenith` and the hour-ending top-of-atmosphere flux, per station.

    Args:
        frame: Tidy radiation rows carrying `src_id` and hour-ending `time`.
        locations: Station coordinates from `_read_station_locations`.

    Returns:
        `frame` with `cos_zenith` and `etr_w_m2` (extraterrestrial flux on a horizontal plane at
        the hour's mid-point, W/m2) columns.
    """
    parts = []
    for (src_id,), group in frame.group_by(["src_id"], maintain_order=True):
        latitude, longitude = locations[str(src_id)]
        stamps = group["time"].dt.offset_by("-30m")
        zenith_deg = zenith(stamps=stamps, latitude=latitude, longitude=longitude)
        parts.append(
            group.with_columns(
                cos_zenith=pl.Series(np.cos(np.radians(zenith_deg))),
                etr_w_m2=pl.Series(
                    extraterrestrial_horizontal(stamps=stamps, zenith_deg=zenith_deg)
                ),
            )
        )
    return pl.concat(parts)


def _round(value: Any, digits: int = 3) -> Any:
    """Return `value` rounded if it is a float, else unchanged (for JSON)."""
    return round(float(value), digits) if isinstance(value, float | np.floating) else value


def _gaps(*, frame: pl.DataFrame, top: int = 5) -> list[dict[str, Any]]:
    """Return the `top` longest runs of missing hourly stamps, with the station id and length."""
    ranked = (
        frame.sort("src_id", "time")
        .with_columns(
            step_h=(pl.col("time") - pl.col("time").shift(1)).over("src_id").dt.total_hours()
        )
        .filter(pl.col("step_h") > 1)
        .select(
            "src_id", start=pl.col("time").dt.offset_by("-1h"), missing_hours=pl.col("step_h") - 1
        )
        .sort("missing_hours", descending=True)
        .head(top)
    )
    return [
        {"src_id": r["src_id"], "before": str(r["start"]), "missing_hours": r["missing_hours"]}
        for r in ranked.iter_rows(named=True)
    ]


def _stuck_runs(
    *, frame: pl.DataFrame, value_cols: list[str], min_hours: int, top: int = 5
) -> list:
    """Return the longest runs where every column in `value_cols` repeats exactly, non-null."""
    ranked = (
        frame.sort("src_id", "time")
        .filter(pl.all_horizontal(pl.col(c).is_not_null() for c in value_cols))
        .with_columns(
            run_id=pl.any_horizontal(pl.col(c) != pl.col(c).shift(1) for c in value_cols)
            .cum_sum()
            .over("src_id")
        )
        .group_by("src_id", "run_id")
        .agg(
            length=pl.len(),
            start=pl.col("time").first(),
            **{c: pl.col(c).first() for c in value_cols},
        )
        .filter(pl.col("length") >= min_hours)
        .sort("length", descending=True)
        .head(top)
    )
    return [
        {k: (str(v) if k == "start" else _round(v)) for k, v in r.items() if k != "run_id"}
        for r in ranked.iter_rows(named=True)
    ]


def _step_changes(*, monthly: pl.DataFrame, value: str, top: int = 6) -> list[dict[str, Any]]:
    """Find months where a station's level shifts relative to the other stations.

    For each month the station's monthly mean is compared with the median of the other stations'
    monthly means, which removes weather shared across stations. The shift is the mean of that
    difference over the 12 months from a month on, minus its mean over the 12 months before.

    Args:
        monthly: One row per (`src_id`, `month`) with the monthly mean in column `value`.
        value: The monthly-mean column.
        top: How many stations' largest shifts to return.

    Returns:
        The `top` largest absolute shifts, as dicts.
    """
    medians = monthly.group_by("month").agg(median=pl.col(value).median())
    deviation = (
        monthly.join(medians, on="month")
        .with_columns(dev=pl.col(value) - pl.col("median"))
        .sort("src_id", "month")
    )
    shifts = deviation.with_columns(
        after=pl.col("dev").rolling_mean(12, min_samples=9).shift(-11).over("src_id"),
        before=pl.col("dev").rolling_mean(12, min_samples=9).shift(1).over("src_id"),
    ).with_columns(shift=pl.col("after") - pl.col("before"))
    ranked = (
        shifts.filter(pl.col("shift").is_not_null())
        .with_columns(abs_shift=pl.col("shift").abs())
        .sort("abs_shift", descending=True)
        .unique(subset=["src_id"], keep="first", maintain_order=True)
        .head(top)
    )
    return [
        {"src_id": r["src_id"], "month": str(r["month"].date()), "shift": _round(r["shift"])}
        for r in ranked.iter_rows(named=True)
    ]


def validate_radiation() -> dict[str, Any]:
    """Run every applicable check on the tidy radiation parquet.

    Returns:
        A JSON-ready dict, one key per check.
    """
    raw = pl.read_parquet(PRODUCT_DIR / "uk_radiation_obs_hourly.parquet")
    locations = _read_station_locations(dataset="uk-radiation-obs")
    frame = _solar_columns(frame=raw, locations=locations).with_columns(
        ghi_w_m2=pl.col("glbl_irad_amt") / KJ_PER_M2_PER_W_HOUR
    )
    frame = frame.with_columns(kt=pl.col("ghi_w_m2") / pl.col("etr_w_m2"))
    day = frame.filter(pl.col("cos_zenith") > DAYTIME_COS_ZENITH)
    night = frame.filter(pl.col("cos_zenith") < NIGHT_COS_ZENITH)
    out: dict[str, Any] = {"rows": frame.height}

    out["duplicate_keys"] = frame.group_by("src_id", "time").len().filter(pl.col("len") > 1).height
    out["time_dtype"] = str(frame.schema["time"])
    out["nulls_in_glbl_irad_amt"] = frame["glbl_irad_amt"].null_count()
    out["nan_in_glbl_irad_amt"] = int(frame["glbl_irad_amt"].is_nan().sum())
    out["non_null_rows_per_other_column"] = {
        c: frame[c].drop_nulls().len()
        for c in ("difu_irad_amt", "direct_irad", "irad_bal_amt", "glbl_s_lat_irad_amt")
        if c in frame.columns
    }
    out["longest_gaps_hours"] = _gaps(frame=frame)
    out["completeness_pct_of_hours_between_first_and_last"] = {
        r["src_id"]: _round(100 * r["n"] / r["expected"], 1)
        for r in frame.group_by("src_id")
        .agg(n=pl.len(), t0=pl.col("time").min(), t1=pl.col("time").max())
        .with_columns(expected=(pl.col("t1") - pl.col("t0")).dt.total_hours() + 1)
        .sort("src_id")
        .iter_rows(named=True)
    }

    out["value_range_kj_m2"] = {
        "min": _round(frame["glbl_irad_amt"].min()),
        "max": _round(frame["glbl_irad_amt"].max()),
        "negative_rows": frame.filter(pl.col("glbl_irad_amt") < 0).height,
    }
    out["night_time"] = {
        "night_rows": night.height,
        "max_w_m2": _round(night["ghi_w_m2"].max()),
        "rows_above_5_w_m2": night.filter(pl.col("ghi_w_m2") > 5).height,
        "rows_above_5_w_m2_listed": [
            {
                "src_id": r["src_id"],
                "time": str(r["time"]),
                "kj_m2": r["glbl_irad_amt"],
                "flag": r["glbl_irad_amt_q"],
            }
            for r in night.filter(pl.col("ghi_w_m2") > 5)
            .sort("ghi_w_m2", descending=True)
            .head(10)
            .iter_rows(named=True)
        ],
        "rows_above_5_w_m2_by_station": {
            r["src_id"]: r["n"]
            for r in night.filter(pl.col("ghi_w_m2") > 5)
            .group_by("src_id")
            .agg(n=pl.len())
            .sort("src_id")
            .iter_rows(named=True)
        },
    }
    out["daytime_clearness_index"] = {
        "rows": day.height,
        "p50": _round(day["kt"].quantile(0.5)),
        "p99": _round(day["kt"].quantile(0.99)),
        "max": _round(day["kt"].max()),
        "rows_above_1_0": day.filter(pl.col("kt") > 1.0).height,
        "rows_above_1_1": day.filter(pl.col("kt") > 1.1).height,
        "high_sun_zero_rows": day.filter(
            (pl.col("cos_zenith") > 0.4) & (pl.col("glbl_irad_amt") == 0)
        ).height,
        "p99_by_station": {
            r["src_id"]: _round(r["p99"])
            for r in day.group_by("src_id")
            .agg(p99=pl.col("kt").quantile(0.99))
            .sort("src_id")
            .iter_rows(named=True)
        },
    }
    out["daytime_rows_exceeding_extraterrestrial_flux_by_5_w_m2"] = day.filter(
        pl.col("ghi_w_m2") > pl.col("etr_w_m2") + 5
    ).height

    profile = (
        frame.group_by(pl.col("time").dt.hour().alias("hour_label_utc"))
        .agg(mean_w_m2=pl.col("ghi_w_m2").mean(), mean_etr_w_m2=pl.col("etr_w_m2").mean())
        .sort("hour_label_utc")
    )
    out["hour_of_day_profile"] = {
        int(r["hour_label_utc"]): [_round(r["mean_w_m2"], 1), _round(r["mean_etr_w_m2"], 1)]
        for r in profile.iter_rows(named=True)
    }
    out["label_convention"] = _label_convention(frame=frame, locations=locations)
    out["stuck_daytime_runs"] = _stuck_runs(
        frame=day.filter(pl.col("glbl_irad_amt") > 0), value_cols=["glbl_irad_amt"], min_hours=3
    )

    monthly = (
        day.group_by(pl.col("time").dt.truncate("1mo").alias("month"), "src_id")
        .agg(kt=pl.col("kt").mean(), n=pl.len())
        .filter(pl.col("n") >= 100)
    )
    annual: dict[str, dict[str, float]] = {}
    for r in (
        monthly.group_by("src_id", pl.col("month").dt.year().alias("year"))
        .agg(kt=pl.col("kt").mean())
        .sort("src_id", "year")
        .iter_rows(named=True)
    ):
        annual.setdefault(r["src_id"], {})[str(r["year"])] = _round(r["kt"])
    out["annual_mean_clearness_index_by_station"] = annual
    out["largest_level_shifts_vs_other_stations_kt"] = _step_changes(monthly=monthly, value="kt")
    out["qc_flag_summary"] = _qc_summary(
        frame=frame, flag="glbl_irad_amt_q", extra=day, extra_value="kt"
    )
    out["comparison_with_era5"] = _compare_radiation_with_era5(frame=frame, locations=locations)
    return out


def _label_convention(*, frame: pl.DataFrame, locations: dict[str, tuple[float, float]]) -> dict:
    """Return, per station, the label shift (hours) that best correlates the data with sun height.

    A radiation total labelled `T` should track the top-of-atmosphere flux over the hour ending at
    `T`, so the best shift is 0. A start-of-hour label would peak at a shift of -1 or +1 hours.
    """
    result: dict[str, Any] = {}
    for (src_id,), group in frame.group_by(["src_id"], maintain_order=True):
        latitude, longitude = locations[str(src_id)]
        best_shift, best_corr = 0, -2.0
        for shift in (-2, -1, 0, 1, 2):
            shifted = group["time"].dt.offset_by(f"{shift}h").dt.offset_by("-30m")
            cosz = cos_zenith(
                zenith_deg=zenith(stamps=shifted, latitude=latitude, longitude=longitude)
            )
            corr = float(np.corrcoef(group["ghi_w_m2"].fill_null(0).to_numpy(), cosz)[0, 1])
            if corr > best_corr:
                best_shift, best_corr = shift, corr
        result[str(src_id)] = {"best_shift_hours": best_shift, "correlation": _round(best_corr)}
    return result


def _qc_summary(
    *, frame: pl.DataFrame, flag: str, extra: pl.DataFrame | None = None, extra_value: str = ""
) -> list[dict[str, Any]]:
    """Return, per QC flag value, the row count and (optionally) the daytime clearness index."""
    counts = frame.group_by(flag).agg(rows=pl.len()).sort("rows", descending=True)
    result = []
    for r in counts.iter_rows(named=True):
        entry: dict[str, Any] = {"flag": r[flag], "rows": r["rows"]}
        if extra is not None and extra_value:
            subset = extra.filter(
                pl.col(flag).is_null() if r[flag] is None else pl.col(flag) == r[flag]
            )
            entry["daytime_rows"] = subset.height
            entry["daytime_mean_kt"] = _round(subset[extra_value].mean())
            entry["daytime_kt_above_1_1"] = subset.filter(pl.col(extra_value) > 1.1).height
        result.append(entry)
    return result


def _nearest_era5_cells(*, locations: dict[str, tuple[float, float]], stations: list[str]) -> dict:
    """Map each station to the ERA5 cell containing it, or leave it out when none does."""
    cells = pl.read_parquet(ERA5_GRID_PATH, columns=["latitude", "longitude"]).unique()
    lats, lons = cells["latitude"].to_numpy(), cells["longitude"].to_numpy()
    mapping: dict[str, tuple[float, float]] = {}
    for src_id in stations:
        latitude, longitude = locations[src_id]
        distance = np.maximum(np.abs(lats - latitude), np.abs(lons - longitude))
        i = int(np.argmin(distance))
        if distance[i] <= ERA5_HALF_CELL_DEG:
            mapping[src_id] = (float(lats[i]), float(lons[i]))
    return mapping


def _compare_radiation_with_era5(
    *, frame: pl.DataFrame, locations: dict[str, tuple[float, float]]
) -> dict[str, Any]:
    """Compare hour-ending global irradiance with the ERA5 cell that contains each station."""
    stations = sorted(frame["src_id"].unique().to_list())
    mapping = _nearest_era5_cells(locations=locations, stations=stations)
    era5 = pl.read_parquet(ERA5_GRID_PATH)
    result: dict[str, Any] = {"stations_compared": len(mapping)}
    for src_id, (lat, lon) in mapping.items():
        cell = era5.filter((pl.col("latitude") == lat) & (pl.col("longitude") == lon)).select(
            "time", "ghi_w_m2"
        )
        joined = (
            frame.filter(pl.col("src_id") == src_id)
            .select("time", "glbl_irad_amt", "cos_zenith")
            .join(cell, on="time", how="inner")
            .filter(pl.col("cos_zenith") > DAYTIME_COS_ZENITH)
            .with_columns(midas=pl.col("glbl_irad_amt") / KJ_PER_M2_PER_W_HOUR)
            .drop_nulls(["midas", "ghi_w_m2"])
        )
        result[src_id] = {
            "daytime_hours": joined.height,
            "correlation": _round(float(np.corrcoef(joined["midas"], joined["ghi_w_m2"])[0, 1])),
            "mean_midas_w_m2": _round(joined["midas"].mean(), 1),
            "mean_era5_w_m2": _round(joined["ghi_w_m2"].mean(), 1),
            "mae_w_m2": _round((joined["midas"] - joined["ghi_w_m2"]).abs().mean(), 1),
        }
    return result


def validate_weather() -> dict[str, Any]:
    """Run every applicable check on the tidy hourly-weather parquet.

    Returns:
        A JSON-ready dict, one key per check.
    """
    frame = pl.read_parquet(PRODUCT_DIR / "uk_hourly_weather_obs.parquet")
    out: dict[str, Any] = {"rows": frame.height}
    hourly_ids = (
        frame.group_by("src_id")
        .agg(share=pl.col("time").dt.hour().n_unique())
        .filter(pl.col("share") > 12)["src_id"]
        .to_list()
    )
    out["stations_with_more_than_daily_rows"] = len(hourly_ids)
    out["duplicate_keys"] = frame.group_by("src_id", "time").len().filter(pl.col("len") > 1).height
    out["time_dtype"] = str(frame.schema["time"])
    hourly = frame.filter(pl.col("src_id").is_in(hourly_ids))
    out["longest_gaps_hours_hourly_stations"] = _gaps(frame=hourly)
    out["value_ranges"] = {
        c: [_round(frame[c].min()), _round(frame[c].max())]
        for c in (
            "wind_speed_m_s",
            "wind_direction",
            "q10mnt_mxgst_spd",
            "air_temperature",
            "dewpoint",
            "rltv_hum",
            "msl_pressure",
            "cld_ttl_amt_id",
            "wmo_hr_sun_dur",
        )
        if c in frame.columns
    }
    out["null_counts"] = {
        c: int(frame[c].null_count())
        for c in frame.columns
        if c.endswith(("speed", "direction", "temperature", "_m_s"))
    }
    out["nan_counts"] = {
        c: int(frame[c].is_nan().sum()) for c in frame.columns if frame.schema[c].is_float()
    }
    out["wind_speed_unit_id_counts"] = {
        str(r["wind_speed_unit_id"]): r["n"]
        for r in frame.group_by("wind_speed_unit_id").agg(n=pl.len()).iter_rows(named=True)
    }
    wind = frame.filter(pl.col("wind_speed_m_s").is_not_null())
    direction = wind.filter(pl.col("wind_direction").is_not_null())
    out["wind"] = {
        "stations_reporting": sorted(wind["src_id"].unique().to_list()),
        "rows": wind.height,
        "zero_speed_share": _round(
            wind.filter(pl.col("wind_speed_m_s") == 0).height / wind.height, 4
        ),
        "zero_speed_with_nonzero_direction": wind.filter(
            (pl.col("wind_speed_m_s") == 0) & (pl.col("wind_direction") > 0)
        ).height,
        "direction_10_degree_bins_occupied": direction.select(
            (pl.col("wind_direction") // 10).n_unique()
        ).item(),
        "direction_exactly_360": direction.filter(pl.col("wind_direction") == 360).height,
        "gust_below_mean_speed_rows": frame.filter(
            pl.col("q10mnt_mxgst_spd") * 0.514444 < pl.col("wind_speed_m_s") - 0.6
        ).height,
        "mean_speed_m_s_by_station": {
            r["src_id"]: _round(r["m"], 2)
            for r in wind.group_by("src_id")
            .agg(m=pl.col("wind_speed_m_s").mean())
            .sort("src_id")
            .iter_rows(named=True)
        },
    }
    out["temperature"] = {
        "dewpoint_above_air_temperature_by_more_than_0_5_rows": frame.filter(
            pl.col("dewpoint") > pl.col("air_temperature") + 0.5
        ).height,
        "stations_reporting": sorted(
            frame.filter(pl.col("air_temperature").is_not_null())["src_id"].unique().to_list()
        ),
    }
    out["stuck_runs"] = {
        "air_temperature_24h_or_more": _stuck_runs(
            frame=hourly, value_cols=["air_temperature"], min_hours=24
        ),
        "wind_speed_and_direction_12h_or_more_nonzero": _stuck_runs(
            frame=hourly.filter(pl.col("wind_speed_m_s") > 0),
            value_cols=["wind_speed_m_s", "wind_direction"],
            min_hours=12,
        ),
    }
    out["hour_of_day_mean_air_temperature_c"] = {
        int(r["h"]): _round(r["t"], 2)
        for r in hourly.group_by(pl.col("time").dt.hour().alias("h"))
        .agg(t=pl.col("air_temperature").mean())
        .sort("h")
        .iter_rows(named=True)
    }
    temp_monthly = (
        hourly.group_by(pl.col("time").dt.truncate("1mo").alias("month"), "src_id")
        .agg(t=pl.col("air_temperature").mean(), n=pl.col("air_temperature").count())
        .filter(pl.col("n") >= 500)
    )
    out["largest_level_shifts_air_temperature_c"] = _step_changes(monthly=temp_monthly, value="t")
    wind_monthly = (
        wind.filter(pl.col("src_id").is_in(hourly_ids))
        .group_by(pl.col("time").dt.truncate("1mo").alias("month"), "src_id")
        .agg(w=pl.col("wind_speed_m_s").mean(), n=pl.len())
        .filter(pl.col("n") >= 500)
    )
    out["largest_level_shifts_wind_speed_m_s"] = _step_changes(monthly=wind_monthly, value="w")
    out["qc_flag_summary"] = {
        flag: _qc_summary(frame=frame, flag=flag)
        for flag in ("air_temperature_q", "wind_speed_q", "wind_direction_q")
        if flag in frame.columns
    }
    out["flag_106_share_of_rows_by_station"] = {
        flag: {
            r["src_id"]: _round(r["share"], 3)
            for r in frame.group_by("src_id")
            .agg(share=(pl.col(flag) == 106).mean())
            .filter(pl.col("share") > 0.01)
            .sort("src_id")
            .iter_rows(named=True)
        }
        for flag in ("air_temperature_q", "wind_speed_q")
    }
    out["qc_flag_vs_value"] = {
        "air_temperature_by_flag": _qc_value_summary(
            frame=frame, value="air_temperature", flag="air_temperature_q"
        ),
        "wind_speed_m_s_by_flag": _qc_value_summary(
            frame=frame, value="wind_speed_m_s", flag="wind_speed_q"
        ),
    }
    out["comparison_with_era5_temperature"] = _compare_temperature_with_era5(frame=hourly)
    return out


def _qc_value_summary(*, frame: pl.DataFrame, value: str, flag: str) -> list[dict[str, Any]]:
    """Return, per QC flag value, the count and the min, mean, max of `value`."""
    return [
        {
            "flag": r[flag],
            "rows": r["n"],
            "min": _round(r["lo"]),
            "mean": _round(r["m"]),
            "max": _round(r["hi"]),
        }
        for r in frame.group_by(flag)
        .agg(n=pl.len(), lo=pl.col(value).min(), m=pl.col(value).mean(), hi=pl.col(value).max())
        .sort("n", descending=True)
        .iter_rows(named=True)
    ]


def _compare_temperature_with_era5(*, frame: pl.DataFrame) -> dict[str, Any]:
    """Compare instantaneous air temperature with the ERA5 cell containing each station."""
    locations = _read_station_locations(dataset="uk-hourly-weather-obs")
    stations = sorted(frame["src_id"].unique().to_list())
    mapping = _nearest_era5_cells(locations=locations, stations=stations)
    era5 = pl.read_parquet(ERA5_GRID_PATH)
    result: dict[str, Any] = {"stations_compared": len(mapping)}
    for src_id, (lat, lon) in mapping.items():
        cell = era5.filter((pl.col("latitude") == lat) & (pl.col("longitude") == lon)).select(
            "time", "temp_c"
        )
        joined = (
            frame.filter(pl.col("src_id") == src_id)
            .select("time", "air_temperature")
            .join(cell, on="time", how="inner")
            .drop_nulls()
        )
        if joined.height < 1000:
            continue
        difference = joined["air_temperature"] - joined["temp_c"]
        result[src_id] = {
            "hours": joined.height,
            "correlation": _round(
                float(np.corrcoef(joined["air_temperature"], joined["temp_c"])[0, 1])
            ),
            "mean_midas_minus_era5_c": _round(difference.mean()),
            "mae_c": _round(difference.abs().mean()),
        }
    return result


def main() -> int:
    """Run both validations, print the report, and write `validation.json` and the markdown.

    Returns:
        The process exit code, 0.
    """
    report = {"uk-radiation-obs": validate_radiation(), "uk-hourly-weather-obs": validate_weather()}
    text = json.dumps(report, indent=2, default=str)
    (PRODUCT_DIR / "validation.json").write_text(text)
    (PRODUCT_DIR / "validation_report.md").write_text(f"```json\n{text}\n```\n")
    print(text)
    _LOG.info("wrote validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
