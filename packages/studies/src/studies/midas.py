"""Reading the Met Office MIDAS Open station files, and choosing the stations nearest a site.

**The reader returns what the tidy parquets hold, in the units a study wants, and filters on
nothing MIDAS flags.** The quality-control flag `glbl_irad_amt_q` marks whole stations rather than
bad hours (a station's flag is the same on nearly every row), so a filter on it drops a station's
whole record or none of it. The reader therefore never reads a flag column.

**A station file has three defects the reader repairs, and no others.** Relative humidity reaches
107.5 % and is clipped at 100. Global irradiation dips slightly below zero in a few hours, and is
clipped at zero. A pre-dawn spike, where a station reports more than 5 W m⁻² in an hour that starts
and ends with the sun below the horizon, is set to missing by `null_night_spikes`, which needs the
station's coordinates and so is a separate call.

**The station-to-site mapping is sensitive, so nothing here prints one.** A metered generator's
location must not be narrowed from published output. `select_nearest_stations` returns the
mapping to the calling script, which may report only pooled distance ranges.
"""

from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Final

import polars as pl

from studies.grid_sampling import distance_matrix_km
from studies.solar import zenith

KJ_M2_PER_W_M2_HOUR: Final[float] = 3.6
"""One hour at 1 W m⁻² delivers 3.6 kJ m⁻², so a kJ m⁻² hourly total divided by this is W m⁻²."""

MAX_RELATIVE_HUMIDITY: Final[float] = 100.0
"""The ceiling relative humidity is clipped to; the files carry values up to 107.5."""

NIGHT_SPIKE_THRESHOLD_W_M2: Final[float] = 5.0
"""An hour spent wholly with the sun below the horizon may not report more than this, in W m⁻²."""

HORIZON_ZENITH_DEG: Final[float] = 90.0
"""The apparent solar zenith angle at which the sun's centre, refracted, is on the horizon."""

METADATA_COLUMNS: Final[dict[str, str]] = {
    "src_id": "src_id",
    "station_latitude": "latitude",
    "station_longitude": "longitude",
    "station_elevation": "elevation_m",
}
"""The station-metadata CSV's columns this module keeps, mapped to their tidy names.

The station name is deliberately absent: choosing a station does not need it, and it identifies one.
"""


def _raise_on_duplicate_keys(*, frame: pl.DataFrame) -> None:
    """Raise if a `(src_id, time)` pair appears twice."""
    if frame.select("src_id", "time").is_duplicated().any():
        msg = "the file holds a (src_id, time) pair twice"
        raise ValueError(msg)


def read_radiation(*, path: Path) -> pl.DataFrame:
    """Read the tidy hourly-radiation parquet as global irradiance in W m⁻² per station and hour.

    Args:
        path: `uk_radiation_obs_hourly.parquet`.

    Returns:
        One row per station and hour, sorted by `src_id` and `time`, with `src_id`, `time` (the
        END of the hour, UTC) and `ghi_w_m2` (the mean over that hour, `Float64`, clipped at zero).

    Raises:
        ValueError: If a `(src_id, time)` pair appears twice, or an hour count is not 1.
    """
    raw = pl.read_parquet(path, columns=["src_id", "time", "ob_hour_count", "glbl_irad_amt"])
    if (raw["ob_hour_count"] != 1).any():
        msg = "the radiation file holds a row that does not cover exactly one hour"
        raise ValueError(msg)
    _raise_on_duplicate_keys(frame=raw)
    return raw.select(
        "src_id",
        "time",
        ghi_w_m2=(pl.col("glbl_irad_amt") / KJ_M2_PER_W_M2_HOUR).clip(lower_bound=0.0),
    ).sort("src_id", "time")


def read_hourly_weather(*, path: Path, columns: Sequence[str]) -> pl.DataFrame:
    """Read named columns of the tidy hourly-weather parquet, one row per station and instant.

    Values are as delivered, in the units the parquet's own README states, except relative humidity,
    which is clipped at 100.

    Args:
        path: `uk_hourly_weather_obs.parquet`.
        columns: Value columns to read, such as `air_temperature`. Quality-control flag columns
            are refused, because the flags mark stations rather than hours.

    Returns:
        `src_id`, `time` (an instant, UTC), then `columns`, sorted by `src_id` and `time`.

    Raises:
        ValueError: If a requested column is a quality-control flag or is absent, or a
            `(src_id, time)` pair appears twice.
    """
    flags = [name for name in columns if name.endswith("_q")]
    if flags:
        msg = f"{flags} are quality-control flags, which mark whole stations and are never read"
        raise ValueError(msg)
    raw = pl.read_parquet(path, columns=["src_id", "time", *columns])
    _raise_on_duplicate_keys(frame=raw)
    if "rltv_hum" in columns:
        raw = raw.with_columns(pl.col("rltv_hum").clip(upper_bound=MAX_RELATIVE_HUMIDITY))
    return raw.sort("src_id", "time")


def read_station_metadata(*, path: Path) -> pl.DataFrame:
    """Read a station-metadata CSV, keeping the id, the coordinates and the elevation.

    The file is BADC-CSV: a header block, a line reading `data`, a column-name row, one row per
    station, and a line reading `end data`.

    Args:
        path: A `..._station-metadata.csv` under `MIDAS-OPEN/_station_metadata/`.

    Returns:
        One row per station with `src_id`, `latitude`, `longitude` (degrees) and `elevation_m`.

    Raises:
        ValueError: If the file has no `data` line, or lacks one of `METADATA_COLUMNS`.
    """
    lines = path.read_text().splitlines()
    if "data" not in lines:
        msg = f"{path} has no line reading 'data'"
        raise ValueError(msg)
    start = lines.index("data") + 1
    end = lines.index("end data") if "end data" in lines else len(lines)
    body = "\n".join(lines[start:end])
    raw = pl.read_csv(body.encode(), infer_schema_length=0)
    missing = [name for name in METADATA_COLUMNS if name not in raw.columns]
    if missing:
        msg = f"{path} lacks the columns {missing}"
        raise ValueError(msg)
    return raw.select(
        pl.col(source).cast(pl.String if target == "src_id" else pl.Float64).alias(target)
        for source, target in METADATA_COLUMNS.items()
    )


def null_night_spikes(*, radiation: pl.DataFrame, stations: pl.DataFrame) -> pl.DataFrame:
    """Set `ghi_w_m2` to missing where a station reports sunshine in an hour spent wholly at night.

    An hour is "wholly at night" when the sun's centre, refracted, is on or below the horizon at
    both the hour's start and its end. A sunrise hour and a sunset hour each have the sun above the
    horizon at one end, carry real light, and are kept.

    Args:
        radiation: `read_radiation`'s output.
        stations: One row per station with `src_id`, `latitude` and `longitude`. Every station in
            `radiation` must be present.

    Returns:
        `radiation` with the same rows, and `ghi_w_m2` null on each spike.

    Raises:
        ValueError: If a station in `radiation` has no coordinates.
    """
    unknown = set(radiation["src_id"].unique()) - set(stations["src_id"])
    if unknown:
        msg = f"no coordinates for stations {sorted(unknown)}"
        raise ValueError(msg)
    parts = []
    by_station = radiation.partition_by("src_id", as_dict=True, maintain_order=True)
    for (station,), hours in by_station.items():
        row = stations.filter(pl.col("src_id") == station).row(0, named=True)
        at_hour_end = zenith(
            stamps=hours["time"], latitude=row["latitude"], longitude=row["longitude"]
        )
        at_hour_start = zenith(
            stamps=hours["time"] - timedelta(hours=1),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        below_horizon = (at_hour_end >= HORIZON_ZENITH_DEG) & (at_hour_start >= HORIZON_ZENITH_DEG)
        spike = below_horizon & (hours["ghi_w_m2"].to_numpy() > NIGHT_SPIKE_THRESHOLD_W_M2)
        parts.append(
            hours.with_columns(
                ghi_w_m2=pl.when(pl.Series(spike)).then(None).otherwise(pl.col("ghi_w_m2"))
            )
        )
    return pl.concat(parts).sort("src_id", "time")


def select_nearest_stations(
    *,
    sites: pl.DataFrame,
    stations: pl.DataFrame,
    observed: pl.DataFrame,
    required: pl.DataFrame,
    k: int,
    min_coverage: float,
) -> pl.DataFrame:
    """Choose, for each site, the `k` nearest stations that cover enough of the site's hours.

    **The rule, in full.** Stations are ranked by great-circle distance from the site. A tie in
    distance is broken by the lower `src_id`. A station is
    eligible when it has a value at no less than `min_coverage` of the site's `required` hours.
    The site takes its first `k` eligible stations in rank order. A station nearer than a
    chosen one that fails the coverage test is skipped, and is counted in the `skipped_nearer` of
    every chosen station beyond it. The rule reads no
    score, no forecast and no target value, so it can be written down before any fit.

    **The result is the station-to-site mapping, which is sensitive.** A caller may report only
    pooled distance ranges across sites, never a row.

    Args:
        sites: One row per site with `site`, `latitude` and `longitude`.
        stations: One row per candidate station with `src_id`, `latitude` and `longitude`.
        observed: The `(src_id, time)` pairs where a station has a usable value.
        required: The `(site, time)` pairs the site's rows need, fixed before any scoring.
        k: How many stations to choose per site.
        min_coverage: The share of a site's required hours a station must cover, in (0, 1].

    Returns:
        One row per site and rank (from 1) with `site`, `rank`, `src_id`, `distance_km`,
        `coverage` (the share of the site's required hours the station covers) and
        `skipped_nearer` (how many stations nearer than this row's station failed the coverage
        test), sorted by `site` and `rank`.

    Raises:
        ValueError: If `k` is below 1, `min_coverage` is outside (0, 1], a site has no required
            hour, or a site has fewer than `k` eligible stations.
    """
    if k < 1:
        msg = f"k must be at least 1, not {k}"
        raise ValueError(msg)
    if not 0.0 < min_coverage <= 1.0:
        msg = f"min_coverage must lie in (0, 1], not {min_coverage}"
        raise ValueError(msg)
    observed_keys = observed.select("src_id", "time").unique()
    distances = distance_matrix_km(sites=sites, cells=stations)
    chosen = []
    for site, site_distances in zip(sites["site"], distances, strict=True):
        hours = required.filter(pl.col("site") == site).select("time").unique()
        if hours.height == 0:
            msg = f"site {site} has no required hours"
            raise ValueError(msg)
        ranked = (
            stations.select("src_id")
            .with_columns(distance_km=pl.Series(site_distances))
            .sort("distance_km", "src_id")
        )
        covered = (
            hours.join(observed_keys, on="time", how="inner")
            .group_by("src_id")
            .agg(n_covered=pl.len())
        )
        ranked = ranked.join(covered, on="src_id", how="left").with_columns(
            coverage=pl.col("n_covered").fill_null(0) / hours.height
        )
        eligible = ranked.with_row_index("position").filter(pl.col("coverage") >= min_coverage)
        if eligible.height < k:
            msg = f"site {site} has {eligible.height} eligible stations, fewer than k={k}"
            raise ValueError(msg)
        chosen.append(
            eligible.head(k).select(
                site=pl.lit(site),
                rank=pl.int_range(1, k + 1),
                src_id="src_id",
                distance_km="distance_km",
                coverage="coverage",
                skipped_nearer=pl.col("position") - pl.int_range(0, k),
            )
        )
    return pl.concat(chosen).sort("site", "rank")
