"""Build stage 1a's inputs once: ERA5 and CAMS on the half-hourly grid, and the ENS rows' features.

Writes under `data/studies/per_study/stage1_pilot/inputs/`:

- `analysis_features.parquet`: one row per series and half-hour of ERA5 (and, for the solar farms,
  CAMS) weather, with stage 1a's columns.
- `ens_features/train.parquet` and `ens_features/val/init=<date>.parquet`: the rows the control arm
  was trained and scored on, with the same columns computed from ENS weather.
- `checks.md`: the checks run on both, printed from the data. The script stops if a check fails.

    uv run python studies/stage1_pilot/stage1_build_weather.py
"""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import stage1_inputs as inputs
from studies.solar import cos_zenith_hour_mean

INPUT_DIR: Final[Path] = inputs.INPUTS_DIR

WIND_SPEED_LIMIT: Final[float] = 60.0
"""A wind speed above this, in m s-1, is a decoding fault."""


def _wind_columns(*, heights: tuple[str, ...]) -> list[pl.Expr]:
    """Speed and bearing sine and cosine from components `u<h>` and `v<h>`."""
    columns: list[pl.Expr] = []
    for height in heights:
        u, v = pl.col(f"u{height}"), pl.col(f"v{height}")
        bearing = (pl.arctan2(u, v) * 180.0 / np.pi + 180.0) % 360.0
        columns += [
            ((u**2 + v**2).sqrt()).alias(f"wind_speed_{height}m"),
            (bearing * np.pi / 180.0).sin().alias(f"wind_dir_sin_{height}m"),
            (bearing * np.pi / 180.0).cos().alias(f"wind_dir_cos_{height}m"),
        ]
    return columns


def analysis_features(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return stage 1a's inputs from ERA5 (and CAMS) for every series at every half-hour."""
    pieces = []
    solar = sites.filter(pl.col("technology") == "solar")
    era5 = inputs.solar_hourly(sites=solar)
    cams = inputs.cams_hourly(sites=solar)
    for row in solar.iter_rows(named=True):
        hourly = (
            era5.filter(pl.col("time_series_id") == row["time_series_id"])
            .join(
                cams.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                    "time", "ghi_cams"
                ),
                on="time",
                how="left",
            )
            .sort("time")
        )
        grid = inputs.half_hourly(
            hourly=hourly,
            instants=[
                "temperature_2m",
                "dew_point_temperature_2m",
                "pressure_surface",
                "u10",
                "v10",
            ],
            period=["longwave", "precipitation"],
            irradiance=["ghi", "ghi_cams"],
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        geometry = inputs.solar_geometry(
            stamps=grid["valid_time"], latitude=row["latitude"], longitude=row["longitude"]
        )
        pieces.append(
            grid.join(geometry, on="valid_time", how="left", validate="1:1")
            .with_columns(
                *_wind_columns(heights=("10",)),
                time_series_id=pl.lit(row["time_series_id"]).cast(pl.Int32),
                shortwave=pl.col("ghi"),
                shortwave_cams=pl.col("ghi_cams"),
            )
            .drop("ghi", "ghi_cams", "u10", "v10")
        )
    wind = inputs.wind_hourly(sites=sites.filter(pl.col("technology") == "wind"))
    for row in sites.filter(pl.col("technology") == "wind").iter_rows(named=True):
        hourly = wind.filter(pl.col("time_series_id") == row["time_series_id"]).sort("time")
        grid = inputs.half_hourly(
            hourly=hourly,
            instants=["u10", "v10", "u100", "v100"],
            period=[],
            irradiance=[],
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        pieces.append(
            grid.with_columns(
                *_wind_columns(heights=("10", "100")),
                time_series_id=pl.lit(row["time_series_id"]).cast(pl.Int32),
            ).drop("u10", "v10", "u100", "v100")
        )
    return (
        pl.concat(pieces, how="diagonal")
        .with_columns(*inputs.calendar_columns())
        .sort("time_series_id", "valid_time")
    )


def ens_features(*, frame: pl.DataFrame, geometry: pl.DataFrame) -> pl.DataFrame:
    """Return stage 1a's inputs computed from the ENS weather columns of a fold A frame."""
    columns = [pl.col(source).alias(name) for name, source in inputs.ENS_COLUMNS.items()]
    for height in ("10", "100"):
        bearing = pl.col(f"wind_direction_{height}m") * np.pi / 180.0
        columns += [
            bearing.sin().alias(f"wind_dir_sin_{height}m"),
            bearing.cos().alias(f"wind_dir_cos_{height}m"),
        ]
    return (
        frame.select(*inputs.KEY_COLUMNS, "power", *columns)
        .join(geometry, on=["time_series_id", "valid_time"], how="left")
        .with_columns(*inputs.calendar_columns())
    )


def _geometry(*, sites: pl.DataFrame, analysis: pl.DataFrame) -> pl.DataFrame:
    """Solar zenith and azimuth for each solar farm over the ENS era's valid times."""
    stamps = pl.datetime_range(
        datetime(2024, 4, 1, tzinfo=UTC), datetime(2026, 8, 1, tzinfo=UTC), "30m", eager=True
    )
    pieces = [
        inputs.solar_geometry(
            stamps=stamps, latitude=row["latitude"], longitude=row["longitude"]
        ).with_columns(time_series_id=pl.lit(row["time_series_id"]).cast(pl.Int32))
        for row in sites.filter(pl.col("technology") == "solar").iter_rows(named=True)
    ]
    if analysis.height == 0:
        msg = "the analysis table is empty"
        raise ValueError(msg)
    return pl.concat(pieces)


def check_analysis(*, analysis: pl.DataFrame, sites: pl.DataFrame) -> list[str]:
    """Run the data checks on the ERA5 and CAMS table; return report lines and raise on failure."""
    lines = ["## ERA5 and CAMS on the half-hourly grid", ""]
    counts = analysis.group_by("time_series_id").agg(
        rows=pl.len(),
        first=pl.col("valid_time").min(),
        last=pl.col("valid_time").max(),
        unique_rows=pl.col("valid_time").n_unique(),
    )
    expected = counts.with_columns(
        expected=((pl.col("last") - pl.col("first")).dt.total_minutes() // 30 + 1)
    )
    if expected.filter(
        (pl.col("rows") != pl.col("expected")) | (pl.col("rows") != pl.col("unique_rows"))
    ).height:
        msg = "an analysis series has gaps or duplicate half-hours"
        raise ValueError(msg)
    lines += [
        (
            f"- Series: {counts.height}. Rows per series: "
            f"{sorted(counts['rows'].unique().to_list())}, "
            "each exactly the half-hours from its first to its last stamp, with no duplicate key."
        ),
        (
            f"- Span: {counts['first'].min():%Y-%m-%d %H:%M} to "
            f"{counts['last'].max():%Y-%m-%d %H:%M} UTC."
        ),
    ]
    nulls = analysis.select(pl.all().null_count()).row(0, named=True)
    nans = analysis.select(pl.col(pl.Float32, pl.Float64).is_nan().sum()).row(0, named=True)
    bad = {name: n for name, n in nans.items() if n}
    lines.append(f"- NaN counts: {bad or 'none'}.")
    lines.append(
        "- Null counts by column (a column is null for a series whose technology does not use it, "
        "and "
        "CAMS is null for its first hour): " + ", ".join(f"{k}={v}" for k, v in nulls.items() if v)
    )
    wind_ids = sites.filter(pl.col("technology") == "wind")["time_series_id"].to_list()
    wind_max = analysis.filter(pl.col("time_series_id").is_in(wind_ids)).select(
        pl.col("wind_speed_10m").max(), pl.col("wind_speed_100m").max()
    )
    if wind_max.max_horizontal().item() > WIND_SPEED_LIMIT:
        msg = "implausible wind speed"
        raise ValueError(msg)
    lines.append(
        f"- Wind speed maxima at the wind farms: {wind_max['wind_speed_10m'].item():.1f} m/s at "
        f"10 m, {wind_max['wind_speed_100m'].item():.1f} m/s at 100 m (limit "
        f"{WIND_SPEED_LIMIT:.0f})."
    )
    # Hour-ending convention: an hourly ERA5 irradiance labelled T correlates best with the mean
    # top-of-atmosphere cosine over the hour ending at T. Rebuild the hourly mean from the two
    # half-hours ending at T - 30 min and T, then correlate with that cosine shifted by -1, 0, +1 h.
    row = sites.filter(pl.col("technology") == "solar").row(0, named=True)
    one = analysis.filter(pl.col("time_series_id") == row["time_series_id"]).sort("valid_time")
    hourly = (
        one.filter(pl.col("valid_time").dt.minute() == 0)
        .join(
            one.select(
                (pl.col("valid_time") + pl.duration(minutes=30)).alias("valid_time"),
                pl.col("shortwave").alias("previous_half"),
            ),
            on="valid_time",
            how="inner",
        )
        .with_columns(hour_mean=(pl.col("shortwave") + pl.col("previous_half")) / 2.0)
        .drop_nulls("hour_mean")
    )
    correlations = {}
    for shift in (-1, 0, 1):
        cosine = cos_zenith_hour_mean(
            stamps=hourly["valid_time"].dt.offset_by(f"{shift}h"),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        correlations[shift] = float(np.corrcoef(hourly["hour_mean"].to_numpy(), cosine)[0, 1])
    best = max(correlations, key=lambda k: correlations[k])
    if best != 0:
        msg = f"ERA5 irradiance is not hour-ending: correlations by shift {correlations}"
        raise ValueError(msg)
    lines.append(
        "- Hour-ending convention (one solar farm): correlation of the hour mean rebuilt from the "
        "two half-hours with the top-of-atmosphere cosine of the zenith, hour ending at the label "
        f"shifted by -1, 0, +1 h: "
        f"{', '.join(f'{k:+d} h: {v:.4f}' for k, v in correlations.items())}. "
        "The unshifted label fits best."
    )
    # CAMS against ERA5 on the same hour, same convention.
    both = hourly.join(
        one.filter(pl.col("valid_time").dt.minute() == 0).select(
            "valid_time", pl.col("shortwave_cams")
        ),
        on="valid_time",
    ).drop_nulls("shortwave_cams")
    lines.append(
        f"- CAMS against ERA5 irradiance at the label, same hour: correlation "
        f"{np.corrcoef(both['shortwave'], both['shortwave_cams'])[0, 1]:.3f} over "
        f"{both.height:,} half-hours (CAMS shaped to half-hours by the same rule)."
    )
    lines.append(
        f"- Night-time irradiance: maximum shortwave where the sun is below the horizon at the "
        f"half-hour midpoint (zenith above 95 degrees): "
        f"{analysis.filter(pl.col('solar_zenith') > 95)['shortwave'].max():.3f} W/m2."
    )
    night = analysis.filter(pl.col("solar_zenith") > 95)["shortwave"].max()
    if night is not None and night > 5.0:  # ty: ignore[unsupported-operator]
        msg = "irradiance at night"
        raise ValueError(msg)
    lines.append(
        f"- Highest half-hourly shortwave at a solar farm: {analysis['shortwave'].max():.0f} W/m2 "
        "(top of atmosphere is 1,361 W/m2 normal to the beam)."
    )
    return [*lines, ""]


def check_ens(*, ens: pl.DataFrame, analysis: pl.DataFrame, sites: pl.DataFrame) -> list[str]:
    """Compare ENS member 0 at short lead with ERA5 at the same valid time, by technology."""
    lines = [
        "## ENS rows against ERA5 at the same valid time",
        "",
        (
            "Rows are control-member rows of the validation runs with lead under 36 h. A match in "
            "the "
            "correlation and a small bias confirms the timestamps, units, and conventions line up."
        ),
        "",
        "| technology | column | rows | correlation | mean ENS | mean ERA5 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    joined = (
        ens.filter(
            (pl.col("ensemble_member") == 0)
            & ((pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours() < 36)
        )
        .join(sites.select("time_series_id", "technology"), on="time_series_id")
        .join(analysis, on=["time_series_id", "valid_time"], suffix="_era5", how="inner")
    )
    for technology, columns in inputs.FEATURES.items():
        part = joined.filter(pl.col("technology") == technology)
        for column in columns:
            if column.startswith(("hour", "doy", "solar")):
                continue
            pair = part.select(column, f"{column}_era5").drop_nulls().drop_nans()
            corr = float(np.corrcoef(pair[column], pair[f"{column}_era5"])[0, 1])
            lines.append(
                f"| {technology} | {column} | {pair.height:,} | {corr:.3f} | "
                f"{pair[column].mean():.4g} | {pair[f'{column}_era5'].mean():.4g} |"
            )
    return [*lines, ""]


def main() -> None:
    """Build and check the inputs, skipping files that exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    (INPUT_DIR / "ens_features" / "val").mkdir(parents=True, exist_ok=True)
    sites = inputs.site_table()
    analysis_path = INPUT_DIR / "analysis_features.parquet"
    if not analysis_path.exists():
        analysis = analysis_features(sites=sites)
        analysis.write_parquet(analysis_path.with_suffix(".tmp"))
        analysis_path.with_suffix(".tmp").rename(analysis_path)
    analysis = pl.read_parquet(analysis_path)
    report = check_analysis(analysis=analysis, sites=sites)
    geometry = _geometry(sites=sites, analysis=analysis)
    ids = sites["time_series_id"].to_list()

    train_path = INPUT_DIR / "ens_features" / "train.parquet"
    if not train_path.exists():
        started = time.monotonic()
        train = ens_features(
            frame=pl.read_parquet(inputs.FRAMES_DIR / "train.parquet").filter(
                pl.col("time_series_id").is_in(ids)
            ),
            geometry=geometry,
        )
        train.write_parquet(train_path.with_suffix(".tmp"))
        train_path.with_suffix(".tmp").rename(train_path)
        print(f"train: {train.height:,} rows in {time.monotonic() - started:.0f} s", flush=True)
    for path in sorted((inputs.FRAMES_DIR / "val").glob("init=*.parquet")):
        target = INPUT_DIR / "ens_features" / "val" / path.name
        if target.exists():
            continue
        frame = ens_features(
            frame=pl.read_parquet(path).filter(pl.col("time_series_id").is_in(ids)),
            geometry=geometry,
        )
        frame.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(f"{path.name}: {frame.height:,} rows", flush=True)

    val_files = sorted((INPUT_DIR / "ens_features" / "val").glob("init=*.parquet"))
    sample = pl.concat([pl.read_parquet(p) for p in val_files[::6]])
    report += check_ens(ens=sample, analysis=analysis, sites=sites)
    (INPUT_DIR / "checks.md").write_text("\n".join(["# Input checks", "", *report]) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
