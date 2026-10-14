"""Validate the CAMS irradiance downloaded at the 22 demand series' locations.

Run after `fetch_cams_demand_points.py` finishes. It applies the `data-validation` skill's checks to
`cams_demand_points.parquet` and writes `validation.json` beside it:

- the exact row count (22 series times every hour from 2019-09-01 01:00 to 2026-09-11 00:00), and
  gaps and duplicate keys per series;
- nulls, NaN, negative values, and a global horizontal irradiance above the clear-sky value;
- the mean daily profile by hour of day (one smooth daytime peak near local solar noon);
- the timestamp convention, by correlating the hourly irradiance with ERA5's hourly shortwave
  at shifts of -1, 0, and +1 hour: the unshifted label (hour-ending) should correlate best;
- steps in the monthly mean at any series.

No coordinate is read or written: the series are the labels D1 to D22.

    uv run python studies/weather_downloads/validate_cams_demand_points.py \
        --era5 <era5 features> --series <private series table>

`--era5` is experiment 00003's half-hourly ERA5 features table (`era5_features.parquet`), whose
`shortwave` holds an hour-ending mean over both half-hours of the hour. `--series` is the private
table that maps each label D1 to D22 to its `time_series_id`; it is only joined, never printed.
"""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import product_dir_for

CAMS_DIR: Final[Path] = product_dir_for(product="CAMS") / "demand_points"
FIRST: Final[datetime] = datetime(2019, 9, 1, 1, tzinfo=UTC)
LAST: Final[datetime] = datetime(2026, 9, 11, 0, tzinfo=UTC)
SERIES: Final[int] = 22


def main() -> None:
    """Run every check, print the findings, and write `validation.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True)
    arguments = parser.parse_args()
    frame = pl.read_parquet(CAMS_DIR / "cams_demand_points.parquet")
    expected_hours = int((LAST - FIRST) / timedelta(hours=1)) + 1
    per_series = frame.group_by("site").len()
    duplicate_keys = int(frame.select(["site", "time"]).is_duplicated().sum())
    findings: dict[str, object] = {
        "expected_rows": SERIES * expected_hours,
        "rows": frame.height,
        "series": frame["site"].n_unique(),
        "series_with_wrong_row_count": per_series.filter(pl.col("len") != expected_hours)[
            "site"
        ].to_list(),
        "duplicate_keys": duplicate_keys,
        "nulls": {c: frame[c].null_count() for c in frame.columns},
        "nans": {
            c: int(frame[c].is_nan().sum()) for c in frame.columns if frame[c].dtype.is_float()
        },
        "negative_ghi": int((frame["ghi_w_m2"] < 0).sum()),
        "ghi_above_clear_sky_by_more_than_1pct": int(
            (frame["ghi_w_m2"] > 1.01 * frame["clear_sky_ghi_w_m2"] + 1.0).sum()
        ),
        "ghi_max_w_m2": float(frame.select(pl.col("ghi_w_m2").max()).item()),
    }
    profile = (
        frame.with_columns(hour=pl.col("time").dt.hour())
        .group_by("hour")
        .agg(pl.col("ghi_w_m2").mean())
        .sort("hour")
    )
    findings["mean_ghi_by_utc_hour_of_label"] = {
        int(h): round(float(v), 1) for h, v in profile.iter_rows()
    }
    findings["peak_hour_utc_of_label"] = int(profile.sort("ghi_w_m2", descending=True)["hour"][0])

    series = pl.read_parquet(arguments.series, columns=["site", "time_series_id"])
    era5 = pl.read_parquet(
        arguments.era5, columns=["time_series_id", "valid_time", "shortwave"]
    ).join(series, on="time_series_id")
    hourly = (
        era5.with_columns(
            time=pl.col("valid_time").dt.offset_by("-1m").dt.truncate("1h").dt.offset_by("1h")
        )
        .group_by("site", "time")
        .agg(pl.col("shortwave").mean().alias("era5"))
    )
    correlations = {}
    for shift in (-1, 0, 1):
        shifted = frame.with_columns(time=pl.col("time").dt.offset_by(f"{shift}h"))
        joined = shifted.join(hourly, on=["site", "time"], how="inner")
        correlations[f"cams_label_shifted_{shift:+d}h"] = float(
            joined.select(pl.corr("ghi_w_m2", "era5")).item()
        )
    findings["correlation_with_era5_hourly_shortwave"] = correlations
    findings["best_shift_hours"] = max(
        (-1, 0, 1), key=lambda k: correlations[f"cams_label_shifted_{k:+d}h"]
    )

    monthly = (
        frame.with_columns(month=pl.col("time").dt.truncate("1mo"))
        .group_by(["site", "month"])
        .agg(pl.col("ghi_w_m2").mean().alias("mean"))
        .sort("site", "month")
    )
    jumps = monthly.with_columns(
        ratio=pl.col("mean") / pl.col("mean").shift(12).over("site")
    ).filter(pl.col("ratio").is_not_null())
    findings["largest_year_on_year_monthly_ratio"] = float(
        jumps.select(pl.col("ratio").max()).item()
    )
    findings["smallest_year_on_year_monthly_ratio"] = float(
        jumps.select(pl.col("ratio").min()).item()
    )
    (CAMS_DIR / "validation.json").write_text(json.dumps(findings, indent=2, default=str) + "\n")
    print(json.dumps(findings, indent=2, default=str))


if __name__ == "__main__":
    main()
