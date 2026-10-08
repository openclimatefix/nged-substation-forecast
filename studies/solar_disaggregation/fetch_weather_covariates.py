"""Download hourly ERA5 wind speed and air temperature at the 18 public CAMS grid points.

The solar separation of `stage2_synthetic_separation.py` treats an aggregate's non-solar output as
a function of the calendar. Wind generation and heating demand also follow the weather, and that
weather correlates with cloud, so the script fetches two covariates to test whether adding them to
the separation removes the bias: the wind speed at 100 m and the air temperature at 2 m, from
Open-Meteo's archive of the ERA5 reanalysis. The 18 points are the public grid points of the CAMS
download (`inputs.grid_point_ids`), not the private trial-area box.

One request per point, each written to its own file before the next is sent, so a rerun skips the
points already fetched. A full run is 18 requests and about 160,000 values.

Run: `uv run python studies/solar_disaggregation/fetch_weather_covariates.py`.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
from inputs import INPUTS_DIR, grid_point_positions

URL: Final[str] = "https://archive-api.open-meteo.com/v1/archive"
START_DATE: Final[str] = "2025-08-31"
END_DATE: Final[str] = "2026-09-02"
"""A day either side of the study window, so every hour of the window is present."""
MODEL: Final[str] = "era5"
VARIABLES: Final[tuple[str, ...]] = ("wind_speed_100m", "temperature_2m")
MAX_ATTEMPTS: Final[int] = 5
PAUSE_SECONDS: Final[float] = 2.0
COVARIATES_DIR: Final[Path] = INPUTS_DIR / "weather_covariates"
COMBINED_PATH: Final[Path] = INPUTS_DIR / "open_meteo_era5_gb_points.parquet"


def fetch_point(*, point_id: str, latitude: float, longitude: float) -> pl.DataFrame:
    """Fetch one point's hourly series, or raise after `MAX_ATTEMPTS` failures.

    Args:
        point_id: The identifier written to the `point_id` column.
        latitude: The point's latitude, in degrees north.
        longitude: The point's longitude, in degrees east.

    Returns:
        One row per hour, with the point's identifier, the time, and each variable.
    """
    query = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": ",".join(VARIABLES),
            "models": MODEL,
            "wind_speed_unit": "ms",
            "timezone": "GMT",
        }
    )
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(f"{URL}?{query}", timeout=120) as response:
                payload = json.loads(response.read())
            break
        except urllib.error.URLError, TimeoutError:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(2.0**attempt)
    hourly = payload["hourly"]
    return pl.DataFrame(
        {
            "point_id": point_id,
            "time": pl.Series(hourly["time"]).str.to_datetime("%Y-%m-%dT%H:%M", time_zone="UTC"),
            "wind_speed_100m_m_s": pl.Series(hourly["wind_speed_100m"], dtype=pl.Float32),
            "temperature_2m_c": pl.Series(hourly["temperature_2m"], dtype=pl.Float32),
        }
    )


def main() -> None:
    """Fetch every point not yet cached, then write the combined file and a lineage note."""
    COVARIATES_DIR.mkdir(parents=True, exist_ok=True)
    for row in grid_point_positions().iter_rows(named=True):
        path = COVARIATES_DIR / f"{row['point_id']}.parquet"
        if path.exists():
            print(f"{row['point_id']}: cached, skipping")
            continue
        frame = fetch_point(
            point_id=row["point_id"], latitude=row["latitude"], longitude=row["longitude"]
        )
        partial = path.with_suffix(".parquet.partial")
        frame.write_parquet(partial)
        partial.rename(path)
        print(f"{row['point_id']}: {frame.height} hours")
        time.sleep(PAUSE_SECONDS)
    combined = pl.concat(
        [pl.read_parquet(path) for path in sorted(COVARIATES_DIR.glob("*.parquet"))]
    )
    combined.write_parquet(COMBINED_PATH)
    lineage = {
        "source": URL,
        "model": MODEL,
        "variables": list(VARIABLES),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "points": combined["point_id"].n_unique(),
        "rows": combined.height,
        "nulls": combined.null_count().row(0),
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "note": "Wind speed is instantaneous at `time`; temperature is instantaneous at `time`.",
    }
    (INPUTS_DIR / "open_meteo_era5_gb_points_lineage.json").write_text(
        json.dumps(lineage, indent=2)
    )
    print(lineage)


if __name__ == "__main__":
    main()
