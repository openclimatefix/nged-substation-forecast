"""Build the joined frame every arm of the beam/diffuse experiment reads, from one source.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>. It is deliberately outside
the Dagster asset graph and writes nothing the rest of the repo reads. `studies.pv_dataset` holds
the readers, the filters and the conventions, and `studies.arm_runner` then chooses which irradiance
columns to show the model.

Run it with `uv run --with netcdf4 python studies/beam_diffuse_split/build_dataset.py`.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import polars as pl
from studies.pv_dataset import (
    MIN_CAMS_RELIABILITY,
    MIN_SOLAR_ELEVATION_DEGREES,
    add_separation_models,
    add_solar_geometry,
    add_synthetic_control_target,
    drop_false_zeros,
    drop_outages_and_spikes,
    nearest_era5_cell,
    pv_sites,
    read_cams,
    read_era5,
    read_open_meteo_point,
    solar_hourly_power,
)
from studies.sources import PER_SITE_SOURCES, SOURCE_CHOICES, STUDY_INPUTS_DIR, SourceType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("build_dataset")


def output_path_for(*, dataset_name: str) -> Path:
    """Return where one built frame is written.

    Args:
        dataset_name: The irradiance source, plus any suffix distinguishing a variant build from
            the main one for the same source.

    Returns:
        The parquet path every arm of that run reads.
    """
    return STUDY_INPUTS_DIR / f"beam_diffuse_dataset_{dataset_name}.parquet"


def main() -> int:
    """Build the joined frame for the irradiance source named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE_CHOICES, default="open-meteo")
    parser.add_argument(
        "--min-cams-reliability",
        type=float,
        default=MIN_CAMS_RELIABILITY,
        help="Drop CAMS hours flagged below this fraction. Zero keeps every hour.",
    )
    parser.add_argument(
        "--point-temporal",
        choices=("hourly", "instant"),
        default="hourly",
        help=(
            "Which UKV columns the arms see. Pair it with --suffix, or the variant build "
            "overwrites the main one."
        ),
    )
    parser.add_argument(
        "--first-date",
        default=None,
        help=(
            "Drop rows before this YYYY-MM-DD. Pair it with --suffix. What it exists for is "
            "running a source over the span another source can be checked against."
        ),
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Appended to the output filename, so a variant build does not overwrite the main one.",
    )
    arguments = parser.parse_args()
    source: SourceType = arguments.source

    sites = pv_sites()
    _LOG.info("using %d PV sites, irradiance source %s", sites.height, source)

    gridded = read_era5(source="open-meteo" if source in PER_SITE_SOURCES else source)
    _LOG.info(
        "gridded fields: %d rows, %s to %s",
        gridded.height,
        gridded["time"].min(),
        gridded["time"].max(),
    )

    power = drop_outages_and_spikes(power=solar_hourly_power(sites=sites), sites=sites)
    _LOG.info("hourly power after outage and spike filtering: %d rows", power.height)

    sites_with_cells = nearest_era5_cell(sites=sites, era5=gridded)
    _LOG.info(
        "the %d sites resolve to %d distinct ERA5 grid cells",
        sites_with_cells.height,
        sites_with_cells.select("cell_latitude", "cell_longitude").n_unique(),
    )
    # Every source needs the gridded frame's air temperature, which is a shared feature rather than
    # an irradiance one, so the gridded join runs for the CAMS build too and only its two irradiance
    # columns are then replaced.
    joined = (
        power.join(sites_with_cells, on=["site", "effective_capacity_mw"])
        .join(
            gridded,
            left_on=["time", "cell_latitude", "cell_longitude"],
            right_on=["time", "latitude", "longitude"],
            how="inner",
        )
        .drop("time_series_id")
    )
    if source in PER_SITE_SOURCES:
        per_site = (
            read_cams(min_reliability=arguments.min_cams_reliability)
            if source == "cams"
            else read_open_meteo_point(source=source, temporal=arguments.point_temporal)
        )
        joined = joined.drop("ghi_w_m2", "bhi_w_m2").join(
            per_site, on=["site", "time"], how="inner"
        )
    _LOG.info("after joining irradiance: %d rows", joined.height)

    with_geometry = add_solar_geometry(joined=drop_false_zeros(joined=joined))
    daylight = with_geometry.filter(pl.col("solar_elevation_deg") > MIN_SOLAR_ELEVATION_DEGREES)
    _LOG.info("daylight rows: %d", daylight.height)

    dataset = add_synthetic_control_target(frame=add_separation_models(frame=daylight)).drop(
        "cell_latitude", "cell_longitude", "latitude", "longitude"
    )
    # The filter runs after the synthetic control target, so a shorter span keeps the same seeded
    # noise on the rows it shares with the full build rather than redrawing it.
    if arguments.first_date is not None:
        before = dataset.height
        dataset = dataset.filter(
            pl.col("time")
            >= pl.lit(f"{arguments.first_date} 00:00:00")
            .str.to_datetime()
            .dt.replace_time_zone("UTC")
        )
        _LOG.info("first-date filter kept %d of %d rows", dataset.height, before)
    output_path = output_path_for(dataset_name=f"{source}{arguments.suffix}")
    dataset.write_parquet(output_path)
    _LOG.info("wrote %d rows to %s", dataset.height, output_path)
    _LOG.info(
        "span %s to %s; per-site rows %s",
        dataset["time"].min(),
        dataset["time"].max(),
        dataset.group_by("site").len().sort("site").to_dicts(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
