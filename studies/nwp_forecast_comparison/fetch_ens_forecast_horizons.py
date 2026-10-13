"""Extract every ECMWF ENS member at every metered generator, at the leads the horizon study scores.

One-off throwaway script for the first step of
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>, read by
`ens_forecast_horizons.py`.

**Every member is kept, because the study uses the members three ways**: the control member alone,
the mean of the members' fields, and each member through the power model in turn. The fields kept
are global horizontal irradiance, air temperature at 2 m, and wind speed and direction at 100 m and
at 10 m.

**Rows are keyed by the generator's anonymous label, and each generator reads the H3 cell it sits
in.** The table stores each field as the area-weighted mean over an H3 resolution-5 cell, which is
what the live service reads. Coordinates are read from the private site list at run time and never
written.

**Each lead band is a whole UTC day after the run's own day**, so every band covers the same hours
of the day. Band `d` scores leads 24d to 24d + 24, and the extract keeps 6 hours either side of
that, so every scored hour has a stamp on both sides of it to interpolate between and a 6-hourly
radiation window either side to anchor a clear-sky index on. `ens_forecast_horizons.py` owns which
leads each technology scores.

The extract is read through the Delta transaction log, never by globbing the parquet files, because
a partition written twice keeps its superseded files on disk.

Run with `uv run python studies/nwp_forecast_comparison/fetch_ens_forecast_horizons.py`
followed by `--source COMPONENT_TABLE --source-version VERSION --output-directory NEW_DIRECTORY`.
The main extract and supplement must use the same pinned source snapshot and output directory.
Existing polar caches remain frozen and cannot be paired with newly derived values.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

import h3.api.basic_int as h3
import polars as pl
from ml_core.features import wind_feature_expressions
from studies.ens_members import ENSEMBLE_SIZE
from studies.guards import refuse_to_overwrite
from studies.pv_dataset import pv_sites, wind_sites
from studies.sources import ENS_PRODUCT_DIR

OUTPUT_PATH: Final[Path] = ENS_PRODUCT_DIR / "ens_members.parquet"
"""Frozen extract used by existing analyses; new extractions require an output directory."""

_LOG: Final[logging.Logger] = logging.getLogger("fetch_ens_forecast_horizons")

NWP_MODEL_ID: Final[str] = "ECMWF_ENS_0_25_degree"
"""The ENS partition of the source table."""

H3_RESOLUTION: Final[int] = 5
"""The resolution the ENS table's `h3_index` is keyed at."""

BAND_DAYS: Final[tuple[int, ...]] = (0, 1, 2, 3, 5, 7, 10, 14)
"""The days after the run's own day that the study scores, one lead band each."""

MARGIN_HOURS: Final[int] = 6
"""How many hours of leads either side of each band the extract keeps."""


RADIATION: Final[str] = "downward_short_wave_radiation_flux_surface"
"""Global horizontal irradiance, a mean over the step ending at `valid_time`, in W m⁻²."""


def _leads() -> list[int]:
    """Return every lead hour any band needs, with `MARGIN_HOURS` either side of each band.

    Returns:
        The lead hours, ascending.
    """
    return sorted(
        {
            lead
            for day in BAND_DAYS
            for lead in range(max(24 * day - MARGIN_HOURS, 0), 24 * day + 25 + MARGIN_HOURS)
        }
    )


def _cells(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Map each generator to the H3 cell it sits in.

    Args:
        sites: One row per generator, with `site`, `latitude`, and `longitude`.

    Returns:
        One row per generator, with `site` and `h3_index`.
    """
    return sites.select(
        "site",
        h3_index=pl.Series(
            [
                h3.latlng_to_cell(latitude, longitude, H3_RESOLUTION)
                for latitude, longitude in sites.select("latitude", "longitude").iter_rows()
            ],
            dtype=pl.Int64,
        ),
    )


def _members(
    *, source: Path, source_version: int, cells: list[int], leads: list[int] | None = None
) -> pl.DataFrame:
    """Read every member of every run at the wanted cells and leads.

    Args:
        source: Component-wind Delta archive.
        source_version: Pinned Delta snapshot.
        cells: The H3 cells to read.
        leads: The lead hours to read; `_leads()` where `None`.

    Returns:
        One row per cell, run, valid time and member, with the lead and the six fields the study
        reads.
    """
    return (
        pl.scan_delta(str(source), version=source_version)
        .filter(pl.col("nwp_model_id") == NWP_MODEL_ID, pl.col("h3_index").is_in(cells))
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("init_time")).dt.total_hours().cast(pl.Int32)
        )
        .filter(pl.col("lead_hours").is_in(_leads() if leads is None else leads))
        .with_columns(
            *wind_feature_expressions(height="10m"),
            *wind_feature_expressions(height="100m"),
        )
        .select(
            "h3_index",
            "init_time",
            "valid_time",
            "lead_hours",
            "ensemble_member",
            ghi_w_m2=pl.col(RADIATION),
            temp_c=pl.col("temperature_2m"),
            speed_100m=pl.col("wind_speed_100m"),
            direction_100m=pl.col("wind_direction_100m"),
            speed_10m=pl.col("wind_speed_10m"),
            direction_10m=pl.col("wind_direction_10m"),
        )
        .collect()
    )


def source_metadata(*, source: Path, source_version: int) -> dict:
    """Identify the native component snapshot shared by an extract and its supplement."""
    return {
        "source": str(source.resolve()),
        "version": source_version,
        "schema": {
            name: str(dtype)
            for name, dtype in pl.scan_delta(str(source), version=source_version)
            .collect_schema()
            .items()
        },
        "wind_representation": "components-derived-polar-native-v1",
    }


def main(*, source: Path, source_version: int, output_directory: Path) -> int:
    """Write the per-member extract.

    Args:
        source: Component-wind Delta archive.
        source_version: Pinned Delta snapshot.
        output_directory: New dataset directory, separate from historical polar extracts.

    Returns:
        The process exit status.

    Raises:
        FileNotFoundError: If the ENS table is not on disk.
        ValueError: If a generator's H3 cell has no rows in the table.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_path = output_directory / "ens_members.parquet"
    metadata_path = output_directory / "ens_members.source.json"
    refuse_to_overwrite(paths=[output_path, metadata_path])
    if not source.exists():
        msg = f"{source} is missing; build the component archive first"
        raise FileNotFoundError(msg)

    sites = pl.concat(
        [
            pv_sites().select("site", "latitude", "longitude"),
            wind_sites().select("site", "latitude", "longitude"),
        ]
    )
    lookup = _cells(sites=sites)
    members = _members(
        source=source, source_version=source_version, cells=lookup["h3_index"].unique().to_list()
    )
    rows = (
        lookup.join(members, on="h3_index")
        .drop("h3_index")
        .sort("site", "init_time", "valid_time", "ensemble_member")
    )
    missing = set(sites["site"].to_list()) - set(rows["site"].unique().to_list())
    if missing:
        msg = f"no ENS rows for generators {sorted(missing)}: their H3 cells are not in the table"
        raise ValueError(msg)
    output_directory.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.partial")
    rows.write_parquet(temporary)
    metadata_path.write_text(
        json.dumps(source_metadata(source=source, source_version=source_version), indent=2) + "\n"
    )
    temporary.replace(output_path)
    _LOG.info(
        "wrote %d rows for %d generators, %d runs (%s to %s), %d members, to %s",
        rows.height,
        rows["site"].n_unique(),
        rows["init_time"].n_unique(),
        rows["init_time"].min(),
        rows["init_time"].max(),
        rows["ensemble_member"].n_unique(),
        output_path,
    )
    per_stamp = rows.group_by("site", "init_time", "valid_time").len()
    _LOG.info(
        "stamps with fewer than %d members: %d of %d",
        ENSEMBLE_SIZE,
        per_stamp.filter(pl.col("len") < ENSEMBLE_SIZE).height,
        per_stamp.height,
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-version", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(
        main(
            source=args.source,
            source_version=args.source_version,
            output_directory=args.output_directory,
        )
    )
