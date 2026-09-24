"""Pull ECMWF ENS wind at each wind meter's coordinates, at a run of forecast horizons.

One-off throwaway script for the wind-products study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/826>, sibling to
`fetch_ens_point.py` (issue #784), which pulls irradiance for the six solar meters. This script
reads the same ENS Delta table and reuses `fetch_ens_point.py`'s `NWP_ROOT`, `H3_RESOLUTION`,
`HORIZONS`, `_cell_for_each_meter`, `_wanted_leads`, and `_labelled_by_horizon` rather than
duplicating them: only the site roster (wind meters, not solar), the columns read, and the output
path differ. A shared module was not worth the extra indirection for two scripts.

**Wind speed and direction are instantaneous, unlike ENS's radiation.** The [NWP variable
conventions
page](https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/)
lists `wind_speed_10m`, `wind_speed_100m`, `wind_direction_10m`, and `wind_direction_100m` as
instantaneous at `valid_time`, where `fetch_ens_point.py`'s radiation column is period-ending over
the preceding forecast step. So, unlike radiation, wind carries no legitimate null at lead 0 — this
script still starts at lead 3 only because it reuses `HORIZONS`' bands unchanged, not because wind
needs it.

**Direction is circular**, per the same page: 359 degrees and 1 degree are two degrees apart in
reality and 358 degrees apart arithmetically, so `wind_direction_10m`/`wind_direction_100m` must
never be averaged, interpolated, or differenced as plain numbers. They are written through
unchanged here for a downstream study to turn into sine/cosine features, as
`wind_products.py` already does for its other wind sources.

Run it with `uv run python studies/beam_diffuse_split/fetch_ens_point_wind.py`.
"""

import logging
import sys
from pathlib import Path
from typing import Final

import polars as pl
from build_dataset import _wind_sites
from fetch_ens_point import (
    HORIZONS,
    NWP_ROOT,
    _cell_for_each_meter,
    _labelled_by_horizon,
    _wanted_leads,
)
from sources import WEATHER_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH: Final[Path] = WEATHER_DATA_DIR / "ENS" / "beam_diffuse_ens_wind.parquet"
"""Where this script writes the per-meter, per-member, per-horizon frame."""

WIND_COLUMNS: Final[tuple[str, ...]] = (
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
)
"""The ENS wind columns read, all instantaneous at `valid_time`. Speeds are in m/s, directions in
degrees (meteorological convention: the direction the wind blows FROM), matching the units
`nwp-variable-conventions.md` documents for the Delta table."""


def _read_meters(*, cells: list[int], leads: list[int]) -> pl.DataFrame:
    """Read every run's wind rows for the meters' cells and the horizons' leads.

    Same Delta-log-not-glob reasoning as `fetch_ens_point._read_meters`: a glob over the partition
    directories would double-count superseded files still on disk. See that function's docstring
    for the measured duplicate count.

    Args:
        cells: The H3 cells to keep.
        leads: The lead hours to keep.

    Returns:
        Every kept row, carrying the four wind columns and the lead.
    """
    return (
        pl.scan_delta(str(NWP_ROOT.parent))
        .filter(pl.col("h3_index").is_in(cells))
        .select("h3_index", "init_time", "valid_time", "ensemble_member", *WIND_COLUMNS)
        .with_columns(
            lead_hours=((pl.col("valid_time") - pl.col("init_time")).dt.total_minutes() // 60).cast(
                pl.Int32
            )
        )
        .filter(pl.col("lead_hours").is_in(leads))
        .collect()
    )


def main() -> int:
    """Write one frame holding every wind meter, member, and horizon.

    Returns:
        The process exit status.

    Raises:
        FileNotFoundError: If the ENS table is not on disk.
    """
    if not NWP_ROOT.exists():
        msg = f"{NWP_ROOT} is missing; run the ecmwf_ens asset first"
        raise FileNotFoundError(msg)

    lookup = _cell_for_each_meter(sites=_wind_sites())
    cells = lookup["h3_index"].unique().to_list()
    leads = _wanted_leads()
    logger.info("%d meters in %d cells, %d lead hours", lookup.height, len(cells), len(leads))

    frame = _read_meters(cells=cells, leads=leads)
    rows = _labelled_by_horizon(frame=frame).join(lookup, on="h3_index")
    rows = rows.select(
        "site",
        "init_time",
        "valid_time",
        "lead_hours",
        "horizon",
        "ensemble_member",
        speed_10m_ms=pl.col("wind_speed_10m"),
        speed_100m_ms=pl.col("wind_speed_100m"),
        direction_10m_deg=pl.col("wind_direction_10m"),
        direction_100m_deg=pl.col("wind_direction_100m"),
    ).sort("site", "horizon", "valid_time", "ensemble_member")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(OUTPUT_PATH)
    logger.info("wrote %s rows to %s", f"{rows.height:,}", OUTPUT_PATH)
    for name in HORIZONS:
        at = rows.filter(pl.col("horizon") == name)
        logger.info(
            "  %-7s %9s rows, %5d valid times, %d members",
            name,
            f"{at.height:,}",
            at["valid_time"].n_unique(),
            at["ensemble_member"].n_unique(),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
