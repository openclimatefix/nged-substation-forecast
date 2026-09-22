"""Pull ECMWF ENS irradiance at each meter's coordinates, at a run of forecast horizons.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**ENS is the only source here that is a forecast rather than an estimate of an hour that has
already happened**, which is what makes it worth scoring: every other source answers "what was the
irradiance?", and ENS answers the question the live service actually faces. The cost is that a
comparison against the other three mixes forecast skill with information content, so an ENS number
is read against another ENS number at a different horizon, never against CAMS.

Two conventions have to be honoured, and both are documented on the [NWP variable conventions
page](https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/):

- **Radiation is period-ending over the preceding forecast step**, 3 hours out to lead 144 and 6
  hours beyond it, where every other source in this experiment is period-ending over 1 hour. The
  scoring therefore happens on ENS's own stamps with the other sources aggregated up to them,
  rather than ENS interpolated down to an hour it never resolved.
- **Radiation is null at lead 0**, because a period-ending rate has no period to end. The shortest
  horizon here starts at lead 3 for that reason.

The whole table runs to 135 GB across 905 daily runs, which is far more than a reader of six
meters needs. Filtering each run to the four H3 cells the meters fall in, before anything is
collected, takes the download to about 6 million rows.

Run it with `uv run --with h3 --with pvlib --with polars python
scripts/experiments/beam_diffuse_split/fetch_ens_point.py`.
"""

import datetime as dt
import logging
import sys
import urllib.parse
from pathlib import Path
from typing import Final

import h3.api.basic_int as h3
import polars as pl
from sources import REPO_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NWP_ROOT: Final[Path] = REPO_DATA_DIR / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
"""Where the ENS Delta table's daily run partitions live."""

OUTPUT_PATH: Final[Path] = REPO_DATA_DIR / "ENS" / "beam_diffuse_ens.parquet"
"""Where this script writes the per-meter, per-member, per-horizon frame."""

H3_RESOLUTION: Final[int] = 5
"""The resolution the ENS table's `h3_index` column is keyed at.

Read off the table rather than assumed, in `_cell_for_each_meter`, so a table rewritten at another
resolution fails loudly instead of silently matching no cell.
"""

HORIZONS: Final[dict[str, tuple[int, int]]] = {
    "T+3": (3, 24),
    "day+1": (24, 48),
    "day+2": (48, 72),
    "day+7": (168, 192),
    "day+14": (336, 360),
}
"""Each reported horizon, as the half-open range of lead hours it covers.

A horizon is a *band* of leads rather than one lead, because one lead would give a single valid
time per run and too few rows to score. The bands match how a horizon is spoken about
operationally: `day+1` is every step a run resolves for the following day.

**`day+7` and `day+14` sit on the 6-hourly half of the step grid and the first three do not**, so
those two are a different row set and are reported separately. Shortwave radiation is the variable
worst affected by that coarsening, so part of any skill lost at the long horizons is the step width
rather than the forecast.
"""

RADIATION_COLUMN: Final[str] = "downward_short_wave_radiation_flux_surface"
"""The ENS global horizontal irradiance column, period-ending in W m⁻².

ENS publishes no direct-beam field, so a source built from this column can feed only the arms that
take global irradiance alone, and never the arm that reads a product's own split.
"""


def _cell_for_each_meter(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Map each meter to the H3 cell the ENS table keys its rows by.

    Args:
        sites: One row per meter, carrying `site`, `latitude`, and `longitude`.

    Returns:
        One row per meter, carrying `site` and `h3_index`.
    """
    return pl.DataFrame(
        {
            "site": sites["site"].to_list(),
            "h3_index": [
                h3.latlng_to_cell(latitude, longitude, H3_RESOLUTION)
                for latitude, longitude in zip(
                    sites["latitude"].to_list(), sites["longitude"].to_list(), strict=True
                )
            ],
        }
    )


def _init_time_of(*, partition: Path) -> dt.datetime:
    """Read a run's initialisation time out of its partition directory name.

    The column is a Hive partition key rather than a column inside the parquet files, so a plain
    `scan_parquet` of the files does not carry it.

    Args:
        partition: The `init_time=...` directory.

    Returns:
        The run's initialisation time, in UTC.
    """
    stamp = urllib.parse.unquote(partition.name.split("=", 1)[1])
    return dt.datetime.fromisoformat(stamp).replace(tzinfo=dt.UTC)


def _wanted_leads() -> list[int]:
    """Return every lead hour any horizon covers.

    Returns:
        The lead hours to keep, ascending.
    """
    return sorted({lead for first, last in HORIZONS.values() for lead in range(first, last)})


def _read_one_run(*, partition: Path, cells: list[int], leads: list[int]) -> pl.DataFrame:
    """Read one run's rows for the meters' cells and the horizons' leads.

    Args:
        partition: The `init_time=...` directory to read.
        cells: The H3 cells to keep.
        leads: The lead hours to keep.

    Returns:
        That run's rows, carrying the irradiance, the temperature, and the lead.
    """
    init_time = _init_time_of(partition=partition)
    return (
        pl.scan_parquet(f"{partition}/*.parquet")
        .filter(pl.col("h3_index").is_in(cells))
        .select("h3_index", "valid_time", "ensemble_member", RADIATION_COLUMN, "temperature_2m")
        .with_columns(init_time=pl.lit(init_time, dtype=pl.Datetime("us", "UTC")))
        .with_columns(
            lead_hours=((pl.col("valid_time") - pl.col("init_time")).dt.total_minutes() // 60).cast(
                pl.Int32
            )
        )
        .filter(pl.col("lead_hours").is_in(leads))
        .collect()
    )


def _labelled_by_horizon(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Tag every row with the horizon whose lead band contains it.

    Args:
        frame: Rows carrying `lead_hours`.

    Returns:
        The same rows with a `horizon` column.
    """
    label = pl.lit(None, dtype=pl.String)
    for name, (first, last) in HORIZONS.items():
        label = (
            pl.when((pl.col("lead_hours") >= first) & (pl.col("lead_hours") < last))
            .then(pl.lit(name))
            .otherwise(label)
        )
    return frame.with_columns(horizon=label)


def main() -> int:
    """Write one frame holding every meter, member, and horizon.

    Returns:
        The process exit status.

    Raises:
        FileNotFoundError: If the ENS table holds no run partitions.
    """
    from build_dataset import _pv_sites

    partitions = sorted(NWP_ROOT.glob("init_time=*"))
    if not partitions:
        msg = f"{NWP_ROOT} holds no init_time partitions; run the ecmwf_ens asset first"
        raise FileNotFoundError(msg)

    lookup = _cell_for_each_meter(sites=_pv_sites())
    cells = lookup["h3_index"].unique().to_list()
    leads = _wanted_leads()
    logger.info(
        "%d meters in %d cells, %d runs, %d lead hours",
        lookup.height,
        len(cells),
        len(partitions),
        len(leads),
    )

    frames = [
        _read_one_run(partition=partition, cells=cells, leads=leads) for partition in partitions
    ]
    rows = _labelled_by_horizon(frame=pl.concat(frames)).join(lookup, on="h3_index")
    rows = rows.select(
        "site",
        "init_time",
        "valid_time",
        "lead_hours",
        "horizon",
        "ensemble_member",
        ghi_w_m2=pl.col(RADIATION_COLUMN),
        temp_c=pl.col("temperature_2m"),
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
