"""Extract the ENS leads that the day-4 band needs and `ens_members.parquet` lacks.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/974>, read by
`nwp_forecast_comparison/build_forecast_inputs.py`.

**`fetch_ens_forecast_horizons.py` extracts one lead band for each of days 0, 1, 2, 3, 5, 7, 10,
and 14, so leads 105, 108, and 111 fall in no band.** The day-4 band scores leads 90 to 120 and
needs the 3-hourly steps from 90 to 126. The extract holds 90 to 102 (band 3's margin) and 114 to
126 (band 5's margin), so a day-4 build finds no step for the hours between them.

**The supplement holds exactly the day-4 leads that the extract lacks, read from the same local
Delta table, at the same H3 cells, for the runs the extract holds.** The script writes a new
folder and never touches `ens_forecast_horizons/`, and the Delta table is read only. Before writing,
the script re-reads the leads that the extract and the table share, and raises unless every row and
every field agrees. That check is what shows the supplement means the same as the extract: same
table, same units, same anonymous site labels.

Run it with `uv run python studies/beam_diffuse_split/fetch_ens_day4_supplement.py`.
"""

import logging
import sys
from typing import Final

import ens_forecast_horizons as efh
import polars as pl
from build_dataset import _pv_sites, _wind_sites
from fetch_ens_forecast_horizons import (
    ENSEMBLE_SIZE,
    MARGIN_HOURS,
    NWP_TABLE,
    OUTPUT_PATH,
    _cells,
    _members,
)
from sources import STUDIES_DATA_DIR
from studies.guards import refuse_to_overwrite

_LOG: Final[logging.Logger] = logging.getLogger("fetch_ens_day4_supplement")

DAY: Final[int] = 4
"""The band the extract lacks."""

SUPPLEMENT_DIR: Final = STUDIES_DATA_DIR / "ens_forecast_horizons_day4"
"""The new folder holding the supplement, written once."""

SUPPLEMENT_PATH: Final = SUPPLEMENT_DIR / "ens_members_day4.parquet"
"""The supplement: the extract's columns, for the leads the extract lacks at day 4."""

MISSING_LEADS: Final[tuple[int, ...]] = (105, 108, 111)
"""The 3-hourly leads inside the day-4 band that the extract lacks."""

KEY: Final[tuple[str, ...]] = ("site", "init_time", "valid_time", "ensemble_member")
"""What identifies one row of the extract."""


def band_leads() -> list[int]:
    """Return every hour of the day-4 band and its margin.

    Returns:
        The lead hours from 24 * 4 - `MARGIN_HOURS` to 24 * 5 + `MARGIN_HOURS`, ascending.
    """
    return list(range(24 * DAY - MARGIN_HOURS, 24 * DAY + 25 + MARGIN_HOURS))


def check_complete(*, supplement: pl.DataFrame, extract_pairs: pl.DataFrame) -> None:
    """Raise unless the supplement holds every missing lead, whole, for every (site, run).

    Args:
        supplement: The rows about to be written.
        extract_pairs: The (`site`, `init_time`) pairs the extract holds inside the day-4 band.

    Raises:
        ValueError: If the supplement's leads are not `MISSING_LEADS`, if a (site, run, lead) lacks
            a member, or if a (site, run) the extract holds lacks a lead.
    """
    leads = sorted(supplement["lead_hours"].unique().to_list())
    if leads != list(MISSING_LEADS):
        msg = f"the supplement holds leads {leads}, not {list(MISSING_LEADS)}"
        raise ValueError(msg)
    counts = supplement.group_by("site", "init_time", "lead_hours").len()
    short = counts.filter(pl.col("len") != ENSEMBLE_SIZE)
    if not short.is_empty():
        msg = f"{short.height} (site, run, lead) groups do not hold {ENSEMBLE_SIZE} members"
        raise ValueError(msg)
    per_pair = (
        counts.group_by("site", "init_time").len().filter(pl.col("len") == len(MISSING_LEADS))
    )
    lacking = extract_pairs.join(per_pair, on=["site", "init_time"], how="anti")
    if not lacking.is_empty():
        msg = (
            f"{lacking.height} (site, run) pairs in the extract lack one of leads "
            f"{list(MISSING_LEADS)} in the table"
        )
        raise ValueError(msg)


def log_surviving_runs(*, extract_band: pl.DataFrame, supplement: pl.DataFrame) -> None:
    """Log how many (site, run) pairs `band_steps` keeps at day 4, against those the extract holds.

    Args:
        extract_band: The extract's rows at the day-4 band's leads, for every site.
        supplement: The rows about to be written.
    """
    for domain, roster in (("solar", _pv_sites()), ("wind", _wind_sites())):
        members = pl.concat([extract_band, supplement.select(extract_band.columns)]).filter(
            pl.col("site").is_in(roster["site"])
        )
        steps = efh.band_steps(members=members, day=DAY, domain=domain, ensemble_size=ENSEMBLE_SIZE)
        kept = steps.keys.select("site", "init_time").unique().height
        held = members.select("site", "init_time").unique().height
        _LOG.info(
            "%s day 4: band_steps keeps %d of the %d (site, run) pairs the extract holds",
            domain,
            kept,
            held,
        )


def main() -> int:
    """Write the supplement after checking it against the extract.

    Returns:
        The process exit status.

    Raises:
        FileNotFoundError: If the extract or the NWP table is not on disk.
        ValueError: If the table lacks a lead of the band, has no lead in the band that the extract
            lacks, disagrees with the extract on a lead both hold, or would give a supplement that
            lacks any of `MISSING_LEADS` for any (site, run) the extract holds.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    refuse_to_overwrite(paths=[SUPPLEMENT_PATH])
    for path in (OUTPUT_PATH, NWP_TABLE):
        if not path.exists():
            msg = f"{path} is missing"
            raise FileNotFoundError(msg)

    sites = pl.concat(
        [
            _pv_sites().select("site", "latitude", "longitude"),
            _wind_sites().select("site", "latitude", "longitude"),
        ]
    )
    lookup = _cells(sites=sites)
    extract = pl.scan_parquet(OUTPUT_PATH)
    held = set(extract.select("lead_hours").unique().collect()["lead_hours"].to_list())
    runs = extract.select("init_time").unique().collect()["init_time"]
    table = (
        lookup.join(
            _members(cells=lookup["h3_index"].unique().to_list(), leads=band_leads()), on="h3_index"
        )
        .drop("h3_index")
        .filter(pl.col("init_time").is_in(runs.implode()))
        .sort(*KEY)
    )
    table_leads = set(table["lead_hours"].unique().to_list())
    if not {90, 102, 114, 126} <= table_leads:
        msg = f"the table lacks a lead of the day-4 band: it holds {sorted(table_leads)}"
        raise ValueError(msg)
    shared = sorted(table_leads & held)
    supplement = table.filter(~pl.col("lead_hours").is_in(shared))
    if supplement.is_empty():
        msg = "the extract already holds every day-4 lead"
        raise ValueError(msg)

    ours = table.filter(pl.col("lead_hours").is_in(shared))
    theirs = (
        extract.filter(pl.col("lead_hours").is_in(list(shared)))
        .filter(pl.col("site").is_in(sites["site"]))
        .collect()
        .sort(*KEY)
    )
    if ours.height != theirs.height or not ours.select(theirs.columns).equals(theirs):
        msg = (
            f"the table and the extract disagree on the leads they share, {shared}: "
            f"{ours.height} rows against {theirs.height}"
        )
        raise ValueError(msg)
    _LOG.info("the table agrees with the extract on %d rows at leads %s", ours.height, shared)

    extract_band = extract.filter(pl.col("lead_hours").is_in(band_leads())).collect()
    check_complete(
        supplement=supplement,
        extract_pairs=extract_band.select("site", "init_time").unique(),
    )
    log_surviving_runs(extract_band=extract_band, supplement=supplement)

    SUPPLEMENT_DIR.mkdir(parents=True, exist_ok=True)
    supplement.select(theirs.columns).write_parquet(SUPPLEMENT_PATH)
    _LOG.info(
        "wrote %d rows at leads %s for %d generators and %d runs to %s",
        supplement.height,
        sorted(supplement["lead_hours"].unique().to_list()),
        supplement["site"].n_unique(),
        supplement["init_time"].n_unique(),
        SUPPLEMENT_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
