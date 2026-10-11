"""Build the joined hourly frame the ERA5 variable ladder is fitted on.

One-off throwaway script for the study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1097>. It joins, for each of the
six solar farms and each daylight hour, the farm's hourly output, CAMS satellite irradiance, and
every ERA5 variable of the ladder (`studies.era5_ladder`) read from the ERA5 cell nearest the farm.
Optionally it adds CAMS EAC4 aerosol optical depth. It writes one parquet and one markdown file of
checks, both under `data/studies/per_study/era5_solar_variables/inputs/`.

**The rows are set by the clock, the place, the targets, and the span, and never by an ERA5 value.**
An hour is kept when all of these hold:

- the sun is up: the top-of-atmosphere horizontal flux exceeds
  `era5_ladder.MIN_EXTRATERRESTRIAL_W_M2` at the hour's midpoint (`add_solar_geometry`) and in the
  hour-integrated value the CAMS files carry;
- both targets exist: the farm's output has both half-hours, and CAMS has the hour;
- no half-hour of the output reads exactly zero, because a zero is a meter dropout or a snow-covered
  panel and the two cannot be told apart (`--keep-zero-hours-with-snow` keeps the ones where
  ERA5 reports snow on the ground, for the exploratory snow arm);
- the hour is after the commissioning ramp's end and within the span in which every ERA5 file
  is final ERA5 (`expver` 0001) in every hour.

**The build raises if any ERA5 variable other than `cbh` and `cin` is missing on a kept row.**
ERA5 leaves cloud base height missing where there is no cloud, and probably leaves convective
inhibition missing too, so those two keep their missing values.

**Hour conventions.** The accumulations (`ssrd`, `ssrdc`, `fdir`, `cdir`, `strd`, `tp`, `sf`, `uvb`)
are totals over the hour ending at the label and become hourly rates. Every other variable is a
snapshot, and is averaged over the labels one hour earlier and at the label, through
`hourly_from_snapshots`, with `cbh` and `cin` each in a call of their own so that a missing snapshot
makes only that variable's hour missing.

**The snow variant also restores the hours the outage filter removed where ERA5 reports snow.**
`drop_outages_and_spikes` drops every run of 24 or more zero hours, and the zero night on either
side of a day of snow-covered panels joins the day to a run that long. The variant restores those
hours where `sd` is above zero, so that the exploratory snow arm sees whole snow days.

**The aerosol columns are added to the full build whenever `eac4_aod.parquet` exists**, because the
aerosol view needs them and the build refuses to overwrite its output. `--no-aerosol` leaves them
out.

Run it with `uv run --with netcdf4 python
studies/era5_solar_variables/era5_ladder_build_dataset.py`, adding `--through-rung g2` to build
from only the variables downloaded so far. No coordinate appears in the output.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Final

import polars as pl
from era5_ladder_arms import INPUTS_DIR, checks_path, dataset_path
from studies.arm_runner import add_time_features
from studies.commissioning import drop_commissioning_ramp
from studies.era5_ladder import (
    ACCUMULATED_VARIABLES,
    AEROSOL_COLUMNS,
    INSTANTANEOUS_VARIABLES,
    MIN_EXTRATERRESTRIAL_W_M2,
    MISSING_UNDER_CLEAR_SKY_VARIABLES,
    RUNGS,
    RungType,
    accumulation_to_hourly_rate,
    aerosol_hour_ending_mean,
    ratio_index,
    rung_variables,
)
from studies.export_cap import with_export_cap
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.hourly_means import KEY_COLUMN, hourly_from_snapshots
from studies.pv_dataset import (
    CAMS_PATH,
    IMPLAUSIBLE_CAPACITY_MULTIPLE,
    add_solar_geometry,
    drop_outages_and_spikes,
    nearest_era5_cell,
    pv_sites,
    read_era5,
    solar_hourly_power,
)
from studies.sources import (
    CAMS_PRODUCT_DIR,
    ERA5_PRODUCT_DIR,
    REANALYSIS_DOWNLOADS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("era5_ladder_build_dataset")

ERA5_SOLAR_VARIABLES_DIR: Final[Path] = ERA5_PRODUCT_DIR / "solar_variables"
"""Where the downloaded ERA5 variables are: one parquet per variable, named by its short name."""

EAC4_PATH: Final[Path] = REANALYSIS_DOWNLOADS_DIR / "CAMS_EAC4_AOD" / "eac4_aod.parquet"
"""The CAMS EAC4 aerosol optical depths over the box, in a wide frame, 3-hourly."""

HELD_COPY_COLUMNS: Final[dict[str, str]] = {"ghi_w_m2": "ssrd", "bhi_w_m2": "fdir", "temp_c": "t2m"}
"""The held `beam_diffuse/` copy's columns and the ERA5 names they hold.

The held copy's fluxes are already hourly rates in W m⁻², and its `t2m` is in degrees Celsius.
"""

FINAL_RELEASE: Final[str] = "0001"
"""The `expver` of final ERA5. Preliminary ERA5T is `0005` and ECMWF may replace it."""

SNAPSHOT_OFFSETS_MINUTES: Final[tuple[int, int]] = (-60, 0)
"""The labels, relative to an hour's label, whose snapshots are averaged into the hour."""

MIN_CLEAR_SKY_W_M2: Final[float] = 1.0
"""The clear-sky index is missing where the clear-sky irradiance is at or below this."""

NAN_SPLIT_CLOUD_COVER: Final[float] = 0.05
"""The total cloud cover that the missing-value shares of `cbh` and `cin` are split at."""

NEAREST_CELL_COLUMNS: Final[tuple[str, str]] = ("cell_latitude", "cell_longitude")
"""The columns that name the ERA5 cell of a site. They are dropped before anything is written."""

CAMS_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "observation_period",
    "toa_wh_m2",
    "clear_sky_ghi_wh_m2",
    "clear_sky_bhi_wh_m2",
    "clear_sky_dhi_wh_m2",
    "clear_sky_bni_wh_m2",
    "ghi_wh_m2",
    "bhi_wh_m2",
    "dhi_wh_m2",
    "bni_wh_m2",
    "reliability",
)
"""The 11 columns the CAMS radiation service writes, in order, under its commented header."""


def downloaded_variables(*, through_rung: RungType) -> tuple[str, ...]:
    """Return the variables of a rung that come from the new download, not the held copy.

    Args:
        through_rung: The highest rung wanted.

    Returns:
        The ERA5 short names whose parquet files are in `ERA5_SOLAR_VARIABLES_DIR`.
    """
    held = set(HELD_COPY_COLUMNS.values())
    return tuple(name for name in rung_variables(rung=through_rung) if name not in held)


def read_downloaded(*, variable: str) -> pl.DataFrame:
    """Read one downloaded variable's table.

    Args:
        variable: The ERA5 short name.

    Returns:
        One row per (time, latitude, longitude) with `value` in Float64 and `expver`.

    Raises:
        FileNotFoundError: If the variable has not been downloaded.
    """
    path = ERA5_SOLAR_VARIABLES_DIR / f"{variable}.parquet"
    if not path.exists():
        msg = f"{path} missing; the download has not reached {variable}"
        raise FileNotFoundError(msg)
    return pl.read_parquet(path).select(
        "time", "latitude", "longitude", pl.col("value").cast(pl.Float64), "expver"
    )


def span_end(*, tables: dict[str, pl.DataFrame]) -> datetime | None:
    """Return the first instant that is not within a month of final ERA5 in every file.

    Args:
        tables: Each downloaded variable's table, with `time` and `expver`.

    Returns:
        The start of the first month holding a preliminary hour in any table, so that rows before
        it are final ERA5 in every file, or `None` if every hour of every file is final.
    """
    firsts = [
        table.filter(pl.col("expver") != FINAL_RELEASE)["time"].min() for table in tables.values()
    ]
    present = [first for first in firsts if first is not None]
    if not present:
        return None
    return pl.DataFrame({"first": present}).select(pl.col("first").dt.truncate("1mo").min()).item()


def raise_if_releases_differ(*, tables: dict[str, pl.DataFrame]) -> None:
    """Raise if two variables of one kind carry different `expver` for the same hour and cell.

    Accumulations and snapshots are checked separately, because an accumulation stamped 00 to 06 UTC
    on the first day of a month comes from the previous day's forecast.

    Args:
        tables: Each downloaded variable's table, with `time`, `latitude`, `longitude`, `expver`.

    Raises:
        ValueError: If any hour and cell has more than one `expver` within a kind.
    """
    for kind, names in (
        ("accumulation", ACCUMULATED_VARIABLES),
        ("snapshot", INSTANTANEOUS_VARIABLES),
    ):
        present = [tables[name] for name in names if name in tables]
        if len(present) < 2:
            continue
        disagreeing = (
            pl.concat(table.select("time", "latitude", "longitude", "expver") for table in present)
            .group_by("time", "latitude", "longitude")
            .agg(releases=pl.col("expver").n_unique())
            .filter(pl.col("releases") > 1)
        )
        if not disagreeing.is_empty():
            msg = f"{disagreeing.height} hour-cells of {kind} variables mix ERA5 releases"
            raise ValueError(msg)


def snapshot_hourly(*, tables: dict[str, pl.DataFrame], names: Sequence[str]) -> pl.DataFrame:
    """Average snapshot variables over the hour, one `hourly_from_snapshots` call per group.

    The variables that may be missing under clear sky each go through their own call, so that a
    missing snapshot of `cbh` does not remove the hour from the other variables. The rest share one
    call, which keeps an hour only if every one of them has both snapshots.

    Args:
        tables: Each downloaded variable's table.
        names: The snapshot variables to average.

    Returns:
        One row per (latitude, longitude, time) holding the hourly means that exist. A variable
        with no value in an hour is null in the hour's row.
    """

    def keyed(*, columns: Sequence[str]) -> pl.DataFrame:
        joined = tables[columns[0]].select(
            "time", "latitude", "longitude", pl.col("value").alias(columns[0])
        )
        for name in columns[1:]:
            joined = joined.join(
                tables[name].select("time", "latitude", "longitude", pl.col("value").alias(name)),
                on=["time", "latitude", "longitude"],
                how="inner",
            )
        return joined.with_columns(
            pl.format("{}_{}", pl.col("latitude"), pl.col("longitude")).alias(KEY_COLUMN)
        )

    complete = [name for name in names if name not in MISSING_UNDER_CLEAR_SKY_VARIABLES]
    groups: list[list[str]] = [complete] if complete else []
    groups += [[name] for name in names if name in MISSING_UNDER_CLEAR_SKY_VARIABLES]
    hourly: pl.DataFrame | None = None
    for group in groups:
        frame = keyed(columns=group)
        averaged = hourly_from_snapshots(
            frame=frame.select(KEY_COLUMN, "time", *group),
            value_columns=group,
            slot_offsets_minutes=SNAPSHOT_OFFSETS_MINUTES,
        )
        coordinates = frame.select(KEY_COLUMN, "latitude", "longitude").unique()
        averaged = averaged.join(coordinates, on=KEY_COLUMN, how="left").drop(KEY_COLUMN)
        hourly = (
            averaged
            if hourly is None
            else hourly.join(
                averaged, on=["time", "latitude", "longitude"], how="full", coalesce=True
            )
        )
    if hourly is None:
        msg = "no snapshot variables to average"
        raise ValueError(msg)
    return hourly


def era5_columns(*, through_rung: RungType) -> tuple[pl.DataFrame, datetime | None]:
    """Read the held copy and every downloaded variable up to a rung, as hourly columns.

    Args:
        through_rung: The highest rung wanted.

    Returns:
        The wide table with one row per (time, latitude, longitude) and one column per variable,
        and the start of the first month that is not final ERA5 in every downloaded file. A build
        with no downloaded variable reads the total cloud cover's releases, because the held copy
        carries no `expver`.
    """
    held = (
        read_era5(source="cds")
        .select("time", "latitude", "longitude", *HELD_COPY_COLUMNS)
        .rename(HELD_COPY_COLUMNS)
    )
    wanted = rung_variables(rung=through_rung)
    names = downloaded_variables(through_rung=through_rung)
    tables = {name: read_downloaded(variable=name) for name in names}
    raise_if_releases_differ(tables=tables)
    end = span_end(tables=tables or {"tcc": read_downloaded(variable="tcc")})

    snapshot_names = [name for name in names if name in INSTANTANEOUS_VARIABLES]
    accumulation_names = [name for name in names if name in ACCUMULATED_VARIABLES]

    # The held `t2m` is a snapshot, so it joins the snapshot average with the downloaded ones.
    held_temperature = held.select("time", "latitude", "longitude", pl.col("t2m").alias("value"))
    hourly = snapshot_hourly(
        tables={**tables, "t2m": held_temperature}, names=["t2m", *snapshot_names]
    )
    wide = held.drop("t2m").join(hourly, on=["time", "latitude", "longitude"], how="left")
    for name in accumulation_names:
        rate = (
            tables[name]
            .rename({"value": name})
            .with_columns(accumulation_to_hourly_rate(variable=name))
            .select("time", "latitude", "longitude", name)
        )
        wide = wide.join(rate, on=["time", "latitude", "longitude"], how="left")
    missing = [name for name in wanted if name not in wide.columns]
    if missing:
        msg = f"the wide table lacks {missing}"
        raise ValueError(msg)
    return wide, end


def read_cams_top_of_atmosphere(*, sites: Sequence[str]) -> pl.DataFrame:
    """Read the hour-integrated top-of-atmosphere irradiation from the CAMS site files.

    Args:
        sites: The anonymised site labels, which name the downloaded files.

    Returns:
        One row per (site, time) with `cams_toa_w_m2`. The service sums Wh m⁻² over a 1-hour step,
        so the number is also the mean flux in W m⁻².
    """
    frames: list[pl.DataFrame] = []
    for site in sites:
        for path in sorted(CAMS_PRODUCT_DIR.glob(f"cams_site_{site}_*.csv")):
            frame = pl.read_csv(
                path,
                separator=";",
                comment_prefix="#",
                has_header=False,
                new_columns=list(CAMS_CSV_COLUMNS),
                null_values=["nan"],
            )
            frames.append(
                frame.select(
                    site=pl.lit(site),
                    time=pl.col("observation_period")
                    .str.split("/")
                    .list.last()
                    .str.to_datetime("%Y-%m-%dT%H:%M:%S%.f")
                    .dt.replace_time_zone("UTC"),
                    cams_toa_w_m2=pl.col("toa_wh_m2"),
                ).drop_nulls()
            )
    return pl.concat(frames).unique(subset=["site", "time"], keep="first").sort("site", "time")


def read_cams_irradiance() -> pl.DataFrame:
    """Read CAMS global and clear-sky irradiance with its reliability flag, for every hour.

    Returns:
        One row per (site, time) with `cams_ghi_w_m2`, `cams_clear_sky_ghi_w_m2`, and
        `cams_reliability`. No hour is dropped for low reliability, because dropping on a flag the
        retrieval sets would pick rows by one target's value.
    """
    return pl.read_parquet(CAMS_PATH).select(
        "site",
        "time",
        cams_ghi_w_m2=pl.col("ghi_w_m2"),
        cams_clear_sky_ghi_w_m2=pl.col("clear_sky_ghi_w_m2"),
        cams_reliability=pl.col("reliability"),
    )


def aerosol_columns(*, labels: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the hour-ending EAC4 aerosol optical depths at each site's nearest EAC4 cell.

    Args:
        labels: The (site, time) rows to return values for.
        sites: The site list, with each site's coordinates.

    Returns:
        `labels` with `aod550` and `duaod550`, null outside the EAC4 span.

    Raises:
        FileNotFoundError: If the EAC4 file has not been written.
        ValueError: If the file lacks the expected columns.
    """
    if not EAC4_PATH.exists():
        msg = f"{EAC4_PATH} missing; fetch_cams_eac4_aod.py has not finished"
        raise FileNotFoundError(msg)
    eac4 = pl.read_parquet(EAC4_PATH)
    expected = {"time", "latitude", "longitude", *AEROSOL_COLUMNS}
    if not expected <= set(eac4.columns):
        msg = f"{EAC4_PATH} has columns {eac4.columns}, expected {sorted(expected)}"
        raise ValueError(msg)
    eac4 = eac4.with_columns(
        longitude=pl.when(pl.col("longitude") > 180.0)
        .then(pl.col("longitude") - 360.0)
        .otherwise(pl.col("longitude"))
    )
    cells = eac4.select("latitude", "longitude").unique()
    per_site: list[pl.DataFrame] = []
    for row in sites.select("site", "latitude", "longitude").iter_rows(named=True):
        nearest = cells.with_columns(
            distance=(pl.col("latitude") - row["latitude"]) ** 2
            + (pl.col("longitude") - row["longitude"]) ** 2
        ).sort("distance")[0]
        series = eac4.filter(
            (pl.col("latitude") == nearest["latitude"].item())
            & (pl.col("longitude") == nearest["longitude"].item())
        ).select(pl.lit(row["site"]).alias("site"), "time", *AEROSOL_COLUMNS)
        per_site.append(series)
    return aerosol_hour_ending_mean(
        aerosol=pl.concat(per_site),
        labels=labels.select("site", "time"),
        value_columns=AEROSOL_COLUMNS,
    ).sort("site", "time")


def missing_value_shares(*, rows: pl.DataFrame, names: Sequence[str]) -> str:
    """Write a markdown table of each variable's share of missing values on the kept rows.

    A value is missing if it is null or not-a-number. The shares of `cbh` and `cin` are split by
    total cloud cover below and from `NAN_SPLIT_CLOUD_COVER`, when the frame holds total cloud
    cover.

    Args:
        rows: The kept rows.
        names: The variables to report.

    Returns:
        The table, as markdown.
    """
    lines = [
        "| variable | share missing | share missing, `tcc` below 0.05 | `tcc` from 0.05 |",
        "|---|---|---|---|",
    ]
    has_cover = "tcc" in rows.columns
    for name in names:
        missing = pl.col(name).is_null() | pl.col(name).is_nan().fill_null(value=False)
        overall = rows.select(missing.mean()).item()
        split = "n/a | n/a"
        if has_cover:
            low = rows.filter(pl.col("tcc") < NAN_SPLIT_CLOUD_COVER).select(missing.mean()).item()
            high = rows.filter(pl.col("tcc") >= NAN_SPLIT_CLOUD_COVER).select(missing.mean()).item()
            split = f"{low:.3f} | {high:.3f}"
        lines.append(f"| `{name}` | {overall:.4f} | {split} |")
    return "\n".join(lines)


def restore_outage_hours(
    *, raw_power: pl.DataFrame, kept_power: pl.DataFrame, sites: pl.DataFrame
) -> pl.DataFrame:
    """Put back the hours the outage filter removed, apart from meter spikes.

    The snow variant keeps a removed hour only if ERA5 then reports snow, and applies that test
    after the ERA5 join, in the zero rule. An hour above the spike limit stays removed.

    Args:
        raw_power: Hourly power before `drop_outages_and_spikes`.
        kept_power: Hourly power after it.
        sites: The site list, for each site's capacity.

    Returns:
        `kept_power` and the removed hours that are not spikes, sorted by site and time.
    """
    removed = (
        raw_power.join(kept_power.select("site", "time"), on=["site", "time"], how="anti")
        .join(sites.select("site", "effective_capacity_mw"), on="site")
        .filter(
            pl.col("power_mw") <= IMPLAUSIBLE_CAPACITY_MULTIPLE * pl.col("effective_capacity_mw")
        )
        .select(kept_power.columns)
    )
    return pl.concat([kept_power, removed]).sort("site", "time")


def build(
    *, through_rung: RungType, keep_zero_hours_with_snow: bool, with_aerosol: bool
) -> tuple[pl.DataFrame, str]:
    """Build the frame and the text of its checks.

    Args:
        through_rung: The highest rung whose variables are joined.
        keep_zero_hours_with_snow: Whether to keep hours with a zero half-hour where `sd` is above
            zero.
        with_aerosol: Whether to add the EAC4 aerosol columns.

    Returns:
        The kept rows, and the checks as markdown.

    Raises:
        ValueError: If the snow variant is asked of a rung below `g8`, which has no `sd`.
    """
    if keep_zero_hours_with_snow and RUNGS.index(through_rung) < RUNGS.index("g8"):
        msg = "the snow variant needs snow depth, which the g8 rung adds"
        raise ValueError(msg)
    sites = pv_sites()
    wide, end = era5_columns(through_rung=through_rung)
    sites_with_cells = nearest_era5_cell(sites=sites, era5=wide)
    _LOG.info(
        "the %d sites resolve to %d distinct ERA5 cells",
        sites_with_cells.height,
        sites_with_cells.select(*NEAREST_CELL_COLUMNS).n_unique(),
    )

    raw_power = solar_hourly_power(sites=sites)
    power = drop_outages_and_spikes(power=raw_power, sites=sites)
    if keep_zero_hours_with_snow:
        power = restore_outage_hours(raw_power=raw_power, kept_power=power, sites=sites)
    joined = (
        power.join(sites_with_cells.drop("time_series_id"), on=["site", "effective_capacity_mw"])
        .join(
            wide,
            left_on=["time", *NEAREST_CELL_COLUMNS],
            right_on=["time", "latitude", "longitude"],
            how="left",
        )
        .join(read_cams_irradiance(), on=["site", "time"], how="inner")
        .join(read_cams_top_of_atmosphere(sites=sites["site"].to_list()), on=["site", "time"])
    )
    with_geometry = add_solar_geometry(joined=joined)
    in_span = pl.col("time") < end if end is not None else pl.lit(value=True)
    daylight = with_geometry.filter(
        in_span,
        pl.col("extraterrestrial_horizontal_w_m2") > MIN_EXTRATERRESTRIAL_W_M2,
        pl.col("cams_toa_w_m2") > MIN_EXTRATERRESTRIAL_W_M2,
    )
    _LOG.info("daylight rows with both targets, inside the final span: %d", daylight.height)

    zero_rule = ~pl.col("has_zero_half_hour")
    if keep_zero_hours_with_snow:
        zero_rule = zero_rule | (pl.col("sd") > 0.0)
    kept = drop_commissioning_ramp(dataset=daylight.filter(zero_rule))
    if kept.select("site", "time").is_duplicated().any():
        msg = "the kept rows hold a duplicate (site, time)"
        raise ValueError(msg)
    if with_aerosol:
        kept = kept.join(
            aerosol_columns(labels=kept.select("site", "time"), sites=sites),
            on=["site", "time"],
            how="left",
        )

    wanted = rung_variables(rung=through_rung)
    kept = kept.with_columns(
        clearness_index=ratio_index(
            numerator=pl.col("ssrd"),
            denominator=pl.col("extraterrestrial_horizontal_w_m2"),
            minimum_denominator=MIN_EXTRATERRESTRIAL_W_M2,
        ),
        cams_clearness_index=ratio_index(
            numerator=pl.col("cams_ghi_w_m2"),
            denominator=pl.col("cams_toa_w_m2"),
            minimum_denominator=MIN_EXTRATERRESTRIAL_W_M2,
        ),
        cams_clear_sky_index=ratio_index(
            numerator=pl.col("cams_ghi_w_m2"),
            denominator=pl.col("cams_clear_sky_ghi_w_m2"),
            minimum_denominator=MIN_CLEAR_SKY_W_M2,
        ),
    )
    if "ssrdc" in wanted:
        kept = kept.with_columns(
            clear_sky_index=ratio_index(
                numerator=pl.col("ssrd"),
                denominator=pl.col("ssrdc"),
                minimum_denominator=MIN_CLEAR_SKY_W_M2,
            )
        )
    kept = add_time_features(dataset=kept)
    complete = [name for name in wanted if name not in MISSING_UNDER_CLEAR_SKY_VARIABLES]
    check_no_missing(
        frame=kept,
        columns=[
            *complete,
            "power_mw",
            "cams_ghi_w_m2",
            "cams_toa_w_m2",
            "cams_clearness_index",
            "cams_clear_sky_ghi_w_m2",
        ],
    )
    final = with_export_cap(dataset=kept).drop(*NEAREST_CELL_COLUMNS, "latitude", "longitude")
    final = final.sort("site", "time")

    return final, build_checks(rows=final, through_rung=through_rung, end=end)


def build_checks(*, rows: pl.DataFrame, through_rung: RungType, end: datetime | None) -> str:
    """Write the build's checks as markdown: the span, the rows per farm, and the missing values.

    Args:
        rows: The kept rows.
        through_rung: The highest rung the frame holds.
        end: The start of the first month that is not final ERA5 in every file, or `None`.

    Returns:
        The markdown text.
    """
    toa_ratio = (
        rows.group_by("site")
        .agg(ratio=(pl.col("cams_toa_w_m2") / pl.col("extraterrestrial_horizontal_w_m2")).median())
        .sort("site")
        .to_dicts()
    )
    missing_clear_sky = (
        int(rows["clear_sky_index"].is_null().sum()) if "clear_sky_index" in rows.columns else "n/a"
    )
    final_span = (
        f"- Final-ERA5 span ends before {end}."
        if end is not None
        else "- Every hour is final ERA5."
    )
    lines = [
        f"# Build checks, rows through {through_rung}",
        "",
        f"- Rows: {rows.height:,}, span {rows['time'].min()} to {rows['time'].max()}.",
        final_span,
        f"- Rows per farm: {rows.group_by('site').len().sort('site').to_dicts()}.",
        f"- CAMS top-of-atmosphere over the midpoint estimate, median per farm: {toa_ratio}.",
        f"- Rows with a missing clear-sky index: {missing_clear_sky}.",
        "",
        missing_value_shares(
            rows=rows,
            names=[name for name in MISSING_UNDER_CLEAR_SKY_VARIABLES if name in rows.columns],
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build one frame and write it with its checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through-rung", choices=RUNGS, default=RUNGS[-1])
    parser.add_argument("--keep-zero-hours-with-snow", action="store_true")
    parser.add_argument("--no-aerosol", action="store_true")
    arguments = parser.parse_args()
    variant = "snow_zero_hours" if arguments.keep_zero_hours_with_snow else "main"
    frame_path = dataset_path(through_rung=arguments.through_rung, variant=variant)
    report_path = checks_path(through_rung=arguments.through_rung, variant=variant)
    refuse_to_overwrite(paths=[frame_path, report_path])
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    frame, checks = build(
        through_rung=arguments.through_rung,
        keep_zero_hours_with_snow=arguments.keep_zero_hours_with_snow,
        with_aerosol=not arguments.no_aerosol
        and arguments.through_rung == RUNGS[-1]
        and EAC4_PATH.exists(),
    )
    frame.write_parquet(frame_path)
    report_path.write_text(checks)
    _LOG.info("wrote %d rows to %s", frame.height, frame_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
