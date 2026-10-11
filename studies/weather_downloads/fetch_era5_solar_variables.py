"""Download the ERA5 variables that might explain the sunlight reaching a solar farm.

One-off throwaway script for the study of which ERA5 variables help predict solar PV output beyond
`ssrd`. It fetches 31 ERA5 hourly single-level variables from the Climate Data Store (CDS) over the
same 20 cells (the public box in `studies.era5_grid`) and the same hours (2019-09-01 to 2026-09-10)
as the held copy in `data/studies/downloads/reanalysis/ERA5/beam_diffuse/`. That copy already holds
`ssrd`, `fdir`, and `t2m`, which are not fetched again.

**Variables come in four tiers, selectable with `--tier`.** `tier1a` is total, low, medium, and
high cloud cover. `tier1b` is cloud water, cloud base, clear-sky and thermal radiation, humidity,
boundary layer height, and 10 m wind. `tier2` is snow, albedo, precipitation, convective energy,
skin temperature, surface pressure, and gusts. `tier3` is rain and snow water columns, ozone,
ultraviolet radiation, and the zero-degree level, and it is the tier to stop at if time runs
short. See
`era5_solar_variables.VARIABLES` for the list and each variable's physical limits.

**Accumulated and instantaneous fields go in separate requests.** CDS splits a mixed request by
GRIB step type, and the accumulations (`ssrdc`, `cdir`, `strd`, `tp`, `sf`, `uvb`) are totals over
the hour ending at the stamp, where every other field is a snapshot at the stamp.

**Each request covers at most 4 variables over one calendar half-year.** CDS counts a request in
fields and rejects one over 121,000. One variable-hour counts for 6 fields (measured: 78,192 for 3
variables over 4,344 hours, and 104,832 for 4 variables over 4,368 hours), so a half-year of 4
variables costs at most 105,984. Larger requests would not be faster, because CDS runs one job per
account at a time and its time is set by the fields requested. Exactly one request is in flight at a
time. Each chunk is written to `_chunks/<chunk id>.zip` as soon as it arrives, and a re-run skips
every chunk that is already there and valid.

**Whole months are requested and trimmed to the span afterwards.** The tables hold hourly stamps
from 2019-09-01 00:00 to 2026-09-10 23:00 UTC, 20 cells each.

Run it with `uv run --with cdsapi --with netCDF4 python
studies/weather_downloads/fetch_era5_solar_variables.py`, adding:

- `--dry-run` to list the chunks, their sizes, and the first request body without contacting CDS;
- `--pilot` to fetch one month (2025-06) of three groups into `_pilot/` and print the measured
  seconds of each: the first `tier1a` instantaneous group, the first `tier1b` accumulation group,
  and the first `tier2` instantaneous group;
- `--tier tier1a tier1b` (any of the four names) to choose tiers; the default is all four;
- `--no-assemble` to download only.

**Interrupting the run deletes the queued job.** `Ctrl-C` and `SIGTERM` raise inside the wait, and
`retrieve_with_cleanup` deletes the remote job before the process exits, so the account's single
job slot is freed. A hard kill (`SIGKILL`, an out-of-memory kill, a power cut) runs no cleanup and
leaves the job queued. The job id is logged when the request is submitted, and
`ecmwf.datastores.Client().delete(job_id)` removes it. A wait over 12 hours also deletes the job.

Requires a CDS token in `~/.cdsapirc` and an account that has accepted the "licence to use
Copernicus products". The key is read by `cdsapi` and is never logged or written.
"""

import argparse
import json
import logging
import signal
import statistics
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import polars as pl
from era5_solar_variables import (
    CDS_DATASET,
    CELL_COUNT,
    COST_FIELDS_PER_VARIABLE_HOUR,
    FIRST_HOUR,
    LAST_HOUR,
    MAX_VARIABLES_PER_REQUEST,
    PILOT_MONTH,
    TIERS,
    VARIABLES_BY_NAME,
    ChunkRecord,
    PlannedChunk,
    TierType,
    check_cells,
    chunk_is_valid,
    expected_rows,
    last_final_month,
    pilot_chunks,
    plan_chunks,
    read_archive,
    request_body,
    retrieve_with_cleanup,
    run_chunks,
    trim_to_span,
)
from lineage import write_lineage_note, write_readme
from studies.era5_grid import GRID_LATITUDES, GRID_LONGITUDES
from studies.sources import ERA5_SOLAR_VARIABLES_DIR, SCRATCH_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_era5_solar_variables")

SCRIPT_PATH: Final[str] = "studies/weather_downloads/fetch_era5_solar_variables.py"
SOURCE_PAGE: Final[str] = "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels"
PILOT_DIR_NAME: Final[str] = "_pilot"
CHUNK_DIR_NAME: Final[str] = "_chunks"
SECONDS_PER_VARIABLE_HOUR: Final[float] = 150 / 2200
"""The measured CDS rate for a small box: about 2.5 minutes per 2,200 variable-hours."""


def _credentials() -> tuple[str, str]:
    """Read the Copernicus URL and API key from `~/.cdsapirc`.

    `ecmwf.datastores.Client` looks for `~/.ecmwfdatastoresrc` when it is given no credentials, so
    the script passes the ones `cdsapi` already uses.

    Returns:
        The URL and the key. The key must never be logged or written.

    Raises:
        RuntimeError: If the file holds no `url:` line or no `key:` line.
    """
    values: dict[str, str] = {}
    for line in Path.home().joinpath(".cdsapirc").read_text().splitlines():
        name, _, value = line.partition(":")
        values[name.strip()] = value.strip()
    if "url" not in values or "key" not in values:
        msg = "~/.cdsapirc needs a url: line and a key: line"
        raise RuntimeError(msg)
    return values["url"], values["key"]


def _download(chunk: PlannedChunk, destination: Path) -> None:
    """Submit one request to CDS and write the archive to `destination`.

    The remote job is deleted if the wait is interrupted or fails; see `retrieve_with_cleanup`.
    """
    from ecmwf.datastores import Client  # ty: ignore[unresolved-import]  # `--with cdsapi` only

    url, key = _credentials()
    retrieve_with_cleanup(
        client=Client(url=url, key=key, progress=False),
        collection=CDS_DATASET,
        request=request_body(chunk=chunk),
        destination=destination,
        log=_LOG.info,
    )


def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
    """Turn `SIGTERM` into `KeyboardInterrupt`, so the running job is deleted on the way out."""
    raise KeyboardInterrupt


def _print_plan(*, chunks: Sequence[PlannedChunk]) -> None:
    """Print the chunks per group, their sizes, and the first request body."""
    groups: dict[tuple[str, str, int], list[PlannedChunk]] = {}
    for chunk in chunks:
        groups.setdefault((chunk.tier, chunk.kind, chunk.group_index), []).append(chunk)
    for (tier, kind, index), members in groups.items():
        largest = max(members, key=lambda c: c.cost_fields)
        print(
            f"{tier} {kind} group {index}: {len(members)} chunks, variables "
            f"{','.join(members[0].variables)}; largest chunk {largest.variable_hours} "
            f"variable-hours = {largest.cost_fields} store fields"
        )
    for tier in TIERS:
        in_tier = [c for c in chunks if c.tier == tier]
        if in_tier:
            hours = sum(c.variable_hours for c in in_tier)
            print(
                f"{tier}: {len(in_tier)} chunks, {hours} variable-hours, about "
                f"{hours * SECONDS_PER_VARIABLE_HOUR / 3600:.1f} h of queue at the measured rate"
            )
    print(f"total: {len(chunks)} chunks")
    if chunks:
        print("first request body:")
        print(json.dumps(request_body(chunk=chunks[0])))


def _read_log(*, path: Path) -> dict[str, dict[str, Any]]:
    """Return the chunk log written by earlier runs, keyed by chunk id."""
    return json.loads(path.read_text()) if path.exists() else {}


def _update_log(
    *, path: Path, records: Sequence[ChunkRecord], log: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Merge this run's records into the log and write it. A skipped chunk keeps its old seconds."""
    for record in records:
        previous = log.get(record.chunk_id, {})
        entry = record._asdict()
        entry["variables"] = list(record.variables)
        if record.skipped:
            entry["seconds"] = previous.get("seconds")
        entry.pop("skipped")
        log[record.chunk_id] = entry
    path.write_text(json.dumps(log, indent=2))
    return log


def _assemble_variable(
    *,
    name: str,
    parts: list[pl.DataFrame],
    first_hour: datetime,
    last_hour: datetime,
    output_dir: Path,
) -> dict[str, Any]:
    """Write one variable's table and return its manifest entry.

    Args:
        name: The variable's short name.
        parts: The variable's frames, one per chunk.
        first_hour: The first hourly stamp kept.
        last_hour: The last hourly stamp kept.
        output_dir: Where the parquet goes.

    Returns:
        The row counts, schema, span, `expver` counts, last final month, and missing counts.

    Raises:
        ValueError: If the cells are not the public 20, a (time, cell) repeats, or the row count is
            not 20 cells times the hours.
    """
    frame = trim_to_span(frame=pl.concat(parts), first_hour=first_hour, last_hour=last_hour)
    check_cells(frame=frame)
    frame = frame.sort("time", "latitude", "longitude")
    key = ["time", "latitude", "longitude"]
    duplicates = frame.height - frame.select(key).unique().height
    wanted = expected_rows(first_hour=first_hour, last_hour=last_hour)
    if duplicates or frame.height != wanted:
        msg = f"{name}: {frame.height} rows, {duplicates} duplicates, expected {wanted}"
        raise ValueError(msg)
    path = output_dir / f"{name}.parquet"
    partial = path.with_suffix(".parquet.partial")
    frame.write_parquet(partial)
    partial.rename(path)
    hours = frame.select("time", "expver").unique()
    return {
        "rows": frame.height,
        "expected_rows": wanted,
        "schema": {column: str(dtype) for column, dtype in frame.schema.items()},
        "first_time": str(frame["time"].min()),
        "last_time": str(frame["time"].max()),
        "hours_by_expver": dict(hours.group_by("expver").len().sort("expver").iter_rows()),
        "last_final_month": last_final_month(frame=frame),
        "nan_rows": int(frame["value"].is_nan().sum()),
        "null_rows": int(frame["value"].null_count()),
    }


def assemble(
    *,
    chunks: Sequence[PlannedChunk],
    chunk_dir: Path,
    output_dir: Path,
    first_hour: datetime,
    last_hour: datetime,
) -> dict[str, dict[str, Any]]:
    """Write a parquet for every variable whose chunks are all on disk, and return their entries.

    Args:
        chunks: The planned chunks of the run.
        chunk_dir: Where the archives are.
        output_dir: Where the parquets go.
        first_hour: The first hourly stamp kept.
        last_hour: The last hourly stamp kept.

    Returns:
        The manifest entry of each variable written, keyed by short name.
    """
    present = {c.chunk_id for c in chunks if chunk_is_valid(path=chunk_dir / f"{c.chunk_id}.zip")}
    parts: dict[str, list[pl.DataFrame]] = {}
    units: dict[str, str] = {}
    scratch = SCRATCH_DIR / "era5_solar_variables"
    for chunk in chunks:
        if chunk.chunk_id not in present:
            continue
        archive = read_archive(path=chunk_dir / f"{chunk.chunk_id}.zip", scratch_dir=scratch)
        missing = set(chunk.variables) - set(archive.frames)
        if missing:
            msg = (
                f"{chunk.chunk_id}: the archive lacks {sorted(missing)}; "
                f"it holds {sorted(archive.frames)}"
            )
            raise ValueError(msg)
        for name in chunk.variables:
            parts.setdefault(name, []).append(archive.frames[name])
            units[name] = archive.units[name]
    entries = {}
    for name, frames in parts.items():
        wanted = {c.chunk_id for c in chunks if name in c.variables}
        if not wanted <= present:
            _LOG.info("%s: %d chunks still missing, not assembled", name, len(wanted - present))
            continue
        entry = _assemble_variable(
            name=name,
            parts=frames,
            first_hour=first_hour,
            last_hour=last_hour,
            output_dir=output_dir,
        )
        variable = VARIABLES_BY_NAME[name]
        entry |= {
            "tier": variable.tier,
            "kind": variable.kind,
            "units_in_file": units[name],
            "units_expected": variable.expected_units,
            "cds_name": variable.cds_name,
        }
        entries[name] = entry
        _LOG.info("%s: wrote %d rows", name, entry["rows"])
    return entries


def _wall_time_summary(*, log: dict[str, dict[str, Any]]) -> str:
    """Return a sentence on the measured seconds per chunk."""
    seconds = [e["seconds"] for e in log.values() if e.get("seconds") is not None]
    if not seconds:
        return "No chunk wall times were recorded."
    return (
        f"{len(seconds)} chunks were timed: the median took {statistics.median(seconds):.0f} s, "
        f"the fastest {min(seconds):.0f} s, the slowest {max(seconds):.0f} s, and all together "
        f"{sum(seconds) / 3600:.2f} h."
    )


def write_documentation(
    *,
    output_dir: Path,
    manifest: dict[str, dict[str, Any]],
    log: dict[str, dict[str, Any]],
    first_hour: datetime,
    last_hour: datetime,
) -> None:
    """Write `manifest.json`, `lineage.json`, and `README.md` from the measured manifest.

    Every number and dtype in the README is read from `manifest`, which is computed from the
    written frames.

    Args:
        output_dir: The product folder.
        manifest: The manifest entry of every variable written so far.
        log: The chunk log.
        first_hour: The first hourly stamp of the span.
        last_hour: The last hourly stamp of the span.
    """
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "scope": {
                    "first_hour": str(first_hour),
                    "last_hour": str(last_hour),
                    "cells": CELL_COUNT,
                    "expected_rows": expected_rows(first_hour=first_hour, last_hour=last_hour),
                },
                "variables": manifest,
            },
            indent=2,
        )
    )
    write_lineage_note(
        product_dir=output_dir,
        source_address="https://cds.climate.copernicus.eu/api",
        request_description=(
            f"ERA5 hourly data on single levels (`{CDS_DATASET}`), netCDF in zip, product type "
            f"reanalysis, every hour of every day, the public box in `studies.era5_grid.AREA`. "
            f"At most {MAX_VARIABLES_PER_REQUEST} variables per request over one calendar "
            "half-year, accumulations and instantaneous fields in separate requests, one request "
            "at a time. Whole months were requested and trimmed to the span."
        ),
        variables=sorted(manifest),
        extra={
            "span": [str(first_hour), str(last_hour)],
            "cost_fields_per_variable_hour": COST_FIELDS_PER_VARIABLE_HOUR,
            "chunks": log,
            "note": "Per-variable row counts, expver counts and last final month: manifest.json.",
        },
    )
    done = sorted(manifest)
    nan_text = "; ".join(
        f"`{n}` {manifest[n]['nan_rows']} of {manifest[n]['rows']}"
        for n in done
        if manifest[n]["nan_rows"]
    )
    final_text = "; ".join(f"`{n}` {manifest[n]['last_final_month'] or 'none'}" for n in done)
    cell_text = f"{len(GRID_LATITUDES)} latitudes by {len(GRID_LONGITUDES)} longitudes"
    schema = next(iter(manifest.values()))["schema"] if manifest else {}
    write_readme(
        product_dir=output_dir,
        product_name="ERA5 hourly single-level variables for the solar-variables study",
        source_web_page=SOURCE_PAGE,
        script_path=SCRIPT_PATH,
        lineage_filenames=["lineage.json"],
        columns={
            "time": f"Valid time, UTC, stored as {schema.get('time', 'n/a')}. ERA5 labels the hour "
            "ending at this stamp, the same stamp the held `beam_diffuse/` copy carries.",
            "latitude": f"Cell latitude, degrees north, {schema.get('latitude', 'n/a')}; "
            f"{len(GRID_LATITUDES)} values on the 0.25 degree grid.",
            "longitude": f"Cell longitude, degrees east, {schema.get('longitude', 'n/a')}; "
            f"{len(GRID_LONGITUDES)} values on the 0.25 degree grid.",
            "value": f"The variable named by the file, {schema.get('value', 'n/a')}, in the unit "
            "ERA5 documents for it (listed per file below). NaN is kept as NaN.",
            "expver": f"ERA5 release of the hour, {schema.get('expver', 'n/a')}: `0001` final, "
            "`0005` preliminary (ERA5T).",
            **{
                f"value in {n}.parquet": f"{manifest[n]['cds_name']}, {manifest[n]['kind']}, "
                f"unit in the file: `{manifest[n]['units_in_file']}`"
                for n in done
            },
        },
        missing_value_convention=(
            f"Missing values are NaN, never null (null count 0 in every file). "
            f"NaN rows per file: {nan_text or 'none in any file written so far'}. "
            "`cbh` (cloud base height) is NaN where there is no cloud, and `cin` (convective "
            "inhibition) can be NaN too, so a NaN in those two files is a value. The files keep "
            "the NaN: nothing is filled or dropped. A NaN in any other file is a defect."
        ),
        gotchas=[
            (
                "**Hour convention.** The accumulations (`ssrdc`, `cdir`, `strd`, `tp`, `sf`, "
                "`uvb`) are totals over the hour ending at `time`, in J m-2 (radiation) or m "
                "of water (`tp`, `sf`); divide radiation by 3600 for the mean W m-2. Every "
                "other file is a snapshot at `time`."
            ),
            (
                f"**Release.** ERA5 mixes final (`0001`) and preliminary (`0005`) hours. Each hour "
                f"carries its own `expver`. Last month up to which every hour is final, per file: "
                f"{final_text or 'none written yet'}."
            ),
            (
                f"**Cells.** {cell_text}, the same {CELL_COUNT} cells and hours as the held "
                f"`beam_diffuse/` copy, from {first_hour:%Y-%m-%d %H:%M} to "
                f"{last_hour:%Y-%m-%d %H:%M} UTC. Whole months were requested and trimmed to "
                "this span."
            ),
            (
                f"**Request limits.** CDS rejects a request over 121,000 fields. One variable-hour "
                f"counts for {COST_FIELDS_PER_VARIABLE_HOUR} fields, so each request holds at most "
                f"{MAX_VARIABLES_PER_REQUEST} variables over one half-year, and accumulated and "
                "instantaneous fields never share a request."
            ),
            (
                "**Release across kinds.** An accumulation stamped 00 to 06 UTC on the first day "
                "of a month comes from the previous day's forecast, so its `expver` can differ "
                "from an instantaneous field at the same stamp. Each archive holds one kind, "
                "and the `expver` agreement check runs within an archive only. A study build "
                "should compare `expver` across variables of the same kind, not across kinds."
            ),
            f"**Wall time.** {_wall_time_summary(log=log)}",
            (
                "**Validation.** The findings of the `data-validation` checklist for each file "
                "are in `validation_<variable>.json`, written by "
                "`validate_era5_solar_variables.py`."
            ),
        ],
        external_docs={
            "ERA5 hourly data on single levels": SOURCE_PAGE,
            "ERA5 data documentation": "https://confluence.ecmwf.int/display/CKB/"
            "ERA5%3A+data+documentation",
            "Hersbach et al. (2020)": "https://doi.org/10.1002/qj.3803",
        },
    )


def _load_manifest(*, output_dir: Path) -> dict[str, dict[str, Any]]:
    """Return the variable entries of an earlier run's manifest."""
    path = output_dir / "manifest.json"
    return json.loads(path.read_text())["variables"] if path.exists() else {}


def main(argv: Sequence[str] | None = None) -> int:
    """Plan, fetch, and assemble the chunks of the chosen tiers."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tier", nargs="+", choices=TIERS, default=list(TIERS))
    parser.add_argument(
        "--pilot", action="store_true", help="one month of three groups into _pilot/"
    )
    parser.add_argument("--dry-run", action="store_true", help="list the plan; contact nothing")
    parser.add_argument("--no-assemble", action="store_true", help="download only")
    args = parser.parse_args(argv)

    tiers: list[TierType] = args.tier
    first_month = (FIRST_HOUR.year, FIRST_HOUR.month)
    last_month = (LAST_HOUR.year, LAST_HOUR.month)
    output_dir = (
        ERA5_SOLAR_VARIABLES_DIR / PILOT_DIR_NAME if args.pilot else ERA5_SOLAR_VARIABLES_DIR
    )
    first_hour = (
        FIRST_HOUR.replace(year=PILOT_MONTH[0], month=PILOT_MONTH[1]) if args.pilot else FIRST_HOUR
    )
    last_hour = (
        LAST_HOUR.replace(year=PILOT_MONTH[0], month=PILOT_MONTH[1], day=30)
        if args.pilot
        else LAST_HOUR
    )
    chunks = (
        pilot_chunks()
        if args.pilot
        else plan_chunks(tiers=tiers, first_month=first_month, last_month=last_month)
    )
    if args.dry_run:
        _print_plan(chunks=chunks)
        return 0
    _LOG.info(
        "%d chunks over %d variables into %s",
        len(chunks),
        len({name for chunk in chunks for name in chunk.variables}),
        output_dir.name,
    )
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    chunk_dir = output_dir / CHUNK_DIR_NAME
    log_path = chunk_dir / "chunk_log.json"
    log = _read_log(path=log_path)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    def _persist(record: ChunkRecord) -> None:
        _update_log(path=log_path, records=[record], log=log)

    records = run_chunks(
        chunks=chunks,
        chunk_dir=chunk_dir,
        download=_download,
        log=_LOG.info,
        on_record=_persist,
    )
    if args.pilot:
        for record in records:
            print(f"pilot: {record.chunk_id}: {record.seconds} s measured")
    if args.no_assemble:
        return 0
    manifest = _load_manifest(output_dir=output_dir) | assemble(
        chunks=chunks,
        chunk_dir=chunk_dir,
        output_dir=output_dir,
        first_hour=first_hour,
        last_hour=last_hour,
    )
    write_documentation(
        output_dir=output_dir,
        manifest=manifest,
        log=log,
        first_hour=first_hour,
        last_hour=last_hour,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
