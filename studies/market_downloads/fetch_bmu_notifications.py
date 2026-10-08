r"""Download the Elexon physical notifications of the study's BMUs for 2025-09-01 to 2026-09-30.

Written for the battery-versus-solar-PV study (PR 1094). A BMU is a Balancing Mechanism Unit. Three
datasets are downloaded for the BMUs in `--bmu-list` (default: the 113-BMU list of the study):

- `elexon_pn`: Physical Notifications (PN), the output a BMU tells the system operator it will
  deliver, in MW (positive is export, negative is import).
- `elexon_mels`: Maximum Export Limit (MELS), the most a BMU can export, in MW.
- `elexon_mils`: Maximum Import Limit (MILS), the most a BMU can import, in MW (zero or negative).

Run:

    uv run python studies/market_downloads/fetch_bmu_notifications.py
    uv run python studies/market_downloads/fetch_bmu_notifications.py --sources elexon_pn \\
        --start 2026-03-10 --end 2026-03-10 --output-root <scratch dir>

Each folder gets two parquet files. `<name>_points.parquet` holds the points exactly as Elexon
publishes them: a start time, an end time, and the level at each end, joined by a straight line.
`<name>_half_hourly.parquet` holds the time-weighted mean level per BMU per half-hour, found by
integrating that piecewise-linear profile over each UTC half-hour window. See each folder's
`README.md` for the method.

Requests are one UTC day for one hash batch of BMUs, cached as
`_day_cache/batch_<hash>/<day>.parquet`, so that a re-run resumes and removing one BMU from the list
refetches only that BMU's batch. Requests are keyless, use 4 threads, and back off exponentially on
HTTP 429 and 5xx.
"""

import argparse
import hashlib
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Final

import polars as pl
from market_common import (
    ELEXON_API,
    ELEXON_ATTRIBUTION,
    ELEXON_LICENCE,
    FETCH_THREADS,
    SETTLEMENT_PERIOD,
    WINDOW_END,
    WINDOW_START,
    IncompleteChunkError,
    days_between,
    expected_half_hour_starts,
    expected_settlement_starts,
    fetch_missing_chunks,
    get_json,
    read_chunks,
    summarise_gaps,
    utc_midnight,
    write_lineage,
    write_parquet_atomic,
    write_readme,
)
from studies.sources import MARKET_DOWNLOADS_DIR

SCRIPT_PATH: Final[str] = "studies/market_downloads/fetch_bmu_notifications.py"
DEFAULT_BMU_LIST: Final[Path] = Path(
    "/home/jack/dev/nged-substation-forecast/data/studies/per_study/battery_pv_separation/"
    "bmu_list.csv"
)
BATCH_COUNT: Final[int] = 16
"""BMUs are split into this many batches by a hash of each identifier, so that adding or deleting a
BMU changes the cache key of one batch only."""
MAX_URL_LENGTH: Final[int] = 4000
"""A batch whose URL would be longer than this raises instead of being sent."""
WINDOW_SECONDS: Final[int] = int(SETTLEMENT_PERIOD.total_seconds())
MICROSECONDS: Final[int] = 1_000_000
ISO_SECONDS: Final[str] = "%Y-%m-%dT%H:%M:%SZ"
UTC_TIME: Final[pl.Datetime] = pl.Datetime(time_unit="us", time_zone="UTC")

DATASETS: Final[dict[str, dict[str, str]]] = {
    "elexon_pn": {
        "dataset": "PN",
        "title": "Elexon Physical Notifications (PN), for the listed BMUs",
        "page": "https://bmrs.elexon.co.uk/physical-notifications",
        "meaning": "the output the BMU notified it would deliver (positive is export)",
    },
    "elexon_mels": {
        "dataset": "MELS",
        "title": "Elexon Maximum Export Limit (MELS), for the listed BMUs",
        "page": "https://bmrs.elexon.co.uk/maximum-export-limit",
        "meaning": "the most the BMU notified it could export",
    },
    "elexon_mils": {
        "dataset": "MILS",
        "title": "Elexon Maximum Import Limit (MILS), for the listed BMUs",
        "page": "https://bmrs.elexon.co.uk/maximum-import-limit",
        "meaning": "the most the BMU notified it could import (zero or negative)",
    },
}

POINT_SCHEMA: Final[dict[str, Any]] = {
    "bmu_id": pl.String,
    "time_from": UTC_TIME,
    "time_to": UTC_TIME,
    "level_from_mw": pl.Float64,
    "level_to_mw": pl.Float64,
    "settlement_date": pl.Date,
    "settlement_period": pl.Int16,
    "notification_time": UTC_TIME,
    "notification_sequence": pl.Int64,
}
HALF_HOURLY_SCHEMA: Final[dict[str, Any]] = {
    "bmu_id": pl.String,
    "time": UTC_TIME,
    "settlement_date": pl.Date,
    "settlement_period": pl.Int16,
    "mean_level_mw": pl.Float64,
    "covered_seconds": pl.Int32,
}


# BMU batches


def read_bmu_list(*, path: Path) -> list[str]:
    """Return the sorted, de-duplicated `elexon_bmu_id` values of a BMU list CSV."""
    ids = pl.read_csv(path, infer_schema_length=0)["elexon_bmu_id"].drop_nulls().to_list()
    return sorted({bmu_id.strip() for bmu_id in ids if bmu_id.strip()})


def list_hash(*, bmu_ids: Sequence[str]) -> str:
    """Return a short hash of a BMU list, used to name the cache folder."""
    return hashlib.sha1("\n".join(sorted(bmu_ids)).encode()).hexdigest()[:8]


def bmu_batches(*, bmu_ids: Sequence[str], count: int = BATCH_COUNT) -> list[list[str]]:
    """Split BMU identifiers into up to `count` batches by a hash of each identifier.

    A BMU's batch depends on its own identifier alone, so editing the list changes only the batches
    that gain or lose a BMU.

    Args:
        bmu_ids: The BMU identifiers.
        count: The number of hash buckets.

    Returns:
        The non-empty batches, each sorted, in bucket order.

    Raises:
        ValueError: If a batch would make a URL longer than `MAX_URL_LENGTH`.
    """
    buckets: list[list[str]] = [[] for _ in range(count)]
    for bmu_id in sorted(bmu_ids):
        bucket = int(hashlib.sha1(bmu_id.encode()).hexdigest()[:8], 16) % count
        buckets[bucket].append(bmu_id)
    batches = [batch for batch in buckets if batch]
    for batch in batches:
        length = 200 + sum(len(bmu_id) + len("&bmUnit=") for bmu_id in batch)
        if length > MAX_URL_LENGTH:
            raise ValueError(f"A batch of {len(batch)} BMUs would make a {length}-byte URL")
    return batches


# Parsing and fetching


def parse_points(*, rows: list[dict[str, Any]], day: date) -> pl.DataFrame:
    """Turn one UTC day of PN, MELS, or MILS rows into a table of points.

    Elexon answers a request for a day with every point that touches the day, including the points
    that end exactly at its start and the ones that start exactly at its end. Each point is kept in
    the day its `timeFrom` falls in, so a point is cached once. If two rows share a BMU and
    `timeFrom` (MELS and MILS notifications can be revised), the one with the highest notification
    sequence wins.

    Args:
        rows: The decoded JSON rows.
        day: The UTC day requested.

    Returns:
        The points whose `time_from` is in `day`, one row per BMU and `time_from`.
    """
    frame = pl.DataFrame(
        {
            "bmu_id": [row["bmUnit"] for row in rows],
            "time_from": [row["timeFrom"] for row in rows],
            "time_to": [row["timeTo"] for row in rows],
            "level_from_mw": [row["levelFrom"] for row in rows],
            "level_to_mw": [row["levelTo"] for row in rows],
            "settlement_date": [row["settlementDate"] for row in rows],
            "settlement_period": [row["settlementPeriod"] for row in rows],
            "notification_time": [row.get("notificationTime") for row in rows],
            "notification_sequence": [row.get("notificationSequence") for row in rows],
        },
        schema={
            "bmu_id": pl.String,
            "time_from": pl.String,
            "time_to": pl.String,
            "level_from_mw": pl.Float64,
            "level_to_mw": pl.Float64,
            "settlement_date": pl.String,
            "settlement_period": pl.Int16,
            "notification_time": pl.String,
            "notification_sequence": pl.Int64,
        },
    ).with_columns(
        pl.col("time_from", "time_to", "notification_time").str.to_datetime(
            format=ISO_SECONDS, time_zone="UTC", time_unit="us"
        ),
        pl.col("settlement_date").str.to_date(),
    )
    first = utc_midnight(day=day)
    after = first + timedelta(days=1)
    return (
        frame.filter(pl.col("time_from") >= first, pl.col("time_from") < after)
        .sort("notification_sequence", nulls_last=False)
        .unique(subset=["bmu_id", "time_from"], keep="last")
        .sort("bmu_id", "time_from")
        .select(list(POINT_SCHEMA))
        .cast(pl.Schema(POINT_SCHEMA))
    )


def fetch_day(key: str, *, dataset: str, batch: list[str]) -> pl.DataFrame:
    """Fetch one UTC day of `dataset` for one batch of BMUs."""
    day = date.fromisoformat(key)
    first = utc_midnight(day=day)
    after = first + timedelta(days=1)
    rows = get_json(
        url=f"{ELEXON_API}/datasets/{dataset}/stream",
        params=[
            ("from", f"{first:%Y-%m-%dT%H:%MZ}"),
            ("to", f"{after:%Y-%m-%dT%H:%MZ}"),
            *[("bmUnit", bmu_id) for bmu_id in batch],
        ],
    )
    if not isinstance(rows, list):
        raise TypeError(f"{dataset} {key} returned {str(rows)[:200]!r}, not a list of rows")
    return parse_points(rows=rows, day=day)


# Half-hourly means


def half_hourly_means(*, points: pl.DataFrame, period_starts: pl.DataFrame) -> pl.DataFrame:
    """Average piecewise-linear profiles over UTC half-hour windows, weighting by time.

    Each point is a straight line from `(time_from, level_from_mw)` to `(time_to, level_to_mw)`. A
    point that crosses a window boundary is split at the boundary, and the level at the cut is
    found by linear interpolation. The area of each piece is its duration times the mean of its
    two end levels. A window's mean is the summed area divided by the summed duration of the pieces
    in it, so a window only partly covered by points is the mean over the covered part, and
    `covered_seconds` says how much that was. A point with `time_to <= time_from` has no duration
    and adds nothing.

    Args:
        points: Points with the `POINT_SCHEMA` columns.
        period_starts: A `time`, `settlement_date`, `settlement_period` table for every window.

    Returns:
        One row per BMU and window that has at least one second of coverage.
    """
    seconds = (
        points.select(
            "bmu_id",
            start=pl.col("time_from").dt.epoch("s"),
            stop=pl.col("time_to").dt.epoch("s"),
            level_start=pl.col("level_from_mw"),
            level_stop=pl.col("level_to_mw"),
        )
        .filter(pl.col("stop") > pl.col("start"))
        .with_columns(
            window=pl.int_ranges(
                pl.col("start") // WINDOW_SECONDS,
                (pl.col("stop") - 1) // WINDOW_SECONDS + 1,
            )
        )
        .explode("window")
        .with_columns(
            piece_start=pl.max_horizontal(pl.col("start"), pl.col("window") * WINDOW_SECONDS),
            piece_stop=pl.min_horizontal(pl.col("stop"), (pl.col("window") + 1) * WINDOW_SECONDS),
        )
    )

    def level_at(column: str) -> pl.Expr:
        slope = (pl.col("level_stop") - pl.col("level_start")) / (pl.col("stop") - pl.col("start"))
        return pl.col("level_start") + slope * (pl.col(column) - pl.col("start"))

    pieces = seconds.with_columns(
        duration=pl.col("piece_stop") - pl.col("piece_start"),
        area=(level_at("piece_start") + level_at("piece_stop"))
        / 2
        * (pl.col("piece_stop") - pl.col("piece_start")),
    )
    return (
        pieces.group_by("bmu_id", "window")
        .agg(area=pl.col("area").sum(), covered_seconds=pl.col("duration").sum())
        .select(
            "bmu_id",
            time=(pl.col("window") * WINDOW_SECONDS * MICROSECONDS).cast(UTC_TIME),
            mean_level_mw=pl.col("area") / pl.col("covered_seconds"),
            covered_seconds=pl.col("covered_seconds").cast(pl.Int32),
        )
        .join(period_starts, on="time", how="left")
        .select(list(HALF_HOURLY_SCHEMA))
        .sort("bmu_id", "time")
    )


def period_start_table(*, start: date, end: date) -> pl.DataFrame:
    """Return the UTC start, settlement date, and settlement period of every period in the window.

    The table covers the settlement dates from the day before `start` to the day after `end`, so
    that every UTC half-hour of `start..end` is in it.
    """
    rows = []
    for day in days_between(start=start - timedelta(days=1), end=end + timedelta(days=1)):
        starts = expected_settlement_starts(start=day, end=day)
        rows += [(stamp, day, period) for period, stamp in enumerate(starts, start=1)]
    return pl.DataFrame(
        rows,
        schema={"time": UTC_TIME, "settlement_date": pl.Date, "settlement_period": pl.Int16},
        orient="row",
    )


# Orchestration


def empty_chunks(*, cache_dir: Path, keys: Sequence[str]) -> list[str]:
    """Return the cached day chunks of a batch that hold no rows, in key order."""
    return [
        key
        for key in keys
        if (path := cache_dir / f"{key}.parquet").exists()
        and pl.scan_parquet(path).select(pl.len()).collect().item() == 0
    ]


def check_no_empty_month(*, label: str, empty: Sequence[str], keys: Sequence[str]) -> None:
    """Raise if every requested day of any calendar month is an empty chunk.

    Elexon answers HTTP 200 with an empty list for data it has not published, so one empty day
    can be real (a BMU switched off), but a whole month of empty days for a batch means the batch
    was not published or the request was wrong.

    Args:
        label: Names the dataset and batch in the error message.
        empty: The keys of the batch's empty chunks.
        keys: Every key requested.

    Raises:
        IncompleteChunkError: If some month has all its requested days empty. Delete that batch's
            cached chunks for the month once the cause is understood.
    """
    empty_set = set(empty)
    for month in sorted({key[:7] for key in keys}):
        month_keys = [key for key in keys if key.startswith(month)]
        if all(key in empty_set for key in month_keys):
            raise IncompleteChunkError(f"{label}: all {len(month_keys)} days of {month} are empty")


def run_dataset(
    *,
    name: str,
    root: Path,
    bmu_ids: Sequence[str],
    start: date,
    end: date,
    threads: int,
) -> None:
    """Download one dataset for every BMU, write its points and half-hourly means, and README."""
    info = DATASETS[name]
    output_dir = root / name
    batches = bmu_batches(bmu_ids=bmu_ids)
    days = [day.isoformat() for day in days_between(start=start, end=end)]
    chunks: dict[str, list[str]] = {"fetched": [], "cached": [], "not_published": []}
    parts = []
    empty_by_batch: dict[str, list[str]] = {}
    for batch in batches:
        cache_dir = output_dir / "_day_cache" / f"batch_{list_hash(bmu_ids=batch)}"
        outcome = fetch_missing_chunks(
            cache_dir=cache_dir,
            keys=days,
            fetch=partial(fetch_day, dataset=info["dataset"], batch=batch),
            label=f"{name} {cache_dir.name}",
            threads=threads,
        )
        for kind, keys in outcome.items():
            chunks[kind] += keys
        empty = empty_chunks(cache_dir=cache_dir, keys=days)
        check_no_empty_month(label=f"{name} {cache_dir.name}", empty=empty, keys=days)
        empty_by_batch[cache_dir.name] = empty
        parts.append(read_chunks(cache_dir=cache_dir, keys=days, schema=POINT_SCHEMA))
    points = pl.concat(parts).sort("bmu_id", "time_from")
    write_parquet_atomic(frame=points, path=output_dir / f"{name}_points.parquet")
    means = half_hourly_means(
        points=points, period_starts=period_start_table(start=start, end=end)
    ).filter(
        pl.col("time") >= utc_midnight(day=start),
        pl.col("time") < utc_midnight(day=end + timedelta(days=1)),
    )
    write_parquet_atomic(frame=means, path=output_dir / f"{name}_half_hourly.parquet")
    quality = quality_summary(means=means, bmu_ids=bmu_ids, start=start, end=end)
    write_lineage(
        product_dir=output_dir,
        note={
            "source_address": f"{ELEXON_API}/datasets/{info['dataset']}/stream",
            "request": f"{len(bmu_ids)} BMUs in {len(batches)} hash batches, one request per "
            f"UTC day and batch, {start} to {end}",
            "window": [start.isoformat(), end.isoformat()],
            "script": SCRIPT_PATH,
            "bmus_requested": list(bmu_ids),
            "bmu_list_hash": list_hash(bmu_ids=bmu_ids),
            "point_rows": points.height,
            "half_hourly_rows": means.height,
            "zero_row_chunks_by_batch": {k: v for k, v in empty_by_batch.items() if v},
            "chunks_not_published": sorted(set(chunks["not_published"])),
            **quality,
        },
    )
    write_notifications_readme(
        name=name,
        output_dir=output_dir,
        points=points,
        half_hourly_rows=means.height,
        quality=quality,
        bmu_count=len(bmu_ids),
        start=start,
        end=end,
    )
    print(f"{name}: {points.height} points, {means.height} half-hourly rows")


def quality_summary(
    *, means: pl.DataFrame, bmu_ids: Sequence[str], start: date, end: date
) -> dict[str, Any]:
    """Measure how much of the window the half-hourly means cover.

    Args:
        means: The half-hourly table.
        bmu_ids: The BMUs requested.
        start: First UTC day of the window.
        end: Last UTC day of the window.

    Returns:
        The BMUs with no rows, how many windows have a partial or an over-full cover, the windows no
        BMU covers, and the rows each BMU has against the windows in the window.
    """
    expected = expected_half_hour_starts(start=start, end=end)
    per_bmu = means.group_by("bmu_id").agg(rows=pl.len())
    return {
        "bmus_with_rows": per_bmu.height,
        "bmus_without_rows": sorted(set(bmu_ids) - set(per_bmu["bmu_id"].to_list())),
        "windows_per_bmu_expected": len(expected),
        "rows_per_bmu_min_median_max": (
            [per_bmu["rows"].min(), per_bmu["rows"].median(), per_bmu["rows"].max()]
            if per_bmu.height
            else []
        ),
        "windows_partly_covered": means.filter(pl.col("covered_seconds") < WINDOW_SECONDS).height,
        "windows_covered_more_than_once": means.filter(
            pl.col("covered_seconds") > WINDOW_SECONDS
        ).height,
        "windows_no_bmu_has_rows": summarise_gaps(expected=expected, actual=means["time"]),
    }


def write_notifications_readme(
    *,
    name: str,
    output_dir: Path,
    points: pl.DataFrame,
    half_hourly_rows: int,
    quality: dict[str, Any],
    bmu_count: int,
    start: date,
    end: date,
) -> None:
    """Write the README of one notification dataset."""
    info = DATASETS[name]
    no_row_windows = quality["windows_no_bmu_has_rows"]["missing_count"]
    write_readme(
        product_dir=output_dir,
        title=info["title"],
        source_page=info["page"],
        script_path=SCRIPT_PATH,
        attribution=ELEXON_ATTRIBUTION,
        cache_hint="_day_cache/batch_<hash>/",
        licence=ELEXON_LICENCE,
        timestamp_convention=(
            f"Every time is UTC. `{name}_points.parquet` holds Elexon's points: from `time_from` "
            "to `time_to` the level moves in a straight line from `level_from_mw` to "
            "`level_to_mw`. `time_from` is the instant the point starts, not a settlement "
            f"period start. `{name}_half_hourly.parquet` has one row per BMU and UTC half-hour "
            "window; `time` is the start of the window, which equals the start of a settlement "
            "period (settlement periods start on the hour and half hour in UTC). `settlement_date` "
            "and `settlement_period` are the Elexon period that window is: period 1 starts at "
            "00:00 UK local time. The window is by UTC day, "
            f"{start} to {end}, and a point is kept in the UTC day its `time_from` falls in."
        ),
        columns={
            "bmu_id": "Elexon BMU identifier",
            "time_from": "points file: start of the point, UTC",
            "time_to": "points file: end of the point, UTC",
            "level_from_mw": f"points file: level at `time_from`, MW; {info['meaning']}",
            "level_to_mw": "points file: level at `time_to`, MW",
            "settlement_date": "Elexon settlement date (UK local day)",
            "settlement_period": "Elexon settlement period within the date, from 1",
            "notification_time": "points file: when the notification was made, UTC. Null for PN",
            "notification_sequence": "points file: Elexon notification sequence. Null for PN",
            "time": "half-hourly file: start of the half-hour window, UTC",
            "mean_level_mw": (
                "half-hourly file: time-weighted mean level over the part of the window the "
                "points cover, MW"
            ),
            "covered_seconds": (
                "half-hourly file: seconds of the window the points cover. 1800 is full coverage"
            ),
        },
        row_summary=(
            f"- Points written: {points.height}, half-hourly rows: {half_hourly_rows}; "
            f"{quality['bmus_with_rows']} of the {bmu_count} BMUs requested have rows; BMUs "
            f"without rows: {quality['bmus_without_rows']}.\n"
            f"- Windows per BMU in the window: {quality['windows_per_bmu_expected']}. Rows per "
            f"BMU (min, median, max): {quality['rows_per_bmu_min_median_max']}.\n"
            f"- Rows with `covered_seconds` below 1800: {quality['windows_partly_covered']}; "
            f"above 1800: {quality['windows_covered_more_than_once']}.\n"
            f"- Windows in which no BMU has a row: {no_row_windows}."
            f" The first of them are in `lineage.json`.\n"
            f"- Window: {start} to {end}."
        ),
        gotchas=[
            (
                "The half-hourly mean is the exact integral of the straight-line profile, not "
                "a sample of it. Where points do not cover the whole window (`covered_seconds` "
                "below 1800) the mean is over the covered part only, so do not treat it as a "
                "full-window mean. Where points overlap (`covered_seconds` above 1800) the "
                "level is averaged over the doubled time."
            ),
            (
                "Elexon's stream returns the points as last published, so a PN, MELS, or MILS "
                "revised before the retrieval date shows only its final version. The table does "
                "not say what was known at a given earlier time."
            ),
            (
                "A BMU that did not exist, or was not required to notify, on a day has no points "
                "that day. Absence is not zero."
            ),
            (
                "The BMU list is the CSV passed with `--bmu-list`; `lineage.json` lists the BMUs "
                "requested. Each batch of BMUs is cached separately."
            ),
        ],
    )


def main() -> None:
    """Download the requested notification datasets."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, default=MARKET_DOWNLOADS_DIR)
    parser.add_argument("--bmu-list", type=Path, default=DEFAULT_BMU_LIST)
    parser.add_argument("--start", type=date.fromisoformat, default=WINDOW_START)
    parser.add_argument("--end", type=date.fromisoformat, default=WINDOW_END)
    parser.add_argument("--sources", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--threads", type=int, default=FETCH_THREADS)
    args = parser.parse_args()
    bmu_ids = read_bmu_list(path=args.bmu_list)
    if not bmu_ids:
        raise ValueError(f"{args.bmu_list} lists no BMUs")
    print(f"{len(bmu_ids)} BMUs")
    for name in args.sources:
        run_dataset(
            name=name,
            root=args.output_root,
            bmu_ids=bmu_ids,
            start=args.start,
            end=args.end,
            threads=args.threads,
        )
    print(f"Finished at {datetime.now(UTC):%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
