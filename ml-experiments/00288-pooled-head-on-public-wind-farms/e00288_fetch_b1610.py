"""Download Elexon B1610 half-hourly metered output of the experiment 00288 wind BMUs.

Reads `work/bmu_list.csv` (written by `e00288_farms.py`) and fetches, for each BMU, one request per
calendar half-year from 2019-09-01 to 2026-07-01 (exclusive). Each chunk is written atomically to
`data/studies/downloads/market_wind00288/elexon_b1610/_chunks/<bmu>_<start>.parquet`, and a re-run
skips the chunks on disk. The combined file `elexon_b1610.parquet` holds `elexon_bmu_id`,
`half_hour_end_time` (UTC) and `output_mwh` (energy in the half-hour, so MW is twice the value).

    uv run --no-sync python ml-experiments/00288-*/e00288_fetch_b1610.py

Contains BMRS data © Elexon Limited copyright and database right 2026.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Final

import httpx
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

WORK_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00288-pooled-head-on-public-wind-farms" / "work"
OUT_DIR: Final = REPO_DATA_DIR / "studies/downloads/market_wind00288/elexon_b1610"
CHUNK_DIR: Final = OUT_DIR / "_chunks"
API: Final[str] = "https://data.elexon.co.uk/bmrs/api/v1/datasets/B1610/stream"
FIRST: Final = datetime(2019, 9, 1, tzinfo=UTC)
END: Final = datetime(2026, 7, 1, tzinfo=UTC)
THREADS: Final[int] = 4
RETRIES: Final[int] = 5


def chunk_starts() -> list[datetime]:
    """Return the first instant of each half-year (1 September, 1 March, ...) before `END`."""
    starts, year, month = [], FIRST.year, FIRST.month
    while (current := datetime(year, month, 1, tzinfo=UTC)) < END:
        starts.append(current)
        year, month = (year + 1, month - 6) if month > 6 else (year, month + 6)
    return starts


def request(*, bmu: str, start: datetime, end: datetime) -> list[dict]:
    """GET one chunk, retrying transient failures; raise on a final failure."""
    params = {
        "from": f"{start:%Y-%m-%dT%H:%MZ}",
        "to": f"{end:%Y-%m-%dT%H:%MZ}",
        "bmUnit": bmu,
    }
    for attempt in range(RETRIES):
        try:
            response = httpx.get(API, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError, httpx.TransportError:
            if attempt == RETRIES - 1:
                raise
            time.sleep(5 * 2**attempt)
    raise AssertionError("unreachable")


def fetch_chunk(job: tuple[str, datetime, datetime]) -> int:
    """Fetch one BMU chunk to its own file unless cached; return the row count (0 if cached)."""
    bmu, start, end = job
    path = CHUNK_DIR / f"{bmu}_{start:%Y%m%d}.parquet"
    if path.exists():
        return 0
    rows = request(bmu=bmu, start=start, end=end)
    frame = (
        pl.DataFrame(
            {
                "half_hour_end_time": [row["halfHourEndTime"] for row in rows],
                "output_mwh": [row["quantity"] for row in rows],
            },
            schema={"half_hour_end_time": pl.String, "output_mwh": pl.Float64},
        )
        .with_columns(
            pl.col("half_hour_end_time").str.to_datetime(
                format=None, time_zone="UTC", time_unit="us"
            )
        )
        .filter(pl.col("half_hour_end_time") > start, pl.col("half_hour_end_time") <= end)
        .unique(subset="half_hour_end_time", keep="last")
        .sort("half_hour_end_time")
    )
    partial = path.with_suffix(".parquet.partial")
    frame.write_parquet(partial)
    partial.rename(path)
    return frame.height


def main() -> None:
    """Fetch every missing chunk, then combine the chunks of the requested BMUs."""
    bmus = pl.read_csv(WORK_DIR / "bmu_list.csv")["elexon_bmu_id"].to_list()
    starts = chunk_starts()
    ends = [*starts[1:], END]
    jobs = [(bmu, s, e) for bmu in bmus for s, e in zip(starts, ends, strict=True)]
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        for done, _ in enumerate(pool.map(fetch_chunk, jobs), start=1):
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} chunks, {time.monotonic() - began:.0f} s", flush=True)
    frames = []
    for bmu, start, _ in jobs:
        path = CHUNK_DIR / f"{bmu}_{start:%Y%m%d}.parquet"
        frames.append(pl.read_parquet(path).with_columns(elexon_bmu_id=pl.lit(bmu)))
    combined = pl.concat(frames).select("elexon_bmu_id", "half_hour_end_time", "output_mwh")
    combined.sort("elexon_bmu_id", "half_hour_end_time").write_parquet(
        OUT_DIR / "elexon_b1610.parquet"
    )
    print(f"wrote {combined.height} rows for {combined['elexon_bmu_id'].n_unique()} BMUs")


if __name__ == "__main__":
    main()
