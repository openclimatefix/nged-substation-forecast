"""Experiment 177, stage 2: fetch public Elexon system prices and the NESO DFS event list.

Resumable: each day of system prices is written to its own parquet file, atomically, and a re-run
skips the days already cached. Total transfer is about 15 MB. Writes under the experiment's private
data folder, never committed. Run: `PYTHONUNBUFFERED=1 uv run --no-sync python <this file>`.
"""

import json
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00177-dp-worst-days-other-data-sources" / "public"
PRICES: Final[Path] = OUT / "system_prices"
FIRST: Final[date] = date(2025, 7, 2)
LAST: Final[date] = date(2026, 7, 1)
URL: Final[str] = (
    "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{day}?format=json"
)
DFS_URL: Final[str] = (
    "https://api.neso.energy/dataset/829e325f-8926-40ab-9dd6-8b7aea6cb244/resource/"
    "f5605e2b-b677-424c-8df7-d0ce4ee03cef/download/dfs-service-requirement.csv"
)
KEEP: Final[list[str]] = [
    "settlementDate",
    "settlementPeriod",
    "startTime",
    "systemSellPrice",
    "systemBuyPrice",
    "netImbalanceVolume",
]


def get(url: str) -> bytes:
    """Fetch a URL once; raise on any HTTP error."""
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main() -> None:
    """Fetch every day of system prices and the DFS list, skipping what is cached."""
    PRICES.mkdir(parents=True, exist_ok=True)
    day = FIRST
    while day <= LAST:
        path = PRICES / f"{day}.parquet"
        if not path.exists():
            rows = json.loads(get(URL.format(day=day)))["data"]
            frame = pl.DataFrame(rows, infer_schema_length=None).select(KEEP)
            partial = path.with_suffix(".parquet.partial")
            frame.write_parquet(partial)
            partial.rename(path)
            time.sleep(0.2)
        day += timedelta(days=1)
    dfs = OUT / "dfs_service_requirement.csv"
    if not dfs.exists():
        partial = dfs.with_suffix(".csv.partial")
        partial.write_bytes(get(DFS_URL))
        partial.rename(dfs)
    print("done", len(list(PRICES.glob("*.parquet"))), "days cached")


if __name__ == "__main__":
    main()
