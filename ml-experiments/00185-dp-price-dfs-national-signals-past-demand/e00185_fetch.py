"""Experiment 185: fetch the public national data (Elexon system prices, NESO historic demand).

Resumable: each day of system prices and each year of demand is written to its own file through a
temporary name, and a re-run skips what exists. About 30 MB in total. The DFS list and the
2025-07-02 to 2026-07-01 prices are read from experiment 177's folder. Run:
`PYTHONUNBUFFERED=1 uv run --no-sync python <this file>` from the repository root with
`PYTHONPATH` unset (the script imports `e00185_common` by its folder).
"""

import json
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import e00185_common as common
import polars as pl

PRICES: Final[Path] = common.PUBLIC_DIR / "system_prices"
FIRST: Final[date] = date(2024, 7, 1)
LAST: Final[date] = date(2025, 7, 1)
URL: Final[str] = (
    "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{day}?format=json"
)
KEEP: Final[list[str]] = [
    "settlementDate", "settlementPeriod", "startTime", "systemSellPrice", "systemBuyPrice",
    "netImbalanceVolume",
]  # fmt: skip
DEMAND_URLS: Final[dict[int, str]] = {
    2024: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/f6d02c0f-957b-48cb-82ee-09003f2ba759/download/demanddata_2024.csv",
    2025: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/b2bde559-3455-4021-b179-dfe60c0337b0/download/demanddata_2025.csv",
    2026: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/8a4a771c-3929-4e56-93ad-cdf13219dea5/download/demanddataupdate_2026.csv",
}  # fmt: skip


def get(url: str) -> bytes:
    """Fetch a URL once; raise on any HTTP error."""
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


WIND_URL: Final[str] = (
    "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH/stream?settlementDateFrom={first}"
    "&settlementDateTo={last}&fuelType=WIND&format=json"
)
WIND_CHUNK_DAYS: Final[int] = 30


def fetch_wind() -> None:
    """Fetch half-hourly transmission-connected wind output (Elexon FUELHH) in 30-day chunks."""
    out = common.PUBLIC_DIR / "wind_fuelhh"
    out.mkdir(parents=True, exist_ok=True)
    first = FIRST
    while first <= date(2026, 7, 1):
        last = min(first + timedelta(days=WIND_CHUNK_DAYS - 1), date(2026, 7, 1))
        path = out / f"{first}.parquet"
        if not path.exists():
            rows = json.loads(get(WIND_URL.format(first=first, last=last)))
            frame = pl.DataFrame(rows, infer_schema_length=None).select(
                "settlementDate", "settlementPeriod", "startTime", "generation"
            )
            partial = path.with_suffix(".parquet.partial")
            frame.write_parquet(partial)
            partial.rename(path)
            time.sleep(0.5)
        first = last + timedelta(days=1)


def main() -> None:
    """Fetch every missing day of prices and year of demand."""
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
    for year, url in DEMAND_URLS.items():
        path = common.PUBLIC_DIR / f"demanddata_{year}.csv"
        if not path.exists():
            partial = path.with_suffix(".csv.partial")
            partial.write_bytes(get(url))
            partial.rename(path)
    fetch_wind()
    print(
        "done",
        len(list(PRICES.glob("*.parquet"))),
        "price days;",
        sorted(p.name for p in common.PUBLIC_DIR.glob("demanddata_*.csv")),
    )


if __name__ == "__main__":
    main()
