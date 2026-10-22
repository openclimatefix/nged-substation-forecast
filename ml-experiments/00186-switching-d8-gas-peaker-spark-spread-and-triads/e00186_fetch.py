"""Experiment 186: fetch the public prices (Elexon system prices, National Gas daily SAP).

Resumable: each day of system prices is its own file written via a temporary name, and a re-run
skips files that exist. The 2024-07-01 to 2026-07-01 prices are read from experiments 185 and 177,
so only 2023-06-01 to 2024-06-30 is fetched here. The gas price is one request (one value per gas
day, pence per kWh; National Gas Data Portal item PUBOB47, "SAP, hourly actual" with one row per
gas day). Under 5 MB in total. Run from the repository root:
`PYTHONUNBUFFERED=1 uv run --no-sync python <this file>` with the folder on the import path.
"""

import json
import time
import urllib.request
from datetime import date, timedelta
from typing import Final

import e00186_common as common
import polars as pl

FIRST: Final[date] = date(2023, 6, 1)
LAST: Final[date] = date(2024, 6, 30)
PRICE_URL: Final[str] = (
    "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{day}?format=json"
)
GAS_URL: Final[str] = (
    "https://data.nationalgas.com/api/find-gas-data-download?applicableFor=Y"
    "&dateFrom=2023-06-01&dateTo=2026-07-01&dateType=GASDAY&latestFlag=Y&ids=PUBOB47&type=CSV"
)
KEEP: Final[list[str]] = [
    "settlementDate", "settlementPeriod", "startTime", "systemSellPrice", "systemBuyPrice",
]  # fmt: skip


def get(url: str) -> bytes:
    """Fetch a URL once, raising on an HTTP error."""
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> None:
    """Fetch the missing days of prices and the gas price."""
    out = common.PUBLIC_DIR / "system_prices"
    out.mkdir(parents=True, exist_ok=True)
    day = FIRST
    while day <= LAST:
        path = out / f"{day}.parquet"
        if not path.exists():
            rows = json.loads(get(PRICE_URL.format(day=day)))["data"]
            frame = pl.DataFrame(rows).select(KEEP)
            tmp = path.with_suffix(".tmp")
            frame.write_parquet(tmp)
            tmp.rename(path)
            time.sleep(0.2)
        day += timedelta(days=1)
    gas = common.PUBLIC_DIR / "gas_sap.csv"
    if not gas.exists():
        gas.write_bytes(get(GAS_URL))
    print(len(list(out.glob("*.parquet"))), "price days;", gas.stat().st_size, "gas bytes")


if __name__ == "__main__":
    main()
