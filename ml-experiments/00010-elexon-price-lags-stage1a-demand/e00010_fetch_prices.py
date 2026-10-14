"""Download Elexon Market Index Data (both providers) for 2019-01-01 to 2026-10-12.

Written for experiment 10. The Insights API serves the `MID` dataset without a key or a payment.
Each UTC day is fetched with both providers (`APXMIDP` and `N2EXMIDP`) and written to
`_day_cache/` as soon as it arrives, so a crash costs one day and a re-run fetches only the missing
days. A day the API returns empty is recorded, not treated as an error, because the archive has
empty days in the middle of the window. The combined parquet holds every row returned.

Run: `uv run python ml-experiments/00010-elexon-price-lags-stage1a-demand/e00010_fetch_prices.py`
Writes `data/ml-experiments/00010-elexon-price-lags-stage1a-demand/prices/`.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import httpx
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

URL: Final[str] = "https://data.elexon.co.uk/bmrs/api/v1/datasets/MID/stream"
START: Final[date] = date(2019, 1, 1)
END: Final[date] = date(2026, 10, 12)
OUTPUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00010-elexon-price-lags-stage1a-demand" / "prices"
)
THREADS: Final[int] = 4
SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    "startTime": pl.String,
    "dataProvider": pl.String,
    "settlementDate": pl.String,
    "settlementPeriod": pl.Int64,
    "price": pl.Float64,
    "volume": pl.Float64,
}


def fetch_day(day: date) -> pl.DataFrame:
    """Return every row Elexon holds for one UTC day, retrying on rate limits and server errors."""
    params = {"from": f"{day}T00:00Z", "to": f"{day + timedelta(days=1)}T00:00Z"}
    for attempt in range(6):
        response = httpx.get(URL, params=params, timeout=120.0)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(5.0 * 2**attempt)
            continue
        response.raise_for_status()
        rows = response.json()
        return pl.DataFrame(
            [{name: row.get(name) for name in SCHEMA} for row in rows], schema=SCHEMA
        )
    raise RuntimeError(f"{day}: still failing after 6 attempts")


def fetch_and_cache(day: date) -> None:
    """Fetch one day unless cached, writing atomically."""
    path = OUTPUT_DIR / "_day_cache" / f"{day}.parquet"
    if path.exists():
        return
    partial = path.with_suffix(".parquet.partial")
    fetch_day(day).write_parquet(partial)
    partial.rename(path)


def main() -> None:
    """Fetch every missing day, then write the combined parquet."""
    (OUTPUT_DIR / "_day_cache").mkdir(parents=True, exist_ok=True)
    days = [START + timedelta(days=i) for i in range((END - START).days + 1)]
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        list(pool.map(fetch_and_cache, days))
    frame = pl.concat(
        [pl.read_parquet(OUTPUT_DIR / "_day_cache" / f"{day}.parquet") for day in days]
    ).with_columns(pl.col("startTime").str.to_datetime(time_zone="UTC").alias("time"))
    frame.drop("startTime").sort("time", "dataProvider").write_parquet(
        OUTPUT_DIR / "market_index.parquet", compression="zstd"
    )
    print(f"wrote {frame.height} rows for {len(days)} days")


if __name__ == "__main__":
    main()
