"""Download Octopus Agile half-hourly import prices for the East Midlands (region letter B).

The price is a signal that domestic batteries on the Agile tariff follow, for the unmetered-battery
study. Octopus's public REST API serves a tariff's unit rates at
`/v1/products/<product>/electricity-tariffs/E-1R-<product>-B/standard-unit-rates/` without an
account, a key, or a payment. The script checks that before fetching anything: a refusal (HTTP 401
or 403) stops the script, because it would mean the terms now require something the maintainer must
accept. The script also records the API guide page at fetch time.

Every Agile product version that overlaps the window is fetched, one calendar month at a time. Each
month is written to `_month_cache/` as soon as it arrives, so a crash costs one month and a re-run
skips what is cached. The combined file is built from the months of the current window only.

Timestamps are the start of the half-hour in UTC (`valid_from`). Prices are pence per kWh,
including and excluding VAT.

Run: `uv run python studies/market_downloads/fetch_agile.py`
Writes `data/studies/downloads/market/octopus_agile_east_midlands/` (parquet, lineage.json, README).
"""

import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl
import requests
from studies.sources import MARKET_DOWNLOADS_DIR

API_ROOT: Final[str] = "https://api.octopus.energy/v1"
GUIDE_PAGE: Final[str] = "https://developer.octopus.energy/rest/guides/api-basics/"
REGION_LETTER: Final[str] = "B"
"""Octopus's letter for the East Midlands distribution region."""
WINDOW_START: Final[datetime] = datetime(2025, 8, 31, tzinfo=UTC)
WINDOW_END: Final[datetime] = datetime(2026, 9, 2, tzinfo=UTC)
"""The study year (September 2025 to August 2026) plus a day either side, so that every Agile
delivery day (23:00 to 23:00 local time) that touches the year is whole."""
OUTPUT_DIR: Final[Path] = MARKET_DOWNLOADS_DIR / "octopus_agile_east_midlands"
PAGE_SIZE: Final[int] = 1500
"""Octopus's maximum page size; a 31-day month holds 1,488 half-hours, so a month is one page."""
PAUSE_SECONDS: Final[float] = 0.3
REQUEST_TIMEOUT_SECONDS: Final[int] = 60
PRODUCT_CODE_PREFIX: Final[str] = "AGILE-"
EXCLUDED_PRODUCT_WORDS: Final[tuple[str, ...]] = ("OUTGOING", "FLEX")
"""Agile Outgoing is an export tariff, and Agile Flex is a different product."""
PLAUSIBLE_P_PER_KWH_INC_VAT: Final[tuple[float, float]] = (-30.0, 110.0)
"""Agile's cap is 100 p/kWh including VAT (a product's description says so) and its floor is
negative; the bounds leave room for either."""


def _get_json(*, url: str, params: dict[str, str | int] | None = None) -> dict:
    """Return a JSON response, stopping the script if the API refuses an anonymous request.

    Args:
        url: The endpoint.
        params: Query parameters.

    Returns:
        The decoded body.

    Raises:
        PermissionError: If the API answers 401 or 403.
    """
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code in (401, 403):
        raise PermissionError(
            f"{url} answered {response.status_code}: the terms may now need accepting. "
            "Stop and ask the maintainer."
        )
    response.raise_for_status()
    time.sleep(PAUSE_SECONDS)
    return response.json()


def agile_products() -> list[dict]:
    """Return every Agile import product version whose availability overlaps the window.

    Returns:
        The products' records from `/v1/products/`, oldest first.
    """
    listing = _get_json(
        url=f"{API_ROOT}/products/", params={"brand": "OCTOPUS_ENERGY", "page_size": 200}
    )
    if listing.get("next"):
        raise RuntimeError("The product listing has a second page; the script reads only one.")
    products = []
    for product in listing["results"]:
        code = product["code"]
        if not code.startswith(PRODUCT_CODE_PREFIX) or any(
            word in code for word in EXCLUDED_PRODUCT_WORDS
        ):
            continue
        available_from = datetime.fromisoformat(product["available_from"])
        available_to = (
            datetime.fromisoformat(product["available_to"]) if product["available_to"] else None
        )
        if available_from < WINDOW_END and (available_to is None or available_to > WINDOW_START):
            products.append(product)
    return sorted(products, key=lambda product: product["available_from"])


def month_starts() -> list[datetime]:
    """Return the first instant of every calendar month the window touches (UTC).

    Returns:
        Month starts from the month holding the window's start to the one holding its end.
    """
    starts = []
    cursor = WINDOW_START.replace(day=1)
    while cursor < WINDOW_END:
        starts.append(cursor)
        cursor = (cursor + timedelta(days=32)).replace(day=1)
    return starts


def fetch_month(*, product_code: str, month_start: datetime) -> pl.DataFrame:
    """Fetch one product's unit rates for one calendar month, clipped to the window.

    Args:
        product_code: The Agile product, e.g. `AGILE-24-10-01`.
        month_start: The first instant of the month, UTC.

    Returns:
        Columns `time`, `price_inc_vat_p_per_kwh`, `price_exc_vat_p_per_kwh`, `product_code`.
    """
    month_end = (month_start + timedelta(days=32)).replace(day=1)
    period_from = max(month_start, WINDOW_START)
    period_to = min(month_end, WINDOW_END)
    tariff = f"E-1R-{product_code}-{REGION_LETTER}"
    url = f"{API_ROOT}/products/{product_code}/electricity-tariffs/{tariff}/standard-unit-rates/"
    params: dict[str, str | int] | None = {
        "period_from": period_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "period_to": period_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "page_size": PAGE_SIZE,
    }
    rows = []
    while True:
        body = _get_json(url=url, params=params)
        rows.extend(body["results"])
        if not body.get("next"):
            break
        url, params = body["next"], None
    return pl.DataFrame(
        {
            "time": [row["valid_from"] for row in rows],
            "price_inc_vat_p_per_kwh": [row["value_inc_vat"] for row in rows],
            "price_exc_vat_p_per_kwh": [row["value_exc_vat"] for row in rows],
        },
        schema={
            "time": pl.String,
            "price_inc_vat_p_per_kwh": pl.Float64,
            "price_exc_vat_p_per_kwh": pl.Float64,
        },
    ).with_columns(
        time=pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_unit="us", time_zone="UTC"),
        product_code=pl.lit(product_code),
    )


def fetch_all(*, products: list[dict]) -> list[Path]:
    """Fetch and checkpoint every product-month, skipping those already cached.

    Args:
        products: The products to fetch.

    Returns:
        The cache files for the current window, in product then month order.
    """
    cache_dir = OUTPUT_DIR / "_month_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for product in products:
        for month_start in month_starts():
            path = cache_dir / f"{product['code']}_{month_start:%Y-%m}.parquet"
            paths.append(path)
            if path.exists():
                print(f"{path.name}: cached, skipping")
                continue
            frame = fetch_month(product_code=product["code"], month_start=month_start)
            print(f"{path.name}: {frame.height} rows")
            partial = path.with_suffix(".parquet.partial")
            frame.write_parquet(partial)
            partial.rename(path)
    return paths


def combine(*, paths: list[Path]) -> pl.DataFrame:
    """Join the cached months, keeping one row per half-hour.

    Where two product versions both price a half-hour, the later-starting version wins, because a
    customer is on the newest version.

    Args:
        paths: Cache files.

    Returns:
        The combined frame, sorted by time.
    """
    return (
        pl.concat([pl.read_parquet(path) for path in paths if path.stat().st_size > 0])
        .sort("time", "product_code")
        .unique(subset="time", keep="last", maintain_order=True)
        .sort("time")
    )


def validate(*, frame: pl.DataFrame) -> dict:
    """Run the data-validation checks that apply to a half-hourly price series.

    Args:
        frame: The combined price frame.

    Returns:
        The findings, as a JSON-serialisable dictionary.
    """
    expected = pl.datetime_range(
        WINDOW_START,
        WINDOW_END - timedelta(minutes=30),
        interval="30m",
        time_unit="us",
        time_zone="UTC",
        eager=True,
    ).to_frame("time")
    missing = expected.join(frame.select("time"), on="time", how="anti")
    unexpected = frame.select("time").join(expected, on="time", how="anti")
    price = pl.col("price_inc_vat_p_per_kwh")
    price_column = frame["price_inc_vat_p_per_kwh"]
    low, high = PLAUSIBLE_P_PER_KWH_INC_VAT
    local_hour = pl.col("time").dt.convert_time_zone("Europe/London").dt.hour()
    profile = frame.group_by(hour=local_hour).agg(mean=price.mean()).sort("hour")
    monthly = (
        frame.group_by(month=pl.col("time").dt.strftime("%Y-%m"))
        .agg(mean=price.mean(), std=price.std(), n=pl.len())
        .sort("month")
    )
    runs = (
        frame.sort("time")
        .with_columns(run=(price != price.shift(1)).cum_sum())
        .group_by("run")
        .agg(length=pl.len(), value=price.first())
        .sort("length", descending=True)
    )
    return {
        "expected_rows": expected.height,
        "rows": frame.height,
        "distinct_times": frame["time"].n_unique(),
        "missing_count": missing.height,
        "missing_first": [str(t) for t in missing["time"].head(5).to_list()],
        "unexpected_count": unexpected.height,
        "null_counts": frame.null_count().row(0, named=True),
        "nan_counts": frame.select(pl.col(pl.Float64).is_nan().sum()).row(0, named=True),
        "price_inc_vat_min_max": [price_column.min(), price_column.max()],
        "outside_plausible_range": frame.filter((price < low) | (price > high)).height,
        "negative_price_half_hours": frame.filter(price < 0).height,
        "capped_at_100_half_hours": frame.filter(price >= 100.0).height,
        "longest_unchanging_runs": runs.head(3).to_dicts(),
        "mean_by_uk_clock_hour": dict(
            zip(profile["hour"].to_list(), profile["mean"].round(2).to_list(), strict=True)
        ),
        "monthly": monthly.to_dicts(),
        "vat_ratio_min_max": frame.filter(pl.col("price_exc_vat_p_per_kwh").abs() > 1.0)
        .select(r=pl.col("price_inc_vat_p_per_kwh") / pl.col("price_exc_vat_p_per_kwh"))
        .select(pl.col("r").min().alias("min"), pl.col("r").max().alias("max"))
        .row(0, named=True),
        "product_codes": frame["product_code"].unique().to_list(),
    }


def guide_page_record() -> dict:
    """Record the API guide page at fetch time.

    Returns:
        The page's address, status, retrieval time, and a hash of its body.
    """
    response = requests.get(GUIDE_PAGE, timeout=REQUEST_TIMEOUT_SECONDS)
    return {
        "url": GUIDE_PAGE,
        "http_status": response.status_code,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "body_sha256": hashlib.sha256(response.content).hexdigest(),
    }


def write_documents(*, frame: pl.DataFrame, findings: dict, products: list[dict]) -> None:
    """Write lineage.json and README.md, with every fact taken from the written frame.

    Args:
        frame: The combined frame.
        findings: The validation findings.
        products: The Agile products fetched.
    """
    lineage = {
        "source_address": (
            f"{API_ROOT}/products/<product>/electricity-tariffs/"
            f"E-1R-<product>-{REGION_LETTER}/standard-unit-rates/"
        ),
        "request": (
            f"Unit rates for region {REGION_LETTER}, {WINDOW_START:%Y-%m-%d} to "
            f"{WINDOW_END:%Y-%m-%d}, one month per request"
        ),
        "products": [
            {key: product[key] for key in ("code", "full_name", "available_from", "available_to")}
            for product in products
        ],
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "script": "studies/market_downloads/fetch_agile.py",
        "anonymous_access": (
            "The tariff endpoints answered without credentials; no licence was accepted."
        ),
        "guide_page": guide_page_record(),
        "validation": findings,
    }
    (OUTPUT_DIR / "lineage.json").write_text(json.dumps(lineage, indent=2, default=str))
    schema_lines = "\n".join(f"- `{name}`: {dtype}" for name, dtype in frame.schema.items())
    readme = f"""# Octopus Agile import price, East Midlands, half-hourly

Public data for the unmetered-battery-capacity study. Nothing under `data/` is committed.

- **Source:** <https://developer.octopus.energy/rest/> (tariff endpoints need no account or key).
- **Re-download with:** `studies/market_downloads/fetch_agile.py`. Delete a file in `_month_cache/`
  to fetch that month again.
- **Request details and the validation findings:** `lineage.json` next to this file.

## Columns (schema read from the written file)

{schema_lines}

`time` is the start of the half-hour, UTC. `price_inc_vat_p_per_kwh` and `price_exc_vat_p_per_kwh`
are pence per kWh. `product_code` is the Agile product version that priced the half-hour.

## Missing values

Null counts: {findings["null_counts"]}. NaN counts: {findings["nan_counts"]}. Half-hours with no
row: {findings["missing_count"]} of {findings["expected_rows"]}.

## Gotchas

- An Agile delivery day runs 23:00 to 23:00 UK clock time, and its prices are published in the
  afternoon before. The file is in UTC, so the delivery day's edges move with daylight saving.
- The price including VAT is capped at 100 p/kWh: {findings["capped_at_100_half_hours"]} half-hours
  sit at the cap, and {findings["negative_price_half_hours"]} are negative.
- The tariff is region {REGION_LETTER} (East Midlands) only. Other regions differ in level and
  slightly in shape.
"""
    (OUTPUT_DIR / "README.md").write_text(readme)


def main() -> None:
    """Probe anonymous access, fetch every month, validate, and write the outputs."""
    products = agile_products()
    if not products:
        raise RuntimeError("No Agile product overlaps the window.")
    print("Products:", [product["code"] for product in products])
    frame = combine(paths=fetch_all(products=products))
    findings = validate(frame=frame)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT_DIR / "octopus_agile_east_midlands.parquet.partial"
    frame.write_parquet(partial)
    partial.rename(OUTPUT_DIR / "octopus_agile_east_midlands.parquet")
    write_documents(frame=frame, findings=findings, products=products)
    print(json.dumps({k: v for k, v in findings.items() if k != "monthly"}, indent=1, default=str))


if __name__ == "__main__":
    main()
