"""Download four GB electricity price and carbon-intensity series for 2025-09-01 to 2026-09-30.

Written for the battery-versus-solar-PV study (PR 1094). Run all four sources, or name some:

    uv run python studies/market_downloads/fetch_gb_prices.py
    uv run python studies/market_downloads/fetch_gb_prices.py --sources carbon_intensity

Each source gets a folder under `data/studies/downloads/market/` holding one tidy parquet, a
`README.md` and a `lineage.json` that the script writes from the values it measured, and a
`_day_cache/` of per-chunk parquet files. Each chunk is written the moment it arrives, so a crash
costs one chunk and a re-run fetches only the chunks that are missing. Requests are keyless, use 4
threads, and back off exponentially on HTTP 429 and 5xx.

- `neso_n2ex_day_ahead`: the hourly N2EX day-ahead auction price, from the National Energy System
  Operator's (NESO) CKAN datastore. One request.
- `elexon_system_prices`: the imbalance settlement prices (Elexon dataset DISEBSP), one request for
  each settlement day.
- `elexon_mid_apx`: the Market Index Data from the `APXMIDP` provider only (the `N2EXMIDP` provider
  publishes zeros), one request for each UTC day. This is the EPEX short-term half-hourly index, not
  the day-ahead auction price.
- `carbon_intensity`: the national half-hourly carbon intensity, 14 days for each request.

Pass `--start` and `--end` (inclusive) and `--output-root` for a small test run.
"""

import argparse
import io
from collections.abc import Callable
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
    WINDOW_END,
    WINDOW_START,
    days_between,
    expected_half_hour_starts,
    expected_hour_starts,
    expected_settlement_starts,
    fetch_missing_chunks,
    get_json,
    get_response,
    period_time_mismatches,
    periods_in_settlement_day,
    read_chunks,
    require_complete,
    summarise_gaps,
    utc_midnight,
    write_lineage,
    write_parquet_atomic,
    write_readme,
)
from studies.sources import MARKET_DOWNLOADS_DIR

SCRIPT_PATH: Final[str] = "studies/market_downloads/fetch_gb_prices.py"
SOURCE_NAMES: Final[tuple[str, ...]] = (
    "elexon_system_prices",
    "elexon_mid_apx",
    "carbon_intensity",
    "neso_n2ex_day_ahead",
)
"""The sources in run order. The N2EX source runs last because its clock check reads the Elexon
APX table."""
CLOCK_LAGS_HOURS: Final[tuple[int, ...]] = (-1, 0, 1)
N2EX_RESOURCE: Final[str] = "4f27eea5-7038-4f73-9740-e3e4ad47c26a"
N2EX_DUMP_URL: Final[str] = f"https://api.neso.energy/datastore/dump/{N2EX_RESOURCE}"
N2EX_PAGE: Final[str] = "https://www.neso.energy/data-portal/gb-n2ex-day-ahead-price"
SYSTEM_PRICES_URL: Final[str] = f"{ELEXON_API}/balancing/settlement/system-prices"
MID_URL: Final[str] = f"{ELEXON_API}/datasets/MID/stream"
MID_PROVIDER: Final[str] = "APXMIDP"
CARBON_API: Final[str] = "https://api.carbonintensity.org.uk/intensity"
CARBON_PAGE: Final[str] = "https://carbonintensity.org.uk/"
CARBON_CHUNK_DAYS: Final[int] = 14
CLOCK_CHANGE_DAYS: Final[tuple[date, ...]] = (date(2025, 10, 26), date(2026, 3, 29))
"""The two days in the window on which UK clocks change."""

UTC_TIME: Final[pl.Datetime] = pl.Datetime(time_unit="us", time_zone="UTC")
ISO_SECONDS: Final[str] = "%Y-%m-%dT%H:%M:%SZ"
ISO_MINUTES: Final[str] = "%Y-%m-%dT%H:%MZ"

N2EX_SCHEMA: Final[dict[str, Any]] = {
    "time": UTC_TIME,
    "delivery_date": pl.Date,
    "price_gbp_per_mwh": pl.Float64,
}
SYSTEM_PRICE_SCHEMA: Final[dict[str, Any]] = {
    "time": UTC_TIME,
    "settlement_date": pl.Date,
    "settlement_period": pl.Int16,
    "system_sell_price_gbp_per_mwh": pl.Float64,
    "system_buy_price_gbp_per_mwh": pl.Float64,
    "net_imbalance_volume_mwh": pl.Float64,
    "total_accepted_offer_volume_mwh": pl.Float64,
    "total_accepted_bid_volume_mwh": pl.Float64,
    "price_derivation_code": pl.String,
}
MID_SCHEMA: Final[dict[str, Any]] = {
    "time": UTC_TIME,
    "settlement_date": pl.Date,
    "settlement_period": pl.Int16,
    "price_gbp_per_mwh": pl.Float64,
    "volume_mwh": pl.Float64,
}
CARBON_SCHEMA: Final[dict[str, Any]] = {
    "time": UTC_TIME,
    "time_end": UTC_TIME,
    "forecast_gco2_per_kwh": pl.Int32,
    "actual_gco2_per_kwh": pl.Int32,
    "index": pl.String,
}


def _utc(*, column: str, fmt: str) -> pl.Expr:
    """Parse a UTC ISO string column with `fmt` into a UTC datetime."""
    return pl.col(column).str.to_datetime(format=fmt, time_zone="UTC", time_unit="us")


def parse_n2ex(*, csv_text: str, start: date, end: date) -> pl.DataFrame:
    """Turn the N2EX datastore dump into one row for each hour in `start..end`.

    The dump has `Date`, `Delivery Period` (`HH:MM - HH:MM`), and `Price`. NESO's clock-change days
    have 24 rows, so the grid is read as UTC: `time` is the date plus the first hour of the delivery
    period, in UTC.

    Args:
        csv_text: The CSV body.
        start: First day to keep.
        end: Last day to keep, inclusive.

    Returns:
        Columns `time` (UTC hour start), `delivery_date`, and `price_gbp_per_mwh`, sorted by time.

    Raises:
        ValueError: If the CSV lacks one of the three columns.
    """
    raw = pl.read_csv(io.StringIO(csv_text), infer_schema_length=0)
    missing = {"Date", "Delivery Period", "Price"} - set(raw.columns)
    if missing:
        raise ValueError(f"N2EX dump lacks columns {sorted(missing)}; it has {raw.columns}")
    return (
        raw.select(
            delivery_date=pl.col("Date").str.to_date(format="%Y-%m-%d"),
            start_hour=pl.col("Delivery Period").str.slice(0, 2).cast(pl.Int32),
            price_gbp_per_mwh=pl.col("Price").cast(pl.Float64),
        )
        .filter(pl.col("delivery_date").is_between(start, end))
        .select(
            time=(
                pl.col("delivery_date").cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
                + pl.duration(hours=pl.col("start_hour"))
            ),
            delivery_date="delivery_date",
            price_gbp_per_mwh="price_gbp_per_mwh",
        )
        .unique(subset="time", keep="last")
        .sort("time")
    )


def apx_lag_correlations(*, n2ex: pl.DataFrame, apx: pl.DataFrame) -> dict[str, float | None]:
    """Correlate hourly N2EX prices with the hourly mean of the half-hourly APX index at each lag.

    A lag of `k` hours pairs the N2EX price for hour `t` with the APX mean for hour `t + k`. If the
    N2EX grid is UTC, the correlation peaks at lag 0. If the grid were UK clock time, it would peak
    at one hour during summer time.

    Args:
        n2ex: The output of `parse_n2ex`.
        apx: The `elexon_mid_apx` table, with `time` and `price_gbp_per_mwh`.

    Returns:
        The Pearson correlation at each lag in `CLOCK_LAGS_HOURS`, keyed by the lag as a string,
        or None where fewer than 2 hours overlap.
    """
    hourly = (
        apx.sort("time")
        .group_by_dynamic("time", every="1h")
        .agg(apx_price=pl.col("price_gbp_per_mwh").mean())
    )
    result: dict[str, float | None] = {}
    for lag in CLOCK_LAGS_HOURS:
        shifted = hourly.with_columns(time=pl.col("time") - timedelta(hours=lag))
        joined = n2ex.join(shifted, on="time", how="inner")
        result[str(lag)] = (
            joined.select(pl.corr("price_gbp_per_mwh", "apx_price")).item()
            if joined.height > 1
            else None
        )
    return result


def check_n2ex_hours(
    *, frame: pl.DataFrame, start: date, end: date, apx: pl.DataFrame | None = None
) -> dict[str, Any]:
    """Check that the N2EX hour mapping is consistent and that its clock is UTC.

    The mapping is consistent when every day has exactly 24 rows (including the two clock-change
    days) and the rows form an unbroken UTC hourly series with no duplicate hour. The clock is UTC
    when the correlation with the hourly APX index peaks at lag 0.

    Args:
        frame: The output of `parse_n2ex`.
        start: First day of the window.
        end: Last day of the window, inclusive.
        apx: The `elexon_mid_apx` table, or None if it is not available.

    Returns:
        The count of days that do not have 24 rows, the rows on each clock-change day, whether the
        hourly series is unbroken, the number of hours priced exactly 0, and, when `apx` is given,
        the correlation at each lag and the lag with the highest one.
    """
    per_day = frame.group_by("delivery_date").len()
    irregular = per_day.filter(pl.col("len") != 24)
    change_rows = {
        day.isoformat(): int(per_day.filter(pl.col("delivery_date") == day)["len"].sum())
        for day in CLOCK_CHANGE_DAYS
        if start <= day <= end
    }
    steps = frame["time"].diff().drop_nulls().unique().to_list()
    result: dict[str, Any] = {
        "days_without_24_rows": irregular.height,
        "rows_on_clock_change_days": change_rows,
        "hourly_series_unbroken": steps in ([], [timedelta(hours=1)]),
        "hours_priced_exactly_zero": int((frame["price_gbp_per_mwh"] == 0).sum()),
    }
    if apx is not None and apx.height:
        correlations = apx_lag_correlations(n2ex=frame, apx=apx)
        valid = {lag: value for lag, value in correlations.items() if value is not None}
        result["apx_correlation_by_lag_hours"] = correlations
        result["best_lag_hours"] = int(max(valid, key=lambda lag: valid[lag])) if valid else None
    return result


def parse_system_prices(*, rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Turn the rows of one DISEBSP response into a table, one row for each settlement period."""
    frame = pl.DataFrame(
        {
            "time": [row["startTime"] for row in rows],
            "settlement_date": [row["settlementDate"] for row in rows],
            "settlement_period": [row["settlementPeriod"] for row in rows],
            "system_sell_price_gbp_per_mwh": [row["systemSellPrice"] for row in rows],
            "system_buy_price_gbp_per_mwh": [row["systemBuyPrice"] for row in rows],
            "net_imbalance_volume_mwh": [row["netImbalanceVolume"] for row in rows],
            "total_accepted_offer_volume_mwh": [row["totalAcceptedOfferVolume"] for row in rows],
            "total_accepted_bid_volume_mwh": [row["totalAcceptedBidVolume"] for row in rows],
            "price_derivation_code": [row["priceDerivationCode"] for row in rows],
        },
        schema={
            "time": pl.String,
            "settlement_date": pl.String,
            "settlement_period": pl.Int16,
            "system_sell_price_gbp_per_mwh": pl.Float64,
            "system_buy_price_gbp_per_mwh": pl.Float64,
            "net_imbalance_volume_mwh": pl.Float64,
            "total_accepted_offer_volume_mwh": pl.Float64,
            "total_accepted_bid_volume_mwh": pl.Float64,
            "price_derivation_code": pl.String,
        },
    )
    return frame.with_columns(
        time=_utc(column="time", fmt=ISO_SECONDS),
        settlement_date=pl.col("settlement_date").str.to_date(format="%Y-%m-%d"),
    ).sort("time")


def parse_mid(*, rows: list[dict[str, Any]], start: datetime, end: datetime) -> pl.DataFrame:
    """Turn Market Index Data rows into a table, keeping only `start <= time < end`.

    A stream window includes both its ends, so the row that starts exactly at `end` is dropped and
    the next day's chunk holds it.
    """
    frame = pl.DataFrame(
        {
            "time": [row["startTime"] for row in rows],
            "settlement_date": [row["settlementDate"] for row in rows],
            "settlement_period": [row["settlementPeriod"] for row in rows],
            "price_gbp_per_mwh": [row["price"] for row in rows],
            "volume_mwh": [row["volume"] for row in rows],
        },
        schema={
            "time": pl.String,
            "settlement_date": pl.String,
            "settlement_period": pl.Int16,
            "price_gbp_per_mwh": pl.Float64,
            "volume_mwh": pl.Float64,
        },
    )
    return (
        frame.with_columns(
            time=_utc(column="time", fmt=ISO_SECONDS),
            settlement_date=pl.col("settlement_date").str.to_date(format="%Y-%m-%d"),
        )
        .filter(pl.col("time") >= start, pl.col("time") < end)
        .unique(subset="time", keep="last")
        .sort("time")
    )


def parse_carbon(*, rows: list[dict[str, Any]], start: datetime, end: datetime) -> pl.DataFrame:
    """Turn Carbon Intensity API rows into a table, keeping only `start <= time < end`."""
    frame = pl.DataFrame(
        {
            "time": [row["from"] for row in rows],
            "time_end": [row["to"] for row in rows],
            "forecast_gco2_per_kwh": [row["intensity"]["forecast"] for row in rows],
            "actual_gco2_per_kwh": [row["intensity"]["actual"] for row in rows],
            "index": [row["intensity"]["index"] for row in rows],
        },
        schema={
            "time": pl.String,
            "time_end": pl.String,
            "forecast_gco2_per_kwh": pl.Int32,
            "actual_gco2_per_kwh": pl.Int32,
            "index": pl.String,
        },
    )
    return (
        frame.with_columns(
            time=_utc(column="time", fmt=ISO_MINUTES),
            time_end=_utc(column="time_end", fmt=ISO_MINUTES),
        )
        .filter(pl.col("time") >= start, pl.col("time") < end)
        .unique(subset="time", keep="last")
        .sort("time")
    )


def carbon_chunks(*, start: date, end: date) -> list[tuple[date, date]]:
    """Split `start..end` (inclusive) into `(first_day, day_after_last)` chunks of 14 days."""
    chunks = []
    first = start
    while first <= end:
        after = min(first + timedelta(days=CARBON_CHUNK_DAYS), end + timedelta(days=1))
        chunks.append((first, after))
        first = after
    return chunks


def _period_expectation(*, start: date, end: date) -> str:
    """Describe the settlement-period count of each settlement date in the window."""
    irregular = [
        f"{periods_in_settlement_day(settlement_date=day)} on {day}"
        for day in days_between(start=start, end=end)
        if periods_in_settlement_day(settlement_date=day) != 48
    ]
    return "48 for each settlement date" + (f", except {', '.join(irregular)}" if irregular else "")


def _chunk_key(*, first: date, after: date) -> str:
    return f"{first.isoformat()}_{after.isoformat()}"


def _row_summary(*, name: str, gaps: dict[str, Any], expectation: str) -> str:
    listed = ", ".join(gaps["missing_first"]) or "none"
    return (
        f"- Rows written: {gaps['distinct_rows']}.\n"
        f"- Rows expected: {gaps['expected_rows']} ({expectation}).\n"
        f"- Expected timestamps with no row: {gaps['missing_count']}. Rows outside the expected "
        f"timestamps: {gaps['unexpected_count']}.\n"
        f"- First missing timestamps (UTC): {listed}.\n"
        f"- `{name}.parquet` is built from `_day_cache/`, so a missing chunk there is a gap here."
    )


def _finish(
    *,
    name: str,
    output_dir: Path,
    frame: pl.DataFrame,
    gaps: dict[str, Any],
    request: str,
    source_address: str,
    window: tuple[date, date],
    chunks: dict[str, list[str]],
    extra: dict[str, Any],
    readme: dict[str, Any],
    expectation: str,
) -> None:
    """Write the combined parquet, `lineage.json`, and `README.md` for one source."""
    write_parquet_atomic(frame=frame, path=output_dir / f"{name}.parquet")
    write_lineage(
        product_dir=output_dir,
        note={
            "source_address": source_address,
            "request": request,
            "window": [day.isoformat() for day in window],
            "script": SCRIPT_PATH,
            "rows": frame.height,
            "gaps": gaps,
            "chunks_not_published": chunks["not_published"],
        }
        | extra,
    )
    write_readme(
        product_dir=output_dir,
        script_path=SCRIPT_PATH,
        cache_hint="_day_cache/",
        row_summary=_row_summary(name=name, gaps=gaps, expectation=expectation),
        **readme,
    )
    print(f"{name}: wrote {frame.height} rows, {gaps['missing_count']} expected rows missing")


def fetch_n2ex(*, root: Path, start: date, end: date) -> None:
    """Download the N2EX day-ahead price dump, keep the window, and write the folder."""
    output_dir = root / "neso_n2ex_day_ahead"
    raw_path = output_dir / "_day_cache" / "datastore_dump.csv"
    if raw_path.exists():
        print(f"neso_n2ex_day_ahead: {raw_path.name} cached, skipping")
    else:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        partial = raw_path.with_suffix(".csv.partial")
        text = get_response(url=N2EX_DUMP_URL).text
        last_date = pl.read_csv(io.StringIO(text), infer_schema_length=0)["Date"].max()
        if last_date is None or str(last_date) < end.isoformat():
            raise RuntimeError(
                f"The N2EX dump ends on {last_date}, before the window's last day {end}"
            )
        partial.write_text(text, encoding="utf-8")
        partial.rename(raw_path)
    frame = parse_n2ex(csv_text=raw_path.read_text(encoding="utf-8"), start=start, end=end)
    gaps = summarise_gaps(expected=expected_hour_starts(start=start, end=end), actual=frame["time"])
    apx_path = root / "elexon_mid_apx" / "elexon_mid_apx.parquet"
    apx = pl.read_parquet(apx_path) if apx_path.exists() else None
    hour_check = check_n2ex_hours(frame=frame, start=start, end=end, apx=apx)
    if "apx_correlation_by_lag_hours" in hour_check:
        correlations = ", ".join(
            f"lag {lag} h: {value:.2f}" if value is not None else f"lag {lag} h: n/a"
            for lag, value in hour_check["apx_correlation_by_lag_hours"].items()
        )
        best = hour_check["best_lag_hours"]
        clock_verdict = (
            f"The correlation between N2EX and the hourly mean of the Elexon APX index over the "
            f"window is highest at lag {best} h ({correlations}). "
            + ("That places the N2EX grid on UTC." if best == 0 else "The grid is NOT UTC.")
        )
    else:
        clock_verdict = (
            "The correlation check against the Elexon APX index did not run, because "
            "`elexon_mid_apx.parquet` was not present. Run `elexon_mid_apx` first."
        )
    series_state = "unbroken" if hour_check["hourly_series_unbroken"] else "BROKEN"
    clock_rows = ", ".join(
        f"{day}: {rows} rows" for day, rows in hour_check["rows_on_clock_change_days"].items()
    )
    _finish(
        name="neso_n2ex_day_ahead",
        output_dir=output_dir,
        frame=frame,
        gaps=gaps,
        request=f"Whole datastore dump of resource {N2EX_RESOURCE}, cut to {start} to {end}",
        source_address=N2EX_DUMP_URL,
        window=(start, end),
        chunks={"not_published": []},
        extra={"hour_mapping_check": hour_check},
        expectation=f"24 for each of {(end - start).days + 1} days",
        readme={
            "title": "NESO N2EX day-ahead price, hourly",
            "source_page": N2EX_PAGE,
            "attribution": None,
            "licence": (
                "NESO Open Data Licence, as stated on the NESO data portal "
                "(https://www.neso.energy/data-portal/neso-open-licence)."
            ),
            "timestamp_convention": (
                "`time` is the start of the delivery hour in UTC: the `Date` column plus the first "
                "hour of the `Delivery Period` column. The source labels hours `HH:MM - HH:MM`. "
                "The source has 24 rows on both clock-change days "
                f"({clock_rows or 'none in this window'}), so the grid is read as UTC rather than "
                "UK clock time. Check done by this script: "
                f"{hour_check['days_without_24_rows']} days have a row count other than 24, and "
                f"the hourly series is {series_state}. "
                f"{clock_verdict}"
            ),
            "columns": {
                "time": "Start of the delivery hour, UTC",
                "delivery_date": "The source's `Date` column",
                "price_gbp_per_mwh": (
                    "N2EX day-ahead auction price, GBP per MWh (the source column carries no unit)"
                ),
            },
            "gotchas": [
                (
                    "The price is hourly. The Elexon APX series is half-hourly and is a different "
                    "product."
                ),
                "The source dump covers 2021 to the present. This table is cut to the window.",
                (
                    f"{hour_check['hours_priced_exactly_zero']} hours in the window are priced "
                    "exactly 0.0. They are real prices, not missing values."
                ),
            ],
        },
    )


def fetch_system_prices_day(key: str) -> pl.DataFrame:
    """Fetch one settlement day of DISEBSP system prices."""
    body = get_json(url=f"{SYSTEM_PRICES_URL}/{key}")
    return require_complete(
        frame=parse_system_prices(rows=body["data"]),
        expected_rows=periods_in_settlement_day(settlement_date=date.fromisoformat(key)),
        key=key,
    )


def fetch_system_prices(*, root: Path, start: date, end: date, threads: int) -> None:
    """Download Elexon system prices one settlement day at a time and write the folder."""
    output_dir = root / "elexon_system_prices"
    cache_dir = output_dir / "_day_cache"
    keys = [day.isoformat() for day in days_between(start=start, end=end)]
    chunks = fetch_missing_chunks(
        cache_dir=cache_dir,
        keys=keys,
        fetch=fetch_system_prices_day,
        label="elexon_system_prices",
        threads=threads,
    )
    frame = read_chunks(cache_dir=cache_dir, keys=keys, schema=SYSTEM_PRICE_SCHEMA)
    gaps = summarise_gaps(
        expected=expected_settlement_starts(start=start, end=end), actual=frame["time"]
    )
    mismatches = period_time_mismatches(frame=frame)
    first_row = frame["time"].min()
    first_text = f"{first_row:%H:%M} UTC on {first_row:%Y-%m-%d}" if first_row else "none"
    _finish(
        name="elexon_system_prices",
        output_dir=output_dir,
        frame=frame,
        gaps=gaps,
        request=f"One call for each settlement date from {start} to {end}",
        source_address=f"{SYSTEM_PRICES_URL}/{{settlement_date}}",
        window=(start, end),
        chunks=chunks,
        extra={"rows_whose_time_differs_from_date_and_period": mismatches},
        expectation=_period_expectation(start=start, end=end),
        readme={
            "title": "Elexon imbalance system prices (DISEBSP), half-hourly",
            "source_page": "https://bmrs.elexon.co.uk/system-prices",
            "attribution": ELEXON_ATTRIBUTION,
            "licence": ELEXON_LICENCE,
            "timestamp_convention": (
                "`time` is the UTC start of the settlement period (Elexon's `startTime`). "
                "`settlement_date` and `settlement_period` are Elexon's own: period 1 starts at "
                "00:00 UK local time, so a settlement date has 48 periods, 46 on the day the "
                "clocks go forward, and 50 on the day they go back. The window is by settlement "
                f"date, so the first row starts at {first_text}. Date and period pairs whose "
                f"`time` differs from the UTC start computed from them: {mismatches}."
            ),
            "columns": {
                "time": "Start of the settlement period, UTC",
                "settlement_date": "Elexon settlement date (UK local day)",
                "settlement_period": "Elexon settlement period within the settlement date, from 1",
                "system_sell_price_gbp_per_mwh": "System sell price (SSP), GBP per MWh",
                "system_buy_price_gbp_per_mwh": "System buy price (SBP), GBP per MWh",
                "net_imbalance_volume_mwh": (
                    "Net imbalance volume (NIV), MWh; positive means the system was short"
                ),
                "total_accepted_offer_volume_mwh": "Total accepted offer volume, MWh",
                "total_accepted_bid_volume_mwh": "Total accepted bid volume, MWh (negative)",
                "price_derivation_code": "Elexon's code for how the price was derived",
            },
            "gotchas": [
                "SSP and SBP are both kept. Check whether they differ before using only one.",
                (
                    "Elexon revises these values in later settlement runs. The table holds what "
                    "the API returned on the retrieval date in `lineage.json`."
                ),
            ],
        },
    )


def fetch_mid_day(key: str) -> pl.DataFrame:
    """Fetch one UTC day of Market Index Data from the `APXMIDP` provider."""
    day = date.fromisoformat(key)
    first = utc_midnight(day=day)
    after = first + timedelta(days=1)
    body = get_json(
        url=MID_URL,
        params=[
            ("from", f"{first:%Y-%m-%dT%H:%MZ}"),
            ("to", f"{after:%Y-%m-%dT%H:%MZ}"),
            ("dataProviders", MID_PROVIDER),
        ],
    )
    rows = body["data"] if isinstance(body, dict) else body
    return require_complete(
        frame=parse_mid(rows=rows, start=first, end=after), expected_rows=48, key=key
    )


def fetch_mid(*, root: Path, start: date, end: date, threads: int) -> None:
    """Download the `APXMIDP` Market Index Data one UTC day at a time and write the folder."""
    output_dir = root / "elexon_mid_apx"
    cache_dir = output_dir / "_day_cache"
    keys = [day.isoformat() for day in days_between(start=start, end=end)]
    chunks = fetch_missing_chunks(
        cache_dir=cache_dir,
        keys=keys,
        fetch=fetch_mid_day,
        label="elexon_mid_apx",
        threads=threads,
    )
    frame = read_chunks(cache_dir=cache_dir, keys=keys, schema=MID_SCHEMA)
    gaps = summarise_gaps(
        expected=expected_half_hour_starts(start=start, end=end), actual=frame["time"]
    )
    mismatches = period_time_mismatches(frame=frame)
    _finish(
        name="elexon_mid_apx",
        output_dir=output_dir,
        frame=frame,
        gaps=gaps,
        request=f"One call for each UTC day from {start} to {end}, dataProviders={MID_PROVIDER}",
        source_address=f"{MID_URL}?dataProviders={MID_PROVIDER}",
        window=(start, end),
        chunks=chunks,
        extra={"rows_whose_time_differs_from_date_and_period": mismatches},
        expectation="48 half-hours for each UTC day",
        readme={
            "title": "Elexon Market Index Data, APXMIDP provider, half-hourly",
            "source_page": "https://bmrs.elexon.co.uk/market-index-data",
            "attribution": ELEXON_ATTRIBUTION,
            "licence": ELEXON_LICENCE,
            "timestamp_convention": (
                "`time` is the UTC start of the settlement period (Elexon's `startTime`). The "
                "window is by UTC day, so every UTC day has 48 rows. `settlement_date` and "
                "`settlement_period` are Elexon's own (period 1 starts at 00:00 UK local time). "
                f"Pairs whose `time` differs from the UTC start computed from them: {mismatches}."
            ),
            "columns": {
                "time": "Start of the half-hour, UTC",
                "settlement_date": "Elexon settlement date (UK local day)",
                "settlement_period": "Elexon settlement period within the settlement date, from 1",
                "price_gbp_per_mwh": "APX (EPEX) market index price, GBP per MWh",
                "volume_mwh": "Volume traded in the index, MWh",
            },
            "gotchas": [
                (
                    "This is the EPEX short-term half-hourly index (provider `APXMIDP`). It is NOT "
                    "the day-ahead auction price. The N2EX day-ahead auction is in "
                    "`neso_n2ex_day_ahead/`."
                ),
                (
                    "The other provider, `N2EXMIDP`, published price 0 and volume 0 in the "
                    "periods checked, so it was not downloaded."
                ),
            ],
        },
    )


def fetch_carbon_chunk(key: str) -> pl.DataFrame:
    """Fetch one chunk of the national Carbon Intensity series. The key is `<first>_<after>`."""
    first_text, after_text = key.split("_")
    first = utc_midnight(day=date.fromisoformat(first_text))
    after = utc_midnight(day=date.fromisoformat(after_text))
    body = get_json(url=f"{CARBON_API}/{first:%Y-%m-%dT%H:%MZ}/{after:%Y-%m-%dT%H:%MZ}")
    return require_complete(
        frame=parse_carbon(rows=body["data"], start=first, end=after),
        expected_rows=48 * (after - first).days,
        key=key,
    )


def fetch_carbon(*, root: Path, start: date, end: date, threads: int) -> None:
    """Download the national Carbon Intensity series in 14-day chunks and write the folder."""
    output_dir = root / "carbon_intensity"
    cache_dir = output_dir / "_day_cache"
    keys = [
        _chunk_key(first=first, after=after) for first, after in carbon_chunks(start=start, end=end)
    ]
    chunks = fetch_missing_chunks(
        cache_dir=cache_dir,
        keys=keys,
        fetch=fetch_carbon_chunk,
        label="carbon_intensity",
        threads=threads,
    )
    frame = read_chunks(cache_dir=cache_dir, keys=keys, schema=CARBON_SCHEMA)
    gaps = summarise_gaps(
        expected=expected_half_hour_starts(start=start, end=end), actual=frame["time"]
    )
    null_actual = int(frame["actual_gco2_per_kwh"].null_count())
    _finish(
        name="carbon_intensity",
        output_dir=output_dir,
        frame=frame,
        gaps=gaps,
        request=f"National intensity in {CARBON_CHUNK_DAYS}-day chunks from {start} to {end}",
        source_address=f"{CARBON_API}/{{from}}/{{to}}",
        window=(start, end),
        chunks=chunks,
        extra={"rows_with_null_actual": null_actual},
        expectation="48 half-hours for each UTC day",
        readme={
            "title": "National Carbon Intensity, half-hourly",
            "source_page": CARBON_PAGE,
            "attribution": "Carbon Intensity API, National Energy System Operator and partners",
            "licence": (
                "CC BY 4.0, from the API provider's documentation. Not independently verified: "
                "the API responses carry no licence field."
            ),
            "timestamp_convention": (
                "`time` is the UTC start of the half-hour (the API's `from`) and `time_end` is its "
                "end (the API's `to`). Every UTC day has 48 rows."
            ),
            "columns": {
                "time": "Start of the half-hour, UTC",
                "time_end": "End of the half-hour, UTC",
                "forecast_gco2_per_kwh": "Forecast carbon intensity, grams CO2 per kWh",
                "actual_gco2_per_kwh": (
                    "Actual (estimated outturn) carbon intensity, grams CO2 per kWh; null where "
                    "the API has no actual yet"
                ),
                "index": "The API's band: very low, low, moderate, high, or very high",
            },
            "gotchas": [
                "National values only. The API's regional values are not downloaded.",
                f"Rows with a null actual value: {null_actual}.",
            ],
        },
    )


def main() -> None:
    """Download the named sources, or all four."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--sources", nargs="+", choices=SOURCE_NAMES, default=list(SOURCE_NAMES))
    parser.add_argument("--start", type=date.fromisoformat, default=WINDOW_START)
    parser.add_argument("--end", type=date.fromisoformat, default=WINDOW_END)
    parser.add_argument("--output-root", type=Path, default=MARKET_DOWNLOADS_DIR)
    parser.add_argument("--threads", type=int, default=FETCH_THREADS)
    args = parser.parse_args()
    runners: dict[str, Callable[[], None]] = {
        "neso_n2ex_day_ahead": partial(
            fetch_n2ex, root=args.output_root, start=args.start, end=args.end
        ),
        "elexon_system_prices": partial(
            fetch_system_prices,
            root=args.output_root,
            start=args.start,
            end=args.end,
            threads=args.threads,
        ),
        "elexon_mid_apx": partial(
            fetch_mid, root=args.output_root, start=args.start, end=args.end, threads=args.threads
        ),
        "carbon_intensity": partial(
            fetch_carbon,
            root=args.output_root,
            start=args.start,
            end=args.end,
            threads=args.threads,
        ),
    }
    for name in sorted(args.sources, key=SOURCE_NAMES.index):
        runners[name]()
    print(f"Finished at {datetime.now(UTC):%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
