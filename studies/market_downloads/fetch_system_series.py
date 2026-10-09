r"""Download GB system-wide market series and bid-offer prices for 2025-09-01 to 2026-09-30.

Written for the battery-versus-solar-PV study, and extended for the study of INDDEM and INDGEN.
Run every default source, or name some:

    uv run python studies/market_downloads/fetch_system_series.py
    uv run python studies/market_downloads/fetch_system_series.py --sources elexon_frequency
    uv run python studies/market_downloads/fetch_system_series.py --sources elexon_bod \
        --first-pair-only --bmu-list data/studies/per_study/battery_pv_separation/bmu_list.csv

Each source gets a folder under `data/studies/downloads/market/` holding one tidy parquet (or two,
for frequency); a `README.md` and a `lineage.json`, both written from the values the script
measured; and a `_day_cache/` of per-chunk parquet files. Each chunk is written the moment it
arrives, so a crash loses at most one chunk and a re-run fetches only the missing chunks. Requests
are keyless, use 4 threads, and back off exponentially on HTTP 429 and 5xx. Pass `--start`, `--end`
(inclusive), and `--output-root` for a small test run.

- `elexon_bod`: the bid-offer prices every listed Balancing Mechanism Unit (BMU) submitted (Elexon
  dataset BOD), one request for each calendar month and batch of BMUs. Each BMU submits numbered
  bid-offer pairs; `--first-pair-only` keeps pairs 1 and -1, the offer and bid nearest the physical
  notification.
- `elexon_frequency`: system frequency every 15 seconds, one request for each UTC day. Writes the
  raw values as Float32 and a half-hourly summary (mean, minimum, maximum, standard deviation,
  count).
- `neso_frequency`: the National Energy System Operator's (NESO) one-second frequency CSV, one file
  for each month. Writes the half-hourly summary only and never the raw seconds. Not run by default,
  because the Elexon series covers the same grid frequency.
- `elexon_demand_outturn`: initial national demand outturn (INDO) and initial transmission system
  demand outturn (ITSDO), which one Elexon endpoint serves together.
- `elexon_fuelhh`: half-hourly generation by fuel type.
- `elexon_windfor`: the hourly wind generation forecast, with its publish time.
- `elexon_ndf` and `elexon_tsdf`: the rolling national demand forecast and transmission system
  demand forecast for the national boundary, with their publish times.
- `elexon_netbsad` and `elexon_disbsad`: balancing services adjustment data.
- `elexon_lolpdrm`: loss of load probability and de-rated margin forecasts, with publish times.
- `elexon_syswarn`: system warnings.
- `elexon_inddem` and `elexon_indgen`: the indicated demand (INDDEM) and indicated generation
  (INDGEN) sums of the final Physical Notifications, for the national boundary `N` and the 17
  transmission boundaries B1 to B17, with their publish times. Not run by default, because the two
  series together hold about 40 million rows for a 13-month window. Their publish window starts
  one day before the study window, so that the first study day has an issue published before its
  midnight:

      uv run python studies/market_downloads/fetch_system_series.py \\
          --sources elexon_inddem elexon_indgen --start 2025-08-31
"""

import argparse
import io
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Final, NamedTuple

import polars as pl
from fetch_bmu_dispatch import bmu_batches, bmu_params, list_hash, month_chunks, read_bmu_list
from market_common import (
    ELEXON_API,
    ELEXON_ATTRIBUTION,
    ELEXON_LICENCE,
    FETCH_THREADS,
    WINDOW_END,
    WINDOW_START,
    IncompleteChunkError,
    QueryParams,
    days_between,
    expected_half_hour_starts,
    expected_settlement_starts,
    fetch_missing_chunks,
    get_json,
    get_response,
    period_start_utc,
    read_chunks,
    settlement_day_start_utc,
    summarise_gaps,
    utc_midnight,
    write_lineage,
    write_parquet_atomic,
    write_readme,
)
from studies.sources import MARKET_DOWNLOADS_DIR

SCRIPT_PATH: Final[str] = "studies/market_downloads/fetch_system_series.py"
DEFAULT_BMU_LIST: Final[Path] = Path(
    "/home/jack/dev/nged-substation-forecast/data/studies/per_study/battery_pv_separation/"
    "bmu_list.csv"
)
BOD_URL: Final[str] = f"{ELEXON_API}/datasets/BOD/stream"
FREQUENCY_URL: Final[str] = f"{ELEXON_API}/system/frequency/stream"
NESO_FREQUENCY_PACKAGE_URL: Final[str] = (
    "https://api.neso.energy/api/3/action/package_show?id=system-frequency-data"
)
NESO_LICENCE: Final[str] = (
    "NESO publishes this dataset on its data portal under the NESO Open Data Licence. The licence "
    "name comes from the portal and its text has not been independently verified."
)
RAW_FREQUENCY_MAX_BYTES: Final[int] = 1_000_000_000
"""The raw 15-second values are written only if the cached chunks total less than this."""
BOD_FIRST_PAIRS: Final[tuple[int, ...]] = (-1, 1)
"""Pair identifiers nearest the physical notification: -1 is the first bid, 1 the first offer."""
SAMPLES_PER_HALF_HOUR_15S: Final[int] = 120
SAMPLES_PER_HALF_HOUR_1S: Final[int] = 1800
ELEXON_DEFAULT_SOURCES: Final[tuple[str, ...]] = (
    "elexon_bod",
    "elexon_frequency",
    "elexon_demand_outturn",
    "elexon_fuelhh",
    "elexon_windfor",
    "elexon_ndf",
    "elexon_tsdf",
    "elexon_netbsad",
    "elexon_disbsad",
    "elexon_lolpdrm",
    "elexon_syswarn",
)
ALL_SOURCES: Final[tuple[str, ...]] = (
    *ELEXON_DEFAULT_SOURCES,
    "elexon_inddem",
    "elexon_indgen",
    "neso_frequency",
)

UTC_TIME: Final[pl.Datetime] = pl.Datetime(time_unit="us", time_zone="UTC")
ISO_SECONDS: Final[str] = "%Y-%m-%dT%H:%M:%SZ"
REQUEST_TIME: Final[str] = "%Y-%m-%dT%H:%MZ"


class Col(NamedTuple):
    """One column of a parsed table: the source field, the column name, its type, and a gloss."""

    raw: str
    name: str
    dtype: Any
    doc: str


def frame_from_rows(*, rows: Any, columns: Sequence[Col], what: str) -> pl.DataFrame:
    """Turn a list of JSON objects into a typed table, parsing timestamps as UTC.

    Args:
        rows: The decoded JSON body, which must be a list of objects.
        columns: The fields to keep, in output order.
        what: Names the request in the error message.

    Returns:
        One row per object, with the columns in `columns`.

    Raises:
        TypeError: If the body is not a list, such as a validation error object.
    """
    if not isinstance(rows, list):
        raise TypeError(f"{what}: expected a JSON list, got {str(rows)[:200]!r}")
    raw_types = {
        col.name: pl.String if col.dtype in (UTC_TIME, pl.Date) else col.dtype for col in columns
    }
    frame = pl.DataFrame(
        {col.name: [row.get(col.raw) for row in rows] for col in columns}, schema=raw_types
    )
    for col in columns:
        if col.dtype == UTC_TIME:
            frame = frame.with_columns(
                pl.col(col.name).str.to_datetime(
                    format=ISO_SECONDS, time_zone="UTC", time_unit="us"
                )
            )
        elif col.dtype == pl.Date:
            frame = frame.with_columns(pl.col(col.name).str.to_date(format="%Y-%m-%d"))
    return frame


def schema_of(*, columns: Sequence[Col]) -> dict[str, Any]:
    """Return the `{name: dtype}` schema of a column list."""
    return {col.name: col.dtype for col in columns}


def add_period_time(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `time`, the UTC start of each row's settlement period, computed from date and period."""
    pairs = frame.select("settlement_date", "settlement_period").unique()
    starts = pairs.with_columns(
        time=pl.Series(
            [
                period_start_utc(settlement_date=settlement_date, settlement_period=period)
                for settlement_date, period in pairs.iter_rows()
            ],
            dtype=UTC_TIME,
        )
    )
    return frame.join(starts, on=["settlement_date", "settlement_period"], how="left").select(
        "time", pl.exclude("time")
    )


def chunk_bounds(*, key: str) -> tuple[date, date]:
    """Split a `<first>_<day after last>` chunk key into its two dates."""
    first_text, after_text = key.split("_")
    return date.fromisoformat(first_text), date.fromisoformat(after_text)


def day_chunks(*, start: date, end: date) -> list[str]:
    """Return one `<day>_<next day>` key for each day in `start..end`, both included."""
    return [f"{day}_{day + timedelta(days=1)}" for day in days_between(start=start, end=end)]


# The generic Elexon series: Balancing Mechanism Reporting Service (BMRS) datasets with no BMU
# filter


def publish_params(*, first: date, after: date, extra: QueryParams) -> QueryParams:
    """Query a dataset by publish time, for the UTC days `first` up to but excluding `after`."""
    return [
        ("publishDateTimeFrom", f"{utc_midnight(day=first):{REQUEST_TIME}}"),
        ("publishDateTimeTo", f"{utc_midnight(day=after):{REQUEST_TIME}}"),
        *extra,
    ]


def settlement_date_params(*, first: date, after: date, extra: QueryParams) -> QueryParams:
    """Query a dataset by settlement date, for the settlement dates `first` up to `after`."""
    return [
        ("settlementDateFrom", first.isoformat()),
        ("settlementDateTo", (after - timedelta(days=1)).isoformat()),
        *extra,
    ]


def period_time_params(*, first: date, after: date, extra: QueryParams) -> QueryParams:
    """Query a dataset by settlement-period start time, which is inclusive at both ends."""
    start = settlement_day_start_utc(settlement_date=first)
    stop = settlement_day_start_utc(settlement_date=after)
    return [("from", f"{start:{REQUEST_TIME}}"), ("to", f"{stop:{REQUEST_TIME}}"), *extra]


@dataclass(frozen=True)
class SeriesSpec:
    """The endpoint, query parameters, chunking, columns, and README text of one Elexon series."""

    name: str
    title: str
    page: str
    url: str
    columns: tuple[Col, ...]
    params: Callable[..., QueryParams]
    chunks: Callable[..., list[str]]
    window_column: str
    """The column the chunk window is applied to: `publish_time` or `settlement_date`."""
    sort: tuple[str, ...]
    timestamp_convention: str
    gotchas: tuple[str, ...]
    extra_params: tuple[tuple[str, str], ...] = ()
    add_time: bool = False
    """Whether the rows carry no timestamp, so `time` is computed from date and period."""
    sparse: bool = False
    """Whether a chunk may legitimately be empty (a warning feed)."""
    expect_every_period: bool = False
    """Whether every settlement period of the window should have at least one row."""
    expect_daily_publish: bool = False
    """Whether every UTC day of the window should have at least one publication."""
    publications_per_day: int | None = None
    """How many distinct publish times a full UTC day holds, if the schedule is regular."""
    purpose: str | None = None
    """The sentence for the README that says which study the download was made for."""


_SETTLEMENT_KEYS: Final[tuple[Col, ...]] = (
    Col("settlementDate", "settlement_date", pl.Date, "Settlement date (UK clock-time day)."),
    Col(
        "settlementPeriod",
        "settlement_period",
        pl.Int16,
        "Settlement period within the settlement date, 1 to 48 (46 or 50 on clock-change days).",
    ),
)

INDGEN_INDDEM_PURPOSE: Final[str] = (
    "Public data for the study of Elexon's indicated generation and demand "
    "(<https://github.com/openclimatefix/nged-substation-forecast/issues/1108>)."
)

SERIES: Final[dict[str, SeriesSpec]] = {
    "elexon_demand_outturn": SeriesSpec(
        name="elexon_demand_outturn",
        title="Elexon initial demand outturn (INDO and ITSDO)",
        page="https://bmrs.elexon.co.uk/demand-outturn",
        url=f"{ELEXON_API}/demand/outturn/stream",
        columns=(
            Col("startTime", "time", UTC_TIME, "UTC start of the half-hour the demand covers."),
            Col("publishTime", "publish_time", UTC_TIME, "UTC time the figure was published."),
            *_SETTLEMENT_KEYS,
            Col(
                "initialDemandOutturn",
                "indo_mw",
                pl.Float64,
                (
                    "Initial national demand outturn (INDO), MW. Excludes station load, "
                    "pumping, and interconnector exports."
                ),
            ),
            Col(
                "initialTransmissionSystemDemandOutturn",
                "itsdo_mw",
                pl.Float64,
                (
                    "Initial transmission system demand outturn (ITSDO), MW. Includes "
                    "station load, pumping, and interconnector exports."
                ),
            ),
        ),
        params=settlement_date_params,
        chunks=month_chunks,
        window_column="settlement_date",
        sort=("time",),
        timestamp_convention=(
            "`time` is the UTC start of the half-hour (settlement period) the demand covers. "
            "`publish_time` is when Elexon published that figure, about 30 minutes later. "
            "Both are UTC."
        ),
        gotchas=(
            (
                "INDO and ITSDO come from one endpoint, so they share this folder "
                "instead of having "
                "one each."
            ),
            "These are initial outturn figures, which later settlement runs do not revise.",
        ),
        expect_every_period=True,
    ),
    "elexon_fuelhh": SeriesSpec(
        name="elexon_fuelhh",
        title="Elexon half-hourly generation outturn by fuel type (FUELHH)",
        page="https://bmrs.elexon.co.uk/generation-by-fuel-type",
        url=f"{ELEXON_API}/datasets/FUELHH/stream",
        columns=(
            Col("startTime", "time", UTC_TIME, "UTC start of the half-hour the output covers."),
            Col("publishTime", "publish_time", UTC_TIME, "UTC time the figure was published."),
            *_SETTLEMENT_KEYS,
            Col("fuelType", "fuel_type", pl.String, "Fuel type, such as CCGT, WIND, or INTFR."),
            Col(
                "generation", "generation_mw", pl.Float64, "Average output over the half-hour, MW."
            ),
        ),
        params=settlement_date_params,
        chunks=month_chunks,
        window_column="settlement_date",
        sort=("time", "fuel_type"),
        timestamp_convention=(
            "`time` is the UTC start of the half-hour (settlement period) the output covers. "
            "`publish_time` is when Elexon published the figure."
        ),
        gotchas=(
            (
                "Interconnector fuel types (names starting `INT`) are net imports and so can be "
                "negative or positive depending on the flow."
            ),
            "Pumped storage (`PS`) is negative while it pumps.",
        ),
        expect_every_period=True,
    ),
    "elexon_windfor": SeriesSpec(
        name="elexon_windfor",
        title="Elexon wind generation forecast (WINDFOR)",
        page="https://bmrs.elexon.co.uk/wind-generation-forecast",
        url=f"{ELEXON_API}/datasets/WINDFOR/stream",
        columns=(
            Col("publishTime", "publish_time", UTC_TIME, "UTC time the forecast was published."),
            Col("startTime", "time", UTC_TIME, "UTC start of the hour the forecast covers."),
            Col("generation", "forecast_mw", pl.Float64, "Forecast wind output, MW."),
        ),
        params=publish_params,
        chunks=day_chunks,
        window_column="publish_time",
        sort=("publish_time", "time"),
        timestamp_convention=(
            "`publish_time` is when the forecast was published (UTC). `time` is the UTC start of "
            "the hour the forecast covers, so `time - publish_time` is the lead time. Every "
            "publication is kept, so a consumer can reconstruct what was known at a given time."
        ),
        gotchas=(
            (
                "The window is applied to `publish_time`, so the first and last `time` values "
                "reach "
                "outside the study window."
            ),
            "The forecast covers transmission-connected and large embedded wind only.",
        ),
        expect_daily_publish=True,
    ),
    "elexon_ndf": SeriesSpec(
        name="elexon_ndf",
        title="Elexon national demand forecast, national boundary (NDF)",
        page="https://bmrs.elexon.co.uk/day-ahead-demand-forecast",
        url=f"{ELEXON_API}/datasets/NDF/stream",
        columns=(
            Col("publishTime", "publish_time", UTC_TIME, "UTC time the forecast was published."),
            Col("startTime", "time", UTC_TIME, "UTC start of the half-hour the forecast covers."),
            *_SETTLEMENT_KEYS,
            Col("boundary", "boundary", pl.String, "Boundary; `N` is the national boundary."),
            Col("demand", "demand_mw", pl.Float64, "Forecast national demand, MW."),
        ),
        params=publish_params,
        chunks=day_chunks,
        window_column="publish_time",
        sort=("publish_time", "time"),
        timestamp_convention=(
            "`publish_time` is when the forecast was published (UTC). `time` is the UTC start of "
            "the half-hour it covers. Every publication is kept, so a consumer can reconstruct "
            "what was known at a given time."
        ),
        gotchas=(
            "The window is applied to `publish_time`, so `time` values reach past the window end.",
            (
                "NESO publishes a new forecast about every half-hour, so each `time` appears many "
                "times with different `publish_time`."
            ),
        ),
        expect_daily_publish=True,
    ),
    "elexon_tsdf": SeriesSpec(
        name="elexon_tsdf",
        title="Elexon transmission system demand forecast, national boundary (TSDF)",
        page="https://bmrs.elexon.co.uk/day-ahead-demand-forecast",
        url=f"{ELEXON_API}/datasets/TSDF/stream",
        columns=(
            Col("publishTime", "publish_time", UTC_TIME, "UTC time the forecast was published."),
            Col("startTime", "time", UTC_TIME, "UTC start of the half-hour the forecast covers."),
            *_SETTLEMENT_KEYS,
            Col("boundary", "boundary", pl.String, "Boundary; `N` is the national boundary."),
            Col("demand", "demand_mw", pl.Float64, "Forecast transmission system demand, MW."),
        ),
        params=publish_params,
        chunks=day_chunks,
        window_column="publish_time",
        sort=("publish_time", "time"),
        timestamp_convention=(
            "`publish_time` is when the forecast was published (UTC). `time` is the UTC start of "
            "the half-hour it covers. Every publication is kept."
        ),
        gotchas=(
            (
                "Only the national boundary `N` is requested. The dataset also has about 17 "
                "transmission boundaries, which would multiply the rows by that factor."
            ),
            "The window is applied to `publish_time`, so `time` values reach past the window end.",
        ),
        extra_params=(("boundary", "N"),),
        expect_daily_publish=True,
    ),
    "elexon_inddem": SeriesSpec(
        name="elexon_inddem",
        title="Elexon indicated demand (INDDEM), national and 17 boundaries",
        page="https://bmrs.elexon.co.uk/indicated-generation-and-demand",
        url=f"{ELEXON_API}/datasets/INDDEM/stream",
        columns=(
            Col("publishTime", "publish_time", UTC_TIME, "UTC time the issue was published."),
            Col("startTime", "time", UTC_TIME, "UTC start of the half-hour the value covers."),
            *_SETTLEMENT_KEYS,
            Col(
                "boundary",
                "boundary",
                pl.String,
                "`N` is the national total; B1 to B17 are the 17 overlapping boundaries.",
            ),
            Col(
                "demand",
                "demand_mw",
                pl.Float64,
                "Sum of the final Physical Notifications of the importing BMUs, MW. Negative.",
            ),
        ),
        params=publish_params,
        chunks=day_chunks,
        window_column="publish_time",
        sort=("publish_time", "boundary", "time"),
        timestamp_convention=(
            "`publish_time` is when the issue was published (UTC). `time` is the UTC start of "
            "the half-hour the value covers. Every issue is kept, and an issue holds about 44 to "
            "82 half-hours depending on when it is published."
        ),
        gotchas=(
            "Values are negative for import, so a larger demand is a more negative number.",
            (
                "Boundaries B1 to B17 are nested sums of 17 study zones (Elexon CVA Change "
                "Circular 235, Appendix 1), so they overlap and do not add up to `N`."
            ),
            "The window is applied to `publish_time`, so `time` values reach past the window end.",
        ),
        expect_daily_publish=True,
        publications_per_day=48,
        purpose=INDGEN_INDDEM_PURPOSE,
    ),
    "elexon_indgen": SeriesSpec(
        name="elexon_indgen",
        title="Elexon indicated generation (INDGEN), national and 17 boundaries",
        page="https://bmrs.elexon.co.uk/indicated-generation-and-demand",
        url=f"{ELEXON_API}/datasets/INDGEN/stream",
        columns=(
            Col("publishTime", "publish_time", UTC_TIME, "UTC time the issue was published."),
            Col("startTime", "time", UTC_TIME, "UTC start of the half-hour the value covers."),
            *_SETTLEMENT_KEYS,
            Col(
                "boundary",
                "boundary",
                pl.String,
                "`N` is the national total; B1 to B17 are the 17 overlapping boundaries.",
            ),
            Col(
                "generation",
                "generation_mw",
                pl.Float64,
                "Sum of the final Physical Notifications of the exporting BMUs, MW.",
            ),
        ),
        params=publish_params,
        chunks=day_chunks,
        window_column="publish_time",
        sort=("publish_time", "boundary", "time"),
        timestamp_convention=(
            "`publish_time` is when the issue was published (UTC). `time` is the UTC start of "
            "the half-hour the value covers. Every issue is kept, and an issue holds about 44 to "
            "82 half-hours depending on when it is published."
        ),
        gotchas=(
            (
                "Boundaries B1 to B17 are nested sums of 17 study zones (Elexon CVA Change "
                "Circular 235, Appendix 1), so they overlap and do not add up to `N`."
            ),
            "The window is applied to `publish_time`, so `time` values reach past the window end.",
        ),
        expect_daily_publish=True,
        publications_per_day=48,
        purpose=INDGEN_INDDEM_PURPOSE,
    ),
    "elexon_netbsad": SeriesSpec(
        name="elexon_netbsad",
        title="Elexon net balancing services adjustment data (NETBSAD)",
        page="https://bmrs.elexon.co.uk/net-balancing-services-adjustment-data",
        url=f"{ELEXON_API}/datasets/NETBSAD/stream",
        columns=(
            *_SETTLEMENT_KEYS,
            *(
                Col(raw, name, pl.Float64, doc)
                for raw, name, doc in (
                    (
                        "netBuyPriceCostAdjustmentEnergy",
                        "net_buy_price_cost_adjustment_energy_gbp",
                        "Net cost adjustment to the buy price, energy actions, GBP.",
                    ),
                    (
                        "netBuyPriceVolumeAdjustmentEnergy",
                        "net_buy_price_volume_adjustment_energy_mwh",
                        "Net volume adjustment to the buy price, energy actions, MWh.",
                    ),
                    (
                        "netBuyPriceVolumeAdjustmentSystem",
                        "net_buy_price_volume_adjustment_system_mwh",
                        "Net volume adjustment to the buy price, system actions, MWh.",
                    ),
                    (
                        "buyPricePriceAdjustment",
                        "buy_price_price_adjustment_gbp_per_mwh",
                        "Price adjustment to the buy price, GBP/MWh.",
                    ),
                    (
                        "netSellPriceCostAdjustmentEnergy",
                        "net_sell_price_cost_adjustment_energy_gbp",
                        "Net cost adjustment to the sell price, energy actions, GBP.",
                    ),
                    (
                        "netSellPriceVolumeAdjustmentEnergy",
                        "net_sell_price_volume_adjustment_energy_mwh",
                        "Net volume adjustment to the sell price, energy actions, MWh.",
                    ),
                    (
                        "netSellPriceVolumeAdjustmentSystem",
                        "net_sell_price_volume_adjustment_system_mwh",
                        "Net volume adjustment to the sell price, system actions, MWh.",
                    ),
                    (
                        "sellPricePriceAdjustment",
                        "sell_price_price_adjustment_gbp_per_mwh",
                        "Price adjustment to the sell price, GBP/MWh.",
                    ),
                )
            ),
        ),
        params=period_time_params,
        chunks=month_chunks,
        window_column="settlement_date",
        sort=("time",),
        timestamp_convention=(
            "The source gives only the settlement date and period. `time` is computed by this "
            "script as the UTC start of that settlement period (period 1 starts at 00:00 UK local "
            "time)."
        ),
        gotchas=(
            (
                "These adjustments feed the system buy and sell prices. The field glosses are "
                "paraphrased from the field names and have not been checked against Elexon's "
                "definitions."
            ),
        ),
        add_time=True,
        expect_every_period=True,
    ),
    "elexon_disbsad": SeriesSpec(
        name="elexon_disbsad",
        title="Elexon disaggregated balancing services adjustment data (DISBSAD)",
        page="https://bmrs.elexon.co.uk/disaggregated-balancing-services-adjustment-data",
        url=f"{ELEXON_API}/datasets/DISBSAD/stream",
        columns=(
            *_SETTLEMENT_KEYS,
            Col("id", "adjustment_id", pl.Int64, "Adjustment identifier within the period."),
            Col("cost", "cost_gbp", pl.Float64, "Cost of the action, GBP (negative is a credit)."),
            Col("volume", "volume_mwh", pl.Float64, "Volume of the action, MWh."),
            Col("soFlag", "so_flag", pl.Boolean, "System-operator flag: a system action."),
            Col("storFlag", "stor_flag", pl.Boolean, "Short-term operating reserve flag."),
            Col("partyId", "party_id", pl.String, "Party providing the service."),
            Col("assetId", "asset_id", pl.String, "Asset providing the service."),
            Col("isTendered", "is_tendered", pl.Boolean, "Whether the service was tendered."),
            Col("service", "service", pl.String, "Service type, such as Energy or System."),
        ),
        params=period_time_params,
        chunks=month_chunks,
        window_column="settlement_date",
        sort=("time", "adjustment_id"),
        timestamp_convention=(
            "The source gives only the settlement date and period. `time` is computed by this "
            "script as the UTC start of that settlement period (period 1 starts at 00:00 UK local "
            "time)."
        ),
        gotchas=(
            "A period can have no rows, one row, or many, so a missing period is not a gap.",
            "`party_id` holds a party name in the rows sampled, not a short code.",
            (
                "A period with no actions can still carry one placeholder row whose party, "
                "asset, tender, and service fields are null."
            ),
        ),
        add_time=True,
    ),
    "elexon_lolpdrm": SeriesSpec(
        name="elexon_lolpdrm",
        title="Elexon loss of load probability and de-rated margin forecasts (LOLPDRM)",
        page="https://bmrs.elexon.co.uk/loss-of-load-probability-and-de-rated-margin",
        url=f"{ELEXON_API}/datasets/LOLPDRM/stream",
        columns=(
            Col("dataset", "dataset", pl.String, "Dataset code in the row (`LOLPDM`)."),
            Col("publishTime", "publish_time", UTC_TIME, "UTC time the forecast was published."),
            Col(
                "publishingPeriodCommencingTime",
                "publishing_period_start",
                UTC_TIME,
                "UTC start of the half-hour in which the forecast was published.",
            ),
            Col("startTime", "time", UTC_TIME, "UTC start of the half-hour the forecast covers."),
            *_SETTLEMENT_KEYS,
            Col(
                "lossOfLoadProbability",
                "loss_of_load_probability",
                pl.Float64,
                "Probability of lost load in the period, between 0 and 1.",
            ),
            Col(
                "deratedMargin",
                "derated_margin_mw",
                pl.Float64,
                "De-rated margin, MW.",
            ),
        ),
        params=publish_params,
        chunks=day_chunks,
        window_column="publish_time",
        sort=("publish_time", "time"),
        timestamp_convention=(
            "`publish_time` is when the forecast was published (UTC). `time` is the UTC start of "
            "the half-hour it covers. Every publication is kept, so a consumer can reconstruct "
            "what was known at a given time."
        ),
        gotchas=(
            "The window is applied to `publish_time`, so `time` values reach past the window end.",
        ),
        expect_daily_publish=True,
    ),
    "elexon_syswarn": SeriesSpec(
        name="elexon_syswarn",
        title="Elexon system warnings (SYSWARN)",
        page="https://bmrs.elexon.co.uk/system-warnings",
        url=f"{ELEXON_API}/datasets/SYSWARN/stream",
        columns=(
            Col("publishTime", "publish_time", UTC_TIME, "UTC time the warning was published."),
            Col("warningType", "warning_type", pl.String, "Warning category."),
            Col("warningText", "warning_text", pl.String, "Free text of the warning."),
        ),
        params=publish_params,
        chunks=month_chunks,
        window_column="publish_time",
        sort=("publish_time", "warning_type"),
        timestamp_convention="`publish_time` is when the warning was published (UTC).",
        gotchas=(
            "Warnings are irregular events, so an empty day is normal.",
            "`warning_text` keeps the source's line breaks, including literal backslash-n pairs.",
        ),
        sparse=True,
    ),
}
"""The Elexon series that need no BMU list, keyed by source name."""


def fetch_series_chunk(key: str, *, spec: SeriesSpec) -> pl.DataFrame:
    """Fetch one chunk of a generic Elexon series and keep only rows inside the chunk window."""
    first, after = chunk_bounds(key=key)
    rows = get_json(
        url=spec.url, params=spec.params(first=first, after=after, extra=list(spec.extra_params))
    )
    frame = frame_from_rows(rows=rows, columns=spec.columns, what=f"{spec.name} {key}")
    if spec.add_time and frame.height:
        frame = add_period_time(frame=frame)
    elif spec.add_time:
        frame = frame.with_columns(time=pl.lit(None, dtype=UTC_TIME)).select(
            "time", pl.exclude("time")
        )
    if spec.window_column == "publish_time":
        inside = pl.col("publish_time").is_between(
            utc_midnight(day=first), utc_midnight(day=after), closed="left"
        )
    else:
        inside = pl.col("settlement_date").is_between(first, after, closed="left")
    frame = frame.filter(inside).unique(maintain_order=True)
    if frame.is_empty() and not spec.sparse:
        raise IncompleteChunkError(f"{spec.name} chunk {key} is empty, so is not published yet")
    return frame


def empty_chunks(*, cache_dir: Path, keys: Sequence[str]) -> list[str]:
    """Return the cached chunks among `keys` that hold zero rows."""
    return [
        key
        for key in keys
        if (cache_dir / f"{key}.parquet").exists()
        and pl.scan_parquet(cache_dir / f"{key}.parquet").select(pl.len()).collect().item() == 0
    ]


def series_schema(*, spec: SeriesSpec) -> dict[str, Any]:
    """Return the schema of a series' written table."""
    schema = schema_of(columns=spec.columns)
    return {"time": UTC_TIME, **schema} if spec.add_time else schema


def days_with_missing_issues(*, frame: pl.DataFrame, expected_per_day: int) -> dict[str, Any]:
    """Count the UTC days whose number of distinct publish times is below `expected_per_day`."""
    per_day = (
        frame.select("publish_time")
        .unique()
        .group_by(day=pl.col("publish_time").dt.date())
        .agg(issues=pl.len())
        .filter(pl.col("issues") < expected_per_day)
        .sort("day")
    )
    return {
        "expected_per_day": expected_per_day,
        "count": per_day.height,
        "first": [
            {"day": day.isoformat(), "issues": issues}
            for day, issues in per_day.head(50).iter_rows()
        ],
    }


def run_series(
    *, spec: SeriesSpec, root: Path, start: date, end: date, threads: int
) -> pl.DataFrame:
    """Download one generic Elexon series, then write its parquet, lineage, and README."""
    output_dir = root / spec.name
    keys = spec.chunks(start=start, end=end)
    cache_dir = output_dir / "_day_cache"
    outcome = fetch_missing_chunks(
        cache_dir=cache_dir,
        keys=keys,
        fetch=partial(fetch_series_chunk, spec=spec),
        label=spec.name,
        threads=threads,
    )
    frame = read_chunks(cache_dir=cache_dir, keys=keys, schema=series_schema(spec=spec))
    frame = frame.unique(maintain_order=True).sort(*spec.sort)
    write_parquet_atomic(frame=frame, path=output_dir / f"{spec.name}.parquet")
    checks: dict[str, Any] = {}
    if spec.expect_every_period:
        checks["period_coverage"] = summarise_gaps(
            expected=expected_settlement_starts(start=start, end=end), actual=frame["time"]
        )
    if spec.expect_daily_publish:
        published_days = set(frame["publish_time"].dt.date().to_list())
        missing_days = [
            day for day in days_between(start=start, end=end) if day not in published_days
        ]
        checks["days_without_publication"] = {
            "count": len(missing_days),
            "first": [day.isoformat() for day in missing_days[:50]],
        }
    if spec.publications_per_day is not None:
        checks["days_with_missing_issues"] = days_with_missing_issues(
            frame=frame, expected_per_day=spec.publications_per_day
        )
    summary = _summary_text(rows=frame.height, checks=checks)
    chunk_kind = "daily" if spec.chunks is day_chunks else "monthly"
    request = f"{spec.url} for {start} to {end}, {len(keys)} {chunk_kind} chunks"
    write_lineage(
        product_dir=output_dir,
        note={
            "source_address": spec.url,
            "request": request,
            "window": [start.isoformat(), end.isoformat()],
            "script": SCRIPT_PATH,
            "rows": frame.height,
            "chunks_not_published": sorted(outcome["not_published"]),
            "chunks_with_zero_rows": empty_chunks(cache_dir=cache_dir, keys=keys),
            "checks": checks,
        },
    )
    write_readme(
        product_dir=output_dir,
        title=spec.title,
        source_page=spec.page,
        script_path=SCRIPT_PATH,
        attribution=ELEXON_ATTRIBUTION,
        cache_hint="_day_cache/",
        licence=ELEXON_LICENCE,
        timestamp_convention=spec.timestamp_convention,
        columns={col.name: col.doc for col in _columns_with_time(spec=spec)},
        row_summary=summary,
        gotchas=list(spec.gotchas),
        **({} if spec.purpose is None else {"purpose": spec.purpose}),
    )
    print(f"{spec.name}: wrote {frame.height} rows")
    return frame


def _columns_with_time(*, spec: SeriesSpec) -> list[Col]:
    columns = list(spec.columns)
    if spec.add_time:
        columns.insert(0, Col("", "time", UTC_TIME, "UTC start of the settlement period."))
    return columns


def _summary_text(*, rows: int, checks: dict[str, Any]) -> str:
    lines = [f"- {rows} rows."]
    coverage = checks.get("period_coverage")
    if coverage:
        lines.append(
            f"- Settlement periods: {coverage['distinct_rows']} of {coverage['expected_rows']} "
            f"expected have at least one row, {coverage['missing_count']} are missing "
            f"(first missing: {coverage['missing_first'][:5] or 'none'})."
        )
    publications = checks.get("days_without_publication")
    if publications:
        lines.append(f"- UTC days with no publication: {publications['count']}.")
    short_days = checks.get("days_with_missing_issues")
    if short_days:
        lines.append(
            f"- UTC days with fewer than {short_days['expected_per_day']} issues: "
            f"{short_days['count']}."
        )
    return "\n".join(lines)


# BOD: bid-offer prices for the listed BMUs

PLACEHOLDER_PRICE_GBP_PER_MWH: Final[float] = 9999.0
"""A bid or offer price at or beyond this magnitude is treated as a placeholder, not a price."""

BOD_COLUMNS: Final[tuple[Col, ...]] = (
    Col("bmUnit", "bmu_id", pl.String, "Elexon BMU identifier."),
    Col("timeFrom", "time", UTC_TIME, "UTC start of the settlement period the prices apply to."),
    Col("timeTo", "time_to", UTC_TIME, "UTC end of the settlement period (start + 30 minutes)."),
    *_SETTLEMENT_KEYS,
    Col(
        "pairId",
        "pair_id",
        pl.Int8,
        (
            "Bid-offer pair. Positive pairs are offers (levels at or above the physical "
            "notification), negative pairs are bids. Pairs 1 and -1 are nearest the notification."
        ),
    ),
    Col("levelFrom", "level_from_mw", pl.Float64, "Level at the start of the period, MW."),
    Col("levelTo", "level_to_mw", pl.Float64, "Level at the end of the period, MW."),
    Col("offer", "offer_gbp_per_mwh", pl.Float64, "Offer price, GBP/MWh."),
    Col("bid", "bid_gbp_per_mwh", pl.Float64, "Bid price, GBP/MWh."),
)


BOD_SCHEMA: Final[dict[str, Any]] = {
    **schema_of(columns=BOD_COLUMNS),
    "price_is_placeholder": pl.Boolean,
}


def fetch_bod_month(key: str, *, batches: list[list[str]]) -> pl.DataFrame:
    """Fetch BOD for every batch of BMUs over a `<first>_<after>` chunk."""
    first, after = chunk_bounds(key=key)
    start, stop = utc_midnight(day=first), utc_midnight(day=after)
    frames = []
    for batch in batches:
        rows = get_json(
            url=BOD_URL,
            params=[
                ("from", f"{start:{REQUEST_TIME}}"),
                ("to", f"{stop:{REQUEST_TIME}}"),
                *bmu_params(batch=batch),
            ],
        )
        frames.append(frame_from_rows(rows=rows, columns=BOD_COLUMNS, what=f"BOD {key}"))
    frame = pl.concat(frames)
    frame = frame.filter(pl.col("time").is_between(start, stop, closed="left")).unique()
    if frame.is_empty():
        raise IncompleteChunkError(f"BOD chunk {key} for {batches} has no rows")
    return frame.with_columns(
        price_is_placeholder=(
            (pl.col("offer_gbp_per_mwh").abs() >= PLACEHOLDER_PRICE_GBP_PER_MWH)
            | (pl.col("bid_gbp_per_mwh").abs() >= PLACEHOLDER_PRICE_GBP_PER_MWH)
        )
    )


def keep_first_pair(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep only bid-offer pairs 1 and -1."""
    return frame.filter(pl.col("pair_id").is_in(BOD_FIRST_PAIRS))


def run_bod(
    *,
    root: Path,
    bmu_list: Path,
    start: date,
    end: date,
    first_pair_only: bool,
    threads: int,
) -> None:
    """Download BOD for the BMUs in `bmu_list`, then write the parquet, lineage, and README."""
    bmu_ids = read_bmu_list(path=bmu_list)
    if not bmu_ids:
        raise ValueError(f"{bmu_list} lists no BMUs")
    batches = bmu_batches(bmu_ids=bmu_ids)
    output_dir = root / "elexon_bod"
    keys = month_chunks(start=start, end=end)
    chunks: dict[str, list[str]] = {"fetched": [], "cached": [], "not_published": []}
    parts = []
    zero_chunks: list[str] = []
    for batch in batches:
        cache_dir = output_dir / "_day_cache" / f"batch_{list_hash(bmu_ids=batch)}"
        outcome = fetch_missing_chunks(
            cache_dir=cache_dir,
            keys=keys,
            fetch=partial(fetch_bod_month, batches=[batch]),
            label=f"elexon_bod {cache_dir.name}",
            threads=threads,
        )
        for kind, done in outcome.items():
            chunks[kind] += done
        zero_chunks += [
            f"{cache_dir.name}/{key}" for key in empty_chunks(cache_dir=cache_dir, keys=keys)
        ]
        parts.append(read_chunks(cache_dir=cache_dir, keys=keys, schema=BOD_SCHEMA))
    frame = pl.concat(parts)
    dropped = 0
    if first_pair_only:
        kept = keep_first_pair(frame=frame)
        dropped = frame.height - kept.height
        frame = kept
    frame = frame.sort("bmu_id", "time", "pair_id")
    write_parquet_atomic(frame=frame, path=output_dir / "elexon_bod.parquet")
    coverage = summarise_gaps(
        expected=expected_half_hour_starts(start=start, end=end), actual=frame["time"]
    )
    bmus_with_rows = frame["bmu_id"].n_unique()
    pairs = sorted(frame["pair_id"].unique().to_list())
    kept_text = (
        "Only pairs 1 and -1 are kept, the offer and bid nearest the physical notification. "
        f"The script dropped {dropped} rows of the other pairs."
        if first_pair_only
        else "Every bid-offer pair is kept (no `--first-pair-only`)."
    )
    write_lineage(
        product_dir=output_dir,
        note={
            "source_address": BOD_URL,
            "request": f"{len(bmu_ids)} BMUs in {len(batches)} hash batches, {start} to {end}",
            "window": [start.isoformat(), end.isoformat()],
            "script": SCRIPT_PATH,
            "bmus_requested": bmu_ids,
            "bmu_list_hash": list_hash(bmu_ids=bmu_ids),
            "rows": frame.height,
            "bmus_with_rows": bmus_with_rows,
            "first_pair_only": first_pair_only,
            "rows_dropped_by_first_pair_only": dropped,
            "pair_ids_present": pairs,
            "chunks_not_published": sorted(set(chunks["not_published"])),
            "chunks_with_zero_rows": zero_chunks,
            "rows_with_placeholder_price": int(frame["price_is_placeholder"].sum()),
            "period_coverage_any_bmu": coverage,
        },
    )
    write_readme(
        product_dir=output_dir,
        title="Elexon bid-offer prices (BOD), for the listed balancing mechanism units (BMUs)",
        source_page="https://bmrs.elexon.co.uk/bid-offer-data",
        script_path=SCRIPT_PATH,
        attribution=ELEXON_ATTRIBUTION,
        cache_hint="_day_cache/batch_<hash>/",
        licence=ELEXON_LICENCE,
        timestamp_convention=(
            "`time` is the UTC start of the settlement period the bid-offer prices apply to, "
            "taken from the source's `timeFrom`, and `time_to` is its end. The prices are the "
            "ones submitted for that period. The window is whole UTC days, so in summer it "
            "starts at settlement period 3 of the first date and ends at period 2 of the day "
            "after the last date."
        ),
        columns={
            **{col.name: col.doc for col in BOD_COLUMNS},
            "price_is_placeholder": (
                f"True if the offer or bid price is at least {PLACEHOLDER_PRICE_GBP_PER_MWH:.0f} "
                "in magnitude. The raw price is kept."
            ),
        },
        row_summary=(
            f"- {frame.height} rows for {bmus_with_rows} of {len(bmu_ids)} requested BMUs.\n"
            f"- Pair identifiers present: {pairs}. {kept_text}\n"
            f"- Settlement periods with at least one row from any BMU: {coverage['distinct_rows']} "
            f"of {coverage['expected_rows']}; {coverage['missing_count']} are missing."
        ),
        gotchas=[
            (
                "Each BMU submits a few pairs per period. Most periods repeat the previous "
                "period's prices, so the table is long but compresses well."
            ),
            (
                "Prices of 99999 and -99999, and values near 9999 such as -9999, are placeholders, "
                "not prices. The placeholders appear on the outer pairs and also on pairs 1 and "
                "-1 for some BMUs. A value just inside the threshold, such as -9979, is not "
                "flagged. Filter on `price_is_placeholder` before any average."
            ),
            "A BMU only has rows for periods in which it submitted prices.",
        ],
    )
    print(f"elexon_bod: wrote {frame.height} rows for {bmus_with_rows} of {len(bmu_ids)} BMUs")


# Frequency

FREQUENCY_SUMMARY_SCHEMA: Final[dict[str, Any]] = {
    "time": UTC_TIME,
    "frequency_mean_hz": pl.Float64,
    "frequency_min_hz": pl.Float32,
    "frequency_max_hz": pl.Float32,
    "frequency_std_hz": pl.Float64,
    "sample_count": pl.Int32,
}
FREQUENCY_RAW_SCHEMA: Final[dict[str, Any]] = {"time": UTC_TIME, "frequency_hz": pl.Float32}
FREQUENCY_SUMMARY_COLUMNS: Final[dict[str, str]] = {
    "time": "UTC start of the half-hour the statistics cover (window is start inclusive).",
    "frequency_mean_hz": "Mean of the samples in the half-hour, Hz.",
    "frequency_min_hz": "Lowest sample in the half-hour, Hz.",
    "frequency_max_hz": "Highest sample in the half-hour, Hz.",
    "frequency_std_hz": "Sample standard deviation (n - 1) of the samples in the half-hour, Hz.",
    "sample_count": "Number of samples in the half-hour (120 if every 15-second value is present).",
}


def summarise_frequency(*, raw: pl.DataFrame, column: str) -> pl.DataFrame:
    """Summarise a frequency series to UTC half-hours: mean, minimum, maximum, std, count.

    Args:
        raw: A table with a UTC `time` column and a frequency column in hertz.
        column: Name of the frequency column.

    Returns:
        One row per half-hour that holds at least one sample, with `FREQUENCY_SUMMARY_SCHEMA`.
    """
    value = pl.col(column).cast(pl.Float64)
    return (
        raw.sort("time")
        .group_by_dynamic("time", every="30m", closed="left", label="left")
        .agg(
            frequency_mean_hz=value.mean(),
            frequency_min_hz=value.min().cast(pl.Float32),
            frequency_max_hz=value.max().cast(pl.Float32),
            frequency_std_hz=value.std(),
            sample_count=value.count().cast(pl.Int32),
        )
        .filter(pl.col("sample_count") > 0)
        .select(*FREQUENCY_SUMMARY_SCHEMA)
    )


def fetch_frequency_day(key: str) -> pl.DataFrame:
    """Fetch the 15-second frequency for one UTC day (`<day>_<next day>`)."""
    first, after = chunk_bounds(key=key)
    start, stop = utc_midnight(day=first), utc_midnight(day=after)
    rows = get_json(
        url=FREQUENCY_URL,
        params=[("from", f"{start:{REQUEST_TIME}}"), ("to", f"{stop:{REQUEST_TIME}}")],
    )
    columns = (
        Col("measurementTime", "time", UTC_TIME, ""),
        Col("frequency", "frequency_hz", pl.Float32, ""),
    )
    frame = frame_from_rows(rows=rows, columns=columns, what=f"frequency {key}")
    frame = frame.filter(pl.col("time").is_between(start, stop, closed="left")).unique(
        maintain_order=True
    )
    if frame.is_empty():
        raise IncompleteChunkError(f"frequency chunk {key} is empty, so is not published yet")
    return frame


def run_elexon_frequency(*, root: Path, start: date, end: date, threads: int) -> None:
    """Download Elexon's 15-second frequency; write the summary and, if small, the raw values."""
    output_dir = root / "elexon_frequency"
    cache_dir = output_dir / "_day_cache"
    keys = day_chunks(start=start, end=end)
    outcome = fetch_missing_chunks(
        cache_dir=cache_dir,
        keys=keys,
        fetch=fetch_frequency_day,
        label="elexon_frequency",
        threads=threads,
    )
    raw = read_chunks(cache_dir=cache_dir, keys=keys, schema=FREQUENCY_RAW_SCHEMA)
    raw = raw.unique(maintain_order=True).sort("time")
    summary = summarise_frequency(raw=raw, column="frequency_hz")
    write_parquet_atomic(frame=summary, path=output_dir / "elexon_frequency.parquet")
    cache_bytes = sum(
        (cache_dir / f"{key}.parquet").stat().st_size
        for key in keys
        if (cache_dir / f"{key}.parquet").exists()
    )
    store_raw = cache_bytes < RAW_FREQUENCY_MAX_BYTES
    if store_raw:
        write_parquet_atomic(frame=raw, path=output_dir / "elexon_frequency_15s.parquet")
    coverage = summarise_gaps(
        expected=expected_half_hour_starts(start=start, end=end), actual=summary["time"]
    )
    short = summary.filter(pl.col("sample_count") < SAMPLES_PER_HALF_HOUR_15S).height
    _write_frequency_docs(
        output_dir=output_dir,
        title="Elexon system frequency, 15-second values and half-hourly summary",
        source_page="https://bmrs.elexon.co.uk/system-frequency",
        source_address=FREQUENCY_URL,
        attribution=ELEXON_ATTRIBUTION,
        licence=ELEXON_LICENCE,
        raw_rows=raw.height,
        store_raw=store_raw,
        coverage=coverage,
        short=short,
        expected_samples=SAMPLES_PER_HALF_HOUR_15S,
        outcome=outcome,
        start=start,
        end=end,
        raw_name="elexon_frequency_15s.parquet",
        summary_name="elexon_frequency.parquet",
        sample_text="15-second",
    )
    print(f"elexon_frequency: {raw.height} raw rows, {summary.height} half-hours")


def _write_frequency_docs(
    *,
    output_dir: Path,
    title: str,
    source_page: str,
    source_address: str,
    attribution: str | None,
    licence: str,
    raw_rows: int,
    store_raw: bool,
    coverage: dict[str, Any],
    short: int,
    expected_samples: int,
    outcome: dict[str, list[str]],
    start: date,
    end: date,
    raw_name: str,
    summary_name: str,
    sample_text: str,
    raw_never_stored: bool = False,
    extra_gotchas: Sequence[str] = (),
) -> None:
    write_lineage(
        product_dir=output_dir,
        note={
            "source_address": source_address,
            "window": [start.isoformat(), end.isoformat()],
            "script": SCRIPT_PATH,
            "raw_rows": raw_rows,
            "raw_stored": store_raw,
            "half_hour_coverage": coverage,
            "half_hours_with_fewer_samples_than_expected": short,
            "expected_samples_per_half_hour": expected_samples,
            "chunks_not_published": sorted(outcome["not_published"]),
        },
    )
    if raw_never_stored:
        raw_text = f"The {sample_text} values are summarised during download and never stored."
    elif store_raw:
        raw_text = (
            f"`{raw_name}` holds every {sample_text} value as Float32, in {raw_rows} rows with "
            "columns `time` and `frequency_hz`."
        )
    else:
        raw_text = (
            f"The {raw_rows} raw {sample_text} values are not stored because the cache exceeds "
            f"{RAW_FREQUENCY_MAX_BYTES / 1e9:g} GB."
        )
    write_readme(
        product_dir=output_dir,
        title=title,
        source_page=source_page,
        script_path=SCRIPT_PATH,
        attribution=attribution,
        cache_hint="_day_cache/",
        licence=licence,
        timestamp_convention=(
            "`time` is UTC. In the summary it is the start of a half-hour window that includes "
            "its first instant and excludes its last. The summary carries no settlement date or "
            "period number, so on a clock-change day match it to settlement periods by `time`."
        ),
        columns=FREQUENCY_SUMMARY_COLUMNS,
        row_summary=(
            f"- `{summary_name}`: {coverage['distinct_rows']} of {coverage['expected_rows']} "
            f"half-hours have a sample; {coverage['missing_count']} are missing. "
            f"{short} half-hours have fewer than {expected_samples} samples.\n"
            f"- {raw_text}"
        ),
        gotchas=[
            *extra_gotchas,
            "Frequency is a single national value, not a per-site measurement.",
            "A half-hour with fewer samples than expected has a less reliable minimum and maximum.",
        ],
    )


def neso_frequency_urls() -> dict[str, str]:
    """Map each `YYYY-MM` to the download URL of NESO's one-second frequency CSV for that month."""
    package = get_json(url=NESO_FREQUENCY_PACKAGE_URL)
    urls: dict[str, str] = {}
    for resource in package["result"]["resources"]:
        match = re.search(r"/(fnew-(\d{4})-(\d{1,2})\.csv)$", resource.get("url") or "")
        if match:
            urls[f"{match.group(2)}-{int(match.group(3)):02d}"] = resource["url"]
    return urls


def fetch_neso_frequency_month(key: str, *, urls: dict[str, str]) -> pl.DataFrame:
    """Download one month of one-second CSV and return its half-hourly summary only."""
    first, after = chunk_bounds(key=key)
    url = urls.get(f"{first:%Y-%m}")
    if url is None:
        raise IncompleteChunkError(f"no NESO frequency CSV is listed for {first:%Y-%m}")
    response = get_response(url=url)
    raw = pl.read_csv(
        io.BytesIO(response.content),
        schema={"dtm": pl.String, "f": pl.Float64},
    ).select(
        time=pl.col("dtm")
        .str.to_datetime(format="%Y-%m-%d %H:%M:%S", time_unit="us")
        .dt.replace_time_zone("UTC"),
        frequency_hz=pl.col("f"),
    )
    start, stop = utc_midnight(day=first), utc_midnight(day=after)
    raw = raw.filter(pl.col("time").is_between(start, stop, closed="left"))
    if raw.is_empty():
        raise IncompleteChunkError(f"NESO frequency CSV for {first:%Y-%m} is empty")
    return summarise_frequency(raw=raw, column="frequency_hz")


def run_neso_frequency(*, root: Path, start: date, end: date, threads: int) -> None:
    """Download NESO's one-second frequency CSVs and store half-hourly summaries only."""
    output_dir = root / "neso_frequency"
    cache_dir = output_dir / "_day_cache"
    keys = month_chunks(start=start, end=end)
    urls = neso_frequency_urls()
    outcome = fetch_missing_chunks(
        cache_dir=cache_dir,
        keys=keys,
        fetch=partial(fetch_neso_frequency_month, urls=urls),
        label="neso_frequency",
        threads=min(threads, 2),
    )
    summary = read_chunks(cache_dir=cache_dir, keys=keys, schema=FREQUENCY_SUMMARY_SCHEMA)
    summary = summary.unique(maintain_order=True).sort("time")
    write_parquet_atomic(frame=summary, path=output_dir / "neso_frequency.parquet")
    coverage = summarise_gaps(
        expected=expected_half_hour_starts(start=start, end=end), actual=summary["time"]
    )
    short = summary.filter(pl.col("sample_count") < SAMPLES_PER_HALF_HOUR_1S).height
    _write_frequency_docs(
        output_dir=output_dir,
        title="NESO one-second system frequency, half-hourly summary",
        source_page="https://www.neso.energy/data-portal/system-frequency-data",
        source_address=NESO_FREQUENCY_PACKAGE_URL,
        attribution=None,
        licence=NESO_LICENCE,
        raw_rows=0,
        store_raw=False,
        coverage=coverage,
        short=short,
        expected_samples=SAMPLES_PER_HALF_HOUR_1S,
        outcome=outcome,
        start=start,
        end=end,
        raw_name="",
        summary_name="neso_frequency.parquet",
        sample_text="one-second",
        raw_never_stored=True,
        extra_gotchas=(
            (
                "The CSV's `dtm` column carries no time zone, and the script reads it as UTC. "
                "Over 96 half-hours of 30 and 31 March 2026 (British Summer Time), the "
                "half-hourly means differed from Elexon's 15-second values by 0.9 millihertz on "
                "average with `dtm` read as UTC, against about 58 millihertz with `dtm` shifted "
                "by one hour."
            ),
        ),
    )
    print(f"neso_frequency: {summary.height} half-hours")


def main() -> None:
    """Parse the command line and run each requested source."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, default=MARKET_DOWNLOADS_DIR)
    parser.add_argument("--start", type=date.fromisoformat, default=WINDOW_START)
    parser.add_argument("--end", type=date.fromisoformat, default=WINDOW_END)
    parser.add_argument("--sources", nargs="+", choices=ALL_SOURCES, default=ELEXON_DEFAULT_SOURCES)
    parser.add_argument("--bmu-list", type=Path, default=DEFAULT_BMU_LIST)
    parser.add_argument("--first-pair-only", action="store_true")
    parser.add_argument("--threads", type=int, default=FETCH_THREADS)
    args = parser.parse_args()
    for name in args.sources:
        if name == "elexon_bod":
            run_bod(
                root=args.output_root,
                bmu_list=args.bmu_list,
                start=args.start,
                end=args.end,
                first_pair_only=args.first_pair_only,
                threads=args.threads,
            )
        elif name == "elexon_frequency":
            run_elexon_frequency(
                root=args.output_root, start=args.start, end=args.end, threads=args.threads
            )
        elif name == "neso_frequency":
            run_neso_frequency(
                root=args.output_root, start=args.start, end=args.end, threads=args.threads
            )
        else:
            run_series(
                spec=SERIES[name],
                root=args.output_root,
                start=args.start,
                end=args.end,
                threads=args.threads,
            )
    print(f"Finished at {datetime.now(UTC):%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
