"""Download Met Office MIDAS Open hourly radiation and hourly weather observations from CEDA.

One-off throwaway script for the weather downloads in
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. It fetches
`dataset-version-202607` of two MIDAS Open datasets, `uk-radiation-obs` and
`uk-hourly-weather-obs`, at quality-control version 1 only, for the stations the studies need and
for each calendar year from 2017 to 2025. CEDA serves one BADC-CSV file per station per year, and
this script mirrors CEDA's own layout under `MIDAS-OPEN/raw/`, then writes one tidy parquet per
dataset, a `lineage.json`, and a `README.md` into `MIDAS-OPEN/`.

**Station identity stays private apart from the numeric MIDAS id.** The station-metadata CSVs in
`MIDAS-OPEN/_station_metadata/` carry each station's name, latitude, longitude, and elevation. This
script reads only `station_file_name` and `historic_county` from them, because those two fields
build the download URL. No log line, exception message, parquet, lineage note, or README carries a
station name, a coordinate, or a download URL, because the URL contains the station's file name.
The raw directory tree mirrors CEDA's layout, so the station's file name does appear in directory
names on disk.

**Downloads need a CEDA access token.** The token is read from `CEDA_TOKEN` in the main checkout's
`.env` and sent in an `Authorization` header that is never logged, never written to disk, and never
placed in an exception message. No HTTP redirect is followed, because CEDA answers a bad or expired
token by redirecting to its login page.

**The station lists are hand-typed.** The rule that reproduces them is: `last_year` 2025 or later
in the station-metadata CSV, and within 100 km of at least one of the nine anonymised study sites.
That gives all 10 radiation stations and 37 of the 38 hourly-weather stations; station 24219
(record ends 2024) is the exception. No other station with `last_year` 2025 is within 100 km of
any site.

**CEDA's own directory listing decides which files exist.** The station-metadata CSV's
`first_year` and `last_year` bound a station's record but do not promise a file for every year in
between: station 00390 has no `uk-hourly-weather-obs` file for 2017 to 2024, for example. So the
script first reads each station's anonymous JSON directory listing, which also gives every file's
MD5 checksum. A year missing from the listing is recorded as expected and not requested. A listed
file that fails to download, or whose MD5 differs from the listing's, is recorded as an error, the
run continues, and the script exits non-zero at the end, listing every such file. HTTP 401, HTTP
403, or a redirect aborts at once, because the token is wrong or has expired. HTTP 429, any 5xx
status, and network failures are retried with exponential backoff.

**Checkpointed per file**, following the `data-download` skill: each file must match the MD5
checksum in CEDA's directory listing before it is written to a `.partial` file and renamed into
place. A re-run skips every file already on disk, so a crash costs at most one file.

Run it with `uv run python studies/weather_downloads/fetch_midas_open.py`. `--sample` fetches one
file per dataset and prints its column names with their non-null counts, `--dataset` restricts the
run to one dataset, and `--readme-only` regenerates `README.md` and `lineage.json` from the tidy
parquets and the raw files already on disk, without downloading.
"""

import argparse
import csv
import hashlib
import http.client
import io
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from calendar import isleap
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import IO, Any, Final, Literal

import polars as pl
from dotenv import load_dotenv
from lineage import write_lineage_note, write_readme
from paths import REPO_DATA_DIR, WEATHER_DOWNLOADS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_midas_open")

DatasetType = Literal["uk-radiation-obs", "uk-hourly-weather-obs"]
"""The two MIDAS Open datasets this script fetches, in CEDA's own naming."""

DATASETS: Final[tuple[DatasetType, ...]] = ("uk-radiation-obs", "uk-hourly-weather-obs")

FetchStatusType = Literal["downloaded", "cached", "expected_missing", "error"]
"""What happened to one station-year file: fetched now, already on disk, absent from CEDA's
directory listing, or a failure that the end-of-run summary lists."""

DATASET_VERSION: Final[str] = "dataset-version-202607"
QC_VERSION: Final[str] = "qc-version-1"
_FILENAME_VERSION_TAG: Final[str] = "dv-202607"
_FILENAME_QC_TAG: Final[str] = "qcv-1"

DOWNLOAD_BASE_URL: Final[str] = "https://dap.ceda.ac.uk/badc/ukmo-midas-open/data"
"""CEDA's download server. A file download needs a bearer token; a directory listing does not."""

SOURCE_ADDRESS: Final[str] = "https://data.ceda.ac.uk/badc/ukmo-midas-open/data/"
"""The browsable archive root recorded in `lineage.json`. Appending `?json` to any directory below
it returns that directory's file listing, with each file's size and MD5 checksum, anonymously."""

DEFAULT_FIRST_YEAR: Final[int] = 2017
DEFAULT_LAST_YEAR: Final[int] = 2025
SAMPLE_YEAR: Final[int] = 2023

RADIATION_STATION_IDS: Final[tuple[str, ...]] = (
    "00370",
    "00384",
    "00395",
    "00456",
    "00461",
    "00554",
    "00556",
    "00583",
    "00595",
    "61986",
)
"""The `uk-radiation-obs` stations the studies need, as zero-padded MIDAS `src_id` values."""

HOURLY_WEATHER_STATION_IDS: Final[tuple[str, ...]] = (
    "00370",
    "00373",
    "00382",
    "00384",
    "00386",
    "00390",
    "00393",
    "00395",
    "00405",
    "00407",
    "00409",
    "00413",
    "00421",
    "00426",
    "00435",
    "00454",
    "00455",
    "00456",
    "00461",
    "00465",
    "00525",
    "00542",
    "00554",
    "00556",
    "00583",
    "00595",
    "03805",
    "16725",
    "19204",
    "24219",
    "30476",
    "30529",
    "56423",
    "56986",
    "61986",
    "62152",
    "62216",
    "62265",
)
"""The `uk-hourly-weather-obs` stations the studies need, as zero-padded MIDAS `src_id` values."""

_EXPECTED_STATION_COUNTS: Final[dict[DatasetType, int]] = {
    "uk-radiation-obs": 10,
    "uk-hourly-weather-obs": 38,
}
"""How many stations each list above was specified to hold, checked at import time so a dropped
or duplicated id in the hand-typed lists fails before any request is made."""

STATION_IDS: Final[dict[DatasetType, tuple[str, ...]]] = {
    "uk-radiation-obs": RADIATION_STATION_IDS,
    "uk-hourly-weather-obs": HOURLY_WEATHER_STATION_IDS,
}

for _dataset, _ids in STATION_IDS.items():
    if len(_ids) != _EXPECTED_STATION_COUNTS[_dataset] or len(set(_ids)) != len(_ids):
        _msg = (
            f"{_dataset}: expected {_EXPECTED_STATION_COUNTS[_dataset]} distinct station ids, "
            f"found {len(set(_ids))} distinct among {len(_ids)}"
        )
        raise RuntimeError(_msg)

PRODUCT_DIR: Final[Path] = WEATHER_DOWNLOADS_DIR / "MIDAS-OPEN"
RAW_DIR: Final[Path] = PRODUCT_DIR / "raw"
STATION_METADATA_DIR: Final[Path] = PRODUCT_DIR / "_station_metadata"
"""Read only. Holds CEDA's station-metadata CSVs, which carry coordinates and station names."""

ENV_PATH: Final[Path] = REPO_DATA_DIR.parent / ".env"
"""The main checkout's `.env`, which holds `CEDA_TOKEN`. A linked worktree has no `.env` of its
own, and `paths.py` loads only the running checkout's, so this path is loaded explicitly."""

REQUEST_TIMEOUT_SECONDS: Final[float] = 120.0
MAX_ATTEMPTS: Final[int] = 5
BACKOFF_BASE_SECONDS: Final[float] = 2.0
"""The first retry waits this long; each later retry waits twice as long as the one before."""
POLITE_DELAY_SECONDS: Final[float] = 0.2
"""Pause after every network request, so the sequential loop never hammers CEDA."""

KNOTS_TO_M_S: Final[float] = 0.514444
"""One international knot (1,852 m per hour) in metres per second."""

LICENCE: Final[str] = "Open Government Licence v3.0"
LICENCE_URL: Final[str] = (
    "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"
)
QC_FLAGS_URL: Final[str] = "https://dap.ceda.ac.uk/badc/ukmo-midas/metadata/doc/QC_J_flags.html"
USER_GUIDE_URL: Final[str] = "https://help.ceda.ac.uk/article/4982-midas-open-user-guide"
SCRIPT_PATH: Final[str] = "studies/weather_downloads/fetch_midas_open.py"

VALIDATION_GOTCHAS: Final[tuple[str, ...]] = (
    (
        "Validated on 2026-09-24 by `validate_midas_open.py`, which writes `validation.json` "
        "next to this file. No `(src_id, time)` key repeats in either parquet, and no float column "
        "holds a NaN. The findings below are that run's."
    ),
    (
        "Only global irradiation is reported: at all 10 radiation stations the diffuse, direct, "
        "balance, and tilted-plane columns are null on every row (the illuminance column is null "
        "too, so the tidy file drops it). A study cannot use these stations for a beam or diffuse "
        "split."
    ),
    (
        "`uk-radiation-obs` has gaps: 3,216 consecutive hours are missing at one station from "
        "2021-05-14, 840 hours at another from 2018-03-26, and every other gap is 432 hours or "
        "shorter. Completeness between each station's first and last hour is 94.6% to 99.8%. "
        "Station 61986 starts on 2017-05-27."
    ),
    (
        "Hourly global irradiation is hour-ending, in UTC: the label that best matches the "
        "top-of-atmosphere flux is the label itself at all 10 stations, and the daily profile "
        "peaks at the 12:00 and 13:00 labels. Night-time values are zero, except two rows of "
        "station 61986 in October 2019 (886 and 123 kJ/m2 at 06:00 UTC, before sunrise), which "
        "carry flag 6006, the same flag as 5,075 ordinary rows, so the flag does not single them "
        "out. "
        "Three rows at station 61986 are negative (the lowest is -41 kJ/m2)."
    ),
    (
        "The 99th percentile of the clearness index (hourly global irradiance divided by the "
        "top-of-atmosphere flux) is 0.79 to 0.80 at every station, and the daytime rows above 1.0 "
        "number 45 of 293,149. Annual mean clearness index differs between stations by up to 0.07 "
        "in a year, and the largest 12-month shift of one station against the others is 0.028, so "
        "no station shows a step change."
    ),
    (
        "Against ERA5's hourly global irradiance in the cell containing the station (two "
        "stations, 20,000 daytime hours each), the correlation is 0.87, the mean difference is "
        "+2.6 and +8.4 W/m2 (station minus ERA5), and the mean absolute difference is 73 to 75 "
        "W/m2. That is the size of difference expected between a point pyranometer and a 25 km "
        "cell, so the hour labels and the unit conversion agree."
    ),
    (
        "Only 18 of the 38 hourly-weather stations report a wind speed: 12 stations of message "
        "type `AWSHRLY` carry no wind in this dataset at all (0% non-null), and 8 stations are "
        "`DLY3208` daily-only (see above). Air temperature is reported at all 38. The 12 "
        "wind-less stations may hold wind in another MIDAS Open dataset, which this download does "
        "not include."
    ),
    (
        "Hourly-weather completeness at the 30 hourly stations is 90.5% to 100.0% of hours in "
        "the fetched years; the longest gaps are 803 hours at one station from 2018-03-25 and 664 "
        "at another from 2021-02-03. Two hourly stations start late: 00465 (2022-02-22) and "
        "62265 (2023-03-31)."
    ),
    (
        "Wind direction runs 0 to 360 and occupies every 10-degree bin. MIDAS writes north as 360 "
        "(21,889 rows are exactly 360) and calm as 0, so a wind-direction average needs the 360 "
        "and 0 cases handled: only 2 rows have zero speed and a non-zero direction. Wind speed "
        "peaks at 24.7 m/s. Only one row has a 10-minute gust more than 0.6 m/s below "
        "the mean speed."
    ),
    (
        "Air temperature runs from -11.0 to 39.6 degC. The dew point is never more than 0.5 degC "
        "above it. Relative humidity reaches 107.5%, so clip it at 100 before use. Air "
        "temperature has no run of 24 identical hours, and wind speed with direction has no "
        "identical run of 12 non-zero hours."
    ),
    (
        "Against ERA5's 2 m temperature (7 stations, about 55,000 hours each), the correlation is "
        "0.97 to 0.99, the mean difference (station minus ERA5) is -0.1 to -0.6 degC, and the mean "
        "absolute difference is 0.6 to 1.1 degC. No gridded ERA5 wind product is on disk, so the "
        "wind speed and direction were checked for plausibility and steps only, not against "
        "another product."
    ),
    (
        "The largest 12-month level shift against the other stations is 0.54 degC for air "
        "temperature and 0.63 m/s (about 12% of that station's mean) for wind speed, at one "
        "station in February 2018; no other wind shift exceeds 0.45 m/s."
    ),
    (
        "Quality-control flag 106 marks whole stations, not bad hours: it is on almost "
        "every air-temperature row at stations 00405 and 62216, and on almost every "
        "wind-speed row at station 62265, and it does not make the values implausible (station "
        "00405's temperature agrees with ERA5 as closely as flag-6 stations do). Flag 6 is the "
        "usual value (94.7% of air-temperature rows). Filtering on the flag would drop those "
        "stations wholesale."
    ),
)
"""Findings of `validate_midas_open.py`, appended to the README's gotchas. Written by hand from
that script's `validation.json`, so re-run the script and re-check these numbers after any
re-download."""


@dataclass(frozen=True)
class StationMetadata:
    """The station-metadata fields this script uses, and nothing that locates the station."""

    src_id: str
    station_file_name: str
    historic_county: str


@dataclass(frozen=True)
class DatasetSpec:
    """How one dataset's BADC-CSV files become its tidy parquet.

    `string_columns`, `int_columns`, and `float_columns` together are every column the tidy step
    reads besides `src_id` and `time_column`; a file missing any of them raises. `value_columns` are
    the measurements counted by the de-duplication rule in `_deduplicate`. `output_columns` is the
    tidy parquet's column order.
    """

    time_column: str
    string_columns: tuple[str, ...]
    int_columns: tuple[str, ...]
    float_columns: tuple[str, ...]
    value_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    coverage_columns: tuple[str, ...]
    tidy_filename: str


_RADIATION_VALUES: Final[tuple[str, ...]] = (
    "glbl_irad_amt",
    "difu_irad_amt",
    "direct_irad",
    "irad_bal_amt",
    "glbl_s_lat_irad_amt",
    "glbl_horz_ilmn",
)

_WEATHER_FLOAT_VALUES: Final[tuple[str, ...]] = (
    "wind_direction",
    "wind_speed",
    "q10mnt_mxgst_spd",
    "air_temperature",
    "dewpoint",
    "rltv_hum",
    "msl_pressure",
    "wmo_hr_sun_dur",
    "drv_hr_sun_dur",
    "cs_hr_sun_dur",
)
_WEATHER_INT_FLAGS: Final[tuple[str, ...]] = (
    "wind_direction_q",
    "wind_speed_q",
    "q10mnt_mxgst_spd_q",
    "air_temperature_q",
    "dewpoint_q",
    "msl_pressure_q",
    "cld_ttl_amt_id_q",
    "wmo_hr_sun_dur_q",
    "cs_hr_sun_dur_q",
)

SPECS: Final[dict[DatasetType, DatasetSpec]] = {
    "uk-radiation-obs": DatasetSpec(
        time_column="ob_end_time",
        string_columns=("id_type", "met_domain_name"),
        int_columns=(
            "ob_hour_count",
            "version_num",
            "rec_st_ind",
            *(f"{name}_q" for name in _RADIATION_VALUES),
        ),
        float_columns=_RADIATION_VALUES,
        value_columns=_RADIATION_VALUES,
        output_columns=(
            "src_id",
            "time",
            "ob_hour_count",
            "id_type",
            "met_domain_name",
            "rec_st_ind",
            *(column for name in _RADIATION_VALUES for column in (name, f"{name}_q")),
        ),
        coverage_columns=("glbl_irad_amt", "difu_irad_amt", "direct_irad"),
        tidy_filename="uk_radiation_obs_hourly.parquet",
    ),
    "uk-hourly-weather-obs": DatasetSpec(
        time_column="ob_time",
        string_columns=("id_type", "met_domain_name"),
        int_columns=(
            "version_num",
            "rec_st_ind",
            "wind_speed_unit_id",
            "cld_ttl_amt_id",
            *_WEATHER_INT_FLAGS,
        ),
        # `drv_hr_sun_dur_q` is declared `float` in the file header, so it is read as Float64
        # and narrowed to Int64 by `_narrow_whole_float` only if every value is whole.
        float_columns=(*_WEATHER_FLOAT_VALUES, "drv_hr_sun_dur_q"),
        value_columns=(*_WEATHER_FLOAT_VALUES, "cld_ttl_amt_id"),
        output_columns=(
            "src_id",
            "time",
            "id_type",
            "met_domain_name",
            "rec_st_ind",
            "wind_speed_unit_id",
            "wind_direction",
            "wind_direction_q",
            "wind_speed",
            "wind_speed_q",
            "wind_speed_m_s",
            "q10mnt_mxgst_spd",
            "q10mnt_mxgst_spd_q",
            "air_temperature",
            "air_temperature_q",
            "dewpoint",
            "dewpoint_q",
            "rltv_hum",
            "msl_pressure",
            "msl_pressure_q",
            "cld_ttl_amt_id",
            "cld_ttl_amt_id_q",
            "wmo_hr_sun_dur",
            "wmo_hr_sun_dur_q",
            "drv_hr_sun_dur",
            "drv_hr_sun_dur_q",
            "cs_hr_sun_dur",
            "cs_hr_sun_dur_q",
        ),
        coverage_columns=("wind_speed", "wind_direction", "air_temperature"),
        tidy_filename="uk_hourly_weather_obs.parquet",
    ),
}
"""One entry per dataset. Every retained value column keeps its `_q` quality-control flag exactly
as delivered, and no row is filtered on a flag: which flags to trust is the study's decision."""


@dataclass(frozen=True)
class FetchOutcome:
    """What happened to one station-year file. `reason` never contains a URL or the token."""

    dataset: DatasetType
    src_id: str
    year: int
    status: FetchStatusType
    n_bytes: int = 0
    reason: str = ""


@dataclass
class TidyReport:
    """Row counts the tidy step removed, keyed by the reason each row was removed.

    Every per-station dictionary lists only stations with a non-zero count.
    """

    rows_read: int = 0
    rows_dropped_by_ob_hour_count: dict[str, int] = field(default_factory=dict)
    off_hour_rows_dropped_per_station: dict[str, int] = field(default_factory=dict)
    duplicate_rows_removed_per_station: dict[str, int] = field(default_factory=dict)
    columns_dropped_as_entirely_null: list[str] = field(default_factory=list)
    drv_hr_sun_dur_q_dtype: str = ""
    rows_written: int = 0


class CedaAuthError(RuntimeError):
    """CEDA refused the token with HTTP 401 or 403, so every later request would fail too."""


class _TransientError(Exception):
    """A failure worth retrying: HTTP 429, a 5xx status, or a network error."""


# --------------------------------------------------------------------------------------------------
# Station metadata and paths
# --------------------------------------------------------------------------------------------------


def _badc_table(*, text: str, label: str) -> str:
    """Return the CSV table between a BADC-CSV file's `data` line and its `end data` line.

    Args:
        text: The whole file's text.
        label: A description of the file for error messages, carrying no station name or path.

    Returns:
        The table text, starting with its header row.

    Raises:
        ValueError: If either marker line is missing.
    """
    text = text.replace("\r\n", "\n")
    _, data_marker, rest = text.partition("\ndata\n")
    table, end_marker, _ = rest.rpartition("\nend data")
    if not data_marker or not end_marker:
        msg = f"{label}: no `data` ... `end data` block found in the BADC-CSV file"
        raise ValueError(msg)
    return table


def _read_station_metadata(*, dataset: DatasetType) -> dict[str, StationMetadata]:
    """Read one dataset's station-metadata CSV, keeping only the fields this script uses.

    Args:
        dataset: Which dataset's metadata file to read.

    Returns:
        Every station in the file, keyed by zero-padded `src_id`.
    """
    path = (
        STATION_METADATA_DIR / f"midas-open_{dataset}_{_FILENAME_VERSION_TAG}_station-metadata.csv"
    )
    table = _badc_table(text=path.read_text(encoding="utf-8"), label=f"{dataset} station metadata")
    stations: dict[str, StationMetadata] = {}
    for row in csv.DictReader(io.StringIO(table)):
        src_id = row["src_id"].strip().zfill(5)
        stations[src_id] = StationMetadata(
            src_id=src_id,
            station_file_name=row["station_file_name"].strip(),
            historic_county=row["historic_county"].strip(),
        )
    return stations


def _stations_for(*, dataset: DatasetType) -> list[StationMetadata]:
    """Return the metadata of every station `STATION_IDS` lists for one dataset, in list order.

    Args:
        dataset: Which dataset's station list and metadata CSV to use.

    Returns:
        One `StationMetadata` per listed station id.

    Raises:
        RuntimeError: If any listed station id is absent from the dataset's metadata CSV.
    """
    metadata = _read_station_metadata(dataset=dataset)
    missing = [src_id for src_id in STATION_IDS[dataset] if src_id not in metadata]
    if missing:
        msg = f"{dataset}: station ids {missing} are not in the dataset's station-metadata CSV"
        raise RuntimeError(msg)
    return [metadata[src_id] for src_id in STATION_IDS[dataset]]


def _station_dir_name(*, station: StationMetadata) -> str:
    """Return CEDA's directory name for one station, `<src_id>_<station_file_name>`."""
    return f"{station.src_id}_{station.station_file_name}"


def _file_name(*, dataset: DatasetType, station: StationMetadata, year: int) -> str:
    """Return CEDA's file name for one station-year of one dataset."""
    return (
        f"midas-open_{dataset}_{_FILENAME_VERSION_TAG}_{station.historic_county}_"
        f"{_station_dir_name(station=station)}_{_FILENAME_QC_TAG}_{year}.csv"
    )


def _relative_path(*, dataset: DatasetType, station: StationMetadata, year: int) -> str:
    """Return one file's path below the dataset version, identical on CEDA and on local disk."""
    return (
        f"{station.historic_county}/{_station_dir_name(station=station)}/{QC_VERSION}/"
        f"{_file_name(dataset=dataset, station=station, year=year)}"
    )


def _raw_path(*, dataset: DatasetType, station: StationMetadata, year: int) -> Path:
    """Return where one station-year file lives on disk, mirroring CEDA's layout."""
    return RAW_DIR / dataset / _relative_path(dataset=dataset, station=station, year=year)


def _file_url(*, dataset: DatasetType, station: StationMetadata, year: int) -> str:
    """Return CEDA's download URL for one station-year file. Never log the result."""
    relative = _relative_path(dataset=dataset, station=station, year=year)
    return f"{DOWNLOAD_BASE_URL}/{dataset}/{DATASET_VERSION}/{relative}"


def _listed_files(*, dataset: DatasetType, station: StationMetadata) -> dict[str, str]:
    """Return every file in the station's quality-control-version-1 directory on CEDA.

    The listing is anonymous, so no token is sent.

    Args:
        dataset: Which dataset's directory to list.
        station: The station's metadata.

    Returns:
        Each file's name mapped to its MD5 checksum, as hexadecimal text.

    Raises:
        RuntimeError: If CEDA has no such directory, or the listing request fails.
    """
    directory = f"{station.historic_county}/{_station_dir_name(station=station)}/{QC_VERSION}"
    body = _download(
        url=f"{SOURCE_ADDRESS}{dataset}/{DATASET_VERSION}/{directory}/?json", token=None
    )
    if body is None:
        msg = f"{dataset} {station.src_id}: CEDA has no {QC_VERSION} directory for this station"
        raise RuntimeError(msg)
    items = json.loads(body)["items"]
    return {item["name"]: item["md5"] for item in items if item["type"] == "file"}


# --------------------------------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------------------------------


def _ceda_token() -> str:
    """Return `CEDA_TOKEN` from the main checkout's `.env`. Never log or print the result.

    Returns:
        The CEDA access token.

    Raises:
        RuntimeError: If `CEDA_TOKEN` is unset or empty.
    """
    load_dotenv(ENV_PATH)
    token = os.environ.get("CEDA_TOKEN", "").strip()
    if not token:
        msg = f"CEDA_TOKEN is not set: add it to {ENV_PATH} (a CEDA access token)"
        raise RuntimeError(msg)
    return token


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect, so a 3xx status reaches `_request_once` as an `HTTPError`."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> None:
        """Return `None`, which makes `urllib` raise the redirect as an `HTTPError`."""


_OPENER: Final[urllib.request.OpenerDirector] = urllib.request.build_opener(_NoRedirect)


def _request_once(*, url: str, token: str | None) -> bytes | None:
    """Make one GET request, authenticated when `token` is given.

    No redirect is followed. CEDA answers a bad or expired token with a redirect to its login page,
    so a redirect is treated like HTTP 401. No exception raised here carries the URL, the request
    headers, or the response body.

    Args:
        url: The file's download URL, or a directory listing's URL.
        token: The CEDA access token, or `None` for an anonymous directory listing.

    Returns:
        The response body, or `None` for HTTP 404.

    Raises:
        CedaAuthError: For HTTP 401, HTTP 403, or any redirect.
        _TransientError: For HTTP 429, a 5xx status, or a network failure.
        RuntimeError: For any other HTTP status.
    """
    request = urllib.request.Request(url)
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with _OPENER.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        code = error.code
        error.close()
        if code == 404:
            return None
        if code in {401, 403} or 300 <= code < 400:
            msg = (
                f"CEDA refused the access token (HTTP {code}; a 3xx is a redirect to its login "
                f"page); renew CEDA_TOKEN in {ENV_PATH}"
            )
            raise CedaAuthError(msg) from None
        if code == 429 or code >= 500:
            raise _TransientError(f"HTTP {code}") from None
        raise RuntimeError(f"HTTP {code}") from None
    except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
        # `URLError.reason` names the failure (DNS, refused connection, timeout) without the URL.
        reason = getattr(error, "reason", None) or type(error).__name__
        raise _TransientError(f"network error: {reason}") from None


def _download(*, url: str, token: str | None) -> bytes | None:
    """Fetch one URL, retrying transient failures with exponential backoff.

    Args:
        url: The file's download URL, or a directory listing's URL.
        token: The CEDA access token, or `None` for an anonymous directory listing.

    Returns:
        The response body, or `None` for HTTP 404.

    Raises:
        RuntimeError: If every attempt failed transiently, or for a non-retryable HTTP status.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _request_once(url=url, token=token)
        except _TransientError as error:
            if attempt == MAX_ATTEMPTS:
                msg = f"{error} after {MAX_ATTEMPTS} attempts"
                raise RuntimeError(msg) from None
            wait = BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
            _LOG.warning("attempt %d failed (%s); retrying in %.0f s", attempt, error, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover - the loop always returns or raises.


def _fetch_file(
    *,
    dataset: DatasetType,
    station: StationMetadata,
    year: int,
    listing: dict[str, str],
    token: str,
) -> FetchOutcome:
    """Fetch one station-year file into the raw layout, unless it is on disk or not on CEDA.

    Args:
        dataset: Which dataset the file belongs to.
        station: The station's metadata.
        year: The calendar year the file covers.
        listing: The station directory's file names mapped to MD5, from `_listed_files`.
        token: The CEDA access token.

    Returns:
        What happened. A file absent from `listing` is `expected_missing` and is not requested;
        a listed file that returns HTTP 404 or fails its MD5 check is an `error`.

    Raises:
        CedaAuthError: For HTTP 401, HTTP 403, or a redirect, which ends the run.
    """
    path = _raw_path(dataset=dataset, station=station, year=year)
    outcome = FetchOutcome(dataset=dataset, src_id=station.src_id, year=year, status="error")
    if path.exists():
        return replace(outcome, status="cached", n_bytes=path.stat().st_size)
    if path.name not in listing:
        return replace(outcome, status="expected_missing", reason="not in CEDA's directory listing")
    try:
        content = _download(url=_file_url(dataset=dataset, station=station, year=year), token=token)
    except RuntimeError as error:
        if isinstance(error, CedaAuthError):
            raise
        return replace(outcome, status="error", reason=str(error))
    finally:
        time.sleep(POLITE_DELAY_SECONDS)
    if content is None:
        return replace(outcome, reason="HTTP 404 for a file CEDA's directory listing shows")
    if hashlib.md5(content).hexdigest() != listing[path.name]:
        reason = f"{len(content)} bytes whose MD5 differs from CEDA's directory listing; discarded"
        return replace(outcome, reason=reason)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.write_bytes(content)
    partial.rename(path)
    return replace(outcome, status="downloaded", n_bytes=len(content))


def _log_outcome(*, outcome: FetchOutcome) -> None:
    """Print one progress line: status, dataset, station id, year, and bytes. No path or URL."""
    _LOG.info(
        "%-16s %-22s %s %d %10d bytes%s",
        outcome.status,
        outcome.dataset,
        outcome.src_id,
        outcome.year,
        outcome.n_bytes,
        f" ({outcome.reason})" if outcome.reason else "",
    )


def _fetch_dataset(*, dataset: DatasetType, years: range, token: str) -> list[FetchOutcome]:
    """Fetch every station-year file of one dataset, sequentially.

    Args:
        dataset: Which dataset to fetch.
        years: The calendar years to request for every station.
        token: The CEDA access token.

    Returns:
        One outcome per station-year.
    """
    outcomes: list[FetchOutcome] = []
    for station in _stations_for(dataset=dataset):
        listing = _listed_files(dataset=dataset, station=station)
        for year in years:
            outcome = _fetch_file(
                dataset=dataset, station=station, year=year, listing=listing, token=token
            )
            _log_outcome(outcome=outcome)
            outcomes.append(outcome)
    return outcomes


# --------------------------------------------------------------------------------------------------
# Tidying
# --------------------------------------------------------------------------------------------------


def _read_badc_csv(*, path: Path, label: str) -> pl.DataFrame:
    """Read one BADC-CSV data file's table with every column as a string and `NA` as null.

    Args:
        path: The raw file.
        label: A description of the file for error messages, carrying no station name or path.

    Returns:
        The table, all columns `String`.
    """
    table = _badc_table(text=path.read_text(encoding="utf-8"), label=label)
    return pl.read_csv(io.StringIO(table), null_values=["NA"], infer_schema_length=0)


def _typed_station_year(
    *, dataset: DatasetType, src_id: str, year: int, path: Path
) -> pl.DataFrame:
    """Read one raw file and cast the columns the tidy step keeps, strictly.

    Args:
        dataset: Which dataset the file belongs to.
        src_id: The zero-padded station id from the file's directory.
        year: The calendar year the file covers.
        path: The raw file.

    Returns:
        `src_id`, `time` (UTC), and every column `SPECS[dataset]` names, cast to its dtype.

    Raises:
        RuntimeError: If a column the tidy step relies on is missing, or if the file's own `src_id`
            column disagrees with the directory's station id.
    """
    spec = SPECS[dataset]
    label = f"{dataset} {src_id} {year}"
    raw = _read_badc_csv(path=path, label=label)
    needed = ("src_id", spec.time_column, *spec.string_columns, *spec.int_columns)
    missing = [name for name in (*needed, *spec.float_columns) if name not in raw.columns]
    if missing:
        msg = f"{label}: columns {missing} are missing from the file"
        raise RuntimeError(msg)
    casts: dict[Any, pl.DataType] = {
        **dict.fromkeys(spec.int_columns, pl.Int64()),
        **dict.fromkeys(spec.float_columns, pl.Float64()),
    }
    typed = (
        raw.select(needed + spec.float_columns)
        .with_columns(pl.col(pl.String).str.strip_chars())
        .cast(casts, strict=True)
        .with_columns(
            src_id=pl.col("src_id").cast(pl.Int64, strict=True).cast(pl.String).str.zfill(5),
            time=pl.col(spec.time_column).str.to_datetime(
                format="%Y-%m-%d %H:%M:%S", time_unit="us", time_zone="UTC", strict=True
            ),
        )
        .drop(spec.time_column)
    )
    file_ids = typed["src_id"].unique().to_list()
    if file_ids != [src_id]:
        msg = f"{label}: the file's src_id column holds {file_ids}, not the directory's {src_id}"
        raise RuntimeError(msg)
    return typed


def _counts_per_station(*, frame: pl.DataFrame) -> dict[str, int]:
    """Return the row count of each station in `frame`, keyed by `src_id`."""
    counts = frame.group_by("src_id").len().sort("src_id")
    return dict(zip(counts["src_id"].to_list(), counts["len"].to_list(), strict=True))


def _drop_off_hour_rows(*, frame: pl.DataFrame, report: TidyReport) -> pl.DataFrame:
    """Keep only rows whose `time` falls exactly on the hour, counting the rest per station."""
    on_hour = (pl.col("time").dt.minute() == 0) & (pl.col("time").dt.second() == 0)
    report.off_hour_rows_dropped_per_station = _counts_per_station(frame=frame.filter(~on_hour))
    return frame.filter(on_hour)


def _deduplicate(*, frame: pl.DataFrame, spec: DatasetSpec, report: TidyReport) -> pl.DataFrame:
    """Keep one row per (`src_id`, `time`), by a fixed rule, counting the rows removed per station.

    Where several rows share a key, the row kept is the first after sorting by, in order:

    1. the highest `version_num` (each file's header defines version 0 as the original received
       message and version 1 as the current best version, so version 1 outranks version 0);
    2. the most non-null `value_columns`;
    3. the lowest `rec_st_ind` (record state indicator);
    4. `met_domain_name`, then `id_type`, alphabetically;
    5. the row's position in the concatenated raw files, which are read in station-then-year order.

    The last criterion makes the rule a total order, so the result is deterministic.

    Args:
        frame: The concatenated, filtered rows of one dataset.
        spec: The dataset's spec.
        report: Receives the per-station count of removed rows.

    Returns:
        `frame` with one row per (`src_id`, `time`), sorted by that key.

    Raises:
        RuntimeError: If the key is still not unique afterwards.
    """
    ranked = frame.with_row_index("_row").with_columns(
        _n_values=pl.sum_horizontal(pl.col(spec.value_columns).is_not_null())
    )
    deduplicated = (
        ranked.sort(
            [
                "src_id",
                "time",
                "version_num",
                "_n_values",
                "rec_st_ind",
                "met_domain_name",
                "id_type",
                "_row",
            ],
            descending=[False, False, True, True, False, False, False, False],
            nulls_last=True,
        )
        .unique(subset=["src_id", "time"], keep="first", maintain_order=True)
        .drop("_row", "_n_values")
    )
    before = _counts_per_station(frame=frame)
    after = _counts_per_station(frame=deduplicated)
    report.duplicate_rows_removed_per_station = {
        src_id: before[src_id] - after.get(src_id, 0)
        for src_id in before
        if before[src_id] != after.get(src_id, 0)
    }
    if deduplicated.select(pl.struct("src_id", "time").is_duplicated().any()).item():
        msg = "(src_id, time) is not unique after de-duplication"
        raise RuntimeError(msg)
    return deduplicated.sort("src_id", "time")


def _read_dataset_files(
    *, dataset: DatasetType, files: list[tuple[str, int, Path]]
) -> pl.DataFrame:
    """Read and concatenate every raw file of one dataset, in the order given.

    Args:
        dataset: Which dataset the files belong to.
        files: `(src_id, year, path)` for each raw file, station-then-year order.

    Returns:
        The typed rows of every file, concatenated.

    Raises:
        RuntimeError: If `files` is empty.
    """
    if not files:
        msg = f"{dataset}: no raw files to tidy"
        raise RuntimeError(msg)
    return pl.concat(
        [
            _typed_station_year(dataset=dataset, src_id=src_id, year=year, path=path)
            for src_id, year, path in files
        ]
    )


def tidy_radiation(*, files: list[tuple[str, int, Path]]) -> tuple[pl.DataFrame, TidyReport]:
    """Build the tidy hourly radiation table from raw `uk-radiation-obs` files.

    A `uk-radiation-obs` file mixes hourly totals (`ob_hour_count` 1) with daily totals
    (`ob_hour_count` 24, timestamped 23:59), so only the hourly rows are kept, and the dropped rows
    are counted per `ob_hour_count` value. Then rows not on the hour are dropped, duplicates of
    (`src_id`, `time`) are resolved by the rule in `_deduplicate`, and `glbl_horz_ilmn` with its
    flag is dropped if `glbl_horz_ilmn` is null in every row. `time` is the end of the hour the
    total covers.

    Args:
        files: `(src_id, year, path)` for each raw file, station-then-year order.

    Returns:
        The tidy frame, sorted by (`src_id`, `time`), and the counts of removed rows.
    """
    dataset: DatasetType = "uk-radiation-obs"
    spec = SPECS[dataset]
    frame = _read_dataset_files(dataset=dataset, files=files)
    report = TidyReport(rows_read=frame.height)
    dropped = (
        frame.filter(pl.col("ob_hour_count").ne_missing(1))
        .group_by("ob_hour_count")
        .len()
        .sort("ob_hour_count")
    )
    report.rows_dropped_by_ob_hour_count = {
        str(value): count
        for value, count in zip(dropped["ob_hour_count"], dropped["len"], strict=True)
    }
    frame = frame.filter(pl.col("ob_hour_count") == 1)
    frame = _drop_off_hour_rows(frame=frame, report=report)
    frame = _deduplicate(frame=frame, spec=spec, report=report).select(spec.output_columns)
    if frame["glbl_horz_ilmn"].is_null().all():
        report.columns_dropped_as_entirely_null = ["glbl_horz_ilmn", "glbl_horz_ilmn_q"]
        frame = frame.drop(report.columns_dropped_as_entirely_null)
    report.rows_written = frame.height
    return frame, report


def _narrow_whole_float(*, frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """Cast a Float64 column to Int64 if every non-null value is a whole number, else leave it."""
    values = frame[column].drop_nulls()
    if (values == values.round(0)).all():
        return frame.cast({column: pl.Int64}, strict=True)
    return frame


def tidy_hourly_weather(*, files: list[tuple[str, int, Path]]) -> tuple[pl.DataFrame, TidyReport]:
    """Build the tidy hourly weather table from raw `uk-hourly-weather-obs` files.

    Rows not on the hour are dropped and counted per station, and duplicates of (`src_id`, `time`)
    — a station can file several message types, such as SYNOP and METAR, for one hour — are
    resolved by the rule in `_deduplicate`. `wind_speed_m_s` is derived here from `wind_speed` and
    `wind_speed_unit_id`: MIDAS unit ids 3 and 4 mean knots, multiplied by `KNOTS_TO_M_S`; ids 0
    and 1 mean metres per second, copied unchanged; any other id gives null. `drv_hr_sun_dur_q` is
    narrowed from Float64 to Int64 only if every value is whole.

    Args:
        files: `(src_id, year, path)` for each raw file, station-then-year order.

    Returns:
        The tidy frame, sorted by (`src_id`, `time`), and the counts of removed rows.
    """
    dataset: DatasetType = "uk-hourly-weather-obs"
    spec = SPECS[dataset]
    frame = _read_dataset_files(dataset=dataset, files=files)
    report = TidyReport(rows_read=frame.height)
    frame = _drop_off_hour_rows(frame=frame, report=report)
    frame = _deduplicate(frame=frame, spec=spec, report=report)
    unit = pl.col("wind_speed_unit_id")
    frame = frame.with_columns(
        wind_speed_m_s=pl.when(unit.is_in([3, 4]))
        .then(pl.col("wind_speed") * KNOTS_TO_M_S)
        .when(unit.is_in([0, 1]))
        .then(pl.col("wind_speed"))
        .otherwise(None)
    ).select(spec.output_columns)
    frame = _narrow_whole_float(frame=frame, column="drv_hr_sun_dur_q")
    report.drv_hr_sun_dur_q_dtype = str(frame.schema["drv_hr_sun_dur_q"])
    report.rows_written = frame.height
    return frame, report


TIDY_FUNCTIONS: Final[dict[DatasetType, Callable[..., tuple[pl.DataFrame, TidyReport]]]] = {
    "uk-radiation-obs": tidy_radiation,
    "uk-hourly-weather-obs": tidy_hourly_weather,
}


def _present_files(*, dataset: DatasetType, years: range) -> list[tuple[str, int, Path]]:
    """Return `(src_id, year, path)` for every raw file of one dataset already on disk."""
    return [
        (station.src_id, year, path)
        for station in _stations_for(dataset=dataset)
        for year in years
        if (path := _raw_path(dataset=dataset, station=station, year=year)).exists()
    ]


def _tidy_and_write(*, dataset: DatasetType, years: range) -> TidyReport:
    """Tidy every raw file of one dataset on disk and write its parquet atomically.

    Args:
        dataset: Which dataset to tidy.
        years: The calendar years to include.

    Returns:
        The counts of rows removed on the way.
    """
    frame, report = TIDY_FUNCTIONS[dataset](files=_present_files(dataset=dataset, years=years))
    path = PRODUCT_DIR / SPECS[dataset].tidy_filename
    partial = path.with_name(f"{path.name}.partial")
    frame.write_parquet(partial)
    partial.rename(path)
    _LOG.info("%s: wrote %d rows to %s", dataset, frame.height, path.name)
    return report


# --------------------------------------------------------------------------------------------------
# README and lineage
# --------------------------------------------------------------------------------------------------

_COLUMN_MEANINGS: Final[dict[str, str]] = {
    "src_id": "MIDAS station id, zero-padded to 5 characters.",
    "id_type": "MIDAS identifier type of the reporting message, e.g. `DCNN` or `WMO`.",
    "met_domain_name": "MIDAS message type the row came from, e.g. `SYNOP` or `HCM`.",
    "rec_st_ind": "MIDAS record state indicator (see the state-indicator page under Further "
    "reading).",
    "ob_hour_count": "Hours the observation period covers; always 1 in this file.",
    "glbl_irad_amt": "Global horizontal irradiation over the hour ending at `time`, kJ/m2.",
    "difu_irad_amt": "Diffuse horizontal irradiation over the hour ending at `time`, kJ/m2.",
    "direct_irad": "Direct (beam) irradiation over the hour ending at `time`, kJ/m2.",
    "irad_bal_amt": "Radiation balance (net irradiation) over the hour ending at `time`, kJ/m2.",
    "glbl_s_lat_irad_amt": "MIDAS 'mean global S latitude radiation' over the hour ending at "
    "`time`, kJ/m2, as delivered.",
    "glbl_horz_ilmn": "MIDAS 'global horizontal illumination' over the hour ending at `time`, "
    "in the kJ/m2 the file header gives.",
    "wind_speed_unit_id": "MIDAS unit code for `wind_speed`: 0 is estimated and 1 is from an "
    "anemometer, both in metres per second; 3 is estimated and 4 is from an anemometer, both in "
    "knots.",
    "wind_direction": "Mean wind direction, degrees from true north, over the 10 minutes from 20 "
    "to 10 minutes before `time`.",
    "wind_speed": "Mean wind speed over the 10 minutes from 20 to 10 minutes before `time`, in "
    "the unit `wind_speed_unit_id` gives.",
    "wind_speed_m_s": "Derived by this script: `wind_speed` in metres per second (knots x "
    f"{KNOTS_TO_M_S}); null where `wind_speed_unit_id` is not 0, 1, 3, or 4.",
    "q10mnt_mxgst_spd": "Maximum gust speed over the 10 minutes from 20 to 10 minutes before "
    "`time`, knots.",
    "air_temperature": "Air temperature at `time`, degC.",
    "dewpoint": "Dew-point temperature at `time`, degC.",
    "rltv_hum": "Relative humidity at `time`, percent.",
    "msl_pressure": "Mean-sea-level pressure at `time`, hPa.",
    "cld_ttl_amt_id": "Total cloud amount code at `time`: 0 to 8 are oktas, and 9 means the sky "
    "is obscured.",
    "wmo_hr_sun_dur": "MIDAS 'WMO hour sunshine duration' for the hour labelled `time`, hours.",
    "drv_hr_sun_dur": "MIDAS 'derived hour sunshine' for the hour labelled `time`, hours.",
    "cs_hr_sun_dur": "MIDAS 'Campbell-Stokes hour sunshine duration' for the hour labelled "
    "`time`, hours.",
}
"""What each tidy column means. Every fact about the data itself (dtype, null count) is added from
the written frame by `_column_descriptions`."""


def _column_meaning(*, name: str) -> str:
    """Return the meaning of one tidy column, including the `_q` flag columns and `time`."""
    if name.endswith("_q"):
        return (
            f"MIDAS quality-control flag for `{name.removesuffix('_q')}`, as delivered; the digits "
            "are Met Office codes, not interpreted here."
        )
    if name == "time":
        return "UTC, timezone-aware; see Timestamp convention and units below."
    return _COLUMN_MEANINGS[name]


def _column_descriptions(*, frame: pl.DataFrame, filename: str) -> dict[str, str]:
    """Return one README column line per tidy column, with dtype and null count read off `frame`."""
    null_counts = frame.null_count().row(0, named=True)
    return {
        f"{filename}: {name}": (
            f"{_column_meaning(name=name)} Dtype `{dtype}`; "
            f"{null_counts[name]:,} nulls in {frame.height:,} rows."
        )
        for name, dtype in frame.schema.items()
    }


def _file_inventory(*, dataset: DatasetType, years: range) -> list[FetchOutcome]:
    """Classify every station-year file of one dataset by what is on disk and on CEDA.

    Reads each station's anonymous CEDA directory listing but downloads no data file. A file on
    disk counts as `cached`; an absent file counts as `expected_missing` if CEDA's listing lacks it
    too, and as `error` if CEDA lists it, the same rule `_fetch_file` applies.

    Args:
        dataset: Which dataset to inventory.
        years: The calendar years requested.

    Returns:
        One outcome per station-year.
    """
    inventory: list[FetchOutcome] = []
    for station in _stations_for(dataset=dataset):
        listing = _listed_files(dataset=dataset, station=station)
        for year in years:
            path = _raw_path(dataset=dataset, station=station, year=year)
            outcome = FetchOutcome(
                dataset=dataset, src_id=station.src_id, year=year, status="error"
            )
            if path.exists():
                outcome = replace(outcome, status="cached", n_bytes=path.stat().st_size)
            elif path.name in listing:
                outcome = replace(outcome, reason="listed on CEDA but absent on disk")
            else:
                outcome = replace(
                    outcome, status="expected_missing", reason="not in CEDA's directory listing"
                )
            inventory.append(outcome)
    return inventory


def _coverage_table(
    *, dataset: DatasetType, frame: pl.DataFrame, inventory: list[FetchOutcome]
) -> str:
    """Return the Markdown station-coverage table for one dataset, from the frame and inventory.

    Args:
        dataset: Which dataset the table describes.
        frame: The dataset's tidy frame.
        inventory: Every station-year of the dataset, classified by `_file_inventory`.

    Returns:
        A Markdown table with one row per station id.
    """
    spec = SPECS[dataset]
    per_station = (
        frame.group_by("src_id")
        .agg(
            pl.col("time").min().alias("first_time"),
            pl.col("time").max().alias("last_time"),
            pl.len().alias("rows"),
            pl.col("met_domain_name").unique().sort().str.join(", ").alias("message_types"),
            *(
                pl.col(name).is_not_null().mean().mul(100).alias(name)
                for name in spec.coverage_columns
            ),
        )
        .sort("src_id")
    )
    by_station = {row["src_id"]: row for row in per_station.to_dicts()}
    header = (
        "| src_id | station-years fetched | years not on CEDA "
        "| years listed on CEDA but not fetched | first `time` | last `time` | rows "
        "| % of hours in fetched years with a row "
        "| message types | "
        + " | ".join(f"% `{name}` non-null" for name in spec.coverage_columns)
        + " |"
    )
    divider = "|" + "---|" * (9 + len(spec.coverage_columns))
    lines = [header, divider]
    for src_id in STATION_IDS[dataset]:
        station_files = [outcome for outcome in inventory if outcome.src_id == src_id]
        fetched = sorted(o.year for o in station_files if o.status in {"cached", "downloaded"})
        expected = sorted(o.year for o in station_files if o.status == "expected_missing")
        errors = sorted(o.year for o in station_files if o.status == "error")
        row = by_station.get(src_id)
        cells = [
            src_id,
            str(len(fetched)),
            ", ".join(map(str, expected)) or "-",
            ", ".join(map(str, errors)) or "-",
            f"{row['first_time']:%Y-%m-%d %H:%M}" if row else "-",
            f"{row['last_time']:%Y-%m-%d %H:%M}" if row else "-",
            f"{row['rows']:,}" if row else "0",
            f"{100 * row['rows'] / sum(24 * (365 + isleap(y)) for y in fetched):.1f}"
            if row
            else "-",
            row["message_types"] if row else "-",
            *(f"{row[name]:.1f}" if row else "-" for name in spec.coverage_columns),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _extra_readme_sections(
    *, frames: dict[DatasetType, pl.DataFrame], inventories: dict[DatasetType, list[FetchOutcome]]
) -> str:
    """Return the `Station coverage` and `Timestamp convention and units` README sections."""
    coverage = "\n\n".join(
        f"### `{dataset}` (`{SPECS[dataset].tidy_filename}`)\n\n"
        + _coverage_table(dataset=dataset, frame=frame, inventory=inventories[dataset])
        for dataset, frame in frames.items()
    )
    return f"""
## Station coverage

**Which stations were fetched.** The station lists were typed by hand and the script did not
record the rule behind them. The rule below reproduces them from the station-metadata CSVs: a
station is fetched if its `last_year` is 2025 or later and it lies within 100 km of at least one of
the nine anonymised study sites. That rule gives exactly the 10 of 241 `uk-radiation-obs` stations
and 37 of the 38 `uk-hourly-weather-obs` stations. The exception is station 24219, whose record
ends in 2024 but which was fetched as well. Every other station with `last_year` 2025 lies further
than 100 km from every site. This README gives no per-station distances, because a distance to a
study site would help locate that site.

Computed from the tidy parquets and the raw files on disk when this README was written. The
station-metadata CSVs do not say which variables a station reports or how often, so the last
columns (the share of the tidy file's rows where the variable is non-null) and the share of hours
with a row are the only record of that.

{coverage}

## Timestamp convention and units

- **Radiation:** `time` is the END of the observation hour (MIDAS `ob_end_time`). The row with
  `time` T holds the total irradiation over the hour ending at T, in kJ/m2. The hour's mean
  irradiance in W/m2 is the kJ/m2 total divided by 3.6.
- **Hourly weather:** `time` is an instant (MIDAS `ob_time`); a reading describes the moment of
  the label. The Met Office Surface Data Users Guide gives the wind speed, wind direction and
  maximum gust as covering HH-20 to HH-10: the 10 minutes ending 10 minutes before `time`. The
  file header calls each sunshine duration an "hour" duration without saying whether the hour ends
  or starts at `time`.
- **Both:** every `time` is UTC and stored timezone-aware; no naive timestamp is written.
"""


def _nan_count(*, frame: pl.DataFrame) -> int:
    """Return the total NaN count across every float column of `frame`."""
    floats = [name for name, dtype in frame.schema.items() if dtype.is_float()]
    if not floats:
        return 0
    return int(frame.select(pl.sum_horizontal(pl.col(floats).is_nan().sum())).item())


def _write_docs(
    *,
    years: range,
    tidy_reports: dict[str, dict[str, Any]],
    errors_this_run: list[FetchOutcome],
    retrieved_at_utc: str | None,
) -> Path:
    """Write `lineage.json` and `README.md` for every dataset whose tidy parquet exists.

    Args:
        years: The calendar years requested.
        tidy_reports: Per dataset, the `TidyReport` fields of its most recent tidy run.
        errors_this_run: Failed fetches from this run, with their reasons; empty in
            `--readme-only` mode.
        retrieved_at_utc: The earlier fetch's timestamp to keep, in `--readme-only` mode; `None`
            to stamp the current time.

    Returns:
        The README path.

    Raises:
        RuntimeError: If no tidy parquet exists yet.
    """
    frames: dict[DatasetType, pl.DataFrame] = {
        dataset: pl.read_parquet(PRODUCT_DIR / SPECS[dataset].tidy_filename)
        for dataset in DATASETS
        if (PRODUCT_DIR / SPECS[dataset].tidy_filename).exists()
    }
    if not frames:
        msg = "no tidy parquet exists yet; run a fetch first"
        raise RuntimeError(msg)
    inventories = {dataset: _file_inventory(dataset=dataset, years=years) for dataset in frames}
    _write_lineage(
        years=years,
        frames=frames,
        inventories=inventories,
        tidy_reports=tidy_reports,
        errors_this_run=errors_this_run,
        retrieved_at_utc=retrieved_at_utc,
    )
    columns: dict[str, str] = {}
    for dataset, frame in frames.items():
        columns |= _column_descriptions(frame=frame, filename=SPECS[dataset].tidy_filename)
    nan_counts = {
        SPECS[dataset].tidy_filename: _nan_count(frame=frame) for dataset, frame in frames.items()
    }
    readme = write_readme(
        product_dir=PRODUCT_DIR,
        product_name=f"Met Office MIDAS Open ({DATASET_VERSION}, {QC_VERSION})",
        source_web_page=USER_GUIDE_URL,
        script_path=SCRIPT_PATH,
        columns=columns,
        missing_value_convention=(
            "A missing reading is a Polars null (MIDAS writes `NA`); null counts per column are "
            "listed under Columns. NaN counts across the float columns: "
            + ", ".join(f"`{name}` {count:,}" for name, count in nan_counts.items())
            + ". A station-year with no file on CEDA is simply absent: see Station coverage."
        ),
        gotchas=[
            (
                "`uk-radiation-obs` files mix hourly totals (`ob_hour_count` 1) with daily "
                "totals (`ob_hour_count` 24, timestamped 23:59) in one table; the tidy file keeps "
                "only the hourly rows, and `lineage.json` counts the dropped rows per "
                "`ob_hour_count` value."
            ),
            (
                "`wind_speed` is delivered in knots unless `wind_speed_unit_id` says otherwise; "
                "`wind_speed_m_s` is derived by this script, not delivered by MIDAS."
            ),
            (
                "`uk-hourly-weather-obs` files can hold several message types for one hour; the "
                "tidy step keeps one row per (`src_id`, `time`) by the rule in the script's "
                "`_deduplicate` and counts the removed rows per station in `lineage.json`."
            ),
            (
                "Not every `uk-hourly-weather-obs` station reports hourly: a station whose only "
                "message type in Station coverage is `DLY3208` (a once-daily climate return) has "
                "one row per day at 09:00. Those rows have a null `wind_speed_unit_id`, and so a "
                "null `wind_speed_m_s`; their `wind_speed` values sit on Beaufort-force midpoints "
                "in knots, which suggests the wind is estimated, not measured."
            ),
            (
                "The station-metadata CSVs do not say which variables a station reports; see "
                "Station coverage for the measured share of non-null values per station."
            ),
            (
                "The `_q` quality-control flag digits are Met Office codes (see the MIDAS "
                f"quality-control page, <{QC_FLAGS_URL}>); they are kept as delivered, not "
                "interpreted, and no row is filtered on them."
            ),
            *VALIDATION_GOTCHAS,
        ],
        external_docs={
            "MIDAS Open user guide (CEDA)": USER_GUIDE_URL,
            "MIDAS quality-control flags (`_q` and `_j`)": QC_FLAGS_URL,
            "MIDAS record state indicators (`rec_st_ind`)": (
                "https://dap.ceda.ac.uk/badc/ukmo-midas/metadata/doc/state_indicators.html"
            ),
            LICENCE: LICENCE_URL,
        },
        lineage_filenames=["lineage.json"],
    )
    with readme.open("a", encoding="utf-8") as handle:
        handle.write(_extra_readme_sections(frames=frames, inventories=inventories))
    _LOG.info("wrote %s and lineage.json", readme.name)
    return readme


def _write_lineage(
    *,
    years: range,
    frames: dict[DatasetType, pl.DataFrame],
    inventories: dict[DatasetType, list[FetchOutcome]],
    tidy_reports: dict[str, dict[str, Any]],
    errors_this_run: list[FetchOutcome],
    retrieved_at_utc: str | None,
) -> None:
    """Write `lineage.json`, carrying no coordinate, station name, URL, or token.

    Args:
        years: The calendar years requested.
        frames: The tidy frame of every dataset being documented.
        inventories: Every station-year of each dataset, classified by `_file_inventory`.
        tidy_reports: Per dataset, the `TidyReport` fields of its most recent tidy run.
        errors_this_run: Failed fetches from this run, with their reasons.
        retrieved_at_utc: The earlier fetch's timestamp to keep; `None` to stamp the current time.
    """

    def _listed(outcomes: list[FetchOutcome], status: FetchStatusType) -> list[dict[str, Any]]:
        return [
            {"dataset": o.dataset, "src_id": o.src_id, "year": o.year, "reason": o.reason}
            for o in outcomes
            if o.status == status
        ]

    files = {
        dataset: {
            "fetched": sum(o.status in {"cached", "downloaded"} for o in inventory),
            "expected_missing": sum(o.status == "expected_missing" for o in inventory),
            "error": sum(o.status == "error" for o in inventory),
            "raw_bytes": sum(o.n_bytes for o in inventory),
            "expected_missing_files": _listed(inventory, "expected_missing"),
            "error_files": _listed(inventory, "error"),
        }
        for dataset, inventory in inventories.items()
    }
    extra: dict[str, Any] = {
        "datasets": list(frames),
        "dataset_version": DATASET_VERSION,
        "qc_version": QC_VERSION,
        "years": [years.start, years.stop - 1],
        "station_ids": {dataset: list(STATION_IDS[dataset]) for dataset in frames},
        "files": files,
        "total_raw_bytes": sum(o.n_bytes for inventory in inventories.values() for o in inventory),
        "errors_this_run": [asdict(outcome) for outcome in errors_this_run],
        "tidy": {dataset: tidy_reports.get(dataset) for dataset in frames},
        "tidy_row_counts": {dataset: frame.height for dataset, frame in frames.items()},
        "licence": LICENCE,
        "licence_url": LICENCE_URL,
        "redownload_command": f"uv run python {SCRIPT_PATH}",
    }
    if retrieved_at_utc is not None:
        extra["retrieved_at_utc"] = retrieved_at_utc
    write_lineage_note(
        product_dir=PRODUCT_DIR,
        source_address=SOURCE_ADDRESS,
        request_description=(
            f"MIDAS Open {', '.join(frames)}, {DATASET_VERSION}, {QC_VERSION}, one BADC-CSV per "
            f"station per calendar year {years.start}-{years.stop - 1}"
        ),
        variables=list(
            dict.fromkeys(
                name
                for frame in frames.values()
                for name in frame.columns
                if name not in {"src_id", "time"}
            )
        ),
        extra=extra,
    )


def _previous_lineage() -> dict[str, Any]:
    """Return the existing `lineage.json`, or an empty dict if there is none yet."""
    path = PRODUCT_DIR / "lineage.json"
    return json.loads(path.read_text()) if path.exists() else {}


# --------------------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------------------


def _run_sample(*, datasets: tuple[DatasetType, ...], token: str) -> int:
    """Fetch one file per dataset and print its column names with their non-null counts.

    Args:
        datasets: The datasets to sample.
        token: The CEDA access token.

    Returns:
        The process exit code.
    """
    for dataset in datasets:
        station = _stations_for(dataset=dataset)[0]
        outcome = _fetch_file(
            dataset=dataset,
            station=station,
            year=SAMPLE_YEAR,
            listing=_listed_files(dataset=dataset, station=station),
            token=token,
        )
        _log_outcome(outcome=outcome)
        if outcome.status not in {"cached", "downloaded"}:
            return 1
        frame = _read_badc_csv(
            path=_raw_path(dataset=dataset, station=station, year=SAMPLE_YEAR),
            label=f"{dataset} {station.src_id} {SAMPLE_YEAR}",
        )
        print(f"{dataset} {station.src_id} {SAMPLE_YEAR}: {frame.height} rows")
        for name, count in zip(frame.columns, frame.count().row(0), strict=True):
            print(f"  {name:<24} {count:>6} non-null")
    return 0


def _parse_arguments() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--start-year", type=int, default=DEFAULT_FIRST_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_LAST_YEAR)
    parser.add_argument("--sample", action="store_true", help="fetch one file per dataset only")
    parser.add_argument(
        "--readme-only", action="store_true", help="rewrite README.md and lineage.json only"
    )
    return parser.parse_args()


def main() -> int:
    """Fetch, tidy, and document MIDAS Open radiation and hourly weather observations."""
    arguments = _parse_arguments()
    datasets: tuple[DatasetType, ...] = (arguments.dataset,) if arguments.dataset else DATASETS
    years = range(arguments.start_year, arguments.end_year + 1)
    for dataset in datasets:
        _stations_for(dataset=dataset)  # Fails before any request if a station id is unknown.

    previous = _previous_lineage()
    tidy_reports: dict[str, dict[str, Any]] = dict(previous.get("tidy") or {})
    if arguments.readme_only:
        _write_docs(
            years=years,
            tidy_reports=tidy_reports,
            errors_this_run=[],
            retrieved_at_utc=previous.get("retrieved_at_utc"),
        )
        return 0

    token = _ceda_token()
    if arguments.sample:
        return _run_sample(datasets=datasets, token=token)

    outcomes: list[FetchOutcome] = []
    try:
        for dataset in datasets:
            outcomes += _fetch_dataset(dataset=dataset, years=years, token=token)
    except CedaAuthError as error:
        _LOG.error("%s", error)
        return 2
    for dataset in datasets:
        tidy_reports[dataset] = asdict(_tidy_and_write(dataset=dataset, years=years))
    errors = [outcome for outcome in outcomes if outcome.status == "error"]
    _write_docs(
        years=years, tidy_reports=tidy_reports, errors_this_run=errors, retrieved_at_utc=None
    )
    for status in ("downloaded", "cached", "expected_missing", "error"):
        _LOG.info("%s: %d files", status, sum(o.status == status for o in outcomes))
    if errors:
        _LOG.error("%d files failed:", len(errors))
        for outcome in errors:
            _LOG.error(
                "  %s %s %d: %s", outcome.dataset, outcome.src_id, outcome.year, outcome.reason
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
