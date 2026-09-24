"""Download Open-Meteo's Previous Runs API at every solar and wind meter's own coordinates.

One-off throwaway script for the forecast study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>. The Historical Forecast
archives `fetch_open_meteo_grid.py` and `studies/beam_diffuse_split/fetch_open_meteo_point.py`
already downloaded only hold each model's most-recent-run value at each hour (lead 0-3h), so they
cannot score a forecast at a genuine lead time. The Previous Runs API
(<https://open-meteo.com/en/docs/previous-runs-api>) serves the same hour from several different
model runs side by side: `<variable>` is the freshest run covering that hour (lead 0), and
`<variable>_previous_dayN` for `N` in 1-7 is the run made `N` x 24 hours earlier — so
`shortwave_radiation_previous_day3` at a given `time` is what a forecast issued three days earlier
predicted for that same hour. That is exactly the lead-time axis the forecast study needs and the
Historical Forecast archives do not carry.

**Sites, not the trial-area grid.** This script builds the same anonymised meter roster
`studies/beam_diffuse_split/build_dataset._pv_sites`/`_wind_sites` does (`_roster` below is a
deliberate duplicate — see its own docstring for why) and follows the same request/response shape
as `studies/beam_diffuse_split/fetch_open_meteo_point.py`, rather than the trial-area grid
`fetch_open_meteo_grid.py` uses: a forecast study scores a specific generator's forecast, not a
grid cell's. No coordinate or `time_series_id` reaches the written frame or any log line — rows
are keyed by the anonymised `site` label the roster assigns, exactly as the sibling per-site
scripts do.

**Every model's own `models=` identifier for this API was read from the Previous Runs docs page
itself** (the checkbox `id` attributes server-rendered into
<https://open-meteo.com/en/docs/previous-runs-api>), not guessed and not carried over from
`sources.OPEN_METEO_MODELS`, because the Previous Runs API's own identifiers differ from the
Historical Forecast API's for the same model: ICON is `dwd_icon_d2`/`dwd_icon_eu`/`dwd_icon_global`
here, not the bare `icon_d2`/`icon_eu`/`icon_global` `fetch_open_meteo_point.py` requests; GFS is
`ncep_gfs_seamless`, not `gfs_seamless`; ECMWF's 9 km HRES is `ecmwf_ifs`, not `ecmwf_ifs04`. See
`MODELS` below for the full registry and which candidates from the issue were skipped and why.

**Checkpointed per (model, year)**, following the `data-download` skill: each model's own request
is split into whole calendar years (the coarsest chunk that still lets a crash mid-run lose at most
one year of one model), written atomically to `_year_cache/<year>.parquet` under each product's
`previous_runs/` directory as soon as it is fetched, and combined by reading back whatever is
cached — so re-running the same command after a crash or a daily-quota refusal resumes rather than
re-fetching. `_first_year_with_data` measures each model's own archive start with a one-month probe
before committing to the multi-year backfill, per the skill's "measure one chunk first" rule.

Run it with `uv run python studies/weather_downloads/fetch_open_meteo_previous_runs.py --model ukv`,
or with no `--model` to fetch every registered model in turn.
"""

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from lineage import write_lineage_note, write_readme
from paths import REPO_DATA_DIR, WEATHER_DOWNLOADS_DIR, open_meteo_api_key
from studies.anonymise import (
    LABEL_PERMUTATION_SEED,
    SITE_LABELS,
    WIND_LABEL_PERMUTATION_SEED,
    WIND_SITE_LABELS,
    site_labels_for,
)
from studies.solar import cos_zenith_hour_mean, extraterrestrial_horizontal, zenith

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_open_meteo_previous_runs")

PREVIOUS_RUNS_URL: Final[str] = (
    "https://customer-previous-runs-api.open-meteo.com/v1/forecast"
    if open_meteo_api_key()
    else "https://previous-runs-api.open-meteo.com/v1/forecast"
)
"""The commercial host once `OPEN_METEO_API_KEY` is set (see `paths.open_meteo_api_key`), which
lifts the free tier's daily/hourly/minutely rate limits entirely; the free host otherwise."""
REQUEST_TIMEOUT_SECONDS: Final[float] = 600.0
MAX_ATTEMPTS: Final[int] = 5
REQUEST_SLEEP_SECONDS: Final[float] = 2.0
"""Paced between requests, since a previous-runs request carries many more columns per call than
the single-lead point fetches (`BASE_VARIABLES` x 8 leads), which weighs more against Open-Meteo's
call budget than the row count alone suggests."""

LEAD_DAYS: Final[tuple[int, ...]] = tuple(range(8))
"""0 (the freshest run covering the hour) through 7 (the run made a week earlier)."""

BASE_VARIABLES: Final[tuple[str, ...]] = (
    "shortwave_radiation",
    "direct_radiation",
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_direction_100m",
)
"""Open-Meteo's normalised variable names, before the `_previous_dayN` suffix. `direct_radiation`
rather than `diffuse_radiation`, matching `studies/beam_diffuse_split/fetch_open_meteo_point.py`'s
choice, so a later join onto that study's arms uses the same flux. `wind_speed_100m` /
`wind_direction_100m` are Open-Meteo's own rescaling to 100 m for a model whose native upper level
sits elsewhere (120 m for the ICON family — see `fetch_wind_point.py`'s docstring for the measured
0.98 scale factor), not a per-model hub-height chosen here."""

FIRST_DATE: Final[str] = "2024-01-01"
"""Earliest date the Previous Runs API is documented to serve for most models.

`_first_year_with_data` measures each model's own start against this rather than assuming every
model reaches back this far."""


def _last_date() -> str:
    """Return yesterday's date as `YYYY-MM-DD`, in UTC.

    The Previous Runs API's freshest (day-0) column needs the current run to have completed, so
    "yesterday" rather than "today" avoids a partially-populated final day.
    """
    from datetime import UTC, datetime, timedelta

    return (datetime.now(UTC).date() - timedelta(days=1)).isoformat()


POWER_DELTA_URI: Final[str] = str(REPO_DATA_DIR / "NGED" / "power_time_series.delta")
METADATA_PATH: Final[Path] = REPO_DATA_DIR / "NGED" / "metadata.parquet"
CAPACITY_DELTA_URI: Final[str] = str(REPO_DATA_DIR / "effective_capacity")
MIN_YEARS_OF_READINGS: Final[float] = 1.0
"""Matches `studies/beam_diffuse_split/build_dataset.py`'s own constants of the same name. The
roster reader below (`_roster`/`_pv_sites`/`_wind_sites`) is a deliberate duplicate of that
module's functions of the same name, not an import: `paths.py`'s own `_main_checkout` docstring
records the same choice for `sources.py`, so that this directory stays self-contained the way
every other study directory is, rather than depending on a sibling study's own internal layout.
The anonymisation itself — the label permutation and the minimum-history filter — is not
duplicated: both copies call the shared `studies.anonymise.site_labels_for`, so the labelling
cannot drift between the two copies."""


def _roster(*, time_series_type: str, labels: tuple[str, ...], seed: int) -> pl.DataFrame:
    """Return the series of one technology with enough history, labelled anonymously.

    Args:
        time_series_type: The metadata's technology, such as `PV` or `Wind`.
        labels: The anonymous labels to shuffle onto the series.
        seed: The seed paired with those labels.

    Returns:
        One row per site with `site`, `latitude`, and `longitude`.
    """
    metadata = pl.read_parquet(METADATA_PATH).filter(pl.col("time_series_type") == time_series_type)
    row_counts = (
        pl.scan_delta(POWER_DELTA_URI)
        .group_by("time_series_id")
        .agg(pl.len().alias("n_rows"))
        .collect()
    )
    min_rows = int(MIN_YEARS_OF_READINGS * 365.25 * 48)
    sites = (
        metadata.select("time_series_id", "latitude", "longitude")
        .join(row_counts, on="time_series_id", how="inner")
        .filter(pl.col("n_rows") >= min_rows)
        .sort("time_series_id")
    )
    mapping = site_labels_for(
        eligible_ids=sites["time_series_id"].to_list(), labels=labels, seed=seed
    )
    return sites.with_columns(
        site=pl.col("time_series_id").replace_strict(mapping, return_dtype=pl.Utf8)
    ).select("site", "latitude", "longitude")


def _pv_sites() -> pl.DataFrame:
    """Return the usable PV sites, labelled `A`-`F`, carrying `site`, `latitude`, `longitude`."""
    return _roster(time_series_type="PV", labels=SITE_LABELS, seed=LABEL_PERMUTATION_SEED)


def _wind_sites() -> pl.DataFrame:
    """Return the usable wind sites, labelled `W1`-`W3`, carrying `site`/`latitude`/`longitude`."""
    return _roster(
        time_series_type="Wind", labels=WIND_SITE_LABELS, seed=WIND_LABEL_PERMUTATION_SEED
    )


@dataclass(frozen=True)
class PreviousRunsModel:
    """One model the Previous Runs API serves, and where its output is written."""

    output_dir: str
    models_parameter: str
    label: str
    docs_url: str


MODELS: Final[dict[str, PreviousRunsModel]] = {
    "ukv": PreviousRunsModel(
        output_dir="UKV",
        models_parameter="ukmo_uk_deterministic_2km",
        label="UK Met Office UKV (2 km deterministic)",
        docs_url="https://www.metoffice.gov.uk/research/weather/numerical-modelling/models",
    ),
    "ecmwf-ifs-hres": PreviousRunsModel(
        output_dir="ECMWF-IFS-HRES",
        models_parameter="ecmwf_ifs",
        label="ECMWF IFS HRES 9 km",
        docs_url="https://www.ecmwf.int/en/forecasts/documentation-and-support",
    ),
    "ecmwf-ifs025": PreviousRunsModel(
        output_dir="ECMWF-IFS-025",
        models_parameter="ecmwf_ifs025",
        label="ECMWF IFS 0.25 degree",
        docs_url="https://www.ecmwf.int/en/forecasts/documentation-and-support",
    ),
    "icon-eu": PreviousRunsModel(
        output_dir="ICON-EU",
        models_parameter="dwd_icon_eu",
        label="DWD ICON-EU (6.5 km)",
        docs_url="https://www.dwd.de/EN/research/weatherforecasting/num_modelling/01_num_weather_prediction_modells/icon_eu_en.html",
    ),
    "icon-d2": PreviousRunsModel(
        output_dir="ICON-D2",
        models_parameter="dwd_icon_d2",
        label="DWD ICON-D2 (2 km)",
        docs_url="https://www.dwd.de/EN/research/weatherforecasting/num_modelling/01_num_weather_prediction_modells/icon_d2_en.html",
    ),
    "icon-global": PreviousRunsModel(
        output_dir="ICON-GLOBAL",
        models_parameter="dwd_icon_global",
        label="DWD ICON global (13 km)",
        docs_url="https://www.dwd.de/EN/research/weatherforecasting/num_modelling/01_num_weather_prediction_modells/icon_global_en.html",
    ),
    "arpege-europe": PreviousRunsModel(
        output_dir="ARPEGE-EUROPE",
        models_parameter="meteofrance_arpege_europe",
        label="Meteo-France ARPEGE Europe",
        docs_url="https://meteofrance.com/en/weather-forecast-and-services/weather-data",
    ),
    "arome-france": PreviousRunsModel(
        output_dir="AROME-FRANCE",
        models_parameter="meteofrance_arome_france",
        label="Meteo-France AROME France",
        docs_url="https://meteofrance.com/en/weather-forecast-and-services/weather-data",
    ),
    "dmi-harmonie-arome": PreviousRunsModel(
        output_dir="DMI-HARMONIE-AROME",
        models_parameter="dmi_harmonie_arome_europe",
        label="DMI HARMONIE-AROME (Europe domain)",
        docs_url="https://opendatadocs.dmi.googleapis.com/",
    ),
    "knmi-harmonie-arome": PreviousRunsModel(
        output_dir="KNMI-HARMONIE-AROME",
        models_parameter="knmi_harmonie_arome_europe",
        label="KNMI HARMONIE-AROME (Europe domain)",
        docs_url="https://english.knmidata.nl/",
    ),
    "gfs": PreviousRunsModel(
        output_dir="GFS-SEAMLESS",
        models_parameter="ncep_gfs_seamless",
        label="NCEP GFS Seamless",
        docs_url="https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php",
    ),
}
"""Every model requested for issue #810's Previous Runs download.

**AROME France's own domain is centred on France and may not reach every NGED generator** — it is
requested anyway, per the issue's "at minimum, try" list, and `_check_domain_coverage` raises if a
site comes back entirely null so the gap is caught rather than silently written as a missing site.
If it turns out not to cover any generator, delete its `MODELS` entry and record why here.

**Skipped, and why:** `ecmwf_aifs025_single` (a machine-learning model, not the physics-based
family the issue asks for), the Canadian GEM family and Japan's JMA/Korea's KMA family (no Great
Britain domain), MeteoSwiss and GeoSphere Austria (Alpine-region domains only), `ukmo_seamless` and
`ukmo_global_deterministic_10km` (the issue asks for UKV's own 2 km deterministic run, already
covered by `ukv` above), `bom_access_global`/`cma_grapes_global` (Australia/China, no reason to
expect better GB skill than GFS/ECMWF already registered). Confirmed against the Previous Runs
docs page's own model checkboxes (`https://open-meteo.com/en/docs/previous-runs-api`), not against
the general-purpose `/v1/forecast` API's model list, because the two APIs' `models=` enums differ.
"""


def _requested_variables() -> tuple[str, ...]:
    """Return every `hourly=` variable name: each base variable at every lead in `LEAD_DAYS`."""
    names: list[str] = []
    for base in BASE_VARIABLES:
        names.append(base)
        names.extend(f"{base}_previous_day{lead}" for lead in range(1, 8))
    return tuple(names)


def _get_json(*, url: str) -> Any:
    """Fetch one URL, retrying a transport failure but never an API refusal.

    Copied from `studies/beam_diffuse_split/fetch_open_meteo_point.py`'s `_get_json`: see that
    function's docstring for why a rate-limit refusal must not be retried, and why the refusal body
    is read only for its `reason` and never echoed whole (it can quote a request parameter back,
    and the coordinates in this request are meter locations).
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as refusal:
            reason = json.loads(refusal.read() or b"{}").get("reason", "no reason given")
            msg = f"Open-Meteo refused the request with HTTP {refusal.code}: {reason}"
            raise RuntimeError(msg) from refusal
        except Exception:  # noqa: BLE001
            _LOG.warning("attempt %d failed, retrying", attempt + 1)
            time.sleep(5.0 * (attempt + 1))
        else:
            if isinstance(payload, dict) and payload.get("error"):
                msg = f"Open-Meteo refused the request: {payload.get('reason')}"
                raise RuntimeError(msg)
            return payload
    msg = f"Open-Meteo failed after {MAX_ATTEMPTS} attempts"
    raise RuntimeError(msg)


def fetch_previous_runs_frame(
    *,
    sites: pl.DataFrame,
    models_parameter: str,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Fetch one date range of every lead at every site's own coordinates, in a single request.

    Every site goes in one request, the same batching `fetch_open_meteo_point.fetch_point_frame`
    uses. **Nulls are kept, not dropped**: near the archive's own start, `previous_day6`/`day7`
    genuinely have no run to report yet, and a row missing only a deep lead still carries a real
    day-0 reading that a blanket `drop_nulls()` (as the single-lead point fetcher uses) would throw
    away. The "does this model cover this site" check below only requires `shortwave_radiation` (a
    lead every model serves once its archive has started) to be non-null somewhere.

    Args:
        sites: The roster, carrying `site`, `latitude`, and `longitude`.
        models_parameter: The value of the API's `models=` query parameter.
        start_date: First date to request, as `YYYY-MM-DD`.
        end_date: Last date to request, as `YYYY-MM-DD`.

    Returns:
        One row per (site, time), one column per requested variable/lead, nulls kept.

    Raises:
        RuntimeError: If the response does not carry one block per site, or a site is entirely
            null for the baseline variable (no domain coverage at that site).
    """
    variables = _requested_variables()
    url = (
        f"{PREVIOUS_RUNS_URL}"
        f"?latitude={','.join(str(value) for value in sites['latitude'])}"
        f"&longitude={','.join(str(value) for value in sites['longitude'])}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={','.join(variables)}"
        f"&models={models_parameter}&timezone=UTC"
    )
    api_key = open_meteo_api_key()
    if api_key:
        url += f"&apikey={api_key}"
    payload = _get_json(url=url)
    blocks = payload if isinstance(payload, list) else [payload]
    if len(blocks) != sites.height:
        msg = f"asked for {sites.height} sites and got {len(blocks)} blocks"
        raise RuntimeError(msg)

    def _block_frame(site: str, block: dict[str, Any]) -> pl.DataFrame:
        hourly = block["hourly"]
        n_rows = len(hourly["time"])
        columns = {name: hourly.get(name, [None] * n_rows) for name in variables}
        return pl.DataFrame(
            {"site": site, "time": hourly["time"]} | columns,
            schema_overrides=dict.fromkeys(variables, pl.Float64),
        )

    frame = pl.concat(
        _block_frame(site, block) for site, block in zip(sites["site"], blocks, strict=True)
    ).with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M").dt.replace_time_zone("UTC"))

    coverage = frame.group_by("site").agg(pl.col("shortwave_radiation").is_not_null().any())
    uncovered = coverage.filter(~pl.col("shortwave_radiation"))["site"].to_list()
    if uncovered:
        msg = f"{models_parameter} returned only nulls for {sorted(uncovered)}; may not cover them"
        raise RuntimeError(msg)
    return frame


def _year_cache_dir(*, output_dir: Path) -> Path:
    """Return the per-year checkpoint directory for one model's download."""
    return output_dir / "_year_cache"


def _first_year_with_data(*, models_parameter: str, sites: pl.DataFrame) -> int:
    """Measure the first calendar year this model actually serves, with a one-month probe.

    Per the `data-download` skill's "measure one chunk before committing to the rest": a start date
    earlier than the model's real archive start returns an all-null month rather than an HTTP error,
    so probing `FIRST_DATE`'s own month first (rather than assuming every model reaches back to it)
    is what `_first_date_of` in `fetch_open_meteo_point.py` establishes by yearly probing; this does
    the same at one-month grain since the Previous Runs archive is documented to start in 2024
    for most models rather than spanning several years like the Historical Forecast archive.

    **A model can start mid-year, not just mid-month.** `ecmwf_ifs025`'s archive starts
    2024-02-03: the one-month probe below finds nothing in January, but the year still has 11
    months of real data. `fetch_previous_runs_frame` itself raises the moment a site comes back
    all-null, before this function gets to inspect anything, so the one-month probe is wrapped in
    `try`/`except` and a whole-year probe — the same call weight `_fetch_model_checkpointed`
    spends on year 1 for real once this returns — is only made if the cheap one-month probe
    raised that same "returned only nulls" error.

    Args:
        models_parameter: The value of the API's `models=` query parameter.
        sites: The roster to probe with.

    Returns:
        The first year (as an int) with a non-null `shortwave_radiation` reading.

    Raises:
        RuntimeError: If no data is found anywhere in `FIRST_DATE`'s year.
    """
    try:
        fetch_previous_runs_frame(
            sites=sites,
            models_parameter=models_parameter,
            start_date=FIRST_DATE,
            end_date=f"{FIRST_DATE[:4]}-01-31",
        )
    except RuntimeError:
        fetch_previous_runs_frame(
            sites=sites,
            models_parameter=models_parameter,
            start_date=FIRST_DATE,
            end_date=f"{FIRST_DATE[:4]}-12-31",
        )
        # The call above raises `RuntimeError` itself if the whole year is also all-null, which
        # is the genuine "no data" case and propagates with its own message. Reaching this line
        # means the year has real data even though January did not.
        _LOG.info(
            "%s: January %s is all-null, but the rest of %s has data",
            models_parameter,
            FIRST_DATE[:4],
            FIRST_DATE[:4],
        )
    return int(FIRST_DATE[:4])


def _fetch_model_checkpointed(
    *, model: PreviousRunsModel, sites: pl.DataFrame, output_dir: Path
) -> pl.DataFrame:
    """Fetch every year of one model's Previous Runs data, checkpointing per year.

    Args:
        model: The registry entry to fetch.
        sites: The roster (solar and wind sites combined) to request.
        output_dir: `<WEATHER_DOWNLOADS_DIR>/<model.output_dir>/previous_runs/`.

    Returns:
        The combined frame, read back from the per-year cache rather than accumulated in memory
        during the loop, so a second run after a crash finishes the job rather than redoing it.
    """
    cache_dir = _year_cache_dir(output_dir=output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    first_year = _first_year_with_data(models_parameter=model.models_parameter, sites=sites)
    last_year = int(_last_date()[:4])
    for year in range(first_year, last_year + 1):
        year_path = cache_dir / f"{year}.parquet"
        # The final year's own end date is "yesterday", which moves forward on every calendar day
        # this script is resumed on (the exact scenario a daily-quota refusal creates) — so a cache
        # hit on that one year is only really complete if the cached rows already reach yesterday's
        # date. Every earlier year's end date is fixed at its own 31 December and a cache hit there
        # is unconditionally final.
        if year_path.exists():
            if year != last_year:
                _LOG.info("%s %d: already cached, skipping", model.output_dir, year)
                continue
            cached_max_date = (
                pl.scan_parquet(year_path).select(pl.col("time").max()).collect().item()
            )
            if cached_max_date is not None and str(cached_max_date.date()) >= _last_date():
                _LOG.info(
                    "%s %d: already cached through %s, skipping",
                    model.output_dir,
                    year,
                    _last_date(),
                )
                continue
            _LOG.info(
                "%s %d: cached only through %s, refetching to extend to %s",
                model.output_dir,
                year,
                cached_max_date,
                _last_date(),
            )
        start = FIRST_DATE if year == first_year else f"{year}-01-01"
        end = _last_date() if year == last_year else f"{year}-12-31"
        frame = fetch_previous_runs_frame(
            sites=sites, models_parameter=model.models_parameter, start_date=start, end_date=end
        )
        partial_path = year_path.with_suffix(".partial.parquet")
        frame.write_parquet(partial_path)
        partial_path.rename(year_path)
        _LOG.info("%s %d: wrote %d rows to %s", model.output_dir, year, frame.height, year_path)
        time.sleep(REQUEST_SLEEP_SECONDS)

    # A narrow glob matching only a bare `<year>.parquet` filename, never a `<year>.partial.parquet`
    # a crash could leave behind between `write_parquet` and the rename above (or an old attempt for
    # a model since dropped from `--model`): `*.parquet` would match both and double-count that
    # year's rows into the combined frame.
    year_paths = sorted(cache_dir.glob("[0-9][0-9][0-9][0-9].parquet"))
    return pl.concat(pl.read_parquet(path) for path in year_paths).sort("site", "time")


def _check_lead_differs(*, frame: pl.DataFrame) -> dict[str, float]:
    """Measure, per lead, how often `shortwave_radiation` exactly matches the day-0 value.

    The guard against a lead-mislabelling bug that would show every lead as an identical copy of
    day 0. Restricted to rows where the day-0 reading exceeds 10 W/m^2, because a night-time zero is
    legitimately identical across every lead and would otherwise mask a real bug.

    Args:
        frame: The combined frame for one model, carrying every `shortwave_radiation_previous_dayN`
            column.

    Returns:
        One fraction per lead (`"day1"`..`"day7"`), for the lineage note.
    """
    daytime = frame.filter(pl.col("shortwave_radiation") > 10.0)
    fractions: dict[str, float] = {}
    for lead in range(1, 8):
        column = f"shortwave_radiation_previous_day{lead}"
        comparable = daytime.filter(pl.col(column).is_not_null())
        if comparable.is_empty():
            fractions[f"day{lead}"] = float("nan")
            continue
        identical = comparable.filter(pl.col(column) == pl.col("shortwave_radiation")).height
        fractions[f"day{lead}"] = identical / comparable.height
    return fractions


def _check_timestamp_convention(*, frame: pl.DataFrame, sites: pl.DataFrame) -> dict[str, float]:
    """Measure whether `shortwave_radiation` (day 0) tracks hour-ending or hour-starting better.

    Also checks that it is non-zero only while the sun is up. Both are measured from the fetched
    data itself rather than assumed to carry over from the Historical Forecast API's convention.
    For each site, solar geometry is evaluated at three candidate midpoints — the label itself (an
    instantaneous snapshot rather than an hourly mean), the label shifted 30 minutes earlier (the
    hour-*ending* mid-point `fetch_open_meteo_point._solar_geometry` uses), and the label shifted 30
    minutes later (the hour-*starting* mid-point) — and the correlation of the clear-sky envelope
    (`extraterrestrial_horizontal`) against the served value is compared across all three. Testing
    only the label and the earlier shift would leave the later shift unmeasured, so a served value
    that is genuinely period-starting could be reported as "hour STARTING" from the label-offset
    candidate alone even though that candidate is an instantaneous reading, not the true
    period-starting midpoint. **Reported, not asserted**: whichever candidate correlates best in the
    fetched data is what the lineage note and README record.

    Args:
        frame: The combined frame, carrying `site`, `time`, `shortwave_radiation`.
        sites: The roster, carrying `site`, `latitude`, `longitude`.

    Returns:
        `corr_at_label`, `corr_shifted_30min_earlier`, `corr_shifted_30min_later`, and
        `frac_daytime_radiation_at_night`.
    """
    coordinates = {
        str(row["site"]): (float(row["latitude"]), float(row["longitude"]))
        for row in sites.to_dicts()
    }
    at_label: list[np.ndarray] = []
    shifted_earlier: list[np.ndarray] = []
    shifted_later: list[np.ndarray] = []
    served: list[np.ndarray] = []
    night_violation = 0
    total = 0
    for (site,), rows in frame.sort("site", "time").group_by(["site"], maintain_order=True):
        latitude, longitude = coordinates[str(site)]
        stamps = rows["time"]
        values = rows["shortwave_radiation"].fill_null(float("nan")).to_numpy()
        zenith_at_label = zenith(stamps=stamps, latitude=latitude, longitude=longitude)
        zenith_earlier = zenith(
            stamps=stamps.dt.offset_by("-30m"), latitude=latitude, longitude=longitude
        )
        zenith_later = zenith(
            stamps=stamps.dt.offset_by("30m"), latitude=latitude, longitude=longitude
        )
        at_label.append(extraterrestrial_horizontal(stamps=stamps, zenith_deg=zenith_at_label))
        shifted_earlier.append(
            extraterrestrial_horizontal(stamps=stamps, zenith_deg=zenith_earlier)
        )
        shifted_later.append(extraterrestrial_horizontal(stamps=stamps, zenith_deg=zenith_later))
        served.append(values)
        mask = ~np.isnan(values)
        night_violation += int(
            np.sum(
                mask
                & (values > 10.0)
                & (
                    cos_zenith_hour_mean(stamps=stamps, latitude=latitude, longitude=longitude)
                    <= 0.0
                )
            )
        )
        total += int(np.sum(mask))

    served_all = np.concatenate(served)
    valid = ~np.isnan(served_all)

    def _corr(candidate: list[np.ndarray]) -> float:
        return float(np.corrcoef(np.concatenate(candidate)[valid], served_all[valid])[0, 1])

    return {
        "corr_at_label": _corr(at_label),
        "corr_shifted_30min_earlier": _corr(shifted_earlier),
        "corr_shifted_30min_later": _corr(shifted_later),
        "frac_daytime_radiation_at_night": night_violation / total if total else float("nan"),
    }


def _check_native_step(*, frame: pl.DataFrame) -> dict[str, float]:
    """Measure, per lead, how often adjacent daytime hours carry an exactly repeated value.

    A coarser-than-hourly native step (ECMWF's 3-hourly output beyond a point, for instance)
    repeats `shortwave_radiation` across several consecutive hourly labels, where a genuine hourly
    mean almost never does.

    Args:
        frame: The combined frame for one model.

    Returns:
        One fraction per lead (`"day0"`..`"day7"`).
    """
    fractions: dict[str, float] = {}
    for lead in range(8):
        column = "shortwave_radiation" if lead == 0 else f"shortwave_radiation_previous_day{lead}"
        repeats = 0
        pairs = 0
        for _, rows in frame.sort("site", "time").group_by(["site"], maintain_order=True):
            values = rows[column].to_numpy()
            daytime = rows["shortwave_radiation"].to_numpy() > 10.0
            mask = daytime[1:] & daytime[:-1] & ~np.isnan(values[1:]) & ~np.isnan(values[:-1])
            repeats += int(np.sum(mask & (values[1:] == values[:-1])))
            pairs += int(np.sum(mask))
        fractions[f"day{lead}"] = repeats / pairs if pairs else float("nan")
    return fractions


def _column_description(*, name: str) -> str:
    """Return a one-line description (quantity, unit, lead) for one requested column name."""
    if name.startswith("shortwave"):
        quantity = "Global horizontal irradiance"
    elif name.startswith("direct"):
        quantity = "Direct (beam) horizontal irradiance"
    elif name.startswith("temperature"):
        quantity = "Air temperature at 2 m"
    elif name.startswith("wind_speed"):
        quantity = "Wind speed"
    else:
        quantity = "Wind direction"

    if "radiation" in name:
        unit = "W/m^2"
    elif "temperature" in name:
        unit = "degC"
    elif "speed" in name:
        unit = "km/h"
    else:
        unit = "degrees"

    if name in BASE_VARIABLES:
        lead = ""
    else:
        day = name.rsplit("_previous_day", 1)[1]
        lead = (
            f", from the model run made {day} x 24h before the hour it values (day 0, the bare "
            "column name, is the freshest run)."
        )
    return f"{quantity}, {unit}{lead}"


def _write_docs_for_model(
    *, frame: pl.DataFrame, model: PreviousRunsModel, sites: pl.DataFrame, output_dir: Path
) -> None:
    """Write `lineage.json` and `README.md` for one already-fetched model, from measured data only.

    Every number quoted in the README comes from `frame` itself — the lead-differencing fractions,
    the timestamp-convention correlations, and the native-step fractions are all computed here,
    never hand-typed, so a later re-run cannot leave the docs describing a stale fetch.
    """
    lead_diff = _check_lead_differs(frame=frame)
    timestamp = _check_timestamp_convention(frame=frame, sites=sites)
    native_step = _check_native_step(frame=frame)
    null_counts = frame.select(_requested_variables()).null_count().row(0, named=True)
    null_fraction = {
        name: count / frame.height if frame.height else float("nan")
        for name, count in null_counts.items()
    }
    best_candidate = max(
        ("corr_at_label", "corr_shifted_30min_earlier", "corr_shifted_30min_later"),
        key=lambda name: timestamp[name],
    )
    # A standalone label for a sentence ("best fit: hour ENDING") and an adjectival phrase for a
    # sentence that already supplies its own noun ("hour-ending convention") — built separately so
    # neither reads with a doubled "hour" (e.g. "hour ending hour") or a doubled "convention".
    variant_label, variant_adjective = {
        "corr_at_label": (
            "an instantaneous snapshot at the label (matches neither hourly-mean convention well)",
            "instantaneous-snapshot",
        ),
        "corr_shifted_30min_earlier": ("hour ENDING", "hour-ending"),
        "corr_shifted_30min_later": ("hour STARTING", "hour-starting"),
    }[best_candidate]

    write_lineage_note(
        product_dir=output_dir,
        source_address=PREVIOUS_RUNS_URL,
        request_description=(
            f"{model.label}, models={model.models_parameter}, hourly="
            f"{','.join(BASE_VARIABLES)} x previous_day0-7, {sites.height} meter sites, "
            f"{FIRST_DATE} to {_last_date()}"
        ),
        variables=list(_requested_variables()),
        extra={
            "n_sites": sites.height,
            "row_count": frame.height,
            "date_range_fetched": [FIRST_DATE, _last_date()],
            "null_fraction_per_column": null_fraction,
            "note": (
                "Lead-differencing check (fraction of daytime rows where shortwave_radiation at "
                "each lead exactly equals day 0 — near-zero is expected, near-1.0 would indicate a "
                f"lead-mislabelling bug): {lead_diff}. Timestamp-convention check (clear-sky "
                "correlation against the label itself vs. the label shifted 30 minutes earlier, "
                "and the fraction of daytime-threshold radiation reported while the sun is below "
                f"the horizon): {timestamp} — best fit: {variant_label}. Native-step "
                "check (fraction of adjacent daytime hours with an exactly repeated "
                "shortwave_radiation value, per lead — near-zero means a genuine hourly mean, "
                "materially above zero means the served hourly value is held constant across a "
                f"coarser native step): {native_step}."
            ),
        },
    )
    write_readme(
        product_dir=output_dir,
        product_name=f"{model.label} — Previous Runs",
        source_web_page="https://open-meteo.com/en/docs/previous-runs-api",
        script_path="studies/weather_downloads/fetch_open_meteo_previous_runs.py",
        lineage_filenames=["lineage.json"],
        columns={
            "site": (
                "Anonymised meter label (`A`-`F` for solar, `W1`-`W3` for wind) — not a "
                "coordinate or a `time_series_id`."
            ),
            "time": (
                "UTC, timezone-aware. See `lineage.json`'s `note` field for the measured "
                f"timestamp convention ({variant_adjective}, from the clear-sky check below)."
            ),
            **{name: _column_description(name=name) for name in _requested_variables()},
        },
        missing_value_convention=(
            "Polars null. Null fraction per column in this fetch is recorded in "
            "`lineage.json`'s `null_fraction_per_column` field. A deep lead (previous_day6/7) "
            "is null near the start of the fetched window, before that many days of archive "
            "existed yet — expected, not a defect."
        ),
        gotchas=[
            (
                "**previous_dayN is not the same as an N-day-ahead forecast product**: it is the "
                "value the model predicted for this hour from the run made N x 24h earlier, so "
                "for an hour early in a run's cycle the effective lead is close to N x 24h, and "
                "for an hour late in the cycle it is closer to N x 24h plus the run's own "
                "forecast length into that cycle. See `lineage.json`'s `note` for the measured "
                "lead-differencing fractions confirming each lead's column genuinely differs from "
                "day 0."
            ),
            (
                f"Timestamp convention measured as {variant_adjective} from the clear-sky "
                "check in `lineage.json`'s `note` field — confirm this still holds before "
                "joining against power readings with a different convention."
            ),
            (
                "wind_speed_100m/wind_direction_100m is Open-Meteo's own rescaling for a model "
                "whose native upper level is not 100 m (120 m for the ICON family, scaled by "
                "0.98 per `studies/beam_diffuse_split/fetch_wind_point.py`'s measurement) — not "
                "independently re-measured here."
            ),
        ],
        external_docs={
            f"{model.label} technical documentation": model.docs_url,
            "Open-Meteo Previous Runs API": "https://open-meteo.com/en/docs/previous-runs-api",
        },
    )


def main() -> int:
    """Download every registered model's Previous Runs data at every meter site, and write it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), default=None)
    arguments = parser.parse_args()
    models = [MODELS[arguments.model]] if arguments.model else list(MODELS.values())

    sites = pl.concat([_pv_sites(), _wind_sites()], how="diagonal_relaxed").select(
        "site", "latitude", "longitude"
    )
    _LOG.info("fetching %d models for %d sites", len(models), sites.height)

    for model in models:
        output_dir = WEATHER_DOWNLOADS_DIR / model.output_dir / "previous_runs"
        output_dir.mkdir(parents=True, exist_ok=True)
        frame = _fetch_model_checkpointed(model=model, sites=sites, output_dir=output_dir)
        combined_path = output_dir / "combined.parquet"
        frame.write_parquet(combined_path)
        _LOG.info(
            "%s: wrote %d rows covering %s to %s for %d sites to %s",
            model.output_dir,
            frame.height,
            frame["time"].min(),
            frame["time"].max(),
            frame["site"].n_unique(),
            combined_path,
        )
        _write_docs_for_model(frame=frame, model=model, sites=sites, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
