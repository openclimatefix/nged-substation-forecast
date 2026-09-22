"""Which irradiance sources the experiment runs on, and what each Open-Meteo model serves.

One-off throwaway module for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784> and its UKV extension in
<https://github.com/openclimatefix/nged-substation-forecast/issues/800>.

`SOURCE_CHOICES` is the single copy of the source list, imported by every `argparse` parser that
offers `--source`. A second Open-Meteo model still needs its own entries in `SourceType`,
`PER_SITE_SOURCES`, and `build_dataset`, and its own labels in the two figure scripts; what the
registry saves is the download itself, which needs no new code at all.
"""

import os
from pathlib import Path
from typing import Final, Literal, NamedTuple

from contracts.settings import PROJECT_ROOT

SourceType = Literal["cds", "open-meteo", "cams", "ukv", "icon-d2", "icon-eu", "icon-global"]
"""Which irradiance download to build from.

`open-meteo` is the reanalysis route the experiment runs on, because it serves the same fields in
about a minute where the Copernicus archive takes most of a night. `cds` is that Copernicus archive,
and it is the reference the mirror is checked against rather than a second result:
`verify_era5_sources.py` compares the two over every hour both cover, and a run of it is what
licenses reading an `open-meteo` result as an ERA5 result.

`cams` is a different instrument rather than a second route to the same one. The CAMS radiation
service infers cloud from Meteosat at around 5 km and publishes the global, beam and diffuse
horizontal irradiances at each meter's own coordinates, where ERA5 averages its cloud field over
roughly 31 km and lands the meter in a grid cell up to 17 km away. Running the same arms on both
separates "the split carries no information" from "ERA5's grid has already smoothed the beam away".

`ukv` is the Met Office's 2 km deterministic UK model, taken from Open-Meteo's historical-forecast
archive. UKV separates resolution from delivery: ERA5 against UKV is a resolution contrast inside
one product class, whereas ERA5 against CAMS crosses from a reanalysis to a satellite retrieval as
well as from 31 km to 5 km.

`icon-d2` is the German weather service's 2 km model, also from Open-Meteo's historical-forecast
archive. A second model at the same grid spacing from a different forecasting centre is what turns
a single 2 km result into a resolution claim: if the published split helps on both, the finding is
about grid spacing rather than about one centre's radiation scheme. Its own radiation output is
accumulated where UKV's is instantaneous, so Open-Meteo reaches the hourly column by
de-accumulation rather than by reconstruction.

`icon-eu` and `icon-global` are the same German modelling system on wider domains: about 7 km over
Europe and about 11 km worldwide. ICON-D2's domain stops around 2.5°W, so it excludes South West
England and South Wales, and a forecast for the whole of Great Britain has to use one of these two.

Each of `cams`, `ukv`, `icon-d2`, `icon-eu`, and `icon-global` takes its air temperature from the
Open-Meteo ERA5 frame, because the temperature feature is shared by every arm and a source must
differ from ERA5 only in its irradiance columns.
"""

SOURCE_CHOICES: Final[tuple[SourceType, ...]] = (
    "cds",
    "open-meteo",
    "cams",
    "ukv",
    "icon-d2",
    "icon-eu",
    "icon-global",
)
"""Every source name, as `argparse` `choices` for the scripts that take `--source`."""

PER_SITE_SOURCES: Final[tuple[SourceType, ...]] = (
    "cams",
    "ukv",
    "icon-d2",
    "icon-eu",
    "icon-global",
)
"""Sources downloaded at each meter's own coordinates rather than on the ERA5 grid.

A build from one of these still reads the gridded ERA5 frame, for the air temperature every arm
shares, and then replaces only the two irradiance columns.
"""

NativeRadiationType = Literal["instantaneous", "accumulated", "unmeasured"]
"""What a model's own output holds before Open-Meteo converts it to an hourly value.

An `instantaneous` model publishes a snapshot at the step, which Open-Meteo divides by the ratio of
the instantaneous to the hour-mean cosine of the solar zenith angle to reach a backward-looking
hourly mean. An `accumulated` model publishes a running total since model start, which Open-Meteo
de-accumulates into a genuine hourly mean with no solar geometry involved. The distinction decides
whether the hourly value carries a sub-hourly approximation, so `_instant` is worth requesting
alongside the default only for an `instantaneous` model.

`unmeasured` means nobody has checked this model's own output, and a registry entry carrying it must
not be trained on until someone has.
"""


PointTemporalType = Literal["hourly", "instant"]
"""Which pair of served columns a per-site Open-Meteo build feeds the arms.

`hourly` is the default column, a backward-looking mean over the hour ending at the label. That is
the same temporal object as ERA5's hourly integral, as the CAMS hourly integration, and as the
period-ending hourly mean of metered power the experiment predicts, which is why it is the primary.
`instant` is the snapshot at the label, half an hour later than that window's centre, and exists so
the sensitivity can be measured rather than argued about.
"""


ICON_WIDE_DOMAIN_ARCHIVE_STARTS: Final[str] = "2022-11-23"
"""The first day ICON-EU and ICON global carry real values in Open-Meteo's archive.

**The archive accepts earlier dates than it can serve.** Requests back to 2016-01-01 return HTTP 200
with `shortwave_radiation` null throughout, and bisecting the dates put the first day with values
at 2022-11-23 for both models. The fetcher drops null rows, so a start date taken from the range the
API accepts would ask for six years of nothing rather than fail.

Both models are `accumulated` for the same reason ICON-D2 is: Open-Meteo reads every ICON domain
through one downloader, `DownloadIconCommand`, which de-averages a field according to its GRIB
step type, and DWD publishes the surface radiation of every domain as an average since the run
started.
"""


class OpenMeteoModel(NamedTuple):
    """One model the historical-forecast archive serves, and what it is known to hold.

    Attributes:
        source: The `--source` name, which also names the output directory and filename.
        models_parameter: The value of the API's `models=` query parameter.
        archive_starts: The first date the archive claims to serve, as `YYYY-MM-DD`.
        live_ingest_starts: The date Open-Meteo's own downloader for this model first existed, as
            `YYYY-MM-DD`, or `None` where nobody has established it. Everything in the archive
            before this date was backfilled from a source Open-Meteo does not name, so it is a
            different product until measurement says otherwise — see `verify_ukv_lineage.py`.
        native_radiation: What the upstream model publishes, before Open-Meteo's conversion.
    """

    source: SourceType
    models_parameter: str
    archive_starts: str
    live_ingest_starts: str | None
    native_radiation: NativeRadiationType


OPEN_METEO_MODELS: Final[dict[str, OpenMeteoModel]] = {
    "ukv": OpenMeteoModel(
        source="ukv",
        models_parameter="ukmo_uk_deterministic_2km",
        archive_starts="2022-03-01",
        live_ingest_starts="2024-08-12",
        native_radiation="instantaneous",
    ),
    "icon-d2": OpenMeteoModel(
        source="icon-d2",
        models_parameter="icon_d2",
        archive_starts="2022-12-01",
        live_ingest_starts=None,
        native_radiation="accumulated",
    ),
    "icon-eu": OpenMeteoModel(
        source="icon-eu",
        models_parameter="icon_eu",
        archive_starts=ICON_WIDE_DOMAIN_ARCHIVE_STARTS,
        live_ingest_starts=None,
        native_radiation="accumulated",
    ),
    "icon-global": OpenMeteoModel(
        source="icon-global",
        models_parameter="icon_global",
        archive_starts=ICON_WIDE_DOMAIN_ARCHIVE_STARTS,
        live_ingest_starts=None,
        native_radiation="accumulated",
    ),
}
"""Every model `fetch_open_meteo_point.py` can download, keyed by its `--model` name.

Adding a model means adding an entry and measuring what goes in it. `native_radiation` in
particular is a claim about the upstream model, and `unmeasured` is the honest value until somebody
has read the upstream documentation or the downloader that converts it.
"""


def _main_checkout(root: Path) -> Path:
    """Return the repository's main working tree, given any working tree's root.

    A linked worktree carries `uv.lock` of its own, so `contracts.settings.PROJECT_ROOT` is the
    worktree's own root, and every worktree would otherwise get an empty `data/` of its own. The
    downloads under `data/` run to tens of gigabytes and are shared by every branch, so a
    per-worktree copy would mean re-fetching the lot. Git marks a linked worktree by making `.git` a
    file holding `gitdir: <main>/.git/worktrees/<name>`, which names the main checkout two levels
    up.

    Args:
        root: A working tree's root directory.

    Returns:
        The main working tree's root, or `root` unchanged when it is already the main one.
    """
    marker = root / ".git"
    if not marker.is_file():
        return root
    pointer = marker.read_text().removeprefix("gitdir:").strip()
    if not pointer:
        return root
    git_dir = Path(pointer)
    if git_dir.parent.name != "worktrees":
        return root
    return git_dir.parent.parent.parent


REPO_DATA_DIR: Final[Path] = Path(
    os.environ.get("DATA_PATH_INTERNAL") or _main_checkout(PROJECT_ROOT) / "data"
)
"""Where every download and every built frame lands.

Run from the main checkout, the same directory `contracts.Settings.data_path_internal` names: the
`DATA_PATH_INTERNAL` environment variable if set, otherwise `data/` under the directory holding
`uv.lock`. Only the environment is read, where `Settings` also reads the workspace `.env`, so a
path configured solely in `.env` has to be exported before running any script here.

**Run from a linked worktree, the default still resolves to the main checkout's `data/`**, which
is what `_main_checkout` is for. Every branch shares one copy of the downloads rather than
re-fetching tens of gigabytes per worktree.

**A remote URI is not supported here, where `Settings` allows one.** These scripts read and write
through `pathlib`, which mangles `s3://` into `s3:/`, so a workspace configured against object
storage has to point this variable at a local directory instead.
"""


STUDIES_DATA_DIR: Final[Path] = REPO_DATA_DIR / "studies"
"""Where every study's inputs and outputs live, apart from the pipeline's own tables.

Under `data/studies/` rather than beside the pipeline's own tables, so that a weather product a
study alone fetches cannot be mistaken for one the Dagster asset graph ingests. `data/NWP/` holds
what production ingests; `data/studies/weather/ICON-D2/` holds what a study fetched to answer one
question. `data/NGED/` stays outside it: the pipeline's own power and metadata tables, which a study
reads and does not own.
"""

WEATHER_DATA_DIR: Final[Path] = STUDIES_DATA_DIR / "weather"
"""Where downloaded weather lands, one subdirectory per product (`ERA5`, `CAMS`, `ENS`, `UKV`,
`ICON-D2`).

Kept apart from any one study's outputs because a download is an input a later study can reuse, and
some take most of a night to fetch again.
"""

ANM_DATA_DIR: Final[Path] = STUDIES_DATA_DIR / "anm"
"""Where NGED's active network management setpoint exports are filed, one CSV per `time_series_id`,
beside the export-cap parquet `anm_setpoints.py` derives from each.
"""

STUDY_DATA_DIR: Final[Path] = STUDIES_DATA_DIR / "beam_diffuse_split"
"""Where everything this study builds from its inputs lives: the joined datasets, each arm's
results, and the figures.
"""


def point_output_path_for(*, source: SourceType) -> Path:
    """Return where one per-site download is written.

    The fetcher writes this path and `build_dataset` reads it, so both call this function rather
    than spelling the path twice and finding out they disagree only when a build comes up empty.

    Args:
        source: Which source's download to locate.

    Returns:
        The parquet path holding that source's per-site fluxes.
    """
    return WEATHER_DATA_DIR / source.upper() / f"beam_diffuse_{source}.parquet"


HISTORICAL_FORECAST_URL: Final[str] = "https://historical-forecast-api.open-meteo.com/v1/forecast"
"""Where a forecast model's archive is served from.

A different service from the `archive-api` endpoint `fetch_era5_open_meteo.py` uses for ERA5, with
its own call-weight accounting, though the JSON `hourly` block has the same shape.
"""
