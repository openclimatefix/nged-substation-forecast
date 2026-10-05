"""Which irradiance sources the experiment runs on, and what each Open-Meteo model serves.

Written for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784> and its UKV extension in
<https://github.com/openclimatefix/nged-substation-forecast/issues/800>. Scripts in
`studies/beam_diffuse_split/`, `studies/nwp_forecast_comparison/`,
`studies/open_meteo_ensemble_means/`, `studies/past_weather/`, and `studies/weather_downloads/`
import it.

`SOURCE_CHOICES` is the single copy of the source list, imported by every `argparse` parser that
offers `--source`. A further Open-Meteo model needs its own entries in `SourceType`,
`SOURCE_CHOICES`, `PER_SITE_SOURCES`, and `OPEN_METEO_MODELS`, and its own labels in the two figure
scripts before its results are charted. The download and the dataset build need no new code.
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, NamedTuple

from contracts.settings import PROJECT_ROOT

SourceType = Literal[
    "cds",
    "open-meteo",
    "cams",
    "ukv",
    "icon-d2",
    "icon-eu",
    "icon-global",
    "sarah-3",
    "icon-dream-eu",
    "ecmwf-ifs-hres",
    "arpege-europe",
    "dmi-harmonie-arome",
    "knmi-harmonie-arome",
]
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

`icon-eu` and `icon-global` are the same German modelling system on wider domains: 6.5 km over
Europe and 13 km worldwide, which Open-Meteo serves on grids of about 7 km and 11 km. ICON-D2's
domain stops around 2.5°W, so it excludes South West England and South Wales, and an ICON forecast
for the whole of Great Britain has to use one of these two. All three ICON domains are
`accumulated` upstream: Open-Meteo reads every ICON domain through
one downloader, `DownloadIconCommand`, which de-averages a field according to its GRIB step type,
and DWD publishes the surface radiation of every domain as an average since the run started.

`sarah-3` is EUMETSAT's SARAH-3 satellite retrieval from CM SAF, a second satellite product beside
CAMS, on a 0.05° latitude-longitude grid. `icon-dream-eu` is the German weather service's ICON-DREAM
reanalysis over Europe, a second reanalysis beside ERA5, at about 6.5 km. Both are read from gridded
downloads at each site's nearest cell by `extract_site_series.py`, which also turns their native
time steps into hourly means.

`ecmwf-ifs-hres`, `arpege-europe`, `dmi-harmonie-arome`, and `knmi-harmonie-arome` are four more
forecast models from Open-Meteo's historical-forecast archive, fetched at each site's own
coordinates by `fetch_open_meteo_point.py` like UKV and the ICON products: ECMWF's 9 km global
model, Météo-France's global model on its European grid, and the HARMONIE-AROME models the Danish
and the Dutch weather services run over Europe.

Each of the per-site sources takes its air temperature from the Open-Meteo ERA5 frame, because the
temperature feature is shared by every arm and a source must differ from ERA5 only in its irradiance
columns.
"""

SOURCE_CHOICES: Final[tuple[SourceType, ...]] = (
    "cds",
    "open-meteo",
    "cams",
    "ukv",
    "icon-d2",
    "icon-eu",
    "icon-global",
    "sarah-3",
    "icon-dream-eu",
    "ecmwf-ifs-hres",
    "arpege-europe",
    "dmi-harmonie-arome",
    "knmi-harmonie-arome",
)
"""Every source name, as `argparse` `choices` for the scripts that take `--source`."""

PER_SITE_SOURCES: Final[tuple[SourceType, ...]] = (
    "cams",
    "ukv",
    "icon-d2",
    "icon-eu",
    "icon-global",
    "sarah-3",
    "icon-dream-eu",
    "ecmwf-ifs-hres",
    "arpege-europe",
    "dmi-harmonie-arome",
    "knmi-harmonie-arome",
)
"""Sources read at each meter's own coordinates, or its nearest cell, rather than on the ERA5 grid.

A build from one of these still reads the gridded ERA5 frame, for the air temperature every arm
shares, and then replaces only the two irradiance columns.
"""

EXTRACTED_SOURCES: Final[tuple[SourceType, ...]] = ("sarah-3", "icon-dream-eu")
"""Per-site sources `extract_site_series.py` cuts from a gridded download, rather than fetched."""

UNSCORED_EXTRACTED_SPLITS: Final[frozenset[SourceType]] = frozenset({"sarah-3"})
"""Extracted sources whose direct flux is modelled from their own global flux, so no arm reads it.

CM SAF computes SARAH-3's direct flux (SID) from its global flux (SIS) with a diffuse-fraction
model, and the whole record fails `check_direct_is_not_a_separation_model`: the direct fraction's
median spread within a bin is 0.041, against a threshold of 0.05. The fetched models carry the same
judgement as `OpenMeteoModel.split_scored`.
"""

NativeRadiationType = Literal["instantaneous", "accumulated", "unmeasured"]
"""What a model's own output holds before Open-Meteo converts it to an hourly value.

An `instantaneous` model publishes a snapshot at the step, which Open-Meteo divides by the ratio of
the instantaneous to the hour-mean cosine of the solar zenith angle to reach a backward-looking
hourly mean. An `accumulated` model publishes a running total or average since model start, which
Open-Meteo de-accumulates into a genuine hourly mean with no solar geometry involved. The
distinction decides whether the hourly value carries a sub-hourly approximation, so `_instant` is
worth requesting alongside the default only for an `instantaneous` model.

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


class OpenMeteoModel(NamedTuple):
    """One model the historical-forecast archive serves, and what it is known to hold.

    Attributes:
        source: The `--source` name, which also names the output directory and filename.
        models_parameter: The value of the API's `models=` query parameter.
        archive_starts: The first date the archive serves real values for, as `YYYY-MM-DD`. The
            API accepts earlier dates and answers them with nulls, so this is found by probing
            rather than read from the range the API accepts.
        live_ingest_starts: The date Open-Meteo's own downloader for this model first existed, as
            `YYYY-MM-DD`, or `None` where nobody has established it. Everything in the archive
            before this date was backfilled from a source Open-Meteo does not name, so it is a
            different product until measurement says otherwise — see `verify_ukv_lineage.py`.
        native_radiation: What the upstream model publishes, before Open-Meteo's conversion.
        split_scored: Whether the served direct flux is the model's own, so a split arm may read
            it. `False` where the served direct flux is defective or is a separation model's
            output, which `fetch_open_meteo_point.py` then reports on rather than raising.
    """

    source: SourceType
    models_parameter: str
    archive_starts: str
    live_ingest_starts: str | None
    native_radiation: NativeRadiationType
    split_scored: bool = True


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
        archive_starts="2022-11-23",
        live_ingest_starts=None,
        native_radiation="accumulated",
    ),
    "icon-global": OpenMeteoModel(
        source="icon-global",
        models_parameter="icon_global",
        archive_starts="2022-11-17",
        live_ingest_starts=None,
        native_radiation="accumulated",
    ),
    "ecmwf-ifs-hres": OpenMeteoModel(
        source="ecmwf-ifs-hres",
        models_parameter="ecmwf_ifs",
        archive_starts="2017-01-01",
        live_ingest_starts=None,
        native_radiation="accumulated",
    ),
    "arpege-europe": OpenMeteoModel(
        source="arpege-europe",
        models_parameter="meteofrance_arpege_europe",
        archive_starts="2024-01-02",
        live_ingest_starts=None,
        native_radiation="accumulated",
        split_scored=False,
    ),
    "dmi-harmonie-arome": OpenMeteoModel(
        source="dmi-harmonie-arome",
        models_parameter="dmi_harmonie_arome_europe",
        archive_starts="2024-07-01",
        live_ingest_starts=None,
        native_radiation="accumulated",
        split_scored=False,
    ),
    "knmi-harmonie-arome": OpenMeteoModel(
        source="knmi-harmonie-arome",
        models_parameter="knmi_harmonie_arome_europe",
        archive_starts="2024-07-01",
        live_ingest_starts=None,
        native_radiation="accumulated",
        split_scored=False,
    ),
}
"""Every model `fetch_open_meteo_point.py` can download, keyed by its `--model` name.

A start date earlier than the first day with values fails the fetch on its first all-null year, with
a message blaming domain coverage; a start date later than it truncates the download silently. The
ICON-EU and ICON global starts are the first whole days with values at a probe in London: ICON-EU
from 2022-11-23 07:00 UTC, ICON global from 2022-11-16 08:00 UTC.

**ICON-EU's archive holds one corrupt block, 2023-06-21 01:00 to 06:00 UTC, at every site.** Its
values there run about three hours early, reaching 150 W m⁻² at 03:00 UTC, where ICON global and
ICON-D2 read zero. No other hour in either wide-domain download disagrees with its siblings that
way.

**The four models added for issue #809's second round were first downloaded on a coarse grid of
points across the trial area**, by the issue #841 downloader, whose `lineage.json` files record the
`models=` values used here. Those grids hold 49 points about 0.15° apart, not the 0.05° the lineage
files state, so the nearest point sits 0.7 km to 5.3 km from a solar farm; fetching at each site's
own coordinates removes that handicap, which would fall hardest on DMI's 2 km model. DMI's and
KNMI's HARMONIE-AROME feeds both carry the same 2 km run that the United Weather Centres-West
(UWC-West) collaboration of the Danish, Dutch, Icelandic and Irish weather services operates over
north-west Europe up to Iceland; KNMI distributes it on a reduced 0.05° grid, about 5.5 km
(<https://english.knmidata.nl/open-data/harmonie>,
<https://open-meteo.com/en/docs/dmi-api>, <https://open-meteo.com/en/docs/knmi-api>).

- `ecmwf-ifs-hres`: fetched as `ecmwf_ifs`, not `ecmwf_ifs_hres` — the API rejects `ecmwf_ifs_hres`
  outright with "Cannot initialize MultiDomains from invalid String value". The grid download was
  requested as `ecmwf_ifs04`, the 0.4° open-data name, yet serves values from 2017-01-01 at 37
  distinct series among 49 points 0.15° apart, which a 0.4° grid could not produce; a one-week
  check at the grid point nearest site B confirmed `ecmwf_ifs` and `ecmwf_ifs04` serve identical
  values there (zero difference, correlation 1.0), so both names reach the same underlying field.
- `arpege-europe`: both fluxes step up at 2024-01-01 against ECMWF-IFS-HRES, by about 20% for the
  global and 40% for the direct flux, and both are null for 35 hours from 2023-12-31 07:00 UTC to
  2024-01-01 17:00 UTC. The archive before the step is treated as a different product and not
  fetched; the start is the first whole day after the gap.
- `dmi-harmonie-arome`: the served direct flux is exactly zero or exceeds the served global flux on
  a share of daytime hours `check_new_products.py` prints into `product_checks.md`
  (`_dmi_beam_defect_lines`), so the model's split is unusable and only its global flux is scored.
- `arpege-europe` and `knmi-harmonie-arome`: the served direct flux fails
  `check_direct_is_not_a_separation_model` on the grid downloads, with a within-bin spread of the
  direct fraction of 0.018 against a threshold of 0.05, so it is a separation model's output rather
  than the model's own beam. Only their global flux is scored.
- `native_radiation` for all four is `accumulated` on the models' published GRIB conventions, not on
  a measurement: ECMWF's `ssrd` is accumulated since the run started, as is the surface radiation of
  the ALADIN code family that ARPEGE and both HARMONIE-AROME configurations belong to. No upstream
  file has been compared with the archive, as `verify_icon_lineage.py` does for ICON; the timing
  that matters to the arms is checked instead against the sun by `check_new_products.py`.

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

DOWNLOADS_DIR: Final[Path] = STUDIES_DATA_DIR
"""The layer of `data/studies/` that holds shared downloads, one folder per kind of data.

Equal to `STUDIES_DATA_DIR` until the data moves. No script reads this constant yet.
"""

PER_STUDY_DIR: Final[Path] = STUDIES_DATA_DIR
"""The layer of `data/studies/` that holds one folder per study.

Equal to `STUDIES_DATA_DIR` until the data moves. Every study's folder below is built from this
constant.
"""

SCRATCH_DIR: Final[Path] = REPO_DATA_DIR / "_scratch"
"""Where a whole-domain download or an archive extraction lands transiently, and is then deleted.

Under `data/`, not `/tmp`: `/tmp` on this machine is tmpfs, and a multi-gigabyte file there
consumes memory rather than disk. The CERRA fetches and the ERA5 archive reader use this folder.
"""

WEATHER_DATA_DIR: Final[Path] = STUDIES_DATA_DIR / "weather"
"""Where downloaded weather lands, one subdirectory per product (`ERA5`, `CAMS`, `ENS`, `UKV`,
`ICON-D2`, `ICON-EU`, `ICON-GLOBAL`, `SARAH-3`, `ICON-DREAM-EU`, `ECMWF-IFS-HRES`, `ARPEGE-EUROPE`,
`DMI-HARMONIE-AROME`, `KNMI-HARMONIE-AROME`).

Kept apart from any one study's outputs because a download is an input a later study can reuse, and
some take most of a night to fetch again.
"""

TRIAL_AREA_BOX_PATH: Final[Path] = WEATHER_DATA_DIR / "_trial_area_box.json"
"""Where the trial-area box's bounds are kept.

**This file is never read by anything outside this process's private working state, and its
contents must never be logged, printed, committed, or quoted back in a report.** The bounds are
derived from the private generator roster (`packages/contracts` `TimeSeriesMetadata`), and NGED's
generator locations must never appear in anything published — see CLAUDE.md.
"""

ANM_DATA_DIR: Final[Path] = STUDIES_DATA_DIR / "anm"
"""Where NGED's active network management setpoint exports are filed, one CSV per `time_series_id`,
beside the export-cap parquet `anm_setpoints.py` derives from each.
"""


def product_dir_for(*, product: str) -> Path:
    """Return the folder of a weather product whose name is only known at run time.

    A script that names one product writes a constant below. A script that loops over products, or
    takes the product's name from a registry or the command line, calls this function, so the
    layout of the weather folders is spelled out in this module alone.

    Args:
        product: The product's folder name, such as `ICON-D2` or `ECMWF-IFS-025`.

    Returns:
        The product's folder.
    """
    return WEATHER_DATA_DIR / product


def previous_runs_product_dir_for(*, product: str) -> Path:
    """Return the folder of an Open-Meteo Previous Runs product, given the product's folder name.

    Args:
        product: The product's folder name, such as `ICON-D2` or `ECMWF-IFS-025`.

    Returns:
        The product's folder, which holds its `previous_runs/` download.
    """
    return product_dir_for(product=product)


ERA5_PRODUCT_DIR: Final[Path] = product_dir_for(product="ERA5")
"""ERA5, fetched from Open-Meteo's mirror and from the Copernicus Climate Data Store."""

ERA5_WIND_2019_2023_PRODUCT_DIR: Final[Path] = product_dir_for(product="ERA5-WIND-2019-2023")
"""ERA5 native-level wind for 2019 to 2023."""

CAMS_PRODUCT_DIR: Final[Path] = product_dir_for(product="CAMS")
"""The CAMS radiation service's satellite retrieval."""

ENS_PRODUCT_DIR: Final[Path] = product_dir_for(product="ENS")
"""The per-site extract of ECMWF's ensemble."""

CERRA_PRODUCT_DIR: Final[Path] = product_dir_for(product="CERRA")
"""The CERRA regional reanalysis."""

NORA3_PRODUCT_DIR: Final[Path] = product_dir_for(product="NORA3")
"""The NORA3 reanalysis, wind at several heights."""

NORA3_10M_PRODUCT_DIR: Final[Path] = product_dir_for(product="NORA3_10m")
"""The NORA3 reanalysis, 10 m wind."""

ICON_DREAM_EU_PRODUCT_DIR: Final[Path] = product_dir_for(product="ICON-DREAM-EU")
"""The ICON-DREAM-EU reanalysis."""

MIDAS_OPEN_PRODUCT_DIR: Final[Path] = product_dir_for(product="MIDAS-OPEN")
"""The Met Office's MIDAS Open station observations."""

SARAH_3_PRODUCT_DIR: Final[Path] = product_dir_for(product="SARAH-3")
"""The SARAH-3 satellite retrieval."""

ECMWF_IFS_HRES_PRODUCT_DIR: Final[Path] = product_dir_for(product="ECMWF-IFS-HRES")
"""Open-Meteo's historical-forecast archive of ECMWF's high-resolution forecast."""

ECMWF_IFS_SINGLE_RUNS_PRODUCT_DIR: Final[Path] = product_dir_for(product="ECMWF-IFS-SINGLE-RUNS")
"""Open-Meteo's Single Runs archive of ECMWF's high-resolution forecast."""

ECMWF_AIFS_PRODUCT_DIR: Final[Path] = product_dir_for(product="ECMWF-AIFS")
"""ECMWF's AIFS Single forecast."""

ECMWF_AIFS_ENS_PRODUCT_DIR: Final[Path] = product_dir_for(product="ECMWF-AIFS-ENS")
"""ECMWF's AIFS ensemble forecast."""

GEFS_WINDOW_DIR: Final[Path] = product_dir_for(product="GEFS_window_2024-11-01_None")
"""The finished GEFS download, with its `_month_cache/`."""

GFS_PRODUCT_DIR: Final[Path] = product_dir_for(product="GFS")
"""The native GFS store from Dynamical.org."""

GFS_WINDOW_DIR: Final[Path] = product_dir_for(product="GFS_window_2025-07-01_2025-07-02")
"""Open-Meteo's GFS Previous Runs archive over a two-day window."""

WEATHERNEXT3_PRODUCT_DIR: Final[Path] = product_dir_for(product="WeatherNext3_trial_area")
"""The local copy of WeatherNext 3 over the trial area."""

UKV_CEDA_T120_PRODUCT_DIR: Final[Path] = product_dir_for(product="UKV-CEDA-T120")
"""The Met Office's UKV archive on CEDA, run-time 120 forecasts."""

OPEN_METEO_ENSEMBLE_MEANS_PRODUCT_DIR: Final[Path] = product_dir_for(
    product="OPEN-METEO-ENSEMBLE-MEANS"
)
"""Open-Meteo's ensemble-mean downloads."""


def study_dir_for(*, study: str) -> Path:
    """Return a study's folder, given the study's name.

    Args:
        study: The study's folder name, such as `ens_forecast_horizons`.

    Returns:
        The study's folder.
    """
    return PER_STUDY_DIR / study


STUDY_DATA_DIR: Final[Path] = study_dir_for(study="beam_diffuse_split")
"""Where everything this study builds from its inputs lives: the joined datasets, each arm's
results, and the figures.
"""

ENS_FORECAST_HORIZONS_DIR: Final[Path] = study_dir_for(study="ens_forecast_horizons")
"""The ENS forecast-horizons study's inputs and results."""

ENS_FORECAST_HORIZONS_DAY4_DIR: Final[Path] = study_dir_for(study="ens_forecast_horizons_day4")
"""The day-4 supplement to the ENS member extract."""

OPEN_METEO_ENSEMBLE_MEANS_DIR: Final[Path] = study_dir_for(study="open_meteo_ensemble_means")
"""The Open-Meteo ensemble-means study."""

OPEN_METEO_ENS_GAP_DIR: Final[Path] = study_dir_for(study="open_meteo_ens_gap")
"""The study of the gap between a local ensemble and ENS."""

ICON_EU_COMPARE_DIR: Final[Path] = study_dir_for(study="icon_eu_compare")
"""The ICON-EU comparison between Dynamical.org and Open-Meteo."""

ERA5_WIND_COMPARE_DIR: Final[Path] = study_dir_for(study="era5_wind_compare")
"""The ERA5 wind comparison between Open-Meteo and the Climate Data Store."""

ENS_BACKFILL_PILOT_DIR: Final[Path] = study_dir_for(study="ens_backfill_pilot")
"""The ENS backfill pilot's checkpoint files."""

CERRA_WIND_LEVELS_DIR: Final[Path] = study_dir_for(study="cerra_wind_levels")
"""The CERRA wind-levels study."""

CERRA_WIND_LEVELS_POST_HOC_DIR: Final[Path] = study_dir_for(study="cerra_wind_levels_post_hoc")
"""The CERRA wind-levels study's post-hoc shear analysis."""

CERRA_WIND_LEVELS_SHEAR_DIR: Final[Path] = study_dir_for(study="cerra_wind_levels_shear")
"""An earlier output of the CERRA wind-shear analysis."""

CERRA_WIND_DIRECTION_DIR: Final[Path] = study_dir_for(study="cerra_wind_direction")
"""The CERRA wind-direction study."""

UKV_CEDA_BLENDS_DIR: Final[Path] = study_dir_for(study="ukv_ceda_blends")
"""The planned run of the UKV-on-CEDA blends study."""

UKV_CEDA_BLENDS_RUN15_DIR: Final[Path] = study_dir_for(study="ukv_ceda_blends_run15")
"""The UKV-on-CEDA blends study's run on the 15 UTC cycle."""

NFC_DIR: Final[Path] = study_dir_for(study="nwp_forecast_comparison")
"""The NWP forecast comparison's original batch, which holds the published fit."""


def nfc_batch_dir_for(*, batch: str) -> Path:
    """Return the folder of one batch of the NWP forecast comparison.

    Args:
        batch: The batch's name without its study prefix, such as `aifs_blends`.

    Returns:
        The batch's folder.
    """
    return study_dir_for(study=f"nwp_forecast_comparison_{batch}")


def per_study_relative(*, folder: Path) -> Path:
    """Return a study folder's path relative to `PER_STUDY_DIR`.

    A script that takes the per-study folder as a command-line argument joins this path onto the
    argument, so the script's tests can point the argument at a temporary directory.

    Args:
        folder: A folder under `PER_STUDY_DIR`, such as `NFC_AIFS_BLENDS_DIR`.

    Returns:
        The folder's path below `PER_STUDY_DIR`.
    """
    return folder.relative_to(PER_STUDY_DIR)


# One folder per batch of the NWP forecast comparison, each holding that batch's inputs and
# fits. The `NFC_` prefix abbreviates `nwp_forecast_comparison`, and a batch's name is the folder's
# name without that prefix. Some batches have no reader of their own: `NFC_BATCH_DIRS` lists them
# all, for the data moves that rename every batch folder.
NFC_AIFS_DIR: Final[Path] = nfc_batch_dir_for(batch="aifs")
NFC_AIFS_BLENDS_DIR: Final[Path] = nfc_batch_dir_for(batch="aifs_blends")
NFC_AIFS_EXTRA_DAYS_DIR: Final[Path] = nfc_batch_dir_for(batch="aifs_extra_days")
NFC_DAY4_SHARED_DIR: Final[Path] = nfc_batch_dir_for(batch="day4_shared")
NFC_DAY5_AIFS_WN3_DIR: Final[Path] = nfc_batch_dir_for(batch="day5_aifs_wn3")
NFC_LEADERBOARD_BY_DAY_DIR: Final[Path] = nfc_batch_dir_for(batch="leaderboard_by_day")
NFC_LEADERBOARD_BY_DAY_FIG3_DIR: Final[Path] = nfc_batch_dir_for(batch="leaderboard_by_day_fig3")
NFC_LEADS_DIR: Final[Path] = nfc_batch_dir_for(batch="leads")
NFC_LEADS_DAY10_DIR: Final[Path] = nfc_batch_dir_for(batch="leads_day10")
NFC_LEADS_DAY10B_DIR: Final[Path] = nfc_batch_dir_for(batch="leads_day10b")
NFC_LEADS_DAY10C_DIR: Final[Path] = nfc_batch_dir_for(batch="leads_day10c")
NFC_LEADS_DAY10D_DIR: Final[Path] = nfc_batch_dir_for(batch="leads_day10d")
NFC_P4_SEEDS_DIR: Final[Path] = nfc_batch_dir_for(batch="p4_seeds")
NFC_PRODUCT_BLENDS_DIR: Final[Path] = nfc_batch_dir_for(batch="product_blends")
NFC_PRODUCT_BLENDS_REPORT_DIR: Final[Path] = nfc_batch_dir_for(batch="product_blends_report")
NFC_VS_ENS_DOTS_DIR: Final[Path] = nfc_batch_dir_for(batch="vs_ens_dots")
NFC_VS_ENS_DOTS_ALL_DAYS_DIR: Final[Path] = nfc_batch_dir_for(batch="vs_ens_dots_all_days")
NFC_VS_ENS_DOTS_BLENDS_DIR: Final[Path] = nfc_batch_dir_for(batch="vs_ens_dots_blends")
NFC_VS_ENS_DOTS_BLENDS_FINAL_DIR: Final[Path] = nfc_batch_dir_for(batch="vs_ens_dots_blends_final")
NFC_VS_ENS_DOTS_FINAL_DIR: Final[Path] = nfc_batch_dir_for(batch="vs_ens_dots_final")
NFC_WN3_DIR: Final[Path] = nfc_batch_dir_for(batch="wn3")
NFC_WN3_EXTRA_DAYS_DIR: Final[Path] = nfc_batch_dir_for(batch="wn3_extra_days")

NFC_BATCH_DIRS: Final[tuple[Path, ...]] = (
    NFC_AIFS_DIR,
    NFC_AIFS_BLENDS_DIR,
    NFC_AIFS_EXTRA_DAYS_DIR,
    NFC_DAY4_SHARED_DIR,
    NFC_DAY5_AIFS_WN3_DIR,
    NFC_LEADERBOARD_BY_DAY_DIR,
    NFC_LEADERBOARD_BY_DAY_FIG3_DIR,
    NFC_LEADS_DIR,
    NFC_LEADS_DAY10_DIR,
    NFC_LEADS_DAY10B_DIR,
    NFC_LEADS_DAY10C_DIR,
    NFC_LEADS_DAY10D_DIR,
    NFC_P4_SEEDS_DIR,
    NFC_PRODUCT_BLENDS_DIR,
    NFC_PRODUCT_BLENDS_REPORT_DIR,
    NFC_VS_ENS_DOTS_DIR,
    NFC_VS_ENS_DOTS_ALL_DAYS_DIR,
    NFC_VS_ENS_DOTS_BLENDS_DIR,
    NFC_VS_ENS_DOTS_BLENDS_FINAL_DIR,
    NFC_VS_ENS_DOTS_FINAL_DIR,
    NFC_WN3_DIR,
    NFC_WN3_EXTRA_DAYS_DIR,
)
"""Every batch folder of the NWP forecast comparison, apart from the original batch `NFC_DIR`."""

NFC_STAMP_GLOB: Final[str] = "nwp_forecast_comparison_*/*_losses.json"
"""Matches every earlier batch's `*_losses.json` stamp, relative to `PER_STUDY_DIR`.

The stamps record the columns each fit used, which `check_arm_columns_unchanged.py` compares.
"""


UPDATE_OUTPUT_DIR: Final[Path] = STUDY_DATA_DIR / "past_weather_v2"
"""Where every output of the second round of the past-weather studies is written.

The second round adds six products to the sunshine study and the ERA5 year-by-year tables to both
studies. Its outputs live apart from the first round's `beam_diffuse_weather_products` and
`beam_diffuse_wind_products`, which the published pages quote and the blending study and the ENS
forecast study read as their reference rows, so no run of the second round can overwrite them.
"""


SOLAR_LEADERBOARD_DIR: Final[Path] = UPDATE_OUTPUT_DIR / "solar_leaderboard_3"
"""Where `past_solar_leaderboard.py` writes the past-solar page's leaderboard and contrasts.

The earlier folder `solar_leaderboard_2` holds four row sets, and this folder holds five: the
same four and the CERRA rows.

Write-once: the script refuses to run where this folder exists, so the numbers a page quotes from
its `report.md` and `intervals.parquet` cannot be overwritten by a later run.
"""


WIND_LEADERBOARD_DIR: Final[Path] = UPDATE_OUTPUT_DIR / "wind_leaderboard_2"
"""Where `past_wind_leaderboard.py` writes the past-wind page's leaderboard and contrasts.

The first run's folder, `wind_leaderboard`, holds a contrast heading that the main block no longer
uses, so the script writes to `wind_leaderboard_2`.

Write-once: the script refuses to run where this folder exists, so the numbers a page quotes from
its `report.md` and `intervals.parquet` cannot be overwritten by a later run.
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
    return product_dir_for(product=source.upper()) / f"beam_diffuse_{source}.parquet"


def temperature_site_b_path_for(*, source: SourceType) -> Path:
    """Return where one model's single-site 2 m temperature fetch is written.

    A throwaway download, one site (B) only, for `check_new_products.py`'s night-jump table:
    `time` and `temperature_2m`, hourly. Temperature rather than radiation, because it is served
    around the clock, so a night-time reading isolates a run switch from the diurnal solar cycle
    that swamps the same measure on radiation.

    Args:
        source: Which model's download to locate.

    Returns:
        The parquet path holding that model's single-site hourly temperature.
    """
    return product_dir_for(product=source.upper()) / "temperature_2m_site_b.parquet"


IFS_OPEN_DATA_CUTOVER: Final[datetime] = datetime(2025, 10, 1, tzinfo=UTC)
"""When Open-Meteo's historical-forecast archive switched ECMWF-IFS-HRES to ECMWF's own open-data
catalogue.

Before this date the archive served IFS-HRES with roughly a one-hour publication delay; from this
date it serves the native 9 km O1280 HRES hourly to 90 hours with no such delay, following
[ECMWF's real-time catalogue opening on 2025-10-01](https://openmeteo.substack.com/p/ecmwf-transitions-to-open-data).
`check_new_products.py`'s night-jump table measures IFS-HRES's run cadence on both sides of this
date separately, because the switch to a faster catalogue is expected to change how often a new
run appears in the archive as well as how quickly.
"""


HISTORICAL_FORECAST_URL: Final[str] = "https://historical-forecast-api.open-meteo.com/v1/forecast"
"""Where a forecast model's archive is served from.

A different service from the `archive-api` endpoint `fetch_era5_open_meteo.py` uses for ERA5, with
its own call-weight accounting, though the JSON `hourly` block has the same shape.
"""
