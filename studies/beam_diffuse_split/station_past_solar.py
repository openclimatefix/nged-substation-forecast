"""Score nearby Met Office weather stations as a stand-in for a gridded product in past sunshine.

One-off throwaway script for the addition to
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>, extending
`weather_products.py`'s comparison with the Met Office's MIDAS Open station observations. The plan
was committed before the first fit, and stays in the git history after `plans/` is emptied at merge.

**The question.** Instead of estimating the sunshine at a solar farm from a gridded product (CAMS or
ERA5), what if the nearest Met Office weather station is used, or a blend of a gridded product and a
station? A station arm tests how well a pyranometer some tens of kilometres away stands in for the
site's own sunshine. The stations publish global irradiance only (diffuse and direct are null at
every station), so no arm here carries a beam split.

**Data.** `data/studies/weather/MIDAS-OPEN/` (its `README.md` and `lineage.json`), read through
`studies.midas`: hourly global irradiance from 10 stations (kJ m⁻² over the hour ending at `time`,
converted to W m⁻²) and air temperature from 38 stations (an instant at `time`). Coverage ends on
2025-12-31, so this section's row set ends there, about eight months before the page's main row set.

**The selection rule, fixed before any score existed.** `studies.midas.select_nearest_stations`
with `MIN_COVERAGE`: stations are ranked by great-circle distance (ties to the lower station id) and
a station is eligible when it has a usable value at no less than `MIN_COVERAGE` of the site's
candidate hours, the blend study's common rows (`blend_products._solar_frame`) before
`ROW_SET_END`. The nearest radiation stations and
the nearest air-temperature stations are chosen independently. Hours where any station input an arm
needs is missing are dropped from every arm's rows, so every arm scores the same rows. The choice
frame maps stations to generators, which is sensitive: this script never saves, logs or prints it,
and the report carries only pooled distance ranges and counts of distinct stations.

**Arms.** Every arm carries the same number of feature columns as the arm it is contrasted with, and
every fit uses `colsample_bytree=1`. `jobs()` builds each arm's columns from one function and
`_check_column_counts` raises if a contrast's two arms differ in width; the report prints every
arm's columns.

**Planned contrasts** (`PLANNED_CONTRASTS`, written into the plan file before the first
fit): the nearest station's irradiance and temperature against CAMS and against ERA5, and
CAMS blended with the station against CAMS padded with a climatology-permuted copy of the station's
irradiance (equal column counts). Each is also run at the second hyperparameter setting. Every
other contrast is exploratory and labelled so in the report.

Run it with `uv run python studies/beam_diffuse_split/station_past_solar.py`, after
`weather_products.py` has built its datasets. `--report-only` rebuilds `report.md` from the saved
`losses.parquet` alone, still checking the saved fingerprint. A re-run first moves `losses.parquet`,
`losses.fingerprint` and `report.md` to a `superseded/` subfolder (`refuse_to_overwrite`). Only one
agent may run it at a time, because every worktree shares one data folder.
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

import numpy as np
import polars as pl
from blend_products import (
    PERMUTATION_GROUPS,
    SOLAR,
    _solar_frame,
)
from build_dataset import _pv_sites
from run_experiment import MAX_CONCURRENT_FITS, Job, run_all
from sources import STUDY_DATA_DIR, WEATHER_DATA_DIR
from studies.blending import climatology_permutation
from studies.bootstrap import bootstrap_absolute
from studies.charts import report_errors
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS, SEEDS, SENSITIVITY_HYPER_PARAMETERS
from studies.grid_sampling import distance_matrix_km
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.midas import (
    null_night_spikes,
    read_hourly_weather,
    read_radiation,
    read_station_metadata,
    select_nearest_stations,
)
from weather_products import (
    CONTRAST_HEADER,
    METRIC,
    PERCENTAGE_POINTS,
    _contrast_line,
    _mae,
    geometry_lines,
    with_eras,
)

_LOG: Final[logging.Logger] = logging.getLogger("station_past_solar")

MIDAS_DIR: Final[Path] = WEATHER_DATA_DIR / "MIDAS-OPEN"
RADIATION_PATH: Final[Path] = MIDAS_DIR / "uk_radiation_obs_hourly.parquet"
WEATHER_PATH: Final[Path] = MIDAS_DIR / "uk_hourly_weather_obs.parquet"
RADIATION_METADATA_PATH: Final[Path] = (
    MIDAS_DIR / "_station_metadata" / "midas-open_uk-radiation-obs_dv-202607_station-metadata.csv"
)
WEATHER_METADATA_PATH: Final[Path] = (
    MIDAS_DIR
    / "_station_metadata"
    / "midas-open_uk-hourly-weather-obs_dv-202607_station-metadata.csv"
)

OUTPUT_DIR: Final[Path] = STUDY_DATA_DIR / "past_weather_v2" / "station_past_solar"

ROW_SET_START_YEAR: Final[int] = 2022
"""The calendar year of the row set's first hour, December 2022."""

ROW_SET_END: Final[datetime] = datetime(2026, 1, 1, tzinfo=UTC)
"""The first instant after the station files' last hour (2025-12-31 23:00 UTC); rows end before."""

MIN_COVERAGE: Final[float] = 0.99
"""The share of a site's candidate hours a station must cover to be eligible.

1.0 leaves at least one site with no eligible station, because no station is complete.
"""

RANK_DEPTH: Final[int] = 3
"""How many nearest stations of each kind are read: the nearest, the second and the third."""

USUAL_FLAG: Final[int] = 6
"""The quality-control flag value on 766,702 of the 772,093 radiation rows in the files.

The flags are kept as delivered and never interpreted: no hour is dropped on its flag. The report
counts the hours where the nearest station's flag differs from this value, as a diagnostic.
"""

STATION_PERMUTATION_SEED: Final[int] = 20260927
"""The seed of the permutation that builds the padded controls' station irradiance column."""

KM_FORMAT: Final[str] = ".0f"

GHI: Final[str] = "ghi_station_r{rank}"
TEMPERATURE: Final[str] = "temp_station_r{rank}"
GHI_MEAN: Final[str] = "ghi_station_mean3"
TEMPERATURE_MEAN: Final[str] = "temp_station_mean3"
SHUFFLED: Final[str] = "ghi_station_r1_shuffled"

STATION_ARM: Final[str] = "station_global"
BLEND_ARM: Final[str] = "cams_station_xgb"
BLEND_CONTROL_ARM: Final[str] = "cams_station_control"

PLANNED_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (STATION_ARM, "cams_global"),
    (STATION_ARM, "era5_global"),
    (BLEND_ARM, BLEND_CONTROL_ARM),
)
"""The three contrasts written into the plan before the first fit: (treatment, reference)."""

EXPLORATORY_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (STATION_ARM, "station_ghi_era5temp"),
    (STATION_ARM, "station_shuffled"),
    ("station_mean3", STATION_ARM),
    ("station_mean3", "cams_global"),
    ("station_mean3", "era5_global"),
    ("station_rank2", STATION_ARM),
    ("station_rank3", STATION_ARM),
    ("station_rank2", "era5_global"),
    ("station_rank3", "era5_global"),
    ("station_era5_xgb", "station_era5_control"),
    (BLEND_ARM, "cams_global"),
    (BLEND_CONTROL_ARM, "cams_global"),
    ("station_era5_control", "era5_global"),
)
"""Every contrast beyond the three planned ones, in the order the report prints them."""

UNEQUAL_COLUMN_CONTRASTS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        (BLEND_ARM, "cams_global"),
        (BLEND_CONTROL_ARM, "cams_global"),
        ("station_era5_control", "era5_global"),
    }
)
"""The contrasts whose arms differ in width, descriptive only; the width check allows them.

The last two pair a padded control with the plain product, so each measures what a permuted
station column adds to that product: nothing, if the control is sound.
"""

ARM_ORDER: Final[tuple[str, ...]] = (
    "cams_global",
    "era5_global",
    STATION_ARM,
    "station_ghi_era5temp",
    "station_mean3",
    "station_rank2",
    "station_rank3",
    "station_shuffled",
    BLEND_ARM,
    BLEND_CONTROL_ARM,
    "station_era5_xgb",
    "station_era5_control",
)

SENSITIVITY_ARMS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(arm for pair in PLANNED_CONTRASTS for arm in pair)
)
"""The arms refit at the second hyperparameter setting: every arm in a planned contrast."""

HOUR_PROFILE_MIN_ROWS: Final[int] = 500
"""Hours of day with fewer scored rows than this are left out of the hour-of-day profile."""


def _shared_no_temperature() -> tuple[str, ...]:
    """Return the study's shared feature columns without ERA5's temperature."""
    return tuple(name for name in SOLAR.shared_features if name != "temp_c")


def _arm_features() -> dict[str, tuple[str, ...]]:
    """Return every arm's feature columns, from one place, in the order the model sees them.

    Returns:
        Arm name to feature columns.
    """
    shared = SOLAR.shared_features
    no_temperature = _shared_no_temperature()
    station = (*no_temperature, TEMPERATURE.format(rank=1), GHI.format(rank=1))
    return {
        "cams_global": (*shared, "ghi_cams"),
        "era5_global": (*shared, "ghi_era5"),
        STATION_ARM: station,
        "station_ghi_era5temp": (*shared, GHI.format(rank=1)),
        "station_mean3": (*no_temperature, TEMPERATURE_MEAN, GHI_MEAN),
        "station_rank2": (*no_temperature, TEMPERATURE.format(rank=2), GHI.format(rank=2)),
        "station_rank3": (*no_temperature, TEMPERATURE.format(rank=3), GHI.format(rank=3)),
        "station_shuffled": (*no_temperature, TEMPERATURE.format(rank=1), SHUFFLED),
        BLEND_ARM: (*shared, "ghi_cams", GHI.format(rank=1)),
        BLEND_CONTROL_ARM: (*shared, "ghi_cams", SHUFFLED),
        "station_era5_xgb": (*shared, "ghi_era5", GHI.format(rank=1)),
        "station_era5_control": (*shared, "ghi_era5", SHUFFLED),
    }


def _check_column_counts(*, features: dict[str, tuple[str, ...]]) -> None:
    """Raise unless every contrast's two arms carry the same number of feature columns.

    Args:
        features: Arm name to feature columns.

    Raises:
        ValueError: If an arm is missing, an arm repeats a column, or a contrast that is not in
            `UNEQUAL_COLUMN_CONTRASTS` pairs two arms of different widths.
    """
    for arm, columns in features.items():
        if len(set(columns)) != len(columns):
            msg = f"{arm} repeats a feature column: {columns}"
            raise ValueError(msg)
    for treatment, reference in (*PLANNED_CONTRASTS, *EXPLORATORY_CONTRASTS):
        for arm in (treatment, reference):
            if arm not in features:
                msg = f"{arm} is in a contrast but has no feature columns"
                raise ValueError(msg)
        unequal = len(features[treatment]) != len(features[reference])
        if unequal and (treatment, reference) not in UNEQUAL_COLUMN_CONTRASTS:
            msg = (
                f"{treatment} has {len(features[treatment])} columns and {reference} has "
                f"{len(features[reference])}: equal counts are required"
            )
            raise ValueError(msg)


def jobs() -> list[Job]:
    """Return every arm at `pooled`, and the planned contrasts' arms at `sensitivity` too.

    Returns:
        One job per arm at `pooled`, plus one per arm in `SENSITIVITY_ARMS` at `sensitivity`.

    Raises:
        ValueError: If `_check_column_counts` fails.
    """
    features = _arm_features()
    _check_column_counts(features=features)
    job_list: list[Job] = [
        (arm, "pooled", "power_mw", features[arm], PRIMARY_HYPER_PARAMETERS, False)
        for arm in ARM_ORDER
    ]
    job_list += [
        (arm, "sensitivity", "power_mw", features[arm], SENSITIVITY_HYPER_PARAMETERS, False)
        for arm in SENSITIVITY_ARMS
    ]
    return job_list


class Selection:
    """The stations chosen for every site, held in memory only.

    The frames map stations to generators, which is sensitive. Nothing in this class is saved,
    logged or printed except through `pooled_lines`, which reports pooled ranges and counts.
    """

    def __init__(
        self,
        *,
        radiation: pl.DataFrame,
        temperature: pl.DataFrame,
        unusual_flags: pl.DataFrame,
        undownloaded: tuple[int, float],
    ) -> None:
        """Hold the two choice frames and the hours with an unusual quality-control flag.

        Args:
            radiation: `select_nearest_stations`'s result for the radiation stations.
            temperature: The same for the air-temperature stations.
            unusual_flags: The `(site, time)` pairs where the nearest radiation station's flag
                differs from `USUAL_FLAG`, a diagnostic and never a filter.
            undownloaded: How many radiation stations the metadata file lists that were not
                downloaded, and the smallest distance in km from any farm to any of them.
        """
        self.radiation = radiation
        self.temperature = temperature
        self.unusual_flags = unusual_flags
        self.undownloaded = undownloaded

    def pooled_lines(self) -> list[str]:
        """Render the choices as pooled ranges and counts, never as a station-to-site mapping.

        Returns:
            Markdown lines.
        """
        lines = [
            "#### Which stations stood in (pooled; no station-to-site mapping is printed)",
            "",
            (
                "| Kind | Rank | Distance range (km) | Distinct stations | Lowest coverage "
                "| Most nearer stations skipped |"
            ),
            "|---|---|---|---|---|---|",
        ]
        for kind, chosen in (("radiation", self.radiation), ("air temperature", self.temperature)):
            for rank in range(1, RANK_DEPTH + 1):
                ranked = chosen.filter(pl.col("rank") == rank)
                lines.append(
                    f"| {kind} | {rank} | {ranked['distance_km'].min():{KM_FORMAT}} to "
                    f"{ranked['distance_km'].max():{KM_FORMAT}} | {ranked['src_id'].n_unique()} | "
                    f"{ranked['coverage'].min():.4f} | {ranked['skipped_nearer'].max()} |"
                )
        count, nearest_km = self.undownloaded
        lines += [
            "",
            (
                f"The radiation station-metadata file also lists {count} stations whose record "
                "overlaps the row set's years and that were not downloaded. The nearest of them "
                f"is {nearest_km:{KM_FORMAT}} km or more from every generator."
            ),
            "",
            (
                f"Stations across the {RANK_DEPTH} ranks: "
                f"{self.radiation['src_id'].n_unique()} distinct radiation stations and "
                f"{self.temperature['src_id'].n_unique()} distinct air-temperature stations, "
                f"for {self.radiation['site'].n_unique()} generators."
            ),
        ]
        return lines

    def nearest_distance_range_km(self) -> tuple[float, float]:
        """Return the pooled range of the nearest radiation station's distance, in km."""
        nearest = self.radiation.filter(pl.col("rank") == 1)["distance_km"]
        return cast("float", nearest.min()), cast("float", nearest.max())


def _station_columns(
    *, base: pl.DataFrame, chosen: pl.DataFrame, observations: pl.DataFrame, value: str, name: str
) -> pl.DataFrame:
    """Return `(site, time)` with each site's chosen stations' value, one column per rank.

    Args:
        base: The candidate rows, with `site` and `time`.
        chosen: `select_nearest_stations`'s result.
        observations: `src_id`, `time` and `value`.
        value: The observation column to read.
        name: A format string with a `{rank}` field naming the output columns.

    Returns:
        One row per `(site, time)` where all ranks have a value, with one column per rank.
    """
    joined = base.select("site", "time")
    for rank in range(1, RANK_DEPTH + 1):
        stations = chosen.filter(pl.col("rank") == rank).select("site", "src_id")
        column = (
            joined.join(stations, on="site", how="inner")
            .join(observations.select("src_id", "time", value), on=["src_id", "time"], how="inner")
            .select("site", "time", pl.col(value).alias(name.format(rank=rank)))
        )
        joined = joined.join(column, on=["site", "time"], how="inner")
    return joined


def _station_inputs(*, base: pl.DataFrame) -> tuple[pl.DataFrame, Selection, dict[str, int]]:
    """Read the station files, choose the stations, and build each site's station columns.

    Args:
        base: The candidate rows: the page's common rows before `ROW_SET_END`.

    Returns:
        The station columns per `(site, time)`, the in-memory `Selection`, and counts of the
        repairs the reader made (`spike_hours`, `clipped_negative_hours`).
    """
    sites = _pv_sites().select("site", "latitude", "longitude")
    required = base.select("site", "time")

    radiation_metadata = read_station_metadata(path=RADIATION_METADATA_PATH)
    raw = read_radiation(path=RADIATION_PATH)
    listed = radiation_metadata
    radiation_metadata = radiation_metadata.filter(
        pl.col("src_id").is_in(raw["src_id"].unique().implode())
    )
    others = listed.filter(
        ~pl.col("src_id").is_in(raw["src_id"].unique().implode()),
        pl.col("first_year") <= ROW_SET_END.year - 1,
        pl.col("last_year") >= ROW_SET_START_YEAR,
    )
    other_distances = distance_matrix_km(sites=sites, cells=others)
    undownloaded = (others.height, float(other_distances.min()))
    negative = int(
        (pl.read_parquet(RADIATION_PATH, columns=["glbl_irad_amt"])["glbl_irad_amt"] < 0.0).sum()
    )
    radiation = null_night_spikes(radiation=raw, stations=radiation_metadata)
    spikes = radiation["ghi_w_m2"].null_count() - raw["ghi_w_m2"].null_count()
    usable = radiation.drop_nulls("ghi_w_m2")
    radiation_choice = select_nearest_stations(
        sites=sites,
        stations=radiation_metadata,
        observed=usable.select("src_id", "time"),
        required=required,
        k=RANK_DEPTH,
        min_coverage=MIN_COVERAGE,
    )

    weather = read_hourly_weather(path=WEATHER_PATH, columns=["air_temperature"]).drop_nulls(
        "air_temperature"
    )
    weather_metadata = read_station_metadata(path=WEATHER_METADATA_PATH).filter(
        pl.col("src_id").is_in(weather["src_id"].unique().implode())
    )
    temperature_choice = select_nearest_stations(
        sites=sites,
        stations=weather_metadata,
        observed=weather.select("src_id", "time"),
        required=required,
        k=RANK_DEPTH,
        min_coverage=MIN_COVERAGE,
    )

    ghi = _station_columns(
        base=base, chosen=radiation_choice, observations=usable, value="ghi_w_m2", name=GHI
    )
    temperature = _station_columns(
        base=base,
        chosen=temperature_choice,
        observations=weather,
        value="air_temperature",
        name=TEMPERATURE,
    )
    both = ghi.join(temperature, on=["site", "time"], how="inner").with_columns(
        (pl.mean_horizontal(*(GHI.format(rank=r) for r in range(1, RANK_DEPTH + 1)))).alias(
            GHI_MEAN
        ),
        (pl.mean_horizontal(*(TEMPERATURE.format(rank=r) for r in range(1, RANK_DEPTH + 1)))).alias(
            TEMPERATURE_MEAN
        ),
    )
    flags = pl.read_parquet(RADIATION_PATH, columns=["src_id", "time", "glbl_irad_amt_q"])
    nearest = radiation_choice.filter(pl.col("rank") == 1).select("site", "src_id")
    unusual = (
        required.join(nearest, on="site", how="inner")
        .join(flags, on=["src_id", "time"], how="inner")
        .filter(pl.col("glbl_irad_amt_q") != USUAL_FLAG)
        .select("site", "time")
    )
    repairs = {"spike_hours": spikes, "clipped_negative_hours": negative}
    selection = Selection(
        radiation=radiation_choice,
        temperature=temperature_choice,
        unusual_flags=unusual,
        undownloaded=undownloaded,
    )
    return both, selection, repairs


def build_rows() -> tuple[pl.DataFrame, Selection, dict[str, int], int]:
    """Build this section's row set.

    The page's common rows before `ROW_SET_END`, restricted to the hours where every station input
    any arm needs is present, with the folds recomputed on the restricted row set.

    Returns:
        The rows, the in-memory station `Selection`, the reader's repair counts, and the number of
        candidate rows before the station intersection.

    Raises:
        ValueError: If a `(site, time)` is duplicated, or a station column holds a missing value.
    """
    base = _solar_frame().filter(pl.col("time") < ROW_SET_END)
    stations, selection, repairs = _station_inputs(base=base)
    joined = (
        base.join(stations, on=["site", "time"], how="inner")
        .drop("era", "era_code", "fold")
        .sort("site", "time")
    )
    if joined.select("site", "time").is_duplicated().any():
        msg = "build_rows: a (site, time) is duplicated"
        raise ValueError(msg)
    padded = climatology_permutation(
        frame=joined,
        column_groups=[(GHI.format(rank=1),)],
        by=PERMUTATION_GROUPS,
        seed=STATION_PERMUTATION_SEED,
    )
    rows = with_eras(frame=padded)
    check_no_missing(
        frame=rows, columns=[column for columns in _arm_features().values() for column in columns]
    )
    _LOG.info("%d rows in this section's own row set, from %d candidates", rows.height, base.height)
    return rows, selection, repairs, base.height


def _fingerprint(*, frame: pl.DataFrame, job_list: list[Job]) -> str:
    """Return a hash covering every row's values, every job's columns, and the seeds.

    `--report-only` refuses to reuse a saved `losses.parquet` when this does not match. Every float
    column is cast to `Float32` before hashing, which makes a flip from floating-point noise in a
    rebuilt column less likely, though not impossible; a flip only refuses `--report-only`. The
    saved `losses.parquet` keeps full precision.

    Args:
        frame: The row set every job is fitted on.
        job_list: Every job this run means to fit.

    Returns:
        A hex digest.
    """
    ordered = frame.select(sorted(frame.columns)).sort("site", "time")
    float_columns = [name for name, dtype in ordered.schema.items() if dtype.is_float()]
    stable = ordered.cast(dict.fromkeys(float_columns, pl.Float32))
    row_hashes = stable.hash_rows(seed=0).to_list()
    payload = repr(
        (
            row_hashes,
            [
                (arm, setting, target, tuple(features), tuple(sorted(hyper_parameters.items())))
                for arm, setting, target, features, hyper_parameters, _ in job_list
            ],
            SEEDS,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _arm_columns_lines(*, job_list: list[Job]) -> list[str]:
    """Render every fitted arm's feature columns, once per arm, as markdown.

    Args:
        job_list: Every job `jobs()` returns.

    Returns:
        Markdown lines.
    """
    seen: dict[str, tuple[str, ...]] = {}
    for arm, _, _, columns, _, _ in job_list:
        seen.setdefault(arm, columns)
    lines = ["#### Every arm's feature columns", ""]
    lines += [
        f"- `{arm}` ({len(columns)} columns): {', '.join(f'`{c}`' for c in columns)}"
        for arm, columns in seen.items()
    ]
    return lines


def _absolute_table_lines(*, pooled: pl.DataFrame) -> list[str]:
    """Render every arm's mean absolute error and 95% interval as a markdown table.

    Args:
        pooled: Every arm's losses at the `pooled` setting.

    Returns:
        Markdown lines.
    """
    lines = ["| Arm | All sites | 95% interval |", "|---|---|---|"]
    for arm in ARM_ORDER:
        interval = bootstrap_absolute(losses=pooled, arm=arm, metric=METRIC)
        lower, upper = (interval[key] * PERCENTAGE_POINTS for key in ("lower_95", "upper_95"))
        lines.append(f"| {arm} | {_mae(losses=pooled, arm=arm):.3f} | [{lower:.3f}, {upper:.3f}] |")
    return lines


def _hour_profile_lines(*, frame: pl.DataFrame) -> list[str]:
    """Render mean irradiance by hour of day for the nearest station, CAMS and ERA5.

    A station series shifted by an hour against the products would show as a profile that peaks an
    hour early or late.

    Args:
        frame: This section's row set.

    Returns:
        Markdown lines.
    """
    profile = (
        frame.group_by("hour_of_day")
        .agg(
            n=pl.len(),
            station=pl.col(GHI.format(rank=1)).mean(),
            cams=pl.col("ghi_cams").mean(),
            era5=pl.col("ghi_era5").mean(),
        )
        .filter(pl.col("n") >= HOUR_PROFILE_MIN_ROWS)
        .sort("hour_of_day")
    )
    lines = [
        "#### Hour-of-day profile of the station, CAMS and ERA5 irradiance",
        "",
        "| Hour ending (UTC) | Rows | Station (W m⁻²) | CAMS (W m⁻²) | ERA5 (W m⁻²) |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['hour_of_day']:02d}:00 | {row['n']:,} | {row['station']:.1f} | "
        f"{row['cams']:.1f} | {row['era5']:.1f} |"
        for row in profile.iter_rows(named=True)
    ]
    peaks = {
        name: int(profile.sort(name, descending=True)["hour_of_day"][0])
        for name in ("station", "cams", "era5")
    }
    lines += [
        "",
        (
            f"Peak hour (hour ending, UTC): station {peaks['station']:02d}:00, "
            f"CAMS {peaks['cams']:02d}:00, ERA5 {peaks['era5']:02d}:00."
        ),
    ]
    return lines


def _agreement_lines(*, frame: pl.DataFrame) -> list[str]:
    """Render how closely CAMS and ERA5 follow the nearest station's irradiance.

    Args:
        frame: This section's row set.

    Returns:
        Markdown lines.
    """
    station = frame[GHI.format(rank=1)].to_numpy().astype(np.float64)
    lines = [
        "#### How closely each gridded product follows the nearest station (exploratory)",
        "",
        (
            "| Product | Correlation | Mean difference, product minus station (W m⁻²) "
            "| Mean station (W m⁻²) |"
        ),
        "|---|---|---|---|",
    ]
    for name, column in (("CAMS", "ghi_cams"), ("ERA5", "ghi_era5")):
        product = frame[column].to_numpy().astype(np.float64)
        lines.append(
            f"| {name} | {np.corrcoef(station, product)[0, 1]:.3f} | "
            f"{(product - station).mean():+.1f} | {station.mean():.1f} |"
        )
    return lines


def _files_lines() -> list[str]:
    """Render what the two MIDAS files hold: stations, and the first and last hour.

    Returns:
        Markdown lines.
    """
    radiation = read_radiation(path=RADIATION_PATH)
    weather = read_hourly_weather(path=WEATHER_PATH, columns=["air_temperature"])
    lineage = json.loads((MIDAS_DIR / "lineage.json").read_text())
    return [
        "#### What the MIDAS files hold",
        "",
        f"The files are MIDAS Open `{lineage['dataset_version']}`.",
        "",
        "| File | Stations | First hour (UTC) | Last hour (UTC) |",
        "|---|---|---|---|",
        *(
            f"| {name} | {frame['src_id'].n_unique()} | {frame['time'].min():%Y-%m-%d %H:%M} "
            f"| {frame['time'].max():%Y-%m-%d %H:%M} |"
            for name, frame in (
                ("hourly global irradiance", radiation),
                ("air temperature", weather),
            )
        ),
    ]


def _row_lines(*, frame: pl.DataFrame, candidates: int, repairs: dict[str, int]) -> list[str]:
    """Render the row counts and the reader's repairs.

    Args:
        frame: This section's row set.
        candidates: The common rows before `ROW_SET_END`, before the station intersection.
        repairs: The reader's repair counts.

    Returns:
        Markdown lines.
    """
    per_site = frame.group_by("site").agg(n=pl.len()).sort("site")
    dropped = candidates - frame.height
    months = frame["month"].n_unique()
    return [
        "#### The row set",
        "",
        (
            f"The common rows before {ROW_SET_END:%Y-%m-%d} number {candidates:,}. Keeping only "
            f"the hours where every station input is present leaves {frame.height:,} "
            f"({dropped:,} dropped, {dropped / candidates:.2%}), spanning {months} calendar months."
        ),
        "",
        "| Generator | Site-hours |",
        "|---|---|",
        *(f"| {row['site']} | {row['n']:,} |" for row in per_site.iter_rows(named=True)),
        "",
        (
            f"Reader repairs: {repairs['spike_hours']} radiation hours set to missing as night "
            f"spikes, {repairs['clipped_negative_hours']} negative radiation values clipped to "
            f"zero (station-file totals, before the row set is cut)."
        ),
    ]


def _main_panel_lines(*, pooled: pl.DataFrame, frame: pl.DataFrame) -> list[str]:
    """Render how far ERA5 and CAMS refit here differ from the page's main row set.

    The last column scores the main row set's own fits on the `(site, time)` keys both row sets
    hold, and its heading states how many keys those are.

    Args:
        pooled: Every arm's losses at the `pooled` setting.
        frame: This section's row set, whose `(site, time)` keys restrict the main row set's losses.

    Returns:
        Markdown lines.
    """
    path = OUTPUT_DIR.parent / "solar_long" / "report.md"
    heading = path.read_text().splitlines()[0]
    match = re.search(r"on ([\d,]+) common site-hours \(([\d-]+) to ([\d-]+)\)", heading)
    if match is None:
        msg = f"{path}: cannot read the row count from {heading!r}"
        raise ValueError(msg)
    main = report_errors(report_path=path, column="Global only")
    main_losses = pl.read_parquet(path.with_name("losses.parquet")).filter(
        pl.col("setting") == "pooled"
    )
    keys = frame.select("site", "time")
    both = main_losses.join(keys, on=["site", "time"], how="inner")
    matched = both.select("site", "time").unique().height
    lines = [
        "#### Against the page's main row set",
        "",
        (
            f"The main row set holds {match[1]} common site-hours ({match[2]} to {match[3]}); this "
            f"section's row set holds {frame.height:,} ({frame['time'].min():%Y-%m-%d} to "
            f"{frame['time'].max():%Y-%m-%d}), of which {matched:,} are also in the main row set "
            f"and {frame.height - matched:,} are not."
        ),
        "",
        (
            "| Arm | This section | Main row set | Difference "
            f"| Main row set's fit, scored on the {matched:,} rows both row sets hold "
            "| This section minus that: the shorter training span |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for arm, name in (("era5_global", "era5"), ("cams_global", "cams")):
        here = _mae(losses=pooled, arm=arm)
        lines.append(
            f"| {arm} | {here:.3f} | {main[name]:.3f} | {here - main[name]:+.3f} "
            f"| {_mae(losses=both, arm=arm):.3f} | {here - _mae(losses=both, arm=arm):+.3f} |"
        )
    return lines


def _shared_rows_lines(*, pooled: pl.DataFrame) -> list[str]:
    """Render the planned contrasts on the rows this section shares with the main row set.

    Args:
        pooled: Every arm's losses at the `pooled` setting.

    Returns:
        Markdown lines.
    """
    path = OUTPUT_DIR.parent / "solar_long" / "losses.parquet"
    main_keys = pl.read_parquet(path, columns=["site", "time"]).unique()
    shared = pooled.join(main_keys, on=["site", "time"], how="inner")
    rows = shared.filter(pl.col("arm") == STATION_ARM).select("site", "time").unique().height
    return [
        "#### The planned contrasts on the rows the main row set also holds (exploratory)",
        "",
        *CONTRAST_HEADER,
        *(
            _contrast_line(losses=shared, treatment=t, reference=r, label=f"{rows:,} shared rows")
            for t, r in PLANNED_CONTRASTS
        ),
    ]


def _flag_lines(*, frame: pl.DataFrame, pooled: pl.DataFrame, selection: Selection) -> list[str]:
    """Render the hours with an unusual quality-control flag, and the planned contrasts without.

    Args:
        frame: This section's row set.
        pooled: Every arm's losses at the `pooled` setting.
        selection: The in-memory station choice, whose `unusual_flags` names the hours.

    Returns:
        Markdown lines.
    """
    unusual = selection.unusual_flags.join(
        frame.select("site", "time"), on=["site", "time"], how="inner"
    )
    kept = pooled.join(unusual, on=["site", "time"], how="anti")
    return [
        "#### The planned contrasts without the hours flagged unusually (exploratory, post hoc)",
        "",
        (
            "The flags are kept as delivered and no hour is dropped on its flag. The nearest "
            f"radiation station's flag differs from its usual value, {USUAL_FLAG}, at "
            f"{unusual.height:,} of the {frame.height:,} site-hours."
        ),
        "",
        *CONTRAST_HEADER,
        *(
            _contrast_line(losses=kept, treatment=t, reference=r, label="flag usual")
            for t, r in PLANNED_CONTRASTS
        ),
    ]


def _half_year_lines(*, pooled: pl.DataFrame) -> list[str]:
    """Render the planned contrasts for April to September and for October to March.

    Args:
        pooled: Every arm's losses at the `pooled` setting.

    Returns:
        Markdown lines.
    """
    summer = pl.col("month").str.slice(5, 2).cast(pl.Int32).is_between(4, 9)
    lines = [
        "#### The planned contrasts by half of the year (exploratory, post hoc)",
        "",
        *CONTRAST_HEADER,
    ]
    halves = (("April to September", summer), ("October to March", ~summer))
    for label, condition in halves:
        lines += [
            _contrast_line(losses=pooled.filter(condition), treatment=t, reference=r, label=label)
            for t, r in PLANNED_CONTRASTS
        ]
    months = [pooled.filter(condition)["month"].n_unique() for _, condition in halves]
    lines += [
        "",
        f"April to September holds {months[0]} calendar months and October to March {months[1]}.",
    ]
    return lines


def _lag_lines(*, frame: pl.DataFrame) -> list[str]:
    """Render how strongly output tracks the station's and CAMS's irradiance one hour either side.

    Output measured over the hour ending at `time` should track irradiance over the same hour more
    strongly than the hour before or after; a station stamped an hour off would peak elsewhere.

    Args:
        frame: This section's row set.

    Returns:
        Markdown lines.
    """
    fraction = frame.select(
        "site",
        "time",
        output=pl.col("power_mw").cast(pl.Float64) / pl.col("effective_capacity_mw"),
        station=pl.col(GHI.format(rank=1)),
        cams=pl.col("ghi_cams"),
    )
    shifted = fraction.select(
        "site",
        (pl.col("time") + pl.duration(hours=1)).alias("time"),
        station_before=pl.col("station"),
        cams_before=pl.col("cams"),
    ).join(
        fraction.select(
            "site",
            (pl.col("time") - pl.duration(hours=1)).alias("time"),
            station_after=pl.col("station"),
            cams_after=pl.col("cams"),
        ),
        on=["site", "time"],
        how="inner",
    )
    joined = fraction.join(shifted, on=["site", "time"], how="inner")
    lines = [
        "#### Output against irradiance one hour before, the same hour, and one hour after",
        "",
        f"Correlation on the {joined.height:,} site-hours with both neighbouring hours scored.",
        "",
        "| Irradiance from | Hour before | Same hour | Hour after |",
        "|---|---|---|---|",
    ]
    for name in ("station", "cams"):
        cells = [
            f"{np.corrcoef(joined['output'].to_numpy(), joined[column].to_numpy())[0, 1]:.3f}"
            for column in (f"{name}_before", name, f"{name}_after")
        ]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    return lines


def _report(
    *,
    frame: pl.DataFrame,
    losses: pl.DataFrame,
    sites: pl.DataFrame,
    job_list: list[Job],
    selection: Selection,
    repairs: dict[str, int],
    candidates: int,
) -> str:
    """Assemble the markdown report.

    Args:
        frame: This section's own row set.
        losses: Every arm's losses, at both `pooled` and `sensitivity` settings.
        sites: The solar roster, for the geometry lines.
        job_list: Every job `jobs()` returns, for the feature-column section.
        selection: The in-memory station choice, reported only as pooled ranges.
        repairs: The reader's repair counts.
        candidates: The common rows before the station intersection.

    Returns:
        The report.
    """
    pooled = losses.filter(pl.col("setting") == "pooled")
    sensitivity = losses.filter(pl.col("setting") == "sensitivity")
    site_labels = sorted(frame["site"].unique().to_list())
    lines = [
        (
            f"### Station observations on {frame.height:,} common site-hours of solar "
            f"({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d})"
        ),
        "",
        *_absolute_table_lines(pooled=pooled),
        "",
        (
            "Mean absolute error as a percentage of each site's P99 output. The interval is a 95% "
            "bound from resampling whole months and a fitting seed."
        ),
        "",
        *_arm_columns_lines(job_list=job_list),
        "",
        "#### Planned contrasts",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(losses=pooled, treatment=t, reference=r, label="all")
        for t, r in PLANNED_CONTRASTS
    ]
    lines += [
        "",
        "#### Planned contrasts at the second hyperparameter setting",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(losses=sensitivity, treatment=t, reference=r, label="sensitivity")
        for t, r in PLANNED_CONTRASTS
    ]
    lines += ["", "#### The planned contrasts, per generator (exploratory)", "", *CONTRAST_HEADER]
    for treatment, reference in PLANNED_CONTRASTS:
        lines += [
            _contrast_line(
                losses=pooled.filter(pl.col("site") == site),
                treatment=treatment,
                reference=reference,
                label=f"site {site}",
            )
            for site in site_labels
        ]
    lines += ["", "#### Exploratory contrasts", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(losses=pooled, treatment=t, reference=r, label="all")
        for t, r in EXPLORATORY_CONTRASTS
    ]
    nearest_low, nearest_high = selection.nearest_distance_range_km()
    lines += [
        "",
        (
            f"The nearest radiation station is {nearest_low:{KM_FORMAT}} to "
            f"{nearest_high:{KM_FORMAT}} km from a generator."
        ),
        "",
        *selection.pooled_lines(),
        "",
        *_files_lines(),
        "",
        *_row_lines(frame=frame, candidates=candidates, repairs=repairs),
        "",
        *_hour_profile_lines(frame=frame),
        "",
        *_agreement_lines(frame=frame),
        "",
        *_main_panel_lines(pooled=pooled, frame=frame),
        "",
        *_shared_rows_lines(pooled=pooled),
        "",
        *_flag_lines(frame=frame, pooled=pooled, selection=selection),
        "",
        *_half_year_lines(pooled=pooled),
        "",
        *_lag_lines(frame=frame),
        "",
    ]
    lines += geometry_lines(sites=sites, noun="solar farms")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build the row set, fit every arm, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Fit nothing; rebuild report.md from the saved losses.parquet alone.",
    )
    arguments = parser.parse_args()

    sites = _pv_sites()
    frame, selection, repairs, candidates = build_rows()
    _LOG.info("%d rows, %s to %s", frame.height, frame["time"].min(), frame["time"].max())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "losses.parquet"
    fingerprint_path = OUTPUT_DIR / "losses.fingerprint"
    report_path = OUTPUT_DIR / "report.md"

    all_jobs = jobs()
    fingerprint = _fingerprint(frame=frame, job_list=all_jobs)

    if arguments.report_only:
        saved_fingerprint = (
            fingerprint_path.read_text().strip() if fingerprint_path.exists() else None
        )
        if saved_fingerprint != fingerprint:
            msg = (
                f"--report-only: {path} was fitted on a different row set, column set, seed set, "
                "or hyperparameter setting than this code now produces; re-run without "
                "--report-only"
            )
            raise ValueError(msg)
        losses = pl.read_parquet(path)
    else:
        refuse_to_overwrite(paths=[path, fingerprint_path, report_path])
        losses = run_all(dataset=frame, jobs=all_jobs, max_workers=MAX_CONCURRENT_FITS)
        losses.write_parquet(path)
        fingerprint_path.write_text(fingerprint)

    report = _report(
        frame=frame,
        losses=losses,
        sites=sites,
        job_list=all_jobs,
        selection=selection,
        repairs=repairs,
        candidates=candidates,
    )
    report_path.write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
