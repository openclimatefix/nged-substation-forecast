"""Find why the local ECMWF ENS mean scores worse than Open-Meteo's, and what the leads can explain.

One-off follow-up to `ensemble_means_mae.py`, whose report shows the local ECMWF ENS series (this
repository's 51-member table, one 00 UTC run a day) scoring worse than Open-Meteo's ECMWF ENS 0.25
degree mean: about 1.2 points of capacity for solar and 0.2 to 0.3 for wind. **The comparison is
descriptive.** It states errors over the same 88-day summer window and puts no interval on any
difference.

**What it reads.** The rows, folds, power, and every product's columns come from the saved frames of
the earlier run (`data/studies/open_meteo_ensemble_means/frame_*.parquet`), which this script never
writes. It adds columns, each built for the same generator-hours:

- `local_kt` (solar): the local ENS mean with its 3-hour irradiance interpolated through the
  clearness index instead of held flat over each step
  (`studies.stitched_ensemble.interpolate_clearness_hourly`). Open-Meteo serves hourly values, so
  the hold is one candidate cause of the gap.
- `local_dayold` (solar and wind): the local ENS mean built from the newest run that is at least 27
  hours ahead of the valid time, so each step comes from a run about a day older than the stored
  series. The increase in error from using an older run bounds how much of the gap a
  difference in lead can explain.
- Deterministic Open-Meteo runs from `data/studies/weather/<model>/previous_runs/combined.parquet`
  (ICON-D2, ICON-EU, ECMWF IFS 0.25 degree, UKV) at `previous_day0`, the freshest run, and
  `previous_day1`, the run 24 hours older. The two days give each model's error at a short lead and
  at a lead a day longer, which is the only run-age evidence in these data. Each ensemble mean is
  also compared with its deterministic sibling, and the distance to `day0` against the distance to
  `day1` says which run age the ensemble mean sits closer to.

**Arms and fits.** Each arm is one product's column or columns, given to the XGBoost model of the
earlier study at its primary setting, three seeds, the same shared features, the same folds, and
the same rows. Rows where any added column is missing are dropped from every arm, and the report
gives the count. Solar arms read one irradiance column and wind arms one 10 m speed column; the
hub-height design reads the 100 m and 10 m speeds. Every arm of a design has the same number of
columns.

**What it tabulates besides power error.** For every product: the mean absolute error against CAMS
(solar) or ERA5 (wind) at the hourly resolution and, for solar, over the 3-hour windows that the
local steps span; the local series' difference from Open-Meteo's ECMWF ENS mean over the same
windows; and each series' power error by the 3-hour step of the day, beside the mean lead of the
local steps behind that part of the day. Because the local series comes from one 00 UTC run a day,
its lead is fixed by the hour of day, so the by-step table shows how the gap between the two series
changes with local lead.

Run it with `uv run python studies/open_meteo_ensemble_means/local_ens_gap.py`. It writes to a new
folder, `data/studies/open_meteo_ens_gap/`, and refuses to overwrite. It refuses to fit while the
one-minute load average is above `LOAD_LIMIT`. `--report-only` rebuilds the tables and `report.md`
from the saved frames and losses. Fits run on the CPU.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import ensemble_means_mae as earlier
import polars as pl
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS, SEEDS
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.stitched_ensemble import (
    STEP_HOURS,
    hold_backward_mean_hourly,
    interpolate_clearness_hourly,
    interpolate_instants_hourly,
    newest_run_member_means,
)

from build_dataset import _add_solar_geometry, _pv_sites  # isort: skip
from run_experiment import Job, run_all  # isort: skip
from sources import STUDIES_DATA_DIR, WEATHER_DATA_DIR  # isort: skip

_LOG: Final[logging.Logger] = logging.getLogger("local_ens_gap")

OUTPUT_DIR: Final[Path] = STUDIES_DATA_DIR / "open_meteo_ens_gap"
"""Where this script writes. A re-run needs the old files moved to `superseded/` by hand."""

LOAD_LIMIT: Final[float] = 24.0
"""The one-minute load average above which the script refuses to fit."""

DAY_OLD_MIN_LEAD_HOURS: Final[int] = 27
"""The shortest lead the `local_dayold` series may use: three hours past a day."""

LOCAL_KT: Final[str] = "local_kt"
LOCAL_DAY_OLD: Final[str] = "local_dayold"
ENS_OPEN_METEO: Final[str] = "ecmwf_ens_025_mean"
LOCAL: Final[str] = earlier.LOCAL_ENS

DETERMINISTIC_MODELS: Final[dict[str, str]] = {
    "icon_d2": "ICON-D2",
    "icon_eu": "ICON-EU",
    "ifs025": "ECMWF-IFS-025",
    "ukv": "UKV",
}
"""Each deterministic model's key, and its folder under `data/studies/weather/`."""

RUN_AGES: Final[tuple[int, ...]] = (0, 1)
"""The `previous_day` values read: 0 is the freshest run, 1 the run 24 hours older."""

SIBLINGS: Final[dict[str, str]] = {
    "icon_d2_eps_mean": "icon_d2",
    "icon_eu_eps_mean": "icon_eu",
    ENS_OPEN_METEO: "ifs025",
    LOCAL: "ifs025",
    "mogreps_uk_mean": "ukv",
}
"""Each ensemble mean, and the deterministic model that shares its physics."""

DesignType = earlier.DesignType
DomainType = earlier.DomainType

HUB_PRODUCTS: Final[tuple[str, ...]] = (
    ENS_OPEN_METEO,
    LOCAL,
    LOCAL_DAY_OLD,
    "icon_d2_eps_mean",
    "icon_d2_det0",
    "icon_d2_det1",
    "ifs025_det0",
    "ifs025_det1",
    earlier.WIND_REFERENCE,
)
"""The wind products with a 100 m speed, for the hub-height design."""

HUB_DETERMINISTIC_MODELS: Final[frozenset[str]] = frozenset(
    product.removesuffix("_det0").removesuffix("_det1")
    for product in HUB_PRODUCTS
    if "_det" in product
)
"""The deterministic models whose 100 m speed the hub-height design reads. Other models' 100 m
speeds are not read, so a null in one cannot shrink the rows."""

COMPARED_ARMS: Final[tuple[str, ...]] = (ENS_OPEN_METEO, LOCAL, LOCAL_KT, LOCAL_DAY_OLD)
"""The arms the by-step and agreement tables show."""

HOURS_PER_DAY: Final[int] = 24
STORED_TOLERANCE_W_M2: Final[float] = 0.01
"""The largest difference between the rebuilt and saved local series: Float32 rounding."""

RESULT_TABLES: Final[tuple[str, ...]] = (
    "mae_by_arm",
    "mae_by_step",
    "weather_error",
    "agreement",
    "sibling_distance",
)
"""The names of the result tables `_analyse` builds."""

WITHIN_W_M2: Final[float] = 5.0
"""How close two irradiance series must be over a 3-hour window to count as agreeing."""


def _det_products() -> list[str]:
    return [f"{model}_det{age}" for model in DETERMINISTIC_MODELS for age in RUN_AGES]


def _product_lists() -> dict[DesignType, list[str]]:
    """Return every product of each design, the reference last."""
    open_meteo = list(earlier.OPEN_METEO_PRODUCTS)
    return {
        "solar": [
            *open_meteo,
            LOCAL,
            LOCAL_KT,
            LOCAL_DAY_OLD,
            *_det_products(),
            earlier.SOLAR_REFERENCE,
        ],
        "wind_10m": [
            *open_meteo,
            LOCAL,
            LOCAL_DAY_OLD,
            *_det_products(),
            earlier.WIND_REFERENCE,
        ],
        "wind_hub": list(HUB_PRODUCTS),
    }


def jobs(*, design: DesignType) -> list[Job]:
    """Return one primary-setting job per arm of a design.

    Args:
        design: The design.

    Returns:
        One job per arm, the target `power_mw`, no quantile model.

    Raises:
        ValueError: Unless every arm of the design carries the same number of columns.
    """
    shared = earlier.SOLAR_SHARED_FEATURES if design == "solar" else earlier.WIND_SHARED_FEATURES
    arms = {
        product: (*shared, *earlier._feature_columns(design=design, product=product))
        for product in _product_lists()[design]
    }
    if len({len(columns) for columns in arms.values()}) != 1:
        msg = f"{design}: every arm should carry the same number of columns"
        raise ValueError(msg)
    return [
        (f"{design}:{product}", "primary", "power_mw", columns, PRIMARY_HYPER_PARAMETERS, False)
        for product, columns in arms.items()
    ]


def _previous_runs(*, model: str, domain: DomainType) -> pl.DataFrame:
    """Read one deterministic model's freshest and one-day-older runs from the window start.

    Args:
        model: A key of `DETERMINISTIC_MODELS`.
        domain: `solar` reads irradiance; `wind` reads the 10 m and 100 m speeds.

    Returns:
        One row per (site, hour) with a column per variable and run age.
    """
    fields = {"shortwave_radiation": "ghi"} if domain == "solar" else {"wind_speed_10m": "speed10"}
    if domain == "wind" and model in HUB_DETERMINISTIC_MODELS:
        fields["wind_speed_100m"] = "speed100"
    path = WEATHER_DATA_DIR / DETERMINISTIC_MODELS[model] / "previous_runs" / "combined.parquet"
    columns = {
        (source if age == 0 else f"{source}_previous_day{age}"): f"{name}_{model}_det{age}"
        for source, name in fields.items()
        for age in RUN_AGES
    }
    return (
        pl.read_parquet(path)
        .select("site", "time", *(pl.col(source).alias(name) for source, name in columns.items()))
        .filter(pl.col("time") >= earlier.WINDOW_START)
    )


def _hourly_extraterrestrial(*, first: datetime, last: datetime) -> pl.DataFrame:
    """Return each solar site's extraterrestrial irradiance for every hour label in a range.

    Args:
        first: The first hour label.
        last: The last hour label.

    Returns:
        One row per (site, hour label) with `extraterrestrial_horizontal_w_m2`. The site
        coordinates are used inside this function and never returned.
    """
    sites = _pv_sites().select("site", "latitude", "longitude")
    hours = pl.DataFrame(
        {"time": pl.datetime_range(first, last, "1h", time_zone="UTC", eager=True)}
    )
    grid = sites.join(hours, how="cross")
    return _add_solar_geometry(joined=grid).select(
        "site", "time", "extraterrestrial_horizontal_w_m2"
    )


def _step_leads(*, steps: pl.DataFrame, variant: str) -> pl.DataFrame:
    """Summarise the lead behind each 3-hour step of the day.

    Args:
        steps: The stitched steps, carrying `valid_time` and `init_time`.
        variant: The series' name.

    Returns:
        One row per step-ending hour of day with the mean lead in hours and the step count.
    """
    return (
        steps.with_columns(
            step_end_hour=pl.col("valid_time").dt.hour(),
            lead_hours=(pl.col("valid_time") - pl.col("init_time")).dt.total_hours(),
        )
        .group_by("step_end_hour")
        .agg(mean_lead_hours=pl.col("lead_hours").mean(), n_steps=pl.len())
        .with_columns(variant=pl.lit(variant))
        .sort("step_end_hour")
    )


def _local_solar_variants(*, frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the `local_kt` and `local_dayold` irradiance series and their lead summaries.

    Args:
        frame: The earlier study's solar rows, whose local series the rebuilt stored series must
            equal.

    Returns:
        One row per (site, hour) with `ghi_local_kt` and `ghi_local_dayold`, and the lead summary
        for the stored series and the day-old series.

    Raises:
        ValueError: If the rebuilt stored series differs from the frame's, or a day-old step's run
            is not exactly 24 hours older than the stored step's.
    """
    members = earlier._local_ens_members(
        path=earlier.ENS_DIR / "beam_diffuse_ens.parquet", value_columns=["ghi_w_m2"]
    )
    settings = {
        "expected_members": earlier.EXPECTED_MEMBERS,
        "max_lead_hours": earlier.LOCAL_ENS_MAX_LEAD_HOURS,
        "value_columns": ["ghi_w_m2"],
    }
    stored = newest_run_member_means(members=members, **settings)
    day_old = newest_run_member_means(
        members=members.filter(pl.col("lead_hours") >= DAY_OLD_MIN_LEAD_HOURS), **settings
    )
    window = frame.select(first=pl.col("time").min(), last=pl.col("time").max())
    _check_day_old(
        stored=stored.filter(
            pl.col("valid_time").is_between(
                window["first"][0], window["last"][0] + timedelta(hours=STEP_HOURS)
            )
        ),
        day_old=day_old,
    )
    rebuilt = hold_backward_mean_hourly(steps=stored, value_columns=["ghi_w_m2"])
    _check_stored_matches(rebuilt=rebuilt, frame=frame)
    extraterrestrial = _hourly_extraterrestrial(
        first=stored.select(pl.col("valid_time").min()).item() - timedelta(hours=STEP_HOURS),
        last=stored.select(pl.col("valid_time").max()).item(),
    )
    kt = interpolate_clearness_hourly(
        steps=stored, extraterrestrial_hourly=extraterrestrial, value_column="ghi_w_m2"
    ).rename({"ghi_w_m2": f"ghi_{LOCAL_KT}"})
    held = hold_backward_mean_hourly(steps=day_old, value_columns=["ghi_w_m2"]).rename(
        {"ghi_w_m2": f"ghi_{LOCAL_DAY_OLD}"}
    )
    leads = pl.concat(
        [
            _step_leads(steps=stored, variant=LOCAL),
            _step_leads(steps=day_old, variant=LOCAL_DAY_OLD),
        ]
    )
    return kt.join(held, on=["site", "time"], how="full", coalesce=True), leads


def _check_day_old(*, stored: pl.DataFrame, day_old: pl.DataFrame) -> None:
    """Raise unless every day-old step's run is exactly 24 hours older than the stored step's.

    Args:
        stored: The stored series' steps inside the scored window, with `init_time`. The series
            also runs past both ends of the window, where a day-old partner may not exist.
        day_old: The day-old series' steps, with `init_time`.

    Raises:
        ValueError: Naming how many steps differ from 24 hours, or lack a partner.
    """
    gaps = stored.join(day_old, on=["site", "valid_time"], how="left", suffix="_day_old").select(
        gap_hours=(pl.col("init_time") - pl.col("init_time_day_old")).dt.total_hours()
    )
    wrong = gaps.filter(pl.col("gap_hours").is_null() | (pl.col("gap_hours") != HOURS_PER_DAY))
    if not wrong.is_empty():
        msg = f"{wrong.height} steps' day-old run is not exactly {HOURS_PER_DAY} hours older"
        raise ValueError(msg)


def _check_stored_matches(*, rebuilt: pl.DataFrame, frame: pl.DataFrame) -> None:
    """Raise unless the rebuilt stored series equals the earlier study's local series.

    Args:
        rebuilt: The rebuilt stored series, `ghi_w_m2` per (site, hour).
        frame: The earlier study's rows, carrying `ghi_ecmwf_ens_local_mean`.

    Raises:
        ValueError: If any row of the frame lacks a rebuilt value or differs by more than the
            Float32 rounding of the frame's column.
    """
    joined = frame.select("site", "time", f"ghi_{LOCAL}").join(
        rebuilt, on=["site", "time"], how="left"
    )
    difference = (pl.col(f"ghi_{LOCAL}") - pl.col("ghi_w_m2")).abs()
    if joined.filter(difference.is_null() | (difference > STORED_TOLERANCE_W_M2)).height:
        msg = "the rebuilt stored series differs from the earlier study's local series"
        raise ValueError(msg)


def _local_wind_day_old() -> pl.DataFrame:
    """Build the `local_dayold` wind series, linearly interpolated onto hours, in km/h.

    Returns:
        One row per (site, hour) with `speed10_local_dayold` and `speed100_local_dayold`.
    """
    columns = ["speed_10m_ms", "speed_100m_ms"]
    members = earlier._local_ens_members(
        path=earlier.ENS_DIR / "beam_diffuse_ens_wind.parquet", value_columns=columns
    )
    steps = newest_run_member_means(
        members=members.filter(pl.col("lead_hours") >= DAY_OLD_MIN_LEAD_HOURS),
        value_columns=columns,
        expected_members=earlier.EXPECTED_MEMBERS,
        max_lead_hours=earlier.LOCAL_ENS_MAX_LEAD_HOURS,
    )
    return interpolate_instants_hourly(steps=steps, value_columns=columns).select(
        "site",
        "time",
        pl.col("speed_10m_ms").mul(earlier.MS_TO_KM_PER_H).alias(f"speed10_{LOCAL_DAY_OLD}"),
        pl.col("speed_100m_ms").mul(earlier.MS_TO_KM_PER_H).alias(f"speed100_{LOCAL_DAY_OLD}"),
    )


def _with_columns_from(
    *, frame: pl.DataFrame, extras: list[pl.DataFrame]
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Join extra columns onto a frame, then drop the rows where any of them is missing.

    Args:
        frame: The earlier study's rows.
        extras: Frames keyed by `site` and `time`, carrying the new columns.

    Returns:
        The frame without the rows that lack a new column, and the row counts before and after.
    """
    joined = frame
    added: list[str] = []
    for extra in extras:
        added += [name for name in extra.columns if name not in ("site", "time")]
        joined = joined.join(extra, on=["site", "time"], how="left")
    nulls = {name: joined[name].null_count() for name in added}
    _LOG.info("null counts of the added columns before the drop: %s", nulls)
    print(f"null counts of the added columns before the drop: {nulls}")
    kept = joined.filter(pl.all_horizontal(pl.col(added).is_not_null()))
    return kept, {"rows_before": joined.height, "rows_after": kept.height}


def build_frames() -> tuple[
    dict[DomainType, pl.DataFrame], pl.DataFrame, dict[str, dict[str, int]]
]:
    """Build each domain's rows with the added columns.

    Returns:
        The frames by domain, the lead summaries, and each domain's row counts before and after
        the rows lacking an added column are dropped.
    """
    solar_frame = pl.read_parquet(earlier.OUTPUT_DIR / "frame_solar.parquet")
    wind_frame = pl.read_parquet(earlier.OUTPUT_DIR / "frame_wind.parquet")
    local_solar, leads = _local_solar_variants(frame=solar_frame)
    solar, solar_counts = _with_columns_from(
        frame=solar_frame,
        extras=[
            local_solar,
            *(_previous_runs(model=m, domain="solar") for m in DETERMINISTIC_MODELS),
        ],
    )
    wind, wind_counts = _with_columns_from(
        frame=wind_frame,
        extras=[
            _local_wind_day_old(),
            *(_previous_runs(model=m, domain="wind") for m in DETERMINISTIC_MODELS),
        ],
    )
    return {"solar": solar, "wind": wind}, leads, {"solar": solar_counts, "wind": wind_counts}


def _check_load() -> None:
    """Raise if the machine is too busy to fit on.

    Raises:
        RuntimeError: If the one-minute load average is above `LOAD_LIMIT`.
    """
    load = os.getloadavg()[0]
    if load > LOAD_LIMIT:
        msg = f"one-minute load average {load:.1f} is above {LOAD_LIMIT}; wait and re-run"
        raise RuntimeError(msg)


def _fit(*, frames: dict[DomainType, pl.DataFrame]) -> dict[DesignType, pl.DataFrame]:
    """Fit every design's arms, checking the columns each arm reads first.

    Args:
        frames: Each domain's rows.

    Returns:
        Each design's losses, `arm` labelled `<design>:<product>`.
    """
    _check_load()
    losses: dict[DesignType, pl.DataFrame] = {}
    for design in ("solar", "wind_10m", "wind_hub"):
        frame = frames[earlier._domain_of(design=design)]
        design_jobs = jobs(design=design)
        check_no_missing(frame=frame, columns=[c for job in design_jobs for c in job[3]])
        losses[design] = run_all(
            dataset=frame, jobs=design_jobs, max_workers=earlier.MAX_CONCURRENT_FITS
        )
        earlier._check_same_rows(losses=losses[design])
    return losses


def _step_end_hour(*, column: str = "time") -> pl.Expr:
    """Return the hour of day at which the 3-hour step holding an hour label ends."""
    hour = pl.col(column).dt.hour()
    return ((hour + STEP_HOURS - 1) // STEP_HOURS * STEP_HOURS % HOURS_PER_DAY).alias(
        "step_end_hour"
    )


def _mae_by_arm(*, losses: dict[DesignType, pl.DataFrame]) -> pl.DataFrame:
    """Return every arm's error against Open-Meteo's ECMWF ENS mean, per design."""
    return pl.concat(
        [
            earlier._absolute_table(
                losses=design_losses, reference=ENS_OPEN_METEO, design=design
            ).drop("setting")
            for design, design_losses in losses.items()
        ]
    ).sort("design", "mae_pct")


def _mae_by_step(*, losses: dict[DesignType, pl.DataFrame]) -> pl.DataFrame:
    """Return the compared arms' power error by 3-hour step of the day, for each design."""
    parts = []
    for design, design_losses in losses.items():
        arms = [f"{design}:{arm}" for arm in COMPARED_ARMS]
        parts.append(
            design_losses.filter(pl.col("arm").is_in(arms))
            .with_columns(_step_end_hour(), design=pl.lit(design))
            .group_by("design", "arm", "step_end_hour")
            .agg(
                mae_pct=pl.col(earlier.METRIC).mean() * earlier.PERCENT,
                n_rows=pl.len() // len(SEEDS),
            )
            .with_columns(arm=pl.col("arm").str.split(":").list.last())
        )
    return pl.concat(parts).sort("design", "step_end_hour", "arm")


def _weather_errors(*, frames: dict[DomainType, pl.DataFrame]) -> pl.DataFrame:
    """Measure every product against the reference at hourly and 3-hour resolution.

    Args:
        frames: Each domain's rows.

    Returns:
        One row per (design, product, resolution) with the mean absolute and signed error and the
        row count. The 3-hour rows exist for solar only and use windows whose three hours are all
        present.
    """
    rows = []
    products = _product_lists()
    for design, prefix, reference in (
        ("solar", "ghi", earlier.SOLAR_REFERENCE),
        ("wind_10m", "speed10", earlier.WIND_REFERENCE),
        ("wind_hub", "speed100", earlier.WIND_REFERENCE),
    ):
        frame = frames[earlier._domain_of(design=design)]
        resolutions = {"hourly": frame}
        if design == "solar":
            windows = (
                frame.with_columns(_step_end_hour(), day=pl.col("time").dt.truncate("1d"))
                .group_by("site", "day", "step_end_hour")
                .agg(
                    pl.len().alias("n_hours"),
                    *(pl.col(f"{prefix}_{p}").mean() for p in products[design]),
                )
                .filter(pl.col("n_hours") == STEP_HOURS)
            )
            resolutions["3-hour windows"] = windows
        for resolution, data in resolutions.items():
            for product in products[design]:
                if product == reference:
                    continue
                difference = pl.col(f"{prefix}_{product}") - pl.col(f"{prefix}_{reference}")
                rows.append(
                    data.select(
                        design=pl.lit(design),
                        product=pl.lit(product),
                        resolution=pl.lit(resolution),
                        mean_absolute_error=difference.abs().mean(),
                        mean_signed_error=difference.mean(),
                        n_rows=pl.len(),
                    )
                )
    return pl.concat(rows).sort("design", "resolution", "mean_absolute_error")


def _agreement(*, solar: pl.DataFrame) -> pl.DataFrame:
    """Compare the local series with Open-Meteo's ECMWF ENS mean over 3-hour windows.

    Args:
        solar: The solar rows.

    Returns:
        One row per (series, step-ending hour): the windows' count, the mean signed difference from
        Open-Meteo's mean, the mean and median absolute difference, and the share of windows within
        `WITHIN_W_M2`. The series are `local` and `local_kt` against Open-Meteo, and `local_kt`
        against `local`, which shows how far the interpolation departs from its step means.
    """
    windows = (
        solar.with_columns(_step_end_hour(), day=pl.col("time").dt.truncate("1d"))
        .group_by("site", "day", "step_end_hour")
        .agg(
            pl.len().alias("n_hours"),
            pl.col(f"ghi_{ENS_OPEN_METEO}").mean().alias("open_meteo"),
            pl.col(f"ghi_{LOCAL}").mean().alias(LOCAL),
            pl.col(f"ghi_{LOCAL_KT}").mean().alias(LOCAL_KT),
        )
        .filter(pl.col("n_hours") == STEP_HOURS)
    )
    parts = []
    comparisons = {
        LOCAL: (LOCAL, "open_meteo"),
        LOCAL_KT: (LOCAL_KT, "open_meteo"),
        f"{LOCAL_KT}_vs_{LOCAL}": (LOCAL_KT, LOCAL),
    }
    for series, (compared, baseline) in comparisons.items():
        difference = pl.col(compared) - pl.col(baseline)
        parts.append(
            windows.group_by("step_end_hour")
            .agg(
                n_windows=pl.len(),
                mean_signed_difference=difference.mean(),
                mean_absolute_difference=difference.abs().mean(),
                median_absolute_difference=difference.abs().median(),
                share_within=(difference.abs() <= WITHIN_W_M2).mean(),
            )
            .with_columns(series=pl.lit(series))
        )
    return pl.concat(parts).sort("series", "step_end_hour")


def _sibling_distance(*, frames: dict[DomainType, pl.DataFrame]) -> pl.DataFrame:
    """Measure how far each ensemble mean sits from its deterministic sibling at two run ages.

    Args:
        frames: Each domain's rows.

    Returns:
        One row per (variable, ensemble mean): the mean absolute distance to the sibling's freshest
        run, to its run a day older, and the ratio of the second to the first.
    """
    rows = []
    for prefix, domain in (("ghi", "solar"), ("speed10", "wind")):
        for ensemble, sibling in SIBLINGS.items():
            fresh = pl.col(f"{prefix}_{ensemble}") - pl.col(f"{prefix}_{sibling}_det0")
            older = pl.col(f"{prefix}_{ensemble}") - pl.col(f"{prefix}_{sibling}_det1")
            rows.append(
                frames[domain].select(
                    variable=pl.lit(prefix),
                    ensemble=pl.lit(ensemble),
                    sibling=pl.lit(sibling),
                    distance_to_freshest=fresh.abs().mean(),
                    distance_to_day_older=older.abs().mean(),
                    n_rows=pl.len(),
                )
            )
    return pl.concat(rows).with_columns(
        ratio=pl.col("distance_to_day_older") / pl.col("distance_to_freshest")
    )


def _paths() -> dict[str, Path]:
    """Return every file the script writes, by name."""
    names = ["report.md", "run_facts.json", "local_leads.parquet"]
    names += [f"{name}.parquet" for name in RESULT_TABLES]
    names += [f"frame_{domain}.parquet" for domain in ("solar", "wind")]
    names += [f"losses_{design}.parquet" for design in ("solar", "wind_10m", "wind_hub")]
    return {name: OUTPUT_DIR / name for name in names}


def _report(
    *,
    tables: dict[str, pl.DataFrame],
    leads: pl.DataFrame,
    counts: dict[str, dict[str, int]],
) -> str:
    """Render every number the study page quotes, as markdown.

    Args:
        tables: The result tables by name.
        leads: The mean lead of the local steps.
        counts: Each domain's row counts before and after the missing rows drop.

    Returns:
        The report.
    """
    table = earlier._markdown_table
    by_step = tables["mae_by_step"].pivot(
        on="arm", index=["design", "step_end_hour"], values="mae_pct"
    )
    by_step = by_step.select(
        column for column in by_step.columns if by_step[column].null_count() < by_step.height
    ).with_columns(local_minus_open_meteo=pl.col(LOCAL) - pl.col(ENS_OPEN_METEO))
    lines = ["# Local ECMWF ENS against Open-Meteo's ECMWF ENS mean", "", "## Rows", ""]
    lines += [
        f"- {domain}: {c['rows_before']} rows before the drop, {c['rows_after']} after"
        for domain, c in counts.items()
    ]
    sections = [
        ("Power error by arm, % of capacity (primary setting)", tables["mae_by_arm"]),
        ("Mean lead of the local steps, by step-ending hour (UTC)", leads),
        ("Power error by 3-hour step of the day, % of capacity", by_step),
        ("Weather error against the reference", tables["weather_error"]),
        ("Local series against Open-Meteo's ECMWF ENS mean, 3-hour windows", tables["agreement"]),
        (
            "Distance of each ensemble mean from its deterministic sibling",
            tables["sibling_distance"],
        ),
    ]
    for title, frame in sections:
        lines += ["", f"## {title}", "", *table(frame=frame)]
    return "\n".join(lines) + "\n"


def _analyse(
    *, frames: dict[DomainType, pl.DataFrame], losses: dict[DesignType, pl.DataFrame]
) -> dict[str, pl.DataFrame]:
    """Compute every result table."""
    return {
        "mae_by_arm": _mae_by_arm(losses=losses),
        "mae_by_step": _mae_by_step(losses=losses),
        "weather_error": _weather_errors(frames=frames),
        "agreement": _agreement(solar=frames["solar"]),
        "sibling_distance": _sibling_distance(frames=frames),
    }


def _write_results(
    *,
    frames: dict[DomainType, pl.DataFrame],
    losses: dict[DesignType, pl.DataFrame],
    leads: pl.DataFrame,
    counts: dict[str, dict[str, int]],
) -> None:
    """Write the result tables and the report from the frames and losses."""
    paths = _paths()
    tables = _analyse(frames=frames, losses=losses)
    for name, result in tables.items():
        result.write_parquet(paths[f"{name}.parquet"])
    report = _report(tables=tables, leads=leads, counts=counts)
    paths["report.md"].write_text(report)
    _LOG.info("wrote %s", OUTPUT_DIR)
    print(report)


def main() -> int:
    """Build the added columns, fit every arm, and write the tables and the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    paths = _paths()
    designs: tuple[DesignType, ...] = ("solar", "wind_10m", "wind_hub")
    if arguments.report_only:
        results = ["report.md", *(f"{name}.parquet" for name in RESULT_TABLES)]
        refuse_to_overwrite(paths=[paths[name] for name in results])
        saved_frames: dict[DomainType, pl.DataFrame] = {
            "solar": pl.read_parquet(paths["frame_solar.parquet"]),
            "wind": pl.read_parquet(paths["frame_wind.parquet"]),
        }
        _write_results(
            frames=saved_frames,
            losses={d: pl.read_parquet(paths[f"losses_{d}.parquet"]) for d in designs},
            leads=pl.read_parquet(paths["local_leads.parquet"]),
            counts=json.loads(paths["run_facts.json"].read_text())["counts"],
        )
        return 0
    refuse_to_overwrite(paths=paths.values())
    frames, leads, counts = build_frames()
    for frame in frames.values():
        earlier._check_anonymous(frame=frame)
    losses = _fit(frames=frames)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for domain, frame in frames.items():
        frame.write_parquet(paths[f"frame_{domain}.parquet"])
    for design, design_losses in losses.items():
        design_losses.write_parquet(paths[f"losses_{design}.parquet"])
    leads.write_parquet(paths["local_leads.parquet"])
    paths["run_facts.json"].write_text(json.dumps({"counts": counts}, indent=2))
    _write_results(frames=frames, losses=losses, leads=leads, counts=counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
