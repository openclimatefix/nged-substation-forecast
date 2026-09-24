"""Score power forecasts from every planned and exploratory weather product, at matched leads.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>. Reads the per-technology
arm-input frames `build_forecast_inputs.py` writes, assigns folds, fits every arm out of fold,
brackets each Previous Runs product against ECMWF ENS, blends the open-data forecasts against ENS
alone, and writes `report.md`. `plans/nwp-forecast-comparison.md` holds the design.

**Rows.** `rows()` reads `<domain>_forecast_inputs.parquet`, keeps 2024-12-01 onwards (the first
whole month after IFS Cycle 49r1), drops the part-month that straddles the UKV PS47 upgrade
(2026-01), and keeps only the rows where the target, the baselines' inputs, and every *planned*
arm's columns are present. An exploratory arm is fitted on the same rows with its own small gaps
left as missing values, which XGBoost routes natively.

**Folds.** `assign_folds_with_eras()` cuts five folds of whole months inside three eras using
`studies.cross_validation.cut_eras`, which PR #885 adds. TODO(#885): the import lives inside the
fold-assigning functions until #885 is merged.

**Arms.** `arm_columns()` gives every arm's feature columns: the calendar columns, the sun-position
columns for solar, and each named product's own weather fields. `jobs()` builds one `Job` per
(arm, hyperparameter setting): the planned contrasts' arms are fitted at both settings, every other
arm once, at the primary setting. A product whose columns are absent from the input frame (GEFS,
while its download is unvalidated) contributes no job.

**Contrasts.** Every contrast is computed within one hyperparameter setting. A planned verdict
stands only if both settings give it (`studies.bootstrap.combine_setting_verdicts`). A bracket
against ENS rests on ENS's error not falling as its lead rises, checked per 3-hour band by
`check_ens_monotonicity`; a band where it fails is voided and its own bracket is reported.

**Modes.** `--dry-run` builds rows, folds and jobs, prints the coverage table and the job list, and
stops. The default fits every job and saves `<domain>_losses.parquet` and
`<domain>_predictions.parquet`. `--fit-missing` fits only the (arm, setting) pairs the saved losses
lack. `--report-only` writes `report.md` from the saved losses. `--synthetic-losses` fabricates
losses in place of fitting, to exercise the report path, and refuses to write under `data/studies/`.

Run it with `uv run python studies/nwp_forecast_comparison/nwp_forecast_comparison.py`.
"""

import argparse
import concurrent.futures
import hashlib
import logging
import os
import sys
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, NamedTuple, TypedDict

import numpy as np
import polars as pl
from build_forecast_inputs import (
    ENS_DAYS,
    GEFS_DAYS,
    PRODUCT_DAY_OFFSETS,
    PRODUCT_SLUGS,
    SOLAR_ONLY_PRODUCTS,
)
from contracts.settings import PROJECT_ROOT
from studies.baselines import climatology, shrunk_persistence
from studies.blending import climatology_permutation
from studies.bootstrap import (
    NO_DETECTABLE_DIFFERENCE,
    BootstrapInterval,
    bootstrap_absolute,
    bootstrap_difference,
    bracket_verdict,
    combine_setting_verdicts,
)
from studies.bootstrap import blend_verdict as blend_verdict_from_intervals
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SEEDS,
    SENSITIVITY_HYPER_PARAMETERS,
    HyperParameters,
    out_of_fold_losses,
    score_prediction,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

DomainType = Literal["solar", "wind"]

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss every table reports: each row's clamped error over its own generator's capacity."""

TARGET: Final[str] = "power_mw"
"""The column every arm's XGBoost model is fitted to predict."""

PERCENTAGE_POINTS: Final[float] = 100.0
"""Losses are fractions of capacity; every table prints percentage points."""

MAX_CONCURRENT_FITS: Final[int] = 8
"""How many (arm, site) fits run at once; each releases the GIL while XGBoost trains."""

ROW_SET_START: Final[datetime] = datetime(2024, 12, 1, tzinfo=UTC)
"""The first hour the study's rows cover (the first whole month after IFS Cycle 49r1)."""

DROPPED_MONTHS: Final[tuple[str, ...]] = ("2026-01",)
"""Calendar months dropped entirely: the part-month straddling the UKV PS47 upgrade (2026-01-21),
which #885 also drops."""

UKV_UPGRADE: Final[datetime] = datetime(2026, 1, 21, tzinfo=UTC)
"""The UKV PS47 upgrade; rows before it are UKV's earlier era, rows after it its later era."""

HORIZONS_REPORT: Final[str] = "ens_forecast_horizons/report.md"
"""Under `data/studies/`, the ENS horizons page's report, read for the day-1 reconciliation."""

# --- Folds --------------------------------------------------------------------------------------

NWP_ERA_START_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2026-02")
"""The first month of each era after the first: era 0 is before 2025-10; era 1 is 2025-10 to
2025-12; era 2 starts once the UKV upgrade has settled into a whole month."""

NWP_ERA_FOLD_OFFSETS: Final[dict[int, int]] = {0: 0, 1: 0, 2: 3}
"""How far each era's fold numbers are rotated, so no calendar month is held out of every era at
once (#868). `search_fold_offsets`, which reads only which hours exist and no error, returned this
rotation first among those covering every month in both technologies; `coverage_table` re-checks
it on every run."""


def assign_folds_with_eras(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Cut five folds of whole months inside three eras, using #885's helper.

    TODO(#885): `studies.cross_validation.cut_eras` and its companions are added by PR #885, so
    the import lives inside this function.

    Args:
        frame: Rows carrying `site` and `time`.

    Returns:
        `frame` with `month` (`%Y-%m`), `era_code`, `era` and `fold`.
    """
    from studies.cross_validation import cut_eras

    labelled = frame.with_columns(month=pl.col("time").dt.strftime("%Y-%m"))
    return cut_eras(
        frame=labelled, first_months=NWP_ERA_START_MONTHS, fold_offsets=NWP_ERA_FOLD_OFFSETS
    )


def coverage_table(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the (site, fold, calendar month) coverage table, raising on any uncovered month.

    TODO(#885): imports `calendar_month_coverage` and `raise_on_uncovered_months` from #885.

    Args:
        frame: Rows carrying `site`, `fold` and `time`.

    Returns:
        `studies.cross_validation.calendar_month_coverage`'s result.
    """
    from studies.cross_validation import calendar_month_coverage, raise_on_uncovered_months

    coverage = calendar_month_coverage(frame=frame)
    raise_on_uncovered_months(coverage=coverage)
    return coverage


# --- Arm columns ----------------------------------------------------------------------------

CALENDAR_COLUMNS: Final[tuple[str, ...]] = ("hour_of_day", "day_of_year", "era_code")
"""Every arm's calendar columns, solar and wind alike."""

SOLAR_POSITION_COLUMNS: Final[tuple[str, ...]] = ("solar_elevation_deg", "solar_azimuth_deg")
"""Every solar arm's sun-position columns, on top of `CALENDAR_COLUMNS`."""


def _solar_weather_fields(*, prefix: str) -> tuple[str, str]:
    """Return one product's two solar weather columns: irradiance, then temperature."""
    return (f"{prefix}_ghi", f"{prefix}_temp")


def _wind_weather_fields(*, prefix: str) -> tuple[str, str, str, str]:
    """Return one product's four wind weather columns.

    Hub-height speed, direction sine and cosine, then 10 m speed.
    """
    return (
        f"{prefix}_speed_100m",
        f"{prefix}_sin_100m",
        f"{prefix}_cos_100m",
        f"{prefix}_speed_10m",
    )


def _weather_fields(*, domain: DomainType, prefix: str) -> tuple[str, ...]:
    """Return one product's weather columns for `domain`."""
    if domain == "wind":
        return _wind_weather_fields(prefix=prefix)
    return _solar_weather_fields(prefix=prefix)


def arm_columns(*, domain: DomainType, prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """Return one arm's full, fixed-length feature-column tuple.

    Every arm of one domain that names the same number of products gets the same column count,
    because `colsample_bytree=1` means an unequal count would favour the wider arm.

    Args:
        domain: `solar` or `wind`.
        prefixes: Each named product's own weather-column prefix, such as `ens_mean_day1`.

    Returns:
        The calendar columns, the sun-position columns for solar, then each prefix's own weather
        fields in order.
    """
    calendar = (
        CALENDAR_COLUMNS if domain == "wind" else (*CALENDAR_COLUMNS, *SOLAR_POSITION_COLUMNS)
    )
    weather = tuple(
        column for prefix in prefixes for column in _weather_fields(domain=domain, prefix=prefix)
    )
    return (*calendar, *weather)


# --- Rows -----------------------------------------------------------------------------------

PLANNED_PREFIXES: Final[tuple[str, ...]] = (
    "ens_mean_day0",
    "ens_mean_day1",
    "ukv_day1",
    "icon_eu_day1",
    "icon_eu_day2",
    "ifs025_day1",
    "ifs025_day2",
    "gefs_mean_day1",
)
"""The planned arms (P1 to P4 and their bracket sides), by weather-column prefix, for both
technologies. Fitted at both hyperparameter settings; every other arm is exploratory."""


def _repo_data_dir() -> Path:
    """Return the shared `data/` directory, resolving a linked worktree to the main checkout.

    Duplicated from `verify_previous_runs_leads._repo_data_dir`, because study scripts cannot
    import one another's private helpers.

    Returns:
        The directory holding `studies/`, `NGED/` and the rest of the shared downloads.
    """
    root = PROJECT_ROOT
    marker = root / ".git"
    if marker.is_file():
        pointer = marker.read_text().removeprefix("gitdir:").strip()
        git_dir = Path(pointer)
        if git_dir.parent.name == "worktrees":
            root = git_dir.parent.parent.parent
    return Path(os.environ.get("DATA_PATH_INTERNAL") or root / "data")


def baseline_input_columns(*, domain: DomainType) -> tuple[str, ...]:
    """Return the no-weather baselines' input columns, which the shared rows must also hold.

    Args:
        domain: `solar` or `wind`.

    Returns:
        Persistence and diurnal persistence at every ENS day, and for solar smart persistence's
        clear-sky columns.
    """
    columns = tuple(
        f"{name}_day{day}" for day in ENS_DAYS for name in ("persistence", "diurnal_persistence")
    )
    if domain == "solar":
        columns += ("clear_sky_w_m2", *(f"clear_sky_index_day{day}" for day in ENS_DAYS))
    return columns


def candidate_rows(*, input_dir: Path, domain: DomainType) -> pl.DataFrame:
    """Load one technology's input frame, restricted to the study's span and months.

    Args:
        input_dir: Where `build_forecast_inputs.py` wrote `<domain>_forecast_inputs.parquet`.
        domain: `solar` or `wind`.

    Returns:
        Rows from `ROW_SET_START`, `DROPPED_MONTHS` removed, with a `%Y-%m` `month` label.
    """
    frame = pl.read_parquet(input_dir / f"{domain}_forecast_inputs.parquet").filter(
        pl.col("time") >= ROW_SET_START
    )
    return frame.with_columns(month=pl.col("time").dt.strftime("%Y-%m")).filter(
        ~pl.col("month").is_in(DROPPED_MONTHS)
    )


def _required_columns(*, candidates: pl.DataFrame, domain: DomainType) -> dict[str, list[str]]:
    """Return the columns each requirement of the shared row set needs, by requirement name.

    Args:
        candidates: `candidate_rows()`'s result.
        domain: `solar` or `wind`.

    Returns:
        The target, `baselines`, and each planned prefix whose columns are all in `candidates`.
        A planned prefix absent from `candidates` (GEFS, while ungated) is left out and logged.
    """
    required: dict[str, list[str]] = {
        "target": [TARGET],
        "baselines": [
            column
            for column in baseline_input_columns(domain=domain)
            if column in candidates.columns
        ],
    }
    for prefix in PLANNED_PREFIXES:
        fields = _weather_fields(domain=domain, prefix=prefix)
        if all(column in candidates.columns for column in fields):
            required[prefix] = list(fields)
        else:
            _LOG.warning("%s: planned arm %s is not in the input frame, skipped", domain, prefix)
    return required


def rows_dropped_by_requirement(*, candidates: pl.DataFrame, domain: DomainType) -> dict[str, int]:
    """Count the candidate rows each requirement of the shared row set lacks.

    Args:
        candidates: `candidate_rows()`'s result.
        domain: `solar` or `wind`.

    Returns:
        For the target, the baselines and each planned product, how many candidate rows have a
        null in that requirement's columns. One row can be counted under several requirements.
    """
    required = _required_columns(candidates=candidates, domain=domain)
    return {
        name: candidates.filter(
            pl.any_horizontal(pl.col(column).is_null() for column in columns)
        ).height
        for name, columns in required.items()
    }


def rows(*, input_dir: Path, domain: DomainType) -> pl.DataFrame:
    """Load one technology's shared row set: the rows every planned arm can score.

    Args:
        input_dir: Where `build_forecast_inputs.py` wrote `<domain>_forecast_inputs.parquet`.
        domain: `solar` or `wind`.

    Returns:
        The candidate rows with no null in the target, the baselines' inputs or any planned arm's
        columns, with `fold` and `era_code` assigned.
    """
    candidates = candidate_rows(input_dir=input_dir, domain=domain)
    required = _required_columns(candidates=candidates, domain=domain)
    columns = [column for group in required.values() for column in group]
    complete = candidates.filter(pl.all_horizontal(pl.col(c).is_not_null() for c in columns))
    return assign_folds_with_eras(frame=complete)


def ukv_straddle_share(*, frame: pl.DataFrame, cycle_hours: int) -> float:
    """Return the share of solar rows whose two UKV snapshots would straddle a run change.

    UKV's hourly radiation is the mean of the snapshots at the hour's start and end. Under an
    assumed run cycle of `cycle_hours`, the snapshot at the hour's end comes from a new run when
    the end hour is a multiple of the cycle, so the pair straddles a run change on that share of
    hours. The cycle is an assumption: the run-switch check finds no clear UKV signature.

    Args:
        frame: Solar rows carrying `time`.
        cycle_hours: The assumed run cycle in hours.

    Returns:
        The share of rows, from 0 to 1.
    """
    return float(frame.select((pl.col("time").dt.hour() % cycle_hours == 0).mean()).item())


# --- Jobs -----------------------------------------------------------------------------------

SETTINGS: Final[dict[str, HyperParameters]] = {
    "primary": PRIMARY_HYPER_PARAMETERS,
    "sensitivity": SENSITIVITY_HYPER_PARAMETERS,
}

Job = tuple[str, str, tuple[str, ...], HyperParameters]
"""One (arm, setting) fit: the arm's name, the setting's name, its feature columns, and the
hyperparameters."""


def _product_prefixes(*, domain: DomainType) -> list[str]:
    """Return every Previous Runs product's `<slug>_day<N>` prefix that applies to `domain`."""
    return [
        f"{PRODUCT_SLUGS[product]}_day{day}"
        for product, days in PRODUCT_DAY_OFFSETS.items()
        if not (domain == "wind" and product in SOLAR_ONLY_PRODUCTS)
        for day in days
    ]


def _ens_prefixes() -> list[str]:
    """Return the ENS arms' prefixes: the mean at every ENS day and the control at day 1."""
    return [*(f"ens_mean_day{day}" for day in ENS_DAYS), "ens_control_day1"]


def _gefs_prefixes(*, domain: DomainType) -> list[str]:
    """Return the GEFS mean's prefix at every GEFS day."""
    return [f"gefs_mean_day{day}" for day in GEFS_DAYS[domain]]


BLEND_ARMS: Final[dict[str, tuple[str, str, str]]] = {
    "blend_p4a": ("ens_mean_day1", "icon_eu_day1", "ifs025_day1"),
    "blend_p4b": ("ens_mean_day1", "icon_eu_day2", "ifs025_day2"),
}
"""The two planned blends' product prefixes: ENS's own day-1 mean plus the other two products, at
the optimistic (P4a) and conservative (P4b) lead."""

BLEND_GUARD_SUFFIX: Final[str] = "_permuted"
"""Appended to a permuted product's prefix, forming each guard's own column names."""


def _blend_guard_prefixes(*, blend: str) -> tuple[str, str, str]:
    """Return a blend's guard's product prefixes: ENS real, the other two permuted."""
    ens_prefix, first_other, second_other = BLEND_ARMS[blend]
    return (
        ens_prefix,
        f"{first_other}{BLEND_GUARD_SUFFIX}",
        f"{second_other}{BLEND_GUARD_SUFFIX}",
    )


def jobs(*, domain: DomainType, frame: pl.DataFrame) -> list[Job]:
    """Build one job per (arm, setting), for every arm whose columns are in `frame`.

    Args:
        domain: `solar` or `wind`.
        frame: `rows()`'s result with the blend guard columns added.

    Returns:
        One `Job` per (arm, setting): planned arms, the blends and their controls at `primary` and
        `sensitivity`; every other arm at `primary` only.

    Raises:
        ValueError: If a blend's own columns are present but its guard's are not, which would
            silently drop the guard the blend's verdict needs.
    """
    available = set(frame.columns)
    planned = set(PLANNED_PREFIXES)
    single_product_prefixes = [
        *_product_prefixes(domain=domain),
        *_ens_prefixes(),
        *_gefs_prefixes(domain=domain),
    ]
    output: list[Job] = []
    for prefix in single_product_prefixes:
        columns = arm_columns(domain=domain, prefixes=(prefix,))
        if not all(column in available for column in columns):
            continue
        settings = SETTINGS.keys() if prefix in planned else ("primary",)
        output.extend((prefix, setting, columns, SETTINGS[setting]) for setting in settings)
    for blend, prefixes in BLEND_ARMS.items():
        columns = arm_columns(domain=domain, prefixes=prefixes)
        if not all(column in available for column in columns):
            continue
        output.extend((blend, setting, columns, SETTINGS[setting]) for setting in SETTINGS)
        guard_columns = arm_columns(domain=domain, prefixes=_blend_guard_prefixes(blend=blend))
        if not all(column in available for column in guard_columns):
            msg = f"{blend}: its guard columns are missing; call add_blend_guard_columns first"
            raise ValueError(msg)
        output.extend(
            (f"{blend}_control", setting, guard_columns, SETTINGS[setting]) for setting in SETTINGS
        )
    return output


def add_blend_guard_columns(*, frame: pl.DataFrame, domain: DomainType) -> pl.DataFrame:
    """Add each blend's guard columns: the non-ENS products' weather, permuted among matched hours.

    `studies.blending.climatology_permutation` moves each product's weather among the rows sharing
    a site, year-month and hour of day, so the guard keeps the blend's column count while removing
    the other products' real weather. A product's direction sine and cosine (wind) are permuted
    together as one group, so a permuted row still sits on the unit circle.

    Args:
        frame: Rows carrying `site`, `month`, `hour_of_day` and every blend product's weather.
        domain: `solar` or `wind`.

    Returns:
        `frame` in its own row order, with `<prefix>_permuted_<field>` for every non-ENS blend
        product's weather fields.
    """
    by = ("site", "month", "hour_of_day")
    seed = 20260920
    output = frame
    for prefixes in BLEND_ARMS.values():
        for index, prefix in enumerate(prefixes[1:], start=1):
            fields = _weather_fields(domain=domain, prefix=prefix)
            if not all(column in output.columns for column in fields):
                continue
            groups = (
                [(fields[0],), (fields[1], fields[2]), (fields[3],)]
                if domain == "wind"
                else [(fields[0],), (fields[1],)]
            )
            output = climatology_permutation(
                frame=output,
                column_groups=groups,
                by=by,
                seed=seed + index,
                suffix=BLEND_GUARD_SUFFIX,
            )
            # `climatology_permutation` names a copy `<column>_permuted`; `arm_columns` reads a
            # permuted product as the prefix `<prefix>_permuted`, so its fields are
            # `<prefix>_permuted_<field>`.
            output = output.rename(
                {
                    f"{column}{BLEND_GUARD_SUFFIX}": (
                        f"{prefix}{BLEND_GUARD_SUFFIX}_{column.removeprefix(f'{prefix}_')}"
                    )
                    for group in groups
                    for column in group
                }
            )
    return output


def run_jobs(*, frame: pl.DataFrame, jobs_: list[Job]) -> pl.DataFrame:
    """Fit every (arm, site) job out of fold, concurrently, and stack the losses.

    Args:
        frame: `rows()`'s result, carrying `fold` and the blend guard columns.
        jobs_: The jobs to fit.

    Returns:
        Every job's losses, stacked, labelled with `arm` and `setting`.
    """
    sites = sorted(frame["site"].unique().to_list())
    outputs: list[pl.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FITS) as pool:
        futures = {}
        for arm, setting, features, hyper_parameters in jobs_:
            for site in sites:
                future = pool.submit(
                    out_of_fold_losses,
                    site_rows=frame.filter(pl.col("site") == site),
                    features=list(features),
                    target=TARGET,
                    hyper_parameters=hyper_parameters,
                    with_quantiles=False,
                )
                futures[future] = (arm, setting, site)
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            arm, setting, site = futures[future]
            outputs.append(future.result().with_columns(arm=pl.lit(arm), setting=pl.lit(setting)))
            _LOG.info("%d/%d done: %s / %s / site %s", done, len(futures), setting, arm, site)
    return pl.concat(outputs)


SYNTHETIC_SIGMA: Final[dict[str, float]] = {
    "ens_mean_day0": 0.08,
    "ens_mean_day1": 0.10,
    "ens_mean_day2": 0.12,
    "ens_mean_day3": 0.14,
    "ens_control_day1": 0.11,
    "ukv_day1": 0.075,
    "icon_eu_day1": 0.13,
    "icon_eu_day2": 0.15,
    "ifs025_day1": 0.105,
    "ifs025_day2": 0.125,
    "gefs_mean_day1": 0.102,
    "blend_p4a": 0.09,
    "blend_p4b": 0.097,
    "blend_p4a_control": 0.101,
    "blend_p4b_control": 0.102,
}
"""Each fabricated arm's error standard deviation as a fraction of capacity, chosen so the
synthetic report shows a mixture of verdicts. Any other arm gets `SYNTHETIC_DEFAULT_SIGMA`."""

SYNTHETIC_DEFAULT_SIGMA: Final[float] = 0.11
"""The fabricated error standard deviation for an arm `SYNTHETIC_SIGMA` does not name."""


def synthetic_losses(*, frame: pl.DataFrame, jobs_: list[Job]) -> pl.DataFrame:
    """Fabricate losses for every job: the target plus noise, with no model fitted.

    For exercising the report path only. Each job's prediction is the target plus Gaussian noise
    scaled by the site's capacity, from a random stream seeded by the arm, setting and seed, and
    scored by `score_prediction` as a fitted arm is. `ens_mean_day1` is made unusually accurate on
    hours 0 to 2 UTC, so the monotonicity check meets a failing band.

    Args:
        frame: `rows()`'s result, carrying `fold` and the blend guard columns.
        jobs_: The jobs to fabricate.

    Returns:
        Stacked losses labelled with `arm` and `setting`.
    """
    capacity = frame["effective_capacity_mw"].cast(pl.Float64).to_numpy()
    early = (frame["time"].dt.hour() < 3).to_numpy()
    outputs: list[pl.DataFrame] = []
    for arm, setting, _features, _hyper_parameters in jobs_:
        sigma = SYNTHETIC_SIGMA.get(arm, SYNTHETIC_DEFAULT_SIGMA)
        setting_scale = 1.0 if setting == "primary" else 1.02
        scale = np.full(frame.height, sigma * setting_scale)
        if arm == "ens_mean_day1":
            scale = np.where(early, scale * 0.5, scale)
        for seed in SEEDS:
            generator = np.random.default_rng(zlib.crc32(f"{arm}|{setting}|{seed}".encode()))
            noise = generator.normal(0.0, 1.0, frame.height) * scale * capacity
            prediction = frame.select("site", "time").with_columns(
                seed=pl.lit(seed, dtype=pl.Int32),
                prediction=pl.Series(frame[TARGET].cast(pl.Float64).to_numpy() + noise),
            )
            scored = score_prediction(rows=frame, prediction=prediction, target=TARGET)
            outputs.append(scored.with_columns(arm=pl.lit(arm), setting=pl.lit(setting)))
    return pl.concat(outputs)


# --- Baselines --------------------------------------------------------------------------------


def baseline_losses(*, frame: pl.DataFrame, domain: DomainType) -> pl.DataFrame:
    """Score every no-weather baseline at every ENS day, the same forecast for every seed.

    A baseline has no fitting seed, so its forecast is repeated for each of `SEEDS`, which lets the
    paired bootstrap pair it with a fitted arm seed by seed. It has no hyperparameters either, so
    the same losses stand under both settings. Each forecast is held to the export cap and scored
    by `score_prediction`, as a fitted arm is.

    Args:
        frame: `rows()`'s result, carrying the baselines' input columns.
        domain: `solar` or `wind`.

    Returns:
        Losses for `climatology` and, at each ENS day, `persistence_day<N>`,
        `diurnal_persistence_day<N>` and `smart_persistence_day<N>`, labelled with `arm` and
        `setting`.
    """
    frame = frame.with_columns(climatology=climatology(frame=frame))
    forecasts: dict[str, pl.Expr] = {"climatology": pl.col("climatology")}
    for day in ENS_DAYS:
        for name in ("persistence", "diurnal_persistence"):
            forecasts[f"{name}_day{day}"] = pl.col(f"{name}_day{day}")
        smart = f"smart_persistence_day{day}"
        if domain == "solar":
            forecasts[smart] = pl.col(f"clear_sky_index_day{day}") * pl.col("clear_sky_w_m2")
        else:
            shrunk = shrunk_persistence(frame=frame, persisted=f"persistence_day{day}")
            frame = frame.with_columns(shrunk["shrunk"].alias(smart))
            forecasts[smart] = pl.col(smart)
    seeds = pl.DataFrame({"seed": list(SEEDS)}, schema={"seed": pl.Int32})
    outputs = []
    for arm, expression in forecasts.items():
        prediction = frame.select("site", "time", prediction=expression).join(seeds, how="cross")
        scored = score_prediction(rows=frame, prediction=prediction, target=TARGET)
        outputs.extend(
            scored.with_columns(arm=pl.lit(arm), setting=pl.lit(setting)) for setting in SETTINGS
        )
    return pl.concat(outputs)


def is_baseline_arm(*, arm: str) -> bool:
    """Whether `arm` names a no-weather baseline rather than a fitted arm."""
    return arm == "climatology" or arm.startswith(
        ("persistence_day", "diurnal_persistence_day", "smart_persistence_day")
    )


# --- Contrasts ----------------------------------------------------------------------------------


def assert_equal_rows(*, losses: pl.DataFrame, treatment: str, reference: str) -> None:
    """Raise unless `treatment` and `reference` score exactly the same (site, time, seed) rows.

    `studies.bootstrap.paired_differences` inner-joins the two arms, which would silently shrink a
    contrast's rows, or pair rows across hyperparameter settings, rather than fail; every contrast
    in this module calls this first.

    Args:
        losses: Per-row losses carrying both arms, restricted to one setting.
        treatment: One arm's name.
        reference: The other arm's name.

    Raises:
        ValueError: If either arm holds a duplicated (site, time, seed), or the two arms' key sets
            differ.
    """
    keys = ["site", "time", "seed"]
    left = losses.filter(pl.col("arm") == treatment).select(keys)
    right = losses.filter(pl.col("arm") == reference).select(keys)
    for name, side in ((treatment, left), (reference, right)):
        if side.is_duplicated().any():
            msg = f"assert_equal_rows: {name!r} holds a duplicated (site, time, seed)"
            raise ValueError(msg)
    unshared = (
        left.join(right, on=keys, how="anti").height + right.join(left, on=keys, how="anti").height
    )
    if unshared:
        msg = (
            f"assert_equal_rows: {treatment!r} and {reference!r} score different rows "
            f"({left.height} vs {right.height}, {unshared} not shared)"
        )
        raise ValueError(msg)


def arms_present(*, losses: pl.DataFrame, arms: tuple[str, ...]) -> bool:
    """Whether every arm in `arms` has at least one row in `losses`."""
    present = set(losses["arm"].unique().to_list())
    return all(arm in present for arm in arms)


def difference(*, losses: pl.DataFrame, treatment: str, reference: str) -> BootstrapInterval:
    """Return `treatment − reference`'s interval, after checking both arms hold the same rows.

    Args:
        losses: Per-row losses at one setting, carrying both arms.
        treatment: The arm whose error is compared.
        reference: The arm it is compared against.

    Returns:
        The paired month-and-seed bootstrap interval.
    """
    assert_equal_rows(losses=losses, treatment=treatment, reference=reference)
    return bootstrap_difference(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )


class BracketResult(TypedDict):
    """One product's bracket against ENS: both sides, the verdict, and whether it broke."""

    lower: BootstrapInterval
    upper: BootstrapInterval
    verdict: str
    assumption_failed: bool


def bracket(*, losses: pl.DataFrame, product_arm: str, day: int) -> BracketResult:
    """Bracket a Previous Runs product's day-`day` arm between ENS days `day − 1` and `day`.

    Args:
        losses: Per-row losses at one setting, carrying the product and both ENS arms.
        product_arm: The product's arm, such as `ukv_day1`.
        day: The product's day offset.

    Returns:
        The lower side (product − ENS day `day − 1`), the upper side (product − ENS day `day`), the
        verdict, and whether both sides were significant, which means ENS's error fell with lead.
    """
    lower = difference(losses=losses, treatment=product_arm, reference=f"ens_mean_day{day - 1}")
    upper = difference(losses=losses, treatment=product_arm, reference=f"ens_mean_day{day}")
    return {
        "lower": lower,
        "upper": upper,
        "verdict": bracket_verdict(lower_side=lower, upper_side=upper),
        "assumption_failed": lower["upper_95"] < 0.0 and upper["lower_95"] > 0.0,
    }


def three_hour_band(*, hour: pl.Expr) -> pl.Expr:
    """Return the start hour of the 3-hour UTC band containing `hour`."""
    return (hour // 3) * 3


def check_ens_monotonicity(
    *, losses: pl.DataFrame, day_pairs: tuple[tuple[int, int], ...]
) -> pl.DataFrame:
    """Check that ENS's error does not fall as its lead rises, per day pair and 3-hour UTC band.

    A bracket rests on this holding: its lower side needs ENS day N−1 to be no worse than ENS day
    N. A point estimate at or below zero in a band voids, in that band, every bracket resting on
    the pair.

    Args:
        losses: Per-row losses at one setting, carrying `ens_mean_day<N>` arms and `time`.
        day_pairs: The (earlier, later) day pairs to check.

    Returns:
        One row per (day pair, band) whose two arms are both present, with the `later − earlier`
        interval and whether its point estimate is positive.
    """
    banded = losses.with_columns(band=three_hour_band(hour=pl.col("time").dt.hour()))
    records = []
    for earlier, later in day_pairs:
        earlier_arm, later_arm = f"ens_mean_day{earlier}", f"ens_mean_day{later}"
        if not arms_present(losses=banded, arms=(earlier_arm, later_arm)):
            continue
        for band in sorted(banded["band"].unique().to_list()):
            interval = difference(
                losses=banded.filter(pl.col("band") == band),
                treatment=later_arm,
                reference=earlier_arm,
            )
            records.append(
                {
                    "earlier_day": earlier,
                    "later_day": later,
                    "band_start_utc": band,
                    "difference": interval["difference"],
                    "lower_95": interval["lower_95"],
                    "upper_95": interval["upper_95"],
                    "monotone": interval["difference"] > 0.0,
                }
            )
    return pl.DataFrame(
        records,
        schema={
            "earlier_day": pl.Int64,
            "later_day": pl.Int64,
            "band_start_utc": pl.Int8,
            "difference": pl.Float64,
            "lower_95": pl.Float64,
            "upper_95": pl.Float64,
            "monotone": pl.Boolean,
        },
    )


def voided_bands(*, monotonicity: pl.DataFrame, day: int) -> list[int]:
    """Return the bands where ENS's error fell from day `day − 1` to day `day`.

    Args:
        monotonicity: `check_ens_monotonicity()`'s result.
        day: The product's day offset; the bracket rests on the pair (`day − 1`, `day`).

    Returns:
        The start hours of the voided bands, ascending.
    """
    failed = monotonicity.filter(
        (pl.col("later_day") == day) & (pl.col("earlier_day") == day - 1) & ~pl.col("monotone")
    )
    return sorted(failed["band_start_utc"].to_list())


def bracket_in_band(
    *, losses: pl.DataFrame, product_arm: str, day: int, band: int
) -> BracketResult:
    """Return a product's bracket on the rows of one 3-hour UTC band.

    Args:
        losses: Per-row losses at one setting.
        product_arm: The product's arm.
        day: The product's day offset.
        band: The band's start hour, UTC.

    Returns:
        `bracket()`'s result on that band's rows.
    """
    in_band = losses.filter(three_hour_band(hour=pl.col("time").dt.hour()) == band)
    return bracket(losses=in_band, product_arm=product_arm, day=day)


EXACT_LEAD_WIND_HOURS: Final[dict[str, tuple[int, ...]]] = {
    "icon_eu_day1": tuple(range(6)),
    "ukv_day1": (0,),
}
"""Each product's UTC hours where its day-1 lead equals ENS day 1's exactly: `h < n` for a run
cycle `n`. ICON-EU runs 6-hourly, so hours 0 to 5; hour 0 is exact for any cycle, which is all
UKV is given because the run-switch check found no clear UKV cycle."""


def exact_lead_wind_rereads(*, losses: pl.DataFrame) -> dict[str, BootstrapInterval]:
    """Re-read the upper-side wind contrast on the hours where no bracket is needed.

    At these hours a product's day-1 lead equals ENS day 1's exactly, so `product − ENS day 1` is
    itself a matched-lead contrast.

    Args:
        losses: Per-row wind losses at one setting.

    Returns:
        Each product's `product − ens_mean_day1` interval on its exact-lead hours.
    """
    output: dict[str, BootstrapInterval] = {}
    for arm, hours in EXACT_LEAD_WIND_HOURS.items():
        if not arms_present(losses=losses, arms=(arm, "ens_mean_day1")):
            continue
        subset = losses.filter(pl.col("time").dt.hour().is_in(hours))
        output[arm] = difference(losses=subset, treatment=arm, reference="ens_mean_day1")
    return output


def leaderboard(*, losses: pl.DataFrame, arms: list[str]) -> pl.DataFrame:
    """Return each arm's own absolute error with its interval.

    Args:
        losses: Per-row losses at one setting.
        arms: The arms to include, in the order shown; an arm with no rows is skipped.

    Returns:
        One row per arm with `arm`, `value`, `lower_95`, `upper_95`, `n_rows` and `n_months`.
    """
    records = [
        {"arm": arm, **bootstrap_absolute(losses=losses, arm=arm, metric=METRIC)}
        for arm in arms
        if arms_present(losses=losses, arms=(arm,))
    ]
    return pl.DataFrame(records)


class BlendResult(TypedDict):
    """One blend at one setting: both contrasts against ENS alone, both guards, and the verdict."""

    p4a: BootstrapInterval
    p4a_guard: BootstrapInterval
    p4b: BootstrapInterval
    p4b_guard: BootstrapInterval
    verdict: str
    largest_gain_not_excluded: float | None


def blend_result(*, losses: pl.DataFrame) -> BlendResult:
    """Compute both blend bounds, their guards and the published verdict at one setting.

    Args:
        losses: Per-row losses at one setting, carrying both blends, their controls and
            `ens_mean_day1`.

    Returns:
        The four intervals and the verdict from `studies.bootstrap.blend_verdict`.
    """
    p4a = difference(losses=losses, treatment="blend_p4a", reference="ens_mean_day1")
    p4a_guard = difference(losses=losses, treatment="blend_p4a", reference="blend_p4a_control")
    p4b = difference(losses=losses, treatment="blend_p4b", reference="ens_mean_day1")
    p4b_guard = difference(losses=losses, treatment="blend_p4b", reference="blend_p4b_control")
    verdict = blend_verdict_from_intervals(
        p4a=p4a, p4a_guard=p4a_guard, p4b=p4b, p4b_guard=p4b_guard
    )
    return {
        "p4a": p4a,
        "p4a_guard": p4a_guard,
        "p4b": p4b,
        "p4b_guard": p4b_guard,
        "verdict": verdict["verdict"],
        "largest_gain_not_excluded": verdict["largest_gain_not_excluded"],
    }


# --- Saved outputs -------------------------------------------------------------------------------


def fingerprint(*, frame: pl.DataFrame) -> str:
    """Return a short hash of a losses frame, independent of row order, floats cast to Float32.

    Args:
        frame: The saved per-row losses, carrying `setting`, `arm`, `site`, `time` and `seed`.

    Returns:
        A hex digest, stable across re-runs on the same data and code.
    """
    ordered = frame.sort("setting", "arm", "site", "time", "seed")
    floats = [name for name, dtype in ordered.schema.items() if dtype in (pl.Float32, pl.Float64)]
    cast = ordered.with_columns([pl.col(name).cast(pl.Float32) for name in floats])
    row_hashes = cast.hash_rows(seed=0).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:16]


def losses_path(*, output_dir: Path, domain: DomainType) -> Path:
    """Return where one domain's saved per-row losses live."""
    return output_dir / f"{domain}_losses.parquet"


def predictions_path(*, output_dir: Path, domain: DomainType) -> Path:
    """Return where one domain's saved per-row predictions live."""
    return output_dir / f"{domain}_predictions.parquet"


def predictions_from_losses(*, losses: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """Recover each arm's capped prediction from its capped signed error and the target.

    Args:
        losses: Per-row losses carrying `signed_error_capped_mw`.
        frame: The rows, carrying the target.

    Returns:
        `arm`, `setting`, `site`, `time`, `seed`, and `prediction_capped_mw`.
    """
    return (
        losses.select("arm", "setting", "site", "time", "seed", "signed_error_capped_mw")
        .join(frame.select("site", "time", TARGET), on=["site", "time"])
        .select(
            "arm",
            "setting",
            "site",
            "time",
            "seed",
            prediction_capped_mw=pl.col(TARGET).cast(pl.Float64) + pl.col("signed_error_capped_mw"),
        )
    )


def save_outputs(
    *, output_dir: Path, domain: DomainType, losses: pl.DataFrame, frame: pl.DataFrame
) -> None:
    """Write one domain's per-row losses and predictions as parquet.

    Args:
        output_dir: The directory to write to.
        domain: `solar` or `wind`.
        losses: Every arm's per-row losses, baselines included.
        frame: The rows the losses score.
    """
    ordered = losses.sort("setting", "arm", "site", "time", "seed")
    ordered.write_parquet(losses_path(output_dir=output_dir, domain=domain))
    predictions_from_losses(losses=ordered, frame=frame).write_parquet(
        predictions_path(output_dir=output_dir, domain=domain)
    )


def read_saved_losses(*, output_dir: Path, domain: DomainType) -> pl.DataFrame | None:
    """Return one domain's saved losses, or `None` if none were saved."""
    path = losses_path(output_dir=output_dir, domain=domain)
    return pl.read_parquet(path) if path.exists() else None


# --- Report -----------------------------------------------------------------------------------


def _pp(*, value: float) -> str:
    """Format a fraction of capacity as signed percentage points."""
    return f"{value * PERCENTAGE_POINTS:+.3f}"


def _interval_text(*, interval: BootstrapInterval) -> str:
    """Format a difference interval in percentage points."""
    return (
        f"{_pp(value=interval['difference'])} "
        f"[{_pp(value=interval['lower_95'])}, {_pp(value=interval['upper_95'])}]"
    )


def horizons_ens_day1_mae(*, domain: DomainType, path: Path) -> float | None:
    """Read the ENS horizons page's day-1 ENS mean error, in percentage points of capacity.

    Args:
        domain: `solar` or `wind`.
        path: The horizons page's `report.md`.

    Returns:
        The first `ens_mean_day1` leaderboard row in the domain's section, or `None` if the file
        or the row is absent.
    """
    if not path.exists():
        return None
    heading = f"### {domain.capitalize()}:"
    in_section = False
    for line in path.read_text().splitlines():
        if line.startswith("### "):
            in_section = line.startswith(heading)
        elif in_section and line.startswith("| ens_mean_day1 |"):
            return float(line.split("|")[2])
    return None


def _column_lines(*, job_list: list[Job]) -> list[str]:
    """Return the table of every arm's column list."""
    lines = ["| Arm | Setting | Columns |", "|---|---|---|"]
    lines.extend(
        f"| {arm} | {setting} | {', '.join(columns)} |" for arm, setting, columns, _ in job_list
    )
    return lines


def _row_lines(
    *, domain: DomainType, candidates: pl.DataFrame, frame: pl.DataFrame, dropped: dict[str, int]
) -> list[str]:
    """Return the shared-row counts, the rows each requirement drops, and the UKV straddle share."""
    lines = [
        (
            f"{candidates.height:,} candidate rows from {ROW_SET_START:%Y-%m-%d}; "
            f"{frame.height:,} shared rows after the target, the baselines and every planned "
            "product are required."
        ),
        "",
        "| Requirement | Candidate rows lacking it | Share |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {name} | {count:,} | {count / candidates.height:.1%} |"
        for name, count in dropped.items()
    )
    if domain == "solar" and "ukv_day1_ghi" in frame.columns:
        shares = ", ".join(
            f"{ukv_straddle_share(frame=frame, cycle_hours=cycle):.1%} at a {cycle}-hour cycle"
            for cycle in (3, 6)
        )
        lines += [
            "",
            (
                "UKV straddle share (hours whose two snapshots would straddle a run change, "
                "assuming a run cycle; the run-switch check found no clear UKV cycle): "
                f"{shares}."
            ),
        ]
    return lines


def _leaderboard_lines(*, losses: pl.DataFrame, arms: list[str]) -> list[str]:
    """Return one setting's leaderboard table, each arm's own error with its interval."""
    board = leaderboard(losses=losses, arms=arms)
    lines = [
        "| Arm | Error (% of capacity) | 95% interval | Rows | Months |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {record['arm']} | {record['value'] * PERCENTAGE_POINTS:.3f} | "
        f"[{record['lower_95'] * PERCENTAGE_POINTS:.3f}, "
        f"{record['upper_95'] * PERCENTAGE_POINTS:.3f}] | {record['n_rows']:,} | "
        f"{record['n_months']} |"
        for record in board.iter_rows(named=True)
    )
    return lines


CONTRAST_HEADER: Final[list[str]] = [
    "| ID | Status | Contrast | Setting | Difference (points) [95% interval] | Rows | Months |",
    "|---|---|---|---|---|---|---|",
]


def _contrast_line(
    *,
    identifier: str,
    status: str,
    label: str,
    setting: str,
    interval: BootstrapInterval,
) -> str:
    """Format one contrast's row."""
    return (
        f"| {identifier} | {status} | {label} | {setting} | {_interval_text(interval=interval)} | "
        f"{interval['n_rows']:,} | {interval['n_months']} |"
    )


PLANNED_BRACKETS: Final[dict[str, tuple[str, str, str]]] = {
    "ukv": ("P1a", "P1b", "UKV"),
    "icon_eu": ("P2a", "P2b", "ICON-EU"),
}
"""Each planned bracket's product slug to its upper-side ID, lower-side ID and name. Both are
at day 1."""


def _bracket_lines(
    *,
    by_setting: dict[str, pl.DataFrame],
    monotonicity: dict[str, pl.DataFrame],
    domain: DomainType,
) -> list[str]:
    """Return the bracket sections: planned P1 and P2, then every exploratory bracket."""
    lines = ["#### Planned brackets (day 1)", "", *CONTRAST_HEADER]
    verdicts: list[str] = []
    voided_notes: list[str] = []
    for slug, (upper_id, lower_id, name) in PLANNED_BRACKETS.items():
        arm = f"{slug}_day1"
        per_setting: dict[str, str] = {}
        for setting, losses in by_setting.items():
            if not arms_present(losses=losses, arms=(arm, "ens_mean_day0", "ens_mean_day1")):
                continue
            result = bracket(losses=losses, product_arm=arm, day=1)
            per_setting[setting] = result["verdict"]
            lines.append(
                _contrast_line(
                    identifier=upper_id,
                    status="planned",
                    label=f"{name} day 1 − ENS mean day 1 (upper side)",
                    setting=setting,
                    interval=result["upper"],
                )
            )
            lines.append(
                _contrast_line(
                    identifier=lower_id,
                    status="planned",
                    label=f"{name} day 1 − ENS mean day 0 (lower side)",
                    setting=setting,
                    interval=result["lower"],
                )
            )
            if result["assumption_failed"]:
                voided_notes.append(
                    f"{name} at {setting}: both sides are significant, so ENS day 1 scored better "
                    f"than ENS day 0 and the bracket's assumption failed."
                )
            voided_notes += _voided_band_lines(
                losses=losses,
                monotonicity=monotonicity[setting],
                arm=arm,
                day=1,
                setting=setting,
                name=name,
            )
        if len(per_setting) == len(by_setting):
            combined = combine_setting_verdicts(**{k: per_setting[k] for k in SETTINGS})
            verdicts.append(
                f"| {name} | {per_setting['primary']} | {per_setting['sensitivity']} | "
                f"**{combined}** |"
            )
    lines += [
        "",
        "| Product (day 1) | Primary | Sensitivity | Verdict (both settings must agree) |",
        "|---|---|---|---|",
        *verdicts,
        "",
        *voided_notes,
    ]
    primary = by_setting["primary"]
    lines += ["", "#### Exploratory brackets, every other Previous Runs product and day", ""]
    lines += CONTRAST_HEADER
    exploratory_notes: list[str] = []
    for arm in sorted(set(primary["arm"].unique().to_list())):
        slug, _, day_text = arm.rpartition("_day")
        if not day_text.isdigit() or slug not in _all_slugs(domain=domain):
            continue
        day = int(day_text)
        if (slug in PLANNED_BRACKETS and day == 1) or day == 0:
            continue
        if not arms_present(losses=primary, arms=(f"ens_mean_day{day - 1}", f"ens_mean_day{day}")):
            continue
        result = bracket(losses=primary, product_arm=arm, day=day)
        failed = " - bracket assumption failed" if result["assumption_failed"] else ""
        for side, ens_day in (("upper", day), ("lower", day - 1)):
            lines.append(
                _contrast_line(
                    identifier=f"X-{arm}-{side}",
                    status="exploratory",
                    label=(
                        f"{arm} − ens_mean_day{ens_day} ({side} side; {result['verdict']}{failed})"
                    ),
                    setting="primary",
                    interval=result["upper"] if side == "upper" else result["lower"],
                )
            )
        exploratory_notes += _voided_band_lines(
            losses=primary,
            monotonicity=monotonicity["primary"],
            arm=arm,
            day=day,
            setting="primary",
            name=arm,
        )
    lines += ["", *exploratory_notes]
    return lines


def _all_slugs(*, domain: DomainType) -> set[str]:
    """Return every Previous Runs product slug that applies to `domain`."""
    return {
        PRODUCT_SLUGS[product]
        for product in PRODUCT_DAY_OFFSETS
        if not (domain == "wind" and product in SOLAR_ONLY_PRODUCTS)
    }


def _voided_band_lines(
    *,
    losses: pl.DataFrame,
    monotonicity: pl.DataFrame,
    arm: str,
    day: int,
    setting: str,
    name: str,
) -> list[str]:
    """Return the per-band bracket lines for each band ENS's monotonicity failed in."""
    lines = []
    for band in voided_bands(monotonicity=monotonicity, day=day):
        result = bracket_in_band(losses=losses, product_arm=arm, day=day, band=band)
        lines.append(
            f"{name} at {setting}, day {day}: band {band:02d}-{band + 3:02d} UTC is voided (ENS "
            f"day {day - 1} did not beat day {day}); its own bracket is lower "
            f"{_interval_text(interval=result['lower'])}, upper "
            f"{_interval_text(interval=result['upper'])}, {result['verdict']}."
        )
    return lines


def _monotonicity_lines(*, monotonicity: dict[str, pl.DataFrame]) -> list[str]:
    """Return the ENS monotonicity table by band, per setting."""
    lines = [
        "| Setting | Pair (day N−1 → N) | Band (UTC) | Change (points) | 95% interval | Monotone |",
        "|---|---|---|---|---|---|",
    ]
    for setting, table in monotonicity.items():
        lines.extend(
            f"| {setting} | {row['earlier_day']} → {row['later_day']} | "
            f"{row['band_start_utc']:02d}-{row['band_start_utc'] + 3:02d} | "
            f"{_pp(value=row['difference'])} | [{_pp(value=row['lower_95'])}, "
            f"{_pp(value=row['upper_95'])}] | {'yes' if row['monotone'] else '**no, voided**'} |"
            for row in table.iter_rows(named=True)
        )
    return lines


def _gefs_lines(*, by_setting: dict[str, pl.DataFrame]) -> list[str]:
    """Return P3, or a note that it is skipped while the GEFS columns are absent."""
    if not arms_present(losses=by_setting["primary"], arms=("gefs_mean_day1",)):
        return ["P3 (GEFS mean day 1 − ENS mean day 1) is skipped: the GEFS columns are absent."]
    lines = list(CONTRAST_HEADER)
    lines.extend(
        _contrast_line(
            identifier="P3",
            status="planned",
            label="GEFS mean day 1 − ENS mean day 1 (exact lead)",
            setting=setting,
            interval=difference(
                losses=losses, treatment="gefs_mean_day1", reference="ens_mean_day1"
            ),
        )
        for setting, losses in by_setting.items()
    )
    return lines


def _blend_lines(*, by_setting: dict[str, pl.DataFrame]) -> list[str]:
    """Return P4a, P4b, both guards and the combined blend verdict."""
    if not arms_present(losses=by_setting["primary"], arms=("blend_p4a", "blend_p4b")):
        return ["The blends are skipped: their product columns are absent."]
    lines = list(CONTRAST_HEADER)
    results: dict[str, BlendResult] = {}
    for setting, losses in by_setting.items():
        result = blend_result(losses=losses)
        results[setting] = result
        for identifier, label, interval in (
            ("P4a", "blend P4a − ENS mean day 1", result["p4a"]),
            ("P4a guard", "blend P4a − its permutation control", result["p4a_guard"]),
            ("P4b", "blend P4b − ENS mean day 1", result["p4b"]),
            ("P4b guard", "blend P4b − its permutation control", result["p4b_guard"]),
        ):
            lines.append(
                _contrast_line(
                    identifier=identifier,
                    status="planned",
                    label=label,
                    setting=setting,
                    interval=interval,
                )
            )
        primary = by_setting["primary"]
        if setting == "primary":
            lines.extend(
                _contrast_line(
                    identifier=f"X-{arm}",
                    status="exploratory",
                    label=f"{arm} − ENS mean day 1",
                    setting="primary",
                    interval=difference(losses=primary, treatment=arm, reference="ens_mean_day1"),
                )
                for arm in ("blend_p4a_control", "blend_p4b_control")
            )
    combined = combine_setting_verdicts(
        primary=results["primary"]["verdict"],
        sensitivity=results["sensitivity"]["verdict"],
        unresolved=NO_DETECTABLE_DIFFERENCE,
    )
    lines += [
        "",
        (
            f"Blend verdict: primary {results['primary']['verdict']}, sensitivity "
            f"{results['sensitivity']['verdict']}; **combined: {combined}**."
        ),
    ]
    bound = results["primary"]["largest_gain_not_excluded"]
    if combined == NO_DETECTABLE_DIFFERENCE and bound is not None:
        lines.append(
            f"A gain as large as {bound * PERCENTAGE_POINTS:.3f} points is not excluded "
            "(primary setting, P4b's lower bound)."
        )
    return lines


def _era_lines(*, losses: pl.DataFrame) -> list[str]:
    """Return UKV's P1 bracket split by UKV era."""
    if not arms_present(losses=losses, arms=("ukv_day1", "ens_mean_day0", "ens_mean_day1")):
        return ["No UKV arm."]
    lines = list(CONTRAST_HEADER)
    for era, subset in (
        ("before the 2026-01-21 upgrade", losses.filter(pl.col("time") < UKV_UPGRADE)),
        ("after the upgrade", losses.filter(pl.col("time") >= UKV_UPGRADE)),
    ):
        result = bracket(losses=subset, product_arm="ukv_day1", day=1)
        for side, ens_day in (("upper", 1), ("lower", 0)):
            lines.append(
                _contrast_line(
                    identifier=f"X-P1-{side}",
                    status="exploratory",
                    label=f"UKV day 1 − ENS mean day {ens_day} ({era}; {result['verdict']})",
                    setting="primary",
                    interval=result["upper"] if side == "upper" else result["lower"],
                )
            )
    return lines


def _reconciliation_line(*, domain: DomainType, losses: pl.DataFrame, path: Path) -> str:
    """Return this study's ENS day-1 error beside the horizons page's."""
    if not arms_present(losses=losses, arms=("ens_mean_day1",)):
        return "ENS day-1 reconciliation: no ens_mean_day1 arm."
    ours = bootstrap_absolute(losses=losses, arm="ens_mean_day1", metric=METRIC)["value"]
    theirs = horizons_ens_day1_mae(domain=domain, path=path)
    other = f"{theirs:.3f}" if theirs is not None else "not available"
    return (
        f"ENS day-1 reconciliation (exploratory): this study's ens_mean_day1 error is "
        f"{ours * PERCENTAGE_POINTS:.3f}% of capacity; the horizons page's is {other}% "
        f"(its rows start 2024-04, so the two differ in span)."
    )


class DomainInputs(NamedTuple):
    """Everything the report needs about one technology besides its losses."""

    candidates: pl.DataFrame
    frame: pl.DataFrame
    dropped: dict[str, int]
    jobs: list[Job]
    coverage: pl.DataFrame


def _domain_lines(*, domain: DomainType, inputs: DomainInputs, losses: pl.DataFrame) -> list[str]:
    """Return one technology's whole report section."""
    by_setting = {setting: losses.filter(pl.col("setting") == setting) for setting in SETTINGS}
    monotonicity = {
        setting: check_ens_monotonicity(
            losses=by_setting[setting], day_pairs=((0, 1), (1, 2), (2, 3))
        )
        for setting in SETTINGS
    }
    uncovered = inputs.coverage.filter(~pl.col("covered")).height
    arms = losses["arm"].unique(maintain_order=True).to_list()
    baselines = [arm for arm in arms if is_baseline_arm(arm=arm)]
    fitted = [arm for arm in arms if not is_baseline_arm(arm=arm)]
    lines = [
        f"## {domain.capitalize()}",
        "",
        "### Rows",
        "",
        *_row_lines(
            domain=domain, candidates=inputs.candidates, frame=inputs.frame, dropped=inputs.dropped
        ),
        "",
        (
            f"Coverage table: {inputs.coverage.height} (site, fold, month) cells, {uncovered} "
            "not covered by training data (`raise_on_uncovered_months` allows only months that "
            "occur in one year)."
        ),
        "",
        "### Arm columns",
        "",
        *_column_lines(job_list=inputs.jobs),
        "",
        f"Losses fingerprint: `{fingerprint(frame=losses)}`.",
        "",
    ]
    for setting, setting_losses in by_setting.items():
        lines += [
            f"### Leaderboard, {setting} setting",
            "",
            *_leaderboard_lines(losses=setting_losses, arms=[*baselines, *fitted]),
            "",
        ]
    lines += [
        "### Planned contrasts and verdicts",
        "",
        *_bracket_lines(by_setting=by_setting, monotonicity=monotonicity, domain=domain),
        "",
        "#### P3, GEFS against ENS",
        "",
        *_gefs_lines(by_setting=by_setting),
        "",
        "#### P4, the blend",
        "",
        *_blend_lines(by_setting=by_setting),
        "",
        "### ENS monotonicity by band",
        "",
        *_monotonicity_lines(monotonicity=monotonicity),
        "",
        "### Exploratory: UKV P1 by era",
        "",
        *_era_lines(losses=by_setting["primary"]),
        "",
    ]
    if domain == "wind":
        lines += [
            "### Exploratory: exact-lead wind re-read (primary setting)",
            "",
            *CONTRAST_HEADER,
        ]
        for arm, interval in exact_lead_wind_rereads(losses=by_setting["primary"]).items():
            hours = EXACT_LEAD_WIND_HOURS[arm]
            lines.append(
                _contrast_line(
                    identifier=f"X-exact-{arm}",
                    status="exploratory",
                    label=f"{arm} − ens_mean_day1 on hours {hours[0]:02d}-{hours[-1]:02d} UTC",
                    setting="primary",
                    interval=interval,
                )
            )
        lines.append("")
    lines += [
        "### Exploratory: ENS control against ENS mean, day 1",
        "",
        *CONTRAST_HEADER,
    ]
    if arms_present(losses=by_setting["primary"], arms=("ens_control_day1", "ens_mean_day1")):
        lines.append(
            _contrast_line(
                identifier="X-control",
                status="exploratory",
                label="ENS mean day 1 − ENS control day 1",
                setting="primary",
                interval=difference(
                    losses=by_setting["primary"],
                    treatment="ens_mean_day1",
                    reference="ens_control_day1",
                ),
            )
        )
    lines += [
        "",
        _reconciliation_line(
            domain=domain,
            losses=by_setting["primary"],
            path=_repo_data_dir() / "studies" / HORIZONS_REPORT,
        ),
        "",
    ]
    return lines


def write_report(
    *,
    output_dir: Path,
    inputs: dict[DomainType, DomainInputs],
    losses: dict[DomainType, pl.DataFrame],
) -> None:
    """Write `report.md` from each technology's inputs and saved losses.

    Args:
        output_dir: Where `report.md` is written.
        inputs: Each domain's rows, jobs and coverage.
        losses: Each domain's stacked losses, baselines included.
    """
    lines = [
        "# Weather forecasts for power, compared at matched lead times: report",
        "",
        (
            "Differences are first arm minus second, in percentage points of capacity. Planned "
            "contrasts carry the plan's IDs; every other contrast is exploratory, and about 1 in "
            "20 exploratory intervals reaches significance at 5% by chance."
        ),
        "",
    ]
    for domain, domain_inputs in inputs.items():
        lines += _domain_lines(domain=domain, inputs=domain_inputs, losses=losses[domain])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


# --- CLI ----------------------------------------------------------------------------------------


def _obtain_losses(
    *,
    args: argparse.Namespace,
    domain: DomainType,
    inputs: DomainInputs,
    output_dir: Path,
) -> pl.DataFrame:
    """Return one domain's model losses and baselines by fabricating, reading, or fitting."""
    saved = read_saved_losses(output_dir=output_dir, domain=domain)
    if args.report_only:
        if saved is None:
            msg = f"--report-only: no saved losses for {domain} in {output_dir}"
            raise FileNotFoundError(msg)
        return saved
    if args.synthetic_losses:
        model = synthetic_losses(frame=inputs.frame, jobs_=inputs.jobs)
    else:
        kept = pl.DataFrame()
        todo = inputs.jobs
        if args.fit_missing and saved is not None:
            baseline_arms = [a for a in saved["arm"].unique().to_list() if is_baseline_arm(arm=a)]
            kept = saved.filter(~pl.col("arm").is_in(baseline_arms))
            done = set(kept.select("arm", "setting").unique().iter_rows())
            todo = [job for job in inputs.jobs if (job[0], job[1]) not in done]
            _LOG.info("%s: %d of %d jobs still to fit", domain, len(todo), len(inputs.jobs))
        fitted = run_jobs(frame=inputs.frame, jobs_=todo) if todo else pl.DataFrame()
        model = pl.concat([kept, fitted], how="diagonal") if not kept.is_empty() else fitted
    baselines = baseline_losses(frame=inputs.frame, domain=domain)
    return pl.concat([model, baselines], how="diagonal")


def _is_under_real_data(*, path: Path) -> bool:
    """Whether `path` lies inside the shared `data/studies/` folder."""
    return path.resolve().is_relative_to((_repo_data_dir() / "studies").resolve())


def _build_inputs(*, input_dir: Path, domain: DomainType) -> DomainInputs:
    """Build one technology's rows, guard columns, coverage table and job list."""
    candidates = candidate_rows(input_dir=input_dir, domain=domain)
    dropped = rows_dropped_by_requirement(candidates=candidates, domain=domain)
    frame = add_blend_guard_columns(frame=rows(input_dir=input_dir, domain=domain), domain=domain)
    _LOG.info("%s: %d rows after the shared-row filter", domain, frame.height)
    coverage = coverage_table(frame=frame)
    job_list = jobs(domain=domain, frame=frame)
    _LOG.info("%s: %d jobs", domain, len(job_list))
    return DomainInputs(candidates, frame, dropped, job_list, coverage)


def main() -> int:
    """Build rows, folds and jobs, then dry-run, fit, fabricate, or report from saved losses."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = _repo_data_dir() / "studies" / "nwp_forecast_comparison"
    parser.add_argument("--input-dir", type=Path, default=default_dir)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Build rows, folds and jobs; stop.")
    parser.add_argument("--report-only", action="store_true", help="Report from saved losses.")
    parser.add_argument("--fit-missing", action="store_true", help="Fit only unsaved arms.")
    parser.add_argument(
        "--synthetic-losses",
        action="store_true",
        help="Fabricate losses instead of fitting; needs an explicit --output-dir outside data/.",
    )
    args = parser.parse_args()
    if args.synthetic_losses and (
        args.output_dir is None or _is_under_real_data(path=args.output_dir)
    ):
        msg = "--synthetic-losses needs an explicit --output-dir outside data/studies/"
        raise SystemExit(msg)
    output_dir: Path = args.output_dir or default_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs: dict[DomainType, DomainInputs] = {
        "solar": _build_inputs(input_dir=args.input_dir, domain="solar"),
        "wind": _build_inputs(input_dir=args.input_dir, domain="wind"),
    }
    if args.dry_run:
        for domain, domain_inputs in inputs.items():
            print(f"\n{domain}: {domain_inputs.frame.height} rows, {len(domain_inputs.jobs)} jobs")
            for arm, setting, columns, _hyper_parameters in domain_inputs.jobs:
                print(f"  {arm} / {setting}: {columns}")
        return 0

    losses: dict[DomainType, pl.DataFrame] = {}
    for domain, domain_inputs in inputs.items():
        losses[domain] = _obtain_losses(
            args=args, domain=domain, inputs=domain_inputs, output_dir=output_dir
        )
        if not args.report_only:
            save_outputs(
                output_dir=output_dir,
                domain=domain,
                losses=losses[domain],
                frame=domain_inputs.frame,
            )
    write_report(output_dir=output_dir, inputs=inputs, losses=losses)
    return 0


if __name__ == "__main__":
    sys.exit(main())
