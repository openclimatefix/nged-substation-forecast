"""Score power forecasts from every planned and exploratory weather product, at matched leads.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>. Reads the per-technology
arm-input frames `build_forecast_inputs.py` writes, assigns folds, fits every arm out of fold,
brackets each Previous Runs product against ENS, blends the open-data forecasts against ENS alone,
and writes `report.md`. See `plans/nwp-forecast-comparison.md` for the full design and
`studies/nwp_forecast_comparison/README.md` for what has actually been run.

**Rows.** `rows()` reads `<domain>_forecast_inputs.parquet`, keeps 2024-12-01 onwards (the first
whole month after IFS Cycle 49r1), drops the part-month that straddles the UKV PS47 upgrade
(2026-01), and keeps only the rows where every *planned* arm's columns are present — an exploratory
arm is fitted on the same rows with its own small gaps left as missing values, which XGBoost routes
natively.

**Folds.** `assign_folds_with_eras()` cuts five folds of whole months inside three eras (before
2025-10-01; 2025-10-01 to the UKV upgrade; after it) using `studies.cross_validation.cut_eras`,
which PR #885 (not yet merged) adds. The import happens inside that one function, so the rest of
this module — including `--dry-run`, which stops before folds are used for anything but the
coverage table — stays importable while #885 is outstanding. TODO(#885): once merged, this import
needs no guard and `ERA_FOLD_OFFSETS`/`ERA_START_MONTHS` should move to whatever #885 names them,
in place of this module's own placeholders (see `NWP_ERA_FOLD_OFFSETS` below).

**Arms.** Every arm's feature columns come from `arm_columns()`: the calendar columns, the
sun-position columns for solar, and each named product's own weather fields — so every arm of one
domain carries the same column count, and a blend's three products each contribute their own
fields once. `jobs()` builds one `Job` per (arm, hyperparameter setting): the planned contrasts'
arms are fitted at both `PRIMARY_HYPER_PARAMETERS` and `SENSITIVITY_HYPER_PARAMETERS`; every other
arm is exploratory and fitted once. A product whose columns are not in the input frame (GEFS, while
its download is incomplete) contributes no job.

**Fitting, baselines, intervals, brackets.** `run_jobs()` calls
`studies.cross_validation.out_of_fold_losses` per (arm, site), reusing the fold, seed and
absolute-error design the ENS-horizons page uses. `baseline_losses()` scores persistence and
diurnal persistence directly (no model). `bracket_verdicts()` and `blend_verdicts()` turn saved
losses into the plan's published verdicts through `studies.bootstrap.bracket_verdict` and
`bootstrap_difference`. `check_ens_monotonicity()` is the guard the whole bracket rests on: ENS's
error must not fall as its lead rises, checked per day pair and per 3-hour UTC band.

Run it with `uv run python studies/nwp_forecast_comparison/nwp_forecast_comparison.py
--output-dir <dir> --input-dir <dir>`. `--dry-run` builds the rows, the folds, and the job list,
prints the coverage table and the job list, and stops before any fit. `--report-only` rebuilds
`report.md` from previously saved losses. `--fit-missing` fits only the arms `report.md` does not
already cover.
"""

import argparse
import concurrent.futures
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, TypedDict

import polars as pl
from build_forecast_inputs import (
    ENS_DAYS,
    PRODUCT_DAY_OFFSETS,
    PRODUCT_SLUGS,
    SOLAR_ONLY_PRODUCTS,
)
from contracts.settings import PROJECT_ROOT
from studies.blending import climatology_permutation
from studies.bootstrap import (
    BootstrapInterval,
    bootstrap_absolute,
    bootstrap_difference,
    bracket_verdict,
)
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_HYPER_PARAMETERS,
    HyperParameters,
    out_of_fold_losses,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

DomainType = Literal["solar", "wind"]

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss every table reports: each row's clamped error over its own generator's capacity."""

TARGET: Final[str] = "power_mw"
"""The column every arm's XGBoost model is fitted to predict."""

MAX_CONCURRENT_FITS: Final[int] = 8
"""How many (arm, site) fits run at once; each releases the GIL while XGBoost trains."""

ROW_SET_START: Final[datetime] = datetime(2024, 12, 1, tzinfo=UTC)
"""The first hour the study's rows cover (the first whole month after IFS Cycle 49r1)."""

DROPPED_MONTHS: Final[tuple[str, ...]] = ("2026-01",)
"""Calendar months dropped entirely: the part-month straddling the UKV PS47 upgrade
(2026-01-21), which #885 also drops."""

# --- Folds --------------------------------------------------------------------------------------

NWP_ERA_START_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2026-02")
"""The first month of each era after the first: era 0 is before 2025-10; era 1 is 2025-10 to
2025-12 plus the dropped part-month; era 2 starts once the UKV upgrade has settled into a whole
month. Matches the plan's three-era design, which #885 also uses for its own reasons (HRES's
archive source changes on 2025-10-01)."""

NWP_ERA_FOLD_OFFSETS: Final[dict[int, int]] = {0: 0, 1: 0, 2: 3}
"""How far each era's fold numbers are rotated, so no calendar month is held out of every era at
once (#868). #885's own rotation, `{0: 0, 1: 0, 2: 2}`, leaves two solar (site, fold, calendar
month) cells uncovered on this study's rows. `search_fold_offsets` on this study's solar and wind
rows, which reads only which hours exist and no error, returns this rotation first among those
that cover every month in both technologies. `coverage_table` re-checks it on every run."""


def assign_folds_with_eras(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Cut five folds of whole months inside three eras, using #885's not-yet-merged helper.

    TODO(#885): `studies.cross_validation.cut_eras`, `.rotate_folds`, `.calendar_month_coverage`,
    `.uncovered_months` and `.raise_on_uncovered_months` are added by PR #885
    (`origin/ens-hres-past-wind`), not yet on `main`. The import lives inside this function, not
    at module level, so the rest of this module stays importable — including `--dry-run` up to the
    point folds are assigned — while #885 is outstanding.

    Args:
        frame: Rows carrying `site` and `time`.

    Returns:
        `frame` with `month` (`%Y-%m`), `era_code`, `era` and `fold`.

    Raises:
        ImportError: If #885 is not on the branch this runs against.
    """
    from studies.cross_validation import cut_eras

    labelled = frame.with_columns(month=pl.col("time").dt.strftime("%Y-%m"))
    return cut_eras(
        frame=labelled, first_months=NWP_ERA_START_MONTHS, fold_offsets=NWP_ERA_FOLD_OFFSETS
    )


def coverage_table(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the (site, fold, calendar month) coverage table, and raise on any uncovered month.

    TODO(#885): imports `calendar_month_coverage` and `raise_on_uncovered_months` from #885, same
    reasoning as `assign_folds_with_eras`.

    Args:
        frame: Rows carrying `site`, `fold` and `time`.

    Returns:
        `studies.cross_validation.calendar_month_coverage`'s result.

    Raises:
        ValueError: If a held-out calendar month occurring in two years has no training row.
    """
    from studies.cross_validation import (
        calendar_month_coverage,
        raise_on_uncovered_months,
    )

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


def arm_columns(*, domain: DomainType, prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """Return one arm's full, fixed-length feature-column tuple.

    Every arm of one domain that names the same number of products gets the same column count
    (`colsample_bytree=1`, no column subsampling, so an unequal count would favour the wider arm —
    see `docs/roadmap` on the blending study's control). A single-product arm passes one prefix; a
    blend passes each of its products' prefixes, in the order they are shown.

    Args:
        domain: `solar` or `wind`.
        prefixes: Each named product's own weather-column prefix, such as `ens_mean_day1` or
            `ukv_day1`.

    Returns:
        The calendar columns, the sun-position columns for solar, then each prefix's own weather
        fields in order.
    """
    calendar = (
        CALENDAR_COLUMNS if domain == "wind" else (*CALENDAR_COLUMNS, *SOLAR_POSITION_COLUMNS)
    )
    weather_fn = _wind_weather_fields if domain == "wind" else _solar_weather_fields
    weather = tuple(column for prefix in prefixes for column in weather_fn(prefix=prefix))
    return (*calendar, *weather)


# --- Rows -----------------------------------------------------------------------------------

PLANNED_PREFIXES: Final[dict[DomainType, tuple[str, ...]]] = {
    "solar": (
        "ens_mean_day0",
        "ens_mean_day1",
        "ukv_day1",
        "icon_eu_day1",
        "icon_eu_day2",
        "ifs025_day1",
        "ifs025_day2",
        "gefs_mean_day1",
    ),
    "wind": (
        "ens_mean_day0",
        "ens_mean_day1",
        "ukv_day1",
        "icon_eu_day1",
        "icon_eu_day2",
        "ifs025_day1",
        "ifs025_day2",
        "gefs_mean_day1",
    ),
}
"""Each domain's planned arms (P1-P4 and their bracket sides), by weather-column prefix. Fitted at
both hyperparameter settings; every other arm is exploratory and fitted once."""


def _repo_data_dir() -> Path:
    """Return the shared `data/` directory, resolving a linked worktree to the main checkout.

    See `verify_previous_runs_leads._repo_data_dir` for the reasoning; duplicated here because
    study scripts in different directories cannot import one another's private helpers.
    """
    root = PROJECT_ROOT
    marker = root / ".git"
    if marker.is_file():
        pointer = marker.read_text().removeprefix("gitdir:").strip()
        git_dir = Path(pointer)
        if git_dir.parent.name == "worktrees":
            root = git_dir.parent.parent.parent
    import os

    return Path(os.environ.get("DATA_PATH_INTERNAL") or root / "data")


def rows(*, input_dir: Path, domain: DomainType) -> pl.DataFrame:
    """Load one technology's shared row set: the rows every planned arm can score.

    Args:
        input_dir: Where `build_forecast_inputs.py` wrote `<domain>_forecast_inputs.parquet`.
        domain: `solar` or `wind`.

    Returns:
        Rows from `ROW_SET_START`, `DROPPED_MONTHS` removed, `fold`/`era_code` assigned, kept only
        where every planned arm whose columns are actually present in the input frame is complete.
        A planned arm entirely missing from the input frame (GEFS, while ungated) is skipped rather
        than dropping every row, and is reported by `main` as not built this pass.
    """
    frame = pl.read_parquet(input_dir / f"{domain}_forecast_inputs.parquet").filter(
        pl.col("time") >= ROW_SET_START
    )
    frame = frame.with_columns(month=pl.col("time").dt.strftime("%Y-%m")).filter(
        ~pl.col("month").is_in(DROPPED_MONTHS)
    )
    prefixes = PLANNED_PREFIXES[domain]
    weather_fields_fn = _wind_weather_fields if domain == "wind" else _solar_weather_fields
    weather_columns: list[str] = []
    missing_prefixes: list[str] = []
    for prefix in prefixes:
        fields = weather_fields_fn(prefix=prefix)
        if all(column in frame.columns for column in fields):
            weather_columns.extend(fields)
        else:
            missing_prefixes.append(prefix)
    if missing_prefixes:
        _LOG.warning(
            "%s: planned arm(s) not in the input frame, skipped from the shared-row filter: %s",
            domain,
            sorted(missing_prefixes),
        )
    complete = frame.filter(pl.all_horizontal(pl.col(c).is_not_null() for c in weather_columns))
    return assign_folds_with_eras(frame=complete)


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
    """Return every ENS arm's `ens_<way>_day<N>` prefix, mean and control, at `ENS_DAYS`."""
    return [f"ens_{way}_day{day}" for day in ENS_DAYS for way in ("mean", "control")]


BLEND_ARMS: Final[dict[str, tuple[str, str, str]]] = {
    "blend_p4a": ("ens_mean_day1", "icon_eu_day1", "ifs025_day1"),
    "blend_p4b": ("ens_mean_day1", "icon_eu_day2", "ifs025_day2"),
}
"""The two planned blends' product prefixes: ENS's own day-1 mean plus the other two products, at
the optimistic (P4a) and conservative (P4b) lead."""

BLEND_GUARD_SUFFIX: Final[str] = "_permuted"
"""Appended to a blend's non-ENS columns by `climatology_permutation`, forming each guard's own
column names."""


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
        frame: `rows()`'s result, or any frame carrying the same columns (used to skip a product
            not yet built, such as GEFS).

    Returns:
        One `Job` per (arm, setting): planned arms at both `primary` and `sensitivity`; every
        other arm (exploratory single products, the ENS control member, and the blend guards) at
        `primary` only.
    """
    available = set(frame.columns)
    planned = set(PLANNED_PREFIXES[domain])
    single_product_prefixes = [*_product_prefixes(domain=domain), *_ens_prefixes()]
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
        guard_prefixes = _blend_guard_prefixes(blend=blend)
        guard_columns = arm_columns(domain=domain, prefixes=guard_prefixes)
        if all(column in available for column in guard_columns):
            output.extend(
                (f"{blend}_control", setting, guard_columns, SETTINGS[setting])
                for setting in SETTINGS
            )
    return output


def add_blend_guard_columns(*, frame: pl.DataFrame, domain: DomainType) -> pl.DataFrame:
    """Add each blend's guard columns: the non-ENS products' weather, permuted among matched hours.

    `studies.blending.climatology_permutation` moves each product's weather among the rows sharing
    a site, calendar month and hour of day, so the guard keeps the blend's column count while
    removing the other products' real weather. A product's direction sine and cosine (wind) are
    permuted together as one group, so a permuted row still sits on the unit circle.

    Args:
        frame: Rows carrying every blend product's weather columns.
        domain: `solar` or `wind`.

    Returns:
        `frame` with `<column>_permuted` for every non-ENS blend product's weather columns.
    """
    by = ("site", "month", "hour_of_day")
    seed = 20260920
    output = frame
    for prefixes in BLEND_ARMS.values():
        for index, prefix in enumerate(prefixes[1:], start=1):
            fields = (
                _wind_weather_fields(prefix=prefix)
                if domain == "wind"
                else _solar_weather_fields(prefix=prefix)
            )
            if not all(column in output.columns for column in fields):
                continue
            groups = (
                [
                    (f"{prefix}_speed_100m",),
                    (f"{prefix}_sin_100m", f"{prefix}_cos_100m"),
                    (f"{prefix}_speed_10m",),
                ]
                if domain == "wind"
                else [(f"{prefix}_ghi",), (f"{prefix}_temp",)]
            )
            permuted = climatology_permutation(
                frame=output,
                column_groups=groups,
                by=by,
                seed=seed + index,
                suffix=BLEND_GUARD_SUFFIX,
            )
            permuted_columns = [
                f"{column}{BLEND_GUARD_SUFFIX}" for group in groups for column in group
            ]
            output = output.join(
                permuted.select(*by, "site", "time", *permuted_columns),
                on=["site", "time", *by],
                how="left",
            )
    return output


def run_jobs(*, domain: DomainType, frame: pl.DataFrame, jobs_: list[Job]) -> pl.DataFrame:
    """Fit every (arm, site) job out of fold, concurrently, and stack the losses.

    Args:
        domain: `solar` or `wind`.
        frame: `rows()`'s result, already carrying `fold`.
        jobs_: `jobs()`'s result.

    Returns:
        Every job's losses, stacked, labelled with `arm` and `setting`.
    """
    del domain
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


# --- Baselines, brackets and the ENS monotonicity guard --------------------------------------


def baseline_losses(*, frame: pl.DataFrame, day: int) -> pl.DataFrame:
    """Score persistence and diurnal persistence directly, no model, at one band.

    Args:
        frame: Rows carrying `power_mw`, `cap_mw`, `effective_capacity_mw`, `constrained`, `month`,
            and `persistence_day<day>`/`diurnal_persistence_day<day>`.
        day: The band.

    Returns:
        One row per (site, time, baseline) with `arm`, `month` and `METRIC`.
    """
    outputs = []
    for name in ("persistence", "diurnal_persistence"):
        column = f"{name}_day{day}"
        if column not in frame.columns:
            continue
        capacity = pl.col("effective_capacity_mw")
        error = (pl.col(column) - pl.col("power_mw")).abs().clip(upper_bound=capacity)
        outputs.append(
            frame.select(
                "site",
                "time",
                "month",
                arm=pl.lit(f"{name}_day{day}"),
                **{METRIC: error / capacity},
            )
        )
    if not outputs:
        return pl.DataFrame()
    return pl.concat(outputs)


class BlendVerdict(TypedDict):
    """One blend's contrast against ENS alone, its guard, and the plan's published verdict."""

    against_ens: BootstrapInterval
    guard: BootstrapInterval
    verdict: str


def blend_verdicts(*, losses: pl.DataFrame, blend: str) -> BlendVerdict:
    """Turn one blend's contrast and guard into the plan's published verdict.

    A blend **lowers the day-ahead error** if P4b is negative and significant and its guard is
    negative and significant (the gain survives the conservative lead and comes from the weather);
    it **may lower the error** if only P4a and its guard are; otherwise it makes no detectable
    difference.

    Args:
        losses: Per-row losses carrying `blend`, `ens_mean_day1` and `{blend}_control` as `arm`
            values.
        blend: `"blend_p4a"` or `"blend_p4b"`.

    Returns:
        The blend-minus-ENS interval, the guard interval, and the verdict string.
    """
    assert_equal_rows(losses=losses, treatment=blend, reference="ens_mean_day1")
    against_ens = bootstrap_difference(
        losses=losses, treatment=blend, reference="ens_mean_day1", metric=METRIC
    )
    assert_equal_rows(losses=losses, treatment=blend, reference=f"{blend}_control")
    guard = bootstrap_difference(
        losses=losses, treatment=blend, reference=f"{blend}_control", metric=METRIC
    )
    negative_significant = against_ens["upper_95"] < 0.0 and guard["upper_95"] < 0.0
    if blend == "blend_p4b" and negative_significant:
        verdict = "lowers the day-ahead error"
    elif blend == "blend_p4a" and negative_significant:
        verdict = "may lower the error"
    else:
        verdict = "no detectable difference"
    return {"against_ens": against_ens, "guard": guard, "verdict": verdict}


EXACT_LEAD_WIND_HOURS: Final[dict[str, tuple[int, ...]]] = {
    "icon_eu_day1": tuple(range(6)),
    "ukv_day1": tuple(range(3)),
}
"""Each product's UTC hours where its day-1 lead equals ENS day 1's exactly (`h < n` for its run
cycle `n`): 00-05 for a 6-hourly product, 00-02 for UKV. Exploratory; no extra fit — a re-read of
the saved losses."""


def exact_lead_wind_rereads(*, losses: pl.DataFrame) -> dict[str, BootstrapInterval]:
    """Re-read the upper-side wind contrast on the hours where no bracket is needed.

    At these hours a product's day-1 lead equals ENS day 1's exactly, so `product − ENS day 1`
    is itself a matched-lead contrast rather than the upper side of a bracket, and tests whether
    the bracket's inference holds where it is not needed.

    Args:
        losses: Per-row wind losses carrying `time` and each product's day-1 arm.

    Returns:
        Each product's `product − ens_mean_day1` interval, restricted to its exact-lead hours.
    """
    output: dict[str, BootstrapInterval] = {}
    for arm, hours in EXACT_LEAD_WIND_HOURS.items():
        subset = losses.filter(pl.col("time").dt.hour().is_in(hours))
        if subset.filter(pl.col("arm") == arm).is_empty():
            continue
        assert_equal_rows(losses=subset, treatment=arm, reference="ens_mean_day1")
        output[arm] = bootstrap_difference(
            losses=subset, treatment=arm, reference="ens_mean_day1", metric=METRIC
        )
    return output


def leaderboard(*, losses: pl.DataFrame, arms: tuple[str, ...]) -> pl.DataFrame:
    """Return each arm's own absolute error, with its interval, for the day-1 leaderboard.

    Args:
        losses: Per-row losses carrying every arm in `arms`.
        arms: The arms to leaderboard, in the order they are shown.

    Returns:
        One row per arm with `value`, `lower_95` and `upper_95`, in `arms`' order.
    """
    rows_out = []
    for arm in arms:
        if losses.filter(pl.col("arm") == arm).is_empty():
            continue
        interval = bootstrap_absolute(losses=losses, arm=arm, metric=METRIC)
        rows_out.append({"arm": arm, **interval})
    return pl.DataFrame(rows_out)


def assert_equal_rows(*, losses: pl.DataFrame, treatment: str, reference: str) -> None:
    """Raise unless `treatment` and `reference` score exactly the same (site, time) rows.

    `studies.bootstrap.paired_differences` inner-joins the two arms, which would silently shrink a
    contrast's rows rather than fail if they were not already equal; every contrast in this module
    calls this first.

    Args:
        losses: Per-row losses carrying both arms.
        treatment: One arm's name.
        reference: The other arm's name.

    Raises:
        ValueError: If the two arms' (site, time) row sets differ.
    """
    treatment_rows = set(
        losses.filter(pl.col("arm") == treatment).select("site", "time").iter_rows()
    )
    reference_rows = set(
        losses.filter(pl.col("arm") == reference).select("site", "time").iter_rows()
    )
    if treatment_rows != reference_rows:
        msg = (
            f"assert_equal_rows: {treatment!r} and {reference!r} score different rows "
            f"({len(treatment_rows)} vs {len(reference_rows)}, "
            f"{len(treatment_rows ^ reference_rows)} not shared)"
        )
        raise ValueError(msg)


class BracketVerdict(TypedDict):
    """One product's day-1 bracket against ENS: both sides' intervals and the published verdict."""

    lower: BootstrapInterval
    upper: BootstrapInterval
    verdict: str


def bracket_verdicts(*, losses: pl.DataFrame, product_day1: str) -> BracketVerdict:
    """Turn one product's day-1 bracket against ENS into a published verdict.

    Args:
        losses: Per-row losses carrying `ens_mean_day0`, `ens_mean_day1` and `product_day1` as
            `arm` values.
        product_day1: The product's day-1 arm name, such as `ukv_day1`.

    Returns:
        `lower` (product − ENS day 0), `upper` (product − ENS day 1), and `verdict` (`"beats"`,
        `"loses"` or `"unresolved"`).
    """
    for arm in (product_day1, "ens_mean_day0", "ens_mean_day1"):
        if losses.filter(pl.col("arm") == arm).is_empty():
            msg = f"bracket_verdicts: no rows for arm {arm!r}"
            raise ValueError(msg)
    assert_equal_rows(losses=losses, treatment=product_day1, reference="ens_mean_day0")
    lower = bootstrap_difference(
        losses=losses, treatment=product_day1, reference="ens_mean_day0", metric=METRIC
    )
    assert_equal_rows(losses=losses, treatment=product_day1, reference="ens_mean_day1")
    upper = bootstrap_difference(
        losses=losses, treatment=product_day1, reference="ens_mean_day1", metric=METRIC
    )
    return {
        "lower": lower,
        "upper": upper,
        "verdict": bracket_verdict(lower_side=lower, upper_side=upper),
    }


def check_ens_monotonicity(
    *, losses: pl.DataFrame, day_pairs: tuple[tuple[int, int], ...]
) -> pl.DataFrame:
    """Check that ENS's error does not fall as its lead rises, per day pair and 3-hour UTC band.

    The bracket rests on this holding: the lower side needs `ENS day N−1 <= ENS day N` at the
    matching lead. A point estimate at or below zero in any band voids, in that band, every
    bracket resting on that pair.

    Args:
        losses: Per-row losses carrying `ens_mean_day<N>` arms, `time` and `month`.
        day_pairs: The (earlier, later) day pairs the study's brackets use: (0, 1), (1, 2), (2, 3).

    Returns:
        One row per (day pair, 3-hour band) with the bootstrap interval of `later − earlier` and
        whether its point estimate is positive (ENS's error rose with lead, as it should).
    """
    banded = losses.with_columns(band=(pl.col("time").dt.hour() // 3) * 3)
    rows_out = []
    for earlier, later in day_pairs:
        earlier_arm, later_arm = f"ens_mean_day{earlier}", f"ens_mean_day{later}"
        if banded.filter(pl.col("arm") == earlier_arm).is_empty():
            continue
        for band in sorted(banded["band"].unique().to_list()):
            subset = banded.filter(pl.col("band") == band)
            assert_equal_rows(losses=subset, treatment=later_arm, reference=earlier_arm)
            interval = bootstrap_difference(
                losses=subset, treatment=later_arm, reference=earlier_arm, metric=METRIC
            )
            rows_out.append(
                {
                    "earlier_day": earlier,
                    "later_day": later,
                    "band_start_utc": band,
                    "difference": interval["difference"],
                    "lower_95": interval["lower_95"],
                    "upper_95": interval["upper_95"],
                    "monotone": interval["difference"] > 0,
                }
            )
    return pl.DataFrame(rows_out)


# --- Report -----------------------------------------------------------------------------------


def fingerprint(*, frame: pl.DataFrame) -> str:
    """Return a short hash of a losses frame, floats cast to Float32 first.

    Args:
        frame: The frame to fingerprint (typically the saved per-row losses).

    Returns:
        A hex digest, stable across re-runs on the same data and code.
    """
    import hashlib

    floats = [name for name, dtype in frame.schema.items() if dtype in (pl.Float32, pl.Float64)]
    cast = frame.with_columns([pl.col(name).cast(pl.Float32) for name in floats])
    return hashlib.sha256(cast.write_ipc(None).getbuffer()).hexdigest()[:16]


def write_report(
    *,
    output_dir: Path,
    domain_jobs: dict[DomainType, list[Job]],
    coverage: dict[DomainType, pl.DataFrame],
    losses: dict[DomainType, pl.DataFrame] | None,
) -> None:
    """Write `report.md`: every arm's columns, the coverage table, and (if fitted) the losses.

    Args:
        output_dir: Where `report.md` is written.
        domain_jobs: Each domain's `jobs()` result.
        coverage: Each domain's `coverage_table()` result.
        losses: Each domain's stacked losses, or `None` before any fit (a dry run).
    """
    lines = ["# Weather forecasts for power, compared at matched lead times — report", ""]
    for domain, job_list in domain_jobs.items():
        lines.append(f"## {domain}")
        lines.append("")
        lines.append(f"{len(job_list)} jobs.")
        lines.append("")
        lines.append("| Arm | Setting | Columns |")
        lines.append("|---|---|---|")
        for arm, setting, columns, _hyper_parameters in job_list:
            lines.append(f"| {arm} | {setting} | {', '.join(columns)} |")
        lines.append("")
        cov = coverage.get(domain)
        if cov is not None:
            uncovered = cov.filter(~pl.col("covered"))
            lines.append(
                f"Coverage table: {cov.height} (site, fold, month) cells, "
                f"{uncovered.height} not covered by training data (single-year months excepted)."
            )
            lines.append("")
        if losses is not None and domain in losses:
            lines.append(f"Losses fingerprint: `{fingerprint(frame=losses[domain])}`.")
            lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


# --- CLI ----------------------------------------------------------------------------------------


def main() -> int:
    """Build rows and folds, print the coverage table and job list, and (unless `--dry-run`) fit."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_repo_data_dir() / "studies" / "nwp_forecast_comparison",
        help="Where build_forecast_inputs.py wrote the per-technology arm-input parquets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_data_dir() / "studies" / "nwp_forecast_comparison",
        help="Where report.md and the saved losses are written.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build rows, folds and jobs; stop.")
    parser.add_argument("--report-only", action="store_true", help="Rebuild report.md, no fit.")
    parser.add_argument("--fit-missing", action="store_true", help="Fit only arms not yet saved.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    domain_frames: dict[DomainType, pl.DataFrame] = {}
    domain_jobs: dict[DomainType, list[Job]] = {}
    coverage: dict[DomainType, pl.DataFrame] = {}
    for domain in ("solar", "wind"):
        frame = rows(input_dir=args.input_dir, domain=domain)
        _LOG.info("%s: %d rows after the shared-row filter", domain, frame.height)
        cov = coverage_table(frame=frame)
        coverage[domain] = cov
        job_list = jobs(domain=domain, frame=frame)
        _LOG.info("%s: %d jobs", domain, len(job_list))
        domain_frames[domain] = frame
        domain_jobs[domain] = job_list

    if args.dry_run:
        for domain, job_list in domain_jobs.items():
            print(f"\n{domain}: {domain_frames[domain].height} rows, {len(job_list)} jobs")
            for arm, setting, columns, _hp in job_list:
                print(f"  {arm} / {setting}: {columns}")
        return 0

    if args.report_only:
        write_report(
            output_dir=args.output_dir, domain_jobs=domain_jobs, coverage=coverage, losses=None
        )
        return 0

    # --fit-missing and the full run both need an actual fit, which this pass does not run (see
    # the study coordinator's gate on the GEFS download and the worktree data-folder rule in
    # plans/nwp-forecast-comparison.md, "Process and constraints").
    _LOG.error(
        "Fitting is not run by this pass (see the module docstring and the plan's process "
        "constraints); use --dry-run or --report-only."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
