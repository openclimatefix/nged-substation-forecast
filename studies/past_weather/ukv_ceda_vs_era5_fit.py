"""Set B of the UKV-CEDA against ERA5 study: fit one XGBoost model per generator and product.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1024>. It reads the rows that
`ukv_ceda_vs_era5_build.py` wrote, fits an XGBoost model per metered generator and per arm, scores
each model's power error out of fold, applies the decision rule by code, and writes `report.md` and
`decision.md`. Set A's intervals come from `ukv_ceda_station_scores.py`.

**Arms.** Each wind arm has 7 columns and each solar arm has 10, built by one function per arm type
(`ukv_ceda_vs_era5_build.wind_arm_columns` and `solar_arm_columns`) and printed into the report. The
two wind arms differ in the product's four wind columns: ERA5 reads its 100 m speed and direction
and its 10 m speed, and UKV-CEDA has no 100 m wind, so it reads its 10 m speed and direction and its
925 hPa speed. The contrast therefore mixes product and height, and the report claims no cause. The
two solar arms differ only in the temperature column, and both read CAMS global, beam and diffuse
irradiance.

**Planned contrasts (UKV-CEDA minus ERA5, a negative value favouring UKV-CEDA).** P3 is the
as-available wind arms and P4 is the solar arms, each fitted at both hyperparameter settings. P1 and
P2 are set A's. The margins are frozen in `ukv_ceda_vs_era5_build` before any result: 0.16 points of
capacity for wind and 0.06 for solar. The rule is `decide_wind` and `decide_temperature`.

**Controls and checks, all exploratory, at the primary setting.** The negative control shuffles the
product's weather columns within a generator, year-month, and hour of day. For wind, one joint
permutation moves the four columns of a product, and for solar only the temperature column moves, so
the CAMS irradiance stays intact. The shuffled UKV-CEDA arm minus the shuffled ERA5 arm should
differ by about zero. The hour-ending pair builds the power hour as the solar studies do, ending at
the label, and scans the power-hour offset for both products. The keep-zero-hours block refits both
wind arms on rows that keep the hours holding an exactly zero half-hour. One wind arm is refitted on
the CPU, to give the difference between a GPU and a CPU fit.

**Folds.** `studies.ukv_ceda_stores.with_ukv_eras` cuts folds of whole months inside each of the
three UKV eras, `era_code` is a feature, and `colsample_bytree` stays at 1. The folds are fixed when
the rows are built.

**Intervals.** Each interval resamples whole calendar months, paired across arms, and one of the
three fitting seeds, 2,000 times (`studies.bootstrap.bootstrap_difference`). A scope with fewer than
six months gets a point estimate and no interval. Every score is divided by its row's own
generator's capacity, and an export-cap `constrained` solar hour is excluded from training and still
scored.

Run it with `uv run python studies/past_weather/ukv_ceda_vs_era5_fit.py`. `--dry-run` lists every
fit and fits nothing. A fit needs `--verified`, which says `verify.md` was run and read.
`--report-only` rebuilds the intervals and the reports from the saved losses after checking each
fingerprint. A fresh run stops (`refuse_to_overwrite`) while an output exists. Only one agent may
run it at a time, because every worktree shares one data folder.
"""

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, TypedDict

import polars as pl
import xgboost
from ens_past_solar import _arm_columns_lines, _fingerprint
from studies.arm_runner import MAX_CONCURRENT_FITS, Job, run_all
from studies.bootstrap import (
    MIN_MONTHS_FOR_INTERVAL,
    bootstrap_absolute,
    bootstrap_difference,
    paired_differences,
)
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_HYPER_PARAMETERS,
    DeviceType,
)
from studies.guards import refuse_to_overwrite
from studies.ukv_ceda_stores import ERA_FIRST_MONTHS, STRADDLING_MONTHS
from ukv_ceda_station_scores import INTERVALS_NAME as STATION_INTERVALS_NAME
from ukv_ceda_station_scores import PRIMARY_SCORE, TREND_BLOCKS, monthly_trend
from ukv_ceda_vs_era5_build import (
    EARLY_END_MONTH,
    MARGIN_SOLAR_PP,
    MARGIN_WIND_PP,
    MAX_MONTH_LOSS_SHARE,
    OUTPUT_DIR,
    SOLAR_ROWS_NAME,
    WIND_HOUR_STARTING_ROWS_NAME,
    WIND_KEEP_ZERO_ROWS_NAME,
    WIND_MATCHED_ROWS_NAME,
    WIND_ROWS_NAME,
    ContrastReadingType,
    check_arm_widths,
    contrast_reading,
    matched_wind_arm_columns,
    open_ukv_stores,
    run_status_lines,
    solar_arm_columns,
    wind_arm_columns,
)

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss every table reports: each row's clamped error over its own generator's capacity."""

PERCENTAGE_POINTS: Final[float] = 100.0

PRIMARY_SETTING: Final[str] = "pooled"
SECOND_SETTING: Final[str] = "sensitivity"

DomainType = Literal[
    "wind",
    "solar",
    "wind_keep_zero",
    "wind_matched",
    "wind_hour_starting",
    "wind_lead0",
    "wind_leads01",
    "solar_lead0",
    "solar_leads01",
]

RESTRICTED_DOMAINS: Final[Mapping[DomainType, tuple[DomainType, int]]] = {
    "wind_lead0": ("wind", 1),
    "wind_leads01": ("wind", 2),
    "solar_lead0": ("solar", 1),
    "solar_leads01": ("solar", 2),
}
"""Each analysis-only domain, as its base domain and how many leads it keeps (0 to n - 1).

The rows are the base domain's rows whose hour of day modulo 6 is below the count, so the XGBoost
models are trained and scored on the first leads of UKV-CEDA's runs only.
"""

OPTIONAL_DOMAINS: Final[tuple[DomainType, ...]] = (
    "wind_matched",
    "wind_hour_starting",
    *RESTRICTED_DOMAINS,
)
"""Domains whose rows and fits are added after the first report, so they exist only once built."""

VERIFY_NAME: Final[str] = "verify.md"
REPORT_NAME: Final[str] = "report.md"
DECISION_NAME: Final[str] = "decision.md"
INTERVALS_NAME: Final[str] = "intervals.parquet"
STAMP_NAME: Final[str] = "fit_stamp.json"
MATCHED_STAMP_NAME: Final[str] = "fit_stamp_matched.json"
LEAD_STAMP_NAME: Final[str] = "fit_stamp_lead_restricted.json"
HOUR_STARTING_STAMP_NAME: Final[str] = "fit_stamp_hour_starting.json"
CPU_REFIT_ARM: Final[str] = "era5_wind_cpu_refit"
"""The arm refitted on the CPU for the noise floor: ERA5's wind arm."""

ROW_NAMES: Final[Mapping[DomainType, str]] = {
    "wind": WIND_ROWS_NAME,
    "solar": SOLAR_ROWS_NAME,
    "wind_keep_zero": WIND_KEEP_ZERO_ROWS_NAME,
    "wind_matched": WIND_MATCHED_ROWS_NAME,
    "wind_hour_starting": WIND_HOUR_STARTING_ROWS_NAME,
}
"""The build's row file of each domain. The matched-height rows exist only after a later step."""

MARGINS_PP: Final[Mapping[DomainType, float]] = {
    "wind": MARGIN_WIND_PP,
    "solar": MARGIN_SOLAR_PP,
    "wind_keep_zero": MARGIN_WIND_PP,
    "wind_matched": MARGIN_WIND_PP,
    "wind_hour_starting": MARGIN_WIND_PP,
    "wind_lead0": MARGIN_WIND_PP,
    "wind_leads01": MARGIN_WIND_PP,
    "solar_lead0": MARGIN_SOLAR_PP,
    "solar_leads01": MARGIN_SOLAR_PP,
}
"""The margin of each domain, in percentage points of capacity."""

LICENCE_CONDITION: Final[str] = (
    "Subject to the licence: the CEDA catalogue record gives the Creative Commons "
    "Attribution-NonCommercial-ShareAlike 4.0 licence, and whether the main work's use of UKV-CEDA "
    "as training history is non-commercial is a decision for the maintainer."
)
"""Every recommendation of UKV-CEDA carries this."""

GPU_WORKERS: Final[int] = 2
"""How many (arm, generator) fits run at once on the GPU, each needing about one CPU core."""


def workers_for(*, device: DeviceType) -> int:
    """Return how many fits to run at once on a device.

    Args:
        device: XGBoost's device.

    Returns:
        `GPU_WORKERS` on a GPU, and `MAX_CONCURRENT_FITS` on the CPU.
    """
    return GPU_WORKERS if device == "cuda" else MAX_CONCURRENT_FITS


class Planned(NamedTuple):
    """One planned contrast of set B."""

    label: str
    domain: DomainType
    treatment: str
    reference: str
    registered: bool = True
    """Whether the plan named the contrast before any result. A post hoc contrast is not."""


PLANNED_CONTRASTS: Final[tuple[Planned, ...]] = (
    Planned("P3", "wind", "ukv_ceda_wind", "era5_wind"),
    Planned("P4", "solar", "solar_ukv_ceda_temp", "solar_era5_temp"),
)
"""The planned contrasts of set B, UKV-CEDA minus ERA5, each at both settings."""

MATCHED_CONTRAST: Final[Planned] = Planned(
    "post hoc matched 10 m",
    "wind_matched",
    "ukv_ceda_wind_10m",
    "era5_wind_10m",
    registered=False,
)
"""The post hoc matched-height pair: both products' 10 m wind alone, UKV-CEDA minus ERA5."""

ANALYSIS_ONLY_CONTRASTS: Final[tuple[Planned, ...]] = tuple(
    Planned(
        f"post hoc analysis-only {base.label} ({'lead 0' if n_leads == 1 else 'leads 0 to 1'})",
        domain,
        base.treatment,
        base.reference,
        registered=False,
    )
    for domain, (base_domain, n_leads) in RESTRICTED_DOMAINS.items()
    for base in PLANNED_CONTRASTS
    if base.domain == base_domain
)
"""P3 and P4 refitted on the first leads only: UKV-CEDA minus ERA5, post hoc."""

SPLIT_CONTRASTS: Final[tuple[Planned, ...]] = (
    *PLANNED_CONTRASTS,
    MATCHED_CONTRAST,
    *ANALYSIS_ONLY_CONTRASTS,
)
"""The contrasts split by scope, setting, and generator."""

POST_HOC_DOMAINS: Final[tuple[DomainType, ...]] = ("wind", "wind_matched", "solar")
"""The domains that get the post hoc lead, window, and era splits."""

PUBLISHED_WIND_WINDOW_START: Final[datetime] = datetime(2024, 8, 12, tzinfo=UTC)
"""The first day of the published wind page's window, when Open-Meteo's UKV hub wind starts."""

LEAD_CYCLE_HOURS: Final[int] = 6
"""UKV-CEDA's runs start every 6 hours, so an hour's lead is its UTC hour modulo this."""

EXPLORATORY_CONTRASTS: Final[Mapping[DomainType, tuple[tuple[str, str, str], ...]]] = {
    "wind": (
        ("control", "ukv_ceda_wind_shuffled", "era5_wind_shuffled"),
        ("era5 against its shuffled arm", "era5_wind", "era5_wind_shuffled"),
        ("UKV-CEDA against its shuffled arm", "ukv_ceda_wind", "ukv_ceda_wind_shuffled"),
        ("hour-ending pair", "ukv_ceda_wind_hour_ending", "era5_wind_hour_ending"),
        ("ERA5 power-hour offset", "era5_wind_hour_ending", "era5_wind"),
        ("UKV-CEDA power-hour offset", "ukv_ceda_wind_hour_ending", "ukv_ceda_wind"),
        ("GPU against CPU", CPU_REFIT_ARM, "era5_wind"),
    ),
    "solar": (
        ("control", "solar_ukv_ceda_temp_shuffled", "solar_era5_temp_shuffled"),
        ("era5 against its shuffled arm", "solar_era5_temp", "solar_era5_temp_shuffled"),
        (
            "UKV-CEDA against its shuffled arm",
            "solar_ukv_ceda_temp",
            "solar_ukv_ceda_temp_shuffled",
        ),
    ),
    "wind_keep_zero": (("keep zero hours", "ukv_ceda_wind", "era5_wind"),),
    "wind_matched": (),
    "wind_hour_starting": (
        ("scan, centred pair", "ukv_ceda_wind_centred", "era5_wind_centred"),
        ("scan, hour-ending pair", "ukv_ceda_wind_hour_ending", "era5_wind_hour_ending"),
        ("scan, hour-starting pair", "ukv_ceda_wind_hour_starting", "era5_wind_hour_starting"),
        ("scan, ERA5 hour-ending offset", "era5_wind_hour_ending", "era5_wind_centred"),
        ("scan, ERA5 hour-starting offset", "era5_wind_hour_starting", "era5_wind_centred"),
        ("scan, UKV-CEDA hour-ending offset", "ukv_ceda_wind_hour_ending", "ukv_ceda_wind_centred"),
        (
            "scan, UKV-CEDA hour-starting offset",
            "ukv_ceda_wind_hour_starting",
            "ukv_ceda_wind_centred",
        ),
    ),
}
"""The exploratory contrasts of each domain, as (label, treatment, reference)."""


class IntervalRecord(TypedDict):
    """One contrast of set B, UKV-CEDA minus ERA5 unless named otherwise, in the intervals table."""

    domain: str
    setting: str
    label: str
    planned: bool
    kind: str
    scope: str
    treatment: str
    reference: str
    treatment_mae_pp: float
    reference_mae_pp: float
    difference_pp: float
    lower_95_pp: float
    upper_95_pp: float
    margin_pp: float
    reading: str
    n_rows: int
    n_months: int
    enough_months: bool
    seed_spread_pp: float
    product_contrast: bool
    post_hoc: bool


# --- Jobs -----------------------------------------------------------------------------------------


def _job(*, arm: str, setting: str, columns: tuple[str, ...], target: str = "power_mw") -> Job:
    hyper_parameters = (
        PRIMARY_HYPER_PARAMETERS if setting == PRIMARY_SETTING else SENSITIVITY_HYPER_PARAMETERS
    )
    return (arm, setting, target, columns, hyper_parameters, False)


def domain_jobs(*, domain: DomainType) -> list[Job]:
    """List every fit of one domain.

    Args:
        domain: `wind`, `solar`, `wind_keep_zero`, `wind_matched`, or
            `wind_hour_starting`.

    Returns:
        The jobs: the planned arms at both settings, then the controls and checks at the primary
        setting. Every job fits one arm at each generator of the domain.
    """
    if domain in RESTRICTED_DOMAINS:
        base = RESTRICTED_DOMAINS[domain][0]
        return [job for job in domain_jobs(domain=base) if job[1] == PRIMARY_SETTING][:2]
    if domain == "wind_hour_starting":
        return [
            _job(
                arm=f"{product}_wind_{convention}",
                setting=PRIMARY_SETTING,
                columns=wind_arm_columns(product=product),
                target=target,
            )
            for product in ("era5", "ukv_ceda")
            for convention, target in (
                ("centred", "power_mw"),
                ("hour_ending", "power_hour_ending_mw"),
                ("hour_starting", "power_hour_starting_mw"),
            )
        ]
    if domain == "wind_matched":
        return [
            _job(
                arm=f"{product}_wind_10m",
                setting=setting,
                columns=matched_wind_arm_columns(product=product),
            )
            for setting in (PRIMARY_SETTING, SECOND_SETTING)
            for product in ("era5", "ukv_ceda")
        ]
    if domain == "solar":
        jobs = [
            _job(
                arm=f"solar_{product}_temp",
                setting=setting,
                columns=solar_arm_columns(product=product),
            )
            for setting in (PRIMARY_SETTING, SECOND_SETTING)
            for product in ("era5", "ukv_ceda")
        ]
        jobs += [
            _job(
                arm=f"solar_{product}_temp_shuffled",
                setting=PRIMARY_SETTING,
                columns=solar_arm_columns(product=product, shuffled=True),
            )
            for product in ("era5", "ukv_ceda")
        ]
        return jobs
    jobs = [
        _job(arm=f"{product}_wind", setting=setting, columns=wind_arm_columns(product=product))
        for setting in (PRIMARY_SETTING, SECOND_SETTING)
        for product in ("era5", "ukv_ceda")
    ]
    if domain == "wind_keep_zero":
        return [job for job in jobs if job[1] == PRIMARY_SETTING]
    jobs += [
        _job(
            arm=f"{product}_wind_shuffled",
            setting=PRIMARY_SETTING,
            columns=wind_arm_columns(product=product, shuffled=True),
        )
        for product in ("era5", "ukv_ceda")
    ]
    jobs += [
        _job(
            arm=f"{product}_wind_hour_ending",
            setting=PRIMARY_SETTING,
            columns=wind_arm_columns(product=product),
            target="power_hour_ending_mw",
        )
        for product in ("era5", "ukv_ceda")
    ]
    return jobs


def cpu_refit_jobs() -> list[Job]:
    """List the one arm refitted on the CPU: ERA5's wind arm at the primary setting.

    Returns:
        One job.
    """
    return [
        _job(arm=CPU_REFIT_ARM, setting=PRIMARY_SETTING, columns=wind_arm_columns(product="era5"))
    ]


# --- Fitting --------------------------------------------------------------------------------------


def with_actual_and_prediction(*, fitted: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the measured power and the out-of-fold prediction to the fitted losses.

    The measured power is read from each job's own target column, because the hour-ending arms fit
    a different power hour. The prediction is the measured power plus the signed error.

    Args:
        fitted: `run_all`'s losses, carrying `target`, `site`, `time` and `signed_error_mw`.
        frame: The rows the jobs were fitted on.

    Returns:
        The losses with `actual_mw` and `prediction_mw`.
    """
    parts = [
        fitted.filter(pl.col("target") == target)
        .join(
            frame.select("site", "time", actual_mw=pl.col(target).cast(pl.Float64)),
            on=["site", "time"],
            how="left",
        )
        .with_columns(prediction_mw=pl.col("actual_mw") + pl.col("signed_error_mw"))
        for target in fitted["target"].unique().sort().to_list()
    ]
    return pl.concat(parts)


def in_stable_order(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Sort the losses by arm, setting, target, site, time and seed.

    The fits finish in an order that differs from run to run, so the saved per-row file is sorted
    to make two runs comparable bit for bit.

    Args:
        losses: Fitted losses, in completion order.

    Returns:
        The same rows in a fixed order.
    """
    return losses.sort("arm", "setting", "target", "site", "time", "seed")


def hardware_stamp(*, device: DeviceType) -> dict[str, str]:
    """Record the device, the XGBoost version, and the machine's load before a fit.

    Args:
        device: The device the fits will use.

    Returns:
        The stamp. A GPU is named from `nvidia-smi`.

    Raises:
        RuntimeError: If `device` is `cuda` and `nvidia-smi` reports no GPU.
    """
    stamp = {
        "device": device,
        "xgboost": xgboost.__version__,
        "machine": platform.node(),
        "uptime": subprocess.run(
            ["uptime"], capture_output=True, text=True, check=False
        ).stdout.strip(),
    }
    if device == "cuda":
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if not gpu:
            msg = "device is cuda, but nvidia-smi reports no GPU"
            raise RuntimeError(msg)
        stamp["gpu"] = gpu
    return stamp


def fit_domain(
    *,
    domain: DomainType,
    frame: pl.DataFrame,
    directory: Path,
    device: DeviceType,
) -> pl.DataFrame:
    """Fit every job of one domain, refusing to overwrite, and save the losses.

    Args:
        domain: The domain.
        frame: The domain's rows.
        directory: The output folder.
        device: XGBoost's device.

    Returns:
        The losses, with the measured power and the prediction.
    """
    jobs = domain_jobs(domain=domain)
    losses_path, fingerprint_path = _paths(directory=directory, stem=f"losses_{domain}")
    refuse_to_overwrite(paths=[losses_path, fingerprint_path])
    fitted = run_all(
        dataset=frame, jobs=jobs, max_workers=workers_for(device=device), device=device
    )
    losses = in_stable_order(losses=with_actual_and_prediction(fitted=fitted, frame=frame))
    losses.write_parquet(losses_path)
    fingerprint_path.write_text(_fingerprint(frame=frame, job_list=jobs))
    return losses


def fit_cpu_refit(*, frame: pl.DataFrame, directory: Path) -> pl.DataFrame:
    """Refit one wind arm on the CPU, and save the losses.

    Args:
        frame: The wind rows.
        directory: The output folder.

    Returns:
        The losses, with the measured power and the prediction.
    """
    jobs = cpu_refit_jobs()
    losses_path, fingerprint_path = _paths(directory=directory, stem="losses_cpu_refit")
    refuse_to_overwrite(paths=[losses_path, fingerprint_path])
    fitted = run_all(dataset=frame, jobs=jobs, max_workers=workers_for(device="cpu"), device="cpu")
    losses = in_stable_order(losses=with_actual_and_prediction(fitted=fitted, frame=frame))
    losses.write_parquet(losses_path)
    fingerprint_path.write_text(_fingerprint(frame=frame, job_list=jobs))
    return losses


def _paths(*, directory: Path, stem: str) -> tuple[Path, Path]:
    return directory / f"{stem}.parquet", directory / f"{stem}.fingerprint"


def load_losses(
    *, stem: str, frame: pl.DataFrame, jobs: Sequence[Job], directory: Path
) -> pl.DataFrame:
    """Load saved losses after checking that they were fitted on these rows and jobs.

    Args:
        stem: The losses' file stem, such as `losses_wind`.
        frame: The rows as the build now produces them.
        jobs: The jobs the losses were fitted with.
        directory: The output folder.

    Returns:
        The losses.

    Raises:
        ValueError: If the saved fingerprint differs from the one the rows and jobs now give, or a
            file is missing.
    """
    losses_path, fingerprint_path = _paths(directory=directory, stem=stem)
    expected = _fingerprint(frame=frame, job_list=list(jobs))
    saved = fingerprint_path.read_text().strip() if fingerprint_path.exists() else None
    if saved != expected or not losses_path.exists():
        msg = (
            f"{losses_path}: the saved losses were fitted on a different row set, column set, seed "
            "set, feature values or hyperparameter setting than this code now produces, or a file "
            "is missing; re-run without --report-only"
        )
        raise ValueError(msg)
    return pl.read_parquet(losses_path)


# --- Intervals ------------------------------------------------------------------------------------


def scopes_of(*, losses: pl.DataFrame) -> list[tuple[str, str, pl.Expr | None]]:
    """List the scopes a contrast is split into, as (kind, label, filter).

    Args:
        losses: Losses carrying `time`.

    Returns:
        The whole row set, each calendar year, the two half-years, and the early and late windows.
    """
    year = pl.col("time").dt.year()
    winter = pl.col("time").dt.month().is_in([10, 11, 12, 1, 2, 3])
    scopes: list[tuple[str, str, pl.Expr | None]] = [("all", "all", None)]
    scopes += [
        ("year", f"year {value}", year == value)
        for value in sorted(losses["time"].dt.year().unique().to_list())
    ]
    scopes += [
        ("half-year", "October to March", winter),
        ("half-year", "April to September", ~winter),
        ("window", "early window", pl.col("month") < EARLY_END_MONTH),
        ("window", "late window", pl.col("month") >= EARLY_END_MONTH),
    ]
    return scopes


def post_hoc_scopes_of() -> list[tuple[str, str, pl.Expr]]:
    """List the post hoc scopes of the wind contrasts, as (kind, label, filter).

    Returns:
        Each single lead, each UTC hour, the published wind page's window, and each UKV era. A
        lead is the UTC hour modulo 6, so it is confounded with the hour of day.
    """
    hour = pl.col("time").dt.hour()
    first_late, first_upgraded = ERA_FIRST_MONTHS
    scopes: list[tuple[str, str, pl.Expr]] = [
        ("lead", f"lead {lead} h", hour % LEAD_CYCLE_HOURS == lead)
        for lead in range(LEAD_CYCLE_HOURS)
    ]
    scopes += [("hour", f"UTC hour {h:02d}", hour == h) for h in range(24)]
    scopes.append(("lead group", "leads 0 to 1", hour % LEAD_CYCLE_HOURS <= 1))
    scopes.append(
        (
            "published window",
            f"from {PUBLISHED_WIND_WINDOW_START:%Y-%m-%d}",
            pl.col("time") >= PUBLISHED_WIND_WINDOW_START,
        )
    )
    scopes += [
        ("era", f"era 0 (before {first_late})", pl.col("month") < first_late),
        (
            "era",
            f"era 1 ({first_late} to {first_upgraded}, exclusive)",
            (pl.col("month") >= first_late) & (pl.col("month") < first_upgraded),
        ),
        ("era", f"era 2 ({first_upgraded} onward)", pl.col("month") >= first_upgraded),
    ]
    return scopes


def contrast_record(
    *,
    losses: pl.DataFrame,
    domain: DomainType,
    setting: str,
    label: str,
    planned: bool,
    kind: str,
    scope: str,
    treatment: str,
    reference: str,
    product_contrast: bool = True,
    post_hoc: bool = False,
) -> IntervalRecord:
    """Interval one paired contrast on the given losses.

    Args:
        losses: Losses at one setting holding both arms, restricted to the scope.
        domain: The domain, which sets the margin.
        setting: The setting the losses were fitted at.
        label: The contrast's label, such as `P3` or `control`.
        planned: Whether the contrast was written into the plan before any result.
        kind: `all`, `year`, `half-year`, `window`, or `site`.
        scope: The scope's label.
        treatment: The treatment arm.
        reference: The reference arm.
        product_contrast: Whether the contrast is UKV-CEDA minus ERA5, which is read against the
            domain's margin. A control or a replication is not, so it has no margin and reads
            `significant` or `not significant`.
        post_hoc: Whether the contrast was added after a result was seen.

    Returns:
        The record, in percentage points of capacity. A scope with fewer than
        `MIN_MONTHS_FOR_INTERVAL` months has a null interval and the reading `no_interval`.
    """
    interval = bootstrap_difference(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    margin = MARGINS_PP[domain]
    enough = interval["n_months"] >= MIN_MONTHS_FOR_INTERVAL
    difference, lower, upper, spread = (
        interval[key] * PERCENTAGE_POINTS
        for key in ("difference", "lower_95", "upper_95", "seed_spread")
    )
    reading: str
    if not enough:
        reading = "no_interval"
    elif product_contrast:
        reading = contrast_reading(difference=difference, lower=lower, upper=upper, margin=margin)
    else:
        reading = "significant" if lower > 0.0 or upper < 0.0 else "not significant"
    return {
        "domain": domain,
        "setting": setting,
        "label": label,
        "planned": planned,
        "kind": kind,
        "scope": scope,
        "treatment": treatment,
        "reference": reference,
        "treatment_mae_pp": _mean_pp(losses=losses, arm=treatment),
        "reference_mae_pp": _mean_pp(losses=losses, arm=reference),
        "difference_pp": difference,
        "lower_95_pp": lower if enough else float("nan"),
        "upper_95_pp": upper if enough else float("nan"),
        "margin_pp": margin if product_contrast else float("nan"),
        "reading": reading,
        "n_rows": interval["n_rows"],
        "n_months": interval["n_months"],
        "enough_months": enough,
        "seed_spread_pp": spread,
        "product_contrast": product_contrast,
        "post_hoc": post_hoc,
    }


def trend_record(
    *,
    losses: pl.DataFrame,
    domain: DomainType,
    setting: str,
    planned: Planned,
    scope: str = "slope per year",
    block_months: int = 1,
) -> IntervalRecord:
    """Record the slope per year of a contrast's monthly mean difference, post hoc.

    Args:
        losses: Losses at one setting holding both arms.
        domain: The domain.
        setting: The setting the losses were fitted at.
        planned: The contrast.
        scope: The record's scope.
        block_months: The run length of the month resampling.

    Returns:
        A record whose differences are the slope per year in points of capacity, with no margin. It
        covers month-to-month weather only, and a physics change inside the record is a step that
        a line smooths.
    """
    differences, months = paired_differences(
        losses=losses, treatment=planned.treatment, reference=planned.reference, metric=METRIC
    )
    slope, lower, upper = (
        value * PERCENTAGE_POINTS
        for value in monthly_trend(
            values=differences.mean(axis=0), months=months, block_months=block_months
        )
    )
    return {
        "domain": domain,
        "setting": setting,
        "label": f"{planned.label} trend per year",
        "planned": False,
        "kind": "trend",
        "scope": scope,
        "treatment": planned.treatment,
        "reference": planned.reference,
        "treatment_mae_pp": float("nan"),
        "reference_mae_pp": float("nan"),
        "difference_pp": slope,
        "lower_95_pp": lower,
        "upper_95_pp": upper,
        "margin_pp": float("nan"),
        "reading": "significant" if lower > 0.0 or upper < 0.0 else "not significant",
        "n_rows": int(differences.shape[1]),
        "n_months": len(set(months.tolist())),
        "enough_months": True,
        "seed_spread_pp": float("nan"),
        "product_contrast": False,
        "post_hoc": True,
    }


def _mean_pp(*, losses: pl.DataFrame, arm: str) -> float:
    """Return one arm's mean loss in percentage points of capacity."""
    mean = losses.filter(pl.col("arm") == arm)[METRIC].mean()
    return float(mean) * PERCENTAGE_POINTS  # ty: ignore[invalid-argument-type]


def domain_records(*, domain: DomainType, losses: pl.DataFrame) -> list[IntervalRecord]:
    """Interval every contrast of one domain.

    Args:
        domain: The domain.
        losses: The domain's losses at every setting.

    Returns:
        The split contrasts at both settings in every scope and by generator, the post hoc lead,
        window, and era splits of the wind contrasts, and the controls and checks at the primary
        setting. Only the planned contrasts' whole-row-set rows and P3's early and late windows are
        planned.
    """
    records: list[IntervalRecord] = []
    for planned in (p for p in SPLIT_CONTRASTS if p.domain == domain):
        for setting in (PRIMARY_SETTING, SECOND_SETTING):
            at_setting = losses.filter(pl.col("setting") == setting)
            pair = at_setting.filter(pl.col("arm").is_in([planned.treatment, planned.reference]))
            if pair.is_empty():
                continue
            scopes = [(k, sc, cond, False) for k, sc, cond in scopes_of(losses=pair)]
            if domain in POST_HOC_DOMAINS:
                scopes += [(k, sc, cond, True) for k, sc, cond in post_hoc_scopes_of()]
            for kind, scope, condition, post_hoc in scopes:
                subset = pair if condition is None else pair.filter(condition)
                if subset.is_empty():
                    continue
                records.append(
                    contrast_record(
                        losses=subset,
                        domain=domain,
                        setting=setting,
                        label=planned.label,
                        planned=planned.registered
                        and (kind == "all" or (planned.label == "P3" and kind == "window")),
                        kind=kind,
                        scope=scope,
                        treatment=planned.treatment,
                        reference=planned.reference,
                        post_hoc=post_hoc or not planned.registered,
                    )
                )
            if planned.registered:
                records += [
                    trend_record(losses=pair, domain=domain, setting=setting, planned=planned),
                    *(
                        trend_record(
                            losses=pair,
                            domain=domain,
                            setting=setting,
                            planned=planned,
                            scope=f"slope per year, {block}-month blocks",
                            block_months=block,
                        )
                        for block in TREND_BLOCKS
                    ),
                    trend_record(
                        losses=pair.filter(pl.col("month") < ERA_FIRST_MONTHS[1]),
                        domain=domain,
                        setting=setting,
                        planned=planned,
                        scope="slope per year, without era 2",
                    ),
                ]
            if domain in POST_HOC_DOMAINS:
                records += [
                    contrast_record(
                        losses=pair.filter(pl.col("site") != site),
                        domain=domain,
                        setting=setting,
                        label=planned.label,
                        planned=False,
                        kind="leave one out",
                        scope=f"without generator {site}",
                        treatment=planned.treatment,
                        reference=planned.reference,
                        post_hoc=True,
                    )
                    for site in sorted(pair["site"].unique().to_list())
                    if pair["site"].n_unique() > 1
                ]
            records += [
                contrast_record(
                    losses=pair.filter(pl.col("site") == site),
                    domain=domain,
                    setting=setting,
                    label=f"{planned.label} by generator",
                    planned=False,
                    kind="site",
                    scope=f"generator {site}",
                    treatment=planned.treatment,
                    reference=planned.reference,
                    post_hoc=not planned.registered,
                )
                for site in sorted(pair["site"].unique().to_list())
            ]
    primary = losses.filter(pl.col("setting") == PRIMARY_SETTING)
    for label, treatment, reference in EXPLORATORY_CONTRASTS.get(domain, ()):
        pair = primary.filter(pl.col("arm").is_in([treatment, reference]))
        if pair["arm"].n_unique() < 2:
            continue
        wanted = ("all", "early window") if domain == "wind_keep_zero" else ("all",)
        for kind, scope, condition in scopes_of(losses=pair):
            if scope not in wanted:
                continue
            subset = pair if condition is None else pair.filter(condition)
            records.append(
                contrast_record(
                    losses=subset,
                    domain=domain,
                    setting=PRIMARY_SETTING,
                    label=label,
                    planned=False,
                    kind=kind,
                    scope=scope,
                    treatment=treatment,
                    reference=reference,
                    product_contrast=False,
                )
            )
    return records


# --- The decision rule ----------------------------------------------------------------------------


class Contrast(NamedTuple):
    """One contrast, UKV-CEDA minus ERA5, with the margin it is read against.

    Attributes:
        difference: The point estimate.
        lower: The 95% interval's lower bound.
        upper: The 95% interval's upper bound.
        margin: The margin, in the contrast's own unit.
    """

    difference: float
    lower: float
    upper: float
    margin: float

    @property
    def reading(self) -> ContrastReadingType:
        """How the contrast reads against its margin."""
        return contrast_reading(
            difference=self.difference, lower=self.lower, upper=self.upper, margin=self.margin
        )


ProductType = Literal["ukv_ceda", "era5"]
EarlyType = Literal["from 2019", "from 2021 only", "not applicable"]


class Decision(NamedTuple):
    """The study's recommendation for one variable.

    Attributes:
        variable: `wind` or `temperature`.
        product: The recommended product.
        early_years: Whether a UKV-CEDA recommendation covers training history from 2019.
        reasons: One line per fact the recommendation rests on.
    """

    variable: str
    product: ProductType
    early_years: EarlyType
    reasons: tuple[str, ...]


def early_years_test(*, early: Contrast) -> EarlyType:
    """Say whether UKV-CEDA is recommended for training history from 2019.

    The deciding contrast's early-window point estimate must be negative, and its upper 95% bound
    must be below the margin, which is zero plus the margin.

    Args:
        early: The deciding contrast on the early window.

    Returns:
        `from 2019` if both hold, else `from 2021 only`.
    """
    if early.difference < 0.0 and early.upper < early.margin:
        return "from 2019"
    return "from 2021 only"


def _describe(*, name: str, contrast: Contrast, unit: str) -> str:
    return (
        f"{name}: {contrast.difference:+.3f} {unit} "
        f"[{contrast.lower:+.3f}, {contrast.upper:+.3f}], margin {contrast.margin:.3f}, "
        f"reads {contrast.reading}."
    )


def decide_wind(
    *, primary: Contrast, second: Contrast, early: Contrast, context: Contrast
) -> Decision:
    """Apply the wind rule: P3 decides, and it must agree at both hyperparameter settings.

    Args:
        primary: P3 at the primary setting.
        second: P3 at the second setting.
        early: P3 on the early window at the primary setting.
        context: P1, which is printed beside P3 and never decides.

    Returns:
        UKV-CEDA only if P3 clearly favours UKV-CEDA at both settings, and ERA5 otherwise, which
        covers ERA5 clearly better, no clear difference, a difference smaller than the margin, and
        two settings that disagree.
    """
    reasons = [
        _describe(name="P3, primary setting", contrast=primary, unit="points of capacity"),
        _describe(name="P3, second setting", contrast=second, unit="points of capacity"),
        _describe(name="P3, early window", contrast=early, unit="points of capacity"),
        _describe(name="P1 (set A, context only)", contrast=context, unit="m/s"),
    ]
    if primary.reading != context.reading:
        reasons.append(f"P1 reads {context.reading} and P3 reads {primary.reading}; P3 decides.")
    if primary.reading == second.reading == "ukv_clearly_better":
        return Decision("wind", "ukv_ceda", early_years_test(early=early), tuple(reasons))
    reasons.append("ERA5 is the default unless UKV-CEDA clears the margin at both settings.")
    return Decision("wind", "era5", "not applicable", tuple(reasons))


def decide_temperature(
    *,
    p2: Contrast,
    p2_late_leads: Contrast,
    p4_primary: Contrast,
    p4_second: Contrast,
    early: Contrast,
) -> Decision:
    """Apply the temperature rule: P2 decides, P2-lead is a second condition, P4 can veto.

    Args:
        p2: P2, set A's temperature contrast on the primary score.
        p2_late_leads: P2 on leads 3 to 5 hours only.
        p4_primary: P4 at the primary setting.
        p4_second: P4 at the second setting.
        early: P2 on the early window.

    Returns:
        UKV-CEDA only if P2 clearly favours UKV-CEDA overall and at leads 3 to 5, and neither P4
        setting clearly favours ERA5. ERA5 otherwise.
    """
    reasons = [
        _describe(name="P2", contrast=p2, unit="K"),
        _describe(name="P2-lead, leads 3 to 5", contrast=p2_late_leads, unit="K"),
        _describe(name="P4, primary setting", contrast=p4_primary, unit="points of capacity"),
        _describe(name="P4, second setting", contrast=p4_second, unit="points of capacity"),
        _describe(name="P2, early window", contrast=early, unit="K"),
    ]
    vetoed = "era5_clearly_better" in (p4_primary.reading, p4_second.reading)
    if p2.reading == p2_late_leads.reading == "ukv_clearly_better" and not vetoed:
        return Decision("temperature", "ukv_ceda", early_years_test(early=early), tuple(reasons))
    if p2.reading == "ukv_clearly_better" and p2_late_leads.reading != "ukv_clearly_better":
        reasons.append(
            "P2 favours UKV-CEDA overall but not at leads 3 to 5, so it does not qualify."
        )
    if vetoed:
        reasons.append("P4 clearly favours ERA5 at a setting, which vetoes UKV-CEDA temperature.")
    reasons.append("ERA5 is the default unless UKV-CEDA clears the margin.")
    return Decision("temperature", "era5", "not applicable", tuple(reasons))


def _contrast_from(*, records: Sequence[Mapping[str, Any]], **where: object) -> Contrast:
    """Pick the one record matching every field of `where`, and read it as a `Contrast`.

    Args:
        records: Interval records, from set B (`*_pp` fields) or set A.
        **where: Field values the record must hold.

    Returns:
        The contrast.

    Raises:
        ValueError: If no record or more than one record matches.
    """
    found = [r for r in records if all(r.get(key) == value for key, value in where.items())]
    if len(found) != 1:
        msg = f"{len(found)} records match {where}, not exactly one"
        raise ValueError(msg)
    record = found[0]
    if "difference_pp" in record:
        keys = ("difference_pp", "lower_95_pp", "upper_95_pp", "margin_pp")
    else:
        keys = ("difference", "lower_95", "upper_95", "margin")
    difference, lower, upper, margin = (float(record[key]) for key in keys)
    return Contrast(difference, lower, upper, margin)


def decisions(
    *, set_b: Sequence[Mapping[str, Any]], set_a: Sequence[Mapping[str, Any]]
) -> tuple[Decision, Decision]:
    """Apply the rule to the interval tables of both sets.

    Args:
        set_b: Set B's interval records.
        set_a: Set A's interval records.

    Returns:
        The wind decision and the temperature decision.
    """

    def p3(setting: str, scope: str) -> Contrast:
        return _contrast_from(
            records=set_b, label="P3", setting=setting, scope=scope, domain="wind"
        )

    def p4(setting: str) -> Contrast:
        return _contrast_from(
            records=set_b, label="P4", setting=setting, scope="all", domain="solar"
        )

    def station(**where: object) -> Contrast:
        return _contrast_from(records=set_a, score=PRIMARY_SCORE, **where)

    wind = decide_wind(
        primary=p3(PRIMARY_SETTING, "all"),
        second=p3(SECOND_SETTING, "all"),
        early=p3(PRIMARY_SETTING, "early window"),
        context=station(variable="wind", label="P1", scope="all"),
    )
    temperature = decide_temperature(
        p2=station(variable="temperature", label="P2", scope="all"),
        p2_late_leads=station(variable="temperature", label="P2-lead", scope="leads 3 to 5"),
        p4_primary=p4(PRIMARY_SETTING),
        p4_second=p4(SECOND_SETTING),
        early=station(variable="temperature", label="P2", scope="early window"),
    )
    return wind, temperature


def decision_text(*, wind: Decision, temperature: Decision) -> str:
    """Render the decision rule's result as markdown.

    Args:
        wind: The wind decision.
        temperature: The temperature decision.

    Returns:
        The text of `decision.md`.
    """
    names = {"ukv_ceda": "UKV-CEDA", "era5": "ERA5"}
    lines = ["### Decision, by the rule fixed before any result", ""]
    for decision in (wind, temperature):
        lines += [f"#### {decision.variable.capitalize()}: {names[decision.product]}", ""]
        lines += [f"- {reason}" for reason in decision.reasons]
        if decision.product == "ukv_ceda":
            history = f"- Training history: UKV-CEDA {decision.early_years}."
            if decision.early_years != "from 2019":
                history += " An era-mixed design is untested."
            lines += [history, f"- {LICENCE_CONDITION}"]
        lines.append("")
    return "\n".join(lines)


# --- The report -----------------------------------------------------------------------------------


def _status(*, record: IntervalRecord) -> str:
    if record["planned"]:
        return "planned"
    return "post hoc" if record["post_hoc"] else "exploratory"


def _record_line(*, record: IntervalRecord) -> str:
    interval = (
        f"[{record['lower_95_pp']:+.3f}, {record['upper_95_pp']:+.3f}]"
        if record["enough_months"]
        else "too few months"
    )
    margin = f"{record['margin_pp']:.2f}" if record["product_contrast"] else "none"
    return (
        f"| {_status(record=record)} | {record['label']} | {record['setting']} | {record['scope']} "
        f"| {record['treatment']} − {record['reference']} | {record['difference_pp']:+.3f} "
        f"| {interval} | {margin} | {record['reading']} | {record['n_rows']:,} "
        f"| {record['n_months']} |"
    )


RECORD_HEADER: Final[tuple[str, str]] = (
    (
        "| Status | Label | Setting | Scope | Contrast | Difference (pp of capacity) "
        "| 95% interval, months and seed | Margin | Reading | Rows | Months |"
    ),
    "|---|---|---|---|---|---|---|---|---|---|---|",
)


def absolute_lines(*, domain: DomainType, losses: pl.DataFrame) -> list[str]:
    """Render every arm's absolute error and 95% interval.

    Args:
        domain: The domain.
        losses: The domain's losses.

    Returns:
        Markdown lines.
    """
    lines = [
        f"#### {domain}: every arm's mean absolute error",
        "",
        "| Arm | Setting | Mean absolute error (% of capacity) | 95% interval | Rows |",
        "|---|---|---|---|---|",
    ]
    for setting in (PRIMARY_SETTING, SECOND_SETTING):
        at_setting = losses.filter(pl.col("setting") == setting)
        for arm in sorted(at_setting["arm"].unique().to_list()):
            interval = bootstrap_absolute(losses=at_setting, arm=arm, metric=METRIC)
            lower, upper = (interval[k] * PERCENTAGE_POINTS for k in ("lower_95", "upper_95"))
            lines.append(
                f"| {arm} | {setting} | {interval['value'] * PERCENTAGE_POINTS:.3f} "
                f"| [{lower:.3f}, {upper:.3f}] | {interval['n_rows']:,} |"
            )
    return lines


CONTROL_LABELS: Final[tuple[str, ...]] = (
    "control",
    "era5 against its shuffled arm",
    "UKV-CEDA against its shuffled arm",
)
"""The contrasts that are controls, which a result near the 5% line sends to the second setting."""

NEAR_LINE_SHARE: Final[float] = 0.2
"""A result is near the 5% line when a bound of its interval is within this share of the interval's
width from zero."""


def near_the_line(*, record: IntervalRecord) -> bool:
    """Say whether a result lies near the 5% line.

    Args:
        record: One interval record.

    Returns:
        True when the record has an interval and one bound lies within `NEAR_LINE_SHARE` of the
        interval's width from zero, on either side.
    """
    if not record["enough_months"]:
        return False
    lower, upper = record["lower_95_pp"], record["upper_95_pp"]
    return min(abs(lower), abs(upper)) <= NEAR_LINE_SHARE * (upper - lower)


def near_line_lines(*, records: Sequence[IntervalRecord]) -> list[str]:
    """List every control near the 5% line, which must be rerun at the second setting.

    Args:
        records: Every interval record of set B.

    Returns:
        Markdown lines.
    """
    flagged = [r for r in records if r["label"] in CONTROL_LABELS and near_the_line(record=r)]
    lines = ["#### Controls near the 5% line", ""]
    if not flagged:
        return [
            *lines,
            "- None: no control has a bound within 20% of its interval's width of zero.",
        ]
    return [
        *lines,
        (
            "- Each control below has a bound within 20% of its interval's width of zero, so the "
            "study skill asks for a rerun at the second hyperparameter setting. Nothing reruns it "
            "here."
        ),
        *(
            f"- {r['domain']}, {r['label']}: {r['treatment']} minus {r['reference']}, "
            f"{r['difference_pp']:+.3f} [{r['lower_95_pp']:+.3f}, {r['upper_95_pp']:+.3f}]."
            for r in flagged
        ),
    ]


def records_lines(*, records: Sequence[IntervalRecord], title: str) -> list[str]:
    """Render records as one markdown table.

    Args:
        records: The records.
        title: The heading.

    Returns:
        Markdown lines.
    """
    return [f"#### {title}", "", *RECORD_HEADER, *(_record_line(record=r) for r in records)]


def era_lines(*, records: Sequence[IntervalRecord], era2_months: Sequence[str] = ()) -> list[str]:
    """Report P3 in the third UKV era, which is the closest to today's UKV, at both settings.

    Args:
        records: Every interval record of set B.
        era2_months: The scored months of the third era, as `%Y-%m`.

    Returns:
        Markdown lines. The sign and the significance at each setting are stated, so a result that
        is significant at one setting only is not read as settled, and the calendar make-up of the
        era is counted.
    """
    lines = ["#### The third era (post hoc)", ""]
    for setting in (PRIMARY_SETTING, SECOND_SETTING):
        found = [
            r
            for r in records
            if r["label"] == "P3"
            and r["kind"] == "era"
            and r["scope"].startswith("era 2")
            and r["setting"] == setting
        ]
        if len(found) != 1:
            continue
        record = found[0]
        favours = "UKV-CEDA" if record["difference_pp"] < 0.0 else "ERA5"
        significant = record["enough_months"] and (
            record["lower_95_pp"] > 0.0 or record["upper_95_pp"] < 0.0
        )
        lines.append(
            f"- {setting} setting, {record['scope']}: P3 {record['difference_pp']:+.3f} "
            f"[{record['lower_95_pp']:+.3f}, {record['upper_95_pp']:+.3f}] points of capacity over "
            f"{record['n_months']} months, favouring {favours}, "
            f"{'statistically significant' if significant else 'not statistically significant'} "
            "at the 5% level."
        )
    if era2_months:
        summer = [m for m in era2_months if 4 <= int(m[5:7]) <= 9]
        lines.append(
            f"- Of the era's {len(era2_months)} scored months ({min(era2_months)} to "
            f"{max(era2_months)}), {len(summer)} fall in April to September."
        )
    lines.append(
        "- The era is exploratory, its folds trained mostly on the second era, and set A holds no "
        "station data from it, so P1 and P2 say nothing about it."
    )
    return lines


def hour_lines(*, records: Sequence[IntervalRecord]) -> list[str]:
    """Show P3 at each run boundary and ERA5's own error by lead and by UTC hour, post hoc.

    Each lead pools four UTC hours spread across the day, so a diurnal cycle can masquerade as a
    lead only if it repeats every 6 hours. The gap between adjacent hours at a run boundary is the
    sharper test.

    Args:
        records: Every interval record of set B.

    Returns:
        Markdown lines.
    """
    p3 = [
        r
        for r in records
        if r["label"] == "P3" and r["setting"] == PRIMARY_SETTING and r["domain"] == "wind"
    ]
    by_hour = {int(r["scope"][-2:]): r for r in p3 if r["kind"] == "hour"}
    by_lead = [r for r in p3 if r["kind"] == "lead"]
    if len(by_hour) != 24 or not by_lead:
        return []
    boundaries = [
        f"{before:02d} UTC {by_hour[before]['difference_pp']:+.3f} to {after:02d} UTC "
        f"{by_hour[after]['difference_pp']:+.3f}"
        for before, after in ((5, 6), (11, 12), (17, 18), (23, 0))
    ]
    era5_by_hour = [r["reference_mae_pp"] for r in by_hour.values()]
    era5_by_lead = [r["reference_mae_pp"] for r in by_lead]
    return [
        "#### P3 by UTC hour and ERA5's own error (post hoc)",
        "",
        f"- P3 across the four run boundaries, in points of capacity: {'; '.join(boundaries)}.",
        (
            f"- ERA5's own mean absolute error ranges from {min(era5_by_hour):.2f}% to "
            f"{max(era5_by_hour):.2f}% across the 24 UTC hours and from "
            f"{min(era5_by_lead):.2f}% to {max(era5_by_lead):.2f}% across the six leads."
        ),
    ]


def farm_lines(*, records: Sequence[IntervalRecord]) -> list[str]:
    """Print P3 leaving each wind farm out, post hoc.

    Args:
        records: Every interval record of set B.

    Returns:
        Markdown lines.
    """
    found = sorted(
        (
            r
            for r in records
            if r["label"] == "P3"
            and r["kind"] == "leave one out"
            and r["domain"] == "wind"
            and r["setting"] == PRIMARY_SETTING
        ),
        key=lambda r: r["scope"],
    )
    return [
        "#### P3 leaving one wind farm out (post hoc)",
        "",
        *(
            f"- {r['scope']}: {r['difference_pp']:+.3f} [{r['lower_95_pp']:+.3f}, "
            f"{r['upper_95_pp']:+.3f}] points of capacity."
            for r in found
        ),
    ]


def veto_lines(*, records: Sequence[IntervalRecord], n_farms: int) -> list[str]:
    """Show that the P4 veto could not have fired, and what P4's bound says instead.

    Args:
        records: Every interval record of set B.
        n_farms: The number of solar farms.

    Returns:
        Markdown lines.
    """

    def pick(**where: object) -> IntervalRecord:
        found = [r for r in records if all(r[k] == v for k, v in where.items())]  # ty: ignore[invalid-key]
        if len(found) != 1:
            msg = f"{len(found)} records match {where}"
            raise ValueError(msg)
        return found[0]

    era5 = pick(domain="solar", label="era5 against its shuffled arm", scope="all")
    ukv = pick(domain="solar", label="UKV-CEDA against its shuffled arm", scope="all")
    bounds = [
        pick(domain="solar", label="P4", scope="all", setting=setting)
        for setting in (PRIMARY_SETTING, SECOND_SETTING)
    ]
    margin = MARGINS_PP["solar"]
    era5_value, ukv_value = -era5["difference_pp"], -ukv["difference_pp"]
    return [
        "#### The P4 veto could not have fired",
        "",
        (
            f"- Temperature as a whole is worth {era5_value:.3f} points of capacity to an XGBoost "
            f"model given ERA5's (its error against its own shuffled arm) and {ukv_value:.3f} to "
            f"one given UKV-CEDA's. A clear ERA5 verdict from P4 needs a difference above "
            f"{margin:.2f} points, which is {margin / max(era5_value, 1e-9):.1f} times the whole "
            "value of ERA5's temperature."
        ),
        (
            f"- P4's information is its bound: on these {n_farms} solar farms the choice of "
            "temperature product moves the error by no more than "
            f"{max(abs(b[k]) for b in bounds for k in ('lower_95_pp', 'upper_95_pp')):.3f} points "
            "at either setting."
        ),
    ]


def station_lead_lines(*, set_a: Sequence[Mapping[str, Any]]) -> list[str]:
    """Print set A by single lead and by leaving one station out, both post hoc.

    Args:
        set_a: Set A's interval records.

    Returns:
        Markdown lines.
    """
    lines = [
        "#### Set A by UKV-CEDA lead (post hoc)",
        "",
        (
            "Each row is UKV-CEDA minus ERA5 on the primary score, with each product's error "
            "against the station. Lead equals the UTC hour modulo 6, so a lead is also an hour of "
            "day."
        ),
        "",
        "| Variable | Lead (h) | ERA5 MAE | UKV-CEDA MAE | UKV-CEDA minus ERA5 | 95% interval |",
        "|---|---|---|---|---|---|",
    ]
    for variable in ("wind", "temperature"):
        lines += [
            f"| {variable} | {record['scope'].removeprefix('lead ').removesuffix(' h')} "
            f"| {record['era5_mae']:.3f} | {record['ukv_mae']:.3f} | {record['difference']:+.3f} "
            f"| [{record['lower_95']:+.3f}, {record['upper_95']:+.3f}] |"
            for record in set_a
            if record["variable"] == variable and record["label"] == "post hoc lead"
        ]
    lines += ["", "#### Set A leaving one station out (post hoc)", ""]
    for variable in ("wind", "temperature"):
        rows = [
            r for r in set_a if r["variable"] == variable and r["label"] == "post hoc leave one out"
        ]
        values = [r["difference"] for r in rows]
        if values:
            lines.append(
                f"- {variable}: UKV-CEDA minus ERA5 between {min(values):+.3f} and "
                f"{max(values):+.3f} over the {len(values)} leave-one-out sets, so no single "
                "station decides the sign."
            )
    return lines


def costs_lines(
    *, build_stamp: Mapping[str, Any], frames: Mapping[DomainType, pl.DataFrame]
) -> list[str]:
    """List what a UKV-CEDA training history costs, beside the temperature recommendation.

    Args:
        build_stamp: `build.json`.
        frames: Each domain's rows.

    Returns:
        Markdown lines.
    """
    dropped = build_stamp["dropped_months"]
    months_per_era = dict(
        sorted(frames["wind"].group_by("era_code").agg(m=pl.col("month").n_unique()).iter_rows())
    )
    return [
        "#### What a UKV-CEDA training history costs",
        "",
        *run_status_lines(ukv=open_ukv_stores()),
        "",
        (
            f"- {len(dropped)} months lose more than {MAX_MONTH_LOSS_SHARE:.0%} of their hours to "
            "incomplete runs and are dropped from every arm: "
            + ", ".join(f"{m} ({share:.0%})" for m, share in sorted(dropped.items()))
            + f". Two more months, {', '.join(STRADDLING_MONTHS)}, straddle a physics change."
        ),
        (
            "- The main work would have to fill the dropped hours from another source, a "
            "mixed-source history that this study did not test."
        ),
        (
            "- UKV-CEDA carries a 6-hourly lead sawtooth: the error rises with lead and drops at "
            "each run boundary (the set A lead table), which a feature built over several hours, "
            "such as a lag or a rolling mean, inherits."
        ),
        (
            f"- A history from 2019 spans three physics eras, with {months_per_era} scored months "
            "each. The XGBoost models here were given `era_code`, and the main work would need it "
            "too."
        ),
    ]


def deviation_lines(*, n_dropped_months: int) -> list[str]:
    """List the plan's departures, for the report.

    Args:
        n_dropped_months: The number of months the 25% guard dropped, from `build.json`.

    Returns:
        One line per departure.
    """
    return [
        (
            f"The 25% monthly-loss guard failed for {n_dropped_months} months, which are dropped "
            "from every arm and every set. The maintainer's delegate decided this after the "
            "implementer had seen the set A tables in memory."
        ),
        "The plan's re-check of the unlisted days and the partial runs against CEDA was not done.",
        "A keep-zero-hours block (all months, not only the early window) was added.",
        "The wind row set is the intersection of the centred and the hour-ending power targets.",
        (
            "The search for the Met Office's PS44 was not repeated. The repository's roadmap "
            "records that none was found."
        ),
        (
            "Post hoc rows were added after the first results: each single UKV-CEDA lead and "
            "leave-one-station-out in set A; the lead, published-window, and era splits of P3; "
            "and a matched-height pair of wind arms (each product's 10 m wind alone)."
        ),
    ]


def report_text(
    *,
    records: Sequence[IntervalRecord],
    losses: Mapping[DomainType, pl.DataFrame],
    frames: Mapping[DomainType, pl.DataFrame],
    stamp: Mapping[str, str],
    set_a: Sequence[Mapping[str, Any]],
    build_stamp: Mapping[str, Any],
) -> str:
    """Render every table the page quotes.

    Args:
        records: Every interval record of set B.
        losses: Each domain's losses.
        frames: Each domain's rows.
        stamp: The hardware stamp of the fits.
        set_a: Set A's interval records.
        build_stamp: `build.json`.

    Returns:
        The text of `report.md`.
    """
    lines = ["### Set B: XGBoost models, out of fold", ""]
    lines += [f"- {key}: {value}." for key, value in stamp.items()]
    lines += [
        (
            "- Every score is the mean absolute error of the capped prediction, each row divided "
            "by its own generator's capacity. A negative difference favours UKV-CEDA. The "
            "intervals resample whole calendar months and one of three fitting seeds."
        ),
        (
            "- Planned: set B's P3 and P4 on the whole row set and P3's early and late windows, "
            "and set A's P1, P2 and P2-lead and P2's early and late windows. Every other row is "
            "exploratory, or post hoc where it was added after a result was seen. A margin applies "
            "only to a contrast of UKV-CEDA against ERA5. A control or a replication has no margin "
            "and reads `significant` or `not significant`."
        ),
        "",
    ]
    for domain, frame in frames.items():
        per_site = sorted(frame.group_by("site").len().iter_rows())
        lines += [
            f"#### {domain}: rows",
            "",
            f"- {frame.height:,} rows on {len(per_site)} generators, {frame['month'].n_unique()} "
            f"months, {int(frame['constrained'].sum()):,} constrained (excluded from training, "
            "still scored); rows per generator: "
            + ", ".join(f"{site} {count:,}" for site, count in per_site)
            + ".",
            "",
        ]
    job_list = [job for domain in frames for job in domain_jobs(domain=domain)]
    lines += [*_arm_columns_lines(job_list=job_list), ""]
    for domain, domain_losses in losses.items():
        lines += [*absolute_lines(domain=domain, losses=domain_losses), ""]
    product = [r for r in records if r["product_contrast"]]
    planned = [r for r in product if r["planned"]]
    lines += [*records_lines(records=planned, title="Planned contrasts"), ""]
    whole = [r for r in product if not r["planned"] and r["kind"] == "all"]
    lines += [
        *records_lines(records=whole, title="Post hoc contrasts on the whole row set"),
        "",
    ]
    for kind, title in (
        ("window", "Early and late windows"),
        ("year", "By calendar year"),
        ("half-year", "By half-year"),
        ("site", "By generator"),
        ("lead", "By UKV-CEDA lead (post hoc; a lead is also an hour of day)"),
        ("lead group", "By the first two leads together (post hoc)"),
        ("published window", "From the published wind page's first day (post hoc)"),
        ("era", "By UKV era (post hoc)"),
    ):
        chosen = [r for r in product if r["kind"] == kind]
        if chosen:
            lines += [*records_lines(records=chosen, title=title), ""]
    analysis_only = [
        r for r in product if r["domain"] in RESTRICTED_DOMAINS and r["kind"] in ("all", "window")
    ]
    if analysis_only:
        lines += [
            *records_lines(
                records=analysis_only,
                title="Analysis-only refits: trained and scored on the first leads only (post hoc)",
            ),
            "",
        ]
    trends = [r for r in records if r["kind"] == "trend"]
    lines += [
        *records_lines(records=trends, title="Slope per year of the monthly difference (post hoc)"),
        "",
    ]
    controls = [r for r in records if not r["product_contrast"] and r["kind"] != "trend"]
    lines += [*records_lines(records=controls, title="Controls and replications"), ""]
    lines += [*near_line_lines(records=records), ""]
    lines += [
        *era_lines(
            records=records,
            era2_months=sorted(
                m for m in frames["wind"]["month"].unique().to_list() if m >= ERA_FIRST_MONTHS[1]
            ),
        ),
        "",
        *hour_lines(records=records),
        "",
        *farm_lines(records=records),
        "",
    ]
    lines += [*veto_lines(records=records, n_farms=frames["solar"]["site"].n_unique()), ""]
    lines += [*station_lead_lines(set_a=set_a), ""]
    lines += [*costs_lines(build_stamp=build_stamp, frames=frames), ""]
    splits = [
        r
        for r in product
        if not r["planned"] and r["setting"] == PRIMARY_SETTING and r["kind"] != "all"
    ]
    significant = [
        r
        for r in splits
        if r["enough_months"] and (r["lower_95_pp"] > 0.0 or r["upper_95_pp"] < 0.0)
    ]
    n_matched = sum(r["domain"] == "wind_matched" for r in significant)
    n_hour = sum(r["domain"] != "wind_matched" and r["kind"] == "hour" for r in significant)
    controls_primary = [r for r in controls if r["setting"] == PRIMARY_SETTING]
    significant_controls = [r for r in controls_primary if r["reading"] == "significant"]
    lines += [
        "#### How many exploratory splits reach statistical significance",
        "",
        (
            f"- {len(significant)} of {len(splits)} exploratory and post hoc splits of the "
            "contrasts of UKV-CEDA against ERA5 (by year, half-year, generator, lead, UTC hour, "
            "window, era, and with one generator or station left out) at the primary setting are "
            "statistically significant at the 5% level. A row with no real effect behind it has a "
            "nominal 5% chance of reaching that level, the number of rows with no real effect is "
            "unknown, and the rows share their months, so spurious results cluster. The page does "
            "not correct for multiple comparisons."
        ),
        (
            f"- Of those, {n_matched} come from the matched 10 m pair, a further {n_hour} are "
            "splits of the other contrasts by UTC hour, and the remaining "
            f"{len(significant) - n_matched - n_hour} are other splits, so the statistically "
            "significant splits are not that many independent findings."
        ),
        (
            f"- Separately, {len(significant_controls)} of {len(controls_primary)} controls and "
            "replications are statistically significant. An arm against its own shuffled arm is "
            "expected to be, so these rows are not evidence about the products."
        ),
        "",
        "#### Departures from the plan",
        "",
        *(
            f"- {line}"
            for line in deviation_lines(n_dropped_months=len(build_stamp["dropped_months"]))
        ),
    ]
    return "\n".join(lines) + "\n"


def read_station_records(*, directory: Path) -> list[dict[str, Any]]:
    """Read set A's interval table.

    Args:
        directory: The output folder.

    Returns:
        The records.
    """
    return pl.read_parquet(directory / STATION_INTERVALS_NAME).to_dicts()


# --- Command line ---------------------------------------------------------------------------------


def dry_run_lines(*, frames: Mapping[DomainType, pl.DataFrame]) -> list[str]:
    """List every fit, count the fit-sets, and check the arm widths.

    Args:
        frames: Each domain's rows.

    Returns:
        Text lines.
    """
    check_arm_widths()
    lines = ["--dry-run: nothing is fitted or written."]
    total = 0
    for domain, frame in frames.items():
        n_sites = frame["site"].n_unique()
        jobs = domain_jobs(domain=domain)
        total += len(jobs) * n_sites
        lines += [f"{domain}: {len(jobs)} jobs on {n_sites} generators"]
        lines += [
            f"- {arm} / {setting} / target {target} / {len(columns)} columns"
            for arm, setting, target, columns, _, _ in jobs
        ]
    cpu = cpu_refit_jobs()
    cpu_fit_sets = len(cpu) * frames["wind"]["site"].n_unique()
    lines += [
        f"CPU refit: {CPU_REFIT_ARM}, {cpu_fit_sets} fit-sets.",
        (
            f"Fit-sets: {total} on the GPU, plus {cpu_fit_sets} on the CPU. "
            "Each fits 5 folds at 3 seeds."
        ),
    ]
    return lines


def load_saved(
    *, frames: dict[DomainType, pl.DataFrame], directory: Path
) -> tuple[dict[DomainType, pl.DataFrame], dict[str, str]]:
    """Load every domain's saved losses and the hardware stamp, for `--report-only`.

    A domain whose rows exist but whose losses do not (the matched-height pair before it is
    fitted) is left out of `frames`.

    Args:
        frames: Each domain's rows, which this function may remove an entry from.
        directory: The output folder.

    Returns:
        Each domain's losses, with the CPU refit added to wind, and the hardware stamp.
    """
    for domain in OPTIONAL_DOMAINS:
        if domain in frames and not (directory / f"losses_{domain}.parquet").exists():
            del frames[domain]
    losses = {
        domain: load_losses(
            stem=f"losses_{domain}",
            frame=frame,
            jobs=domain_jobs(domain=domain),
            directory=directory,
        )
        for domain, frame in frames.items()
    }
    cpu = load_losses(
        stem="losses_cpu_refit", frame=frames["wind"], jobs=cpu_refit_jobs(), directory=directory
    )
    losses["wind"] = pl.concat([losses["wind"], cpu])
    return losses, json.loads((directory / STAMP_NAME).read_text())


def restricted_frame(*, frame: pl.DataFrame, n_leads: int) -> pl.DataFrame:
    """Keep the rows whose lead in UKV-CEDA's 6-hourly runs is below `n_leads`.

    The lead is the hour of day modulo 6. A solar hour's temperature averages the instants at both
    ends of the hour, so a solar row at lead 0 also reads the previous run's lead 5.

    Args:
        frame: A domain's rows, carrying `hour_of_day`.
        n_leads: How many leads to keep, from lead 0.

    Returns:
        The rows at leads 0 to `n_leads - 1`.
    """
    return frame.filter(pl.col("hour_of_day") % LEAD_CYCLE_HOURS < n_leads)


def load_frames(*, directory: Path) -> dict[DomainType, pl.DataFrame]:
    """Read every domain's rows, leaving out the matched-height rows until they exist.

    Args:
        directory: The output folder.

    Returns:
        Each domain's rows.
    """
    frames = {
        domain: pl.read_parquet(directory / name)
        for domain, name in ROW_NAMES.items()
        if domain not in OPTIONAL_DOMAINS or (directory / name).exists()
    }
    for domain, (base, n_leads) in RESTRICTED_DOMAINS.items():
        frames[domain] = restricted_frame(frame=frames[base], n_leads=n_leads)
    return frames


def run_extra_fits(
    *, arguments: argparse.Namespace, frames: dict[DomainType, pl.DataFrame], directory: Path
) -> bool:
    """Fit the post hoc domains that a flag asks for, and say whether one was fitted.

    Args:
        arguments: The parsed command line.
        frames: Each domain's rows.
        directory: The output folder.

    Returns:
        True if a post hoc fit ran, in which case no report is written.
    """
    if arguments.fit_lead_restricted:
        stamp = hardware_stamp(device=arguments.device)
        refuse_to_overwrite(paths=[directory / LEAD_STAMP_NAME])
        (directory / LEAD_STAMP_NAME).write_text(json.dumps(stamp, indent=2))
        for restricted in RESTRICTED_DOMAINS:
            fit_domain(
                domain=restricted,
                frame=frames[restricted],
                directory=directory,
                device=arguments.device,
            )
        sys.stdout.write(f"Fitted the analysis-only refits on {arguments.device}.\n")
        return True
    if arguments.fit_matched or arguments.fit_hour_starting:
        domain: DomainType = "wind_matched" if arguments.fit_matched else "wind_hour_starting"
        stamp_name = MATCHED_STAMP_NAME if arguments.fit_matched else HOUR_STARTING_STAMP_NAME
        stamp = hardware_stamp(device=arguments.device)
        refuse_to_overwrite(paths=[directory / stamp_name])
        (directory / stamp_name).write_text(json.dumps(stamp, indent=2))
        fit_domain(
            domain=domain, frame=frames[domain], directory=directory, device=arguments.device
        )
        sys.stdout.write(f"Fitted {domain} on {arguments.device}.\n")
        return True

    return False


def main() -> int:
    """Fit every arm, or list the fits, or rebuild the reports from saved losses."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="List the fits; fit nothing.")
    mode.add_argument("--report-only", action="store_true", help="Rebuild the reports.")
    mode.add_argument(
        "--fit-lead-restricted",
        action="store_true",
        help="Fit only the post hoc analysis-only refits, and write no report.",
    )
    mode.add_argument(
        "--fit-hour-starting",
        action="store_true",
        help="Fit only the post hoc power-hour scan, and write no report.",
    )
    mode.add_argument(
        "--fit-matched",
        action="store_true",
        help="Fit only the post hoc matched-height pair, and write no report.",
    )
    parser.add_argument(
        "--verified", action="store_true", help="State that verify.md was run and read."
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    directory: Path = arguments.output_dir
    frames = load_frames(directory=directory)

    if arguments.dry_run:
        sys.stdout.write("\n".join(dry_run_lines(frames=frames)) + "\n")
        return 0
    if not (directory / STATION_INTERVALS_NAME).exists():
        msg = f"run ukv_ceda_station_scores.py first: {STATION_INTERVALS_NAME} is missing"
        raise SystemExit(msg)

    if run_extra_fits(arguments=arguments, frames=frames, directory=directory):
        return 0

    if arguments.report_only:
        losses, stamp = load_saved(frames=frames, directory=directory)
    else:
        if not arguments.verified or not (directory / VERIFY_NAME).exists():
            msg = (
                "run ukv_ceda_vs_era5_verify.py until it passes (a failed run writes "
                "verify_failed.md and no verify.md), read verify.md, then pass --verified"
            )
            raise SystemExit(msg)
        stamp = hardware_stamp(device=arguments.device)
        refuse_to_overwrite(paths=[directory / STAMP_NAME])
        (directory / STAMP_NAME).write_text(json.dumps(stamp, indent=2))
        for optional in OPTIONAL_DOMAINS:
            frames.pop(optional, None)
        losses = {
            domain: fit_domain(
                domain=domain, frame=frame, directory=directory, device=arguments.device
            )
            for domain, frame in frames.items()
        }
        cpu = fit_cpu_refit(frame=frames["wind"], directory=directory)
        losses["wind"] = pl.concat([losses["wind"], cpu])

    paths = [directory / name for name in (INTERVALS_NAME, REPORT_NAME, DECISION_NAME)]
    refuse_to_overwrite(paths=paths)
    records = [
        r
        for domain, domain_losses in losses.items()
        for r in domain_records(domain=domain, losses=domain_losses)
    ]
    set_a = read_station_records(directory=directory)
    wind_decision, temperature_decision = decisions(set_b=records, set_a=set_a)
    report = report_text(
        records=records,
        losses=losses,
        frames=frames,
        stamp=stamp,
        set_a=set_a,
        build_stamp=json.loads((directory / "build.json").read_text()),
    )
    decision = decision_text(wind=wind_decision, temperature=temperature_decision)
    pl.DataFrame(records).write_parquet(paths[0])
    paths[1].write_text(report)
    paths[2].write_text(decision)
    sys.stdout.write(report + "\n" + decision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
