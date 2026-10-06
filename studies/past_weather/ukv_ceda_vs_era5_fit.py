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
)
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_HYPER_PARAMETERS,
    DeviceType,
)
from studies.guards import refuse_to_overwrite
from ukv_ceda_station_scores import INTERVALS_NAME as STATION_INTERVALS_NAME
from ukv_ceda_station_scores import PRIMARY_SCORE
from ukv_ceda_vs_era5_build import (
    EARLY_END_MONTH,
    MARGIN_SOLAR_PP,
    MARGIN_WIND_PP,
    OUTPUT_DIR,
    SOLAR_ROWS_NAME,
    WIND_KEEP_ZERO_ROWS_NAME,
    WIND_ROWS_NAME,
    ContrastReadingType,
    check_arm_widths,
    contrast_reading,
    solar_arm_columns,
    wind_arm_columns,
)

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss every table reports: each row's clamped error over its own generator's capacity."""

PERCENTAGE_POINTS: Final[float] = 100.0

PRIMARY_SETTING: Final[str] = "pooled"
SECOND_SETTING: Final[str] = "sensitivity"

DomainType = Literal["wind", "solar", "wind_keep_zero"]

VERIFY_NAME: Final[str] = "verify.md"
REPORT_NAME: Final[str] = "report.md"
DECISION_NAME: Final[str] = "decision.md"
INTERVALS_NAME: Final[str] = "intervals.parquet"
STAMP_NAME: Final[str] = "fit_stamp.json"
CPU_REFIT_ARM: Final[str] = "era5_wind_cpu_refit"
"""The arm refitted on the CPU for the noise floor: ERA5's wind arm."""

ROW_NAMES: Final[Mapping[DomainType, str]] = {
    "wind": WIND_ROWS_NAME,
    "solar": SOLAR_ROWS_NAME,
    "wind_keep_zero": WIND_KEEP_ZERO_ROWS_NAME,
}
"""The build's row file of each domain."""

MARGINS_PP: Final[Mapping[DomainType, float]] = {
    "wind": MARGIN_WIND_PP,
    "solar": MARGIN_SOLAR_PP,
    "wind_keep_zero": MARGIN_WIND_PP,
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


PLANNED_CONTRASTS: Final[tuple[Planned, ...]] = (
    Planned("P3", "wind", "ukv_ceda_wind", "era5_wind"),
    Planned("P4", "solar", "solar_ukv_ceda_temp", "solar_era5_temp"),
)
"""The planned contrasts of set B, UKV-CEDA minus ERA5, each at both settings."""

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


# --- Jobs -----------------------------------------------------------------------------------------


def _job(*, arm: str, setting: str, columns: tuple[str, ...], target: str = "power_mw") -> Job:
    hyper_parameters = (
        PRIMARY_HYPER_PARAMETERS if setting == PRIMARY_SETTING else SENSITIVITY_HYPER_PARAMETERS
    )
    return (arm, setting, target, columns, hyper_parameters, False)


def domain_jobs(*, domain: DomainType) -> list[Job]:
    """List every fit of one domain.

    Args:
        domain: `wind`, `solar`, or `wind_keep_zero`.

    Returns:
        The jobs: the planned arms at both settings, then the controls and checks at the primary
        setting. Every job fits one arm at each generator of the domain.
    """
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


def fit_set_count(*, jobs: Sequence[Job], n_sites: int) -> int:
    """Count the fit-sets: one per job and generator, each fitting five folds at three seeds.

    Args:
        jobs: The jobs.
        n_sites: The number of generators.

    Returns:
        The count.
    """
    return len(jobs) * n_sites


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
    losses = with_actual_and_prediction(fitted=fitted, frame=frame)
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
    losses = with_actual_and_prediction(fitted=fitted, frame=frame)
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
    reading: str = (
        contrast_reading(difference=difference, lower=lower, upper=upper, margin=margin)
        if enough
        else "no_interval"
    )
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
        "margin_pp": margin,
        "reading": reading,
        "n_rows": interval["n_rows"],
        "n_months": interval["n_months"],
        "enough_months": enough,
        "seed_spread_pp": spread,
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
        The planned contrasts at both settings in every scope, the planned contrasts by generator,
        and the exploratory contrasts at the primary setting.
    """
    records: list[IntervalRecord] = []
    for planned in (p for p in PLANNED_CONTRASTS if p.domain == domain):
        for setting in (PRIMARY_SETTING, SECOND_SETTING):
            at_setting = losses.filter(pl.col("setting") == setting)
            pair = at_setting.filter(pl.col("arm").is_in([planned.treatment, planned.reference]))
            for kind, scope, condition in scopes_of(losses=pair):
                subset = pair if condition is None else pair.filter(condition)
                records.append(
                    contrast_record(
                        losses=subset,
                        domain=domain,
                        setting=setting,
                        label=planned.label,
                        planned=kind == "all" or (planned.label == "P3" and kind == "window"),
                        kind=kind,
                        scope=scope,
                        treatment=planned.treatment,
                        reference=planned.reference,
                    )
                )
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
                )
                for site in sorted(pair["site"].unique().to_list())
            ]
    primary = losses.filter(pl.col("setting") == PRIMARY_SETTING)
    for label, treatment, reference in EXPLORATORY_CONTRASTS[domain]:
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


def _record_line(*, record: IntervalRecord) -> str:
    interval = (
        f"[{record['lower_95_pp']:+.3f}, {record['upper_95_pp']:+.3f}]"
        if record["enough_months"]
        else "too few months"
    )
    return (
        f"| {record['label']} | {record['setting']} | {record['scope']} "
        f"| {record['treatment']} − {record['reference']} | {record['difference_pp']:+.3f} "
        f"| {interval} | {record['margin_pp']:.2f} | {record['reading']} | {record['n_rows']:,} "
        f"| {record['n_months']} |"
    )


RECORD_HEADER: Final[tuple[str, str]] = (
    (
        "| Label | Setting | Scope | Contrast | Difference (pp of capacity) | 95% interval, months "
        "and seed | Margin | Reading | Rows | Months |"
    ),
    "|---|---|---|---|---|---|---|---|---|---|",
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


def records_lines(*, records: Sequence[IntervalRecord], title: str) -> list[str]:
    """Render records as one markdown table.

    Args:
        records: The records.
        title: The heading.

    Returns:
        Markdown lines.
    """
    return [f"#### {title}", "", *RECORD_HEADER, *(_record_line(record=r) for r in records)]


def report_text(
    *,
    records: Sequence[IntervalRecord],
    losses: Mapping[DomainType, pl.DataFrame],
    frames: Mapping[DomainType, pl.DataFrame],
    stamp: Mapping[str, str],
) -> str:
    """Render every table the page quotes.

    Args:
        records: Every interval record of set B.
        losses: Each domain's losses.
        frames: Each domain's rows.
        stamp: The hardware stamp of the fits.

    Returns:
        The text of `report.md`.
    """
    lines = ["### Set B: XGBoost models, out of fold", ""]
    lines += [f"- {key}: {value}." for key, value in stamp.items()]
    lines += [
        (
            "- Every score is the mean absolute error of the capped prediction, each row divided "
            "by its own generator's capacity. A negative difference favours UKV-CEDA. The "
            "intervals resample whole calendar months and one of three fitting seeds. Rows "
            "labelled P3 and P4 are planned, and every other row is exploratory."
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
    planned = [r for r in records if r["planned"] and r["kind"] == "all"]
    lines += [*records_lines(records=planned, title="Planned contrasts"), ""]
    for kind, title in (
        ("year", "Planned contrasts by calendar year (exploratory splits)"),
        ("half-year", "Planned contrasts by half-year (exploratory splits)"),
        ("window", "Planned contrasts, early and late windows"),
        ("site", "Planned contrasts by generator (exploratory)"),
    ):
        chosen = [r for r in records if r["kind"] == kind]
        lines += [*records_lines(records=chosen, title=title), ""]
    exploratory = [r for r in records if not r["planned"] and r["kind"] != "site"]
    lines += [*records_lines(records=exploratory, title="Controls and checks (exploratory)"), ""]
    primary = [r for r in records if not r["planned"] and r["setting"] == PRIMARY_SETTING]
    excluding = [
        r
        for r in primary
        if r["enough_months"] and (r["lower_95_pp"] > 0.0 or r["upper_95_pp"] < 0.0)
    ]
    lines += [
        "#### How many exploratory rows reach statistical significance",
        "",
        (
            f"- {len(excluding)} of {len(primary)} exploratory rows at the primary setting are "
            "statistically significant at the 5% level. A row with no real effect behind it has "
            "a nominal 5% chance of reaching that level, the number of rows with no real effect "
            "is unknown, and the rows share their months, so spurious results cluster. The page "
            "does not correct for multiple comparisons."
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
        total += fit_set_count(jobs=jobs, n_sites=n_sites)
        lines += [f"{domain}: {len(jobs)} jobs on {n_sites} generators"]
        lines += [
            f"- {arm} / {setting} / target {target} / {len(columns)} columns"
            for arm, setting, target, columns, _, _ in jobs
        ]
    cpu = cpu_refit_jobs()
    n_wind = frames["wind"]["site"].n_unique()
    lines += [
        f"CPU refit: {CPU_REFIT_ARM}, {fit_set_count(jobs=cpu, n_sites=n_wind)} fit-sets.",
        (
            f"Fit-sets: {total} on the GPU, plus {fit_set_count(jobs=cpu, n_sites=n_wind)} on the "
            "CPU. Each fits 5 folds at 3 seeds."
        ),
    ]
    return lines


def main() -> int:
    """Fit every arm, or list the fits, or rebuild the reports from saved losses."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="List the fits; fit nothing.")
    mode.add_argument("--report-only", action="store_true", help="Rebuild the reports.")
    parser.add_argument(
        "--verified", action="store_true", help="State that verify.md was run and read."
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    directory: Path = arguments.output_dir
    frames: dict[DomainType, pl.DataFrame] = {
        domain: pl.read_parquet(directory / name) for domain, name in ROW_NAMES.items()
    }

    if arguments.dry_run:
        sys.stdout.write("\n".join(dry_run_lines(frames=frames)) + "\n")
        return 0

    if arguments.report_only:
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
            stem="losses_cpu_refit",
            frame=frames["wind"],
            jobs=cpu_refit_jobs(),
            directory=directory,
        )
        losses["wind"] = pl.concat([losses["wind"], cpu])
        stamp = json.loads((directory / STAMP_NAME).read_text())
    else:
        if not arguments.verified or not (directory / VERIFY_NAME).exists():
            msg = "run ukv_ceda_vs_era5_verify.py, read verify.md, then pass --verified"
            raise SystemExit(msg)
        stamp = hardware_stamp(device=arguments.device)
        refuse_to_overwrite(paths=[directory / STAMP_NAME])
        (directory / STAMP_NAME).write_text(json.dumps(stamp, indent=2))
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
    wind_decision, temperature_decision = decisions(
        set_b=records, set_a=read_station_records(directory=directory)
    )
    report = report_text(records=records, losses=losses, frames=frames, stamp=stamp)
    decision = decision_text(wind=wind_decision, temperature=temperature_decision)
    pl.DataFrame(records).write_parquet(paths[0])
    paths[1].write_text(report)
    paths[2].write_text(decision)
    sys.stdout.write(report + "\n" + decision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
