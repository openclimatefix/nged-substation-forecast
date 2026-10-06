"""Fit one XGBoost model per generator and archive, and score it on both archives of UKV.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1051>. It reads the rows that
`ukv_ceda_vs_openmeteo_build.py` wrote, fits an XGBoost model per metered generator and per arm,
scores each model's power error out of fold, applies the decision rule by code, and writes
`report.md` and `decision.md`.

**Arms.** Each wind arm has 6 columns and each solar arm 8, built by one function per arm type
(`ukv_ceda_vs_openmeteo_build.wind_arm_columns` and `solar_arm_columns`) and printed into the
report. `ceda_wind_10m` and `om_wind_10m` read each archive's 10 m speed and the sine and cosine of
its 10 m direction. `ceda_ghi_temp` and `om_ghi_temp` read each archive's global irradiance and
temperature, built as the build script describes.

**Planned contrasts (CEDA minus Open-Meteo, a negative value favouring CEDA), each fitted at both
hyperparameter settings.** P1 is the wind arms, P2 the solar arms, and P3 the transfer penalty: the
CEDA-trained model scored on Open-Meteo's values, minus the Open-Meteo-trained model scored on
Open-Meteo's values, for wind and for solar. The margins are those of the CEDA-against-ERA5 study,
frozen in `ukv_ceda_vs_openmeteo_build` before any result: 0.16 points of capacity for wind and
0.06 for solar. **The solar contrasts P2 and P3 are read on era 0 only** (before the PS47 upgrade
of 2026-01-21), because Open-Meteo's hourly irradiance is built differently afterwards and the two
archives' irradiance columns are then not like for like. The solar rows of era 1 and every scope
that includes them are exploratory and carry a note. Wind and temperature keep the whole overlap.
P1 and P2 read `interchangeable`, `differ`, or `unresolved`, and P3 is read
one-sided as `no_penalty`, `penalty`, or `unresolved`. A verdict stands only if both settings
agree. An unresolved P3 leads to "do not mix the two archives".

**One fitted model scores every frame.** The CEDA-trained model of each (generator, fold, seed)
predicts CEDA's own values, Open-Meteo's values under CEDA's column names, and a few partial swaps
(`studies.cross_validation.out_of_fold_losses` with `scoring_site_rows`), so the transfer penalty
needs no second fit and `ceda_*` scored on CEDA is exactly the arm P1 reads. The partial swaps and
the temperature-offset read are exploratory.

**Controls, exploratory, at the primary setting.** The negative control shuffles each archive's
weather columns within a generator, year-month, and hour of day, each archive under its own
permutation. One wind arm is refitted on the CPU, to give the difference between a GPU and a CPU
fit. No positive control runs, so a null contrast is reported with its bound.

**Folds.** `cerra_past_solar.with_covering_folds` cuts folds of whole months inside the two UKV
eras (before and after the 2026-01-21 PS47 upgrade), `era_code` is a feature, and `colsample_bytree`
stays at 1. The folds are fixed when the rows are built.

**Intervals.** Each interval resamples whole calendar months, paired across arms, and one of the
three fitting seeds, 2,000 times (`studies.bootstrap.bootstrap_difference`). A scope with fewer than
six months gets a point estimate and no interval. Every score is divided by its row's own
generator's capacity, and an export-cap `constrained` solar hour is excluded from training and still
scored. The lead-0 reads filter the same fits' per-row losses to the hours at which CEDA's lead is 0
and refit nothing.

Run it with `uv run python studies/past_weather/ukv_ceda_vs_openmeteo_fit.py`. `--dry-run` lists
every fit and fits nothing. A fit needs `--verified`, which says the build's `--check-only` passed
and `direct_report.md` was read. `--report-only` rebuilds the intervals and the reports from the
saved losses after checking each fingerprint. A fresh run stops (`refuse_to_overwrite`) while an
output exists. Only one agent may run it at a time, because every worktree shares one data folder.
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal, NamedTuple, TypedDict

import numpy as np
import polars as pl
from ens_past_solar import _arm_columns_lines, _fingerprint
from studies.arm_runner import Job, run_all
from studies.bootstrap import MIN_MONTHS_FOR_INTERVAL, bootstrap_absolute, bootstrap_difference
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_HYPER_PARAMETERS,
    DeviceType,
    out_of_fold_losses,
)
from studies.guards import refuse_to_overwrite
from ukv_ceda_vs_era5_fit import (
    METRIC,
    PERCENTAGE_POINTS,
    hardware_stamp,
    in_stable_order,
    with_actual_and_prediction,
    workers_for,
)
from ukv_ceda_vs_openmeteo_build import (
    LEAD_CYCLE_HOURS,
    LOW_SUN_ZENITH_DEG,
    MARGIN_SOLAR_PP,
    MARGIN_WIND_PP,
    OUTPUT_DIR,
    SOLAR_ERA_0_ROWS_NAME,
    SOLAR_ROWS_NAME,
    WIND_ROWS_NAME,
    check_arm_widths,
    solar_arm_columns,
    solar_columns,
    transfer_frame,
    wind_arm_columns,
    wind_columns,
)
from ukv_ceda_vs_openmeteo_build import (
    STAMP_NAME as BUILD_STAMP_NAME,
)
from ukv_ceda_vs_openmeteo_compare import REPORT_NAME as DIRECT_REPORT_NAME

_LOG: Final[logging.Logger] = logging.getLogger("ukv_ceda_vs_openmeteo_fit")

DomainType = Literal["wind", "solar", "solar_era0"]
"""The fitted row sets. `solar_era0` is the solar rows of era 0 alone, with its own folds."""

BaseDomainType = Literal["wind", "solar"]


def base_domain(*, domain: DomainType) -> BaseDomainType:
    """Return the technology of a row set: `solar_era0` is solar.

    Args:
        domain: A fitted row set.

    Returns:
        `wind` or `solar`.
    """
    return "wind" if domain == "wind" else "solar"


ROW_NAMES: Final[Mapping[DomainType, str]] = {
    "wind": WIND_ROWS_NAME,
    "solar": SOLAR_ROWS_NAME,
    "solar_era0": SOLAR_ERA_0_ROWS_NAME,
}
MARGINS_PP: Final[Mapping[DomainType, float]] = {
    "wind": MARGIN_WIND_PP,
    "solar": MARGIN_SOLAR_PP,
    "solar_era0": MARGIN_SOLAR_PP,
}
"""The margin of each domain, in percentage points of capacity."""

PRIMARY_SETTING: Final[str] = "primary"
SECOND_SETTING: Final[str] = "second"
"""The names of the two hyperparameter settings in every table."""

REPORT_NAME: Final[str] = "report.md"
DECISION_NAME: Final[str] = "decision.md"
INTERVALS_NAME: Final[str] = "intervals.parquet"
STAMP_NAME: Final[str] = "fit_stamp.json"
CPU_REFIT_STEM: Final[str] = "losses_cpu_refit"

ERA_BOUNDARY_MONTH: Final[str] = "2026-02"
"""The first month after the PS47 upgrade, which starts era 1."""

LICENCE_CONDITION: Final[str] = (
    "Subject to the licence: CEDA's catalogue record gives the Creative Commons "
    "Attribution-NonCommercial-ShareAlike 4.0 licence, and whether the main work's use of CEDA's "
    "UKV is non-commercial is a decision for the maintainer."
)

CEDA_OWN: Final[str] = "ceda"
OM_VALUES: Final[str] = "om"


# --- Arms and jobs --------------------------------------------------------------------------------


def arm_names(*, domain: DomainType) -> tuple[str, str]:
    """Return the names of the CEDA and Open-Meteo arms of a domain.

    Args:
        domain: `wind` or `solar`.

    Returns:
        The CEDA arm, then the Open-Meteo arm.
    """
    return (
        ("ceda_wind_10m", "om_wind_10m")
        if base_domain(domain=domain) == "wind"
        else ("ceda_ghi_temp", "om_ghi_temp")
    )


def arm_columns(*, domain: DomainType, archive: str, shuffled: bool = False) -> tuple[str, ...]:
    """Return an arm's feature columns.

    Args:
        domain: `wind` or `solar`.
        archive: `ceda` or `om`.
        shuffled: Whether the arm reads the archive's shuffled copy.

    Returns:
        The columns, from the build's one function per arm type.
    """
    function = wind_arm_columns if base_domain(domain=domain) == "wind" else solar_arm_columns
    return function(archive=archive, shuffled=shuffled)


def _job(*, arm: str, setting: str, columns: tuple[str, ...]) -> Job:
    hyper_parameters = (
        PRIMARY_HYPER_PARAMETERS if setting == PRIMARY_SETTING else SENSITIVITY_HYPER_PARAMETERS
    )
    return (arm, setting, "power_mw", columns, hyper_parameters, False)


def ordinary_jobs(*, domain: DomainType) -> list[Job]:
    """List the fits scored on their own archive's values: Open-Meteo's arm and the controls.

    Args:
        domain: `wind` or `solar`.

    Returns:
        Open-Meteo's arm at both settings, then each archive's shuffled arm at the primary setting
        (the era-0 solar sensitivity has no controls).
    """
    ceda_arm, om_arm = arm_names(domain=domain)
    jobs = [
        _job(arm=om_arm, setting=setting, columns=arm_columns(domain=domain, archive="om"))
        for setting in (PRIMARY_SETTING, SECOND_SETTING)
    ]
    if domain == "solar_era0":
        return jobs
    jobs += [
        _job(
            arm=f"{arm}_shuffled",
            setting=PRIMARY_SETTING,
            columns=arm_columns(domain=domain, archive=archive, shuffled=True),
        )
        for arm, archive in ((ceda_arm, "ceda"), (om_arm, "om"))
    ]
    return jobs


def transfer_jobs(*, domain: DomainType) -> list[Job]:
    """List the CEDA-trained fits, each scored on several frames of values.

    Args:
        domain: `wind` or `solar`.

    Returns:
        The CEDA arm at both settings.
    """
    ceda_arm, _ = arm_names(domain=domain)
    return [
        _job(arm=ceda_arm, setting=setting, columns=arm_columns(domain=domain, archive="ceda"))
        for setting in (PRIMARY_SETTING, SECOND_SETTING)
    ]


def cpu_refit_jobs() -> list[Job]:
    """List the one arm refitted on the CPU: Open-Meteo's wind arm at the primary setting.

    Returns:
        One job.
    """
    return [
        _job(
            arm="om_wind_10m_cpu_refit",
            setting=PRIMARY_SETTING,
            columns=arm_columns(domain="wind", archive="om"),
        )
    ]


def scoring_names(*, domain: DomainType) -> tuple[str, ...]:
    """List the frames a CEDA-trained model is scored on.

    Args:
        domain: `wind` or `solar`.

    Returns:
        CEDA's own values, Open-Meteo's values, and the exploratory partial swaps.
    """
    partial = (
        ("om_direction", "om_speed_rescaled")
        if base_domain(domain=domain) == "wind"
        else ("om_temp", "om_ghi", "om_temp_offset_removed")
    )
    return (CEDA_OWN, OM_VALUES, *partial)


def scored_arm_name(*, arm: str, scoring: str) -> str:
    """Name a CEDA-trained arm's losses on one scoring frame.

    Args:
        arm: The CEDA arm.
        scoring: The scoring frame's name.

    Returns:
        The arm itself when scored on CEDA's own values, else `<arm>_scored_on_<scoring>`.
    """
    return arm if scoring == CEDA_OWN else f"{arm}_scored_on_{scoring}"


def scoring_frames(*, site_rows: pl.DataFrame, domain: DomainType) -> dict[str, pl.DataFrame]:
    """Build the frames one site's CEDA-trained model is scored on.

    Args:
        site_rows: One site's rows.
        domain: `wind` or `solar`.

    Returns:
        By name, the frames: CEDA's own values, Open-Meteo's values under CEDA's column names, and
        the partial swaps. Each holds `time`, `fold` and every column of the CEDA arm.
    """
    columns = arm_columns(domain=domain, archive="ceda")
    base = base_domain(domain=domain)
    swapped = transfer_frame(frame=site_rows, domain=base)
    frames = {
        CEDA_OWN: site_rows.select("time", "fold", *columns),
        OM_VALUES: swapped.select("time", "fold", *columns),
    }
    if base == "wind":
        _, ceda_sin, ceda_cos = wind_columns(archive="ceda")
        frames["om_direction"] = site_rows.select("time", "fold", *columns).with_columns(
            swapped[ceda_sin].alias(ceda_sin), swapped[ceda_cos].alias(ceda_cos)
        )
        ceda_speed = wind_columns(archive="ceda")[0]
        frames["om_speed_rescaled"] = swapped.select("time", "fold", *columns).with_columns(
            site_rows["om_speed_10m_rescaled"].alias(ceda_speed)
        )
        return frames
    ceda_ghi, ceda_temp = solar_columns(archive="ceda")
    own = site_rows.select("time", "fold", *columns)
    frames["om_temp"] = own.with_columns(swapped[ceda_temp].alias(ceda_temp))
    frames["om_ghi"] = own.with_columns(swapped[ceda_ghi].alias(ceda_ghi))
    frames["om_temp_offset_removed"] = own.with_columns(
        site_rows["om_temp_offset_removed"].alias(ceda_temp)
    )
    return frames


def _transfer_site(
    *, domain: DomainType, job: Job, site_rows: pl.DataFrame, device: DeviceType
) -> pl.DataFrame:
    arm, setting, target, columns, hyper_parameters, _ = job
    losses = out_of_fold_losses(
        site_rows=site_rows,
        features=columns,
        target=target,
        hyper_parameters=hyper_parameters,
        with_quantiles=False,
        device=device,
        scoring_site_rows=scoring_frames(site_rows=site_rows, domain=domain),
    )
    return losses.with_columns(
        arm=pl.struct("scoring_archive").map_elements(
            lambda row: scored_arm_name(arm=arm, scoring=row["scoring_archive"]),
            return_dtype=pl.String,
        ),
        setting=pl.lit(setting),
        target=pl.lit(target),
    ).drop("scoring_archive")


def run_transfer(
    *, domain: DomainType, frame: pl.DataFrame, device: DeviceType, max_workers: int
) -> pl.DataFrame:
    """Fit the CEDA arm at each site and score every fold's model on every scoring frame.

    Args:
        domain: `wind` or `solar`.
        frame: The domain's rows.
        device: XGBoost's device.
        max_workers: How many (job, site) fits run at once.

    Returns:
        The losses of every scoring frame, with the arm named by `scored_arm_name`.
    """
    sites = sorted(frame["site"].unique().to_list())
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _transfer_site,
                domain=domain,
                job=job,
                site_rows=frame.filter(pl.col("site") == site),
                device=device,
            )
            for job in transfer_jobs(domain=domain)
            for site in sites
        ]
        parts = [future.result() for future in concurrent.futures.as_completed(futures)]
    return pl.concat(parts)


# --- Fitting --------------------------------------------------------------------------------------


def _paths(*, directory: Path, stem: str) -> tuple[Path, Path]:
    return directory / f"{stem}.parquet", directory / f"{stem}.fingerprint"


def all_jobs(*, domain: DomainType) -> list[Job]:
    """List every fit of one domain.

    Args:
        domain: `wind` or `solar`.

    Returns:
        The ordinary fits, then the CEDA-trained fits.
    """
    return [*ordinary_jobs(domain=domain), *transfer_jobs(domain=domain)]


def fit_domain(
    *, domain: DomainType, frame: pl.DataFrame, directory: Path, device: DeviceType
) -> pl.DataFrame:
    """Fit every job of one domain, refusing to overwrite, and save the losses.

    Args:
        domain: `wind` or `solar`.
        frame: The domain's rows.
        directory: The output folder.
        device: XGBoost's device.

    Returns:
        The losses, with the measured power and the prediction.
    """
    jobs = all_jobs(domain=domain)
    losses_path, fingerprint_path = _paths(directory=directory, stem=f"losses_{domain}")
    refuse_to_overwrite(paths=[losses_path, fingerprint_path])
    workers = workers_for(device=device)
    ordinary = run_all(
        dataset=frame, jobs=ordinary_jobs(domain=domain), max_workers=workers, device=device
    )
    transferred = run_transfer(domain=domain, frame=frame, device=device, max_workers=workers)
    fitted = pl.concat([ordinary, transferred.select(ordinary.columns)])
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
    losses_path, fingerprint_path = _paths(directory=directory, stem=CPU_REFIT_STEM)
    refuse_to_overwrite(paths=[losses_path, fingerprint_path])
    fitted = run_all(dataset=frame, jobs=jobs, max_workers=workers_for(device="cpu"), device="cpu")
    losses = in_stable_order(losses=with_actual_and_prediction(fitted=fitted, frame=frame))
    losses.write_parquet(losses_path)
    fingerprint_path.write_text(_fingerprint(frame=frame, job_list=jobs))
    return losses


def load_losses(
    *, stem: str, frame: pl.DataFrame, jobs: Sequence[Job], directory: Path
) -> pl.DataFrame:
    """Load saved losses after checking that they were fitted on these rows and jobs.

    Args:
        stem: The losses' file stem.
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


# --- Readings -------------------------------------------------------------------------------------

TwoSidedReadingType = Literal["interchangeable", "differ", "unresolved"]
PenaltyReadingType = Literal["no_penalty", "penalty", "unresolved"]


def two_sided_reading(
    *, difference: float, lower: float, upper: float, margin: float
) -> TwoSidedReadingType:
    """Read one contrast, CEDA minus Open-Meteo, against its margin.

    Args:
        difference: The point estimate.
        lower: The interval's lower bound.
        upper: The interval's upper bound.
        margin: The margin, a positive number in the contrast's own unit.

    Returns:
        `interchangeable` where the interval lies wholly inside the margin, `differ` where it
        excludes zero and the estimate lies beyond the margin, else `unresolved`.
    """
    if lower > -margin and upper < margin:
        return "interchangeable"
    if (upper < 0.0 or lower > 0.0) and abs(difference) > margin:
        return "differ"
    return "unresolved"


def penalty_reading(
    *, difference: float, lower: float, upper: float, margin: float
) -> PenaltyReadingType:
    """Read one transfer penalty one-sided, because only a penalty is actionable.

    Args:
        difference: The point estimate, the CEDA-trained model's error on Open-Meteo's values minus
            the Open-Meteo-trained model's error on the same values.
        lower: The interval's lower bound.
        upper: The interval's upper bound.
        margin: The margin, a positive number in the contrast's own unit.

    Returns:
        `no_penalty` where the upper bound is below the margin, `penalty` where the lower bound is
        above zero and the estimate exceeds the margin, else `unresolved`.
    """
    if upper < margin:
        return "no_penalty"
    if lower > 0.0 and difference > margin:
        return "penalty"
    return "unresolved"


def combine_settings(*, readings: Sequence[str]) -> str:
    """Combine a contrast's readings at the two settings: a verdict stands only if both agree.

    Args:
        readings: The reading at each setting.

    Returns:
        The shared reading, or `unresolved` where the settings disagree.
    """
    return readings[0] if len(set(readings)) == 1 else "unresolved"


class Planned(NamedTuple):
    """One planned contrast."""

    label: str
    domain: DomainType
    treatment: str
    reference: str
    one_sided: bool = False


PLANNED_CONTRASTS: Final[tuple[Planned, ...]] = (
    Planned("P1", "wind", "ceda_wind_10m", "om_wind_10m"),
    Planned("P2", "solar", "ceda_ghi_temp", "om_ghi_temp"),
    Planned("P3", "wind", "ceda_wind_10m_scored_on_om", "om_wind_10m", one_sided=True),
    Planned("P3", "solar", "ceda_ghi_temp_scored_on_om", "om_ghi_temp", one_sided=True),
    Planned("P2", "solar_era0", "ceda_ghi_temp", "om_ghi_temp"),
    Planned("P3", "solar_era0", "ceda_ghi_temp_scored_on_om", "om_ghi_temp", one_sided=True),
)
"""The planned contrasts, each at both settings.

The solar contrasts are fitted twice. The rows of both eras train and are scored on era 0, and the
rows of era 0 alone train and are scored on era 0, so that the irradiance construction that differs
after PS47 cannot bias the planned read. A solar verdict stands only if the two fits agree.
"""

EXPLORATORY_CONTRASTS: Final[Mapping[DomainType, tuple[tuple[str, str, str], ...]]] = {
    "wind": (
        ("control", "ceda_wind_10m_shuffled", "om_wind_10m_shuffled"),
        ("CEDA against its shuffled arm", "ceda_wind_10m", "ceda_wind_10m_shuffled"),
        ("Open-Meteo against its shuffled arm", "om_wind_10m", "om_wind_10m_shuffled"),
        (
            "CEDA model, Open-Meteo direction",
            "ceda_wind_10m_scored_on_om_direction",
            "ceda_wind_10m",
        ),
        (
            "CEDA model, Open-Meteo speed rescaled (calibrator test)",
            "ceda_wind_10m_scored_on_om_speed_rescaled",
            "om_wind_10m",
        ),
        ("GPU against CPU", "om_wind_10m_cpu_refit", "om_wind_10m"),
    ),
    "solar_era0": (),
    "solar": (
        ("control", "ceda_ghi_temp_shuffled", "om_ghi_temp_shuffled"),
        ("CEDA against its shuffled arm", "ceda_ghi_temp", "ceda_ghi_temp_shuffled"),
        ("Open-Meteo against its shuffled arm", "om_ghi_temp", "om_ghi_temp_shuffled"),
        ("CEDA model, Open-Meteo temperature", "ceda_ghi_temp_scored_on_om_temp", "ceda_ghi_temp"),
        ("CEDA model, Open-Meteo irradiance", "ceda_ghi_temp_scored_on_om_ghi", "ceda_ghi_temp"),
        (
            "CEDA model, Open-Meteo temperature without its lead-0 offset",
            "ceda_ghi_temp_scored_on_om_temp_offset_removed",
            "ceda_ghi_temp",
        ),
    ),
}
"""The exploratory contrasts of each domain, as (label, treatment, reference)."""


class IntervalRecord(TypedDict):
    """One contrast in the intervals table."""

    domain: str
    setting: str
    label: str
    planned: bool
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
    note: str


PLANNED_SCOPE: Final[Mapping[DomainType, str]] = {
    "wind": "all",
    "solar": "era 0",
    "solar_era0": "all",
}
"""The scope each domain's planned contrasts are read on.

Wind and temperature keep the whole overlap. Solar reads era 0 only, because Open-Meteo's hourly
irradiance is built differently after the PS47 upgrade, so the two archives' irradiance columns are
not like for like in era 1. Every other solar scope is exploratory and carries the era-1 note.
"""


LEAD_SCOPE_FORMAT: Final[str] = "lead {lead} only"
ERA_0_LEAD_SCOPE_FORMAT: Final[str] = "era 0, lead {lead} only"
"""The scopes at one CEDA lead, over all rows and over era 0 (the lead is the UTC hour mod 6)."""

ERA_0_SCOPES: Final[tuple[str, ...]] = (
    "era 0",
    "era 0, sun above 5 degrees",
    *(ERA_0_LEAD_SCOPE_FORMAT.format(lead=lead) for lead in range(6)),
)
"""The scopes that hold no era-1 row, and so carry no era-1 note."""


def scopes_of(
    *, losses: pl.DataFrame, domain: DomainType = "wind"
) -> list[tuple[str, pl.Expr | None]]:
    """List the scopes a contrast is split into, as (label, filter).

    Args:
        losses: Losses carrying `time`, `month` and `site`, and for solar `solar_zenith_deg`.
        domain: The row set. The era-0 solar sensitivity is read on all its rows and at each lead.

    Returns:
        All rows, each UKV era, the two half-years, the lead-0 hours (every row at which CEDA's
        lead is 0), and each site. Where the losses carry the solar zenith, two geometry scopes
        follow: the hours with the sun more than 5 degrees above the horizon, in all rows and in
        era 0. The filter reads the geometry and neither archive's value.
    """
    if domain == "solar_era0":
        return [
            ("all", None),
            *(
                (
                    LEAD_SCOPE_FORMAT.format(lead=lead),
                    pl.col("time").dt.hour() % LEAD_CYCLE_HOURS == lead,
                )
                for lead in range(LEAD_CYCLE_HOURS)
            ),
        ]
    winter = pl.col("time").dt.month().is_in([10, 11, 12, 1, 2, 3])
    scopes: list[tuple[str, pl.Expr | None]] = [
        ("all", None),
        ("era 0", pl.col("month") < ERA_BOUNDARY_MONTH),
        ("era 1", pl.col("month") >= ERA_BOUNDARY_MONTH),
        ("October to March", winter),
        ("April to September", ~winter),
        ("without 2025-01", pl.col("month") != "2025-01"),
    ]
    for lead in range(LEAD_CYCLE_HOURS):
        at_lead = pl.col("time").dt.hour() % LEAD_CYCLE_HOURS == lead
        scopes.append((LEAD_SCOPE_FORMAT.format(lead=lead), at_lead))
        in_era_0 = pl.col("month") < ERA_BOUNDARY_MONTH
        scopes.append((ERA_0_LEAD_SCOPE_FORMAT.format(lead=lead), at_lead & in_era_0))
    if "solar_zenith_deg" in losses.columns:
        sunny = pl.col("solar_zenith_deg") < LOW_SUN_ZENITH_DEG
        scopes += [
            ("sun above 5 degrees", sunny),
            ("era 0, sun above 5 degrees", sunny & (pl.col("month") < ERA_BOUNDARY_MONTH)),
        ]
    scopes += [
        (f"site {site}", pl.col("site") == site)
        for site in sorted(losses["site"].unique().to_list())
    ]
    return scopes


def _mean_pp(*, losses: pl.DataFrame, arm: str) -> float:
    rows = losses.filter(pl.col("arm") == arm)
    return float(np.mean(rows[METRIC].to_numpy())) * PERCENTAGE_POINTS


def contrast_record(
    *,
    losses: pl.DataFrame,
    domain: DomainType,
    setting: str,
    label: str,
    planned: bool,
    scope: str,
    treatment: str,
    reference: str,
    one_sided: bool = False,
    read_margin: bool = True,
    note: str = "",
) -> IntervalRecord:
    """Interval one paired contrast on the given losses.

    Args:
        losses: Losses at one setting holding both arms, restricted to the scope.
        domain: The domain, which sets the margin.
        setting: The setting the losses were fitted at.
        label: The contrast's label.
        planned: Whether the plan named the contrast before any result.
        scope: The scope's label.
        treatment: The treatment arm.
        reference: The reference arm.
        one_sided: Whether to read the contrast as a transfer penalty.
        read_margin: Whether to read the contrast against the margin. A control or a replication
            has no margin and reads `significant` or `not significant`.
        note: A caveat the report and the figures print beside the row.

    Returns:
        The record, in percentage points of capacity. A scope with fewer than
        `MIN_MONTHS_FOR_INTERVAL` months has a null interval and the reading `no_interval`.
    """
    interval = bootstrap_difference(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    margin = MARGINS_PP[domain]
    enough = interval["n_months"] >= MIN_MONTHS_FOR_INTERVAL
    difference, lower, upper = (
        interval[key] * PERCENTAGE_POINTS for key in ("difference", "lower_95", "upper_95")
    )
    reading: str
    if not enough:
        reading = "no_interval"
    elif not read_margin:
        reading = "significant" if lower > 0.0 or upper < 0.0 else "not significant"
    elif one_sided:
        reading = penalty_reading(difference=difference, lower=lower, upper=upper, margin=margin)
    else:
        reading = two_sided_reading(difference=difference, lower=lower, upper=upper, margin=margin)
    return {
        "domain": domain,
        "setting": setting,
        "label": label,
        "planned": planned,
        "scope": scope,
        "treatment": treatment,
        "reference": reference,
        "treatment_mae_pp": _mean_pp(losses=losses, arm=treatment),
        "reference_mae_pp": _mean_pp(losses=losses, arm=reference),
        "difference_pp": difference,
        "lower_95_pp": lower if enough else float("nan"),
        "upper_95_pp": upper if enough else float("nan"),
        "margin_pp": margin if read_margin else float("nan"),
        "reading": reading,
        "n_rows": interval["n_rows"],
        "n_months": interval["n_months"],
        "enough_months": enough,
        "note": note,
    }


def domain_records(
    *, domain: DomainType, losses: pl.DataFrame, era_1_note: str = ""
) -> list[IntervalRecord]:
    """Interval every contrast of one domain, in every scope.

    Args:
        domain: `wind` or `solar`.
        losses: The domain's losses, and for wind the CPU refit's.
        era_1_note: The caveat that Open-Meteo's irradiance is built differently after PS47, which
            every solar record whose scope includes era-1 rows carries.

    Returns:
        The planned contrasts at both settings in every scope, labelled planned only on the
        domain's `PLANNED_SCOPE`, then the exploratory contrasts at the primary setting on that
        scope.
    """
    present = set(losses["arm"].unique().to_list())
    records: list[IntervalRecord] = []
    for scope, condition in scopes_of(losses=losses, domain=domain):
        scoped = losses if condition is None else losses.filter(condition)
        if scoped.is_empty():
            continue
        for planned in PLANNED_CONTRASTS:
            if planned.domain != domain:
                continue
            records += [
                contrast_record(
                    losses=scoped.filter(pl.col("setting") == setting),
                    domain=domain,
                    setting=setting,
                    label=planned.label,
                    planned=scope == PLANNED_SCOPE[domain],
                    scope=scope,
                    treatment=planned.treatment,
                    reference=planned.reference,
                    one_sided=planned.one_sided,
                    note=(era_1_note if domain == "solar" and scope not in ERA_0_SCOPES else ""),
                )
                for setting in (PRIMARY_SETTING, SECOND_SETTING)
            ]
        if scope != PLANNED_SCOPE[domain]:
            continue
        primary = scoped.filter(pl.col("setting") == PRIMARY_SETTING)
        records += [
            contrast_record(
                losses=primary,
                domain=domain,
                setting=PRIMARY_SETTING,
                label=label,
                planned=False,
                scope=scope,
                treatment=treatment,
                reference=reference,
                read_margin=False,
            )
            for label, treatment, reference in EXPLORATORY_CONTRASTS[domain]
            if {treatment, reference} <= present
        ]
    return records


# --- The decision ---------------------------------------------------------------------------------


DOMAIN_FITS: Final[Mapping[BaseDomainType, tuple[DomainType, ...]]] = {
    "wind": ("wind",),
    "solar": ("solar", "solar_era0"),
}
"""The row sets whose readings a verdict combines. Solar has two: the rows of both eras, and the
rows of era 0 alone."""


class Verdict(NamedTuple):
    """One planned contrast's verdict across its settings and fits."""

    label: str
    domain: BaseDomainType
    reading: str
    primary: IntervalRecord
    second: IntervalRecord
    sensitivity: tuple[IntervalRecord, ...] = ()
    """The era-0-trained fit's records at both settings, for solar."""


def verdicts(*, records: Sequence[IntervalRecord]) -> list[Verdict]:
    """Read each planned contrast across both settings and every fit, on its planned scope.

    **A verdict stands only if every reading agrees.** Wind has two readings, one per setting.
    Solar has four: the rows of both eras, and the rows of era 0 alone, each at both settings. The
    era-0-trained fit cannot be biased by the irradiance construction that differs after PS47, and
    the all-rows fit trains on era-1 rows built that way, so a disagreement is read as
    `unresolved`.

    Args:
        records: Every interval record.

    Returns:
        One verdict per planned contrast of a technology that has records.
    """
    found: list[Verdict] = []
    domains = {record["domain"] for record in records}
    for planned in PLANNED_CONTRASTS:
        if planned.domain == "solar_era0":
            continue
        if planned.domain not in domains:
            continue
        readings: list[str] = []
        chosen_by_domain: dict[str, dict[str, IntervalRecord]] = {}
        for domain in DOMAIN_FITS[planned.domain]:
            if domain not in domains:
                continue
            chosen = {
                record["setting"]: record
                for record in records
                if record["scope"] == PLANNED_SCOPE[domain]
                and record["label"] == planned.label
                and record["domain"] == domain
                and record["treatment"] == planned.treatment
                and record["planned"]
            }
            chosen_by_domain[domain] = chosen
            readings += [chosen[PRIMARY_SETTING]["reading"], chosen[SECOND_SETTING]["reading"]]
        main = chosen_by_domain[planned.domain]
        sensitivity = tuple(
            record
            for domain, chosen in chosen_by_domain.items()
            if domain != planned.domain
            for record in (chosen[PRIMARY_SETTING], chosen[SECOND_SETTING])
        )
        found.append(
            Verdict(
                label=planned.label,
                domain=planned.domain,
                reading=combine_settings(readings=readings),
                primary=main[PRIMARY_SETTING],
                second=main[SECOND_SETTING],
                sensitivity=sensitivity,
            )
        )
    return found


RECOMMENDATIONS: Final[Mapping[str, str]] = {
    "no_penalty": (
        "A model trained on CEDA's leads 0 to 5 can be scored on Open-Meteo's UKV analysis for "
        "this input, at the precision tested."
    ),
    "penalty": (
        "For this input, train on Open-Meteo's UKV history only (from 2024-08). A calibrator is "
        "an alternative only where the exploratory calibrator test in the report shows it "
        "absorbs the penalty."
    ),
    "unresolved": (
        "Unresolved, so by the rule fixed in the plan, treat as a penalty: do not mix the two "
        "archives until a longer overlap exists. This is a default for an unresolved reading, and "
        "not a measured penalty."
    ),
}
"""The scoped recommendation of each P3 reading. P3 measures transfer to Open-Meteo's lead-0
analysis only, and says nothing about leads beyond 5 hours."""


def verdict_bound(*, verdict: Verdict) -> float:
    """Return the largest upper bound of a verdict's readings, in points of capacity.

    Args:
        verdict: One planned contrast's verdict.

    Returns:
        The largest 95% upper bound across its settings and fits, the effect that is not excluded.
    """
    records = (verdict.primary, verdict.second, *verdict.sensitivity)
    return max(record["upper_95_pp"] for record in records)


def decision_text(*, found: Sequence[Verdict], observations: Sequence[str] = ()) -> str:
    """Apply the plan's rule to the verdicts.

    Args:
        found: `verdicts`' result.
        observations: Lines of what the saved losses show about the solar fits, which the
            decision prints in place of a conjecture.

    Returns:
        The decision, in Markdown.
    """
    lines = ["### Decision, by the rule fixed before any result", ""]
    for verdict in found:
        unit = f"{verdict.domain}, {verdict.label}, scope {PLANNED_SCOPE[verdict.domain]}"
        sensitivity = "".join(
            f"; era-0-trained {record['setting']} {record['reading']}: "
            f"{record['difference_pp']:+.3f} "
            f"[{record['lower_95_pp']:+.3f}, {record['upper_95_pp']:+.3f}]"
            for record in verdict.sensitivity
        )
        lines.append(
            f"- {unit}: {verdict.reading} (primary {verdict.primary['reading']}: "
            f"{verdict.primary['difference_pp']:+.3f} points of capacity "
            f"[{verdict.primary['lower_95_pp']:+.3f}, {verdict.primary['upper_95_pp']:+.3f}]; "
            f"second {verdict.second['reading']}: {verdict.second['difference_pp']:+.3f} "
            f"[{verdict.second['lower_95_pp']:+.3f}, {verdict.second['upper_95_pp']:+.3f}]; "
            f"margin {verdict.primary['margin_pp']:.2f}{sensitivity})."
        )
    lines += [
        "",
        (
            "Solar contrasts are read on era 0 only, by two fits: one trained on the rows of both "
            "eras and one trained on era 0 alone, and a verdict stands only if the two agree. "
            "After the PS47 upgrade Open-Meteo's hourly irradiance is built differently, so the "
            "all-rows fit trains on era-1 rows whose irradiance is not like for like. The "
            "observations below say what that did. Every solar scope that includes era 1 is "
            "exploratory and carries the era-1 note."
        ),
        "",
        "#### Training history (question 5), scoped",
        "",
    ]
    for verdict in found:
        if verdict.label == "P3":
            bound = (
                " The largest upper bound across the readings is "
                f"{verdict_bound(verdict=verdict):+.3f} points of capacity."
                if verdict.reading == "unresolved"
                else ""
            )
            lines.append(f"- {verdict.domain}: {RECOMMENDATIONS[verdict.reading]}{bound}")
    if observations:
        lines += ["", "#### What the saved losses show (exploratory)", "", *observations]
    lines += [
        "",
        (
            "The study says nothing about leads beyond 5 hours, and Open-Meteo's wind and "
            "temperature have not been checked against the Met Office's files."
        ),
        "",
        LICENCE_CONDITION,
        "",
    ]
    return "\n".join(lines)


# --- The report -----------------------------------------------------------------------------------


def _record_line(*, record: IntervalRecord) -> str:
    return (
        f"| {record['label']} | {record['scope']} | {record['setting']} | {record['treatment']} | "
        f"{record['reference']} | {record['difference_pp']:+.3f} "
        f"[{record['lower_95_pp']:+.3f}, {record['upper_95_pp']:+.3f}] | {record['reading']} | "
        f"{record['n_months']} | {'planned' if record['planned'] else 'exploratory'} | "
        f"{record['note']} |"
    )


RECORD_HEADER: Final[tuple[str, str]] = (
    (
        "| Label | Scope | Setting | Treatment | Reference | Difference, points of capacity | "
        "Reading | Months | Kind | Note |"
    ),
    "|---|---|---|---|---|---|---|---|---|---|",
)


def absolute_lines(*, losses: pl.DataFrame) -> list[str]:
    """Report every arm's absolute error with its month-resampled interval.

    Args:
        losses: A domain's losses.

    Returns:
        Markdown lines.
    """
    lines = [
        "| Arm | Setting | Mean absolute error, % of capacity | 95% interval |",
        "|---|---|---|---|",
    ]
    for arm in sorted(losses["arm"].unique().to_list()):
        for setting in (PRIMARY_SETTING, SECOND_SETTING):
            rows = losses.filter((pl.col("arm") == arm) & (pl.col("setting") == setting))
            if rows.is_empty():
                continue
            interval = bootstrap_absolute(losses=rows, arm=arm, metric=METRIC)
            lines.append(
                f"| {arm} | {setting} | {interval['value'] * PERCENTAGE_POINTS:.3f} | "
                f"[{interval['lower_95'] * PERCENTAGE_POINTS:.3f}, "
                f"{interval['upper_95'] * PERCENTAGE_POINTS:.3f}] |"
            )
    return lines


def planned_scope_losses(*, domain: DomainType, losses: pl.DataFrame) -> pl.DataFrame:
    """Restrict a domain's losses to its planned scope.

    Args:
        domain: The row set.
        losses: The domain's losses, carrying `month`.

    Returns:
        The losses of the planned scope: all rows for wind and the era-0 solar fit, and the
        months before the PS47 upgrade for the solar fit on both eras.
    """
    if PLANNED_SCOPE[domain] == "era 0":
        return losses.filter(pl.col("month") < ERA_BOUNDARY_MONTH)
    return losses


def planned_scope_absolute_lines(*, losses: Mapping[DomainType, pl.DataFrame]) -> list[str]:
    """Report each arm's absolute error on its domain's planned scope, with its interval.

    Args:
        losses: Each domain's losses.

    Returns:
        Markdown lines: one table per domain, on the scope the planned contrasts are read on.
    """
    lines: list[str] = []
    for domain, frame in losses.items():
        lines += [
            f"## {domain}: every arm's absolute error, planned scope ({PLANNED_SCOPE[domain]})",
            "",
            *absolute_lines(losses=planned_scope_losses(domain=domain, losses=frame)),
            "",
        ]
    return lines


def mean_signed_error_pp(*, losses: pl.DataFrame, arm: str, setting: str) -> float:
    """Return an arm's mean signed capped error as a share of capacity, in percentage points.

    Args:
        losses: Losses carrying `signed_error_capped_mw` and `effective_capacity_mw`.
        arm: The arm.
        setting: The hyperparameter setting.

    Returns:
        The mean of (prediction minus actual) over capacity, so a negative value is an
        under-prediction.
    """
    rows = losses.filter((pl.col("arm") == arm) & (pl.col("setting") == setting))
    share = rows["signed_error_capped_mw"] / rows["effective_capacity_mw"]
    return float(np.mean(share.to_numpy())) * PERCENTAGE_POINTS


def signed_error_lines(*, losses: Mapping[DomainType, pl.DataFrame]) -> list[str]:
    """Report each arm's mean signed error on the planned scope, in all hours and at lead 0.

    Args:
        losses: Each domain's losses.

    Returns:
        Markdown lines. A level bias that the absolute error hides, such as a transfer penalty that
        is a systematic under-prediction, shows here.
    """
    lines: list[str] = []
    for domain, frame in losses.items():
        scoped = planned_scope_losses(domain=domain, losses=frame)
        at_lead_0 = scoped.filter(pl.col("time").dt.hour() % LEAD_CYCLE_HOURS == 0)
        lines += [
            f"## {domain}: mean signed error on the planned scope ({PLANNED_SCOPE[domain]})",
            "",
            (
                "Prediction minus measured power as a share of capacity, in percentage points, "
                "so a negative value is an under-prediction."
            ),
            "",
            "| Arm | Setting | All hours | CEDA lead 0 only |",
            "|---|---|---|---|",
        ]
        for arm in sorted(scoped["arm"].unique().to_list()):
            for setting in (PRIMARY_SETTING, SECOND_SETTING):
                if scoped.filter(
                    (pl.col("arm") == arm) & (pl.col("setting") == setting)
                ).is_empty():
                    continue
                lines.append(
                    f"| {arm} | {setting} | "
                    f"{mean_signed_error_pp(losses=scoped, arm=arm, setting=setting):+.3f} | "
                    f"{mean_signed_error_pp(losses=at_lead_0, arm=arm, setting=setting):+.3f} |"
                )
        lines.append("")
    return lines


def by_lead_lines(*, records: Sequence[IntervalRecord]) -> list[str]:
    """Tabulate each planned contrast by CEDA lead, on its planned era, at both settings.

    **P1 and P2 read all hours, so they measure CEDA's lead 0 to 5 archive against Open-Meteo's
    lead-0 analysis as much as a difference between the two archives.** This table shows how each
    contrast changes with CEDA's lead.

    Args:
        records: Every interval record.

    Returns:
        Markdown lines.
    """
    lines = [
        "## Each planned contrast by CEDA lead (exploratory)",
        "",
        (
            "A row's CEDA lead is its UTC hour modulo 6. Wind reads all months, and solar reads "
            "era 0. The models are trained on all leads, so a lead is a scoring subset."
        ),
        "",
        "| Contrast | Row set | Setting | "
        + " | ".join(f"Lead {lead}" for lead in range(LEAD_CYCLE_HOURS))
        + " |",
        "|---|---|---|" + "---|" * LEAD_CYCLE_HOURS,
    ]
    for planned in PLANNED_CONTRASTS:
        domain = planned.domain
        for setting in (PRIMARY_SETTING, SECOND_SETTING):
            cells = []
            for lead in range(LEAD_CYCLE_HOURS):
                scope = (
                    LEAD_SCOPE_FORMAT.format(lead=lead)
                    if PLANNED_SCOPE[domain] == "all"
                    else ERA_0_LEAD_SCOPE_FORMAT.format(lead=lead)
                )
                found = [
                    r
                    for r in records
                    if r["domain"] == domain
                    and r["label"] == planned.label
                    and r["treatment"] == planned.treatment
                    and r["setting"] == setting
                    and r["scope"] == scope
                ]
                cells.append(
                    f"{found[0]['difference_pp']:+.3f} "
                    f"[{found[0]['lower_95_pp']:+.3f}, {found[0]['upper_95_pp']:+.3f}]"
                    if found
                    else "-"
                )
            lines.append(
                f"| {planned.label}, {planned.treatment} | {domain} | {setting} | "
                + " | ".join(cells)
                + " |"
            )
    return [*lines, ""]


def _find(
    *,
    records: Sequence[IntervalRecord],
    domain: str,
    label: str,
    scope: str,
    setting: str = PRIMARY_SETTING,
    treatment: str | None = None,
) -> IntervalRecord | None:
    for record in records:
        if (
            record["domain"] == domain
            and record["label"] == label
            and record["scope"] == scope
            and record["setting"] == setting
            and (treatment is None or record["treatment"] == treatment)
        ):
            return record
    return None


def _show(*, record: IntervalRecord | None) -> str:
    if record is None:
        return "not computed"
    return (
        f"{record['difference_pp']:+.3f} [{record['lower_95_pp']:+.3f}, "
        f"{record['upper_95_pp']:+.3f}] ({record['reading']})"
    )


def observation_lines(
    *, records: Sequence[IntervalRecord], losses: Mapping[DomainType, pl.DataFrame]
) -> list[str]:
    """State what the saved losses show about the contrasts, for the decision and the page.

    Args:
        records: Every interval record.
        losses: Each domain's losses.

    Returns:
        Markdown bullet lines. All are exploratory: the lead-0 reads, the solar fits' absolute
        errors, the wind transfer penalty's level bias, the calibrator test, the controls, and the
        refit noise floor.
    """
    wind_p1 = _find(records=records, domain="wind", label="P1", scope="lead 0 only")
    solar_p2 = _find(records=records, domain="solar", label="P2", scope="era 0, lead 0 only")
    lines = [
        (
            "- **P1 and P2 read all hours, so they measure CEDA's lead 0 to 5 archive against "
            "Open-Meteo's lead-0 analysis.** At CEDA lead 0, P1 reads "
            f"{_show(record=wind_p1)} and P2 reads {_show(record=solar_p2)}. The by-lead table "
            "shows how each contrast grows with lead."
        )
    ]
    solar_all = planned_scope_losses(domain="solar", losses=losses["solar"])
    solar_0 = losses["solar_era0"]
    lines.append(
        "- **The Open-Meteo-trained solar model scores "
        f"{_mae(solar_all, 'om_ghi_temp')} on era 0 when trained on both eras and "
        f"{_mae(solar_0, 'om_ghi_temp')} when trained on era 0 alone** (percent of capacity, "
        "primary setting). The CEDA-trained model scored on Open-Meteo's inputs went from "
        f"{_mae(solar_all, 'ceda_ghi_temp_scored_on_om')} "
        f"(both eras) to {_mae(solar_0, 'ceda_ghi_temp_scored_on_om')} (era 0 alone)."
    )
    zero_0 = solar_0.filter(pl.col("time").dt.hour() % LEAD_CYCLE_HOURS == 0)
    lines.append(
        "- **At CEDA lead 0 the era-0-trained CEDA solar model scores "
        f"{_mae(zero_0, 'ceda_ghi_temp')} on CEDA's inputs and "
        f"{_mae(zero_0, 'ceda_ghi_temp_scored_on_om')} on Open-Meteo's,** so its transfer penalty "
        "is a difference between the two models and not an input mismatch: the CEDA-trained "
        "model also learned from CEDA's lead-1-to-5 inputs."
    )
    wind = losses["wind"]
    signed = {
        arm: mean_signed_error_pp(losses=wind, arm=arm, setting=PRIMARY_SETTING)
        for arm in ("ceda_wind_10m", "om_wind_10m", "ceda_wind_10m_scored_on_om")
    }
    direction = _find(
        records=records, domain="wind", label="CEDA model, Open-Meteo direction", scope="all"
    )
    rescaled = _find(
        records=records,
        domain="wind",
        label="CEDA model, Open-Meteo speed rescaled (calibrator test)",
        scope="all",
    )
    lines.append(
        "- **The wind transfer penalty is a level bias.** The mean signed error is "
        f"{signed['ceda_wind_10m']:+.2f} for the CEDA model on CEDA's inputs, "
        f"{signed['om_wind_10m']:+.2f} for the Open-Meteo model, and "
        f"{signed['ceda_wind_10m_scored_on_om']:+.2f} for the CEDA model on Open-Meteo's inputs "
        "(percentage points of capacity; negative is an under-prediction). Open-Meteo's 10 m "
        "speed sits about 3% below the nearest CEDA cell's at lead 0, and swapping the direction "
        f"alone reads {_show(record=direction)}."
    )
    lines.append(
        "- **Calibrator test (exploratory):** the CEDA model scored on Open-Meteo's speed rescaled "
        f"to CEDA's level, against the Open-Meteo model, reads {_show(record=rescaled)}. The scale "
        "is each site's median CEDA-to-Open-Meteo speed ratio at lead-0 instants, learned on the "
        "training folds."
    )
    wind_control = _find(
        records=records,
        domain="wind",
        label="control",
        scope="all",
        treatment="ceda_wind_10m_shuffled",
    )
    solar_control = _find(
        records=records,
        domain="solar",
        label="control",
        scope="era 0",
        treatment="ceda_ghi_temp_shuffled",
    )
    cpu = _find(records=records, domain="wind", label="GPU against CPU", scope="all")
    lines.append(
        "- **The shuffled controls are not a clean null here.** The wind pair reads "
        f"{_show(record=wind_control)} and the solar pair {_show(record=solar_control)}. A shuffle "
        "within a site, month, and hour keeps each archive's monthly-hourly distribution, which "
        "carries Open-Meteo's lower speed level and CEDA's lead-dependent spread, so the two "
        "shuffled arms are not equally informative by construction. The GPU-against-CPU refit "
        f"reads {_show(record=cpu)}, which is the noise floor of a refit."
    )
    lines.append(
        "- **About one exploratory row in 20 with no real effect reaches the 5% level by chance,** "
        "and the report holds many such rows over shared months. Read the exploratory labels "
        "with their scope and months."
    )
    return lines


def _mae(losses: pl.DataFrame, arm: str) -> str:
    rows = losses.filter((pl.col("arm") == arm) & (pl.col("setting") == PRIMARY_SETTING))
    return f"{float(np.mean(rows[METRIC].to_numpy())) * PERCENTAGE_POINTS:.2f}"


def report_text(
    *,
    records: Sequence[IntervalRecord],
    losses: Mapping[DomainType, pl.DataFrame],
    job_list: Sequence[Job],
    found: Sequence[Verdict],
) -> str:
    """Render every table the page quotes.

    Args:
        records: Every interval record.
        losses: Each domain's losses.
        job_list: Every fit's job, for the arm columns.
        found: `verdicts`' result.

    Returns:
        The report, in Markdown.
    """
    lines = ["# UKV from CEDA against UKV from Open-Meteo: set B report", ""]
    lines += [*_arm_columns_lines(job_list=list(job_list)), ""]
    for domain, frame in losses.items():
        lines += [f"## {domain}: every arm's absolute error", "", *absolute_lines(losses=frame), ""]
        lines += [f"## {domain}: contrasts", "", *RECORD_HEADER]
        lines += [_record_line(record=record) for record in records if record["domain"] == domain]
        lines.append("")
    lines += [
        *planned_scope_absolute_lines(losses=losses),
        *signed_error_lines(losses=losses),
        *by_lead_lines(records=records),
        decision_text(found=found, observations=observation_lines(records=records, losses=losses)),
    ]
    return "\n".join(lines)


# --- Running --------------------------------------------------------------------------------------


def load_frames(*, directory: Path) -> dict[DomainType, pl.DataFrame]:
    """Read the build's three row files.

    Args:
        directory: The output folder.

    Returns:
        The rows of each domain.
    """
    return {domain: pl.read_parquet(directory / name) for domain, name in ROW_NAMES.items()}


def dry_run_lines(*, frames: Mapping[DomainType, pl.DataFrame]) -> list[str]:
    """List every fit and fit nothing.

    Args:
        frames: The rows of each domain.

    Returns:
        Markdown lines.
    """
    lines = ["### Dry run: nothing is fitted", ""]
    total = 0
    for domain, frame in frames.items():
        sites = frame["site"].n_unique()
        jobs = all_jobs(domain=domain)
        total += len(jobs) * sites
        lines.append(
            f"- {domain}: {frame.height:,} rows, {sites} sites, {len(jobs)} jobs "
            f"({len(transfer_jobs(domain=domain))} scored on {len(scoring_names(domain=domain))} "
            f"frames) = {len(jobs) * sites} fit-sets."
        )
        lines += [
            f"  - {arm} / {setting}: {len(columns)} columns"
            for arm, setting, _, columns, _, _ in jobs
        ]
    lines.append(f"- Total {total} fit-sets, plus {len(cpu_refit_jobs()) * 3} on the CPU.")
    return lines


def check_verified(*, directory: Path) -> None:
    """Raise unless the build passed every guard and the row files are the ones it stamped.

    Args:
        directory: The output folder.

    Raises:
        ValueError: If `build.json` is missing, says a guard failed, or holds no hash for a row
            file; if a row file's hash differs from the stamped one; or if `direct_report.md`
            does not exist.
    """
    stamp_path = directory / BUILD_STAMP_NAME
    if not stamp_path.exists():
        msg = f"{stamp_path} is missing: run the build first"
        raise ValueError(msg)
    stamp = json.loads(stamp_path.read_text())
    if not stamp.get("guards_passed"):
        msg = f"{stamp_path} does not say every guard passed"
        raise ValueError(msg)
    for name in ROW_NAMES.values():
        stamped = stamp.get("row_file_hashes", {}).get(name)
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if stamped != actual:
            msg = f"{name} does not hash to the value the build stamped"
            raise ValueError(msg)
    if not (directory / DIRECT_REPORT_NAME).exists():
        msg = f"{DIRECT_REPORT_NAME} is missing: run the compare script and read its report"
        raise ValueError(msg)


def planned_outputs(*, directory: Path) -> list[Path]:
    """List every file a full fit run writes, so a stale one stops the run before it starts.

    Args:
        directory: The output folder.

    Returns:
        The hardware stamp, every domain's losses and fingerprints, the CPU refit's, and the
        three reports.
    """
    paths = [directory / STAMP_NAME]
    for stem in (*(f"losses_{domain}" for domain in ROW_NAMES), CPU_REFIT_STEM):
        paths += list(_paths(directory=directory, stem=stem))
    return [*paths, directory / REPORT_NAME, directory / DECISION_NAME, directory / INTERVALS_NAME]


def main() -> int:
    """Fit, or rebuild the reports from saved losses."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="List every fit; fit nothing.")
    mode.add_argument("--report-only", action="store_true", help="Rebuild the reports.")
    parser.add_argument("--verified", action="store_true", help="The build check passed.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--directory", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    directory: Path = arguments.directory
    check_arm_widths()
    frames = load_frames(directory=directory)

    if arguments.dry_run:
        sys.stdout.write("\n".join(dry_run_lines(frames=frames)) + "\n")
        return 0
    if not arguments.report_only:
        if not arguments.verified:
            sys.stdout.write("Refusing to fit without --verified.\n")
            return 2
        check_verified(directory=directory)
        refuse_to_overwrite(paths=planned_outputs(directory=directory))
        stamp_path = directory / STAMP_NAME
        stamp_path.write_text(json.dumps(hardware_stamp(device=arguments.device), indent=2))
        for domain, frame in frames.items():
            fit_domain(domain=domain, frame=frame, directory=directory, device=arguments.device)
        fit_cpu_refit(frame=frames["wind"], directory=directory)

    losses: dict[DomainType, pl.DataFrame] = {}
    for domain, frame in frames.items():
        losses[domain] = load_losses(
            stem=f"losses_{domain}", frame=frame, jobs=all_jobs(domain=domain), directory=directory
        )
    cpu = load_losses(
        stem=CPU_REFIT_STEM, frame=frames["wind"], jobs=cpu_refit_jobs(), directory=directory
    )
    losses["wind"] = pl.concat([losses["wind"], cpu.select(losses["wind"].columns)])
    losses["solar"] = losses["solar"].join(
        frames["solar"].select("site", "time", "solar_zenith_deg"), on=["site", "time"], how="left"
    )
    era_1_note = json.loads((directory / BUILD_STAMP_NAME).read_text())["era_1_irradiance_note"]
    records = [
        record
        for domain, domain_losses in losses.items()
        for record in domain_records(domain=domain, losses=domain_losses, era_1_note=era_1_note)
    ]
    found = verdicts(records=records)
    job_list = [job for domain in frames for job in all_jobs(domain=domain)] + cpu_refit_jobs()
    report = report_text(records=records, losses=losses, job_list=job_list, found=found)
    paths = [directory / REPORT_NAME, directory / DECISION_NAME, directory / INTERVALS_NAME]
    if arguments.report_only:
        refuse_to_overwrite(paths=paths)
    paths[0].write_text(report)
    paths[1].write_text(
        decision_text(found=found, observations=observation_lines(records=records, losses=losses))
    )
    pl.DataFrame(records).write_parquet(paths[2])
    sys.stdout.write(f"Wrote the reports to {directory}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
