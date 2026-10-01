"""Fit ENS plus one weather product (ICON-EU, UKV, WeatherNext 3) into one write-once folder.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1007>.
It answers which single weather product to add to the ECMWF ENS mean first. A blend is one XGBoost
model given ENS's mean columns plus one product's columns; its reference is an XGBoost model given
ENS's mean columns alone, and its control has the product's columns shuffled within generator,
year-month, and hour of day (`fit_aifs.add_shuffled_columns`). The settings, seeds, folds, eras,
column counts, and GPU device are those of `fit_aifs.py`, which this script imports and does not
change.

**AIFS Single's blends are reused, not refitted.** `nwp_forecast_comparison_aifs_blends` holds the
AIFS Single blend, its control, and ENS's mean at days 1, 2, 7, and 14 on the `single` rows. This
script builds the same rows, computes every (arm, setting) pair the contrasts C1 to C5 need that
the folder lacks (mostly the second hyperparameter setting), and refits exactly those pairs. Before
any fit it raises unless its build stamp (inputs, settings, seeds, GPU, XGBoost version) equals the
reused folder's, and after each fit it raises unless every new arm holds the `(site, time, seed,
fold)` keys of the saved ENS mean.

The fits, each on the GPU:

- `single` rows, days 1 and 2: the ICON-EU blend at the optimistic lead (`icon_eu_day<N>`) and at
  the conservative lead (`icon_eu_day<N+1>`), and the UKV blend at day 1, each with its control, at
  both settings. The missing AIFS Single pairs at both settings.
- `wn3` rows (February to April and June to 10 September 2026), days 1, 2, 7, and 14: ENS's mean,
  the WeatherNext 3 blend, and its control, at the primary setting. Descriptive, never ranked.

`report.md` prints every arm's columns, the contrasts C1 to C5 at both settings, and the ranking
rule's verdicts. Every output carries only the anonymised `site` label.

`--dry-run` builds every frame and prints the fits without fitting or writing anything. `--check`
fits one arm at one wind site twice on the GPU and prints the time.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final, NamedTuple

import fit_aifs
import polars as pl
from fit_aifs import (
    PRIMARY,
    SENSITIVITY,
    SETTINGS,
    Job,
    blend_arm_name,
    control_shuffles,
    product_blend_arms,
)
from fit_extra_leads import interval_text
from nwp_forecast_comparison import (
    METRIC,
    PERCENTAGE_POINTS,
    DomainType,
    difference,
    predictions_from_losses,
)
from studies.bootstrap import (
    NO_DETECTABLE_DIFFERENCE,
    BootstrapInterval,
    bootstrap_difference_at_level,
    combine_setting_verdicts,
)
from studies.guards import refuse_to_overwrite

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

OUTPUT_DIR_NAME: Final[str] = "nwp_forecast_comparison_product_blends"
"""Under `data/studies/`, the only folder this script writes to."""

WN3_DIR_NAME: Final[str] = "nwp_forecast_comparison_wn3"
"""Under `data/studies/`, the folder holding `<domain>_wn3_inputs.parquet`, which is only read."""

REPORT_NAME: Final[str] = "report.md"
README_NAME: Final[str] = "README.md"

ROW_KEYS: Final[tuple[str, ...]] = ("site", "time", "seed", "fold")
"""The columns that identify one scored row of one arm."""

SAME_BUILD_KEYS: Final[tuple[str, ...]] = (
    "inputs_sha256",
    "published_sha256",
    "extra_sha256",
    "device",
    "gpu",
    "xgboost",
    "settings",
    "seeds",
)
"""The stamp entries that must equal the reused folder's. `columns` differs by design, because the
new arms are not in the reused folder, and the Polars version does not change a fit."""

REUSED_DAYS: Final[tuple[int, ...]] = (1, 2, 7)
"""The days at which the AIFS Single blend is a contrast's treatment (C2), so its arms need both
settings."""

WN3_DAYS: Final[tuple[int, ...]] = fit_aifs.BLEND_DAYS

C5_DAYS: Final[tuple[int, ...]] = (1, 2)
"""The days at which ICON-EU has a blend to compare AIFS Single's against."""

README_TEXT: Final[
    str
] = """# ENS plus one weather product: ICON-EU, UKV, WeatherNext 3 (write-once)

Fits for `docs/studies/forecasts/blends-with-ens.md`. Never overwrite a file in this folder.

- `<domain>_single_day<N>_losses.parquet` holds, on the `single` rows, the ICON-EU and UKV blends
  with their controls at both settings, and the (arm, setting) pairs of the AIFS Single blend, its
  control, and ENS's mean that `nwp_forecast_comparison_aifs_blends` lacks. The other pairs are read
  from that folder.
- `<domain>_wn3_day<N>_losses.parquet` holds ENS's mean, the WeatherNext 3 blend, and its control on
  the `wn3` rows, at the primary setting.
- Each losses file has a `_predictions.parquet` and a `.json` stamp naming the device, the inputs'
  SHA-256, and the settings. `report.md` prints every arm's columns and the contrasts C1 to C5.
"""


class Stage(NamedTuple):
    """One frame to build and fit: a technology, a row set, and a lead day."""

    domain: DomainType
    row_set: str
    day: int


def reused_arms(*, day: int) -> tuple[str, ...]:
    """Return the AIFS Single arms that the contrasts read at `day`: the blend, its control, ENS."""
    return (
        blend_arm_name(product="aifs_single", day=day),
        blend_arm_name(product="aifs_single", day=day, role="_control"),
        f"ens_mean_day{day}",
    )


def missing_jobs(*, saved: pl.DataFrame, arms: tuple[str, ...]) -> list[Job]:
    """Return every (arm, setting) pair of `arms` that the saved losses lack."""
    have = set(saved.select("arm", "setting").unique().iter_rows())
    return [(arm, setting) for arm in arms for setting in SETTINGS if (arm, setting) not in have]


def stage_jobs(*, stage: Stage, saved: pl.DataFrame | None) -> list[Job]:
    """Return the (arm, setting) fits of one stage.

    Args:
        stage: The stage.
        saved: On `single`, the reused folder's losses at this day, whose missing pairs are
            refitted. On `wn3`, `None`.

    Returns:
        On `single`, every new arm at both settings, then each AIFS Single pair that `saved`
        lacks. On `wn3`, every arm at the primary setting.
    """
    arms = product_blend_arms(row_set=stage.row_set, day=stage.day)
    if stage.row_set == fit_aifs.WN3_ROW_SET:
        return [(arm, PRIMARY) for arm in arms]
    jobs = [(arm, setting) for arm in arms for setting in SETTINGS]
    if saved is not None and stage.day in REUSED_DAYS:
        jobs += missing_jobs(saved=saved, arms=reused_arms(day=stage.day))
    return jobs


def check_same_build(*, stamp: dict[str, str], reused_stamp: dict[str, str]) -> None:
    """Raise unless this build's stamp equals the reused folder's on `SAME_BUILD_KEYS`.

    Args:
        stamp: This run's `fit_aifs.build_stamp` result.
        reused_stamp: The stamp saved beside the reused folder's losses.

    Raises:
        ValueError: Naming each entry that differs, so a refit never pairs with reused losses from
            another input file, setting, seed set, GPU, or XGBoost version.
    """
    differing = [key for key in SAME_BUILD_KEYS if stamp.get(key) != reused_stamp.get(key)]
    if differing:
        msg = f"this build differs from the reused folder's in {differing}"
        raise ValueError(msg)


def check_keys_match(
    *, losses: pl.DataFrame, saved: pl.DataFrame, reference_arm: str, label: str
) -> None:
    """Raise unless every new (arm, setting) holds the saved reference arm's scored rows.

    Args:
        losses: The new fits' per-row losses.
        saved: The reused folder's losses at the same day.
        reference_arm: The saved arm whose `ROW_KEYS` every new arm must equal, ENS's mean.
        label: The stage, for the message.

    Raises:
        ValueError: Naming the arm and setting whose keys differ: a row, a seed, a fold, or a
            duplicate.
    """
    want = (
        saved.filter(pl.col("arm") == reference_arm, pl.col("setting") == PRIMARY)
        .select(ROW_KEYS)
        .sort(ROW_KEYS)
    )
    for (arm, setting), group in losses.group_by(["arm", "setting"], maintain_order=True):
        if not group.select(ROW_KEYS).sort(ROW_KEYS).equals(want):
            msg = (
                f"{label}: {arm} at {setting} does not hold {reference_arm}'s "
                "(site, time, seed, fold)"
            )
            raise ValueError(msg)


def check_frame_matches_saved(*, frame: pl.DataFrame, saved: pl.DataFrame, stage: Stage) -> None:
    """Raise unless the stage's rows are the `(site, time, fold)` rows of the saved ENS mean.

    This runs before any fit, so `--dry-run` catches a row set that differs from the reused
    folder's.

    Args:
        frame: The stage's rows.
        saved: The reused folder's losses at the stage's day.
        stage: The stage, for the message.

    Raises:
        ValueError: If the rows differ in any site, time, or fold.
    """
    keys = ["site", "time", "fold"]
    want = (
        saved.filter(pl.col("arm") == f"ens_mean_day{stage.day}", pl.col("setting") == PRIMARY)
        .select(keys)
        .unique()
        .sort(keys)
    )
    if not frame.select(keys).unique().sort(keys).equals(want):
        msg = f"{stage.domain}/single_day{stage.day}: the rows differ from the saved ENS mean's"
        raise ValueError(msg)


def check_reproduces_saved(
    *, frame: pl.DataFrame, saved: pl.DataFrame, arm: str, domain: DomainType
) -> bool:
    """Refit one saved arm at its first site and return whether the rows equal the saved ones.

    The comparison covers `(site, time, seed, fold, absolute_error_mw)`, so it tests the rows, the
    folds, the shuffle, and the device together.

    Args:
        frame: A stage's rows, carrying the arm's columns.
        saved: The reused folder's losses at the stage's day.
        arm: The saved arm to refit at the primary setting.
        domain: `solar` or `wind`.

    Returns:
        Whether the refit's rows equal the saved rows exactly.
    """
    site = min(frame["site"].unique().to_list())
    losses = fit_aifs.fit_jobs(
        frame=frame.filter(pl.col("site") == site), domain=domain, jobs=[(arm, PRIMARY)], workers=1
    )
    columns = [*ROW_KEYS, "absolute_error_mw"]
    want = saved.filter(
        pl.col("arm") == arm, pl.col("setting") == PRIMARY, pl.col("site") == site
    ).select(columns)
    return losses.select(columns).sort(ROW_KEYS).equals(want.sort(ROW_KEYS))


def write_atomically(*, path: Path, frame: pl.DataFrame) -> None:
    """Write a parquet file to a temporary name and rename it, so a crash leaves no partial file."""
    temporary = path.with_name(path.name + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


def check_output_dir(*, output_dir: Path, read_only: list[Path]) -> None:
    """Raise unless `output_dir` is the one folder this script may write to.

    Args:
        output_dir: Where the fits would write.
        read_only: The folders the script reads, which it never writes to.

    Raises:
        ValueError: If `output_dir` is a folder the script reads, or has another name than
            `OUTPUT_DIR_NAME`.
    """
    resolved = {folder.resolve() for folder in read_only}
    if output_dir.resolve() in resolved or output_dir.name != OUTPUT_DIR_NAME:
        msg = f"this script writes only to a folder named {OUTPUT_DIR_NAME}, not {output_dir}"
        raise ValueError(msg)


def fit_single_stage(
    *, frame: pl.DataFrame, stage: Stage, saved: pl.DataFrame, workers: int
) -> pl.DataFrame:
    """Fit one `single` stage and check its rows equal the saved ENS mean's.

    Args:
        frame: The stage's rows.
        stage: The stage.
        saved: The reused folder's losses at this day.
        workers: How many (arm, site) fits run at once.

    Returns:
        The new per-row losses, stamped with the device.
    """
    jobs = stage_jobs(stage=stage, saved=saved)
    losses = fit_aifs.fit_jobs(frame=frame, domain=stage.domain, jobs=jobs, workers=workers)
    check_keys_match(
        losses=losses,
        saved=saved,
        reference_arm=f"ens_mean_day{stage.day}",
        label=f"{stage.domain}/single_day{stage.day}",
    )
    return losses.with_columns(device=pl.lit(fit_aifs.DEVICE))


def combine_losses(
    *, saved: pl.DataFrame, new: pl.DataFrame | None, arms: tuple[str, ...]
) -> pl.DataFrame:
    """Return the saved losses of `arms` stacked on the new losses, one row per (arm, setting, key).

    Args:
        saved: The reused folder's losses at one day.
        new: This run's losses at the same day, or `None` where the day has no stage.
        arms: The saved arms to keep.

    Returns:
        The saved rows of `arms`, then the new rows.

    Raises:
        ValueError: If an (arm, setting) is in both, which would count its rows twice.
    """
    kept = saved.filter(pl.col("arm").is_in(arms))
    if new is None:
        return kept
    kept = kept.select(new.columns)
    both = set(kept.select("arm", "setting").unique().iter_rows()) & set(
        new.select("arm", "setting").unique().iter_rows()
    )
    if both:
        msg = f"these (arm, setting) pairs are both saved and refitted: {sorted(both)}"
        raise ValueError(msg)
    return pl.concat([kept, new])


# --- Contrasts and the ranking rule --------------------------------------------------------------


class ProductContrast(NamedTuple):
    """One paired contrast, treatment minus reference, at a lead day."""

    code: str
    day: int
    treatment: str
    reference: str
    planned: bool


def product_contrasts() -> list[ProductContrast]:
    """Return the planned contrasts C1 to C5 and the exploratory optimistic ICON-EU ones.

    C1 to C3 are each blend minus ENS's mean alone, C4 is each blend minus its own control, and C5
    is the AIFS Single blend minus each ICON-EU blend. Only the conservative ICON-EU lead is a
    planned C1 and C4 contrast; the optimistic one is exploratory, and UKV (C3) is an optimistic
    upper bound that is never ranked.
    """
    output: list[ProductContrast] = []
    for day in REUSED_DAYS:
        mean = f"ens_mean_day{day}"
        aifs = blend_arm_name(product="aifs_single", day=day)
        output += [
            ProductContrast("C2", day, aifs, mean, planned=True),
            ProductContrast("C4", day, aifs, f"{aifs}_control", planned=True),
        ]
    for product, days in fit_aifs.PRODUCT_BLEND_DAYS.items():
        for day in days:
            blend = blend_arm_name(product=product, day=day)
            mean = f"ens_mean_day{day}"
            planned = product != "icon_eu"
            code = "C3" if product == "ukv" else "C1"
            output += [
                ProductContrast(code, day, blend, mean, planned=planned),
                ProductContrast("C4", day, blend, f"{blend}_control", planned=planned),
            ]
    for day in C5_DAYS:
        aifs = blend_arm_name(product="aifs_single", day=day)
        for product in ("icon_eu_conservative", "icon_eu"):
            output.append(
                ProductContrast(
                    "C5", day, aifs, blend_arm_name(product=product, day=day), planned=True
                )
            )
    return output


def setting_intervals(
    *, losses: pl.DataFrame, contrast: ProductContrast
) -> dict[str, BootstrapInterval]:
    """Return the contrast's interval at each setting the losses hold both arms at."""
    return {
        setting: difference(
            losses=losses.filter(pl.col("setting") == setting),
            treatment=contrast.treatment,
            reference=contrast.reference,
        )
        for setting in SETTINGS
    }


def blend_verdict_at_both_settings(
    *,
    day: int,
    versus_ens: dict[str, BootstrapInterval],
    versus_control: dict[str, BootstrapInterval],
) -> str:
    """Return a blend's verdict, which stands only if both settings give it.

    The verdict at each setting is `fit_aifs.lead_verdict`: the blend lowers the error only where
    its difference from ENS's mean alone and from its control both have an upper 95% bound below
    zero.
    """
    verdicts = {
        setting: fit_aifs.lead_verdict(
            day=day, versus_ens=versus_ens[setting], versus_control=versus_control[setting]
        )["verdict"]
        for setting in SETTINGS
    }
    return combine_setting_verdicts(
        primary=verdicts[PRIMARY],
        sensitivity=verdicts[SENSITIVITY],
        unresolved=NO_DETECTABLE_DIFFERENCE,
    )


def ranking_verdict(
    *,
    versus_optimistic: dict[str, BootstrapInterval],
    versus_conservative: dict[str, BootstrapInterval],
) -> str:
    """Return whether AIFS Single or ICON-EU is ranked first as the product to add to ENS.

    The rule was fixed before any fit. AIFS Single ranks above ICON-EU only if the AIFS Single blend
    minus the optimistic ICON-EU blend has an upper 95% bound below zero at both settings, which
    holds although the optimistic ICON-EU blend reads the fresher run. ICON-EU ranks above AIFS
    Single only if the same difference against the conservative ICON-EU blend has a lower 95% bound
    above zero at both settings, which holds although the conservative blend reads an older run.
    Otherwise the two are not separable.

    Args:
        versus_optimistic: AIFS Single blend minus optimistic ICON-EU blend, at each setting.
        versus_conservative: AIFS Single blend minus conservative ICON-EU blend, at each setting.

    Returns:
        `AIFS Single ranks above ICON-EU`, `ICON-EU ranks above AIFS Single`, or `not separable`.
    """
    aifs_first = all(interval["upper_95"] < 0.0 for interval in versus_optimistic.values())
    icon_first = all(interval["lower_95"] > 0.0 for interval in versus_conservative.values())
    if aifs_first and not icon_first:
        return "AIFS Single ranks above ICON-EU"
    if icon_first and not aifs_first:
        return "ICON-EU ranks above AIFS Single"
    return "not separable"


# --- Report --------------------------------------------------------------------------------------


def interval_cell(*, interval: BootstrapInterval) -> str:
    """Format an interval as `difference [lower, upper]` in points of capacity."""
    return interval_text(
        point=interval["difference"], lower=interval["lower_95"], upper=interval["upper_95"]
    )


def columns_lines(*, arms: list[str]) -> list[str]:
    """Print every arm's feature columns, for each technology, so a reviewer can check them."""
    lines = ["## Columns of every arm", ""]
    for domain in fit_aifs.DOMAINS:
        lines += [f"### {domain}", ""]
        lines += [
            f"- `{arm}` ({len(fit_aifs.arm_features(arm=arm, domain=domain))} columns): "
            + ", ".join(fit_aifs.arm_features(arm=arm, domain=domain))
            for arm in arms
        ]
        lines.append("")
    return lines


def multiplicity_lines(
    *, domain: DomainType, combined: dict[int, pl.DataFrame], n_listed: int
) -> list[str]:
    """Print each blend's control minus ENS's mean and each blend minus ENS's mean, widened.

    A control that is itself worse than ENS's mean alone makes the control guard (C4) weak, so each
    control's difference from ENS's mean is printed at both settings. The blend-minus-ENS contrasts
    (C1 to C3) are also printed with a Bonferroni-widened interval, at a coverage that corrects the
    95% level across every interval the report lists.

    Args:
        domain: `solar` or `wind`.
        combined: The saved and new losses at each day.
        n_listed: How many intervals the report lists at each setting.

    Returns:
        The two tables and a count of controls that are significantly worse than ENS's mean.
    """
    level = 100.0 - 5.0 / n_listed
    lines = [
        f"### {domain}: controls, and Bonferroni-widened blend-minus-ENS intervals",
        "",
        (
            "`Control minus ENS` is the blend's control minus ENS's mean alone. The widened "
            f"interval covers {level:.3f}% (the 95% level divided across {n_listed} intervals). "
            f"The family is all {n_listed} intervals, but only the blend-minus-ENS intervals are "
            "widened, so the correction is conservative. `Survives` follows the `lead_verdict` "
            "rule: the widened upper bound of blend minus ENS and the 95% upper bound of blend "
            "minus its own control are both below zero at both settings."
        ),
        "",
        (
            "| Blend | Day | Blend minus ENS, primary | Blend minus ENS, sensitivity | "
            "Control minus ENS, primary | Control minus ENS, sensitivity | "
            "Widened upper bound, primary | Widened upper bound, sensitivity | Survives |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    worse_controls = 0
    blends = [("aifs_single", day) for day in REUSED_DAYS] + [
        (product, day) for product, days in fit_aifs.PRODUCT_BLEND_DAYS.items() for day in days
    ]
    for product, day in blends:
        blend = blend_arm_name(product=product, day=day)
        mean = f"ens_mean_day{day}"
        cells: dict[str, list[str]] = {"blend": [], "control": [], "upper": []}
        survives = True
        for setting in SETTINGS:
            scope = combined[day].filter(pl.col("setting") == setting)
            versus_ens = difference(losses=scope, treatment=blend, reference=mean)
            control = difference(losses=scope, treatment=f"{blend}_control", reference=mean)
            versus_control = difference(losses=scope, treatment=blend, reference=f"{blend}_control")
            _, upper = bootstrap_difference_at_level(
                losses=scope, treatment=blend, reference=mean, metric=METRIC, level=level
            )
            cells["blend"].append(interval_cell(interval=versus_ens))
            cells["control"].append(interval_cell(interval=control))
            cells["upper"].append(f"{upper * PERCENTAGE_POINTS:+.3f}")
            survives = survives and upper < 0.0 and versus_control["upper_95"] < 0.0
            worse_controls += control["lower_95"] > 0.0
        lines.append(
            f"| `{blend}` | {day} | {' | '.join(cells['blend'])} | {' | '.join(cells['control'])} "
            f"| {' | '.join(cells['upper'])} | {'yes' if survives else 'no'} |"
        )
    lines += [
        "",
        (
            f"{worse_controls} (blend, setting) controls have a lower 95% bound above zero "
            "against ENS's mean alone, so they are significantly worse than ENS's mean."
        ),
        "",
    ]
    return lines


def single_lines(*, domain: DomainType, combined: dict[int, pl.DataFrame]) -> list[str]:
    """Print C1 to C5 at both settings and the verdicts, for one technology."""
    lines = [f"## {domain}: ENS plus one product, `single` rows", ""]
    lines += [
        "| Contrast | Day | Treatment minus reference | Planned | Primary | Sensitivity |",
        "|---|---|---|---|---|---|",
    ]
    found: dict[tuple[str, int, str], dict[str, BootstrapInterval]] = {}
    for contrast in product_contrasts():
        intervals = setting_intervals(losses=combined[contrast.day], contrast=contrast)
        found[(contrast.code, contrast.day, contrast.treatment + contrast.reference)] = intervals
        lines.append(
            f"| {contrast.code} | {contrast.day} | `{contrast.treatment}` minus "
            f"`{contrast.reference}` | {'planned' if contrast.planned else 'exploratory'} | "
            f"{interval_cell(interval=intervals[PRIMARY])} | "
            f"{interval_cell(interval=intervals[SENSITIVITY])} |"
        )
    lines += ["", "Verdicts by the `lead_verdict` rule, which needs both settings to agree:", ""]
    for product, days in fit_aifs.PRODUCT_BLEND_DAYS.items():
        for day in days:
            blend = blend_arm_name(product=product, day=day)
            verdict = blend_verdict_at_both_settings(
                day=day,
                versus_ens=found[
                    ("C3" if product == "ukv" else "C1", day, blend + f"ens_mean_day{day}")
                ],
                versus_control=found[("C4", day, blend + f"{blend}_control")],
            )
            lines.append(f"- `{blend}`: {verdict}")
    for day in REUSED_DAYS:
        blend = blend_arm_name(product="aifs_single", day=day)
        verdict = blend_verdict_at_both_settings(
            day=day,
            versus_ens=found[("C2", day, blend + f"ens_mean_day{day}")],
            versus_control=found[("C4", day, blend + f"{blend}_control")],
        )
        lines.append(f"- `{blend}`: {verdict}")
    lines += ["", "Ranking of AIFS Single against ICON-EU (C5):", ""]
    for day in C5_DAYS:
        aifs = blend_arm_name(product="aifs_single", day=day)
        verdict = ranking_verdict(
            versus_optimistic=found[("C5", day, aifs + blend_arm_name(product="icon_eu", day=day))],
            versus_conservative=found[
                ("C5", day, aifs + blend_arm_name(product="icon_eu_conservative", day=day))
            ],
        )
        lines.append(f"- day {day}: {verdict}")
    lines.append("")
    return lines


def wn3_lines(*, domain: DomainType, losses: dict[int, pl.DataFrame]) -> list[str]:
    """Print the descriptive WeatherNext 3 blend contrasts at the primary setting."""
    lines = [
        f"## {domain}: ENS plus WeatherNext 3, `wn3` rows (7 months, descriptive)",
        "",
        "| Day | Treatment minus reference | Primary |",
        "|---|---|---|",
    ]
    for day in WN3_DAYS:
        blend = blend_arm_name(product="wn3", day=day)
        for reference in (f"ens_mean_day{day}", f"{blend}_control"):
            interval = difference(
                losses=losses[day].filter(pl.col("setting") == PRIMARY),
                treatment=blend,
                reference=reference,
            )
            lines.append(
                f"| {day} | `{blend}` minus `{reference}` | {interval_cell(interval=interval)} |"
            )
    lines.append("")
    return lines


def n_intervals(*, n_domains: int) -> int:
    """Return how many intervals the report lists at each setting."""
    return len(product_contrasts()) * n_domains


def report_text(
    *,
    singles: dict[DomainType, dict[int, pl.DataFrame]],
    wn3: dict[DomainType, dict[int, pl.DataFrame]],
    refits: list[str],
) -> str:
    """Return `report.md`: the header, the refitted pairs, the columns, and every contrast."""
    n_listed = n_intervals(n_domains=len(singles))
    lines = [
        "# ENS plus one weather product: report",
        "",
        (
            "Every fit is on the GPU. Differences are first arm minus second, in percentage points "
            "of capacity, so a negative difference means the first arm has the lower error. "
            "C1 to C3 are each blend minus ENS's mean alone, C4 is each blend minus its own "
            "control, and C5 is the AIFS Single blend minus each ICON-EU blend. The conservative "
            "ICON-EU blend reads the freshest run at least 48 hours before the valid hour, "
            "which is older than the AIFS Single run, so C5 against it is biased towards AIFS "
            "Single; C5 is therefore read "
            "against both ICON-EU blends. UKV is an optimistic upper bound and is never ranked. "
            f"The report prints {n_listed} intervals at each setting, so about {n_listed / 20:.1f} "
            "would reach statistical significance at the 5% level by chance, and every verdict "
            "is uncorrected for multiplicity."
        ),
        "",
        "## AIFS Single pairs refitted because the reused folder lacks them",
        "",
        *(refits or ["None."]),
        "",
        *columns_lines(
            arms=[
                *dict.fromkeys(
                    arm
                    for day in WN3_DAYS
                    for row_set in ("single", fit_aifs.WN3_ROW_SET)
                    for arm in product_blend_arms(row_set=row_set, day=day)
                ),
                *reused_arms(day=1),
            ]
        ),
    ]
    for domain in fit_aifs.DOMAINS:
        lines += single_lines(domain=domain, combined=singles[domain])
        lines += multiplicity_lines(domain=domain, combined=singles[domain], n_listed=n_listed)
        lines += wn3_lines(domain=domain, losses=wn3[domain])
    return "\n".join(lines)


# --- Run -----------------------------------------------------------------------------------------


class PlannedStage(NamedTuple):
    """One stage ready to fit: its rows, its fits, its stamp, and the saved losses it extends."""

    stage: Stage
    frame: pl.DataFrame
    jobs: list[Job]
    stamp: dict[str, str]
    saved: pl.DataFrame | None


def single_days() -> list[int]:
    """Return the days with a `single` stage: each day a contrast reads a blend at."""
    return sorted(
        {*REUSED_DAYS, *(day for days in fit_aifs.PRODUCT_BLEND_DAYS.values() for day in days)}
    )


def plan_stages(
    *,
    published_dir: Path,
    reused_dir: Path,
    wn3_dir: Path,
    existing_dir: Path,
    extra_dirs: dict[str, Path],
) -> list[PlannedStage]:
    """Build every stage's rows and fits, and raise unless each reuses a matching build.

    Args:
        published_dir: The folder holding the published inputs.
        reused_dir: The AIFS blends folder, whose inputs, saved losses, and stamps are reused.
        wn3_dir: The folder holding `<domain>_wn3_inputs.parquet`.
        existing_dir: The day-1 and day-2 AIFS folder that `fit_aifs.blend_inputs` checks against.
        extra_dirs: The extra-lead folders by `fit_aifs.EXTRA_FOLDERS` short name.

    Returns:
        The stages with at least one fit, `single` stages first within each technology.
    """
    planned: list[PlannedStage] = []
    for domain in fit_aifs.DOMAINS:
        inputs = fit_aifs.blend_inputs(
            aifs_dir=reused_dir, extra_dirs=extra_dirs, domain=domain, existing_dir=existing_dir
        )
        for day in single_days():
            stage = Stage(domain=domain, row_set="single", day=day)
            saved = pl.read_parquet(reused_dir / f"{domain}_single_day{day}_losses.parquet")
            jobs = stage_jobs(stage=stage, saved=saved)
            if not jobs:
                continue
            arms = list(dict.fromkeys(arm for arm, _ in jobs))
            stamp = {
                **fit_aifs.build_stamp(
                    published_dir=published_dir,
                    aifs_dir=reused_dir,
                    domain=domain,
                    arms=arms,
                    extra_dirs=extra_dirs,
                ),
                "reused_dir": reused_dir.name,
            }
            reused_stamp = json.loads(
                (reused_dir / f"{domain}_single_day{day}_losses.json").read_text()
            )
            check_same_build(stamp=stamp, reused_stamp=reused_stamp)
            frame = fit_aifs.aifs_rows(
                published_dir=published_dir,
                aifs=inputs,
                domain=domain,
                row_set="single",
                arms=arms,
                day=day,
                shuffles=control_shuffles(arms=arms),
                nullable=fit_aifs.NULLABLE_PREFIXES,
            )
            check_frame_matches_saved(frame=frame, saved=saved, stage=stage)
            planned.append(PlannedStage(stage, frame, jobs, stamp, saved))
        wn3_inputs = pl.read_parquet(wn3_dir / f"{domain}_wn3_inputs.parquet")
        for day in WN3_DAYS:
            stage = Stage(domain=domain, row_set=fit_aifs.WN3_ROW_SET, day=day)
            jobs = stage_jobs(stage=stage, saved=None)
            arms = list(dict.fromkeys(arm for arm, _ in jobs))
            stamp = fit_aifs.build_stamp(
                published_dir=published_dir,
                aifs_dir=wn3_dir,
                domain=domain,
                arms=arms,
                inputs_name="wn3",
            )
            frame = fit_aifs.aifs_rows(
                published_dir=published_dir,
                aifs=wn3_inputs,
                domain=domain,
                row_set=fit_aifs.WN3_ROW_SET,
                arms=arms,
                day=day,
                shuffles=control_shuffles(arms=arms),
            )
            planned.append(PlannedStage(stage, frame, jobs, stamp, None))
    return planned


def stage_name(*, stage: Stage) -> str:
    """Return a stage's file stem, such as `single_day2`."""
    return f"{stage.row_set}_day{stage.day}"


def stage_losses(*, output_dir: Path, planned: PlannedStage, workers: int) -> pl.DataFrame:
    """Fit one stage and write its outputs once, or check and read a rerun's saved losses.

    Args:
        output_dir: The write-once folder.
        planned: The stage.
        workers: How many (arm, site) fits run at once.

    Returns:
        The stage's new per-row losses.
    """
    stage = planned.stage
    stem = f"{stage.domain}_{stage_name(stage=stage)}"
    losses_file = output_dir / f"{stem}_losses.parquet"
    predictions_file = output_dir / f"{stem}_predictions.parquet"
    stamp_file = losses_file.with_suffix(".json")
    if losses_file.exists():
        losses = pl.read_parquet(losses_file)
        fit_aifs.check_saved_losses(
            losses=losses,
            frame=planned.frame,
            stage=stem,
            arms={arm for arm, setting in planned.jobs if setting == PRIMARY},
            stamp_file=stamp_file,
            stamp=planned.stamp,
        )
    else:
        # A crash after the stamp and before the losses leaves a stamp a rerun rewrites; a crash
        # mid-write leaves only a temporary file, so no losses file ever lacks its stamp.
        refuse_to_overwrite(paths=[predictions_file])
        stamp_file.write_text(json.dumps(planned.stamp))
        if planned.saved is not None:
            losses = fit_single_stage(
                frame=planned.frame, stage=stage, saved=planned.saved, workers=workers
            )
        else:
            losses = fit_aifs.fit_jobs(
                frame=planned.frame, domain=stage.domain, jobs=planned.jobs, workers=workers
            ).with_columns(device=pl.lit(fit_aifs.DEVICE))
        write_atomically(path=losses_file, frame=losses)
    if not predictions_file.exists():
        write_atomically(
            path=predictions_file,
            frame=predictions_from_losses(losses=losses, frame=planned.frame),
        )
    return losses


def refit_lines(*, planned: list[PlannedStage]) -> list[str]:
    """List the AIFS Single (arm, setting) pairs that this run refits."""
    return [
        f"- {item.stage.domain} day {item.stage.day}: `{arm}` at {setting}"
        for item in planned
        if item.saved is not None
        for arm, setting in missing_jobs(saved=item.saved, arms=reused_arms(day=item.stage.day))
    ]


def print_plan(*, planned: list[PlannedStage]) -> None:
    """Print every stage's fits, and the number of (arm, site) fits, without fitting."""
    total = 0
    for item in planned:
        sites = item.frame["site"].n_unique()
        total += len(item.jobs) * sites
        sys.stdout.write(
            f"{item.stage.domain} {stage_name(stage=item.stage)}: {item.frame.height} rows, "
            f"{sites} sites, {len(item.jobs)} (arm, setting) fits: "
            + ", ".join(f"{arm}@{setting}" for arm, setting in item.jobs)
            + "\n"
        )
    sys.stdout.write(f"{total} (arm, site) fits in all\n")


def run_product_blends(
    *, planned: list[PlannedStage], output_dir: Path, reused_dir: Path, workers: int
) -> int:
    """Fit every stage, write its outputs once, and write `README.md` and `report.md`."""
    refuse_to_overwrite(paths=[output_dir / REPORT_NAME])
    output_dir.mkdir(exist_ok=True)
    readme = output_dir / README_NAME
    if not readme.exists():
        readme.write_text(README_TEXT)
    new = {
        (item.stage.domain, item.stage.row_set, item.stage.day): stage_losses(
            output_dir=output_dir, planned=item, workers=workers
        )
        for item in planned
    }
    singles: dict[DomainType, dict[int, pl.DataFrame]] = {}
    wn3: dict[DomainType, dict[int, pl.DataFrame]] = {}
    for domain in fit_aifs.DOMAINS:
        singles[domain] = {}
        for day in single_days():
            saved = pl.read_parquet(reused_dir / f"{domain}_single_day{day}_losses.parquet")
            fitted = new.get((domain, "single", day))
            singles[domain][day] = combine_losses(
                saved=saved, new=fitted, arms=reused_arms(day=day)
            )
        wn3[domain] = {day: new[(domain, fit_aifs.WN3_ROW_SET, day)] for day in WN3_DAYS}
    (output_dir / REPORT_NAME).write_text(
        report_text(singles=singles, wn3=wn3, refits=refit_lines(planned=planned))
    )
    return 0


def run_check(*, planned: list[PlannedStage], reused_dir: Path) -> int:
    """Time one arm twice, and refit a saved AIFS Single control to compare it with the saved rows.

    Args:
        planned: Every stage.
        reused_dir: The AIFS blends folder whose saved losses the refit is compared with.

    Returns:
        0 if both checks pass, else 1, after printing `CHECK PASS` or `CHECK FAIL`.
    """
    first = next(item for item in planned if item.stage.domain == "wind")
    agree, seconds = fit_aifs.time_two_fits(frame=first.frame, arm=first.jobs[0][0], domain="wind")
    arm = blend_arm_name(product="aifs_single", day=1, role="_control")
    stage = next(item for item in planned if item.stage == Stage("wind", "single", 1))
    if stage.saved is None:
        msg = "the wind day-1 `single` stage has no saved losses to compare with"
        raise ValueError(msg)
    reproduces = check_reproduces_saved(
        frame=stage.frame, saved=stage.saved, arm=arm, domain="wind"
    )
    n_fits = sum(len(item.jobs) * item.frame["site"].n_unique() for item in planned)
    passed = agree and reproduces
    sys.stdout.write(
        f"one arm at one site: {seconds:.0f} s; {n_fits} (arm, site) fits are about "
        f"{n_fits * seconds / 3600:.1f} h on one worker\n"
        f"two GPU runs agree: {agree}\n"
        f"`{arm}` refitted at one wind site equals the saved rows in {reused_dir.name}: "
        f"{reproduces}\n"
        f"CHECK {'PASS' if passed else 'FAIL'}\n"
    )
    return 0 if passed else 1


def main() -> int:
    """Fit the product blends, list the fits (`--dry-run`), or time one fit twice (`--check`)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--published-dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=fit_aifs.workers_argument,
        default=1,
        help=f"(arm, site) fits run at once, at most {fit_aifs.MAX_WORKERS}.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List the fits; fit nothing.")
    parser.add_argument("--check", action="store_true", help="Compare two GPU runs of one arm.")
    parser.add_argument(
        "--lookahead-cleared",
        action="store_true",
        help="Confirm that the run log of `build_wn3_inputs.py --read-store` and the page's "
        "lookahead section have been read.",
    )
    args = parser.parse_args()
    studies_dir = args.published_dir.resolve().parent
    reused_dir = studies_dir / fit_aifs.BLENDS_DIR_NAME
    wn3_dir = studies_dir / WN3_DIR_NAME
    existing_dir = studies_dir / fit_aifs.EXISTING_AIFS_DIR_NAME
    extra_dirs = {name: studies_dir / folder for name, folder in fit_aifs.EXTRA_FOLDERS.items()}
    check_output_dir(
        output_dir=args.output_dir,
        read_only=[args.published_dir, reused_dir, wn3_dir, existing_dir, *extra_dirs.values()],
    )
    fit_aifs.require_lookahead_cleared(cleared=args.lookahead_cleared)
    if not args.dry_run:
        fit_aifs.check_gpu_visible()
    planned = plan_stages(
        published_dir=args.published_dir,
        reused_dir=reused_dir,
        wn3_dir=wn3_dir,
        existing_dir=existing_dir,
        extra_dirs=extra_dirs,
    )
    if args.dry_run:
        print_plan(planned=planned)
        return 0
    if args.check:
        return run_check(planned=planned, reused_dir=reused_dir)
    return run_product_blends(
        planned=planned, output_dir=args.output_dir, reused_dir=reused_dir, workers=args.workers
    )


if __name__ == "__main__":
    sys.exit(main())
