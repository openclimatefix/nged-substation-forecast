"""The arms, targets, planned contrasts, and paths that the IFS ladder study's scripts share.

The study is planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1137>.
`ifs_ladder_build_dataset.py`, `ifs_ladder_fit.py`, `ifs_ladder_report.py`, and
`ifs_ladder_charts.py` import this module, so the fit, the report, and the charts cannot disagree
about which arm holds which columns or which contrasts were planned.

**An arm is an XGBoost model given a named set of columns.** The ladder arms `f0` to `f6` are
`studies.ifs_ladder.rung_features`. `fp` is the production reference, `fb` and `f6b` add a second
forecast, and the rest are controls, drop-one-group runs, and ERA5 arms refitted on the IFS rows.
`arm_features` is the one function that says which columns each arm is shown.
"""

from pathlib import Path
from typing import Final, Literal, NamedTuple

from studies.ifs_ladder import (
    GROUP_NAMES,
    RUNG_ADDITIONS,
    RUNGS,
    cloud_cover_control_features,
    cloud_layers_control_features,
    drop_one_group_features,
    era5_comparison_features,
    group_control_features,
    later_groups_columns,
    later_groups_control_features,
    partner_control_features,
    positive_control_features,
    production_features,
    rung_features,
    with_partner_features,
)
from studies.ifs_lead_days import LEAD_DAYS
from studies.sources import IFS_LADDER_INPUTS_DIR, IFS_LADDER_RESULTS_DIR

INPUTS_DIR: Final[Path] = IFS_LADDER_INPUTS_DIR
"""The built frames and their checks."""

RESULTS_DIR: Final[Path] = IFS_LADDER_RESULTS_DIR
"""The out-of-fold losses, the arms' column lists, the report, and the interval tables."""

TargetType = Literal["pv", "cams"]
"""The two targets: the solar farms' output, and CAMS clearness index at the same farms."""

TARGETS: Final[tuple[TargetType, ...]] = ("pv", "cams")
"""The targets, in the order the page presents them."""

TARGET_COLUMNS: Final[dict[TargetType, str]] = {"pv": "power_mw", "cams": "cams_clearness_index"}
"""The column each target's XGBoost models predict."""

ViewType = Literal["ladder", "blend"]
"""Which rows a fit uses: every kept row, or the blend rows that AIFS Single also covers."""

VIEWS: Final[tuple[ViewType, ...]] = ("ladder", "blend")
"""The views, in the order the fit script runs them."""

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss column the study ranks on.

For the output target it is the error as a fraction of the farm's capacity, held to the export cap
that was in force. For the CAMS target the capacity is 1 and no cap applies, so it is the error in
clearness-index units.
"""

SIGNED_ERROR: Final[str] = "signed_error_capped_mw"
"""The loss column holding the prediction minus the measured value, in the target's own unit."""

PERMUTATION_SEED: Final[int] = 20261010
"""The seed of the negative controls' permutations."""

PERMUTATION_GROUPING: Final[tuple[str, ...]] = ("site", "month", "hour_of_day")
"""The columns whose shared values define the rows a permuted value may move between."""

PRIMARY_SETTING: Final[str] = "primary"
"""The name of `PRIMARY_HYPER_PARAMETERS` in the losses."""

SENSITIVITY_SETTING: Final[str] = "sensitivity"
"""The name of `SENSITIVITY_HYPER_PARAMETERS` in the losses."""

PLANNED_LEAD_DAYS: Final[tuple[int, ...]] = (1, 2, 3)
"""The lead days pooled into the planned contrasts, and the only ones that get controls."""

DROP_ONE_LEAD_DAYS: Final[tuple[int, ...]] = PLANNED_LEAD_DAYS
"""The lead days that get the drop-one-group runs."""

BLEND_LEAD_DAYS: Final[tuple[int, ...]] = PLANNED_LEAD_DAYS
"""The lead days that the second forecast's blend rows cover."""

ERA5_COMPARISON_LEAD_DAY: Final[int] = 1
"""The lead day at which ERA5 arms are refitted on the IFS rows."""

ERA5_COMPARISON_RUNGS: Final[tuple[str, ...]] = ("g0", "g1", "g2", "g9")
"""The ERA5 rungs refitted on the IFS rows."""

FAMILY_WISE_ALPHA_PERCENT: Final[float] = 5.0
"""The family-wise level of the five planned contrasts."""

PLANNED_CONTRAST_COUNT: Final[int] = 5
"""How many planned contrasts the family holds."""

ADJUSTED_LEVEL_PERCENT: Final[float] = 100.0 - FAMILY_WISE_ALPHA_PERCENT / PLANNED_CONTRAST_COUNT
"""The coverage of a planned contrast's interval, Bonferroni-adjusted: 99% for five."""

PLANNED_RESAMPLES: Final[int] = 10_000
"""The resamples behind a planned contrast's adjusted interval."""

SMALLEST_EFFECT: Final[dict[TargetType, float]] = {"pv": 0.001, "cams": 0.01}
"""The smallest improvement in the metric worth acting on, fixed before any result.

0.1 percentage points of capacity on the output target (as a fraction), and 0.01 on the CAMS
clearness index.
"""

ERA5_P4_RESULTS_NAME: Final[str] = "contrasts_main_through_g9.parquet"
"""The ERA5 study's contrast table, which holds its planned contrast P4."""

ERA5_LOSSES_NAME: Final[str] = "losses_main_through_g9_pv_ladder.parquet"
"""The ERA5 study's output-target losses, which the width check cuts to the IFS months."""

ERA5_DATASET_NAME: Final[str] = "dataset_main_through_g9.parquet"
"""The ERA5 study's built frame, which supplies the rows."""

F6_ARM: Final[str] = "f6"
"""The arm given every IFS variable."""
PRODUCTION_ARM: Final[str] = "fp"
"""The production reference: `f0` and the four variables the production ensemble feed carries."""
BLEND_ARM: Final[str] = "fb"
"""The minimal IFS set plus AIFS Single's radiation and temperature."""
BLEND_FULL_ARM: Final[str] = "f6b"
"""Every IFS variable plus AIFS Single's radiation and temperature."""
CONTROL_CLOUD_COVER_ARM: Final[str] = "control_f0_cloud_cover"
"""The arm `f0` plus a permuted copy of total cloud cover, which pads contrast P1."""
CONTROL_CLOUD_LAYERS_ARM: Final[str] = "control_f1_layers"
"""The arm `f1` plus permuted copies of the three cloud layers, which pads contrast P2."""
CONTROL_LATER_GROUPS_ARM: Final[str] = "control_f2_later_groups"
"""The arm `f2` plus permuted copies of every `f3` to `f6` variable, which pads contrast P3."""
CONTROL_PARTNER_MINIMAL_ARM: Final[str] = "control_f0_partner"
"""The arm `f0` plus two permuted partner columns, which pads contrast P4."""
CONTROL_PARTNER_FULL_ARM: Final[str] = "control_f6_partner"
"""The arm `f6` plus two permuted partner columns, which pads contrast P5."""
CONTROL_DIRECT_ARM: Final[str] = "control_f2_direct"
"""The arm `f2` plus a permuted copy of direct radiation, which pads the step to `f3`."""
CONTROL_HUMIDITY_ARM: Final[str] = "control_f2_humidity"
"""The arm `f2` plus permuted copies of the `f4` variables, which pads the step to `f4`."""
CONTROL_CONVECTION_ARM: Final[str] = "control_f2_convection"
"""The arm `f2` plus permuted copies of the `f5` variables, which pads the step to `f5`."""
POSITIVE_CONTROL_ARM: Final[str] = "positive_control"
"""The output-target arm given `f2` and CAMS global irradiance."""
DROP_PREFIX: Final[str] = "drop_"
"""The prefix of a drop-one-group arm's name, followed by the dropped rung."""
ERA5_PREFIX: Final[str] = "era5_"
"""The prefix of an ERA5 arm's name, followed by the ERA5 rung."""

CONTROL_ARMS: Final[tuple[str, ...]] = (
    CONTROL_CLOUD_COVER_ARM,
    CONTROL_CLOUD_LAYERS_ARM,
    CONTROL_LATER_GROUPS_ARM,
    CONTROL_PARTNER_MINIMAL_ARM,
    CONTROL_PARTNER_FULL_ARM,
    CONTROL_DIRECT_ARM,
    CONTROL_HUMIDITY_ARM,
    CONTROL_CONVECTION_ARM,
    POSITIVE_CONTROL_ARM,
)
"""Every control arm."""

GROUP_CONTROL_ARMS: Final[dict[str, str]] = {
    "f3": CONTROL_DIRECT_ARM,
    "f4": CONTROL_HUMIDITY_ARM,
    "f5": CONTROL_CONVECTION_ARM,
}
"""The control of each exploratory rung `f3` to `f5`, by the rung it pads."""


class PlannedContrast(NamedTuple):
    """One planned contrast: a treatment arm minus a reference arm, and the control that pads it.

    Attributes:
        label: The contrast's name on the page, `P1` to `P5`.
        treatment: The arm whose error is compared.
        reference: The arm it is compared against.
        control: The negative control with the treatment's columns and the new information
            removed. The contrast stands only if the treatment also beats the control.
        view: The rows the contrast is read on.
        question: What the contrast asks, in words.
    """

    label: str
    treatment: str
    reference: str
    control: str
    view: ViewType
    question: str


PLANNED_CONTRASTS: Final[tuple[PlannedContrast, ...]] = (
    PlannedContrast(
        label="P1",
        treatment="f1",
        reference="f0",
        control=CONTROL_CLOUD_COVER_ARM,
        view="ladder",
        question="Does total cloud help beyond IFS's radiation and temperature?",
    ),
    PlannedContrast(
        label="P2",
        treatment="f2",
        reference="f1",
        control=CONTROL_CLOUD_LAYERS_ARM,
        view="ladder",
        question="Do the cloud layers help beyond total cloud?",
    ),
    PlannedContrast(
        label="P3",
        treatment=F6_ARM,
        reference="f2",
        control=CONTROL_LATER_GROUPS_ARM,
        view="ladder",
        question="Does any other IFS variable help beyond the cloud layers?",
    ),
    PlannedContrast(
        label="P4",
        treatment=BLEND_ARM,
        reference="f0",
        control=CONTROL_PARTNER_MINIMAL_ARM,
        view="blend",
        question="What does a second forecast add to the minimal IFS set?",
    ),
    PlannedContrast(
        label="P5",
        treatment=BLEND_FULL_ARM,
        reference=F6_ARM,
        control=CONTROL_PARTNER_FULL_ARM,
        view="blend",
        question="What does a second forecast add once every IFS variable is present?",
    ),
)
"""The planned contrasts as data, written before any result. A negative difference is a gain."""

SENSITIVITY_ARMS: Final[tuple[str, ...]] = (
    "f0",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    F6_ARM,
    BLEND_ARM,
    BLEND_FULL_ARM,
    *CONTROL_ARMS,
)
"""The arms also fitted at the second hyperparameter setting at the planned lead days.

They are the arms of the planned contrasts, every control, and `f3` to `f5`, which the priority
list's exploratory rule needs at both settings.
"""

CAMS_ARMS: Final[tuple[str, ...]] = ("f0", "f2", F6_ARM)
"""The only arms fitted on the exploratory CAMS target."""

ARM_LABELS: Final[dict[str, str]] = {
    "f0": "F0 minimal (radiation, temperature)",
    "f1": "F1 + total cloud",
    "f2": "F2 + cloud layers",
    "f3": "F3 + direct radiation",
    "f4": "F4 + humidity, column water, boundary layer",
    "f5": "F5 + convection, visibility",
    "f6": "F6 + surface and weather state (all 20)",
    PRODUCTION_ARM: "FP production reference",
    BLEND_ARM: "FB F0 + AIFS Single",
    BLEND_FULL_ARM: "F6B F6 + AIFS Single",
    CONTROL_CLOUD_COVER_ARM: "Control: F0 + shuffled total cloud",
    CONTROL_CLOUD_LAYERS_ARM: "Control: F1 + shuffled layers",
    CONTROL_LATER_GROUPS_ARM: "Control: F2 + shuffled F3 to F6",
    CONTROL_PARTNER_MINIMAL_ARM: "Control: F0 + shuffled AIFS Single",
    CONTROL_PARTNER_FULL_ARM: "Control: F6 + shuffled AIFS Single",
    CONTROL_DIRECT_ARM: "Control: F2 + shuffled direct radiation",
    CONTROL_HUMIDITY_ARM: "Control: F2 + shuffled F4 variables",
    CONTROL_CONVECTION_ARM: "Control: F2 + shuffled F5 variables",
    POSITIVE_CONTROL_ARM: "Positive control: F2 + CAMS",
    **{f"{DROP_PREFIX}{rung}": f"F6 without {GROUP_NAMES[rung]}" for rung in RUNGS[1:]},
    **{f"{ERA5_PREFIX}{rung}": f"ERA5 {rung.upper()}" for rung in ERA5_COMPARISON_RUNGS},
}
"""The words each arm carries on a chart and in the report."""


IFS_FIGURE_NUMBERS: Final[dict[str, int]] = {
    "headline": 1,
    "weather": 2,
    "lead_day_skill": 3,
    "models_work": 4,
    "controls": 5,
    "leaderboard": 6,
    "era5_against_ifs": 7,
    "regimes": 8,
    "drop_one": 9,
    "variables_or_forecast": 10,
}
"""The page's figure numbers by figure name, in the order of the plan's page structure."""


class FitKey(NamedTuple):
    """What names one fit's files: the view, the lead day, and the target.

    Attributes:
        view: Which rows were used.
        lead_day: The lead day of the forecast the arms were shown.
        target: The target the models predicted.
    """

    view: ViewType
    lead_day: int
    target: TargetType


def lead_day_dataset_path(*, lead_day: int) -> Path:
    """Return where the frame of one lead day's rows is written."""
    return INPUTS_DIR / f"dataset_lead_day_{lead_day}.parquet"


def blend_dataset_path() -> Path:
    """Return where the blend rows of every blend lead day are written, one `lead_day` column."""
    return INPUTS_DIR / "dataset_blend.parquet"


def checks_path() -> Path:
    """Return where the build's checks are written."""
    return INPUTS_DIR / "checks.md"


def dataset_path_for(*, key: FitKey) -> Path:
    """Return the frame a fit reads."""
    if key.view == "blend":
        return blend_dataset_path()
    return lead_day_dataset_path(lead_day=key.lead_day)


def results_path(*, key: FitKey) -> Path:
    """Return where one fit's per-row losses are written."""
    return RESULTS_DIR / f"losses_{key.view}_lead_day_{key.lead_day}_{key.target}.parquet"


def checkpoint_dir_for(*, key: FitKey) -> Path:
    """Return the directory where one fit's groups of jobs checkpoint their losses."""
    return results_path(key=key).with_suffix(".parts")


def arms_path(*, key: FitKey) -> Path:
    """Return where the JSON naming each arm's columns, the device, and the settings is written."""
    return RESULTS_DIR / f"arms_{key.view}_lead_day_{key.lead_day}_{key.target}.json"


class ReportPaths(NamedTuple):
    """Where the report script writes its outputs.

    Attributes:
        report: The markdown report.
        leaderboard: Each arm's error by lead day, target, and row set, with intervals.
        contrasts: Every contrast with its intervals.
        splits: The regime, season, and hour-of-day splits.
        farms: Each farm's mean absolute error by lead day and arm.
        verdicts: The planned contrasts' verdicts at each setting, combined, and after the control.
        priority_list: Each group of variables with its evidence class and gain.
        decisions: The second and third decisions' outcomes.
    """

    report: Path
    leaderboard: Path
    contrasts: Path
    splits: Path
    farms: Path
    verdicts: Path
    priority_list: Path
    decisions: Path


def width_preview_path() -> Path:
    """Return where the width preview, which needs no IFS fit, is written."""
    return RESULTS_DIR / "width_preview.md"


def report_paths() -> ReportPaths:
    """Return the report's output paths."""
    return ReportPaths(
        report=RESULTS_DIR / "report.md",
        leaderboard=RESULTS_DIR / "leaderboard.parquet",
        contrasts=RESULTS_DIR / "contrasts.parquet",
        splits=RESULTS_DIR / "splits.parquet",
        farms=RESULTS_DIR / "farms.parquet",
        verdicts=RESULTS_DIR / "planned_verdicts.parquet",
        priority_list=RESULTS_DIR / "priority_list.parquet",
        decisions=RESULTS_DIR / "decisions.parquet",
    )


def permutation_groups(*, view: ViewType) -> list[tuple[str, ...]]:
    """Return the column groups the controls permute, each group moving together.

    Args:
        view: The rows of the fit.

    Returns:
        For the ladder view, total cloud, the three layers, and the `f3` to `f6` variables, as three
        groups with three permutations. For the blend view, the partner's two variables.
    """
    if view == "blend":
        return [("partner_shortwave_radiation", "partner_temperature_2m")]
    return [RUNG_ADDITIONS["f1"], RUNG_ADDITIONS["f2"], later_groups_columns()]


def _ladder_pv_arms(*, lead_day: int) -> dict[str, tuple[str, ...]]:
    """Return the output-target arms of the ladder view at one lead day."""
    arms: dict[str, tuple[str, ...]] = {rung: rung_features(rung=rung) for rung in RUNGS}
    arms[PRODUCTION_ARM] = production_features()
    if lead_day in PLANNED_LEAD_DAYS:
        arms[CONTROL_CLOUD_COVER_ARM] = cloud_cover_control_features()
        arms[CONTROL_CLOUD_LAYERS_ARM] = cloud_layers_control_features()
        arms[CONTROL_LATER_GROUPS_ARM] = later_groups_control_features()
        for rung, control in GROUP_CONTROL_ARMS.items():
            arms[control] = group_control_features(rung=rung)  # ty: ignore[invalid-argument-type]
        arms[POSITIVE_CONTROL_ARM] = positive_control_features()
    if lead_day in DROP_ONE_LEAD_DAYS:
        for rung in RUNGS[1:]:
            arms[f"{DROP_PREFIX}{rung}"] = drop_one_group_features(dropped=rung)
    if lead_day == ERA5_COMPARISON_LEAD_DAY:
        for era5_rung in ERA5_COMPARISON_RUNGS:
            arms[f"{ERA5_PREFIX}{era5_rung}"] = era5_comparison_features(
                rung=era5_rung  # ty: ignore[invalid-argument-type]
            )
    return arms


def arm_features(
    *, view: ViewType, target: TargetType, lead_day: int
) -> dict[str, tuple[str, ...]]:
    """Return every arm of one fit and the columns it is shown.

    Args:
        view: The rows of the fit.
        target: The target the arms predict.
        lead_day: The lead day of the forecast the arms are shown.

    Returns:
        The arms from the smallest to the largest, then the controls and the other arms. The
        blend view holds only the output target, and the CAMS target holds only `CAMS_ARMS`.

    Raises:
        ValueError: If the blend view is asked of the CAMS target or of a lead day without blend
            rows.
    """
    if view == "blend":
        if target != "pv" or lead_day not in BLEND_LEAD_DAYS:
            msg = f"the blend view holds the pv target at lead days {BLEND_LEAD_DAYS} only"
            raise ValueError(msg)
        return {
            "f0": rung_features(rung="f0"),
            BLEND_ARM: with_partner_features(rung="f0"),
            F6_ARM: rung_features(rung="f6"),
            BLEND_FULL_ARM: with_partner_features(rung="f6"),
            CONTROL_PARTNER_MINIMAL_ARM: partner_control_features(rung="f0"),
            CONTROL_PARTNER_FULL_ARM: partner_control_features(rung="f6"),
        }
    arms = _ladder_pv_arms(lead_day=lead_day)
    if target == "cams":
        return {name: arms[name] for name in CAMS_ARMS}
    return arms


def settings_for(
    *, view: ViewType, arm: str, lead_day: int, extra: tuple[str, ...] = ()
) -> list[str]:
    """Return the hyperparameter settings an arm is fitted at.

    Args:
        view: The rows of the fit.
        arm: The arm.
        lead_day: The lead day.
        extra: Further arms to fit at both settings.

    Returns:
        The primary setting, and the second setting as well for the arms of `SENSITIVITY_ARMS`
        and `extra` at the planned lead days (and in the blend view).
    """
    both = (arm in SENSITIVITY_ARMS or arm in extra) and (
        view == "blend" or lead_day in PLANNED_LEAD_DAYS
    )
    return [PRIMARY_SETTING, SENSITIVITY_SETTING] if both else [PRIMARY_SETTING]


def fit_keys() -> list[FitKey]:
    """List every fit: each lead day's ladder view on both targets, and the blend view's output."""
    keys = [FitKey(view="ladder", lead_day=day, target="pv") for day in LEAD_DAYS]
    keys += [FitKey(view="ladder", lead_day=day, target="cams") for day in LEAD_DAYS]
    keys += [FitKey(view="blend", lead_day=day, target="pv") for day in BLEND_LEAD_DAYS]
    return keys
