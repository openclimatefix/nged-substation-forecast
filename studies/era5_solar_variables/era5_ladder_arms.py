"""The arms, targets, planned contrasts, and paths that the ERA5 ladder study's scripts share.

The study is planned in <https://github.com/openclimatefix/nged-substation-forecast/pull/1097>.
`era5_ladder_build_dataset.py`, `era5_ladder_fit.py`, `era5_ladder_report.py`, and
`era5_ladder_charts.py` import this module, so the fit, the report, and the charts cannot disagree
about which arm holds which columns or which contrasts were planned.

**An arm is an XGBoost model given a named set of columns.** The ladder arms `g0` to `g9` are
`studies.era5_ladder.rung_features`. The other arms are the controls, the drop-one-group runs, and
the aerosol rung. `arm_features` is the one function that says which columns each arm is shown.
"""

from pathlib import Path
from typing import Final, Literal, NamedTuple

from studies.blending import PERMUTED_SUFFIX
from studies.era5_ladder import (
    AEROSOL_COLUMNS,
    RUNGS,
    SHARED_FEATURES,
    RungType,
    drop_one_group_features,
    negative_control_features,
    rung_features,
)
from studies.sources import study_dir_for

STUDY_DIR: Final[Path] = study_dir_for(study="era5_solar_variables")
"""Where the study's datasets and results live, under `data/studies/per_study/`."""

INPUTS_DIR: Final[Path] = STUDY_DIR / "inputs"
"""The built frames and their checks."""

RESULTS_DIR: Final[Path] = STUDY_DIR / "results"
"""The out-of-fold losses, the arms' column lists, the report, and the interval tables."""

TargetType = Literal["pv", "cams"]
"""The two targets: the solar farms' output, and CAMS clearness index at the same farms."""

TARGETS: Final[tuple[TargetType, ...]] = ("pv", "cams")
"""The targets, in the order the page presents them."""

TARGET_COLUMNS: Final[dict[TargetType, str]] = {"pv": "power_mw", "cams": "cams_clearness_index"}
"""The column each target's XGBoost models predict."""

ViewType = Literal["ladder", "aerosol_rows"]
"""Which rows a fit used: every kept row, or only the rows that EAC4 aerosol covers."""

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss column the study ranks on.

For the output target it is the error as a fraction of the farm's capacity, held to the export cap
that was in force. For the CAMS target the capacity is 1 and no cap applies, so it is the error in
clearness-index units.
"""

SIGNED_ERROR: Final[str] = "signed_error_capped_mw"
"""The loss column holding the prediction minus the measured value, in the target's own unit."""

NEGATIVE_CONTROL_SEED: Final[int] = 20261008
"""The seed of the negative control's permutation."""

PERMUTATION_GROUPING: Final[tuple[str, ...]] = ("site", "month", "hour_of_day")
"""The columns whose shared values define the rows a permuted value may move between."""

SENSITIVITY_ARMS: Final[tuple[str, ...]] = ("g0", "g1", "g2", "g9")
"""The arms also fitted at the second hyperparameter setting: the arms of the planned contrasts."""

PLANNED_CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("P0", "g9", "g0"),
    ("P1", "g1", "g0"),
    ("P2", "g2", "g0"),
    ("P3", "g9", "g2"),
)
"""The planned contrasts as (label, treatment arm, reference arm), written before any result.

P0 is every ERA5 variable against the minimal set, P1 adds total cloud, P2 adds the three cloud
layers, and P3 asks whether anything beyond the cloud layers helps. Each is run on both targets.
"""

FAMILY_WISE_ALPHA_PERCENT: Final[float] = 5.0
"""The family-wise level of the eight planned contrasts (four contrasts on two targets)."""

PLANNED_CONTRAST_COUNT: Final[int] = len(PLANNED_CONTRASTS) * len(TARGETS)
"""How many planned contrasts the family holds."""

ADJUSTED_LEVEL_PERCENT: Final[float] = 100.0 - FAMILY_WISE_ALPHA_PERCENT / PLANNED_CONTRAST_COUNT
"""The coverage of a planned contrast's interval, Bonferroni-adjusted: 99.375% for eight."""

SMALLEST_EFFECT: Final[dict[TargetType, float]] = {"pv": 0.001, "cams": 0.01}
"""The smallest improvement in the metric worth acting on, fixed before any result.

0.1 percentage points of capacity on the output target (as a fraction), and 0.01 on the CAMS
clearness index.
"""

NEAR_LINE_SHARE: Final[float] = 0.2
"""A result is near the 5% line if a bound of its 95% interval lies within this share of the
interval's width from zero."""

AEROSOL_RUNG: Final[str] = "g10"
"""The arm that adds EAC4 aerosol optical depth to every ERA5 variable."""

AEROSOL_REFERENCE: Final[str] = "g9_aerosol_rows"
"""The arm of every ERA5 variable refitted on exactly the rows that EAC4 aerosol covers."""

KNOWN_ANSWER_ARM: Final[str] = "known_answer_ssrd_only"
"""The CAMS-target arm shown only `ssrd` and the solar geometry."""

NEGATIVE_CONTROL_ARM: Final[str] = "negative_control"
"""The arm given `g2` and permuted copies of every later column."""

POSITIVE_CONTROL_ARM: Final[str] = "positive_control"
"""The output-target arm given `g2` and CAMS global irradiance."""

DROP_PREFIX: Final[str] = "drop_"
"""The prefix of a drop-one-group arm's name, followed by the dropped rung."""

ARM_LABELS: Final[dict[str, str]] = {
    "g0": "G0 minimal: ssrd, t2m",
    "g1": "G1 + total cloud",
    "g2": "G2 + low, medium, high cloud",
    "g3": "G3 + clear-sky irradiance",
    "g4": "G4 + cloud water and base",
    "g5": "G5 + direct beam",
    "g6": "G6 + wind and thermal radiation",
    "g7": "G7 + humidity and haze",
    "g8": "G8 + snow and albedo",
    "g9": "G9 + every other ERA5 variable",
    "g10": "G10 + CAMS aerosol (not ERA5)",
    "g9_aerosol_rows": "G9 on the aerosol rows",
    NEGATIVE_CONTROL_ARM: "Negative control: G2 + permuted G3 to G9",
    POSITIVE_CONTROL_ARM: "Positive control: G2 + CAMS irradiance",
    KNOWN_ANSWER_ARM: "ssrd and sun position only",
    **{f"{DROP_PREFIX}{rung}": f"G9 without {rung.upper()}'s variables" for rung in RUNGS[1:]},
}
"""The words each arm carries on a chart and in the report."""

PRIMARY_SETTING: Final[str] = "primary"
"""The name of `PRIMARY_HYPER_PARAMETERS` in the losses."""

SENSITIVITY_SETTING: Final[str] = "sensitivity"
"""The name of `SENSITIVITY_HYPER_PARAMETERS` in the losses."""


class FitKey(NamedTuple):
    """What names one fit's files: the variant, the rungs in the frame, the target, and the rows.

    Attributes:
        variant: `main` or `snow_zero_hours`.
        through_rung: The highest rung the fit's frame holds.
        target: The target the models predicted.
        view: Which rows were used.
    """

    variant: str
    through_rung: RungType
    target: TargetType
    view: ViewType


def dataset_path(*, through_rung: RungType, variant: str) -> Path:
    """Return where the frame for one rung and variant is written.

    Args:
        through_rung: The highest rung whose variables the frame holds.
        variant: `main` or `snow_zero_hours`.

    Returns:
        The parquet path.
    """
    return INPUTS_DIR / f"dataset_{variant}_through_{through_rung}.parquet"


def checks_path(*, through_rung: RungType, variant: str) -> Path:
    """Return where the build's checks are written, beside its frame."""
    return INPUTS_DIR / f"checks_{variant}_through_{through_rung}.md"


def results_path(*, key: FitKey) -> Path:
    """Return where one fit's per-row losses are written."""
    stem = f"{key.variant}_through_{key.through_rung}_{key.target}_{key.view}"
    return RESULTS_DIR / f"losses_{stem}.parquet"


def arms_path(*, key: FitKey) -> Path:
    """Return where the JSON naming each arm's columns, the device, and the settings is written."""
    stem = f"{key.variant}_through_{key.through_rung}_{key.target}_{key.view}"
    return RESULTS_DIR / f"arms_{stem}.json"


def arm_features(*, target: TargetType, through_rung: RungType) -> dict[str, tuple[str, ...]]:
    """Return every arm of the ladder view and the columns it is shown.

    The negative control and the drop-one-group arms are defined relative to `g9`, so they are
    present only when the frame holds every rung. The positive control and the known-answer arm
    are each defined for one target only.

    Args:
        target: The target the arms predict.
        through_rung: The highest rung the frame holds.

    Returns:
        The arms from the smallest to the largest, then the controls and the drop-one-group arms.
    """
    arms: dict[str, tuple[str, ...]] = {
        rung: rung_features(rung=rung) for rung in RUNGS[: RUNGS.index(through_rung) + 1]
    }
    if through_rung == RUNGS[-1]:
        arms[NEGATIVE_CONTROL_ARM] = negative_control_features(suffix=PERMUTED_SUFFIX)
        for rung in RUNGS[1:]:
            arms[f"{DROP_PREFIX}{rung}"] = drop_one_group_features(dropped=rung)
    if target == "pv":
        arms[POSITIVE_CONTROL_ARM] = (*rung_features(rung="g2"), "cams_ghi_w_m2")
    else:
        arms[KNOWN_ANSWER_ARM] = (*SHARED_FEATURES, "ssrd")
    return arms


def aerosol_arm_features() -> dict[str, tuple[str, ...]]:
    """Return the two arms of the aerosol view: every ERA5 variable, with and without aerosol.

    Returns:
        `g9_aerosol_rows` (the `g9` columns) and `g10` (the `g9` columns and the aerosol columns).
    """
    base = rung_features(rung="g9")
    return {AEROSOL_REFERENCE: base, AEROSOL_RUNG: (*base, *AEROSOL_COLUMNS)}
