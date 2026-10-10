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
    without_mars_only_features,
)
from studies.sources import ERA5_LADDER_INPUTS_DIR, ERA5_LADDER_RESULTS_DIR

INPUTS_DIR: Final[Path] = ERA5_LADDER_INPUTS_DIR
"""The built frames and their checks."""

RESULTS_DIR: Final[Path] = ERA5_LADDER_RESULTS_DIR
"""The out-of-fold losses, the arms' column lists, the report, and the interval tables."""

TargetType = Literal["pv", "cams"]
"""The two targets: the solar farms' output, and CAMS clearness index at the same farms."""

TARGETS: Final[tuple[TargetType, ...]] = ("pv", "cams")
"""The targets, in the order the page presents them."""

TARGET_COLUMNS: Final[dict[TargetType, str]] = {"pv": "power_mw", "cams": "cams_clearness_index"}
"""The column each target's XGBoost models predict."""

ViewType = Literal["ladder", "ladder_extra", "aerosol_rows"]
"""Which fit a losses file holds.

`ladder` is every arm on every kept row, `ladder_extra` is further arms fitted at the second
hyperparameter setting only (arms near the 5% line), and `aerosol_rows` is the aerosol rung and its
reference on the rows that EAC4 aerosol covers.
"""

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

SENSITIVITY_ARMS: Final[tuple[str, ...]] = ("g0", "g1", "g2", "g9", "g9_without_mars_only")
"""The arms also fitted at the second hyperparameter setting: the arms of the planned contrasts."""

PLANNED_CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("P0", "g9", "g0"),
    ("P1", "g1", "g0"),
    ("P2", "g2", "g0"),
    ("P3", "g9", "g2"),
    ("P4", "g9", "g9_without_mars_only"),
)
"""The planned contrasts as (label, treatment arm, reference arm), written before any result.

P0 is every ERA5 variable against the minimal set, P1 adds total cloud, P2 adds the three cloud
layers, and P3 asks whether anything beyond the cloud layers helps. P4 asks whether the 12 MARS-only
variables help beyond everything a production forecast can already get. Each is run on both
targets, and P4 is judged on the output target.
"""

FAMILY_WISE_ALPHA_PERCENT: Final[float] = 5.0
"""The family-wise level of the ten planned contrasts (five contrasts on two targets)."""

PLANNED_CONTRAST_COUNT: Final[int] = len(PLANNED_CONTRASTS) * len(TARGETS)
"""How many planned contrasts the family holds."""

ADJUSTED_LEVEL_PERCENT: Final[float] = 100.0 - FAMILY_WISE_ALPHA_PERCENT / PLANNED_CONTRAST_COUNT
"""The coverage of a planned contrast's interval, Bonferroni-adjusted: 99.5% for ten."""

SMALLEST_EFFECT: Final[dict[TargetType, float]] = {"pv": 0.001, "cams": 0.01}
"""The smallest improvement in the metric worth acting on, fixed before any result.

0.1 percentage points of capacity on the output target (as a fraction), and 0.01 on the CAMS
clearness index.
"""

PLANNED_RESAMPLES: Final[int] = 10_000
"""The resamples behind a planned contrast's adjusted interval: each 0.25% tail holds about 25."""

NEAR_LINE_SHARE: Final[float] = 0.2
"""A result is near the 5% line if a bound of its 95% interval lies within this share of the
interval's width from zero."""

AEROSOL_RUNG: Final[str] = "g10"
"""The arm that adds EAC4 aerosol optical depth to every ERA5 variable."""

AEROSOL_REFERENCE: Final[str] = "g9_aerosol_rows"
"""The arm of every ERA5 variable refitted on exactly the rows that EAC4 aerosol covers."""

KNOWN_ANSWER_ARM: Final[str] = "known_answer_ssrd_only"
"""The CAMS-target arm shown only `ssrd` and the solar geometry."""

MARS_FREE_ARM: Final[str] = "g9_without_mars_only"
"""The arm of every ERA5 variable except the 12 found only in MARS."""

NEGATIVE_CONTROL_ARM: Final[str] = "negative_control"
"""The arm given `g2` and permuted copies of every later column."""

POSITIVE_CONTROL_ARM: Final[str] = "positive_control"
"""The output-target arm given `g2` and CAMS global irradiance."""

DROP_PREFIX: Final[str] = "drop_"
"""The prefix of a drop-one-group arm's name, followed by the dropped rung."""

QUANTILE_ARMS: Final[tuple[str, ...]] = (
    "g0",
    "g2",
    "g9",
    MARS_FREE_ARM,
    NEGATIVE_CONTROL_ARM,
    AEROSOL_REFERENCE,
    AEROSOL_RUNG,
)
"""The arms that also get a quantile fit, at the primary setting only.

The point fit ranks every arm. The quantile fit asks the second question, whether an input helps
the model say how uncertain it is, and it costs about nine times a point fit, so it covers only
the arms that question needs.
"""

ARM_LABELS: Final[dict[str, str]] = {
    "g0": "G0 minimal (radiation, temperature)",
    "g1": "G1 + total cloud",
    "g2": "G2 + cloud layers",
    "g3": "G3 + clear-sky irradiance",
    "g4": "G4 + cloud water, base",
    "g5": "G5 + direct beam",
    "g6": "G6 + wind, thermal radiation",
    "g7": "G7 + humidity, haze",
    "g8": "G8 + snow, albedo",
    "g9": "G9 + all other ERA5",
    "g10": "G10 + CAMS aerosol",
    MARS_FREE_ARM: "G9 without the MARS-only variables",
    "g9_aerosol_rows": "G9 on aerosol rows",
    NEGATIVE_CONTROL_ARM: "Negative control",
    POSITIVE_CONTROL_ARM: "Positive control (+ CAMS)",
    KNOWN_ANSWER_ARM: "Solar radiation and sun position only",
    **{
        f"{DROP_PREFIX}{rung}": f"G9 without the variables {rung.upper()} adds"
        for rung in RUNGS[1:]
    },
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


def report_path(*, name: str, variant: str, through_rung: RungType) -> Path:
    """Return where one of the report's outputs is written, keyed by variant and rung.

    Args:
        name: The output's name and extension, such as `report.md` or `contrasts.parquet`.
        variant: `main` or `snow_zero_hours`.
        through_rung: The highest rung the report covers.

    Returns:
        The path, so that a report on a partial build, the full build, and the snow variant never
        overwrite or feed one another.
    """
    stem, _, extension = name.rpartition(".")
    return RESULTS_DIR / f"{stem}_{variant}_through_{through_rung}.{extension}"


class ReportPaths(NamedTuple):
    """Where the report script writes its outputs for one variant and rung.

    Attributes:
        report: The markdown report.
        leaderboard: Each arm's error and correlation.
        contrasts: Every contrast with its intervals.
        splits: The regime, season, farm, and hour-of-day splits.
        worst_days: The minimal arm's worst farm-days.
        aerosol_conditions: The aerosol rung against its reference inside each aerosol condition.
        probabilistic: The quantile fits' scores, contrasts, reliability, and regime scores.
    """

    report: Path
    leaderboard: Path
    contrasts: Path
    splits: Path
    worst_days: Path
    aerosol_conditions: Path
    probabilistic: Path


def report_paths(*, variant: str, through_rung: RungType) -> ReportPaths:
    """Return the report's output paths, keyed by variant and rung."""
    return ReportPaths(
        report=report_path(name="report.md", variant=variant, through_rung=through_rung),
        leaderboard=report_path(
            name="leaderboard.parquet", variant=variant, through_rung=through_rung
        ),
        contrasts=report_path(name="contrasts.parquet", variant=variant, through_rung=through_rung),
        splits=report_path(name="splits.parquet", variant=variant, through_rung=through_rung),
        worst_days=report_path(
            name="worst_days.parquet", variant=variant, through_rung=through_rung
        ),
        aerosol_conditions=report_path(
            name="aerosol_conditions.parquet", variant=variant, through_rung=through_rung
        ),
        probabilistic=report_path(
            name="probabilistic.parquet", variant=variant, through_rung=through_rung
        ),
    )


def importance_path(*, variant: str, through_rung: RungType) -> Path:
    """Return where `era5_ladder_importance.py` writes each column's share of gain."""
    return RESULTS_DIR / f"importance_{variant}_through_{through_rung}.parquet"


def results_path(*, key: FitKey) -> Path:
    """Return where one fit's per-row losses are written."""
    stem = f"{key.variant}_through_{key.through_rung}_{key.target}_{key.view}"
    return RESULTS_DIR / f"losses_{stem}.parquet"


def checkpoint_dir_for(*, key: FitKey) -> Path:
    """Return the directory where one fit's groups of jobs checkpoint their losses."""
    return results_path(key=key).with_suffix(".parts")


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
        arms[MARS_FREE_ARM] = without_mars_only_features()
        arms[NEGATIVE_CONTROL_ARM] = negative_control_features(suffix=PERMUTED_SUFFIX)
        for rung in RUNGS[1:]:
            arms[f"{DROP_PREFIX}{rung}"] = drop_one_group_features(dropped=rung)
    if target == "pv":
        if RUNGS.index(through_rung) >= RUNGS.index("g2"):
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
