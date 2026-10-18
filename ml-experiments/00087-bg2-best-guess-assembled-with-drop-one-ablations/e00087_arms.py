"""The arm registry of experiment 00087: names, plain-words labels, staircases, and ablations.

`e00087_assemble.py` writes the arms, `e00087_score.py` scores them, and `e00087_curtailment.py`
prices them. All three read the names from here so that a rename happens in one place.

An arm is one table of quantile forecasts, `work/arms/<name>.parquet`, that covers one or more
series groups (`Primary`, `BSP`, `GSP`, `solar`, `wind`). The four target classes of the page are
`Primary` (16 series), `BSP/GSP` (2 + 2), `Solar farms` (6), and `Wind farms` (3). An alias is a
name that stands for another arm's table (for example ablation A1 is the same forecast as staircase
step S5), so no file is written twice.
"""

from dataclasses import dataclass
from typing import Final, Literal

ArmRoleType = Literal["step", "ablation", "reference", "addon"]
"""`step` is a rung of a class's staircase, `ablation` a drop-one ablation of the best guess,
`reference` a baseline, and `addon` an add-one arm that is not in the best guess."""

CLASSES: Final[tuple[str, ...]] = ("Primary", "BSP/GSP", "Solar farms", "Wind farms")
GROUPS_OF_CLASS: Final[dict[str, tuple[str, ...]]] = {
    "Primary": ("Primary",),
    "BSP/GSP": ("BSP", "GSP"),
    "Solar farms": ("solar",),
    "Wind farms": ("wind",),
}
CLASS_OF_GROUP: Final[dict[str, str]] = {
    group: name for name, groups in GROUPS_OF_CLASS.items() for group in groups
}


@dataclass(frozen=True)
class ArmSpec:
    """One arm.

    Attributes:
        name: The file and prediction-set name.
        target_class: The class the arm belongs to, or `all` for a reference that covers every
            series.
        role: What the arm is for.
        label: A plain-words name for charts and the page.
        alias_of: The arm whose table this name stands for, or `None`.
    """

    name: str
    target_class: str
    role: ArmRoleType
    label: str
    alias_of: str | None = None


def _arm(
    name: str, target_class: str, role: ArmRoleType, label: str, alias_of: str | None = None
) -> ArmSpec:
    return ArmSpec(name=name, target_class=target_class, role=role, label=label, alias_of=alias_of)


ARMS: Final[tuple[ArmSpec, ...]] = (
    # Primary staircase S1 to S6, then the drop-one ablations A1 to A7.
    _arm("p_s1_frozen", "Primary", "step", "S1 Frozen model"),
    _arm("p_s2_stack", "Primary", "step", "S2 Frozen model with 14-day correction (stack)"),
    _arm("p_s3_monthly", "Primary", "step", "S3 Refitted monthly"),
    _arm("p_s4_weekly", "Primary", "step", "S4 Refitted weekly"),
    _arm("p_s5_blend_no_offset", "Primary", "step", "S5 Blend of five models"),
    _arm("p_s6_bg2", "Primary", "step", "S6 Blend plus solar offset (best guess)"),
    _arm(
        "p_a1_no_offset",
        "Primary",
        "ablation",
        "A1 Without the solar offset",
        alias_of="p_s5_blend_no_offset",
    ),
    _arm("p_a2_no_blend", "Primary", "ablation", "A2 Without the blend (weekly model plus offset)"),
    _arm("p_a3_monthly_in_weekly_slot", "Primary", "ablation", "A3 Monthly instead of weekly"),
    _arm("p_a4_correction_all", "Primary", "ablation", "A4 14-day correction on every model"),
    _arm("p_a5_spread_off", "Primary", "ablation", "A5 Without the spread factor 1.15"),
    _arm("p_a6_capacity_held", "Primary", "ablation", "A6 Capacity held at its 2024/25 value"),
    _arm("p_a7_three_members", "Primary", "ablation", "A7 Three-model blend (exploratory)"),
    _arm(
        "p_noise_repeat_weekly",
        "Primary",
        "addon",
        "Repeat fit of the weekly model (noise floor)",
    ),
    # BSP and GSP.
    _arm("b_frozen", "BSP/GSP", "step", "Frozen model"),
    _arm("b_stack", "BSP/GSP", "step", "Frozen model with 14-day correction (previous best)"),
    _arm("b_monthly", "BSP/GSP", "step", "Refitted monthly"),
    _arm("b_weekly", "BSP/GSP", "step", "Refitted weekly (best guess)"),
    _arm(
        "b_a_weekly_to_monthly",
        "BSP/GSP",
        "ablation",
        "Monthly instead of weekly",
        alias_of="b_monthly",
    ),
    _arm("b_a_correction", "BSP/GSP", "ablation", "Weekly with the 14-day correction added back"),
    # Solar and wind farms (the same generator arms serve both classes).
    _arm("g_prev_best_era5", "generators", "step", "Weather-trained model (previous best)"),
    _arm("g_ens_frozen", "generators", "step", "Forecast-trained model, frozen"),
    _arm(
        "g_ens_weekly", "generators", "step", "Forecast-trained model, refitted weekly (best guess)"
    ),
    _arm(
        "g_a_weekly_to_frozen",
        "generators",
        "ablation",
        "Frozen instead of weekly",
        alias_of="g_ens_frozen",
    ),
    _arm(
        "g_a_era5_swap",
        "generators",
        "ablation",
        "Weather-trained model instead of forecast-trained",
        alias_of="g_prev_best_era5",
    ),
    _arm("g_a_spread_on", "generators", "ablation", "With a fitted spread factor"),
    _arm("g_a_wind_margin", "Wind farms", "addon", "Plus power-curve margin"),
    # References.
    _arm("ref_climatology", "all", "reference", "Climatology"),
    _arm("ref_manual_heuristic", "all", "reference", "NGED manual heuristic"),
    _arm("ref_xgboost_baseline", "all", "reference", "Production-style XGBoost model"),
    _arm("ref_perfect", "all", "reference", "Perfect forecast"),
)
BY_NAME: Final[dict[str, ArmSpec]] = {arm.name: arm for arm in ARMS}

BG2_OF_CLASS: Final[dict[str, str]] = {
    "Primary": "p_s6_bg2",
    "BSP/GSP": "b_weekly",
    "Solar farms": "g_ens_weekly",
    "Wind farms": "g_ens_weekly",
}
PREVIOUS_BEST_OF_CLASS: Final[dict[str, str]] = {
    "Primary": "p_s2_stack",
    "BSP/GSP": "b_stack",
    "Solar farms": "g_prev_best_era5",
    "Wind farms": "g_prev_best_era5",
}
NON_INFERIORITY_MARGIN: Final[dict[str, float]] = {"Solar farms": 0.10, "Wind farms": 0.30}
"""Points of quantile score by which BG2 may exceed the weather-trained model and still pass."""
VERDICT_COUNT: Final[int] = 4
VERDICT_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / VERDICT_COUNT)
"""98.75 percent: 5 percent divided by the four verdict contrasts (Bonferroni)."""

STAIRCASE: Final[dict[str, tuple[str, ...]]] = {
    "Primary": (
        "p_s1_frozen",
        "p_s2_stack",
        "p_s3_monthly",
        "p_s4_weekly",
        "p_s5_blend_no_offset",
        "p_s6_bg2",
    ),
    "BSP/GSP": ("b_frozen", "b_stack", "b_monthly", "b_weekly"),
    "Solar farms": ("g_prev_best_era5", "g_ens_frozen", "g_ens_weekly"),
    "Wind farms": ("g_prev_best_era5", "g_ens_frozen", "g_ens_weekly", "g_a_wind_margin"),
}
"""Each class's rungs in order; a step's contrast is against the rung before it."""
ABLATIONS: Final[dict[str, tuple[str, ...]]] = {
    "Primary": (
        "p_a1_no_offset",
        "p_a2_no_blend",
        "p_a3_monthly_in_weekly_slot",
        "p_a4_correction_all",
        "p_a5_spread_off",
        "p_a6_capacity_held",
        "p_a7_three_members",
    ),
    "BSP/GSP": ("b_a_weekly_to_monthly", "b_a_correction"),
    "Solar farms": ("g_a_weekly_to_frozen", "g_a_era5_swap", "g_a_spread_on"),
    "Wind farms": ("g_a_weekly_to_frozen", "g_a_era5_swap", "g_a_spread_on", "g_a_wind_margin"),
}
REFERENCES: Final[tuple[str, ...]] = (
    "ref_climatology",
    "ref_manual_heuristic",
    "ref_xgboost_baseline",
)
"""The three references scored beside the best guess; `ref_perfect` scores zero by definition."""
NOT_RUN_STEPS: Final[tuple[tuple[str, str], ...]] = (
    ("Zero start (experiment 80)", "tested and not adopted: no gain"),
    ("Switching masks (experiment 75)", "tested and not adopted: no gain over a random mask"),
)
"""Rows the staircase draws as flat steps with a note; they are never run here."""


def resolve(*, name: str) -> str:
    """Return the file name behind an arm name, following an alias."""
    alias = BY_NAME[name].alias_of
    return name if alias is None else alias


def arm_label(*, name: str) -> str:
    """Return an arm's plain-words label."""
    return BY_NAME[name].label
