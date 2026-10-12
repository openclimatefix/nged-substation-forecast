"""The arms of the quick-wins study: what each is shown, how it is trained, and how it is cleaned.

An arm is a frozen description. `fit_arms.py` turns it into 31 per-series boosters, predictions, and
scores. Arms differ from production in one named respect each (a column list, a cleaning rule, a
training tweak), so a contrast between two arms isolates that respect. The columns are chosen by a
series' type: the generator types are `PV` and `Wind`, and every other type is a demand-like series.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import quick_wins_inputs as inputs
from cleaning_rules import CleaningRule
from extra_features import WEEKLY_ANALOGUE_COLUMNS

GENERATOR_TYPES: Final[frozenset[str]] = frozenset({"PV", "Wind"})
"""Series types driven by weather alone, for which power lags and weekday columns carry little."""

PRODUCTION_COLUMNS: Final[tuple[str, ...]] = tuple(
    sorted(inputs.production_config().selected_features)
)
"""The 24 columns the production forecast is shown."""

LAG_COLUMNS: Final[frozenset[str]] = frozenset(
    name for name in PRODUCTION_COLUMNS if name.startswith("power_lag_")
)
"""The four power-lag columns."""

WEEKDAY_COLUMNS: Final[frozenset[str]] = frozenset(
    {"local_day_of_week", "local_day_of_week_sin", "local_day_of_week_cos"}
)
"""The three day-of-week columns, which mean nothing to a wind or solar farm."""

PV_PHYSICS_COLUMNS: Final[tuple[str, ...]] = (
    "cos_zenith",
    "clear_sky_index",
    "clear_sky_ghi_window_mean",
)
WIND_PHYSICS_COLUMNS: Final[tuple[str, ...]] = (
    "wind_u_100m",
    "wind_v_100m",
    "wind_power_proxy",
)
DEMAND_PHYSICS_COLUMNS: Final[tuple[str, ...]] = (
    "heating_degrees_c",
    "temperature_2m_rolling_mean_24h",
    "temperature_2m_rolling_mean_72h",
)
CALENDAR_COLUMNS: Final[tuple[str, ...]] = (
    "is_bank_holiday",
    "days_to_nearest_bank_holiday",
    "half_hour_of_day",
    "day_of_year",
)

ColumnPolicy = Callable[[str], tuple[str, ...]]
"""Maps a series type, such as `"PV"`, to the feature columns an arm shows that series."""


def production_columns(series_type: str) -> tuple[str, ...]:
    """Show every series the 24 production columns."""
    return PRODUCTION_COLUMNS


def generators_without_lags(series_type: str) -> tuple[str, ...]:
    """Drop the power lags and weekday columns for PV and wind; leave other series as production."""
    if series_type in GENERATOR_TYPES:
        return tuple(c for c in PRODUCTION_COLUMNS if c not in LAG_COLUMNS | WEEKDAY_COLUMNS)
    return PRODUCTION_COLUMNS


def per_type_physics(series_type: str) -> tuple[str, ...]:
    """`generators_without_lags` plus the physics columns that suit each type."""
    base = generators_without_lags(series_type)
    if series_type == "PV":
        return (*base, *PV_PHYSICS_COLUMNS)
    if series_type == "Wind":
        return (*base, *WIND_PHYSICS_COLUMNS)
    return (*base, *DEMAND_PHYSICS_COLUMNS)


PV_IRRELEVANT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "wind_direction_100m",
        "wind_direction_10m",
        "wind_speed_100m",
        "wind_speed_10m",
        "windchill",
        "geopotential_height_500hpa",
        "pressure_reduced_to_mean_sea_level",
        "pressure_surface",
    }
)
"""Production columns with no direct physical route to solar output."""

WIND_IRRELEVANT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "downward_short_wave_radiation_flux_surface",
        "dew_point_temperature_2m",
        "categorical_precipitation_type_surface",
        "local_time_of_year_sin",
        "local_time_of_year_cos",
    }
)
"""Production columns with no direct physical route to wind output."""


def lean_generators(series_type: str) -> tuple[str, ...]:
    """Keep the lags but cut the weekday columns and the weather columns a generator ignores."""
    if series_type == "PV":
        drop = WEEKDAY_COLUMNS | PV_IRRELEVANT_COLUMNS
    elif series_type == "Wind":
        drop = WEEKDAY_COLUMNS | WIND_IRRELEVANT_COLUMNS
    else:
        return PRODUCTION_COLUMNS
    return tuple(c for c in PRODUCTION_COLUMNS if c not in drop)


def lean_wind_only(series_type: str) -> tuple[str, ...]:
    """`lean_generators` for wind farms; every other series, PV included, keeps production's."""
    return lean_generators(series_type) if series_type == "Wind" else PRODUCTION_COLUMNS


def with_weekly_analogues(series_type: str) -> tuple[str, ...]:
    """Show every series the production columns and the manual heuristic's inputs."""
    return (*PRODUCTION_COLUMNS, *WEEKLY_ANALOGUE_COLUMNS)


def with_lead_time(series_type: str) -> tuple[str, ...]:
    """Show every series the production columns and the forecast lead time."""
    return (*PRODUCTION_COLUMNS, "nwp_lead_time_hours")


def with_calendar(series_type: str) -> tuple[str, ...]:
    """Add holiday and raw ordinal time columns for the demand-like series."""
    if series_type in GENERATOR_TYPES:
        return PRODUCTION_COLUMNS
    return (*PRODUCTION_COLUMNS, *CALENDAR_COLUMNS)


@dataclass(frozen=True)
class Arm:
    """One configuration of the forecast, fitted as one booster per series.

    Attributes:
        name: A short identifier used in file names and on the leaderboard.
        family: The group the arm belongs to: `reference`, `cleaning`, `features`, `training`, or
            `combination`.
        description: One line saying how the arm differs from production, for the page.
        columns: The column policy.
        cleaning: The training-data cleaning rule, or `"none"` for production's.
        extra_cleaning: Further cleaning rules applied together with `cleaning`.
        objective: The XGBoost objective.
        recency_half_life_days: Training rows lose half their weight every this many days of age,
            counted back from the end of the training window; `None` weights rows equally.
        early_stopping: Choose the number of trees on the last six training weeks, then refit on
            every training row with that number.
        colsample_bytree: Column subsampling per tree. Production uses 0.8; every contrast uses 1.0.
        min_child_weight: XGBoost's `min_child_weight`; production uses 1.
        pooled_by_type: Whether one booster per series type is fitted on every series of that type,
            with each series' power divided by its training capacity, instead of one per series.
        pooled_site_id: Whether a pooled booster is also shown the series identifier as a number.
        quantile_levels: Quantile levels of a `reg:quantileerror` booster. The forecast is every
            ensemble member crossed with every level, a sample of the predictive distribution.
        train_last_months: Train only on the last this many months of the training window.
    """

    name: str
    family: str
    description: str
    columns: ColumnPolicy = production_columns
    cleaning: CleaningRule = "none"
    extra_cleaning: tuple[CleaningRule, ...] = ()
    objective: str = "reg:squarederror"
    recency_half_life_days: float | None = None
    early_stopping: bool = False
    colsample_bytree: float = 1.0
    min_child_weight: int = 1
    pooled_by_type: bool = False
    pooled_site_id: bool = False
    quantile_levels: tuple[float, ...] | None = None
    train_last_months: int | None = None


def needs_weekly_analogues(arm: Arm) -> bool:
    """Whether any series type's columns include the weekly analogue columns."""
    return any(
        column in WEEKLY_ANALOGUE_COLUMNS
        for series_type in ("PV", "Wind", "Disaggregated Demand", "Raw Flow", "BESS", "Biofuel")
        for column in arm.columns(series_type)
    )


QUANTILE_LEVELS: Final[tuple[float, ...]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
"""The quantile levels the quantile arms fit, so the pinball losses sum to a coarse CRPS."""

CHOSEN_CLEANING: Final[CleaningRule] = "new_meter_settling"
"""The cleaning rule round 0 chose, which every later arm applies (see the report's round 0)."""

REFERENCE_ARMS: Final[tuple[Arm, ...]] = (
    Arm(
        name="production_colsample0.8",
        family="reference",
        description="Production as stored: 24 columns, colsample_bytree 0.8.",
        colsample_bytree=0.8,
    ),
    Arm(
        name="production",
        family="reference",
        description="Production with colsample_bytree 1.0, the control for round 0.",
    ),
    Arm(
        name="production_cleaned",
        family="reference",
        description="Production with colsample_bytree 1.0 and the chosen cleaning: the control for "
        "every round 1 and later contrast.",
        cleaning=CHOSEN_CLEANING,
    ),
)

CLEANING_ARMS: Final[tuple[Arm, ...]] = (
    *(
        Arm(
            name=f"clean_first_{n}_months",
            family="cleaning",
            description=f"Drop the first {n} month(s) after each series' first non-zero reading.",
            cleaning=rule,
        )
        for n, rule in (
            (1, "first_1_month"),
            (2, "first_2_months"),
            (3, "first_3_months"),
            (6, "first_6_months"),
        )
    ),
    Arm(
        name="clean_isolated_zeros",
        family="cleaning",
        description="Drop exact zeros whose neighbours are clearly non-zero.",
        cleaning="isolated_zeros",
    ),
    Arm(
        name="clean_spikes",
        family="cleaning",
        description="Drop single readings that jump away from agreeing neighbours.",
        cleaning="spikes",
    ),
    Arm(
        name="clean_commissioning_ramp",
        family="cleaning",
        description="Drop the days before a solar or wind farm reaches its peers' relative level.",
        cleaning="commissioning_ramp",
    ),
    Arm(
        name="clean_new_meter_settling",
        family="cleaning",
        description="Drop the 6 weeks after a meter first reads live (post hoc rule).",
        cleaning="new_meter_settling",
    ),
    Arm(
        name="clean_pv_dropouts",
        family="cleaning",
        description="Drop PV and wind half-hours near zero while peers produce (post hoc).",
        cleaning="pv_dropouts",
    ),
    Arm(
        name="clean_switching_days",
        family="cleaning",
        description="Drop runs of days a demand series steps away from its peers (post hoc).",
        cleaning="switching_days",
    ),
    Arm(
        name="clean_combined",
        family="cleaning",
        description="Meter settling, commissioning ramp, PV dropouts, and switching days together.",
        cleaning="new_meter_settling",
        extra_cleaning=("commissioning_ramp", "pv_dropouts", "switching_days"),
    ),
)

FEATURE_ARMS: Final[tuple[Arm, ...]] = (
    Arm(
        name="a1_generators_without_lags",
        cleaning=CHOSEN_CLEANING,
        family="features",
        description="PV and wind lose the power lags and weekday columns.",
        columns=generators_without_lags,
    ),
    Arm(
        name="a2_per_type_physics",
        cleaning=CHOSEN_CLEANING,
        family="features",
        description="A1 plus solar columns for PV, wind components for wind, heating for demand.",
        columns=per_type_physics,
    ),
    Arm(
        name="a3_lean_generators",
        cleaning=CHOSEN_CLEANING,
        family="features",
        description="PV and wind keep their lags but lose weekday and irrelevant weather columns.",
        columns=lean_generators,
    ),
    Arm(
        name="a3w_lean_wind_only",
        cleaning=CHOSEN_CLEANING,
        family="features",
        description="Only wind farms lose the weekday and irrelevant weather columns (post hoc).",
        columns=lean_wind_only,
    ),
    Arm(
        name="c1_lead_time",
        cleaning=CHOSEN_CLEANING,
        family="features",
        description="Production columns plus the forecast lead time.",
        columns=with_lead_time,
    ),
    Arm(
        name="c7_calendar",
        cleaning=CHOSEN_CLEANING,
        family="features",
        description="Demand-like series gain holiday and raw ordinal time columns.",
        columns=with_calendar,
    ),
)

TRAINING_ARMS: Final[tuple[Arm, ...]] = (
    Arm(
        name="d1_min_child_weight_20",
        cleaning=CHOSEN_CLEANING,
        family="training",
        description="min_child_weight 20, as in the earlier studies' boosters.",
        min_child_weight=20,
    ),
    Arm(
        name="d2_pooled_by_type",
        cleaning=CHOSEN_CLEANING,
        family="training",
        description="One booster per series type on capacity-normalised power (lag study design).",
        pooled_by_type=True,
    ),
    Arm(
        name="d3_pooled_with_site_id",
        cleaning=CHOSEN_CLEANING,
        family="training",
        description="d2 with the series identifier as a column, so a booster can tell sites apart.",
        pooled_by_type=True,
        pooled_site_id=True,
    ),
    Arm(
        name="g1_heuristic_inputs",
        cleaning=CHOSEN_CLEANING,
        family="features",
        description="Production columns plus the manual heuristic's inputs: 6 weekly analogues "
        "and their mean, median, minimum, and maximum.",
        columns=with_weekly_analogues,
    ),
    Arm(
        name="e1_quantile",
        cleaning=CHOSEN_CLEANING,
        family="training",
        description="Nine quantile levels with the pinball loss, a proxy for training on CRPS.",
        objective="reg:quantileerror",
        quantile_levels=QUANTILE_LEVELS,
    ),
    *(
        Arm(
            name=f"f1_train_last_{months}_months",
            cleaning=CHOSEN_CLEANING,
            family="training",
            description=f"Train on only the last {months} months of the training window.",
            train_last_months=months,
        )
        for months in (6, 9, 12)
    ),
    Arm(
        name="c2_early_stopping",
        cleaning=CHOSEN_CLEANING,
        family="training",
        description="Number of trees chosen on the last six training weeks.",
        early_stopping=True,
    ),
    Arm(
        name="c3_recency_weights",
        cleaning=CHOSEN_CLEANING,
        family="training",
        description="Training rows lose half their weight every 120 days of age.",
        recency_half_life_days=120.0,
    ),
    Arm(
        name="c4_absolute_error",
        cleaning=CHOSEN_CLEANING,
        family="training",
        description="Absolute-error objective instead of squared error.",
        objective="reg:absoluteerror",
    ),
)

ALL_ARMS: Final[dict[str, Arm]] = {
    arm.name: arm for arm in (*REFERENCE_ARMS, *CLEANING_ARMS, *FEATURE_ARMS, *TRAINING_ARMS)
}
"""Every arm defined so far, by name. Round 2 and 3 combinations are added to this when they are
chosen, by `combine`."""


def combine(*, name: str, parts: tuple[Arm, ...], description: str) -> Arm:
    """Build an arm that merges the differences from production of several arms.

    Each part's difference from production is the set of columns it removes and the set it adds,
    for each series type. A combination removes every column any part removes and adds every column
    any part adds, so a later part that merely keeps a column cannot undo an earlier removal. The
    cleaning rules of all parts apply together. Each other setting takes the value of the last part
    that changes it from production's.

    Args:
        name: The combination's name.
        parts: Arms to merge.
        description: One line for the page.

    Returns:
        The combined arm.
    """

    def columns(series_type: str) -> tuple[str, ...]:
        removed: set[str] = set()
        added: list[str] = []
        for part in parts:
            offered = part.columns(series_type)
            removed |= set(PRODUCTION_COLUMNS) - set(offered)
            added += [c for c in offered if c not in PRODUCTION_COLUMNS and c not in added]
        kept = [c for c in PRODUCTION_COLUMNS if c not in removed]
        return (*kept, *added)

    cleaning_rules = [rule for part in parts for rule in (part.cleaning, *part.extra_cleaning)]
    distinct_rules = tuple(dict.fromkeys(rule for rule in cleaning_rules if rule != "none"))
    objectives = [p.objective for p in parts if p.objective != "reg:squarederror"]
    return Arm(
        name=name,
        family="combination",
        description=description,
        columns=columns,
        cleaning=distinct_rules[0] if distinct_rules else "none",
        extra_cleaning=distinct_rules[1:],
        objective=objectives[-1] if objectives else "reg:squarederror",
        recency_half_life_days=next(
            (p.recency_half_life_days for p in reversed(parts) if p.recency_half_life_days), None
        ),
        early_stopping=any(p.early_stopping for p in parts),
        min_child_weight=max(p.min_child_weight for p in parts),
        pooled_by_type=any(p.pooled_by_type for p in parts),
        pooled_site_id=any(p.pooled_site_id for p in parts),
        quantile_levels=next(
            (p.quantile_levels for p in reversed(parts) if p.quantile_levels), None
        ),
        train_last_months=next(
            (p.train_last_months for p in reversed(parts) if p.train_last_months), None
        ),
    )


_A3W = next(a for a in FEATURE_ARMS if a.name == "a3w_lean_wind_only")
_D3 = next(a for a in TRAINING_ARMS if a.name == "d3_pooled_with_site_id")
_E1 = next(a for a in TRAINING_ARMS if a.name == "e1_quantile")
_G1 = next(a for a in TRAINING_ARMS if a.name == "g1_heuristic_inputs")
_C1 = next(a for a in FEATURE_ARMS if a.name == "c1_lead_time")
_C2 = next(a for a in TRAINING_ARMS if a.name == "c2_early_stopping")
_C4 = next(a for a in TRAINING_ARMS if a.name == "c4_absolute_error")

ROUND2_ARMS: Final[tuple[Arm, ...]] = (
    combine(
        name="r2_e1_c1_a3w_d3_g1",
        parts=(_C1, _A3W, _D3, _E1, _G1),
        description="The quantile combination plus the manual heuristic's inputs.",
    ),
    combine(
        name="r2_e1_c1_a3w_d3",
        parts=(_C1, _A3W, _D3, _E1),
        description="Quantile training with the lead-time column, lean wind columns, and pooling.",
    ),
    combine(
        name="r2_c4_c2_c1_a3w_d3",
        parts=(_C4, _C2, _C1, _A3W, _D3),
        description="The best ladder step plus one pooled booster per type with a site identifier.",
    ),
    combine(
        name="r2_c4_c1_a3w_d3",
        parts=(_C4, _C1, _A3W, _D3),
        description="As the previous arm, without early stopping.",
    ),
    combine(
        name="r2_c4_c2_c1_a3w",
        parts=(_C4, _C2, _C1, _A3W),
        description="The best ladder step plus lean wind-farm columns (post hoc).",
    ),
    combine(
        name="r2_c4_c2",
        parts=(_C4, _C2),
        description="Absolute-error objective with early stopping.",
    ),
    combine(
        name="r2_c4_c1",
        parts=(_C4, _C1),
        description="Absolute-error objective with the lead-time column.",
    ),
    combine(
        name="r2_c4_c2_c1",
        parts=(_C4, _C2, _C1),
        description="Absolute-error objective, early stopping, and the lead-time column.",
    ),
)
"""Round 2: the round 1 arms that beat `production_cleaned` with intervals clear of zero, in a
ladder. Chosen by judgement after reading the round 1 table (post hoc)."""

ALL_ARMS.update({arm.name: arm for arm in ROUND2_ARMS})
