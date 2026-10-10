"""Score the ERA5 variable ladder from its saved losses, and write the report and the tables.

One-off throwaway script for the study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1097>. It reads the per-row losses
that `era5_ladder_fit.py` saved and refits nothing. It writes `report.md` and up to six
tables (`leaderboard`, `contrasts`, `splits`, `worst_days`, `aerosol_conditions`, and
`probabilistic`, each a parquet file) into
`data/studies/per_study/era5_solar_variables/results/`, each name ending in the variant and the
highest rung, so a report on a partial build, the full build, and the snow variant never overwrite
one another. The chart script reads the tables. Every number on the page comes from `report.md`, and
the split tables it prints are the same numbers as `splits.parquet`.

**Planned contrasts are those the plan named before any result existed**: P0 (every ERA5 variable
against the minimal set), P1 (total cloud), P2 (the three cloud layers), P3 (anything beyond the
cloud layers), and P4 (the 12 MARS-only variables, against everything else), each on both targets.
Each is reported at the Bonferroni-adjusted level for the family of ten, at 95%, and at the second
hyperparameter setting, and its verdict stands only if both settings agree. Every other contrast is
exploratory.

**The adjusted intervals of the planned contrasts rest on 10,000 resamples**, so each tail of a
99.5% interval is set by about 25 of them. The report's `MARS fetch decision` section applies the
plan's rule to P4 on the output target at both settings.

**The correlation** is the Pearson correlation of the out-of-fold prediction with the measured
value, over all rows, as a fraction of each farm's capacity, with an interval from the same
month-and-seed resampling. It is exploratory.

**A farm's dates are not printed.** The worst days of the minimal arm are listed with the farm's
anonymous label and the month and year, never the day of the month.

Run it with `uv run python studies/era5_solar_variables/era5_ladder_report.py`.
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from itertools import pairwise
from typing import Final

import numpy as np
import polars as pl
from era5_ladder_arms import (
    ADJUSTED_LEVEL_PERCENT,
    AEROSOL_REFERENCE,
    AEROSOL_RUNG,
    ARM_LABELS,
    DROP_PREFIX,
    MARS_FREE_ARM,
    METRIC,
    NEAR_LINE_SHARE,
    NEGATIVE_CONTROL_ARM,
    PLANNED_CONTRASTS,
    PLANNED_RESAMPLES,
    POSITIVE_CONTROL_ARM,
    PRIMARY_SETTING,
    QUANTILE_ARMS,
    SENSITIVITY_SETTING,
    SHUFFLED_ARM,
    SHUFFLED_SUFFIX,
    SIGNED_ERROR,
    SMALLEST_EFFECT,
    TARGET_COLUMNS,
    TARGETS,
    FitKey,
    TargetType,
    ViewType,
    arms_path,
    checks_path,
    dataset_path,
    importance_path,
    report_paths,
    results_path,
)
from studies.bootstrap import (
    MIN_MONTHS_FOR_INTERVAL,
    N_BOOTSTRAP_RESAMPLES,
    BootstrapInterval,
    bootstrap_absolute,
    bootstrap_difference,
    bootstrap_difference_at_level,
)
from studies.correlation import pooled_correlation_interval
from studies.cross_validation import (
    N_FOLDS,
    PRIMARY_HYPER_PARAMETERS,
    QUANTILE_LEVELS,
    SEEDS,
    SENSITIVITY_HYPER_PARAMETERS,
)
from studies.era5_ladder import (
    AEROSOL_COLUMNS,
    AEROSOL_CONDITIONS,
    CLEAR_SKY_INDEX_THRESHOLDS,
    MARS_ONLY_VARIABLES,
    MIN_AEROSOL_DAYS,
    MIN_AEROSOL_MONTHS,
    MIN_EXTRATERRESTRIAL_W_M2,
    RUNGS,
    RungType,
    aerosol_condition_flags,
    aerosol_trial_recommendation,
    mars_fetch_recommendation,
    raise_unless_same_rows,
    season_of_month,
    sky_regime,
    sky_regime_from_cloud_cover,
)
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("era5_ladder_report")

PERCENTAGE_POINTS: Final[float] = 100.0
"""Turns a fraction of capacity into percentage points."""

PRINT_DECIMALS: Final[int] = 3
"""How many decimals a table prints, so a chart and the page quote the same digits."""

MIN_HOURS_FOR_WORST_DAY: Final[int] = 6
"""A farm-day needs this many kept hours to be listed, so one stray hour is not a day."""

WORST_DAY_COUNT: Final[int] = 20
"""How many of the minimal arm's worst farm-days the report lists."""

SPLIT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (("g1", "g0"), ("g2", "g0"), ("g9", "g0"))
"""The (treatment, reference) contrasts the regime and season splits show."""

HOUR_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("g2", "g0"),
    ("g4", "g3"),
    ("g9", "g0"),
)
"""The (treatment, reference) contrasts the hour-of-day split shows, one interval per UTC hour."""

HOUR_BY_SEASON_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (("g4", "g3"),)
"""The contrasts split by UTC hour within each season, which separate the hour from the season."""

PROBABILISTIC_METRICS: Final[dict[str, str]] = {
    "crps": "crps_floored_fraction_of_capacity",
    "coverage_80": "covered_80",
    "width_80": "width_80_fraction_of_capacity",
}
"""The probabilistic scores the report differences, by the name it prints, and their loss column."""

PROBABILISTIC_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("g2", "g0"),
    ("g9", "g2"),
    ("g9", NEGATIVE_CONTROL_ARM),
    ("g9", MARS_FREE_ARM),
    (AEROSOL_RUNG, AEROSOL_REFERENCE),
)
"""The (treatment, reference) pairs scored on the probabilistic metrics."""

NO_GAIN_BEYOND_EFFECT: Final[str] = "rules out a gain larger than the smallest effect"
GAIN_NOT_EXCLUDED: Final[str] = "does not rule out a gain larger than the smallest effect"
IMPROVES_BEYOND_EFFECT: Final[str] = "improves by more than the smallest effect"
IMPROVES: Final[str] = "improves; whether the gain exceeds the smallest effect is unresolved"
IMPROVES_BY_LESS_THAN_EFFECT: Final[str] = "improves, by less than the smallest effect"
WORSENS: Final[str] = "worsens"
UNRESOLVED: Final[str] = "unresolved"

RULES_OUT_LARGE_GAIN: Final[frozenset[str]] = frozenset(
    {NO_GAIN_BEYOND_EFFECT, IMPROVES_BY_LESS_THAN_EFFECT, WORSENS}
)
"""The verdicts that each rule out a gain larger than the smallest effect."""

LARGE_GAIN_POSSIBLE: Final[frozenset[str]] = frozenset(
    {IMPROVES_BEYOND_EFFECT, IMPROVES, GAIN_NOT_EXCLUDED}
)
"""The verdicts that each leave a gain larger than the smallest effect possible."""


def combine_planned(*, primary: str, sensitivity: str) -> str:
    """Combine a planned contrast's verdicts at the two settings into one.

    The plan's question is whether a gain larger than the smallest effect is ruled out, so two
    settings that both rule it out agree even where one says `improves, by less than the smallest
    effect` and the other says `rules out a gain larger than the smallest effect`.

    Args:
        primary: The verdict at the primary setting.
        sensitivity: The verdict at the second setting.

    Returns:
        The shared verdict if the two are equal. `improves; whether the gain exceeds the smallest
        effect is unresolved` if one setting shows a gain beyond the smallest effect and the other
        leaves that unresolved. Otherwise `rules out a gain larger than the
        smallest effect` if both rule it out, `does not rule out a gain larger than the smallest
        effect` if both leave it possible, and `unresolved` if they disagree on that.
    """
    if primary == sensitivity:
        return primary
    pair = {primary, sensitivity}
    if pair <= RULES_OUT_LARGE_GAIN:
        return NO_GAIN_BEYOND_EFFECT
    if pair == {IMPROVES_BEYOND_EFFECT, IMPROVES}:
        return IMPROVES
    if pair <= LARGE_GAIN_POSSIBLE:
        return GAIN_NOT_EXCLUDED
    return UNRESOLVED


def contrast_verdict(*, lower: float, upper: float, smallest_effect: float) -> str:
    """Return the verdict on an error difference (treatment minus reference) from its interval.

    A negative difference is a lower error, so an interval wholly below zero is an improvement.

    Args:
        lower: The interval's lower bound.
        upper: The interval's upper bound.
        smallest_effect: The smallest improvement worth acting on, in the same unit.

    Returns:
        `improves by more than the smallest effect` if the whole interval lies below minus the
        smallest effect. `improves, by less than the smallest effect` if the whole interval lies
        between minus the smallest effect and zero. `improves; whether the gain exceeds the
        smallest effect is unresolved` if the interval lies below zero and straddles minus the
        smallest effect. `worsens` if it is above zero. Otherwise `rules out a gain larger than the
        smallest effect` if the lower bound is above minus the smallest effect, and `does not rule
        out a gain larger than the smallest effect` if it is not.
    """
    if upper < -smallest_effect:
        return IMPROVES_BEYOND_EFFECT
    if upper < 0.0:
        return IMPROVES_BY_LESS_THAN_EFFECT if lower > -smallest_effect else IMPROVES
    if lower > 0.0:
        return WORSENS
    if lower > -smallest_effect:
        return NO_GAIN_BEYOND_EFFECT
    return GAIN_NOT_EXCLUDED


def is_near_line(*, lower: float, upper: float) -> bool:
    """Return whether a 95% interval has a bound within `NEAR_LINE_SHARE` of its width from zero."""
    width = upper - lower
    return min(abs(lower), abs(upper)) <= NEAR_LINE_SHARE * width


def read_losses(*, key: FitKey) -> pl.DataFrame:
    """Read one fit's per-row losses, and the extra sensitivity-setting fit's if it exists.

    Args:
        key: Which fit. The extra fit shares its variant, rung, and target, and holds arms at the
            sensitivity setting that the main fit did not.

    Returns:
        The per-row losses.

    Raises:
        ValueError: If the two files hold the same arm at the same setting.
    """
    losses = pl.read_parquet(results_path(key=key))
    if key.view != "ladder":
        return losses
    extra_path = results_path(key=key._replace(view="ladder_extra"))
    if not extra_path.exists():
        return losses
    extra = pl.read_parquet(extra_path)
    pairs = (
        extra.select("arm", "setting")
        .unique()
        .join(losses.select("arm", "setting").unique(), on=["arm", "setting"])
    )
    if not pairs.is_empty():
        msg = f"the extra fit repeats arms already fitted: {pairs.to_dicts()}"
        raise ValueError(msg)
    return pl.concat([losses, extra])


def with_watts(*, losses: pl.DataFrame, dataset: pl.DataFrame) -> pl.DataFrame:
    """Add the CAMS target's absolute error in W m⁻², which is the clearness-index error times TOA.

    Args:
        losses: The CAMS target's losses.
        dataset: The kept rows, carrying `cams_toa_w_m2`.

    Returns:
        `losses` with `absolute_error_w_m2`.
    """
    return losses.join(
        dataset.select("site", "time", "cams_toa_w_m2"), on=["site", "time"]
    ).with_columns(absolute_error_w_m2=pl.col(METRIC) * pl.col("cams_toa_w_m2"))


def with_split_columns(*, losses: pl.DataFrame, dataset: pl.DataFrame) -> pl.DataFrame:
    """Add the regime, season, and hour-of-day columns the splits group by.

    Args:
        losses: Per-row losses.
        dataset: The kept rows with `cams_clear_sky_index`, `hour_of_day`, `time`, and (from `g1`)
            `tcc`.

    Returns:
        `losses` with `regime_cams`, `regime_era5` (null if `tcc` is absent), `season`, and
        `hour_of_day`.
    """
    columns = ["site", "time", "cams_clear_sky_index", "hour_of_day"]
    has_cover = "tcc" in dataset.columns
    if has_cover:
        columns.append("tcc")
    extra = dataset.select(columns).with_columns(
        regime_cams=sky_regime(
            index=pl.col("cams_clear_sky_index"), thresholds=CLEAR_SKY_INDEX_THRESHOLDS
        ),
        season=season_of_month(month_number=pl.col("time").dt.month()),
    )
    if has_cover:
        extra = extra.with_columns(
            regime_era5=sky_regime_from_cloud_cover(cloud_cover=pl.col("tcc"))
        )
    return losses.join(extra.drop("cams_clear_sky_index"), on=["site", "time"])


def arm_arrays(
    *, losses: pl.DataFrame, dataset: pl.DataFrame, arm: str, setting: str, target: TargetType
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the measured value, the per-seed prediction, and the month of every row of an arm.

    Both are fractions of the farm's capacity, so a large farm does not dominate the correlation.

    Args:
        losses: Per-row losses.
        dataset: The kept rows, carrying the target and `effective_capacity_mw`.
        arm: The arm.
        setting: The hyperparameter setting.
        target: The target.

    Returns:
        The measured values (n_rows,), the predictions (n_seeds, n_rows), and the months (n_rows,),
        in (site, time) order.
    """
    column = TARGET_COLUMNS[target]
    rows = losses.filter((pl.col("arm") == arm) & (pl.col("setting") == setting)).join(
        dataset.select("site", "time", column), on=["site", "time"]
    )
    capacity = pl.col("effective_capacity_mw") if target == "pv" else pl.lit(1.0)
    by_seed = [
        rows.filter(pl.col("seed") == seed).sort("site", "time")
        for seed in sorted(rows["seed"].unique().to_list())
    ]
    first = by_seed[0]
    actual = first.select(actual=pl.col(column) / capacity)["actual"].to_numpy()
    prediction = np.stack(
        [
            frame.select(prediction=(pl.col(column) + pl.col(SIGNED_ERROR)) / capacity)[
                "prediction"
            ].to_numpy()
            for frame in by_seed
        ]
    )
    return actual, prediction, first["month"].to_numpy()


def leaderboard(
    *, losses: pl.DataFrame, dataset: pl.DataFrame, target: TargetType, view: ViewType, device: str
) -> pl.DataFrame:
    """Return each arm's mean absolute error, correlation, and intervals at the primary setting.

    Args:
        losses: Per-row losses of one fit.
        dataset: The kept rows.
        target: The target.
        view: Which rows the fit used.
        device: The XGBoost device the arms were fitted on.

    Returns:
        One row per arm.
    """
    primary = losses.filter(pl.col("setting") == PRIMARY_SETTING)
    scored = with_watts(losses=primary, dataset=dataset) if target == "cams" else primary
    rows: list[dict[str, object]] = []
    for arm in sorted(primary["arm"].unique().to_list(), key=_arm_order):
        _LOG.info("%s target, %s view: leaderboard for %s", target, view, arm)
        error = bootstrap_absolute(losses=scored, arm=arm, metric=METRIC)
        actual, prediction, months = arm_arrays(
            losses=primary, dataset=dataset, arm=arm, setting=PRIMARY_SETTING, target=target
        )
        correlation = pooled_correlation_interval(
            actual=actual, prediction=prediction, months=months
        )
        row: dict[str, object] = {
            "target": target,
            "view": view,
            "arm": arm,
            "label": ARM_LABELS.get(arm, arm),
            "device": device,
            "mae": error["value"],
            "mae_lower": error["lower_95"],
            "mae_upper": error["upper_95"],
            "correlation": correlation["correlation"],
            "correlation_lower": correlation["lower"],
            "correlation_upper": correlation["upper"],
            "n_rows": error["n_rows"],
            "n_months": error["n_months"],
        }
        if target == "cams":
            watts = bootstrap_absolute(losses=scored, arm=arm, metric="absolute_error_w_m2")
            row |= {
                "mae_w_m2": watts["value"],
                "mae_w_m2_lower": watts["lower_95"],
                "mae_w_m2_upper": watts["upper_95"],
            }
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def _arm_order(arm: str) -> tuple[int, int, str]:
    """Order arms: the ladder, then the aerosol arms, the controls, and the drop-one-group arms."""
    if arm in RUNGS:
        return (0, RUNGS.index(arm), arm)
    if arm in (AEROSOL_REFERENCE, AEROSOL_RUNG):
        return (1, 0 if arm == AEROSOL_REFERENCE else 1, arm)
    if arm.startswith(DROP_PREFIX):
        return (3, RUNGS.index(arm.removeprefix(DROP_PREFIX)), arm)
    return (2, 0, arm)


def contrast_row(
    *,
    losses: pl.DataFrame,
    target: TargetType,
    family: str,
    label: str,
    treatment: str,
    reference: str,
    setting: str,
    planned: bool,
) -> dict[str, object]:
    """Return one contrast's difference with its 95% and adjusted intervals.

    The difference is the treatment's error minus the reference's, so a negative value is a gain.

    Args:
        losses: Per-row losses of the setting, holding both arms.
        target: The target.
        family: What kind of contrast it is, such as `planned` or `drop_one`.
        label: A short name for the contrast, such as `P0`.
        treatment: The arm whose error is compared.
        reference: The arm it is compared against.
        setting: The hyperparameter setting.
        planned: Whether the plan named the contrast before any result existed.

    Returns:
        The contrast's table row.

    Raises:
        ValueError: If the two arms hold different rows.
    """
    in_setting = losses.filter(pl.col("setting") == setting)
    raise_unless_same_rows(losses=in_setting, arms=[treatment, reference])
    interval = bootstrap_difference(
        losses=in_setting, treatment=treatment, reference=reference, metric=METRIC
    )
    adjusted_lower, adjusted_upper = bootstrap_difference_at_level(
        losses=in_setting,
        treatment=treatment,
        reference=reference,
        metric=METRIC,
        level=ADJUSTED_LEVEL_PERCENT,
        n_resamples=PLANNED_RESAMPLES if planned else None,
    )
    smallest = SMALLEST_EFFECT[target]
    reference_mean = float(
        in_setting.filter(pl.col("arm") == reference).select(pl.col(METRIC).mean()).item()
    )
    return {
        "target": target,
        "family": family,
        "label": label,
        "treatment": treatment,
        "reference": reference,
        "setting": setting,
        "planned": planned,
        "difference": interval["difference"],
        "reference_mean": reference_mean,
        "relative_difference": interval["difference"] / reference_mean,
        "lower_95": interval["lower_95"],
        "upper_95": interval["upper_95"],
        "lower_adjusted": adjusted_lower,
        "upper_adjusted": adjusted_upper,
        "verdict_adjusted": contrast_verdict(
            lower=adjusted_lower, upper=adjusted_upper, smallest_effect=smallest
        ),
        "verdict_95": contrast_verdict(
            lower=interval["lower_95"], upper=interval["upper_95"], smallest_effect=smallest
        ),
        "near_line": is_near_line(lower=interval["lower_95"], upper=interval["upper_95"]),
        "seed_spread": interval["seed_spread"],
        "n_rows": interval["n_rows"],
        "n_months": interval["n_months"],
    }


def contrast_pairs(*, arms: Sequence[str]) -> list[tuple[str, str, str, str, bool]]:
    """List every contrast the report computes for the arms present.

    Returns tuples of (family, label, treatment, reference, planned). Planned contrasts come first.

    Args:
        arms: The arms the fit holds.

    Returns:
        The contrasts whose two arms are both present.
    """
    present = set(arms)
    pairs: list[tuple[str, str, str, str, bool]] = [
        ("planned", label, treatment, reference, True)
        for label, treatment, reference in PLANNED_CONTRASTS
    ]
    pairs += [
        ("against_g0", f"{rung} - g0", rung, "g0", False)
        for rung in RUNGS[1:]
        if rung not in ("g1", "g2", "g9")
    ]
    pairs += [
        ("step", f"{upper} - {lower}", upper, lower, False)
        for lower, upper in pairwise(RUNGS)
        if lower != "g0"
    ]
    pairs += [
        ("control", "negative control - g2", NEGATIVE_CONTROL_ARM, "g2", False),
        ("control", "g9 - negative control", "g9", NEGATIVE_CONTROL_ARM, False),
        ("control", "positive control - g2", POSITIVE_CONTROL_ARM, "g2", False),
        ("aerosol", "g10 - g9 on the aerosol rows", AEROSOL_RUNG, AEROSOL_REFERENCE, False),
    ]
    pairs += [
        ("drop_one", f"without {rung} - g9", f"{DROP_PREFIX}{rung}", "g9", False)
        for rung in RUNGS[1:]
    ]
    return [pair for pair in pairs if pair[2] in present and pair[3] in present]


def all_contrasts(*, losses: pl.DataFrame, target: TargetType) -> pl.DataFrame:
    """Compute every contrast of a fit at each setting that holds both its arms.

    Args:
        losses: Per-row losses of one fit.
        target: The target.

    Returns:
        One row per (contrast, setting).
    """
    rows: list[dict[str, object]] = []
    arms_by_setting = {
        setting: set(losses.filter(pl.col("setting") == setting)["arm"].unique().to_list())
        for setting in losses["setting"].unique().to_list()
    }
    for family, label, treatment, reference, planned in contrast_pairs(
        arms=sorted(arms_by_setting[PRIMARY_SETTING])
    ):
        for setting, arms in arms_by_setting.items():
            if treatment in arms and reference in arms:
                _LOG.info("%s: %s at %s", target, label, setting)
                rows.append(
                    contrast_row(
                        losses=losses,
                        target=target,
                        family=family,
                        label=label,
                        treatment=treatment,
                        reference=reference,
                        setting=setting,
                        planned=planned,
                    )
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def planned_verdicts(*, contrasts: pl.DataFrame) -> pl.DataFrame:
    """Combine each planned contrast's verdicts at the two settings.

    Args:
        contrasts: The contrast table.

    Returns:
        One row per planned contrast with both settings' verdicts and the combined verdict.
    """
    if "planned" not in contrasts.columns:
        return pl.DataFrame()
    rows: list[dict[str, object]] = []
    planned = contrasts.filter(pl.col("planned"))
    for (target, label), group in planned.group_by("target", "label", maintain_order=True):
        by_setting = {row["setting"]: row for row in group.to_dicts()}
        primary = by_setting[PRIMARY_SETTING]
        second = by_setting.get(SENSITIVITY_SETTING)
        if second is None:
            msg = f"planned contrast {label} on the {target} target has no sensitivity-setting run"
            raise ValueError(msg)
        combined = combine_planned(
            primary=primary["verdict_adjusted"], sensitivity=second["verdict_adjusted"]
        )
        rows.append(
            {
                "target": target,
                "label": label,
                "primary": primary["verdict_adjusted"],
                "sensitivity": second["verdict_adjusted"],
                "combined": combined,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def split_rows(
    *, losses: pl.DataFrame, dataset: pl.DataFrame, target: TargetType, setting: str
) -> pl.DataFrame:
    """Compute the exploratory splits: by regime, season, farm, and hour of day.

    Args:
        losses: Per-row losses of one fit.
        dataset: The kept rows.
        target: The target.
        setting: The hyperparameter setting to split.

    Returns:
        One row per (split, group, contrast or arm).
    """
    in_setting = with_split_columns(
        losses=losses.filter(pl.col("setting") == setting), dataset=dataset
    )
    arms = set(in_setting["arm"].unique().to_list())
    contrasts = [(t, r) for t, r in SPLIT_CONTRASTS if t in arms and r in arms]
    rows: list[dict[str, object]] = []
    hour_contrasts = [(t, r) for t, r in HOUR_CONTRASTS if t in arms and r in arms]
    hour_by_season_contrasts = [
        (t, r) for t, r in HOUR_BY_SEASON_CONTRASTS if t in arms and r in arms
    ]
    groupings: list[tuple[str, tuple[str, ...], list[tuple[str, str]]]] = [
        ("regime_cams", ("regime_cams",), contrasts),
        ("season", ("season",), contrasts),
        ("regime_cams_by_season", ("season", "regime_cams"), contrasts),
        ("hour_of_day_contrast", ("hour_of_day",), hour_contrasts),
        ("hour_by_season_contrast", ("season", "hour_of_day"), hour_by_season_contrasts),
    ]
    if "regime_era5" in in_setting.columns:
        groupings.append(("regime_era5", ("regime_era5",), contrasts))
    for split, columns, split_contrasts in groupings:
        known = in_setting.filter(pl.all_horizontal(pl.col(name).is_not_null() for name in columns))
        for key, subset in known.group_by(*columns, maintain_order=True):
            group = " / ".join(str(part) for part in key)
            for treatment, reference in split_contrasts:
                rows.append(
                    _split_contrast(
                        subset=subset,
                        target=target,
                        split=split,
                        group=group,
                        treatment=treatment,
                        reference=reference,
                    )
                )
    for arm in sorted(arms, key=_arm_order):
        arm_rows = in_setting.filter(pl.col("arm") == arm)
        for column, split in (("site", "farm"), ("hour_of_day", "hour_of_day")):
            for key, subset in arm_rows.group_by(column, maintain_order=True):
                rows.append(
                    {
                        "target": target,
                        "split": split,
                        "group": str(key[0]),
                        "treatment": arm,
                        "reference": None,
                        "value": subset.select(pl.col(METRIC).mean()).item(),
                        "difference": None,
                        "lower_95": None,
                        "upper_95": None,
                        "n_rows": subset.height,
                        "n_months": subset["month"].n_unique(),
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def _split_contrast(
    *,
    subset: pl.DataFrame,
    target: TargetType,
    split: str,
    group: str,
    treatment: str,
    reference: str,
) -> dict[str, object]:
    """Return one contrast inside one regime, season, or both, with a month-resampled interval."""
    n_months = subset["month"].n_unique()
    row: dict[str, object] = {
        "target": target,
        "split": split,
        "group": group,
        "treatment": treatment,
        "reference": reference,
        "value": None,
        "difference": None,
        "lower_95": None,
        "upper_95": None,
        "n_rows": subset.filter(pl.col("arm") == treatment).height,
        "n_months": n_months,
        "reference_value": float(
            subset.filter(pl.col("arm") == reference).select(pl.col(METRIC).mean()).item()
        ),
    }
    if n_months < MIN_MONTHS_FOR_INTERVAL:
        return row
    interval = bootstrap_difference(
        losses=subset, treatment=treatment, reference=reference, metric=METRIC
    )
    row |= {
        "difference": interval["difference"],
        "lower_95": interval["lower_95"],
        "upper_95": interval["upper_95"],
    }
    return row


def worst_days(*, losses: pl.DataFrame, dataset: pl.DataFrame, target: TargetType) -> pl.DataFrame:
    """List the minimal arm's worst farm-days, with what the full arm changed on each.

    A farm-day is identified by the farm's anonymous label and the month and year, and never by the
    day of the month.

    Args:
        losses: Per-row losses at the primary setting.
        dataset: The kept rows.
        target: The target.

    Returns:
        The worst `WORST_DAY_COUNT` farm-days by mean `g0` error, with the `g0` and `g9` errors and
        the day's mean snow depth when the frame holds it.
    """
    primary = losses.filter(pl.col("setting") == PRIMARY_SETTING)
    daily = (
        primary.filter(pl.col("arm").is_in(["g0", "g9"]))
        .with_columns(day=pl.col("time").dt.date())
        .group_by("site", "day", "arm")
        .agg(error=pl.col(METRIC).mean())
        .pivot(on="arm", index=["site", "day"], values="error")
    )
    if "g9" not in daily.columns:
        return pl.DataFrame()
    context = (
        dataset.with_columns(day=pl.col("time").dt.date())
        .group_by("site", "day")
        .agg(
            *(
                pl.col(name).mean().alias(f"mean_{name}")
                for name in ("sd", "tcc")
                if name in dataset.columns
            ),
            hours=pl.len(),
            mean_output_pct=(pl.col("power_mw") / pl.col("effective_capacity_mw")).mean()
            * PERCENTAGE_POINTS,
            mean_clear_sky_index=pl.col("cams_clear_sky_index").mean(),
        )
        .filter(pl.col("hours") >= MIN_HOURS_FOR_WORST_DAY)
    )
    return (
        daily.join(context, on=["site", "day"])
        .sort("g0", descending=True)
        .head(WORST_DAY_COUNT)
        .with_columns(month=pl.col("day").dt.strftime("%Y-%m"), target=pl.lit(target))
        .drop("day")
    )


def table(*, rows: Sequence[Sequence[str]], header: Sequence[str]) -> str:
    """Render rows as a markdown table."""
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return "\n".join(lines)


def interval_text(*, lower: float, upper: float, factor: float = 1.0) -> str:
    """Render an interval as `[lower, upper]` at the report's precision, scaled by `factor`."""
    return f"[{lower * factor:.{PRINT_DECIMALS}f}, {upper * factor:.{PRINT_DECIMALS}f}]"


def scale(*, target: TargetType) -> float:
    """Return what turns the metric into the unit the report prints: points of capacity or index."""
    return PERCENTAGE_POINTS if target == "pv" else 1.0


def render_leaderboard(*, board: pl.DataFrame, target: TargetType) -> str:
    """Render one target's leaderboard as markdown."""
    unit = "% of capacity" if target == "pv" else "clearness index"
    header = [
        "arm",
        f"mean absolute error ({unit})",
        "95% interval",
        "correlation",
        "95% interval",
        "columns",
    ]
    if target == "cams":
        header = [*header[:3], "error (W m⁻²)", *header[3:]]
    factor = scale(target=target)
    rows = []
    for row in board.iter_rows(named=True):
        cells = [
            row["label"],
            f"{row['mae'] * factor:.{PRINT_DECIMALS}f}",
            interval_text(lower=row["mae_lower"], upper=row["mae_upper"], factor=factor),
            f"{row['correlation']:.{PRINT_DECIMALS}f}",
            interval_text(lower=row["correlation_lower"], upper=row["correlation_upper"]),
            str(row["n_columns"]),
        ]
        if target == "cams":
            cells.insert(3, f"{row['mae_w_m2']:.1f}")
        rows.append(cells)
    return table(rows=rows, header=header)


def render_contrasts(*, contrasts: pl.DataFrame, target: TargetType, planned: bool) -> str:
    """Render one target's planned or exploratory contrasts as markdown."""
    unit = "points of capacity" if target == "pv" else "clearness index"
    factor = scale(target=target)
    header = [
        "contrast",
        "setting",
        f"difference ({unit})",
        "difference as % of the reference's error",
        "95% interval",
        f"{ADJUSTED_LEVEL_PERCENT:.3f}% interval" if planned else "near the 5% line",
        "verdict" if planned else "months",
    ]
    rows = []
    selected = contrasts.filter((pl.col("target") == target) & (pl.col("planned") == planned))
    for row in selected.iter_rows(named=True):
        fourth = (
            interval_text(lower=row["lower_adjusted"], upper=row["upper_adjusted"], factor=factor)
            if planned
            else ("yes" if row["near_line"] else "no")
        )
        last = row["verdict_adjusted"] if planned else str(row["n_months"])
        rows.append(
            [
                f"{row['label']} ({row['treatment']} minus {row['reference']})",
                row["setting"],
                f"{row['difference'] * factor:+.{PRINT_DECIMALS}f}",
                f"{row['relative_difference'] * PERCENTAGE_POINTS:+.1f}",
                interval_text(lower=row["lower_95"], upper=row["upper_95"], factor=factor),
                fourth,
                last,
            ]
        )
    return table(rows=rows, header=header)


def render_worst_days(*, worst: pl.DataFrame) -> str:
    """Render the worst farm-days as markdown, by farm label and month, never by day.

    Args:
        worst: The table `worst_days` returns, for each target.

    Returns:
        The markdown table. Errors are in the target's own unit: percentage points of capacity for
        the output target, and clearness index for the CAMS target.
    """
    context = [
        name
        for name in ("hours", "mean_output_pct", "mean_clear_sky_index", "mean_sd", "mean_tcc")
        if name in worst.columns
    ]
    rows = []
    for row in worst.iter_rows(named=True):
        factor = scale(target=row["target"])
        rows.append(
            [
                row["target"],
                row["site"],
                row["month"],
                f"{row['g0'] * factor:.{PRINT_DECIMALS}f}",
                f"{row['g9'] * factor:.{PRINT_DECIMALS}f}",
                *(
                    "n/a"
                    if row[name] is None
                    else (f"{row[name]}" if name == "hours" else f"{row[name]:.{PRINT_DECIMALS}f}")
                    for name in context
                ),
            ]
        )
    return table(rows=rows, header=["target", "farm", "month", "g0 error", "g9 error", *context])


def near_line_without_second_setting(*, contrasts: pl.DataFrame) -> pl.DataFrame:
    """List the exploratory contrasts near the 5% line that have no second-setting run.

    Args:
        contrasts: The contrast table.

    Returns:
        The (target, treatment, reference) of each such contrast at the primary setting.
    """
    if "near_line" not in contrasts.columns:
        return pl.DataFrame()
    second = contrasts.filter(pl.col("setting") == SENSITIVITY_SETTING).select(
        "target", "treatment", "reference"
    )
    return (
        contrasts.filter(
            (pl.col("setting") == PRIMARY_SETTING) & pl.col("near_line") & ~pl.col("planned")
        )
        .select("target", "treatment", "reference")
        .join(second, on=["target", "treatment", "reference"], how="anti")
    )


def render_mars_decision(*, contrasts: pl.DataFrame) -> list[str]:
    """Return the report lines stating what the MARS decision rule says about contrast P4.

    The rule reads the output target's adjusted interval at both settings, and recommends a pilot or
    recommends against only if the two settings agree.

    Args:
        contrasts: The contrast table.

    Returns:
        Lines of markdown, empty if P4 on the output target has not been computed.
    """
    p4 = contrasts.filter(
        (pl.col("label") == "P4") & (pl.col("target") == "pv") & pl.col("planned")
    )
    if p4.is_empty():
        return []
    by_setting = {row["setting"]: row for row in p4.iter_rows(named=True)}
    recommendations = {
        setting: mars_fetch_recommendation(
            lower=row["lower_adjusted"],
            upper=row["upper_adjusted"],
            smallest_effect=SMALLEST_EFFECT["pv"],
        )
        for setting, row in by_setting.items()
    }
    distinct = set(recommendations.values())
    if SENSITIVITY_SETTING not in by_setting or len(distinct) > 1:
        overall = "unresolved"
    else:
        overall = next(iter(distinct))
    return [
        "## MARS fetch decision (planned contrast P4, output target)",
        "",
        *(
            f"- {setting} setting: {recommendation}"
            for setting, recommendation in recommendations.items()
        ),
        f"- **Overall: {overall}**",
        "",
    ]


def probabilistic_rows(
    *, losses: pl.DataFrame, dataset: pl.DataFrame, target: TargetType, setting: str
) -> pl.DataFrame:
    """Summarise the quantile fits: scores by arm, contrasts, reliability, and scores by regime.

    Args:
        losses: Per-row losses of one fit, holding the probabilistic columns (null for an arm
            without a quantile fit).
        dataset: The kept rows, carrying `tcc` and `cams_clear_sky_index`.
        target: The target.
        setting: The hyperparameter setting to summarise, which is the one the quantile fits ran at.

    Returns:
        A long frame with `kind` of `arm`, `contrast`, `reliability`, or `regime`, then `arm`
        (or `treatment` and `reference`), `group`, `measure`, `value`, `difference`, and the 95%
        interval bounds of a contrast.
    """
    crps_column = PROBABILISTIC_METRICS["crps"]
    in_setting = losses.filter(
        (pl.col("setting") == setting)
        & pl.col("arm").is_in(QUANTILE_ARMS)
        & pl.col(crps_column).is_not_null()
    )
    if in_setting.is_empty():
        return pl.DataFrame()
    arms = sorted(in_setting["arm"].unique().to_list(), key=_arm_order)
    raise_unless_same_rows(losses=in_setting, arms=arms)
    rows: list[dict[str, object]] = []
    for arm in arms:
        arm_rows = in_setting.filter(pl.col("arm") == arm)
        for measure, column in PROBABILISTIC_METRICS.items():
            rows.append(
                {
                    "target": target,
                    "kind": "arm",
                    "arm": arm,
                    "measure": measure,
                    "value": arm_rows[column].mean(),
                }
            )
        rows.extend(
            {
                "target": target,
                "kind": "reliability",
                "arm": arm,
                "group": f"{level:.1f}",
                "measure": "share_below",
                "value": arm_rows[f"below_q{round(level * 100)}"].mean(),
            }
            for level in QUANTILE_LEVELS
        )
    # A single interval from the arm's own out-of-fold errors needs no second model, so it is the
    # reference a regime-aware quantile model has to beat.
    for arm in arms:
        errors = in_setting.filter(pl.col("arm") == arm)
        capacity = errors["effective_capacity_mw"]
        shares = errors["signed_error_capped_mw"] / capacity
        rows.append(
            {
                "target": target,
                "kind": "residual_reference",
                "arm": arm,
                "measure": "width_80",
                "value": float(shares.quantile(0.9, interpolation="linear") or 0.0)
                - float(shares.quantile(0.1, interpolation="linear") or 0.0),
            }
        )
    # The same interval, but holding as many outcomes as the arm's quantile interval did, so the two
    # widths answer for the same coverage.
    for arm in arms:
        errors = in_setting.filter(pl.col("arm") == arm)
        shares = errors["signed_error_capped_mw"] / errors["effective_capacity_mw"]
        coverage = float(errors[PROBABILISTIC_METRICS["coverage_80"]].mean())  # ty: ignore[invalid-argument-type]
        upper = float(shares.quantile(0.5 + coverage / 2, interpolation="linear") or 0.0)
        lower = float(shares.quantile(0.5 - coverage / 2, interpolation="linear") or 0.0)
        rows.append(
            {
                "target": target,
                "kind": "residual_reference_matched",
                "arm": arm,
                "measure": "width_80",
                "value": upper - lower,
            }
        )
    for treatment, reference in PROBABILISTIC_CONTRASTS:
        if treatment not in arms or reference not in arms:
            continue
        for measure, column in PROBABILISTIC_METRICS.items():
            if (
                in_setting.filter(pl.col("arm") == treatment)
                .select(pl.col(column))
                .to_series()
                .is_null()
                .all()
            ):
                continue
            interval = bootstrap_difference(
                losses=in_setting, treatment=treatment, reference=reference, metric=column
            )
            rows.append(
                {
                    "target": target,
                    "kind": "contrast",
                    "treatment": treatment,
                    "reference": reference,
                    "measure": measure,
                    "difference": interval["difference"],
                    "lower_95": interval["lower_95"],
                    "upper_95": interval["upper_95"],
                }
            )
    with_regimes = with_split_columns(losses=in_setting, dataset=dataset)
    if "regime_era5" in with_regimes.columns:
        for (arm, regime), subset in with_regimes.group_by("arm", "regime_era5"):
            rows.extend(
                {
                    "target": target,
                    "kind": "regime",
                    "arm": arm,
                    "group": regime,
                    "measure": measure,
                    "value": subset[column].mean(),
                    "n_rows": subset.height,
                }
                for measure, column in PROBABILISTIC_METRICS.items()
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def reference_key(*, arm: str, matched: bool = False) -> tuple[str, str, None, str]:
    """Return the key of an arm's constant-interval reference width in the long frame."""
    kind = "residual_reference_matched" if matched else "residual_reference"
    return (kind, arm, None, "width_80")


def render_probabilistic(*, rows: pl.DataFrame) -> str:
    """Render the arm scores, the contrasts, the reliability table, and the regime scores."""
    if rows.is_empty():
        return "No quantile fits."
    parts: list[str] = []
    for target in TARGETS:
        subset = rows.filter(pl.col("target") == target)
        if subset.is_empty():
            continue
        unit = {
            "crps": scale(target=target),
            "coverage_80": PERCENTAGE_POINTS,
            "width_80": scale(target=target),
        }
        measures = tuple(unit)
        values = {
            (row["kind"], row["arm"], row["group"], row["measure"]): row
            for row in subset.filter(pl.col("kind") != "contrast").iter_rows(named=True)
        }
        arms = sorted({key[1] for key in values}, key=_arm_order)
        mean_width = {arm: values[("arm", arm, None, "width_80")]["value"] for arm in arms}
        matched_width = {arm: values[reference_key(arm=arm, matched=True)]["value"] for arm in arms}
        matched_ratio = {arm: matched_width[arm] / mean_width[arm] for arm in arms}
        parts += [f"### {target} target: scores by arm", ""]
        parts.append(
            table(
                header=[
                    "arm",
                    "CRPS",
                    "coverage of the 10-90 interval (%)",
                    "mean width",
                    "width of one constant interval holding 80% of the arm's errors",
                    "width of one constant interval holding the arm's own coverage",
                    "that constant width over the quantile fit's mean width",
                ],
                rows=[
                    [
                        arm,
                        *(
                            f"{values[('arm', arm, None, m)]['value'] * unit[m]:.{PRINT_DECIMALS}f}"
                            for m in measures
                        ),
                        f"{values[reference_key(arm=arm)]['value'] * unit['width_80']:.3f}",
                        f"{matched_width[arm] * unit['width_80']:.3f}",
                        f"{matched_ratio[arm]:.3f}",
                    ]
                    for arm in arms
                ],
            )
        )
        if "g0" in mean_width and "g9" in mean_width:
            narrowing = mean_width["g9"] / mean_width["g0"] - 1.0
            parts += [
                "",
                (
                    f"The full set's mean interval width is {narrowing * PERCENTAGE_POINTS:+.1f}% "
                    "of the minimal set's."
                ),
            ]
        parts += ["", f"### {target} target: contrasts (exploratory, 95% intervals)", ""]
        parts.append(
            table(
                header=["treatment minus reference", "measure", "difference", "95% interval"],
                rows=[
                    [
                        f"{row['treatment']} minus {row['reference']}",
                        row["measure"],
                        f"{row['difference'] * unit[row['measure']]:+.{PRINT_DECIMALS}f}",
                        interval_text(
                            lower=row["lower_95"],
                            upper=row["upper_95"],
                            factor=unit[row["measure"]],
                        ),
                    ]
                    for row in subset.filter(pl.col("kind") == "contrast").iter_rows(named=True)
                ],
            )
        )
        levels = sorted({key[2] for key in values if key[0] == "reliability"})
        parts += ["", f"### {target} target: share of outcomes at or below each quantile", ""]
        parts.append(
            table(
                header=["arm", *levels],
                rows=[
                    [
                        arm,
                        *(
                            f"{values[('reliability', arm, level, 'share_below')]['value']:.3f}"
                            for level in levels
                        ),
                    ]
                    for arm in arms
                ],
            )
        )
        regime_keys = sorted({key[1:3] for key in values if key[0] == "regime"})
        if regime_keys:
            parts += ["", f"### {target} target: scores by ERA5 cloud regime", ""]
            parts.append(
                table(
                    header=[
                        "arm",
                        "regime",
                        "CRPS",
                        "coverage (%)",
                        "mean width",
                        "row-seed pairs",
                    ],
                    rows=[
                        [
                            arm,
                            regime,
                            *(
                                f"{values[('regime', arm, regime, m)]['value'] * unit[m]:.3f}"
                                for m in measures
                            ),
                            f"{values[('regime', arm, regime, 'crps')]['n_rows']:,}",
                        ]
                        for arm, regime in regime_keys
                    ],
                )
            )
        parts.append("")
    return "\n".join(parts)


def aerosol_condition_rows(
    *, losses: pl.DataFrame, dataset: pl.DataFrame, target: TargetType, setting: str
) -> pl.DataFrame:
    """Compare the aerosol rung with its reference inside each pre-specified aerosol condition.

    The conditions come from `aerosol_condition_flags`. Each condition reports its event counts
    and, for the aerosol rung minus the reference, the mean absolute error with a month-resampled
    interval, the 95th percentile and the maximum of the farm-day mean absolute error, and the mean
    signed error as a share of capacity. A condition spanning fewer than `MIN_MONTHS_FOR_INTERVAL`
    months gets no interval.

    Args:
        losses: Per-row losses of the aerosol view.
        dataset: The kept rows, with `tcc`, `aod550`, and `duaod550`.
        target: The target.
        setting: The hyperparameter setting to analyse.

    Returns:
        One row per (condition, measure), with `treatment`, `reference`, the two arms' values, the
        difference, the interval bounds where one exists, and the event counts.
    """
    covered = dataset.filter(
        pl.all_horizontal(pl.col(name).is_not_null() for name in ("tcc", "aod550", "duaod550"))
    )
    flags = aerosol_condition_flags(frame=covered)
    scored = (
        losses.filter(
            (pl.col("setting") == setting) & pl.col("arm").is_in([AEROSOL_RUNG, AEROSOL_REFERENCE])
        )
        .join(flags, on=["site", "time"])
        .with_columns(
            signed_share=pl.col("signed_error_capped_mw") / pl.col("effective_capacity_mw"),
            day=pl.col("time").dt.date(),
        )
    )
    raise_unless_same_rows(losses=scored, arms=[AEROSOL_RUNG, AEROSOL_REFERENCE])
    rows: list[dict[str, object]] = []
    for condition in AEROSOL_CONDITIONS:
        subset = scored.filter(pl.col(condition))
        one_arm = subset.filter(pl.col("arm") == AEROSOL_RUNG)
        counts = {
            "hours": one_arm.select(pl.struct("site", "time").n_unique()).item(),
            "farm_days": one_arm.select(pl.struct("site", "day").n_unique()).item(),
            "days": one_arm["day"].n_unique(),
            "months": one_arm["month"].n_unique(),
        }
        if subset.is_empty():
            continue
        daily = (
            subset.group_by("arm", "site", "day")
            .agg(error=pl.col(METRIC).mean())
            .group_by("arm")
            .agg(
                p95=pl.col("error").quantile(0.95, interpolation="linear"),
                worst=pl.col("error").max(),
            )
        )
        has_quantiles = subset[PROBABILISTIC_METRICS["crps"]].null_count() < subset.height
        means = {
            "mean_absolute_error": METRIC,
            "signed_error": "signed_share",
            **(dict(PROBABILISTIC_METRICS) if has_quantiles else {}),
        }
        by_arm = {
            arm: {
                **{
                    name: subset.filter(pl.col("arm") == arm)[column].mean()
                    for name, column in means.items()
                },
                "farm_day_p95": daily.filter(pl.col("arm") == arm)["p95"].item(),
                "farm_day_worst": daily.filter(pl.col("arm") == arm)["worst"].item(),
            }
            for arm in (AEROSOL_RUNG, AEROSOL_REFERENCE)
        }
        intervals: dict[str, BootstrapInterval] = {}
        if counts["months"] >= MIN_MONTHS_FOR_INTERVAL:
            for name in ("mean_absolute_error", "crps"):
                if name in means:
                    intervals[name] = bootstrap_difference(
                        losses=subset,
                        treatment=AEROSOL_RUNG,
                        reference=AEROSOL_REFERENCE,
                        metric=means[name],
                    )
        for measure in by_arm[AEROSOL_RUNG]:
            treated = float(by_arm[AEROSOL_RUNG][measure])  # ty: ignore[invalid-argument-type]
            reference = float(by_arm[AEROSOL_REFERENCE][measure])  # ty: ignore[invalid-argument-type]
            interval = intervals.get(measure)
            rows.append(
                {
                    "target": target,
                    "setting": setting,
                    "condition": condition,
                    "measure": measure,
                    "treatment": AEROSOL_RUNG,
                    "reference": AEROSOL_REFERENCE,
                    "treatment_value": treated,
                    "reference_value": reference,
                    "difference": treated - reference,
                    "lower_95": interval["lower_95"] if interval is not None else None,
                    "upper_95": interval["upper_95"] if interval is not None else None,
                    **counts,
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def _points_interval(*, row: dict[str, object]) -> str:
    """Return a row's 95% interval in points of capacity."""
    return interval_text(
        lower=float(row["lower_95"]),  # ty: ignore[invalid-argument-type]
        upper=float(row["upper_95"]),  # ty: ignore[invalid-argument-type]
        factor=PERCENTAGE_POINTS,
    )


def render_aerosol_decision(*, conditions: pl.DataFrame) -> list[str]:
    """State the pre-specified aerosol rule's outcome from the clear-and-dusty PV rows.

    Args:
        conditions: The aerosol-condition rows of every target and setting.

    Returns:
        Report lines giving the event counts, each setting's interval, and the rule's outcome.
    """
    rows = conditions.filter(
        (pl.col("target") == "pv")
        & (pl.col("condition") == "clear_and_dusty")
        & (pl.col("measure") == "mean_absolute_error")
    )
    if rows.is_empty():
        return []
    days = int(rows["days"].max())  # ty: ignore[invalid-argument-type]
    months = int(rows["months"].max())  # ty: ignore[invalid-argument-type]
    with_interval = rows.filter(pl.col("upper_95").is_not_null())
    outcome = aerosol_trial_recommendation(
        uppers=with_interval["upper_95"].to_list(),
        smallest_effect=SMALLEST_EFFECT["pv"],
        days=days,
        months=months,
    )
    return [
        "",
        "### Outcome of the aerosol reading rule (PV target, clear and dusty hours)",
        "",
        f"- Distinct days: {days}. Calendar months: {months}.",
        *(
            f"- {row['setting']} setting: G10 minus G9 {_points_interval(row=row)} points."
            for row in with_interval.iter_rows(named=True)
        ),
        f"- Outcome: **{outcome.replace('_', ' ')}**.",
    ]


IMPORTANCE_TOP_COLUMNS: Final[int] = 12
"""How many columns of the full arm's importance the report lists, best first."""


def render_importance(*, through_rung: RungType, variant: str) -> str:
    """Render the full arm's XGBoost importance and the shuffled-copy noise line, per target.

    The share is each column's gain as a fraction of the model's total gain, averaged over farms,
    folds, and seeds, in the refit that holds the full set and a shuffled copy of its columns. The
    importance of an XGBoost model is descriptive and not a test: correlated columns split the
    credit among themselves. The noise line is the largest share that any shuffled copy of a
    variable takes.

    Args:
        through_rung: The highest rung of the build.
        variant: The build variant.

    Returns:
        The markdown, empty if the importance file does not exist.
    """
    path = importance_path(variant=variant, through_rung=through_rung)
    if not path.exists():
        return ""
    shares = pl.read_parquet(path)
    parts: list[str] = []
    for target in TARGETS:
        full = (
            shares.filter(
                (pl.col("target") == target)
                & (pl.col("arm") == SHUFFLED_ARM)
                & ~pl.col("column").str.ends_with(SHUFFLED_SUFFIX)
            )
            .group_by("column")
            .agg(share=pl.col("share").mean())
            .sort("share", descending=True)
        )
        noise = (
            shares.filter(
                (pl.col("target") == target)
                & (pl.col("arm") == SHUFFLED_ARM)
                & pl.col("column").str.ends_with(SHUFFLED_SUFFIX)
            )
            .group_by("column")
            .agg(share=pl.col("share").mean())
            .sort("share", descending=True)
        )
        if full.is_empty() or noise.is_empty():
            continue
        parts += [
            f"### {target} target: share of the full arm's gain, mean over farms, folds, and seeds",
            "",
            table(
                header=["column", "share of gain (%)"],
                rows=[
                    [row["column"], f"{row['share'] * PERCENTAGE_POINTS:.2f}"]
                    for row in full.head(IMPORTANCE_TOP_COLUMNS).iter_rows(named=True)
                ],
            ),
            "",
            (
                f"Noise line: the largest shuffled copy takes {noise['share'][0] * 100:.2f}% "
                f"({noise['column'][0]}), and the shuffled copies together take "
                f"{noise['share'].sum() * 100:.2f}%."
            ),
            "",
        ]
    return "\n".join(parts)


def render_splits(*, splits: pl.DataFrame) -> str:
    """Render the regime, season, and hour-of-day contrasts as markdown, one table per target.

    Args:
        splits: The splits table, with contrast rows carrying `difference` and an interval.

    Returns:
        The markdown, empty if there are no contrast rows.
    """
    if splits.is_empty():
        return ""
    contrast_rows = splits.filter(pl.col("reference").is_not_null())
    parts: list[str] = []
    for target in TARGETS:
        subset = contrast_rows.filter(pl.col("target") == target)
        if subset.is_empty():
            continue
        factor = scale(target=target)
        parts += [
            f"### {target} target: contrasts by split group, primary setting (exploratory)",
            "",
            table(
                header=[
                    "split",
                    "group",
                    "contrast",
                    "difference",
                    "difference as % of the reference's error",
                    "95% interval",
                    "months",
                ],
                rows=[
                    [
                        row["split"],
                        row["group"],
                        f"{row['treatment']} minus {row['reference']}",
                        "n/a"
                        if row["difference"] is None
                        else f"{row['difference'] * factor:+.3f}",
                        "n/a"
                        if row["difference"] is None
                        else (
                            f"{row['difference'] / row['reference_value'] * PERCENTAGE_POINTS:+.1f}"
                        ),
                        "n/a"
                        if row["difference"] is None
                        else interval_text(
                            lower=row["lower_95"], upper=row["upper_95"], factor=factor
                        ),
                        str(row["n_months"]),
                    ]
                    for row in subset.iter_rows(named=True)
                ],
            ),
            "",
        ]
    return "\n".join(parts)


def render_aerosol_conditions(*, conditions: pl.DataFrame) -> str:
    """Render the aerosol-condition table, with the event counts that let a reader discount it."""
    if conditions.is_empty():
        return "No aerosol-condition rows."
    body = []
    for row in conditions.iter_rows(named=True):
        factor = (
            PERCENTAGE_POINTS if row["measure"] == "coverage_80" else scale(target=row["target"])
        )
        interval = (
            interval_text(lower=row["lower_95"], upper=row["upper_95"], factor=factor)
            if row["lower_95"] is not None
            else "no interval"
        )
        body.append(
            [
                row["target"],
                row["setting"],
                row["condition"],
                row["measure"],
                f"{row['reference_value'] * factor:.{PRINT_DECIMALS}f}",
                f"{row['treatment_value'] * factor:.{PRINT_DECIMALS}f}",
                f"{row['difference'] * factor:+.{PRINT_DECIMALS}f}",
                interval,
                f"{row['hours']:,}",
                f"{row['farm_days']:,}",
                f"{row['days']:,}",
                f"{row['months']:,}",
            ]
        )
    return table(
        rows=body,
        header=[
            "target",
            "setting",
            "condition",
            "measure",
            "g9 on aerosol rows",
            "g10",
            "g10 minus g9",
            "95% interval (mean absolute error and CRPS only)",
            "hours",
            "farm-days",
            "days",
            "months",
        ],
    )


def render_settings(
    *,
    contrasts: pl.DataFrame,
    arms_json: dict[TargetType, dict[str, object]],
    dataset: pl.DataFrame,
) -> str:
    """Render the settings and the derived numbers that the page quotes, as markdown.

    Args:
        contrasts: The contrast table.
        arms_json: Each target's arms and their columns.
        dataset: The kept rows.

    Returns:
        The markdown, with one bullet per setting or derived number.
    """
    pv_arms: dict[str, list[str]] = arms_json["pv"]["arms"]  # ty: ignore[invalid-assignment]
    smallest_pv = SMALLEST_EFFECT["pv"] * PERCENTAGE_POINTS
    extra_columns = len(pv_arms[NEGATIVE_CONTROL_ARM]) - len(pv_arms["g2"])
    lines = [
        f"- Hyperparameters, main setting: {dict(PRIMARY_HYPER_PARAMETERS)}.",
        f"- Hyperparameters, second setting: {dict(SENSITIVITY_HYPER_PARAMETERS)}.",
        "- Both settings use `colsample_bytree=1`, which the code leaves unset.",
        f"- Folds per farm: {N_FOLDS} contiguous blocks of whole months. Seeds: {len(SEEDS)}.",
        (
            f"- Resamples: {PLANNED_RESAMPLES:,} for planned contrasts, "
            f"{N_BOOTSTRAP_RESAMPLES:,} for the others, each drawing one seed and whole months."
        ),
        (
            "- Daylight hours need a top-of-atmosphere horizontal flux above "
            f"{MIN_EXTRATERRESTRIAL_W_M2:g} W m⁻²."
        ),
        (
            "- Effective capacity is the 99th percentile of a farm's metered output "
            "(`studies.pv_dataset`)."
        ),
        (
            f"- Smallest effects: {smallest_pv:g} points of capacity on output, "
            f"{SMALLEST_EFFECT['cams']:g} on the clearness index."
        ),
        (
            "- Near-the-line rule for a second-setting run: a bound of the 95% interval within "
            f"{NEAR_LINE_SHARE:g} of the interval's width from zero."
        ),
        (
            f"- Aerosol reading rule: at least {MIN_AEROSOL_DAYS} days in {MIN_AEROSOL_MONTHS} "
            f"months, and an interval wholly below minus {smallest_pv:g} points at both settings."
        ),
        f"- MARS-only variables ({len(MARS_ONLY_VARIABLES)}): {', '.join(MARS_ONLY_VARIABLES)}.",
        (
            f"- Columns: `g0` {len(pv_arms['g0'])}, `g2` {len(pv_arms['g2'])}, `g9` "
            f"{len(pv_arms['g9'])}, negative control {len(pv_arms[NEGATIVE_CONTROL_ARM])}, so the "
            f"negative control adds {extra_columns} shuffled columns to `g2`."
        ),
        (
            f"- Rows: {dataset.height:,}, from {dataset['time'].min():%B %Y} to "
            f"{dataset['time'].max():%B %Y}."
        ),
    ]
    if AEROSOL_COLUMNS[0] in dataset.columns:
        covered = dataset.filter(pl.col(AEROSOL_COLUMNS[0]).is_not_null())
        lines.append(
            f"- Rows with CAMS aerosol: {covered.height:,}, from {covered['time'].min():%B %Y} to "
            f"{covered['time'].max():%B %Y}."
        )
    primary = contrasts.filter(
        (pl.col("target") == "pv") & (pl.col("setting") == PRIMARY_SETTING)
    ).with_columns(key=pl.format("{}-{}", pl.col("treatment"), pl.col("reference")))
    difference = dict(zip(primary["key"].to_list(), primary["difference"].to_list(), strict=True))
    needed = ("g9-g0", "g2-g0", "g1-g0", "g2-g1", "g4-g3", "g4-g0")
    if all(name in difference for name in needed):
        total = difference["g9-g0"]
        cloud_steps = difference["g1-g0"] + difference["g2-g1"] + difference["g4-g3"]
        lines += [
            (
                "- Share of the full set's gain from total and layered cloud cover (`g2` minus "
                f"`g0`): {difference['g2-g0'] / total * PERCENTAGE_POINTS:.0f}%."
            ),
            (
                "- Share from the G1, G2, and G4 steps together: "
                f"{cloud_steps / total * PERCENTAGE_POINTS:.0f}% ({-cloud_steps:.2f} of "
                f"{-total:.2f} points). The G3 step is left out."
            ),
            (
                "- Share of the gain at `g4` (`g4` minus `g0`): "
                f"{difference['g4-g0'] / total * PERCENTAGE_POINTS:.0f}%."
            ),
        ]
    return "\n".join(lines)


def render_report(
    *,
    variant: str,
    through_rung: RungType,
    boards: dict[TargetType, pl.DataFrame],
    contrasts: pl.DataFrame,
    verdicts: pl.DataFrame,
    arms_json: dict[TargetType, dict[str, object]],
    build_notes: str,
    worst: pl.DataFrame,
    aerosol_conditions: pl.DataFrame,
    probabilistic: pl.DataFrame,
    splits: pl.DataFrame,
    dataset: pl.DataFrame,
) -> str:
    """Assemble the report text."""
    parts = [
        f"# ERA5 variable ladder: report ({variant} rows, through {through_rung})",
        "",
        "## Build checks",
        "",
        build_notes,
        "## Device and settings",
        "",
        *(
            f"- {target} target: fitted on `{info['device']}`, {info['rows']:,} rows."
            for target, info in arms_json.items()
        ),
        "",
        "## Columns each arm was shown",
        "",
    ]
    for target, info in arms_json.items():
        parts.append(f"### {target} target")
        parts.append("")
        parts.extend(
            f"- `{arm}` ({len(columns)}): {', '.join(columns)}"
            for arm, columns in info["arms"].items()  # ty: ignore[unresolved-attribute]
        )
        parts.append("")
    parts += [
        "## Settings and derived numbers the page quotes",
        "",
        render_settings(contrasts=contrasts, arms_json=arms_json, dataset=dataset),
        "",
    ]
    near = near_line_without_second_setting(contrasts=contrasts)
    if not near.is_empty():
        second = contrasts.filter(pl.col("setting") == SENSITIVITY_SETTING)
        already = {*second["treatment"].to_list(), *second["reference"].to_list()}
        arms_to_add = sorted({*near["treatment"].to_list(), *near["reference"].to_list()} - already)
        parts += [
            "## Exploratory contrasts near the 5% line with no second-setting run",
            "",
            *(
                f"- {row['target']}: {row['treatment']} minus {row['reference']}"
                for row in near.iter_rows(named=True)
            ),
            "",
            (
                f"Fit the second setting with `era5_ladder_fit.py --variant {variant} "
                f"--through-rung {through_rung} --view extra_sensitivity "
                f"--sensitivity-arms {' '.join(arms_to_add)}`, then re-run the report."
            ),
            "",
        ]
    parts += render_mars_decision(contrasts=contrasts)
    parts += ["## Planned verdicts, both settings combined", ""]
    parts.append(
        table(
            rows=[
                [
                    row["target"],
                    row["label"],
                    str(row["primary"]),
                    str(row["sensitivity"]),
                    str(row["combined"]),
                ]
                for row in verdicts.iter_rows(named=True)
            ],
            header=["target", "contrast", "primary setting", "second setting", "combined"],
        )
    )
    for target in TARGETS:
        if target not in boards:
            continue
        parts += [
            "",
            f"## Mean absolute error and correlation, {target} target",
            "",
            render_leaderboard(board=boards[target], target=target),
            "",
            f"## Planned contrasts, {target} target (planned)",
            "",
            render_contrasts(contrasts=contrasts, target=target, planned=True),
            "",
            f"## Exploratory contrasts, {target} target (exploratory)",
            "",
            render_contrasts(contrasts=contrasts, target=target, planned=False),
        ]
    if not worst.is_empty():
        parts += [
            "",
            "## Worst farm-days of the minimal arm (exploratory)",
            "",
            render_worst_days(worst=worst),
        ]
    if not splits.is_empty():
        parts += [
            "",
            "## Contrasts by cloud regime, season, and UTC hour of day (exploratory)",
            "",
            render_splits(splits=splits),
        ]
    importance_text = render_importance(through_rung=through_rung, variant=variant)
    if importance_text:
        parts += [
            "",
            "## XGBoost importance of the full arm (descriptive)",
            "",
            importance_text,
        ]
    if not probabilistic.is_empty():
        parts += [
            "",
            "## Probabilistic scores (exploratory, pre-specified)",
            "",
            (
                "CRPS and width are in points of capacity for the PV target and in clearness-index "
                "units for the CAMS target. A negative difference is a gain. The quantiles are "
                "sorted, held at the export cap, and floored at zero."
            ),
            "",
            render_probabilistic(rows=probabilistic),
        ]
    if not aerosol_conditions.is_empty():
        parts += [
            "",
            "## Aerosol in unusual conditions (exploratory, pre-specified)",
            "",
            (
                "Values are scaled as in the other tables; a negative difference is a gain. The "
                "signed error is prediction minus measurement."
            ),
            "",
            render_aerosol_conditions(conditions=aerosol_conditions),
            *render_aerosol_decision(conditions=aerosol_conditions),
        ]
    return "\n".join(parts) + "\n"


def aerosol_conditions_by_setting(
    *, losses: pl.DataFrame, dataset: pl.DataFrame, target: TargetType
) -> list[pl.DataFrame]:
    """Run the aerosol-condition analysis at each hyperparameter setting the fit holds."""
    present = set(losses["setting"].unique().to_list())
    return [
        aerosol_condition_rows(losses=losses, dataset=dataset, target=target, setting=setting)
        for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING)
        if setting in present
    ]


def stack_nonempty(*, frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Stack the frames that hold rows, or return an empty frame if none does."""
    kept = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(kept, how="diagonal_relaxed") if kept else pl.DataFrame()


def main() -> int:
    """Score the saved fits and write the report and tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through-rung", choices=RUNGS, default=RUNGS[-1])
    parser.add_argument("--variant", choices=("main", "snow_zero_hours"), default="main")
    parser.add_argument("--skip-splits", action="store_true")
    arguments = parser.parse_args()
    through_rung: RungType = arguments.through_rung
    variant: str = arguments.variant
    paths = report_paths(variant=variant, through_rung=through_rung)
    refuse_to_overwrite(paths=list(paths))

    dataset = pl.read_parquet(dataset_path(through_rung=through_rung, variant=variant))
    boards: dict[TargetType, pl.DataFrame] = {}
    contrast_frames: list[pl.DataFrame] = []
    split_frames: list[pl.DataFrame] = []
    worst_frames: list[pl.DataFrame] = []
    condition_frames: list[pl.DataFrame] = []
    probabilistic_frames: list[pl.DataFrame] = []
    arms_json: dict[TargetType, dict[str, object]] = {}
    for target in TARGETS:
        key = FitKey(variant=variant, through_rung=through_rung, target=target, view="ladder")
        losses = read_losses(key=key)
        arms_json[target] = json.loads(arms_path(key=key).read_text())
        board = leaderboard(
            losses=losses,
            dataset=dataset,
            target=target,
            view="ladder",
            device=str(arms_json[target]["device"]),
        )
        boards[target] = board.with_columns(
            n_columns=pl.col("arm").replace_strict(
                {arm: len(columns) for arm, columns in arms_json[target]["arms"].items()},  # ty: ignore[unresolved-attribute]
                return_dtype=pl.Int64,
            )
        )
        contrast_frames.append(all_contrasts(losses=losses, target=target))
        if not arguments.skip_splits:
            split_frames.append(
                split_rows(losses=losses, dataset=dataset, target=target, setting=PRIMARY_SETTING)
            )
        worst_frames.append(worst_days(losses=losses, dataset=dataset, target=target))
        probabilistic_frames.append(
            probabilistic_rows(
                losses=losses, dataset=dataset, target=target, setting=PRIMARY_SETTING
            )
        )
        aerosol_key = FitKey(
            variant=variant, through_rung=through_rung, target=target, view="aerosol_rows"
        )
        if results_path(key=aerosol_key).exists():
            aerosol_losses = read_losses(key=aerosol_key)
            contrast_frames.append(all_contrasts(losses=aerosol_losses, target=target))
            probabilistic_frames.append(
                probabilistic_rows(
                    losses=aerosol_losses,
                    dataset=dataset,
                    target=target,
                    setting=PRIMARY_SETTING,
                )
            )
            condition_frames.extend(
                aerosol_conditions_by_setting(losses=aerosol_losses, dataset=dataset, target=target)
            )
    contrasts = pl.concat(contrast_frames, how="diagonal")
    verdicts = planned_verdicts(contrasts=contrasts)
    leaderboard_table = pl.concat(list(boards.values()), how="diagonal")
    leaderboard_table.write_parquet(paths.leaderboard)
    contrasts.write_parquet(paths.contrasts)
    worst = stack_nonempty(frames=worst_frames)
    aerosol_conditions = stack_nonempty(frames=condition_frames)
    probabilistic = stack_nonempty(frames=probabilistic_frames)
    splits = stack_nonempty(frames=split_frames)
    for frame, path in (
        (splits, paths.splits),
        (worst, paths.worst_days),
        (aerosol_conditions, paths.aerosol_conditions),
        (probabilistic, paths.probabilistic),
    ):
        if not frame.is_empty():
            frame.write_parquet(path)
    report = render_report(
        variant=variant,
        through_rung=through_rung,
        boards=boards,
        contrasts=contrasts,
        verdicts=verdicts,
        arms_json=arms_json,
        build_notes=checks_path(through_rung=through_rung, variant=variant).read_text(),
        worst=worst,
        aerosol_conditions=aerosol_conditions,
        probabilistic=probabilistic,
        splits=splits,
        dataset=dataset,
    )
    paths.report.write_text(report)
    _LOG.info("wrote %s", paths.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
