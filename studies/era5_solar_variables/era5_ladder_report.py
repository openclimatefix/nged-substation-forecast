"""Score the ERA5 variable ladder from its saved losses, and write the report and the tables.

One-off throwaway script for the study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1097>. It reads the per-row losses
that `era5_ladder_fit.py` saved and refits nothing. It writes `report.md` and four tables
(`leaderboard.parquet`, `contrasts.parquet`, `splits.parquet`, `worst_days.parquet`) into
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
    METRIC,
    NEAR_LINE_SHARE,
    NEGATIVE_CONTROL_ARM,
    PLANNED_CONTRASTS,
    PLANNED_RESAMPLES,
    POSITIVE_CONTROL_ARM,
    PRIMARY_SETTING,
    SENSITIVITY_SETTING,
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
    report_paths,
    results_path,
)
from studies.bootstrap import (
    MIN_MONTHS_FOR_INTERVAL,
    bootstrap_absolute,
    bootstrap_difference,
    bootstrap_difference_at_level,
)
from studies.correlation import pooled_correlation_interval
from studies.era5_ladder import (
    CLEAR_SKY_INDEX_THRESHOLDS,
    RUNGS,
    RungType,
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

WORST_DAY_COUNT: Final[int] = 20
"""How many of the minimal arm's worst farm-days the report lists."""

SPLIT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (("g1", "g0"), ("g2", "g0"), ("g9", "g0"))
"""The (treatment, reference) contrasts the regime and season splits show."""

NO_GAIN_BEYOND_EFFECT: Final[str] = "rules out a gain larger than the smallest effect"
GAIN_NOT_EXCLUDED: Final[str] = "does not rule out a gain larger than the smallest effect"
IMPROVES: Final[str] = "improves; a gain larger than the smallest effect is not ruled out"
IMPROVES_BY_LESS_THAN_EFFECT: Final[str] = "improves, by less than the smallest effect"
WORSENS: Final[str] = "worsens"
UNRESOLVED: Final[str] = "unresolved"

RULES_OUT_LARGE_GAIN: Final[frozenset[str]] = frozenset(
    {NO_GAIN_BEYOND_EFFECT, IMPROVES_BY_LESS_THAN_EFFECT, WORSENS}
)
"""The verdicts that each rule out a gain larger than the smallest effect."""

LARGE_GAIN_POSSIBLE: Final[frozenset[str]] = frozenset({IMPROVES, GAIN_NOT_EXCLUDED})
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
        The shared verdict if the two are equal. Otherwise `rules out a gain larger than the
        smallest effect` if both rule it out, `does not rule out a gain larger than the smallest
        effect` if both leave it possible, and `unresolved` if they disagree on that.
    """
    if primary == sensitivity:
        return primary
    pair = {primary, sensitivity}
    if pair <= RULES_OUT_LARGE_GAIN:
        return NO_GAIN_BEYOND_EFFECT
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
        `improves` if the interval is below zero and reaches past the smallest effect, and
        `improves, by less than the smallest effect` if it is below zero and does not. `worsens`
        if it is above zero. Otherwise `rules out a gain larger than the smallest effect` if the
        lower bound is above minus the smallest effect, and `does not rule out a gain larger than
        the smallest effect` if it is not.
    """
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
    return {
        "target": target,
        "family": family,
        "label": label,
        "treatment": treatment,
        "reference": reference,
        "setting": setting,
        "planned": planned,
        "difference": interval["difference"],
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
    groupings: list[tuple[str, tuple[str, ...]]] = [
        ("regime_cams", ("regime_cams",)),
        ("season", ("season",)),
        ("regime_cams_by_season", ("season", "regime_cams")),
    ]
    if "regime_era5" in in_setting.columns:
        groupings.append(("regime_era5", ("regime_era5",)))
    for split, columns in groupings:
        known = in_setting.filter(pl.all_horizontal(pl.col(name).is_not_null() for name in columns))
        for key, subset in known.group_by(*columns, maintain_order=True):
            group = " / ".join(str(part) for part in key)
            for treatment, reference in contrasts:
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
            )
        )
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
    context = [name for name in ("mean_sd", "mean_tcc") if name in worst.columns]
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
                    "n/a" if row[name] is None else f"{row[name]:.{PRINT_DECIMALS}f}"
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
    return "\n".join(parts) + "\n"


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
        aerosol_key = FitKey(
            variant=variant, through_rung=through_rung, target=target, view="aerosol_rows"
        )
        if results_path(key=aerosol_key).exists():
            aerosol_losses = read_losses(key=aerosol_key)
            contrast_frames.append(all_contrasts(losses=aerosol_losses, target=target))
    contrasts = pl.concat(contrast_frames, how="diagonal")
    verdicts = planned_verdicts(contrasts=contrasts)
    leaderboard_table = pl.concat(list(boards.values()), how="diagonal")
    leaderboard_table.write_parquet(paths.leaderboard)
    contrasts.write_parquet(paths.contrasts)
    if split_frames:
        pl.concat(split_frames, how="diagonal").write_parquet(paths.splits)
    worst = pl.concat(worst_frames, how="diagonal") if worst_frames else pl.DataFrame()
    if not worst.is_empty():
        worst.write_parquet(paths.worst_days)
    report = render_report(
        variant=variant,
        through_rung=through_rung,
        boards=boards,
        contrasts=contrasts,
        verdicts=verdicts,
        arms_json=arms_json,
        build_notes=checks_path(through_rung=through_rung, variant=variant).read_text(),
        worst=worst,
    )
    paths.report.write_text(report)
    _LOG.info("wrote %s", paths.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
