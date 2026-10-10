"""Score the IFS variable ladder from its saved losses, and write the report and the tables.

One-off throwaway script for the study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1137>. It reads the per-row losses
that `ifs_ladder_fit.py` saved and refits nothing. It writes `report.md` and seven parquet tables
(`leaderboard`, `contrasts`, `splits`, `farms`, `planned_verdicts`, `priority_list`, and
`decisions`) into `data/studies/per_study/ifs_solar_variables/results/`. The chart script reads
the tables. Every number on the page comes from `report.md`. With `--width-preview-only` it reads
only the ERA5 study's losses and writes `width_preview.md`, so that the width a planned interval
would have is stated before any IFS fit exists.

**Planned contrasts are P1 to P5, written into the plan before any result existed** (see
`ifs_ladder_arms.PLANNED_CONTRASTS`). P1 to P3 are pooled over lead days 1 to 3 on the ladder
rows, and P4 and P5 are pooled over lead days 1 to 3 on the blend rows. Each is a treatment's error
minus a reference's, so a negative difference is a gain. Each is read at both hyperparameter
settings from a 99% interval (the 5% family-wise level shared by five contrasts) with 10,000 month
resamples, and its verdict is `gain`, `no gain`, or `unresolved` by the formula in
`studies.ifs_decisions.contrast_verdict`. Each is also read against the negative control that has
the treatment's columns, and a gain stands only if it is also a gain over the control.

**Pooling lead days.** The lead days are separate XGBoost models. The pooled difference is the mean
of the differences over the farm-hours that all of lead days 1, 2, and 3 hold, so it is the
equal-weight mean of the three lead days' differences. One month resample is shared by the three
lead days because each lead day's rows get a `site` key of their own (`A-L1`, `A-L2`, and so on;
`studies.ifs_decisions.stack_lead_days`). A pooled scope raises if any of the three lead days is
missing.

**Every other number is exploratory**: the single lead days, the rungs one by one, the contrasts
against the production reference, the CAMS target, the splits, the drop-one runs, and the
comparison with ERA5. The report counts the exploratory intervals.

**The decisions are formulas** in `studies.ifs_decisions`: the third decision's outcome, the second
feed, and the priority list's evidence classes. The ERA5 study's P4 numbers are read from its
results table. Open-data availability is a constant in `studies.ifs_ladder`.

**A farm's dates are not printed.** The report names only farm letters, months, and hours of day.

Run it with `uv run python studies/ifs_solar_variables/ifs_ladder_report.py`.
"""

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Final, NamedTuple, cast

import polars as pl
from ifs_ladder_arms import (
    ADJUSTED_LEVEL_PERCENT,
    BLEND_ARM,
    BLEND_LEAD_DAYS,
    CONTROL_CLOUD_COVER_ARM,
    CONTROL_CLOUD_LAYERS_ARM,
    CONTROL_CONVECTION_ARM,
    CONTROL_DIRECT_ARM,
    CONTROL_HUMIDITY_ARM,
    CONTROL_LATER_GROUPS_ARM,
    CONTROL_PARTNER_FULL_ARM,
    CONTROL_PARTNER_MINIMAL_ARM,
    DROP_ONE_LEAD_DAYS,
    DROP_PREFIX,
    ERA5_COMPARISON_LEAD_DAY,
    ERA5_LOSSES_NAME,
    ERA5_P4_RESULTS_NAME,
    ERA5_PREFIX,
    F6_ARM,
    METRIC,
    PLANNED_CONTRAST_COUNT,
    PLANNED_CONTRASTS,
    PLANNED_LEAD_DAYS,
    PLANNED_RESAMPLES,
    POSITIVE_CONTROL_ARM,
    PRIMARY_SETTING,
    PRODUCTION_ARM,
    SENSITIVITY_SETTING,
    SMALLEST_EFFECT,
    FitKey,
    PlannedContrast,
    TargetType,
    arms_path,
    checks_path,
    lead_day_dataset_path,
    report_paths,
    results_path,
    width_preview_path,
)
from ifs_ladder_days import WEATHER_LEAD_DAYS, chosen_days, month_lines
from studies.bootstrap import (
    MIN_MONTHS_FOR_INTERVAL,
    N_BOOTSTRAP_RESAMPLES,
    bootstrap_absolute,
    bootstrap_difference,
    bootstrap_difference_at_level,
    paired_differences,
)
from studies.cross_validation import (
    N_FOLDS,
    PRIMARY_HYPER_PARAMETERS,
    SEEDS,
    SENSITIVITY_HYPER_PARAMETERS,
)
from studies.era5_ladder import (
    TOTAL_CLOUD_COVER_THRESHOLDS,
    raise_unless_same_rows,
    season_of_month,
    sky_regime_from_cloud_cover,
)
from studies.guards import refuse_to_overwrite
from studies.ifs_decisions import (
    NEAR_LINE_SHARE,
    SETTINGS_THAT_MUST_AGREE,
    EvidenceClassType,
    VerdictType,
    combine_verdicts,
    contrast_verdict,
    drop_one_matters,
    evidence_class,
    exploratory_gain,
    gain_after_control,
    is_near_line,
    priority_order,
    second_feed_recommended,
    second_forecast_or_mars_variables,
    stack_lead_days,
)
from studies.ifs_ladder import (
    GROUP_NAMES,
    OPEN_DATA_AVAILABILITY,
    OPEN_DATA_CHECK_NOTE,
    PRODUCTION_FEED_VARIABLES,
    RUNG_ADDITIONS,
    RUNGS,
    RungType,
)
from studies.ifs_lead_days import DROPPED_MONTHS, LAST_ERA_MONTH_EXCLUSIVE, LEAD_DAYS
from studies.sources import ERA5_LADDER_RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("ifs_ladder_report")

PERCENTAGE_POINTS: Final[float] = 100.0
"""Turns a fraction of capacity into percentage points."""

PRINT_DECIMALS: Final[int] = 3
"""How many decimals a table prints, so a chart and the page quote the same digits."""

SPLIT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (("f2", "f0"), (F6_ARM, "f0"))
"""The (treatment, reference) contrasts the regime, season, and hour-of-day splits show."""

LADDER_STEPS: Final[tuple[tuple[str, str], ...]] = tuple(
    (upper, lower) for lower, upper in pairwise(RUNGS)
)
"""Each rung minus the rung below it."""

AGAINST_MINIMAL: Final[tuple[tuple[str, str], ...]] = tuple((rung, "f0") for rung in RUNGS[1:])
"""Each rung minus the minimal set."""

AGAINST_PRODUCTION: Final[tuple[tuple[str, str], ...]] = (
    ("f1", PRODUCTION_ARM),
    ("f2", PRODUCTION_ARM),
    (F6_ARM, PRODUCTION_ARM),
    (PRODUCTION_ARM, "f0"),
)
"""The contrasts against the production reference, all exploratory."""

CAMS_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (("f2", "f0"), (F6_ARM, "f0"), (F6_ARM, "f2"))
"""The contrasts on the exploratory CAMS target."""

BLEND_EXPLORATORY: Final[tuple[tuple[str, str], ...]] = (
    (BLEND_ARM, "f0"),
    (F6_ARM, "f0"),
    (F6_ARM, BLEND_ARM),
)
"""The blend rows' exploratory contrasts: the partner, the IFS variables, and one against other."""

CONTROL_BASES: Final[tuple[tuple[str, str, str], ...]] = (
    (CONTROL_CLOUD_COVER_ARM, "f0", "ladder"),
    (CONTROL_CLOUD_LAYERS_ARM, "f1", "ladder"),
    (CONTROL_LATER_GROUPS_ARM, "f2", "ladder"),
    (CONTROL_DIRECT_ARM, "f2", "ladder"),
    (CONTROL_HUMIDITY_ARM, "f2", "ladder"),
    (CONTROL_CONVECTION_ARM, "f2", "ladder"),
    (POSITIVE_CONTROL_ARM, "f2", "ladder"),
    (CONTROL_PARTNER_MINIMAL_ARM, "f0", "blend"),
    (CONTROL_PARTNER_FULL_ARM, F6_ARM, "blend"),
)
"""Each control, the arm it pads, and the rows it is read on."""

POSITIVE_CONTROL_MINIMUM_GAIN: Final[float] = 0.01
"""The positive control must lower the error by more than this fraction of capacity (1 point)."""

GROUP_STEPS: Final[dict[RungType, tuple[str, str]]] = {
    "f1": ("f1", "f0"),
    "f2": ("f2", "f1"),
    "f3": ("f3", "f2"),
    "f4": ("f4", "f3"),
    "f5": ("f5", "f4"),
    "f6": ("f6", "f5"),
}
"""Each group's own ladder step: its rung minus the rung below."""

PLANNED_COVERS_SEVERAL_GROUPS: Final[frozenset[RungType]] = frozenset({"f3", "f4", "f5", "f6"})
"""The groups that contrast P3 covers together, so each needs its own drop-one run to agree."""

REGIME_COLUMN: Final[str] = "regime_era5"
"""The regime column of the splits, from ERA5 total cloud cover."""

ERA5_ADJUSTED_LEVEL_PERCENT: Final[float] = 99.5
"""The coverage of the ERA5 study's adjusted interval, which is Bonferroni-adjusted for ten."""

PLAN_ERA5_P4_PRIMARY_UPPER: Final[float] = -0.000827
"""The upper bound of the ERA5 study's P4 adjusted interval at the main setting in the plan."""

WIDTH_WARNING: Final[float] = 0.002
"""A planned interval wider than this fraction of capacity (0.2 points) triggers a caveat."""

FIRST_IFS_MONTH: Final[str] = "2024-03"
"""The first month with IFS runs, a part-month."""

GROUP_CONTROLS: Final[dict[str, str]] = {
    "f3": CONTROL_DIRECT_ARM,
    "f4": CONTROL_HUMIDITY_ARM,
    "f5": CONTROL_CONVECTION_ARM,
}
"""The control of each exploratory rung that has one of its own; each pads `f2`."""


# ---------------------------------------------------------------------------------------------
# Reading the fits
# ---------------------------------------------------------------------------------------------


def read_losses(*, key: FitKey) -> pl.DataFrame | None:
    """Read one fit's per-row losses with a `lead_day` column, or `None` if it was not fitted.

    Args:
        key: Which fit.

    Returns:
        The losses, or `None` if the fit has no losses file.
    """
    path = results_path(key=key)
    if not path.exists():
        return None
    return pl.read_parquet(path).with_columns(lead_day=pl.lit(key.lead_day, dtype=pl.Int8))


class Fits:
    """Every fit's losses and arms, keyed by view, target, and lead day."""

    def __init__(self) -> None:
        """Read every fit that exists."""
        self.losses: dict[tuple[str, str, int], pl.DataFrame] = {}
        self.arms: dict[tuple[str, str, int], dict[str, object]] = {}
        self.datasets: dict[int, pl.DataFrame] = {}
        for view, days in (("ladder", LEAD_DAYS), ("blend", BLEND_LEAD_DAYS)):
            for target in ("pv", "cams"):
                for lead_day in days:
                    key = FitKey(view=view, lead_day=lead_day, target=target)
                    losses = read_losses(key=key)
                    if losses is not None:
                        self.losses[(view, target, lead_day)] = losses
                        self.arms[(view, target, lead_day)] = json.loads(
                            arms_path(key=key).read_text()
                        )
        for lead_day in LEAD_DAYS:
            if lead_day_dataset_path(lead_day=lead_day).exists():
                self.datasets[lead_day] = pl.read_parquet(lead_day_dataset_path(lead_day=lead_day))

    def get(self, *, view: str, target: str, lead_days: Sequence[int]) -> dict[int, pl.DataFrame]:
        """Return the losses of the lead days that were fitted, by lead day."""
        return {
            day: self.losses[(view, target, day)]
            for day in lead_days
            if (view, target, day) in self.losses
        }

    def pooled(self, *, view: str, target: str, lead_days: Sequence[int]) -> pl.DataFrame | None:
        """Return the stacked losses of the lead days, or `None` if none was fitted.

        Args:
            view: `ladder` or `blend`.
            target: The target.
            lead_days: The lead days to stack.

        Returns:
            `stack_lead_days`'s result for the lead days, or `None` if none was fitted.

        Raises:
            ValueError: If some but not all of the lead days were fitted, because a pooled
                contrast over a subset would be published as the pooled contrast.
        """
        fitted = self.get(view=view, target=target, lead_days=lead_days)
        if not fitted:
            return None
        missing = sorted(set(lead_days) - set(fitted))
        if missing:
            msg = (
                f"{view} {target}: lead days {missing} are not fitted, so lead days "
                f"{list(lead_days)} cannot be pooled"
            )
            raise ValueError(msg)
        return stack_lead_days(losses_by_lead_day=fitted)


def has_arms(*, losses: pl.DataFrame, setting: str, arms: Sequence[str]) -> bool:
    """Say whether every named arm was fitted at a setting, at every lead day the losses hold."""
    in_setting = losses.filter(pl.col("setting") == setting)
    by_day = in_setting.group_by("lead_day").agg(pl.col("arm").unique())
    return by_day.height == losses["lead_day"].n_unique() and all(
        set(arms) <= set(found) for found in by_day["arm"].to_list()
    )


# ---------------------------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------------------------


def contrast_row(
    *,
    losses: pl.DataFrame,
    target: TargetType,
    family: str,
    label: str,
    scope: str,
    treatment: str,
    reference: str,
    setting: str,
    planned: bool,
) -> dict[str, object]:
    """Return one contrast's difference with its 95% interval and, if planned, its 99% interval.

    The difference is the treatment's error minus the reference's, so a negative value is a gain.

    Args:
        losses: Per-row losses holding both arms, possibly stacked over lead days.
        target: The target.
        family: What kind of contrast it is, such as `planned` or `drop_one`.
        label: A short name for the contrast, such as `P1`.
        scope: `pooled 1-3`, `lead day 5`, or similar.
        treatment: The arm whose error is compared.
        reference: The arm it is compared against.
        setting: The hyperparameter setting.
        planned: Whether the plan named the contrast before any result existed.

    Returns:
        The contrast's table row.
    """
    in_setting = losses.filter(pl.col("setting") == setting)
    raise_unless_same_rows(losses=in_setting, arms=[treatment, reference])
    interval = bootstrap_difference(
        losses=in_setting, treatment=treatment, reference=reference, metric=METRIC
    )
    smallest = SMALLEST_EFFECT[target]
    row: dict[str, object] = {
        "target": target,
        "family": family,
        "label": label,
        "scope": scope,
        "treatment": treatment,
        "reference": reference,
        "setting": setting,
        "planned": planned,
        "difference": interval["difference"],
        "reference_mean": float(
            in_setting.filter(pl.col("arm") == reference).select(pl.col(METRIC).mean()).item()
        ),
        "lower_95": interval["lower_95"],
        "upper_95": interval["upper_95"],
        "lower_adjusted": None,
        "upper_adjusted": None,
        "verdict_adjusted": None,
        "verdict_95": contrast_verdict(
            lower=interval["lower_95"], upper=interval["upper_95"], smallest_effect=smallest
        ),
        "near_line": is_near_line(lower=interval["lower_95"], upper=interval["upper_95"]),
        "seed_spread": interval["seed_spread"],
        "n_rows": interval["n_rows"],
        "n_months": interval["n_months"],
    }
    if planned:
        lower, upper = bootstrap_difference_at_level(
            losses=in_setting,
            treatment=treatment,
            reference=reference,
            metric=METRIC,
            level=ADJUSTED_LEVEL_PERCENT,
            n_resamples=PLANNED_RESAMPLES,
        )
        row |= {
            "lower_adjusted": lower,
            "upper_adjusted": upper,
            "verdict_adjusted": contrast_verdict(
                lower=lower, upper=upper, smallest_effect=smallest
            ),
        }
    return row


def settings_in(*, losses: pl.DataFrame, arms: Sequence[str]) -> list[str]:
    """Return the settings at which every named arm was fitted."""
    return [
        setting
        for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING)
        if has_arms(losses=losses, setting=setting, arms=arms)
    ]


def add_pairs(
    *,
    rows: list[dict[str, object]],
    losses: pl.DataFrame | None,
    target: TargetType,
    family: str,
    scope: str,
    pairs: Sequence[tuple[str, str]],
    planned: bool = False,
    labels: Mapping[tuple[str, str], str] | None = None,
) -> None:
    """Append the contrast rows of some (treatment, reference) pairs at every setting fitted.

    Args:
        rows: The list to append to.
        losses: The losses to read, or `None` to add nothing.
        target: The target.
        family: The contrast family.
        scope: The scope label.
        pairs: The (treatment, reference) pairs.
        planned: Whether these are the planned contrasts.
        labels: A label for each pair, if not `treatment - reference`.
    """
    if losses is None:
        return
    for treatment, reference in pairs:
        for setting in settings_in(losses=losses, arms=[treatment, reference]):
            label = (labels or {}).get((treatment, reference), f"{treatment} - {reference}")
            _LOG.info("%s %s: %s at %s", target, scope, label, setting)
            rows.append(
                contrast_row(
                    losses=losses,
                    target=target,
                    family=family,
                    label=label,
                    scope=scope,
                    treatment=treatment,
                    reference=reference,
                    setting=setting,
                    planned=planned,
                )
            )


def planned_rows(*, fits: Fits) -> list[dict[str, object]]:
    """Compute the planned contrasts and their controls, pooled over lead days 1 to 3.

    Args:
        fits: Every fit.

    Returns:
        The rows for `P1` to `P5`, each treatment minus reference, and each treatment minus its
        negative control (family `planned_control`), at every setting fitted.
    """
    rows: list[dict[str, object]] = []
    for planned in PLANNED_CONTRASTS:
        losses = fits.pooled(view=planned.view, target="pv", lead_days=PLANNED_LEAD_DAYS)
        add_pairs(
            rows=rows,
            losses=losses,
            target="pv",
            family="planned",
            scope="pooled 1-3",
            pairs=[(planned.treatment, planned.reference)],
            planned=True,
            labels={(planned.treatment, planned.reference): planned.label},
        )
        add_pairs(
            rows=rows,
            losses=losses,
            target="pv",
            family="planned_control",
            scope="pooled 1-3",
            pairs=[(planned.treatment, planned.control)],
            planned=True,
            labels={(planned.treatment, planned.control): f"{planned.label} vs control"},
        )
    return rows


def exploratory_rows(*, fits: Fits) -> list[dict[str, object]]:
    """Compute every exploratory contrast on the output target and the CAMS target.

    Args:
        fits: Every fit.

    Returns:
        The rows, each at 95% only.
    """
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, Sequence[int]]] = [
        ("pooled 1-3", PLANNED_LEAD_DAYS),
        *((f"lead day {day}", (day,)) for day in LEAD_DAYS),
    ]
    for scope, days in scopes:
        ladder = fits.pooled(view="ladder", target="pv", lead_days=days)
        add_pairs(
            rows=rows, losses=ladder, target="pv", family="step", scope=scope, pairs=LADDER_STEPS
        )
        add_pairs(
            rows=rows,
            losses=ladder,
            target="pv",
            family="against_f0",
            scope=scope,
            pairs=AGAINST_MINIMAL,
        )
        add_pairs(
            rows=rows,
            losses=ladder,
            target="pv",
            family="against_production",
            scope=scope,
            pairs=AGAINST_PRODUCTION,
        )
        add_pairs(
            rows=rows,
            losses=ladder,
            target="pv",
            family="control",
            scope=scope,
            pairs=[(control, base) for control, base, view in CONTROL_BASES if view == "ladder"],
        )
        if set(days) <= set(DROP_ONE_LEAD_DAYS):
            add_pairs(
                rows=rows,
                losses=ladder,
                target="pv",
                family="drop_one",
                scope=scope,
                pairs=[(f"{DROP_PREFIX}{rung}", F6_ARM) for rung in RUNGS[1:]],
            )
        add_pairs(
            rows=rows,
            losses=fits.pooled(view="ladder", target="cams", lead_days=days),
            target="cams",
            family="cams",
            scope=scope,
            pairs=CAMS_CONTRASTS,
        )
        if set(days) <= set(BLEND_LEAD_DAYS):
            blend = fits.pooled(view="blend", target="pv", lead_days=days)
            add_pairs(
                rows=rows,
                losses=blend,
                target="pv",
                family="blend",
                scope=scope,
                pairs=BLEND_EXPLORATORY,
            )
            add_pairs(
                rows=rows,
                losses=blend,
                target="pv",
                family="control",
                scope=scope,
                pairs=[(control, base) for control, base, view in CONTROL_BASES if view == "blend"],
            )
    comparison = fits.pooled(view="ladder", target="pv", lead_days=(ERA5_COMPARISON_LEAD_DAY,))
    add_pairs(
        rows=rows,
        losses=comparison,
        target="pv",
        family="era5",
        scope=f"lead day {ERA5_COMPARISON_LEAD_DAY}",
        pairs=[(f"{ERA5_PREFIX}{rung}", f"{ERA5_PREFIX}g0") for rung in ("g1", "g2", "g9")],
    )
    return rows


def era5_p4_rows() -> list[dict[str, object]]:
    """Return the ERA5 study's P4 rows from its results table, for the third decision.

    Returns:
        One row per hyperparameter setting, in the columns of the contrast table. The scope is
        `era5 study`, and the intervals are the ERA5 study's own.
    """
    table = pl.read_parquet(ERA5_LADDER_RESULTS_DIR / ERA5_P4_RESULTS_NAME).filter(
        (pl.col("label") == "P4") & (pl.col("target") == "pv") & pl.col("planned")
    )
    primary_upper = table.filter(pl.col("setting") == PRIMARY_SETTING)["upper_adjusted"].to_list()
    if not primary_upper or abs(primary_upper[0] - PLAN_ERA5_P4_PRIMARY_UPPER) > 5e-7:
        _LOG.warning(
            "the ERA5 study's P4 primary upper bound is %s, not the plan's %s: the plan's text "
            "about the third decision must be rechecked",
            primary_upper,
            PLAN_ERA5_P4_PRIMARY_UPPER,
        )
    return [
        {
            "target": "pv",
            "family": "era5_p4",
            "label": "ERA5 P4",
            "scope": "era5 study",
            "treatment": row["treatment"],
            "reference": row["reference"],
            "setting": row["setting"],
            "planned": True,
            "difference": row["difference"],
            "reference_mean": row["reference_mean"],
            "lower_95": row["lower_95"],
            "upper_95": row["upper_95"],
            "lower_adjusted": row["lower_adjusted"],
            "upper_adjusted": row["upper_adjusted"],
            "verdict_adjusted": None,
            "verdict_95": None,
            "near_line": False,
            "seed_spread": row["seed_spread"],
            "n_rows": row["n_rows"],
            "n_months": row["n_months"],
        }
        for row in table.iter_rows(named=True)
    ]


def one_row(
    *,
    contrasts: pl.DataFrame,
    family: str,
    treatment: str,
    reference: str,
    setting: str,
    scope: str = "pooled 1-3",
    target: str = "pv",
) -> dict[str, object] | None:
    """Return the one contrast row that matches, or `None` if the fit was not run.

    Args:
        contrasts: The contrast table.
        family: The contrast family.
        treatment: The treatment arm.
        reference: The reference arm.
        setting: The hyperparameter setting.
        scope: The scope label.
        target: The target.

    Returns:
        The row.

    Raises:
        ValueError: If more than one row matches.
    """
    found = contrasts.filter(
        (pl.col("family") == family)
        & (pl.col("treatment") == treatment)
        & (pl.col("reference") == reference)
        & (pl.col("setting") == setting)
        & (pl.col("scope") == scope)
        & (pl.col("target") == target)
    )
    if found.height > 1:
        msg = f"{found.height} rows match {family} {treatment} {reference} {setting} {scope}"
        raise ValueError(msg)
    return found.row(0, named=True) if found.height else None


def settings_verdict(
    *, contrasts: pl.DataFrame, family: str, treatment: str, reference: str
) -> VerdictType:
    """Combine a planned contrast's 99% verdicts at the two settings.

    Args:
        contrasts: The contrast table.
        family: `planned` or `planned_control`.
        treatment: The treatment arm.
        reference: The reference arm or control.

    Returns:
        The combined verdict, `unresolved` if a setting was not fitted.
    """
    verdicts: list[VerdictType] = []
    for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING):
        row = one_row(
            contrasts=contrasts,
            family=family,
            treatment=treatment,
            reference=reference,
            setting=setting,
        )
        if row is not None:
            verdicts.append(row["verdict_adjusted"])  # ty: ignore[invalid-argument-type]
    return combine_verdicts(verdicts=verdicts)


def planned_verdict_table(*, contrasts: pl.DataFrame) -> pl.DataFrame:
    """Combine each planned contrast's verdicts at the two settings, then against its control.

    Args:
        contrasts: The contrast table.

    Returns:
        One row per planned contrast with both settings' verdicts, the combined verdict, the
        verdict against the control, and the final verdict after the control.
    """
    rows: list[dict[str, object]] = []
    for planned in PLANNED_CONTRASTS:
        by_setting = {
            setting: one_row(
                contrasts=contrasts,
                family="planned",
                treatment=planned.treatment,
                reference=planned.reference,
                setting=setting,
            )
            for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING)
        }
        if by_setting[PRIMARY_SETTING] is None:
            continue
        combined = settings_verdict(
            contrasts=contrasts,
            family="planned",
            treatment=planned.treatment,
            reference=planned.reference,
        )
        versus_control = settings_verdict(
            contrasts=contrasts,
            family="planned_control",
            treatment=planned.treatment,
            reference=planned.control,
        )
        rows.append(
            {
                "label": planned.label,
                "treatment": planned.treatment,
                "reference": planned.reference,
                "control": planned.control,
                "primary": by_setting[PRIMARY_SETTING]["verdict_adjusted"],  # ty: ignore[not-subscriptable]
                "sensitivity": None
                if by_setting[SENSITIVITY_SETTING] is None
                else by_setting[SENSITIVITY_SETTING]["verdict_adjusted"],  # ty: ignore[not-subscriptable]
                "combined": combined,
                "versus_control": versus_control,
                "final": gain_after_control(
                    versus_reference=combined, versus_control=versus_control
                ),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def final_verdict(*, verdicts: pl.DataFrame, label: str) -> VerdictType:
    """Return a planned contrast's final verdict, `unresolved` if it was not computed."""
    found = verdicts.filter(pl.col("label") == label) if not verdicts.is_empty() else verdicts
    return found["final"][0] if found.height else "unresolved"


# ---------------------------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------------------------


def pooled_difference(
    *, contrasts: pl.DataFrame, family: str, treatment: str, reference: str
) -> float | None:
    """Return a contrast's pooled difference at the primary setting, or `None` if not computed."""
    row = one_row(
        contrasts=contrasts,
        family=family,
        treatment=treatment,
        reference=reference,
        setting=PRIMARY_SETTING,
    )
    return None if row is None else float(row["difference"])  # ty: ignore[invalid-argument-type]


def step_uppers(*, contrasts: pl.DataFrame, treatment: str, reference: str) -> list[float]:
    """Return a step's 95% upper bounds at every setting and planned lead day that was fitted.

    Args:
        contrasts: The contrast table.
        treatment: The rung.
        reference: The rung below it.

    Returns:
        Up to six bounds: two settings times lead days 1 to 3. A missing fit is left out, so the
        caller must compare the length with six.
    """
    uppers: list[float] = []
    for day in PLANNED_LEAD_DAYS:
        for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING):
            row = one_row(
                contrasts=contrasts,
                family="step",
                treatment=treatment,
                reference=reference,
                setting=setting,
                scope=f"lead day {day}",
            )
            if row is not None:
                uppers.append(float(row["upper_95"]))  # ty: ignore[invalid-argument-type]
    return uppers


def gain_interval(
    *, contrasts: pl.DataFrame, rung: RungType, evidence: EvidenceClassType
) -> tuple[float | None, float | None, str]:
    """Return the interval of the quantity that ranks a group, positive for a gain, and its level.

    Total cloud and the cloud layers read their planned contrast's 99% interval. A group in the
    exploratory class reads the 95% interval of its pooled ladder step. Every other group reads the
    95% interval of its drop-one run, which is exploratory.

    Args:
        contrasts: The contrast table.
        rung: The group's rung.
        evidence: The group's evidence class.

    Returns:
        The lower and upper bounds of the gain, and `99%` or `95%`. Bounds are `None` if the
        contrast was not computed.
    """
    if rung in ("f1", "f2"):
        treatment, reference = GROUP_STEPS[rung]
        row = one_row(
            contrasts=contrasts,
            family="planned",
            treatment=treatment,
            reference=reference,
            setting=PRIMARY_SETTING,
        )
        level = "99%"
        if row is None:
            return None, None, level
        return -float(row["upper_adjusted"]), -float(row["lower_adjusted"]), level  # ty: ignore[invalid-argument-type]
    if evidence == "exploratory gain":
        treatment, reference = GROUP_STEPS[rung]
        step = one_row(
            contrasts=contrasts,
            family="step",
            treatment=treatment,
            reference=reference,
            setting=PRIMARY_SETTING,
        )
        if step is None:
            return None, None, "95%"
        return -float(step["upper_95"]), -float(step["lower_95"]), "95%"  # ty: ignore[invalid-argument-type]
    row = one_row(
        contrasts=contrasts,
        family="drop_one",
        treatment=f"{DROP_PREFIX}{rung}",
        reference=F6_ARM,
        setting=PRIMARY_SETTING,
    )
    if row is None:
        return None, None, "95%"
    return float(row["lower_95"]), float(row["upper_95"]), "95%"  # ty: ignore[invalid-argument-type]


def group_evidence(*, contrasts: pl.DataFrame, verdicts: pl.DataFrame) -> pl.DataFrame:
    """Build the priority list's rows: each group's evidence class, gain, and availability.

    A group's step is read against a negative control with as many added columns as the step. `f1`
    and `f2` have the controls of P1 and P2. `f3`, `f4`, and `f5` have a control that pads `f2`
    with only that group's permuted variables. `f6` has the P3 control, which pads `f2` with all
    the variables of `f3` to `f6` (14 columns against the step's 7), the only control the ladder
    holds for it.

    Args:
        contrasts: The contrast table.
        verdicts: The planned verdict table.

    Returns:
        One row per group, in priority order, with its class, pooled gain, and the open-data
        status of its variables.
    """
    control_difference = pooled_difference(
        contrasts=contrasts,
        family="control",
        treatment=CONTROL_LATER_GROUPS_ARM,
        reference="f2",
    )
    matching_control = {
        "f1": (CONTROL_CLOUD_COVER_ARM, "f0"),
        "f2": (CONTROL_CLOUD_LAYERS_ARM, "f1"),
        **{rung: (control, "f2") for rung, control in GROUP_CONTROLS.items()},
    }
    classes: dict[str, EvidenceClassType] = {}
    gains: dict[str, float] = {}
    details: dict[str, dict[str, object]] = {}
    for rung in RUNGS[1:]:
        treatment, reference = GROUP_STEPS[rung]
        step = pooled_difference(
            contrasts=contrasts, family="step", treatment=treatment, reference=reference
        )
        drop = pooled_difference(
            contrasts=contrasts,
            family="drop_one",
            treatment=f"{DROP_PREFIX}{rung}",
            reference=F6_ARM,
        )
        control_arm, control_base = matching_control.get(rung, (CONTROL_LATER_GROUPS_ARM, "f2"))
        group_control = pooled_difference(
            contrasts=contrasts, family="control", treatment=control_arm, reference=control_base
        )
        planned_label = {"f1": "P1", "f2": "P2"}.get(rung, "P3")
        planned_verdict = final_verdict(verdicts=verdicts, label=planned_label)
        uppers = step_uppers(contrasts=contrasts, treatment=treatment, reference=reference)
        complete = len(uppers) == len(PLANNED_LEAD_DAYS) * 2
        has_exploratory = (
            complete
            and step is not None
            and group_control is not None
            and exploratory_gain(
                uppers=uppers,
                difference=step,
                control_difference=group_control,
                smallest_effect=SMALLEST_EFFECT["pv"],
            )
        )
        has_drop_one = (
            drop is not None
            and control_difference is not None
            and drop_one_matters(raise_in_error=drop, control_difference=control_difference)
        )
        classes[rung] = evidence_class(
            planned_verdict=planned_verdict if rung != "f2" else None,
            planned_needs_drop_one=rung in PLANNED_COVERS_SEVERAL_GROUPS,
            has_exploratory_gain=bool(has_exploratory),
            has_drop_one_gain=bool(has_drop_one),
        )
        if classes[rung] == "exploratory gain" and step is not None:
            gains[rung] = -step
        elif (
            classes[rung] in ("planned gain", "drop-one gain")
            and rung in PLANNED_COVERS_SEVERAL_GROUPS
            and drop is not None
        ):
            gains[rung] = drop
        else:
            gains[rung] = -step if step is not None else 0.0
        interval_lower, interval_upper, level = gain_interval(
            contrasts=contrasts,
            rung=rung,
            evidence=classes[rung],
        )
        details[rung] = {
            "gain_lower": interval_lower,
            "gain_upper": interval_upper,
            "gain_interval_level": level,
            "step_difference": step,
            "drop_one_raise": drop,
            "control_arm": control_arm,
            "control_difference": group_control,
            "f2_control_difference": control_difference,
            "step_bounds_found": len(uppers),
            "exploratory_gain": bool(has_exploratory),
            "drop_one_gain": bool(has_drop_one),
            "planned_verdict": planned_verdict,
        }
    rows = []
    for rank, name in enumerate(priority_order(classes=classes, gains=gains), start=1):
        rung = cast("RungType", name)
        variables = RUNG_ADDITIONS[rung]
        availability = [OPEN_DATA_AVAILABILITY[name] for name in variables]
        rows.append(
            {
                "rank": rank,
                "rung": rung,
                "group": GROUP_NAMES[rung],
                "evidence_class": classes[rung],
                "gain": gains[rung],
                "variables": ", ".join(variables),
                "carried": ", ".join(
                    name for name in variables if OPEN_DATA_AVAILABILITY[name] == "carried"
                ),
                "not_carried": ", ".join(
                    name for name in variables if OPEN_DATA_AVAILABILITY[name] == "not carried"
                ),
                "unchecked": ", ".join(
                    name for name in variables if OPEN_DATA_AVAILABILITY[name] == "unchecked"
                ),
                "in_open_data": "carried" in availability or "unchecked" in availability,
                "in_production_feed": ", ".join(
                    name for name in variables if name in PRODUCTION_FEED_VARIABLES
                ),
                **details[rung],
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


class ThirdDecision(NamedTuple):
    """The third decision's outcome and the numbers it rests on.

    Attributes:
        p5_final_verdict: P5's verdict after its control.
        era5_p4_uppers: The ERA5 study's P4 adjusted upper bounds, one per setting.
        outcome: The rule's outcome.
    """

    p5_final_verdict: VerdictType
    era5_p4_uppers: list[float]
    outcome: str


def third_decision(*, verdicts: pl.DataFrame, era5_p4: pl.DataFrame) -> ThirdDecision:
    """Apply the third decision's rule: a second forecast, the MARS-only variables, or neither.

    Args:
        verdicts: The planned verdict table.
        era5_p4: The ERA5 study's P4 rows.

    Returns:
        The outcome and the numbers it rests on.
    """
    p5 = final_verdict(verdicts=verdicts, label="P5")
    uppers = era5_p4.sort("setting")["upper_adjusted"].to_list()
    outcome = second_forecast_or_mars_variables(
        second_forecast_verdict=p5, mars_uppers=uppers, smallest_effect=SMALLEST_EFFECT["pv"]
    )
    return ThirdDecision(p5_final_verdict=p5, era5_p4_uppers=uppers, outcome=outcome)


def second_feed(*, evidence: pl.DataFrame, verdicts: pl.DataFrame) -> bool:
    """Apply the second decision's rule: whether the production forecast needs a second feed.

    The groups the open-data feed lacks are the cloud layers and direct radiation. The
    boundary-layer height is one variable of a group that also holds carried variables, so no
    drop-one run isolates it.

    Args:
        evidence: The priority list's rows.
        verdicts: The planned verdict table.

    Returns:
        Whether the second feed is recommended.
    """
    missing = [
        bool(evidence.filter(pl.col("rung") == rung)["drop_one_gain"][0]) for rung in ("f2", "f3")
    ]
    return second_feed_recommended(
        cloud_layers_verdict=final_verdict(verdicts=verdicts, label="P2"),
        missing_group_drop_one_gains=missing,
    )


# ---------------------------------------------------------------------------------------------
# Leaderboard, splits, and the width check
# ---------------------------------------------------------------------------------------------


def leaderboard_rows(*, fits: Fits) -> pl.DataFrame:
    """Return each arm's mean absolute error with a 95% interval at the primary setting.

    Each fit is scored on all its rows (`rows == "all"`), and the ladder's output target is scored
    again on the rows that every lead day holds (`rows == "common"`), so that lead days compare.

    Args:
        fits: Every fit.

    Returns:
        One row per (view, target, lead day, arm, rows).
    """
    rows: list[dict[str, object]] = []
    ladder = fits.get(view="ladder", target="pv", lead_days=LEAD_DAYS)
    common = None
    if len(ladder) == len(LEAD_DAYS):
        keys = [frame.select("site", "time").unique() for frame in ladder.values()]
        common = keys[0]
        for other in keys[1:]:
            common = common.join(other, on=["site", "time"])
    for (view, target, lead_day), losses in sorted(fits.losses.items()):
        primary = losses.filter(pl.col("setting") == PRIMARY_SETTING)
        scopes: list[tuple[str, pl.DataFrame]] = [("all", primary)]
        if common is not None and (view, target) == ("ladder", "pv"):
            scopes.append(("common", primary.join(common, on=["site", "time"])))
        for rows_name, scoped in scopes:
            for arm in sorted(scoped["arm"].unique().to_list()):
                interval = bootstrap_absolute(losses=scoped, arm=arm, metric=METRIC)
                rows.append(
                    {
                        "view": view,
                        "target": target,
                        "lead_day": lead_day,
                        "arm": arm,
                        "rows": rows_name,
                        "mae": interval["value"],
                        "mae_lower": interval["lower_95"],
                        "mae_upper": interval["upper_95"],
                        "seed_spread": interval["seed_spread"],
                        "n_rows": interval["n_rows"],
                        "n_months": interval["n_months"],
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def per_farm_rows(*, fits: Fits) -> pl.DataFrame:
    """Return each farm's mean absolute error for each ladder arm at the primary setting."""
    frames = [
        losses.filter(pl.col("setting") == PRIMARY_SETTING)
        .group_by("lead_day", "arm", "site")
        .agg(value=pl.col(METRIC).mean(), n_rows=pl.len())
        for (view, target, _), losses in sorted(fits.losses.items())
        if (view, target) == ("ladder", "pv")
    ]
    return pl.concat(frames).sort("lead_day", "arm", "site") if frames else pl.DataFrame()


def split_rows(*, fits: Fits) -> pl.DataFrame:
    """Compute the exploratory splits of contrasts by cloud regime, season, and hour of day.

    The rows are the output target's ladder rows at lead days 1 to 3, stacked, at the primary
    setting. The regime is from ERA5 total cloud cover, the season from the valid month, and the
    hour from the valid time.

    Args:
        fits: Every fit.

    Returns:
        One row per (split, group, contrast).

    Raises:
        ValueError: If the losses or the frame of any of lead days 1 to 3 is missing.
    """
    parts = []
    for lead_day in PLANNED_LEAD_DAYS:
        losses = fits.losses.get(("ladder", "pv", lead_day))
        dataset = fits.datasets.get(lead_day)
        if losses is None or dataset is None:
            msg = (
                f"the splits pool lead days {PLANNED_LEAD_DAYS}, but lead day {lead_day} is missing"
            )
            raise ValueError(msg)
        extra = dataset.select(
            "site",
            "time",
            "hour_of_day",
            season=season_of_month(month_number=pl.col("time").dt.month()),
            **{REGIME_COLUMN: sky_regime_from_cloud_cover(cloud_cover=pl.col("era5_tcc"))},
        )
        parts.append(
            losses.filter(pl.col("setting") == PRIMARY_SETTING).join(
                extra, on=["site", "time"], how="inner"
            )
        )
    stacked = stack_lead_days(losses_by_lead_day={int(p["lead_day"][0]): p for p in parts})
    rows: list[dict[str, object]] = []
    for split in (REGIME_COLUMN, "season", "hour_of_day"):
        for group in sorted(stacked[split].drop_nulls().unique().to_list()):
            subset = stacked.filter(pl.col(split) == group)
            n_months = subset["month"].n_unique()
            for treatment, reference in SPLIT_CONTRASTS:
                row: dict[str, object] = {
                    "split": split,
                    "group": str(group),
                    "treatment": treatment,
                    "reference": reference,
                    "difference": None,
                    "lower_95": None,
                    "upper_95": None,
                    "reference_value": float(
                        subset.filter(pl.col("arm") == reference)
                        .select(pl.col(METRIC).mean())
                        .item()
                    ),
                    "n_rows": subset.filter(pl.col("arm") == treatment).height,
                    "n_months": n_months,
                }
                if n_months >= MIN_MONTHS_FOR_INTERVAL:
                    interval = bootstrap_difference(
                        losses=subset, treatment=treatment, reference=reference, metric=METRIC
                    )
                    row |= {
                        "difference": interval["difference"],
                        "lower_95": interval["lower_95"],
                        "upper_95": interval["upper_95"],
                    }
                rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def ifs_months() -> list[str]:
    """Return the `%Y-%m` months the IFS rows can hold: every month with runs and in an era."""
    months: list[str] = []
    year, month = int(FIRST_IFS_MONTH[:4]), int(FIRST_IFS_MONTH[5:])
    while f"{year}-{month:02d}" < LAST_ERA_MONTH_EXCLUSIVE:
        label = f"{year}-{month:02d}"
        if label not in DROPPED_MONTHS:
            months.append(label)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def width_check(*, months: Sequence[str]) -> list[dict[str, object]]:
    """Return the width a planned interval would have, from the ERA5 losses cut to the same months.

    The ERA5 arms stand in for the IFS arms: `g1` minus `g0` for P1, and `g9` minus `g2` for P3.
    ERA5 is an analysis, so its errors are smaller than a forecast's, and the width understates
    the width at lead. Only the columns the interval needs are read, because the file holds
    millions of rows.

    Args:
        months: The `%Y-%m` months of the IFS rows.

    Returns:
        One row per ERA5 contrast with the 99% interval's bounds and width, in fractions of
        capacity.
    """
    losses = (
        pl.scan_parquet(ERA5_LADDER_RESULTS_DIR / ERA5_LOSSES_NAME)
        .filter((pl.col("setting") == PRIMARY_SETTING) & pl.col("month").is_in(list(months)))
        .select("site", "time", "seed", "month", "arm", "setting", METRIC)
        .collect()
    )
    rows: list[dict[str, object]] = []
    for treatment, reference in (("g1", "g0"), ("g9", "g2")):
        raise_unless_same_rows(losses=losses, arms=[treatment, reference])
        lower, upper = bootstrap_difference_at_level(
            losses=losses,
            treatment=treatment,
            reference=reference,
            metric=METRIC,
            level=ADJUSTED_LEVEL_PERCENT,
            n_resamples=PLANNED_RESAMPLES,
        )
        rows.append(
            {
                "treatment": treatment,
                "reference": reference,
                "lower": lower,
                "upper": upper,
                "width": upper - lower,
                "n_months": losses["month"].n_unique(),
            }
        )
    return rows


def render_widths(*, widths: Sequence[Mapping[str, object]]) -> list[str]:
    """Render the width a planned interval would have, and what the page says about it."""
    lines = [
        "### Width a planned interval would have, from the ERA5 losses cut to the same months",
        "",
    ]
    lines.extend(
        f"- ERA5 `{width['treatment']}` minus `{width['reference']}` on {width['n_months']} "
        f"months: {ADJUSTED_LEVEL_PERCENT:g}% interval "
        f"{interval(width['lower'], width['upper'])} points, width "  # ty: ignore[invalid-argument-type]
        f"{points(width['width'])} points."  # ty: ignore[invalid-argument-type]
        for width in widths
    )
    if widths:
        wide = max(float(w["width"]) for w in widths) > WIDTH_WARNING  # ty: ignore[invalid-argument-type]
        lines.append(
            "- The widest of these "
            + (
                f"exceeds {WIDTH_WARNING * PERCENTAGE_POINTS:g} points, so a contrast with no real "
                "effect cannot return `no gain`, and the study can rule out only gains larger "
                "than about half this width."
                if wide
                else f"is at most {WIDTH_WARNING * PERCENTAGE_POINTS:g} points."
            )
            + " ERA5 is an analysis, so this understates the width at forecast lead."
        )
    return lines


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------


def table(*, rows: Sequence[Sequence[str]], header: Sequence[str]) -> str:
    """Render rows as a markdown table."""
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return "\n".join(lines)


def points(value: float | None, *, target: str = "pv", signed: bool = False) -> str:
    """Render a metric value in points of capacity (output) or index units (CAMS)."""
    if value is None:
        return "n/a"
    factor = PERCENTAGE_POINTS if target == "pv" else 1.0
    return f"{value * factor:{'+' if signed else ''}.{PRINT_DECIMALS}f}"


def interval(lower: float | None, upper: float | None, *, target: str = "pv") -> str:
    """Render an interval at the report's precision."""
    if lower is None or upper is None:
        return "n/a"
    return f"[{points(lower, target=target)}, {points(upper, target=target)}]"


def render_leaderboard(*, board: pl.DataFrame) -> str:
    """Render every leaderboard row, one table per view, target, and row set."""
    parts: list[str] = []
    for (view, target, rows_name), group in board.group_by(
        "view", "target", "rows", maintain_order=True
    ):
        unit = "% of capacity" if target == "pv" else "clearness index"
        parts += [
            f"### {view} view, {target} target, {rows_name} rows",
            "",
            table(
                rows=[
                    [
                        str(row["lead_day"]),
                        row["arm"],
                        points(row["mae"], target=target),
                        interval(row["mae_lower"], row["mae_upper"], target=target),
                        f"{row['n_rows']:,}",
                        str(row["n_months"]),
                    ]
                    for row in group.sort("lead_day", "arm").iter_rows(named=True)
                ],
                header=[
                    "lead day",
                    "arm",
                    f"mean absolute error ({unit})",
                    "95% interval",
                    "rows (per seed)",
                    "months",
                ],
            ),
            "",
        ]
    return "\n".join(parts)


def render_contrasts(*, contrasts: pl.DataFrame, families: Sequence[str]) -> str:
    """Render the contrasts of some families as markdown tables, one per family and target."""
    parts: list[str] = []
    for family in families:
        for target in ("pv", "cams"):
            subset = contrasts.filter((pl.col("family") == family) & (pl.col("target") == target))
            if subset.is_empty():
                continue
            planned = bool(subset["planned"][0])
            unit = "points of capacity" if target == "pv" else "clearness index"
            header = [
                "contrast",
                "scope",
                "setting",
                f"difference ({unit})",
                "95% interval",
                "verdict at 95%",
                "months",
            ]
            if planned:
                header[5:5] = [f"{ADJUSTED_LEVEL_PERCENT:g}% interval", "verdict at 99%"]
            rows = []
            for row in subset.iter_rows(named=True):
                cells = [
                    f"{row['label']} ({row['treatment']} minus {row['reference']})",
                    row["scope"],
                    row["setting"],
                    points(row["difference"], target=target, signed=True),
                    interval(row["lower_95"], row["upper_95"], target=target),
                    str(row["verdict_95"]),
                    str(row["n_months"]),
                ]
                if planned:
                    cells[5:5] = [
                        interval(row["lower_adjusted"], row["upper_adjusted"], target=target),
                        str(row["verdict_adjusted"]),
                    ]
                rows.append(cells)
            parts += [f"### {family}, {target} target", "", table(rows=rows, header=header), ""]
    return "\n".join(parts)


def render_era5_p4(*, contrasts: pl.DataFrame) -> str:
    """Render the ERA5 study's P4 at the ERA5 study's own adjusted level, not this study's."""
    rows = [
        [
            row["setting"],
            f"{row['treatment']} minus {row['reference']}",
            points(row["difference"], signed=True),
            interval(row["lower_adjusted"], row["upper_adjusted"]),
            interval(row["lower_95"], row["upper_95"]),
            str(row["n_months"]),
        ]
        for row in contrasts.filter(pl.col("family") == "era5_p4").iter_rows(named=True)
    ]
    return table(
        rows=rows,
        header=[
            "setting",
            "arms",
            "difference (points)",
            f"{ERA5_ADJUSTED_LEVEL_PERCENT:g}% interval (the ERA5 study's adjusted interval)",
            "95% interval",
            "months",
        ],
    )


def render_planned_verdicts(*, verdicts: pl.DataFrame) -> str:
    """Render the combined planned verdicts."""
    return table(
        rows=[
            [
                row["label"],
                f"{row['treatment']} minus {row['reference']}",
                str(row["primary"]),
                str(row["sensitivity"]),
                str(row["combined"]),
                f"{row['treatment']} minus {row['control']}: {row['versus_control']}",
                f"**{row['final']}**",
            ]
            for row in verdicts.iter_rows(named=True)
        ],
        header=[
            "contrast",
            "arms",
            "main setting (99%)",
            "second setting (99%)",
            "both settings",
            "against the control",
            "final verdict after the control",
        ],
    )


def render_controls(*, contrasts: pl.DataFrame) -> list[str]:
    """Render the controls' differences from their base arms, and the positive control's check."""
    lines = [
        (
            "A control has the columns of the arm it pads. Its difference from the base arm is "
            "what the extra columns produce from nothing, in points of capacity (positive means "
            "worse)."
        ),
        "",
    ]
    rows = []
    for control, base, _ in CONTROL_BASES:
        for scope in ("pooled 1-3", "lead day 1", "lead day 3"):
            for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING):
                row = one_row(
                    contrasts=contrasts,
                    family="control",
                    treatment=control,
                    reference=base,
                    setting=setting,
                    scope=scope,
                )
                if row is not None:
                    rows.append(
                        [
                            control,
                            base,
                            scope,
                            setting,
                            points(row["difference"], signed=True),  # ty: ignore[invalid-argument-type]
                            interval(row["lower_95"], row["upper_95"]),  # ty: ignore[invalid-argument-type]
                        ]
                    )
    lines.append(
        table(
            rows=rows,
            header=[
                "control",
                "base arm",
                "scope",
                "setting",
                "difference (points)",
                "95% interval",
            ],
        )
    )
    for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING):
        positive = one_row(
            contrasts=contrasts,
            family="control",
            treatment=POSITIVE_CONTROL_ARM,
            reference="f2",
            setting=setting,
        )
        if positive is None:
            continue
        passed = float(positive["upper_95"]) < -POSITIVE_CONTROL_MINIMUM_GAIN  # ty: ignore[invalid-argument-type]
        lines += [
            "",
            (
                "**Positive control** (`f2` plus CAMS irradiance at the valid hour), pooled lead "
                f"days 1 to 3, {setting} setting: "
                f"{points(positive['difference'], signed=True)} points "  # ty: ignore[invalid-argument-type]
                f"{interval(positive['lower_95'], positive['upper_95'])}. It must lower the "  # ty: ignore[invalid-argument-type]
                f"error by more than {POSITIVE_CONTROL_MINIMUM_GAIN * PERCENTAGE_POINTS:g} point: "
                f"**{'passed' if passed else 'FAILED'}**. A pass shows only that the instrument "
                "detects a large effect."
            ),
        ]
    return lines


def render_priority(*, evidence: pl.DataFrame) -> list[str]:
    """Render the priority list and the groups the open-data feed lacks."""
    listed = evidence.filter(pl.col("in_open_data"))
    return [
        OPEN_DATA_CHECK_NOTE,
        "",
        "### Priority list: groups with at least one variable the open-data feed carries",
        "",
        table(
            rows=[
                [
                    str(row["rank"]),
                    f"{row['group']} ({row['rung'].upper()})",
                    row["evidence_class"],
                    (
                        f"{points(row['gain'], signed=True)} "
                        f"{interval(row['gain_lower'], row['gain_upper'])} "
                        f"({row['gain_interval_level']})"
                    ),
                    row["carried"] or "none",
                    row["unchecked"] or "none",
                    row["not_carried"] or "none",
                    row["in_production_feed"] or "none",
                    points(row["drop_one_raise"], signed=True),
                    points(row["step_difference"], signed=True),
                ]
                for row in listed.iter_rows(named=True)
            ],
            header=[
                "rank",
                "group",
                "evidence class",
                "gain with interval (points; positive is a gain)",
                "carried by open data",
                "open data unchecked",
                "not carried",
                "in the production feed",
                "drop-one: error added (points)",
                "ladder step (points; negative is a gain)",
            ],
        ),
        "",
        "### Groups the open-data feed does not carry, read for the second-feed decision",
        "",
        table(
            rows=[
                [
                    f"{row['group']} ({row['rung'].upper()})",
                    row["evidence_class"],
                    points(row["gain"], signed=True),
                    row["not_carried"],
                ]
                for row in evidence.filter(~pl.col("in_open_data")).iter_rows(named=True)
            ],
            header=["group", "evidence class", "gain (points)", "variables"],
        ),
        "",
        (
            "The ranking is for IFS at lead days 1 to 3 and these months. The drop-one runs "
            "remove whole groups, so the report does not find the smallest set of variables "
            "that keeps a gain."
        ),
    ]


def render_arms(*, fits: Fits) -> list[str]:
    """Render each arm's column list for the ladder, the CAMS target, and the blend view."""
    lines: list[str] = []
    for view, target, lead_day in (
        ("ladder", "pv", 1),
        ("ladder", "pv", 5),
        ("ladder", "cams", 1),
        ("blend", "pv", 1),
    ):
        info = fits.arms.get((view, target, lead_day))
        if info is None:
            continue
        lines += [f"### {view} view, {target} target, lead day {lead_day}", ""]
        lines += [
            f"- `{arm}` ({len(columns)}): {', '.join(columns)}"
            for arm, columns in info["arms"].items()  # ty: ignore[unresolved-attribute]
        ]
        lines.append("")
    return lines


def lead_day_mean_differences(
    *, fits: Fits, planned: PlannedContrast
) -> dict[int, tuple[float, int]]:
    """Return a planned contrast's mean difference and row count at each planned lead day.

    The rows are the farm-hours that every planned lead day holds, the rows the pooled contrast
    uses, at the main setting.

    Args:
        fits: Every fit.
        planned: The contrast.

    Returns:
        For each of lead days 1 to 3, the mean paired difference and the number of farm-hours, or
        an empty dictionary if the lead days were not fitted.
    """
    pooled = fits.pooled(view=planned.view, target="pv", lead_days=PLANNED_LEAD_DAYS)
    if pooled is None:
        return {}
    in_setting = pooled.filter(pl.col("setting") == PRIMARY_SETTING)
    result: dict[int, tuple[float, int]] = {}
    for lead_day in PLANNED_LEAD_DAYS:
        day = in_setting.filter(pl.col("lead_day") == lead_day)
        paired, _ = paired_differences(
            losses=day, treatment=planned.treatment, reference=planned.reference, metric=METRIC
        )
        rows = day.filter(pl.col("arm") == planned.treatment).height // len(SEEDS)
        result[lead_day] = (float(paired.mean()), rows)
    return result


def near_line_without_second_setting(*, contrasts: pl.DataFrame) -> pl.DataFrame:
    """List the exploratory contrasts near the 5% line that have no second-setting run.

    Args:
        contrasts: The contrast table.

    Returns:
        The (target, scope, treatment, reference) of each such contrast at the primary setting.
    """
    key = ["target", "scope", "treatment", "reference"]
    second = contrasts.filter(pl.col("setting") == SENSITIVITY_SETTING).select(key)
    return (
        contrasts.filter(
            (pl.col("setting") == PRIMARY_SETTING) & pl.col("near_line").fill_null(False)
        )
        .filter(~pl.col("planned"))
        .select(key)
        .join(second, on=key, how="anti")
    )


def render_settings(
    *,
    fits: Fits,
    widths: Sequence[Mapping[str, object]],
    contrasts: pl.DataFrame,
    splits: pl.DataFrame,
) -> list[str]:
    """Render the settings and derived numbers the page quotes."""
    lines = [
        f"- Hyperparameters, main setting: {dict(PRIMARY_HYPER_PARAMETERS)}.",
        f"- Hyperparameters, second setting: {dict(SENSITIVITY_HYPER_PARAMETERS)}.",
        "- Both settings use `colsample_bytree=1`, which the code leaves unset.",
        f"- Folds: {N_FOLDS} blocks of whole months inside each era. Seeds: {len(SEEDS)}.",
        (
            f"- Resamples: {PLANNED_RESAMPLES:,} for planned contrasts, "
            f"{N_BOOTSTRAP_RESAMPLES:,} for the others, each drawing one seed and whole months."
        ),
        (
            f"- Planned family: {PLANNED_CONTRAST_COUNT} contrasts at a 5% family-wise level, so "
            f"each interval covers {ADJUSTED_LEVEL_PERCENT:g}%. Intervals from about 24 resampled "
            "months undercover, so they are approximate."
        ),
        (
            f"- Smallest effects: {SMALLEST_EFFECT['pv'] * PERCENTAGE_POINTS:g} points of capacity "
            f"on output, {SMALLEST_EFFECT['cams']:g} on the clearness index."
        ),
        (
            "- ERA5 cloud regimes: clear below total cloud cover "
            f"{TOTAL_CLOUD_COVER_THRESHOLDS[0]}, overcast from {TOTAL_CLOUD_COVER_THRESHOLDS[1]}."
        ),
        (
            "- Era 3 (2026-06 onwards) is too short to cut into folds, so it is dropped and no "
            "check scores the newest model cycle."
        ),
    ]
    devices = sorted({str(info["device"]) for info in fits.arms.values()})
    lines.append(f"- Devices the arms were fitted on: {', '.join(devices)}.")
    for lead_day in LEAD_DAYS:
        info = fits.arms.get(("ladder", "pv", lead_day))
        if info is not None:
            lines.append(f"- Ladder rows at lead day {lead_day}: {int(info['rows']):,}.")  # ty: ignore[invalid-argument-type]
    for lead_day in BLEND_LEAD_DAYS:
        info = fits.arms.get(("blend", "pv", lead_day))
        if info is not None:
            lines.append(f"- Blend rows at lead day {lead_day}: {int(info['rows']):,}.")  # ty: ignore[invalid-argument-type]
    pv1 = fits.arms.get(("ladder", "pv", 1))
    if pv1 is not None:
        arms: dict[str, list[str]] = pv1["arms"]  # ty: ignore[invalid-assignment]
        lines.append(
            "- Columns at lead day 1: "
            + ", ".join(
                f"`{arm}` {len(arms[arm])}" for arm in arms if arm in ("f0", "f1", "f2", "f6")
            )
            + f", control `{CONTROL_LATER_GROUPS_ARM}` "
            + f"{len(arms.get(CONTROL_LATER_GROUPS_ARM, []))}."
        )
    for planned in PLANNED_CONTRASTS:
        views = [(v, d) for (v, t, d) in fits.arms if v == planned.view and t == "pv"]
        if not views:
            continue
        day = min(d for _, d in views)
        arms = fits.arms[(planned.view, "pv", day)]["arms"]  # ty: ignore[invalid-assignment]
        if all(name in arms for name in (planned.treatment, planned.reference, planned.control)):
            lines.append(
                f"- {planned.label}: `{planned.treatment}` has {len(arms[planned.treatment])} "
                f"columns, `{planned.reference}` has {len(arms[planned.reference])}, and its "
                f"control `{planned.control}` has {len(arms[planned.control])}."
            )
    for planned in PLANNED_CONTRASTS:
        by_day = lead_day_mean_differences(fits=fits, planned=planned)
        pooled = pooled_difference(
            contrasts=contrasts,
            family="planned",
            treatment=planned.treatment,
            reference=planned.reference,
        )
        if pooled is not None and by_day:
            lines.append(
                f"- {planned.label}: pooled over the farm-hours that lead days 1, 2, and 3 all "
                f"hold ({', '.join(f'{rows:,}' for _, rows in by_day.values())} farm-hours per "
                "lead day), the lead days' differences are "
                f"{', '.join(points(value, signed=True) for value, _ in by_day.values())} points; "
                "their equal-weight mean is "
                f"{points(sum(v for v, _ in by_day.values()) / len(by_day), signed=True)}, and "
                f"the pooled difference is {points(pooled, signed=True)}."
            )
    lines += ["", *render_widths(widths=widths), ""]
    exploratory = contrasts.filter(~pl.col("planned"))
    split_intervals = (
        splits.filter(pl.col("difference").is_not_null()).height if not splits.is_empty() else 0
    )
    lines.append(
        f"- Exploratory intervals computed: {exploratory.height + split_intervals} "
        f"({exploratory.height} contrast intervals, one per contrast, scope, and setting, and "
        f"{split_intervals} split intervals), all at 95%. About 1 in 20 with no real effect "
        "behind it reaches statistical significance at the 5% level, the intervals share "
        "months so that spurious results cluster, and the report applies no correction."
    )
    near = near_line_without_second_setting(contrasts=contrasts)
    if near.is_empty():
        lines.append("- No exploratory result near the 5% line lacks a second-setting run.")
    else:
        lines.append(
            f"- {near.height} exploratory results are near the 5% line (a bound within "
            f"{NEAR_LINE_SHARE:g} of the interval's width from zero) and have no second-setting "
            "run. Fit them with `--sensitivity-arms` before the page reads them:"
        )
        lines += [
            f"  - {row['target']} target, {row['scope']}: `{row['treatment']}` minus "
            f"`{row['reference']}`"
            for row in near.iter_rows(named=True)
        ]
    return lines


def render_splits(*, splits: pl.DataFrame) -> str:
    """Render the regime, season, and hour-of-day splits."""
    if splits.is_empty():
        return "No splits."
    parts: list[str] = []
    for split in (REGIME_COLUMN, "season", "hour_of_day"):
        subset = splits.filter(pl.col("split") == split)
        rows = [
            [
                row["group"],
                f"{row['treatment']} minus {row['reference']}",
                points(row["difference"], signed=True),
                interval(row["lower_95"], row["upper_95"]),
                points(row["reference_value"]),
                str(row["n_months"]),
            ]
            for row in subset.iter_rows(named=True)
        ]
        parts += [
            f"### Split by {split}",
            "",
            table(
                rows=rows,
                header=[
                    "group",
                    "contrast",
                    "difference (points)",
                    "95% interval",
                    "reference error (points)",
                    "months",
                ],
            ),
            "",
        ]
    return "\n".join(parts)


def render_farms(*, farms: pl.DataFrame) -> str:
    """Render each farm's mean absolute error for F0, F1, F2, F6, and FP at every lead day."""
    subset = farms.filter(pl.col("arm").is_in(["f0", "f1", "f2", F6_ARM, PRODUCTION_ARM]))
    rows = [
        [str(row["lead_day"]), row["arm"], row["site"], points(row["value"]), f"{row['n_rows']:,}"]
        for row in subset.iter_rows(named=True)
    ]
    return table(
        rows=rows,
        header=[
            "lead day",
            "arm",
            "farm",
            "mean absolute error (% of capacity)",
            "rows (all three seeds)",
        ],
    )


def render_decisions(
    *, verdicts: pl.DataFrame, evidence: pl.DataFrame, third: ThirdDecision, needs_second: bool
) -> list[str]:
    """Render the three decisions."""
    uppers = third.era5_p4_uppers
    mars_can_occur = len(uppers) == SETTINGS_THAT_MUST_AGREE and all(
        upper < -SMALLEST_EFFECT["pv"] for upper in uppers
    )
    return [
        "## Decisions",
        "",
        "### Which variables to ask Dynamical.org to add",
        "",
        "The priority list below ranks the groups.",
        "",
        *render_priority(evidence=evidence),
        "",
        "### Whether the production forecast needs a second IFS feed",
        "",
        (
            f"- **{'Second feed recommended' if needs_second else 'No second feed recommended'}.** "
            "The rule: P2 is a gain after its control, or the drop-one run of the cloud layers "
            "or direct radiation finds the group carrying a gain. The drop-one part is "
            "exploratory. Boundary-layer height, the third variable the open-data feed lacks, "
            "sits in a group with carried variables, so no drop-one run isolates it and the "
            "rule cannot test it."
        ),
        "",
        "### Variables or another forecast",
        "",
        f"- P5 final verdict after its control: {third.p5_final_verdict}.",
        f"- ERA5 P4 {ERA5_ADJUSTED_LEVEL_PERCENT:g}% upper bounds (the ERA5 study's own adjusted "
        "level) at the two settings, sorted by setting name (points): "
        + ", ".join(points(upper, signed=True) for upper in uppers)
        + ".",
        (
            f"- `mars_variables` {'can' if mars_can_occur else 'cannot'} occur: it needs both ERA5 "
            "P4 upper bounds below minus the smallest effect "
            f"({points(-SMALLEST_EFFECT['pv'], signed=True)} points). The page says so."
        ),
        "- ICON-EU was not tested.",
        (
            f"- **Outcome: {third.outcome}.** The rule: `second_forecast` if P5 is a gain; "
            "`mars_variables` if P5 is no gain and the ERA5 P4 interval lies wholly below minus "
            "the smallest effect at both settings; otherwise `not_separated`."
        ),
        "",
        "### Planned verdicts, both settings combined",
        "",
        render_planned_verdicts(verdicts=verdicts),
    ]


def render_report(
    *,
    fits: Fits,
    build_notes: str,
    board: pl.DataFrame,
    contrasts: pl.DataFrame,
    verdicts: pl.DataFrame,
    evidence: pl.DataFrame,
    third: ThirdDecision,
    needs_second: bool,
    splits: pl.DataFrame,
    farms: pl.DataFrame,
    widths: Sequence[Mapping[str, object]],
    day_lines: Sequence[str],
) -> str:
    """Assemble the report text."""
    parts = [
        "# IFS variable ladder: report",
        "",
        "## Build checks",
        "",
        build_notes,
        "## Settings and derived numbers the page quotes",
        "",
        *render_settings(fits=fits, widths=widths, contrasts=contrasts, splits=splits),
        "",
        "### Months of the days the weather and prediction figures draw",
        "",
        *day_lines,
        "",
        *render_decisions(
            verdicts=verdicts, evidence=evidence, third=third, needs_second=needs_second
        ),
        "",
        "## Planned contrasts (planned)",
        "",
        render_contrasts(contrasts=contrasts, families=["planned", "planned_control"]),
        "## Controls",
        "",
        *render_controls(contrasts=contrasts),
        "",
        "## Mean absolute error by arm and lead day",
        "",
        render_leaderboard(board=board),
        "## Exploratory contrasts (exploratory)",
        "",
        render_contrasts(
            contrasts=contrasts,
            families=[
                "step",
                "against_f0",
                "against_production",
                "drop_one",
                "blend",
                "cams",
                "era5",
            ],
        ),
        "### ERA5 study P4 (the ERA5 study's own planned contrast, at its own interval level)",
        "",
        (
            f"The ERA5 study adjusted for ten planned contrasts, so its interval covers "
            f"{ERA5_ADJUSTED_LEVEL_PERCENT:g}%, not the {ADJUSTED_LEVEL_PERCENT:g}% of this "
            "study's intervals."
        ),
        "",
        render_era5_p4(contrasts=contrasts),
        "## Contrasts by cloud regime, season, and hour of day (exploratory)",
        "",
        render_splits(splits=splits),
        "## Per-farm errors",
        "",
        render_farms(farms=farms),
        "",
        "## Columns each arm was shown",
        "",
        *render_arms(fits=fits),
    ]
    return "\n".join(parts) + "\n"


def write_width_preview() -> int:
    """Write the width a planned interval would have, from the ERA5 losses alone."""
    path = width_preview_path()
    refuse_to_overwrite(paths=[path])
    months = ifs_months()
    lines = [
        "# Width a planned interval would have, before any IFS fit",
        "",
        (
            f"The IFS rows can hold {len(months)} months, {months[0]} to {months[-1]}, less "
            f"{sorted(DROPPED_MONTHS)}."
        ),
        "",
        *render_widths(widths=width_check(months=months)),
    ]
    path.write_text("\n".join(lines) + "\n")
    _LOG.info("wrote %s", path)
    return 0


def main() -> int:
    """Score the saved fits and write the report and tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--width-preview-only",
        action="store_true",
        help="Read only the ERA5 losses and write width_preview.md, before any IFS fit exists.",
    )
    if parser.parse_args().width_preview_only:
        return write_width_preview()
    paths = report_paths()
    refuse_to_overwrite(paths=list(paths))
    fits = Fits()
    if not fits.losses:
        msg = "no fit has saved losses; run ifs_ladder_fit.py first"
        raise FileNotFoundError(msg)

    contrast_rows = [*planned_rows(fits=fits), *exploratory_rows(fits=fits), *era5_p4_rows()]
    contrasts = pl.DataFrame(contrast_rows, infer_schema_length=None)
    era5_p4 = contrasts.filter(pl.col("family") == "era5_p4")
    verdicts = planned_verdict_table(contrasts=contrasts)
    evidence = group_evidence(contrasts=contrasts, verdicts=verdicts)
    third = third_decision(verdicts=verdicts, era5_p4=era5_p4)
    needs_second = second_feed(evidence=evidence, verdicts=verdicts)
    board = leaderboard_rows(fits=fits)
    farms = per_farm_rows(fits=fits)
    splits = split_rows(fits=fits)
    first_dataset = fits.datasets[min(fits.datasets)]
    widths = width_check(months=first_dataset["month"].unique().to_list())
    near_dataset = fits.datasets.get(WEATHER_LEAD_DAYS[0])
    far_dataset = fits.datasets.get(WEATHER_LEAD_DAYS[1])
    if near_dataset is None or far_dataset is None:
        day_lines = ["- The frames of lead days 1 and 7 are missing, so no days were chosen."]
    else:
        weather_days, farm_days = chosen_days(near=near_dataset, far=far_dataset)
        day_lines = month_lines(weather_days=weather_days, farm_days=farm_days)

    board.write_parquet(paths.leaderboard)
    contrasts.write_parquet(paths.contrasts)
    farms.write_parquet(paths.farms)
    if not splits.is_empty():
        splits.write_parquet(paths.splits)
    decisions = pl.DataFrame(
        {
            "decision": ["third_decision", "second_feed"],
            "outcome": [
                str(third.outcome),
                "recommended" if needs_second else "not recommended",
            ],
        }
    )
    decisions.write_parquet(paths.decisions)
    paths.report.write_text(
        render_report(
            fits=fits,
            build_notes=checks_path().read_text(),
            board=board,
            contrasts=contrasts,
            verdicts=verdicts,
            evidence=evidence,
            third=third,
            needs_second=needs_second,
            splits=splits,
            farms=farms,
            widths=widths,
            day_lines=day_lines,
        )
    )
    evidence.write_parquet(paths.priority_list)
    verdicts.write_parquet(paths.verdicts)
    _LOG.info("wrote %s", paths.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
