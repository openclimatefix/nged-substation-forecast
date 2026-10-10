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
    CONTROL_E0_PARTNER_ARM,
    CONTROL_HUMIDITY_ARM,
    CONTROL_LATER_GROUPS_ARM,
    CONTROL_PARTNER_FULL_ARM,
    CONTROL_PARTNER_MINIMAL_ARM,
    DROP_ONE_LEAD_DAYS,
    DROP_PREFIX,
    E0_ARM,
    E_A_ARM,
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
    POST_HOC_CONTRASTS,
    PRIMARY_SETTING,
    PRODUCTION_ARM,
    SENSITIVITY_SETTING,
    SMALLEST_EFFECT,
    FitKey,
    PlannedContrast,
    TargetType,
    arms_path,
    checks_path,
    ens_dataset_path,
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
    display_class,
    drop_one_gain_at_settings,
    drop_one_margin,
    evidence_class,
    exploratory_gain,
    gain_after_control,
    half_width_allows_no_gain,
    is_near_line,
    mean_absolute_error_on_common_rows,
    mean_paired_difference_by,
    merge_extra_arms,
    percentage_rise,
    priority_order,
    production_scope_sentence,
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
    (CONTROL_E0_PARTNER_ARM, E0_ARM, "ens"),
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

FIRST_IFS_MONTH: Final[str] = "2024-03"
"""The first month with IFS runs, a part-month."""

POST_HOC_NOT_FITTED: Final[str] = "post hoc analysis not yet fitted"
"""What a post hoc section prints when the losses it needs have not been fitted."""

WINTER_MONTH_NUMBERS: Final[frozenset[str]] = frozenset({"12", "01", "02"})
"""The calendar months, as the `%m` part of a `%Y-%m` label, that count as winter."""

LOW_SUN_HOURS: Final[tuple[int, ...]] = (6, 7, 19, 20)
"""The UTC hours of day at which the full set is worse than the minimal set in the ERA5 study."""

RAW_RADIATION_COLUMNS: Final[dict[str, str]] = {
    "shortwave_radiation": "Open-Meteo IFS 9 km",
    "partner_shortwave_radiation": "AIFS Single",
    "ens_shortwave_radiation": "IFS ensemble mean",
}
"""The radiation columns of the blend frame with ensemble mean, by the name the report prints."""

CAMS_IRRADIANCE_COLUMN: Final[str] = "cams_ghi_w_m2"
"""The CAMS irradiance the raw radiation errors are scored against."""

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
        self.blend_ens: pl.DataFrame | None = (
            pl.read_parquet(ens_dataset_path()) if ens_dataset_path().exists() else None
        )
        self.missing_post_hoc: list[str] = []
        for view, days in (
            ("ladder", LEAD_DAYS),
            ("blend", BLEND_LEAD_DAYS),
            ("ens", BLEND_LEAD_DAYS),
        ):
            for target in ("pv", "cams"):
                for lead_day in days:
                    key = FitKey(view=view, lead_day=lead_day, target=target)
                    losses = read_losses(key=key)
                    if losses is None and view == "ens" and target == "pv":
                        self.missing_post_hoc.append(results_path(key=key).name)
                    if losses is not None:
                        self.losses[(view, target, lead_day)] = losses
                        self.arms[(view, target, lead_day)] = json.loads(
                            arms_path(key=key).read_text()
                        )
        for lead_day in PLANNED_LEAD_DAYS:
            key = FitKey(view="extra_sensitivity", lead_day=lead_day, target="pv")
            extra = read_losses(key=key)
            if extra is None:
                self.missing_post_hoc.append(results_path(key=key).name)
            elif ("ladder", "pv", lead_day) in self.losses:
                self.losses[("ladder", "pv", lead_day)] = merge_extra_arms(
                    main=self.losses[("ladder", "pv", lead_day)], extra=extra
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

    def ens_with_f0(self, *, lead_days: Sequence[int]) -> pl.DataFrame | None:
        """Return the stacked `ens` view losses with the blend view's `f0` added.

        The `ens` view and the `blend` view are fitted on the same rows, so `f0_e` and `f0` can be
        contrasted once the two fits' losses sit in one frame.

        Args:
            lead_days: The lead days to stack.

        Returns:
            The stacked losses, or `None` if no `ens` lead day was fitted.

        Raises:
            ValueError: If some but not all of the lead days were fitted in either view.
        """
        ens = self.get(view="ens", target="pv", lead_days=lead_days)
        if not ens:
            return None
        blend = self.get(view="blend", target="pv", lead_days=lead_days)
        missing = sorted(set(lead_days) - set(ens)) + sorted(set(lead_days) - set(blend))
        if missing:
            msg = f"lead days {missing} are not fitted in the ens or the blend view"
            raise ValueError(msg)
        return stack_lead_days(
            losses_by_lead_day={
                day: pl.concat(
                    [ens[day], blend[day].filter(pl.col("arm") == "f0").select(ens[day].columns)],
                    how="vertical_relaxed",
                )
                for day in lead_days
            }
        )

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
    adjusted: bool = False,
) -> dict[str, object]:
    """Return one contrast's difference with its 95% interval and, if asked, its 99% interval.

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
        planned: Whether the plan named the contrast before any result existed. A planned contrast
            always gets the adjusted interval.
        adjusted: Whether to add the adjusted interval to a contrast that is not planned, as the
            post hoc production-relevant contrasts do.

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
    if planned or adjusted:
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
    adjusted: bool = False,
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
        adjusted: Whether to add the adjusted interval to contrasts that are not planned.
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
                    adjusted=adjusted,
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


def winter_folds(*, losses: pl.DataFrame) -> list[int]:
    """Return the folds that hold a December, January, or February month.

    Args:
        losses: A fit's per-row losses, carrying `fold` and `month` (`%Y-%m`).

    Returns:
        The sorted fold numbers. The blend rows' folds are blocks of whole months, so a fold that
        holds a winter month trains on no winter month of the other folds' years.
    """
    winter = losses.filter(pl.col("month").str.slice(5, 2).is_in(list(WINTER_MONTH_NUMBERS)))
    return sorted(winter["fold"].unique().to_list())


def without_winter(*, losses: pl.DataFrame, folds: Sequence[int]) -> pl.DataFrame:
    """Return the losses of the folds that hold no winter month."""
    return losses.filter(~pl.col("fold").is_in(list(folds)))


def blend_without_winter_rows(*, fits: Fits) -> list[dict[str, object]]:
    """Compute P4 and P5 again without the folds that hold the blend rows' winter months.

    Args:
        fits: Every fit.

    Returns:
        The rows, each at 95%, pooled over lead days 1 to 3 and at each lead day alone.
    """
    reference = fits.get(view="blend", target="pv", lead_days=BLEND_LEAD_DAYS)
    if not reference:
        return []
    folds = winter_folds(losses=pl.concat(reference.values()))
    pairs = [(planned.treatment, planned.reference) for planned in PLANNED_CONTRASTS[3:]]
    labels = {(p.treatment, p.reference): p.label for p in PLANNED_CONTRASTS[3:]}
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, Sequence[int]]] = [
        ("pooled 1-3 without winter fold", BLEND_LEAD_DAYS),
        *((f"lead day {day} without winter fold", (day,)) for day in BLEND_LEAD_DAYS),
    ]
    for scope, days in scopes:
        pooled = fits.pooled(view="blend", target="pv", lead_days=days)
        if pooled is None:
            continue
        add_pairs(
            rows=rows,
            losses=without_winter(losses=pooled, folds=folds),
            target="pv",
            family="blend_no_winter",
            scope=scope,
            pairs=pairs,
            labels=labels,
        )
    return rows


def post_hoc_rows(*, fits: Fits) -> list[dict[str, object]]:
    """Compute the production-relevant contrasts, which were added after the first review.

    Each is pooled over lead days 1 to 3 on the blend rows, at every setting fitted, with the 95%
    and the 99% interval. Contrasts are labelled post hoc and counted as exploratory.

    Args:
        fits: Every fit.

    Returns:
        The rows of family `post_hoc`, of `post_hoc_control` (each treatment minus its padding
        control), and of `control` for the ensemble-mean arm's control, or no rows if the `ens`
        view was not fitted.
    """
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, Sequence[int]]] = [
        ("pooled 1-3", BLEND_LEAD_DAYS),
        *((f"lead day {day}", (day,)) for day in BLEND_LEAD_DAYS),
    ]
    for scope, days in scopes:
        losses = fits.ens_with_f0(lead_days=days)
        if losses is None:
            continue
        adjusted = scope == "pooled 1-3"
        for contrast in POST_HOC_CONTRASTS:
            add_pairs(
                rows=rows,
                losses=losses,
                target="pv",
                family="post_hoc",
                scope=scope,
                pairs=[(contrast.treatment, contrast.reference)],
                adjusted=adjusted,
                labels={(contrast.treatment, contrast.reference): contrast.label},
            )
            if contrast.control is not None:
                add_pairs(
                    rows=rows,
                    losses=losses,
                    target="pv",
                    family="post_hoc_control",
                    scope=scope,
                    pairs=[(contrast.treatment, contrast.control)],
                    adjusted=adjusted,
                    labels={(contrast.treatment, contrast.control): f"{contrast.label} vs control"},
                )
        add_pairs(
            rows=rows,
            losses=losses,
            target="pv",
            family="control",
            scope=scope,
            pairs=[(CONTROL_E0_PARTNER_ARM, E0_ARM)],
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
    rows += blend_without_winter_rows(fits=fits)
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
    *,
    contrasts: pl.DataFrame,
    family: str,
    treatment: str,
    reference: str,
    setting: str = PRIMARY_SETTING,
    scope: str = "pooled 1-3",
) -> float | None:
    """Return a contrast's difference at one setting, or `None` if it was not computed."""
    row = one_row(
        contrasts=contrasts,
        family=family,
        treatment=treatment,
        reference=reference,
        setting=setting,
        scope=scope,
    )
    return None if row is None else float(row["difference"])  # ty: ignore[invalid-argument-type]


def drop_one_by_lead_day(*, contrasts: pl.DataFrame, rung: RungType, setting: str) -> str:
    """Return a group's drop-one raise at each planned lead day, for the report to print.

    Args:
        contrasts: The contrast table.
        rung: The group's rung.
        setting: The hyperparameter setting.

    Returns:
        The values in points, such as `L1 -0.003, L2 -0.005, L3 +0.116`, or `n/a` if none exist.
    """
    parts = []
    for day in PLANNED_LEAD_DAYS:
        value = pooled_difference(
            contrasts=contrasts,
            family="drop_one",
            treatment=f"{DROP_PREFIX}{rung}",
            reference=F6_ARM,
            setting=setting,
            scope=f"lead day {day}",
        )
        if value is not None:
            parts.append(f"L{day} {points(value, signed=True)}")
    return ", ".join(parts) if parts else "n/a"


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

    Total cloud, and the cloud layers unless the layers' class comes from their drop-one run, read
    their planned contrast's 99% interval. A group in the exploratory class reads the 95% interval
    of its pooled ladder step. Every other group reads the 95% interval of its drop-one run, which
    is exploratory.

    Args:
        contrasts: The contrast table.
        rung: The group's rung.
        evidence: The group's evidence class.

    Returns:
        The lower and upper bounds of the gain, and `99%` or `95%`. Bounds are `None` if the
        contrast was not computed.
    """
    if rung == "f1" or (rung == "f2" and evidence != "drop-one gain"):
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

    The drop-one rule is applied at every setting at which both the group's drop-one run and the
    later-groups control were fitted, and the group counts as a drop-one gain only if all of them
    agree. The row's `drop_one_settings` says whether that was both settings or the main one alone.

    Args:
        contrasts: The contrast table.
        verdicts: The planned verdict table.

    Returns:
        One row per group, in priority order, with its class, pooled gain, and the open-data
        status of its variables. `rank` is the position among all groups, and `rank_in_table` the
        position within the table that prints the group.
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
        drop_raises: list[float] = []
        drop_controls: list[float] = []
        for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING):
            raised = pooled_difference(
                contrasts=contrasts,
                family="drop_one",
                treatment=f"{DROP_PREFIX}{rung}",
                reference=F6_ARM,
                setting=setting,
            )
            later_control = pooled_difference(
                contrasts=contrasts,
                family="control",
                treatment=CONTROL_LATER_GROUPS_ARM,
                reference="f2",
                setting=setting,
            )
            if raised is not None and later_control is not None:
                drop_raises.append(raised)
                drop_controls.append(later_control)
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
        has_drop_one = drop_one_gain_at_settings(
            raises=drop_raises, control_differences=drop_controls
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
            classes[rung] == "drop-one gain" or rung in PLANNED_COVERS_SEVERAL_GROUPS
        ) and drop is not None:
            gains[rung] = drop
        else:
            gains[rung] = -step if step is not None else 0.0
        interval_lower, interval_upper, level = gain_interval(
            contrasts=contrasts,
            rung=rung,
            evidence=classes[rung],
        )
        margin = (
            None
            if drop is None or control_difference is None
            else drop_one_margin(raise_in_error=drop, control_difference=control_difference)
        )
        details[rung] = {
            "display_class": display_class(
                evidence=classes[rung],
                gain_lower=interval_lower if rung in ("f1", "f2") else None,
                smallest_effect=SMALLEST_EFFECT["pv"],
            ),
            "drop_one_settings": (
                "not fitted"
                if not drop_raises
                else "both settings"
                if len(drop_raises) == SETTINGS_THAT_MUST_AGREE
                else "main setting only"
            ),
            "drop_one_margin": margin,
            "drop_one_by_lead_day": drop_one_by_lead_day(
                contrasts=contrasts, rung=rung, setting=PRIMARY_SETTING
            ),
            "drop_one_raise_second": pooled_difference(
                contrasts=contrasts,
                family="drop_one",
                treatment=f"{DROP_PREFIX}{rung}",
                reference=F6_ARM,
                setting=SENSITIVITY_SETTING,
            ),
            "f2_control_difference_second": pooled_difference(
                contrasts=contrasts,
                family="control",
                treatment=CONTROL_LATER_GROUPS_ARM,
                reference="f2",
                setting=SENSITIVITY_SETTING,
            ),
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
    in_table_count = {True: 0, False: 0}
    for rank, name in enumerate(priority_order(classes=classes, gains=gains), start=1):
        rung = cast("RungType", name)
        listed = "carried" in [OPEN_DATA_AVAILABILITY[v] for v in RUNG_ADDITIONS[rung]]
        in_table_count[listed] += 1
        variables = RUNG_ADDITIONS[rung]
        availability = [OPEN_DATA_AVAILABILITY[name] for name in variables]
        rows.append(
            {
                "rank": rank,
                "rank_in_table": in_table_count[listed],
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
                "in_open_data": "carried" in availability,
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
        f"{points(width['width'])} points, half-width "  # ty: ignore[invalid-argument-type]
        f"{points(float(width['width']) / 2)} points."  # ty: ignore[invalid-argument-type]
        for width in widths
    )
    if widths:
        half_widths = [float(w["width"]) / 2 for w in widths]  # ty: ignore[invalid-argument-type]
        smallest = SMALLEST_EFFECT["pv"]
        allowed = all(
            half_width_allows_no_gain(half_width=half_width, smallest_effect=smallest)
            for half_width in half_widths
        )
        lines.append(
            "- A contrast with no real effect returns `no gain` only if its "
            f"{ADJUSTED_LEVEL_PERCENT:g}% half-width is below {smallest * PERCENTAGE_POINTS:g} "
            f"points. The planned half-widths here are {points(min(half_widths))} to "
            f"{points(max(half_widths))} points, so a contrast with no real effect "
            + ("can" if allowed else "cannot always")
            + " return `no gain`. ERA5 is an analysis, so this understates the width at forecast "
            "lead."
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
            planned = bool(subset["lower_adjusted"].is_not_null().any())
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
                        "n/a" if row["verdict_adjusted"] is None else str(row["verdict_adjusted"]),
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
    """Render the priority list and the groups the open-data feed lacks.

    Ranks count from 1 within each of the two tables. The class column holds the label from
    `studies.ifs_decisions.display_class`, and the gain column holds the drop-one raise for a group
    whose class comes from its drop-one run, and the ladder step otherwise.
    """
    listed = evidence.filter(pl.col("in_open_data")).sort("rank_in_table")
    return [
        OPEN_DATA_CHECK_NOTE,
        "",
        "### Priority list: groups with at least one variable the open-data feed carries",
        "",
        table(
            rows=[
                [
                    str(row["rank_in_table"]),
                    f"{row['group']} ({row['rung'].upper()})",
                    row["display_class"],
                    (
                        f"{points(row['gain'], signed=True)} "
                        f"{interval(row['gain_lower'], row['gain_upper'])} "
                        f"({row['gain_interval_level']})"
                    ),
                    row["carried"] or "none",
                    row["not_carried"] or "none",
                    row["in_production_feed"] or "none",
                    points(row["drop_one_raise"], signed=True),
                    points(row["step_difference"], signed=True),
                ]
                for row in listed.iter_rows(named=True)
            ],
            header=[
                "rank in this table",
                "group",
                "evidence class",
                "gain with interval (points; positive is a gain)",
                "carried by open data",
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
                    str(row["rank_in_table"]),
                    f"{row['group']} ({row['rung'].upper()})",
                    row["display_class"],
                    points(row["gain"], signed=True),
                    row["not_carried"],
                ]
                for row in evidence.filter(~pl.col("in_open_data"))
                .sort("rank_in_table")
                .iter_rows(named=True)
            ],
            header=["rank in this table", "group", "evidence class", "gain (points)", "variables"],
        ),
        "",
        (
            "The ranking is for IFS at lead days 1 to 3 and these months. The drop-one runs "
            "remove whole groups, so the report does not find the smallest set of variables "
            "that keeps a gain. A group's gain is its drop-one raise when its class comes from "
            "the drop-one run, and its ladder step otherwise."
        ),
    ]


def render_drop_one_rule(*, evidence: pl.DataFrame) -> list[str]:
    """Render the drop-one rule group by group, at each setting fitted, and what it implies.

    Args:
        evidence: The priority list's rows.

    Returns:
        The table of raises, controls, margins, and per-lead-day values, then the lines about F4,
        the cloud layers, and the settings the rule was applied at.
    """
    rows = [
        [
            f"{row['group']} ({row['rung'].upper()})",
            points(row["drop_one_raise"], signed=True),
            points(row["f2_control_difference"], signed=True),
            points(row["drop_one_margin"], signed=True),
            points(row["drop_one_raise_second"], signed=True),
            points(row["f2_control_difference_second"], signed=True),
            row["drop_one_settings"],
            row["drop_one_by_lead_day"],
            row["evidence_class"],
        ]
        for row in evidence.sort("rung").iter_rows(named=True)
    ]
    lines = [
        "### The drop-one rule, group by group",
        "",
        (
            "A group passes the rule at a setting if removing it from the full set raises the "
            "error by more than the negative control that pads F2 with the later groups' permuted "
            "variables (0 if the control lowered the error). The margin is the raise minus the "
            "larger of the control and zero. The group counts as a drop-one gain only if it "
            "passes at every setting at which it was fitted. All values are in points of "
            "capacity, pooled over lead days 1 to 3 unless a lead day is named."
        ),
        "",
        table(
            rows=rows,
            header=[
                "group",
                "raise, main setting",
                "control, main setting",
                "margin, main setting",
                "raise, second setting",
                "control, second setting",
                "rule applied at",
                "raise by lead day, main setting",
                "evidence class",
            ],
        ),
        "",
    ]
    f4 = evidence.filter(pl.col("rung") == "f4")
    if f4.height and f4["drop_one_margin"][0] is not None:
        margin = float(f4["drop_one_margin"][0])
        lines.append(
            f"- F4 {'passes' if margin > 0 else 'fails'} the drop-one rule at the main setting by "
            f"{points(margin, signed=True)} points; its raise by lead day is "
            f"{f4['drop_one_by_lead_day'][0]}."
        )
    f2 = evidence.filter(pl.col("rung") == "f2")
    if f2.height and f2["drop_one_raise"][0] is not None:
        raised = float(f2["drop_one_raise"][0])
        smallest = SMALLEST_EFFECT["pv"]
        lines.append(
            f"- Removing the cloud layers raises the full set's error by {points(raised)} points "
            f"(main setting), against the smallest effect of {points(smallest)} points: it "
            f"{'reaches' if raised > smallest else 'does not reach'} the smallest effect."
        )
    applied = sorted({str(value) for value in evidence["drop_one_settings"].to_list()})
    groups_by_setting = {
        setting: ", ".join(
            f"`{rung}`"
            for rung in evidence.filter(pl.col("drop_one_settings") == setting)
            .sort("rung")["rung"]
            .to_list()
        )
        for setting in applied
    }
    lines.append(
        "- The rule was applied at: "
        + "; ".join(f"{setting} ({groups})" for setting, groups in groups_by_setting.items())
        + "."
    )
    if "both settings" not in applied:
        lines.append(
            "- The second-setting drop-one runs are absent, so the rule was applied at the main "
            "setting only."
        )
    return lines


def render_against_production(*, contrasts: pl.DataFrame) -> list[str]:
    """Render F1 minus FP and F2 minus FP with 95% intervals, the contrasts against production."""
    lines = [
        "### Total cloud and the cloud layers against the production reference (exploratory)",
        "",
    ]
    for treatment in ("f1", "f2"):
        for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING):
            row = one_row(
                contrasts=contrasts,
                family="against_production",
                treatment=treatment,
                reference=PRODUCTION_ARM,
                setting=setting,
            )
            if row is not None:
                lines.append(
                    f"- {treatment.upper()} minus FP, pooled over lead days 1 to 3, {setting} "
                    f"setting: {points(row['difference'], signed=True)} points, 95% interval "  # ty: ignore[invalid-argument-type]
                    f"{interval(row['lower_95'], row['upper_95'])}."  # ty: ignore[invalid-argument-type]
                )
    return lines


def render_low_sun(*, splits: pl.DataFrame) -> list[str]:
    """Render the full set minus the minimal set at the low-sun hours, with 95% intervals."""
    lines = ["### The low-sun penalty (F6 minus F0 by hour of day, exploratory)", ""]
    rows = []
    for hour in LOW_SUN_HOURS:
        found = (
            splits.filter(
                (pl.col("split") == "hour_of_day")
                & (pl.col("group") == str(hour))
                & (pl.col("treatment") == F6_ARM)
                & (pl.col("reference") == "f0")
            )
            if not splits.is_empty()
            else splits
        )
        if found.height:
            row = found.row(0, named=True)
            rows.append(
                [
                    f"{hour:02d}",
                    points(row["difference"], signed=True),
                    interval(row["lower_95"], row["upper_95"]),
                ]
            )
    if not rows:
        return [*lines, "No hour-of-day split."]
    return [
        *lines,
        table(rows=rows, header=["hour (UTC)", "F6 minus F0 (points)", "95% interval"]),
        "",
        "A positive difference means the full set is worse than the minimal set at that hour.",
    ]


def post_hoc_verdict_rows(*, contrasts: pl.DataFrame) -> list[dict[str, object]]:
    """Read the post hoc contrasts by the plan's verdict formula, for information.

    Args:
        contrasts: The contrast table.

    Returns:
        One row per post hoc contrast with the verdict at each setting, combined, against its
        control (if it has one), and after the control. `None` values mark a missing fit.
    """
    rows: list[dict[str, object]] = []
    for contrast in POST_HOC_CONTRASTS:
        by_setting = {
            setting: one_row(
                contrasts=contrasts,
                family="post_hoc",
                treatment=contrast.treatment,
                reference=contrast.reference,
                setting=setting,
            )
            for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING)
        }
        if by_setting[PRIMARY_SETTING] is None:
            continue
        combined = settings_verdict(
            contrasts=contrasts,
            family="post_hoc",
            treatment=contrast.treatment,
            reference=contrast.reference,
        )
        versus_control = (
            None
            if contrast.control is None
            else settings_verdict(
                contrasts=contrasts,
                family="post_hoc_control",
                treatment=contrast.treatment,
                reference=contrast.control,
            )
        )
        rows.append(
            {
                "label": contrast.label,
                "treatment": contrast.treatment,
                "reference": contrast.reference,
                "control": contrast.control,
                "primary": by_setting[PRIMARY_SETTING]["verdict_adjusted"],  # ty: ignore[not-subscriptable]
                "sensitivity": None
                if by_setting[SENSITIVITY_SETTING] is None
                else by_setting[SENSITIVITY_SETTING]["verdict_adjusted"],  # ty: ignore[not-subscriptable]
                "combined": combined,
                "versus_control": versus_control,
                "final": combined
                if versus_control is None
                else gain_after_control(versus_reference=combined, versus_control=versus_control),
            }
        )
    return rows


def production_verdict(*, contrasts: pl.DataFrame) -> VerdictType | None:
    """Return the combined verdict of AIFS Single added to the ensemble mean, or `None`.

    Args:
        contrasts: The contrast table.

    Returns:
        The verdict of `e_a` minus `e0` from `combine_verdicts` over the two settings, or `None` if
        the `ens` view was not fitted.
    """
    if not any(row["label"] == "E-A" for row in post_hoc_verdict_rows(contrasts=contrasts)):
        return None
    return settings_verdict(
        contrasts=contrasts, family="post_hoc", treatment=E_A_ARM, reference=E0_ARM
    )


def render_post_hoc_production(*, fits: Fits, contrasts: pl.DataFrame) -> list[str]:
    """Render the production-relevant second-forecast contrast, which is post hoc and exploratory.

    Args:
        fits: Every fit.
        contrasts: The contrast table.

    Returns:
        The section's lines, or a line saying the analysis is not yet fitted.
    """
    lines = [
        "## Production-relevant second-forecast contrast (post hoc, exploratory)",
        "",
        (
            "P4 and P5 add a second forecast to a single deterministic 9 km point forecast "
            "(Open-Meteo's IFS run). This section adds it to the IFS ensemble mean, which is what "
            "the production service reads. The contrasts were added after the first scientific "
            "review, so they are post hoc and exploratory. Their verdicts below apply the plan's "
            "formula for information only."
        ),
        "",
    ]
    verdict_rows = post_hoc_verdict_rows(contrasts=contrasts)
    if not verdict_rows:
        return [*lines, f"{POST_HOC_NOT_FITTED}: {', '.join(fits.missing_post_hoc)}.", ""]
    lines += [
        render_contrasts(contrasts=contrasts, families=["post_hoc", "post_hoc_control"]),
        "### Verdicts by the plan's formula (post hoc, for information)",
        "",
        table(
            rows=[
                [
                    str(row["label"]),
                    f"{row['treatment']} minus {row['reference']}",
                    str(row["primary"]),
                    str(row["sensitivity"]),
                    str(row["combined"]),
                    "n/a"
                    if row["control"] is None
                    else f"{row['treatment']} minus {row['control']}: {row['versus_control']}",
                    f"**{row['final']}**",
                ]
                for row in verdict_rows
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
        ),
        "",
        "### The ERA5 study's P4 beside them (99.5% interval, the ERA5 study's own level)",
        "",
        render_era5_p4(contrasts=contrasts),
        "",
        "### Lead-day values (95% intervals, post hoc, exploratory)",
        "",
    ]
    rows = []
    for contrast in POST_HOC_CONTRASTS:
        for day in BLEND_LEAD_DAYS:
            for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING):
                row = one_row(
                    contrasts=contrasts,
                    family="post_hoc",
                    treatment=contrast.treatment,
                    reference=contrast.reference,
                    setting=setting,
                    scope=f"lead day {day}",
                )
                if row is not None:
                    rows.append(
                        [
                            contrast.label,
                            str(day),
                            setting,
                            points(row["difference"], signed=True),  # ty: ignore[invalid-argument-type]
                            interval(row["lower_95"], row["upper_95"]),  # ty: ignore[invalid-argument-type]
                        ]
                    )
    lines.append(
        table(
            rows=rows,
            header=["contrast", "lead day", "setting", "difference (points)", "95% interval"],
        )
    )
    return lines


def render_by_fold(*, fits: Fits, contrasts: pl.DataFrame) -> list[str]:
    """Render P4 and P5 by blend fold, and pooled and by lead day without the winter fold.

    Args:
        fits: Every fit.
        contrasts: The contrast table.

    Returns:
        The section's lines. The values are at the main setting; they are exploratory.
    """
    lines = [
        "## P4 and P5 by blend fold and without the winter fold (exploratory)",
        "",
    ]
    blend = fits.get(view="blend", target="pv", lead_days=BLEND_LEAD_DAYS)
    if not blend:
        return [*lines, "No blend fits."]
    primary = {
        day: losses.filter(pl.col("setting") == PRIMARY_SETTING) for day, losses in blend.items()
    }
    all_rows = pl.concat(primary.values())
    folds = winter_folds(losses=all_rows)
    months = (
        all_rows.group_by("fold")
        .agg(first=pl.col("month").min(), last=pl.col("month").max())
        .sort("fold")
    )
    lines += [
        (
            "Each fold holds a block of whole months. The folds that hold a December, January, or "
            f"February month are {', '.join(str(f) for f in folds)}; the models that score those "
            "months train on no winter month of the other folds. Main setting throughout."
        ),
        "",
        table(
            rows=[
                [str(row["fold"]), row["first"], row["last"]]
                for row in months.iter_rows(named=True)
            ],
            header=["fold", "first month", "last month"],
        ),
        "",
    ]
    blend_planned = [planned for planned in PLANNED_CONTRASTS if planned.view == "blend"]
    rows = []
    for planned in blend_planned:
        for day, losses in primary.items():
            by_fold = mean_paired_difference_by(
                losses=losses,
                treatment=planned.treatment,
                reference=planned.reference,
                metric=METRIC,
                by="fold",
            )
            rows.extend(
                [
                    planned.label,
                    str(day),
                    str(row["fold"]),
                    points(row["difference"], signed=True),
                    f"{row['rows'] // len(SEEDS):,}",
                ]
                for row in by_fold.iter_rows(named=True)
            )
    lines += [
        "### By fold",
        "",
        table(
            rows=rows,
            header=["contrast", "lead day", "fold", "difference (points)", "rows (per seed)"],
        ),
        "",
        "### Without the winter fold",
        "",
    ]
    rows = []
    for planned in blend_planned:
        by_day = lead_day_mean_differences(fits=fits, planned=planned)
        for scope, label in [
            *((f"lead day {day} without winter fold", str(day)) for day in BLEND_LEAD_DAYS),
            ("pooled 1-3 without winter fold", "pooled 1-3"),
        ]:
            without = one_row(
                contrasts=contrasts,
                family="blend_no_winter",
                treatment=planned.treatment,
                reference=planned.reference,
                setting=PRIMARY_SETTING,
                scope=scope,
            )
            if label == "pooled 1-3":
                with_all = pooled_difference(
                    contrasts=contrasts,
                    family="planned",
                    treatment=planned.treatment,
                    reference=planned.reference,
                )
            else:
                with_all = by_day.get(int(label), (None, 0))[0]
            rows.append(
                [
                    planned.label,
                    label,
                    points(with_all, signed=True),
                    "n/a" if without is None else points(without["difference"], signed=True),  # ty: ignore[invalid-argument-type]
                    "n/a"
                    if without is None
                    else interval(without["lower_95"], without["upper_95"]),  # ty: ignore[invalid-argument-type]
                ]
            )
    lines.append(
        table(
            rows=rows,
            header=[
                "contrast",
                "lead day",
                "all folds (points)",
                "without the winter fold (points)",
                "95% interval without the winter fold",
            ],
        )
    )
    return lines


def render_raw_errors(*, fits: Fits) -> list[str]:
    """Render the forecasts' raw radiation errors against CAMS irradiance, and F0 on both fits.

    Args:
        fits: Every fit.

    Returns:
        The section's lines. The raw errors need `dataset_blend_ens.parquet`.
    """
    lines = ["## The reference's own error against CAMS irradiance (post hoc)", ""]
    if fits.blend_ens is None:
        return [
            *lines,
            (
                f"{POST_HOC_NOT_FITTED}: dataset_blend_ens.parquet is absent; run the build with "
                "`--ens-only`."
            ),
        ]
    lines += [
        (
            "Raw mean absolute error in W m-2 of each forecast's radiation against CAMS "
            "irradiance at the valid hour, on the blend rows where all three forecasts are "
            "present."
        ),
        "",
    ]
    errors: dict[int, dict[str, float]] = {}
    counts: dict[int, int] = {}
    for day in BLEND_LEAD_DAYS:
        frame = fits.blend_ens.filter(pl.col("lead_day") == day)
        errors[day] = mean_absolute_error_on_common_rows(
            frame=frame,
            forecast_columns=list(RAW_RADIATION_COLUMNS),
            truth_column=CAMS_IRRADIANCE_COLUMN,
        )
        counts[day] = frame.select(
            pl.all_horizontal(
                pl.col(name).is_not_null() & ~pl.col(name).is_nan().fill_null(value=False)
                for name in (*RAW_RADIATION_COLUMNS, CAMS_IRRADIANCE_COLUMN)
            ).sum()
        ).item()
    first, last = BLEND_LEAD_DAYS[0], BLEND_LEAD_DAYS[-1]
    lines += [
        table(
            rows=[
                [
                    str(day),
                    f"{counts[day]:,}",
                    *(f"{errors[day][name]:.1f}" for name in RAW_RADIATION_COLUMNS),
                ]
                for day in BLEND_LEAD_DAYS
            ]
            + [
                [
                    f"rise, lead day {first} to {last} (%)",
                    "",
                    *(
                        f"{percentage_rise(first=errors[first][name], last=errors[last][name]):.1f}"
                        for name in RAW_RADIATION_COLUMNS
                    ),
                ]
            ],
            header=[
                "lead day",
                "rows",
                *(f"{label} (W m-2)" for label in RAW_RADIATION_COLUMNS.values()),
            ],
        ),
        "",
        "### F0 on the ladder fit and on the blend refit, on the same rows (main setting)",
        "",
    ]
    rows = []
    for day in BLEND_LEAD_DAYS:
        ladder = fits.losses.get(("ladder", "pv", day))
        blend = fits.losses.get(("blend", "pv", day))
        if ladder is None or blend is None:
            continue
        blend_f0 = blend.filter((pl.col("arm") == "f0") & (pl.col("setting") == PRIMARY_SETTING))
        ladder_f0 = ladder.filter(
            (pl.col("arm") == "f0") & (pl.col("setting") == PRIMARY_SETTING)
        ).join(blend_f0.select("site", "time").unique(), on=["site", "time"], how="semi")
        ladder_error = float(ladder_f0[METRIC].mean())  # ty: ignore[invalid-argument-type]
        blend_error = float(blend_f0[METRIC].mean())  # ty: ignore[invalid-argument-type]
        rows.append(
            [
                str(day),
                f"{blend_f0.height // len(SEEDS):,}",
                points(ladder_error),
                points(blend_error),
                points(blend_error - ladder_error, signed=True),
            ]
        )
    lines.append(
        table(
            rows=rows,
            header=[
                "lead day",
                "rows (per seed)",
                "ladder F0 (% of capacity)",
                "blend refit of F0 (% of capacity)",
                "blend minus ladder (points)",
            ],
        )
    )
    return lines


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
    planned_months = (
        contrasts.filter(pl.col("family") == "planned")
        .group_by("label")
        .agg(pl.col("n_months").max())
        .sort("label")
    )
    months_text = ", ".join(f"{label} {months}" for label, months in planned_months.iter_rows())
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
            f"each interval covers {ADJUSTED_LEVEL_PERCENT:g}%. The intervals resample whole "
            f"months ({months_text}), and so few months undercover, so they are approximate."
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
        f"{split_intervals} split intervals), all with 95% intervals; the pooled post hoc "
        "production-relevant contrasts also carry 99% intervals. About 1 in 20 with no real effect "
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
    *,
    verdicts: pl.DataFrame,
    evidence: pl.DataFrame,
    contrasts: pl.DataFrame,
    third: ThirdDecision,
    needs_second: bool,
    production: VerdictType | None,
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
        *render_drop_one_rule(evidence=evidence),
        "",
        *render_against_production(contrasts=contrasts),
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
        (
            "- The cloud layers' drop-one raise and the smallest effect are printed under the "
            "drop-one rule above. F2 minus FP, printed above, is the contrast closest to the "
            "production feed. The production feed's long-wave radiation was not tested."
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
        (
            "**Scope of the third decision.** P5 adds the second forecast to a single 9 km point "
            "forecast (Open-Meteo's IFS run), and the ERA5 study's P4 adds MARS-only variables to "
            "a reanalysis. Neither is the gain the production service would see, because it reads "
            "the IFS ensemble mean. The production-relevant contrast is the post hoc `e_a` minus "
            "`e0` (AIFS Single added to the ensemble mean). The outcome above keeps its "
            "pre-registered formula. "
            + production_scope_sentence(verdict=production)
            + " The smallest effect is "
            f"{points(SMALLEST_EFFECT['pv'])} points."
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
    production: VerdictType | None,
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
            verdicts=verdicts,
            evidence=evidence,
            contrasts=contrasts,
            third=third,
            needs_second=needs_second,
            production=production,
        ),
        "",
        "## Planned contrasts (planned)",
        "",
        render_contrasts(contrasts=contrasts, families=["planned", "planned_control"]),
        *render_post_hoc_production(fits=fits, contrasts=contrasts),
        "",
        *render_by_fold(fits=fits, contrasts=contrasts),
        "",
        *render_raw_errors(fits=fits),
        "",
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
        *render_low_sun(splits=splits),
        "",
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


def report_refit_needed() -> int:
    """Print which post hoc fits and frames are missing, with the commands that make them.

    Returns:
        The process exit code, always 0, because missing post hoc fits are not an error.
    """
    fits = Fits()
    missing = list(fits.missing_post_hoc)
    if fits.blend_ens is None:
        missing.append(ens_dataset_path().name)
    if not missing:
        print("every post hoc fit and frame exists")
        return 0
    print(f"{POST_HOC_NOT_FITTED}: {', '.join(missing)}")
    print("build:  ifs_ladder_build_dataset.py --ens-only")
    print("fit:    ifs_ladder_fit.py --view ens --lead-days 1 2 3")
    print("fit:    ifs_ladder_fit.py --view extra_sensitivity --lead-days 1 2 3")
    return 0


def main() -> int:
    """Score the saved fits and write the report and tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--width-preview-only",
        action="store_true",
        help="Read only the ERA5 losses and write width_preview.md, before any IFS fit exists.",
    )
    parser.add_argument(
        "--refit-needed",
        action="store_true",
        help="Print the post hoc fits and frames that are missing, then exit without a report.",
    )
    arguments = parser.parse_args()
    if arguments.width_preview_only:
        return write_width_preview()
    if arguments.refit_needed:
        return report_refit_needed()
    paths = report_paths()
    refuse_to_overwrite(paths=list(paths))
    fits = Fits()
    if not fits.losses:
        msg = "no fit has saved losses; run ifs_ladder_fit.py first"
        raise FileNotFoundError(msg)

    contrast_rows = [
        *planned_rows(fits=fits),
        *exploratory_rows(fits=fits),
        *post_hoc_rows(fits=fits),
        *era5_p4_rows(),
    ]
    contrasts = pl.DataFrame(contrast_rows, infer_schema_length=None)
    era5_p4 = contrasts.filter(pl.col("family") == "era5_p4")
    verdicts = planned_verdict_table(contrasts=contrasts)
    evidence = group_evidence(contrasts=contrasts, verdicts=verdicts)
    third = third_decision(verdicts=verdicts, era5_p4=era5_p4)
    needs_second = second_feed(evidence=evidence, verdicts=verdicts)
    production = production_verdict(contrasts=contrasts)
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
            production=production,
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
