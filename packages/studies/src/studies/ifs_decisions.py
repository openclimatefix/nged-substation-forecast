"""The decision rules of the IFS solar-variables study, as functions the report script applies.

Written for the study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1137>. The plan fixes every rule
before any result exists, so each rule is a pure function of intervals and differences, and
`studies/ifs_solar_variables/ifs_ladder_report.py` only supplies the numbers.

**Every contrast is a treatment's error minus a reference's error, so a negative difference is a
gain.** `contrast_verdict` reads one interval against the smallest effect worth acting on,
`combine_verdicts` needs both hyperparameter settings to agree, and `gain_after_control` makes a
planned gain stand only if the same gain also shows against the negative control that has the same
columns.

**Lead days are pooled by stacking.** The lead days are separate XGBoost models whose rows stand
for the same months, so `stack_lead_days` gives each lead day's rows a `site` key of their own and
lets one month resample serve all of them.
"""

from collections.abc import Mapping, Sequence
from typing import Final, Literal

import polars as pl

VerdictType = Literal["gain", "no gain", "unresolved"]
"""What an interval says about the smallest effect: a gain, the absence of one, or neither."""

OutcomeType = Literal["second_forecast", "mars_variables", "not_separated"]
"""What the third decision recommends: a second forecast, the MARS-only variables, or neither."""

EvidenceClassType = Literal["planned gain", "exploratory gain", "drop-one gain", "no gain shown"]
"""How strongly the study supports asking for a group of variables."""

EVIDENCE_CLASSES: Final[tuple[EvidenceClassType, ...]] = (
    "planned gain",
    "exploratory gain",
    "drop-one gain",
    "no gain shown",
)
"""The evidence classes from the strongest to the weakest, which is the priority list's order."""

SETTINGS_THAT_MUST_AGREE: Final[int] = 2
"""How many hyperparameter settings a verdict needs, all returning it."""

LEAD_DAY_SITE_SEPARATOR: Final[str] = "-L"
"""Joins a farm label and a lead day in a stacked `site` key, as in `A-L1`."""


def contrast_verdict(*, lower: float, upper: float, smallest_effect: float) -> VerdictType:
    """Read a treatment-minus-reference interval against the smallest effect worth acting on.

    Args:
        lower: The interval's lower bound.
        upper: The interval's upper bound.
        smallest_effect: The smallest improvement worth acting on, a positive number in the same
            unit as the interval.

    Returns:
        `gain` if the upper bound is below minus the smallest effect, so that the whole interval is
        a larger gain than the smallest. `no gain` if the lower bound is above minus the smallest
        effect, so that a gain as large as the smallest effect is ruled out. Otherwise
        `unresolved`. A bound exactly on minus the smallest effect satisfies neither test.
    """
    if upper < -smallest_effect:
        return "gain"
    if lower > -smallest_effect:
        return "no gain"
    return "unresolved"


def combine_verdicts(*, verdicts: Sequence[VerdictType]) -> VerdictType:
    """Combine one contrast's verdicts at the hyperparameter settings into one.

    Args:
        verdicts: The verdict at each setting that was fitted.

    Returns:
        `gain` or `no gain` only if every one of exactly `SETTINGS_THAT_MUST_AGREE` settings
        returns it, and `unresolved` otherwise, including when a setting is missing.
    """
    if len(verdicts) != SETTINGS_THAT_MUST_AGREE:
        return "unresolved"
    first = verdicts[0]
    return first if all(verdict == first for verdict in verdicts) else "unresolved"


def gain_after_control(
    *, versus_reference: VerdictType, versus_control: VerdictType
) -> VerdictType:
    """Let a planned gain stand only if it also shows against the negative control.

    The control has the treatment's columns with the new variables' information removed, so a gain
    over the reference that is not also a gain over the control may come from the extra columns.

    Args:
        versus_reference: The combined verdict of treatment minus reference.
        versus_control: The combined verdict of treatment minus the control.

    Returns:
        `versus_reference`, except that `gain` becomes `unresolved` when `versus_control` is not
        `gain`.
    """
    if versus_reference == "gain" and versus_control != "gain":
        return "unresolved"
    return versus_reference


def second_forecast_or_mars_variables(
    *, second_forecast_verdict: VerdictType, mars_uppers: Sequence[float], smallest_effect: float
) -> OutcomeType:
    """Apply the third decision's rule: a second forecast, the MARS-only variables, or neither.

    Args:
        second_forecast_verdict: The combined verdict of the gain from a second forecast on top of
            every IFS variable (contrast P5).
        mars_uppers: The upper bound of the adjusted interval for the gain from the MARS-only
            variables on top of every other ERA5 variable (the ERA5 study's P4), one per
            hyperparameter setting.
        smallest_effect: The smallest improvement worth acting on, a positive number.

    Returns:
        `second_forecast` if `second_forecast_verdict` is `gain`. `mars_variables` if it is
        `no gain` and every MARS upper bound, from exactly `SETTINGS_THAT_MUST_AGREE` settings, is
        below minus the smallest effect. `not_separated` otherwise.
    """
    if second_forecast_verdict == "gain":
        return "second_forecast"
    mars_gain = len(mars_uppers) == SETTINGS_THAT_MUST_AGREE and all(
        upper < -smallest_effect for upper in mars_uppers
    )
    if second_forecast_verdict == "no gain" and mars_gain:
        return "mars_variables"
    return "not_separated"


def second_feed_recommended(
    *, cloud_layers_verdict: VerdictType, missing_group_drop_one_gains: Sequence[bool]
) -> bool:
    """Apply the second decision's rule: whether the production forecast needs a second IFS feed.

    Args:
        cloud_layers_verdict: The combined verdict of the cloud layers' gain (contrast P2).
        missing_group_drop_one_gains: For each group of variables that the free open-data feed
            lacks, whether the drop-one run finds the group carrying a gain.

    Returns:
        Whether the cloud layers are a planned gain or any missing group carries a gain.
    """
    return cloud_layers_verdict == "gain" or any(missing_group_drop_one_gains)


def exploratory_gain(
    *, uppers: Sequence[float], difference: float, control_difference: float
) -> bool:
    """Say whether an exploratory gain holds at every setting and lead day and beats the control.

    Args:
        uppers: The upper bound of the 95% interval of the group's step at each combination of
            hyperparameter setting and lead day that must agree.
        difference: The group's pooled difference from its base arm.
        control_difference: The negative control's pooled difference from the same base arm.

    Returns:
        Whether there is at least one bound, every bound is below zero, and the group's difference
        is more negative than the control's.
    """
    return (
        len(uppers) > 0 and all(upper < 0.0 for upper in uppers) and difference < control_difference
    )


def drop_one_matters(*, raise_in_error: float, control_difference: float) -> bool:
    """Say whether removing a group raises the full set's error by more than the control's change.

    Args:
        raise_in_error: The error of the full set without the group minus the full set's error.
        control_difference: The negative control's difference from its base arm. A control that
            lowered the error counts as a difference of zero.

    Returns:
        Whether `raise_in_error` is more than `control_difference`, and above zero.
    """
    return raise_in_error > max(control_difference, 0.0)


def evidence_class(
    *,
    planned_verdict: VerdictType | None,
    planned_needs_drop_one: bool,
    has_exploratory_gain: bool,
    has_drop_one_gain: bool,
) -> EvidenceClassType:
    """Return the evidence class of a group of variables, the strongest that applies.

    Args:
        planned_verdict: The group's planned contrast verdict after the control, or `None` if the
            group has none.
        planned_needs_drop_one: Whether the planned contrast covers several groups, so that a
            group takes the planned class only if its own drop-one run also finds it carrying the
            gain (contrast P3 covers four groups).
        has_exploratory_gain: The result of `exploratory_gain`.
        has_drop_one_gain: The result of `drop_one_matters`.

    Returns:
        `planned gain` if the planned verdict is `gain` and, where the contrast covers several
        groups, the drop-one run agrees. Otherwise `exploratory gain`, then `drop-one gain`, then
        `no gain shown`.
    """
    if planned_verdict == "gain" and (has_drop_one_gain or not planned_needs_drop_one):
        return "planned gain"
    if has_exploratory_gain:
        return "exploratory gain"
    if has_drop_one_gain:
        return "drop-one gain"
    return "no gain shown"


def priority_order(
    *, classes: Mapping[str, EvidenceClassType], gains: Mapping[str, float]
) -> list[str]:
    """Order groups for the priority list: by evidence class, then by the size of the gain.

    Args:
        classes: Each group's evidence class.
        gains: Each group's pooled gain in the metric's unit, positive for a gain.

    Returns:
        The group names from the first to ask for to the last. Ties keep the order of `classes`.
    """
    return sorted(
        classes,
        key=lambda name: (EVIDENCE_CLASSES.index(classes[name]), -gains[name]),
    )


def stack_lead_days(*, losses_by_lead_day: Mapping[int, pl.DataFrame]) -> pl.DataFrame:
    """Stack several lead days' per-row losses so that one month resample serves all of them.

    Each lead day's rows get a `site` key of the form `A-L1`, so that the same farm and hour at
    two lead days are two rows, and the bootstrap resamples whole months over all the lead days at
    once. The pooled mean is a mean over the stacked rows, so it weights each lead day by its row
    count.

    Args:
        losses_by_lead_day: Each lead day's per-row losses, carrying `site`.

    Returns:
        The stacked losses, with the relabelled `site` and a `lead_day` column.

    Raises:
        ValueError: If no lead day is given, or a lead day's `site` already holds the separator.
    """
    if not losses_by_lead_day:
        msg = "no lead days to stack"
        raise ValueError(msg)
    parts: list[pl.DataFrame] = []
    for lead_day, losses in sorted(losses_by_lead_day.items()):
        if losses["site"].str.contains(LEAD_DAY_SITE_SEPARATOR).any():
            msg = f"a site label already holds {LEAD_DAY_SITE_SEPARATOR!r}: lead day {lead_day}"
            raise ValueError(msg)
        parts.append(
            losses.with_columns(
                site=pl.col("site") + f"{LEAD_DAY_SITE_SEPARATOR}{lead_day}",
                lead_day=pl.lit(lead_day, dtype=pl.Int8),
            )
        )
    return pl.concat(parts)
