"""Say why a capacity estimate moved, in one sentence, and score how explainable each estimator is.

Part of the study in issue #1143. The maintainer wants the capacity estimator to be explainable, and
the study to lay groundwork for models built wholly from differentiable physics. This module holds
the fixed rubric, scored by reading each estimator's code and outputs before any result exists, and
the sentence the report prints for a planted event or a real low month.
"""

from collections.abc import Mapping
from typing import Final, Literal

import polars as pl

from studies.capacity_estimators import ESTIMATOR_FAMILY

GradeType = Literal["yes", "partly", "no"]
"""A rubric grade."""

CRITERIA: Final[tuple[tuple[str, str], ...]] = (
    (
        "one_sentence",
        "An operator can be told in one sentence why the capacity fell on a given day",
    ),
    ("physical_units", "The estimator's fitted quantities are physical quantities with units"),
    (
        "attributable",
        "A change can be attributed to a cause: soiling, a lost string, curtailment, or bias",
    ),
    ("spread", "The estimator returns a spread, not only a value"),
    (
        "differentiable",
        "The estimator can be differentiated through, for joint fitting with other models",
    ),
)
"""The rubric's five criteria: a key and the question the grader answers."""

RUBRIC: Final[Mapping[str, Mapping[str, GradeType]]] = {
    "EM": {
        "one_sentence": "yes",
        "physical_units": "partly",
        "attributable": "no",
        "spread": "no",
        "differentiable": "no",
    },
    "R30": {
        "one_sentence": "yes",
        "physical_units": "partly",
        "attributable": "no",
        "spread": "no",
        "differentiable": "no",
    },
    "CSQ": {
        "one_sentence": "yes",
        "physical_units": "partly",
        "attributable": "no",
        "spread": "no",
        "differentiable": "no",
    },
    "NR": {
        "one_sentence": "yes",
        "physical_units": "partly",
        "attributable": "partly",
        "spread": "no",
        "differentiable": "no",
    },
    "PR": {
        "one_sentence": "yes",
        "physical_units": "partly",
        "attributable": "partly",
        "spread": "no",
        "differentiable": "no",
    },
    "KF": {
        "one_sentence": "yes",
        "physical_units": "yes",
        "attributable": "partly",
        "spread": "yes",
        "differentiable": "partly",
    },
    "HM": {
        "one_sentence": "yes",
        "physical_units": "yes",
        "attributable": "partly",
        "spread": "partly",
        "differentiable": "no",
    },
    "CX1": {
        "one_sentence": "partly",
        "physical_units": "partly",
        "attributable": "partly",
        "spread": "no",
        "differentiable": "partly",
    },
    "CX2": {
        "one_sentence": "yes",
        "physical_units": "yes",
        "attributable": "yes",
        "spread": "no",
        "differentiable": "partly",
    },
    "DP1": {
        "one_sentence": "yes",
        "physical_units": "yes",
        "attributable": "partly",
        "spread": "yes",
        "differentiable": "yes",
    },
}
"""Each estimator's grades, from reading its code. The reasons are in the table below."""

RUBRIC_REASONS: Final[Mapping[str, str]] = {
    "EM": "reports the highest power seen so far; it can only rise, so a loss is invisible",
    "R30": "reports the 30-day 99th percentile of output; no model of the cause",
    "CSQ": "reports a quantile of output per unit of clear-sky irradiance; no model of the cause",
    "NR": "removes shared weather by comparing with the other plants, so a loss is not the weather",
    "PR": "divides out the satellite irradiance, so a loss is not the weather the satellite saw",
    "KF": "an availability factor with a standard deviation; a random walk cannot separate causes",
    "HM": "capacity is a share of whole units (strings or inverters); a fall reads as units lost",
    "CX1": "the fused lasso gives a changepoint with a date and a size; one trace mixes causes",
    "CX2": (
        "separate direct-current and alternating-current traces tell a lost string from a lost"
        " inverter, and the hinge handles curtailment"
    ),
    "DP1": (
        "tilt, azimuth, and both capacities are physical, with a posterior; curtailment is"
        " handled by the loss, not named"
    ),
}
"""One line per estimator on why it earned its grades."""


def rubric_table() -> pl.DataFrame:
    """Return the rubric as a table, one row per estimator.

    Returns:
        `estimator`, `family`, one column per criterion key, `score` (yes = 1, partly = 0.5,
        no = 0, summed), and `reason`.
    """
    points = {"yes": 1.0, "partly": 0.5, "no": 0.0}
    rows = []
    for name, grades in RUBRIC.items():
        rows.append(
            {
                "estimator": name,
                "family": ESTIMATOR_FAMILY[name],  # type: ignore[index]
                **grades,
                "score": sum(points[grade] for grade in grades.values()),
                "reason": RUBRIC_REASONS[name],
            }
        )
    return pl.DataFrame(rows)


def fitted_physical_quantities(
    *,
    orientation: Mapping[str, float],
    scale_mw: float,
    static_p99_mw: float,
    relative_sd: float | None,
) -> dict[str, str]:
    """Describe a plant's fitted physical quantities with units, on one day.

    Args:
        orientation: The plant's `tilt_deg`, `azimuth_deg`, `dc_capacity_mw`, and `ac_capacity_mw`
            from its orientation fit.
        scale_mw: The estimator's capacity on the day.
        static_p99_mw: The plant's static 99th-percentile power.
        relative_sd: The estimate's relative standard deviation, or `None` for an estimator
            without one.

    Returns:
        Each quantity as a string with its unit. The availability factor is the capacity over the
        static power, a dimensionless number.
    """
    factor = scale_mw / static_p99_mw
    ratio = orientation["dc_capacity_mw"] / orientation["ac_capacity_mw"]
    spread = "" if relative_sd is None else f" +/- {100 * relative_sd * factor:.1f} points"
    return {
        "tilt": f"{orientation['tilt_deg']:.1f} degrees",
        "azimuth": f"{orientation['azimuth_deg']:.1f} degrees clockwise from north",
        "dc_to_ac_ratio": f"{ratio:.2f} (MW per MW)",
        "availability_factor": f"{100 * factor:.1f}% of the static 99th-percentile power{spread}",
        "capacity": f"{scale_mw:.2f} MW",
    }


def explanation_sentence(
    *,
    estimator: str,
    when: str,
    scale_mw: float,
    earlier_scale_mw: float,
    earlier_days: int,
    relative_sd: float | None,
    where: str = "the plant",
) -> str:
    """Return the one-sentence reason an estimator gives for the capacity on a day.

    Args:
        estimator: The estimator's name.
        when: The day, such as `2025-03-14`.
        scale_mw: The capacity on the day.
        earlier_scale_mw: The capacity `earlier_days` days before.
        earlier_days: How far back the comparison reaches.
        relative_sd: The relative standard deviation, or `None`.
        where: How to refer to the plant; the low-output plant's label is never printed next to a
            date.

    Returns:
        A sentence naming the change and what the estimator compared to reach it.
    """
    change = 100 * (scale_mw / earlier_scale_mw - 1.0)
    basis = {
        "EM": "the highest hourly power it has ever recorded",
        "R30": "the 99th percentile of its last 30 days of hourly power",
        "CSQ": "the 90th percentile of its last 30 days of output per unit of clear-sky irradiance",
        "NR": "its last 7 days of energy compared with the other plants' over the same days",
        "PR": "its last 7 days of energy per unit of satellite irradiance",
        "KF": "the filtered ratio of its daily energy to the physics model's output",
        "HM": "the capacity level, as a share of whole units, that best fits its daily energy",
        "CX1": "a fused-lasso trace fitted to its output against the clear-sky envelope",
        "CX2": (
            "a fused-lasso direct-current trace fitted to its output against the satellite-driven"
            " physics model"
        ),
        "DP1": "a weekly direct-current capacity fitted through the physics model",
    }[estimator]
    spread = "" if relative_sd is None else f" (standard deviation {100 * relative_sd:.1f}%)"
    direction = "fell" if change < 0 else "rose"
    return (
        f"{estimator}: on {when}, {where}'s capacity {direction} {abs(change):.1f}% in "
        f"{earlier_days} days to {scale_mw:.2f} MW{spread}, judged from {basis}."
    )
