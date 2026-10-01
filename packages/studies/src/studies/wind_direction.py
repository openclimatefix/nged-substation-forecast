"""Encode wind direction for a tree model, and build the information-free controls around it.

**A direction in degrees cannot go into a tree as it is,** because 359 degrees and 1 degree are
neighbours on the compass and far apart on the number line. `sine_cosine` encodes a direction as the
sine and the cosine of its angle, which puts every direction on a circle, so that the two columns
together are continuous across north. The encoding is the same whether the angle is the direction
the wind blows from or the direction it blows towards, since the two differ by 180 degrees and
negate both columns.

**Veer is the turning of the wind with height,** the difference between the direction at an upper
height and at a lower one. `veer_degrees` wraps that difference onto [-180, 180) so that a turn from
350 to 10 degrees is +20, not -340.

**A control that carries no information shuffles a column by month.** `shuffled_by_month` fills each
month's rows with another month's values, so the column keeps its distribution and its diurnal
spacing and carries nothing about the hour it sits on.
"""

from typing import Final

import numpy as np
import polars as pl

DEGREES_PER_TURN: Final[float] = 360.0
HALF_TURN_DEGREES: Final[float] = 180.0


def sine_cosine(*, direction_deg: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Return the sine and the cosine of a direction given in degrees.

    Args:
        direction_deg: A direction in degrees, on any turn (0 and 360 are the same direction).

    Returns:
        The sine and the cosine, each in [-1, 1].
    """
    radians = direction_deg.radians()
    return radians.sin(), radians.cos()


def veer_degrees(*, upper_deg: pl.Expr, lower_deg: pl.Expr) -> pl.Expr:
    """Return the turning of the wind from a lower height to an upper one, in degrees.

    Args:
        upper_deg: The direction at the upper height, in degrees.
        lower_deg: The direction at the lower height, in degrees.

    Returns:
        `upper_deg - lower_deg` wrapped to [-180, 180). A positive value is a clockwise turn with
        height.
    """
    return (upper_deg - lower_deg + HALF_TURN_DEGREES) % DEGREES_PER_TURN - HALF_TURN_DEGREES


def shuffled_by_month(
    *, values: np.ndarray, months: np.ndarray, generator: np.random.Generator
) -> np.ndarray:
    """Fill each month's rows with the values of a different month, so no row keeps its own.

    The months are put in a random cyclic order and each takes its successor's values, repeated or
    cut to its own row count.

    Args:
        values: One column of one site's values in time order.
        months: Each row's month label.
        generator: The random source.

    Returns:
        An array shaped like `values`.

    Raises:
        ValueError: If the rows hold fewer than two months.
    """
    unique = np.unique(months)
    if len(unique) < 2:
        msg = "shuffling by month needs at least two months"
        raise ValueError(msg)
    order = generator.permutation(len(unique))
    donor = {
        unique[order[position]]: unique[order[(position + 1) % len(unique)]]
        for position in range(len(unique))
    }
    shuffled = np.empty_like(values)
    for month in unique:
        rows = np.flatnonzero(months == month)
        donor_values = values[months == donor[month]]
        shuffled[rows] = donor_values[np.arange(len(rows)) % len(donor_values)]
    return shuffled
