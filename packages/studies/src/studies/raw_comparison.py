"""Compare two products' raw values directly, with no forecasting model between them.

A paired forecast contrast (`studies.bootstrap`) always measures a weather product through the lens
of the XGBoost model fitted to it, so it cannot say whether two products already agree, or disagree,
before any model sees them. `raw_column_comparison` answers that narrower question: how one column
compares to a reference column, row for row.
"""

from typing import TypedDict

import polars as pl


class RawColumnComparison(TypedDict):
    """The bias, mean absolute difference, and correlation between two columns."""

    bias: float
    mad: float
    correlation: float


def raw_column_comparison(
    *, frame: pl.DataFrame, treatment: str, reference: str
) -> RawColumnComparison:
    """Return the treatment column's bias, mean absolute difference, and correlation with reference.

    No model or aggregation sits between the two columns: this is a row-for-row comparison of raw
    values, meant to check whether a forecast contrast built from these columns is describing a real
    difference between the products, or one that a shared model would recalibrate away.

    Args:
        frame: Rows holding both columns, already restricted to the rows worth comparing (for
            example, daylight hours only).
        treatment: The column being compared.
        reference: The column it is compared against.

    Returns:
        `bias` (treatment minus reference, mean), `mad` (mean absolute difference), and
        `correlation` (Pearson correlation between the two columns).
    """
    difference = pl.col(treatment) - pl.col(reference)
    row = frame.select(
        bias=difference.mean(),
        mad=difference.abs().mean(),
        correlation=pl.corr(treatment, reference),
    ).row(0, named=True)
    return RawColumnComparison(
        bias=float(row["bias"]), mad=float(row["mad"]), correlation=float(row["correlation"])
    )
