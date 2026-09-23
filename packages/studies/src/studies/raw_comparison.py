"""Compare two products' raw values directly, with no forecasting model between them.

A paired forecast contrast (`studies.bootstrap`) always measures a weather product through the lens
of the XGBoost model fitted to it, so it cannot say whether two products already agree, or disagree,
before any model sees them. `raw_column_comparison` answers that narrower question: how one column
compares to a reference column, row for row.
"""

from typing import TypedDict

import polars as pl

from studies.bootstrap import BootstrapInterval, bootstrap_row_difference


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


def mean_per_site_correlation(
    *, frame: pl.DataFrame, column: str, reference: str, site_col: str = "site"
) -> float:
    """Return one column's Pearson correlation with a reference column, averaged over each site.

    Correlating every site's rows pooled together can inherit a spurious component from
    site-to-site differences in scale or in mean level; correlating within each site first, then
    averaging, removes it. `raw_column_comparison`'s correlation uses CAMS as the reference, which
    shares SARAH-3's Meteosat inputs; a reference-free check instead correlates each product's raw
    irradiance with the generator's own measured output, which uses no weather product at all.

    Args:
        frame: Rows holding `column`, `reference` and `site_col`.
        column: The column being correlated.
        reference: The column it is correlated against.
        site_col: The column identifying each generator.

    Returns:
        The unweighted mean of each site's own Pearson correlation.
    """
    per_site = frame.group_by(site_col).agg(pl.corr(column, reference).alias("r"))
    return float(per_site.select(pl.col("r").mean()).item())


def raw_mad_difference(
    *, frame: pl.DataFrame, treatment: str, reference: str, baseline: str, month_col: str = "month"
) -> BootstrapInterval:
    """Bootstrap one product's mean absolute difference from `baseline` minus another's.

    Resamples whole months (`studies.bootstrap.bootstrap_row_difference`), since these raw values
    carry no fitting seed to resample alongside the months.

    Args:
        frame: Rows holding `treatment`, `reference`, `baseline` and `month_col`.
        treatment: The column being compared.
        reference: The column it is compared against.
        baseline: The column both are measured against (for example, CAMS's raw irradiance).
        month_col: The column carrying each row's month label.

    Returns:
        The bootstrapped difference between the two products' mean absolute difference from
        `baseline`, in the columns' own unit.
    """
    treatment_absolute = (frame[treatment] - frame[baseline]).abs()
    reference_absolute = (frame[reference] - frame[baseline]).abs()
    values = (treatment_absolute - reference_absolute).to_numpy()
    months = frame[month_col].to_numpy()
    return bootstrap_row_difference(values=values, months=months)
