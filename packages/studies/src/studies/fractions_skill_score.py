"""Score a forecast on whether it put a threshold exceedance near the right hour.

Mean absolute error charges a forecast twice for a peak it places an hour late: once for the power
it predicted that did not arrive, and once for the power that arrived unpredicted. A forecast that
gets a spike's magnitude right and its timing wrong therefore scores worse than one that never
predicts a spike at all. The Fractions Skill Score measures how much of that penalty is timing
rather than magnitude.

[Evaluation
metrics](https://openclimatefix.github.io/nged-substation-forecast/techniques/evaluation-metrics/#fractions-skill-score-fss)
explains what the score measures and the two limits on reading it.
"""

from typing import Final

import polars as pl

MONTH_FORMAT: Final[str] = "%Y-%m"
"""How a month is labelled when the bootstrap resamples whole months.

A calendar month rather than a month-of-year, so that March 2024 and March 2025 are different
blocks. Errors within a day are not independent, and a block has to hold each weather episode
together.
"""


def on_a_complete_hourly_grid(*, frame: pl.DataFrame, thresholds: pl.DataFrame) -> pl.DataFrame:
    """Place one arm's exceedance indicators on an unbroken hourly index per site.

    **A rolling window over the scored rows alone would silently span a night**, joining the hours
    either side of a 12-hour gap into one 3-hour window. Reindexing to every hour between a site's
    first and last scored hour leaves a null wherever a row is absent, and `monthly_components`
    rejects any window holding one.

    Args:
        frame: One arm and seed's rows, carrying `site`, `time`, `power_mw` and `forecast_mw`.
        thresholds: Each site's exceedance threshold, as `site` and `threshold_mw`.

    Returns:
        The complete grid, carrying `month` and with `exceeded_observed` and `exceeded_forecast`
        null off the scored rows.
    """
    spans = frame.group_by("site").agg(
        first_time=pl.col("time").min(), last_time=pl.col("time").max()
    )
    grid = (
        spans.with_columns(
            time=pl.datetime_ranges(
                pl.col("first_time"), pl.col("last_time"), interval="1h", time_zone="UTC"
            )
        )
        # Never empty here, because first_time <= last_time, so the value cannot matter. It is
        # stated because Polars warns that the default changes in 2.0.
        .explode("time", empty_as_null=True)
        .select("site", "time")
    )
    return (
        grid.join(frame.join(thresholds, on="site", how="inner"), on=["site", "time"], how="left")
        .with_columns(
            month=pl.col("time").dt.strftime(MONTH_FORMAT),
            exceeded_observed=(pl.col("power_mw") > pl.col("threshold_mw")).cast(pl.Float64),
            exceeded_forecast=(pl.col("forecast_mw") > pl.col("threshold_mw")).cast(pl.Float64),
        )
        .sort("site", "time")
    )


def monthly_components(*, gridded: pl.DataFrame, window_hours: int) -> pl.DataFrame:
    """Reduce one arm and seed to the per-month sums a score is assembled from.

    The score is a ratio of two means over windows, so keeping the numerator and denominator as
    monthly sums lets a bootstrap resample whole months and re-form the ratio, rather than
    resampling hours that are correlated within a day.

    Args:
        gridded: One arm and seed's exceedance indicators, from `on_a_complete_hourly_grid`.
        window_hours: The width of the centred window, in hours. A width of 1 is the point score,
            carrying the full double penalty; each wider width forgives a displacement of up to
            half the window either side.

    Returns:
        One row per month, carrying the Fractions Brier Score numerator, its reference, and how
        many complete windows each was summed over.
    """
    fractions = gridded.with_columns(
        observed_fraction=pl.col("exceeded_observed")
        .rolling_mean(window_size=window_hours, min_samples=window_hours, center=True)
        .over("site"),
        forecast_fraction=pl.col("exceeded_forecast")
        .rolling_mean(window_size=window_hours, min_samples=window_hours, center=True)
        .over("site"),
    ).drop_nulls(["observed_fraction", "forecast_fraction"])
    return fractions.group_by("month").agg(
        squared_difference=(pl.col("forecast_fraction") - pl.col("observed_fraction")).pow(2).sum(),
        reference=(pl.col("forecast_fraction").pow(2) + pl.col("observed_fraction").pow(2)).sum(),
        windows=pl.len(),
    )


def fss_from(*, squared_difference: float, reference: float) -> float:
    """Form the Fractions Skill Score from its two summed components.

    Args:
        squared_difference: The summed squared difference of the two window fractions.
        reference: The summed reference, being both fractions' summed squares.

    Returns:
        The score, 1 for a forecast whose exceedances coincide with the observed exceedances at
        this tolerance, and 0 for a forecast with no skill over the reference. `nan` where no
        window at this tolerance held an exceedance in either series, leaving the reference at
        zero.
    """
    if reference == 0.0:
        return float("nan")
    return 1.0 - squared_difference / reference
