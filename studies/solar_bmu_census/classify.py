"""Classify each BMU as solar, not solar, or without output, from its settled half-hourly output.

No Elexon field identifies every solar Balancing Mechanism Unit (BMU). The BMU register's fuel type
never says solar, and the Installed Generation Capacity per Unit (IGCPU) report types only some
solar BMUs as Solar, so the census also asks whether a BMU's output follows the sun. The feature is
the Pearson correlation between half-hourly output and the cosine of the solar zenith angle (clipped
at zero below the horizon) at one central point in Great Britain. Run:
`uv run python studies/solar_bmu_census/classify.py`.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final, Literal

import numpy as np
import polars as pl
from fetch_sources import (
    OUTPUT_DIR,
    STUDY_DIR,
    b1610_bmu_ids,
    fetch_bmu_reference,
    fetch_igcpu,
    recorded_run,
)
from studies.solar import cos_zenith, zenith

REFERENCE_LATITUDE: Final[float] = 53.0
REFERENCE_LONGITUDE: Final[float] = -1.5
"""A point near the middle of Great Britain.

A BMU has no coordinates, so one point stands in for every BMU. Across Great Britain, solar noon
moves by about 16 minutes with longitude. A 16-minute shift is small beside the half-hour
resolution of the output.
"""
SOLAR_CORRELATION_THRESHOLD: Final[float] = 0.6
"""The correlation above which a BMU's output counts as following the sun.

Among single-site BMUs (`T_`, `E_`, `M_`) with at least `MIN_POSITIVE_HALF_HOURS` positive
half-hours, a wide gap separates the correlations of the BMUs that follow the sun from the
correlations of the BMUs that do not. The census page's first figure shows the gap. The threshold
sits inside the gap. Aggregate BMUs show no gap between the two groups.
"""
COMMISSIONING_SKIP: Final[timedelta] = timedelta(days=30)
"""How long after a BMU's first positive output the BMU's half-hours are left out of the analysis.

The exception is a BMU that was already running when the window opened (see `RUNNING_AT_START`). A
site often commissions in stages over a few weeks, so its output in those 30 days follows the sun
badly or only partly, whatever the site's technology.
"""
RUNNING_AT_START: Final[timedelta] = timedelta(days=7)
"""How soon after the window starts a BMU's first positive output must fall for the BMU to count as
already running.

The first month of an already-running BMU is not a commissioning month, so the month stays in the
analysis.
"""
POSITIVE_FLOOR_MWH: Final[float] = 0.01
"""Output at or below this many megawatt-hours in a half-hour is meter noise, not generation.

BMUs that never generated in the window were seen to publish readings of a few thousandths of a
megawatt-hour.
"""
SINGLE_SITE_PREFIXES: Final[tuple[str, ...]] = ("T_", "E_", "M_")
"""A BMU with one of these prefixes is one generating site. `2_` (supplier), `V_`, and `C_` BMUs
can aggregate many sites."""
DAYLIGHT_COS_ZENITH: Final[float] = 0.1
"""The sun is clearly up when the cosine of its zenith angle exceeds this (a zenith of about 84°).

At the one reference point, a smaller cosine is twilight, where zero output is not a fault. In
twilight the sun may also be down at one end of the country and up at the other.
"""
MIN_POSITIVE_HALF_HOURS: Final[int] = 100
"""Fewer positive half-hours than this leaves too little output to judge.

The count is taken over the judged series, after the commissioning period and the daytime zeros are
removed, and a BMU below it is classed `no_output`.
"""

BehaviourType = Literal["solar", "no_output", "not_solar"]


@dataclass(frozen=True)
class Behaviour:
    """What a BMU's output looks like over the window."""

    correlation: float | None
    raw_correlation: float | None
    positive_half_hours: int
    behaviour: BehaviourType


def analysis_series(*, output: pl.DataFrame, window_start: datetime) -> pl.DataFrame:
    """Return the half-hours the classifier judges, with the sun's height at each.

    A BMU already running when the window opens is judged on every half-hour. Any other BMU is
    judged from `COMMISSIONING_SKIP` after its first positive output. A BMU can publish months of
    zero or near-zero readings before it first generates. Those readings would dilute the
    correlation of a BMU that follows the sun closely once running. A site also commissions in
    stages over its first weeks, and a half-built site follows the sun badly.

    Args:
        output: Columns `half_hour_end_time` (UTC) and `output_mwh`.
        window_start: The start of the study window, in UTC.

    Returns:
        Columns `half_hour_end_time`, `output_mwh`, and `cos_zenith` (the cosine of the solar zenith
        at the half-hour's midpoint, zero below the horizon), sorted by time. The frame is empty
        when the BMU's output never exceeds `POSITIVE_FLOOR_MWH`.
    """
    ordered = output.sort("half_hour_end_time")
    positive_times = ordered.filter(pl.col("output_mwh") > POSITIVE_FLOOR_MWH)["half_hour_end_time"]
    if positive_times.is_empty():
        return ordered.clear().with_columns(cos_zenith=pl.lit(None, dtype=pl.Float64))
    first_positive = positive_times.min()
    if not isinstance(first_positive, datetime):
        raise TypeError("The half-hour end times must be datetimes")
    if first_positive < window_start + RUNNING_AT_START:
        series = ordered
    else:
        series = ordered.filter(pl.col("half_hour_end_time") >= first_positive + COMMISSIONING_SKIP)
    midpoints = series["half_hour_end_time"].dt.offset_by("-15m")
    sun = cos_zenith(
        zenith_deg=zenith(
            stamps=midpoints, latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE
        )
    )
    return series.with_columns(cos_zenith=pl.Series(sun))


def drop_daytime_zeros(*, series: pl.DataFrame) -> pl.DataFrame:
    """Remove the half-hours with exactly zero output while the sun is clearly up.

    A solar BMU does not output exactly zero at midday, so an exact zero at midday is a metering
    fault. A wind or gas unit's daytime zeros are real. Removing them leaves that unit's output
    positive by day and zero by night, which raises the unit's correlation with the sun. The census
    therefore reports the correlation with and without this step.

    Args:
        series: `analysis_series`'s output.

    Returns:
        The series without rows whose output is zero and whose `cos_zenith` exceeds
        `DAYLIGHT_COS_ZENITH`.
    """
    return series.filter(
        (pl.col("output_mwh") != 0) | (pl.col("cos_zenith") <= DAYLIGHT_COS_ZENITH)
    )


def sun_following_correlation(*, series: pl.DataFrame) -> float | None:
    """Return the correlation of output with the cosine of the solar zenith.

    Args:
        series: `analysis_series`'s output, with or without `drop_daytime_zeros` applied.

    Returns:
        The Pearson correlation, or None when it is undefined: fewer than two half-hours, or an
        output or sun series with no variation.
    """
    if series.height < 2:
        return None
    values = series["output_mwh"].to_numpy()
    sun = series["cos_zenith"].to_numpy()
    if np.std(values) == 0 or np.std(sun) == 0:
        return None
    return float(np.corrcoef(values, sun)[0, 1])


def classify_behaviour(*, output: pl.DataFrame, window_start: datetime) -> Behaviour:
    """Classify a BMU by its output alone.

    The rules apply in order. A BMU is `no_output` when the BMU has fewer than
    `MIN_POSITIVE_HALF_HOURS` positive half-hours or an undefined correlation. Otherwise the BMU is
    `solar` when its correlation exceeds `SOLAR_CORRELATION_THRESHOLD`, and `not_solar` when it does
    not. `no_output` comes first, because a constant series has no defined correlation.

    Args:
        output: Columns `half_hour_end_time` (UTC) and `output_mwh`.
        window_start: The start of the study window, in UTC.

    Returns:
        The correlation with daytime zeros removed, the correlation with them kept
        (`raw_correlation`), the number of positive half-hours in the judged series, and the class.
    """
    series = analysis_series(output=output, window_start=window_start)
    cleaned = drop_daytime_zeros(series=series)
    positive_half_hours = int((cleaned["output_mwh"] > POSITIVE_FLOOR_MWH).sum())
    correlation = sun_following_correlation(series=cleaned)
    raw_correlation = sun_following_correlation(series=series)
    if positive_half_hours < MIN_POSITIVE_HALF_HOURS or correlation is None:
        behaviour: BehaviourType = "no_output"
    elif correlation > SOLAR_CORRELATION_THRESHOLD:
        behaviour = "solar"
    else:
        behaviour = "not_solar"
    return Behaviour(
        correlation=correlation,
        raw_correlation=raw_correlation,
        positive_half_hours=positive_half_hours,
        behaviour=behaviour,
    )


def p99_output_mw(*, output: pl.DataFrame, window_start: datetime) -> float | None:
    """Return the 99th percentile of a BMU's half-hourly output, in megawatts.

    The percentile is taken over the series the classifier judges: the half-hours from
    `analysis_series` (so after the commissioning period) with `drop_daytime_zeros` applied (so
    without the exact zeros while the sun is up). Zeros at night, and negative readings, stay in.
    The percentile is linear-interpolated, and the half-hourly megawatt-hours are multiplied by 2 to
    give megawatts. The result is a measure of the BMU's observed output, not a registered capacity.

    Args:
        output: Columns `half_hour_end_time` (UTC) and `output_mwh`.
        window_start: The start of the study window, in UTC.

    Returns:
        The 99th percentile in megawatts, or None when no half-hour is judged.
    """
    judged = drop_daytime_zeros(series=analysis_series(output=output, window_start=window_start))
    if judged.is_empty():
        return None
    percentile = judged["output_mwh"].quantile(0.99, interpolation="linear")
    return None if percentile is None else float(percentile) * 2


def igcpu_solar_ids(*, igcpu: list[dict[str, Any]]) -> set[str]:
    """Return the BMU identifiers that IGCPU registers with resource type "Solar"."""
    return {str(row["bmUnit"]) for row in igcpu if row["psrType"] == "Solar" and row["bmUnit"]}


def classify_all() -> pl.DataFrame:
    """Classify every BMU, and return one row per BMU.

    The BMUs are the non-interconnector BMUs in the reference data, and every BMU that IGCPU types
    as Solar that the reference data lacks. The window and the run date come from
    `fetch_sources.py`'s `lineage.json`. The function raises for a fetched BMU with no file for that
    window, because an empty frame would class the BMU `no_output` silently.

    Returns:
        Columns: `elexon_bmu_id`, `scope` (`single-site` or `aggregate`), `correlation` (daytime
        zeros removed), `raw_correlation` (not removed), `positive_half_hours` (in the judged
        series), `half_hours` (the rows in the BMU's B1610 file, which has no row for a half-hour
        that B1610 did not publish), `behaviour`, `igcpu_solar`, `is_solar` (behaviour solar or
        IGCPU Solar), and `basis`, which says what put the BMU in the census: `type and behaviour`,
        `type only`, `behaviour only`, or `neither`.

    Raises:
        FileNotFoundError: If a BMU in the reference data has no B1610 file for the window.
    """
    today, window = recorded_run()
    reference = fetch_bmu_reference()
    igcpu_ids = igcpu_solar_ids(igcpu=fetch_igcpu(today=today))
    fetched_ids = b1610_bmu_ids(reference=reference)
    rows = []
    for bmu_id in fetched_ids + sorted(igcpu_ids - set(fetched_ids)):
        path = OUTPUT_DIR / f"{bmu_id}_{window.label}.parquet"
        if path.exists():
            output = pl.read_parquet(path)
        elif bmu_id in fetched_ids:
            raise FileNotFoundError(f"No B1610 file for {bmu_id} in window {window.label}")
        else:
            output = empty_output()
        result = classify_behaviour(output=output, window_start=window.start)
        by_type = bmu_id in igcpu_ids
        by_behaviour = result.behaviour == "solar"
        rows.append(
            {
                "elexon_bmu_id": bmu_id,
                "scope": "single-site" if bmu_id.startswith(SINGLE_SITE_PREFIXES) else "aggregate",
                "correlation": result.correlation,
                "raw_correlation": result.raw_correlation,
                "positive_half_hours": result.positive_half_hours,
                "half_hours": output.height,
                "behaviour": result.behaviour,
                "igcpu_solar": by_type,
                "is_solar": by_type or by_behaviour,
                "basis": census_basis(by_type=by_type, by_behaviour=by_behaviour),
            }
        )
    return pl.DataFrame(
        rows, schema_overrides={"correlation": pl.Float64, "raw_correlation": pl.Float64}
    )


def empty_output() -> pl.DataFrame:
    """Return the output frame of an IGCPU-typed BMU that B1610 has no file for."""
    return pl.DataFrame(
        schema={"half_hour_end_time": pl.Datetime("us", "UTC"), "output_mwh": pl.Float64}
    )


def census_basis(*, by_type: bool, by_behaviour: bool) -> str:
    """Say what put a BMU in the census: its output, its IGCPU type, both, or neither."""
    if by_type and by_behaviour:
        return "type and behaviour"
    if by_type:
        return "type only"
    if by_behaviour:
        return "behaviour only"
    return "neither"


def main() -> None:
    """Classify every BMU and write `classes.parquet`."""
    classes = classify_all()
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    classes.write_parquet(STUDY_DIR / "classes.parquet")
    print(classes["basis"].value_counts().sort("basis"))
    print(f"Wrote {STUDY_DIR / 'classes.parquet'}")


if __name__ == "__main__":
    main()
