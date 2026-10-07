"""Classify each BMU as solar, not solar, or without output, from its settled half-hourly output.

No Elexon field says which BMUs are solar, so the census asks whether a BMU's output follows the
sun. The feature is the Pearson correlation between half-hourly output and the cosine of the solar
zenith angle (clipped at zero below the horizon) at one central point in Great Britain. Run:
`uv run python studies/solar_bmu_census/classify.py`.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final, Literal

import numpy as np
import polars as pl
from fetch_sources import (
    INPUTS_DIR,
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

A BMU has no coordinates, and across Great Britain the solar noon moves by about 16 minutes with
longitude, which is small beside the half-hour resolution of the output.
"""
SOLAR_CORRELATION_THRESHOLD: Final[float] = 0.6
"""The correlation above which a BMU's output counts as following the sun.

Among single-site BMUs (`T_`, `E_`, `M_`) with enough output to judge, the correlations of the
BMUs that follow the sun and those that do not are separated by a wide gap, which the census
page's first figure shows. The threshold sits inside that gap. Aggregate BMUs have no such gap.
"""
COMMISSIONING_SKIP: Final[timedelta] = timedelta(days=30)
"""How long after its first positive output a BMU is left out of the analysis.

A site often commissions in stages over a few weeks, so its output in that month follows the sun
badly or only partly, whatever the site is.
"""
DAYLIGHT_COS_ZENITH: Final[float] = 0.1
"""The sun is clearly up when the cosine of its zenith angle exceeds this (a zenith of about 84°).

At the one reference point, a smaller value is twilight, where zero output is not a fault and the
sun may be down at one end of the country and up at the other.
"""
MIN_POSITIVE_HALF_HOURS: Final[int] = 100
"""Fewer positive half-hours than this in the window leaves too little output to judge."""

BehaviourType = Literal["solar", "no_output", "not_solar"]


@dataclass(frozen=True)
class Behaviour:
    """What a BMU's output looks like over the window."""

    correlation: float | None
    raw_correlation: float | None
    positive_half_hours: int
    behaviour: BehaviourType


def analysis_series(*, output: pl.DataFrame) -> pl.DataFrame:
    """Return the half-hours the classifier judges, with the sun's height at each.

    The series starts `COMMISSIONING_SKIP` after the first positive output. B1610 publishes no row
    for a unit before it begins generating, and a site commissions in stages over its first weeks,
    so months of silence before the first output would dilute the correlation of a unit that
    follows the sun closely once running, and a half-built site follows it badly.

    Args:
        output: Columns `half_hour_end_time` (UTC) and `output_mwh`.

    Returns:
        Columns `half_hour_end_time`, `output_mwh`, and `cos_zenith` (the cosine of the solar zenith
        at the half-hour's midpoint, zero below the horizon), sorted by time. The frame is empty
        when the BMU never has positive output.
    """
    ordered = output.sort("half_hour_end_time")
    positive_times = ordered.filter(pl.col("output_mwh") > 0)["half_hour_end_time"]
    if positive_times.is_empty():
        return ordered.clear().with_columns(cos_zenith=pl.lit(None, dtype=pl.Float64))
    start = positive_times.dt.offset_by(f"{COMMISSIONING_SKIP.days}d").min()
    series = ordered.filter(pl.col("half_hour_end_time") >= start)
    midpoints = series["half_hour_end_time"].dt.offset_by("-15m")
    sun = cos_zenith(
        zenith_deg=zenith(
            stamps=midpoints, latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE
        )
    )
    return series.with_columns(cos_zenith=pl.Series(sun))


def drop_daytime_zeros(*, series: pl.DataFrame) -> pl.DataFrame:
    """Remove the half-hours with exactly zero output while the sun is clearly up.

    A solar BMU does not output exactly zero at midday, so such a reading is a metering fault. A
    wind or gas unit's daytime zeros are real, so removing them leaves that unit's output
    positive by day and zero by night, which raises its correlation with the sun. The census
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


def classify_behaviour(*, output: pl.DataFrame) -> Behaviour:
    """Classify a BMU by its output alone.

    The rules apply in order. `no_output` comes first, because a constant series has no defined
    correlation and a NaN would otherwise compare as greater than the threshold.

    Args:
        output: Columns `half_hour_end_time` (UTC) and `output_mwh`.

    Returns:
        The correlation, the number of positive half-hours after the commissioning month, and the
        class.
    """
    series = analysis_series(output=output)
    cleaned = drop_daytime_zeros(series=series)
    positive_half_hours = int((cleaned["output_mwh"] > 0).sum())
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


def igcpu_solar_ids(*, igcpu: list[dict[str, Any]]) -> set[str]:
    """Return the BMU identifiers that IGCPU registers with resource type "Solar"."""
    return {str(row["bmUnit"]) for row in igcpu if row["psrType"] == "Solar" and row["bmUnit"]}


def classify_all() -> pl.DataFrame:
    """Classify every non-interconnector BMU, and return one row per BMU.

    The window and the run date come from `fetch_sources.py`'s `lineage.json`. A BMU whose file for
    that window is missing raises, because an empty frame would class it `no_output` silently.

    Returns:
        Columns: `elexon_bmu_id`, `correlation` (daytime zeros removed), `raw_correlation` (not
        removed), `positive_half_hours` (after the commissioning month), `half_hours` (in the whole
        window), `behaviour`, `igcpu_solar`, `is_solar` (behaviour solar or IGCPU Solar), and
        `basis`, a string saying
        which of the two put the BMU in the census.
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
            output = _empty_output()
        result = classify_behaviour(output=output)
        by_type = bmu_id in igcpu_ids
        by_behaviour = result.behaviour == "solar"
        rows.append(
            {
                "elexon_bmu_id": bmu_id,
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


def _empty_output() -> pl.DataFrame:
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
    """Classify every BMU and write `classes.parquet` and a private listing."""
    classes = classify_all()
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    classes.write_parquet(STUDY_DIR / "classes.parquet")
    classes.write_csv(STUDY_DIR / "classes.csv")
    print(classes["basis"].value_counts().sort("basis"))
    print(f"Wrote {INPUTS_DIR.parent / 'classes.parquet'}")


if __name__ == "__main__":
    main()
