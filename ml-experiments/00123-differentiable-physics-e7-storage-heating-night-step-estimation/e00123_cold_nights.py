"""Actual demand against experiment 114's baseline forecast (B0) on cold winter nights.

The rule was fixed in the plan's pre-fit clarification 3: cold nights are the cold-snap nights of
winters 2023-24 and 2024-25 (plus 2025-26 once PR 1397 has merged). For each series the script
prints the median and the 25th and 75th percentiles across nights of actual demand and of B0's
median forecast, for the half-hours starting 22:00 to 07:30, in points of effective capacity, and
a list of the largest median misses. The eight E7-X series have no stored B0 forecast.

    uv run python ml-experiments/00123-*/e00123_cold_nights.py headline
"""

import argparse
import json
from datetime import timedelta
from typing import Final

import e00123_common as common
import e00123_fit as fit
import numpy as np
import polars as pl

SLOT_MINUTES: Final[tuple[int, ...]] = tuple(range(22 * 60, 24 * 60, 30)) + tuple(
    range(24 * 60, 32 * 60, 30)
)
"""Start minutes from the evening date's midnight: 22:00 to 07:30 the next morning (20 slots)."""
MORNING_FIRST: Final[int] = (29 * 60 - 22 * 60) // 30
"""Index of the half-hour starting 05:00, where the morning-rise search starts."""
MORNING_LAST: Final[int] = (32 * 60 - 22 * 60) // 30 - 1
BAND: Final[tuple[float, float]] = (25.0, 75.0)


def slot_label(*, index: int) -> str:
    """Return the `HH:MM` start of a half-hour slot."""
    minute = SLOT_MINUTES[index] % (24 * 60)
    return f"{minute // 60:02d}:{minute % 60:02d}"


def night_matrix(*, lookup: dict, evenings: list, scale: float = 1.0) -> np.ndarray:
    """Return an array (nights, 20 slots) of values, NaN where a half-hour is missing."""
    out = np.full((len(evenings), len(SLOT_MINUTES)), np.nan)
    for row, evening in enumerate(evenings):
        base = common.datetime.combine(evening, common.time.min).replace(tzinfo=common.UTC)
        for column, minute in enumerate(SLOT_MINUTES):
            value = lookup.get(base + timedelta(minutes=minute + 30))
            if value is not None:
                out[row, column] = value * scale
    return out


def summarise(*, matrix: np.ndarray) -> dict[str, list[float]]:
    """Return the median and the 25th and 75th percentiles per slot, ignoring NaN."""
    return {
        "median": np.nanmedian(matrix, axis=0).tolist(),
        "low": np.nanpercentile(matrix, BAND[0], axis=0).tolist(),
        "high": np.nanpercentile(matrix, BAND[1], axis=0).tolist(),
    }


def largest_rise(*, profile: np.ndarray) -> int:
    """Return the index of the half-hour entered by the largest rise between 05:00 and 08:00."""
    increments = np.diff(profile)
    window = increments[MORNING_FIRST - 1 : MORNING_LAST]
    return int(np.argmax(window)) + MORNING_FIRST


def main() -> None:
    """Compute the figure's numbers and write `cold_nights_<stage>.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["headline", "all"])
    stage = parser.parse_args().stage
    fit.require(stage=stage)
    winters = (2023, 2024) if stage == "headline" else (2023, 2024, 2025)
    tags = ("tuning", "replicate", *(("scored",) if stage == "all" else ()))
    infos = common.series_infos()
    table = pl.read_parquet(common.WORK_DIR / "night_tables.parquet")
    flags = fit.cold_snap_nights(table=table)
    cold_evenings = set(flags.filter(pl.col("cold_snap"))["evening"].to_list())
    ids = [infos[s].time_series_id for s in common.FITTED_SERIES]
    power = common.load_power(time_series_ids=ids)
    result: dict[str, object] = {
        "winters": list(winters),
        "labels": [slot_label(index=i) for i in range(20)],
    }
    series: dict[str, dict] = {}
    for site in common.FITTED_SERIES:
        info = infos[site]
        nights = table.filter(
            (pl.col("site") == site)
            & (pl.col("season") == "gmt")
            & ~pl.col("excluded")
            & pl.col("winter").is_in(list(winters))
        )["evening"].to_list()
        evenings = [e for e in nights if e in cold_evenings]
        own = power.filter(pl.col("time_series_id") == info.time_series_id)
        scale = 100.0 / info.effective_capacity_mw
        actual = night_matrix(
            lookup=dict(zip(own["time"].to_list(), own["power"].to_list(), strict=True)),
            evenings=evenings,
            scale=scale,
        )
        entry: dict[str, object] = {"nights": len(evenings), "actual": summarise(matrix=actual)}
        if site in common.DECISION_SERIES:
            frames = [
                pl.read_parquet(common.EXPERIMENT_114_DIR / f"{tag}_b0" / "predictions.parquet")
                for tag in tags
            ]
            stored = pl.concat(frames).filter(pl.col("time_series_id") == info.time_series_id)
            if stored.height:
                stored = stored.unique(subset="valid_time", keep="first")
                times = stored["valid_time"].to_list()
                forecast = night_matrix(
                    lookup=dict(zip(times, stored["q50"].cast(pl.Float64).to_list(), strict=True)),
                    evenings=evenings,
                )
                measured = night_matrix(
                    lookup=dict(
                        zip(times, stored["actual"].cast(pl.Float64).to_list(), strict=True)
                    ),
                    evenings=evenings,
                )
                residual = measured - forecast
                median_residual = np.nanmedian(residual, axis=0)
                worst = int(np.argmax(np.abs(median_residual)))
                entry["forecast"] = summarise(matrix=forecast)
                entry["forecast_nights"] = int(np.isfinite(forecast).all(axis=1).sum())
                entry["median_residual"] = median_residual.tolist()
                entry["largest_median_miss"] = {
                    "slot": slot_label(index=worst),
                    "points": float(median_residual[worst]),
                }
                entry["morning_rise_actual"] = slot_label(
                    index=largest_rise(profile=np.array(entry["actual"]["median"]))  # type: ignore[index]
                )
                entry["morning_rise_forecast"] = slot_label(
                    index=largest_rise(profile=np.array(entry["forecast"]["median"]))  # type: ignore[index]
                )
        series[site] = entry
    result["series"] = series
    (common.WORK_DIR / f"cold_nights_{stage}.json").write_text(json.dumps(result, indent=1))
    for site, entry in series.items():
        line = f"{site}: {entry['nights']} cold nights"
        if "largest_median_miss" in entry:
            miss = entry["largest_median_miss"]
            line += (
                f"; largest median miss {miss['points']:+.1f} points in the half-hour starting "  # type: ignore[index]
                f"{miss['slot']}; morning rise at {entry['morning_rise_actual']} (actual) and "  # type: ignore[index]
                f"{entry['morning_rise_forecast']} (forecast)"
            )
        print(line)


if __name__ == "__main__":
    main()
