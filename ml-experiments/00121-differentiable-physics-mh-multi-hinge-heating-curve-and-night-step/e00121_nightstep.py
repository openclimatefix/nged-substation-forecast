"""Experiment 121: does the forecast show the 01:00 step on cold winter nights?

A **cold winter night** is a night from 22:00 to 08:00 local time in November to February whose
local calendar day (the day on which the night's midnight falls) has a daily mean ERA5 temperature
below 8 C at the substation's cell. For each decision substation, year, and arm, the script prints
the median over cold winter weekday nights of the forecast step (median forecast of the half-hour
starting 00:30 minus the median forecast of the half-hour starting 00:00) against the actual step
(the same difference in observed power), in points, with a 95% interval from resampling whole
nights, and the median profile from 22:00 to 08:00 for the cards' figure.

    OMP_NUM_THREADS=1 uv run python ml-experiments/00121-*/e00121_nightstep.py
"""

import json
from typing import Any, Final

import e00121_common as common
import e00121_report as report
import numpy as np
import polars as pl

ARMS: Final[tuple[str, ...]] = ("b0_k", "ns_l_k", "ns_u_k", "ns_h_k", "ns_hw_k", "ns_ut_k")
YEARS: Final[tuple[str, ...]] = ("replicate", "scored")
COLD_NIGHT_C: Final[float] = report.COLD_BST_THRESHOLD_C
N_RESAMPLES: Final[int] = 2000
STEP_HALF_HOUR: Final[int] = 1
BASE_HALF_HOUR: Final[int] = 0
NIGHT_HALF_HOURS_LOCAL: Final[tuple[int, ...]] = (*range(44, 48), *range(16))
SEED_TAG: Final[int] = 8


def night_frame(*, year: str, arm: str) -> pl.DataFrame:
    """Return the rows of cold winter weekday nights with the night date and half-hour of day."""
    frame = report.load_arm(year=year, arm=arm, drop_switching=False)
    frame = frame.with_columns(
        half_hour=pl.col("local_start").dt.hour() * 2 + pl.col("local_start").dt.minute() // 30,
        night_date=(pl.col("local_start") + pl.duration(hours=2)).dt.date(),
    )
    temperatures = (
        report.cold_days()
        .select("site_name", "local_date", "tmean")
        .rename({"local_date": "night_date", "tmean": "night_tmean"})
    )
    frame = frame.join(temperatures, on=["site_name", "night_date"], how="left")
    return frame.filter(
        pl.col("winter_night")
        & (pl.col("night_tmean") < COLD_NIGHT_C)
        & (pl.col("local_start").dt.weekday() <= 5)
        & pl.col("site_name").is_in(common.DECISION_SITES)
    )


def step_row(*, frame: pl.DataFrame, site: str, rng: np.random.Generator) -> dict[str, Any]:
    """Return one substation's forecast step, actual step, and 95% intervals."""
    part = frame.filter(pl.col("site_name") == site)
    wide = (
        part.filter(pl.col("half_hour").is_in([BASE_HALF_HOUR, STEP_HALF_HOUR]))
        .group_by("night_date", "half_hour")
        .agg(pl.col("q50").first(), pl.col("actual").first())
        .pivot(on="half_hour", index="night_date", values=["q50", "actual"])
        .drop_nulls()
    )
    names = {c: c for c in wide.columns}
    forecast = (
        wide[f"q50_{STEP_HALF_HOUR}"].to_numpy() - wide[f"q50_{BASE_HALF_HOUR}"].to_numpy()
        if f"q50_{STEP_HALF_HOUR}" in names
        else np.array([])
    )
    actual = (
        wide[f"actual_{STEP_HALF_HOUR}"].to_numpy() - wide[f"actual_{BASE_HALF_HOUR}"].to_numpy()
        if f"actual_{STEP_HALF_HOUR}" in names
        else np.array([])
    )
    nights = len(forecast)
    if nights == 0:
        return {"site": site, "nights": 0}
    picks = rng.integers(0, nights, size=(N_RESAMPLES, nights))
    forecast_draws = np.median(forecast[picks], axis=1)
    actual_draws = np.median(actual[picks], axis=1)
    gap_draws = np.median((forecast - actual)[picks], axis=1)
    return {
        "site": site,
        "nights": nights,
        "forecast_step": float(np.median(forecast)),
        "forecast_step_95": [float(x) for x in np.percentile(forecast_draws, [2.5, 97.5])],
        "actual_step": float(np.median(actual)),
        "actual_step_95": [float(x) for x in np.percentile(actual_draws, [2.5, 97.5])],
        "forecast_minus_actual": float(np.median(forecast - actual)),
        "forecast_minus_actual_95": [float(x) for x in np.percentile(gap_draws, [2.5, 97.5])],
    }


def profile(*, frame: pl.DataFrame, site: str) -> list[dict[str, Any]]:
    """Return the median observed power and forecast median and 80% band by half-hour of night."""
    part = frame.filter(pl.col("site_name") == site)
    grouped = part.group_by("half_hour").agg(
        pl.col("actual").median().alias("actual"),
        pl.col("q50").median().alias("q50"),
        pl.col("q10").median().alias("q10"),
        pl.col("q90").median().alias("q90"),
    )
    order = {h: position for position, h in enumerate(NIGHT_HALF_HOURS_LOCAL)}
    rows = grouped.to_dicts()
    return sorted(rows, key=lambda row: order[row["half_hour"]])


def other_misses(*, frame: pl.DataFrame, site: str) -> dict[str, Any]:
    """Return the half-hour with the largest median miss (forecast median minus actual)."""
    part = frame.filter(pl.col("site_name") == site)
    grouped = part.group_by("half_hour").agg((pl.col("q50") - pl.col("actual")).median().alias("m"))
    worst = grouped.sort(pl.col("m").abs(), descending=True).row(0, named=True)
    return {"site": site, "half_hour_start": worst["half_hour"], "median_miss_points": worst["m"]}


def main() -> None:
    """Print the readout and write `work/nightstep.json`."""
    capacity = {"D5": 5.723, "D10": 5.398, "D11": 10.777, "D14": 11.706, "D16": 9.115}
    out: dict[str, Any] = {"cold_night_threshold_c": COLD_NIGHT_C, "capacity_mw": capacity}
    for year in YEARS:
        for arm in ARMS:
            frame = night_frame(year=year, arm=arm)
            rng = np.random.default_rng([common.EXPERIMENT_ID, SEED_TAG, YEARS.index(year)])
            rows = [step_row(frame=frame, site=s, rng=rng) for s in common.DECISION_SITES]
            out[f"{year}_{arm}"] = {
                "steps": rows,
                "profiles": {s: profile(frame=frame, site=s) for s in common.DECISION_SITES},
                "largest_miss": [other_misses(frame=frame, site=s) for s in common.DECISION_SITES],
            }
            print(f"\n== {year} {arm} ==")
            for row in rows:
                print(row)
    (common.WORK_DIR / "nightstep.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
