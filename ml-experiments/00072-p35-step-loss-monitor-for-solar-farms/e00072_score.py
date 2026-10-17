"""Score the frozen monitors on injected losses, the scored placebo, and the unmodified record.

**Run once, after `e00072_tune.py` has frozen the parameters.** Writes four tables under
`data/ml-experiments/00072-p35-step-loss-monitor-for-solar-farms/`:

- `events.parquet`: one row per planted loss and monitor, with whether the monitor detected it.
- `placement.parquet`: one row per attempted run, saying whether its losses could be placed.
- `placebo.parquet`: one row per scored shuffle, farm, and monitor, with the alarm count.
- `unmodified.parquet`: one row per alarm of each monitor on the unmodified record.
- `low_months.parquet`: one row per low month of farm A and monitor, with whether it was found.
"""

import json
from collections.abc import Mapping
from typing import Final

import numpy as np
import polars as pl
from e00072_detectors import Alarm, fitted_level
from e00072_inject import (
    DURATIONS,
    EARLY_WINDOW_DAYS,
    METHODS,
    PERMANENT_SIZES,
    SIZES,
    AlarmFunction,
    Loss,
    MethodType,
    SeasonType,
    detected,
    farm_years,
    inject,
    make_alarm_function,
    place_losses,
)
from e00072_signal import (
    FARM_A_LOW_MONTHS,
    POOL_SITES,
    clean_baseline,
    farm_signal,
    load_record,
)
from e00072_tune import FROZEN_PATH, placebo_signals
from studies.guards import refuse_to_overwrite
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUTPUT_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00072-p35-step-loss-monitor-for-solar-farms"
SCORED_DRAWS: Final[int] = 20
SCORED_SEED_BASE: Final[int] = 1000
SEASONS: Final[tuple[SeasonType, ...]] = ("apr_sep", "oct_mar")
LOW_MONTH_SHARE_REQUIRED: Final[float] = 0.5
LOW_MONTH_DROP: Final[float] = 0.05
LOW_MONTH_GUARD_DAYS: Final[int] = 90
"""Onsets at farm A stay this many days from its low months, and alarms within 30 days of them are
set aside on the unmodified record."""


def _runs() -> list[tuple[str, list[tuple[float, int | None]]]]:
    """Return each run's name and its losses' sizes and durations."""
    runs: list[tuple[str, list[tuple[float, int | None]]]] = [
        (f"temporary_{days}", [(size, days) for size in SIZES]) for days in DURATIONS
    ]
    runs += [(f"permanent_{size}", [(size, None)]) for size in PERMANENT_SIZES]
    return runs


def injected_rows(
    *,
    parameters: dict[MethodType, float],
    extra_alarm_functions: Mapping[str, AlarmFunction] | None = None,
    only_extra: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Plant losses in each pool farm and record whether each monitor detects them.

    Args:
        parameters: The frozen parameter of each monitor.
        extra_alarm_functions: Further monitors to score on the same losses, by name.
        only_extra: Score only the extra monitors.

    Returns:
        One row per planted loss and monitor, and one row per attempted run saying whether its
        losses could be placed.
    """
    record = load_record()
    baseline = clean_baseline(record=record)
    alarm_functions: dict[str, AlarmFunction] = (
        {}
        if only_extra
        else {m: make_alarm_function(method=m, parameter=parameters[m]) for m in METHODS}
    )
    alarm_functions.update(extra_alarm_functions or {})
    rows = []
    attempts: list[dict[str, object]] = []
    for site_index, site in enumerate(POOL_SITES):
        farm = record.sites.index(site)
        finite = np.flatnonzero(np.isfinite(baseline.signal[farm]))
        valid_range = (int(finite[0]), int(finite[-1]) + 1)
        allowed = np.ones(len(record.dates), dtype=bool)
        if site == "A":
            for month in FARM_A_LOW_MONTHS:
                days_in_month = np.flatnonzero(
                    record.dates.astype("datetime64[M]") == np.datetime64(month)
                )
                allowed[
                    max(days_in_month[0] - LOW_MONTH_GUARD_DAYS, 0) : days_in_month[-1]
                    + LOW_MONTH_GUARD_DAYS
                ] = False
        for draw in range(SCORED_DRAWS):
            for season_index, season in enumerate(SEASONS):
                for run_index, (run_name, specs) in enumerate(_runs()):
                    seed = np.random.SeedSequence(
                        [SCORED_SEED_BASE + draw, season_index, site_index, run_index]
                    )
                    losses = place_losses(
                        rng=np.random.default_rng(seed),
                        dates=record.dates,
                        valid_range=valid_range,
                        allowed=allowed,
                        specs=specs,
                        season=season,
                    )
                    attempts.append(
                        {
                            "site": site,
                            "season": season,
                            "run": run_name,
                            "placed": losses is not None,
                        }
                    )
                    if losses is None:
                        continue
                    signal = farm_signal(
                        record=record,
                        baseline=baseline,
                        farm=farm,
                        power_mw=inject(power_mw=record.power_mw[farm], losses=losses),
                    )
                    for method, alarm_function in alarm_functions.items():
                        alarms = alarm_function(signal)
                        rows.extend(
                            _row(
                                loss=loss,
                                alarms=alarms,
                                site=site,
                                draw=draw,
                                run=run_name,
                                method=method,
                            )
                            for loss in losses
                        )
    return pl.DataFrame(rows), pl.DataFrame(attempts)


def _row(
    *, loss: Loss, alarms: list[Alarm], site: str, draw: int, run: str, method: str
) -> dict[str, object]:
    """Return one table row for a planted loss."""
    alarm = detected(alarms=alarms, loss=loss)
    return {
        "detected_early_window": detected(alarms=alarms, loss=loss, early_days=EARLY_WINDOW_DAYS)
        is not None,
        "site": site,
        "draw": draw,
        "run": run,
        "method": method,
        "season": loss.season,
        "size": loss.size,
        "days": -1 if loss.days is None else loss.days,
        "first_day": loss.first_day,
        "detected": alarm is not None,
        "delay_days": None if alarm is None else alarm.day - loss.first_day,
        "size_error": None if alarm is None else alarm.size - loss.size,
    }


def placebo_rows(*, parameters: dict[MethodType, float]) -> pl.DataFrame:
    """Count each monitor's alarms on the scored week-shuffled placebo.

    Args:
        parameters: The frozen parameter of each monitor.

    Returns:
        One row per shuffle, pool farm, and monitor.
    """
    rows = []
    for draw in range(SCORED_DRAWS):
        signals = placebo_signals(seed=SCORED_SEED_BASE + draw)
        for site, (signal, farm_span) in zip(POOL_SITES, signals, strict=True):
            for method in METHODS:
                alarms = make_alarm_function(method=method, parameter=parameters[method])(signal)
                rows.append(
                    {
                        "draw": draw,
                        "site": site,
                        "method": method,
                        "alarms": len(alarms),
                        "farm_years": farm_span,
                    }
                )
    return pl.DataFrame(rows)


def unmodified_rows(*, parameters: dict[MethodType, float]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """List the alarms on the unmodified record and test farm A's low months.

    Args:
        parameters: The frozen parameter of each monitor.

    Returns:
        The alarm table (every farm including E, flagged when within 30 days of farm A's low
        months) and the low-month table.
    """
    record = load_record()
    baseline = clean_baseline(record=record)
    months_of_days = record.dates.astype("datetime64[M]")
    month_spans = []
    for month in FARM_A_LOW_MONTHS:
        days_in_month = np.flatnonzero(months_of_days == np.datetime64(month))
        month_spans.append((int(days_in_month[0]), int(days_in_month[-1])))
    alarm_rows = []
    for farm, site in enumerate(record.sites):
        signal = baseline.signal[farm]
        for method in METHODS:
            for alarm in make_alarm_function(method=method, parameter=parameters[method])(signal):
                near_low = site == "A" and any(
                    first - 30 <= alarm.day <= last + 60 for first, last in month_spans
                )
                alarm_rows.append(
                    {
                        "site": site,
                        "method": method,
                        "date": str(record.dates[alarm.day]),
                        "size": alarm.size,
                        "near_farm_a_low_months": near_low,
                        "farm_years": farm_years(signal=signal),
                    }
                )
    farm_a = record.sites.index("A")
    signal_a = baseline.signal[farm_a]
    median_level = float(np.nanmedian(np.where(baseline.low_months, np.nan, signal_a)))
    level = fitted_level(signal=signal_a, penalty_scale=parameters["changepoint"])
    months = record.dates.astype("datetime64[M]")
    low_rows = []
    for month in FARM_A_LOW_MONTHS:
        in_month = months == np.datetime64(month)
        valid = in_month & np.isfinite(signal_a)
        below = (level[valid] <= median_level + np.log(1.0 - LOW_MONTH_DROP)).mean()
        low_rows.append(
            {
                "month": month,
                "method": "changepoint",
                "valid_days": int(valid.sum()),
                "share_below": float(below),
                "found": bool(below >= LOW_MONTH_SHARE_REQUIRED),
            }
        )
        first, last = np.flatnonzero(in_month)[[0, -1]]
        for method in ("rolling_median", "hyst"):
            alarms = make_alarm_function(method=method, parameter=parameters[method])(signal_a)
            hit = any(first - 14 <= alarm.day <= last for alarm in alarms)
            low_rows.append(
                {
                    "month": month,
                    "method": method,
                    "valid_days": int(valid.sum()),
                    "share_below": None,
                    "found": hit,
                }
            )
    return pl.DataFrame(alarm_rows), pl.DataFrame(low_rows)


def main() -> None:
    """Score everything and write the four tables."""
    frozen = json.loads(FROZEN_PATH.read_text())
    parameters = {method: float(frozen[method]["parameter"]) for method in METHODS}
    paths = {
        name: OUTPUT_DIR / f"{name}.parquet"
        for name in ("events", "placement", "placebo", "unmodified", "low_months")
    }
    refuse_to_overwrite(paths=paths.values())
    unmodified, low_months = unmodified_rows(parameters=parameters)
    unmodified.write_parquet(paths["unmodified"])
    low_months.write_parquet(paths["low_months"])
    placebo_rows(parameters=parameters).write_parquet(paths["placebo"])
    events, placement = injected_rows(parameters=parameters)
    events.write_parquet(paths["events"])
    placement.write_parquet(paths["placement"])
    unplaced = placement.filter(~pl.col("placed")).height
    if unplaced:
        print(f"WARNING: {unplaced} of {placement.height} runs could not place their losses")
    print("wrote", ", ".join(str(path) for path in paths.values()))


if __name__ == "__main__":
    main()
