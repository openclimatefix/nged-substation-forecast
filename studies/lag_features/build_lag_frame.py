"""Build the frames the lagged-power-features study fits on.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/1138>. The question is whether
an XGBoost solar power forecast improves when it is also shown lagged power. `docs/studies/` holds
the design and the results once published, and `plans/study-lag-features-1138.md` holds the plan.

**Rows.** `shared_rows()` rebuilds the matched-lead study's shared row set from
`original/solar_forecast_inputs.parquet` (the same requirements, the same era-aware folds), because
a study script may not import another study folder's script. `PUBLISHED_ROW_COUNT` pins the result
to the published count. Folds and `era_code` are assigned once, on that set, and every later row
drop keeps them.

**Weather.** `--weather-product ens_mean` (the default) reads `ens_mean_day<N>_ghi` and `_temp`;
`ifs_single` reads `ifs_single_day<N>_ghi` and `_temp`. Both are renamed to `nwp_ghi` and
`nwp_temp`, so nothing else differs between the two products.

**The lag.** A forecast for lead-day `N` is issued at `baselines.issue_time` (09:00 UTC on the run's
own day, or the run's 00 UTC init time at lead-day 0), so the latest whole day it has seen is
`N + 1` days before the target day. The lag of a target hour is the same clock hour on that day,
`24 (N + 1)` hours earlier. `studies.baselines.same_clock_hour_window` reads every lag and window,
anchored at the issue day, strictly, with no fallback to an earlier day.

**Cleaning.** The lag source is NGED's hourly power with the multi-day zero runs, the meter spikes,
the commissioning ramp and the export-capped hours removed, so "part of a plant offline" is outside
what the study measures. The target is the published shared rows' `power_mw`.

**Row filter.** A row is dropped for every arm of its lead-day if the weather, the target, or a
single-lag arm's column is null (`REQUIRED_COLUMNS`). The statistics over several days keep their
nulls, which XGBoost routes natively. Each drop's count goes into `build_report.md`.

**Outputs**, under `<output-root>/<weather-product>/`: `lag_frame_<product>_day<N>.parquet` per
lead-day, `stage1_hours_<product>.parquet` (every hour the stage-1 models predict, lead-day 1 only),
`positive_control_<product>_s<percent>.parquet`, and `build_report_<product>.md`.

Run it with `uv run python studies/lag_features/build_lag_frame.py`.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from contracts.config_schemas import load_cv_config
from studies.baselines import (
    MIN_OBSERVED_CLEAR_SKY_SHARE,
    hourly_clear_sky,
    hourly_grid,
    issue_time,
    same_clock_hour_window,
)
from studies.commissioning import drop_commissioning_ramp
from studies.cross_validation import cut_eras
from studies.export_cap import with_export_cap
from studies.guards import refuse_to_overwrite
from studies.power import CV_CONFIG_PATH
from studies.pv_dataset import drop_outages_and_spikes, pv_sites, solar_hourly_power
from studies.sources import (
    NFC_DIR,
    NFC_LEADS_DAY10_DIR,
    NFC_LEADS_DAY10B_DIR,
    NFC_LEADS_DAY10D_DIR,
    study_dir_for,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("build_lag_frame")

WeatherProduct = Literal["ens_mean", "ifs_single"]

WEATHER_PRODUCTS: Final[tuple[WeatherProduct, ...]] = ("ens_mean", "ifs_single")
"""The weather products the study reads. The ENS mean is primary; IFS HRES is a replicate."""

LAG_FEATURES_DIR: Final[Path] = study_dir_for(study="lag_features")
"""Where the study's built frames, fits and report live, unless `--output-root` says otherwise."""

ROW_SET_START: Final[datetime] = datetime(2024, 12, 1, tzinfo=UTC)
"""The first hour of the shared rows: the first whole month after IFS Cycle 49r1."""

DROPPED_MONTHS: Final[tuple[str, ...]] = ("2026-01",)
"""The part-month that straddles the UKV upgrade, dropped from the shared rows."""

NWP_ERA_START_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2026-02")
"""The first month of each era after the first, as the matched-lead study cuts them."""

NWP_ERA_FOLD_OFFSETS: Final[dict[int, int]] = {0: 0, 1: 0, 2: 3}
"""How far each era's fold numbers are rotated, as the matched-lead study rotates them."""

PUBLISHED_ROW_COUNT: Final[int] = 35263
"""How many rows the matched-lead study's solar shared row set holds."""

ENS_DAYS: Final[tuple[int, ...]] = (0, 1, 2, 3)
"""The lead-days the original inputs file holds for the ENS mean, and for the baselines."""

PLANNED_PREFIXES: Final[tuple[str, ...]] = (
    "ens_mean_day0",
    "ens_mean_day1",
    "ukv_day1",
    "icon_eu_day1",
    "icon_eu_day2",
    "ifs025_day1",
    "ifs025_day2",
    "gefs_mean_day1",
)
"""The matched-lead study's planned arms, whose columns the shared rows must hold."""

TARGET: Final[str] = "power_mw"
"""The column every arm predicts."""

ENS_MEAN_LEAD_DAYS: Final[tuple[int, ...]] = (0, 1, 2, 3, 5, 7, 10, 14)
"""The lead-days run with the ENS mean: lead-day 1 is the full sweep, the rest are longer leads."""

IFS_LEAD_DAYS: Final[tuple[int, ...]] = (1, 2)
"""The lead-days run with IFS HRES, for B0 and L1 only."""

FULL_SWEEP_LEAD_DAY: Final[int] = 1
"""The day-ahead lead the live service delivers, at which every sweep arm is fitted."""

WEATHER_FILES: Final[dict[WeatherProduct, dict[int, Path]]] = {
    "ens_mean": {
        **dict.fromkeys(ENS_DAYS, NFC_DIR / "solar_forecast_inputs.parquet"),
        5: NFC_LEADS_DAY10_DIR / "solar_extra_lead_inputs.parquet",
        7: NFC_LEADS_DAY10B_DIR / "solar_extra_lead_inputs.parquet",
        10: NFC_LEADS_DAY10_DIR / "solar_extra_lead_inputs.parquet",
        14: NFC_LEADS_DAY10_DIR / "solar_extra_lead_inputs.parquet",
    },
    "ifs_single": dict.fromkeys(
        IFS_LEAD_DAYS, NFC_LEADS_DAY10D_DIR / "solar_extra_lead_inputs.parquet"
    ),
}
"""The file holding each product's weather columns at each lead-day, joined on `(site, time)`."""

B0_COLUMNS: Final[tuple[str, ...]] = (
    "hour_of_day",
    "day_of_year",
    "era_code",
    "solar_elevation_deg",
    "solar_azimuth_deg",
    "nwp_ghi",
    "nwp_temp",
)
"""The baseline arm B0: hour of day, day of year, era, the sun's elevation and azimuth, and the
product's global horizontal irradiance and 2 m air temperature at its own lead."""

STAGE1_TARGET_TEMPLATE: Final[str] = "stage1_target_fold{fold}"
STAGE1_LAG_TEMPLATE: Final[str] = "stage1_lag_fold{fold}"
RESIDUAL_LAG_TEMPLATE: Final[str] = "resid_lag_fold{fold}"
RESIDUAL_MEAN_TEMPLATES: Final[tuple[str, ...]] = (
    "resid_mean_1d_fold{fold}",
    "resid_mean_7d_fold{fold}",
    "resid_mean_30d_fold{fold}",
)
RATIO_TEMPLATE: Final[str] = "energy_ratio_7d_fold{fold}"
"""The columns `fit_lag_arms.py` derives from the stage-1 models, one per scored fold."""

EXTRA_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "B0": (),
    "L1": ("lag_d1",),
    "L7": tuple(f"lag_d{day}" for day in range(1, 8)),
    "L2": ("lag_d1", "lag_nwp_ghi", "lag_nwp_temp"),
    "W7": ("week_min", "week_max", "week_mean"),
    "Q30": ("q30_p90", "q30_median", "peak_slope_30d"),
    "DS": ("day_peak", "day_energy", "day_csi"),
    "S2": (
        STAGE1_TARGET_TEMPLATE,
        STAGE1_LAG_TEMPLATE,
        "lag_d1",
        RESIDUAL_LAG_TEMPLATE,
    ),
    "S3": RESIDUAL_MEAN_TEMPLATES,
    "FL": ("fleet_lag_mean",),
    "T1": ("days_since_start",),
    "KS": (
        "lag_d1",
        "week_min",
        "week_max",
        "week_mean",
        "q30_p90",
        "q30_median",
        "peak_slope_30d",
        *RESIDUAL_MEAN_TEMPLATES,
        "fleet_lag_mean",
    ),
    "N1": ("null_lag_8_to_28",),
    "N2": ("null_lag_random",),
}
"""Each fitted arm's columns beyond B0's seven. A `{fold}` is filled with the scored fold."""

SWEEP_ARMS: Final[tuple[str, ...]] = tuple(EXTRA_COLUMNS)
"""Every fitted arm of the lead-day 1 sweep."""

LONGER_LEAD_ARMS: Final[tuple[str, ...]] = ("B0", "L1", "W7", "Q30", "T1", "N2")
"""The arms fitted at every other lead-day."""

REPLICATE_ARMS: Final[tuple[str, ...]] = ("B0", "L1")
"""The arms fitted with IFS HRES, as an exploratory replicate."""

POWER_COLUMN_PREFIXES: Final[tuple[str, ...]] = (
    "power_mw",
    "cap_mw",
    "lag_d",
    "week_",
    "q30_",
    "peak_slope",
    "day_peak",
    "day_energy",
    "day_csi",
    "fleet_lag",
    "null_lag",
    "stage1_",
    "resid_",
)
"""Name prefixes of the columns in megawatts: the global model divides them
by capacity. `energy_ratio_7d` is a ratio, and is left alone."""

REQUIRED_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "weather": ("nwp_ghi", "nwp_temp"),
    "target": (TARGET,),
    "L1 lag": ("lag_d1",),
    "L2 and S2 lag-hour weather": ("lag_nwp_ghi", "lag_nwp_temp"),
    "N1 lag": ("null_lag_8_to_28",),
    "N2 lag": ("null_lag_random",),
}
"""The columns a row must hold to be kept, by requirement. A requirement applies only if the
frame builds its column."""

LAG_SOURCE_START_DAYS: Final[int] = 30
"""How many days before the first row the lag source is read from, for the 30-day windows."""

WEEK_DAYS: Final[int] = 7
WEEK_MIN_DAYS: Final[int] = 5
"""The weekly statistics' window, and the fewest of its days that must hold the hour."""

MONTH_DAYS: Final[int] = 30
MONTH_MIN_DAYS: Final[int] = 15
"""The slow trackers' window, and the fewest of its days that must hold the hour (or the peak)."""

NULL_LAG_FIRST_DAY: Final[int] = 8
NULL_LAG_LAST_DAY: Final[int] = 28
"""N1 reads one hour from this window of days before the issue day, never the L1 day."""

NULL_CONTROL_SEED: Final[int] = 1138
"""Seeds N1's and N2's random draws, so a rebuild reproduces them."""

DRIFT_START: Final[datetime] = datetime(2024, 3, 1, tzinfo=UTC)
"""T1 counts days from this date."""

POSITIVE_CONTROL_DATE: Final[datetime] = datetime(2025, 6, 1, tzinfo=UTC)
"""From this date, the positive control multiplies power by one minus the shift."""

POSITIVE_CONTROL_SHIFTS: Final[tuple[float, ...]] = (0.02, 0.05, 0.10)
"""The fractions of power lost after `POSITIVE_CONTROL_DATE`: 2%, 5% and 10%."""

LAG_TOLERANCE_MW: Final[float] = 1e-4
"""How far a strict lag may differ from `diurnal_persistence`'s Float32 value and still agree."""

FRAME_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "site",
    "time",
    "month",
    "fold",
    TARGET,
    "cap_mw",
    "constrained",
    "effective_capacity_mw",
    "clear_sky_w_m2",
)
"""The columns every frame carries besides the arms' features."""

MIN_SLOPE_DAYS: Final[int] = 15
"""The fewest days with a valid peak that the slope of the daily peak is fitted to."""


def shared_rows() -> pl.DataFrame:
    """Rebuild the matched-lead study's solar shared rows, with folds and `era_code`.

    Returns:
        The rows of `original/solar_forecast_inputs.parquet` from `ROW_SET_START`, without
        `DROPPED_MONTHS`, with no null in the target, the baselines' inputs or any planned arm's
        columns, carrying `month`, `era_code`, `era` and `fold`.

    Raises:
        ValueError: If the row count differs from `PUBLISHED_ROW_COUNT`.
    """
    frame = pl.read_parquet(NFC_DIR / "solar_forecast_inputs.parquet").filter(
        pl.col("time") >= ROW_SET_START
    )
    frame = frame.with_columns(month=pl.col("time").dt.strftime("%Y-%m")).filter(
        ~pl.col("month").is_in(DROPPED_MONTHS)
    )
    baselines = [
        f"{name}_day{day}" for day in ENS_DAYS for name in ("persistence", "diurnal_persistence")
    ]
    baselines += ["clear_sky_w_m2", *(f"clear_sky_index_day{day}" for day in ENS_DAYS)]
    required = [TARGET, *(c for c in baselines if c in frame.columns)]
    for prefix in PLANNED_PREFIXES:
        fields = [f"{prefix}_ghi", f"{prefix}_temp"]
        if all(c in frame.columns for c in fields):
            required += fields
    complete = frame.filter(pl.all_horizontal(pl.col(c).is_not_null() for c in required))
    rows = cut_eras(
        frame=complete, first_months=NWP_ERA_START_MONTHS, fold_offsets=NWP_ERA_FOLD_OFFSETS
    )
    if rows.height != PUBLISHED_ROW_COUNT:
        msg = f"shared rows hold {rows.height}, not the published {PUBLISHED_ROW_COUNT}"
        raise ValueError(msg)
    return rows.sort("site", "time")


def weather_table(*, product: WeatherProduct, lead_day: int) -> pl.DataFrame:
    """Return one product's irradiance and temperature at one lead-day, for every hour it covers.

    Args:
        product: The weather product.
        lead_day: The lead-day.

    Returns:
        One row per `(site, time)` with `nwp_ghi` and `nwp_temp`, null where the product has none.
    """
    ghi, temp = f"{product}_day{lead_day}_ghi", f"{product}_day{lead_day}_temp"
    return pl.read_parquet(
        WEATHER_FILES[product][lead_day], columns=["site", "time", ghi, temp]
    ).rename({ghi: "nwp_ghi", temp: "nwp_temp"})


def raw_hourly_power() -> pl.DataFrame:
    """Return NGED's hourly power as `diurnal_persistence` reads it, for the lag assertions.

    Returns:
        One row per `(site, time)` with `power_mw`.
    """
    return solar_hourly_power(sites=pv_sites()).select("site", "time", "power_mw")


def lag_source_hourly(*, shift: float = 0.0) -> pl.DataFrame:
    """Return the cleaned hourly power every lag and window reads.

    Removes the multi-day zero runs and meter spikes, the commissioning ramp, and the hours the
    export cap held down. Optionally multiplies power from `POSITIVE_CONTROL_DATE` by `1 - shift`,
    which is the positive control's level shift, applied before any lag is built.

    Args:
        shift: The fraction of power lost from `POSITIVE_CONTROL_DATE`; 0 for the real record.

    Returns:
        One row per `(site, time)` with `power_mw`.
    """
    sites = pv_sites()
    hourly = solar_hourly_power(sites=sites)
    cleaned = drop_outages_and_spikes(power=hourly, sites=sites).select("site", "time", "power_mw")
    ramped = drop_commissioning_ramp(dataset=cleaned)
    capped = with_export_cap(dataset=ramped).filter(~pl.col("constrained"))
    return capped.select(
        "site",
        "time",
        power_mw=pl.when(pl.col("time") >= POSITIVE_CONTROL_DATE)
        .then(pl.col("power_mw") * (1.0 - shift))
        .otherwise(pl.col("power_mw")),
    ).sort("site", "time")


def target_date(*, lead_day: int) -> pl.Expr:
    """Return the date of the whole day the forecast's latest whole observed day falls on.

    Args:
        lead_day: The forecast's lead-day.

    Returns:
        The date `lead_day + 1` days before the target hour's own date. A solar hour is labelled by
        its end, so its own date is that of its midpoint.
    """
    return (pl.col("time") - pl.duration(minutes=30)).dt.date() - pl.duration(days=lead_day + 1)


def daily_table(*, hourly: pl.DataFrame) -> pl.DataFrame:
    """Summarise each plant's days: the peak, the energy, the clear-sky index and the 30-day slope.

    A day is valid only if its observed hours hold at least `MIN_OBSERVED_CLEAR_SKY_SHARE` of its
    clear-sky energy, the rule `baselines.clear_sky_index` applies to its 24-hour window.

    Args:
        hourly: The lag source, with `site`, `time` and `power_mw`.

    Returns:
        One row per `(site, date)` with `day_peak`, `day_energy` (megawatt-hours over the observed
        hours), `day_csi` (megawatts per W m⁻² of clear-sky irradiance) and `peak_slope_30d`
        (megawatts per day), each null where the day is invalid.
    """
    sites = pv_sites()
    span = hourly.select(
        first=pl.col("time").min() - pl.duration(days=1),
        last=pl.col("time").max() + pl.duration(days=1),
    ).row(0, named=True)
    clear_sky = hourly_clear_sky(sites=sites, first=span["first"], last=span["last"])
    grid = hourly_grid(hourly=hourly).join(clear_sky, on=["site", "time"], how="inner")
    daily = (
        grid.with_columns(date=(pl.col("time") - pl.duration(minutes=30)).dt.date())
        .group_by("site", "date")
        .agg(
            peak=pl.col("power_mw").max(),
            energy=pl.col("power_mw").sum(),
            observed_clear_sky=pl.col("clear_sky_w_m2")
            .filter(pl.col("power_mw").is_not_null())
            .sum(),
            clear_sky=pl.col("clear_sky_w_m2").sum(),
        )
        .with_columns(
            valid=(pl.col("clear_sky") > 0)
            & (pl.col("observed_clear_sky") >= MIN_OBSERVED_CLEAR_SKY_SHARE * pl.col("clear_sky"))
        )
        .with_columns(
            day_peak=pl.when("valid").then(pl.col("peak")),
            day_energy=pl.when("valid").then(pl.col("energy")),
            day_csi=pl.when("valid").then(pl.col("energy") / pl.col("observed_clear_sky")),
        )
        .sort("site", "date")
    )
    return daily.select("site", "date", "day_peak", "day_energy", "day_csi").join(
        _peak_slopes(daily=daily), on=["site", "date"], how="left"
    )


def _peak_slopes(*, daily: pl.DataFrame) -> pl.DataFrame:
    """Return the least-squares slope of the daily peak over the 30 days ending at each date.

    Args:
        daily: `daily_table`'s intermediate frame, with `site`, `date` and `day_peak`.

    Returns:
        `site`, `date` and `peak_slope_30d`, null with fewer than `MIN_SLOPE_DAYS` valid peaks.
    """
    parts = []
    for site, rows in daily.group_by("site", maintain_order=True):
        grid = rows.select(
            date=pl.date_range(pl.col("date").min(), pl.col("date").max(), interval="1d")
        ).join(rows.select("date", "day_peak"), on="date", how="left")
        peaks = grid["day_peak"].to_numpy()
        slopes = np.full(len(peaks), np.nan)
        for end in range(len(peaks)):
            window = peaks[max(0, end - MONTH_DAYS + 1) : end + 1]
            index = np.flatnonzero(~np.isnan(window))
            if len(index) >= MIN_SLOPE_DAYS:
                slopes[end] = np.polyfit(index, window[index], deg=1)[0]
        parts.append(
            grid.select("date").with_columns(
                site=pl.lit(site[0]), peak_slope_30d=pl.Series(slopes).fill_nan(None)
            )
        )
    return pl.concat(parts)


def _random_hours(
    *, rows: pl.DataFrame, hourly: pl.DataFrame, lead_day: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw N1's and N2's random lags for every row.

    Args:
        rows: The rows, with `site` and `time`.
        hourly: The lag source.
        lead_day: The forecast's lead-day.
        rng: The generator both draws use.

    Returns:
        N1's lag and N2's lag per row, NaN where no draw exists. N1 is the first present hour, in
        a random order, among days `NULL_LAG_FIRST_DAY` to `NULL_LAG_LAST_DAY` before the issue
        day. N2 is the same clock hour on a uniformly random day of the whole record, never the
        row's own day.
    """
    days = range(NULL_LAG_FIRST_DAY, NULL_LAG_LAST_DAY + 1)
    matrix = np.column_stack(
        [
            same_clock_hour_window(
                keys=rows,
                hourly=hourly,
                day=lead_day,
                first_days_back=day,
                last_days_back=day,
                statistic="mean",
                min_count=1,
            )
            .fill_null(np.nan)
            .to_numpy()
            for day in days
        ]
    )
    order = rng.permuted(np.tile(np.arange(len(days)), (rows.height, 1)), axis=1)
    shuffled = np.take_along_axis(matrix, order, axis=1)
    first_present = (~np.isnan(shuffled)).argmax(axis=1)
    n1 = shuffled[np.arange(rows.height), first_present]

    n2 = np.full(rows.height, np.nan)
    keyed = hourly.with_columns(clock=pl.col("time").dt.time())
    pool = {
        (site, clock): group.sort("time")
        for (site, clock), group in keyed.group_by("site", "clock")
    }
    row_keys = rows.with_columns(clock=pl.col("time").dt.time()).select("site", "clock", "time")
    for (site, clock), group in row_keys.with_row_index("row").group_by("site", "clock"):
        source = pool.get((site, clock))
        if source is None:
            continue
        times, values = source["time"].to_numpy(), source["power_mw"].to_numpy()
        own = np.searchsorted(times, group["time"].to_numpy())
        own_present = (own < len(times)) & (
            times[np.minimum(own, len(times) - 1)] == group["time"].to_numpy()
        )
        draw = rng.integers(0, np.maximum(len(times) - own_present.astype(int), 1))
        draw = draw + (own_present & (draw >= own))
        n2[group["row"].to_numpy()] = values[np.minimum(draw, len(times) - 1)]
    return n1, n2


def _fleet_lag_mean(*, rows: pl.DataFrame, hourly: pl.DataFrame, lead_day: int) -> pl.Series:
    """Return the mean of the other plants' lag at the same hour, over whichever are present.

    Args:
        rows: The rows, with `site` and `time`.
        hourly: The lag source.
        lead_day: The forecast's lead-day.

    Returns:
        One value per row, null where no other plant has the lag.
    """
    sites = hourly["site"].unique().sort()
    everywhere = pl.DataFrame({"site": sites}).join(rows.select("time").unique(), how="cross")
    lags = everywhere.with_columns(
        lag=same_clock_hour_window(
            keys=everywhere,
            hourly=hourly,
            day=lead_day,
            first_days_back=1,
            last_days_back=1,
            statistic="mean",
            min_count=1,
        )
    )
    totals = lags.group_by("time").agg(
        total=pl.col("lag").sum(), count=pl.col("lag").is_not_null().sum()
    )
    own = rows.select("site", "time").join(
        lags, on=["site", "time"], how="left", maintain_order="left"
    )
    joined = own.join(totals, on="time", how="left", maintain_order="left")
    others = pl.col("count") - pl.col("lag").is_not_null().cast(pl.UInt32)
    return joined.select(
        fleet_lag_mean=pl.when(others > 0).then(
            (pl.col("total") - pl.col("lag").fill_null(0.0)) / others
        )
    )["fleet_lag_mean"]


def lag_columns(
    *,
    rows: pl.DataFrame,
    hourly: pl.DataFrame,
    daily: pl.DataFrame,
    weather: pl.DataFrame,
    lead_day: int,
    arms: Sequence[str],
) -> pl.DataFrame:
    """Add every column the named arms need beyond B0's.

    Args:
        rows: The rows, with `site`, `time` and the B0 columns.
        hourly: The lag source.
        daily: `daily_table`'s result.
        weather: `weather_table`'s result at the lead-day, for L2's lag-hour weather.
        lead_day: The forecast's lead-day.
        arms: The arms to build columns for.

    Returns:
        `rows` with the columns of `EXTRA_COLUMNS` that do not hold a `{fold}`, for those arms.
    """
    needed = {c for arm in arms for c in EXTRA_COLUMNS[arm] if "{fold}" not in c}
    window = {"hourly": hourly, "day": lead_day}
    columns: dict[str, pl.Series] = {}
    for day in range(1, 8):
        if f"lag_d{day}" in needed:
            columns[f"lag_d{day}"] = same_clock_hour_window(
                keys=rows,
                first_days_back=day,
                last_days_back=day,
                statistic="mean",
                min_count=1,
                **window,
            )
    for name in ("min", "max", "mean"):
        if f"week_{name}" in needed:
            columns[f"week_{name}"] = same_clock_hour_window(
                keys=rows,
                first_days_back=1,
                last_days_back=WEEK_DAYS,
                statistic=name,
                min_count=WEEK_MIN_DAYS,
                **window,
            )
    for name in ("p90", "median"):
        if f"q30_{name}" in needed:
            columns[f"q30_{name}"] = same_clock_hour_window(
                keys=rows,
                first_days_back=1,
                last_days_back=MONTH_DAYS,
                statistic=name,
                min_count=MONTH_MIN_DAYS,
                **window,
            )
    if "fleet_lag_mean" in needed:
        columns["fleet_lag_mean"] = _fleet_lag_mean(rows=rows, hourly=hourly, lead_day=lead_day)
    if "days_since_start" in needed:
        columns["days_since_start"] = rows.select(
            (pl.col("time") - pl.lit(DRIFT_START)).dt.total_days().cast(pl.Int32)
        ).to_series()
    out = rows.with_columns(**columns)
    if "lag_nwp_ghi" in needed:
        lag_weather = weather.select(
            "site",
            lag_time=pl.col("time"),
            lag_nwp_ghi=pl.col("nwp_ghi"),
            lag_nwp_temp=pl.col("nwp_temp"),
        )
        out = (
            out.with_columns(lag_time=pl.col("time") - pl.duration(days=lead_day + 1))
            .join(lag_weather, on=["site", "lag_time"], how="left", maintain_order="left")
            .drop("lag_time")
        )
    daily_needed = [
        c for c in ("day_peak", "day_energy", "day_csi", "peak_slope_30d") if c in needed
    ]
    if daily_needed:
        out = (
            out.with_columns(asof_date=target_date(lead_day=lead_day))
            .join(
                daily.select("site", pl.col("date").alias("asof_date"), *daily_needed),
                on=["site", "asof_date"],
                how="left",
                maintain_order="left",
            )
            .drop("asof_date")
        )
    if "null_lag_random" in needed or "null_lag_8_to_28" in needed:
        n1, n2 = _random_hours(
            rows=rows,
            hourly=hourly,
            lead_day=lead_day,
            rng=np.random.default_rng(NULL_CONTROL_SEED),
        )
        drawn = {
            "null_lag_8_to_28": pl.Series(n1).fill_nan(None),
            "null_lag_random": pl.Series(n2).fill_nan(None),
        }
        out = out.with_columns(**{name: drawn[name] for name in drawn if name in needed})
    return out


def assert_lag_arithmetic(*, frame: pl.DataFrame, lead_day: int, raw: pl.DataFrame) -> list[str]:
    """Check the lag's timing and its agreement with `diurnal_persistence`.

    Args:
        frame: A lead-day's frame with `lag_d1` and `diurnal_persistence_day<lead_day>` (when the
            original inputs hold that lead-day).
        lead_day: The lead-day.
        raw: `raw_hourly_power()`, to recompute the lag independently.

    Returns:
        Report lines giving each assertion's result.

    Raises:
        ValueError: If a lag is read from after its issue time, or differs from the power at the
            lagged hour, or from `diurnal_persistence` where both exist.
    """
    lines = []
    timed = frame.select(
        lag_time=pl.col("time") - pl.duration(days=lead_day + 1),
        issued=issue_time(
            day_start=(pl.col("time") - pl.duration(minutes=30)).dt.truncate("1d"), day=lead_day
        ),
    )
    late = timed.filter(pl.col("lag_time") > pl.col("issued")).height
    if late:
        msg = f"lead-day {lead_day}: {late} lags are read from after the issue time"
        raise ValueError(msg)
    lines.append(
        f"- lead-day {lead_day}: every lag time is at or before its issue time "
        f"({frame.height} rows)."
    )
    independent = frame.select("site", "time", "lag_d1").join(
        raw.select("site", lag_time=pl.col("time"), raw=pl.col("power_mw")).with_columns(
            time=pl.col("lag_time") + pl.duration(days=lead_day + 1)
        ),
        on=["site", "time"],
        how="left",
    )
    wrong = independent.filter(
        pl.col("lag_d1").is_not_null()
        & ((pl.col("lag_d1") - pl.col("raw")).abs() > LAG_TOLERANCE_MW)
    ).height
    if wrong:
        msg = f"lead-day {lead_day}: {wrong} lags differ from the power at the lagged hour"
        raise ValueError(msg)
    lines.append(
        f"- lead-day {lead_day}: every lag equals the power at the hour "
        f"24 x {lead_day + 1} hours earlier."
    )
    reference = f"diurnal_persistence_day{lead_day}"
    if reference in frame.columns:
        both = frame.filter(pl.col("lag_d1").is_not_null() & pl.col(reference).is_not_null())
        different = both.filter(
            (pl.col("lag_d1") - pl.col(reference)).abs() > LAG_TOLERANCE_MW
        ).height
        if different:
            msg = f"lead-day {lead_day}: {different} lags differ from {reference}"
            raise ValueError(msg)
        lines.append(
            f"- lead-day {lead_day}: the strict lag equals `{reference}` on all {both.height} rows "
            f"where both exist ({frame.height - both.height} rows lack one of them)."
        )
    return lines


def arms_columns_table(*, arms: Sequence[str]) -> list[str]:
    """Return a markdown table of each arm's full column list.

    Args:
        arms: The arms to list.

    Returns:
        Table lines, one row per arm.
    """
    lines = ["| Arm | Columns | Count |", "|---|---|---|"]
    for arm in arms:
        columns = (*B0_COLUMNS, *EXTRA_COLUMNS[arm])
        lines.append(f"| {arm} | {', '.join(f'`{c}`' for c in columns)} | {len(columns)} |")
    return lines


def build_frame(
    *,
    product: WeatherProduct,
    lead_day: int,
    arms: Sequence[str],
    shared: pl.DataFrame,
    hourly: pl.DataFrame,
    daily: pl.DataFrame,
    shift: float = 0.0,
) -> tuple[pl.DataFrame, list[str]]:
    """Build one lead-day's frame: the shared rows with the arms' columns and the row filter.

    Args:
        product: The weather product.
        lead_day: The lead-day.
        arms: The arms the frame serves.
        shared: `shared_rows()`'s result.
        hourly: The lag source, already shifted for the positive control if `shift` is not 0.
        daily: `daily_table`'s result for `hourly`.
        shift: The positive control's fraction of power lost, scaling the target to match `hourly`.

    Returns:
        The frame, and report lines giving the row counts and each requirement's drop count.
    """
    weather = weather_table(product=product, lead_day=lead_day)
    cutoff = datetime.combine(load_cv_config(CV_CONFIG_PATH).final_test_start, time.min, tzinfo=UTC)
    rows = shared.join(weather, on=["site", "time"], how="left").sort("site", "time")
    after_cutoff = rows.filter(pl.col("time") >= cutoff).height
    rows = rows.filter(pl.col("time") < cutoff)
    if shift:
        shifted = (
            pl.when(pl.col("time") >= POSITIVE_CONTROL_DATE)
            .then(pl.col(TARGET) * (1.0 - shift))
            .otherwise(pl.col(TARGET))
        )
        rows = rows.with_columns(**{TARGET: shifted.cast(pl.Float64)})
    built = lag_columns(
        rows=rows, hourly=hourly, daily=daily, weather=weather, lead_day=lead_day, arms=arms
    )
    present = [
        (name, [c for c in columns if c in built.columns])
        for name, columns in REQUIRED_COLUMNS.items()
    ]
    present = [(name, columns) for name, columns in present if columns]
    lines = [
        f"- shared rows: {built.height + after_cutoff}",
        (
            f"- rows at or after `final_test_start` ({cutoff:%Y-%m-%d}), dropped because "
            f"`studies.power.scan_power` cannot read their lags: {after_cutoff}"
        ),
    ]
    for name, columns in present:
        count = built.filter(pl.any_horizontal(pl.col(c).is_null() for c in columns)).height
        lines.append(f"- rows lacking {name} (`{', '.join(columns)}`): {count}")
    keep = [c for _, columns in present for c in columns]
    kept = built.filter(pl.all_horizontal(pl.col(c).is_not_null() for c in keep))
    lines.append(f"- rows kept: {kept.height}, dropped in all: {built.height - kept.height}")
    optional = [
        c for arm in arms for c in EXTRA_COLUMNS[arm] if c in kept.columns and c not in keep
    ]
    lines.extend(
        f"- null share of `{column}` among kept rows: {kept[column].is_null().mean():.3f}"
        for column in dict.fromkeys(optional)
    )
    arm_columns = [c for arm in arms for c in EXTRA_COLUMNS[arm] if "{fold}" not in c]
    references = [
        c
        for c in kept.columns
        if c in (f"persistence_day{lead_day}", f"diurnal_persistence_day{lead_day}")
    ]
    columns = [*FRAME_KEY_COLUMNS, *B0_COLUMNS, *arm_columns, *references]
    return kept.select(list(dict.fromkeys(columns))), lines


def stage1_hours(
    *, product: WeatherProduct, shared: pl.DataFrame, hourly: pl.DataFrame
) -> pl.DataFrame:
    """Return every hour the stage-1 models predict, with the observed power.

    Args:
        product: The weather product.
        shared: `shared_rows()`'s result.
        hourly: The lag source.

    Returns:
        One row per hour of `original/solar_forecast_inputs.parquet` with the B0 columns non-null:
        `fold` (null outside the shared rows), `constrained` and the target `power_mw` (null
        outside the shared rows), and `observed_mw` from the lag source.
    """
    calendar = pl.read_parquet(
        NFC_DIR / "solar_forecast_inputs.parquet",
        columns=[
            "site",
            "time",
            "hour_of_day",
            "day_of_year",
            "solar_elevation_deg",
            "solar_azimuth_deg",
        ],
    )
    month = pl.col("time").dt.strftime("%Y-%m")
    era = pl.lit(0, dtype=pl.Int8) + sum(
        (month >= start).cast(pl.Int8) for start in NWP_ERA_START_MONTHS
    )
    hours = (
        calendar.join(
            weather_table(product=product, lead_day=FULL_SWEEP_LEAD_DAY), on=["site", "time"]
        )
        .with_columns(era_code=era)
        .join(
            shared.select("site", "time", "fold", "constrained", TARGET, shared_era="era_code"),
            on=["site", "time"],
            how="left",
        )
        .join(hourly.rename({"power_mw": "observed_mw"}), on=["site", "time"], how="left")
        .drop_nulls(subset=list(B0_COLUMNS))
        .sort("site", "time")
    )
    disagree = hours.filter(
        pl.col("shared_era").is_not_null() & (pl.col("shared_era") != pl.col("era_code"))
    ).height
    if disagree:
        msg = f"era_code from the month disagrees with the shared rows on {disagree} hours"
        raise ValueError(msg)
    return hours.drop("shared_era")


def output_paths(
    *, root: Path, product: WeatherProduct, lead_days: Sequence[int]
) -> dict[str, Path]:
    """Return every file a build writes, by a short name.

    Args:
        root: The output root.
        product: The weather product.
        lead_days: The lead-days built.

    Returns:
        Each output path, all under `root / product` with the product in the file name.
    """
    directory = root / product
    paths = {f"day{day}": directory / f"lag_frame_{product}_day{day}.parquet" for day in lead_days}
    paths["report"] = directory / f"build_report_{product}.md"
    if product == "ens_mean":
        paths["stage1"] = directory / f"stage1_hours_{product}.parquet"
        for shift in POSITIVE_CONTROL_SHIFTS:
            paths[f"control{round(shift * 100):02d}"] = (
                directory / f"positive_control_{product}_s{round(shift * 100):02d}.parquet"
            )
    return paths


def main() -> int:
    """Build the frames and write them with their report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-product", choices=WEATHER_PRODUCTS, default="ens_mean")
    parser.add_argument("--output-root", type=Path, default=LAG_FEATURES_DIR)
    parser.add_argument(
        "--lead-days",
        type=int,
        nargs="+",
        help="Build only these lead-days (default: every lead-day of the product).",
    )
    arguments = parser.parse_args()
    product: WeatherProduct = arguments.weather_product
    default_days = ENS_MEAN_LEAD_DAYS if product == "ens_mean" else IFS_LEAD_DAYS
    lead_days = tuple(arguments.lead_days or default_days)
    paths = output_paths(root=arguments.output_root, product=product, lead_days=lead_days)
    refuse_to_overwrite(paths=paths.values())

    shared = shared_rows()
    hourly = lag_source_hourly()
    daily = daily_table(hourly=hourly)
    raw = raw_hourly_power()
    report = [f"## Frames built for the weather product `{product}`", ""]
    report += [f"Shared rows: {shared.height} (the published count {PUBLISHED_ROW_COUNT}).", ""]

    for lead_day in lead_days:
        if product != "ens_mean":
            arms = REPLICATE_ARMS
        else:
            arms = SWEEP_ARMS if lead_day == FULL_SWEEP_LEAD_DAY else LONGER_LEAD_ARMS
        frame, lines = build_frame(
            product=product, lead_day=lead_day, arms=arms, shared=shared, hourly=hourly, daily=daily
        )
        assertion_lines = assert_lag_arithmetic(frame=frame, lead_day=lead_day, raw=raw)
        paths[f"day{lead_day}"].parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(paths[f"day{lead_day}"])
        report += [
            f"### Lead-day {lead_day}",
            "",
            f"Arms: {', '.join(arms)}.",
            "",
            *lines,
            "",
            *assertion_lines,
            "",
        ]
        report += [*arms_columns_table(arms=arms), ""]
        _LOG.info("lead-day %d: %d rows", lead_day, frame.height)

        if product == "ens_mean" and lead_day == FULL_SWEEP_LEAD_DAY:
            stage1_hours(product=product, shared=shared, hourly=hourly).write_parquet(
                paths["stage1"]
            )
            for shift in POSITIVE_CONTROL_SHIFTS:
                shifted_hourly = lag_source_hourly(shift=shift)
                control, control_lines = build_frame(
                    product=product,
                    lead_day=lead_day,
                    arms=("B0", "L1"),
                    shared=shared,
                    hourly=shifted_hourly,
                    daily=daily,
                    shift=shift,
                )
                control.write_parquet(paths[f"control{round(shift * 100):02d}"])
                report += [
                    (
                        f"### Positive control, {shift:.0%} of power lost from "
                        f"{POSITIVE_CONTROL_DATE:%Y-%m-%d}"
                    ),
                    "",
                    *control_lines,
                    "",
                ]
    paths["report"].write_text("\n".join(report) + "\n")
    sys.stdout.write("\n".join(report) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
