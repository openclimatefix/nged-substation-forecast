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
from lag_arm_columns import (
    DAILY_FEATURE_COLUMNS,
    SATELLITE_LAG_DAYS,
    LagInputs,
    analogue_ensemble,
    clear_sky_table,
    clipping_share,
    daily_features,
    issue_morning,
    lag_context,
    lookup_daily,
    satellite_ratios,
    transfer_function,
    window_anchor_lines,
)
from studies.baselines import issue_time, same_clock_hour_window
from studies.commissioning import drop_commissioning_ramp
from studies.cross_validation import cut_eras
from studies.export_cap import with_export_cap
from studies.guards import check_no_missing, refuse_to_overwrite
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

CALENDAR_AND_SUN_COLUMNS: Final[tuple[str, ...]] = B0_COLUMNS[:5]
"""B0's columns that come with the shared rows rather than from the weather product."""

STAGE1_TARGET_TEMPLATE: Final[str] = "stage1_target_fold{fold}"
STAGE1_LAG_TEMPLATE: Final[str] = "stage1_lag_fold{fold}"
RESIDUAL_LAG_TEMPLATE: Final[str] = "resid_lag_fold{fold}"
RESIDUAL_MEAN_TEMPLATES: Final[tuple[str, ...]] = (
    "resid_mean_1d_fold{fold}",
    "resid_mean_7d_fold{fold}",
    "resid_mean_30d_fold{fold}",
)
RATIO_TEMPLATE: Final[str] = "energy_ratio_7d_fold{fold}"
CLOCK_RATIO_TEMPLATE: Final[str] = "energy_ratio_clock30_fold{fold}"
"""The columns `fit_lag_arms.py` derives from the stage-1 models, one per scored fold. R1s and R2
read the ratios; neither is a feature of a fitted arm."""

IM_COLUMNS: Final[tuple[str, ...]] = ("im_energy", "im_ratio", "im_hours")
CK_COLUMNS: Final[tuple[str, ...]] = ("ck_expanding_p995", "ck_max_60d", "ck_clear_near_share")
TF_COLUMNS: Final[tuple[str, ...]] = ("tf_ratio", "tf_scaled_forecast")
DT_COLUMNS: Final[tuple[str, ...]] = ("dt_centroid_shift_30d", "dt_shoulder_share_30d")
W7_COLUMNS: Final[tuple[str, ...]] = ("week_min", "week_max", "week_mean")
Q30_COLUMNS: Final[tuple[str, ...]] = ("q30_p90", "q30_median")
RP_COLUMNS: Final[tuple[str, ...]] = ("rp_7d", "rp_1d")
"""Column groups several arms share."""

N2_DRAWS: Final[int] = 16
"""How many random lags N2's family draws per row: N2 reads the first, and N2-k the first `k`.
Sixteen
is the most columns any shortlist candidate adds (KS adds 16)."""

N2_WIDE_COLUMNS: Final[tuple[str, ...]] = (
    "null_lag_random",
    *(f"null_lag_random_{draw}" for draw in range(2, N2_DRAWS + 1)),
)
"""Every random-lag column the frame holds. N2-k reads the first `k`."""

EXTRA_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "B0": (),
    "L1": ("lag_d1",),
    "IM": IM_COLUMNS,
    "L2": ("lag_d1", "lag_nwp_ghi", "lag_nwp_temp"),
    "CTX7": (
        *(f"lag_d{day}" for day in range(1, 8)),
        *(f"lag_ghi_d{day}" for day in range(1, 8)),
    ),
    "W7": W7_COLUMNS,
    "Q30": Q30_COLUMNS,
    "CK": CK_COLUMNS,
    "TF": TF_COLUMNS,
    "AN": ("an_mean", "an_csi", "an_spread"),
    "PC": ("pc_power_to_cams_7d", "pc_cams_to_forecast_30d"),
    "S2": (
        STAGE1_TARGET_TEMPLATE,
        STAGE1_LAG_TEMPLATE,
        "lag_d1",
        RESIDUAL_LAG_TEMPLATE,
    ),
    "S3": RESIDUAL_MEAN_TEMPLATES,
    "RP": RP_COLUMNS,
    "T1": ("days_since_start",),
    "KS": (
        "lag_d1",
        *IM_COLUMNS,
        *W7_COLUMNS,
        *Q30_COLUMNS,
        *TF_COLUMNS,
        *RESIDUAL_MEAN_TEMPLATES,
        *RP_COLUMNS,
    ),
    "N1": ("null_lag_8_to_28",),
    "N2": ("null_lag_random",),
    "N2-wide": N2_WIDE_COLUMNS,
    "G-ID": ("plant_code",),
    "G-FP": (*TF_COLUMNS, *CK_COLUMNS, *DT_COLUMNS),
}
"""Each fitted arm's columns beyond B0's seven. A `{fold}` is filled with the scored fold. The two
arms named `G-` are fitted only as one model across the plants."""

FRAME_ONLY_ARMS: Final[tuple[str, ...]] = ("N2-wide",)
"""Arms that only name columns the frame must hold; N2-k is fitted from them when X is wide."""

GLOBAL_ONLY_ARMS: Final[tuple[str, ...]] = ("G-ID", "G-FP")
"""Arms fitted only in the global scope; they stay out of the per-plant shortlist rule."""

SWEEP_ARMS: Final[tuple[str, ...]] = tuple(
    a for a in EXTRA_COLUMNS if a not in GLOBAL_ONLY_ARMS and a not in FRAME_ONLY_ARMS
)
"""Every fitted arm of the lead-day 1 per-plant sweep."""

LONGER_LEAD_ARMS: Final[tuple[str, ...]] = ("B0", "L1", "W7", "Q30", "T1", "N2")
"""The arms fitted at every other lead-day."""

REPLICATE_ARMS: Final[tuple[str, ...]] = ("B0", "L1")
"""The arms fitted with IFS HRES, as an exploratory replicate."""

ARM_LABELS: Final[dict[str, str]] = {"T1": "T1 (interpolation bound)"}
"""How a table or chart names an arm whose bare name would mislead. T1 gives trees the date, and
month-block folds interleave, so its gain is what interpolating between months buys, never drift
a live forecast could use."""

POWER_COLUMN_PREFIXES: Final[tuple[str, ...]] = (
    "power_mw",
    "cap_mw",
    "lag_d",
    "week_",
    "q30_",
    "im_energy",
    "im_ratio",
    "ck_expanding",
    "ck_max",
    "tf_ratio",
    "tf_scaled",
    "an_mean",
    "an_spread",
    "pc_power",
    "null_lag",
    "stage1_",
    "resid_",
)
"""Name prefixes of the columns in megawatts (or megawatts per unit of irradiance): the global model
divides them by capacity. The ratios, shares, counts and irradiances are left alone."""

WINDOW_FIRST_DAY: Final[dict[str, int]] = {
    **{f"lag_d{day}": day for day in range(1, 8)},
    **{f"lag_ghi_d{day}": day for day in range(1, 8)},
    "lag_nwp_ghi": 1,
    **dict.fromkeys(W7_COLUMNS, 1),
    **dict.fromkeys(Q30_COLUMNS, 1),
    **dict.fromkeys(TF_COLUMNS, 1),
    "an_mean": 1,
    "pc_power_to_cams_7d": SATELLITE_LAG_DAYS,
    "pc_cams_to_forecast_30d": SATELLITE_LAG_DAYS,
    "null_lag_8_to_28": 8,
}
"""The nearest day (counted from the latest whole day) of each same-clock-hour window. The anchor
assertions check each one. N2's random day is deliberately unanchored, because a true null reads
any day of the record."""

REQUIRED_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "weather": ("nwp_ghi", "nwp_temp"),
    "target": (TARGET,),
    "L1 strict lag": ("lag_d1",),
}
"""The row set: the shared rows minus those where B0's columns or L1's strict lag are null. Every
other arm's columns may be null, which XGBoost routes natively, and no row is dropped for them. B0's
calendar and sun columns are non-null on every shared row, which `build_frame` checks."""

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
        N1's lag per row, and N2's draws (`_random_other_fold_lags`), NaN where none exists. N1 is
        the first present hour, in a random order, among days `NULL_LAG_FIRST_DAY` to
        `NULL_LAG_LAST_DAY` before the issue day.
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

    return n1, _random_other_fold_lags(rows=rows, hourly=hourly, rng=rng)


def _random_other_fold_lags(
    *, rows: pl.DataFrame, hourly: pl.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Draw N2's random same-clock-hour lags from months outside each row's own fold.

    A candidate hour is the same clock hour at the same plant on any day of the record whose month
    does not lie in the row's fold (a month in no fold is allowed), so a draw can never be the
    row's own day or a neighbour in its own test months.

    Args:
        rows: Rows with `site`, `time`, `month` and `fold`.
        hourly: The lag source.
        rng: The random generator.

    Returns:
        An array of shape (rows, `N2_DRAWS`) of independent draws, NaN where no candidate exists.
    """
    month_folds = rows.select("site", "month", "fold").unique(subset=["site", "month"])
    pool = (
        hourly.with_columns(
            month=pl.col("time").dt.strftime("%Y-%m"), clock=pl.col("time").dt.time()
        )
        .join(month_folds, on=["site", "month"], how="left")
        .with_columns(fold=pl.col("fold").fill_null(-1))
        .sort("site", "time")
    )
    sources = {
        (site, clock): (group["power_mw"].to_numpy(), group["fold"].to_numpy())
        for (site, clock), group in pool.group_by("site", "clock")
    }
    draws = np.full((rows.height, N2_DRAWS), np.nan)
    keyed = rows.select("site", "fold", clock=pl.col("time").dt.time()).with_row_index("row")
    for (site, clock, fold), group in keyed.group_by("site", "clock", "fold"):
        source = sources.get((site, clock))
        if source is None:
            continue
        values, folds = source
        candidates = np.flatnonzero(folds != fold)
        if len(candidates) == 0:
            continue
        chosen = rng.integers(0, len(candidates), size=(group.height, N2_DRAWS))
        draws[group["row"].to_numpy()] = values[candidates[chosen]]
    return draws


def _same_hour_columns(
    *, rows: pl.DataFrame, hourly: pl.DataFrame, lead_day: int, needed: set[str]
) -> dict[str, pl.Series]:
    """Build the lags, weekly statistics and 30-day percentiles that some arm needs.

    Args:
        rows: Rows with `site` and `time`.
        hourly: The lag source.
        lead_day: The forecast's lead-day.
        needed: Every column the arms need.

    Returns:
        The `lag_d<k>`, `week_*` and `q30_*` columns in `needed`.
    """
    columns: dict[str, pl.Series] = {}
    specs = [
        *((f"lag_d{day}", "mean", day, day, 1) for day in range(1, 8)),
        *((f"week_{name}", name, 1, WEEK_DAYS, WEEK_MIN_DAYS) for name in ("min", "max", "mean")),
        *((f"q30_{name}", name, 1, MONTH_DAYS, MONTH_MIN_DAYS) for name in ("p90", "median")),
    ]
    for column, statistic, first, last, minimum in specs:
        if column in needed:
            columns[column] = same_clock_hour_window(
                keys=rows,
                hourly=hourly,
                day=lead_day,
                first_days_back=first,
                last_days_back=last,
                statistic=statistic,  # ty: ignore[invalid-argument-type]
                min_count=minimum,
            )
    return columns


def lag_columns(
    *,
    rows: pl.DataFrame,
    inputs: LagInputs,
    weather: pl.DataFrame,
    weather_lead0: pl.DataFrame | None,
    lead_day: int,
    arms: Sequence[str],
) -> tuple[pl.DataFrame, pl.Series | None]:
    """Add every column the named arms need beyond B0's.

    Args:
        rows: The rows, with `site`, `time` and the B0 columns.
        inputs: The lead-independent inputs: the lag source, clear-sky irradiance, CAMS and the
            daily features.
        weather: `weather_table`'s result at the lead-day, for the lag-hour weather.
        weather_lead0: `weather_table`'s result at lead-day 0, for IM's morning irradiance.
        lead_day: The forecast's lead-day.
        arms: The arms to build columns for.

    Returns:
        `rows` with the columns of `EXTRA_COLUMNS` that do not hold a `{fold}`, for those arms, and
        each row's latest issue-morning hour read, or `None` if no arm needs IM.
    """
    needed = {c for arm in arms for c in EXTRA_COLUMNS[arm] if "{fold}" not in c}
    hourly = inputs.hourly
    columns: dict[str, pl.Series] = {}
    if "lag_ghi_d1" in needed:
        columns |= lag_context(
            rows=rows, hourly=hourly, weather=weather, lead_day=lead_day
        ).to_dict()
    columns |= {
        name: series
        for name, series in _same_hour_columns(
            rows=rows, hourly=hourly, lead_day=lead_day, needed=needed
        ).items()
        if name not in columns
    }
    morning_latest = None
    if "im_energy" in needed:
        if weather_lead0 is None:
            msg = "IM needs the lead-day 0 weather"
            raise ValueError(msg)
        morning, morning_latest = issue_morning(
            rows=rows, hourly=hourly, weather_lead0=weather_lead0, lead_day=lead_day
        )
        columns |= morning.to_dict()
    if "tf_ratio" in needed:
        columns |= transfer_function(
            rows=rows, hourly=hourly, weather=weather, lead_day=lead_day
        ).to_dict()
    if "an_mean" in needed:
        columns |= analogue_ensemble(
            rows=rows, hourly=hourly, weather=weather, clear_sky=inputs.clear_sky, lead_day=lead_day
        ).to_dict()
    if "pc_power_to_cams_7d" in needed:
        columns |= satellite_ratios(
            rows=rows, hourly=hourly, weather=weather, cams=inputs.cams, lead_day=lead_day
        ).to_dict()
    daily_needed = [c for c in DAILY_FEATURE_COLUMNS if c in needed]
    if daily_needed:
        columns |= lookup_daily(
            rows=rows, daily=inputs.daily, lead_day=lead_day, columns=daily_needed
        ).to_dict()
    if "ck_clear_near_share" in needed:
        columns["ck_clear_near_share"] = clipping_share(
            rows=rows,
            hourly=hourly,
            weather=weather,
            clear_sky=inputs.clear_sky,
            daily=inputs.daily,
            lead_day=lead_day,
        )
    if "days_since_start" in needed:
        columns["days_since_start"] = rows.select(
            (pl.col("time") - pl.lit(DRIFT_START)).dt.total_days().cast(pl.Int32)
        ).to_series()
    if "plant_code" in needed:
        columns["plant_code"] = rows.select(
            pl.col("site").str.to_integer(base=36, strict=False).cast(pl.Int32)
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
    if "null_lag_random" in needed or "null_lag_8_to_28" in needed:
        n1, n2 = _random_hours(
            rows=rows,
            hourly=hourly,
            lead_day=lead_day,
            rng=np.random.default_rng(NULL_CONTROL_SEED),
        )
        drawn = {"null_lag_8_to_28": pl.Series(n1).fill_nan(None)} | {
            name: pl.Series(n2[:, index]).fill_nan(None)
            for index, name in enumerate(N2_WIDE_COLUMNS)
        }
        out = out.with_columns(**{name: drawn[name] for name in drawn if name in needed})
    return out, morning_latest


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


def extra_columns_of(*, arm: str) -> tuple[str, ...]:
    """Return an arm's columns beyond B0's, resolving the dynamic arm `N2-<k>`.

    Args:
        arm: A key of `EXTRA_COLUMNS`, or `N2-<k>` for `k` random lags.

    Returns:
        The columns; for `N2-<k>`, the first `k` of `N2_WIDE_COLUMNS`.
    """
    if arm.startswith("N2-") and arm != "N2-wide":
        return N2_WIDE_COLUMNS[: int(arm.removeprefix("N2-"))]
    return EXTRA_COLUMNS[arm]


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
    inputs: LagInputs,
    weather_lead0: pl.DataFrame | None,
    shift: float = 0.0,
) -> tuple[pl.DataFrame, list[str]]:
    """Build one lead-day's frame: the shared rows with the arms' columns and the row filter.

    Args:
        product: The weather product.
        lead_day: The lead-day.
        arms: The arms the frame serves.
        shared: `shared_rows()`'s result.
        inputs: The lag source (already shifted for the positive control if `shift` is not 0) and
            the inputs built from it.
        weather_lead0: The ENS mean's lead-day 0 weather, for IM, or `None`.
        shift: The positive control's fraction of power lost, scaling the target to match `hourly`.

    Returns:
        The frame, and report lines giving the row counts and each requirement's drop count.
    """
    weather = weather_table(product=product, lead_day=lead_day)
    cutoff = datetime.combine(load_cv_config(CV_CONFIG_PATH).final_test_start, time.min, tzinfo=UTC)
    rows = shared.join(weather, on=["site", "time"], how="left").sort("site", "time")
    check_no_missing(frame=rows, columns=CALENDAR_AND_SUN_COLUMNS)
    after_cutoff = rows.filter(pl.col("time") >= cutoff).height
    rows = rows.filter(pl.col("time") < cutoff)
    if shift:
        shifted = (
            pl.when(pl.col("time") >= POSITIVE_CONTROL_DATE)
            .then(pl.col(TARGET) * (1.0 - shift))
            .otherwise(pl.col(TARGET))
        )
        rows = rows.with_columns(**{TARGET: shifted.cast(pl.Float64)})
    built, morning_latest = lag_columns(
        rows=rows,
        inputs=inputs,
        weather=weather,
        weather_lead0=weather_lead0,
        lead_day=lead_day,
        arms=arms,
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
    needed = {c for arm in arms for c in EXTRA_COLUMNS[arm]}
    anchors = window_anchor_lines(
        rows=built,
        lead_day=lead_day,
        windows={c: first for c, first in WINDOW_FIRST_DAY.items() if c in needed},
        daily_columns=[c for c in (*DAILY_FEATURE_COLUMNS, "ck_clear_near_share") if c in needed],
        morning_latest=morning_latest,
    )
    lines += anchors
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


def lag_inputs(*, hourly: pl.DataFrame) -> LagInputs:
    """Build the lead-independent inputs from a lag source.

    Args:
        hourly: The lag source, which a positive control has already shifted.

    Returns:
        The lag source, clear-sky irradiance, CAMS irradiance, and the daily features.
    """
    sites = pv_sites()
    clear_sky = clear_sky_table(hourly=hourly, sites=sites)
    cams = pl.read_parquet(
        NFC_DIR / "solar_forecast_inputs.parquet", columns=["site", "time", "ghi_cams"]
    )
    daily = daily_features(
        hourly=hourly, clear_sky=clear_sky, capacity=sites.select("site", "effective_capacity_mw")
    )
    return LagInputs(hourly=hourly, clear_sky=clear_sky, cams=cams, daily=daily)


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
            "clear_sky_w_m2",
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
    control_month = f"{POSITIVE_CONTROL_DATE:%Y-%m}"
    if control_month in NWP_ERA_START_MONTHS or POSITIVE_CONTROL_DATE.day != 1:
        msg = "the positive control's date must be the start of a month that is not an era start"
        raise ValueError(msg)

    shared = shared_rows()
    hourly = lag_source_hourly()
    inputs = lag_inputs(hourly=hourly)
    weather_lead0 = weather_table(product="ens_mean", lead_day=0) if product == "ens_mean" else None
    raw = raw_hourly_power()
    report = [f"## Frames built for the weather product `{product}`", ""]
    report += [f"Shared rows: {shared.height} (the published count {PUBLISHED_ROW_COUNT}).", ""]

    for lead_day in lead_days:
        if product != "ens_mean":
            arms = REPLICATE_ARMS
        else:
            arms = (
                (*SWEEP_ARMS, *GLOBAL_ONLY_ARMS, *FRAME_ONLY_ARMS)
                if lead_day == FULL_SWEEP_LEAD_DAY
                else LONGER_LEAD_ARMS
            )
        frame, lines = build_frame(
            product=product,
            lead_day=lead_day,
            arms=arms,
            shared=shared,
            inputs=inputs,
            weather_lead0=weather_lead0,
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
                    inputs=lag_inputs(hourly=shifted_hourly),
                    weather_lead0=weather_lead0,
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
