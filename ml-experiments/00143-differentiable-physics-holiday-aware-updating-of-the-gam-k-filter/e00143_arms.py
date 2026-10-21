"""The arms of experiment 143: GAM-K's filter with holiday readings kept out of the update.

| Arm | Readings the filter ignores | Correction on a holiday target |
|---|---|---|
| `K0` | none (experiment 111's GAM-K, the control) | the carried level and transient |
| `K-skip` | every reading on a holiday day | the carried level and transient |
| `K-skip-core` | bank holidays and 24 to 26 December | the carried level and transient |
| `K-skip+1` | every holiday day and the day after one | the carried level and transient |
| `K-zero` | every reading on a holiday day | zero (experiment 118's drop-and-zero idea) |

Every arm is the member forecasts of experiment 115 plus a correction. All four use the filter
hyperparameters, the spread tables, and the spread factors of K0, so the arms differ only in the
correction. The code follows experiment 134's `e00134_arms.py`.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import e00143_common as common
import e00143_holidays as hol
import e00143_shape as shape
import e00143_site as site_module
import numpy as np
import polars as pl
from studies.dp_ens import LeadHourSpread
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours
from studies.dp_gam import QUANTILE_COLUMNS, ResidualSpread
from studies.ens_members import mixture_quantiles

ARMS: Final[tuple[str, ...]] = ("K0", "K-skip", "K-skip-core", "K-skip+1", "K-zero")
DELAY_HOURS: Final[float] = 3.5
"""The telemetry delay, as in experiments 115 and 134."""
LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
WORK_DIR: Final[Path] = common.WORK_DIR
FRAMES_DIR: Final[Path] = WORK_DIR / "frames"
SRC_115_FRAMES_DIR: Final[Path] = common.SRC_115_WORK_DIR / "frames"


@dataclass(frozen=True)
class Correction:
    """An arm's correction at every row of a site.

    Attributes:
        level: The persistent level part.
        anomaly: The decaying transient part.
    """

    level: np.ndarray
    anomaly: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """The correction to add to the member means."""
        return self.level + self.anomaly


@lru_cache(maxsize=1)
def holiday_numbers() -> np.ndarray:
    """Return the holiday set over the whole grid, as sorted days since 1970-01-01."""
    cache = common.load_cache()
    dates = cache.calendar.local_date.astype("datetime64[D]")
    first = dates.min().astype(object)
    last = dates.max().astype(object)
    return hol.day_numbers(days=hol.holiday_days(first=first, last=last))


def grid_days(*, data: site_module.SiteData) -> np.ndarray:
    """Return each grid row's local date as days since 1970-01-01."""
    return data.cache.calendar.local_date.astype("datetime64[D]").astype(np.int64)


def stream_for(
    *, data: site_module.SiteData, arm: str, stream: np.ndarray | None = None
) -> np.ndarray:
    """Return the residual stream the arm's filter reads: skipped days become missing.

    Args:
        data: The site data.
        arm: One of `ARMS`.
        stream: The residual stream to mask; the site's real stream if `None`.

    Returns:
        The stream with the skipped days set to `NaN`.
    """
    base = data.stream if stream is None else stream
    if arm == "K0":
        return base
    if arm == "K-skip-core":
        skip = hol.core_days(holiday=holiday_numbers())
    else:
        skip = hol.skip_days(holiday=holiday_numbers(), also_day_after=arm == "K-skip+1")
    return np.where(np.isin(grid_days(data=data), skip), np.nan, base)


def correction_from_stream(
    *, data: site_module.SiteData, stream: np.ndarray, zero_holiday_targets: bool = False
) -> Correction:
    """Run K0's filter over a stream and look up the correction at every row.

    Args:
        data: The site data.
        stream: The residual stream.
        zero_holiday_targets: True to set the correction to zero on rows whose target is a
            holiday day.

    Returns:
        The correction at every row.
    """
    structure = shape.build_structure(variant="KW", band_of_half_hour=data.band_of_half_hour)
    trace = shape.run_shape_filter(
        residual=stream, half_hour=data.half_hour, structure=structure, hyper=data.kalman_hyper
    )
    cutoff = data.cutoffs(delay_hours=DELAY_HOURS)[data.run_of_row]
    found = shape.lookup(
        trace=trace,
        structure=structure,
        cutoff=cutoff,
        target=data.position,
        half_hour=data.row_half_hour,
    )
    level, anomaly = found.level, found.anomaly
    if zero_holiday_targets:
        holiday_row = np.isin(grid_days(data=data)[data.position], holiday_numbers())
        level = np.where(holiday_row, 0.0, level)
        anomaly = np.where(holiday_row, 0.0, anomaly)
    return Correction(level=level, anomaly=anomaly)


def arm_correction(
    *, data: site_module.SiteData, arm: str, stream: np.ndarray | None = None
) -> Correction:
    """Return an arm's correction at every row of a site, for the real or a changed stream."""
    return correction_from_stream(
        data=data,
        stream=stream_for(data=data, arm=arm, stream=stream),
        zero_holiday_targets=arm == "K-zero",
    )


def k0_spread(*, site: str) -> LeadHourSpread:
    """Return experiment 115's K0 spread with the factors it chose."""
    with np.load(common.SRC_115_WORK_DIR / "spread" / f"K0_{site}.npz") as stored:
        tables = stored["tables"]
    factors = json.loads((common.SRC_115_WORK_DIR / "tuned.json").read_text())["spread_factors"][
        "K0"
    ]
    return LeadHourSpread(
        by_lead_band=tuple(ResidualSpread(table=table) for table in tables),
        factors=np.array(factors),
    )


def spread_offsets(
    *, spread: LeadHourSpread, band: np.ndarray, quarter: np.ndarray, lead_hours: np.ndarray
) -> np.ndarray:
    """Return the spread's 11 offsets at each row, shape (rows, 11), factors applied."""
    lead = lead_band_of_hours(lead_hours=lead_hours)
    out = np.empty((len(band), len(LEVELS)))
    for index in range(N_LEAD_BANDS):
        chosen = lead == index
        if chosen.any():
            out[chosen] = (
                spread.factors[index]
                * spread.by_lead_band[index].table[band[chosen], quarter[chosen]]
            )
    return out


def quantiles_for_runs(
    *,
    data: site_module.SiteData,
    correction: np.ndarray,
    spread: LeadHourSpread,
    run_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mixture quantiles of chosen runs, and the rows they cover."""
    pieces, covered = [], []
    offsets = spread_offsets(
        spread=spread, band=data.hour_band, quarter=data.quarter, lead_hours=data.lead_hours
    )
    for run in run_indices:
        rows = np.arange(data.offsets[run], data.offsets[run + 1])
        corrected = data.member_means[run] + correction[rows][None, :]
        per_member = corrected.T[:, :, None] + offsets[rows][:, None, :]
        pieces.append(mixture_quantiles(member_quantiles=per_member))
        covered.append(rows)
    return np.concatenate(pieces), np.concatenate(covered)


def frame_for_arm(
    *,
    data: site_module.SiteData,
    arm: str,
    correction: Correction,
    spread: LeadHourSpread,
    run_indices: np.ndarray | None = None,
) -> pl.DataFrame:
    """Return an arm's forecast rows with a reading, in the stored column layout.

    Columns: `time_series_id`, `run`, `power_fcst_init_time`, `valid_time`, `lead_hours`, `seed`,
    `arm`, `q01` to `q99`, `actual`, `mean`, `level`, `anomaly`, `hour_band`, `quarter`,
    `switching_mask`, and `period`, in points.
    """
    runs = np.arange(len(data.runs)) if run_indices is None else run_indices
    quantiles, rows = quantiles_for_runs(
        data=data, correction=correction.total, spread=spread, run_indices=runs
    )
    keep = np.isfinite(data.actual[rows])
    rows, quantiles = rows[keep], quantiles[keep]
    n = len(rows)
    valid = data.cache.end_times[data.position[rows]]
    origin = data.run_origins[data.run_of_row[rows]]
    table = data.features.table
    columns: dict[str, object] = {
        "time_series_id": np.full(n, table.time_series_id, dtype=np.int32),
        "run": data.runs[data.run_of_row[rows]].astype(np.int32),
        "power_fcst_init_time": pl.Series(origin).dt.replace_time_zone("UTC"),
        "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
        "lead_hours": data.lead_hours[rows],
        "seed": np.zeros(n, dtype=np.int8),
        "arm": np.full(n, arm),
        **{
            column: quantiles[:, position].astype(np.float32)
            for position, column in enumerate(QUANTILE_COLUMNS)
        },
        "actual": data.actual[rows].astype(np.float32),
        "mean": (data.member_mean + correction.total)[rows].astype(np.float32),
        "level": correction.level[rows].astype(np.float32),
        "anomaly": correction.anomaly[rows].astype(np.float32),
        "hour_band": data.hour_band[rows].astype(np.int8),
        "quarter": data.quarter[rows].astype(np.int8),
        "switching_mask": data.features.switching_mask[data.position[rows]],
        "period": np.full(n, ""),
    }
    frame = pl.DataFrame(columns)
    names = {int(i): str(p) for i, p in zip(data.runs, data.run_period, strict=True)}
    periods = np.array([names[int(r)] for r in frame["run"].to_numpy()])
    return frame.with_columns(period=pl.Series(periods))


def replace_period_masks(*, frame: pl.DataFrame, data: site_module.SiteData) -> pl.DataFrame:
    """Add `row_period`: the period whose scoring rule the row meets, or an empty string.

    A tuning or replicate row has its valid time in the period's months too (see
    `SiteData.period_rows`).
    """
    codes = data.month_code[
        np.searchsorted(
            data.cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
    ]
    tuning = np.isin(codes, [100 * y + m for y, m in common.TUNING_MONTHS])
    replicate = np.isin(codes, [100 * y + m for y, m in common.REPLICATE_MONTHS])
    period = frame["period"].to_numpy()
    row_period = np.where(
        period == "scored",
        "scored",
        np.where(
            (period == "tuning") & tuning,
            "tuning",
            np.where((period == "replicate") & replicate, "replicate", ""),
        ),
    )
    return frame.with_columns(row_period=pl.Series(row_period))


def frame_path(*, arm: str, site: str) -> Path:
    """Return the file that holds an arm's frame at a site."""
    return FRAMES_DIR / f"{arm}_{site}.parquet"
