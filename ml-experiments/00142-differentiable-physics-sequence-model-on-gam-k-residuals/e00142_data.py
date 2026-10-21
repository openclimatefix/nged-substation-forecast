"""Build the sequence model's samples: one per (substation, run), on flat arrays.

A sample holds the 14 UTC days before the origin's day plus the origin's own partial day (to the
latest usable reading, 3.5 hours before the origin) as 48-bin residual vectors with a reading mask
and calendar flags, the target day D+1's calendar and forecast weather, K0's median and quantiles
at the target's 48 half-hours, and the target `actual - K0 median`. Nothing from the target day's
readings, or any reading after the cutoff, enters the inputs (`assert_no_lookahead`).

**Half-hours are indexed by UTC position within the day (0 to 47), not by local clock time.** The
network sees a UTC day's 48 bins; the day type and holiday flags use the UTC date.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

import e00142_arms as arms
import e00142_common as common
import e00142_profiles as profiles
import e00142_site as site_module
import numpy as np
import torch

HISTORY_DAYS: Final[int] = 14
N_DAYS: Final[int] = HISTORY_DAYS + 1
"""The 14 full UTC days before the origin's day, then the origin's partial day."""
DELAY_HOURS: Final[float] = 3.5
N_BINS: Final[int] = 48
N_DAY_FEATURES: Final[int] = 2 * N_BINS + 5 + 1
"""Residuals, reading mask, day type (5 one-hot), holiday-like flag."""
N_TARGET_FEATURES: Final[int] = 5 + 1 + 1 + 3 * N_BINS
"""Day type, holiday-like flag, Christmas-window flag, temperature, shortwave, K0 median."""
RESIDUAL_SCALE: Final[float] = 5.0
TEMPERATURE_CENTRE_C: Final[float] = 10.0
TEMPERATURE_SCALE_C: Final[float] = 8.0
SHORTWAVE_SCALE: Final[float] = 400.0
K0_SCALE: Final[float] = 50.0
EPOCH: Final[np.datetime64] = np.datetime64("1970-01-01", "D")


@dataclass(frozen=True)
class Samples:
    """Every sample of one substation.

    Attributes:
        site: The label.
        run: Index into `SiteData.runs` of each sample.
        target_day: UTC day number (days since 1970-01-01) of each target day D+1.
        history: Shape (n, N_DAYS, N_DAY_FEATURES).
        target: Shape (n, N_TARGET_FEATURES).
        k0_offsets: K0's quantile minus K0's median at the target, shape (n, 48, 11).
        y: Actual minus K0's median at the target, shape (n, 48), `NaN` where not scorable.
        rows: The `SiteData` row of each target bin, shape (n, 48).
        period: Each sample's run period.
        month_code: `100 * year + month` of each target day.
    """

    site: str
    run: np.ndarray
    target_day: np.ndarray
    history: np.ndarray
    target: np.ndarray
    k0_offsets: np.ndarray
    y: np.ndarray
    rows: np.ndarray
    period: np.ndarray
    month_code: np.ndarray


def _day_flags(*, day_numbers: np.ndarray, banned: set[int]) -> np.ndarray:
    """Return day type one-hot (5) and the holiday-like flag for UTC day numbers, shape (n, 6)."""
    weekday = (day_numbers + 3) % 7
    types = np.array(profiles.DAY_TYPES)[weekday]
    out = np.zeros((len(day_numbers), 6))
    out[np.arange(len(day_numbers)), types] = 1.0
    out[:, 5] = [float(int(d) in banned) for d in day_numbers]
    return out


def _christmas(*, day_numbers: np.ndarray) -> np.ndarray:
    first = date(1970, 1, 1)
    flags = []
    for d in day_numbers:
        day = first + timedelta(days=int(d))
        flags.append(float(day.month == 12 and day.day in profiles.CHRISTMAS_RUN_DAYS))
    return np.array(flags)


def build_samples(  # noqa: PLR0915
    *,
    data: site_module.SiteData,
    k0_q50: np.ndarray,
    k0_quantiles: np.ndarray,
    setup_runs: list,
    actual: np.ndarray | None = None,
    stream: np.ndarray | None = None,
) -> Samples:
    """Build a substation's samples.

    Args:
        data: The site data.
        k0_q50: K0's median at every row (`NaN` where K0 has no forecast).
        k0_quantiles: K0's 11 quantiles at every row, shape (n_rows, 11).
        setup_runs: The ensemble runs of the site (`SiteSetup.runs`), indexed by run index.
        actual: Readings to learn from, `data.actual` by default; an injected copy for the gate.
        stream: The residual stream, `data.stream` by default; an injected copy for the gate.

    Returns:
        The samples of every run with a full day D+1 and a K0 forecast.
    """
    actual = data.actual if actual is None else actual
    ends = data.cache.end_times
    assert (np.diff(ends) == np.timedelta64(30, "m")).all(), "the grid must be regular"
    stream = data.stream if stream is None else stream
    cutoff = data.cutoffs(delay_hours=DELAY_HOURS)
    d1 = data.d1_rows()
    first_day = (ends[0] - np.timedelta64(30, "m")).astype("datetime64[D]")
    origin_days = data.run_origins.astype("datetime64[D]")
    first = date(1970, 1, 1)
    low = int((origin_days.min() - EPOCH).astype(int)) - HISTORY_DAYS - 30
    high = int((origin_days.max() - EPOCH).astype(int)) + 40
    banned = {
        (d - first).days
        for d in profiles.holiday_like_days(
            first=first + timedelta(days=low), last=first + timedelta(days=high)
        )
    }
    keep, histories, targets, offsets, ys, rows_out, target_days = [], [], [], [], [], [], []
    for run in range(len(data.runs)):
        rows = np.arange(data.offsets[run], data.offsets[run + 1])
        rows = rows[d1[rows]]
        if len(rows) != N_BINS or not np.isfinite(k0_q50[rows]).all():
            continue
        origin_day = origin_days[run]
        day_numbers = (origin_day - EPOCH).astype(int) + np.arange(-HISTORY_DAYS, 1)
        history = np.zeros((N_DAYS, N_DAY_FEATURES))
        for slot, day_number in enumerate(day_numbers):
            start = (
                int((EPOCH + np.timedelta64(int(day_number), "D") - first_day).astype(np.int64))
                * N_BINS
            )
            if start < 0 or start + N_BINS > len(stream):
                continue
            positions = start + np.arange(N_BINS)
            values = stream[positions]
            usable = np.isfinite(values) & (positions <= cutoff[run])
            history[slot, :N_BINS] = np.where(usable, values, 0.0) / RESIDUAL_SCALE
            history[slot, N_BINS : 2 * N_BINS] = usable
        history[:, 2 * N_BINS :] = _day_flags(day_numbers=day_numbers, banned=banned)
        target_day = int(day_numbers[-1]) + 1
        flags = _day_flags(day_numbers=np.array([target_day]), banned=banned)[0]
        ens = setup_runs[int(data.runs[run])]
        local = rows - data.offsets[run]
        temperature = np.nanmean(ens.temperature_c[:, local], axis=0)
        shortwave = np.nanmean(ens.shortwave_w_m2[:, local], axis=0)
        target = np.concatenate(
            [
                flags,
                _christmas(day_numbers=np.array([target_day])),
                (temperature - TEMPERATURE_CENTRE_C) / TEMPERATURE_SCALE_C,
                shortwave / SHORTWAVE_SCALE,
                k0_q50[rows] / K0_SCALE,
            ]
        )
        reading = actual[rows]
        usable_target = np.isfinite(reading) & data.scorable[rows]
        y = np.where(usable_target, reading - k0_q50[rows], np.nan)
        keep.append(run)
        histories.append(history)
        targets.append(target)
        offsets.append(k0_quantiles[rows] - k0_q50[rows][:, None])
        ys.append(y)
        rows_out.append(rows)
        target_days.append(target_day)
    run_array = np.array(keep)
    target_day_array = np.array(target_days)
    target_dates = EPOCH + target_day_array.astype("timedelta64[D]")
    month = target_dates.astype("datetime64[M]")
    year_number = month.astype("datetime64[Y]").astype(int) + 1970
    month_number = month.astype(int) % 12 + 1
    return Samples(
        site=data.site,
        run=run_array,
        target_day=target_day_array,
        history=np.array(histories, dtype=np.float32),
        target=np.array(targets, dtype=np.float32),
        k0_offsets=np.array(offsets, dtype=np.float32),
        y=np.array(ys, dtype=np.float32),
        rows=np.array(rows_out),
        period=data.run_period[run_array],
        month_code=year_number * 100 + month_number,
    )


def assert_no_lookahead(*, data: site_module.SiteData, samples: Samples) -> None:
    """Fail if any history bin has a reading mask set after the run's cutoff position."""
    cutoff = data.cutoffs(delay_hours=DELAY_HOURS)
    ends = data.cache.end_times
    first_day = (ends[0] - np.timedelta64(30, "m")).astype("datetime64[D]")
    for index, run in enumerate(samples.run):
        origin_day = data.run_origins[run].astype("datetime64[D]")
        start = int((origin_day - first_day).astype(np.int64)) * N_BINS
        partial_mask = samples.history[index, -1, N_BINS : 2 * N_BINS]
        positions = start + np.arange(N_BINS)
        assert not partial_mask[positions > cutoff[run]].any(), (samples.site, int(run))
        assert samples.history[index, -1, :N_BINS][positions > cutoff[run]].tolist() == [0.0] * int(
            (positions > cutoff[run]).sum()
        )


def to_tensors(*, samples: Samples, index: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    """Return a subset of the samples as tensors on a device."""
    return {
        "history": torch.as_tensor(samples.history[index], device=device),
        "target": torch.as_tensor(samples.target[index], device=device),
        "offsets": torch.as_tensor(samples.k0_offsets[index], device=device),
        "y": torch.as_tensor(samples.y[index], device=device),
    }


def load_k0_with_quantiles(
    *, data: site_module.SiteData, site: str
) -> tuple[arms.Correction, np.ndarray, np.ndarray]:
    """Return experiment 115's K0 correction, K0's median, and K0's 11 quantiles at every row."""
    from studies.dp_ens import LeadHourSpread
    from studies.dp_gam import ResidualSpread

    with np.load(common.SRC_115_WORK_DIR / "corrections" / f"K0_{site}.npz") as stored:
        k0 = arms.Correction(
            level=stored["level"], anomaly=stored["anomaly"], shape=stored["shape"]
        )
    with np.load(common.SRC_115_WORK_DIR / "spread" / f"K0_{site}.npz") as stored:
        tables, factors = stored["tables"], stored["factors"]
    spread = LeadHourSpread(
        by_lead_band=tuple(ResidualSpread(table=t) for t in tables), factors=factors
    )
    quantiles, rows = arms.quantiles_for_runs(
        data=data, correction=k0.total, spread=spread, run_indices=np.arange(len(data.runs))
    )
    full = np.full((data.n_rows, quantiles.shape[1]), np.nan)
    full[rows] = quantiles
    return k0, full[:, 5].copy(), full
