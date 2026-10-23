"""The arms of experiment 141 and their scoring at one site in one world.

A world is the real residual stream plus the moved megawatts of one batch of injected transfers
(or no transfers, for the real year). Every arm is GAM-K's cached member forecasts plus a
correction from a filter run on the world's stream, with experiment 115's K0 spread (shared, so a
contrast is a location-only difference). The scored rows are the rows near an injected event or a
detector alarm, because far from both every arm equals K0.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import e00283_g_arms as arms134
import e00283_g_common as common134
import e00283_g_site as site_module
import e00283_kf as kf
import numpy as np
import polars as pl
from e00283_common import GRID_START
from studies.dp_assimilation import (
    FilterHyperparameters,
    correction_at,
)
from studies.dp_ens import LeadHourSpread
from studies.dp_gam import ResidualSpread
from studies.ens_members import mixture_quantiles

SERIES_OF_SITE: Final[dict[str, int]] = {
    "D5": 17,
    "D10": 10,
    "D11": 8,
    "D14": 16,
    "D16": 18,
    "D9": 7,
}
CAPACITY_MW: Final[dict[str, float]] = {
    "D5": 5.723,
    "D10": 5.398,
    "D11": 10.777,
    "D14": 11.706,
    "D16": 9.115,
    "D9": 7.104,
}
"""Effective capacity of each site; a stream in points is 100 times megawatts over this."""
ONSET_LAG: Final[int] = 96
DELAY_HOURS: Final[float] = 3.5
AFTER_STAMPS: Final[int] = 96
"""48 hours of target time after a causal alarm is known; rows in it are the C1 rows."""
END_STAMPS: Final[int] = 96
"""48 hours of target time the post hoc arms keep the prior on after the end is called."""
ONSET_TOLERANCE: Final[int] = 4
"""Stamps (2 hours) within which an alarm counts as at the start of an injected transfer."""
TAIL_STAMPS: Final[int] = 144
"""72 hours of target time after an event ends are scored as well."""


@lru_cache(maxsize=2)
def get_site(site: str) -> site_module.SiteData:
    """Load a site once per process."""
    return site_module.load_site(site=site)


@lru_cache(maxsize=8)
def k0_spread(site: str) -> LeadHourSpread:
    """Experiment 115's K0 spread with its factors."""
    with np.load(common134.SRC_115_WORK_DIR / "spread" / f"K0_{site}.npz") as stored:
        tables, factors = stored["tables"], stored["factors"]
    return LeadHourSpread(
        by_lead_band=tuple(ResidualSpread(table=t) for t in tables), factors=factors
    )


def hyper_of(data: site_module.SiteData) -> FilterHyperparameters:
    """GAM-K's fitted hyperparameters at a site."""
    k = data.kalman_hyper
    return FilterHyperparameters(k.log_q, k.logit_phi, k.log_sigma_z, k.log_sigma_eps)


def stamp_offset(data: site_module.SiteData) -> int:
    """The grid position of detector stamp 0 (a stamp's position is the stamp plus this)."""
    first_end = np.datetime64(GRID_START.replace(tzinfo=None), "us") + np.timedelta64(30, "m")
    return int(np.searchsorted(data.cache.end_times, first_end))


def to_positions(*, array: np.ndarray, data: site_module.SiteData, fill: float = 0.0) -> np.ndarray:
    """Re-index a detector-grid array (per stamp) onto the site's grid positions."""
    offset = stamp_offset(data)
    out = np.full(len(data.cache.end_times), fill)
    n = min(len(array), len(out) - offset)
    out[offset : offset + n] = array[:n]
    return out


@dataclass(frozen=True)
class Signals:
    """Per-position priors of the switched regime for one site in one world.

    Attributes:
        causal: The prior from causal alarms (active from onset-known to end-known).
        hindsight: The prior from the hindsight end rule, active from the true onset.
        oracle: The prior from the injected truth (zeros in the real year).
        step_points: The signed step of the active causal alarm, in points (0 when none).
        age_days: Days since the active causal alarm was known (0 when none).
        alarm_window: True within 48 hours after an alarm on an injected transfer is known (the
            C1 rows' target window). Alarms from the real stream alone are left out.
        event_window: True within an injected transfer and 72 hours after it.
        transfer: True inside an injected transfer.
        causal_end: The causal prior kept on for 48 hours after the end is called (post hoc).
        oracle_end: The true-interval prior kept on for 72 hours after the transfer (post hoc).
    """

    causal: np.ndarray
    hindsight: np.ndarray
    oracle: np.ndarray
    step_points: np.ndarray
    age_days: np.ndarray
    alarm_window: np.ndarray
    event_window: np.ndarray
    transfer: np.ndarray
    causal_end: np.ndarray
    oracle_end: np.ndarray


def build_signals(
    *,
    data: site_module.SiteData,
    site: str,
    intervals: pl.DataFrame,
    delta_mw: np.ndarray | None,
    p_on: float,
) -> Signals:
    """Turn detector intervals and injected events into per-position arrays.

    Args:
        data: The site data.
        site: The site label.
        intervals: Rows with `setting` (`R2_M48` or `R0_M48`), `series`, `onset`, `end`,
            `known_from`, and `step_mw`.
        delta_mw: The megawatts the injected transfers add to this site, per detector stamp, or
            `None` for the real year.
        p_on: The prior probability of the switched regime while an alarm is active.

    Returns:
        The signals.
    """
    n = len(data.cache.end_times)
    offset = stamp_offset(data)
    series = SERIES_OF_SITE[site]
    causal, hindsight, oracle = np.zeros(n), np.zeros(n), np.zeros(n)
    step, age = np.zeros(n), np.zeros(n)
    alarm_window, event_window = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    causal_end, oracle_end = np.zeros(n), np.zeros(n)
    moving = np.zeros(n, dtype=bool)
    if delta_mw is not None:
        moving = (
            to_positions(array=(np.abs(np.nan_to_num(delta_mw)) > 0).astype(float), data=data) > 0
        )
    near_transfer = (
        np.convolve(moving.astype(float), np.ones(2 * ONSET_TOLERANCE + 1), mode="same") > 0
    )
    for row in intervals.filter(pl.col("series") == series).iter_rows(named=True):
        onset, end, known = row["onset"] + offset, row["end"] + offset, row["known_from"] + offset
        if row["setting"] == "R2_M48":
            start = onset + ONSET_LAG
            stop = max(known, start + 1)
            lo, hi = max(start, 0), min(stop, n)
            if lo < hi:
                causal[lo:hi] = p_on
                causal_end[lo : min(hi + END_STAMPS, n)] = p_on
                points = 100.0 * row["step_mw"] / CAPACITY_MW[site]
                step[lo:hi] = points if np.isfinite(points) else 0.0
                age[lo:hi] = (np.arange(lo, hi) - start) / 48.0
            wlo, whi = max(start, 0), min(start + AFTER_STAMPS, n)
            if 0 <= onset < n and near_transfer[onset]:
                alarm_window[wlo:whi] = True
        else:
            lo, hi = max(onset, 0), min(max(end, onset + 1), n)
            hindsight[lo:hi] = p_on
    if delta_mw is not None:
        oracle[moving] = p_on
        cumulative = np.concatenate([[0], np.cumsum(moving)])
        lagged = np.concatenate([np.zeros(TAIL_STAMPS + 1, dtype=cumulative.dtype), cumulative])[
            : len(cumulative)
        ]
        event_window = (cumulative[1:] - lagged[: len(moving)]) > 0
        oracle_end[event_window] = p_on
    return Signals(
        causal,
        hindsight,
        oracle,
        step,
        age,
        alarm_window,
        event_window,
        moving,
        causal_end,
        oracle_end,
    )


def correction_for(
    *,
    data: site_module.SiteData,
    stream: np.ndarray,
    prior: np.ndarray,
    multiplier: float,
    update_regime: bool = True,
    noise_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Return the total correction at every row from the two-regime filter.

    `noise_scale` inflates the observation-noise variance at each grid row (a soft mask).
    """
    trace, _ = kf.run_two_regime(
        residual=stream,
        band=data.cache.calendar.band.astype(np.int64),
        prior=prior,
        hyper=hyper_of(data),
        multiplier=multiplier,
        update_regime=update_regime,
        noise_scale=noise_scale,
    )
    cutoff = data.cutoffs(delay_hours=DELAY_HOURS)[data.run_of_row]
    return correction_at(
        trace=trace, cutoff=cutoff, target=data.position, band=data.hour_band
    ).total


def quantiles_at(
    *, data: site_module.SiteData, correction: np.ndarray, spread: LeadHourSpread, rows: np.ndarray
) -> np.ndarray:
    """Mixture quantiles at chosen rows, shape (rows, 11), with the correction applied."""
    out = np.empty((len(rows), 11))
    offsets = arms134.spread_offsets(
        spread=spread,
        band=data.hour_band[rows],
        quarter=data.quarter[rows],
        lead_hours=data.lead_hours[rows],
    )
    runs = data.run_of_row[rows]
    for run in np.unique(runs):
        chosen = np.flatnonzero(runs == run)
        columns = rows[chosen] - data.offsets[run]
        corrected = data.member_means[run][:, columns] + correction[rows[chosen]][None, :]
        per_member = corrected.T[:, :, None] + offsets[chosen][:, None, :]
        out[chosen] = mixture_quantiles(member_quantiles=per_member)
    return out


def rows_where(
    *, data: site_module.SiteData, mask_by_position: np.ndarray, period: str
) -> np.ndarray:
    """Rows of a period whose target is flagged, with a reading, outside the logged masks."""
    flagged = mask_by_position[data.position]
    ok = np.isfinite(data.actual) & ~data.features.switching_mask[data.position]
    return np.flatnonzero(flagged & ok & data.period_rows(period=period))


def score_arms(
    *,
    data: site_module.SiteData,
    site: str,
    corrections: dict[str, np.ndarray],
    delta_points: np.ndarray | None,
    rows: np.ndarray,
) -> pl.DataFrame:
    """Score every arm on `rows`: quantile score, and errors of the median and the mean forecast.

    Args:
        data: The site data.
        site: The site label.
        corrections: Each arm's total correction at every row.
        delta_points: The injected points at each grid position (added to the actual), or `None`.
        rows: The rows to score.

    Returns:
        One row per (arm, row).
    """
    actual = data.actual[rows] + (
        0.0 if delta_points is None else delta_points[data.position[rows]]
    )
    spread = k0_spread(site)
    frames = []
    for arm, correction in corrections.items():
        q = quantiles_at(data=data, correction=correction, spread=spread, rows=rows)
        score = arms134.quantile_score(actual=actual, quantiles=q)
        mean_forecast = data.member_mean[rows] + correction[rows]
        frames.append(
            pl.DataFrame(
                {
                    "arm": arm,
                    "site": site,
                    "row": rows,
                    "position": data.position[rows],
                    "run": data.runs[data.run_of_row[rows]],
                    "lead_hours": data.lead_hours[rows],
                    "qs": score,
                    "err_median": q[:, 5] - actual,
                    "err_mean": mean_forecast - actual,
                    "q05": q[:, 1],
                    "q95": q[:, 9],
                    "q01": q[:, 0],
                    "q99": q[:, 10],
                    "actual": actual,
                }
            )
        )
    return pl.concat(frames)
