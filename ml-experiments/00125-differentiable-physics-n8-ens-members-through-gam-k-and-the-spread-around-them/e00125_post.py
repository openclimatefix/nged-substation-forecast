"""Build every arm of experiment 125 from the cache of corrected member means.

An arm is a choice of centres (the corrected members, their mean, or members rescaled about their
mean) and of the residual table placed around each centre. `build_rows` returns the quantiles the
scorers read, pooled the way `studies.dp_ens.forecast_run` pools them.

| Arm | Centres | Residual table |
|---|---|---|
| K0 | corrected members | ensemble-mean residuals (`k0_table`) |
| MP | corrected members | proxy-driven residuals (`mp_table`) |
| MP-s | members rescaled by s | as MP |
| MM | the corrected ensemble mean | as K0 |
| MO | corrected members | none |
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import e00125_cache as cache_module
import numpy as np
from studies.dp_ens import LeadHourSpread
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours
from studies.ens_members import MIXTURE_LEVELS, mixture_quantiles

N_LEVELS: Final[int] = len(MIXTURE_LEVELS)
FACTOR_GRID: Final[tuple[float, ...]] = (0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5)
"""The wide factor grid of K0-w, MP, MM, and MP-s."""
STORED_FACTOR_GRID: Final[tuple[float, ...]] = (0.9, 1.0, 1.15, 1.3)
"""Experiment 111's factor grid, kept only for the stored benchmark K0."""
S_GRID: Final[tuple[float, ...]] = (0.8, 1.0, 1.25, 1.5, 2.0, 3.0)
TAIL_LEVEL_POSITIONS: Final[tuple[int, ...]] = (0, 1, 9, 10)
"""Positions of q01, q05, q95, and q99 among the 11 levels."""


@dataclass(frozen=True)
class Rows:
    """The quantiles and columns the scorers read, one row per scored half-hour.

    Attributes:
        quantiles: Shape (rows, 11), in points.
        actual: The observed power.
        run: The run index of each row.
        valid_time: Naive UTC valid time of each row.
        lead_hours: The lead from the origin.
        hour_band: The hour band.
        quarter: The calendar quarter.
        switching_mask: True inside a hand-filtered switching event.
    """

    quantiles: np.ndarray
    actual: np.ndarray
    run: np.ndarray
    valid_time: np.ndarray
    lead_hours: np.ndarray
    hour_band: np.ndarray
    quarter: np.ndarray
    switching_mask: np.ndarray


def _stack(*, runs: Sequence[cache_module.RunCache], name: str) -> np.ndarray:
    return np.concatenate([getattr(run, name) for run in runs])


def residual_table(
    *, runs: Sequence[cache_module.RunCache], centre: str, row_filter: np.ndarray | None = None
) -> LeadHourSpread:
    """Fit the centred residual table of the chosen centres outside the switching masks.

    Args:
        runs: The cached runs the table is fitted on.
        centre: `mean` for the corrected ensemble mean, `proxy` for the proxy-driven forecast.
        row_filter: Optional boolean mask over the stacked rows that keeps only some of them, for
            example the rows whose valid time lies in the tuning months.

    Returns:
        The table with every factor 1.
    """
    if centre == "mean":
        centres = np.concatenate([run.corrected.mean(axis=0) for run in runs])
    elif centre == "proxy":
        centres = _stack(runs=runs, name="proxy")
    else:
        msg = f"centre must be 'mean' or 'proxy', not {centre!r}"
        raise ValueError(msg)
    actual = _stack(runs=runs, name="actual").astype(np.float64)
    keep = ~_stack(runs=runs, name="switching_mask") & np.isfinite(centres)
    if row_filter is not None:
        keep &= row_filter
    return LeadHourSpread.fit(
        residuals=(actual - centres)[keep],
        band=_stack(runs=runs, name="hour_band")[keep].astype(np.int64),
        quarter=_stack(runs=runs, name="quarter")[keep].astype(np.int64),
        lead_hours=_stack(runs=runs, name="lead_hours")[keep],
    )


def rescale_members(
    *, corrected: np.ndarray, lead_hours: np.ndarray, s: Sequence[float]
) -> np.ndarray:
    """Move each member away from the ensemble mean by the factor of its row's lead band.

    Args:
        corrected: Shape (members, rows).
        lead_hours: The lead of each row.
        s: One factor per lead band.

    Returns:
        Shape (members, rows) with the same mean over members.
    """
    factor = np.asarray(s, dtype=np.float64)[lead_band_of_hours(lead_hours=lead_hours)]
    mean = corrected.mean(axis=0, keepdims=True)
    return mean + factor[None, :] * (corrected - mean)


def run_quantiles(
    *, run: cache_module.RunCache, spread: LeadHourSpread | None, centres: np.ndarray, pool: bool
) -> np.ndarray:
    """Return one run's 11 quantiles per row.

    Args:
        run: The cached run.
        spread: The table placed around each centre, or `None` for no table (the centres' own
            empirical distribution).
        centres: Shape (centres, rows).
        pool: True to pool the centres as a mixture; False for a single centre.

    Returns:
        Shape (rows, 11), non-decreasing along the levels.
    """
    if spread is None:
        per_centre = np.repeat(centres[:, :, None], N_LEVELS, axis=2)
    else:
        per_centre = np.stack(
            [
                spread.quantiles(
                    mean=row,
                    band=run.hour_band.astype(np.int64),
                    quarter=run.quarter.astype(np.int64),
                    lead_hours=run.lead_hours,
                )
                for row in centres
            ]
        )
    if not pool:
        return per_centre[0]
    return mixture_quantiles(member_quantiles=np.transpose(per_centre, (1, 0, 2)))


def build_rows(
    *,
    runs: Sequence[cache_module.RunCache],
    spread: LeadHourSpread | None,
    centre: str,
    s: Sequence[float] | None = None,
) -> Rows:
    """Build an arm's rows from the cached runs.

    Args:
        runs: The cached runs.
        spread: The residual table with its factors, or `None` for arm MO.
        centre: `members` for the corrected members pooled as a mixture, `mean` for their mean.
        s: Optional rescaling of the members about their mean, per lead band.

    Returns:
        The rows of every run, in run order.
    """
    pieces = []
    for run in runs:
        corrected = run.corrected
        if s is not None:
            corrected = rescale_members(corrected=corrected, lead_hours=run.lead_hours, s=s)
        if centre == "mean":
            pieces.append(
                run_quantiles(
                    run=run,
                    spread=spread,
                    centres=corrected.mean(axis=0, keepdims=True),
                    pool=False,
                )
            )
        else:
            pieces.append(run_quantiles(run=run, spread=spread, centres=corrected, pool=True))
    return Rows(
        quantiles=np.concatenate(pieces),
        actual=_stack(runs=runs, name="actual").astype(np.float64),
        run=np.concatenate([np.full(len(r.actual), r.run) for r in runs]),
        valid_time=_stack(runs=runs, name="valid_time"),
        lead_hours=_stack(runs=runs, name="lead_hours"),
        hour_band=_stack(runs=runs, name="hour_band"),
        quarter=_stack(runs=runs, name="quarter"),
        switching_mask=_stack(runs=runs, name="switching_mask"),
    )


def pinball(*, actual: np.ndarray, quantiles: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Return the pinball loss of each quantile column, shape (rows, len(levels))."""
    tau = np.asarray(levels)[None, :]
    error = actual[:, None] - quantiles
    return np.maximum(tau * error, (tau - 1.0) * error)


def quantile_score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 levels, per row, in points."""
    return 2.0 * pinball(actual=actual, quantiles=quantiles, levels=MIXTURE_LEVELS).mean(axis=1)


def tail_pinball(*, rows: Rows) -> np.ndarray:
    """Return the mean pinball loss over q01, q05, q95, and q99 per row."""
    levels = [MIXTURE_LEVELS[i] for i in TAIL_LEVEL_POSITIONS]
    return pinball(
        actual=rows.actual, quantiles=rows.quantiles[:, TAIL_LEVEL_POSITIONS], levels=levels
    ).mean(axis=1)


def band_means(*, values: np.ndarray, rows: Rows) -> np.ndarray:
    """Return the mean of per-row values in each lead band outside the switching masks."""
    band = lead_band_of_hours(lead_hours=rows.lead_hours)
    keep = ~rows.switching_mask
    return np.array(
        [values[keep & (band == index)].mean() for index in range(N_LEAD_BANDS)], dtype=np.float64
    )


def month_mask(*, valid_time: np.ndarray, months: frozenset[tuple[int, int]]) -> np.ndarray:
    """Return a mask of the rows whose half-hour starts in one of the (year, month) pairs."""
    starts = valid_time - np.timedelta64(30, "m")
    year = starts.astype("datetime64[Y]").astype(np.int64) + 1970
    month = starts.astype("datetime64[M]").astype(np.int64) % 12 + 1
    mask = np.zeros(len(starts), dtype=bool)
    for y, m in months:
        mask |= (year == y) & (month == m)
    return mask


def select(*, rows: Rows, mask: np.ndarray) -> Rows:
    """Return the rows where the mask is true."""
    return Rows(**{name: getattr(rows, name)[mask] for name in rows.__dataclass_fields__})


def stacked_months_mask(
    *, runs: Sequence[cache_module.RunCache], months: frozenset[tuple[int, int]] | None
) -> np.ndarray | None:
    """Return the month mask over the stacked rows of the runs, or `None` for no restriction."""
    if months is None:
        return None
    return month_mask(valid_time=_stack(runs=runs, name="valid_time"), months=months)
