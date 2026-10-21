"""A joint fit of the convex demand model and its Kalman correction for the DP series.

**The two-stage fit minimises the model's own squared error, and the filter then runs on what is
left. The joint fit minimises the squared error that is left after the filter has corrected the
day-ahead forecast.** The coefficients are therefore not asked to explain a level or a transient
that the filter would track anyway.

**For fixed noise settings the joint fit is an exact bounded quadratic problem.** The Kalman gains
depend on which readings are present and never on their values, so the day-ahead correction at a
case is a fixed linear combination of the residuals before it. The residual stream is linear in the
coefficients (`r = y - X theta`), so the corrected error of a case is

    e = (y_t - corr(y)) - sum_j theta_j * (X_tj - corr(X_j))

where `corr(.)` is the correction the filter gives when it runs on that stream. Running the filter
on every design column and on the target (a batch of streams, which `studies.dp_filter_grid` does
on the GPU) gives the corrected design matrix, and its Gram matrix goes into the same bounded
solver as the two-stage fit (`studies.dp_gam.solve_bounded_quadratic`) with the same penalties. No
gradient descent is involved, and the noise settings are inputs, not parameters.

**The cases are the day-ahead cases of the scoring.** Each training half-hour is the target of one
case: the forecast made at the 09:00 UTC initialisation whose window of valid times 12 to 36 hours
ahead contains it, from the residuals that ended at least `DEFAULT_DELAY_HOURS` before that
initialisation. The windows of consecutive days tile the time axis, so no half-hour is a target
twice.

**An optional ridge ties the coefficients to an anchor fit** (normally the two-stage fit at the same
cut), `lambda * (theta - anchor)' G (theta - anchor)` with `G` the plain Gram matrix of the same
rows, so `lambda` is measured against the data's own weight and the shapes move only as far as the
corrected error needs.

**The filter reads the training residuals, which are in-sample.** They are smaller than the
out-of-sample residuals that the filter reads at forecast time, so the joint fit may under-weight
the correction's noise. The first `BURN_IN_DAYS` days of the window are left out of the loss, while
the filter converges.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np
import torch

from studies.dp_assimilation import DEFAULT_DELAY_HOURS, usable_cutoff_positions
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_filter_grid import correction_at_cutoffs, run_kalman_grid
from studies.dp_gam import (
    DEFAULT_MAX_ITERATIONS,
    N_BANDS,
    NODE_MAX_ITERATIONS,
    GamFit,
    GamLayout,
    GamSpec,
    TrainingWindow,
    accumulate_gram,
    design_matrix,
    last_signal_season_year,
    penalty_matrix,
    pinned_solar_points,
    prepare_fit,
    row_chunks,
    solve_bounded_quadratic,
)
from studies.dp_pinned_capacity import PinnedCapacity

BURN_IN_DAYS: Final[int] = 14
"""The first days of the window are left out of the loss, while the filter converges."""

INITIALISATION_HOUR_UTC: Final[int] = 9
"""The hour of the daily initialisation, 09:00 UTC."""

FIRST_LEAD_HOURS: Final[int] = 12
"""The first lead of a day-ahead case's window of valid times, in hours after the initialisation."""

LAST_LEAD_HOURS: Final[int] = 36
"""The last lead of a day-ahead case's window of valid times, in hours after the initialisation."""


def day_ahead_initialisations(*, end_times: np.ndarray) -> np.ndarray:
    """Return the initialisation whose 12-to-36-hour window of valid times contains each time.

    The window of the initialisation at 09:00 UTC on day `d` holds the half-hour ends `e` with
    `d + 21 h < e <= d + 45 h`, and the windows of consecutive days tile the time axis.

    Args:
        end_times: Valid times as the half-hour ends, `datetime64[us]`.

    Returns:
        `datetime64[us]` initialisation times, one per valid time.
    """
    offset = np.timedelta64(INITIALISATION_HOUR_UTC + FIRST_LEAD_HOURS, "h")
    one_microsecond = np.timedelta64(1, "us")
    day = (end_times - offset - one_microsecond).astype("datetime64[D]")
    return day.astype("datetime64[us]") + np.timedelta64(INITIALISATION_HOUR_UTC, "h")


@dataclass(frozen=True)
class CorrectedProblem:
    """The normal equations of the joint loss.

    Attributes:
        gram: `Z' Z`, shape `(p, p)`, float64, with `Z` the corrected design matrix of the cases.
        rhs: `Z' z`, shape `(p,)`, with `z` the corrected target.
        target_sum_squares: `z' z`, the loss at zero coefficients.
        n_cases: The number of cases in the loss.
    """

    gram: torch.Tensor
    rhs: torch.Tensor
    target_sum_squares: float
    n_cases: int


def corrected_normal_equations(
    *,
    columns: np.ndarray,
    target: np.ndarray,
    rows: np.ndarray,
    band: np.ndarray,
    case_target: np.ndarray,
    case_cutoff: np.ndarray,
    hyper: np.ndarray,
    device: str | None = None,
) -> CorrectedProblem:
    """Build the Gram matrix and right-hand side of the loss after the filter's correction.

    Args:
        columns: The design matrix of the stream rows, shape `(n, p)`.
        target: The target of the stream rows, shape `(n,)`.
        rows: The grid positions of the stream rows, strictly ascending. The filter reads only
            these positions.
        band: The hour band of every grid position, shape `(T,)` with `T > rows[-1]`.
        case_target: The grid position of each case's valid time; every one is in `rows`.
        case_cutoff: The grid position of the newest half-hour each case may read; -1 for none.
        hyper: The filter's unconstrained settings (log `q`, logit `phi`, log `sigma_z`, log
            `sigma_eps`): shape `(4,)`, or `(B, 4)` with one row per hour band.
        device: A torch device name, or `None` for CUDA when available, otherwise the CPU.

    Returns:
        The normal equations.

    Raises:
        ValueError: If a case's target is not a stream row, or the shapes disagree.
    """
    if columns.shape[0] != len(rows) or target.shape != rows.shape:
        msg = "columns, target, and rows must have one entry per stream row"
        raise ValueError(msg)
    first, last = int(rows[0]), int(rows[-1])
    local = rows - first
    n_local = last - first + 1
    streams = np.full((columns.shape[1] + 1, n_local), np.nan)
    streams[:-1, local] = columns.T
    streams[-1, local] = target
    position = np.full(n_local, -1, dtype=np.int64)
    position[local] = np.arange(len(rows))
    case_row = position[case_target - first]
    if (case_row < 0).any():
        msg = "every case's target must be a stream row"
        raise ValueError(msg)
    local_band = band[first : last + 1]
    trace = run_kalman_grid(
        residuals=streams,
        band=local_band,
        hyper=np.asarray(hyper, dtype=np.float64)[None],
        n_bands=N_BANDS,
        device=device,
    )
    correction = correction_at_cutoffs(
        trace=trace,
        cutoff=case_cutoff - first,
        target=case_target - first,
        band=local_band[case_target - first],
    )[:, 0, :].cpu()
    corrected_columns = torch.as_tensor(columns[case_row], dtype=torch.float64) - correction[:-1].T
    corrected_target = torch.as_tensor(target[case_row], dtype=torch.float64) - correction[-1]
    return CorrectedProblem(
        gram=corrected_columns.T @ corrected_columns,
        rhs=corrected_columns.T @ corrected_target,
        target_sum_squares=float(corrected_target @ corrected_target),
        n_cases=len(case_target),
    )


def loss_cases(
    *, cache: FeatureCache, index: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the grid positions of the loss's targets and of the newest reading each may use.

    A training row is a target when it is a stream row, ends more than `BURN_IN_DAYS` days after the
    first stream row, and has a usable reading before its initialisation's cutoff.

    Args:
        cache: The cache.
        index: The training rows.
        rows: The stream rows (see `stream_rows`).

    Returns:
        `(targets, cutoffs)`, both grid positions, one entry per case.
    """
    burn_in_end = cache.end_times[rows[0]] + np.timedelta64(BURN_IN_DAYS, "D")
    targets = index[np.isin(index, rows) & (cache.end_times[index] > burn_in_end)]
    initialisation = day_ahead_initialisations(end_times=cache.end_times[targets])
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times, initialisation=initialisation, delay_hours=DEFAULT_DELAY_HOURS
    )
    usable = (cutoff >= rows[0]) & (cutoff < targets)
    return targets[usable], cutoff[usable]


def stream_rows(
    *, cache: FeatureCache, site: str, cut: datetime, start: np.datetime64
) -> np.ndarray:
    """Return the grid rows the filter reads: finite reading and inputs, after `start`, up to `cut`.

    These are the rows `studies.dp_ens.proxy_residual_stream` makes a residual for, and include
    the rows that the fit masks out of its loss.

    Args:
        cache: The cache.
        site: The substation's label.
        cut: The fit's cut.
        start: The window's start; only later half-hour ends count.

    Returns:
        Sorted grid positions.
    """
    features = cache.sites[site]
    end = cache.end_times
    keep = (
        (end > start)
        & (end <= naive_utc(moment=cut))
        & np.isfinite(features.smoothed_temperature_c).all(axis=1)
        & np.isfinite(features.pv_unit_shape)
        & np.isfinite(features.temperature_c)
        & np.isfinite(features.y)
    )
    return np.flatnonzero(keep)


def fit_joint_gam(
    *,
    cache: FeatureCache,
    site: str,
    spec: GamSpec,
    cut: datetime,
    window: TrainingWindow,
    season_years: Sequence[int],
    hyper: np.ndarray,
    pinned: PinnedCapacity | None = None,
    anchor: GamFit | None = None,
    ridges: Sequence[float] = (0.0,),
    device: str | None = None,
) -> list[GamFit]:
    """Fit the convex model of one substation through the Kalman filter.

    Args:
        cache: The cache.
        site: The substation's label.
        spec: The penalties and drift form.
        cut: The cut; no row ends after it.
        window: The window rule.
        season_years: The season-years with a capacity increment.
        hyper: The filter's unconstrained settings, shape `(4,)` or `(B, 4)`.
        pinned: The pinned capacity built for `cut`; required exactly when the spec pins it.
        anchor: The fit whose coefficients the ridge pulls towards, with the same layout.
        ridges: The ridge weights `lambda` against the plain Gram matrix, one fit each; 0 for
            none. Every weight shares one pass of the filter over the design columns.
        device: Where the filter and the solve run.

    Returns:
        One fitted model per ridge weight, each with the same layout and fields as
        `studies.dp_gam.fit_gam`'s.

    Raises:
        ValueError: If a ridge is asked for without an anchor, or the anchor's layout differs.
    """
    if any(ridge > 0.0 for ridge in ridges) and anchor is None:
        msg = "a ridge needs an anchor fit"
        raise ValueError(msg)
    layout, index = prepare_fit(
        cache=cache, site=site, spec=spec, cut=cut, window=window, season_years=season_years
    )
    if anchor is not None and anchor.layout.n_columns != layout.n_columns:
        msg = "the anchor's layout differs from the joint fit's"
        raise ValueError(msg)
    features = cache.sites[site]
    start = cache.end_times[index[0]] - np.timedelta64(30, "m")
    rows = stream_rows(cache=cache, site=site, cut=cut, start=start)
    columns, target = stream_design(cache=cache, site=site, layout=layout, rows=rows, pinned=pinned)
    case_target, case_cutoff = loss_cases(cache=cache, index=index, rows=rows)
    problem = corrected_normal_equations(
        columns=columns,
        target=target,
        rows=rows,
        band=cache.calendar.band,
        case_target=case_target,
        case_cutoff=case_cutoff,
        hyper=hyper,
        device=device,
    )
    matrix_np, cost_np = penalty_matrix(layout=layout, n_rows=len(index))
    base = problem.gram + torch.as_tensor(matrix_np, dtype=torch.float64)
    base_rhs = problem.rhs - 0.5 * torch.as_tensor(cost_np, dtype=torch.float64)
    matrices, right_hand_sides = [], []
    if anchor is not None:
        plain, _ = accumulate_gram(
            cache=cache, features=features, layout=layout, index=index, pinned=pinned
        )
        pull = plain[0] @ torch.as_tensor(anchor.theta[0], dtype=torch.float64)
    for ridge in ridges:
        if ridge > 0.0:
            matrices.append(base + ridge * plain[0])
            right_hand_sides.append(base_rhs + ridge * pull)
        else:
            matrices.append(base)
            right_hand_sides.append(base_rhs)
    where = device or "cpu"
    theta, violation = solve_bounded_quadratic(
        matrix=torch.stack(matrices).to(where),
        rhs=torch.stack(right_hand_sides).to(where),
        constrained=torch.as_tensor(layout.constrained),
        max_iterations=NODE_MAX_ITERATIONS if spec.node is not None else DEFAULT_MAX_ITERATIONS,
    )
    last_season_year = last_signal_season_year(
        cache=cache, site=site, index=index, cut=cut, pinned=pinned
    )
    return [
        GamFit(
            layout=layout,
            theta=theta[member : member + 1].cpu().numpy(),
            cut=cut,
            n_rows=len(index),
            site=site,
            last_season_year=last_season_year,
            violation=violation[member : member + 1].cpu().numpy(),
            pinned=pinned,
        )
        for member in range(len(ridges))
    ]


def stream_design(
    *,
    cache: FeatureCache,
    site: str,
    layout: GamLayout,
    rows: np.ndarray,
    pinned: PinnedCapacity | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the design matrix and the target of the stream rows, as the fit builds them.

    Args:
        cache: The cache.
        site: The substation's label.
        layout: The columns.
        rows: The stream rows (see `stream_rows`).
        pinned: The pinned capacity, whose solar output is added to the target; required exactly
            when the layout pins the capacity.

    Returns:
        The design matrix of shape `(len(rows), p)` and the target of shape `(len(rows),)`.
    """
    features = cache.sites[site]
    designs, targets = [], []
    for chunk in row_chunks(index=rows):
        chunk_rows = rows[chunk]
        designs.append(
            design_matrix(cache=cache, features=features, layout=layout, index=chunk_rows)
        )
        target = features.y[chunk_rows]
        if pinned is not None:
            target = target + pinned_solar_points(
                cache=cache,
                features=features,
                pinned=pinned,
                index=chunk_rows,
                clip=layout.spec.clip,
            )
        targets.append(target)
    return np.concatenate(designs), np.concatenate(targets)
