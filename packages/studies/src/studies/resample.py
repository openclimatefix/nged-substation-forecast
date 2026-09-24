"""Upsample a weather forecast from its 3- or 6-hourly steps to hourly values, four ways.

A forecast series here is one row of a 2-D array: one row per series (a generator, a run and an
ensemble member, say), one column per forecast step, every row sharing one grid of step leads. The
target hours share a grid too. Every function works on whole arrays at once, so upsampling millions
of series costs a few array operations.

**Which way is right depends on what the value at a step describes**, which [NWP variable
conventions](https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/)
sets out per variable:

- `interpolate_linear` treats every value as instantaneous at its step, which is right for
  temperature and wind speed, and wrong for radiation, whose value is a mean over the step before
  its stamp.
- `interpolate_pchip` is a shape-preserving cubic: it passes through every value without
  overshooting between them, which keeps more of a daily temperature cycle than a straight line
  between 6-hourly samples does.
- `wind_components` and `wind_polar` convert a wind to its eastward and northward components and
  back, so a wind can be interpolated as a vector rather than across the 0°/360° wrap of its
  direction.
- `clear_sky_index_resample` rebuilds hourly radiation from period means by interpolating the
  clear-sky index rather than the radiation.
"""

from typing import Final

import numpy as np
from scipy.interpolate import PchipInterpolator

DEFAULT_DAYLIGHT_FLOOR_W_M2: Final[float] = 50.0
"""The least clear-sky mean over a step for that step to anchor a clear-sky index.

Below this the index divides a small, noisy flux by a small clear-sky flux, and a few watts of
difference become an index far above 1.
"""


def interpolate_linear(*, values: np.ndarray, x: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Interpolate every row linearly between its steps, holding the end values beyond them.

    Args:
        values: Shape (n_series, n_steps), each row's value at each step, NaN where missing.
        x: Shape (n_steps,), each step's position, strictly increasing.
        targets: Shape (n_targets,), the positions to evaluate at.

    Returns:
        Shape (n_series, n_targets). A target between two steps is NaN if either step is.
    """
    clamped = np.clip(targets, x[0], x[-1])
    right = np.clip(np.searchsorted(x, clamped, side="right"), 1, len(x) - 1)
    left = right - 1
    weight = (clamped - x[left]) / (x[right] - x[left])
    return values[:, left] * (1.0 - weight) + values[:, right] * weight


def interpolate_pchip(*, values: np.ndarray, x: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Interpolate every row with a shape-preserving cubic, holding the end values beyond them.

    Args:
        values: Shape (n_series, n_steps), with no NaN.
        x: Shape (n_steps,), each step's position, strictly increasing.
        targets: Shape (n_targets,), the positions to evaluate at.

    Returns:
        Shape (n_series, n_targets).
    """
    interpolator = PchipInterpolator(x, values, axis=1, extrapolate=False)
    return interpolator(np.clip(targets, x[0], x[-1]))


def wind_components(
    *, speed: np.ndarray, direction_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the eastward and northward components of a wind given as speed and direction.

    The direction is meteorological: the bearing the wind blows from, clockwise from north.

    Args:
        speed: The wind speed.
        direction_deg: The bearing the wind blows from, in degrees.

    Returns:
        The eastward component `u` and the northward component `v`, in the speed's unit.
    """
    radians = np.radians(direction_deg)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def wind_polar(*, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a wind's speed and meteorological direction from its components.

    Args:
        u: The eastward component.
        v: The northward component.

    Returns:
        The speed, and the bearing the wind blows from in degrees, in [0, 360).
    """
    return np.hypot(u, v), np.mod(np.degrees(np.arctan2(-u, -v)), 360.0)


def _forward_fill(values: np.ndarray) -> np.ndarray:
    """Carry each row's last non-NaN value forward along the row.

    Args:
        values: Shape (n_series, n_steps).

    Returns:
        The same shape, NaN only before a row's first non-NaN value.
    """
    positions = np.where(~np.isnan(values), np.arange(values.shape[1])[None, :], -1)
    np.maximum.accumulate(positions, axis=1, out=positions)
    filled = np.take_along_axis(values, np.clip(positions, 0, None), axis=1)
    return np.where(positions >= 0, filled, np.nan)


def hold_flat_outside_daylight(*, index: np.ndarray, morning: np.ndarray) -> np.ndarray:
    """Fill each step that anchors no index with the nearest daylight step's index on its side.

    A step before noon takes the next daylight index, so the index at dawn is the morning's. A
    step after noon takes the last daylight index, so the index at dusk is the evening's. A step
    with no daylight step on its own side takes the one on the other side.

    Args:
        index: Shape (n_series, n_steps), NaN at every step that anchors no index.
        morning: Shape (n_steps,), whether each step's midpoint falls before noon.

    Returns:
        The same shape, NaN only in a row with no daylight step at all.
    """
    earlier = _forward_fill(index)
    later = _forward_fill(index[:, ::-1])[:, ::-1]
    on_side = np.where(morning[None, :], later, earlier)
    other_side = np.where(morning[None, :], earlier, later)
    fill = np.where(np.isnan(on_side), other_side, on_side)
    return np.where(np.isnan(index), fill, index)


def clear_sky_index_resample(
    *,
    values: np.ndarray,
    step_clear_sky: np.ndarray,
    step_midpoints: np.ndarray,
    morning: np.ndarray,
    target_clear_sky: np.ndarray,
    target_midpoints: np.ndarray,
    daylight_floor_w_m2: float = DEFAULT_DAYLIGHT_FLOOR_W_M2,
) -> np.ndarray:
    """Rebuild hourly radiation from period means through the clear-sky index.

    The four requirements [NWP variable
    conventions](https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/)
    sets out for a period-ending field, in the order they are applied:

    1. The index is each step's mean radiation over the clear-sky mean over the same step, not
       over the clear-sky value at the step's stamp.
    2. The index sits at the step's midpoint, where the mean it came from is centred.
    3. Only a step whose clear-sky mean reaches `daylight_floor_w_m2` anchors an index, and every
       other step holds the nearest daylight index flat (`hold_flat_outside_daylight`).
    4. The index, interpolated linearly to each target's midpoint, is multiplied by the clear-sky
       mean over the target's own period, so the result is a mean over that period like the
       power it is compared with.

    A clear sky whose every step reads a fixed fraction of the clear-sky mean therefore comes back
    as that fraction of the clear-sky mean over every target period.

    Args:
        values: Shape (n_series, n_steps), each step's mean radiation over the step before it.
        step_clear_sky: Shape (n_series, n_steps), the clear-sky mean over each step.
        step_midpoints: Shape (n_steps,), each step's midpoint, strictly increasing.
        morning: Shape (n_steps,), whether each step's midpoint falls before noon.
        target_clear_sky: Shape (n_series, n_targets), the clear-sky mean over each target period.
        target_midpoints: Shape (n_targets,), each target period's midpoint.
        daylight_floor_w_m2: The least clear-sky mean that anchors an index.

    Returns:
        Shape (n_series, n_targets), the mean radiation over each target period: zero where the
        target's clear-sky mean is zero, and NaN where a row has no daylight step.
    """
    daylight = step_clear_sky >= daylight_floor_w_m2
    index = np.where(daylight, values / np.where(daylight, step_clear_sky, 1.0), np.nan)
    filled = hold_flat_outside_daylight(index=index, morning=morning)
    interpolated = interpolate_linear(values=filled, x=step_midpoints, targets=target_midpoints)
    return np.where(target_clear_sky > 0.0, interpolated * target_clear_sky, 0.0)


def step_means(
    *, hourly: np.ndarray, first_hour: int, step_leads: np.ndarray, step_widths: np.ndarray
) -> np.ndarray:
    """Average hourly period means over each forecast step.

    Args:
        hourly: Shape (n_series, n_hours), the mean over each hour, column `k` being the hour that
            ends at lead `first_hour + k`.
        first_hour: The lead at which the first column's hour ends.
        step_leads: Shape (n_steps,), each step's end.
        step_widths: Shape (n_steps,), each step's width in whole hours.

    Returns:
        Shape (n_series, n_steps), the mean over the hours ending after `lead - width` and at or
        before `lead`.

    Raises:
        ValueError: If a step reaches outside the hours given.
    """
    starts = (step_leads - step_widths + 1 - first_hour).astype(int)
    ends = (step_leads + 1 - first_hour).astype(int)
    if starts.min() < 0 or ends.max() > hourly.shape[1]:
        msg = "a step reaches outside the hours given"
        raise ValueError(msg)
    return np.stack(
        [hourly[:, start:end].mean(axis=1) for start, end in zip(starts, ends, strict=True)],
        axis=1,
    )


def rescale_to_step_means(
    *,
    values: np.ndarray,
    target_leads: np.ndarray,
    step_values: np.ndarray,
    step_leads: np.ndarray,
    step_widths: np.ndarray,
) -> np.ndarray:
    """Scale the hours inside each step so their mean equals the step's own mean.

    A clear-sky-index resample is close to conserving each step's energy but not exact, because the
    index is interpolated between steps. Scaling each step's hours by the step's mean over their
    mean makes it exact. A step whose hours sum to zero is left at zero.

    Args:
        values: Shape (n_series, n_targets), hourly means, each hour ending at its target lead.
        target_leads: Shape (n_targets,), each hour's end, whole hours.
        step_values: Shape (n_series, n_steps), each step's mean over the step.
        step_leads: Shape (n_steps,), each step's end.
        step_widths: Shape (n_steps,), each step's width in whole hours.

    Returns:
        `values`, each step's hours scaled.

    Raises:
        ValueError: If a target hour falls in no step, or a step's hours are not all targets.
    """
    scaled = values.copy()
    covered = np.zeros(len(target_leads), dtype=bool)
    for index, (lead, width) in enumerate(zip(step_leads, step_widths, strict=True)):
        inside = (target_leads > lead - width) & (target_leads <= lead)
        if not inside.any():
            continue
        if inside.sum() != width:
            msg = (
                f"only {inside.sum()} of the {width} hours of the step ending at {lead} are targets"
            )
            raise ValueError(msg)
        covered |= inside
        hourly_mean = values[:, inside].mean(axis=1)
        factor = np.where(
            hourly_mean > 0.0,
            step_values[:, index] / np.where(hourly_mean > 0.0, hourly_mean, 1.0),
            1.0,
        )
        scaled[:, inside] = values[:, inside] * factor[:, None]
    if not covered.all():
        msg = "a target hour falls in no step"
        raise ValueError(msg)
    return scaled


def coarsen_to_six_hourly(
    *,
    leads: np.ndarray,
    values: dict[str, np.ndarray],
    period_means: frozenset[str],
    last_three_hourly_lead: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Emulate a 6-hourly forecast from one published on 3-hour steps to a lead, 6-hour beyond.

    The steps at multiples of 6 hours are kept. An instantaneous field keeps its value at the step.
    A period-mean field on the 3-hour part of the grid takes the mean of its value and the one 3
    hours before, which is the 6-hour mean exactly, and a step whose earlier half is not given is
    dropped.

    Args:
        leads: Shape (n_steps,), each step's lead, ascending.
        values: Each field's values, shape (n_series, n_steps).
        period_means: The fields that are a mean over the step ending at the lead.
        last_three_hourly_lead: The last lead on the 3-hour part of the grid.

    Returns:
        The kept leads and each field's values at them.
    """
    position = {int(lead): index for index, lead in enumerate(leads)}
    kept = [
        index
        for index, lead in enumerate(leads)
        if int(lead) % 6 == 0
        and (int(lead) > last_three_hourly_lead or int(lead) - 3 in position or not period_means)
    ]
    coarse: dict[str, np.ndarray] = {}
    for name, array in values.items():
        columns = []
        for index in kept:
            lead = int(leads[index])
            if name in period_means and lead <= last_three_hourly_lead:
                columns.append((array[:, index] + array[:, position[lead - 3]]) / 2.0)
            else:
                columns.append(array[:, index])
        coarse[name] = np.stack(columns, axis=1)
    return leads[kept], coarse
