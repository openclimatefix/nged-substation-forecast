"""A level step per hour band: experiment 252's T-all design plus four unpenalised columns.

The step columns are one per hour band, equal to 1 on rows whose valid time is on or after
`step_from` in that band and 0 elsewhere. They carry a ridge of 1e-6 (numerical conditioning only)
and no sign bound, so each band's level can move freely at the step. `install` replaces
`e00252_design.arm_matrix` and `arm_penalty` with versions that append the columns for a
`StepArm`, so `e00252_tall.predict_tall` and `fit_arm` pick them up unchanged. Fits are pickled
against this module, not `__main__`.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import e00252_design as design
import numpy as np
from studies.dp_features import naive_utc
from studies.dp_gam import N_BANDS

STEP_RIDGE: Final[float] = 1e-6

_original_matrix = design.arm_matrix
_original_penalty = design.arm_penalty


@dataclass(frozen=True)
class StepArm(design.ArmSpec):
    """T-all plus an optional level step per hour band.

    Attributes:
        step_from: The instant the level step starts, or `None` for no step columns.
    """

    step_from: datetime | None = None


def step_arm(*, base: design.ArmSpec, step_from: datetime | None) -> StepArm:
    """Return `base`'s settings as a `StepArm`."""
    return StepArm(
        name=base.name,  # ty: ignore[unknown-argument]
        heating_factors=base.heating_factors,  # ty: ignore[unknown-argument]
        lighting_factors=base.lighting_factors,  # ty: ignore[unknown-argument]
        penalty_weight=base.penalty_weight,  # ty: ignore[unknown-argument]
        hinge_threshold_c=base.hinge_threshold_c,  # ty: ignore[unknown-argument]
        step_from=step_from,
    )


def _matrix(*, cache, layout, site, index, x0, arm):  # noqa: ANN001, ANN202
    x = _original_matrix(cache=cache, layout=layout, site=site, index=index, x0=x0, arm=arm)
    step_from = getattr(arm, "step_from", None)
    if step_from is None:
        return x
    after = cache.end_times[index] >= naive_utc(moment=step_from)
    band = cache.calendar.band[index]
    columns = (after[:, None] & (band[:, None] == np.arange(N_BANDS)[None, :])).astype(np.float64)
    return np.hstack([x, columns])


def _penalty(*, layout, arm, n_rows):  # noqa: ANN001, ANN202
    base = _original_penalty(layout=layout, arm=arm, n_rows=n_rows)
    if getattr(arm, "step_from", None) is None:
        return base
    size = base.matrix.shape[0]
    matrix = np.zeros((size + N_BANDS, size + N_BANDS))
    matrix[:size, :size] = base.matrix
    matrix[size:, size:] = STEP_RIDGE * np.eye(N_BANDS)
    constrained = np.concatenate([base.constrained, np.zeros(N_BANDS, dtype=bool)])
    return design.ArmPenalty(
        matrix=matrix, constrained=constrained, n_new_columns=base.n_new_columns + N_BANDS
    )


def install() -> None:
    """Make `e00252_design` build the step columns for a `StepArm`."""
    design.arm_matrix = _matrix
    design.arm_penalty = _penalty
