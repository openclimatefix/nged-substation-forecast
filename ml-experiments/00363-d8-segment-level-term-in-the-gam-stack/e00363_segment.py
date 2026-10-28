"""A penalised level term for each stretch between detected steps, appended to the T-all design.

`install` must run after experiment 354's `step.install`, because it wraps whichever `arm_matrix`
and `arm_penalty` are installed at that moment. A `SegmentArm` carries the stretch boundaries
(instants, all at or before the fit's cut), so the same design is rebuilt at prediction time and
every row after the cut takes the level of the latest stretch. The columns carry a ridge of
`penalty_scale * lambda`, where `penalty_scale` is the interaction penalty's own scale
(`penalty_weight * n_rows / SITE_YEAR_ROWS`), so lambda 1 means "as strong as the weather terms'
interaction ridge". Fits are pickled against this module, not `__main__`.
"""

from dataclasses import dataclass
from typing import Final

import e00252_design as design  # ty: ignore[unresolved-import]
import e00354_step as step  # ty: ignore[unresolved-import]
import numpy as np
from studies.dp_gam import SITE_YEAR_ROWS

NO_SEGMENT_RIDGE: Final[float] = 1e-6

_before: dict = {}


@dataclass(frozen=True)
class SegmentArm(step.StepArm):
    """T-all plus the level step, plus one level per stretch between detected steps.

    Attributes:
        boundaries: Sorted stretch boundaries as `datetime64[ns]` integers (naive UTC); stretch
            `k` runs from boundary `k-1` (inclusive) to boundary `k` (exclusive).
        segment_lambda: The ridge multiplier of the stretch columns.
        last_in_fall: Whether the latest stretch before the cut is a detected fall in power.
        fall_share: Share of training rows inside detected falls.
        fall_depth_points: Median depth of the detected falls, in points of effective capacity.
    """

    boundaries: tuple[int, ...] = ()
    segment_lambda: float = 1.0
    last_in_fall: bool = False
    fall_share: float = 0.0
    fall_depth_points: float = 0.0


def stretch_of(*, end_times: np.ndarray, boundaries: tuple[int, ...]) -> np.ndarray:
    """Return the stretch number of each row: the count of boundaries at or before its end time."""
    stamps = np.asarray(end_times, dtype="datetime64[ns]").astype(np.int64)
    return np.searchsorted(np.asarray(boundaries, dtype=np.int64), stamps, side="right")


def _matrix(*, cache, layout, site, index, x0, arm):  # noqa: ANN001, ANN202
    x = _before["matrix"](cache=cache, layout=layout, site=site, index=index, x0=x0, arm=arm)
    if not isinstance(arm, SegmentArm) or arm.segment_lambda <= 0.0:
        return x
    stretch = stretch_of(end_times=cache.end_times[index], boundaries=arm.boundaries)
    columns = np.zeros((len(index), len(arm.boundaries) + 1))
    columns[np.arange(len(index)), stretch] = 1.0
    return np.hstack([x, columns])


def _penalty(*, layout, arm, n_rows):  # noqa: ANN001, ANN202
    base = _before["penalty"](layout=layout, arm=arm, n_rows=n_rows)
    if not isinstance(arm, SegmentArm) or arm.segment_lambda <= 0.0:
        return base
    size, extra = base.matrix.shape[0], len(arm.boundaries) + 1
    matrix = np.zeros((size + extra, size + extra))
    matrix[:size, :size] = base.matrix
    scale = arm.penalty_weight * n_rows / SITE_YEAR_ROWS
    matrix[size:, size:] = max(arm.segment_lambda * scale, NO_SEGMENT_RIDGE) * np.eye(extra)
    constrained = np.concatenate([base.constrained, np.zeros(extra, dtype=bool)])
    return design.ArmPenalty(
        matrix=matrix, constrained=constrained, n_new_columns=base.n_new_columns + extra
    )


def install() -> None:
    """Make `e00252_design` build the stretch columns for a `SegmentArm`."""
    _before["matrix"], _before["penalty"] = design.arm_matrix, design.arm_penalty
    design.arm_matrix = _matrix
    design.arm_penalty = _penalty
