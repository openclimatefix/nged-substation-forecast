"""The four arms: RO-NC as in experiment 130 (B) and three ways to catch weaker, slower onsets.

Every arm scores candidates with a logistic regression and keeps those above a threshold. The arms
differ in which candidates they consider and which features the score sees:

- B: candidates at 2.5 noise units or more on the 24-hour step (experiment 130's arm B).
- W1: candidates from 1.5 noise units, the same features.
- W2: candidates from 1.5 noise units, and feature `x3`, the step still present 1 to 2 days later.
- W3: B's candidates plus peaks of a 72-hour step at 3.0 noise units or more, and feature `x4`,
  the 72-hour step at the candidate in noise units along the candidate's sign.

Nothing here reads a switching log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from e00148_combined import (
    CANDIDATE_Z,
    FEATURES_PERSIST,
    FEATURES_SAME,
    FEATURES_SLOW,
    candidate_features,
    candidates,
)
from e00148_detectors import DetectorInputs, alarms
from e00148_variants import WINDOW, variant_inputs

SLOW_WINDOW: Final[int] = 144
"""Half-hours in 72 hours."""
SLOW_CANDIDATE_Z: Final[float] = 3.0
SLOW_MERGE: Final[int] = WINDOW
"""A slow peak within this many stamps of a 24-hour candidate on the same series adds nothing."""
PERSIST_GAP: Final[int] = WINDOW
"""The persistence statistic compares the day after a one-day gap with the day before."""
WEAK_CANDIDATE_Z: Final[float] = 1.5


@dataclass(frozen=True)
class ArmSpec:
    """One arm's candidate bar, extra statistics, and score features."""

    name: str
    candidate_z: float
    persistence: bool
    slow: bool
    features: tuple[str, ...]


ARMS: Final[dict[str, ArmSpec]] = {
    "B": ArmSpec("B", CANDIDATE_Z, False, False, FEATURES_SAME),
    "W1": ArmSpec("W1", WEAK_CANDIDATE_Z, False, False, FEATURES_SAME),
    "W2": ArmSpec("W2", WEAK_CANDIDATE_Z, True, False, FEATURES_PERSIST),
    "W3": ArmSpec("W3", CANDIDATE_Z, False, True, FEATURES_SLOW),
}


@dataclass(frozen=True)
class ArmInputs:
    """The step statistics an arm reads: the 24-hour step, and the persistence or slow step."""

    fast: DetectorInputs
    persist: DetectorInputs | None
    slow: DetectorInputs | None


def arm_inputs(
    *, spec: ArmSpec, residual: dict, offset: dict, scales: dict[str, dict[int, float]]
) -> ArmInputs:
    """Build the statistics `spec` needs from a (possibly injected) residual.

    `scales` maps `"X"`, `"persist"`, and `"slow"` to each series' odd-month noise unit of that
    statistic on the uninjected residual.
    """

    def build(*, scale: dict[int, float], window: int = WINDOW, gap: int = 0) -> DetectorInputs:
        return variant_inputs(
            variant="X", residual=residual, offset=offset, scale=scale, window=window, gap=gap
        )

    return ArmInputs(
        fast=build(scale=scales["X"]),
        persist=build(scale=scales["persist"], gap=PERSIST_GAP) if spec.persistence else None,
        slow=build(scale=scales["slow"], window=SLOW_WINDOW) if spec.slow else None,
    )


def _signed_z(*, inputs: DetectorInputs, table: pl.DataFrame) -> np.ndarray:
    """The statistic at each candidate in noise units along the candidate's sign, 0 if missing."""
    out = np.zeros(table.height)
    for i, (series, stamp, sign) in enumerate(table.select("series", "stamp", "sign").iter_rows()):
        value = sign * inputs.shift[series][stamp] / inputs.scale[series]
        out[i] = value if np.isfinite(value) else 0.0
    return out


def arm_candidates(*, spec: ArmSpec, inputs: ArmInputs) -> pl.DataFrame:
    """Every candidate of the arm, with the features its score reads."""
    table = candidates(inputs=inputs.fast, candidate_z=spec.candidate_z)
    if inputs.slow is not None:
        slow = alarms(inputs=inputs.slow, detector="D0", threshold=SLOW_CANDIDATE_Z)
        near: dict[int, np.ndarray] = {
            s: g["stamp"].to_numpy() for (s,), g in table.group_by("series")
        }
        extra = [
            (int(s), int(t), float(np.sign(z)))
            for s, t, z in slow.select("series", "stamp", "z").iter_rows()
            if not (np.abs(near.get(int(s), np.array([], dtype=int)) - t) <= SLOW_MERGE).any()
        ]
        table = pl.concat([table, candidate_features(inputs=inputs.fast, peaks=extra)])
        table = table.with_columns(x4=pl.Series(_signed_z(inputs=inputs.slow, table=table)))
    if inputs.persist is not None:
        table = table.with_columns(x3=pl.Series(_signed_z(inputs=inputs.persist, table=table)))
    return table
