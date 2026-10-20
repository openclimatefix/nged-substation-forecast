"""The arms of experiment 118: which weights move, whether the anomaly is on, and the noise matrix.

| Arm | Weights that move | Anomaly | Noise matrix |
|---|---|---|---|
| GK | level (4) | yes | experiment 111's |
| KE | all 14 | no | fitted |
| KE-infl | all 14 | no | fitted, sigma_eps held at sqrt(27) times KE's |
| KEZ | all 14 | yes | fitted |
| KEZ-Q0 | level (4) | yes | fitted for the level |
| KEZ-QI | all 14 | yes | one q for every weight, fitted |
| KEZ-drop-g | all but group g | yes | fitted |
| KEZ-1s | all 14 | yes | the first run's, fitted by the one-step likelihood (q bound of 2) |
| KEZ-DT | all 14 | yes | fitted on a stream without holiday-day readings |
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from studies.dp_effects_filter import (
    GROUP_SLICES,
    LEVEL,
    N_WEIGHTS,
    FilterNoise,
    FilterVariant,
    initial_sd_for,
)

INFLATION_VARIANCE: Final[float] = 27.0
"""The variance inflation of autocorrelated residuals that experiment 113 measured."""
INFLATED_SIGMA_EPS_SPREAD: Final[float] = 1e-3
"""The prior spread of `log sigma_eps` that holds KE-infl's observation noise at its start."""
DROPPABLE_GROUPS: Final[tuple[str, ...]] = ("profile", "heating", "pv", "holiday")


@dataclass(frozen=True)
class ArmConfig:
    """One arm.

    Attributes:
        name: The arm's name.
        moving: True for the weights that follow a random walk, shape (14,).
        use_anomaly: Whether the anomaly states are in use.
        shared_q: Whether one q is fitted for every moving weight.
        inflated: Whether the observation noise is held at the inflated value of KE.
        one_step: Whether the noise was fitted by the one-step likelihood (the first run, kept as an
            arm) and not by the lead-matched likelihood.
        skip_holidays: Whether readings on holiday days are left out of the filter and the level
            and anomaly corrections are zero on holiday target days (the day-type-aware arm).
    """

    name: str
    moving: np.ndarray
    use_anomaly: bool
    shared_q: bool = False
    inflated: bool = False
    skip_holidays: bool = False
    one_step: bool = False

    @property
    def variant(self) -> FilterVariant:
        """The arm as the module's `FilterVariant`."""
        return FilterVariant(
            moving=self.moving, use_anomaly=self.use_anomaly, shared_q=self.shared_q
        )


def _moving_except(*, group: str | None) -> np.ndarray:
    moving = np.ones(N_WEIGHTS, dtype=bool)
    if group is not None:
        moving[GROUP_SLICES[group]] = False
    return moving


LEVEL_ONLY: Final[np.ndarray] = np.isin(np.arange(N_WEIGHTS), np.arange(LEVEL.start, LEVEL.stop))
FITTED_ARMS: Final[dict[str, ArmConfig]] = {
    "KE": ArmConfig("KE", _moving_except(group=None), use_anomaly=False),
    "KE-infl": ArmConfig("KE-infl", _moving_except(group=None), use_anomaly=False, inflated=True),
    "KEZ": ArmConfig("KEZ", _moving_except(group=None), use_anomaly=True),
    "KEZ-Q0": ArmConfig("KEZ-Q0", LEVEL_ONLY, use_anomaly=True),
    "KEZ-QI": ArmConfig("KEZ-QI", _moving_except(group=None), use_anomaly=True, shared_q=True),
    "KEZ-1s": ArmConfig("KEZ-1s", _moving_except(group=None), use_anomaly=True, one_step=True),
    "KEZ-DT": ArmConfig("KEZ-DT", _moving_except(group=None), use_anomaly=True, skip_holidays=True),
    **{
        f"KEZ-drop-{group}": ArmConfig(
            f"KEZ-drop-{group}", _moving_except(group=group), use_anomaly=True
        )
        for group in DROPPABLE_GROUPS
    },
}
ALL_ARM_NAMES: Final[tuple[str, ...]] = ("GK", *FITTED_ARMS)
"""GK first, then every fitted arm."""


def gk_noise(*, hyper: list[float], scale: np.ndarray) -> FilterNoise:
    """Return GAM-K's noise as a `FilterNoise` with only the four level weights moving.

    Args:
        hyper: Experiment 111's four unconstrained hyperparameters of the site: log q, logit phi,
            log sigma_z, and log sigma_eps.
        scale: The regressor scales, shape (14,).

    Returns:
        The noise; the level's q is the same for every hour band.
    """
    log_q, logit_phi, log_sigma_z, log_sigma_eps = hyper
    q = np.where(LEVEL_ONLY, np.exp(log_q), 0.0)
    return FilterNoise(
        q=q,
        phi=float(1.0 / (1.0 + np.exp(-logit_phi))),
        sigma_z=float(np.exp(log_sigma_z)),
        sigma_eps=float(np.exp(log_sigma_eps)),
        initial_sd=initial_sd_for(scale=scale, moving=LEVEL_ONLY),
        use_anomaly=True,
    )
