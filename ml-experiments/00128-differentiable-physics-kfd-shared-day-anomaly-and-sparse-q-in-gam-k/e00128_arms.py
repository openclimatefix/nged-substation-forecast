"""The arms of experiment 128: which hyperparameters each arm's filter uses and how d is read.

| Arm | Filter constants | Day-level state in the forecast | Spread |
|---|---|---|---|
| GK | experiment 111's GAM-K (d off) | none | experiment 111's stored spread |
| GK-sp | GAM-K's | none | fitted by this experiment's procedure |
| KD | KD, one shared level variance | own d | fitted |
| KS | KS, sparse level variances, d off | none | fitted |
| KDS | KDS, sparse level variances, d on | own d | fitted |
| KDS-shuf | KDS's | d taken from another day of the same month | fitted |
| KDS-nod | KDS's | removed from the forecast only | fitted |
| KDS-1n, KDS-4n | KDS with thresholds of 1 and 4 nats | own d | fitted |
| KDS-floor | KDS with every `q_b` at least 1e-4 | own d | fitted |
| KDS-da | KDS with KD's other values fitted by the day-ahead likelihood | own d | fitted |

The filter itself is `e00128_filter`; the fitted constants come from `e00128_fit`.
"""

from dataclasses import dataclass
from typing import Final, Literal

DayModeType = Literal["own", "none", "shuffled"]
"""How a forecast reads d: its own filtered value, left out, or the value of another day."""


@dataclass(frozen=True)
class ArmSpec:
    """One arm.

    Attributes:
        name: The arm's name.
        params_arm: The fitted arm whose constants the filter runs with, `None` for GAM-K's.
        day_mode: How the forecast reads d.
        own_spread: Whether the arm's spread is fitted on the tuning months (not the stored one).
    """

    name: str
    params_arm: str | None
    day_mode: DayModeType
    own_spread: bool = True


ARM_SPECS: Final[dict[str, ArmSpec]] = {
    "GK": ArmSpec("GK", None, "none", own_spread=False),
    "GK-sp": ArmSpec("GK-sp", None, "none"),
    "KD": ArmSpec("KD", "KD", "own"),
    "KS": ArmSpec("KS", "KS", "none"),
    "KDS": ArmSpec("KDS", "KDS", "own"),
    "KDS-shuf": ArmSpec("KDS-shuf", "KDS", "shuffled"),
    "KDS-nod": ArmSpec("KDS-nod", "KDS", "none"),
    "KDS-1n": ArmSpec("KDS-1n", "KDS-1n", "own"),
    "KDS-4n": ArmSpec("KDS-4n", "KDS-4n", "own"),
    "KDS-floor": ArmSpec("KDS-floor", "KDS-floor", "own"),
    "KDS-da": ArmSpec("KDS-da", "KDS-da", "own"),
}
FORECAST_ARMS: Final[tuple[str, ...]] = tuple(ARM_SPECS)
"""Every arm the experiment forecasts."""
OWN_SPREAD_ARMS: Final[tuple[str, ...]] = tuple(
    name for name, spec in ARM_SPECS.items() if spec.own_spread
)
"""Arms whose spread is fitted on the tuning months."""
PLANNED_ARMS: Final[tuple[str, ...]] = ("GK", "KD", "KS", "KDS")
"""The four arms of the gate and of the planned contrasts."""
