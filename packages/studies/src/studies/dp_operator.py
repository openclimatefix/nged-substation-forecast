"""Per-row observation operators of the differentiable-physics (DP) series.

**The model predicts signed real power `P = D - PV` for every row, and an operator maps `P` to what
the meter reads.** An MW meter reads `P`. An MVA meter reads the apparent power `|S|`, which is
never negative, so it cannot show a substation that exports at midday.

- `identity` (MW substations): the reading is `P`.
- `apparent` (MVA substations): reactive power is `Q = tan(phi) * D + q0`, and the reading is
  `|S| = sqrt(P^2 + Q^2 + epsilon)`. Both `tan(phi)` and `q0` are per-substation parameters. The
  small `epsilon` keeps the gradient finite where `P` and `Q` are both zero. With `tan(phi) = 0`,
  `q0 = 0`, and the solar term gated off, `|S|` is the demand in its own units, which is the rule
  the series' early parts use for MVA substations.
- `sum_of_nodes`: a bulk supply point informed by its Primaries. It is a typed stub here.

A **solar gate** zeroes the solar term on rows where it must not train, such as the dark-row rule
for MVA substations (the rows with irradiance below 1 W/m²): the model then reads demand alone.
"""

from dataclasses import dataclass
from typing import Final, Literal

import torch

OperatorType = Literal["identity", "apparent", "sum_of_nodes"]
"""Which operator maps real power to the meter's reading."""

DEFAULT_EPSILON: Final[float] = 1e-6
"""Added under the square root of the apparent-power operator, in points squared."""


@dataclass(frozen=True)
class ApparentPowerParameters:
    """The per-row parameters of the `apparent` operator.

    Attributes:
        tan_phi: The ratio of reactive power to demand, broadcast to the rows.
        q0: The reactive power at zero demand, in the same units as the demand.
        epsilon: Added under the square root.
    """

    tan_phi: torch.Tensor
    q0: torch.Tensor
    epsilon: float = DEFAULT_EPSILON


def real_power(
    *, demand: torch.Tensor, solar: torch.Tensor, solar_gate: torch.Tensor | None = None
) -> torch.Tensor:
    """Return signed real power: demand minus the (gated) solar output.

    Args:
        demand: Demand per row.
        solar: Solar output per row, non-negative.
        solar_gate: True where the solar term applies; `None` means everywhere.

    Returns:
        `demand - solar` where the gate is true, `demand` elsewhere.
    """
    if solar_gate is None:
        return demand - solar
    return demand - torch.where(solar_gate, solar, torch.zeros_like(solar))


def apparent_power(
    *, real: torch.Tensor, demand: torch.Tensor, parameters: ApparentPowerParameters
) -> torch.Tensor:
    """Return `sqrt(P^2 + Q^2 + epsilon)` with `Q = tan(phi) * D + q0`.

    Args:
        real: Signed real power `P`.
        demand: Demand `D`, which sets the reactive power.
        parameters: `tan(phi)`, `q0`, and `epsilon`.

    Returns:
        The apparent power, non-negative.
    """
    reactive = parameters.tan_phi * demand + parameters.q0
    return torch.sqrt(real * real + reactive * reactive + parameters.epsilon)


def observe(
    *,
    kind: OperatorType,
    real: torch.Tensor,
    demand: torch.Tensor,
    parameters: ApparentPowerParameters | None = None,
) -> torch.Tensor:
    """Map real power to the meter's reading with the named operator.

    Args:
        kind: The operator.
        real: Signed real power `P`.
        demand: Demand `D`, used by `apparent`.
        parameters: The `apparent` operator's parameters.

    Returns:
        The reading.

    Raises:
        ValueError: If `apparent` is asked for without parameters.
        NotImplementedError: For `sum_of_nodes`.
    """
    if kind == "identity":
        return real
    if kind == "apparent":
        if parameters is None:
            msg = "the apparent operator needs tan_phi and q0"
            raise ValueError(msg)
        return apparent_power(real=real, demand=demand, parameters=parameters)
    msg = "the sum_of_nodes operator (a bulk supply point from its Primaries) is not built yet"
    raise NotImplementedError(msg)


def power_factor_from_tan_phi(*, tan_phi: torch.Tensor) -> torch.Tensor:
    """Return the power factor `cos(phi)` that a ratio `tan(phi)` implies."""
    return 1.0 / torch.sqrt(1.0 + tan_phi * tan_phi)


def tan_phi_from_power_factor(*, power_factor: float) -> float:
    """Return `tan(phi)` for a power factor between 0 and 1."""
    return float((1.0 - power_factor**2) ** 0.5 / power_factor)
