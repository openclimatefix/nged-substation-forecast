"""A differentiable state-space battery estimator, fitted by gradient descent on a GPU or a CPU.

Each *unit* is one battery (or one fleet that moves as one) with a power `P` in MW. Units belong to
*classes*, and the units of one class share a usable duration `d` (hours at full power) and a
round-trip efficiency `eta`. A unit's *policy* is a pair of signals in [0, 1] that say how hard it
tries to charge and to discharge at each half-hour:

- A *stack unit* (a price taker: the merchant battery, or the Agile tariff) follows the dispatch of
  a day-by-day linear programme on its price. The dispatch is interpolated, bilinearly in
  `(log d, eta)` and linearly in the cycle cap (1 or 2 cycles a day, mixed by a learned weight),
  between precomputed schedules on a fine grid of durations and efficiencies. The dispatch of a
  linear programme is piecewise linear in the energy capacity, so the interpolation is exact within
  each piece.
- A *window unit* follows a fixed daily charge window and discharge window (a fixed tariff, or the
  distribution-charge red band). Its signals are known indicators.

The policy is an intent; the state of charge `s`, a fraction of the usable energy, enforces
physics through the recurrence

```text
discharge_t = softmin(policy_discharge_t, s_t * d * sqrt(eta) / 0.5h)
charge_t    = softmin(policy_charge_t, (1 - s_t) * d / (0.5h * sqrt(eta)))
s_t+1       = s_t + 0.5h / d * (sqrt(eta) * charge_t - discharge_t / sqrt(eta))
export_t    = P * (discharge_t - charge_t)
```

The softmin is a smooth minimum, so a battery never discharges more than it holds or charges more
than it has room for, and the gradient with respect to `d` and `eta` exists. The state of charge is
measured in usable energy, so the state-of-charge limits of a physical battery do not appear: `d` is
the usable duration, as in the grid estimator.

The aggregate is `y = free columns + noise - sum of exports`, where the free columns (the monthly
baseline, the solar curves, and the charge-only nuisance columns) enter linearly. They are profiled
out by projection: with `M` the projector off the free columns, the residual is `r = M (y + export)`
and the loss is `(n_eff / 2) log(r . r)` minus the log prior, in an unconstrained parameter space
(`log P`, `log d`, `logit eta`, and the logit of the cap weight). `n_eff` is the number of
equations divided by the residual's integrated autocorrelation time, which tempers the likelihood.

`Estimator.fit` runs Adam in float32 on the device, polishes the optimum by Levenberg-Marquardt
steps in float64, and returns the Laplace approximation: the Gaussian at the optimum whose
covariance is the inverse of a float64 Gauss-Newton Hessian of the loss. A fit whose Hessian is not
positive definite has no covariance and no interval.

Requires the optional `gpu` dependency group (`torch`); the CPU path in float64 needs no GPU.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

from studies.battery_recurrence import recurrence

AUTOCORRELATION_LAGS: Final[int] = 336
"""One week of half-hours: the longest lag counted in the integrated autocorrelation time."""
RSS_FLOOR: Final[float] = 1e-30
FINITE_DIFFERENCE_STEP: Final[float] = 1e-5
LOG_DURATION_BOUNDS: Final[tuple[float, float]] = (float(np.log(0.65)), float(np.log(5.5)))
"""Usable durations between 0.65 and 5.5 hours, inside the stacks' range."""
LOGIT_EFFICIENCY_BOUNDS: Final[tuple[float, float]] = (1.15, 3.44)
"""Round-trip efficiency between 0.76 and 0.969, inside the stacks' range."""
LOGIT_CAP_BOUNDS: Final[tuple[float, float]] = (-8.0, 8.0)
LOG_POWER_BOUNDS_RELATIVE: Final[tuple[float, float]] = (-14.0, 3.0)
"""Bounds on `log P` relative to the log of the power prior's scale."""
LEVENBERG_MARQUARDT_ITERATIONS: Final[int] = 25
LAPLACE_CHUNK_LANES: Final[int] = 4096


@dataclass(frozen=True)
class Layout:
    """How a flat parameter vector is split.

    The vector holds, in order: `log P` of each unit, `log d` of each class, `logit eta` of each
    class, and the logit of the cycle-cap weight of each stack unit. Stack units are the first
    `n_stack` units.

    Attributes:
        unit_class: The class index of each unit.
        n_classes: The number of classes.
        n_stack: The number of stack units, which come first.
    """

    unit_class: tuple[int, ...]
    n_classes: int
    n_stack: int

    @property
    def n_units(self) -> int:
        """The number of units."""
        return len(self.unit_class)

    @property
    def n_parameters(self) -> int:
        """The length of the parameter vector."""
        return self.n_units + 2 * self.n_classes + self.n_stack

    @property
    def power(self) -> slice:
        """Positions of `log P`."""
        return slice(0, self.n_units)

    @property
    def log_duration(self) -> slice:
        """Positions of `log d`."""
        return slice(self.n_units, self.n_units + self.n_classes)

    @property
    def logit_efficiency(self) -> slice:
        """Positions of `logit eta`."""
        return slice(self.n_units + self.n_classes, self.n_units + 2 * self.n_classes)

    @property
    def logit_cap(self) -> slice:
        """Positions of the cycle-cap weight logits."""
        start = self.n_units + 2 * self.n_classes
        return slice(start, start + self.n_stack)


@dataclass(frozen=True)
class Signals:
    """The known signals that drive the units' policies, as tensors on the device.

    Attributes:
        stacks: For each stack unit, the schedules on the grid of durations, efficiencies, and
            cycle caps, positive for export; shape (stack units, n_d * n_e * 2, T), where the node
            index is `(i_d * n_e + i_e) * 2 + cap`.
        duration_nodes: The stack's usable durations in hours, increasing and log-spaced; (n_d,).
        efficiency_nodes: The stack's round-trip efficiencies, increasing; (n_e,).
        fixed_charge: For each window unit, the share of each half-hour inside its charge window.
        fixed_discharge: The same for the discharge window; shape (window units, T).
    """

    stacks: torch.Tensor
    duration_nodes: torch.Tensor
    efficiency_nodes: torch.Tensor
    fixed_charge: torch.Tensor
    fixed_discharge: torch.Tensor


def make_signals(
    *,
    stacks: np.ndarray,
    duration_nodes: np.ndarray,
    efficiency_nodes: np.ndarray,
    fixed_charge: np.ndarray,
    fixed_discharge: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> Signals:
    """Move the policy signals onto the device.

    Args:
        stacks: Shape (stack units, n_d * n_e * 2, T).
        duration_nodes: Shape (n_d,).
        efficiency_nodes: Shape (n_e,).
        fixed_charge: Shape (window units, T).
        fixed_discharge: Same shape.
        device: The device.
        dtype: The floating-point type of the signals.

    Returns:
        The signals.
    """

    def to_tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)

    return Signals(
        stacks=to_tensor(stacks),
        duration_nodes=to_tensor(duration_nodes),
        efficiency_nodes=to_tensor(efficiency_nodes),
        fixed_charge=to_tensor(fixed_charge),
        fixed_discharge=to_tensor(fixed_discharge),
    )


def _unpack(*, theta: torch.Tensor, layout: Layout) -> dict[str, torch.Tensor]:
    """Return the natural-space parameters of each lane.

    Args:
        theta: Shape (lanes, parameters).
        layout: The parameter layout.

    Returns:
        `power` (lanes, units) in MW, `duration` and `efficiency` (lanes, classes), and `cap_weight`
        (lanes, stack units), the weight of the 2-cycle schedule against the 1-cycle schedule.
    """
    return {
        "power": torch.exp(theta[:, layout.power]),
        "duration": torch.exp(theta[:, layout.log_duration]),
        "efficiency": torch.sigmoid(theta[:, layout.logit_efficiency]),
        "cap_weight": torch.sigmoid(theta[:, layout.logit_cap]),
    }


@dataclass(frozen=True)
class Sharpness:
    """How close the smooth gate is to its hard limit.

    Attributes:
        smoothing: The squared width added inside the softmin's square root.
    """

    smoothing: float


FINAL_SHARPNESS: Final[Sharpness] = Sharpness(smoothing=1e-4)


def _node_weights(
    *,
    duration: torch.Tensor,
    efficiency: torch.Tensor,
    cap_weight: torch.Tensor,
    signals: Signals,
) -> torch.Tensor:
    """Return the interpolation weights over a stack's nodes, shape (lanes, nodes).

    Args:
        duration: Shape (lanes,), hours.
        efficiency: Shape (lanes,), round trip.
        cap_weight: Shape (lanes,), the weight of the 2-cycle schedule.
        signals: The signals that hold the node grids.

    Returns:
        Eight non-zero weights per lane (two durations, two efficiencies, two caps) that sum to 1,
        and zero elsewhere. The weights are differentiable in all three arguments.
    """
    d_nodes, e_nodes = signals.duration_nodes, signals.efficiency_nodes
    n_d, n_e = len(d_nodes), len(e_nodes)
    position = (torch.log(duration) - torch.log(d_nodes[0])) / (
        (torch.log(d_nodes[-1]) - torch.log(d_nodes[0])) / (n_d - 1)
    )
    i_d = position.detach().floor().clamp(0, n_d - 2).long()
    f_d = (position - i_d).clamp(0.0, 1.0)
    i_e = (torch.searchsorted(e_nodes, efficiency.detach().contiguous()) - 1).clamp(0, n_e - 2)
    f_e = ((efficiency - e_nodes[i_e]) / (e_nodes[i_e + 1] - e_nodes[i_e])).clamp(0.0, 1.0)
    weights = torch.zeros(
        duration.shape[0], n_d * n_e * 2, dtype=duration.dtype, device=duration.device
    )
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w = (
                    (f_d if a else 1.0 - f_d)
                    * (f_e if b else 1.0 - f_e)
                    * (cap_weight if c else 1.0 - cap_weight)
                )
                index = (((i_d + a) * n_e + (i_e + b)) * 2 + c)[:, None]
                weights.scatter_add_(1, index, w[:, None])
    return weights


def simulate(
    *,
    theta: torch.Tensor,
    layout: Layout,
    signals: Signals,
    sharpness: Sharpness,
    return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the state-space recurrence for a batch of parameter vectors.

    Args:
        theta: Shape (lanes, parameters).
        layout: The parameter layout.
        signals: The policy signals.
        sharpness: The gate's smoothing.
        return_state: Also return each unit's state of charge and per-unit power.

    Returns:
        The summed export in MW, positive for export, shape (lanes, T). With `return_state`, also
        the state of charge before each half-hour (lanes, units, T) and the per-unit power in
        fractions of full power (lanes, units, T), positive for discharge.
    """
    lanes = theta.shape[0]
    parameters = _unpack(theta=theta, layout=layout)
    classes = torch.as_tensor(layout.unit_class, device=theta.device)
    duration = parameters["duration"][:, classes]
    efficiency = parameters["efficiency"][:, classes]
    intents = []
    for unit in range(layout.n_stack):
        weights = _node_weights(
            duration=duration[:, unit],
            efficiency=efficiency[:, unit],
            cap_weight=parameters["cap_weight"][:, unit],
            signals=signals,
        )
        intents.append(weights @ signals.stacks[unit])
    intent = torch.stack(intents, dim=1) if intents else duration.new_zeros(lanes, 0, 1)
    n_fixed = signals.fixed_charge.shape[0]
    charge = torch.cat(
        [torch.relu(-intent), signals.fixed_charge[None].expand(lanes, n_fixed, -1)], dim=1
    )
    discharge = torch.cat(
        [torch.relu(intent), signals.fixed_discharge[None].expand(lanes, n_fixed, -1)], dim=1
    )
    n_units = layout.n_units
    charge = charge.permute(2, 0, 1).reshape(-1, lanes * n_units)
    discharge = discharge.permute(2, 0, 1).reshape(-1, lanes * n_units)
    per_unit, state = recurrence(
        charge=charge,
        discharge=discharge,
        duration=duration.reshape(-1),
        eta_one_way=torch.sqrt(efficiency).reshape(-1),
        smoothing=sharpness.smoothing,
        return_state=return_state,
    )
    per_unit = per_unit.reshape(-1, lanes, n_units)
    export = (per_unit * parameters["power"][None]).sum(dim=2).transpose(0, 1)
    if return_state:
        assert state is not None
        return (
            export,
            state.reshape(-1, lanes, n_units).permute(1, 2, 0),
            per_unit.permute(1, 2, 0),
        )
    return export


def log_prior(
    *,
    theta: torch.Tensor,
    layout: Layout,
    power_scale: torch.Tensor,
    log_duration_mean: torch.Tensor,
    log_duration_sd: torch.Tensor,
    efficiency_alpha: torch.Tensor,
    efficiency_beta: torch.Tensor,
) -> torch.Tensor:
    """Return the log prior density of unconstrained parameters, with the change-of-variable terms.

    Power is half-normal, duration is log-normal, efficiency is Beta, and each cap weight is
    uniform between 0 and 1.

    Args:
        theta: Shape (lanes, parameters).
        layout: The parameter layout.
        power_scale: The half-normal scale of every unit's power, shape (lanes,).
        log_duration_mean: The mean of `log d` of each class, shape (classes,).
        log_duration_sd: The standard deviation of `log d` of each class.
        efficiency_alpha: The first Beta shape of each class.
        efficiency_beta: The second Beta shape of each class.

    Returns:
        Shape (lanes,), up to an additive constant.
    """
    log_power = theta[:, layout.power]
    power = torch.exp(log_power)
    power_term = (-0.5 * (power / power_scale[:, None]) ** 2 + log_power).sum(dim=1)
    log_d = theta[:, layout.log_duration]
    duration_term = (-0.5 * ((log_d - log_duration_mean) / log_duration_sd) ** 2).sum(dim=1)
    logit_eta = theta[:, layout.logit_efficiency]
    efficiency_term = (
        -efficiency_alpha * torch.nn.functional.softplus(-logit_eta)
        - efficiency_beta * torch.nn.functional.softplus(logit_eta)
    ).sum(dim=1)
    cap = theta[:, layout.logit_cap]
    cap_term = (-torch.nn.functional.softplus(-cap) - torch.nn.functional.softplus(cap)).sum(dim=1)
    return power_term + duration_term + efficiency_term + cap_term


@dataclass(frozen=True)
class Priors:
    """The class priors, as tensors.

    Attributes:
        log_duration_mean: The mean of `log d` of each class.
        log_duration_sd: Its standard deviation.
        efficiency_alpha: The first Beta shape of the efficiency of each class.
        efficiency_beta: The second Beta shape.
    """

    log_duration_mean: np.ndarray
    log_duration_sd: np.ndarray
    efficiency_alpha: np.ndarray
    efficiency_beta: np.ndarray

    def tensors(self, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        """Return the priors as keyword arguments of `log_prior`."""
        return {
            name: torch.as_tensor(value, dtype=dtype, device=device)
            for name, value in (
                ("log_duration_mean", self.log_duration_mean),
                ("log_duration_sd", self.log_duration_sd),
                ("efficiency_alpha", self.efficiency_alpha),
                ("efficiency_beta", self.efficiency_beta),
            )
        }


@dataclass(frozen=True)
class Problems:
    """The aggregates to fit, grouped so that each group shares one projector.

    A *group* is one demand series in one block. A group holds `lanes` different aggregates (for
    example a battery of each share and duration subtracted from the same demand series), and the
    fit runs `starts` initialisations of each.

    Attributes:
        aggregate: Shape (groups, lanes, T), the import-positive aggregate in MW, 0 where invalid.
        valid: Shape (groups, T), True where the aggregate and every free column are finite.
        basis: Shape (groups, T, B), an orthonormal basis of the free columns' span at the valid
            half-hours, zero-padded to a common B.
        rank: Shape (groups,), the true rank of each basis.
        power_scale: Shape (groups, lanes), the half-normal scale of every unit's power in MW.
    """

    aggregate: np.ndarray
    valid: np.ndarray
    basis: np.ndarray
    rank: np.ndarray
    power_scale: np.ndarray


def orthonormal_basis(*, free: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int]:
    """Return an orthonormal basis of the free columns at the valid rows, zero at the others.

    Args:
        free: Shape (T, columns).
        valid: Shape (T,), True where a row is used.

    Returns:
        The basis, shape (T, rank), and its rank.
    """
    masked = np.where(valid[:, None], free, 0.0)
    left, singular, _ = np.linalg.svd(masked, full_matrices=False)
    keep = singular > singular[0] * 1e-9
    return left[:, keep], int(keep.sum())


def make_problems(
    *,
    aggregate: np.ndarray,
    valid: np.ndarray,
    free: Sequence[np.ndarray],
    power_scale: np.ndarray,
) -> Problems:
    """Assemble groups from per-group free columns.

    Args:
        aggregate: Shape (groups, lanes, T).
        valid: Shape (groups, T).
        free: One (T, columns) array per group.
        power_scale: Shape (groups, lanes).

    Returns:
        The problems, with the bases padded to a common width.
    """
    bases, ranks = zip(
        *[
            orthonormal_basis(free=columns, valid=valid[group])
            for group, columns in enumerate(free)
        ],
        strict=True,
    )
    width = max(basis.shape[1] for basis in bases)
    padded = np.stack(
        [np.pad(basis, ((0, 0), (0, width - basis.shape[1]))) for basis in bases], axis=0
    )
    return Problems(
        aggregate=np.where(valid[:, None, :], aggregate, 0.0),
        valid=valid,
        basis=padded,
        rank=np.array(ranks),
        power_scale=power_scale,
    )


def integrated_autocorrelation_time(
    *, residual: torch.Tensor, valid: torch.Tensor, max_lag: int = AUTOCORRELATION_LAGS
) -> torch.Tensor:
    """Return the Bartlett-weighted integrated autocorrelation time of each residual, at least 1.

    Args:
        residual: Shape (lanes, T), zero where invalid.
        valid: Shape (lanes, T), boolean.
        max_lag: The most lags counted.

    Returns:
        Shape (lanes,): `1 + 2 * sum_k (1 - k / (max_lag + 1)) * rho_k`.
    """
    count = valid.sum(dim=1, keepdim=True)
    mean = residual.sum(dim=1, keepdim=True) / count
    centred = torch.where(valid, residual - mean, torch.zeros_like(residual))
    size = 2 * centred.shape[1]
    spectrum = torch.fft.rfft(centred, n=size, dim=1)
    correlation = torch.fft.irfft(spectrum * torch.conj(spectrum), n=size, dim=1)
    lags = torch.arange(1, max_lag + 1, device=residual.device, dtype=residual.dtype)
    weights = 1.0 - lags / (max_lag + 1)
    ratio = correlation[:, 1 : max_lag + 1] / correlation[:, :1].clamp_min(RSS_FLOOR)
    tau = 1.0 + 2.0 * (ratio * weights).sum(dim=1)
    return tau.clamp_min(1.0)


@dataclass(frozen=True)
class FitResult:
    """The fitted parameters and their Laplace approximation, per group, lane, and start.

    Every array has leading shape (groups, lanes, starts).

    Attributes:
        theta: The optimum, in the unconstrained space; shape (..., parameters), float64.
        covariance: The Laplace covariance, shape (..., parameters, parameters); NaN where the
            Hessian was not positive definite.
        loss: The loss at the optimum (negative log posterior, up to a constant).
        rss: The residual sum of squares at the optimum.
        tau: The integrated autocorrelation time of the residual.
        has_interval: Whether the covariance exists.
        at_bound: Whether any parameter sits on a bound of the box it is fitted in.
        log_evidence: The Laplace log marginal likelihood of the model with batteries, NaN where
            there is no covariance.
        null_log_evidence: The log marginal likelihood of the model with no battery, from the same
            tempered likelihood. `log_evidence - null_log_evidence` is the log Bayes factor.
        best_start: For each group and lane, the start with the lowest loss; shape (groups, lanes).
    """

    theta: np.ndarray
    covariance: np.ndarray
    loss: np.ndarray
    rss: np.ndarray
    tau: np.ndarray
    has_interval: np.ndarray
    at_bound: np.ndarray
    log_evidence: np.ndarray
    null_log_evidence: np.ndarray
    best_start: np.ndarray


class Estimator:
    """Fits batches of aggregates with the differentiable state-space battery.

    Attributes:
        layout: The parameter layout.
        signals: The policy signals, in the fit dtype on the device.
        signals64: The same signals in float64, for the polish and the Hessian.
        priors: The class priors.
        device: The device.
    """

    def __init__(
        self,
        *,
        layout: Layout,
        signals_numpy: dict[str, np.ndarray],
        priors: Priors,
        device: torch.device,
    ) -> None:
        """Create an estimator.

        Args:
            layout: The parameter layout.
            signals_numpy: The keyword arguments of `make_signals` other than the device and dtype.
            priors: The class priors.
            device: The device the fit runs on.
        """
        self.layout = layout
        self.device = device
        self.priors = priors
        self.signals = make_signals(**signals_numpy, device=device, dtype=torch.float32)
        self.signals64 = make_signals(**signals_numpy, device=device, dtype=torch.float64)
        self._prior_args = priors.tensors(device=device, dtype=torch.float64)

    def _prior(self, *, theta: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        args = {k: v.to(theta.dtype) for k, v in self._prior_args.items()}
        return log_prior(theta=theta, layout=self.layout, power_scale=scale, **args)

    def _prior_constant(self, *, scale: torch.Tensor) -> torch.Tensor:
        """Return the log normalising constant of the log prior of every lane."""
        args = self._prior_args
        n_units = self.layout.n_units
        power = n_units * (math.log(2.0) - 0.5 * math.log(2 * math.pi)) - n_units * torch.log(scale)
        duration = -(torch.log(args["log_duration_sd"]) + 0.5 * math.log(2 * math.pi)).sum()
        alpha, beta = args["efficiency_alpha"], args["efficiency_beta"]
        efficiency = -(torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)).sum()
        return power + duration + efficiency

    def _bounds(self, *, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the lower and upper bound of every parameter of every lane."""
        layout = self.layout
        lanes = scale.shape[0]
        low = torch.empty(lanes, layout.n_parameters, dtype=scale.dtype, device=scale.device)
        high = torch.empty_like(low)
        log_scale = torch.log(scale)[:, None]
        low[:, layout.power] = log_scale + LOG_POWER_BOUNDS_RELATIVE[0]
        high[:, layout.power] = log_scale + LOG_POWER_BOUNDS_RELATIVE[1]
        for part, bounds in (
            (layout.log_duration, LOG_DURATION_BOUNDS),
            (layout.logit_efficiency, LOGIT_EFFICIENCY_BOUNDS),
            (layout.logit_cap, LOGIT_CAP_BOUNDS),
        ):
            low[:, part], high[:, part] = bounds
        return low, high

    def _project(
        self, *, export: torch.Tensor, basis: torch.Tensor, valid: torch.Tensor, groups: int
    ) -> torch.Tensor:
        """Return `M export`, the export projected off the free columns of its group."""
        lanes = export.shape[0] // groups
        masked = export.reshape(groups, lanes, -1) * valid[:, None, :]
        coefficients = torch.bmm(masked, basis)
        return (masked - torch.bmm(coefficients, basis.transpose(1, 2))).reshape(export.shape)

    def _adam(
        self,
        *,
        theta: torch.Tensor,
        projected_aggregate: torch.Tensor,
        basis: torch.Tensor,
        valid: torch.Tensor,
        groups: int,
        scale: torch.Tensor,
        effective: torch.Tensor,
        bounds: tuple[torch.Tensor, torch.Tensor],
        stages: Sequence[tuple[int, float, Sharpness]],
        verbose: bool,
    ) -> torch.Tensor:
        """Run Adam in float32 through the annealing stages and return the float64 optimum."""
        layout = self.layout
        theta32 = theta.to(torch.float32).requires_grad_(True)
        low32, high32 = bounds[0].to(torch.float32), bounds[1].to(torch.float32)
        aggregate32 = projected_aggregate.to(torch.float32)
        basis32 = basis.to(torch.float32)
        valid32 = valid.to(torch.float32)
        scale32 = scale.to(torch.float32)
        effective32 = effective.to(torch.float32)
        prior_args32 = {k: v.to(torch.float32) for k, v in self._prior_args.items()}
        for index, (iterations, learning_rate, sharpness) in enumerate(stages):
            optimiser = torch.optim.Adam([theta32], lr=learning_rate)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=iterations)
            for step in range(iterations):
                optimiser.zero_grad(set_to_none=True)
                export = simulate(
                    theta=theta32, layout=layout, signals=self.signals, sharpness=sharpness
                )
                residual = aggregate32 + self._project(
                    export=export, basis=basis32, valid=valid32, groups=groups
                )
                rss = (residual**2).sum(dim=1).clamp_min(RSS_FLOOR)
                loss = 0.5 * effective32 * torch.log(rss) - log_prior(
                    theta=theta32, layout=layout, power_scale=scale32, **prior_args32
                )
                loss.sum().backward()
                optimiser.step()
                schedule.step()
                with torch.no_grad():
                    theta32.copy_(torch.maximum(torch.minimum(theta32, high32), low32))
                if verbose and step % 50 == 0:
                    print(
                        f"  stage {index} step {step}: mean loss {float(loss.mean()):.3f}",
                        flush=True,
                    )
        return theta32.detach().to(torch.float64)

    def fit(
        self,
        *,
        problems: Problems,
        starts: np.ndarray,
        stages: Sequence[tuple[int, float, Sharpness]],
        seed: int = 0,
        verbose: bool = False,
    ) -> FitResult:
        """Fit every aggregate from every start, then polish and approximate the posterior.

        Args:
            problems: The aggregates.
            starts: Initial parameters relative to each group's power scale, shape (starts,
                parameters): the `log P` entries are offsets from the log of the group's power
                scale, the others are absolute.
            stages: Adam stages, each `(iterations, learning rate, sharpness)`. The last stage's
                sharpness defines the model that is polished and approximated.
            seed: Unused apart from reproducibility of future randomised starts.
            verbose: Print progress.

        Returns:
            The fit.
        """
        del seed
        layout = self.layout
        groups, lanes, n_time = problems.aggregate.shape
        n_starts = starts.shape[0]
        device = self.device
        valid64 = torch.as_tensor(problems.valid, dtype=torch.float64, device=device)
        basis64 = torch.as_tensor(problems.basis, dtype=torch.float64, device=device)
        aggregate64 = torch.as_tensor(problems.aggregate, dtype=torch.float64, device=device)
        # Lane order: group, lane, start. `base` repeats each aggregate once per start.
        flat_aggregate = aggregate64.repeat_interleave(n_starts, dim=1).reshape(-1, n_time)
        per_group = lanes * n_starts
        n_lanes = groups * per_group
        flat_valid = valid64.repeat_interleave(per_group, dim=0)
        basis_per_lane_group = basis64
        projected_aggregate = self._project(
            export=flat_aggregate, basis=basis_per_lane_group, valid=valid64, groups=groups
        )
        n_equations = torch.as_tensor(
            problems.valid.sum(axis=1) - problems.rank, dtype=torch.float64, device=device
        ).repeat_interleave(per_group)
        scale = torch.as_tensor(problems.power_scale, dtype=torch.float64, device=device)
        scale = scale.reshape(-1).repeat_interleave(n_starts)
        tau = torch.ones(n_lanes, dtype=torch.float64, device=device)
        start_theta = torch.as_tensor(starts, dtype=torch.float64, device=device)
        theta = start_theta.repeat(groups * lanes, 1).clone()
        theta[:, layout.power] += torch.log(scale)[:, None]
        low, high = self._bounds(scale=scale)
        theta = torch.maximum(torch.minimum(theta, high), low)

        def loss_fn(
            *, theta_in: torch.Tensor, export: torch.Tensor, effective: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual = projected_aggregate + self._project(
                export=export, basis=basis_per_lane_group, valid=valid64, groups=groups
            ).to(projected_aggregate.dtype)
            rss = (residual**2).sum(dim=1).clamp_min(RSS_FLOOR)
            loss = 0.5 * effective * torch.log(rss) - self._prior(theta=theta_in, scale=scale)
            return loss, rss

        # Stage 1: Adam in float32 on the device.
        theta = self._adam(
            theta=theta,
            projected_aggregate=projected_aggregate,
            basis=basis64,
            valid=valid64,
            groups=groups,
            scale=scale,
            effective=n_equations / tau,
            bounds=(low, high),
            stages=stages,
            verbose=verbose,
        )
        final_sharpness = stages[-1][2]

        # Stage 2: Levenberg-Marquardt in float64, re-estimating the tempering twice.
        for _ in range(3):
            export = self._export64(theta=theta, sharpness=final_sharpness)
            residual = projected_aggregate + self._project(
                export=export, basis=basis64, valid=valid64, groups=groups
            )
            tau = integrated_autocorrelation_time(residual=residual, valid=flat_valid > 0)
            theta = self._levenberg_marquardt(
                theta=theta,
                tau=tau,
                scale=scale,
                low=low,
                high=high,
                sharpness=final_sharpness,
                n_equations=n_equations,
                projected_aggregate=projected_aggregate,
                basis=basis64,
                valid=valid64,
                groups=groups,
            )
        export = self._export64(theta=theta, sharpness=final_sharpness)
        residual = projected_aggregate + self._project(
            export=export, basis=basis64, valid=valid64, groups=groups
        )
        tau = integrated_autocorrelation_time(residual=residual, valid=flat_valid > 0)
        effective = n_equations / tau
        loss, rss = loss_fn(theta_in=theta, export=export, effective=effective)
        covariance, has_interval, log_determinant = self._laplace(
            theta=theta,
            effective=effective,
            scale=scale,
            sharpness=final_sharpness,
            projected_aggregate=projected_aggregate,
            basis=basis64,
            valid=valid64,
            groups=groups,
        )
        at_bound = ((theta <= low + 1e-6) | (theta >= high - 1e-6)).any(dim=1)
        likelihood_constant = torch.lgamma(effective / 2) - effective / 2 * math.log(math.pi)
        log_evidence = (
            -(loss - self._prior_constant(scale=scale) - likelihood_constant)
            + 0.5 * layout.n_parameters * math.log(2 * math.pi)
            - 0.5 * log_determinant
        )
        null_rss = (projected_aggregate**2).sum(dim=1).clamp_min(RSS_FLOOR)
        null_tau = integrated_autocorrelation_time(
            residual=projected_aggregate, valid=flat_valid > 0
        )
        null_effective = n_equations / null_tau
        null_log_evidence = torch.lgamma(null_effective / 2) - null_effective / 2 * (
            math.log(math.pi) + torch.log(null_rss)
        )
        shape = (groups, lanes, n_starts)
        loss_np = loss.reshape(shape).cpu().numpy()
        return FitResult(
            theta=theta.reshape(*shape, -1).cpu().numpy(),
            covariance=covariance.reshape(*shape, layout.n_parameters, layout.n_parameters)
            .cpu()
            .numpy(),
            loss=loss_np,
            rss=rss.reshape(shape).cpu().numpy(),
            tau=tau.reshape(shape).cpu().numpy(),
            has_interval=has_interval.reshape(shape).cpu().numpy(),
            at_bound=at_bound.reshape(shape).cpu().numpy(),
            log_evidence=log_evidence.reshape(shape).cpu().numpy(),
            null_log_evidence=null_log_evidence.reshape(shape).cpu().numpy(),
            best_start=loss_np.argmin(axis=2),
        )

    @torch.no_grad()
    def _export64(self, *, theta: torch.Tensor, sharpness: Sharpness) -> torch.Tensor:
        """Return the float64 export of every lane, in chunks."""
        chunks = [
            simulate(
                theta=theta[start : start + LAPLACE_CHUNK_LANES],
                layout=self.layout,
                signals=self.signals64,
                sharpness=sharpness,
            )
            for start in range(0, theta.shape[0], LAPLACE_CHUNK_LANES)
        ]
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def _jacobian(
        self,
        *,
        theta: torch.Tensor,
        sharpness: Sharpness,
        basis: torch.Tensor,
        valid: torch.Tensor,
        groups: int,
    ) -> torch.Tensor:
        """Return `M d export / d theta` by central differences, shape (lanes, T, parameters)."""
        lanes, n_parameters = theta.shape
        step = FINITE_DIFFERENCE_STEP
        eye = torch.eye(n_parameters, dtype=theta.dtype, device=theta.device) * step
        plus = (theta[:, None, :] + eye[None]).reshape(-1, n_parameters)
        minus = (theta[:, None, :] - eye[None]).reshape(-1, n_parameters)
        columns = [
            self._export64(theta=perturbed, sharpness=sharpness) for perturbed in (plus, minus)
        ]
        derivative = (columns[0] - columns[1]) / (2 * step)
        derivative = derivative.reshape(lanes, n_parameters, -1)
        per_group = lanes // groups
        flat = derivative.reshape(groups, per_group * n_parameters, -1)
        masked = flat * valid[:, None, :]
        coefficients = torch.bmm(masked, basis)
        projected = masked - torch.bmm(coefficients, basis.transpose(1, 2))
        return projected.reshape(lanes, n_parameters, -1).transpose(1, 2)

    def _prior_gradient_hessian(
        self, *, theta: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the gradient and Hessian of the log prior, per lane.

        The prior is a sum of terms that each depend on one parameter, so its Hessian is diagonal
        and two reverse-mode passes give it exactly.
        """
        with torch.enable_grad():
            vector = theta.detach().clone().requires_grad_(True)
            total = self._prior(theta=vector, scale=scale).sum()
            (gradient,) = torch.autograd.grad(total, vector, create_graph=True)
            (second,) = torch.autograd.grad(gradient.sum(), vector)
        return gradient.detach(), torch.diag_embed(second)

    def _gauss_newton(
        self,
        *,
        theta: torch.Tensor,
        effective: torch.Tensor,
        scale: torch.Tensor,
        sharpness: Sharpness,
        projected_aggregate: torch.Tensor,
        basis: torch.Tensor,
        valid: torch.Tensor,
        groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the loss gradient, the PSD Gauss-Newton Hessian, the rank-one term, and RSS."""
        export = self._export64(theta=theta, sharpness=sharpness)
        residual = projected_aggregate + self._project(
            export=export, basis=basis, valid=valid, groups=groups
        )
        rss = (residual**2).sum(dim=1).clamp_min(RSS_FLOOR)
        jacobian = self._jacobian(
            theta=theta, sharpness=sharpness, basis=basis, valid=valid, groups=groups
        )
        score = torch.einsum("ltk,lt->lk", jacobian, residual)
        gram = torch.einsum("ltk,ltj->lkj", jacobian, jacobian)
        prior_gradient, prior_hessian = self._prior_gradient_hessian(theta=theta, scale=scale)
        gradient = (effective / rss)[:, None] * score - prior_gradient
        psd = (effective / rss)[:, None, None] * gram - prior_hessian
        correction = (2 * effective / rss**2)[:, None, None] * score[:, :, None] * score[:, None, :]
        return gradient, psd, correction, rss

    def _levenberg_marquardt(
        self,
        *,
        theta: torch.Tensor,
        tau: torch.Tensor,
        scale: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        sharpness: Sharpness,
        n_equations: torch.Tensor,
        projected_aggregate: torch.Tensor,
        basis: torch.Tensor,
        valid: torch.Tensor,
        groups: int,
    ) -> torch.Tensor:
        """Polish every lane's optimum with damped Gauss-Newton steps, keeping only improvements."""
        effective = n_equations / tau
        damping = torch.full_like(tau, 1e-2)
        eye = torch.eye(theta.shape[1], dtype=theta.dtype, device=theta.device)

        def loss_at(vector: torch.Tensor) -> torch.Tensor:
            export = self._export64(theta=vector, sharpness=sharpness)
            residual = projected_aggregate + self._project(
                export=export, basis=basis, valid=valid, groups=groups
            )
            rss = (residual**2).sum(dim=1).clamp_min(RSS_FLOOR)
            return 0.5 * effective * torch.log(rss) - self._prior(theta=vector, scale=scale)

        current = loss_at(theta)
        for _ in range(LEVENBERG_MARQUARDT_ITERATIONS):
            gradient, psd, _, _ = self._gauss_newton(
                theta=theta,
                effective=effective,
                scale=scale,
                sharpness=sharpness,
                projected_aggregate=projected_aggregate,
                basis=basis,
                valid=valid,
                groups=groups,
            )
            diagonal = torch.diagonal(psd, dim1=1, dim2=2).clamp_min(1e-12)
            system = psd + damping[:, None, None] * diagonal[:, :, None] * eye[None]
            step = torch.linalg.solve(system, gradient[:, :, None])[:, :, 0]
            proposal = torch.maximum(torch.minimum(theta - step, high), low)
            proposed = loss_at(proposal)
            better = torch.isfinite(proposed) & (proposed < current)
            theta = torch.where(better[:, None], proposal, theta)
            current = torch.where(better, proposed, current)
            damping = torch.where(better, damping / 3.0, damping * 10.0).clamp(1e-9, 1e9)
        return theta

    def _laplace(
        self,
        *,
        theta: torch.Tensor,
        effective: torch.Tensor,
        scale: torch.Tensor,
        sharpness: Sharpness,
        projected_aggregate: torch.Tensor,
        basis: torch.Tensor,
        valid: torch.Tensor,
        groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return each lane's Laplace covariance, its definiteness, and its log determinant.

        Returns:
            The covariance (NaN where the Hessian is not positive definite), whether the Hessian is
            positive definite, and the log determinant of the Hessian (NaN where it is not).
        """
        _, psd, correction, _ = self._gauss_newton(
            theta=theta,
            effective=effective,
            scale=scale,
            sharpness=sharpness,
            projected_aggregate=projected_aggregate,
            basis=basis,
            valid=valid,
            groups=groups,
        )
        hessian = psd - correction
        hessian = 0.5 * (hessian + hessian.transpose(1, 2))
        eigenvalues = torch.linalg.eigvalsh(hessian)
        positive = eigenvalues[:, 0] > 0
        covariance = torch.full_like(hessian, float("nan"))
        log_determinant = torch.full_like(eigenvalues[:, 0], float("nan"))
        if positive.any():
            covariance[positive] = torch.linalg.inv(hessian[positive])
            log_determinant[positive] = torch.log(eigenvalues[positive]).sum(dim=1)
        return covariance, positive, log_determinant


def natural_draws(
    *,
    theta: np.ndarray,
    covariance: np.ndarray,
    layout: Layout,
    n_draws: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw from a Laplace approximation and map the draws to natural units.

    Args:
        theta: The optimum, shape (parameters,).
        covariance: The covariance, shape (parameters, parameters), positive definite.
        layout: The parameter layout.
        n_draws: The number of draws.
        rng: The random generator.

    Returns:
        `power` (draws, units) in MW, `duration` (draws, classes) in hours, `efficiency` (draws,
        classes), `cap_weight` (draws, stack units), and `energy` (draws, units) in MWh of usable
        energy, each unit's power times its class's duration.
    """
    draws = rng.multivariate_normal(theta, covariance, size=n_draws, method="eigh")
    return _natural(draws=draws, layout=layout)


def natural_point(*, theta: np.ndarray, layout: Layout) -> dict[str, np.ndarray]:
    """Map one optimum to natural units, with the same keys as `natural_draws` (one row)."""
    return {k: v[0] for k, v in _natural(draws=theta[None], layout=layout).items()}


def _natural(*, draws: np.ndarray, layout: Layout) -> dict[str, np.ndarray]:
    """Map unconstrained parameters, one row each, to natural units."""
    power = np.exp(draws[:, layout.power])
    duration = np.exp(draws[:, layout.log_duration])
    return {
        "power": power,
        "duration": duration,
        "efficiency": 1.0 / (1.0 + np.exp(-draws[:, layout.logit_efficiency])),
        "cap_weight": 1.0 / (1.0 + np.exp(-draws[:, layout.logit_cap])),
        "energy": power * duration[:, list(layout.unit_class)],
    }
