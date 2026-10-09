"""The battery state-of-charge recurrence, with a reference loop and a fused GPU kernel.

For each lane (one unit of one fit) and each half-hour `t`, given the policy signals `charge_t` and
`discharge_t` in [0, 1], the usable duration `d` in hours, and the one-way efficiency `e`:

```text
discharged_t = softmin(discharge_t, s_t * d * e / 0.5)
charged_t    = softmin(charge_t, (1 - s_t) * d / (0.5 * e))
s_t+1        = s_t + 0.5 / d * (e * charged_t - discharged_t / e)
output_t     = discharged_t - charged_t
```

where `softmin(a, b) = max(0, (a + b - sqrt((a - b)^2 + smoothing)) / 2)`.

`recurrence_reference` is a Python loop over the half-hours, the oracle for the tests and the only
path on a CPU. `recurrence` runs a Triton kernel on a CUDA device: one thread per lane walks the
whole year in registers, and a second kernel walks it backwards for the gradient, so a year costs
milliseconds instead of the tens of thousands of small kernel launches that the loop needs.
"""

from typing import Final

import torch

HOURS_PER_HALF_HOUR: Final[float] = 0.5
LANES_PER_PROGRAM: Final[int] = 32


def _softmin(*, a: torch.Tensor, b: torch.Tensor, smoothing: float) -> torch.Tensor:
    """A smooth minimum that never exceeds the true minimum, floored at zero."""
    return torch.relu(0.5 * (a + b - torch.sqrt((a - b) ** 2 + smoothing)))


def recurrence_reference(
    *,
    charge: torch.Tensor,
    discharge: torch.Tensor,
    duration: torch.Tensor,
    eta_one_way: torch.Tensor,
    smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the recurrence as a Python loop over the half-hours.

    Args:
        charge: The charge policy, shape (T, lanes).
        discharge: The discharge policy, shape (T, lanes).
        duration: The usable duration in hours, shape (lanes,).
        eta_one_way: The one-way efficiency, shape (lanes,).
        smoothing: The squared width inside the softmin's square root.

    Returns:
        The per-unit output, positive for discharge, shape (T, lanes), and the state of charge
        before each half-hour as a fraction of the usable energy, same shape.
    """
    headroom_gain = duration * eta_one_way / HOURS_PER_HALF_HOUR
    room_gain = duration / (HOURS_PER_HALF_HOUR * eta_one_way)
    step = HOURS_PER_HALF_HOUR / duration
    state = torch.zeros_like(duration)
    outputs = []
    states = []
    for t in range(charge.shape[0]):
        discharged = _softmin(a=discharge[t], b=state * headroom_gain, smoothing=smoothing)
        charged = _softmin(a=charge[t], b=(1.0 - state) * room_gain, smoothing=smoothing)
        states.append(state)
        state = state + step * (eta_one_way * charged - discharged / eta_one_way)
        outputs.append(discharged - charged)
    return torch.stack(outputs), torch.stack(states)


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - triton ships with the CUDA build of torch
    triton = None


if triton is not None:

    @triton.jit
    def _forward_kernel(
        charge_ptr: tl.tensor,
        discharge_ptr: tl.tensor,
        duration_ptr: tl.tensor,
        eta_ptr: tl.tensor,
        output_ptr: tl.tensor,
        state_ptr: tl.tensor,
        n_time: int,
        n_lanes: int,
        smoothing: float,
        block_size: tl.constexpr,
    ) -> None:
        lane = tl.program_id(0) * block_size + tl.arange(0, block_size)
        inside = lane < n_lanes
        duration = tl.load(duration_ptr + lane, mask=inside, other=1.0)
        eta = tl.load(eta_ptr + lane, mask=inside, other=1.0)
        headroom_gain = duration * eta * 2.0
        room_gain = duration * 2.0 / eta
        step = 0.5 / duration
        state = tl.zeros([block_size], dtype=duration.dtype)
        for t in range(n_time):
            offset = t * n_lanes + lane
            discharge = tl.load(discharge_ptr + offset, mask=inside, other=0.0)
            charge = tl.load(charge_ptr + offset, mask=inside, other=0.0)
            tl.store(state_ptr + offset, state, mask=inside)
            limit_d = state * headroom_gain
            diff_d = discharge - limit_d
            discharged = tl.maximum(
                0.5 * (discharge + limit_d - tl.sqrt(diff_d * diff_d + smoothing)), 0.0
            )
            limit_c = (1.0 - state) * room_gain
            diff_c = charge - limit_c
            charged = tl.maximum(
                0.5 * (charge + limit_c - tl.sqrt(diff_c * diff_c + smoothing)), 0.0
            )
            tl.store(output_ptr + offset, discharged - charged, mask=inside)
            state = state + step * (eta * charged - discharged / eta)

    @triton.jit
    def _backward_kernel(
        charge_ptr: tl.tensor,
        discharge_ptr: tl.tensor,
        duration_ptr: tl.tensor,
        eta_ptr: tl.tensor,
        state_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        grad_charge_ptr: tl.tensor,
        grad_discharge_ptr: tl.tensor,
        grad_duration_ptr: tl.tensor,
        grad_eta_ptr: tl.tensor,
        n_time: int,
        n_lanes: int,
        smoothing: float,
        block_size: tl.constexpr,
    ) -> None:
        lane = tl.program_id(0) * block_size + tl.arange(0, block_size)
        inside = lane < n_lanes
        duration = tl.load(duration_ptr + lane, mask=inside, other=1.0)
        eta = tl.load(eta_ptr + lane, mask=inside, other=1.0)
        headroom_gain = duration * eta * 2.0
        room_gain = duration * 2.0 / eta
        step = 0.5 / duration
        adjoint = tl.zeros([block_size], dtype=duration.dtype)
        grad_duration = tl.zeros([block_size], dtype=duration.dtype)
        grad_eta = tl.zeros([block_size], dtype=duration.dtype)
        for i in range(n_time):
            t = n_time - 1 - i
            offset = t * n_lanes + lane
            discharge = tl.load(discharge_ptr + offset, mask=inside, other=0.0)
            charge = tl.load(charge_ptr + offset, mask=inside, other=0.0)
            state = tl.load(state_ptr + offset, mask=inside, other=0.0)
            grad_out = tl.load(grad_output_ptr + offset, mask=inside, other=0.0)
            limit_d = state * headroom_gain
            diff_d = discharge - limit_d
            root_d = tl.sqrt(diff_d * diff_d + smoothing)
            pre_d = 0.5 * (discharge + limit_d - root_d)
            discharged = tl.maximum(pre_d, 0.0)
            limit_c = (1.0 - state) * room_gain
            diff_c = charge - limit_c
            root_c = tl.sqrt(diff_c * diff_c + smoothing)
            pre_c = 0.5 * (charge + limit_c - root_c)
            charged = tl.maximum(pre_c, 0.0)
            # The loss gradient with respect to discharged_t and charged_t.
            weight_d = grad_out - adjoint * step / eta
            weight_c = -grad_out + adjoint * step * eta
            weight_d = tl.where(pre_d > 0.0, weight_d, 0.0)
            weight_c = tl.where(pre_c > 0.0, weight_c, 0.0)
            grad_discharge = weight_d * 0.5 * (1.0 - diff_d / root_d)
            grad_limit_d = weight_d * 0.5 * (1.0 + diff_d / root_d)
            grad_charge = weight_c * 0.5 * (1.0 - diff_c / root_c)
            grad_limit_c = weight_c * 0.5 * (1.0 + diff_c / root_c)
            tl.store(grad_discharge_ptr + offset, grad_discharge, mask=inside)
            tl.store(grad_charge_ptr + offset, grad_charge, mask=inside)
            moved = eta * charged - discharged / eta
            grad_duration += (
                -adjoint * step / duration * moved
                + grad_limit_d * state * eta * 2.0
                + grad_limit_c * (1.0 - state) * 2.0 / eta
            )
            grad_eta += (
                adjoint * step * (charged + discharged / (eta * eta))
                + grad_limit_d * state * duration * 2.0
                - grad_limit_c * (1.0 - state) * 2.0 * duration / (eta * eta)
            )
            adjoint = adjoint + grad_limit_d * headroom_gain - grad_limit_c * room_gain
        tl.store(grad_duration_ptr + lane, grad_duration, mask=inside)
        tl.store(grad_eta_ptr + lane, grad_eta, mask=inside)


class _KernelRecurrence(torch.autograd.Function):
    """The recurrence on a CUDA device, with its hand-written reverse pass."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        charge: torch.Tensor,
        discharge: torch.Tensor,
        duration: torch.Tensor,
        eta_one_way: torch.Tensor,
        smoothing: float,
    ) -> torch.Tensor:
        charge, discharge = charge.contiguous(), discharge.contiguous()
        n_time, n_lanes = charge.shape
        output = torch.empty_like(charge)
        state = torch.empty_like(charge)
        grid = (triton.cdiv(n_lanes, LANES_PER_PROGRAM),)
        _forward_kernel[grid](
            charge,
            discharge,
            duration.contiguous(),
            eta_one_way.contiguous(),
            output,
            state,
            n_time,
            n_lanes,
            smoothing,
            block_size=LANES_PER_PROGRAM,
        )
        ctx.save_for_backward(charge, discharge, duration, eta_one_way, state)
        ctx.smoothing = smoothing
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        charge, discharge, duration, eta_one_way, state = ctx.saved_tensors
        n_time, n_lanes = charge.shape
        grad_charge = torch.empty_like(charge)
        grad_discharge = torch.empty_like(charge)
        grad_duration = torch.empty_like(duration)
        grad_eta = torch.empty_like(eta_one_way)
        grid = (triton.cdiv(n_lanes, LANES_PER_PROGRAM),)
        _backward_kernel[grid](
            charge,
            discharge,
            duration.contiguous(),
            eta_one_way.contiguous(),
            state,
            grad_output.contiguous(),
            grad_charge,
            grad_discharge,
            grad_duration,
            grad_eta,
            n_time,
            n_lanes,
            ctx.smoothing,
            block_size=LANES_PER_PROGRAM,
        )
        return grad_charge, grad_discharge, grad_duration, grad_eta, None


def recurrence(
    *,
    charge: torch.Tensor,
    discharge: torch.Tensor,
    duration: torch.Tensor,
    eta_one_way: torch.Tensor,
    smoothing: float,
    return_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the recurrence on the fastest path for the tensors' device.

    Args:
        charge: The charge policy, shape (T, lanes).
        discharge: The discharge policy, shape (T, lanes).
        duration: The usable duration in hours, shape (lanes,).
        eta_one_way: The one-way efficiency, shape (lanes,).
        smoothing: The squared width inside the softmin's square root.
        return_state: Also return the state of charge before each half-hour. On a CUDA device
            this runs the reference loop, because only the output has a reverse pass.

    Returns:
        The per-unit output (T, lanes), and the state of charge or None.
    """
    if charge.is_cuda and triton is not None and not return_state:
        return _KernelRecurrence.apply(charge, discharge, duration, eta_one_way, smoothing), None
    output, state = recurrence_reference(
        charge=charge,
        discharge=discharge,
        duration=duration,
        eta_one_way=eta_one_way,
        smoothing=smoothing,
    )
    return output, state if return_state else None
