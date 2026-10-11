"""Estimate a plant's capacity with a variational physics model fitted by gradient descent (DP1).

DP1 is the roadmap's Candidate B, cut down to what the study needs. The plant's tilt, azimuth, and
alternating-current limit are fixed from its orientation fit, and the model estimates a
piecewise-constant weekly direct-current capacity trace. The power is the direct-current output of
the satellite-driven physics model, softly clipped at the alternating-current limit. The loss is a
pinball loss on the reconstruction, which makes the trace follow the upper envelope of the data so
that curtailment does not drag it down, plus the Kullback-Leibler divergence of a mean-field normal
posterior over each week's logarithmic capacity from a wide normal prior, plus a total-variation
penalty on the weekly means.

**Every refit reads the latest 63 days (nine weeks), and the estimate is the last week's capacity.**
The refits happen every seventh day and warm-start from the previous refit's posterior, so the
estimate for day `d` is the latest refit that ended on or before `d`, and it can be up to 6 days
old. Every plant and every record (the original and each planted draw) fits in one batch, so a
refit takes a few milliseconds on a GPU. Nothing outside `packages/studies` may import torch.
"""

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from studies.capacity_estimators import CapacityRecord, RawEstimate, valid_days

WINDOW_DAYS: Final[int] = 63
"""Each refit reads this many days (nine weeks) ending on the refit day."""

WEEKS: Final[int] = 9
"""The number of weekly capacity blocks in a window."""

REFIT_EVERY_DAYS: Final[int] = 7
"""A refit happens on every seventh day."""

MIN_UNIT_OUTPUT: Final[float] = 0.15
"""Hours whose modelled output per unit of direct-current capacity is lower are left out."""

MIN_VALID_DAYS: Final[int] = 20
"""A window needs this many valid days for its estimate to count."""

PRIOR_SD: Final[float] = 0.5
"""The prior standard deviation of a week's logarithmic capacity around the fitted capacity."""

CLIP_SHARPNESS: Final[float] = 50.0
"""The soft clip's sharpness as a multiple of one over the alternating-current limit."""

DP1_DEFAULT_PARAMS: Final[Mapping[str, float]] = {
    "quantile": 0.7,
    "total_variation_weight": 10.0,
    "learning_rate": 0.05,
    "samples": 4.0,
    "first_steps": 300.0,
    "later_steps": 80.0,
}
"""The settings the tuning draw starts from."""


def soft_clip(*, power: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    """Return a differentiable `min(power, limit)`, linear well below the limit.

    Args:
        power: The unclipped power.
        limit: The alternating-current limit, broadcastable against `power`.

    Returns:
        The softly clipped power.
    """
    beta = CLIP_SHARPNESS / limit
    return limit - F.softplus(beta * (limit - power)) / beta


class _Windows:
    """The batched data of every plant and record, with the window a refit reads."""

    def __init__(
        self,
        *,
        records: Sequence[CapacityRecord],
        dc_prior: np.ndarray,
        ac_limit: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Stack the records' inputs into tensors of shape (batch, days, 24).

        Args:
            records: The records, which share dates, plants, and static power.
            dc_prior: Each plant's fitted direct-current capacity over its static power.
            ac_limit: Each plant's fitted alternating-current limit over its static power.
            device: Where the tensors live.
            dtype: Their floating-point type.
        """
        plants = len(records[0].sites)
        static = records[0].static_p99_mw[:, None, None]
        power = np.stack([r.power_mw / static for r in records])
        shape = np.stack([r.cams_shape for r in records])
        valid = np.stack([valid_days(record=r) for r in records])
        usable = (
            valid[..., None] & np.isfinite(power) & np.isfinite(shape) & (shape >= MIN_UNIT_OUTPUT)
        )
        days = power.shape[2]
        self.records = len(records)
        self.plants = plants
        self.days = days
        self.power = torch.as_tensor(
            np.nan_to_num(power).reshape(-1, days, 24), device=device, dtype=dtype
        )
        self.shape = torch.as_tensor(
            np.nan_to_num(shape).reshape(-1, days, 24), device=device, dtype=dtype
        )
        self.mask = torch.as_tensor(usable.reshape(-1, days, 24), device=device)
        self.valid = torch.as_tensor(valid.reshape(-1, days), device=device)
        self.dc_prior = torch.as_tensor(np.tile(dc_prior, self.records), device=device, dtype=dtype)
        self.ac_limit = torch.as_tensor(np.tile(ac_limit, self.records), device=device, dtype=dtype)

    def window(
        self, *, end_day: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """Return the tensors of the 63 days ending at `end_day` and the valid days in each.

        Args:
            end_day: The window's last day.

        Returns:
            The power, the modelled output, the mask, and the number of valid days per batch entry.
        """
        days = slice(end_day - WINDOW_DAYS + 1, end_day + 1)
        valid_count = self.valid[:, days].sum(dim=1).cpu().numpy()
        return self.power[:, days], self.shape[:, days], self.mask[:, days], valid_count


def _fit_window(
    *,
    windows: _Windows,
    end_day: int,
    posterior: tuple[torch.Tensor, torch.Tensor],
    params: Mapping[str, float],
    steps: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Fit one window for every batch entry by gradient descent on the loss described above.

    Args:
        windows: The batched data.
        end_day: The window's last day.
        posterior: The warm start: the weekly means and the raw standard deviations, each of shape
            (batch, `WEEKS`).
        params: `quantile`, `total_variation_weight`, `learning_rate`, and `samples`.
        steps: The number of Adam steps.
        generator: The random generator for the reparameterisation noise, drawn once per plant
            and shared across records.

    Returns:
        The fitted means, the fitted raw standard deviations, and the valid days in each window.
    """
    power, shape, mask, valid_count = windows.window(end_day=end_day)
    mean = posterior[0].detach().clone().requires_grad_()
    raw_sd = posterior[1].detach().clone().requires_grad_()
    optimiser = torch.optim.Adam([mean, raw_sd], lr=params["learning_rate"])
    quantile = params["quantile"]
    samples = int(params["samples"])
    week_of_day = torch.arange(WINDOW_DAYS, device=power.device) // REFIT_EVERY_DAYS
    dc = windows.dc_prior[:, None, None]
    limit = windows.ac_limit[:, None, None]
    for _ in range(steps):
        optimiser.zero_grad()
        sd = F.softplus(raw_sd)
        # Common random numbers: one draw per (sample, plant, week), shared by every record in the
        # batch, so that two records with the same data get the same estimate and a difference
        # between the original and a planted record is not Monte Carlo noise.
        noise = torch.randn(
            (samples, windows.plants, WEEKS),
            generator=generator,
            device=mean.device,
            dtype=mean.dtype,
        ).repeat(1, windows.records, 1)
        log_capacity = mean[None] + sd[None] * noise
        per_day = torch.exp(log_capacity)[:, :, week_of_day]
        predicted = soft_clip(power=dc[None] * per_day[..., None] * shape[None], limit=limit[None])
        residual = power[None] - predicted
        pinball = torch.where(residual >= 0, quantile * residual, (quantile - 1.0) * residual)
        reconstruction = (pinball * mask[None]).sum(dim=(2, 3)).mean(dim=0)
        divergence = (
            (sd**2 + mean**2) / (2 * PRIOR_SD**2) - torch.log(sd) + np.log(PRIOR_SD) - 0.5
        ).sum(dim=1)
        variation = params["total_variation_weight"] * (mean[:, 1:] - mean[:, :-1]).abs().sum(dim=1)
        (reconstruction + divergence + variation).sum().backward()
        optimiser.step()
    return mean.detach(), raw_sd.detach(), valid_count


def run_dp1(
    *,
    records: Sequence[CapacityRecord],
    dc_prior: np.ndarray,
    ac_limit: np.ndarray,
    params: Mapping[str, float] | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> list[RawEstimate]:
    """Run DP1 over every record at once, refitting every seventh day.

    Args:
        records: The original record and any planted records. They share dates, plants, and static
            power.
        dc_prior: Each plant's fitted direct-current capacity over its static power.
        ac_limit: Each plant's fitted alternating-current limit over its static power.
        params: The settings, or `None` for `DP1_DEFAULT_PARAMS`.
        device: The torch device.
        dtype: The floating-point type.
        seed: The seed of the reparameterisation noise.

    Returns:
        One raw estimate per record. The level is the last week's direct-current capacity over the
        static power (dimensionless), the standard deviation its posterior's, and both are held
        from each refit to the next. Days before the first full window are NaN.
    """
    settings = DP1_DEFAULT_PARAMS if params is None else params
    torch_device = torch.device(device)
    windows = _Windows(
        records=records, dc_prior=dc_prior, ac_limit=ac_limit, device=torch_device, dtype=dtype
    )
    batch = windows.records * windows.plants
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(seed)
    mean = torch.zeros((batch, WEEKS), device=torch_device, dtype=dtype)
    raw_sd = torch.full((batch, WEEKS), 0.0, device=torch_device, dtype=dtype)
    level = np.full((batch, windows.days), np.nan)
    sd = np.full((batch, windows.days), np.nan)
    refit_days = list(range(WINDOW_DAYS - 1, windows.days, REFIT_EVERY_DAYS))
    for index, end_day in enumerate(refit_days):
        steps = int(settings["first_steps" if index == 0 else "later_steps"])
        mean, raw_sd, valid_count = _fit_window(
            windows=windows,
            end_day=end_day,
            posterior=(mean, raw_sd),
            params=settings,
            steps=steps,
            generator=generator,
        )
        last_mean = mean[:, -1].cpu().numpy().astype(np.float64)
        last_sd = F.softplus(raw_sd[:, -1]).cpu().numpy().astype(np.float64)
        capacity = np.exp(last_mean) * windows.dc_prior.cpu().numpy()
        enough = valid_count >= MIN_VALID_DAYS
        until = refit_days[index + 1] if index + 1 < len(refit_days) else windows.days
        level[:, end_day:until] = np.where(enough, capacity, np.nan)[:, None]
        sd[:, end_day:until] = np.where(enough, capacity * last_sd, np.nan)[:, None]
        # The next window moves by one week: shift the blocks left and repeat the newest.
        mean = torch.cat([mean[:, 1:], mean[:, -1:]], dim=1)
        raw_sd = torch.cat([raw_sd[:, 1:], raw_sd[:, -1:]], dim=1)
    shaped_level = level.reshape(windows.records, windows.plants, windows.days)
    shaped_sd = sd.reshape(windows.records, windows.plants, windows.days)
    return [RawEstimate(level=shaped_level[r], sd=shaped_sd[r]) for r in range(windows.records)]
