"""The ENC encoder, its training loop, and the arms of experiment 119.

The encoder reads a Primary's 37-cell ENS patch (member mean and standard deviation of 5 fields)
and outputs either 11 signed shifts of GAM-K's quantiles (design a, `kind="c"`) or a temperature
offset and an irradiance multiplier that move the forecast through GAM-K's own sensitivities
(design b, `kind="r"`). The output layer starts at zero, so the untrained encoder reproduces GAM-K
(K0) exactly. Weights are shared across sites; a site sees a cell's offset from its centre cell,
never its coordinates.
"""

from dataclasses import dataclass, field
from typing import Final, Literal

import e00119_common as common
import numpy as np
import torch
from torch import nn

Kind = Literal["c", "r"]
InputMode = Literal["patch", "centre", "distant", "foreign_date"]
N_CHANNELS: Final[int] = 10
N_LEVELS: Final[int] = len(common.QUANTILE_LEVELS)
WIDTH: Final[int] = 16
CELL_DROPOUT: Final[float] = 0.2
LEARNING_RATE: Final[float] = 3e-3
BATCH_SIZE: Final[int] = 4096
MAX_EPOCHS: Final[int] = 60
PATIENCE: Final[int] = 8
SITE_DECAY: Final[float] = 0.5
"""Decoupled weight decay on the per-site offsets, large so the offsets stay near zero."""
TEMPERATURE_RANGE_K: Final[float] = 5.0
IRRADIANCE_RANGE: Final[float] = 0.3
GEOMETRY_SCALE: Final[tuple[float, float, float]] = (0.4, 0.4, 3.0)
DEVICE: Final[str] = "cuda" if torch.cuda.is_available() else "cpu"
LEVELS: Final[torch.Tensor] = torch.tensor(common.QUANTILE_LEVELS, dtype=torch.float32)


@dataclass(frozen=True)
class Arm:
    """One arm: a design and the weather it reads.

    Attributes:
        name: The arm's label (`ENC-c`, `ENC-c0`, `ENC-cps`, `ENC-cpt`, `ENC-r`, `ENC-r0`).
        kind: `c` for the correction, `r` for the replacement.
        mode: `patch` reads the site's own patch, `centre` only its centre cell, `distant` the most
            distant decision site's patch at the same run, and `foreign_date` the same site's patch
            from another run.
    """

    name: str
    kind: Kind
    mode: InputMode


ARMS: Final[dict[str, Arm]] = {
    arm.name: arm
    for arm in (
        Arm("ENC-c", "c", "patch"),
        Arm("ENC-c0", "c", "centre"),
        Arm("ENC-cps", "c", "distant"),
        Arm("ENC-cpt", "c", "foreign_date"),
        Arm("ENC-r", "r", "patch"),
        Arm("ENC-r0", "r", "centre"),
    )
}


@dataclass
class Bundle:
    """The rows and patch tensors on the device.

    Attributes:
        columns: Row tensors: `site`, `run`, `step`, `band`, `lead`, `hod_sin`, `hod_cos`, `mean`,
            `actual`, `q` (rows, 11), `sens_t`, `sens_g`, `src_site`, `src_run`, `month`.
        patch: Float tensor (sites, runs, steps, 37, channels), NaN replaced by 0.
        cell_ok: Bool tensor (sites, runs, steps, 37) of cells with a value.
        geometry: Float tensor (sites, 37, 3), already scaled.
    """

    columns: dict[str, torch.Tensor]
    patch: torch.Tensor
    cell_ok: torch.Tensor
    geometry: torch.Tensor

    def subset(self, index: torch.Tensor) -> Bundle:
        """Return the bundle restricted to some rows; the patch tensors are shared."""
        return Bundle(
            {k: v[index] for k, v in self.columns.items()}, self.patch, self.cell_ok, self.geometry
        )


@dataclass(frozen=True)
class Standardiser:
    """Training-month statistics per lead band; the scored year is transformed with these."""

    centre_mean: torch.Tensor
    centre_std: torch.Tensor
    rel_std: torch.Tensor
    scalar_mean: torch.Tensor
    scalar_std: torch.Tensor


class Encoder(nn.Module):
    """The shared-weight patch encoder."""

    def __init__(self, *, kind: Kind, n_sites: int) -> None:
        """Build the layers; the output layer and the site offsets start at zero."""
        super().__init__()
        self.kind = kind
        self.cell = nn.Sequential(nn.Linear(N_CHANNELS + 3, WIDTH), nn.SiLU())
        self.trunk = nn.Sequential(nn.Linear(WIDTH + N_CHANNELS + 4, WIDTH), nn.SiLU())
        n_out = N_LEVELS if kind == "c" else 2
        self.head = nn.Linear(WIDTH, n_out)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.site_offset = nn.Parameter(torch.zeros(n_sites, n_out))

    def forward(
        self,
        *,
        rel: torch.Tensor,
        geometry: torch.Tensor,
        cell_mask: torch.Tensor,
        centre: torch.Tensor,
        scalars: torch.Tensor,
        site: torch.Tensor,
    ) -> torch.Tensor:
        """Return the raw outputs (rows, 11) for `c` or (rows, 2) for `r`."""
        hidden = self.cell(torch.cat([rel, geometry], dim=-1))
        weight = cell_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        trunk = self.trunk(torch.cat([pooled, centre, scalars], dim=-1))
        return self.head(trunk) + self.site_offset[site]


@dataclass
class FitResult:
    """A fitted encoder with its standardiser and learning curves."""

    model: Encoder
    standardiser: Standardiser
    arm: Arm
    best_epoch: int
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def pinball(*, actual: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss per row over the 11 levels, in points."""
    error = actual.unsqueeze(-1) - quantiles
    levels = LEVELS.to(error.device)
    return torch.maximum(levels * error, (levels - 1.0) * error).mean(dim=-1)


def fit_standardiser(*, bundle: Bundle) -> Standardiser:
    """Return the per-band means and standard deviations of a training bundle."""
    col = bundle.columns
    centre_mean = torch.zeros(4, N_CHANNELS, device=DEVICE)
    centre_std = torch.ones(4, N_CHANNELS, device=DEVICE)
    rel_std = torch.ones(4, N_CHANNELS, device=DEVICE)
    scalars = torch.stack([col["lead"], col["mean"]], dim=-1)
    scalar_mean, scalar_std = scalars.mean(dim=0), scalars.std(dim=0).clamp_min(1e-6)
    for band in range(4):
        rows = (col["band"] == band).nonzero().squeeze(-1)
        if len(rows) == 0:
            continue
        sample = rows[torch.randperm(len(rows), device=DEVICE)[:50000]]
        patch = bundle.patch[col["site"][sample], col["run"][sample], col["step"][sample]]
        centre = patch[:, 0]
        centre_mean[band] = centre.mean(dim=0)
        centre_std[band] = centre.std(dim=0).clamp_min(1e-9)
        rel = patch[:, 1:] - centre[:, None]
        ok = bundle.cell_ok[col["site"][sample], col["run"][sample], col["step"][sample]][:, 1:]
        mean_sq = (rel**2 * ok.unsqueeze(-1)).sum(dim=(0, 1)) / ok.sum().clamp_min(1)
        rel_std[band] = mean_sq.sqrt().clamp_min(1e-9)
    return Standardiser(centre_mean, centre_std, rel_std, scalar_mean, scalar_std)


def encoder_inputs(
    *, bundle: Bundle, std: Standardiser, arm: Arm, drop: bool, generator: torch.Generator | None
) -> dict[str, torch.Tensor]:
    """Return the encoder's tensors for a bundle's rows."""
    col = bundle.columns
    src_site, src_run, step, band = col["src_site"], col["src_run"], col["step"], col["band"]
    patch = bundle.patch[src_site, src_run, step]
    ok = bundle.cell_ok[src_site, src_run, step]
    centre = patch[:, 0]
    rel = (patch[:, 1:] - centre[:, None]) / std.rel_std[band][:, None]
    centre_n = (centre - std.centre_mean[band]) / std.centre_std[band]
    cell_mask = ok[:, 1:]
    if arm.mode == "centre":
        cell_mask = torch.zeros_like(cell_mask)
    if drop:
        keep = (
            torch.rand(cell_mask.shape, device=cell_mask.device, generator=generator) > CELL_DROPOUT
        )
        cell_mask = cell_mask & keep
    geometry = bundle.geometry[src_site][:, 1:]
    scalars = torch.stack([col["lead"], col["mean"]], dim=-1)
    scalars = (scalars - std.scalar_mean) / std.scalar_std
    scalars = torch.cat([scalars, col["hod_sin"][:, None], col["hod_cos"][:, None]], dim=-1)
    return {
        "rel": rel * cell_mask.unsqueeze(-1),
        "geometry": geometry,
        "cell_mask": cell_mask,
        "centre": centre_n,
        "scalars": scalars,
        "site": col["site"].long(),
    }


def apply_outputs(
    *, kind: Kind, raw: torch.Tensor, bundle: Bundle, shrink: float = 1.0
) -> torch.Tensor:
    """Turn raw outputs into 11 quantiles (rows, 11); `shrink` is lambda for the correction."""
    col = bundle.columns
    if kind == "c":
        return torch.sort(col["q"] + shrink * raw, dim=-1).values
    temperature = TEMPERATURE_RANGE_K * torch.tanh(raw[:, 0])
    multiplier = IRRADIANCE_RANGE * torch.tanh(raw[:, 1])
    shift = col["sens_t"] * temperature + col["sens_g"] * multiplier
    return col["q"] + shrink * shift.unsqueeze(-1)


def predict_raw(*, result: FitResult, bundle: Bundle, batch: int = 65536) -> torch.Tensor:
    """Return the raw outputs of a fitted encoder for every row, in batches, without dropout."""
    result.model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(bundle.columns["site"]), batch):
            part = bundle.subset(
                torch.arange(start, min(start + batch, len(bundle.columns["site"])), device=DEVICE)
            )
            inputs = encoder_inputs(
                bundle=part, std=result.standardiser, arm=result.arm, drop=False, generator=None
            )
            out.append(result.model(**inputs))
    return torch.cat(out)


def quantiles_of(*, result: FitResult, bundle: Bundle, shrink: float = 1.0) -> torch.Tensor:
    """Return the fitted arm's 11 quantiles for every row of a bundle."""
    raw = predict_raw(result=result, bundle=bundle)
    return apply_outputs(kind=result.arm.kind, raw=raw, bundle=bundle, shrink=shrink)


def mean_loss(*, result: FitResult, bundle: Bundle, shrink: float = 1.0) -> float:
    """Return the mean pinball loss of a fitted arm on a bundle."""
    quantiles = quantiles_of(result=result, bundle=bundle, shrink=shrink)
    return float(pinball(actual=bundle.columns["actual"], quantiles=quantiles).mean())


def fit(
    *,
    arm: Arm,
    train: Bundle,
    validation: Bundle | None,
    seed: int,
    decay_c: float,
    n_days: int,
    epochs: int | None = None,
) -> FitResult:
    """Fit an arm on a training bundle.

    Args:
        arm: The arm.
        train: The training rows.
        validation: Rows for early stopping, or `None` to train for exactly `epochs`.
        seed: The seed of the weights, the batches, and the cell dropout.
        decay_c: The constant c; the weight decay is `c / n_days`.
        n_days: The number of training days (runs).
        epochs: The number of epochs when there is no validation set.

    Returns:
        The fit, at its best validation epoch (or after `epochs` epochs).
    """
    torch.manual_seed(seed)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    std = fit_standardiser(bundle=train)
    model = Encoder(kind=arm.kind, n_sites=len(common.SCORED_SITES)).to(DEVICE)
    shared = [p for n, p in model.named_parameters() if n != "site_offset"]
    optimiser = torch.optim.AdamW(
        [
            {"params": shared, "weight_decay": decay_c / n_days},
            {"params": [model.site_offset], "weight_decay": SITE_DECAY},
        ],
        lr=LEARNING_RATE,
    )
    n_rows = len(train.columns["site"])
    result = FitResult(model=model, standardiser=std, arm=arm, best_epoch=0)
    best = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    limit = MAX_EPOCHS if validation is not None else int(epochs or MAX_EPOCHS)
    stale = 0
    for epoch in range(1, limit + 1):
        model.train()
        order = torch.randperm(n_rows, device=DEVICE, generator=generator)
        total = torch.zeros((), device=DEVICE)
        for start in range(0, n_rows, BATCH_SIZE):
            part = train.subset(order[start : start + BATCH_SIZE])
            inputs = encoder_inputs(bundle=part, std=std, arm=arm, drop=True, generator=generator)
            quantiles = apply_outputs(kind=arm.kind, raw=model(**inputs), bundle=part)
            loss = pinball(actual=part.columns["actual"], quantiles=quantiles).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += loss.detach() * len(part.columns["site"])
        result.train_loss.append(float(total) / n_rows)
        if validation is None:
            result.best_epoch = epoch
            continue
        value = mean_loss(result=result, bundle=validation)
        result.val_loss.append(value)
        if value < best - 1e-9:
            best, stale, result.best_epoch = value, 0, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if validation is not None:
        model.load_state_dict(best_state)
    model.eval()
    return result


def parameter_count(*, kind: Kind) -> int:
    """Return the number of trainable parameters of an encoder."""
    return sum(p.numel() for p in Encoder(kind=kind, n_sites=len(common.SCORED_SITES)).parameters())


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Return a tensor as a NumPy array on the CPU."""
    return tensor.detach().cpu().numpy()
