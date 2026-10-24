# ruff: noqa: E501
"""The four neural networks of experiment 334: the MDN (M1, M2), the GRU (R), and a compact TFT (T).

All four predict the site-standardised power `z = (power - mean) / spread` of every lead step.
M1, M2, and R end in a three-component Gaussian mixture (`mixture_*` helpers); T ends in 11 quantiles.
"""

from typing import Final

import numpy as np
import torch
from scipy.special import ndtr
from torch import nn

COMPONENTS: Final[int] = 3
Z_LIMIT: Final[float] = 40.0
LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99)


def mixture_head_bias() -> list[float]:
    """Return the output bias that starts the mixture as three components at -0.5, 0, +0.5."""
    return [0.0] * 3 + [-0.5, 0.0, 0.5] + [0.0] * 3


def mixture_parts(*, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split raw outputs (..., 9) into log weights, means, and standard deviations."""
    k = COMPONENTS
    return (
        torch.log_softmax(out[..., :k], dim=-1),
        out[..., k : 2 * k],
        torch.exp(out[..., 2 * k :].clamp(-3.0, 3.0)),
    )


def mixture_nll(*, out: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Return the negative log-likelihood of `z` (shape of `out` without its last axis) per element."""
    log_w, mean, std = mixture_parts(out=out)
    log_pdf = -0.5 * ((z[..., None] - mean) / std) ** 2 - torch.log(std)
    return -torch.logsumexp(log_w + log_pdf, dim=-1) + 0.5 * float(np.log(2 * np.pi))


def mixture_quantiles(*, out: torch.Tensor) -> np.ndarray:
    """Return the 11 quantiles of the mixture in standardised units, by bisection on its CDF."""
    with torch.no_grad():
        log_w, mean, std = mixture_parts(out=out.detach().float().cpu())
    w, mean, std = (t.numpy().astype(np.float64) for t in (log_w.exp(), mean, std))
    result = np.empty((len(w), len(LEVELS)))
    for k, level in enumerate(LEVELS):
        lo = np.full(len(w), -Z_LIMIT)
        hi = np.full(len(w), Z_LIMIT)
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            cdf = (w * ndtr((mid[:, None] - mean) / std)).sum(axis=1)
            below = cdf < level
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        result[:, k] = 0.5 * (lo + hi)
    return result


def pinball_loss(*, q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss over the 11 levels per element, `q` of shape (..., 11)."""
    levels = torch.tensor(LEVELS, dtype=q.dtype, device=q.device)
    error = z[..., None] - q
    return torch.maximum(levels * error, (levels - 1.0) * error).mean(dim=-1)


class MixtureMlp(nn.Module):
    """M1 and M2: two hidden layers of 64 units and a mixture output, one row at a time."""

    def __init__(self, *, n_inputs: int, hidden: int = 64) -> None:
        """Build the network; the output layer starts at zero weights."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3 * COMPONENTS),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor(mixture_head_bias()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw mixture outputs."""
        return self.net(x)

    def loss(self, *, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return the mean negative log-likelihood."""
        return mixture_nll(out=self(x), z=z).mean()


class Seq2SeqGru(nn.Module):
    """R: a GRU over the power history, then a GRU over the lead steps that starts from its state."""

    def __init__(self, *, n_history: int, n_known: int, hidden: int = 64) -> None:
        """Build the encoder, decoder, and mixture head."""
        super().__init__()
        self.encoder = nn.GRU(n_history, hidden, batch_first=True)
        self.decoder = nn.GRU(n_known, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3 * COMPONENTS)
        )
        last = self.head[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor(mixture_head_bias()))

    def forward(self, *, hist: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        """Return raw mixture outputs of shape (runs, steps, 9)."""
        _, state = self.encoder(hist)
        out, _ = self.decoder(known, state)
        return self.head(out)

    def loss(
        self, *, hist: torch.Tensor, known: torch.Tensor, z: torch.Tensor, keep: torch.Tensor
    ) -> torch.Tensor:
        """Return the mean negative log-likelihood over kept steps."""
        nll = mixture_nll(out=self(hist=hist, known=known), z=z)
        return (nll * keep).sum() / keep.sum().clamp(min=1.0)


class GatedResidualNetwork(nn.Module):
    """The TFT's gated residual network (Lim et al. 2021, figure 2), with an optional context."""

    def __init__(
        self,
        *,
        d_in: int,
        d: int,
        d_out: int | None = None,
        d_context: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Build the network."""
        super().__init__()
        d_out = d_out or d
        self.fc1 = nn.Linear(d_in, d)
        self.context = nn.Linear(d_context, d, bias=False) if d_context else None
        self.fc2 = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d, 2 * d_out)
        self.skip = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """Return the transformed input."""
        h = self.fc1(x)
        if self.context is not None and context is not None:
            h = h + self.context(context)
        h = self.fc2(nn.functional.elu(h))
        gated = nn.functional.glu(self.gate(self.dropout(h)), dim=-1)
        return self.norm(self.skip(x) + gated)


class VariableSelection(nn.Module):
    """The TFT's variable selection: softmax weights over per-variable embeddings, then one shared network.

    Departure from the paper (stated in the write-up): one gated residual network processes the
    weighted sum, not one network per variable, which cuts the cost of the 33 known inputs.
    """

    def __init__(self, *, n_vars: int, d: int, d_context: int, dropout: float) -> None:
        """Build the embeddings, the weight network, and the shared network."""
        super().__init__()
        self.embed = nn.Parameter(torch.randn(n_vars, d) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n_vars, d))
        self.weights = GatedResidualNetwork(
            d_in=n_vars * d, d=d, d_out=n_vars, d_context=d_context, dropout=dropout
        )
        self.shared = GatedResidualNetwork(d_in=d, d=d, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return the selected embedding; `x` is (batch, steps, n_vars), `context` (batch, d)."""
        emb = x[..., None] * self.embed + self.bias
        weights = torch.softmax(self.weights(emb.flatten(-2), context[:, None, :]), dim=-1)
        return self.shared((emb * weights[..., None]).sum(dim=-2))


class TemporalFusionTransformer(nn.Module):
    """T: a compact temporal fusion transformer with an 11-level quantile output."""

    def __init__(
        self,
        *,
        n_static: int,
        n_past: int,
        n_known: int,
        d: int = 16,
        dropout: float = 0.1,
        history: int = 336,
        steps: int = 480,
    ) -> None:
        """Build static encoders, variable selection, the LSTM encoder-decoder, attention, and output."""
        super().__init__()
        self.d, self.history, self.steps = d, history, steps
        self.static = GatedResidualNetwork(d_in=n_static, d=d, dropout=dropout)
        self.contexts = nn.ModuleList(
            GatedResidualNetwork(d_in=d, d=d, dropout=dropout) for _ in range(4)
        )
        self.select_past = VariableSelection(n_vars=n_past, d=d, d_context=d, dropout=dropout)
        self.select_known = VariableSelection(n_vars=n_known, d=d, d_context=d, dropout=dropout)
        self.encoder = nn.LSTM(d, d, batch_first=True)
        self.decoder = nn.LSTM(d, d, batch_first=True)
        self.lstm_gate = nn.Linear(d, 2 * d)
        self.lstm_norm = nn.LayerNorm(d)
        self.enrich = GatedResidualNetwork(d_in=d, d=d, d_context=d, dropout=dropout)
        self.attention = nn.MultiheadAttention(d, num_heads=1, dropout=dropout, batch_first=True)
        self.attn_gate = nn.Linear(d, 2 * d)
        self.attn_norm = nn.LayerNorm(d)
        self.position = GatedResidualNetwork(d_in=d, d=d, dropout=dropout)
        self.out_gate = nn.Linear(d, 2 * d)
        self.out_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, len(LEVELS))
        mask = torch.ones(steps, history + steps, dtype=torch.bool)
        for i in range(steps):
            mask[i, : history + i + 1] = False
        self.register_buffer("attn_mask", mask)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.copy_(
                torch.tensor(
                    [float(torch.distributions.Normal(0, 1).icdf(torch.tensor(q))) for q in LEVELS]
                )
            )

    def forward(
        self, *, static: torch.Tensor, past: torch.Tensor, known: torch.Tensor
    ) -> torch.Tensor:
        """Return quantiles of shape (runs, steps, 11)."""
        c_select, c_enrich, c_h, c_c = (net(self.static(static)) for net in self.contexts)
        past_in = self.select_past(past, c_select)
        known_in = self.select_known(known, c_select)
        state = (c_h[None], c_c[None])
        enc_out, state = self.encoder(past_in, state)
        dec_out, _ = self.decoder(known_in, state)
        lstm_out = torch.cat([enc_out, dec_out], dim=1)
        skip = torch.cat([past_in, known_in], dim=1)
        lstm_out = self.lstm_norm(skip + nn.functional.glu(self.lstm_gate(lstm_out), dim=-1))
        enriched = self.enrich(lstm_out, c_enrich[:, None, :])
        queries = enriched[:, self.history :]
        attended, _ = self.attention(
            queries, enriched, enriched, attn_mask=self.attn_mask, need_weights=False
        )
        attended = self.attn_norm(queries + nn.functional.glu(self.attn_gate(attended), dim=-1))
        refined = self.position(attended)
        final = self.out_norm(
            lstm_out[:, self.history :] + nn.functional.glu(self.out_gate(refined), dim=-1)
        )
        return self.output(final)
