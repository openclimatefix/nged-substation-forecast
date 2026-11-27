"""The forward pass of the step-5 neural network and the read-out of its quantiles.

`mlp` and `forward` are ported line by line from the functional reference in each bundle's
`reference_check.py`, applying the stored weights by their `state_dict` names, so the two can be
compared side by side. The forward pass runs in float32 on the CPU; the quantile read-out bisects
each Gaussian mixture's cumulative distribution function in float64.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from substation_demand_forecaster.bundle import Bundle, NetworkConfig
from substation_demand_forecaster.features import ExpectedRows

_LAG_NAMES: Final[tuple[str, str]] = ("day", "week")


@dataclass(frozen=True)
class QuantileForecast:
    """The quantiles of one forecast, as fractions of capacity at the native origin."""

    per_seed: np.ndarray
    """Float32 (seeds, rows, levels), each row sorted into ascending order."""
    mean: np.ndarray
    """Float32 (rows, levels): the seeds' average level by level, each row sorted again."""


def mlp(
    *, w: dict[str, torch.Tensor], prefix: str, x: torch.Tensor, layers: tuple[int, ...]
) -> torch.Tensor:
    """Apply Linear layers named `prefix.<i>` with an exact GELU between them (no dropout)."""
    for k, i in enumerate(layers):
        x = F.linear(x, w[f"{prefix}.{i}.weight"], w[f"{prefix}.{i}.bias"])
        if k < len(layers) - 1:
            x = F.gelu(x)
    return x


def forward(
    *, w: dict[str, torch.Tensor], x: torch.Tensor, primary: torch.Tensor,
    expected: tuple[torch.Tensor, torch.Tensor], config: NetworkConfig,
) -> torch.Tensor:  # fmt: skip
    """Return the nine raw outputs: 3 mixture logits, 3 means, and 3 log standard deviations.

    Args:
        w: One seed's tensors.
        x: Float32 (rows, 60) standardised inputs.
        primary: Int64 (rows,) Primary index of each row.
        expected: The float16 expected-weather row (rows, 28) that each row's day lag and week lag
            read; a zero row for a missing lag.
        config: The bundle's column layout and read-out constants.

    Returns:
        Float32 (rows, 9).
    """
    cols = config.column_slices
    n = len(x)
    emb = w["embedding.weight"][primary]
    weather_in = torch.cat(
        [x[:, cols["weather"][0] : cols["weather"][1]], emb, x[:, cols["b1"][0] : cols["b1"][1]]],
        dim=-1,
    )
    hw = F.gelu(mlp(w=w, prefix="weather", x=weather_in, layers=(0, 3)))
    hd = F.gelu(
        mlp(
            w=w,
            prefix="datetime",
            x=torch.cat([x[:, cols["datetime"][0] : cols["datetime"][1]], emb], dim=-1),
            layers=(0, 3),
        )
    )
    n_weather = cols["weather"][1] - cols["weather"][0]
    outs = []
    for j, name in enumerate(_LAG_NAMES):
        rows = expected[j].float()
        pad = torch.zeros(n, config.n_b1)
        ehw = F.gelu(
            mlp(
                w=w,
                prefix="weather",
                x=torch.cat([rows[:, :n_weather], emb, pad], dim=-1),
                layers=(0, 3),
            )
        )
        ehd = F.gelu(
            mlp(
                w=w,
                prefix="datetime",
                x=torch.cat([rows[:, n_weather:], emb], dim=-1),
                layers=(0, 3),
            )
        )
        lag_id = w["lag_id.weight"][torch.full((n,), j, dtype=torch.long)]
        lag_cols = cols[f"{name}_lag"]
        parts = [x[:, lag_cols[0] : lag_cols[1]], ehw, ehd, emb, lag_id]
        outs.append(mlp(w=w, prefix="lag.0", x=torch.cat(parts, dim=-1), layers=(0, 3)))
    hl = F.gelu(torch.cat(outs, dim=-1))
    core = torch.cat([hw, hd, hl, emb], dim=-1)
    mixer_in = torch.cat(
        [
            core,
            x[:, cols["spread"][0] : cols["spread"][1]],
            x[:, cols["lead"][0] : cols["lead"][1]],
        ],
        dim=-1,
    )
    out = mlp(w=w, prefix="mixer_ft", x=mixer_in, layers=(0, 3, 6))
    knots = torch.cat([w["knots_old"], w["knot_new"][:, None, :]], dim=1)
    n_knots = knots.shape[1]
    tau = x[:, config.knot_position_column].clamp(0.0, float(n_knots - 1))
    lo = tau.floor().long().clamp(max=n_knots - 2)
    frac = (tau - lo.to(tau.dtype))[:, None]
    offset = (1.0 - frac) * knots[primary, lo] + frac * knots[primary, lo + 1]
    k = config.mixture_components
    return torch.cat([out[:, :k], out[:, k : 2 * k] + offset, out[:, 2 * k :]], dim=-1)


def mixture_quantiles(
    *, raw: torch.Tensor, levels: tuple[float, ...], config: NetworkConfig
) -> torch.Tensor:
    """Return the standardised quantiles of each row's mixture, by bisection on its CDF in float64.

    Args:
        raw: Float32 (rows, 9) `forward` outputs.
        levels: Quantile levels strictly between 0 and 1.
        config: The bundle's mixture and bisection constants.

    Returns:
        Float64 (rows, levels), in the target's standardised units.
    """
    k = config.mixture_components
    out = raw.double()
    log_w = torch.log_softmax(out[:, :k], dim=-1)
    mean = out[:, k : 2 * k]
    lo_s, hi_s = config.log_std_clamp
    std = torch.exp(out[:, 2 * k :].clamp(lo_s, hi_s))
    level_tensor = torch.tensor(levels, dtype=torch.float64)
    w = log_w.exp()
    limit = config.bisection_range
    lo = torch.full((len(w), len(levels)), -float(limit), dtype=torch.float64)
    hi = torch.full_like(lo, float(limit))
    for _ in range(config.bisection_steps):
        mid = 0.5 * (lo + hi)
        z = (mid[..., None] - mean[:, None, :]) / (std[:, None, :] * float(np.sqrt(2.0)))
        cdf = (w[:, None, :] * 0.5 * (1.0 + torch.erf(z))).sum(-1)
        below = cdf < level_tensor
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


def _expected_for_lags(
    *, x: np.ndarray, primary: np.ndarray, expected_rows: ExpectedRows, config: NetworkConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the expected-weather row each row's day lag and week lag read, float16 (rows, 28).

    A missing lag (grid index below 0) reads a zero row.

    Args:
        x: Float32 (rows, 60) standardised inputs, holding the lags' grid indices.
        primary: Int64 (rows,) Primary index of each row.
        expected_rows: `build_expected_rows`' output.
        config: The bundle's column layout.

    Returns:
        The day lag's rows and the week lag's rows.

    Raises:
        ValueError: If a lag's half-hour has no expected-weather row, which is our own bug.
    """
    lookup = {
        (int(p), int(g)): i
        for i, (p, g) in enumerate(
            zip(expected_rows.primary_index, expected_rows.grid_index, strict=True)
        )
    }
    width = expected_rows.standardised.shape[1]
    padded = np.concatenate([expected_rows.standardised, np.zeros((1, width), dtype=np.float16)])
    out = []
    for column in config.lag_grid_index_columns:
        grid = x[:, column].astype(np.int64)
        try:
            index = np.array(
                [
                    lookup[(int(p), int(g))] if g >= 0 else len(padded) - 1
                    for p, g in zip(primary, grid, strict=True)
                ],
                dtype=np.int64,
            )
        except KeyError as missing:
            msg = f"no expected-weather row for (primary, grid index) {missing.args[0]}"
            raise ValueError(msg) from missing
        out.append(torch.from_numpy(padded[index]))
    return out[0], out[1]


def forecast(
    *, bundle: Bundle, matrix: np.ndarray, primary_index: np.ndarray,
    expected_rows: ExpectedRows, levels: tuple[float, ...],
) -> QuantileForecast:  # fmt: skip
    """Return the quantiles of every seed and of the seed average, as fractions of capacity.

    Each seed's quantiles are converted from the standardised target with the Primary's target
    mean and standard deviation, then sorted. The seeds are averaged level by level and each row is
    sorted again.

    Args:
        bundle: The bundle's weights and standardisers.
        matrix: Float32 (rows, 60) `InputMatrix.standardised`.
        primary_index: Int64 (rows,) Primary index of each row.
        expected_rows: `build_expected_rows`' output for the same forecast.
        levels: The quantile levels to read out.

    Returns:
        The quantiles of every seed and of the seed average.
    """
    config = bundle.network
    expected = _expected_for_lags(
        x=matrix, primary=primary_index, expected_rows=expected_rows, config=config
    )
    target_mean = torch.from_numpy(bundle.standardisers.target_mean[primary_index])[:, None]
    target_std = torch.from_numpy(bundle.standardisers.target_std[primary_index])[:, None]
    x = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))
    primary = torch.from_numpy(primary_index.astype(np.int64))
    per_seed = []
    with torch.inference_mode():
        for weights in bundle.weights:
            raw = forward(w=weights, x=x, primary=primary, expected=expected, config=config)
            quantiles = mixture_quantiles(raw=raw, levels=levels, config=config)
            per_seed.append(
                torch.sort(quantiles * target_std + target_mean, dim=1).values.float().numpy()
            )
    stacked = np.stack(per_seed)
    mean = np.sort(np.mean(stacked, axis=0, dtype=np.float32), axis=1)
    return QuantileForecast(per_seed=stacked, mean=mean)
