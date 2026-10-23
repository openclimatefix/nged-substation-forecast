# ruff: noqa: E501
"""The mixture over 0 to 3 running engine units (arms 3 and 4).

The reading is modelled as the reference forecast's distribution, its spread multiplied by a noise
scale `c`, shifted by `delta - k * g`, where `delta` is the per-row `g * E[k]`, for `k` running units of size `g`. The probability of `k`
follows a multinomial logistic model fitted by expectation-maximisation on residuals that were
resolved at the fit time. The reference distribution is the piecewise-linear cumulative
distribution through its 11 quantiles. Below q01 and above q99 the density decays exponentially,
with a rate set so that the density is continuous at q01 (from the q01-to-q05 slope) and at q99
(from the q95-to-q99 slope).
"""

from dataclasses import dataclass

import e00264_asof as asof
import e00264_common as common
import numpy as np
import polars as pl
from scipy.optimize import minimize

TAIL_MASS: float = 0.01
CHUNK_ROWS: int = 6000
Q50: int = common.QUANTILES.index("q50")


def make_monotone(*, quantiles: np.ndarray) -> np.ndarray:
    """Return the quantile rows made strictly increasing by at least 1e-4 points."""
    out = quantiles.copy()
    for j in range(1, out.shape[1]):
        out[:, j] = np.maximum(out[:, j], out[:, j - 1] + 1e-4)
    return out


def scaled(*, quantiles: np.ndarray, scale: float) -> np.ndarray:
    """Return the quantiles with their distance from q50 multiplied by `scale`."""
    median = quantiles[:, [Q50]]
    return make_monotone(quantiles=median + scale * (quantiles - median))


def cdf_and_density(*, y: np.ndarray, quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the reference distribution's cumulative distribution and density at `y`.

    Args:
        y: Points, shape (n, m).
        quantiles: Strictly increasing quantile rows, shape (n, 11).
    """
    levels = common.LEVELS
    index = (y[:, :, None] >= quantiles[:, None, :]).sum(axis=-1)
    inside = np.clip(index, 1, 10)
    q_lo = np.take_along_axis(quantiles, inside - 1, axis=1)
    q_hi = np.take_along_axis(quantiles, inside, axis=1)
    l_lo, l_hi = levels[inside - 1], levels[inside]
    slope = (l_hi - l_lo) / (q_hi - q_lo)
    cdf = l_lo + (y - q_lo) * slope
    density = slope.copy()
    lower_rate = (levels[1] - levels[0]) / (quantiles[:, [1]] - quantiles[:, [0]]) / TAIL_MASS
    upper_rate = (levels[10] - levels[9]) / (quantiles[:, [10]] - quantiles[:, [9]]) / TAIL_MASS
    below, above = index == 0, index == 11
    lower_cdf = TAIL_MASS * np.exp(np.minimum((y - quantiles[:, [0]]) * lower_rate, 0.0))
    upper_cdf = 1.0 - TAIL_MASS * np.exp(np.minimum(-(y - quantiles[:, [10]]) * upper_rate, 0.0))
    cdf = np.where(below, lower_cdf, np.where(above, upper_cdf, cdf))
    density = np.where(
        below, lower_cdf * lower_rate, np.where(above, (1.0 - upper_cdf) * upper_rate, density)
    )
    return cdf, density


def component_densities(
    *, y: np.ndarray, quantiles: np.ndarray, scale: float, delta: np.ndarray
) -> np.ndarray:
    """Return the density of `y` under each of the four components, shape (n, 4).

    `delta` is the per-row shift in points, so that the mixture mean equals the reference mean.
    """
    base = scaled(quantiles=quantiles, scale=scale)
    shifts = np.arange(common.N_UNITS)[None, :] * common.UNIT_POINTS - delta[:, None]
    points = y[:, None] + shifts
    out = np.empty_like(points)
    for k in range(common.N_UNITS):
        out[:, k] = cdf_and_density(y=points[:, [k]], quantiles=base)[1][:, 0]
    return np.maximum(out, 1e-300)


def row_delta(*, probs: np.ndarray) -> np.ndarray:
    """Return the per-row shift `g * E[k]` that keeps the mixture mean at the reference mean."""
    return common.UNIT_POINTS * (probs * np.arange(common.N_UNITS)).sum(axis=1)


def softmax_probabilities(*, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return class probabilities, shape (n, 4); `weights` has shape (3, p) for classes 1 to 3."""
    logits = np.concatenate([np.zeros((features.shape[0], 1)), features @ weights.T], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _fit_weights(
    *, features: np.ndarray, responsibilities: np.ndarray, start: np.ndarray
) -> np.ndarray:
    """Minimise the ridge-penalised cross-entropy against soft labels; return weights (3, p)."""
    p = features.shape[1]

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        weights = flat.reshape(3, p)
        probs = softmax_probabilities(features=features, weights=weights)
        loss = -(responsibilities * np.log(np.maximum(probs, 1e-300))).sum() / len(features)
        penalty = common.RIDGE * (weights[:, 1:] ** 2).sum() / len(features)
        grad = ((probs - responsibilities).T @ features)[1:] / len(features)
        grad_penalty = (
            2.0
            * common.RIDGE
            * np.concatenate([np.zeros((3, 1)), weights[:, 1:]], axis=1)
            / len(features)
        )
        return loss + penalty, (grad + grad_penalty).ravel()

    result = minimize(
        objective, start.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 60}
    )
    return result.x.reshape(3, p)


@dataclass(frozen=True)
class MixtureFit:
    """A fitted mixture: logistic weights, noise scale, and log-likelihood."""

    weights: np.ndarray
    scale: float
    log_likelihood: float


def fit_mixture(*, y: np.ndarray, quantiles: np.ndarray, features: np.ndarray) -> MixtureFit:
    """Fit the mixture by expectation-maximisation, keeping the noise scale with the best likelihood."""
    best: MixtureFit | None = None
    p = features.shape[1]
    for scale in common.NOISE_SCALES:
        weights = np.zeros((3, p))
        weights[:, 0] = np.log(0.1 / 0.7)
        for _ in range(common.EM_ITERATIONS):
            probs = softmax_probabilities(features=features, weights=weights)
            delta = row_delta(probs=probs)
            densities = component_densities(y=y, quantiles=quantiles, scale=scale, delta=delta)
            joint = probs * densities
            responsibilities = joint / joint.sum(axis=1, keepdims=True)
            weights = _fit_weights(
                features=features, responsibilities=responsibilities, start=weights
            )
        probs = softmax_probabilities(features=features, weights=weights)
        densities = component_densities(
            y=y, quantiles=quantiles, scale=scale, delta=row_delta(probs=probs)
        )
        log_likelihood = float(np.log((probs * densities).sum(axis=1)).sum())
        if best is None or log_likelihood > best.log_likelihood:
            best = MixtureFit(weights=weights, scale=scale, log_likelihood=log_likelihood)
    assert best is not None
    return best


def design_matrix(
    *, valid_ns: np.ndarray, share: np.ndarray, share_fill: float, price: np.ndarray | None
) -> np.ndarray:
    """Return the logistic design matrix: intercept, hour block, season, weekend, 14-day share.

    Arm 4 adds two hindsight price flags as the last two columns.
    """
    hours, seasons = asof.hour_block(valid_ns=valid_ns), asof.season(valid_ns=valid_ns)
    columns = [np.ones(len(valid_ns))]
    columns += [(hours == h).astype(np.float64) for h in (1, 2, 3)]
    columns += [(seasons == s).astype(np.float64) for s in (1, 2, 3)]
    columns.append(asof.weekend(valid_ns=valid_ns).astype(np.float64))
    columns.append(np.where(np.isnan(share), share_fill, share))
    if price is not None:
        columns += [price[:, 0], price[:, 1]]
    return np.stack(columns, axis=1)


def mixture_quantiles(*, quantiles: np.ndarray, probs: np.ndarray, scale: float) -> np.ndarray:
    """Return the mixture's 11 quantiles per row by bisection on its cumulative distribution."""
    out = np.empty((len(quantiles), len(common.LEVELS)))
    for start in range(0, len(quantiles), CHUNK_ROWS):
        part = slice(start, start + CHUNK_ROWS)
        base = scaled(quantiles=quantiles[part], scale=scale)
        weights = probs[part]
        delta = row_delta(probs=weights)[:, None]
        span = base[:, [10]] - base[:, [0]]
        low = np.repeat(
            base[:, [0]] + delta - 3 * common.UNIT_POINTS - 5 * span, len(common.LEVELS), axis=1
        )
        high = np.repeat(base[:, [10]] + delta + 5 * span, len(common.LEVELS), axis=1)
        for _ in range(36):
            middle = 0.5 * (low + high)
            total = np.zeros_like(middle)
            for k in range(common.N_UNITS):
                total += (
                    weights[:, [k]]
                    * cdf_and_density(y=middle - delta + k * common.UNIT_POINTS, quantiles=base)[0]
                )
            go_up = total < common.LEVELS[None, :]
            low = np.where(go_up, middle, low)
            high = np.where(go_up, high, middle)
        out[part] = 0.5 * (low + high)
    return np.maximum.accumulate(out, axis=1)


def mixture_arm(
    *, reference: pl.DataFrame, fallback: pl.DataFrame, use_price: bool, force_visible: bool
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the arm 3 (or arm 4) forecast frame and a per-block log of the fit.

    Args:
        reference: The reference forecast frame from `common.load_reference`.
        fallback: The forecast used for rows whose switch is off (arm 2b), same rows as `reference`.
        use_price: Whether the hindsight price flags enter the logistic model (arm 4).
        force_visible: Whether the visibility switch is forced on (the placebo stress test).

    Returns:
        The forecast frame, with `switch_on` and `p_any` (probability of at least one running
        unit) added, and a block log with the fit time, rows used, noise scale, and the share of
        the block's rows with the switch on. A block with too few resolved rows keeps the reference.
    """
    resolved = asof.resolved_from_reference(reference=reference)
    quantiles = reference.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    origin_ns = reference["power_fcst_init_time"].dt.epoch("ns").to_numpy()
    valid_ns = reference["valid_time"].dt.epoch("ns").to_numpy()
    price_all = (
        np.stack(
            [
                reference[name].to_numpy().astype(np.float64)
                for name in ("spread_positive", "price_p75")
            ],
            axis=1,
        )
        if use_price
        else None
    )
    out = fallback.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    mean = reference["mean"].to_numpy().astype(np.float64)
    switch_on = np.zeros(len(reference), dtype=bool)
    p_any = np.zeros(len(reference))
    log: list[dict] = []
    unique_origins = np.unique(origin_ns)
    states = asof.switch_states(resolved=resolved, origins_ns=unique_origins)
    row_state = lambda ns: np.array([states[int(o)] for o in ns.tolist()])  # noqa: E731
    share14_all = resolved.drop_share(at_ns=origin_ns, window_days=common.SHARE_WINDOW_DAYS)
    switch_all = np.ones(len(reference), dtype=bool) if force_visible else row_state(origin_ns)
    for fit_ns, origins in asof.blocks(origins_ns=unique_origins):
        rows = asof.fit_rows(resolved=resolved, fit_ns=fit_ns)
        applies = np.isin(origin_ns, origins)
        entry = {
            "fit_time": str(np.datetime64(fit_ns, "ns").astype("datetime64[s]")),
            "rows": 0, "scale": None, "p_any_mean": None,
            "switch_share": float(switch_all[applies].mean()),
        }  # fmt: skip
        log.append(entry)
        if len(rows) < common.MIN_FIT_ROWS:
            out[applies] = quantiles[applies]
            continue
        row_share = resolved.drop_share(
            at_ns=resolved.origin_ns[rows], window_days=common.SHARE_WINDOW_DAYS
        )
        keep = (
            np.ones(len(rows), dtype=bool) if force_visible else row_state(resolved.origin_ns[rows])
        )
        if keep.sum() < common.MIN_FIT_ROWS:
            continue
        share_fill = (
            float(np.nanmean(row_share[keep])) if np.isfinite(row_share[keep]).any() else 0.0
        )
        fit_valid = resolved.valid_ns[rows][keep]
        fit_price = (
            np.stack(
                [
                    _price_flags(reference=reference, name=name, valid_ns=fit_valid)
                    for name in ("spread_positive", "price_p75")
                ],
                axis=1,
            )
            if use_price
            else None
        )
        features = design_matrix(
            valid_ns=fit_valid, share=row_share[keep], share_fill=share_fill, price=fit_price
        )
        fit = fit_mixture(
            y=resolved.actual[rows][keep],
            quantiles=resolved.quantiles[rows][keep],
            features=features,
        )
        entry.update(rows=int(keep.sum()), scale=fit.scale)
        live = applies & switch_all
        if live.any():
            pf = None if price_all is None else price_all[live]
            x = design_matrix(
                valid_ns=valid_ns[live], share=share14_all[live], share_fill=share_fill, price=pf
            )
            probs = softmax_probabilities(features=x, weights=fit.weights)
            out[live] = mixture_quantiles(quantiles=quantiles[live], probs=probs, scale=fit.scale)
            switch_on[live] = True
            p_any[live] = 1.0 - probs[:, 0]
            entry["p_any_mean"] = float(p_any[live].mean())
    frame = reference.with_columns(
        [pl.Series(name, out[:, i].astype(np.float32)) for i, name in enumerate(common.QUANTILES)]
        + [
            pl.Series("mean", mean.astype(np.float32)),
            pl.Series("switch_on", switch_on),
            pl.Series("p_any", p_any),
        ]
    )
    return frame, pl.DataFrame(log, infer_schema_length=None)


def _price_flags(*, reference: pl.DataFrame, name: str, valid_ns: np.ndarray) -> np.ndarray:
    """Return the hindsight flag `name` at each valid time (the flag does not depend on the lead)."""
    flags = reference.select("valid_time", name).unique(subset="valid_time").sort("valid_time")
    times = flags["valid_time"].dt.epoch("ns").to_numpy()
    values = flags[name].to_numpy().astype(np.float64)
    return values[np.searchsorted(times, valid_ns)]
