"""Scores of a detector's posterior against an injection: recall, precision, energy, intervals."""

from typing import Any

import numpy as np
from e00199_common import UNIT_MW
from e00199_inject import Injection

THRESHOLD = 0.5
N_RESAMPLES = 4000


def flagged(*, smoothed: np.ndarray) -> np.ndarray:
    """True where the smoothed posterior P(k at least 1) exceeds 0.5."""
    return smoothed[:, 1:].sum(axis=1) > THRESHOLD


def score(
    *, inj: Injection, smoothed: np.ndarray, filtered: np.ndarray, g: float, seed: int = 0
) -> dict[str, Any]:
    """Pooled scores over the scored half-hours of `inj`, with day-resampled intervals."""
    sc = inj.scored
    truth = inj.truth_units[sc]
    flag = flagged(smoothed=smoothed)[sc]
    flag_f = flagged(smoothed=filtered)[sc]
    ek = (smoothed[sc] * np.arange(4)).sum(axis=1)
    injected = np.nan_to_num(inj.output[sc])
    days = (np.flatnonzero(sc) // 48) + 100000 * inj.window_id[sc]

    def stats(idx: np.ndarray) -> tuple[float, float, float, float, float]:
        t, f = truth[idx], flag[idx]
        r1 = f[t == 1].mean() if (t == 1).any() else np.nan
        r23 = f[t >= 2].mean() if (t >= 2).any() else np.nan
        prec = (t[f] >= 1).mean() if f.any() else np.nan
        e_err = (ek[idx] * g).sum() / max(injected[idx].sum(), 1e-9) - 1.0
        return r1, r23, prec, e_err, f.mean()

    base = stats(np.arange(len(truth)))
    rng = np.random.default_rng(seed)
    order = np.argsort(days, kind="stable")
    sorted_days = days[order]
    uniq, first = np.unique(sorted_days, return_index=True)
    bounds = [*list(first), len(order)]
    groups = [order[bounds[i] : bounds[i + 1]] for i in range(len(uniq))]
    draws = []
    for _ in range(N_RESAMPLES):
        pick = rng.integers(len(groups), size=len(groups))
        draws.append(stats(np.concatenate([groups[i] for i in pick])))
    draws = np.array(draws)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    names = ("recall_1_unit", "recall_2_or_3_units", "precision", "energy_error", "flagged_share")
    out: dict[str, Any] = {
        n: {"value": float(v), "lo": float(a), "hi": float(b)}
        for n, v, a, b in zip(names, base, lo, hi, strict=True)
    }
    out["recall_filtered_any_running"] = float(flag_f[truth >= 1].mean())
    out["exact_k_accuracy"] = float((smoothed[sc].argmax(axis=1) == truth).mean())
    out["n_scored"] = int(sc.sum())
    out["n_units"] = {int(k): int((truth == k).sum()) for k in range(4)}
    out["unit_size_mw"] = UNIT_MW
    return out


def passes(*, scores: dict) -> dict[str, bool]:
    """The pre-registered thresholds of the decision rule."""
    return {
        "recall_2_or_3_units_ge_0.80": scores["recall_2_or_3_units"]["value"] >= 0.80,
        "recall_1_unit_ge_0.50": scores["recall_1_unit"]["value"] >= 0.50,
        "precision_ge_0.80": scores["precision"]["value"] >= 0.80,
        "energy_within_15pct": abs(scores["energy_error"]["value"]) <= 0.15,
    }
