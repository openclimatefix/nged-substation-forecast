"""Per-card features F1 to F7 and the change-point estimate of onset and end."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from e00041_common import MVA_SERIES, SERIES_IDS
from e00041_panel import (
    ADJACENCY,
    INDEX,
    Panel,
    excess,
    first_affected,
    last_affected,
    null_steps,
    percentile_of,
    reference,
    step_at,
    window_length,
)


@dataclass
class CardSpec:
    """What a labeller is told about one logged operation."""

    card_id: str
    side_a: int
    side_b: int | None
    close_utc: datetime
    open_utc: datetime
    note: str = ""

    @property
    def first(self) -> int:
        """First half-hour stamp affected by the close."""
        return first_affected(close_utc=self.close_utc)

    @property
    def last(self) -> int:
        """Last half-hour stamp affected before the open."""
        return last_affected(open_utc=self.open_utc)


@dataclass
class Features:
    """Everything computed for a card; `neighbours` is one dict per metered neighbour."""

    values: dict[str, float] = field(default_factory=dict)
    neighbours: list[dict[str, float | int | str]] = field(default_factory=list)


def _r2_and_slope(*, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 6 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return float("nan"), float("nan")
    slope = float(np.polyfit(x[ok], y[ok], 1)[0])
    r = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return r * r, slope


def refine_edge(
    *, panel: Panel, rows: list[int], edge: int, reach: int = 8, window: int = 6
) -> int:
    """Stamp within `reach` of `edge` that maximises the summed absolute step over `rows`."""
    best, best_score = edge, -1.0
    for candidate in range(edge - reach, edge + reach + 1):
        score = 0.0
        for row in rows:
            step = step_at(panel=panel, row=row, edge=candidate, window=window)
            score += 0.0 if np.isnan(step) else abs(step)
        if score > best_score:
            best, best_score = candidate, score
    return best


def _neighbour_row(
    *,
    panel: Panel,
    donor: int,
    other: int,
    exc: dict[int, np.ndarray],
    first: int,
    last: int,
    window: int,
    avoid: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    """F3 and F4 for one (donor, other) pair; fills `exc` for any series not yet in it."""
    lo, hi = first - 48, last + 49
    inside = slice(48, 48 + (last - first + 1))
    for sid in (donor, other):
        if sid not in exc:
            exc[sid] = excess(panel=panel, row=INDEX[sid], lo=lo, hi=hi)
    x, y = exc[donor][inside], exc[other][inside]
    r2_pair, slope = _r2_and_slope(x=-x, y=y)
    ref_donor = reference(panel=panel, row=INDEX[donor], stamps=np.arange(first, last + 1))
    r2_shape, share = _r2_and_slope(x=ref_donor, y=y)
    half = len(y) // 2
    share_1 = _r2_and_slope(x=ref_donor[:half], y=y[:half])[1] if half >= 6 else float("nan")
    share_2 = _r2_and_slope(x=ref_donor[half:], y=y[half:])[1] if half >= 6 else float("nan")
    step_other = step_at(panel=panel, row=INDEX[other], edge=first, window=window)
    step_donor = step_at(panel=panel, row=INDEX[donor], edge=first, window=window)
    null_n = null_steps(
        panel=panel,
        row=INDEX[other],
        edge=first,
        window=window,
        avoid=avoid,
        n_draws=200,
        rng=rng,
    )
    return {
        "donor": donor,
        "other": other,
        "other_unit": "MVA" if other in MVA_SERIES else "MW",
        "donor_close_step_mw": step_donor,
        "other_close_step_mw": step_other,
        "other_close_step_pct": percentile_of(value=step_other, null=null_n),
        "mean_transfer_mw": float(np.nanmean((y - x) / 2)),
        "mean_common_mw": float(np.nanmean((x + y) / 2)),
        "slope_y_on_minus_x": slope,
        "r2_y_on_minus_x": r2_pair,
        "shape_share": share,
        "shape_r2": r2_shape,
        "share_ratio_second_over_first": share_2 / share_1
        if share_1 and not np.isnan(share_1) and share_1 != 0
        else float("nan"),
    }


def _neighbourhood_features(
    *, panel: Panel, spec: CardSpec, first: int, window: int
) -> dict[str, float]:
    """F5: step in the sum of the donor and its metered neighbours, over the largest step."""
    group = [spec.side_a, *ADJACENCY.get(spec.side_a, ())]
    steps = {s: step_at(panel=panel, row=INDEX[s], edge=first, window=window) for s in group}
    total = sum(0.0 if np.isnan(t) else t for t in steps.values())
    largest = max((abs(t) for t in steps.values() if not np.isnan(t)), default=float("nan"))
    return {
        "nbhd_sum_step_mw": total,
        "nbhd_largest_step_mw": largest,
        "nbhd_sum_over_largest": (
            abs(total) / largest if largest and not np.isnan(largest) else float("nan")
        ),
    }


def _fleet_same_sign_share(
    *,
    panel: Panel,
    spec: CardSpec,
    ref_step: float,
    first: int,
    window: int,
    avoid: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """F6: share of the other series whose step has the same sign and exceeds two null SDs."""
    ref_sign = np.sign(ref_step)
    same, counted = 0, 0
    for sid in SERIES_IDS:
        if sid == spec.side_a:
            continue
        step = step_at(panel=panel, row=INDEX[sid], edge=first, window=window)
        if np.isnan(step):
            continue
        counted += 1
        if np.sign(step) == ref_sign and abs(step) > 0:
            draws = null_steps(
                panel=panel,
                row=INDEX[sid],
                edge=first,
                window=window,
                avoid=avoid,
                rng=rng,
                n_draws=60,
            )
            sd = float(np.std(draws)) if len(draws) else float("inf")
            same += abs(step) > 2 * sd
    return same / counted if counted else float("nan")


def _hindsight_features(
    *, panel: Panel, sides: list[int], first: int, last: int, window: int
) -> dict[str, float]:
    """F7: hindsight z-score of the observed power, after minus before the close."""
    out: dict[str, float] = {}
    for tag, sid in zip("ab", sides, strict=False):
        row = INDEX[sid]
        scale = (panel.q90[row] - panel.q10[row]) / 2.56
        z = (panel.obs[row] - panel.q50[row]) / scale
        after = np.nanmean(z[first : first + window])
        before = np.nanmean(z[first - window : first])
        out[f"hindsight_z_{tag}"] = float(after - before)
        out[f"hindsight_mean_z_in_event_{tag}"] = float(np.nanmean(z[first : last + 1]))
    return out


def compute_features(*, panel: Panel, spec: CardSpec, avoid: np.ndarray, seed: int) -> Features:
    """Compute F1 to F7 for one card on `panel` (with any injection already applied)."""
    rng = np.random.default_rng(seed)
    first, last = spec.first, spec.last
    window = window_length(first=first, last=last)
    sides = [spec.side_a] + ([spec.side_b] if spec.side_b in INDEX else [])
    out = Features()
    v = out.values
    v["window_half_hours"] = float(window)
    exc: dict[int, np.ndarray] = {}
    lo, hi = first - 48, last + 49
    for tag, sid in zip("ab", sides, strict=False):
        row = INDEX[sid]
        exc[sid] = excess(panel=panel, row=row, lo=lo, hi=hi)
        close = step_at(panel=panel, row=row, edge=first, window=window)
        opened = step_at(panel=panel, row=row, edge=last + 1, window=window)
        null_c = null_steps(panel=panel, row=row, edge=first, window=window, avoid=avoid, rng=rng)
        null_o = null_steps(
            panel=panel, row=row, edge=last + 1, window=window, avoid=avoid, rng=rng
        )
        v[f"close_step_{tag}_mw"] = close
        v[f"close_step_{tag}_pct"] = percentile_of(value=close, null=null_c)
        v[f"open_step_{tag}_mw"] = opened
        v[f"open_step_{tag}_pct"] = percentile_of(value=opened, null=null_o)
        v[f"null_sd_{tag}_mw"] = float(np.std(null_c)) if len(null_c) else float("nan")
        v[f"ref_mean_{tag}_mw"] = float(
            np.nanmean(reference(panel=panel, row=row, stamps=np.arange(first, last + 1)))
        )
    # F3 and F4 for the logged pair and for every metered neighbour of side a and side b
    partners: list[tuple[int, int]] = []
    for base in sides:
        for nb in ADJACENCY.get(base, ()):
            if (base, nb) not in partners and (nb, base) not in partners:
                partners.append((base, nb))
    if spec.side_b in INDEX and (spec.side_a, spec.side_b) not in partners:
        partners.insert(0, (spec.side_a, spec.side_b))
    for donor, other in partners:
        out.neighbours.append(
            _neighbour_row(
                panel=panel,
                donor=donor,
                other=other,
                exc=exc,
                first=first,
                last=last,
                window=window,
                avoid=avoid,
                rng=rng,
            )
        )
    v.update(_neighbourhood_features(panel=panel, spec=spec, first=first, window=window))
    v["fleet_same_sign_share"] = _fleet_same_sign_share(
        panel=panel,
        spec=spec,
        ref_step=v["close_step_a_mw"],
        first=first,
        window=window,
        avoid=avoid,
        rng=rng,
    )
    v.update(_hindsight_features(panel=panel, sides=sides, first=first, last=last, window=window))
    return out
