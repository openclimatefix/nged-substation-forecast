"""The rules-only labeller: thresholds on the model-free features, tuned on dev cards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from e00041_features import CardSpec, Features, refine_edge
from e00041_panel import INDEX, Panel, excess


@dataclass(frozen=True)
class RuleThresholds:
    """Percentile bars on the step features; values are tuned on the development cards."""

    close_pct: float = 0.97
    open_pct: float = 0.90
    one_sided_close_pct: float = 0.985
    ratio_low: float = 0.35
    ratio_high: float = 2.8
    min_in_event_z: float = 0.0


DEFAULT: Final[RuleThresholds] = RuleThresholds()


def _num(x: float | None) -> float:
    return float(x) if x is not None else float("nan")


def _side_steps(*, spec: CardSpec, features: Features) -> dict[int, dict[str, float]]:
    """Close and open steps, with their null percentiles, for each logged side."""
    v = features.values
    sides = [spec.side_a] + ([spec.side_b] if spec.side_b in INDEX else [])
    steps = {}
    for tag, sid in zip("ab", sides, strict=False):
        steps[sid] = {
            "close": _num(v.get(f"close_step_{tag}_mw")),
            "close_pct": _num(v.get(f"close_step_{tag}_pct")),
            "open": _num(v.get(f"open_step_{tag}_mw")),
            "open_pct": _num(v.get(f"open_step_{tag}_pct")),
        }
    return steps


def _receivers(*, features: Features, donor: int) -> list[dict]:
    """Metered neighbours of `donor` with a positive close step and a confident percentile."""
    chosen = [
        n
        for n in features.neighbours
        if n["donor"] == donor
        and float(n["other_close_step_mw"]) > 0
        and float(n["other_close_step_pct"]) >= 0.95
    ]
    total = sum(float(n["other_close_step_mw"]) for n in chosen)
    if not total:
        return []
    return [
        {"series": n["other"], "share": float(n["other_close_step_mw"]) / total} for n in chosen
    ]


def label_card(
    *, panel: Panel, spec: CardSpec, features: Features, thresholds: RuleThresholds = DEFAULT
) -> dict:
    """Return a label row with the schema of the Sonnet labeller (without evidence text)."""
    steps = _side_steps(spec=spec, features=features)
    if all(np.isnan(s["close_pct"]) for s in steps.values()):
        return {"card_id": spec.card_id, "visibility": "unassessable", "confidence": "low"}

    def reverses(s: dict) -> bool:
        return (
            (not np.isnan(s["open"]))
            and np.sign(s["open"]) == -np.sign(s["close"])
            and s["open_pct"] >= thresholds.open_pct
        )

    strong = [
        sid for sid, s in steps.items() if s["close_pct"] >= thresholds.close_pct and reverses(s)
    ]
    visible_sides = [
        sid
        for sid, s in steps.items()
        if s["close_pct"] >= thresholds.one_sided_close_pct and reverses(s)
    ]
    conserved = False
    if len(strong) == 2:
        a, b = (steps[s]["close"] for s in strong)
        conserved = (
            np.sign(a) != np.sign(b) and thresholds.ratio_low <= abs(a / b) <= thresholds.ratio_high
        )
    if conserved:
        visibility = "conserved"
    elif visible_sides:
        visibility = "one_sided"
    else:
        visibility = "not_visible"
    row: dict = {"card_id": spec.card_id, "visibility": visibility, "confidence": "med"}
    if visibility == "not_visible":
        return row
    use = strong if conserved else visible_sides
    rows_used = [INDEX[s] for s in use]
    onset = refine_edge(panel=panel, rows=rows_used, edge=spec.first)
    end = refine_edge(panel=panel, rows=rows_used, edge=spec.last + 1) - 1
    if end <= onset:
        onset, end = spec.first, spec.last
    exc = {s: excess(panel=panel, row=INDEX[s], lo=onset, hi=end + 1) for s in use}
    if conserved:
        donor = min(use, key=lambda s: steps[s]["close"])
        other = next(s for s in use if s != donor)
        series = (exc[other] - exc[donor]) / 2
        receivers = [{"series": other, "share": 1.0}]
    else:
        donor = use[0] if steps[use[0]]["close"] < 0 else None
        series = np.abs(exc[use[0]])
        receivers = [{"series": use[0], "share": 1.0}] if donor is None else []
    # name any metered neighbour of the donor with a positive close step and a confident percentile
    if donor is not None:
        receivers = _receivers(features=features, donor=donor)
    p10, p50, p90 = (float(np.nanpercentile(series, q)) for q in (10, 50, 90))
    row.update(
        onset_stamp=int(onset),
        end_stamp=int(end),
        onset_unc_hh=2,
        end_unc_hh=4,
        donor=donor,
        receivers=receivers,
        moved_mw_p50=p50,
        moved_mw_p10=p10,
        moved_mw_p90=p90,
        unmetered_receiver=bool(donor is not None and not receivers),
        confidence="high" if min(steps[s]["close_pct"] for s in use) > 0.995 else "med",
    )
    return row
