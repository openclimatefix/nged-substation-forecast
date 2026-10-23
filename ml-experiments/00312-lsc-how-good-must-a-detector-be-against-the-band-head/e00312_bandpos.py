# ruff: noqa: E501, PLR0912, PLR0915
"""Band position as a detector on experiment 297's injected worlds, beside RO-NC.

Reads the per-run band-position series, the injected events, and the RO-NC alarm spans stored by
`e00312_degraded.py`. An alarm state opens when the signal exceeds k at two consecutive runs and
closes when it stays under k/2 at two runs or after 14 days (experiment 309's rule). A shift is
matched when an alarm opens between its onset and its end; an alarm outside every shift and its
72-hour tail is false (real transfers in the stream count as false). k is picked on odd months to
reach a target number of alarms per matched shift, and recall, median onset delay, and alarms per
match are read on even months. RO-NC is scored with the same match rule at its own settings.

    PYTHONPATH=... uv run --no-sync python e00312_bandpos.py
"""

from __future__ import annotations

import json
from typing import Final

import numpy as np
import polars as pl
from e00312_common import BAND_DIR, EVENTS_DIR, RESULTS_DIR, TAIL_STAMPS

MAX_RUNS: Final[int] = 56
TARGETS: Final[tuple[float, ...]] = (1.33, 2.0, 2.7)
K_GRID: Final[dict[str, np.ndarray]] = {
    "pit24": np.round(np.arange(0.3, 3.01, 0.1), 2),
    "ext24": np.round(np.arange(0.9, 2.71, 0.05), 2),
}
N_DRAWS: Final[int] = 2000
SEED: Final[int] = 312


def opens_of(*, value: np.ndarray, k: float) -> np.ndarray:
    """Run indices at which an alarm opens, under the two-run enter and exit rule."""
    over = np.nan_to_num(value, nan=0.0) > k
    under = np.nan_to_num(value, nan=0.0) < k / 2.0
    out, state, since = [], False, 0
    for i in range(1, len(value)):
        if not state and over[i] and over[i - 1]:
            state, since = True, i
            out.append(i)
        elif state and ((under[i] and under[i - 1]) or i - since >= MAX_RUNS):
            state = False
    return np.array(out, dtype=np.int64)


def evaluate(*, opens_cut: list[np.ndarray], opens_month: list[np.ndarray], events: list[pl.DataFrame], parity: int) -> dict[str, float]:
    """Pooled recall, median delay (hours), and alarms per match on events of one month parity."""
    matched, delays, alarms, n_events = 0, [], 0, 0
    per_world = []
    for cut, mon, ev in zip(opens_cut, opens_month, events, strict=True):
        onset, end = ev["onset"].to_numpy(), ev["end"].to_numpy()
        month = ev["onset_month"].to_numpy()
        sel = (month % 2) == parity
        m_w, d_w = 0, []
        for o, e in zip(onset[sel], end[sel], strict=True):
            hit = cut[(cut >= o) & (cut <= e)]
            if len(hit):
                m_w += 1
                d_w.append((hit.min() - o) / 2.0)
        busy = np.zeros(len(cut), dtype=bool)
        for o, e in zip(onset, end, strict=True):
            busy |= (cut >= o) & (cut <= e + TAIL_STAMPS)
        keep = (mon % 2) == parity
        a_w = int(np.sum(keep))
        per_world.append((int(sel.sum()), m_w, a_w, d_w))
        n_events += int(sel.sum())
        matched += m_w
        alarms += a_w
        delays += d_w
    return {
        "events": n_events,
        "matched": matched,
        "recall": matched / max(n_events, 1),
        "median_delay_hours": float(np.median(delays)) if delays else float("nan"),
        "alarms": alarms,
        "alarms_per_match": alarms / max(matched, 1),
        "per_world": per_world,
    }


def bootstrap_recall(*, per_world: list, rng: np.random.Generator) -> tuple[float, float]:
    """95% interval of recall, resampling whole worlds."""
    arr = np.array([(n, m) for n, m, _, _ in per_world], dtype=float)
    draws = rng.integers(0, len(arr), size=(N_DRAWS, len(arr)))
    rec = arr[draws, 1].sum(axis=1) / np.maximum(arr[draws, 0].sum(axis=1), 1)
    lo, hi = np.percentile(rec, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    """Calibrate each signal and RO-NC and print the numbers the page quotes."""
    rng = np.random.default_rng(SEED)
    keys = sorted(
        {p.name.removeprefix("events_") for p in EVENTS_DIR.glob("events_*.parquet")}
    )
    bands, events, ronc = [], [], []
    for name in keys:
        bands.append(pl.read_parquet(BAND_DIR / name))
        events.append(pl.read_parquet(EVENTS_DIR / f"events_{name}"))
        ronc.append(pl.read_parquet(EVENTS_DIR / f"ronc_{name}"))
    print(f"{len(keys)} site-worlds, {sum(e.height for e in events)} events")
    out: dict = {"n_site_worlds": len(keys)}
    # RO-NC under the same rule: an alarm is known at its `start`.
    ro_cut = [r["start"].to_numpy() for r in ronc]
    ro_month = [r["month"].to_numpy() for r in ronc]
    for parity, label in ((1, "odd"), (0, "even")):
        res = evaluate(opens_cut=ro_cut, opens_month=ro_month, events=events, parity=parity)
        lo, hi = bootstrap_recall(per_world=res.pop("per_world"), rng=rng)
        res["recall_lo"], res["recall_hi"] = lo, hi
        out[f"RO_NC_{label}"] = res
        print(f"RO-NC {label}: {res}")
    target_set = {"RO_NC": out["RO_NC_odd"]["alarms_per_match"]}
    for t in TARGETS:
        target_set[f"{t}"] = t
    for score, grid in K_GRID.items():
        values = [b[score].to_numpy() if score == "ext24" else np.abs(b[score].to_numpy()) for b in bands]
        cutoffs = [b["cutoff"].to_numpy() for b in bands]
        months = [b["month"].to_numpy() for b in bands]
        table = []
        for k in grid:
            oc, om = [], []
            for v, c, m in zip(values, cutoffs, months, strict=True):
                idx = opens_of(value=v, k=float(k))
                oc.append(c[idx])
                om.append(m[idx])
            odd = evaluate(opens_cut=oc, opens_month=om, events=events, parity=1)
            odd.pop("per_world")
            table.append({"k": float(k), "odd_alarms_per_match": odd["alarms_per_match"], "odd_recall": odd["recall"], "_oc": oc, "_om": om})
        for tname, target in target_set.items():
            pick = min(table, key=lambda r: abs(r["odd_alarms_per_match"] - target))
            even = evaluate(opens_cut=pick["_oc"], opens_month=pick["_om"], events=events, parity=0)
            lo, hi = bootstrap_recall(per_world=even.pop("per_world"), rng=rng)
            even.update({"recall_lo": lo, "recall_hi": hi, "k": pick["k"], "target": target, "odd_alarms_per_match": pick["odd_alarms_per_match"]})
            out[f"{score}_at_{tname}"] = even
            print(f"{score} at target {tname} ({target:.2f}): k={pick['k']} even: {even}")
        out[f"{score}_curve"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in table]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "bandpos.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
