"""Known answers and false alarms for detector v2 and its separate features.

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00199-*/e00199_run.py [known|alarms|all]

Arms: v1 (experiment 196 as it was), v1 plus the edge test (S), v1 plus the price gate (P), the
night-evidence detector (N), and N plus S plus P (v2). Everything is scored on night rows (the
headline) and on all rows (secondary).
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any

import numpy as np
from e00199_common import (
    DATA_DIR,
    G2_FIRST,
    N_UNITS,
    g2_output,
    stamp,
)
from e00199_gate import false_alarms
from e00199_inject import Injection, build, build_k3
from e00199_pipeline import clean_mask, fit_noise, load_series
from e00199_scoring import flagged, passes, score
from e00199_v2 import (
    Gates,
    apply_gates,
    detect,
    injection_spread,
    load_frozen_fit,
    night_mask,
    spread_array,
)

ARMS: dict[str, tuple[bool, Gates]] = {
    "v1": (False, Gates()),
    "v1+S": (False, Gates(shape=True)),
    "v1+P": (False, Gates(price=True)),
    "N": (True, Gates()),
    "v2 = N+S+P": (True, Gates(shape=True, price=True)),
    "post hoc N+P": (True, Gates(price=True)),
    "post hoc N+S(either edge)+P": (True, Gates(shape=True, price=True, either_edge=True)),
}
SHARE_SWEEP = (0.5, 0.7)
"""Edge-test sharpness levels reported beside the pre-registered 0.6 (v2 only)."""
PRIMARIES = ("D3", "D10", "D11", "D14", "D16", "D5")
WINDOWS = {
    "D8_clean": None,
    "D8_2025-03_to_2025-12": ("2025-03-01", "2026-01-01"),
    "D8_2026-01_to_2026-06": ("2026-01-01", "2026-07-01"),
}


def _scores_on(*, inj: Injection, smoothed: np.ndarray, filtered: np.ndarray, g: float) -> dict:
    sc = score(inj=inj, smoothed=smoothed, filtered=filtered, g=g)
    sc["passes"] = passes(scores=sc)
    return sc


def known_answers() -> dict[str, Any]:
    """K1 and K2 scored for every arm, on night rows and on all rows."""
    noise, params = load_frozen_fit()
    s = load_series(label="D8")
    n = len(s.y)
    g2 = g2_output(n=n)
    night = night_mask(label="D8", n=n)
    spread, gate = spread_array(n=n)
    first = stamp(G2_FIRST)
    out: dict[str, Any] = {"gate_spread": gate, "n_units": N_UNITS}
    for kname, kwargs in (
        ("K1", {"with_k2": False}),
        ("K2", {"with_k1": False}),
        ("K12", {}),
        ("K3", {"k3": True}),
    ):
        inj = build_k3(y=s.y, g2=g2) if kwargs.get("k3") else build(y=s.y, g2=g2, **kwargs)
        sp = injection_spread(inj=inj, spread=spread, first=first)
        night_inj = replace(inj, window_id=np.where(night, inj.window_id, -1))
        si = s.with_reading(inj.y)
        for arm, (use_night, gates) in ARMS.items():
            post, x = detect(s=si, noise=noise, params=params, night=night if use_night else None)
            variants = {arm: gates}
            if arm.startswith("v2"):  # the sharpness sweep is for the verdict arm only
                variants |= {f"{arm} sharp{sh}": replace(gates, share=sh) for sh in SHARE_SWEEP}
            for name, gt in variants.items():
                sm = apply_gates(
                    post=post, x=x, unit=params.g, spread=sp, gate_threshold=gate, gates=gt
                )
                out[f"{kname} | {name} | night"] = _scores_on(
                    inj=night_inj, smoothed=sm, filtered=post.filtered, g=params.g
                )
                out[f"{kname} | {name} | all"] = _scores_on(
                    inj=inj, smoothed=sm, filtered=post.filtered, g=params.g
                )
        print(kname, "done", flush=True)
    return out


def _alarm_rows(*, label: str, window: tuple[str, str] | None) -> tuple[np.ndarray, np.ndarray]:
    """Rows scored at a Primary (before the night restriction) and the noise-fit rows."""
    from datetime import UTC, datetime

    s = load_series(label=label)
    n = len(s.y)
    finite = np.isfinite(s.y) & np.isfinite(s.load)
    even = (((s.month - 1) % 12) + 1) % 2 == 0
    if label == "D8":
        if window is None:
            return clean_mask(n=n) & even, clean_mask(n=n)
        a, b = (datetime.fromisoformat(w).replace(tzinfo=UTC) for w in window)
        rows = np.zeros(n, dtype=bool)
        rows[stamp(a) : stamp(b)] = True
        return rows & finite, clean_mask(n=n)
    return finite & even, finite


def alarms_for(task: tuple[str, str]) -> tuple[str, dict[str, Any]]:
    """False alarms of every arm at one Primary (or one D8 window)."""
    key, label = task
    window = WINDOWS.get(key)
    d8_noise, params = load_frozen_fit()
    s = load_series(label=label)
    n = len(s.y)
    rows, noise_rows = _alarm_rows(label=label, window=window)
    noise = d8_noise if label == "D8" else fit_noise(s=s, rows=noise_rows)
    night = night_mask(label=label, n=n)
    spread, gate = spread_array(n=n)
    has_price = np.isfinite(spread)
    res: dict[str, Any] = {
        "night_share_of_rows": float(night[rows].mean()),
        "left_tail_night": float((s.resid[rows & night] < -2.5).mean()),
        "price_rows": int((rows & has_price).sum()),
    }
    for arm, (use_night, gates) in ARMS.items():
        post, x = detect(s=s, noise=noise, params=params, night=night if use_night else None)
        variants = {arm: gates}
        if arm.startswith("v2"):  # the sharpness sweep is for the verdict arm only
            variants |= {f"{arm} sharp{sh}": replace(gates, share=sh) for sh in SHARE_SWEEP}
        for name, gt in variants.items():
            sm = apply_gates(
                post=post, x=x, unit=params.g, spread=spread, gate_threshold=gate, gates=gt
            )
            flag = flagged(smoothed=sm)
            res[f"{name} | night"] = false_alarms(s=s, smoothed=sm, rows=rows & night)
            res[f"{name} | all"] = false_alarms(s=s, smoothed=sm, rows=rows)
            np.save(DATA_DIR / f"flags_{key}_{name.split(' ')[0]}_{len(name)}.npy", flag)
        print(key, arm, "done", flush=True)
    return key, res


def main() -> None:
    """Write the known-answer and false-alarm tables."""
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if what in ("known", "all"):
        (DATA_DIR / "known_answers.json").write_text(
            json.dumps(known_answers(), indent=1, default=float)
        )
    if what in ("alarms", "all"):
        tasks = [(k, "D8") for k in WINDOWS] + [(p, p) for p in PRIMARIES]
        with ProcessPoolExecutor(max_workers=3) as pool:
            results = dict(pool.map(alarms_for, tasks))
        (DATA_DIR / "false_alarms.json").write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
