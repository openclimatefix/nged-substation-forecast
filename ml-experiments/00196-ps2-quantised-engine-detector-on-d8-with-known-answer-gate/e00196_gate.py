"""Run the frozen detector on D8: clean history, K1, K2, synthetic injections, and the regimes.

OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00196-*/e00196_gate.py
"""

import json
import pickle
import sys

import numpy as np
import polars as pl
from e00196_common import (
    DATA_DIR,
    FALL_START,
    FIT_END,
    FIT_START,
    POST_START,
    Q_FLOOR,
    g2_output,
    stamp,
)
from e00196_inject import build, synthetic
from e00196_pipeline import Series, clean_mask, load_series, run_loop
from e00196_scoring import flagged, passes, score

Q_SENSITIVITY = (0.0, 0.5, 1.0)


def false_alarms(*, s: Series, smoothed: np.ndarray, rows: np.ndarray) -> dict:
    """Share of rows flagged, and false active days (4 or more flagged half-hours) per month."""
    flag = flagged(smoothed=smoothed)
    rows = rows & np.isfinite(s.y)
    day = np.arange(len(flag)) // 48
    n_days = np.unique(day[rows]).size
    per_day = np.bincount(day[rows], weights=flag[rows].astype(float))
    active_days = int((per_day >= 4).sum())
    months = max(n_days / 30.4, 1e-9)
    return {
        "rows": int(rows.sum()),
        "flagged_share": float(flag[rows].mean()),
        "false_active_days": active_days,
        "days": int(n_days),
        "false_active_days_per_month": active_days / months,
    }


def main() -> None:
    """Score every arm and write the tables."""
    local = "local" in sys.argv[1:]
    suffix = "_local" if local else "_fixed"
    fits = pickle.loads(
        (DATA_DIR / ("fit_wide_local.pkl" if local else "fits_fixed.pkl")).read_bytes()
    )
    noise = fits["noise"]
    s = load_series(label="D8")
    n = len(s.y)
    g2 = g2_output(n=n)
    clean = clean_mask(n=n)
    ref = clean if local else None
    if local:
        noise = noise.__class__(noise.rho, noise.s0, noise.s1, None)
    even = (((s.month - 1) % 12) + 1) % 2 == 0
    results: dict = {}
    for name, fit in fits["fits"].items():
        params = fit["params"]
        res: dict = {"g": params.g, "g_path": fit["g_path"]}
        # --- uninjected: false alarms, regimes, months
        _offset, post = run_loop(s=s, noise=noise, params=params, local_reference=ref)
        np.save(DATA_DIR / f"smoothed_{name}{suffix}.npy", post.smoothed.astype(np.float32))
        res["false_alarms_clean_even_months"] = false_alarms(
            s=s, smoothed=post.smoothed, rows=clean & even
        )
        res["false_alarms_clean_all"] = false_alarms(s=s, smoothed=post.smoothed, rows=clean)
        flag = flagged(smoothed=post.smoothed)
        months = (
            pl.DataFrame({"month": s.month, "flag": flag, "left": s.resid < -2.5})
            .group_by("month")
            .agg(pl.col("flag").mean(), pl.col("left").mean())
            .sort("month")
        )
        res["monthly"] = months.to_dicts()
        for label, lo, hi in (
            ("fit_window", FIT_START, FIT_END),
            ("fall", FALL_START, POST_START),
            ("post_fall", POST_START, None),
        ):
            a = stamp(lo)
            b = stamp(hi) if hi else n
            res[f"flagged_share_{label}"] = float(flag[a:b].mean())
            res[f"left_tail_share_{label}"] = float((s.resid[a:b] < -2.5).mean())
        # --- known answers
        for arm, kwargs in (("K1", {"with_k2": False}), ("K2", {"with_k1": False})):
            inj = build(y=s.y, g2=g2, **kwargs)
            _, p = run_loop(
                s=s.with_reading(inj.y), noise=noise, params=params, local_reference=ref
            )
            sc = score(inj=inj, smoothed=p.smoothed, filtered=p.filtered, g=params.g)
            sc["passes"] = passes(scores=sc)
            res[arm] = sc
            print(name, arm, json.dumps({k: v for k, v in sc.items() if k == "passes"}), flush=True)
        for rate in (0.5, 2.0):
            inj = synthetic(y=s.y, g2=g2, rate=rate, seed=196)
            _, p = run_loop(
                s=s.with_reading(inj.y), noise=noise, params=params, local_reference=ref
            )
            sc = score(inj=inj, smoothed=p.smoothed, filtered=p.filtered, g=params.g)
            sc["passes"] = passes(scores=sc)
            res[f"synthetic_x{rate}"] = sc
        results[name] = res
    # --- reactive-floor sensitivity (design-prior and wide fits, K1 and K2 together)
    for name, fit in fits["fits"].items():
        for q in Q_SENSITIVITY:
            sq = load_series(label="D8", q=q)
            inj = build(y=s.y, g2=g2, q=Q_FLOOR)
            _, p = run_loop(
                s=sq.with_reading(inj.y),
                noise=noise,
                params=fit["params"],
                q=q,
                local_reference=ref,
            )
            sc = score(inj=inj, smoothed=p.smoothed, filtered=p.filtered, g=fit["params"].g)
            sc["passes"] = passes(scores=sc)
            results[name][f"q_{q}"] = sc
    (DATA_DIR / f"gate_results{suffix}.json").write_text(
        json.dumps(results, indent=1, default=float)
    )
    print("written", file=sys.stderr)


if __name__ == "__main__":
    main()
