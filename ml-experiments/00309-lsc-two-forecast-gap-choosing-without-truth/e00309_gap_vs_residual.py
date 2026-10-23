# ruff: noqa: E501
"""Paired comparison of the plain gap with the residual windows at equal false alarm rates.

Reads experiment 308's stored per-shift match flags, so it fits nothing. The thresholds were set on
odd months to the carried detector's odd-month false alarm rate, so each pair below shares an
operating point. Intervals resample whole shifts, 20,000 draws.

    uv run --no-sync python e00309_gap_vs_residual.py
"""

from __future__ import annotations

import json

import e00309_data as dat
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

REPORT = (
    ML_EXPERIMENTS_DATA_DIR / "00308-lsc-two-forecast-gap-as-a-detector" / "results" / "report.json"
)
N_DRAWS = 20_000


def paired(*, a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[int, float, float]:
    """Matched-shift difference a minus b with a 95% interval over whole shifts."""
    idx = rng.integers(0, len(a), size=(N_DRAWS, len(a)))
    diff = (a.astype(int) - b.astype(int))[idx].sum(axis=1)
    lo, hi = np.quantile(diff, [0.025, 0.975])
    return int(a.sum() - b.sum()), float(lo), float(hi)


def main() -> None:
    """Print the paired differences."""
    report = json.loads(REPORT.read_text())
    gap = report["gap"]
    rng = np.random.default_rng(309)
    out = []
    for src in ("LR_b20", "RO_b80"):
        base = gap[f"d_a_{src}"]
        for other in ("r6_a", "r24_a", "r72_a", "d24_a"):
            row = gap[f"{other}_{src}"]
            for key, label in (("_matched", "while on"), ("_late", "within 4 days of end")):
                d, lo, hi = paired(a=np.array(row[key]), b=np.array(base[key]), rng=rng)
                out.append(
                    {
                        "rate": src,
                        "score": other,
                        "rule": label,
                        "gap": int(sum(base[key])),
                        "other": int(sum(row[key])),
                        "diff": d,
                        "lo": lo,
                        "hi": hi,
                        "fa_odd_gap": base["fa_odd"],
                        "fa_odd_other": row["fa_odd"],
                        "fa_even_gap": base["fa_even"],
                        "fa_even_other": row["fa_even"],
                    }
                )
                print(
                    f"@{src} {other:6s} minus d_a, {label:22s}: gap {sum(base[key])} other {sum(row[key])} diff {d:+d} ({lo:+.0f} to {hi:+.0f}); FA odd {base['fa_odd']:.2f} vs {row['fa_odd']:.2f}, even {base['fa_even']:.2f} vs {row['fa_even']:.2f}"
                )
    (dat.WORK_DIR / "results").mkdir(parents=True, exist_ok=True)
    (dat.WORK_DIR / "results" / "gap_vs_residual.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
