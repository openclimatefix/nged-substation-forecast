"""False alarms of the frozen D8 detector on the decision Primaries (and D8's own clean history).

Each Primary keeps D8's unit size and transition tables and gets its own noise fit (odd months of
its whole history). False alarms are scored on even months, which the noise fit did not see.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00196-*/e00196_primaries.py
"""

import json
import pickle
import sys

import numpy as np
from e00196_common import DATA_DIR, DECISION_LABELS
from e00196_gate import false_alarms
from e00196_pipeline import fit_noise, load_series, run_loop
from e00196_scoring import flagged


def main() -> None:
    """Write per-Primary flagged shares and false active days."""
    local = "local" in sys.argv[1:]
    fits = pickle.loads((DATA_DIR / ("fit_wide_local.pkl" if local else "fits_fixed.pkl")).read_bytes())
    out: dict = {}
    for label in DECISION_LABELS:
        s = load_series(label=label)
        rows = np.isfinite(s.y) & np.isfinite(s.load)
        noise = fit_noise(s=s, rows=rows)
        even = (((s.month - 1) % 12) + 1) % 2 == 0
        out[label] = {"noise": [noise.rho, noise.s0, noise.s1]}
        for name, fit in fits["fits"].items():
            _, post = run_loop(s=s, noise=noise, params=fit["params"], local_reference=rows if local else None)
            fa = false_alarms(s=s, smoothed=post.smoothed, rows=rows & even)
            by_year = {}
            years = (s.month - 1) // 12
            for y in np.unique(years[rows]):
                sel = rows & even & (years == y)
                by_year[int(y)] = float(flagged(smoothed=post.smoothed)[sel].mean())
            out[label][name] = {**fa, "flagged_share_by_year": by_year}
            print(label, name, round(fa["flagged_share"], 4), round(fa["false_active_days_per_month"], 2), flush=True)
    (DATA_DIR / f"primaries_results_{'local' if local else 'fixed'}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
