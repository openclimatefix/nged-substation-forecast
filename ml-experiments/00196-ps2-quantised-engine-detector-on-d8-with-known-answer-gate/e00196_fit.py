"""Fit the noise on D8's clean history and the detector on 2023-09 to 2025-12; save the fits.

OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00196-*/e00196_fit.py
"""

import pickle
import sys
from datetime import UTC, datetime

import numpy as np
from e00196_common import DATA_DIR, FIT_END, FIT_START, Q_FLOOR, UNIT_MW, stamp
from e00196_pipeline import clean_mask, fit_noise, load_series, train_rounds, with_local_noise

PRIORS = {"wide": (2.5, 1.0), "design": (UNIT_MW, 0.3)}


def main() -> None:
    """Fit and print both priors' unit-size paths."""
    s = load_series(label="D8")
    clean = clean_mask(n=len(s.y))
    noise = fit_noise(s=s, rows=clean)
    print(f"noise (odd months of clean history): {noise}")
    print(f"clean rows {clean.sum()}, lag-1 rho {noise.rho:.3f}")
    presolar = "presolar" in sys.argv[1:]
    lo, hi = stamp(FIT_START), stamp(datetime(2025, 3, 1, tzinfo=UTC) if presolar else FIT_END)
    fits = {}
    local = "local" in sys.argv[1:]
    if local:
        noise = with_local_noise(s=s, noise=noise, reference_rows=clean)
        assert noise.mult is not None
        print(
            "local-noise variant: multiplier on 2023-09 to 2025-12 has median "
            f"{float(np.median(noise.mult[stamp(FIT_START) : stamp(FIT_END)])):.2f}"
        )
    names = [a for a in sys.argv[1:] if a in PRIORS] or list(PRIORS)
    for name in names:
        mean, sd = PRIORS[name]
        print(f"prior {name}: centre {mean} MW, sd {sd} MW")
        params, path = train_rounds(
            s=s,
            lo=lo,
            hi=hi,
            noise=noise,
            prior_mean=mean,
            prior_sd=sd,
            q=Q_FLOOR,
        )
        fits[name] = {"params": params, "g_path": path}
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (
            DATA_DIR / f"fit_{name}{'_local' if local else ''}{'_presolar' if presolar else ''}.pkl"
        ).write_bytes(pickle.dumps({"noise": noise, "fits": {name: fits[name]}}))


if __name__ == "__main__":
    main()
