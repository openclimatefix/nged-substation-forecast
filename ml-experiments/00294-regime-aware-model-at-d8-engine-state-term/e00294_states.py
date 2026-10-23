"""Engine-running states for D8 and the control Primaries, filtered and smoothed.

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00294-*/e00294_states.py

Runs experiment 196's frozen detector (arm v1, as experiment 291 ran it) and stores, per site and
on the 2022-01-01 half-hour grid: `p_filtered_<site>.npy` and `p_smoothed_<site>.npy`, the
probability of one or more running units from readings up to the half-hour and from all readings,
and `p_units_filtered_<site>.npy`, the expected number of running units from the filtered posterior.
The smoothed file feeds only the hindsight arm.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
from e00294_detcommon import DATA_DIR
from e00294_pipeline import fit_noise, load_series
from e00294_v2 import detect, load_frozen_fit

SITES: Final[tuple[str, ...]] = ("D8", "D5", "D10", "D11")


def state_path(*, site: str, kind: str) -> Path:
    """Return the file of a site's stored probability (`kind` is `filtered` or `smoothed`)."""
    return DATA_DIR / f"p_{kind}_{site}.npy"


def one(site: str) -> str:
    """Run the detector at one site and store both probabilities."""
    d8_noise, params = load_frozen_fit()
    s = load_series(label=site)
    noise = d8_noise
    if site != "D8":
        noise = fit_noise(s=s, rows=np.isfinite(s.y) & np.isfinite(s.load))
    post, _ = detect(s=s, noise=noise, params=params, night=None)
    np.save(
        state_path(site=site, kind="filtered"), post.filtered[:, 1:].sum(axis=1).astype(np.float32)
    )
    np.save(
        state_path(site=site, kind="smoothed"), post.smoothed[:, 1:].sum(axis=1).astype(np.float32)
    )
    units = np.arange(post.filtered.shape[1])
    np.save(
        state_path(site=site, kind="units_filtered"), (post.filtered @ units).astype(np.float32)
    )
    return site


def main() -> None:
    """Run every site."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for site in pool.map(one, SITES):
            print(site, "done", flush=True)
    for site in SITES:
        f = np.load(state_path(site=site, kind="filtered"))
        m = np.load(state_path(site=site, kind="smoothed"))
        print(
            site,
            len(f),
            "flag share filtered",
            float((f > 0.5).mean()),
            "smoothed",
            float((m > 0.5).mean()),
        )


if __name__ == "__main__":
    main()
