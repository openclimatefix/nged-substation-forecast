"""Engine-running probabilities for D8 and the control Primaries over their whole history.

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00291-*/e00291_states.py

Writes, per site and arm, the smoothed probability that one or more units run (`p_run_<site>_<arm>
.npy`, one value per half-hour of the 2022-01-01 grid) under `DATA_DIR`. Arm v1 is experiment 196's
detector with its frozen fit; arm N is experiment 199's night-evidence version of the same detector.
D8 uses experiment 196's frozen noise; each control fits its own noise on its finite half-hours, as
experiment 199 did.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
from e00291_common import DATA_DIR
from e00291_pipeline import fit_noise, load_series
from e00291_v2 import detect, load_frozen_fit, night_mask
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SITES: Final[tuple[str, ...]] = ("D8", "D5", "D10", "D11")
ARMS: Final[tuple[str, ...]] = ("v1", "N")
STORED_V1_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00196-ps2-quantised-engine-detector-on-d8-with-known-answer-gate"
    / "smoothed_wide_fixed.npy"
)  # fmt: skip


def p_run_path(*, site: str, arm: str) -> Path:
    """File holding the smoothed probability of one or more units running."""
    return DATA_DIR / f"p_run_{site}_{arm}.npy"


def one(task: tuple[str, str]) -> tuple[str, str]:
    """Run one detector arm at one site and store P(one or more units)."""
    site, arm = task
    d8_noise, params = load_frozen_fit()
    s = load_series(label=site)
    noise = d8_noise
    if site != "D8":
        noise = fit_noise(s=s, rows=np.isfinite(s.y) & np.isfinite(s.load))
    night = night_mask(label=site, n=len(s.y)) if arm == "N" else None
    post, _ = detect(s=s, noise=noise, params=params, night=night)
    np.save(p_run_path(site=site, arm=arm), post.smoothed[:, 1:].sum(axis=1).astype(np.float32))
    return site, arm


def main() -> None:
    """Run every site and arm, then compare D8's arm v1 with experiment 196's stored posterior."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(site, arm) for site in SITES for arm in ARMS]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for site, arm in pool.map(one, tasks):
            print(site, arm, "done", flush=True)
    mine = np.load(p_run_path(site="D8", arm="v1")) > 0.5
    stored = np.load(STORED_V1_PATH)[:, 1:].sum(axis=1) > 0.5
    check = {
        "d8_v1_flag_share_recomputed": float(mine.mean()),
        "d8_v1_flag_share_stored": float(stored.mean()),
        "d8_v1_agreement": float((mine == stored).mean()),
    }
    (DATA_DIR / "states_check.json").write_text(json.dumps(check, indent=1))
    print(check)


if __name__ == "__main__":
    main()
