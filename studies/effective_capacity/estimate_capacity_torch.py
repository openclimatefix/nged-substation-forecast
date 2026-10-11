"""Run DP1, the variational physics estimator, on every record and both of its settings.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1143>.
`studies.capacity_physics_torch` holds the model. Every plant and every record fits in one batch,
and each refit warm-starts from the one before it, so a full run is a few minutes on a GPU. The
records are the original, the 11 draws, the positive control, the ANM-blind original, and the
time-reversed original and control (for the hindsight reference).

Each setting's estimates are saved as `estimates/DP1/<setting>/<record>.npz`, and a setting whose
files all exist is skipped. `--smoke` runs every record with 20 and 5 steps on
the CPU and writes under `_smoke/`; it is not a result.

Run it with `uv run python studies/effective_capacity/estimate_capacity_torch.py --device cuda`.
"""

import argparse
import json
import logging
import sys
from typing import Final

import numpy as np
from capacity_paths import (
    CONTROL,
    ORIGINAL,
    TUNING_GRIDS,
    claim_root,
    draw_id,
    estimate_path,
    inputs_dir,
    load_events,
    record_for,
    save_estimate,
    setting_name,
)
from studies.capacity_physics_torch import run_dp1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("estimate_capacity_torch")

SMOKE_STEPS: Final[tuple[float, float]] = (20.0, 5.0)
"""A smoke run's first-refit and later-refit steps."""


def main() -> int:
    """Run DP1 at both settings over every record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="The torch device, such as cuda or cpu.")
    parser.add_argument("--smoke", action="store_true", help="Tiny run on the CPU; not a result.")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    events = load_events(root=root)
    draws = [draw_id(index=i) for i in range(len(events.draws))]
    file_ids = [
        ORIGINAL,
        *draws,
        CONTROL,
        "anm_blind",
        f"{ORIGINAL}__reversed",
        f"{CONTROL}__reversed",
    ]
    orientation = json.loads((inputs_dir(root=root) / "orientation.json").read_text())
    original = record_for(root=root, record_id=ORIGINAL)
    static = original.static_p99_mw
    dc_prior = np.array([orientation[s]["dc_capacity_mw"] for s in original.sites]) / static
    ac_limit = np.array([orientation[s]["ac_capacity_mw"] for s in original.sites]) / static
    records = [record_for(root=root, record_id=file_id) for file_id in file_ids]
    for params in TUNING_GRIDS["DP1"]:
        setting = setting_name(params=params)
        paths = [
            estimate_path(root=root, estimator="DP1", setting=setting, record_id=f)
            for f in file_ids
        ]
        if all(path.exists() for path in paths):
            _LOG.info("DP1 %s is complete", setting)
            continue
        settings = dict(params)
        device = "cpu" if arguments.smoke else arguments.device
        if arguments.smoke:
            settings["first_steps"], settings["later_steps"] = SMOKE_STEPS
        estimates = run_dp1(
            records=records, dc_prior=dc_prior, ac_limit=ac_limit, params=settings, device=device
        )
        for path, estimate in zip(paths, estimates, strict=True):
            save_estimate(path=path, level=estimate.level, sd=estimate.sd)
        _LOG.info("DP1 %s saved %d records", setting, len(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
