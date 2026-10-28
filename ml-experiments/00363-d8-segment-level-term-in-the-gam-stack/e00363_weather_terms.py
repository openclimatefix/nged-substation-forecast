"""Compare each term's fitted coefficients with and without the segment term, at the last refit.

    PYTHONPATH=$E252:$E264:$E354:$E361:$E363 uv run --no-sync python -W ignore ml-experiments/00363-*/e00363_weather_terms.py
"""

import json
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

import pickle

import e00354_step  # noqa: F401  # ty: ignore[unresolved-import]  (unpickling)
import e00363_fit as fit_module
import e00363_segment  # noqa: F401  (unpickling)
import numpy as np


def main() -> None:
    """Write `work/weather_terms.json` and print it."""
    out: dict = {}
    for site in ("D8", "D2"):
        ctl = pickle.loads((fit_module.arm_dir(site=site, arm="ctl") / "tall_fits.pkl").read_bytes())[-1]
        seg = pickle.loads((fit_module.arm_dir(site=site, arm="seg_mid") / "tall_fits.pkl").read_bytes())[-1]
        out[site] = {
            name: float(np.linalg.norm(ctl.fit.theta[sl] - seg.fit.theta[sl]) / max(np.linalg.norm(ctl.fit.theta[sl]), 1e-9))
            for name, sl in ctl.layout.slices.items()
        }
        levels = seg.fit.theta[ctl.fit.theta.shape[0] :]
        out[site + "_levels"] = {"n": len(levels), "sd": float(levels.std()), "max_abs": float(np.abs(levels).max())}
    print(json.dumps(out, indent=1))
    weather = {k: v for k, v in out.items() if not k.endswith("_levels")}
    (fit_module.WORK / "weather_terms.json").write_text(json.dumps(weather))
    (fit_module.WORK / "level_stats.json").write_text(json.dumps({k: v for k, v in out.items() if k.endswith("_levels")}))


if __name__ == "__main__":
    main()
