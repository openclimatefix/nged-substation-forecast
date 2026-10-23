"""Print how the ensemble hindcast period differs from the scored year, farm by farm.

Prints, for each farm, the mean ensemble 100 m wind speed and the mean observed fraction of
capacity of the hindcast cases (initialised 2024-04-01 to 2025-06-16) and of the scored cases
(the 53 Tuesday runs of July 2025 to June 2026), and the observed fraction in bins of the members'
mean 100 m speed. Output goes to `work/shift.json`.

    uv run --no-sync python e00303_shift.py
"""

import json
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import e00303_common as common  # noqa: E402
import e00303_fit as fit  # noqa: E402
import numpy as np  # noqa: E402

BINS = ((0, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, 40))


def main() -> None:
    """Write the shift table."""
    report: dict = {}
    for farm in ("W1", "W2", "W3"):
        data = fit.Hindcast(farm=farm)
        cases = common.cases_table(sites=data.sites)
        scored_actual = cases["actual_norm"].to_numpy()
        hind_speed = data.hind_columns["speed_100m"].reshape(-1, common.MEMBERS).mean(axis=1)
        scored_speed = data.scored_columns["speed_100m"].reshape(-1, common.MEMBERS).mean(axis=1)
        entry: dict = {
            "hindcast_mean_speed": float(hind_speed.mean()),
            "scored_mean_speed": float(scored_speed.mean()),
            "hindcast_mean_observed": float(data.hind_actual[data.usable].mean()),
            "scored_mean_observed": float(np.nanmean(scored_actual)),
            "bins": {},
        }
        for low, high in BINS:
            h = data.usable & (hind_speed >= low) & (hind_speed < high)
            s = (scored_speed >= low) & (scored_speed < high) & np.isfinite(scored_actual)
            entry["bins"][f"{low}-{high}"] = {
                "hindcast": float(data.hind_actual[h].mean()) if h.any() else None,
                "scored": float(scored_actual[s].mean()) if s.any() else None,
            }
        report[farm] = entry
        print(farm, json.dumps(entry), flush=True)
    (common.WORK_DIR / "shift.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
