"""Experiment 00342, first step: per-case summaries of the ensemble members' 100 m wind speed.

For each farm and each of the hindcast and scored case sets, in the case order of experiment
00336's stored statistics, write an array (cases, 3): the mean and standard deviation of the 51
members' 100 m speed, and the share of members at or above the farm's rated (saturation) speed.
The rated speed is the first speed at which the farm's ERA5 power curve (ERA5 valid times before
2024-04-01) reaches 95% of its maximum.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue env OMP_NUM_THREADS=1 \
      uv run --no-sync python ml-experiments/00342-*/e00342_stats.py --farms W1
    (PYTHONPATH: see e00342_common)
"""

import argparse
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_ens as ens  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_fit as fit303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00326_score as score326  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00342_common as common  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    """Write the member speed summaries of the requested farms."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default=",".join(common.FARMS))
    farms = parser.parse_args().farms.split(",")
    common.STATS_DIR.mkdir(parents=True, exist_ok=True)
    rated = score326.rated_speeds(sites=common.c303.site_table())
    for farm in farms:
        data = fit303.Hindcast(farm=farm)
        for name, columns in (("hindcast", data.hind_columns), ("scored", data.scored_columns)):
            speed = columns["speed_100m"].reshape(-1, common.c336.MEMBERS).astype(np.float64)
            summary = np.column_stack(
                [speed.mean(axis=1), speed.std(axis=1), (speed >= rated[farm]).mean(axis=1)]
            ).astype(np.float32)
            # Consistency with experiment 336's stored ensemble mean (same case order).
            stored = np.load(common.c336.stats_path(kind="speed", set_name=name, farm=farm))
            if not np.allclose(stored[:, 0], summary[:, 0], atol=1e-3):
                msg = f"{farm} {name}: ensemble mean differs from experiment 336's"
                raise ValueError(msg)
            ens.write_atomic(path=common.stats_path(set_name=name, farm=farm), array=summary)
        print(f"{farm}: rated {rated[farm]:.2f} m/s", flush=True)


if __name__ == "__main__":
    main()
