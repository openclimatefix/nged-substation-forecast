# ruff: noqa: E501
r"""Score experiment 258's real-record arms at the two bulk supply points (descriptive, no injection).

Arms: K0 (the base), GAM_ONLY, ORACLE (the 66 labelled load shifts as logged, known at once), and
LR_B20 (the carried mask; a Primary scored in two folds gets the mean of the two folds' corrections).
Each arm's forecast change is the sum over the 15 Primaries of its stored correction minus K0's.
Row groups use the labelled real load shifts, which touch a BSP through the Primaries they involve.

```bash
systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \\
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \\
    uv run --no-sync python e00271_real.py
```
"""

from __future__ import annotations

import e00271_data as dat
import e00271_harness as h
import e00271_inputs as inputs
import e00271_score as sc
import e00271_worlds as worlds
import numpy as np
import polars as pl

REAL_ARMS: tuple[str, ...] = ("K0", "GAM_ONLY", "ORACLE", "LR_B20")


def real_shifts() -> list[worlds.Shift]:
    """The labelled real load shifts as shifts: a one-series event is its own donor and receiver."""
    out = []
    for row in h.labelled_events().iter_rows(named=True):
        ids = [i for i in row["series"] if i in h.SERIES]
        if not ids:
            continue
        donor, receiver = (ids[0], ids[-1])
        out.append(worlds.Shift(donor, receiver, int(row["onset"]), int(row["end"]), 0.0, False))
    return out


def real_deltas() -> dict[str, dict[str, np.ndarray]]:
    """Each real-record arm's forecast change against K0 at each BSP, in MW."""
    bsp = worlds.bsp_of_series()
    capacity = {
        int(r["time_series_id"]): float(r["effective_capacity_mw"])
        for r in worlds.children().iter_rows(named=True)
    }
    _, _, targets = h.origin_grid()
    names = sorted(set(bsp.values()))
    out = {arm: {n: np.zeros(targets.shape) for n in names} for arm in REAL_ARMS}
    for series in h.SERIES:
        stored = np.load(dat.WORK_DIR / "harness" / f"series_{series}.npz", allow_pickle=True)
        k0 = stored["K0"].astype(np.float64)
        folds = [k for k in stored.files if k.startswith("LR_b20_f")]
        arms = {
            "GAM_ONLY": stored["GAM_ONLY"],
            "ORACLE": stored["ORACLE"],
            "LR_B20": np.mean([stored[k] for k in folds], axis=0),
        }
        for arm, correction in arms.items():
            out[arm][bsp[series]] += (correction.astype(np.float64) - k0) * capacity[series] / 100.0
    return out


def main() -> None:
    """Score the four arms and write the cost and quality tables."""
    ctx = sc.build_context()
    forecast = real_deltas()
    cost, quality = sc.score_world(
        ctx=ctx,
        shifts=real_shifts(),
        forecast=forecast,
        observed={},
        arm_names=REAL_ARMS,
        targets=("observed",),
    )
    out = inputs.RESULTS_DIR / "real_record"
    out.mkdir(parents=True, exist_ok=True)
    cost.write_parquet(out / "cost.parquet")
    quality.write_parquet(out / "quality.parquet")
    pl.Config.set_tbl_rows(100)
    print(cost.filter(pl.col("arm") != "perfect").sort("group", "arm"))
    print(quality.sort("group", "arm"))


if __name__ == "__main__":
    main()
