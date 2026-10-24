"""Known-answer test for x1: recover a speed scale that is known by direction sector.

Takes one farm's ERA5 training rows, builds a no-adapter core on them, multiplies its hub-height
speed by a known scale per direction sector (eight values from 0.90 to 1.12), and sets the target
to the core's noise-free power at that speed. A fresh x1 table core is then fitted to that target.
The test passes when, in every sector, the mean ratio of x1's effective speed to the true effective
speed lies within 3% of one. Writes `work/known_answer.json`.

    PYTHONPATH as for the fit; uv run --no-sync python ml-experiments/00338-*/e00338_known_answer.py
"""

import json
import os

os.environ.setdefault("E00303_TORCH_THREADS", "2")

import e00303_fit as fit303  # isort: skip  # ty: ignore[unresolved-import]
import e00303_models as models  # isort: skip  # ty: ignore[unresolved-import]
import e00326_fit as fit326  # isort: skip  # ty: ignore[unresolved-import]
import e00338_adapters as adapters
import e00338_common as common
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

FARM = "W1"
SECTOR_SCALES = np.linspace(0.90, 1.12, common.DIRECTION_SECTORS)[[3, 0, 6, 2, 7, 1, 5, 4]].astype(
    np.float32
)
"""A fixed, non-monotone assignment of the eight known scales to the eight sectors."""


def main() -> None:
    """Fit x1 to power generated with known sector scales and write the recovery by sector."""
    sites = common.c303.site_table().filter(pl.col("site") == FARM)
    era5 = common.c303.era5_rows(sites=sites)
    frame = models.add_derived_columns(
        frame=common.c303.training_table(sites=sites, era5=era5)
    ).filter(pl.col("valid_time") < common.c303.BASE_END)

    class Rows:
        train = frame

    columns = fit326.train_columns(data=Rows)
    core, _ = fit303.fit_era5_core(train=frame, adapter="none", seed=0, with_idle_layer=True)
    sector = adapters.sector_of(sin=columns["direction_sin"], cos=columns["direction_cos"])
    true_speed = adapters.base_speed(core=core, columns=columns) * SECTOR_SCALES[sector]
    with torch.no_grad():
        target = adapters.power_from_speed(core=core, speed=torch.as_tensor(true_speed)).numpy()
    edges = adapters.make_edges(
        columns=columns, ratio_bins=common.RATIO_BINS, temperature_bins=common.TEMPERATURE_BINS
    )
    fitted = adapters.fit_table_core(
        train_columns=columns, target=target, edges=edges, with_idle_layer=True, seed=0
    )
    recovered = adapters.hub_speed(core=fitted, columns=columns)
    usable = true_speed > 3.0
    by_sector = {}
    for index in range(common.DIRECTION_SECTORS):
        inside = usable & (sector == index)
        by_sector[index] = float(np.mean(recovered[inside] / true_speed[inside]))
    worst = max(abs(v - 1.0) for v in by_sector.values())
    report = {
        "known_scales": SECTOR_SCALES.tolist(),
        "mean_recovered_over_true_speed_by_sector": by_sector,
        "worst_abs_error": worst,
        "passes": bool(worst < common.KNOWN_ANSWER_TOLERANCE),
    }
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (common.WORK_DIR / "known_answer.json").write_text(json.dumps(report, indent=1))
    print(report)


if __name__ == "__main__":
    main()
