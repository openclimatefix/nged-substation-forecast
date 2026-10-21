"""Control P1: the two-stage arm reproduces experiment 111's stored `gamk` forecasts.

With experiment 111's fixed filter settings, the corrected ensemble mean of this folder's pipeline
(copied from experiment 111's) must equal experiment 111's stored mean forecast in the scored year,
to 1e-4 points. The check reads only the mean, which does not depend on the spread.

    uv run --no-sync python ml-experiments/00135-*/e00135_p1.py
"""

import json
from typing import Final

import e00135_arms as arms
import e00135_common as common
import e00135_ens as ens
import numpy as np
import polars as pl
import torch

TOLERANCE_POINTS: Final[float] = 1e-4


def check_site(*, site: str) -> dict:
    """Return the largest absolute difference in the mean forecast at one substation."""
    torch.set_num_threads(1)
    cache = common.load_cache()
    tuned = arms.read_tuned()
    setup = ens.load_setup(cache=cache, site=site).with_calibration(
        calibration=arms.load_calibration(site=site).with_solar_shrink(
            solar_shrink=tuned["solar_shrink"]
        )
    )
    arm = arms.ARMS["ts"]
    fits = arms.load_arm_fits(arm=arm, site=site)
    trace = arms.build_trace(
        cache=cache,
        setup=setup,
        arm=arm,
        fits=fits,
        hyper=np.array(tuned["filter_hyper"]["gamk"][site]),
    )
    frame = ens.forecast_frame(
        cache=cache,
        setup=setup,
        fits=fits,
        spread=ens.ZERO_SPREAD,
        trace=trace,
        run_indices=setup.indices(periods={"scored"}),
        arm="ts",
    )
    stored = pl.read_parquet(common.SHARED_WORK_DIR / "dp_preds" / f"gamk_scored_{site}.parquet")
    keys = ["power_fcst_init_time", "valid_time"]
    joined = frame.select(*keys, "mean").join(
        stored.select(*keys, pl.col("mean").alias("stored_mean")), on=keys, how="inner"
    )
    difference = (joined["mean"] - joined["stored_mean"]).abs().to_numpy()
    return {
        "site": site,
        "rows": joined.height,
        "rows_in_either": max(frame.height, stored.height),
        "max_abs_difference": float(difference.max()),
    }


def main() -> None:
    """Check every decision substation and write `work/p1.json`."""
    results = [check_site(site=site) for site in common.DECISION_SITES]
    common.write_json_atomic(
        path=common.WORK_DIR / "p1.json",
        content={
            "tolerance": TOLERANCE_POINTS,
            "passed": all(r["max_abs_difference"] <= TOLERANCE_POINTS for r in results),
            "sites": results,
        },
    )
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
