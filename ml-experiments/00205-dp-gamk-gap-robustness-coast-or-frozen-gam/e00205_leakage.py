"""Leakage control: arm D's forecast of a run must not change when later actuals change.

For the first event runs at one substation this replaces, with random numbers, the actual of every
row whose valid time is at or after the run's origin (the winter night offset and recipe R read
actuals), rebuilds arm D, and compares the run's own forecast rows. Rows of the run itself keep
their stored mean and quantiles by construction, so the control compares the run's mean forecast
(which holds the offset and recipe R shifts), not its actuals.

    uv run --no-sync python ml-experiments/00205-*/e00205_leakage.py [--site D10] [--events 3]
"""

import argparse

import e00205_base as base
import e00205_common as common
import e00205_corrections as corr
import e00205_filters as filters
import e00205_prep as prep
import e00205_run as run
import numpy as np
import polars as pl


def main() -> None:
    """Run the control and write `work/leakage.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="D10")
    parser.add_argument("--events", type=int, default=3)
    args = parser.parse_args()
    frame0 = pl.read_parquet(prep.base_path(site=args.site))
    stream, band = prep.read_stream(site=args.site)
    trace = filters.filter_stream(residual=stream, band=band, settings=base.pool_settings()).trace
    frame = corr.add_calendar(
        frame=base.arm_frame(mean_frame=base.arm_mean_frame(base=frame0, trace=trace))
    )
    actual = frame["actual"].to_numpy().astype(np.float64)
    reference = run.d_pipeline(frame=frame, hidden_actual=actual)
    runs = frame.group_by("run").agg(pl.col("power_fcst_init_time").first()).sort("run")
    rng = np.random.default_rng(205)
    results = []
    for event in runs["run"].to_list()[300 : 300 + args.events]:
        origin = runs.filter(pl.col("run") == event)["power_fcst_init_time"][0]
        later = (frame["valid_time"] >= origin).to_numpy()
        noisy = np.where(later, rng.normal(50.0, 20.0, len(actual)), actual)
        perturbed_frame = frame.with_columns(pl.Series("actual", noisy.astype(np.float32)))
        perturbed = run.d_pipeline(frame=perturbed_frame, hidden_actual=noisy)
        mine = pl.col("run") == event
        gap = (reference.filter(mine)["mean"] - perturbed.filter(mine)["mean"]).abs().max()
        results.append(
            {"run": int(event), "origin": str(origin), "max_abs_mean_difference": float(gap)}
        )
        print(results[-1], flush=True)
    common.write_json_atomic(path=common.WORK_DIR / "leakage.json", content={"events": results})


if __name__ == "__main__":
    main()
