"""Check that this experiment's A0 and A2 reproduce experiment 106's stored sets.

The 2024 and 2025 fits of the six MW sites must give the stored `replicate_a0`, `scored_a0`,
`replicate_a2`, `scored_a2` (and `_d2` to `_d7`) quantiles and means to 1e-4 points, and the `a0`
filter hyperparameters refitted from this experiment's 2023 residuals must match experiment 106's
`tuned.json` to 1e-3. Either failure stops the experiment.

    uv run python ml-experiments/00108-*/e00108_reproduce.py
"""

import json
import sys

import e00108_common as common
import e00108_factor as factor_lib
import e00108_pipeline as pipeline
import numpy as np
import polars as pl
from studies.dp_assimilation import fit_filter_hyperparameters
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import read_predictions

TOLERANCE_POINTS = 1e-4
HYPER_TOLERANCE = 1e-3
KEYS = ["time_series_id", "valid_time", "power_fcst_init_time"]


def compare(*, mine: pl.DataFrame, stored: pl.DataFrame, name: str) -> float:
    """Return the largest absolute difference in mean and quantiles between two sets."""
    joined = mine.join(stored, on=KEYS, how="inner", suffix="_stored")
    if joined.height != stored.height or joined.height != mine.height:
        print(f"  {name}: row mismatch ({mine.height}, {stored.height}, {joined.height} rows)")
        return float("inf")
    worst = 0.0
    for column in ["mean", *QUANTILE_COLUMNS]:
        difference = (joined[column] - joined[f"{column}_stored"]).to_numpy()
        worst = max(worst, float(np.nanmax(np.abs(difference))))
    return worst


def main() -> int:
    """Rebuild A0 and A2, compare them with experiment 106's sets, and refit `a0`."""
    cache = common.load_cache()
    all_fits = factor_lib.load_fits(cache=cache)
    panel = factor_lib.build_panel(cache=cache, all_fits=all_fits)
    a0_hyper = pipeline.a0_hyper_from_part4()
    failed = False
    for label in ("replicate", "scored"):
        mine: dict[str, list[pl.DataFrame]] = {}
        for site in common.SCORED_SITES:
            inputs = pipeline.year_inputs(cache=cache, all_fits=all_fits, site=site, label=label)
            if inputs is None or inputs.replica is None:
                continue
            for suffix, frame in pipeline.a0_frame(
                cache=cache, inputs=inputs, same_day=False
            ).items():
                mine.setdefault(f"{label}_a0{suffix}", []).append(frame)
            for suffix, frame in pipeline.filter_arm_frames(
                cache=cache,
                all_fits=all_fits,
                panel=panel,
                setup=None,
                a0_hyper=a0_hyper,
                inputs=inputs,
                spec=pipeline.ArmSpec("a2", local="a0", shared=False, days=pipeline.ALL_DAYS),
            ).items():
                mine.setdefault(f"{label}_a2{suffix}", []).append(frame)
        for name, frames in mine.items():
            stored = read_predictions(experiment_id=106, name=name, row_set_id=common.ROW_SET_ID)
            gap = compare(mine=pl.concat(frames), stored=stored, name=name)
            status = "ok" if gap <= TOLERANCE_POINTS else "FAIL"
            failed |= gap > TOLERANCE_POINTS
            print(f"{name}: max abs difference {gap:.2e} points [{status}]")
    window = (
        common.to_64(moment=common.cut_of(year=common.TUNING_YEAR)),
        common.to_64(moment=common.year_end(cut=common.cut_of(year=common.TUNING_YEAR))),
    )
    streams = {
        site: factor_lib.keep_window(
            stream=panel.residual[site],
            window=factor_lib.window_positions(
                cache=cache,
                after=common.cut_of(year=common.TUNING_YEAR),
                through=common.year_end(cut=common.cut_of(year=common.TUNING_YEAR)),
            ),
        )
        for site in common.DECISION_SITES
    }
    refit = fit_filter_hyperparameters(
        residuals=streams, band=cache.calendar.band, end_times=cache.end_times, window=window
    )
    tuned = json.loads((common.PART4_DIR / "work" / "tuned.json").read_text())["filter_hyper"]["a0"]
    for site, hyper in refit.items():
        gap = float(np.max(np.abs(hyper.as_array() - np.array(tuned[site]))))
        status = "ok" if gap <= HYPER_TOLERANCE else "FAIL"
        failed |= gap > HYPER_TOLERANCE
        print(f"a0 hyperparameters {site}: max abs difference {gap:.2e} [{status}]")
    print("REPRODUCTION FAILED" if failed else "reproduction passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
