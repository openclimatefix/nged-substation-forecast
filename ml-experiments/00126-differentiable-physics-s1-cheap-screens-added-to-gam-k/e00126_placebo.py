"""Placebo arms for experiment 126's T2 gain, written after the Opus audit asked for them.

**T2 splits each heating column between the two halves of its hour band.** A split of the same
number of columns, with the same difference penalty (lambda 1), by a parity that has no time-of-day
or weekly structure should gain nothing if T2's gain comes from the structure of the split and not
from extra flexibility. Two placebo arms split each heating column into two pieces:

- `p_halfhour`: even against odd local half-hours of the day;
- `p_day`: even against odd days of the year.

Every arm here runs through B0's filter hyperparameters, so that only the terms differ, and T2 is
rerun the same way (`t2_kb0`). The sets are written beside the main arms as `<year>_<arm>` and
`<year>_<arm>_kb0` (filtered), for `e00126_report.py` to contrast with B0-K.

    uv run python ml-experiments/00114-*/e00126_placebo.py [--workers 6]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final

import e00126_common as common
import e00126_gate2 as gate2
import e00126_pipeline as pipeline
import numpy as np
import polars as pl
from studies.dp_assimilation import FilterHyperparameters
from studies.dp_features import FeatureCache
from studies.dp_pinned_capacity import read_vintages

PLACEBO_ARMS: Final[tuple[str, ...]] = ("t2", "p_halfhour", "p_day")
PENALTY_WEIGHT: Final[float] = 1.0
LABELS: Final[tuple[str, ...]] = ("replica_tuning", "tuning", "replicate", "scored")
SCORED_LABELS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")

_STATE: dict[str, Any] = {}


def _init_worker() -> None:
    """Load the cache and vintages once per worker process, with one thread."""
    common.configure_threads()
    _STATE["cache"] = common.load_cache()
    _STATE["vintages"] = read_vintages()


def fit_site(job: tuple[str, dict[str, Any]]) -> dict[str, pl.DataFrame]:
    """Fit the placebo arms at one substation and return their prediction frames.

    Args:
        job: The substation's label and the verified decisions.

    Returns:
        The frames by set name.
    """
    site, decisions = job
    cache: FeatureCache = _STATE["cache"]
    hyper = FilterHyperparameters.from_array(values=np.array(decisions["filter_hyper"]["b0"][site]))
    frames: dict[str, pl.DataFrame] = {}
    for arm_name in PLACEBO_ARMS:
        runs = {}
        for label in LABELS:
            run = pipeline.fit_year(
                cache=cache,
                vintages=_STATE["vintages"],
                site=site,
                fit_label=label,
                arm_name=arm_name,
                penalty_weight=PENALTY_WEIGHT,
            )
            if run is not None:
                runs[label] = run
        for label in SCORED_LABELS:
            if label not in runs or pipeline.PREVIOUS_LABEL[label] not in runs:
                continue
            scored = pipeline.score_year(cache=cache, runs=runs, label=label, hyper=hyper)
            pair = pipeline.scored_frames(cache=cache, site=site, arm_name=arm_name, scored=scored)
            if arm_name != "t2":
                frames[f"{label}_{arm_name}"] = pair[arm_name]
            frames[f"{label}_{arm_name}_kb0"] = pair[f"{arm_name}_k"].with_columns(
                arm=pl.lit(f"{arm_name}_kb0")
            )
    print(f"{site}: {len(frames)} sets", flush=True)
    return frames


def main() -> None:
    """Fit the placebo arms at every substation and write their prediction sets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    decisions = gate2.verified_decisions()
    common.load_cache()
    jobs = [(site, decisions) for site in common.SCORED_SITES]
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        results = list(pool.map(fit_site, jobs))
    names = dict.fromkeys(name for result in results for name in result)
    for name in names:
        frame = pl.concat([r[name] for r in results if name in r])
        common.store_arm(name=name, frame=frame)
    print(f"wrote {len(names)} prediction sets")


if __name__ == "__main__":
    main()
