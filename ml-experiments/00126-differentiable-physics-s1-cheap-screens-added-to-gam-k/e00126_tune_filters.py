"""Fit F-mix's four log factors on the tuning year and write them, with a hash, before any fit.

The factors come from T-all's out-of-sample tuning-year residuals: the fit at 2023-07-01 forecasts
the tuning year, and the fit at 2022-07-01 is its replica. D9 takes the mean of the five decision
substations' log factors. The file is `work/decisions_filters.json`, with
`decisions_filters.sha256`.

    uv run --no-sync python ml-experiments/00126-*/e00126_tune_filters.py [--workers 5]
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import e00126_common as common
import e00126_fmix as fmix
import e00126_pipeline as pipeline
import e00126_tune as tune
import numpy as np
from studies.dp_assimilation import FilterHyperparameters
from studies.dp_pinned_capacity import read_vintages

_STATE: dict[str, Any] = {}


def _init_worker() -> None:
    """Load the cache and vintages once per worker process, with one thread."""
    common.configure_threads()
    _STATE["cache"] = common.load_cache()
    _STATE["vintages"] = read_vintages()


def _fit_site(job: tuple[str, list[float]]) -> tuple[str, list[float], float]:
    """Return one substation's log factors and its cold-snap threshold."""
    site, hyper_values = job
    cache = _STATE["cache"]
    weight = tune.t_all_weight()
    run = pipeline.fit_year(
        cache=cache,
        vintages=_STATE["vintages"],
        site=site,
        fit_label="tuning",
        arm_name="t_all",
        penalty_weight=weight,
    )
    assert run is not None
    regimes = fmix.regime_map(cache=cache, site=site)
    factors = fmix.fit_factors(
        residual=run.residual_stream(cache=cache),
        band=cache.calendar.band,
        regimes=regimes.regime,
        hyper=FilterHyperparameters.from_array(values=np.array(hyper_values)),
    )
    print(f"{site}: log factors {np.round(factors, 3).tolist()}", flush=True)
    return site, factors.tolist(), regimes.cold_threshold_c


def main() -> None:
    """Fit the factors and write the hashed decisions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    arguments = parser.parse_args()
    decisions = json.loads((common.WORK_DIR / "decisions.json").read_text())
    common.load_cache()
    jobs = [(site, decisions["filter_hyper"]["t_all"][site]) for site in common.DECISION_SITES]
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        results = list(pool.map(_fit_site, jobs))
    factors = {site: values for site, values, _ in results}
    factors["D9"] = np.mean(list(factors.values()), axis=0).tolist()
    record = {
        "log_factors": factors,
        "cold_threshold_c": {site: threshold for site, _, threshold in results},
    }
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    (common.WORK_DIR / "decisions_filters.json").write_text(text)
    (common.WORK_DIR / "decisions_filters.sha256").write_text(
        hashlib.sha256(text.encode()).hexdigest() + "\n"
    )
    print(text)


if __name__ == "__main__":
    main()
