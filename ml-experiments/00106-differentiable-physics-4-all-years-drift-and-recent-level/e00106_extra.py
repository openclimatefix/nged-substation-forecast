"""Fit experiment 106's delay sweep, lead-matched filter arms, and D10-mask arms.

- **Delay sweep.** Arm A3 with the filter's telemetry delay at 0, 9, and 24 hours instead of 3.5
  (`<cut>_a3_delay0`, `_delay9`, `_delay24`). The spread is rebuilt with the same delay.
- **Lead-matched arms** for the descriptive comparison with lagged XGBoost, which is initialised at
  00:00 UTC on the day before the valid day (leads of 24.5 to 48 hours) or six days before (144.5
  to 168 hours): A3 and A1 under those initialisations (`<cut>_a3_lag24`, `_a3_lag144`,
  `_a1_lag24`, `_a1_lag144`).
- **D10's November 2024 jump.** A1 refitted at the scored cut with D10's rows from 2024-11-08 to
  2024-12-01 masked from training (`scored_a1_d10mask`, D10 only).

    uv run python ml-experiments/00106-*/e00106_extra.py [--workers 6]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Final

import e00106_common as common
import e00106_g as g
import numpy as np
import polars as pl
from studies.dp_assimilation import FilterHyperparameters, post_filter_lead_spread
from studies.dp_features import FeatureCache
from studies.dp_pinned_capacity import read_vintages

DELAYS_HOURS: Final[tuple[int, ...]] = (0, 9, 24)
LAGGED: Final[dict[str, tuple[int, int, int]]] = {
    "lag24": (1, 0, 1),
    "lag144": (6, 0, 2),
}
"""Days before the valid day, the UTC hour of the initialisation, and the lead band."""
D10_JUMP: Final[tuple[datetime, datetime]] = (
    datetime(2024, 11, 8, tzinfo=UTC),
    datetime(2024, 12, 1, tzinfo=UTC),
)
"""D10's unlabelled November 2024 jump, masked from training in one arm."""


def _lag_initialisation(*, end_times: np.ndarray, days_before: int, hour: int) -> np.ndarray:
    """Return each valid time's initialisation: `days_before` days earlier at `hour` UTC."""
    day = (end_times - np.timedelta64(30, "m")).astype("datetime64[D]")
    return (day - np.timedelta64(days_before, "D")).astype("datetime64[us]") + np.timedelta64(
        hour, "h"
    )


def fit_site(*, site: str) -> dict[str, Any]:
    """Fit the extra arms of one substation and return the prediction frames.

    Args:
        site: The substation's label.

    Returns:
        A dict with `frames`, the set name to predictions.
    """
    cache: FeatureCache = common.load_cache()
    vintages = read_vintages()
    tuned = g.read_tuned()
    definitions = g.arm_definitions(tuned=tuned)
    hyper = FilterHyperparameters.from_array(values=np.array(tuned["filter_hyper"]["a1"][site]))
    frames: dict[str, pl.DataFrame] = {}
    for label in ("replicate", "scored"):
        if site == "D9" and label == "replicate":
            continue
        cut = common.CUTS[label]
        run = g.run_site(
            cache=cache, vintages=vintages, site=site, cut=cut, definition=definitions["a1"]
        )
        assert run.replica is not None
        for delay in DELAYS_HOURS:
            spread = post_filter_lead_spread(
                end_times=cache.end_times,
                replica=run.replica,
                run=g.trace_builder(kind="kalman", hyper=hyper),
                delay_hours=float(delay),
            )
            result = g.filtered_forecast(
                cache=cache,
                run=run,
                kind="kalman",
                hyper=hyper,
                delay_hours=float(delay),
                lead_spread=spread,
            )
            frames[f"{label}_a3_delay{delay}"] = common.prediction_frame(
                cache=cache,
                site=site,
                index=run.index,
                quantiles=result.quantiles,
                mean=result.corrected_mean,
                arm=f"a3_delay{delay}",
                level=result.correction.level,
                anomaly=result.correction.anomaly,
            )
        spread = post_filter_lead_spread(
            end_times=cache.end_times,
            replica=run.replica,
            run=g.trace_builder(kind="kalman", hyper=hyper),
        )
        valid = cache.end_times[run.index]
        for name, (days_before, hour, lead_band) in LAGGED.items():
            init = _lag_initialisation(end_times=valid, days_before=days_before, hour=hour)
            result = g.filtered_forecast(
                cache=cache,
                run=run,
                kind="kalman",
                hyper=hyper,
                lead_spread=spread,
                initialisation=init,
            )
            frames[f"{label}_a3_{name}"] = common.prediction_frame(
                cache=cache,
                site=site,
                index=run.index,
                quantiles=spread.quantiles(
                    mean=result.corrected_mean,
                    band=cache.calendar.band[run.index],
                    quarter=cache.calendar.quarter[run.index],
                    lead_band=lead_band,
                ),
                mean=result.corrected_mean,
                arm=f"a3_{name}",
                init=init,
                level=result.correction.level,
                anomaly=result.correction.anomaly,
            )
            frames[f"{label}_a1_{name}"] = common.prediction_frame(
                cache=cache,
                site=site,
                index=run.index,
                quantiles=g.uncorrected_quantiles(cache=cache, run=run),
                mean=run.mean,
                arm=f"a1_{name}",
                init=init,
            )
    if site == "D10":
        features = cache.sites[site]
        jump = (cache.end_times > common.to_64(moment=D10_JUMP[0])) & (
            cache.end_times <= common.to_64(moment=D10_JUMP[1])
        )
        masked = replace(
            cache,
            sites={
                **cache.sites,
                site: replace(
                    features,
                    switching_mask=features.switching_mask | jump,
                    default_rows=features.default_rows & ~jump,
                ),
            },
        )
        run = g.run_site(
            cache=masked,
            vintages=vintages,
            site=site,
            cut=common.SCORED_CUT,
            definition=definitions["a1"],
        )
        frames["scored_a1_d10mask"] = common.prediction_frame(
            cache=cache,
            site=site,
            index=run.index,
            quantiles=g.uncorrected_quantiles(cache=masked, run=run),
            mean=run.mean,
            arm="a1_d10mask",
        )
    print(f"{site}: {len(frames)} sets", flush=True)
    return {"frames": frames}


def _worker(site: str) -> dict[str, Any]:
    """Run `fit_site` in a worker process with one thread."""
    import torch

    torch.set_num_threads(1)
    return fit_site(site=site)


def main() -> None:
    """Fit all substations and write the prediction sets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.load_cache()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(_worker, common.SCORED_SITES))
    names = dict.fromkeys(name for result in results for name in result["frames"])
    for name in names:
        frame = pl.concat([r["frames"][name] for r in results if name in r["frames"]])
        common.store_arm(name=name, frame=frame)
    print(f"wrote {len(names)} prediction sets")


if __name__ == "__main__":
    main()
