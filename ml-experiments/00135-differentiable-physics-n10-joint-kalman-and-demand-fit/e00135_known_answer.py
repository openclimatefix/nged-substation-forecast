"""Control P4: can the joint fit recover known shapes when its assumptions hold?

For each decision substation at one cut, the target is replaced by a synthetic one: the two-stage
fit's own curves (the truth) plus a slowly moving level, a day-level anomaly with decay, and white
noise, on the rows the fit uses. TS, J1, and J2 (ridge weight 1) are refitted on it, and each fitted
curve is compared with the truth. The pre-registered pass condition is that J1's error is no larger
than TS's.

    uv run --no-sync python ml-experiments/00135-*/e00135_known_answer.py
"""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Final

import e00135_arms as arms
import e00135_common as common
import e00135_controls as controls
import e00135_fits as fits_module
import numpy as np
from studies.dp_features import FeatureCache
from studies.dp_gam import GamFit, prepare_fit
from studies.dp_joint_fit import HALF_HOUR, stream_design, stream_rows
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity, read_vintages

CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
SEED: Final[int] = 135
LEVEL_STEP_POINTS: Final[float] = 0.03
ANOMALY_POINTS: Final[float] = 1.5
ANOMALY_DECAY: Final[float] = 0.6
NOISE_POINTS: Final[float] = 1.0
HALF_HOURS_PER_DAY: Final[int] = 48


def synthetic_cache(
    *, cache: FeatureCache, site: str, truth: GamFit, pinned: PinnedCapacity
) -> FeatureCache:
    """Return the cache whose target for `site` is the truth plus the known disturbances."""
    rng = np.random.default_rng(SEED)
    features = cache.sites[site]
    start = (
        cache.end_times[
            prepare_fit(
                cache=cache,
                site=site,
                spec=fits_module.gamk_spec(),
                cut=CUT,
                window=controls.WINDOW,
                season_years=common.SEASON_YEARS,
            )[1][0]
        ]
        - HALF_HOUR
    )
    rows = stream_rows(cache=cache, site=site, cut=CUT, start=start)
    design, target = stream_design(
        cache=cache, site=site, layout=truth.layout, rows=rows, pinned=pinned
    )
    solar = target - features.y[rows]
    n = len(rows)
    level = np.cumsum(rng.normal(0.0, LEVEL_STEP_POINTS, n))
    n_days = n // HALF_HOURS_PER_DAY + 1
    anomaly = np.zeros(n_days)
    for day in range(1, n_days):
        anomaly[day] = ANOMALY_DECAY * anomaly[day - 1] + rng.normal(0.0, ANOMALY_POINTS)
    disturbance = (
        level + np.repeat(anomaly, HALF_HOURS_PER_DAY)[:n] + rng.normal(0.0, NOISE_POINTS, n)
    )
    y = features.y.copy()
    y[rows] = design @ truth.theta[0] + disturbance - solar
    return replace(cache, sites={**cache.sites, site: replace(features, y=y)})


def recover(*, site: str) -> dict:
    """Return each arm's curve error against the truth at one substation."""
    cache = common.load_cache()
    hyper = arms.read_hyper(site=site)
    pinned = pinned_capacity(vintages=read_vintages(), table=cache.sites[site].table, cut=CUT)
    truth = controls.fit_three(cache=cache, site=site, cut=CUT, hyper=hyper)["ts"]
    changed = synthetic_cache(cache=cache, site=site, truth=truth, pinned=pinned)
    fits = controls.fit_three(cache=changed, site=site, cut=CUT, hyper=hyper)
    _, index = prepare_fit(
        cache=cache,
        site=site,
        spec=fits_module.gamk_spec(),
        cut=CUT,
        window=controls.WINDOW,
        season_years=common.SEASON_YEARS,
    )
    true_curves = controls.curves(cache=cache, fit=truth, index=index)
    out = {}
    for arm, fit in fits.items():
        fitted = controls.curves(cache=cache, fit=fit, index=index)
        out[arm] = {
            term: float(
                np.sqrt(
                    np.mean(
                        (
                            (fitted[term] - true_curves[term])
                            - (fitted[term] - true_curves[term]).mean()
                        )
                        ** 2
                    )
                )
            )
            for term in controls.TERMS
        }
    return out


def main() -> None:
    """Run the known-answer control at every decision substation and write `work/p4.json`."""
    results = {site: recover(site=site) for site in common.DECISION_SITES}
    total = {
        arm: float(sum(sum(r[arm].values()) for r in results.values()))
        for arm in ("ts", "j1", "j2")
    }
    common.write_json_atomic(
        path=common.WORK_DIR / "p4.json",
        content={
            "sites": results,
            "summed_centred_rms": total,
            "passed": total["j1"] <= total["ts"],
        },
    )
    print(total)


if __name__ == "__main__":
    main()
