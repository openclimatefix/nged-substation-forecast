"""Refit the convex model every 4 weeks or every week, and write the refit arms' quantiles.

For each substation the script fits arm A1 (D1, all years, pinned capacity) at a cut every 28 or 7
days from 2023-07-01 (from 2024-07-01 at D9) to the end of June 2026, each fit predicting only the
days until the next cut. The out-of-sample residuals of the fits form one stream, which carries the
Kalman filter across the refits (`studies.dp_assimilation.stitch_out_of_sample_residuals` joins
streams the same way).

The spread of a refit arm in a scored year is the centred residual quantiles per hour band and
calendar quarter of the same refit arm in the year before, which has the staleness of a refit
forecast (0 to 4 weeks, or 0 to 1 week). The spread of the refit arm with the filter is the
previous year's residuals run through the same filter. The sets are `<cut>_a1_4weekly`,
`<cut>_a1_weekly`, `<cut>_a3_4weekly`, and `<cut>_a3_weekly`.

    uv run python ml-experiments/00106-*/e00106_refit.py [--workers 6]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Final

import e00106_common as common
import e00106_g as g
import numpy as np
import polars as pl
from studies.dp_assimilation import (
    FilterHyperparameters,
    correction_at,
    day_ahead_initialisation,
    post_filter_lead_spread,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import ReplicaResiduals, ResidualSpread, fit_gam, predict_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

CADENCE_DAYS: Final[dict[str, int]] = {"4weekly": 28, "weekly": 7}
LAST_DAY: Final[datetime] = common.year_end(cut=common.SCORED_CUT)


def refit_chain(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    site: str,
    first_cut: datetime,
    step_days: int,
    definition: g.ArmDefinition,
) -> np.ndarray:
    """Refit every `step_days` days from `first_cut` and return the stitched mean forecast.

    Args:
        cache: The feature cache.
        vintages: The real-time capacity vintages.
        site: The substation's label.
        first_cut: The first fit's cut.
        step_days: Days between cuts.
        definition: The arm.

    Returns:
        The mean forecast on the grid, `NaN` where no fit predicted: each fit predicts the rows
        with a reading that end after its cut and no later than the next cut.
    """
    mean = np.full(len(cache.end_times), np.nan)
    cut = first_cut
    while cut < LAST_DAY:
        following = min(cut + timedelta(days=step_days), LAST_DAY)
        pinned = pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut)
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=[definition.spec],
            cut=cut,
            window=definition.window,
            season_years=common.SEASON_YEARS,
            pinned=pinned,
        )
        index = common.scoring_index(cache=cache, site=site, after=cut, through=following)
        mean[index] = predict_gam(cache=cache, fit=fit, index=index)
        cut = following
    return mean


def _year_rows(*, cache: FeatureCache, site: str, label: str) -> np.ndarray:
    """Return the rows with a reading in the scored year that starts at the cut `label`."""
    cut = common.CUTS[label]
    return common.scoring_index(cache=cache, site=site, after=cut, through=common.year_end(cut=cut))


def _previous_year_residuals(
    *, cache: FeatureCache, site: str, mean: np.ndarray, label: str
) -> ReplicaResiduals:
    """Return the refit arm's residuals in the year before `label`, as a spread's source."""
    previous = {"replicate": "tuning", "scored": "replicate"}[label]
    features = cache.sites[site]
    index = _year_rows(cache=cache, site=site, label=previous)
    index = index[np.isfinite(mean[index])]
    return ReplicaResiduals(
        residuals=features.y[index] - mean[index],
        band=cache.calendar.band[index],
        quarter=cache.calendar.quarter[index],
        index=index,
        fit_cut=common.CUTS[previous],
        switching=features.switching_mask[index],
    )


def fit_site(*, site: str) -> dict[str, Any]:
    """Fit both cadences for one substation and return the prediction frames.

    Args:
        site: The substation's label.

    Returns:
        A dict with `frames`, the set name to predictions.
    """
    cache: FeatureCache = common.load_cache()
    vintages = read_vintages()
    tuned = g.read_tuned()
    definition = g.arm_definitions(tuned=tuned)["a1"]
    hyper = FilterHyperparameters.from_array(values=np.array(tuned["filter_hyper"]["a1"][site]))
    features = cache.sites[site]
    first_cut = common.REPLICATE_CUT if site == "D9" else common.TUNING_CUT
    frames: dict[str, pl.DataFrame] = {}
    for cadence, step_days in CADENCE_DAYS.items():
        mean = refit_chain(
            cache=cache,
            vintages=vintages,
            site=site,
            first_cut=first_cut,
            step_days=step_days,
            definition=definition,
        )
        stream = np.where(np.isfinite(mean), features.y - mean, np.nan)
        trace = run_kalman(residual=stream, band=cache.calendar.band, hyper=hyper)
        for label in ("replicate", "scored"):
            if site == "D9" and label == "replicate":
                continue
            index = _year_rows(cache=cache, site=site, label=label)
            previous = _previous_year_residuals(cache=cache, site=site, mean=mean, label=label)
            keep = ~previous.switching
            plain_spread = ResidualSpread.fit(
                residuals=previous.residuals[keep],
                band=previous.band[keep],
                quarter=previous.quarter[keep],
                centre=True,
            )
            band = cache.calendar.band[index]
            quarter = cache.calendar.quarter[index]
            frames[f"{label}_a1_{cadence}"] = common.prediction_frame(
                cache=cache,
                site=site,
                index=index,
                quantiles=plain_spread.quantiles(mean=mean[index], band=band, quarter=quarter),
                mean=mean[index],
                arm=f"a1_{cadence}",
            )
            lead_spread = post_filter_lead_spread(
                end_times=cache.end_times,
                replica=previous,
                run=lambda *, residual, band, _h=hyper: run_kalman(
                    residual=residual, band=band, hyper=_h
                ),
            )
            cutoff = usable_cutoff_positions(
                end_times=cache.end_times,
                initialisation=day_ahead_initialisation(
                    end_times=cache.end_times[index], days_ahead=1
                ),
                delay_hours=3.5,
            )
            correction = correction_at(trace=trace, cutoff=cutoff, target=index, band=band)
            corrected = mean[index] + correction.total
            frames[f"{label}_a3_{cadence}"] = common.prediction_frame(
                cache=cache,
                site=site,
                index=index,
                quantiles=lead_spread.quantiles(
                    mean=corrected, band=band, quarter=quarter, lead_band=0
                ),
                mean=corrected,
                arm=f"a3_{cadence}",
                level=correction.level,
                anomaly=correction.anomaly,
            )
        print(f"{site} {cadence}: done", flush=True)
    return {"frames": frames}


def _worker(site: str) -> dict[str, Any]:
    """Run `fit_site` in a worker process with one thread."""
    import torch

    torch.set_num_threads(1)
    return fit_site(site=site)


def main() -> None:
    """Fit all substations at both cadences and write the prediction sets."""
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
