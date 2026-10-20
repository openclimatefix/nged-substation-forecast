"""Lagless XGBoost on 4 weeks or 3 months of a substation's history (X-4w and X-3m).

These are the descriptive tree-model comparators of the onboarding and short-history arms. Each
held-out substation's model is trained only on its last 4 weeks, or last 3 months, before the
2025-07-01 cut, with experiment 87's hyperparameters and the same inputs as `e00105_xgb.py`. The
quantiles are XGBoost's own, so no cross-fitted spread is added, and the hyperparameters are not
tuned for the short windows.

    uv run --no-sync python ml-experiments/00105-*/e00105_xgb_short.py
"""

from datetime import timedelta

import e00105_common as common
import polars as pl
from e00105_xgb import SiteJob, feature_matrix, fit_booster, predict
from studies.dp_gam import QUANTILE_LEVELS, TrainingWindow, training_rows


def fit_site(*, site: str) -> dict[str, pl.DataFrame]:
    """Return the X-4w and X-3m predictions of one substation."""
    cache = common.load_cache()
    features = cache.sites[site]
    cut = common.FINAL_CUT
    job = SiteJob(
        cache=cache,
        site=site,
        cut=cut,
        index=common.scoring_index(cache=cache, site=site, after=cut, through=common.SCORING_END),
    )
    window = training_rows(
        features=features, cache=cache, cut=cut, window=TrainingWindow(months=12)
    )
    ends = cache.end_times[window]
    levels = tuple(QUANTILE_LEVELS)
    frames = {}
    for name, start in (
        ("x_4w", common.to_64(moment=cut - timedelta(weeks=4))),
        ("x_3m", common.to_64(moment=common.months_before(moment=cut, months=3))),
    ):
        train = window[ends > start]
        booster = fit_booster(
            x=feature_matrix(cache=cache, site=site, index=train),
            y=features.y[train],
            levels=levels,
            start="estimated",
            margin=None,
        )
        quantiles = predict(booster=booster, x=job.x_score(), margin=None, n_levels=len(levels))
        frames[name] = job.frame_of(quantiles=quantiles)
        print(f"{site} {name}: {len(train)} training rows", flush=True)
    return frames


def main() -> None:
    """Fit the arms of every scored substation and store them."""
    results = [fit_site(site=site) for site in common.SCORED_SITES]
    for name in ("x_4w", "x_3m"):
        common.store_arm(name=name, frame=pl.concat([result[name] for result in results]))


if __name__ == "__main__":
    main()
