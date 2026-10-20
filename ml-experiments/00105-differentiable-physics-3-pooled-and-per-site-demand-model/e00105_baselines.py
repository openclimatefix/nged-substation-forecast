"""Two cheap baselines for the onboarding and short-history arms, scored by the mean forecast.

(b) **Per-site convex model on the same history.** Part 2's convex model (`studies.dp_gam`, no
drift) fitted to a held-out substation's last 4 weeks or 3 months before the 2025-07-01 cut, with
its ridge chosen on the tuning year (the same truncation at 2024-07-01, scored 2024-07 to 2025-06
by the error of the mean forecast) from a grid around Part 2's chosen ridge.

(c) **Fleet-profile baseline.** The mean daily profile (day type and half-hour) of the other
scored substations over the 2 years before the cut, shifted so that its mean over the substation's
own last 4 weeks (or 3 months) equals the substation's mean over them.

Both are scored by the mean absolute error of the mean forecast, in points, over the six scored
substations on the scored year, outside the switching masks. The pooled arms' mean forecast error
comes from the same column. No quantiles are built for these two baselines.

    uv run --no-sync python ml-experiments/00105-*/e00105_baselines.py
"""

import json
from datetime import datetime, timedelta
from typing import Final

import e00105_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam

HISTORIES: Final[dict[str, str]] = {"4w": "weeks", "3m": "months"}
RIDGE_FACTORS: Final[tuple[float, ...]] = (0.1, 1.0, 10.0, 100.0)


def window_start(*, cut: datetime, history: str) -> datetime:
    """Return the first instant of the 4-week or 3-month history before `cut`."""
    if history == "4w":
        return cut - timedelta(weeks=4)
    return common.months_before(moment=cut, months=3)


def convex_forecast(
    *, cache: FeatureCache, site: str, cut: datetime, history: str, ridge: float, index: np.ndarray
) -> np.ndarray:
    """Return the mean forecast of the convex model on the substation's short history."""
    spec = GamSpec(drift="none", ridge=ridge)
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=cut,
        window=TrainingWindow(start=window_start(cut=cut, history=history)),
        season_years=common.SEASON_YEARS,
    )
    return predict_gam(cache=cache, fit=fit, index=index)


def tuned_ridges(*, cache: FeatureCache) -> dict[str, float]:
    """Choose the convex model's ridge for each history on the tuning year."""
    base = json.loads((common.PART_2_DIR / "work" / "tuned.json").read_text())["ridge"]["chosen"]
    out = {}
    for history in HISTORIES:
        errors = {}
        for factor in RIDGE_FACTORS:
            ridge = base * factor
            per_site = []
            for site in common.ELIGIBLE_SITES:
                index = common.scoring_index(
                    cache=cache, site=site, after=common.TUNING_CUT, through=common.TUNING_END
                )
                predicted = convex_forecast(
                    cache=cache,
                    site=site,
                    cut=common.TUNING_CUT,
                    history=history,
                    ridge=ridge,
                    index=index,
                )
                per_site.append(float(np.abs(cache.sites[site].y[index] - predicted).mean()))
            errors[factor] = float(np.mean(per_site))
        out[history] = base * min(errors, key=lambda factor: errors[factor])
        print(f"ridge {history}: {errors}", flush=True)
    return out


def fleet_forecast(
    *, cache: FeatureCache, site: str, cut: datetime, history: str, index: np.ndarray
) -> np.ndarray:
    """Return the fleet-profile forecast: the fleet's mean profile shifted to the site's level."""
    others = [s for s in common.SCORED_SITES if s != site]
    start = common.to_64(moment=window_start(cut=cut, history=history))
    cut64 = common.to_64(moment=cut)
    two_years = common.to_64(moment=common.months_before(moment=cut, months=common.WINDOW_MONTHS))
    calendar = cache.calendar
    profile_sum = np.zeros((4, 48))
    profile_count = np.zeros((4, 48))
    for other in others:
        features = cache.sites[other]
        keep = (
            features.default_rows
            & features.present
            & np.isfinite(features.y)
            & (cache.end_times > two_years)
            & (cache.end_times <= cut64)
        )
        rows = np.flatnonzero(keep)
        np.add.at(
            profile_sum, (calendar.day_type[rows], calendar.half_hour[rows]), features.y[rows]
        )
        np.add.at(profile_count, (calendar.day_type[rows], calendar.half_hour[rows]), 1.0)
    profile = profile_sum / np.maximum(profile_count, 1.0)
    features = cache.sites[site]
    own = np.flatnonzero(
        features.default_rows
        & features.present
        & np.isfinite(features.y)
        & (cache.end_times > start)
        & (cache.end_times <= cut64)
    )
    shift = float(
        features.y[own].mean() - profile[calendar.day_type[own], calendar.half_hour[own]].mean()
    )
    return profile[calendar.day_type[index], calendar.half_hour[index]] + shift


def main() -> None:
    """Score both baselines and write `work/baselines.json`."""
    cache = common.load_cache()
    ridges = tuned_ridges(cache=cache)
    rows = []
    for site in common.SCORED_SITES:
        index = common.scoring_index(
            cache=cache, site=site, after=common.FINAL_CUT, through=common.SCORING_END
        )
        keep = ~cache.sites[site].switching_mask[index]
        index = index[keep]
        actual = cache.sites[site].y[index]
        for history in HISTORIES:
            convex = convex_forecast(
                cache=cache,
                site=site,
                cut=common.FINAL_CUT,
                history=history,
                ridge=ridges[history],
                index=index,
            )
            fleet = fleet_forecast(
                cache=cache, site=site, cut=common.FINAL_CUT, history=history, index=index
            )
            rows.append(
                {
                    "site": site,
                    "history": history,
                    "convex_mae": float(np.abs(actual - convex).mean()),
                    "fleet_profile_mae": float(np.abs(actual - fleet).mean()),
                }
            )
    table = pl.DataFrame(rows)
    print(table)
    print(table.group_by("history").agg(pl.col("convex_mae", "fleet_profile_mae").mean()))
    common.write_json(name="baselines.json", payload={"ridges": ridges, "rows": rows})


if __name__ == "__main__":
    main()
