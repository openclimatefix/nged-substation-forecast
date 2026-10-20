"""The onboarding arms (C2) and the short-history arms (C3), one held-out substation at a time.

**Onboarding (C2).** For each scored MW substation s, the pooled model P is fitted on every other
substation (leave one out), then `onboard_site` fits only s's site-level parameters on s's last 0,
2, 4, or 8 weeks before the 2025-07-01 cut. The per-site comparator S-4w is the per-site model
fitted on those 4 weeks alone.

**Short history (C3).** For each s, P is fitted on the full window in which s keeps only its last 3
months (2025-04-01 to 2025-06-30), so the shared modules are refitted with s inside. The
comparator S-3m is the per-site model on those 3 months.

**Spread.** A new substation has no residuals of its own, so the spread for s comes from the other
scored substations. For onboarding arm w, substation t's residuals come from a replica: P fitted
without t at the cut 12 months earlier, onboarded with t's w weeks before that cut, and scored over
the following 12 months. For the short-history arm, t's residuals come from P fitted with t holding
only its last 3 months before the replica cut. The replica fits for t include s's rows before the
replica cut, which a truly new substation would not have: the write-up states that caveat.

    uv run --no-sync python ml-experiments/00105-*/e00105_onboard.py [--stages c2 c3]
"""

import argparse
import copy
import json
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Final

import e00105_arms as arms
import e00105_common as common
import numpy as np
import torch
from studies.dp_features import FeatureCache
from studies.dp_onboarding import (
    FitCopy,
    PooledResiduals,
    after_mask,
    cross_fitted_spread,
    onboard_site,
    pooled_residuals,
    select_rows,
    site_mask,
)
from studies.dp_pooled import DpConfig, PooledBatch, PooledDemandModel

WEEKS: Final[tuple[int, ...]] = (0, 2, 4, 8)
SHORT_START: Final[datetime] = datetime(2025, 4, 1, tzinfo=common.FINAL_CUT.tzinfo)
SHORT_MONTHS: Final[int] = 3
ONBOARD_STEPS: Final[int] = 300
HELD_OUT_SITES: Final[tuple[str, ...]] = common.SCORED_SITES


def weeks_rows(*, batch: PooledBatch, cache: FeatureCache, site: str, weeks: int, cut: datetime):  # noqa: ANN201
    """Return the site's rows from the last `weeks` weeks before `cut` (none for 0 weeks)."""
    start = common.to_64(moment=cut - timedelta(weeks=weeks))
    keep = site_mask(batch=batch, label=site) & after_mask(batch=batch, cache=cache, after=start)
    if weeks == 0:
        keep = torch.zeros_like(keep)
    return select_rows(batch=batch, keep=keep)


def onboarded_copy(
    *,
    model: PooledDemandModel,
    batch: PooledBatch,
    cache: FeatureCache,
    site: str,
    weeks: int,
    cut: datetime,
) -> PooledDemandModel:
    """Return a copy of the leave-one-out `model` onboarded with `weeks` weeks of `site`."""
    onboarded = copy.deepcopy(model)
    rows = weeks_rows(batch=batch, cache=cache, site=site, weeks=weeks, cut=cut)
    onboard_site(
        model=onboarded,
        batch=rows,
        label=site,
        cache=cache,
        history_end=common.to_64(moment=cut),
        forecast_season_year=cut.year,
        steps=ONBOARD_STEPS,
        weeks_of_history=weeks,
    )
    return onboarded


def replica_residuals_by_arm(
    *, cache: FeatureCache, config: DpConfig, cut: datetime, stage: str
) -> dict[tuple[str, int], PooledResiduals]:
    """Return each held-out substation's replica residuals, per arm (weeks, or 0 for short)."""
    cut_r = common.months_before(moment=cut, months=common.REPLICA_MONTHS)
    train_r = common.pooled_batch_for(cache=cache, sites=common.POOL, cut=cut_r)
    out: dict[tuple[str, int], PooledResiduals] = {}
    for site in HELD_OUT_SITES:
        year = common.scoring_batch_for(cache=cache, sites=(site,), after=cut_r, through=cut)
        year = arms.remap_sites(batch=year, model_sites=common.POOL)
        if stage == "c2":
            keep = ~site_mask(batch=train_r, label=site)
            model, _, _ = arms.fit(
                batch=train_r, config=config, copy=FitCopy(name="loo", keep=keep)
            )
            for weeks in WEEKS:
                onboarded = onboarded_copy(
                    model=model, batch=train_r, cache=cache, site=site, weeks=weeks, cut=cut_r
                )
                out[(site, weeks)] = pooled_residuals(
                    model=onboarded, batch=year, weeks_of_history=weeks
                )
        else:
            keep = short_keep(batch=train_r, cache=cache, site=site, cut=cut_r)
            model, _, _ = arms.fit(
                batch=train_r, config=config, copy=FitCopy(name="short", keep=keep)
            )
            out[(site, 0)] = pooled_residuals(model=model, batch=year)
        print(f"replica residuals for {site} ({stage})", flush=True)
    return out


def short_keep(*, batch: PooledBatch, cache: FeatureCache, site: str, cut: datetime):  # noqa: ANN201
    """Return the rows to keep when `site` holds only its last 3 months before `cut`."""
    start = common.to_64(moment=common.months_before(moment=cut, months=SHORT_MONTHS))
    return ~site_mask(batch=batch, label=site) | after_mask(batch=batch, cache=cache, after=start)


def spread_without(  # noqa: ANN201
    *, residuals: dict[tuple[str, int], PooledResiduals], held_out: str, weeks: int
):
    """Return the spread for `held_out` from the other substations' residuals at `weeks`."""
    others = [r for (site, w), r in residuals.items() if site != held_out and w == weeks]
    combined = PooledResiduals(
        site=np.concatenate([r.site for r in others]),
        residual=np.concatenate([r.residual for r in others]),
        band=np.concatenate([r.band for r in others]),
        quarter=np.concatenate([r.quarter for r in others]),
        fit_excluded=tuple(sorted({s for r in others for s in r.fit_excluded})),
        weeks_of_history=others[0].weeks_of_history,
    )
    return cross_fitted_spread(
        residuals=combined,
        held_out=held_out,
        weeks_of_history=others[0].weeks_of_history,
        allow_seen=True,
    )


def run_c2(*, cache: FeatureCache) -> None:
    """Fit the onboarding arms and write `p_onb<weeks>w` and `s_4w`."""
    config = arms.config_for(kind="p", settings=arms.tuned())
    residuals = replica_residuals_by_arm(
        cache=cache, config=config, cut=common.FINAL_CUT, stage="c2"
    )
    train = common.pooled_batch_for(cache=cache, sites=common.POOL, cut=common.FINAL_CUT)
    forecasts: dict[int, dict] = {w: {} for w in WEEKS}
    for site in HELD_OUT_SITES:
        keep = ~site_mask(batch=train, label=site)
        model, _, peak = arms.fit(batch=train, config=config, copy=FitCopy(name="loo", keep=keep))
        score = arms.remap_sites(
            batch=common.scoring_batch_for(
                cache=cache,
                sites=(site,),
                after=common.FINAL_CUT,
                through=common.SCORING_END,
                all_present_rows=True,
            ),
            model_sites=common.POOL,
        )
        for weeks in WEEKS:
            onboarded = onboarded_copy(
                model=model, batch=train, cache=cache, site=site, weeks=weeks, cut=common.FINAL_CUT
            )
            spread = spread_without(residuals=residuals, held_out=site, weeks=weeks)
            forecasts[weeks] |= arms.forecasts_by_site(
                model=onboarded, score=score, replica=None, spread_for={site: spread}
            )
        print(f"onboarded {site} (peak {peak:.2f} GB)", flush=True)
    for weeks, result in forecasts.items():
        common.store_arm(
            name=f"p_onb{weeks}w", frame=arms.frames_from(cache=cache, forecasts=result)
        )
    run_per_site_short(cache=cache, residuals=residuals, weeks=4, name="s_4w", c2=True)


def run_c3(*, cache: FeatureCache) -> None:
    """Fit the short-history arms and write p_short3m and s_3m."""
    config = arms.config_for(kind="p", settings=arms.tuned())
    residuals = replica_residuals_by_arm(
        cache=cache, config=config, cut=common.FINAL_CUT, stage="c3"
    )
    train = common.pooled_batch_for(cache=cache, sites=common.POOL, cut=common.FINAL_CUT)
    forecasts: dict = {}
    for site in HELD_OUT_SITES:
        keep = short_keep(batch=train, cache=cache, site=site, cut=common.FINAL_CUT)
        model, _, _ = arms.fit(batch=train, config=config, copy=FitCopy(name="short", keep=keep))
        score = arms.remap_sites(
            batch=common.scoring_batch_for(
                cache=cache,
                sites=(site,),
                after=common.FINAL_CUT,
                through=common.SCORING_END,
                all_present_rows=True,
            ),
            model_sites=common.POOL,
        )
        spread = spread_without(residuals=residuals, held_out=site, weeks=0)
        forecasts |= arms.forecasts_by_site(
            model=model, score=score, replica=None, spread_for={site: spread}
        )
        print(f"short {site}", flush=True)
    common.store_arm(name="p_short3m", frame=arms.frames_from(cache=cache, forecasts=forecasts))
    run_per_site_short(cache=cache, residuals=residuals, weeks=0, name="s_3m", c2=False)


def run_per_site_short(
    *,
    cache: FeatureCache,
    residuals: dict[tuple[str, int], PooledResiduals],
    weeks: int,
    name: str,
    c2: bool,
) -> None:
    """Fit the per-site model on 4 weeks (C2) or 3 months (C3) and store its forecasts.

    The penalty is the one tuned for short per-site fits (`short_deviation`), and the spread is the
    same cross-fitted spread as the pooled arm of the same history.
    """
    settings = arms.tuned()
    settings["site_deviation"] = settings.get("short_deviation", settings["site_deviation"])
    config = arms.config_for(kind="s", settings=settings)
    forecasts: dict = {}
    for site in HELD_OUT_SITES:
        full = common.pooled_batch_for(cache=cache, sites=(site,), cut=common.FINAL_CUT)
        if c2:
            rows = weeks_rows(batch=full, cache=cache, site=site, weeks=weeks, cut=common.FINAL_CUT)
        else:
            start = common.to_64(
                moment=common.months_before(moment=common.FINAL_CUT, months=SHORT_MONTHS)
            )
            rows = select_rows(batch=full, keep=after_mask(batch=full, cache=cache, after=start))
        model, _, _ = arms.fit(batch=replace(rows), config=config)
        score = common.scoring_batch_for(
            cache=cache,
            sites=(site,),
            after=common.FINAL_CUT,
            through=common.SCORING_END,
            all_present_rows=True,
        )
        spread = spread_without(residuals=residuals, held_out=site, weeks=weeks if c2 else 0)
        forecasts |= arms.forecasts_by_site(
            model=model, score=score, replica=None, spread_for={site: spread}
        )
    common.store_arm(name=name, frame=arms.frames_from(cache=cache, forecasts=forecasts))


def main() -> None:
    """Run the requested stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=["c2", "c3"])
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    cache = common.load_cache()
    if "c2" in arguments.stages:
        run_c2(cache=cache)
    if "c3" in arguments.stages:
        run_c3(cache=cache)
    common.write_json(name="onboard_done.json", payload=json.loads(json.dumps(arguments.stages)))


if __name__ == "__main__":
    main()
