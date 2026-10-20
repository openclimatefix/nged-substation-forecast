"""The onboarding arms (C2v) and the short-history arms (C3v) of the variable-projection fit.

**Onboarding (C2v).** For each scored MW substation s, VP-P is refitted on every other substation
(leave one out), from a cold start. Then `onboard_site` fits only s's site-level
parameters on s's last 4 weeks before the cut, with every shared parameter and population mean
frozen from that leave-one-out fit and the site's own parameters solved exactly
(`vp_site_fitter`). The per-site comparator VP-S-4w is the per-site VP model on those 4 weeks.

**Short history (C3v).** For each s, VP-P is refitted on the full window in which s keeps only its
last 3 months (the shared modules are refitted with s inside), from a cold start. The
comparator VP-S-3m is the per-site VP model on those 3 months.

**Spread.** A new substation has no residuals of its own, so the spread for s comes from the other
scored substations: substation t's residuals come from the same procedure at the replica cut, 12
months earlier, scored over the following 12 months. The replica fits include s's rows before the
replica cut, which a truly new substation would not have.

The leave-one-out and short-history refits at the three cuts are saved with `--stage fit`, each
from a cold start; the forecasts of the replicate year and the scored year follow with
`--stage forecast`.

    uv run --no-sync python ml-experiments/00112-*/e00112_onboard.py --stage fit|forecast
"""

import argparse
import copy
from datetime import datetime, timedelta
from typing import Final

import e00112_common as common
import e00112_forecast as forecast
import e00112_gradient as gradient
import e00112_vp as vp
import numpy as np
from studies.dp_features import FeatureCache
from studies.dp_onboarding import (
    PooledResiduals,
    after_mask,
    cross_fitted_spread,
    onboard_site,
    pooled_residuals,
    select_rows,
    site_mask,
)
from studies.dp_pooled import PooledBatch, PooledDemandModel
from studies.dp_pooled_vp import vp_site_fitter

HELD_OUT_SITES: Final[tuple[str, ...]] = common.SCORED_SITES
ONBOARD_WEEKS: Final[int] = 4
SHORT_MONTHS: Final[int] = 3
REFIT_FALLBACK_STEPS: Final[int] = 0
"""The leave-one-out and short-history refits use L-BFGS only, to keep inside the GPU budget."""


def train_batch(*, cache: FeatureCache, cut: datetime) -> PooledBatch:
    """Return the float64 pooled training rows of the whole pool at `cut`."""
    return forecast.vp_train(cache=cache, cut=cut)


def short_keep(*, batch: PooledBatch, cache: FeatureCache, site: str, cut: datetime):  # noqa: ANN201
    """Return the rows to keep when `site` holds only its last 3 months before `cut`."""
    start = common.to_64(moment=common.months_before(moment=cut, months=SHORT_MONTHS))
    return ~site_mask(batch=batch, label=site) | after_mask(batch=batch, cache=cache, after=start)


def weeks_rows(  # noqa: ANN201
    *, batch: PooledBatch, cache: FeatureCache, site: str, weeks: int, cut: datetime
):
    """Return the site's rows from the last `weeks` weeks before `cut`."""
    start = common.to_64(moment=cut - timedelta(weeks=weeks))
    keep = site_mask(batch=batch, label=site) & after_mask(batch=batch, cache=cache, after=start)
    return select_rows(batch=batch, keep=keep)


def stage_fit(*, cache: FeatureCache) -> None:
    """Save the leave-one-out and short-history refits at every cut.

    Each refit starts cold, as the arm it is compared with does: a start taken from the full VP-P
    fit would carry the held-out substation's older rows into its shared parameters.
    """
    for cut in common.CUTS:
        batch = train_batch(cache=cache, cut=cut)
        settings = common.vp_settings(cut=cut, round_name="round2")
        for site in HELD_OUT_SITES:
            for kind, keep in (
                ("loo", ~site_mask(batch=batch, label=site)),
                ("short3m", short_keep(batch=batch, cache=cache, site=site, cut=cut)),
            ):
                vp.fit_cached(
                    name=f"vp_{kind}_{site.lower()}",
                    cut=cut,
                    batch=select_rows(batch=batch, keep=keep),
                    settings=settings,
                    start="s0",
                    fallback_steps=REFIT_FALLBACK_STEPS,
                )


def onboarded_copy(
    *, cache: FeatureCache, site: str, cut: datetime, weeks: int
) -> tuple[PooledDemandModel, PooledBatch, PooledBatch]:
    """Return the leave-one-out fit onboarded with `weeks` weeks of `site`, and its batches."""
    batch = train_batch(cache=cache, cut=cut)
    # The saved fit has the whole pool's layout, and its `site_active` flag marks the held-out
    # substation as having no rows, which `onboard_site` requires.
    model = vp.load_fit(name=f"vp_loo_{site.lower()}", cut=cut, batch=batch)
    onboarded = copy.deepcopy(model)
    rows = weeks_rows(batch=batch, cache=cache, site=site, weeks=weeks, cut=cut)
    settings = common.vp_settings(cut=cut, round_name="round2")
    onboard_site(
        model=onboarded,
        batch=rows,
        label=site,
        cache=cache,
        history_end=common.to_64(moment=cut),
        forecast_season_year=cut.year,
        steps=0,
        weeks_of_history=weeks,
        fitter=vp_site_fitter(settings=settings),
    )
    return onboarded, batch, rows


def replica_residuals(
    *, cache: FeatureCache, cut: datetime, kind: str
) -> dict[str, PooledResiduals]:
    """Return each held-out substation's residuals from the replica fits at `cut_r`.

    For `kind="onboard"` the leave-one-out fit is onboarded with 4 weeks before the replica cut; for
    `kind="short"` the short-history refit is used as it is. The residuals cover the 12 months after
    the replica cut. A substation with no row in its onboarding weeks is left out.
    """
    cut_r = common.months_before(moment=cut, months=common.REPLICA_MONTHS)
    out: dict[str, PooledResiduals] = {}
    for site in HELD_OUT_SITES:
        year = gradient.remap_sites(
            batch=forecast.score_batch(
                cache=cache, sites=(site,), cut=cut_r, through=cut, for_spread=True
            ),
            model_sites=common.POOL,
        )
        if kind == "onboard":
            model, _, rows = onboarded_copy(cache=cache, site=site, cut=cut_r, weeks=ONBOARD_WEEKS)
            if rows.n_rows == 0 or year.n_rows == 0:
                continue
            out[site] = pooled_residuals(model=model, batch=year, weeks_of_history=ONBOARD_WEEKS)
        else:
            batch = train_batch(cache=cache, cut=cut_r)
            model = vp.load_fit(name=f"vp_short3m_{site.lower()}", cut=cut_r, batch=batch)
            short = per_site_short(cache=cache, cut=cut_r, site=site, weeks=None)
            if year.n_rows == 0 or short.n_rows == 0:
                continue
            out[site] = pooled_residuals(model=model, batch=year)
        print(f"replica residuals for {site} at {cut_r.date()} ({kind})", flush=True)
    return out


def spread_without(*, residuals: dict[str, PooledResiduals], held_out: str, weeks: int | None):  # noqa: ANN201
    """Return the spread for `held_out` from the other substations' replica residuals."""
    others = [r for site, r in residuals.items() if site != held_out]
    combined = PooledResiduals(
        site=np.concatenate([r.site for r in others]),
        residual=np.concatenate([r.residual for r in others]),
        band=np.concatenate([r.band for r in others]),
        quarter=np.concatenate([r.quarter for r in others]),
        fit_excluded=tuple(sorted({s for r in others for s in r.fit_excluded})),
        weeks_of_history=others[0].weeks_of_history,
    )
    return cross_fitted_spread(
        residuals=combined, held_out=held_out, weeks_of_history=weeks, allow_seen=True
    )


def per_site_short(
    *, cache: FeatureCache, cut: datetime, site: str, weeks: int | None
) -> PooledBatch:
    """Return the single-site rows of the per-site short fit: 4 weeks, or the last 3 months."""
    full = forecast.common.pooled_batch_for(
        cache=cache, sites=(site,), cut=cut, device=vp.DEVICE, dtype=common.BATCH_DTYPE
    )
    if weeks is not None:
        return weeks_rows(batch=full, cache=cache, site=site, weeks=weeks, cut=cut)
    start = common.to_64(moment=common.months_before(moment=cut, months=SHORT_MONTHS))
    return select_rows(batch=full, keep=after_mask(batch=full, cache=cache, after=start))


def stage_forecast(*, cache: FeatureCache) -> None:
    """Store the onboarding and short-history arms of the replicate year and the scored year."""
    for year, (cut, _, through) in forecast.YEARS.items():
        tag = forecast.suffix(year=year)
        settings = common.vp_settings(cut=cut, round_name="round2")
        onboard_residuals = replica_residuals(cache=cache, cut=cut, kind="onboard")
        short_residuals = replica_residuals(cache=cache, cut=cut, kind="short")
        onb: dict[str, gradient.SiteForecast] = {}
        onb_s: dict[str, gradient.SiteForecast] = {}
        short: dict[str, gradient.SiteForecast] = {}
        short_s: dict[str, gradient.SiteForecast] = {}
        for site in HELD_OUT_SITES:
            score = gradient.remap_sites(
                batch=forecast.score_batch(cache=cache, sites=(site,), cut=cut, through=through),
                model_sites=common.POOL,
            )
            single = forecast.score_batch(cache=cache, sites=(site,), cut=cut, through=through)
            model, _, _ = onboarded_copy(cache=cache, site=site, cut=cut, weeks=ONBOARD_WEEKS)
            spread = spread_without(residuals=onboard_residuals, held_out=site, weeks=ONBOARD_WEEKS)
            onb |= gradient.forecasts_by_site(
                model=model, score=score, replica=None, spread_for={site: spread}
            )
            tr = per_site_short(cache=cache, cut=cut, site=site, weeks=ONBOARD_WEEKS)
            if tr.n_rows:
                per_model, _ = vp.fit_cached(
                    name=f"vp_s_4w_{site.lower()}",
                    cut=cut,
                    batch=tr,
                    settings=settings,
                    start="s0",
                    fallback_steps=common.FALLBACK_STEPS_SECONDARY,
                )
                onb_s |= gradient.forecasts_by_site(
                    model=per_model, score=single, replica=None, spread_for={site: spread}
                )
            batch = train_batch(cache=cache, cut=cut)
            short_model = vp.load_fit(name=f"vp_short3m_{site.lower()}", cut=cut, batch=batch)
            short_spread = spread_without(residuals=short_residuals, held_out=site, weeks=None)
            short |= gradient.forecasts_by_site(
                model=short_model, score=score, replica=None, spread_for={site: short_spread}
            )
            tr3 = per_site_short(cache=cache, cut=cut, site=site, weeks=None)
            if tr3.n_rows:
                per_model3, _ = vp.fit_cached(
                    name=f"vp_s_3m_{site.lower()}",
                    cut=cut,
                    batch=tr3,
                    settings=settings,
                    start="s0",
                    fallback_steps=common.FALLBACK_STEPS_SECONDARY,
                )
                short_s |= gradient.forecasts_by_site(
                    model=per_model3, score=single, replica=None, spread_for={site: short_spread}
                )
            print(f"{year}: onboarding and short history for {site}", flush=True)
        for name, result in (
            ("vp_onb4w", onb),
            ("vp_s_4w", onb_s),
            ("vp_short3m", short),
            ("vp_s_3m", short_s),
        ):
            forecast.store(cache=cache, name=f"{name}{tag}", forecasts=result)


def main() -> None:
    """Run the requested stage."""
    common.require_gate_passed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["fit", "forecast"])
    arguments = parser.parse_args()
    cache = common.load_cache()
    if arguments.stage == "fit":
        stage_fit(cache=cache)
    else:
        stage_forecast(cache=cache)


if __name__ == "__main__":
    main()
