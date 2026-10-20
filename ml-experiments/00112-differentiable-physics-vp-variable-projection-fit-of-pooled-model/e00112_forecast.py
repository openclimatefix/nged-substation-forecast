"""Turn the saved fits into prediction sets for the replicate year and the scored year.

A year's forecast of an arm needs the arm's fit at the cut and its replica, the fit 12 months
earlier, whose residuals give the spread (the arm refitted 12 months earlier by the same method,
residual quantiles per hour band and quarter, centred on the mean forecast). The replicate year
(2024-07-01 to 2025-06-30) uses the fits at 2024-07-01 and 2023-07-01; the scored year
(2025-07-01 to 2026-06-30) uses the fits at 2025-07-01 and 2024-07-01. Prediction sets of the
replicate year carry the suffix `_rep`.

The arm is the lower-objective start: VP-P is one start for the whole pool, and VP-S one start for
each substation, decided by the training objective at the forecast cut. Both starts are stored too.

    uv run --no-sync python ml-experiments/00112-*/e00112_forecast.py [--years rep scored]
"""

import argparse
from datetime import datetime
from typing import Final

import e00112_common as common
import e00112_gradient as gradient
import e00112_vp as vp
import numpy as np
from studies.dp_features import FeatureCache
from studies.dp_gam import MIN_SPREAD_GROUP_ROWS
from studies.dp_onboarding import PooledResiduals, fit_centred_spread, pooled_residuals

MIN_SITE_ROWS: Final[int] = 1440
"""A substation with fewer training rows than this has no per-site fit at the cut."""
YEARS: Final[dict[str, tuple[datetime, datetime, datetime]]] = {
    "rep": (common.REPLICATE_CUT, common.REPLICA_CUT, common.FINAL_CUT),
    "scored": (common.FINAL_CUT, common.REPLICATE_CUT, common.SCORING_END),
}
"""For each year: its cut, its replica's cut, and the end of the scored rows."""


def suffix(*, year: str) -> str:
    """Return the set-name suffix of a year."""
    return "_rep" if year == "rep" else ""


def score_batch(  # noqa: ANN201
    *,
    cache: FeatureCache,
    sites: tuple[str, ...],
    cut: datetime,
    through: datetime,
    for_spread: bool = False,
):
    """Return the float64 rows of `sites` in (`cut`, `through`].

    Scored rows are every present row. The replica year that fits a spread uses the default rows
    only (outside the switching masks, the lockdown window, and experiment 103's pool filter), as
    experiment 105's spread did.
    """
    return common.scoring_batch_for(
        cache=cache,
        sites=sites,
        after=cut,
        through=through,
        all_present_rows=not for_spread,
        device=vp.DEVICE,
        dtype=common.BATCH_DTYPE,
    )


def lower_objective_start(*, names: dict[str, str], cut: datetime) -> str:
    """Return the start (`s0` or `s1`) whose fit at `cut` has the lower training objective."""
    objectives = {
        start: vp.read_record(name=name, cut=cut)["objective"] for start, name in names.items()
    }
    return min(objectives, key=lambda start: objectives[start])


def pooled_forecasts(
    *, cache: FeatureCache, name: str, start: str | None, year: str, network: bool = True
) -> dict[str, gradient.SiteForecast]:
    """Return a pooled arm's forecasts of every reported substation.

    Args:
        cache: The feature cache.
        name: The stored fit's name at each cut (`{name}` for `start=None`, else `{name}_{start}`).
        start: The start to use, or `None` for a fit that has none (P-map-same).
        year: `rep` or `scored`.
        network: False loads a fit made without the interaction network.

    Returns:
        A forecast per substation.
    """
    cut, cut_r, through = YEARS[year]
    fit_name = name if start is None else f"{name}_{start}"
    train = vp_train(cache=cache, cut=cut)
    train_r = vp_train(cache=cache, cut=cut_r)
    model = vp.load_fit(name=fit_name, cut=cut, batch=train, network=network)
    model_r = vp.load_fit(name=fit_name, cut=cut_r, batch=train_r, network=network)
    score = gradient.remap_sites(
        batch=score_batch(cache=cache, sites=common.REPORT_SITES, cut=cut, through=through),
        model_sites=common.POOL,
    )
    replica_year = gradient.remap_sites(
        batch=score_batch(
            cache=cache, sites=common.REPORT_SITES, cut=cut_r, through=cut, for_spread=True
        ),
        model_sites=common.POOL,
    )
    return gradient.forecasts_by_site(model=model, score=score, replica=(model_r, replica_year))


def vp_train(*, cache: FeatureCache, cut: datetime):  # noqa: ANN201
    """Return the float64 pooled training rows of the whole pool at `cut`."""
    return common.pooled_batch_for(
        cache=cache, sites=common.POOL, cut=cut, device=vp.DEVICE, dtype=common.BATCH_DTYPE
    )


def per_site_forecasts(
    *, cache: FeatureCache, name_prefix: str, year: str, start: str | None
) -> dict[str, gradient.SiteForecast]:
    """Return VP-S forecasts: each reported substation from its own fits at the cut and replica.

    A substation with no fit at the replica cut (its history began later) has its spread fitted on
    the other substations' replica residuals, as the gradient per-site arm does. With `start` set
    to `None` the lower-objective start of each substation is used.
    """
    cut, cut_r, through = YEARS[year]
    chosen: dict[str, str] = {}
    residuals: dict[str, PooledResiduals] = {}
    models = {}
    for site in common.REPORT_SITES:
        names = {s: f"{name_prefix}_{site.lower()}_{s}" for s in common.STARTS}
        train = common.pooled_batch_for(
            cache=cache, sites=(site,), cut=cut, device=vp.DEVICE, dtype=common.BATCH_DTYPE
        )
        if train.n_rows < MIN_SITE_ROWS:
            continue
        missing = [n for n in names.values() if not vp.has_fit(name=n, cut=cut)]
        if missing:
            msg = f"missing fits at {cut.date()}: {missing}"
            raise RuntimeError(msg)
        begin = start or lower_objective_start(names=names, cut=cut)
        chosen[site] = begin
        models[site] = vp.load_fit(name=names[begin], cut=cut, batch=train)
        if vp.has_fit(name=names[begin], cut=cut_r):
            train_r = common.pooled_batch_for(
                cache=cache, sites=(site,), cut=cut_r, device=vp.DEVICE, dtype=common.BATCH_DTYPE
            )
            model_r = vp.load_fit(name=names[begin], cut=cut_r, batch=train_r)
            replica_year = score_batch(
                cache=cache, sites=(site,), cut=cut_r, through=cut, for_spread=True
            )
            residuals[site] = pooled_residuals(model=model_r, batch=replica_year)
    out: dict[str, gradient.SiteForecast] = {}
    for site, model in models.items():
        own = residuals.get(site)
        pool = [r for other, r in residuals.items() if other != site] if own is None else [own]
        enough = [r for r in pool if len(r.residual) >= MIN_SPREAD_GROUP_ROWS]
        spread = fit_centred_spread(
            residual=np.concatenate([r.residual for r in enough]),
            band=np.concatenate([r.band for r in enough]),
            quarter=np.concatenate([r.quarter for r in enough]),
        )
        score = score_batch(cache=cache, sites=(site,), cut=cut, through=through)
        out |= gradient.forecasts_by_site(
            model=model, score=score, replica=None, spread_for={site: spread}
        )
    return out


def store(*, cache: FeatureCache, name: str, forecasts: dict[str, gradient.SiteForecast]) -> None:
    """Store a set of forecasts as one prediction set (an existing set is left alone)."""
    common.store_arm(name=name, frame=gradient.frames_from(cache=cache, forecasts=forecasts))
    print(f"stored {name}", flush=True)


def run_year(*, cache: FeatureCache, year: str) -> None:
    """Store every VP arm of a year."""
    cut = YEARS[year][0]
    tag = suffix(year=year)
    pooled_names = {s: f"vp_p_{s}" for s in common.STARTS}
    for start in common.STARTS:
        store(
            cache=cache,
            name=f"vp_p_{start}{tag}",
            forecasts=pooled_forecasts(cache=cache, name="vp_p", start=start, year=year),
        )
    chosen = lower_objective_start(names=pooled_names, cut=cut)
    store(
        cache=cache,
        name=f"vp_p{tag}",
        forecasts=pooled_forecasts(cache=cache, name="vp_p", start=chosen, year=year),
    )
    for start in common.STARTS:
        store(
            cache=cache,
            name=f"vp_s_{start}{tag}",
            forecasts=per_site_forecasts(cache=cache, name_prefix="vp_s", year=year, start=start),
        )
    store(
        cache=cache,
        name=f"vp_s{tag}",
        forecasts=per_site_forecasts(cache=cache, name_prefix="vp_s", year=year, start=None),
    )
    if all(vp.has_fit(name=f"vp_p_nonet_{s}", cut=cut) for s in common.STARTS):
        nonet_names = {s: f"vp_p_nonet_{s}" for s in common.STARTS}
        for start in common.STARTS:
            store(
                cache=cache,
                name=f"vp_p_nonet_{start}{tag}",
                forecasts=pooled_forecasts(
                    cache=cache, name="vp_p_nonet", start=start, year=year, network=False
                ),
            )
        store(
            cache=cache,
            name=f"vp_p_nonet{tag}",
            forecasts=pooled_forecasts(
                cache=cache,
                name="vp_p_nonet",
                start=lower_objective_start(names=nonet_names, cut=cut),
                year=year,
                network=False,
            ),
        )
    for same in ("p_map_same", "p_map_polish"):
        store(
            cache=cache,
            name=f"{same}{tag}",
            forecasts=pooled_forecasts(cache=cache, name=same, start=None, year=year),
        )


def main() -> None:
    """Store the prediction sets of the requested years."""
    common.require_gate_passed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", default=["rep", "scored"], choices=list(YEARS))
    arguments = parser.parse_args()
    cache = common.load_cache()
    for year in arguments.years:
        run_year(cache=cache, year=year)


if __name__ == "__main__":
    main()
