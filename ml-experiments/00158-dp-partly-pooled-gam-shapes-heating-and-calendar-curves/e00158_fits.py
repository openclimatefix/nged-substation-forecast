"""Fit the partly pooled convex model of experiment 109 at every monthly cut, for experiment 158.

Every arm is experiment 111's GAM-K design (24-month window, ridge 0.1, no drift, capacity pinned
to experiment 45's real-time vintage) fitted jointly across the six-substation fleet with
`studies.dp_gam_pooled.fit_pooled`. An arm is named by its kind and its deviation penalties, in rows
of evidence:

- `k0`: no pooling (penalty 0), the per-site fit, which reproduces experiment 111's `gamk` fits.
- `ph<h>`: the heating curve pooled at penalty `h`, 0 on every other family.
- `ps<h>_<o>`: the heating curve pooled at `h`, and the lighting, daily profile, holiday periods,
  Christmas tent, and harmonics pooled at `o`.
- `tied`: complete pooling of the families of `ps`, one shared coefficient vector with a free band
  mean per substation.

**Two histories.** `full` gives every substation its last 24 months. `3m` gives the target
substation its last 3 months and every other substation its last 24 months, and takes the target in
turn, so the stored fits of substation `s` come from the joint fit in which `s` was the target. The
per-site comparator on 3 months uses experiment 109's tuned ridge of 10; the pooled arms keep 0.1.

Each stored file holds one list of `GamFit` per arm and substation, ascending by cut.

    uv run python ml-experiments/00158-*/e00158_fits.py --configs k0 ph300 ph1000 [--history full]
"""

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Final

import e00158_common as common
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamFit, TrainingWindow
from studies.dp_gam_pooled import PooledSite, PoolingStrength, fit_pooled
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

HISTORIES: Final[tuple[str, ...]] = ("full", "3m")


def strength_of(*, config: str) -> PoolingStrength:
    """Return the deviation penalties an arm name stands for.

    Args:
        config: `k0`, `tied`, `ph<h>`, or `ps<h>_<o>`.

    Returns:
        The penalties; zero for `k0` and `tied`.

    Raises:
        ValueError: If the name is not one of those forms.
    """
    if config in {"k0", "tied"}:
        return PoolingStrength.uniform(value=0.0)
    if config.startswith("ph"):
        return PoolingStrength(heating=float(config[2:]), other=0.0)
    if config.startswith("ps"):
        heating, other = config[2:].split("_")
        return PoolingStrength(heating=float(heating), other=float(other))
    msg = f"unknown arm {config!r}"
    raise ValueError(msg)


def fit_path(*, config: str, history: str, site: str) -> Path:
    """Return the file that holds an arm's fits at a substation."""
    return common.FITS_ROOT / history / config / f"{site}.pkl"


def load_fits(*, config: str, history: str, site: str) -> list[GamFit]:
    """Return the stored monthly fits of an arm at a substation, ascending by cut."""
    return pickle.loads(fit_path(config=config, history=history, site=site).read_bytes())


def pooled_sites(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    cut: datetime,
    members: tuple[str, ...],
    target: str | None,
    ridge: float,
) -> list[PooledSite]:
    """Return the inputs of one joint fit; the target, if any, has the short window and `ridge`.

    A substation fitted alone may use a late vintage, as experiment 111's per-site fits do, because
    no other substation reads its level.
    """
    sites = []
    for site in members:
        short = site == target
        sites.append(
            PooledSite(
                site=site,
                spec=common.base_spec(ridge=ridge if short else common.RIDGE),
                window=TrainingWindow(months=common.SHORT_MONTHS)
                if short
                else common.base_window(),
                season_years=tuple(common.SEASON_YEARS),
                pinned=pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut),
                allow_late_vintage=len(members) == 1,
            )
        )
    return sites


def fleet_at(*, cache: FeatureCache, vintages: pl.DataFrame, cut: datetime) -> tuple[str, ...]:
    """Return the substations that can join a joint fit at a cut.

    A substation whose pinned capacity is a `growth_prior` vintage, fitted more than six weeks after
    the cut, is left out, because that hindsight level would move the fleet mean and so every other
    substation's forecast. Such a substation is fitted alone at that cut in every arm.
    """
    return tuple(
        site
        for site in common.FLEET
        if pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut).source
        != "growth_prior"
    )


def _alone(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    cut: datetime,
    site: str,
    target: str | None,
    ridge: float,
) -> GamFit:
    """Return a substation's fit alone: it is excluded from the fleet, or has a fleet of one."""
    return fit_pooled(
        cache=cache,
        sites=pooled_sites(
            cache=cache, vintages=vintages, cut=cut, members=(site,), target=target, ridge=ridge
        ),
        cut=cut,
        strengths=[PoolingStrength.uniform(value=0.0)],
    ).fits[site]


def fit_cut(  # noqa: PLR0912
    *, cut: datetime, configs: list[str], history: str
) -> dict[tuple[str, str], GamFit]:
    """Return every arm's fit of every substation at one cut."""
    cache = common.load_cache()
    vintages = read_vintages()
    fleet = fleet_at(cache=cache, vintages=vintages, cut=cut)
    targets: list[str | None] = [None] if history == "full" else list(common.FLEET)
    out: dict[tuple[str, str], GamFit] = {}
    for target in targets:
        scored = list(common.FLEET) if target is None else [target]
        alone = target is not None and target not in fleet
        members = (target,) if alone else fleet
        # The per-site comparator and the pooled arms differ only in the target's ridge on a short
        # history.
        groups: dict[float, list[str]] = {}
        for name in configs:
            ridge = common.SHORT_RIDGE if (name == "k0" and target) else common.RIDGE
            groups.setdefault(ridge, []).append(name)
        for ridge, names in groups.items():
            partial = [n for n in names if n != "tied"]
            joint_sites = pooled_sites(
                cache=cache, vintages=vintages, cut=cut, members=members, target=target, ridge=ridge
            )
            if partial:
                joint = fit_pooled(
                    cache=cache,
                    sites=joint_sites,
                    cut=cut,
                    strengths=[
                        PoolingStrength.uniform(value=0.0) if alone else strength_of(config=n)
                        for n in partial
                    ],
                )
                for site in scored:
                    for member, name in enumerate(partial):
                        if site in members:
                            fit = joint.fits[site]
                            out[(name, site)] = replace(fit, theta=fit.theta[member : member + 1])
            if "tied" in names:
                if alone or len(members) == 1:
                    tied_fits = {
                        s: _alone(
                            cache=cache,
                            vintages=vintages,
                            cut=cut,
                            site=s,
                            target=target,
                            ridge=ridge,
                        )
                        for s in scored
                    }
                else:
                    tied_fits = fit_pooled(cache=cache, sites=joint_sites, cut=cut, tied=True).fits
                for site in scored:
                    if site in tied_fits:
                        out[("tied", site)] = replace(
                            tied_fits[site], theta=tied_fits[site].theta[:1]
                        )
        # A full-history substation left out of the fleet is fitted alone in every arm.
        for site in scored:
            if site in members:
                continue
            for name in configs:
                out[(name, site)] = _alone(
                    cache=cache,
                    vintages=vintages,
                    cut=cut,
                    site=site,
                    target=target,
                    ridge=common.RIDGE,
                )
    return out


def _job(job: tuple[datetime, list[str], str]) -> tuple[datetime, dict]:
    torch.set_num_threads(1)
    cut, configs, history = job
    return cut, fit_cut(cut=cut, configs=configs, history=history)


def main() -> None:
    """Fit the arms at every monthly cut in parallel and store each arm's fits per substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--history", choices=HISTORIES, default="full")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    started = time.monotonic()
    cuts = common.monthly_cuts()
    jobs = [(cut, args.configs, args.history) for cut in cuts]
    by_key: dict[tuple[str, str], list[GamFit]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for cut, fits in pool.map(_job, jobs):
            for key, fit in fits.items():
                by_key.setdefault(key, []).append(fit)
            print(f"{cut:%Y-%m-%d} done, {time.monotonic() - started:.0f} s", flush=True)
    for (config, site), fits in by_key.items():
        target = fit_path(config=config, history=args.history, site=site)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        temporary.write_bytes(pickle.dumps(fits))
        temporary.rename(target)


if __name__ == "__main__":
    main()
