"""Tune every arm's spread on the tuning months, then forecast the periods, for experiment 118.

**Step `tune`.** For each scored site the script forecasts the tuning runs with every arm and a zero
spread, fits each arm's centred residual spread on the corrected ensemble mean, and scores a spread
factor per lead band on `SPREAD_FACTOR_GRID`. **Step `choose`** takes, per arm and lead band, the
factor with the lowest mean quantile score at the five decision sites and writes `tuned.json`.
**Step `forecast`** forecasts the chosen periods with the tuned spreads and writes one parquet file
per arm, period, and site under `work/preds/`.

The replicate months and the scored year are forecast only after the known-answer gate has passed
(`work/gate.json`) and the module's Opus review has landed (`work/OPUS_REVIEW_DONE`); the tuning
months have no such guard, because no replicate month and no scored run is read.

    uv run python ml-experiments/00118-*/e00118_run.py tune [--workers 6]
    uv run python ml-experiments/00118-*/e00118_run.py choose
    uv run python ml-experiments/00118-*/e00118_run.py forecast --periods tuning [replicate scored]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00118_common as common
import e00118_forecast as fc
import e00118_streams as streams
import numpy as np
import polars as pl
import torch
from studies.dp_ens import LeadHourSpread, choose_fit
from studies.dp_gam import ResidualSpread

PREDS_DIR: Final[Path] = common.WORK_DIR / "preds"
SPREAD_DIR: Final[Path] = common.WORK_DIR / "spread"
TUNED_PATH: Final[Path] = common.WORK_DIR / "tuned.json"
GATE_PATH: Final[Path] = common.WORK_DIR / "gate.json"
REVIEW_PATH: Final[Path] = common.WORK_DIR / "OPUS_REVIEW_DONE"
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
GUARDED_PERIODS: Final[frozenset[str]] = frozenset({"replicate", "scored"})


def require_clearance(*, periods: list[str]) -> None:
    """Raise unless the gate has passed and the module's Opus review has landed.

    Args:
        periods: The periods about to be forecast; only replicate and scored are guarded.

    Raises:
        RuntimeError: If a guarded period is asked for and the gate or the review is missing.
    """
    if not GUARDED_PERIODS & set(periods):
        return
    if not GATE_PATH.exists() or json.loads(GATE_PATH.read_text())["blocking_failed"] != 0:
        msg = "the known-answer gate has not passed; no replicate or scored run may start"
        raise RuntimeError(msg)
    if not REVIEW_PATH.exists():
        msg = "the module's Opus review has not landed; no replicate or scored run may start"
        raise RuntimeError(msg)


def spread_path(*, arm: str, site: str) -> Path:
    """Return the file that holds an arm's fitted spread at a site."""
    return SPREAD_DIR / f"{arm}_{site}.npz"


def save_spread(*, arm: str, site: str, spread: LeadHourSpread) -> None:
    """Store a spread's tables."""
    path = spread_path(arm=arm, site=site)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, tables=np.stack([band.table for band in spread.by_lead_band]))


def load_spread(*, arm: str, site: str, factors: list[float]) -> LeadHourSpread:
    """Return a stored spread with the given factors."""
    with np.load(spread_path(arm=arm, site=site)) as stored:
        tables = stored["tables"]
    spread = LeadHourSpread(
        by_lead_band=tuple(ResidualSpread(table=table) for table in tables),
        factors=np.ones(len(tables)),
    )
    return spread.with_factors(factors=factors)


def state_arm(*, arm: str) -> str:
    """Return the arm whose filter state `arm` uses."""
    return fc.STATE_ARM_OF.get(arm, arm)


def site_inputs(*, site: str) -> tuple:
    """Load a site's cache, setup, fits, streams, and every arm's filter."""
    cache = common.load_cache()
    setup = common.load_setup(cache=cache, site=site)
    fits = common.load_fits(site=site)
    data = streams.build_site_streams(cache=cache, setup=setup, fits=fits)
    return cache, setup, fits, data, fc.build_filters(cache=cache, data=data)


def tune_site(site: str) -> dict:
    """Fit every arm's spread at one site on the tuning runs and score the factor grid.

    Args:
        site: The substation's label.

    Returns:
        Per arm, the factor grid's per-lead-band score sums, as `{arm: {factor: [[sum, n]...]}}`.
    """
    torch.set_num_threads(1)
    cache, setup, fits, data, filters = site_inputs(site=site)
    runs = setup.indices(periods={"tuning"})
    pieces = [
        piece
        for index in runs
        if (piece := fc.run_members(cache=cache, setup=setup, fits=fits, index=index)) is not None
    ]
    own_arms = fc.available_arms(filters=filters, arms_wanted=fc.OWN_SPREAD_ARMS)
    zero_frames: dict[str, list[pl.DataFrame]] = {arm: [] for arm in own_arms}
    corrected: dict[str, list[fc.Corrected]] = {arm: [] for arm in own_arms}
    for members in pieces:
        for arm in own_arms:
            result = fc.corrected_members(
                cache=cache,
                setup=setup,
                data=data,
                filtered=filters[state_arm(arm=arm)],
                members=members,
            )
            corrected[arm].append(result)
            frame = fc.frame_of(
                cache=cache,
                setup=setup,
                members=members,
                corrected=result,
                quantiles=np.zeros((result.means.shape[1], 11)),
                arm=arm,
            )
            if frame is not None:
                zero_frames[arm].append(frame)
    out: dict = {"site": site, "factor_scores": {}}
    for arm in own_arms:
        spread = fc.fit_spread(frame=pl.concat(zero_frames[arm]))
        save_spread(arm=arm, site=site, spread=spread)
        scores = {}
        for factor in common.SPREAD_FACTOR_GRID:
            tuned = spread.with_factors(factors=[factor] * common.N_LEAD_BANDS)
            frames = []
            for members, result in zip(pieces, corrected[arm], strict=True):
                quantiles = fc.quantiles_of(
                    cache=cache, members=members, corrected=result.means, spread=tuned
                )
                frame = fc.frame_of(
                    cache=cache,
                    setup=setup,
                    members=members,
                    corrected=result,
                    quantiles=quantiles,
                    arm=arm,
                )
                if frame is not None:
                    frames.append(frame)
            scores[str(factor)] = fc.band_scores(frame=pl.concat(frames)).tolist()
        out["factor_scores"][arm] = scores
    return out


def choose_factors(*, results: list[dict]) -> dict[str, list[float]]:
    """Return, per arm and lead band, the factor with the lowest decision-site mean score."""
    chosen: dict[str, list[float]] = {}
    for arm in fc.OWN_SPREAD_ARMS:
        if arm not in results[0]["factor_scores"]:
            continue
        totals = np.zeros((len(common.SPREAD_FACTOR_GRID), common.N_LEAD_BANDS, 2))
        for result in results:
            if result["site"] not in common.DECISION_SITES:
                continue
            for position, factor in enumerate(common.SPREAD_FACTOR_GRID):
                totals[position] += np.array(result["factor_scores"][arm][str(factor)])
        mean = totals[:, :, 0] / totals[:, :, 1]
        chosen[arm] = np.array(common.SPREAD_FACTOR_GRID)[mean.argmin(axis=0)].tolist()
    return chosen


def read_tuned() -> dict:
    """Return the tuned spread factors."""
    return json.loads(TUNED_PATH.read_text())


def spread_for(*, arm: str, site: str, tuned: dict) -> LeadHourSpread:
    """Return the spread an arm uses at a site: experiment 111's GK spread or this experiment's."""
    if arm in fc.SPREAD_ARM_OF:
        return common.load_gamk_spread(site=site)
    return load_spread(arm=arm, site=site, factors=tuned["spread_factors"][arm])


def forecast_site(job: tuple[str, list[str]]) -> str:
    """Forecast the periods with every arm at one site and write the parquet files.

    Args:
        job: The site's label and the periods to forecast.

    Returns:
        A progress line.
    """
    site, periods = job
    torch.set_num_threads(1)
    started = time.monotonic()
    cache, setup, fits, data, filters = site_inputs(site=site)
    tuned = read_tuned()
    todo = fc.available_arms(filters=filters, arms_wanted=fc.FORECAST_ARMS)
    spreads = {arm: spread_for(arm=arm, site=site, tuned=tuned) for arm in todo}
    rows = 0
    for period in periods:
        frames: dict[str, list[pl.DataFrame]] = {arm: [] for arm in todo}
        for index in setup.indices(periods={period}):
            first_usable = choose_fit(fits=fits, origin=setup.origins[index]) if fits else None
            if first_usable is None:
                continue
            members = fc.run_members(cache=cache, setup=setup, fits=fits, index=index)
            if members is None:
                continue
            for arm in todo:
                result = fc.corrected_members(
                    cache=cache,
                    setup=setup,
                    data=data,
                    filtered=filters[state_arm(arm=arm)],
                    members=members,
                )
                quantiles = fc.quantiles_of(
                    cache=cache, members=members, corrected=result.means, spread=spreads[arm]
                )
                frame = fc.frame_of(
                    cache=cache,
                    setup=setup,
                    members=members,
                    corrected=result,
                    quantiles=quantiles,
                    arm=arm,
                )
                if frame is not None:
                    frames[arm].append(frame)
        for arm, pieces in frames.items():
            target = PREDS_DIR / f"{arm}_{period}_{site}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            frame = pl.concat(pieces)
            frame.write_parquet(temporary)
            temporary.rename(target)
            rows += frame.height
    return f"{site}: {rows:,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Run the chosen step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("tune", "choose", "forecast"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--periods", nargs="+", choices=PERIODS, default=["tuning"])
    args = parser.parse_args()
    started = time.monotonic()
    if args.step == "tune":
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(tune_site, common.SCORED_SITES))
        common.write_json_atomic(
            path=common.WORK_DIR / "factor_scores.json", content={"results": results}
        )
    elif args.step == "choose":
        results = json.loads((common.WORK_DIR / "factor_scores.json").read_text())["results"]
        factors = choose_factors(results=results)
        common.write_json_atomic(path=TUNED_PATH, content={"spread_factors": factors})
        print(f"factors: {factors}")
    else:
        require_clearance(periods=args.periods)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for line in pool.map(
                forecast_site, [(site, args.periods) for site in common.SCORED_SITES]
            ):
                print(line, flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
