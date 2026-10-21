"""Choose each substation's filter cell and the shared cell, on the tuning months only.

**Step 1 (GPU)** runs `studies.dp_filter_grid.run_kalman_grid` over every cell of the grid (and
each substation's FIX setting as one more cell) on the substation's proxy-driven residual stream,
and scores each cell by the mean absolute error of the corrected residual at day-ahead cases:
the tuning-month runs, targets 12 to 36 hours after the origin, state at the newest reading that
ended 3.5 hours before the origin, rows outside the hand-filtered switching events.

**Step 2 (CPU)** re-scores the FIX cell and the 20 cells with the lowest step-1 error per
substation by the quantile score at 12 to 36 hours of the full ensemble forecast on the tuning
months (the spread refitted for each cell at a factor of 1). The lowest wins and is the GRID cell.

**POOL** is the one grid cell with the lowest sum over the scored substations of its step-1 error
relative to the same substation's FIX-cell error, which pools the substations without a second
ensemble pass.

    uv run python ml-experiments/00133-*/e00133_select.py grid
    uv run python ml-experiments/00133-*/e00133_select.py rescore [--workers 5]
    uv run python ml-experiments/00133-*/e00133_select.py pool
"""

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from typing import Final

import e00133_arms as arms
import e00133_common as common
import e00133_ens as ens
import e00133_filters as filters
import numpy as np
import polars as pl
from studies.dp_assimilation import DEFAULT_DELAY_HOURS, usable_cutoff_positions
from studies.dp_features import naive_utc
from studies.dp_filter_grid import correction_at_cutoffs, run_kalman_grid

SELECT_DIR: Final = common.WORK_DIR / "select"
LOG_Q: Final[tuple[float, ...]] = tuple(np.arange(-4.0, 6.01, 0.5))
LOGIT_PHI: Final[tuple[float, ...]] = (1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 4.0)
LOG_SIGMA_Z: Final[tuple[float, ...]] = tuple(np.arange(-0.5, 3.51, 0.5))
TOP_CELLS: Final[int] = 20
LEAD_LOW_HOURS: Final[float] = 12.0
LEAD_HIGH_HOURS: Final[float] = 36.0


def cells() -> np.ndarray:
    """Return the grid cells as rows of (log q, logit phi, log sigma_z, log sigma_eps = 0)."""
    grid = list(itertools.product(LOG_Q, LOGIT_PHI, LOG_SIGMA_Z))
    return np.array([[q, phi, z, 0.0] for q, phi, z in grid])


def day_ahead_cases(
    *,
    context: arms.SiteContext,
    periods: frozenset[str] = frozenset({"tuning"}),
    stream: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return the day-ahead cases of the named periods with a finite residual outside switching.

    Args:
        context: The substation's stored inputs.
        periods: The periods whose runs supply the cases.
        stream: A residual stream to use instead of the substation's real one.

    Returns:
        Arrays `cutoff`, `target`, `band`, and `actual` (the residual before correction).
    """
    cache, setup = context.cache, context.setup
    features = cache.sites[context.site]
    cutoffs, targets = [], []
    stream = context.stream if stream is None else stream
    first = context.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    runs = [i for i in setup.indices(periods=set(periods)) if setup.origins[i] >= first]
    for index in runs:
        origin = np.datetime64(naive_utc(moment=setup.origins[index]), "us")
        lead = (cache.end_times - origin) / np.timedelta64(1, "h")
        rows = np.flatnonzero((lead > LEAD_LOW_HOURS) & (lead <= LEAD_HIGH_HOURS))
        cutoff = usable_cutoff_positions(
            end_times=cache.end_times,
            initialisation=np.array([origin]),
            delay_hours=DEFAULT_DELAY_HOURS,
        )[0]
        cutoffs.append(np.full(len(rows), cutoff))
        targets.append(rows)
    cutoff, target = np.concatenate(cutoffs), np.concatenate(targets)
    keep = np.isfinite(stream[target]) & ~features.switching_mask[target] & (cutoff >= 0)
    cutoff, target = cutoff[keep], target[keep]
    return {
        "cutoff": cutoff,
        "target": target,
        "band": cache.calendar.band[target].astype(np.int64),
        "actual": stream[target],
    }


def step_one(*, site: str) -> dict:
    """Score every cell and the FIX cell on the proxy stream's day-ahead cases.

    Args:
        site: The substation's label.

    Returns:
        The site's JSON-ready record: the mean absolute error of each grid cell, of the FIX cell,
        the number of cases, and the lowest cells.
    """
    context = arms.load_site(site=site)
    grid = np.vstack([cells(), arms.fix_hyper(site=site)[None, :]])
    cases = day_ahead_cases(context=context)
    trace = run_kalman_grid(
        residuals=context.stream[None, :],
        band=context.cache.calendar.band,
        hyper=grid,
        n_bands=4,
        keep_states=True,
    )
    correction = correction_at_cutoffs(
        trace=trace, cutoff=cases["cutoff"], target=cases["target"], band=cases["band"]
    )[0]
    error = np.abs(cases["actual"][None, :] - correction.cpu().numpy()).mean(axis=1)
    order = np.argsort(error[:-1])[:TOP_CELLS]
    return {
        "site": site,
        "n_cases": len(cases["target"]),
        "error": error[:-1].tolist(),
        "fix_error": float(error[-1]),
        "top": order.tolist(),
    }


def run_grid() -> None:
    """Run step 1 for every scored substation on the GPU, one after the other."""
    for site in common.SCORED_SITES:
        started = time.monotonic()
        record = step_one(site=site)
        common.write_json_atomic(path=SELECT_DIR / f"step1_{site}.json", content=record)
        best = record["top"][0]
        print(
            f"step 1 {site}: {record['n_cases']:,} cases, best cell {best} "
            f"error {record['error'][best]:.4f} against FIX {record['fix_error']:.4f} "
            f"({time.monotonic() - started:.0f} s)",
            flush=True,
        )


def tuning_score(
    *,
    context: arms.SiteContext,
    settings: filters.FilterSettings,
) -> float:
    """Return the quantile score at 12 to 36 hours of a setting's ensemble forecast, tuning months.

    Args:
        context: The substation's stored inputs.
        settings: The filter's setting.

    Returns:
        The mean quantile score in points, outside the switching events.
    """
    run = filters.filter_stream(
        residual=context.stream, band=context.cache.calendar.band, settings=settings
    )
    indices = arms.tuning_runs(context=context)
    base = ens.forecast_frame(
        cache=context.cache,
        setup=context.setup,
        fits=context.fits,
        spread=ens.ZERO_SPREAD,
        trace=run.trace,
        run_indices=indices,
        arm="x",
    )
    spread = ens.fit_spread_from_frame(frame=base)
    frame = ens.forecast_frame(
        cache=context.cache,
        setup=context.setup,
        fits=context.fits,
        spread=spread,
        trace=run.trace,
        run_indices=indices,
        arm="x",
    )
    return day_ahead_score(frame=frame)


def day_ahead_score(*, frame: pl.DataFrame) -> float:
    """Return the mean quantile score of the rows at leads 12 to 36 hours outside switching."""
    kept = frame.filter(
        ~pl.col("switching_mask")
        & (pl.col("lead_hours") > LEAD_LOW_HOURS)
        & (pl.col("lead_hours") <= LEAD_HIGH_HOURS)
    )
    quantiles = kept.select(
        [c for c in kept.columns if c.startswith("q") and c[1:].isdigit()]
    ).to_numpy()
    scores = ens.quantile_score(
        actual=kept["actual"].to_numpy().astype(np.float64), quantiles=quantiles
    )
    return float(scores.mean())


def rescore_site(site: str) -> dict:
    """Re-score the FIX cell and the top cells of a substation by the ensemble quantile score.

    Args:
        site: The substation's label.

    Returns:
        The record: the candidate cells, their scores, and the winning cell's index (-1 for FIX).
    """
    import torch

    torch.set_num_threads(1)
    context = arms.load_site(site=site)
    step1 = json.loads((SELECT_DIR / f"step1_{site}.json").read_text())
    grid = cells()
    candidates = [-1, *step1["top"]]
    scores = []
    for index in candidates:
        hyper = arms.fix_hyper(site=site) if index < 0 else grid[index]
        scores.append(tuning_score(context=context, settings=arms.settings_of_hyper(hyper=hyper)))
    winner = candidates[int(np.argmin(scores))]
    hyper = arms.fix_hyper(site=site) if winner < 0 else grid[winner]
    record = {
        "site": site,
        "candidates": candidates,
        "scores": scores,
        "winner": winner,
        "hyper": hyper.tolist(),
        "rank_correlation": rank_correlation(
            first=[step1["fix_error"], *[step1["error"][i] for i in step1["top"]]], second=scores
        ),
    }
    common.write_json_atomic(path=SELECT_DIR / f"step2_{site}.json", content=record)
    return record


def rank_correlation(*, first: list[float], second: list[float]) -> float:
    """Return the Spearman rank correlation of two equally long lists."""
    ranks = [np.argsort(np.argsort(values)) for values in (first, second)]
    return float(np.corrcoef(ranks[0], ranks[1])[0, 1])


def choose_pool() -> dict:
    """Return the shared cell: the lowest sum over sites of step-1 error relative to FIX's."""
    total = None
    for site in common.SCORED_SITES:
        step1 = json.loads((SELECT_DIR / f"step1_{site}.json").read_text())
        relative = np.array(step1["error"]) / step1["fix_error"]
        total = relative if total is None else total + relative
    assert total is not None
    best = int(np.argmin(total))
    record = {
        "cell": best,
        "hyper": cells()[best].tolist(),
        "mean_relative_error": float(total[best] / len(common.SCORED_SITES)),
    }
    common.write_json_atomic(path=SELECT_DIR / "pool.json", content=record)
    return record


def main() -> None:
    """Run one stage: `grid` (GPU), `rescore` (CPU), or `pool`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("grid", "rescore", "pool"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.stage == "grid":
        run_grid()
    elif args.stage == "rescore":
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for record in pool.map(rescore_site, common.SCORED_SITES):
                print(
                    f"step 2 {record['site']}: winner cell {record['winner']}, "
                    f"scores {np.round(record['scores'], 3).tolist()}, "
                    f"rank correlation {record['rank_correlation']:.2f}",
                    flush=True,
                )
    else:
        print(choose_pool())


if __name__ == "__main__":
    main()
