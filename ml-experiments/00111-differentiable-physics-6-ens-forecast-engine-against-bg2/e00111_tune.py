"""Tune experiment 111's weather calibration, solar shrink, filter, and spread on the tuning months.

Everything here reads the tuning months only (the odd months of 2024-07 to 2025-06). No replicate
month and no scored run is forecast. The steps, in order:

1. **Calibration** of each substation's ensemble mean against ERA5 and CAMS (fitted, not chosen).
2. **Kalman hyperparameters** of `gamk` by marginal likelihood on the proxy-driven residuals of the
   tuning months, with Part 4's fitted prior as the prior and Part 4's fit as the start.
3. **Solar shrink** per lead band on {0, 0.25, 0.42, 0.55, 0.69, 1.0}, chosen by the quantile
   score of `gamk` at the decision substations (D9 is left out of the tuning score), with the
   spread fitted at a shrink of 1.
4. **Spread** per arm: the centred residual quantiles of the arm's corrected ensemble mean in the
   tuning months, fitted at the chosen shrink, and a **spread factor** per lead band on
   {0.9, 1.0, 1.15, 1.3}. Every arm gets the same grid on the same months.

The spread of an arm is fitted on the tuning months and then scores them, so the tuning scores
are optimistic; the replicate months and the scored year are out of sample.

    uv run python ml-experiments/00111-*/e00111_tune.py [--workers 4]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Final

import e00111_arms as arms
import e00111_common as common
import e00111_ens as ens
import numpy as np
import polars as pl
import torch
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours

SHRINK_GRID: Final[tuple[float, ...]] = (0.0, 0.25, 0.42, 0.55, 0.69, 1.0)
FACTOR_GRID: Final[tuple[float, ...]] = (0.9, 1.0, 1.15, 1.3)
TUNED_ARMS: Final[tuple[str, ...]] = ("gamk", "gam0", "game", "gamk_era5")
TUNING_SITES: Final[tuple[str, ...]] = (*common.SCORED_SITES,)
"""D18 has no readings in the tuning months."""


def band_scores(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the sum and count of quantile scores per calibration lead band, shape (4, 2).

    Rows inside the hand-filtered switching events do not count.
    """
    kept = frame.filter(~pl.col("switching_mask"))
    quantiles = kept.select(
        [f"q{c}" for c in ("01", "05", "10", "25", "40", "50", "60", "75", "90", "95", "99")]
    ).to_numpy()
    scores = ens.quantile_score(
        actual=kept["actual"].to_numpy().astype(np.float64), quantiles=quantiles
    )
    band = lead_band_of_hours(lead_hours=kept["lead_hours"].to_numpy())
    out = np.zeros((N_LEAD_BANDS, 2))
    for index in range(N_LEAD_BANDS):
        out[index] = (scores[band == index].sum(), (band == index).sum())
    return out


def tune_calibration_and_shrink(*, site: str) -> dict:
    """Fit a site's calibration and filter hyperparameters, then score the shrink grid.

    Args:
        site: The substation's label.

    Returns:
        The site's hyperparameters (or `None`) and, per shrink value, the per-band score sums.
    """
    torch.set_num_threads(1)
    cache = common.load_cache()
    setup = ens.load_setup(cache=cache, site=site)
    calibration = ens.fit_calibration(cache=cache, setup=setup)
    arms.save_calibration(site=site, calibration=calibration)
    setup = setup.with_calibration(calibration=calibration)
    result: dict = {"site": site, "hyper": None, "shrink_scores": {}}
    if site not in TUNING_SITES:
        return result
    arm = arms.ARMS["gamk"]
    fits = arms.load_arm_fits(arm=arm, site=site)
    hyper = arms.fit_filter_hyper(cache=cache, setup=setup, fits=fits)
    result["hyper"] = hyper.tolist()
    trace = arms.build_trace(cache=cache, setup=setup, arm=arm, fits=fits, hyper=hyper)
    runs = setup.indices(periods={"tuning"})
    base = ens.forecast_frame(
        cache=cache,
        setup=setup,
        fits=fits,
        spread=ens.ZERO_SPREAD,
        trace=trace,
        run_indices=runs,
        arm="gamk",
    )
    spread = ens.fit_spread_from_frame(frame=base)
    for shrink in SHRINK_GRID:
        shrunk = setup.with_calibration(
            calibration=replace(calibration, solar_shrink=np.full(N_LEAD_BANDS, shrink))
        )
        frame = ens.forecast_frame(
            cache=cache,
            setup=shrunk,
            fits=fits,
            spread=spread,
            trace=trace,
            run_indices=runs,
            arm="gamk",
        )
        result["shrink_scores"][str(shrink)] = band_scores(frame=frame).tolist()
    return result


def tune_spread(
    *,
    site: str,
    shrink: list[float],
    hypers: dict[str, list[float] | None],
    only: tuple[str, ...] = TUNED_ARMS,
) -> dict:
    """Fit each arm's spread at the chosen shrink and score the factor grid.

    Args:
        site: The substation's label.
        shrink: The chosen solar shrink per lead band.
        hypers: The Kalman hyperparameters of each site, from the first step.
        only: The arms to tune.

    Returns:
        For each arm, the per-band score sums for each factor.
    """
    torch.set_num_threads(1)
    cache = common.load_cache()
    setup = ens.load_setup(cache=cache, site=site)
    calibration = replace(arms.load_calibration(site=site), solar_shrink=np.array(shrink))
    setup = setup.with_calibration(calibration=calibration)
    runs = setup.indices(periods={"tuning"})
    out: dict = {"site": site, "factor_scores": {}}
    for name in only:
        arm = arms.ARMS[name]
        fits = arms.load_arm_fits(arm=arm, site=site)
        hyper = hypers[site] if arm.correction == "kalman" and arm.residuals == "proxy" else None
        if arm.name == "gamk_era5":
            hyper = arms.part_4_hyper(site=site)
        trace = arms.build_trace(
            cache=cache,
            setup=setup,
            arm=arm,
            fits=fits,
            hyper=None if hyper is None else np.array(hyper),
        )
        base = ens.forecast_frame(
            cache=cache,
            setup=setup,
            fits=fits,
            spread=ens.ZERO_SPREAD,
            trace=trace,
            run_indices=runs,
            arm=name,
        )
        spread = ens.fit_spread_from_frame(frame=base)
        arms.save_spread(arm=name, site=site, spread=spread)
        scores = {}
        for factor in FACTOR_GRID:
            frame = ens.forecast_frame(
                cache=cache,
                setup=setup,
                fits=fits,
                spread=spread.with_factors(factors=[factor] * N_LEAD_BANDS),
                trace=trace,
                run_indices=runs,
                arm=name,
            )
            scores[str(factor)] = band_scores(frame=frame).tolist()
        out["factor_scores"][name] = scores
    return out


def _step_one(site: str) -> dict:
    return tune_calibration_and_shrink(site=site)


def _step_two(job: tuple[str, list[float], dict, tuple[str, ...]]) -> dict:
    return tune_spread(site=job[0], shrink=job[1], hypers=job[2], only=job[3])


def choose(
    *, results: list[dict], key: str, grid: tuple[float, ...], sites: tuple[str, ...]
) -> np.ndarray:
    """Return, per lead band, the grid value with the lowest mean quantile score at `sites`."""
    totals = np.zeros((len(grid), N_LEAD_BANDS, 2))
    for result in results:
        if result["site"] not in sites:
            continue
        for position, value in enumerate(grid):
            totals[position] += np.array(result[key][str(value)])
    mean = totals[:, :, 0] / totals[:, :, 1]
    return np.array(grid)[mean.argmin(axis=0)]


def main() -> None:
    """Run both tuning steps over the sites in parallel and write `work/tuned.json`.

    With `--spread-arms`, only the spread step reruns for the named arms, with the shrink and
    hyperparameters of the stored `work/tuned.json`, and the stored file is updated.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--spread-arms", nargs="+", choices=TUNED_ARMS, default=None)
    args = parser.parse_args()
    started = time.monotonic()
    decision = common.DECISION_SITES
    stored = arms.read_tuned() if args.spread_arms else {}
    only = tuple(args.spread_arms) if args.spread_arms else TUNED_ARMS
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.spread_arms:
            first = []
            hypers = stored["filter_hyper"]["gamk"]
            shrink = np.array(stored["solar_shrink"])
        else:
            first = list(pool.map(_step_one, common.ALL_SITES))
            hypers = {result["site"]: result["hyper"] for result in first}
            shrink = choose(results=first, key="shrink_scores", grid=SHRINK_GRID, sites=decision)
            elapsed = time.monotonic() - started
            print(f"solar shrink per lead band: {shrink.tolist()} ({elapsed:.0f} s)")
        second = list(
            pool.map(_step_two, [(site, shrink.tolist(), hypers, only) for site in TUNING_SITES])
        )
    factors = dict(stored.get("spread_factors", {}))
    scores = dict(stored.get("factor_scores", {}))
    for name in only:
        per_site = [
            {"site": result["site"], "scores": result["factor_scores"][name]} for result in second
        ]
        factors[name] = choose(
            results=per_site, key="scores", grid=FACTOR_GRID, sites=decision
        ).tolist()
        scores[name] = {item["site"]: item["scores"] for item in per_site}
    common.write_json_atomic(
        path=arms.tuned_path(),
        content={
            "solar_shrink": shrink.tolist(),
            "filter_hyper": {"gamk": hypers},
            "spread_factors": factors,
            "shrink_scores": stored.get("shrink_scores")
            or {result["site"]: result["shrink_scores"] for result in first},
            "factor_scores": scores,
        },
    )
    print(f"factors: {factors}")
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
