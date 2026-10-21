"""Fit each arm's spread and spread factors on the tuning months, for experiment 135.

Every arm uses experiment 111's calibration and solar shrink and the filter settings of
`e00135_arms.read_hyper`. For each arm and substation, the script forecasts the tuning months (the
odd months of 2024-07 to 2025-06) with no spread, fits the centred residual quantiles of the
corrected ensemble mean (`LeadHourSpread`), and scores a factor grid on the same months. The factor
per lead band is the grid value with the lowest mean quantile score at the decision substations, as
in experiment 111. The J2 arm's ridge weight is the weight with the lowest tuning score at the
substation, which `choose_ridge` writes to `work/ridge.json`.

The spread of an arm is fitted on the tuning months and then scores them, so the tuning scores are
optimistic; the replicate months and the scored year are out of sample.

    uv run --no-sync python ml-experiments/00135-*/e00135_tune.py [--workers 4]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00135_arms as arms
import e00135_common as common
import e00135_ens as ens
import numpy as np
import polars as pl
import torch
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours

FACTOR_GRID: Final[tuple[float, ...]] = (0.9, 1.0, 1.15, 1.3)
TUNING_SITES: Final[tuple[str, ...]] = common.SCORED_SITES


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


def tune_site(job: tuple[str, tuple[str, ...]]) -> dict:
    """Fit each arm's spread at one substation and score the factor grid.

    Args:
        job: The substation's label and the arms to tune.

    Returns:
        For each arm, the per-band score sums for each factor.
    """
    site, names = job
    torch.set_num_threads(1)
    cache = common.load_cache()
    tuned = arms.read_tuned()
    setup = ens.load_setup(cache=cache, site=site).with_calibration(
        calibration=arms.load_calibration(site=site).with_solar_shrink(
            solar_shrink=tuned["solar_shrink"]
        )
    )
    runs = setup.indices(periods={"tuning"})
    out: dict = {"site": site, "factor_scores": {}}
    for name in names:
        arm = arms.ARMS[name]
        fits = arms.load_arm_fits(arm=arm, site=site)
        trace = arms.build_trace(
            cache=cache, setup=setup, arm=arm, fits=fits, hyper=arms.read_hyper(site=site)
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


def choose(*, results: list[dict], name: str, sites: tuple[str, ...]) -> tuple[list[float], float]:
    """Return an arm's factor per lead band and its mean tuning score, over `sites`."""
    totals = np.zeros((len(FACTOR_GRID), N_LEAD_BANDS, 2))
    for result in results:
        if result["site"] in sites:
            for position, value in enumerate(FACTOR_GRID):
                totals[position] += np.array(result["factor_scores"][name][str(value)])
    mean = totals[:, :, 0] / totals[:, :, 1]
    chosen = np.array(FACTOR_GRID)[mean.argmin(axis=0)]
    best = totals[mean.argmin(axis=0), np.arange(N_LEAD_BANDS)]
    return chosen.tolist(), float(best[:, 0].sum() / best[:, 1].sum())


def main() -> None:
    """Tune the spread of every arm in parallel and write `work/factors.json` and the scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arms", nargs="+", default=list(arms.ARMS))
    args = parser.parse_args()
    started = time.monotonic()
    names = tuple(args.arms)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(tune_site, [(site, names) for site in TUNING_SITES]))
    factors = {}
    tuning_scores = {}
    for name in names:
        factors[name], tuning_scores[name] = choose(
            results=results, name=name, sites=common.DECISION_SITES
        )
    common.write_json_atomic(path=arms.factors_path(), content=factors)
    common.write_json_atomic(
        path=common.WORK_DIR / "tuning_scores.json",
        content={"mean_tuning_score": tuning_scores, "per_site": results},
    )
    print(f"factors: {factors}")
    print(f"tuning scores: {tuning_scores}")
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
