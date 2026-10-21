"""Forecast the replicate months and the scored year with experiment 111's DP arms.

Runs the tuned arms (`e00111_arms.ARMS`) at every substation over the replicate months and the 292
scored runs, and writes one parquet file per arm, period, and substation under
`work/dp_preds/`. It refuses to run unless the known-answer gate passed (`work/gate.json`).

The tuning months are forecast too (arm `*_tuning`), for the tuning-score reference rows; they are
in-sample for the spread and must not be quoted as results.

    uv run python ml-experiments/00111-*/e00111_dp_run.py [--workers 4] [--arms gamk gam0]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00111_arms as arms
import e00111_common as common
import e00111_ens as ens
import numpy as np
import torch
from studies.dp_assimilation import DEFAULT_DELAY_HOURS

PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
OUT_DIR: Final[Path] = common.WORK_DIR / "dp_preds"


def out_path(*, arm: str, period: str, site: str) -> Path:
    """Return the file that holds an arm's forecasts for one period at one site."""
    return OUT_DIR / f"{arm}_{period}_{site}.parquet"


def require_gate() -> None:
    """Raise unless the last gate run passed every blocking reading.

    Raises:
        RuntimeError: If `work/gate.json` is missing or records a failure.
    """
    path = common.WORK_DIR / "gate.json"
    if not path.exists() or json.loads(path.read_text())["failed"] != 0:
        msg = "the known-answer gate has not passed; no replicate or scored run may start"
        raise RuntimeError(msg)


def forecast_site(job: tuple[str, str]) -> str:
    """Forecast every period of one arm at one substation and write the files.

    Args:
        job: The arm's name and the substation's label.

    Returns:
        A progress line.
    """
    name, site = job
    torch.set_num_threads(1)
    started = time.monotonic()
    cache = common.load_cache()
    arm = arms.ARMS[name]
    tuned = arms.read_tuned()
    calibration = arms.load_calibration(site=site)
    setup = ens.load_setup(cache=cache, site=site).with_calibration(
        calibration=calibration.with_solar_shrink(solar_shrink=tuned["solar_shrink"])
    )
    fits = arms.load_arm_fits(arm=arm, site=site)
    if arm.name == "gamk_era5":
        hyper = arms.part_4_hyper(site=site)
        if hyper is None:
            hyper = arms.part_4_prior()[0]
    else:
        hyper = tuned["filter_hyper"]["gamk"].get(site)
        if hyper is None:
            hyper = arms.part_4_prior()[0]
    trace = arms.build_trace(
        cache=cache,
        setup=setup,
        arm=arm,
        fits=fits,
        hyper=None if hyper is None else np.array(hyper),
    )
    spread = arms.spread_with_factors(arm=name, site=site, tuned=tuned)
    rows = 0
    for period in PERIODS:
        first_usable_origin = fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
        runs = [
            index
            for index in setup.indices(periods={period})
            if setup.origins[index] >= first_usable_origin
        ]
        if not runs:
            continue
        frame = ens.forecast_frame(
            cache=cache,
            setup=setup,
            fits=fits,
            spread=spread,
            trace=trace,
            run_indices=runs,
            arm=name,
        )
        target = out_path(arm=name, period=period, site=site)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        frame.write_parquet(temporary)
        temporary.rename(target)
        rows += frame.height
    return f"{name} {site}: {rows:,} rows, {time.monotonic() - started:.0f} s"


def spread_for_missing_site(*, name: str, site: str, donors: tuple[str, ...]) -> None:
    """Store a spread for a site with no tuning residuals as the mean of its donors' tables.

    Args:
        name: The arm's name.
        site: The site without a spread (D18, which has no readings in the tuning months).
        donors: The sites whose spreads are averaged.
    """
    from studies.dp_ens import LeadHourSpread
    from studies.dp_gam import ResidualSpread

    spreads = [arms.load_spread(arm=name, site=donor) for donor in donors]
    tables = np.mean([[band.table for band in spread.by_lead_band] for spread in spreads], axis=0)
    arms.save_spread(
        arm=name,
        site=site,
        spread=LeadHourSpread(
            by_lead_band=tuple(ResidualSpread(table=table) for table in tables),
            factors=np.ones(len(tables)),
        ),
    )


def main() -> None:
    """Forecast every arm and site in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arms", nargs="+", default=list(arms.ARMS))
    args = parser.parse_args()
    require_gate()
    for name in args.arms:
        spread_for_missing_site(name=name, site="D18", donors=common.DECISION_SITES)
    jobs = [(name, site) for name in args.arms for site in common.ALL_SITES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(forecast_site, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
