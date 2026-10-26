# ruff: noqa: E501
"""Fix 3: fit D8's stack monthly at the stored cuts with a longer history, and forecast.

Fit arms (all monthly, at the cuts of experiment 252's stored D8 fits, T-all with a fitted solar
capacity):

| Arm | History | Level step |
|---|---|---|
| `a` | from 2025-04-01 (24-month window, as stored) | none |
| `b` | from 2023-09-01 (every row since) | one per hour band at 2025-04-01 |
| `c` | from 2023-09-01 | none |

Each arm's fits drive experiment 252's forecast engine (the same filter, ensemble weather, and
donated spread as the stored forecasts) and then its stage-2 steps. Frames are written under
`work/fits_<arm>/`: `base_scored_D8.parquet` (no corrections, arm A of stage 2) and
`d_scored_D8.parquet` (night offset and recipe R, arm D).

    PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries:ml-experiments/00354-d8-own-residual-bands-and-longer-fit-history \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00354-*/e00354_fit.py --arms a b c
"""

import argparse
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

import pickle
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00252_common as common
import e00252_run as run
import e00252_tall as tall
import e00252s2_corrections as corr
import e00252s2_run as stage2
import e00354_step as step
import polars as pl
import torch
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SITE: Final[str] = "D8"
WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00354-d8-own-residual-bands-and-longer-fit-history" / "work"
)
LONG_START: Final[datetime] = datetime(2023, 9, 1, tzinfo=UTC)
STEP_AT: Final[datetime] = datetime(2025, 4, 1, tzinfo=UTC)
LONG_WINDOW_MONTHS: Final[int] = 48
ARMS: Final[dict[str, dict]] = {
    "a": {"start": common.D8_FIRST_DAY, "window": common.WINDOW_MONTHS, "step": None},
    "b": {"start": LONG_START, "window": LONG_WINDOW_MONTHS, "step": STEP_AT},
    "c": {"start": LONG_START, "window": LONG_WINDOW_MONTHS, "step": None},
}
STORED_CUT_FILE: Final[str] = "monthly"


def arm_dir(*, arm: str) -> Path:
    """Return the folder of a fit arm's fits and forecasts."""
    return WORK / f"fits_{arm}"


def fit_arm_cuts(*, arm: str) -> list:
    """Fit T-all at every stored monthly cut of D8 for one arm and return the fits."""
    settings = ARMS[arm]
    common.D8_FIRST_DAY = settings["start"]
    common.WINDOW_MONTHS = settings["window"]
    cache = common.variant_cache(cache=common.load_cache(), site=SITE)
    vintages = common.vintages_for(cache=cache)
    cuts = [f.cut for f in pickle.loads(tall.fit_path(cadence="monthly", site=SITE).read_bytes())]
    fits = []
    for cut in cuts:
        base = tall.design.build_base(
            cache=cache, vintages=vintages, site=SITE, cut=cut, fitted=common.FITTED
        )
        spec = step.step_arm(
            base=tall.design.interaction_arm(name="t_all", penalty_weight=common.T_ALL_LAMBDA),
            step_from=settings["step"],
        )
        fits.append(
            tall.TallFit(
                site=SITE,
                cut=cut,
                layout=base.layout,
                pinned=base.pinned,
                arm=spec,
                fit=tall.design.fit_arm(cache=cache, base=base, arm=spec),
            )
        )
    return fits


def forecast_scored(*, arm: str) -> pl.DataFrame:
    """Forecast the scored runs of D8 from a fit arm's stored fits, with the donated spread."""
    context = run.load_site(site=SITE)
    forecaster = run.ArmForecaster(arm="a2", context=context)
    first = forecaster.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    runs = [
        i for i in context.setup.indices(periods={"scored"}) if context.setup.origins[i] >= first
    ]
    return forecaster.forecast(runs=runs)


def main() -> None:
    """Fit and forecast every requested arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    step.install()
    for arm in args.arms:
        started = time.monotonic()
        folder = arm_dir(arm=arm)
        folder.mkdir(parents=True, exist_ok=True)
        fits_path = folder / "tall_fits.pkl"
        if arm == "a":
            fits = pickle.loads(tall.fit_path(cadence="monthly", site=SITE).read_bytes())
        else:
            fits = fit_arm_cuts(arm=arm)
        fits_path.write_bytes(pickle.dumps(fits))
        print(arm, "fits", len(fits), f"{time.monotonic() - started:.0f} s", flush=True)
        tall.FITS_DIR = folder / "tall_fits_root"
        target = tall.fit_path(cadence="monthly", site=SITE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pickle.dumps(fits))
        common.D8_FIRST_DAY = ARMS[arm]["start"]
        frame = forecast_scored(arm=arm)
        base = frame.with_columns(period=pl.lit("scored"))
        base.write_parquet(folder / "base_raw_scored_D8.parquet")
        arms = stage2.build_arms(site=SITE, frame=corr.add_calendar(frame=base))
        for name in ("a", "d"):
            arms[name].write_parquet(folder / f"{name}_scored_D8.parquet")
        print(arm, "forecast", frame.height, f"{time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
