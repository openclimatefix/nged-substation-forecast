"""Re-forecast arm D's base engine (experiment 136's A2) with realised CAMS sunshine.

For each substation the script writes `work/swap/<site>.parquet` with, for every scored run and
valid time, the engine's mean forecast under three weather inputs:

| Column | Sunshine |
|---|---|
| `mean_ens` | the calibrated and shrunk ensemble, as stored (gate: equals experiment 136's `a2`) |
| `mean_ens_id` | the ensemble with the irradiance ratio and the solar shrink set to 1 |
| `mean_cams` | realised CAMS irradiance at the target half-hours, ratio and shrink 1 |

Temperature is the calibrated ensemble mean in every column. Nothing is refitted.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00289-*/e00289_swap.py [--workers 3]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import timedelta
from typing import Final

import e00289_common as common
import e00289_ens as ens
import e00289_run as run
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.dp_ens_weather import Calibration, EnsSiteRun

OUT_DIR: Final = common.WORK_DIR / "swap"
GATE_TOLERANCE_POINTS: Final[float] = 1e-6


def identity_irradiance(*, calibration: Calibration) -> Calibration:
    """Return the calibration with the irradiance ratio and the solar shrink set to 1."""
    return replace(
        calibration,
        irradiance_ratio=np.ones_like(calibration.irradiance_ratio),
        solar_shrink=np.ones_like(calibration.solar_shrink),
    )


def cams_runs(*, setup: ens.SiteSetup, ghi_w_m2: np.ndarray) -> list[EnsSiteRun]:
    """Return the setup's runs with one member whose shortwave is CAMS at the target half-hours.

    A half-hour with no CAMS value takes the ensemble mean, so the run stays usable.
    """
    runs = []
    for item in setup.runs:
        count = item.n_targets
        rows = np.arange(item.origin_position + 1, item.origin_position + 1 + count)
        padded = np.full(count, np.nan)
        inside = rows < len(ghi_w_m2)
        padded[inside] = ghi_w_m2[rows[inside]]
        missing = ~np.isfinite(padded)
        padded[missing] = np.nanmean(item.shortwave_w_m2[:, :count], axis=0)[missing]
        runs.append(
            EnsSiteRun(
                origin_position=item.origin_position,
                temperature_c=item.temperature_c[item.usable_members()].mean(axis=0)[None, :],
                shortwave_w_m2=padded[None, :],
            )
        )
    return runs


def _frame(*, forecaster: run.ArmForecaster, setup: ens.SiteSetup, runs: list[int]) -> pl.DataFrame:
    context = replace(forecaster.context, setup=setup)
    swapped = run.ArmForecaster.__new__(run.ArmForecaster)
    swapped.__dict__.update(forecaster.__dict__)
    swapped.context = context
    return swapped.forecast(runs=runs).select("run", "valid_time", "mean", "actual", "lead_hours")


def swap_site(site: str) -> str:
    """Write one substation's three mean forecasts."""
    torch.set_num_threads(1)
    started = time.monotonic()
    context = run.load_site(site=site)
    forecaster = run.ArmForecaster(arm="a2", context=context)
    setup = context.setup
    first = forecaster.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    runs = [i for i in setup.indices(periods={"scored"}) if setup.origins[i] >= first]
    base = _frame(forecaster=forecaster, setup=setup, runs=runs).rename({"mean": "mean_ens"})
    calibration = identity_irradiance(calibration=setup.context.calibration)
    ens_id_setup = setup.with_calibration(calibration=calibration)
    ens_id = _frame(forecaster=forecaster, setup=ens_id_setup, runs=runs).select(
        "run", "valid_time", pl.col("mean").alias("mean_ens_id")
    )
    ghi = context.cache.sites[site].ghi_w_m2
    cams_setup = replace(setup, runs=cams_runs(setup=setup, ghi_w_m2=ghi)).with_calibration(
        calibration=calibration
    )
    cams = _frame(forecaster=forecaster, setup=cams_setup, runs=runs).select(
        "run", "valid_time", pl.col("mean").alias("mean_cams")
    )
    out = base.join(ens_id, on=["run", "valid_time"], how="left", validate="1:1").join(
        cams, on=["run", "valid_time"], how="left", validate="1:1"
    )
    out = out.with_columns(site=pl.lit(site))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / f"{site}.parquet"
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    out.write_parquet(temporary)
    temporary.rename(target)
    return f"{site}: {out.height:,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Run every substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(swap_site, args.sites):
            print(line, flush=True)


if __name__ == "__main__":
    main()
