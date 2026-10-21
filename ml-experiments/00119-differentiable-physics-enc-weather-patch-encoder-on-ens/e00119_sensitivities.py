"""GAM-K's own sensitivities to temperature and irradiance, for ENC-r (design b).

For every K0 row the script re-forecasts GAM-K from experiment 111's stored monthly fits with the
ensemble's temperature moved by plus and minus 1 K and its irradiance scaled by 1.1 and 0.9, in the
forecast window only. The central differences are `sens_t` (points of effective capacity per
kelvin) and `sens_g` (points per unit of irradiance multiplier). These are the same per-month
out-of-sample fits that produced each stored prediction, so no scored-year information enters.

    uv run python ml-experiments/00119-*/e00119_sensitivities.py [--workers 3]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import e00119_common as common
import e00119_gamk as gamk
import polars as pl
import torch

TEMPERATURE_STEP_K = 1.0
IRRADIANCE_STEP = 0.1
KEYS = ["run", "valid_time"]


def sensitivities_path(*, site: str) -> Path:
    """Return the file that holds a site's sensitivities."""
    return common.WORK_DIR / "sens" / f"{site}.parquet"


def sites_sensitivities(site: str) -> str:
    """Compute and store one site's sensitivities."""
    torch.set_num_threads(1)
    started = time.monotonic()
    indices = list(range(len(common.run_dates())))
    pieces = {
        name: gamk.forecast_runs(
            site=site, run_indices=indices, temperature_shift_k=shift, irradiance_scale=scale
        ).select(*KEYS, pl.col("mean").alias(name))
        for name, shift, scale in (
            ("t_up", TEMPERATURE_STEP_K, 1.0),
            ("t_down", -TEMPERATURE_STEP_K, 1.0),
            ("g_up", 0.0, 1.0 + IRRADIANCE_STEP),
            ("g_down", 0.0, 1.0 - IRRADIANCE_STEP),
        )
    }
    joined = pieces["t_up"]
    for name in ("t_down", "g_up", "g_down"):
        joined = joined.join(pieces[name], on=KEYS, how="inner")
    result = joined.select(
        *KEYS,
        sens_t=(pl.col("t_up") - pl.col("t_down")) / (2.0 * TEMPERATURE_STEP_K),
        sens_g=(pl.col("g_up") - pl.col("g_down")) / (2.0 * IRRADIANCE_STEP),
    )
    target = sensitivities_path(site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(target)
    return f"{site}: {result.height:,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Compute the sensitivities of every scored site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(sites_sensitivities, common.SCORED_SITES):
            print(line, flush=True)


if __name__ == "__main__":
    main()
