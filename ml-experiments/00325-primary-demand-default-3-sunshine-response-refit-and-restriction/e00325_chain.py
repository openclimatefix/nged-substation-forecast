# ruff: noqa: E501
r"""Stage B: rebuild Primary demand research default 1's convex chain for a response variant.

The chain is experiment 317's (T-all with the Kalman filter, winter night offset and recipe R, the
spread table by lead band, fix O), run with the variant's refitted T-all coefficients. The Kalman
filter runs on the variant's own residuals against the analysis proxy, and the offset, recipe R and
spread table are refitted on the variant's own forecasts. `v0r` installs no response and must
reproduce experiment 317's stored chain (the gate). A second output, `arm_d`, is experiment 155's
arm D (offset and recipe R, spread refitted on the tuning months) at leads of 168 hours or less,
with the filter's level and anomaly, which feeds the median-correction network of experiment 201.

    source /tmp/env325.sh && systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00325-*/e00325_chain.py --variant f1 [--workers 6]
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import timedelta

import e00136_run as run136  # ty: ignore[unresolved-import]
import e00155_corrections as corr  # ty: ignore[unresolved-import]
import e00317_chain as c317  # ty: ignore[unresolved-import]
import e00317_common as common317  # ty: ignore[unresolved-import]
import e00325_common as common
import e00325_variant as variant_module
import numpy as np
import polars as pl
import studies.dp_ens_weather as weather
import torch
from studies.dp_assimilation import DEFAULT_DELAY_HOURS

BASE_COLUMNS = (*c317.BASE_COLUMNS, "level", "anomaly")


def forecast_base(*, site: str) -> pl.DataFrame:
    """Return arm A2's frames (mean, level, anomaly, keys) for the three periods out to 240 hours."""
    context = run136.load_site(site=site)
    forecaster = run136.ArmForecaster(arm="a2", context=context)
    first = forecaster.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    pieces = []
    for period in common.PERIODS:
        runs = [
            i for i in context.setup.indices(periods={period}) if context.setup.origins[i] >= first
        ]
        if runs:
            pieces.append(forecaster.forecast(runs=runs).with_columns(period=pl.lit(period)))
    return (
        pl.concat(pieces)
        .select([*BASE_COLUMNS, "period"])
        .sort(["power_fcst_init_time", "valid_time"])
    )


def job(arguments: tuple[str, str]) -> str:
    """Build the chain and arm D of one variant at one Primary."""
    variant, site = arguments
    torch.set_num_threads(1)
    started = time.monotonic()
    variant_module.install(variant=variant)
    weather.HORIZON_HALF_HOURS = common317.HORIZON_HALF_HOURS
    base = forecast_base(site=site)
    gates: dict = {}
    if variant == "v0r":
        gates["a2"] = c317.gate_against(
            built=base,
            stored_path_of=lambda p: common317.E136_PREDS / f"a2_{p}_{site}.parquet",
            columns=("mean",),
        )
    frame = corr.add_calendar(frame=base)
    mean_a = frame["mean"].to_numpy().astype(np.float64)
    offset = c317.offset_shift_fitted_short(frame=frame)
    mean_b = mean_a + offset
    shift_d = offset + corr.recipe_r_shift(frame=frame, forecast=mean_b)
    mean_d = mean_a + shift_d
    frame = frame.with_columns(mean_d=pl.Series(mean_d))
    if variant == "v0r":
        gates["arm_d"] = c317.gate_against(
            built=frame.with_columns(mean=pl.col("mean_d")),
            stored_path_of=lambda p: common317.E155_PREDS / f"d_{p}_{site}.parquet",
            columns=("mean",),
        )
    # Experiment 155's arm D at leads of 168 hours or less, for the median-correction network.
    short = frame.filter(pl.col("lead_hours") <= common317.STORED_HORIZON_HOURS)
    short_mean_a = short["mean"].to_numpy().astype(np.float64)
    short_offset, _ = corr.winter_offset_shift(frame=short, variant="winter")
    short_shift = short_offset + corr.recipe_r_shift(
        frame=short, forecast=short_mean_a + short_offset
    )
    arm_d = corr.apply_shift_and_refit_spread(frame=short, shift=short_shift)
    if variant == "v0r":
        gates["arm_d_quantiles"] = c317.gate_against(
            built=arm_d,
            stored_path_of=lambda p: common317.E155_PREDS / f"d_{p}_{site}.parquet",
            columns=common317.QUANTILES,
        )
    # Experiment 160's arm I with the table of the variant's arm A2 extended to seven lead bands.
    table = c317.fit_table(frame=frame, mean=mean_a)
    band = frame["hour_band"].to_numpy().astype(np.int64)
    quarter = frame["quarter"].to_numpy().astype(np.int64)
    lead = frame["lead_hours"].to_numpy().astype(np.float64)
    offsets = c317.interpolated_offsets(table=table, band=band, quarter=quarter, lead_hours=lead)
    quantiles_i = mean_d[:, None] + offsets
    built_i = frame.with_columns(
        *[
            pl.Series(q, quantiles_i[:, k].astype(np.float32))
            for k, q in enumerate(common317.QUANTILES)
        ],
        pl.Series("mean", mean_d.astype(np.float32)),
    )
    multipliers = json.loads((common317.E164_WORK / "fit_parameters.json").read_text())["o"]
    ens_mean_c = c317.ensemble_temperature(frame=frame, site=site)
    cold = np.nan_to_num(ens_mean_c, nan=99.0) < common317.COLD_GATE_C
    scale = np.ones_like(quantiles_i)
    for k, level in enumerate(common317.LEVELS.tolist()):
        if level in common317.UPPER_FIX_LEVELS:
            scale[:, k] = np.where(cold, multipliers[str(level)], 1.0)
    quantiles_o = np.sort(mean_d[:, None] + offsets * scale, axis=1)
    built_o = built_i.with_columns(
        *[
            pl.Series(q, quantiles_o[:, k].astype(np.float32))
            for k, q in enumerate(common317.QUANTILES)
        ],
        ens_mean_c=pl.Series(ens_mean_c),
    )
    if variant == "v0r":
        gates["arm_o"] = c317.gate_against(
            built=built_o,
            stored_path_of=lambda p: common317.E164_WORK / "dp_preds" / f"o_{p}_{site}.parquet",
            columns=common317.QUANTILES,
        )
    for directory, content in (
        (common.CHAIN_DIR, built_o.drop("mean_d")),
        (common.ARM_D_DIR, arm_d),
    ):
        target = directory / variant / f"{site}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        content.write_parquet(temporary)
        temporary.rename(target)
    common.write_json_atomic(path=common.CHAIN_DIR / variant / f"gates_{site}.json", content=gates)
    return (
        f"{variant} {site}: {built_o.height:,} rows, {time.monotonic() - started:.0f} s, "
        f"gates {json.dumps(gates)[:500]}"
    )


def main() -> None:
    """Build the chain of every variant and Primary in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(common.VARIANTS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sites", nargs="+", default=list(common.SITES))
    args = parser.parse_args()
    jobs = [(v, s) for v in args.variants for s in args.sites]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for line in pool.imap_unordered(job, jobs, chunksize=1):
            print(line, flush=True)


if __name__ == "__main__":
    main()
