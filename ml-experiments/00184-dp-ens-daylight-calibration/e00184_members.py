# ruff: noqa: E501
"""Pass every ensemble member through the T-all fit's solar term and through all of its inputs.

For each run in experiment 136's stored arm A2 forecasts, this script computes, at every scored
half-hour,

- `delta_m` for each member: the T-all fit's forecast with the member's unit PV shape (and every
  other input at the ensemble mean) minus the forecast with the ensemble mean's unit PV shape;
- `delta_f` for each member: the forecast with all of the member's weather minus the forecast with
  the ensemble mean's weather;
- the spread statistics of the members' clear-sky index, calibrated and raw.

Each member's forecast under arms M and F is then arm A's mean plus its delta. The reproduction gate
G1 compares the ensemble-mean forecast plus the filter's level and anomaly with experiment 136's
stored mean.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00184-*/e00184_members.py [--workers 4]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00184_common as common
import e00184_ens as ens
import e00184_tall as tall
import numpy as np
import polars as pl
import torch
from studies.dp_ens import choose_fit
from studies.dp_ens_weather import RunWindow, replace_site
from studies.dp_features import FeatureCache

N_MEMBERS: Final[int] = 51
OUT_DIR = common.WORK_DIR / "members"
MIN_CLEAR_SKY_W_M2: Final[float] = 20.0
NEAR_CLEAR_INDEX: Final[float] = 0.8
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")


def out_path(*, site: str, period: str):  # noqa: ANN201
    """Return the file that holds a site's member deltas for one period."""
    return OUT_DIR / f"{site}_{period}.npz"


def load_context(*, site: str, cache: FeatureCache) -> ens.SiteSetup:
    """Return a site's ensemble setup with experiment 111's calibration and solar shrink."""
    tuned = json.loads((common.SOURCE_WORK_DIR / "tuned.json").read_text())
    with np.load(common.SOURCE_WORK_DIR / "calibration" / f"{site}.npz") as stored:
        from studies.dp_ens_weather import Calibration

        calibration = Calibration(
            temperature_shift_c=stored["shift"],
            irradiance_ratio=stored["ratio"],
            solar_shrink=stored["shrink"],
        ).with_solar_shrink(solar_shrink=tuned["solar_shrink"])
    return ens.load_setup(cache=cache, site=site).with_calibration(calibration=calibration)


def member_statistics(*, flux: np.ndarray, clear: np.ndarray) -> dict[str, np.ndarray]:
    """Return the spread statistics of the members' clear-sky index at each half-hour.

    Args:
        flux: Member irradiance, shape (members, targets), W/m2.
        clear: Haurwitz clear-sky irradiance, shape (targets,).

    Returns:
        `std`, `range`, and `bimodal` (`4 p (1 - p)`, with `p` the share of members at a clear-sky
        index of `NEAR_CLEAR_INDEX` or more), and the ensemble-mean index `mean`, each `NaN` where
        the clear-sky irradiance is under `MIN_CLEAR_SKY_W_M2`.
    """
    valid = clear > MIN_CLEAR_SKY_W_M2
    index = np.where(valid[None, :], flux / np.where(valid, clear, 1.0)[None, :], np.nan)
    share = np.nanmean(index >= NEAR_CLEAR_INDEX, axis=0)
    nan = np.full(len(clear), np.nan)
    with np.errstate(invalid="ignore"):
        return {
            "std": np.where(valid, np.nanstd(index, axis=0), nan),
            "range": np.where(valid, np.nanmax(index, axis=0) - np.nanmin(index, axis=0), nan),
            "bimodal": np.where(valid, 4.0 * share * (1.0 - share), nan),
            "mean": np.where(valid, np.nanmean(index, axis=0), nan),
        }


def compute_run(
    *, cache: FeatureCache, setup: ens.SiteSetup, fits: list[tall.TallFit], index: int
) -> dict[str, np.ndarray]:
    """Return the deltas and statistics of one run at every half-hour of its window."""
    run = setup.runs[index]
    origin = setup.origins[index]
    fit: tall.TallFit = choose_fit(fits=fits, origin=origin)  # ty: ignore[invalid-argument-type, invalid-assignment]
    window = RunWindow.build(cache=cache, context=setup.context, run=run)
    members = run.usable_members()
    n = window.window.stop - window.window.start
    grid = np.arange(n)
    mean_driven = window.driven_caches(
        context=setup.context, run=ens.mean_run(run=run), members=[0]
    )[0]
    base = tall.predict_tall(cache=mean_driven, fit=fit, index=grid)
    driven = window.driven_caches(context=setup.context, run=run, members=members)
    site = setup.site
    delta_m = np.full((N_MEMBERS, n), np.nan)
    delta_f = np.full((N_MEMBERS, n), np.nan)
    flux = np.empty((len(members), n))
    for position, (member, cache_m) in enumerate(zip(members, driven, strict=True)):
        features_m = cache_m.sites[site]
        flux[position] = features_m.ghi_w_m2
        hybrid = replace_site(
            cache=mean_driven,
            site=site,
            pv_unit_shape=features_m.pv_unit_shape,
            ghi_w_m2=features_m.ghi_w_m2,
        )
        delta_m[member] = tall.predict_tall(cache=hybrid, fit=fit, index=grid) - base
        delta_f[member] = tall.predict_tall(cache=cache_m, fit=fit, index=grid) - base
    clear = window.sun.clear_sky_w_m2
    calibrated = member_statistics(flux=flux, clear=clear)
    raw = member_statistics(flux=run.shortwave_w_m2[members, :n].astype(np.float64), clear=clear)
    return {
        "window_start": np.array([window.window.start]),
        "delta_m": delta_m.T,
        "delta_f": delta_f.T,
        "z_std": calibrated["std"],
        "z_range": calibrated["range"],
        "z_bimodal": calibrated["bimodal"],
        "csi_cal": calibrated["mean"],
        "raw_std": raw["std"],
        "csi_raw": raw["mean"],
        "clear": clear,
        "base": base,
        "n_members": np.full(n, len(members), dtype=np.int16),
    }


def one_run(
    *,
    cache: FeatureCache,
    setup: ens.SiteSetup,
    fits: list[tall.TallFit],
    index: int,
    rows: pl.DataFrame,
) -> dict[str, np.ndarray]:
    """Return the deltas and statistics of one run at the rows of experiment 136's forecast."""
    full = compute_run(cache=cache, setup=setup, fits=fits, index=index)
    start = int(full.pop("window_start")[0])
    n = len(full["base"])
    position = (
        np.searchsorted(cache.end_times, rows["valid_time"].dt.replace_time_zone(None).to_numpy())
        - start
    )
    if position.min() < 0 or position.max() >= n:
        msg = "a stored valid time lies outside the run's window"
        raise ValueError(msg)
    out = {name: values[position] for name, values in full.items()}
    out["delta_m"] = out["delta_m"].astype(np.float32)
    out["delta_f"] = out["delta_f"].astype(np.float32)
    out["g1"] = (
        full["base"][position]
        + rows["level"].to_numpy()
        + rows["anomaly"].to_numpy()
        - rows["mean"].to_numpy().astype(np.float64)
    )
    return out


def _job(job: tuple[str, str]) -> str:
    torch.set_num_threads(1)
    site, period = job
    started = time.monotonic()
    cache = common.load_cache()
    setup = load_context(site=site, cache=cache)
    fits = tall.load_fits(cadence="monthly", site=site)
    frame = pl.read_parquet(common.E136_DIR / "dp_preds" / f"a2_{period}_{site}.parquet")
    pieces: list[dict[str, np.ndarray]] = []
    keys = []
    for run_index, rows in frame.group_by("run", maintain_order=True):
        index = int(run_index[0])
        pieces.append(one_run(cache=cache, setup=setup, fits=fits, index=index, rows=rows))
        keys.append(
            {
                "run": rows["run"].to_numpy(),
                "valid_time": rows["valid_time"].dt.replace_time_zone(None).to_numpy(),
            }
        )
    out = {name: np.concatenate([p[name] for p in pieces]) for name in pieces[0]}
    out["run"] = np.concatenate([k["run"] for k in keys])
    out["valid_time"] = np.concatenate([k["valid_time"] for k in keys])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = out_path(site=site, period=period)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **out)  # ty: ignore[invalid-argument-type]
    temporary.rename(target)
    worst = float(np.nanmax(np.abs(out["g1"])))
    return f"{site} {period}: {len(out['run']):,} rows, G1 max |diff| {worst:.2e}, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Compute the member deltas of every site and period."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    parser.add_argument("--periods", nargs="+", default=list(PERIODS))
    args = parser.parse_args()
    jobs = [(s, p) for p in args.periods for s in args.sites]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
