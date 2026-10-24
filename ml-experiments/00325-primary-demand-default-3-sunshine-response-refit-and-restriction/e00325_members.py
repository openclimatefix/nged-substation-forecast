# ruff: noqa: E501
r"""Stage B: pass the 51 ensemble members through the T-all fit with and without the sunshine response.

Follows experiment 184's `compute_run` with the forecast window set to 480 half-hours (experiment
317's setting). For each run, the ensemble-mean weather gives `base` for each variant, and each
member's weather (its PV shape and irradiance in place of the mean's) gives `delta_m`, the change
in the forecast. Variants:

- `v0`: experiment 184's stored fit, no response (the gate: its `delta_m` equals the stored ones).
- `f1`, `f1s`: the fit with the unrestricted or restricted response of `e00325_fits`.

The level change a response makes is not stored: the chain of stage B is rebuilt on each
variant's own fit. Rows are the rows of experiment 317's convex chain (0.5 to 240 hours), one file
per Primary and period.

    source /tmp/env325.sh && systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00325-*/e00325_members.py [--workers 6]
"""

import argparse
import multiprocessing as mp
import os
import time

import e00184_common as c184  # ty: ignore[unresolved-import]
import e00184_design as design  # ty: ignore[unresolved-import]
import e00184_members as m184  # ty: ignore[unresolved-import]
import e00184_tall as tall  # ty: ignore[unresolved-import]
import e00325_basis as sun
import e00325_common as common
import e00325_fits as fits325
import numpy as np
import polars as pl
import studies.dp_ens_weather as weather
import torch
from studies.dp_ens import choose_fit
from studies.dp_ens_weather import RunWindow, replace_site
from studies.dp_features import FeatureCache
from studies.dp_gam import design_matrix, pinned_solar_points

VARIANTS: tuple[str, ...] = ("v0", *common.RESPONSE_VARIANTS)


def predict_variants(
    *,
    cache: FeatureCache,
    fit: tall.TallFit,
    params: dict[str, tuple[np.ndarray, np.ndarray]],
    index: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return each variant's net-power prediction in points, building the design once."""
    features = cache.sites[fit.site]
    x0 = design_matrix(cache=cache, features=features, layout=fit.layout, index=index)
    x = design.arm_matrix(
        cache=cache, layout=fit.layout, site=fit.site, index=index, x0=x0, arm=fit.arm
    )
    if not np.isfinite(x).all():
        msg = f"{fit.site} has a non-finite input among the rows to predict"
        raise ValueError(msg)
    pinned = pinned_solar_points(cache=cache, features=features, pinned=fit.pinned, index=index)
    out = {}
    for name, (theta, gamma) in params.items():
        response = 0.0
        if name != "v0":
            response = (
                sun.basis(
                    ghi_w_m2=features.ghi_w_m2[index],
                    elevation_deg=features.elevation_deg[index],
                    variant=name,
                )
                @ gamma
            )
        out[name] = x @ theta + response - pinned
    return out


def run_params(*, fit: tall.TallFit, tables: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return `(theta, gamma)` of each variant for the stored fit's cut."""
    out = {"v0": (fit.fit.theta, np.zeros(sun.N_COLUMNS))}
    for name in common.RESPONSE_VARIANTS:
        out[name] = tables[name][(fit.site, fit.cut)]
    return out


def compute_run(
    *,
    cache: FeatureCache,
    setup,  # noqa: ANN001
    fits: list,
    index: int,
    tables: dict,
) -> dict[str, np.ndarray]:
    """Return every variant's `base` and member deltas at every half-hour of one run's window."""
    run = setup.runs[index]
    origin = setup.origins[index]
    fit = choose_fit(fits=fits, origin=origin)
    params = run_params(fit=fit, tables=tables)
    window = RunWindow.build(cache=cache, context=setup.context, run=run)
    members = run.usable_members()
    n = window.window.stop - window.window.start
    grid = np.arange(n)
    mean_driven = window.driven_caches(
        context=setup.context, run=m184.ens.mean_run(run=run), members=[0]
    )[0]
    base = predict_variants(cache=mean_driven, fit=fit, params=params, index=grid)
    driven = window.driven_caches(context=setup.context, run=run, members=members)
    delta = {name: np.full((common.N_MEMBERS, n), np.nan) for name in VARIANTS}
    for member, cache_m in zip(members, driven, strict=True):
        features_m = cache_m.sites[setup.site]
        hybrid = replace_site(
            cache=mean_driven,
            site=setup.site,
            pv_unit_shape=features_m.pv_unit_shape,
            ghi_w_m2=features_m.ghi_w_m2,
        )
        predicted = predict_variants(cache=hybrid, fit=fit, params=params, index=grid)
        for name in VARIANTS:
            delta[name][member] = predicted[name] - base[name]
    out: dict[str, np.ndarray] = {
        "window_start": np.array([window.window.start]),
        "n_members": np.full(n, len(members), dtype=np.int16),
    }
    for name in VARIANTS:
        out[f"delta_{name}"] = delta[name].T
    return out


def job(arguments: tuple[str, str, int]) -> str:
    """Compute one Primary and period's member responses for every chain row."""
    site, period, limit = arguments
    torch.set_num_threads(1)
    weather.HORIZON_HALF_HOURS = common.HORIZON_HALF_HOURS
    started = time.monotonic()
    cache = c184.load_cache()
    setup = m184.load_context(site=site, cache=cache)
    fits = tall.load_fits(cadence="monthly", site=site)
    tables = {name: fits325.load_table(variant=name) for name in common.RESPONSE_VARIANTS}
    chain = pl.read_parquet(
        common.CHAIN_DIR / "v0r" / f"{site}.parquet", columns=["run", "valid_time", "period"]
    ).filter(pl.col("period") == period)
    pieces = []
    for number, (run_index, rows) in enumerate(chain.group_by("run", maintain_order=True)):
        if limit and number >= limit:
            break
        full = compute_run(
            cache=cache, setup=setup, fits=fits, index=int(run_index[0]), tables=tables
        )
        start = int(full["window_start"][0])
        position = (
            np.searchsorted(
                cache.end_times, rows["valid_time"].dt.replace_time_zone(None).to_numpy()
            )
            - start
        )
        if position.min() < 0 or position.max() >= len(full["n_members"]):
            msg = "a valid time lies outside the run's window"
            raise ValueError(msg)
        piece = {
            "run": rows["run"].to_numpy(),
            "valid_time": rows["valid_time"].dt.replace_time_zone(None).to_numpy(),
            "n_members": full["n_members"][position],
        }
        for name in VARIANTS:
            piece[f"delta_{name}"] = full[f"delta_{name}"][position].astype(np.float32)
        pieces.append(piece)
    out = {name: np.concatenate([p[name] for p in pieces]) for name in pieces[0]}
    if limit:
        return (
            f"SMOKE {site} {period}: {len(out['run']):,} rows, {time.monotonic() - started:.0f} s"
        )
    target = common.MEMBERS_DIR / f"{site}_{period}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **out)  # ty: ignore[invalid-argument-type]
    temporary.rename(target)
    return f"{site} {period}: {len(out['run']):,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Compute every Primary and period in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sites", nargs="+", default=list(c184.SCORED_SITES))
    parser.add_argument("--periods", nargs="+", default=list(common.PERIODS))
    parser.add_argument(
        "--limit-runs", type=int, default=0, help="Smoke test: time this many runs."
    )
    args = parser.parse_args()
    jobs = [
        (s, p, args.limit_runs)
        for p in args.periods
        for s in args.sites
        if args.limit_runs or not (common.MEMBERS_DIR / f"{s}_{p}.npz").exists()
    ]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for line in pool.imap_unordered(job, jobs, chunksize=1):
            print(line, flush=True)


if __name__ == "__main__":
    main()
