"""Time the framework module on the CPU and on the GPU, and decide whether Part 3 runs on the GPU.

Run this only while the manager has paused other GPU work (about 15 minutes). The same code and
cached tensors run on both devices, with the CPU using the slot's 8 threads.

1. **Single fit.** The convex model with drift D1 on the six scored substations at the 2025-07-01
   cut, from the cached tensors to predictions on every scored row. Float64 on both devices: the
   normal equations are too badly conditioned for float32 (the solver's tolerance is dtype-aware
   for that reason). Predictions must agree within 0.01 points.
2. **Stacked bootstrap.** Twenty month-block bootstrap weightings of the same fit as one batched
   model per substation (a resample is a row of row weights). Float64; every member's predictions
   must agree within 0.01 points between the devices.
3. **Part 3 proxy.** The pooled configuration with the interaction network (two hidden layers of
   64) on all MW substations plus the MVA substations' dark rows, 200 Adam steps in float32, from
   building the batch to the final centring pass. The two devices start from the same seed but
   float rounding makes the paths diverge, so the report gives each device's final loss rather than
   asking for agreement.

**Rule:** the differentiable-physics series uses the GPU from Part 3 on if workload 3 runs more
than 3 times faster end to end on the GPU. Workloads 1 and 2 are reported beside it. Peak GPU
memory is recorded for each workload.

    uv run python ml-experiments/00104-*/e00104_benchmark.py [--smoke] [--devices cpu cuda]
"""

import argparse
import json
import time
from collections.abc import Callable
from typing import Any, Final

import e00104_common as common
import numpy as np
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    GamSpec,
    TrainingWindow,
    fit_gam,
    month_block_weights,
    predict_gam,
    prepare_fit,
)
from studies.dp_pooled import DpConfig, PooledDemandModel, build_pooled_batch, fit_pooled

N_BOOTSTRAP: Final[int] = 20
POOLED_STEPS: Final[int] = 200
SPEEDUP_RULE: Final[float] = 3.0
AGREEMENT_POINTS: Final[float] = 0.01
CPU_THREADS: Final[int] = 8
POOLED_SITES_ROLES: Final[tuple[str, ...]] = ("mw_long", "mw_short", "mva_dark")


def timed(*, device: str, work: Callable[[], object]) -> tuple[float, int, object]:
    """Run `work`, return its wall seconds, the peak GPU memory in bytes, and its result."""
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = work()
    if device == "cuda":
        torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    return time.perf_counter() - started, peak, result


def single_fit(
    *, cache: FeatureCache, device: str, sites: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Workload 1: fit G-all+D1 per substation and predict the scored rows."""
    spec = GamSpec(drift="d1")
    out: dict[str, np.ndarray] = {}
    for site in sites:
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=[spec],
            cut=common.FINAL_CUT,
            window=TrainingWindow(),
            season_years=common.SEASON_YEARS,
            device=device,
            dtype=torch.float64,
        )
        index = common.scoring_index(
            cache=cache, site=site, after=common.FINAL_CUT, through=common.SCORING_END
        )
        out[site] = predict_gam(cache=cache, fit=fit, index=index)
    return out


def stacked_bootstrap(
    *, cache: FeatureCache, device: str, sites: tuple[str, ...], n_resamples: int
) -> dict[str, np.ndarray]:
    """Workload 2: all month-block bootstrap fits of each substation as one batched model."""
    spec = GamSpec(drift="d1")
    out: dict[str, np.ndarray] = {}
    for site in sites:
        _, index = prepare_fit(
            cache=cache,
            site=site,
            spec=spec,
            cut=common.FINAL_CUT,
            window=TrainingWindow(),
            season_years=common.SEASON_YEARS,
        )
        weights = month_block_weights(cache=cache, index=index, n_resamples=n_resamples, seed=0)
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=[spec],
            cut=common.FINAL_CUT,
            window=TrainingWindow(),
            season_years=common.SEASON_YEARS,
            weights=weights,
            device=device,
            dtype=torch.float64,
        )
        rows = common.scoring_index(
            cache=cache, site=site, after=common.FINAL_CUT, through=common.SCORING_END
        )
        out[site] = np.stack(
            [predict_gam(cache=cache, fit=fit, index=rows, member=m) for m in range(n_resamples)]
        )
    return out


def pooled_proxy(*, cache: FeatureCache, device: str, steps: int) -> dict[str, float]:
    """Workload 3: the pooled model with the interaction network on all MW and MVA-dark rows."""
    sites = tuple(site for site, f in cache.sites.items() if f.table.role in POOLED_SITES_ROLES)
    batch = build_pooled_batch(
        cache=cache, sites=sites, cut_end=common.to_64(moment=common.FINAL_CUT), device=device
    )
    torch.manual_seed(0)
    initial = torch.tensor(
        [float(np.nanmean(cache.sites[s].y[cache.sites[s].default_rows])) for s in sites],
        dtype=torch.float32,
        device=device,
    )
    model = PooledDemandModel(
        config=DpConfig(interaction_hidden=(64, 64)),
        n_sites=len(sites),
        n_season_years=batch.n_season_years,
        initial_level=initial,
    ).to(device)
    result = fit_pooled(model=model, batch=batch, steps=steps, learning_rate=0.05)
    return {
        "rows": float(batch.n_rows),
        "sites": float(len(sites)),
        "first_loss": result.losses[0],
        "final_loss": result.losses[-1],
        "optimiser_seconds": result.seconds,
    }


def main() -> None:
    """Run the three workloads on each device and write `work/benchmark.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    arguments = parser.parse_args()
    torch.set_num_threads(CPU_THREADS)
    cache = common.load_cache()
    sites = common.SCORED_SITES[:2] if arguments.smoke else common.SCORED_SITES
    n_resamples = 3 if arguments.smoke else N_BOOTSTRAP
    steps = 5 if arguments.smoke else POOLED_STEPS
    report: dict[str, dict] = {}
    outputs: dict[str, dict[str, Any]] = {}
    for device in arguments.devices:
        if device == "cuda":
            # A first call creates the CUDA context and loads the kernels; it is not the workload.
            _ = torch.zeros(1, device="cuda") @ torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
        report[device] = {}
        outputs[device] = {}
        for name, work in (
            ("single_fit", lambda d=device: single_fit(cache=cache, device=d, sites=sites)),
            (
                "stacked_bootstrap",
                lambda d=device: stacked_bootstrap(
                    cache=cache, device=d, sites=sites, n_resamples=n_resamples
                ),
            ),
            ("pooled_proxy", lambda d=device: pooled_proxy(cache=cache, device=d, steps=steps)),
        ):
            seconds, peak, result = timed(device=device, work=work)
            report[device][name] = {"seconds": seconds, "peak_gpu_bytes": peak}
            outputs[device][name] = result
            print(
                f"{device:5} {name:18} {seconds:8.1f} s   peak GPU {peak / 1e9:6.2f} GB", flush=True
            )
    summary: dict[str, object] = {"devices": report, "smoke": arguments.smoke}
    if {"cpu", "cuda"} <= set(arguments.devices):
        for name in ("single_fit", "stacked_bootstrap"):
            worst = max(
                float(np.abs(outputs["cpu"][name][site] - outputs["cuda"][name][site]).max())
                for site in sites
            )
            summary[f"{name}_max_difference_points"] = worst
            print(f"{name}: largest CPU-GPU difference {worst:.2e} points")
        speedup = (
            report["cpu"]["pooled_proxy"]["seconds"] / report["cuda"]["pooled_proxy"]["seconds"]
        )
        summary["pooled_proxy_speedup"] = speedup
        summary["use_gpu_from_part_3"] = bool(speedup > SPEEDUP_RULE)
        for name in ("single_fit", "stacked_bootstrap"):
            summary[f"{name}_speedup"] = (
                report["cpu"][name]["seconds"] / report["cuda"][name]["seconds"]
            )
        verdict = summary["use_gpu_from_part_3"]
        print(f"pooled proxy speedup {speedup:.1f}x; GPU from Part 3: {verdict}")
    summary["pooled_proxy_detail"] = {d: outputs[d]["pooled_proxy"] for d in arguments.devices}
    suffix = "_smoke" if arguments.smoke else ""
    (common.WORK_DIR / f"benchmark{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
