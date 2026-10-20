"""Fit every convex-model arm of experiment 109 at the replicate, scored, and winter cuts.

Each arm is one frozen fit per cut (the other sites keep their 2 years before the cut), scored on
the 12 months that follow. The quantiles are the mean plus a median-centred residual spread from
the replica fit 12 months earlier (`e00109_arms`). `lambda*` and the per-site comparator's
short-history ridge come from `work/tuned.json` (`e00109_tune.py`), fitted on 2023-24 only.

**Arms.**

- All rows: `g` (per site, `lambda = 0`), `n` (`lambda*`), `n_full` (complete pooling), and `n2l_*`
  (separate heating and other-family `lambda`, 3 by 3 around `lambda*`).
- Short history, one held-out site at a time: `n_4w`, `g_4w`, `n_full_4w`, `n_3m`, `g_3m`; the
  onboarding curve `n_2w`, `n_8w`, `n_26w` with their `g_` counterparts; `n_4w_pin`, `g_4w_pin`
  (vintage capacity, an oracle leak); `n_4w_cappool` (capacity shrunk towards the other sites').
- Winter cut (2025-02-01, scored to 2026-01-31): `n_4w` and `g_4w` only.

Each set is stored as `<cut>_<arm>`.

    uv run python ml-experiments/00109-*/e00109_fit.py \
        --cuts replicate scored winter [--workers 6]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Final

import e00109_arms as arms
import e00109_common as common
import polars as pl
from studies.dp_gam_pooled import PoolingStrength

TARGETS: Final[tuple[str, ...]] = common.HELD_OUT_SITES
HEATING_FACTORS: Final[tuple[float, ...]] = (0.1, 1.0, 10.0)
"""The 3 by 3 grid of `n2l_*` multiplies `lambda*` by these, for heating and for the others."""
FACTOR_NAMES: Final[dict[float, str]] = {0.1: "tenth", 1.0: "same", 10.0: "tenfold"}
"""Set names cannot hold a decimal point, so the factors are named."""


@dataclass(frozen=True)
class FitJob:
    """One arm at one cut, for every target site."""

    cut_name: str
    config: arms.ArmConfig


def curve_histories() -> tuple[common.History, ...]:
    """Return the onboarding curve's histories: 2, 8, and 26 weeks."""
    return common.ONBOARDING_CURVE


def catalogue(*, tuned: dict[str, Any], cut_name: str) -> list[arms.ArmConfig]:
    """Return every arm fitted at a cut.

    Args:
        tuned: The contents of `work/tuned.json`.
        cut_name: `replicate`, `scored`, or `winter`.

    Returns:
        The arm configurations.
    """
    star = PoolingStrength.uniform(value=tuned["lambda_star"])
    zero = PoolingStrength.uniform(value=0.0)
    ridge = tuned["g_short_ridge"]

    def gam(name: str, history: common.History, **kwargs: Any) -> arms.ArmConfig:
        return arms.ArmConfig(
            name=name,
            strength=zero,
            history=history,
            alone=True,
            short_ridge=ridge["4w" if history.label not in ridge else history.label],
            **kwargs,
        )

    four, three = common.FOUR_WEEKS, common.THREE_MONTHS
    if cut_name == "winter":
        return [arms.ArmConfig(name="n_4w", strength=star, history=four), gam("g_4w", four)]
    configs = [
        arms.ArmConfig(name="g", strength=zero),
        arms.ArmConfig(name="n", strength=star),
        arms.ArmConfig(name="n_full", kind="tied"),
    ]
    configs += [
        arms.ArmConfig(
            name=f"n2l_h{FACTOR_NAMES[heating]}_o{FACTOR_NAMES[other]}",
            strength=PoolingStrength(
                heating=tuned["lambda_star"] * heating, other=tuned["lambda_star"] * other
            ),
        )
        for heating in HEATING_FACTORS
        for other in HEATING_FACTORS
    ]
    configs += [
        arms.ArmConfig(name="n_4w", strength=star, history=four),
        gam("g_4w", four),
        arms.ArmConfig(name="n_full_4w", kind="tied", history=four),
        arms.ArmConfig(name="n_3m", strength=star, history=three),
        gam("g_3m", three),
        arms.ArmConfig(name="n_4w_pin", strength=star, history=four, oracle_pin=True),
        gam("g_4w_pin", four, oracle_pin=True),
        arms.ArmConfig(name="n_4w_cappool", strength=star, history=four, capacity_prior=True),
    ]
    for history in curve_histories():
        configs += [
            arms.ArmConfig(name=f"n_{history.label}", strength=star, history=history),
            gam(f"g_{history.label}", history),
        ]
    return configs


def run_job(job: FitJob) -> tuple[str, dict[str, dict[str, str]]]:
    """Fit one arm at one cut for every target site.

    Args:
        job: The arm and cut.

    Returns:
        The stored set's name and how each target's fit set the capacities. The worker stores the
        set itself, so a later failure cannot lose an arm that finished.
    """
    import torch

    torch.set_num_threads(1)
    cache, vintages = arms.load_source(source=arms.Source())
    cut = common.CUTS[job.cut_name]
    frames, sources = [], {}
    for site in TARGETS:
        try:
            frame, capacity = arms.fit_and_score_site(
                cache=cache,
                vintages=vintages,
                cut=cut,
                config=job.config,
                target=site,
                decision_pool=common.DECISION_SITES,
            )
        except ValueError as error:
            print(f"{job.cut_name}_{job.config.name} {site}: skipped ({error})", flush=True)
            continue
        frames.append(frame)
        sources[site] = capacity
        print(f"{job.cut_name}_{job.config.name} {site}: {len(frame)} rows", flush=True)
    name = f"{job.cut_name}_{job.config.name}"
    if frames:
        common.store_arm(name=name, frame=pl.concat(frames))
    return name, sources


def main() -> None:
    """Fit all arms at the requested cuts and store their predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuts", nargs="+", default=["replicate", "scored", "winter"])
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    tuned = json.loads((common.WORK_DIR / "tuned.json").read_text())
    common.load_cache()
    jobs = [
        FitJob(cut_name=cut_name, config=config)
        for cut_name in arguments.cuts
        for config in catalogue(tuned=tuned, cut_name=cut_name)
        if not (common.DATA_DIR / "predictions" / f"{cut_name}_{config.name}").exists()
    ]
    path = common.WORK_DIR / "capacity_sources.json"
    capacity_sources = json.loads(path.read_text()) if path.exists() else {}
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        capacity_sources |= dict(pool.map(run_job, jobs))
    path.write_text(json.dumps(capacity_sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
