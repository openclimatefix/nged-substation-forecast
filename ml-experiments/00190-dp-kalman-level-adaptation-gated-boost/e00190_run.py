"""Tune the gate, then build arms A0 to A3 of experiment 190 at the five decision Primaries.

| Arm | Level process noise | Counter | Level shared across bands |
|---|---|---|---|
| `a0` | experiment 136's POOL value | none | no |
| `a1` | times m while the counter is on | pooled over the four bands | no |
| `a2` | times m at every reading | none | no |
| `a3` | as `a1` | as `a1` | yes, while the counter is on |

Each arm's stage-1 forecast is the T-all prediction on the ensemble-mean weather plus the filter's
correction. The final forecast of an arm is experiment 155's arm D built on that stage-1 forecast
(winter night offset, recipe R, spread refitted on the tuning months), written as `d<n>`.

Stages: `base` (the unfiltered forecasts), `tune` (the grid on the tuning months), `build` (arms).

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00190-*/e00190_run.py --stage base [--workers 4]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Final

import e00190_base as base
import e00190_common as common
import e00190_corrections as corr
import e00190_ens as ens
import e00190_gated as gated
import e00190_stage2_common as stage2
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

SITES: Final[tuple[str, ...]] = common.DECISION_SITES
GRID_M: Final[tuple[float, ...]] = (3.0, 5.0, 10.0)
GRID_K: Final[tuple[int, ...]] = (4, 6, 8)
GRID_Z: Final[tuple[float, ...]] = (0.5, 1.0)
PREREGISTERED: Final[gated.Gate] = gated.Gate(mode="gated", m=5.0, k=6, z=1.0)
CLEAR_WINNER_POINTS: Final[float] = 0.005
"""A grid cell replaces the pre-registered point when it lowers the tuning score by this much."""
ARMS: Final[tuple[str, ...]] = ("a0", "a1", "a2", "a3")
ARM_MODES: Final[dict[str, str]] = {"a0": "none", "a1": "gated", "a2": "always", "a3": "shared"}
WORK: Final = common.WORK_DIR
FROZEN_PATH: Final = WORK / "frozen.json"
OUT_COLUMNS: Final[tuple[str, ...]] = (*stage2.BASE_COLUMNS, "arm")


def quantile_score(*, frame: pl.DataFrame) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 levels, per row, in points."""
    return ens.quantile_score(
        actual=frame["actual"].to_numpy().astype(np.float64),
        quantiles=frame.select(list(stage2.QUANTILES)).to_numpy().astype(np.float64),
    )


def _write(*, frame: pl.DataFrame, path) -> None:  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(path)


def _base_job(site: str) -> str:
    torch.set_num_threads(1)
    started = time.monotonic()
    context = base.load_site(site=site)
    frame = base.build_base(context=context)
    _write(frame=frame, path=base.base_path(site=site))
    return f"{site}: base {frame.height:,} rows, {time.monotonic() - started:.0f} s"


def grid() -> list[gated.Gate]:
    """Return the pre-registered tuning grid."""
    return [gated.Gate(mode="gated", m=m, k=k, z=z) for m, k, z in product(GRID_M, GRID_K, GRID_Z)]


def _tune_job(site: str) -> dict:
    """Return, per grid cell, the tuning-month score sum and row count at a site."""
    torch.set_num_threads(1)
    started = time.monotonic()
    context = base.load_site(site=site)
    stream = base.residual_stream(context=context)
    settings = base.pool_settings()
    band = context.cache.calendar.band
    frame = pl.read_parquet(base.base_path(site=site))
    tuning = frame.filter(pl.col("period") == "tuning")

    def mean_of(gate: gated.Gate) -> pl.DataFrame:
        run = gated.gated_stream(residual=stream, band=band, settings=settings, gate=gate)
        return base.arm_mean_frame(base=tuning, trace=run.stream.trace)

    reference = mean_of(gated.Gate(mode="none"))
    spread = ens.fit_spread_from_frame(frame=reference)
    keep = (
        ~tuning["switching_mask"].to_numpy()
        & np.isfinite(tuning["actual"].to_numpy())
        & (tuning["lead_hours"].to_numpy() <= stage2.HORIZON_HOURS)
    )
    out = {"rows": int(keep.sum()), "cells": {}}
    for label, gate in [("a0", gated.Gate(mode="none")), *[(_label(g), g) for g in grid()]]:
        with_mean = mean_of(gate)
        mean = with_mean["mean"].to_numpy().astype(np.float64)
        quantiles = spread.quantiles(
            mean=mean,
            band=with_mean["hour_band"].to_numpy().astype(np.int64),
            quarter=with_mean["quarter"].to_numpy().astype(np.int64),
            lead_hours=with_mean["lead_hours"].to_numpy(),
        )
        score = ens.quantile_score(
            actual=with_mean["actual"].to_numpy().astype(np.float64), quantiles=quantiles
        )
        out["cells"][label] = float(score[keep].sum())
    out["seconds"] = time.monotonic() - started
    out["site"] = site
    return out


def _label(gate: gated.Gate) -> str:
    return f"m{gate.m:g}_k{gate.k}_z{gate.z:g}"


def freeze(*, results: list[dict]) -> dict:
    """Choose the gate from the pooled tuning scores and write `frozen.json`."""
    rows = sum(r["rows"] for r in results)
    pooled = {
        label: sum(r["cells"][label] for r in results) / rows for label in results[0]["cells"]
    }
    registered = _label(PREREGISTERED)
    best = min((label for label in pooled if label != "a0"), key=pooled.__getitem__)
    chosen = best if pooled[registered] - pooled[best] >= CLEAR_WINNER_POINTS else registered
    parts = {p[0]: float(p[1:]) for p in chosen.split("_")}
    frozen = {
        "chosen": chosen,
        "m": parts["m"],
        "k": int(parts["k"]),
        "z": parts["z"],
        "best_cell": best,
        "pooled_tuning_scores": pooled,
        "tuning_rows": rows,
        "clear_winner_points": CLEAR_WINNER_POINTS,
    }
    common.write_json_atomic(path=FROZEN_PATH, content=frozen)
    return frozen


def frozen_gate(*, mode: str) -> gated.Gate:
    """Return the frozen gate with a mode."""
    frozen = json.loads(FROZEN_PATH.read_text())
    return gated.Gate(mode=mode, m=frozen["m"], k=frozen["k"], z=frozen["z"])  # ty: ignore[invalid-argument-type]


def stage2_frame(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 155's arm D built on a stage-1 forecast frame.

    Args:
        frame: The stage-1 forecast with `period`, sorted by origin and valid time.

    Returns:
        The final forecast frame with the same rows and columns.
    """
    calendar = corr.add_calendar(frame=frame)
    mean_a = calendar["mean"].to_numpy().astype(np.float64)
    offset, _ = corr.winter_offset_shift(frame=calendar, variant="winter")
    mean_b = mean_a + offset
    shift_d = offset + corr.recipe_r_shift(frame=calendar, forecast=mean_b)
    return corr.apply_shift_and_refit_spread(frame=calendar, shift=shift_d).select(
        [*stage2.BASE_COLUMNS, "period"]
    )


def _build_job(site: str) -> str:
    torch.set_num_threads(1)
    started = time.monotonic()
    context = base.load_site(site=site)
    stream = base.residual_stream(context=context)
    settings = base.pool_settings()
    band = context.cache.calendar.band
    frame = pl.read_parquet(base.base_path(site=site))
    triggers = []
    for arm in ARMS:
        run = gated.gated_stream(
            residual=stream, band=band, settings=settings, gate=frozen_gate(mode=ARM_MODES[arm])
        )
        if arm == "a1":
            triggers.append(
                pl.DataFrame(
                    {
                        "site": site,
                        "trigger_end_time": pl.Series(
                            context.cache.end_times[run.trigger_positions]
                        ).dt.replace_time_zone("UTC"),
                        "trigger_position": run.trigger_positions,
                    }
                )
            )
            on = pl.DataFrame({"site": site, "on_position": run.on_positions})
            _write(frame=on, path=WORK / "triggers" / f"on_{site}.parquet")
        stage1 = base.arm_frame(
            mean_frame=base.arm_mean_frame(base=frame, trace=run.stream.trace)
        ).sort(["power_fcst_init_time", "valid_time"])
        final = stage2_frame(frame=stage1.select([*stage2.BASE_COLUMNS, "period"]))
        for period in stage2.PERIODS:
            for prefix, built in (("s", stage1), ("d", final)):
                part = built.filter(pl.col("period") == period).with_columns(
                    arm=pl.lit(f"{prefix}{arm[1:]}")
                )
                _write(
                    frame=part.select(list(OUT_COLUMNS)),
                    path=common.WORK_DIR
                    / "dp_preds"
                    / f"{prefix}{arm[1:]}_{period}_{site}.parquet",
                )
    _write(frame=pl.concat(triggers), path=WORK / "triggers" / f"trigger_{site}.parquet")
    return f"{site}: arms built, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Run one stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("base", "tune", "build"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sites", nargs="+", default=list(SITES))
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.stage == "base":
            for line in pool.map(_base_job, args.sites):
                print(line, flush=True)
        elif args.stage == "tune":
            results = list(pool.map(_tune_job, args.sites))
            print(json.dumps(freeze(results=results), indent=2), flush=True)
        else:
            for line in pool.map(_build_job, args.sites):
                print(line, flush=True)


if __name__ == "__main__":
    main()
