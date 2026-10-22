"""Score each fold's network on its held-out real transfers, the 17 block days, and injected worlds.

The alarm threshold of every fold's network is the one that leaves 80 unexplained odd-month onset
alarms on the whole record (RO-NC's budget in experiment 181), and likewise for end alarms. A real
transfer counts as found by experiment 130's rule. Run after e00195_train.py for folds 0 to 2:

    OMP_NUM_THREADS=2 uv run --no-sync python ml-experiments/00189-*/e00195_score.py [SEED]
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
from e00195_alarms import (
    alarm_intervals,
    candidates,
    load_model,
    match,
    threshold_for_budget,
)
from e00195_common import (
    BG2_PREDICTIONS,
    GRID_START,
    ML_EXPERIMENTS_DATA_DIR,
    WORK_DIR,
    load_panel,
    noise_units,
    pair_folds,
    real_transfers,
    stamp_months,
)
from e00195_injector import fit_injector
from e00195_net import SERIES_INDEX, Context, build_context
from e00195_realism import recall, standin_peaks, standin_scales
from e00195_worlds import disjoint_world, inject

N_INJECTED: Final[int] = 200
BUDGETS: Final[tuple[int, ...]] = (20, 40, 80, 160, 320)
E174_DAYS: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00174-dp-unlabelled-worst-days-weather-shared-or-single-site"
    / "work"
    / "worst_days_split.parquet"
)
E175_SHAPES: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00175-dp-unlabelled-day-shapes" / "work" / "day_shapes.parquet"
)
E175_GRID_FIRST: Final = datetime(2025, 6, 1, tzinfo=UTC)
BLOCK_TOLERANCE: Final[int] = 6


def block_days() -> pl.DataFrame:
    """The 17 unlabelled worst days that experiment 175 classed as sharp-edged blocks."""
    unlabelled = pl.read_parquet(E174_DAYS).filter(~pl.col("labelled")).select("site", "local_date")
    days = (
        pl.read_parquet(E175_SHAPES)
        .join(unlabelled, on=["site", "local_date"])
        .filter(pl.col("shape") == "a_block")
        .sort("site", "local_date")
    )
    pairs = pl.read_parquet(BG2_PREDICTIONS, columns=["site", "time_series_id"]).unique()
    series = {r["site"]: int(r["time_series_id"]) for r in pairs.iter_rows(named=True)}
    offset = int((E175_GRID_FIRST - GRID_START).total_seconds() // 1800)
    start = (
        pl.col("local_date")
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("Europe/London")
        .dt.convert_time_zone("UTC")
    )
    days = days.with_columns(
        series=pl.col("site").replace_strict(series, return_dtype=pl.Int64),
        edge_stamp=pl.col("edge_position") + offset,
        first=((start.dt.epoch("s") - int(GRID_START.timestamp())) // 1800).cast(pl.Int64),
    ).with_columns(last=pl.col("first") + 47)
    if days.height != 17:
        raise ValueError(f"expected 17 block days, found {days.height}")
    return days


def score_fold(
    *,
    fold: int,
    seed: int,
    ctx: Context,
    panel: dict,
    sigma: dict[int, float],
    folds: dict[str, int],
    rng: np.random.Generator,
    device: str,
) -> dict:
    """Score one fold's network. Returns the summary and the per-event tables."""
    model = load_model(fold=fold, seed=seed, device=device)
    real = candidates(model=model, ctx=ctx, device=device)
    thr_on = threshold_for_budget(table=real.onset)
    thr_end = threshold_for_budget(table=real.end)
    held = [p for p, f in folds.items() if f == fold]
    transfers = real_transfers().filter(pl.col("pair").is_in(held))
    rows = []
    intervals = alarm_intervals(found=real, onset_threshold=thr_on, end_threshold=thr_end)
    for r in transfers.iter_rows(named=True):
        m = match(intervals=intervals, series=tuple(r["series"]), onset=r["onset"], end=r["end"])
        rows.append(
            {
                "card_id": r["card_id"],
                "fold": fold,
                "hours": r["hours"],
                "moved_mw_p50": r["moved_mw_p50"],
                "found_ronc": r["found_ronc"],
                **m,
            }
        )
    # Found against the number of unexplained odd-month alarms.
    curve = {}
    for budget in BUDGETS:
        t_on = threshold_for_budget(table=real.onset, budget=budget)
        t_en = threshold_for_budget(table=real.end, budget=budget)
        iv = alarm_intervals(found=real, onset_threshold=t_on, end_threshold=t_en)
        curve[budget] = [
            int(
                match(intervals=iv, series=tuple(r["series"]), onset=r["onset"], end=r["end"])[
                    "found"
                ]
            )
            for r in transfers.filter(pl.col("hours") > 24).iter_rows(named=True)
        ]
    # The 17 block days: an onset alarm on the series within the local day.
    days = block_days()
    months = stamp_months()
    day_rows = []
    onset_alarms = real.onset.filter(pl.col("p") > thr_on)
    for d in days.iter_rows(named=True):
        near = onset_alarms.filter(
            pl.col("series") == d["series"],
            pl.col("stamp").is_between(d["first"] - BLOCK_TOLERANCE, d["last"] + BLOCK_TOLERANCE),
        )
        day_rows.append(
            {
                "site": d["site"],
                "fold": fold,
                "found_day": near.height > 0,
                "odd_month": int(months[d["edge_stamp"]]) in {1, 3, 5, 7, 9, 11},
            }
        )
    # Injected worlds on odd months: held-out pairs and the pairs the injector was fitted on.
    fit = fit_injector(held_out_pairs=frozenset(held), sigma=sigma)
    held_tuples = tuple(tuple(int(s) for s in p.split(",")) for p in held)
    injected_rows = []
    scales = standin_scales(residual=panel["residual"])
    for label, world_fit in (
        ("held_out_pairs", replace(fit, pairs=held_tuples)),
        ("fitted_pairs", fit),
    ):
        events = disjoint_world(
            fit=world_fit,
            n_events=N_INJECTED,
            residual=panel["residual"],
            power=panel["power"],
            sigma=sigma,
            calendar=ctx.calendar,
            rng=rng,
            allow_months=frozenset({1, 3, 5, 7, 9, 11}),
        )
        residual_injected = inject(residual=panel["residual"], events=events)
        z_injected = ctx.z.copy()
        for s, i in SERIES_INDEX.items():
            if np.isfinite(sigma.get(s, np.nan)):
                z_injected[i] = residual_injected[s] / sigma[s]
        found = candidates(model=model, ctx=ctx, z=z_injected, device=device)
        iv = alarm_intervals(found=found, onset_threshold=thr_on, end_threshold=thr_end)
        peaks = standin_peaks(residual=residual_injected, scales=scales)
        for placed in events:
            m = match(intervals=iv, series=placed.series, onset=placed.onset, end=placed.end)
            injected_rows.append(
                {
                    "world": label,
                    "fold": fold,
                    "hours": (placed.end - placed.onset) / 2,
                    "depth_z": placed.event.depth_z,
                    "shape": placed.event.shape,
                    "standin_recalled": recall(
                        peaks=peaks, series=placed.series, onset=placed.onset
                    ),
                    **m,
                }
            )
    return {
        "fold": fold,
        "thr_on": thr_on,
        "thr_end": thr_end,
        "n_alarms_all": real.onset.filter(pl.col("p") > thr_on).height,
        "real": pl.DataFrame(rows),
        "days": pl.DataFrame(day_rows),
        "injected": pl.DataFrame(injected_rows),
        "curve": curve,
    }


def main(*, seed: int) -> None:
    """Score all three folds, pool the held-out results, and write the tables."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = load_panel()
    sigma = noise_units(residual=panel["residual"])
    ctx = build_context(residual=panel["residual"], sigma=sigma)
    folds = pair_folds()
    rng = np.random.default_rng(189 + seed)
    results = [
        score_fold(
            fold=f,
            seed=seed,
            ctx=ctx,
            panel=panel,
            sigma=sigma,
            folds=folds,
            rng=rng,
            device=device,
        )
        for f in range(3)
    ]
    real = pl.concat([r["real"] for r in results])
    days = pl.concat([r["days"] for r in results])
    injected = pl.concat([r["injected"] for r in results])
    real.write_parquet(WORK_DIR / f"scored_real_seed{seed}.parquet")
    days.write_parquet(WORK_DIR / f"scored_days_seed{seed}.parquet")
    injected.write_parquet(WORK_DIR / f"scored_injected_seed{seed}.parquet")
    long = real.filter(pl.col("hours") > 24)
    errors = long.filter(pl.col("found"))["onset_error"].drop_nans().to_numpy() / 2
    summary = {
        "seed": seed,
        "thresholds": {r["fold"]: [r["thr_on"], r["thr_end"]] for r in results},
        "n_alarms_all_months": {r["fold"]: r["n_alarms_all"] for r in results},
        "long_found": int(long["found"].sum()),
        "long_onset_found": int(long["onset_found"].sum()),
        "long_n": long.height,
        "long_ronc_found": int(long["found_ronc"].sum()),
        "all_found": int(real["found"].sum()),
        "all_onset_found": int(real["onset_found"].sum()),
        "all_n": real.height,
        "all_ronc_found": int(real["found_ronc"].sum()),
        "median_onset_error_h_found": float(np.median(errors)) if errors.size else None,
        "median_onset_error_h_onset_found": float(
            np.nanmedian(long.filter(pl.col("onset_found"))["onset_error"].to_numpy() / 2)
        ),
        "found_by_budget_long": {
            str(b): int(sum(sum(r["curve"][b]) for r in results)) for b in BUDGETS
        },
        "block_days_found": int(days["found_day"].sum()),
        "block_days_found_odd_month": int(days.filter(pl.col("odd_month"))["found_day"].sum()),
        "block_days_odd_month_n": int(days["odd_month"].sum()),
        "block_days_found_by_fold": {
            int(f): int(g["found_day"].sum()) for (f,), g in days.group_by("fold")
        },
        "injected_found": {w: float(g["found"].mean()) for (w,), g in injected.group_by("world")},
        "injected_onset_found": {
            w: float(g["onset_found"].mean()) for (w,), g in injected.group_by("world")
        },
        "injected_standin_recall": {
            w: float(g["standin_recalled"].mean()) for (w,), g in injected.group_by("world")
        },
        "injected_long_found": {
            w: float(g.filter(pl.col("hours") > 24)["found"].mean())
            for (w,), g in injected.group_by("world")
        },
    }
    (WORK_DIR / f"summary_seed{seed}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
