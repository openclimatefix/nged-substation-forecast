# ruff: noqa: E501
"""Curtailment cost of the label-free mask on experiment 297's injected worlds (Part 1).

Each (site, world) file of experiment 327 holds five quantiles per arm for a sample of rows. Every
row group becomes a node, all with the Primary's limit of f times its largest real observed power
before the earliest scored row, and `studies.node_curtailment` scores them node-only (a Primary has
no published limit, so the result ranks arms and states no money saved).

    PYTHONPATH=ml-experiments/00297-lsc-need-a-detector-no-detector-arms-against-the-mask:ml-experiments/00309-lsc-two-forecast-gap-choosing-without-truth:ml-experiments/00134-differentiable-physics-same-weekday-error-profile-on-gam-k:ml-experiments/00130-ro-nc-detector-on-the-gam-k-residual \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 uv run --no-sync python e00332_inj_cost.py run
"""

from __future__ import annotations

import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Final

import e00297_fed as fed
import numpy as np
import polars as pl
from studies.node_curtailment import (
    NodeCurtailmentAssumptions,
    NodeTable,
    node_curtailment,
    table_with_scaled_limits,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

E327: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"
OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00332-lsc-curtailment-cost-of-label-free-mask"
FRACTIONS: Final[tuple[float, ...]] = (0.5, 0.75, 0.9)
TUNING_MONTHS: Final[frozenset[int]] = frozenset({1, 3, 5, 7, 9, 11})
"""Odd months: offsets are fitted there, so only even months are scored (as in experiment 327)."""
ARM_FILE_TO_NAME: Final[dict[str, str]] = {"K0": "C", "PERFECT": "D", "A": "A", "B": "B"}
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "G0_all": ("ORDINARY", "BEFORE_ALARM", "DURING", "AFTER", "MARGIN"),
    "G1_ordinary": ("ORDINARY",),
    "G2_before_alarm": ("BEFORE_ALARM",),
    "G3_during": ("DURING",),
    "G4_after": ("AFTER",),
    "G5_near": ("BEFORE_ALARM", "DURING", "AFTER", "MARGIN"),
    "DA_during_after": ("DURING", "AFTER"),
}
GROUP_ID: Final[dict[str, int]] = {name: i + 1 for i, name in enumerate(GROUPS)}
QUANTILE_COLUMNS: Final[tuple[str, ...]] = ("q05", "q95")
EARLIEST_SCORED: Final[np.datetime64] = np.datetime64("2024-08-04T00:00")
PRICE: Final[float] = 2000.0
INIT_BASE: Final[np.datetime64] = np.datetime64("2000-01-01T00:00")


def node_table(*, series: int) -> NodeTable:
    """A node-only table: one node per row group, positive power the flow that can breach."""
    ids = list(GROUP_ID.values())
    nodes = pl.DataFrame(
        {
            "node_id": [f"g{i}" for i in ids],
            "time_series_id": ids,
            "node_type": ["BSP"] * len(ids),
            "parent_node_id": [None] * len(ids),
            "export_sign": [1] * len(ids),
            "sign_source": ["assumed: a positive demand reading is the flow that can breach"]
            * len(ids),
        },
        schema_overrides={"parent_node_id": pl.String, "export_sign": pl.Int64},
    )
    limits = pl.DataFrame(
        schema={
            "node_id": pl.String,
            "season": pl.String,
            "export_limit_mw": pl.Float64,
            "import_limit_mw": pl.Float64,
            "limit_source": pl.String,
        }
    )
    generators = pl.DataFrame(
        schema={
            "generator_id": pl.Int64,
            "node_id": pl.String,
            "capacity_mw": pl.Float64,
            "connection_date": pl.Date,
            "is_forecast": pl.Boolean,
        }
    )
    return NodeTable(nodes=nodes, limits=limits, generators=generators)


def limit_mw(*, series: int, fraction: float) -> float:
    """Return f times the largest real observed power of the Primary before the earliest scored row."""
    power = np.load(E327 / "power" / f"power_{series}.npy")
    stamps = np.datetime64("2022-01-01T00:00") + np.arange(len(power)) * np.timedelta64(30, "m")
    return float(fraction * np.nanmax(power[stamps < EARLIEST_SCORED]))


def job(args: tuple[str, int]) -> list[dict]:
    """Score one site in one world on both targets and all three limits."""
    site, world = args
    path = E327 / "inj" / f"{site}_{world}.parquet"
    if not path.exists():
        return []
    series = fed.SERIES_OF_SITE[site]
    cap = fed.CAPACITY_MW[site]
    scale = cap / 100.0
    frame = pl.read_parquet(path).filter(
        ~pl.col("month").is_in(list(TUNING_MONTHS)),
        pl.col("lead_hours") >= 12.0,
        pl.col("lead_hours") < 36.0,
    )
    if frame.is_empty():
        return []
    table0 = node_table(series=series)
    out: list[dict] = []
    all_months = frame
    for season, target in [(s, t) for s in ("all", "winter") for t in ("obs", "nra")]:
        frame = (
            all_months
            if season == "all"
            else all_months.filter(pl.col("month").is_in([11, 12, 1, 2, 3]))
        )
        if frame.is_empty():
            continue
        per_group_frames: dict[str, list[pl.DataFrame]] = {}
        actual = pl.col(f"actual_{target}").cast(pl.Float64) * scale
        variants = [(a, n, False) for a, n in ARM_FILE_TO_NAME.items()]
        if target == "obs":
            variants += [(a, f"{n}_plus", True) for a, n in ARM_FILE_TO_NAME.items() if n != "C"]
        for arm_file, arm, add_back in variants:
            sub = frame.filter(pl.col("arm") == arm_file)
            if add_back:
                moved = (
                    pl.when(pl.col("phase") == "DURING")
                    .then(
                        pl.col("actual_obs").cast(pl.Float64)
                        - pl.col("actual_nra").cast(pl.Float64)
                    )
                    .otherwise(0.0)
                )
                sub = sub.with_columns(
                    q05=pl.col("q05").cast(pl.Float64) + moved,
                    q95=pl.col("q95").cast(pl.Float64) + moved,
                )
            for group, phases in GROUPS.items():
                g = sub.filter(pl.col("phase").is_in(phases))
                base = g.select(
                    time_series_id=pl.lit(GROUP_ID[group], dtype=pl.Int64),
                    power_fcst_init_time=(
                        pl.lit(INIT_BASE).dt.replace_time_zone("UTC")
                        + pl.duration(hours=pl.col("row").cast(pl.Int64))
                    ),
                    lead=pl.duration(minutes=(pl.col("lead_hours") * 60).cast(pl.Int64)),
                    seed=pl.lit(0, dtype=pl.Int64),
                    actual=actual,
                    q05=pl.col("q05").cast(pl.Float64) * scale,
                    q95=pl.col("q95").cast(pl.Float64) * scale,
                ).with_columns(valid_time=pl.col("power_fcst_init_time") + pl.col("lead"))
                per_group_frames.setdefault(arm, []).append(base.drop("lead"))
                if arm == "C":
                    perfect = base.with_columns(q05=pl.col("actual"), q95=pl.col("actual"))
                    per_group_frames.setdefault("perfect", []).append(
                        perfect.drop("lead", strict=False)
                    )
        frames = {arm: pl.concat(parts) for arm, parts in per_group_frames.items()}
        counts = {
            group: int(frames["C"].filter(pl.col("time_series_id") == GROUP_ID[group]).height)
            for group in GROUPS
        }
        for fraction in FRACTIONS:
            lim = limit_mw(series=series, fraction=fraction)
            limits = pl.DataFrame(
                {
                    "node_id": [f"g{i}" for i in GROUP_ID.values()],
                    "season": ["Winter"] * len(GROUP_ID),
                    "export_limit_mw": [lim] * len(GROUP_ID),
                    "import_limit_mw": [None] * len(GROUP_ID),
                    "limit_source": [f"scaled f={fraction}"] * len(GROUP_ID),
                },
                schema_overrides={"import_limit_mw": pl.Float64},
            )
            scored_table = table_with_scaled_limits(
                table=table0,
                scaled=limits.select("node_id", "export_limit_mw", "import_limit_mw"),
                fraction=fraction,
            )
            result = node_curtailment(
                table=scored_table,
                node_forecasts=frames,
                generator_forecasts={},
                assumptions=replace(
                    NodeCurtailmentAssumptions(use_generators=False),
                    lead_min_hours=12.0,
                    lead_max_hours=36.0,
                    headline_price_gbp_per_mwh=PRICE,
                ),
            )
            cost = (
                result.per_node_season.group_by("arm", "node_id")
                .agg(cost=pl.col("energy_proxy_gbp").sum())
                .to_dicts()
            )
            for row in cost:
                group = next(g for g, i in GROUP_ID.items() if f"g{i}" == row["node_id"])
                out.append(
                    {
                        "site": site,
                        "world": world,
                        "season": season,
                        "target": target,
                        "fraction": fraction,
                        "arm": row["arm"],
                        "group": group,
                        "cost": row["cost"],
                        "n": counts[group],
                    }
                )
    return out


def run() -> None:
    """Score every (site, world) in a spawn pool and write the long table."""
    OUT.mkdir(parents=True, exist_ok=True)
    worlds = sorted({int(p.stem.split("_")[1]) for p in (E327 / "inj").glob("*.parquet")})
    tasks = [(s, w) for s in fed.SERIES_OF_SITE for w in worlds]
    rows: list[dict] = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as pool:
        for i, part in enumerate(pool.map(job, tasks, chunksize=1)):
            rows.extend(part)
            if i % 20 == 0:
                print("done", i, "of", len(tasks), flush=True)
    pl.DataFrame(rows).write_parquet(OUT / "inj_cost_long.parquet")
    print("rows", len(rows))


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        run()
    elif sys.argv[1:2] == ["one"]:
        print(job((sys.argv[2], int(sys.argv[3])))[:12])
