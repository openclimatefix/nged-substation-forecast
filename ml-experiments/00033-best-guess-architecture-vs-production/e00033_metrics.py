"""Compute every per-case metric of every arm of experiment 00033.

Arms come from `work/arms/` (demand, one table per arm) and `work/generator_tables/` (solar and
wind). Each arm's table holds every case it has with finite quantiles and a finite observation
before 2026-07-01. The `shared` column marks the cases that every arm of the family holds; the
absolute scores use those, and each contrast uses the cases its own two arms share. The thresholds
are each series' `historical_p99` and `historical_p95` (fractions of capacity).

Output: `work/metrics/<group>/<arm>.parquet` with the run keys, `seed`, `lead_hours`, the metrics
of `case_metrics`, and `work/metrics/shared_counts.parquet` (cases kept and dropped per group).

    uv run python ml-experiments/00033-*/e00033_metrics.py
"""

from pathlib import Path
from typing import Final

import e00033_common as common
import e00033_fit_generators as generators
import numpy as np
import polars as pl
from e00033_gen_common import site_table

METRICS_DIR: Final[Path] = common.WORK_DIR / "metrics"
TABLES_DIR: Final[Path] = common.WORK_DIR / "generator_tables"
DEMAND_GROUPS: Final[tuple[str, ...]] = ("Primary", "BSP", "GSP")
GENERATOR_GROUPS: Final[tuple[str, ...]] = ("solar", "wind")
SHARED_FLOOR: Final[float] = 0.9
"""Each group must keep at least this share of its largest arm's cases."""
FINITE: Final = pl.all_horizontal(pl.col(*common.LEVEL_COLUMNS).is_finite())


def demand_arm_names() -> list[str]:
    """Return the demand arm names found in the arms folder."""
    names = {
        p.stem
        for p in common.ARMS_DIR.glob("*.parquet")
        if "_seed" not in p.stem and not p.stem.endswith(".tmp")
    }
    return sorted(names)


def generator_arm_names() -> list[str]:
    """Return the generator arm names found in the tables folder."""
    return sorted(p.stem for p in TABLES_DIR.glob("*.parquet"))


def load_demand(*, name: str, groups: pl.DataFrame) -> pl.DataFrame:
    """Return a demand arm with its group and lead, restricted to finite cases."""
    return (
        pl.read_parquet(common.ARMS_DIR / f"{name}.parquet")
        .join(groups, on="time_series_id", how="left", validate="m:1")
        .filter(FINITE & pl.col("actual").is_finite() & (pl.col("valid_time") < common.HOLDOUT))
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
    )


def load_generator(*, name: str) -> pl.DataFrame:
    """Return a generator arm restricted to finite cases, the observation named `actual`."""
    return (
        pl.read_parquet(TABLES_DIR / f"{name}.parquet")
        .rename({"actual_norm": "actual", "technology": "group"})
        .filter(FINITE & pl.col("actual").is_finite() & (pl.col("valid_time") < common.HOLDOUT))
    )


def main() -> None:
    """Write every arm's metrics for every group."""
    series = common.series_table()
    arrays = common.load_series_arrays(series=series)
    groups = series.select("time_series_id", group=pl.col("substation_type").cast(pl.String))
    thresholds = {
        int(row["time_series_id"]): common.historical_thresholds(power=arrays[row["site"]].power)
        for row in series.iter_rows(named=True)
    }
    sites = site_table()
    power = generators.full_power(sites=sites)
    thresholds.update(
        {int(s): common.historical_thresholds(power=power[int(s)]) for s in sites["time_series_id"]}
    )
    pl.DataFrame([{"time_series_id": k, **v} for k, v in thresholds.items()]).write_parquet(
        common.WORK_DIR / "thresholds.parquet"
    )
    demand_names, generator_names = demand_arm_names(), generator_arm_names()
    counts = []
    for group in (*DEMAND_GROUPS, *GENERATOR_GROUPS):
        is_demand = group in DEMAND_GROUPS
        available = demand_names if is_demand else generator_names
        keys: pl.DataFrame | None = None
        sizes = {}
        for name in available:
            frame = (
                load_demand(name=name, groups=groups) if is_demand else load_generator(name=name)
            )
            cases = frame.filter(pl.col("group") == group).select(*common.KEYS).unique()
            if cases.is_empty():
                continue
            sizes[name] = cases.height
            keys = cases if keys is None else keys.join(cases, on=list(common.KEYS), how="inner")
        names = list(sizes)
        if keys is None or keys.is_empty() or keys.height < SHARED_FLOOR * max(sizes.values()):
            shared = 0 if keys is None else keys.height
            msg = f"{group}: {shared} shared cases fall below the floor; arm sizes {sizes}"
            raise ValueError(msg)
        out = METRICS_DIR / group
        out.mkdir(parents=True, exist_ok=True)
        keys.write_parquet(out / "shared_keys.parquet")
        for name in names:
            frame = (
                load_demand(name=name, groups=groups) if is_demand else load_generator(name=name)
            )
            kept = (
                frame.filter(pl.col("group") == group)
                .join(
                    keys.with_columns(shared=pl.lit(value=True)),
                    on=list(common.KEYS),
                    how="left",
                    maintain_order="left",
                )
                .with_columns(pl.col("shared").fill_null(value=False))
                .sort(*common.KEYS, "seed")
            )
            quantiles = np.sort(
                kept.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1
            )
            actual = kept["actual"].to_numpy().astype(np.float64)
            ids = kept["time_series_id"].to_numpy()
            levels = {
                label: np.array([thresholds[int(i)][label] for i in ids])
                for label in ("p99", "p95")
            }
            metrics = common.case_metrics(quantiles=quantiles, actual=actual, thresholds=levels)
            table = kept.select(
                *common.KEYS,
                "seed",
                "shared",
                lead_hours=(
                    (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                    / 60.0
                ),
            ).with_columns(
                *[pl.Series(label, values.astype(np.float32)) for label, values in metrics.items()]
            )
            table.write_parquet(out / f"{name}.parquet")
            counts.append(
                {
                    "group": group,
                    "arm": name,
                    "cases_before": sizes[name],
                    "cases_shared": keys.height,
                }
            )
        print(f"{group}: {len(names)} arms on {keys.height:,} shared cases", flush=True)
    pl.DataFrame(counts).write_parquet(METRICS_DIR / "shared_counts.parquet")


if __name__ == "__main__":
    main()
