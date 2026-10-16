"""Assemble the arms of experiment 00048 into one prediction table per arm.

An arm is a regime (which refit forecasts which daily run) applied to one configuration's refit
predictions (`work/anchor_preds/<config>/`, from `e00048_fit.py`; `work/generator_preds/`, from
`e00048_fit_generators.py`). Every table is keyed by series, run, valid time, and seed, holds the
nine quantiles, the observed power (`actual`, a fraction of capacity), the series label (`site`),
and the group (`Primary`, `BSP`, `GSP`, `solar`, or `wind`), and covers all 365 daily runs. A later
step restricts the rows to a row set.

Demand arms. The base arms are `frozen` (the stack's model), `monthly` and `weekly` (`n1h`), and
`roll12`. Each is written raw, with the spread factor 1.15 (`_s`), and with the 14-day correction
and the spread factor (`_cs`); `weekly` also with the 7-day correction and the spread factor
(`_c7s`). The correction reads, for each past half-hour, the forecast as issued: the forecast of
the latest earlier daily run, from the refit that served that run. Each correction therefore reads
no run before 2025-07-01, the first run of the tables.

Generator arms. `gen_frozen` and `gen_monthly` are the pooled mixtures of the ERA5-trained models,
and `gen_frozen_s` and `gen_monthly_s` apply experiment 00033's spread-factor rule month by month,
each month fitted only on cases whose valid time is at least 3.5 hours before the month's first
origin, the same for both arms.

Baselines. `manual_heuristic` and `climatology` are read from the `power_forecasts` Delta table for
the 292 runs of `fold_a_daily_gap_v1`.

    uv run python ml-experiments/00048-*/e00048_assemble.py [--parts demand generators baselines]
        [--smoke]
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Final

import e00048_common as common
import e00048_fit as fitting
import e00048_gen_common as gen
import e00048_regimes as regimes
import numpy as np
import polars as pl

FOLD_ID: Final[str] = "mid_2025_to_mid_2026"
BASELINES: Final[dict[str, str]] = {
    "manual_heuristic": "manual_heuristic",
    "climatology": "climatology",
}
"""Arm name to the stored `experiment_name` of the `power_forecasts` table."""
BASELINE_CHUNK_RUNS: Final[int] = 20
DEMAND_PLAN: Final[dict[str, tuple[str, str, tuple[str, ...]]]] = {
    "frozen": ("frozen", "frozen", ("s", "cs")),
    "monthly": ("n1h", "monthly", ("s", "cs")),
    "weekly": ("n1h", "weekly", ("s", "cs", "c7s")),
    "roll12": ("roll12", "monthly", ("s", "cs")),
}
"""Each demand base arm: the configuration folder, the regime, and the forms written beyond raw."""
SMOKE_DAYS: Final[frozenset[tuple[int, int]]] = frozenset(
    {(7, 1), (7, 2), (7, 3), (7, 8), (7, 9), (7, 10), (8, 1), (8, 2), (8, 3)}
)
"""The (month, day) of the runs that `e00048_fit.py --smoke` forecasts."""
CORRECTION_WINDOW: Final[dict[str, int | None]] = {"s": None, "cs": 14, "c7s": 7}
"""Each form's correction window in days, or `None` for a form with no correction."""


def regime_rows(
    *,
    root: Path,
    config: str,
    regime: str,
    site: str,
    origins: list[datetime],
    strict: bool,
) -> pl.DataFrame:
    """Return one series' quantiles for a configuration and regime, from the chosen refits.

    Args:
        root: The folder of refit predictions.
        config: The configuration folder under `root`.
        regime: A key of `regimes.REGIMES`.
        site: The series label.
        origins: The `power_fcst_init_time` of the runs to cover.
        strict: Raise if a refit file or a run assigned to it is missing. A smoke test writes only
            some.

    Returns:
        The quantiles keyed by series, run, valid time, and seed.

    Raises:
        ValueError: If `strict` and a refit file lacks a run assigned to it.
        FileNotFoundError: If no refit file of the series exists for the runs.
    """
    pieces = []
    for refit, chosen in regimes.served(name=regime, origins=origins).items():
        path = fitting.job_path(root=root, config=config, site=site, refit=refit)
        if not strict and not path.exists():
            continue
        frame = pl.read_parquet(path).filter(pl.col("power_fcst_init_time").is_in(chosen))
        if strict and frame["power_fcst_init_time"].n_unique() != len(chosen):
            msg = f"{config} {site} refit {refit}: the file lacks some assigned runs"
            raise ValueError(msg)
        pieces.append(frame)
    if not pieces:
        msg = f"no refit file of {config} {site} serves the runs"
        raise FileNotFoundError(msg)
    return pl.concat(pieces)


def attach_actual(
    *,
    table: pl.DataFrame,
    labels: dict[int, tuple[str, str]],
    power_of: dict[int, np.ndarray],
) -> pl.DataFrame:
    """Add `site`, `group`, and `actual` (observed power over capacity) to a quantile table.

    Args:
        table: Quantiles keyed by series, run, valid time, and seed.
        labels: Each series identifier's site label and group.
        power_of: Each series identifier's observed power over capacity on the dense grid.

    Returns:
        The table with the three columns added, `actual` NaN where the grid has no observation.
    """
    pieces = []
    for (series_id,), part in table.partition_by("time_series_id", as_dict=True).items():
        site, group = labels[int(series_id)]
        index = np.asarray(common.grid_index(times=part["valid_time"]))
        pieces.append(
            part.with_columns(
                site=pl.lit(site),
                group=pl.lit(group),
                actual=pl.Series(common.take(values=power_of[int(series_id)], index=index)),
            )
        )
    return pl.concat(pieces).sort(*common.KEYS, "seed")


def write_arm(*, name: str, table: pl.DataFrame, smoke: bool) -> None:
    """Write one arm's table."""
    folder = common.ARMS_DIR.with_name("arms_smoke") if smoke else common.ARMS_DIR
    common.atomic_write(frame=table, target=folder / f"{name}.parquet")


def assemble_demand(*, smoke: bool) -> None:
    """Write every demand base arm in its raw, spread, and corrected forms."""
    series = common.series_table()
    arrays = common.load_series_arrays(series=series)
    labels = {
        int(row["time_series_id"]): (row["site"], row["substation_type"])
        for row in series.iter_rows(named=True)
    }
    power_of = {
        int(row["time_series_id"]): arrays[row["site"]].power
        for row in series.iter_rows(named=True)
    }
    root = (
        common.ANCHOR_PREDS_DIR.with_name("anchor_preds_smoke")
        if smoke
        else common.ANCHOR_PREDS_DIR
    )
    origins = common.all_origins()
    if smoke:
        origins = [o for o in origins if (o.month, o.day) in SMOKE_DAYS]
    for name, (config_name, regime, forms) in DEMAND_PLAN.items():
        config = fitting.CONFIGS[config_name]
        wanted = series.filter(pl.col("substation_type").is_in(config.groups))
        pieces = []
        for site in wanted["site"].to_list():
            try:
                pieces.append(
                    regime_rows(
                        root=root,
                        config=config_name,
                        regime=regime,
                        site=site,
                        origins=origins,
                        strict=not smoke,
                    )
                )
            except FileNotFoundError:
                continue
        if not pieces:
            print(f"{name}: refit predictions missing, skipped", flush=True)
            continue
        started = time.monotonic()
        raw = pl.concat(pieces)
        write_arm(
            name=name,
            table=attach_actual(table=raw, labels=labels, power_of=power_of),
            smoke=smoke,
        )
        for form in forms:
            window = CORRECTION_WINDOW[form]
            if window is None:
                table = common.stack(frame=raw, correct=False, spread=True)
            else:
                corrected = common.with_corrections(
                    base=raw, series=series, arrays=arrays, window_days=window
                )
                table = common.stack(frame=corrected, correct=True, spread=True)
            write_arm(
                name=f"{name}_{form}",
                table=attach_actual(table=table, labels=labels, power_of=power_of),
                smoke=smoke,
            )
        print(
            f"{name}: raw and {len(forms)} forms, {raw.height:,} rows, "
            f"{time.monotonic() - started:.0f} s",
            flush=True,
        )


def assemble_generators(*, smoke: bool) -> None:
    """Write the frozen and monthly generator arms, raw and with the spread factor."""
    sites = gen.site_table()
    labels = {
        int(row["time_series_id"]): (row["site"], row["technology"])
        for row in sites.iter_rows(named=True)
    }
    power_of = gen.full_power(sites=sites)
    root = (
        common.GENERATOR_PREDS_DIR.with_name("generator_preds_smoke")
        if smoke
        else common.GENERATOR_PREDS_DIR
    )
    for arm in ("frozen", "monthly"):
        files = sorted((root / arm).glob("*.parquet")) if (root / arm).exists() else []
        if not files:
            print(f"gen_{arm}: forecasts missing, skipped", flush=True)
            continue
        raw = pl.concat([pl.read_parquet(path) for path in files])
        table = attach_actual(table=raw, labels=labels, power_of=power_of)
        cases = table.filter(
            pl.col("actual").is_finite() & (pl.col("valid_time") < common.HOLDOUT)
        ).with_columns(
            lead_hours=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
            ),
            actual_norm=pl.col("actual"),
        )
        cases = cases.with_columns(
            band=pl.Series(gen.band_names(lead_hours=cases["lead_hours"].to_numpy()))
        )
        spread_pieces = []
        for technology in ("solar", "wind"):
            part = cases.filter(pl.col("group") == technology)
            if part.is_empty():
                continue
            mixture = part.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
            scaled = gen.apply_spread(cases=part, mixture=mixture, technology=technology)
            spread_pieces.append(
                part.drop(common.LEVEL_COLUMNS).hstack(
                    pl.DataFrame(scaled.astype(np.float32), schema=list(common.LEVEL_COLUMNS))
                )
            )
        keep = [*common.KEYS, "seed", *common.LEVEL_COLUMNS, "site", "group", "actual"]
        write_arm(name=f"gen_{arm}", table=cases.select(keep), smoke=smoke)
        write_arm(
            name=f"gen_{arm}_s",
            table=pl.concat(spread_pieces).select(keep).sort(*common.KEYS, "seed"),
            smoke=smoke,
        )
        print(f"gen_{arm}: {cases.height:,} cases with observed power", flush=True)


def stored_quantiles(
    *, experiment_name: str, capacity: pl.DataFrame, origins: list[datetime]
) -> pl.DataFrame:
    """Return one stored baseline's capacity-normalised member quantiles for some origins.

    Args:
        experiment_name: The `experiment_name` of the `power_forecasts` table.
        capacity: `time_series_id` and `capacity` (the effective capacity in MW) of the series.
        origins: The runs' `power_fcst_init_time`.

    Returns:
        The nine quantiles of each case's members divided by the series' capacity, keyed by series,
        run, and valid time, with the member count.

    Raises:
        ValueError: If a case holds duplicate member rows.
    """
    frame = (
        pl.scan_delta(str(common.POWER_FORECASTS_DIR))
        .filter(
            (pl.col("experiment_name") == experiment_name)
            & (pl.col("fold_id") == FOLD_ID)
            & pl.col("power_fcst_init_time").is_in(origins)
            & pl.col("time_series_id").is_in(capacity["time_series_id"].to_list())
        )
        .group_by("time_series_id", "power_fcst_init_time", "valid_time")
        .agg(
            *[
                pl.col("power_fcst").cast(pl.Float64).quantile(float(level), "linear").alias(name)
                for level, name in zip(common.LEVELS, common.LEVEL_COLUMNS, strict=True)
            ],
            members=pl.col("ensemble_member").n_unique(),
            stored_rows=pl.len(),
        )
        .join(capacity.lazy(), on="time_series_id", how="left", validate="m:1")
        .with_columns(*[pl.col(name) / pl.col("capacity") for name in common.LEVEL_COLUMNS])
        .drop("capacity")
        .collect(engine="streaming")
    )
    if frame.filter(pl.col("members") != pl.col("stored_rows")).height:
        msg = f"{experiment_name}: a case holds duplicate member rows"
        raise ValueError(msg)
    return frame.drop("stored_rows")


def assemble_baselines(*, smoke: bool) -> None:
    """Write `manual_heuristic` and `climatology` for the 292 gap runs, all 29 series."""
    series = common.series_table()
    arrays = common.load_series_arrays(series=series)
    sites = gen.site_table()
    power_grid = gen.full_power(sites=sites)
    labels = {
        **{
            int(r["time_series_id"]): (r["site"], r["substation_type"])
            for r in series.iter_rows(named=True)
        },
        **{
            int(r["time_series_id"]): (r["site"], r["technology"])
            for r in sites.iter_rows(named=True)
        },
    }
    power_of = {
        **{int(r["time_series_id"]): arrays[r["site"]].power for r in series.iter_rows(named=True)},
        **power_grid,
    }
    capacity = pl.concat(
        [
            series.select(
                "time_series_id", capacity=pl.col("effective_capacity_mw").cast(pl.Float64)
            ),
            sites.select(
                "time_series_id", capacity=pl.col("effective_capacity_mw").cast(pl.Float64)
            ),
        ]
    )
    origins = common.origins_of_subset(subset="gap")
    if smoke:
        capacity = capacity.head(3)
        origins = origins[:2]
    for name, experiment_name in BASELINES.items():
        started = time.monotonic()
        pieces = [
            stored_quantiles(
                experiment_name=experiment_name,
                capacity=capacity,
                origins=origins[start : start + BASELINE_CHUNK_RUNS],
            )
            for start in range(0, len(origins), BASELINE_CHUNK_RUNS)
        ]
        stored = pl.concat(pieces).with_columns(seed=pl.lit(common.SEED).cast(pl.Int8))
        table = attach_actual(table=stored.drop("members"), labels=labels, power_of=power_of)
        write_arm(name=name, table=table, smoke=smoke)
        print(
            f"{name}: {table.height:,} rows, fewest members {int(stored['members'].min())}, "  # ty: ignore[invalid-argument-type]
            f"{time.monotonic() - started:.0f} s",
            flush=True,
        )


def main() -> None:
    """Build and write the requested parts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("demand", "generators", "baselines"),
        default=["demand", "generators", "baselines"],
    )
    parser.add_argument("--smoke", action="store_true", help="read smoke refit files, write apart")
    args = parser.parse_args()
    if "demand" in args.parts:
        assemble_demand(smoke=args.smoke)
    if "generators" in args.parts:
        assemble_generators(smoke=args.smoke)
    if "baselines" in args.parts:
        assemble_baselines(smoke=args.smoke)


if __name__ == "__main__":
    main()
