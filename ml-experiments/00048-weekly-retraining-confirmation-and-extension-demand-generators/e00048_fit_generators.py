"""Fit the ERA5-trained stage 1a models of the generators and forecast the daily runs.

For each of the 6 solar farms (stage 1a `base`: the weather columns, no CAMS, no stage 0) and the 3
wind farms (stage 1a `margin`: the weather columns with the power curve as the base margin of all
nine levels) this fits two kinds of model, both on ERA5 weather and the farm's cleaned power:

- `frozen`: trained once on valid times before 2025-06-17, forecasting all 365 daily runs;
- `monthly`: refitted on the first day of each month from 2025-07-01 to 2026-06-01 on valid times
  before 5 days earlier (ERA5 reaches the project 5 days late), forecasting the runs whose origin
  falls in the month.

Each model is applied to every ENS member of a run and the members' quantiles are pooled into a
mixture, as in experiment 00033, whose spread-factor rule is applied later by
`e00048_assemble.py`. Seed 0 only.

Output: `work/generator_preds/<arm>/<site>_<label>.parquet`, keyed by series, run, and valid time,
with the nine quantile columns as a fraction of capacity; `<label>` is `frozen`, or the refit date
for `monthly`. Existing files stay, so a stopped run resumes.

    uv run python ml-experiments/00048-*/e00048_fit_generators.py [--workers 3] [--smoke]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Final

import e00048_common as common
import e00048_gen_common as gen
import polars as pl

CORES: Final[int] = 10


def job_path(*, root: Path, arm: str, site: str, label: str) -> Path:
    """Return the file one model's forecasts for one series are written to."""
    return root / arm / f"{site}_{label}.parquet"


def training_table(
    *, row: dict, analysis: pl.DataFrame, observed: pl.DataFrame, train_end: datetime
) -> pl.DataFrame:
    """Return one series' ERA5 rows that have observed power, with the `target` column.

    Args:
        row: The series' row of `gen.site_table()`.
        analysis: The stage 1 pilot's ERA5 analysis rows, keyed by series and valid time.
        observed: `gen.observed_training_power` for the series to `train_end`.
        train_end: Exclusive end of the training window.

    Returns:
        The analysis columns and `target` for valid times from the ERA5 start to `train_end`.
    """
    return (
        analysis.filter(pl.col("time_series_id") == row["time_series_id"])
        .join(
            observed.filter(pl.col("time_series_id") == row["time_series_id"]),
            on=["time_series_id", "valid_time"],
            how="inner",
            validate="1:1",
        )
        .filter((pl.col("valid_time") >= gen.ERA5_START) & (pl.col("valid_time") < train_end))
    )


def forecast_runs(
    *, model: gen.FittedQuantiles, row: dict, origins: list[datetime]
) -> pl.DataFrame:
    """Return the pooled-mixture quantiles of a series for the runs with the given origins."""
    pieces = []
    for origin in origins:
        frame = pl.read_parquet(
            gen.generator_frame_path(run=origin - common.PUBLICATION_DELAY)
        ).filter(pl.col("time_series_id") == row["time_series_id"])
        pieces.append(gen.pooled_mixture(part=frame, member_quantiles=model.predict(frame=frame)))
    return pl.concat(pieces)


def run_series(
    *,
    site: str,
    out_root: Path,
    rounds: int,
    nthread: int,
    limit_runs: int | None,
    limit_refits: int,
) -> list[str]:
    """Fit and write every frozen and monthly model of one generator series.

    Args:
        site: The series' anonymous label (A to F, W1 to W3).
        out_root: The folder holding one subfolder per arm.
        rounds: Boosting rounds.
        nthread: Threads for each fit.
        limit_runs: Forecast only this many runs per model (smoke test), or `None` for all.
        limit_refits: Fit only this many monthly refits (smoke test), or all if not positive.

    Returns:
        One progress line per model fitted.
    """
    sites = gen.site_table()
    row = sites.filter(pl.col("site") == site).row(0, named=True)
    one = sites.filter(pl.col("site") == site)
    technology = row["technology"]
    analysis = pl.read_parquet(gen.ANALYSIS_PATH).filter(
        pl.col("time_series_id") == row["time_series_id"]
    )
    origins = common.all_origins()
    refits = gen.MONTHLY_REFITS[:limit_refits] if limit_refits > 0 else gen.MONTHLY_REFITS
    models: list[tuple[str, str, datetime, list[datetime]]] = [
        ("frozen", "frozen", gen.FROZEN_TRAIN_END, origins)
    ] + [
        (
            "monthly",
            f"{refit:%Y%m%d}",
            gen.training_end_for(refit=refit),
            [o for o in origins if gen.refit_serving(origin=o) == refit],
        )
        for refit in refits
    ]
    lines = []
    for arm, label, train_end, served in models:
        target = job_path(root=out_root, arm=arm, site=site, label=label)
        if target.exists():
            continue
        started = time.monotonic()
        observed = gen.observed_training_power(sites=one, train_end=train_end)
        rows = training_table(row=row, analysis=analysis, observed=observed, train_end=train_end)
        model = gen.FittedQuantiles(
            rows=rows,
            columns=gen.FEATURES[technology],
            seed=common.SEED,
            rounds=rounds,
            use_margin=technology == "wind",
            nthread=nthread,
        )
        used = served[:limit_runs] if limit_runs else served
        forecast = forecast_runs(model=model, row=row, origins=used)
        common.atomic_write(
            frame=forecast.with_columns(seed=pl.lit(common.SEED).cast(pl.Int8)), target=target
        )
        lines.append(
            f"{arm} {site} {label}: {rows.height:,} training rows to {train_end:%Y-%m-%d}, "
            f"{len(used)} runs, {time.monotonic() - started:.0f} s"
        )
    return lines


def main() -> None:
    """Fit every generator series' models, one series per worker process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3, help="processes; each gets 10 // workers")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one solar and one wind farm, the frozen model and 2 monthly refits, 3 runs each",
    )
    args = parser.parse_args()
    sites = gen.site_table()["site"].to_list()
    out_root = common.GENERATOR_PREDS_DIR
    rounds = common.N_ROUNDS
    workers = args.workers
    limit_runs, limit_refits = None, 0
    if args.smoke:
        sites = [sites[0], sites[-1]]
        out_root = out_root.with_name("generator_preds_smoke")
        rounds, workers, limit_runs, limit_refits = 20, 1, 3, 2
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_series,
                site=site,
                out_root=out_root,
                rounds=rounds,
                nthread=max(1, CORES // workers),
                limit_runs=limit_runs,
                limit_refits=limit_refits,
            )
            for site in sites
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
