"""Fit the rounds family of experiment 00080: do more rounds or early stopping remove the gap?

For each series and each starting value (`def`: XGBoost's estimate from the labels; `m0`: zero,
`base_score` 0), on the frozen configuration of the series' class:

- Fit A trains on the whole frozen window for 2000 rounds. Boosting is sequential, so the first 500,
  1000, and 2000 rounds of the booster are the models that 500, 1000, and 2000 rounds would give,
  and the script writes the predictions of each prefix as the arms `frozen_<start>_r<rounds>`.
- Fit B trains on the frozen window without its last 90 days and scores those 90 days every round
  (the mean pinball loss over the nine levels, with the recency weights of the training rows for
  demand), stopping after 100 rounds without a lower loss, at most 2000 rounds. It writes the arms
  `frozenes_<start>_best` (the first `best_iteration + 1` rounds) and `frozenes_<start>_r500` (the
  500-round prefix of the same booster, the matched fixed-round arm), and the learning curves to
  `work/curves/`.

Wind farms use models without the power-curve margin, because a margin overrides the starting
value. When early stopping ends before round 500 the booster holds fewer than 500 rounds, and the
matched 500-round arm of that series is not written. Predictions cover the 292 runs of
`fold_a_daily_gap_v1`. A job whose files all exist is skipped, so a stopped run resumes. Seed 0
only.

    uv run python ml-experiments/00080-*/e00080_rounds.py --group Primary [--workers 3] [--smoke]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00080_common as common
import e00080_fit as fitting
import e00080_fit_generators as gen_fit
import e00080_gen_common as gen
import e00080_regimes as regimes
import polars as pl
import xgboost as xgb

CORES: Final[int] = 6
STARTS: Final[tuple[str, ...]] = ("def", "m0")
FIXED_ROUNDS: Final[tuple[int, ...]] = (500, 1000, 2000)
MAX_ROUNDS: Final[int] = 2000
PATIENCE: Final[int] = 100
HOLDOUT_DAYS: Final[int] = 90
GROUPS: Final[tuple[str, ...]] = ("Primary", "BSP", "GSP", "solar", "wind")
CURVES_DIR: Final[Path] = common.WORK_DIR / "curves"


def curve_frame(*, history: dict, group: str, site: str, start: str) -> pl.DataFrame:
    """Return the training and validation losses of every round."""
    train = history["train"]["quantile"] if "train" in history else []
    valid = history["valid"]["quantile"]
    return pl.DataFrame(
        {
            "group": group,
            "site": site,
            "start": start,
            "round": list(range(1, len(valid) + 1)),
            "valid": valid,
            "train": train or [None] * len(valid),
        },
        schema_overrides={"train": pl.Float64},
    )


def demand_job(*, site: str, nthread: int, smoke: bool) -> list[str]:
    """Fit A and fit B for one demand series, both starts, and write their predictions."""
    series = common.series_table()
    row = series.filter(pl.col("site") == site).row(0, named=True)
    arrays = common.load_series_arrays(series=series.filter(pl.col("site") == site))[site]
    config = fitting.CONFIGS["frozen_m0"]
    refit = regimes.FROZEN_REFIT
    root = common.ANCHOR_PREDS_DIR.with_name("anchor_preds_smoke" if smoke else "anchor_preds")
    origins = common.origins_of_subset(subset="gap")
    if smoke:
        origins = origins[:3]
    all_rows = fitting.known_rows(
        rows=fitting.training_rows(row=row, arrays=arrays, frozen=True), refit=refit, config=config
    )
    cut = refit - timedelta(days=HOLDOUT_DAYS)
    fit_b_rows = all_rows.filter(pl.col("valid_time") <= cut)
    valid_rows = all_rows.filter(pl.col("valid_time") > cut)
    rounds_a = 20 if smoke else MAX_ROUNDS
    fixed = (5, 10, 20) if smoke else FIXED_ROUNDS
    lines = []
    for start in STARTS:
        started = time.monotonic()
        parameters = {
            **common.QUANTILE_PARAMETERS,
            "seed": common.SEED,
            "nthread": nthread,
            **({"base_score": 0.0} if start == "m0" else {}),
        }
        targets = [
            fitting.job_path(root=root, config=f"{family}_{start}_{label}", site=site, refit=refit)
            for family, label in [
                *[("frozen", f"r{n}") for n in fixed],
                ("frozenes", "best"),
                ("frozenes", f"r{fixed[0]}"),
            ]
        ]
        if all(target.exists() for target in targets):
            lines.append(f"{site} {start}: all files exist")
            continue
        half_life = config.half_life_days
        models: dict[str, tuple[xgb.Booster, tuple[tuple[str, int], ...]]] = {}
        # Fit A: the whole window, fixed prefixes.
        booster_a = xgb.train(
            parameters,
            fitting.train_matrix(rows=all_rows, row=row, refit=refit, half_life_days=half_life),
            num_boost_round=rounds_a,
        )
        models["frozen"] = (booster_a, tuple((f"r{n}", n) for n in fixed))
        # Fit B: the window without its last 90 days, early stopping on those days.
        history: dict = {}
        train_b = fitting.train_matrix(
            rows=fit_b_rows, row=row, refit=refit, half_life_days=half_life
        )
        valid_b = fitting.train_matrix(
            rows=valid_rows, row=row, refit=refit, half_life_days=half_life
        )
        booster_b = xgb.train(
            parameters,
            train_b,
            num_boost_round=rounds_a,
            evals=[(train_b, "train"), (valid_b, "valid")],
            evals_result=history,
            early_stopping_rounds=PATIENCE,
            verbose_eval=False,
        )
        best = int(booster_b.best_iteration) + 1
        models["frozenes"] = (booster_b, (("best", best), (f"r{fixed[0]}", fixed[0])))
        common.atomic_write(
            frame=curve_frame(
                history=history, group=row["substation_type"], site=site, start=start
            ),
            target=(CURVES_DIR / ("smoke" if smoke else "") / f"{site}_{start}.parquet"),
        )
        for family, (booster, prefixes) in models.items():
            wanted = {
                (family, label): n
                for label, n in prefixes
                if n <= booster.num_boosted_rounds()
                and not fitting.job_path(
                    root=root, config=f"{family}_{start}_{label}", site=site, refit=refit
                ).exists()
            }
            if not wanted:
                continue
            pieces: dict[tuple[str, str], list[pl.DataFrame]] = {key: [] for key in wanted}
            for origin in origins:
                frame = pl.read_parquet(
                    common.demand_frame_path(run=origin - common.PUBLICATION_DELAY)
                ).filter(pl.col("time_series_id") == row["time_series_id"])
                predicted = fitting.predict_run(
                    model=booster,
                    run_rows=frame,
                    row=row,
                    arrays=arrays,
                    prefixes=tuple(wanted.values()),
                )
                for key, n in wanted.items():
                    pieces[key].append(predicted[n])
            for (fam, label), frames in pieces.items():
                common.atomic_write(
                    frame=pl.concat(frames).with_columns(seed=pl.lit(common.SEED).cast(pl.Int8)),
                    target=fitting.job_path(
                        root=root, config=f"{fam}_{start}_{label}", site=site, refit=refit
                    ),
                )
        lines.append(f"{site} {start}: best_iteration {best}, {time.monotonic() - started:.0f} s")
    return lines


def generator_job(*, site: str, nthread: int, smoke: bool) -> list[str]:
    """Fit A and fit B for one generator series, both starts, and write their predictions."""
    sites = gen.site_table()
    one = sites.filter(pl.col("site") == site)
    row = one.row(0, named=True)
    technology = row["technology"]
    analysis = pl.read_parquet(gen.ANALYSIS_PATH).filter(
        pl.col("time_series_id") == row["time_series_id"]
    )
    root = common.GENERATOR_PREDS_DIR.with_name(
        "generator_preds_smoke" if smoke else "generator_preds"
    )
    origins = common.origins_of_subset(subset="gap")
    if smoke:
        origins = origins[:3]
    observed = gen.observed_training_power(sites=one, train_end=gen.FROZEN_TRAIN_END)
    rows = gen_fit.training_table(
        row=row, analysis=analysis, observed=observed, train_end=gen.FROZEN_TRAIN_END
    )
    cut = gen.FROZEN_TRAIN_END - timedelta(days=HOLDOUT_DAYS)
    fit_b_rows = rows.filter(pl.col("valid_time") < cut)
    valid_rows = rows.filter(pl.col("valid_time") >= cut)
    rounds_a = 20 if smoke else MAX_ROUNDS
    fixed = (5, 10, 20) if smoke else FIXED_ROUNDS
    lines = []
    for start in STARTS:
        started = time.monotonic()
        zero = start == "m0"
        columns = gen.FEATURES[technology]
        targets = [
            gen_fit.job_path(root=root, arm=f"{family}_{start}_{label}", site=site, label="frozen")
            for family, label in [
                *[("frozen", f"r{n}") for n in fixed],
                ("frozenes", "best"),
                ("frozenes", f"r{fixed[0]}"),
            ]
        ]
        if all(target.exists() for target in targets):
            lines.append(f"{site} {start}: all files exist")
            continue
        model_a = gen.FittedQuantiles(
            rows=rows,
            columns=columns,
            seed=common.SEED,
            rounds=rounds_a,
            use_margin=False,
            nthread=nthread,
            zero_start=zero,
        )
        model_b = gen.FittedQuantiles(
            rows=fit_b_rows,
            columns=columns,
            seed=common.SEED,
            rounds=rounds_a,
            use_margin=False,
            nthread=nthread,
            zero_start=zero,
            valid_rows=valid_rows,
            patience=PATIENCE,
        )
        common.atomic_write(
            frame=curve_frame(history=model_b.history, group=technology, site=site, start=start),
            target=(CURVES_DIR / ("smoke" if smoke else "") / f"{site}_{start}.parquet"),
        )
        families = {
            "frozen": (model_a, tuple((f"r{n}", n) for n in fixed)),
            "frozenes": (model_b, (("best", model_b.best_rounds), (f"r{fixed[0]}", fixed[0]))),
        }
        for family, (model, prefixes) in families.items():
            for label, n in prefixes:
                target = gen_fit.job_path(
                    root=root, arm=f"{family}_{start}_{label}", site=site, label="frozen"
                )
                if target.exists() or n > model.booster.num_boosted_rounds():
                    continue
                pieces = []
                for origin in origins:
                    frame = pl.read_parquet(
                        gen.generator_frame_path(run=origin - common.PUBLICATION_DELAY)
                    ).filter(pl.col("time_series_id") == row["time_series_id"])
                    pieces.append(
                        gen.pooled_mixture(
                            part=frame, member_quantiles=model.predict(frame=frame, n_rounds=n)
                        )
                    )
                common.atomic_write(
                    frame=pl.concat(pieces).with_columns(seed=pl.lit(common.SEED).cast(pl.Int8)),
                    target=target,
                )
        lines.append(
            f"{site} {start}: best_iteration {model_b.best_rounds}, "
            f"{time.monotonic() - started:.0f} s"
        )
    return lines


def main() -> None:
    """Fit the rounds family of the requested groups, one series per worker process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", nargs="+", choices=GROUPS, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--primary-stride",
        type=int,
        default=2,
        help="fit every Nth Primary substation (the 2000-round fits are slow)",
    )
    parser.add_argument("--smoke", action="store_true", help="two series, 20 rounds, 3 runs")
    args = parser.parse_args()
    table = common.series_table().filter(pl.col("substation_type").is_in(args.group))
    primary = table.filter(pl.col("substation_type") == "Primary")["site"].to_list()[
        :: args.primary_stride
    ]
    demand = pl.Series(
        [
            site
            for site in table["site"]
            if table.filter(pl.col("site") == site)["substation_type"][0] != "Primary"
            or site in primary
        ]
    )
    generators = gen.site_table().filter(pl.col("technology").is_in(args.group))["site"]
    jobs = [(demand_job, s) for s in demand] + [(generator_job, s) for s in generators]
    if args.smoke:
        jobs = jobs[:2]
    nthread = max(1, CORES // args.workers)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(job, site=site, nthread=nthread, smoke=args.smoke) for job, site in jobs
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
