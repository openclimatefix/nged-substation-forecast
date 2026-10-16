"""Fit the by-type pooled arm of experiment 00024 (arm `type_id`) under monthly refits.

One model per metered-units pool of Primary substations (MW and MVA, 8 series each), with the
series' integer code as a categorical column, trained at each monthly refit on the pool's
control-member rows under the rules of `e00034_fit.py` (cutoff, recency weights, configuration).
Each refit forecasts, for every series of the pool, the validation runs that the monthly-family
regimes assign to it. Output: `work/anchor_preds/type_id_<config>/`, one file per series, seed,
and refit, in the format of `e00034_fit.py`.

    uv run python ml-experiments/00034-*/e00034_fit_pool.py --config n1h --seeds 0 [--smoke]
"""

import argparse
import time

import e00034_common as common
import e00034_fit as fitting
import e00034_regimes as regimes
import numpy as np
import polars as pl
import xgboost as xgb


def identifier_matrix(*, matrix: np.ndarray, code: int) -> xgb.DMatrix:
    """Return a DMatrix with the series code appended as a categorical column."""
    column = np.full((matrix.shape[0], 1), code, dtype=np.float32)
    return xgb.DMatrix(
        np.hstack([matrix, column]),
        feature_types=["q"] * matrix.shape[1] + ["c"],
        enable_categorical=True,
    )


def main() -> None:
    """Fit and write the pooled arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("n1h", "roll12"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--smoke", action="store_true", help="four series, two refits, scratch dir")
    args = parser.parse_args()
    config = fitting.CONFIGS[args.config]
    everyone = common.series_table()
    codes = {site: position for position, site in enumerate(everyone["site"].to_list())}
    primary = everyone.filter(pl.col("substation_type") == "Primary")
    if args.smoke:
        primary = primary.filter(pl.col("site").is_in(["D5", "D9", "D2", "D3"]))
    origins = regimes.run_origins()
    jobs = regimes.monthly_family_jobs(origins=origins)
    if args.smoke:
        jobs = dict(list(jobs.items())[:2])
    arrays = common.load_series_arrays(series=primary)
    needed = {o for served in jobs.values() for o in served}
    folder = f"type_id_{config.name}"
    out_root = fitting.ANCHOR_PREDS_DIR.with_name(
        "anchor_preds_smoke" if args.smoke else "anchor_preds"
    )
    rows = {}
    prepared = {}
    for row in primary.iter_rows(named=True):
        rows[row["site"]] = fitting.training_rows(row=row, arrays=arrays[row["site"]])
        prepared[row["site"]] = fitting.prepared_runs(
            row=row, arrays=arrays[row["site"]], origins=needed
        )
        print(f"prepared {row['site']}", flush=True)
    for units in ("MW", "MVA"):
        pool = primary.filter(pl.col("units") == units)
        for seed in args.seeds:
            for refit in jobs:
                targets = {
                    r["site"]: fitting.job_path(
                        root=out_root, config=folder, site=r["site"], seed=seed, refit=refit
                    )
                    for r in pool.iter_rows(named=True)
                }
                if all(p.exists() for p in targets.values()):
                    continue
                started = time.monotonic()
                parts = []
                for r in pool.iter_rows(named=True):
                    known = fitting.known_rows(
                        rows=rows[r["site"]], refit=refit, window_days=config.window_days
                    )
                    age_days = (refit - known["valid_time"]).dt.total_seconds() / 86_400
                    matrix = known.select(fitting.input_columns()).to_numpy().astype(np.float32)
                    parts.append(
                        (
                            np.hstack(
                                [
                                    matrix,
                                    np.full((matrix.shape[0], 1), codes[r["site"]], np.float32),
                                ]
                            ),
                            (known["power"] / r["effective_capacity_mw"]).to_numpy(),
                            0.5 ** (age_days.to_numpy() / config.half_life_days),
                        )
                    )
                width = parts[0][0].shape[1]
                model = xgb.train(
                    {**common.QUANTILE_PARAMETERS, "seed": seed, "eta": config.eta},
                    xgb.DMatrix(
                        np.vstack([p[0] for p in parts]),
                        label=np.concatenate([p[1] for p in parts]),
                        weight=np.concatenate([p[2] for p in parts]),
                        feature_types=["q"] * (width - 1) + ["c"],
                        enable_categorical=True,
                    ),
                    num_boost_round=config.rounds,
                )
                for r in pool.iter_rows(named=True):
                    pieces = []
                    for origin in jobs[refit]:
                        keys, matrix, members = prepared[r["site"]][origin]
                        raw = model.predict(
                            identifier_matrix(matrix=matrix, code=codes[r["site"]])
                        ).reshape(-1, members, len(common.LEVELS))
                        pieces.append(
                            keys.hstack(
                                pl.DataFrame(
                                    fitting.mixture_quantiles(member_quantiles=raw),
                                    schema=list(common.LEVEL_COLUMNS),
                                )
                            )
                        )
                    target = targets[r["site"]]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    frame = pl.concat(pieces).with_columns(seed=pl.lit(seed).cast(pl.Int8))
                    frame.write_parquet(target.with_suffix(".tmp"))
                    target.with_suffix(".tmp").rename(target)
                print(
                    f"{folder} {units} seed {seed} refit {refit:%Y-%m-%d}: "
                    f"{time.monotonic() - started:.0f} s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
