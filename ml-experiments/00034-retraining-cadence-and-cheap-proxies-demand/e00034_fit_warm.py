"""Fit the warm-start proxy (P-warm) of experiment 00034.

Each series starts from its frozen booster (the 2025-07-01 refit of `e00034_fit.py`, config
`n1h`) and, at every update time, adds `k` boosting rounds fitted on the rows that became usable
since the previous update (valid time in `(previous update - 3.5 h, update - 3.5 h]`, run begun at
or before the update), weighted by the 0.5-year half-life anchored at the update. The updated
booster forecasts the validation runs its regime assigns to it. Updates chain: each starts from the
booster the previous one produced.

Update times:

- `monthly`: the first of each month from 2025-08-01.
- `weekly`: every 7 days from 2025-07-08.

Output: `work/anchor_preds/warm<k>_<regime>/`, in the format of `e00034_fit.py`.

    uv run python ml-experiments/00034-*/e00034_fit_warm.py --regime monthly --rounds 20 50 \
        --groups Primary --seeds 0 [--smoke]
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import e00034_common as common
import e00034_fit as fitting
import e00034_regimes as regimes
import polars as pl
import xgboost as xgb


def update_times(*, regime: str) -> list[datetime]:
    """Return the update times of a regime's chain, in order."""
    if regime == "monthly":
        return list(regimes.MONTHLY[1:])
    if regime == "weekly":
        return list(regimes.WEEKLY[1:])
    msg = f"unknown regime {regime}"
    raise ValueError(msg)


def served_origins(*, regime: str, origins: list[datetime]) -> dict[datetime, list[datetime]]:
    """Return each update time and the validation origins it forecasts.

    The monthly and weekly chains forecast the origins their refit times serve in the cold
    regimes.
    """
    return regimes.served(name=regime, origins=origins)


def write_atomic(*, frame: pl.DataFrame, seed: int, target: Path) -> None:
    """Write a refit's predictions through a temporary file."""
    tmp = target.with_suffix(".tmp")
    frame.with_columns(seed=pl.lit(seed).cast(pl.Int8)).write_parquet(tmp)
    tmp.rename(target)


def main() -> None:
    """Run the warm-start chains and write their predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("monthly", "weekly"), required=True)
    parser.add_argument("--rounds", type=int, nargs="+", default=[20])
    parser.add_argument("--groups", nargs="+", default=["Primary"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--smoke", action="store_true", help="two series, scratch folders")
    args = parser.parse_args()
    config = fitting.CONFIGS["n1h"]
    series = common.series_table().filter(pl.col("substation_type").is_in(args.groups))
    if args.smoke:
        series = series.head(2)
    origins = regimes.run_origins()
    serve = served_origins(regime=args.regime, origins=origins)
    updates = update_times(regime=args.regime)
    arrays = common.load_series_arrays(series=series)
    suffix = "_smoke" if args.smoke else ""
    boosters_dir = fitting.BOOSTERS_DIR.with_name(f"boosters{suffix}")
    out_root = fitting.ANCHOR_PREDS_DIR.with_name(f"anchor_preds{suffix}")
    for row in series.iter_rows(named=True):
        site = arrays[row["site"]]
        rows = fitting.training_rows(row=row, arrays=site)
        prepared = fitting.prepared_runs(
            row=row, arrays=site, origins={o for s in serve.values() for o in s}
        )
        for seed in args.seeds:
            for k in args.rounds:
                name = f"warm{k}_{args.regime}"
                started = time.monotonic()
                booster = xgb.Booster()
                booster.load_model(boosters_dir / f"{row['site']}_s{seed}_frozen.ubj")
                previous = regimes.MONTHLY[0]
                first = regimes.MONTHLY[0]
                if first in serve:
                    target = fitting.job_path(
                        root=out_root, config=name, site=row["site"], seed=seed, refit=first
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        frame = fitting.predict_runs(
                            model=booster, prepared=prepared, origins=serve[first]
                        )
                        write_atomic(frame=frame, seed=seed, target=target)
                for update in updates:
                    new = rows.filter(
                        (pl.col("valid_time") > previous - regimes.REFIT_CUTOFF_AGE)
                        & (pl.col("valid_time") <= update - regimes.REFIT_CUTOFF_AGE)
                        & (pl.col("power_fcst_init_time") <= update)
                    )
                    previous = update
                    if new.is_empty():
                        continue
                    booster = xgb.train(
                        {**common.QUANTILE_PARAMETERS, "seed": seed, "eta": config.eta},
                        fitting.train_matrix(
                            rows=new, row=row, refit=update, half_life_days=config.half_life_days
                        ),
                        num_boost_round=k,
                        xgb_model=booster,
                    )
                    if update not in serve:
                        continue
                    target = fitting.job_path(
                        root=out_root, config=name, site=row["site"], seed=seed, refit=update
                    )
                    frame = fitting.predict_runs(
                        model=booster, prepared=prepared, origins=serve[update]
                    )
                    write_atomic(frame=frame, seed=seed, target=target)
                print(
                    f"{name} {row['site']} seed {seed}: {len(updates)} updates, "
                    f"{time.monotonic() - started:.0f} s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
