# ruff: noqa: E501
r"""Stage 3 refits: experiment 305's depth-1 XGBoost twin, refitted in the replicate year.

Copies the additive (depth-1, four times the rounds) variant of experiment 305 and changes only the
refit dates and the rows predicted. A refit at the first of April or June 2025 sees control-member
and ensemble rows whose valid time is at least 3.5 hours before the refit and whose run began at or
before the refit (experiment 305's `known_rows` rule, with the training rows ending 2025-06-30),
and predicts that month's daily 09:00 UTC runs on the control member. Each (site, refit) writes a
booster, the 11 quantiles in points of effective capacity, and the input columns, atomically.

    PYTHONPATH=ml-experiments/00305-summer-gap-what-bg2-knows-that-arm-d-does-not:<this folder> \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00322-*/e00322_refit.py --workers 2 --threads 3
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from typing import Final

import e00305_common as c305  # ty: ignore[unresolved-import]
import e00305_recipe as recipe  # ty: ignore[unresolved-import]
import e00322_common as common
import numpy as np
import polars as pl
import xgboost as xgb

SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D16")
REFITS: Final[tuple[datetime, ...]] = tuple(datetime(2025, m, 1, tzinfo=UTC) for m in (4, 6))
SCALE: Final[float] = 100.0
MODEL_DIR = common.WORK_DIR / "refit" / "models"
PRED_DIR = common.WORK_DIR / "refit" / "preds"


def run_site(*, site: str, threads: int) -> list[str]:
    """Fit and predict both refits of one site, keeping the site's training rows in memory once."""
    series, arrays = recipe._series_and_arrays()  # noqa: SLF001
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    config = recipe.CONFIGS["n1h"]
    rows = recipe.training_rows(row=row, arrays=arrays[site], frozen=True, a7=False)
    messages = []
    for refit in REFITS:
        out = PRED_DIR / f"{site}_{refit:%Y-%m}.parquet"
        if out.exists():
            messages.append(f"{site} {refit:%Y-%m}: exists")
            continue
        known = recipe.known_rows(rows=rows, refit=refit, config=config)
        params = {
            **recipe.parameters(config=config, seed=0, nthread=threads),
            "device": "cpu",
            "max_depth": 1,
        }
        train = recipe.train_inputs(known=known, row=row, refit=refit, config=config)
        model = xgb.train(params, train, num_boost_round=config.rounds * 4)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_DIR / f"{site}_{refit:%Y-%m}.ubj"))
        month = refit.strftime("%Y-%m")
        wanted = rows.filter(
            (pl.col("ensemble_member") == 0)
            & (pl.col("power_fcst_init_time").dt.strftime("%Y-%m") == month)
            & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
        ).sort("power_fcst_init_time", "valid_time")
        columns = recipe.input_columns(a7=False)
        raw = model.predict(xgb.DMatrix(wanted.select(columns).to_numpy().astype(np.float32)))
        quantiles = np.sort(raw, axis=1) * SCALE
        table = wanted.select("power_fcst_init_time", "valid_time", *columns).with_columns(
            site=pl.lit(site),
            *[pl.Series(f"r_q{round(level * 100):02d}", quantiles[:, i]) for i, level in enumerate(c305.LEVELS)],
        )
        c305.atomic_write(frame=table, target=out)
        messages.append(f"{site} {month}: {known.height:,} training rows, {table.height:,} predictions")
    return messages


def main() -> None:
    """Run every site, one worker per site."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--sites", nargs="+", default=list(SITES))
    args = parser.parse_args()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_site, site=s, threads=args.threads) for s in args.sites]
        for future in futures:
            for line in future.result():
                print(line, flush=True)


if __name__ == "__main__":
    main()
