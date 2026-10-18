"""Block B of experiment 00094: experiment 00064's arm `am` with a heating term added to its margin.

Arm `amh` is experiment 00064's `am` (XGBoost's base margin is the fitted solar term, with the
`d1l` recipe: half-life 2 years, 500 rounds, `eta` 0.05, whole history to 2025-06-30, no refits),
with the heating term of `e00094_fit.py` added to the margin: `b_h * H`, with `H` the degrees the
row's ENS 2 m temperature is below 15.5 C and `b_h` the hourly coefficient fitted on the training
rows. The margin sums the solar and the heating terms. The model is trained once, on rows to
2025-06-30, and forecasts the 73 runs of `fold_a_every5d_v1` for the five series that have a solar
term. One job is one series and seed; it writes
`work/b_arms_raw/amh_seed<s>_<site>.parquet`, atomically, and skips an existing file.

    uv run python ml-experiments/00094-*/e00094b_fit.py --seed 0 --site D9 [--smoke]
"""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00094_fit as heating
import e00094b_common as common
import e00094b_model as model
import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import ML_EXPERIMENTS_DATA_DIR

B_WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00094-explicit-heating-offset-primary-demand" / "work"
)
RAW_DIR: Final[Path] = B_WORK_DIR / "b_arms_raw"
COEFFICIENT_DIR: Final[Path] = B_WORK_DIR / "b_coefficients"
ANCHOR: Final[datetime] = datetime(2025, 6, 30, 23, 30, tzinfo=UTC)
SMOKE_ROUNDS: Final[int] = 20
SMOKE_RUNS: Final[int] = 2


def training_coefficients(*, rows: pl.DataFrame, ctx: model.SeriesContext) -> np.ndarray:
    """Return the series' 24 hourly heating coefficients, fitted on its training rows."""
    prepared = rows.with_columns(
        power_over_capacity=pl.col("power") / ctx.row["effective_capacity_mw"]
    )
    return heating.heating_coefficients(
        rows=prepared, refit=ANCHOR, half_life_days=common.RECENCY_HALF_LIFE_DAYS
    )


def fit_amh(
    *, ctx: model.SeriesContext, seed: int, rounds: int, device: str
) -> tuple[xgb.Booster, np.ndarray]:
    """Fit `amh` for one series and seed and return the booster and the hourly coefficients."""
    weights = model.read_weights()
    rows = model.training_rows(ctx=ctx)
    matrix = model.training_matrix(
        arm="am", rows=rows, ctx=ctx, columns=model.d1l_columns(), weights=weights
    )
    coefficients = training_coefficients(rows=rows, ctx=ctx)
    solar = matrix.get_base_margin().reshape(rows.height, model.N_QUANTILES)
    matrix.set_base_margin(solar + heating.heating_margin(frame=rows, coefficients=coefficients))
    booster = model.fit_booster(matrix=matrix, seed=seed, n_rounds=rounds, device=device)
    return booster, coefficients


def predict_runs(
    *,
    booster: xgb.Booster,
    coefficients: np.ndarray,
    ctx: model.SeriesContext,
    seed: int,
    max_runs: int | None,
) -> pl.DataFrame:
    """Predict every validation run: pooled quantiles of the 51 members, one margin per member."""
    weights = model.read_weights()
    columns = model.d1l_columns()
    pieces = []
    for run in model.validation_runs(ctx=ctx, weights=weights, max_runs=max_runs):
        solar = np.repeat(
            model.margin_rows(
                weight=run.solar.weight, anomaly=run.solar.anomaly_mean, scale=ctx.scale
            ),
            run.n_members,
            axis=0,
        )
        margin = solar + heating.heating_margin(frame=run.members, coefficients=coefficients)
        matrix = xgb.DMatrix(
            run.members.select(columns).to_numpy().astype(np.float32), base_margin=margin
        )
        raw = booster.predict(matrix).reshape(-1, run.n_members, model.N_QUANTILES)
        pieces.append(
            model.variant_frame(
                keys=run.keys,
                quantiles=common.mixture_quantiles(member_quantiles=raw.astype(np.float32)),
                variant="base",
                seed=seed,
            )
        )
    return pl.concat(pieces)


def main() -> None:
    """Fit one series and seed and write the raw predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--site", choices=common.SITES, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="20 rounds, 2 runs, nothing written")
    args = parser.parse_args()
    target = RAW_DIR / f"amh_seed{args.seed}_{args.site}.parquet"
    if target.exists() and not args.smoke:
        return
    started = time.monotonic()
    ctx = model.load_context(site=args.site)
    rounds = SMOKE_ROUNDS if args.smoke else common.N_ROUNDS
    booster, coefficients = fit_amh(ctx=ctx, seed=args.seed, rounds=rounds, device=args.device)
    table = predict_runs(
        booster=booster,
        coefficients=coefficients,
        ctx=ctx,
        seed=args.seed,
        max_runs=SMOKE_RUNS if args.smoke else None,
    )
    print(
        f"{args.site} seed {args.seed}: {table.height:,} rows, coefficients "
        f"{coefficients.min():.4f} to {coefficients.max():.4f}, "
        f"{time.monotonic() - started:.0f} s",
        flush=True,
    )
    if args.smoke:
        return
    common.write_atomically(frame=table, target=target)
    common.write_atomically(
        frame=pl.DataFrame({"hour": list(range(24)), "coefficient": coefficients.tolist()}),
        target=COEFFICIENT_DIR / f"seed{args.seed}_{args.site}.parquet",
    )


if __name__ == "__main__":
    main()
