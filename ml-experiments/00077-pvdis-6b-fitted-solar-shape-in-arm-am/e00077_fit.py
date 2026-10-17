"""Fit arm `am` (and its placebo twin) on one solar shape, one job per series and seed.

The solar shape is the one named by the `E77_ARM` environment variable. Each job writes
`work/arm/<shape>/arms_raw/<arm>_seed<s>_<site>.parquet`, atomically, holding the pooled nine
quantiles of the 51-member mixture for every validation run (variant `base`):

- arm `am`: XGBoost given the base margin `w * scale * anomaly`, with the anomaly of the shape;
- arm `am_twin`: the same with the forecast sunshine of another date (experiment 00064's placebo).

A fit also saves its booster to `work/arm/<shape>/boosters/<arm>_seed<s>_<site>.ubj` and reloads
it when the raw file is missing. Each job uses one thread (`QUANTILE_PARAMETERS`), so four run side
by side (see `e00077_run.sh`). The fit reads only rows up to the last training valid time.

    E77_ARM=blend uv run python ml-experiments/00077-*/e00077_fit.py --arm am --seed 0 --site D9
"""

import argparse
import time
from typing import Final, Literal

import e00077_common as common
import e00077_model as model
import polars as pl
import xgboost as xgb

ArmType = Literal["am", "am_twin"]
ARMS: Final[tuple[ArmType, ...]] = ("am", "am_twin")


def predict_arm(
    *,
    arm: ArmType,
    seed: int,
    ctx: model.SeriesContext,
    booster: xgb.Booster,
    weights: pl.DataFrame,
) -> pl.DataFrame:
    """Predict every validation run of one series and arm.

    Args:
        arm: The arm.
        seed: The fitting seed, stored in the `seed` column.
        ctx: The series' context.
        booster: The fitted model.
        weights: `model.read_weights()`.

    Returns:
        Columns `KEYS`, `variant`, `seed`, and the nine quantiles.
    """
    columns = model.d1l_columns()
    pieces = []
    for run in model.validation_runs(ctx=ctx, weights=weights):
        solar = run.solar
        anomaly = solar.anomaly_twin if arm == "am_twin" else solar.anomaly_mean
        margin = model.margin_rows(weight=solar.weight, anomaly=anomaly, scale=ctx.scale)
        members = model.predict_members(
            booster=booster,
            members=run.members,
            columns=columns,
            margin=margin,
            n_members=run.n_members,
        )
        pieces.append(
            model.variant_frame(
                keys=run.keys,
                quantiles=common.mixture_quantiles(member_quantiles=members),
                variant="base",
                seed=seed,
            )
        )
    return pl.concat(pieces)


def obtain_booster(
    *,
    arm: ArmType,
    seed: int,
    ctx: model.SeriesContext,
    weights: pl.DataFrame,
    n_rounds: int,
    device: str,
) -> xgb.Booster:
    """Load an arm's booster, or fit and save it."""
    path = common.BOOSTER_DIR / f"{arm}_seed{seed}_{ctx.site}.ubj"
    if path.exists():
        booster = xgb.Booster(params={"nthread": 1, "device": device})
        booster.load_model(path)
        return booster
    rows = model.training_rows(ctx=ctx)
    matrix = model.training_matrix(
        arm=arm, rows=rows, ctx=ctx, columns=model.d1l_columns(), weights=weights
    )
    booster = model.fit_booster(matrix=matrix, seed=seed, n_rounds=n_rounds, device=device)
    model.save_booster(booster=booster, path=path)
    return booster


def run_job(*, arm: ArmType, seed: int, site: str, n_rounds: int, device: str) -> None:
    """Fit (or reload) one arm's booster for a series and seed, and write its raw file."""
    target = common.RAW_ARMS_DIR / f"{arm}_seed{seed}_{site}.parquet"
    if target.exists():
        return
    started = time.monotonic()
    ctx = model.load_context(site=site)
    weights = model.read_weights()
    booster = obtain_booster(
        arm=arm, seed=seed, ctx=ctx, weights=weights, n_rounds=n_rounds, device=device
    )
    table = predict_arm(arm=arm, seed=seed, ctx=ctx, booster=booster, weights=weights)
    common.write_atomically(frame=table, target=target)
    print(
        f"{common.SHAPE_ARM} {arm} seed {seed} {site}: {time.monotonic() - started:.0f} s",
        flush=True,
    )


def main() -> None:
    """Run one job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--site", choices=common.SITES, required=True)
    parser.add_argument("--rounds", type=int, default=common.N_ROUNDS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_job(arm=args.arm, seed=args.seed, site=args.site, n_rounds=args.rounds, device=args.device)


if __name__ == "__main__":
    main()
