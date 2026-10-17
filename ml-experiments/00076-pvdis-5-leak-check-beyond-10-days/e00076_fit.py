"""Fit `am0` and predict `am_train_only`, one job per series and seed, for experiment 00076.

- `am0`: experiment 00064's `am` recipe with a base margin of zero in every training row and at
  prediction. It starts from zero like `am` and carries no solar term.
- `am_train_only`: experiment 00064's stored `am` booster (fitted with the solar margin), predicted
  with a margin of zero at every lead. It needs no fit. At leads of 10 days and beyond the margin
  of `am` is already zero, so its quantiles must equal `am`'s there.

Each job writes `work/arms_raw/<arm>_seed<s>_<site>.parquet`, atomically, with the pooled nine
quantiles of the 51-member mixture for every validation run. Each job uses one thread.

    uv run python ml-experiments/00076-*/e00076_fit.py --arm am0 --seed 0 --site D9
    uv run python ml-experiments/00076-*/e00076_fit.py --arm am_train_only --seed 0 --site D9
"""

import argparse
import time

import e00076_common as common
import e00076_model as model
import numpy as np
import polars as pl
import xgboost as xgb

ARMS = ("am0", "am_train_only")


def obtain_booster(*, arm: str, seed: int, ctx: model.SeriesContext, device: str) -> xgb.Booster:
    """Load experiment 00064's `am` booster for `am_train_only`, or fit and save `am0`."""
    booster = xgb.Booster(params={"nthread": 1, "device": device})
    if arm == "am_train_only":
        path = common.E64_BOOSTER_DIR / f"am_seed{seed}_{ctx.site}.ubj"
        if not path.exists():
            msg = f"no stored am booster at {path}"
            raise FileNotFoundError(msg)
        booster.load_model(path)
        return booster
    path = common.BOOSTER_DIR / f"{arm}_seed{seed}_{ctx.site}.ubj"
    if path.exists():
        booster.load_model(path)
        return booster
    rows = model.training_rows(ctx=ctx)
    latest = common.latest_time(times=rows["valid_time"])
    if latest > common.LAST_TRAINING_VALID:
        msg = f"training rows reach {latest}, after {common.LAST_TRAINING_VALID}"
        raise ValueError(msg)
    matrix = model.training_matrix(
        arm=arm, rows=rows, ctx=ctx, columns=model.d1l_columns(), weights=None
    )
    booster = model.fit_booster(matrix=matrix, seed=seed, n_rounds=common.N_ROUNDS, device=device)
    model.save_booster(booster=booster, path=path)
    return booster


def run_job(*, arm: str, seed: int, site: str, device: str) -> None:
    """Fit (or reload) one booster and write the pooled quantiles of every validation run."""
    target = common.RAW_ARMS_DIR / f"{arm}_seed{seed}_{site}.parquet"
    if target.exists():
        return
    started = time.monotonic()
    ctx = model.load_context(site=site)
    booster = obtain_booster(arm=arm, seed=seed, ctx=ctx, device=device)
    pieces = []
    for run in model.validation_runs(ctx=ctx, weights=None):
        margin = np.zeros((run.keys.height, model.N_QUANTILES), dtype=np.float32)
        members = model.predict_members(
            booster=booster,
            members=run.members,
            columns=model.d1l_columns(),
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
    common.write_atomically(frame=pl.concat(pieces), target=target)
    print(f"{arm} seed {seed} {site}: {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Run one job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--site", choices=common.SITES, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_job(arm=args.arm, seed=args.seed, site=args.site, device=args.device)


if __name__ == "__main__":
    main()
