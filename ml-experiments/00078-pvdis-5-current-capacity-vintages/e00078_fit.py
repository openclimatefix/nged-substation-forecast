"""Predict the vintage arms and fit `d1l_r`, `d0_r`, and `d0_m0`, for experiment 00078.

- `am_sy`, `am_sy_twin`, `am_wk`: experiment 00064's fitted `am` (or `am_twin`) booster, predicted
  with a base margin `w * scale * anomaly` whose `scale` is the run's vintage capacity over the
  series' effective capacity (see `e00078_capacity.py`). No model is refitted.
- `d1l_r`: `d1l` refitted with experiment 00064's recipe and no base margin, so XGBoost estimates
  its own base score.
- `d0_r`: the same with `d0`'s 17 columns (`d1l`'s 15 plus `shortwave` and `longwave`).
- `d0_m0`: `d0_r` with a base margin of zero in every training row and at prediction.

Each job writes `work/arms_raw/<arm>_seed<s>_<site>.parquet`, atomically, with the pooled nine
quantiles of the 51-member mixture for every validation run. A fitted arm also writes
`work/base_scores/<arm>_seed<s>_<site>.json` with XGBoost's base score for each quantile level and
the number of boosting rounds. Each job uses one thread.

    uv run python ml-experiments/00078-*/e00078_fit.py --arm am_sy --seed 0 --site D9
"""

import argparse
import json
import time
from datetime import date

import e00078_common as common
import e00078_model as model
import numpy as np
import polars as pl
import xgboost as xgb

VINTAGE_ARMS = {
    "am_sy": ("am", "season_year_mw", False),
    "am_sy_twin": ("am_twin", "season_year_mw", True),
    "am_wk": ("am", "weekly_mw", False),
}
"""Each predict-only arm: the stored booster's arm, the capacity column, and whether the anomaly is
the placebo twin."""
FITTED_ARMS = ("d1l_r", "d0_r", "d0_m0")
ARMS = (*VINTAGE_ARMS, *FITTED_ARMS)
BASE_SCORE_DIR = common.WORK_DIR / "base_scores"


def run_date_of(*, run: model.RunData) -> date:
    """Return the initialisation date in a run file's name, such as `init=2025-07-01.parquet`."""
    return date.fromisoformat(run.file_name.removesuffix(".parquet").removeprefix("init="))


def fitted_columns(*, arm: str) -> list[str]:
    """Return the input columns of a fitted arm."""
    return model.d1l_columns() if arm == "d1l_r" else model.d0_columns()


def predict_vintage_arm(*, arm: str, seed: int, ctx: model.SeriesContext) -> pl.DataFrame:
    """Predict every validation run of a vintage arm with its stored booster."""
    stored_arm, column, twin = VINTAGE_ARMS[arm]
    path = common.E64_BOOSTER_DIR / f"{stored_arm}_seed{seed}_{ctx.site}.ubj"
    if not path.exists():
        msg = f"no stored booster at {path}"
        raise FileNotFoundError(msg)
    booster = xgb.Booster(params={"nthread": 1, "device": "cpu"})
    booster.load_model(path)
    vintages = pl.read_parquet(common.VINTAGES_FILE).filter(pl.col("site") == ctx.site)
    capacity = dict(zip(vintages["run_date"], vintages[column], strict=True))
    effective = ctx.row["effective_capacity_mw"]
    pieces = []
    for run in model.validation_runs(ctx=ctx, weights=model.read_weights()):
        solar = run.solar
        anomaly = solar.anomaly_twin if twin else solar.anomaly_mean
        margin = model.margin_rows(
            weight=solar.weight, anomaly=anomaly, scale=capacity[run_date_of(run=run)] / effective
        )
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
    return pl.concat(pieces)


def obtain_fitted_booster(
    *, arm: str, seed: int, ctx: model.SeriesContext, device: str
) -> xgb.Booster:
    """Load or fit a fitted arm's booster, and write its base scores."""
    booster = xgb.Booster(params={"nthread": 1, "device": device})
    path = common.BOOSTER_DIR / f"{arm}_seed{seed}_{ctx.site}.ubj"
    if path.exists():
        booster.load_model(path)
        return booster
    rows = model.training_rows(ctx=ctx)
    latest = common.latest_time(times=rows["valid_time"])
    if latest > common.LAST_TRAINING_VALID:
        msg = f"training rows reach {latest}, after {common.LAST_TRAINING_VALID}"
        raise ValueError(msg)
    base = {"d1l_r": "d1l", "d0_r": "d0", "d0_m0": "d0_m0"}[arm]
    matrix = model.training_matrix(
        arm=base, rows=rows, ctx=ctx, columns=fitted_columns(arm=arm), weights=None
    )
    booster = model.fit_booster(matrix=matrix, seed=seed, n_rounds=common.N_ROUNDS, device=device)
    model.save_booster(booster=booster, path=path)
    parameters = json.loads(booster.save_config())["learner"]["learner_model_param"]
    BASE_SCORE_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_SCORE_DIR / f"{arm}_seed{seed}_{ctx.site}.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "seed": seed,
                "site": ctx.site,
                "base_score": parameters["base_score"],
                "boosted_rounds": booster.num_boosted_rounds(),
            }
        )
    )
    return booster


def predict_fitted_arm(
    *, arm: str, seed: int, ctx: model.SeriesContext, booster: xgb.Booster
) -> pl.DataFrame:
    """Predict every validation run of a fitted arm (a zero margin for `d0_m0`)."""
    pieces = []
    for run in model.validation_runs(ctx=ctx, weights=None):
        margin = (
            np.zeros((run.keys.height, model.N_QUANTILES), dtype=np.float32)
            if arm == "d0_m0"
            else None
        )
        members = model.predict_members(
            booster=booster,
            members=run.members,
            columns=fitted_columns(arm=arm),
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


def run_job(*, arm: str, seed: int, site: str, device: str) -> None:
    """Predict or fit one arm for a series and seed, and write its raw file."""
    target = common.RAW_ARMS_DIR / f"{arm}_seed{seed}_{site}.parquet"
    if target.exists():
        return
    started = time.monotonic()
    ctx = model.load_context(site=site)
    if arm in VINTAGE_ARMS:
        table = predict_vintage_arm(arm=arm, seed=seed, ctx=ctx)
    else:
        booster = obtain_fitted_booster(arm=arm, seed=seed, ctx=ctx, device=device)
        table = predict_fitted_arm(arm=arm, seed=seed, ctx=ctx, booster=booster)
    common.write_atomically(frame=table, target=target)
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
