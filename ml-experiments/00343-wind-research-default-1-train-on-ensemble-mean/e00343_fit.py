"""Fit experiment 00343's arms for one farm and store their quantiles.

Quantiles are stored in `work/mix/<arm>_s<seed>_<hindcast|scored>_<farm>.npy`, shape (cases, 11),
before the spread factor, in experiment 00303's layout (one row per case, so there is no member
mixture). Arms are described in `e00343_common.FIT_ARMS`. The hindcast quantiles are out-of-fold
(three blocks of consecutive run dates, `Hindcast.split`), so the spread factor and the offset
table never see an in-sample prediction. The scored quantiles come from the fit on all usable
hindcast cases.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      env PYTHONPATH=<the 00303, 00326, 00336, and 00340 folders, colon separated> \
      uv run --no-sync python ml-experiments/00343-*/e00343_fit.py --farms W1 --nthread 4
"""

import argparse
import os
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_ens as ens  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_fit as fit303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_models as models  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00343_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402

SPEED_100M = 1  # column index of the 100 m speed in `case_features`


def fit_and_predict(
    *,
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    predict_sets: list[np.ndarray],
    seed: int,
    nthread: int,
) -> list[np.ndarray]:
    """Fit the quantile model on cases `train` and return sorted quantiles for each feature set."""
    centres, curve = models.binned_curve(speed=features[train, SPEED_100M], target=target[train])
    base = np.interp(features[train, SPEED_100M], centres, curve)
    params = {**models.XGBOOST_PARAMETERS, "seed": seed, "device": "cpu", "nthread": nthread}
    booster = xgb.train(
        params,
        xgb.DMatrix(
            features[train], label=target[train], base_margin=np.tile(base[:, None], (1, 11))
        ),
        num_boost_round=models.N_ROUNDS,
    )
    out = []
    for x in predict_sets:
        margin = np.tile(np.interp(x[:, SPEED_100M], centres, curve)[:, None], (1, 11))
        predicted = booster.predict(xgb.DMatrix(x, base_margin=margin))
        out.append(np.sort(np.clip(predicted, 0.0, 1.0), axis=1).astype(np.float32))
    return out


def run_farm(*, farm: str, arms: list[str], nthread: int) -> None:
    """Fit and store the requested arms for one farm."""
    data = fit303.Hindcast(farm=farm)
    common.MIX_DIR.mkdir(parents=True, exist_ok=True)
    actual = np.nan_to_num(data.hind_actual, nan=0.0).astype(np.float32)
    block = data.blocks()
    for arm in arms:
        with_spread = arm == "MS"
        hind_x = common.case_features(columns=data.hind_columns, with_spread=with_spread)
        scored_x = common.case_features(columns=data.scored_columns, with_spread=with_spread)
        for seed in common.SEEDS:
            paths = {
                name: common.MIX_DIR / f"{arm}_s{seed}_{name}_{farm}.npy"
                for name in ("hindcast", "scored")
            }
            if all(p.exists() for p in paths.values()):
                continue
            started = time.time()
            target = actual
            if arm == "Mpl":
                source = common.month_permutation(valid=data.valid, seed=common.PLACEBO_SEED + seed)
                target = actual[source]
            out_of_fold = np.empty((data.hind_cases, 11), dtype=np.float32)
            scored = None
            for held_out in (*range(common.FOLDS), None):
                training, validation = data.split(held_out=held_out)
                rows = np.concatenate([training, validation])
                if held_out is None:
                    (scored,) = fit_and_predict(
                        features=hind_x,
                        target=target,
                        train=rows,
                        predict_sets=[scored_x],
                        seed=seed,
                        nthread=nthread,
                    )
                    break
                cases = np.flatnonzero(block == held_out)
                lo, hi = data.init[cases].min(), data.valid[cases].max()
                if ((data.valid[rows] >= lo) & (data.valid[rows] <= hi)).any():
                    msg = f"{farm} fold {held_out}: training valid times overlap the held-out runs"
                    raise RuntimeError(msg)
                (out_of_fold[cases],) = fit_and_predict(
                    features=hind_x,
                    target=target,
                    train=rows,
                    predict_sets=[hind_x[cases]],
                    seed=seed,
                    nthread=nthread,
                )
            ens.write_atomic(path=paths["hindcast"], array=out_of_fold)
            ens.write_atomic(path=paths["scored"], array=scored)
            print(f"{farm} {arm} seed {seed}: {time.time() - started:.0f} s", flush=True)


def main() -> None:
    """Fit the requested arms for the requested farms."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default=",".join(common.FARMS))
    parser.add_argument("--arms", default=",".join(common.FIT_ARMS))
    parser.add_argument("--nthread", type=int, default=4)
    args = parser.parse_args()
    for farm in args.farms.split(","):
        run_farm(farm=farm, arms=args.arms.split(","), nthread=args.nthread)
    _ = pl  # polars is imported before xgboost workers start, as in experiment 00303


if __name__ == "__main__":
    main()
