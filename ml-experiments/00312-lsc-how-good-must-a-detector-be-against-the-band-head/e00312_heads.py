# ruff: noqa: PLR0915
"""Fit and score the new quantile heads: band position and perfect-knowledge inputs.

Every head is experiment 297's BAND_HEAD (K0 centre plus 11 fitted residual quantiles against
observed power; XGBoost depth 3, 100 rounds, eta 0.1, seed 297, ordinary rows weigh 4) with extra
inputs. Variants: PIT (band position `pit24`, `ext24`), IND (three perfect-knowledge indicators),
PIT_IND (both), IND_MW (indicators and the added-back megawatts). Modes: `cross` cross-fits on
month parity over all worlds, as experiment 297 did; `held` fits on even-seed worlds and the other
parity and scores odd-seed worlds only; `size` fits on rows near shifts smaller than the median
size (plus ordinary rows) and scores rows near larger shifts (plus ordinary rows), on the other
parity. The base head is refitted under the same modes (`BASE`) so that every contrast compares
like with like.

    PYTHONPATH=... systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00312_heads.py run
"""

from __future__ import annotations

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import e00297_fed as fed
import e00297_pass2 as p2
import e00297_score as score
import numpy as np
import polars as pl
import xgboost as xgb
from e00312_common import EXTRA_DIR, HEADS_DIR

BOOSTERS = HEADS_DIR / "boosters"
"""Cross-fitted BASE and PIT heads, by test parity, applied to the real record."""

N_WORKERS = 4
VARIANT_COLUMNS: dict[str, tuple[str, ...]] = {
    "BASE": (),
    "PIT": ("pit24", "ext24"),
    "IND": ("ind_active", "ind_recent", "ind_target"),
    "PIT_IND": ("pit24", "ext24", "ind_active", "ind_recent", "ind_target"),
    "IND_MW": ("ind_active", "ind_recent", "ind_target", "added_mw"),
}
MODES: dict[str, tuple[str, ...]] = {
    "cross": ("BASE", "PIT", "IND", "PIT_IND", "IND_MW"),
    "held": ("BASE", "PIT", "IND", "PIT_IND", "IND_MW"),
    "size": ("BASE", "PIT", "IND_MW"),
}


def matrix(*, frame: pl.DataFrame, variant: str) -> np.ndarray:
    """Head inputs: experiment 297's six, plus the variant's columns (missing values become 0)."""
    base = p2.head_features(frame)
    extra = [
        np.nan_to_num(frame[c].to_numpy().astype(np.float32)) for c in VARIANT_COLUMNS[variant]
    ]
    return np.column_stack([base, *extra]).astype(np.float32) if extra else base


def fit(*, x: np.ndarray, residual: np.ndarray, weight: np.ndarray) -> xgb.Booster:
    """The head, with experiment 297's settings and its row cap."""
    rng = np.random.default_rng(p2.SEED)
    if len(x) > p2.HEAD_ROWS:
        keep = np.sort(rng.choice(len(x), p2.HEAD_ROWS, replace=False))
        x, residual, weight = x[keep], residual[keep], weight[keep]
    dtrain = xgb.DMatrix(x, label=residual.astype(np.float32), weight=weight)
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": p2.LEVELS,
        "tree_method": "hist",
        "max_depth": 3,
        "eta": 0.1,
        "nthread": 1,
        "seed": p2.SEED,
    }
    return xgb.train(params, dtrain, num_boost_round=p2.HEAD_ROUNDS)


def site_job(site: str) -> str:
    """Fit and score every head variant and mode at one site, then write one file per world."""
    started = time.time()
    data = fed.get_site(site)
    frames = []
    for path in sorted(score.FEATURES.glob(f"{site}_*.parquet")):
        world = int(path.stem.split("_")[1])
        k0 = (
            pl.read_parquet(score.OUT / path.name)
            .filter(pl.col("arm") == "K0")
            .select("row", "event", "onset_month", "actual_nra", "actual_obs", "fraction")
        )
        extra = pl.read_parquet(EXTRA_DIR / path.name).select(
            "row", "added_mw", "pit24", "ext24", "ind_active", "ind_recent", "ind_target"
        )
        frames.append(
            pl.read_parquet(path)
            .join(k0, on="row")
            .join(extra, on="row")
            .with_columns(world=pl.lit(world, dtype=pl.Int32))
        )
    frame = pl.concat(frames)
    ordinary = (frame["event"] < 0).to_numpy()
    fold_month = np.where(ordinary, frame["month"].to_numpy(), frame["onset_month"].to_numpy())
    parity = (fold_month % 2).astype(np.int64)
    weight = np.where(ordinary, p2.ORDINARY_WEIGHT, 1.0)
    k0_mean = frame["k0_mean"].to_numpy().astype(np.float64)
    residual = frame["actual_obs"].to_numpy().astype(np.float64) - k0_mean
    even_world = (frame["world"].to_numpy() % 2) == 0
    fraction = frame["fraction"].to_numpy().astype(np.float64)
    median_size = float(np.nanmedian(fraction))
    small = ordinary | (fraction < median_size)
    large = ordinary | (fraction >= median_size)
    q = {}
    for mode, variants in MODES.items():
        for variant in variants:
            x = matrix(frame=frame, variant=variant)
            out = np.full((len(frame), 11), np.nan)
            for p in (0, 1):
                if mode == "cross":
                    train, test = parity == 1 - p, parity == p
                elif mode == "held":
                    train, test = even_world & (parity == 1 - p), ~even_world & (parity == p)
                else:
                    train, test = small & (parity == 1 - p), large & (parity == p)
                booster = fit(x=x[train], residual=residual[train], weight=weight[train])
                if mode == "cross" and variant in ("BASE", "PIT"):
                    BOOSTERS.mkdir(parents=True, exist_ok=True)
                    booster.save_model(BOOSTERS / f"{site}_{variant}_{p}.ubj")
                idx = np.flatnonzero(test)
                out[idx] = k0_mean[idx][:, None] + booster.predict(xgb.DMatrix(x[idx]))
            q[f"{variant}_{mode}"] = out
        print(f"{site} {mode} fitted, {time.time() - started:.0f} s", flush=True)
    HEADS_DIR.mkdir(parents=True, exist_ok=True)
    edges = score.edges_of(data=data)
    for world in sorted(frame["world"].unique().to_list()):
        sel = (frame["world"] == world).to_numpy()
        sub = frame.filter(pl.Series(sel))
        rows = sub["row"].to_numpy().astype(np.int64)
        delta = np.zeros(len(data.cache.end_times))
        delta[data.position[rows]] = sub["actual_obs"].to_numpy() - sub["actual_nra"].to_numpy()
        arms = {}
        for name, out in q.items():
            part = out[sel]
            have = np.isfinite(part[:, 0])
            if not have.any():
                continue
            arms[f"HEAD_{name}"] = (
                p2.sorted_quantiles(np.where(have[:, None], part, 0.0)),
                k0_mean[sel],
            )
        scored = score.score_arms(arms=arms, rows=rows, data=data, delta_points=delta, edges=edges)
        # rows a mode did not score are marked by NaN quantiles in `scored_mask`
        have_frame = pl.DataFrame(
            {"arm": [], "row": []}, schema={"arm": pl.String, "row": pl.Int64}
        )
        parts = []
        for name, out in q.items():
            part = out[sel]
            have = np.isfinite(part[:, 0])
            parts.append(pl.DataFrame({"arm": f"HEAD_{name}", "row": rows[have]}))
        have_frame = pl.concat(parts)
        scored = scored.join(have_frame, on=["arm", "row"], how="inner")
        keep = [
            c
            for c in scored.columns
            if c in ("arm", "row") or c.startswith(("qs_", "tw95_", "q0", "q5", "q9", "ae_"))
        ]
        tmp = HEADS_DIR / f"{site}_{world}.tmp"
        scored.select(keep).with_columns(
            site=pl.lit(site), world=pl.lit(int(world), dtype=pl.Int32)
        ).with_columns(pl.col(pl.Float64).cast(pl.Float32)).write_parquet(tmp)
        tmp.rename(HEADS_DIR / f"{site}_{world}.parquet")
    print(f"{site} done, {time.time() - started:.0f} s (median size {median_size:.3f})", flush=True)
    return site


def main() -> None:
    """Run the heads."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "one"])
    parser.add_argument("--site", default="D5")
    args = parser.parse_args()
    if args.stage == "one":
        site_job(args.site)
        return
    with ProcessPoolExecutor(N_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(site_job, list(fed.SERIES_OF_SITE)))


if __name__ == "__main__":
    main()
