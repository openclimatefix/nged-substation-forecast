"""Fit the GAM-stage network with cross-fitting and write its corrections.

The pooled rows are every half-hour of the six sites' residual streams before the scored year
(`e00222_inputs.site_rows`). Four cross-fit networks each leave out one calendar-month fold, and
one full network uses every pre-scored fold. A stream row or forecast row before 2025-07-01 is
corrected by the network that did not train on its month; a row from 2025-07-01 onward is
corrected by the full network. Each network is trained for three seeds and the corrections are
averaged. Writes `work/nn/<site>.npz` (the correction on the stream grid) and
`work/nn/<site>_rows.parquet` (the correction on experiment 136's arm A2 forecast rows).

    uv run --no-sync python ml-experiments/00222-*/e00222_fit.py [--threads 4]
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00222_common as common
import e00222_inputs as inputs
import e00222_net as net
import e00222_stream as stream_module
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

NN_DIR: Final[Path] = common.WORK_DIR / f"nn{common.VARIANT}"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
BLOCKED: Final[bool] = common.VARIANT == "_blocked"
"""Post-hoc variant: one network trained on the stream before 2024-07-01, stopped on the tuning and
replicate year, with no site indicator, and applied to every row from 2024-04 on."""
BLOCK_SPLIT: Final[np.datetime64] = np.datetime64("2024-07-01T00:00:00", "us")


@dataclass(frozen=True)
class Pooled:
    """The pooled training rows and the per-site row sets the corrections are applied to.

    Attributes:
        x: The stream rows' inputs, shape (rows, `N_INPUTS`).
        y: The stream rows' residual (observed minus convex forecast), in points.
        fold: The stream rows' cross-fit fold.
        pre_scored: Whether each stream row is before the scored year.
        switching: Whether each stream row lies in a hand-filtered switching event.
        site: The site index of each stream row.
        grid_index: The stream row's position on its site's cache grid.
        temp: The stream rows' driver temperature.
        g: The stream rows' convex forecast.
        times: The stream rows' UTC times.
        forecast: Each site's arm A2 forecast rows with the inputs the network reads.
        n_grid: The length of each site's cache grid.
    """

    x: np.ndarray
    y: np.ndarray
    fold: np.ndarray
    pre_scored: np.ndarray
    switching: np.ndarray
    site: np.ndarray
    grid_index: np.ndarray
    temp: np.ndarray
    g: np.ndarray
    times: np.ndarray
    forecast: dict[str, tuple[pl.DataFrame, np.ndarray]]
    n_grid: dict[str, int]


def forecast_rows(*, site: str, site_index: int) -> tuple[pl.DataFrame, np.ndarray]:
    """Return a site's arm A2 forecast rows and their input matrix."""
    base = pl.concat(
        [
            pl.read_parquet(common.E136_WORK_DIR / "dp_preds" / f"a2_{period}_{site}.parquet")
            for period in PERIODS
        ]
    )
    features = pl.read_parquet(
        common.E201_WORK_DIR / "features" / f"{site}.parquet",
        columns=["run", "valid_time", "drv_temp_c"],
    )
    frame = base.join(features, on=["run", "valid_time"], how="left")
    if frame.height != base.height or frame["drv_temp_c"].null_count() > 0:
        msg = f"{site}: the forecast rows do not match experiment 201's feature rows"
        raise ValueError(msg)
    g = (frame["mean"] - frame["level"] - frame["anomaly"]).to_numpy().astype(np.float64)
    times = frame["valid_time"].dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
    x = inputs.design_matrix(
        g=g,
        temp=frame["drv_temp_c"].to_numpy().astype(np.float64),
        times=times,
        site_index=site_index,
    )
    if BLOCKED:
        x[:, -len(inputs.SITES) :] = 0.0
    return frame, x


def load_pooled() -> Pooled:
    """Build the pooled stream rows and every site's forecast rows."""
    cache = common.load_cache()
    xs, ys, folds, pres, swit, sites, gidx, temps, gs, times = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    forecast: dict[str, tuple[pl.DataFrame, np.ndarray]] = {}
    n_grid: dict[str, int] = {}
    for index, site in enumerate(inputs.SITES):
        site_stream = stream_module.load_site_stream(site=site, cache=cache)
        rows = inputs.site_rows(site=site, site_stream=site_stream)
        take = np.flatnonzero(np.isfinite(rows.stream) & np.isfinite(rows.g))
        t = rows.times[take]
        xs.append(
            inputs.design_matrix(g=rows.g[take], temp=rows.temp[take], times=t, site_index=index)
        )
        ys.append(rows.stream[take])
        folds.append(inputs.fold_of(times=t))
        pres.append(t < inputs.SCORED_START)
        swit.append(rows.switching[take])
        sites.append(np.full(len(take), index))
        gidx.append(take)
        temps.append(rows.temp[take])
        gs.append(rows.g[take])
        times.append(t)
        forecast[site] = forecast_rows(site=site, site_index=index)
        n_grid[site] = len(rows.times)
    if BLOCKED:
        for part in xs:
            part[:, -len(inputs.SITES) :] = 0.0
    return Pooled(
        x=np.concatenate(xs),
        y=np.concatenate(ys),
        fold=np.concatenate(folds),
        pre_scored=np.concatenate(pres),
        switching=np.concatenate(swit),
        site=np.concatenate(sites),
        grid_index=np.concatenate(gidx),
        temp=np.concatenate(temps),
        g=np.concatenate(gs),
        times=np.concatenate(times),
        forecast=forecast,
        n_grid=n_grid,
    )


def split_masks(*, pooled: Pooled, held_out: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return the training and early-stopping masks of the network that leaves `held_out` out.

    `held_out = None` is the full network: it trains on folds 0, 1, and 3 and stops on fold 2. A
    cross-fit network leaves fold `held_out` out, stops on fold `held_out + 1`, and trains on the
    other two folds. Rows of the scored year and rows in hand-filtered switching events are never
    used.
    """
    usable = pooled.pre_scored & ~pooled.switching
    if BLOCKED:
        before = pooled.times < BLOCK_SPLIT
        return usable & before, usable & ~before
    if held_out is None:
        val_fold = 2
        train = usable & (pooled.fold != val_fold)
    else:
        val_fold = (held_out + 1) % inputs.N_FOLDS
        train = usable & (pooled.fold != held_out) & (pooled.fold != val_fold)
    return train, usable & (pooled.fold == val_fold)


def model_for(*, fold: np.ndarray, pre_scored: np.ndarray) -> np.ndarray:
    """Return which network corrects each row: its fold (0 to 3), or 4 for the full network."""
    if BLOCKED:
        return np.zeros(len(fold), dtype=np.int64)
    return np.where(pre_scored, fold, inputs.N_FOLDS)


def correct_rows(
    *, fitted: list[net.Fitted], x: np.ndarray, fold: np.ndarray, pre_scored: np.ndarray
) -> np.ndarray:
    """Return each row's correction from the network that did not train on it."""
    which = model_for(fold=fold, pre_scored=pre_scored)
    out = np.zeros(len(x))
    for model_index, model in enumerate(fitted):
        rows = which == model_index
        if rows.any():
            out[rows] = model.predict(x=x[rows])
    return out


def train_networks(
    *, pooled: Pooled, seed: int, y: np.ndarray | None = None, folds: tuple[int | None, ...] = ()
) -> list[net.Fitted]:
    """Train the four cross-fit networks and the full network for one seed.

    Args:
        pooled: The pooled rows.
        seed: The seed.
        y: The target to train on, or `None` for the real residuals.
        folds: The held-out folds to train (`None` is the full network); empty trains all five.

    Returns:
        The networks indexed by `model_for` (a fold's network, then the full network last).
    """
    target = pooled.y if y is None else y
    wanted = folds or ((None,) if BLOCKED else (*range(inputs.N_FOLDS), None))
    fitted = []
    for held_out in wanted:
        train, val = split_masks(pooled=pooled, held_out=held_out)
        fitted.append(net.fit(x=pooled.x, y=target, train=train, val=val, seed=seed))
    return fitted


def main() -> None:
    """Train every network, average the seeds, and write the corrections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    pooled = load_pooled()
    print(f"stream rows {len(pooled.y):,}; pre-scored {int(pooled.pre_scored.sum()):,}", flush=True)
    stream_delta = np.zeros(len(pooled.y))
    forecast_delta = {site: np.zeros(frame.height) for site, (frame, _) in pooled.forecast.items()}
    history = []
    for seed in SEEDS:
        fitted = train_networks(pooled=pooled, seed=seed)
        history += [
            {
                "seed": seed,
                "model": i,
                "epochs": m.epochs,
                "val_loss": m.best_val,
                "val_loss_zero_correction": m.base_val,
            }
            for i, m in enumerate(fitted)
        ]
        stream_delta += correct_rows(
            fitted=fitted, x=pooled.x, fold=pooled.fold, pre_scored=pooled.pre_scored
        ) / len(SEEDS)
        for site, (frame, x) in pooled.forecast.items():
            times = (
                frame["valid_time"].dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
            )
            forecast_delta[site] += correct_rows(
                fitted=fitted,
                x=x,
                fold=inputs.fold_of(times=times),
                pre_scored=times < inputs.SCORED_START,
            ) / len(SEEDS)
        print(f"seed {seed} done", flush=True)
    NN_DIR.mkdir(parents=True, exist_ok=True)
    for index, site in enumerate(inputs.SITES):
        rows = pooled.site == index
        grid = np.zeros(pooled.n_grid[site])
        grid[pooled.grid_index[rows]] = stream_delta[rows]
        mask = np.zeros(pooled.n_grid[site], dtype=bool)
        mask[pooled.grid_index[rows]] = True
        frame, _ = pooled.forecast[site]
        tmp = NN_DIR / f"{site}.npz.{os.getpid()}.tmp.npz"
        np.savez(tmp, delta=grid, has_delta=mask)
        tmp.rename(NN_DIR / f"{site}.npz")
        part = frame.select("run", "valid_time").with_columns(
            delta=pl.Series(forecast_delta[site].astype(np.float32))
        )
        part.write_parquet(NN_DIR / f"{site}_rows.parquet")
    (common.WORK_DIR / f"nn_fits{common.VARIANT}.json").write_text(json.dumps(history, indent=2))
    print("written", flush=True)


if __name__ == "__main__":
    main()
