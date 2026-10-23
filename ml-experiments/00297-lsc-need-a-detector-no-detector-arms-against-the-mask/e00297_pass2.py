# ruff: noqa: E501, PLR0915
"""Pass 2: fit and score the no-detector arms by cross-fitting on month parity.

Reads the per-row features and the K0 rows of pass 1's scores. For each site and each fold parity
p, every table, quantile head, and decay time is fitted on the rows of the other parity (all worlds
pooled, ordinary rows weighted 4) against residuals from observed power, which is the stream a
forecaster without a detector sees. Rows of parity p are then scored out of sample. Parity is the
target month for ordinary rows and the onset month of the shift for rows near a shift.

Arms: BAND_TABLE_PLAIN, BAND_TABLE_DISAGREE, BAND_HEAD (K0 centre with fitted quantile offsets);
ASOF_WIDEN (GAM_ONLY with its spread scaled by the as-of residual ratio); LEAD_DECAY (the
correction c0 * exp(-lead / T(u)) with the K0 spread).

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00297_pass2.py run
"""

from __future__ import annotations

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00297_fed as fed
import e00297_g_arms as arms134
import e00297_score as score
import numpy as np
import polars as pl
import xgboost as xgb
from e00297_common import WORK_DIR
from studies.dp_ens import LeadHourSpread
from studies.dp_ens_weather import lead_band_of_hours
from studies.ens_members import mixture_quantiles

N_WORKERS: Final[int] = 4
OUT: Final[Path] = WORK_DIR / "scores2"
LEVELS: Final[np.ndarray] = score.LEVELS
ORDINARY_WEIGHT: Final[float] = 4.0
MIN_CELL: Final[int] = 200
U_EDGES: Final[np.ndarray] = np.array([-4.0, -2.0, -1.0, 1.0, 2.0, 4.0])
ABS_U_EDGES: Final[np.ndarray] = np.array([1.0, 2.0, 4.0])
T_GRID: Final[tuple[float, ...]] = (3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 200.0, 1.0e9)
HEAD_ROWS: Final[int] = 1_000_000
HEAD_ROUNDS: Final[int] = 100
SEED: Final[int] = 297
NEW_ARMS: Final[tuple[str, ...]] = (
    "BAND_TABLE_PLAIN",
    "BAND_TABLE_DISAGREE",
    "BAND_HEAD",
    "ASOF_WIDEN",
    "LEAD_DECAY",
)


def weighted_quantiles(*, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The 11 levels' weighted quantiles of `values`."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(LEVELS, cum, v)


def table_fit(
    *, residual: np.ndarray, weight: np.ndarray, cell: np.ndarray, fallback: np.ndarray
) -> dict[int, np.ndarray]:
    """Weighted residual quantiles per cell; a cell under `MIN_CELL` rows uses its fallback cell."""
    overall = weighted_quantiles(values=residual, weights=weight)
    out: dict[int, np.ndarray] = {}
    for fb in np.unique(fallback):
        chosen = fallback == fb
        out[-1 - int(fb)] = (
            weighted_quantiles(values=residual[chosen], weights=weight[chosen])
            if chosen.sum() >= MIN_CELL
            else overall
        )
    for c in np.unique(cell):
        chosen = cell == c
        if chosen.sum() >= MIN_CELL:
            out[int(c)] = weighted_quantiles(values=residual[chosen], weights=weight[chosen])
    return out


def table_apply_fast(
    *, table: dict[int, np.ndarray], cell: np.ndarray, fallback: np.ndarray
) -> np.ndarray:
    """Vectorised `table_apply`: one assignment per distinct cell."""
    out = np.empty((len(cell), len(LEVELS)))
    for c in np.unique(cell):
        chosen = cell == c
        if int(c) in table:
            out[chosen] = table[int(c)]
        else:
            for f in np.unique(fallback[chosen]):
                out[chosen & (fallback == f)] = table[-1 - int(f)]
    return out


def sorted_quantiles(q: np.ndarray) -> np.ndarray:
    """Sort each row's quantiles ascending, so none cross."""
    return np.sort(q, axis=1)


def u_bin(*, u: np.ndarray) -> np.ndarray:
    """The signed bin (0 to 6) of the scaled filter level."""
    return np.searchsorted(U_EDGES, np.nan_to_num(u))


def abs_u_bin(*, u: np.ndarray) -> np.ndarray:
    """The bin (0 to 3) of the magnitude of the scaled filter level."""
    return np.searchsorted(ABS_U_EDGES, np.abs(np.nan_to_num(u)))


def head_features(frame: pl.DataFrame) -> np.ndarray:
    """The quantile head's inputs."""
    return np.column_stack(
        [
            np.nan_to_num(frame["u"].to_numpy()),
            np.nan_to_num(frame["k0_correction"].to_numpy()),
            frame["lead_hours"].to_numpy(),
            frame["hour_band"].to_numpy(),
            frame["quarter"].to_numpy(),
            frame["asof_ratio"].to_numpy(),
        ]
    ).astype(np.float32)


def fit_head(*, frame: pl.DataFrame, residual: np.ndarray, weight: np.ndarray) -> xgb.Booster:
    """A small multi-quantile XGBoost model of the residual."""
    rng = np.random.default_rng(SEED)
    if len(frame) > HEAD_ROWS:
        keep = np.sort(rng.choice(len(frame), HEAD_ROWS, replace=False))
        frame, residual, weight = frame[keep], residual[keep], weight[keep]
    dtrain = xgb.DMatrix(head_features(frame), label=residual.astype(np.float32), weight=weight)
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": LEVELS,
        "tree_method": "hist",
        "max_depth": 3,
        "eta": 0.1,
        "nthread": 1,
        "seed": SEED,
    }
    return xgb.train(params, dtrain, num_boost_round=HEAD_ROUNDS)


def fit_decay(*, frame: pl.DataFrame, weight: np.ndarray) -> dict[int, float]:
    """For each |u| bin, the decay time T (hours) that minimises the centre's weighted squared error."""
    c0 = np.nan_to_num(frame["c0"].to_numpy())
    lead = frame["lead_hours"].to_numpy()
    base = frame["gam_mean"].to_numpy()
    actual = frame["actual_obs"].to_numpy()
    bins = abs_u_bin(u=frame["u"].to_numpy())
    out = {}
    for b in range(len(ABS_U_EDGES) + 1):
        chosen = bins == b
        if not chosen.any():
            out[b] = 24.0
            continue
        errors = []
        for t in T_GRID:
            centre = base[chosen] + c0[chosen] * np.exp(-lead[chosen] / t)
            errors.append(float(np.sum(weight[chosen] * (actual[chosen] - centre) ** 2)))
        out[b] = T_GRID[int(np.argmin(errors))]
    return out


def asof_quantiles(
    *,
    data: fed.site_module.SiteData,
    site: str,
    rows: np.ndarray,
    ratio: np.ndarray,
    spreads: dict[int, LeadHourSpread],
) -> np.ndarray:
    """GAM_ONLY's quantiles with each row's spread offsets multiplied by its as-of ratio."""
    month = score.row_month(data=data, rows=rows)
    out = np.empty((len(rows), 11))
    for parity in (0, 1):
        chosen = np.flatnonzero((month % 2) == parity)
        if not len(chosen):
            continue
        sel = rows[chosen]
        offsets = (
            arms134.spread_offsets(
                spread=spreads[parity],
                band=data.hour_band[sel],
                quarter=data.quarter[sel],
                lead_hours=data.lead_hours[sel],
            )
            * ratio[chosen][:, None]
        )
        runs = data.run_of_row[sel]
        for run in np.unique(runs):
            inner = np.flatnonzero(runs == run)
            cols = sel[inner] - data.offsets[run]
            centre = data.member_means[run][:, cols].T
            out[chosen[inner]] = mixture_quantiles(
                member_quantiles=centre[:, :, None] + offsets[inner][:, None, :]
            )
    return out


def site_job(site: str) -> str:
    """Fit and score every new arm at one site, then write one file per world."""
    started = time.time()
    data = fed.get_site(site)
    frames = []
    for path in sorted(score.FEATURES.glob(f"{site}_*.parquet")):
        world = int(path.stem.split("_")[1])
        feat = pl.read_parquet(path)
        k0 = (
            pl.read_parquet(score.OUT / path.name)
            .filter(pl.col("arm") == "K0")
            .select("row", "event", "onset_month", "actual_nra", "actual_obs")
        )
        frames.append(feat.join(k0, on="row").with_columns(world=pl.lit(world, dtype=pl.Int32)))
    # features are written per period; join on (row) is unique within a site-world file.
    frame = pl.concat(frames)
    ordinary = (frame["event"] < 0).to_numpy()
    fold_month = np.where(ordinary, frame["month"].to_numpy(), frame["onset_month"].to_numpy())
    parity = (fold_month % 2).astype(np.int64)
    weight = np.where(ordinary, ORDINARY_WEIGHT, 1.0)
    k0_mean = frame["k0_mean"].to_numpy().astype(np.float64)
    obs = frame["actual_obs"].to_numpy().astype(np.float64)
    residual = obs - k0_mean
    lead_band = lead_band_of_hours(lead_hours=frame["lead_hours"].to_numpy())
    hour_band = frame["hour_band"].to_numpy().astype(np.int64)
    ubin = u_bin(u=frame["u"].to_numpy())
    cell_plain = lead_band * 100 + hour_band
    cell_dis = ubin * 100 + lead_band
    fits: dict[int, dict] = {}
    for p in (0, 1):
        train = parity == 1 - p
        fits[p] = {
            "plain": table_fit(
                residual=residual[train],
                weight=weight[train],
                cell=cell_plain[train],
                fallback=lead_band[train],
            ),
            "disagree": table_fit(
                residual=residual[train],
                weight=weight[train],
                cell=cell_dis[train],
                fallback=lead_band[train],
            ),
            "head": fit_head(
                frame=frame.filter(pl.Series(train)), residual=residual[train], weight=weight[train]
            ),
            "decay": fit_decay(frame=frame.filter(pl.Series(train)), weight=weight[train]),
        }
    print(
        f"{site} fitted, {time.time() - started:.0f} s; T by parity: {[fits[p]['decay'] for p in (0, 1)]}",
        flush=True,
    )
    spreads = score.gam_only_spreads(site)
    spread = fed.k0_spread(site)
    OUT.mkdir(parents=True, exist_ok=True)
    edges = score.edges_of(data=data)
    for world in sorted(frame["world"].unique().to_list()):
        sel_world = (frame["world"] == world).to_numpy()
        sub = frame.filter(pl.Series(sel_world))
        rows = sub["row"].to_numpy().astype(np.int64)
        par = parity[sel_world]
        sub_k0 = k0_mean[sel_world]
        q_plain = np.empty((len(sub), 11))
        q_dis = np.empty((len(sub), 11))
        q_head = np.empty((len(sub), 11))
        correction = np.zeros(data.n_rows)
        for p in (0, 1):
            chosen = np.flatnonzero(par == p)
            if not len(chosen):
                continue
            part = sub[chosen]
            lb = lead_band[sel_world][chosen]
            hb = hour_band[sel_world][chosen]
            ub = ubin[sel_world][chosen]
            q_plain[chosen] = sub_k0[chosen][:, None] + table_apply_fast(
                table=fits[p]["plain"], cell=lb * 100 + hb, fallback=lb
            )
            q_dis[chosen] = sub_k0[chosen][:, None] + table_apply_fast(
                table=fits[p]["disagree"], cell=ub * 100 + lb, fallback=lb
            )
            q_head[chosen] = sub_k0[chosen][:, None] + fits[p]["head"].predict(
                xgb.DMatrix(head_features(part))
            )
            bins = abs_u_bin(u=part["u"].to_numpy())
            tt = np.array([fits[p]["decay"][int(b)] for b in bins])
            correction[rows[chosen]] = np.nan_to_num(part["c0"].to_numpy()) * np.exp(
                -part["lead_hours"].to_numpy() / tt
            )
        q_decay = fed.quantiles_at(data=data, correction=correction, spread=spread, rows=rows)
        q_asof = asof_quantiles(
            data=data,
            site=site,
            rows=rows,
            ratio=sub["asof_ratio"].to_numpy().astype(np.float64),
            spreads=spreads,
        )
        gam_mean = data.member_mean[rows]
        arms = {
            "BAND_TABLE_PLAIN": (sorted_quantiles(q_plain), sub_k0),
            "BAND_TABLE_DISAGREE": (sorted_quantiles(q_dis), sub_k0),
            "BAND_HEAD": (sorted_quantiles(q_head), sub_k0),
            "ASOF_WIDEN": (q_asof, gam_mean),
            "LEAD_DECAY": (q_decay, gam_mean + correction[rows]),
        }
        delta_points = np.zeros(len(data.cache.end_times))
        pos = data.position[rows]
        delta_points[pos] = sub["actual_obs"].to_numpy() - sub["actual_nra"].to_numpy()
        scored = score.score_arms(
            arms=arms, rows=rows, data=data, delta_points=delta_points, edges=edges
        )
        meta = (
            pl.read_parquet(score.OUT / f"{site}_{world}.parquet")
            .filter(pl.col("arm") == "K0")
            .drop(
                [
                    c
                    for c in pl.read_parquet(score.OUT / f"{site}_{world}.parquet").columns
                    if c.startswith(("qs_", "ae_", "tw", "q0", "q5", "q9")) or c == "arm"
                ]
            )
        )
        tmp = OUT / f"{site}_{world}.tmp"
        scored.join(meta, on="row").with_columns(pl.col(pl.Float64).cast(pl.Float32)).write_parquet(
            tmp
        )
        tmp.rename(OUT / f"{site}_{world}.parquet")
    print(f"{site} done, {time.time() - started:.0f} s", flush=True)
    return site


def main() -> None:
    """Run pass 2."""
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
