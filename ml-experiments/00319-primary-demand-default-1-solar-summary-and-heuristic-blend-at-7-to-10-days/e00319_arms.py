# ruff: noqa: E501
"""Build arm x1 (solar summary S to 240 hours) and arm x2 (heuristic blend) past 168 hours.

Rows are the rows of experiment 317's convex chain with a lead above 168 hours, at six Primaries,
in all three periods. Both arms are refitted for each scored month on the rows of the trailing 365
days whose run had completed (origin plus 240 hours) before the month's first origin, as
experiment 202 does for S below 168 hours.

- **x1.** Experiment 193's Gaussian two-number arm: the 51 members' solar responses are summarised
  as a mean and a standard deviation (float32), the daylight bands are the mixture of 51 centres
  `mean + mu + s * sigma * z`, each with the chain's offsets times `f`. `f` and `s` are fitted per
  24-hour lead band by experiment 193's grid search. Night and twilight rows keep the chain's
  quantiles.
- **x2.** The chain's median is blended with the manual heuristic's median, `m = w * chain +
  (1 - w) * heuristic`, and all 11 quantiles of the chain shift by `m - chain`. `w` is fitted per
  24-hour lead band and season group (November to March, April to October, by valid month) by the
  mean pinball loss on a grid of 0, 0.05, ..., 1. The heuristic is rebuilt from the feature cache
  with experiment 161's recipe and gated against experiment 87's stored heuristic.

    PYTHONPATH=ml-experiments/00184-dp-ens-daylight-calibration:ml-experiments/00202-dp-ens-two-number-rolling-replay \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00319-*/e00319_arms.py
"""

import json
import multiprocessing as mp
from datetime import timedelta
from typing import Final

import e00184_common as c184  # ty: ignore[unresolved-import]
import e00202_arms as arms  # ty: ignore[unresolved-import]
import e00319_common as common
import numpy as np
import polars as pl

WEEK: Final[np.timedelta64] = np.timedelta64(7, "D")
WEEKLY_LAGS: Final[tuple[int, ...]] = tuple(range(1, 7))
ANNUAL_LAGS: Final[tuple[int, ...]] = tuple(range(49, 56))
TELEMETRY_DELAY: Final[np.timedelta64] = np.timedelta64(210, "m")
W_GRID: Final[np.ndarray] = np.round(np.arange(0.0, 1.0001, 0.05), 2)
DAYLIGHT_ELEVATION_DEG: Final[float] = 5.0
STATE: dict = {}


def heuristic_median(*, cache, site: str, frame: pl.DataFrame) -> np.ndarray:  # noqa: ANN001
    """Return the manual heuristic's median per row (experiment 161's recipe), NaN with no member."""
    y = cache.sites[site].y
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    members = np.full((len(frame), len(WEEKLY_LAGS) + len(ANNUAL_LAGS)), np.nan)
    for column, lag in enumerate((*WEEKLY_LAGS, *ANNUAL_LAGS)):
        stamps = valid - lag * WEEK
        index = np.clip(np.searchsorted(cache.end_times, stamps), 0, len(cache.end_times) - 1)
        position = np.where(cache.end_times[index] == stamps, index, -1)
        ok = (position >= 0) & (stamps <= origin - TELEMETRY_DELAY)
        members[ok, column] = y[position[ok]]
    out = np.full(len(frame), np.nan)
    has = np.isfinite(members).any(axis=1)
    out[has] = np.nanquantile(members[has], 0.5, axis=1)
    return out


def load_rows() -> tuple[pl.DataFrame, np.ndarray]:
    """Return every site's long-lead rows of all periods, labelled, and the (rows, 51) responses."""
    cache = c184.load_cache()
    frames, deltas = [], []
    for site in common.SCORED_SITES:
        chain = pl.read_parquet(common.CHAIN_DIR / f"{site}.parquet").filter(
            pl.col("lead_hours") > common.STORED_HORIZON_HOURS
        )
        pieces = []
        for period in common.PERIODS:
            stored = np.load(common.MEMBERS_DIR / f"{site}_{period}.npz")
            keys = pl.DataFrame(
                {
                    "run": pl.Series(stored["run"]).cast(chain["run"].dtype),
                    "valid_time": pl.Series(stored["valid_time"]).dt.replace_time_zone("UTC"),
                    "n_members": stored["n_members"],
                    "row": np.arange(len(stored["run"])),
                }
            )
            part = chain.filter(pl.col("period") == period).join(
                keys, on=["run", "valid_time"], how="inner", validate="1:1"
            )
            if part.height != chain.filter(pl.col("period") == period).height:
                msg = f"{site} {period}: rows without member responses"
                raise ValueError(msg)
            deltas.append(stored["delta_m"][part["row"].to_numpy()].astype(np.float64))
            pieces.append(part.drop("row"))
        site_frame = pl.concat(pieces)
        index = np.searchsorted(
            cache.end_times, site_frame["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        elevation = cache.sites[site].elevation_deg[index]
        kind = np.where(elevation >= DAYLIGHT_ELEVATION_DEG, "daylight", "other")
        site_frame = site_frame.with_columns(
            site=pl.lit(site),
            kind=pl.Series(kind),
            heuristic_median=pl.Series(heuristic_median(cache=cache, site=site, frame=site_frame)),
        )
        frames.append(site_frame)
    return pl.concat(frames), np.concatenate(deltas)


def usable(*, frame: pl.DataFrame) -> np.ndarray:
    """Return rows with a reading outside the hand-filtered events."""
    return (~frame["switching_mask"] & frame["actual"].is_finite()).to_numpy()


def fit_factors(
    *, frame: pl.DataFrame, delta: np.ndarray, rows: np.ndarray
) -> list[tuple[float, float]]:
    """Return `(f, s)` per long lead band on the chosen daylight rows (experiment 193's search)."""
    sub = frame[rows]
    stats = arms.summaries(delta=delta[rows])
    stats = {k: v.astype(np.float32).astype(np.float64) for k, v in stats.items()}
    mean = sub["mean"].to_numpy().astype(np.float64)
    offsets = sub.select(list(common.QUANTILES)).to_numpy().astype(np.float64) - mean[:, None]
    actual = sub["actual"].to_numpy().astype(np.float64)
    band = common.long_band(lead_hours=sub["lead_hours"].to_numpy())
    day = ((sub["kind"] == "daylight") & (sub["n_members"] > 0)).to_numpy()
    rng = np.random.default_rng(arms.SEED)
    out = []
    for b in range(3):
        chosen = np.flatnonzero(day & (band == b))
        chosen = np.sort(rng.choice(chosen, size=min(arms.FIT_ROWS, len(chosen)), replace=False))

        def loss(f: float, s: float, chosen: np.ndarray = chosen) -> float:
            centre = {k: v[chosen] for k, v in stats.items()}
            points = np.nan_to_num(arms.gaussian_points(stats=centre, spread=s))
            q = arms.mixture(mean=mean[chosen], delta=points, offsets=offsets[chosen], f=f)
            return float(arms.pinball_rows(actual=actual[chosen], quantiles=q).mean())

        f, s = 1.0, 1.0
        f = float(arms.F_GRID[int(np.argmin([loss(g, s) for g in arms.F_GRID]))])
        s = float(arms.S_GRID[int(np.argmin([loss(f, g) for g in arms.S_GRID]))])
        f = float(arms.F_GRID[int(np.argmin([loss(g, s) for g in arms.F_GRID]))])
        out.append((f, s))
    return out


def apply_x1(
    *, frame: pl.DataFrame, delta: np.ndarray, rows: np.ndarray, factors: list[tuple[float, float]]
) -> np.ndarray:
    """Return the 11 quantiles of x1 at `rows`: the mixture on daylight rows, the chain elsewhere."""
    sub = frame[rows]
    q = sub.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    mean = sub["mean"].to_numpy().astype(np.float64)
    offsets = q - mean[:, None]
    stats = {
        k: v.astype(np.float32).astype(np.float64)
        for k, v in arms.summaries(delta=delta[rows]).items()
    }
    band = common.long_band(lead_hours=sub["lead_hours"].to_numpy())
    day = ((sub["kind"] == "daylight") & (sub["n_members"] > 0)).to_numpy()
    for b, (f, s) in enumerate(factors):
        chosen = np.flatnonzero(day & (band == b))
        if len(chosen) == 0:
            continue
        centre = {k: v[chosen] for k, v in stats.items()}
        points = np.nan_to_num(arms.gaussian_points(stats=centre, spread=s))
        q[chosen] = arms.mixture(mean=mean[chosen], delta=points, offsets=offsets[chosen], f=f)
    return np.sort(q, axis=1)


def season_of(*, frame: pl.DataFrame) -> np.ndarray:
    """Return 0 for November to March and 1 for April to October, by valid month."""
    month = frame["valid_time"].dt.convert_time_zone("Europe/London").dt.month().to_numpy()
    return np.where(np.isin(month, list(common.WINTER_MONTHS)), 0, 1)


def fit_weights(*, frame: pl.DataFrame, rows: np.ndarray) -> np.ndarray:
    """Return the chain's weight `w`, shape (3 bands, 2 seasons), by pinball loss on the rows."""
    sub = frame[rows]
    q = sub.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    actual = sub["actual"].to_numpy().astype(np.float64)
    heuristic = sub["heuristic_median"].to_numpy()
    ok = np.isfinite(heuristic)
    band = common.long_band(lead_hours=sub["lead_hours"].to_numpy())
    season = season_of(frame=sub)
    weights = np.ones((3, 2))
    for b in range(3):
        for s in range(2):
            chosen = ok & (band == b) & (season == s)
            if chosen.sum() < 200:
                continue
            shift_base = heuristic[chosen] - q[chosen, 5]
            losses = [
                arms.pinball_rows(
                    actual=actual[chosen], quantiles=q[chosen] + (1.0 - w) * shift_base[:, None]
                ).mean()
                for w in W_GRID
            ]
            weights[b, s] = float(W_GRID[int(np.argmin(losses))])
    return weights


def apply_x2(*, frame: pl.DataFrame, rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the 11 quantiles of x2 at `rows`: the chain shifted to the blended median."""
    sub = frame[rows]
    q = sub.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    heuristic = sub["heuristic_median"].to_numpy()
    band = common.long_band(lead_hours=sub["lead_hours"].to_numpy())
    season = season_of(frame=sub)
    w = weights[band, season]
    shift = np.where(np.isfinite(heuristic), (1.0 - w) * (heuristic - q[:, 5]), 0.0)
    return np.sort(q + shift[:, None], axis=1)


def init_state() -> None:
    """Load rows and responses in the calling process (workers are spawned)."""
    frame, delta = load_rows()
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    STATE.update(
        frame=frame,
        delta=delta,
        origin=origin,
        scored=(frame["period"] == "scored").to_numpy(),
        month=frame["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy(),
        usable=usable(frame=frame),
    )


def replay_month(m: str) -> tuple[str, np.ndarray, dict]:
    """Fit on completed history and apply both arms for one scored month."""
    frame, delta, origin = STATE["frame"], STATE["delta"], STATE["origin"]
    rows_m = np.flatnonzero(STATE["scored"] & (STATE["month"] == m))
    first = origin[rows_m].min()
    horizon = np.timedelta64(int(timedelta(hours=common.HORIZON_HOURS).total_seconds()), "s")
    completed = (
        (origin + horizon <= first)
        & (origin >= first - np.timedelta64(common.WINDOW_DAYS, "D"))
        & STATE["usable"]
    )
    fit_rows = np.flatnonzero(completed)
    assert origin[fit_rows].max() + horizon <= first
    factors = fit_factors(frame=frame, delta=delta, rows=fit_rows)
    weights = fit_weights(frame=frame, rows=fit_rows)
    q1 = apply_x1(frame=frame, delta=delta, rows=rows_m, factors=factors)
    q2 = apply_x2(frame=frame, rows=rows_m, weights=weights)
    zero = np.zeros((frame.height, 1))
    placebo_factors = fit_factors(frame=frame, delta=zero, rows=fit_rows)
    qr = apply_x1(frame=frame, delta=zero, rows=rows_m, factors=placebo_factors)
    print(m, "done", len(fit_rows), factors, weights.tolist(), flush=True)
    log = {
        "n_fit_rows": len(fit_rows),
        "factors": factors,
        "weights": weights.tolist(),
        "placebo_factors": placebo_factors,
    }
    return m, rows_m, {"x1": q1, "x2": q2, "xr": qr, "log": log}


def write_replicate() -> None:
    """Fit on the tuning months and apply to the replicate months (the second period of the rule)."""
    frame, delta, period = STATE["frame"], STATE["delta"], frame_period()
    fit_rows = np.flatnonzero((period == "tuning") & STATE["usable"])
    rows = np.flatnonzero((period == "replicate") & STATE["usable"])
    factors = fit_factors(frame=frame, delta=delta, rows=fit_rows)
    weights = fit_weights(frame=frame, rows=fit_rows)
    zero = np.zeros((frame.height, 1))
    placebo = fit_factors(frame=frame, delta=zero, rows=fit_rows)
    quantiles = {
        "x1": apply_x1(frame=frame, delta=delta, rows=rows, factors=factors),
        "x2": apply_x2(frame=frame, rows=rows, weights=weights),
        "xr": apply_x1(frame=frame, delta=zero, rows=rows, factors=placebo),
    }
    out = frame[rows].with_columns(
        *[
            pl.Series(f"{name}_{q}", values[:, k].astype(np.float32))
            for name, values in quantiles.items()
            for k, q in enumerate(common.QUANTILES)
        ]
    )
    out.write_parquet(common.ARMS_DIR / "replicate_long_lead.parquet")
    common.write_json_atomic(
        path=common.ARMS_DIR / "replicate_fit.json",
        content={
            "factors": factors,
            "weights": weights.tolist(),
            "placebo_factors": placebo,
            "n_fit_rows": len(fit_rows),
            "n_rows": len(rows),
        },
    )


def frame_period() -> np.ndarray:
    """Return each row's period label."""
    return STATE["frame"]["period"].to_numpy()


def main() -> None:
    """Replay every scored month and write the scored rows with both arms' quantiles."""
    init_state()
    frame = STATE["frame"]
    scored = np.flatnonzero(STATE["scored"])
    months = sorted(set(STATE["month"][scored]))
    with mp.get_context("spawn").Pool(3, initializer=init_state) as pool:
        results = pool.map(replay_month, months, chunksize=1)
    out = {
        name: frame.select(list(common.QUANTILES)).to_numpy().astype(np.float64)[scored]
        for name in ("x1", "x2", "xr")
    }
    log = {}
    for m, rows_m, res in results:
        position = np.searchsorted(scored, rows_m)
        out["x1"][position] = res["x1"]
        out["x2"][position] = res["x2"]
        out["xr"][position] = res["xr"]
        log[m] = res["log"]
    scored_frame = frame[scored].with_columns(
        *[
            pl.Series(f"x1_{q}", out["x1"][:, k].astype(np.float32))
            for k, q in enumerate(common.QUANTILES)
        ],
        *[
            pl.Series(f"x2_{q}", out["x2"][:, k].astype(np.float32))
            for k, q in enumerate(common.QUANTILES)
        ],
        *[
            pl.Series(f"xr_{q}", out["xr"][:, k].astype(np.float32))
            for k, q in enumerate(common.QUANTILES)
        ],
    )
    common.ARMS_DIR.mkdir(parents=True, exist_ok=True)
    scored_frame.write_parquet(common.ARMS_DIR / "scored_long_lead.parquet")
    write_replicate()
    # Heuristic gate on the scored year against experiment 87's stored heuristic median.
    stored = pl.read_parquet(
        common.E87_PREDICTIONS / "ref_manual_heuristic" / "predictions.parquet",
        columns=[*["time_series_id", "power_fcst_init_time", "valid_time"], "q50"],
    ).unique(subset=["time_series_id", "power_fcst_init_time", "valid_time"])
    joined = scored_frame.join(
        stored,
        on=["time_series_id", "power_fcst_init_time", "valid_time"],
        how="inner",
        suffix="_87",
    ).filter(pl.col("heuristic_median").is_finite())
    diff = np.abs(joined["heuristic_median"].to_numpy() - joined["q50_87"].to_numpy() * 100.0)
    gate = {
        "rows_compared": joined.height,
        "heuristic_median_max_abs_diff": float(diff.max()),
        "heuristic_median_mean_abs_diff": float(diff.mean()),
        "rows_scored": scored_frame.height,
    }
    common.write_json_atomic(path=common.ARMS_DIR / "gates_arms.json", content=gate)
    common.write_json_atomic(path=common.ARMS_DIR / "fit_log.json", content=log)
    print(json.dumps(gate), flush=True)


if __name__ == "__main__":
    main()
