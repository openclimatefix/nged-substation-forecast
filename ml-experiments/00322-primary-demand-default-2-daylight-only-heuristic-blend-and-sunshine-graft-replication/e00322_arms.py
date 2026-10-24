# ruff: noqa: PLR0915
"""Build the daylight-only blend (D1), the refitted day and night blend (D2), and D3.

All three act on rows with a lead above 168 hours and start from experiment 319's arm x1 (the
default) and arm r319 (the blend at experiment 319's weights, stored by experiment 320).

- **D1.** The blend at experiment 319's weights on half-hours with the sun above 5 degrees; the
  default's quantiles at night.
- **D2 (reference).** The blend with its chain weight refitted separately for day and night per
  24-hour lead band and season group.
- **D3.** D1 by day. At night the chain weight is `sigmoid(a + c * P)`, where P is the share of the
  51 ensemble members whose 24-hour mean temperature at the half-hour is below 3 C, with `(a, c)`
  fitted per lead band and season group on night rows.

Every fit is refitted for each scored month on rows known at the month's first origin, as
experiments 319 and 320 do; the replicate year uses the tuning period's fits.

    PYTHONPATH=<folders of experiments 184, 202, 317, 319, 320, 322> \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00322-*/e00322_arms.py
"""

import json
from datetime import timedelta

import e00320_arms as a320  # ty: ignore[unresolved-import]
import e00322_common as common
import numpy as np
import polars as pl
from scipy.optimize import minimize

a319 = a320.a319
QUANTILES = a320.common.QUANTILES


def member_share(*, frame: pl.DataFrame, site: str) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's ensemble-mean 24-hour temperature and the share of members below 3 C."""
    with np.load(common.E111_STORE / f"{site}.npz") as stored:
        origins = stored["origins"]
        temperature = stored["temperature"].astype(np.float64)
    finite = np.isfinite(temperature)
    cumulative = np.zeros((*temperature.shape[:2], temperature.shape[2] + 1))
    count = np.zeros_like(cumulative)
    np.cumsum(np.where(finite, temperature, 0.0), axis=2, out=cumulative[:, :, 1:])
    np.cumsum(finite, axis=2, out=count[:, :, 1:])
    run = frame["run"].to_numpy().astype(np.int64)
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    if not (origins[run] == origin).all():
        msg = f"{site}: the run column does not index experiment 111's store"
        raise ValueError(msg)
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    position = ((valid - origin) // np.timedelta64(30, "m")).astype(np.int64) - 1
    last = temperature.shape[2] - 1
    window = 24  # experiment 317's WINDOW_HALF_HOURS
    low = np.clip(position - window, 0, last)
    high = np.clip(position + window, 0, last)
    total = cumulative[run, :, high + 1] - cumulative[run, :, low]
    number = count[run, :, high + 1] - count[run, :, low]
    with np.errstate(invalid="ignore", divide="ignore"):
        per_member = np.where(number > 0, total / number, np.nan)
    valid_members = np.isfinite(per_member).sum(axis=1)
    cold = (per_member < common.COLD_MEMBER_C).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(valid_members > 0, cold / valid_members, np.nan)
    return np.nanmean(per_member, axis=1), share


def fit_night_logistic(*, frame: pl.DataFrame, rows: np.ndarray) -> np.ndarray:
    """Return `(a, c)` of `w = sigmoid(a + c * P)`, shape (3 bands, 2 seasons, 2), on night rows."""
    sub = frame[rows]
    q = sub.select(list(QUANTILES)).to_numpy().astype(np.float64)
    actual = sub["actual"].to_numpy().astype(np.float64)
    heuristic = sub["heuristic_median"].to_numpy()
    share = np.nan_to_num(sub["p_share"].to_numpy(), nan=0.0)
    band = a319.common.long_band(lead_hours=sub["lead_hours"].to_numpy())
    season = a319.season_of(frame=sub)
    night = (sub["kind"] != "daylight").to_numpy()
    ok = np.isfinite(heuristic) & night
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    params = np.zeros((3, 2, 2))
    params[:, :, 0] = 20.0
    for b in range(3):
        for s in range(2):
            chosen = np.flatnonzero(ok & (band == b) & (season == s))
            if len(chosen) < 200:
                continue
            chosen = np.sort(
                rng.choice(chosen, size=min(common.FIT_ROWS, len(chosen)), replace=False)
            )

            def loss(p: np.ndarray, chosen: np.ndarray = chosen) -> float:
                w = a320.sigmoid(x=p[0] + p[1] * share[chosen])
                shift = (1.0 - w) * (heuristic[chosen] - q[chosen, 5])
                return float(
                    a319.arms.pinball_rows(
                        actual=actual[chosen], quantiles=q[chosen] + shift[:, None]
                    ).mean()
                )

            best = min(
                (
                    minimize(loss, x0=np.array(x0), method="Nelder-Mead")
                    for x0 in ((-2.0, 0.0), (0.0, 0.0), (-2.0, 4.0), (2.0, 0.0))
                ),
                key=lambda r: r.fun,
            )
            params[b, s] = best.x
    return params


def night_weight(*, frame: pl.DataFrame, rows: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Return the chain weight of the member-share rule on `rows`."""
    sub = frame[rows]
    share = np.nan_to_num(sub["p_share"].to_numpy(), nan=0.0)
    band = a319.common.long_band(lead_hours=sub["lead_hours"].to_numpy())
    season = a319.season_of(frame=sub)
    return a320.sigmoid(x=params[band, season, 0] + params[band, season, 1] * share)


def blend_arms(
    *,
    frame: pl.DataFrame,
    rows: np.ndarray,
    base: np.ndarray,
    blended: np.ndarray,
    w_day: np.ndarray,
    w_night: np.ndarray,
    w_share: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the quantiles of D1, D2, and D3 on `rows`."""
    sub = frame[rows]
    day = (sub["kind"] == "daylight").to_numpy()[:, None]
    heuristic = sub["heuristic_median"].to_numpy()
    table = np.where(day[:, 0], w_day, w_night)
    return {
        "d1": np.where(day, blended, base),
        "d2": a320.apply_blend(base=base, heuristic=heuristic, w=table),
        "d3": np.where(day, blended, a320.apply_blend(base=base, heuristic=heuristic, w=w_share)),
    }


def main() -> None:
    """Replay every scored month and the replicate year, and write the arms' long-lead quantiles."""
    frame, _ = a319.load_rows()
    p_mean = np.full(frame.height, np.nan)
    p_share = np.full(frame.height, np.nan)
    for site in a320.common.SCORED_SITES:
        index = np.flatnonzero((frame["site"] == site).to_numpy())
        mean, share = member_share(frame=frame[index], site=site)
        p_mean[index], p_share[index] = mean, share
    own = frame["ens_mean_c"].to_numpy()
    both = np.isfinite(own) & np.isfinite(p_mean)
    gate = float(np.abs(own[both] - p_mean[both]).max())
    print("ensemble mean temperature gate, max abs difference:", gate, flush=True)
    if gate > 1e-3:
        msg = f"the rebuilt ensemble-mean temperature differs from experiment 317's by {gate}"
        raise ValueError(msg)
    frame = frame.with_columns(p_share=pl.Series(p_share))
    scored_stored = pl.read_parquet(common.E320_ARMS / "scored_long_lead.parquet")
    repl_stored = pl.read_parquet(common.E320_ARMS / "replicate_long_lead.parquet")
    period = frame["period"].to_numpy()
    usable = a319.usable(frame=frame)
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    month = frame["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    scored = np.flatnonzero(period == "scored")
    replicate = np.flatnonzero((period == "replicate") & usable)
    base = np.full((frame.height, 11), np.nan)
    blended = np.full((frame.height, 11), np.nan)
    for idx, stored in ((scored, scored_stored), (replicate, repl_stored)):
        keys = ["site", "power_fcst_init_time", "valid_time", "lead_hours"]
        if stored.height != len(idx) or not stored.select(keys).equals(frame[idx].select(keys)):
            msg = "experiment 320's stored rows do not line up with the rebuilt rows"
            raise ValueError(msg)
        for name, target in (("x1", base), ("r319", blended)):
            target[idx] = (
                stored.select([f"{name}_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
            )
    horizon = np.timedelta64(int(timedelta(hours=a320.common.HORIZON_HOURS).total_seconds()), "s")
    out = {name: np.full((frame.height, 11), np.nan) for name in ("d1", "d2", "d3")}
    log: dict = {}

    def table_weight(*, rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return a320._table_w(frame=frame, rows=rows, weights=weights)

    for m in sorted(set(month[scored])):
        rows_m = scored[month[scored] == m]
        first = origin[rows_m].min()
        completed = (
            (origin + horizon <= first)
            & (origin >= first - np.timedelta64(a319.common.WINDOW_DAYS, "D"))
            & usable
        )
        fit_rows = np.flatnonzero(completed)
        is_day = (frame["kind"] == "daylight").to_numpy()
        weights_day = a319.fit_weights(frame=frame, rows=fit_rows[is_day[fit_rows]])
        weights_night = a319.fit_weights(frame=frame, rows=fit_rows[~is_day[fit_rows]])
        night_params = fit_night_logistic(frame=frame, rows=fit_rows)
        arms = blend_arms(
            frame=frame,
            rows=rows_m,
            base=base[rows_m],
            blended=blended[rows_m],
            w_day=table_weight(rows=rows_m, weights=weights_day),
            w_night=table_weight(rows=rows_m, weights=weights_night),
            w_share=night_weight(frame=frame, rows=rows_m, params=night_params),
        )
        for name, q in arms.items():
            out[name][rows_m] = q
        log[m] = {
            "n_fit_rows": len(fit_rows),
            "weights_day": weights_day.tolist(),
            "weights_night": weights_night.tolist(),
            "night_logistic": night_params.tolist(),
        }
        print(m, "done", len(fit_rows), flush=True)

    tuning_fit = np.flatnonzero((period == "tuning") & usable)
    is_day = (frame["kind"] == "daylight").to_numpy()
    weights_day = a319.fit_weights(frame=frame, rows=tuning_fit[is_day[tuning_fit]])
    weights_night = a319.fit_weights(frame=frame, rows=tuning_fit[~is_day[tuning_fit]])
    night_params = fit_night_logistic(frame=frame, rows=tuning_fit)
    arms = blend_arms(
        frame=frame,
        rows=replicate,
        base=base[replicate],
        blended=blended[replicate],
        w_day=table_weight(rows=replicate, weights=weights_day),
        w_night=table_weight(rows=replicate, weights=weights_night),
        w_share=night_weight(frame=frame, rows=replicate, params=night_params),
    )
    for name, q in arms.items():
        out[name][replicate] = q
    log["replicate"] = {
        "weights_day": weights_day.tolist(),
        "weights_night": weights_night.tolist(),
        "night_logistic": night_params.tolist(),
    }

    common.ARMS_DIR.mkdir(parents=True, exist_ok=True)
    for tag, idx in (("scored", scored), ("replicate", replicate)):
        columns = [
            pl.Series(f"{name}_{q}", out[name][idx, k].astype(np.float32))
            for name in out
            for k, q in enumerate(QUANTILES)
        ] + [
            pl.Series(f"{name}_{q}", array[idx, k].astype(np.float32))
            for name, array in (("x1", base), ("r319", blended))
            for k, q in enumerate(QUANTILES)
        ]
        keep = frame[idx].select(
            "site",
            "power_fcst_init_time",
            "valid_time",
            "lead_hours",
            "ens_mean_c",
            "p_share",
            "kind",
            "heuristic_median",
        )
        keep.with_columns(columns).write_parquet(common.ARMS_DIR / f"{tag}_long_lead.parquet")
    common.write_json_atomic(path=common.ARMS_DIR / "fit_log.json", content=log)
    print(json.dumps({"scored_rows": len(scored), "replicate_rows": len(replicate)}), flush=True)


if __name__ == "__main__":
    main()
