# ruff: noqa: E501, PLR0915
"""Fit and apply the night-aware blends (B1, B2) and the night band-width correction (N).

B1 and B2 act on rows with a lead above 168 hours; N acts on every lead to 240 hours. All fits are
refitted for each scored month on rows known at the month's first origin, as experiment 319 does,
and for the replicate year they are fitted on the tuning period. The fits use the convex chain's
rows, because the default's own rows exist only for the scored year; the fitted weights and
factors are applied to the default's quantiles.

    PYTHONPATH=ml-experiments/00184-dp-ens-daylight-calibration:ml-experiments/00202-dp-ens-two-number-rolling-replay:ml-experiments/00317-primary-demand-default-1-at-7-to-10-days:ml-experiments/00319-primary-demand-default-1-solar-summary-and-heuristic-blend-at-7-to-10-days \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00320-*/e00320_arms.py
"""

import json
from datetime import timedelta

import e00319_arms as a319  # ty: ignore[unresolved-import]
import e00320_common as common
import numpy as np
import polars as pl
from scipy.optimize import minimize

LOGISTIC_FIT_ROWS = 60_000


def is_night_minute(*, valid_time: pl.Series) -> np.ndarray:
    """Return True for half-hours in 21:00 to 07:30 UTC."""
    minute = (
        valid_time.dt.hour().cast(pl.Int32) * 60 + valid_time.dt.minute().cast(pl.Int32)
    ).to_numpy()
    return (minute >= common.NIGHT_START_MINUTE) | (minute <= common.NIGHT_END_MINUTE)


def forecast_cold_winter_night(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the B1 rule: winter, dark, local 18:00 to 07:00, ensemble-mean temperature below 2 C."""
    local_hour = frame["valid_time"].dt.convert_time_zone("Europe/London").dt.hour().to_numpy()
    month = frame["valid_time"].dt.month().to_numpy()
    cold = np.nan_to_num(frame["ens_mean_c"].to_numpy(), nan=99.0) < common.COLD_FORECAST_C
    return (
        cold
        & (frame["kind"] != "daylight").to_numpy()
        & np.isin(month, list(common.WINTER_MONTHS))
        & ((local_hour >= 18) | (local_hour < 7))
    )


def apply_blend(*, base: np.ndarray, heuristic: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return `base` shifted so its median is `w * median + (1 - w) * heuristic`."""
    shift = np.where(np.isfinite(heuristic), (1.0 - w) * (heuristic - base[:, 5]), 0.0)
    return np.sort(base + shift[:, None], axis=1)


def sigmoid(*, x: np.ndarray) -> np.ndarray:
    """Return the logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def fit_logistic(*, frame: pl.DataFrame, rows: np.ndarray) -> np.ndarray:
    """Return `(a, b)` of `w = sigmoid(a + b * z)`, shape (3 bands, 2 day-or-night, 2), by pinball loss."""
    sub = frame[rows]
    q = sub.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    actual = sub["actual"].to_numpy().astype(np.float64)
    heuristic = sub["heuristic_median"].to_numpy()
    z = (
        np.nan_to_num(sub["ens_mean_c"].to_numpy(), nan=common.TEMPERATURE_CENTRE_C)
        - common.TEMPERATURE_CENTRE_C
    ) / common.TEMPERATURE_SCALE_C
    band = a319.common.long_band(lead_hours=sub["lead_hours"].to_numpy())
    night = (sub["kind"] != "daylight").to_numpy().astype(int)
    ok = np.isfinite(heuristic)
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    params = np.zeros((3, 2, 2))
    for b in range(3):
        for d in range(2):
            chosen = np.flatnonzero(ok & (band == b) & (night == d))
            if len(chosen) < 200:
                params[b, d] = (20.0, 0.0)
                continue
            chosen = np.sort(
                rng.choice(chosen, size=min(LOGISTIC_FIT_ROWS, len(chosen)), replace=False)
            )

            def loss(p: np.ndarray, chosen: np.ndarray = chosen) -> float:
                w = sigmoid(x=p[0] + p[1] * z[chosen])
                shift = (1.0 - w) * (heuristic[chosen] - q[chosen, 5])
                return float(
                    a319.arms.pinball_rows(
                        actual=actual[chosen], quantiles=q[chosen] + shift[:, None]
                    ).mean()
                )

            best = min(
                (
                    minimize(loss, x0=np.array([a0, 0.0]), method="Nelder-Mead")
                    for a0 in (-2.0, 0.0, 2.0)
                ),
                key=lambda r: r.fun,
            )
            params[b, d] = best.x
    return params


def apply_logistic(
    *, frame: pl.DataFrame, rows: np.ndarray, base: np.ndarray, params: np.ndarray
) -> np.ndarray:
    """Return `base` blended with the heuristic at the logistic chain weight."""
    sub = frame[rows]
    z = (
        np.nan_to_num(sub["ens_mean_c"].to_numpy(), nan=common.TEMPERATURE_CENTRE_C)
        - common.TEMPERATURE_CENTRE_C
    ) / common.TEMPERATURE_SCALE_C
    band = a319.common.long_band(lead_hours=sub["lead_hours"].to_numpy())
    night = (sub["kind"] != "daylight").to_numpy().astype(int)
    w = sigmoid(x=params[band, night, 0] + params[band, night, 1] * z)
    return apply_blend(base=base, heuristic=sub["heuristic_median"].to_numpy(), w=w)


def load_chain_all_leads() -> pl.DataFrame:
    """Return the convex chain's rows of every period and lead to 240 hours at the six Primaries."""
    pieces = [
        pl.read_parquet(common.CHAIN_DIR / f"{site}.parquet").with_columns(site=pl.lit(site))
        for site in common.SCORED_SITES
    ]
    frame = pl.concat(pieces).filter(~pl.col("switching_mask") & pl.col("actual").is_finite())
    block = is_night_minute(valid_time=frame["valid_time"])
    return frame.with_columns(night=pl.Series(block))


def fit_width(*, history: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Return the pinball-optimal width multiplier per site and block (`night`, `day`) on the rows."""
    out: dict[str, dict[str, float]] = {}
    for site in common.SCORED_SITES:
        out[site] = {}
        for name, flag in (("night", True), ("day", False)):
            part = history.filter((pl.col("site") == site) & (pl.col("night") == flag))
            if part.height < 500:
                out[site][name] = 1.0
                continue
            q = part.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
            actual = part["actual"].to_numpy().astype(np.float64)
            median = q[:, [5]]
            offsets = q - median
            losses = [
                float(a319.arms.pinball_rows(actual=actual, quantiles=median + m * offsets).mean())
                for m in common.WIDTH_GRID
            ]
            out[site][name] = float(common.WIDTH_GRID[int(np.argmin(losses))])
    return out


def apply_width(*, base: np.ndarray, site: np.ndarray, night: np.ndarray, mult: dict) -> np.ndarray:
    """Scale each quantile's offset from the median by the site and block multiplier."""
    m = np.array([mult[s]["night" if n else "day"] for s, n in zip(site, night, strict=True)])
    median = base[:, [5]]
    return np.sort(median + m[:, None] * (base - median), axis=1)


def main() -> None:
    """Replay every scored month and the replicate year, and write the arms' long-lead quantiles."""
    frame, _ = a319.load_rows()
    scored_stored = pl.read_parquet(common.E319_ARMS / "scored_long_lead.parquet")
    repl_stored = pl.read_parquet(common.E319_ARMS / "replicate_long_lead.parquet")
    period = frame["period"].to_numpy()
    usable = a319.usable(frame=frame)
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    month = frame["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    scored = np.flatnonzero(period == "scored")
    replicate = np.flatnonzero((period == "replicate") & usable)
    base = np.full((frame.height, 11), np.nan)
    for idx, stored in ((scored, scored_stored), (replicate, repl_stored)):
        if stored.height != len(idx) or not np.array_equal(
            stored["q50"].to_numpy(), frame["q50"].to_numpy()[idx]
        ):
            msg = "experiment 319's stored rows do not line up with the rebuilt rows"
            raise ValueError(msg)
        base[idx] = (
            stored.select([f"x1_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)
        )
    cold_rule = forecast_cold_winter_night(frame=frame)
    chain_all = load_chain_all_leads()
    chain_origin = chain_all["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    chain_valid = chain_all["valid_time"].dt.replace_time_zone(None).to_numpy()
    delay = np.timedelta64(common.TELEMETRY_DELAY_MINUTES, "m")
    horizon = np.timedelta64(int(timedelta(hours=common.HORIZON_HOURS).total_seconds()), "s")

    out = {name: np.full((frame.height, 11), np.nan) for name in ("b1", "b2", "r319")}
    log: dict = {}
    for m in sorted(set(month[scored])):
        rows_m = scored[month[scored] == m]
        first = origin[rows_m].min()
        completed = (
            (origin + horizon <= first)
            & (origin >= first - np.timedelta64(a319.common.WINDOW_DAYS, "D"))
            & usable
        )
        fit_rows = np.flatnonzero(completed)
        weights = a319.fit_weights(frame=frame, rows=fit_rows)
        logistic = fit_logistic(frame=frame, rows=fit_rows)
        window = (
            (chain_origin < first)
            & (chain_origin >= first - np.timedelta64(common.WIDTH_WINDOW_DAYS, "D"))
            & (chain_valid + delay <= first)
        )
        width = fit_width(history=chain_all.filter(pl.Series(window)))
        w_table = _table_w(frame=frame, rows=rows_m, weights=weights)
        heuristic_m = frame["heuristic_median"].to_numpy()[rows_m]
        out["r319"][rows_m] = apply_blend(base=base[rows_m], heuristic=heuristic_m, w=w_table)
        out["b1"][rows_m] = apply_blend(
            base=base[rows_m], heuristic=heuristic_m, w=np.where(cold_rule[rows_m], 1.0, w_table)
        )
        out["b2"][rows_m] = apply_logistic(
            frame=frame, rows=rows_m, base=base[rows_m], params=logistic
        )
        log[m] = {
            "n_fit_rows": len(fit_rows),
            "weights": weights.tolist(),
            "logistic": logistic.tolist(),
            "width": width,
            "n_cold_rule_rows": int(cold_rule[rows_m].sum()),
        }
        print(m, "done", len(fit_rows), flush=True)

    # Replicate year: fits on the tuning period, applied to the replicate period at 168 to 240 hours.
    tuning_fit = np.flatnonzero((period == "tuning") & usable)
    weights = a319.fit_weights(frame=frame, rows=tuning_fit)
    logistic = fit_logistic(frame=frame, rows=tuning_fit)
    width = fit_width(history=chain_all.filter(pl.col("period") == "tuning"))
    w_table = _table_w(frame=frame, rows=replicate, weights=weights)
    heuristic = frame["heuristic_median"].to_numpy()
    out["r319"][replicate] = apply_blend(
        base=base[replicate], heuristic=heuristic[replicate], w=w_table
    )
    out["b1"][replicate] = apply_blend(
        base=base[replicate],
        heuristic=heuristic[replicate],
        w=np.where(cold_rule[replicate], 1.0, w_table),
    )
    out["b2"][replicate] = apply_logistic(
        frame=frame, rows=replicate, base=base[replicate], params=logistic
    )
    log["replicate"] = {"weights": weights.tolist(), "logistic": logistic.tolist(), "width": width}

    common.ARMS_DIR.mkdir(parents=True, exist_ok=True)
    for tag, idx in (("scored", scored), ("replicate", replicate)):
        columns = [
            pl.Series(f"{name}_{q}", out[name][idx, k].astype(np.float32))
            for name in out
            for k, q in enumerate(common.QUANTILES)
        ] + [
            pl.Series(f"x1_{q}", base[idx, k].astype(np.float32))
            for k, q in enumerate(common.QUANTILES)
        ]
        keep = frame[idx].select(
            "site",
            "power_fcst_init_time",
            "valid_time",
            "lead_hours",
            "ens_mean_c",
            "kind",
            "heuristic_median",
        )
        keep = keep.with_columns(
            cold_rule=pl.Series(cold_rule[idx]),
            night=pl.Series(is_night_minute(valid_time=frame["valid_time"][idx])),
        ).with_columns(columns)
        keep.write_parquet(common.ARMS_DIR / f"{tag}_long_lead.parquet")
    common.write_json_atomic(path=common.ARMS_DIR / "fit_log.json", content=log)
    print(json.dumps({"scored_rows": len(scored), "replicate_rows": len(replicate)}), flush=True)


def _table_w(*, frame: pl.DataFrame, rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return experiment 319's table weight per row."""
    sub = frame[rows]
    band = a319.common.long_band(lead_hours=sub["lead_hours"].to_numpy())
    return weights[band, a319.season_of(frame=sub)]


if __name__ == "__main__":
    main()
