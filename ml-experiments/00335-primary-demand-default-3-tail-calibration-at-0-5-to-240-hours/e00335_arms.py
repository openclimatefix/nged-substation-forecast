# ruff: noqa: E501
"""Fit the three as-of tail calibrations C1, C2, and C3 of Primary demand research default 3.

Every calibration for origins in month m is fitted at the cut on the first day of m, on completed
origins (origin plus 10 days on or before the cut) in the 12 months before the cut. History is the
replicate year (even months of 2024-25, 168 to 240 hours only) plus earlier scored months. C3 is
updated daily instead of monthly.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00335-*/e00335_arms.py
"""

import e00335_common as common
import numpy as np
import polars as pl
from scipy import optimize, stats

MAX_T_ROWS = 20_000
CONFORMAL_STEP = 0.005
CONFORMAL_WINDOW_DAYS = 60
TAIL_PROB = {name: (level if level < 0.5 else 1.0 - level) for name, _, level in common.TAILS}


def clipped(*, factor: float) -> float:
    """Return the factor bounded to the pre-registered range."""
    return float(np.clip(factor, *common.FACTOR_BOUNDS))


def ratio_factor(*, ratios: np.ndarray, tail_prob: float) -> float:
    """Return the factor at which the share of ratios above it equals the tail probability."""
    return float(np.quantile(ratios[np.isfinite(ratios)], 1.0 - tail_prob))


def fit_c1(*, hist: dict, n_groups: int) -> np.ndarray:
    """Return the C1 factor table, shape (bands, groups, 4), shrunk to 1 where a cell is thin."""
    table = np.ones((len(common.BAND_LABELS), n_groups, len(common.TAILS)))
    for b in range(len(common.BAND_LABELS)):
        for g in range(n_groups):
            m = (hist["band"] == b) & (hist["group"] == g)
            n = int(m.sum())
            if n == 0:
                continue
            weight = min(1.0, n / common.MIN_CELL_ROWS)
            for j, (name, col, _) in enumerate(common.TAILS):
                r = common.tail_ratio(actual=hist["actual"][m], quantiles=hist["q"][m], column=col)
                f = clipped(factor=ratio_factor(ratios=r, tail_prob=TAIL_PROB[name]))
                table[b, g, j] = 1.0 + weight * (f - 1.0)
    return table


def fit_t_side(*, z: np.ndarray) -> tuple[float, float, float] | None:
    """Fit location, df, and scale of a Student-t to standardised errors z so that its 0.9 quantile is 1.

    z is the error divided by the distance of the stored q90 (or q10) from the median, so the stored
    quantile sits at 1. The scale follows from the location and the degrees of freedom.
    """
    z = z[np.isfinite(z)]
    if len(z) < common.MIN_CELL_ROWS:
        return None
    if len(z) > MAX_T_ROWS:
        z = np.random.default_rng(0).choice(z, MAX_T_ROWS, replace=False)

    def nll(p: np.ndarray) -> float:
        mu, log_nu = p
        nu = 2.1 + np.exp(log_nu)
        scale = (1.0 - mu) / stats.t.ppf(0.9, nu)
        if scale <= 0.05:
            return 1e12
        return float(-stats.t.logpdf(z, nu, loc=mu, scale=scale).sum())

    best = optimize.minimize(
        nll,
        x0=np.array([0.0, 1.5]),
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-2, "maxiter": 200},
    )
    mu, nu = float(best.x[0]), 2.1 + float(np.exp(best.x[1]))
    if not np.isfinite(best.fun) or best.fun >= 1e11:
        return None
    return mu, nu, (1.0 - mu) / float(stats.t.ppf(0.9, nu))


def fit_c2(*, hist: dict, n_groups: int) -> np.ndarray:
    """Return C2 factors, shape (bands, groups, 4).

    Each factor is the fitted Student-t tail quantile over the stored q90 or q10 distance.
    """
    table = np.ones((len(common.BAND_LABELS), n_groups, len(common.TAILS)))
    for b in range(len(common.BAND_LABELS)):
        for g in range(n_groups):
            m = (hist["band"] == b) & (hist["group"] == g)
            if m.sum() < common.MIN_CELL_ROWS:
                continue
            for side, columns in (("upper", (3, 2)), ("lower", (0, 1))):
                col90 = 8 if side == "upper" else 2
                z = common.tail_ratio(
                    actual=hist["actual"][m], quantiles=hist["q"][m], column=col90
                )
                fit = fit_t_side(z=z)
                if fit is None:
                    continue
                mu, nu, scale = fit
                for j in columns:
                    _name, col, level = common.TAILS[j]
                    p = level if side == "upper" else 1.0 - level
                    ratio_to_q90 = mu + scale * stats.t.ppf(p, nu)
                    stored = np.median(
                        np.abs(hist["q"][m][:, col] - hist["q"][m][:, common.MEDIAN])
                        / np.maximum(
                            np.abs(hist["q"][m][:, col90] - hist["q"][m][:, common.MEDIAN]), 1e-6
                        )
                    )
                    table[b, g, j] = float(ratio_to_q90 / max(stored, 1e-6))
    return table


def lookup(*, table: np.ndarray, band: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Return the per-row factors, shape (rows, 4), from a (bands, groups, 4) table."""
    return table[band, group]


def conformal_factors(*, scored: dict, replicate_rows: int) -> np.ndarray:
    """Return C3's per-row factors, shape (rows, 4): adaptive conformal per band and tail, updated daily.

    Each day at 09:00 UTC, the rows whose valid time has newly passed are the feedback. The tail's
    miscoverage level moves by step x (nominal - observed miscoverage) and the factor is the
    empirical quantile at the adapted level of the ratios of the last 60 days of observed rows.
    """
    n = len(scored["actual"])
    out = np.ones((n, len(common.TAILS)))
    origin_day = scored["origin"].astype("datetime64[D]")
    days = np.unique(origin_day)
    valid = scored["valid"]
    ratios = np.stack(
        [
            common.tail_ratio(actual=scored["actual"], quantiles=scored["q"], column=c)
            for _, c, _ in common.TAILS
        ],
        axis=1,
    )
    alpha = {
        (b, j): TAIL_PROB[name]
        for b in range(len(common.BAND_LABELS))
        for j, (name, _, _) in enumerate(common.TAILS)
    }
    now_prev = days[0].astype("datetime64[us]")
    for day in days:
        now = day.astype("datetime64[us]") + np.timedelta64(9, "h")
        window_start = now - np.timedelta64(CONFORMAL_WINDOW_DAYS, "D")
        observed = valid <= now
        new = observed & (valid > now_prev)
        in_window = observed & (valid > window_start)
        rows_today = np.flatnonzero(origin_day == day)
        for b in range(len(common.BAND_LABELS)):
            band_new = new & (scored["band"] == b)
            band_win = in_window & (scored["band"] == b)
            for j, (name, _, _) in enumerate(common.TAILS):
                nominal = TAIL_PROB[name]
                factor = 1.0
                if band_win.sum() >= common.MIN_CELL_ROWS:
                    # The factor in force was applied to rows observed now, so err is their miscoverage.
                    if band_new.sum() > 0:
                        err = float((ratios[band_new, j] > out_prev[b, j]).mean())
                        alpha[b, j] = float(
                            np.clip(alpha[b, j] + CONFORMAL_STEP * (nominal - err), 1e-3, 0.5)
                        )
                    factor = clipped(
                        factor=float(np.quantile(ratios[band_win, j], 1.0 - alpha[b, j]))
                    )
                out_prev[b, j] = factor
        if len(rows_today):
            out[rows_today] = out_prev[scored["band"][rows_today]]
        now_prev = now
    return out


out_prev = np.ones((len(common.BAND_LABELS), len(common.TAILS)))


def main() -> None:
    """Fit C1 and C2 at each monthly cut, run C3 daily, and write the arm quantiles."""
    scored = pl.read_parquet(common.WORK_DIR / "default3_scored.parquet")
    replicate = pl.read_parquet(common.WORK_DIR / "default3_replicate.parquet")
    qcols = [f"d3_{n}" for n in common.QUANTILE_NAMES]
    n_groups = len(common.GROUP_NAMES)

    def pack(frame: pl.DataFrame, *, with_group: bool) -> dict:
        d = {
            "actual": frame["actual"].to_numpy().astype(np.float64),
            "q": frame.select(qcols).to_numpy().astype(np.float64),
            "band": common.band_index(lead_hours=frame["lead_hours"].to_numpy()),
            "origin": frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy(),
            "valid": frame["valid_time"].dt.replace_time_zone(None).to_numpy(),
        }
        d["group"] = (
            common.group_index(frame=frame) if with_group else np.zeros(frame.height, dtype=int)
        )
        return d

    # The replicate frame lacks the group columns, so its rows go to the group of their season and a
    # day/night label from the valid time is unavailable: they join the Nov-Mar/Apr-Oct night-free
    # groups by season only through the scored frame's flags. They are used for the pooled
    # band-level cells only (group -1 is not used); see hist_for.
    hist_scored = pack(scored, with_group=True)
    hist_rep = pack(
        replicate.with_columns(
            daylight=pl.lit(None, dtype=pl.Boolean), p_share=pl.lit(None, dtype=pl.Float64)
        ),
        with_group=True,
    )
    # Replicate rows have an unknown day/night flag, so they are labelled by season as "day" (group 0 or 2).
    month = replicate["valid_time"].dt.convert_time_zone("Europe/London").dt.month().to_numpy()
    hist_rep["group"] = np.where(np.isin(month, common.WINTER_MONTHS), 2, 0)

    cuts = sorted({(t.year, t.month) for t in scored["power_fcst_init_time"].to_list()})
    out = {k: np.zeros_like(hist_scored["q"]) for k in ("c1", "c2")}
    factor_log: dict = {}
    cut_of = scored["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    for year, mon in cuts:
        cut = np.datetime64(f"{year:04d}-{mon:02d}-01T00:00:00")
        complete = np.timedelta64(10, "D")
        start = cut - np.timedelta64(365, "D")
        parts = []
        for h in (hist_rep, hist_scored):
            keep = (h["origin"] + complete <= cut) & (h["origin"] > start)
            parts.append({k: v[keep] for k, v in h.items()})
        hist = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        label = f"{year:04d}-{mon:02d}"
        rows = np.flatnonzero(cut_of == label)
        t1 = fit_c1(hist=hist, n_groups=n_groups)
        t2 = fit_c2(hist=hist, n_groups=n_groups)
        for key, table in (("c1", t1), ("c2", t2)):
            f = lookup(
                table=table, band=hist_scored["band"][rows], group=hist_scored["group"][rows]
            )
            out[key][rows] = common.apply_factor(quantiles=hist_scored["q"][rows], factors=f)
        factor_log[label] = {
            "history_rows": len(hist["actual"]),
            "c1": t1.tolist(),
            "c2": t2.tolist(),
        }
        print(
            label,
            len(hist["actual"]),
            "c1 mean factor",
            float(t1.mean()),
            "c2",
            float(t2.mean()),
            flush=True,
        )
    c3_f = conformal_factors(scored=hist_scored, replicate_rows=len(hist_rep["actual"]))
    out["c3"] = common.apply_factor(quantiles=hist_scored["q"], factors=c3_f)
    result = scored.select(
        "site",
        "power_fcst_init_time",
        "valid_time",
        "lead_hours",
        "actual",
        "cold_winter_night",
        "sunny_daylight",
        "daylight",
        *[f"mh_{n}" for n in common.QUANTILE_NAMES],
        *[f"cl_{n}" for n in common.QUANTILE_NAMES],
        *qcols,
    ).with_columns(
        *[
            pl.Series(f"{k}_{n}", v[:, i].astype(np.float32))
            for k, v in out.items()
            for i, n in enumerate(common.QUANTILE_NAMES)
        ],
        group=pl.Series(hist_scored["group"]),
        c3_factor_q01=pl.Series(c3_f[:, 0]),
        c3_factor_q99=pl.Series(c3_f[:, 3]),
    )
    result.write_parquet(common.WORK_DIR / "arms.parquet")
    common.write_json_atomic(path=common.WORK_DIR / "factors.json", content=factor_log)
    print("done", result.shape)


if __name__ == "__main__":
    main()
