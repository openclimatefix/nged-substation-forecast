# ruff: noqa: E501, PLR0912, PLR0915
"""Score the default and the three tail calibrations: tail rates, pinball, protected groups, intervals.

OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00335-*/e00335_score.py
"""

import json

import e00335_common as common
import numpy as np
import polars as pl

ARMS = ("d3", "c1", "c2", "c3")
REFERENCES = ("mh", "cl")
CONTRASTS = (*tuple((a, "d3") for a in ("c1", "c2", "c3")), ("c2", "c1"), ("c3", "c1"))
N_DRAWS = 20_000


def interval(*, diff: np.ndarray, mask: np.ndarray, blocks: np.ndarray) -> dict:
    """Return the mean difference over the mask and its 95% and 99.5% intervals, resampling origin months."""
    labels = np.unique(blocks[mask])
    sums = np.array([diff[mask & (blocks == b)].sum() for b in labels])
    counts = np.array([(mask & (blocks == b)).sum() for b in labels], dtype=float)
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    draws = rng.integers(0, len(labels), size=(N_DRAWS, len(labels)))
    res = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    est = float(sums.sum() / counts.sum())
    lo, hi = np.percentile(res, [2.5, 97.5])
    lo2, hi2 = np.percentile(res, [0.25, 99.75])
    return {
        "est": est,
        "lo95": float(lo),
        "hi95": float(hi),
        "lo995": float(lo2),
        "hi995": float(hi2),
        "n": int(mask.sum()),
    }


def main() -> None:
    """Write the report and print the headline tables."""
    rows = pl.read_parquet(common.WORK_DIR / "arms.parquet")
    actual = rows["actual"].to_numpy().astype(np.float64)
    q = {
        a: rows.select([f"{a}_{n}" for n in common.QUANTILE_NAMES]).to_numpy().astype(np.float64)
        for a in (*ARMS, *REFERENCES)
    }
    lead = rows["lead_hours"].to_numpy()
    band = common.band_index(lead_hours=lead)
    site = rows["site"].to_numpy()
    month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    group = rows["group"].to_numpy()
    local = rows["valid_time"].dt.convert_time_zone("Europe/London")
    winter = np.isin(local.dt.month().to_numpy(), common.WINTER_MONTHS)
    pin = {a: common.pinball(actual=actual, quantiles=v) for a, v in q.items()}
    loss = {a: 2.0 * v.mean(axis=1) for a, v in pin.items()}
    # Threshold-weighted CRPS at each Primary's 0.9 quantile of the scored actuals.
    thr = np.array([np.quantile(actual[site == s], 0.9) for s in np.unique(site)])
    thr_row = thr[np.searchsorted(np.unique(site), site)]
    tw = {}
    for a, v in q.items():
        d = np.maximum(actual, thr_row)[:, None] - np.maximum(v, thr_row[:, None])
        tw[a] = 2.0 * np.maximum(common.LEVELS * d, (common.LEVELS - 1.0) * d).mean(axis=1)
    windows = {"0-240 h": np.ones(len(lead), bool), "0-168 h": lead <= 168, "168-240 h": lead > 168}
    groups = {
        "all rows": np.ones(len(lead), bool),
        "cold winter nights": rows["cold_winter_night"].fill_null(False).to_numpy(),
        "sunny daylight": rows["sunny_daylight"].fill_null(False).to_numpy(),
        "forecast cold nights": group == 4,
        "winter (Nov-Mar)": winter,
        "April to October": ~winter,
    }
    rep: dict = {"n_rows": len(lead)}
    rep["gate"] = {a: float(loss[a].mean()) for a in loss}
    rep["by_group"] = {}
    for gname, gm in groups.items():
        rep["by_group"][gname] = {}
        for wname, wm in windows.items():
            m = gm & wm
            if m.sum() < 50:
                continue
            e: dict = {"n": int(m.sum())}
            for a in (*ARMS, *REFERENCES):
                e[a] = interval(diff=loss[a], mask=m, blocks=month)["est"]
            for a, b in CONTRASTS:
                e[f"{a}_vs_{b}"] = interval(diff=loss[a] - loss[b], mask=m, blocks=month)
                e[f"{a}_twcrps_vs_{b}"] = interval(diff=tw[a] - tw[b], mask=m, blocks=month)
                for name, pos in (("q01", 0), ("q05", 1), ("q95", 9), ("q99", 10)):
                    e[f"{a}_{name}_pinball_diff_vs_{b}"] = float(
                        pin[a][m, pos].mean() - pin[b][m, pos].mean()
                    )
            rep["by_group"][gname][wname] = e
    rep["tails_by_band"] = {}
    for b, label in enumerate(common.BAND_LABELS):
        m = band == b
        rep["tails_by_band"][label] = {
            a: {
                "below_q01": float((actual < q[a][:, 0])[m].mean()),
                "below_q05": float((actual < q[a][:, 1])[m].mean()),
                "above_q95": float((actual > q[a][:, 9])[m].mean()),
                "above_q99": float((actual > q[a][:, 10])[m].mean()),
                "in_5_95": float(((actual >= q[a][:, 1]) & (actual <= q[a][:, 9]))[m].mean()),
                "q01_pinball": float(pin[a][m, 0].mean()),
                "q99_pinball": float(pin[a][m, 10].mean()),
                "q05_pinball": float(pin[a][m, 1].mean()),
                "q95_pinball": float(pin[a][m, 9].mean()),
                "score": float(loss[a][m].mean()),
            }
            for a in (*ARMS, *REFERENCES)
        }
    rep["cold_night_upper_by_band"] = {
        label: {
            a: {
                "n": int((groups["cold winter nights"] & (band == b)).sum()),
                "above_q95": float(
                    (actual > q[a][:, 9])[groups["cold winter nights"] & (band == b)].mean()
                ),
                "above_q99": float(
                    (actual > q[a][:, 10])[groups["cold winter nights"] & (band == b)].mean()
                ),
            }
            for a in ARMS
        }
        for b, label in enumerate(common.BAND_LABELS)
    }
    months = sorted(set(month))
    rep["monthly_tails"] = {
        mo: {
            a: {
                "below_q01": float((actual < q[a][:, 0])[month == mo].mean()),
                "above_q99": float((actual > q[a][:, 10])[month == mo].mean()),
                "in_5_95": float(
                    ((actual >= q[a][:, 1]) & (actual <= q[a][:, 9]))[month == mo].mean()
                ),
            }
            for a in ARMS
        }
        for mo in months
    }
    rep["worst_month_tail_miss"] = {
        a: float(
            max(
                max(abs(v[a]["below_q01"] - 0.01), abs(v[a]["above_q99"] - 0.01))
                for v in rep["monthly_tails"].values()
            )
        )
        for a in ARMS
    }
    rep["worst_month_band_miss"] = {
        a: float(max(abs(v[a]["in_5_95"] - 0.9) for v in rep["monthly_tails"].values()))
        for a in ARMS
    }
    rep["by_site"], rep["leave_one_out"] = {}, {}
    for wname in ("0-240 h", "168-240 h"):
        rep["by_site"][wname] = {
            s: {
                f"{a}_vs_d3": float((loss[a] - loss["d3"])[windows[wname] & (site == s)].mean())
                for a in ("c1", "c2", "c3")
            }
            for s in np.unique(site)
        }
        rep["leave_one_out"][wname] = {
            s: {
                f"{a}_vs_d3": interval(
                    diff=loss[a] - loss["d3"], mask=windows[wname] & (site != s), blocks=month
                )
                for a in ("c1", "c2", "c3")
            }
            for s in np.unique(site)
        }
    rep["median_minus_mean_by_site"] = {
        a: {
            s: float((q[a][site == s, 5] - q[a][site == s].mean(axis=1)).mean())
            for s in np.unique(site)
        }
        for a in ARMS
    }
    rep["abs_median_error"] = {
        a: float(np.abs(actual - q[a][:, 5]).mean()) for a in (*ARMS, *REFERENCES)
    }
    # Months with a full history below 168 hours: the first cuts only have replicate rows at 168 to 240 hours.
    late = np.isin(month, months[3:])
    rep["after_three_months"] = {a: float(loss[a][late].mean()) for a in (*ARMS, *REFERENCES)}
    rep["after_three_months_vs_d3"] = {
        a: interval(diff=loss[a] - loss["d3"], mask=late, blocks=month) for a in ("c1", "c2", "c3")
    }
    common.write_json_atomic(path=common.WORK_DIR / "report.json", content=rep)
    print(json.dumps(rep["gate"]))
    for w in ("0-240 h", "168-240 h"):
        e = rep["by_group"]["all rows"][w]
        print(w, {a: round(e[a], 4) for a in (*ARMS, *REFERENCES)})
        for a, b in CONTRASTS:
            c = e[f"{a}_vs_{b}"]
            print(
                "  ",
                a,
                "vs",
                b,
                round(c["est"], 4),
                round(c["lo95"], 4),
                round(c["hi95"], 4),
                "q01",
                round(e[f"{a}_q01_pinball_diff_vs_{b}"], 4),
                "q99",
                round(e[f"{a}_q99_pinball_diff_vs_{b}"], 4),
            )
    for label, r in rep["tails_by_band"].items():
        print(
            label,
            *[
                f"{a}: {r[a]['below_q01']:.3f}/{r[a]['above_q99']:.3f}/{r[a]['in_5_95']:.3f}"
                for a in ARMS
            ],
        )
    print(
        "worst month tail miss", rep["worst_month_tail_miss"], "band", rep["worst_month_band_miss"]
    )
    for g in ("cold winter nights", "sunny daylight"):
        for w in ("0-240 h", "168-240 h"):
            e = rep["by_group"][g][w]
            print(g, w, {f"{a}": round(e[f"{a}_vs_d3"]["est"], 4) for a in ("c1", "c2", "c3")})


if __name__ == "__main__":
    main()
