"""Variance accounting, cold-night coverage, rank histogram, and extra scores for experiment 125.

Variance accounting splits each arm's mixture variance into the members' variance (σ_w²) and the
variance of the table's 11 atoms (σ_m²), beside the squared error of the ensemble mean. The
cold-night read uses the cold-snap episodes fixed in `work/decisions.json` before any score was
read. The rank histogram and the fair CRPS read the 51 corrected members of the scored year; the
tail-weighted CRPS reads the 11-atom mixtures of K0-w and MP.

    uv run python ml-experiments/00125-*/e00125_accounting.py
"""

import json
from typing import Final

import e00125_cache as cache_module
import e00125_common as common
import e00125_gate as gate
import e00125_post as post
import e00125_predict as predict
import e00125_results as results
import numpy as np
import polars as pl
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours
from studies.ensemble_scores import fair_crps, rank_histogram, tail_weighted_crps

NIGHT_HOURS: Final[range] = range(6)


def accounting(*, site: str, tuned: dict) -> dict:
    """Return the scored-year variance accounting of one site, per lead band."""
    runs = cache_module.read_cache(site=site, period="scored")
    tuning = cache_module.read_cache(site=site, period="tuning")
    mask = post.stacked_months_mask(runs=tuning, months=common.TUNING_MONTHS)
    mp_table = post.residual_table(runs=tuning, centre="proxy", row_filter=mask).with_factors(
        factors=tuned["mp_factors"]
    )
    k0_table = predict.arms.load_spread(arm="gamk", site=site).with_factors(
        factors=tuned["k0w_factors"]
    )
    out = {band: np.zeros(6) for band in range(N_LEAD_BANDS)}
    for run in runs:
        band = lead_band_of_hours(lead_hours=run.lead_hours)
        members = run.corrected.var(axis=0, ddof=0)
        k0 = gate.table_atom_variance(spread=k0_table, rows=run)
        mp = gate.table_atom_variance(spread=mp_table, rows=run)
        error = (run.actual - run.corrected.mean(axis=0)) ** 2
        keep = ~run.switching_mask
        for b in range(N_LEAD_BANDS):
            chosen = keep & (band == b)
            out[b] += np.array(
                [
                    members[chosen].sum(),
                    k0[chosen].sum(),
                    mp[chosen].sum(),
                    error[chosen].sum(),
                    chosen.sum(),
                    0,
                ]
            )
    return {str(b): value.tolist() for b, value in out.items()}


def members_scores(*, site: str) -> dict:
    """Return the rank histogram and fair CRPS of the corrected members, per lead band."""
    runs = cache_module.read_cache(site=site, period="scored")
    rng = np.random.default_rng(125)
    hist = np.zeros((N_LEAD_BANDS, 52), dtype=np.int64)
    crps = np.zeros((N_LEAD_BANDS, 2))
    for run in runs:
        if run.corrected.shape[0] != 51:
            continue
        keep = ~run.switching_mask
        band = lead_band_of_hours(lead_hours=run.lead_hours)
        for b in range(N_LEAD_BANDS):
            chosen = keep & (band == b)
            if not chosen.any():
                continue
            members = run.corrected[:, chosen].T
            actual = run.actual[chosen].astype(np.float64)
            hist[b] += rank_histogram(members=members, actual=actual, rng=rng)
            score = fair_crps(members=members, actual=actual)
            crps[b] += (score.sum(), len(score))
    return {"rank_histogram": hist.tolist(), "fair_crps": crps.tolist()}


def tail_weighted(*, site: str, tuned: dict) -> dict:
    """Return upper- and lower-tail CRPS of the K0-w and MP mixtures, per lead band."""
    tuning = cache_module.read_cache(site=site, period="tuning")
    actual_tuning = np.concatenate([run.actual for run in tuning]).astype(np.float64)
    upper_threshold = float(np.percentile(actual_tuning, 90))
    lower_threshold = float(np.percentile(actual_tuning, 10))
    out = {}
    for arm in ("K0-w", "MP"):
        frame = pl.read_parquet(common.WORK_DIR / "preds" / f"{arm}_scored_{site}.parquet").filter(
            ~pl.col("switching_mask")
        )
        values = frame.select(list(results.QUANTILES)).to_numpy().astype(np.float64)
        weights = np.diff(
            np.concatenate([[0.0], (gate._LEVELS[:-1] + gate._LEVELS[1:]) / 2, [1.0]])
        )
        actual = frame["actual"].to_numpy().astype(np.float64)
        band = lead_band_of_hours(lead_hours=frame["lead_hours"].to_numpy())
        for tail, threshold in (("upper", upper_threshold), ("lower", lower_threshold)):
            score = tail_weighted_crps(
                values=values, weights=weights, actual=actual, threshold=threshold, tail=tail
            )
            out[f"{arm}|{tail}"] = [
                [float(score[band == b].sum()), int((band == b).sum())] for b in range(N_LEAD_BANDS)
            ]
    return out


def cold_night_coverage() -> dict:
    """Return q05 and q95 night-band coverage on cold-snap days of the scored winter, per arm."""
    decisions = json.loads((common.WORK_DIR / "decisions.json").read_text())
    out: dict = {}
    for arm in ("K0", "K0-w", "MP", "MM"):
        pieces = []
        for site in common.DECISION_SITES:
            cold_days = set(decisions[site]["cold_days_after_rule_window"])
            frame = results.load(arm=arm, period="scored", sites=(site,))
            frame = results.with_bands(frame=frame).filter(
                (pl.col("lead_hours") >= 15.5)
                & (pl.col("lead_hours") <= 39.0)
                & pl.col("local_hour").is_in(list(NIGHT_HOURS))
                & pl.col("local_date").cast(pl.String).is_in(cold_days)
            )
            pieces.append(frame)
        frame = pl.concat(pieces)
        out[arm] = {
            "n_rows": frame.height,
            "n_site_days": frame.select(pl.struct("site", "local_date").n_unique()).item(),
            "q05": float((frame["actual"] <= frame["q05"]).mean()),
            "q95": float((frame["actual"] <= frame["q95"]).mean()),
            "q99": float((frame["actual"] <= frame["q99"]).mean()),
        }
    return out


def main() -> None:
    """Write `work/accounting.json`."""
    tuned = json.loads((common.WORK_DIR / "tuned_n8.json").read_text())
    content: dict = {"variance": {}, "members": {}, "tail_weighted": {}}
    for site in common.DECISION_SITES:
        content["variance"][site] = accounting(site=site, tuned=tuned)
        content["members"][site] = members_scores(site=site)
        content["tail_weighted"][site] = tail_weighted(site=site, tuned=tuned)
    content["cold_nights"] = cold_night_coverage()
    common.write_json_atomic(path=common.WORK_DIR / "accounting.json", content=content)
    total = np.sum([np.array(list(v.values())) for v in content["variance"].values()], axis=0)
    for band in range(N_LEAD_BANDS):
        w, k0, mp, err, n, _ = total[band]
        print(
            f"band {band}: sigma_w2 {w / n:.2f} atoms K0w {k0 / n:.2f} MP {mp / n:.2f} "
            f"mean error^2 {err / n:.2f}; total ratio K0w/MP {(w + k0) / (w + mp):.3f}"
        )
    hist = np.sum([np.array(v["rank_histogram"]) for v in content["members"].values()], axis=0)
    for band in range(N_LEAD_BANDS):
        outside = (hist[band, 0] + hist[band, -1]) / hist[band].sum()
        print(f"band {band}: observed value outside all 51 members in {outside:.1%} of rows")
    crps = np.sum([np.array(v["fair_crps"]) for v in content["members"].values()], axis=0)
    print("fair CRPS of the members per band", (crps[:, 0] / crps[:, 1]).round(2).tolist())
    for key in ("K0-w|upper", "K0-w|lower", "MP|upper", "MP|lower"):
        pooled = np.sum([np.array(v[key]) for v in content["tail_weighted"].values()], axis=0)
        print("tail-weighted CRPS", key, (pooled[:, 0] / pooled[:, 1]).round(4).tolist())
    print(json.dumps(content["cold_nights"], indent=1))


if __name__ == "__main__":
    main()
