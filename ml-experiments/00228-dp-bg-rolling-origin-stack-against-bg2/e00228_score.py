"""Score the arms of a row set: gate, absolute scores, monthly/site/season contrasts, tails.

The quantile score is twice the mean pinball loss over the 11 quantiles, in points of effective
capacity. Pooled intervals resample the months of the forecast run's initialisation time, 20,000
draws; a month's own interval resamples that month's initialisation days.

    uv run --no-sync python ml-experiments/00228-*/e00228_score.py scored
"""

import sys
from typing import Final

import numpy as np
import polars as pl

import e00228_common as common  # isort: skip

N_RESAMPLES: Final[int] = 20_000
SEED: Final[int] = 228
TAILS: Final[tuple[tuple[str, int], ...]] = (("q01", 0), ("q05", 1), ("q95", 9), ("q99", 10))
SEASONS: Final[dict[str, frozenset[int]]] = {
    "winter (Dec-Feb)": frozenset({12, 1, 2}),
    "spring (Mar-May)": frozenset({3, 4, 5}),
    "summer (Jun-Aug)": frozenset({6, 7, 8}),
    "autumn (Sep-Nov)": frozenset({9, 10, 11}),
}
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("under 24 h", 0.0, 24.0), ("24-72 h", 24.0, 72.0), ("72-168 h", 72.0, 168.0),
    ("12-36 h", 12.0, 36.0), ("all", 0.0, 168.0),
)  # fmt: skip
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (("dh", "bg2"), ("d", "bg2"), ("dhs", "bg2"))


def per_row(*, rows: pl.DataFrame, q: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-row quantile score, median error, and the tail pinball and coverage columns."""
    actual = rows["actual"].to_numpy().astype(np.float64)
    error = actual[:, None] - q.astype(np.float64)
    pin = np.maximum(common.LEVELS[None, :] * error, (common.LEVELS[None, :] - 1.0) * error)
    out = {"qs": 2.0 * pin.mean(axis=1), "abs_median": np.abs(error[:, 5])}
    for name, pos in TAILS:
        out[f"pin_{name}"] = pin[:, pos]
        out[f"cov_{name}"] = (error[:, pos] <= 0).astype(np.float64)
    return out


def diff_interval(*, values: np.ndarray, mask: np.ndarray, blocks: np.ndarray, rng) -> dict:  # noqa: ANN001
    """Mean of `values` over `mask` with a block-resampled 95% interval."""
    labels, codes = np.unique(blocks[mask], return_inverse=True)
    n = len(labels)
    sums = np.bincount(codes, weights=values[mask], minlength=n)
    counts = np.bincount(codes, minlength=n).astype(np.float64)
    draws = rng.integers(0, n, size=(N_RESAMPLES, n))
    res = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(res, [2.5, 97.5])
    return {
        "est": float(sums.sum() / counts.sum()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(mask.sum()),
    }


def main(*, name: str) -> None:
    """Load a built row set, score it, and write the report."""
    base = common.ARM_DIR / name
    rows = pl.read_parquet(base / "rows.parquet")
    arms = {p.stem: np.load(p) for p in sorted(base.glob("*.npy"))}
    m = {a: per_row(rows=rows, q=q) for a, q in arms.items()}
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    day = rows["power_fcst_init_time"].dt.strftime("%Y-%m-%d").to_numpy()
    site = rows["site"].to_numpy()
    lead = rows["lead_hours"].to_numpy()
    vmonth = rows["valid_month"].to_numpy()
    five = np.isin(site, common.DECISION_SITES)
    allrows = np.ones(rows.height, dtype=bool)
    rep: dict = {"n_rows": rows.height}

    rep["absolute"] = {
        a: {
            "five": float(m[a]["qs"][five].mean()),
            "six": float(m[a]["qs"].mean()),
            **{f"{k}_pin": float(m[a][f"pin_{k}"].mean()) for k, _ in TAILS},
            **{f"{k}_cov": float(m[a][f"cov_{k}"].mean()) for k, _ in TAILS},
            "abs_median": float(m[a]["abs_median"].mean()),
        }
        for a in arms
    }
    rep["gate"] = {
        "d_five": rep["absolute"]["d"]["five"],
        "bg2_five": rep["absolute"]["bg2"]["five"],
    }
    rep["contrasts"] = {}
    for treat, ref in CONTRASTS:
        if treat not in m:
            continue
        diff = m[treat]["qs"] - m[ref]["qs"]
        c: dict = {
            "six_pooled": diff_interval(values=diff, mask=allrows, blocks=month, rng=rng),
            "five_pooled": diff_interval(values=diff, mask=five, blocks=month, rng=rng),
            "abs_median_diff_six": float((m[treat]["abs_median"] - m[ref]["abs_median"]).mean()),
        }
        c["by_month"] = {
            mo: diff_interval(values=diff, mask=month == mo, blocks=day, rng=rng)
            for mo in sorted(set(month))
        }
        c["months_won"] = int(sum(v["est"] < 0 for v in c["by_month"].values()))
        c["by_site"] = {
            s: diff_interval(values=diff, mask=site == s, blocks=month, rng=rng)
            for s in common.SCORED_SITES
        }
        c["sites_won"] = int(sum(v["est"] < 0 for v in c["by_site"].values()))
        c["by_season"] = {
            k: diff_interval(values=diff, mask=np.isin(vmonth, list(v)), blocks=month, rng=rng)
            for k, v in SEASONS.items()
        }
        c["by_lead"] = {
            k: diff_interval(values=diff, mask=(lead >= lo) & (lead <= hi), blocks=month, rng=rng)
            for k, lo, hi in LEAD_BANDS
        }
        c["christmas"] = diff_interval(
            values=diff, mask=rows["christmas"].to_numpy(), blocks=month, rng=rng
        )
        c["bank_holiday"] = diff_interval(
            values=diff, mask=rows["bank_holiday"].to_numpy(), blocks=month, rng=rng
        )
        rep["contrasts"][f"{treat}_minus_{ref}"] = c
    rep["by_month_absolute"] = {
        a: {mo: float(m[a]["qs"][month == mo].mean()) for mo in sorted(set(month))} for a in arms
    }
    common.write_json_atomic(path=common.REPORT_DIR / f"report_{name}.json", content=rep)
    print({k: round(v["five"], 4) for k, v in rep["absolute"].items()})
    for k, c in rep["contrasts"].items():
        print(k, c["six_pooled"], "months won", c["months_won"], "sites won", c["sites_won"])


if __name__ == "__main__":
    main(name=sys.argv[1])
