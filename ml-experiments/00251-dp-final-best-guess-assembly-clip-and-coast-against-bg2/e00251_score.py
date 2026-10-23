# ruff: noqa: E501, PLR0915
"""Score every arm of experiment 251: gate, absolute scores, contrasts, tails, and cuts.

The quantile score (CRPS) is twice the mean pinball loss over the 11 quantiles, in points of
effective capacity. The tail-weighted score raises the quantiles and the observation to the
series' historical 95th (or 99th) percentile of power before 2025-07-02 first. Intervals
resample the months of the forecast run's initialisation time, 20,000 draws; a month's own interval
resamples that month's initialisation days.

    uv run --no-sync python ml-experiments/00251-*/e00251_score.py
"""

from typing import Final

import numpy as np
import polars as pl

import e00251_common as common  # isort: skip

N_RESAMPLES: Final[int] = 20_000
TAILS: Final[tuple[tuple[str, int], ...]] = (("q01", 0), ("q05", 1), ("q95", 9), ("q99", 10))
METRICS: Final[tuple[str, ...]] = ("qs", "mae", "tw95", "tw99")
SEASONS: Final[dict[str, frozenset[int]]] = {
    "winter (Dec-Feb)": frozenset({12, 1, 2}),
    "spring (Mar-May)": frozenset({3, 4, 5}),
    "summer (Jun-Aug)": frozenset({6, 7, 8}),
    "autumn (Sep-Nov)": frozenset({9, 10, 11}),
}
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("under 24 h", 0.0, 24.0), ("24-72 h", 24.0, 72.0), ("72-168 h", 72.0, 168.0),
    ("12-36 h", 12.0, 36.0),
)  # fmt: skip
HOUR_BANDS: Final[tuple[str, ...]] = ("00-06 UTC", "06-12 UTC", "12-18 UTC", "18-24 UTC")
LEAD_BIN_HOURS: Final[float] = 24.0
ARMS_OF_INTEREST: Final[tuple[str, ...]] = (
    "climatology", "xgboost", "manual_heuristic", "bg2", "d", "dh", "a_s", "arm1", "arm2", "arm2b", "arm2c",
)  # fmt: skip
REFS: Final[tuple[str, ...]] = ("bg2", "manual_heuristic", "xgboost", "climatology")
D5_CAPACITY_MW: Final[float] = 5.723


def thresholds() -> dict[str, dict[str, float]]:
    """Return each site's historical 95th and 99th percentile of power, in points."""
    pieces = [
        pl.read_parquet(
            common.E169_PREDS / f"ref_{period}_{site}.parquet", columns=["actual", "valid_time"]
        ).with_columns(site=pl.lit(site))
        for site in common.SCORED_SITES
        for period in ("tuning", "replicate")
    ]
    frame = (
        pl.concat(pieces)
        .filter(pl.col("valid_time") < common.SCORED_START)
        .unique(subset=["site", "valid_time"])
    )
    out: dict[str, dict[str, float]] = {}
    for (site,), group in frame.group_by("site"):
        values = group["actual"].to_numpy().astype(np.float64)
        out[str(site)] = {
            "95": float(np.quantile(values, 0.95)),
            "99": float(np.quantile(values, 0.99)),
        }
    return out


def pinball(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 quantiles, per row."""
    error = actual[:, None] - q
    pin = np.maximum(common.LEVELS[None, :] * error, (common.LEVELS[None, :] - 1.0) * error)
    return 2.0 * pin.mean(axis=1)


def per_row(
    *, rows: pl.DataFrame, q: np.ndarray, edges: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return per-row score columns for one arm."""
    actual = rows["actual"].to_numpy().astype(np.float64)
    q = q.astype(np.float64)
    error = actual[:, None] - q
    pin = np.maximum(common.LEVELS[None, :] * error, (common.LEVELS[None, :] - 1.0) * error)
    out = {"qs": 2.0 * pin.mean(axis=1), "mae": np.abs(error[:, 5])}
    for label in ("95", "99"):
        edge = edges[label]
        out[f"tw{label}"] = pinball(actual=np.maximum(actual, edge), q=np.maximum(q, edge[:, None]))
    for name, pos in TAILS:
        out[f"pin_{name}"] = pin[:, pos]
        out[f"cov_{name}"] = (error[:, pos] <= 0).astype(np.float64)
    return out


def mean_interval(*, values: np.ndarray, mask: np.ndarray, blocks: np.ndarray, rng) -> dict:  # noqa: ANN001
    """Mean of `values` over `mask` with a block-resampled 95% interval."""
    labels, codes = np.unique(blocks[mask], return_inverse=True)
    n = len(labels)
    if n == 0:
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "blocks": 0}
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


def main() -> None:
    """Load the built rows, score every arm, and write the report."""
    base = common.ARM_DIR / "scored"
    rows = pl.read_parquet(base / "rows.parquet")
    arms = {p.stem: np.load(p) for p in sorted(base.glob("*.npy"))}
    for name, q in arms.items():
        assert q.shape == (rows.height, 11), name
    site = rows["site"].to_numpy()
    th = thresholds()
    edges = {
        label: np.array([th[s][label] for s in site], dtype=np.float64) for label in ("95", "99")
    }
    m = {a: per_row(rows=rows, q=arms[a], edges=edges) for a in ARMS_OF_INTEREST}
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    day = rows["power_fcst_init_time"].dt.strftime("%Y-%m-%d").to_numpy()
    lead = rows["lead_hours"].to_numpy()
    vmonth = rows["valid_month"].to_numpy()
    hour = rows["valid_time"].dt.hour().to_numpy()
    hour_band = hour // 6
    five = np.isin(site, common.DECISION_SITES)
    every = np.ones(rows.height, dtype=bool)
    daylight = rows["daylight"].to_numpy()
    sunny = rows["sunny"].to_numpy() & daylight
    d5 = site == "D5"
    masks: dict[str, np.ndarray] = {
        "all rows": every, "five decision substations": five,
        "cold nights (below 2 C)": rows["cold2"].to_numpy(),
        "Christmas": rows["christmas"].to_numpy(), "bank holidays": rows["bank_holiday"].to_numpy(),
        "daylight": daylight, "sunny daylight": sunny, "D5 sunny daylight": sunny & d5,
        "D5 daylight": daylight & d5,
        **{k: np.isin(vmonth, list(v)) for k, v in SEASONS.items()},
        **{k: (lead >= lo) & (lead < hi) for k, lo, hi in LEAD_BANDS},
        **{f"hour {b}": hour_band == i for i, b in enumerate(HOUR_BANDS)},
        **{f"site {s}": site == s for s in common.SCORED_SITES},
    }  # fmt: skip
    rep: dict = {"n_rows": rows.height, "thresholds_points": th}
    rep["absolute"] = {
        a: {
            "five": float(m[a]["qs"][five].mean()),
            "six": float(m[a]["qs"].mean()),
            **{f"{k}_pin": float(m[a][f"pin_{k}"].mean()) for k, _ in TAILS},
            **{f"{k}_cov": float(m[a][f"cov_{k}"].mean()) for k, _ in TAILS},
            **{k: float(m[a][k].mean()) for k in METRICS},
        }
        for a in ARMS_OF_INTEREST
    }
    rep["gate"] = {
        "a_five": float(per_row(rows=rows, q=arms["d"], edges=edges)["qs"][five].mean()),
        "bg2_five": rep["absolute"]["bg2"]["five"],
        "arm1_five": rep["absolute"]["arm1"]["five"],
    }
    for key, ref, tol in (("a_five", common.GATE_ARM_A, 0.005), ("bg2_five", common.GATE_BG2, 0.005),
                          ("arm1_five", common.GATE_ARM_1, 0.005)):  # fmt: skip
        rep["gate"][f"{key}_pass"] = bool(abs(rep["gate"][key] - ref) <= tol)
    # hour-of-day cut for the tails: coverage and pinball loss by arm and hour band
    rep["tails_by_hour"] = {
        a: {
            b: {
                k: {
                    "cov": float(m[a][f"cov_{k}"][hour_band == i].mean()),
                    "pin": float(m[a][f"pin_{k}"][hour_band == i].mean()),
                }
                for k, _ in TAILS
            }
            for i, b in enumerate(HOUR_BANDS)
        }
        for a in ARMS_OF_INTEREST
    }
    rep["tails_by_cut"] = {
        a: {
            cut: {k: float(m[a][f"cov_{k}"][masks[cut]].mean()) for k, _ in TAILS}
            for cut in (
                "all rows",
                "cold nights (below 2 C)",
                "sunny daylight",
                "Christmas",
                "bank holidays",
            )
        }
        for a in ARMS_OF_INTEREST
    }
    rep["absolute_by_cut"] = {
        a: {
            cut: {k: float(m[a][k][mask].mean()) for k in METRICS}
            for cut, mask in masks.items()
            if mask.any()
        }
        for a in ARMS_OF_INTEREST
    }
    contrasts: dict = {}
    for treat in ("d", "dh", "arm1", "arm2", "arm2b", "arm2c"):
        for ref in REFS:
            c: dict = {}
            for metric in METRICS:
                diff = m[treat][metric] - m[ref][metric]
                c[metric] = {
                    cut: mean_interval(values=diff, mask=masks[cut], blocks=month, rng=rng)
                    for cut in ("all rows", "five decision substations", "cold nights (below 2 C)",
                                "Christmas", "bank holidays", "daylight", "sunny daylight")
                }  # fmt: skip
            diff = m[treat]["qs"] - m[ref]["qs"]
            c["by_month"] = {
                mo: mean_interval(values=diff, mask=month == mo, blocks=day, rng=rng)
                for mo in sorted(set(month))
            }
            c["months_won"] = int(sum(v["est"] < 0 for v in c["by_month"].values()))
            c["by_cut"] = {
                cut: mean_interval(values=diff, mask=mask, blocks=month, rng=rng)
                for cut, mask in masks.items() if mask.any()
            }  # fmt: skip
            contrasts[f"{treat}_minus_{ref}"] = c
    rep["contrasts"] = contrasts
    # D5 clip contrasts (pre-registered): rows are D5 sunny daylight and D5 daylight
    clip: dict = {}
    for treat, ref in (("arm2", "arm1"), ("arm2b", "arm1"), ("arm2b", "a_s")):
        entry: dict = {}
        for metric in ("qs", "mae"):
            diff = m[treat][metric] - m[ref][metric]
            entry[metric] = {
                cut: mean_interval(values=diff, mask=masks[cut], blocks=month, rng=rng)
                for cut in ("D5 sunny daylight", "D5 daylight", "site D5")
            }
        entry["mae_mw_d5_sunny"] = {
            k: v * D5_CAPACITY_MW / 100.0
            for k, v in entry["mae"]["D5 sunny daylight"].items()
            if k != "n"
        }
        diff_cov = {
            k: float(m[treat][f"cov_{k}"][d5].mean() - m[ref][f"cov_{k}"][d5].mean())
            for k, _ in TAILS
        }
        entry["d5_coverage_change"] = diff_cov
        clip[f"{treat}_minus_{ref}"] = entry
    rep["clip"] = clip
    other = ~d5
    rep["arm2_equals_arm1_off_d5"] = bool(np.array_equal(arms["arm1"][other], arms["arm2"][other]))
    # MAE and QS against lead time: arms by 24-hour lead bin
    edges_lead = np.arange(0.0, 168.0 + LEAD_BIN_HOURS, LEAD_BIN_HOURS)
    bins = np.digitize(lead, edges_lead[1:-1], right=False)
    rep["by_lead_bin"] = {
        a: {
            str(int(edges_lead[i])): {
                metric: mean_interval(values=m[a][metric], mask=bins == i, blocks=month, rng=rng)
                for metric in ("mae", "qs")
            }
            for i in range(len(edges_lead) - 1)
        }
        for a in ARMS_OF_INTEREST
    }
    common.write_json_atomic(path=common.REPORT_DIR / "report_scored.json", content=rep)
    print("gate", rep["gate"])
    print({a: round(v["five"], 4) for a, v in rep["absolute"].items()})
    for k in (
        "arm1_minus_bg2",
        "arm2_minus_bg2",
        "arm1_minus_manual_heuristic",
        "arm2_minus_manual_heuristic",
    ):
        print(k, contrasts[k]["qs"]["all rows"], contrasts[k]["mae"]["all rows"])
    print("clip", clip)


if __name__ == "__main__":
    main()
