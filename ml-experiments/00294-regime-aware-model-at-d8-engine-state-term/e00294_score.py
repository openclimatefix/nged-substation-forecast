# ruff: noqa: E501, PLR0915
"""Score every arm of experiment 294 and print each number the write-up quotes.

The quantile score is twice the mean pinball loss over the 11 quantiles, in points of effective
capacity (one point is 1% of the site's effective capacity, 0.1 MW at D8). The twCRPS applies the
same score to min(quantile, t) and min(reading, t) with t the site's 10th percentile of readings in
the scored year. Intervals resample the 12 calendar months of the scored year (20,000 draws).

    uv run --no-sync python ml-experiments/00294-*/e00294_score.py
"""

import json
import pickle
from pathlib import Path
from typing import Final

import e00294_common as common
import numpy as np
import polars as pl
from e00294_detcommon import DATA_DIR as STATE_DIR
from studies.sources import ML_EXPERIMENTS_DATA_DIR

N_RESAMPLES: Final[int] = 20_000
LEVEL_BONFERRONI: Final[float] = 1.0 - 0.05 / 4
QUANTILES: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
TAILS: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)
KEYS: Final[list[str]] = ["run", "valid_time"]
E264_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00264-dp-gas-engine-asymmetric-bands-at-d8" / "work" / "preds"
)
STATE_ENGINE: Final[float] = 0.5
"""A half-hour counts as engine-running when the filtered probability at the target exceeds this."""
D8_MW_PER_POINT: Final[float] = 0.1
OUT_PATH: Final[Path] = common.DATA_DIR / "score.json"
D8_ARMS: Final[dict[str, tuple[str, str]]] = {
    "arm1": ("whole", "a0"),
    "arm2": ("post", "a0"),
    "arm3": ("whole", "regF_persist"),
    "arm3_hindsight": ("whole", "regS_true"),
    "arm3_indicator": ("whole", "ind"),
    "arm3_window": ("post", "eng_persist"),
    "arm3_units": ("whole", "regU_persist"),
    "arm3_matched": ("whole", "regP_persist"),
}
CONTROL_ARMS: Final[dict[str, tuple[str, str]]] = {
    "arm1": ("whole", "a0"),
    "arm3": ("whole", "regF_persist"),
    "arm3_indicator": ("whole", "ind"),
    "arm3_matched": ("whole", "regP_persist"),
}


def load_arm(*, config: str, kind: str, site: str) -> pl.DataFrame:
    """Return one stored arm's scored rows with experiment 264's row filters."""
    frame = pl.read_parquet(common.PREDS_DIR / config / f"scored_{site}.parquet")
    return frame.filter(
        (pl.col("arm") == kind)
        & (pl.col("lead_hours") > 0.0)
        & (pl.col("lead_hours") <= 168.0)
        & ~pl.col("switching_mask")
        & pl.col("actual").is_not_null()
    ).sort(KEYS)


def load_e264(*, arm: str) -> pl.DataFrame:
    """Return one of experiment 264's stored D8 arms."""
    return pl.read_parquet(E264_PREDS / arm / "D8.parquet").sort(KEYS)


def align(*, frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Keep only the keys that every frame holds, so arms are scored on the same rows."""
    keys = None
    for frame in frames.values():
        k = frame.select(KEYS)
        keys = k if keys is None else keys.join(k, on=KEYS, how="inner")
    assert keys is not None
    return {
        name: frame.join(keys, on=KEYS, how="inner").sort(KEYS) for name, frame in frames.items()
    }


def quantile_matrix(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the 11 quantiles as a float64 array, one row per forecast."""
    return frame.select(list(QUANTILES)).to_numpy().astype(np.float64)


def row_losses(*, frame: pl.DataFrame, threshold: float, low: float) -> dict[str, np.ndarray]:
    """Return per-row quantile score, twCRPS, mean absolute error, coverage, tails, and Brier."""
    q = np.sort(quantile_matrix(frame=frame), axis=1)
    actual = frame["actual"].to_numpy().astype(np.float64)
    error = actual[:, None] - q
    pinball = np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error)
    capped_q = np.minimum(q, threshold)
    capped_error = np.minimum(actual, threshold)[:, None] - capped_q
    tw = np.maximum(LEVELS[None, :] * capped_error, (LEVELS[None, :] - 1.0) * capped_error)
    out = {
        "qs": 2.0 * pinball.mean(axis=1),
        "tw": 2.0 * tw.mean(axis=1),
        "ae": np.abs(actual - frame["mean"].to_numpy().astype(np.float64)),
        "err": actual - frame["mean"].to_numpy().astype(np.float64),
        "in90": ((actual >= q[:, 1]) & (actual <= q[:, 9])).astype(np.float64),
        "median_minus_mean": q[:, 5] - frame["mean"].to_numpy().astype(np.float64),
    }
    for name, level in TAILS:
        j = QUANTILES.index(name)
        e = actual - q[:, j]
        out[f"pin_{name}"] = np.maximum(level * e, (level - 1.0) * e)
        out[f"below_{name}"] = (actual < q[:, j]).astype(np.float64)
    out["brier"] = (probability_below(q=q, value=low) - (actual < low).astype(np.float64)) ** 2
    return out


def probability_below(*, q: np.ndarray, value: float) -> np.ndarray:
    """Return P(reading < value) from the 11 quantiles by linear interpolation (stand-in)."""
    p = np.empty(len(q))
    p[:] = 0.005
    p[q[:, -1] <= value] = 0.995
    for j in range(len(LEVELS) - 1):
        inside = (q[:, j] <= value) & (value < q[:, j + 1])
        width = np.maximum(q[inside, j + 1] - q[inside, j], 1e-9)
        p[inside] = LEVELS[j] + (value - q[inside, j]) / width * (LEVELS[j + 1] - LEVELS[j])
    return p


def month_of(*, frame: pl.DataFrame) -> np.ndarray:
    """Return year*100 + month of each row's valid time."""
    return (frame["valid_time"].dt.year() * 100 + frame["valid_time"].dt.month()).to_numpy()


def ratio_interval(
    *,
    numerator: np.ndarray,
    denominator: np.ndarray,
    months: np.ndarray,
    level: float,
    seed: int = 294,
) -> tuple[float, float, float]:
    """Return sum(numerator) / sum(denominator) and its month-resampled interval at `level`."""
    keys = np.unique(months)
    num = np.array([numerator[months == m].sum() for m in keys])
    den = np.array([denominator[months == m].sum() for m in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(N_RESAMPLES, len(keys)))
    stats = num[draws].sum(axis=1) / den[draws].sum(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(stats, [tail, 1.0 - tail])
    return float(num.sum() / den.sum()), float(low), float(high)


def relative_contrast(
    *,
    losses: dict[str, dict[str, np.ndarray]],
    arm: str,
    reference: str,
    metric: str,
    months: np.ndarray,
    mask: np.ndarray | None = None,
    level: float = LEVEL_BONFERRONI,
) -> dict[str, float]:
    """Return (arm - reference) / reference for a metric, with a month-resampled interval."""
    keep = np.ones(len(months), dtype=bool) if mask is None else mask
    diff = (losses[arm][metric] - losses[reference][metric])[keep]
    base = losses[reference][metric][keep]
    point, low, high = ratio_interval(
        numerator=diff, denominator=base, months=months[keep], level=level
    )
    return {"relative": point, "low": low, "high": high}


def subsets(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the row masks: all, ordinary, engine-running, and winter (December to February)."""
    state = frame["p_engine_filtered"].to_numpy().astype(np.float64)
    month = frame["valid_time"].dt.month().to_numpy()
    return {
        "all": np.ones(len(frame), dtype=bool),
        "ordinary": state <= STATE_ENGINE,
        "running": state > STATE_ENGINE,
        "winter": np.isin(month, (12, 1, 2)),
        "winter_running": np.isin(month, (12, 1, 2)) & (state > STATE_ENGINE),
    }


def summarise(*, loss: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    """Return one arm's means over a row subset."""
    row = {key: float(values[mask].mean()) for key, values in loss.items() if key != "err"}
    row["mean_error"] = float(loss["err"][mask].mean())
    row["rows"] = int(mask.sum())
    return row


def persisted_state(*, frame: pl.DataFrame, site: str) -> np.ndarray:
    """Return the filtered probability at the same half-hour of the last day settled at the origin."""
    states = np.load(STATE_DIR / f"p_filtered_{site}.npy").astype(np.float64)
    origin = np.datetime64(common.STATE_GRID_START.replace(tzinfo=None), "us")
    valid = frame["valid_time"].dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
    init = (
        frame["power_fcst_init_time"].dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
    )
    ahead_days = np.ceil((valid - init) / np.timedelta64(24, "h")).astype(np.int64)
    source_end = valid - np.timedelta64(24, "h") * np.maximum(ahead_days, 1)
    position = ((source_end - np.timedelta64(30, "m") - origin) / np.timedelta64(30, "m")).astype(
        np.int64
    )
    return np.where(
        (position >= 0) & (position < len(states)),
        states[np.clip(position, 0, len(states) - 1)],
        0.0,
    )


def slope_interval(
    *, x: np.ndarray, y: np.ndarray, months: np.ndarray
) -> tuple[float, float, float]:
    """Return the least-squares slope of y on x with an intercept and a month-resampled interval."""
    keys = np.unique(months)
    n = np.array([(months == m).sum() for m in keys], dtype=np.float64)
    sx = np.array([x[months == m].sum() for m in keys])
    sy = np.array([y[months == m].sum() for m in keys])
    sxy = np.array([(x[months == m] * y[months == m]).sum() for m in keys])
    sxx = np.array([(x[months == m] ** 2).sum() for m in keys])

    def slope(w: np.ndarray) -> np.ndarray:
        """Return the slopes of resamples whose month counts are `w` (rows are resamples)."""
        tn, tx, ty = w @ n, w @ sx, w @ sy
        return (w @ sxy - tx * ty / tn) / (w @ sxx - tx**2 / tn)

    rng = np.random.default_rng(2940)
    draws = rng.integers(0, len(keys), size=(N_RESAMPLES, len(keys)))
    counts = np.stack([np.bincount(row, minlength=len(keys)) for row in draws]).astype(np.float64)
    low, high = np.quantile(slope(counts), [0.025, 0.975])
    point = float(slope(np.ones((1, len(keys))))[0])
    return point, float(low), float(high)


def cut_coefficients(*, site: str) -> list[dict]:
    """Return each monthly refit's break and engine coefficients (points) for D8's whole history."""
    fits = pickle.loads((common.FITS_DIR / "whole" / f"{site}.pkl").read_bytes())
    rows = []
    for cut in fits:
        row = {"cut": cut.cut.date().isoformat(), "n_rows": cut.n_rows}
        for family in ("regF", "regU", "regP", "ind"):
            if family in cut.families:
                row[family] = [float(x) for x in cut.families[family].theta_extra]
        rows.append(row)
    post = (
        pickle.loads((common.FITS_DIR / "post" / f"{site}.pkl").read_bytes())
        if site == "D8"
        else []
    )
    rows.extend(
        {
            "cut": cut.cut.date().isoformat(),
            "post_eng": [float(x) for x in cut.families["eng"].theta_extra],
        }
        for cut in post
    )
    return rows


def main() -> None:
    """Print the absolute scores, contrasts, and checks."""
    report: dict = {"row_counts": {}, "scores": {}, "contrasts": {}, "checks": {}}
    # --- D8
    frames = {name: load_arm(config=c, kind=k, site="D8") for name, (c, k) in D8_ARMS.items()}
    frames["arm4_emp"] = load_e264(arm="arm_emp")
    frames["arm4_2b"] = load_e264(arm="arm2b")
    frames = align(frames=frames)
    base = frames["arm1"]
    months = month_of(frame=base)
    actual = base["actual"].to_numpy().astype(np.float64)
    threshold = float(np.quantile(actual, 0.10))
    low = float(np.quantile(actual, 0.02))
    report["row_counts"]["D8"] = base.height
    report["thresholds"] = {"D8_p10_twcrps": threshold, "D8_p02_brier": low}
    masks = subsets(frame=base)
    losses = {arm: row_losses(frame=f, threshold=threshold, low=low) for arm, f in frames.items()}
    # carry the target-time state of arm 1 for subsets (other arms hold the same rows)
    print(f"D8: {base.height:,} rows; twCRPS threshold {threshold:.2f}, Brier threshold {low:.2f}")
    for name, mask in masks.items():
        report["scores"][f"D8/{name}"] = {
            arm: summarise(loss=loss, mask=mask) for arm, loss in losses.items()
        }
        print(f"\n--- D8 subset {name}: {int(mask.sum()):,} rows")
        for arm in frames:
            row = report["scores"][f"D8/{name}"][arm]
            print(
                f"{arm:15s} qs={row['qs']:.4f} tw={row['tw']:.4f} mae={row['ae']:.3f} err={row['mean_error']:+.3f} cov90={row['in90']:.3f} med-mean={row['median_minus_mean']:+.3f} brier={row['brier']:.4f} "
                + " ".join(
                    f"{n}:b={row[f'below_{n}']:.3f},p={row[f'pin_{n}']:.3f}" for n, _ in TAILS
                )
            )
    # planned contrasts
    contrasts = {
        "C1_arm3_vs_arm1_qs": ("arm3", "arm1", "qs"),
        "C2_arm3_vs_arm4_emp_qs": ("arm3", "arm4_emp", "qs"),
        "C2b_arm3_vs_arm4_2b_qs": ("arm3", "arm4_2b", "qs"),
        "C4_arm3_vs_arm1_tw": ("arm3", "arm1", "tw"),
        "arm2_vs_arm1_qs": ("arm2", "arm1", "qs"),
        "arm3_indicator_vs_arm1_qs": ("arm3_indicator", "arm1", "qs"),
        "arm3_hindsight_vs_arm1_qs": ("arm3_hindsight", "arm1", "qs"),
        "arm3_window_vs_arm2_qs": ("arm3_window", "arm2", "qs"),
        "arm3_units_vs_arm1_qs": ("arm3_units", "arm1", "qs"),
        "arm3_matched_vs_arm1_qs": ("arm3_matched", "arm1", "qs"),
        "arm3_matched_vs_arm4_emp_qs": ("arm3_matched", "arm4_emp", "qs"),
        "arm3_matched_vs_arm1_tw": ("arm3_matched", "arm1", "tw"),
        "arm3_matched_vs_arm3_indicator_qs": ("arm3_matched", "arm3_indicator", "qs"),
        "arm3_vs_arm3_indicator_qs": ("arm3", "arm3_indicator", "qs"),
    }
    print("\n=== D8 contrasts (relative to the reference, 98.75% interval)")
    for name, (arm, ref, metric) in contrasts.items():
        res = {
            subset: relative_contrast(
                losses=losses,
                arm=arm,
                reference=ref,
                metric=metric,
                months=months,
                mask=masks[subset],
            )
            for subset in ("all", "ordinary", "running", "winter")
        }
        report["contrasts"][f"D8/{name}"] = res
        print(
            f"{name:32s} "
            + " | ".join(
                f"{s}: {r['relative'] * 100:+.2f}% [{r['low'] * 100:+.2f}, {r['high'] * 100:+.2f}]"
                for s, r in res.items()
            )
        )
    # leave-one-month-out for C1
    keys = np.unique(months)
    loo = [
        relative_contrast(
            losses=losses,
            arm="arm3",
            reference="arm1",
            metric="qs",
            months=months,
            mask=months != m,
        )["relative"]
        for m in keys
    ]
    report["checks"]["C1_leave_one_month_out"] = {"min": float(min(loo)), "max": float(max(loo))}
    print(f"\nC1 leave-one-month-out range: {min(loo) * 100:+.2f}% to {max(loo) * 100:+.2f}%")
    # coverage by month at D8
    by_month = {}
    for arm in ("arm1", "arm3", "arm4_emp"):
        by_month[arm] = {int(m): float(losses[arm]["in90"][months == m].mean()) for m in keys}
    report["checks"]["coverage_by_month"] = by_month
    # --- the term out of sample
    state_p = persisted_state(frame=base, site="D8")
    err1 = losses["arm1"]["err"]
    slope = slope_interval(x=-state_p, y=err1, months=months)
    slope_target = slope_interval(
        x=-base["p_engine_filtered"].to_numpy().astype(np.float64), y=err1, months=months
    )
    ordinary = masks["ordinary"]
    out_of_sample = {
        "slope_error_on_persisted_state_mw": [v * D8_MW_PER_POINT for v in slope],
        "slope_error_on_target_state_mw": [v * D8_MW_PER_POINT for v in slope_target],
        "persisted_state_mean": float(state_p.mean()),
        "persisted_state_agreement_with_target_flag": float(
            ((state_p > STATE_ENGINE) == (~ordinary)).mean()
        ),
    }
    report["checks"]["term_out_of_sample"] = out_of_sample
    flag_target = ~ordinary
    flag_state = state_p > STATE_ENGINE
    print(
        f"Persisted-state flag vs target flag: P(state flag | running) = {float(flag_state[flag_target].mean()):.3f}, P(state flag | ordinary) = {float(flag_state[~flag_target].mean()):.3f}"
    )
    report["checks"]["persisted_flag"] = {
        "given_running": float(flag_state[flag_target].mean()),
        "given_ordinary": float(flag_state[~flag_target].mean()),
    }
    print("\nTerm out of sample (MW per unit of state): ", out_of_sample)
    coefficients = {site: cut_coefficients(site=site) for site in ("D8", *common.CONTROL_SITES)}
    report["checks"]["coefficients"] = coefficients
    scored_cuts = [
        r for r in coefficients["D8"] if "regF" in r and "2025-07-01" <= r["cut"] <= "2026-06-01"
    ]
    mw = [r["regF"][1] * D8_MW_PER_POINT for r in scored_cuts]
    mw_units = [r["regU"][1] * D8_MW_PER_POINT for r in scored_cuts]
    mw_matched = [r["regP"][1] * D8_MW_PER_POINT for r in scored_cuts]
    print(
        "Engine coefficient on the lagged (matched) state, MW:", [round(x, 2) for x in mw_matched]
    )
    report["checks"]["engine_coefficient_mw_scored_year"] = {
        "regF": mw,
        "regU_per_unit": mw_units,
        "regP_matched": mw_matched,
        "share_regF_1_to_3": float(np.mean([1.0 <= x <= 3.0 for x in mw])),
        "share_regU_1_to_3": float(np.mean([1.0 <= x <= 3.0 for x in mw_units])),
    }
    print(
        "\nEngine coefficient per monthly refit, scored year, MW (P(1+ unit) column):",
        [round(x, 2) for x in mw],
    )
    print(
        "Engine coefficient per running unit (expected-units column), MW:",
        [round(x, 2) for x in mw_units],
    )
    print("break coefficient D8 (points):", [round(r["regF"][0], 2) for r in scored_cuts])
    # --- controls
    for site in common.CONTROL_SITES:
        cf = {name: load_arm(config=c, kind=k, site=site) for name, (c, k) in CONTROL_ARMS.items()}
        cf = align(frames=cf)
        cbase = cf["arm1"]
        cmonths = month_of(frame=cbase)
        cactual = cbase["actual"].to_numpy().astype(np.float64)
        cthr, clow = float(np.quantile(cactual, 0.10)), float(np.quantile(cactual, 0.02))
        closs = {arm: row_losses(frame=f, threshold=cthr, low=clow) for arm, f in cf.items()}
        report["row_counts"][site] = cbase.height
        report["scores"][f"{site}/all"] = {
            arm: summarise(loss=loss, mask=np.ones(cbase.height, dtype=bool))
            for arm, loss in closs.items()
        }
        print(
            f"\n=== {site}: {cbase.height:,} rows, engine-flag share (filtered, target) {float((cbase['p_engine_filtered'].to_numpy() > STATE_ENGINE).mean()):.3f}"
        )
        for arm in cf:
            row = report["scores"][f"{site}/all"][arm]
            print(
                f"{arm:15s} qs={row['qs']:.4f} tw={row['tw']:.4f} mae={row['ae']:.3f} cov90={row['in90']:.3f} med-mean={row['median_minus_mean']:+.3f} brier={row['brier']:.4f}"
            )
        for name, (arm, ref) in {
            "arm3_vs_arm1": ("arm3", "arm1"),
            "arm3_indicator_vs_arm1": ("arm3_indicator", "arm1"),
            "arm3_vs_arm3_indicator": ("arm3", "arm3_indicator"),
            "arm3_matched_vs_arm1": ("arm3_matched", "arm1"),
        }.items():
            res = relative_contrast(
                losses=closs, arm=arm, reference=ref, metric="qs", months=cmonths
            )
            report["contrasts"][f"{site}/{name}_qs"] = res
            print(
                f"{name:26s} {res['relative'] * 100:+.3f}% [{res['low'] * 100:+.3f}, {res['high'] * 100:+.3f}]"
            )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
