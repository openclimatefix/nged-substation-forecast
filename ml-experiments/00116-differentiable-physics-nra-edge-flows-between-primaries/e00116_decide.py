"""Apply the pre-registered decision rule of experiment 116 and write `verdict.json`.

The gate comes first: a failed reading caps the verdict at `inconclusive`. Then C1, C2, and C3 are
read against the thresholds of the plan, and the verdict follows the plan's rule:

- `supported`: the gate passes; C1, C2, and C3 pass in both years.
- `partly supported`: the gate and C2 pass, and only one of C1 and C3 passes, or C3 passes in the
  scored year but reverses sign in the replicate year.
- `small gain`: the gate, C1, and C2 pass, and C3's interval lies wholly below zero but misses its
  size, site, or gap condition.
- `refuted`: the gate passes and C2's interval lies wholly above +0.05, or C1's estimate is under
  0.25 and C3's interval lies wholly above zero.
- `inconclusive`: anything else.

    uv run python ml-experiments/00116-*/e00116_decide.py
"""

import json
from typing import Any, Final

import e00116_common as common
import e00116_score as score
import numpy as np
import polars as pl

C1_ESTIMATE: Final[float] = 0.5
C1_LOWER: Final[float] = 0.25
C2_MARGIN: Final[float] = 0.05
C3_SIZE: Final[float] = -0.5
C3_SITES: Final[int] = 3
C3_GAP: Final[float] = 0.25
C3_REAL_ROWS: Final[float] = 0.03
VERDICT_PATH: Final = common.WORK_DIR / "verdict.json"


def c1_summary() -> dict[str, float]:
    """Read the seeds of C1 and return the reconstruction share and its interval."""
    frames = [pl.read_parquet(p) for p in sorted((common.WORK_DIR / "c1").glob("seed_*.parquet"))]
    transfers = pl.concat(frames, how="diagonal").filter(pl.col("kind") == "transfer")
    seeds = sorted(transfers["seed"].unique().to_list())
    cells = []
    for seed in seeds:
        part = transfers.filter(pl.col("seed") == seed)
        by_month = part.group_by("month").agg(
            pl.col("error").sum().alias("error"), pl.col("injected").sum().alias("injected")
        )
        cells.append((by_month["error"].to_numpy(), by_month["injected"].to_numpy()))
    estimate = float(np.mean([1.0 - e.sum() / i.sum() for e, i in cells]))
    generator = np.random.default_rng(116)
    draws = score.N_RESAMPLES
    resampled = np.zeros(draws)
    for error, injected in cells:
        pick = generator.integers(0, len(error), size=(draws, len(error)))
        weight = np.stack([np.bincount(row, minlength=len(error)) for row in pick])
        resampled += 1.0 - (weight @ error) / (weight @ injected)
    resampled /= len(cells)
    tail = (100.0 - score.BONFERRONI_LEVEL) / 2.0
    clean = pl.concat(frames, how="diagonal").filter(pl.col("kind") == "clean_year")
    return {
        "estimate": estimate,
        "lower": float(np.percentile(resampled, tail)),
        "upper": float(np.percentile(resampled, 100.0 - tail)),
        "n_transfers": transfers.height,
        "clean_year_mae": {arm: float(clean[arm].mean()) for arm in ("a0_inj", "a4_inj", "or_inj")},
    }


def c3_passes(*, year: dict[str, Any]) -> dict[str, bool]:
    """Return each condition of C3 in one year."""
    contrast = year["contrasts"]["a5_minus_a4"]
    real = year["real_rows_outside_events"]
    return {
        "interval_below_zero": contrast["upper"] < 0.0,
        "interval_above_zero": contrast["lower"] > 0.0,
        "size": contrast["estimate"] <= C3_SIZE,
        "sites": sum(1 for d in year["sites"].values() if d < 0.0) >= C3_SITES,
        "gap": year["gap_closed_share"] >= C3_GAP,
        "real_rows": real is not None and real["difference"] <= C3_REAL_ROWS,
    }


def decide() -> dict[str, Any]:
    """Read every result file and return the verdict with its reasons."""
    gate = json.loads((common.WORK_DIR / "gate.json").read_text())
    c2 = json.loads((common.WORK_DIR / "c2.json").read_text())
    c3 = json.loads((common.WORK_DIR / "c3.json").read_text())
    c1 = c1_summary()
    gate_passes = bool(gate["gate_passes"]["value"])
    c1_passes = c1["estimate"] >= C1_ESTIMATE and c1["lower"] > C1_LOWER
    c2_scope = "outside_events"
    c2_by_year = {year: c2[f"{year}:{c2_scope}"]["a4_minus_a0"] for year in ("scored", "replicate")}
    c2_passes = all(v["upper"] <= C2_MARGIN for v in c2_by_year.values())
    c2_harm = any(v["lower"] > C2_MARGIN for v in c2_by_year.values())
    conditions = {year: c3_passes(year=c3[year]) for year in ("scored", "replicate")}
    needed = ("interval_below_zero", "size", "sites", "gap", "real_rows")
    full = {year: all(c[k] for k in needed) for year, c in conditions.items()}
    c3_passes_both = all(full.values())
    c3_scored_wholly_below = conditions["scored"]["interval_below_zero"]
    c3_scored_wholly_above = conditions["scored"]["interval_above_zero"]
    reversed_sign = full["scored"] and c3["replicate"]["contrasts"]["a5_minus_a4"]["estimate"] > 0.0
    if not gate_passes:
        verdict = "inconclusive"
        reason = "a gate reading failed, which caps the verdict at inconclusive"
    elif c2_harm or (c1["estimate"] < C1_LOWER and c3_scored_wholly_above):
        verdict = "refuted"
        reason = "C2 interval wholly above the margin, or C1 under 0.25 with C3 harmful"
    elif c1_passes and c2_passes and c3_passes_both:
        verdict = "supported"
        reason = "the gate, C1, C2, and C3 pass in both years"
    elif c1_passes and c2_passes and c3_scored_wholly_below and not c3_passes_both:
        verdict = "small gain"
        reason = "C3's interval lies below zero but misses a size, site, gap, or real-row condition"
    elif c2_passes and ((c1_passes != c3_passes_both) or reversed_sign):
        verdict = "partly supported"
        reason = "the gate and C2 pass, and one of C1 and C3 passes"
    else:
        verdict = "inconclusive"
        reason = "no branch of the decision rule applies"
    return {
        "verdict": verdict,
        "reason": reason,
        "gate_passes": gate_passes,
        "c1": {**c1, "passes": c1_passes},
        "c2": {"by_year": c2_by_year, "passes": c2_passes, "harm": c2_harm},
        "c3": {"conditions": conditions, "passes_both_years": c3_passes_both},
    }


def main() -> None:
    """Write and print the verdict."""
    result = decide()
    score.write_json(path=VERDICT_PATH, payload=result)
    print(json.dumps({k: result[k] for k in ("verdict", "reason")}, indent=2))


if __name__ == "__main__":
    main()
