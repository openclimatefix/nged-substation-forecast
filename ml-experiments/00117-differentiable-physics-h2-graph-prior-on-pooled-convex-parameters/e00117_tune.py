"""Choose the graph weight gamma and the holiday weight lambda_hol on the tuning data only.

**Gamma, in the plan's order.** (1) The synthetic tuning set (smooth-cell fleets with their own
seeds) scores the graph arm at each gamma by the 80% interval score: width plus ten times the miss,
relative to the parameter's typical size. (2) The real tuning year (the 2023-07-01 cut, scored
2023-24) bounds the harm: only gammas for which H minus G and NH minus N on the five decision sites
cost at most 0.02 points are kept. (3) Gamma is the largest kept gamma whose synthetic score is
within one standard error of the best. If only gamma = 0 is kept, the verdict follows
`not supported`.

**Lambda_hol.** The holiday-only fleet mean (P-hol) is scored on the tuning year's event days of the
five decision sites; lambda_hol is the largest weight within one event-day standard error of the
lowest score.

    uv run python ml-experiments/00117-*/e00117_tune.py synthetic   # reads the tuning fleets
    uv run python ml-experiments/00117-*/e00117_tune.py real        # fits the tuning year
"""

import json
import sys
from typing import Any

import e00117_arms as arms
import e00117_common as common
import numpy as np
import polars as pl
import torch
from studies.dp_gam_graph import GraphPenalty
from studies.dp_pinned_capacity import read_vintages

HARM_BOUND = 0.02
MISS_WEIGHT = 10.0
"""An 80% interval's miss is charged 2 / 0.2 times the distance to the interval."""
TUNED_PATH = common.WORK_DIR / "tuned.json"


def interval_scores(*, tag: str) -> pl.DataFrame:
    """Return the relative interval score per fleet and arm over the identified rows."""
    frames = [
        pl.read_parquet(p) for p in sorted((common.WORK_DIR / "gate" / tag).glob("*.parquet"))
    ]
    frame = pl.concat(frames).filter(pl.col("identified"))
    typical = frame.group_by("functional").agg(
        pl.col("truth").abs().median().clip(lower_bound=1e-6).alias("typical")
    )
    scored = frame.join(typical, on="functional").with_columns(
        score=(
            (pl.col("hi") - pl.col("lo"))
            + MISS_WEIGHT * (pl.col("lo") - pl.col("truth")).clip(lower_bound=0.0)
            + MISS_WEIGHT * (pl.col("truth") - pl.col("hi")).clip(lower_bound=0.0)
        )
        / pl.col("typical")
    )
    return scored.group_by("arm", "fleet").agg(pl.col("score").mean())


def tune_synthetic() -> dict[str, Any]:
    """Return the synthetic interval score of each gamma and the within-one-SE set."""
    scores = interval_scores(tag="tuning")
    rows = []
    for arm in scores["arm"].unique().to_list():
        part = scores.filter(pl.col("arm") == arm)["score"].to_numpy()
        gamma = 0.0 if arm == "g" else float(arm[2:])
        rows.append(
            {
                "gamma": gamma,
                "arm": arm,
                "mean": float(part.mean()),
                "se": float(part.std(ddof=1) / np.sqrt(len(part))),
                "fleets": len(part),
            }
        )
    rows.sort(key=lambda r: r["gamma"])
    best = min(rows, key=lambda r: r["mean"])
    within = [r["gamma"] for r in rows if r["mean"] <= best["mean"] + best["se"]]
    return {"table": rows, "best_gamma": best["gamma"], "within_one_se": within}


def real_scores() -> dict[str, Any]:
    """Fit the tuning year's arms and return the five-site scores and the harm of each gamma."""
    torch.set_num_threads(6)
    cache = common.load_cache()
    vintages = read_vintages()
    context = arms.build_context(cache=cache, vintages=vintages, cut=common.TUNING_CUT)
    ids = {cache.sites[s].table.time_series_id: s for s in common.DECISION_SITES}

    def five_site_score(frame: pl.DataFrame) -> float:
        scores = arms.score_by_site(frame=frame.filter(pl.col("time_series_id").is_in(list(ids))))
        return common.as_float(scores["score"].mean())

    def run(name: str, penalty: GraphPenalty) -> tuple[float, pl.DataFrame]:
        frame, _, _ = arms.predict_arm(context=context, name=name, penalty=penalty)
        return five_site_score(frame), frame

    g, _ = run("g", GraphPenalty.none())
    n, _ = run("n", GraphPenalty.of(fleet_rows=common.FLEET_ROWS))
    table = []
    for gamma in common.GAMMA_GRID:
        h, _ = run(f"h@{gamma:g}", GraphPenalty.of(graph_rows=gamma))
        nh, _ = run(
            f"nh@{gamma:g}", GraphPenalty.of(fleet_rows=common.FLEET_ROWS, graph_rows=gamma)
        )
        table.append(
            {
                "gamma": gamma,
                "h": h,
                "nh": nh,
                "h_minus_g": h - g,
                "nh_minus_n": nh - n,
                "kept": (h - g <= HARM_BOUND) and (nh - n <= HARM_BOUND),
            }
        )
        print(table[-1], flush=True)
    holiday = []
    five = arms.build_context(
        cache=cache, vintages=vintages, cut=common.TUNING_CUT, labels=common.DECISION_SITES
    )
    for lam in common.LAMBDA_HOLIDAY_GRID:
        frame, _, _ = arms.predict_arm(
            context=five, name=f"phol@{lam:g}", penalty=GraphPenalty.of(holiday_fleet_rows=lam)
        )
        event = frame.filter(
            pl.col("event_day")
            & ~pl.col("switching_mask")
            & pl.col("time_series_id").is_in(list(ids))
        )
        quantiles = event.select(common.QUANTILE_COLUMNS).to_numpy().astype(np.float64)
        row = common.quantile_score(
            actual=event["actual"].to_numpy().astype(np.float64), quantiles=quantiles
        )
        days = (
            event.select(day=pl.col("valid_time").dt.truncate("1d"), score=pl.Series(row))
            .group_by("day")
            .agg(pl.col("score").mean())["score"]
            .to_numpy()
        )
        holiday.append(
            {
                "lambda_hol": lam,
                "event_day_score": float(days.mean()),
                "se": float(days.std(ddof=1) / np.sqrt(len(days))),
                "event_days": len(days),
            }
        )
        print(holiday[-1], flush=True)
    return {"g": g, "n": n, "gamma": table, "holiday": holiday}


def decide(*, synthetic: dict[str, Any], real: dict[str, Any]) -> dict[str, Any]:
    """Apply the plan's rule for gamma and lambda_hol."""
    kept = {row["gamma"] for row in real["gamma"] if row["kept"]}
    allowed = sorted(kept & set(synthetic["within_one_se"]))
    gamma = max(allowed) if allowed else 0.0
    grid = list(common.GAMMA_GRID)
    above = grid[grid.index(gamma) + 1] if gamma in grid[:-1] else None
    best = min(real["holiday"], key=lambda r: r["event_day_score"])
    near = [
        r["lambda_hol"]
        for r in real["holiday"]
        if r["event_day_score"] <= best["event_day_score"] + best["se"]
    ]
    return {
        "gamma_lo": gamma,
        "gamma_hi": above,
        "only_zero_kept": gamma == 0.0,
        "at_top_of_grid": gamma == grid[-1],
        "lambda_hol": max(near),
    }


def main() -> None:
    """Run one stage and write or update `tuned.json`."""
    stage = sys.argv[1]
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    stored = json.loads(TUNED_PATH.read_text()) if TUNED_PATH.exists() else {}
    if stage == "synthetic":
        stored["synthetic"] = tune_synthetic()
    elif stage == "real":
        stored["real"] = real_scores()
    else:
        msg = f"unknown stage {stage}"
        raise SystemExit(msg)
    if "synthetic" in stored and "real" in stored:
        stored["decision"] = decide(synthetic=stored["synthetic"], real=stored["real"])
    TUNED_PATH.write_text(json.dumps(stored, indent=2) + "\n")
    print(json.dumps(stored.get("decision", {}), indent=2))


if __name__ == "__main__":
    main()
