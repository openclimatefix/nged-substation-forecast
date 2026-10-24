"""Score experiment 00288's arms on the public farms: the held-out check and the learning curve.

Reads the public farms' scored runs (53 Tuesdays, lead days 1 and 2) and prints the quantile score
of R0 (stored base), the pooled head trained on every farm (P2, in sample for the pool), the same
head with each fifth of the farms held out (fold), the pooled XGBoost stage (X2), and R0 refitted on
the last 12, 24, 36, and 48 months before 2024-04-01. Writes `work/pool_score.json`.
"""

import json

import e00288_common as common
import polars as pl

BASE_DIR = common.WORK_DIR / "base"
HEADS_DIR = common.WORK_DIR / "heads"
HISTORIES = (12, 24, 36, 48)
KEYS = ["farm", "power_fcst_init_time", "valid_time"]


def arm_scores(*, frame: pl.DataFrame, actual_name: str = "actual_norm") -> pl.DataFrame:
    """Return the quantile score in points of capacity for every row of an arm's frame."""
    score = common.quantile_score(
        quantiles=frame.select(common.LEVEL_COLUMNS).to_numpy().astype(float),
        actual=frame[actual_name].to_numpy().astype(float),
    )
    return frame.select(*KEYS).with_columns(score=pl.Series(score) * common.POINTS_PER_FRACTION)


def main() -> None:
    """Print and store the per-farm and pool-wide scores."""
    cases = pl.read_parquet(BASE_DIR / "cases_scored.parquet").select(
        *KEYS, "actual_norm", "lead_day"
    )
    arms: dict[str, pl.DataFrame] = {}
    for name, path in {"R0": BASE_DIR / "R0_scored_hall_seed0.parquet"}.items():
        arms[name] = cases.join(pl.read_parquet(path), on=KEYS)
    for history in HISTORIES:
        path = BASE_DIR / f"R0_scored_h{history}_seed0.parquet"
        if path.exists():
            arms[f"R0 last {history} months"] = cases.join(pl.read_parquet(path), on=KEYS)
    for name in ("P2_pool", "X2_pool", "fold_pool", "P2_ns_pool", "X2_ns_pool", "fold_ns_pool"):
        stored = pl.read_parquet(HEADS_DIR / f"{name}.parquet")
        arms[name] = stored.filter(pl.col("seed") == 0).drop("lead_day", "actual_norm")
        arms[name] = cases.join(arms[name], on=KEYS)
    result: dict[str, dict[str, float]] = {}
    per_farm = []
    for name, frame in arms.items():
        window = frame.filter(pl.col("lead_day").is_between(1, 2))
        scores = arm_scores(frame=window).join(window.select(*KEYS, "lead_day"), on=KEYS)
        result[name] = {
            "days 1-2": float(scores["score"].mean()),  # ty: ignore[invalid-argument-type]
            "all leads": float(arm_scores(frame=frame)["score"].mean()),  # ty: ignore[invalid-argument-type]
        }
        per_farm.append(
            scores.group_by("farm").agg(pl.col("score").mean()).with_columns(arm=pl.lit(name))
        )
    table = pl.concat(per_farm).pivot(on="arm", index="farm", values="score").sort("farm")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(12)
    print(table)
    print(json.dumps(result, indent=1))
    held = result["fold_pool"]["days 1-2"]
    inside = result["P2_pool"]["days 1-2"]
    print(
        f"held-out fifth vs in pool: {held:.3f} vs {inside:.3f} ({100 * (held / inside - 1):+.1f}%)"
    )
    farms_better = int((table["fold_pool"] < table["R0"]).sum())
    farms_better_ns = int((table["fold_ns_pool"] < table["R0"]).sum())
    print(f"no-static held-out head beats R0 on {farms_better_ns} of {table.height} farms")
    print(f"held-out head beats R0 on {farms_better} of {table.height} farms")
    table.write_parquet(common.WORK_DIR / "pool_score_per_farm.parquet")
    (common.WORK_DIR / "pool_score.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
