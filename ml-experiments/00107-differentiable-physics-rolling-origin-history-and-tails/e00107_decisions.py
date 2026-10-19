"""Choose experiment 107's XGBoost bar and measure its seed floor on the burn-in windows only.

**The bar for C4 is whichever of `x_12` and `x_12_nodoy` scores better over the four burn-in
origins** (scored windows 2022-07 to 2023-06, which no contrast reads). The choice and the XGBoost
seed floor are written to `work/decisions.json` before any scored window is read. `e00107_score.py`
refuses to run without that file, and appends the tie-band hash to it.

The seed floor is the mean over the four burn-in origins of the absolute difference between the
quantile scores of `x_12` (seed 0) and `x_12_seed1`, pooled over the five decision sites, on rows
outside the hand-filtered switching masks.

    uv run python ml-experiments/00107-*/e00107_decisions.py
"""

import json
from datetime import UTC, datetime

import e00107_common as common
import numpy as np
import polars as pl
from studies.dp_gam import QUANTILE_COLUMNS

DECISIONS_PATH = common.WORK_DIR / "decisions.json"


def burn_in_scores() -> dict[tuple[str, str], float]:
    """Return the mean quantile score of each XGBoost arm at each burn-in origin.

    Returns:
        A map from (arm, origin label) to the mean quantile score over the decision sites.
    """
    labels = [common.cut_label(cut=cut) for cut in common.BURN_IN_CUTS]
    frames = [
        pl.read_parquet(common.WORK_DIR / "xgb" / "base" / f"{site}_{label}.parquet")
        for site in common.DECISION_SITES
        for label in labels
    ]
    rows = pl.concat(frames).filter(~pl.col("switching_mask"))
    scores: dict[tuple[str, str], float] = {}
    for (arm, origin), group in rows.group_by(["arm", "origin"]):
        per_row = common.quantile_score(
            actual=group["actual"].to_numpy(), quantiles=group.select(QUANTILE_COLUMNS).to_numpy()
        )
        scores[(str(arm), str(origin))] = float(per_row.mean())
    return scores


def main() -> None:
    """Write the bar choice and the seed floor."""
    scores = burn_in_scores()
    origins = [common.cut_label(cut=cut) for cut in common.BURN_IN_CUTS]
    pooled = {
        arm: float(np.mean([scores[(arm, origin)] for origin in origins]))
        for arm in ("x_12", "x_12_nodoy", "x_12_seed1")
    }
    bar = "x_12" if pooled["x_12"] <= pooled["x_12_nodoy"] else "x_12_nodoy"
    seed_differences = [abs(scores[("x_12", o)] - scores[("x_12_seed1", o)]) for o in origins]
    decisions = {
        "written_at": datetime.now(UTC).isoformat(),
        "bar_arm": bar,
        "burn_in_mean_score": pooled,
        "burn_in_score_by_origin": {f"{arm}@{origin}": v for (arm, origin), v in scores.items()},
        "seed_floor": float(np.mean(seed_differences)),
        "seed_floor_max": float(np.max(seed_differences)),
        "seed_floor_by_origin": dict(zip(origins, seed_differences, strict=True)),
    }
    DECISIONS_PATH.write_text(json.dumps(decisions, indent=2) + "\n")
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
