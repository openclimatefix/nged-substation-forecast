"""Experiment 214: the numbers the write-up quotes that `asof.json` does not hold.

Prints the reproduction of experiment 180's fixed-60% forecasts at D5, whether the stage-2 mean
equals q50, and the pooled sunny contrast of the other Primaries with each one left out.

    uv run --no-sync python ml-experiments/00214-*/e00214_checks.py
"""

import e00214_asof as asof
import e00214_common as common
import e00214_score as score
import polars as pl


def main() -> None:
    """Print the checks."""
    new = pl.read_parquet(common.arm_file(arm="k60", period="scored", site="D5"))
    old = pl.read_parquet(
        common.EXPERIMENT_180_WORK_DIR / "arms" / "k60" / "dp_preds_s2" / "d_scored_D5.parquet"
    )
    same_rows = new.shape == old.shape
    print("reproduction: same shape", same_rows)
    for column in ("q50", "q99"):
        print(
            f"  max |new - old| in {column}:",
            float((new[column] - old[column]).abs().to_numpy().max()),
        )
    print(
        "stage-2 mean equals q50 (max |mean - q50|):",
        float((new["mean"] - new["q50"]).abs().to_numpy().max()),
    )
    cache = common.load_cache()
    bg2 = pl.read_parquet(score.BG2_PATH)
    joined = {}
    for site in asof.REST_SITES:
        control = asof.load(cache=cache, site=site, arm="c", period="scored", bg2=bg2)
        arm = asof.load(cache=cache, site=site, arm="k60", period="scored", bg2=bg2)
        _, joined[site] = asof.contrast(arm=arm, control=control, bg2=bg2)
    for left_out in (None, *asof.REST_SITES):
        sites = {s: j for s, j in joined.items() if s != left_out}
        result = asof.pooled(joined_by_site=sites)["mae_minus_control"]
        print(f"pooled rest, leaving out {left_out}:", [round(x, 4) for x in result])


if __name__ == "__main__":
    main()
