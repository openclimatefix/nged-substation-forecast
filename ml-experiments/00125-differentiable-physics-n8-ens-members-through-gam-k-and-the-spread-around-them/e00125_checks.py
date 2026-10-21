"""The audit checks of experiment 125: median minus mean, mean error, and leave-one-site-out.

uv run python ml-experiments/00125-*/e00125_checks.py
"""

import json

import e00125_cache as cache_module
import e00125_common as common
import e00125_results as results
import numpy as np
import polars as pl

CHECK_ARMS = ("K0", "K0-w", "MP", "MP-s", "MM")


def median_minus_mean() -> dict:
    """Return, per arm and decision site, the mean of q50 minus the corrected ensemble mean."""
    out: dict = {arm: {} for arm in CHECK_ARMS}
    for site in common.DECISION_SITES:
        runs = cache_module.read_cache(site=site, period="scored")
        mean_forecast = np.concatenate([run.corrected.mean(axis=0) for run in runs])
        for arm in CHECK_ARMS:
            frame = pl.read_parquet(common.WORK_DIR / "preds" / f"{arm}_scored_{site}.parquet")
            out[arm][site] = float((frame["q50"].to_numpy() - mean_forecast).mean())
    return out


def leave_one_out() -> dict:
    """Return the C1 differences (MP minus K0-w) with each decision site left out."""
    out: dict = {}
    for period in ("replicate", "scored"):
        frames = {
            arm: results.with_bands(
                frame=results.load(arm=arm, period=period, sites=common.DECISION_SITES)
            )
            for arm in ("MP", "K0-w")
        }
        out[period] = {}
        for left_out in (None, *common.DECISION_SITES):
            kept = {
                arm: frame.filter(
                    (pl.col("lead_hours") >= 0.5)
                    & (pl.col("lead_hours") <= 168.0)
                    & (pl.col("site") != left_out if left_out else pl.lit(True))
                )
                for arm, frame in frames.items()
            }
            item = results.contrast(treatment=kept["MP"], reference=kept["K0-w"])
            out[period][left_out or "none"] = {
                "score": item["score"]["difference"],
                "tail": item["tail"]["difference"],
            }
    return out


def main() -> None:
    """Write `work/checks.json`."""
    content = {"median_minus_mean": median_minus_mean(), "leave_one_out": leave_one_out()}
    common.write_json_atomic(path=common.WORK_DIR / "checks.json", content=content)
    print(json.dumps(content, indent=1))


if __name__ == "__main__":
    main()
