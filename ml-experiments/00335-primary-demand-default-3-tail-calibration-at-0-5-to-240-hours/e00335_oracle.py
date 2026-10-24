# ruff: noqa: E501
"""Post hoc check: fit C1 on the scored year itself (hindsight) to see whether the as-of fit or the method limits the tail correction.

OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00335-*/e00335_oracle.py
"""

import e00335_arms as arms
import e00335_common as common
import numpy as np
import polars as pl


def main() -> None:
    """Fit the C1 table on all scored rows, apply it to the same rows, and write the tail rates by band."""
    rows = pl.read_parquet(common.WORK_DIR / "arms.parquet")
    actual = rows["actual"].to_numpy().astype(np.float64)
    q = rows.select([f"d3_{n}" for n in common.QUANTILE_NAMES]).to_numpy().astype(np.float64)
    band = common.band_index(lead_hours=rows["lead_hours"].to_numpy())
    group = rows["group"].to_numpy()
    table = arms.fit_c1(hist={"actual": actual, "q": q, "band": band, "group": group}, n_groups=5)
    adjusted = common.apply_factor(quantiles=q, factors=table[band, group])
    score_d3 = common.quantile_score(actual=actual, quantiles=q)
    score_o = common.quantile_score(actual=actual, quantiles=adjusted)
    out = {
        "score_diff_vs_d3": float((score_o - score_d3).mean()),
        "factors_mean": table.mean(axis=(0, 1)).tolist(),
        "by_band": {},
    }
    for b, label in enumerate(common.BAND_LABELS):
        m = band == b
        out["by_band"][label] = {
            "below_q01": float((actual < adjusted[:, 0])[m].mean()),
            "above_q99": float((actual > adjusted[:, 10])[m].mean()),
            "in_5_95": float(((actual >= adjusted[:, 1]) & (actual <= adjusted[:, 9]))[m].mean()),
        }
    common.write_json_atomic(path=common.WORK_DIR / "oracle.json", content=out)
    print(out["score_diff_vs_d3"], out["factors_mean"])
    for label, v in out["by_band"].items():
        print(label, v)


if __name__ == "__main__":
    main()
