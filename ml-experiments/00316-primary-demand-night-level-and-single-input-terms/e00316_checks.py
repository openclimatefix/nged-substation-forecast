"""Checks for experiment 316: median minus mean forecast per site, and mean-forecast error at D10.

uv run --no-sync python ml-experiments/00316-*/e00316_checks.py
"""

import e00316_common as common
import numpy as np
import polars as pl


def main() -> None:
    """Write `checks.json`."""
    rows = pl.read_parquet(common.E251_SCORED / "rows.parquet")
    default = np.load(common.E251_SCORED / "arm1.npy").astype(np.float64)
    head = np.load(common.E314_SCORED / "P.npy").astype(np.float64)
    site = rows["site"].to_numpy()
    actual = rows["actual"].to_numpy().astype(np.float64)
    mean_d = rows["mean_d"].to_numpy().astype(np.float64)
    d10 = site == "D10"
    out = {
        "median_minus_mean_per_site": {
            s: float((default[site == s, 5] - mean_d[site == s]).mean()) for s in np.unique(site)
        },
        "d10_mean_abs_error_of_median": {
            "default": float(np.abs(actual[d10] - default[d10, 5]).mean()),
            "P": float(np.abs(actual[d10] - head[d10, 5]).mean()),
        },
        "d10_mean_abs_error_of_quantile_mean": {
            "default": float(np.abs(actual[d10] - default[d10].mean(axis=1)).mean()),
            "P": float(np.abs(actual[d10] - head[d10].mean(axis=1)).mean()),
        },
    }
    common.write_json_atomic(path=common.REPORT_DIR / "checks.json", content=out)
    print(out)


if __name__ == "__main__":
    main()
