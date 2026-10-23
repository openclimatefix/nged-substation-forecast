"""Experiment 304: why the member-wise sum covers more than its nominal level, and the checks.

Compares the correlation the ensemble's ranks imply between farms with the correlation of the
farms' residual levels, and prints the median-minus-mean check per farm. Run with
`OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 uv run --no-sync python e00304_diagnose.py`.
"""

import e00304_data as data
import e00304_run as run
import numpy as np
import polars as pl
from scipy.stats import norm, rankdata
from studies.quantile_sums import pit_normal_scores


def ensemble_correlation(*, members: list[np.ndarray]) -> np.ndarray:
    """Return, per row, the mean pairwise correlation of the farms' member normal scores."""
    n_members = members[0].shape[1]
    scores = []
    for member in members:
        z = norm.ppf((rankdata(member, axis=1, method="average") - 0.5) / n_members)
        z = (z - z.mean(axis=1, keepdims=True)) / np.maximum(z.std(axis=1, keepdims=True), 1e-9)
        scores.append(z)
    pairs = [(i, j) for i in range(len(scores)) for j in range(i + 1, len(scores))]
    return np.mean([(scores[i] * scores[j]).mean(axis=1) for i, j in pairs], axis=0)


def mean_forecast(*, quantiles: np.ndarray) -> np.ndarray:
    """Return the mean of the quantile function between 0.01 and 0.99 (trapezoid over levels)."""
    levels = np.asarray(data.LEVELS)
    widths = np.diff(levels)
    mids = (quantiles[:, 1:] + quantiles[:, :-1]) / 2.0
    return (mids * widths).sum(axis=1) / widths.sum()


def main() -> None:
    """Print the two diagnostics."""
    rows = []
    checks = []
    for node, sites in data.NODES.items():
        farms = data.node_rows(node=node, arm="blend")
        band = run.csi_band(csi=np.mean([f.csi for f in farms], axis=0))
        implied = ensemble_correlation(members=[f.members for f in farms])
        scores = np.column_stack(
            [
                pit_normal_scores(
                    quantiles=f.quantiles, levels=data.LEVELS, actual=f.actual, seed=run.SEED + i
                )
                for i, f in enumerate(farms)
            ]
        )
        corr = np.corrcoef(scores.T)
        fitted = corr[np.triu_indices(3, k=1)].mean()
        for b, name in enumerate(run.CSI_BANDS):
            use = band == b
            c = np.corrcoef(scores[use].T)
            rows.append(
                {
                    "node": node,
                    "band": name,
                    "rows": int(use.sum()),
                    "ensemble_rank_correlation": float(implied[use].mean()),
                    "residual_level_correlation": float(c[np.triu_indices(3, k=1)].mean()),
                }
            )
        rows.append(
            {
                "node": node,
                "band": "all",
                "rows": len(band),
                "ensemble_rank_correlation": float(implied.mean()),
                "residual_level_correlation": float(fitted),
            }
        )
        for site, farm in zip(sites, farms, strict=True):
            median = farm.quantiles[:, data.LEVELS.index(0.5)]
            mean = mean_forecast(quantiles=farm.quantiles)
            checks.append(
                {
                    "site": site,
                    "median_minus_mean_mw": float((median - mean).mean()),
                    "capacity_mw": farm.capacity_mw,
                }
            )
    pl.DataFrame(rows).write_csv(data.DATA_DIR / "dependence_diagnostic.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=160):
        print("## Correlation of farm levels: implied by the ensemble ranks versus the residuals")
        print(
            pl.DataFrame(rows).with_columns(
                pl.col("ensemble_rank_correlation", "residual_level_correlation").round(3)
            )
        )
        print("\n## Check: median minus mean forecast per farm (blend arm, MW)")
        print(pl.DataFrame(checks).with_columns(pl.col("median_minus_mean_mw").round(4)))


if __name__ == "__main__":
    main()
