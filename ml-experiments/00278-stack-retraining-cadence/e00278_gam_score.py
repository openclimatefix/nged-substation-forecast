"""Score the convex-model refit cadences (arm D on monthly, weekly, and quarterly refits).

Gate first: the rebuilt monthly arm D must reproduce experiment 155's stored arm D (mean within
1e-3 points and quantile score within 0.005). Then the planned contrasts C1 (quarterly minus
monthly) and C2 (weekly minus monthly), the stop rule (sites where monthly and quarterly differ by
under 0.01 points), and the score by months since the last quarterly refit, winter and summer.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00278-*/e00278_gam_score.py
"""

import numpy as np
import polars as pl

import e00278_common as common  # isort: skip

CADENCES: dict[str, str] = {"monthly": "a2_d", "weekly": "a3_d", "quarterly": "a4_d"}
SEED = 278


def read_arm(*, arm: str, directory, prefix: str = "") -> pl.DataFrame:  # noqa: ANN001
    """Return an arm's scored-year frames, all six substations, with a `site` column."""
    pieces = [
        pl.read_parquet(directory / f"{arm}_scored_{site}.parquet").with_columns(site=pl.lit(site))
        for site in common.SCORED_SITES
    ]
    return pl.concat(pieces)


def aligned(*, rows: pl.DataFrame, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the frame's quantiles and mean on the rows' order, `NaN` where missing."""
    joined = (
        rows.select(common.KEYS)
        .with_row_index("i")
        .join(
            frame.select([*common.KEYS, *common.QUANTILES, "mean"]).unique(subset=common.KEYS),
            on=common.KEYS,
            how="left",
            validate="1:1",
        )
        .sort("i")
    )
    return (
        joined.select(list(common.QUANTILES)).to_numpy().astype(np.float64),
        joined["mean"].to_numpy().astype(np.float64),
    )


def main() -> None:
    """Gate, score, and write the report."""
    rows = pl.read_parquet(common.E220_ARMS / "rows.parquet").filter(pl.col("lead_hours") <= 168.0)
    arms: dict[str, np.ndarray] = {}
    means: dict[str, np.ndarray] = {}
    for name, arm in CADENCES.items():
        arms[name], means[name] = aligned(
            rows=rows, frame=read_arm(arm=arm, directory=common.PREDS_DIR)
        )
    stored_q, stored_mean = aligned(rows=rows, frame=read_arm(arm="d", directory=common.E155_PREDS))
    keep = np.ones(rows.height, dtype=bool)
    for q in (*arms.values(), stored_q):
        keep &= np.isfinite(q).all(axis=1)
    rows = rows.filter(pl.Series(keep))
    arms = {k: v[keep] for k, v in arms.items()}
    means = {k: v[keep] for k, v in means.items()}
    stored_q, stored_mean = stored_q[keep], stored_mean[keep]
    edges = common.edges_of(rows=rows)
    actual = rows["actual"].to_numpy().astype(np.float64)
    metrics = {k: common.per_row(actual=actual, q=v, edges=edges) for k, v in arms.items()}
    stored = common.per_row(actual=actual, q=stored_q, edges=edges)
    assert np.abs(means["monthly"] - stored_mean).max() < 1e-3, "monthly gate: mean differs"
    assert abs(metrics["monthly"]["qs"].mean() - stored["qs"].mean()) < 0.005, "monthly gate: score"
    report: dict = {"n_rows": int(rows.height)}
    report["gate"] = {
        "max_abs_mean_diff_monthly_vs_155": float(np.abs(means["monthly"] - stored_mean).max()),
        "qs_monthly_rebuilt": float(metrics["monthly"]["qs"].mean()),
        "qs_155_stored": float(stored["qs"].mean()),
    }
    months = rows["month"].to_numpy()
    origin_month = rows["power_fcst_init_time"].dt.month().to_numpy()
    rng = np.random.default_rng(SEED)
    everything = np.ones(rows.height, dtype=bool)
    winter = np.isin(origin_month, list(common.WINTER_MONTHS))
    summer = np.isin(origin_month, list(common.SUMMER_MONTHS))
    since = (origin_month - 1) % 3
    report["absolute"] = {
        name: {
            m: float(v[m].mean())
            for m in (
                "qs",
                "mae",
                "tw95",
                "tw99",
                "cov_q01",
                "cov_q05",
                "cov_q95",
                "cov_q99",
                "pin_q01",
                "pin_q05",
                "pin_q95",
                "pin_q99",
            )
        }
        for name, v in metrics.items()
    }
    contrasts = {
        "C1 quarterly minus monthly": ("quarterly", "monthly"),
        "C2 weekly minus monthly": ("weekly", "monthly"),
    }
    report["contrasts"] = {}
    for label, (a, b) in contrasts.items():
        d = metrics[a]["qs"] - metrics[b]["qs"]
        dm = metrics[a]["mae"] - metrics[b]["mae"]
        report["contrasts"][label] = {
            "qs_99": common.mean_interval(
                values=d, mask=everything, blocks=months, level=99.0, rng=rng
            ),
            "qs_95": common.mean_interval(
                values=d, mask=everything, blocks=months, level=95.0, rng=rng
            ),
            "mae_95": common.mean_interval(
                values=dm, mask=everything, blocks=months, level=95.0, rng=rng
            ),
            "winter_95": common.mean_interval(
                values=d, mask=winter, blocks=months, level=95.0, rng=rng
            ),
            "summer_95": common.mean_interval(
                values=d, mask=summer, blocks=months, level=95.0, rng=rng
            ),
            "by_site": {
                site: float(d[(rows["site"] == site).to_numpy()].mean())
                for site in common.SCORED_SITES
            },
            "by_months_since_refit": {
                str(k): {
                    "all": float(d[since == k].mean()),
                    "winter": float(d[(since == k) & winter].mean())
                    if ((since == k) & winter).any()
                    else None,
                    "summer": float(d[(since == k) & summer].mean())
                    if ((since == k) & summer).any()
                    else None,
                }
                for k in (0, 1, 2)
            },
            "by_month": {m: float(d[months == m].mean()) for m in sorted(set(months.tolist()))},
        }
    by_site = report["contrasts"]["C1 quarterly minus monthly"]["by_site"]
    report["stop_rule"] = {
        "sites_under_0.01": [s for s, v in by_site.items() if abs(v) < 0.01],
        "n_sites_under_0.01": sum(abs(v) < 0.01 for v in by_site.values()),
        "stop": sum(abs(v) < 0.01 for v in by_site.values()) >= 5,
    }
    common.write_json_atomic(path=common.REPORT_DIR / "gam_report.json", content=report)
    import json

    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
