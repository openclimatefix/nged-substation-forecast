"""Leave-one-substation-out check of the cold-night q95 coverage and the all-row contrasts.

Reads the same arms as `e00164_report.py` and writes `work/report/leave_one_out.json`.

    uv run --no-sync python ml-experiments/00164-*/e00164_leave_one_out.py
"""

import e00164_common as common
import e00164_report as report
import numpy as np
import polars as pl

N_BOOTSTRAP = 5000


def day_block_interval(*, cold: pl.DataFrame, arm: str, arm_minus: str | None = None) -> dict:
    """Return the pooled cold q95 coverage with a bootstrap over valid dates (day blocks)."""
    part = cold.filter(pl.col("arm") == arm).with_columns(day=pl.col("valid_time").dt.date())
    days = part.group_by("day").agg(hits=pl.col("cov_q95").sum(), rows=pl.len()).sort("day")
    hits = days["hits"].to_numpy().astype(float)
    rows = days["rows"].to_numpy().astype(float)
    draws = np.random.default_rng(0).integers(0, len(days), size=(N_BOOTSTRAP, len(days)))
    coverage = hits[draws].sum(axis=1) / rows[draws].sum(axis=1)
    return {
        "arm": arm,
        "coverage": float(hits.sum() / rows.sum()),
        "low": float(np.quantile(coverage, 0.025)),
        "high": float(np.quantile(coverage, 0.975)),
        "p_at_least_0_90": float((coverage >= 0.90).mean()),
        "n_days": len(days),
    }


def main() -> None:
    """Print the per-substation cold-night coverage and the leave-one-out pooled figures."""
    cache = common.load_cache()
    out: dict = {}
    for period in ("scored", "replicate"):
        loaded = {a: report.load_ours(arm=a, period=period, cache=cache) for a in report.OURS}
        losses = report.stack(frames=report.common_rows(frames=loaded), cache=cache)
        decision = losses.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
        cold = decision.filter(report.subsets()[report.COLD_KEY])
        per_site = (
            cold.group_by(["arm", "site"])
            .agg(cov_q95=pl.col("cov_q95").mean(), cov_q99=pl.col("cov_q99").mean(), rows=pl.len())
            .sort(["arm", "site"])
        )
        leave_out: list[dict] = []
        for site in common.DECISION_SITES:
            rest = cold.filter(pl.col("site") != site)
            pooled = rest.group_by("arm").agg(cov_q95=pl.col("cov_q95").mean())
            score = (
                decision.filter(pl.col("site") != site).group_by("arm").agg(qs=pl.col("qs").mean())
            )
            leave_out.append(
                {"left_out": site, "cold_q95": pooled.to_dicts(), "all_row_qs": score.to_dicts()}
            )
        blocks = [day_block_interval(cold=cold, arm=a) for a in report.OURS]
        print(period, blocks, flush=True)
        out[period] = {
            "per_site": per_site.to_dicts(),
            "leave_one_out": leave_out,
            "day_blocks": blocks,
        }
        print(period, per_site.filter(pl.col("arm").is_in(["ref", "o", "t"])), flush=True)
        for row in leave_out:
            cov = {r["arm"]: round(r["cov_q95"], 3) for r in row["cold_q95"]}
            qs = {r["arm"]: round(r["qs"], 4) for r in row["all_row_qs"]}
            print(period, row["left_out"], cov, qs, flush=True)
    common.write_json_atomic(path=common.WORK_DIR / "report" / "leave_one_out.json", content=out)


if __name__ == "__main__":
    main()
