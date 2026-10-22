"""Score experiment 204 and print every number the write-up quotes.

Writes `report.json` under the work folder. The headline is C1 (bottom-up minus top-down) at each
BSP over leads up to 168 hours in the scored year, at the 97.5% level of the month-resampled paired
bootstrap, with the decision rule of the plan.

    uv run --no-sync python ml-experiments/00204-*/e00204_report.py
"""

from typing import Final, cast

import e00204_common as common
import e00204_score as score
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level

LEVEL_PERCENT: Final[float] = 97.5
N_RESAMPLES: Final[int] = 20_000
THRESHOLD_SHARE: Final[float] = 0.05
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "0.5 to 24 h": (0.0, 24.0),
    "24 to 72 h": (24.0, 72.0),
    "72 to 168 h": (72.0, 168.0),
}
REPORT_PATH: Final = common.WORK_DIR / "report.json"


def _mean(series: pl.Series) -> float:
    """Return a numeric series' mean as a float."""
    return cast(float, series.mean())


def contrast(*, losses: pl.DataFrame, treatment: str, reference: str, metric: str = "qs") -> dict:
    """Return a paired contrast with its month-resampled interval and its mean-error companion."""
    t = losses.filter(pl.col("arm") == treatment)
    r = losses.filter(pl.col("arm") == reference)
    lower, upper = bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric=metric,
        level=LEVEL_PERCENT,
        n_resamples=N_RESAMPLES,
    )
    by_month = (
        t.group_by("month")
        .agg(t=pl.col(metric).mean())
        .join(r.group_by("month").agg(r=pl.col(metric).mean()), on="month")
    )
    return {
        "estimate": _mean(t[metric]) - _mean(r[metric]),
        "lower": float(lower),
        "upper": float(upper),
        "mae_mean_estimate": _mean(t["abs_mean"]) - _mean(r["abs_mean"]),
        "months_negative": int((by_month["t"] < by_month["r"]).sum()),
        "n_months": by_month.height,
        "reference_score": _mean(r[metric]),
    }


def arm_table(*, losses: pl.DataFrame) -> dict:
    """Return each arm's absolute scores: overall, by lead band, and the tail table."""
    table = {}
    for arm in losses["arm"].unique().sort():
        part = losses.filter(pl.col("arm") == arm)
        row = {
            "rows": part.height,
            "qs": _mean(part["qs"]),
            "abs_mean": _mean(part["abs_mean"]),
            "median_minus_mean": _mean(part["q50"] - part["mean"]),
        }
        for name, (low, high) in LEAD_BANDS.items():
            band = part.filter((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))
            row[f"qs {name}"] = _mean(band["qs"])
        for quantile, _ in score.TAIL_QUANTILES:
            row[f"cov_{quantile}"] = _mean(part[f"cov_{quantile}"])
            row[f"pin_{quantile}"] = _mean(part[f"pin_{quantile}"])
        table[arm] = row
    return table


def covered_share(*, cache, bsp: str, children: tuple[str, ...]) -> float:  # noqa: ANN001
    """Return the share of the BSP's effective capacity that the children make up."""
    cap = cache.sites[bsp].table.effective_capacity_mw
    return sum(cache.sites[c].table.effective_capacity_mw for c in children) / cap


def main() -> None:
    """Score every arm, apply the plan's decision rule, and write `report.json`."""
    cache = common.load_cache()
    report: dict = {"weights": {}, "covered_share": {}, "scored": {}, "replicate": {}}
    for bsp in common.BSPS:
        report["weights"][bsp] = {
            v: common.fit_weights(
                cache=cache, bsp=bsp, end=common.WEIGHT_WINDOW_END, variant=v, bounded=True
            )
            for v in common.VARIANT_SUFFIX
        }
        report["weights"][bsp]["fold_unbounded"] = common.fit_weights(
            cache=cache, bsp=bsp, end=common.WEIGHT_WINDOW_END, variant="fold", bounded=False
        )
        report["covered_share"][bsp] = {
            "b_partial": covered_share(cache=cache, bsp=bsp, children=common.MW_COVERED[bsp]),
            "c_d": covered_share(cache=cache, bsp=bsp, children=common.BSPS[bsp]),
        }
    for period in ("scored", "replicate"):
        for bsp in common.BSPS:
            losses = score.loss_frame(bsp=bsp, period=period)
            out = {"arms": arm_table(losses=losses), "contrasts": {}}
            pairs = {
                "C1_d_minus_a": ("d", "a_td"),
                "C1_c_minus_a": ("c", "a_td"),
                "C2_d_minus_c": ("d", "c"),
                "C2_dp50_minus_c": ("dp50", "c"),
                "C2_dp50_minus_d": ("dp50", "d"),
                "C3_c_minus_b_tdspread": ("c", "b_partial_tdspread"),
                "d_minus_b_tdspread": ("d", "b_partial_tdspread"),
                "b_partial_minus_a": ("b_partial", "a_td"),
                "b_tdspread_minus_a": ("b_partial_tdspread", "a_td"),
                "C7_oracle_minus_a": ("oracle", "a_td"),
            }
            for name, (treatment, reference) in pairs.items():
                out["contrasts"][name] = contrast(
                    losses=losses, treatment=treatment, reference=reference
                )
            for band, (low, high) in LEAD_BANDS.items():
                part = losses.filter((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))
                out["contrasts"][f"C4_d_minus_a {band}"] = contrast(
                    losses=part, treatment="d", reference="a_td"
                )
            for quantile, _ in score.TAIL_QUANTILES:
                out["contrasts"][f"C6_d_minus_a {quantile}"] = contrast(
                    losses=losses, treatment="d", reference="a_td", metric=f"pin_{quantile}"
                )
            report[period][bsp] = out
            print(period, bsp, "done", flush=True)
    common.write_json_atomic(path=REPORT_PATH, content=report)
    for bsp in common.BSPS:
        for name in ("C1_d_minus_a", "C2_d_minus_c"):
            c = report["scored"][bsp]["contrasts"][name]
            print(bsp, name, f"{c['estimate']:+.3f} ({c['lower']:+.3f} to {c['upper']:+.3f})")


if __name__ == "__main__":
    main()
