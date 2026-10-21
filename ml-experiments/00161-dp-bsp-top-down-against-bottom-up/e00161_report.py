"""Score experiment 161 and print every number the write-up quotes.

Writes `report.json` under the work folder. The headline is C1 (bottom-up minus top-down) at each
BSP over leads up to 168 hours in the scored year, at the 97.5% level of the month-resampled paired
bootstrap, with the decision rule of the plan.

    uv run --no-sync python ml-experiments/00161-*/e00161_report.py
"""

import json
from typing import Final, cast

import e00161_common as common
import e00161_refs as refs
import e00161_score as score
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level

LEVEL_PERCENT: Final[float] = 97.5
N_RESAMPLES: Final[int] = 20_000
THRESHOLD_SHARE: Final[float] = 0.02
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


def verdict(*, c1: dict[str, dict]) -> str:
    """Apply the plan's decision rule to the C1 intervals of the two BSPs."""
    below_threshold = [v["upper"] < -THRESHOLD_SHARE * v["reference_score"] for v in c1.values()]
    below_zero = [v["upper"] < 0 for v in c1.values()]
    above_zero = [v["lower"] > 0 for v in c1.values()]
    if all(below_threshold):
        return "supported"
    if any(below_threshold):
        return "mixed" if any(above_zero) else "partly supported"
    if all(below_zero):
        return "small gain"
    if all(above_zero):
        return "refuted"
    return "inconclusive"


def oracle_control(*, c7: dict[str, dict], c1: dict[str, dict], share: dict[str, float]) -> dict:
    """Return the oracle control: a gain of at least half the covered share times TD's score."""
    out = {}
    for bsp in c7:
        needed = -0.5 * share[bsp] * c7[bsp]["reference_score"]
        out[bsp] = {
            "gain": c7[bsp]["estimate"],
            "needed": needed,
            "passes": bool(c7[bsp]["estimate"] <= needed),
        }
    return out


def error_decomposition(*, bsp: str, cache, period: str = "scored") -> dict:  # noqa: ANN001
    """Return the standard deviations (points) of the top-down error and of its two parts."""
    import e00161_bu as bu

    weights = common.weights_for(cache=cache, bsp=bsp, variant="learned")
    cap = cache.sites[bsp].table.effective_capacity_mw
    frame = score.scorable(frame=bu.joined_period(bsp=bsp, suffix="r", period=period))
    error_r = (frame["actual_r"] - frame["mean_r"]).to_numpy().astype(np.float64)
    children = np.zeros(len(frame))
    for child, weight in weights.items():
        scale = cache.sites[child].table.effective_capacity_mw / cap
        children += weight * scale * (frame[f"actual_{child}"] - frame[f"mean_{child}"]).to_numpy()
    total = (frame["actual"] - frame["mean"]).to_numpy().astype(np.float64)
    return {
        "sd_top_down_error": float(total.std()),
        "sd_remainder_error": float(error_r.std()),
        "sd_children_error": float(children.std()),
        "corr_remainder_children": float(np.corrcoef(error_r, children)[0, 1]),
        "sd_bottom_up_error": float((error_r + children).std()),
    }


def main() -> None:
    """Compute and print the report."""
    cache = common.load_cache()
    report: dict = {"weights": {}, "closure": {}, "share": {}, "periods": {}}
    for bsp in common.BSPS:
        report["weights"][bsp] = {
            v: common.weights_for(cache=cache, bsp=bsp, variant=v) for v in common.VARIANT_SUFFIX
        }
        report["closure"][bsp] = common.fit_weights(
            cache=cache, bsp=bsp, end=common.WEIGHT_WINDOW_END
        )
        report["share"][bsp] = float(
            sum(cache.sites[c].table.effective_capacity_mw for c in common.BSPS[bsp])
            / cache.sites[bsp].table.effective_capacity_mw
        )
    for period in ("replicate", "scored"):
        result: dict = {}
        losses = {bsp: score.loss_frame(bsp=bsp, period=period) for bsp in common.BSPS}
        result["arms"] = {bsp: arm_table(losses=frame) for bsp, frame in losses.items()}
        result["C1"] = {
            bsp: contrast(losses=f, treatment="bu_r", reference="td") for bsp, f in losses.items()
        }
        result["C2"] = {
            bsp: contrast(losses=f, treatment="bu_s", reference="td") for bsp, f in losses.items()
        }
        result["C3"] = {
            bsp: contrast(losses=f, treatment="bu_r", reference="bu_s") for bsp, f in losses.items()
        }
        result["C4"] = {
            bsp: {
                name: contrast(
                    losses=f.filter((pl.col("lead_hours") > lo) & (pl.col("lead_hours") <= hi)),
                    treatment="bu_r",
                    reference="td",
                )
                for name, (lo, hi) in LEAD_BANDS.items()
            }
            for bsp, f in losses.items()
        }
        result["C5"] = {
            bsp: contrast(losses=f, treatment="bu_r_tdspread", reference="td")
            for bsp, f in losses.items()
        }
        result["C6"] = {
            bsp: {
                q: contrast(losses=f, treatment="bu_r", reference="td", metric=f"pin_{q}")
                for q, _ in score.TAIL_QUANTILES
            }
            for bsp, f in losses.items()
        }
        result["C7"] = {
            bsp: contrast(losses=f, treatment="bu_oracle", reference="td")
            for bsp, f in losses.items()
        }
        result["free_minus_learned"] = {
            bsp: contrast(losses=f, treatment="bu_f", reference="bu_r") for bsp, f in losses.items()
        }
        report["periods"][period] = result
    scored = report["periods"]["scored"]
    report["verdict_rule"] = verdict(c1=scored["C1"])
    report["oracle_control"] = oracle_control(
        c7=scored["C7"], c1=scored["C1"], share=report["share"]
    )
    report["error_decomposition"] = {
        bsp: error_decomposition(bsp=bsp, cache=cache) for bsp in common.BSPS
    }
    # References and the top-down arm on all of its own rows.
    report["references"] = {}
    report["td_all_rows"] = {}
    for bsp in common.BSPS:
        td = score.scorable(frame=score.read_arm(arm="td", bsp=bsp, period="scored"))
        mh = refs.manual_heuristic(cache=cache, bsp=bsp, frame=td)
        keep = td.join(mh.select(score.KEYS), on=score.KEYS, how="inner")
        report["references"][bsp] = {
            "rows": keep.height,
            "td": _mean(score.with_losses(frame=keep)["qs"]),
            "climatology": _mean(
                score.with_losses(frame=refs.climatology(cache=cache, bsp=bsp, frame=keep))["qs"]
            ),
            "manual_heuristic": _mean(
                score.with_losses(
                    frame=mh.join(keep.select(score.KEYS), on=score.KEYS, how="inner")
                )["qs"]
            ),
        }
        own = score.scorable(
            frame=pl.read_parquet(common.OUT_DIR / f"d_scored_{bsp}.parquet").with_columns(
                switching_mask=pl.col("switching_mask")
            )
        )
        report["td_all_rows"][bsp] = {
            "rows": own.height,
            "qs": _mean(score.with_losses(frame=own)["qs"]),
        }
    common.write_json_atomic(path=REPORT_PATH, content=report)
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
