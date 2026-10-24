# ruff: noqa: E501
"""Score the stored forecasts of the Primaries and the bulk supply points with the curtailment metric.

Part A scores experiment 251's six-Primary rows (outside the hand-filtered load shifts) with MAE,
CRPS, twCRPS and the node-only curtailment metric, with month-resampled intervals and a
detectability table. Part B scores the bulk supply points D13 and D17. Part C scores experiment
163's rows, which include the logged transfers, split into the row groups of the decision program.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00277-*/e00277_score.py
"""

import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Final

import e00277_common as common
import numpy as np
import polars as pl
from studies.node_curtailment import (
    NodeCurtailmentAssumptions,
    NodeTable,
    month_total_arrays,
    node_curtailment,
    scaled_limits,
    table_with_scaled_limits,
)

METRICS: Final[tuple[str, ...]] = ("qs", "mae", "mae_mean", "tw95", "tw99")
CONTRAST_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("climatology", "arm1"),
    ("manual_heuristic", "arm1"),
    ("bg2", "arm1"),
    ("climatology", "bg2"),
    ("manual_heuristic", "bg2"),
    ("climatology", "manual_heuristic"),
)
"""(worse expected, better expected): the contrast is the first arm minus the second."""
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-12 h", 0.0, 12.0),
    ("12-36 h", 12.0, 36.0),
    ("36 h-7 d", 36.0, 169.0),
)
COST_COLUMN: Final[str] = "energy_proxy_mwh_equivalent"
ENERGY_EVENT_MIN_HALF_HOURS: Final[int] = 1


def block_bootstrap(
    *, sums: np.ndarray, counts: np.ndarray, rng: np.random.Generator, ratio: bool
) -> dict[str, float]:
    """Resample whole blocks of per-block sums; return the estimate, 95% interval, and std error.

    With `ratio` true the statistic is the mean per row (sum of drawn sums over sum of drawn
    counts). With `ratio` false it is the total over the drawn blocks, which has the scale of one
    full pass over the blocks.
    """
    n = len(sums)
    if n == 0:
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "se": float("nan")}
    draws = rng.integers(0, n, size=(common.N_RESAMPLES, n))
    drawn_sums = sums[draws].sum(axis=1)
    if ratio:
        resampled = drawn_sums / counts[draws].sum(axis=1)
        est = float(sums.sum() / counts.sum())
    else:
        resampled = drawn_sums
        est = float(sums.sum())
    lo, hi = np.percentile(resampled, [2.5, 97.5])
    return {"est": est, "lo": float(lo), "hi": float(hi), "se": float(resampled.std())}


def blocks_of(*, values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each block's sum of `values` and row count, blocks in sorted label order."""
    unique, codes = np.unique(labels, return_inverse=True)
    sums = np.bincount(codes, weights=values, minlength=len(unique))
    counts = np.bincount(codes, minlength=len(unique)).astype(np.float64)
    return sums, counts


def mean_with_interval(
    *, values: np.ndarray, labels: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    """Return the mean of `values` with a block-resampled interval; NaNs are left out."""
    keep = np.isfinite(values)
    if not keep.any():
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "se": float("nan")}
    sums, counts = blocks_of(values=values[keep], labels=labels[keep])
    return block_bootstrap(sums=sums, counts=counts, rng=rng, ratio=True)


def census(*, node_frame: pl.DataFrame, limits: pl.DataFrame) -> list[dict]:
    """Count scored half-hours and half-hours above the scaled limit, per node (12-36 h leads)."""
    joined = node_frame.join(limits, left_on="node_id", right_on="node_id")
    return (
        joined.group_by("node_id")
        .agg(
            n_half_hours=pl.len(),
            n_above_limit=(pl.col("actual") > pl.col("export_limit_mw")).sum(),
            limit_mw=pl.col("export_limit_mw").first(),
        )
        .sort("node_id")
        .to_dicts()
    )


def run_cost(
    *,
    table: NodeTable,
    flows: pl.DataFrame,
    frames: Mapping[str, pl.DataFrame],
    fraction: float,
    lead_min: float = 12.0,
    lead_max: float = 36.0,
) -> tuple[dict[str, dict[str, np.ndarray]], pl.DataFrame, pl.DataFrame]:
    """Score the arms at one scaled limit; return per-arm month arrays, per-arm totals, limits."""
    scaled = scaled_limits(table=table, flows=flows, cutoff=common.SCORED_START, fraction=fraction)
    result = node_curtailment(
        table=table_with_scaled_limits(table=table, scaled=scaled, fraction=fraction),
        node_forecasts=dict(frames),
        generator_forecasts={},
        assumptions=replace(
            NodeCurtailmentAssumptions(use_generators=False),
            lead_min_hours=lead_min,
            lead_max_hours=lead_max,
            headline_price_gbp_per_mwh=common.PRICE_GBP_PER_MWH,
        ),
    )
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for arm in frames:
        sums, counts = month_total_arrays(
            month_totals=result.month_totals, arm=arm, column=COST_COLUMN
        )
        arrays[arm] = {"sums": sums[0] * common.PRICE_GBP_PER_MWH, "counts": counts}
    per_arm = result.per_arm.filter(pl.col("is_headline_price"))
    return arrays, per_arm, scaled


def cost_summary(
    *, arrays: Mapping[str, Mapping[str, np.ndarray]], rng: np.random.Generator
) -> dict[str, Any]:
    """Return total cost, avoidable cost, and contrasts, with month-resampled intervals, in GBP."""
    out: dict[str, Any] = {"arms": {}, "contrasts": {}}
    perfect = arrays["perfect"]["sums"]
    counts = arrays["perfect"]["counts"]
    for arm, array in arrays.items():
        if arm == "perfect":
            continue
        total = block_bootstrap(sums=array["sums"], counts=counts, rng=rng, ratio=False)
        avoid = block_bootstrap(sums=array["sums"] - perfect, counts=counts, rng=rng, ratio=False)
        out["arms"][arm] = {"cost": total, "avoidable": avoid}
    for worse, better in CONTRAST_PAIRS:
        if worse in arrays and better in arrays:
            diff = arrays[worse]["sums"] - arrays[better]["sums"]
            out["contrasts"][f"{worse}-{better}"] = block_bootstrap(
                sums=diff, counts=counts, rng=rng, ratio=False
            )
    out["perfect_cost"] = float(perfect.sum())
    out["n_node_half_hours"] = float(counts.sum())
    return out


def detectability(*, interval: Mapping[str, float], reference: float) -> dict[str, float]:
    """Return the smallest paired difference the interval design resolves (80% power, 5% level)."""
    mde = common.POWER_FOR_80_PERCENT * interval["se"]
    return {
        "se": interval["se"],
        "mde": mde,
        "mde_relative_percent": 100.0 * mde / reference if reference else float("nan"),
    }


def main() -> None:  # noqa: PLR0912, PLR0915
    """Run part A and write the report."""
    started = time.time()
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    capacity = common.capacity_mw()
    report: dict = {}

    # ------------------------------------------------------------------ part A: Primaries
    rows, arms = common.read_primary_rows()
    sites = rows["site"].to_numpy()
    series = rows["time_series_id"].to_numpy()
    edges_by_series = common.thresholds(sites=common.PRIMARIES, capacity=capacity)
    edges = {
        label: np.array([edges_by_series[int(s)][label] for s in series]) for label in ("95", "99")
    }
    means = common.stored_means(rows=rows)
    actual = rows["actual"].to_numpy().astype(np.float64)
    per_row = {
        arm: common.per_row_scores(actual=actual, q=arms[arm], mean=means.get(arm), edges=edges)
        for arm in common.PRIMARY_ARMS
    }
    init_month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    valid_month = rows["valid_time"].dt.month().to_numpy()
    lead = rows["lead_hours"].to_numpy()
    winter = np.isin(valid_month, list(WINTER_MONTHS))
    groups: dict[str, np.ndarray] = {
        "G0 all rows": np.ones(rows.height, dtype=bool),
        "season: November-March": winter,
        "season: April-October": ~winter,
        **{f"lead {name}": (lead >= lo) & (lead < hi) for name, lo, hi in LEAD_BANDS},
        **{f"substation {s}": sites == s for s in common.PRIMARIES},
    }
    print(f"rows {rows.height}, {time.time() - started:.0f} s", flush=True)

    metrics: dict = {}
    for arm in common.PRIMARY_ARMS:
        metrics[arm] = {}
        for gname, mask in groups.items():
            metrics[arm][gname] = {
                metric: mean_with_interval(
                    values=per_row[arm][metric][mask], labels=init_month[mask], rng=rng
                )
                for metric in METRICS
            }
    contrasts: dict = {}
    for worse, better in CONTRAST_PAIRS:
        contrasts[f"{worse}-{better}"] = {}
        for gname, mask in groups.items():
            contrasts[f"{worse}-{better}"][gname] = {
                metric: mean_with_interval(
                    values=per_row[worse][metric][mask] - per_row[better][metric][mask],
                    labels=init_month[mask],
                    rng=rng,
                )
                for metric in METRICS
            }
    report["A_metrics"] = metrics
    report["A_contrasts"] = contrasts
    report["A_rows"] = {
        "n_rows": rows.height,
        "n_by_site": {s: int((sites == s).sum()) for s in common.PRIMARIES},
        "n_12_36h": int(groups["lead 12-36 h"].sum()),
    }
    # only arm 1 has a stored mean; the crossing share is a check on the stored quantiles
    report["A_crossing_share"] = {
        arm: float((np.diff(arms[arm].astype(np.float64), axis=1) < 0).any(axis=1).mean())
        for arm in common.PRIMARY_ARMS
    }
    report["A_median_minus_mean"] = {
        arm: {
            s: float(np.nanmean((arms[arm][:, 5] - means[arm])[sites == s]))
            for s in common.PRIMARIES
        }
        for arm in means
    }
    report["A_mae_of_mean_vs_median"] = {
        arm: {
            m: float(
                np.nanmean(per_row[arm][m]) if np.isfinite(per_row[arm][m]).any() else float("nan")
            )
            for m in ("mae", "mae_mean")
        }
        for arm in common.PRIMARY_ARMS
    }

    # curtailment on the Primaries
    table = common.node_table(sites=common.PRIMARIES)
    flows = common.read_flows(series=list(common.PRIMARIES.values()))
    frames = {
        arm: common.quantile_frame(rows=rows, quantiles=arms[arm], capacity=capacity)
        for arm in common.PRIMARY_ARMS
    }
    frames["perfect"] = frames["bg2"].with_columns(
        *[pl.col("actual").alias(name) for name in common.QUANTILES]
    )
    cost: dict = {}
    for fraction in common.FRACTIONS:
        arrays, _, scaled = run_cost(table=table, flows=flows, frames=frames, fraction=fraction)
        key = str(fraction)
        cost[key] = {"all": cost_summary(arrays=arrays, rng=rng)}
        lim = scaled.rename({"export_limit_mw": "limit_mw"})
        cost[key]["limits_mw"] = lim.select("node_id", "limit_mw").to_dicts()
        # events: scored half-hours (12-36 h leads) above the scaled limit
        scored = frames["perfect"].filter(pl.col("valid_time") >= common.SCORED_START)
        scored = scored.with_columns(
            lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours()
        ).filter(pl.col("lead") >= 12, pl.col("lead") < 36)
        names = {v: k for k, v in common.PRIMARIES.items()}
        events = scored.with_columns(
            node_id=pl.col("time_series_id").replace_strict(names, return_dtype=pl.String)
        )
        cost[key]["census"] = census(node_frame=events, limits=scaled)
        # G6-style cuts: season, lead band, substation (cost only)
        cuts: dict = {}
        for gname, mask in groups.items():
            if gname == "G0 all rows":
                continue
            part = {a: f.filter(pl.Series(mask)) for a, f in frames.items()}
            if gname.startswith("lead "):
                lo, hi = next((low, high) for n, low, high in LEAD_BANDS if f"lead {n}" == gname)
            else:
                lo, hi = 12.0, 36.0
            arrays_g, _, _ = run_cost(
                table=table, flows=flows, frames=part, fraction=fraction, lead_min=lo, lead_max=hi
            )
            cuts[gname] = cost_summary(arrays=arrays_g, rng=rng)
        cost[key]["cuts"] = cuts
        print(f"cost f={fraction} done at {time.time() - started:.0f} s", flush=True)
    # leave one substation out: the BG2-minus-stack and climatology-minus-stack cost contrasts
    loo: dict = {}
    for fraction in (0.5, 0.75):
        loo[str(fraction)] = {}
        for left_out in common.PRIMARIES:
            keep = sites != left_out
            part = {a: f.filter(pl.Series(keep)) for a, f in frames.items()}
            sub_table = common.node_table(
                sites={k: v for k, v in common.PRIMARIES.items() if k != left_out}
            )
            arrays_l, _, _ = run_cost(table=sub_table, flows=flows, frames=part, fraction=fraction)
            summary = cost_summary(arrays=arrays_l, rng=rng)
            loo[str(fraction)][left_out] = {
                k: summary["contrasts"][k]
                for k in ("bg2-arm1", "climatology-arm1", "manual_heuristic-bg2")
            }
    report["A_leave_one_out"] = loo
    report["A_cost"] = cost

    # detectability on the pooled year (12-36 h leads for cost; all leads for the score metrics)
    det: dict = {"cost": {}, "score": {}}
    f_key = str(common.HEADLINE_FRACTION)
    for pair, interval in cost[f_key]["all"]["contrasts"].items():
        worse = pair.split("-")[0]
        reference = cost[f_key]["all"]["arms"][worse]["avoidable"]["est"]
        det["cost"][pair] = {
            **detectability(interval=interval, reference=reference),
            "observed": interval["est"],
        }
    for pair, by_group in contrasts.items():
        worse = pair.split("-")[0]
        det["score"][pair] = {
            metric: {
                **detectability(
                    interval=by_group["G0 all rows"][metric],
                    reference=metrics[worse]["G0 all rows"][metric]["est"],
                ),
                "observed": by_group["G0 all rows"][metric]["est"],
            }
            for metric in METRICS
        }
    report["A_detectability"] = det

    common.write_json_atomic(path=common.OUT_DIR / "report_a.json", content=report)
    print(f"done at {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
