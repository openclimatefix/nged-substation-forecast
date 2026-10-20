"""Aggregate experiment 113's cell files into coverage tables and the gate labels.

    uv run python ml-experiments/00113-*/e00113_summary.py

Writes `work/summary.parquet` (one row per window, truth, noise, arm, interval method, family),
`work/by_site.parquet`, `work/noise_stats.parquet`, and `work/gate.json` (the decision rule's
labels and verdict).
"""

import json
from typing import Final

import e00113_common as common
import numpy as np
import polars as pl

FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    "band": ("night", "morning", "midday", "evening"),
    "capacity": ("capacity",),
}
METHODS: Final[tuple[str, ...]] = ("i1", "i1_raw", "i1u", "i2", "sandwich")
N_CLUSTER_RESAMPLES: Final[int] = 2000
LOW, HIGH = 0.70, 0.90
PRECISE_WIDTH: Final[dict[str, float]] = {"band": 0.30, "capacity": 0.20}


def read_cells(*, tag: str) -> pl.DataFrame:
    """Read every result file of a stage (`main` or `control`)."""
    paths = [
        path
        for path in sorted(common.CELLS_DIR.glob(f"{tag}_*.parquet"))
        if not path.stem.endswith("_noise")
    ]
    return pl.concat([pl.read_parquet(path) for path in paths])


def with_intervals(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the lower and upper end of each interval method as columns `<method>_low/_up`."""
    z = common.INTERVAL_Z
    return frame.with_columns(
        family=pl.when(pl.col("functional") == "capacity")
        .then(pl.lit("capacity"))
        .otherwise(pl.lit("band")),
        i1_low=pl.col("i1_lo"),
        i1_up=pl.col("i1_hi"),
        i1_raw_low=pl.col("i1_raw_lo"),
        i1_raw_up=pl.col("i1_raw_hi"),
        i1u_low=pl.col("i1u_lo"),
        i1u_up=pl.col("i1u_hi"),
        i2_low=pl.col("point") - z * pl.col("i2_sd"),
        i2_up=pl.col("point") + z * pl.col("i2_sd"),
        sandwich_low=pl.col("point") - z * pl.col("sandwich_sd"),
        sandwich_up=pl.col("point") + z * pl.col("sandwich_sd"),
    )


def long_form(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Stack the methods into rows with `covers_truth`, `covers_noise_free`, and `width`."""
    parts = []
    for method in METHODS:
        low, up = pl.col(f"{method}_low"), pl.col(f"{method}_up")
        parts.append(
            frame.with_columns(
                method=pl.lit(method),
                covers_truth=(low <= pl.col("truth_value")) & (pl.col("truth_value") <= up),
                covers_noise_free=(low <= pl.col("noise_free_value"))
                & (pl.col("noise_free_value") <= up),
                width=up - low,
            )
        )
    return pl.concat(parts)


def clustered_interval(*, rows: pl.DataFrame, column: str, seed: int = 0) -> tuple[float, float]:
    """Return a 95% interval for a coverage mean, resampling sites and then draws within sites."""
    rng = np.random.default_rng(seed)
    by_site: dict[str, list[np.ndarray]] = {}
    for site, group in rows.group_by("site", maintain_order=True):
        by_draw = group.group_by("draw", maintain_order=True).agg(pl.col(column).mean())
        by_site[site[0]] = by_draw[column].to_numpy()  # ty: ignore[invalid-assignment]
    sites = list(by_site)
    means = np.empty(N_CLUSTER_RESAMPLES)
    for resample in range(N_CLUSTER_RESAMPLES):
        chosen = rng.choice(len(sites), size=len(sites))
        values = [rng.choice(by_site[sites[i]], size=len(by_site[sites[i]])) for i in chosen]
        means[resample] = np.concatenate(values).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarise(*, long: pl.DataFrame) -> pl.DataFrame:
    """Return coverage, errors, and widths per cell, interval method, and family."""
    keys = ["window", "truth", "noise", "arm", "method", "family"]
    rows = []
    for key, group in long.group_by(keys, maintain_order=True):
        per_draw = group.group_by(["site", "draw"]).agg(
            pl.col("covers_truth").mean(), pl.col("covers_noise_free").mean()
        )
        low, high = clustered_interval(rows=per_draw, column="covers_truth")
        relative_error = (group["point"] - group["truth_value"]) / group["truth_value"]
        relative_width = group["width"] / group["truth_value"]
        rows.append(
            dict(zip(keys, key, strict=True))
            | {
                "n_intervals": group.height,
                "coverage_truth": common.as_float(group["covers_truth"].mean()),
                "coverage_truth_low": low,
                "coverage_truth_high": high,
                "coverage_noise_free": common.as_float(group["covers_noise_free"].mean()),
                "median_signed_relative_error": common.as_float(relative_error.median()),
                "rms_relative_error": float(np.sqrt((relative_error**2).mean())),  # ty: ignore[no-matching-overload]
                "median_width": common.as_float(group["width"].median()),
                "median_relative_width": common.as_float(relative_width.median()),
                "median_abs_bias": common.as_float(
                    (group["noise_free_value"] - group["truth_value"]).abs().median()
                ),
                "median_active_bound_share": common.as_float(group["active_bound_share"].median()),
                "mean_boot_variance": common.as_float(
                    (group["boot_sd"] ** 2 * group["i1_inflate"] ** 2).mean()
                ),
                "mean_sandwich_variance": common.as_float((group["sandwich_sd"] ** 2).mean()),
            }
        )
    return pl.DataFrame(rows).with_columns(
        bias_over_width=pl.col("median_abs_bias") / pl.col("median_width"),
        tau=pl.col("mean_boot_variance") / pl.col("mean_sandwich_variance"),
    )


def by_site(*, long: pl.DataFrame) -> pl.DataFrame:
    """Return coverage per real site for the I1 interval, per cell and family."""
    return (
        long.filter(pl.col("method") == "i1")
        .group_by(["window", "truth", "noise", "arm", "family", "site"])
        .agg(
            pl.col("covers_truth").mean().alias("coverage_truth"),
            pl.col("covers_noise_free").mean().alias("coverage_noise_free"),
            (pl.col("width") / pl.col("truth_value")).median().alias("median_relative_width"),
        )
        .sort(["window", "truth", "noise", "arm", "family", "site"])
    )


def label(*, coverage_decision: float, coverage_iid: float) -> str:
    """Return the gate label of one family from its two I1 coverages."""
    if min(coverage_decision, coverage_iid) < LOW:
        return "overconfident"
    if max(coverage_decision, coverage_iid) > HIGH:
        return "too wide"
    return "calibrated"


def verdict(*, labels: dict[str, str]) -> str:
    """Return the verdict the decision rule gives for the two family labels."""
    values = set(labels.values())
    if "overconfident" in values:
        return "refuted"
    if values == {"calibrated"}:
        return "supported"
    if values == {"too wide"}:
        return "inconclusive"
    return "partly supported"


def gate(*, summary: pl.DataFrame) -> dict:
    """Return the per-family labels and the verdict from the decision and iid5 cells."""
    out: dict = {"labels": {}, "detail": {}}
    for family in FAMILIES:
        pick = summary.filter(
            (pl.col("window") == "all")
            & (pl.col("truth") == "hd")
            & (pl.col("arm") == "g_full")
            & (pl.col("method") == "i1")
            & (pl.col("family") == family)
        )
        decision = pick.filter(pl.col("noise") == "real").row(0, named=True)
        iid = pick.filter(pl.col("noise") == "iid5").row(0, named=True)
        out["labels"][family] = label(
            coverage_decision=decision["coverage_truth"], coverage_iid=iid["coverage_truth"]
        )
        out["detail"][family] = {
            "decision_coverage": decision["coverage_truth"],
            "decision_clustered_interval": [
                decision["coverage_truth_low"],
                decision["coverage_truth_high"],
            ],
            "iid5_coverage": iid["coverage_truth"],
            "iid5_clustered_interval": [iid["coverage_truth_low"], iid["coverage_truth_high"]],
            "median_relative_width": decision["median_relative_width"],
            "precision": "precise"
            if decision["median_relative_width"] <= PRECISE_WIDTH[family]
            else "weak",
        }
    out["verdict"] = verdict(labels=out["labels"])
    return out


def main() -> None:
    """Write the summary tables and the gate."""
    frame = with_intervals(frame=read_cells(tag="main"))
    long = long_form(frame=frame)
    summary = summarise(long=long)
    summary.write_parquet(common.WORK_DIR / "summary.parquet")
    by_site(long=long).write_parquet(common.WORK_DIR / "by_site.parquet")
    noise_paths = sorted(common.CELLS_DIR.glob("main_*_noise.parquet"))
    pl.concat([pl.read_parquet(path) for path in noise_paths], how="diagonal").write_parquet(
        common.WORK_DIR / "noise_stats.parquet"
    )
    result = gate(summary=summary)
    (common.WORK_DIR / "gate.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
