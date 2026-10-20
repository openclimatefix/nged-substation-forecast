"""Score the real-data arms: absolute scores, the planned contrasts, winter, tails, and checks.

Every contrast pairs two arms on the same rows and resamples whole blocks of rows with the same
blocks at every site (`studies.bootstrap` groups rows by the label in the `month` column across
sites, so the draw is joint). The blocks are calendar months for R1, R2, R3b and the all-days
rows, weeks for the winter rows, and days for the event days and the cold-snap days.

    uv run python ml-experiments/00117-*/e00117_report.py [--cuts scored replicate]
"""

import argparse
import json
from dataclasses import dataclass
from typing import Any, Final

import e00117_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

LEVEL_PLANNED: Final[float] = 98.75
"""Bonferroni across the four planned contrasts R1, R2, R3, and R3b."""
N_RESAMPLES: Final[int] = 4000
MIN_BLOCKS: Final[int] = 5
"""A scope with fewer blocks is reported as "too few days" and gets no interval."""
EXPERIMENT_106 = (
    ML_EXPERIMENTS_DATA_DIR / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
)
EXTERNAL: Final[dict[str, str]] = {
    "climatology": "{cut}_climatology",
    "manual_heuristic": "{cut}_manual_heuristic",
}
"""Reference sets that experiment 106 stored for the scored and replicate years."""
TAIL_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")


@dataclass(frozen=True)
class Contrast:
    """A treatment arm minus a reference arm on one scope.

    Attributes:
        name: The contrast's label.
        treatment: The treatment arm.
        reference: The reference arm.
        scope: A key of `SCOPES`.
        block: `month`, `week`, or `day`: the unit that is resampled.
        planned: True for the four planned contrasts, which use the Bonferroni level.
    """

    name: str
    treatment: str
    reference: str
    scope: str
    block: str
    planned: bool = False


SCOPES: Final[dict[str, str]] = {
    "all": "all days",
    "event": "event days",
    "non_event": "days that are not event days",
    "winter": "November to March",
    "winter_evening_peak": "evening peak, November to March",
    "winter_event": "event days, November to March",
    "cold_snap": "cold-snap days",
}
CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast("R1 H - G", "h_lo", "g", "all", "month", planned=True),
    Contrast("R2 NH - N", "nh_lo", "n", "all", "month", planned=True),
    Contrast("R3 P-hol - G, event days", "phol", "g", "event", "day", planned=True),
    Contrast("R3b P-hol - G, other days", "phol", "g", "non_event", "month", planned=True),
    Contrast("R1w H - G, winter", "h_lo", "g", "winter", "week", planned=True),
    Contrast("R2w NH - N, winter", "nh_lo", "n", "winter", "week", planned=True),
    Contrast("R3w P-hol - G, winter event days", "phol", "g", "winter_event", "day", planned=True),
    Contrast("H - G, winter evening peak", "h_lo", "g", "winter_evening_peak", "week"),
    Contrast("NH - N, winter evening peak", "nh_lo", "n", "winter_evening_peak", "week"),
    Contrast("H - G, cold snaps", "h_lo", "g", "cold_snap", "day"),
    Contrast("NH - N, cold snaps", "nh_lo", "n", "cold_snap", "day"),
    Contrast("Exploratory: H heating only - G", "h_heat", "g", "all", "month"),
    Contrast("Exploratory: H heating only - G, winter", "h_heat", "g", "winter", "week"),
    Contrast("Exploratory: H holiday only - G", "h_hol", "g", "all", "month"),
    Contrast("N - G", "n", "g", "all", "month"),
    Contrast("NH - G", "nh_lo", "g", "all", "month"),
    Contrast("H - N", "h_lo", "n", "all", "month"),
    Contrast("H(next gamma) - G", "h_hi", "g", "all", "month"),
    Contrast("NH(next gamma) - N", "nh_hi", "n", "all", "month"),
)


def scope_mask(*, frame: pl.DataFrame, scope: str) -> pl.Expr:
    """Return the expression that selects a scope's rows."""
    table = {
        "all": pl.lit(True),
        "event": pl.col("event_day"),
        "non_event": ~pl.col("event_day"),
        "winter": pl.col("winter"),
        "winter_evening_peak": pl.col("winter") & pl.col("evening_peak"),
        "winter_event": pl.col("winter") & pl.col("event_day"),
        "cold_snap": pl.col("cold_snap"),
    }
    del frame
    return table[scope]


def load_arm(*, cut_name: str, arm: str) -> pl.DataFrame | None:
    """Read an arm's stored predictions, or None if the set does not exist."""
    path = common.DATA_DIR / "predictions" / f"{cut_name}_{arm}" / "predictions.parquet"
    return pl.read_parquet(path) if path.exists() else None


def load_external(*, cut_name: str, key: str) -> pl.DataFrame | None:
    """Read a reference set that experiment 106 stored, or None."""
    path = (
        EXPERIMENT_106 / "predictions" / EXTERNAL[key].format(cut=cut_name) / "predictions.parquet"
    )
    return pl.read_parquet(path) if path.exists() else None


def row_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return per-row losses: quantile score, mean-forecast error, and the four tail pinballs."""
    quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    score = common.quantile_score(actual=actual, quantiles=quantiles)
    levels = dict(zip(QUANTILE_COLUMNS, QUANTILE_LEVELS, strict=True))
    out = frame.select(
        site=pl.col("time_series_id").cast(pl.Utf8),
        time=pl.col("valid_time"),
        seed=pl.col("seed").cast(pl.Int64),
        switching=pl.col("switching_mask"),
    ).with_columns(
        qs=pl.Series(score),
        mean_error=pl.Series(
            np.abs(actual - frame["mean"].to_numpy().astype(np.float64))
            if "mean" in frame.columns
            else np.abs(actual - quantiles[:, 5])
        ),
    )
    for column in TAIL_LEVELS:
        level = levels[column]
        error = actual - frame[column].to_numpy().astype(np.float64)
        out = out.with_columns(
            pl.Series(f"pinball_{column}", np.maximum(level * error, (level - 1.0) * error))
        )
    for column in ("event_day", "winter", "evening_peak", "cold_snap"):
        if column in frame.columns:
            out = out.with_columns(frame[column].alias(column))
    return out.with_columns(
        month=pl.col("time").dt.strftime("%Y-%m"),
        week=pl.col("time").dt.strftime("%G-%V"),
        day=pl.col("time").dt.strftime("%Y-%m-%d"),
    )


def contrast_result(
    *, treatment: pl.DataFrame, reference: pl.DataFrame, spec: Contrast, metric: str
) -> dict[str, Any]:
    """Return one contrast's difference, interval, and per-site signs on the shared rows."""
    scope = scope_mask(frame=treatment, scope=spec.scope)
    keys = ["site", "time"]
    ref = reference.filter(scope & ~pl.col("switching")).select(
        *keys, "seed", "month", "week", "day", pl.col(metric)
    )
    tre = treatment.filter(scope & ~pl.col("switching")).select(*keys, "seed", pl.col(metric))
    paired = ref.join(tre, on=[*keys, "seed"], suffix="_t")
    if paired.is_empty():
        return {"name": spec.name, "metric": metric, "rows": 0, "note": "no rows"}
    blocks = paired[spec.block].n_unique()
    difference = paired[f"{metric}_t"] - paired[metric]
    per_site = (
        paired.with_columns(difference=difference).group_by("site").agg(pl.col("difference").mean())
    )
    result: dict[str, Any] = {
        "name": spec.name,
        "metric": metric,
        "scope": SCOPES[spec.scope],
        "block": spec.block,
        "rows": paired.height,
        "blocks": blocks,
        "difference": common.as_float(difference.mean()),
        "sites_better": int((per_site["difference"] < 0).sum()),
        "sites_worse": int((per_site["difference"] > 0).sum()),
    }
    if blocks < MIN_BLOCKS:
        result["note"] = "too few days"
        return result
    level = LEVEL_PLANNED if spec.planned else 95.0
    frame = pl.concat(
        [
            paired.select(
                arm=pl.lit("reference"),
                site="site",
                time="time",
                seed="seed",
                month=spec.block,
                value=pl.col(metric),
            ),
            paired.select(
                arm=pl.lit("treatment"),
                site="site",
                time="time",
                seed="seed",
                month=spec.block,
                value=pl.col(f"{metric}_t"),
            ),
        ]
    ).rename({"value": metric})
    low, high = bootstrap_difference_at_level(
        losses=frame,
        treatment="treatment",
        reference="reference",
        metric=metric,
        level=level,
        n_resamples=N_RESAMPLES,
    )
    result.update({"level": level, "lower": low, "upper": high})
    return result


def tails_table(*, losses: pl.DataFrame, frame: pl.DataFrame) -> dict[str, float]:
    """Return coverage and pinball at q01, q05, q95, and q99 for one arm on all rows."""
    keep = ~losses["switching"]
    out: dict[str, float] = {}
    actual = frame["actual"].to_numpy().astype(np.float64)[keep.to_numpy()]
    for column in TAIL_LEVELS:
        predicted = frame[column].to_numpy().astype(np.float64)[keep.to_numpy()]
        out[f"coverage_{column}"] = float(np.mean(actual <= predicted))
        out[f"pinball_{column}"] = common.as_float(losses.filter(keep)[f"pinball_{column}"].mean())
    return out


def decision_ids() -> list[int]:
    """Return the series identifiers of the five decision sites, which every score is read on."""
    cache = common.load_cache()
    return [cache.sites[site].table.time_series_id for site in common.DECISION_SITES]


def median_minus_mean(*, frame: pl.DataFrame) -> dict[str, float]:
    """Return each site's median of (q50 minus mean forecast), which exposes a shifted spread."""
    shifted = frame.with_columns(difference=pl.col("q50") - pl.col("mean")).group_by(
        pl.col("time_series_id").cast(pl.Utf8).alias("site")
    )
    return {
        row["site"]: common.as_float(row["difference"])
        for row in shifted.agg(pl.col("difference").median()).sort("site").to_dicts()
    }


def report_cut(*, cut_name: str, arms: list[str]) -> dict[str, Any]:
    """Return the report of one cut: scores, contrasts, tails, and checks."""
    losses: dict[str, pl.DataFrame] = {}
    frames: dict[str, pl.DataFrame] = {}
    ids = decision_ids()
    for arm in arms:
        frame = load_arm(cut_name=cut_name, arm=arm)
        if frame is not None:
            frame = frame.filter(pl.col("time_series_id").is_in(ids))
            frames[arm] = frame
            losses[arm] = row_losses(frame=frame)
    for key in EXTERNAL:
        frame = load_external(cut_name=cut_name, key=key)
        if frame is not None:
            frame = frame.filter(pl.col("time_series_id").is_in(ids))
            flags = losses["g"].select(
                "site", "time", "event_day", "winter", "evening_peak", "cold_snap"
            )
            ext = row_losses(frame=frame).join(flags, on=["site", "time"], how="inner")
            losses[key] = ext
    scores = {}
    for arm, frame in losses.items():
        part = frame.filter(~pl.col("switching"))
        scores[arm] = {
            "all_sites": common.as_float(part["qs"].mean()),
            "by_site": dict(
                part.group_by("site").agg(pl.col("qs").mean()).sort("site").iter_rows()
            ),
            "mean_error": common.as_float(part["mean_error"].mean()),
            "by_scope": {
                scope: common.as_float(
                    part.filter(scope_mask(frame=part, scope=scope))["qs"].mean()
                )
                for scope in SCOPES
                if not part.filter(scope_mask(frame=part, scope=scope)).is_empty()
            },
        }
    contrasts = [
        contrast_result(
            treatment=losses[spec.treatment],
            reference=losses[spec.reference],
            spec=spec,
            metric=metric,
        )
        for spec in CONTRASTS
        if spec.treatment in losses and spec.reference in losses
        for metric in ("qs", "mean_error")
    ]
    tails = {arm: tails_table(losses=losses[arm], frame=frames[arm]) for arm in frames}
    checks = {arm: median_minus_mean(frame=frame) for arm, frame in frames.items()}
    return {"scores": scores, "contrasts": contrasts, "tails": tails, "median_minus_mean": checks}


def main() -> None:
    """Write the report for the requested cuts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts", nargs="+", default=["scored", "replicate"])
    arguments = parser.parse_args()
    arms = [
        "g",
        "n",
        "h_lo",
        "nh_lo",
        "h_hi",
        "nh_hi",
        "phol",
        "h_heat",
        "h_hol",
        "h_perm0",
        "h_perm1",
        "h_perm2",
    ]
    report = {cut: report_cut(cut_name=cut, arms=arms) for cut in arguments.cuts}
    path = common.WORK_DIR / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for cut, part in report.items():
        print(cut, {arm: round(v["all_sites"], 3) for arm, v in part["scores"].items()})
        for row in part["contrasts"]:
            if row["metric"] == "qs":
                print(
                    " ", {k: row.get(k) for k in ("name", "difference", "lower", "upper", "note")}
                )


if __name__ == "__main__":
    main()
