"""Score experiment 205's rules and print every number the write-up quotes.

Reads `work/rows/<site>.parquet` and writes `work/report/report.json`: each rule's quantile score
by gap length, the eight headline contrasts with 99.375% month-resampled intervals, the descriptive
contrasts against the frozen model alone, recovery after each gap, tail coverage, the checks, and
the per-substation and leave-one-substation-out views.

    uv run --no-sync python ml-experiments/00205-*/e00205_report.py
"""

import json
from typing import Final

import e00205_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level

LEVEL_PERCENT: Final[float] = 100.0 - 5.0 / 8.0
N_RESAMPLES: Final[int] = 20_000
GAPS: Final[tuple[int, ...]] = (3, 6, 24, 72)
DISTANCES: Final[tuple[int, ...]] = (0, 3, 6, 12, 24, 48)
RULES_WITH_GAP: Final[tuple[str, ...]] = ("a", "b", "c2", "a0", "b0")
FIXED_RULES: Final[tuple[str, ...]] = ("n", "n0", "c")
"""Rules that do not depend on the gap: no gap, no gap with the filter alone, frozen model alone."""
SCOPES: Final[dict[str, tuple[float, float]]] = {
    "all": (0.0, 168.0),
    "under 24 h": (0.0, 24.0),
    "12-36 h": (12.0, 36.0),
}
RECOVERY_MARGIN: Final[float] = 0.02
NOMINAL: Final[dict[str, float]] = {"q05": 0.05, "q95": 0.95}


def load() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the gap-dependent rows and the fixed rules' rows of every substation."""
    frame = pl.concat(
        [pl.read_parquet(common.WORK_DIR / "rows" / f"{s}.parquet") for s in common.DECISION_SITES]
    ).with_columns(
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        time=pl.concat_str(
            pl.col("power_fcst_init_time").dt.strftime("%Y-%m-%dT%H:%M:%S"),
            pl.col("valid_time").dt.strftime("%Y-%m-%dT%H:%M:%S"),
            separator="|",
        ),
        seed=pl.lit(0),
    )
    fixed = frame.filter(pl.col("rule").is_in(FIXED_RULES))
    varying = frame.filter(~pl.col("rule").is_in(FIXED_RULES))
    return varying, fixed


def scope_rows(*, frame: pl.DataFrame, scope: str) -> pl.DataFrame:
    """Keep the rows whose lead lies in the scope (above the low and at most the high bound)."""
    low, high = SCOPES[scope]
    return frame.filter((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))


def arms(*, varying: pl.DataFrame, fixed: pl.DataFrame, hours: int, distance: int) -> pl.DataFrame:
    """Return the rows of every rule at one (gap length, distance), the fixed rules included."""
    gapped = varying.filter((pl.col("gap_hours") == hours) & (pl.col("distance_hours") == distance))
    return (
        pl.concat([gapped, fixed])
        .rename({"site": "site_label"})
        .with_columns(site=pl.col("site_label"), arm=pl.col("rule"))
    )


def contrast(*, losses: pl.DataFrame, treatment: str, reference: str, metric: str = "qs") -> dict:
    """Return a paired contrast with a Bonferroni-level month-resampled interval."""
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
    estimate = float(np.mean(t[metric].to_numpy()) - np.mean(r[metric].to_numpy()))
    state = "below" if upper < 0 else "above" if lower > 0 else "straddling"
    return {"estimate": estimate, "lower": float(lower), "upper": float(upper), "state": state}


def mean_by(*, losses: pl.DataFrame, metric: str) -> dict[str, float]:
    """Return the mean of a metric by rule."""
    return {r: float(v) for r, v in losses.group_by("arm").agg(pl.col(metric).mean()).iter_rows()}


def recovery(*, scores: dict[int, float], reference: float) -> int | None:
    """Return the smallest distance from which the score stays within the margin of no gap."""
    ok = [scores[k] - reference <= RECOVERY_MARGIN for k in DISTANCES]
    for index, k in enumerate(DISTANCES):
        if all(ok[index:]):
            return k
    return None


def main() -> None:
    """Score every rule and write the report."""
    varying, fixed = load()
    report: dict = {"level_percent": LEVEL_PERCENT, "headline": {}, "scores": {}, "coverage": {}}
    per_gap: dict[int, pl.DataFrame] = {}
    for hours in GAPS:
        losses = arms(varying=varying, fixed=fixed, hours=hours, distance=0)
        per_gap[hours] = losses
        n_events = losses.filter(pl.col("arm") == "n")["run"].n_unique()
        entry = {"n_rows": losses.filter(pl.col("arm") == "a").height}
        for scope in SCOPES:
            scoped = scope_rows(frame=losses, scope=scope)
            entry[scope] = {
                "qs": mean_by(losses=scoped, metric="qs"),
                "abs_mean": mean_by(losses=scoped, metric="abs_mean"),
                "abs_median": mean_by(losses=scoped, metric="abs_median"),
            }
        scoped = scope_rows(frame=losses, scope="all")
        entry["coverage"] = {q: mean_by(losses=scoped, metric=f"cov_{q}") for q in NOMINAL}
        entry["pinball"] = {
            q: mean_by(losses=scoped, metric=f"pin_{q}") for q in ("q01", "q05", "q95", "q99")
        }
        entry["coverage_q01_q99"] = {
            q: mean_by(losses=scoped, metric=f"cov_{q}") for q in ("q01", "q99")
        }
        entry["median_minus_mean_by_site"] = {
            rule: {
                site: float(v)
                for site, v in scoped.filter(pl.col("arm") == rule)
                .group_by("site_label")
                .agg(pl.col("median_minus_mean").mean())
                .iter_rows()
            }
            for rule in ("n", "a", "b", "c2")
        }
        entry["n_run_site_pairs"] = int(
            losses.filter(pl.col("arm") == "n").select("site", "run").unique().height
        )
        entry["n_events_by_site"] = {
            s: int(n)
            for s, n in losses.filter(pl.col("arm") == "n")
            .group_by("site")
            .agg(pl.col("run").n_unique())
            .iter_rows()
        }
        report["scores"][str(hours)] = entry
        scoped = scope_rows(frame=losses, scope="all")
        report["headline"][str(hours)] = {
            "C_a_minus_c2": contrast(losses=scoped, treatment="a", reference="c2"),
            "C_b_minus_a": contrast(losses=scoped, treatment="b", reference="a"),
            "a_minus_c": contrast(losses=scoped, treatment="a", reference="c"),
            "a_minus_n": contrast(losses=scoped, treatment="a", reference="n"),
            "c2_minus_n": contrast(losses=scoped, treatment="c2", reference="n"),
            "b_minus_n": contrast(losses=scoped, treatment="b", reference="n"),
            "c_minus_n": contrast(losses=scoped, treatment="c", reference="n"),
            "a_minus_c2_12_36h": contrast(
                losses=scope_rows(frame=losses, scope="12-36 h"), treatment="a", reference="c2"
            ),
            "a_minus_c2_mean_abs_error": contrast(
                losses=scoped, treatment="a", reference="c2", metric="abs_mean"
            ),
            "a0_minus_c2": contrast(losses=scoped, treatment="a0", reference="c2"),
            "a_under_24h_minus_c2": contrast(
                losses=scope_rows(frame=losses, scope="under 24 h"), treatment="a", reference="c2"
            ),
            "coverage_b_minus_a": {
                q: contrast(losses=scoped, treatment="b", reference="a", metric=f"cov_{q}")
                for q in NOMINAL
            },
            "per_site": {},
            "leave_one_site_out": {},
        }
        for site in common.DECISION_SITES:
            one = scoped.filter(pl.col("site_label") == site)
            report["headline"][str(hours)]["per_site"][site] = {
                "a_minus_c2": float(
                    np.mean(one.filter(pl.col("arm") == "a")["qs"].to_numpy())
                    - np.mean(one.filter(pl.col("arm") == "c2")["qs"].to_numpy())
                ),
                "n_events": int(one.filter(pl.col("arm") == "a")["run"].n_unique()),
            }
            rest = scoped.filter(pl.col("site_label") != site)
            report["headline"][str(hours)]["leave_one_site_out"][site] = float(
                np.mean(rest.filter(pl.col("arm") == "a")["qs"].to_numpy())
                - np.mean(rest.filter(pl.col("arm") == "c2")["qs"].to_numpy())
            )
        report["headline"][str(hours)]["n_events"] = int(n_events)
    # Recovery: the all-lead and under-24-hour score of rule (a) against the no-gap score.
    report["recovery"] = {}
    for hours in GAPS:
        entry = {}
        for scope in ("all", "under 24 h"):
            scores: dict[int, float] = {}
            scores_filter_only: dict[int, float] = {}
            reference = reference_filter_only = 0.0
            for distance in DISTANCES:
                losses = scope_rows(
                    frame=arms(varying=varying, fixed=fixed, hours=hours, distance=distance),
                    scope=scope,
                )
                means = mean_by(losses=losses, metric="qs")
                scores[distance] = means["a"]
                scores_filter_only[distance] = means["a0"]
                reference = means["n"]
                reference_filter_only = means["n0"]
            entry[scope] = {
                "a_score_by_distance": scores,
                "no_gap": reference,
                "recovery_hours": recovery(scores=scores, reference=reference),
                "a0_score_by_distance": scores_filter_only,
                "no_gap_filter_only": reference_filter_only,
                "a0_recovery_hours": recovery(
                    scores=scores_filter_only, reference=reference_filter_only
                ),
            }
        report["recovery"][str(hours)] = entry
    # Recovery interval for the 24-hour gap on the all-lead score: a minus n at each distance.
    report["recovery_contrast_24"] = {}
    for distance in DISTANCES:
        losses = scope_rows(
            frame=arms(varying=varying, fixed=fixed, hours=24, distance=distance), scope="all"
        )
        report["recovery_contrast_24"][str(distance)] = contrast(
            losses=losses, treatment="a", reference="n"
        )
    controls = [
        json.loads((common.WORK_DIR / "controls" / f"{s}.json").read_text())
        for s in common.DECISION_SITES
    ]
    report["controls"] = controls
    leakage = common.WORK_DIR / "leakage.json"
    report["leakage"] = json.loads(leakage.read_text()) if leakage.exists() else None
    # The median minus the mean forecast per site, rule (a) at the 24-hour gap.
    out = common.WORK_DIR / "report" / "report.json"
    common.write_json_atomic(path=out, content=report)
    _print(report=report)


def _print(*, report: dict) -> None:
    for hours, entry in report["headline"].items():
        print(f"gap {hours} h  ({entry['n_events']} runs per site on average)")
        for name in ("C_a_minus_c2", "C_b_minus_a", "a_minus_c", "a_minus_n", "c2_minus_n"):
            c = entry[name]
            print(
                f"  {name:14s} {c['estimate']:+.3f}  [{c['lower']:+.3f}, {c['upper']:+.3f}] "
                f"{c['state']}"
            )
    for hours, entry in report["recovery"].items():
        print(
            f"recovery L={hours}: all {entry['all']['recovery_hours']}, "
            f"under 24 h {entry['under 24 h']['recovery_hours']}"
        )


if __name__ == "__main__":
    main()
