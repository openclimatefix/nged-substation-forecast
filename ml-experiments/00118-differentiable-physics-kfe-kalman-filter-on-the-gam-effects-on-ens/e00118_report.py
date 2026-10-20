"""Score experiment 118's arms and print every number the write-up quotes.

Reads the arms that exist, puts the arms of each contrast on the rows they share, and writes
`work/results_<period>.json` and `work/report_<period>.md`. Intervals resample whole calendar months
of run initialisations, paired across arms; the five planned contrasts C1 to C5 are read at 99%
(a Bonferroni correction across five), the rest at 95%. Subsets of days (cold snaps, holiday days)
resample whole local days, and a subset with fewer than 15 days has no interval.

    uv run python ml-experiments/00118-*/e00118_report.py --periods replicate scored
"""

import argparse
import json
from typing import Final

import e00118_common as common
import e00118_forecast as fc
import e00118_score as score
import numpy as np
import polars as pl

PLANNED: Final[dict[str, tuple[str, str]]] = {
    "C1 KEZ minus GK": ("KEZ", "GK"),
    "C2 KE minus GK": ("KE", "GK"),
    "C3 KEZ minus KE": ("KEZ", "KE"),
    "C4 KEZ minus KEZ-Q0": ("KEZ", "KEZ-Q0"),
    "C5 KEZ minus KEZ-QI": ("KEZ", "KEZ-QI"),
}
PLANNED_LEVEL: Final[float] = 99.0
EXPLORATORY: Final[dict[str, tuple[str, str]]] = {
    "KEZ minus GK-sp": ("KEZ", "GK-sp"),
    "GK-sp minus GK": ("GK-sp", "GK"),
    "KEZ-shift minus GK": ("KEZ-shift", "GK"),
    "KEZ minus KEZ-shift": ("KEZ", "KEZ-shift"),
    "KE-infl minus KEZ": ("KE-infl", "KEZ"),
    "KEZ-drop-profile minus KEZ": ("KEZ-drop-profile", "KEZ"),
    "KEZ-drop-heating minus KEZ": ("KEZ-drop-heating", "KEZ"),
    "KEZ-drop-pv minus KEZ": ("KEZ-drop-pv", "KEZ"),
    "KEZ-drop-holiday minus KEZ": ("KEZ-drop-holiday", "KEZ"),
    "KEZ-DT minus KEZ": ("KEZ-DT", "KEZ"),
    "KEZ-DT minus GK": ("KEZ-DT", "GK"),
    "KEZ-Q0 minus GK": ("KEZ-Q0", "GK"),
}
REFERENCE_CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "KEZ minus GAM-0": ("KEZ", "gam0"),
    "GK minus GAM-0": ("GK", "gam0"),
    "KEZ minus X-nl+K": ("KEZ", "xnl_k"),
    "KEZ minus Xlag-d": ("KEZ", "xlag_d"),
    "KEZ minus BG2": ("KEZ", "bg2"),
    "KEZ minus manual_heuristic": ("KEZ", "manual_heuristic"),
    "KEZ minus climatology": ("KEZ", "climatology"),
}
SUBSETS: Final[tuple[str, ...]] = (
    "all rows",
    "winter",
    "evening_peak",
    "winter_night",
    "cold_snap",
    "holiday",
    "non_holiday",
)
LEADS: Final[tuple[str, ...]] = ("D+1", "0-24 h", "24-48 h", "48-144 h", "144-168 h", "all leads")
MY_ARMS: Final[tuple[str, ...]] = fc.FORECAST_ARMS
SITES: Final[tuple[str, ...]] = common.SCORED_SITES
BAND_OF_LEAD: Final[dict[str, tuple[str, float, float]]] = {
    band[0]: band for band in score.LEAD_BANDS
}


def holiday_flags(*, losses: pl.DataFrame) -> pl.Series:
    """Return True on rows whose half-hour is in a holiday day (named period or bank holiday)."""
    cache = common.load_cache()
    from e00118_streams import BANK_HOLIDAY

    flags = (cache.calendar.holiday_period >= 0) | (cache.calendar.day_type == BANK_HOLIDAY)
    position = np.searchsorted(
        cache.end_times, losses["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    return pl.Series("holiday", flags[position])


def cold_flags(
    *, losses: pl.DataFrame, labels: dict[int, str]
) -> tuple[pl.Series, dict[str, dict]]:
    """Return True on rows of a cold-snap day at their site, and each site's rule and day counts."""
    sets = {site: score.cold_snap_dates(site=site) for site in set(labels.values())}
    series = losses["site"].to_list()
    dates = losses["local_date"].to_list()
    flags = [dates[i] in sets[labels[series[i]]][1] for i in range(losses.height)]
    summary = {
        site: {"threshold_c": threshold, "days_total": len(days)}
        for site, (threshold, days) in sets.items()
    }
    return pl.Series("cold_snap", flags), summary


def subset_rows(*, losses: pl.DataFrame, subset: str) -> pl.DataFrame:
    """Return the rows of a subset of days or hours."""
    if subset == "all rows":
        return losses
    if subset in ("winter", "evening_peak", "winter_night"):
        return score.day_labels(losses=losses, kind=subset)
    if subset == "cold_snap":
        return losses.filter(pl.col("cold_snap"))
    if subset == "holiday":
        return losses.filter(pl.col("holiday"))
    return losses.filter(~pl.col("holiday"))


def contrast_table(
    *,
    losses: pl.DataFrame,
    pairs: dict[str, tuple[str, str]],
    level: float,
    labels: dict[int, str],
) -> dict:
    """Return every pair's contrast by subset and lead band, quantile score and mean error."""
    out: dict = {}
    for name, (treatment, reference) in pairs.items():
        out[name] = {}
        for subset in SUBSETS:
            rows = subset_rows(losses=losses, subset=subset)
            by_day = subset in ("cold_snap", "holiday", "non_holiday")
            out[name][subset] = {}
            for lead in LEADS:
                band = BAND_OF_LEAD[lead]
                part = rows.filter(
                    (pl.col("lead_hours") > band[1]) & (pl.col("lead_hours") <= band[2])
                )
                if part.is_empty():
                    continue
                result = score.contrast(
                    losses=part,
                    treatment=treatment,
                    reference=reference,
                    level=level,
                    by_day=by_day,
                )
                result["mean_error"] = score.contrast(
                    losses=part,
                    treatment=treatment,
                    reference=reference,
                    metric="abs_mean",
                    level=95.0,
                    by_day=by_day,
                )["estimate"]
                result["median_error"] = score.contrast(
                    losses=part,
                    treatment=treatment,
                    reference=reference,
                    metric="abs_median",
                    level=95.0,
                    by_day=by_day,
                )["estimate"]
                # Leaving each decision site out in turn: the sign of the estimate.
                result["leave_one_out_signs"] = {
                    site: float(
                        np.sign(
                            score.contrast(
                                losses=part.filter(pl.col("site") != series),
                                treatment=treatment,
                                reference=reference,
                                level=95.0,
                                by_day=True,
                            )["estimate"]
                        )
                    )
                    for series, site in labels.items()
                    if site in common.DECISION_SITES
                }
                out[name][subset][lead] = result
    return out


def load_period(*, period: str) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    """Load this experiment's arms on common rows, and the references that cover the period."""
    mine = {
        arm: score.load_mine(arm=arm, periods={period}, sites=SITES)
        for arm in MY_ARMS
        if any((score.PREDS_DIR / f"{arm}_{period}_{site}.parquet").exists() for site in SITES)
    }
    mine = score.common_rows(arms_by_name=mine)
    references: dict[str, pl.DataFrame] = {}
    references["gam0"] = score.load_gam0(periods={period}, sites=SITES)
    references["xnl_k"] = score.load_xnl_k(periods={period}, sites=SITES)
    references["xlag_d"] = score.load_xlag_d(periods={period}, sites=SITES)
    if period == "scored":
        for name in score.E87_NAMES:
            references[name] = score.load_e87(name=name, sites=SITES)
    return mine, {name: frame for name, frame in references.items() if frame.height}


def with_mean_error(*, frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Add the absolute error of the mean forecast (the `mean` column), where the frame has one."""
    return {
        name: frame.with_columns(abs_mean=(pl.col("actual") - pl.col("mean")).abs())
        if "mean" in frame.columns
        else frame.with_columns(abs_mean=(pl.col("actual") - pl.col("q50")).abs())
        for name, frame in frames.items()
    }


def mine_with_mean(*, mine: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Return this experiment's arms with the absolute error of the mean forecast added."""
    return with_mean_error(frames=mine)


def absolute_scores(*, losses: pl.DataFrame, labels: dict[int, str]) -> dict:
    """Return each arm's mean quantile score by subset and lead band, at the decision sites.

    Each entry also holds the per-site means at the next-day and all-lead bands.
    """
    decision = [series for series, site in labels.items() if site in common.DECISION_SITES]
    scope = losses.filter(pl.col("site").is_in(decision))
    out: dict = {}
    for arm in sorted(scope["arm"].unique().to_list()):
        rows_arm = scope.filter(pl.col("arm") == arm)
        out[arm] = {}
        for subset in SUBSETS:
            rows = subset_rows(losses=rows_arm, subset=subset)
            out[arm][subset] = {}
            for lead in LEADS:
                band = BAND_OF_LEAD[lead]
                part = rows.filter(
                    (pl.col("lead_hours") > band[1]) & (pl.col("lead_hours") <= band[2])
                )
                if part.height:
                    out[arm][subset][lead] = {
                        "qs": float(part["qs"].mean()),
                        "abs_median": float(part["abs_median"].mean()),
                        "abs_mean": float(part["abs_mean"].mean()),
                        "n": part.height,
                    }
        out[arm]["per_site"] = {
            labels[int(series)]: {
                lead: float(
                    rows_arm.filter(
                        (pl.col("site") == series)
                        & (pl.col("lead_hours") > BAND_OF_LEAD[lead][1])
                        & (pl.col("lead_hours") <= BAND_OF_LEAD[lead][2])
                    )["qs"].mean()
                )
                for lead in ("D+1", "all leads")
            }
            for series in labels
        }
    return out


def tails(*, arms_by_name: dict[str, pl.DataFrame]) -> dict:
    """Return coverage and pinball loss at q01, q05, q95, and q99 per arm and lead band."""
    out: dict = {}
    for arm, frame in arms_by_name.items():
        out[arm] = {}
        for lead in ("D+1", "0-24 h", "24-48 h", "all leads"):
            band = BAND_OF_LEAD[lead]
            part = frame.filter(
                (pl.col("lead_hours") > band[1]) & (pl.col("lead_hours") <= band[2])
            )
            actual = part["actual"].to_numpy().astype(np.float64)
            out[arm][lead] = {}
            for level, column in ((0.01, "q01"), (0.05, "q05"), (0.95, "q95"), (0.99, "q99")):
                quantile = part[column].to_numpy().astype(np.float64)
                error = actual - quantile
                out[arm][lead][column] = {
                    "coverage": float(np.mean(actual <= quantile)),
                    "pinball": float(np.mean(np.maximum(level * error, (level - 1.0) * error))),
                }
    return out


def median_minus_mean(*, arms_by_name: dict[str, pl.DataFrame], labels: dict[int, str]) -> dict:
    """Return the mean median-forecast-minus-mean-forecast per arm and site, in points."""
    out: dict = {}
    by_label = {site: series for series, site in labels.items()}
    for arm, frame in arms_by_name.items():
        out[arm] = {
            site: float(
                (
                    frame.filter(pl.col("time_series_id") == series)["q50"]
                    - frame.filter(pl.col("time_series_id") == series)["mean"]
                ).mean()
            )
            for site, series in by_label.items()
        }
    return out


def reference_contrasts(
    *, mine: dict[str, pl.DataFrame], references: dict[str, pl.DataFrame], labels: dict[int, str]
) -> dict:
    """Return KEZ's contrasts against each reference on the rows the pair shares, exploratory."""
    out: dict = {}
    for name, (treatment, reference) in REFERENCE_CONTRASTS.items():
        if reference not in references or treatment not in mine:
            continue
        pair = score.common_rows(
            arms_by_name={
                treatment: mine[treatment].drop("mean", "level", "anomaly", strict=False),
                reference: references[reference].drop("mean", strict=False),
            }
        )
        stacked = score.stack_losses(arms_by_name=with_mean_error(frames=pair))
        out[name] = {}
        for lead in ("D+1", "all leads"):
            band = BAND_OF_LEAD[lead]
            part = stacked.filter(
                (pl.col("lead_hours") > band[1]) & (pl.col("lead_hours") <= band[2])
            )
            out[name][lead] = score.contrast(
                losses=part, treatment=treatment, reference=reference, level=95.0
            )
        out[name]["n_rows"] = pair[treatment].height
    return out


def run_period(*, period: str) -> dict:
    """Score one period and return the results."""
    mine, references = load_period(period=period)
    ids, _ = score.site_meta(sites=SITES)
    labels = {series: site for site, series in ids.items()}
    results: dict = {"period": period, "arms": sorted(mine), "reference_arms": sorted(references)}
    stacked = score.stack_losses(arms_by_name=mine_with_mean(mine=mine))
    stacked = stacked.with_columns(holiday_flags(losses=stacked))
    cold, summary = cold_flags(losses=stacked, labels=labels)
    stacked = stacked.with_columns(cold)
    results["cold_snap_rule"] = summary
    results["cold_snap_days_in_rows"] = {
        labels[int(series)]: int(
            stacked.filter((pl.col("site") == series) & pl.col("cold_snap"))[
                "local_date"
            ].n_unique()
        )
        for series in labels
    }
    results["holiday_days_in_rows"] = int(
        stacked.filter(pl.col("holiday"))["local_date"].n_unique()
    )
    present = set(mine)
    planned = {k: v for k, v in PLANNED.items() if set(v) <= present}
    exploratory = {k: v for k, v in EXPLORATORY.items() if set(v) <= present}
    results["planned"] = contrast_table(
        losses=stacked, pairs=planned, level=PLANNED_LEVEL, labels=labels
    )
    results["exploratory"] = contrast_table(
        losses=stacked, pairs=exploratory, level=95.0, labels=labels
    )
    results["absolute"] = absolute_scores(losses=stacked, labels=labels)
    results["tails"] = tails(arms_by_name=mine)
    results["median_minus_mean"] = median_minus_mean(arms_by_name=mine, labels=labels)
    results["against_references"] = reference_contrasts(
        mine=mine, references=references, labels=labels
    )
    return results


def fmt(*, entry: dict | None) -> str:
    """Format a contrast as `estimate [lower, upper] (k of n)`."""
    if not entry:
        return "n/a"
    if entry.get("too_few_days"):
        return f"{entry['estimate']:+.3f} (too few days: {entry['n_days']})"
    interval = "" if entry["lower"] is None else f" [{entry['lower']:+.3f}, {entry['upper']:+.3f}]"
    return f"{entry['estimate']:+.3f}{interval} ({entry['sites_better']} of {entry['n_sites']})"


def markdown(*, results: dict) -> str:
    """Return the period's key tables as markdown."""
    lines = [f"# Experiment 118: {results['period']} period", ""]
    for group in ("planned", "exploratory"):
        lines += [f"## Contrasts ({group})", ""]
        for name, by_subset in results[group].items():
            lines.append(f"**{name}**")
            for subset in SUBSETS:
                cells = [
                    f"{lead}: {fmt(entry=by_subset.get(subset, {}).get(lead))}" for lead in LEADS
                ]
                lines.append(f"- {subset}: " + "; ".join(cells))
            lines.append("")
    lines += ["## Absolute scores at the decision sites (quantile score, points)", ""]
    for arm, by_subset in results["absolute"].items():
        row = by_subset["all rows"]
        cells = ", ".join(f"{lead} {row[lead]['qs']:.3f}" for lead in LEADS if lead in row)
        lines.append(f"- {arm}: {cells}")
    lines += ["", "## Cold-snap days per site in the rows", ""]
    lines.append(json.dumps(results["cold_snap_days_in_rows"]))
    return "\n".join(lines)


def main() -> None:
    """Score the chosen periods."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", nargs="+", default=["replicate", "scored"])
    args = parser.parse_args()
    for period in args.periods:
        results = run_period(period=period)
        common.write_json_atomic(path=common.WORK_DIR / f"results_{period}.json", content=results)
        (common.WORK_DIR / f"report_{period}.md").write_text(markdown(results=results))
        print(markdown(results=results))


if __name__ == "__main__":
    main()
