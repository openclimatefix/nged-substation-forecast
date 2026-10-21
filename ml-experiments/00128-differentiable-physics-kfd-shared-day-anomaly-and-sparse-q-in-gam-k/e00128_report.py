"""Score experiment 128's arms and print every number the write-up quotes.

Reads the arms that exist, puts the arms of each contrast on the rows they share, and writes
`work/results_<period>.json` and `work/report_<period>.md`. Intervals resample whole calendar months
of run initialisations, paired across arms; the five planned contrasts C1 to C5 are read at 99%
(a Bonferroni correction across five), the rest at 95%. Subsets of days (cold snaps, holiday days)
resample whole local days, and a subset with fewer than 15 days has no interval.

Beside the contrasts the script writes, per arm, the absolute scores by subset and lead band, the
coverage and pinball loss at q01, q05, q95, and q99 (all rows, the midday hours of April to
September where reverse flow matters, and winter evening peaks), the median-minus-mean forecast per
substation, the fitted constants of KD and the supports of the sparse arms, and the scores of the
references (experiment 118's KEZ, BG2, climatology, and the manual heuristic).

    uv run python ml-experiments/00128-*/e00128_report.py --periods replicate scored
"""

import argparse
import json
from typing import Final

import e00128_arms as arm_module
import e00128_common as common
import e00128_fit as fit
import e00128_score as score
import numpy as np
import polars as pl
from studies.dp_calendar import DAY_TYPES

PLANNED: Final[dict[str, tuple[str, str]]] = {
    "C1 KDS minus GK": ("KDS", "GK"),
    "C2 KD minus GK": ("KD", "GK"),
    "C3 KS minus GK": ("KS", "GK"),
    "C4 KDS minus KD": ("KDS", "KD"),
    "C5 KDS minus KS": ("KDS", "KS"),
}
PLANNED_LEVEL: Final[float] = 99.0
EXPLORATORY: Final[dict[str, tuple[str, str]]] = {
    "KDS minus GK-sp": ("KDS", "GK-sp"),
    "GK-sp minus GK": ("GK-sp", "GK"),
    "KDS-shuf minus KDS": ("KDS-shuf", "KDS"),
    "KDS-nod minus KDS": ("KDS-nod", "KDS"),
    "KDS-shuf minus KDS-nod": ("KDS-shuf", "KDS-nod"),
    "KDS-1n minus KDS": ("KDS-1n", "KDS"),
    "KDS-4n minus KDS": ("KDS-4n", "KDS"),
    "KDS-floor minus KDS": ("KDS-floor", "KDS"),
    "KDS-da minus KDS": ("KDS-da", "KDS"),
    "KDS-da minus GK": ("KDS-da", "GK"),
    "KDS-floor minus GK": ("KDS-floor", "GK"),
}
REFERENCE_CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "KDS minus KEZ (experiment 118)": ("KDS", "kez"),
    "GK minus KEZ (experiment 118)": ("GK", "kez"),
    "KDS minus BG2": ("KDS", "bg2"),
    "KDS minus manual_heuristic": ("KDS", "manual_heuristic"),
    "KDS minus climatology": ("KDS", "climatology"),
}
SUBSETS: Final[tuple[str, ...]] = (
    "all rows",
    "winter",
    "evening_peak",
    "winter_night",
    "cold_snap",
    "cold_winter_night",
    "holiday_and_next",
    "other_days",
)
LEADS: Final[tuple[str, ...]] = ("D+1", "0-24 h", "24-48 h", "48-144 h", "144-168 h", "all leads")
MY_ARMS: Final[tuple[str, ...]] = arm_module.FORECAST_ARMS
SITES: Final[tuple[str, ...]] = common.SCORED_SITES
BAND_OF_LEAD: Final[dict[str, tuple[str, float, float]]] = {
    band[0]: band for band in score.LEAD_BANDS
}
NON_INFERIORITY_MARGIN: Final[float] = 0.02
"""Points of quantile score by which KDS may be worse than GK on the days outside the events."""
TAIL_ARMS: Final[tuple[str, ...]] = ("GK", "KD", "KS", "KDS")
TAIL_LEVELS: Final[tuple[tuple[float, str], ...]] = (
    (0.01, "q01"),
    (0.05, "q05"),
    (0.95, "q95"),
    (0.99, "q99"),
)
BAND_ROWS_BY_SUBSET: Final[tuple[str, ...]] = (
    "all rows",
    "midday_apr_sep",
    "winter_evening_peak",
)
"""The row sets of the tail table: all rows, April to September midday, winter evening peaks."""
COLD_NIGHT_START_HOUR: Final[float] = 22.0
COLD_NIGHT_END_HOUR: Final[float] = 8.0


def holiday_row_flags(*, losses: pl.DataFrame) -> pl.Series:
    """Return True on rows whose half-hour is in a holiday day (named period or bank holiday)."""
    cache = common.load_cache()
    bank_holiday = DAY_TYPES.index("bank_holiday")
    flags = (cache.calendar.holiday_period >= 0) | (cache.calendar.day_type == bank_holiday)
    position = np.searchsorted(
        cache.end_times, losses["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    return pl.Series("holiday", flags[position])


def holiday_and_next_flags(*, losses: pl.DataFrame) -> pl.Series:
    """Return True on rows of a holiday day and of the day after each holiday day."""
    holiday_dates = set(losses.filter(pl.col("holiday"))["local_date"].unique().to_list())
    after = {str(np.datetime64(date) + np.timedelta64(1, "D")) for date in holiday_dates}
    flagged = holiday_dates | after
    return pl.Series(
        "holiday_and_next", [date in flagged for date in losses["local_date"].to_list()]
    )


def cold_flags(
    *, losses: pl.DataFrame, labels: dict[int, str]
) -> tuple[pl.DataFrame, dict[str, dict]]:
    """Return the cold-snap and cold-winter-night flags of every row, and each site's rule.

    A cold-snap row lies on a cold-snap local day at its site. A cold-winter-night row starts at or
    after 22:00 local on a cold-snap day or before 08:00 local on the day after one.
    """
    sets = {site: score.cold_snap_dates(site=site) for site in set(labels.values())}
    series = losses["site"].to_list()
    dates = losses["local_date"].to_list()
    hours = losses["start_hour"].to_list()
    snap, night = [], []
    for i in range(losses.height):
        cold_days = sets[labels[series[i]]][1]
        snap.append(dates[i] in cold_days)
        previous = str(np.datetime64(dates[i]) - np.timedelta64(1, "D"))
        night.append(
            (dates[i] in cold_days and hours[i] >= COLD_NIGHT_START_HOUR)
            or (previous in cold_days and hours[i] < COLD_NIGHT_END_HOUR)
        )
    flags = pl.DataFrame({"cold_snap": snap, "cold_winter_night": night})
    summary = {
        site: {"threshold_c": threshold, "days_total": len(days)}
        for site, (threshold, days) in sets.items()
    }
    return flags, summary


def subset_rows(*, losses: pl.DataFrame, subset: str) -> pl.DataFrame:
    """Return the rows of a subset of days or hours."""
    if subset == "all rows":
        return losses
    if subset in ("winter", "evening_peak", "winter_night"):
        return score.day_labels(losses=losses, kind=subset)
    if subset == "cold_snap":
        return losses.filter(pl.col("cold_snap"))
    if subset == "cold_winter_night":
        return losses.filter(pl.col("cold_winter_night"))
    if subset == "holiday_and_next":
        return losses.filter(pl.col("holiday_and_next"))
    return losses.filter(~pl.col("holiday_and_next"))


def lead_rows(*, rows: pl.DataFrame, lead: str) -> pl.DataFrame:
    """Return the rows whose lead lies in the named band."""
    band = BAND_OF_LEAD[lead]
    return rows.filter((pl.col("lead_hours") > band[1]) & (pl.col("lead_hours") <= band[2]))


def contrast_table(
    *,
    losses: pl.DataFrame,
    pairs: dict[str, tuple[str, str]],
    level: float,
    labels: dict[int, str],
) -> dict:
    """Return every pair's contrast by subset and lead band, with the mean-forecast error.

    Each entry also holds the contrast with each decision substation left out in turn.
    """
    out: dict = {}
    for name, (treatment, reference) in pairs.items():
        out[name] = {}
        for subset in SUBSETS:
            rows = subset_rows(losses=losses, subset=subset)
            by_day = subset not in ("all rows", "winter", "evening_peak", "winter_night")
            out[name][subset] = {}
            for lead in LEADS:
                part = lead_rows(rows=rows, lead=lead)
                if part.is_empty():
                    continue
                result = score.contrast(
                    losses=part,
                    treatment=treatment,
                    reference=reference,
                    level=level,
                    by_day=by_day,
                )
                for key, metric in (("mean_error", "abs_mean"), ("median_error", "abs_median")):
                    result[key] = score.contrast(
                        losses=part,
                        treatment=treatment,
                        reference=reference,
                        metric=metric,
                        by_day=by_day,
                        interval=False,
                    )["estimate"]
                result["leave_one_out"] = {
                    site: score.contrast(
                        losses=part.filter(pl.col("site") != series),
                        treatment=treatment,
                        reference=reference,
                        by_day=True,
                        interval=False,
                    )["estimate"]
                    for series, site in labels.items()
                    if site in common.DECISION_SITES
                }
                out[name][subset][lead] = result
    return out


def non_inferiority(*, losses: pl.DataFrame) -> dict:
    """Return KDS minus GK on the days outside the holidays, against the margin of 0.02 points."""
    part = lead_rows(rows=subset_rows(losses=losses, subset="other_days"), lead="D+1")
    result = score.contrast(losses=part, treatment="KDS", reference="GK", level=95.0, by_day=True)
    result["margin"] = NON_INFERIORITY_MARGIN
    result["non_inferior"] = bool(
        result["upper"] is not None and result["upper"] <= NON_INFERIORITY_MARGIN
    )
    return result


def load_period(*, period: str) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    """Load this experiment's arms on common rows, and the references that cover the period."""
    mine = {
        arm: score.load_mine(arm=arm, periods={period}, sites=SITES)
        for arm in MY_ARMS
        if any((score.PREDS_DIR / f"{arm}_{period}_{site}.parquet").exists() for site in SITES)
    }
    mine = score.common_rows(arms_by_name=mine)
    references: dict[str, pl.DataFrame] = {}
    if period in ("replicate", "scored"):
        references["kez"] = score.load_kez(periods={period}, sites=SITES)
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
                part = lead_rows(rows=rows, lead=lead)
                if part.height:
                    out[arm][subset][lead] = {
                        "qs": score.mean_of(part["qs"]),
                        "abs_median": score.mean_of(part["abs_median"]),
                        "abs_mean": score.mean_of(part["abs_mean"]),
                        "n": part.height,
                    }
        all_sites_arm = losses.filter(pl.col("arm") == arm)
        out[arm]["per_site"] = {
            labels[int(series)]: {
                lead: score.mean_of(
                    lead_rows(rows=all_sites_arm.filter(pl.col("site") == series), lead=lead)["qs"]
                )
                for lead in ("D+1", "all leads")
            }
            for series in labels
        }
    return out


def tail_rows(*, stacked: pl.DataFrame, which: str) -> pl.DataFrame:
    """Return the rows of one tail-table row set.

    `midday_apr_sep` holds the half-hours that start 11:00 to 15:59 local from April to September,
    where demand is positive and reverse flow is negative. `winter_evening_peak` is the winter
    evening peak of `score.day_labels`.
    """
    if which == "all rows":
        return stacked
    if which == "midday_apr_sep":
        return stacked.filter(
            pl.col("local_month").is_between(4, 9) & pl.col("start_hour").is_between(11.0, 15.5)
        )
    return score.day_labels(losses=stacked, kind="evening_peak")


def tails(*, stacked: pl.DataFrame, decision_series: list[int]) -> dict:
    """Return coverage and pinball loss at q01, q05, q95, and q99 per arm, row set, and lead band.

    Only the decision substations count.
    """
    out: dict = {}
    decision = stacked.filter(pl.col("site").is_in(decision_series))
    for arm in TAIL_ARMS:
        out[arm] = {}
        for which in BAND_ROWS_BY_SUBSET:
            rows = tail_rows(stacked=decision.filter(pl.col("arm") == arm), which=which)
            out[arm][which] = {}
            for lead in ("D+1", "all leads"):
                part = lead_rows(rows=rows, lead=lead)
                actual = part["actual"].to_numpy().astype(np.float64)
                out[arm][which][lead] = {}
                for level, column in TAIL_LEVELS:
                    quantile = part[column].to_numpy().astype(np.float64)
                    error = actual - quantile
                    out[arm][which][lead][column] = {
                        "coverage": float(np.mean(actual <= quantile)),
                        "pinball": float(np.mean(np.maximum(level * error, (level - 1.0) * error))),
                        "n": part.height,
                    }
    return out


def median_minus_mean(*, arms_by_name: dict[str, pl.DataFrame], labels: dict[int, str]) -> dict:
    """Return the mean median-forecast-minus-mean-forecast per arm and site, in points."""
    out: dict = {}
    by_label = {site: series for series, site in labels.items()}
    for arm, frame in arms_by_name.items():
        out[arm] = {}
        for site, series in by_label.items():
            part = frame.filter(pl.col("time_series_id") == series)
            out[arm][site] = score.mean_of(part["q50"] - part["mean"])
    return out


def reference_contrasts(
    *, mine: dict[str, pl.DataFrame], references: dict[str, pl.DataFrame]
) -> dict:
    """Return the arms' contrasts against each reference on the rows the pair shares."""
    out: dict = {}
    for name, (treatment, reference) in REFERENCE_CONTRASTS.items():
        if reference not in references or treatment not in mine:
            continue
        pair = score.common_rows(
            arms_by_name={
                treatment: mine[treatment].drop(
                    "mean", "level", "anomaly", "day_term", strict=False
                ),
                reference: references[reference].drop("mean", strict=False),
            }
        )
        stacked = score.stack_losses(arms_by_name=with_mean_error(frames=pair))
        out[name] = {
            lead: score.contrast(
                losses=lead_rows(rows=stacked, lead=lead),
                treatment=treatment,
                reference=reference,
                level=95.0,
            )
            for lead in ("D+1", "all leads")
        }
        out[name]["n_rows"] = pair[treatment].height
    return out


def reference_scores(
    *, mine: dict[str, pl.DataFrame], references: dict[str, pl.DataFrame], labels: dict[int, str]
) -> dict:
    """Return the headline arm's and each reference's mean quantile score by lead band.

    Only the decision sites count. A reference's rows are those it shares with the headline arm.
    """
    anchor = "KDS" if "KDS" in mine else "GK"
    out: dict = {}
    decision = [site for site in labels.values() if site in common.DECISION_SITES]
    for name, frame in references.items():
        pair = score.common_rows(
            arms_by_name={
                anchor: mine[anchor].drop("mean", "level", "anomaly", "day_term", strict=False),
                name: frame.drop("mean", strict=False),
            }
        )
        out[name] = {}
        for arm, rows in pair.items():
            kept = rows.filter(pl.col("site").is_in(decision))
            scored = kept.with_columns(score.quantile_score(frame=kept))
            out[name][arm] = {
                lead: score.mean_of(lead_rows(rows=scored, lead=lead)["qs"]) for lead in LEADS
            }
    return out


def fitted_constants() -> dict:
    """Return every arm's stored constants and sparse supports, for the substation cards."""
    out: dict = {}
    for arm in fit.FIT_ARMS:
        path = fit.hyper_path(arm=arm)
        if path.exists():
            out[arm] = json.loads(path.read_text())["sites"]
    return out


def run_period(*, period: str) -> dict:
    """Score one period and return the results."""
    mine, references = load_period(period=period)
    ids, _ = score.site_meta(sites=SITES)
    labels = {series: site for site, series in ids.items()}
    results: dict = {"period": period, "arms": sorted(mine), "reference_arms": sorted(references)}
    stacked = score.stack_losses(arms_by_name=with_mean_error(frames=mine))
    stacked = stacked.with_columns(holiday_row_flags(losses=stacked))
    stacked = stacked.with_columns(holiday_and_next_flags(losses=stacked))
    flags, summary = cold_flags(losses=stacked, labels=labels)
    stacked = stacked.with_columns(flags)
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
    decision_series = [series for series, site in labels.items() if site in common.DECISION_SITES]
    decision = stacked.filter(pl.col("site").is_in(decision_series))
    d9 = stacked.filter(~pl.col("site").is_in(decision_series))
    results["planned"] = contrast_table(
        losses=decision, pairs=planned, level=PLANNED_LEVEL, labels=labels
    )
    results["exploratory"] = contrast_table(
        losses=decision, pairs=exploratory, level=95.0, labels=labels
    )
    results["d9_planned"] = contrast_table(losses=d9, pairs=planned, level=95.0, labels=labels)
    results["absolute"] = absolute_scores(losses=stacked, labels=labels)
    if {"KDS", "GK"} <= present:
        results["non_inferiority_other_days"] = non_inferiority(losses=decision)
    results["tails"] = tails(stacked=stacked, decision_series=decision_series)
    results["median_minus_mean"] = median_minus_mean(arms_by_name=mine, labels=labels)
    results["against_references"] = reference_contrasts(mine=mine, references=references)
    results["reference_scores"] = reference_scores(mine=mine, references=references, labels=labels)
    results["fits"] = fitted_constants()
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
    lines = [f"# Experiment 128: {results['period']} period", ""]
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
    if "non_inferiority_other_days" in results:
        entry = results["non_inferiority_other_days"]
        lines += [
            "## Non-inferiority of KDS on the days outside the holidays (D+1)",
            "",
            f"{fmt(entry=entry)}; margin {entry['margin']}; non-inferior: {entry['non_inferior']}",
            "",
        ]
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
