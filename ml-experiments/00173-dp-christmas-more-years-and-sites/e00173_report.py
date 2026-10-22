"""Score the Christmas windows of experiment 173 and write the numbers the write-up quotes.

The ratio is the mean of a loss over the half-hours valid from 24 December to 2 January, divided by
the mean of the same loss over ordinary weekdays, each ordinary half-hour weighted by how many
Christmas rows it has (so a Christmas that falls on quiet half-hours is not blamed for them). An
ordinary weekday is a Monday to Friday from 8 December to 31 January, outside the Christmas window
and outside any bank holiday. Rows inside a hand-filtered switching event are left out. Intervals
resample the local dates of the Christmas window and of the ordinary weekdays.

    uv run --no-sync python ml-experiments/00173-*/e00173_report.py
"""

from typing import Final

import e00173_common as common
import numpy as np
import polars as pl
from studies.dp_calendar import bank_holiday_names

GROUPS: Final[dict[str, tuple[tuple[int, int], ...]]] = {
    "24-25 Dec": ((12, 24), (12, 25)),
    "26 Dec": ((12, 26),),
    "27-31 Dec": ((12, 27), (12, 28), (12, 29), (12, 30), (12, 31)),
    "1-2 Jan": ((1, 1), (1, 2)),
}
SCOPES: Final[dict[str, tuple[float, float]]] = {"all": (0.0, 168.0), "D+1": (15.0, 39.0)}
TAIL_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
N_RESAMPLES: Final[int] = 4000
RNG_SEED: Final[int] = 173


def _mean(series: pl.Series) -> float:
    """Return a series' mean as a float."""
    return float(series.to_numpy().mean())


def _sum_ratio(numerator: pl.Series, denominator: pl.Series) -> float:
    """Return the sum of one series over the sum of another."""
    return float(numerator.to_numpy().sum() / denominator.to_numpy().sum())


def pinball(*, actual: pl.Expr, quantile: pl.Expr, level: float) -> pl.Expr:
    """Return the pinball loss of one quantile."""
    error = actual - quantile
    return pl.max_horizontal(level * error, (level - 1.0) * error)


def load(*, arm: str) -> pl.DataFrame:
    """Return an arm's forecast rows with their losses, event group, and reference flag."""
    folder = common.forecast_dir(arm=arm)
    frames = [pl.read_parquet(path) for path in sorted(folder.glob("*.parquet"))]
    frame = pl.concat(frames).filter(~pl.col("switching_mask"))
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    holidays = pl.Series(sorted(bank_holiday_names()), dtype=pl.Date)
    group_expr = pl.lit(None, dtype=pl.String)
    for name, days in GROUPS.items():
        in_group = pl.lit(False)
        for month, day in days:
            in_group = in_group | ((local.dt.month() == month) & (local.dt.day() == day))
        group_expr = pl.when(in_group).then(pl.lit(name)).otherwise(group_expr)
    levels = dict(zip(common.QUANTILES, common.LEVELS, strict=True))
    qs = sum(
        pinball(actual=pl.col("actual"), quantile=pl.col(q), level=level)
        for q, level in levels.items()
    )
    frame = frame.with_columns(
        local_date=local.dt.date(),
        half_hour=pl.col("local_hh"),
        group=group_expr,
        qs=2.0 * qs / len(levels),
        abs_mean=(pl.col("actual") - pl.col("mean")).abs(),
        signed=pl.col("actual") - pl.col("mean"),
        median_minus_mean=pl.col("q50") - pl.col("mean"),
        **{f"cover_{q}": (pl.col("actual") <= pl.col(q)).cast(pl.Float64) for q in TAIL_LEVELS},
        **{
            f"pin_{q}": pinball(actual=pl.col("actual"), quantile=pl.col(q), level=levels[q])
            for q in TAIL_LEVELS
        },
    )
    ordinary = (
        (pl.col("local_date").dt.weekday() <= 5)
        & ~pl.col("local_date").is_in(holidays)
        & pl.col("group").is_null()
        & ~(
            ((pl.col("local_date").dt.month() == 12) & (pl.col("local_date").dt.day() >= 24))
            | ((pl.col("local_date").dt.month() == 1) & (pl.col("local_date").dt.day() <= 2))
        )
        & (
            (pl.col("local_date").dt.month() == 1)
            | ((pl.col("local_date").dt.month() == 12) & (pl.col("local_date").dt.day() >= 8))
        )
    )
    return frame.with_columns(reference=ordinary)


def lead_scope(*, frame: pl.DataFrame, scope: str) -> pl.DataFrame:
    """Keep the rows whose lead is in the scope."""
    low, high = SCOPES[scope]
    return frame.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") <= high))


def date_arrays(*, rows: pl.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return, per local date, the sum and the count of a metric by half-hour (dates x 48)."""
    dates = sorted(rows["local_date"].unique().to_list())
    index = {day: i for i, day in enumerate(dates)}
    total = np.zeros((len(dates), 48))
    count = np.zeros((len(dates), 48))
    grouped = rows.group_by(["local_date", "half_hour"]).agg(s=pl.col(metric).sum(), n=pl.len())
    for day, hh, s, n in grouped.iter_rows():
        total[index[day], hh] = s
        count[index[day], hh] = n
    return total, count


def ratio_of(
    *, e_sum: np.ndarray, e_cnt: np.ndarray, r_sum: np.ndarray, r_cnt: np.ndarray
) -> float:
    """Return the matched ratio from summed per-half-hour arrays (each of length 48)."""
    weights = e_cnt
    usable = (weights > 0) & (r_cnt > 0)
    if not usable.any():
        return float("nan")
    observed = e_sum[usable].sum() / e_cnt[usable].sum()
    expected = (weights[usable] * (r_sum[usable] / r_cnt[usable])).sum() / weights[usable].sum()
    return float(observed / expected)


def matched_ratio(*, event: pl.DataFrame, reference: pl.DataFrame, metric: str) -> dict:
    """Return the matched ratio of a metric with a 95% interval from resampling local dates."""
    event = event.filter(pl.col(metric).is_not_null())
    reference = reference.filter(pl.col(metric).is_not_null())
    if event.is_empty() or reference.is_empty():
        return {"n_event_rows": event.height, "n_reference_rows": reference.height}
    e_s, e_c = date_arrays(rows=event, metric=metric)
    r_s, r_c = date_arrays(rows=reference, metric=metric)
    point = ratio_of(e_sum=e_s.sum(0), e_cnt=e_c.sum(0), r_sum=r_s.sum(0), r_cnt=r_c.sum(0))
    rng = np.random.default_rng(RNG_SEED)
    draws = np.empty(N_RESAMPLES)
    for i in range(N_RESAMPLES):
        ei = rng.integers(0, len(e_s), len(e_s))
        ri = rng.integers(0, len(r_s), len(r_s))
        draws[i] = ratio_of(
            e_sum=e_s[ei].sum(0), e_cnt=e_c[ei].sum(0), r_sum=r_s[ri].sum(0), r_cnt=r_c[ri].sum(0)
        )
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return {
        "n_event_rows": event.height,
        "n_event_dates": event["local_date"].n_unique(),
        "n_reference_rows": reference.height,
        "n_reference_dates": reference["local_date"].n_unique(),
        "ratio": point,
        "low": float(low),
        "high": float(high),
        "event_mean": _mean(event[metric]),
    }


def christmas_window(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the rows in any named group."""
    return frame.filter(pl.col("group").is_not_null())


def ratio_tables(*, frame: pl.DataFrame) -> list[dict]:
    """Return ratios by scope, metric, season, site, and group (each pooled over the rest)."""
    out = []
    for scope in SCOPES:
        scoped = lead_scope(frame=frame, scope=scope)
        for metric in ("qs", "abs_mean"):
            seasons = [*common.SEASONS, "all"]
            for season in seasons:
                part = scoped if season == "all" else scoped.filter(pl.col("season") == season)
                for site in ["all", *common.ALL_SITES]:
                    sited = part if site == "all" else part.filter(pl.col("site") == site)
                    event = christmas_window(frame=sited)
                    reference = sited.filter(pl.col("reference"))
                    out.append(
                        {
                            "scope": scope, "metric": metric, "season": season, "site": site,
                            "group": "christmas",
                            **matched_ratio(event=event, reference=reference, metric=metric),
                        }
                    )  # fmt: skip
                    if site != "all":
                        continue
                    for group in GROUPS:
                        out.append(  # noqa: PERF401
                            {
                                "scope": scope, "metric": metric, "season": season, "site": site,
                                "group": group,
                                **matched_ratio(
                                    event=sited.filter(pl.col("group") == group),
                                    reference=reference,
                                    metric=metric,
                                ),
                            }
                        )  # fmt: skip
    return out


def bias_tables(*, frame: pl.DataFrame) -> list[dict]:
    """Return the signed error as a share of measured power, by season, site, and group."""
    out = []
    scoped = lead_scope(frame=frame, scope="all")
    groups = [*GROUPS, "25-26 Dec", "ordinary weekday"]
    for season in [*common.SEASONS, "all"]:
        part = scoped if season == "all" else scoped.filter(pl.col("season") == season)
        for site in ["all", *common.ALL_SITES]:
            sited = part if site == "all" else part.filter(pl.col("site") == site)
            for group in groups:
                if group == "25-26 Dec":
                    rows = sited.filter(
                        (pl.col("local_date").dt.month() == 12)
                        & pl.col("local_date").dt.day().is_in([25, 26])
                    )
                elif group == "ordinary weekday":
                    rows = sited.filter(pl.col("reference"))
                else:
                    rows = sited.filter(pl.col("group") == group)
                if rows.is_empty():
                    continue
                out.append(
                    {
                        "season": season, "site": site, "group": group, "rows": rows.height,
                        "signed_mean": _mean(rows["signed"]),
                        "measured_mean": _mean(rows["actual"]),
                        "signed_share": _sum_ratio(rows["signed"], rows["actual"]),
                        "median_minus_mean": rows["median_minus_mean"].mean(),
                    }
                )  # fmt: skip
    return out


def tail_tables(*, frame: pl.DataFrame) -> list[dict]:
    """Return coverage and pinball loss at the four tail quantiles by season and group."""
    out = []
    scoped = lead_scope(frame=frame, scope="all")
    for season in [*common.SEASONS, "all"]:
        part = scoped if season == "all" else scoped.filter(pl.col("season") == season)
        for name, rows in (
            *[(g, part.filter(pl.col("group") == g)) for g in GROUPS],
            ("christmas", christmas_window(frame=part)),
            ("ordinary weekday", part.filter(pl.col("reference"))),
        ):
            if rows.is_empty() or rows["q05"].null_count() == rows.height:
                continue
            out.append(
                {
                    "season": season, "group": name, "rows": rows.height,
                    **{f"cover_{q}": _mean(rows[f"cover_{q}"]) for q in TAIL_LEVELS},
                    **{f"pin_{q}": _mean(rows[f"pin_{q}"]) for q in TAIL_LEVELS},
                    "qs": _mean(rows["qs"]),
                }
            )  # fmt: skip
    return out


def leave_one_out(*, frame: pl.DataFrame) -> list[dict]:
    """Return each season's pooled quantile-score ratio with each substation left out."""
    out = []
    scoped = lead_scope(frame=frame, scope="all")
    for season in common.SEASONS:
        part = scoped.filter(pl.col("season") == season)
        for site in sorted(part["site"].unique().to_list()):
            rest = part.filter(pl.col("site") != site)
            ratio = matched_ratio(
                event=christmas_window(frame=rest), reference=rest.filter(pl.col("reference")),
                metric="qs",
            )  # fmt: skip
            out.append({"season": season, "left_out": site, "ratio": ratio.get("ratio")})
    return out


def control(*, frame: pl.DataFrame) -> list[dict]:
    """Return the positive control: the 2024 and 2025 seasons at the five decision substations."""
    part = lead_scope(frame=frame, scope="all").filter(
        pl.col("season").is_in([2024, 2025]) & pl.col("site").is_in(list(common.DECISION_SITES))
    )
    out = []
    for season in ("both", 2024, 2025):
        rows = part if season == "both" else part.filter(pl.col("season") == season)
        ratio = matched_ratio(
            event=christmas_window(frame=rows), reference=rows.filter(pl.col("reference")),
            metric="qs",
        )  # fmt: skip
        out.append(
            {
                "season": season, **ratio,
                "ordinary_weekday_qs": _mean(rows.filter(pl.col("reference"))["qs"]),
                "all_rows_qs": _mean(rows["qs"]),
            }
        )  # fmt: skip
    return out


def matched_control(*, frame: pl.DataFrame) -> dict:
    """Return the arm's score against experiment 155's arm D on the rows they share.

    The rows are the decision substations' runs of November 2025 to January 2026 that both
    experiments forecast (experiment 155's scored runs, deduplicated), paired on the substation, the
    origin, and the valid time. Experiment 155's forecast is driven by ensemble-mean weather, so a
    lower score here is expected, and its arm has the winter night offset and a 24-month window.
    """
    levels = dict(zip(common.QUANTILES, common.LEVELS, strict=True))
    reference = pl.concat(
        [
            pl.read_parquet(path).with_columns(site=pl.lit(path.stem.rsplit("_", 1)[1]))
            for path in sorted(common.E155_PREDS_DIR.glob("d_scored_D*.parquet"))
        ]
    ).select(
        "site", "power_fcst_init_time", "valid_time", "lead_hours",
        *[pl.col(q).alias(f"ref_{q}") for q in common.QUANTILES],
    )  # fmt: skip
    reference = reference.unique(subset=["site", "power_fcst_init_time", "valid_time"])
    mine = frame.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
    joined = mine.join(
        reference.drop("lead_hours"), on=["site", "power_fcst_init_time", "valid_time"], how="inner"
    )
    ref_qs = (
        2.0
        * sum(
            pinball(actual=pl.col("actual"), quantile=pl.col(f"ref_{q}"), level=level)
            for q, level in levels.items()
        )
        / len(levels)
    )
    joined = joined.filter(pl.col("qs").is_not_null()).with_columns(ref_qs=ref_qs)
    weekday = joined.filter(pl.col("reference"))
    return {
        "weekday_rows": weekday.height,
        "weekday_this_qs": _mean(weekday["qs"]),
        "weekday_reference_qs": _mean(weekday["ref_qs"]),
        "rows": joined.height,
        "this_qs": _mean(joined["qs"]),
        "reference_qs": _mean(joined["ref_qs"]),
        "difference": _mean(joined["qs"] - joined["ref_qs"]),
    }


def subset_tables(*, frame: pl.DataFrame) -> list[dict]:
    """Return ratios and bias on fits with at least N days of usable training rows (post hoc)."""
    out = []
    scoped = lead_scope(frame=frame, scope="all")
    for days in common.ADEQUATE_TRAINING_DAYS:
        kept = scoped.filter(pl.col("n_train_rows") >= days * 48)
        for season in [*common.SEASONS, "all"]:
            part = kept if season == "all" else kept.filter(pl.col("season") == season)
            for metric in ("qs", "abs_mean"):
                ratio = matched_ratio(
                    event=christmas_window(frame=part),
                    reference=part.filter(pl.col("reference")),
                    metric=metric,
                )
                out.append(
                    {
                        "min_days": days, "season": season, "metric": metric,
                        "n_fits": part.select("site", "season").unique().height, **ratio,
                    }
                )  # fmt: skip
            holiday = part.filter(
                (pl.col("local_date").dt.month() == 12)
                & pl.col("local_date").dt.day().is_in([25, 26])
            )
            if not holiday.is_empty():
                out[-1]["signed_share_25_26"] = _sum_ratio(holiday["signed"], holiday["actual"])
        by_site = [
            {"min_days": days, "site": site, **matched_ratio(
                event=christmas_window(frame=kept.filter(pl.col("site") == site)),
                reference=kept.filter((pl.col("site") == site) & pl.col("reference")),
                metric="qs",
            )}
            for site in common.ALL_SITES
            if not kept.filter(pl.col("site") == site).is_empty()
        ]  # fmt: skip
        out.extend({"kind": "site", **row} for row in by_site)
    return out


def coverage_of_data(*, frame: pl.DataFrame) -> list[dict]:
    """Return the substations and rows present in each season."""
    return (
        frame.group_by("season")
        .agg(sites=pl.col("site").n_unique(), rows=pl.len())
        .sort("season")
        .to_dicts()
    )


def main() -> None:
    """Score every season of every arm and write `report_<arm>.json`."""
    for arm in common.ARMS:
        write_arm(arm=arm)


def write_arm(*, arm: str) -> None:
    """Score one arm and write its report."""
    frame = load(arm=arm)
    result = {
        "arm": arm,
        "matched_control": matched_control(frame=frame),
        "adequate_subset": subset_tables(frame=frame),
        "data": coverage_of_data(frame=frame),
        "ratios": ratio_tables(frame=frame),
        "bias": bias_tables(frame=frame),
        "tails": tail_tables(frame=frame),
        "leave_one_out": leave_one_out(frame=frame),
        "control": control(frame=frame),
    }
    common.write_json_atomic(path=common.WORK_DIR / "report" / f"report_{arm}.json", content=result)
    print(f"report {arm} written")


if __name__ == "__main__":
    main()
