"""Score Christmas, Easter, and bank holidays against ordinary days (Part 2 of experiment 170).

Every event class is compared with ordinary days at the same local half-hours of the day, so a
class that falls on quiet half-hours is not blamed for them. The ratio is the mean quantile score
over the event rows divided by the same mean taken over ordinary rows, with each ordinary
half-hour weighted by how many event rows it has. The tuning months fit the spread, so only the
replicate year (2024-25) and the scored year (2025-26) are scored.

| Class | Local dates |
|---|---|
| `christmas` | 24 December to 2 January |
| `easter` | Good Friday, Easter Saturday, Easter Sunday, and Easter Monday |
| `bank_holiday` | May Day, Spring Bank Holiday, Late Summer Bank Holiday, and one-off holidays |

| Reference | Local dates |
|---|---|
| `weekday` | Monday to Friday outside any holiday period and outside 24 December to 2 January |
| `weekday_winter` | `weekday`, November to March only |
| `sunday` | Sundays outside any holiday period and outside 24 December to 2 January |

The report also gives the with-minus-without recipe R and drift differences on the event days, by
Christmas, and the number of dates on which the with arm scored worse.

    uv run --no-sync python ml-experiments/00170-*/e00170_events.py
"""

from typing import Final

import e00170_report as report
import e00170_stage2_common as common
import polars as pl
from studies.dp_calendar import HOLIDAY_PERIODS

ARMS: Final[tuple[str, ...]] = (
    "a155",
    "c155",
    "d155",
    "s24_a",
    "s24_c",
    "s24_d",
    "sall_a",
    "sall_d",
)
PERIODS: Final[tuple[str, ...]] = ("replicate", "scored")
EASTER_NAMES: Final[tuple[str, ...]] = (
    "Good Friday", "Easter Saturday", "Easter Sunday", "Easter Monday",
)  # fmt: skip
BANK_HOLIDAY_NAMES: Final[tuple[str, ...]] = (
    "May Day", "Spring Bank Holiday", "Late Summer Bank Holiday", "one-off bank holiday",
)  # fmt: skip
SUNDAY: Final[int] = 7
LEAD_SCOPES: Final[dict[str, tuple[float, float]]] = {"all": (0.0, 168.0), "D+1": (15.0, 39.0)}
DIFFERENCES: Final[dict[str, tuple[str, str]]] = {
    "R without drift (c155 minus a155)": ("c155", "a155"),
    "R with drift (s24_c minus s24_a)": ("s24_c", "s24_a"),
    "drift without R (s24_a minus a155)": ("s24_a", "a155"),
    "drift with R (s24_c minus c155)": ("s24_c", "c155"),
    "drift with offset and R (s24_d minus d155)": ("s24_d", "d155"),
}


def classify(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add each row's event class and reference class, `None` where it has none."""
    names = list(HOLIDAY_PERIODS)
    easter = [names.index(n) for n in EASTER_NAMES]
    bank = [names.index(n) for n in BANK_HOLIDAY_NAMES]
    month, day = pl.col("local_date").dt.month(), pl.col("local_date").dt.day()
    window = ((month == 12) & (day >= 24)) | ((month == 1) & (day <= 2))
    ordinary = (pl.col("holiday_code") < 0) & ~window
    weekday = pl.col("local_date").dt.weekday()
    return frame.with_columns(
        event=pl.when(window)
        .then(pl.lit("christmas"))
        .when(pl.col("holiday_code").is_in(easter))
        .then(pl.lit("easter"))
        .when(pl.col("holiday_code").is_in(bank))
        .then(pl.lit("bank_holiday"))
        .otherwise(None),
        reference=pl.when(ordinary & (weekday <= 5))
        .then(pl.lit("weekday"))
        .when(ordinary & (weekday == SUNDAY))
        .then(pl.lit("sunday"))
        .otherwise(None),
        winter=pl.col("valid_month").is_in(list(common.WINTER_MONTHS)),
    )


def matched_ratio(*, event: pl.DataFrame, reference: pl.DataFrame, metric: str = "qs") -> dict:
    """Return the event mean, the reference mean at the event's half-hours, and their ratio."""
    if event.is_empty() or reference.is_empty():
        return {"n_event_rows": event.height, "n_reference_rows": reference.height}
    weights = event.group_by("half_hour").agg(w=pl.len())
    ordinary = reference.group_by("half_hour").agg(r=pl.col(metric).mean())
    joined = weights.join(ordinary, on="half_hour", how="inner")
    w, r = joined["w"].to_numpy().astype(float), joined["r"].to_numpy()
    expected = float((w * r).sum() / w.sum())
    observed = float(
        event.join(joined.select("half_hour"), on="half_hour")[metric].to_numpy().mean()
    )
    return {
        "n_event_rows": event.height,
        "n_reference_rows": reference.height,
        "event_mean": observed,
        "reference_at_same_half_hours": expected,
        "ratio": observed / expected,
    }


def ratio_tables(*, losses: pl.DataFrame) -> list[dict]:
    """Return the matched ratio by arm, event class, reference, scope, period, and Christmas."""
    out = []
    for arm in ARMS:
        arm_rows = losses.filter(pl.col("arm") == arm)
        for scope_name, (low, high) in LEAD_SCOPES.items():
            scoped = report.lead_filter(frame=arm_rows, low=low, high=high)
            for period in (*PERIODS, "both"):
                part = scoped if period == "both" else scoped.filter(pl.col("period") == period)
                for event_name in ("christmas", "easter", "bank_holiday"):
                    event = part.filter(pl.col("event") == event_name)
                    for reference_name in ("weekday", "weekday_winter", "sunday"):
                        reference = part.filter(
                            (pl.col("reference") == reference_name.removesuffix("_winter"))
                            & (pl.col("winter") if reference_name.endswith("_winter") else True)
                        )
                        out.append(
                            {
                                "arm": arm,
                                "scope": scope_name,
                                "period": period,
                                "event": event_name,
                                "reference": reference_name,
                                **matched_ratio(event=event, reference=reference),
                            }
                        )
    return out


def difference_tables(*, losses: pl.DataFrame) -> list[dict]:
    """Return each with-minus-without difference on event days, by event, scope, and period."""
    out = []
    keys = ["site", "time_series_id", "power_fcst_init_time", "valid_time", "seed"]
    for label, (treatment, reference) in DIFFERENCES.items():
        t = losses.filter(pl.col("arm") == treatment).select(
            [*keys, "qs", "abs_mean", "event", "period", "local_date", "lead_hours"]
        )
        r = losses.filter(pl.col("arm") == reference).select([*keys, "qs", "abs_mean"])
        paired = t.join(r, on=keys, suffix="_ref").with_columns(
            d_qs=pl.col("qs") - pl.col("qs_ref"), d_abs=pl.col("abs_mean") - pl.col("abs_mean_ref")
        )
        for scope_name, (low, high) in LEAD_SCOPES.items():
            scoped = report.lead_filter(frame=paired, low=low, high=high)
            for event_name in ("christmas", "easter", "bank_holiday"):
                event = scoped.filter(pl.col("event") == event_name)
                groups = [("both", event)] + [
                    (p, event.filter(pl.col("period") == p)) for p in PERIODS
                ]
                for period, part in groups:
                    if part.is_empty():
                        continue
                    by_date = part.group_by("local_date").agg(d=pl.col("d_qs").mean())
                    by_site = part.group_by("site").agg(d=pl.col("d_qs").mean())
                    out.append(
                        {
                            "difference": label,
                            "scope": scope_name,
                            "event": event_name,
                            "period": period,
                            "rows": part.height,
                            "dates": by_date.height,
                            "mean_qs": float(part["d_qs"].to_numpy().mean()),
                            "mean_abs_error_of_mean": float(part["d_abs"].to_numpy().mean()),
                            "dates_worse": int((by_date["d"] > 0).sum()),
                            "sites_worse": int((by_site["d"] > 0).sum()),
                            "n_sites": by_site.height,
                        }
                    )
    return out


def day_table(*, losses: pl.DataFrame) -> list[dict]:
    """Return the mean quantile score of every Christmas date by arm and period (D+1 leads)."""
    scoped = report.lead_filter(frame=losses, low=15.0, high=39.0).filter(
        pl.col("event") == "christmas"
    )
    return (
        scoped.group_by(["arm", "period", "local_date"])
        .agg(qs=pl.col("qs").mean(), n=pl.len())
        .sort(["arm", "period", "local_date"])
        .with_columns(local_date=pl.col("local_date").cast(pl.String))
        .to_dicts()
    )


def main() -> None:
    """Score the event days of the replicate year and the scored year, and write the report."""
    cache = common.load_cache()
    frames = {}
    for period in PERIODS:
        loaded = {
            arm: frame
            for arm in ARMS
            if (frame := report.load(arm=arm, period=period, cache=cache)) is not None
        }
        for arm, frame in report.common_rows(frames=loaded).items():
            frames.setdefault(arm, []).append(frame.with_columns(period=pl.lit(period)))
    stacked = {arm: pl.concat(parts) for arm, parts in frames.items()}
    # `stack` builds the bootstrap loss frame; the period is re-attached from the origin dates.
    losses = report.stack(frames=stacked)
    origins = {period: common.period_origins(period=period) for period in PERIODS}
    losses = losses.with_columns(
        period=pl.when(pl.col("power_fcst_init_time").is_in(origins["replicate"]))
        .then(pl.lit("replicate"))
        .otherwise(pl.lit("scored"))
    )
    losses = classify(frame=losses).filter(pl.col("site").is_in(list(common.DECISION_SITES)))
    result = {
        "n_rows": losses.height / len(ARMS),
        "ratios": ratio_tables(losses=losses),
        "differences": difference_tables(losses=losses),
        "days": day_table(losses=losses),
        "events_found": (
            losses.filter(pl.col("arm") == ARMS[0])
            .group_by(["period", "event"])
            .agg(dates=pl.col("local_date").n_unique(), rows=pl.len())
            .sort(["period", "event"])
            .to_dicts()
        ),
    }
    common.write_json_atomic(path=common.WORK_DIR / "report" / "events.json", content=result)
    print("events written")


if __name__ == "__main__":
    main()
