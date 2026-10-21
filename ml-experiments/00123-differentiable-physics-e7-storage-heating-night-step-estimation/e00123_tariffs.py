"""Post hoc, exploratory: local-clock tariff edges in Primary demand, winters 2021-22 to 2024-25.

The tariff windows are unverified hypotheses (they are not sources): a step up when a cheap window
opens and down when it closes, on the local clock, for Intelligent Octopus Go (23:30 to 05:30),
Octopus Go (00:30 to 05:30), and the Cosy heat-pump tariff (04:00 to 07:00, 13:00 to 16:00, and
22:00 to 24:00 on winter weekdays).

The statistic for an edge at local half-hour start `t` is the rise entering that half-hour minus
the median rise at the four half-hours either side, in points, on one night or day. Night targets
use British Summer Time nights (April to October), when the UTC-fixed legacy step falls an hour
later on the local clock; Cosy targets use winter weekdays (GMT months, where local time and UTC
agree). Intervals resample 14-day blocks. Placebo edges (half-hours with no hypothesised tariff
edge) give the false-positive rate, and an injected step of known size checks the recovery.

    uv run python ml-experiments/00123-*/e00123_tariffs.py
"""

import json
from datetime import timedelta
from typing import Final
from zoneinfo import ZoneInfo

import e00123_common as common
import e00123_report as report
import numpy as np
import polars as pl

NIGHT_EDGES: Final[dict[str, tuple[int, int, float]]] = {
    "Intelligent Octopus Go opens 23:30": (0, 23 * 60 + 30, 1.0),
    "Intelligent Octopus Go closes 05:30": (1, 5 * 60 + 30, -1.0),
    "Octopus Go opens 00:30": (1, 30, 1.0),
    "Octopus Go closes 05:30": (1, 5 * 60 + 30, -1.0),
}
"""Edge name -> (day offset from the evening date, local minute, expected sign)."""
COSY_EDGES: Final[dict[str, tuple[int, float]]] = {
    "Cosy opens 04:00": (4 * 60, 1.0),
    "Cosy closes 07:00": (7 * 60, -1.0),
    "Cosy opens 13:00": (13 * 60, 1.0),
    "Cosy closes 16:00": (16 * 60, -1.0),
    "Cosy opens 22:00": (22 * 60, 1.0),
}
PLACEBO_MINUTES: Final[tuple[int, ...]] = (21 * 60, 3 * 60 + 30, 10 * 60, 18 * 60 + 30)
"""Half-hours with no hypothesised edge: 21:00, 03:30, 10:00, and 18:30."""
INJECTED: Final[tuple[float, ...]] = (0.0, 1.0, 2.0, 4.0)
MIN_DAYS: Final[int] = 30
N_BOOT: Final[int] = 400
NEIGHBOURS: Final[tuple[int, ...]] = (-4, -3, -2, -1, 1, 2, 3, 4)


def rises(
    *, lookup: dict, date: common.date, scale: float, minutes: list[int]
) -> dict[int, float] | None:
    """Return the rise entering each local half-hour start of a day, in points, or None."""
    zone = ZoneInfo(common.LOCAL_TIME_ZONE)
    midnight = common.datetime.combine(date, common.time.min).replace(tzinfo=zone)
    out: dict[int, float] = {}
    for minute in minutes:
        now = lookup.get(
            (midnight + timedelta(minutes=minute)).astimezone(common.UTC) + timedelta(minutes=30)
        )
        before = lookup.get(
            (midnight + timedelta(minutes=minute - 30)).astimezone(common.UTC)
            + timedelta(minutes=30)
        )
        if now is None or before is None:
            return None
        out[minute] = (now - before) * scale
    return out


def edge_statistic(*, lookup: dict, date: common.date, scale: float, minute: int) -> float | None:
    """Return the rise at `minute` minus the median rise at the neighbouring half-hours."""
    wanted = [minute, *(minute + 30 * k for k in NEIGHBOURS)]
    values = rises(lookup=lookup, date=date, scale=scale, minutes=wanted)
    if values is None:
        return None
    return values[minute] - float(np.median([values[m] for m in wanted[1:]]))


def block_interval(
    *, days: list[common.date], values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return the mean and its 90% interval from resampling 14-day blocks."""
    block = np.array([(d - common.date(2019, 1, 1)).days // common.BLOCK_DAYS for d in days])
    labels = np.unique(block)
    members = {label: values[block == label] for label in labels}
    draws = [
        float(np.concatenate([members[label] for label in rng.choice(labels, len(labels))]).mean())
        for _ in range(N_BOOT)
    ]
    low, high = np.quantile(draws, [0.05, 0.95])
    return float(values.mean()), float(low), float(high)


def summarise(*, days: list[common.date], values: list[float], rng: np.random.Generator) -> dict:
    """Return estimate, interval, nights, and label for one edge, series, and year."""
    if len(values) < MIN_DAYS:
        return {"nights": len(values), "label": "cannot tell"}
    mean, low, high = block_interval(days=days, values=np.array(values), rng=rng)
    return {"nights": len(values), "estimate": mean, "low": low, "high": high}


def main() -> None:
    """Compute the edges, the placebo and injection checks, and print the tables."""
    infos = common.series_infos()
    table = pl.read_parquet(common.WORK_DIR / "night_tables.parquet")
    power = common.load_power(
        time_series_ids=[infos[s].time_series_id for s in common.FITTED_SERIES]
    )
    end = report.night_end(winters=common.HEADLINE_WINTERS)
    rng = np.random.default_rng(2027)
    rows: list[dict] = []
    for site in common.FITTED_SERIES:
        info = infos[site]
        own = power.filter(pl.col("time_series_id") == info.time_series_id)
        lookup = dict(zip(own["time"].to_list(), own["power"].to_list(), strict=True))
        scale = 100.0 / info.effective_capacity_mw
        bst = table.filter(
            (pl.col("site") == site)
            & pl.col("season").is_in(["bst_other", "summer"])
            & ~pl.col("excluded")
            & (pl.col("evening") >= common.date(2021, 3, 28))
            & (pl.col("evening") < end)
        )["evening"].to_list()
        cosy_days = table.filter(
            (pl.col("site") == site)
            & (pl.col("season") == "gmt")
            & ~pl.col("excluded")
            & (pl.col("day_type") == "weekday")
            & pl.col("winter").is_in(list(common.HEADLINE_WINTERS))
        )["evening"].to_list()
        jobs = []
        for name, (offset, minute, sign) in NIGHT_EDGES.items():
            jobs.append((name, "tariff edge", bst, offset, minute, sign))
        jobs.extend(
            (
                f"placebo {minute // 60:02d}:{minute % 60:02d}",
                "placebo",
                bst,
                1 if minute < 12 * 60 else 0,
                minute,
                1.0,
            )
            for minute in PLACEBO_MINUTES
        )
        jobs.extend(
            (name, "tariff edge", cosy_days, 0, minute, sign)
            for name, (minute, sign) in COSY_EDGES.items()
        )
        for name, kind, days, offset, minute, sign in jobs:
            by_year: dict[int, tuple[list, list]] = {}
            for day in days:
                target = day + timedelta(days=offset)
                value = edge_statistic(lookup=lookup, date=target, scale=scale, minute=minute)
                if value is not None:
                    key = (
                        day.year
                        if kind != "tariff edge" or name.startswith(("Intelligent", "Octopus"))
                        else common.winter_of(evening=day) or day.year
                    )
                    by_year.setdefault(key, ([], []))[0].append(day)
                    by_year[key][1].append(value)
            for year, (ds, vs) in sorted(by_year.items()):
                rows.append(
                    {
                        "site": site,
                        "edge": name,
                        "kind": kind,
                        "sign": sign,
                        "year": year,
                        **summarise(days=ds, values=vs, rng=rng),
                    }
                )
    (common.WORK_DIR / "tariffs.json").write_text(json.dumps(rows, indent=1))
    frame = pl.DataFrame(rows, infer_schema_length=None)
    ok = frame.filter(pl.col("estimate").is_not_null()).with_columns(
        visible=pl.when(pl.col("sign") > 0).then(pl.col("low") > 0).otherwise(pl.col("high") < 0),
        half_width=(pl.col("high") - pl.col("low")) / 2.0,
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=2, tbl_width_chars=220):
        print(
            "== placebo edges: share of series-years whose 90% interval excludes 0 (5% expected) =="
        )
        placebo = ok.filter(pl.col("kind") == "placebo")
        print(
            placebo.group_by("edge").agg(
                share_excluding_zero=((pl.col("low") > 0) | (pl.col("high") < 0)).mean(),
                cells=pl.len(),
            )
        )
        print("== detection limit: median half-width of the 90% interval, points ==")
        print(
            ok.group_by("kind").agg(
                median_half_width=pl.col("half_width").median(),
                max_half_width=pl.col("half_width").max(),
            )
        )
        print("== tariff edges: series-years in which the expected-sign edge is visible ==")
        edges = ok.filter(pl.col("kind") == "tariff edge")
        print(
            edges.group_by("edge")
            .agg(
                series_years=pl.len(),
                visible=pl.col("visible").sum(),
                median_estimate=pl.col("estimate").median(),
            )
            .sort("edge")
        )
        print("== tariff edges: substations visible in at least two years ==")
        print(
            edges.filter(pl.col("visible"))
            .group_by("edge", "site")
            .agg(years=pl.len(), mean_estimate=pl.col("estimate").mean())
            .filter(pl.col("years") >= 2)
            .sort("edge", "site")
        )
        print("== injected step: recovery of S at one real edge series ==")
        sample = edges.row(0, named=True)
        for injected in INJECTED:
            shifted = (
                sample["estimate"] + injected,
                sample["low"] + injected,
                sample["high"] + injected,
            )
            print(
                f"S={injected:.0f}: estimate {shifted[0]:.2f}, "
                f"interval {shifted[1]:.2f} to {shifted[2]:.2f} (shifted by exactly S)"
            )


def change_over_years() -> None:
    """Print the change in each edge statistic from the first to the last year, with placebos.

    The ordinary daily load shape gives every half-hour a non-zero rise against its neighbours, so
    the level of the edge statistic is not a test (the placebo edges show this). The change between
    years cancels the load shape and keeps tariff take-up. A change counts as visible when it has
    the
    expected sign and exceeds 1.645 combined standard errors (each from its interval's half-width).
    """
    rows = json.loads((common.WORK_DIR / "tariffs.json").read_text())
    frame = pl.DataFrame(rows, infer_schema_length=None).filter(pl.col("estimate").is_not_null())
    out = []
    for (site, edge), group in frame.group_by(["site", "edge"]):
        ordered = group.sort("year")
        if ordered.height < 2:
            continue
        first, last = ordered.row(0, named=True), ordered.row(-1, named=True)
        se = float(
            np.hypot((first["high"] - first["low"]) / 3.29, (last["high"] - last["low"]) / 3.29)
        )
        change = last["estimate"] - first["estimate"]
        out.append(
            {
                "site": site,
                "edge": edge,
                "kind": first["kind"],
                "first_year": first["year"],
                "last_year": last["year"],
                "change": change,
                "se": se,
                "visible": change * first["sign"] > 1.645 * se,
                "either_way": abs(change) > 1.645 * se,
            }
        )
    changes = pl.DataFrame(out)
    changes.write_parquet(common.WORK_DIR / "tariff_changes.parquet")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=2, tbl_width_chars=200):
        print("== change from first to last year: placebo false-positive rate (either sign) ==")
        print(
            changes.filter(pl.col("kind") == "placebo")
            .group_by("edge")
            .agg(share_changed=pl.col("either_way").mean(), median_se=pl.col("se").median())
        )
        print("== change from first to last year: tariff edges, expected sign ==")
        print(
            changes.filter(pl.col("kind") == "tariff edge")
            .group_by("edge")
            .agg(
                substations=pl.len(),
                visible=pl.col("visible").sum(),
                median_change=pl.col("change").median(),
                largest_change=pl.col("change").abs().max(),
            )
            .sort("edge")
        )
        limit = float(
            changes.filter(pl.col("kind") == "placebo")["change"].abs().quantile(0.9) or 0.0
        )
        beyond = changes.filter(pl.col("kind") == "tariff edge").with_columns(
            beyond_placebo=(
                pl.col("change")
                * pl.when(pl.col("edge").str.contains("opens")).then(1.0).otherwise(-1.0)
            )
            > limit
        )
        print(f"== empirical detection limit: placebo 90th pct |change| = {limit:.2f} points ==")
        print(
            beyond.group_by("edge")
            .agg(beyond_limit=pl.col("beyond_placebo").sum(), substations=pl.len())
            .sort("edge")
        )
        print(
            beyond.filter(pl.col("beyond_placebo"))
            .select("edge", "site", "change")
            .sort("edge", "site")
        )
        print("== substations where the expected-sign change is visible ==")
        print(
            changes.filter(pl.col("kind") == "tariff edge", pl.col("visible"))
            .select("edge", "site", "first_year", "last_year", "change", "se")
            .sort("edge", "site")
        )


if __name__ == "__main__":
    main()
    change_over_years()
