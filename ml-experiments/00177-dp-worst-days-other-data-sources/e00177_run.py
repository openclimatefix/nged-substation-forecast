"""Experiment 177, stage 1: do data sources we already hold explain the 37 unlabelled worst days?

Computes four indicators (S1 raw switching log, S2 metered-generator step, S3 reading quality,
S4 the substation's own history) for every scored (site, local_date) of experiment 172, then
compares the 37 days with 1,000 draws of 37 matched ordinary days. Needs `openpyxl` to read the
switching log: `uv run --no-sync --with openpyxl python -I <this file>`.
"""

import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import openpyxl  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

SEED: Final[int] = 177
N_DRAWS: Final[int] = 1000
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00177-dp-worst-days-other-data-sources" / "work"
E172: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00172-dp-error-budget-worst-days" / "work"
E175: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00175-dp-unlabelled-day-shapes" / "work"
BG2: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
    / "p_s6_bg2"
    / "predictions.parquet"
)
HIER: Final[Path] = REPO_DATA_DIR / "NGED" / "substation_hierarchy_2026-10-15"
NGED: Final[Path] = REPO_DATA_DIR / "NGED"
CONFIRMED: Final[set[str]] = {"Confirmed", "Field Confirmed"}
ACTIONS: Final[tuple[str, ...]] = (
    "Telecontrol OPEN",
    "Telecontrol CLOSE",
    "Open from Earth",
    "Confirm Opened",
)
GEN_TYPES: Final[list[str]] = ["Wind", "PV", "BESS", "Biofuel", "Other (Generation)"]


def log_events() -> pl.DataFrame:
    """Return every row of every switching-log sheet: pair ids, manual time, state, action."""
    wb = openpyxl.load_workbook(HIER / "Switching Logs.xlsx", read_only=True)
    rows = []
    for ws in wb:
        m = re.match(r"(\d+)-(\w+)", ws.title)
        if not m:
            continue
        first, second = int(m.group(1)), m.group(2)
        for r in ws.iter_rows(values_only=True):
            if len(r) < 6 or not isinstance(r[3], datetime):
                continue
            rows.append(
                {
                    "sheet": ws.title,
                    "a": first,
                    "b": int(second) if second.isdigit() else None,
                    "manual_time": r[3],
                    "state": str(r[4]).strip(),
                    "action": str(r[5]).strip(),
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def s1_flags(*, days: pl.DataFrame, ids: dict[str, int], nbr: dict[str, set[str]]) -> pl.DataFrame:
    """Flag each day for S1 under the planned rule and the labelled sensitivities."""
    ev = log_events()
    ev = ev.with_columns(d=pl.col("manual_time").dt.date(), h=pl.col("manual_time").dt.hour())
    ok = ev.filter(pl.col("state").is_in(CONFIRMED), pl.col("action").is_in(ACTIONS))

    def hit(events: pl.DataFrame, series: set[int], day: date, mode: str) -> bool:
        e = events.filter(pl.col("a").is_in(series) | pl.col("b").is_in(series))
        if mode == "plan":
            return (
                e.filter(
                    (pl.col("d") == day)
                    | ((pl.col("d") == day - timedelta(days=1)) & (pl.col("h") >= 18))
                ).height
                > 0
            )
        return (
            e.filter(
                pl.col("d").is_between(day - timedelta(days=1), day + timedelta(days=1))
            ).height
            > 0
        )

    out = []
    for r in days.iter_rows(named=True):
        own = {ids[r["site"]]}
        wide = own | {ids[n] for n in nbr[r["site"]] if n in ids}
        out.append(
            {
                "s1": hit(ok, own, r["local_date"], "plan")
                or hit(ok, wide - own, r["local_date"], "plan"),
                "s1_own_sheet": hit(ok, own, r["local_date"], "plan"),
                "s1_any_state": hit(ev, own | (wide - own), r["local_date"], "plan"),
                "s1_pm1": hit(ok, wide, r["local_date"], "pm1"),
            }
        )
    cover = ev.group_by("sheet").agg(
        pl.col("d").min().alias("first"), pl.col("d").max().alias("last")
    )
    cover.write_parquet(WORK / "log_coverage_by_sheet.parquet")
    return days.select("site", "local_date").hstack(pl.DataFrame(out))


def power(series: list[int]) -> pl.DataFrame:
    """Return raw power for the series, with local half-hour start date and index."""
    lf = pl.scan_delta(str(NGED / "power_time_series.delta")).filter(
        pl.col("time_series_id").is_in(series),
        pl.col("time") >= datetime(2025, 5, 1, tzinfo=UTC),
        pl.col("time") < datetime(2026, 8, 1, tzinfo=UTC),
    )
    return (
        lf.select("time_series_id", "time", "power")
        .collect()
        .with_columns(
            start=(pl.col("time") - pl.duration(minutes=30)).dt.convert_time_zone("Europe/London")
        )
        .with_columns(local_date=pl.col("start").dt.date())
    )


def s3_flags(*, days: pl.DataFrame, ids: dict[str, int]) -> pl.DataFrame:
    """Flag missing half-hours (6 or more) and stuck runs (8 or more identical readings)."""
    p = power(list(ids.values()))
    inv = {v: k for k, v in ids.items()}
    rows = []
    for (tsid,), part in p.group_by("time_series_id"):
        g = part.sort("time")
        # identical-consecutive run length
        same = (g["power"] == g["power"].shift(1)).fill_null(False)
        run = (~same).cum_sum()
        marked = g.with_columns(run=run).with_columns(rl=pl.len().over("run"))
        per = marked.group_by("local_date").agg(n=pl.len(), max_run=pl.col("rl").max())
        per = per.with_columns(site=pl.lit(inv[tsid]))
        rows.append(per)
    per = pl.concat(rows)
    out = days.select("site", "local_date").join(per, on=["site", "local_date"], how="left")
    return (
        out.with_columns(
            s3_missing=(48 - pl.col("n").fill_null(0)) >= 6,
            s3_stuck=pl.col("max_run").fill_null(0) >= 8,
        )
        .with_columns(s3=pl.col("s3_missing") | pl.col("s3_stuck"))
        .select("site", "local_date", "s3", "s3_missing", "s3_stuck")
    )


def generator_steps(*, gen_ids: list[int]) -> pl.DataFrame:
    """Return (series, local_date) where a generator steps: day-on-day change shifts for 4 hours."""
    p = power(gen_ids)
    rows = []
    for (tsid,), part in p.group_by("time_series_id"):
        g = part.sort("time")
        x = g["power"].to_numpy().astype(float)
        t = g["start"].to_list()
        q = x[48:] - x[:-48]
        qt = t[48:]
        s = np.full(len(q), np.nan)
        for i in range(8, len(q) - 8):
            s[i] = np.nanmean(q[i : i + 8]) - np.nanmean(q[i - 8 : i])
        floor = 0.05 * np.nanpercentile(x, 99)
        dates = np.array([v.date() for v in qt])
        for d in np.unique(dates):
            prior = (dates < d) & (dates >= d - timedelta(days=28))
            sigma = np.nanstd(s[prior]) if prior.sum() > 200 else np.nan
            thr = max(3 * sigma, floor) if not np.isnan(sigma) else np.inf
            if np.nanmax(np.abs(s[dates == d]), initial=0) > thr:
                rows.append({"time_series_id": tsid, "local_date": d})
    return pl.DataFrame(rows, schema={"time_series_id": pl.Int32, "local_date": pl.Date})


def s2_flags(*, days: pl.DataFrame, ids: dict[str, int], nbr: dict[str, set[str]]) -> pl.DataFrame:
    """Flag generator steps on the same or an interconnected Primary, and anywhere in the area."""
    meta = pl.read_parquet(NGED / "metadata.parquet").select(
        "time_series_id", "time_series_type", "substation_number"
    )
    gens = meta.filter(pl.col("time_series_type").is_in(GEN_TYPES))
    steps = generator_steps(gen_ids=gens["time_series_id"].to_list())
    steps.write_parquet(WORK / "generator_steps.parquet")
    mon = pl.read_csv(HIER / "Monitor Direction.csv").select("TimeSeriesID", "SubstationNumber")
    num = {r["TimeSeriesID"]: r["SubstationNumber"] for r in mon.iter_rows(named=True)}
    gen_num = {r["time_series_id"]: r["substation_number"] for r in gens.iter_rows(named=True)}
    out = []
    for r in days.iter_rows(named=True):
        here = {num[ids[r["site"]]]} | {num[ids[n]] for n in nbr[r["site"]] if n in ids}
        near = {g for g, n in gen_num.items() if n in here}
        day_steps = set(steps.filter(pl.col("local_date") == r["local_date"])["time_series_id"])
        out.append(
            {
                "s2": bool(near & day_steps),
                "s2_has_generator": bool(near),
                "s2_any_area": bool(day_steps),
            }
        )
    return days.select("site", "local_date").hstack(pl.DataFrame(out))


def s4_flags(*, days: pl.DataFrame) -> pl.DataFrame:
    """Flag a day with two other worst-5% days on its weekday within 8 weeks of the same site."""
    worst = pl.read_parquet(E172 / "worst_days.parquet").select("site", "local_date")
    out = []
    for r in days.iter_rows(named=True):
        w = worst.filter(
            pl.col("site") == r["site"],
            pl.col("local_date") != r["local_date"],
            pl.col("local_date").dt.weekday() == r["local_date"].isoweekday(),
            (pl.col("local_date") - pl.lit(r["local_date"])).dt.total_days().abs() <= 56,
        )
        out.append({"s4": w.height >= 2})
    return days.select("site", "local_date").hstack(pl.DataFrame(out))


def main() -> None:
    """Compute the indicators and compare the 37 days with matched ordinary days."""
    WORK.mkdir(parents=True, exist_ok=True)
    days = pl.read_parquet(E172 / "days.parquet").filter(pl.col("n_rows") >= 24)
    bg2 = pl.read_parquet(BG2, columns=["site", "time_series_id"]).unique()
    ids = {r["site"]: int(r["time_series_id"]) for r in bg2.iter_rows(named=True)}
    mon = pl.read_csv(HIER / "Monitor Direction.csv").select("TimeSeriesID", "SubstationNumber")
    num = {r["TimeSeriesID"]: r["SubstationNumber"] for r in mon.iter_rows(named=True)}
    lab: dict[int, set[str]] = {}
    for s, i in ids.items():
        lab.setdefault(num[i], set()).add(s)
    nbr: dict[str, set[str]] = {s: set() for s in ids}
    for a, b in pl.read_csv(HIER / "Primary Substation Interconnections.csv").iter_rows():
        for x, y in ((a, b), (b, a)):
            for s in lab.get(x, set()):
                nbr[s] |= lab.get(y, set())

    key = ["site", "local_date"]
    flags = (
        days.select(key)
        .join(s1_flags(days=days, ids=ids, nbr=nbr), on=key)
        .join(s2_flags(days=days, ids=ids, nbr=nbr), on=key)
        .join(s3_flags(days=days, ids=ids), on=key)
        .join(s4_flags(days=days), on=key)
    )
    flags.write_parquet(WORK / "indicator_flags.parquet")

    target = pl.read_parquet(E175 / "day_shapes.parquet").select(key)
    pool = pl.read_parquet(E175 / "ordinary_day_shapes.parquet").select(key)
    srcs = ["s1", "s2", "s3", "s4"]
    sens = ["s1_own_sheet", "s1_any_state", "s1_pm1", "s2_any_area", "s3_missing", "s3_stuck"]
    flags = flags.with_columns(union=pl.any_horizontal(srcs))
    t = target.join(flags, on=key)
    pool_f = pool.join(flags, on=key)
    rng = np.random.default_rng(SEED)
    cols = [*srcs, *sens, "union"]
    draws = {c: [] for c in cols}
    fallback = 0
    for _ in range(N_DRAWS):
        picks = []
        for r in target.iter_rows(named=True):
            month = r["local_date"].month
            cand = pool_f.filter(
                pl.col("site") == r["site"], pl.col("local_date").dt.month() == month
            )
            if cand.height == 0:
                cand = pool_f.filter(pl.col("site") == r["site"])
                fallback += 1
            picks.append(cand.row(int(rng.integers(cand.height)), named=True))
        d = pl.DataFrame(picks)
        for c in cols:
            draws[c].append(int(d[c].sum()))
    report = {}
    for c in cols:
        obs = int(t[c].sum())
        arr = np.array(draws[c])
        report[c] = {
            "observed_of_37": obs,
            "ordinary_mean_of_37": float(arr.mean()),
            "ordinary_95pct_range": [
                float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)),
            ],
            "one_sided_p": float((arr >= obs).mean()),
            "all_scored_days_rate": int(flags[c].sum()) / flags.height,
        }
    report["_fallback_draw_picks"] = fallback
    report["_days_with_generator_on_same_or_interconnected_primary"] = int(
        flags["s2_has_generator"].sum()
    )
    (WORK / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(t.group_by("site").agg(pl.col(c).sum() for c in cols))


if __name__ == "__main__":
    main()
