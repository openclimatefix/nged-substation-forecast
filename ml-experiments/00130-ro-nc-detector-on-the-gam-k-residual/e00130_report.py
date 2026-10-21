"""Print every number of the experiment 101 write-up from the tables `e00130_real.py` wrote.

Also writes `event_table.parquet` (the anonymised event-by-event table) for the charts.

    uv run python ml-experiments/00101-*/e00130_report.py
"""

from __future__ import annotations

import itertools
import json
import textwrap
from datetime import timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00130_common import (
    GRID_LENGTH,
    GRID_START,
    LOGGED_PATH,
    SERIES_IDS,
    WORK_DIR,
    load_panel,
    stamp_months,
)

MVA_SERIES: Final[frozenset[int]] = frozenset({3, 9, 11, 14, 15, 24, 25, 26})
"""Non-directional apparent-power series, as in experiment 41."""
ARMS: Final[tuple[str, ...]] = ("D0", "PF14", "X", "LOGIT_SAME", "RO-NC")
EXPERIMENT_91_ROWS: Final = (
    WORK_DIR.parent
    / "00091-sw2-26-calibrated-probability-of-abnormal-arrangement"
    / "work"
    / "rows"
    / "rows_REAL_0.parquet"
)
INTRO: Final[str] = (
    "Each row is one labelled real transfer, sorted by start. Substations are Primary "
    "substations, labelled P followed by the series number; no NGED generator is involved. "
    "Share is the moved MW as a share of the metered receivers' usual load (blank when no "
    "receiver is metered). Step is the largest 24-hour step in the stored residual within 2 "
    "hours of the start, in noise units (blank when the residual panel holds no readings). "
    "The last five columns say whether the detector found the event. "
    "[Back to the write-up](result.md)."
)
"""Opening paragraph of the events page."""
N_RESAMPLES: Final[int] = 20_000
SEED: Final[int] = 101
INJECTED_RECALL: Final[tuple[float, float, float]] = (0.770, 0.665, 0.857)
"""Experiment 82's RO-NC (`XL`) MW-weighted recall on injected headline-band transfers."""
SHARE_BANDS: Final[tuple[str, ...]] = ("under 25%", "25% or more", "no metered receiver")
DURATION_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("2 to 6 hours", 0.0, 6.0),
    ("6 to 24 hours", 6.0, 24.0),
    ("1 to 3 days", 24.0, 72.0),
    ("over 3 days", 72.0, 1e9),
)


def wilson(*, k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a binomial share."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def bootstrap_weighted(*, found: np.ndarray, weight: np.ndarray) -> tuple[float, float, float]:
    """MW-weighted recall and a 95% interval from resampling whole events."""
    if len(found) == 0 or weight.sum() == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(found), size=(N_RESAMPLES, len(found)))
    draws = (found[picks] * weight[picks]).sum(axis=1) / weight[picks].sum(axis=1)
    point = float((found * weight).sum() / weight.sum())
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def event_table() -> pl.DataFrame:
    """One row per labelled event with its anonymised label, attributes, and each arm's outcome."""
    events = pl.read_parquet(WORK_DIR / "events_real.parquet")
    arms = pl.read_parquet(WORK_DIR / "event_arms.parquet")
    wide = arms.pivot(
        on="arm",
        index="card_id",
        values=["found", "onset_found", "flagged_half_hours", "detection_lag_h", "onset_error_h"],
        separator="|",
    )
    peak = arms.filter(pl.col("arm") == "PF14").select("card_id", "peak_step_z")
    table = events.join(wide, on="card_id").join(peak, on="card_id").sort("onset")
    labels = {card: f"E{i + 1:02d}" for i, card in enumerate(table["card_id"])}
    donors = []
    units = []
    for row in table.iter_rows(named=True):
        series = [int(s) for s in row["series"].split(",")]
        donors.append(" + ".join(f"P{s:02d}" for s in series))
        in_mva = [s in MVA_SERIES for s in series]
        units.append("MVA only" if all(in_mva) else "MW only" if not any(in_mva) else "mixed")
    power = load_panel()["power"]
    present = [
        max(
            float(np.isfinite(power[int(s)][row["onset"] : row["end"]]).mean())
            for s in row["series"].split(",")
            if int(s) in power
        )
        for row in table.iter_rows(named=True)
    ]
    return table.with_columns(
        power_present_share=pl.Series(present),
        event=pl.col("card_id").replace_strict(labels),
        substations=pl.Series(donors),
        units=pl.Series(units),
        month=pl.Series(
            [(GRID_START + timedelta(minutes=30 * o)).strftime("%Y-%m") for o in table["onset"]]
        ),
        share_band=pl.when(pl.col("share").is_null())
        .then(pl.lit(SHARE_BANDS[2]))
        .when(pl.col("share") >= 0.25)
        .then(pl.lit(SHARE_BANDS[1]))
        .otherwise(pl.lit(SHARE_BANDS[0])),
    )


def recall_row(*, frame: pl.DataFrame, arm: str) -> str:
    """Found count with Wilson interval, and MW-weighted recall with bootstrap interval."""
    found = frame[f"found|{arm}"].to_numpy().astype(float)
    weight = frame["moved_mw_p50"].fill_null(0.0).to_numpy()
    k, n = int(found.sum()), len(found)
    lo, hi = wilson(k=k, n=n)
    point, blo, bhi = bootstrap_weighted(found=found, weight=weight)
    return (
        f"{arm:11s} found {k:2d}/{n:2d} ({k / n:.2f}; {lo:.2f} to {hi:.2f})  "
        f"MW-weighted {point:.3f} ({blo:.3f} to {bhi:.3f})"
    )


def section_recall(*, table: pl.DataFrame) -> None:
    """Recall of every arm on all events, the 35 over a day, and by band."""
    long = table.filter(pl.col("hours") > 24)
    print(f"\n== all labelled real events: {table.height}; over 1 day: {long.height}")
    print(
        f"total moved MW (p50 sum): all {table['moved_mw_p50'].sum():.1f}, over 1 day "
        f"{long['moved_mw_p50'].sum():.1f}; null moved MW: {table['moved_mw_p50'].null_count()}"
    )
    for name, frame in (("ALL 66", table), ("OVER 1 DAY (35)", long)):
        print(f"-- {name}")
        for arm in ARMS:
            print(recall_row(frame=frame, arm=arm))
    print("\n-- by share of receiving load, events over 1 day")
    for band in SHARE_BANDS:
        frame = long.filter(pl.col("share_band") == band)
        print(f"  {band}: n={frame.height}, MW {frame['moved_mw_p50'].sum():.1f}")
        for arm in ARMS:
            print("   ", recall_row(frame=frame, arm=arm)) if frame.height else None
    print("\n-- by share of receiving load, all 66")
    for band in SHARE_BANDS:
        frame = table.filter(pl.col("share_band") == band)
        print(f"  {band}: n={frame.height}")
        for arm in ("PF14", "RO-NC"):
            print("   ", recall_row(frame=frame, arm=arm)) if frame.height else None
    print("\n-- by duration, all 66")
    for name, lo, hi in DURATION_BANDS:
        frame = table.filter((pl.col("hours") > lo) & (pl.col("hours") <= hi))
        print(f"  {name}: n={frame.height}, MW {frame['moved_mw_p50'].sum():.1f}")
        for arm in ARMS:
            print("   ", recall_row(frame=frame, arm=arm)) if frame.height else None
    head = long.filter(pl.col("share_band") == SHARE_BANDS[1])
    print(f"\n-- headline band on real events (25% or more, over 1 day): n={head.height}")
    for arm in ARMS:
        print(recall_row(frame=head, arm=arm))
    print(f"injected RO-NC MW-weighted recall: {INJECTED_RECALL}")


def span_total(frame: pl.DataFrame) -> int:
    """Half-hours of the events in `frame`."""
    return int((frame["end"].to_numpy() - frame["onset"].to_numpy()).sum())


def section_timing(*, table: pl.DataFrame) -> None:
    """Onset error, detection lag and flagged half-hours."""
    print("\n== timing and coverage (events over 1 day / all 66)")
    for name, frame in (("over 1 day", table.filter(pl.col("hours") > 24)), ("all", table)):
        print(f"-- {name}")
        for arm in ARMS:
            lag = frame[f"detection_lag_h|{arm}"].drop_nulls()
            err = frame[f"onset_error_h|{arm}"].drop_nulls()
            found = frame.filter(pl.col(f"found|{arm}"))
            flag = frame[f"flagged_half_hours|{arm}"].sum()
            frame["event_half_hours"].sum() if "event_half_hours" in frame.columns else 0
            print(
                f"{arm:11s} onset-matched {len(err):2d}: median onset error "
                f"{err.median() if len(err) else float('nan'):+.1f} h, "
                f"median detection lag {lag.median() if len(lag) else float('nan'):.1f} h; "
                f"found {found.height}; flagged half-hours {flag} of "
                f"{span_total(frame)} "
                f"({flag / (span_total(frame)):.2f})"
            )
            if found.height:
                ff = found[f"flagged_half_hours|{arm}"].sum()
                tt = span_total(found)
                print(f"{'':11s} among found events: {ff} of {tt} ({ff / tt:.2f})")


def section_misses(*, table: pl.DataFrame) -> None:
    """Why RO-NC misses events over 1 day."""
    long = table.filter(pl.col("hours") > 24).with_columns(
        small_step=pl.col("peak_step_z") < 4.0,
    )
    print("\n== RO-NC misses among the 35 events over 1 day")
    print(
        long.group_by("share_band", "units", "small_step")
        .agg(
            n=pl.len(),
            found=pl.col("found|RO-NC").sum(),
            mw=pl.col("moved_mw_p50").sum(),
            median_hours=pl.col("hours").median(),
        )
        .sort("share_band", "units", "small_step")
    )
    print(
        long.group_by("small_step").agg(
            n=pl.len(), found=pl.col("found|RO-NC").sum(), onset=pl.col("onset_found|RO-NC").sum()
        )
    )
    print("moved MW quantiles, found vs missed (RO-NC), over 1 day")
    print(
        long.group_by("found|RO-NC").agg(
            n=pl.len(),
            mw_median=pl.col("moved_mw_p50").median(),
            share_median=pl.col("share").median(),
            step_median=pl.col("peak_step_z").median(),
            hours_median=pl.col("hours").median(),
        )
    )
    print("peak step (noise units) of 24-hour residual near onset, median by band, all 66")
    print(
        table.group_by("share_band").agg(
            n=pl.len(),
            step_median=pl.col("peak_step_z").median(),
            mw_median=pl.col("moved_mw_p50").median(),
        )
    )


def section_alarms() -> None:
    """Unexplained alarms at the matched thresholds."""
    print("\n== alarms over the whole history (2022-01 to 2026-06)")
    print(pl.read_parquet(WORK_DIR / "alarm_counts.parquet"))


def section_half_hours(*, table: pl.DataFrame) -> None:
    """Share of experiment 91's logged half-hours that lie inside labelled transfers."""
    rows = pl.read_parquet(EXPERIMENT_91_ROWS).filter(pl.col("y"))
    print(
        f"\n== experiment 91's logged real half-hours: {rows.height} rows, "
        f"{rows.select(pl.struct('series', 'lag_stamp').n_unique()).item()} distinct"
    )
    stamps = rows["lag_stamp"].to_numpy()
    series = rows["series"].to_numpy()

    def inside(frame: pl.DataFrame) -> int:
        hit = np.zeros(len(stamps), dtype=bool)
        for event in frame.iter_rows(named=True):
            ids = [int(s) for s in event["series"].split(",")]
            hit |= np.isin(series, ids) & (stamps >= event["onset"]) & (stamps < event["end"])
        return int(hit.sum())

    for name, frame in (
        ("all labelled visible events", table),
        ("share of receiving load 25% or more", table.filter(pl.col("share") >= 0.25)),
        (
            "share 25% or more and over 1 day",
            table.filter((pl.col("share") >= 0.25) & (pl.col("hours") > 24)),
        ),
        ("share under 25%", table.filter(pl.col("share") < 0.25)),
        ("no metered receiver", table.filter(pl.col("share").is_null())),
        ("over 1 day, any share", table.filter(pl.col("hours") > 24)),
    ):
        k = inside(frame)
        lo, hi = wilson(k=k, n=rows.height)
        print(
            f"{name}: {k} of {rows.height} ({k / rows.height:.3f}; {lo:.3f} to {hi:.3f}); "
            f"events {frame.height}"
        )


def section_missing(*, table: pl.DataFrame) -> None:
    """Events whose metered series have little telemetry, and how RO-NC did on them."""
    print("\n== telemetry present on the best-covered series of each event (over 1 day)")
    long = table.filter(pl.col("hours") > 24).with_columns(
        covered=pl.col("power_present_share") >= 0.5
    )
    print(
        long.group_by("covered").agg(
            n=pl.len(), found=pl.col("found|RO-NC").sum(), mw=pl.col("moved_mw_p50").sum()
        )
    )
    print(long.filter(~pl.col("covered")).select("event", "substations", "power_present_share"))


def section_logged_history(*, table: pl.DataFrame) -> None:
    """Over the whole history, which logged half-hours lie inside which labelled events."""
    logged = pl.read_parquet(LOGGED_PATH)
    mask = {s: np.zeros(GRID_LENGTH, dtype=bool) for s in SERIES_IDS}
    short = {s: np.zeros(GRID_LENGTH, dtype=bool) for s in SERIES_IDS}
    long_closed = {s: np.zeros(GRID_LENGTH, dtype=bool) for s in SERIES_IDS}
    for row in logged.iter_rows(named=True):
        for side in (row["side_a"], row["side_b"]):
            if side is not None and side.isdigit() and int(side) in mask:
                lo, hi = max(row["first_stamp"], 0), min(row["last_stamp_exclusive"], GRID_LENGTH)
                mask[int(side)][lo:hi] = True
                if row["hours"] < 2:
                    short[int(side)][lo:hi] = True
                if row["hours"] > 60 * 24:
                    long_closed[int(side)][lo:hi] = True

    def flag_events(frame: pl.DataFrame) -> dict[int, np.ndarray]:
        out = {s: np.zeros(GRID_LENGTH, dtype=bool) for s in SERIES_IDS}
        for event in frame.iter_rows(named=True):
            for s in (int(x) for x in event["series"].split(",")):
                if s in out:
                    out[s][event["onset"] : event["end"]] = True
        return out

    big = flag_events(table.filter(pl.col("share") >= 0.25))
    some = flag_events(table)
    total = sum(int(mask[s].sum()) for s in SERIES_IDS)
    in_big = sum(int((mask[s] & big[s]).sum()) for s in SERIES_IDS)
    in_some = sum(int((mask[s] & some[s]).sum()) for s in SERIES_IDS)
    in_other = in_some - in_big
    rest_long = sum(int((mask[s] & ~some[s] & long_closed[s]).sum()) for s in SERIES_IDS)
    rest_short = sum(
        int((mask[s] & ~some[s] & ~long_closed[s] & short[s]).sum()) for s in SERIES_IDS
    )
    rest_in = total - in_some - rest_long - rest_short
    scope = {s: mask[s] & ~long_closed[s] & ~short[s] for s in SERIES_IDS}
    n_scope = sum(int(scope[s].sum()) for s in SERIES_IDS)
    s_big = sum(int((scope[s] & big[s]).sum()) for s in SERIES_IDS)
    s_some = sum(int((scope[s] & some[s]).sum()) for s in SERIES_IDS)
    print(f"\n== whole history, 16 series: {total:,} logged half-hours (union over closures)")
    for name, k in (
        ("inside a labelled transfer of 25% or more of receiving load", in_big),
        ("inside another labelled transfer", in_other),
        ("in a closure over 60 days (outside the labelled set)", rest_long),
        ("in a closure under 2 hours (outside the labelled set)", rest_short),
        ("in a 2 h to 60 day closure with no transfer seen", rest_in),
    ):
        print(f"  {name}: {k:,} ({k / total:.3f})")
    print(
        f"  within closures of 2 hours to 60 days only: {n_scope:,} logged half-hours; "
        f"{s_big:,} ({s_big / n_scope:.3f}) inside a transfer of 25% or more; "
        f"{s_some:,} ({s_some / n_scope:.3f}) inside any labelled transfer"
    )


STEP_EDGES: Final[tuple[float, ...]] = (0.0, 2.0, 4.0, 6.0, 10.0, float("inf"))
"""Donor-step bands, in noise units, of experiment 82's recall-by-step table."""
EXPERIMENT_82_DIR: Final = (
    WORK_DIR.parent / "00082-sw2-18-19-combined-refit-offset-and-same-sign-count"
)


def injected_recall_by_cell(*, base_only: bool = False) -> dict[tuple[bool, int], float]:
    """RO-NC recall on experiment 82's injected transfers, by metered receiver and step band.

    Even months, shapes S2 to S4, headline-band and long transfers together. The key is
    `(has metered receiver, band index)`.
    """
    events = pl.read_parquet(EXPERIMENT_82_DIR / "events.parquet")
    results = pl.read_parquet(EXPERIMENT_82_DIR / "event_results.parquet").filter(
        pl.col("setting") == "XL", pl.col("shape").is_in(["S2", "S3", "S4"])
    )
    frame = results.join(events, on="event_id")
    if base_only:
        frame = frame.filter(pl.col("population") == "base")
    frame = frame.filter(pl.Series(stamp_months()[frame["onset"].to_numpy()] % 2 == 0))
    out = {}
    for flag in (True, False):
        for i, (lo, hi) in enumerate(itertools.pairwise(STEP_EDGES)):
            cell = frame.filter(
                (pl.col("metered_receivers") > 0) == flag,
                pl.col("donor_step_z").abs().is_between(lo, hi, "left"),
            )
            out[flag, i] = float(cell["found"].to_numpy().mean()) if cell.height else float("nan")
    return out


def section_expected(*, table: pl.DataFrame) -> None:
    """How many real events RO-NC would find if each behaved like an injected one of its step."""
    cells = injected_recall_by_cell()
    base_cells = injected_recall_by_cell(base_only=True)
    print("\n== injected RO-NC recall by (metered receiver, step band), experiment 82 data")
    for (flag, i), value in cells.items():
        print(f"  metered={flag} step [{STEP_EDGES[i]:g}, {STEP_EDGES[i + 1]:g}): {value:.3f}")

    def probabilities(frame: pl.DataFrame, table: dict | None = None) -> list[float]:
        out = []
        for row in frame.iter_rows(named=True):
            step = row["peak_step_z"]
            if step is None or not np.isfinite(step):
                out.append(0.0)
                continue
            i = max(j for j in range(5) if abs(step) >= STEP_EDGES[j])
            out.append((table or cells)[row["share"] is not None, i])
        return out

    def expected(frame: pl.DataFrame) -> float:
        return float(sum(p for p in probabilities(frame) if np.isfinite(p)))

    long = table.filter(pl.col("hours") > 24)
    for name, frame in (
        ("35 events over 1 day", long),
        ("14 events 25% or more, over 1 day", long.filter(pl.col("share") >= 0.25)),
        ("11 events under 25%, over 1 day", long.filter(pl.col("share") < 0.25)),
        ("10 events with no metered receiver, over 1 day", long.filter(pl.col("share").is_null())),
    ):
        print(
            f"{name}: expected found {expected(frame):.1f}, "
            f"actual {int(frame['found|RO-NC'].sum())} of {frame.height}"
        )
    base_probs = [p for p in probabilities(long, base_cells) if np.isfinite(p)]
    print(
        "base-only injected recall cells (metered [2,4), unmetered [2,4)): "
        f"{base_cells[True, 1]:.3f}, {base_cells[False, 1]:.3f}; "
        f"expected found over the 35 from base-only cells: {sum(base_probs):.1f}"
    )
    print("found by PF14:", long.filter(pl.col("found|PF14"))["event"].to_list())
    print("found by RO-NC:", long.filter(pl.col("found|RO-NC"))["event"].to_list())
    print(
        "onset found, RO-NC, missed events with step 4 or more:",
        long.filter(
            pl.col("peak_step_z").is_finite()
            & (pl.col("peak_step_z") >= 4)
            & ~pl.col("found|RO-NC")
        )
        .select("event", "onset_found|RO-NC")
        .rows(),
    )
    probs = np.array([p for p in probabilities(long) if np.isfinite(p)])
    print(
        "standard error of the expected count over the 35: "
        f"{np.sqrt((probs * (1 - probs)).sum()):.2f}"
    )
    covered = long.filter(pl.col("peak_step_z").is_finite() & (pl.col("peak_step_z") >= 4))
    print(
        f"{covered.height} events with data and a step of 4 or more: expected found "
        f"{expected(covered):.1f}, actual {int(covered['found|RO-NC'].sum())}"
    )
    small = long.filter(pl.col("peak_step_z").is_finite() & (pl.col("peak_step_z") < 4))
    print(
        f"{small.height} events with data and a step under 4: expected found "
        f"{expected(small):.1f}, actual {int(small['found|RO-NC'].sum())}"
    )
    print("steps of the 14 events (noise units), sorted:")
    head = long.filter(pl.col("share") >= 0.25).sort("peak_step_z")
    print(
        head.select("event", "peak_step_z", "found|RO-NC", "found|X", "onset_found|RO-NC", "units")
    )
    print(
        "real events with step 4 or more:",
        int((long["peak_step_z"].is_finite() & (long["peak_step_z"] >= 4)).sum()),
        "of",
        long.height,
    )


def section_miss_classes(*, table: pl.DataFrame) -> None:
    """Sort the 35 events over 1 day by why RO-NC could or could not find them."""
    long = table.filter(pl.col("hours") > 24).with_columns(
        cause=pl.when(pl.col("power_present_share") < 0.5)
        .then(pl.lit("no telemetry on any series of the event"))
        .when(pl.col("peak_step_z") < 4.0)
        .then(pl.lit("step under 4 noise units"))
        .otherwise(pl.lit("step of 4 or more noise units"))
    )
    print("\n== the 35 events over 1 day by cause class (RO-NC)")
    print(
        long.group_by("cause").agg(
            n=pl.len(),
            found=pl.col("found|RO-NC").sum(),
            onset_found=pl.col("onset_found|RO-NC").sum(),
            found_x=pl.col("found|X").sum(),
            mw=pl.col("moved_mw_p50").sum(),
            no_metered_receiver=pl.col("share").is_null().sum(),
        )
    )
    missed = long.filter(~pl.col("found|RO-NC"), pl.col("cause") == "step of 4 or more noise units")
    print("step 4 or more, missed:", missed["event"].to_list())
    print(
        "MW found by RO-NC and by PF14 of the 42.7 MW:",
        long.filter(pl.col("found|RO-NC"))["moved_mw_p50"].sum(),
        long.filter(pl.col("found|PF14"))["moved_mw_p50"].sum(),
    )


def write_events_page(*, table: pl.DataFrame) -> None:
    """Write `events.md`, the event-by-event table, beside the write-up."""
    docs = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / WORK_DIR.name
    mark = {True: "yes", False: "-"}
    lines = [
        "# Experiment 101: the 66 labelled real transfers, event by event",
        "",
        *textwrap.wrap(INTRO, 98),
        "",
        (
            "| Event | Month | Substations | Series type | Hours | Share | MW | Step | D0 | PF14 "
            "| X "
            "| LOGIT_SAME | RO-NC |"
        ),
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in table.iter_rows(named=True):
        share = "" if row["share"] is None else f"{row['share']:.2f}"
        step = row["peak_step_z"]
        step_text = "" if step is None or not np.isfinite(step) else f"{step:.1f}"
        flags = " | ".join(mark[row[f"found|{arm}"]] for arm in ARMS)
        lines.append(
            f"| {row['event']} | {row['month']} | {row['substations']} | {row['units']} | "
            f"{row['hours']:.1f} | {share} | {row['moved_mw_p50']:.1f} | {step_text} | {flags} |"
        )
    (docs / "events.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Print all numbers and write the event table."""
    table = event_table()
    table.write_parquet(WORK_DIR / "event_table.parquet")
    print(table.columns)
    section_recall(table=table)
    section_timing(table=table)
    section_misses(table=table)
    section_alarms()
    section_half_hours(table=table)
    section_logged_history(table=table)
    section_missing(table=table)
    section_expected(table=table)
    section_miss_classes(table=table)
    write_events_page(table=table)
    print(json.dumps({"series": list(SERIES_IDS)}))


if __name__ == "__main__":
    main()
