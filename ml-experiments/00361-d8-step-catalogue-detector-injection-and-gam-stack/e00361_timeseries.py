# ruff: noqa: E501
"""Draw experiment 361's time series figures from stored outputs and the experiment 252 cache.

Weeks are chosen by a rule fixed before any chart was drawn. Scoring year: Monday-start weeks
from 2025-07-07 to 2026-06-22. Flagged hours are the hours of detected falls and drops in the
stored D8 catalogue. Weeks 1 and 2 are the two non-overlapping weeks with the most flagged hours
(ties go to the earlier week). Week 3 is the zero-flagged-hour week nearest the middle of the year.
Injection panels use host D2, one draw per size with a fixed seed. Each draw shows the first
missed spell (none of its half-hours detected, and at least 2 days into the window so a baseline
exists) and the first false alarm (a detected segment that overlaps no injected spell).

    PYTHONPATH=$E252:$E264:$E354:$E361 uv run --no-sync python ml-experiments/00361-*/e00361_timeseries.py
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00361_detect as detect
import e00361_inject as inject
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00361-d8-step-catalogue-detector-injection-and-gam-stack"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 480
YEAR_START: Final[datetime] = datetime(2025, 7, 7)  # noqa: DTZ001
N_WEEKS: Final[int] = 51
RED: Final[str] = "#B30000"


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def _segments() -> pl.DataFrame:
    falls = pl.read_parquet(WORK / "d8_negative_segments.parquet").with_columns(
        kind=pl.lit("Fall of 6 hours or more")
    )
    drops = pl.read_parquet(WORK / "d8_drops.parquet").with_columns(kind=pl.lit("Multi-day drop"))
    return pl.concat([falls, drops])


def choose_weeks(*, segments: pl.DataFrame) -> list[tuple[datetime, str]]:
    """Apply the module docstring's rule."""
    rows = []
    for i in range(N_WEEKS):
        a = YEAR_START + timedelta(weeks=i)
        b = a + timedelta(weeks=1)
        hours = 0.0
        for s, e in zip(segments["start"], segments["end"], strict=True):
            lo, hi = max(s, a), min(e + timedelta(minutes=30), b)
            if hi > lo:
                hours += (hi - lo).total_seconds() / 3600
        rows.append((a, hours, i))
    ranked = sorted(rows, key=lambda r: (-r[1], r[0]))
    chosen: list[tuple[datetime, str]] = []
    for a, hours, _ in ranked:
        if all(abs((a - c).days) >= 7 for c, _ in chosen):
            chosen.append((a, f"{hours:.0f} flagged hours"))
        if len(chosen) == 2:
            break
    calm = min((r for r in rows if r[1] == 0), key=lambda r: abs(r[2] - N_WEEKS // 2))
    chosen.append((calm[0], "0 flagged hours"))
    print("weeks", chosen)
    return chosen


def _label(a: datetime, note: str) -> str:
    return f"Week of {a:%d %b %Y} ({note})"


def figure_4(
    *,
    weeks: list[tuple[datetime, str]],
    segments: pl.DataFrame,
    times: np.ndarray,
    power: np.ndarray,
) -> None:
    """D8 demand with detected falls and drops shaded."""
    panels = []
    for a, note in weeks:
        b = a + timedelta(weeks=1)
        m = (times > np.datetime64(a)) & (times <= np.datetime64(b))
        line = pl.DataFrame({"time": times[m], "mw": power[m]})
        seg = segments.filter((pl.col("end") >= a) & (pl.col("start") <= b)).with_columns(
            pl.col("start").clip(a, b), pl.col("end").clip(a, b)
        )
        x = alt.X(
            "time:T", title=None, scale=alt.Scale(domain=[a, b]), axis=alt.Axis(format="%a %d")
        )
        base = (
            alt.Chart(line)
            .mark_line(color=ocf.DATA_BLUE, strokeWidth=1.2)
            .encode(  # ty: ignore[unresolved-attribute]
                x=x, y=alt.Y("mw:Q", title="D8 demand (MW)", scale=alt.Scale(zero=False))
            )
        )
        layers = [base]
        if seg.height:
            band = alt.Chart(seg).mark_rect(opacity=0.25).encode(  # ty: ignore[unresolved-attribute]
                x="start:T", x2="end:T",
                color=alt.Color("kind:N", scale=alt.Scale(domain=["Fall of 6 hours or more", "Multi-day drop"], range=[ocf.BRAND_ORANGE, ocf.DATA_PURPLE]), legend=alt.Legend(title=None, orient="top")),
            )  # fmt: skip
            layers = [band, base]
        panels.append(alt.layer(*layers).properties(width=WIDTH, height=130, title=_label(a, note)))
    chart = alt.vconcat(*panels).resolve_scale(color="shared")
    _save(chart, name="figure_4_d8_demand_with_detected_steps")


def _pairs(seg: detect.Segments) -> list[tuple[int, int]]:
    return list(zip(seg.start.tolist(), seg.end.tolist(), strict=True))


def _py(t: np.datetime64) -> datetime:
    return t.astype("datetime64[us]").item()


def _host_series(*, host: str) -> tuple[np.ndarray, np.ndarray]:
    times, power = detect.load_power_mw(site=host)
    lo = int(np.searchsorted(times, np.datetime64(inject.WINDOW[0])))
    hi = int(np.searchsorted(times, np.datetime64(inject.WINDOW[1])))
    return times[lo:hi], power[lo:hi]


def figure_5(*, catalogue: pl.DataFrame) -> None:
    """Simulated spells injected into D2 with the detector's output, for 2 MW and 4 MW."""
    dwells = catalogue["dwell_days"].to_numpy()
    gaps = (catalogue["start"].to_numpy()[1:] - catalogue["end"].to_numpy()[:-1]) / np.timedelta64(
        1, "D"
    )
    gaps = gaps[gaps > 0.25]
    times, base = _host_series(host="D2")
    n = len(base)
    panels = []
    for size in (2.0, 4.0):
        rng = np.random.default_rng(inject.SEED + int(size))
        spells = inject.spells_for(rng=rng, n=n, dwells=dwells, gaps=gaps)
        injected = np.zeros(n)
        for s, e in spells:
            injected[s:e] += size
        series = base - injected
        found = _pairs(detect.detect(power_mw=series)) + _pairs(
            detect.detect_drops(power_mw=series)
        )
        flag, level = inject.union_flag(n=n, power=series)
        scored = inject.score_spells(spells=spells, truth=injected, flag=flag, level=level)
        missed = [
            sp
            for sp, r in zip(spells, scored, strict=True)
            if r["covered"] == 0 and sp[0] >= 2 * 48
        ]
        false = [seg for seg in found if not any(seg[0] < e and s < seg[1] for s, e in spells)]
        picks = []
        if missed:
            picks.append(("Missed spell", missed[0][0], missed[0][1]))
        if false:
            picks.append(("False alarm", false[0][0], false[0][1]))
        print(size, "spells", len(spells), "missed", len(missed), "false alarms", len(false), picks)
        for label, s, e in picks:
            mid = (s + e) // 2
            lo, hi = max(mid - 4 * 48, 0), min(mid + 4 * 48, n)
            line = pl.DataFrame({"time": times[lo:hi], "mw": series[lo:hi]})
            a, b = (t.astype("datetime64[us]").item() for t in (times[lo], times[hi - 1]))
            x = alt.X(
                "time:T", title=None, scale=alt.Scale(domain=[a, b]), axis=alt.Axis(format="%d %b")
            )
            layers = [
                alt.Chart(line)
                .mark_line(color=ocf.DATA_BLUE, strokeWidth=1.2)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=x, y=alt.Y("mw:Q", title="D2 + simulated (MW)", scale=alt.Scale(zero=False))
                )
            ]
            truth = [(s2, e2) for s2, e2 in spells if s2 < hi and e2 > lo]
            det = [(s2, e2) for s2, e2 in found if s2 < hi and e2 > lo]
            rects = [
                {
                    "start": _py(times[max(s2, lo)]),
                    "end": _py(times[min(e2, hi) - 1]),
                    "what": "Injected spell",
                }
                for s2, e2 in truth
            ]
            rects += [
                {
                    "start": _py(times[max(s2, lo)]),
                    "end": _py(times[min(e2, hi) - 1]),
                    "what": "Detected segment",
                }
                for s2, e2 in det
            ]
            if rects:
                frame = pl.DataFrame(rects).with_columns(
                    row=pl.when(pl.col("what") == "Injected spell")
                    .then(pl.lit(1.0))
                    .otherwise(pl.lit(0.0))
                )
                layers.insert(0, alt.Chart(frame).mark_rect(opacity=0.3).encode(x="start:T", x2="end:T", color=alt.Color("what:N", scale=alt.Scale(domain=["Injected spell", "Detected segment"], range=[ocf.BRAND_ORANGE, ocf.DATA_PURPLE]), legend=alt.Legend(title=None, orient="top"))))  # ty: ignore[unresolved-attribute]  # fmt: skip
            panels.append(
                alt.layer(*layers).properties(
                    width=WIDTH,
                    height=120,
                    title=f"{size:.0f} MW injected: {label.lower()} (from {str(times[lo])[:10]})",
                )
            )
    _save(
        alt.vconcat(*panels).resolve_scale(color="shared"),
        name="figure_5_injected_spells_with_detector",
    )


def figure_6(*, weeks: list[tuple[datetime, str]], capacity: float) -> None:
    """D8 forecasts at 12 to 36 hours ahead with and without the detected steps removed."""
    wide = pl.read_parquet(WORK / "work" / "scores_wide.parquet").filter(
        (pl.col("lead_hours") >= 12) & (pl.col("lead_hours") < 36)
    )
    cols = {
        "actual": "Actual demand",
        "ctl_arm3__q50": "Control (raw target)",
        "t1_arm3__q50": "Levels added back",
        "t0_arm3__q50": "Rows masked",
    }
    wide = (
        wide.select("valid_time", *cols)
        .with_columns(pl.col(c) * capacity / 100 for c in cols)
        .unique(subset="valid_time")
        .sort("valid_time")
    )
    wide = wide.with_columns(pl.col("valid_time").dt.replace_time_zone(None))
    grid = (
        pl.datetime_range(wide["valid_time"].min(), wide["valid_time"].max(), "30m", eager=True)  # ty: ignore[no-matching-overload]
        .alias("valid_time")
        .to_frame()
    )
    wide = grid.join(wide, on="valid_time", how="left")
    order = list(cols.values())
    panels = []
    for a, note in weeks:
        b = a + timedelta(weeks=1)
        sub = wide.filter((pl.col("valid_time") > a) & (pl.col("valid_time") <= b)).rename(cols)
        long = sub.unpivot(index="valid_time", variable_name="series", value_name="mw")
        x = alt.X(
            "valid_time:T",
            title=None,
            scale=alt.Scale(domain=[a, b]),
            axis=alt.Axis(format="%a %d"),
        )
        colour = alt.Color(
            "series:N",
            sort=order,
            scale=alt.Scale(
                domain=order, range=["#333333", ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]
            ),
            legend=alt.Legend(title=None, orient="top"),
        )
        top = (
            alt.Chart(long)
            .mark_line(strokeWidth=1.2)
            .encode(  # ty: ignore[unresolved-attribute]
                x=x, y=alt.Y("mw:Q", title="Demand (MW)", scale=alt.Scale(zero=False)), color=colour
            )
            .properties(width=WIDTH, height=130, title=_label(a, note))
        )
        diff = sub.select(
            "valid_time",
            added_back=pl.col("Levels added back") - pl.col("Control (raw target)"),
            masked=pl.col("Rows masked") - pl.col("Control (raw target)"),
        ).unpivot(index="valid_time", variable_name="series", value_name="mw")
        diff = diff.with_columns(
            series=pl.col("series").replace(
                {"added_back": "Levels added back", "masked": "Rows masked"}
            )
        )
        bottom = (
            alt.Chart(diff)
            .mark_line(strokeWidth=1.2)
            .encode(  # ty: ignore[unresolved-attribute]
                x=x, y=alt.Y("mw:Q", title="Change vs control (MW)"), color=colour
            )
            .properties(width=WIDTH, height=60)
        )
        panels.append(alt.vconcat(top, bottom).resolve_scale(color="shared"))
    _save(
        alt.vconcat(*panels).resolve_scale(color="shared"),
        name="figure_6_forecasts_with_and_without_steps",
    )


def main() -> None:
    """Draw every time series figure."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    import e00252_common as common  # ty: ignore[unresolved-import]

    segments = _segments()
    weeks = choose_weeks(segments=segments)
    times, power = detect.load_power_mw(site="D8")
    capacity = float(common.load_cache().sites["D8"].table.effective_capacity_mw)
    figure_4(weeks=weeks, segments=segments, times=times, power=power)
    wide = (
        pl.read_parquet(WORK / "work" / "scores_wide.parquet")
        .filter(pl.col("lead_hours") == 0.5)
        .select("valid_time", "actual")
        .with_columns(pl.col("valid_time").dt.replace_time_zone(None))
        .head(300)
    )
    cache = pl.DataFrame({"valid_time": times, "mw": power}).join(wide, on="valid_time")
    print(
        "alignment check (max |actual*cap/100 - cache MW|):",
        (cache["actual"] * capacity / 100 - cache["mw"]).abs().max(),
        "rows",
        cache.height,
    )
    cat = (
        pl.concat(
            [
                pl.read_parquet(WORK / "d8_negative_segments.parquet"),
                pl.read_parquet(WORK / "d8_drops.parquet"),
            ]
        )
        .unique()
        .sort("start", "end")
    )
    figure_5(catalogue=cat)
    figure_6(weeks=weeks, capacity=capacity)


if __name__ == "__main__":
    main()
