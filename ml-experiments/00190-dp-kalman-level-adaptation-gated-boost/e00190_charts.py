"""Draw the figures of experiment 190's write-up from `work/report/report.json`.

uv run --no-sync python ml-experiments/00190-*/e00190_charts.py
"""

import json
import textwrap
from datetime import timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00190_common as common
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

E175_DAYS: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00175-dp-unlabelled-day-shapes" / "work"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
GREY: Final[str] = "#808080"
ARM_LABELS: Final[dict[str, str]] = {
    "a0": "A0 reference (arm D)",
    "a1": "A1 gated boost",
    "a2": "A2 always-on boost",
    "a3": "A3 gated, shared level",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "a0": GREY,
    "a1": ocf.BRAND_ORANGE,
    "a2": ocf.DATA_BLUE,
    "a3": ocf.DATA_PURPLE,
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _zero_rule(*, value: float, label: str) -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"x": [value], "label": [label]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )


def figure_contrasts(*, data: dict) -> alt.TopLevelMixin:
    """The planned contrasts over all rows, with and without refitting the spread."""
    names = {
        "C1 gated minus reference": "C1 A1 minus A0 (gated boost)",
        "C2 always-on minus reference": "C2 A2 minus A0 (always-on boost)",
        "C3 gated minus always-on": "C3 A1 minus A2",
        "C4 shared minus gated": "C4 A3 minus A1 (shared level)",
    }
    rows = []
    for key, label in (
        ("contrasts", "spread refitted"),
        ("fixed_spread_contrasts", "spread of experiment 169 held fixed"),
    ):
        for contrast, name in names.items():
            c = data[key][contrast]
            rows.append(
                {"contrast": name, "spread": label, "estimate": c["estimate"],
                 "low": c["lower"], "high": c["upper"]}
            )  # fmt: skip
    colours = {
        "spread refitted": ocf.BRAND_ORANGE,
        "spread of experiment 169 held fixed": ocf.DATA_BLUE,
    }
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("contrast:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("spread:N", sort=list(colours)),
        color=alt.Color(
            "spread:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top", direction="horizontal", labelLimit=240),
        ),
    )
    rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title="Difference in quantile score (points; below zero favours the first arm)",
        )
    )
    return (_zero_rule(value=0.0, label="same score") + rule + dot).properties(
        width=WIDTH - 260,
        height=220,
        title=_title(
            text="The gated boost leaves the score where it was; the shared-level boost raises it",
            subtitle=(
                "Dot: difference in mean quantile score over leads 0.5 to 168 hours, five decision "
                "substations, scored year, outside switching events. Line: 95% interval from "
                "resampling whole calendar months."
            ),
        ),
    )


def figure_recovery(*, data: dict) -> alt.TopLevelMixin:
    """The known-answer gate: share of the injected offset's excess loss that each arm removes."""
    rows = []
    for key, value in data["gate"]["recovery"].items():
        size, hours, arm = key.split("|")
        if arm not in ("a1", "a2", "a3") or float(size) not in (0.05, 0.1, 0.2):
            continue
        rows.append(
            {"size": f"offset of {float(size):.2f} of the median load", "hours": f"{hours} h",
             "arm": ARM_LABELS[arm], "recovery": value["recovery"], "low": value["lower"],
             "high": value["upper"], "order": int(hours)}
        )  # fmt: skip
    labels = [ARM_LABELS[a] for a in ("a1", "a2", "a3")]
    frame = pl.DataFrame(rows)
    panels = []
    for size in sorted(frame["size"].unique().to_list()):
        base = alt.Chart(frame.filter(pl.col("size") == size)).encode(
            y=alt.Y("hours:N", sort=["8 h", "16 h", "36 h"], title="Offset lasts"),
            yOffset=alt.YOffset("arm:N", sort=labels),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=labels, range=[ARM_COLOURS[a] for a in ("a1", "a2", "a3")]),
                legend=alt.Legend(title=None, orient="top", direction="horizontal", labelLimit=240),
            ),
        )
        rule = base.mark_rule(strokeWidth=2, clip=True).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", scale=alt.Scale(domain=[-1.0, 1.0])), x2="high:Q"
        )
        dot = base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "recovery:Q",
                scale=alt.Scale(domain=[-1.0, 1.0]),
                title="Share of the offset's excess loss removed, against A0 (higher is better)",
            )
        )
        target = (
            alt.Chart(pl.DataFrame({"x": [0.5]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color=ocf.DATA_GREEN, strokeWidth=2)
            .encode(x="x:Q")
        )
        panels.append(
            (_zero_rule(value=0.0, label="no recovery") + target + rule + dot).properties(
                width=WIDTH - 140, height=110, title=size
            )
        )
    return (
        alt.vconcat(*panels)
        .resolve_scale(color="shared")
        .properties(
            title=_title(
                text="No boost removes more than a few percent of an injected offset's excess loss",
                subtitle=(
                    "Known-answer gate: a one-signed offset added to the observed power on 40 "
                    "ordinary days per panel row (8 per substation), forecasts with "
                    "3.5-hour-old power from "
                    "origins every 6 hours. Dot: share removed against A0. Line: 95% interval from "
                    "resampling the 40 cases. Green rule: the pre-registered pass mark of 50%."
                ),
            )
        )
    )


def figure_curve(*, data: dict) -> alt.TopLevelMixin:
    """The recovery by hours since the offset started, for the 0.10 offset lasting 36 hours."""
    order = ["0-3.5 h", "3.5-6 h", "6-12 h", "12-24 h", "24-36 h"]
    rows = []
    for key, value in data["gate"]["curve"].items():
        hours, arm, label = key.split("|")
        if hours != "36":
            continue
        rows.append({"arm": ARM_LABELS[arm], "since": label, "recovery": value["recovery"]})
    labels = [ARM_LABELS[a] for a in ("a1", "a2", "a3")]
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "since:N",
                sort=order,
                title="Hours since the offset started (the target half-hour's time)",
            ),
            y=alt.Y(
                "recovery:Q",
                title="Share of excess loss removed (higher is better)",
                scale=alt.Scale(domain=[-0.4, 0.5]),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=labels, range=[ARM_COLOURS[a] for a in ("a1", "a2", "a3")]),
                legend=alt.Legend(title=None, orient="top", direction="horizontal", labelLimit=240),
            ),
        )
    )
    return chart.properties(
        width=WIDTH - 120,
        height=200,
        title=_title(
            text="The recovery stays at zero at every hour since the offset started",
            subtitle=(
                "Known-answer gate, offset of 0.10 of the median load lasting 36 hours, 40 cases. "
                "The telemetry delay of 3.5 hours means the first hours cannot be recovered."
            ),
        ),
    )


def figure_example_day() -> alt.TopLevelMixin:
    """Return the smooth-offset day with the largest error of arm D: power and each arm's mean."""
    shapes = pl.read_parquet(E175_DAYS / "day_shapes.parquet").filter(pl.col("shape") == "b_offset")
    best = None
    for site, date in shapes.select("site", "local_date").iter_rows():
        frame = _day(site=site, date=date, arm="d0")
        if frame is None:
            continue
        error = float((frame["actual"] - frame["mean"]).abs().to_numpy().mean())
        if best is None or error > best[0]:
            best = (error, site, date)
    assert best is not None
    _, site, date = best
    rows = []
    for arm in ("d0", "d1", "d2", "d3"):
        frame = _day(site=site, date=date, arm=arm)
        assert frame is not None
        rows.extend(
            {"hour": h, "series": ARM_LABELS["a" + arm[1:]], "value": v}
            for h, v in zip(frame["hour"], frame["mean"], strict=True)
        )
    reference = _day(site=site, date=date, arm="d0")
    assert reference is not None
    rows.extend(
        {"hour": h, "series": "Observed", "value": v}
        for h, v in zip(reference["hour"], reference["actual"], strict=True)
    )
    domain = [*ARM_LABELS.values(), "Observed"]
    colours = [*(ARM_COLOURS[a] for a in ARM_LABELS), "#000000"]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            x=alt.X(
                "hour:Q", title="Hour of the local day", axis=alt.Axis(values=[0, 6, 12, 18, 24])
            ),
            y=alt.Y("value:Q", title="Power (points of capacity)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=domain, range=colours),
                legend=alt.Legend(title=None, orient="top", columns=3, labelLimit=240),
            ),
        )
        .properties(
            width=WIDTH - 120,
            height=230,
            title=_title(
                text="On a smooth-offset day the four arms' forecasts differ little",
                subtitle=(
                    f"One of the 15 smooth-offset days of experiment 175 ({site}, {date}), "
                    "the one where arm D's mean error is largest. Run issued the "
                    "previous morning (lead 15 to 39 hours)."
                ),
            ),
        )
    )


def _day(*, site: str, date, arm: str) -> pl.DataFrame | None:  # noqa: ANN001
    """Return the mean and observed power of the run issued the morning before a local date."""
    frame = pl.read_parquet(common.WORK_DIR / "dp_preds" / f"{arm}_scored_{site}.parquet")
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    origin_day = date - timedelta(days=1)
    part = frame.filter(
        (pl.col("power_fcst_init_time").dt.date() == origin_day) & (local.dt.date() == date)
    ).with_columns(hour=local.dt.hour() + local.dt.minute() / 60.0)
    return part.sort("valid_time") if part.height >= 40 else None


def main() -> None:
    """Draw every figure."""
    data = json.loads((common.WORK_DIR / "report" / "report.json").read_text())
    _save(figure_contrasts(data=data), name="figure_1_contrasts")
    _save(figure_recovery(data=data), name="figure_2_recovery")
    _save(figure_curve(data=data), name="figure_3_curve")
    _save(figure_example_day(), name="figure_4_example_day")


if __name__ == "__main__":
    main()
