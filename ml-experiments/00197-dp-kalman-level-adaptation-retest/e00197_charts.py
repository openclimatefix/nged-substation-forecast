"""Draw the figures of experiment 197's write-up from `work/report/report.json`.

uv run --no-sync python ml-experiments/00197-*/e00197_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00197_common as common
import plotting.ocf_theme as ocf
import polars as pl

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
BINS: Final[tuple[str, ...]] = ("0-3.5 h", "3.5-6 h", "6-12 h", "12-24 h", "24-36 h")
CLIP: Final[tuple[float, float]] = (-1.5, 1.5)


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


def _rule(*, value: float) -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"x": [value]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )


def figure_gate(*, gate: dict) -> alt.TopLevelMixin:
    """Recovery on the pairs each arm was pre-registered for, at the 0.10 size."""
    chosen = gate["chosen"]
    arms = {
        "a1": f"A1 selected ({chosen['a1']})",
        "a1_e190": "A1 at experiment 190's setting",
        "a3": f"A3 selected ({chosen['a3']})",
        "a3_e190": "A3 at experiment 190's setting",
    }
    classes = {
        "a1": "band_read",
        "a1_e190": "band_read",
        "a3": "other_read",
        "a3_e190": "other_read",
    }
    rows = []
    for key, label in arms.items():
        for hours in (16, 36):
            cell = gate["test"][f"{key}|{hours}|{classes[key]}"]
            rows.append(
                {
                    "arm": label,
                    "offset": f"{hours}-hour offset",
                    "estimate": cell["recovery"],
                    "low": max(cell["lower"], CLIP[0]),
                    "high": min(cell["upper"], CLIP[1]),
                }
            )
    colours = {"16-hour offset": ocf.BRAND_ORANGE, "36-hour offset": ocf.DATA_BLUE}
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", sort=list(arms.values()), title=None, axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("offset:N", sort=list(colours)),
        color=alt.Color(
            "offset:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top", direction="horizontal"),
        ),
    )
    rule = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", scale=alt.Scale(domain=list(CLIP))), x2="high:Q"
    )
    dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title="Share of the injected offset's extra loss recovered (pass mark 0.5)",
        )
    )
    return (_rule(value=0.0) + _rule(value=0.5) + rule + dot).properties(
        width=WIDTH - 260,
        height=220,
        title=_title(
            text="Neither faster filter reaches the pass mark on the pairs it can know about",
            subtitle=(
                "Dot: recovery at the 0.10 size, pooled over 40 injected days. Line: 95% interval "
                "from resampling days, clipped at -1.5 and 1.5. A1 is scored on pairs whose target "
                "band was read since the offset began, A3 on "
                "pairs whose band was not but another was."
            ),
        ),
    )


def figure_limit(*, gate: dict) -> alt.TopLevelMixin:
    """What the filter could know, by hours since the offset began (36-hour offset)."""
    rows = []
    labels = {
        "band_read": "target band read since onset",
        "other_read": "only another band read",
        "unread": "no band read since onset",
    }
    for bin_label in BINS:
        shares = gate["read_class_shares"][f"36|{bin_label}"]
        for key, label in labels.items():
            rows.append({"bin": bin_label, "class": label, "share": shares[key]})
    colours = {
        labels["band_read"]: ocf.DATA_GREEN,
        labels["other_read"]: ocf.DATA_SKY,
        labels["unread"]: GREY,
    }
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "bin:N",
                sort=list(BINS),
                title="Hours since the offset began",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("share:Q", title="Share of scored pairs", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "class:N",
                sort=list(colours),
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=400),
            ),
            order=alt.Order("class:N", sort="ascending"),
        )
        .properties(
            width=WIDTH - 80,
            height=220,
            title=_title(
                text="In the first 6 hours after an offset begins, no filter can know about it",
                subtitle=(
                    "Pairs of (origin, target half-hour) of the 36-hour offsets, 5 substations, 40 "
                    "days. The telemetry delay is 3.5 hours and origins come every 6 hours."
                ),
            ),
        )
    )


def figure_curve(*, gate: dict) -> alt.TopLevelMixin:
    """Recovery by hours since onset for the selected arms (0.10 size, 36-hour offset)."""
    chosen = gate["chosen"]
    pairs = (
        ("a1", "band_read", f"A1 selected, target band read ({chosen['a1']})"),
        ("a3", "other_read", f"A3 selected, only another band read ({chosen['a3']})"),
    )
    rows = []
    for key, read_class, label in pairs:
        for bin_label in BINS:
            cell = gate["curve"][f"{key}|36|{read_class}|{bin_label}"]
            if cell.get("recovery") is None:
                continue
            rows.append(
                {
                    "bin": bin_label,
                    "series": label,
                    "recovery": max(min(cell["recovery"], 1.5), -1.5),
                }
            )
    colours = {pairs[0][2]: ocf.BRAND_ORANGE, pairs[1][2]: ocf.DATA_PURPLE}
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "bin:N",
                sort=list(BINS),
                title="Hours since the offset began",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "recovery:Q",
                title="Share of extra loss recovered",
                scale=alt.Scale(domain=[-1.5, 1.5]),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top", direction="vertical"),
            ),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [0.5]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(y="y:Q")
    )  # ty: ignore[unresolved-attribute]
    return (rule + chart).properties(
        width=WIDTH - 80,
        height=220,
        title=_title(
            text="Recovery stays below the pass mark at every delay since onset",
            subtitle="0.10 size, 36-hour offset, 40 days. Pairs with no reading in the class are omitted.",  # noqa: E501
        ),
    )


def figure_tradeoff(*, gate: dict) -> alt.TopLevelMixin:
    """Recovery against left-alone harm over the whole grid (0.10 size, 36-hour offset)."""
    rows = []
    for key, harm in gate["harm"].items():
        if not (key.startswith(("a1_m", "a3_m"))):
            continue
        prefix = key[:2]
        read_class = "band_read" if prefix == "a1" else "other_read"
        cell = gate["grid"][f"{key}|36|{read_class}"]
        if cell.get("recovery") is None:
            continue
        rows.append(
            {
                "arm": "A1 (target band read)" if prefix == "a1" else "A3 (only another band read)",
                "recovery": cell["recovery"],
                "harm": harm["difference"],
                "setting": key[3:],
            }
        )
    colours = {
        "A1 (target band read)": ocf.BRAND_ORANGE,
        "A3 (only another band read)": ocf.DATA_PURPLE,
    }
    frame = pl.DataFrame(rows)
    points = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X(
                "harm:Q",
                title="Quantile score on left-alone days minus A0 (points)",
                scale=alt.Scale(type="symlog"),
            ),
            y=alt.Y(
                "recovery:Q", title="Share of extra loss recovered", scale=alt.Scale(domain=[-1, 1])
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top", direction="horizontal", labelLimit=300),
            ),
            tooltip=["setting", "recovery", "harm"],
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [0.5]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(y="y:Q")
    )  # ty: ignore[unresolved-attribute]
    return (rule + points).properties(
        width=WIDTH - 80,
        height=240,
        title=_title(
            text="Every setting that recovers more of the offset also scores worse on quiet days",
            subtitle=(
                "One dot per grid setting (m 3 to 50, k 2 to 6), 0.10 size, 36-hour offset. "
                "Harm: 40 "
                "days with no injected offset."
            ),
        ),
    )


def figure_real(*, real: dict) -> alt.TopLevelMixin:
    """A1 and A3 minus A0 on the scored year's real rows (experiment 190's settings)."""
    rows = []
    for key, label in (
        ("A1 minus A0", "A1 minus A0 (gated boost)"),
        ("A3 minus A0", "A3 minus A0 (shared level)"),
    ):
        cell = real[key]
        rows.append(
            {
                "contrast": label,
                "estimate": cell["estimate"],
                "low": cell["lower"],
                "high": cell["upper"],
            }
        )
    base = alt.Chart(pl.DataFrame(rows)).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("contrast:N", title=None, axis=alt.Axis(labelLimit=300))
    )
    rule = base.mark_rule(strokeWidth=2, color=ocf.BRAND_ORANGE).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=60, color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title="Difference in quantile score (points; below zero favours the first arm)",
        )
    )
    return (_rule(value=0.0) + rule + dot).properties(
        width=WIDTH - 260,
        height=110,
        title=_title(
            text="On real rows the shared level raises the score by 0.08 points",
            subtitle=(
                "Mean quantile score over leads 0.5 to 168 hours, five decision substations, "
                "scored year. Line: 95% interval from resampling calendar months. "
                "Settings: m 5, k 8, z 0.5."
            ),
        ),
    )


def main() -> None:
    """Draw every figure."""
    report = json.loads((common.WORK_DIR / "report" / "report.json").read_text())
    gate, real = report["gate"], report["real_rows"]
    _save(figure_gate(gate=gate), name="figure_1_gate")
    _save(figure_limit(gate=gate), name="figure_2_limit")
    _save(figure_curve(gate=gate), name="figure_3_curve")
    _save(figure_tradeoff(gate=gate), name="figure_4_tradeoff")
    _save(figure_real(real=real), name="figure_5_real")


if __name__ == "__main__":
    main()
