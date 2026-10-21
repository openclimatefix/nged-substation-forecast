"""Draw the figures of experiment 158's write-up from `work/report/report.json`.

uv run python ml-experiments/00158-*/e00158_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00158_common as common
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
PERIOD_COLOURS: Final[dict[str, str]] = {
    "even months (replicate)": ocf.DATA_BLUE,
    "scored year": ocf.BRAND_ORANGE,
}
LABELS: Final[dict[str, str]] = {
    "full_k0": "24 months, per site (K0)",
    "full_ps300_3000": "24 months, partly pooled (PS)",
    "full_tied": "24 months, completely pooled",
    "3m_k0": "3 months, per site (K0-3m)",
    "3m_ps300_3000": "3 months, partly pooled (PS-3m)",
    "3m_tied": "3 months, completely pooled",
}
COLOURS: Final[dict[str, str]] = {
    "full_k0": ocf.DATA_SKY,
    "full_ps300_3000": ocf.BRAND_ORANGE,
    "full_tied": GREY,
    "3m_k0": ocf.DATA_BLUE,
    "3m_ps300_3000": ocf.DATA_PURPLE,
    "3m_tied": "#BBBBBB",
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


def _forest(*, data: dict, names: dict[str, str], axis_values: list[float], title: alt.TitleParams):  # noqa: ANN202
    rows = []
    for period, label in (("replicate", "even months (replicate)"), ("scored", "scored year")):
        for key, name in names.items():
            c = data[period]["contrasts"][key]["all"]["qs"]
            rows.append(
                {
                    "contrast": name,
                    "period": label,
                    "estimate": c["estimate"],
                    "low": c["lower"],
                    "high": c["upper"],
                }
            )
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("contrast:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=320)),
        yOffset=alt.YOffset("period:N", sort=list(PERIOD_COLOURS)),
        color=alt.Color(
            "period:N",
            scale=alt.Scale(domain=list(PERIOD_COLOURS), range=list(PERIOD_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top", direction="horizontal"),
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title=(
                "Difference in quantile score (points of capacity; "
                "below zero favours the first arm)"
            ),
            axis=alt.Axis(values=axis_values),
        )
    )
    return (zero + rule + dot).properties(width=WIDTH - 280, height=170, title=title)


def figure_short_history(*, data: dict) -> alt.TopLevelMixin:
    """Contrasts for the 3-month history."""
    names = {
        "C4": "C4 PS-3m minus K0-3m",
        "C5": "C5 PH-3m minus K0-3m (heating only)",
        "C6": "C6 PS-3m minus K0 (24 months)",
    }
    return _forest(
        data=data,
        names=names,
        axis_values=[-3.0, -2.0, -1.0, 0.0, 1.0],
        title=_title(
            text="With 3 months of history, pooling every shared shape cuts the error; pooling "
            "only the heating curve does not",
            subtitle=(
                "Dot: difference in mean quantile score over leads 0.5 to 168 hours, five decision "
                "substations, outside switching events. Line: 99% interval from resampling whole "
                "calendar months."
            ),
        ),
    )


def figure_full_history(*, data: dict) -> alt.TopLevelMixin:
    """Contrasts for the 24-month history."""
    names = {
        "C1": "C1 PH minus K0 (heating pooled)",
        "C2": "C2 PS minus K0 (all shared shapes pooled)",
        "C3": "C3 PS minus PH",
        "C8": "C8 PS minus K0, ordinary rows",
        "C7": "C7 PS minus K0, holiday and Christmas rows",
    }
    return _forest(
        data=data,
        names=names,
        axis_values=[-0.05, -0.02, 0.0, 0.02, 0.05],
        title=_title(
            text="With 24 months of history, pooling changes the score by less than 0.02 points",
            subtitle=(
                "Same scoring as the first figure. The threshold for porting is a gain of 0.02 "
                "points."
            ),
        ),
    )


def figure_breakouts(*, data: dict) -> alt.TopLevelMixin:
    """Absolute scores by row subset, scored year."""
    rows = []
    for subset in ("all", "ordinary", "holiday", "christmas", "cold nights"):
        entry = data["scored"]["tables"]["breakouts"][subset]
        rows.extend(
            {
                "subset": f"{subset} ({entry['rows']:,} rows)",
                "arm": LABELS[item["arm"]],
                "score": item["qs"],
            }
            for item in entry["scores"]
            if item["arm"] in LABELS
        )
    domain = list(LABELS.values())
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", sort=domain, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("score:Q", title="Mean quantile score (points of capacity; lower is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=domain, range=[COLOURS[k] for k in LABELS]),
                legend=None,
            ),
            row=alt.Row("subset:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        )
        .properties(
            width=WIDTH - 260,
            height=90,
            title=_title(
                text="Pooling matters on Christmas and holidays when history is short",
                subtitle=(
                    "Scored year, all leads, five decision substations. Christmas rows are those "
                    "within the Christmas tent's window of Christmas Day."
                ),
            ),
        )
    )


def figure_seasons(*, data: dict) -> alt.TopLevelMixin:
    """The 3-month history's score by season, per site and partly pooled."""
    rows = []
    for season in ("winter (DJF)", "spring (MAM)", "summer (JJA)", "autumn (SON)"):
        entry = data["scored"]["tables"]["breakouts"][season]
        rows.extend(
            {"season": season, "arm": LABELS[item["arm"]], "score": item["qs"]}
            for item in entry["scores"]
            if item["arm"] in ("3m_k0", "3m_ps300_3000", "full_k0")
        )
    domain = [LABELS[k] for k in ("3m_k0", "3m_ps300_3000", "full_k0")]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", sort=domain, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("score:Q", title="Mean quantile score (points of capacity; lower is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=domain,
                    range=[COLOURS[k] for k in ("3m_k0", "3m_ps300_3000", "full_k0")],
                ),
                legend=None,
            ),
            row=alt.Row("season:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        )
        .properties(
            width=WIDTH - 260,
            height=60,
            title=_title(
                text="The 3-month gain sits in winter, where the heating curve matters",
                subtitle=(
                    "Scored year, all leads, five decision substations, outside switching events."
                ),
            ),
        )
    )


def main() -> None:
    """Draw every figure."""
    data = json.loads((common.WORK_DIR / "report" / "report.json").read_text())
    _save(figure_short_history(data=data), name="figure_1_short_history")
    _save(figure_full_history(data=data), name="figure_2_full_history")
    _save(figure_breakouts(data=data), name="figure_3_breakouts")
    _save(figure_seasons(data=data), name="figure_4_seasons")


if __name__ == "__main__":
    main()
