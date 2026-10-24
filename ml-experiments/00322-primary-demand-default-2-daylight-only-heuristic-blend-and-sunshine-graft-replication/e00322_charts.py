# ruff: noqa: E501
"""Draw the figures of experiment 322 from `work/report/*.json`.

PYTHONPATH=<folders of experiments 320, 322> uv run --no-sync python ml-experiments/00322-*/e00322_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00322_common as common
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
BANDS: Final[tuple[str, ...]] = tuple(f"{a}-{a + 24} h" for a in range(0, 240, 24))
CENTRES: Final[dict[str, int]] = {b: i * 24 + 12 for i, b in enumerate(BANDS)}
NAMES: Final[dict[str, str]] = {
    "default": "Default (Primary demand research default 2)",
    "r319": "Blend of experiment 319 at all hours (reference)",
    "d1": "Default plus D1 (blend by day only)",
    "d3": "Default plus D3 (D1, and a member-share night weight)",
    "d3b": "Default plus D3b (post hoc: blend off on forecast cold nights)",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
COLOURS: Final[dict[str, str]] = {
    NAMES["default"]: ocf.DATA_BLUE,
    NAMES["r319"]: "#7A7A7A",
    NAMES["d1"]: ocf.DATA_GREEN,
    NAMES["d3"]: ocf.DATA_PURPLE,
    NAMES["d3b"]: ocf.DATA_SKY,
    NAMES["manual_heuristic"]: ocf.BRAND_ORANGE,
    NAMES["climatology"]: "#B0B0B0",
}


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def _title(*, title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(title, 85),
        subtitle=textwrap.wrap(subtitle, 110),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: Any, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _colour(*, keys: tuple[str, ...] | None = None) -> Any:
    names = [NAMES[k] for k in keys] if keys else list(COLOURS)
    return alt.Color(
        "arm:N",
        scale=alt.Scale(domain=names, range=[COLOURS[n] for n in names]),
        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=420, symbolType="stroke"),
    )


def headline(*, report: dict) -> None:
    """Quantile score against lead for every forecast, with 95% intervals."""
    keys = ("default", "r319", "d1", "d3", "manual_heuristic", "climatology")
    data = pl.DataFrame(
        [
            {
                "arm": NAMES[k],
                "lead": CENTRES[b],
                **{f: report["by_band"][b][k][f] for f in ("est", "lo", "hi")},
            }
            for b in BANDS
            for k in keys
        ]
    )
    yt = "Mean quantile score (points of capacity; smaller is better)"
    x = alt.X(
        "lead:Q",
        title="Lead time (hours; each point is a 24-hour band, plotted at its centre)",
        scale=alt.Scale(domain=[0, 240]),
        axis=alt.Axis(values=[0, 48, 96, 144, 192, 240]),
    )
    base = _chart(data)
    chart = (
        base.mark_errorbar(thickness=1.5).encode(
            x=x,
            y=alt.Y("lo:Q", title=yt, scale=alt.Scale(zero=False)),
            y2="hi:Q",
            color=_colour(keys=keys),
        )
        + base.mark_line(strokeWidth=2).encode(
            x=x, y=alt.Y("est:Q", title=yt), color=_colour(keys=keys)
        )
    ).properties(
        width=WIDTH,
        height=300,
        title=_title(
            title="Blending by day only keeps about half of the all-hours blend's gain at 168 to 240 hours",
            subtitle="Line: mean over six Primary substations, scored year 2025-07-02 to 2026-06-30. Bar: 95% interval from resampling the 12 months of forecast date. Below 168 hours every arm equals the default.",
        ),
    )
    _save(chart, name="headline_score_by_lead")


def gain_by_group(*, report: dict) -> None:
    """Each arm's score minus the default's by group, at 168 to 240 hours, with 99.5% intervals."""
    keys = ("r319", "d1", "d3", "d3b")
    rows = []
    for g, windows in report["by_group"].items():
        e = windows.get("168-240 h")
        if e is None:
            continue
        for k in keys:
            c = e[f"{k}_vs_default"]
            rows.append(
                {
                    "group": g,
                    "arm": NAMES[k],
                    "diff": c["diff"],
                    "lo": c["diff_lo_995"],
                    "hi": c["diff_hi_995"],
                    "n": e["n"],
                }
            )
    data = pl.DataFrame(rows).with_columns(
        label=pl.col("group") + " (" + pl.col("n").cast(pl.String) + ")"
    )
    order = data["label"].unique(maintain_order=True).to_list()
    yt = "Score minus the default's (points of capacity; negative is better)"
    base = _chart(data)
    y = alt.Y("label:N", title=None, sort=order)
    chart = (
        base.mark_rule(thickness=2).encode(
            y=y, x=alt.X("lo:Q", title=yt), x2="hi:Q", color=_colour(keys=keys), yOffset="arm:N"
        )
        + base.mark_point(filled=True, size=50).encode(
            y=y, x="diff:Q", color=_colour(keys=keys), yOffset="arm:N"
        )
    ).properties(
        width=WIDTH,
        height=420,
        title=_title(
            title="Blending by day only leaves cold winter nights untouched and gives up the gain at night",
            subtitle="168 to 240 hours; the number of half-hours is in brackets. Bar: 99.5% interval from resampling forecast months.",
        ),
    )
    _save(chart, name="gain_by_group")


def recall(*, report: dict) -> None:
    """Share of realised cold winter nights flagged, and false-flag rate, by rule and lead band."""
    rules = {
        "mean_below_2c": "Ensemble mean below 2 C",
        "share_at_least_0.3": "At least 30% of members below 3 C",
        "share_at_least_0.5": "At least 50% of members below 3 C",
        "share_at_least_0.7": "At least 70% of members below 3 C",
    }
    rows = []
    for band in ("168-192 h", "192-216 h", "216-240 h"):
        e = report["recall"][band]
        for key, label in rules.items():
            rows.append(
                {
                    "band": band,
                    "rule": label,
                    "measure": "Share of cold winter nights flagged",
                    "value": e[key]["recall"],
                    "lo": e[key]["recall_lo"],
                    "hi": e[key]["recall_hi"],
                }
            )
            rows.append(
                {
                    "band": band,
                    "rule": label,
                    "measure": "Share of other winter nights flagged",
                    "value": e[key]["false_flag_rate"],
                    "lo": e[key]["false_flag_rate"],
                    "hi": e[key]["false_flag_rate"],
                }
            )
    data = pl.DataFrame(rows)
    base = _chart(data)
    x = alt.X("band:N", title="Lead band", sort=None)
    colour = alt.Color(
        "rule:N",
        sort=list(rules.values()),
        legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=300),
        scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_SKY, ocf.DATA_GREEN, ocf.DATA_PURPLE]),
    )
    chart = (
        (
            base.mark_errorbar(thickness=1.5).encode(
                x=x,
                y=alt.Y("lo:Q", title="Share flagged"),
                y2="hi:Q",
                color=colour,
                xOffset="rule:N",
            )
            + base.mark_point(filled=True, size=50).encode(
                x=x, y=alt.Y("value:Q"), color=colour, xOffset="rule:N"
            )
        )
        .properties(width=280, height=240)
        .facet(
            column=alt.Column(
                "measure:N",
                title=None,
                sort=[
                    "Share of cold winter nights flagged",
                    "Share of other winter nights flagged",
                ],
            )
        )
        .properties(
            title=_title(
                title="At 7 to 10 days the member share finds more cold nights than the ensemble mean, at a price in false flags",
                subtitle="Scored year, six Primaries. Bar on the left panel: 95% interval from resampling origin days.",
            )
        )
    )
    _save(chart, name="recall_by_rule")


def half_hour(*, report: dict) -> None:
    """Score change of each blend against the default by UTC half-hour of the day, 168 to 240 hours."""
    keys = {"r319": "all hours", "d1": "by day only"}
    rows = [
        {"hour": h / 2, "arm": NAMES[k], "diff": v}
        for k in keys
        for h, v in enumerate(report["gain_by_half_hour"][k])
    ]
    data = pl.DataFrame(rows)
    chart = (
        _chart(data)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "hour:Q", title="Hour of the day (UTC, valid time)", scale=alt.Scale(domain=[0, 24])
            ),
            y=alt.Y("diff:Q", title="Score minus the default's (negative is better)"),
            color=_colour(keys=tuple(keys)),
        )
        .properties(
            width=WIDTH,
            height=280,
            title=_title(
                title="The blend of experiment 319 gains at night as well as by day",
                subtitle="168 to 240 hours, scored year, six Primaries.",
            ),
        )
    )
    _save(chart, name="gain_by_half_hour")


def graft(*, stage3: dict) -> None:
    """Share of the refit's advantage recovered by each graft, by Primary, in the replicate and scored years."""
    families = {"solar": "Sunshine", "lags": "Recent power", "calendar": "Calendar"}
    rows = []
    for section, label in (
        ("replicate_year_2025", "April and June 2025"),
        ("scored_year_bridge_same_selection", "April to June 2026 (same selection)"),
    ):
        for subset, e in stage3[section].items():
            for key, name in families.items():
                s = e["S_level_matched"][key]
                rows.append(
                    {
                        "year": label,
                        "subset": subset.replace(" worst quarter", ""),
                        "graft": name,
                        "S": s["est"],
                        "lo": s["lo"],
                        "hi": s["hi"],
                    }
                )
    data = pl.DataFrame(rows)
    base = _chart(data)
    y = alt.Y("subset:N", title=None, sort=["D5+D9", "D5", "D9", "D10", "D16"])
    colour = alt.Color(
        "graft:N",
        sort=list(families.values()),
        legend=alt.Legend(title=None, orient="top"),
        scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]),
    )
    chart = (
        (
            base.mark_rule(thickness=2).encode(
                y=y,
                x=alt.X(
                    "lo:Q",
                    title="Share of the refit's advantage recovered (axis clipped to -1 to 1.5)",
                    scale=alt.Scale(domain=[-1, 1.5], clamp=True),
                ),
                x2="hi:Q",
                color=colour,
                yOffset="graft:N",
            )
            + base.mark_point(filled=True, size=50).encode(
                y=y,
                x=alt.X("S:Q", scale=alt.Scale(domain=[-1, 1.5], clamp=True)),
                color=colour,
                yOffset="graft:N",
            )
        )
        .properties(width=420, height=200)
        .facet(row=alt.Row("year:N", title=None))
        .properties(
            title=_title(
                title="The sunshine graft recovers a third to a half of the refit's advantage in both years",
                subtitle="Worst quarter of daylight half-hours by arm D's own score, 0.5 to 168 hours. D10 and D16 are placebos. Bar: 95% interval from resampling ISO weeks; bars past the axis are clipped (D9 and D10 have a small refit advantage, which makes the ratio unstable).",
            )
        )
    )
    _save(chart, name="graft_shares")


def main() -> None:
    """Draw every figure."""
    report = json.loads((common.REPORT_DIR / "report.json").read_text())
    stage3 = json.loads((common.REPORT_DIR / "stage3.json").read_text())
    headline(report=report)
    gain_by_group(report=report)
    recall(report=report)
    half_hour(report=report)
    graft(stage3=stage3)


if __name__ == "__main__":
    main()
