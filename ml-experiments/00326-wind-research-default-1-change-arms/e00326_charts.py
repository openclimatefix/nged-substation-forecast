"""Draw experiment 00326's figures as SVG and print the numbers the write-up quotes.

Reads `work/score.json` and `work/rows.parquet`. Afterwards optimise with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00326-*/e00326_charts.py   (PYTHONPATH as for the fit)
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00326_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00326-wind-research-default-1-change-arms"
)
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
GREY: Final[str] = "#999999"
LABELS: Final[dict[str, str]] = {
    "c": "(c) adapter's effective speed",
    "b": "(b) hour and day of year",
    "a": "(a) temperature and density",
    "d": "(d) speed factor 1.13",
    "c_pl": "placebo for (c)",
    "b_pl": "placebo for (b)",
    "a_pl": "placebo for (a)",
}
COLOURS: Final[dict[str, str]] = {
    "c": ocf.DATA_BLUE,
    "b": ocf.DATA_GREEN,
    "a": ocf.DATA_PURPLE,
    "d": ocf.DATA_SKY,
    "c_pl": GREY,
    "b_pl": GREY,
    "a_pl": GREY,
}
SCOPE: Final[str] = "3 wind farms, 3 seeds, rows kept by the mask, scored at 1 to 10 days."


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _contrast_panel(
    *, score: dict, period: str, level: str, title: alt.TitleParams
) -> alt.TopLevelMixin:
    rows = []
    for key in LABELS:
        entry = score["contrasts"][period].get(key)
        if entry is None:
            continue
        found = entry["kept | qs"]
        low, high = found[level]
        rows.append(
            {"arm": LABELS[key], "key": key, "estimate": found["mean"], "lower": low, "upper": high}
        )
    data = pl.DataFrame(rows)
    order = [LABELS[k] for k in LABELS if k in set(data["key"])]
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color(
            "key:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=None,
        ),
    )
    points = base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title="Score minus the default's (points of capacity; more negative is better)",
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH, height=26 * len(order), title=title
    )


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """The scored year's and the replicate's contrasts against the default."""
    top = _contrast_panel(
        score=score,
        period="scored",
        level="0.9975",
        title=_title(
            text="Only the adapter's speed gains with an interval below zero",
            lines=[
                "Quantile score minus Wind farms research default 1's. Grey line: same as the "
                "default. Dot: estimate. Line: 99.75% interval from re-drawing whole months and "
                "seeds. Scored year, July 2025 to June 2026, 53 Tuesday runs. " + SCOPE
            ],
        ),
    )
    bottom = _contrast_panel(
        score=score,
        period="replicate",
        level="0.9975",
        title=_title(
            text="The same contrasts on the replicate period",
            lines=[
                (
                    "Daily forecast runs from January to June 2025, with the offset table fitted "
                    "on April to December 2024. Same marks. Six months only, so the 99.75% "
                    "interval is wide."
                )
            ],
        ),
    )
    return alt.vconcat(top, bottom).resolve_scale(x="shared")  # ty: ignore[invalid-argument-type]


def figure_2(*, score: dict) -> alt.TopLevelMixin:
    """Each arm's gain at lead days 1 to 2 and days 3 to 10, 95% intervals."""
    rows = []
    for key in ("c", "b", "a", "d"):
        for label, name in (("days 1 to 2", "days_1_2"), ("days 3 to 10", "days_3_10")):
            found = score["contrasts"]["scored"][key][f"{name} | qs"]
            low, high = found["0.9500"]
            rows.append(
                {
                    "arm": LABELS[key],
                    "key": key,
                    "band": label,
                    "estimate": found["mean"],
                    "lower": low,
                    "upper": high,
                }
            )
    data = pl.DataFrame(rows)
    order = [LABELS[k] for k in ("c", "b", "a", "d")]
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("band:N"),
        color=alt.Color(
            "key:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=None,
        ),
        opacity=alt.Opacity(
            "band:N",
            scale=alt.Scale(domain=["days 1 to 2", "days 3 to 10"], range=[1.0, 0.5]),
            legend=alt.Legend(title="Lead", orient="top"),
        ),
    )
    points = base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title="Score minus the default's (points of capacity; more negative is better)",
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH,
        height=90,
        title=_title(
            text="The adapter's gain is the same at short and long leads",
            lines=[
                "Quantile score minus the default's by lead. Dark: lead days 1 to 2. Light: days "
                "3 to 10. Line: 95% interval. " + SCOPE
            ],
        ),
    )


def figure_3(*, rows: pl.DataFrame) -> alt.TopLevelMixin:
    """Monthly score minus the default's for the arms that gain."""
    pieces = []
    for key in ("c", "b"):
        monthly = (
            rows.filter(~pl.col("masked"))
            .group_by("month")
            .agg((pl.col(f"qs__{key}") - pl.col("qs__stored_default")).mean().alias("difference"))
            .with_columns(key=pl.lit(key), arm=pl.lit(LABELS[key]))
        )
        pieces.append(monthly)
    data = pl.concat(pieces).sort("month")
    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:N", title="Month of the forecast run", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(
                "difference:Q",
                title="Score minus default (points)",
            ),
            color=alt.Color(
                "key:N",
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=None,
            ),
            detail="arm:N",
        )
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=GREY).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, chart).properties(
        width=WIDTH,
        height=200,
        title=_title(
            text="Arm (c) gains in most months",
            lines=[
                "Blue: (c) the adapter's effective speed. Green: (b) hour and day of year. Grey "
                "line: same as the default. More negative is better. " + SCOPE
            ],
        ),
    )


def main() -> None:
    """Write the three figures."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    rows = pl.read_parquet(common.WORK_DIR / "rows.parquet")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_1(score=score).save(DOCS_DIR / "figure_1.svg")
    figure_2(score=score).save(DOCS_DIR / "figure_2.svg")
    figure_3(rows=rows).save(DOCS_DIR / "figure_3.svg")
    monthly = (
        rows.filter(~pl.col("masked"))
        .group_by("month")
        .agg((pl.col("qs__c") - pl.col("qs__stored_default")).mean().alias("c_minus_default"))
        .sort("month")
    )
    print(monthly)


if __name__ == "__main__":
    main()
