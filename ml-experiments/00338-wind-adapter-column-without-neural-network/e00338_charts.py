"""Draw experiment 00338's figures as SVG and print the numbers the write-up quotes.

Reads `work/score.json` and the tables in `work/tables/`. Afterwards optimise with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00338-*/e00338_charts.py   (PYTHONPATH as for the fit)
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00338_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00338-wind-adapter-column-without-neural-network"
)
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
GREY: Final[str] = "#999999"
LABELS: Final[dict[str, str]] = {
    "x4": "x4 neural adapter (experiment 326's (c))",
    "x1": "x1 lookup table",
    "x2": "x2 boosted trees",
    "x3": "x3 table copying the network",
    "x1_no_direction": "x1 without direction sector",
    "x1_no_ratio": "x1 without speed ratio",
    "x1_pl": "placebo for x1",
    "x2_pl": "placebo for x2",
}
COLOURS: Final[dict[str, str]] = {
    "x4": ocf.DATA_BLUE,
    "x1": ocf.DATA_GREEN,
    "x2": ocf.DATA_PURPLE,
    "x3": ocf.DATA_SKY,
    "x1_no_direction": ocf.DATA_GREEN,
    "x1_no_ratio": ocf.DATA_GREEN,
    "x1_pl": GREY,
    "x2_pl": GREY,
}
SCOPE: Final[str] = "3 wind farms, 3 seeds, rows kept by the mask, scored at 1 to 10 days."


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _colour(*, keys: list[str]) -> alt.Color:
    return alt.Color(
        "key:N",
        scale=alt.Scale(domain=keys, range=[COLOURS[k] for k in keys]),
        legend=None,
    )


def _interval_panel(*, rows: list[dict], title: alt.TitleParams, x_title: str) -> alt.TopLevelMixin:
    data = pl.DataFrame(rows)
    keys = list(dict.fromkeys(data["key"].to_list()))
    order = [LABELS[k] for k in keys]
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        color=_colour(keys=keys),
    )
    points = base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("estimate:Q", title=x_title)
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH, height=26 * len(order), title=title
    )


def _contrast_rows(*, score: dict, period: str, level: str) -> list[dict]:
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
    return rows


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """The scored year's and the replicate's contrasts against the default."""
    x_title = "Score minus the default's (points of capacity; more negative is better)"
    top = _interval_panel(
        rows=_contrast_rows(score=score, period="scored", level="0.9975"),
        x_title=x_title,
        title=_title(
            text="How much of the neural adapter's gain each non-neural column keeps",
            lines=[
                "Quantile score minus Wind farms research default 1's. Grey line: same as the "
                "default. Dot: estimate. Line: 99.75% interval from re-drawing whole months and "
                "seeds. Scored year, July 2025 to June 2026, 53 Tuesday runs. " + SCOPE
            ],
        ),
    )
    bottom = _interval_panel(
        rows=_contrast_rows(score=score, period="replicate", level="0.9975"),
        x_title=x_title,
        title=_title(
            text="The same contrasts on the replicate period",
            lines=[
                (
                    "Daily forecast runs from January to June 2025, with the offset table fitted "
                    "on April to December 2024. Six months only, so the 99.75% interval is wide."
                )
            ],
        ),
    )
    return alt.vconcat(top, bottom).resolve_scale(x="shared")  # ty: ignore[invalid-argument-type]


def figure_2(*, score: dict) -> alt.TopLevelMixin:
    """Each arm's share of the neural adapter's gain, overall and by lead."""
    rows = []
    for key in ("x1", "x2", "x3", "x1_no_direction", "x1_no_ratio"):
        for label, name in (
            ("all leads", "kept"),
            ("days 1 to 2", "days_1_2"),
            ("days 3 to 10", "days_3_10"),
        ):
            found = score["contrasts"]["scored"]["share_of_x4_gain"][key][name]
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
    order = [LABELS[k] for k in dict.fromkeys(data["key"].to_list())]
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("band:N"),
        color=_colour(keys=list(dict.fromkeys(data["key"].to_list()))),
        opacity=alt.Opacity(
            "band:N",
            scale=alt.Scale(
                domain=["all leads", "days 1 to 2", "days 3 to 10"], range=[1.0, 0.65, 0.35]
            ),
            legend=alt.Legend(title="Lead", orient="top"),
        ),
    )
    points = base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("estimate:Q", title="Share of the neural adapter's gain kept (1 = all of it)")
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    one = alt.Chart(pl.DataFrame({"x": [1.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(one, bars, points).properties(
        width=WIDTH,
        height=170,
        title=_title(
            text="Share of the neural adapter's gain kept, by lead",
            lines=[
                "Each arm's gain over the default divided by the neural adapter's gain over the "
                "default. Grey line: all of the gain. Line: 95% interval from re-drawing whole "
                "months and seeds. " + SCOPE
            ],
        ),
    )


def figure_3() -> alt.TopLevelMixin:
    """Table values by direction sector for x1 and x3, averaged over the other bins."""
    rows = []
    centres = np.arange(common.DIRECTION_SECTORS) * 360 / common.DIRECTION_SECTORS
    for key in ("x1", "x3"):
        for farm in common.FARMS:
            tables = np.stack(
                [np.load(common.TABLE_DIR / f"{key}_s{seed}_{farm}.npy") for seed in common.SEEDS]
            ).mean(axis=0)
            by_sector = tables.reshape(common.DIRECTION_SECTORS, -1).mean(axis=1)
            rows += [
                {
                    "sector": float(c),
                    "scale": float(v),
                    "farm": farm,
                    "key": key,
                    "arm": LABELS[key],
                }
                for c, v in zip(centres, by_sector, strict=True)
            ]
    data = pl.DataFrame(rows)
    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("sector:Q", title="Direction (degrees)"),
            y=alt.Y(
                "scale:Q",
                title="Speed scale, mean over the other bins",
                scale=alt.Scale(zero=False),
            ),
            color=_colour(keys=["x1", "x3"]),
            detail="arm:N",
        )
        .properties(width=170, height=170)
        .facet(column=alt.Column("farm:N", title=None))
    )
    return chart.properties(
        title=_title(
            text="The fitted table scales the wind speed by direction",
            lines=[
                (
                    "Green: x1, the fitted table. Light blue: x3, the table copying the network. "
                    "Each value is the unweighted mean over the 24 speed-ratio and temperature "
                    "bins of a sector, then over 3 seeds. Anonymised farms W1 to W3."
                )
            ],
        )
    )


def main() -> None:
    """Write the three figures and print the table's range."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_1(score=score).save(DOCS_DIR / "figure_1.svg")
    figure_2(score=score).save(DOCS_DIR / "figure_2.svg")
    figure_3().save(DOCS_DIR / "figure_3.svg")
    for key in ("x1", "x3"):
        for farm in common.FARMS:
            table = np.stack(
                [np.load(common.TABLE_DIR / f"{key}_s{s}_{farm}.npy") for s in common.SEEDS]
            ).mean(axis=0)
            print(key, farm, "min", float(table.min()), "max", float(table.max()))


if __name__ == "__main__":
    main()
