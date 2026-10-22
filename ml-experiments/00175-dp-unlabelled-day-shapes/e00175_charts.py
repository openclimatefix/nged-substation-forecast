"""Draw the figures of experiment 175's write-up from the stored tables.

uv run --no-sync python ml-experiments/00175-*/e00175_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00175_common as common
import e00175_run as run
import numpy as np
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
WRAP: Final[int] = 100
GREY: Final[str] = "#9A9A9A"
OWN: Final[str] = "Substation (arm D)"
OTHERS: Final[str] = "Neighbours (BG2)"
SHAPE_LABELS: Final[dict[str, str]] = {
    "a_block": "Sharp-edged block",
    "b_offset": "Smooth offset",
    "c_ramp_or_midday": "Ramp or midday-only",
    "d_spike": "Spike",
    "mixed": "None of the four",
}
PANEL_GROUPS: Final[tuple[tuple[str, ...], ...]] = (("D10",), ("D11",), ("D14",), ("D16", "D5"))
HOURS_BEFORE: Final[int] = 24
HOURS_AFTER: Final[int] = 48


def lines(*, text: str) -> list[str]:
    """Wrap text to the plot width."""
    return textwrap.wrap(text, WRAP)


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG into the docs folder and PNG into the private folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)


def class_shares_chart() -> alt.TopLevelMixin:
    """Return the class shares of the unlabelled worst days next to ordinary days."""
    parts = []
    for label, file in (
        ("37 unlabelled worst days", "day_shapes.parquet"),
        ("200 ordinary days", "ordinary_day_shapes.parquet"),
    ):
        table = pl.read_parquet(common.WORK_DIR / file)
        n = table.height
        for shape, name in SHAPE_LABELS.items():
            count = int((table["shape"] == shape).sum())
            low, high = run.wilson(hits=count, n=n)
            parts.append(
                {"days": label, "shape": name, "share": count / n, "low": low, "high": high}
            )
    data = pl.DataFrame(parts)
    order = list(SHAPE_LABELS.values())
    base = alt.Chart(data).encode(
        y=alt.Y("shape:N", sort=order, title=None),
        yOffset=alt.YOffset("days:N"),
        color=alt.Color(
            "days:N",
            scale=alt.Scale(
                domain=["37 unlabelled worst days", "200 ordinary days"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    bars = base.mark_bar(size=14).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("share:Q", title="Share of the days", axis=alt.Axis(format="%"))
    )
    rule = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x="low:Q", x2="high:Q", color=alt.value("#111111")
    )
    return (bars + rule).properties(
        width=WIDTH,
        height=230,
        title=alt.TitleParams(
            "Sharp-edged blocks are commoner on the unlabelled worst days than on ordinary days",
            subtitle=lines(
                text="Share of days in each error shape, from rules fixed before any day was "
                "looked at. Black line: 95% Wilson interval. Ordinary days: 40 random days per "
                "substation from the middle half of its days by excess loss."
            ),
        ),
    )


def neighbour_chart() -> alt.TopLevelMixin:
    """Return the share of block days with an opposite-signed neighbour step, against chance."""
    unlabelled = pl.read_parquet(common.WORK_DIR / "day_shapes.parquet")
    ordinary = pl.read_parquet(common.WORK_DIR / "ordinary_day_shapes.parquet")
    chance = pl.read_parquet(common.WORK_DIR / "chance.parquet")
    rows = []
    for label, table in (("Unlabelled worst days", unlabelled), ("Ordinary days", ordinary)):
        blocks = table.filter(pl.col("shape") == "a_block", pl.col("n_metered_neighbours") > 0)
        hits = int(blocks["neighbour_step"].sum())
        low, high = run.wilson(hits=hits, n=blocks.height)
        rows.append(
            {
                "row": f"{label}: observed ({blocks.height} block days)",
                "share": hits / blocks.height,
                "low": low,
                "high": high,
            }
        )
    rows.append(
        {
            "row": "Unlabelled days: neighbours rolled by days",
            "share": float(chance["rate"].to_numpy().mean()),
            "low": float(np.quantile(chance["rate"].to_numpy(), 0.025)),
            "high": float(np.quantile(chance["rate"].to_numpy(), 0.975)),
        }
    )
    data = pl.DataFrame(rows)
    order = list(data["row"])
    base = alt.Chart(data).encode(
        y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=300))
    )
    dots = base.mark_point(filled=True, size=90, color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "share:Q",
            title="Share of block days",
            axis=alt.Axis(format="%"),
            scale=alt.Scale(domain=[0, 1]),
        )
    )
    rule = base.mark_rule(strokeWidth=2, color="#111111").encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    third = (
        alt.Chart(pl.DataFrame({"x": [1 / 3]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 4], color=GREY)
        .encode(x="x:Q")
    )
    return (rule + dots + third).properties(
        width=WIDTH,
        height=140,
        title=alt.TitleParams(
            "A neighbour steps the other way about as often by chance as it does on the real days",
            subtitle=lines(
                text="Share of sharp-edged block days on which an interconnected Primary shows a "
                "step of opposite sign within 2 hours. Dot: estimate. Line: 95% interval (Wilson "
                "for the observed rows, 2.5th to 97.5th percentile of 1,000 draws for the rolled "
                "row). Dashed line: the one-third threshold."
            ),
        ),
    )


def one_panel(*, ctx: run.Context, row: dict) -> alt.TopLevelMixin:
    """Return the panel of one day: the residual, neighbours, masked rows, onsets, and edge."""
    grid, site, day = ctx.grid, row["site"], row["local_date"]
    start = int(grid.positions_of(day=day)[0])
    window = np.arange(start - 2 * HOURS_BEFORE, start + 2 * HOURS_AFTER)
    hours = (window - start) / 2.0
    colours = alt.Scale(domain=[OTHERS, OWN], range=[GREY, ocf.DATA_BLUE])
    own = pl.DataFrame({"hour": hours, "mw": grid.residual[site][window], "who": OWN})
    others = pl.concat(
        [
            pl.DataFrame({"hour": hours, "mw": ctx.bg2[n][window], "who": OTHERS, "neighbour": n})
            for n in sorted(ctx.link[site])
            if n in ctx.bg2
        ]
        or [
            pl.DataFrame(
                {"hour": [0.0], "mw": [None], "who": OTHERS, "neighbour": "none"},
                schema={
                    "hour": pl.Float64,
                    "mw": pl.Float64,
                    "who": pl.String,
                    "neighbour": pl.String,
                },
            )
        ]
    )
    x = alt.X(
        "hour:Q",
        title=None,
        scale=alt.Scale(domain=[-HOURS_BEFORE, HOURS_AFTER]),
        axis=alt.Axis(values=[-24, 0, 24, 48]),
    )
    color = alt.Color("who:N", scale=colours, legend=alt.Legend(orient="top", title=None))
    layers = [
        alt.Chart(others)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=0.7, opacity=0.8)
        .encode(x=x, y=alt.Y("mw:Q", title=None), detail="neighbour:N", color=color),
        alt.Chart(own).mark_line(strokeWidth=1.2).encode(x=x, y="mw:Q", color=color),  # ty: ignore[unresolved-attribute]
        alt.Chart(pl.DataFrame({"hour": [0.0, 24.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#CCCCCC")
        .encode(x="hour:Q"),
    ]
    masked = common.arm_d_masked_times(site=site)
    first, last = grid.end_times[int(window[0])], grid.end_times[int(window[-1])]
    ticks = [
        (grid.end_times.search_sorted(t) - start) / 2.0
        for t in masked.filter(masked.is_between(first, last))
    ]
    if ticks:
        layers.append(
            alt.Chart(pl.DataFrame({"hour": ticks}))  # ty: ignore[unresolved-attribute]
            .mark_tick(color=ocf.BRAND_ORANGE, size=8, orient="vertical", thickness=1)
            .encode(x="hour:Q", y=alt.value(92))
        )
    onsets = []
    for table in ctx.detectors.values():
        for o in table.filter(pl.col("series") == ctx.ids[site])["onset"].to_list():
            p = grid.position_of_stamp(stamp=int(o))
            if window[0] <= p <= window[-1]:
                onsets.append((p - start) / 2.0)
    if onsets:
        layers.append(
            alt.Chart(pl.DataFrame({"hour": onsets}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color=ocf.DATA_PURPLE, strokeWidth=2)
            .encode(x="hour:Q")
        )
    if row.get("edge_position") is not None:
        layers.append(
            alt.Chart(pl.DataFrame({"hour": [(row["edge_position"] - start) / 2.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color="#111111", strokeDash=[3, 3])
            .encode(x="hour:Q")
        )
    return alt.layer(*layers).properties(
        width=300,
        height=95,
        title=alt.TitleParams(f"{site}, {day}: {SHAPE_LABELS[row['shape']]}", fontSize=10),
    )


def day_panels(
    *, sites: tuple[str, ...], ctx: run.Context, table: pl.DataFrame
) -> alt.TopLevelMixin:
    """Return a two-column grid of day panels for the given sites."""
    panels = [
        one_panel(ctx=ctx, row=row)
        for row in table.filter(pl.col("site").is_in(list(sites))).iter_rows(named=True)
    ]
    return (
        alt.concat(*panels, columns=2)  # ty: ignore[invalid-argument-type]
        .resolve_scale(y="independent", color="shared")
        .properties(
            title=alt.TitleParams(
                f"Residual on the unlabelled worst days: {', '.join(sites)}",
                subtitle=lines(
                    text="Blue: substation's residual (observed minus forecast, points of "
                    "effective capacity, arm D). Grey: "
                    "interconnected Primaries' residual (BG2). Dashed black: the day's sharpest "
                    "edge, if any. Purple: detector onset. Orange ticks: hand-filtered rows left "
                    "out of the scoring. Grey rules mark the day's start and end; the x axis "
                    "shows hours from the day's start."
                ),
            )
        )
    )


def main() -> None:
    """Draw all figures."""
    run_ctx = None
    save(chart=class_shares_chart(), name="fig1_classes")
    save(chart=neighbour_chart(), name="fig2_neighbour")
    table = pl.read_parquet(common.WORK_DIR / "day_shapes.parquet")
    sites = tuple(sorted(table["site"].unique().to_list()))
    residuals = common.arm_d_residuals(sites=sites)
    bg2 = common.bg2_residuals()
    grid = common.ResidualGrid(residual=residuals)
    run_ctx = run.Context(
        grid=grid,
        bg2=bg2,
        bg2_scale={},
        link=common.neighbours(),
        ids=common.series_ids(),
        detectors=run.detector_tables(),
    )
    for number, group in enumerate(PANEL_GROUPS, start=3):
        save(
            chart=day_panels(sites=group, ctx=run_ctx, table=table),
            name=f"fig{number}_days_{'_'.join(group).lower()}",
        )


if __name__ == "__main__":
    main()
