"""Draw experiment 84's four figures from the saved results, as SVG.

No generator appears: the charts hold node flows, node limits, and sums over every generator. Run
`e00084_report.py` and `e00084_known_answer.py` first. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00084-*/e00084_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00084_inputs as inputs
import e00084_report as report
import plotting.ocf_theme as ocf
import polars as pl
from studies.node_curtailment import SEASON_NAMES, season_expression

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{inputs.EXPERIMENT_ID:05d}-{inputs.SLUG}"
)
WIDTH: Final[int] = 640
WRAP: Final[int] = 100
SHOWN_NODES: Final[tuple[str, ...]] = (inputs.BOSTON, inputs.SKEGNESS, inputs.BICKER_FEN)
LIMIT_ORDER: Final[tuple[str, ...]] = ("real", "scaled f=0.9", "scaled f=0.75", "scaled f=0.5")
LIMIT_COLOURS: Final[tuple[str, ...]] = (
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_PURPLE,
    ocf.DATA_SKY,
)
LIMIT_DASHES: Final[tuple[tuple[int, int], ...]] = ((1, 0), (6, 3), (3, 3), (1, 3))
LIMIT_NAMES: Final[dict[str, str]] = {
    "real": "Published thermal limit",
    "scaled f=0.9": "0.9 x maximum",
    "scaled f=0.75": "0.75 x maximum",
    "scaled f=0.5": "0.5 x maximum",
}
SHORT_LIMIT_NAMES: Final[dict[str, str]] = {
    "real": "thermal limit",
    "scaled f=0.9": "0.9 x max",
    "scaled f=0.75": "0.75 x max",
    "scaled f=0.5": "0.5 x max",
}
SEASON_LABEL_EXPRESSION: Final[str] = (
    "replace(replace(replace(datum.label, 'Intermediate ', 'Int. '), 'Warm', "
    "'warm'), 'Cool', 'cool')"
)
RESULTS: Final[Path] = inputs.DATA_DIR / "results"
SLUG_OF_KIND: Final[dict[str, str]] = {k: k.replace(" ", "_").replace("=", "") for k in LIMIT_ORDER}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _node(*, node_id: str) -> str:
    suffix = " (sign unconfirmed)" if node_id == inputs.BICKER_FEN else ""
    return inputs.NODE_LABELS[node_id] + suffix


def figure_1_census() -> alt.VConcatChart:
    """Export and import exceedances per node, season, limit and period, as a labelled heat map."""
    counts = pl.read_parquet(RESULTS / "census.parquet").filter(
        pl.col("node_id").is_in(SHOWN_NODES)
    )
    counts = counts.with_columns(
        row=pl.col("node_id").replace_strict(
            {
                n: inputs.NODE_LABELS[n] + ("*" if n == inputs.BICKER_FEN else "")
                for n in SHOWN_NODES
            }
        )
        + pl.lit(": ")
        + pl.col("limit_kind").replace_strict(SHORT_LIMIT_NAMES)
        + pl.lit(" ")
        + pl.col("limit_mw").round(1).cast(pl.String)
        + pl.lit(" MW"),
        order=pl.col("node_id").replace_strict({n: i for i, n in enumerate(SHOWN_NODES)}) * 10
        + pl.col("limit_kind").replace_strict({k: i for i, k in enumerate(LIMIT_ORDER)}),
    )
    panels = []
    for direction in ("export", "import"):
        part = counts.filter(pl.col("direction") == direction).sort("order")
        rows = part.sort("order")["row"].unique(maintain_order=True).to_list()
        base = alt.Chart(part).encode(
            x=alt.X(
                "season:N",
                sort=list(SEASON_NAMES),
                title=None,
                axis=alt.Axis(
                    labelAngle=-35,
                    labelExpr=SEASON_LABEL_EXPRESSION,
                ),
            ),
            y=alt.Y("row:N", sort=rows, title=None, axis=alt.Axis(labelLimit=260)),
        )
        heat = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
            color=alt.Color(
                "n_exceedances:Q",
                scale=alt.Scale(type="symlog", range=[ocf.WHITE, ocf.BRAND_ORANGE]),
                legend=None,
            )
        )
        text = base.mark_text(fontSize=10, aria=False).encode(  # ty: ignore[unresolved-attribute]
            text="n_exceedances:Q",
            color=alt.value(ocf.BLACK_1),
        )
        panels.append(
            alt.layer(heat, text)
            .properties(
                width=150, height=13 * len(rows), title=f"{direction.capitalize()} exceedances"
            )
            .facet(column=alt.Column("period:N", title=None, sort=["all history", "scored year"]))
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="Few half-hours cross the thermal limit; hundreds cross half the observed peak",
            lines=[
                (
                    "Number of half-hours whose flow is beyond the limit, by season. The "
                    "published thermal limit is a ceiling NGED does not operate to; each scaled "
                    "limit (0.9, 0.75 or 0.5 times the node's maximum before 2025-07-02) stands "
                    "in for the headroom NGED keeps. All history is 2019-09 (Skegness, Bicker "
                    "Fen) or 2022-07 (Boston) to 2026-06-30; the scored year is 2025-07-02 to "
                    "2026-06-30. * Bicker Fen's sign is unconfirmed, and its thermal limit is a "
                    "Technical Limit. No published import limit is used. Thermal limits are "
                    "NGED's, Supported by NGED Open Data."
                )
            ],
        )
    )


def figure_2_distribution() -> alt.VConcatChart:
    """Each node's net-export distribution by season, with the real and scaled limits marked."""
    exports = pl.read_parquet(RESULTS / "exports.parquet").filter(
        pl.col("node_id").is_in(SHOWN_NODES)
    )
    width_of = {inputs.BOSTON: 1.0, inputs.SKEGNESS: 1.0, inputs.BICKER_FEN: 5.0}
    histogram = (
        exports.with_columns(
            season=season_expression(time_column="valid_time"),
            width=pl.col("node_id").replace_strict(width_of),
        )
        .with_columns(bin=(pl.col("export_mw") / pl.col("width")).floor() * pl.col("width"))
        .group_by("node_id", "season", "bin")
        .agg(half_hours=pl.len())
        .with_columns(
            label=pl.col("node_id").replace_strict({n: _node(node_id=n) for n in SHOWN_NODES})
        )
    )
    scaled = pl.read_parquet(RESULTS / "scaled_limits.parquet").filter(
        pl.col("node_id").is_in(SHOWN_NODES)
    )
    real = pl.DataFrame(
        {
            n: [
                max(inputs.EXPORT_LIMITS_MW[n])
                if n == inputs.BICKER_FEN
                else min(inputs.EXPORT_LIMITS_MW[n])
            ]
            for n in SHOWN_NODES
        }
    ).unpivot(variable_name="node_id", value_name="limit_mw")
    marks = pl.concat(
        [
            real.with_columns(kind=pl.lit("real")),
            scaled.select(
                "node_id",
                limit_mw="export_limit_mw",
                kind=pl.format("scaled f={}", pl.col("fraction")),
            ),
        ]
    ).join(pl.DataFrame({"season": list(SEASON_NAMES)}), how="cross")
    marks = marks.with_columns(
        label=pl.col("node_id").replace_strict({n: _node(node_id=n) for n in SHOWN_NODES}),
        kind_name=pl.col("kind").replace_strict(LIMIT_NAMES),
    )
    colour = alt.Color(
        "kind_name:N",
        title=None,
        scale=alt.Scale(domain=[LIMIT_NAMES[k] for k in LIMIT_ORDER], range=list(LIMIT_COLOURS)),
        legend=alt.Legend(orient="top"),
    )
    dash = alt.StrokeDash(
        "kind_name:N",
        scale=alt.Scale(
            domain=[LIMIT_NAMES[k] for k in LIMIT_ORDER], range=[list(d) for d in LIMIT_DASHES]
        ),
        legend=None,
    )
    rows = []
    for season in SEASON_NAMES:
        cells = []
        for node in SHOWN_NODES:
            label = _node(node_id=node)
            hist = histogram.filter(pl.col("node_id") == node, pl.col("season") == season)
            lines = (
                alt.Chart(hist)  # ty: ignore[unresolved-attribute]
                .mark_line(interpolate="step-after", color=ocf.BLACK_1, aria=False)
                .encode(
                    x=alt.X("bin:Q", title="Net export (MW)" if season == "Winter" else None),
                    y=alt.Y(
                        "half_hours:Q",
                        scale=alt.Scale(type="log", domain=[1, 10000]),
                        title=season if node == SHOWN_NODES[0] else None,
                        axis=alt.Axis(values=[1, 10, 100, 1000, 10000]),
                    ),
                )
            )
            rules = (
                alt.Chart(marks.filter(pl.col("node_id") == node, pl.col("season") == season))  # ty: ignore[unresolved-attribute]
                .mark_rule(strokeWidth=1.6, aria=False)
                .encode(x="limit_mw:Q", color=colour, strokeDash=dash)
            )
            cells.append(
                alt.layer(lines, rules).properties(
                    width=170, height=62, title=label if season == "Winter" else ""
                )
            )
        rows.append(alt.hconcat(*cells, spacing=14))
    return (
        alt.vconcat(*rows, spacing=6)
        .properties(
            title=_title(
                text="Each node's limit sits far out in the tail of its export",
                lines=[
                    (
                        "Half-hours of net export per MW (Bicker Fen: per 5 MW), all history, from "
                        "cleaned unflagged power. Rows are seasons. Bicker Fen's thermal limit is "
                        "a Technical Limit and its sign is unconfirmed. The published thermal "
                        "limit is a ceiling NGED does not operate to. Limits are NGED's, "
                        "Supported by NGED Open Data."
                    )
                ],
            )
        )
        .resolve_scale(color="shared")
    )


def figure_3_known_answer() -> alt.Chart:
    """One made-up half-hour through the whole chain, beside the hand-worked answer."""
    rows = json.loads((RESULTS / "known_answer.json").read_text())
    quantities = [r["quantity"] for r in rows]
    cells = pl.DataFrame(
        [
            {"quantity": r["quantity"], "column": col, "text": text}
            for r in rows
            for col, text in (
                ("By hand", f"{r['hand']:,.1f}"),
                ("By the code", f"{r['code']:,.1f}"),
                ("Agree?", "yes" if r["agrees"] else "NO"),
            )
        ]
    )
    return (
        alt.Chart(cells)  # ty: ignore[unresolved-attribute]
        .mark_text(fontSize=12, aria=False)
        .encode(
            x=alt.X(
                "column:N",
                sort=["By hand", "By the code", "Agree?"],
                title=None,
                axis=alt.Axis(orient="top", labelAngle=0, labelFontSize=11, ticks=False),
            ),
            y=alt.Y("quantity:N", sort=quantities, title=None, axis=alt.Axis(labelLimit=320)),
            text="text:N",
            color=alt.condition(
                alt.datum.text == "NO", alt.value(ocf.BRAND_ORANGE), alt.value(ocf.BLACK_1)
            ),
        )
        .properties(
            width=WIDTH - 360,
            height=24 * len(quantities),
            title=_title(
                text="The code reproduces a hand-worked half-hour exactly",
                lines=[
                    (
                        "Made-up bulk supply point, 45.0 MW limit. Forecast 95% export 70 MW; "
                        "actual export 65 MW. Two made-up generators of 10 MW and 20 MW forecast "
                        "8 MW "
                        "and 18 MW and produce 8 MW and 12 MW. The energy proxy is "
                        "GBP 2,000 x (MWh "
                        "curtailed + 20 x MWh of breach). The first generator's share is cut "
                        "to its "
                        "own forecast energy and the second takes the rest."
                    )
                ],
            ),
        )
    )


def figure_4_headline() -> alt.VConcatChart:
    """Avoidable cost per arm at the three scaled limits, with other prices beside it."""
    order = [
        report.NGED_BASELINE,
        report.CLIMATOLOGY,
        report.STACK,
        report.BEST_GUESS,
        report.DIAGNOSTIC,
    ]
    labels = [report.ARM_LABELS[a] for a in order]
    colours = [ocf.DATA_PURPLE, ocf.DATA_SKY, ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]
    panels = []
    for kind in ("scaled f=0.9", "scaled f=0.75", "scaled f=0.5"):
        slug = SLUG_OF_KIND[kind]
        intervals = pl.read_parquet(RESULTS / f"intervals_{slug}.parquet")
        avoidable = pl.read_parquet(RESULTS / f"avoidable_{slug}.parquet")
        marks = intervals.filter(pl.col("arm").is_in(order)).with_columns(
            label=pl.col("arm").replace_strict(report.ARM_LABELS),
        )
        dot = alt.Chart(marks).encode(
            y=alt.Y("label:N", sort=labels, title=None, axis=alt.Axis(labelLimit=235)),
        )
        bars = dot.mark_rule(strokeWidth=2, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("avoidable_low_gbp:Q", title="Avoidable cost (GBP; smaller is better)"),
            x2="avoidable_high_gbp:Q",
            color=alt.Color("label:N", scale=alt.Scale(domain=labels, range=colours), legend=None),
        )
        points = dot.mark_point(filled=True, size=70, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x="avoidable_gbp:Q",
            color=alt.Color("label:N", scale=alt.Scale(domain=labels, range=colours), legend=None),
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3], aria=False)
            .encode(x="x:Q")
        )
        main = alt.layer(zero, bars, points).properties(width=170, height=24 * len(order))
        beside = avoidable.filter(pl.col("arm").is_in(order)).select(
            "arm", "price_gbp_per_mwh", "energy_proxy_gbp", "dcusa_form_gbp"
        )
        text_rows = []
        for arm_row in beside.iter_rows(named=True):
            if arm_row["price_gbp_per_mwh"] == 2000.0:
                text_rows.append(
                    (arm_row["arm"], "DCUSA form, GBP 2,000", arm_row["dcusa_form_gbp"] or 0.0)
                )
            else:
                text_rows.append(
                    (
                        arm_row["arm"],
                        f"GBP {arm_row['price_gbp_per_mwh']:,.0f} per MWh",
                        arm_row["energy_proxy_gbp"],
                    )
                )
        text_frame = pl.DataFrame(
            text_rows, schema=["arm", "column", "value"], orient="row"
        ).with_columns(
            label=pl.col("arm").replace_strict(report.ARM_LABELS),
            text=pl.col("value").map_elements(lambda v: f"{v:,.0f}", return_dtype=pl.String),
        )
        columns = [
            "GBP 100 per MWh",
            "GBP 380 per MWh",
            "GBP 11,100 per MWh",
            "DCUSA form, GBP 2,000",
        ]
        table = (
            alt.Chart(text_frame)  # ty: ignore[unresolved-attribute]
            .mark_text(fontSize=10, aria=False)
            .encode(
                x=alt.X(
                    "column:N",
                    sort=columns,
                    title=None,
                    axis=alt.Axis(orient="bottom", labelAngle=-30, labelFontSize=9),
                ),
                y=alt.Y("label:N", sort=labels, axis=None),
                text="text:N",
            )
            .properties(width=215, height=24 * len(order))
        )
        kind_note = {
            "scaled f=0.9": "Limit 0.9 x maximum (headline): calibration to history, not a saving",
            "scaled f=0.75": "Limit 0.75 x maximum: calibration to history, not a saving",
            "scaled f=0.5": "Limit 0.5 x maximum: calibration to history, not a saving",
        }[kind]
        panels.append(alt.hconcat(main, table, spacing=4).properties(title=kind_note))
    return alt.vconcat(*panels).properties(
        title=_title(
            text="No forecast can be ranked at the headline limit: a few events decide each arm",
            lines=[
                (
                    "Avoidable cost is each arm's cost minus the perfect forecast's, over the "
                    "scored year, July 2025 to June 2026, Boston and Skegness, Curtailment Limit "
                    "0. Dot: estimate. Line: 95% interval from resampling whole months. Right: "
                    "avoidable cost in GBP at the other prices, and in the DCUSA form. The limit "
                    "is a multiple of each node's maximum before the scored year, a stand-in for "
                    "the headroom NGED keeps. The NGED baseline is the maintainer's unconfirmed "
                    "understanding of NGED's practice."
                )
            ],
        )
    )


def main() -> None:
    """Write the four figures as SVG into the docs folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    charts = {
        "figure_1_event_census": figure_1_census(),
        "figure_2_export_distribution": figure_2_distribution(),
        "figure_3_known_answer": figure_3_known_answer(),
        "figure_4_avoidable_cost": figure_4_headline(),
    }
    for name, chart in charts.items():
        chart.save(DOCS_DIR / f"{name}.svg")
        print("wrote", name)


if __name__ == "__main__":
    main()
