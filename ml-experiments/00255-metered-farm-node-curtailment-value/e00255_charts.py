# ruff: noqa: E501
"""Draw experiment 255's figures from the saved results, as SVG.

No farm appears singly: the charts hold sums over every generator beneath a bulk supply point.
Run `e00255_report.py` first. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00255-*/e00255_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00255_arms as arms
import e00255_inputs as inputs
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{inputs.EXPERIMENT_ID:05d}-{inputs.SLUG}"
)
RESULTS: Final[Path] = inputs.RESULTS_DIR
WIDTH: Final[int] = 600
WRAP: Final[int] = 100
ARM_ORDER: Final[tuple[str, ...]] = (
    arms.CAPACITY_BASELINE,
    arms.FLOOR_HEURISTIC,
    arms.FLOOR_CLIMATOLOGY,
    arms.PLANT,
    arms.XGBOOST,
    arms.H5,
    arms.H5_ZERO,
)
ARM_COLOURS: Final[tuple[str, ...]] = (
    ocf.BRAND_ORANGE,
    ocf.GREY_3,
    ocf.BLACK_1,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_PURPLE_LIGHT,
)
KIND_FILES: Final[dict[str, str]] = {
    "0.9 x maximum": "scaled_f0.9",
    "0.75 x maximum": "scaled_f0.75",
    "0.5 x maximum": "scaled_f0.5",
}
PAIRING_NAMES: Final[dict[str, str]] = {
    "causal": "Causal rows (headline)",
    "valid_time": "Valid-time pairing (sensitivity)",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _arm_scale() -> alt.Scale:
    return alt.Scale(domain=[arms.ARM_LABELS[a] for a in ARM_ORDER], range=list(ARM_COLOURS))


def figure_1_census() -> alt.LayerChart:
    """Event half-hours on the scored rows, by limit and pairing."""
    rows = []
    for pairing in PAIRING_NAMES:
        for name, slug in KIND_FILES.items():
            census = pl.read_parquet(RESULTS / f"{pairing}_census_{slug}.parquet")
            exported = census.filter(pl.col("direction") == "export")["n_exceedances"].sum()
            rows.append({"pairing": PAIRING_NAMES[pairing], "limit": name, "events": exported})
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame).encode(
        y=alt.Y("limit:N", sort=list(KIND_FILES), title=None),
        x=alt.X("events:Q", title="Half-hours of export beyond the limit (two bulk supply points)"),
        color=alt.Color(
            "pairing:N",
            scale=alt.Scale(
                domain=list(PAIRING_NAMES.values()), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
            ),
            legend=alt.Legend(orient="top", title=None, direction="vertical", labelLimit=400),
        ),
        yOffset="pairing:N",
    )
    bars = base.mark_bar()
    text = base.mark_text(align="left", dx=3, fontSize=11, aria=False).encode(  # ty: ignore[unresolved-attribute]
        text="events:Q", color=alt.value(ocf.BLACK_1)
    )
    return (bars + text).properties(
        width=WIDTH - 140,
        height=130,
        title=_title(
            text="Few half-hours exceed a limit above half the earlier maximum",
            lines=[
                (
                    "Half-hours of metered export beyond each limit among the scored node rows,"
                    " 2025-09 to 2026-06. A limit is a fraction of each node's maximum export before"
                    " 2025-09-01."
                )
            ],
        ),
    )


def figure_2_headline() -> alt.LayerChart:
    """Avoidable cost per arm at 0.5 times the maximum, causal rows, with 95% intervals."""
    frame = pl.read_parquet(RESULTS / "causal_intervals_scaled_f0.5.parquet").filter(
        pl.col("arm").is_in(ARM_ORDER)
    )
    frame = frame.with_columns(
        label=pl.col("arm").replace_strict(arms.ARM_LABELS),
        low=pl.col("avoidable_low_gbp") / 1e6,
        high=pl.col("avoidable_high_gbp") / 1e6,
        value=pl.col("avoidable_gbp") / 1e6,
    )
    order = [arms.ARM_LABELS[a] for a in ARM_ORDER]
    base = alt.Chart(frame).encode(y=alt.Y("label:N", sort=order, title=None))
    rule = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x="low:Q", x2="high:Q", color=alt.Color("label:N", scale=_arm_scale(), legend=None)
    )
    dot = base.mark_point(filled=True, size=90, opacity=1, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("value:Q", title="Avoidable cost (GBP million, smaller is better)"),
        color=alt.Color("label:N", scale=_arm_scale(), legend=None),
    )
    return (rule + dot).properties(
        width=WIDTH - 200,
        height=200,
        title=_title(
            text="Every farm forecast avoids about a quarter of the baseline's cost, within noise",
            lines=[
                (
                    "Avoidable cost at 0.5 times each node's earlier maximum export, GBP 2,000 per MWh,"
                    " causal rows (2025-09 to 2026-06). Dot: estimate. Line: 95% interval from"
                    " resampling whole months. Smaller is better."
                )
            ],
        ),
    )


def figure_3_nodes() -> alt.Chart:
    """Avoidable cost per arm at each bulk supply point, causal rows, 0.5 times the maximum."""
    frame = pl.read_parquet(RESULTS / "causal_per_node_scaled_f0.5.parquet").filter(
        pl.col("arm").is_in(ARM_ORDER)
    )
    frame = frame.with_columns(
        label=pl.col("arm").replace_strict(arms.ARM_LABELS),
        node=pl.col("node_id").str.split(" ").list.first().str.to_titlecase(),
        value=pl.col("avoidable_gbp") / 1e6,
    )
    order = [arms.ARM_LABELS[a] for a in ARM_ORDER]
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labels=True)),
            x=alt.X("value:Q", title="Avoidable cost (GBP million, smaller is better)"),
            color=alt.Color("label:N", scale=_arm_scale(), legend=None),
            row=alt.Row("node:N", title=None, header=alt.Header(labelFontSize=12)),
        )
        .properties(
            width=WIDTH - 200,
            height=130,
            title=_title(
                text="The farm forecasts help at Boston and do worse than the baseline at Skegness",
                lines=[
                    (
                        "Avoidable cost at 0.5 times each node's earlier maximum export, causal rows,"
                        " a sum over the generators beneath each node. Smaller is better."
                    )
                ],
            ),
        )
    )


def figure_4_contrast() -> alt.VConcatChart:
    """H5 minus XGBoost and H5 minus the plant, as a share of XGBoost's avoidable cost."""
    rows = []
    for pairing in PAIRING_NAMES:
        for name, slug in KIND_FILES.items():
            contrasts = pl.read_parquet(RESULTS / f"{pairing}_contrasts_{slug}.parquet")
            avoidable = pl.read_parquet(RESULTS / f"{pairing}_intervals_{slug}.parquet")
            scale = avoidable.filter(pl.col("arm") == arms.XGBOOST)["avoidable_gbp"].item()
            for contrast in ("h5 minus xgboost", "h5 minus plant_two_number"):
                row = contrasts.filter(pl.col("contrast") == contrast).row(0, named=True)
                rows.append(
                    {
                        "limit": name,
                        "pairing": PAIRING_NAMES[pairing],
                        "contrast": "H5 minus XGBoost"
                        if "xgboost" in contrast
                        else "H5 minus the plant",
                        "value": 100 * row["difference_gbp"] / scale,
                        "low": 100 * row["low95_gbp"] / scale,
                        "high": 100 * row["high95_gbp"] / scale,
                    }
                )
    frame = pl.DataFrame(rows)
    panels = []
    for pairing_name in PAIRING_NAMES.values():
        part = frame.filter(pl.col("pairing") == pairing_name)
        base = alt.Chart(part).encode(
            y=alt.Y("limit:N", sort=list(KIND_FILES), title=None),
            yOffset="contrast:N",
            color=alt.Color(
                "contrast:N",
                scale=alt.Scale(
                    domain=["H5 minus XGBoost", "H5 minus the plant"],
                    range=[ocf.DATA_PURPLE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=70, opacity=1, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "value:Q",
                title="Change in avoidable cost (% of XGBoost's; negative is better)",
                scale=alt.Scale(domain=[-40, 10]),
            )
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3])
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            alt.layer(zero, rule, dot).properties(
                width=WIDTH - 140,
                height=110,
                title=alt.TitleParams(pairing_name, anchor="start", fontSize=12),
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="The neural head lowers the avoidable cost by 1% to 10% against XGBoost",
            lines=[
                (
                    "Difference in avoidable cost as a share of XGBoost's avoidable cost. Dot: estimate."
                    " Line: 95% interval from resampling whole months. Dashed line: no difference."
                )
            ],
        )
    )


def main() -> None:
    """Write the four figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1_event_census.svg": figure_1_census(),
        "figure_2_avoidable_cost.svg": figure_2_headline(),
        "figure_3_by_node.svg": figure_3_nodes(),
        "figure_4_neural_head.svg": figure_4_contrast(),
    }
    for name, chart in figures.items():
        chart.save(DOCS_DIR / name)
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
