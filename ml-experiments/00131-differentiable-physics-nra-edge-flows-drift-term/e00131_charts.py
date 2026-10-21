"""Draw the figures of experiment 131's write-up from the stored result files.

Every figure carries its scope, unit, and direction in its title and subtitle. Substations are drawn
by D-label only, with no dates. SVGs go to the write-up's folder and PNGs to a scratch folder for
the render check.

    uv run python ml-experiments/00131-*/e00131_charts.py [--only name ...]
"""

import argparse
import json
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Final

import altair as alt
import e00131_common as common
import e00131_data as data_module
import e00131_inject as inject
import e00131_local as local
import e00131_threshold as threshold
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
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
GREY: Final[str] = "#808080"
DARK: Final[str] = "#333333"
GROUP_LABEL: Final[dict[str, str]] = {"drift": "with a drift term", "none": "no drift term"}
GROUP_COLOURS: Final[list[str]] = [ocf.BRAND_ORANGE, ocf.DATA_BLUE]


def title(*, text: str, subtitle: str) -> alt.TitleParams:
    """Return a left-aligned title with a wrapped subtitle."""
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def save(chart: alt.TopLevelMixin, *, name: str) -> None:
    """Write a figure as SVG beside the write-up and as PNG for the render check."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def read_json(*, name: str) -> dict:
    """Read a result file."""
    return json.loads((common.WORK_DIR / name).read_text())


def rule(*, value: float, axis: str, dash: bool = True) -> alt.Chart:
    """Return a dashed reference rule at a value on x or y."""
    mark = alt.Chart(pl.DataFrame({"v": [value]})).mark_rule(
        color=DARK, strokeDash=[4, 3] if dash else []
    )
    return mark.encode(**{axis: "v:Q"})  # ty: ignore[unresolved-attribute]


def colour_encoding(*, field: str = "arm") -> alt.Color:
    """Return the colour encoding of the two arms."""
    return alt.Color(
        f"{field}:N",
        scale=alt.Scale(domain=list(GROUP_LABEL.values()), range=GROUP_COLOURS),
        legend=alt.Legend(title=None, orient="top", direction="horizontal"),
    )


def screen_table() -> pl.DataFrame:
    """Return every stage-1 setting with its placebo flow and recall, where recall was read."""
    rows = []
    for path in sorted((common.WORK_DIR / "tune").glob("*.json")):
        result = json.loads(path.read_text())
        group = "none" if result["config"]["drift_q"] is None else "drift"
        recall = result.get("recall")
        rows.append(
            {
                "arm": GROUP_LABEL[group],
                "placebo": max(result["placebo"].values()),
                "recall_3_24h": None if recall is None else recall["3-24h"],
                "recall_1_3d": None if recall is None else recall["1-3 days"],
            }
        )
    return pl.DataFrame(rows)


def figure_tuning() -> alt.TopLevelMixin:
    """Recall against placebo flow for every setting of the tuning screen, two panels."""
    table = screen_table()
    panels = []
    for column, name, target in (
        ("recall_3_24h", "3 to 24 hours", 0.5),
        ("recall_1_3d", "24 to 72 hours", 0.77),
    ):
        points = (
            alt.Chart(table.filter(pl.col(column).is_not_null()))  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=50, opacity=0.8)
            .encode(
                x=alt.X(
                    "placebo:Q",
                    title="Flow on the worse unconnected pair (MW-days a year; smaller is better)",
                    scale=alt.Scale(type="symlog", domain=[0, 30]),
                    axis=alt.Axis(values=[0, 0.05, 1, 3, 10, 30]),
                ),
                y=alt.Y(
                    f"{column}:Q",
                    title=f"Recall, {name} (higher is better)",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=colour_encoding(),
            )
        )
        panels.append(
            (rule(value=0.05, axis="x") + rule(value=target, axis="y") + points).properties(
                width=WIDTH - 60, height=170
            )
        )
    return alt.vconcat(*panels).properties(
        title=title(
            text="A drift term cuts the false flow on unconnected pairs but not to the limit",
            subtitle=(
                "Each dot is one setting of the step penalty, level penalty, and drift variance, "
                "scored on the big injected transfers of the tuning year (recall) and on the "
                "tuning year's two unconnected pairs (flow). Dashed lines: the 0.05 MW-days "
                "limit and the recall target. Recall was read for the first 24 settings, before "
                "the screen skipped it for settings over the flow limit, and for every setting "
                "under the limit."
            ),
        )
    )


def figure_gate() -> alt.TopLevelMixin:
    """The gate's recall and placebo readings in the tuning and replicate years."""
    rows = []
    gate = read_json(name="gate.json")["readings"]
    for group, years in gate.items():
        for year, readings in years.items():
            rows.append(
                {
                    "arm": GROUP_LABEL[group],
                    "year": year,
                    "recall_3_24h": readings["2b_recall_3_to_24_h"]["value"],
                    "recall_1_3d": readings["2a_recall_24_to_72_h"]["value"],
                    "ratio": readings["1_moved_mw_ratio"]["value"],
                    "placebo": readings["3_placebo_mw_days"]["worst"],
                }
            )
    table = pl.DataFrame(rows)
    panels = []
    for column, text, target, domain in (
        ("recall_3_24h", "Recall, transfers of 3 to 24 hours (target 0.5)", 0.5, [0, 1]),
        ("recall_1_3d", "Recall, transfers of 24 to 72 hours (target 0.77)", 0.77, [0, 1]),
        ("ratio", "Moved-MW ratio, median (target 0.8 to 1.25)", 0.8, [0, 1.4]),
        ("placebo", "Flow on unconnected pairs, MW-days a year (limit 0.05)", 0.05, None),
    ):
        x = alt.X(
            f"{column}:Q",
            title=text,
            scale=alt.Scale(domain=domain) if domain else alt.Scale(type="symlog"),
            axis=alt.Axis(values=[0, 0.05, 0.5, 1, 3, 10]) if not domain else alt.Axis(),
        )
        bars = (
            alt.Chart(table)  # ty: ignore[unresolved-attribute]
            .mark_bar()
            .encode(
                y=alt.Y("year:N", title=None, sort=["tuning", "replicate"]),
                x=x,
                color=colour_encoding(),
                yOffset="arm:N",
            )
        )
        layers = rule(value=target, axis="x") + bars
        if column == "ratio":
            layers = rule(value=1.25, axis="x") + layers
        panels.append(layers.properties(width=WIDTH - 60, height=70))
    return alt.vconcat(*panels).properties(
        title=title(
            text="The drift setting misses every recall target and leaks flow out of sample",
            subtitle=(
                "Gate readings on 182 (tuning year) and 173 (replicate year) "
                "big injected transfers, with the settings frozen after tuning. Dashed lines: "
                "the gate's target for each reading, which a bar must reach (recall, ratio) or "
                "stay under (flow on unconnected pairs)."
            ),
        )
    )


def figure_recall_by_length() -> alt.TopLevelMixin:
    """Per-transfer recall against transfer length for the chosen setting, both years."""
    rows = pl.read_parquet(common.WORK_DIR / "gate_rows.parquet").filter(pl.col("set") == "G")
    rows = rows.with_columns(
        arm=pl.col("group").replace_strict(GROUP_LABEL), hours=pl.col("hours").clip(lower_bound=1)
    )
    points = (
        alt.Chart(rows)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=25, opacity=0.5)
        .encode(
            x=alt.X(
                "hours:Q",
                title="Transfer length (hours, log scale)",
                scale=alt.Scale(type="log", domain=[3, 72]),
                axis=alt.Axis(values=[3, 6, 12, 24, 48, 72]),
            ),
            y=alt.Y(
                "recall:Q",
                title="Recall of the transfer (higher is better)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=colour_encoding(),
        )
        .properties(width=WIDTH - 60, height=150)
    )
    return alt.vconcat(
        points.transform_filter(alt.datum.year == "tuning").properties(title="Tuning year"),
        points.transform_filter(alt.datum.year == "replicate").properties(title="Replicate year"),
    ).properties(
        title=title(
            text="Short transfers are found less often than long ones",
            subtitle=(
                "Each dot is one big injected transfer of 25% or 40% of the receiving load, "
                "fitted on its own window. The setting without a drift term is the one the "
                "same tuning rule chose, which has to be very stiff to keep unconnected pairs "
                "clean and so finds almost nothing."
            ),
        )
    )


def figure_long_arm() -> alt.TopLevelMixin:
    """Recall of 7 to 30 day transfers, with and without a drift term."""
    rows = pl.read_parquet(common.WORK_DIR / "gate_rows.parquet").filter(pl.col("set") == "L")
    rows = rows.with_columns(
        arm=pl.col("group").replace_strict(GROUP_LABEL), days=pl.col("hours") / 24.0
    )
    points = (
        alt.Chart(rows)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=25, opacity=0.5)
        .encode(
            x=alt.X(
                "days:Q",
                title="Transfer length (days, log scale)",
                scale=alt.Scale(type="log", domain=[7, 30]),
                axis=alt.Axis(values=[7, 10, 14, 21, 30]),
            ),
            y=alt.Y(
                "recall:Q",
                title="Recall of the transfer (higher is better)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=colour_encoding(),
        )
        .properties(width=WIDTH - 60, height=150)
    )
    return points.properties(
        title=title(
            text="Transfers of 7 to 30 days are found only in part",
            subtitle=(
                "Each dot is one injected transfer of 25% or 40% of the receiving load, both "
                "years pooled. This arm is descriptive and carries no verdict."
            ),
        )
    )


def figure_examples() -> alt.TopLevelMixin:
    """Two injected transfers with the true flow, the fitted flow, and the fitted drifts."""
    cache = common.load_cache()
    config = data_module.FlowConfig(**read_json(name="decisions.json")["drift"]["config"])
    data = data_module.load_year(cache=cache, cut_name="replicate", arm="a0")
    limit = threshold.thresholds(
        cache=cache, data=data_module.load_year(cache=cache, cut_name="tuning", arm="a0")
    )
    edges = data_module.decision_graph(data=data)
    paths = data_module.build_graph_paths(data=data, edges=edges, config=config)
    transfers = inject.draw_set(set_name="G", year_name="replicate", cache=cache)
    panels = []
    for edge in common.GATE_EDGES:
        node_a, node_b = sorted(edge)
        candidates = [
            row
            for row in transfers.iter_rows(named=True)
            if (row["node_a"], row["node_b"]) == (node_a, node_b)
            and row["end"] - row["onset"] >= 48
            and threshold.is_big(transfer=row, cache=cache, data=data, limit=limit)
        ]
        transfer = min(candidates, key=lambda row: row["transfer_id"])
        window = local.fit_local(
            cache=cache, data=data, paths=paths, config=config, transfer=transfer
        )
        days = (np.arange(window.stop - window.first) - window.onset) / 48.0
        table = pl.concat(
            [
                pl.DataFrame({"day": days, "mw": window.true_flow, "series": "true flow"}),
                pl.DataFrame({"day": days, "mw": window.fitted_edge(), "series": "fitted flow"}),
                pl.DataFrame(
                    {
                        "day": days,
                        "mw": window.fit.drifts[node_a],
                        "series": "drift of the first one",
                    }
                ),
                pl.DataFrame(
                    {
                        "day": days,
                        "mw": window.fit.drifts[node_b],
                        "series": "drift of the second one",
                    }
                ),
            ]
        )
        domain = [
            "true flow",
            "fitted flow",
            "drift of the first one",
            "drift of the second one",
        ]
        panels.append(
            alt.Chart(table)  # ty: ignore[unresolved-attribute]
            .mark_line()
            .encode(
                x=alt.X("day:Q", title="Days from the transfer's start"),
                y=alt.Y(
                    "mw:Q",
                    title=f"Flow onto {node_a} from {node_b}, or drift (MW)",
                ),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=domain,
                        range=[DARK, ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY],
                    ),
                    legend=alt.Legend(title=None, orient="top", direction="horizontal"),
                ),
                strokeDash=alt.StrokeDash(
                    "series:N",
                    scale=alt.Scale(domain=domain, range=[[1, 0], [1, 0], [1, 0], [4, 3]]),
                    legend=None,
                ),
            )
            .properties(
                width=WIDTH - 60, height=150, title=f"{node_a} (first) and {node_b} (second)"
            )
        )
    return alt.vconcat(*panels).properties(
        title=title(
            text="The fit recovers part of a transfer, and the drift stays small",
            subtitle=(
                "Two injected transfers of the replicate year, the first by identifier on each "
                "gate edge that lasts 24 hours or more and is big. The window runs from 7 days "
                "before the transfer to 7 days after its 48 post-event hours. Positive flow "
                "moves load onto the first named substation."
            ),
        )
    )


FIGURES: Final[dict[str, Callable[[], alt.TopLevelMixin]]] = {
    "tuning": figure_tuning,
    "gate": figure_gate,
    "recall_by_length": figure_recall_by_length,
    "long_arm": figure_long_arm,
    "examples": figure_examples,
}


def main() -> None:
    """Draw the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=tuple(FIGURES))
    arguments = parser.parse_args()
    for name in arguments.only or FIGURES:
        save(FIGURES[name](), name=name)


if __name__ == "__main__":
    main()
