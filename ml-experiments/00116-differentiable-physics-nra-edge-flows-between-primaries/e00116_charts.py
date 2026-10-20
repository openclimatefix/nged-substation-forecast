"""Draw the figures of experiment 116's write-up from the stored result files.

Every figure carries its own scope, unit, and direction in its title and subtitle. A substation is
drawn by its D-label with dates; the two series labelled G-c never appear in a figure. SVGs go to
the write-up's folder and PNGs to a scratch folder for the render check.

    uv run python ml-experiments/00116-*/e00116_charts.py [--only name ...]
"""

import argparse
import json
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Final

import altair as alt
import e00116_common as common
import e00116_data as data_module
import e00116_flows as flows_module
import e00116_inject as inject
import e00116_local as local
import e00116_tune as tune
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


def zero_rule() -> alt.Chart:
    """Return a dashed vertical rule at zero."""
    return (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )


def figure_tuning() -> alt.TopLevelMixin:
    """Recall against placebo flow for every configuration tried on the tuning seeds."""
    rows = []
    for result in tune.read_results():
        config = result["config"]
        rows.append(
            {
                "form": "3-hour blocks (A2-b)" if config["block"] > 1 else "half-hourly (A2)",
                "recall": result["recall"],
                "placebo": max(result["placebo_mw_days"].values()),
                "label": f"c {config['step_c']:g}, l {config['level_c']:g}",
            }
        )
    table = pl.DataFrame(rows)
    chosen = read_json(name="decisions.json")["chosen"]
    points = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X(
                "placebo:Q",
                title=[
                    "Flow on the worse of two pairs with no interconnection",
                    "(MW-days per year; smaller is better)",
                ],
                scale=alt.Scale(type="symlog"),
            ),
            y=alt.Y(
                "recall:Q", title="MW-weighted recall of injected transfers (higher is better)"
            ),
            color=alt.Color(
                "form:N",
                scale=alt.Scale(
                    domain=["half-hourly (A2)", "3-hour blocks (A2-b)"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
    )
    limit = (
        alt.Chart(pl.DataFrame({"x": [tune.PLACEBO_LIMIT_MW_DAYS]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=DARK, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (limit + points).properties(
        width=WIDTH - 60,
        height=240,
        title=title(
            text="Stricter penalties cut false flows and lose short transfers",
            subtitle=(
                "Each dot is one configuration of the step and level penalties, scored on the 400 "
                "injected transfers of the tuning seeds (recall) and on the tuning year's two "
                "placebo pairs (flow). Dashed line: the 0.05 MW-days limit a chosen configuration "
                f"must stay under. Chosen: step constant {chosen['config']['step_c']:g}, level "
                f"constant {chosen['config']['level_c']:g}, block {chosen['config']['block']}."
            ),
        ),
    )


def figure_recall_cells() -> alt.TopLevelMixin:
    """Recall by transfer size and duration for the chosen configuration on the tuning seeds."""
    chosen = read_json(name="decisions.json")["chosen"]
    table = pl.DataFrame(chosen["recall_by_cell"]).with_columns(
        label=pl.col("recall").round(2).cast(pl.String)
    )
    base = alt.Chart(table).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "duration:N",
            sort=list(inject.DURATION_BANDS_HOURS),
            title="Duration of the injected transfer",
        ),
        y=alt.Y(
            "size:N", sort=list(inject.SIZE_LEVELS), title="Size (MW, or share of receiving load)"
        ),
    )
    cells = base.mark_rect().encode(
        color=alt.Color(
            "recall:Q",
            scale=alt.Scale(domain=[0, 1], range=["#FFFFFF", ocf.DATA_BLUE]),
            legend=alt.Legend(title="Recall", orient="top", direction="horizontal"),
        )
    )
    text = base.mark_text(fontSize=11).encode(text="label:N")
    return (cells + text).properties(
        width=WIDTH - 140,
        height=170,
        title=title(
            text="Long, large transfers are found; short or small ones are not",
            subtitle=(
                "MW-weighted recall of the chosen configuration, by the size and duration of the "
                "injected transfer, on the 400 tuning-seed transfers. 1 means every injected "
                "megawatt-hour was found on the right edge with the right sign."
            ),
        ),
    )


def example_transfers() -> list[dict]:
    """Choose two gate transfers by identifier, before any score is read."""
    cache = common.load_cache()
    table = inject.draw_set(set_name="G", year_name="tuning", cache=cache)
    wanted = [("25%", "over 1 day"), ("25%", "6-24h")]
    return [
        table.filter(pl.col("size") == size, pl.col("duration") == duration)
        .sort("transfer_id")
        .row(0, named=True)
        for size, duration in wanted
    ]


def figure_examples() -> alt.TopLevelMixin:
    """Two injected transfers: the true flow against the flow the hindsight fit finds."""
    cache = common.load_cache()
    config = flows_module.chosen_config()
    data = data_module.load_year(cache=cache, cut_name="tuning", arm="a0")
    paths = data_module.build_graph_paths(
        data=data, edges=data_module.decision_graph(data=data), config=config
    )
    panels = []
    for transfer in example_transfers():
        window = local.fit_local(
            cache=cache, data=data, paths=paths, config=config, transfer=transfer
        )
        hours = (np.arange(window.stop - window.first) - window.onset) / 2.0
        table = pl.DataFrame(
            {
                "hours": np.concatenate([hours, hours]),
                "flow": np.concatenate([window.true_flow, window.fitted_edge()]),
                "series": ["Injected (true)"] * len(hours)
                + ["Fitted from the residuals"] * len(hours),
            }
        )
        edge = f"{transfer['node_a']}–{transfer['node_b']}"
        panels.append(
            alt.Chart(table)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=2, interpolate="step-after")
            .encode(
                x=alt.X("hours:Q", title="Hours since the transfer began"),
                y=alt.Y("flow:Q", title="Flow onto the first-named substation (MW)"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=["Injected (true)", "Fitted from the residuals"],
                        range=[DARK, ocf.BRAND_ORANGE],
                    ),
                    legend=alt.Legend(title=None, orient="top", direction="horizontal"),
                ),
            )
            .properties(
                width=WIDTH - 60,
                height=150,
                title=(
                    f"Edge {edge}: {transfer['duration']} transfer, size {transfer['size']} "
                    "of the receiving load"
                ),
            )
        )
    return alt.vconcat(*panels).properties(
        title=title(
            text="Two injected transfers and the flows the fit finds",
            subtitle=(
                "Tuning year, hindsight fit of the paths at the two ends. The two examples are the "
                "first transfers by identifier of the 25% size with a duration over 1 day and "
                "6 to 24 hours; they were chosen before any score was read."
            ),
        )
    )


FIGURES: Final[dict[str, Callable[[], alt.TopLevelMixin]]] = {
    "tuning": figure_tuning,
    "recall_cells": figure_recall_cells,
    "examples": figure_examples,
}


def main() -> None:
    """Draw every figure, or those named."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=list(FIGURES))
    for name in parser.parse_args().only:
        save(FIGURES[name](), name=name)


if __name__ == "__main__":
    main()
