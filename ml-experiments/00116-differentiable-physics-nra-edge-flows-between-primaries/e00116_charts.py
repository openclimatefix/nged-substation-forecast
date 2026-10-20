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
    base = alt.Chart(table).encode(
        x=alt.X(
            "duration:N",
            sort=list(inject.DURATION_BANDS_HOURS),
            title="Duration of the injected transfer",
        ),
        y=alt.Y(
            "size:N", sort=list(inject.SIZE_LEVELS), title="Size (MW, or share of receiving load)"
        ),
    )
    cells = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "recall:Q",
            scale=alt.Scale(domain=[0, 1], range=["#FFFFFF", ocf.DATA_BLUE]),
            legend=alt.Legend(title="Recall", orient="top", direction="horizontal"),
        )
    )
    text = base.mark_text(fontSize=11).encode(text="label:N")  # ty: ignore[unresolved-attribute]
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


def figure_variants() -> alt.TopLevelMixin:
    """Recall over one day against placebo flow, for the registered setting and the variants."""
    variants = read_json(name="variants.json")["post_hoc_variants"]
    gate = read_json(name="gate_tuning_only_first_run.json")
    rows = [
        {
            "label": "Registered (step 300, level 0, Huber 1.5)",
            "recall": gate["3a_recall_over_1_day"]["value"],
            "placebo": 0.0,
            "kind": "registered",
        }
    ]
    names = {
        "huber10": "Post hoc: registered penalties, Huber 10",
        "loose": "Post hoc: step 10, level 0.3, Huber 1.5",
        "loose_huber10": "Post hoc: step 10, level 0.3, Huber 10",
        "loosest_huber10": "Post hoc: step 3, level 0, Huber 10",
    }
    rows.extend(
        {
            "label": names[entry["variant"]],
            "recall": entry["recall_over_1_day"],
            "placebo": max(entry["placebo_mw_days"].values()),
            "kind": "post hoc",
        }
        for entry in variants
    )
    table = pl.DataFrame(rows)
    order = table["label"].to_list()
    points = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=90)
        .encode(
            x=alt.X(
                "placebo:Q",
                title=["Flow on the worse unconnected pair (MW-days per year; smaller is better)"],
                scale=alt.Scale(type="symlog", domain=[0, 50]),
                axis=alt.Axis(values=[0, 0.05, 1, 5, 20, 50]),
            ),
            y=alt.Y(
                "recall:Q",
                title=["Recall of transfers over 1 day (higher is better)"],
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "label:N",
                sort=order,
                scale=alt.Scale(
                    domain=order,
                    range=[DARK, ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=380),
            ),
        )
    )
    recall_line = (
        alt.Chart(pl.DataFrame({"y": [0.77]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(y="y:Q")
    )
    placebo_line = (
        alt.Chart(pl.DataFrame({"x": [0.05]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (recall_line + placebo_line + points).properties(
        width=WIDTH - 60,
        height=260,
        title=title(
            text="Finding real transfers and finding false ones go together",
            subtitle=(
                "Tuning year, 100 of the gate's transfers of 25% or 40% of the receiving load "
                "(recall; the registered dot reads all 200 such transfers) and the two placebo "
                "pairs with no interconnection (flow). Dashed "
                "lines: the gate's thresholds, recall 0.77 and flow 0.05 MW-days. Variants were "
                "chosen after the registered gate failed and cannot change the verdict."
            ),
        ),
    )


def figure_fleet() -> alt.TopLevelMixin:
    """Megawatt-days of flow on each edge of the decision graph, mean of two years."""
    rows = []
    for year in ("replicate", "scored"):
        flows = flows_module.read_flows(year_name=year)
        for name, flow in flows.items():
            if name == "position" or name.startswith("outside"):
                continue
            rows.append({"pair": name, "year": year, "mw_days": float(np.sum(np.abs(flow)) / 48.0)})
    table = pl.DataFrame(rows).group_by("pair").agg(pl.col("mw_days").mean())
    order = table.sort("mw_days", descending=True)["pair"].to_list()
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            y=alt.Y("pair:N", sort=order, title=None),
            x=alt.X(
                "mw_days:Q", title="Flow the registered fit puts on the edge (MW-days per year)"
            ),
        )
        .properties(
            width=WIDTH - 100,
            height=14 * len(order),
            title=title(
                text="Where the registered fit puts flow",
                subtitle=(
                    "Mean over the replicate and scored years of the megawatt-days of flow on each "
                    "edge of the decision graph, with the registered penalties. The placebo pairs "
                    "carry about zero in this fit."
                ),
            ),
        )
    )


WINTER_WEEK: Final[tuple[str, str]] = ("2025-12-01", "2025-12-08")
SUMMER_WEEK: Final[tuple[str, str]] = ("2026-06-01", "2026-06-08")
"""The two weeks every card shows. Fixed before any score was read: the first full weeks of
December 2025 and June 2026, Monday to Sunday."""


def _week_frame(*, site: str, week: tuple[str, str], arm: str = "a4") -> pl.DataFrame:
    """Return the day-ahead forecast rows of a site over a week, in megawatts."""
    losses = pl.read_parquet(common.WORK_DIR / "c2_losses.parquet").filter(
        pl.col("arm") == arm, pl.col("site") == site, pl.col("cut") == "scored"
    )
    start = pl.lit(week[0]).str.to_datetime(time_zone="UTC")
    stop = pl.lit(week[1]).str.to_datetime(time_zone="UTC")
    return losses.filter(pl.col("time") > start, pl.col("time") <= stop)


def _forecast_panel(
    *, site: str, week: tuple[str, str], label: str, scale: float
) -> alt.TopLevelMixin:
    """Draw one week of observed power against the A4 forecast and its 80% and 90% bands."""
    frame = _week_frame(site=site, week=week).with_columns(
        pl.col(["actual", "mean", "q05", "q10", "q90", "q95"]) * scale
    )
    base = alt.Chart(frame).encode(x=alt.X("time:T", title=None, axis=alt.Axis(format="%a %d %b")))
    band90 = base.mark_area(opacity=0.18, color=ocf.DATA_BLUE).encode(y="q05:Q", y2="q95:Q")  # ty: ignore[unresolved-attribute]
    band80 = base.mark_area(opacity=0.30, color=ocf.DATA_BLUE).encode(y="q10:Q", y2="q90:Q")  # ty: ignore[unresolved-attribute]
    forecast = base.mark_line(color=ocf.DATA_BLUE, strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("mean:Q", title="Power (MW)", scale=alt.Scale(zero=False))
    )
    observed = base.mark_line(color=DARK, strokeWidth=1.2).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
    event_rows = frame.filter(pl.col("in_event")).with_columns(
        start=pl.col("time") - pl.duration(minutes=30)
    )
    layers = [band90, band80, forecast, observed]
    if event_rows.height:
        layers.insert(
            0,
            alt.Chart(event_rows)  # ty: ignore[unresolved-attribute]
            .mark_rect(color=ocf.DATA_AMBER, opacity=0.15)
            .encode(x="start:T", x2="time:T"),
        )
    return alt.layer(*layers).properties(width=WIDTH - 60, height=120, title=label)


def _flow_panel(
    *, site: str, week: tuple[str, str], label: str, cache_end: np.ndarray
) -> alt.Chart:
    """Draw one week of the hindsight edge flows at a site, and its outside path."""
    flows = flows_module.read_flows(year_name="scored")
    positions = flows["position"]
    times = cache_end[positions]
    start, stop = np.datetime64(week[0], "us"), np.datetime64(week[1], "us")
    chosen = (times > start) & (times <= stop)
    edges = [
        name for name in flows if "-" in name and site in name.split("-") and "outside" not in name
    ]
    parts = []
    for name in edges:
        first, second = name.split("-")
        sign = 1.0 if first == site else -1.0
        parts.append(
            pl.DataFrame(
                {
                    "time": times[chosen],
                    "flow": sign * flows[name][chosen],
                    "path": f"edge to {second if first == site else first}",
                }
            )
        )
    parts.append(
        pl.DataFrame(
            {
                "time": times[chosen],
                "flow": flows[f"outside:{site}"][chosen],
                "path": "outside path",
            }
        )
    )
    table = pl.concat(parts).with_columns(pl.col("time").dt.replace_time_zone("UTC"))
    if float(np.max(np.abs(table["flow"].to_numpy()))) < 0.01:
        label = f"{label}: no flow above 0.01 MW on any path"
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(interpolate="step-after", strokeWidth=1.4)
        .encode(
            x=alt.X("time:T", title=None, axis=alt.Axis(format="%a %d %b")),
            y=alt.Y("flow:Q", title="Flow onto the substation (MW)"),
            color=alt.Color(
                "path:N",
                scale=alt.Scale(range=list(ocf.DATA_COLOURS)),
                legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=4),
            ),
        )
        .properties(width=WIDTH - 60, height=90, title=label)
    )


def figure_card(*, site: str) -> alt.TopLevelMixin:
    """Draw a substation's network card (provisional): forecasts and flows in two weeks."""
    cache = common.load_cache()
    scale = cache.sites[site].table.effective_capacity_mw / 100.0
    return alt.vconcat(
        _forecast_panel(  # ty: ignore[invalid-argument-type]
            site=site, week=WINTER_WEEK, label="Winter week, 1 to 7 December 2025", scale=scale
        ),
        _forecast_panel(  # ty: ignore[invalid-argument-type]
            site=site, week=SUMMER_WEEK, label="Summer week, 1 to 7 June 2026", scale=scale
        ),
        _flow_panel(
            site=site,
            week=WINTER_WEEK,
            label="Fitted flows, winter week",
            cache_end=cache.end_times,
        ),
        _flow_panel(
            site=site,
            week=SUMMER_WEEK,
            label="Fitted flows, summer week",
            cache_end=cache.end_times,
        ),
        _cold_snap_panel(site=site),
    ).properties(
        title=title(
            text=f"Network card for {site} (provisional: past weather, screening run)",
            subtitle=(
                "Top: day-ahead forecast of the model trained on power with the edge flows taken "
                "out (line), its 80% and 90% bands, and the observed power (dark line); amber: "
                "hours inside a hand-filtered switching event. Bottom: the registered fit's "
                "hindsight flows on the substation's edges and its outside path. Weeks fixed "
                "before any score was read."
            ),
        )
    )


def _forest(
    *, table: pl.DataFrame, x_title: str, chart_title: str, zero_at: float = 0.0
) -> alt.Chart:
    """Draw dots with interval lines for a table of label, estimate, lower, upper, and colour."""
    order = table["label"].to_list()
    base = alt.Chart(table).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=300))
    )
    rule = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q", title=x_title), x2="upper:Q", color=alt.Color("group:N", legend=None)
    )
    dot = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x="estimate:Q", color=alt.Color("group:N", legend=None)
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [zero_at]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (zero + rule + dot).properties(
        width=WIDTH - 250, height=22 * len(order), title=chart_title
    )


def figure_headline() -> alt.TopLevelMixin:
    """The three planned contrasts, with winter breakdowns, as dots with 98.33% intervals."""
    c1 = decide_summary_c1()
    c2 = read_json(name="c2.json")
    c3 = read_json(name="c3.json")
    panel_1 = _forest(
        table=pl.DataFrame(
            [
                {
                    "label": "share rebuilt (needs 0.5)",
                    "estimate": c1["estimate"],
                    "lower": c1["lower"],
                    "upper": c1["upper"],
                    "group": "c1",
                }
            ]
        ),
        x_title="Share of the injected megawatts that the edge flows rebuild (higher is better)",
        chart_title="C1: does the fit rebuild injected transfers?",
        zero_at=0.5,
    )
    rows_2 = []
    for year in ("scored", "replicate"):
        for scope, name in (
            ("outside_events", "all rows outside events"),
            ("winter", "winter"),
            ("winter_cold_snap", "winter cold-snap days"),
            ("winter_evening_peak", "winter evening peaks"),
        ):
            entry = c2[f"{year}:{scope}"]["a4_minus_a0"]
            rows_2.append(
                {
                    "label": f"{year}, {name}",
                    "estimate": entry["difference"],
                    "lower": entry["lower"],
                    "upper": entry["upper"],
                    "group": year,
                }
            )
    panel_2 = _forest(
        table=pl.DataFrame(rows_2),
        x_title="A4 minus A0, quantile score (points; negative is better; margin +0.05)",
        chart_title="C2: does removing flows from training do harm outside events?",
    )
    rows_3 = []
    for year in ("scored", "replicate"):
        entry = c3[year]["contrasts"]["a5_minus_a4"]
        rows_3.append(
            {
                "label": f"{year}, all injected rows",
                "estimate": entry["estimate"],
                "lower": entry["lower"],
                "upper": entry["upper"],
                "group": year,
            }
        )
        winter = c3[year]["winter_scopes"]["winter"]["a5_minus_a4"]
        if winter is not None:
            rows_3.append(
                {
                    "label": f"{year}, winter rows",
                    "estimate": winter["estimate"],
                    "lower": winter["lower"],
                    "upper": winter["upper"],
                    "group": year,
                }
            )
    panel_3 = _forest(
        table=pl.DataFrame(rows_3),
        x_title="A5 minus A4, quantile score (points; negative is better; needs -0.5 or lower)",
        chart_title="C3: do causal flows help the forecast during and after a transfer?",
    )
    return alt.vconcat(panel_1, panel_2, panel_3).properties(
        title=title(
            text="No contrast shows a gain from fitted flows",
            subtitle=(
                "Dots: estimates; lines: 98.33% intervals from resampling whole calendar months "
                "(C2 and C3 over the decision substations D5, D10, D11, D14, D16 and the four gate "
                "sites; C1 over injected transfers). Dashed line: no difference (C1: the 0.5 "
                "threshold). Registered configuration; C3 on a 100-transfer sample per year."
            ),
        )
    )


def decide_summary_c1() -> dict:
    """Return the C1 summary from the stored seeds."""
    import e00116_decide as decide

    return decide.c1_summary()


def _cold_snap_panel(*, site: str) -> alt.Chart:
    """Draw A4 minus A0 on the site's cold-snap days (winter days with a daily mean under 2 C)."""
    losses = pl.read_parquet(common.WORK_DIR / "c2_losses.parquet").filter(
        pl.col("site") == site, pl.col("cut") == "scored", pl.col("cold_snap"), ~pl.col("in_event")
    )
    days = losses["time"].dt.date().n_unique()
    scores = {
        arm: float(np.mean(losses.filter(pl.col("arm") == arm)["score"].to_numpy()))
        for arm in ("a0", "a4")
    }
    too_few = days < common.MIN_COLD_SNAP_DAYS
    label = (
        f"Cold snap: {days} days (under the {common.MIN_COLD_SNAP_DAYS} needed: too few days)"
        if too_few
        else f"Cold snap: {days} days"
    )
    table = pl.DataFrame(
        {
            "arm": ["A0 (trained on every row)", "A4 (trained with edge flows removed)"],
            "score": [scores["a0"], scores["a4"]],
        }
    )
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X(
                "score:Q",
                title="Day-ahead quantile score on cold-snap days (points; smaller is better)",
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(table["arm"]), range=[ocf.DATA_BLUE_LIGHT, ocf.BRAND_ORANGE]
                ),
                legend=None,
            ),
        )
        .properties(width=WIDTH - 200, height=60, title=label)
    )


def figure_transfer_weeks() -> alt.TopLevelMixin:
    """Forecasts around one injected transfer at each gate site."""
    cache = common.load_cache()
    rows = pl.read_parquet(common.WORK_DIR / "c3_rows_scored.parquet")
    table = inject.draw_set(set_name="C3", year_name="scored", cache=cache)
    present = set(rows["transfer_id"].unique().to_list())
    panels = []
    for site in common.GATE_SITES:
        chosen = (
            table.filter(
                pl.col("transfer_id").is_in(present),
                pl.col("receiver") == site,
                (pl.col("end") - pl.col("onset")) >= 48,
            )
            .sort("transfer_id")
            .row(0, named=True)
        )
        part = rows.filter(
            pl.col("transfer_id") == chosen["transfer_id"],
            pl.col("node") == site,
            pl.col("lead_hours") >= 12.0,
            pl.col("lead_hours") < 18.0,
        )
        scale = cache.sites[site].table.effective_capacity_mw / 100.0
        frames = []
        for arm, name in (("a4", "A4 plain filter"), ("a5", "A5 causal flows"), ("or", "Oracle")):
            arm_rows = part.filter(pl.col("arm") == arm).sort("target")
            frames.append(
                pl.DataFrame(
                    {
                        "time": cache.end_times[arm_rows["target"].to_numpy()],
                        "mw": arm_rows["mean"].to_numpy() * scale,
                        "low": arm_rows["q10"].to_numpy() * scale,
                        "high": arm_rows["q90"].to_numpy() * scale,
                        "series": name,
                    }
                )
            )
        observed = part.filter(pl.col("arm") == "a4").sort("target")
        frames.append(
            pl.DataFrame(
                {
                    "time": cache.end_times[observed["target"].to_numpy()],
                    "mw": observed["actual"].to_numpy() * scale,
                    "low": observed["actual"].to_numpy() * scale,
                    "high": observed["actual"].to_numpy() * scale,
                    "series": "Observed (with the injected transfer)",
                }
            )
        )
        data = pl.concat(frames).with_columns(pl.col("time").dt.replace_time_zone("UTC"))
        domain = [
            "Observed (with the injected transfer)",
            "A4 plain filter",
            "A5 causal flows",
            "Oracle",
        ]
        lines = (
            alt.Chart(data)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1.4)
            .encode(
                x=alt.X("time:T", title=None, axis=alt.Axis(format="%d %b")),
                y=alt.Y("mw:Q", title="Power (MW)", scale=alt.Scale(zero=False)),
                strokeDash=alt.StrokeDash(
                    "series:N",
                    scale=alt.Scale(domain=domain, range=[[1, 0], [1, 0], [5, 3], [1, 0]]),
                    legend=None,
                ),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=domain, range=[DARK, ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]
                    ),
                    legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=2),
                ),
            )
        )
        band = (
            alt.Chart(data.filter(pl.col("series") == "A4 plain filter"))  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.2, color=ocf.DATA_BLUE)
            .encode(x="time:T", y="low:Q", y2="high:Q")
        )
        panels.append(
            (band + lines).properties(
                width=WIDTH - 60,
                height=110,
                title=(
                    f"{site}: injected transfer of {chosen['size_mw']:.1f} MW "
                    f"({(chosen['end'] - chosen['onset']) / 2:.0f} hours)"
                    if chosen["size_mw"] > 0
                    else f"{site}: injected transfer"
                ),
            )
        )
    return alt.vconcat(*panels).properties(
        title=title(
            text="Forecasts around four injected transfers",
            subtitle=(
                "Scored year. For each gate site, the first injected transfer (by identifier) of "
                "at least 24 hours that moves load onto the site, from the 100-transfer sample. "
                "Each line is the forecast made 12 to 18 hours ahead; the dashed A5 line lies on "
                "the plain filter wherever the fit finds no flow. The shaded band is the plain "
                "filter's 80% band. The transfers were chosen before any score was read."
            ),
        )
    )


FIGURES: Final[dict[str, Callable[[], alt.TopLevelMixin]]] = {
    "tuning": figure_tuning,
    "recall_cells": figure_recall_cells,
    "examples": figure_examples,
    "variants": figure_variants,
    "headline": figure_headline,
    "transfer_weeks": figure_transfer_weeks,
    "fleet": figure_fleet,
    **{
        f"card_{site}": (lambda site=site: figure_card(site=site)) for site in common.DECISION_SITES
    },
}


def main() -> None:
    """Draw every figure, or those named."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=list(FIGURES))
    for name in parser.parse_args().only:
        save(FIGURES[name](), name=name)


if __name__ == "__main__":
    main()
