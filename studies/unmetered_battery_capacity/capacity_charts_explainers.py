"""Draw the explainer figures of the unmetered-battery-capacity page (2 to 4).

Figure 2 follows one simulated sum through the estimator, from the inputs to the posterior.
Figure 3 shows how the differentiable estimator reads a schedule off the precomputed
linear-programme dispatches. Figure 4 shows how to read a coverage plot. All three read saved
results (`rung1_posteriors.parquet`, `rung1b_rank_rule_posteriors.parquet`, `stacks.npz`) and the
inputs; nothing is refitted.

The worked example follows a fixed rule rather than the best result: the first series of rung 1
(in label order) that was not a tuning series (tuning used S6 and S2), S3, with the 2-hour battery
at the middle share (20%), in the first block (Sep-Nov), in the first full week of October 2025.
The posterior's error in the example is 7% of the true power, near the median error at this share
(5.1%).

Run: `uv run python studies/unmetered_battery_capacity/capacity_charts_explainers.py`.
"""

from datetime import UTC, datetime
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from capacity_charts_common import (
    PANEL_HEIGHT_PX,
    PERCENT,
    PLOT_WIDTH_PX,
    REFERENCE_COLOUR,
    draw_figure,
    save,
    write_notes,
)
from capacity_inputs import OUTPUT_DIR, day_ahead_on_grid, demand_series, window_half_hours
from capacity_rung1 import SEED
from capacity_rung3 import units
from capacity_runs import simulated_merchant_battery
from capacity_stacks import STACKS_PATH

EXAMPLE_SERIES: Final[str] = "S3"
EXAMPLE_HOURS: Final[float] = 2.0
EXAMPLE_SHARE: Final[float] = 0.2
EXAMPLE_BLOCK: Final[int] = 0
REAL_UNIT: Final[str] = "U01"
"""The first named public battery of rung 3, used for the second worked example by the same rule."""
WEEK_START: Final[datetime] = datetime(2025, 10, 6, tzinfo=UTC)
DAYS_PER_WEEK: Final[int] = 7
DEMAND_ORDER: Final[tuple[str, ...]] = ("S1", "S2", "S3", "S5", "S6", "S7", "S8", "BSP2", "GSP1")
NOTES: list[str] = []
BLOCK_NAMES: Final[tuple[str, ...]] = ("Sep-Nov", "Dec-Feb", "Mar-May", "Jun-Aug")


def stack_schedule(
    *, stack: np.ndarray, duration: float, efficiency: float, cap_weight: float
) -> np.ndarray:
    """Interpolate a one-megawatt schedule from the precomputed linear-programme stack.

    This repeats the estimator's interpolation (bilinear in log duration and efficiency, linear in
    the cycle-cap weight) for one parameter vector, in numpy.

    Args:
        stack: The merchant stack, shape (nodes, half-hours), node index
            `(i_duration * n_efficiency + i_efficiency) * 2 + cap`.
        duration: The usable duration in hours at full power.
        efficiency: The round-trip efficiency.
        cap_weight: The weight of the 2-cycle schedule against the 1-cycle schedule.

    Returns:
        The schedule in MW per MW of power, positive for export.
    """
    nodes = np.load(STACKS_PATH)
    d_nodes, e_nodes = nodes["duration_nodes"], nodes["efficiency_nodes"]
    n_d, n_e = len(d_nodes), len(e_nodes)
    position = (np.log(duration) - np.log(d_nodes[0])) / (
        (np.log(d_nodes[-1]) - np.log(d_nodes[0])) / (n_d - 1)
    )
    i_d = int(np.clip(np.floor(position), 0, n_d - 2))
    f_d = float(np.clip(position - i_d, 0.0, 1.0))
    i_e = int(np.clip(np.searchsorted(e_nodes, efficiency) - 1, 0, n_e - 2))
    f_e = float(np.clip((efficiency - e_nodes[i_e]) / (e_nodes[i_e + 1] - e_nodes[i_e]), 0.0, 1.0))
    out = np.zeros(stack.shape[1])
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                weight = (
                    (f_d if a else 1.0 - f_d)
                    * (f_e if b else 1.0 - f_e)
                    * (cap_weight if c else 1.0 - cap_weight)
                )
                out += weight * stack[((i_d + a) * n_e + (i_e + b)) * 2 + c]
    return out


def node_schedule(*, stack: np.ndarray, duration: float, efficiency: float, cap: int) -> np.ndarray:
    """Return the schedule at the stack node nearest a duration and efficiency."""
    nodes = np.load(STACKS_PATH)
    i_d = int(np.argmin(np.abs(np.log(nodes["duration_nodes"]) - np.log(duration))))
    i_e = int(np.argmin(np.abs(nodes["efficiency_nodes"] - efficiency)))
    return stack[(i_d * len(nodes["efficiency_nodes"]) + i_e) * 2 + (cap - 1)]


def _example_row() -> dict:
    """Return the saved posterior row of the worked example."""
    rows = pl.read_parquet(OUTPUT_DIR / "rung1_posteriors.parquet").filter(
        (pl.col("series") == EXAMPLE_SERIES)
        & (pl.col("nameplate_hours") == EXAMPLE_HOURS)
        & (pl.col("share") == EXAMPLE_SHARE)
        & (pl.col("block") == EXAMPLE_BLOCK)
    )
    if rows.height != 1:
        raise ValueError(f"Expected one example row, found {rows.height}")
    return rows.row(0, named=True)


def _week(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the example week of a frame with a UTC `start` column."""
    start = pl.lit(WEEK_START)
    return frame.filter(
        (pl.col("start") >= start) & (pl.col("start") < start + pl.duration(days=DAYS_PER_WEEK))
    )


def _fit_and_posterior_panels(
    *,
    week: pl.DataFrame,
    row: dict,
    true_power: float,
    true_energy: float,
    truth_label: str,
    step_prefix: str,
    posterior_prefix: str,
    x_domain: tuple[float, float] = (0.4, 1.8),
) -> tuple[alt.Chart, alt.LayerChart | alt.FacetChart]:
    """Draw the fitted-against-true schedule panel and the posterior-beside-truth panel.

    Args:
        week: The example week with `start`, `truth`, and `fitted` columns (MW).
        row: The saved posterior row.
        true_power: The true (or registered) power in MW.
        true_energy: The true usable energy, or the energy reference, in MWh.
        truth_label: The legend label of the truth line.
        step_prefix: Prefixes the schedule panel's title.
        posterior_prefix: Prefixes the posterior panel's title.
        x_domain: The posterior panel's range, as multiples of the truth.

    Returns:
        The schedule panel and the posterior panel.
    """
    x = alt.X("start:T", axis=alt.Axis(format="%a", tickCount=7), title=None)
    comparison = pl.concat(
        [
            week.select("start", pl.col("truth").alias("mw"), pl.lit(truth_label).alias("series")),
            week.select(
                "start",
                pl.col("fitted").alias("mw"),
                pl.lit("Fitted battery (posterior median power times the template)").alias(
                    "series"
                ),
            ),
        ]
    )
    fitted_chart = (
        alt.Chart(comparison)
        .mark_line(strokeWidth=1.3, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("mw:Q", title="MW"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=[
                        truth_label,
                        "Fitted battery (posterior median power times the template)",
                    ],
                    range=[REFERENCE_COLOUR, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, columns=1, labelLimit=420),
            ),
        )
        .properties(
            width=PLOT_WIDTH_PX,
            height=95,
            title=alt.TitleParams(
                f"{step_prefix}scaling the template by the posterior power", anchor="start"
            ),
        )
    )
    intervals = pl.DataFrame(
        {
            "quantity": ["Power (MW)", "Energy (MWh)"],
            "q05": [row["merchant_power_q05"], row["merchant_energy_q05"]],
            "q25": [row["merchant_power_q25"], row["merchant_energy_q25"]],
            "median": [row["merchant_power_median"], row["merchant_energy_median"]],
            "q75": [row["merchant_power_q75"], row["merchant_energy_q75"]],
            "q95": [row["merchant_power_q95"], row["merchant_energy_q95"]],
            "truth": [true_power, true_energy],
        }
    ).with_columns(
        **{c: pl.col(c) / pl.col("truth") * 1.0 for c in ("q05", "q25", "median", "q75", "q95")},
        one=pl.lit(1.0),
    )
    base = alt.Chart(intervals)
    y = alt.Y("quantity:N", title=None)
    posterior = alt.layer(
        base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE, clip=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=y,
            x=alt.X(
                "q05:Q",
                scale=alt.Scale(domain=list(x_domain)),
                title="Posterior as a multiple of the truth (1 = exact)",
            ),
            x2="q95:Q",
        ),
        base.mark_bar(height=10, color=ocf.DATA_BLUE, opacity=0.5, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=y, x="q25:Q", x2="q75:Q"
        ),
        base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=y, x="median:Q"
        ),
        base.mark_tick(color=ocf.BRAND_ORANGE, thickness=3, size=28, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=y, x="one:Q"
        ),
    ).properties(
        width=PLOT_WIDTH_PX,
        height=70,
        title=alt.TitleParams(
            f"{posterior_prefix}the posterior of power and energy beside the truth", anchor="start"
        ),
    )
    return fitted_chart, posterior


def _real_battery_panels(
    *, demand: np.ndarray, p99: float
) -> tuple[alt.Chart, alt.LayerChart | alt.FacetChart, dict]:
    """Draw the worked example's second half: a real public battery, by the same rule.

    The rule is the first series (S3), the first block, the middle share (20%), and the first named
    battery of rung 3. The truth line is the battery's metered output scaled to that share, and the
    energy truth is the smallest capacity that holds its state of charge (a lower bound).

    Args:
        demand: The example series' flow in MW.
        p99: The series' 99th-percentile absolute flow.

    Returns:
        The schedule panel, the posterior panel, and the saved posterior row.
    """
    rows = pl.read_parquet(OUTPUT_DIR / "rung3_posteriors.parquet").filter(
        (pl.col("series") == EXAMPLE_SERIES)
        & (pl.col("unit") == REAL_UNIT)
        & (pl.col("share") == EXAMPLE_SHARE)
        & (pl.col("block") == EXAMPLE_BLOCK)
    )
    if rows.height != 1:
        raise ValueError(f"Expected one real example row, found {rows.height}")
    row = rows.row(0, named=True)
    unit = next(u for u in units() if u["unit"] == REAL_UNIT)
    true_power = EXAMPLE_SHARE * p99
    battery = np.nan_to_num(unit["output_mw"]) * (true_power / unit["registered_mw"])
    template = stack_schedule(
        stack=np.load(STACKS_PATH)["merchant"],
        duration=row["merchant_duration_point"],
        efficiency=row["merchant_efficiency_point"],
        cap_weight=row["merchant_cap_weight_point"],
    )
    frame = pl.DataFrame(
        {
            "start": window_half_hours().dt.offset_by("-30m"),
            "truth": battery,
            "fitted": row["merchant_power_median"] * template,
        }
    )
    fitted_chart, posterior = _fit_and_posterior_panels(
        week=_week(frame=frame),
        row=row,
        true_power=true_power,
        true_energy=row["true_energy_reference_mwh"],
        truth_label="Real battery (metered output scaled to the share)",
        step_prefix="A real public battery: ",
        posterior_prefix="A real public battery: ",
        x_domain=(0.0, 1.8),
    )
    return fitted_chart, posterior, row


def worked_example_figure() -> alt.VConcatChart:
    """Draw figure 2: one sum, from its inputs to the posterior beside the truth."""
    row = _example_row()
    group = DEMAND_ORDER.index(EXAMPLE_SERIES)
    demand = demand_series()[EXAMPLE_SERIES]
    p99 = float(np.nanquantile(np.abs(demand), 0.99))
    hours_index = {1.0: 0, 2.0: 1, 4.0: 2}[EXAMPLE_HOURS]
    unit, usable, _ = simulated_merchant_battery(
        nameplate_hours=EXAMPLE_HOURS, seed=SEED + 100 * group + hours_index
    )
    true_power = EXAMPLE_SHARE * p99
    battery = true_power * unit
    stack = np.load(STACKS_PATH)["merchant"]
    template = stack_schedule(
        stack=stack,
        duration=row["merchant_duration_point"],
        efficiency=row["merchant_efficiency_point"],
        cap_weight=row["merchant_cap_weight_point"],
    )
    fitted = row["merchant_power_median"] * template
    frame = pl.DataFrame(
        {
            "start": window_half_hours().dt.offset_by("-30m"),
            "aggregate": (demand - battery),
            "price": day_ahead_on_grid(),
            "template": template,
            "truth": battery,
            "fitted": fitted,
        }
    ).with_columns(pl.col(pl.Float64).fill_nan(None))
    week = _week(frame=frame)
    x = alt.X("start:T", axis=alt.Axis(format="%a", tickCount=7), title=None)

    def line(*, column: str, title: str, y_title: str, colour: str) -> alt.Chart:
        return (
            alt.Chart(week.select("start", column))
            .mark_line(strokeWidth=1.3, color=colour, aria=False)
            .encode(x=x, y=alt.Y(f"{column}:Q", title=y_title, scale=alt.Scale(zero=False)))  # ty: ignore[unresolved-attribute]
            .properties(
                width=PLOT_WIDTH_PX,
                height=85,
                title=alt.TitleParams(title, anchor="start"),
            )
        )

    fitted_chart, posterior = _fit_and_posterior_panels(
        week=week,
        row=row,
        true_power=true_power,
        true_energy=true_power * usable,
        truth_label="True battery",
        step_prefix="Step 3: ",
        posterior_prefix="Step 4: ",
    )
    real_fitted_chart, real_posterior, real_row = _real_battery_panels(demand=demand, p99=p99)
    _note_example(row=row, true_power=true_power, usable=usable)
    _note_real_example(row=real_row, true_power=true_power)
    return draw_figure(
        panels=[
            line(
                column="aggregate",
                title="Step 1: the input, a primary's flow with a 20% battery already subtracted",
                y_title="MW",
                colour=ocf.BLACK_1,
            ),
            line(
                column="price",
                title="Step 1: the input, the N2EX day-ahead price the battery follows",
                y_title="£ per MWh",
                colour=ocf.DATA_SKY,
            ),
            line(
                column="template",
                title=(
                    "Step 2: the template, a one-megawatt schedule at the fitted "
                    "duration and efficiency"
                ),
                y_title="MW per MW",
                colour=ocf.DATA_PURPLE,
            ),
            fitted_chart,
            posterior,
            real_fitted_chart,
            real_posterior,
        ],
        number=2,
        title=(
            "A simulated in-family battery is sized well; a real public battery's power "
            "is underestimated"
        ),
        subtitle=[
            (
                f"Primary {EXAMPLE_SERIES}, September to November, a {EXAMPLE_HOURS:g}-hour"
                f" battery at a {EXAMPLE_SHARE * PERCENT:g}% share, week from 6 October 2025."
                " Series, battery, and block follow a fixed rule (the first of each), not the"
                " best result. The simulated battery follows the estimator's own dispatch."
                " The last two panels repeat steps 3 and 4 for the first real public battery"
                " of rung 3, in the same series, block, week, and share."
            ),
            (
                "Step 4: dot, posterior median; thick bar, 50% interval; thin line, 90%"
                " interval; orange tick, the truth. The fitted schedule before the "
                "smooth state-of-charge gate is drawn, so its edges are sharper than "
                "the estimator's."
            ),
        ],
        figure_planning=None,
    ).resolve_scale(color="independent")


def _note_real_example(*, row: dict, true_power: float) -> None:
    """Record the second worked example's numbers for the page."""
    true_energy = row["true_energy_reference_mwh"]
    NOTES.append(
        f"Second worked example (real public battery {REAL_UNIT}, {EXAMPLE_SERIES},"
        f" {BLOCK_NAMES[EXAMPLE_BLOCK]}, {EXAMPLE_SHARE:.0%} share): the posterior median power is"
        f" {row['merchant_power_median'] / true_power:.3f} times the registered share, its 90%"
        f" interval runs from {row['merchant_power_q05'] / true_power:.3f} to"
        f" {row['merchant_power_q95'] / true_power:.3f} times, the energy median is"
        f" {row['merchant_energy_median'] / true_energy:.3f} times the energy reference, the log"
        f" Bayes factor is {row['log_bayes_factor']:.1f} and tau is {row['tau']:.1f}."
    )


def _note_example(*, row: dict, true_power: float, usable: float) -> None:
    """Record the worked example's numbers for the page."""
    true_energy = true_power * usable
    NOTES.append(
        f"Worked example ({EXAMPLE_SERIES}, {BLOCK_NAMES[EXAMPLE_BLOCK]}, 2-hour, 20% share):"
        f" the posterior median power is {row['merchant_power_median'] / true_power:.3f} times"
        f" the truth, its 90% interval runs from {row['merchant_power_q05'] / true_power:.3f} to"
        f" {row['merchant_power_q95'] / true_power:.3f} times, the energy median is"
        f" {row['merchant_energy_median'] / true_energy:.3f} times the true usable energy (90%"
        f" interval {row['merchant_energy_q05'] / true_energy:.3f} to"
        f" {row['merchant_energy_q95'] / true_energy:.3f}), the log Bayes factor is"
        f" {row['log_bayes_factor']:.1f} and tau is {row['tau']:.1f}."
    )


def dispatch_figure() -> alt.VConcatChart:
    """Draw figure 3: how one dispatch is read off the precomputed linear-programme schedules."""
    row = _example_row()
    stack = np.load(STACKS_PATH)["merchant"]
    efficiency = row["merchant_efficiency_point"]
    frame_columns = {
        "start": window_half_hours().dt.offset_by("-30m"),
        "price": day_ahead_on_grid(),
    }
    labelled = {}
    for hours in (1.0, 2.0, 4.0):
        labelled[f"{hours:g}-hour node"] = node_schedule(
            stack=stack, duration=hours, efficiency=efficiency, cap=1
        )
    frame = pl.DataFrame({**frame_columns, **labelled}).with_columns(
        pl.col(pl.Float64).fill_nan(None)
    )
    day = frame.filter(
        (pl.col("start") >= pl.lit(WEEK_START))
        & (pl.col("start") < pl.lit(WEEK_START) + pl.duration(days=2))
    )
    x = alt.X("start:T", axis=alt.Axis(format="%a %H:%M", tickCount=8), title=None)
    price = (
        alt.Chart(day.select("start", "price"))
        .mark_line(strokeWidth=1.3, color=ocf.DATA_SKY, aria=False)
        .encode(x=x, y=alt.Y("price:Q", title="£ per MWh", scale=alt.Scale(zero=False)))  # ty: ignore[unresolved-attribute]
        .properties(
            width=PLOT_WIDTH_PX,
            height=80,
            title=alt.TitleParams("The day-ahead price", anchor="start"),
        )
    )
    nodes = day.select("start", *labelled).unpivot(
        index="start", variable_name="node", value_name="mw_per_mw"
    )
    names = list(labelled)
    node_chart = (
        alt.Chart(nodes)
        .mark_line(strokeWidth=1.3, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("mw_per_mw:Q", title="MW per MW"),
            color=alt.Color(
                "node:N",
                scale=alt.Scale(
                    domain=names, range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]
                ),
                legend=alt.Legend(title=None, columns=3, labelLimit=300),
            ),
        )
        .properties(
            width=PLOT_WIDTH_PX,
            height=100,
            title=alt.TitleParams(
                f"Dispatches for three usable durations (efficiency {efficiency:.2f},"
                " 1 cycle a day)",
                anchor="start",
            ),
        )
    )
    mixed = stack_schedule(
        stack=stack,
        duration=row["merchant_duration_point"],
        efficiency=efficiency,
        cap_weight=row["merchant_cap_weight_point"],
    )
    lower, upper = (
        node_schedule(stack=stack, duration=d, efficiency=efficiency, cap=1)
        for d in (row["merchant_duration_point"] / 1.15, row["merchant_duration_point"] * 1.15)
    )
    interpolation = (
        pl.DataFrame(
            {
                **frame_columns,
                "A neighbouring node, shorter": lower,
                "A neighbouring node, longer": upper,
                "Interpolated at the fitted duration": mixed,
            }
        )
        .with_columns(pl.col(pl.Float64).fill_nan(None))
        .filter(
            (pl.col("start") >= pl.lit(WEEK_START))
            & (pl.col("start") < pl.lit(WEEK_START) + pl.duration(days=2))
        )
        .drop("price")
        .unpivot(index="start", variable_name="schedule", value_name="mw_per_mw")
    )
    domain = [
        "A neighbouring node, shorter",
        "A neighbouring node, longer",
        "Interpolated at the fitted duration",
    ]
    interp_chart = (
        alt.Chart(interpolation)
        .mark_line(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("mw_per_mw:Q", title="MW per MW"),
            color=alt.Color(
                "schedule:N",
                scale=alt.Scale(
                    domain=domain, range=[ocf.DATA_BLUE, ocf.DATA_SKY, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(title=None, columns=1, labelLimit=400),
            ),
            strokeWidth=alt.StrokeWidth(
                "schedule:N",
                scale=alt.Scale(domain=domain, range=[1.0, 1.0, 2.4]),
                legend=None,
            ),
        )
        .properties(
            width=PLOT_WIDTH_PX,
            height=100,
            title=alt.TitleParams(
                "Read between neighbouring nodes at the fitted duration of"
                f" {row['merchant_duration_point']:.2f} hours",
                anchor="start",
            ),
        )
    )
    return draw_figure(
        panels=[price, node_chart, interp_chart],
        number=3,
        title=(
            "The differentiable battery follows the price through precomputed "
            "dispatches and fits only a few numbers"
        ),
        subtitle=[
            (
                "Two days from 6 October 2025 (UTC). Positive is export. Each node is "
                "the day-by-day linear programme for one usable duration and "
                "efficiency; the stack holds 3,800 nodes."
            ),
            (
                "The estimator fits the power, the duration, the efficiency, and a cap "
                "weight by gradient descent, reading the schedule between nodes, then "
                "passes it through a smooth state-of-charge limit."
            ),
        ],
        figure_planning=None,
    ).resolve_scale(color="independent", strokeWidth="independent")


def coverage_figure() -> alt.VConcatChart:
    """Draw figure 4: how to read a coverage plot, with an honest case and an overconfident case."""
    cases = {
        "In the estimator's family, 2% share": (
            pl.read_parquet(OUTPUT_DIR / "rung1_posteriors.parquet").filter(
                pl.col("share") == 0.02
            ),
            ocf.DATA_BLUE,
        ),
        "Rank-rule battery, 40% share": (
            pl.read_parquet(OUTPUT_DIR / "rung1b_rank_rule_posteriors.parquet").filter(
                pl.col("share") == 0.4
            ),
            ocf.BRAND_ORANGE,
        ),
    }
    panels = []
    reliability = []
    for name, (frame, _) in cases.items():
        ratio = frame.filter(pl.col("has_interval")).with_columns(
            low=pl.col("merchant_power_q05") / pl.col("true_power_mw"),
            high=pl.col("merchant_power_q95") / pl.col("true_power_mw"),
            mid=pl.col("merchant_power_median") / pl.col("true_power_mw"),
            low50=pl.col("merchant_power_q25") / pl.col("true_power_mw"),
            high50=pl.col("merchant_power_q75") / pl.col("true_power_mw"),
        )
        ratio = (
            ratio.with_columns(
                holds=(pl.col("low") <= 1.0) & (pl.col("high") >= 1.0),
                holds50=(pl.col("low50") <= 1.0) & (pl.col("high50") >= 1.0),
            )
            .sort("mid")
            .with_columns(rank=pl.int_range(pl.len()))
        )
        n = ratio.height
        covered = int(ratio["holds"].sum())
        NOTES.append(
            f"Coverage figure, {name}: the 90% power interval holds the truth in {covered} of {n}"
            f" sums ({covered / n:.3f}); the 50% interval in {int(ratio['holds50'].sum())}"
            f" ({ratio['holds50'].mean():.3f})."
        )
        reliability.append(
            pl.DataFrame(
                {
                    "case": [name, name],
                    "nominal": [0.5, 0.9],
                    "actual": [int(ratio["holds50"].sum()) / n, covered / n],
                }
            )
        )
        shown = ratio.with_columns(
            outcome=pl.when(pl.col("holds"))
            .then(pl.lit("Interval holds the truth"))
            .otherwise(pl.lit("Interval misses the truth"))
        )
        base = alt.Chart(shown)
        colour = alt.Color(
            "outcome:N",
            scale=alt.Scale(
                domain=["Interval holds the truth", "Interval misses the truth"],
                range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
            ),
            legend=alt.Legend(title=None) if not panels else None,
        )
        panels.append(
            alt.layer(
                base.mark_rule(strokeWidth=1.2, aria=False).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("rank:O", axis=None),
                    x=alt.X(
                        "low:Q",
                        scale=alt.Scale(type="log", domain=[0.1, 30], clamp=True),
                        title=(
                            "Posterior power as a multiple of the true power (1 ="
                            " exact; log axis, ends clipped)"
                        ),
                        axis=alt.Axis(values=[0.1, 0.3, 1, 3, 10, 30], format="g"),
                    ),
                    x2="high:Q",
                    color=colour,
                ),
                alt.Chart(pl.DataFrame({"one": [1.0]}))
                .mark_rule(color=REFERENCE_COLOUR, strokeDash=[4, 3], aria=False)
                .encode(x="one:Q"),  # ty: ignore[unresolved-attribute]
            ).properties(
                width=PLOT_WIDTH_PX,
                height=PANEL_HEIGHT_PX * 1.5,
                title=alt.TitleParams(
                    f"{name}: {covered} of {n} intervals hold the truth", anchor="start"
                ),
            )
        )
    diagonal = pl.DataFrame({"nominal": [0.0, 1.0], "actual": [0.0, 1.0]})
    reliability_chart = alt.layer(
        alt.Chart(diagonal)
        .mark_line(color=REFERENCE_COLOUR, strokeDash=[4, 3], aria=False)
        .encode(x="nominal:Q", y="actual:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(pl.concat(reliability))
        .mark_line(point=True, strokeWidth=1.8, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "nominal:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="Nominal level of the interval",
                axis=alt.Axis(values=[0, 0.5, 0.9]),
            ),
            y=alt.Y(
                "actual:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="Share of sums that hold the truth",
            ),
            color=alt.Color(
                "case:N",
                scale=alt.Scale(domain=list(cases), range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title=None, columns=1, labelLimit=400),
            ),
        ),
    ).properties(
        width=PLOT_WIDTH_PX // 2,
        height=PANEL_HEIGHT_PX * 1.6,
        title=alt.TitleParams("On the dashed line, an interval means what it says", anchor="start"),
    )
    return draw_figure(
        panels=[*panels, reliability_chart],
        number=4,
        title="A coverage plot shows whether an interval means what it says",
        subtitle=[
            (
                "Each line in the upper two panels is one simulated sum's 90% interval "
                "for the battery's power, divided by the true power and sorted by its "
                "median; the dashed line is the truth."
            ),
            (
                "The lower panel plots, for the 50% and 90% intervals, the share that "
                "hold the truth against the nominal level. Below the dashed line, an "
                "interval is overconfident."
            ),
        ],
        figure_planning=None,
    ).resolve_scale(color="independent")


def main() -> None:
    """Write the explainer figures and the numbers derived while drawing them."""
    for name, chart in {
        "worked_example": worked_example_figure(),
        "dispatch": dispatch_figure(),
        "coverage": coverage_figure(),
    }.items():
        print(f"Wrote {save(chart=chart, name=name)}")
    print(f"Wrote {write_notes(name='explainers', notes=NOTES)}")


if __name__ == "__main__":
    main()
