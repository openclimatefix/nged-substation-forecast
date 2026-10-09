"""Draw the battery study's rung 7c figures (22 and 23).

Figure 22 is built from `rung7c_known_answer_fits.parquet` and Figure 23 from
`rung7c_costs_fits.parquet`. The demand-like halves are public supplier and virtual BMUs, so the
figures name them.

Run after `battery_rung7c_known_answer.py` and `battery_rung7c_costs.py`:
`uv run python studies/battery_pv_separation/battery_charts_rung7c.py`.
"""

from typing import Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from battery_charts import ASSETS_DIR, PANEL_HEIGHT_PX, PLOT_WIDTH_PX, Layer
from battery_inputs import OUTPUT_DIR
from battery_rung7c_costs import COST_POWERS_MW, SETTINGS
from battery_rung7c_costs_summary import cost_table
from battery_rung7c_known_answer import DEMAND_BMUS, DEMAND_KINDS, SOLAR_P99_MW
from battery_rung7c_known_answer_summary import TRUE_LABEL, assumed_order, with_assumed_power
from studies.charts import figure

PERCENT: Final[float] = 100.0
REAL_LABEL: Final[str] = "Real output"
REPLICA_LABEL: Final[str] = "Calendar replica"
KIND_LABELS: Final[dict[str, str]] = {"real": REAL_LABEL, "replica": REPLICA_LABEL}
DEMAND_COLOURS: Final[tuple[str, ...]] = (ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE)


def _known_answer_panel(*, frame: pl.DataFrame, solar_p99: float, column: str, title: str) -> Layer:
    """Draw one metric against assumed battery power for one solar size, a line per demand half.

    Args:
        frame: The scored fits with the assumed power, one row per aggregate and power.
        solar_p99: The solar half's 99th-percentile output in megawatts.
        column: The metric.
        title: The panel's title.

    Returns:
        The panel.
    """
    means = (
        frame.filter(pl.col("solar_p99_mw") == solar_p99)
        .group_by("demand_bmu", "demand_kind", "assumed")
        .agg(pl.col(column).mean())
        .with_columns(
            series=pl.col("demand_bmu")
            + ", "
            + pl.col("demand_kind").replace_strict(KIND_LABELS, return_dtype=pl.String)
        )
    )
    domain = [f"{bmu}, {KIND_LABELS[kind]}" for bmu in DEMAND_BMUS for kind in DEMAND_KINDS]
    colours = [DEMAND_COLOURS[i // len(DEMAND_KINDS)] for i in range(len(domain))]
    dashes = [[1, 0] if i % 2 == 0 else [4, 3] for i in range(len(domain))]
    x = alt.X(
        "assumed:O",
        sort=assumed_order(),
        title="Assumed battery power (MW; 2 hours of energy; True is the battery half's own power)",
    )
    encoding = {
        "x": x,
        "y": alt.Y(f"{column}:Q", title=title),
        "color": alt.Color(
            "series:N",
            scale=alt.Scale(domain=domain, range=colours),
            legend=alt.Legend(title=None, columns=1, symbolLimit=0, labelLimit=400),
        ),
    }
    lines = (
        alt.Chart(means.filter(pl.col("assumed") != TRUE_LABEL))
        .mark_line(point=True, strokeWidth=1.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            **encoding,
            strokeDash=alt.StrokeDash(
                "series:N", scale=alt.Scale(domain=domain, range=dashes), legend=None
            ),
        )
    )
    # The fit at the battery half's own power is a separate point, not the next step of the sweep.
    true_points = (
        alt.Chart(means.filter(pl.col("assumed") == TRUE_LABEL))
        .mark_point(filled=True, size=70, aria=False)
        .encode(**encoding)  # ty: ignore[unresolved-attribute]
    )
    layers: list[alt.Chart] = [lines, true_points]
    if column == "ratio_fleet":
        layers.insert(
            0,
            alt.Chart(pl.DataFrame({"y": [1.0]}))
            .mark_rule(color=ocf.BLACK_1, strokeWidth=1.0, aria=False)
            .encode(y="y:Q"),  # ty: ignore[unresolved-attribute]
        )
    return cast(Layer, alt.layer(*layers)).properties(
        width=PLOT_WIDTH_PX - 60,
        height=PANEL_HEIGHT_PX * 1.3,
        title=alt.TitleParams(
            f"Solar half with a {solar_p99:g} MW 99th percentile", anchor="start"
        ),
    )


def known_answer_figure(*, number: int) -> alt.VConcatChart:
    """Draw the capacity ratio and the solar error against assumed battery power.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    fits = pl.read_parquet(OUTPUT_DIR / "rung7c_known_answer_fits.parquet")
    scored = (
        with_assumed_power(fits=fits)
        .filter(pl.col("solar_p99_mw") > 0)
        .with_columns(
            ratio_fleet=pl.col("fitted_ac_mw") / pl.col("fleet_alone_ac_mw"),
            solar_error_pct=pl.col("nmae_of_solar_p99") * PERCENT,
        )
    )
    sizes = SOLAR_P99_MW[1:]

    def mean_ratio(*, label: str) -> float:
        part = scored.filter(
            (pl.col("assumed") == label)
            & (pl.col("solar_p99_mw") == SOLAR_P99_MW[1])
            & (pl.col("demand_kind") == "real")
        )
        return float(part["ratio_fleet"].mean())  # ty: ignore[invalid-argument-type]

    low, high = assumed_order()[0], "200"
    panels = [
        _known_answer_panel(
            frame=scored,
            solar_p99=size,
            column="ratio_fleet",
            title="Fitted solar capacity over the solar half's own fit",
        )
        for size in sizes
    ] + [
        _known_answer_panel(
            frame=scored,
            solar_p99=size,
            column="solar_error_pct",
            title="Solar series error (% of solar p99)",
        )
        for size in sizes
    ]
    return figure(
        panels=panels,
        number=number,
        title=(
            f"On synthetic aggregates with a known answer, rung 7b's model fits "
            f"{mean_ratio(label=low):.2f} times the solar "
            f"capacity with no battery and {mean_ratio(label=high):.2f} times with a {high} MW "
            f"battery, for a {SOLAR_P99_MW[1]:g} MW solar half and a real demand-like half"
        ),
        subtitle=[
            (
                "Each aggregate is a solar half, a battery half, and a demand-like half "
                f"({', '.join(DEMAND_BMUS)}, real or as its calendar replica, scaled to a 50 MW "
                "99th percentile), fitted with four fleet curves, a seasonal calendar baseline, "
                "and a battery of the assumed power. Each point is the mean over 16 aggregates "
                "(4 solar sets by 4 batteries). In the upper panels the black line at 1 is the "
                "same model fitted to the solar half alone. The lone dots at True are the fits "
                "given the battery half's own power (75 MW for the 25 MW solar half, 50 MW for "
                "the 50 MW one)."
            )
        ],
        figure_planning=None,
    )


def cost_label(*, throughput: float, solar: float) -> str:
    """Return the axis label of a cost setting."""
    return f"battery {throughput:g}, solar {solar:g}"


def cost_panel(*, table: pl.DataFrame, power: float) -> Layer:
    """Draw the total fitted solar capacity under each cost setting at one assumed power.

    Args:
        table: `battery_rung7c_costs_summary.cost_table`'s output.
        power: The assumed battery power in megawatts.

    Returns:
        The panel.
    """
    labels = [cost_label(throughput=t, solar=s) for t, s in SETTINGS]
    long = (
        table.filter(pl.col("power_mw") == power)
        .unpivot(
            on=["real_all_mw", "replica_all_mw", "real_cloud_mw"],
            index=["throughput_cost", "solar_cost"],
            variable_name="quantity",
            value_name="mw",
        )
        .with_columns(
            setting=pl.struct("throughput_cost", "solar_cost").map_elements(
                lambda s: cost_label(throughput=s["throughput_cost"], solar=s["solar_cost"]),
                return_dtype=pl.String,
            ),
            line=pl.col("quantity").replace_strict(
                {
                    "real_all_mw": "Real output, all 25 BMUs",
                    "replica_all_mw": "Calendar replicas, all 25 BMUs",
                    "real_cloud_mw": "Real output, 9 cloud-signal BMUs",
                },
                return_dtype=pl.String,
            ),
        )
    )
    domain = [
        "Real output, all 25 BMUs",
        "Calendar replicas, all 25 BMUs",
        "Real output, 9 cloud-signal BMUs",
    ]
    no_battery = table.filter((pl.col("power_mw") == 0.0) & (pl.col("solar_cost") == 0.0))[
        "real_all_mw"
    ][0]
    lines = (
        alt.Chart(long)
        .mark_point(filled=True, size=70, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "setting:O",
                sort=labels,
                title="Cost setting (per MW)",
                axis=alt.Axis(labelAngle=-20),
            ),
            y=alt.Y("mw:Q", title="Total fitted solar capacity (MW)", scale=alt.Scale(zero=True)),
            color=alt.Color(
                "line:N",
                scale=alt.Scale(
                    domain=domain, range=[ocf.BRAND_ORANGE, ocf.ENSEMBLE_LINE, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=400),
            ),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [no_battery]}))
        .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3], strokeWidth=1.0, aria=False)
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return cast(Layer, alt.layer(rule, lines)).properties(
        width=PLOT_WIDTH_PX - 60,
        height=PANEL_HEIGHT_PX * 1.5,
        title=alt.TitleParams(f"Assumed battery power {power:g} MW", anchor="start"),
    )


def costs_figure(*, number: int) -> alt.VConcatChart:
    """Draw the total fitted solar capacity against the cost settings.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    table = cost_table(fits=pl.read_parquet(OUTPUT_DIR / "rung7c_costs_fits.parquet"))
    real = table.filter(pl.col("power_mw") > 0)["real_all_mw"]
    return figure(
        panels=[cost_panel(table=table, power=power) for power in COST_POWERS_MW],
        number=number,
        title=(
            f"Across all cost settings tried, the fitted solar capacity of the 25 BMUs stays "
            f"between {real.min():,.0f} MW and {real.max():,.0f} MW"
        ),
        subtitle=[
            (
                "Rung 7b's model fitted to the 25 real aggregate BMUs and their calendar replicas "
                "while varying the cost per megawatt of battery throughput (against 1 per "
                "megawatt of residual) and adding a cost per megawatt of fitted solar capacity. "
                "The dashed line is the fit with no battery and no solar cost. Battery 0.1 and "
                "solar 0 are rung 7b's settings. The solar costs of 0.01 and 0.05 are charged once "
                "per megawatt of capacity against a residual summed over about 17,500 half-hours, "
                "so they are too small to move the fit and test nothing."
            )
        ],
        figure_planning=None,
    )


def main() -> None:
    """Write figures 22 and 23 as SVGs under `docs/studies/assets/` and PNG previews."""
    charts = {
        "rung7c_known_answer": known_answer_figure(number=22),
        "rung7c_costs": costs_figure(number=23),
    }
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    previews = OUTPUT_DIR / "previews"
    previews.mkdir(exist_ok=True)
    for name, chart in charts.items():
        target = ASSETS_DIR / f"battery_pv_separation_{name}.svg"
        chart.save(target)
        chart.save(previews / f"{name}.png", scale_factor=1.3)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
