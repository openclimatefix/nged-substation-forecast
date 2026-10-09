"""Draw the primer figures P1 to P5 of the background pages on GB batteries.

Every figure is built from the tables `battery_primer.py` saved. The batteries, the prices, and the
market data are public, so the figures name each battery and show calendar dates and megawatts.

Run after `battery_primer.py`:
`uv run python studies/battery_pv_separation/battery_charts_primer.py`.
"""

from datetime import timedelta
from typing import Final, cast

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from battery_charts import BACKGROUND_ASSETS_DIR, PANEL_HEIGHT_PX, PLOT_WIDTH_PX, Layer, line_panel
from battery_inputs import BATTERIES, NAMES, OUTPUT_DIR, WEEK_START
from battery_primer import FRAME_PATH, FREQUENCY_BINS, MIN_BIN_COUNT, SEASONS, SUMMARY_PATH
from studies.charts import figure

BATTERY_COLOURS: Final[dict[str, str]] = {
    "Lakeside": ocf.BRAND_ORANGE,
    "Dollymans": ocf.DATA_BLUE,
    "Thurrock": ocf.DATA_PURPLE,
    "Ocker Hill": ocf.DATA_GREEN,
}
"""One colour per battery, the same on every primer figure."""
PRICE_COLOUR: Final[str] = ocf.BLACK_1
SCATTER_BATTERIES: Final[tuple[str, str]] = ("T_LKSDB-1", "E_DOLLB-1")
"""The two batteries of the frequency figure."""
SCATTER_SAMPLE: Final[int] = 1500
"""Half-hours drawn as faint dots in each frequency panel."""
SCATTER_SEED: Final[int] = 6
GROUP_COLOURS: Final[dict[str, str]] = {
    "Inside a response block (candidate)": ocf.BRAND_ORANGE,
    "Outside a response block": ocf.DATA_BLUE,
}


def _tod_panel(
    *,
    table: pl.DataFrame,
    value: str,
    y_title: str,
    title: str,
    colours: dict[str, str],
    last: bool,
    height: int = PANEL_HEIGHT_PX,
) -> Layer:
    """Draw mean values against the half-hour of the UTC day.

    Args:
        table: Columns `hour` (0 to 23.5), `series`, and the value column.
        value: The value column.
        y_title: The y axis title.
        title: The panel title.
        colours: The colour of each series.
        last: Whether the panel carries the x axis labels.
        height: The panel height in pixels.

    Returns:
        The panel.
    """
    chart = (
        alt.Chart(table, title=alt.TitleParams(title, anchor="start", fontSize=11, offset=2))
        .mark_line(interpolate="step-after", strokeWidth=1.6, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                scale=alt.Scale(domain=[0, 24], nice=False),
                axis=alt.Axis(
                    values=list(range(0, 25, 3)),
                    labels=last,
                    ticks=last,
                    title="Hour of the day (UTC)" if last else None,
                ),
            ),
            y=alt.Y(f"{value}:Q", axis=alt.Axis(tickCount=4, title=y_title)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, columns=5, labelLimit=200, symbolLimit=0),
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=height)
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(color=ocf.BLACK_1, strokeWidth=0.7, strokeDash=[4, 3], aria=False)
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return cast(Layer, alt.layer(chart, zero) if value == "output_mw" else alt.layer(chart))


def typical_day_figure() -> alt.VConcatChart:
    """Draw P1: mean output and mean day-ahead price by hour of the day, in winter and summer.

    Returns:
        The figure.
    """
    day = pl.read_parquet(OUTPUT_DIR / "primer_typical_day.parquet").with_columns(
        hour=pl.col("tod") / 2.0
    )
    palette: dict[str, str] = {**BATTERY_COLOURS, "Day-ahead price": PRICE_COLOUR}
    panels = []
    for index, season in enumerate(SEASONS):
        part = day.filter(pl.col("season") == season)
        price = (
            part.filter(pl.col("bmu_id") == BATTERIES[0])
            .select("hour", "day_ahead_gbp_per_mwh")
            .with_columns(series=pl.lit("Day-ahead price"))
        )
        output = part.with_columns(
            series=pl.col("bmu_id").replace_strict(NAMES, return_dtype=pl.String)
        )
        panels.append(
            _tod_panel(
                table=price,
                value="day_ahead_gbp_per_mwh",
                y_title="£ per MWh",
                title=f"{season}: mean day-ahead price (N2EX)",
                colours=palette,
                last=False,
                height=80,
            )
        )
        panels.append(
            _tod_panel(
                table=output,
                value="output_mw",
                y_title="MW (positive: export)",
                title=f"{season}: mean battery output",
                colours=palette,
                last=index == len(SEASONS) - 1,
            )
        )
    return figure(
        panels=panels,
        number="P1",
        title=(
            "On a typical day the four batteries charge in the cheap half-hours and export in the "
            "dear half-hours"
        ),
        subtitle=[
            (
                "Mean over the days of each season, 1 September 2025 to 31 August 2026, by "
                "half-hour of the UTC day. Output is settled metering (B1610) as megawatts. "
                "Summer prices peak later and higher than winter prices, and the batteries' "
                "export peak moves with them."
            )
        ],
        figure_planning=None,
    )


def decomposition_figure() -> alt.VConcatChart:
    """Draw P2: one battery-week as metered output, planned output, accepted volume, remainder.

    Returns:
        The figure.
    """
    frame = (
        pl.read_parquet(FRAME_PATH)
        .filter(pl.col("bmu_id") == BATTERIES[0])
        .filter((pl.col("time") >= WEEK_START) & (pl.col("time") < WEEK_START + timedelta(days=7)))
    )
    specs = (
        ("output_mw", "Metered output (B1610)", ocf.BLACK_1),
        ("plan_mw", "Planned output (physical notification)", ocf.DATA_BLUE),
        (
            "accepted_mw",
            "Accepted balancing volume (BOAV)",
            ocf.BRAND_ORANGE,
        ),
        ("remainder_mw", "Remainder: metered minus planned minus accepted", ocf.DATA_PURPLE),
    )
    palette: dict[str, str] = {label: colour for _, label, colour in specs}
    panels = []
    for index, (column, label, colour) in enumerate(specs):
        panel = line_panel(
            series={label: (frame, column, colour)},
            title=label,
            y_title="MW",
            last=index == len(specs) - 1,
            interpolate="step-after",
            rules={"zero": 0.0},
            palette=palette,
            height=85,
        )
        panels.append(panel)
    return figure(
        panels=panels,
        number="P2",
        title=(
            "The planned output and the balancing acceptances explain most of a battery's output"
        ),
        subtitle=[
            (
                "Lakeside, one week from Monday 15 December 2025, UTC. In each half-hour, "
                "metered output equals the planned output plus the accepted volume plus the "
                "remainder (all in megawatts, half-hour means). Each panel has its own y axis."
            )
        ],
        figure_planning=None,
    )


def _bin_means(*, x: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    """Return the mean of y in equal-count bins of x.

    Args:
        x: The frequency deviations.
        y: The remainders.

    Returns:
        Columns `deviation_hz`, `remainder_mw`, and `count`, one row per bin.
    """
    order = np.argsort(x)
    rows = [
        {
            "deviation_hz": float(x[chunk].mean()),
            "remainder_mw": float(y[chunk].mean()),
            "count": len(chunk),
        }
        for chunk in np.array_split(order, FREQUENCY_BINS)
        if len(chunk) >= MIN_BIN_COUNT
    ]
    return pl.DataFrame(rows)


def frequency_figure() -> alt.VConcatChart:
    """Draw P3: the remainder against the frequency deviation, with and without a response block.

    Returns:
        The figure.
    """
    frames = pl.read_parquet(FRAME_PATH).drop_nulls(["remainder_mw", "frequency_deviation_hz"])
    rng = np.random.default_rng(SCATTER_SEED)
    summary = pl.read_parquet(SUMMARY_PATH)
    panels = []
    for index, bmu_id in enumerate(SCATTER_BATTERIES):
        frame = frames.filter(pl.col("bmu_id") == bmu_id).with_columns(
            group=pl.when(pl.col("contracted_response_mw") > 0)
            .then(pl.lit("Inside a response block (candidate)"))
            .otherwise(pl.lit("Outside a response block"))
        )
        means, dots = [], []
        for group in GROUP_COLOURS:
            part = frame.filter(pl.col("group") == group)
            x = part["frequency_deviation_hz"].to_numpy()
            y = part["remainder_mw"].to_numpy()
            means.append(_bin_means(x=x, y=y).with_columns(group=pl.lit(group)))
            pick = rng.choice(len(x), size=min(SCATTER_SAMPLE, len(x)), replace=False)
            dots.append(
                pl.DataFrame({"deviation_hz": x[pick], "remainder_mw": y[pick]}).with_columns(
                    group=pl.lit(group)
                )
            )
        colour = alt.Color(
            "group:N",
            scale=alt.Scale(domain=list(GROUP_COLOURS), range=list(GROUP_COLOURS.values())),
            legend=alt.Legend(title=None, columns=2, labelLimit=300, symbolLimit=0),
        )
        x_axis = alt.X(
            "deviation_hz:Q",
            scale=alt.Scale(domain=[-0.15, 0.15], clamp=True),
            axis=alt.Axis(
                title="50 Hz minus the half-hour's mean frequency (Hz; positive: low frequency)"
                if index == len(SCATTER_BATTERIES) - 1
                else None,
                labels=index == len(SCATTER_BATTERIES) - 1,
                ticks=index == len(SCATTER_BATTERIES) - 1,
            ),
        )
        y_axis = alt.Y(
            "remainder_mw:Q",
            scale=alt.Scale(domain=[-45, 45], clamp=True),
            axis=alt.Axis(title="Remainder (MW)"),
        )
        points = (
            alt.Chart(pl.concat(dots))
            .mark_circle(size=7, opacity=0.18, aria=False)
            .encode(x=x_axis, y=y_axis, color=colour)  # ty: ignore[unresolved-attribute]
        )
        lines = (
            alt.Chart(pl.concat(means))
            .mark_line(point=True, strokeWidth=2, aria=False)
            .encode(x=x_axis, y=y_axis, color=colour)  # ty: ignore[unresolved-attribute]
        )
        row = summary.filter(pl.col("bmu_id") == bmu_id).row(0, named=True)
        title = (
            f"{NAMES[bmu_id]}: remainder SD {row['with_response_remainder_sd_mw']:.1f} MW inside "
            f"response blocks, {row['without_response_remainder_sd_mw']:.1f} MW outside"
        )
        panels.append(
            cast(Layer, alt.layer(points, lines)).properties(
                width=PLOT_WIDTH_PX,
                height=150,
                title=alt.TitleParams(title, anchor="start", fontSize=11, offset=2),
            )
        )
    return figure(
        panels=panels,
        number="P3",
        title=(
            "The remainder is large mostly inside response-contract blocks, and mostly when "
            "frequency is high"
        ),
        subtitle=[
            (
                "Remainder is metered output minus planned output minus accepted balancing "
                "volume, half-hour means, 1 September 2025 to 31 August 2026. Lines: mean of "
                f"{FREQUENCY_BINS} equal-count bins of frequency deviation. Dots: a random "
                f"sample of {SCATTER_SAMPLE} half-hours per group. Response blocks come from "
                "matching NESO auction units to batteries by identifier, so they are candidates."
            )
        ],
        figure_planning=None,
    )


def behaviour_figure() -> alt.VConcatChart:
    """Draw P4: what each battery does, as shares of half-hours and accepted energy.

    Returns:
        The figure.
    """
    summary = pl.read_parquet(SUMMARY_PATH).with_columns(
        battery=pl.col("bmu_id").replace_strict(NAMES, return_dtype=pl.String)
    )
    order = [NAMES[b] for b in BATTERIES]

    kind_colours = {
        "Inside a response block (candidate)": ocf.DATA_BLUE,
        "Inside a reserve block (candidate)": ocf.DATA_PURPLE,
    }

    def bars(
        *, columns: dict[str, str], x_title: str, title: str, last: bool, percent: bool
    ) -> Layer:
        long = pl.concat(
            summary.select(
                "battery", value=pl.col(column) * (100.0 if percent else 1.0)
            ).with_columns(kind=pl.lit(label))
            for column, label in columns.items()
        )
        grouped = len(columns) > 1
        encodings = {
            "y": alt.Y("battery:N", sort=order, axis=alt.Axis(title=None)),
            "x": alt.X(
                "value:Q", axis=alt.Axis(title=x_title if last else None, labels=last, ticks=last)
            ),
            "color": alt.Color(
                "kind:N",
                scale=alt.Scale(domain=list(kind_colours), range=list(kind_colours.values())),
                legend=alt.Legend(title=None, columns=2, labelLimit=300),
            )
            if grouped
            else alt.value(ocf.BRAND_ORANGE),
        }
        if grouped:
            encodings["yOffset"] = alt.YOffset(
                "kind:N", scale=alt.Scale(domain=list(columns.values()))
            )
        chart = (
            alt.Chart(long, title=alt.TitleParams(title, anchor="start", fontSize=11, offset=2))
            .mark_bar(aria=False)
            .encode(**encodings)  # ty: ignore[unresolved-attribute]
        )
        text = chart.mark_text(align="left", dx=3, fontSize=9, aria=False).encode(
            text=alt.Text("value:Q", format=".1f"), color=alt.value(ocf.BLACK_1)
        )
        return cast(Layer, alt.layer(chart, text)).properties(
            width=PLOT_WIDTH_PX - 120, height=130 if grouped else 95
        )

    panels = [
        bars(
            columns={"share_with_acceptance": "Half-hours with a balancing acceptance"},
            x_title="",
            title="Share of half-hours with a balancing acceptance (% of the year)",
            last=False,
            percent=True,
        ),
        bars(
            columns={"mean_abs_accepted_mwh_when_accepted": "Mean absolute accepted energy"},
            x_title="",
            title="Mean absolute accepted energy in those half-hours (MWh)",
            last=False,
            percent=False,
        ),
        bars(
            columns={
                "share_in_response_block": "Inside a response block (candidate)",
                "share_in_reserve_block": "Inside a reserve block (candidate)",
            },
            x_title="Share of the year's half-hours (%)",
            title="Share of half-hours inside a contract block of a matched auction unit",
            last=True,
            percent=True,
        ),
    ]
    return figure(
        panels=panels,
        number="P4",
        title=(
            "All four batteries take balancing acceptances in about half of all half-hours, "
            "and they differ in how much of the year they are contracted for response"
        ),
        subtitle=[
            (
                "1 September 2025 to 31 August 2026. Acceptances are Elexon bid-offer "
                "acceptance levels (BOALF). Contract blocks come from NESO auction results "
                "matched to batteries by identifier only, so they are candidates."
            )
        ],
        figure_planning=None,
    )


EXAMPLE_PRICES: Final[tuple[float, ...]] = (62.0, 45.0, 38.0, 71.0, 55.0, 90.0)
"""Six invented day-ahead prices (£ per MWh), the worked example of the price rank."""


def price_rank_figure() -> alt.VConcatChart:
    """Draw P5: six invented prices, each labelled with its place in the cheapest-to-dearest order.

    Returns:
        The figure.
    """
    ranks = {price: rank for rank, price in enumerate(sorted(EXAMPLE_PRICES), start=1)}
    table = pl.DataFrame(
        {
            "half_hour": [f"Half-hour {i}" for i in range(1, len(EXAMPLE_PRICES) + 1)],
            "price": list(EXAMPLE_PRICES),
            "rank": [ranks[price] for price in EXAMPLE_PRICES],
        }
    ).with_columns(label=pl.format("rank {}", pl.col("rank")))
    order = table["half_hour"].to_list()
    bars = (
        alt.Chart(table)
        .mark_bar(color=ocf.DATA_BLUE, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "half_hour:N",
                sort=order,
                axis=alt.Axis(title="Half-hours of the example day, in time order", labelAngle=0),
            ),
            y=alt.Y("price:Q", axis=alt.Axis(title="Day-ahead price (£ per MWh)")),
        )
    )
    text = bars.mark_text(dy=-6, fontSize=11, aria=False).encode(
        text="label:N", color=alt.value(ocf.BLACK_1)
    )
    panel = cast(Layer, alt.layer(bars, text)).properties(width=PLOT_WIDTH_PX, height=160)
    return figure(
        panels=[panel],
        number="P5",
        title="A half-hour's price rank is its place in the day's cheapest-to-dearest ordering",
        subtitle=[
            (
                "Six invented day-ahead prices. Rank 1 is the cheapest half-hour of the day and "
                "the highest rank is the dearest. A real day has 48 half-hours."
            )
        ],
        figure_planning=None,
    )


def main() -> None:
    """Write the primer figures as SVGs under `docs/background/assets/` and PNG previews."""
    charts = {
        "primer_typical_day": typical_day_figure(),
        "primer_decomposition": decomposition_figure(),
        "primer_frequency": frequency_figure(),
        "primer_behaviour": behaviour_figure(),
        "primer_price_rank": price_rank_figure(),
    }
    BACKGROUND_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    previews = OUTPUT_DIR / "previews"
    previews.mkdir(exist_ok=True)
    for name, chart in charts.items():
        target = BACKGROUND_ASSETS_DIR / f"battery_pv_separation_{name}.svg"
        chart.save(target)
        chart.save(previews / f"{name}.png", scale_factor=1.3)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
