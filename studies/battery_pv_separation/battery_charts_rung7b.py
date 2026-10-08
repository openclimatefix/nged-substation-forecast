"""Draw the battery study's rung 7b figures (20 and 21).

Every figure is built from `rung7b_fits.parquet`, which `battery_rung7b.py` saved, and the solar
study's `stage3_separations.parquet`. The BMUs are public supplier and virtual BMUs, so the figures
name them and show megawatts.

Run after `battery_rung7b.py`:
`uv run python studies/battery_pv_separation/battery_charts_rung7b.py`.
"""

from typing import Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from battery_charts import ASSETS_DIR, PANEL_HEIGHT_PX, PLOT_WIDTH_PX, Layer
from battery_inputs import OUTPUT_DIR
from battery_rung7 import CLOUD_SIGNAL_BMUS
from battery_rung7b import POWERS_MW
from studies.charts import figure
from studies.sources import SOLAR_BMU_DISAGGREGATION_DIR

REAL_LABEL: Final[str] = "Real output"
REPLICA_LABEL: Final[str] = "Calendar replica"
SYMLOG_CONSTANT: Final[float] = 2.0
LARGEST_COUNT: Final[int] = 4
"""The number of BMUs with the largest fitted capacity at no battery that Figure 21 summarises."""


def _x_axis() -> alt.X:
    """Return the battery power axis, stretched near 0."""
    return alt.X(
        "power_mw:Q",
        title="Assumed battery power (MW; 2 hours of energy; 0 is no battery)",
        scale=alt.Scale(type="symlog", constant=SYMLOG_CONSTANT, domain=[0, max(POWERS_MW)]),
        axis=alt.Axis(values=list(POWERS_MW)),
    )


def solar_study_totals(*, bmus: tuple[str, ...] | None) -> dict[str, float]:
    """Return the solar study's totals over some BMUs.

    Args:
        bmus: The BMUs to sum over; None for all 25.

    Returns:
        The regional difference separation's and the physical fit's total AC capacity in
        megawatts, keyed by a legend label.
    """
    table = pl.read_parquet(SOLAR_BMU_DISAGGREGATION_DIR / "stage3_separations.parquet")
    if bmus is not None:
        table = table.filter(pl.col("bmu").is_in(bmus))
    return {
        "Solar study, difference separation": float(table["regional_difference_capacity_mw"].sum()),
        "Solar study, physical fit": float(table["physical_ac_mw"].sum()),
    }


def _total_panel(*, fits: pl.DataFrame, bmus: tuple[str, ...] | None, title: str) -> Layer:
    """Draw real and replica total capacity against power, with the solar study's totals.

    Args:
        fits: `rung7b_fits.parquet`.
        bmus: The BMUs to sum over; None for all 25.
        title: The panel's title.

    Returns:
        The panel.
    """
    frame = fits if bmus is None else fits.filter(pl.col("bmu").is_in(bmus))
    totals = (
        frame.group_by("source", "power_mw")
        .agg(pl.col("fitted_ac_mw").sum())
        .with_columns(
            line=pl.col("source").replace_strict(
                {"real": REAL_LABEL, "replica": REPLICA_LABEL}, return_dtype=pl.String
            )
        )
    )
    lines = (
        alt.Chart(totals)
        .mark_line(point=True, strokeWidth=1.6, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=_x_axis(),
            y=alt.Y("fitted_ac_mw:Q", title="Total fitted solar capacity (MW)"),
            color=alt.Color(
                "line:N",
                scale=alt.Scale(
                    domain=[REAL_LABEL, REPLICA_LABEL], range=[ocf.BRAND_ORANGE, ocf.ENSEMBLE_LINE]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    study = solar_study_totals(bmus=bmus)
    top = float(totals["fitted_ac_mw"].max())  # ty: ignore[invalid-argument-type]
    layers: list[alt.Chart] = []
    for index, ((name, value), dash) in enumerate(zip(study.items(), ([], [4, 3]), strict=True)):
        label = pl.DataFrame(
            {
                "y": [value],
                "label_y": [top * (0.62 - 0.1 * index)],
                "label": [f"{name}: {value:,.0f} MW ({'solid' if not dash else 'dashed'} line)"],
            }
        )
        layers += [
            alt.Chart(label)
            .mark_rule(color=ocf.DATA_BLUE, strokeDash=dash, strokeWidth=1.0, aria=False)
            .encode(y="y:Q"),  # ty: ignore[unresolved-attribute]
            alt.Chart(label)
            .mark_text(align="left", dx=4, fontSize=10, color=ocf.DATA_BLUE, aria=False)
            .encode(x=alt.value(4), y="label_y:Q", text="label:N"),  # ty: ignore[unresolved-attribute]
        ]
    return cast(Layer, alt.layer(*layers, lines)).properties(
        width=PLOT_WIDTH_PX - 60,
        height=PANEL_HEIGHT_PX * 1.5,
        title=alt.TitleParams(title, anchor="start"),
    )


def total_figure(*, number: int) -> alt.VConcatChart:
    """Draw total fitted solar capacity against assumed battery power, with a calendar baseline.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    fits = pl.read_parquet(OUTPUT_DIR / "rung7b_fits.parquet")
    real = fits.filter(pl.col("source") == "real")
    replica = fits.filter(pl.col("source") == "replica")

    def total(frame: pl.DataFrame, bmus: tuple[str, ...] | None, power: float) -> float:
        part = frame if bmus is None else frame.filter(pl.col("bmu").is_in(bmus))
        return float(part.filter(pl.col("power_mw") == power)["fitted_ac_mw"].sum())

    low, high = min(POWERS_MW), max(POWERS_MW)
    study = solar_study_totals(bmus=None)
    panels = [
        _total_panel(fits=fits, bmus=None, title="All 25 aggregate BMUs"),
        _total_panel(fits=fits, bmus=CLOUD_SIGNAL_BMUS, title="The nine cloud-signal BMUs"),
    ]
    return figure(
        panels=panels,
        number=number,
        title=(
            f"With a calendar baseline, the fitted solar capacity of the 25 BMUs is "
            f"{total(real, None, low):,.0f} MW with no battery and {total(real, None, high):,.0f} "
            f"MW with a {high:.0f} MW battery, against the solar study's "
            f"{study['Solar study, difference separation']:,.0f} MW and "
            f"{study['Solar study, physical fit']:,.0f} MW"
        ),
        subtitle=[
            (
                "Sum of the fitted AC capacity of the four fleet curves (regional CAMS sky), "
                "with a seasonal calendar baseline and a battery of the assumed power and 2 hours "
                "of energy, fitted to half-hourly levels. The grey line is the same fit to each "
                "BMU's calendar replica, which has no cloud, so solar fitted there could be shape "
                "alone "
                f"(all 25 BMUs: {total(replica, None, low):,.0f} MW with no battery and "
                f"{total(replica, None, high):,.0f} MW at {high:.0f} MW). The blue lines are the "
                "solar study's totals over the same BMUs. No ground truth exists. The horizontal "
                "axis is stretched near 0."
            )
        ],
        figure_planning=None,
    )


def per_bmu_figure(*, number: int) -> alt.VConcatChart:
    """Draw the fitted solar capacity of each cloud-signal BMU against assumed battery power.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    fits = pl.read_parquet(OUTPUT_DIR / "rung7b_fits.parquet").filter(
        pl.col("bmu").is_in(CLOUD_SIGNAL_BMUS)
    )
    real = fits.filter(pl.col("source") == "real").with_columns(line=pl.col("bmu"))
    replica = fits.filter(pl.col("source") == "replica")
    colours = [
        ocf.BRAND_ORANGE,
        ocf.DATA_BLUE,
        ocf.DATA_PURPLE,
        ocf.DATA_GREEN,
        ocf.DATA_SKY,
        ocf.DATA_AMBER,
        ocf.DATA_DEEP_TEAL,
        ocf.DATA_MAGENTA,
        ocf.DATA_BURNT_ORANGE,
    ]
    y = alt.Y(
        "fitted_ac_mw:Q",
        title="Fitted solar capacity (MW)",
        scale=alt.Scale(type="symlog", constant=1.0),
    )
    grey = (
        alt.Chart(replica)
        .mark_line(color=ocf.BLACK_1, opacity=0.35, strokeWidth=1.0, aria=False)
        .encode(x=_x_axis(), y=y, detail="bmu:N")  # ty: ignore[unresolved-attribute]
    )
    coloured = (
        alt.Chart(real)
        .mark_line(point=True, strokeWidth=1.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=_x_axis(),
            y=y,
            color=alt.Color(
                "bmu:N",
                scale=alt.Scale(domain=[*CLOUD_SIGNAL_BMUS], range=colours),
                legend=alt.Legend(title=None, columns=3, symbolLimit=0),
            ),
        )
    )
    panel = cast(Layer, alt.layer(grey, coloured)).properties(
        width=PLOT_WIDTH_PX - 60, height=PANEL_HEIGHT_PX * 3
    )
    at_zero = real.filter(pl.col("power_mw") == 0.0).sort("fitted_ac_mw", descending=True)
    largest = at_zero["bmu"].head(LARGEST_COUNT).to_list()
    spread = max(
        abs(
            row["fitted_ac_mw"] / at_zero.filter(pl.col("bmu") == row["bmu"])["fitted_ac_mw"][0] - 1
        )
        for row in real.filter(pl.col("bmu").is_in(largest)).iter_rows(named=True)
    )
    largest_replicas = replica.filter(pl.col("bmu").is_in(largest))
    replica_growth = float(
        largest_replicas.filter(pl.col("power_mw") == max(POWERS_MW))["fitted_ac_mw"].sum()
    ) / float(largest_replicas.filter(pl.col("power_mw") == 0.0)["fitted_ac_mw"].sum())
    return figure(
        panels=[panel],
        number=number,
        title=(
            f"With a calendar baseline, the fitted solar capacity of the four largest "
            f"BMUs stays within {spread:.0%} of its no-battery value at every assumed battery "
            f"size, while their calendar replicas' capacity grows {replica_growth:.1f} times"
        ),
        subtitle=[
            (
                "The nine aggregate BMUs in which the solar study found a cloud-correlated "
                "component. Each coloured line is one BMU's real output, fitted with solar (four "
                "fleet curves, regional CAMS sky), a seasonal calendar baseline, and a battery of "
                "the assumed power and 2 hours of energy. Each grey line is the same fit to that "
                "BMU's calendar replica, which has no cloud. No ground truth exists for these "
                "BMUs. Both axes are stretched near 0."
            )
        ],
        figure_planning=None,
    )


def main() -> None:
    """Write figures 20 and 21 as SVGs under `docs/studies/assets/` and PNG previews."""
    charts = {
        "rung7b_total": total_figure(number=20),
        "rung7b_per_bmu": per_bmu_figure(number=21),
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
