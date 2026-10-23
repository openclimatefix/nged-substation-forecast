# ruff: noqa: E501
"""Draw experiment 270's figures from the score tables (farm series are never drawn; farms pooled)."""

import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00270_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 320
ARM_LABELS: Final[dict[str, str]] = {
    "x2_plant_forecast_only": "X2: XGBoost on the plant forecast only",
    "p_ens": "P: plant refitted on ensemble weather",
    "p_cams": "P: plant refitted on observed weather",
    "pq_elev": "Plant, offsets by forecast and sun height",
    "pq_csi": "Plant, offsets by forecast and clear-sky index",
    "pq_spread": "Plant, offsets by forecast and ensemble spread",
    "x1_plant_inputs": "X1: XGBoost on the plant's inputs",
    "l1": "L1: + plant forecast",
    "l2": "L2: + ensemble spread",
    "l3": "L3: + lagged power",
    "l4": "L4: + hour and day of year",
    "l5": "L5: + humidity, longwave, rain, pressure",
    "l6": "L6: + wind, height, clear-sky index, lead",
    "gam": "G: additive model on the plant forecast",
}
SETTING_LABELS: Final[dict[str, str]] = {
    "own_history": "Own history",
    "zero_history": "Zero history",
}
SETTING_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE]


def _title(lines: list[str]) -> alt.TitleParams:
    lines = [part for line in lines for part in textwrap.wrap(line, 100)]
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


def _dots(
    *,
    data: pl.DataFrame,
    y: str,
    x_title: str,
    sort: list[str],
    colour: str,
    colour_domain: list[str],
    colours: list[str],
    height: int,
    title: list[str],
    column: str | None = None,
) -> alt.Chart:
    base: Any = alt.Chart(data)
    colour_enc = alt.Color(
        f"{colour}:N",
        scale=alt.Scale(domain=colour_domain, range=colours),
        legend=alt.Legend(title=None, orient="top"),
    )
    y_enc = alt.Y(f"{y}:N", sort=sort, title=None, axis=alt.Axis(labelLimit=300))
    zero_chart: Any = alt.Chart(pl.DataFrame({"zero": [0.0]}))
    rule = zero_chart.mark_rule(strokeDash=[4, 3], color="#444").encode(x="zero:Q")
    chart = (
        base.mark_rule(size=2).encode(
            x=alt.X("lo:Q", title=x_title),
            x2="hi:Q",
            y=y_enc,
            color=colour_enc,
            yOffset=f"{colour}:N",
        )
        + base.mark_point(filled=True, size=50).encode(
            x="value:Q", y=y_enc, color=colour_enc, yOffset=f"{colour}:N"
        )
        + rule
    )
    return chart.properties(width=PLOT_WIDTH, height=height, title=_title(title))


def headline(*, contrasts: pl.DataFrame) -> None:
    """Figure 1: change in quantile score against the plant, per arm and setting."""
    data = contrasts.filter(
        (pl.col("slice") == "all")
        & (pl.col("metric") == "score")
        & (pl.col("reference") == "plant")
        & pl.col("arm").is_in(list(ARM_LABELS))
    ).with_columns(
        label=pl.col("arm").replace_strict(ARM_LABELS),
        setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
        value=pl.col("percent_change"),
    )
    chart = _dots(
        data=data,
        y="label",
        x_title="Quantile score minus the plant's (%; more negative is better)",
        sort=list(ARM_LABELS.values()),
        colour="setting_label",
        colour_domain=list(SETTING_LABELS.values()),
        colours=SETTING_COLOURS,
        height=380,
        title=[
            "Figure 1: XGBoost on the plant's inputs alone beats the plant by about 3%; the other rungs add little.",
            "Dot: estimate. Line: 98.75% interval from resampling the 12 scored months. Dashed line: same as the plant. Metered farms A to F, 12 to 36 h, daylight.",
        ],
    )
    chart.save(DOCS_DIR / "figure_1_headline.svg")


def ladder(*, contrasts: pl.DataFrame) -> None:
    """Figure 2: each rung against the previous rung and against its shuffled control."""
    rows = []
    for k in range(2, 7):
        rows.append(("prev", f"L{k} minus L{k - 1}", f"l{k}", f"l{k - 1}"))
        rows.append(("control", f"L{k} minus L{k} with shuffled added columns", f"l{k}", f"s{k}"))
    parts = []
    for kind, label, arm, ref in rows:
        part = contrasts.filter(
            (pl.col("slice") == "all")
            & (pl.col("metric") == "score")
            & (pl.col("setting") == "own_history")
            & (pl.col("arm") == arm)
            & (pl.col("reference") == ref)
        ).with_columns(
            label=pl.lit(label),
            kind=pl.lit(
                "Against the previous rung" if kind == "prev" else "Against the shuffled control"
            ),
            value=pl.col("percent_change"),
        )
        parts.append(part)
    data = pl.concat(parts)
    chart = _dots(
        data=data,
        y="label",
        x_title="Change in quantile score (%; more negative is better)",
        sort=[r[1] for r in rows],
        colour="kind",
        colour_domain=["Against the previous rung", "Against the shuffled control"],
        colours=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
        height=330,
        title=[
            "Figure 2: only the lagged-power rung helps; the calendar rung hurts, and the shuffled control shows the fault is the calendar columns.",
            "Own history. Dot: estimate. Line: 98.75% interval from resampling months. Dashed line: no change.",
        ],
    )
    chart.save(DOCS_DIR / "figure_2_ladder.svg")


def location_and_width(*, contrasts: pl.DataFrame) -> None:
    """Figure 3: where the gain sits: the median, or the lower and upper tails."""
    metrics = {
        "ae_median": "Error of the median",
        "ae_mean": "Error of the mean forecast",
        "pin01": "Pinball loss, 0.01",
        "pin05": "Pinball loss, 0.05",
        "pin95": "Pinball loss, 0.95",
        "pin99": "Pinball loss, 0.99",
    }
    arms = {
        "x1_plant_inputs": "X1: plant's inputs",
        "l6": "L6: all inputs",
        "gam": "G: additive model",
    }
    data = contrasts.filter(
        (pl.col("slice") == "all")
        & (pl.col("setting") == "own_history")
        & (pl.col("reference") == "plant")
        & pl.col("metric").is_in(list(metrics))
        & pl.col("arm").is_in(list(arms))
    ).with_columns(
        label=pl.col("metric").replace_strict(metrics),
        arm_label=pl.col("arm").replace_strict(arms),
        value=pl.col("percent_change"),
    )
    chart = _dots(
        data=data,
        y="label",
        x_title="Loss minus the plant's (%; more negative is better)",
        sort=list(metrics.values()),
        colour="arm_label",
        colour_domain=list(arms.values()),
        colours=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN],
        height=300,
        title=[
            "Figure 3: XGBoost's gain is in the 0.05 and 0.95 tails and not in the median; only the additive model also wins at 0.99.",
            "Own history. Dot: estimate. Line: 98.75% interval from resampling months. Dashed line: same as the plant.",
        ],
    )
    chart.save(DOCS_DIR / "figure_3_location_width.svg")


def by_hour(*, contrasts: pl.DataFrame) -> None:
    """Figure 4: gain by hour of day (UTC, end of the half-hour)."""
    data = contrasts.filter(
        (pl.col("slice_kind") == "hour")
        & (pl.col("setting") == "own_history")
        & (pl.col("metric") == "score")
        & (pl.col("reference") == "plant")
        & pl.col("arm").is_in(["x1_plant_inputs", "l6"])
    ).with_columns(
        hour=pl.col("slice").cast(pl.Int64),
        arm_label=pl.col("arm").replace_strict(
            {"x1_plant_inputs": "X1: plant's inputs", "l6": "L6: all inputs"}
        ),
    )
    zero_chart: Any = alt.Chart(pl.DataFrame({"zero": [0.0]}))
    base: Any = alt.Chart(data)
    colour = alt.Color(
        "arm_label:N",
        scale=alt.Scale(
            domain=["X1: plant's inputs", "L6: all inputs"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    x = alt.X(
        "hour:Q",
        title="Hour of day (UTC)",
        scale=alt.Scale(domain=[5, 19]),
        axis=alt.Axis(values=[6, 9, 12, 15, 18]),
    )
    chart = (
        base.mark_errorband(opacity=0.2).encode(
            x=x,
            y=alt.Y("lo:Q", title="Quantile score minus the plant's (%)"),
            y2="hi:Q",
            color=colour,
        )
        + base.mark_line(point=True).encode(x=x, y="percent_change:Q", color=colour)
        + zero_chart.mark_rule(strokeDash=[4, 3], color="#444").encode(y="zero:Q")
    ).properties(
        width=PLOT_WIDTH + 100,
        height=220,
        title=_title(
            [
                "Figure 4: the gain is at low sun in the morning and evening, and absent around noon.",
                "Own history. Line: estimate. Band: 98.75% interval from resampling months. Dashed line: same as the plant. Daylight hours only.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_4_by_hour.svg")


def binned_gain() -> None:
    """Figure 5: the plant's score minus L6's, averaged in bins of three inputs."""
    rows = pl.read_parquet(common.DATA_DIR / "row_losses.parquet").filter(
        pl.col("setting") == "own_history"
    )
    wide = (
        rows.filter(pl.col("arm").is_in(["plant", "l6"]))
        .pivot(
            on="arm",
            index=["time_series_id", "time", "csi", "temperature_mean", "ghi_std"],
            values="score",
        )
        .with_columns(gain=pl.col("plant") - pl.col("l6"))
    )
    panels = []
    for column, title in (
        ("csi", "Clear-sky index"),
        ("temperature_mean", "Ensemble-mean air temperature (degrees C)"),
        ("ghi_std", "Ensemble spread of irradiance (W/m2)"),
    ):
        binned = (
            wide.with_columns(bin=(pl.col(column).rank("ordinal") * 10 // (pl.len() + 1)))
            .group_by("bin")
            .agg(x=pl.col(column).mean(), gain=pl.col("gain").mean())
        )
        base: Any = alt.Chart(binned)
        panels.append(
            base.mark_line(point=True, color=ocf.DATA_BLUE)
            .encode(x=alt.X("x:Q", title=title), y=alt.Y("gain:Q", title="Plant minus L6 (points)"))
            .properties(width=PLOT_WIDTH + 100, height=130)
        )
    chart = alt.vconcat(*panels).properties(
        title=_title(
            [
                "Figure 5: L6's gain over the plant, in points of quantile score (positive is better), by decile of three inputs.",
                "Own history; each dot is the mean over one tenth of the scored rows.",
            ]
        )
    )
    chart.save(DOCS_DIR / "figure_5_binned_gain.svg")


def main() -> None:
    """Draw every figure."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    contrasts = pl.read_csv(common.DATA_DIR / "contrasts.csv")
    headline(contrasts=contrasts)
    ladder(contrasts=contrasts)
    location_and_width(contrasts=contrasts)
    by_hour(contrasts=contrasts)
    binned_gain()


if __name__ == "__main__":
    main()
