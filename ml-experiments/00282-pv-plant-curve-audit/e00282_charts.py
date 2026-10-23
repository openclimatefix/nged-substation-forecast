"""Draw the anonymised figures of experiment 282 from the tables the earlier scripts wrote."""

from pathlib import Path
from typing import Final

import altair as alt
import e00282_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
COLOURS: Final[dict[str, str]] = {
    "Measured": "#222222",
    "Plant fit, per farm": ocf.DATA_BLUE,
    "Plant fit, fleet shape": ocf.DATA_SKY,
    "Stack's solar term": ocf.BRAND_ORANGE,
    "Pooled shape of the other five farms": ocf.DATA_PURPLE,
}
MODEL_LABELS: Final[dict[str, str]] = {
    "plant_fleet": "Plant fit, fleet shape",
    "plant_per_farm": "Plant fit, per farm",
    "stack_term": "Stack's solar term",
    "pooled_shape": "Pooled shape of the other five farms",
}
ENGINES: Final[tuple[str, ...]] = ("plant_per_farm", "stack_term")


def _title(lines: list[str]) -> alt.TitleParams:
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


def _colour(*, models: list[str], title: str = "") -> alt.Color:
    labels = [MODEL_LABELS[m] for m in models]
    return alt.Color(
        "label:N",
        scale=alt.Scale(domain=labels, range=[COLOURS[label] for label in labels]),
        legend=alt.Legend(orient="top", title=title or None),
    )


def _labelled(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(label=pl.col("model").replace_strict(MODEL_LABELS))


def residual_by_bin() -> None:
    """Figure 1: measured minus modelled output by irradiance bin, one panel per farm."""
    table = _labelled(
        pl.read_csv(common.DATA_DIR / "check1_bins.csv").filter(
            pl.col("model").is_in(ENGINES) & (pl.col("n") >= common.MIN_BIN_ROWS)
        )
    )
    base = alt.Chart(table).encode(
        x=alt.X(
            "poa_bin:Q",
            title="Irradiance on the fleet plane (W/m²)",
            scale=alt.Scale(domain=[0, 1100]),
        ),
        color=_colour(models=list(ENGINES)),
    )
    band = base.mark_errorband(opacity=0.25).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("low:Q", title="Measured minus modelled (points)"), y2="high:Q"
    )
    line = base.mark_line().encode(y="mean:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(table).mark_rule(color="#888888").encode(y=alt.datum(0.0))  # ty: ignore[unresolved-attribute]
    limits = alt.layer(
        *[
            alt.Chart(table)
            .mark_rule(color="#888888", strokeDash=[4, 3])
            .encode(y=alt.datum(level))  # ty: ignore[unresolved-attribute]
            for level in (-5.0, 5.0)
        ]
    )
    chart = (
        alt.layer(band, line, zero, limits)
        .properties(width=190, height=150)
        .facet(facet=alt.Facet("site:N", title=None), columns=3)
        .properties(
            title=_title(
                [
                    "Measured farm output minus the engines' output, by irradiance, farms A to F.",
                    "Engines tuned before 2025-09, scored 2025-09 to 2026-08. Line: mean.",
                    "Shade: 95% interval from resampling months. Dashed: the 5-point departure.",
                    "Below zero: the engine predicts more than the farm produced.",
                ]
            )
        )
    )
    chart.save(DOCS_DIR / "figure_1_residual_by_irradiance.svg")


def flat_top() -> None:
    """Figure 2: the 99th percentile of output where the plane's irradiance is above 900 W/m²."""
    table = pl.read_csv(common.DATA_DIR / "check2_flat_top.csv")
    long = table.unpivot(
        index="site", on=[c for c in table.columns if c.startswith("p99_")]
    ).with_columns(
        model=pl.col("variable").str.replace("p99_", ""),
    )
    long = long.with_columns(
        label=pl.col("model").replace_strict({"output": "Measured", **MODEL_LABELS})
    ).filter(pl.col("label") != MODEL_LABELS["pooled_shape"])
    domain = [
        "Measured",
        *[MODEL_LABELS[m] for m in ("plant_fleet", "plant_per_farm", "stack_term")],
    ]
    chart = (
        alt.Chart(long)
        .mark_point(filled=True, size=90)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("site:N", title="Farm"),
            y=alt.Y(
                "value:Q",
                title="99th percentile of output over capacity",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=domain, range=[COLOURS[d] for d in domain]),
                legend=alt.Legend(orient="top", title=None),
            ),
            xOffset="label:N",
        )
        .properties(
            width=CONTENT_WIDTH_PX - 120,
            height=200,
            title=_title(
                [
                    "Flat-top level: the 99th percentile of output over capacity on half-hours",
                    "when the fleet plane receives at least 900 W/m², scored year, farms A to F.",
                ]
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_2_flat_top.svg")


def sun_and_temperature() -> None:
    """Figure 3: residuals at low sun, at clear midday, and by air-temperature tercile."""
    parts = []
    for name, slice_label in (
        ("check3_low_sun", "Sun under 15 degrees"),
        ("check3_clear_midday", "Clear sky, sun over 40 degrees"),
    ):
        parts.append(
            pl.read_csv(common.DATA_DIR / f"{name}.csv").with_columns(panel=pl.lit(slice_label))
        )
    temperature = pl.read_csv(common.DATA_DIR / "check4_temperature.csv").with_columns(
        panel=pl.lit("Fleet plane 700 to 900 W/m², by air temperature: ") + pl.col("temp_tercile")
    )
    table = _labelled(
        pl.concat([*parts, temperature.drop("temp_tercile")], how="vertical_relaxed").filter(
            pl.col("model").is_in(ENGINES)
        )
    )
    panels = sorted(table["panel"].unique().to_list())
    base = alt.Chart(table).encode(
        y=alt.Y("site:N", title="Farm"),
        color=_colour(models=list(ENGINES)),
        yOffset="label:N",
    )
    chart = (
        alt.layer(
            base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("low:Q", title="Measured minus modelled (points of capacity)"), x2="high:Q"
            ),
            base.mark_point(filled=True, size=40).encode(x="mean:Q"),  # ty: ignore[unresolved-attribute]
            alt.Chart(table).mark_rule(color="#888888").encode(x=alt.datum(0.0)),  # ty: ignore[unresolved-attribute]
        )
        .properties(width=250, height=110)
        .facet(facet=alt.Facet("panel:N", title=None, sort=panels), columns=2)
        .properties(
            title=_title(
                [
                    "Residuals at low sun, at clear midday, and by air temperature, scored year.",
                    "Dot: mean. Line: 95% interval from resampling months.",
                    "Below zero: the engine predicts more than the farm produced.",
                ]
            )
        )
    )
    chart.save(DOCS_DIR / "figure_3_sun_and_temperature.svg")


def orientation() -> None:
    """Figure 4: the tilt that fits each farm best, beside the engines' tilts."""
    table = pl.read_csv(common.DATA_DIR / "check5_orientation.csv")
    best = (
        table.filter(pl.col("tilt_deg") >= 0)
        .sort("tuning_sse")
        .group_by("site", maintain_order=True)
        .first()
        .sort("site")
    )
    fleet = table.filter((pl.col("tilt_deg") == 35.0) & (pl.col("azimuth_deg") == 180.0)).select(
        "site", fleet_fraction=pl.col("scored_fraction_fitted")
    )
    east_west = table.filter(pl.col("tilt_deg") == -1.0).select(
        "site", east_west_fraction=pl.col("scored_fraction_fitted")
    )
    out = best.join(fleet, on="site").join(east_west, on="site")
    out.write_csv(common.DATA_DIR / "check5_summary.csv")
    long = out.select(
        "site",
        pl.col("tilt_deg").alias("Best free tilt"),
    ).unpivot(index="site")
    rule = (
        alt.Chart(pl.DataFrame({"y": [35.0]}))
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    points = (
        alt.Chart(long)
        .mark_point(filled=True, size=90, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("site:N", title="Farm"),
            y=alt.Y("value:Q", title="Panel tilt (degrees)", scale=alt.Scale(domain=[0, 60])),
        )
    )
    (rule + points).properties(
        width=CONTENT_WIDTH_PX - 120,
        height=180,
        title=_title(
            [
                "Tilt that fits each farm best on the tuning months (5 degree grid).",
                "Dot: best tilt. Orange dashed line: the engines' assumed tilt (35 degrees).",
            ]
        ),
    ).save(DOCS_DIR / "figure_4_orientation.svg")


def seasonal() -> None:
    """Figure 5: measured over modelled output by calendar month, relative to each farm's year."""
    table = _labelled(pl.read_csv(common.DATA_DIR / "check6_seasonal_ratio.csv"))
    lines = (
        alt.Chart(table)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("calendar_month:O", title="Calendar month", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "relative_ratio:Q",
                title="Measured / modelled, relative",
                scale=alt.Scale(domain=[0.4, 1.3]),
            ),
            color=alt.Color("site:N", legend=alt.Legend(orient="top", title="Farm")),
        )
    )
    one = alt.Chart(table).mark_rule(color="#888888").encode(y=alt.datum(1.0))  # ty: ignore[unresolved-attribute]
    alt.layer(lines, one).properties(width=CONTENT_WIDTH_PX - 120, height=170).facet(
        row=alt.Row("label:N", title=None, sort=["Stack's solar term", "Plant fit, per farm"])
    ).properties(
        title=_title(
            [
                "Seasonal shape error of each engine at the six farms, all years of data.",
                "Clear half-hours with 400 to 900 W/m² on the fleet plane. Above 1: the farm",
                "produced more than the engine implies, relative to the rest of the year.",
            ]
        ),
    ).save(DOCS_DIR / "figure_5_seasonal_shape.svg")


def fix_test() -> None:
    """Figure 6: fraction of output fitted per farm for each shape."""
    table = _labelled(pl.read_csv(common.DATA_DIR / "check7_per_farm.csv").rename({"arm": "model"}))
    models = ["plant_fleet", "plant_per_farm", "stack_term", "pooled_shape"]
    (
        alt.Chart(table)
        .mark_point(filled=True, size=80)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("site:N", title="Farm"),
            y=alt.Y(
                "fraction_fitted:Q",
                title="Fraction of output fitted (higher is better)",
                scale=alt.Scale(zero=False),
            ),
            color=_colour(models=models),
            xOffset="label:N",
        )
        .properties(
            width=CONTENT_WIDTH_PX - 120,
            height=200,
            title=_title(
                [
                    "Fraction of output fitted, scored year, one scale per farm for every shape.",
                    "The pooled shape is fitted to the other five farms' tuning months only.",
                ]
            ),
        )
    ).save(DOCS_DIR / "figure_6_fix_test.svg")


def main() -> None:
    """Draw every figure."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    residual_by_bin()
    flat_top()
    sun_and_temperature()
    orientation()
    seasonal()
    fix_test()


if __name__ == "__main__":
    main()
