"""Print the numbers of experiment 206 and draw its figures as SVG into the docs folder.

Only Primaries (D-labels) and anonymised farms (A to F, as ratios) appear. Afterwards optimise each
SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00206-*/e00206_report.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00206_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00206-dp-meter-regime-steps-e1-farm-outages-in-primary"
)
PNG_DIR: Final[Path] = Path("/tmp/e00206_png")
WIDTH: Final[int] = 600
WRAP: Final[int] = 100
CLASSES: Final[list[str]] = [
    "Assigned MW Primary",
    "Other MW Primary",
    "MVA Primary (size only)",
    "Bulk supply point D13 (post hoc)",
]
CLASS_COLOURS: Final[list[str]] = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.DATA_GREEN]


def load() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return the outages, the tested shifts (with a target class), and the placebo table."""
    outages = pl.read_parquet(common.WORK_DIR / "outages.parquet")
    shifts = pl.read_parquet(common.WORK_DIR / "shifts.parquet").filter(pl.col("tested"))
    shifts = shifts.with_columns(
        target=pl.when(pl.col("primary") == "D13")
        .then(pl.lit(CLASSES[3]))
        .when(pl.col("units") == "MVA")
        .then(pl.lit(CLASSES[2]))
        .when(pl.col("assigned"))
        .then(pl.lit(CLASSES[0]))
        .otherwise(pl.lit(CLASSES[1])),
        label=pl.col("farm") + " " + pl.col("kind") + " " + pl.col("start").dt.strftime("%b %Y"),
    )
    placebo = pl.read_parquet(common.WORK_DIR / "placebo.parquet").filter(pl.col("tested"))
    return outages, shifts, placebo


def print_numbers(*, outages: pl.DataFrame, shifts: pl.DataFrame, placebo: pl.DataFrame) -> None:
    """Print the numbers the write-up quotes."""
    with pl.Config(tbl_rows=80, float_precision=3, tbl_cols=16, tbl_width_chars=220):
        print(outages.group_by("farm", "kind").len().sort("farm", "kind"))
        print(
            "outages found",
            outages.height,
            "of which zero",
            outages.filter(pl.col("kind") == "zero").height,
        )
        decision = shifts.filter(
            pl.col("target") == CLASSES[0],
            pl.col("primary").is_in(common.DECISION_SITES),
            pl.col("kind") == "zero",
        )
        print("decision-set zero outages tested at assigned Primary:")
        print(
            decision.select(
                "farm",
                "start",
                "primary",
                "farm_p99_mw",
                "shift_mw",
                "ratio_more_pv",
                "ratio_lo",
                "ratio_hi",
                "shows_more_pv",
            )
        )
        print(
            "share showing the shift:",
            decision["shows_more_pv"].sum(),
            "of",
            decision.height,
        )
        for kind in ("zero", "missing"):
            for target in CLASSES:
                sub = shifts.filter(pl.col("kind") == kind, pl.col("target") == target)
                print(
                    kind,
                    target,
                    "tested",
                    sub.height,
                    "more_pv",
                    sub["shows_more_pv"].sum(),
                    "less_pv",
                    sub["shows_less_pv"].sum(),
                    "either",
                    sub["shows_either"].sum(),
                )
        print(
            placebo.group_by("farm")
            .agg(
                n=pl.len(),
                false_more=pl.col("shows_more_pv").sum(),
                false_less=pl.col("shows_less_pv").sum(),
                q05=pl.col("ratio_more_pv").quantile(0.05),
                q50=pl.col("ratio_more_pv").median(),
                q95=pl.col("ratio_more_pv").quantile(0.95),
            )
            .sort("farm")
        )
        print(
            "placebo false positives (either direction):",
            placebo["shows_more_pv"].sum() + placebo["shows_less_pv"].sum(),  # ty: ignore[unsupported-operator]
            "of",
            placebo.height,
        )
        print(
            shifts.filter(pl.col("target") == CLASSES[0])
            .select(
                "farm",
                "kind",
                "start",
                "end",
                "primary",
                "km",
                "n_sunny_in",
                "n_sunny_ref",
                "shift_mw",
                "ratio_more_pv",
                "ratio_lo",
                "ratio_hi",
            )
            .sort("farm", "start")
        )


def save_chart(chart: alt.TopLevelMixin, *, name: str) -> None:
    """Write the chart as an SVG into the docs folder and as a PNG for checking."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)


def chart_title(*, text: str, subtitle: str) -> alt.TitleParams:
    """Return a left-aligned title with the subtitle wrapped to the plot width."""
    return alt.TitleParams(
        text=text, subtitle=textwrap.wrap(subtitle, WRAP), anchor="start", subtitleColor="#444444"
    )


def figure_forest(*, shifts: pl.DataFrame) -> alt.TopLevelMixin:
    """Return each tested outage's shift in nearby substations, over the farm's output."""
    x_scale = alt.Scale(domain=[-2.0, 2.5], clamp=True)
    data = shifts.with_columns(
        low=pl.col("ratio_lo").clip(-2.0, 2.5), high=pl.col("ratio_hi").clip(-2.0, 2.5)
    )
    order = data.sort("farm", "start")["label"].unique(maintain_order=True).to_list()
    band = (
        alt.Chart(
            pl.DataFrame({"low": [common.SHIFT_RATIO_LOW], "high": [common.SHIFT_RATIO_HIGH]})
        )
        .mark_rect(color=ocf.DATA_GREEN, opacity=0.2)
        .encode(x=alt.X("low:Q", scale=x_scale), x2="high:Q")  # ty: ignore[unresolved-attribute]
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#808080").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None),
        yOffset=alt.YOffset("primary:N"),
        color=alt.Color(
            "target:N",
            scale=alt.Scale(domain=CLASSES, range=CLASS_COLOURS),
            legend=alt.Legend(title=None, orient="top", columns=1),
        ),
    )
    ranges = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "low:Q",
            scale=x_scale,
            title="Shift towards more solar, over the farm's 99th-percentile output",
        ),
        x2="high:Q",
    )
    points = base.mark_point(filled=True, size=40).encode(x=alt.X("ratio_more_pv:Q", scale=x_scale))  # ty: ignore[unresolved-attribute]
    return alt.layer(band, zero, ranges, points).properties(
        width=WIDTH,
        height=520,
        title=chart_title(
            text="Almost no nearby substation shifts by the 0.5 to 1.5 farm outputs predicted",
            subtitle=(
                "Each row is one farm outage; each dot is one nearby substation, with a 95%"
                " interval over days. Green band: the pre-registered shift of 0.5 to 1.5. Farms"
                " A to F are anonymised; Primaries and the bulk supply point are not."
            ),
        ),
    )


def figure_placebo(*, shifts: pl.DataFrame, placebo: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the placebo spread per farm with the real shifts at the assigned Primary."""
    spread = (
        placebo.group_by("farm")
        .agg(
            low=pl.col("ratio_more_pv").quantile(0.05),
            high=pl.col("ratio_more_pv").quantile(0.95),
        )
        .with_columns(low=pl.col("low").clip(-2.0, 2.5), high=pl.col("high").clip(-2.0, 2.5))
    )
    real = shifts.filter(pl.col("target") == CLASSES[0], pl.col("kind") == "zero")
    scale = alt.Scale(domain=[-2.0, 2.5], clamp=True)
    bars = (
        alt.Chart(spread)
        .mark_bar(color=ocf.DATA_BLUE_LIGHT, height=14)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("farm:N", title="Farm"),
            x=alt.X("low:Q", scale=scale, title="Shift towards more solar, over the farm's output"),
            x2="high:Q",
        )
    )
    dots = (
        alt.Chart(real)
        .mark_point(filled=True, size=70, color=ocf.BRAND_ORANGE)
        .encode(y="farm:N", x=alt.X("ratio_more_pv:Q", scale=scale))  # ty: ignore[unresolved-attribute]
    )
    return (bars + dots).properties(
        width=WIDTH,
        height=170,
        title=chart_title(
            text="The test's noise is far smaller than the shift it looks for, except at farm D",
            subtitle=(
                "Bar: 5th to 95th percentile of the shift in windows with no outage, at the farm's"
                " assigned Primary. Dot: a zero-output outage at the same Primary. Farm D's output"
                " is 1.2 MW, so the same noise in MW is a larger share of it."
            ),
        ),
    )


def main() -> None:
    """Print the numbers and write the figures."""
    outages, shifts, placebo = load()
    print_numbers(outages=outages, shifts=shifts, placebo=placebo)
    save_chart(figure_forest(shifts=shifts), name="figure_1")
    save_chart(figure_placebo(shifts=shifts, placebo=placebo), name="figure_2")


if __name__ == "__main__":
    main()
