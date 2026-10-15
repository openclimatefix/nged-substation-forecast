"""Draw experiment 00029's figures from the saved results, as SVG.

1. CRPS by lead band for each arm.
2. The planned contrasts with 98.75% intervals.
3. Recovery of tilt and DC capacity on synthetic plants.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00029-*/e00029_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00029_common import LEAD_BANDS, WORK_DIR

RESULTS_DIR: Final[Path] = WORK_DIR / "results"
CHART_DIR: Final[Path] = WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 520
WRAP_CHARACTERS: Final[int] = 100
SCOPE: Final[str] = (
    "6 solar farms; 53 forecast runs on Tuesdays, July 2025 to June 2026; pooled mixture of 51 "
    "ENS members, 3 seeds."
)
ARM_LABELS: Final[dict[str, str]] = {
    "base": "XGBoost stage 1 (experiment 00020 recipe)",
    "scipy": "SciPy-fitted physics stage 1",
    "torch": "PyTorch physics stage 1",
    "torch_hybrid": "PyTorch physics + XGBoost weather",
    "climatology": "Climatology",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "base": ocf.DATA_BLUE,
    "scipy": ocf.DATA_SKY,
    "torch": ocf.BRAND_ORANGE,
    "torch_hybrid": ocf.DATA_GREEN,
    "climatology": "#5a5a5a",
}
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "C1: PyTorch physics minus XGBoost",
    "C2": "C2: hybrid minus XGBoost",
    "C3": "C3: PyTorch minus SciPy physics",
    "C4": "C4: SciPy physics minus XGBoost",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """CRPS by lead band."""
    bands = [name for name, _, _ in LEAD_BANDS]
    data = pl.read_parquet(RESULTS_DIR / "by_band.parquet").with_columns(
        label=pl.col("arm").replace_strict(ARM_LABELS)
    )
    labels = list(ARM_LABELS.values())
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("band:N", sort=bands, title="Lead time", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "crps:Q",
                title="CRPS (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "label:N",
                sort=labels,
                scale=alt.Scale(domain=labels, range=list(ARM_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=400),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=240,
            title=_title(
                text="Every physics arm scores within 0.11 points of XGBoost at every lead",
                lines=[
                    SCOPE,
                    "Each line is the mean continuous ranked probability score of one arm.",
                ],
            ),
        )
    )


def figure_2() -> alt.TopLevelMixin:
    """The planned contrasts."""
    contrasts = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").with_columns(
        label=pl.col("contrast").replace_strict(CONTRAST_LABELS)
    )
    labels = contrasts.sort("contrast")["label"].to_list()
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#292b2b", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    threshold = (
        alt.Chart(pl.DataFrame({"x": [-0.05]}))
        .mark_rule(color=ocf.DATA_GREEN, strokeDash=[2, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    base = alt.Chart(contrasts).encode(
        y=alt.Y("label:N", sort=labels, title=None, axis=alt.Axis(labelLimit=330)),
        color=alt.condition(
            alt.datum.contrast == "C1", alt.value(ocf.BRAND_ORANGE), alt.value(ocf.DATA_BLUE)
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="CRPS difference (points of capacity; negative is better)")
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, threshold, bars, points).properties(
        width=PLOT_WIDTH,
        height=110,
        title=_title(
            text="No physics arm beats the XGBoost stage 1",
            lines=[
                SCOPE,
                (
                    "Dot: estimate. Line: 98.75% interval from re-drawing whole months (a 5% "
                    "level corrected for 4 planned contrasts). Black dashed line: no change. "
                    "Green dotted line: the -0.05 threshold for a gain."
                ),
            ],
        ),
    )


def figure_3() -> alt.TopLevelMixin:
    """Recovery of tilt and DC capacity on synthetic plants."""
    data = pl.read_parquet(WORK_DIR / "synthetic.parquet").with_columns(
        chain=pl.when(pl.col("version") == "same")
        .then(pl.lit("Same physics as the fit"))
        .otherwise(pl.lit("Different sky model (Hay-Davies)"))
    )
    panels = []
    for name, title, domain in (
        ("tilt", "Tilt (degrees)", [0, 70]),
        ("dc", "DC capacity (fraction of effective capacity)", [0.85, 1.45]),
    ):
        diagonal = (
            alt.Chart(pl.DataFrame({"a": domain, "b": domain}))
            .mark_line(color="#292b2b", strokeDash=[4, 3])
            .encode(x="a:Q", y="b:Q")  # ty: ignore[unresolved-attribute]
        )
        points = (
            alt.Chart(data)  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=40, opacity=0.8)
            .encode(
                x=alt.X(
                    f"truth_{name}:Q", title=f"True {title.lower()}", scale=alt.Scale(domain=domain)
                ),
                y=alt.Y(f"est_{name}:Q", title="Fitted", scale=alt.Scale(domain=domain)),
                color=alt.Color(
                    "chain:N",
                    scale=alt.Scale(
                        domain=["Same physics as the fit", "Different sky model (Hay-Davies)"],
                        range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                    ),
                    legend=alt.Legend(title=None, orient="top", labelLimit=400),
                ),
            )
        )
        panels.append(
            alt.layer(diagonal, points).properties(width=PLOT_WIDTH, height=190, title=title)
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="Tilt and DC capacity are recovered closely only when the chain matches",
            lines=[
                (
                    "48 synthetic plants at 6 farm locations, each with its own two-year window "
                    "and autocorrelated noise. Dashed line: fitted equals true."
                ),
            ],
        )
    )


def main() -> None:
    """Write the figures as SVG."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for number, chart in ((1, figure_1()), (2, figure_2()), (3, figure_3())):
        chart.save(CHART_DIR / f"figure_{number}.svg")


if __name__ == "__main__":
    main()
