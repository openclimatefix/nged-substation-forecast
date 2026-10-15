"""Draw experiment 00020's figures from the saved results, as SVG.

1. The planned CRPS contrasts with 99.58% intervals, one panel per technology.
2. Mean CRPS of each arm with 95% intervals, one panel per technology.
3. Coverage of the 5% to 95% band by lead band, for the arms with and without the spread factor.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00020-*/e00020_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00020_common import LEAD_BANDS, WORK_DIR

RESULTS_DIR: Final[Path] = WORK_DIR / "results"
CHART_DIR: Final[Path] = WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 520
WRAP_CHARACTERS: Final[int] = 100
SCOPE: Final[str] = (
    "6 solar farms and 3 wind farms; 53 forecast runs on Tuesdays, July 2025 to June 2026; "
    "pooled mixture of 51 ENS members, 3 seeds."
)
ARM_LABELS: Final[dict[str, str]] = {
    "base": "Production recipe (ERA5-trained, pooled mixture)",
    "cams": "CAMS-trained",
    "cams_s0": "CAMS-trained + stage 0",
    "margin": "Power-curve base margin",
    "base_spread": "Spread factor",
    "full": "Full stack",
    "full_no_cams": "Full stack without CAMS-trained",
    "full_no_s0": "Full stack without stage 0",
    "climatology": "Climatology",
}
CONTRAST_LABELS: Final[dict[str, str]] = {
    "S1": "Full stack minus production recipe",
    "S2": "CAMS-trained minus recipe",
    "S3": "Stage 0 added to CAMS-trained",
    "S4": "Spread factor added to recipe",
    "S5": "Drop CAMS-trained from full stack",
    "S6": "Drop stage 0 from full stack",
    "S7": "Drop spread factor from full stack",
    "W1": "Full stack minus production recipe",
    "W2": "Base margin added to recipe",
    "W3": "Spread factor added to recipe",
    "W4": "Drop base margin from full stack",
    "W5": "Drop spread factor from full stack",
}
COVERAGE_ARMS: Final[tuple[str, ...]] = ("base", "base_spread", "full")
COVERAGE_COLOURS: Final[tuple[str, ...]] = ("#5a5a5a", ocf.DATA_BLUE, ocf.BRAND_ORANGE)
TOLERANCE: Final[float] = 2.5
Y_SCALE: Final[alt.Scale] = alt.Scale(domain=[84, 100])


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """The planned contrasts, one panel per technology."""
    contrasts = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").with_columns(
        label=pl.col("contrast").replace_strict(CONTRAST_LABELS)
    )
    panels = []
    for technology, prefix in (("wind", "W"), ("solar", "S")):
        part = contrasts.filter(pl.col("technology") == technology)
        order = [f"{prefix}{i}" for i in range(1, part.height + 1)]
        part = part.with_columns(
            label=pl.col("contrast") + ": " + pl.col("label"),
        )
        labels = part.sort("contrast")["label"].to_list()
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
        base = alt.Chart(part).encode(
            y=alt.Y("label:N", sort=labels, title=None, axis=alt.Axis(labelLimit=330)),
            color=alt.condition(
                alt.datum.contrast == order[0],
                alt.value(ocf.BRAND_ORANGE),
                alt.value(ocf.DATA_BLUE),
            ),
        )
        points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="CRPS difference (points of capacity; negative = better)",
            )
        )
        bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, threshold, bars, points).properties(
                width=PLOT_WIDTH, height=24 * part.height, title=f"{technology.title()} farms"
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="The stack helps wind and does not help solar",
            lines=[
                SCOPE,
                (
                    "Dot: estimate. Line: 99.58% interval from re-drawing whole months and seeds "
                    "(a 5% level corrected for 12 planned contrasts). Orange: the headline "
                    "contrast. Black dashed line: no change. Green dotted line: the -0.05 "
                    "threshold for a gain."
                ),
            ],
        )
    )


def figure_2() -> alt.TopLevelMixin:
    """Mean CRPS of each arm with 95% intervals, one panel per technology."""
    absolute = pl.read_parquet(RESULTS_DIR / "absolute.parquet").with_columns(
        label=pl.col("arm").replace_strict(ARM_LABELS)
    )
    panels = []
    for technology in ("wind", "solar"):
        part = absolute.filter(pl.col("technology") == technology)
        labels = [ARM_LABELS[a] for a in ARM_LABELS if a in set(part["arm"])]
        base = alt.Chart(part).encode(
            y=alt.Y("label:N", sort=labels, title=None, axis=alt.Axis(labelLimit=330)),
        )
        points = base.mark_point(filled=True, size=60, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("value:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=False))
        )
        bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower_95:Q", x2="upper_95:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(bars, points).properties(
                width=PLOT_WIDTH, height=24 * part.height, title=f"{technology.title()} farms"
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="Arms differ by hundredths of a point; farms and months differ by far more",
            lines=[
                SCOPE,
                "Dot: mean CRPS. Line: 95% interval from re-drawing whole months and seeds.",
            ],
        )
    )


def figure_3() -> alt.TopLevelMixin:
    """Coverage by lead band for the arms without and with the spread factor."""
    data = (
        pl.read_parquet(RESULTS_DIR / "coverage.parquet")
        .filter(pl.col("arm").is_in(list(COVERAGE_ARMS)))
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    order = [name for name, _, _ in LEAD_BANDS]
    labels = [ARM_LABELS[a] for a in COVERAGE_ARMS]
    panels = []
    for technology in ("wind", "solar"):
        part = data.filter(pl.col("technology") == technology)
        target = alt.Chart(
            pl.DataFrame({"low": [90.0 - TOLERANCE], "high": [90.0 + TOLERANCE]})
        ).mark_rect(color=ocf.DATA_GREEN, opacity=0.15)
        target = target.encode(y=alt.Y("low:Q", scale=Y_SCALE), y2="high:Q")  # ty: ignore[unresolved-attribute]
        base = alt.Chart(part).encode(
            x=alt.X(
                "band:N", sort=order, title="Lead time of the forecast", axis=alt.Axis(labelAngle=0)
            ),
            xOffset=alt.XOffset("label:N", sort=labels),
            color=alt.Color(
                "label:N",
                sort=labels,
                scale=alt.Scale(domain=labels, range=list(COVERAGE_COLOURS)),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=330),
            ),
        )
        points = base.mark_point(filled=True, size=36).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("coverage:Q", title="Cases inside the 5% to 95% band (%)", scale=Y_SCALE)
        )
        bars = base.mark_rule().encode(y=alt.Y("lower:Q", scale=Y_SCALE), y2="upper:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(target, bars, points).properties(
                width=PLOT_WIDTH, height=190, title=f"{technology.title()} farms"
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="The spread factor brings coverage close to 90% in every lead band",
            lines=[
                SCOPE,
                (
                    "Dot: estimate. Line: 95% interval from re-drawing whole months. Green band: "
                    "within 2.5 points of the 90% the band should cover. Cases with a 95% "
                    "quantile below 1% of capacity (night-time solar) are left out."
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
