"""Draw experiment 00096's figures from the saved results, as SVG.

1. The four planned contrasts with their 98.75% intervals.
2. Each arm's mean absolute error against the true unmodified power.
3. Synthetic-trained minus real-trained error by time since onset.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00096-*/e00096_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00096_common as common
import plotting.ocf_theme as ocf
import polars as pl

PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / (f"{common.EXPERIMENT_ID:05d}-{common.SLUG}")
)
RESULTS: Final[Path] = common.RESULTS_DIR
SCOPE: Final[str] = (
    "Injected transfers at 16 Primary substations, starting in October 2025 and February, "
    "April, and June 2026; shapes S2, S3, S4, and S6."
)
MW_LABELS: Final[dict[str, str]] = {
    "inj": "Injected MW (the truth)",
    "caus": "RO-NC MW as built",
    "cmax": "RO-NC MW, largest step",
}
MW_COLOURS: Final[dict[str, str]] = {
    "inj": ocf.DATA_BLUE,
    "caus": ocf.DATA_PURPLE,
    "cmax": ocf.DATA_SKY,
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """The planned contrasts."""
    data = pl.read_parquet(RESULTS / "headline.parquet")
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    base = alt.Chart(data).encode(
        y=alt.Y("contrast:N", sort=None, title=None, axis=alt.Axis(labelLimit=260))
    )
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Change in error (points of capacity; below zero is better)"),
        x2="high:Q",
    )
    points = base.mark_point(filled=True, size=90, color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        x="point:Q"
    )
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=170,
        title=_title(
            text="Synthetic switches in training did not lower the error",
            lines=[
                SCOPE,
                (
                    "Each synthetic-trained arm minus its real-trained twin with the same "
                    "forecast-time MW (C1 to C3), and the largest-step arm minus the plain booster "
                    "(C4). Dot: estimate. Line: 98.75% interval from re-drawing whole months."
                ),
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """Each arm's error."""
    data = pl.read_parquet(RESULTS / "absolute_a.parquet").filter(pl.col("metric") == "mae")
    rows = []
    for arm, value, low, high in data.select("arm", "value", "low", "high").iter_rows():
        if arm in ("plain", "ceiling"):
            rows.append({"group": arm, "mw": "ref", "value": value, "low": low, "high": high})
        elif arm.startswith(("real_", "synth_")):
            kind, mw = arm.split("_")
            rows.append(
                {"group": f"{kind}-trained", "mw": mw, "value": value, "low": low, "high": high}
            )
    frame = pl.DataFrame(rows).with_columns(
        label=pl.col("mw").replace({**MW_LABELS, "ref": "Reference"})
    )
    order = ["plain", "real-trained", "synth-trained", "ceiling"]
    colours = alt.Scale(
        domain=[*MW_LABELS.values(), "Reference"],
        range=[*MW_COLOURS.values(), ocf.BRAND_ORANGE],
    )
    base = alt.Chart(frame).encode(
        x=alt.X("group:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("label:N"),
        color=alt.Color(
            "label:N", scale=colours, legend=alt.Legend(orient="top", title="Forecast-time MW")
        ),
    )
    points = base.mark_point(filled=True, size=80).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "value:Q",
            scale=alt.Scale(zero=False),
            title="Mean absolute error (points of capacity)",
        )
    )
    bars = base.mark_rule().encode(y="low:Q", y2="high:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            text="The detector's MW, not the training data, sets how much of the gap closes",
            lines=[
                SCOPE,
                (
                    "Error against the true unmodified power, forecast origins inside the "
                    "transfer. Plain is the booster on contaminated lags; ceiling is the same "
                    "booster on lags the transfer never touched. Line: 95% interval."
                ),
            ],
        ),
    )


def figure_3() -> alt.TopLevelMixin:
    """Synthetic minus real by time since onset."""
    data = pl.read_parquet(RESULTS / "by_onset.parquet").with_columns(
        mw=pl.col("contrast").str.extract(r"synth_(\w+) -", 1)
    )
    data = data.with_columns(label=pl.col("mw").replace(MW_LABELS))
    scale = alt.Scale(domain=list(MW_LABELS.values()), range=list(MW_COLOURS.values()))
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="#888").encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    base = alt.Chart(data).encode(
        x=alt.X(
            "bin:N",
            sort=["0-24 h", "1-3 d", "over 3 d"],
            title="Time since the transfer began",
            axis=alt.Axis(labelAngle=0),
        ),
        xOffset=alt.XOffset("label:N", sort=list(MW_LABELS.values())),
        color=alt.Color(
            "label:N", scale=scale, legend=alt.Legend(orient="top", title="Forecast-time MW")
        ),
    )
    points = base.mark_point(filled=True, size=80).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("point:Q", title="Synthetic-trained minus real-trained error (points of capacity)")
    )
    bars = base.mark_rule().encode(y="low:Q", y2="high:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            text="Synthetic training raised the error slightly, mostly after the first day",
            lines=[SCOPE, "Below zero favours synthetic training. Line: 95% interval."],
        ),
    )


def main() -> None:
    """Write the three figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for number, figure in enumerate((figure_1(), figure_2(), figure_3()), start=1):
        figure.save(OUT_DIR / f"figure_{number}.svg")
        print(f"wrote figure_{number}.svg")


if __name__ == "__main__":
    main()
