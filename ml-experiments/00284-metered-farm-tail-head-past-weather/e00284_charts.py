# ruff: noqa: E501
"""Draw experiment 284's figures from the score tables (farms appear as labels A to F only)."""

from pathlib import Path
from typing import Final

import altair as alt
import e00284_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 330
ARMS: Final[dict[str, str]] = {
    "l2": "L2: H5, past weather + ensemble, era flag",
    "t4b": "T4b: tail head, past-weather power shuffled",
    "t3": "T3: tail head, past weather + ensemble, era flag",
    "t0": "T0: tail head, ensemble rows only",
    "t4a": "T4a: tail head, no era flag",
    "p2": "P2: plant with forecast and spread bins",
    "t5": "T5: XGBoost on the tail levels",
    "x2": "X2: XGBoost, past weather + ensemble, era flag",
}
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "C1: T3 against X2 at 0.99",
    "C3": "C3: T3 against T5 at 0.99",
    "C4": "C4: T3 against shuffled-power T4b at 0.99",
    "t3_vs_l2_99": "T3 against L2 at 0.99",
    "t3_vs_p2_99": "T3 against P2 at 0.99",
    "t3_vs_t0_99": "T3 against T0 at 0.99 (what past weather adds)",
    "C2": "C2: T3 against X2 at 0.01",
    "t3_vs_l2_01": "T3 against L2 at 0.01",
    "t3_vs_p2_01": "T3 against P2 at 0.01",
    "t3_vs_t0_01": "T3 against T0 at 0.01 (what past weather adds)",
}


def _title(lines: list[str]) -> alt.TitleParams:
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


def _dots(*, data: pl.DataFrame, y: alt.Y, x: str, lo: str, hi: str, xtitle: str, height: int):  # noqa: ANN202
    base = alt.Chart(data)
    return (
        base.mark_rule(size=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(f"{lo}:Q", title=xtitle), x2=f"{hi}:Q", y=y
        )
        + base.mark_point(filled=True, size=55, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=f"{x}:Q", y=y
        )
    ).properties(width=PLOT_WIDTH, height=height)


def tails() -> None:
    """Figure 1: pinball loss at the 0.99 and 0.01 quantiles per arm."""
    scores = (
        pl.read_csv(common.DATA_DIR / "scores.csv")
        .filter(pl.col("metric").is_in(["pin99", "pin01"]) & pl.col("arm").is_in(list(ARMS)))
        .with_columns(label=pl.col("arm").replace_strict(ARMS))
    )
    y = alt.Y("label:N", sort=list(ARMS.values()), title=None, axis=alt.Axis(labelLimit=330))
    panels = [
        _dots(
            data=scores.filter(pl.col("metric") == metric),
            y=y,
            x="value",
            lo="lo",
            hi="hi",
            xtitle=f"Pinball loss at the {name} quantile (points of capacity; smaller is better)",
            height=230,
        )
        for metric, name in (("pin99", "0.99"), ("pin01", "0.01"))
    ]
    alt.vconcat(*panels).properties(
        title=_title(
            [
                "Pinball loss at the two tails, six metered solar farms (A to F), mean over farms, 12 to 36 h",
                "leads, daylight half-hours, 2025-09 to 2026-08. Dot: estimate (mean over five seeds for",
                "networks). Line: 95% interval from resampling whole months.",
            ]
        )
    ).save(DOCS_DIR / "figure_1_tails.svg")


def contrasts() -> None:
    """Figure 2: percentage change in tail pinball loss for the planned and secondary contrasts."""
    table = (
        pl.read_csv(common.DATA_DIR / "contrasts.csv")
        .filter(pl.col("contrast").is_in(list(CONTRAST_LABELS)))
        .with_columns(label=pl.col("contrast").replace_strict(CONTRAST_LABELS))
        .rename({"lo98.75": "lo", "hi98.75": "hi"})
    )
    y = alt.Y(
        "label:N", sort=list(CONTRAST_LABELS.values()), title=None, axis=alt.Axis(labelLimit=330)
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        _dots(
            data=table,
            y=y,
            x="percent_change",
            lo="lo",
            hi="hi",
            xtitle="Change in pinball loss (%; negative is better)",
            height=300,
        )
        + zero
    ).properties(
        title=_title(
            [
                "Change in tail pinball loss between arms, mean over six farms. Grey line: no change.",
                "Dot: estimate. Line: 98.75% interval from resampling whole months.",
            ]
        )
    )
    chart.save(DOCS_DIR / "figure_2_contrasts.svg")


def coverage() -> None:
    """Figure 3: share of scored rows below the 0.01 quantile and above the 0.99 quantile, per farm."""
    farms = pl.read_csv(common.DATA_DIR / "per_farm.csv").filter(
        pl.col("arm").is_in(["t3", "l2", "x2"])
    )
    long = pl.concat(
        [
            farms.select("arm", "time_series_id", value=pl.col("cover01") * 100).with_columns(
                level=pl.lit("Below the 0.01 quantile (target 1%)")
            ),
            farms.select(
                "arm", "time_series_id", value=(1.0 - pl.col("cover99")) * 100
            ).with_columns(level=pl.lit("Above the 0.99 quantile (target 1%)")),
        ]
    ).with_columns(label=pl.col("arm").replace_strict({k: ARMS[k] for k in ("t3", "l2", "x2")}))
    chart = (
        alt.Chart(long)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("value:Q", title="Share of scored half-hours outside the quantile (%)"),
            y=alt.Y("time_series_id:N", title="Farm"),
            color=alt.Color("label:N", title=None, scale=alt.Scale(range=ocf.DATA_COLOURS[:3])),
        )
        .properties(width=PLOT_WIDTH - 40, height=150)
        .facet(row=alt.Row("level:N", title=None), columns=1)
        .properties(
            title=_title(
                [
                    "Share of half-hours outside the two tail quantiles per farm, 2025-09 to 2026-08, 12 to 36 h leads."
                ]
            )
        )
    )
    chart.save(DOCS_DIR / "figure_3_coverage.svg")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    tails()
    contrasts()
    coverage()
