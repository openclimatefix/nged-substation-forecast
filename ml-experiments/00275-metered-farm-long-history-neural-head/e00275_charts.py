# ruff: noqa: E501
"""Draw experiment 275's figures from the score tables (no farm series are drawn)."""

from pathlib import Path
from typing import Final

import altair as alt
import e00275_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 330
ARMS: Final[dict[str, str]] = {
    "x2": "X2: XGBoost, past weather + ensemble, era flag",
    "l3": "L3: H5 pretrained on past weather, fine-tuned",
    "l2": "L2: H5, past weather + ensemble, era flag",
    "l2_ens_offsets": "Post hoc: L2, offsets from ensemble rows only",
    "l2_pre_ens": "Post hoc: L2, past rows before the ensemble era only",
    "l0": "L0: H5, ensemble rows only",
    "x0": "X0: XGBoost, ensemble rows only",
    "l1": "L1: H5, past weather + ensemble, no era flag",
    "control": "Control: L2 with past-weather power shuffled",
}
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "C1: L2 against L0 (H5, with and without past weather)",
    "x2_vs_x0": "X2 against X0 (XGBoost, with and without past weather)",
    "l3_vs_l0": "L3 against L0 (pretrain, then fine-tune)",
    "l1_vs_l0": "L1 against L0 (no era flag)",
    "l2_vs_l1": "L2 against L1 (what the era flag adds)",
    "C3": "C3: L2 against the shuffled-power control",
}


def _title(lines: list[str]) -> alt.TitleParams:
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


def headline() -> None:
    """Figure 1: quantile score per arm."""
    scores = (
        pl.read_csv(common.DATA_DIR / "scores.csv")
        .filter((pl.col("metric") == "score") & pl.col("arm").is_in(list(ARMS)))
        .with_columns(label=pl.col("arm").replace_strict(ARMS))
    )
    y = alt.Y("label:N", sort=list(ARMS.values()), title=None, axis=alt.Axis(labelLimit=330))
    base = alt.Chart(scores)
    chart = (
        base.mark_rule(size=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(domain=[4.8, 6.2]),
            ),
            x2="hi:Q",
            y=y,
        )
        + base.mark_point(filled=True, size=55, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x="value:Q", y=y
        )
    ).properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            [
                "Quantile score of the day-ahead forecast of six metered solar farms (A to F), mean over farms,",
                "12 to 36 h leads, daylight half-hours, 2025-09 to 2026-08. Dot: estimate (mean over five seeds",
                "for networks). Line: 95% interval from resampling whole months.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_1_headline.svg")


def contrasts() -> None:
    """Figure 2: percentage change in quantile score for the planned and secondary contrasts."""
    table = (
        pl.read_csv(common.DATA_DIR / "contrasts.csv")
        .filter((pl.col("metric") == "score") & pl.col("contrast").is_in(list(CONTRAST_LABELS)))
        .with_columns(label=pl.col("contrast").replace_strict(CONTRAST_LABELS))
        .rename({"lo95.0": "lo95", "hi95.0": "hi95"})
    )
    y = alt.Y(
        "label:N", sort=list(CONTRAST_LABELS.values()), title=None, axis=alt.Axis(labelLimit=330)
    )
    base = alt.Chart(table)
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        base.mark_rule(size=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo95:Q", title="Change in quantile score (%; negative is better)"),
            x2="hi95:Q",
            y=y,
        )
        + base.mark_point(filled=True, size=55, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x="percent_change:Q", y=y
        )
        + zero
    ).properties(
        width=PLOT_WIDTH,
        height=200,
        title=_title(
            [
                "Change in quantile score between arms, mean over six farms, 2025-09 to 2026-08. Grey line:",
                "no change. Dot: estimate. Line: 95% interval from resampling whole months.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_2_contrasts.svg")


def tails() -> None:
    """Figure 3: pinball loss at the 0.99 and 0.01 quantiles per arm."""
    scores = (
        pl.read_csv(common.DATA_DIR / "scores.csv")
        .filter(pl.col("metric").is_in(["pin99", "pin01"]) & pl.col("arm").is_in(list(ARMS)))
        .with_columns(
            label=pl.col("arm").replace_strict(ARMS),
            quantile=pl.col("metric").replace_strict(
                {"pin99": "0.99 quantile", "pin01": "0.01 quantile"}
            ),
        )
    )
    y = alt.Y("label:N", sort=list(ARMS.values()), title=None, axis=alt.Axis(labelLimit=330))
    panels = []
    for name in ("0.99 quantile", "0.01 quantile"):
        base = alt.Chart(scores.filter(pl.col("quantile") == name))
        panels.append(
            (
                base.mark_rule(size=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X(
                        "lo:Q",
                        title=f"Pinball loss at the {name} (points of capacity; smaller is better)",
                    ),
                    x2="hi:Q",
                    y=y,
                )
                + base.mark_point(filled=True, size=55, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                    x="value:Q", y=y
                )
            ).properties(width=PLOT_WIDTH, height=210)
        )
    chart = alt.vconcat(*panels).properties(
        title=_title(
            [
                "Pinball loss at the two tails, mean over six farms, 2025-09 to 2026-08. Dot: estimate.",
                "Line: 95% interval from resampling whole months.",
            ]
        )
    )
    chart.save(DOCS_DIR / "figure_3_tails.svg")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    headline()
    contrasts()
    tails()
