# ruff: noqa: E501
"""Draw experiment 287's figures from the score tables (farm series are never drawn; farms pooled)."""

import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00287_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 320
ARM_LABELS: Final[dict[str, str]] = {
    "p2": "P2: plant binned by forecast and spread",
    "gam": "GAM: additive model on P2",
    "nn": "NN: monotone-quantile network on P2",
    "nn_ens": "NN, five seeds averaged",
    "mdn": "MDN: mixture-density network on P2",
    "mdn_ens": "MDN, five seeds averaged",
    "xgb_x2": "XGBoost with past weather",
    "blend": "Blend of GAM, NN, XGBoost",
    "blend_mdn": "Blend of GAM, MDN, XGBoost (post hoc)",
    "ctrl_spread": "NN, spread shuffled",
    "ctrl_lag": "NN, lagged power shuffled",
    "ctrl_blind": "NN, no forecast input",
}
SETTING_LABELS: Final[dict[str, str]] = {
    "own_history": "Own history",
    "zero_history": "Zero history",
}
SETTING_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE]
HEADLINE_PAIRS: Final[dict[tuple[str, str], str]] = {
    ("nn", "gam"): "C1: NN minus GAM",
    ("mdn", "gam"): "C2: MDN minus GAM",
    ("gam", "p2"): "C4: GAM minus P2",
    ("blend_mdn", "blend"): "Post hoc: MDN blend minus NN blend",
}


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


def _load() -> tuple[pl.DataFrame, pl.DataFrame]:
    scores = pl.read_csv(common.DATA_DIR / "scores.csv")
    contrasts = pl.read_csv(common.DATA_DIR / "contrasts.csv")
    return scores, contrasts


def _label(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        label=pl.col("arm").replace_strict(ARM_LABELS, default=pl.col("arm")),
        setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
    )


def figure_1(*, scores: pl.DataFrame) -> None:
    """Absolute quantile score per arm and setting."""
    data = _label(
        scores.filter(
            (pl.col("slice") == "all")
            & (pl.col("metric") == "score")
            & pl.col("arm").is_in(list(ARM_LABELS))
        )
    )
    _dots(
        data=data,
        y="label",
        x_title="Quantile score (points of capacity; lower is better)",
        sort=list(ARM_LABELS.values()),
        colour="setting_label",
        colour_domain=list(SETTING_LABELS.values()),
        colours=SETTING_COLOURS,
        height=420,
        title=[
            "Figure 1: absolute quantile score of every arm, farms A to F, 12 to 36 h, daylight.",
            "Dot: score. Line: 98.75% interval from resampling the 12 scored months.",
        ],
    ).save(DOCS_DIR / "figure_1_scores.svg")


def figure_2(*, contrasts: pl.DataFrame) -> None:
    """The pre-registered contrasts, in percent of the reference arm's score."""
    rows = []
    for (arm, ref), label in HEADLINE_PAIRS.items():
        rows.append(
            contrasts.filter(
                (pl.col("slice") == "all")
                & (pl.col("metric") == "score")
                & (pl.col("arm") == arm)
                & (pl.col("reference") == ref)
            ).with_columns(label=pl.lit(label))
        )
    best = contrasts.filter(
        (pl.col("slice") == "all")
        & (pl.col("metric") == "score")
        & (pl.col("arm") == "blend")
        & pl.col("is_headline")
    )
    rows.append(best.with_columns(label=pl.lit("C3: blend minus best single arm")))
    data = pl.concat(rows).with_columns(
        setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
        value=pl.col("percent_change"),
    )
    _dots(
        data=data,
        y="label",
        x_title="Change in quantile score (%; more negative is better)",
        sort=[
            "C1: NN minus GAM",
            "C2: MDN minus GAM",
            "C3: blend minus best single arm",
            "C4: GAM minus P2",
            "Post hoc: MDN blend minus NN blend",
        ],
        colour="setting_label",
        colour_domain=list(SETTING_LABELS.values()),
        colours=SETTING_COLOURS,
        height=260,
        title=[
            "Figure 2: the pre-registered contrasts. The neural head would have needed an interval wholly below -1.5%.",
            "Dot: estimate. Line: 98.75% interval. Dashed line: no change.",
        ],
    ).save(DOCS_DIR / "figure_2_contrasts.svg")


def figure_3(*, contrasts: pl.DataFrame) -> None:
    """Tail pinball loss against P2, own history, at the four outer levels."""
    arms = ["gam", "nn", "mdn", "xgb_x2", "blend"]
    data = pl.concat(
        [
            contrasts.filter(
                (pl.col("slice") == "all")
                & (pl.col("setting") == "own_history")
                & (pl.col("metric") == f"pin{lv}")
                & (pl.col("reference") == "p2")
                & pl.col("arm").is_in(arms)
            ).with_columns(level=pl.lit(f"0.{lv}"))
            for lv in ("01", "05", "95", "99")
        ]
    ).with_columns(label=pl.col("arm").replace_strict(ARM_LABELS), value=pl.col("percent_change"))
    _dots(
        data=data,
        y="label",
        x_title="Change in pinball loss against P2 (%; more negative is better)",
        sort=[ARM_LABELS[a] for a in arms],
        colour="level",
        colour_domain=["0.01", "0.05", "0.95", "0.99"],
        colours=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, "#2a9d8f", "#8a5a9e"],
        height=300,
        title=[
            "Figure 3: pinball loss at the four outer quantile levels against P2, own history.",
            "Dot: estimate. Line: 98.75% interval (the 4 headline contrasts set this level).",
        ],
    ).save(DOCS_DIR / "figure_3_tails.svg")


def figure_4(*, contrasts: pl.DataFrame) -> None:
    """Score against P2 by clear-sky band, own history."""
    arms = ["gam", "nn", "mdn", "xgb_x2", "blend"]
    data = contrasts.filter(
        (pl.col("slice_kind") == "csi_band")
        & (pl.col("setting") == "own_history")
        & (pl.col("metric") == "score")
        & (pl.col("reference") == "p2")
        & pl.col("arm").is_in(arms)
    ).with_columns(label=pl.col("arm").replace_strict(ARM_LABELS), value=pl.col("percent_change"))
    _dots(
        data=data,
        y="label",
        x_title="Change in quantile score against P2 (%; more negative is better)",
        sort=[ARM_LABELS[a] for a in arms],
        colour="slice",
        colour_domain=["<0.3", "0.3-0.6", "0.6-0.9", ">=0.9"],
        colours=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, "#2a9d8f", "#8a5a9e"],
        height=300,
        title=[
            "Figure 4: gain over P2 by clear-sky index band (cloudy to clear), own history.",
            "Dot: estimate. Line: 98.75% interval.",
        ],
    ).save(DOCS_DIR / "figure_4_clear_sky_bands.svg")


def main() -> None:
    """Draw all figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    scores, contrasts = _load()
    figure_1(scores=scores)
    figure_2(contrasts=contrasts)
    figure_3(contrasts=contrasts)
    figure_4(contrasts=contrasts)


if __name__ == "__main__":
    main()
