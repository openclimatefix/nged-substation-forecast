# ruff: noqa: E501
"""Draw experiment 300's figures from the score tables (farm series are never drawn; farms pooled)."""

import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00300_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 320
ARM_LABELS: Final[dict[str, str]] = {
    "floor_heuristic": "manual_heuristic floor",
    "floor_climatology": "Climatology floor",
    "p2": "P2: plant binned by forecast and spread",
    "xgb": "XGBoost, same inputs",
    "gam": "GAM: additive model on P2",
    "mdn_ens": "MDN: mixture-density network on P2 (5 seeds averaged)",
    "blend": "Candidate: blend of GAM, MDN, XGBoost",
    "blend_gam_xgb": "Blend of GAM and XGBoost (no MDN)",
}
ABLATION_LABELS: Final[dict[str, str]] = {
    "gam": "GAM (all inputs)",
    "gam_nospread": "GAM without spread",
    "gam_nolag": "GAM without lagged power",
    "gam_noavail": "GAM without availability ratio",
    "p2": "No band model (P2)",
    "blend_gam_xgb": "Blend without MDN",
    "blend": "Full candidate (blend)",
}
SETTING_LABELS: Final[dict[str, str]] = {
    "own_history": "Own history",
    "zero_history": "Zero history",
}
SETTING_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE]
LEVEL_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, "#2a9d8f", "#8a5a9e"]


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
        height=360,
        title=[
            "Figure 1: absolute quantile score of every arm, farms A to F, 12 to 36 h leads, daylight, monthly refits.",
            "Dot: score. Line: 98.75% interval from resampling the 12 scored months.",
        ],
    ).save(DOCS_DIR / "figure_1_scores.svg")


def _by_slice(*, scores: pl.DataFrame, kind: str, metric: str, setting: str) -> pl.DataFrame:
    return _label(
        scores.filter(
            (pl.col("slice_kind") == kind)
            & (pl.col("metric") == metric)
            & (pl.col("setting") == setting)
            & pl.col("arm").is_in(list(ARM_LABELS))
        )
    )


def figure_2(*, scores: pl.DataFrame) -> None:
    """Quantile score and error of the median against lead, own history."""
    panels = []
    for metric, title in (
        ("score", "Quantile score"),
        ("ae_median", "Absolute error of the median"),
    ):
        data = _by_slice(scores=scores, kind="lead_bin", metric=metric, setting="own_history")
        base: Any = alt.Chart(data)
        panels.append(
            base.mark_line(point=True)
            .encode(
                x=alt.X("slice:N", title="Lead time (hours, start of 3 h bin)"),
                y=alt.Y(
                    "value:Q", title=f"{title} (points of capacity)", scale=alt.Scale(zero=False)
                ),
                color=alt.Color(
                    "label:N",
                    sort=list(ARM_LABELS.values()),
                    legend=alt.Legend(title=None, orient="bottom", columns=2),
                ),
            )
            .properties(width=PLOT_WIDTH // 2, height=260, title=_title([title]))
        )
    alt.hconcat(*panels).properties(
        title=_title(["Figure 2: score against lead time, own history, all arms and both floors."])
    ).save(DOCS_DIR / "figure_2_by_lead.svg")


def figure_3(*, contrasts: pl.DataFrame) -> None:
    """The planned contrasts, in percent of the reference arm's score."""
    wanted = [
        ("blend", "xgb", "score", "C1/C2: candidate minus XGBoost, quantile score"),
        ("blend", "xgb", "pin99", "C3: candidate minus XGBoost, 0.99 pinball loss"),
        ("blend", "xgb", "ae_median", "C4: candidate minus XGBoost, error of the median"),
        ("blend", "p2", "score", "Candidate minus P2, quantile score"),
        ("gam", "p2", "score", "GAM minus P2, quantile score"),
        ("mdn_ens", "gam", "score", "MDN minus GAM, quantile score"),
        ("blend", "blend_gam_xgb", "score", "Candidate minus blend without MDN"),
    ]
    rows = [
        contrasts.filter(
            (pl.col("slice") == "all")
            & (pl.col("metric") == metric)
            & (pl.col("arm") == arm)
            & (pl.col("reference") == ref)
        ).with_columns(label=pl.lit(label))
        for arm, ref, metric, label in wanted
    ]
    data = pl.concat(rows).with_columns(
        setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
        value=pl.col("percent_change"),
    )
    _dots(
        data=data,
        y="label",
        x_title="Change (%; more negative is better)",
        sort=[w[3] for w in wanted],
        colour="setting_label",
        colour_domain=list(SETTING_LABELS.values()),
        colours=SETTING_COLOURS,
        height=300,
        title=[
            "Figure 3: the planned contrasts and the supporting ones.",
            "Dot: estimate. Line: 98.75% interval. Dashed line: no change.",
        ],
    ).save(DOCS_DIR / "figure_3_contrasts.svg")


def figure_4(*, scores: pl.DataFrame) -> None:
    """Coverage at the four outer levels, own history and zero history."""
    levels = {"01": 0.01, "05": 0.05, "95": 0.95, "99": 0.99}
    data = pl.concat(
        [
            scores.filter(
                (pl.col("slice") == "all")
                & (pl.col("metric") == f"cover{lv}")
                & pl.col("arm").is_in(list(ARM_LABELS))
            ).with_columns(nominal=pl.lit(f"{nominal}"))
            for lv, nominal in levels.items()
        ]
    ).with_columns(
        label=pl.col("arm").replace_strict(ARM_LABELS),
        setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
    )
    base: Any = alt.Chart(data)
    chart = base.mark_point(filled=True, size=60).encode(
        x=alt.X("nominal:N", title="Nominal level"),
        y=alt.Y("value:Q", title="Share of actual output at or below the quantile"),
        color=alt.Color(
            "label:N", sort=list(ARM_LABELS.values()), legend=alt.Legend(title=None, orient="right")
        ),
        shape="setting_label:N",
    )
    chart.properties(
        width=PLOT_WIDTH // 2,
        height=300,
        title=_title(
            [
                "Figure 4: tail calibration. A perfectly calibrated arm sits at its nominal level (0.01, 0.05, 0.95, 0.99)."
            ]
        ),
    ).save(DOCS_DIR / "figure_4_coverage.svg")


def figure_5(*, scores: pl.DataFrame) -> None:
    """Quantile score by clear-sky band, valid hour, and season, own history."""
    panels = []
    for kind, title in (
        ("csi_band", "Clear-sky index band"),
        ("hour_utc", "Valid hour (UTC)"),
        ("season", "Season"),
    ):
        data = _by_slice(scores=scores, kind=kind, metric="score", setting="own_history")
        base: Any = alt.Chart(data)
        panels.append(
            base.mark_line(point=True)
            .encode(
                x=alt.X("slice:N", title=title),
                y=alt.Y(
                    "value:Q",
                    title="Quantile score (points of capacity)",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "label:N",
                    sort=list(ARM_LABELS.values()),
                    legend=alt.Legend(title=None, orient="bottom", columns=2)
                    if kind == "csi_band"
                    else None,
                ),
            )
            .properties(width=PLOT_WIDTH // 3 - 20, height=240, title=_title([title]))
        )
    alt.hconcat(*panels).properties(
        title=_title(["Figure 5: score by clear-sky band, hour, and season, own history."])
    ).save(DOCS_DIR / "figure_5_slices.svg")


def figure_6(*, scores: pl.DataFrame) -> None:
    """Ablations: quantile score with one component dropped, both settings."""
    data = scores.filter(
        (pl.col("slice") == "all")
        & (pl.col("metric") == "score")
        & pl.col("arm").is_in(list(ABLATION_LABELS))
    ).with_columns(
        label=pl.col("arm").replace_strict(ABLATION_LABELS),
        setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
    )
    _dots(
        data=data,
        y="label",
        x_title="Quantile score (points of capacity; lower is better)",
        sort=list(ABLATION_LABELS.values()),
        colour="setting_label",
        colour_domain=list(SETTING_LABELS.values()),
        colours=SETTING_COLOURS,
        height=260,
        title=[
            "Figure 6: ablations. Each row drops one component of the candidate.",
            "Dot: score. Line: 98.75% interval from resampling the 12 scored months.",
        ],
    ).save(DOCS_DIR / "figure_6_ablations.svg")


def main() -> None:
    """Draw all figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    scores, contrasts = _load()
    figure_1(scores=scores)
    figure_2(scores=scores)
    figure_3(contrasts=contrasts)
    figure_4(scores=scores)
    figure_5(scores=scores)
    figure_6(scores=scores)


if __name__ == "__main__":
    main()
