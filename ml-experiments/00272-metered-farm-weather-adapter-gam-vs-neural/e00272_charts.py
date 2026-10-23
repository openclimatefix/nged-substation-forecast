# ruff: noqa: E501
"""Draw experiment 272's figures from the score tables (no farm series are drawn)."""

from pathlib import Path
from typing import Final

import altair as alt
import e00272_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 300
ARMS: Final[dict[str, str]] = {
    "xgboost": "XGBoost (experiment 243)",
    "h5_quantile": "H5: plant + neural network offsets (243)",
    "p2": "P2: plant + linear quantile regression (257)",
    "a1_identity": "A1': plant, adapter at identity",
    "a2_gam_min": "A2: additive curves, minimal inputs",
    "a3_gam_full": "A3: additive curves, all variables",
    "a3_placebo": "A3 with placebo variables",
    "a4_nn_min": "A4: neural network, minimal inputs (post hoc)",
    "a5_nn_full": "A5: neural network, all variables (post hoc)",
    "a5_placebo": "A5 with placebo variables (post hoc)",
    "blind": "Blind: neural network, no plant (post hoc)",
    "floor_climatology": "Floor: climatology",
    "floor_heuristic": "Floor: manual heuristic",
}
SETTING_LABELS: Final[dict[str, str]] = {
    "own_history": "Own history",
    "zero_history": "Zero history",
}
SETTING_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE]
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "C1: A3 minus A1' (all variables vs plant)",
    "C2": "C2: A3 minus A2 (all variables vs minimal)",
    "C3": "C3: A5 minus A3 (neural vs curves, post hoc)",
    "C4": "C4: A5 minus A4 (all vs minimal, neural, post hoc)",
    "C6": "C6: A3 minus its placebo (post hoc)",
    "C7": "C7: A5 minus its placebo (post hoc)",
    "X1": "Post hoc: A2 minus A1' (minimal inputs vs plant)",
    "X2": "Post hoc: A4 minus A1' (neural, minimal inputs vs plant)",
    "X3": "Post hoc: blind network minus A1' (no plant vs plant)",
}


def _title(lines: list[str]) -> alt.TitleParams:
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


def headline() -> None:
    """Figure 1: quantile score per arm and setting."""
    scores = (
        pl.read_csv(common.DATA_DIR / "scores.csv")
        .filter(
            (pl.col("slice") == "all")
            & (pl.col("metric") == "score")
            & pl.col("arm").is_in(list(ARMS))
        )
        .with_columns(
            label=pl.col("arm").replace_strict(ARMS),
            setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
        )
    )
    colour = alt.Color(
        "setting_label:N",
        scale=alt.Scale(domain=list(SETTING_LABELS.values()), range=SETTING_COLOURS),
        legend=alt.Legend(title=None, orient="top"),
    )
    y = alt.Y("label:N", sort=list(ARMS.values()), title=None, axis=alt.Axis(labelLimit=330))
    base = alt.Chart(scores)
    chart = (
        base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(domain=[4.8, 9.4]),
            ),
            x2="hi:Q",
            y=y,
            color=colour,
            yOffset="setting_label:N",
        )
        + base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
            x="value:Q", y=y, color=colour, yOffset="setting_label:N"
        )
    ).properties(
        width=PLOT_WIDTH,
        height=380,
        title=_title(
            [
                "Quantile score of the day-ahead forecast of six metered solar farms (A to F), mean over farms, 12 to 36 h",
                "leads, daylight half-hours, 2025-09 to 2026-08. Own history: fitted on the farm's own earlier months.",
                "Zero history: fitted on the other five farms. Dot: estimate (mean over seeds). Line: 95% interval from",
                "resampling whole months. Arms marked post hoc ran after the stop rule fired.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_1_headline.svg")


def contrasts() -> None:
    """Figure 2: the planned contrasts with familywise intervals."""
    c = (
        pl.read_csv(common.DATA_DIR / "contrasts.csv")
        .filter(
            (pl.col("slice") == "all")
            & (pl.col("metric") == "score")
            & (pl.col("setting") == "own_history")
        )
        .filter(pl.col("contrast").is_in(list(CONTRAST_LABELS)))
        .with_columns(name=pl.col("contrast").replace_strict(CONTRAST_LABELS))
    )
    y = alt.Y(
        "name:N", sort=list(CONTRAST_LABELS.values()), title=None, axis=alt.Axis(labelLimit=330)
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    base = alt.Chart(c)
    chart = (
        base.mark_rule(size=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo_family:Q",
                title="Change in quantile score (%; negative means the first arm is better)",
            ),
            x2="hi_family:Q",
            y=y,
        )
        + base.mark_point(filled=True, size=60, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x="percent_change:Q", y=y
        )
        + zero
    ).properties(
        width=PLOT_WIDTH,
        height=240,
        title=_title(
            [
                "Planned contrasts in quantile score, own history, mean over six farms, 12 to 36 h, 2025-09 to 2026-08.",
                "Dot: estimate. Line: 99.4% interval from resampling whole months (Bonferroni over eight contrasts).",
                "Grey rule: no change.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_2_contrasts.svg")


def shapes() -> None:
    """Figure 3: the irradiance factor each adapter applies against each variable, own history."""
    side = common.DATA_DIR / "sidecars"
    frames = [pl.read_parquet(f) for f in sorted(side.glob("shapes_*.parquet"))]
    s = pl.concat(frames).with_columns(
        label=pl.col("arm").replace_strict(
            {"a3_gam_full": "Additive curves (A3)", "a5_nn_full": "Neural network (A5)"}
        )
    )
    band = s.group_by("label", "variable", "value").agg(
        lo=pl.col("factor_mean").min(), hi=pl.col("factor_mean").max()
    )
    # Each farm has its own value grid, so plot one line per farm, thin, with the mean shown by colour only.
    charts = []
    for name in common.FULL_VARS:
        sub = s.filter(pl.col("variable") == name)
        charts.append(
            alt.Chart(sub)
            .mark_line(opacity=0.6, clip=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("value:Q", title=name, scale=alt.Scale(zero=False)),
                y=alt.Y(
                    "factor_mean:Q", title="Irradiance factor", scale=alt.Scale(domain=[0.65, 1.5])
                ),
                color=alt.Color(
                    "label:N",
                    scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                detail="farm:N",
            )
            .properties(width=PLOT_WIDTH / 2 - 40, height=110)
        )
    rows = [alt.hconcat(*charts[i : i + 2]) for i in range(0, len(charts), 2)]
    chart = alt.vconcat(*rows).properties(
        title=_title(
            [
                "Factor applied to the ensemble's irradiance (1 means no change) against each variable, with the other",
                "variables averaged over the training rows, own history, six farms (one line each), seed 0.",
            ]
        )
    )
    del band
    chart.save(DOCS_DIR / "figure_3_shapes.svg")


if __name__ == "__main__":
    headline()
    contrasts()
    shapes()
