# ruff: noqa: E501
"""Draw experiment 248's figures from the score tables (no farm series are drawn)."""

from pathlib import Path
from typing import Final

import altair as alt
import e00248_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 300
ARMS: Final[dict[str, str]] = {
    "h5_ukv": "H5 with UKV columns",
    "xgboost_ukv": "XGBoost with UKV columns",
    "h5_ukv_placebo": "H5 with UKV from another day (control)",
    "h5_quantile": "H5, ensemble inputs only",
    "xgboost": "XGBoost, ensemble inputs only",
    "h2_ukv": "H2 with UKV columns",
    "h2_residual": "H2, ensemble inputs only",
    "plant": "Plant alone",
    "floor_climatology": "Floor: climatology",
    "floor_heuristic": "Floor: manual heuristic",
}
FOLDS: Final[dict[str, str]] = {
    "pooled": "Both folds (10 months)",
    "scored": "2025-09 to 2025-12",
    "replicate": "2025-03 to 2025-08",
}
FOLD_COLOURS: Final[list[str]] = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY]


def _title(lines: list[str]) -> alt.TitleParams:
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


def _colour() -> alt.Color:
    return alt.Color(
        "fold_label:N",
        scale=alt.Scale(domain=list(FOLDS.values()), range=FOLD_COLOURS),
        legend=alt.Legend(title=None, orient="top"),
    )


def _dots(
    *, table: pl.DataFrame, x: str, lo: str, hi: str, title: str, order: list[str], height: int
) -> alt.Chart:
    base = alt.Chart(table)
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    layers = base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(f"{lo}:Q", title=title),
        x2=f"{hi}:Q",
        y=y,
        color=_colour(),
        yOffset="fold_label:N",
    ) + base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
        x=f"{x}:Q", y=y, color=_colour(), yOffset="fold_label:N"
    )
    return (layers + zero if x == "percent_change" else layers).properties(
        width=PLOT_WIDTH, height=height
    )


def headline() -> None:
    """Figure 1: quantile score per arm for both folds pooled and each fold."""
    scores = (
        pl.read_csv(common.DATA_DIR / "scores.csv")
        .filter((pl.col("slice") == "all") & (pl.col("metric") == "score"))
        .with_columns(
            label=pl.col("arm").replace_strict(ARMS),
            fold_label=pl.col("setting").replace_strict(FOLDS),
        )
    )
    chart = _dots(
        table=scores,
        x="value",
        lo="lo",
        hi="hi",
        title="Quantile score (points of capacity; smaller is better)",
        order=list(ARMS.values()),
        height=380,
    )
    chart.properties(
        title=_title(
            [
                "Quantile score of the day-ahead forecast of six metered solar farms (A to F), mean over farms, 12 to 36 h",
                "leads, daylight half-hours with a usable UKV run. Dot: estimate (mean over five seeds for neural heads). Line: 95%",
                "interval from resampling whole months.",
            ]
        )
    ).save(DOCS_DIR / "figure_1_headline.svg")


def contrasts() -> None:
    """Figure 2: the change in quantile score from adding UKV columns, and the control."""
    pairs = [
        ("h5_ukv", "h5_quantile"),
        ("h2_ukv", "h2_residual"),
        ("xgboost_ukv", "xgboost"),
        ("h5_ukv_placebo", "h5_quantile"),
    ]
    names = {
        ("h5_ukv", "h5_quantile"): "H5: UKV columns vs ensemble only",
        ("h2_ukv", "h2_residual"): "H2: UKV columns vs ensemble only",
        ("xgboost_ukv", "xgboost"): "XGBoost: UKV columns vs ensemble only",
        ("h5_ukv_placebo", "h5_quantile"): "H5 control: UKV from another day",
    }
    c = pl.read_csv(common.DATA_DIR / "contrasts.csv").filter(
        (pl.col("slice") == "all") & (pl.col("metric") == "score")
    )
    frames = [
        c.filter((pl.col("arm") == a) & (pl.col("reference") == r)).with_columns(
            label=pl.lit(names[(a, r)])
        )
        for a, r in pairs
    ]
    table = pl.concat(frames).with_columns(fold_label=pl.col("setting").replace_strict(FOLDS))
    chart = _dots(
        table=table,
        x="percent_change",
        lo="lo",
        hi="hi",
        title="Change in quantile score (%; negative is better)",
        order=list(names.values()),
        height=230,
    )
    chart.properties(
        title=_title(
            [
                "Change in quantile score from adding UKV irradiance and cloud columns, mean over six farms, 12 to 36 h.",
                "Dot: estimate. Line: 95% interval from resampling whole months. Grey rule: no change.",
            ]
        )
    ).save(DOCS_DIR / "figure_2_contrasts.svg")


def per_farm() -> None:
    """Figure 3: the change from UKV columns at each farm, both folds pooled."""
    farms = pl.read_csv(common.DATA_DIR / "per_farm.csv").filter(pl.col("setting") == "pooled")
    rows = []
    for arm, ref, label in (("h5_ukv", "h5_quantile", "H5"), ("xgboost_ukv", "xgboost", "XGBoost")):
        a = farms.filter(pl.col("arm") == arm).select("time_series_id", a="score")
        b = farms.filter(pl.col("arm") == ref).select("time_series_id", b="score")
        rows.append(
            a.join(b, on="time_series_id").with_columns(
                percent_change=100 * (pl.col("a") - pl.col("b")) / pl.col("b"), model=pl.lit(label)
            )
        )
    table = pl.concat(rows)
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        alt.Chart(table)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "percent_change:Q",
                title="Change in quantile score from adding UKV columns (%; negative is better)",
            ),
            y=alt.Y("time_series_id:N", title="Farm"),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(domain=["H5", "XGBoost"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        + zero
    ).properties(
        width=PLOT_WIDTH,
        height=180,
        title=_title(
            [
                "Change in quantile score from adding UKV columns at each farm, 12 to 36 h, both folds pooled.",
                "Dot: estimate. Grey rule: no change.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_3_per_farm.svg")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    headline()
    contrasts()
    per_farm()
