"""Draw experiment 00033's summary figures from the saved results, as SVG.

1. Quantile score of every headline arm per family (the headline chart).
2. The three decisive contrasts: best guess minus production.
3. Calibration of the 0.95 quantile: the share of cases above it.
4. Quantile score by lead band for the best guess and the baselines.
5. The notable ideas against their reference arms.
6. The weather ceiling: the ERA5 (or CAMS) arm minus its reference, over all leads and by lead band.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00033-*/e00033_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00033_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 520
WRAP_CHARACTERS: Final[int] = 100
GREY: Final[str] = "#8A8A8A"
CATEGORY_COLOURS: Final[dict[str, str]] = {
    "Best guess": ocf.BRAND_ORANGE,
    "Production": ocf.DATA_BLUE,
    "N1": ocf.DATA_SKY,
    "manual_heuristic": ocf.DATA_PURPLE,
    "climatology": GREY,
    "Notable idea": ocf.DATA_GREEN,
}
# group: arm -> (label, category)
HEADLINE: Final[dict[str, dict[str, tuple[str, str]]]] = {
    "Primary": {
        "n1_cs": ("Best guess: N1 + correction + spread", "Best guess"),
        "n1_c": ("N1 + correction", "N1"),
        "n1": ("N1", "N1"),
        "production_c": ("Production + correction", "Production"),
        "production": ("Production", "Production"),
        "manual_heuristic_c": ("manual_heuristic + correction", "manual_heuristic"),
        "manual_heuristic": ("manual_heuristic", "manual_heuristic"),
        "climatology_c": ("climatology + correction", "climatology"),
        "climatology": ("climatology", "climatology"),
        "a7_c": ("Weekly analogues + correction", "Notable idea"),
        "type_id_cs": ("One model per type + correction + spread", "Notable idea"),
        "monthly": ("Monthly retraining, no correction", "Notable idea"),
        "monthly_c": ("Monthly retraining + correction", "Notable idea"),
    },
    "BSP": {
        "n1": ("Best guess: N1", "Best guess"),
        "n1_c": ("N1 + correction", "N1"),
        "production": ("Production", "Production"),
        "manual_heuristic": ("manual_heuristic", "manual_heuristic"),
        "climatology": ("climatology", "climatology"),
    },
    "GSP": {
        "n1": ("Best guess: N1", "Best guess"),
        "n1_c": ("N1 + correction", "N1"),
        "production": ("Production", "Production"),
        "manual_heuristic": ("manual_heuristic", "manual_heuristic"),
        "climatology": ("climatology", "climatology"),
    },
    "solar": {
        "bg_base_spread": ("Best guess: ERA5-trained + spread", "Best guess"),
        "n1": ("N1", "N1"),
        "production": ("Production", "Production"),
        "manual_heuristic": ("manual_heuristic", "manual_heuristic"),
        "climatology": ("climatology", "climatology"),
        "cams_spread": ("CAMS-trained + spread", "Notable idea"),
        "physics_hybrid": ("Physics + XGBoost hybrid", "Notable idea"),
        "physics_scipy": ("Physics only", "Notable idea"),
    },
    "wind": {
        "bg_full": ("Best guess: ERA5-trained + margin + spread", "Best guess"),
        "n1": ("N1", "N1"),
        "production": ("Production", "Production"),
        "manual_heuristic": ("manual_heuristic", "manual_heuristic"),
        "climatology": ("climatology", "climatology"),
        "capacity_a4": ("Margin + power-over-wind-cubed", "Notable idea"),
    },
}
GROUP_TITLES: Final[dict[str, str]] = {
    "Primary": "16 Primary substations",
    "BSP": "2 BSP substations",
    "GSP": "2 GSP substations",
    "solar": "6 solar farms",
    "wind": "3 wind farms",
}
SCOPE: Final[str] = (
    "Demand: 73 forecast runs, one every 5 days. Solar and wind: 53 Tuesday runs. "
    "Validation year July 2025 to June 2026."
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _zero_rule() -> alt.Chart:
    return alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]


def _colour() -> alt.Color:
    return alt.Color(
        "category:N",
        scale=alt.Scale(domain=list(CATEGORY_COLOURS), range=list(CATEGORY_COLOURS.values())),
        legend=alt.Legend(title=None, orient="top", direction="horizontal"),
    )


def _headline_data(*, metric: str) -> pl.DataFrame:
    absolute = pl.read_parquet(RESULTS_DIR / "absolute.parquet").filter(
        pl.col("scope") == "all leads"
    )
    pieces = []
    for group, arms in HEADLINE.items():
        for order, (arm, (label, category)) in enumerate(arms.items()):
            row = absolute.filter((pl.col("group") == group) & (pl.col("arm") == arm))
            if row.is_empty():
                continue
            lower = f"{metric}_lower" if f"{metric}_lower" in row.columns else None
            pieces.append(
                row.select(
                    group=pl.lit(group),
                    label=pl.lit(label),
                    category=pl.lit(category),
                    order=pl.lit(order),
                    value=pl.col(metric),
                    lower=pl.col(lower) if lower else pl.col(metric),
                    upper=pl.col(f"{metric}_upper") if lower else pl.col(metric),
                )
            )
    return pl.concat(pieces)


def _panel(*, data: pl.DataFrame, group: str, x_title: str) -> alt.LayerChart | alt.FacetChart:
    part = data.filter(pl.col("group") == group).sort("order")
    base = alt.Chart(part).encode(
        y=alt.Y("label:N", sort=part["label"].to_list(), title=None, axis=alt.Axis(labelLimit=260))
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("value:Q", title=x_title, scale=alt.Scale(zero=False)), color=_colour()
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q", color=_colour())  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(
        width=PLOT_WIDTH,
        height=22 * part.height + 30,
        title=alt.TitleParams(GROUP_TITLES[group], anchor="start", fontSize=12),
    )


def figure_headline() -> alt.TopLevelMixin:
    """Quantile score of every headline arm, per family."""
    data = _headline_data(metric="qs")
    panels = [
        _panel(
            data=data, group=group, x_title="Quantile score (points of capacity; smaller is better)"
        )
        for group in HEADLINE
    ]
    return (
        alt.vconcat(*panels)
        .properties(
            title=_title(
                text="Production has the highest quantile score in every family",
                lines=[
                    "Quantile score (the CRPS estimate, in points of capacity; 1 point is 1% of "
                    "the series' effective capacity) over all leads. " + SCOPE,
                    (
                        "Dot: mean over the seeds each arm has (three; monthly retraining has "
                        "one). Line: 95% interval from re-drawing whole months."
                    ),
                ],
            )
        )
        .resolve_scale(color="shared")
    )


def figure_decisive() -> alt.TopLevelMixin:
    """The three decisive contrasts."""
    labels = {
        "C1": "Primary demand",
        "C2": "Solar",
        "C3": "Wind",
    }
    data = (
        pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
        .filter(
            pl.col("id").is_in(list(labels))
            & (pl.col("scope") == "all leads")
            & (pl.col("metric") == "qs")
            & (pl.col("level_label") == "planned")
        )
        .with_columns(label=pl.col("id").replace_strict(labels))
    )
    thresholds = pl.DataFrame({"x": [-1.0, -0.5, -1.0], "label": list(labels.values())})
    base = alt.Chart(data).encode(y=alt.Y("label:N", sort=list(labels.values()), title=None))
    points = base.mark_point(filled=True, size=90, color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Best guess minus production (points of capacity; more negative is better)",
        )
    )
    bars = base.mark_rule(color=ocf.BRAND_ORANGE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    ticks = (
        alt.Chart(thresholds)
        .mark_tick(color="#444444", thickness=2, size=30)
        .encode(x="x:Q", y=alt.Y("label:N", sort=list(labels.values())))  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(_zero_rule(), ticks, bars, points).properties(
        width=PLOT_WIDTH,
        height=140,
        title=_title(
            text="Best guess against production on the quantile score",
            lines=[
                (
                    "Each row is the best guess minus the deployed production recipe, over all "
                    "leads. Dot: estimate. Line: 98.33% interval (Bonferroni across the three) "
                    "from re-drawing whole months and seeds. Black tick: the smallest gain worth "
                    "porting."
                ),
                SCOPE,
            ],
        ),
    )


def figure_tail() -> alt.TopLevelMixin:
    """The exceedance share of the 0.95 quantile for the Primary, solar, and wind arms."""
    data = _headline_data(metric="ex95").filter(pl.col("group").is_in(["Primary", "solar", "wind"]))
    panels = [
        _panel(data=data, group=g, x_title="Share of cases above the 0.95 quantile (aim: 0.05)")
        for g in ("Primary", "solar", "wind")
    ]
    return (
        alt.vconcat(*panels)
        .properties(
            title=_title(
                text="Production's 0.95 quantile is exceeded in 3 to 5 cases in 10, not 1 in 20",
                lines=[
                    "Share of validation cases in which the observed power exceeds the arm's 0.95 "
                    "quantile; the aim is 0.05. " + SCOPE,
                ],
            )
        )
        .resolve_scale(color="shared")
    )


def figure_bands(*, groups: tuple[str, ...] = ("Primary", "solar", "wind")) -> alt.TopLevelMixin:
    """Quantile score by lead band for the best guess and the baselines."""
    wanted = {
        "Primary": {
            "n1_cs": "Best guess",
            "production": "Production",
            "n1": "N1",
            "manual_heuristic": "manual_heuristic",
        },
        "solar": {
            "bg_base_spread": "Best guess",
            "production": "Production",
            "n1": "N1",
            "manual_heuristic": "manual_heuristic",
        },
        "wind": {
            "bg_full": "Best guess",
            "production": "Production",
            "n1": "N1",
            "manual_heuristic": "manual_heuristic",
        },
    }
    bands = [b[0] for b in common.LEAD_BANDS]
    absolute = pl.read_parquet(RESULTS_DIR / "absolute.parquet").filter(
        pl.col("scope").is_in(bands)
    )
    panels = []
    for group in groups:
        rows = pl.concat(
            [
                absolute.filter((pl.col("group") == group) & (pl.col("arm") == arm)).with_columns(
                    category=pl.lit(category)
                )
                for arm, category in wanted[group].items()
            ]
        )
        base = alt.Chart(rows).encode(x=alt.X("scope:N", sort=bands, title="Lead"), color=_colour())
        lines = base.mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "qs:Q", title="Quantile score (points of capacity)", scale=alt.Scale(zero=False)
            )
        )
        panels.append(
            lines.properties(
                width=PLOT_WIDTH,
                height=170,
                title=alt.TitleParams(GROUP_TITLES[group], anchor="start"),
            )
        )
    return (
        alt.vconcat(*panels)
        .properties(
            title=_title(
                text="Quantile score by lead band",
                lines=[
                    "Quantile score by lead band, mean over three seeds; smaller is better. "
                    + SCOPE
                ],
            )
        )
        .resolve_scale(color="shared")
    )


def figure_notable() -> alt.TopLevelMixin:
    """The notable ideas against their reference arms."""
    labels = {
        "C9": "Weekly analogues + correction, minus N1 + correction (Primary)",
        "C10": "One model per type, minus one per series (Primary, with correction and spread)",
        "C11": "Monthly retraining + correction, minus N1 + correction (Primary, seed 0)",
        "C12": "Monthly retraining, minus N1 (Primary, seed 0)",
        "C23": "CAMS-trained + spread, minus best guess (solar)",
        "C24": "Margin + power-over-wind-cubed, minus margin (wind)",
        "C25": "Physics + XGBoost hybrid, minus ERA5-trained (solar)",
        "C26": "Physics only, minus ERA5-trained (solar)",
    }
    data = (
        pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
        .filter(
            pl.col("id").is_in(list(labels))
            & (pl.col("scope") == "all leads")
            & (pl.col("metric") == "qs")
            & (pl.col("level_label") == "95")
        )
        .with_columns(label=pl.col("id").replace_strict(labels))
    )
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=list(labels.values()), title=None, axis=alt.Axis(labelLimit=420))
    )
    points = base.mark_point(filled=True, size=70, color=ocf.DATA_GREEN).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Difference in quantile score (points of capacity; negative is better)",
        )
    )
    bars = base.mark_rule(color=ocf.DATA_GREEN).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(_zero_rule(), bars, points).properties(
        width=PLOT_WIDTH,
        height=230,
        title=_title(
            text="The notable ideas against their reference arms",
            lines=[
                "Each row is a notable idea minus its reference arm on the quantile score over all "
                "leads. Dot: estimate. Line: 95% interval from re-drawing whole months and seeds; "
                "not corrected for multiple contrasts. " + SCOPE,
            ],
        ),
    )


def figure_ceiling() -> alt.TopLevelMixin:
    """The weather-ceiling contrasts over all leads."""
    labels = {
        "C27": "Primary demand: ERA5 weather, minus ENS weather",
        "C30": "Primary demand: retrained on ERA5 and fed ERA5, minus N1 on ENS",
        "C31": "Primary demand: ERA5 with CAMS shortwave, minus ENS weather",
        "C32": "BSP and GSP demand: ERA5 weather, minus ENS weather",
        "C28": "Solar: ERA5 weather, minus ENS best guess",
        "C29": "Wind: ERA5 weather, minus ENS best guess",
    }
    data = (
        pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
        .filter(
            pl.col("id").is_in(list(labels))
            & (pl.col("scope") == "all leads")
            & (pl.col("metric") == "qs")
            & (pl.col("level_label") == "95")
        )
        .with_columns(label=pl.col("id").replace_strict(labels))
    )
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=list(labels.values()), title=None, axis=alt.Axis(labelLimit=420))
    )
    points = base.mark_point(filled=True, size=70, color=ocf.DATA_PURPLE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Difference in quantile score (points of capacity; negative is better)",
        )
    )
    bars = base.mark_rule(color=ocf.DATA_PURPLE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(_zero_rule(), bars, points).properties(
        width=PLOT_WIDTH,
        height=180,
        title=_title(
            text="Ceiling: how much better weather could add",
            lines=[
                "ERA5 (and CAMS) are estimates of past weather and arrive days late, so no "
                "serving system could use them: each row is a ceiling for a weather product of "
                "that accuracy. Dot: estimate. Line: 95% interval from re-drawing whole months "
                "and seeds. " + SCOPE,
            ],
        ),
    )


def main() -> None:
    """Write every summary figure."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1_headline": figure_headline,
        "figure_2_decisive": figure_decisive,
        "figure_3_tail": figure_tail,
        "figure_4_bands": figure_bands,
        "figure_5_notable": figure_notable,
        "figure_6_ceiling": figure_ceiling,
    }
    for name, build in figures.items():
        build().save(CHART_DIR / f"{name}.svg")
        print(f"wrote {name}", flush=True)


if __name__ == "__main__":
    main()
