"""Draw experiment 00081's figures from the saved results, as SVG.

1. The six planned contrasts of fold B with their 99.17% intervals, and the same contrasts on fold C
   at 95% (the headline).
2. Each arm's absolute quantile score on fold B ("control-only fold"), with its 95% interval.
3. The quantile score by lead band for the stack and the retrained arms (fold B).
4. Each series' own difference on C1 and C3 (fold B), with its 95% interval.
5. The 9 km check: the frozen model's score relative to `manual_heuristic`, by month of forecast
   start.

Only substations appear, labelled D1 to D22; no generator is involved. The SVGs are afterwards
optimised with `npx svgo@4 --multipass --precision=1 --final-newline`. The figure text is reviewed
before the figures are first rendered, as the `study` skill says.

    uv run python ml-experiments/00081-*/e00081_charts.py [--smoke]
"""

import argparse
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00081_common as common
import plotting.ocf_theme as ocf
import polars as pl

PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
SCOPE: Final[str] = (
    "Control member of the ECMWF ensemble only, 13 Primary substations, 274 daily forecast runs "
    "from July 2023 to March 2024 (fold B). Scores are control-only and are not comparable with "
    "51-member scores."
)
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "Weekly retraining minus the stack",
    "C2": "Monthly retraining minus the stack",
    "C3": "Weekly minus monthly retraining",
    "C4": "Rolling window without the Christmas column minus with it",
    "C5": "Rolling window, half-life 2 years minus 0.5 years",
    "C6": "Blend of four models minus its best member",
}
ARM_LABELS: Final[dict[str, str]] = {
    "frozen_cs": "Stack: frozen, 14-day correction",
    "frozen_s": "Frozen, no correction",
    "monthly_s": "Monthly retraining",
    "weekly_s": "Weekly retraining",
    "roll12_s": "Monthly, 12-month window",
    "roll12_no_christmas_s": "12-month window, no Christmas column",
    "roll12_hl2y_s": "12-month window, half-life 2 years",
    "blend": "Blend of four models",
    "manual_heuristic": "manual_heuristic",
    "climatology": "climatology",
}
ARM_KIND: Final[dict[str, str]] = {
    "frozen_cs": "Frozen",
    "frozen_s": "Frozen",
    "monthly_s": "Retrained",
    "weekly_s": "Retrained",
    "roll12_s": "Retrained",
    "roll12_no_christmas_s": "Retrained",
    "roll12_hl2y_s": "Retrained",
    "blend": "Blend",
    "manual_heuristic": "Baseline",
    "climatology": "Baseline",
}
KIND_COLOURS: Final[dict[str, str]] = {
    "Frozen": ocf.BRAND_ORANGE,
    "Retrained": ocf.DATA_BLUE,
    "Blend": ocf.DATA_GREEN,
    "Baseline": ocf.DATA_PURPLE,
}
LEAD_ARMS: Final[tuple[str, ...]] = ("frozen_cs", "monthly_s", "weekly_s", "roll12_s")


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _zero_rule() -> alt.Chart:
    return alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]


def _planned(*, folder: Path, fold: str, level_label: str) -> pl.DataFrame:
    return pl.read_parquet(folder / fold / "contrasts.parquet").filter(
        (pl.col("family") == "planned")
        & (pl.col("site") == "pooled")
        & (pl.col("scope") == "all leads")
        & (pl.col("metric") == "qs")
        & (pl.col("level_label") == level_label)
    )


def figure_1(*, folder: Path) -> alt.TopLevelMixin:
    """Fold B's planned contrasts at 99.17% and fold C's at 95%."""
    panels = []
    for fold, level_label, title in (
        ("b", "planned", "Fold B (July 2023 to March 2024): 99.17% intervals, decide the verdict"),
        ("c", "95", "Fold C (July 2022 to March 2023): 95% intervals, descriptive"),
    ):
        if not (folder / fold / "contrasts.parquet").exists():
            continue
        data = _planned(folder=folder, fold=fold, level_label=level_label).with_columns(
            label=pl.col("id").replace_strict(CONTRAST_LABELS, default=None)
        )
        data = data.filter(pl.col("label").is_not_null())
        base = alt.Chart(data).encode(
            y=alt.Y(
                "label:N",
                sort=list(CONTRAST_LABELS.values()),
                title=None,
                axis=alt.Axis(labelLimit=380),
            )
        )
        points = base.mark_point(filled=True, size=60, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title=(
                    "Quantile score, first arm minus second "
                    "(points of capacity; below zero is better)"
                ),
            )
        )
        bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(_zero_rule(), bars, points).properties(
                width=PLOT_WIDTH,
                height=22 * data.height + 40,
                title=alt.TitleParams(title, anchor="start"),
            )
        )
    return (
        alt.vconcat(*panels)
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text="Planned contrasts of weekly and monthly retraining, control-member forecasts",
                lines=[
                    (
                        "Stack: frozen model with the 14-day correction and spread factor 1.15. "
                        "Every other arm has the spread factor and, unless named, no correction. "
                        + SCOPE
                    ),
                    (
                        "Dot: estimate. Line: interval from re-drawing whole 4-week blocks of "
                        "forecast start. Zero: same score."
                    ),
                ],
            )
        )
    )


def figure_2(*, folder: Path) -> alt.TopLevelMixin:
    """Each arm's absolute quantile score on fold B."""
    absolute = pl.read_parquet(folder / "b" / "absolute.parquet").filter(
        (pl.col("scope") == "all leads") & pl.col("arm").is_in(list(ARM_LABELS))
    )
    data = absolute.with_columns(
        label=pl.col("arm").replace_strict(ARM_LABELS),
        kind=pl.col("arm").replace_strict(ARM_KIND),
    )
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort="x", title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=list(KIND_COLOURS), range=list(KIND_COLOURS.values())),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "qs:Q",
            title="Quantile score (points of capacity; smaller is better)",
            scale=alt.Scale(zero=False),
        )
    )
    bars = base.mark_rule().encode(x="qs_lower:Q", x2="qs_upper:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(
        width=PLOT_WIDTH,
        height=24 * data.height + 20,
        title=_title(
            text="Absolute quantile score of every arm, control-only fold",
            lines=[
                "Mean over the cases that every arm holds. " + SCOPE,
                (
                    "Dot: mean. Line: 95% interval from re-drawing whole 4-week blocks of "
                    "forecast start."
                ),
            ],
        ),
    )


def figure_3(*, folder: Path) -> alt.TopLevelMixin:
    """Quantile score by lead band for the stack and the retrained arms."""
    absolute = pl.read_parquet(folder / "b" / "absolute.parquet").filter(
        (pl.col("scope") != "all leads") & pl.col("arm").is_in(LEAD_ARMS)
    )
    bands = [name for name, _, _ in common.LEAD_BANDS]
    data = absolute.with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("scope:N", sort=bands, title="Lead time", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "qs:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "label:N",
                sort=[ARM_LABELS[a] for a in LEAD_ARMS],
                scale=alt.Scale(
                    domain=[ARM_LABELS[a] for a in LEAD_ARMS],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_SKY, ocf.DATA_BLUE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(orient="top", title=None, columns=2),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=220,
            title=_title(
                text="Quantile score by lead time, control-only fold",
                lines=["Mean quantile score by lead band. " + SCOPE],
            ),
        )
    )


def figure_4(*, folder: Path) -> alt.TopLevelMixin:
    """Each series' own difference on C1 and C3."""
    contrasts = pl.read_parquet(folder / "b" / "contrasts.parquet").filter(
        (pl.col("family") == "planned")
        & pl.col("id").is_in(["C1", "C3"])
        & (pl.col("site") != "pooled")
        & (pl.col("scope") == "all leads")
        & (pl.col("metric") == "qs")
        & (pl.col("level_label") == "95")
    )
    titles = {
        "C1": "Weekly retraining minus the stack",
        "C3": "Weekly minus monthly retraining",
    }
    panels = []
    for contrast_id, title in titles.items():
        data = contrasts.filter(pl.col("id") == contrast_id)
        if data.is_empty():
            continue
        order = sorted(data["site"].to_list(), key=lambda s: int(s[1:]))
        base = alt.Chart(data).encode(y=alt.Y("site:N", sort=order, title=None))
        points = base.mark_point(filled=True, size=50, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("difference:Q", title="Quantile score difference (points of capacity)")
        )
        bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(_zero_rule(), bars, points).properties(
                width=PLOT_WIDTH,
                height=16 * len(order) + 30,
                title=alt.TitleParams(title, anchor="start"),
            )
        )
    return (
        alt.vconcat(*panels)
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text="Each substation on its own, control-only fold",
                lines=[
                    (
                        "The pooled interval re-draws blocks of forecast start only and says "
                        "nothing about other substations, so each is shown too. " + SCOPE
                    ),
                    (
                        "Dot: estimate. Line: 95% interval from re-drawing whole 4-week blocks. "
                        "Below zero favours the first arm."
                    ),
                ],
            )
        )
    )


def figure_5(*, folder: Path) -> alt.TopLevelMixin:
    """The frozen model's score relative to manual_heuristic, by month of forecast start."""
    data = pl.read_parquet(folder / "d" / "nine_km.parquet").filter(
        pl.col("grouping") == "month of forecast start"
    )
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("group:N", title="Month of forecast start (2023)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "relative_percent:Q",
                title="Frozen model's score relative to manual_heuristic (%; below zero is better)",
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=220,
            title=_title(
                text="Does a frozen model keep its edge once the ensemble moves to 9 km?",
                lines=[
                    (
                        "Frozen model fitted to data to 31 March 2023, spread factor 1.15, no "
                        "correction, scored on control-member runs from April to September 2023. "
                        "ECMWF served 18 km before 27 June 2023 and 9 km from then on. Descriptive."
                    ),
                ],
            ),
        )
    )


def main() -> None:
    """Write the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    smoke = parser.parse_args().smoke
    folder = common.RESULTS_DIR.with_name("results_smoke") if smoke else common.RESULTS_DIR
    chart_dir = common.CHART_DIR.with_name("charts_smoke") if smoke else common.CHART_DIR
    chart_dir.mkdir(parents=True, exist_ok=True)
    for name, chart, needs in (
        ("figure_1", figure_1, "b"),
        ("figure_2", figure_2, "b"),
        ("figure_3", figure_3, "b"),
        ("figure_4", figure_4, "b"),
        ("figure_5", figure_5, "d"),
    ):
        if (
            not (folder / needs / "absolute.parquet").exists()
            and not (folder / needs / "nine_km.parquet").exists()
        ):
            print(f"{name}: results of fold {needs} missing, skipped", flush=True)
            continue
        chart(folder=folder).save(chart_dir / f"{name}.svg")
        print(f"wrote {name}.svg", flush=True)


if __name__ == "__main__":
    main()
