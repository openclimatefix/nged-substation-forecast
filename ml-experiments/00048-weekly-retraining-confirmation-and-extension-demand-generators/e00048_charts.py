"""Draw experiment 00048's figures from the saved results, as SVG.

1. The nine planned contrasts with their 99.44% intervals (the headline).
2. Each arm's absolute quantile score by group, with its 95% interval.
3. The quantile score by lead band for the stack and the retrained arms.
4. Each series' own difference on the planned contrasts, with its 95% interval.

A generator is shown only by its anonymous label (A to F, W1 to W3) and only as a score, never as a
time series. The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1
--final-newline`.

    uv run python ml-experiments/00048-*/e00048_charts.py [--smoke]
"""

import argparse
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00048_common as common
import plotting.ocf_theme as ocf
import polars as pl

PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
SCOPE: Final[str] = (
    "292 daily forecast runs, July 2025 to June 2026 (the runs that experiment 34 left out)."
)
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "Primary: weekly retraining minus the stack",
    "C2": "Primary: monthly retraining minus the stack",
    "C3": "Primary: weekly minus monthly retraining",
    "C4": "Primary, weekly: 7-day correction minus none",
    "C5": "Primary, weekly: 14-day correction minus none",
    "C6": "Primary, weekly: 7-day minus 14-day correction",
    "C7": "BSP and GSP: weekly retraining minus frozen with correction",
    "C8s": "Solar farms: monthly minus frozen retraining",
    "C8w": "Wind farms: monthly minus frozen retraining",
}
ARM_LABELS: Final[dict[str, str]] = {
    "frozen_cs": "Stack: frozen, 14-day correction",
    "frozen_s": "Frozen, no correction",
    "monthly_s": "Monthly retraining",
    "weekly_s": "Weekly retraining",
    "weekly_cs": "Weekly, 14-day correction",
    "weekly_c7s": "Weekly, 7-day correction",
    "roll12_s": "Monthly, 12-month window",
    "gen_frozen_s": "Frozen",
    "gen_monthly_s": "Monthly retraining",
    "manual_heuristic": "manual_heuristic",
    "climatology": "climatology",
}
ARM_KIND: Final[dict[str, str]] = {
    "frozen_cs": "Frozen",
    "frozen_s": "Frozen",
    "gen_frozen_s": "Frozen",
    "monthly_s": "Retrained",
    "weekly_s": "Retrained",
    "weekly_cs": "Retrained",
    "weekly_c7s": "Retrained",
    "roll12_s": "Retrained",
    "gen_monthly_s": "Retrained",
    "manual_heuristic": "Baseline",
    "climatology": "Baseline",
}
KIND_COLOURS: Final[dict[str, str]] = {
    "Frozen": ocf.BRAND_ORANGE,
    "Retrained": ocf.DATA_BLUE,
    "Baseline": ocf.DATA_PURPLE,
}
GROUP_ARMS: Final[dict[str, tuple[str, ...]]] = {
    "Primary substations": (
        "frozen_cs",
        "frozen_s",
        "monthly_s",
        "weekly_s",
        "weekly_cs",
        "weekly_c7s",
        "roll12_s",
        "manual_heuristic",
        "climatology",
    ),
    "BSP substations": ("frozen_cs", "frozen_s", "monthly_s", "weekly_s", "weekly_cs"),
    "GSP substations": ("frozen_cs", "frozen_s", "monthly_s", "weekly_s", "weekly_cs"),
    "Solar farms": ("gen_frozen_s", "gen_monthly_s", "manual_heuristic"),
    "Wind farms": ("gen_frozen_s", "gen_monthly_s", "manual_heuristic"),
}
GROUP_KEYS: Final[dict[str, str]] = {
    "Primary substations": "Primary",
    "BSP substations": "BSP",
    "GSP substations": "GSP",
    "Solar farms": "solar",
    "Wind farms": "wind",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _zero_rule() -> alt.Chart:
    return alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]


def figure_1(*, folder: Path) -> alt.TopLevelMixin:
    """Each planned contrast's difference with its 99.44% interval."""
    contrasts = pl.read_parquet(folder / "contrasts.parquet").filter(
        (pl.col("family") == "planned")
        & (pl.col("site") == "pooled")
        & (pl.col("scope") == "all leads")
        & (pl.col("metric") == "qs")
        & (pl.col("level_label") == "planned")
    )
    data = contrasts.with_columns(label=pl.col("id").replace_strict(CONTRAST_LABELS, default=None))
    band = (
        alt.Chart(pl.DataFrame({"low": [-0.05], "high": [0.05]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color="#DDDDDD", opacity=0.5)
        .encode(x="low:Q", x2="high:Q")
    )
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
                "Quantile score, first arm minus second (points of capacity; below zero is better)"
            ),
        )
    )
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(band, _zero_rule(), bars, points).properties(
        width=PLOT_WIDTH,
        height=280,
        title=_title(
            text="Planned contrasts of weekly and monthly retraining",
            lines=[
                "Stack: frozen model with the 14-day correction and spread factor 1.15. "
                "Every other arm has the spread factor and, unless named, no correction. " + SCOPE,
                (
                    "Dot: estimate. Line: 99.44% interval from re-drawing whole months. Zero: "
                    "same score. Grey band: plus or minus 0.05, the no-change band for solar and "
                    "wind."
                ),
            ],
        ),
    )


def figure_2(*, folder: Path) -> alt.TopLevelMixin:
    """Each arm's absolute quantile score by group."""
    absolute = pl.read_parquet(folder / "absolute.parquet").filter(pl.col("scope") == "all leads")
    panels = []
    for title, arms in GROUP_ARMS.items():
        data = absolute.filter(
            (pl.col("group") == GROUP_KEYS[title]) & pl.col("arm").is_in(arms)
        ).with_columns(
            label=pl.col("arm").replace_strict(ARM_LABELS),
            kind=pl.col("arm").replace_strict(ARM_KIND),
        )
        if data.is_empty():
            continue
        base = alt.Chart(data).encode(
            y=alt.Y(
                "label:N",
                sort=[ARM_LABELS[a] for a in arms],
                title=None,
                axis=alt.Axis(labelLimit=300),
            ),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=list(KIND_COLOURS), range=list(KIND_COLOURS.values())),
                legend=alt.Legend(orient="top", title=None)
                if title == "Primary substations"
                else None,
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
        panels.append(
            alt.layer(bars, points).properties(
                width=PLOT_WIDTH,
                height=22 * len(arms) + 20,
                title=alt.TitleParams(title, anchor="start"),
            )
        )
    return (
        alt.vconcat(*panels)
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text="Absolute quantile score of every arm",
                lines=[
                    "Mean over the cases that every arm of the group holds. " + SCOPE,
                    "Dot: mean. Line: 95% interval from re-drawing whole months.",
                ],
            )
        )
    )


def figure_3(*, folder: Path) -> alt.TopLevelMixin:
    """Quantile score by lead band for the stack and the retrained arms."""
    absolute = pl.read_parquet(folder / "absolute.parquet").filter(pl.col("scope") != "all leads")
    arms = ("frozen_cs", "monthly_s", "weekly_s", "weekly_cs")
    bands = [name for name, _, _ in common.LEAD_BANDS]
    panels = []
    for title in ("Primary substations", "BSP substations", "GSP substations"):
        data = absolute.filter(
            (pl.col("group") == GROUP_KEYS[title]) & pl.col("arm").is_in(arms)
        ).with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
        if data.is_empty():
            continue
        chart = (
            alt.Chart(data)  # ty: ignore[unresolved-attribute]
            .mark_line(point=True)
            .encode(
                x=alt.X("scope:N", sort=bands, title="Lead time", axis=alt.Axis(labelAngle=0)),
                y=alt.Y(
                    "qs:Q",
                    title="Quantile score (points; smaller is better)",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "label:N",
                    sort=[ARM_LABELS[a] for a in arms],
                    scale=alt.Scale(
                        domain=[ARM_LABELS[a] for a in arms],
                        range=[ocf.BRAND_ORANGE, ocf.DATA_SKY, ocf.DATA_BLUE, ocf.DATA_GREEN],
                    ),
                    legend=alt.Legend(orient="top", title=None, columns=2)
                    if title == "Primary substations"
                    else None,
                ),
            )
            .properties(width=PLOT_WIDTH, height=170, title=alt.TitleParams(title, anchor="start"))
        )
        panels.append(chart)
    return (
        alt.vconcat(*panels)
        .resolve_scale(y="independent")
        .properties(
            title=_title(
                text="Quantile score by lead time",
                lines=["Mean quantile score by lead band. " + SCOPE],
            )
        )
    )


def figure_4(*, folder: Path) -> alt.TopLevelMixin:
    """Each series' own difference on the planned contrasts that pool series."""
    contrasts = pl.read_parquet(folder / "contrasts.parquet").filter(
        (pl.col("family") == "planned")
        & pl.col("id").is_in(["C1", "C7", "C8s", "C8w"])
        & (pl.col("site") != "pooled")
        & (pl.col("scope") == "all leads")
        & (pl.col("metric") == "qs")
        & (pl.col("level_label") == "95")
    )
    titles = {
        "C1": "Primary: weekly retraining minus the stack",
        "C7": "BSP and GSP: weekly retraining minus frozen with correction",
        "C8s": "Solar farms: monthly minus frozen retraining",
        "C8w": "Wind farms: monthly minus frozen retraining",
    }
    panels = []
    for contrast_id, title in titles.items():
        data = contrasts.filter(pl.col("id") == contrast_id)
        if data.is_empty():
            continue
        order = sorted(
            data["site"].to_list(), key=lambda s: (s[0], int(s[1:]) if s[1:].isdigit() else 0, s)
        )
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
                text="Each series on its own",
                lines=[
                    "The pooled interval re-draws months only and says nothing about other "
                    "series, so each series is shown too. " + SCOPE,
                    (
                        "Dot: estimate. Line: 95% interval from re-drawing whole months. Below "
                        "zero favours the first arm."
                    ),
                ],
            )
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
    for name, chart in (
        ("figure_1", figure_1(folder=folder)),
        ("figure_2", figure_2(folder=folder)),
        ("figure_3", figure_3(folder=folder)),
        ("figure_4", figure_4(folder=folder)),
    ):
        chart.save(chart_dir / f"{name}.svg")
        print(f"wrote {name}.svg", flush=True)


if __name__ == "__main__":
    main()
