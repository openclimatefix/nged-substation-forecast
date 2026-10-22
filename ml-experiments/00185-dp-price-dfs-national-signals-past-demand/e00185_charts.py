"""Draw the figures of experiment 185's write-up from the stored reports.

uv run --no-sync python ml-experiments/00185-*/e00185_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00185_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00185-dp-price-dfs-national-signals-past-demand"
)
WIDTH: Final[int] = 400


def left(lines: list[str]) -> alt.TitleParams:
    """Return a left-aligned multi-line title."""
    return alt.TitleParams(lines, anchor="start", fontSize=12, subtitleFontSize=12, offset=10)


def zero_rule(*, value: float, color: str, dash: list[int] | None = None) -> alt.Chart:
    """Return a vertical reference rule."""
    return (
        alt.Chart(pl.DataFrame({"x": [value]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=color, strokeDash=dash or [])
        .encode(x="x:Q")
    )


def interval_chart(
    *, frame: pl.DataFrame, title: list[str], x_title: str, rules: alt.Chart
) -> alt.Chart:
    """Return a dot-and-interval chart with one row per label."""
    order = frame["label"].to_list()
    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
    )
    line = base.mark_rule(color=ocf.DATA_BLUE, strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title=x_title), x2="high:Q"
    )
    dot = base.mark_point(filled=True, size=90, color=ocf.BRAND_ORANGE).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
    return (rules + line + dot).properties(
        width=WIDTH, height=26 * len(order) + 30, title=left(title)
    )


def figure_1(*, regress: dict) -> None:
    """Write the out-of-fold R-squared increments of Parts 1 and 2."""
    rows = []
    p1 = regress["part1"]["excess"]
    for key, label in (
        (
            "price_three_summaries_x_temperature_all_days",
            "Price, 3 summaries x temperature (all days)",
        ),
        (
            "price_plus_dfs_x_temperature_dfs_covered_days",
            "Price + DFS flag (days with a DFS record)",
        ),
        ("dfs_only_dfs_covered_days", "DFS flag alone (days with a DFS record)"),
    ):
        d = p1[key]
        rows.append(
            {"label": label, "estimate": d["increment"], "low": d["ci95"][0], "high": d["ci95"][1]}
        )
    for name, d in regress["part2_excess"].items():
        if name.startswith("price"):
            continue
        rows.append(
            {
                "label": f"{name.capitalize()} (all days)",
                "estimate": d["increment"],
                "low": d["ci95"][0],
                "high": d["ci95"][1],
            }
        )
    chart = interval_chart(
        frame=pl.DataFrame(rows),
        title=[
            "Share of the variance in arm D's daily excess loss that a signal explains",
            "beyond temperature and day type (higher is better; the bar was 0.10).",
            "Dot: estimate. Line: 95% interval over months. Six substations, 2025-07 to 2026-06.",
        ],
        x_title="Out-of-fold R-squared gained over temperature, day type, and substation",
        rules=zero_rule(value=0.0, color="#555555")  # ty: ignore[invalid-argument-type]
        + zero_rule(value=0.10, color=ocf.DATA_PURPLE, dash=[4, 4]),
    )
    chart.save(DOCS_DIR / "fig1_variance_explained.svg")


def figure_2(*, regress: dict) -> None:
    """Write the 37 worst days against matched ordinary days."""
    rows = []
    for key, label in (
        ("p2", "System price in the top 5% of settlement periods"),
        ("p1", "DFS event on the day"),
    ):
        d = regress["matched"][key]
        rows.append(
            {
                "source": label,
                "observed": d["observed_of_37"],
                "mean": d["ordinary_mean"],
                "low": d["ordinary_95pct_range"][0],
                "high": d["ordinary_95pct_range"][1],
            }
        )
    frame = pl.DataFrame(rows)
    order = frame["source"].to_list()
    base = alt.Chart(frame).encode(
        y=alt.Y("source:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
    )
    band = base.mark_bar(height=16, opacity=0.35, color="#9A9A9A").encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Days out of 37"), x2="high:Q"
    )
    mean = base.mark_tick(color="#555555", thickness=2, size=20).encode(x="mean:Q")  # ty: ignore[unresolved-attribute]
    obs = base.mark_point(filled=True, size=100, color=ocf.BRAND_ORANGE).encode(x="observed:Q")  # ty: ignore[unresolved-attribute]
    chart = (band + mean + obs).properties(
        width=WIDTH,
        height=110,
        title=left(
            [
                "Orange dot: the 37 unlabelled worst days. Grey band: 95% range of 1,000 draws",
                "of 37 ordinary days matched on substation, weekday, season, and temperature.",
                "Tick: mean of the draws. DFS counts only the 23 days inside its archive.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "fig2_matched_days.svg")


def figure_3(*, score: dict) -> None:
    """Write the hindsight covariate's score contrasts."""
    rows = []
    entry = score["d_plus_1"]
    for variant, name in (("national_demand", "National demand"), ("price", "System price")):
        c = entry[f"contrast_{variant}_minus_control"]
        e, lo, hi = c["quantile_score"]
        rows.append({"label": f"{name}: five substations", "estimate": e, "low": lo, "high": hi})
        for site, (e, lo, hi) in c["per_site_quantile_score"].items():
            rows.append({"label": f"{name}: {site}", "estimate": e, "low": lo, "high": hi})
    chart = interval_chart(
        frame=pl.DataFrame(rows),
        title=[
            "Quantile score with a national signal given to the Kalman filter in hindsight,",
            "minus arm D's (points of capacity; negative is better; not forecastable).",
            "Day-ahead band, scored year. Dot: estimate. Line: 95% interval over months.",
        ],
        x_title="Quantile score minus arm D's (points of effective capacity)",
        rules=zero_rule(value=0.0, color="#555555"),
    )
    chart.save(DOCS_DIR / "fig3_hindsight_score.svg")


def main() -> None:
    """Write all figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    regress = json.loads((common.WORK_DIR / "regress_report.json").read_text())
    score = json.loads((common.WORK_DIR / "score_report.json").read_text())
    figure_1(regress=regress)
    figure_2(regress=regress)
    figure_3(score=score)


if __name__ == "__main__":
    main()
