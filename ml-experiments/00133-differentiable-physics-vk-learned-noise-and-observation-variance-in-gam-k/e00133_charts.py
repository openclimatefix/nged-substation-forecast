"""Draw the figures of experiment 133's write-up from `work/report/report.json`.

Every figure carries its own scope, unit, and direction in its title and subtitle. SVGs go to the
write-up's folder, PNGs to a scratch folder for the render check.

    uv run python ml-experiments/00133-*/e00133_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00133_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
GREY: Final[str] = "#808080"
DARK: Final[str] = "#333333"
ARM_LABELS: Final[dict[str, str]] = {
    "fix": "FIX (experiment 111's values)",
    "grid": "GRID (chosen per substation)",
    "pool": "POOL (one shared setting)",
    "vk0": "VK0 (online variance, mean only)",
    "vk1": "VK1 (online variance, tails scaled)",
    "trail": "TRAIL (trailing innovations, tails scaled)",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "fix": DARK,
    "grid": ocf.BRAND_ORANGE,
    "pool": ocf.DATA_BLUE,
    "vk0": ocf.DATA_SKY,
    "vk1": ocf.DATA_PURPLE,
    "trail": ocf.DATA_GREEN,
}
PERIOD_COLOURS: Final[dict[str, str]] = {
    "odd months (used to choose)": ocf.DATA_SKY,
    "even months (replicate)": ocf.DATA_BLUE,
    "scored year": ocf.BRAND_ORANGE,
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _zero_rule() -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )


def figure_headline(*, data: dict) -> alt.TopLevelMixin:
    """The planned contrasts at 12 to 36 hours in the three periods, with 99% intervals."""
    names = {
        "C1": "C1 GRID minus FIX",
        "C2": "C2 POOL minus GRID",
        "C3a": "C3a VK0 minus GRID",
        "C3b": "C3b VK1 minus GRID",
        "PH1": "PH1 POOL minus FIX (post hoc)",
    }
    rows = []
    for period, label, section in (
        ("tuning", "odd months (used to choose)", data["tuning"]["contrasts"]),
        ("replicate", "even months (replicate)", None),
        ("scored", "scored year", None),
    ):
        for key, name in names.items():
            c = section[key] if section else data[period]["contrasts"][key]["12-36 h"]["qs"]
            rows.append(
                {
                    "contrast": name,
                    "period": label,
                    "estimate": c["estimate"],
                    "low": c["lower"],
                    "high": c["upper"],
                }
            )
    table = pl.DataFrame(rows)
    order = list(names.values())
    base = alt.Chart(table).encode(
        y=alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("period:N", sort=list(PERIOD_COLOURS)),
        color=alt.Color(
            "period:N",
            scale=alt.Scale(domain=list(PERIOD_COLOURS), range=list(PERIOD_COLOURS.values())),
            legend=alt.Legend(
                title=None, orient="top", direction="horizontal", labelLimit=400, columns=3
            ),
        ),
    )
    rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title=(
                "Difference in quantile score (points of capacity; "
                "below zero favours the first arm)"
            ),
        )
    )
    return (_zero_rule() + rule + dot).properties(
        width=WIDTH - 260,
        height=230,
        title=_title(
            text=(
                "Choosing the filter's noise by next-day score helped only on the months "
                "that chose it"
            ),
            subtitle=(
                "Dot: difference in mean quantile score at leads 12 to 36 hours, five decision "
                "substations (D5, D10, D11, D14, D16), outside switching events. Line: 99% "
                "interval from resampling whole calendar months. Dashed line: no difference."
            ),
        ),
    )


def figure_by_lead(*, data: dict) -> alt.TopLevelMixin:
    """Each arm's quantile score minus FIX's, by lead band, in the scored year."""
    bands = ["0-24 h", "24-48 h", "48-144 h", "144-168 h"]
    rows = []
    for band in bands:
        scores = {d["arm"]: d["qs"] for d in data["scored"]["tables"]["by_lead"][band]}
        rows.extend(
            {"band": band, "arm": ARM_LABELS[arm], "difference": value - scores["fix"]}
            for arm, value in scores.items()
            if arm != "fix"
        )
    table = pl.DataFrame(rows)
    domain = [ARM_LABELS[a] for a in ARM_LABELS if a != "fix"]
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "band:N",
                sort=bands,
                title="Lead (hours after the 09:00 UTC forecast origin)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "difference:Q", title="Quantile score minus FIX's (points; below zero is better)"
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=domain, range=[ARM_COLOURS[a] for a in ARM_LABELS if a != "fix"]
                ),
                legend=alt.Legend(
                    title=None, orient="top", columns=2, labelLimit=330, symbolType="stroke"
                ),
            ),
        )
        .properties(
            width=WIDTH - 60,
            height=230,
            title=_title(
                text=(
                    "The arms differ from FIX mostly beyond two days ahead, "
                    "where the filter matters least"
                ),
                subtitle=(
                    "Each arm's mean quantile score minus FIX's, five decision substations, scored "
                    "year 1 July 2025 to 30 June 2026, outside switching events. Dashed line: same "
                    "as FIX."
                ),
            ),
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(y="y:Q")
    )
    return zero + chart


def figure_sites(*, data: dict) -> alt.TopLevelMixin:
    """GRID minus FIX by substation in the three periods."""
    rows = []
    for label, gains in (
        ("odd months (used to choose)", data["tuning"]["site_gains"]["C1"]),
        ("even months (replicate)", data["replicate"]["site_gains"]["C1"]),
        ("scored year", data["scored"]["site_gains"]["C1"]),
    ):
        rows.extend({"site": site, "period": label, "difference": v} for site, v in gains.items())
    table = pl.DataFrame(rows)
    base = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "difference:Q",
                title=(
                    "GRID minus FIX, quantile score at 12 to 36 hours "
                    "(points; below zero favours GRID)"
                ),
            ),
            y=alt.Y("site:N", title=None, sort=["D5", "D10", "D11", "D14", "D16"]),
            color=alt.Color(
                "period:N",
                scale=alt.Scale(domain=list(PERIOD_COLOURS), range=list(PERIOD_COLOURS.values())),
                legend=alt.Legend(
                    title=None, orient="top", direction="horizontal", labelLimit=400, columns=3
                ),
            ),
        )
    )
    return (_zero_rule() + base).properties(
        width=WIDTH - 90,
        height=170,
        title=_title(
            text="Every substation gained on the odd months, and the gain mostly vanished after",
            subtitle=(
                "Per-substation difference in mean quantile score, GRID minus FIX, at leads 12 to "
                "36 hours. Odd months of 2024-07 to 2025-06 chose the cells; even months and the "
                "scored year are out of sample. Dashed line: no difference."
            ),
        ),
    )


def figure_coverage(*, data: dict) -> alt.TopLevelMixin:
    """Coverage of the 0.95 quantile at 12 to 36 hours, for all rows and for cold nights."""
    rows = []
    for panel, key in (("All rows", "12-36 h"), ("Cold nights", "cold nights 12-36 h")):
        rows.extend(
            {"panel": panel, "arm": ARM_LABELS[d["arm"]], "coverage": d["cov_q95"], "key": d["arm"]}
            for d in data["scored"]["tail"][key]
        )
    table = pl.DataFrame(rows)
    order = [ARM_LABELS[a] for a in ARM_LABELS]
    colours = [ARM_COLOURS[a] for a in ARM_LABELS]
    dots = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=80)
        .encode(
            x=alt.X(
                "coverage:Q",
                title="Share of readings at or below the 0.95 quantile",
                scale=alt.Scale(domain=[0.8, 1.0]),
            ),
            y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
            color=alt.Color("arm:N", scale=alt.Scale(domain=order, range=colours), legend=None),
        )
    )
    nominal = (
        alt.Chart(pl.DataFrame({"x": [0.95]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[0.8, 1.0])))
    )
    return (
        alt.layer(nominal, dots)
        .properties(width=WIDTH - 280, height=150)
        .facet(row=alt.Row("panel:N", title=None), data=table)
        if False
        else alt.vconcat(
            *[
                alt.layer(nominal, dots.transform_filter(alt.datum.panel == panel)).properties(
                    width=WIDTH - 280, height=140, title=panel
                )
                for panel in ("All rows", "Cold nights")
            ]
        ).properties(
            title=_title(
                text="The online variance did not lift cold-night coverage",
                subtitle=(
                    "Share of readings at or below each arm's 0.95 quantile, leads 12 to 36 hours, "
                    "five decision substations, scored year. Cold nights: night hour bands with "
                    "ERA5 temperature at or below 3 degrees C. Dashed line: the 0.95 target."
                ),
            )
        )
    )


def figure_values(*, data: dict) -> alt.TopLevelMixin:
    """The fitted settings per substation in ratio units, with the grid's edge."""
    rows = [
        {
            "site": d["site"],
            "arm": ARM_LABELS[d["arm"]],
            "q": d["q_over_sigma_eps_squared"],
            "z": d["sigma_z_over_sigma_eps"],
        }
        for d in data["fitted_values"]
    ]
    table = pl.DataFrame(rows)
    domain = [ARM_LABELS[a] for a in ("fix", "grid", "pool")]
    points = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "q:Q",
                scale=alt.Scale(type="log", domain=[0.01, 100]),
                title="Level variance per day over observation variance (q / sigma_eps squared)",
            ),
            y=alt.Y(
                "z:Q",
                scale=alt.Scale(type="log", domain=[0.4, 12]),
                title="Transient sd over observation sd",
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=domain, range=[DARK, ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            shape=alt.Shape(
                "arm:N",
                scale=alt.Scale(domain=domain, range=["circle", "diamond", "square"]),
                legend=None,
            ),
        )
    )
    edge = (
        alt.Chart(pl.DataFrame({"y": [float(np.exp(-0.5))]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(y="y:Q")
    )
    return (edge + points).properties(
        width=WIDTH - 90,
        height=260,
        title=_title(
            text="Most chosen settings sit at the edge of the grid",
            subtitle=(
                "One dot per substation and arm (D5, D9, D10, D11, D14, D16). The dashed line is "
                "the grid's lowest transient sd. Every GRID cell touches an edge of the grid: "
                "five sit at the lowest transient sd, and five at the highest decay factor "
                "(0.98 per half-hour) or the lowest (0.73, D10)."
            ),
        ),
    )


def main() -> None:
    """Draw every figure."""
    data = json.loads((common.WORK_DIR / "report" / "report.json").read_text())
    _save(figure_headline(data=data), name="figure_1_headline")
    _save(figure_sites(data=data), name="figure_2_by_site")
    _save(figure_by_lead(data=data), name="figure_3_by_lead")
    _save(figure_coverage(data=data), name="figure_4_coverage")
    _save(figure_values(data=data), name="figure_5_values")


if __name__ == "__main__":
    main()
