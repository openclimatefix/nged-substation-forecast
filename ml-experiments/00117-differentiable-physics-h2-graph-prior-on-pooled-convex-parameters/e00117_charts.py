"""Draw the figures of experiment 117 as SVG into the docs folder, from the report files.

Only Primary substations appear, under their D-labels, and the two excluded series never appear.
The numbers come from `work/report.json` (`e00117_report.py`), the gate report
(`e00117_gate_report.py`), and `work/real_intervals_*.json`, so a figure and the page quote the
same values. Afterwards optimise each SVG with `npx svgo@4 --multipass --precision=1
--final-newline`.

    uv run python ml-experiments/00117-*/e00117_charts.py [--gate-tag gate]
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00117_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = Path.home() / ".claude" / "scratch" / "e00117_png"
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
ARM_LABELS: Final[dict[str, str]] = {
    "g": "G: each site alone",
    "n": "N: fleet mean",
    "h_lo": "H: graph prior",
    "nh_lo": "NH: fleet mean and graph",
    "h_hi": "H at the next gamma",
    "nh_hi": "NH at the next gamma",
    "phol": "P-hol: holiday fleet mean",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "g": ocf.BRAND_ORANGE,
    "n": ocf.DATA_BLUE,
    "h_lo": ocf.DATA_PURPLE,
    "nh_lo": ocf.DATA_GREEN,
}
FAMILY_LABELS: Final[dict[str, str]] = {
    "slope": "Heating slopes",
    "scale": "Profile scale",
    "capacity": "Capacity",
    "holiday": "Holiday level",
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    with (DOCS_DIR / f"{name}.svg").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def figure_coverage(*, gate: dict, cells: tuple[str, ...]) -> alt.TopLevelMixin:
    """Coverage of the 80% intervals by arm and family, with fleet-clustered standard errors."""
    frame = pl.DataFrame(gate["coverage"]).filter(
        pl.col("cell").is_in(cells) & pl.col("arm").is_in(list(ARM_COLOURS))
    )
    frame = frame.group_by("cell", "arm", "family").agg(
        pl.col("coverage").mean() * 100.0,
        (pl.col("coverage_se").mean() * 196.0).alias("half"),
    )
    frame = frame.with_columns(
        low=pl.col("coverage") - pl.col("half"),
        high=pl.col("coverage") + pl.col("half"),
        label=pl.col("arm").replace_strict(ARM_LABELS),
        family_label=pl.col("family").replace_strict(FAMILY_LABELS),
    )
    colour = alt.Color(
        "label:N",
        scale=alt.Scale(
            domain=[ARM_LABELS[a] for a in ARM_COLOURS], range=list(ARM_COLOURS.values())
        ),
        legend=alt.Legend(orient="top", title=None),
    )
    y = alt.Y("label:N", sort=[ARM_LABELS[a] for a in ARM_COLOURS], title=None, axis=None)
    band = (
        alt.Chart(pl.DataFrame({"low": [70.0], "high": [90.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color="#EEEEEE")
        .encode(x="low:Q", x2="high:Q")
    )
    base = alt.Chart(frame)
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Share of intervals that hold the truth (%; 80 is calibrated)"),
        x2="high:Q",
        y=y,
        color=colour,
    )
    dots = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x="coverage:Q", y=y, color=colour
    )
    layered = (band + bars + dots).properties(width=WIDTH, height=70)
    return (
        layered.facet(
            row=alt.Row(
                "family_label:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")
            ),
            column=alt.Column("cell:N", title=None),
        )
        .resolve_scale(x="shared")
        .properties(
            title=_title(
                text="Coverage of the 80% parameter intervals, by arm",
                subtitle=(
                    "Dot: coverage. Line: 95% range over fleets. Grey band: 70 to 90%. "
                    "Synthetic fleets with the real noise; smooth: neighbours alike; unlike: not."
                ),
            )
        )
    )


def figure_width_ratio(*, gate: dict, cells: tuple[str, ...]) -> alt.TopLevelMixin:
    """The median interval width of H and NH against G and N, by family."""
    frame = pl.DataFrame(gate["width_ratio"]).filter(
        pl.col("cell").is_in(cells) & pl.col("arm").is_in(["h_lo", "nh_lo"])
    )
    frame = frame.with_columns(
        label=pl.col("arm").replace_strict(ARM_LABELS),
        family_label=pl.col("family").replace_strict(FAMILY_LABELS),
    )
    colour = alt.Color(
        "label:N",
        scale=alt.Scale(
            domain=[ARM_LABELS["h_lo"], ARM_LABELS["nh_lo"]],
            range=[ocf.DATA_PURPLE, ocf.DATA_GREEN],
        ),
        legend=alt.Legend(orient="top", title=None),
    )
    y = alt.Y("family_label:N", sort=list(FAMILY_LABELS.values()), title=None)
    offset = alt.YOffset("label:N")
    rule = (
        alt.Chart(pl.DataFrame({"x": [1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    base = alt.Chart(frame)
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "low:Q", title="Interval width as a share of the reference arm's (below 1 is narrower)"
        ),
        x2="high:Q",
        y=y,
        yOffset=offset,
        color=colour,
    )
    dots = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x="median_ratio:Q", y=y, yOffset=offset, color=colour
    )
    return (
        (rule + bars + dots)
        .properties(width=WIDTH, height=140)
        .facet(row=alt.Row("cell:N", title=None))
        .properties(
            title=_title(
                text="The graph prior narrows heating-slope intervals",
                subtitle=(
                    "Dot: median ratio over fleets. Line: 98.3% interval from resampling fleets. H "
                    "against G, NH against N, on the same parameters of the same fleets."
                ),
            )
        )
    )


def main() -> None:
    """Draw the figures that exist for the files found."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-tag", default="gate")
    arguments = parser.parse_args()
    gate_path = common.WORK_DIR / "gate" / f"{arguments.gate_tag}_report.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text())
        _save(figure_coverage(gate=gate, cells=("smooth", "unlike")), name="fig_gate_coverage")
        _save(figure_width_ratio(gate=gate, cells=("smooth", "unlike")), name="fig_gate_width")


if __name__ == "__main__":
    main()
