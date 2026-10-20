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
    frame = frame.with_columns(band_low=pl.lit(70.0), band_high=pl.lit(90.0))
    band = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_rect(color="#EEEEEE")
        .encode(x="band_low:Q", x2="band_high:Q")
    )
    base = alt.Chart()
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Share of intervals that hold the truth (%; 80 is calibrated)"),
        x2="high:Q",
        y=y,
        color=colour,
    )
    dots = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x="coverage:Q", y=y, color=colour
    )
    layered = alt.layer(band, bars, dots, data=frame).properties(width=WIDTH, height=70)
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
                text="The graph prior leaves scale and capacity intervals overconfident",
                subtitle=(
                    "Share of 80% intervals that hold the true value, over 120 synthetic fleets "
                    "per panel. Dot: estimate. Line: 95% range over fleets. Grey band: 70 to 90%. "
                    "Smooth: neighbours alike. Unlike: neighbours not alike."
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
    frame = frame.with_columns(one=pl.lit(1.0))
    rule = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="one:Q")
    )
    base = alt.Chart()
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
        alt.layer(rule, bars, dots, data=frame)
        .properties(width=WIDTH, height=140)
        .facet(row=alt.Row("cell:N", title=None))
        .properties(
            title=_title(
                text="The graph prior narrows heating-slope intervals and little else",
                subtitle=(
                    "Dot: median ratio over fleets. Line: 98.3% interval from resampling fleets. H "
                    "against G, NH against N, on the same parameters of the same fleets."
                ),
            )
        )
    )


CONTRAST_ORDER: Final[tuple[str, ...]] = (
    "R1 H - G",
    "R2 NH - N",
    "R3 P-hol - G, event days",
    "R3b P-hol - G, other days",
    "R1w H - G, winter",
    "R2w NH - N, winter",
    "R3w P-hol - G, winter event days",
    "H - G, winter evening peak",
    "NH - N, winter evening peak",
    "H - G, cold snaps",
    "NH - N, cold snaps",
)
CONTRAST_LABELS: Final[dict[str, str]] = {
    "R1 H - G": "R1  H minus G (graph prior alone)",
    "R2 NH - N": "R2  NH minus N (graph on top of fleet mean)",
    "R3 P-hol - G, event days": "R3  P-hol minus G, event days",
    "R3b P-hol - G, other days": "R3b  P-hol minus G, other days",
    "R1w H - G, winter": "R1w  H minus G, November to March",
    "R2w NH - N, winter": "R2w  NH minus N, November to March",
    "R3w P-hol - G, winter event days": "R3w  P-hol minus G, winter event days",
    "H - G, winter evening peak": "H minus G, winter evening peak",
    "NH - N, winter evening peak": "NH minus N, winter evening peak",
    "H - G, cold snaps": "H minus G, cold-snap days",
    "NH - N, cold snaps": "NH minus N, cold-snap days",
}


def figure_contrasts(*, report: dict) -> alt.TopLevelMixin:
    """The planned and winter contrasts with their intervals, scored year and replicate year."""
    rows = []
    for cut, year in (
        ("scored", "Scored year (2025-26)"),
        ("replicate", "Replicate year (2024-25)"),
    ):
        for row in report[cut]["contrasts"]:
            if row["metric"] != "qs" or row["name"] not in CONTRAST_ORDER or "lower" not in row:
                continue
            rows.append(
                {
                    "contrast": CONTRAST_LABELS[row["name"]],
                    "year": year,
                    "estimate": row["difference"],
                    "low": row["lower"],
                    "high": row["upper"],
                }
            )
    data = pl.DataFrame(rows)
    colour = alt.Color(
        "year:N",
        scale=alt.Scale(
            domain=["Scored year (2025-26)", "Replicate year (2024-25)"],
            range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
        ),
        legend=alt.Legend(orient="top", title=None),
    )
    y = alt.Y(
        "contrast:N",
        sort=[CONTRAST_LABELS[n] for n in CONTRAST_ORDER],
        title=None,
        axis=alt.Axis(labelLimit=330),
    )
    offset = alt.YOffset("year:N")
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    margin = (
        alt.Chart(pl.DataFrame({"x": [0.03]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#B0B0B0", strokeDash=[1, 3])
        .encode(x="x:Q")
    )
    base = alt.Chart(data)
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Quantile score difference (points; negative is better)"),
        x2="high:Q",
        y=y,
        yOffset=offset,
        color=colour,
    )
    dots = base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
        x="estimate:Q", y=y, yOffset=offset, color=colour
    )
    return (rule + margin + bars + dots).properties(
        width=WIDTH,
        height=330,
        title=_title(
            text="The graph prior costs a little on average and more in winter evenings",
            subtitle=(
                "Dot: estimate. Line: 98.75% interval for R1 to R3b, 95% for the rest, from "
                "resampling whole months (weeks for winter, days for event and cold-snap days). "
                "First arm minus second; 0 is equal; dotted line: the +0.03 margin."
            ),
        ),
    )


def figure_absolute(*, report: dict, cut: str) -> alt.TopLevelMixin:
    """Absolute quantile scores by arm and scope for one year."""
    arms = ["g", "n", "h_lo", "nh_lo", "phol", "manual_heuristic", "climatology"]
    labels = {**ARM_LABELS, "manual_heuristic": "Manual heuristic", "climatology": "Climatology"}
    scopes = {
        "all": "All days",
        "winter": "November to March",
        "winter_evening_peak": "Winter evening peak",
        "event": "Event days",
        "cold_snap": "Cold-snap days",
    }
    rows = [
        {"arm": labels[arm], "scope": label, "score": report[cut]["scores"][arm]["by_scope"][scope]}
        for arm in arms
        for scope, label in scopes.items()
    ]
    data = pl.DataFrame(rows)
    colour = alt.Color(
        "arm:N",
        sort=[labels[a] for a in arms],
        scale=alt.Scale(
            domain=[labels[a] for a in arms],
            range=[
                ocf.BRAND_ORANGE,
                ocf.DATA_BLUE,
                ocf.DATA_PURPLE,
                ocf.DATA_GREEN,
                ocf.DATA_SKY,
                "#808080",
                "#222222",
            ],
        ),
        legend=alt.Legend(orient="top", title=None, columns=2),
    )
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X("score:Q", title="Quantile score (points of capacity; smaller is better)"),
            y=alt.Y("scope:N", sort=list(scopes.values()), title=None),
            color=colour,
        )
        .properties(
            width=WIDTH,
            height=170,
            title=_title(
                text=f"Absolute scores by arm and kind of day, {cut} year",
                subtitle="Mean over the five decision sites, outside switching events.",
            ),
        )
    )


def figure_real_width(*, real: dict[str, dict]) -> alt.TopLevelMixin:
    """Real interval widths against G, with H on the true graph beside H on permuted graphs."""
    rows = []
    for cut, part in real.items():
        for arm, families in part["width_ratio_to_g"].items():
            kind = "H on permuted graphs" if arm.startswith("h_perm") else ARM_LABELS.get(arm, arm)
            for family, ratio in families.items():
                rows.append(
                    {
                        "cut": cut,
                        "arm": arm,
                        "kind": kind,
                        "family": FAMILY_LABELS[family],
                        "ratio": ratio,
                    }
                )
    data = pl.DataFrame(rows)
    order = [ARM_LABELS["n"], ARM_LABELS["h_lo"], ARM_LABELS["nh_lo"], "H on permuted graphs"]
    colour = alt.Color(
        "kind:N",
        sort=order,
        scale=alt.Scale(
            domain=order,
            range=[ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.DATA_GREEN, "#808080"],
        ),
        legend=alt.Legend(orient="top", title=None, columns=2),
    )
    data = data.with_columns(one=pl.lit(1.0))
    rule = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="one:Q")
    )
    dots = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=50)
        .encode(
            x=alt.X(
                "ratio:Q", title="Median 80% interval width as a share of G's (below 1 is narrower)"
            ),
            y=alt.Y("family:N", sort=list(FAMILY_LABELS.values()), title=None),
            yOffset=alt.YOffset("arm:N"),
            color=colour,
        )
    )
    return (
        alt.layer(rule, dots, data=data)
        .properties(width=WIDTH, height=150)
        .facet(row=alt.Row("cut:N", title=None))
        .properties(
            title=_title(
                text="The graph prior narrows real slope intervals as much as a permuted graph",
                subtitle="Dot: median over identified parameters; three permuted graphs.",
            )
        )
    )


def main() -> None:
    """Draw the figures that exist for the files found."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-tag", default="gate")
    arguments = parser.parse_args()
    report_path = common.WORK_DIR / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        _save(figure_contrasts(report=report), name="fig_contrasts")
        for cut in ("scored", "replicate"):
            _save(figure_absolute(report=report, cut=cut), name=f"fig_absolute_{cut}")
    real = {
        cut: json.loads((common.WORK_DIR / f"real_intervals_{cut}.json").read_text())
        for cut in ("scored", "replicate")
        if (common.WORK_DIR / f"real_intervals_{cut}.json").exists()
    }
    if real:
        _save(figure_real_width(real=real), name="fig_real_width")
    gate_path = common.WORK_DIR / "gate" / f"{arguments.gate_tag}_report.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text())
        _save(figure_coverage(gate=gate, cells=("smooth", "unlike")), name="fig_gate_coverage")
        _save(figure_width_ratio(gate=gate, cells=("smooth", "unlike")), name="fig_gate_width")


if __name__ == "__main__":
    main()
