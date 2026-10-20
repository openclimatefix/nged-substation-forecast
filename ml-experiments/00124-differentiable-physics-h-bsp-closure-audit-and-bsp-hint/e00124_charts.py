"""Draw the figures of experiment 124 as SVG into the docs folder, from the stored summaries.

Only anonymised results appear: two bulk supply points by position, and Primaries by D-label.
Afterwards optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00124-*/e00124_charts.py
"""

import json
import re
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00124_common import SLUG, WORK_DIR

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"00124-{SLUG}"
)
PNG_DIR: Final[Path] = Path(__file__).resolve().parents[2] / ".claude" / "worktrees" / "e00124_png"
WIDTH: Final[int] = 560
WRAP: Final[int] = 100
CLEAN_LIMIT: Final[float] = 0.023


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
    with (DOCS_DIR / f"{name}.svg").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _weight_label(weight: float) -> str:
    return f"{weight:g}"


FIRST_RUN: Final[str] = "first run"
CORRECTED_RUN: Final[str] = "re-run, scale corrected"


def _tradeoff_frames() -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Return the damage rows, the clean-rise rows, and the weights in order, for both runs."""
    damage_rows = []
    clean_rows = []
    all_weights: set[float] = set()
    for suffix, run in (("", FIRST_RUN), ("_corrected", CORRECTED_RUN)):
        tuning = json.loads((WORK_DIR / f"tuning{suffix}.json").read_text())
        all_weights.update(tuning["weights"])
        for weight, value in zip(tuning["weights"], tuning["summed_damage"], strict=True):
            damage_rows.append({"weight": weight, "damage": value, "run": run})
        for year, file in (
            ("tuning year", f"clean_grid_tuning{suffix}.json"),
            ("replicate year", f"clean_grid_replicate{suffix}.json"),
            ("scored year", f"clean_grid_scored{suffix}.json"),
        ):
            grid = json.loads((WORK_DIR / file).read_text())
            sites = list(grid["clean_mae"].values())
            mean = [sum(s[i] for s in sites) / len(sites) for i in range(len(grid["weights"]))]
            clean_rows += [
                {"weight": w, "cost": v - mean[0], "year": year, "run": run}
                for w, v in zip(grid["weights"], mean, strict=True)
            ]
    labels = [f"{w:g}" for w in sorted(all_weights)]
    return pl.DataFrame(damage_rows), pl.DataFrame(clean_rows), labels


def figure_tradeoff() -> None:
    """Figure 1: the damage from injected errors and the clean-error rise against the weight."""
    damage, clean, labels = _tradeoff_frames()
    del labels
    x = alt.X(
        "weight:Q",
        scale=alt.Scale(type="symlog", constant=0.003, domain=[0, 30]),
        axis=alt.Axis(
            values=[0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], labelAngle=0, format="~g"
        ),
        title="Weight on the BSP reading (0 = no hint: the separate fits; logarithmic axis)",
    )
    dash = alt.StrokeDash(
        "run:N",
        sort=None,
        scale=alt.Scale(domain=[FIRST_RUN, CORRECTED_RUN], range=[[1, 0], [5, 3]]),
        legend=alt.Legend(title=None, orient="top"),
    )
    top = (
        alt.Chart(damage)  # ty: ignore[unresolved-attribute]
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=60, color=ocf.BRAND_ORANGE))
        .encode(
            x=x,
            y=alt.Y(
                "damage:Q",
                scale=alt.Scale(domain=[0, 36]),
                title="Summed damage (points; smaller is better)",
            ),
            strokeDash=dash,
            color=alt.value(ocf.BRAND_ORANGE),
        )
        .properties(
            width=WIDTH,
            height=140,
            title=alt.TitleParams(
                "With the hint, 200 injected metering errors do less damage",
                fontSize=12,
                anchor="start",
            ),
        )
    )
    colours = alt.Color(
        "year:N",
        sort=None,
        scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.DATA_SKY]),
        legend=alt.Legend(title=None, orient="top"),
    )
    lines = (
        alt.Chart(clean)  # ty: ignore[unresolved-attribute]
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=45))
        .encode(
            x=x,
            y=alt.Y(
                "cost:Q",
                scale=alt.Scale(domain=[-0.5, 5.5]),
                title="Extra clean error (points; smaller is better)",
            ),
            color=colours,
            strokeDash=alt.StrokeDash(
                "run:N",
                sort=None,
                scale=alt.Scale(domain=[FIRST_RUN, CORRECTED_RUN], range=[[1, 0], [5, 3]]),
                legend=None,
            ),
        )
    )
    limit = (
        alt.Chart(pl.DataFrame({"limit": [CLEAN_LIMIT]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[2, 2], color=ocf.BRAND_ORANGE)
        .encode(y="limit:Q")
    )
    bottom = alt.layer(lines, limit).properties(
        width=WIDTH,
        height=160,
        title=alt.TitleParams(
            "but every weight above 0 makes clean forecasts worse",
            fontSize=12,
            anchor="start",
        ),
    )
    chart = alt.vconcat(top, bottom).properties(
        title=_title(
            text="The BSP reading cuts damage from injected errors but makes clean forecasts worse",
            subtitle=(
                "Pilot BSP; frozen fits scored by mean absolute "
                "error in points of effective capacity over the year after each cut, averaged "
                "over its MW Primaries. Top: summed extra error from 200 injected errors in the "
                "tuning year. Bottom: clean error minus weight 0's. Solid: first run. Dashed: "
                "re-run after the BSP-row scale was corrected. Dotted line: the 0.023-point limit."
            ),
        )
    )
    _save(chart, name="figure_1_tradeoff")


def figure_plumbing() -> None:
    """Figure 3: arm B's synthetic check, the clean-error rise of an exactly specified system."""
    result = json.loads((WORK_DIR / "armb_plumbing.json").read_text())
    weights = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    labels = [f"{w:g}" for w in weights]
    frame = pl.DataFrame(
        [
            {"weight": label, "rise": value, "fit": name}
            for name, curve in (
                ("multipliers exact", result["median_curve_exact"][1:]),
                ("multipliers 15% wrong", result["median_curve_wrong"][1:]),
            )
            for label, value in zip(labels, curve, strict=True)
        ]
    )
    x = alt.X(
        "weight:N",
        sort=labels,
        axis=alt.Axis(labelAngle=0),
        title="Weight on the synthetic BSP reading",
    )
    lines = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=60))
        .encode(
            x=x,
            y=alt.Y(
                "rise:Q",
                scale=alt.Scale(type="symlog", constant=0.05),
                axis=alt.Axis(values=[0, 0.023, 0.1, 0.3, 1, 3, 10]),
                title="Extra clean error (points; smaller is better)",
            ),
            color=alt.Color(
                "fit:N",
                sort=None,
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    limit = (
        alt.Chart(pl.DataFrame({"limit": [CLEAN_LIMIT]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[5, 3], color="#555555")
        .encode(y="limit:Q")
    )
    chart = alt.layer(lines, limit).properties(
        width=WIDTH,
        height=220,
        title=_title(
            text="Even a synthetic system that closes exactly loses clean accuracy at every weight",
            subtitle=(
                "Median over 20 synthetic systems built from the pilot's fitted Primaries, "
                "with the audit's multipliers and the Primaries' own autocorrelated noise in "
                "the BSP reading. Extra error is the mean absolute error minus the separate "
                "fits' (points of effective capacity; symmetric-logarithmic axis). Dashed "
                "line: the 0.023-point limit."
            ),
        ),
    )
    _save(chart, name="figure_3_plumbing")


def _parse_audit() -> pl.DataFrame:
    """Read the closure coefficients and their 80% intervals from the audit report."""
    text = (DOCS_DIR / "audit_report.md").read_text()
    rows = []
    bsp = cut = ""
    for line in text.splitlines():
        if line.startswith("### Bulk supply point"):
            bsp = "Bulk supply point " + line.rsplit(" ", 1)[1]
        elif line.startswith("#### Cut"):
            cut = line.split()[2]
        elif line.startswith("- step penalty"):
            for name, pattern in (
                ("a_P (Primaries)", r"a_P ([\d.-]+) \(([\d.-]+) to ([\d.-]+)\)"),
                ("a_G (generation)", r"a_G ([\d.-]+) \(([\d.-]+) to ([\d.-]+)\)"),
                ("c (MVA Primaries)", r"; c ([\d.-]+) \(([\d.-]+) to ([\d.-]+)\)"),
            ):
                match = re.search(pattern, line)
                if match:
                    value, low, high = (float(g) for g in match.groups())
                    rows.append(
                        {
                            "bsp": bsp,
                            "cut": f"{cut} cut",
                            "coefficient": name,
                            "value": value,
                            "low": low,
                            "high": high,
                        }
                    )
    return pl.DataFrame(rows)


def figure_coefficients() -> None:
    """Figure 2: the closure coefficients of the audit, with 80% month-block intervals."""
    frame = _parse_audit()
    colour = alt.Color(
        "coefficient:N",
        sort=None,
        scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    panels = []
    for bsp in sorted(frame["bsp"].unique().to_list()):
        part = frame.filter(pl.col("bsp") == bsp)
        base = alt.Chart(part).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("cut:N", sort=None, title=None)
        )
        rule = base.mark_rule(strokeWidth=2).encode(
            x=alt.X("low:Q", scale=alt.Scale(domain=[0, 1.6]), title="Fitted coefficient"),
            x2="high:Q",
            color=colour,
            yOffset="coefficient:N",
        )
        dots = base.mark_point(filled=True, size=60).encode(
            x="value:Q", color=colour, yOffset="coefficient:N"
        )
        one = (
            alt.Chart(pl.DataFrame({"one": [1.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#555555")
            .encode(x="one:Q")
        )
        panels.append(
            alt.layer(one, rule, dots).properties(
                width=WIDTH,
                height=110,
                title=alt.TitleParams(bsp, fontSize=12, anchor="start"),
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=_title(
            text=(
                "One BSP closes only up to factors of 0.85 to 1.16, and the other does not close "
                "with the generation series available"
            ),
            subtitle=(
                "Closure audit fitted on rows before each cut. Dot: fitted coefficient. Line: 80% "
                "interval from resampling whole months. Dashed line: 1, the value the joint model "
                "fixes. The first panel is the BSP that does not close (an 11-point residual). "
                "A cut without a dot has fewer than one year of complete rows."
            ),
        )
    )
    _save(chart, name="figure_2_coefficients")


if __name__ == "__main__":
    figure_tradeoff()
    figure_coefficients()
    figure_plumbing()
