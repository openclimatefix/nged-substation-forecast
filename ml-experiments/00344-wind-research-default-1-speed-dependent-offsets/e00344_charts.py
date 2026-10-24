"""Draw experiment 00344's figures as SVG. Reads `work/score.json`, `work/final/`, and the stats.

Figure 1: each arm minus the default on four row sets. Figure 2: the fitted offsets by forecast
speed. Figure 3: forecast and observed median power by forecast speed. Figure 4: each arm minus the
default by forecast-speed group. Afterwards optimise with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00344-*/e00344_charts.py   (PYTHONPATH: see common)
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00344_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

c303 = common.c303
c336 = common.c336
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00344-wind-research-default-1-speed-dependent-offsets"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 95
GREY: Final[str] = "#777777"
LEVEL: Final[str] = "0.9967"
SCOPE: Final[str] = "3 wind farms, 3 seeds, scored at 1 to 10 days."
ARM_LABELS: Final[dict[str, str]] = {
    "x2": "x2: smooth curve in speed",
    "x1": "x1: offset per speed bin",
    "x3": "x3: shared curve, scale per farm",
    "x2drift": "post hoc: x2 fitted on 2025 H1 drift",
    "x2oracle": "oracle: x2 fitted on the scored year",
    "Sdrift": "post hoc: constant shift (experiment 340)",
    "x2shuf": "placebo: x2 with shuffled speed",
}
GROUP_LABELS: Final[dict[str, str]] = {
    "speed_lt5": "Below 5 m/s",
    "speed_5_8": "5 to 8 m/s",
    "speed_8_15": "8 to 15 m/s",
    "speed_ge15": "15 m/s or more",
}
BIN: Final[float] = 1.0


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _kind(key: str, adoption: dict) -> str:
    if key == "x2shuf":
        return "placebo"
    return "adoptable" if key in adoption else "descriptive"


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """Each arm minus the default on the kept rows, all half-hours, and two low-wind row sets."""
    panels = []
    sets = (
        ("kept", "Kept rows (the headline)"),
        ("all", "All half-hours"),
        ("mean_masked", "Rows with forecast speed below cut-in"),
        ("era5_masked", "Rows with ERA5 wind below cut-in"),
    )
    for subset, text in sets:
        rows = []
        for key, label in ARM_LABELS.items():
            found = score["contrasts"]["scored"][key][f"{subset} | qs"]
            low, high = found[LEVEL]
            rows.append(
                {
                    "arm": label,
                    "estimate": found["mean"],
                    "lower": low,
                    "upper": high,
                    "kind": _kind(key, score["adoption"]),
                }
            )
        data = pl.DataFrame(rows)
        base = alt.Chart(data).encode(
            y=alt.Y("arm:N", sort=list(data["arm"]), title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["adoptable", "descriptive", "placebo"],
                    range=[ocf.DATA_BLUE, ocf.DATA_PURPLE, GREY],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        points = base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "estimate:Q",
                title="Score minus the default's (points of capacity; lower is better)",
            )
        )
        bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
        zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bars, points).properties(
                width=WIDTH,
                height=24 * data.height,
                title=_title(
                    text=text,
                    lines=[
                        "Scored year, quantile score minus Wind farms research default 1's. Dot: "
                        "estimate. Line: 99.67% interval from re-drawing whole months and seeds. "
                        + SCOPE
                    ],
                ),
            )
        )
    return alt.vconcat(*panels).resolve_scale(x="independent")


def figure_2() -> alt.TopLevelMixin:
    """The fitted correction by forecast speed: x2 curves, x1 bins, and the post hoc drift curve."""
    fits = json.loads((common.FINAL_DIR / "fits.json").read_text())["seed0"]["main"]
    speeds = np.arange(0.0, 25.01, 0.25)
    levels = {1: "q05", 5: "q50", 9: "q95"}
    rows, bins = [], []
    centres = [1.5, 4.5, 7.5, 10.5, 13.5, 17.0]
    for farm in common.FARMS:
        for k, name in levels.items():
            for kind in ("x2", "x2drift"):
                value = common.curve_value(params=tuple(fits[kind][farm][k]), speed=speeds)
                rows += [
                    {"farm": farm, "level": name, "kind": kind, "speed": s, "offset": v * 100}
                    for s, v in zip(speeds, value, strict=True)
                ]
            for bin_index, centre in enumerate(centres):
                values = [
                    np.array(offset)[k] * 100
                    for cell, offset in fits["x1_table"].items()
                    if int(cell) // 1000 == common.FARMS.index(farm) and int(cell) % 10 == bin_index
                ]
                if values:
                    bins.append(
                        {
                            "farm": farm,
                            "level": name,
                            "speed": centre,
                            "offset": float(np.mean(values)),
                        }
                    )
    line_data, bin_data = pl.DataFrame(rows), pl.DataFrame(bins)
    colour = alt.Color(
        "level:N",
        scale=alt.Scale(
            domain=["q05", "q50", "q95"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]
        ),
        legend=alt.Legend(title="Quantile", orient="top"),
    )
    panels = []
    for farm in common.FARMS:
        lines = (
            alt.Chart(line_data.filter(pl.col("farm") == farm))
            .mark_line()
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("speed:Q", title="Ensemble-mean 100 m wind speed (m/s)"),
                y=alt.Y("offset:Q", title="Added to the quantile (points of capacity)"),
                color=colour,
                strokeDash=alt.StrokeDash(
                    "kind:N",
                    scale=alt.Scale(domain=["x2", "x2drift"], range=[[1, 0], [5, 4]]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
        )
        points = (
            alt.Chart(bin_data.filter(pl.col("farm") == farm))
            .mark_point(filled=True, size=40, opacity=0.8)
            .encode(x="speed:Q", y="offset:Q", color=colour)  # ty: ignore[unresolved-attribute]
        )
        zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=GREY).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(zero, lines, points).properties(width=240, height=200, title=farm))
    return alt.hconcat(*panels).properties(
        title=_title(
            text="The fitted correction by forecast wind speed",
            lines=[
                "Lines: the x2 curve (solid) and its post hoc version fitted on 2025 H1 (dashed). "
                "Dots: the x1 offset per speed bin, averaged over lead days. Seed 0, " + SCOPE
            ],
        )
    )


def _binned_median(*, speed: np.ndarray, value: np.ndarray, edges: np.ndarray) -> np.ndarray:
    index = np.digitize(speed, edges) - 1
    out = np.full(edges.size - 1, np.nan)
    for b in range(edges.size - 1):
        picked = value[index == b]
        if picked.size >= 30:
            out[b] = np.median(picked)
    return out


def figure_3() -> alt.TopLevelMixin:
    """Forecast median power and observed median power, by forecast speed, per farm."""
    sites = c303.site_table()
    cases = c303.cases_table(sites=sites)
    ordered = pl.concat([cases.filter(pl.col("site") == f) for f in common.FARMS])
    speed = np.concatenate(
        [np.load(c336.stats_path(kind="speed", set_name="scored", farm=f)) for f in common.FARMS]
    )[:, 0]
    sitecol = ordered["site"].to_numpy()
    actual = ordered["actual_norm"].to_numpy().astype(np.float64) * 100
    kept = ~ordered["masked"].to_numpy()
    curves = {
        "default": np.load(common.FINAL_DIR / "default.npy"),
        "x2": np.load(common.FINAL_DIR / "x2.npy"),
        "x2drift": np.load(common.FINAL_DIR / "x2drift.npy"),
    }
    edges = np.arange(0.0, 26.0 + BIN, BIN)
    centres = (edges[:-1] + edges[1:]) / 2
    rows = []
    for farm in common.FARMS:
        pick = (sitecol == farm) & kept
        observed = _binned_median(speed=speed[pick], value=actual[pick], edges=edges)
        series = {"observed median": observed}
        for name, array in curves.items():
            median = array.mean(axis=0)[:, 5].astype(np.float64) * 100
            series[name] = _binned_median(speed=speed[pick], value=median[pick], edges=edges)
        for name, values in series.items():
            rows += [
                {"farm": farm, "series": name, "speed": c, "power": v}
                for c, v in zip(centres, values, strict=True)
                if np.isfinite(v)
            ]
    data = pl.DataFrame(rows)
    panels = [
        (
            alt.Chart(data.filter(pl.col("farm") == farm))
            .mark_line(point=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("speed:Q", title="Ensemble-mean 100 m wind speed (m/s)"),
                y=alt.Y("power:Q", title="Median power (% of capacity)"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=["observed median", "default", "x2", "x2drift"],
                        range=[GREY, ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
            .properties(width=240, height=220, title=farm)
        )
        for farm in common.FARMS
    ]
    return alt.hconcat(*panels).properties(
        title=_title(
            text="Median power by forecast wind speed, kept rows",
            lines=[
                (
                    "Median of the forecast median (three seeds averaged) and of the observed "
                    "power within 1 m/s bins of the ensemble-mean speed. Bins under 30 half-hours "
                    "are left out. Scored year."
                )
            ],
        )
    )


def figure_4(*, score: dict) -> alt.TopLevelMixin:
    """Each arm minus the default in the four forecast-speed groups, all rows in the group."""
    rows = []
    for key, label in ARM_LABELS.items():
        if key == "Sdrift":
            continue
        for group, text in GROUP_LABELS.items():
            found = score["contrasts"]["scored"][key][f"{group} | qs"]
            low, high = found[LEVEL]
            rows.append(
                {
                    "arm": label,
                    "group": text,
                    "estimate": found["mean"],
                    "lower": low,
                    "upper": high,
                    "kind": _kind(key, score["adoption"]),
                }
            )
    data = pl.DataFrame(rows)
    arms = [v for k, v in ARM_LABELS.items() if k != "Sdrift"]
    panels = []
    for position, group in enumerate(GROUP_LABELS.values()):
        base = alt.Chart(data.filter(pl.col("group") == group)).encode(
            y=alt.Y(
                "arm:N",
                sort=arms,
                title=None,
                axis=alt.Axis(labelLimit=320) if position == 0 else None,
            ),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["adoptable", "descriptive", "placebo"],
                    range=[ocf.DATA_BLUE, ocf.DATA_PURPLE, GREY],
                ),
                legend=alt.Legend(title=None, orient="top") if position == 0 else None,
            ),
        )
        points = base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("estimate:Q", title="Score minus the default's (points)")
        )
        bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
        zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(zero, bars, points).properties(width=170, height=170, title=group))
    return (
        alt.hconcat(*panels)
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text="Score change by forecast wind speed",
                lines=[
                    "Quantile score minus the default's (lower is better), all half-hours in each "
                    "speed group. Line: 99.67% interval. " + SCOPE
                ],
            )
        )
    )


def main() -> None:
    """Write the figures."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_1(score=score).save(DOCS_DIR / "figure_1.svg")
    figure_2().save(DOCS_DIR / "figure_2.svg")
    figure_3().save(DOCS_DIR / "figure_3.svg")
    figure_4(score=score).save(DOCS_DIR / "figure_4.svg")


if __name__ == "__main__":
    main()
