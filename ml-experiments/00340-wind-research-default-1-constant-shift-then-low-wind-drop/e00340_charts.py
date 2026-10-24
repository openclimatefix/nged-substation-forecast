"""Draw experiment 00340's figures as SVG. Reads `work/score.json` and `work/final/`.

Figure 1: each arm's score minus the default's on four row sets. Figures 2 to 4: the power curves
(full range, 0 to 8 m/s, and forecast minus observed median).
Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00340-*/e00340_charts.py   (PYTHONPATH: see common)
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt

import e00326_score as score326  # isort: skip  # ty: ignore[unresolved-import]
import e00340_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

c303 = common.c303
c336 = common.c336
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00340-wind-research-default-1-constant-shift-then-low-wind-drop"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 95
GREY: Final[str] = "#777777"
LEVEL: Final[str] = "0.9975"
SCOPE: Final[str] = "3 wind farms, 3 seeds, scored at 1 to 10 days."
ARM_LABELS: Final[dict[str, str]] = {
    "S": "S: constant shift",
    "Sq": "S-q: shift per quantile",
    "SX2": "S+X2: x2 plus a shift",
    "X2": "X2: low-wind rows dropped",
    "Sdrift": "post hoc: shift fitted on 2025 H1 drift",
    "SX2drift": "post hoc: x2 plus a drift shift",
    "Soracle": "oracle: shift fitted on the scored year",
    "Srev": "placebo: S with its sign reversed",
}
CURVE_ARMS: Final[dict[str, str]] = {
    "default": GREY,
    "S": ocf.DATA_BLUE,
    "X2": ocf.BRAND_ORANGE,
    "SX2": ocf.DATA_GREEN,
    "Soracle": ocf.DATA_PURPLE,
}
CURVE_LABELS: Final[dict[str, str]] = {
    "default": "default",
    "S": "S",
    "X2": "X2",
    "SX2": "S+X2",
    "Soracle": "oracle shift",
}
PANELS: Final[tuple[str, ...]] = ("W1", "W2", "W3", "All three farms")
BIN: Final[float] = 0.5


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """Each arm minus the default on the kept rows, all half-hours, and the low-wind row sets."""
    panels = []
    sets = (
        ("kept", "Kept rows (the headline)"),
        ("all", "All half-hours"),
        ("era5_masked", "Rows with ERA5 wind below cut-in"),
        ("idle210", "Experiment 210 idle rows"),
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
                    "kind": "placebo"
                    if key == "Srev"
                    else ("adoptable" if key in score["adoption"] else "descriptive"),
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
                        "estimate. Line: 99.75% interval from re-drawing whole months and seeds. "
                        + SCOPE
                    ],
                ),
            )
        )
    return alt.vconcat(*panels).resolve_scale(x="independent")


def _binned(
    *, speed: np.ndarray, value: np.ndarray, edges: np.ndarray, quantiles: tuple[float, ...]
) -> np.ndarray:
    """Return per-bin quantiles of `value` (bins, len(quantiles)); NaN where a bin has < 30 rows."""
    index = np.digitize(speed, edges) - 1
    out = np.full((edges.size - 1, len(quantiles)), np.nan)
    for b in range(edges.size - 1):
        picked = value[index == b]
        picked = picked[np.isfinite(picked)]
        if picked.size >= 30:
            out[b] = np.quantile(picked, quantiles)
    return out


def curve_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """Return the observed, forecast, and difference bins and the per-farm marks (scored year)."""
    sites = c303.site_table()
    cases = c303.cases_table(sites=sites)
    cases = cases.join(  # same order as the stored stats: farms in order, then c303's case order
        pl.DataFrame({"site": list(common.FARMS)}), on="site", how="semi"
    )
    ordered = pl.concat([cases.filter(pl.col("site") == f) for f in common.FARMS])
    speed = np.concatenate(
        [np.load(c336.stats_path(kind="speed", set_name="scored", farm=f)) for f in common.FARMS]
    )[:, 0]
    era5 = np.concatenate(
        [np.load(c336.stats_path(kind="era5", set_name="scored", farm=f)) for f in common.FARMS]
    )
    if ordered.height != speed.size:
        msg = "scored cases do not line up with the stored speeds"
        raise ValueError(msg)
    sitecol = ordered["site"].to_numpy()
    actual = ordered["actual_norm"].to_numpy().astype(np.float64) * 100
    forecast = {
        arm: np.load(common.FINAL_DIR / f"{arm}.npy").mean(axis=0).astype(np.float64) * 100
        for arm in CURVE_ARMS
    }
    edges = np.arange(0.0, 26.0 + BIN, BIN)
    centres = (edges[:-1] + edges[1:]) / 2
    rated = score326.rated_speeds(sites=sites)
    cut = c336.cut_in_speeds()
    observed_rows, forecast_rows, diff_rows = [], [], []
    marks = {}
    for panel in PANELS:
        pick = np.ones(sitecol.size, dtype=bool) if panel == "All three farms" else sitecol == panel
        obs = _binned(
            speed=era5[pick], value=actual[pick], edges=edges, quantiles=(0.05, 0.5, 0.95)
        )
        for b, x in enumerate(centres):
            if np.isfinite(obs[b, 0]):
                observed_rows.append(
                    {
                        "panel": panel,
                        "speed": x,
                        "q05": obs[b, 0],
                        "q50": obs[b, 1],
                        "q95": obs[b, 2],
                    }
                )
        obs_by_forecast_speed = _binned(
            speed=speed[pick], value=actual[pick], edges=edges, quantiles=(0.5,)
        )
        for arm, q in forecast.items():
            fq = _binned(speed=speed[pick], value=q[pick, 5], edges=edges, quantiles=(0.5,))
            lo = _binned(speed=speed[pick], value=q[pick, 1], edges=edges, quantiles=(0.5,))
            hi = _binned(speed=speed[pick], value=q[pick, 9], edges=edges, quantiles=(0.5,))
            for b, x in enumerate(centres):
                if np.isfinite(fq[b, 0]):
                    forecast_rows.append(
                        {
                            "panel": panel,
                            "arm": CURVE_LABELS[arm],
                            "speed": x,
                            "q05": lo[b, 0],
                            "q50": fq[b, 0],
                            "q95": hi[b, 0],
                        }
                    )
                    diff_rows.append(
                        {
                            "panel": panel,
                            "arm": CURVE_LABELS[arm],
                            "speed": x,
                            "difference": fq[b, 0] - obs_by_forecast_speed[b, 0],
                        }
                    )
        if panel == "All three farms":
            marks[panel] = {
                "cut_in": float(np.mean(list(cut.values()))),
                "rated": float(np.mean(list(rated.values()))),
                "idle": None,
            }
        else:
            low_wind = pick & (era5 < 2.0)
            band = np.quantile(actual[low_wind], [0.05, 0.95])
            marks[panel] = {"cut_in": cut[panel], "rated": rated[panel], "idle": band.tolist()}
    return pl.DataFrame(observed_rows), pl.DataFrame(forecast_rows), pl.DataFrame(diff_rows), marks


def _curve_panels(
    *,
    observed: pl.DataFrame,
    fc: pl.DataFrame,
    marks: dict,
    x_max: float,
    title: alt.TitleParams,
    quantile_arms: tuple[str, ...],
) -> alt.TopLevelMixin:
    rows = []
    domain = list(CURVE_LABELS.values())
    colours = list(CURVE_ARMS.values())
    charts = []
    for panel in PANELS:
        mark = marks[panel]
        scale_x = alt.Scale(domain=[0, x_max])
        obs = observed.filter((pl.col("panel") == panel) & (pl.col("speed") <= x_max))
        band = (
            alt.Chart(obs)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.25, color="black")
            .encode(
                x=alt.X("speed:Q", scale=scale_x, title="100 m wind speed (m/s)"),
                y="q05:Q",
                y2="q95:Q",
            )
        )
        median = (
            alt.Chart(obs)  # ty: ignore[unresolved-attribute]
            .mark_line(color="black", strokeWidth=2)
            .encode(
                x=alt.X("speed:Q", scale=scale_x), y=alt.Y("q50:Q", title="Power (% of capacity)")
            )
        )
        data = fc.filter((pl.col("panel") == panel) & (pl.col("speed") <= x_max))
        color = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=domain, range=colours),
            legend=alt.Legend(title=None, orient="top"),
        )
        lines = (
            alt.Chart(data)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1.6)
            .encode(x=alt.X("speed:Q", scale=scale_x), y="q50:Q", color=color)
        )
        tails = (
            alt.Chart(data.filter(pl.col("arm").is_in([CURVE_LABELS[a] for a in quantile_arms])))  # ty: ignore[unresolved-attribute]
            .transform_fold(["q05", "q95"], as_=["level", "tail"])
            .mark_line(strokeDash=[3, 3], strokeWidth=1)
            .encode(x=alt.X("speed:Q", scale=scale_x), y="tail:Q", color=color, detail="level:N")
        )
        rules = [
            alt.Chart(pl.DataFrame({"x": [mark["cut_in"]]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color=GREY, strokeDash=[2, 2])
            .encode(x="x:Q"),
        ]
        if mark["rated"] <= x_max:
            rules.append(
                alt.Chart(pl.DataFrame({"x": [mark["rated"]]}))  # ty: ignore[unresolved-attribute]
                .mark_rule(color=GREY, strokeDash=[6, 3])
                .encode(x="x:Q")
            )
        if mark["idle"]:
            rules.append(
                alt.Chart(pl.DataFrame({"y": [mark["idle"][0]], "y2": [mark["idle"][1]]}))  # ty: ignore[unresolved-attribute]
                .mark_rect(color=ocf.DATA_SKY, opacity=0.25)
                .encode(y="y:Q", y2="y2:Q")
            )
        charts.append(
            alt.layer(band, median, tails, lines, *rules).properties(
                width=WIDTH / 2 - 60, height=150, title=panel
            )
        )
    del rows
    grid = alt.vconcat(alt.hconcat(charts[0], charts[1]), alt.hconcat(charts[2], charts[3]))
    return grid.properties(title=title).resolve_scale(color="shared")


def figure_curves(
    *, observed: pl.DataFrame, fc: pl.DataFrame, marks: dict, zoom: bool
) -> alt.TopLevelMixin:
    """The observed power curve and each arm's forecast median (and 5% and 95% quantiles)."""
    return _curve_panels(
        observed=observed,
        fc=fc,
        marks=marks,
        x_max=8.0 if zoom else 25.0,
        quantile_arms=("default", "X2"),
        title=_title(
            text="Low winds: the forecast sits above the observed median"
            if zoom
            else "Power against 100 m wind speed",
            lines=[
                (
                    "Black: observed median and 5% to 95% band in bins of ERA5 speed at the valid "
                    "time. Coloured: each arm's median forecast in bins of the ensemble-mean speed "
                    "at forecast time; dashed: the default's and X2's 5% and 95% quantiles. Grey "
                    "dotted: cut-in; grey dashed: rated speed; blue band: experiment 210 idle "
                    "band (5% to 95% of power below 2 m/s). Scored year, all half-hours, 1 to 10 "
                    "days. W1 to W3 are anonymised farms; the last panel pools them (cut-in and "
                    "rated speed are the farms' means)."
                )
            ],
        ),
    )


def figure_difference(*, diff: pl.DataFrame, marks: dict) -> alt.TopLevelMixin:
    """Forecast median minus observed median per bin of the ensemble-mean speed."""
    charts = []
    domain = list(CURVE_LABELS.values())
    colours = list(CURVE_ARMS.values())
    for panel in PANELS:
        data = diff.filter((pl.col("panel") == panel) & (pl.col("speed") <= 20.0))
        charts.append(
            alt.layer(
                alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=GREY).encode(y="y:Q"),  # ty: ignore[unresolved-attribute]
                alt.Chart(pl.DataFrame({"x": [marks[panel]["cut_in"]]}))  # ty: ignore[unresolved-attribute]
                .mark_rule(color=GREY, strokeDash=[2, 2])
                .encode(x="x:Q"),
                alt.Chart(data)  # ty: ignore[unresolved-attribute]
                .mark_line(strokeWidth=1.6)
                .encode(
                    x=alt.X("speed:Q", title="Ensemble-mean 100 m wind speed (m/s)"),
                    y=alt.Y("difference:Q", title="Forecast minus observed median (points)"),
                    color=alt.Color(
                        "arm:N",
                        scale=alt.Scale(domain=domain, range=colours),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                ),
            ).properties(width=WIDTH / 2 - 60, height=150, title=panel)
        )
    return alt.vconcat(
        alt.hconcat(charts[0], charts[1]), alt.hconcat(charts[2], charts[3])
    ).properties(
        title=_title(
            text="Forecast median minus observed median, by wind speed",
            lines=[
                (
                    "Both medians come from the same half-hours, binned by the ensemble-mean "
                    "speed at forecast time (points of capacity; zero is unbiased). Grey dotted: "
                    "cut-in speed. Scored year, all half-hours, 1 to 10 days."
                )
            ],
        )
    )


def main() -> None:
    """Write the figures."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_1(score=score).save(DOCS_DIR / "figure_1.svg")
    observed, fc, diff, marks = curve_frames()
    figure_curves(observed=observed, fc=fc, marks=marks, zoom=False).save(DOCS_DIR / "figure_2.svg")
    figure_curves(observed=observed, fc=fc, marks=marks, zoom=True).save(DOCS_DIR / "figure_3.svg")
    figure_difference(diff=diff, marks=marks).save(DOCS_DIR / "figure_4.svg")


if __name__ == "__main__":
    main()
