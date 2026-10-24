"""Draw experiment 00342's figures as SVG. Reads `work/score.json` and `work/final/`.

Figure 1: each arm's targeted score (mean of the 0.95 and 0.99 pinball losses) minus the default's.
Figure 2: the share of observed power above the forecast q95 and q99, per arm and farm.
Figures 3 to 5: the power curves (full range, 0 to 8 m/s) of the forecast q95 and q99 against the
observed power, and the forecast q99 minus the observed q99 by wind speed.
Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00342-*/e00342_charts.py   (PYTHONPATH: see common)
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt

import e00326_score as score326  # isort: skip  # ty: ignore[unresolved-import]
import e00342_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

c303 = common.c303
c336 = common.c336
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00342-wind-research-default-1-upper-tail-offsets-and-head"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
GREY: Final[str] = "#777777"
LEVEL: Final[str] = "0.9967"
SCOPE: Final[str] = "3 wind farms, 3 seeds, scored at 1 to 10 days."
ARM_LABELS: Final[dict[str, str]] = {
    "x1": "x1: offsets keyed by wind speed",
    "x2": "x2: tail head",
    "x3": "x3: mass at the farm scale",
    "x1lit": "x1lit: offsets per farm and lead day (literal brief)",
    "x1cap": "x1cap: cap only",
    "x2base": "x2base: head without member summaries",
    "x2plc": "x2plc: head with shuffled member summaries",
    "x2lofo": "x2lofo: head, leave one farm out",
    "ref_drift": "reference: experiment 340 drift shift",
    "ref_mixture": "reference: experiment 339 mixture head",
}
ADOPTABLE: Final[tuple[str, ...]] = ("x1", "x2", "x3")
PLACEBOS: Final[tuple[str, ...]] = ("x2plc",)
CURVE_ARMS: Final[dict[str, str]] = {
    "default": GREY,
    "x1": ocf.DATA_BLUE,
    "x2": ocf.BRAND_ORANGE,
    "x3": ocf.DATA_GREEN,
}
CURVE_LABELS: Final[dict[str, str]] = {"default": "default", "x1": "x1", "x2": "x2", "x3": "x3"}
PANELS: Final[tuple[str, ...]] = ("W1", "W2", "W3", "All three farms")
BIN: Final[float] = 0.5
LEVEL_INDEX: Final[dict[str, int]] = {"q95": 9, "q99": 10}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """Each arm minus the default: targeted metric, headline score, and 0.99 pinball loss."""
    panels = []
    sets = (
        ("tgt", "Mean of the 0.95 and 0.99 pinball losses (the targeted metric)"),
        ("qs", "Quantile score over the 11 levels (the headline)"),
        ("p99", "0.99 pinball loss"),
    )
    for metric, text in sets:
        rows = []
        for key, label in ARM_LABELS.items():
            found = score["contrasts"]["scored"][key][f"kept | {metric}"]
            low, high = found[LEVEL]
            rows.append(
                {
                    "arm": label,
                    "estimate": found["mean"],
                    "lower": low,
                    "upper": high,
                    "kind": "placebo"
                    if key in PLACEBOS
                    else ("adoptable" if key in ADOPTABLE else "descriptive"),
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
                height=22 * data.height,
                title=_title(
                    text=text,
                    lines=[
                        "Scored year, kept rows, score minus Wind farms research default 1's. "
                        "Dot: estimate. Line: 99.67% interval from re-drawing whole months and "
                        "seeds. " + SCOPE
                    ],
                ),
            )
        )
    return alt.vconcat(*panels).resolve_scale(x="independent")


def figure_shares(*, score: dict) -> alt.TopLevelMixin:
    """Share of kept half-hours with observed power above q95 and above q99, per arm and farm."""
    rows = []
    arms = {"default": "default", **{k: ARM_LABELS[k].split(":")[0] for k in ADOPTABLE}}
    arms["ref_mixture"] = "ref 339"
    arms["ref_drift"] = "ref 340"
    for arm, label in arms.items():
        pooled = score["scores"]["scored"][arm]["kept"]
        for metric, nominal in (("above_q95", 5.0), ("above_q99", 1.0)):
            rows.append(
                {
                    "arm": label,
                    "panel": "All three farms",
                    "level": f"above {metric[-3:]} (nominal {nominal:.0f}%)",
                    "share": pooled[metric] * 100,
                }
            )
            by_farm = score["scores"]["scored"][arm]["by_farm_kept"]
            rows.extend(
                {
                    "arm": label,
                    "panel": farm,
                    "level": f"above {metric[-3:]} (nominal {nominal:.0f}%)",
                    "share": by_farm[farm][metric] * 100,
                }
                for farm in common.FARMS
            )
    data = pl.DataFrame(rows)
    halves = []
    for level, nominal in (("above q95 (nominal 5%)", 5.0), ("above q99 (nominal 1%)", 1.0)):
        part = data.filter(pl.col("level") == level)
        points = (
            alt.Chart(part)  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=50)
            .encode(
                x=alt.X("arm:N", sort=list(arms.values()), title=None),
                y=alt.Y("share:Q", title="Share of half-hours (%)"),
                xOffset="panel:N",
                color=alt.Color(
                    "panel:N",
                    scale=alt.Scale(
                        domain=list(PANELS),
                        range=[ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.DATA_PURPLE, GREY],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
        )
        rule = (
            alt.Chart(pl.DataFrame({"share": [nominal]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black", strokeDash=[4, 3])
            .encode(y="share:Q")
        )
        halves.append(alt.layer(rule, points).properties(width=WIDTH / 2 - 40, title=level))
    return alt.hconcat(*halves).properties(
        title=_title(
            text="Observed power above the forecast's upper quantiles",
            lines=[
                "Share of kept half-hours in the scored year with observed power above the "
                "forecast q95 (left) and q99 (right); the dashed line is the nominal share. "
                + SCOPE
            ],
        )
    )


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
    """Return observed, forecast, and difference bins and per-farm marks (scored year).

    The observed curve has the median and 5% to 95% band in bins of ERA5 speed at the valid time,
    and the observed 95% and 99% quantiles in bins of the ensemble-mean speed at forecast time
    (the same bins as the forecast quantiles, so the two are comparable).
    """
    sites = c303.site_table()
    cases = c303.cases_table(sites=sites)
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
        obs_tail = _binned(
            speed=speed[pick], value=actual[pick], edges=edges, quantiles=(0.95, 0.99)
        )
        for b, x in enumerate(centres):
            if np.isfinite(obs[b, 0]):
                observed_rows.append(
                    {
                        "panel": panel,
                        "speed": x,
                        "band_low": obs[b, 0],
                        "median": obs[b, 1],
                        "band_high": obs[b, 2],
                        "obs_q95": obs_tail[b, 0],
                        "obs_q99": obs_tail[b, 1],
                    }
                )
        for arm, q in forecast.items():
            q95 = _binned(speed=speed[pick], value=q[pick, 9], edges=edges, quantiles=(0.5,))
            q99 = _binned(speed=speed[pick], value=q[pick, 10], edges=edges, quantiles=(0.5,))
            for b, x in enumerate(centres):
                if np.isfinite(q99[b, 0]):
                    forecast_rows.append(
                        {
                            "panel": panel,
                            "arm": CURVE_LABELS[arm],
                            "speed": x,
                            "q95": q95[b, 0],
                            "q99": q99[b, 0],
                        }
                    )
                    diff_rows.append(
                        {
                            "panel": panel,
                            "arm": CURVE_LABELS[arm],
                            "speed": x,
                            "q95": q95[b, 0] - obs_tail[b, 0],
                            "q99": q99[b, 0] - obs_tail[b, 1],
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


def _rules(*, mark: dict, x_max: float) -> list:
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
    return rules


def figure_curves(
    *, observed: pl.DataFrame, fc: pl.DataFrame, marks: dict, zoom: bool
) -> alt.TopLevelMixin:
    """The observed power curve and each arm's forecast q95 and q99 against wind speed."""
    x_max = 8.0 if zoom else 25.0
    domain = list(CURVE_LABELS.values())
    colours = list(CURVE_ARMS.values())
    charts = []
    for panel in PANELS:
        scale_x = alt.Scale(domain=[0, x_max])
        obs = observed.filter((pl.col("panel") == panel) & (pl.col("speed") <= x_max))
        band = (
            alt.Chart(obs)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.2, color="black")
            .encode(
                x=alt.X("speed:Q", scale=scale_x, title="100 m wind speed (m/s)"),
                y=alt.Y("band_low:Q", title="Power (% of capacity)"),
                y2="band_high:Q",
            )
        )
        median = (
            alt.Chart(obs)  # ty: ignore[unresolved-attribute]
            .mark_line(color="black", strokeWidth=1.5)
            .encode(x=alt.X("speed:Q", scale=scale_x), y="median:Q")
        )
        observed_tail = (
            alt.Chart(obs)  # ty: ignore[unresolved-attribute]
            .transform_fold(["obs_q95", "obs_q99"], as_=["level", "tail"])
            .mark_line(color="black", strokeWidth=1.8)
            .encode(
                x=alt.X("speed:Q", scale=scale_x),
                y="tail:Q",
                strokeDash=alt.StrokeDash(
                    "level:N",
                    scale=alt.Scale(domain=["obs_q95", "obs_q99"], range=[[1, 0], [4, 3]]),
                    legend=None,
                ),
            )
        )
        data = fc.filter((pl.col("panel") == panel) & (pl.col("speed") <= x_max))
        lines = (
            alt.Chart(data)  # ty: ignore[unresolved-attribute]
            .transform_fold(["q95", "q99"], as_=["level", "tail"])
            .mark_line(strokeWidth=1.4)
            .encode(
                x=alt.X("speed:Q", scale=scale_x),
                y="tail:Q",
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=domain, range=colours),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                strokeDash=alt.StrokeDash(
                    "level:N",
                    scale=alt.Scale(domain=["q95", "q99"], range=[[1, 0], [4, 3]]),
                    legend=None,
                ),
            )
        )
        charts.append(
            alt.layer(
                band, median, observed_tail, lines, *_rules(mark=marks[panel], x_max=x_max)
            ).properties(width=WIDTH / 2 - 60, height=150, title=panel)
        )
    grid = alt.vconcat(alt.hconcat(charts[0], charts[1]), alt.hconcat(charts[2], charts[3]))
    return grid.properties(
        title=_title(
            text="Low winds: the upper quantiles against the observed power"
            if zoom
            else "Forecast q95 and q99 against the observed power curve",
            lines=[
                (
                    "Black band and thin line: observed median and 5% to 95% band in bins of ERA5 "
                    "speed at the valid time. Thick black: observed 95% (solid) and 99% (dashed) "
                    "quantiles of power in bins of the ensemble-mean speed at forecast time. "
                    "Coloured: each arm's mean forecast q95 (solid) and q99 (dashed) in the same "
                    "forecast-speed bins. Grey dotted: cut-in; grey dashed: rated; blue band: "
                    "experiment 210 idle band. Scored year, all half-hours, 1 to 10 days. W1 to "
                    "W3 are anonymised farms; the last panel pools them."
                )
            ],
        )
    ).resolve_scale(color="shared")


def figure_difference(*, diff: pl.DataFrame, marks: dict) -> alt.TopLevelMixin:
    """Forecast q99 minus the observed q99 per bin of the ensemble-mean speed."""
    domain = list(CURVE_LABELS.values())
    colours = list(CURVE_ARMS.values())
    charts = []
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
                    y=alt.Y("q99:Q", title="Forecast q99 minus observed q99 (points)"),
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
            text="Forecast q99 minus the observed q99, by wind speed",
            lines=[
                (
                    "Both come from the same half-hours, binned by the ensemble-mean speed at "
                    "forecast time (points of capacity; zero is calibrated on average). Grey "
                    "dotted: cut-in speed. Scored year, all half-hours, 1 to 10 days."
                )
            ],
        )
    )


def main() -> None:
    """Write the figures."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_1(score=score).save(DOCS_DIR / "figure_1.svg")
    figure_shares(score=score).save(DOCS_DIR / "figure_2.svg")
    observed, fc, diff, marks = curve_frames()
    figure_curves(observed=observed, fc=fc, marks=marks, zoom=False).save(DOCS_DIR / "figure_3.svg")
    figure_curves(observed=observed, fc=fc, marks=marks, zoom=True).save(DOCS_DIR / "figure_4.svg")
    figure_difference(diff=diff, marks=marks).save(DOCS_DIR / "figure_5.svg")


if __name__ == "__main__":
    main()
