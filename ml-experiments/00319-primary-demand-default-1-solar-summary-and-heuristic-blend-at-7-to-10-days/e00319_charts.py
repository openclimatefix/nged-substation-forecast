# ruff: noqa: E501
"""Draw the figures of experiment 319 from `work/report/report.json` and `work/arms/fit_log.json`.

uv run --no-sync python ml-experiments/00319-*/e00319_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00319_common as common
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
BANDS: Final[tuple[str, ...]] = tuple(f"{a}-{a + 24} h" for a in range(0, 240, 24))
CENTRES: Final[dict[str, int]] = {b: i * 24 + 12 for i, b in enumerate(BANDS)}
NAMES: Final[dict[str, str]] = {
    "default": "Default (Primary demand research default 1)",
    "x1": "Default plus x1 (solar summary to 240 h)",
    "x2": "Default plus x2 (heuristic blend)",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
COLOURS: Final[dict[str, str]] = {
    NAMES["default"]: ocf.DATA_BLUE,
    NAMES["x1"]: ocf.DATA_GREEN,
    NAMES["x2"]: ocf.DATA_PURPLE,
    NAMES["manual_heuristic"]: ocf.BRAND_ORANGE,
    NAMES["climatology"]: "#7A7A7A",
}


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def _title(*, title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(title, 85),
        subtitle=textwrap.wrap(subtitle, 110),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: Any, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _colour() -> Any:
    return alt.Color(
        "arm:N",
        scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=420, symbolType="stroke"),
    )


def headline(*, report: dict) -> None:
    """Quantile score against lead for every forecast, with 95% intervals."""
    rows = [
        {
            "arm": NAMES[key],
            "lead": CENTRES[b],
            **{k: report["by_band"][b][key][k] for k in ("est", "lo", "hi")},
        }
        for b in BANDS
        for key in NAMES
    ]
    data = pl.DataFrame(rows)
    x = alt.X(
        "lead:Q",
        title="Lead time (hours; each point is a 24-hour band, plotted at its centre)",
        scale=alt.Scale(domain=[0, 240]),
        axis=alt.Axis(values=[0, 48, 96, 144, 192, 240]),
    )
    yt = "Mean quantile score (points of capacity; smaller is better)"
    base = _chart(data)
    chart = (
        base.mark_errorbar(thickness=1.5).encode(
            x=x, y=alt.Y("lo:Q", title=yt, scale=alt.Scale(zero=False)), y2="hi:Q", color=_colour()
        )
        + base.mark_line(strokeWidth=2).encode(x=x, y=alt.Y("est:Q", title=yt), color=_colour())
        + base.mark_point(filled=True, size=36).encode(
            x=x, y=alt.Y("est:Q", title=yt), color=_colour()
        )
    ).properties(
        width=WIDTH,
        height=300,
        title=_title(
            title="Both changes lower the quantile score from 168 to 240 hours, and the blend also beats the manual heuristic",
            subtitle="Dot: mean over six Primary substations, scored year 2025-07-02 to 2026-06-30. Bar: 95% interval from resampling the 12 months of forecast date. Below 168 hours both arms equal the default.",
        ),
    )
    _save(chart, name="headline_score_by_lead")


def difference(*, report: dict) -> None:
    """Each arm's score minus the default's, with 95% intervals, from 168 to 240 hours."""
    rows = []
    for b in BANDS[7:]:
        for key, label in (
            ("x1", NAMES["x1"]),
            ("x2", NAMES["x2"]),
            ("xr", "Placebo: x1 with every member response set to zero"),
        ):
            c = report["by_band"][b][f"{key}_vs_default"]
            rows.append(
                {
                    "arm": label,
                    "lead": CENTRES[b] + {"x1": -6, "x2": 0, "xr": 6}[key],
                    "diff": c["diff"],
                    "lo95": c["diff_lo_95"],
                    "hi95": c["diff_hi_95"],
                    "lo995": c["diff_lo_995"],
                    "hi995": c["diff_hi_995"],
                }
            )
    data = pl.DataFrame(rows)
    domain = [NAMES["x1"], NAMES["x2"], "Placebo: x1 with every member response set to zero"]
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=domain, range=[ocf.DATA_GREEN, ocf.DATA_PURPLE, "#7A7A7A"]),
        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=420, symbolType="stroke"),
    )
    x = alt.X(
        "lead:Q",
        title="Lead time (hours; 24-hour bands plotted at their centres, arms offset by 6 hours)",
        scale=alt.Scale(domain=[162, 246]),
        axis=alt.Axis(values=[180, 204, 228]),
    )
    yt = "Score minus the default's (points of capacity; more negative is better)"
    base = _chart(data)
    zero = (
        _chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(color="#444444", strokeDash=[4, 3])
        .encode(y="y:Q")
    )
    chart = (
        zero
        + base.mark_errorbar(thickness=1.5).encode(
            x=x, y=alt.Y("lo95:Q", title=yt), y2="hi95:Q", color=colour
        )
        + base.mark_point(filled=True, size=50).encode(
            x=x, y=alt.Y("diff:Q", title=yt), color=colour
        )
    ).properties(
        width=WIDTH,
        height=260,
        title=_title(
            title="The blend lowers the score by about 0.14 points and the solar summary by about 0.06, in every band from 168 to 240 hours",
            subtitle="Dot: difference in mean quantile score, six Primaries, scored year. Bar: 95% interval from resampling the 12 months. Dashed line: same as the default. The placebo refits the offsets but gives no member information.",
        ),
    )
    _save(chart, name="difference_from_default_168_240")


def by_group(*, report: dict) -> None:
    """Gain over the default by protected group at 168 to 240 hours."""
    rows = []
    for group in (
        "all rows",
        "sunny daylight",
        "cold winter nights",
        "winter (Nov-Mar)",
        "April to October",
    ):
        for key in ("x1", "x2"):
            c = report["by_group"][group]["168-240 h"][f"{key}_vs_default"]
            rows.append(
                {
                    "group": group,
                    "arm": NAMES[key],
                    "gain": 100 * c["gain"],
                    "lo": 100 * c["gain_lo_95"],
                    "hi": 100 * c["gain_hi_95"],
                }
            )
    data = pl.DataFrame(rows)
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=[NAMES["x1"], NAMES["x2"]], range=[ocf.DATA_GREEN, ocf.DATA_PURPLE]),
        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=420, symbolType="stroke"),
    )
    y = alt.Y(
        "group:N",
        title=None,
        sort=[
            "all rows",
            "sunny daylight",
            "cold winter nights",
            "winter (Nov-Mar)",
            "April to October",
        ],
    )
    xt = "Lower quantile score than the default (%; larger is better)"
    x = alt.X("gain:Q", title=xt)
    base = _chart(data)
    zero = (
        _chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#444444", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (
        zero
        + base.mark_errorbar(thickness=1.5).encode(
            y=y, yOffset=alt.YOffset("arm:N"), x=alt.X("lo:Q", title=xt), x2="hi:Q", color=colour
        )
        + base.mark_point(filled=True, size=50).encode(
            y=y, yOffset=alt.YOffset("arm:N"), x=x, color=colour
        )
    ).properties(
        width=WIDTH,
        height=260,
        title=_title(
            title="The blend raises the score of cold winter nights; the solar summary leaves them unchanged and helps sunny daylight most",
            subtitle="Leads 168 to 240 hours, six Primaries, scored year. Dot: gain over the default. Bar: 95% interval from resampling the 12 months. Dashed line: same as the default.",
        ),
    )
    _save(chart, name="gain_by_group_168_240")


def weights(*, log: dict) -> None:
    """The chain's blend weight against lead band and season, over the monthly refits."""
    rows = []
    for entry in log.values():
        for band, label in enumerate(("168-192 h", "192-216 h", "216-240 h")):
            for season, name in enumerate(("November to March", "April to October")):
                rows.append({"band": label, "season": name, "w": entry["weights"][band][season]})
    data = (
        pl.DataFrame(rows)
        .group_by(["band", "season"])
        .agg(
            pl.col("w").mean().alias("mean"),
            pl.col("w").min().alias("lo"),
            pl.col("w").max().alias("hi"),
        )
    )
    colour = alt.Color(
        "season:N",
        scale=alt.Scale(
            domain=["November to March", "April to October"],
            range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
        ),
        legend=alt.Legend(title=None, orient="top", symbolType="stroke"),
    )
    x = alt.X("band:N", title="Lead band", sort=["168-192 h", "192-216 h", "216-240 h"])
    yt = "Weight on the convex chain's median (1 means the chain alone)"
    base = _chart(data)
    chart = (
        base.mark_errorbar(thickness=1.5).encode(
            x=x,
            xOffset="season:N",
            y=alt.Y("lo:Q", title=yt, scale=alt.Scale(domain=[0.4, 1.0])),
            y2="hi:Q",
            color=colour,
        )
        + base.mark_point(filled=True, size=60).encode(
            x=x, xOffset="season:N", y=alt.Y("mean:Q", title=yt), color=colour
        )
    ).properties(
        width=WIDTH,
        height=240,
        title=_title(
            title="The fitted blend puts 55% to 90% of its weight on the convex chain, and more in November to March",
            subtitle="Dot: mean of the 12 monthly refits of the scored year. Bar: lowest to highest refit. The rest of the weight goes to the manual heuristic's median.",
        ),
    )
    _save(chart, name="blend_weight_by_lead_and_season")


def tails(*, report: dict) -> None:
    """Coverage of the 90% band and the share of readings below q01 and above q99, by lead."""
    rows = []
    for b in BANDS:
        for key in ("default", "x1", "x2", "manual_heuristic", "climatology"):
            t = report["tails_by_band"][b][key]
            rows.append(
                {
                    "arm": NAMES[key],
                    "lead": CENTRES[b],
                    "metric": "Readings inside the 5% to 95% band (nominal 90%)",
                    "value": 100 * t["in90"],
                }
            )
            rows.append(
                {
                    "arm": NAMES[key],
                    "lead": CENTRES[b],
                    "metric": "Readings below q01 (nominal 1%)",
                    "value": 100 * t["q01_below"],
                }
            )
            rows.append(
                {
                    "arm": NAMES[key],
                    "lead": CENTRES[b],
                    "metric": "Readings above q99 (nominal 1%)",
                    "value": 100 * (1 - t["q99_below"]),
                }
            )
    data = pl.DataFrame(rows)
    x = alt.X(
        "lead:Q",
        title="Lead time (hours; 24-hour bands plotted at their centres)",
        scale=alt.Scale(domain=[0, 240]),
        axis=alt.Axis(values=[0, 48, 96, 144, 192, 240]),
    )
    chart = (
        _chart(data)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y("value:Q", title="Share of readings (%)", scale=alt.Scale(zero=False)),
            color=_colour(),
        )
        .properties(width=WIDTH, height=150)
        .facet(
            row=alt.Row(
                "metric:N",
                title=None,
                sort=[
                    "Readings inside the 5% to 95% band (nominal 90%)",
                    "Readings below q01 (nominal 1%)",
                    "Readings above q99 (nominal 1%)",
                ],
            )
        )
        .resolve_scale(y="independent")
        .properties(
            title=_title(
                title="The blend moves the tails and the coverage by under a percentage point; both stay below nominal",
                subtitle="Six Primaries, scored year, by 24-hour lead band.",
            )
        )
    )
    _save(chart, name="tail_calibration_by_lead")


def main() -> None:
    """Draw every figure."""
    report = json.loads((common.REPORT_DIR / "report.json").read_text())
    log = json.loads((common.ARMS_DIR / "fit_log.json").read_text())
    headline(report=report)
    difference(report=report)
    by_group(report=report)
    weights(log=log)
    tails(report=report)


if __name__ == "__main__":
    main()
