# ruff: noqa: E501
"""Draw the figures of experiment 354 from `work/bands_scores.json` and `work/fits_scores.json`.

uv run --no-sync python ml-experiments/00354-*/e00354_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00354-d8-own-residual-bands-and-longer-fit-history"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
PNG_DIR: Final[Path] = WORK / "png"
WIDTH: Final[int] = 480
GREY: Final[str] = "#8C8C8C"


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _title(text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 85), subtitle=textwrap.wrap(subtitle, 110),
        anchor="start", fontSize=14, subtitleFontSize=11,
    )  # fmt: skip


def forest(
    *,
    rows: list[dict],
    x_title: str,
    title: str,
    subtitle: str,
    name: str,
    colours: dict[str, str] | None = None,
    height: int = 220,
) -> None:
    """Draw estimates with intervals, one row per label; `group` colours the marks."""
    frame = pl.DataFrame(rows)
    order = frame["label"].to_list()
    groups = list(dict.fromkeys(frame["group"].to_list()))
    palette = colours or dict(
        zip(
            groups,
            [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.DATA_PURPLE, GREY],
            strict=False,
        )
    )
    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=400)),
        color=alt.Color("group:N", scale=alt.Scale(domain=groups, range=[palette[g] for g in groups]), legend=alt.Legend(title=None, orient="top")),
    )  # fmt: skip
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        zero
        + base.mark_rule(strokeWidth=2).encode(x=alt.X("low:Q", title=x_title), x2="high:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=45).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=height, title=_title(title, subtitle))
    _save(chart, name=name)


def figure_1(*, bands: dict, fits: dict) -> None:
    """Gap to BG2 on all rows, for the stored-band arms and the refitted arms."""
    d8 = bands["D8"]["contrast"]
    spec = [
        ("Arm 1: stack, donated spread (the card)", d8["g_arm1"], "Donated spread"),
        ("Arm 2: own as-of residual quantiles", d8["g_arm2"], "Own-residual band"),
        ("Arm 3: the same by hour block and season", d8["g_arm3"], "Own-residual band"),
        ("Arm 4: arm 2 shrunk towards BG2's residuals", d8["g_arm4"], "Own-residual band"),
        ("No stage-2 steps + arm 2 band", d8["g_a2"], "Own-residual band"),
        (
            "(a) fit from 2025-04, arm 3 band",
            fits["contrast"]["g_a_d_arm3"],
            "Fit history (arm 3 band)",
        ),
        (
            "(b) fit from 2023-09 with level step",
            fits["contrast"]["g_b_d_arm3"],
            "Fit history (arm 3 band)",
        ),
        (
            "(c) fit from 2023-09, no step",
            fits["contrast"]["g_c_d_arm3"],
            "Fit history (arm 3 band)",
        ),
    ]
    rows = [
        {
            "label": label,
            "estimate": v["all"]["ci99"][0],
            "low": v["all"]["ci99"][1],
            "high": v["all"]["ci99"][2],
            "group": g,
        }
        for label, v, g in spec
    ]
    forest(
        rows=rows,
        x_title="Quantile score minus BG2's (points of capacity; smaller is better)",
        title="Figure 1: D8's own-residual band closes about 60% of the gap to BG2",
        subtitle="Dot: estimate over the 96,986 scored rows. Line: 99% interval from resampling whole months. Zero is BG2.",
        name="figure_1_gap_to_bg2", height=260,
    )  # fmt: skip


def figure_2(*, bands: dict, fits: dict) -> None:
    """Monthly quantile score of the main arms."""
    d8 = bands["D8"]["monthly"]
    fm = fits["monthly"]
    series = {
        "Arm 1: donated spread (the card)": ("arm1", d8, ocf.DATA_BLUE),
        "Arm 3: own-residual band": ("arm3", d8, ocf.BRAND_ORANGE),
        "(b) fit from 2023-09 + arm 3 band": ("b_d_arm3", fm, ocf.DATA_PURPLE),
        "BG2": ("bg2", d8, ocf.DATA_GREEN),
    }
    rows = [
        {"month": m, "arm": label, "score": values[m][key]}
        for label, (key, values, _) in series.items()
        for m in sorted(d8)
    ]
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("month:N", title="Month of the valid time", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("score:Q", title="Quantile score (points of capacity; smaller is better)", scale=alt.Scale(zero=False)),
            color=alt.Color("arm:N", scale=alt.Scale(domain=list(series), range=[v[2] for v in series.values()]), legend=alt.Legend(title=None, orient="top", columns=2)),
        )
        .properties(width=WIDTH, height=280, title=_title("Figure 2: the gain is largest in August and September 2025, when the fits held the least history", "Mean quantile score of each month's scored rows at D8."))
    )  # fmt: skip
    _save(chart, name="figure_2_monthly")


def figure_3(*, bands: dict) -> None:
    """The own-residual band's cost at the placebo Primaries, against D8."""
    rows = []
    for site, label in (("D2", "D2 (placebo)"), ("D3", "D3 (placebo)"), ("D4", "D4 (placebo)")):
        for column, arm in (
            ("c_arm2_arm1", "Arm 2 minus arm 1"),
            ("c_arm3_arm1", "Arm 3 minus arm 1"),
        ):
            point, low, high = bands[site]["contrast"][column]["all"]["ci99"]
            relative = 100.0 / bands[site]["absolute"]["all"]["arm1"]
            rows.append(
                {
                    "label": f"{label}: {arm}",
                    "estimate": point * relative,
                    "low": low * relative,
                    "high": high * relative,
                    "group": arm,
                }
            )
    forest(
        rows=rows,
        x_title="Change in quantile score (% of the donated-spread score; negative is better)",
        title="Figure 3: at Primaries with no engine the own-residual band costs 0 to 7%",
        subtitle="Dot: estimate over each site's scored rows. Line: 99% interval from resampling whole months. The pre-registered limit is +3%.",
        name="figure_3_placebos", height=240,
        colours={"Arm 2 minus arm 1": ocf.BRAND_ORANGE, "Arm 3 minus arm 1": ocf.DATA_BLUE},
    )  # fmt: skip


def figure_4(*, bands: dict) -> None:
    """Share of readings inside q05 to q95, by quarter."""
    d8 = bands["D8"]["coverage_by_quarter"]
    series = {
        "Arm 1: donated spread": ("arm1", ocf.DATA_BLUE),
        "Arm 2: own-residual quantiles": ("arm2", ocf.BRAND_ORANGE),
        "Arm 3: by hour block and season": ("arm3", ocf.DATA_PURPLE),
    }
    rows = [
        {"quarter": q, "arm": label, "share": d8[key][q]["inside_q05_q95"]}
        for label, (key, _) in series.items()
        for q in ("Q1", "Q2", "Q3", "Q4")
    ]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        x=alt.X(
            "quarter:N",
            title="Quarter of the scored year (Q1 is July to September 2025)",
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y(
            "share:Q", title="Share of readings inside q05 to q95", scale=alt.Scale(domain=[0, 1])
        ),
    )
    band = (
        alt.Chart(pl.DataFrame({"low": [0.85], "high": [0.95]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(opacity=0.12, color=ocf.DATA_GREEN)
        .encode(y="low:Q", y2="high:Q")
    )
    nominal = (
        alt.Chart(pl.DataFrame({"y": [0.9]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#333333")
        .encode(y="y:Q")
    )
    lines = base.mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(series), range=[v[1] for v in series.values()]),
            legend=alt.Legend(title=None, orient="top"),
        )
    )
    chart = (band + nominal + lines).properties(
        width=WIDTH,
        height=260,
        title=_title(
            "Figure 4: the own-residual band reaches 0.90 coverage after the first quarter",
            "Dashed line: the nominal 0.90. Green band: the pre-registered range 0.85 to 0.95.",
        ),
    )
    _save(chart, name="figure_4_coverage")


def figure_5(*, fits: dict) -> None:
    """Contrasts between the fit histories."""
    spec = [
        ("(b) minus (a): longer history with step", "c_b_d_arm3__a_d_arm3"),
        ("(c) minus (a): longer history, no step", "c_c_d_arm3__a_d_arm3"),
        ("(b) minus (c): the step alone", "c_b_d_arm3__c_d_arm3"),
    ]
    rows = []
    for label, key in spec:
        for flag, flag_label in (
            ("all", "All rows"),
            ("first3", "July to September 2025"),
            ("y2026", "2026"),
        ):
            point, low, high = fits["contrast"][key][flag]["ci99"]
            rows.append(
                {
                    "label": f"{label}: {flag_label}",
                    "estimate": point,
                    "low": low,
                    "high": high,
                    "group": flag_label,
                }
            )
    forest(
        rows=rows,
        x_title="Quantile score difference (points of capacity; negative favours the first arm)",
        title="Figure 5: a longer history helps in the first months only, and the level step adds nothing measurable",
        subtitle="Both arms use arm 3's band and stage 2. Dot: estimate. Line: 99% interval from resampling whole months.",
        name="figure_5_fit_history", height=340,
        colours={"All rows": ocf.BRAND_ORANGE, "July to September 2025": ocf.DATA_BLUE, "2026": ocf.DATA_GREEN},
    )  # fmt: skip


def main() -> None:
    """Draw every figure."""
    bands = json.loads((WORK / "bands_scores.json").read_text())
    fits = json.loads((WORK / "fits_scores.json").read_text())
    figure_1(bands=bands, fits=fits)
    figure_2(bands=bands, fits=fits)
    figure_3(bands=bands)
    figure_4(bands=bands)
    figure_5(fits=fits)


if __name__ == "__main__":
    main()
