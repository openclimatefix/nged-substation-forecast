"""Draw the figures of experiment 125's write-up.

The inputs are `work/results.json` and `work/accounting.json`.

Every figure carries its scope, unit, and direction in its title and subtitle. SVGs go to the
write-up's folder, PNGs to a scratch folder for the render check.

    uv run python ml-experiments/00125-*/e00125_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final, cast

import altair as alt
import e00125_common as common
import e00125_results as results
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.ens_members import MIXTURE_LEVELS

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 600
WRAP: Final[int] = 100
GREY: Final[str] = "#808080"
BANDS: Final[tuple[str, ...]] = ("0-24 h", "24-48 h", "48-144 h", "144-168 h")
ARM_COLOURS: Final[dict[str, str]] = {
    "K0": ocf.DATA_BLUE_LIGHT,
    "K0-w": ocf.DATA_BLUE,
    "MP": ocf.BRAND_ORANGE,
    "MP-1": ocf.DATA_PURPLE,
    "MO": ocf.DATA_GREEN,
    "MM": ocf.DATA_SKY,
}
ARM_NAMES: Final[dict[str, str]] = {
    "K0": "K0 (today's table, stored factors)",
    "K0-w": "K0-w (today's table, factors re-tuned)",
    "MP": "MP (analysis-proxy table, tuned factors)",
    "MP-1": "MP-1 (analysis-proxy table, factor 1)",
    "MO": "MO (members only, no table)",
    "MM": "MM (ensemble mean, K0 table)",
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
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def figure_headline(*, data: dict) -> alt.TopLevelMixin:
    """Tail pinball differences of the planned and secondary contrasts, both periods."""
    labels = {
        "C1 MP-K0-w": "C1: MP minus K0-w",
        "C1b MP-K0": "MP minus K0",
        "C2 MP-MM": "C2: MP minus MM",
        "C4 MPs-MP": "C4: MP-s minus MP",
        "K0w-K0": "K0-w minus K0",
    }
    rows = []
    for period, name in (("scored", "scored year"), ("replicate", "replicate months")):
        for key, label in labels.items():
            item = data[period]["contrasts"][key]["tail"]
            rows.append(
                {
                    "contrast": label,
                    "period": name,
                    "estimate": item["difference"],
                    "low": item["lower"],
                    "high": item["upper"],
                }
            )
    table = pl.DataFrame(rows)
    order = list(labels.values())
    base = alt.Chart(table).encode(
        y=alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        yOffset="period:N",
        color=alt.Color(
            "period:N",
            scale=alt.Scale(
                domain=["scored year", "replicate months"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
            ),
            legend=alt.Legend(title=None, orient="top", direction="horizontal"),
        ),
    )
    rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title=[
                "Difference in mean tail pinball",
                "(points of capacity; below zero favours the first arm)",
            ],
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (zero + rule + dot).properties(
        width=WIDTH - 220,
        height=40 * len(order),
        title=_title(
            text="The member-aware residual table does not beat today's table, rescaled",
            subtitle=(
                "Dot: difference in mean pinball loss over q01, q05, q95, and q99, five decision "
                "substations, leads 0.5 to 168 hours, rows outside the switching events. Line: "
                "98.3% interval from resampling whole calendar months (the planned contrasts C1, "
                "C2, and C4) or the same level for the others. Dashed line: no difference."
            ),
        ),
    )


def _reliability_frame() -> pl.DataFrame:
    rows = []
    for arm in ("K0", "K0-w", "MP", "MP-1"):
        frame = results.with_bands(
            frame=results.load(arm=arm, period="scored", sites=common.DECISION_SITES)
        ).filter((pl.col("lead_hours") >= 0.5) & (pl.col("lead_hours") <= 168.0))
        for band in range(4):
            chosen = frame.filter(pl.col("lead_band") == band)
            for level, column in zip(MIXTURE_LEVELS, results.QUANTILES, strict=True):
                rows.append(
                    {
                        "arm": ARM_NAMES[arm],
                        "band": BANDS[band],
                        "level": level,
                        "gap": cast("float", (chosen["actual"] <= chosen[column]).mean()) - level,
                    }
                )
    return pl.DataFrame(rows)


def figure_reliability() -> alt.TopLevelMixin:
    """Coverage minus nominal level at each of the 11 levels, per lead band."""
    table = _reliability_frame()
    colours = [ARM_COLOURS[a] for a in ("K0", "K0-w", "MP", "MP-1")]
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "level:Q",
                title="Quantile level",
                axis=alt.Axis(values=[0.01, 0.25, 0.5, 0.75, 0.99], format=".2f"),
            ),
            y=alt.Y("gap:Q", title="Coverage minus level"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(table["arm"].unique(maintain_order=True)), range=colours
                ),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=330),
            ),
        )
        .properties(width=WIDTH - 80, height=110)
        .facet(row=alt.Row("band:N", sort=list(BANDS), title="Lead band"))
    )
    return chart.properties(
        title=_title(
            text="The four arms with a table are slightly too narrow, by up to 2.4 points",
            subtitle=(
                "Share of observed values at or below each forecast quantile, minus the quantile "
                "level, scored year, five decision substations, rows outside the switching "
                "events. Zero is perfect calibration; above zero at low levels and below zero at "
                "high levels means the band is too narrow. Arm MO (members only) is left out "
                "because its gaps reach 36 points."
            ),
        )
    )


def figure_variance(*, accounting: dict) -> alt.TopLevelMixin:
    """Members' variance, table variance of K0-w and MP, and squared error of the mean."""
    total = np.sum([np.array(list(v.values())) for v in accounting["variance"].values()], axis=0)
    rows = []
    for band in range(4):
        w, k0, mp, err, n, _ = total[band]
        for name, value in (
            ("members' variance (σ_w²)", w / n),
            ("K0-w table atoms' variance", k0 / n),
            ("MP table atoms' variance", mp / n),
            ("squared error of the ensemble mean", err / n),
        ):
            rows.append({"band": BANDS[band], "quantity": name, "value": float(value)})
    table = pl.DataFrame(rows)
    names = [str(r["quantity"]) for r in rows[:4]]
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("band:N", sort=list(BANDS), title="Lead band", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("quantity:N", sort=names),
            y=alt.Y("value:Q", title="Variance (points of capacity squared)"),
            color=alt.Color(
                "quantity:N",
                sort=names,
                scale=alt.Scale(
                    domain=names,
                    range=[ocf.DATA_GREEN, ocf.DATA_BLUE, ocf.BRAND_ORANGE, "#333333"],
                ),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=330),
            ),
        )
        .properties(width=WIDTH - 80, height=240)
    )
    return chart.properties(
        title=_title(
            text="Model error, not weather error, sets the width of the table",
            subtitle=(
                "Scored year, five decision substations, rows outside the switching events. The "
                "table's variance is the variance of its 11 atoms weighted by their probability "
                "masses, with the tuned factor applied."
            ),
        )
    )


def figure_cold(*, accounting: dict) -> alt.TopLevelMixin:
    """Night-band q95 and q99 coverage on cold-snap days of the scored winter."""
    rows = []
    for arm, item in accounting["cold_nights"].items():
        for column, level in (("q95", 0.95), ("q99", 0.99)):
            rows.append(
                {
                    "arm": ARM_NAMES[arm],
                    "quantile": f"q{int(level * 100)}",
                    "coverage": item[column],
                    "level": level,
                }
            )
    table = pl.DataFrame(rows)
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", title=None, axis=alt.Axis(labelLimit=330)),
            x=alt.X(
                "coverage:Q",
                title="Share of observed values at or below the quantile",
                scale=alt.Scale(domain=[0, 1]),
            ),
            yOffset="quantile:N",
            color=alt.Color(
                "quantile:N",
                scale=alt.Scale(domain=["q95", "q99"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
    )
    nominal = (
        alt.Chart(pl.DataFrame({"x": [0.95, 0.99]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (chart + nominal).properties(
        width=WIDTH - 260,
        height=220,
        title=_title(
            text="On cold winter nights demand exceeds the 95% quantile a quarter of the time",
            subtitle=(
                "Night hours (00:00 to 05:59 local), day-ahead leads (15.5 to 39 hours), cold-snap "
                "days of the scored winter at the five decision substations, 36 substation-days. "
                "Dashed lines: the nominal levels 0.95 and 0.99."
            ),
        ),
    )


def figure_ranks(*, accounting: dict) -> alt.TopLevelMixin:
    """Rank histogram of the observed value among the 51 corrected members, per lead band."""
    hist = np.sum([np.array(v["rank_histogram"]) for v in accounting["members"].values()], axis=0)
    rows = []
    for band in range(4):
        counts = hist[band]
        share = counts / counts.sum() * counts.shape[0]
        rows.extend(
            {"band": BANDS[band], "rank": rank, "ratio": float(v)} for rank, v in enumerate(share)
        )
    table = pl.DataFrame(rows)
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_GREEN)
        .encode(
            x=alt.X(
                "rank:O",
                title="Rank of the observed value among the 51 members (0 is below all)",
                axis=alt.Axis(values=[0, 10, 20, 30, 40, 51]),
            ),
            y=alt.Y("ratio:Q", title="Relative frequency"),
        )
        .properties(width=WIDTH - 80, height=100)
        .facet(row=alt.Row("band:N", sort=list(BANDS), title="Lead band"))
    )
    return chart.properties(
        title=_title(
            text="The members are far too narrow for demand",
            subtitle=(
                "Rank histogram of the observed power among the 51 corrected member forecasts, "
                "scored year, five decision substations. A flat histogram (1) is calibrated; "
                "tall end bars mean the members' spread is too small."
            ),
        )
    )


def main() -> None:
    """Draw every figure."""
    data = json.loads((common.WORK_DIR / "results.json").read_text())
    accounting = json.loads((common.WORK_DIR / "accounting.json").read_text())
    _save(figure_headline(data=data), name="figure_01_headline")
    _save(figure_cold(accounting=accounting), name="figure_02_cold_nights")
    _save(figure_reliability(), name="figure_03_reliability")
    _save(figure_variance(accounting=accounting), name="figure_04_variance")
    _save(figure_ranks(accounting=accounting), name="figure_05_ranks")


if __name__ == "__main__":
    main()
