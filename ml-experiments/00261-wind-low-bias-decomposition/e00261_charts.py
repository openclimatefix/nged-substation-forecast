"""Draw experiment 00261's figures from `work/report.json` and `work/sources.json`.

1. Share of rows below the median at each stage, by farm (kept rows, scored year).
2. The same shares by lead day.
3. Ensemble-minus-ERA5 wind speed by ERA5 speed bin.
4. Mean median forecast at each weather stage, and the input swaps.
5. Training source: ERA5-trained against ensemble-trained models.

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00261-*/e00261_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00261_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00261-wind-low-bias-decomposition"
)
PLOT_WIDTH: Final[int] = 380
WRAP: Final[int] = 80
SCOPE: Final[str] = (
    "3 wind farms (W1 to W3), 53 Tuesday runs, July 2025 to June 2026, "
    "rows kept by the idle-row mask."
)
STAGE_ORDER: Final[tuple[str, ...]] = (
    "Oracle: ERA5 wind",
    "Ensemble-mean wind",
    "51-member mixture (R1)",
)
STAGE_COLOURS: Final[dict[str, str]] = {
    "Oracle: ERA5 wind": ocf.DATA_GREEN,
    "Ensemble-mean wind": ocf.DATA_BLUE,
    "51-member mixture (R1)": ocf.BRAND_ORANGE,
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _half_rule() -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"x": [0.5]}))
        .mark_rule(color="#666666", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )


def figure_1(*, report: dict) -> alt.TopLevelMixin:
    """Bars of the share below the median at each stage, by farm and pooled."""
    entry = report["scored"]
    rows = [
        {"farm": "All three farms", "stage": stage, "share": entry["share_kept"][stage]}
        for stage in STAGE_ORDER
    ]
    for farm, shares in entry["share_kept_by_farm"].items():
        rows.extend({"farm": farm, "stage": stage, "share": shares[stage]} for stage in STAGE_ORDER)
    frame = pl.DataFrame(rows)
    bars = (
        alt.Chart()
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("stage:N", sort=list(STAGE_ORDER), title=None, axis=None),
            x=alt.X(
                "share:Q",
                title="Share of rows with observed power below the median (0.5 is calibrated)",
                scale=alt.Scale(domain=[0, 0.7]),
            ),
            color=alt.Color(
                "stage:N",
                sort=list(STAGE_ORDER),
                scale=alt.Scale(
                    domain=list(STAGE_ORDER), range=[STAGE_COLOURS[s] for s in STAGE_ORDER]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=PLOT_WIDTH, height=60)
    )
    text = bars.mark_text(align="left", dx=3).encode(text=alt.Text("share:Q", format=".2f"))
    return (
        alt.layer(bars, text, _half_rule())
        .facet(
            row=alt.Row("farm:N", sort=["All three farms", "W1", "W2", "W3"], title=None),
            data=frame,
        )
        .properties(
            title=_title(
                text="The shortfall appears when ensemble wind replaces ERA5",
                lines=[
                    SCOPE,
                    "Dashed line: calibrated. Oracle: core fed ERA5 wind at the valid time.",
                ],
            )
        )
    )


def figure_2(*, report: dict) -> alt.TopLevelMixin:
    """Lines of the share below the median by lead day."""
    rows = []
    for day, shares in report["scored"]["share_kept_by_lead_day"].items():
        rows.extend(
            {"lead_day": int(day), "stage": stage, "share": shares[stage]} for stage in STAGE_ORDER
        )
    frame = pl.DataFrame(rows).drop_nulls()
    base = alt.Chart(frame).encode(
        x=alt.X(
            "lead_day:Q",
            title="Lead (calendar days from the run's date to the valid time)",
            scale=alt.Scale(domain=[0, 10]),
        ),
        y=alt.Y(
            "share:Q",
            title="Share below the median (0.5 is calibrated)",
            scale=alt.Scale(domain=[0.2, 0.65]),
        ),
        color=alt.Color(
            "stage:N",
            sort=list(STAGE_ORDER),
            scale=alt.Scale(
                domain=list(STAGE_ORDER), range=[STAGE_COLOURS[s] for s in STAGE_ORDER]
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [0.5]}))
        .mark_rule(color="#666666", strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return (base.mark_line(point=True) + rule).properties(
        width=PLOT_WIDTH,
        height=200,
        title=_title(
            text="The ERA5-to-ensemble gap is similar at every lead",
            lines=[SCOPE],
        ),
    )


def figure_3(*, report: dict) -> alt.TopLevelMixin:
    """Bars of ensemble-mean minus ERA5 wind speed by ERA5 speed bin."""
    order = ["(-inf, 3]", "(3, 5]", "(5, 8]", "(8, 12]", "(12, 16]", "(16, 20]", "(20, inf]"]
    rows = []
    for key, value in report["scored"]["speed_bias_by_speed_bin"].items():
        for height, label in (("bias_100m", "100 m wind"), ("bias_10m", "10 m wind")):
            rows.append({"bin": key, "height": label, "bias": value[height], "rows": value["rows"]})
    frame = pl.DataFrame(rows).filter(pl.col("rows") >= 100)
    bars = (
        alt.Chart(frame)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("bin:N", sort=order, title="ERA5 100 m wind speed bin (m/s)"),
            y=alt.Y("bias:Q", title="Ensemble mean minus ERA5 (m/s; 0 is unbiased)"),
            xOffset="height:N",
            color=alt.Color(
                "height:N",
                scale=alt.Scale(
                    domain=["100 m wind", "10 m wind"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="#666666").encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    return (bars + zero).properties(
        width=PLOT_WIDTH,
        height=200,
        title=_title(
            text="Ensemble wind is slower than ERA5 above 8 m/s",
            lines=[SCOPE, "Bins with fewer than 100 rows are left out."],
        ),
    )


def figure_4(*, sources: dict) -> alt.TopLevelMixin:
    """Bars of the mean median forecast when one input is swapped from ERA5 to the ensemble mean."""
    arms = [
        ("q50 oracle", "ERA5 wind and temperature (oracle)"),
        ("q50 ens temperature only", "Ensemble-mean temperature only"),
        ("q50 ens 100 m speed only", "Ensemble-mean 100 m wind only"),
        ("q50 ens 10 m speed only", "Ensemble-mean 10 m wind only"),
        ("q50 ens 10 m and 100 m speed", "Ensemble-mean 10 m and 100 m wind"),
        ("q50 all three ens", "All three ensemble-mean inputs"),
    ]
    entry = sources["training_source"]["scored year"]["mean_q50_points"]
    frame = pl.DataFrame([{"arm": label, "points": entry[key]} for key, label in arms])
    bars = (
        alt.Chart(frame)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "arm:N",
                sort=[label for _, label in arms],
                title=None,
                axis=alt.Axis(labelLimit=280),
            ),
            x=alt.X("points:Q", title="Mean forecast median (points of capacity)"),
        )
    )
    text = bars.mark_text(align="left", dx=3).encode(text=alt.Text("points:Q", format=".1f"))
    return (bars + text).properties(
        width=PLOT_WIDTH,
        height=160,
        title=_title(
            text="Ensemble 10 m and 100 m winds each cut the median by 7 points",
            lines=[
                SCOPE,
                "Each bar feeds the same physics core; named inputs come from the ensemble mean.",
            ],
        ),
    )


def figure_5(*, sources: dict) -> alt.TopLevelMixin:
    """Bars of the share below the median for ERA5-trained and ensemble-trained models."""
    arms = [
        ("q50 era5-trained core", "Physics core", "Trained on ERA5"),
        ("q50 ens-trained core", "Physics core", "Trained on ensemble mean"),
        ("q50 era5-trained xgboost", "XGBoost", "Trained on ERA5"),
        ("q50 ens-trained xgboost", "XGBoost", "Trained on ensemble mean"),
    ]
    rows = []
    for subset in ("hindcast, held-out months 2025-01 to 2025-06", "scored year"):
        entry = sources["training_source"][subset]["share"]
        rows.extend(
            {
                "subset": "Held-out hindcast, Jan to Jun 2025"
                if subset.startswith("hindcast")
                else "Scored year",
                "model": model,
                "training": training,
                "share": entry[key],
            }
            for key, model, training in arms
        )
    frame = pl.DataFrame(rows)
    bars = (
        alt.Chart()
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("model:N", title=None),
            xOffset="training:N",
            y=alt.Y(
                "share:Q",
                title="Share below the median (0.5 is calibrated)",
                scale=alt.Scale(domain=[0, 0.6]),
            ),
            color=alt.Color(
                "training:N",
                scale=alt.Scale(
                    domain=["Trained on ERA5", "Trained on ensemble mean"],
                    range=[ocf.DATA_GREEN, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=PLOT_WIDTH // 2, height=180)
    )
    return (
        alt.layer(
            bars,
            alt.Chart().mark_rule(color="#666666", strokeDash=[4, 3]).encode(y=alt.datum(0.5)),  # ty: ignore[unresolved-attribute]
        )
        .facet(column=alt.Column("subset:N", title=None), data=frame)
        .properties(
            title=_title(
                text="Training on ensemble forecasts lifts the share by only 3 to 5 points",
                lines=["Kept rows. Every model reads one ensemble-mean value per case."],
            )
        )
    )


def main() -> None:
    """Write the figures as SVG."""
    work = common.WORK_DIR
    report = json.loads((work / "report.json").read_text())
    sources = json.loads((work / "sources.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        1: figure_1(report=report),
        2: figure_2(report=report),
        3: figure_3(report=report),
        4: figure_4(sources=sources),
        5: figure_5(sources=sources),
    }
    for number, chart in figures.items():
        chart.save(str(DOCS_DIR / f"figure_{number}.svg"))
        print(f"figure_{number}.svg")


if __name__ == "__main__":
    main()
