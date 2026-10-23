"""Draw experiment 00307's figures from the JSON files in `work/`.

1. Share of the gap to the ERA5 oracle that each correction closes (headline, with intervals).
2. Wind speed minus the implied wind, by ERA5 speed bin: ERA5 and the ensemble mean.
3. Ensemble mean, 10th, and 90th percentile minus the implied wind, by lead day.
4. Error of the shifted ensemble mean against the implied wind, by shift in hours.
5. Distance to the implied wind of ERA5, ICON-D2, and UKV, by ERA5 speed bin.

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00307-*/e00307_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00307_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00307-wind-farm-power-as-wind-sensor"
)
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
SCOPE: Final[str] = "3 wind farms (W1 to W3), 53 Tuesday runs, July 2025 to June 2026."
BIN_ORDER: Final[list[str]] = [
    "[-inf, 3)",
    "[3, 5)",
    "[5, 7)",
    "[7, 9)",
    "[9, 11)",
    "[11, 14)",
    "[14, 18)",
    "[18, inf)",
]
ARM_LABELS: Final[dict[str, str]] = {
    "a_mean": "One speed factor, fitted to the implied wind",
    "e_lead": "Speed factor per lead day",
    "d_direction": "Speed factor per direction sector",
    "a_mean_era5": "One speed factor, fitted to ERA5 (control)",
    "e_lead_era5": "Speed factor per lead day, fitted to ERA5 (control)",
    "b_spread": "Spread of members widened per lead day",
    "c_shift_-3h": "Members read 3 h earlier",
    "c_shift_+3h": "Members read 3 h later",
    "all_edb": "Lead day, direction and spread together",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1(*, report: dict) -> alt.TopLevelMixin:
    """Dots and intervals of the share of the gap closed by each correction."""
    entry = report["kept"]["share_of_gap"]
    frame = pl.DataFrame(
        [
            {
                "arm": label,
                "point": entry[key]["point"],
                "low": entry[key]["interval"][0],
                "high": entry[key]["interval"][1],
                "control": "control" in label,
            }
            for key, label in ARM_LABELS.items()
        ]
    )
    base = alt.Chart(frame).encode(
        y=alt.Y("arm:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=330))
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [100.0]}))
        .mark_rule(color="#666666", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "low:Q", title="Share of the gap to the ERA5 oracle that the correction closes (%)"
        ),
        x2="high:Q",
        color=alt.Color(
            "control:N",
            scale=alt.Scale(domain=[False, True], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
            legend=None,
        ),
    )
    dots = base.mark_circle(size=70).encode(  # ty: ignore[unresolved-attribute]
        x="point:Q",
        color=alt.Color(
            "control:N",
            scale=alt.Scale(domain=[False, True], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
            legend=None,
        ),
    )
    return alt.layer(
        bars, dots, rule
    ).properties(
        width=WIDTH,
        height=230,
        title=_title(
            text="A plain speed factor closes most of the shortfall; spread and timing close none",
            lines=[
                SCOPE
                + " Dot: estimate. Line: 95% interval from resampling the 12 months. Orange: fitted to the "  # noqa: E501
                "wind that the farms' own power implies. Blue: fitted to ERA5 (control). Dashed line: "  # noqa: E501
                "all of the 18.7-point gap between the raw ensemble (34.1% of kept rows below the "
                "median) and the ERA5 oracle (52.8%) closed. Larger is better."
            ],
        ),
    )


def figure_2(*, implied: dict, ensemble: dict) -> alt.TopLevelMixin:
    """Mean speed minus implied wind by ERA5 speed bin, ERA5 and ensemble mean."""
    era5 = {
        b["bin"]: b
        for b in implied["out of sample (2024-04 to 2026-06)"]["all_farms"]["by_speed_bin"]
    }
    rows = []
    for key, value in ensemble["scored"]["by_era5_bin"].items():
        if value["rows"] and key in era5:
            rows.append(
                {
                    "bin": key,
                    "product": "ERA5 (control)",
                    "diff": value["era5_minus_implied"],
                    "n": value["rows"],
                }
            )
            rows.append(
                {
                    "bin": key,
                    "product": "Ensemble mean",
                    "diff": value["ens_minus_implied"],
                    "n": value["rows"],
                }
            )
    frame = pl.DataFrame(rows).filter(pl.col("n") >= 500)
    order = [b for b in BIN_ORDER if b in set(frame["bin"])]
    chart = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "bin:N",
                sort=order,
                title="ERA5 100 m wind speed bin (m/s)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("diff:Q", title="Mean speed minus implied wind (m/s)"),
            color=alt.Color(
                "product:N",
                scale=alt.Scale(
                    domain=["Ensemble mean", "ERA5 (control)"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=220,
            title=_title(
                text="The ensemble is slower than the wind the farms' power implies above 5 m/s",
                lines=[
                    SCOPE
                    + " Rows with an implied wind only (observed power between 5% and 90% of the "
                    "curve's plateau). Zero: same as the implied wind. Bins with fewer than 500 rows are left out."  # noqa: E501
                ],
            ),
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(color="#666666", strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(chart, zero)


def figure_3(*, ensemble: dict) -> alt.TopLevelMixin:
    """Ensemble mean, 10th, and 90th percentile minus implied wind, by lead day."""
    rows = []
    for day, value in ensemble["scored"]["by_lead_day"].items():
        rows.append(
            {"day": int(day), "series": "Ensemble mean", "diff": value["ens_minus_implied"]}
        )
        rows.append(
            {
                "day": int(day),
                "series": "10th percentile of members",
                "diff": value["ens_p10_minus_implied"],
            }
        )
        rows.append(
            {
                "day": int(day),
                "series": "90th percentile of members",
                "diff": value["ens_p90_minus_implied"],
            }
        )
    frame = pl.DataFrame(rows)
    order = ["90th percentile of members", "Ensemble mean", "10th percentile of members"]
    chart = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "day:Q",
                title="Lead day (calendar days from the run's date)",
                axis=alt.Axis(values=list(range(10))),
            ),
            y=alt.Y("diff:Q", title="Speed minus implied wind (m/s)"),
            color=alt.Color(
                "series:N",
                sort=order,
                scale=alt.Scale(
                    domain=order, range=[ocf.DATA_SKY, ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=220,
            title=_title(
                text="The shortfall is there from day 0 and grows with lead; the 90th percentile stays near zero",  # noqa: E501
                lines=[SCOPE + " Rows with an implied wind. Zero: same as the implied wind."],
            ),
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(color="#666666", strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(chart, zero)


def figure_4(*, ensemble: dict) -> alt.TopLevelMixin:
    """RMSE of the time-shifted ensemble mean against the implied wind."""
    frame = pl.DataFrame(
        [{"hours": float(k), "rmse": v["rmse"]} for k, v in ensemble["scored"]["timing"].items()]
    )
    return (
        alt.Chart(frame)
        .mark_line(point=True, color=ocf.BRAND_ORANGE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hours:Q",
                title="Hours the ensemble mean is read later than the valid time",
                axis=alt.Axis(values=[-3, -2, -1, 0, 1, 2, 3]),
            ),
            y=alt.Y(
                "rmse:Q", title="RMSE against the implied wind (m/s)", scale=alt.Scale(zero=False)
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="Shifting the ensemble in time does not bring it closer to the implied wind",
                lines=[
                    SCOPE
                    + " Rows with an implied wind. Smaller is better; the minimum is at 0 to 0.5 hours."  # noqa: E501
                ],
            ),
        )
    )


def figure_5(*, highres: dict) -> alt.TopLevelMixin:
    """RMSE against the implied wind of ERA5, ICON-D2, and UKV by ERA5 speed bin."""
    names = {"era5": "ERA5", "icon_d2": "ICON-D2, freshest run", "ukv_lead0": "UKV, lead 0"}
    rows = [
        {"bin": b, "product": names[p], "rmse": v[p]["rmse"]}
        for b, v in highres["distance_to_implied"]["bins"].items()
        for p in names
    ]
    order = list(highres["distance_to_implied"]["bins"])
    domain = list(names.values())
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "bin:N",
                sort=order,
                title="ERA5 100 m wind speed bin (m/s)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("rmse:Q", title="RMSE against the implied wind (m/s)"),
            color=alt.Color(
                "product:N",
                scale=alt.Scale(
                    domain=domain, range=[ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=220,
            title=_title(
                text="UKV's first hour is closer to the implied wind than ERA5 is; ICON-D2's is not",  # noqa: E501
                lines=[
                    "3 wind farms (W1 to W3), July 2025 to June 2026, half-hours with an implied wind. Smaller is better."  # noqa: E501
                ],
            ),
        )
    )


def main() -> None:
    """Write the five figures."""
    work = common.WORK_DIR
    report = json.loads((work / "report.json").read_text())
    implied = json.loads((work / "implied.json").read_text())
    ensemble = json.loads((work / "ensemble.json").read_text())
    highres = json.loads((work / "highres.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    charts = [
        figure_1(report=report),
        figure_2(implied=implied, ensemble=ensemble),
        figure_3(ensemble=ensemble),
        figure_4(ensemble=ensemble),
        figure_5(highres=highres),
    ]
    for number, chart in enumerate(charts, start=1):
        chart.save(str(DOCS_DIR / f"figure_{number}.svg"))
        print(f"figure_{number}.svg")


if __name__ == "__main__":
    main()
