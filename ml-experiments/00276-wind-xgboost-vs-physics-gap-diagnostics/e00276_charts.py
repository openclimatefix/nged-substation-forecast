"""Draw experiment 00276's figures as SVG and print the numbers the write-up quotes.

Reads `work/score.json`, `work/rows.parquet`, `work/loyo.parquet`, and `work/d5.json`.

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    cd ml-experiments/00276-wind-xgboost-vs-physics-gap-diagnostics
    uv run --no-sync python e00276_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00276_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00276-wind-xgboost-vs-physics-gap-diagnostics"
)
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
GREY: Final[str] = "#999999"
SCOPE: Final[str] = "3 wind farms, 53 Tuesday runs, 3 seeds, July 2025 to June 2026."
SETTING_LABELS: Final[dict[str, str]] = {
    "era5": "ERA5 as the weather",
    "ens": "51 ensemble members as the weather",
}
ROW_LABELS: Final[dict[str, str]] = {
    "all": "All rows",
    "kept": "Rows kept by the mask",
    "idle": "Idle-band rows only",
    "above_cut_in": "Wind above cut-in",
    "above_rated": "Wind above rated",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _interval_chart(
    *,
    data: pl.DataFrame,
    y: str,
    order: list[str],
    colour: str | None,
    colours: list[str] | None,
    title: alt.TitleParams,
    xtitle: str,
    height: int,
) -> alt.TopLevelMixin:
    base = alt.Chart(data).encode(
        y=alt.Y(f"{y}:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    )
    if colour and colours is not None:
        domain = data[colour].unique(maintain_order=True).to_list()
        encode = {
            "color": alt.Color(
                f"{colour}:N",
                scale=alt.Scale(domain=domain, range=colours),
                legend=alt.Legend(title=None, orient="top"),
            ),
            "yOffset": alt.YOffset(f"{colour}:N"),
        }
    else:
        encode = {"color": alt.value(ocf.DATA_BLUE)}
    points = base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title=xtitle), **encode
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q", **encode)  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(width=WIDTH, height=height, title=title)


def contrast(*, score: dict, key: str, entry: str) -> list[float]:
    """Return `[estimate, lower, upper]` for one contrast and one `row set | metric` entry."""
    found = score["contrasts"].get(key, {}).get(entry)
    if found is None and " - " in key:
        setting, pair = key.split(" | ")
        first, second = pair.split(" - ")
        reverse = score["contrasts"].get(f"{setting} | {second} - {first}", {}).get(entry)
        if reverse:
            found = [-reverse[0], -reverse[2], -reverse[1]]
    if found is None:
        msg = f"no contrast {key!r} with entry {entry!r}"
        raise KeyError(msg)
    return found


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """D1 and D2: core minus XGBoost under each training and evaluation mask."""
    rows = []
    for setting, label in SETTING_LABELS.items():
        for train, key in (
            ("Trained on all rows", "P - X"),
            ("Trained without idle rows", "P_nm - X_nm"),
        ):
            for row_set, row_label in ROW_LABELS.items():
                entry = contrast(score=score, key=f"{setting} | {key}", entry=f"{row_set} | qs")
                rows.append(
                    {
                        "setting": label,
                        "scored on": row_label,
                        "training": train,
                        "difference": entry[0],
                        "lower": entry[1],
                        "upper": entry[2],
                    }
                )
    data = pl.DataFrame(rows)
    panels = [
        _interval_chart(
            data=data.filter(pl.col("setting") == label),
            y="scored on",
            order=list(ROW_LABELS.values()),
            colour="training",
            colours=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
            title=alt.TitleParams(label, anchor="start", fontSize=12),
            xtitle="Physics core minus XGBoost (points of capacity; positive: XGBoost better)",
            height=150,
        )
        for label in SETTING_LABELS.values()
    ]
    return alt.vconcat(*panels).properties(  # ty: ignore[invalid-argument-type]
        title=_title(
            text="Removing the idle rows does not shrink XGBoost's lead",
            lines=[
                "Quantile score of the physics core minus XGBoost, by the rows scored and the rows "
                "trained on. Grey line: no difference. Dot: estimate. Line: 95% interval from "
                "re-drawing whole months and seeds. " + SCOPE
            ],
        )
    )


def gap_tables(*, rows: pl.DataFrame, loyo: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return D3's mean gap and share of the total gap by dimension."""
    seasons = {12: "Dec-Feb", 1: "Dec-Feb", 2: "Dec-Feb", 3: "Mar-May", 4: "Mar-May", 5: "Mar-May"}
    seasons |= {6: "Jun-Aug", 7: "Jun-Aug", 8: "Jun-Aug"}

    def prepare(frame: pl.DataFrame, gap: pl.Expr) -> pl.DataFrame:
        return frame.with_columns(
            gap=gap,
            speed_bin=pl.col("speed_100m").floor().clip(0, 22).cast(pl.Int32),
            sector=((pl.col("direction_deg") + 22.5) % 360 // 45).cast(pl.Int32),
            season=pl.col("valid_month").replace_strict(seasons, default="Sep-Nov"),
            hour_block=(pl.col("valid_hour") // 4 * 4).cast(pl.Int32),
        )

    era5 = prepare(rows, pl.col("qs__P__era5") - pl.col("qs__X__era5"))
    ens = prepare(rows, pl.col("qs__P__ens") - pl.col("qs__X__ens"))
    every = prepare(loyo, pl.col("qs_P") - pl.col("qs_X"))
    out = {}
    for name, frame, dim in (
        ("speed_bin", every, "speed_bin"),
        ("sector", every, "sector"),
        ("site", every, "site"),
        ("season", every, "season"),
        ("hour_block", every, "hour_block"),
        ("block", every, "block"),
        ("lead_day_ens", ens, "lead_day"),
        ("lead_day_era5", era5, "lead_day"),
        ("site_main", era5, "site"),
    ):
        total = frame["gap"].sum()
        out[name] = (
            frame.group_by(dim)
            .agg(
                n=pl.len(),
                gap=pl.col("gap").mean(),
                share=pl.col("gap").sum() / total,
            )
            .sort(dim)
            .rename({dim: "cell"})
        )
    out["heat"] = (
        every.group_by("speed_bin", "sector")
        .agg(n=pl.len(), gap=pl.col("gap").mean(), total=pl.col("gap").sum())
        .with_columns(share=pl.col("total") / every["gap"].sum())
    )
    return out


def figure_2(*, tables: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """D3: the gap by wind-speed bin and direction sector, with the share of the total gap."""
    heat = tables["heat"].filter(pl.col("n") >= 30)
    sector_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    heat = heat.with_columns(
        direction=pl.col("sector").replace_strict(
            dict(enumerate(sector_names)), return_dtype=pl.String
        )
    )
    cells = (
        alt.Chart(heat)  # ty: ignore[unresolved-attribute]
        .mark_rect()
        .encode(
            x=alt.X("speed_bin:O", title="ERA5 100 m wind speed (m/s, 1 m/s bins)"),
            y=alt.Y("direction:N", sort=sector_names, title="Direction wind blows from"),
            color=alt.Color(
                "share:Q",
                title="Share of total gap",
                scale=alt.Scale(range=["#FFFFFF", ocf.BRAND_ORANGE]),
                legend=alt.Legend(orient="top", format=".0%"),
            ),
            tooltip=["speed_bin", "direction", "n", "gap", "share"],
        )
        .properties(width=WIDTH, height=190)
    )
    speed = tables["speed_bin"]
    line = (
        alt.Chart(speed)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("cell:O", title="ERA5 100 m wind speed (m/s, 1 m/s bins)"),
            y=alt.Y("gap:Q", title="Mean gap (points of capacity)"),
        )
        .properties(width=WIDTH, height=130)
    )
    return alt.vconcat(cells, line).properties(
        title=_title(
            text="The gap sits at 9 to 13 m/s, where the power curve is steep",
            lines=[
                (
                    "Heat map: share of the physics core's total excess over XGBoost in each cell "
                    "(orange: larger). Line: mean excess per half-hour in each speed bin. Every "
                    "half-hour of September 2019 to June 2025 scored out of sample, with each "
                    "July-to-June year held out in turn. Excess is the core's quantile score minus "
                    "XGBoost's; positive means XGBoost is better. W1 to W3."
                )
            ],
        )
    )


def figure_3(*, tables: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """D3: mean gap by farm, season, hour of day, year, and lead day."""
    panels = []
    spec = (
        ("site", "Farm", None),
        ("season", "Season", ["Dec-Feb", "Mar-May", "Jun-Aug", "Sep-Nov"]),
        ("hour_block", "Hour of day (UTC, 4-hour blocks starting at)", None),
        ("block", "July-to-June year (0: Sep 2019 to Jun 2020, 5: Jul 2024 to Jun 2025)", None),
        (
            "lead_day_ens",
            "Lead day, ensemble members as the weather (July 2025 to June 2026)",
            None,
        ),
    )
    for name, label, order in spec:
        frame = tables[name]
        panels.append(
            alt.Chart(frame)  # ty: ignore[unresolved-attribute]
            .mark_bar(color=ocf.DATA_BLUE)
            .encode(
                x=alt.X("cell:O", title=label, sort=order),
                y=alt.Y("gap:Q", title="Mean gap (points)"),
            )
            .properties(width=WIDTH, height=90)
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="The gap appears at every farm, season, and hour of day",
            lines=[
                (
                    "Mean quantile score of the physics core minus XGBoost per half-hour "
                    "(points of capacity; positive: XGBoost better). Panels 1 to 4 use the "
                    "held-out-year fits with ERA5 as the weather; the last panel uses the "
                    "scored cases."
                )
            ],
        )
    )


def figure_4(*, score: dict) -> alt.TopLevelMixin:
    """D4: absolute score of each XGBoost input set against the core, ERA5 as the weather."""
    sets = [
        ("X1", "100 m wind speed only"),
        ("X2", "plus 100 m direction"),
        ("X3", "plus air temperature"),
        ("X4", "plus 10 m speed and direction"),
        ("X5", "plus hour of day and day of year"),
        ("X6", "plus year"),
    ]
    rows = []
    for name, label in sets:
        entry = contrast(score=score, key=f"era5 | P - {name}", entry="kept | qs")
        rows.append(
            {
                "inputs": f"{name}: {label}",
                "difference": entry[0],
                "lower": entry[1],
                "upper": entry[2],
            }
        )
    data = pl.DataFrame(rows)
    return _interval_chart(
        data=data,
        y="inputs",
        order=data["inputs"].to_list(),
        colour=None,
        colours=None,
        title=_title(
            text="XGBoost passes the physics core only after the 10 m wind is added",
            lines=[
                "Quantile score of the physics core minus XGBoost fitted on the inputs shown "
                "(each row adds to the row above), ERA5 as the weather, rows kept by the mask. "
                "Positive: XGBoost better. Grey line: same as the core. Dot: estimate. Line: 95% "
                "interval from re-drawing whole months and seeds. " + SCOPE
            ],
        ),
        xtitle="Physics core minus XGBoost (points of capacity)",
        height=200,
    )


def figure_5(*, d5: dict) -> alt.TopLevelMixin:
    """D5: permutation importance and partial dependence of the core's residual."""
    names = {
        "wind_speed_100m": "100 m wind speed",
        "wind_speed_10m": "10 m wind speed",
        "direction_deg": "100 m wind direction",
        "temperature_c": "air temperature",
        "hour": "hour of day",
        "doy": "day of year",
        "year_frac": "year",
    }
    importance = pl.DataFrame(
        [
            {"farm": farm, "input": names[k], "rise": v}
            for farm, content in d5.items()
            for k, v in content["permutation_importance_mse_rise"].items()
        ]
    )
    bars = (
        alt.Chart(importance)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("input:N", sort="-x", title=None),
            x=alt.X("rise:Q", title="Rise in mean squared error when shuffled (points squared)"),
            color=alt.Color(
                "farm:N",
                scale=alt.Scale(
                    domain=["W1", "W2", "W3"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            yOffset="farm:N",
        )
        .properties(width=WIDTH, height=220)
    )
    pdp = pl.DataFrame(
        [
            {
                "farm": farm,
                "input": names[k],
                "value": g,
                "effect": e,
            }
            for farm, content in d5.items()
            for k in ("direction_deg", "wind_speed_100m", "hour")
            for g, e in zip(
                content["partial_dependence"][k]["grid"],
                content["partial_dependence"][k]["effect_points"],
                strict=True,
            )
        ]
    )
    lines = (
        alt.Chart(pdp)  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            x=alt.X("value:Q", title=None),
            y=alt.Y("effect:Q", title="Predicted residual (points)"),
            color=alt.Color(
                "farm:N",
                scale=alt.Scale(
                    domain=["W1", "W2", "W3"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN],
                ),
                legend=None,
            ),
        )
        .properties(width=170, height=110)
        .facet(
            column=alt.Column(
                "input:N",
                title=None,
                sort=["100 m wind direction", "100 m wind speed", "hour of day"],
            )
        )
        .resolve_scale(x="independent")
    )
    return alt.vconcat(bars, lines).properties(
        title=_title(
            text="The core's error depends on direction, wind speeds, and hour, but only a little",
            lines=[
                "Small XGBoost model of the physics core's median residual (observed minus the "
                "core's median, points of capacity), scored on the held-out year with ERA5 as the "
                "weather. Top: permutation importance. Bottom: partial dependence on direction "
                "(degrees), 100 m wind speed (m/s), and hour of day (UTC). " + SCOPE
            ],
        )
    )


def figure_6(*, score: dict) -> alt.TopLevelMixin:
    """D6 to D8 and D10: each variant of the core minus the core, ERA5 as the weather."""
    variants = [
        ("S", "Binned power curve (no 10 m speed, no density)"),
        ("P_net", "Monotone neural power curve"),
        ("P_noshear", "Core with a fixed shear exponent"),
        ("P_dir", "Core + direction-dependent speed"),
        ("P_dens", "Core + free air-density exponent"),
        ("P_bin_speed", "Core, residual quantiles binned on speed"),
        ("P_pinball", "Core fitted on pinball loss + speed-dependent spread"),
        ("P_year", "Core + per-year scale"),
        ("P_month", "Core + per-month scale"),
        ("P_scaled", "Core on farm-scaled ERA5 speeds"),
    ]
    rows = []
    for name, label in variants:
        entry = contrast(score=score, key=f"era5 | {name} - P", entry="kept | qs")
        rows.append(
            {"variant": label, "difference": entry[0], "lower": entry[1], "upper": entry[2]}
        )
    data = pl.DataFrame(rows)
    return _interval_chart(
        data=data,
        y="variant",
        order=data["variant"].to_list(),
        colour=None,
        colours=None,
        title=_title(
            text="No change to the core's power curve, spread, or scale closes the gap",
            lines=[
                "Quantile score of each variant minus the physics core (negative: the variant is "
                "better), ERA5 as the weather, rows kept by the mask. The gap to XGBoost is "
                "+0.31. Grey line: same as the core. Dot: estimate. Line: 95% interval from "
                "re-drawing whole months and seeds. " + SCOPE
            ],
        ),
        xtitle="Variant minus physics core (points of capacity)",
        height=260,
    )


def figure_7(*, score: dict) -> alt.TopLevelMixin:
    """D9: both models' kept-row score, and their gap, by months of training."""
    months = (3, 6, 12, 24, 48)
    rows = []
    for m in months:
        for model, key in (("Physics core", f"P_w{m}"), ("XGBoost", f"X_w{m}")):
            rows.append(
                {
                    "months": m,
                    "model": model,
                    "score": score["scores"]["era5"][key]["kept"]["qs"],
                }
            )
    rows += [
        {
            "months": 69,
            "model": "Physics core",
            "score": score["scores"]["era5"]["P"]["kept"]["qs"],
        },
        {"months": 69, "model": "XGBoost", "score": score["scores"]["era5"]["X"]["kept"]["qs"]},
    ]
    data = pl.DataFrame(rows)
    scores = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "months:Q",
                scale=alt.Scale(type="log"),
                title="Months of training (log scale; 69: all)",
                axis=alt.Axis(values=[3, 6, 12, 24, 48, 69]),
            ),
            y=alt.Y("score:Q", title="Quantile score (points)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(
                    domain=["Physics core", "XGBoost"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=WIDTH, height=170)
    )
    gap_rows = []
    for m in months:
        entry = contrast(score=score, key=f"era5 | P_w{m} - X_w{m}", entry="kept | qs")
        gap_rows.append({"months": m, "difference": entry[0], "lower": entry[1], "upper": entry[2]})
    entry = contrast(score=score, key="era5 | P - X", entry="kept | qs")
    gap_rows.append({"months": 69, "difference": entry[0], "lower": entry[1], "upper": entry[2]})
    gaps = pl.DataFrame(gap_rows)
    base = alt.Chart(gaps).encode(
        x=alt.X(
            "months:Q",
            scale=alt.Scale(type="log"),
            title="Months of training (log scale; 69: all)",
            axis=alt.Axis(values=[3, 6, 12, 24, 48, 69]),
        )
    )
    band = base.mark_area(opacity=0.25, color=ocf.DATA_BLUE).encode(y="lower:Q", y2="upper:Q")  # ty: ignore[unresolved-attribute]
    line = base.mark_line(point=True, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("difference:Q", title="Core minus XGBoost (points)")
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=GREY).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    return alt.vconcat(
        scores, alt.layer(zero, band, line).properties(width=WIDTH, height=130)
    ).properties(
        title=_title(
            text="The core stops improving after about 2 years; XGBoost keeps improving",
            lines=[
                "Quantile score on the rows kept by the mask, ERA5 as the weather, with training "
                "cut to the last 3 to 48 months before June 2025. Bottom: the core's score minus "
                "XGBoost's (positive: XGBoost better), shaded 95% interval from re-drawing whole "
                "months and seeds. " + SCOPE
            ],
        )
    )


def main() -> None:
    """Write the figures and print the D3 tables."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    d5 = json.loads((common.WORK_DIR / "d5.json").read_text())
    rows = pl.read_parquet(common.WORK_DIR / "rows.parquet")
    loyo = pl.read_parquet(common.WORK_DIR / "loyo.parquet")
    tables = gap_tables(rows=rows, loyo=loyo)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1(score=score)),
        ("figure_2", figure_2(tables=tables)),
        ("figure_3", figure_3(tables=tables)),
        ("figure_4", figure_4(score=score)),
        ("figure_5", figure_5(d5=d5)),
        ("figure_6", figure_6(score=score)),
        ("figure_7", figure_7(score=score)),
    ):
        chart.save(DOCS_DIR / f"{name}.svg")
    pl.Config.set_tbl_rows(40)
    for name, table in tables.items():
        if name != "heat":
            print(name)
            print(table.with_columns(pl.col(pl.Float64).round(3)))
    print(
        "loyo total gap",
        round(float((loyo["qs_P"] - loyo["qs_X"]).mean()), 3),  # ty: ignore[invalid-argument-type]
        "kept",
        round(float((loyo["qs_P"] - loyo["qs_X"]).filter(~loyo["masked"]).mean()), 3),  # ty: ignore[invalid-argument-type]
    )


if __name__ == "__main__":
    main()
