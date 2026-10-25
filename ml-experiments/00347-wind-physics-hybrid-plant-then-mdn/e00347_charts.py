"""Draw experiment 00347's Stage 1 to 3b figures as SVG from the stage tables and plant parameters.

Optimise afterwards with `npx svgo@4 --multipass --precision=1 --final-newline f.svg`. No chart
shows a calendar date or a capacity in megawatts, and every farm is labelled W1 to W3.

    export PYTHONPATH=ml-experiments/00230-wind-physics-core-part-1:\
        ml-experiments/00347-wind-physics-hybrid-plant-then-mdn
    uv run --no-sync python ml-experiments/00347-wind-physics-hybrid-plant-then-mdn/e00347_charts.py
"""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import json  # noqa: E402
import textwrap  # noqa: E402
from dataclasses import replace  # noqa: E402
from typing import Final  # noqa: E402

import altair as alt  # noqa: E402
import e00347_common as common  # noqa: E402
import e00347_eval as evaluation  # noqa: E402
import e00347_flags as flags  # noqa: E402
import e00347_plant as plant  # noqa: E402
import numpy as np  # noqa: E402
import plotting.ocf_theme as ocf  # noqa: E402
import polars as pl  # noqa: E402

PLOT_WIDTH: Final[int] = 560
PANEL_HEIGHT: Final[int] = 170
WRAP: Final[int] = 100
GREY: Final[str] = "#999999"
SCOPE: Final[str] = "3 wind farms, W1 to W3, ERA5 100 m wind."
WORK: Final = common.DATA_DIR / "work"
SECTOR_COLOURS: Final[tuple[str, ...]] = (
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
    ocf.BRAND_ORANGE_LIGHT,
    ocf.DATA_BLUE_LIGHT,
    GREY,
)
YEAR_COLOURS: Final[tuple[str, ...]] = (
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
    GREY,
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _stack(*, panels: list[alt.TopLevelMixin], title: alt.TitleParams) -> alt.VConcatChart:
    return alt.vconcat(*panels).properties(title=title)  # ty: ignore[invalid-argument-type]


def _farm_title(label: str) -> alt.TitleParams:
    return alt.TitleParams(text=label, anchor="start", fontSize=12)


def figure_1(*, density: pl.DataFrame, stats: pl.DataFrame) -> alt.VConcatChart:
    """Power against ERA5 wind speed: density, binned median, and 5% to 95% band."""
    panels = []
    for label in common.FARMS:
        cells = density.filter(pl.col("farm") == label).with_columns(
            speed0=pl.col("speed") - 0.25,
            speed1=pl.col("speed") + 0.25,
            power0=pl.col("power") - 0.025,
            power1=pl.col("power") + 0.025,
        )
        bins = stats.filter((pl.col("farm") == label) & (pl.col("kind") == "all"))
        heat = (
            alt.Chart(cells)
            .mark_rect(aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "speed0:Q",
                    title="ERA5 100 m wind speed (m/s)",
                    scale=alt.Scale(domain=[0, 25]),
                ),
                x2="speed1:Q",
                y=alt.Y(
                    "power0:Q",
                    title="Observed power (share of top level)",
                    scale=alt.Scale(domain=[0, 1.2]),
                ),
                y2="power1:Q",
                color=alt.Color(
                    "count:Q",
                    scale=alt.Scale(type="log", range=[ocf.DATA_SKY_LIGHT, ocf.DATA_BLUE]),
                    legend=None,
                ),
            )
        )
        band = (
            alt.Chart(bins)
            .mark_area(opacity=0.25, color=ocf.BRAND_ORANGE, aria=False)
            .encode(x="speed:Q", y="q05:Q", y2="q95:Q")  # ty: ignore[unresolved-attribute]
        )
        median = (
            alt.Chart(bins)
            .mark_line(color=ocf.BRAND_ORANGE, aria=False)
            .encode(x="speed:Q", y="median:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            alt.layer(heat, band, median).properties(
                width=PLOT_WIDTH, height=PANEL_HEIGHT, title=_farm_title(label)
            )
        )
    return _stack(
        panels=panels,
        title=_title(
            text="Power rises with the cube of wind speed, then flattens at the farm's top level",
            lines=[
                (
                    "Each farm's half-hourly power against ERA5 100 m wind speed, as a share of "
                    "the top level of "
                    "its fitted monotone curve. Blue shade: number of half-hours (darker is "
                    "more, log scale). "
                    "Orange line: median in 0.5 m/s bins. Orange band: 5% to 95% of the "
                    "half-hours in each bin. "
                    "Rows dropped by experiment 210's mask are left out."
                )
            ],
        ),
    )


def _lines(
    *,
    stats: pl.DataFrame,
    kind: str,
    keys: list,
    names: dict,
    colours: tuple[str, ...],
    text: str,
    lines: list[str],
    legend_title: str,
) -> alt.VConcatChart:
    panels = []
    for position, label in enumerate(common.FARMS):
        rows = stats.filter((pl.col("farm") == label) & (pl.col("kind") == kind)).with_columns(
            group=pl.col("key").replace_strict(names, default=None, return_dtype=pl.String)
        )
        chart = (
            alt.Chart(rows)
            .mark_line(aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "speed:Q", title="ERA5 100 m wind speed (m/s)", scale=alt.Scale(domain=[0, 20])
                ),
                y=alt.Y("median:Q", title="Median power (share of top level)"),
                color=alt.Color(
                    "group:N",
                    scale=alt.Scale(domain=list(names.values()), range=list(colours)),
                    legend=alt.Legend(title=legend_title, orient="top", columns=8)
                    if position == 0
                    else None,
                ),
            )
            .properties(width=PLOT_WIDTH, height=PANEL_HEIGHT, title=_farm_title(label))
        )
        panels.append(chart)
    return _stack(panels=panels, title=_title(text=text, lines=lines))


def figure_2(*, stats: pl.DataFrame) -> alt.VConcatChart:
    """Median power curve by calendar year."""
    years = sorted(stats.filter(pl.col("kind") == "year")["key"].unique().to_list())
    years = [y for y in years if y >= 2020]
    names = {y: str(y) for y in years}
    return _lines(
        stats=stats.filter((pl.col("kind") != "year") | pl.col("key").is_in(years)),
        kind="year",
        keys=years,
        names=names,
        colours=YEAR_COLOURS[: len(years)],
        text="The power curve is stable from year to year",
        lines=[
            (
                "Median power by 0.5 m/s wind-speed bin, one line per calendar year, as a share "
                "of the farm's top level. 2019 is left out because it holds only part of a year. "
            )
            + SCOPE
        ],
        legend_title="Calendar year",
    )


def figure_3(*, stats: pl.DataFrame) -> alt.VConcatChart:
    """Median power curve by direction sector."""
    names = dict(enumerate(common.SECTOR_NAMES))
    return _lines(
        stats=stats,
        kind="sector",
        keys=list(names),
        names=names,
        colours=SECTOR_COLOURS,
        text="At the same ERA5 speed, winds from the north-east and east give the most power",
        lines=[
            (
                "Median power by 0.5 m/s wind-speed bin, one line per 45 degree sector of the "
                "direction the wind blows from, as a share of the farm's top level. "
            )
            + SCOPE
        ],
        legend_title="Wind from",
    )


def figure_4(*, direction: pl.DataFrame) -> alt.VConcatChart:
    """Stage 1 direction curve with month-block bootstrap band."""
    panels = []
    for label in common.FARMS:
        rows = direction.filter(pl.col("farm") == label)
        band = (
            alt.Chart(rows)
            .mark_area(opacity=0.3, color=ocf.DATA_BLUE, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "direction_deg:Q",
                    title="ERA5 100 m wind direction (degrees the wind blows from)",
                    axis=alt.Axis(values=[0, 90, 180, 270, 360]),
                    scale=alt.Scale(domain=[0, 360]),
                ),
                y=alt.Y(
                    "lo:Q", title="Observed over fitted power", scale=alt.Scale(domain=[0.8, 1.2])
                ),
                y2="hi:Q",
            )
        )
        line = (
            alt.Chart(rows)
            .mark_line(color=ocf.DATA_BLUE, aria=False)
            .encode(x="direction_deg:Q", y="ratio:Q")  # ty: ignore[unresolved-attribute]
        )
        one = (
            alt.Chart(pl.DataFrame({"y": [1.0]}))
            .mark_rule(color=GREY, strokeDash=[4, 3], aria=False)
            .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            alt.layer(one, band, line).properties(
                width=PLOT_WIDTH, height=PANEL_HEIGHT, title=_farm_title(label)
            )
        )
    return _stack(
        panels=panels,
        title=_title(
            text=(
                "At the same ERA5 wind speed, power differs by 24% to 38% between the best and "
                "worst wind direction"
            ),
            lines=[
                (
                    "Observed power divided by the all-direction monotone curve, on half-hours "
                    "whose fitted power is 40% to 90% of the top level, "
                    "smoothed over about 18 degrees and divided by the all-direction average. "
                    "Grey dashed line: no departure. Blue line: estimate. "
                    "Blue band: 95% interval from re-drawing whole calendar months. The ratio "
                    "mixes wake loss with any speed error in ERA5 that depends on direction."
                )
            ],
        ),
    )


def figure_5(
    *,
    farms: dict[str, common.FarmData],
    priors: dict[str, float],
    stage2: dict[str, plant.PlantParameters],
    tops: dict[str, float],
) -> alt.VConcatChart:
    """Prior-only and fitted plant curves beside the binned median, on the training rows."""
    panels = []
    for label in common.FARMS:
        train = farms[label].train
        kept = ~train.idle_mask
        weather = train.weather.take(np.flatnonzero(kept))
        series = {
            "Binned median of the data": train.target[kept],
            "Prior-only plant": plant.expected_power(
                parameters=plant.prior_parameters(prior_scale=priors[label], shear_density=True),
                weather=weather,
                prior_scale=priors[label],
            ),
            "Fitted plant": plant.expected_power(
                parameters=stage2[label], weather=weather, prior_scale=priors[label]
            ),
        }
        speed_bin = np.floor(weather.speed_100m / 0.5).astype(int)
        rows = []
        for name, values in series.items():
            for b in np.unique(speed_bin):
                chosen = speed_bin == b
                if chosen.sum() >= 30 and (b + 0.5) * 0.5 <= 20:
                    rows.append(
                        {
                            "speed": (b + 0.5) * 0.5,
                            "power": float(np.median(values[chosen]) / tops[label]),
                            "curve": name,
                        }
                    )
        frame = pl.DataFrame(rows)
        order = ["Binned median of the data", "Prior-only plant", "Fitted plant"]
        panels.append(
            alt.Chart(frame.filter(pl.col("speed") <= 20))
            .mark_line(aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("speed:Q", title="ERA5 100 m wind speed (m/s)"),
                y=alt.Y("power:Q", title="Power (share of top level)"),
                color=alt.Color(
                    "curve:N",
                    sort=order,
                    scale=alt.Scale(
                        domain=order, range=[GREY, ocf.DATA_BLUE_LIGHT, ocf.BRAND_ORANGE]
                    ),
                    legend=alt.Legend(title=None, orient="top") if label == "W1" else None,
                ),
                strokeDash=alt.StrokeDash(
                    "curve:N",
                    sort=order,
                    scale=alt.Scale(domain=order, range=[[1, 0], [4, 3], [1, 0]]),
                    legend=None,
                ),
            )
            .properties(width=PLOT_WIDTH, height=PANEL_HEIGHT, title=_farm_title(label))
        )
    return _stack(
        panels=panels,
        title=_title(
            text=(
                "The fitted plant follows the data; the prior-only plant is close at W1 and W2 "
                "and reaches the top level too early at W3"
            ),
            lines=[
                (
                    "Median power in 0.5 m/s bins of ERA5 100 m wind speed, as a share of the "
                    "farm's top level: the data, and the plants evaluated on the same half-hours "
                    "(training rows before April 2024, experiment 210's mask applied). "
                )
                + SCOPE
            ],
        ),
    )


def figure_6(*, direction: pl.DataFrame, plant_ratio: pl.DataFrame) -> alt.VConcatChart:
    """The data's direction curve and the Stage 3 plant's implied curve."""
    panels = []
    for label in common.FARMS:
        data = direction.filter(pl.col("farm") == label).select(
            "direction_deg", ratio=pl.col("ratio"), series=pl.lit("Data (Stage 1)")
        )
        model = plant_ratio.filter(pl.col("farm") == label).select(
            "direction_deg", ratio=pl.col("plant_ratio"), series=pl.lit("Fitted plant (Stage 3)")
        )
        panels.append(
            alt.Chart(pl.concat([data, model]))
            .mark_line(aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "direction_deg:Q",
                    title="ERA5 100 m wind direction (degrees the wind blows from)",
                    axis=alt.Axis(values=[0, 90, 180, 270, 360]),
                    scale=alt.Scale(domain=[0, 360]),
                ),
                y=alt.Y(
                    "ratio:Q",
                    title="Power relative to no direction term",
                    scale=alt.Scale(domain=[0.8, 1.2]),
                ),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=["Data (Stage 1)", "Fitted plant (Stage 3)"],
                        range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                    ),
                    legend=alt.Legend(title=None, orient="top") if label == "W1" else None,
                ),
            )
            .properties(width=PLOT_WIDTH, height=PANEL_HEIGHT, title=_farm_title(label))
        )
    return _stack(
        panels=panels,
        title=_title(
            text="Three dips in wind speed reproduce most of the data's direction pattern",
            lines=[
                (
                    "Blue: the Stage 1 curve of observed over fitted power by direction. Orange: "
                    "the same quantity for the Stage 3 plant, power with its direction term over "
                    "power without it, on the same half-hours. "
                )
                + SCOPE
            ],
        ),
    )


def figure_7(*, scores: pl.DataFrame) -> alt.VConcatChart:
    """Hold-out quantile score of each plant, per farm, kept rows."""
    order = scores["plant"].unique(maintain_order=True).to_list()
    panels = []
    for label in common.FARMS:
        rows = scores.filter(pl.col("farm") == label)
        base = alt.Chart(rows).encode(y=alt.Y("plant:N", sort=order, title=None))
        bars = base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            x2="hi:Q",
            color=alt.value(GREY),
        )
        points = base.mark_point(
            filled=True, size=70, aria=False
        ).encode(  # ty: ignore[unresolved-attribute]
            x="score:Q",
            color=alt.Color(
                "plant:N",
                scale=alt.Scale(
                    domain=order,
                    range=[ocf.DATA_BLUE_LIGHT, ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE][
                        : len(order)
                    ],
                ),
                legend=None,
            ),
        )
        panels.append(
            alt.layer(bars, points).properties(
                width=PLOT_WIDTH, height=90, title=_farm_title(label)
            )
        )
    return _stack(
        panels=panels,
        title=_title(
            text=(
                "The direction term gains 0.05 to 0.13 points; masking flagged blocks changes "
                "the score by less"
            ),
            lines=[
                (
                    "Mean quantile score on the ERA5 hold-out (April 2024 to June 2025, rows "
                    "kept by experiment 210's mask), with ERA5 as the weather: an upper bound on "
                    "what a plant can score. Dot: estimate. Line: 95% interval from re-drawing "
                    "whole calendar months. Residual quantile offsets are fitted on the training "
                    "rows of each plant. "
                )
                + SCOPE
            ],
        ),
    )


def figure_8(*, shares: pl.DataFrame) -> alt.VConcatChart:
    """Share of scored days flagged and the longest flagged block, by rule."""
    rules = ["4 robust sd (strict)", "3 robust sd (main)", "2 robust sd (loose)"]
    colour = alt.Color(
        "rule:N",
        sort=rules,
        scale=alt.Scale(
            domain=rules, range=[ocf.DATA_BLUE_LIGHT, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    top = (
        alt.Chart(shares)
        .mark_bar(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("farm:N", title=None),
            yOffset=alt.YOffset("rule:N", sort=rules),
            x=alt.X("share:Q", title="Share of scored days flagged (%)"),
            color=colour,
        )
        .properties(width=PLOT_WIDTH, height=110)
    )
    bottom = (
        alt.Chart(shares)
        .mark_bar(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("farm:N", title=None),
            yOffset=alt.YOffset("rule:N", sort=rules),
            x=alt.X(
                "longest:Q",
                title="Longest flagged block (days)",
                axis=alt.Axis(tickMinStep=1, format="d"),
            ),
            color=alt.Color(
                "rule:N",
                sort=rules,
                scale=alt.Scale(
                    domain=rules, range=[ocf.DATA_BLUE_LIGHT, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]
                ),
                legend=None,
            ),
        )
        .properties(width=PLOT_WIDTH, height=110)
    )
    return _stack(
        panels=[top, bottom],
        title=_title(
            text="The rule flags 2% to 16% of days, in blocks of 7 days or fewer",
            lines=[
                (
                    "Days on which a farm made much less than the pass-1 plant expects, by the "
                    "number of robust standard deviations below the median of the daily weighted "
                    "log ratios. A block is a run of consecutive flagged days. Training days "
                    "only. "
                )
                + SCOPE
            ],
        ),
    )


def figure_9(
    *,
    payloads: dict[str, dict[str, plant.PlantParameters]],
    priors: dict[str, float],
    tops: dict[str, float],
) -> alt.VConcatChart:
    """Pass-1 and pass-2 plant curves, and their direction losses."""
    grid = np.linspace(0.0, 20.0, 201)
    panels = []
    for label in common.FARMS:
        weather = plant.Weather(
            speed_100m=grid,
            speed_10m=grid / 1.3,
            temperature_c=np.full_like(grid, 10.0),
            direction_rad=np.zeros_like(grid),
        )
        rows = []
        for name, key in (
            ("Pass 1 (all rows)", "pass1"),
            ("Pass 2 (blocks masked)", "pass2"),
        ):
            parameters = payloads[key][label]
            power = plant.expected_power(
                parameters=replace(parameters, depths=(0.0,) * plant.DIP_COUNT),
                weather=weather,
                prior_scale=priors[label],
            )
            rows.extend(
                {"speed": float(s), "power": float(p) / tops[label], "plant": name}
                for s, p in zip(grid, power, strict=True)
            )
        panels.append(
            alt.Chart(pl.DataFrame(rows))
            .mark_line(aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("speed:Q", title="ERA5 100 m wind speed (m/s)"),
                y=alt.Y("power:Q", title="Power (share of top level)"),
                color=alt.Color(
                    "plant:N",
                    scale=alt.Scale(
                        domain=["Pass 1 (all rows)", "Pass 2 (blocks masked)"],
                        range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                    ),
                    legend=alt.Legend(title=None, orient="top") if label == "W1" else None,
                ),
            )
            .properties(width=PLOT_WIDTH, height=PANEL_HEIGHT, title=_farm_title(label))
        )
    return _stack(
        panels=panels,
        title=_title(
            text="Masking flagged blocks moves the fitted curve only slightly",
            lines=[
                (
                    "Plant curves without the direction term at 10 degrees Celsius and a 10 m to "
                    "100 m speed ratio of 1.3, as a share of the Stage 1 top level. "
                )
                + SCOPE
            ],
        ),
    )


def main() -> None:
    """Draw every figure."""
    farms = common.load_farms()
    priors = common.prior_scales()
    tops = {
        label: v["top"]
        for label, v in json.loads((common.DATA_DIR / "stage1_summary.json").read_text()).items()
    }
    density = pl.read_parquet(WORK / "stage1_density.parquet")
    stats = pl.read_parquet(WORK / "stage1_bin_stats.parquet")
    direction = pl.read_parquet(WORK / "stage1_direction.parquet")
    pl.read_parquet(WORK / "stage2_curves.parquet")
    plant_ratio = pl.read_parquet(WORK / "stage3_plant_ratio.parquet")

    def load(name: str) -> dict[str, plant.PlantParameters]:
        return {
            label: plant.PlantParameters.from_dict(v)
            for label, v in common.read_json(name=name).items()
        }

    stage2, pass1, pass2 = (
        load("plant_stage2.json"),
        load("plant_stage3.json"),
        load("plant_stage3b_pass2.json"),
    )
    score_rows = []
    for label, farm in farms.items():
        for name, parameters in (
            (
                "Prior-only plant",
                plant.prior_parameters(prior_scale=priors[label], shear_density=True),
            ),
            ("Stage 2 plant", stage2[label]),
            ("Stage 3 plant (pass 1)", pass1[label]),
            ("Pass 2 plant", pass2[label]),
        ):
            ev = evaluation.evaluate_plant(
                farm=farm, parameters=parameters, prior_scale=priors[label]
            )
            interval = common.month_block_interval(
                values=ev.score[ev.kept], months=ev.month[ev.kept]
            )
            score_rows.append(
                {
                    "farm": label,
                    "plant": name,
                    "score": interval[0],
                    "lo": interval[1],
                    "hi": interval[2],
                }
            )
    share_rows = []
    for label, farm in farms.items():
        evidence = flags.build_evidence(
            farm=farm, parameters=pass1[label], prior_scale=priors[label]
        )
        series = flags.daily_series(
            farm=farm, log_ratio=evidence.log_ratio, weight=evidence.weight, rows=farm.is_train
        )
        reference = flags.reference_of(series)
        for k, name in (
            (4.0, "4 robust sd (strict)"),
            (3.0, "3 robust sd (main)"),
            (2.0, "2 robust sd (loose)"),
        ):
            days = flags.flagged_days(series=series, reference=reference, k=k)
            blocks = flags.blocks_of(days=days)
            share_rows.append(
                {
                    "farm": label,
                    "rule": name,
                    "share": days.size / series.day.size * 100,
                    "longest": max((b - a + 1 for a, b in blocks), default=0),
                }
            )
    common.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1_power_against_wind": figure_1(density=density, stats=stats),
        "figure_2_by_year": figure_2(stats=stats),
        "figure_3_by_direction_sector": figure_3(stats=stats),
        "figure_4_direction_curve": figure_4(direction=direction),
        "figure_5_prior_and_fitted_curves": figure_5(
            farms=farms, priors=priors, stage2=stage2, tops=tops
        ),
        "figure_6_direction_term": figure_6(direction=direction, plant_ratio=plant_ratio),
        "figure_7_holdout_scores": figure_7(scores=pl.DataFrame(score_rows)),
        "figure_8_flagged_share": figure_8(shares=pl.DataFrame(share_rows)),
        "figure_9_pass_1_and_pass_2_curves": figure_9(
            payloads={"pass1": pass1, "pass2": pass2}, priors=priors, tops=tops
        ),
    }
    for name, chart in figures.items():
        chart.save(common.DOCS_DIR / f"{name}.svg")
        preview = os.environ.get("E00347_PREVIEW_DIR")
        if preview:
            chart.save(f"{preview}/{name}.png", scale_factor=1.0)
        print(name)


if __name__ == "__main__":
    main()
