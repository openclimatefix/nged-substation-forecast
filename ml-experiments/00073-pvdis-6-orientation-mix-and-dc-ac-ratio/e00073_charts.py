"""Draw experiment 00073's figures from the saved results, as SVG (raw data first).

Solar farms carry the labels A to F and no chart pairs a label with a name or an identifier;
substations are named. The SVGs are afterwards optimised with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00073-pvdis-6-orientation-mix-and-dc-ac-ratio/e00073_charts.py
"""

import textwrap
from typing import Final

import altair as alt
import e00073_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

WIDTH: Final[int] = 560
WRAP: Final[int] = 88
EAST_SOUTH_WEST: Final[alt.Scale] = alt.Scale(
    domain=["East", "South", "West"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]
)
FARM_COLOURS: Final[alt.Scale] = alt.Scale(
    domain=list(common.FARMS),
    range=[
        ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_PURPLE, ocf.DATA_GREEN, ocf.BRAND_ORANGE,
        ocf.BRAND_ORANGE_LIGHT,
    ],
)  # fmt: skip
MIX_LABELS: Final[list[str]] = [
    "all south", "fleet mix (0.2/0.6/0.2)", "east-heavy (0.5/0.4/0.1)",
    "west-heavy (0.1/0.4/0.5)", "east and west (0.5/0/0.5)",
]  # fmt: skip


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def raw_clipping() -> alt.Chart:
    """Figure 1: clear-day farm output against plane-of-array irradiance, binned medians."""
    frame = pl.read_parquet(common.FRAME_PATH)
    shapes = pl.read_parquet(common.SHAPES_PATH)
    rows = []
    for farm in common.FARMS:
        f = frame.filter(
            (pl.col("site") == farm)
            & pl.col("power").is_not_null()
            & (pl.col("zenith") < common.MAX_ZENITH_DEG)
        ).with_columns(
            cloud=1.0
            - (pl.col("ghi_u") / pl.col("ghi_u_clear")).clip(upper_bound=1.0).fill_nan(0.0),
            date=pl.col("time").dt.date(),
        )
        f = f.filter(~((pl.col("power") <= 0.0) & (pl.col("ghi_u") > common.DROPOUT_GHI_W_M2)))
        f = f.with_columns(day_cloud=pl.col("cloud").mean().over("date"))
        f = f.filter(pl.col("day_cloud") < common.CLEAR_CLOUD_MAX).join(
            shapes.filter(pl.col("site") == farm).select("time", "gS25"), on="time"
        )
        p99 = float(
            np.nanpercentile(frame.filter(pl.col("site") == farm)["power"].drop_nulls(), 99)
        )
        f = f.with_columns(
            x=(pl.col("gS25") / 0.05).floor() * 0.05 + 0.025, y=pl.col("power") / p99
        )
        rows.append(
            f.filter(pl.col("gS25") > 0.3)
            .group_by("x")
            .agg(pl.col("y").median(), pl.len().alias("n"))
            .filter(pl.col("n") >= 50).with_columns(farm=pl.lit(f"Solar farm {farm}"))
        )  # fmt: skip
    data = pl.concat(rows).to_pandas()
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, clip=True)
        .encode(
            x=alt.X("x:Q", title="Plane-of-array irradiance, south-facing 25 degrees (kW/m²)"),
            y=alt.Y("y:Q", title="Median output (share of the farm's 99th percentile)"),
            color=alt.Color(
                "farm:N",
                scale=alt.Scale(
                    domain=[f"Solar farm {f}" for f in common.FARMS], range=FARM_COLOURS.range
                ),
                legend=alt.Legend(orient="top", columns=3, title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=240,
            title=_title(
                text="Output flattens at high irradiance on clear days",
                lines=[
                    (
                        "Median metered output of each solar farm in 0.05 kW/m² bins of "
                        "irradiance, "
                        "on clear days (daylight half-hours only; meter dropouts removed; bins "
                        "with "
                        "under 50 half-hours left out). A flattening is the sign of clipping."
                    )
                ],
            ),
        )
    )


def shapes_one_day() -> alt.Chart:
    """Figure 2: the east, south, and west unit shapes through one clear day at one Primary."""
    shapes = pl.read_parquet(common.SHAPES_PATH).filter(pl.col("site") == "D14")
    day = shapes.filter(pl.col("time").dt.date() == pl.date(2024, 6, 21)).with_columns(
        hour=pl.col("time").dt.convert_time_zone("Europe/London").dt.hour() + 0.0
    )
    rows = []
    for letter, label in (("E", "East"), ("S", "South"), ("W", "West")):
        g = day[f"g{letter}25"].to_numpy()
        hour = day["time"].dt.convert_time_zone("Europe/London").dt.hour().to_numpy() + (
            day["time"].dt.convert_time_zone("Europe/London").dt.minute().to_numpy() / 60.0
        )
        rows.append(pl.DataFrame({"hour": hour, "output": np.minimum(1.15 * g, 1.0), "o": label}))
    data = pl.concat(rows).to_pandas()
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(clip=True)
        .encode(
            x=alt.X(
                "hour:Q",
                title="Local hour of day",
                scale=alt.Scale(domain=[4, 20]),
                axis=alt.Axis(values=[4, 8, 12, 16, 20]),
            ),
            y=alt.Y("output:Q", title="AC output per MW of AC capacity"),
            color=alt.Color(
                "o:N", scale=EAST_SOUTH_WEST, legend=alt.Legend(orient="top", title=None)
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="East, south, and west panels differ only in the time of day of their output",
                lines=[
                    (
                        "Modelled output of 1 MW of panels (25 degrees tilt, DC:AC ratio 1.15) at "
                        "Mount Bridge Boston on one clear June day. Daylight hours only."
                    )
                ],
            ),
        )
    )


def known_orientation() -> alt.Chart:
    """Figure 3: error of recovered shares by true mix, arm against the two guesses."""
    table = pl.read_parquet(common.KNOWN_ORIENTATION_PATH)
    units = table.group_by("site", "farm", "tilt", "mix_id", "arm").agg(pl.col("tv").mean())
    scoring = units.filter(
        (pl.col("tilt") == common.MAIN_TILT) & pl.col("farm").is_in(common.SCORING_FARMS)
    )
    rows = []
    noisy = pl.read_parquet(common.WORK_DIR / "known_noisy.parquet").filter(
        (pl.col("tilt") == common.MAIN_TILT)
        & pl.col("farm").is_in(common.SCORING_FARMS)
        & (pl.col("arm") == "O3_0.0")
    )
    for source, label in (
        (scoring.filter(pl.col("arm") == "O3_0.0"), "Pooled fit, injection only"),
        (noisy, "Pooled fit, with noise"),
    ):
        part = source.group_by("mix_id").agg(pl.col("tv").median())
        rows += [(m, label, v) for m, v in part.iter_rows()]
    for m, mix in enumerate(common.MIXES):
        rows.append(
            (m, "Guess: the fleet mix", 0.5 * np.abs(np.array(common.FLEET_MIX) - mix).sum())
        )
        rows.append((m, "Guess: all south", 0.5 * np.abs(np.array([0, 1, 0]) - mix).sum()))
    data = pl.DataFrame(rows, schema=["mix", "arm", "error"], orient="row").with_columns(
        label=pl.col("mix").map_elements(lambda m: MIX_LABELS[m], return_dtype=pl.String)
    )
    return (
        alt.Chart(data.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=90, opacity=1.0)
        .encode(
            y=alt.Y("label:N", sort=MIX_LABELS, title=None),
            x=alt.X(
                "error:Q", title="Error of the shares (0 = exact, 1 = worst; smaller is better)"
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=[
                        "Pooled fit, injection only",
                        "Pooled fit, with noise",
                        "Guess: the fleet mix",
                        "Guess: all south",
                    ],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_BLUE_LIGHT, "#999999"],
                ),
                legend=alt.Legend(orient="top", columns=2, title=None),
            ),
            shape=alt.Shape("arm:N", legend=None),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="With noise added, the pooled fit still beats a guess but misses by about 0.2",
                lines=[
                    (
                        "Made-up output added to the real power of 8 Primary substations. Dot: "
                        "median "
                        "over scoring units (solar-farm locations D to F), the error of the "
                        "recovered shares (total-variation distance)."
                    )
                ],
            ),
        )
    )


def known_dcac() -> alt.Chart:
    """Figure 4: fitted DC:AC ratio against the true ratio, farm alone and injected."""
    table = pl.read_parquet(common.KNOWN_DCAC_PATH).filter(pl.col("case") != "base")
    data = (
        table.group_by("case", "farm")
        .agg(pl.col("truth").first(), pl.col("fitted").median())
        .with_columns(
            case=pl.when(pl.col("case") == "alone")
            .then(pl.lit("Farm output alone"))
            .otherwise(pl.lit("10 MW added to a Primary"))
        )
        .to_pandas()
    )
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [1.0, 1.6], "y": [1.0, 1.6]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(color="#999999", strokeDash=[4, 3])
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[1.0, 1.6])),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[1.0, 1.6])),
        )
    )
    points = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=90, opacity=1.0)
        .encode(
            x=alt.X(
                "truth:Q",
                title="True DC:AC ratio (experiment 29's fit)",
                scale=alt.Scale(domain=[1.0, 1.6]),
            ),
            y=alt.Y(
                "fitted:Q",
                title="Searched ratio (median over season-years)",
                scale=alt.Scale(domain=[1.0, 1.6]),
            ),
            color=alt.Color(
                "case:N",
                scale=alt.Scale(
                    domain=["Farm output alone", "10 MW added to a Primary"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            shape=alt.Shape("case:N", legend=None),
        )
    )
    return (diagonal + points).properties(
        width=WIDTH,
        height=260,
        title=_title(
            text="The DC:AC ratio is found from a farm's own output but not once it sits in demand",
            lines=[
                (
                    "Each dot is one solar farm (A to F). On the dashed line the search returns "
                    "the "
                    "true ratio. Grid 1.00 to 1.60 in steps of 0.05."
                )
            ],
        ),
    )


def real_shares() -> alt.Chart:
    """Figure 5: the free fit's shares for each Primary and season-year (they swap)."""
    real = (
        pl.read_parquet(common.REAL_PATH)
        .filter(pl.col("arm") == "O1")
        .drop_nulls(["east", "south", "west"])
    )
    long = real.unpivot(
        index=["site", "year"], on=["east", "south", "west"], variable_name="o", value_name="share"
    ).with_columns(
        o=pl.col("o").str.to_titlecase(),
        name=pl.col("site").replace(common.PRIMARY_NAMES),
        year=pl.col("year").cast(pl.String),
    )
    return (
        alt.Chart(long.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Season-year starting in September"),
            y=alt.Y("share:Q", title="Share", stack="normalize", axis=alt.Axis(values=[0, 0.5, 1])),
            color=alt.Color(
                "o:N", scale=EAST_SOUTH_WEST, legend=alt.Legend(orient="top", title=None)
            ),
            row=alt.Row(
                "name:N",
                title=None,
                sort=[common.PRIMARY_NAMES[p] for p in common.DIRECTIONAL_PRIMARIES],
                header=alt.Header(labelAngle=0, labelAlign="left"),
            ),
        )
        .properties(
            width=WIDTH - 140,
            height=60,
            title=_title(
                text="On the real series the fitted mix swaps from one season-year to the next",
                lines=[
                    (
                        "East, south, and west shares of the free fit to each Primary's April to "
                        "September power. Shares are of the fitted solar capacity."
                    )
                ],
            ),
        )
    )


def pooled_shares() -> alt.Chart:
    """Figure 6: the pooled fit's shares with 95% intervals, per Primary."""
    pooled = pl.read_parquet(common.REAL_PATH).filter(pl.col("arm") == "O3_0.0")
    rows = [
        {
            "name": common.PRIMARY_NAMES[row["site"]],
            "o": o.title(),
            "share": row[o],
            "low": row[f"{o}_low"],
            "high": row[f"{o}_high"],
        }
        for row in pooled.iter_rows(named=True)
        for o in ("east", "south", "west")
    ]
    data = pl.DataFrame(rows).to_pandas()
    order = [common.PRIMARY_NAMES[p] for p in common.DIRECTIONAL_PRIMARIES]
    base = alt.Chart(data).encode(
        y=alt.Y("name:N", sort=order, title=None),
        yOffset=alt.YOffset("o:N", sort=["East", "South", "West"]),
        color=alt.Color("o:N", scale=EAST_SOUTH_WEST, legend=alt.Legend(orient="top", title=None)),
    )
    return (
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low:Q", title="Share of the fitted solar capacity", scale=alt.Scale(domain=[0, 1])
            ),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=60, opacity=1.0).encode(x="share:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH - 120,
        height=300,
        title=_title(
            text="Pooled over season-years, the shares still carry wide intervals",
            lines=[
                (
                    "Dot: one mix per Primary fitted on all its clear days. Line: 95% interval "
                    "from "
                    "200 resamples of whole 7-day blocks. Kirton is flagged in experiment 40 and "
                    "Wrangle T1 has little power."
                )
            ],
        ),
    )


def heldout() -> alt.Chart:
    """Figure 7: held-out change in error, fitted mix against the south-only shape."""
    held = pl.read_parquet(common.HELDOUT_PATH)
    data = (
        held.group_by("site").agg(pl.col("change").mean())
        .with_columns(name=pl.col("site").replace(common.PRIMARY_NAMES))
        .to_pandas()
    )  # fmt: skip
    order = [common.PRIMARY_NAMES[p] for p in common.DIRECTIONAL_PRIMARIES]
    zero = (
        alt.Chart(pl.DataFrame({"v": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#999999", strokeDash=[4, 3])
        .encode(x="v:Q")
    )
    dots = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=90, opacity=1.0, color=ocf.DATA_BLUE)
        .encode(
            y=alt.Y("name:N", sort=order, title=None),
            x=alt.X(
                "change:Q",
                title="Change in held-out error (0 = same as south-only; negative is better)",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[-0.05, 0.01]),
            ),
        )
    )
    return (zero + dots).properties(
        width=WIDTH - 140,
        height=220,
        title=_title(
            text="The fitted mix predicts held-out weeks better than all-south panels",
            lines=[
                (
                    "Dot: relative change in root mean square error of daylight power, pooled mix "
                    "against south-only, averaged over season-years and over the two fold "
                    "directions "
                    "(fit on alternate 28-day blocks, score the others)."
                )
            ],
        ),
    )


def main() -> None:
    """Write every figure."""
    common.CHART_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_01": raw_clipping(),
        "figure_02": shapes_one_day(),
        "figure_03": known_orientation(),
        "figure_04": known_dcac(), "figure_05": real_shares(), "figure_06": pooled_shares(),
        "figure_07": heldout(),
    }  # fmt: skip
    for name, chart in figures.items():
        chart.save(common.CHART_DIR / f"{name}.svg")
        print("wrote", name)


if __name__ == "__main__":
    main()
