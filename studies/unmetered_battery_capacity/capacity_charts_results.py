"""Draw the results figures of the unmetered-battery-capacity page (1 and 10 to 18).

Each figure is built from a table that `capacity_report.py` printed into `report.md`, so the
figure and the page quote the same numbers, or from a parquet file a rung script saved. Nothing
is refitted. Charts that hold an NGED primary label it S1 to S8; the chart of NGED battery A shows
its output as multiples of its own metered 99th percentile and carries no dates.

Run after `capacity_report.py`:
`uv run python studies/unmetered_battery_capacity/capacity_charts_results.py`.
"""

from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from capacity_charts_common import (
    IN_FAMILY_COLOUR,
    NOISY_PRICE_COLOUR,
    NOMINAL_FALSE_ALARM,
    PANEL_HEIGHT_PX,
    PLOT_WIDTH_PX,
    RANK_RULE_COLOUR,
    REAL_COLOUR,
    REFERENCE_COLOUR,
    SHAPES,
    draw_figure,
    line_panel,
    report_number,
    report_table,
    save,
    write_notes,
)
from capacity_inputs import OUTPUT_DIR
from capacity_report_tools import clopper_pearson

SHARE_TITLE: Final[str] = "Battery power as a share of the series' 99th-percentile flow"
SHARES: Final[list[float]] = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
SIGMA_VALUES: Final[list[float]] = [0.125, 0.5, 2.0, 8.0, 32.0]
SIGMA_TITLE: Final[str] = "Battery power over the noise unit (bin start, powers of 2; log axis)"
FAMILIES: Final[dict[str, str]] = {
    "rung1 (in the family)": "Simulated, in the estimator's family",
    "rank_rule (outside)": "Simulated, rank rule (outside the family)",
    "noisy_price (outside)": "Simulated, noisy price (outside the family)",
    "rung3 (real public batteries)": "Real public batteries",
}
FAMILY_COLOURS: Final[list[str]] = [
    IN_FAMILY_COLOUR,
    RANK_RULE_COLOUR,
    NOISY_PRICE_COLOUR,
    REAL_COLOUR,
]
SIMULATED_NAMES: Final[dict[str, str]] = {
    "rung1 (in the family).": "Simulated, in the estimator's family",
    "rank_rule.": "Simulated, rank rule (outside the family)",
    "noisy_price.": "Simulated, noisy price (outside the family)",
}
THREE_COLOURS: Final[list[str]] = FAMILY_COLOURS[:3]
PRIMARY_SETS: Final[int] = 13
NOTES: list[str] = []


def _note(text: str) -> None:
    """Record a number derived here, so that the page quotes it from `report_figures.md`."""
    NOTES.append(text)


def headline_figure() -> alt.VConcatChart:
    """Draw figure 1: detection probability against battery power over the noise unit."""
    panels = []
    for marker, label in (
        ("**All blocks.**", "All 9 series"),
        ("**Without GSP1.**", "Without GSP1, whose null blocks are flagged (thresholds rebuilt)"),
    ):
        frame = report_table(marker=marker).with_columns(family=pl.col("family").replace(FAMILIES))
        panels.append(
            line_panel(
                frame=frame,
                x="bin_low",
                y="rate",
                group="family",
                domain=list(FAMILIES.values()),
                colours=FAMILY_COLOURS,
                x_title=SIGMA_TITLE,
                y_title="Share of batteries flagged",
                title=label,
                log_x=True,
                x_values=[0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32],
                y_domain=(0.0, 1.0),
                interval=("ci_low", "ci_high"),
                rule=NOMINAL_FALSE_ALARM,
            )
        )
    return draw_figure(
        panels=panels,
        number=1,
        title="Simulated batteries are found above a size; real public batteries mostly are not",
        subtitle=[
            (
                "Share of 3-month blocks in which the detector flags a battery, against"
                " the battery's power divided by the noise unit (the robust spread of "
                "the series' half-hour changes)."
            ),
            (
                "Band: 95% Clopper-Pearson interval. Dashed line: the 5% false-alarm "
                "level. Rung 1 simulated batteries follow the estimator's own dispatch;"
                " the other simulated families and the real batteries do not. GSP1's "
                "false alarms lift the lowest bins of the top panel."
            ),
        ],
        figure_planning=None,
    )


def positive_control_figure() -> alt.VConcatChart:
    """Draw figure 10: the clean positive control's coverage on replicas and on real demand."""
    frames = []
    for marker, kind in (
        ("**Calendar replicas: 280 fits.**", "Calendar replicas (almost no noise)"),
        ("**Real demand: 280 fits.**", "Real demand"),
    ):
        part = report_table(marker=marker)
        frames.append(
            part.select("block", "power_in_90", "energy_in_90")
            .unpivot(index="block", variable_name="quantity", value_name="coverage")
            .with_columns(
                quantity=pl.col("quantity").replace(
                    {"power_in_90": "Power (MW)", "energy_in_90": "Energy (MWh)"}
                ),
                kind=pl.lit(kind),
            )
        )
    frame = pl.concat(frames)
    panels = []
    for kind in ("Calendar replicas (almost no noise)", "Real demand"):
        base = alt.Chart(frame.filter(pl.col("kind") == kind))
        bars = base.mark_bar(aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("block:N", sort=["Sep-Nov", "Dec-Feb", "Mar-May", "Jun-Aug"], title=None),
            xOffset="quantity:N",
            y=alt.Y(
                "coverage:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="Fits whose 90% interval holds the truth",
            ),
            color=alt.Color(
                "quantity:N",
                scale=alt.Scale(
                    domain=["Power (MW)", "Energy (MWh)"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(title=None),
            ),
        )
        nominal = (
            alt.Chart(pl.DataFrame({"rule": [0.9]}))
            .mark_rule(color=REFERENCE_COLOUR, strokeDash=[4, 3], aria=False)
            .encode(y="rule:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            alt.layer(bars, nominal).properties(
                width=PLOT_WIDTH_PX,
                height=PANEL_HEIGHT_PX,
                title=alt.TitleParams(kind, anchor="start"),
            )
        )
    return draw_figure(
        panels=panels,
        number=10,
        title=(
            "The positive control passes its rule, but its intervals are narrower"
            " than 90% on replicas"
        ),
        subtitle=[
            (
                "A 40% merchant battery whose usable duration and efficiency lie off "
                "the estimator's grid, 10 fresh draws on 7 series in each block: 70 "
                "fits per bar."
            ),
            (
                "Dashed line: the nominal 90%. The rule needed 187 of 280 replica fits "
                "to hold both power and energy; 202 did. On real demand every interval "
                "holds the truth, because the likelihood is tempered by the residual's "
                "autocorrelation, which widens them."
            ),
        ],
        figure_planning=None,
    )


def calibration_figure() -> alt.VConcatChart:
    """Draw figure 12: error and interval coverage against share, in the estimator's family."""
    by_share = report_table(marker="**Rung 1 (merchant class), by share.**")
    rung2 = report_table(marker="**Rung 2 (domestic class), by share.**")
    coverage = pl.concat(
        [
            by_share.select(
                "share",
                pl.col("power_90").alias("coverage"),
                pl.lit("Merchant power, 90% interval").alias("series"),
            ),
            by_share.select(
                "share",
                pl.col("energy_90").alias("coverage"),
                pl.lit("Merchant energy, 90% interval").alias("series"),
            ),
            by_share.select(
                "share",
                pl.col("power_50").alias("coverage"),
                pl.lit("Merchant power, 50% interval").alias("series"),
            ),
            rung2.select(
                "share",
                pl.col("power_90").alias("coverage"),
                pl.lit("Domestic fleet power, 90% interval").alias("series"),
            ),
        ]
    )
    domain = [
        "Merchant power, 90% interval",
        "Merchant energy, 90% interval",
        "Merchant power, 50% interval",
        "Domestic fleet power, 90% interval",
    ]
    error = report_table(marker="**rung1 (in the family).**", occurrence=0).select(
        "share",
        pl.col("median_abs_power_error").alias("error"),
        pl.lit("Median absolute power error").alias("series"),
    )
    error = pl.concat(
        [
            error,
            report_table(marker="**rung1 (in the family).**", occurrence=0).select(
                "share",
                pl.col("median_abs_energy_error").alias("error"),
                pl.lit("Median absolute energy error").alias("series"),
            ),
        ]
    )
    return draw_figure(
        panels=[
            line_panel(
                frame=error,
                x="share",
                y="error",
                group="series",
                domain=["Median absolute power error", "Median absolute energy error"],
                colours=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                x_title=SHARE_TITLE + " (log axis)",
                y_title="Relative error (1 = 100% of the truth; smaller is better)",
                title="Error of the posterior median, simulated merchant batteries (rung 1)",
                log_x=True,
                x_values=SHARES,
                log_y=True,
            ),
            line_panel(
                frame=coverage,
                x="share",
                y="coverage",
                group="series",
                domain=domain,
                colours=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_SKY, ocf.DATA_PURPLE],
                x_title=SHARE_TITLE + " (log axis)",
                y_title="Share of sums whose interval holds the truth",
                title="Coverage of the credible intervals (planned contrast C2)",
                log_x=True,
                x_values=SHARES,
                y_domain=(0.0, 1.0),
                rule=0.9,
            ),
        ],
        number=12,
        title=(
            "The 90% interval for a simulated merchant battery holds the truth "
            "from a 5% share; a domestic fleet's does not"
        ),
        subtitle=[
            (
                "756 rung 1 sums (9 series, 4 blocks, 3 durations) and 252 rung 2 sums."
                " Dashed line: the nominal 90%; the 50% interval should sit at 0.5."
            ),
            (
                "Below a 2% share the posterior is the same as with no battery, so the "
                "coverage and the error describe the prior, not the data."
            ),
        ],
        figure_planning=None,
    )


def false_alarm_figure() -> alt.VConcatChart:
    """Draw figure 11: the log Bayes factor of every block with no added battery."""
    frame = report_table(marker="Log Bayes factor of every null block:").with_columns(
        block_name=pl.col("block").replace_strict(
            {0: "Sep-Nov", 1: "Dec-Feb", 2: "Mar-May", 3: "Jun-Aug"}, return_dtype=pl.String
        ),
        outcome=pl.when(pl.col("flagged")).then(pl.lit("Flagged")).otherwise(pl.lit("Not flagged")),
    )
    base = alt.Chart(frame)
    order = sorted(frame["series"].unique().to_list())
    points = base.mark_point(filled=True, size=70, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("series:N", sort=order, title="Demand-like series (S1 to S8 are NGED primaries)"),
        y=alt.Y("log_bayes_factor:Q", title="Log Bayes factor (battery model against none)"),
        color=alt.Color(
            "outcome:N",
            scale=alt.Scale(
                domain=["Not flagged", "Flagged"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
            ),
            legend=alt.Legend(title=None),
        ),
        shape=alt.Shape(
            "block_name:N",
            scale=alt.Scale(
                domain=["Sep-Nov", "Dec-Feb", "Mar-May", "Jun-Aug"], range=list(SHAPES)
            ),
            legend=alt.Legend(title="Block"),
        ),
    )
    ticks = base.mark_tick(color=REFERENCE_COLOUR, thickness=2, size=34, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("series:N", sort=order), y="threshold:Q"
    )
    panel = alt.layer(points, ticks).properties(
        width=PLOT_WIDTH_PX,
        height=PANEL_HEIGHT_PX * 1.6,
        title=alt.TitleParams("Four blocks per series, no battery added", anchor="start"),
    )
    flagged_blocks = frame.filter(pl.col("flagged"))
    flagged_series = sorted(flagged_blocks["series"].unique().to_list())
    low, high = clopper_pearson(count=flagged_blocks.height, total=frame.height)
    return draw_figure(
        panels=[panel],
        number=11,
        title=(
            f"With no battery added, {flagged_blocks.height} of {frame.height} blocks are "
            f"flagged, all of them {' and '.join(flagged_series)}"
        ),
        subtitle=[
            (
                "Planned contrast C1. Dot: one 3-month block of a series with no added "
                "battery. Black tick: that series' threshold, the 95th percentile over "
                "the other 8 series."
            ),
            (
                "Flagged: the log Bayes factor lies above the threshold. The realised "
                f"rate is {flagged_blocks.height / frame.height:.1%}, 95% interval "
                f"{low:.1%} to {high:.1%}, against a nominal 5%."
            ),
        ],
        figure_planning=None,
    )


def outside_family_figure() -> alt.VConcatChart:
    """Draw figure 13: detection, coverage, and error for batteries from outside the family."""
    tables = {
        name: report_table(marker=f"**{marker}", occurrence=0)
        for marker, name in (
            ("rung1 (in the family).**", "Simulated, in the estimator's family"),
            ("rank_rule.**", "Simulated, rank rule (outside the family)"),
            ("noisy_price.**", "Simulated, noisy price (outside the family)"),
        )
    }
    frame = pl.concat(
        [t.with_columns(family=pl.lit(name)) for name, t in tables.items()], how="vertical"
    )
    domain = list(tables)
    panels = []
    for column, y_title, title, log_y in (
        ("rate", "Share of sums flagged", "Detection at the 5% false-alarm threshold", False),
        ("power_90", "Share holding the truth", "Coverage of the 90% power interval", False),
        (
            "median_abs_power_error",
            "Relative error (smaller is better)",
            "Median absolute error of the power estimate",
            True,
        ),
    ):
        panels.append(
            line_panel(
                frame=frame,
                x="share",
                y=column,
                group="family",
                domain=domain,
                colours=THREE_COLOURS,
                x_title=SHARE_TITLE + " (log axis)",
                y_title=y_title,
                title=title,
                log_x=True,
                x_values=SHARES,
                y_domain=None if log_y else (0.0, 1.0),
                rule=0.9 if column == "power_90" else None,
                log_y=log_y,
            )
        )
    return draw_figure(
        panels=panels,
        number=13,
        title=(
            "Outside the estimator's family, large batteries are flagged less often "
            "and sized wrongly"
        ),
        subtitle=[
            (
                "Each family: 9 series, 4 blocks, and 3 durations, so 108 sums per "
                "share. The rank rule charges in each day's cheapest half-hours; the "
                "noisy price adds independent noise to the price before the dispatch."
            ),
            (
                "Exploratory. Dashed line: the nominal 90%. At a 40% share the rank "
                "rule is flagged in 100% of sums, yet its 90% power interval holds the "
                "truth in 26%."
            ),
        ],
        figure_planning=None,
    )


def fleet_figure() -> alt.VConcatChart:
    """Draw figure 14: simulated domestic fleets, detection, and what is identified."""
    real = report_table(marker="**Rung 2, simulated fleets, real windows.**").with_columns(
        series=pl.lit("Real windows and prices")
    )
    early = report_table(
        marker="**Rung 2, simulated fleets, windows one hour early (negative control).**"
    ).with_columns(series=pl.lit("Windows one hour early (prices real)"))
    both = report_table(
        marker="**Coarse stacks, windows one hour early and prices 7 days later.**"
    ).with_columns(series=pl.lit("Windows one hour early and prices 7 days later"))
    detection = pl.concat([real, early, both])
    domain = [
        "Real windows and prices",
        "Windows one hour early (prices real)",
        "Windows one hour early and prices 7 days later",
    ]
    agile = report_table(marker="Coverage of the Agile unit's 90% interval").with_columns(
        series=pl.lit("Agile unit's power")
    )
    return draw_figure(
        panels=[
            line_panel(
                frame=detection,
                x="share",
                y="rate",
                group="series",
                domain=domain,
                colours=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                x_title="Fleet power as a share of the series' 99th-percentile flow (log axis)",
                y_title="Share of sums flagged",
                title="Detection of a simulated domestic fleet",
                log_x=True,
                x_values=SHARES,
                y_domain=(0.0, 1.0),
                rule=NOMINAL_FALSE_ALARM,
            ),
            line_panel(
                frame=agile,
                x="share",
                y="covered",
                group="series",
                domain=["Agile unit's power"],
                colours=[ocf.DATA_GREEN],
                x_title="Fleet power as a share of the series' 99th-percentile flow (log axis)",
                y_title="Share whose 90% interval holds the truth",
                title="Coverage of the Agile unit's power interval",
                log_x=True,
                x_values=SHARES,
                y_domain=(0.0, 1.0),
                rule=0.9,
            ),
        ],
        number=14,
        title=(
            "A simulated domestic fleet is found only through its Agile homes, "
            "and not below a 20% share"
        ),
        subtitle=[
            (
                "Exploratory, simulated: 36 sums per share (9 series, 4 blocks), fleets"
                " built home by home on the four tariffs. Dashed lines: 5% false alarms"
                " and the nominal 90%."
            ),
            (
                "Moving the windows alone lowers the detections only a little, because the"
                " Agile unit follows a price; moving the prices as well brings flags to"
                " 2 of 36 at every share."
            ),
        ],
        figure_planning=None,
    )


def real_battery_figure() -> alt.VConcatChart:
    """Draw figure 15: real public batteries added to real demand and to calendar replicas."""
    real = report_table(
        marker="**The posterior barely moves when a real public battery is added.**"
    ).select(
        "share",
        pl.col("median_power_over_p99").alias("value"),
        pl.lit("Real demand: posterior median").alias("series"),
    )
    replica = (
        report_table(marker="The same 23 real public batteries")
        .filter(pl.col("share") > 0)
        .select(
            "share",
            pl.col("median_power_over_p99").alias("value"),
            pl.lit("Calendar replica: posterior median").alias("series"),
        )
    )
    truth = pl.DataFrame(
        {"share": [0.05, 0.1, 0.2, 0.4], "value": [0.05, 0.1, 0.2, 0.4], "series": "Truth"}
    )
    power = pl.concat([real, replica.filter(pl.col("share") > 0.05), truth], how="vertical")
    flags = pl.concat(
        [
            report_table(marker="**Rung 3, real public batteries, by share.**").with_columns(
                series=pl.lit("Real demand, all 9 series")
            ),
            report_table(marker="The same rates without GSP1").with_columns(
                series=pl.lit("Real demand, without GSP1")
            ),
            report_table(marker="**Flagged by a threshold from the replicas' own nulls.**")
            .with_columns(series=pl.lit("Calendar replicas"))
            .select("share", "flagged", "total", "rate", "ci_low", "ci_high", "series"),
        ]
    )
    domain = ["Truth", "Real demand: posterior median", "Calendar replica: posterior median"]
    return draw_figure(
        panels=[
            line_panel(
                frame=power,
                x="share",
                y="value",
                group="series",
                domain=domain,
                colours=[REFERENCE_COLOUR, ocf.DATA_BLUE, ocf.DATA_GREEN],
                x_title=(
                    "True battery power as a share of the series' 99th-percentile flow (log axis)"
                ),
                y_title="Estimated power over the 99th percentile",
                title="The estimated power barely rises with the real battery's power",
                log_x=True,
                x_values=[0.05, 0.1, 0.2, 0.4],
                log_y=True,
            ),
            line_panel(
                frame=flags,
                x="share",
                y="rate",
                group="series",
                domain=[
                    "Real demand, all 9 series",
                    "Real demand, without GSP1",
                    "Calendar replicas",
                ],
                colours=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN],
                x_title=(
                    "True battery power as a share of the series' 99th-percentile flow (log axis)"
                ),
                y_title="Share of sums flagged",
                title="Flagged mostly only at a 40% share, and on replicas",
                log_x=True,
                x_values=[0.05, 0.1, 0.2, 0.4],
                y_domain=(0.0, 1.0),
                interval=("ci_low", "ci_high"),
                rule=NOMINAL_FALSE_ALARM,
            ),
        ],
        number=15,
        title="Real public batteries added to NGED demand are mostly not recovered",
        subtitle=[
            (
                "Exploratory. 23 real batteries and fleets of 2, 4, and 8 batteries, "
                "each added to 9 series in 4 blocks: 828 sums per share. The truth is "
                "the registered power."
            ),
            (
                "On calendar replicas (almost no demand noise, thresholds from their own "
                "no-battery blocks) the real batteries are flagged, but the posterior "
                "median is still about 6% of the registered power."
            ),
        ],
        figure_planning=None,
    )


def battery_a_figure() -> alt.VConcatChart:
    """Draw figure 16: NGED battery A inside a bulk supply point's flow, by multiple."""
    frame = pl.read_parquet(OUTPUT_DIR / "rung4_posteriors.parquet")
    metered_p99 = float(frame.filter(pl.col("multiple") == 1)["true_power_mw"][0])
    names = {0: "Sep-Nov", 1: "Dec-Feb", 2: "Mar-May", 3: "Jun-Aug"}
    threshold = report_number(pattern=r"Detection threshold: (-?\d+\.\d+)")
    frame = frame.with_columns(
        block_name=pl.col("block").replace_strict(names, return_dtype=pl.String),
        multiple_f=pl.col("multiple").cast(pl.Float64),
        median=pl.col("merchant_power_median") / metered_p99,
        low=pl.col("merchant_power_q05") / metered_p99,
        high=pl.col("merchant_power_q95") / metered_p99,
        truth=pl.col("multiple").cast(pl.Float64),
    )
    _note(
        "Rung 4: the posterior median of the merchant power over the metered 99th percentile runs"
        f" from {frame['median'].min():.2f} to {frame['median'].max():.2f}, against a truth of"
        f" 0 to {frame['multiple'].max()} times."
    )
    domain = list(names.values())
    colours = [ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_GREEN, ocf.BRAND_ORANGE]
    bf = line_panel(
        frame=frame.select("multiple_f", "log_bayes_factor", "block_name"),
        x="multiple_f",
        y="log_bayes_factor",
        group="block_name",
        domain=domain,
        colours=colours,
        x_title="Further copies of NGED battery A subtracted (0 = its metered output added back)",
        y_title="Log Bayes factor",
        title="Detection statistic",
        x_values=[0, 1, 2, 4, 11],
        rule=threshold,
    )
    power = line_panel(
        frame=frame.select("multiple_f", "median", "low", "high", "block_name"),
        x="multiple_f",
        y="median",
        group="block_name",
        domain=domain,
        colours=colours,
        x_title="Further copies of NGED battery A subtracted (0 = its metered output added back)",
        y_title="Power over metered 99th percentile",
        title="Estimated merchant power (band: 90% interval)",
        x_values=[0, 1, 2, 4, 11],
        interval=("low", "high"),
    )
    truth_line = (
        alt.Chart(frame.select("multiple_f", "truth").unique())
        .mark_line(color=REFERENCE_COLOUR, strokeDash=[4, 3], aria=False)
        .encode(x="multiple_f:Q", y="truth:Q")  # ty: ignore[unresolved-attribute]
    )
    return draw_figure(
        panels=[bf, alt.layer(power, truth_line)],
        number=16,
        title=(
            "NGED battery A is not detected inside its bulk supply point, even at"
            " 11 times its output"
        ),
        subtitle=[
            (
                "Exploratory, one site. Four 3-month blocks of one bulk supply point's "
                "flow. Dashed line in the upper panel: the detection threshold."
            ),
            (
                "Dashed line in the lower panel: the true added power. No block is "
                "flagged at any multiple, including the matched null (0 copies)."
            ),
        ],
        figure_planning=None,
    )


def screen_figure() -> alt.VConcatChart:
    """Draw figure 17: the primary screen's ranks and its power."""
    ranks = report_table(marker="Each primary's log Bayes factor summed over the four blocks")
    labelled = ranks.with_columns(
        label=pl.col("series")
        + pl.when(pl.col("register_connected_storage") > 0)
        .then(pl.lit(" (register: storage connected)"))
        .when(pl.col("register_accepted_storage") > 0)
        .then(pl.lit(" (register: storage accepted)"))
        .otherwise(pl.lit(" (register: no storage)"))
    )
    order = labelled["label"].to_list()
    _note(
        "Per-primary screen table (series | rank of 13 | rank among the 7 "
        "price-differing sets | register):"
    )
    for r in ranks.iter_rows(named=True):
        register = (
            "connected storage"
            if r["register_connected_storage"]
            else "accepted storage"
            if r["register_accepted_storage"]
            else "no storage"
        )
        _note(
            f"| {r['series']} | {r['real_rank']} | {r['real_rank_among_price_placebos']} |"
            f" {register} |"
        )
    rank_chart = alt.layer(
        alt.Chart(labelled)
        .mark_bar(color=ocf.DATA_BLUE, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
            x=alt.X(
                "real_rank:Q",
                scale=alt.Scale(domain=[0, PRIMARY_SETS]),
                title="Rank of the real template set among 13 (1 = best; 7 is the middle)",
            ),
        ),
        alt.Chart(pl.DataFrame({"x": [1.0]}))
        .mark_rule(color=ocf.BRAND_ORANGE, strokeWidth=2, aria=False)
        .encode(x="x:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(
        width=PLOT_WIDTH_PX - 200,
        height=PANEL_HEIGHT_PX * 1.5,
        title=alt.TitleParams("Each primary: no primary ranks first", anchor="start"),
    )
    power = (
        report_table(marker="**With no battery added, the real set ranks first of 13")
        .with_columns(
            lane=pl.when(pl.col("lane_kind") == "null")
            .then(pl.lit("No battery (9 lanes)"))
            .when(pl.col("lane_kind") == "merchant")
            .then(
                pl.lit("Simulated merchant, ")
                + (pl.col("share") * 100).cast(pl.Int64).cast(pl.String)
                + pl.lit("% (27 lanes)")
            )
            .otherwise(pl.lit("Real public batteries, 40% (36 lanes)"))
        )
        .select(
            "lane",
            pl.col("rate_first_of_13").alias("All 13 template sets"),
            pl.col("rate_first_among_price_placebos").alias(
                "Real set and its 6 price-shifted rivals"
            ),
        )
        .unpivot(index="lane", variable_name="screen", value_name="rate")
    )
    lanes = power["lane"].unique(maintain_order=True).to_list()
    power_chart = (
        alt.Chart(power)
        .mark_bar(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("lane:N", sort=lanes, title=None, axis=alt.Axis(labelLimit=300)),
            yOffset="screen:N",
            x=alt.X(
                "rate:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="Share of lanes where the real set ranks first",
            ),
            color=alt.Color(
                "screen:N",
                scale=alt.Scale(
                    domain=["All 13 template sets", "Real set and its 6 price-shifted rivals"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, labelLimit=400),
            ),
        )
        .properties(
            width=PLOT_WIDTH_PX - 200,
            height=PANEL_HEIGHT_PX * 1.5,
            title=alt.TitleParams(
                "The screen's power, measured on lanes with a known answer", anchor="start"
            ),
        )
    )
    return draw_figure(
        panels=[rank_chart, power_chart],
        number=17,
        title=(
            "The 13-set primary screen cannot find a battery, and by the "
            "price-only screen no primary stands out"
        ),
        subtitle=[
            (
                "Exploratory. Each lane sums its log Bayes factor over four blocks and "
                "compares the real templates with 12 placebos (tariff windows shifted "
                "by 1 to 3 hours either way, and prices from 6 other weeks)."
            ),
            (
                "Orange rule: rank 1. The register column says whether NGED's Embedded "
                "Capacity Register lists connected storage of 50 kW or more at the "
                "primary; it lists no MWh."
            ),
        ],
        figure_planning=None,
    )


def grid_figure() -> alt.VConcatChart:
    """Draw figure 18: the grid estimator beside the differentiable estimator."""
    table = report_table(marker="The grid estimator ran on all 756 rung 1 sums")
    frame = pl.concat(
        [
            table.select(
                "share",
                pl.col("dp_power_90").alias("coverage"),
                pl.col("dp_median_absolute_power_error").alias("error"),
                pl.lit("Differentiable estimator").alias("estimator"),
            ),
            table.select(
                "share",
                pl.col("grid_power_90").alias("coverage"),
                pl.col("grid_median_absolute_power_error").alias("error"),
                pl.lit("Grid estimator").alias("estimator"),
            ),
        ]
    )
    domain = ["Differentiable estimator", "Grid estimator"]
    colours = [ocf.DATA_BLUE, ocf.BRAND_ORANGE]
    return draw_figure(
        panels=[
            line_panel(
                frame=frame,
                x="share",
                y="coverage",
                group="estimator",
                domain=domain,
                colours=colours,
                x_title=SHARE_TITLE + " (log axis)",
                y_title="Share holding the truth",
                title="Coverage of the 90% power interval",
                log_x=True,
                x_values=SHARES,
                y_domain=(0.0, 1.0),
                rule=0.9,
            ),
            line_panel(
                frame=frame,
                x="share",
                y="error",
                group="estimator",
                domain=domain,
                colours=colours,
                x_title=SHARE_TITLE + " (log axis)",
                y_title="Relative error (smaller is better)",
                title="Median absolute error of the power estimate",
                log_x=True,
                x_values=SHARES,
                log_y=True,
            ),
        ],
        number=18,
        title=(
            "At a 40% share the grid estimator's 90% interval holds the truth in "
            "11% of sums; the differentiable estimator's in 99%"
        ),
        subtitle=[
            (
                "Exploratory. The same 756 rung 1 sums (9 series, 4 blocks, 3 "
                "durations). The grid estimator places the duration and efficiency on a"
                " fixed grid of 1,728 points."
            ),
            (
                "Dashed line: the nominal 90%. At a 40% share the grid estimator's "
                "interval holds the truth in 11% of sums."
            ),
        ],
        figure_planning=None,
    )


def independent(chart: alt.VConcatChart) -> alt.VConcatChart:
    """Give each panel of a figure its own colour and shape scales and legend."""
    return chart.resolve_scale(color="independent", shape="independent")


def main() -> None:
    """Write every results figure and the numbers derived while drawing them."""
    figures = {
        "headline": headline_figure,
        "positive_control": positive_control_figure,
        "calibration": calibration_figure,
        "false_alarms": false_alarm_figure,
        "outside_family": outside_family_figure,
        "fleets": fleet_figure,
        "real_batteries": real_battery_figure,
        "battery_a": battery_a_figure,
        "screen": screen_figure,
        "grid": grid_figure,
    }
    for name, draw in figures.items():
        print(f"Wrote {save(chart=independent(draw()), name=name)}")
    write_notes(name="results", notes=NOTES)


if __name__ == "__main__":
    main()
