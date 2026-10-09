"""Report sections added in answer to the first science review.

These sections print the out-of-family rungs, the clean positive control, the screen's power, the
detection-power curve, the extra negative controls, and the notes the page must carry. They read
the saved posteriors and write nothing themselves; `capacity_report.main` joins them to the report.
"""

import math
from typing import Final

import numpy as np
import polars as pl
from capacity_inputs import demand_series
from capacity_report_tools import (
    BLOCK_NAMES_BY_INDEX,
    FALSE_ALARM_RATE,
    SUMMER_BLOCK,
    clopper_pearson,
    coverage,
    flag,
    rate_table,
    table,
    thresholds,
    with_errors,
)
from capacity_runs import p99_flow

PASS_FRACTION: Final[float] = 2.0 / 3.0
"""The clean positive control's pass rule: the 90% interval holds both power and energy in at least
this fraction of the replica fits (committed in `capacity_positive_control_clean.py`)."""
RATIO_BIN_START_LOG2: Final[int] = -3
TUNING_SERIES: Final[tuple[str, ...]] = ("S2", "S6")
"""The two series the estimator was tuned on, which every pooled rate also scores."""
PRIOR_SCALE_OVER_P99: Final[float] = 0.2
"""The scale of the half-normal prior on a class's power, as a share of the series' p99."""


def with_season(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `season`: Jun-Aug or Sep-May."""
    return frame.with_columns(
        season=pl.when(pl.col("block") == SUMMER_BLOCK)
        .then(pl.lit("Jun-Aug"))
        .otherwise(pl.lit("Sep-May"))
    )


def flagged_null_series(*, nulls: pl.DataFrame, limits: dict[str, float]) -> tuple[str, ...]:
    """Return the series that have a flagged null block, sorted."""
    flagged = flag(frame=nulls, threshold=limits, default=np.nan).filter(pl.col("flagged"))
    return tuple(sorted(flagged["series"].unique().to_list()))


def default_threshold(*, limits: dict[str, float]) -> float:
    """Return the threshold for a series with no nulls of its own."""
    return float(np.quantile(list(limits.values()), 1 - FALSE_ALARM_RATE))


def positive_control_clean_section(*, draws: pl.DataFrame) -> list[str]:
    """The clean positive control's headline numbers (the pass rule is committed in its script)."""
    replica = draws.filter(pl.col("demand") == "replica")
    real = draws.filter(pl.col("demand") == "real")
    needed = math.ceil(PASS_FRACTION * replica.height)
    both = replica["power_in_90"] & replica["energy_in_90"]
    real_both = float((real["power_in_90"] & real["energy_in_90"]).to_numpy().mean())
    lines = [
        "## The clean positive control (pass rule committed before the run)",
        "",
        (
            f"**{'PASS' if both.sum() >= needed else 'FAIL'}: the 90% interval holds both the "
            f"merchant power and the merchant energy in {int(both.sum())} of {replica.height} "
            f"calendar-replica fits ({both.mean():.1%}); at least {needed} were required.** The "
            "truths are 10 fresh off-node draws on 7 series no earlier diagnostic touched, in all "
            "4 blocks, at a 40% share. This supersedes the earlier pass (2 of 3 blocks, one "
            "truth, after tuning on a scored block, a truth 9% and 11% of a node spacing from the "
            "nodes)."
        ),
        "",
        (
            f"**The pass is a pass of the committed rule, not of the nominal 90%.** On the "
            f"replicas the 90% interval holds the power in {replica['power_in_90'].mean():.1%} of "
            f"fits and the energy in {replica['energy_in_90'].mean():.1%}, below its nominal "
            f"coverage, with intervals about "
            f"{float(np.median(replica['power_width_over_truth'].to_numpy())):.1%} of the truth "
            "wide; the median absolute power error is "
            f"{float(np.median(np.abs(replica['power_median_error'].to_numpy()))):.2%}. On real "
            f"demand the intervals hold both in {real_both:.1%} "
            "of fits, because the tempering by the residual's autocorrelation widens them."
        ),
        "",
    ]
    real_widths = real["power_width_over_truth"].to_numpy()
    real_error = float(np.median(real["power_median_error"].to_numpy()))
    gsp1_error = float(
        np.median(real.filter(pl.col("series") == "GSP1")["power_median_error"].to_numpy())
    )
    lines += [
        (
            f"**On real demand the 90% interval holds both truths in {real_both:.1%} of "
            f"{real.height} fits, and is wide.** The interval is {np.min(real_widths):.0%} to "
            f"{np.max(real_widths):.0%} of the truth wide (median {np.median(real_widths):.0%}). "
            f"The median power error is {real_error:.1%}, and it is {gsp1_error:.1%} for GSP1."
        ),
        "",
        (
            f"**The truths are the estimator's own dispatch, at a 40% share only.** Each truth is "
            "a day-by-day linear programme on the day-ahead price with parameters off the "
            f"estimator's grid, and its usable duration is {draws['truth_usable_hours'].min():.2f} "
            f"to {draws['truth_usable_hours'].max():.2f} hours, so the control tests the "
            "interpolation between precomputed schedules, not a real dispatch. The pass rule "
            "(two thirds of fits holding both truths) is lenient for a nominal 90% interval: two "
            "independent 90% intervals hold both truths in 81% of fits."
        ),
        "",
        "Replica fits holding both truths, by truth:",
        "",
        table(
            replica.with_columns(both=pl.col("power_in_90") & pl.col("energy_in_90"))
            .group_by("truth")
            .agg(fits=pl.len(), both_in_90=pl.col("both").mean())
            .sort("truth")
        ),
    ]
    for name, frame in (("Calendar replicas", replica), ("Real demand", real)):
        lines += [
            f"**{name}: {frame.height} fits.**",
            "",
            table(
                frame.group_by("block")
                .agg(
                    fits=pl.len(),
                    power_in_90=pl.col("power_in_90").mean(),
                    energy_in_90=pl.col("energy_in_90").mean(),
                    median_power_error=pl.col("power_median_error").median(),
                    median_absolute_power_error=pl.col("power_median_error").abs().median(),
                    median_absolute_energy_error=pl.col("energy_median_error").abs().median(),
                    median_tau=pl.col("tau").median(),
                )
                .sort("block")
            ),
        ]
    return lines


def rung1b_section(
    *, rung1: pl.DataFrame, others: dict[str, pl.DataFrame], limits: dict[str, float]
) -> list[str]:
    """Rung 1 beside the batteries dispatched by policies outside the estimator's family."""
    default = default_threshold(limits=limits)
    lines = [
        "## Rung 1 against batteries dispatched outside the estimator's family (exploratory)",
        "",
        (
            "Rung 1's truth is the day-by-day linear programme on the N2EX price, which is the "
            "family the estimator interpolates, so rung 1 is the best case: the battery "
            "dispatches exactly as the estimator assumes. `rank_rule` is a battery that charges "
            "in each day's cheapest and discharges in its dearest half-hours; `noisy_price` is "
            "the linear programme on the price times one plus independent noise of standard "
            "deviation 0.2 per half-hour. The thresholds are rung 1's. Each table gives the share "
            "of sums flagged, the share whose 90% interval holds the truth (power and energy), "
            "and the median absolute relative error of the power median."
        ),
        "",
    ]
    for name, frame in (
        ("rung1 (in the family)", rung1.filter(pl.col("share") > 0)),
        *others.items(),
    ):
        flagged = with_season(frame=flag(frame=frame, threshold=limits, default=default))
        errors = with_errors(frame=frame, power="merchant_power", energy="merchant_energy")
        rates = rate_table(frame=flagged, by=["share"]).select(
            "share", "flagged", "total", "rate", "ci_low", "ci_high"
        )
        covered = coverage(frame=errors, by=["share"]).select(
            "share", "power_90", "energy_90", "power_50"
        )
        sizes = errors.group_by("share").agg(
            median_abs_power_error=pl.col("power_error").median(),
            median_abs_energy_error=pl.col("energy_error").median(),
            median_log_bayes_factor=pl.col("log_bayes_factor").median(),
        )
        lines += [
            f"**{name}.**",
            "",
            table(rates.join(covered, on="share").join(sizes, on="share").sort("share")),
            f"**{name}, flags by share and season.**",
            "",
            table(rate_table(frame=flagged, by=["share", "season"])),
        ]
    return lines


def rung2_controls_section(
    *,
    rung2: pl.DataFrame,
    shifted: pl.DataFrame,
    coarse_real: pl.DataFrame,
    coarse_placebo: pl.DataFrame,
    limits: dict[str, float],
) -> list[str]:
    """Rung 2's negative controls, with the Agile and N2EX prices also moved."""
    default = default_threshold(limits=limits)
    lines = [
        "## Rung 2 negative controls (exploratory)",
        "",
        (
            "**Rung 2's detections come from the Agile unit; the fixed-window units are "
            "unidentified.** The one-hour-early control moves only the four fixed windows, and the "
            "Agile unit (a price taker on the Agile price) is unchanged, so the flag counts "
            "below show how many detections the shift removes. A fixed window repeats every day, "
            "and the monthly baseline's daily profile absorbs a schedule that repeats every day by "
            "construction, so the fixed-window result is a property of the baseline's design as well as of the "
            "data. The control below moves the windows one hour early and takes both prices "
            "from 7 days later (the coarse stacks), beside the same coarse stacks with the real "
            "windows and prices."
        ),
        "",
    ]
    for name, frame in (
        ("Fine stacks, real windows and prices", rung2),
        ("Fine stacks, windows one hour early", shifted),
        ("Coarse stacks, real windows and prices", coarse_real),
        ("Coarse stacks, windows one hour early and prices 7 days later", coarse_placebo),
    ):
        flagged = flag(frame=frame, threshold=limits, default=default)
        lines += [
            f"**{name}.**",
            "",
            table(rate_table(frame=with_season(frame=flagged), by=["share"])),
        ]
    return lines


def screen_power_section(*, screen: pl.DataFrame) -> list[str]:
    """How often the real template set ranks first on lanes with and without a known battery."""
    yearly = (
        screen.group_by(
            "series", "lane", "lane_kind", "share", "nameplate_hours", "template_set", "set_kind"
        )
        .agg(log_bayes_factor=pl.col("log_bayes_factor").sum())
        .filter(pl.col("log_bayes_factor").is_finite())
    )
    keys = ["series", "lane"]
    ranked = yearly.with_columns(
        rank_all=pl.col("log_bayes_factor").rank(method="min", descending=True).over(keys)
    )
    price_only = (
        yearly.filter(pl.col("set_kind").is_in(["real", "price_placebo"]))
        .with_columns(
            rank_price=pl.col("log_bayes_factor").rank(method="min", descending=True).over(keys)
        )
        .filter(pl.col("template_set") == "real")
        .select(*keys, "rank_price")
    )
    real = (
        ranked.filter(pl.col("template_set") == "real")
        .join(price_only, on=keys)
        .with_columns(first_of_13=pl.col("rank_all") == 1, first_of_price=pl.col("rank_price") == 1)
    )
    # A lane's real set needs finite factors for all 13 sets to be ranked against all 13.
    complete = (
        yearly.group_by(keys).agg(n_sets=pl.len()).filter(pl.col("n_sets") == 13).select(keys)
    )
    real = real.join(complete, on=keys)
    groups = real.group_by("lane_kind", "share").agg(
        lanes=pl.len(),
        real_first_of_13=pl.col("first_of_13").sum(),
        rate_first_of_13=pl.col("first_of_13").mean(),
        real_first_among_price_placebos=pl.col("first_of_price").sum(),
        rate_first_among_price_placebos=pl.col("first_of_price").mean(),
        median_rank_of_13=pl.col("rank_all").median(),
    )
    null = real.filter(pl.col("lane_kind") == "null")
    merchant = real.filter(pl.col("lane_kind") == "merchant")
    public = real.filter(pl.col("lane_kind") == "public")
    null_count, null_total = int(null["first_of_13"].sum()), null.height
    low, high = clopper_pearson(count=null_count, total=null_total)
    return [
        "## The power of the primary screen (exploratory)",
        "",
        (
            "The screen ranks the real template set against 12 placebo sets and reads rank 1 of "
            "13 as evidence of a battery. These tables run the same 13 sets (coarse stacks) on "
            "the nine demand-like series with no added battery, with rung 1's simulated merchant "
            "batteries at 10% and 40% of the series' 99th percentile absolute flow, and with the "
            "four named public batteries at 40%, and count how often the real set ranks first, "
            "summing each lane's log Bayes factor over the four blocks."
        ),
        "",
        (
            f"**With no battery added, the real set ranks first of 13 in {null_count} of "
            f"{null_total} series (Clopper-Pearson 95% interval {low:.2f} to {high:.2f}); "
            f"1 in 13 is 0.077.** The placebos are not exchangeable with the real set, so this "
            "measured rate, not 1 in 13, is the screen's chance rate."
        ),
        "",
        table(groups.sort("lane_kind", "share")),
        (
            "**The 13-set screen has no power to find a battery; the screen restricted to the "
            "6 price-shifted placebos does.** Against all 13 sets the real set ranks first in "
            f"{int(merchant['first_of_13'].sum())} of {merchant.height} lanes holding a simulated "
            f"merchant battery of 10% or 40% ({merchant['first_of_13'].mean():.1%}), no more "
            "often than 1 in 13 (7.7%): the shifted-window placebos often outrank the real set "
            "(the median rank of the real set is in the table). Against the real set's 6 "
            "price-shifted rivals alone "
            f"(7 sets in all) it ranks first in {int(merchant['first_of_price'].sum())} of "
            f"{merchant.height} simulated merchant lanes, in {int(null['first_of_price'].sum())} "
            f"of {null.height} lanes with no battery, and in "
            f"{int(public['first_of_price'].sum())} of {public.height} lanes holding a real "
            "public battery at 40%."
        ),
        "",
        "By nameplate duration (merchant lanes only):",
        "",
        table(
            real.filter(pl.col("lane_kind") == "merchant")
            .group_by("share", "nameplate_hours")
            .agg(
                lanes=pl.len(),
                real_first_of_13=pl.col("first_of_13").sum(),
                rate=pl.col("first_of_13").mean(),
            )
            .sort("share", "nameplate_hours")
        ),
    ]


def detection_curve_section(
    *,
    families: dict[str, pl.DataFrame],
    steps: pl.DataFrame,
    limits: dict[str, float],
    nulls: pl.DataFrame,
) -> list[str]:
    """Detection rate against battery power over the noise unit `sigma_step`.

    Args:
        families: The rows of each battery family, with the family's name as the key.
        steps: Rung 1's step-tail table, which holds the noise unit.
        limits: Each series' detection threshold.
        nulls: Rung 1's rows with no added battery, from which the thresholds of the variant
            without the series that false-alarm are rebuilt.

    Returns:
        The report lines.
    """
    default = default_threshold(limits=limits)
    false_alarm_series = flagged_null_series(nulls=nulls, limits=limits)
    clean_limits = thresholds(nulls=nulls.filter(~pl.col("series").is_in(false_alarm_series)))
    clean_default = default_threshold(limits=clean_limits)
    sigma = steps.filter(pl.col("share") == 0).select(
        "series", "block", sigma_step_null="sigma_step_mw"
    )
    tables = []
    for name, frame in families.items():
        flagged = flag(frame=frame, threshold=limits, default=default)
        clean = flag(frame=frame, threshold=clean_limits, default=clean_default).select(
            "series", "block", "share", "true_power_mw", clean_flagged="flagged"
        )
        flagged = with_season(frame=flagged)
        flagged = flagged.with_columns(row=pl.int_range(pl.len())).join(
            clean.with_columns(row=pl.int_range(pl.len())).select("row", "clean_flagged"),
            on="row",
        )
        flagged = flagged.join(sigma, on=["series", "block"], how="left")
        tables.append(
            flagged.with_columns(
                ratio=pl.col("true_power_mw") / pl.col("sigma_step_null"), family=pl.lit(name)
            ).select(
                "family", "series", "block", "season", "share", "ratio", "flagged", "clean_flagged"
            )
        )
    everything = pl.concat(tables).filter(pl.col("ratio").is_finite() & (pl.col("ratio") > 0))
    ratios = everything["ratio"].to_numpy()
    low = math.floor(math.log2(float(ratios.min())))
    high = math.ceil(math.log2(float(ratios.max())))
    edges = 2.0 ** np.arange(max(low, RATIO_BIN_START_LOG2), high + 1)
    bins = np.digitize(everything["ratio"].to_numpy(), edges)
    everything = everything.with_columns(
        bin_low=pl.Series(edges[np.clip(bins - 1, 0, len(edges) - 1)]),
    )
    lines = [
        "## Detection against battery power over the noise unit (exploratory)",
        "",
        (
            "`P / sigma_step` is the battery's true power divided by the robust standard deviation "
            "of the half-hour changes of its demand series' residual with no battery in the same "
            "block (`sigma_step`), so a primary with a noisier flow needs a bigger battery. Bins "
            "are powers of 2. Rung 1 is the in-family best case, `rank_rule` and `noisy_price` "
            "are simulated batteries outside the family, and rung 3 is real public batteries "
            "(the registered power is the truth). The second table drops the series with a "
            "flagged null block and rebuilds the thresholds from the others; the third drops "
            "Jun-Aug."
        ),
        "",
    ]
    clean_name = f"Without {', '.join(false_alarm_series)}"
    for label, frame in (
        ("All blocks", everything),
        (
            clean_name,
            everything.filter(~pl.col("series").is_in(false_alarm_series)).with_columns(
                flagged=pl.col("clean_flagged")
            ),
        ),
        ("Sep-May only", everything.filter(pl.col("season") == "Sep-May")),
    ):
        summary = (
            frame.group_by("family", "bin_low")
            .agg(sums=pl.len(), flagged=pl.col("flagged").sum(), rate=pl.col("flagged").mean())
            .sort("family", "bin_low")
        )
        intervals = [
            clopper_pearson(count=int(f), total=int(n))
            for f, n in zip(summary["flagged"], summary["sums"], strict=True)
        ]
        summary = summary.with_columns(
            ci_low=pl.Series([i[0] for i in intervals]),
            ci_high=pl.Series([i[1] for i in intervals]),
        )
        lines += [f"**{label}.**", "", table(summary)]
        for family in summary["family"].unique().sort().to_list():
            part = summary.filter((pl.col("family") == family) & (pl.col("sums") >= 10))
            reaching = part.filter(pl.col("rate") >= 0.5)
            first = float(reaching["bin_low"].to_numpy().min()) if reaching.height else float("nan")
            lines += [
                (
                    f"{label}, {family}: the first bin (of at least 10 sums) where at least half "
                    f"of the sums are flagged starts at P / sigma_step = {first:.3g}."
                )
            ]
        lines += [""]
    return lines


def start_disagreement_section(*, frames: dict[str, pl.DataFrame]) -> list[str]:
    """How often the three starts disagree by more than the posterior interval is wide."""
    rows = []
    for name, frame in frames.items():
        usable = frame.filter(pl.col("has_interval"))
        spread = usable["start_spread_merchant_power"].to_numpy()
        width = (
            (usable["merchant_power_q95"] - usable["merchant_power_q05"])
            / usable["merchant_power_median"]
        ).to_numpy()
        rows.append(
            {
                "rung": name,
                "sums_with_interval": usable.height,
                "starts_disagree_beyond_interval_width": float(np.mean(spread > width)),
            }
        )
    return [
        "## Do the three starts agree? (exploratory)",
        "",
        (
            "The share of sums whose three starts' merchant powers differ (maximum minus minimum, "
            "relative to the best start) by more than the best start's 90% interval is wide "
            "(relative to its median). A large share means the smoothed objective is multimodal "
            "and the Laplace interval describes one mode."
        ),
        "",
        table(pl.DataFrame(rows)),
    ]


def rung3_replica_section(*, replica: pl.DataFrame) -> list[str]:
    """Real public batteries on calendar replicas, where the demand noise is almost gone."""
    p99 = {label: p99_flow(v) for label, v in demand_series().items()}
    frame = replica.with_columns(
        p99=pl.col("series").replace_strict(p99, return_dtype=pl.Float64),
        registered_over_p99=pl.col("share"),
    ).with_columns(
        median_power_over_p99=pl.col("merchant_power_median") / pl.col("p99"),
        median_over_registered=pl.when(pl.col("true_power_mw") > 0).then(
            pl.col("merchant_power_median") / pl.col("true_power_mw")
        ),
    )
    limits = thresholds(nulls=frame.filter(pl.col("share") == 0))
    flagged = with_season(
        frame=flag(frame=frame.filter(pl.col("share") > 0), threshold=limits, default=np.nan)
    )
    return [
        "## Rung 3 on calendar replicas (exploratory)",
        "",
        (
            "The same 23 real public batteries and fleets, added to the calendar replicas of the "
            "nine series, which have almost no noise. If the posterior still does not recover the "
            "registered power, the real dispatch lies outside the estimator's family; if it does, "
            "real demand noise hides the battery in rung 3."
        ),
        "",
        table(
            frame.group_by("share")
            .agg(
                sums=pl.len(),
                median_power_over_p99=pl.col("median_power_over_p99").median(),
                median_over_registered=pl.col("median_over_registered").median(),
                median_log_bayes_factor=pl.col("log_bayes_factor").median(),
                median_tau=pl.col("tau").median(),
            )
            .sort("share")
        ),
        (
            "**Flagged by a threshold from the replicas' own nulls.** Each series' threshold is "
            "the 95th percentile of the other series' log Bayes factors on the 36 replica blocks "
            "with no added battery, because the real-demand threshold does not apply to a "
            "demand series with almost no noise."
        ),
        "",
        table(rate_table(frame=flagged, by=["share"])),
        table(rate_table(frame=flagged, by=["share", "season"])),
        "By kind of unit, 40% share:",
        "",
        table(
            frame.filter(pl.col("share") == 0.40)
            .group_by("kind")
            .agg(
                sums=pl.len(),
                median_over_registered=pl.col("median_over_registered").median(),
                q25=pl.col("median_over_registered").quantile(0.25),
                q75=pl.col("median_over_registered").quantile(0.75),
            )
            .sort("kind")
        ),
    ]


def low_share_posterior_section(*, rung1: pl.DataFrame) -> list[str]:
    """The merchant power's posterior at the smallest shares, against the prior and the null."""
    p99 = {label: p99_flow(v) for label, v in demand_series().items()}
    frame = rung1.with_columns(
        p99=pl.col("series").replace_strict(p99, return_dtype=pl.Float64)
    ).with_columns(
        median_over_p99=pl.col("merchant_power_median") / pl.col("p99"),
        q05_over_p99=pl.col("merchant_power_q05") / pl.col("p99"),
        q95_over_p99=pl.col("merchant_power_q95") / pl.col("p99"),
    )
    z_median, z_q05 = 0.6745, 0.0627
    return [
        "## Rung 1: the merchant power's posterior at the smallest shares (exploratory)",
        "",
        (
            "The half-normal prior has scale "
            f"{PRIOR_SCALE_OVER_P99:.0%} of the series' 99th percentile, so its median is "
            f"{z_median * PRIOR_SCALE_OVER_P99:.1%} and its 5th percentile "
            f"{z_q05 * PRIOR_SCALE_OVER_P99:.2%} of the 99th percentile. Share 0 is the nulls."
        ),
        "",
        table(
            frame.filter(pl.col("share") <= 0.05)
            .group_by("share")
            .agg(
                sums=pl.len(),
                median_of_posterior_medians=pl.col("median_over_p99").median(),
                median_of_q05=pl.col("q05_over_p99").median(),
                median_of_q95=pl.col("q95_over_p99").median(),
            )
            .sort("share")
        ),
    ]


def tuning_series_section(*, rung1: pl.DataFrame) -> list[str]:
    """Rung 1's false alarms, detection, and calibration with some series removed.

    One variant drops the two series the estimator was tuned on. The other drops the series whose
    null blocks are flagged, because a series flagged with no battery adds its false alarms to
    every detection rate.
    """
    nulls = rung1.filter(pl.col("share") == 0)
    false_alarm_series = flagged_null_series(nulls=nulls, limits=thresholds(nulls=nulls))
    lines = [
        "## Rung 1 with series removed (exploratory)",
        "",
        (
            f"The estimator was tuned on {' and '.join(TUNING_SERIES)}, and every pooled rate also "
            "scores them. The series with a flagged null block ("
            f"{', '.join(false_alarm_series)}) adds its false alarms to every detection rate, "
            "including the rates at the smallest shares. The tables repeat the false-alarm rate, "
            "the detection rates, and the merchant power's 90% coverage with the thresholds "
            "rebuilt from the remaining series."
        ),
        "",
    ]
    variants = (
        ("All 9 series", ()),
        (f"Without {' and '.join(TUNING_SERIES)}", TUNING_SERIES),
        (f"Without {', '.join(false_alarm_series)}", false_alarm_series),
    )
    for name, removed in variants:
        frame = rung1.filter(~pl.col("series").is_in(removed))
        limits = thresholds(nulls=frame.filter(pl.col("share") == 0))
        default = default_threshold(limits=limits)
        nulls = flag(frame=frame.filter(pl.col("share") == 0), threshold=limits, default=default)
        battery = flag(frame=frame.filter(pl.col("share") > 0), threshold=limits, default=default)
        merchant = with_errors(
            frame=frame.filter(pl.col("share") > 0),
            power="merchant_power",
            energy="merchant_energy",
        ).with_columns(all=pl.lit("all"))
        lines += [
            f"**{name}: {int(nulls['flagged'].sum())} of {nulls.height} null blocks flagged.**",
            "",
            table(rate_table(frame=battery, by=["share"])),
            table(
                rate_table(
                    frame=with_season(frame=battery.filter(pl.col("share").is_in([0.05, 0.1]))),
                    by=["share", "season"],
                )
            ),
            table(coverage(frame=merchant, by=["all"])),
        ]
    return lines


def notes_section() -> list[str]:
    """Statements the page must carry, which no table can state."""
    return [
        "## Notes the page must carry",
        "",
        (
            "- The log Bayes factor is a Laplace evidence with the likelihood raised to the power "
            "`1 / tau`, where `tau` is the residual's integrated autocorrelation time (about 49 "
            "on real demand). It is not a Bayes factor in the standard sense, and it is "
            "calibrated only empirically, against 36 null blocks from one year."
        ),
        (
            "- The half-normal prior scale of each class's power is 20% of the 99th percentile of "
            "the aggregate including the battery, so at large shares the battery widens its own "
            "prior. The scale could instead come from a battery-free reference; this was not "
            "done, because it would change every fit."
        ),
        (
            "- The energy reference for real public batteries and for NGED battery A is the "
            "smallest capacity that holds the metered state of charge, a lower bound on the "
            "energy capacity. Energy coverage in rungs 3 and 4 is against a lower bound."
        ),
        (
            "- A schedule that repeats every day, as a fixed tariff window does, is absorbed by "
            "the monthly baseline's daily profile by construction; daylight-saving changes and "
            "the weekday-only red band are the only day-to-day variation left to those units."
        ),
        (
            "- The estimator's power and energy numbers for the domestic and commercial classes "
            "sit at their priors and are never printed as estimates."
        ),
        (
            "- The differentiable estimator has no state-of-charge limits or cycle-cap parameters "
            "to move: the usable duration absorbs the limits, and the merchant battery's cap is a "
            "learned weight between 1 and 2 cycles a day, while the Agile unit's cap is fixed at 1 "
            "cycle a day. The second setting therefore doubles the duration priors' spread and "
            "widens the efficiency prior."
        ),
        (
            "- The simulated batteries' dispatch is a perfect-foresight daily linear programme, "
            "and real GB batteries also earn from frequency response, the balancing mechanism, "
            "and intraday trading, which this estimator does not model."
        ),
    ]


def block_name_column(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `block_name` from the integer `block` column."""
    return frame.with_columns(block_name=pl.col("block").replace_strict(BLOCK_NAMES_BY_INDEX))
