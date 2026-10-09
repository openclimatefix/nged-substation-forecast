"""Print every table the page quotes into `report.md`.

Reads the rungs' saved posteriors (`rung*_posteriors.parquet`), the step tails, the grid
estimator's rung 1 posteriors, and the positive control, and writes: the thresholds and false-alarm
rate (C1), the detection rates, the calibration (C2), the posterior against the step tail (C3), the
energy against the rule (C4), the fleets (C5), the negative controls, rung 4, the primary screen
with its within-primary placebo, and the comparison with the grid estimator. Intervals on the mean
errors of C3, C4, and C5 and on the grid-estimator difference resample whole demand series and
whole batteries (a two-level cluster bootstrap, 2,000 resamples). Rates and coverages carry
Clopper-Pearson intervals on sums, which treat the sums as independent although they share series
and blocks, so they are too narrow.

Run: `uv run python studies/unmetered_battery_capacity/capacity_report.py`
"""

from typing import Final

import capacity_report_review as review
import numpy as np
import polars as pl
from capacity_inputs import OUTPUT_DIR, demand_series, nged_series, storage_presence_by_primary
from capacity_report_tools import (
    BLOCK_NAMES_BY_INDEX,
    FALSE_ALARM_RATE,
    SUMMER_BLOCK,
    clopper_pearson,
    cluster_bootstrap,
    coverage,
    flag,
    rate_table,
    table,
    thresholds,
    with_errors,
)
from capacity_runs import p99_flow
from capacity_templates import DURATION_PRIORS
from scipy.stats import binomtest

# Duplicated from `capacity_state_space` so that the report need not import torch.
UNIT_NAMES: Final[tuple[str, ...]] = (
    "merchant",
    "agile",
    "commercial_and_industrial",
    "intelligent_octopus_go",
    "octopus_go",
    "octopus_flux",
)
"""The estimator's six units, in the order of its parameter vector."""
MIN_SHARE_C3: Final[float] = 0.05
MIN_SHARE_C4: Final[float] = 0.10
Z90: Final[float] = 1.645


def c1_section(
    *, rung1: pl.DataFrame, limits: dict[str, float], heading_suffix: str = ""
) -> list[str]:
    """C1: the false-alarm rate on the blocks with no added battery."""
    nulls = flag(frame=rung1.filter(pl.col("share") == 0), threshold=limits, default=np.nan)
    count, total = int(nulls["flagged"].sum()), nulls.height
    n_unevaluated = int((~nulls["log_bayes_factor"].is_finite()).sum())
    test = binomtest(count, total, FALSE_ALARM_RATE, alternative="greater")
    low, high = clopper_pearson(count=count, total=total)
    return [
        f"## C1: false alarms on the blocks with no added battery (planned){heading_suffix}",
        "",
        (
            f"**The nominal false-alarm rate is 5%; the realised rate is {count} of {total} "
            f"blocks ({count / total:.1%}).** Clopper-Pearson 95% interval "
            f"{low:.3f} to {high:.3f}. One-sided exact binomial test against 5%: p = "
            f"{test.pvalue:.3f}; the contrast {'holds' if test.pvalue >= 0.05 else 'fails'}. "
            f"{n_unevaluated}"
            " null blocks have no log Bayes factor (no positive-definite Hessian) and count as not "
            "flagged."
        ),
        "",
        "Flags by block:",
        "",
        table(
            nulls.with_columns(block_name=pl.col("block").replace_strict(BLOCK_NAMES_BY_INDEX))
            .group_by("block", "block_name")
            .agg(
                flagged=pl.col("flagged").sum(),
                blocks=pl.len(),
                median_tau=pl.col("tau").median(),
                min_tau=pl.col("tau").min(),
                max_tau=pl.col("tau").max(),
                median_log_bayes_factor=pl.col("log_bayes_factor").median(),
            )
            .sort("block")
        ),
        (
            "The flagged series and blocks: "
            + ", ".join(
                f"{r['series']} {BLOCK_NAMES_BY_INDEX[r['block']]}"
                for r in nulls.filter(pl.col("flagged")).sort("series", "block").to_dicts()
            )
            + "."
        ),
        "",
        "Flags by series:",
        "",
        table(
            nulls.group_by("series")
            .agg(flagged=pl.col("flagged").sum(), blocks=pl.len())
            .sort("series")
        ),
        "Thresholds (the 95th percentile of the other 8 series' null log Bayes factors):",
        "",
        table(
            pl.DataFrame({"series": list(limits), "threshold": list(limits.values())}).sort(
                "series"
            )
        ),
        "Log Bayes factor of every null block:",
        "",
        table(
            nulls.select("series", "block", "log_bayes_factor", "threshold", "flagged").sort(
                "series", "block"
            )
        ),
    ]


def season(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add a `season` column: Jun-Aug or Sep-May."""
    return frame.with_columns(
        season=pl.when(pl.col("block") == SUMMER_BLOCK)
        .then(pl.lit("Jun-Aug"))
        .otherwise(pl.lit("Sep-May"))
    )


def detection_section(
    *, rung1: pl.DataFrame, rung2: pl.DataFrame, shifted: pl.DataFrame | None,
    rung3: pl.DataFrame | None, limits: dict[str, float], heading_suffix: str = "",
) -> list[str]:  # fmt: skip
    """Detection rates by share for rungs 1 to 3 and the shifted-window negative control."""
    default = float(np.quantile(list(limits.values()), 1 - FALSE_ALARM_RATE))
    lines = [
        f"## Detection rates at the nominal 5% false-alarm threshold (exploratory){heading_suffix}",
        "",
        (
            "The threshold's realised false-alarm rate is in the C1 section. A series whose null "
            "blocks are flagged adds its false alarms to every rate, including the rates at the "
            "smallest shares, so the tables of rung 1 without that series follow in the section "
            "on series removed. Every table is also split into Jun-Aug and Sep-May."
        ),
        "",
    ]
    tables: list[tuple[str, pl.DataFrame, list[str]]] = [
        ("Rung 1, merchant batteries, by share", rung1.filter(pl.col("share") > 0), ["share"]),
        (
            "Rung 1, by share and nameplate duration",
            rung1.filter(pl.col("share") > 0),
            ["share", "nameplate_hours"],
        ),
        ("Rung 2, simulated fleets, real windows", rung2, ["share"]),
    ]
    if shifted is not None:
        tables.append(
            (
                "Rung 2, simulated fleets, windows one hour early (negative control)",
                shifted,
                ["share"],
            )
        )
    if rung3 is not None:
        tables += [
            ("Rung 3, real public batteries, by share", rung3, ["share"]),
            ("Rung 3, by share and kind", rung3, ["share", "kind"]),
        ]
    for name, frame, by in tables:
        flagged = season(frame=flag(frame=frame, threshold=limits, default=default))
        lines += [f"**{name}.**", "", table(rate_table(frame=flagged, by=by))]
        if by == ["share"]:
            lines += [
                f"**{name}, split by season.**",
                "",
                table(rate_table(frame=flagged, by=[*by, "season"])),
            ]
    return lines


def calibration_section(
    *, rung1: pl.DataFrame, rung2: pl.DataFrame, heading_suffix: str = ""
) -> list[str]:
    """C2: the coverage of the credible intervals on the simulated known answers."""
    merchant = with_errors(
        frame=rung1.filter(pl.col("share") > 0), power="merchant_power", energy="merchant_energy"
    )
    fleets = with_errors(frame=rung2, power="domestic_power", energy="domestic_energy")
    lines = [f"## C2: calibration of the credible intervals (planned){heading_suffix}", ""]
    for name, frame in (("Rung 1 (merchant class)", merchant), ("Rung 2 (domestic class)", fleets)):
        overall = frame.with_columns(all=pl.lit("all"))
        lines += [
            f"**{name}, all shares.**",
            "",
            table(coverage(frame=overall, by=["all"])),
            f"**{name}, by share.**",
            "",
            table(coverage(frame=frame, by=["share"])),
        ]
    agile = rung2.with_columns(
        power_in_90=(pl.col("unit_agile_power_q05") <= pl.col("true_agile_power_mw"))
        & (pl.col("true_agile_power_mw") <= pl.col("unit_agile_power_q95")),
        all=pl.lit("all"),
    )
    lines += [
        (
            "**Fleet calibration scored on the Agile unit alone.** The three fixed-window units "
            "return their prior's mode, because the monthly baseline absorbs any schedule that "
            "repeats every day, so the fleet's summed power is scored here against the simulated "
            "Agile homes' rated power only:"
        ),
        "",
        table(
            agile.group_by("share")
            .agg(n=pl.len(), agile_power_90=pl.col("power_in_90").fill_null(False).mean())
            .sort("share")
        ),
    ]
    simulated = pl.concat(
        [
            merchant.select(
                "series",
                "share",
                "power_in_90",
                "power_in_50",
                "energy_in_90",
                "energy_in_50",
                "has_interval",
            ),
            fleets.select(
                "series",
                "share",
                "power_in_90",
                "power_in_50",
                "energy_in_90",
                "energy_in_50",
                "has_interval",
            ),
        ]
    ).with_columns(all=pl.lit("rungs 1 and 2"))
    summary = coverage(frame=simulated, by=["all"])
    row = summary.row(0, named=True)
    verdict_power = row["power_90"] >= 0.80 and 0.40 <= row["power_50"] <= 0.60
    lines += [
        (
            "**Rungs 1 and 2 pooled (the plan's judgement).** The pooled figure hides the spread "
            "by share and by rung that the tables above give; the page reports the per-share "
            "coverage, not this figure alone."
        ),
        "",
        table(summary),
        (
            f"Power: the 90% interval holds the truth in {row['power_90']:.1%} of sums "
            f"(at least 80% required) and the 50% interval in {row['power_50']:.1%} (40% to 60% "
            f"required): {'holds' if verdict_power else 'fails'}. Energy: "
            f"{row['energy_90']:.1%} and {row['energy_50']:.1%}."
        ),
        "",
    ]
    return lines


def c3_c4_section(
    *, rung1: pl.DataFrame, steps: pl.DataFrame, heading_suffix: str = ""
) -> list[str]:
    """C3 (posterior against the step tail) and C4 (energy against the rule)."""
    keys = ["series", "nameplate_hours", "share", "block"]
    joined = (
        rung1.filter(pl.col("share") > 0)
        .join(steps.select(*keys, "step_tail_power_mw"), on=keys, how="inner")
        .with_columns(
            post=(pl.col("merchant_power_median") - pl.col("true_power_mw")).abs()
            / pl.col("true_power_mw"),
            step=(pl.col("step_tail_power_mw") - pl.col("true_power_mw")).abs()
            / pl.col("true_power_mw"),
        )
        .with_columns(
            difference=pl.col("post") - pl.col("step"),
            battery=pl.col("nameplate_hours").cast(pl.Utf8),
        )
    )
    c3 = joined.filter(pl.col("share") >= MIN_SHARE_C3)
    mean, low, high = cluster_bootstrap(
        frame=c3, value="difference", outer="series", inner="battery"
    )
    lines = [
        f"## C3: posterior median against the step tail, power error (planned){heading_suffix}",
        "",
        (
            f"At shares of {MIN_SHARE_C3:.0%} and above ({c3.height} sums), the mean paired "
            f"difference in relative power error (posterior median minus step tail) is {mean:+.4f} "
            f"(95% cluster-bootstrap interval {low:+.4f} to {high:+.4f}); the contrast "
            f"{'holds' if high < 0 else 'fails'}. Rung 1 is the in-family best case, and the step "
            "tail is a weak comparator (an LP battery rarely flips from full charge to full "
            "discharge within one half-hour), so a pass says the posterior beats a naive "
            "statistic on its own family and nothing about real batteries."
        ),
        "",
        "Median relative power error by share:",
        "",
        table(
            joined.group_by("share")
            .agg(posterior=pl.col("post").median(), step_tail=pl.col("step").median(), n=pl.len())
            .sort("share")
        ),
    ]
    c4 = (
        rung1.filter(
            (pl.col("share") >= MIN_SHARE_C4) & pl.col("nameplate_hours").is_in([1.0, 4.0])
        )
        .with_columns(
            post=(pl.col("merchant_energy_median") - pl.col("true_energy_mwh")).abs()
            / pl.col("true_energy_mwh"),
            rule=(2.0 * pl.col("merchant_power_median") - pl.col("true_energy_mwh")).abs()
            / pl.col("true_energy_mwh"),
            battery=pl.col("nameplate_hours").cast(pl.Utf8),
        )
        .with_columns(difference=pl.col("post") - pl.col("rule"))
    )
    mean, low, high = cluster_bootstrap(
        frame=c4, value="difference", outer="series", inner="battery"
    )
    median, sd = DURATION_PRIORS["merchant"]
    prior_width = float(np.exp(np.log(median) + Z90 * sd) - np.exp(np.log(median) - Z90 * sd))
    widths = rung1.filter(pl.col("share") >= MIN_SHARE_C4).with_columns(
        ratio=(pl.col("merchant_duration_q95") - pl.col("merchant_duration_q05")) / prior_width
    )
    lines += [
        "",
        f"## C4: energy separately from power (planned){heading_suffix}",
        "",
        (
            f"For 1-hour and 4-hour nameplate batteries at shares of {MIN_SHARE_C4:.0%} and above "
            f"({c4.height} sums), the mean paired difference in relative energy error (posterior "
            f"median minus the rule 2 hours times the power median) is {mean:+.4f} (95% interval "
            f"{low:+.4f} to {high:+.4f}); the contrast {'holds' if high < 0 else 'fails'}, in "
            "the family only. For real batteries the energy medians are prior-driven and the "
            "energy reference is a lower bound, so no MWh claim transfers."
        ),
        "",
        (
            "Ratio of the duration's 90% posterior width to its prior width (a ratio near 1 means "
            "the aggregate taught the estimator nothing about duration):"
        ),
        "",
        table(
            widths.group_by("share", "nameplate_hours")
            .agg(median_ratio=pl.col("ratio").median(), n=pl.len())
            .sort("share", "nameplate_hours")
        ),
    ]
    return lines


def c5_section(*, rung3: pl.DataFrame) -> list[str]:
    """C5: fleets of real batteries against the coincident peak and the registered sum."""
    fleets = rung3.filter(
        pl.col("kind").str.starts_with("fleet") & (pl.col("merchant_power_median") > 0)
    )
    log2 = lambda x: pl.col(x).log(2.0)  # noqa: E731
    fleets = fleets.with_columns(
        difference=(log2("merchant_power_median") - log2("coincident_peak_mw")).abs()
        - (log2("merchant_power_median") - log2("true_power_mw")).abs(),
        battery=pl.col("unit"),
        below_peak=pl.col("merchant_power_median") < pl.col("coincident_peak_mw"),
        below_registered=pl.col("merchant_power_median") < pl.col("true_power_mw"),
    )
    mean, low, high = cluster_bootstrap(
        frame=fleets, value="difference", outer="series", inner="battery"
    )
    return [
        "## C5: fleets of real batteries (planned)",
        "",
        (
            f"On {fleets.height} sums of fleets of 2, 4, and 8 public batteries, the mean paired "
            "difference in |log2(P_hat / coincident peak)| minus |log2(P_hat / registered sum)| "
            f"is {mean:+.4f} (95% cluster-bootstrap interval {low:+.4f} to {high:+.4f}), with "
            f"P_hat the merchant class's posterior median power; the contrast "
            f"{'holds' if high < 0 else 'fails'}."
        ),
        "",
        (
            f"**C5 is degenerate, and its pass is not evidence for the coincident-peak reading.** "
            f"The posterior median lies below the coincident peak in "
            f"{fleets['below_peak'].mean():.1%} of the {fleets.height} fleet sums and below the "
            f"registered sum in {fleets['below_registered'].mean():.1%}. Whenever the estimate is "
            "below both references, the nearer one in log terms is the smaller, which is always "
            "the coincident peak, so an estimate near zero would pass C5."
        ),
        "",
        "Median ratio of P_hat to the registered sum and to the coincident peak, by fleet size:",
        "",
        table(
            fleets.group_by("kind")
            .agg(
                to_registered=(pl.col("merchant_power_median") / pl.col("true_power_mw")).median(),
                to_coincident_peak=(
                    pl.col("merchant_power_median") / pl.col("coincident_peak_mw")
                ).median(),
                below_peak=pl.col("below_peak").mean(),
                below_registered=pl.col("below_registered").mean(),
                n=pl.len(),
            )
            .sort("kind")
        ),
    ]


def unit_power_quantiles(*, frame: pl.DataFrame, unit: int) -> pl.DataFrame:
    """Add the 5, 50, and 95% quantiles of one unit's power from the saved Laplace approximation.

    The marginal of `log P` is Gaussian, so the quantiles of the power are exact.

    Args:
        frame: Rows with the saved `theta` and `covariance`.
        unit: The unit's index in the parameter vector.

    Returns:
        The frame with `unit_power_q05`, `unit_power_median`, and `unit_power_q95`.
    """
    n = len(frame["theta"][0])
    theta = np.array(frame["theta"].to_list())
    variance = np.array(frame["covariance"].to_list()).reshape(-1, n, n)[:, unit, unit]
    centre, sd = theta[:, unit], np.sqrt(np.where(np.isfinite(variance), variance, np.nan))
    return frame.with_columns(
        unit_power_q05=pl.Series(np.exp(centre - Z90 * sd)),
        unit_power_median=pl.Series(np.exp(centre)),
        unit_power_q95=pl.Series(np.exp(centre + Z90 * sd)),
        unit_power_over_prior_scale=pl.Series(np.exp(centre)),
    )


def fleet_units_section(*, rung2: pl.DataFrame) -> list[str]:
    """Which units of the domestic class the aggregate identifies in rung 2."""
    if "true_agile_power_mw" not in rung2.columns:
        return []
    prior_scale = pl.col("true_power_mw") / pl.col("share") * 0.2
    lines = [
        "## Rung 2: which domestic units the aggregate identifies (exploratory)",
        "",
        (
            "A fixed tariff window repeats every day, so the monthly baseline absorbs it, and the "
            "likelihood is flat in its power. The table shows each unit's fitted power divided by "
            "the scale of its prior (0.2 times the series' 99th percentile absolute flow): a ratio "
            "that stays near the same value as the fleet grows means the prior, not the aggregate, "
            "sets the power. The Agile tariff follows a price that changes every day."
        ),
        "",
    ]
    rows = []
    for unit, name in (
        (1, "agile"),
        (3, "intelligent_octopus_go"),
        (4, "octopus_go"),
        (5, "octopus_flux"),
    ):
        frame = unit_power_quantiles(frame=rung2, unit=unit)
        rows.append(
            frame.group_by("share")
            .agg(
                (pl.col("unit_power_median") / prior_scale).median().alias("power_over_prior_scale")
            )
            .with_columns(unit=pl.lit(name))
        )
    lines += [
        table(
            pl.concat(rows)
            .pivot(on="unit", index="share", values="power_over_prior_scale")
            .sort("share")
        )
    ]
    agile = unit_power_quantiles(frame=rung2, unit=1).with_columns(
        truth=pl.col("true_agile_power_mw"),
        covered=(pl.col("unit_power_q05") <= pl.col("true_agile_power_mw"))
        & (pl.col("true_agile_power_mw") <= pl.col("unit_power_q95")),
    )
    lines += [
        (
            "Coverage of the Agile unit's 90% interval against the simulated Agile homes' rated "
            "power, and the median ratio of its power to the truth:"
        ),
        "",
        table(
            agile.group_by("share")
            .agg(
                n=pl.len(),
                covered=pl.col("covered").mean(),
                median_ratio=(pl.col("unit_power_median") / pl.col("truth")).median(),
            )
            .sort("share")
        ),
    ]
    return lines


def rung3_section(
    *, rung3: pl.DataFrame, rung1: pl.DataFrame, limits: dict[str, float]
) -> list[str]:
    """Real public batteries against registered power and energy and the no-battery level."""
    series_p99 = {label: p99_flow(v) for label, v in demand_series().items()}
    nulls = rung1.filter(pl.col("share") == 0).with_columns(
        p99=pl.col("series").replace_strict(series_p99, return_dtype=pl.Float64)
    )
    null_fraction = float(np.median((nulls["merchant_power_median"] / nulls["p99"]).to_numpy()))
    null_bf = float(np.median(nulls["log_bayes_factor"].to_numpy()))
    default = float(np.quantile(list(limits.values()), 1 - FALSE_ALARM_RATE))
    flagged = season(frame=flag(frame=rung3, threshold=limits, default=default))
    frame = rung3.with_columns(true_energy_mwh=pl.col("true_energy_reference_mwh"))
    frame = with_errors(frame=frame, power="merchant_power", energy="merchant_energy")
    by_share = rung3.with_columns(
        median_over_p99=pl.col("merchant_power_median")
        / (pl.col("true_power_mw") / pl.col("share"))
    )
    outside = flagged.filter(pl.col("season") == "Sep-May")
    flags_by_season = flagged.group_by("share", "season").agg(
        flagged=pl.col("flagged").sum(), sums=pl.len()
    )
    return [
        "## Rung 3: real public batteries against registered power and energy (exploratory)",
        "",
        (
            "**The posterior barely moves when a real public battery is added.** The posterior "
            f"median of the merchant class's power, as a fraction of the series' 99th percentile "
            f"absolute flow, is {null_fraction:.3f} with no battery (rung 1's {nulls.height} null "
            "blocks) and:"
        ),
        "",
        table(
            by_share.group_by("share")
            .agg(
                median_power_over_p99=pl.col("median_over_p99").median(),
                median_log_bayes_factor=pl.col("log_bayes_factor").median(),
                n=pl.len(),
            )
            .sort("share")
        ),
        f"The median log Bayes factor with no battery is {null_bf:.2f}.",
        "",
        "Flags at the nominal 5% threshold, by share and season:",
        "",
        table(flags_by_season.sort("share", "season")),
        "The same rates with Clopper-Pearson intervals (sums share their series-blocks):",
        "",
        table(rate_table(frame=flagged, by=["share", "season"])),
        (
            f"Outside Jun-Aug, {int(outside['flagged'].sum())} of {outside.height} sums are "
            f"flagged ({outside['flagged'].mean():.1%}), against the nominal 5% of a null."
        ),
        "",
        (
            "Coverage of the 90% and 50% intervals. The power interval for a 5% share is the "
            "prior's, which is why it covers; the energy reference is the smallest holding "
            "capacity of the battery's metered output (a lower bound on the energy capacity), so "
            "the energy coverage is against a lower bound, not a measured capacity."
        ),
        "",
        table(coverage(frame=frame, by=["kind", "share"])),
        (
            "The coverage is a property of the series-block more than of the battery: the 23 "
            "units share 36 series-blocks at each share. The number of distinct 90% power "
            "coverages across the five kinds at each share:"
        ),
        "",
        table(
            coverage(frame=frame, by=["kind", "share"])
            .group_by("share")
            .agg(distinct_power_90=pl.col("power_90").n_unique(), n_kinds=pl.len())
            .sort("share")
        ),
        "Median relative error of the merchant class's power median against the registered power:",
        "",
        table(
            frame.group_by("kind", "share")
            .agg(
                power_error=pl.col("power_error").median(),
                energy_error=pl.col("energy_error").median(),
            )
            .sort("kind", "share")
        ),
    ]


def rung4_section(*, rung4: pl.DataFrame, nulls: pl.DataFrame) -> list[str]:
    """NGED battery A inside a bulk supply point's flow."""
    finite = nulls.filter(pl.col("log_bayes_factor").is_finite())["log_bayes_factor"].to_numpy()
    threshold = float(np.quantile(finite, 1 - FALSE_ALARM_RATE))
    frame = rung4.with_columns(
        flagged=pl.col("log_bayes_factor").is_finite() & (pl.col("log_bayes_factor") > threshold),
        block_name=pl.col("block").replace_strict(BLOCK_NAMES_BY_INDEX),
    )
    theta = np.array(frame["theta"].to_list())
    frame = frame.with_columns(
        **{
            f"{name}_power_over_prior_scale": pl.Series(np.exp(theta[:, index]))
            for index, name in enumerate(UNIT_NAMES)
        }
    )
    unit_columns = [f"{name}_power_over_prior_scale" for name in UNIT_NAMES]
    summer = frame.filter(pl.col("block") == SUMMER_BLOCK)
    other = frame.filter(pl.col("block") != SUMMER_BLOCK)
    matched = frame.filter(pl.col("multiple") == 0)
    nged = nged_series()
    flow_p99 = p99_flow(nged["BSP1"])
    battery_p99 = p99_flow(nged["battery_A"])
    shares = ", ".join(
        f"multiple {m}: {m * battery_p99 / flow_p99:.1%}"
        for m in sorted(frame["multiple"].unique().to_list())
    )
    return [
        "## Rung 4: NGED battery A inside a bulk supply point's flow (exploratory, one site)",
        "",
        (
            f"Detection threshold: {threshold:.2f} (the 95th percentile of the {len(finite)} null "
            "blocks of the other series in rung 1, because BSP1 has no nulls of its own). The "
            "matched null (multiple 0) is a null only if the battery's meter captures the battery "
            "fully inside BSP1's flow: the half-hour changes correlate at -0.29 at lag 0 and at "
            "+0.13 at lags of one half-hour either way (`report_inputs.md`)."
        ),
        "",
        (
            "The battery's metered 99th-percentile output at each multiple, as a share of BSP1's "
            f"99th-percentile flow: {shares}."
        ),
        "",
        (
            "**NGED battery A is not detected, and its power is not recovered, at any multiple of "
            "its metered output.** The multiples run from 0 (the metered output added back, a "
            "matched null) to 11 times the metered output; the truth is the multiple times the "
            f"metered 99th percentile output. From September to May the log Bayes factor ranges "
            f"from {other['log_bayes_factor'].min():.1f} to {other['log_bayes_factor'].max():.1f} "
            f"across all multiples and {int(other['flagged'].sum())} of {other.height} blocks are "
            f"flagged. In Jun-Aug {int(summer['flagged'].sum())} of {summer.height} blocks are "
            f"flagged ({int(matched['flagged'].sum())} of {matched.height} matched nulls, at "
            "multiple 0). The merchant power posterior median ranges from "
            f"{frame['merchant_power_median'].min():.2f} to "
            f"{frame['merchant_power_median'].max():.2f} MW while the truth ranges from "
            f"{frame['true_power_mw'].min():.1f} to {frame['true_power_mw'].max():.1f} MW."
        ),
        "",
        table(
            frame.select(
                "multiple",
                "block_name",
                "log_bayes_factor",
                "flagged",
                "true_power_mw",
                "merchant_power_q05",
                "merchant_power_median",
                "merchant_power_q95",
                "true_energy_reference_mwh",
                "merchant_energy_q05",
                "merchant_energy_median",
                "merchant_energy_q95",
            ).sort("multiple", "block_name")
        ),
        (
            "Jun-Aug log Bayes factor and the fitted power of every unit as a multiple of its "
            "prior scale, by multiple, to show which unit absorbs the factor's rise with the "
            "multiple:"
        ),
        "",
        table(summer.select("multiple", "log_bayes_factor", "tau", *unit_columns).sort("multiple")),
    ]


def rung5_section(*, rung5: pl.DataFrame) -> list[str]:
    """The primary screen: each primary's real log Bayes factor ranked against 12 placebo sets."""
    yearly = (
        rung5.group_by("series", "template_set", "set_kind")
        .agg(log_bayes_factor=pl.col("log_bayes_factor").sum(), blocks=pl.len())
        .sort("series", "template_set")
    )
    ranked = yearly.with_columns(
        rank=pl.col("log_bayes_factor").rank(method="min", descending=True).over("series")
    )
    real = ranked.filter(pl.col("template_set") == "real").select(
        "series", real_log_bayes_factor="log_bayes_factor", real_rank="rank"
    )
    price_only = (
        yearly.filter(pl.col("set_kind").is_in(["real", "price_placebo"]))
        .with_columns(
            rank=pl.col("log_bayes_factor").rank(method="min", descending=True).over("series")
        )
        .filter(pl.col("template_set") == "real")
        .select("series", real_rank_among_price_placebos="rank")
    )
    best_placebo = (
        yearly.filter(pl.col("template_set") != "real")
        .group_by("series")
        .agg(best_placebo_log_bayes_factor=pl.col("log_bayes_factor").max())
    )
    register = storage_presence_by_primary()
    presence = pl.DataFrame(
        {
            "series": list(register),
            "register_connected_storage": [v["connected_storage"] for v in register.values()],
            "register_accepted_storage": [v["accepted_storage"] for v in register.values()],
        }
    )
    screen = (
        real.join(price_only, on="series")
        .join(best_placebo, on="series")
        .join(presence, on="series")
        .sort("series")
    )
    first_all = int(screen.filter(pl.col("real_rank") == 1).height)
    first_price = int(screen.filter(pl.col("real_rank_among_price_placebos") == 1).height)
    return [
        "## Rung 5: the screen of the 8 primaries with a within-primary placebo (exploratory)",
        "",
        (
            "Each primary's log Bayes factor summed over the four blocks, for the real template "
            "set and for 12 placebo sets (tariff windows moved by -3 to +3 hours, and N2EX and "
            "Agile prices taken from 6 other weeks). Under the null the real set's rank is "
            "uniform over 13 only if the 12 placebos are exchangeable with the real set, and they "
            "are not (the early-window placebos rank last in most primaries), so the screen's "
            "false-first rate is measured on rung 1's null lanes in the next section. The MW and "
            "MWh posteriors of the domestic and commercial classes are not printed: rung 2 shows "
            "they sit at the prior."
        ),
        "",
        table(screen),
        (
            f"**The real template set ranks first of 13 for {first_all} of {screen.height} "
            f"primaries, and first among the 7 sets that differ only in their prices for "
            f"{first_price}.** The power of each form of the screen is in the next section."
        ),
        "",
        "Every set's summed log Bayes factor:",
        "",
        table(ranked.select("series", "template_set", "log_bayes_factor", "rank")),
    ]


def grid_section(*, rung1: pl.DataFrame, grid: pl.DataFrame) -> list[str]:
    """The grid estimator beside the differentiable estimator on rung 1."""
    keys = ["series", "nameplate_hours", "share", "block"]
    both = (
        rung1.filter(pl.col("share") > 0)
        .join(grid.filter(pl.col("share") > 0), on=keys, how="inner", suffix="_grid")
        .with_columns(
            dp_power_error=(pl.col("merchant_power_median") - pl.col("true_power_mw_grid"))
            / pl.col("true_power_mw_grid"),
            grid_power_error=(pl.col("grid_power_median") - pl.col("true_power_mw_grid"))
            / pl.col("true_power_mw_grid"),
            dp_energy_error=(pl.col("merchant_energy_median") - pl.col("true_energy_mwh_grid"))
            / pl.col("true_energy_mwh_grid"),
            grid_energy_error=(pl.col("grid_energy_median") - pl.col("true_energy_mwh_grid"))
            / pl.col("true_energy_mwh_grid"),
            dp_power_in_90=(pl.col("merchant_power_q05") <= pl.col("true_power_mw_grid"))
            & (pl.col("true_power_mw_grid") <= pl.col("merchant_power_q95")),
            grid_power_in_90=(pl.col("grid_power_q05") <= pl.col("true_power_mw_grid"))
            & (pl.col("true_power_mw_grid") <= pl.col("grid_power_q95")),
            dp_energy_in_90=(pl.col("merchant_energy_q05") <= pl.col("true_energy_mwh_grid"))
            & (pl.col("true_energy_mwh_grid") <= pl.col("merchant_energy_q95")),
            grid_energy_in_90=(pl.col("grid_energy_q05") <= pl.col("true_energy_mwh_grid"))
            & (pl.col("true_energy_mwh_grid") <= pl.col("grid_energy_q95")),
            battery=pl.col("nameplate_hours").cast(pl.Utf8),
        )
        .with_columns(difference=pl.col("dp_power_error").abs() - pl.col("grid_power_error").abs())
    )
    big = both.filter(pl.col("share") >= MIN_SHARE_C3)
    mean, low, high = cluster_bootstrap(
        frame=big, value="difference", outer="series", inner="battery"
    )
    return [
        "## The grid estimator beside the differentiable estimator on rung 1 (exploratory)",
        "",
        (
            f"The grid estimator ran on all {both.height} rung 1 sums (the same aggregates). "
            "The `*_signed_median_power_error` columns are signed relative errors (positive is an "
            "overestimate); the paired difference uses absolute errors. At "
            f"shares of {MIN_SHARE_C3:.0%} and above, the mean paired difference in absolute "
            f"relative power error (differentiable minus grid) is {mean:+.4f} (95% interval "
            f"{low:+.4f} to {high:+.4f}); the differentiable estimator's medians are "
            f"{'no more accurate than' if high >= 0 else 'more accurate than'} the grid's, and its "
            "gain is the coverage of the 90% intervals in the table."
        ),
        "",
        table(
            both.group_by("share")
            .agg(
                n=pl.len(),
                dp_signed_median_power_error=pl.col("dp_power_error").median(),
                grid_signed_median_power_error=pl.col("grid_power_error").median(),
                dp_median_absolute_power_error=pl.col("dp_power_error").abs().median(),
                grid_median_absolute_power_error=pl.col("grid_power_error").abs().median(),
                dp_power_90=pl.col("dp_power_in_90").mean(),
                grid_power_90=pl.col("grid_power_in_90").mean(),
                dp_energy_90=pl.col("dp_energy_in_90").mean(),
                grid_energy_90=pl.col("grid_energy_in_90").mean(),
            )
            .sort("share")
        ),
    ]


def timing_section(*, frames: dict[str, pl.DataFrame]) -> list[str]:
    """GPU timings of each rung."""
    rows = [
        {
            "rung": name,
            "sums": frame.height,
            "block_batches_seconds_total": float(
                frame.group_by("block").agg(s=pl.col("fit_seconds").first())["s"].sum()
            ),
        }
        for name, frame in frames.items()
    ]
    return ["## Timings of the differentiable estimator", "", table(pl.DataFrame(rows))]


def main() -> None:
    """Write `report.md`."""
    read = lambda name: pl.read_parquet(OUTPUT_DIR / f"{name}.parquet")  # noqa: E731
    rung1, rung2 = read("rung1_posteriors"), read("rung2_posteriors")
    shifted, rung3 = read("rung2_shifted_posteriors"), read("rung3_posteriors")
    rung4, rung5 = read("rung4_posteriors"), read("rung5_posteriors")
    steps, grid = read("rung1_step_tail"), read("rung1_grid_posteriors")
    limits = thresholds(nulls=rung1.filter(pl.col("share") == 0))
    outside = {
        policy: read(f"rung1b_{policy}_posteriors") for policy in ("rank_rule", "noisy_price")
    }
    sensitivity1, sensitivity2 = (
        read("rung1_sensitivity_posteriors"),
        read("rung2_sensitivity_posteriors"),
    )
    limits_sensitivity = thresholds(nulls=sensitivity1.filter(pl.col("share") == 0))
    lines = ["# Report: how well can an aggregate reveal an unmetered battery?", ""]
    lines += review.positive_control_clean_section(draws=read("positive_control_clean_draws"))
    lines += c1_section(rung1=rung1, limits=limits)
    lines += detection_section(
        rung1=rung1, rung2=rung2, shifted=shifted, rung3=rung3, limits=limits
    )
    lines += review.detection_curve_section(
        families={
            "rung1 (in the family)": rung1.filter(pl.col("share") > 0),
            "rank_rule (outside)": outside["rank_rule"],
            "noisy_price (outside)": outside["noisy_price"],
            "rung3 (real public batteries)": rung3,
        },
        steps=steps,
        limits=limits,
        nulls=rung1.filter(pl.col("share") == 0),
    )
    lines += review.rung1b_section(rung1=rung1, others=outside, limits=limits)
    lines += calibration_section(rung1=rung1, rung2=rung2)
    lines += review.low_share_posterior_section(rung1=rung1)
    lines += review.tuning_series_section(rung1=rung1)
    lines += fleet_units_section(rung2=rung2)
    lines += review.rung2_controls_section(
        rung2=rung2,
        shifted=shifted,
        coarse_real=read("rung2_coarse_real_posteriors"),
        coarse_placebo=read("rung2_coarse_placebo_posteriors"),
        limits=limits,
    )
    lines += c3_c4_section(rung1=rung1, steps=steps)
    lines += c5_section(rung3=rung3)
    lines += rung3_section(rung3=rung3, rung1=rung1, limits=limits)
    lines += review.rung3_replica_section(replica=read("rung3_replica_posteriors"))
    lines += rung4_section(rung4=rung4, nulls=rung1.filter(pl.col("share") == 0))
    lines += rung5_section(rung5=rung5)
    lines += review.screen_power_section(screen=read("screen_power_posteriors"))
    lines += grid_section(rung1=rung1, grid=grid)
    suffix = " (second setting)"
    lines += ["# The second setting (doubled duration priors, wider efficiency prior)", ""]
    lines += c1_section(rung1=sensitivity1, limits=limits_sensitivity, heading_suffix=suffix)
    lines += calibration_section(rung1=sensitivity1, rung2=sensitivity2, heading_suffix=suffix)
    lines += c3_c4_section(rung1=sensitivity1, steps=steps, heading_suffix=suffix)
    lines += detection_section(
        rung1=sensitivity1,
        rung2=sensitivity2,
        shifted=None,
        rung3=None,
        limits=limits_sensitivity,
        heading_suffix=suffix,
    )
    lines += review.start_disagreement_section(
        frames={
            "rung1": rung1,
            "rung2": rung2,
            "rung3": rung3,
            "rung1b_rank_rule": outside["rank_rule"],
            "rung1b_noisy_price": outside["noisy_price"],
        }
    )
    lines += timing_section(
        frames={"rung1": rung1, "rung2": rung2, "rung3": rung3, "rung4": rung4, "rung5": rung5}
    )
    lines += review.notes_section()
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
