"""Print every table the page quotes into `report.md`.

Reads the rungs' saved posteriors (`rung*_posteriors.parquet`), the step tails, the grid
estimator's rung 1 posteriors, and the positive control, and writes: the thresholds and false-alarm
rate (C1), the detection rates, the calibration (C2), the posterior against the step tail (C3), the
energy against the rule (C4), the fleets (C5), the negative controls, rung 4, the primary screen
with its within-primary placebo, and the comparison with the grid estimator. Intervals on errors,
coverages, and rates resample whole demand series and whole batteries (a two-level cluster
bootstrap, 2,000 resamples); rates also carry Clopper-Pearson intervals.

Run: `uv run python studies/unmetered_battery_capacity/capacity_report.py`
"""

from typing import Final

import numpy as np
import polars as pl
from capacity_inputs import OUTPUT_DIR, storage_presence_by_primary
from capacity_templates import DURATION_PRIORS
from scipy.stats import beta, binomtest

N_RESAMPLES: Final[int] = 2000
SEED: Final[int] = 20261013
FALSE_ALARM_RATE: Final[float] = 0.05
MIN_SHARE_C3: Final[float] = 0.05
MIN_SHARE_C4: Final[float] = 0.10
Z90: Final[float] = 1.645


def table(frame: pl.DataFrame) -> str:
    """Return a frame as a pipe-separated table with rounded floats."""
    return frame.with_columns(pl.col(pl.Float64).round(4)).write_csv(separator="|")


def clopper_pearson(*, count: int, total: int, level: float = 0.95) -> tuple[float, float]:
    """Return the Clopper-Pearson interval of a binomial proportion."""
    alpha = 1 - level
    low = 0.0 if count == 0 else float(beta.ppf(alpha / 2, count, total - count + 1))
    high = 1.0 if count == total else float(beta.ppf(1 - alpha / 2, count + 1, total - count))
    return low, high


def cluster_bootstrap(
    *, frame: pl.DataFrame, value: str, outer: str, inner: str, seed: int = SEED
) -> tuple[float, float, float]:
    """Return the mean of a column and its 95% interval, resampling outer then inner clusters.

    Whole demand series (outer) are resampled with replacement, then whole batteries (inner) within
    each resampled series, as the prior study resampled whole aggregates.

    Args:
        frame: The rows.
        value: The column to average.
        outer: The outer cluster column (the demand series).
        inner: The inner cluster column (the battery).
        seed: Seeds the resampling.

    Returns:
        The mean, and the 2.5% and 97.5% quantiles of the resampled means.
    """
    rows = (
        frame.filter(pl.col(value).is_finite())
        .group_by(outer, inner)
        .agg(total=pl.col(value).sum(), count=pl.len())
    )
    by_outer: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, part in rows.group_by(outer):
        by_outer[str(key[0])] = (part["total"].to_numpy(), part["count"].to_numpy())
    names = list(by_outer)
    rng = np.random.default_rng(seed)
    means = np.empty(N_RESAMPLES)
    for r in range(N_RESAMPLES):
        total = count = 0.0
        for name in rng.choice(names, size=len(names), replace=True):
            sums, counts = by_outer[str(name)]
            pick = rng.integers(0, len(sums), size=len(sums))
            total += sums[pick].sum()
            count += counts[pick].sum()
        means[r] = total / count
    overall = float(rows["total"].sum() / rows["count"].sum())
    low, high = np.quantile(means, [0.025, 0.975])
    return overall, float(low), float(high)


def thresholds(*, nulls: pl.DataFrame) -> dict[str, float]:
    """Return each series' detection threshold: the 95th percentile of the other series' nulls.

    Args:
        nulls: Rung 1's rows with no added battery; columns `series` and `log_bayes_factor`.

    Returns:
        By series label, the threshold on the log Bayes factor.
    """
    finite = nulls.filter(pl.col("log_bayes_factor").is_finite())
    out = {}
    for label in nulls["series"].unique().to_list():
        others = finite.filter(pl.col("series") != label)["log_bayes_factor"].to_numpy()
        out[label] = float(np.quantile(others, 1 - FALSE_ALARM_RATE))
    return out


def flag(*, frame: pl.DataFrame, threshold: dict[str, float], default: float) -> pl.DataFrame:
    """Add a boolean `flagged` column: the log Bayes factor is finite and above its threshold."""
    return frame.with_columns(
        threshold=pl.col("series").replace_strict(
            threshold, default=default, return_dtype=pl.Float64
        )
    ).with_columns(
        flagged=pl.col("log_bayes_factor").is_finite()
        & (pl.col("log_bayes_factor") > pl.col("threshold"))
    )


def rate_table(*, frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Return the share of flagged sums per group with a Clopper-Pearson interval."""
    grouped = frame.group_by(by).agg(flagged=pl.col("flagged").sum(), total=pl.len()).sort(by)
    lows, highs = zip(
        *[
            clopper_pearson(count=int(f), total=int(t))
            for f, t in zip(grouped["flagged"], grouped["total"], strict=True)
        ],
        strict=True,
    )
    return grouped.with_columns(
        rate=pl.col("flagged") / pl.col("total"), ci_low=pl.Series(lows), ci_high=pl.Series(highs)
    )


def with_errors(*, frame: pl.DataFrame, power: str, energy: str) -> pl.DataFrame:
    """Add relative errors, interval membership, and a finite-interval flag for one class."""
    return frame.with_columns(
        power_error=(pl.col(f"{power}_median") - pl.col("true_power_mw")).abs()
        / pl.col("true_power_mw"),
        energy_error=(pl.col(f"{energy}_median") - pl.col("true_energy_mwh")).abs()
        / pl.col("true_energy_mwh"),
        power_in_90=(pl.col(f"{power}_q05") <= pl.col("true_power_mw"))
        & (pl.col("true_power_mw") <= pl.col(f"{power}_q95")),
        power_in_50=(pl.col(f"{power}_q25") <= pl.col("true_power_mw"))
        & (pl.col("true_power_mw") <= pl.col(f"{power}_q75")),
        energy_in_90=(pl.col(f"{energy}_q05") <= pl.col("true_energy_mwh"))
        & (pl.col("true_energy_mwh") <= pl.col(f"{energy}_q95")),
        energy_in_50=(pl.col(f"{energy}_q25") <= pl.col("true_energy_mwh"))
        & (pl.col("true_energy_mwh") <= pl.col(f"{energy}_q75")),
    )


def coverage(*, frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Return the coverage of the 50% and 90% intervals of power and energy per group.

    Sums without a covariance (no interval) count as not covering.
    """
    return (
        frame.group_by(by)
        .agg(
            n=pl.len(),
            with_interval=pl.col("has_interval").sum(),
            power_90=pl.col("power_in_90").fill_null(False).mean(),
            power_50=pl.col("power_in_50").fill_null(False).mean(),
            energy_90=pl.col("energy_in_90").fill_null(False).mean(),
            energy_50=pl.col("energy_in_50").fill_null(False).mean(),
        )
        .sort(by)
    )


def c1_section(*, rung1: pl.DataFrame, limits: dict[str, float]) -> list[str]:
    """C1: the false-alarm rate on the blocks with no added battery."""
    nulls = flag(frame=rung1.filter(pl.col("share") == 0), threshold=limits, default=np.nan)
    count, total = int(nulls["flagged"].sum()), nulls.height
    test = binomtest(count, total, FALSE_ALARM_RATE, alternative="greater")
    low, high = clopper_pearson(count=count, total=total)
    return [
        "## C1: false alarms on the blocks with no added battery (planned)",
        "",
        (
            f"{count} of {total} blocks flagged ({count / total:.3f}; Clopper-Pearson 95% interval "
            f"{low:.3f} to {high:.3f}). One-sided exact binomial test against 5%: p = "
            f"{test.pvalue:.3f}; the contrast {'holds' if test.pvalue >= 0.05 else 'fails'}. "
            f"{int(nulls['log_bayes_factor'].is_nan().sum() + nulls['log_bayes_factor'].is_null().sum())}"
            " null blocks have no log Bayes factor (no positive-definite Hessian) and count as not "
            "flagged."
        ),
        "",
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


def detection_section(
    *, rung1: pl.DataFrame, rung2: pl.DataFrame, shifted: pl.DataFrame, rung3: pl.DataFrame,
    limits: dict[str, float],
) -> list[str]:  # fmt: skip
    """Detection rates by share for rungs 1 to 3 and the shifted-window negative control."""
    default = float(np.quantile(list(limits.values()), 1 - FALSE_ALARM_RATE))
    lines = ["## Detection rates at the 5% false-alarm threshold (exploratory)", ""]
    for name, frame, by in (
        ("Rung 1, merchant batteries, by share", rung1.filter(pl.col("share") > 0), ["share"]),
        (
            "Rung 1, by share and nameplate duration",
            rung1.filter(pl.col("share") > 0),
            ["share", "nameplate_hours"],
        ),
        ("Rung 2, simulated fleets, real windows", rung2, ["share"]),
        ("Rung 2, simulated fleets, windows one hour early (negative control)", shifted, ["share"]),
        ("Rung 3, real public batteries, by share", rung3, ["share"]),
        ("Rung 3, by share and kind", rung3, ["share", "kind"]),
    ):
        lines += [
            f"**{name}.**",
            "",
            table(rate_table(frame=flag(frame=frame, threshold=limits, default=default), by=by)),
        ]
    return lines


def calibration_section(*, rung1: pl.DataFrame, rung2: pl.DataFrame) -> list[str]:
    """C2: the coverage of the credible intervals on the simulated known answers."""
    merchant = with_errors(
        frame=rung1.filter(pl.col("share") > 0), power="merchant_power", energy="merchant_energy"
    )
    fleets = with_errors(frame=rung2, power="domestic_power", energy="domestic_energy")
    lines = ["## C2: calibration of the credible intervals (planned)", ""]
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
        "**Rungs 1 and 2 together (the plan's judgement).**",
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


def c3_c4_section(*, rung1: pl.DataFrame, steps: pl.DataFrame) -> list[str]:
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
        "## C3: posterior median against the step tail, power error (planned)",
        "",
        (
            f"At shares of {MIN_SHARE_C3:.0%} and above ({c3.height} sums), the mean paired "
            f"difference in relative power error (posterior median minus step tail) is {mean:+.4f} "
            f"(95% cluster-bootstrap interval {low:+.4f} to {high:+.4f}); the contrast "
            f"{'holds' if high < 0 else 'fails'}."
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
        "## C4: energy separately from power (planned)",
        "",
        (
            f"For 1-hour and 4-hour nameplate batteries at shares of {MIN_SHARE_C4:.0%} and above "
            f"({c4.height} sums), the mean paired difference in relative energy error (posterior "
            f"median minus the rule 2 hours times the power median) is {mean:+.4f} (95% interval "
            f"{low:+.4f} to {high:+.4f}); the contrast {'holds' if high < 0 else 'fails'}."
        ),
        "",
        "Ratio of the duration's 90% posterior width to its prior width (a ratio near 1 means the "
        "aggregate taught the estimator nothing about duration):",
        "",
        table(
            widths.group_by("share", "nameplate_hours")
            .agg(median_ratio=pl.col("ratio").median(), n=pl.len())
            .sort("share", "nameplate_hours")
        ),
    ]
    return lines


def c5_section(*, rung3: pl.DataFrame) -> list[str]:
    """C5: fleets of real batteries are seen as their coincident peak."""
    fleets = rung3.filter(
        pl.col("kind").str.starts_with("fleet") & (pl.col("merchant_power_median") > 0)
    )
    log2 = lambda x: pl.col(x).log(2.0)  # noqa: E731
    fleets = fleets.with_columns(
        difference=(log2("merchant_power_median") - log2("coincident_peak_mw")).abs()
        - (log2("merchant_power_median") - log2("true_power_mw")).abs(),
        battery=pl.col("unit"),
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
            f"is {mean:+.4f} (95% cluster-bootstrap interval {low:+.4f} to {high:+.4f}), with P_hat "
            f"the merchant class's posterior median power; the contrast "
            f"{'holds' if high < 0 else 'fails'}."
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
                n=pl.len(),
            )
            .sort("kind")
        ),
    ]


def rung3_section(*, rung3: pl.DataFrame) -> list[str]:
    """Coverage and error of real public batteries against registered power and the energy reference."""
    frame = rung3.with_columns(true_energy_mwh=pl.col("true_energy_reference_mwh"))
    frame = with_errors(frame=frame, power="merchant_power", energy="merchant_energy")
    return [
        "## Rung 3: real public batteries against registered power and the energy reference (exploratory)",
        "",
        table(coverage(frame=frame, by=["kind", "share"])),
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


def rung4_section(*, rung4: pl.DataFrame, limits: dict[str, float]) -> list[str]:
    """NGED battery A inside a bulk supply point's flow."""
    threshold = float(np.quantile(list(limits.values()), 1 - FALSE_ALARM_RATE))
    frame = rung4.with_columns(
        flagged=pl.col("log_bayes_factor").is_finite() & (pl.col("log_bayes_factor") > threshold)
    )
    return [
        "## Rung 4: NGED battery A inside a bulk supply point's flow (exploratory, one site)",
        "",
        f"Detection threshold: {threshold:.2f} (the 95th percentile over the {len(limits)} series' means of thresholds).",
        "",
        table(
            frame.select(
                "multiple",
                "block",
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
            ).sort("multiple", "block")
        ),  # fmt: skip
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
    classes = (
        rung5.filter(pl.col("template_set") == "real")
        .group_by("series")
        .agg(
            merchant_mw=pl.col("merchant_power_point").mean(),
            commercial_mw=pl.col("commercial_and_industrial_power_point").mean(),
            domestic_mw=pl.col("domestic_power_point").mean(),
            merchant_mwh=pl.col("merchant_energy_point").mean(),
        )
    )
    screen = (
        real.join(best_placebo, on="series")
        .join(presence, on="series")
        .join(classes, on="series")
        .sort("series")
    )
    return [
        "## Rung 5: the screen of the 8 primaries with a within-primary placebo (exploratory)",
        "",
        (
            "Each primary's log Bayes factor summed over the four blocks, for the real template set "
            "and for 12 placebo sets (tariff windows moved by -3 to +3 hours, and N2EX and Agile "
            "prices taken from 6 other weeks). A primary shows evidence of a battery when its real "
            "set ranks first of 13; 1 in 13 would rank first by chance."
        ),
        "",
        table(screen),
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
            f"The grid estimator ran on all {both.height} rung 1 sums (the same aggregates). At "
            f"shares of {MIN_SHARE_C3:.0%} and above, the mean paired difference in absolute "
            f"relative power error (differentiable minus grid) is {mean:+.4f} (95% interval "
            f"{low:+.4f} to {high:+.4f})."
        ),
        "",
        table(
            both.group_by("share")
            .agg(
                n=pl.len(),
                dp_median_power_error=pl.col("dp_power_error").median(),
                grid_median_power_error=pl.col("grid_power_error").median(),
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
    lines = ["# Report: how well can an aggregate reveal an unmetered battery?", ""]
    lines += c1_section(rung1=rung1, limits=limits)
    lines += detection_section(
        rung1=rung1, rung2=rung2, shifted=shifted, rung3=rung3, limits=limits
    )
    lines += calibration_section(rung1=rung1, rung2=rung2)
    lines += c3_c4_section(rung1=rung1, steps=steps)
    lines += c5_section(rung3=rung3)
    lines += rung3_section(rung3=rung3)
    lines += rung4_section(rung4=rung4, limits=limits)
    lines += rung5_section(rung5=rung5)
    lines += grid_section(rung1=rung1, grid=grid)
    lines += timing_section(
        frames={"rung1": rung1, "rung2": rung2, "rung3": rung3, "rung4": rung4, "rung5": rung5}
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
