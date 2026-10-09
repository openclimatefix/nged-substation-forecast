"""Print every table the embedded-battery forecast page quotes, from the saved fits.

Reads `fits/<setting>/<issue>/*.parquet` and nothing else that varies, so the tables can be rebuilt
without a refit. Writes `forecast_report.md` and the machine-readable tables under
`report_tables/`.

Scores. Every loss is a percentage of the battery's own 99th-percentile absolute output ("points of
p99"). CRPS is the gap-weighted sum of pinball losses over the 13 delivery levels. Intervals come
from `studies.bootstrap`, which resamples whole calendar months and a fitting seed, 2,000 times,
with the months shared across batteries. Skill is one minus the ratio of mean CRPS values, and its
interval is a month resampling of that ratio (`month_skill_interval`).

Run: `uv run python studies/embedded_battery_forecast/forecast_report.py`.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from forecast_arms import arm_definitions, issue_cutoff_lines
from forecast_fit import FIT_VARIANT, LEVELS, Q_COLUMNS, SettingType, arm_file
from forecast_inputs import load_physical_notifications
from forecast_results import load_losses
from forecast_runner import NGED_BATTERY_A_FILE_ID, battery_for, lead_parties, testbed_ids
from studies.battery_forecast import SYMMETRIC_BANDS, IssueType
from studies.bootstrap import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP_RESAMPLES,
    bootstrap_absolute,
    bootstrap_difference,
)
from studies.party_bootstrap import arm_cells, party_month_difference, party_month_share
from studies.sources import EMBEDDED_BATTERY_FORECAST_DIR

VARIANT_SUFFIX: Final[str] = "" if FIT_VARIANT == "as_written" else f"_{FIT_VARIANT}"
TABLES_DIR: Final[Path] = EMBEDDED_BATTERY_FORECAST_DIR / f"report_tables{VARIANT_SUFFIX}"
REPORT_PATH: Final[Path] = EMBEDDED_BATTERY_FORECAST_DIR / f"forecast_report{VARIANT_SUFFIX}.md"
IDLE_DROPPED_PLANNED_PATH: Final[Path] = (
    EMBEDDED_BATTERY_FORECAST_DIR / "report_tables_idle_dropped" / "planned_contrasts.parquet"
)
CLIM: Final[str] = "clim"
LEAD_BIN_HOURS: Final[int] = 6
TRIVIAL_FPN_ZERO_SHARE: Final[float] = 0.8
"""A battery whose Physical Notification is exactly zero in at least this share of the half-hours
that carry one has a notification that says almost nothing."""
NEAR_LINE_SHARE: Final[float] = 0.2
"""A bound within this share of the interval's width from zero is near the 5% line."""


@dataclass(frozen=True)
class Contrast:
    """A treatment arm compared with a reference arm at one issue time.

    Attributes:
        label: The contrast's label (`D1` to `D5` are planned; others are exploratory).
        issue: The issue type.
        treatment: The treatment arm's name.
        reference: The reference arm's name.
        question: What the contrast asks, for the report.
        planned: Whether the contrast was written into the plan before any result existed.
        part: `A` for the testbed pooled, `B` for NGED battery A.
    """

    label: str
    issue: IssueType
    treatment: str
    reference: str
    question: str
    planned: bool
    part: str


XGB = "xgb_quantile__"
PLANNED_CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast(
        "D1",
        "DA-late",
        f"{XGB}price_actual",
        CLIM,
        "Does a probabilistic forecast issued after the auctions beat the climatology?",
        True,
        "A",
    ),
    Contrast(
        "D2",
        "DA-late",
        f"{XGB}price_actual",
        f"{XGB}price_shuffled",
        "Is the gain the target day's price or the month's typical shape?",
        True,
        "A",
    ),
    Contrast(
        "D3",
        "DA-early",
        f"{XGB}price_model",
        f"{XGB}price_shuffled",
        "Does a price forecast available at 06:00 sharpen or shift the distribution usefully?",
        True,
        "A",
    ),
    Contrast(
        "D4",
        "ID-1h",
        f"{XGB}neighbour_fpn",
        f"{XGB}neighbour_fpn_shuffled",
        "Do other batteries' Physical Notifications improve the forecast for a battery whose "
        "own is unknown?",
        True,
        "A",
    ),
    Contrast(
        "D5",
        "DA-late",
        f"{XGB}price_actual",
        CLIM,
        "NGED battery A: does a forecast issued after the auctions beat the climatology?",
        True,
        "B",
    ),
)
EXPLORATORY_CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast(
        "X1",
        "DA-early",
        f"{XGB}price_actual",
        f"{XGB}price_shuffled",
        "DA-early upper bound: perfect price against shuffled.",
        False,
        "A",
    ),
    Contrast(
        "X2",
        "DA-early",
        f"{XGB}price_naive",
        f"{XGB}price_shuffled",
        "DA-early lower bound: the price a week earlier against shuffled.",
        False,
        "A",
    ),
    Contrast(
        "X3",
        "DA-early",
        f"{XGB}price_model",
        f"{XGB}price_naive",
        "Does the price model beat the price a week earlier as an input?",
        False,
        "A",
    ),
    Contrast(
        "X4",
        "DA-early",
        f"{XGB}price_model",
        CLIM,
        "DA-early: the price-model arm against the climatology.",
        False,
        "A",
    ),
    Contrast(
        "X5",
        "DA-late",
        f"{XGB}price_naive",
        f"{XGB}price_shuffled",
        "DA-late: the price a week earlier against shuffled.",
        False,
        "A",
    ),
    Contrast(
        "X6",
        "DA-late",
        f"{XGB}price_actual",
        "rank_conformal__price_actual",
        "DA-late: XGBoost against the conformal rank rule, both given the actual price.",
        False,
        "A",
    ),
    Contrast(
        "X7",
        "DA-late",
        "rank_conformal__price_actual",
        CLIM,
        "DA-late: the conformal rank rule against the climatology.",
        False,
        "A",
    ),
    Contrast(
        "X8",
        "DA-late",
        "persistence_conformal",
        CLIM,
        "DA-late: conformal persistence against the climatology.",
        False,
        "A",
    ),
    Contrast(
        "X9",
        "ID-1h",
        f"{XGB}own_fpn",
        f"{XGB}no_neighbour",
        "Rung A4: the battery's own Physical Notification against none.",
        False,
        "A",
    ),
    Contrast(
        "X10",
        "ID-1h",
        f"{XGB}neighbour_fpn",
        f"{XGB}no_neighbour",
        "Rung A5: neighbours' Physical Notifications against none.",
        False,
        "A",
    ),
    Contrast(
        "X11",
        "ID-1h",
        f"{XGB}neighbour_fpn_shuffled",
        f"{XGB}no_neighbour",
        "Rung A5 negative control: shuffled neighbours against none.",
        False,
        "A",
    ),
    Contrast(
        "X12",
        "ID-1h",
        f"{XGB}no_neighbour",
        CLIM,
        "ID-1h without any Physical Notification against the climatology.",
        False,
        "A",
    ),
    Contrast(
        "X13",
        "ID-1h",
        f"{XGB}no_neighbour",
        "persistence_conformal",
        "ID-1h without any Physical Notification against conformal persistence.",
        False,
        "A",
    ),
    Contrast(
        "X14",
        "ID-1h",
        f"{XGB}neighbour_fpn",
        CLIM,
        "ID-1h neighbours' Physical Notifications against the climatology.",
        False,
        "A",
    ),
    Contrast(
        "X15",
        "ID-1h",
        f"{XGB}neighbour_fpn_same_party",
        f"{XGB}no_neighbour",
        "Only batteries with another testbed unit of their lead party: same-party Physical "
        "Notifications against none.",
        False,
        "A",
    ),
    Contrast(
        "X16",
        "ID-1h",
        f"{XGB}neighbour_fpn_without_largest_party",
        f"{XGB}neighbour_fpn_without_largest_party_shuffled",
        "D4 repeated with the testbed's largest lead party left out of the neighbour set.",
        False,
        "A",
    ),
    Contrast(
        "X17",
        "ID-1h",
        f"{XGB}neighbour_fpn_without_largest_party",
        f"{XGB}no_neighbour",
        "Neighbours without the largest lead party against none.",
        False,
        "A",
    ),
    Contrast(
        "Y1",
        "DA-early",
        f"{XGB}price_model",
        f"{XGB}price_shuffled",
        "NGED battery A, DA-early: price forecast against shuffled.",
        False,
        "B",
    ),
    Contrast(
        "Y2",
        "DA-late",
        f"{XGB}price_actual",
        f"{XGB}price_shuffled",
        "NGED battery A, DA-late: actual price against shuffled.",
        False,
        "B",
    ),
    Contrast(
        "Y3",
        "ID-1h",
        f"{XGB}fleet_fpn",
        f"{XGB}fleet_fpn_shuffled",
        "Rung B2, NGED battery A: fleet Physical Notifications against shuffled.",
        False,
        "B",
    ),
    Contrast(
        "Y4",
        "ID-1h",
        f"{XGB}fleet_fpn",
        f"{XGB}no_neighbour",
        "Rung B2, NGED battery A: fleet Physical Notifications against none.",
        False,
        "B",
    ),
    Contrast(
        "Y5",
        "DA-early",
        f"{XGB}price_model",
        CLIM,
        "NGED battery A, DA-early: the price-model arm against the climatology.",
        False,
        "B",
    ),
    Contrast(
        "Y6",
        "ID-1h",
        f"{XGB}no_neighbour",
        CLIM,
        "NGED battery A, ID-1h: own telemetry only against the climatology.",
        False,
        "B",
    ),
)
"""Exploratory contrasts: every one is labelled exploratory on the page."""


def save_table(*, frame: pl.DataFrame, name: str) -> None:
    """Write a report table under `TABLES_DIR`."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(TABLES_DIR / name)


def table(*, headers: list[str], rows: list[list[str]]) -> list[str]:
    """Return a Markdown table."""
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def month_skill_interval(
    *, treatment: np.ndarray, reference: np.ndarray, months: np.ndarray
) -> tuple[float, float, float]:
    """Return the CRPS skill score (one minus the ratio of means) and a month-resampled interval.

    Args:
        treatment: Per-row loss of the forecast, any shape; rows are flattened.
        reference: Per-row loss of the reference, the same shape.
        months: The month label of each row, flattened to match.

    Returns:
        The skill, and the 2.5th and 97.5th percentiles over 2,000 resamples of whole months.
    """
    unique, index = np.unique(months, return_inverse=True)
    treatment_sum = np.bincount(index, weights=treatment, minlength=unique.size)
    reference_sum = np.bincount(index, weights=reference, minlength=unique.size)
    skill = 1.0 - treatment_sum.sum() / reference_sum.sum()
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.integers(0, unique.size, size=(N_BOOTSTRAP_RESAMPLES, unique.size))
    resampled = 1.0 - treatment_sum[draws].sum(axis=1) / reference_sum[draws].sum(axis=1)
    return float(skill), float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))


def contrast_losses(
    *, contrast: Contrast, setting: SettingType, batteries: list[str]
) -> pl.DataFrame:
    """Load the two arms of a contrast; a deterministic arm comes from the primary setting."""
    treatment = load_losses(
        setting=setting, issue=contrast.issue, arms=[contrast.treatment], batteries=batteries
    )
    reference_setting: SettingType = setting if contrast.reference.startswith(XGB) else "primary"
    reference = load_losses(
        setting=reference_setting,
        issue=contrast.issue,
        arms=[contrast.reference],
        batteries=batteries,
    )
    sites = sorted(set(treatment["site"].unique()) & set(reference["site"].unique()))
    return pl.concat([treatment, reference]).filter(pl.col("site").is_in(sites))


def significance(*, lower: float, upper: float) -> str:
    """Return whether an interval lies wholly below or above zero."""
    if upper < 0:
        return "lower (significant)"
    if lower > 0:
        return "higher (significant)"
    return "not significant"


def near_line(*, lower: float, upper: float) -> bool:
    """Return whether one bound lies within `NEAR_LINE_SHARE` of the interval's width from zero."""
    width = upper - lower
    return min(abs(lower), abs(upper)) <= NEAR_LINE_SHARE * width


def coverage_summary(*, setting: SettingType, issue: str, arm: str, batteries: list[str]) -> dict:
    """Pool one arm's rows over batteries and seeds: band coverage and width, reliability."""
    hits = np.zeros(len(SYMMETRIC_BANDS))
    widths = np.zeros(len(SYMMETRIC_BANDS))
    below = np.zeros(len(LEVELS))
    total = 0
    for battery in batteries:
        path = arm_file(setting=setting, issue=issue, battery_id=battery, arm=arm)
        if not path.exists():
            continue
        frame = pl.read_parquet(path, columns=["truth_mw", "p99_mw", *Q_COLUMNS])
        truth = frame["truth_mw"].to_numpy()
        p99 = frame["p99_mw"].to_numpy()
        quantiles = frame.select(Q_COLUMNS).to_numpy().astype(np.float64)
        for i, (lower, upper) in enumerate(SYMMETRIC_BANDS):
            low = quantiles[:, LEVELS.index(lower)]
            high = quantiles[:, LEVELS.index(upper)]
            hits[i] += np.sum((truth >= low) & (truth <= high))
            widths[i] += np.sum(100.0 * (high - low) / p99)
        below += np.sum(truth[:, None] <= quantiles, axis=0)
        total += truth.size
    return {
        "coverage": (hits / total).tolist(),
        "width_pct": (widths / total).tolist(),
        "reliability": (below / total).tolist(),
        "n_rows": total,
    }


LEADERBOARD_HEADERS: Final[list[str]] = [
    "Arm",
    "CRPS, points of p99 [95% interval]",
    "Pinball loss",
    "Median absolute error",
    "CRPS skill vs `clim` [95% interval]",
    "p10-p90 coverage (nominal 0.80)",
    "p10-p90 width",
    "p1-p99 coverage (nominal 0.98)",
    "p1-p99 width",
]


def leaderboard_row(
    *, losses: pl.DataFrame, setting: SettingType, issue: IssueType, arm: str, sites: list[str]
) -> tuple[list[str], dict[str, object]]:
    """Return one arm's leaderboard row, scored on the batteries `sites`, and its data."""
    subset = losses.filter((pl.col("arm") == arm) & pl.col("site").is_in(sites))
    interval = bootstrap_absolute(losses=subset, arm=arm, metric="crps_pct")
    stats = coverage_summary(setting=setting, issue=issue, arm=arm, batteries=sites)
    arm_values = subset.sort("site", "time", "seed").select("crps_pct", "month")
    reference_values = (
        losses.filter((pl.col("arm") == CLIM) & pl.col("site").is_in(sites))
        .sort("site", "time", "seed")
        .select("crps_pct", "month")
    )
    skill = (
        month_skill_interval(
            treatment=arm_values["crps_pct"].to_numpy(),
            reference=reference_values["crps_pct"].to_numpy(),
            months=arm_values["month"].to_numpy(),
        )
        if arm_values.height == reference_values.height
        else (float("nan"),) * 3
    )
    index_80 = SYMMETRIC_BANDS.index((0.1, 0.9))
    index_98 = SYMMETRIC_BANDS.index((0.01, 0.99))
    data = {
        "issue": issue,
        "arm": arm,
        "n_batteries": subset["site"].n_unique(),
        "crps": interval["value"],
        "crps_lower": interval["lower_95"],
        "crps_upper": interval["upper_95"],
        "pinball": subset["pinball_pct"].mean(),
        "median_abs_error": subset["median_abs_error_pct"].mean(),
        "skill": skill[0],
        "skill_lower": skill[1],
        "skill_upper": skill[2],
        "coverage_p10_p90": stats["coverage"][index_80],
        "width_p10_p90": stats["width_pct"][index_80],
        "coverage_p1_p99": stats["coverage"][index_98],
        "width_p1_p99": stats["width_pct"][index_98],
        "n_rows": stats["n_rows"],
        "n_months": interval["n_months"],
    }
    row = [
        f"`{arm}`",
        f"{interval['value']:.3f} [{interval['lower_95']:.3f}, {interval['upper_95']:.3f}]",
        f"{subset['pinball_pct'].mean():.3f}",
        f"{subset['median_abs_error_pct'].mean():.3f}",
        f"{skill[0]:+.3f} [{skill[1]:+.3f}, {skill[2]:+.3f}]",
        f"{stats['coverage'][index_80]:.3f}",
        f"{stats['width_pct'][index_80]:.2f}",
        f"{stats['coverage'][index_98]:.3f}",
        f"{stats['width_pct'][index_98]:.2f}",
    ]
    return row, data


def leaderboard(
    *, setting: SettingType, issue: IssueType, batteries: list[str], arms: list[str], label: str
) -> tuple[list[str], pl.DataFrame]:
    """Return the report lines and the data of one issue time's leaderboard.

    The main table holds the arms that cover every battery. An arm that covers only some of the
    batteries goes in a second table, beside the climatology and the `no_neighbour` arm scored on
    those same batteries, so no level is compared across different battery sets.
    """
    losses = load_losses(setting=setting, issue=issue, arms=arms, batteries=batteries)
    covered = {
        arm: sorted(losses.filter(pl.col("arm") == arm)["site"].unique().to_list()) for arm in arms
    }
    rows, data = [], []
    for arm in arms:
        if covered[arm] and len(covered[arm]) == len(batteries):
            row, record = leaderboard_row(
                losses=losses, setting=setting, issue=issue, arm=arm, sites=batteries
            )
            rows.append(row)
            data.append({"label": label, **record})
    lines = table(headers=LEADERBOARD_HEADERS, rows=rows)
    for arm in arms:
        if not covered[arm] or len(covered[arm]) == len(batteries):
            continue
        sites = covered[arm]
        lines += [
            "",
            (
                f"`{arm}` covers {len(sites)} of the {len(batteries)} batteries. On those "
                f"{len(sites)} batteries:"
            ),
            "",
        ]
        subset_rows = []
        for name in (arm, CLIM, f"{XGB}no_neighbour"):
            if covered.get(name):
                row, record = leaderboard_row(
                    losses=losses, setting=setting, issue=issue, arm=name, sites=sites
                )
                subset_rows.append(row)
                data.append({"label": f"{label}, {len(sites)} batteries of {arm}", **record})
        lines += table(headers=LEADERBOARD_HEADERS, rows=subset_rows)
    return lines, pl.DataFrame(data)


def band_and_reliability_lines(
    *, setting: SettingType, issue: IssueType, batteries: list[str], arms: list[str]
) -> list[str]:
    """Return the six-band coverage and width table and the reliability table of an issue time."""
    stats = {
        arm: coverage_summary(setting=setting, issue=issue, arm=arm, batteries=batteries)
        for arm in arms
    }
    stats = {arm: s for arm, s in stats.items() if s["n_rows"] > 0}
    band_rows = []
    for arm, s in stats.items():
        band_rows.append(
            [f"`{arm}`"]
            + [f"{c:.3f} / {w:.1f}" for c, w in zip(s["coverage"], s["width_pct"], strict=True)]
        )
    lines = table(
        headers=["Arm"]
        + [f"p{round(lo * 100)}-p{round(hi * 100)} coverage / width" for lo, hi in SYMMETRIC_BANDS],
        rows=band_rows,
    )
    lines += [""]
    reliability_rows = [
        [f"`{arm}`"] + [f"{share:.3f}" for share in s["reliability"]] for arm, s in stats.items()
    ]
    lines += table(
        headers=["Arm, share of half-hours at or below the quantile of level"]
        + [f"{level:g}" for level in LEVELS],
        rows=reliability_rows,
    )
    return lines


def contrast_row(
    *, contrast: Contrast, setting: SettingType, batteries: list[str]
) -> dict[str, object]:
    """Return one contrast's difference, interval, skills, and coverage at one setting."""
    losses = contrast_losses(contrast=contrast, setting=setting, batteries=batteries)
    result = bootstrap_difference(
        losses=losses,
        treatment=contrast.treatment,
        reference=contrast.reference,
        metric="crps_pct",
    )
    reference_setting: SettingType = setting if contrast.reference.startswith(XGB) else "primary"
    common = sorted(losses["site"].unique().to_list())
    coverage = {}
    for name, arm_setting in (
        (contrast.treatment, setting),
        (contrast.reference, reference_setting),
    ):
        s = coverage_summary(setting=arm_setting, issue=contrast.issue, arm=name, batteries=common)
        index = SYMMETRIC_BANDS.index((0.1, 0.9))
        coverage[name] = (s["coverage"][index], s["width_pct"][index])
    treatment_mean = float(
        np.mean(losses.filter(pl.col("arm") == contrast.treatment)["crps_pct"].to_numpy())
    )
    reference_mean = float(
        np.mean(losses.filter(pl.col("arm") == contrast.reference)["crps_pct"].to_numpy())
    )
    return {
        "label": contrast.label,
        "setting": setting,
        "issue": contrast.issue,
        "treatment": contrast.treatment,
        "reference": contrast.reference,
        "planned": contrast.planned,
        "part": contrast.part,
        "difference": result["difference"],
        "lower_95": result["lower_95"],
        "upper_95": result["upper_95"],
        "seed_spread": result["seed_spread"],
        "n_rows": result["n_rows"],
        "n_months": result["n_months"],
        "n_batteries": len(common),
        "treatment_crps": treatment_mean,
        "reference_crps": reference_mean,
        "treatment_skill": 1.0 - treatment_mean / reference_mean
        if reference_mean
        else float("nan"),
        "treatment_coverage": coverage[contrast.treatment][0],
        "treatment_width": coverage[contrast.treatment][1],
        "reference_coverage": coverage[contrast.reference][0],
        "reference_width": coverage[contrast.reference][1],
        "verdict": significance(lower=result["lower_95"], upper=result["upper_95"]),
        "near_line": near_line(lower=result["lower_95"], upper=result["upper_95"]),
    }


def contrast_lines(*, rows: pl.DataFrame) -> list[str]:
    """Return a Markdown table of contrast rows."""
    body = [
        [
            f"{r['label']}{' (planned)' if r['planned'] else ' (exploratory)'}",
            r["setting"],
            str(r["n_batteries"]),
            f"{r['issue']}: `{r['treatment']}` minus `{r['reference']}`",
            f"{r['difference']:+.3f} [{r['lower_95']:+.3f}, {r['upper_95']:+.3f}]",
            r["verdict"] + (", near the line" if r["near_line"] else ""),
            f"{r['treatment_crps']:.3f} / {r['reference_crps']:.3f}",
            f"{r['treatment_coverage']:.3f} / {r['reference_coverage']:.3f}",
            f"{r['treatment_width']:.1f} / {r['reference_width']:.1f}",
        ]
        for r in rows.iter_rows(named=True)
    ]
    return table(
        headers=[
            "Contrast",
            "Setting",
            "Batteries",
            "Treatment minus reference",
            "CRPS difference, points of p99 [95% interval]",
            "Treatment CRPS is",
            "CRPS treatment / reference",
            "p10-p90 coverage treatment / reference",
            "p10-p90 width treatment / reference",
        ],
        rows=body,
    )


def trivial_fpn_batteries(*, batteries: list[str]) -> list[str]:
    """Return the batteries whose Physical Notification is exactly zero in most half-hours.

    A battery qualifies when at least `TRIVIAL_FPN_ZERO_SHARE` of the half-hours that carry a
    notification hold exactly zero. Its own notification then says almost nothing, and its
    neighbours' mean is pulled towards zero.
    """
    notified = load_physical_notifications().filter(pl.col("bmu_id").is_in(batteries))
    shares = notified.group_by("bmu_id").agg(zero_share=(pl.col("fpn_mw") == 0.0).mean())
    return sorted(shares.filter(pl.col("zero_share") >= TRIVIAL_FPN_ZERO_SHARE)["bmu_id"].to_list())


def share_recovered(
    *, setting: SettingType, batteries: list[str], excluded: tuple[str, ...] = ()
) -> list[str]:
    """Return the share of the own-FPN gain that neighbours recover, with a month resampling.

    Args:
        setting: The hyperparameter setting.
        batteries: The target batteries.
        excluded: Targets left out of the share. Their neighbour statistics are not rebuilt.
    """
    arms = [f"{XGB}no_neighbour", f"{XGB}own_fpn", f"{XGB}neighbour_fpn"]
    kept = [b for b in batteries if b not in excluded]
    losses = load_losses(setting=setting, issue="ID-1h", arms=arms, batteries=kept)
    wide = losses.pivot(on="arm", index=["site", "time", "seed", "month"], values="crps_pct").sort(
        "site", "time", "seed"
    )
    months = wide["month"].to_numpy()
    none, own, neighbour = (wide[a].to_numpy() for a in arms)
    unique, index = np.unique(months, return_inverse=True)
    sums = [np.bincount(index, weights=v, minlength=unique.size) for v in (none, own, neighbour)]
    point = (sums[0].sum() - sums[2].sum()) / (sums[0].sum() - sums[1].sum())
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.integers(0, unique.size, size=(N_BOOTSTRAP_RESAMPLES, unique.size))
    resampled = (sums[0][draws].sum(axis=1) - sums[2][draws].sum(axis=1)) / (
        sums[0][draws].sum(axis=1) - sums[1][draws].sum(axis=1)
    )
    return [
        (
            f"- On {len(kept)} target batteries: own-FPN gain over `no_neighbour`: "
            f"{(sums[0].sum() - sums[1].sum()) / none.size:.3f} points of p99 per half-hour; "
            f"neighbours' gain: {(sums[0].sum() - sums[2].sum()) / none.size:.3f}."
        ),
        (
            f"- **Share of the own-FPN gain that neighbours recover (exploratory): {point:.3f}** "
            f"(95% month-resampled interval {np.percentile(resampled, 2.5):.3f} to "
            f"{np.percentile(resampled, 97.5):.3f}; the interval is unreliable when the own-FPN "
            "gain is near zero)."
        ),
    ]


def party_resampling_section(*, testbed: list[str]) -> tuple[list[str], pl.DataFrame]:
    """Return the party-and-month resampled intervals of D1 to D4 and the share recovered.

    This is a post hoc sensitivity. The month-and-seed intervals hold the 35 batteries fixed.
    Here the lead parties are resampled as well as the months, and the fitting seed is averaged
    out first, so the interval leaves out seed-to-seed spread. D5 has one battery and so no
    parties to resample.
    """
    parties = lead_parties()
    party_of = {b: parties[b] for b in testbed}
    rows = []
    for contrast in PLANNED_CONTRASTS:
        if contrast.part == "B":
            continue
        losses = contrast_losses(contrast=contrast, setting="primary", batteries=testbed)
        result = party_month_difference(
            treatment=arm_cells(losses=losses, arm=contrast.treatment, party_of=party_of),
            reference=arm_cells(losses=losses, arm=contrast.reference, party_of=party_of),
        )
        rows.append({"label": contrast.label, **result})
    frame = pl.DataFrame(rows)
    body = [
        [
            f"{r['label']} (planned; interval post hoc)",
            f"{r['difference']:+.3f}",
            f"[{r['lower_95']:+.3f}, {r['upper_95']:+.3f}]",
            "crosses zero" if r["lower_95"] < 0 < r["upper_95"] else "excludes zero",
            f"{r['party_weighted']:+.3f}",
        ]
        for r in frame.iter_rows(named=True)
    ]
    lines = [
        "## Party-and-month resampling (post hoc sensitivity, primary setting)",
        "",
        (
            f"- The testbed's {len(set(party_of.values()))} lead parties are resampled with "
            "replacement, and so are the calendar months; the fitting seed is averaged out first. "
            "The unit of independence is the lead party, because one party runs "
            f"{max(Counter(party_of.values()).values())} of the {len(testbed)} batteries."
        ),
        "",
        *table(
            headers=[
                "Contrast",
                "Pooled difference",
                "Party-and-month 95% interval",
                "Interval",
                "Party-weighted mean difference",
            ],
            rows=body,
        ),
        "",
    ]
    arms = [f"{XGB}no_neighbour", f"{XGB}own_fpn", f"{XGB}neighbour_fpn"]
    losses = load_losses(setting="primary", issue="ID-1h", arms=arms, batteries=testbed)
    cells = [arm_cells(losses=losses, arm=a, party_of=party_of) for a in arms]
    share = party_month_share(baseline=cells[0], full=cells[1], partial=cells[2])
    lines.append(
        f"- Share of the own-FPN gain that neighbours recover: {share['share']:.3f} "
        f"(party-and-month 95% interval {share['lower_95']:.3f} to {share['upper_95']:.3f})."
    )
    return [*lines, ""], frame


def skill_by_lead(*, setting: SettingType, batteries: list[str]) -> tuple[list[str], pl.DataFrame]:
    """Return CRPS skill against `clim` by lead time for the headline arms."""
    pairs = (
        ("DA-early", f"{XGB}price_model"),
        ("DA-early", f"{XGB}price_actual"),
        ("DA-late", f"{XGB}price_actual"),
        ("DA-late", "rank_conformal__price_actual"),
        ("DA-late", "persistence_conformal"),
    )
    offsets = {"DA-early": 18, "DA-late": 6}
    rows: list[list[str]] = []
    data = []
    for issue, arm in pairs:
        losses = load_losses(setting=setting, issue=issue, arms=[arm, CLIM], batteries=batteries)
        losses = losses.with_columns(
            lead_bin=((pl.col("time").dt.hour().cast(pl.Int64) + offsets[issue]) // LEAD_BIN_HOURS)
            * LEAD_BIN_HOURS
        )
        for lead in sorted(losses["lead_bin"].unique().to_list()):
            part = losses.filter(pl.col("lead_bin") == lead).sort("site", "time", "seed", "arm")
            treat = part.filter(pl.col("arm") == arm)
            reference = part.filter(pl.col("arm") == CLIM)
            skill = month_skill_interval(
                treatment=treat["crps_pct"].to_numpy(),
                reference=reference["crps_pct"].to_numpy(),
                months=treat["month"].to_numpy(),
            )
            rows.append(
                [
                    issue,
                    f"`{arm}`",
                    f"{lead}-{lead + LEAD_BIN_HOURS}",
                    f"{skill[0]:+.3f} [{skill[1]:+.3f}, {skill[2]:+.3f}]",
                ]
            )
            data.append(
                {
                    "issue": issue,
                    "arm": arm,
                    "lead_from_hours": lead,
                    "skill": skill[0],
                    "lower": skill[1],
                    "upper": skill[2],
                }
            )
    lines = table(
        headers=["Issue", "Arm", "Lead, hours", "CRPS skill vs `clim` [95% interval]"], rows=rows
    )
    return lines, pl.DataFrame(data)


def per_battery_skill(
    *, setting: SettingType, batteries: list[str]
) -> tuple[list[str], pl.DataFrame]:
    """Return each battery's CRPS skill against `clim` for the headline arms."""
    columns = (
        ("DA-early", f"{XGB}price_model"),
        ("DA-late", f"{XGB}price_actual"),
        ("ID-1h", f"{XGB}no_neighbour"),
        ("ID-1h", f"{XGB}neighbour_fpn"),
        ("ID-1h", f"{XGB}own_fpn"),
    )
    skills: dict[tuple[str, str], dict[str, float]] = {}
    for issue, arm in columns:
        losses = load_losses(setting=setting, issue=issue, arms=[arm, CLIM], batteries=batteries)
        per = {}
        for battery in batteries:
            part = losses.filter(pl.col("site") == battery)
            if part.is_empty():
                continue
            a = float(np.mean(part.filter(pl.col("arm") == arm)["crps_pct"].to_numpy()))
            c = float(np.mean(part.filter(pl.col("arm") == CLIM)["crps_pct"].to_numpy()))
            per[battery] = 1.0 - a / c
        skills[(issue, arm)] = per
    parties = lead_parties()
    rows = []
    data = []
    for battery in batteries:
        row = [battery, parties.get(battery, "")]
        record: dict[str, object] = {"battery": battery}
        for key in columns:
            value = skills[key].get(battery, float("nan"))
            row.append(f"{value:+.3f}")
            record[f"{key[0]}__{key[1]}"] = value
        rows.append(row)
        data.append(record)
    count_lines = []
    for key in columns:
        values = np.asarray(list(skills[key].values()))
        count_lines.append(
            f"- {key[0]} `{key[1]}`: positive skill for {int((values > 0).sum())} of {values.size} "
            f"batteries; median {np.median(values):+.3f}."
        )
    lines = table(headers=["Battery", "Lead party"] + [f"{i} `{a}`" for i, a in columns], rows=rows)
    return [*count_lines, "", *lines], pl.DataFrame(data)


def arm_lines() -> list[str]:
    """Return a table of the arms: method and feature recipe (columns: see `inputs_report.md`)."""
    rows = []
    for label, battery_id in (
        ("testbed battery", testbed_ids()[0]),
        ("NGED battery A", NGED_BATTERY_A_FILE_ID),
    ):
        battery = battery_for(battery_id=battery_id)
        for issue in ("DA-early", "DA-late", "ID-1h"):
            for setting in ("primary", "sensitivity"):
                for arm in arm_definitions(battery=battery, issue=issue, setting=setting):
                    spec = arm.spec
                    recipe = (
                        "-"
                        if spec is None
                        else f"price `{spec.price_source}`, own slot `{spec.own_slot}`, "
                        f"neighbour slot `{spec.neighbour_slot}`"
                    )
                    rows.append([label, issue, setting, f"`{arm.name}`", arm.method, recipe])
    return table(
        headers=["Battery kind", "Issue", "Setting", "Arm", "Method", "Feature recipe"], rows=rows
    )


def leaderboard_section(*, testbed: list[str], nged: list[str]) -> list[str]:
    """Return the leaderboards, band tables, and reliability tables for both parts."""
    lines: list[str] = []
    boards = []
    for label, batteries in (("testbed", testbed), ("NGED battery A", nged)):
        lines += [f"## Leaderboards: {label}", ""]
        for issue in ("DA-early", "DA-late", "ID-1h"):
            battery = battery_for(battery_id=batteries[0])
            names = [
                a.name for a in arm_definitions(battery=battery, issue=issue, setting="primary")
            ]
            board_lines, board_data = leaderboard(
                setting="primary", issue=issue, batteries=batteries, arms=names, label=label
            )
            boards.append(board_data)
            lines += [f"### {label}, {issue} (primary setting)", "", *board_lines, ""]
            lines += [
                f"#### Six-band coverage / width (points of p99) and reliability, {label}, {issue}",
                "",
                *band_and_reliability_lines(
                    setting="primary", issue=issue, batteries=batteries, arms=names
                ),
                "",
            ]
    save_table(frame=pl.concat(boards, how="diagonal_relaxed"), name="leaderboards.parquet")
    return lines


def planned_section(*, testbed: list[str], nged: list[str]) -> tuple[list[str], pl.DataFrame]:
    """Return the planned contrasts at both settings, with a verdict for each."""
    rows = []
    for contrast in PLANNED_CONTRASTS:
        batteries = nged if contrast.part == "B" else testbed
        rows += [
            contrast_row(contrast=contrast, setting=setting, batteries=batteries)
            for setting in ("primary", "sensitivity")
        ]
    planned = pl.DataFrame(rows)
    save_table(frame=planned, name="planned_contrasts.parquet")
    lines = ["## Planned contrasts", "", *contrast_lines(rows=planned), "", "### Verdicts", ""]
    for contrast in PLANNED_CONTRASTS:
        both = planned.filter(pl.col("label") == contrast.label)
        met = bool((both["upper_95"] < 0).all())
        lines.append(
            f"- **{contrast.label}** ({contrast.question}) {'is met' if met else 'is not met'} "
            f"(primary {both['difference'][0]:+.3f}, sensitivity {both['difference'][1]:+.3f}; "
            "met means negative and statistically significant at the 5% level under both "
            "settings)."
        )
    lines += ["", "### Size of each effect, and the largest improvement each interval allows", ""]
    for r in planned.iter_rows(named=True):
        improvement_bound = -r["lower_95"]
        text = (
            f"- {r['label']}, {r['setting']} setting: the difference is "
            f"{r['difference']:+.3f} points, {abs(r['difference']) / r['reference_crps']:.1%} of "
            f"the reference's CRPS ({r['reference_crps']:.3f})."
        )
        if r["upper_95"] >= 0:
            text += (
                f" An improvement larger than {improvement_bound:.3f} points "
                f"({improvement_bound / r['reference_crps']:.1%} of the reference's CRPS) is "
                "excluded."
            )
        lines.append(text)
    return [*lines, ""], planned


def comparison_lines(*, planned: pl.DataFrame, other_path: Path, title: str, other: str) -> list[str]:
    """Return the planned contrasts of this run beside those of another run's saved table."""
    if not other_path.exists():
        return []
    earlier = pl.read_parquet(other_path)
    rows = []
    for r in planned.iter_rows(named=True):
        match = earlier.filter(
            (pl.col("label") == r["label"]) & (pl.col("setting") == r["setting"])
        )
        if match.is_empty():
            continue
        o = match.row(0, named=True)
        rows.append(
            [
                r["label"],
                r["setting"],
                f"{r['difference']:+.3f} [{r['lower_95']:+.3f}, {r['upper_95']:+.3f}]",
                f"{o['difference']:+.3f} [{o['lower_95']:+.3f}, {o['upper_95']:+.3f}]",
                f"{r['verdict']} / {o['verdict']}",
            ]
        )
    return [
        f"## {title}",
        "",
        *table(
            headers=[
                "Contrast",
                "Setting",
                "This run, difference [95% interval]",
                f"{other}, difference [95% interval]",
                "Verdicts, this run / other",
            ],
            rows=rows,
        ),
        "",
    ]


def exploratory_section(*, testbed: list[str], nged: list[str]) -> tuple[list[str], pl.DataFrame]:
    """Return the exploratory contrasts at the primary setting."""
    lines = ["## Exploratory contrasts (primary setting)", ""]
    rows = []
    for contrast in EXPLORATORY_CONTRASTS:
        batteries = nged if contrast.part == "B" else testbed
        missing = [
            arm
            for arm in (contrast.treatment, contrast.reference)
            if not any(
                arm_file(setting="primary", issue=contrast.issue, battery_id=b, arm=arm).exists()
                for b in batteries
            )
        ]
        if missing:
            lines.append(f"- {contrast.label} was not computed: no saved fit for {missing}.")
            continue
        rows.append(contrast_row(contrast=contrast, setting="primary", batteries=batteries))
    exploratory = pl.DataFrame(rows)
    save_table(frame=exploratory, name="exploratory_contrasts.parquet")
    return [*lines, "", *contrast_lines(rows=exploratory), ""], exploratory


def false_alarm_lines(*, planned: pl.DataFrame, exploratory: pl.DataFrame) -> list[str]:
    """Return the false-alarm rule for D4: the shuffled-neighbour gain must not exceed D4's."""
    d4 = planned.filter((pl.col("label") == "D4") & (pl.col("setting") == "primary"))
    x11 = exploratory.filter(pl.col("label") == "X11")
    if not (d4.height and x11.height):
        return []
    gain_shuffled = -x11["difference"][0]
    d4_effect = -d4["difference"][0]
    verdict = "D4 counts" if gain_shuffled <= d4_effect else "D4 does NOT count"
    return [
        "## False-alarm rule for D4",
        "",
        (
            f"- `neighbour_fpn_shuffled` gain over `no_neighbour`: {gain_shuffled:+.3f} points; "
            f"D4's effect: {d4_effect:+.3f} points."
        ),
        f"- Rule (D4 counts only if the shuffled gain is at most D4's effect): **{verdict}**.",
        "",
    ]


def main() -> None:
    """Write `forecast_report.md` and the tables."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    testbed = testbed_ids()
    nged = [NGED_BATTERY_A_FILE_ID]
    lines = ["# Embedded-battery forecast: results", ""]
    lines += ["## Issue-time cut-offs", "", *issue_cutoff_lines(), ""]
    lines += [
        "## Scope",
        "",
        f"- Testbed batteries: {len(testbed)}. Scored months: October 2025 to August 2026.",
        "- Unit of every loss: points of the battery's own 99th-percentile absolute output.",
        (
            "- Intervals: 95%, whole calendar months and one of 3 fitting seeds resampled "
            f"together ({N_BOOTSTRAP_RESAMPLES:,} resamples), months shared across batteries."
        ),
        (
            "- The deterministic arms (`clim`, `persistence_conformal`, `rank_conformal`) hold "
            "no random draw, so each result is stored under all 3 seed labels."
        ),
        "- All XGBoost fits ran on the CPU (one device for every arm), 2 threads per fit.",
        "",
        "## Arms",
        "",
        *arm_lines(),
        "",
    ]
    for name in ("a0_report_primary.md", "price_model_report.md", "inputs/inputs_report.md"):
        path = EMBEDDED_BATTERY_FORECAST_DIR / name
        if path.exists():
            text = path.read_text().splitlines()
            lines += [
                f"## From `{name}`",
                "",
                *[f"#{line}" if line.startswith("#") else line for line in text[1:]],
                "",
            ]
    lines += leaderboard_section(testbed=testbed, nged=nged)
    planned_lines, planned = planned_section(testbed=testbed, nged=nged)
    exploratory_lines, exploratory = exploratory_section(testbed=testbed, nged=nged)
    lines += [*planned_lines, *exploratory_lines]
    lines += ["## Rung A4 and A5: the share of the own-FPN gain that neighbours recover", ""]
    lines += [*share_recovered(setting="primary", batteries=testbed), ""]
    lines += false_alarm_lines(planned=planned, exploratory=exploratory)
    lines += ["## CRPS skill by lead time (primary setting)", ""]
    lead_lines, lead_data = skill_by_lead(setting="primary", batteries=testbed)
    save_table(frame=lead_data, name="skill_by_lead_testbed.parquet")
    nged_lead_lines, nged_lead_data = skill_by_lead(setting="primary", batteries=nged)
    save_table(frame=nged_lead_data, name="skill_by_lead_nged_battery_a.parquet")
    lines += [*lead_lines, "", "### NGED battery A", "", *nged_lead_lines, ""]
    lines += ["## Per-battery CRPS skill against `clim` (primary setting)", ""]
    battery_lines, battery_data = per_battery_skill(setting="primary", batteries=testbed)
    save_table(frame=battery_data, name="per_battery_skill.parquet")
    lines += [*battery_lines, ""]
    REPORT_PATH.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
