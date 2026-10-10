"""Print every table the lagged-power-features page quotes, from the saved losses, with no refit.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/1138>. It reads
`losses_<product>.parquet` and the frames that `build_lag_frame.py` and `fit_lag_arms.py` wrote, and
writes `report_<product>.md` and `tables_<product>/*.parquet` (which `lag_features_charts.py`
reads, so a chart cannot disagree with the report).

**The planned contrasts, the decision rule, and the screening months are fixed in the plan, before
any fit, and are copied here in their plan wording:**

- **P1:** L1 minus B0, mean absolute error, per-plant, all months.
- **P2:** X minus L1, mean absolute error, per-plant, on the months after 2025-12 only, where X is
  the shortlist rule's arm.
- **P3:** S2 minus L2, mean absolute error, per-plant, all months.
- **P4:** the continuous ranked probability score of L1 minus that of B0, per-plant, all months.
- **P5:** L1 minus B0, mean absolute error, global, all months.

All five are at lead-day 1, at both hyperparameter settings, with 99% intervals (a Bonferroni
correction over five), judged with `bracket_verdict` and `combine_setting_verdicts`.

**Decision rule, fixed before any result:** a contrast "helps" if, at both hyperparameter settings,
its 99% interval lies wholly below zero and its point estimate is at least 2% of the baseline's own
value of that metric; a smaller gain is reported as detectable but not worth the complexity. Every
other number is exploratory and labelled so.

Run it with `uv run python studies/lag_features/report_lag_features.py`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import polars as pl
from build_lag_frame import (
    ARM_LABELS,
    CLOCK_RATIO_TEMPLATE,
    ENS_MEAN_LEAD_DAYS,
    FULL_SWEEP_LEAD_DAY,
    GLOBAL_ONLY_ARMS,
    IFS_LEAD_DAYS,
    LAG_FEATURES_DIR,
    POSITIVE_CONTROL_SHIFTS,
    RATIO_TEMPLATE,
    WEATHER_PRODUCTS,
    WeatherProduct,
    output_paths,
)
from fit_lag_arms import METRIC, SCREENING_MONTHS, SHORTLIST_CANDIDATES
from studies.baselines import climatology
from studies.bootstrap import (
    MIN_MONTHS_FOR_INTERVAL,
    BootstrapInterval,
    arm_values,
    bootstrap_absolute,
    bootstrap_difference,
    bootstrap_difference_at_level,
    bracket_verdict,
    combine_setting_verdicts,
    paired_differences,
)
from studies.cross_validation import N_FOLDS, QUANTILE_LEVELS, SEEDS, clamp_to_cap, crps
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("report_lag_features")

CRPS_METRIC: Final[str] = "crps_capped_fraction_of_capacity"
"""The probabilistic loss: the nine-quantile approximation of the CRPS over the plant's capacity."""

PLANNED_LEVEL: Final[float] = 99.0
"""The planned contrasts' interval level in percent: 95% with a Bonferroni correction over five."""

EXPLORATORY_LEVEL: Final[float] = 95.0
"""The exploratory contrasts' interval level in percent."""

HELPS_SHARE: Final[float] = 0.02
"""A contrast helps only if its point estimate is at least this share of the baseline's value."""

P2_AFTER_MONTH: Final[str] = "2025-12"
"""P2 reads only the months after this one, which the sweep's screening never saw."""

PERCENTAGE_POINTS: Final[float] = 100.0
"""Losses are fractions of capacity; every table prints percentage points."""

NOMINAL_COVERAGE: Final[float] = 0.8
"""The share of rows a 10% to 90% interval should hold."""

CLEAR_SKY_CLASSES: Final[dict[str, tuple[float, float]]] = {
    "clear": (0.8, np.inf),
    "mixed": (0.4, 0.8),
    "cloudy": (-np.inf, 0.4),
}
"""The forecast clear-sky index ranges (global irradiance over clear-sky irradiance) of each sky."""

EXPLORATORY_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("S2", "L1"),
    ("IM", "L1"),
    ("KS", "L1"),
    ("L2", "L1"),
    ("L1", "N1"),
    ("L1", "N2"),
    ("N1", "B0"),
    ("N2", "B0"),
    ("T1", "B0"),
)
"""Exploratory (treatment, reference) pairs at lead-day 1: the two-step arm against the lag, the
lag against each control, and the controls and the drift arm against the baseline. N2 minus B0 is
the column-count bias plus the noise floor."""

REFERENCE_ARMS: Final[tuple[str, ...]] = (
    "persistence",
    "diurnal_persistence",
    "climatology",
    "R1s",
    "R2",
    "R4",
)
"""The no-fit references scored on point error: persistence and diurnal persistence at the
lead-day, climatology, and three post-model corrections of B0's prediction. R1s multiplies it by the
last 7 days' observed-over-predicted stage-1 energy ratio, shrunk halfway to 1. R2 multiplies it by
the same ratio taken per clock hour over 30 days. R4 clamps it at CK's expanding ceiling. R5, which
needs quantiles, is scored on CRPS in its own table."""

R1_SHRINKAGE: Final[float] = 0.5
"""R1s moves the ratio this share of the way from 1 to its value."""

R5_WINDOW_DAYS: Final[int] = 30
R5_MIN_EVENTS: Final[int] = 10
"""R5 reads the stage-1 residuals of the last 30 whole days, and needs at least 10."""

SEED_ORDER: Final[tuple[int, ...]] = SEEDS


MONTH_LABELS: Final[dict[str, str]] = {
    "all": "all months",
    "screening": "the screening months",
    "after": f"months after {P2_AFTER_MONTH}",
}
"""How a contrast's month selector reads in a table."""


class Planned(NamedTuple):
    """One planned contrast."""

    name: str
    treatment: str
    reference: str
    metric: str
    scope: str
    months: str


# --- Interval arithmetic ------------------------------------------------------------------------


def subset(
    *,
    losses: pl.DataFrame,
    scope: str,
    setting: str,
    arms: tuple[str, ...],
    months: str = "all",
) -> pl.DataFrame:
    """Return the losses of some arms in one scope and setting, on some months.

    Args:
        losses: The combined losses.
        scope: The scope, such as `lead1` or `global`.
        setting: `primary` or `sensitivity`.
        arms: The arms to keep.
        months: `all`, `screening` (`SCREENING_MONTHS`), or `after` (months after `P2_AFTER_MONTH`).

    Returns:
        The matching rows.
    """
    kept = losses.filter(
        (pl.col("scope") == scope) & (pl.col("setting") == setting) & pl.col("arm").is_in(arms)
    )
    if months == "screening":
        return kept.filter(pl.col("month").is_in(SCREENING_MONTHS))
    if months == "after":
        return kept.filter(pl.col("month") > P2_AFTER_MONTH)
    return kept


def paired_interval(
    *, scoped: pl.DataFrame, treatment: str, reference: str, metric: str, level: float
) -> BootstrapInterval | None:
    """Bound a paired difference at some level, or return `None` if an arm is missing.

    The returned `lower_95` and `upper_95` hold the bounds at `level`, whatever it is, so the result
    can go to `bracket_verdict`.

    Args:
        scoped: Losses holding both arms in one scope and setting.
        treatment: The arm compared.
        reference: The arm it is compared with.
        metric: The loss column.
        level: The interval's coverage in percent.

    Returns:
        The difference, its bounds at `level`, and what it rests on.
    """
    if scoped.filter(pl.col("arm") == treatment).is_empty():
        return None
    if scoped.filter(pl.col("arm") == reference).is_empty():
        return None
    differences, months = paired_differences(
        losses=scoped, treatment=treatment, reference=reference, metric=metric
    )
    lower, upper = bootstrap_difference_at_level(
        losses=scoped, treatment=treatment, reference=reference, metric=metric, level=level
    )
    return BootstrapInterval(
        difference=float(differences.mean()),
        lower_95=lower,
        upper_95=upper,
        seed_spread=float(differences.mean(axis=1).std()),
        n_rows=differences.shape[1],
        n_months=len(np.unique(months)),
    )


def setting_verdict(*, interval: BootstrapInterval, reference_value: float) -> str:
    """Apply the decision rule at one hyperparameter setting.

    Args:
        interval: The planned contrast's 99% interval.
        reference_value: The baseline's own value of the metric.

    Returns:
        `helps`, `detectable gain below the 2% threshold`, `hurts`, or `unresolved`.
    """
    verdict = bracket_verdict(lower_side=interval, upper_side=interval)
    if verdict == "beats":
        if interval["difference"] <= -HELPS_SHARE * reference_value:
            return "helps"
        return "detectable gain below the 2% threshold"
    return "hurts" if verdict == "loses" else "unresolved"


def planned_contrasts(*, chosen: str) -> list[Planned]:
    """Return the five planned contrasts, with X filled in.

    Args:
        chosen: The shortlist rule's arm.

    Returns:
        P1 to P5.
    """
    lead = f"lead{FULL_SWEEP_LEAD_DAY}"
    return [
        Planned("P1", "L1", "B0", METRIC, lead, "all"),
        Planned("P2", chosen, "L1", METRIC, lead, "after"),
        Planned("P3", "S2", "L2", METRIC, lead, "all"),
        Planned("P4", "L1", "B0", CRPS_METRIC, lead, "all"),
        Planned("P5", "L1", "B0", METRIC, "global", "all"),
    ]


def planned_lines(*, losses: pl.DataFrame, chosen: str) -> tuple[list[str], pl.DataFrame]:
    """Compute the five planned contrasts at both settings.

    Args:
        losses: The combined losses.
        chosen: The shortlist rule's arm.

    Returns:
        The report lines, and a table with one row per (contrast, setting).
    """
    lines = [
        (
            "| Contrast | Setting | Difference (pp) | 99% interval (pp) | Baseline (pp) | Months "
            "| Verdict |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    records = []
    notes = []
    for planned in planned_contrasts(chosen=chosen):
        verdicts = {}
        for setting in ("primary", "sensitivity"):
            scoped = subset(
                losses=losses,
                scope=planned.scope,
                setting=setting,
                arms=(planned.treatment, planned.reference),
                months=planned.months,
            )
            interval = paired_interval(
                scoped=scoped,
                treatment=planned.treatment,
                reference=planned.reference,
                metric=planned.metric,
                level=PLANNED_LEVEL,
            )
            if interval is None:
                lines.append(f"| {planned.name} | {setting} | not fitted | | | | |")
                continue
            reference_value = float(
                arm_values(losses=scoped, arm=planned.reference, metric=planned.metric)[0].mean()
            )
            verdicts[setting] = setting_verdict(interval=interval, reference_value=reference_value)
            records.append(
                {
                    "contrast": planned.name,
                    "treatment": planned.treatment,
                    "reference": planned.reference,
                    "setting": setting,
                    "difference": interval["difference"],
                    "lower": interval["lower_95"],
                    "upper": interval["upper_95"],
                    "reference_value": reference_value,
                    "n_months": interval["n_months"],
                    "verdict": verdicts[setting],
                }
            )
            lines.append(
                f"| {planned.name}: {planned.treatment} − {planned.reference} ({planned.metric}, "
                f"{planned.scope}, {MONTH_LABELS[planned.months]}) | {setting} | "
                f"{interval['difference'] * PERCENTAGE_POINTS:+.3f} | "
                f"[{interval['lower_95'] * PERCENTAGE_POINTS:+.3f}, "
                f"{interval['upper_95'] * PERCENTAGE_POINTS:+.3f}] | "
                f"{reference_value * PERCENTAGE_POINTS:.3f} | {interval['n_months']} | "
                f"{verdicts[setting]} |"
            )
            if interval["n_months"] < MIN_MONTHS_FOR_INTERVAL:
                notes.append(
                    f"- {planned.name} at the {setting} setting rests on {interval['n_months']} "
                    f"months, fewer than the {MIN_MONTHS_FOR_INTERVAL} that count as evidence."
                )
        if len(verdicts) == len(("primary", "sensitivity")):
            final = combine_setting_verdicts(**verdicts)
            lines.append(f"| **{planned.name} verdict, both settings** | | | | | | **{final}** |")
    return [*lines, "", *dict.fromkeys(notes)], pl.DataFrame(records)


# --- Reference arms -----------------------------------------------------------------------------


def reference_losses(*, frame: pl.DataFrame, lead_day: int, losses: pl.DataFrame) -> pl.DataFrame:
    """Score the no-fit references on a lead-day's frame, as losses.

    Args:
        frame: The lead-day's frame, with `fold`, `constrained`, `power_mw`, `cap_mw` and capacity,
            and persistence columns where the lead-day has them.
        lead_day: The lead-day.
        losses: The combined losses, for R1's B0 predictions.

    Returns:
        Rows in the shape of the fitted losses, with `arm` in `REFERENCE_ARMS` and a copy per seed
        (a reference has no seed, and the bootstrap pairs seeds).
    """
    predictions = {
        "climatology": climatology(frame=frame.sort("site", "time")),
    }
    for name in ("persistence", "diurnal_persistence"):
        column = f"{name}_day{lead_day}"
        if column in frame.columns:
            predictions[name] = frame.sort("site", "time")[column].cast(pl.Float64)
    ordered = frame.sort("site", "time")
    base = ordered.select("site", "time", "month", "power_mw", "cap_mw", "effective_capacity_mw")
    parts = []
    for name, prediction in predictions.items():
        parts.append(
            base.with_columns(prediction=prediction, arm=pl.lit(name)),
        )
    parts.extend(_post_model_predictions(frame=ordered, losses=losses, lead_day=lead_day))
    seeds = pl.DataFrame({"seed": list(SEEDS)}, schema={"seed": pl.Int32})
    out = []
    for part in parts:
        if "seed" not in part.columns:
            part = part.join(seeds, how="cross")  # noqa: PLW2901
        capped = clamp_to_cap(prediction=part["prediction"].to_numpy(), cap_mw=part["cap_mw"])
        out.append(
            part.with_columns(
                error=pl.Series(
                    np.abs(part["power_mw"].cast(pl.Float64).to_numpy() - capped) / 1.0
                ).cast(pl.Float64)
                / pl.col("effective_capacity_mw").cast(pl.Float64)
            )
            .rename({"error": METRIC})
            .select("site", "time", "month", "seed", "arm", METRIC)
        )
    stacked = pl.concat(out)
    scored = stacked.filter(pl.col(METRIC).is_not_nan())
    if scored.height != stacked.height:
        _LOG.warning(
            "%d reference rows have no prediction (climatology with no training row) and are "
            "left out of those references",
            stacked.height - scored.height,
        )
    return scored


def _post_model_predictions(
    *, frame: pl.DataFrame, losses: pl.DataFrame, lead_day: int
) -> list[pl.DataFrame]:
    """Return the prediction rows of R1s, R2 and R4, which correct B0's prediction without a fit.

    Args:
        frame: The lead-day's frame, sorted by site and time, with the stage-1 ratio columns and
            `ck_expanding_p995` where the lead-day has them.
        losses: The combined losses, holding B0's lead-day 1 predictions at the primary setting.
        lead_day: The lead-day.

    Returns:
        One frame per correction, per seed, labelled `R1s`, `R2` and `R4`; empty where the
        lead-day's frame lacks the stage-1 columns.
    """
    if lead_day != FULL_SWEEP_LEAD_DAY or "energy_ratio_7d_fold0" not in frame.columns:
        return []

    def by_fold(template: str) -> pl.Expr:
        return pl.coalesce(
            pl.when(pl.col("fold") == fold).then(pl.col(template.format(fold=fold)))
            for fold in range(N_FOLDS)
        )

    corrections = {
        "R1s": 1.0 + R1_SHRINKAGE * (by_fold(RATIO_TEMPLATE).fill_null(1.0) - 1.0),
        "R2": by_fold(CLOCK_RATIO_TEMPLATE).fill_null(1.0),
    }
    base = subset(losses=losses, scope=f"lead{lead_day}", setting="primary", arms=("B0",)).select(
        "site", "time", "seed", "prediction"
    )
    keyed = frame.select(
        "site",
        "time",
        "month",
        "power_mw",
        "cap_mw",
        "effective_capacity_mw",
        "ck_expanding_p995",
        **{f"factor_{name}": expr for name, expr in corrections.items()},
    ).join(base, on=["site", "time"])
    parts = [
        keyed.with_columns(
            prediction=pl.col("prediction") * pl.col(f"factor_{name}"), arm=pl.lit(name)
        )
        for name in corrections
    ]
    parts.append(
        keyed.with_columns(
            prediction=pl.min_horizontal(
                pl.col("prediction"), pl.col("ck_expanding_p995").fill_null(float("inf"))
            ),
            arm=pl.lit("R4"),
        )
    )
    return [
        part.select(
            "site",
            "time",
            "month",
            "power_mw",
            "cap_mw",
            "effective_capacity_mw",
            "seed",
            "prediction",
            "arm",
        )
        for part in parts
    ]


def _residual_events(*, hours: pl.DataFrame, predictions: pl.DataFrame, fold: int) -> pl.DataFrame:
    """Return the stage-1 residuals of one scored fold, with each hour's sky class.

    Args:
        hours: The stage-1 hours, with `observed_mw`, `nwp_ghi` and `clear_sky_w_m2`.
        predictions: `fit_lag_arms.stage1_predictions`' result.
        fold: The scored fold whose stage-1 models made the predictions.

    Returns:
        `site`, `date`, `tercile` (0 to 2 by forecast clear-sky index) and `residual` (observed
        minus predicted, in megawatts).
    """
    index = pl.col("nwp_ghi") / pl.col("clear_sky_w_m2")
    events = (
        hours.select("site", "time", "observed_mw", "nwp_ghi", "clear_sky_w_m2")
        .join(
            predictions.select("site", "time", predicted=pl.col(f"stage1_pred_fold{fold}")),
            on=["site", "time"],
        )
        .filter(pl.col("clear_sky_w_m2") > 0)
        .drop_nulls("observed_mw")
        .with_columns(
            index=index,
            date=(pl.col("time") - pl.duration(minutes=30)).dt.date(),
            residual=pl.col("observed_mw") - pl.col("predicted"),
        )
    )
    cuts = events["index"].quantile(1 / 3), events["index"].quantile(2 / 3)
    return events.with_columns(
        tercile=pl.when(pl.col("index") < cuts[0])
        .then(0)
        .when(pl.col("index") < cuts[1])
        .then(1)
        .otherwise(2)
    ).select("site", "date", "tercile", "residual")


def r5_losses(
    *, frame: pl.DataFrame, hours: pl.DataFrame, predictions: pl.DataFrame, losses: pl.DataFrame
) -> pl.DataFrame:
    """Score R5 on CRPS: B0's point forecast plus the last 30 days' stage-1 residual quantiles.

    The residuals are those within the target's forecast clear-sky tercile (the terciles of the
    forecast clear-sky index over every hour with a stage-1 prediction), over the 30 whole days
    ending at the latest whole day before the issue time.

    Args:
        frame: The lead-day 1 frame, with `fold`, `nwp_ghi` and `clear_sky_w_m2`.
        hours: The stage-1 hours.
        predictions: The stage-1 predictions.
        losses: The combined losses, with B0's quantile run at the primary setting.

    Returns:
        Rows of `arm` `R5` and `B0` with `CRPS_METRIC`, on the rows where R5 has at least
        `R5_MIN_EVENTS` residuals, per seed.
    """
    base = subset(
        losses=losses, scope=f"lead{FULL_SWEEP_LEAD_DAY}", setting="primary", arms=("B0",)
    ).filter(pl.col("with_quantiles"))
    keyed = frame.select(
        "site",
        "time",
        "fold",
        "month",
        "cap_mw",
        "effective_capacity_mw",
        target_tercile_index=pl.col("nwp_ghi") / pl.col("clear_sky_w_m2"),
        asof=(pl.col("time") - pl.duration(minutes=30)).dt.date()
        - pl.duration(days=FULL_SWEEP_LEAD_DAY + 1),
    ).join(base.select("site", "time", "seed", "prediction", "actual"), on=["site", "time"])
    all_index = hours.filter(pl.col("clear_sky_w_m2") > 0).select(
        index=pl.col("nwp_ghi") / pl.col("clear_sky_w_m2")
    )["index"]
    cuts = (all_index.quantile(1 / 3), all_index.quantile(2 / 3))
    keyed = keyed.with_columns(
        tercile=pl.when(pl.col("target_tercile_index") < cuts[0])
        .then(0)
        .when(pl.col("target_tercile_index") < cuts[1])
        .then(1)
        .otherwise(2)
    )
    parts = []
    for fold in range(N_FOLDS):
        events = _residual_events(hours=hours, predictions=predictions, fold=fold)
        rows = keyed.filter(pl.col("fold") == fold)
        quantiles = np.full((rows.height, len(QUANTILE_LEVELS)), np.nan)
        starts = rows["asof"].to_numpy()
        for (site, tercile), group in events.group_by("site", "tercile"):
            ordered = group.sort("date")
            dates = ordered["date"].to_numpy()
            values = ordered["residual"].to_numpy()
            mask = ((rows["site"] == site) & (rows["tercile"] == tercile)).to_numpy()
            for index in np.flatnonzero(mask):
                last = np.searchsorted(dates, starts[index], side="right")
                first = np.searchsorted(
                    dates, starts[index] - np.timedelta64(R5_WINDOW_DAYS, "D"), side="right"
                )
                if last - first >= R5_MIN_EVENTS:
                    quantiles[index] = np.quantile(values[first:last], QUANTILE_LEVELS)
        usable = ~np.isnan(quantiles[:, 0])
        parts.append(
            rows.with_columns(
                **{
                    f"q{level_index}": pl.Series(quantiles[:, level_index])
                    for level_index in range(len(QUANTILE_LEVELS))
                }
            ).filter(pl.Series(usable))
        )
    scored = pl.concat(parts)
    forecast = (
        scored["prediction"].to_numpy()[:, None]
        + scored.select([f"q{i}" for i in range(len(QUANTILE_LEVELS))]).to_numpy()
    )
    capped = clamp_to_cap(prediction=forecast, cap_mw=scored["cap_mw"])
    r5 = scored.with_columns(
        arm=pl.lit("R5"),
        score=pl.Series(crps(actual=scored["actual"].to_numpy(), quantiles=capped))
        / scored["effective_capacity_mw"].cast(pl.Float64),
    ).select("site", "time", "month", "seed", "arm", **{CRPS_METRIC: pl.col("score")})
    b0 = base.join(r5.select("site", "time", "seed"), on=["site", "time", "seed"]).select(
        "site", "time", "month", "seed", "arm", CRPS_METRIC
    )
    return pl.concat([r5, b0])


# --- Tables ---------------------------------------------------------------------------------


def phase1_lines(
    *, losses: pl.DataFrame, references: pl.DataFrame, chosen: str
) -> tuple[list[str], pl.DataFrame]:
    """Tabulate every sweep arm's absolute error on the screening months.

    Args:
        losses: The combined losses.
        references: `reference_losses`' result at lead-day 1.
        chosen: The shortlist rule's arm.

    Returns:
        The report lines, and the table.
    """
    sweep = subset(
        losses=losses,
        scope=f"lead{FULL_SWEEP_LEAD_DAY}",
        setting="primary",
        arms=tuple(losses["arm"].unique().to_list()),
        months="screening",
    )
    both = pl.concat(
        [
            sweep.select("site", "time", "month", "seed", "arm", METRIC),
            references.filter(pl.col("month").is_in(SCREENING_MONTHS)),
        ]
    )
    records = []
    for arm in sorted(both["arm"].unique().to_list()):
        interval = bootstrap_absolute(losses=both, arm=arm, metric=METRIC)
        records.append(
            {
                "arm": arm,
                "error": interval["value"],
                "lower": interval["lower_95"],
                "upper": interval["upper_95"],
                "n_rows": interval["n_rows"],
                "n_months": interval["n_months"],
                "fitted": arm not in REFERENCE_ARMS,
            }
        )
    table = pl.DataFrame(records).sort("error")
    lines = [
        (
            "The shortlist rule (the arm with the lowest error among "
            f"{', '.join(SHORTLIST_CANDIDATES)}) chose **X = {chosen}**."
        ),
        "",
        "| Arm | Mean absolute error (pp of capacity) | 95% interval (pp) | Rows | Months |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {named(row['arm'])}{' (no fit)' if not row['fitted'] else ''} | "
        f"{row['error'] * PERCENTAGE_POINTS:.3f} | "
        f"[{row['lower'] * PERCENTAGE_POINTS:.3f}, {row['upper'] * PERCENTAGE_POINTS:.3f}] | "
        f"{row['n_rows']} | {row['n_months']} |"
        for row in table.iter_rows(named=True)
    ]
    return lines, table


def absolute_lines(*, losses: pl.DataFrame, scope: str, setting: str, title: str) -> list[str]:
    """Tabulate every arm's absolute error in one scope and setting, over all months.

    Args:
        losses: The combined losses.
        scope: The scope.
        setting: The setting.
        title: A heading for the table.

    Returns:
        The report lines.
    """
    scoped = losses.filter((pl.col("scope") == scope) & (pl.col("setting") == setting))
    if scoped.is_empty():
        return []
    lines = [
        f"**{title}**",
        "",
        "| Arm | Mean absolute error (pp) | 95% interval (pp) | Rows |",
        "|---|---|---|---|",
    ]
    for arm in sorted(scoped["arm"].unique().to_list()):
        interval = bootstrap_absolute(losses=scoped, arm=arm, metric=METRIC)
        lines.append(
            f"| {named(arm)} | {interval['value'] * PERCENTAGE_POINTS:.3f} | "
            f"[{interval['lower_95'] * PERCENTAGE_POINTS:.3f}, "
            f"{interval['upper_95'] * PERCENTAGE_POINTS:.3f}] | {interval['n_rows']} |"
        )
    return [*lines, ""]


def exploratory_lines(*, losses: pl.DataFrame, chosen: str) -> list[str]:
    """Tabulate the exploratory paired contrasts at lead-day 1, primary setting.

    Args:
        losses: The combined losses.
        chosen: The shortlist rule's arm, compared with N2-k where N2-k was fitted.

    Returns:
        The report lines.
    """
    lines = [
        "| Contrast (exploratory) | Difference (pp) | 95% interval (pp) |",
        "|---|---|---|",
    ]
    wide_nulls = sorted(a for a in losses["arm"].unique().to_list() if a.startswith("N2-"))
    pairs = (*EXPLORATORY_CONTRASTS, *((chosen, null) for null in wide_nulls))
    for treatment, reference in pairs:
        scoped = subset(
            losses=losses,
            scope=f"lead{FULL_SWEEP_LEAD_DAY}",
            setting="primary",
            arms=(treatment, reference),
        )
        interval = paired_interval(
            scoped=scoped,
            treatment=treatment,
            reference=reference,
            metric=METRIC,
            level=EXPLORATORY_LEVEL,
        )
        if interval is not None:
            lines.append(
                f"| {treatment} − {reference} | "
                f"{interval['difference'] * PERCENTAGE_POINTS:+.3f} | "
                f"[{interval['lower_95'] * PERCENTAGE_POINTS:+.3f}, "
                f"{interval['upper_95'] * PERCENTAGE_POINTS:+.3f}] |"
            )
    return [*lines, ""]


def longer_lead_table(
    *, losses: pl.DataFrame, root: Path, product: WeatherProduct, lead_days: tuple[int, ...]
) -> tuple[list[str], pl.DataFrame]:
    """Tabulate each arm's error and gain over B0 at every lead-day, with the no-fit references.

    Args:
        losses: The combined losses.
        root: The output root.
        product: The weather product.
        lead_days: The lead-days.

    Returns:
        The report lines, and a table with one row per (lead-day, arm).
    """
    records = []
    for lead in lead_days:
        scope = f"lead{lead}"
        path = output_paths(root=root, product=product, lead_days=(lead,))[f"day{lead}"]
        frame = pl.read_parquet(path)
        fitted = subset(
            losses=losses,
            scope=scope,
            setting="primary",
            arms=tuple(losses["arm"].unique().to_list()),
        ).select("site", "time", "month", "seed", "arm", METRIC)
        if fitted.is_empty():
            continue
        references = reference_losses(frame=frame, lead_day=lead, losses=losses)
        both = pl.concat([fitted, references])
        for arm in sorted(both["arm"].unique().to_list()):
            absolute = bootstrap_absolute(losses=both, arm=arm, metric=METRIC)
            gain = (
                bootstrap_difference(losses=both, treatment=arm, reference="B0", metric=METRIC)
                if arm != "B0"
                else None
            )
            records.append(
                {
                    "lead_day": lead,
                    "arm": arm,
                    "error": absolute["value"],
                    "lower": absolute["lower_95"],
                    "upper": absolute["upper_95"],
                    "difference_to_b0": None if gain is None else gain["difference"],
                    "difference_lower": None if gain is None else gain["lower_95"],
                    "difference_upper": None if gain is None else gain["upper_95"],
                    "n_rows": absolute["n_rows"],
                    "fitted": arm not in REFERENCE_ARMS,
                }
            )
    table = pl.DataFrame(records)
    lines = [
        (
            "| Lead-day | Arm | Error (pp) | Difference to B0 (pp) | 95% interval of the difference"
            " | Rows |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for row in table.iter_rows(named=True):
        label = f"{row['lead_day']}" + (
            " (not usable for its early hours in a live service)" if row["lead_day"] == 0 else ""
        )
        diff = (
            ""
            if row["difference_to_b0"] is None
            else f"{row['difference_to_b0'] * PERCENTAGE_POINTS:+.3f}"
        )
        interval = (
            ""
            if row["difference_lower"] is None
            else f"[{row['difference_lower'] * PERCENTAGE_POINTS:+.3f}, "
            f"{row['difference_upper'] * PERCENTAGE_POINTS:+.3f}]"
        )
        lines.append(
            f"| {label} | {named(row['arm'])} | {row['error'] * PERCENTAGE_POINTS:.3f} | {diff} | "
            f"{interval} | {row['n_rows']} |"
        )
    return [*lines, ""], table


def control_lines(*, losses: pl.DataFrame) -> tuple[list[str], pl.DataFrame]:
    """Tabulate the positive control: does L1 recover each synthetic level shift?

    Args:
        losses: The combined losses.

    Returns:
        The report lines, and the table.
    """
    lines = [
        "| Shift | L1 − B0 (pp) | 99% interval (pp) | Baseline (pp) | Recovers (decision rule) |",
        "|---|---|---|---|---|",
    ]
    records = []
    for shift in POSITIVE_CONTROL_SHIFTS:
        scope = f"control_s{round(shift * 100):02d}"
        scoped = subset(losses=losses, scope=scope, setting="primary", arms=("L1", "B0"))
        interval = paired_interval(
            scoped=scoped, treatment="L1", reference="B0", metric=METRIC, level=PLANNED_LEVEL
        )
        if interval is None:
            continue
        reference_value = float(arm_values(losses=scoped, arm="B0", metric=METRIC)[0].mean())
        verdict = setting_verdict(interval=interval, reference_value=reference_value)
        records.append(
            {
                "shift": shift,
                "difference": interval["difference"],
                "lower": interval["lower_95"],
                "upper": interval["upper_95"],
                "reference_value": reference_value,
                "recovers": verdict == "helps",
            }
        )
        lines.append(
            f"| {shift:.0%} | {interval['difference'] * PERCENTAGE_POINTS:+.3f} | "
            f"[{interval['lower_95'] * PERCENTAGE_POINTS:+.3f}, "
            f"{interval['upper_95'] * PERCENTAGE_POINTS:+.3f}] | "
            f"{reference_value * PERCENTAGE_POINTS:.3f} | "
            f"{'yes' if verdict == 'helps' else verdict} |"
        )
    table = pl.DataFrame(records)
    if not table.is_empty() and table["recovers"].any():
        smallest = table.filter(pl.col("recovers"))["shift"].min()
        lines += ["", f"The smallest shift L1 recovers, the detection limit, is {smallest:.0%}."]
    else:
        lines += ["", "L1 recovers none of the shifts."]
    return [*lines, ""], table


def interval_lines(*, directory: Path) -> tuple[list[str], pl.DataFrame]:
    """Tabulate the 10% to 90% interval's coverage and width for B0 and L1.

    Args:
        directory: The checkpoint directory holding `intervals__<arm>.parquet`.

    Returns:
        The report lines, and a table with one row per (arm, sky).
    """
    files = sorted(directory.glob("intervals__*.parquet"))
    if not files:
        return [], pl.DataFrame()
    rows = pl.concat([pl.read_parquet(f) for f in files]).with_columns(
        forecast_index=pl.col("nwp_ghi") / pl.col("clear_sky_w_m2")
    )
    skies = [
        pl.when(
            (pl.col("forecast_index") >= low) & (pl.col("forecast_index") < high),
        )
        .then(pl.lit(name))
        .alias(name)
        for name, (low, high) in CLEAR_SKY_CLASSES.items()
    ]
    classified = rows.filter(pl.col("clear_sky_w_m2") > 0).with_columns(sky=pl.coalesce(skies))
    summary = (
        pl.concat([classified, classified.with_columns(sky=pl.lit("all"))])
        .group_by("arm", "sky")
        .agg(
            coverage=(
                (pl.col("actual") >= pl.col("lower")) & (pl.col("actual") <= pl.col("upper"))
            ).mean(),
            width=(pl.col("upper") - pl.col("lower")).mean(),
            n_rows=pl.len(),
        )
        .sort("arm", "sky")
    )
    lines = [
        (
            "| Arm | Sky (forecast clear-sky index) | Share inside the 10% to 90% interval "
            f"(nominal {NOMINAL_COVERAGE:.0%}) | Mean width (pp of capacity) | Rows |"
        ),
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['arm']} | {row['sky']} | {row['coverage']:.3f} | "
        f"{row['width'] * PERCENTAGE_POINTS:.2f} | {row['n_rows']} |"
        for row in summary.iter_rows(named=True)
    ]
    return [*lines, ""], summary


FINGERPRINT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("G-ID", "G-B0"),
    ("G-L1", "G-B0"),
    ("G-FP", "G-B0"),
    ("G-FP", "G-ID"),
    ("LOPO G-L1", "LOPO G-B0"),
    ("LOPO G-FP", "LOPO G-B0"),
)
"""Exploratory (treatment, reference) pairs of the global fingerprint mini-sweep. G-ID is the
in-sample upper bound for any static fingerprint, and the `LOPO` arms never saw the scored plant."""


def fingerprint_losses(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Relabel the global and leave-one-plant-out fits as the fingerprint mini-sweep's arms.

    Args:
        losses: The combined losses.

    Returns:
        The primary-setting global fits (B0 as `G-B0`, L1 as `G-L1`, plus `G-ID` and `G-FP`) and
        the leave-one-plant-out fits (as `LOPO G-B0`, `LOPO G-L1`, `LOPO G-FP`), all with scope
        `fingerprint`.
    """
    names = {"B0": "G-B0", "L1": "G-L1"}
    pooled = subset(
        losses=losses,
        scope="global",
        setting="primary",
        arms=("B0", "L1", *GLOBAL_ONLY_ARMS),
    ).with_columns(arm=pl.col("arm").replace(names))
    left_out = subset(
        losses=losses, scope="lopo", setting="primary", arms=("B0", "L1", "G-FP")
    ).with_columns(arm="LOPO " + pl.col("arm").replace({**names, "G-FP": "G-FP"}))
    return pl.concat([pooled, left_out], how="diagonal_relaxed").with_columns(
        scope=pl.lit("fingerprint")
    )


def fingerprint_lines(*, losses: pl.DataFrame) -> tuple[list[str], pl.DataFrame]:
    """Tabulate the global fingerprint mini-sweep: absolute errors, then paired contrasts.

    Args:
        losses: The combined losses.

    Returns:
        The report lines, and the table of absolute errors.
    """
    sweep = fingerprint_losses(losses=losses)
    if sweep.is_empty():
        return [], pl.DataFrame()
    records = []
    lines = [
        "| Arm (exploratory) | Mean absolute error (pp of capacity) | 95% interval (pp) | Rows |",
        "|---|---|---|---|",
    ]
    for arm in sorted(sweep["arm"].unique().to_list()):
        interval = bootstrap_absolute(losses=sweep, arm=arm, metric=METRIC)
        records.append(
            {
                "arm": arm,
                "error": interval["value"],
                "lower": interval["lower_95"],
                "upper": interval["upper_95"],
            }
        )
        lines.append(
            f"| {arm} | {interval['value'] * PERCENTAGE_POINTS:.3f} | "
            f"[{interval['lower_95'] * PERCENTAGE_POINTS:.3f}, "
            f"{interval['upper_95'] * PERCENTAGE_POINTS:.3f}] | {interval['n_rows']} |"
        )
    lines += [
        "",
        "| Contrast (exploratory) | Difference (pp) | 95% interval (pp) |",
        "|---|---|---|",
    ]
    for treatment, reference in FINGERPRINT_CONTRASTS:
        scoped = sweep.filter(pl.col("arm").is_in([treatment, reference]))
        interval = paired_interval(
            scoped=scoped,
            treatment=treatment,
            reference=reference,
            metric=METRIC,
            level=EXPLORATORY_LEVEL,
        )
        if interval is not None:
            lines.append(
                f"| {treatment} − {reference} | "
                f"{interval['difference'] * PERCENTAGE_POINTS:+.3f} | "
                f"[{interval['lower_95'] * PERCENTAGE_POINTS:+.3f}, "
                f"{interval['upper_95'] * PERCENTAGE_POINTS:+.3f}] |"
            )
    return [*lines, ""], pl.DataFrame(records)


def r5_lines(*, r5: pl.DataFrame) -> list[str]:
    """Tabulate R5's CRPS against B0's quantile model on the same rows.

    Args:
        r5: `r5_losses`' result.

    Returns:
        The report lines.
    """
    interval = paired_interval(
        scoped=r5, treatment="R5", reference="B0", metric=CRPS_METRIC, level=EXPLORATORY_LEVEL
    )
    if interval is None:
        return []
    absolute = {
        arm: bootstrap_absolute(losses=r5, arm=arm, metric=CRPS_METRIC) for arm in ("R5", "B0")
    }
    return [
        "| Arm (exploratory) | CRPS approximated from nine quantiles (pp of capacity) | Rows |",
        "|---|---|---|",
        *(
            f"| {arm} | {value['value'] * PERCENTAGE_POINTS:.3f} | {value['n_rows']} |"
            for arm, value in absolute.items()
        ),
        "",
        (
            f"R5 minus B0: {interval['difference'] * PERCENTAGE_POINTS:+.3f} pp, 95% interval "
            f"[{interval['lower_95'] * PERCENTAGE_POINTS:+.3f}, "
            f"{interval['upper_95'] * PERCENTAGE_POINTS:+.3f}] pp, on {interval['n_rows']} rows "
            f"per seed where at least {R5_MIN_EVENTS} residuals exist."
        ),
        "",
    ]


def named(arm: str) -> str:
    """Return how a table names an arm, with `ARM_LABELS` applied.

    Args:
        arm: The arm.

    Returns:
        The label, or the arm itself.
    """
    return ARM_LABELS.get(arm, arm)


def report_text(*, root: Path, product: WeatherProduct) -> str:
    """Build the whole report and save the tables the charts read.

    Args:
        root: The output root.
        product: The weather product.

    Returns:
        The report's markdown.
    """
    directory = root / product
    losses = pl.read_parquet(directory / f"losses_{product}.parquet")
    tables = directory / f"tables_{product}"
    tables.mkdir(exist_ok=True)
    paths = output_paths(root=root, product=product, lead_days=())
    lines = [f"# Lag features: report for the weather product `{product}`", ""]
    lines += ["## Build report", "", paths["report"].read_text()]

    if product == "ens_mean":
        shortlist_file = directory / "checkpoints" / "shortlist.json"
        chosen = json.loads(shortlist_file.read_text())["x"]
        frame = pl.read_parquet(
            output_paths(root=root, product=product, lead_days=(FULL_SWEEP_LEAD_DAY,))[
                f"day{FULL_SWEEP_LEAD_DAY}"
            ]
        )
        derived = pl.read_parquet(directory / "checkpoints" / "stage1_columns.parquet")
        frame = frame.join(derived, on=["site", "time"], how="left", maintain_order="left")
        references = reference_losses(frame=frame, lead_day=FULL_SWEEP_LEAD_DAY, losses=losses)
        phase1, phase1_table = phase1_lines(losses=losses, references=references, chosen=chosen)
        phase1_table.write_parquet(tables / "phase1.parquet")
        planned, planned_table = planned_lines(losses=losses, chosen=chosen)
        planned_table.write_parquet(tables / "planned.parquet")
        controls, control_table = control_lines(losses=losses)
        control_table.write_parquet(tables / "positive_control.parquet")
        coverage, coverage_table = interval_lines(directory=directory / "checkpoints")
        if not coverage_table.is_empty():
            coverage_table.write_parquet(tables / "coverage.parquet")
        lines += ["## Phase 1: the sweep, screening months 2024-12 to 2025-12", "", *phase1]
        lines += ["", "## Phase 2: the five planned contrasts", "", *planned]
        lines += [
            "## Exploratory contrasts at lead-day 1",
            "",
            *exploratory_lines(losses=losses, chosen=chosen),
        ]
        lines += ["## Absolute errors over all months", ""]
        for setting in ("primary", "sensitivity"):
            lines += absolute_lines(
                losses=losses, scope="lead1", setting=setting, title=f"Per-plant, {setting} setting"
            )
            lines += absolute_lines(
                losses=losses, scope="global", setting=setting, title=f"Global, {setting} setting"
            )
        lines += ["## Coverage and width of the 10% to 90% interval (descriptive)", "", *coverage]
        lines += ["## Positive control", "", *controls]
        fingerprint, fingerprint_table = fingerprint_lines(losses=losses)
        if not fingerprint_table.is_empty():
            fingerprint_table.write_parquet(tables / "fingerprint.parquet")
        lines += ["## Global fingerprint mini-sweep (exploratory)", "", *fingerprint]
        hours = pl.read_parquet(paths["stage1"])
        stage1 = pl.read_parquet(directory / "checkpoints" / "stage1_predictions.parquet")
        r5 = r5_losses(frame=frame, hours=hours, predictions=stage1, losses=losses)
        lines += ["## R5: residual quantiles added to B0 (exploratory)", "", *r5_lines(r5=r5)]
        lead_days = ENS_MEAN_LEAD_DAYS
    else:
        lead_days = IFS_LEAD_DAYS
    longer, longer_table = longer_lead_table(
        losses=losses, root=root, product=product, lead_days=lead_days
    )
    longer_table.write_parquet(tables / "by_lead.parquet")
    lines += ["## Error and gain over B0 by lead-day (exploratory)", "", *longer]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write the report for one weather product."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-product", choices=WEATHER_PRODUCTS, default="ens_mean")
    parser.add_argument("--output-root", type=Path, default=LAG_FEATURES_DIR)
    arguments = parser.parse_args()
    product: WeatherProduct = arguments.weather_product
    path = arguments.output_root / product / f"report_{product}.md"
    refuse_to_overwrite(paths=[path])
    text = report_text(root=arguments.output_root, product=product)
    path.write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
