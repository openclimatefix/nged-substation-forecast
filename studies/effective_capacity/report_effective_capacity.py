"""Print every table and number the effective-capacity page quotes into `report.md`.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1143>. The
page quotes only numbers that this script prints. Run it after `fit_capacity_arms.py`.

**Sections:** the arms' columns and fallbacks; Stage A (recovery, delay, coverage, false movement,
the gate, the satellite-latency sensitivity, E1 and E2, the shortlist); the descriptive table of
each estimator's ratio at the low-output plant against the audit's ratios; the ANM plant's false
dips; Stage B (every arm's absolute error, P1 with both interval methods at both settings, the
decision rule, the negative control, the no-fit arms, the positive control, the exploratory
additions); and the explainability rubric with one explanation per estimator for a planted event and
for the three real low months. **The low-output plant's label never appears beside a date.**

Run it with `uv run python studies/effective_capacity/report_effective_capacity.py` (add `--smoke`
for the smoke root).
"""

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from build_capacity_inputs import ORIENTATION_END
from capacity_paths import (
    DERATED_PLAN_PATH,
    LAG_FOLLOWUPS2_LOSSES_PATH,
    LAG_FRAME_PATH,
    LAG_LOSSES_PATH,
    OUTPUT_RATIOS_PATH,
    claim_root,
    draw_id,
    load_events,
    load_tuned,
    record_for,
    scored_days,
)
from capacity_scores import scale_for
from scipy import stats
from studies.baselines import MIN_OBSERVED_CLEAR_SKY_SHARE
from studies.bootstrap import bootstrap_difference
from studies.capacity_estimators import (
    ESTIMATOR_FAMILY,
    FLOOR_SHARE_OF_STATIC,
    HM_LEVELS,
    ISSUE_LAG_DAYS,
    MIN_VALID_DAYS_OF_HISTORY,
    REFERENCE_END,
    REFERENCE_START,
    scale_for_rows,
)
from studies.capacity_explain import CRITERIA, rubric_table
from studies.capacity_posthoc import HYSTERESIS_DROP, HYSTERESIS_PERSIST_DAYS
from studies.capacity_recovery import (
    FALLBACK_SHARE_GATE,
    FALSE_MOVEMENT_GATE,
    FALSE_MOVEMENT_THRESHOLD,
    FALSE_MOVEMENT_TRAILING_DAYS,
    FOOTPRINT_AFTER_DAYS,
    FOOTPRINT_BEFORE_DAYS,
    expected_factor,
)
from studies.capacity_scaling import (
    MIN_ORACLE_GAIN_POINTS,
    SCORED_COLUMN,
    SettingResult,
    capped_daylight_share,
    decision_verdict,
    multiplied_forecast_scores,
    recovery_share,
)
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("report_effective_capacity")

THRESHOLD_SHARE: Final[float] = 0.02
"""The decision threshold: 2% of B0's error."""

POINTS: Final[float] = 100.0
"""Errors are printed in points of capacity (percentage points of the static power)."""

DIP_THRESHOLD: Final[float] = 0.05
"""A day dips when the ANM-blind estimate is more than this share below the ANM-aware one."""


class Report:
    """Collects the report's lines."""

    def __init__(self) -> None:
        """Start an empty report."""
        self.lines: list[str] = []

    def heading(self, text: str, *, level: int = 2) -> None:
        """Add a heading."""
        self.lines += ["", "#" * level + " " + text, ""]

    def text(self, text: str) -> None:
        """Add a line of text."""
        self.lines.append(text)

    def table(self, frame: pl.DataFrame) -> None:
        """Add a table, with floats to four decimals."""
        header = "| " + " | ".join(frame.columns) + " |"
        rule = "|" + "|".join("---" for _ in frame.columns) + "|"
        rows = [
            "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |"
            for row in frame.iter_rows()
        ]
        self.lines += ["", header, rule, *rows, ""]


def published_b0(*, setting: str) -> pl.DataFrame:
    """Return the lag study's B0 per-row losses at one setting, lead-day 1."""
    return pl.read_parquet(LAG_LOSSES_PATH).filter(
        (pl.col("scope") == "lead1") & (pl.col("arm") == "B0") & (pl.col("setting") == setting)
    )


def difference(*, losses: pl.DataFrame, treatment: str, reference: str) -> dict[str, float]:
    """Return a paired difference in points, with a month bootstrap and a month t-interval.

    Args:
        losses: Per-row losses of both arms, with `arm`, `site`, `time`, `seed`, `month`, and the
            scored column.
        treatment: The arm compared.
        reference: The arm it is compared against.

    Returns:
        The difference, the bootstrap 95% interval, the t-interval over months, all in points.
    """
    interval = bootstrap_difference(
        losses=losses, treatment=treatment, reference=reference, metric=SCORED_COLUMN
    )
    paired = (
        losses.filter(pl.col("arm") == treatment)
        .select("site", "time", "seed", "month", treatment=pl.col(SCORED_COLUMN))
        .join(
            losses.filter(pl.col("arm") == reference).select(
                "site", "time", "seed", reference=pl.col(SCORED_COLUMN)
            ),
            on=["site", "time", "seed"],
        )
        .group_by("month")
        .agg(difference=(pl.col("treatment") - pl.col("reference")).mean())
    )
    values = paired["difference"].to_numpy()
    half = float(stats.t.ppf(0.975, df=len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {
        "difference": interval["difference"] * POINTS,
        "boot_lower": interval["lower_95"] * POINTS,
        "boot_upper": interval["upper_95"] * POINTS,
        "t_lower": (values.mean() - half) * POINTS,
        "t_upper": (values.mean() + half) * POINTS,
    }


def reading(*, result: dict[str, float]) -> str:
    """Name a difference's direction from its month t-interval, noting a borderline reading.

    Args:
        result: `difference`'s result, where a positive difference means the treatment is worse.

    Returns:
        `worse`, `better`, or `no detectable difference` by the month t-interval, with
        `, borderline` added where the bootstrap interval excludes zero but the t-interval does not.
    """
    if result["t_lower"] > 0:
        return "worse"
    if result["t_upper"] < 0:
        return "better"
    boot_excludes = result["boot_lower"] > 0 or result["boot_upper"] < 0
    direction = "worse" if result["difference"] > 0 else "better"
    return f"{direction}, borderline" if boot_excludes else "no detectable difference"


def bootstrap_text(*, result: dict[str, float]) -> str:
    """Return a difference's point estimate and bootstrap interval as text."""
    return (
        f"{result['difference']:.3f} points, bootstrap "
        f"[{result['boot_lower']:.3f}, {result['boot_upper']:.3f}]"
    )


def stage_a(*, root: Path, report: Report) -> dict[str, Any]:
    """Print Stage A and return `selection.json`."""
    selection = json.loads((root / "selection.json").read_text())
    summary = pl.read_parquet(root / "tables" / "stage_a_summary.parquet")
    events = pl.read_parquet(root / "tables" / "stage_a_events.parquet")
    report.heading("Stage A: recovery of planted losses")
    report.text(
        "Primary metric: mean over events of the gap between the planted-to-original scale ratio "
        "and the expected factor, as a share (lower is better). Selection draws only."
    )
    report.table(summary.sort("mean_selection_error"))
    by_shape = events.group_by("estimator", "shape").agg(
        error=pl.col("error").mean(),
        delay=pl.col("delay").mean(),
        coverage=pl.col("coverage").mean(),
    )
    report.heading("By event shape", level=3)
    report.table(by_shape.sort("estimator", "shape"))
    detected = (
        events.filter(pl.col("shape") == "step")
        .group_by("estimator")
        .agg(
            detected_share=pl.col("delay").is_not_null().mean(),
            mean_delay_days=pl.col("delay").mean(),
        )
    )
    report.heading("Detection of steps", level=3)
    report.table(detected.sort("estimator"))
    report.heading("Selection", level=3)
    report.text(f"- selected: {selection['selected']} (status: {selection['status']})")
    report.text(
        f"- no-skill reference S0 mean recovery error: {selection['s0_error']:.4f}; "
        "every candidate "
        "must beat it"
    )
    if selection["selected"] is None:
        report.text("- **no estimator beat doing nothing; P1 is not run. This is the finding.**")
    if selection["status"] == "ungated_pick":
        report.text("- no candidate passed every gate: the pick is labelled an ungated pick")
    report.text(
        f"- survivors of the {selection['false_movement_gate']:.0%} false-movement gate, the "
        f"{selection['fallback_share_gate']:.0%} fallback gate, and S0: {selection['survivors']}"
    )
    report.text(f"- shortlist: {selection['shortlist']}")
    report.text(f"- representatives (tuning draw): {selection['representatives']}")
    for contrast in selection["contrasts"]:
        report.text(
            f"- {contrast['name']}: {contrast['treatment']} minus {contrast['reference']} = "
            f"{contrast['difference'] * POINTS:.3f} points, 97.5% t-interval "
            f"[{contrast['lower'] * POINTS:.3f}, {contrast['upper'] * POINTS:.3f}] "
            f"-> {contrast['verdict']}"
        )
    for note in selection["contrasts_not_computed"]:
        report.text(f"- not computed: {note}")
    return selection


def fallbacks(*, root: Path, report: Report) -> None:
    """Print the share of scored days on each fallback, per estimator, with no plant labels."""
    record = record_for(root=root, record_id="original")
    days = np.unique(scored_days(record=record, events=load_events(root=root)))
    rows = []
    for estimator in ESTIMATOR_FAMILY:
        scale = scale_for(
            root=root,
            estimator=estimator,
            params=load_tuned(root=root, estimator=estimator),
            file_id="original",
        )
        reason = scale.fallback_reason[:, days]
        null_by_plant = (reason == 2).mean(axis=1)
        rows.append(
            {
                "estimator": estimator,
                "short_history_mean_over_plants": float((reason == 1).mean()),
                "null_estimate_mean_over_plants": float(null_by_plant.mean()),
                "null_estimate_largest_plant": float(null_by_plant.max()),
            }
        )
    report.heading("Fallback shares on the scored days (original record, plants pooled)")
    report.table(pl.DataFrame(rows))


def audit_table(*, root: Path, report: Report) -> None:
    """Print each estimator's monthly ratio at the low-output plant against the audit's ratios."""
    events = load_events(root=root)
    record = record_for(root=root, record_id="original")
    plan = json.loads(DERATED_PLAN_PATH.read_text())
    plant = events.low_plant
    months = record.dates.astype("datetime64[M]").astype(str)
    rows = []
    for estimator in ESTIMATOR_FAMILY:
        scale = scale_for(
            root=root,
            estimator=estimator,
            params=load_tuned(root=root, estimator=estimator),
            file_id="original",
        ).scale_mw[plant]
        monthly = {m: float(np.nanmean(scale[months == m])) for m in events.scored_months}
        normal = float(np.median([v for m, v in monthly.items() if m not in events.low_months]))
        rows.append(
            {
                "estimator": estimator,
                **{
                    f"low month {i + 1}": monthly[m] / normal
                    for i, m in enumerate(events.low_months)
                },
            }
        )
    audit = {
        f"low month {i + 1}": plan["ratio_by_month"][m] for i, m in enumerate(events.low_months)
    }
    rows.append({"estimator": "audit ratio to neighbours", **audit})
    report.heading("Descriptive: the low-output plant's three low months")
    report.text(
        "Each estimator's mean scale in the month over its median across the other scored months. "
        "Used in no selection. The plant's label is deliberately not printed."
    )
    report.table(pl.DataFrame(rows))


def anm_dips(*, root: Path, selection: dict[str, Any], report: Report) -> None:
    """Print the ANM plant's capped-hour shares and each shortlisted estimator's false dips."""
    events = load_events(root=root)
    aware = record_for(root=root, record_id="original")
    blind = record_for(root=root, record_id="anm_blind")
    plant = events.anm_plant
    removed = np.isfinite(blind.power_mw[plant]) & ~np.isfinite(aware.power_mw[plant])
    daylight = aware.clear_sky_w_m2[plant] > 0
    months = aware.dates.astype("datetime64[M]").astype(str)
    shares = [
        {
            "month": m,
            "capped_share_of_daylight_hours": capped_daylight_share(
                removed=removed[months == m], daylight=daylight[months == m]
            ),
        }
        for m in events.scored_months
    ]
    report.heading("The ANM plant: capped hours")
    report.table(pl.DataFrame(shares))
    capped_day = removed.any(axis=1)
    recent = np.convolve(capped_day.astype(float), np.ones(7), mode="full")[: len(capped_day)] > 0
    rows = []
    for estimator in selection["shortlist"]:
        params = load_tuned(root=root, estimator=estimator)
        aware_scale = scale_for(root=root, estimator=estimator, params=params, file_id="original")
        blind_scale = scale_for(root=root, estimator=estimator, params=params, file_id="anm_blind")
        ratio = blind_scale.scale_mw[plant] / aware_scale.scale_mw[plant]
        target = np.zeros(len(ratio), dtype=bool)
        target[
            np.unique(np.concatenate([np.flatnonzero(months == m) for m in events.scored_months]))
        ] = True
        shifted = np.concatenate([[False, False], recent[:-2]])
        for label, mask in (
            ("after capped days", shifted & target),
            ("other days", ~shifted & target),
        ):
            rows.append(
                {
                    "estimator": estimator,
                    "days": label,
                    "share_dipping_more_than_5_percent": float(
                        (ratio[mask] < 1 - DIP_THRESHOLD).mean()
                    ),
                    "mean_ratio_blind_to_aware": float(np.nanmean(ratio[mask])),
                }
            )
    report.heading("The ANM plant: false dips, ANM-blind against ANM-aware (no fits)")
    report.table(pl.DataFrame(rows))


def stage_b(*, root: Path, selection: dict[str, Any], report: Report) -> None:  # noqa: PLR0915
    """Print Stage B."""
    selected = str(selection["selected"])
    events = load_events(root=root)
    shortlist = [str(s) for s in selection["shortlist"]]
    frame = pl.read_parquet(LAG_FRAME_PATH)
    ours = pl.read_parquet(root / "losses_stage_b.parquet")
    keep = ["arm", "site", "time", "seed", "month", "setting", SCORED_COLUMN]
    b0 = {s: published_b0(setting=s).select(*keep) for s in ("primary", "sensitivity")}
    report.heading(
        "Stage B: absolute error of every arm (points of the static 99th-percentile power)"
    )
    arms = pl.concat([b0["primary"], b0["sensitivity"], ours.select(*keep)])
    report.table(
        arms.group_by("arm", "setting")
        .agg(mean_error_points=pl.col(SCORED_COLUMN).mean() * POINTS, rows=pl.len())
        .sort("arm", "setting")
    )
    columns = json.loads((root / "arm_columns.json").read_text())
    report.heading("Every arm's columns", level=3)
    for arm, names in columns.items():
        report.text(f"- {arm}: {names}")
    # P1.
    report.heading(
        "P1 (planned): the selected causal arm minus B0, pooled over the plants", level=3
    )
    p1_arm = f"causal__{selected}"
    decisions = {}
    for setting in ("primary", "sensitivity"):
        pair = pl.concat(
            [
                b0[setting],
                ours.filter((pl.col("arm") == p1_arm) & (pl.col("setting") == setting)).select(
                    *keep
                ),
            ]
        )
        result = difference(losses=pair, treatment=p1_arm, reference="B0")
        threshold = THRESHOLD_SHARE * float(np.mean(b0[setting][SCORED_COLUMN].to_numpy())) * POINTS
        decisions[setting] = (result, threshold)
        report.text(
            f"- {setting}: difference {result['difference']:.3f} points; **headline: month "
            f"t-interval [{result['t_lower']:.3f}, {result['t_upper']:.3f}]**; bootstrap 95% "
            f"[{result['boot_lower']:.3f}, {result['boot_upper']:.3f}]; threshold "
            f"{threshold:.3f} points; reading: {reading(result=result)}"
        )
    nc2_arm = f"nc2__{selected}"
    nc2_pair = pl.concat([b0["primary"], ours.filter(pl.col("arm") == nc2_arm).select(*keep)])
    nc2 = difference(losses=nc2_pair, treatment=nc2_arm, reference="B0")
    nc2_broken = nc2["boot_upper"] < 0
    report.text(
        f"- NC2 (deranged months): {nc2['difference']:.3f} points, bootstrap "
        f"[{nc2['boot_lower']:.3f}, {nc2['boot_upper']:.3f}]; "
        f"interval wholly below zero: {nc2_broken}"
    )
    # Power benchmark and RS (no fits).
    # 2025-08 and 2025-09 have no ratio in the audit (the ANM plant is out of the neighbour set
    # there), so the alternative ratio, which uses every neighbour with data, fills them.
    ratios = pl.read_parquet(OUTPUT_RATIOS_PATH).select(
        "site", "month", ratio=pl.coalesce("ratio", "ratio_alt")
    )
    factor = frame.join(ratios, on=["site", "month"], how="left", maintain_order="left")["ratio"]
    if factor.null_count():
        msg = "a lag-frame row has no neighbour ratio"
        raise ValueError(msg)
    primary_point = published_b0(setting="primary")
    bound = multiplied_forecast_scores(
        baseline_losses=primary_point, frame=frame, factor=factor.to_numpy()
    ).with_columns(arm=pl.lit("capacity_only_bound"), setting=pl.lit("primary"))
    record = record_for(root=root, record_id="original")
    scales = np.load(root / "scales_stage_b.npz")
    rs_factor = scale_for_rows(record=record, scale_mw=scales[f"{p1_arm}__scale"], rows=frame)
    rs = multiplied_forecast_scores(
        baseline_losses=primary_point, frame=frame, factor=rs_factor
    ).with_columns(arm=pl.lit("RS"), setting=pl.lit("primary"))
    report.heading("No-fit arms (exploratory)", level=3)
    for name, arm_losses in (("capacity_only_bound", bound), ("RS", rs)):
        pair = pl.concat([b0["primary"], arm_losses.select(*keep)])
        result = difference(losses=pair, treatment=name, reference="B0")
        report.text(
            f"- {name} minus B0: {result['difference']:.3f} points, bootstrap "
            f"[{result['boot_lower']:.3f}, {result['boot_upper']:.3f}], month t-interval "
            f"[{result['t_lower']:.3f}, {result['t_upper']:.3f}]"
        )
    # Positive control.
    report.heading("Positive control: a 30% step in 3 scored months at every plant", level=3)
    control = ours.filter(pl.col("arm").str.starts_with("control_")).select(*keep)
    oracle = difference(losses=control, treatment="control_oracle", reference="control_b0")
    chosen = difference(losses=control, treatment="control_selected", reference="control_b0")
    real = decisions["primary"][0]["difference"]
    report.text(f"- oracle minus B0 on planted data: {bootstrap_text(result=oracle)}")
    report.text(f"- selected minus B0 on planted data: {bootstrap_text(result=chosen)}")
    report.text(f"- selected minus B0 on real data: {real:.3f} points")
    passed = bool(oracle["boot_upper"] < 0)
    report.text(f"- the oracle's interval lies wholly below zero (instrument can detect): {passed}")
    verdict = decision_verdict(
        settings=[
            SettingResult(r["difference"], r["boot_lower"], r["boot_upper"], t)
            for r, t in decisions.values()
        ],
        nc2_upper=nc2["boot_upper"],
        control_passed=passed,
    )
    label = " (ungated pick)" if selection["status"] == "ungated_pick" else ""
    report.text(f"- **decision rule verdict{label}: {verdict}**")
    counts = Counter(e.plant for e in events.control)
    report.text(
        "- control steps per plant (sorted, no labels): "
        + ", ".join(str(n) for n in sorted(counts.get(i, 0) for i in range(len(record.sites))))
    )
    share = recovery_share(
        planted_gain=-chosen["difference"], real_gain=-real, oracle_gain=-oracle["difference"]
    )
    if share is None:
        report.text(
            f"- recovery share: not computed, because the oracle's gain is below "
            f"{MIN_ORACLE_GAIN_POINTS} points"
        )
    else:
        report.text(
            f"- recovery share (selected planted gain minus real gain, over oracle gain): "
            f"{share:.3f}"
        )
    # Exploratory.
    report.heading("Exploratory: other arms against B0, and hindsight against causal", level=3)
    for name in shortlist:
        for kind in ("causal", "hindsight"):
            arm = f"{kind}__{name}"
            pair = pl.concat(
                [
                    b0["primary"],
                    ours.filter((pl.col("arm") == arm) & (pl.col("setting") == "primary")).select(
                        *keep
                    ),
                ]
            )
            result = difference(losses=pair, treatment=arm, reference="B0")
            report.text(f"- {arm} minus B0: {bootstrap_text(result=result)}")
        pair = (
            ours.filter(pl.col("setting") == "primary")
            .filter(pl.col("arm").is_in([f"hindsight__{name}", f"causal__{name}"]))
            .select(*keep)
        )
        result = difference(
            losses=pair, treatment=f"hindsight__{name}", reference=f"causal__{name}"
        )
        report.text(f"- hindsight__{name} minus causal__{name}: {bootstrap_text(result=result)}")
    # 2x2 of day of year and normalisation.
    no_doy = (
        pl.read_parquet(LAG_FOLLOWUPS2_LOSSES_PATH)
        .filter(pl.col("arm") == "B0-noDOY")
        .select(*keep)
        .with_columns(arm=pl.lit("B0-noDOY"))
    )
    ours_no_doy = ours.filter(pl.col("arm") == f"noDOY__{selected}").select(*keep)
    cells = {
        "B0 (with day of year, static)": b0["primary"],
        "B0-noDOY (no day of year, static)": no_doy,
        f"{p1_arm} (with day of year, normalised)": ours.filter(
            (pl.col("arm") == p1_arm) & (pl.col("setting") == "primary")
        ).select(*keep),
        f"noDOY__{selected} (no day of year, normalised)": ours_no_doy,
    }
    report.heading("2x2 of day of year and normalisation (primary setting)", level=3)
    report.table(
        pl.DataFrame(
            [
                {
                    "cell": k,
                    "mean_error_points": float(np.mean(v[SCORED_COLUMN].to_numpy())) * POINTS,
                }
                for k, v in cells.items()
            ]
        )
    )
    # CRPS.
    crps_ours = (
        ours.filter(
            (pl.col("arm") == p1_arm)
            & (pl.col("setting") == "primary")
            & pl.col("crps_capped_fraction_of_capacity").is_not_null()
        )
        if "crps_capped_fraction_of_capacity" in ours.columns
        else None
    )
    if crps_ours is not None and crps_ours.height:
        b0_crps = published_b0(setting="primary")
        p1_crps = float(np.mean(crps_ours["crps_capped_fraction_of_capacity"].to_numpy())) * POINTS
        b0_value = float(np.mean(b0_crps["crps_capped_fraction_of_capacity"].to_numpy())) * POINTS
        report.text(
            f"- CRPS (fraction of capacity): {p1_arm} {p1_crps:.3f} points, "
            f"B0 {b0_value:.3f} points"
        )
    # Month and plant.
    report.heading(
        "Month-by-month and plant-by-plant differences (selected minus B0, primary, unweighted)",
        level=3,
    )
    pair = pl.concat(
        [
            b0["primary"],
            ours.filter((pl.col("arm") == p1_arm) & (pl.col("setting") == "primary")).select(*keep),
        ]
    )
    wide = (
        pair.group_by("arm", "month", "site")
        .agg(error=pl.col(SCORED_COLUMN).mean())
        .pivot(on="arm", index=["month", "site"], values="error")
    )
    by_month = (
        wide.group_by("month")
        .agg(difference_points=(pl.col(p1_arm) - pl.col("B0")).mean() * POINTS)
        .sort("month")
    )
    by_plant = (
        wide.group_by("site")
        .agg(difference_points=(pl.col(p1_arm) - pl.col("B0")).mean() * POINTS)
        .sort("site")
    )
    report.text("Unweighted: each plant-month counts once, whatever its number of rows.")
    report.table(by_month.rename({"difference_points": "unweighted_difference_points"}))
    report.text(
        "Unweighted by plant, sorted from the largest difference, with no plant labels: "
        + ", ".join(
            f"{v:.3f}" for v in sorted(by_plant["difference_points"].to_list(), reverse=True)
        )
    )
    both = _paired_rows(losses=pair, treatment=p1_arm, reference="B0")
    low_site = record.sites[events.low_plant]
    low_mask = (pl.col("site") == low_site) & pl.col("month").is_in(events.low_months)
    low_gap = _mean(both.filter(low_mask)["diff"]) * POINTS
    report.text(
        f"- the three low-output plant-months together, row weighted: {low_gap:.3f} points "
        "(plant label withheld)"
    )
    posthoc(
        root=root,
        report=report,
        context=PostHocContext(
            ours=ours,
            b0=b0,
            keep=keep,
            frame=frame,
            selected=selected,
            events=events,
            record=record,
            both=both,
            low_mask=low_mask,
            nc2_upper=nc2["boot_upper"],
            control_passed=passed,
        ),
    )


@dataclass(frozen=True)
class PostHocContext:
    """What the post hoc sections read from `stage_b`."""

    ours: pl.DataFrame
    b0: dict[str, pl.DataFrame]
    keep: list[str]
    frame: pl.DataFrame
    selected: str
    events: Any
    record: Any
    both: pl.DataFrame
    low_mask: pl.Expr
    nc2_upper: float
    control_passed: bool


WINTER_MONTHS: Final[tuple[str, ...]] = ("12", "01", "02")
"""The calendar months of the winter split."""

POSTHOC_LABEL: Final[str] = "POST HOC"


def _paired_rows(*, losses: pl.DataFrame, treatment: str, reference: str) -> pl.DataFrame:
    """Return every row's treatment and reference loss and their difference (a fraction)."""
    return (
        losses.filter(pl.col("arm") == treatment)
        .select("site", "time", "seed", "month", treatment=pl.col(SCORED_COLUMN))
        .join(
            losses.filter(pl.col("arm") == reference).select(
                "site", "time", "seed", reference=pl.col(SCORED_COLUMN)
            ),
            on=["site", "time", "seed"],
        )
        .with_columns(diff=pl.col("treatment") - pl.col("reference"))
    )


def _arm_vs_b0(*, context: PostHocContext, arm_losses: pl.DataFrame, arm: str) -> dict[str, float]:
    """Return an arm's primary-setting difference from B0."""
    pair = pl.concat([context.b0["primary"], arm_losses.select(*context.keep)])
    return difference(losses=pair, treatment=arm, reference="B0")


def _mean(values: pl.Series) -> float:
    """Return a series' mean as a float."""
    return float(np.mean(values.to_numpy()))


def _interval_text(*, result: dict[str, float]) -> str:
    return (
        f"{result['difference']:.3f} points; month t-interval "
        f"[{result['t_lower']:.3f}, {result['t_upper']:.3f}]; bootstrap 95% "
        f"[{result['boot_lower']:.3f}, {result['boot_upper']:.3f}]; {reading(result=result)}"
    )


def plant_months(rows: pl.DataFrame) -> int:
    """Return how many distinct plant-months the rows cover."""
    return rows.select(pl.struct("site", "month").n_unique()).item()


def posthoc(*, root: Path, report: Report, context: PostHocContext) -> None:  # noqa: PLR0915
    """Print every analysis that was not pre-registered, each labelled POST HOC."""
    ours, b0, keep = context.ours, context.b0, context.keep
    selected = context.selected
    arm = f"causal__{selected}"
    report.heading(f"{POSTHOC_LABEL}: analyses beyond the pre-registered ones")
    report.text(
        f"**Everything in this section is {POSTHOC_LABEL}. None of it changes the pre-registered "
        "selection or the verdict.**"
    )
    chosen = ours.filter((pl.col("arm") == arm) & (pl.col("setting") == "primary")).select(*keep)
    pair = pl.concat([b0["primary"], chosen])
    # Winter split.
    report.heading(f"{POSTHOC_LABEL}: winter against the other months (selected minus B0)", level=3)
    is_winter = pl.col("month").str.slice(5, 2).is_in(list(WINTER_MONTHS))
    for label, part in (
        ("December to February", pair.filter(is_winter)),
        ("the other months", pair.filter(~is_winter)),
    ):
        months = part["month"].n_unique()
        result = difference(losses=part, treatment=arm, reference="B0")
        report.text(f"- {label} ({months} months, row weighted): {_interval_text(result=result)}")
    # Break-even.
    both = context.both
    low = both.filter(context.low_mask)
    rest = both.filter(~context.low_mask)
    gap_low, gap_rest = _mean(low["diff"]), _mean(rest["diff"])
    break_even = gap_rest / (gap_rest - gap_low)
    report.heading(f"{POSTHOC_LABEL}: break-even share of low-output rows", level=3)
    report.text(
        f"- row-weighted selected-minus-B0 gap: {gap_low * POINTS:.3f} points on the low-output "
        f"plant-months, {gap_rest * POINTS:.3f} points on every other row"
    )
    report.text(
        f"- the selected arm beats B0 overall only if more than {break_even:.1%} of rows are "
        f"low-output plant-months; here the share is {low.height / both.height:.1%} "
        "of rows and "
        f"{plant_months(low) / plant_months(both):.1%} of plant-months "
        f"({plant_months(low)} of {plant_months(both)})"
    )
    # Capacity-only bound against the same rule.
    report.heading(
        f"{POSTHOC_LABEL}: the capacity-only neighbour-ratio bound against the same decision rule",
        level=3,
    )
    ratios = pl.read_parquet(OUTPUT_RATIOS_PATH).select(
        "site", "month", ratio=pl.coalesce("ratio", "ratio_alt")
    )
    factor = context.frame.join(ratios, on=["site", "month"], how="left", maintain_order="left")[
        "ratio"
    ].to_numpy()
    results = []
    for setting in ("primary", "sensitivity"):
        baseline = published_b0(setting=setting)
        bound = multiplied_forecast_scores(
            baseline_losses=baseline, frame=context.frame, factor=factor
        ).with_columns(arm=pl.lit("capacity_only_bound"), setting=pl.lit(setting))
        bound_pair = pl.concat([b0[setting], bound.select(*keep)])
        result = difference(losses=bound_pair, treatment="capacity_only_bound", reference="B0")
        threshold = THRESHOLD_SHARE * float(np.mean(b0[setting][SCORED_COLUMN].to_numpy())) * POINTS
        results.append(
            SettingResult(
                result["difference"], result["boot_lower"], result["boot_upper"], threshold
            )
        )
        report.text(f"- {setting}: {_interval_text(result=result)}; threshold {threshold:.3f}")
    verdict = decision_verdict(
        settings=results, nc2_upper=context.nc2_upper, control_passed=context.control_passed
    )
    report.text(
        f"- the pre-registered rule applied to the bound, with the selected arm's negative control "
        f"and positive control: **{verdict}**"
    )
    # The low-output plant's neighbouring months.
    report.heading(f"{POSTHOC_LABEL}: the low-output plant around March 2026", level=3)
    low_plant = context.record.sites[context.events.low_plant]
    rows = []
    for month in ("2026-02", "2026-03", "2026-04"):
        selected_rows = both.filter((pl.col("site") == low_plant) & (pl.col("month") == month))
        rows.append(
            {
                "month": month,
                "B0_error_points": _mean(selected_rows["reference"]) * POINTS,
                "selected_error_points": _mean(selected_rows["treatment"]) * POINTS,
                "rows": selected_rows.height,
            }
        )
    report.table(pl.DataFrame(rows))
    # Hindsight.
    report.heading(f"{POSTHOC_LABEL}: why hindsight is worse than causal", level=3)
    hind = f"hindsight__{selected}"
    hind_rows = _paired_rows(
        losses=ours.filter(pl.col("setting") == "primary").select(*keep),
        treatment=hind,
        reference=arm,
    )
    monthly = (
        hind_rows.group_by("site", "month")
        .agg(gap=pl.col("diff").mean(), rows=pl.len())
        .sort("gap", descending=True)
    )
    worst = monthly.head(2)
    report.text(
        "- the two plant-months with the largest hindsight-minus-causal gap (labels withheld): "
        + ", ".join(f"{g * POINTS:.1f} points" for g in worst["gap"].to_list())
    )
    keys = pl.struct("site", "month")
    worst_keys = worst.select(keys.alias("k"))["k"].to_list()
    drop = ~keys.is_in(worst_keys)
    for label, part in (
        ("all plant-months", ours.filter(pl.col("setting") == "primary")),
        (
            "without those two plant-months",
            ours.filter(pl.col("setting") == "primary").filter(drop),
        ),
    ):
        pair_h = pl.concat(
            [
                part.filter(pl.col("arm") == hind).select(*keep),
                part.filter(pl.col("arm") == arm).select(*keep),
            ]
        )
        result = difference(losses=pair_h, treatment=hind, reference=arm)
        report.text(f"- {hind} minus {arm}, {label}: {_interval_text(result=result)}")
    earlier = [
        (site, f"{int(month[:4]) - 1}{month[4:]}")
        for site, month in zip(worst["site"].to_list(), worst["month"].to_list(), strict=True)
    ]
    for name in (arm, hind):
        scaled = ours.filter((pl.col("arm") == name) & (pl.col("setting") == "primary"))
        year_before = scaled.filter(
            pl.struct("site", "month").is_in(
                [{"site": site, "month": month} for site, month in earlier]
            )
        )
        report.text(
            f"- {name}: mean scaled power (power over the arm's scale, a fraction) is "
            f"{_mean(scaled.filter(context.low_mask)['actual_scaled']):.3f} in the low-output "
            f"plant-months and {_mean(year_before['actual_scaled']):.3f} in the same plant one "
            "year before the two worst plant-months"
        )
    report.text(
        "- reading: the hindsight scale is lower than the causal scale in those earlier months, "
        "so the training rows there are divided by a smaller number and carry larger targets, and "
        "the day-of-year column is consistent with the fitted trees carrying them into the same "
        "calendar months a year later. The floor is not the cause. The arm works as defined, so "
        "the page calls hindsight a reference, not a bound."
    )
    # New arms.
    path = root / "losses_stage_b_posthoc.parquet"
    if path.exists():
        report.heading(
            f"{POSTHOC_LABEL}: other estimators as Stage B arms (primary setting)", level=3
        )
        extra = pl.read_parquet(path)
        scales = np.load(root / "scales_stage_b_posthoc.npz")
        summary = pl.read_parquet(root / "tables" / "posthoc_stage_a.parquet")
        movement = dict(
            zip(summary["estimator"], summary["false_movement_real_windows"], strict=True)
        )
        rows = []
        for name in sorted(extra["arm"].unique().to_list()):
            estimator = name.removeprefix("posthoc_causal__")
            arm_losses = extra.filter(pl.col("arm") == name).select(*keep)
            result = _arm_vs_b0(context=context, arm_losses=arm_losses, arm=name)
            rs_factor = scale_for_rows(
                record=context.record, scale_mw=scales[name], rows=context.frame
            )
            rs = multiplied_forecast_scores(
                baseline_losses=published_b0(setting="primary"),
                frame=context.frame,
                factor=rs_factor,
            ).with_columns(arm=pl.lit("RS"), setting=pl.lit("primary"))
            rs_result = _arm_vs_b0(context=context, arm_losses=rs, arm="RS")
            rows.append(
                {
                    "estimator": estimator,
                    "fit_minus_B0_points": result["difference"],
                    "fit_t_lower": result["t_lower"],
                    "fit_t_upper": result["t_upper"],
                    "fit_boot_lower": result["boot_lower"],
                    "fit_boot_upper": result["boot_upper"],
                    "RS_minus_B0_points": rs_result["difference"],
                    "RS_t_lower": rs_result["t_lower"],
                    "RS_t_upper": rs_result["t_upper"],
                    "false_movement_23_windows": movement.get(estimator),
                }
            )
        report.table(pl.DataFrame(rows))
        report.text(
            "RS is B0's forecast times the estimator's scale over its training-row mean (no fit)."
        )
        stage_a_table = pl.read_parquet(root / "tables" / "posthoc_stage_a.parquet")
        report.heading(f"{POSTHOC_LABEL}: Stage A recovery and false movement", level=3)
        report.table(stage_a_table)
        hyst_section(root=root, report=report, context=context)


def hyst_section(*, root: Path, report: Report, context: PostHocContext) -> None:
    """Print the POST HOC detail on HYST, the arm designed after the main results."""
    path = root / "losses_stage_b_posthoc_hyst.parquet"
    if not path.exists():
        return
    hyst = pl.read_parquet(path)
    keep, b0 = context.keep, context.b0
    arm = "posthoc_causal__HYST"
    report.heading(f"{POSTHOC_LABEL}: HYST, designed after the main results", level=3)
    for setting in ("primary", "sensitivity"):
        part = hyst.filter(pl.col("setting") == setting).select(*keep)
        result = difference(losses=pl.concat([b0[setting], part]), treatment=arm, reference="B0")
        report.text(f"- {setting}, HYST minus B0: {_interval_text(result=result)}")
    chosen = hyst.filter(pl.col("setting") == "primary").select(*keep)
    both = _paired_rows(losses=pl.concat([b0["primary"], chosen]), treatment=arm, reference="B0")
    low = both.filter(context.low_mask)
    rest = both.filter(~context.low_mask)
    report.text(
        f"- primary, row weighted: {_mean(low['diff']) * POINTS:.3f} points on the three "
        f"low-output plant-months, {_mean(rest['diff']) * POINTS:.3f} points on every other row"
    )
    is_winter = pl.col("month").str.slice(5, 2).is_in(list(WINTER_MONTHS))
    pair = pl.concat([b0["primary"], chosen])
    for label, part in (
        ("December to February", pair.filter(is_winter)),
        ("the other months", pair.filter(~is_winter)),
    ):
        result = difference(losses=part, treatment=arm, reference="B0")
        report.text(f"- primary, {label}: {_interval_text(result=result)}")
    months = sorted(context.events.scored_months)
    for label, subset in (("first 9 months", months[:9]), ("last 9 months", months[9:])):
        result = difference(
            losses=pair.filter(pl.col("month").is_in(subset)), treatment=arm, reference="B0"
        )
        report.text(
            f"- primary, {label} ({subset[0]} to {subset[-1]}): {_interval_text(result=result)}"
        )
    # How often HYST moves.
    scale = np.load(root / "scales_stage_b_posthoc_hyst.npz")[arm]
    record = context.record
    months_of_day = record.dates.astype("datetime64[M]").astype(str)
    static = record.static_p99_mw[:, None]
    moved = np.abs(scale / static - 1.0) > 1e-9
    rows = []
    for month in months:
        mask = months_of_day == month
        rows.append(
            {
                "month": month,
                "share_of_plant_days_off_static": float(moved[:, mask].mean()),
                "plants_off_static_on_some_day": int(moved[:, mask].any(axis=1).sum()),
                "mean_scale_over_static": float((scale[:, mask] / static).mean()),
            }
        )
    scored = np.isin(months_of_day, months)
    report.table(pl.DataFrame(rows))
    report.text(
        f"- over all scored days: HYST left the static power on "
        f"{float(moved[:, scored].mean()):.1%} of plant-days; its mean scale is "
        f"{float((scale[:, scored] / static).mean()):.3f} of the static power"
    )
    # CRPS.
    crps = hyst.filter(
        (pl.col("setting") == "primary") & pl.col("crps_capped_fraction_of_capacity").is_not_null()
    )
    if crps.height:
        baseline = published_b0(setting="primary")
        report.text(
            "- CRPS (fraction of capacity, primary, quantile models): HYST "
            f"{_mean(crps['crps_capped_fraction_of_capacity']) * POINTS:.3f} points, B0 "
            f"{_mean(baseline['crps_capped_fraction_of_capacity']) * POINTS:.3f} points"
        )
    else:
        report.text("- CRPS: skipped")
    # Stage A by shape.
    events_path = root / "tables" / "posthoc_stage_a_events.parquet"
    if events_path.exists():
        shapes = (
            pl.read_parquet(events_path)
            .filter(pl.col("estimator").is_in(["HYST", "DSKF", "KF", "S0"]))
            .group_by("estimator", "shape")
            .agg(error=pl.col("error").mean(), delay=pl.col("delay").mean(), events=pl.len())
            .sort("estimator", "shape")
        )
        report.text("Stage A recovery by event shape (selection draws), HYST with DSKF, KF, S0:")
        report.table(shapes)


def _scale_changes(*, scale: np.ndarray, plant: int, first: int, day: int) -> float:
    """Return the relative change from the median of the 30 days before `first` to `day`."""
    baseline = float(np.nanmedian(scale[plant, max(first - 30, 0) : first]))
    return float(scale[plant, day] / baseline - 1.0)


def explainability(*, root: Path, report: Report) -> None:
    """Print the rubric, each estimator's change across all planted events, and the low months."""
    events = load_events(root=root)
    record = record_for(root=root, record_id="original")
    report.heading("Explainability")
    report.table(rubric_table())
    report.text("Criteria: " + "; ".join(f"{key} = {question}" for key, question in CRITERIA))
    report.heading(
        "Estimated change 20 days into each planted event, against the 30 days before it", level=3
    )
    rows = []
    for estimator in ESTIMATOR_FAMILY:
        planted = scale_for(
            root=root,
            estimator=estimator,
            params=load_tuned(root=root, estimator=estimator),
            file_id="draw00",
        )
        estimated, expected = [], []
        for event in events.draws[0]:
            day = event.first_day + 20
            estimated.append(
                _scale_changes(
                    scale=planted.scale_mw, plant=event.plant, first=event.first_day, day=day
                )
            )
            expected.append(float(expected_factor(event=event, days=len(record.dates))[day] - 1.0))
        rows.append(
            {
                "estimator": estimator,
                "events": len(estimated),
                "median_estimated_change": float(np.median(estimated)),
                "lowest_estimated_change": float(np.min(estimated)),
                "highest_estimated_change": float(np.max(estimated)),
                "median_planted_change": float(np.median(expected)),
            }
        )
    report.table(pl.DataFrame(rows))
    report.heading(
        "Estimated change in each real low month, against the 30 days before it", level=3
    )
    rows = []
    months = record.dates.astype("datetime64[M]").astype(str)
    for estimator in ESTIMATOR_FAMILY:
        scale = scale_for(
            root=root,
            estimator=estimator,
            params=load_tuned(root=root, estimator=estimator),
            file_id="original",
        )
        row: dict[str, object] = {"estimator": estimator}
        for number, month in enumerate(events.low_months, start=1):
            indices = np.flatnonzero(months == month)
            row[f"low month {number}"] = _scale_changes(
                scale=scale.scale_mw,
                plant=events.low_plant,
                first=int(indices[0]),
                day=int(indices[len(indices) // 2]),
            )
        rows.append(row)
    report.table(pl.DataFrame(rows))
    fitted = pl.read_parquet(root / "tables" / "fitted_quantities.parquet")
    report.heading("Fitted physical quantities of KF and DP1 (plant labels withheld)", level=3)
    report.table(
        fitted.drop("plant")
        .sort("estimator", "tilt_deg")
        .select(
            "estimator",
            "tilt_deg",
            "azimuth_deg",
            "ac_limit_mw",
            "fitted_dc_capacity_mw",
            "series",
            "series_min_mw",
            "series_median_mw",
            "series_max_mw",
        )
    )


EXAMPLE_STEP_SIZE: Final[float] = 0.30
"""The planted step the example figure draws is the selection draw's step closest to this size."""

PROFILE_FALLBACK_SHARE: Final[float] = 0.5
"""An event has no seasonal profile when DSKF fell back on this share of its scored days or more."""

EXAMPLE_DAYS_BEFORE: Final[int] = 30
"""How many days before the planted step the example figure starts."""


def constants(*, report: Report) -> None:
    """Print the estimator and scoring constants that the page quotes."""
    report.heading("Constants the page quotes")
    report.text(
        f"- the scale for a target day reads whole days up to {ISSUE_LAG_DAYS} days earlier"
    )
    report.text(
        f"- a day is valid when its observed hours hold at least "
        f"{MIN_OBSERVED_CLEAR_SKY_SHARE:.0%} of its clear-sky energy; an estimator falls back to "
        f"the static power with fewer than "
        f"{MIN_VALID_DAYS_OF_HISTORY} valid days of history; the scale is floored at "
        f"{FLOOR_SHARE_OF_STATIC:.0%} of the static power"
    )
    report.text(f"- calibration year: {REFERENCE_START} to {REFERENCE_END}")
    report.text(f"- plant orientations fitted on data before {ORIENTATION_END:%Y-%m-%d}")
    report.text(f"- HM levels: {', '.join(f'{level:g}' for level in HM_LEVELS)}")
    report.text(
        f"- HYST: follows a deseasonalised drop of more than {HYSTERESIS_DROP:.0%} that lasts "
        f"{HYSTERESIS_PERSIST_DAYS} days, and releases after {HYSTERESIS_PERSIST_DAYS} "
        f"recovered days"
    )
    report.text(
        f"- planted events: footprint of {FOOTPRINT_BEFORE_DAYS} days before and "
        f"{FOOTPRINT_AFTER_DAYS} days after the planted month"
    )
    report.text(
        f"- false movement: a day moves when its estimate is more than "
        f"{FALSE_MOVEMENT_THRESHOLD:.0%} from the median of the previous "
        f"{FALSE_MOVEMENT_TRAILING_DAYS} days; gate at {FALSE_MOVEMENT_GATE:.0%}; fallback-share "
        f"gate at {FALLBACK_SHARE_GATE:.0%}; decision threshold {THRESHOLD_SHARE:.0%} of B0's error"
    )


def design_facts(*, root: Path, report: Report) -> None:
    """Print the counts the page quotes about the rows, the plants, and the planted losses."""
    events = load_events(root=root)
    frame = pl.read_parquet(LAG_FRAME_PATH)
    plant_months = (
        frame.with_columns(month=pl.col("time").dt.strftime("%Y-%m"))
        .select("site", "month")
        .unique()
        .height
    )
    record = record_for(root=root, record_id="original")
    report.heading("Design facts")
    report.text(f"- lead-day-1 daylight hours scored: {frame.height}")
    report.text(f"- plants: {len(record.sites)}; scored months: {len(events.scored_months)}")
    report.text(f"- plant-months with rows: {plant_months}")
    per_draw = [len(draw) for draw in events.draws]
    report.text(
        f"- planted events per draw: {min(per_draw)} to {max(per_draw)} (a step, a ramp, and an "
        f"intermittent loss at each plant, except that the plant with a commissioning ramp has "
        f"room for fewer); selection draws: {len(events.draws) - 1}; tuning draws: 1"
    )
    sizes = np.array([event.size for draw in events.draws for event in draw])
    report.text(
        f"- planted loss sizes: {sizes.min():.0%} to {sizes.max():.0%} of power "
        f"(uniform), {len(sizes)} events over {len(events.draws)} draws"
    )
    report.text(f"- record: {len(record.dates)} days")


def events_without_profile(*, root: Path) -> set[tuple[int, int, int]]:
    """Return the selection-draw events on which DSKF and HYST have no seasonal profile.

    DSKF divides by a profile built only from earlier calendar years, so near the start of the
    record it falls back to the static power, and HYST then holds the static power too. An event
    is without a profile when DSKF fell back on at least half of its scored days.

    Args:
        root: The output root.

    Returns:
        (draw index, plant index, first day) of each such event.
    """
    events = load_events(root=root)
    without: set[tuple[int, int, int]] = set()
    for index, draw in enumerate(events.draws):
        if index == 0:
            continue
        fell_back = scale_for(
            root=root, estimator="DSKF", params={}, file_id=draw_id(index=index)
        ).fallback_reason
        for event in draw:
            start, stop = event.scored_days
            if float(np.mean(fell_back[event.plant, start:stop] != 0)) >= PROFILE_FALLBACK_SHARE:
                without.add((index, event.plant, event.first_day))
    return without


def hyst_detection(*, root: Path, report: Report) -> None:
    """Print how often HYST, DSKF, and KF detected planted steps, with and without a profile."""
    events = load_events(root=root)
    without = events_without_profile(root=root)
    table = pl.read_parquet(root / "tables" / "posthoc_stage_a_events.parquet")
    first_days = {
        (index, event.plant, event.shape, event.size): event.first_day
        for index, draw in enumerate(events.draws)
        for event in draw
    }
    flagged = table.with_columns(
        without_profile=pl.struct("draw", "plant", "shape", "size").map_elements(
            lambda row: (
                (
                    row["draw"],
                    row["plant"],
                    first_days[(row["draw"], row["plant"], row["shape"], row["size"])],
                )
                in without
            ),
            return_dtype=pl.Boolean,
        )
    )
    report.heading(f"{POSTHOC_LABEL}: HYST and DSKF detection, with and without a seasonal profile")
    total = flagged.filter(pl.col("estimator") == "S0").height
    report.text(
        f"- selection-draw events: {total}; without a profile (DSKF fell back on at least "
        f"{PROFILE_FALLBACK_SHARE:.0%} of the scored days): "
        f"{flagged.filter((pl.col('estimator') == 'S0') & pl.col('without_profile')).height}"
    )
    rows = []
    for estimator in ("HYST", "DSKF", "KF", "S0"):
        mine = flagged.filter(pl.col("estimator") == estimator)
        for label, part in (
            ("all events", mine),
            ("events with a profile", mine.filter(~pl.col("without_profile"))),
        ):
            steps = part.filter(pl.col("shape") == "step")
            rows.append(
                {
                    "estimator": estimator,
                    "events": label,
                    "n_events": part.height,
                    "mean_error": float(part["error"].mean()),  # ty: ignore[invalid-argument-type]
                    "n_steps": steps.height,
                    "steps_detected": int(steps["delay"].is_not_null().sum()),
                    "steps_detected_share": float(steps["delay"].is_not_null().mean()),  # ty: ignore[invalid-argument-type]
                }
            )
    report.table(pl.DataFrame(rows))
    missed = flagged.filter(
        (pl.col("estimator") == "HYST")
        & (pl.col("shape") == "step")
        & pl.col("delay").is_null()
        & ~pl.col("without_profile")
    )
    report.text(
        f"- HYST steps missed although a profile existed: {missed.height}, with sizes "
        f"{', '.join(f'{size:.0%}' for size in sorted(missed['size'].to_list()))}"
    )


def example_event(*, root: Path, report: Report) -> None:
    """Print and save one planted step, and every estimator's scale ratio across it.

    The step is chosen by a stated rule: among the selection draws' steps, the one whose size is
    closest to `EXAMPLE_STEP_SIZE`. Days are counted from the step's first day and no plant label
    or calendar date is written.
    """
    events = load_events(root=root)
    days = len(record_for(root=root, record_id="original").dates)
    without = events_without_profile(root=root)
    candidates = [
        (abs(event.size - EXAMPLE_STEP_SIZE), index, event)
        for index, draw in enumerate(events.draws)
        if index > 0
        for event in draw
        if event.shape == "step" and (index, event.plant, event.first_day) not in without
    ]
    _, draw_index, event = min(candidates, key=lambda item: (item[0], item[1]))
    first, last = event.first_day - EXAMPLE_DAYS_BEFORE, event.scored_days[1]
    truth = expected_factor(event=event, days=days)[first:last]
    rows = []
    for estimator in (*ESTIMATOR_FAMILY, "HYST"):
        params = {} if estimator == "HYST" else load_tuned(root=root, estimator=estimator)
        planted = scale_for(
            root=root, estimator=estimator, params=params, file_id=draw_id(index=draw_index)
        )
        original = scale_for(root=root, estimator=estimator, params=params, file_id="original")
        ratio = (
            planted.scale_mw[event.plant, first:last] / original.scale_mw[event.plant, first:last]
        )
        estimated = planted.fallback_reason[event.plant, first:last] == 0
        rows += [
            {
                "estimator": estimator,
                "day": int(day - event.first_day),
                "ratio": float(value) if has_estimate else None,
                "planted_factor": float(factor),
            }
            for day, value, factor, has_estimate in zip(
                range(first, last), ratio, truth, estimated, strict=True
            )
        ]
    table = pl.DataFrame(rows, schema_overrides={"ratio": pl.Float64})
    path = root / "tables" / "example_event.parquet"
    refuse_to_overwrite(paths=[path])
    table.write_parquet(path)
    report.heading("Example planted step (the figure of one planted step)")
    report.text(
        f"- rule: the selection draw's step closest to {EXAMPLE_STEP_SIZE:.0%} among those with a "
        f"seasonal profile (see the HYST detection section); this one is a "
        f"{event.size:.1%} loss lasting {event.month_days} days, from draw {draw_index}; days "
        f"are counted from its first day, and its plant is not named"
    )
    report.text(
        "- a day on which an estimator fell back to the static power (the convex programmes are "
        "solved only for the days a scoring needs) has no ratio and is left blank"
    )
    report.text(f"- file: {path.name}, {table.height} rows")


def kf_seasonal_shape(*, root: Path, report: Report) -> None:
    """Print how the raw Kalman filter level varies through the calendar year, plant by plant.

    A plant's level in a calendar month is the mean, over every day of that calendar month in the
    record, of the filter's issue-time scale on days that did not fall back to the static power,
    over the plant's median scale on the same days. Plants are numbered, not labelled.
    """
    record = record_for(root=root, record_id="original")
    scale = scale_for(
        root=root, estimator="KF", params=load_tuned(root=root, estimator="KF"), file_id="original"
    )
    calendar_month = (record.dates.astype("datetime64[M]").astype(int) % 12) + 1
    rows = []
    for plant in range(len(record.sites)):
        usable = (scale.fallback_reason[plant] == 0) & np.isfinite(scale.scale_mw[plant])
        median = float(np.median(scale.scale_mw[plant][usable]))
        for month in range(1, 13):
            mask = usable & (calendar_month == month)
            rows.append(
                {
                    "plant": plant,
                    "calendar_month": month,
                    "level_over_median": float(scale.scale_mw[plant][mask].mean() / median),
                    "days": int(mask.sum()),
                }
            )
    table = pl.DataFrame(rows)
    path = root / "tables" / "kf_seasonal_shape.parquet"
    refuse_to_overwrite(paths=[path])
    table.write_parquet(path)
    report.heading("Seasonal shape of the raw Kalman filter level")
    report.text(
        "Each plant's mean level in a calendar month over its median level, on days that did not "
        "fall back to the static power. Across plants, plant labels withheld."
    )
    by_month = (
        table.group_by("calendar_month")
        .agg(
            lowest=pl.col("level_over_median").min(),
            median=pl.col("level_over_median").median(),
            highest=pl.col("level_over_median").max(),
        )
        .sort("calendar_month")
    )
    report.table(by_month)
    december = by_month.filter(pl.col("calendar_month") == 12).row(0, named=True)
    deepest = by_month.sort("median").row(0, named=True)
    report.text(
        f"- December: {december['lowest']:.2f} to {december['highest']:.2f} of each plant's median "
        f"(median across plants {december['median']:.2f})"
    )
    report.text(
        f"- the lowest calendar month by the median across plants is month "
        f"{deepest['calendar_month']} at {deepest['median']:.2f}"
    )
    report.text(f"- the file: {path.name}")


def main() -> int:
    """Write `report.md`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Read and write under _smoke/.")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    path = root / "report.md"
    refuse_to_overwrite(paths=[path])
    report = Report()
    report.text(
        "# Effective capacity: report" + (" (SMOKE RUN, NOT A RESULT)" if arguments.smoke else "")
    )
    selection = stage_a(root=root, report=report)
    no_pick = selection["selected"] is None
    fallbacks(root=root, report=report)
    audit_table(root=root, report=report)
    anm_dips(root=root, selection=selection, report=report)
    if not no_pick:
        stage_b(root=root, selection=selection, report=report)
    explainability(root=root, report=report)
    constants(report=report)
    design_facts(root=root, report=report)
    hyst_detection(root=root, report=report)
    example_event(root=root, report=report)
    kf_seasonal_shape(root=root, report=report)
    path.write_text("\n".join(report.lines) + "\n")
    _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
