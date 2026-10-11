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
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from capacity_paths import (
    DERATED_PLAN_PATH,
    LAG_FOLLOWUPS2_LOSSES_PATH,
    LAG_FRAME_PATH,
    LAG_LOSSES_PATH,
    OUTPUT_RATIOS_PATH,
    claim_root,
    inputs_dir,
    load_events,
    load_tuned,
    record_for,
    scored_days,
)
from capacity_scores import scale_for
from scipy import stats
from studies.bootstrap import bootstrap_difference
from studies.capacity_estimators import ESTIMATOR_FAMILY, scale_for_rows
from studies.capacity_explain import (
    CRITERIA,
    explanation_sentence,
    fitted_physical_quantities,
    rubric_table,
)
from studies.capacity_scaling import (
    MIN_ORACLE_GAIN_POINTS,
    SCORED_COLUMN,
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
    report.text(f"- selected: {selection['selected']} (exploratory: {selection['exploratory']})")
    report.text(
        f"- survivors of the {selection['false_movement_gate']:.0%} gate: {selection['survivors']}"
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
    """Print the share of scored days on each fallback, per estimator and plant."""
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
        for plant, site in enumerate(record.sites):
            reason = scale.fallback_reason[plant, days]
            rows.append(
                {
                    "estimator": estimator,
                    "plant": site,
                    "short_history": float((reason == 1).mean()),
                    "null_estimate": float((reason == 2).mean()),
                }
            )
    report.heading("Fallback shares on the scored days (original record)")
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
            "capped_share_of_daylight_hours": float(
                removed[months == m].sum() / daylight[months == m].sum()
            ),
        }
        for m in events.scored_months
    ]
    report.heading("The ANM plant: capped hours")
    report.table(pl.DataFrame(shares))
    capped_day = removed.any(axis=1)
    recent = np.convolve(capped_day.astype(float), np.ones(7), mode="full")[: len(capped_day)] > 0
    rows = []
    for estimator in selection["shortlist"]:  # type: ignore[attr-defined]
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
    shortlist = [str(s) for s in selection["shortlist"]]  # type: ignore[attr-defined]
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
            f"- {setting}: difference {result['difference']:.3f} points; bootstrap 95% "
            f"[{result['boot_lower']:.3f}, {result['boot_upper']:.3f}]; month t-interval "
            f"[{result['t_lower']:.3f}, {result['t_upper']:.3f}]; threshold {threshold:.3f} points"
        )
    worth = all(r["boot_upper"] < 0 and r["difference"] <= -t for r, t in decisions.values())
    not_worth = all(r["boot_lower"] > -t for r, t in decisions.values())
    nc2_arm = f"nc2__{selected}"
    nc2_pair = pl.concat([b0["primary"], ours.filter(pl.col("arm") == nc2_arm).select(*keep)])
    nc2 = difference(losses=nc2_pair, treatment=nc2_arm, reference="B0")
    nc2_broken = nc2["boot_upper"] < 0
    verdict = (
        "unresolved"
        if nc2_broken
        else ("worth building" if worth else "not worth building" if not_worth else "unresolved")
    )
    report.text(
        f"- NC2 (deranged months): {nc2['difference']:.3f} points, bootstrap "
        f"[{nc2['boot_lower']:.3f}, {nc2['boot_upper']:.3f}]; "
        f"interval wholly below zero: {nc2_broken}"
    )
    report.text(f"- **decision rule verdict: {verdict}**")
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
    passed = oracle["boot_upper"] < 0
    report.text(f"- the oracle's interval lies wholly below zero (instrument can detect): {passed}")
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
        "Month-by-month and plant-by-plant differences (selected minus B0, primary)", level=3
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
    report.table(by_month)
    report.table(by_plant)
    events = load_events(root=root)
    low_site = record.sites[events.low_plant]
    low = wide.filter((pl.col("site") == low_site) & pl.col("month").is_in(events.low_months))
    low_gap = float(np.mean((low[p1_arm] - low["B0"]).to_numpy())) * POINTS
    report.text(
        f"- the three low-output plant-months together: {low_gap:.3f} points (plant label withheld)"
    )


def explainability(*, root: Path, report: Report) -> None:
    """Print the rubric and one explanation per estimator for a planted event and the low months."""
    events = load_events(root=root)
    record = record_for(root=root, record_id="original")
    orientation = json.loads((inputs_dir(root=root) / "orientation.json").read_text())
    report.heading("Explainability")
    rubric = rubric_table()
    report.text("Criteria: " + "; ".join(f"{key} = {question}" for key, question in CRITERIA))
    report.table(rubric)
    event = next(e for e in events.draws[0] if e.shape == "step")
    site = record.sites[event.plant]
    day = event.first_day + 20
    report.heading("A planted step in the tuning draw", level=3)
    report.text(f"Plant {site}, step of {event.size:.0%} starting {record.dates[event.first_day]}.")
    for estimator in ESTIMATOR_FAMILY:
        params = load_tuned(root=root, estimator=estimator)
        planted = scale_for(root=root, estimator=estimator, params=params, file_id="draw00")
        report.text(
            "- "
            + explanation_sentence(
                estimator=estimator,
                when=str(record.dates[day]),
                scale_mw=float(planted.scale_mw[event.plant, day]),
                earlier_scale_mw=float(planted.scale_mw[event.plant, event.first_day - 1]),
                earlier_days=day - (event.first_day - 1),
                relative_sd=None
                if np.isnan(planted.relative_sd[event.plant, day])
                else float(planted.relative_sd[event.plant, day]),
                where=f"plant {site}",
            )
        )
        if estimator in ("KF", "DP1"):
            quantities = fitted_physical_quantities(
                orientation=orientation[site],
                scale_mw=float(planted.scale_mw[event.plant, day]),
                static_p99_mw=float(record.static_p99_mw[event.plant]),
                relative_sd=None
                if np.isnan(planted.relative_sd[event.plant, day])
                else float(planted.relative_sd[event.plant, day]),
            )
            report.text(f"  - {estimator} fitted quantities: {quantities}")
    report.heading("The three real low months of the low-output plant (label withheld)", level=3)
    spans = {
        m: np.flatnonzero(record.dates.astype("datetime64[M]").astype(str) == m)
        for m in events.low_months
    }
    for month, indices in spans.items():
        middle = int(indices[len(indices) // 2])
        for estimator in ESTIMATOR_FAMILY:
            scale = scale_for(
                root=root,
                estimator=estimator,
                params=load_tuned(root=root, estimator=estimator),
                file_id="original",
            )
            earlier = int(indices[0])
            sd = scale.relative_sd[events.low_plant, middle]
            report.text(
                "- "
                + explanation_sentence(
                    estimator=estimator,
                    when=month,
                    scale_mw=float(scale.scale_mw[events.low_plant, middle]),
                    earlier_scale_mw=float(scale.scale_mw[events.low_plant, earlier]),
                    earlier_days=middle - earlier,
                    relative_sd=None if np.isnan(sd) else float(sd),
                    where="the low-output plant",
                )
            )


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
    fallbacks(root=root, report=report)
    audit_table(root=root, report=report)
    anm_dips(root=root, selection=selection, report=report)
    stage_b(root=root, selection=selection, report=report)
    explainability(root=root, report=report)
    path.write_text("\n".join(report.lines) + "\n")
    _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
