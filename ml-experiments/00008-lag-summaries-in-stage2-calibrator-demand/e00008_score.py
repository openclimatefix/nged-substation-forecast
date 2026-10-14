"""Score experiment 00008's arms, save their predictions, and write the report.

Every arm is scored on the same cases: the validation runs' rows of the 20 series with an observed
power and a valid time before 2026-07-01. The quantile score is twice the mass-weighted mean
pinball loss of the 9 levels, in points of capacity (percent of effective capacity). Intervals
resample whole validation months and the three fitting seeds. The four planned contrasts are read
at 98.75% (a Bonferroni correction of 5%); every other contrast is exploratory and read at 95%.

Output: `work/arms/<arm>.parquet`, the prediction sets under
`data/ml-experiments/00008-.../predictions/`, and `work/results/{report.md, absolute.parquet,
contrasts.parquet}`.

    uv run python ml-experiments/00008-lag-summaries-in-stage2-calibrator-demand/e00008_score.py
"""

import itertools
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00008_common as common
import e00008_fit as fit
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
PLANNED_LEVEL: Final[float] = 98.75
N_RESAMPLES: Final[int] = 10_000
"""Resamples for the decision contrast, so each tail of the interval holds over 60 draws."""
DEFAULT_RESAMPLES: Final[int] = 2_000
SCOPES: Final[tuple[str, ...]] = (
    "all leads",
    "up to 48 h",
    "0-24 h",
    "1-2 d",
    "3-6 d",
    "7-10 d",
    "11-15 d",
)
GROUPS: Final[tuple[str, ...]] = ("Primary", "BSP", "GSP")
HEADLINE_GROUP: Final[str] = "Primary"
PLANNED: Final[dict[str, tuple[str, str]]] = {
    "C1 (primary): S1 minus L3": ("s1", "l3"),
    "C2: S2 minus L3": ("s2", "l3"),
    "C3: S3 minus L3": ("s3", "l3"),
    "C4: S4 minus R0c": ("s4", "r0c"),
}
"""The planned contrasts: treatment, reference."""
EXPLORATORY: Final[dict[str, tuple[str, str]]] = {
    "E1: S1 minus S2 (the raw power lags)": ("s1", "s2"),
    "E2: S4 minus L3": ("s4", "l3"),
    "E3: L3 minus R0c (00005's C2, refitted L3)": ("l3", "r0c"),
    "E4: S1 minus R0c": ("s1", "r0c"),
    "E5: S1 minus R2 (climatology)": ("s1", "r2"),
    "E6: S4 minus R2 (climatology)": ("s4", "r2"),
}
REPRODUCTION_TOLERANCE: Final[float] = 0.05


def collect_arms() -> dict[str, pl.DataFrame]:
    """Return every arm's validation quantiles and actual power over capacity.

    Returns:
        Each arm's name mapped to rows keyed by series, run, valid time, and seed. `l3_00005` is
        experiment 00005's own L3, kept only for the reproduction check.
    """
    arms = {name: pl.read_parquet(fit.OUT_DIR / f"{name}.parquet") for name in fit.ARM_COLUMNS}
    arms["r0c"] = pl.read_parquet(common.E00005_WORK_DIR / "calibrated" / "r0c_from2023.parquet")
    arms["r2"] = pl.read_parquet(common.E00005_WORK_DIR / "references" / "r2.parquet")
    arms["l3_00005"] = pl.read_parquet(
        common.E00005_WORK_DIR / "calibrated" / "l3_from2023.parquet"
    )
    return {
        name: frame.select(*KEYS, "seed", *common.LEVEL_COLUMNS, "actual")
        for name, frame in arms.items()
    }


def prepare(*, arms: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Restrict every arm to the shared cases and add the lead, loss, and group.

    Args:
        arms: `collect_arms()`.

    Returns:
        The same arms, each with `lead_hours`, `quantile_score`, and `group`.
    """
    shared: pl.DataFrame | None = None
    for frame in arms.values():
        cases = (
            frame.filter(
                pl.all_horizontal(pl.col("actual", *common.LEVEL_COLUMNS).is_finite())
                & (pl.col("valid_time") < HOLDOUT)
            )
            .select(KEYS)
            .unique()
        )
        shared = cases if shared is None else shared.join(cases, on=KEYS, how="inner")
    assert shared is not None
    groups = common.series_table(full=True).select(
        "time_series_id", group=pl.col("substation_type")
    )
    out = {}
    for name, frame in arms.items():
        kept = frame.join(shared, on=KEYS, how="semi").sort(*KEYS, "seed")
        quantiles = np.sort(kept.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
        score = common.pinball_quantile_score(quantiles=quantiles, actual=kept["actual"].to_numpy())
        out[name] = kept.join(groups, on="time_series_id", how="left").with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
            quantile_score=pl.Series(score * 100.0),
        )
    return out


def save_predictions(*, arms: dict[str, pl.DataFrame]) -> None:
    """Write each of this experiment's arms as an immutable prediction set, if not yet written."""
    for name in fit.ARM_COLUMNS:
        try:
            experiment_store.write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=name,
                frame=arms[name].select(*KEYS, "seed", *common.LEVEL_COLUMNS),
                kind="quantiles",
                row_set_id=common.ROW_SET_ID,
            )
        except FileExistsError:
            continue


def _scope(*, frame: pl.DataFrame, scope: str, group: str) -> pl.DataFrame:
    frame = frame.filter(pl.col("group") == group)
    if scope == "all leads":
        return frame
    if scope == "up to 48 h":
        return frame.filter(pl.col("lead_hours") < 48.0)
    low, high = next((lo, hi) for name, lo, hi in common.LEAD_BANDS if name == scope)
    return frame.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))


def contrast(
    *,
    arms: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    scope: str,
    group: str,
    level: float,
) -> dict[str, float | int | str]:
    """Return one paired contrast's estimate, interval, month agreement, and series agreement."""
    losses = experiment_store.losses_for_bootstrap(
        arm_losses={
            treatment: _scope(frame=arms[treatment], scope=scope, group=group),
            reference: _scope(frame=arms[reference], scope=scope, group=group),
        }
    )
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric="quantile_score",
        level=level,
        n_resamples=N_RESAMPLES
        if level == PLANNED_LEVEL and scope == "up to 48 h" and group == HEADLINE_GROUP
        else DEFAULT_RESAMPLES,
    )
    paired = (
        losses.filter(pl.col("arm") == treatment)
        .select("site", "time", "seed", "month", treatment_score=pl.col("quantile_score"))
        .join(
            losses.filter(pl.col("arm") == reference).select(
                "site", "time", "seed", reference_score=pl.col("quantile_score")
            ),
            on=["site", "time", "seed"],
        )
        .with_columns(difference=pl.col("treatment_score") - pl.col("reference_score"))
    )
    by_month = paired.group_by("month").agg(pl.col("difference").mean())
    by_series = paired.group_by("site").agg(pl.col("difference").mean())
    return {
        "treatment": treatment,
        "reference": reference,
        "scope": scope,
        "group": group,
        "level": level,
        "difference": float(np.mean(paired["difference"].to_numpy())),
        "lower": lower,
        "upper": upper,
        "months_negative": int((by_month["difference"] < 0).sum()),
        "months": by_month.height,
        "series_negative": int((by_series["difference"] < 0).sum()),
        "series": by_series.height,
    }


def absolute_table(*, arms: dict[str, pl.DataFrame], groups: tuple[str, ...]) -> pl.DataFrame:
    """Return every arm's mean quantile score by group and scope, with 95% intervals."""
    rows = []
    for name, frame in arms.items():
        for group in groups:
            for scope in SCOPES:
                part = experiment_store.losses_for_bootstrap(
                    arm_losses={name: _scope(frame=frame, scope=scope, group=group)}
                )
                interval = bootstrap.bootstrap_absolute(
                    losses=part, arm=name, metric="quantile_score"
                )
                rows.append({"arm": name, "group": group, "scope": scope, **interval})
    return pl.DataFrame(rows)


def reproduction_check(*, arms: dict[str, pl.DataFrame]) -> str:
    """Compare the refitted L3 with experiment 00005's L3 on the Primary substations."""
    lines = []
    for scope in ("up to 48 h", "all leads"):
        refit, old = (
            float(
                np.mean(
                    _scope(frame=arms[name], scope=scope, group=HEADLINE_GROUP)[
                        "quantile_score"
                    ].to_numpy()
                )
            )
            for name in ("l3", "l3_00005")
        )
        verdict = "within" if abs(refit - old) <= REPRODUCTION_TOLERANCE else "OUTSIDE"
        lines.append(f"{scope}: refitted L3 {refit:.3f}, 00005 L3 {old:.3f}, {verdict} tolerance")
    return "; ".join(lines)


def verdict(*, planned: pl.DataFrame) -> str:
    """Return the pre-registered verdict from C1 at leads up to 48 hours."""
    row = planned.filter(
        pl.col("treatment").eq("s1")
        & pl.col("reference").eq("l3")
        & (pl.col("scope") == "up to 48 h")
        & (pl.col("group") == HEADLINE_GROUP)
    ).row(0, named=True)
    supported = row["difference"] <= -0.15 and row["upper"] < 0 and row["months_negative"] >= 9
    refuted = row["difference"] >= 0.15 and row["lower"] > 0
    label = "supported" if supported else "refuted" if refuted else "inconclusive"
    return (
        f"{label}: C1 up to 48 h is {row['difference']:+.3f} ({row['lower']:+.3f} to "
        f"{row['upper']:+.3f} at {PLANNED_LEVEL}%), {row['months_negative']} of {row['months']} "
        "months negative"
    )


def _table(*, rows: pl.DataFrame, with_group: bool) -> list[str]:
    header = "| contrast | scope |" + (" group |" if with_group else "")
    lines = [header + " difference | interval | months negative | series negative |"]
    lines.append("|" + " --- |" * (header.count("|") + 3))
    for row in rows.iter_rows(named=True):
        group = f" {row['group']} |" if with_group else ""
        lines.append(
            f"| {row['label']} | {row['scope']} |{group} {row['difference']:+.3f} | "
            f"{row['lower']:+.3f} to {row['upper']:+.3f} | {row['months_negative']} of "
            f"{row['months']} | {row['series_negative']} of {row['series']} |"
        )
    return lines


def main() -> None:
    """Score every arm and write the report tables."""
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arms = prepare(arms=collect_arms())
    groups = tuple(g for g in GROUPS if g in set(arms["l3"]["group"].unique().to_list()))
    for name, frame in arms.items():
        frame.write_parquet(ARMS_DIR / f"{name}.parquet")
    if not common.SMOKE:
        save_predictions(arms=arms)
    planned = pl.DataFrame(
        [
            {
                "label": label,
                **contrast(
                    arms=arms, treatment=t, reference=r, scope=s, group=g, level=PLANNED_LEVEL
                ),
            }
            for label, (t, r) in PLANNED.items()
            for s in SCOPES
            for g in groups
        ]
    )
    exploratory = pl.DataFrame(
        [
            {
                "label": label,
                **contrast(
                    arms=arms, treatment=t, reference=r, scope=s, group=HEADLINE_GROUP, level=95.0
                ),
            }
            for label, (t, r) in EXPLORATORY.items()
            for s in SCOPES
        ]
    )
    pl.concat([planned, exploratory], how="diagonal").write_parquet(
        RESULTS_DIR / "contrasts.parquet"
    )
    absolute = absolute_table(arms=arms, groups=groups)
    absolute.write_parquet(RESULTS_DIR / "absolute.parquet")
    lines = [
        "# Experiment 00008: scores",
        "",
        f"Verdict by the pre-registered rule: {verdict(planned=planned)}",
        "",
        f"Reproduction check (Primary): {reproduction_check(arms=arms)}",
        "",
        "## Absolute quantile scores (points of capacity; 95% month-and-seed intervals)",
        "",
        "| arm | group | scope | quantile score |",
        "| --- | --- | --- | --- |",
    ]
    for name, group, scope in itertools.product(arms, groups, SCOPES):
        q = absolute.filter(
            (pl.col("arm") == name) & (pl.col("group") == group) & (pl.col("scope") == scope)
        ).row(0, named=True)
        lines.append(
            f"| {name} | {group} | {scope} | {q['value']:.2f} "
            f"({q['lower_95']:.2f} to {q['upper_95']:.2f}) |"
        )
    lines += [
        "",
        f"## Planned contrasts ({PLANNED_LEVEL}% intervals; negative favours the first arm)",
        "",
    ]
    lines += _table(rows=planned, with_group=True)
    lines += ["", "## Exploratory contrasts, Primary substations (95% intervals)", ""]
    lines += _table(rows=exploratory, with_group=False)
    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:5]))


if __name__ == "__main__":
    main()
