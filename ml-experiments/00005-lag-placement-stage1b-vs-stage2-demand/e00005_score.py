"""Assemble experiment 00005's arms, save their predictions, score them, and write the report.

Every arm is scored on the same cases: the validation runs' rows of the 20 series with an observed
power and a valid time before 2026-07-01. The quantile score is twice the mass-weighted mean
pinball loss of the 9 levels, and the error of the median is its absolute error, both in points of
capacity (percent of effective capacity). Intervals resample whole validation months and the three
fitting seeds. The five planned contrasts are read at 99%; every other contrast is exploratory and
read at 95%.

Output: `work/arms/<arm>.parquet`, the prediction sets under
`data/ml-experiments/00005-.../predictions/`, and `work/results/{report.md, absolute.parquet,
contrasts.parquet}`.

    uv run python ml-experiments/00005-lag-placement-stage1b-vs-stage2-demand/e00005_score.py
"""

import itertools
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00005_calibrate as calibrate
import e00005_common as common
import e00005_ens_stats as ens_stats
import e00005_references as references
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
ARM_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id",
    "power_fcst_init_time",
    "valid_time",
    "seed",
    *common.LEVEL_COLUMNS,
    "actual",
    "group",
    "lead_hours",
    "quantile_score",
    "median_error",
)
PLANNED_LEVEL: Final[float] = 99.0
N_RESAMPLES: Final[int] = 10_000
"""Resamples for the decision contrasts, so each tail of a 99% interval holds 50 draws."""
DEFAULT_RESAMPLES: Final[int] = 2_000
DECISION_SCOPES: Final[tuple[str, ...]] = ("all leads", "up to 48 h")
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
"""The 16 Primary substations are the headline; BSP and GSP have 2 series each."""
RAW_ARMS: Final[dict[str, tuple[str, str]]] = {
    **{f"r0_{config}": (config, "a") for config in common.CONFIGS},
    **{f"l2_{config}": (config, "ab") for config, s in common.CONFIGS.items() if s.stage1b},
}
"""Uncalibrated arms: the member quantiles of stage 1a (R0) or of stage 1a plus 1b (L2)."""
CALIBRATED_ARMS: Final[tuple[str, ...]] = tuple(
    f"{arm}_{config}"
    for config, settings in common.CONFIGS.items()
    if settings.calibrators
    for arm in calibrate.ARMS
)

PLANNED: Final[dict[str, tuple[str, str]]] = {
    "C1 (primary): L2c minus L3": ("l2c_from2023", "l3_from2023"),
    "C2: L3 minus R0c": ("l3_from2023", "r0c_from2023"),
    "C3: L2c minus R0c": ("l2c_from2023", "r0c_from2023"),
    "C4: L2+L3 minus L2c": ("l2l3_from2023", "l2c_from2023"),
    "C5: L2+L3 minus L3": ("l2l3_from2023", "l3_from2023"),
}
"""The planned contrasts: treatment, reference. All from the training window starting 2023-09."""
EXPLORATORY: Final[dict[str, tuple[str, str]]] = {
    "E1: L3 minus R0 (uncalibrated reference)": ("l3_from2023", "r0_from2023"),
    "E2: L2 minus L3": ("l2_from2023", "l3_from2023"),
    "E3: L2c minus L2 (what calibrating adds to stage 1b)": ("l2c_from2023", "l2_from2023"),
    "E4: R0c minus R0 (what calibrating adds to stage 1a)": ("r0c_from2023", "r0_from2023"),
    "E5: L2+L3 minus L2": ("l2l3_from2023", "l2_from2023"),
    "E6: L2 minus R0 (stage 1b alone)": ("l2_from2023", "r0_from2023"),
    "E7: C1 from the 2019-09 window": ("l2c_full", "l3_full"),
    "E8: C2 from the 2019-09 window": ("l3_full", "r0c_full"),
    "E9: C3 from the 2019-09 window": ("l2c_full", "r0c_full"),
    "E10: R0, window 2023-09 minus 2019-09": ("r0_from2023", "r0_full"),
    "E11: R0, window 2024-04 minus 2023-09": ("r0_from2024", "r0_from2023"),
    "E12: R0 without calendar minus with (window 2023-09)": ("r0_from2023_nocal", "r0_from2023"),
    "E13: L2 without calendar minus with (window 2023-09)": ("l2_from2023_nocal", "l2_from2023"),
    "E14: L3 minus R1 (ENS-trained, lags as inputs)": ("l3_from2023", "r1"),
    "E15: L2c minus R1": ("l2c_from2023", "r1"),
    "E16: L3 minus R2 (climatology)": ("l3_from2023", "r2"),
    "E17: R0c minus R2": ("r0c_from2023", "r2"),
}


def _summaries(*, config: str, kind: str, actual: dict[int, np.ndarray]) -> pl.DataFrame:
    """Return one configuration and kind's validation statistics quantiles, with the actual."""
    frame = pl.concat(
        [
            pl.read_parquet(path)
            for path in sorted((ens_stats.STATS_DIR / "validation").glob("init=*.parquet"))
        ]
    ).filter((pl.col("config") == config) & (pl.col("kind") == kind))
    pieces = []
    for series_id, values in actual.items():
        part = frame.filter(pl.col("time_series_id") == series_id)
        index = np.asarray(common.grid_index(times=part["valid_time"]))
        pieces.append(
            part.select(
                "time_series_id",
                "power_fcst_init_time",
                "valid_time",
                "seed",
                *common.LEVEL_COLUMNS,
            ).with_columns(actual=pl.Series(common.take(values=values, index=index)))
        )
    return pl.concat(pieces)


def collect_arms() -> dict[str, pl.DataFrame]:
    """Return every arm's validation quantiles and actual power over capacity.

    Returns:
        Each arm's name mapped to rows keyed by series, run, valid time, and seed, with the 9
        quantile columns and `actual`.
    """
    series = common.series_table()
    arrays = common.load_series_arrays(series=series, config="from2023")
    actual = {
        int(row["time_series_id"]): arrays[row["site"]].power
        for row in series.iter_rows(named=True)
    }
    arms: dict[str, pl.DataFrame] = {}
    for name, (config, kind) in RAW_ARMS.items():
        arms[name] = _summaries(config=config, kind=kind, actual=actual)
    for name in CALIBRATED_ARMS:
        arm, config = name.split("_", 1)
        arms[name] = pl.read_parquet(calibrate.OUT_DIR / f"{arm}_{config}.parquet")
    for name in ("r1", "r2"):
        arms[name] = pl.read_parquet(references.OUT_DIR / f"{name}.parquet")
    return arms


def common_cases(*, arms: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return the cases every arm covers with an observed power before the holdout.

    Args:
        arms: `collect_arms()`.

    Returns:
        The distinct series, run, and valid-time keys.
    """
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    shared: pl.DataFrame | None = None
    for frame in arms.values():
        cases = (
            frame.filter(
                pl.all_horizontal(pl.col("actual", *common.LEVEL_COLUMNS).is_finite())
                & (pl.col("valid_time") < HOLDOUT)
            )
            .select(keys)
            .unique()
        )
        shared = cases if shared is None else shared.join(cases, on=keys, how="inner")
    assert shared is not None
    return shared


def prepare(*, arms: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Restrict every arm to the shared cases and add the lead, losses, and groups.

    Args:
        arms: `collect_arms()`.

    Returns:
        The same arms, each with `lead_hours`, `quantile_score`, `median_error`, and `group`.
    """
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    shared = common_cases(arms=arms)
    groups = common.series_table().select(
        "time_series_id", group=pl.col("substation_type").cast(pl.String)
    )
    out = {}
    for name, frame in arms.items():
        kept = frame.join(shared, on=keys, how="semi").sort(*keys, "seed")
        quantiles = kept.select(common.LEVEL_COLUMNS).to_numpy()
        actual = kept["actual"].to_numpy()
        score = common.pinball_quantile_score(quantiles=np.sort(quantiles, axis=1), actual=actual)
        out[name] = (
            kept.join(groups, on="time_series_id", how="left")
            .with_columns(
                lead_hours=(
                    pl.col("valid_time") - pl.col("power_fcst_init_time")
                ).dt.total_minutes()
                / 60.0,
                quantile_score=pl.Series(score * 100.0),
                median_error=pl.Series(np.abs(quantiles[:, 4] - actual) * 100.0),
            )
            .select(ARM_COLUMNS)
        )
    return out


def save_predictions(*, arms: dict[str, pl.DataFrame]) -> None:
    """Write each arm as an immutable prediction set, skipping sets that already exist.

    Args:
        arms: `prepare()`'s arms.
    """
    for name, frame in arms.items():
        try:
            experiment_store.write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=name,
                frame=frame.select(
                    "time_series_id",
                    "power_fcst_init_time",
                    "valid_time",
                    "seed",
                    *common.LEVEL_COLUMNS,
                ),
                kind="quantiles",
                row_set_id=common.ROW_SET_ID,
            )
        except FileExistsError:
            continue


def present_groups(*, arms: dict[str, pl.DataFrame]) -> tuple[str, ...]:
    """Return the substation types that have cases (all three in the full run)."""
    first = next(iter(arms.values()))
    return tuple(g for g in GROUPS if g in set(first["group"].unique().to_list()))


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
    """Return one paired contrast's estimate, interval, month agreement, and series count.

    Args:
        arms: `prepare()`'s arms.
        treatment: The arm whose score is compared.
        reference: The arm it is compared against.
        scope: One of `SCOPES`.
        group: One of `GROUPS`.
        level: The interval's coverage in percent.

    Returns:
        The mean paired difference in quantile score (and, for `median_difference`, in the error of
        the median), the interval bounds, how many validation months have a negative difference,
        and how many series have a negative difference.
    """
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
        if level == PLANNED_LEVEL and scope in DECISION_SCOPES and group == HEADLINE_GROUP
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


def absolute_table(*, arms: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return every arm's mean quantile score and median error by lead band, with 95% intervals.

    Args:
        arms: `prepare()`'s arms.

    Returns:
        One row per arm, scope, and metric.
    """
    rows = []
    for name, frame in arms.items():
        for group in present_groups(arms=arms):
            for scope in SCOPES:
                part = experiment_store.losses_for_bootstrap(
                    arm_losses={name: _scope(frame=frame, scope=scope, group=group)}
                )
                for metric in ("quantile_score", "median_error"):
                    interval = bootstrap.bootstrap_absolute(losses=part, arm=name, metric=metric)
                    rows.append(
                        {"arm": name, "group": group, "scope": scope, "metric": metric, **interval}
                    )
    return pl.DataFrame(rows)


def verdict(*, planned: pl.DataFrame) -> str:
    """Return the pre-registered verdict from the primary contrast C1 at leads up to 48 hours.

    Args:
        planned: The planned contrasts' table.

    Returns:
        `supported`, `refuted`, or `inconclusive`, with the numbers it rests on.
    """
    row = planned.filter(
        pl.col("treatment").eq("l2c_from2023")
        & pl.col("reference").eq("l3_from2023")
        & (pl.col("scope") == "up to 48 h")
        & (pl.col("group") == HEADLINE_GROUP)
    ).row(0, named=True)
    material_for = row["difference"] <= -0.15 and row["upper"] < 0 and row["months_negative"] >= 9
    material_against = row["difference"] >= 0.15 and row["lower"] > 0
    label = "supported" if material_for else "refuted" if material_against else "inconclusive"
    return (
        f"{label}: C1 up to 48 h is {row['difference']:+.3f} ({row['lower']:+.3f} to "
        f"{row['upper']:+.3f} at 99%), {row['months_negative']} of {row['months']} months negative"
    )


def main() -> None:
    """Score every arm and write the report tables."""
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arms = prepare(arms=collect_arms())
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
            for g in present_groups(arms=arms)
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
    contrasts = pl.concat([planned, exploratory], how="diagonal")
    contrasts.write_parquet(RESULTS_DIR / "contrasts.parquet")
    absolute = absolute_table(arms=arms)
    absolute.write_parquet(RESULTS_DIR / "absolute.parquet")
    lines = [
        "# Experiment 00005: scores",
        "",
        f"Verdict by the pre-registered rule: {verdict(planned=planned)}",
        "",
    ]
    lines += ["## Absolute scores (points of capacity; 95% month-and-seed intervals)", ""]
    lines += [
        "| arm | group | scope | quantile score | median error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, group, scope in itertools.product(arms, present_groups(arms=arms), SCOPES):
        q = absolute.filter(
            (pl.col("arm") == name)
            & (pl.col("group") == group)
            & (pl.col("scope") == scope)
            & (pl.col("metric") == "quantile_score")
        ).row(0, named=True)
        m = absolute.filter(
            (pl.col("arm") == name)
            & (pl.col("group") == group)
            & (pl.col("scope") == scope)
            & (pl.col("metric") == "median_error")
        ).row(0, named=True)
        lines.append(
            f"| {name} | {group} | {scope} | {q['value']:.2f} "
            f"({q['lower_95']:.2f} to {q['upper_95']:.2f})"
            f" | {m['value']:.2f} ({m['lower_95']:.2f} to {m['upper_95']:.2f}) |"
        )
    lines += ["", "## Planned contrasts (99% intervals; negative favours the first arm)", ""]
    lines += [
        "| contrast | scope | group | difference | interval | months negative | series negative |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in planned.iter_rows(named=True):
        lines.append(
            f"| {row['label']} | {row['scope']} | {row['group']} | {row['difference']:+.3f} | "
            f"{row['lower']:+.3f} to {row['upper']:+.3f} | "
            f"{row['months_negative']} of {row['months']} | "
            f"{row['series_negative']} of {row['series']} |"
        )
    lines += ["", "## Exploratory contrasts, Primary substations (95% intervals)", ""]
    lines += [
        "| contrast | scope | difference | interval | months negative |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in exploratory.iter_rows(named=True):
        lines.append(
            f"| {row['label']} | {row['scope']} | {row['difference']:+.3f} | "
            f"{row['lower']:+.3f} to {row['upper']:+.3f} | "
            f"{row['months_negative']} of {row['months']} |"
        )
    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
