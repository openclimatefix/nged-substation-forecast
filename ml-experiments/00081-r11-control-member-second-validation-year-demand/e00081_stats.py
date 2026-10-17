"""Score every arm of experiment 00081 and read the six planned contrasts with their intervals.

Parts, each writing under `work/results/<fold>/`:

- `metrics`: the per-case metrics of every arm table of the fold, to
  `work/metrics/<fold>/<arm>.parquet` (the quantile score `qs`, the absolute error of the median
  `mae`, the exceedance of the 0.95 quantile `ex95`, and the inner-90% coverage `cov90`).
- `contrasts`: the six planned contrasts C1 to C6 and the descriptive ones, on the cases that every
  arm of the fold shares, to `contrasts.parquet`. A contrast pairs the two arms' cases on series,
  run, and valid time. The interval resamples whole 4-week blocks of forecast start (each case
  counts in the block of its run), 50,000 times, at 99.17% (a Bonferroni family of six) and at 95%.
  A t-interval over the monthly mean differences (months of forecast start) sits beside every
  contrast. Reads by lead band, by series, and by month of forecast start are at 95% and
  descriptive.
- `verdict`: the decision rule's reading of fold `b` (rules R-a and R-b for C1 and C3, the
  correction-style rules for C4 and C5, the threshold for C6), to `verdict.json`.
- `absolute`: each arm's mean metrics by lead band and by series, with a 95% block interval on the
  quantile score, to `absolute.parquet` and `absolute_series.parquet`, labelled "control-only fold".
- `spread`: the spread factor re-chosen on the fold's first three months, and its effect later.
- `pooled`: the Christmas-column contrast pooled over the three winters (needs folds `b` and `c`).
- `nine_km`: fold `d`'s frozen model against `manual_heuristic`, by month and by weather product.

`--verify` checks the block resampler against `studies.bootstrap` on synthetic losses.

    uv run python ml-experiments/00081-*/e00081_stats.py [--fold b c d] [--parts ...] [--smoke]
        [--verify]
"""

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import e00081_common as common
import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from studies import bootstrap

N_RESAMPLES: Final[int] = 50_000
N_RESAMPLES_DESCRIPTIVE: Final[int] = 20_000
BLOCK_DAYS: Final[int] = 28
PLANNED_COUNT: Final[int] = 6
BONFERRONI_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / PLANNED_COUNT)
"""99.17%."""
NO_CHANGE_MARGIN: Final[float] = 0.03
"""C4 and C5: a component can go when the interval lies inside plus or minus this many points."""
BLEND_THRESHOLD: Final[float] = -0.05
"""C6's porting threshold, in points of quantile score."""
FACTORS: Final[np.ndarray] = np.round(np.arange(1.0, 1.3001, 0.025), 3)
"""The spread-factor grid of the descriptive check, 1.000 to 1.300."""
SPREAD_CHOICE_MONTHS: Final[int] = 3
SPREAD_ARMS: Final[tuple[str, ...]] = ("weekly", "monthly", "frozen_c")
"""Fold `b` arms without the spread factor; `frozen_c` has the correction and no spread factor."""
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
CHRISTMAS_WINDOWS: Final[dict[str, tuple[date, date]]] = {
    "b": (date(2023, 12, 18), date(2024, 1, 4)),
    "c": (date(2022, 12, 18), date(2023, 1, 4)),
}
"""The valid-time window around Christmas that the plan reads as a number without an interval."""
WITHOUT_SERIES_LABELS: Final[tuple[str, ...]] = ("D8", "D12", "D5")
"""The series that carried experiment 00048's gain, left out of one descriptive read of C1."""
EXPERIMENT_48: Final[dict[str, tuple[float, float, float, float]]] = {
    "C1": (-0.20, -0.40, -0.035, 99.44),
    "C3": (-0.13, -0.24, -0.03, 99.44),
}
"""Experiment 00048's estimate, interval bounds, and interval level (percent) for C1 and C3."""
EXPERIMENT_49_C4: Final[tuple[float, float, float, float]] = (-0.017, -0.058, 0.007, 99.17)
"""Experiment 00049's C1 (the Christmas column dropped): estimate, bounds, and level."""
NINE_KM_CHANGE: Final[date] = date(2023, 6, 27)
"""The first run date that ECMWF's ensemble serves at 9 km native resolution."""
PUBLICATION_DELAY: Final[timedelta] = common.PUBLICATION_DELAY
MEAN_METRICS: Final[tuple[str, ...]] = ("qs", "mae", "ex95", "cov90")
SCOPES: Final[tuple[tuple[str, float, float], ...]] = (
    ("all leads", 0.0, 10_000.0),
    *common.LEAD_BANDS,
)

PLANNED: Final[dict[str, tuple[str, str | None]]] = {
    "C1": ("weekly_s", "frozen_cs"),
    "C2": ("monthly_s", "frozen_cs"),
    "C3": ("weekly_s", "monthly_s"),
    "C4": ("roll12_no_christmas_s", "roll12_s"),
    "C5": ("roll12_hl2y_s", "roll12_s"),
    "C6": ("blend", None),
}
"""Each planned contrast's treatment and reference arm. C6's reference is its best member."""
BLEND_MEMBERS: Final[tuple[str, ...]] = ("weekly_s", "monthly_s", "roll12_s", "frozen_cs")
DESCRIPTIVE: Final[dict[str, tuple[str, str]]] = {
    "X_frozen_correction": ("frozen_cs", "frozen_s"),
    "X_weekly_vs_frozen_s": ("weekly_s", "frozen_s"),
    "X_roll12_vs_monthly": ("roll12_s", "monthly_s"),
    "X_roll12_vs_stack": ("roll12_s", "frozen_cs"),
    "X_weekly_vs_manual": ("weekly_s", "manual_heuristic"),
    "X_stack_vs_manual": ("frozen_cs", "manual_heuristic"),
    "X_blend_vs_stack": ("blend", "frozen_cs"),
}


def _arms_dir(*, fold: str, smoke: bool) -> Path:
    base = common.ARMS_DIR.with_name("arms_smoke") if smoke else common.ARMS_DIR
    return base / fold


def _metrics_dir(*, fold: str, smoke: bool) -> Path:
    base = common.METRICS_DIR.with_name("metrics_smoke") if smoke else common.METRICS_DIR
    return base / fold


def _results_dir(*, fold: str, smoke: bool) -> Path:
    base = common.RESULTS_DIR.with_name("results_smoke") if smoke else common.RESULTS_DIR
    return base / fold


def block_totals(*, values: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each block's sum of `values` and row count, blocks in sorted order."""
    _unique, inverse = np.unique(blocks, return_inverse=True)
    return np.bincount(inverse, weights=values.astype(np.float64)), np.bincount(inverse)


def block_bounds(
    *, sums: np.ndarray, counts: np.ndarray, level: float, n_resamples: int
) -> tuple[float, float]:
    """Return the interval of the mean of rows, resampling whole blocks.

    The draws are those of `studies.bootstrap`'s resampler: for each resample one seed draw, then
    one draw of the blocks with replacement. The resampled mean of the drawn blocks' rows is the sum
    of their sums over the sum of their counts, so no row is touched after the totals.

    Args:
        sums: Each block's sum of the values, from `block_totals`.
        counts: Each block's row count.
        level: The interval's coverage in percent.
        n_resamples: The number of resamples.

    Returns:
        The lower and upper bounds.
    """
    generator = np.random.default_rng(bootstrap.BOOTSTRAP_SEED)
    resampled = np.empty(n_resamples)
    for draw in range(n_resamples):
        generator.integers(0, 1)
        drawn = generator.integers(0, len(sums), size=len(sums))
        resampled[draw] = sums[drawn].sum() / counts[drawn].sum()
    tail = (100.0 - level) / 2.0
    return float(np.percentile(resampled, tail)), float(np.percentile(resampled, 100.0 - tail))


def verify_block_bounds() -> list[str]:
    """Compare `block_bounds` with `studies.bootstrap` on synthetic paired losses.

    Returns:
        Report lines with both intervals at two levels; they agree to float precision.
    """
    rng = np.random.default_rng(1)
    blocks = np.array([f"b{b:02d}" for b in rng.integers(0, 10, size=6000)])
    base = rng.normal(5.0, 1.0, size=6000)
    treatment = base - 0.1 + rng.normal(0.0, 0.3, size=6000)
    frame = pl.DataFrame(
        {
            "site": ["s"] * 6000,
            "time": [f"t{i:06d}" for i in range(6000)],
            "seed": [0] * 6000,
            "month": blocks,
            "treatment": treatment,
            "reference": base,
        }
    )
    losses = pl.concat(
        [
            frame.select("site", "time", "seed", "month", qs=pl.col(arm)).with_columns(
                arm=pl.lit(arm)
            )
            for arm in ("treatment", "reference")
        ]
    )
    sums, counts = block_totals(values=treatment - base, blocks=blocks)
    lines = []
    for level in (95.0, BONFERRONI_LEVEL):
        library = bootstrap.bootstrap_difference_at_level(
            losses=losses,
            treatment="treatment",
            reference="reference",
            metric="qs",
            level=level,
            n_resamples=2000,
        )
        fast = block_bounds(sums=sums, counts=counts, level=level, n_resamples=2000)
        lines.append(
            f"level {level:.2f}: library ({library[0]:.6f}, {library[1]:.6f}), "
            f"block totals ({fast[0]:.6f}, {fast[1]:.6f})"
        )
        if not np.allclose(library, fast, atol=1e-9):
            msg = f"the block resampler differs from studies.bootstrap at level {level}"
            raise ValueError(msg)
    return lines


def compute_metrics(*, fold: str, smoke: bool) -> None:
    """Write every arm's per-case metrics for one fold."""
    folder = _metrics_dir(fold=fold, smoke=smoke)
    folder.mkdir(parents=True, exist_ok=True)
    finite = pl.all_horizontal(pl.col(*common.LEVEL_COLUMNS, "actual").is_finite())
    for path in sorted(_arms_dir(fold=fold, smoke=smoke).glob("*.parquet")):
        frame = (
            pl.read_parquet(path)
            .filter(finite & (pl.col("valid_time") < HOLDOUT))
            .sort(*common.KEYS, "seed")
        )
        quantiles = np.sort(
            frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1
        )
        metrics = common.case_metrics(
            quantiles=quantiles, actual=frame["actual"].to_numpy().astype(np.float64)
        )
        table = frame.select(
            *common.KEYS,
            "seed",
            "site",
            lead_hours=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
            ),
        ).with_columns(
            *[pl.Series(label, values.astype(np.float32)) for label, values in metrics.items()]
        )
        common.atomic_write(frame=table, target=folder / path.name)
        print(f"{fold} {path.stem}: {table.height:,} cases", flush=True)


def read_arm(*, fold: str, arm: str, smoke: bool) -> pl.DataFrame | None:
    """Return one arm's per-case metrics, or `None` if absent."""
    path = _metrics_dir(fold=fold, smoke=smoke) / f"{arm}.parquet"
    return pl.read_parquet(path) if path.exists() else None


def shared_keys(*, fold: str, smoke: bool) -> pl.DataFrame:
    """Return the cases that every arm table of the fold holds.

    Raises:
        FileNotFoundError: If the fold has no metrics.
    """
    keys: pl.DataFrame | None = None
    for path in sorted(_metrics_dir(fold=fold, smoke=smoke).glob("*.parquet")):
        these = pl.read_parquet(path, columns=list(common.KEYS))
        keys = these if keys is None else keys.join(these, on=list(common.KEYS), how="inner")
    if keys is None:
        msg = f"fold {fold} has no metrics"
        raise FileNotFoundError(msg)
    return keys


def _with_blocks(*, frame: pl.DataFrame, fold: common.Fold) -> pl.DataFrame:
    """Add `run_date`, `block` (4-week block of forecast start), `month`, and `week` columns."""
    first = fold.first_run
    run_date = (pl.col("power_fcst_init_time") - PUBLICATION_DELAY).dt.date()
    days = (run_date - pl.lit(first)).dt.total_days()
    return frame.with_columns(
        run_date=run_date,
        block=days // BLOCK_DAYS,
        week=days // 7,
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
    )


def paired_cases(
    *, treatment: pl.DataFrame, reference: pl.DataFrame, keys: pl.DataFrame
) -> pl.DataFrame:
    """Return the shared cases with `qs`, `mae` treatment-minus-reference differences."""
    metrics = ["qs", "mae", "ex95"]
    ref = keys.join(reference, on=list(common.KEYS), how="inner", validate="1:1").select(
        *common.KEYS,
        "site",
        "lead_hours",
        *[pl.col(m).alias(f"{m}_ref") for m in metrics],
    )
    trt = treatment.select(*common.KEYS, *[pl.col(m).alias(f"{m}_trt") for m in metrics])
    return ref.join(trt, on=list(common.KEYS), how="inner", validate="1:1").with_columns(
        **{
            m: pl.col(f"{m}_trt").cast(pl.Float64) - pl.col(f"{m}_ref").cast(pl.Float64)
            for m in metrics
        }
    )


def _interval(
    *, values: np.ndarray, blocks: np.ndarray, level: float, n_resamples: int
) -> tuple[float, float]:
    sums, counts = block_totals(values=values, blocks=blocks)
    return block_bounds(sums=sums, counts=counts, level=level, n_resamples=n_resamples)


def monthly_t_interval(*, part: pl.DataFrame) -> tuple[float, float, int]:
    """Return a 95% t-interval over the monthly mean differences and the month count.

    The interval is NaN when fewer than two months of forecast start are present.
    """
    by_month = part.group_by("month").agg(pl.col("qs").mean()).sort("month")["qs"].to_list()
    if len(by_month) < 2:
        return float("nan"), float("nan"), len(by_month)
    lower, upper = bootstrap.fold_t_interval(fold_differences=by_month)
    return lower, upper, len(by_month)


def contrast_rows(
    *,
    contrast_id: str,
    treatment_arm: str,
    reference_arm: str,
    family: str,
    fold: common.Fold,
    smoke: bool,
    keys: pl.DataFrame,
    exclude_sites: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    """Return one contrast's rows: pooled at two levels, then by lead band, series, and month.

    Args:
        contrast_id: The contrast's label.
        treatment_arm: The arm whose score is being compared.
        reference_arm: The arm it is compared against.
        family: `planned` or `descriptive`.
        fold: The fold.
        smoke: Whether to read the smoke tables and use few resamples.
        keys: The shared cases.
        exclude_sites: Series labels left out of the read.

    Returns:
        One row per scope, metric, and level, with the difference, its block interval, the
        t-interval over months, and the case, block, and month counts.
    """
    treatment = read_arm(fold=fold.name, arm=treatment_arm, smoke=smoke)
    reference = read_arm(fold=fold.name, arm=reference_arm, smoke=smoke)
    if treatment is None or reference is None:
        return []
    paired = _with_blocks(
        frame=paired_cases(treatment=treatment, reference=reference, keys=keys), fold=fold
    )
    if exclude_sites:
        paired = paired.filter(~pl.col("site").is_in(exclude_sites))
    draws = 200 if smoke else N_RESAMPLES
    draws_small = 200 if smoke else N_RESAMPLES_DESCRIPTIVE
    rows: list[dict[str, object]] = []

    def add(
        *,
        part: pl.DataFrame,
        site: str,
        scope: str,
        metric: str,
        level_label: str,
        level: float,
        n_resamples: int,
        grouping: str,
    ) -> None:
        values = part[metric].to_numpy()
        lower, upper = _interval(
            values=values, blocks=part[grouping].to_numpy(), level=level, n_resamples=n_resamples
        )
        t_lower, t_upper, months = (
            monthly_t_interval(part=part) if metric == "qs" else (float("nan"), float("nan"), 0)
        )
        rows.append(
            {
                "id": contrast_id,
                "family": family,
                "treatment": treatment_arm,
                "reference": reference_arm,
                "site": site,
                "scope": scope,
                "metric": metric,
                "level_label": level_label,
                "level": level,
                "difference": float(values.mean()),
                "lower": lower,
                "upper": upper,
                "t_lower": t_lower,
                "t_upper": t_upper,
                "cases": int(values.size),
                "blocks": int(part[grouping].n_unique()),
                "months": months,
            }
        )

    for metric in ("qs", "mae"):
        add(
            part=paired,
            site="pooled",
            scope="all leads",
            metric=metric,
            level_label="95",
            level=95.0,
            n_resamples=draws_small,
            grouping="block",
        )
    add(
        part=paired,
        site="pooled",
        scope="all leads",
        metric="qs",
        level_label="planned",
        level=BONFERRONI_LEVEL,
        n_resamples=draws,
        grouping="block",
    )
    for scope, low, high in SCOPES[1:]:
        part = paired.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
        if not part.is_empty():
            add(
                part=part,
                site="pooled",
                scope=scope,
                metric="qs",
                level_label="95",
                level=95.0,
                n_resamples=draws_small,
                grouping="block",
            )
    for site in sorted(paired["site"].unique().to_list(), key=lambda s: int(s[1:])):
        add(
            part=paired.filter(pl.col("site") == site),
            site=site,
            scope="all leads",
            metric="qs",
            level_label="95",
            level=95.0,
            n_resamples=draws_small,
            grouping="block",
        )
    for month in sorted(paired["month"].unique().to_list()):
        part = paired.filter(pl.col("month") == month)
        add(
            part=part,
            site="pooled",
            scope=f"month {month}",
            metric="qs",
            level_label="95",
            level=95.0,
            n_resamples=draws_small,
            grouping="week",
        )
    return rows


def best_member(*, fold: str, keys: pl.DataFrame, smoke: bool) -> str:
    """Return the blend member with the lowest mean quantile score on the shared cases."""
    scores = {}
    for arm in BLEND_MEMBERS:
        table = read_arm(fold=fold, arm=arm, smoke=smoke)
        if table is not None:
            scores[arm] = float(keys.join(table, on=list(common.KEYS), how="inner")["qs"].mean())  # ty: ignore[invalid-argument-type]
    return min(scores, key=lambda arm: scores[arm])


def compute_contrasts(*, fold_name: str, smoke: bool) -> None:
    """Write the planned and descriptive contrasts of one fold."""
    fold = common.FOLDS[fold_name]  # ty: ignore[invalid-argument-type]
    keys = shared_keys(fold=fold_name, smoke=smoke)
    rows: list[dict[str, object]] = []
    for contrast_id, (treatment, reference) in PLANNED.items():
        resolved = reference or best_member(fold=fold_name, keys=keys, smoke=smoke)
        found = contrast_rows(
            contrast_id=contrast_id,
            treatment_arm=treatment,
            reference_arm=resolved,
            family="planned",
            fold=fold,
            smoke=smoke,
            keys=keys,
        )
        rows += found
        print(
            f"{fold_name} {contrast_id} ({treatment} minus {resolved}): {len(found)} rows",
            flush=True,
        )
    for contrast_id, (treatment, reference) in DESCRIPTIVE.items():
        rows += contrast_rows(
            contrast_id=contrast_id,
            treatment_arm=treatment,
            reference_arm=reference,
            family="descriptive",
            fold=fold,
            smoke=smoke,
            keys=keys,
        )
    rows += [
        {**row, "id": "X_C1_without_D8_D12_D5"}
        for row in contrast_rows(
            contrast_id="C1",
            treatment_arm="weekly_s",
            reference_arm="frozen_cs",
            family="descriptive",
            fold=fold,
            smoke=smoke,
            keys=keys,
            exclude_sites=WITHOUT_SERIES_LABELS,
        )
        if row["site"] == "pooled" and row["scope"] == "all leads"
    ]
    folder = _results_dir(fold=fold_name, smoke=smoke)
    folder.mkdir(parents=True, exist_ok=True)
    common.atomic_write(
        frame=pl.DataFrame(rows, infer_schema_length=None), target=folder / "contrasts.parquet"
    )


def z_for_level(*, level: float) -> float:
    """Return the normal quantile that bounds a two-sided interval of `level` percent."""
    return float(scipy_stats.norm.ppf(1.0 - (1.0 - level / 100.0) / 2.0))


def precision_weighted(*, estimates: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Return the precision-weighted mean and its 95% interval.

    Args:
        estimates: Each estimate and its standard error.

    Returns:
        The weighted mean and the lower and upper ends of its 95% interval.
    """
    # A zero standard error needs a single block, as in the three-run smoke test.
    weights = np.array([1.0 / max(se, 1e-9) ** 2 for _, se in estimates])
    mean = float(sum(w * e for w, (e, _) in zip(weights, estimates, strict=True)) / weights.sum())
    half = 1.96 / float(np.sqrt(weights.sum()))
    return mean, mean - half, mean + half


def _row(*, contrasts: pl.DataFrame, contrast_id: str, level_label: str) -> dict[str, float]:
    return contrasts.filter(
        (pl.col("id") == contrast_id)
        & (pl.col("family") == "planned")
        & (pl.col("site") == "pooled")
        & (pl.col("scope") == "all leads")
        & (pl.col("metric") == "qs")
        & (pl.col("level_label") == level_label)
    ).row(0, named=True)


def decide(*, contrasts: pl.DataFrame) -> dict[str, object]:
    """Return the decision rule's reading of fold `b`'s planned contrasts.

    Args:
        contrasts: Fold `b`'s `contrasts.parquet`.

    Returns:
        The reading of each contrast and the overall verdict, for `verdict.json`.
    """
    out: dict[str, object] = {}
    flags: dict[str, dict[str, object]] = {}
    for contrast_id in ("C1", "C3"):
        wide = _row(contrasts=contrasts, contrast_id=contrast_id, level_label="planned")
        narrow = _row(contrasts=contrasts, contrast_id=contrast_id, level_label="95")
        estimate_48, lower_48, upper_48, level_48 = EXPERIMENT_48[contrast_id]
        se_48 = (upper_48 - lower_48) / 2.0 / z_for_level(level=level_48)
        se_b = (narrow["upper"] - narrow["lower"]) / 2.0 / 1.96
        pooled, pooled_lower, pooled_upper = precision_weighted(
            estimates=[(estimate_48, se_48), (narrow["difference"], se_b)]
        )
        r_a = wide["difference"] < 0 and wide["upper"] < 0
        r_b = narrow["difference"] < 0 and narrow["upper"] < 0 and pooled_upper < 0
        above = wide["lower"] > 0
        t_95 = _row(contrasts=contrasts, contrast_id=contrast_id, level_label="95")
        t_includes_zero = (t_95["t_lower"] <= 0 <= t_95["t_upper"]) if t_95["months"] >= 2 else None
        flags[contrast_id] = {
            "estimate": wide["difference"],
            "interval_99_17": [wide["lower"], wide["upper"]],
            "interval_95": [narrow["lower"], narrow["upper"]],
            "t_interval_95": [t_95["t_lower"], t_95["t_upper"]],
            "t_interval_includes_zero": t_includes_zero,
            "R_a": r_a,
            "R_b": r_b,
            "wholly_above_zero": above,
            "pooled_with_experiment_48": [pooled, pooled_lower, pooled_upper],
        }
    out["C1"], out["C3"] = flags["C1"], flags["C3"]
    c1, c3 = flags["C1"], flags["C3"]
    if c1["R_a"] and c3["R_a"]:
        verdict = "supported"
    elif (c1["R_a"] or c3["R_a"]) or (c1["R_b"] and c3["R_b"]):
        verdict = "partly supported"
    elif c1["wholly_above_zero"] or c3["wholly_above_zero"]:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    out["rule_conflict"] = bool(
        verdict == "partly supported" and (c1["wholly_above_zero"] or c3["wholly_above_zero"])
    )
    fragile = [
        cid
        for cid in ("C1", "C3")
        if flags[cid]["t_interval_includes_zero"] and (flags[cid]["R_a"] or flags[cid]["R_b"])
    ]
    out["verdict"] = verdict
    out["fragile"] = fragile
    for contrast_id in ("C4", "C5"):
        wide = _row(contrasts=contrasts, contrast_id=contrast_id, level_label="planned")
        inside = wide["lower"] >= -NO_CHANGE_MARGIN and wide["upper"] <= NO_CHANGE_MARGIN
        reading = (
            "component stays (removing it hurts)"
            if wide["lower"] > 0
            else "component can go (interval inside plus or minus 0.03)"
            if inside
            else "unresolved"
        )
        out[contrast_id] = {
            "estimate": wide["difference"],
            "interval_99_17": [wide["lower"], wide["upper"]],
            "reading": reading,
        }
    blend = _row(contrasts=contrasts, contrast_id="C6", level_label="planned")
    c6 = (
        ("port" if blend["difference"] <= BLEND_THRESHOLD else "small gain")
        if blend["upper"] < 0
        else "harm"
        if blend["lower"] > 0
        else "null"
    )
    out["C6"] = {
        "estimate": blend["difference"],
        "interval_99_17": [blend["lower"], blend["upper"]],
        "reference": blend["reference"],
        "reading": c6,
    }
    return out


def compute_verdict(*, fold_name: str, smoke: bool) -> None:
    """Write fold `b`'s verdict."""
    folder = _results_dir(fold=fold_name, smoke=smoke)
    verdict = decide(contrasts=pl.read_parquet(folder / "contrasts.parquet"))
    (folder / "verdict.json").write_text(json.dumps(verdict, indent=2, default=float) + "\n")
    print(json.dumps(verdict, indent=2, default=float))


def compute_absolute(*, fold_name: str, smoke: bool) -> None:
    """Write each arm's mean metrics by lead band and series, on the cases every arm shares."""
    fold = common.FOLDS[fold_name]  # ty: ignore[invalid-argument-type]
    keys = shared_keys(fold=fold_name, smoke=smoke)
    draws = 200 if smoke else N_RESAMPLES_DESCRIPTIVE
    by_scope: list[dict[str, object]] = []
    by_series: list[dict[str, object]] = []
    for path in sorted(_metrics_dir(fold=fold_name, smoke=smoke).glob("*.parquet")):
        frame = _with_blocks(
            frame=keys.join(
                pl.read_parquet(path), on=list(common.KEYS), how="inner", validate="1:1"
            ),
            fold=fold,
        )
        for scope, low, high in SCOPES:
            part = frame.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
            if part.is_empty():
                continue
            row: dict[str, object] = {
                "fold": fold_name,
                "arm": path.stem,
                "scope": scope,
                "cases": part.height,
                **{m: float(part[m].mean()) for m in MEAN_METRICS},  # ty: ignore[invalid-argument-type]
            }
            if scope == "all leads":
                row["qs_lower"], row["qs_upper"] = _interval(
                    values=part["qs"].to_numpy(),
                    blocks=part["block"].to_numpy(),
                    level=95.0,
                    n_resamples=draws,
                )
            by_scope.append(row)
        for (site,), site_part in frame.group_by("site"):
            by_series.append(
                {
                    "fold": fold_name,
                    "arm": path.stem,
                    "site": site,
                    "cases": site_part.height,
                    "qs": float(site_part["qs"].mean()),  # ty: ignore[invalid-argument-type]
                }
            )
    folder = _results_dir(fold=fold_name, smoke=smoke)
    common.atomic_write(
        frame=pl.DataFrame(by_scope, infer_schema_length=None), target=folder / "absolute.parquet"
    )
    common.atomic_write(
        frame=pl.DataFrame(by_series, infer_schema_length=None),
        target=folder / "absolute_series.parquet",
    )


def compute_spread(*, fold_name: str, smoke: bool) -> None:
    """Re-choose the spread factor on the fold's first three months and score the later months.

    For each arm in `SPREAD_ARMS` (quantiles without the spread factor), the factor that gives the
    lowest mean quantile score on the first three months of forecast start is chosen, and the
    score of the later months at that factor is compared with the score at 1.15.
    """
    fold = common.FOLDS[fold_name]  # ty: ignore[invalid-argument-type]
    keys = shared_keys(fold=fold_name, smoke=smoke)
    rows = []
    for arm in SPREAD_ARMS:
        table = read_arm(fold=fold_name, arm=arm, smoke=smoke)
        raw_path = _arms_dir(fold=fold_name, smoke=smoke) / f"{arm}.parquet"
        if table is None or not raw_path.exists():
            continue
        frame = _with_blocks(
            frame=keys.join(
                pl.read_parquet(raw_path), on=list(common.KEYS), how="inner", validate="1:1"
            ),
            fold=fold,
        ).filter(pl.col("actual").is_finite())
        quantiles = np.sort(
            frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1
        )
        actual = frame["actual"].to_numpy().astype(np.float64)
        median = quantiles[:, [common.MEDIAN]]
        curve = np.stack(
            [
                100.0
                * common.pinball_quantile_score(
                    quantiles=np.sort(median + f * (quantiles - median), axis=1), actual=actual
                )
                for f in FACTORS
            ],
            axis=1,
        )
        months = sorted(frame["month"].unique().to_list())
        early = np.isin(frame["month"].to_numpy(), months[:SPREAD_CHOICE_MONTHS])
        chosen = float(FACTORS[int(np.argmin(curve[early].mean(axis=0)))])
        fixed_column = int(np.argmin(np.abs(FACTORS - common.SPREAD_FACTOR)))
        chosen_column = int(np.argmin(np.abs(FACTORS - chosen)))
        later = ~early
        difference = curve[later, chosen_column] - curve[later, fixed_column]
        lower, upper = (
            _interval(
                values=difference,
                blocks=frame["block"].to_numpy()[later],
                level=95.0,
                n_resamples=200 if smoke else N_RESAMPLES_DESCRIPTIVE,
            )
            if later.any()
            else (float("nan"), float("nan"))
        )
        rows.append(
            {
                "arm": arm,
                "chosen_factor": chosen,
                "fixed_factor": common.SPREAD_FACTOR,
                "months_chosen_on": ", ".join(months[:SPREAD_CHOICE_MONTHS]),
                "later_cases": int(later.sum()),
                "later_qs_fixed": float(curve[later, fixed_column].mean())
                if later.any()
                else float("nan"),
                "later_qs_chosen": float(curve[later, chosen_column].mean())
                if later.any()
                else float("nan"),
                "chosen_minus_fixed": float(difference.mean()) if later.any() else float("nan"),
                "lower": lower,
                "upper": upper,
            }
        )
    common.atomic_write(
        frame=pl.DataFrame(rows, infer_schema_length=None),
        target=_results_dir(fold=fold_name, smoke=smoke) / "spread.parquet",
    )


def compute_christmas_window(*, fold_name: str, smoke: bool) -> None:
    """Write C4's mean difference over the Christmas window as a number without an interval."""
    if fold_name not in CHRISTMAS_WINDOWS:
        return
    fold = common.FOLDS[fold_name]  # ty: ignore[invalid-argument-type]
    keys = shared_keys(fold=fold_name, smoke=smoke)
    treatment = read_arm(fold=fold_name, arm="roll12_no_christmas_s", smoke=smoke)
    reference = read_arm(fold=fold_name, arm="roll12_s", smoke=smoke)
    if treatment is None or reference is None:
        return
    paired = paired_cases(treatment=treatment, reference=reference, keys=keys)
    first, last = CHRISTMAS_WINDOWS[fold_name]
    window = paired.filter(
        (pl.col("valid_time").dt.date() >= first) & (pl.col("valid_time").dt.date() <= last)
    )
    rest = paired.filter(~pl.col("valid_time").dt.date().is_between(first, last))
    row = {
        "fold": fold.name,
        "window_start": first,
        "window_end": last,
        "cases_in_window": window.height,
        "difference_in_window": float(window["qs"].mean()) if window.height else float("nan"),  # ty: ignore[invalid-argument-type]
        "cases_outside": rest.height,
        "difference_outside": float(rest["qs"].mean()) if rest.height else float("nan"),  # ty: ignore[invalid-argument-type]
    }
    common.atomic_write(
        frame=pl.DataFrame([row]),
        target=_results_dir(fold=fold_name, smoke=smoke) / "christmas_window.parquet",
    )


def compute_pooled(*, smoke: bool) -> None:
    """Pool C4 over the winters of folds `c` and `b` and experiment 00049's, descriptively."""
    estimates, labels = [], []
    for fold_name, label in (("c", "2022-23 (fold C)"), ("b", "2023-24 (fold B)")):
        path = _results_dir(fold=fold_name, smoke=smoke) / "contrasts.parquet"
        if not path.exists():
            return
        row = _row(contrasts=pl.read_parquet(path), contrast_id="C4", level_label="95")
        estimates.append((row["difference"], (row["upper"] - row["lower"]) / 2.0 / 1.96))
        labels.append(label)
    estimate, lower, upper, level = EXPERIMENT_49_C4
    estimates.append((estimate, (upper - lower) / 2.0 / z_for_level(level=level)))
    labels.append("2025-26 (experiment 49)")
    mean, low, high = precision_weighted(estimates=estimates)
    table = pl.DataFrame(
        {
            "winter": [*labels, "pooled"],
            "difference": [e for e, _ in estimates] + [mean],
            "standard_error": [s for _, s in estimates] + [(high - low) / 2.0 / 1.96],
            "lower_95": [e - 1.96 * s for e, s in estimates] + [low],
            "upper_95": [e + 1.96 * s for e, s in estimates] + [high],
        }
    )
    folder = _results_dir(fold="pooled", smoke=smoke)
    folder.mkdir(parents=True, exist_ok=True)
    common.atomic_write(frame=table, target=folder / "pooled_c4.parquet")


def compute_nine_km(*, smoke: bool) -> None:
    """Score fold `d`'s frozen model against `manual_heuristic`, by month and by weather product.

    The frozen model is fitted to 2023-03-31. The score relative to `manual_heuristic` is the
    quantile score of the frozen model over that of `manual_heuristic`, minus one, in percent.
    """
    fold = common.FOLDS["d"]
    keys = shared_keys(fold="d", smoke=smoke)
    frozen = read_arm(fold="d", arm="frozen_s", smoke=smoke)
    manual = read_arm(fold="d", arm="manual_heuristic", smoke=smoke)
    if frozen is None or manual is None:
        return
    paired = _with_blocks(
        frame=paired_cases(treatment=frozen, reference=manual, keys=keys), fold=fold
    ).with_columns(
        product=pl.when(pl.col("run_date") >= pl.lit(NINE_KM_CHANGE))
        .then(pl.lit("9 km (runs from 2023-06-27)"))
        .otherwise(pl.lit("18 km (runs before 2023-06-27)"))
    )
    rows = []
    for grouping, label in (("month", "month of forecast start"), ("product", "weather product")):
        for (name,), part in paired.group_by(grouping, maintain_order=True):
            frozen_score = float(part["qs_trt"].mean())  # ty: ignore[invalid-argument-type]
            manual_score = float(part["qs_ref"].mean())  # ty: ignore[invalid-argument-type]
            lower, upper = _interval(
                values=part["qs"].to_numpy(),
                blocks=part["week"].to_numpy(),
                level=95.0,
                n_resamples=200 if smoke else N_RESAMPLES_DESCRIPTIVE,
            )
            rows.append(
                {
                    "grouping": label,
                    "group": name,
                    "cases": part.height,
                    "frozen_qs": frozen_score,
                    "manual_heuristic_qs": manual_score,
                    "difference": frozen_score - manual_score,
                    "relative_percent": 100.0 * (frozen_score / manual_score - 1.0),
                    "lower_95_weekly_blocks": lower,
                    "upper_95_weekly_blocks": upper,
                }
            )
    folder = _results_dir(fold="d", smoke=smoke)
    folder.mkdir(parents=True, exist_ok=True)
    common.atomic_write(
        frame=pl.DataFrame(rows, infer_schema_length=None), target=folder / "nine_km.parquet"
    )


def main() -> None:
    """Run the requested parts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", nargs="+", choices=sorted(common.FOLDS), default=["b", "c", "d"])
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("metrics", "contrasts", "verdict", "absolute", "spread", "pooled", "nine_km"),
        default=["metrics", "contrasts", "verdict", "absolute", "spread", "pooled", "nine_km"],
    )
    parser.add_argument("--smoke", action="store_true", help="read the smoke arms, few resamples")
    parser.add_argument(
        "--verify", action="store_true", help="compare the resampler with the library"
    )
    args = parser.parse_args()
    if args.verify:
        print("\n".join(verify_block_bounds()))
        return
    folds = ["b"] if args.smoke else args.fold
    for fold_name in folds:
        _results_dir(fold=fold_name, smoke=args.smoke).mkdir(parents=True, exist_ok=True)
        if "metrics" in args.parts:
            compute_metrics(fold=fold_name, smoke=args.smoke)
        if fold_name in ("b", "c"):
            if "contrasts" in args.parts:
                compute_contrasts(fold_name=fold_name, smoke=args.smoke)
                compute_christmas_window(fold_name=fold_name, smoke=args.smoke)
            if "verdict" in args.parts and fold_name == "b":
                compute_verdict(fold_name=fold_name, smoke=args.smoke)
        if "absolute" in args.parts:
            compute_absolute(fold_name=fold_name, smoke=args.smoke)
        if "spread" in args.parts and fold_name == "b":
            compute_spread(fold_name=fold_name, smoke=args.smoke)
        if "nine_km" in args.parts and fold_name == "d":
            compute_nine_km(smoke=args.smoke)
    if "pooled" in args.parts and not args.smoke and {"b", "c"} <= set(args.fold):
        compute_pooled(smoke=args.smoke)


if __name__ == "__main__":
    main()
