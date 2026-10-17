"""Score every arm of experiment 00080 and read the nine planned contrasts with their intervals.

Four parts, each writing under `work/results/`:

- `metrics`: the per-case metrics of every arm table in `work/arms/`, to
  `work/metrics/<arm>.parquet`
  (the quantile score `qs`, the absolute error of the median `mae`, the threshold-weighted CRPS
  `tw_p99`, the exceedance of the 0.95 quantile `ex95`, and the inner-90% coverage `cov90`).
- `contrasts`: the nine planned contrasts C1 to C8w and the descriptive ones, each on the 292 runs
  of `fold_a_daily_gap_v1`, to `contrasts.parquet`. A contrast keeps the cases its two arms share,
  pairs them on series, run, and valid time, and resamples whole origin months (each run counts in
  its origin month). The planned contrasts are read at 99.44% (Bonferroni across nine), every
  contrast also at 95%. Every planned contrast is also read on each series alone, descriptively.
- `absolute`: each arm's mean metrics by group and lead band on the cases that every arm of the
  group shares, with a 95% month-resampled interval on the quantile score, to `absolute.parquet`;
  and each arm's quantile score by series, to `absolute_series.parquet`.
- `reproduction`: the 73 runs of `fold_a_every5d_v1` scored again from the same refits. An arm
  without a correction is compared with experiment 00034's stored seed-0 arm of the same
  configuration, on the cases both hold, to `reproduction.parquet`; the tolerance is 0.001 points.
- `predictions`: every arm's rows on the 292 runs, saved with `experiment_store.write_predictions`
  under the arm's name. An arm already saved is skipped.

The interval resamples whole months with the random stream of `studies.bootstrap`, but sums each
month once instead of re-drawing its rows, so a contrast over millions of rows costs seconds per
100,000 resamples. `--verify` checks the two against each other on synthetic losses.

    uv run python ml-experiments/00080-*/e00080_stats.py [--parts ...] [--smoke] [--verify]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00080_common as common
import e00080_gen_common as gen
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

N_RESAMPLES: Final[int] = 200_000
"""Enough that each tail of the 99.44% interval holds about 300 resampled means."""
N_RESAMPLES_95: Final[int] = 20_000
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 9)
NO_CHANGE_MARGIN: Final[float] = 0.05
"""C8s and C8w read as no change when the interval lies inside plus or minus this many points."""
TOLERANCE: Final[float] = 0.001
MEAN_METRICS: Final[tuple[str, ...]] = ("qs", "mae", "tw_p99", "ex95", "cov90")
SCOPES: Final[tuple[tuple[str, float, float], ...]] = (
    ("all leads", 0.0, 10_000.0),
    *common.LEAD_BANDS,
)
GAP_ORIGINS: Final[list] = common.origins_of_subset(subset="gap")
EVERY5D_ORIGINS: Final[list] = common.origins_of_subset(subset="every5d")


@dataclass(frozen=True)
class ContrastSpec:
    """One contrast: the arm tables and the groups whose rows are pooled.

    Attributes:
        groups: The groups (`Primary`, `BSP`, `GSP`, `solar`, `wind`) whose series are pooled.
        treatment: The arm whose metric is being compared.
        reference: The arm it is compared against.
    """

    groups: tuple[str, ...]
    treatment: str
    reference: str


PLANNED: Final[dict[str, ContrastSpec]] = {
    "C1": ContrastSpec(("Primary",), "weekly_s", "frozen_cs"),
    "C2": ContrastSpec(("Primary",), "monthly_s", "frozen_cs"),
    "C3": ContrastSpec(("Primary",), "weekly_s", "monthly_s"),
    "C4": ContrastSpec(("Primary",), "weekly_c7s", "weekly_s"),
    "C5": ContrastSpec(("Primary",), "weekly_cs", "weekly_s"),
    "C6": ContrastSpec(("Primary",), "weekly_c7s", "weekly_cs"),
    "C7": ContrastSpec(("BSP", "GSP"), "weekly_s", "frozen_cs"),
    "C8s": ContrastSpec(("solar",), "gen_monthly_s", "gen_frozen_s"),
    "C8w": ContrastSpec(("wind",), "gen_monthly_s", "gen_frozen_s"),
}
DESCRIPTIVE: Final[dict[str, ContrastSpec]] = {
    "X_roll12_vs_monthly": ContrastSpec(("Primary",), "roll12_s", "monthly_s"),
    "X_roll12_vs_stack": ContrastSpec(("Primary",), "roll12_s", "frozen_cs"),
    "X_frozen_correction": ContrastSpec(("Primary",), "frozen_cs", "frozen_s"),
    "X_primary_weekly_vs_frozen_s": ContrastSpec(("Primary",), "weekly_s", "frozen_s"),
    "X_bspgsp_weekly_correction": ContrastSpec(("BSP", "GSP"), "weekly_cs", "weekly_s"),
    "X_bspgsp_monthly_vs_stack": ContrastSpec(("BSP", "GSP"), "monthly_s", "frozen_cs"),
    "X_bspgsp_weekly_vs_frozen_s": ContrastSpec(("BSP", "GSP"), "weekly_s", "frozen_s"),
    "X_primary_weekly_vs_manual": ContrastSpec(("Primary",), "weekly_s", "manual_heuristic"),
    "X_primary_stack_vs_manual": ContrastSpec(("Primary",), "frozen_cs", "manual_heuristic"),
    "X_solar_frozen_vs_manual": ContrastSpec(("solar",), "gen_frozen_s", "manual_heuristic"),
    "X_wind_frozen_vs_manual": ContrastSpec(("wind",), "gen_frozen_s", "manual_heuristic"),
}
REPRODUCTION_PAIRS: Final[dict[str, tuple[Path, str]]] = {
    "weekly": (common.EXPERIMENT_00034_ARMS_DIR / "weekly.parquet", "weekly"),
    "weekly_s": (common.EXPERIMENT_00034_ARMS_DIR / "weekly_s.parquet", "weekly_s"),
    "monthly": (common.EXPERIMENT_00034_ARMS_DIR / "monthly.parquet", "monthly"),
    "monthly_s": (common.EXPERIMENT_00034_ARMS_DIR / "monthly_s.parquet", "monthly_s"),
    "roll12": (common.EXPERIMENT_00034_ARMS_DIR / "roll12.parquet", "roll12"),
    "roll12_s": (common.EXPERIMENT_00034_ARMS_DIR / "roll12_s.parquet", "roll12_s"),
    "frozen": (common.EXPERIMENT_00034_WORK_DIR / "reference" / "n1.parquet", "frozen"),
}
"""Each arm's name, mapped to experiment 00034's stored seed-0 arm of the same configuration."""


def _arms_dir(*, smoke: bool) -> Path:
    return common.ARMS_DIR.with_name("arms_smoke") if smoke else common.ARMS_DIR


def _metrics_dir(*, smoke: bool) -> Path:
    return common.METRICS_DIR.with_name("metrics_smoke") if smoke else common.METRICS_DIR


def _results_dir(*, smoke: bool) -> Path:
    return common.RESULTS_DIR.with_name("results_smoke") if smoke else common.RESULTS_DIR


def month_totals(*, values: np.ndarray, months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each month's sum of `values` and row count, months in sorted order."""
    _unique, inverse = np.unique(months, return_inverse=True)
    return np.bincount(inverse, weights=values.astype(np.float64)), np.bincount(inverse)


def month_bounds(
    *, sums: np.ndarray, counts: np.ndarray, level: float, n_resamples: int
) -> tuple[float, float]:
    """Return the interval of the mean of rows, resampling whole months.

    The draws are those of `studies.bootstrap`'s resampler: for each resample one seed draw, then
    one draw of the months with replacement. The resampled mean of the drawn months' rows is the
    sum of their sums over the sum of their counts, so no row is touched after the totals.

    Args:
        sums: Each month's sum of the values, from `month_totals`.
        counts: Each month's row count.
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


def verify_month_bounds() -> list[str]:
    """Compare `month_bounds` with `studies.bootstrap` on synthetic paired losses.

    Returns:
        Report lines with both intervals at two levels; they agree to float precision.
    """
    rng = np.random.default_rng(1)
    months = np.array(
        [f"2025-{m:02d}" for m in rng.integers(1, 13, size=6000)]  # months of unequal size
    )
    runs = np.arange(6000)
    base = rng.normal(5.0, 1.0, size=6000)
    frame = pl.DataFrame(
        {
            "site": ["s"] * 6000,
            "time": [f"t{i:06d}" for i in runs],
            "seed": [0] * 6000,
            "month": months,
            "treatment": base - 0.1 + rng.normal(0.0, 0.3, size=6000),
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
    sums, counts = month_totals(
        values=(frame["treatment"] - frame["reference"]).to_numpy(), months=months
    )
    lines = []
    for level in (95.0, PLANNED_LEVEL):
        library = bootstrap.bootstrap_difference_at_level(
            losses=losses,
            treatment="treatment",
            reference="reference",
            metric="qs",
            level=level,
            n_resamples=2000,
        )
        fast = month_bounds(sums=sums, counts=counts, level=level, n_resamples=2000)
        lines.append(
            f"level {level:.2f}: library ({library[0]:.6f}, {library[1]:.6f}), "
            f"month totals ({fast[0]:.6f}, {fast[1]:.6f})"
        )
    return lines


def compute_metrics(*, smoke: bool) -> None:
    """Write every arm's per-case metrics, skipping an arm whose file is newer than its table."""
    series = common.series_table()
    arrays = common.load_series_arrays(series=series)
    sites = gen.site_table()
    power_grid = gen.full_power(sites=sites)
    thresholds: dict[int, float] = {
        **{
            int(r["time_series_id"]): common.historical_thresholds(power=arrays[r["site"]].power)[
                "p99"
            ]
            for r in series.iter_rows(named=True)
        },
        **{
            int(i): common.historical_thresholds(power=power)["p99"]
            for i, power in power_grid.items()
        },
    }
    folder = _metrics_dir(smoke=smoke)
    folder.mkdir(parents=True, exist_ok=True)
    finite = pl.all_horizontal(pl.col(*common.LEVEL_COLUMNS).is_finite())
    for path in sorted(_arms_dir(smoke=smoke).glob("*.parquet")):
        target = folder / path.name
        if target.exists() and target.stat().st_mtime > path.stat().st_mtime:
            continue
        frame = (
            pl.read_parquet(path)
            .filter(finite & pl.col("actual").is_finite() & (pl.col("valid_time") < common.HOLDOUT))
            .sort(*common.KEYS, "seed")
        )
        quantiles = np.sort(
            frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1
        )
        actual = frame["actual"].to_numpy().astype(np.float64)
        threshold = np.array([thresholds[int(i)] for i in frame["time_series_id"].to_numpy()])
        metrics = common.case_metrics(quantiles=quantiles, actual=actual, threshold_p99=threshold)
        table = frame.select(
            *common.KEYS,
            "seed",
            "group",
            "site",
            lead_hours=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
            ),
        ).with_columns(
            *[pl.Series(label, values.astype(np.float32)) for label, values in metrics.items()]
        )
        common.atomic_write(frame=table, target=target)
        print(f"{path.stem}: {table.height:,} cases", flush=True)


def read_arm(*, arm: str, groups: tuple[str, ...], smoke: bool) -> pl.DataFrame | None:
    """Return one arm's per-case metrics on the gap runs for some groups, or `None` if absent."""
    path = _metrics_dir(smoke=smoke) / f"{arm}.parquet"
    if not path.exists():
        return None
    frame = pl.read_parquet(path).filter(
        pl.col("group").is_in(groups) & pl.col("power_fcst_init_time").is_in(GAP_ORIGINS)
    )
    return None if frame.is_empty() else frame


def _interval_row(
    *, differences: np.ndarray, months: np.ndarray, level: float, n_resamples: int
) -> tuple[float, float]:
    sums, counts = month_totals(values=differences, months=months)
    return month_bounds(sums=sums, counts=counts, level=level, n_resamples=n_resamples)


def contrast_rows(
    *, contrast_id: str, spec: ContrastSpec, planned: bool, smoke: bool
) -> list[dict[str, object]]:
    """Return one contrast's rows: pooled over its groups, then on each series alone.

    Args:
        contrast_id: The contrast's label.
        spec: The groups and the two arms.
        planned: Whether the contrast is read at the Bonferroni level as well as at 95%.
        smoke: Whether to read the smoke tables and use few resamples.

    Returns:
        One row per series scope (`pooled` or the series label), lead scope, metric, and level.
    """
    treatment = read_arm(arm=spec.treatment, groups=spec.groups, smoke=smoke)
    reference = read_arm(arm=spec.reference, groups=spec.groups, smoke=smoke)
    if treatment is None or reference is None:
        return []
    metric_columns = ["qs", "mae", "tw_p99"]
    paired = (
        reference.select(
            *common.KEYS,
            "site",
            "lead_hours",
            *[pl.col(m).alias(f"{m}_ref") for m in metric_columns],
        )
        .join(
            treatment.select(*common.KEYS, *[pl.col(m).alias(f"{m}_trt") for m in metric_columns]),
            on=list(common.KEYS),
            how="inner",
            validate="1:1",
        )
        .with_columns(
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
            **{
                m: pl.col(f"{m}_trt").cast(pl.Float64) - pl.col(f"{m}_ref").cast(pl.Float64)
                for m in metric_columns
            },
        )
    )
    draws = 200 if smoke else N_RESAMPLES
    draws_95 = 200 if smoke else N_RESAMPLES_95
    rows: list[dict[str, object]] = []
    for site in ("pooled", *sorted(paired["site"].unique().to_list())):
        part_site = paired if site == "pooled" else paired.filter(pl.col("site") == site)
        for scope, low, high in SCOPES:
            part = part_site.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
            if part.is_empty():
                continue
            months = part["month"].to_numpy()
            metrics = metric_columns if scope == "all leads" and site == "pooled" else ["qs"]
            for metric in metrics:
                differences = part[metric].to_numpy()
                levels = [("95", 95.0, draws_95)]
                if planned and scope == "all leads" and site == "pooled":
                    levels.insert(0, ("planned", PLANNED_LEVEL, draws))
                for label, level, n_resamples in levels:
                    lower, upper = _interval_row(
                        differences=differences, months=months, level=level, n_resamples=n_resamples
                    )
                    rows.append(
                        {
                            "id": contrast_id,
                            "groups": "+".join(spec.groups),
                            "treatment": spec.treatment,
                            "reference": spec.reference,
                            "site": site,
                            "scope": scope,
                            "metric": metric,
                            "level_label": label,
                            "level": level,
                            "difference": float(differences.mean()),
                            "lower": lower,
                            "upper": upper,
                            "cases": int(differences.size),
                            "months": len(np.unique(months)),
                        }
                    )
    return rows


def compute_contrasts(*, smoke: bool) -> None:
    """Write the planned and descriptive contrasts."""
    rows: list[dict[str, object]] = []
    for family, specs, planned in (("planned", PLANNED, True), ("descriptive", DESCRIPTIVE, False)):
        for contrast_id, spec in specs.items():
            rows += [
                {**row, "family": family}
                for row in contrast_rows(
                    contrast_id=contrast_id, spec=spec, planned=planned, smoke=smoke
                )
            ]
            print(f"{contrast_id} done", flush=True)
    folder = _results_dir(smoke=smoke)
    common.atomic_write(
        frame=pl.DataFrame(rows, infer_schema_length=None), target=folder / "contrasts.parquet"
    )


BASELINE_ARMS: Final[frozenset[str]] = frozenset({"manual_heuristic", "climatology"})


def common_keys(*, group: str, smoke: bool) -> pl.DataFrame:
    """Return the cases of the gap runs that every forecasting-model arm holding the group shares.

    The baselines are left out of the intersection because they hold fewer cases, and including them
    would score the model arms on a smaller set than the planned contrasts use.
    """
    keys: pl.DataFrame | None = None
    for path in sorted(_metrics_dir(smoke=smoke).glob("*.parquet")):
        if path.stem in BASELINE_ARMS:
            continue
        frame = read_arm(arm=path.stem, groups=(group,), smoke=smoke)
        if frame is None:
            continue
        these = frame.select(*common.KEYS).unique()
        keys = these if keys is None else keys.join(these, on=list(common.KEYS), how="inner")
    if keys is None:
        msg = f"no arm holds the group {group!r}"
        raise FileNotFoundError(msg)
    return keys


def compute_absolute(*, smoke: bool) -> None:
    """Write each arm's mean metrics by group and lead band, and its quantile score by series."""
    by_group: list[dict[str, object]] = []
    by_series: list[dict[str, object]] = []
    draws = 200 if smoke else N_RESAMPLES_95
    for group in (*common.GROUPS, "solar", "wind"):
        try:
            keys = common_keys(group=group, smoke=smoke)
        except FileNotFoundError:
            continue
        for path in sorted(_metrics_dir(smoke=smoke).glob("*.parquet")):
            frame = read_arm(arm=path.stem, groups=(group,), smoke=smoke)
            if frame is None:
                continue
            frame = frame.join(keys, on=list(common.KEYS), how="semi")
            for scope, low, high in SCOPES:
                part = frame.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
                if part.is_empty():
                    continue
                row: dict[str, object] = {
                    "group": group,
                    "arm": path.stem,
                    "scope": scope,
                    "cases": part.height,
                    **{m: float(part[m].mean()) for m in MEAN_METRICS},  # ty: ignore[invalid-argument-type]
                }
                if scope == "all leads":
                    sums, counts = month_totals(
                        values=part["qs"].to_numpy(),
                        months=part["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy(),
                    )
                    row["qs_lower"], row["qs_upper"] = month_bounds(
                        sums=sums, counts=counts, level=95.0, n_resamples=draws
                    )
                by_group.append(row)
            for site, site_part in frame.group_by("site"):
                by_series.append(
                    {
                        "group": group,
                        "arm": path.stem,
                        "site": site[0],
                        "cases": site_part.height,
                        "qs": float(site_part["qs"].mean()),  # ty: ignore[invalid-argument-type]
                    }
                )
    folder = _results_dir(smoke=smoke)
    common.atomic_write(
        frame=pl.DataFrame(by_group, infer_schema_length=None), target=folder / "absolute.parquet"
    )
    common.atomic_write(
        frame=pl.DataFrame(by_series, infer_schema_length=None),
        target=folder / "absolute_series.parquet",
    )


def _stored_arm_metrics(
    *, path: Path, groups: dict[int, tuple[str, str]], seed: int
) -> pl.DataFrame:
    """Return an experiment 00034 arm's quantile score per case on the 73 runs, seed `seed`."""
    frame = (
        pl.read_parquet(path)
        .filter(
            (pl.col("seed") == seed)
            & pl.col("actual").is_finite()
            & (pl.col("valid_time") < common.HOLDOUT)
        )
        .sort(*common.KEYS)
    )
    quantiles = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1)
    qs = common.pinball_quantile_score(
        quantiles=quantiles, actual=frame["actual"].to_numpy().astype(np.float64)
    )
    return frame.select(*common.KEYS).with_columns(
        qs_stored=pl.Series(qs * 100.0),
        group=pl.Series([groups[int(i)][1] for i in frame["time_series_id"].to_numpy()]),
    )


def compute_reproduction(*, smoke: bool) -> None:
    """Compare the arms without a correction on the 73 runs with experiment 00034's stored arms."""
    series = common.series_table()
    groups = {
        int(r["time_series_id"]): (r["site"], r["substation_type"])
        for r in series.iter_rows(named=True)
    }
    rows = []
    for arm, (stored_path, _label) in REPRODUCTION_PAIRS.items():
        metrics_path = _metrics_dir(smoke=smoke) / f"{arm}.parquet"
        if not metrics_path.exists() or not stored_path.exists():
            continue
        mine = pl.read_parquet(metrics_path).filter(
            pl.col("group").is_in(common.GROUPS)
            & pl.col("power_fcst_init_time").is_in(EVERY5D_ORIGINS)
        )
        stored = _stored_arm_metrics(path=stored_path, groups=groups, seed=common.SEED)
        joined = mine.join(stored.drop("group"), on=list(common.KEYS), how="inner", validate="1:1")
        for group in common.GROUPS:
            part = joined.filter(pl.col("group") == group)
            if part.is_empty():
                continue
            mine_score, stored_score = float(part["qs"].mean()), float(part["qs_stored"].mean())  # ty: ignore[invalid-argument-type]
            rows.append(
                {
                    "arm": arm,
                    "group": group,
                    "cases": part.height,
                    "qs_new": mine_score,
                    "qs_stored": stored_score,
                    "difference": mine_score - stored_score,
                    "within_tolerance": abs(mine_score - stored_score) <= TOLERANCE,
                }
            )
    common.atomic_write(
        frame=pl.DataFrame(rows, infer_schema_length=None),
        target=_results_dir(smoke=smoke) / "reproduction.parquet",
    )
    with pl.Config(tbl_rows=100):
        print(pl.DataFrame(rows, infer_schema_length=None))


def write_arm_predictions(*, smoke: bool) -> None:
    """Save every arm's rows on the 292 gap runs as a prediction set named after the arm."""
    if smoke:
        print("smoke run: no predictions saved", flush=True)
        return
    for path in sorted(common.ARMS_DIR.glob("*.parquet")):
        frame = pl.read_parquet(path).filter(
            pl.col("power_fcst_init_time").is_in(GAP_ORIGINS)
            & (pl.col("valid_time") < common.HOLDOUT)
        )
        try:
            experiment_store.write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=path.stem,
                frame=frame.with_columns(
                    lead_hours=(
                        (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                        / 60.0
                    )
                ),
                kind="quantiles",
                row_set_id=common.ROW_SET_ID,
            )
        except FileExistsError:
            print(f"{path.stem}: already saved", flush=True)
            continue
        print(f"{path.stem}: saved {frame.height:,} rows", flush=True)


def main() -> None:
    """Run the requested parts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("metrics", "contrasts", "absolute", "reproduction", "predictions"),
        default=["metrics", "contrasts", "absolute", "reproduction", "predictions"],
    )
    parser.add_argument("--smoke", action="store_true", help="read the smoke arms, few resamples")
    parser.add_argument(
        "--verify", action="store_true", help="compare the resampler with the library"
    )
    args = parser.parse_args()
    if args.verify:
        print("\n".join(verify_month_bounds()))
        return
    _results_dir(smoke=args.smoke).mkdir(parents=True, exist_ok=True)
    if "metrics" in args.parts:
        compute_metrics(smoke=args.smoke)
    if "contrasts" in args.parts:
        compute_contrasts(smoke=args.smoke)
    if "absolute" in args.parts:
        compute_absolute(smoke=args.smoke)
    if "reproduction" in args.parts:
        compute_reproduction(smoke=args.smoke)
    if "predictions" in args.parts:
        write_arm_predictions(smoke=args.smoke)


if __name__ == "__main__":
    main()
