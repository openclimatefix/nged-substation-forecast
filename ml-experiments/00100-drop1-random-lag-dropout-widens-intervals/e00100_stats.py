"""Score the arms of experiment 00100 on the real rows, and compute every real-row contrast.

Steps:

1. `metrics`: for each arm, read the monthly refits' quantiles (`e00100_fit.py`), apply the spread
   factor of 1.15, attach the observed power, and write per-case scores, band widths, and the
   flags `nra` (the valid time is outside every labelled event and logged closure, so the target
   has valid normal-arrangement truth) and `lag_oracle` (one of the three lag times is inside a
   labelled event) to `work/metrics/<arm>.parquet`. Rows with `nra` false are never scored.
2. `results`: the contrasts, with whole valid-time months resampled, in
   `work/results/real_contrasts.parquet`, and a reproduction check of `plain11` against
   experiment 00086's stored `plain` arm.

    uv run python ml-experiments/00100-*/e00100_stats.py [--smoke]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00100_common as common
import e00100_fit as fitting
import e00100_labels as labelling
import e00100_regimes as regimes
import numpy as np
import polars as pl

from studies import bootstrap

PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 4.0)
"""The coverage in percent of a two-sided interval Bonferroni-corrected across four contrasts."""
N_RESAMPLES_PLANNED: Final[int] = 20_000
N_RESAMPLES_95: Final[int] = 4_000
REPRODUCTION_MARGIN: Final[float] = 0.01
"""Points of capacity: how far `plain11`'s q05 to q95 score may differ from experiment 00086's."""
EXPERIMENT_00086_PLAIN: Final[Path] = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00086-sw2-12-detector-labels-for-nra-forecast"
    / "work"
    / "metrics"
    / "plain.parquet"
)
ARM_NAMES: Final[tuple[str, ...]] = tuple(fitting.ARMS)
SCOPES: Final[tuple[tuple[str, float, float], ...]] = (
    ("all leads", 0.0, 10_000.0),
    *common.LEAD_BANDS,
)
METRICS: Final[tuple[str, ...]] = (
    "qs",
    "cov90",
    "cov98",
    "w90",
    "w98",
    "pin01",
    "pin05",
    "pin95",
    "pin99",
    "below01",
    "below05",
    "ex95",
    "ex99",
    "crossing",
)
"""Per-case columns contrasted: scores, coverages, widths (points of capacity), tail indicators."""


@dataclass(frozen=True)
class Contrast:
    """A contrast: the treatment arm minus the reference arm, on one case subset.

    Attributes:
        treatment: The arm whose value is differenced.
        reference: The arm it is differenced against.
        subset: `masked` (valid-truth rows with a lag inside a labelled event), `unmasked`
            (valid-truth rows without one), or `nra` (both).
    """

    treatment: str
    reference: str
    subset: str


PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("drop10", "plain11"),
    ("mask", "plain11"),
    ("drop10", "mask"),
    ("drop30", "drop10"),
    ("drop30", "plain11"),
)
PLANNED_METRICS: Final[dict[str, tuple[str, str, str, str]]] = {
    "P1": ("drop10", "plain11", "masked", "cov90"),
    "G2": ("drop10", "plain11", "unmasked", "qs"),
    "G2w": ("drop10", "plain11", "unmasked", "w90"),
}
"""The planned real-row reads (arm, reference, subset, metric); P2 and G1 come from the cases."""


def _dirs(*, smoke: bool) -> tuple[Path, Path, Path]:
    """Return the predictions, metrics, and results folders."""
    suffix = "_smoke" if smoke else ""
    return (
        common.ANCHOR_PREDS_DIR.with_name(f"anchor_preds{suffix}"),
        common.METRICS_DIR.with_name(f"metrics{suffix}"),
        common.RESULTS_DIR.with_name(f"results{suffix}"),
    )


def month_totals(*, values: np.ndarray, months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each month's sum of `values` and row count, months in sorted order."""
    _unique, inverse = np.unique(months, return_inverse=True)
    return np.bincount(inverse, weights=values.astype(np.float64)), np.bincount(inverse)


def month_bounds(
    *,
    numerator: np.ndarray,
    denominator: np.ndarray,
    level: float,
    n_resamples: int,
) -> tuple[float, float]:
    """Return the interval of sum(numerator) / sum(denominator), resampling whole months.

    With the row count as the denominator the statistic is a mean; with the reference arm's width
    sums as the denominator it is a ratio of means over the same rows.
    """
    generator = np.random.default_rng(bootstrap.BOOTSTRAP_SEED)
    resampled = np.empty(n_resamples)
    for draw in range(n_resamples):
        drawn = generator.integers(0, len(numerator), size=len(numerator))
        resampled[draw] = numerator[drawn].sum() / denominator[drawn].sum()
    tail = (100.0 - level) / 2.0
    return float(np.percentile(resampled, tail)), float(np.percentile(resampled, 100.0 - tail))


def compute_metrics(*, smoke: bool) -> None:
    """Write every arm's per-case scores with the `nra` and `lag_oracle` flags."""
    preds_dir, metrics_dir, _ = _dirs(smoke=smoke)
    series = common.series_table().filter(pl.col("substation_type").is_in(common.GROUPS))
    arrays = common.load_series_arrays(series=series)
    labels = labelling.load_labels(series=series)
    origins = sorted(set(common.origins_of_subset(subset="gap")))
    if smoke:
        origins = [o for o in origins if o.month == 7 and o.day in (2, 3, 4, 9, 10)]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARM_NAMES:
        pieces = []
        for row in series.iter_rows(named=True):
            site = row["site"]
            parts = []
            for refit, chosen in regimes.served(name="monthly", origins=origins).items():
                path = fitting.job_path(root=preds_dir, arm=arm, site=site, refit=refit)
                if path.exists():
                    parts.append(
                        pl.read_parquet(path).filter(pl.col("power_fcst_init_time").is_in(chosen))
                    )
            if not parts:
                continue
            raw = pl.concat(parts)
            crossing = raw["crossing"].to_numpy()
            ordered = common.stack(frame=raw, correct=False, spread=True)
            index = np.asarray(common.grid_index(times=ordered["valid_time"]))
            actual = common.take(values=arrays[site].power, index=index).astype(np.float64)
            origin_index = np.asarray(common.grid_index(times=ordered["power_fcst_init_time"]))
            lag_events = common.origin_lags(
                power=labels[site].in_event.astype(np.float32), target=index, origin=origin_index
            )
            threshold = common.historical_thresholds(power=arrays[site].power)["p99"]
            quantiles = ordered.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
            keep = (
                np.isfinite(quantiles).all(axis=1)
                & np.isfinite(actual)
                & (ordered["valid_time"] < common.HOLDOUT).to_numpy()
            )
            quantiles, actual, index = quantiles[keep], actual[keep], index[keep]
            scores = common.case_metrics(
                quantiles=quantiles,
                actual=actual,
                threshold_p99=np.full(actual.shape, threshold),
            )
            untouchable = labels[site].in_event | labels[site].closed
            pieces.append(
                ordered.filter(pl.Series(keep))
                .select(*common.KEYS, "seed")
                .with_columns(
                    site=pl.lit(site),
                    lead_hours=(
                        (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                        / 60.0
                    ),
                    nra=pl.Series(~untouchable[index]),
                    lag_oracle=pl.Series(np.nan_to_num(lag_events[keep]).max(axis=1) > 0.5),
                    w90=pl.Series(((quantiles[:, 9] - quantiles[:, 1]) * 100.0).astype(np.float32)),
                    w98=pl.Series(
                        ((quantiles[:, 10] - quantiles[:, 0]) * 100.0).astype(np.float32)
                    ),
                    crossing=pl.Series(crossing[keep].astype(np.float32)),
                    **{
                        name: pl.Series(values.astype(np.float32))
                        for name, values in scores.items()
                    },
                )
            )
        if not pieces:
            print(f"{arm}: no predictions yet", flush=True)
            continue
        table = pl.concat(pieces).sort(*common.KEYS)
        common.atomic_write(frame=table, target=metrics_dir / f"{arm}.parquet")
        print(f"{arm}: {table.height:,} cases", flush=True)


def read_arm(*, arm: str, smoke: bool) -> pl.DataFrame:
    """Return an arm's per-case scores, restricted to the rows with valid truth."""
    return pl.read_parquet(_dirs(smoke=smoke)[1] / f"{arm}.parquet").filter(pl.col("nra"))


def subset_of(*, frame: pl.DataFrame, subset: str) -> pl.DataFrame:
    """Return the cases of one subset: `masked`, `unmasked`, or `nra`."""
    if subset == "masked":
        return frame.filter(pl.col("lag_oracle"))
    if subset == "unmasked":
        return frame.filter(~pl.col("lag_oracle"))
    return frame


def paired(*, contrast: Contrast, metric: str, smoke: bool) -> pl.DataFrame:
    """Return the treatment and reference value of a metric per case, on the reference's subset."""
    reference = subset_of(
        frame=read_arm(arm=contrast.reference, smoke=smoke), subset=contrast.subset
    )
    treatment = read_arm(arm=contrast.treatment, smoke=smoke).select(
        *common.KEYS, treatment=pl.col(metric)
    )
    return (
        reference.select(*common.KEYS, "site", "lead_hours", reference=pl.col(metric))
        .join(treatment, on=list(common.KEYS), how="inner", validate="1:1")
        .with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))
    )


def contrast_rows(
    *, contrast_id: str, contrast: Contrast, metric: str, planned: bool, smoke: bool
) -> list[dict[str, object]]:
    """Return one contrast's rows: pooled, all leads and each lead band, difference and ratio."""
    frame = paired(contrast=contrast, metric=metric, smoke=smoke)
    rows: list[dict[str, object]] = []
    for scope, low, high in SCOPES:
        part = frame.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
        if part.is_empty():
            continue
        months = part["month"].to_numpy()
        treated = part["treatment"].to_numpy().astype(np.float64)
        referenced = part["reference"].to_numpy().astype(np.float64)
        difference_sums, counts = month_totals(values=treated - referenced, months=months)
        reference_sums, _ = month_totals(values=referenced, months=months)
        treatment_sums, _ = month_totals(values=treated, months=months)
        levels = [("95", 95.0, N_RESAMPLES_95 if not smoke else 200)]
        if planned and scope == "all leads":
            levels.insert(0, ("planned", PLANNED_LEVEL, N_RESAMPLES_PLANNED if not smoke else 200))
        for label, level, n_resamples in levels:
            lower, upper = month_bounds(
                numerator=difference_sums, denominator=counts, level=level, n_resamples=n_resamples
            )
            ratio_lower, ratio_upper = month_bounds(
                numerator=treatment_sums,
                denominator=reference_sums,
                level=level,
                n_resamples=n_resamples,
            )
            rows.append(
                {
                    "id": contrast_id,
                    "treatment": contrast.treatment,
                    "reference": contrast.reference,
                    "subset": contrast.subset,
                    "metric": metric,
                    "scope": scope,
                    "level_label": label,
                    "level": level,
                    "difference": float((treated - referenced).mean()),
                    "lower": lower,
                    "upper": upper,
                    "ratio": float(treated.sum() / referenced.sum()),
                    "ratio_lower": ratio_lower,
                    "ratio_upper": ratio_upper,
                    "reference_mean": float(referenced.mean()),
                    "treatment_mean": float(treated.mean()),
                    "cases": int(part.height),
                    "months": len(np.unique(months)),
                }
            )
    return rows


def compute_results(*, smoke: bool) -> None:
    """Write the planned and descriptive real-row contrasts and print the reproduction check."""
    _, _, results_dir = _dirs(smoke=smoke)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for contrast_id, (treatment, reference, subset, metric) in PLANNED_METRICS.items():
        rows += contrast_rows(
            contrast_id=contrast_id,
            contrast=Contrast(treatment, reference, subset),
            metric=metric,
            planned=True,
            smoke=smoke,
        )
    for treatment, reference in PAIRS:
        for subset in ("masked", "unmasked", "nra"):
            for metric in METRICS:
                rows += contrast_rows(
                    contrast_id=f"D {treatment} minus {reference}",
                    contrast=Contrast(treatment, reference, subset),
                    metric=metric,
                    planned=False,
                    smoke=smoke,
                )
    common.atomic_write(
        frame=pl.DataFrame(rows, infer_schema_length=None),
        target=results_dir / "real_contrasts.parquet",
    )
    print(f"wrote {len(rows):,} contrast rows", flush=True)
    if not smoke:
        reproduction_check(smoke=smoke)


def reproduction_check(*, smoke: bool) -> None:
    """Compare `plain11`'s q05 to q95 quantile score with experiment 00086's stored `plain`.

    Raises:
        ValueError: If the two scores differ by more than `REPRODUCTION_MARGIN` points.
    """
    stored = pl.read_parquet(EXPERIMENT_00086_PLAIN).filter(pl.col("nra"))
    new = read_arm(arm="plain11", smoke=smoke)
    joined = stored.select(*common.KEYS, old=pl.col("qs")).join(
        new.select(*common.KEYS, new=pl.col("qs")), on=list(common.KEYS), how="inner"
    )
    gap = float(np.mean(joined["new"].to_numpy() - joined["old"].to_numpy()))
    masked = int(new.filter(pl.col("lag_oracle")).height)
    print(
        f"reproduction: {joined.height:,} shared rows; plain11 minus stored plain qs {gap:+.4f}; "
        f"masked rows {masked:,}; stored plain mean {float(np.mean(joined['old'].to_numpy())):.3f}",
        flush=True,
    )
    if abs(gap) > REPRODUCTION_MARGIN:
        msg = f"plain11 does not reproduce experiment 00086's plain: {gap:+.4f} points"
        raise ValueError(msg)


def main() -> None:
    """Compute the per-case metrics and the real-row contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    args = parser.parse_args()
    if not args.skip_metrics:
        compute_metrics(smoke=args.smoke)
    compute_results(smoke=args.smoke)


if __name__ == "__main__":
    main()
