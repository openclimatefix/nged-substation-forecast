"""Assemble the arms of experiment 00075, score them, and compute every contrast and table.

Steps, each skipped when its output is newer than its input:

1. `metrics`: for each arm, read the weekly refits' quantiles (`e00075_fit.py`), apply experiment
   00033's spread factor of 1.15, attach the observed power, and write per-case scores plus the
   flags `nra` (the valid time is outside every labelled event and every logged closure of the
   series) and `lag_in_event` (one of the three lag times is inside a labelled event) to
   `work/metrics/<arm>.parquet`.
2. `results`: the planned contrasts C1 to C3 at the Bonferroni level, the descriptive contrasts,
   the absolute scores, the reproduction check against experiment 00048's stored weekly scores,
   and experiment 00048's baseline arms on the same cases, in `work/results/*.parquet`.

    uv run python ml-experiments/00075-*/e00075_stats.py [--smoke]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00075_common as common
import e00075_fit as fitting
import e00075_labels as labelling
import e00075_regimes as regimes
import numpy as np
import polars as pl

from studies import bootstrap

PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 3.0)
"""The coverage in percent of a two-sided interval Bonferroni-corrected across three contrasts."""
N_RESAMPLES_PLANNED: Final[int] = 20_000
N_RESAMPLES_95: Final[int] = 4_000
NON_INFERIORITY_MARGIN: Final[float] = 0.02
"""Points of capacity: 0.34% of the plain arm's 5.44, the most experiment 00044 says a perfect
detector could gain."""
ARM_NAMES: Final[tuple[str, ...]] = fitting.ARMS
EXPERIMENT_00048_ARMS: Final[dict[str, str]] = {
    "weekly_s": "experiment 00048 weekly",
    "weekly_cs": "experiment 00048 weekly + 14-day correction",
    "frozen_cs": "experiment 00048 stack",
    "manual_heuristic": "manual_heuristic",
    "climatology": "climatology",
}
"""Experiment 00048's metrics files read as benchmarks, with their labels."""
SCOPES: Final[tuple[tuple[str, float, float], ...]] = (
    ("all leads", 0.0, 10_000.0),
    *common.LEAD_BANDS,
)


@dataclass(frozen=True)
class Contrast:
    """A contrast: the treatment arm minus the reference arm, on one case subset.

    Attributes:
        treatment: The arm whose score is differenced.
        reference: The arm it is differenced against.
        subset: `nra`, `all`, or `lag_in_event` (NRA rows whose lag is in an event).
    """

    treatment: str
    reference: str
    subset: str = "nra"


PLANNED: Final[dict[str, Contrast]] = {
    "C1": Contrast("nra", "plain"),
    "C2": Contrast("nra_mw", "nra"),
    "C3": Contrast("nra_lag", "nra"),
}
DESCRIPTIVE: Final[dict[str, Contrast]] = {
    "D1 plain minus nra_mw": Contrast("plain", "nra_mw"),
    "D2 plain minus nra_lag": Contrast("plain", "nra_lag"),
    "D3 nra_lag minus nra_mw": Contrast("nra_lag", "nra_mw"),
    "D4 C1 on all half-hours": Contrast("nra", "plain", "all"),
    "D5 C2 on all half-hours": Contrast("nra_mw", "nra", "all"),
    "D6 C3 on all half-hours": Contrast("nra_lag", "nra", "all"),
    "D7 C1 on lag-in-event rows": Contrast("nra", "plain", "lag_in_event"),
    "D8 C2 on lag-in-event rows": Contrast("nra_mw", "nra", "lag_in_event"),
    "D9 C3 on lag-in-event rows": Contrast("nra_lag", "nra", "lag_in_event"),
}


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
    *, sums: np.ndarray, counts: np.ndarray, level: float, n_resamples: int
) -> tuple[float, float]:
    """Return the interval of the mean of rows, resampling whole months.

    The draws are those of `studies.bootstrap`'s resampler for one seed: for each resample one seed
    draw, then one draw of the months with replacement.
    """
    generator = np.random.default_rng(bootstrap.BOOTSTRAP_SEED)
    resampled = np.empty(n_resamples)
    for draw in range(n_resamples):
        generator.integers(0, 1)
        drawn = generator.integers(0, len(sums), size=len(sums))
        resampled[draw] = sums[drawn].sum() / counts[drawn].sum()
    tail = (100.0 - level) / 2.0
    return float(np.percentile(resampled, tail)), float(np.percentile(resampled, 100.0 - tail))


def compute_metrics(*, smoke: bool) -> None:
    """Write every arm's per-case scores with the `nra` and `lag_in_event` flags."""
    preds_dir, metrics_dir, _ = _dirs(smoke=smoke)
    series = common.series_table().filter(pl.col("substation_type").is_in(common.GROUPS))
    arrays = common.load_series_arrays(series=series)
    labels = labelling.load_labels(series=series)
    gap = set(common.origins_of_subset(subset="gap"))
    origins = sorted(gap)
    if smoke:
        origins = [o for o in origins if o.month == 7 and o.day in (2, 3, 4, 9, 10)]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARM_NAMES:
        target = metrics_dir / f"{arm}.parquet"
        pieces = []
        for row in series.iter_rows(named=True):
            site = row["site"]
            parts = []
            for refit, chosen in regimes.served(name="weekly", origins=origins).items():
                path = fitting.job_path(root=preds_dir, arm=arm, site=site, refit=refit)
                if not path.exists():
                    continue
                frame = pl.read_parquet(path).filter(pl.col("power_fcst_init_time").is_in(chosen))
                parts.append(frame)
            if not parts:
                continue
            quantile_frame = pl.concat(parts)
            sorted_q = common.stack(frame=quantile_frame, correct=False, spread=True)
            index = np.asarray(common.grid_index(times=sorted_q["valid_time"]))
            actual = common.take(values=arrays[site].power, index=index).astype(np.float64)
            origin_index = np.asarray(common.grid_index(times=sorted_q["power_fcst_init_time"]))
            lag_events = common.origin_lags(
                power=labels[site].in_event.astype(np.float32), target=index, origin=origin_index
            )
            threshold = common.historical_thresholds(power=arrays[site].power)["p99"]
            finite = np.isfinite(sorted_q.select(common.LEVEL_COLUMNS).to_numpy()).all(axis=1)
            keep = (
                finite & np.isfinite(actual) & (sorted_q["valid_time"] < common.HOLDOUT).to_numpy()
            )
            sorted_q = sorted_q.filter(pl.Series(keep))
            actual = actual[keep]
            index = index[keep]
            scores = common.case_metrics(
                quantiles=sorted_q.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64),
                actual=actual,
                threshold_p99=np.full(actual.shape, threshold),
            )
            pieces.append(
                sorted_q.select(*common.KEYS, "seed").with_columns(
                    site=pl.lit(site),
                    lead_hours=(
                        (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                        / 60.0
                    ),
                    nra=pl.Series(~labels[site].in_event[index] & ~labels[site].closed[index]),
                    lag_in_event=pl.Series(np.nan_to_num(lag_events[keep]).max(axis=1) > 0.5),
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
        common.atomic_write(frame=table, target=target)
        print(f"{arm}: {table.height:,} cases", flush=True)


def _mean(values: pl.Series) -> float:
    """Return the mean, NaN for an empty series."""
    return float(np.mean(values.to_numpy())) if values.len() else float("nan")


def read_arm(*, arm: str, smoke: bool) -> pl.DataFrame:
    """Return an arm's per-case scores."""
    return pl.read_parquet(_dirs(smoke=smoke)[1] / f"{arm}.parquet")


def subset_of(*, frame: pl.DataFrame, subset: str) -> pl.DataFrame:
    """Return the cases of `all`, `nra`, or `lag_in_event` (NRA rows with a lag in an event)."""
    if subset == "all":
        return frame
    if subset == "nra":
        return frame.filter(pl.col("nra"))
    return frame.filter(pl.col("nra") & pl.col("lag_in_event"))


def paired_frame(*, contrast: Contrast, smoke: bool) -> pl.DataFrame:
    """Return the treatment-minus-reference quantile-score difference per case, with its month."""
    reference = subset_of(
        frame=read_arm(arm=contrast.reference, smoke=smoke), subset=contrast.subset
    )
    treatment = read_arm(arm=contrast.treatment, smoke=smoke).select(
        *common.KEYS, treatment=pl.col("qs")
    )
    return (
        reference.select(*common.KEYS, "site", "lead_hours", reference=pl.col("qs"))
        .join(treatment, on=list(common.KEYS), how="inner", validate="1:1")
        .with_columns(
            difference=pl.col("treatment").cast(pl.Float64) - pl.col("reference").cast(pl.Float64),
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        )
    )


def contrast_rows(
    *, contrast_id: str, contrast: Contrast, planned: bool, smoke: bool
) -> list[dict[str, object]]:
    """Return one contrast's rows: pooled and on each series, over all leads and each lead band."""
    paired = paired_frame(contrast=contrast, smoke=smoke)
    rows: list[dict[str, object]] = []
    for site in ("pooled", *sorted(paired["site"].unique().to_list())):
        part_site = paired if site == "pooled" else paired.filter(pl.col("site") == site)
        for scope, low, high in SCOPES:
            part = part_site.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
            if part.is_empty():
                continue
            months = part["month"].to_numpy()
            differences = part["difference"].to_numpy()
            sums, counts = month_totals(values=differences, months=months)
            levels = [("95", 95.0, N_RESAMPLES_95 if not smoke else 200)]
            if planned and scope == "all leads" and site == "pooled":
                levels.insert(
                    0, ("planned", PLANNED_LEVEL, N_RESAMPLES_PLANNED if not smoke else 200)
                )
            for label, level, n_resamples in levels:
                lower, upper = month_bounds(
                    sums=sums, counts=counts, level=level, n_resamples=n_resamples
                )
                rows.append(
                    {
                        "id": contrast_id,
                        "treatment": contrast.treatment,
                        "reference": contrast.reference,
                        "subset": contrast.subset,
                        "site": site,
                        "scope": scope,
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


def verdicts(*, contrasts: pl.DataFrame) -> list[dict[str, object]]:
    """Return the pre-registered reading of C1 to C3 from their Bonferroni-level intervals."""
    out = []
    for contrast_id in PLANNED:
        row = contrasts.filter(
            (pl.col("id") == contrast_id)
            & (pl.col("level_label") == "planned")
            & (pl.col("site") == "pooled")
            & (pl.col("scope") == "all leads")
        ).row(0, named=True)
        gain = row["upper"] < 0.0
        if contrast_id == "C1":
            reading = (
                "gain"
                if gain
                else (
                    "non-inferior"
                    if row["upper"] <= NON_INFERIORITY_MARGIN
                    else "not shown non-inferior"
                )
            )
        else:
            reading = "supported" if gain else "not supported"
        out.append(
            {
                "id": contrast_id,
                "reading": reading,
                "difference": row["difference"],
                "lower": row["lower"],
                "upper": row["upper"],
            }
        )
    return out


def compute_results(*, smoke: bool) -> None:
    """Write the contrasts, the absolute scores, the benchmarks, and the reproduction check."""
    _, _, results_dir = _dirs(smoke=smoke)
    results_dir.mkdir(parents=True, exist_ok=True)
    key_sets = {
        arm: read_arm(arm=arm, smoke=smoke).select(*common.KEYS).sort(*common.KEYS)
        for arm in ARM_NAMES
    }
    for arm, keys in key_sets.items():
        if not keys.equals(key_sets["plain"]):
            msg = f"arm {arm} does not hold the same cases as the plain arm"
            raise ValueError(msg)
    rows: list[dict[str, object]] = []
    for family, specs in (("planned", PLANNED), ("descriptive", DESCRIPTIVE)):
        for contrast_id, contrast in specs.items():
            rows.extend(
                contrast_rows(
                    contrast_id=contrast_id,
                    contrast=contrast,
                    planned=family == "planned",
                    smoke=smoke,
                )
            )
    contrasts = pl.DataFrame(rows)
    contrasts.write_parquet(results_dir / "contrasts.parquet")
    pl.DataFrame(verdicts(contrasts=contrasts)).write_parquet(results_dir / "verdicts.parquet")
    nra_keys = subset_of(frame=read_arm(arm="plain", smoke=smoke), subset="nra").select(
        *common.KEYS
    )
    all_keys = read_arm(arm="plain", smoke=smoke).select(*common.KEYS)
    absolute = []
    for arm in ARM_NAMES:
        frame = read_arm(arm=arm, smoke=smoke)
        for subset in ("nra", "all", "lag_in_event"):
            part = subset_of(frame=frame, subset=subset)
            absolute.append(
                {
                    "arm": arm,
                    "subset": subset,
                    "qs": _mean(part["qs"]),
                    "mae": _mean(part["mae"]),
                    "tw_p99": _mean(part["tw_p99"]),
                    "cases": part.height,
                }
            )
    baseline_dir = common.EXPERIMENT_00048_WORK_DIR / "metrics"
    reproduction = []
    plain = read_arm(arm="plain", smoke=smoke).select(*common.KEYS, plain=pl.col("qs"))
    for name, label in EXPERIMENT_00048_ARMS.items():
        path = baseline_dir / f"{name}.parquet"
        if not path.exists():
            continue
        other = pl.read_parquet(path).filter(pl.col("group") == "Primary")
        for subset, keys in (("nra", nra_keys), ("all", all_keys)):
            part = keys.join(
                other.select(*common.KEYS, "qs"), on=list(common.KEYS), how="inner", validate="1:1"
            )
            absolute.append(
                {
                    "arm": label,
                    "subset": subset,
                    "qs": _mean(part["qs"]),
                    "mae": float("nan"),
                    "tw_p99": float("nan"),
                    "cases": part.height,
                }
            )
        if name == "weekly_s":
            joined = plain.join(
                other.select(*common.KEYS, theirs=pl.col("qs")),
                on=list(common.KEYS),
                how="inner",
                validate="1:1",
            ).with_columns(gap=(pl.col("plain") - pl.col("theirs")).abs())
            reproduction.append(
                {
                    "cases": joined.height,
                    "mean_plain": _mean(joined["plain"]),
                    "mean_experiment_48": _mean(joined["theirs"]),
                    "max_abs_gap": float(np.max(joined["gap"].to_numpy())),
                    "share_within_0_001": _mean(joined["gap"] <= 0.001),
                }
            )
    pl.DataFrame(absolute).write_parquet(results_dir / "absolute.parquet")
    pl.DataFrame(reproduction).write_parquet(results_dir / "reproduction.parquet")


def main() -> None:
    """Run both steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    compute_metrics(smoke=args.smoke)
    compute_results(smoke=args.smoke)
    results_dir = _dirs(smoke=args.smoke)[2]
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=60):
        print(pl.read_parquet(results_dir / "absolute.parquet"))
        print(pl.read_parquet(results_dir / "reproduction.parquet"))
        print(pl.read_parquet(results_dir / "verdicts.parquet"))
        print(
            pl.read_parquet(results_dir / "contrasts.parquet").filter(
                (pl.col("site") == "pooled") & (pl.col("scope") == "all leads")
            )
        )


if __name__ == "__main__":
    main()
