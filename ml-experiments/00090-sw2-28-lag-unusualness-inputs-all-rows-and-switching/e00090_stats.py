"""Assemble the arms of experiment 00090, score them, and compute every contrast and table.

Steps:

1. `metrics`: for each arm, read the monthly refits' quantiles (`e00090_fit.py`), apply experiment
   00033's spread factor of 1.15 (the production recipe), attach the observed power, and write
   per-case scores plus the
   flags `nra` (the valid time is outside every oracle event and every logged closure of the
   series), `lag_oracle` (one of the three lag times is inside an oracle event), and `lag_xl` (one
   is inside an RO-NC event) to `work/metrics/<arm>.parquet`.
2. `results`: the planned contrasts C1 and C2 at the Bonferroni level, the descriptive contrasts,
   the absolute scores and tail table, and the label coverage counts, in
   `work/results/*.parquet`.

    uv run python ml-experiments/00090-*/e00090_stats.py [--smoke]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00090_common as common
import e00090_fit as fitting
import e00090_labels as labelling
import e00090_regimes as regimes
import numpy as np
import polars as pl

from studies import bootstrap

PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 2.0)
"""The coverage in percent of a two-sided interval Bonferroni-corrected across two contrasts."""
N_RESAMPLES_PLANNED: Final[int] = 20_000
N_RESAMPLES_95: Final[int] = 4_000
GAIN_THRESHOLD: Final[float] = -0.02
"""Points of capacity: 0.4% of the plain arm's score of about 5.4, the planned `supported` bar."""
ARM_NAMES: Final[tuple[str, ...]] = fitting.ARMS
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
        subset: `all`, `nra` (rows outside switching), `lag_oracle` (`nra` rows with a lag in an
            oracle event), `lag_xl` (`nra` rows with a lag in an RO-NC event), or `lag_xl_only`
            (`lag_xl` and not `lag_oracle`).
    """

    treatment: str
    reference: str
    subset: str = "nra"


PLANNED: Final[dict[str, Contrast]] = {
    "C1": Contrast("clim_pit", "plain", "nra"),
    "C2": Contrast("clim_pit", "plain", "lag_oracle"),
}
DESCRIPTIVE: Final[dict[str, Contrast]] = {
    "D1 clim_pit on all half-hours": Contrast("clim_pit", "plain", "all"),
    "D2 clim_pit on rows with a lag in an XL event": Contrast("clim_pit", "plain", "lag_xl"),
    "D3 clim_pit on rows with a lag in an XL event only": Contrast(
        "clim_pit", "plain", "lag_xl_only"
    ),
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
    xl_labels = labelling.load_xl_labels(series=series, oracle=labels)
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
            for refit, chosen in regimes.served(name="monthly", origins=origins).items():
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
            lag_events_xl = common.origin_lags(
                power=xl_labels[site].in_event.astype(np.float32),
                target=index,
                origin=origin_index,
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
                    lag_oracle=pl.Series(np.nan_to_num(lag_events[keep]).max(axis=1) > 0.5),
                    lag_xl=pl.Series(np.nan_to_num(lag_events_xl[keep]).max(axis=1) > 0.5),
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
    """Return the cases of one of the subsets named in `Contrast`."""
    if subset == "all":
        return frame
    if subset == "nra":
        return frame.filter(pl.col("nra"))
    rows = frame.filter(pl.col("nra"))
    if subset == "lag_oracle":
        return rows.filter(pl.col("lag_oracle"))
    if subset == "lag_xl":
        return rows.filter(pl.col("lag_xl"))
    return rows.filter(pl.col("lag_xl") & ~pl.col("lag_oracle"))


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


def _planned_row(*, contrasts: pl.DataFrame, contrast_id: str) -> dict[str, float]:
    """Return a contrast's pooled, all-leads row at the Bonferroni level."""
    return contrasts.filter(
        (pl.col("id") == contrast_id)
        & (pl.col("level_label") == "planned")
        & (pl.col("site") == "pooled")
        & (pl.col("scope") == "all leads")
    ).row(0, named=True)


def verdict(*, contrasts: pl.DataFrame) -> pl.DataFrame:
    """Return the pre-registered verdict word of C1 (general) and C2 (switching).

    `refuted` when the lower end is above zero; `supported` when the upper end is below the gain
    threshold; `small gain` when it is below zero only; `inconclusive` otherwise.
    """
    rows = []
    for contrast_id in PLANNED:
        row = _planned_row(contrasts=contrasts, contrast_id=contrast_id)
        if row["lower"] > 0.0:
            word = "refuted"
        elif row["upper"] < GAIN_THRESHOLD:
            word = "supported"
        elif row["upper"] < 0.0:
            word = "small gain"
        else:
            word = "inconclusive"
        rows.append(
            {
                "id": contrast_id,
                "verdict": word,
                "difference": row["difference"],
                "lower": row["lower"],
                "upper": row["upper"],
            }
        )
    return pl.DataFrame(rows)


def label_counts(*, smoke: bool) -> pl.DataFrame:
    """Return how many scored rows each label source puts a lag inside an event for."""
    frame = read_arm(arm="plain", smoke=smoke)
    nra = frame.filter(pl.col("nra"))
    return pl.DataFrame(
        {
            "cases": [frame.height],
            "nra_cases": [nra.height],
            "lag_oracle": [int(nra["lag_oracle"].sum())],
            "lag_xl": [int(nra["lag_xl"].sum())],
            "lag_both": [int((nra["lag_oracle"] & nra["lag_xl"]).sum())],
            "lag_xl_only": [int((nra["lag_xl"] & ~nra["lag_oracle"]).sum())],
        }
    )


TAIL_COLUMNS: Final[tuple[str, ...]] = (
    "qs",
    "mae",
    "tw_p99",
    "cov98",
    "cov90",
    "cov80",
    "ex95",
    "ex99",
    "pin01",
    "pin05",
    "pin95",
    "pin99",
)


def compute_results(*, smoke: bool) -> None:
    """Write the contrasts, the absolute scores and tails by lead band, and the label counts."""
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
    for family, group in (("planned", PLANNED), ("descriptive", DESCRIPTIVE)):
        for contrast_id, contrast in group.items():
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
    verdict(contrasts=contrasts).write_parquet(results_dir / "verdict.parquet")
    label_counts(smoke=smoke).write_parquet(results_dir / "label_counts.parquet")
    absolute = []
    for arm in ARM_NAMES:
        frame = read_arm(arm=arm, smoke=smoke)
        for subset in ("nra", "all", "lag_oracle", "lag_xl", "lag_xl_only"):
            part = subset_of(frame=frame, subset=subset)
            for scope, low, high in SCOPES:
                band = part.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
                absolute.append(
                    {
                        "arm": arm,
                        "subset": subset,
                        "scope": scope,
                        **{name: _mean(band[name]) for name in TAIL_COLUMNS},
                        "cases": band.height,
                    }
                )
    pl.DataFrame(absolute).write_parquet(results_dir / "absolute.parquet")


def main() -> None:
    """Run both steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    compute_metrics(smoke=args.smoke)
    compute_results(smoke=args.smoke)
    results_dir = _dirs(smoke=args.smoke)[2]
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=60):
        print(
            pl.read_parquet(results_dir / "absolute.parquet").filter(pl.col("scope") == "all leads")
        )
        for name in ("label_counts", "verdict"):
            print(name)
            print(pl.read_parquet(results_dir / f"{name}.parquet"))
        print(
            pl.read_parquet(results_dir / "contrasts.parquet").filter(
                (pl.col("site") == "pooled") & (pl.col("scope") == "all leads")
            )
        )


if __name__ == "__main__":
    main()
