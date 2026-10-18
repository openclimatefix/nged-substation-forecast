"""Assemble the arms of experiment 00089, score them, and compute every contrast and table.

Steps:

1. `metrics`: for each arm, read the monthly refits' quantiles (`e00089_fit.py`), apply experiment
   00033's spread factor of 1.15, attach the observed power, and write per-case scores plus the
   flags `nra` (the valid time is outside every ARA: an XL event, a logged event, or a logged
   closure), `lag_ara` (one of the three lag times is inside an ARA), and `lag_oracle` (one is
   inside a logged event) to `work/metrics/<arm>.parquet`.
2. `results`: the planned contrasts C1 to C3 and the no-harm check at the Bonferroni level, the
   descriptive contrasts, the absolute scores and tail table, the label coverage counts, and the
   labelled events per training year and source, in `work/results/*.parquet`.

    uv run python ml-experiments/00089-*/e00089_stats.py [--smoke]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00089_common as common
import e00089_fit as fitting
import e00089_labels as labelling
import e00089_regimes as regimes
import numpy as np
import polars as pl

from studies import bootstrap

PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 3.0)
"""The coverage in percent of a two-sided interval Bonferroni-corrected across three contrasts."""
N_RESAMPLES_PLANNED: Final[int] = 20_000
N_RESAMPLES_95: Final[int] = 4_000
NON_INFERIORITY_MARGIN: Final[float] = 0.02
"""Points of capacity: the no-harm margin of experiments 00075 and 00086 (about 0.4% of a score)."""
ARM_NAMES: Final[tuple[str, ...]] = tuple(fitting.ARMS)
REFERENCE_ARM: Final[str] = "r"
EXPERIMENT_00048_ARMS: Final[dict[str, str]] = {
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
        subset: `all`, `nra` (valid time outside every ARA), `lag_ara` (`nra` rows with a lag
            inside an ARA, the headline rows), or `lag_oracle` (`nra` rows with a lag inside a
            logged event).
    """

    treatment: str
    reference: str
    subset: str = "nra"


PLANNED: Final[dict[str, Contrast]] = {
    "C1": Contrast("lm", "l", "lag_ara"),
    "C2": Contrast("l", "r", "nra"),
    "C3": Contrast("lm", "r", "lag_ara"),
}
NO_HARM: Final[Contrast] = Contrast("lm", "l", "nra")
"""C1 on all rows outside switching, read at the planned level against the margin."""
DESCRIPTIVE: Final[dict[str, Contrast]] = {
    "D1 L minus R on rows with a lag inside an ARA": Contrast("l", "r", "lag_ara"),
    "D2 LM minus R on all rows outside switching": Contrast("lm", "r", "nra"),
    "D3 M minus R on rows with a lag inside an ARA": Contrast("m", "r", "lag_ara"),
    "D4 M minus R on all rows outside switching": Contrast("m", "r", "nra"),
    "D5 LM_ens minus L_ens on rows with a lag inside an ARA": Contrast(
        "lm_ens", "l_ens", "lag_ara"
    ),
    "D6 LM_ens minus L_ens on all rows outside switching": Contrast("lm_ens", "l_ens", "nra"),
    "D7 L_ens minus R on all rows outside switching": Contrast("l_ens", "r", "nra"),
    "D8 L minus L_ens on all rows outside switching": Contrast("l", "l_ens", "nra"),
    "D9 LM minus L on rows with a lag inside a logged event": Contrast("lm", "l", "lag_oracle"),
    "D10 L minus R on all half-hours": Contrast("l", "r", "all"),
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
    ara = {
        site: fitting.ara_array(labels={"oracle": labels[site], "xl": xl_labels[site]})
        for site in labels
    }
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
            lag_ara = common.origin_lags(
                power=ara[site].astype(np.float32), target=index, origin=origin_index
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
                    nra=pl.Series(~ara[site][index]),
                    lag_oracle=pl.Series(np.nan_to_num(lag_events[keep]).max(axis=1) > 0.5),
                    lag_ara=pl.Series(np.nan_to_num(lag_ara[keep]).max(axis=1) > 0.5),
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
    rows = frame.filter(pl.col("nra"))
    if subset == "nra":
        return rows
    return rows.filter(pl.col("lag_oracle") if subset == "lag_oracle" else pl.col("lag_ara"))


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


def verdict(*, contrasts: pl.DataFrame) -> dict[str, object]:
    """Return the pre-registered verdict from C1, C2, and the no-harm check.

    Masking is supported when C1's upper end is below zero and the no-harm upper end is at most
    the margin. The long history is supported when C2's upper end is below zero. The verdict is
    `refuted` when C1's or C2's lower end is above zero, `supported` when both are supported,
    `small gain` when one is, and `inconclusive` otherwise.
    """
    if contrasts.filter(pl.col("id") == "C1").is_empty():
        return {"verdict": "no rows"}
    c1 = _planned_row(contrasts=contrasts, contrast_id="C1")
    c2 = _planned_row(contrasts=contrasts, contrast_id="C2")
    c3 = _planned_row(contrasts=contrasts, contrast_id="C3")
    harm = _planned_row(contrasts=contrasts, contrast_id="N1")
    masking = c1["upper"] < 0.0 and harm["upper"] <= NON_INFERIORITY_MARGIN
    history = c2["upper"] < 0.0
    if c1["lower"] > 0.0 or c2["lower"] > 0.0:
        word = "refuted"
    elif masking and history:
        word = "supported"
    elif masking or history:
        word = "small gain"
    else:
        word = "inconclusive"
    return {
        "verdict": word,
        "masking_supported": masking,
        "history_supported": history,
        "c1": c1["difference"],
        "c1_lower": c1["lower"],
        "c1_upper": c1["upper"],
        "c2": c2["difference"],
        "c2_lower": c2["lower"],
        "c2_upper": c2["upper"],
        "c3": c3["difference"],
        "c3_lower": c3["lower"],
        "c3_upper": c3["upper"],
        "no_harm_upper": harm["upper"],
    }


def label_counts(*, smoke: bool) -> pl.DataFrame:
    """Return how many scored rows have a lag inside an ARA, by label source."""
    frame = read_arm(arm=REFERENCE_ARM, smoke=smoke)
    nra = frame.filter(pl.col("nra"))
    return pl.DataFrame(
        {
            "cases": [frame.height],
            "nra_cases": [nra.height],
            "lag_ara": [int(nra["lag_ara"].sum())],
            "lag_oracle": [int(nra["lag_oracle"].sum())],
        }
    )


def events_per_year() -> pl.DataFrame:
    """Return, per calendar year and label source, the events and labelled half-hours.

    The XL source counts detector intervals by onset; the oracle source counts logged events and
    closures by their onset or logged close. Half-hours are those of the 16 series summed.
    """
    series = common.series_table().filter(pl.col("substation_type").is_in(common.GROUPS))
    oracle = labelling.load_labels(series=series)
    xl = labelling.load_xl_labels(series=series, oracle=oracle)
    times = common.grid_times()
    years = times.dt.year().to_numpy()
    rows = []
    for year in range(2021, 2027):
        mask = years == year
        rows.append(
            {
                "year": year,
                "xl_halfhours": int(sum(xl[s].in_event[mask].sum() for s in xl)),
                "oracle_event_halfhours": int(sum(oracle[s].in_event[mask].sum() for s in oracle)),
                "oracle_closed_halfhours": int(sum(oracle[s].closed[mask].sum() for s in oracle)),
                "ara_halfhours": int(
                    sum(
                        fitting.ara_array(labels={"oracle": oracle[s], "xl": xl[s]})[mask].sum()
                        for s in oracle
                    )
                ),
                "all_halfhours": int(mask.sum()) * len(oracle),
            }
        )
    return pl.DataFrame(rows)


def compute_results(*, smoke: bool) -> None:
    """Write the contrasts, the absolute scores, the benchmarks, and the label checks."""
    _, _, results_dir = _dirs(smoke=smoke)
    results_dir.mkdir(parents=True, exist_ok=True)
    key_sets = {
        arm: read_arm(arm=arm, smoke=smoke).select(*common.KEYS).sort(*common.KEYS)
        for arm in ARM_NAMES
    }
    for arm, keys in key_sets.items():
        if not keys.equals(key_sets[REFERENCE_ARM]):
            msg = f"arm {arm} does not hold the same cases as the reference arm"
            raise ValueError(msg)
    rows: list[dict[str, object]] = []
    specs = (
        ("planned", {**PLANNED, "N1": NO_HARM}),
        ("descriptive", DESCRIPTIVE),
    )
    for family, group in specs:
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
    pl.DataFrame([verdict(contrasts=contrasts)]).write_parquet(results_dir / "verdict.parquet")
    label_counts(smoke=smoke).write_parquet(results_dir / "label_counts.parquet")
    events_per_year().write_parquet(results_dir / "events_per_year.parquet")
    plain = read_arm(arm=REFERENCE_ARM, smoke=smoke)
    absolute = []
    for arm in ARM_NAMES:
        frame = read_arm(arm=arm, smoke=smoke)
        for subset in ("nra", "all", "lag_ara", "lag_oracle"):
            part = subset_of(frame=frame, subset=subset)
            absolute.append(
                {
                    "arm": arm,
                    "subset": subset,
                    "qs": _mean(part["qs"]),
                    "mae": _mean(part["mae"]),
                    "tw_p99": _mean(part["tw_p99"]),
                    "cov90": _mean(part["cov90"]),
                    "pin05": _mean(part["pin05"]),
                    "pin95": _mean(part["pin95"]),
                    "ex05": _mean(part["ex05"]),
                    "ex95": _mean(part["ex95"]),
                    "cases": part.height,
                }
            )
    baseline_dir = common.EXPERIMENT_00048_WORK_DIR / "metrics"
    keys_of = {
        subset: subset_of(frame=plain, subset=subset).select(*common.KEYS)
        for subset in ("nra", "all", "lag_ara")
    }
    for name, label in EXPERIMENT_00048_ARMS.items():
        path = baseline_dir / f"{name}.parquet"
        if not path.exists():
            continue
        other = pl.read_parquet(path).filter(pl.col("group") == "Primary")
        for subset, keys in keys_of.items():
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
                    "cov90": float("nan"),
                    "pin05": float("nan"),
                    "pin95": float("nan"),
                    "ex05": float("nan"),
                    "ex95": float("nan"),
                    "cases": part.height,
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
        for name in ("absolute", "label_counts", "events_per_year", "verdict"):
            print(name)
            print(pl.read_parquet(results_dir / f"{name}.parquet"))
        print(
            pl.read_parquet(results_dir / "contrasts.parquet").filter(
                (pl.col("site") == "pooled") & (pl.col("scope") == "all leads")
            )
        )


if __name__ == "__main__":
    main()
