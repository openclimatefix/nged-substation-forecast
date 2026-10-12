"""Print paired study scores with equal-series weighting and whole-month uncertainty.

These are sampled-run study errors, not official leaderboard submissions or skill numbers.
"""

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from scipy import stats

PAIRS: Final[tuple[tuple[str, str, str], ...]] = (
    ("all_fixed", "baseline", "All corrections (planned)"),
    ("all_fixed", "without_radiation", "Radiation conditional contribution (planned)"),
    ("all_fixed", "without_wind", "Wind conditional contribution (planned)"),
    ("all_fixed", "without_windchill", "Windchill conditional contribution (planned)"),
    ("mask_precipitation", "all_fixed", "Precipitation availability sensitivity (planned)"),
    ("mask_weekday", "all_fixed", "Ordinal weekday masking (exploratory)"),
)
KEYS: Final[list[str]] = ["site", "power_fcst_init_time", "valid_time"]


def markdown(frame: pl.DataFrame) -> str:
    """Render exact script results into a readable evidence table."""

    def cell(value: object) -> str:
        return f"{value:.5f}" if isinstance(value, float) else str(value)

    return "\n".join(
        [
            "| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |",
            *("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.rows()),
        ]
    )


def check_pairs(frames: dict[tuple[str, str, int], pl.DataFrame]) -> None:
    """Refuse duplicate, omitted or changed forecast cases before any inner pairing."""
    for (setting, arm, seed), frame in frames.items():
        if frame.select(KEYS).is_duplicated().any():
            raise ValueError("duplicate score keys")
        baseline = frames[(setting, "baseline", 0)]
        columns = [*KEYS, "members", "actual", "time_series_type", "lead"]
        if not frame.select(columns).sort(KEYS).equals(baseline.select(columns).sort(KEYS)):
            raise ValueError(f"score keys, targets or members differ: {setting}/{arm}/seed{seed}")
        if not frame.select(pl.col("crps", "mae").is_finite().all()).row(0) == (True, True):
            raise ValueError("non-finite losses")


def scopes(frame: pl.DataFrame) -> dict[str, pl.Expr]:
    """Define disjoint type and lead descriptions, with all cases as the primary scope."""
    result = {"All types, all leads": pl.lit(True)}
    result.update(
        {
            f"Type: {kind}": pl.col("time_series_type") == kind
            for kind in sorted(frame["time_series_type"].unique().to_list())
        }
    )
    result.update(
        {
            "Lead: 0–72 h": pl.col("lead") < 72,
            "Lead: 72–144 h": pl.col("lead").is_between(72, 144, closed="left"),
            "Lead: 144–264 h": pl.col("lead").is_between(144, 264, closed="left"),
            "Lead: 264 h and later": pl.col("lead") >= 264,
        }
    )
    return result


def paired_interval(
    parts: list[pl.DataFrame], treatment: str, reference: str, metric: str
) -> dict[str, float | int]:
    """Resample months and one common seed, recomputing each series' count-weighted mean."""
    seed_parts = []
    for frame in parts:
        left = frame.filter(pl.col("arm") == treatment).select(
            *KEYS, "month", treatment=pl.col(metric)
        )
        right = frame.filter(pl.col("arm") == reference).select(*KEYS, reference=pl.col(metric))
        paired = left.join(right, on=KEYS, how="inner").with_columns(
            delta=pl.col("treatment") - pl.col("reference")
        )
        seed_parts.append(
            paired.group_by("site", "month").agg(total=pl.col("delta").sum(), count=pl.len())
        )
    sites = sorted(set(seed_parts[0]["site"]))
    months = sorted(set(seed_parts[0]["month"]))
    sums = np.zeros((len(parts), len(sites), len(months)))
    counts = np.zeros_like(sums)
    for seed, table in enumerate(seed_parts):
        for site, month, total, count in table.rows():
            sums[seed, sites.index(site), months.index(month)] = total
            counts[seed, sites.index(site), months.index(month)] = count
    estimates = np.mean(sums.sum(axis=2) / counts.sum(axis=2), axis=1)
    rng = np.random.default_rng(2026101202)
    draws = []
    for _ in range(2000):
        seed = rng.integers(len(parts))
        multiplicities = np.bincount(
            rng.integers(len(months), size=len(months)), minlength=len(months)
        )
        numerator, denominator = sums[seed] @ multiplicities, counts[seed] @ multiplicities
        eligible = denominator > 0
        draws.append(float(np.mean(numerator[eligible] / denominator[eligible])))
    lower, upper = np.percentile(draws, [2.5, 97.5])
    monthly = np.mean(
        np.nanmean(
            np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0), axis=1
        ),
        axis=0,
    )
    half_width = stats.t.ppf(0.975, len(months) - 1) * stats.sem(monthly)
    return {
        "difference": float(np.mean(estimates)),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "month_mean_difference": float(np.mean(monthly)),
        "month_t_lower": float(np.mean(monthly) - half_width),
        "month_t_upper": float(np.mean(monthly) + half_width),
        "improving_months": int((monthly < 0).sum()),
        "months": len(months),
        "series": len(sites),
        "seed_min": float(estimates.min()),
        "seed_max": float(estimates.max()),
    }


def build_tables(
    frames: dict[tuple[str, str, int], pl.DataFrame],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build absolute errors and all declared paired comparisons, with explicit scope labels."""
    levels, contrasts = [], []
    for setting in sorted({key[0] for key in frames}):
        seeds = sorted({key[2] for key in frames if key[0] == setting})
        arms = sorted({key[1] for key in frames if key[0] == setting})
        for scope, condition in scopes(next(iter(frames.values()))).items():
            parts = [
                pl.concat([frames[(setting, arm, seed)].filter(condition) for arm in arms])
                for seed in seeds
            ]
            for arm in arms:
                for metric in ("crps", "mae"):
                    values = [
                        part.filter(pl.col("arm") == arm)
                        .group_by("site")
                        .agg(pl.col(metric).mean())[metric]
                        .to_numpy()
                        .mean()
                        for part in parts
                    ]
                    levels.append(
                        {
                            "setting": setting,
                            "scope": scope,
                            "arm": arm,
                            "metric": metric,
                            "value": float(np.mean(values)),
                            "seed_min": float(min(values)),
                            "seed_max": float(max(values)),
                        }
                    )
            for treatment, reference, label in PAIRS:
                if treatment not in arms or reference not in arms:
                    continue
                contrasts.extend(
                    {
                        "setting": setting,
                        "scope": scope,
                        "contrast": label,
                        "metric": metric,
                        **paired_interval(parts, treatment, reference, metric),
                    }
                    for metric in ("crps", "mae")
                )
    return pl.DataFrame(levels), pl.DataFrame(contrasts)


def population_evidence(
    frames: dict[tuple[str, str, int], pl.DataFrame], raw_cases: int
) -> pl.DataFrame:
    """Print independently defined case denominators and observed-target exclusions."""
    rows = []
    for (setting, arm, seed), frame in frames.items():
        rows.append(
            {
                "setting": setting,
                "arm": arm,
                "seed": seed,
                "input_cases": raw_cases,
                "scored_cases": frame.height,
                "missing_or_nonfinite_observation_cases": raw_cases - frame.height,
                "series": frame["site"].n_unique(),
                "months": frame["month"].n_unique(),
                "min_members": frame["members"].min(),
                "max_members": frame["members"].max(),
            }
        )
    return pl.DataFrame(rows)


def monthly_evidence(frames: dict[tuple[str, str, int], pl.DataFrame]) -> pl.DataFrame:
    """Expose the monthly differences behind the robustness gate, with cases and series."""
    rows = []
    for setting in sorted({key[0] for key in frames}):
        arms = {key[1] for key in frames if key[0] == setting}
        for treatment, reference, label in PAIRS:
            if treatment not in arms or reference not in arms:
                continue
            for seed in range(3):
                left, right = frames[(setting, treatment, seed)], frames[(setting, reference, seed)]
                paired = left.select(*KEYS, "month", "crps", "mae").join(
                    right.select(
                        *KEYS, pl.col("crps").alias("ref_crps"), pl.col("mae").alias("ref_mae")
                    ),
                    on=KEYS,
                )
                by_series = paired.group_by("site", "month").agg(
                    crps=(pl.col("crps") - pl.col("ref_crps")).mean(),
                    mae=(pl.col("mae") - pl.col("ref_mae")).mean(),
                    cases=pl.len(),
                )
                monthly = by_series.group_by("month").agg(
                    pl.col("crps", "mae").mean(), pl.col("cases").sum(), series=pl.len()
                )
                rows.append(
                    monthly.with_columns(
                        setting=pl.lit(setting), contrast=pl.lit(label), seed=pl.lit(seed)
                    )
                )
    return (
        pl.concat(rows)
        .group_by("setting", "contrast", "month")
        .agg(pl.col("crps", "mae").mean(), pl.col("cases", "series").first())
        .sort("setting", "contrast", "month")
    )


def reference_coverage(path: Path, training_path: Path, baseline: pl.DataFrame) -> str:
    """Use a cached secondary reference only when its scored cases exactly match this study."""
    ids = (
        pl.scan_parquet(training_path)
        .select("time_series_id")
        .unique()
        .collect()["time_series_id"]
        .sort()
        .to_list()
    )
    permutation = np.random.default_rng(2026101201).permutation(len(ids))
    labels = {
        series: f"S{position + 1:02d}" for series, position in zip(ids, permutation, strict=True)
    }
    reference = pl.read_parquet(path).select(
        pl.col("time_series_id").replace_strict(labels).alias("site"),
        "power_fcst_init_time",
        "valid_time",
    )
    if reference.is_duplicated().any():
        raise ValueError("duplicate cached secondary-reference cases")
    missing = baseline.select(KEYS).join(reference, on=KEYS, how="anti").height
    extra = reference.join(baseline.select(KEYS), on=KEYS, how="anti").height
    return (
        f"Cached manual_heuristic: {reference.height} cases; study baseline: {baseline.height}; "
        f"missing relative to study: {missing}; extra: {extra}. "
        "No secondary score quoted: historical cached scores do not establish current "
        "identical targets and complete member labels."
    )


def read_scores(
    directory: Path, *, combined_only: bool
) -> dict[tuple[str, str, int], pl.DataFrame]:
    """Read only complete preregistered fitting matrices and verify exact pairing."""
    frames = {}
    for path in sorted((directory / "scores").glob("*.parquet")):
        if path.stem.endswith("_importance"):
            continue
        frame = pl.read_parquet(path).with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))
        setting, arm, seed = frame.select("setting", "arm", "seed").unique().row(0)
        if setting == "exact_production" and arm not in {"baseline", "all_fixed"}:
            continue
        frames[(setting, arm, seed)] = frame
    expected_arms = (
        ["baseline", "all_fixed"]
        if combined_only
        else [
            "baseline",
            "all_fixed",
            "without_radiation",
            "without_wind",
            "without_windchill",
            "mask_precipitation",
            "mask_weekday",
        ]
    )
    for setting in ("controlled", "sensitivity"):
        for arm in expected_arms:
            for seed in range(3):
                if (setting, arm, seed) not in frames:
                    raise ValueError(f"missing fit {setting}/{arm}/seed{seed}")
    if any(key[0] == "exact_production" for key in frames):
        for arm in ("baseline", "all_fixed"):
            for seed in range(3):
                if ("exact_production", arm, seed) not in frames:
                    raise ValueError("incomplete production-hyperparameter comparison")
    check_pairs(frames)
    return frames


def main() -> None:
    """Require complete fitting jobs, then print reproducible score evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combined-only", action="store_true")
    parser.add_argument("--manual-scores", type=Path)
    parser.add_argument("--training-cache", type=Path)
    args = parser.parse_args()
    frames = read_scores(args.input, combined_only=args.combined_only)
    levels, contrasts = build_tables(frames)
    args.output.mkdir(parents=True, exist_ok=True)
    levels.write_parquet(args.output / "absolute.parquet")
    contrasts.write_parquet(args.output / "contrasts.parquet")
    lines = [
        "# Generated forecast-score evidence",
        "",
        (
            "Study errors in percentage points of effective capacity; lower is better. "
            "No leaderboard score."
        ),
        "",
        (
            "Main estimate: each series' row mean, then equal series weight, "
            "then the mean over three fitting seeds."
        ),
        (
            "Months use valid time. Bootstrap: 2,000 whole-month draws and one paired seed, "
            "recomputing series means from sums/counts."
        ),
        (
            "Month t-interval uses equally weighted monthly differences averaged over seeds. "
            "Its estimand differs from the main estimate."
        ),
        (
            "Leave-one-correction-out results are conditional contributions "
            "and cannot be added together."
        ),
        "",
        "## Feature columns for every arm",
        "",
    ]
    manifest = json.loads((args.input / "fit_manifest.json").read_text())
    for arm in sorted({key[1] for key in frames}):
        lines.extend(
            [
                f"{arm}: 24 columns (masked columns retained):",
                "`" + "`, `".join(manifest["features"]) + "`",
                "",
            ]
        )
    lines.extend(
        [
            "## Absolute errors",
            "",
            markdown(levels),
            "",
            "## Paired differences (treatment minus reference)",
            "",
            markdown(contrasts),
            "",
        ]
    )
    counts = [
        {"file": path.name, **json.loads(path.read_text())}
        for path in sorted((args.input / "prepared").glob("*.counts.json"))
    ]
    raw_cases = sum(row["rows"] for row in counts if not row["file"].startswith("train")) // 51
    population = population_evidence(frames, raw_cases)
    monthly = monthly_evidence(frames)
    population.write_parquet(args.output / "population.parquet")
    monthly.write_parquet(args.output / "monthly.parquet")
    lines.extend(
        [
            "## Population and pairing checks",
            "",
            (
                "PASS: unique keys, identical cases/targets/member counts across arms and seeds; "
                "finite losses."
            ),
            "Exactly member labels 0 through 50 were checked before scoring each run.",
            "",
            markdown(population),
            "",
            "## Monthly paired differences",
            "",
            markdown(monthly),
            "",
        ]
    )
    lines.extend(
        [
            "## Correction support and interval-energy diagnostics",
            "",
            markdown(pl.DataFrame(counts)),
            "",
        ]
    )
    if args.manual_scores:
        if not args.training_cache:
            raise ValueError("secondary reference requires --training-cache")
        lines.extend(
            [
                "## Secondary reference coverage",
                "",
                reference_coverage(
                    args.manual_scores, args.training_cache, frames[("controlled", "baseline", 0)]
                ),
                "",
            ]
        )
    (args.output / "report.md").write_text("\n".join(lines))
    print(levels.filter(pl.col("scope") == "All types, all leads"))
    print(
        contrasts.filter((pl.col("scope") == "All types, all leads") & (pl.col("metric") == "crps"))
    )


if __name__ == "__main__":
    main()
