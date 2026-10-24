"""Score the pure wind arms of experiment 313 beside experiment 276's and 285's stored arms.

Rows are experiment 285's scored cases (`fold_a_tuesdays_v1`, 53 Tuesday runs, W1 to W3). Each arm
is reduced to one per-row metric array: the seed-averaged mixture quantiles for `pure_mdn` (five
seeds) and the mean over the three seeds of the per-row metric for every other arm. Intervals
resample whole months of the run (farms are not resampled) at 98.75%, a Bonferroni level over the
four planned contrasts. Ensemble mixtures get experiment 285's spread factor.

    PYTHONPATH=ml-experiments/00285-wind-effective-wind-adapters \
        uv run --no-sync python ml-experiments/00313-*/e00313_wind_score.py
"""

from typing import Final

import e00285_common as c285  # ty: ignore[unresolved-import]
import e00285_models as m285  # ty: ignore[unresolved-import]
import e00285_score as s285  # ty: ignore[unresolved-import]
import e00313_common as common
import numpy as np
import polars as pl

RESAMPLES: Final[int] = 20_000
LOW, HIGH = 0.625, 99.375
FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
HISTORIES: Final[tuple[str, ...]] = ("own_history", "zero_history")
WEATHER: Final[tuple[str, ...]] = ("era5", "ens")
ROW_SETS: Final[tuple[str, ...]] = ("kept", "lead_12_36", "all")
METRICS: Final[tuple[str, ...]] = ("qs", "p01", "p99", "mean_abs_error", "median_minus_mean")
COMPARATORS: Final[dict[str, str]] = {"core": "P", "xgb_binned_margin": "X", "neural_adapter": "A4"}
PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("pure_mdn", "pure_xgb"),
    ("pure_xgb", "xgb_binned_margin"),
    ("pure_mdn", "core"),
    ("pure_mdn", "neural_adapter"),
    ("pure_xgb", "core"),
)


def load_pure(
    *, arm: str, history: str, weather: str, seeds: tuple[int, ...], cases: pl.DataFrame
) -> np.ndarray:
    """Return (seeds, rows, 11) quantiles of a pure arm, farms stacked in case order."""
    out = []
    for seed in seeds:
        values = np.concatenate(
            [
                np.load(common.WIND_DIR / f"{arm}__{history}__{farm}__s{seed}__{weather}.npy")
                for farm in FARMS
            ]
        )
        out.append(values.astype(np.float64))
    return np.stack(out)


def spread(*, quantiles: np.ndarray, weather: str, cases: pl.DataFrame) -> np.ndarray:
    """Apply experiment 285's spread factor to ensemble mixtures; ERA5 rows are unchanged."""
    if weather == "era5":
        return quantiles
    return np.stack([c285_apply(cases=cases, mixture=q) for q in quantiles])


def c285_apply(*, cases: pl.DataFrame, mixture: np.ndarray) -> np.ndarray:
    """Return the mixture after the spread factor."""
    return c285.apply_spread(cases=cases, mixture=mixture)


def reduce_metrics(*, metrics: dict[str, np.ndarray], average_seeds: bool) -> dict[str, np.ndarray]:
    """Return metric arrays of shape (1, rows)."""
    return {k: v.mean(axis=0, keepdims=True) if average_seeds else v for k, v in metrics.items()}


def interval(
    *, values: np.ndarray, keep: np.ndarray, month: np.ndarray
) -> tuple[float, float, float]:
    """Return the mean over `keep` rows and a month-resampled 98.75% interval."""
    months = np.unique(month[keep])
    sums = np.array([values[0, keep & (month == m)].sum() for m in months])
    counts = np.array([(keep & (month == m)).sum() for m in months], dtype=float)
    rng = np.random.default_rng(313)
    draw = rng.integers(0, months.size, size=(RESAMPLES, months.size))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    low, high = np.percentile(means, [LOW, HIGH])
    return float(sums.sum() / counts.sum()), float(low), float(high)


def main() -> None:  # noqa: PLR0912
    """Write the wind score and contrast tables."""
    cases = s285.ordered_cases()
    specs = m285.build_specs()
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    month = cases["month"].to_numpy()
    flags = {name: cases[name].to_numpy() for name in ROW_SETS}
    site = cases["site"].to_numpy()
    arms: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for weather in WEATHER:
        for history in HISTORIES:
            for arm in ("pure_xgb", "pure_mdn"):
                seeds = (0, 1, 2, 3, 4) if arm == "pure_mdn" else (0, 1, 2)
                raw = load_pure(arm=arm, history=history, weather=weather, seeds=seeds, cases=cases)
                if arm == "pure_mdn":
                    raw = raw.mean(axis=0, keepdims=True)
                quant = spread(quantiles=raw, weather=weather, cases=cases)
                arms[(arm, history, weather)] = reduce_metrics(
                    metrics=s285.row_metrics(quantiles=quant, actual=actual), average_seeds=True
                )
        for name, spec_name in COMPARATORS.items():
            q = s285.load_quantiles(spec=specs[spec_name], setting=weather, cases=cases)
            if q is not None:
                arms[(name, "own_history", weather)] = reduce_metrics(
                    metrics=s285.row_metrics(quantiles=q, actual=actual), average_seeds=True
                )
    scores, contrasts = [], []
    for (arm, history, weather), metrics in arms.items():
        for row_set, keep in flags.items():
            for metric in METRICS:
                value, low, high = interval(values=metrics[metric], keep=keep, month=month)
                scores.append(
                    {
                        "arm": arm,
                        "history": history,
                        "weather": weather,
                        "row_set": row_set,
                        "metric": metric,
                        "value": value,
                        "lo": low,
                        "hi": high,
                    }
                )
            for farm in FARMS:
                value, low, high = interval(
                    values=metrics["qs"], keep=keep & (site == farm), month=month
                )
                scores.append(
                    {
                        "arm": arm,
                        "history": history,
                        "weather": weather,
                        "row_set": row_set,
                        "metric": f"qs_{farm}",
                        "value": value,
                        "lo": low,
                        "hi": high,
                    }
                )
    for weather in WEATHER:
        for history in HISTORIES:
            for first, second in PAIRS:
                if history == "zero_history" and (first, second) != ("pure_mdn", "pure_xgb"):
                    continue
                a = (first, history, weather)
                b = (second, history if second.startswith("pure") else "own_history", weather)
                if a not in arms or b not in arms:
                    continue
                for row_set, keep in flags.items():
                    for metric in METRICS[:4]:
                        diff = arms[a][metric] - arms[b][metric]
                        base = float(arms[b][metric][0, keep].mean())
                        value, low, high = interval(values=diff, keep=keep, month=month)
                        contrasts.append(
                            {
                                "arm": first,
                                "reference": second,
                                "history": history,
                                "weather": weather,
                                "row_set": row_set,
                                "metric": metric,
                                "difference": value,
                                "lo": low,
                                "hi": high,
                                "percent": 100 * value / base,
                            }
                        )
    pl.DataFrame(scores).write_csv(common.DATA_DIR / "wind_scores.csv")
    pl.DataFrame(contrasts).write_csv(common.DATA_DIR / "wind_contrasts.csv")
    pl.Config.set_tbl_rows(80)
    table = pl.DataFrame(scores).filter((pl.col("metric") == "qs") & (pl.col("row_set") == "kept"))
    print(
        table.select("arm", "history", "weather", "value", "lo", "hi").sort(
            "weather", "history", "arm"
        )
    )


if __name__ == "__main__":
    main()
