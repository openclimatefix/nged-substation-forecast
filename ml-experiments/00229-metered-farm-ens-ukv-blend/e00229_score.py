"""Score the arms of experiment 229 and write the tables the write-up quotes.

The quantile score is twice the mean pinball loss over the 11 levels, in points of capacity (x100),
over the daylight rows that every arm holds. Intervals resample whole calendar months (the 10 months
of both folds together), the same months for every farm, with 20,000 draws, at the 99.17% level that
a Bonferroni correction over the six planned contrasts sets. The headline contrast for each pair is
the median over the six farms of the per-farm percentage change.

    uv run --no-sync python ml-experiments/00229-*/e00229_score.py
"""

from typing import Final

import e00229_common as common
import numpy as np
import polars as pl
from studies.tail_calibration import TAIL_LEVELS, level_column, tail_calibration_table

DRAWS: Final[int] = 20_000
SEED: Final[int] = 229
LEVELS: Final[np.ndarray] = np.array(common.LEVELS)
QUANTILE_COLUMNS: Final[list[str]] = [level_column(level=lv) for lv in common.LEVELS]
MASSES: Final[np.ndarray] = np.diff(np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]]))
LOW, HIGH = 100 * (0.0083 / 2), 100 * (1 - 0.0083 / 2)
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1a blend (A3) minus ensemble two-number (A1)": ("a3_blend", "a1_ens_two_number"),
    "C1b tree with all UKV (A4) minus tree on ensemble (A1x)": ("a4_tree_all_ukv", "a1x_tree_ens"),
    "C1c tree with UKV sky only (A5) minus tree on ensemble (A1x)": (
        "a5_tree_ukv_sky",
        "a1x_tree_ens",
    ),
    "C2 sky only (A5) minus all variables (A4)": ("a5_tree_ukv_sky", "a4_tree_all_ukv"),
    "C3 blend minus its shuffled-UKV placebo": ("a3_blend", "p3_blend_shuffle"),
    "C4 tree with all UKV minus its shuffled-UKV placebo": ("a4_tree_all_ukv", "p4_tree_shuffle"),
    "X1 UKV alone (A2) minus ensemble two-number (A1)": ("a2_ukv_alone", "a1_ens_two_number"),
    "X2 blend (A3) minus tree on ensemble (A1x)": ("a3_blend", "a1x_tree_ens"),
    "X3 blend minus manual heuristic": ("a3_blend", "floor_heuristic"),
    "X4 best tree (A4) minus manual heuristic": ("a4_tree_all_ukv", "floor_heuristic"),
}
SUNNY_REFERENCE_OBSERVED: Final[float] = 0.973
"""Experiment 218's fraction fitted on sunny half-hours with observed weather (other rows)."""


def common_rows(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the rows of each band that every arm holds with finite quantiles."""
    arms = frame["arm"].unique().to_list()
    ok = frame.filter(pl.all_horizontal([pl.col(c).is_finite() for c in QUANTILE_COLUMNS]))
    keys = ["fold", "band", "time_series_id", "valid_time"]
    counts = ok.group_by(keys).agg(n_arms=pl.col("arm").n_unique())
    keep = counts.filter(pl.col("n_arms") == len(arms)).select(keys)
    return frame.join(keep, on=keys, how="inner")


def per_row(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the quantile score, the median's absolute error, and the mean forecast's error."""
    q = frame.select(QUANTILE_COLUMNS).to_numpy()
    actual = frame["actual"].to_numpy()
    error = actual[:, None] - q
    pin = np.maximum(LEVELS * error, (LEVELS - 1.0) * error)
    mean_forecast = q @ MASSES
    return frame.with_columns(
        score=pl.Series(200.0 * pin.mean(axis=1)),
        abs_error_median=pl.Series(100.0 * np.abs(actual - q[:, 5])),
        abs_error_mean=pl.Series(100.0 * np.abs(actual - mean_forecast)),
        median_minus_mean=pl.Series(100.0 * (q[:, 5] - mean_forecast)),
        sq_error_mean=pl.Series((actual - mean_forecast) ** 2),
        month=pl.col("valid_time").dt.strftime("%Y-%m"),
    )


def cell_sums(
    *, rows: pl.DataFrame, value: str
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return sums[arm, farm, month], counts[farm, month], and the arm and month labels."""
    arms = sorted(rows["arm"].unique().to_list())
    months = sorted(rows["month"].unique().to_list())
    sums = np.zeros((len(arms), len(common.FARMS), len(months)))
    counts = np.zeros((len(common.FARMS), len(months)))
    grouped = rows.group_by("arm", "time_series_id", "month").agg(
        total=pl.col(value).sum(), n=pl.len()
    )
    for r in grouped.iter_rows(named=True):
        a, f, m = (
            arms.index(r["arm"]),
            common.FARMS.index(r["time_series_id"]),
            months.index(r["month"]),
        )
        sums[a, f, m] = r["total"]
        counts[f, m] = r["n"]
    return sums, counts, arms, months


def month_weights(*, n_months: int) -> np.ndarray:
    """Return multinomial month-resampling weights, shape (draws, months)."""
    rng = np.random.default_rng(SEED)
    return rng.multinomial(n_months, np.ones(n_months) / n_months, size=DRAWS).astype(float)


def farm_scores(*, sums: np.ndarray, counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return per-farm scores per arm and draw, shape (arms, draws, farms)."""
    denominator = weights @ counts.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack(
            [
                (weights @ sums[a].T) / np.where(denominator > 0, denominator, np.nan)
                for a in range(sums.shape[0])
            ]
        )


def main() -> None:
    """Score both folds together at each lead band, then write the tables."""
    frames = [pl.read_parquet(common.ARMS_DIR / f"{fold}.parquet") for fold in common.FOLDS]
    frame = common_rows(frame=pl.concat(frames, how="diagonal"))
    scores_all, contrasts_all, tails_all, checks_all, farm_all, sunny_all, fold_all = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for band in sorted(frame["band"].unique().to_list()):
        rows = per_row(frame=frame.filter(pl.col("band") == band))
        sums, counts, arms, months = cell_sums(rows=rows, value="score")
        weights = month_weights(n_months=len(months))
        boot = farm_scores(sums=sums, counts=counts, weights=weights)
        farm_n = counts.sum(axis=1)
        point = sums.sum(axis=2) / np.where(farm_n > 0, farm_n, np.nan)[None, :]
        index = {a: i for i, a in enumerate(arms)}
        for arm in arms:
            i = index[arm]
            boot_mean = np.nanmean(boot[i], axis=1)
            scores_all.append(
                {
                    "band": band,
                    "arm": arm,
                    "score": float(np.nanmean(point[i])),
                    "lo": float(np.nanpercentile(boot_mean, 2.5)),
                    "hi": float(np.nanpercentile(boot_mean, 97.5)),
                    "n_rows": int(counts.sum()),
                    "n_months": len(months),
                }
            )
        for name, (arm, other) in CONTRASTS.items():
            ratio = 100.0 * (boot[index[arm]] / boot[index[other]] - 1.0)  # (draws, farms)
            median = np.nanmedian(ratio, axis=1)
            base = np.nanmedian(100.0 * (point[index[arm]] / point[index[other]] - 1.0))
            contrasts_all.append(
                {
                    "band": band,
                    "contrast": name,
                    "farm": "median",
                    "percent_change": float(base),
                    "lo": float(np.nanpercentile(median, LOW)),
                    "hi": float(np.nanpercentile(median, HIGH)),
                }
            )
            for k, farm in enumerate(common.FARMS):
                contrasts_all.append(
                    {
                        "band": band,
                        "contrast": name,
                        "farm": farm,
                        "percent_change": float(
                            100.0 * (point[index[arm], k] / point[index[other], k] - 1.0)
                        ),
                        "lo": float(np.nanpercentile(ratio[:, k], LOW)),
                        "hi": float(np.nanpercentile(ratio[:, k], HIGH)),
                    }
                )
        # Fraction of output fitted on sunny half-hours (mean forecast against metered output).
        sunny = rows.filter(pl.col("clear_sky") > common.SUNNY_CLEAR_SKY_W_M2)
        table = (
            sunny.group_by("arm", "time_series_id")
            .agg(sse=pl.col("sq_error_mean").sum(), total=(pl.col("actual") ** 2).sum(), n=pl.len())
            .with_columns(fraction_fitted=1.0 - pl.col("sse") / pl.col("total"))
        )
        sunny_all.append(
            table.group_by("arm")
            .agg(fraction_fitted_median=pl.col("fraction_fitted").median(), n=pl.col("n").sum())
            .with_columns(band=pl.lit(band))
        )
        for farm in common.FARMS:
            sub = rows.filter(pl.col("time_series_id") == farm)
            farm_all.append(
                sub.group_by("arm")
                .agg(
                    score=pl.col("score").mean(),
                    mae_median=pl.col("abs_error_median").mean(),
                    mae_mean=pl.col("abs_error_mean").mean(),
                    median_minus_mean=pl.col("median_minus_mean").mean(),
                    n=pl.len(),
                )
                .with_columns(band=pl.lit(band), farm=pl.lit(farm))
            )
        for arm in arms:
            sub = rows.filter(pl.col("arm") == arm).with_columns(
                **{c: pl.col(c) * 100.0 for c in QUANTILE_COLUMNS}, actual=pl.col("actual") * 100.0
            )
            tails_all.append(
                tail_calibration_table(frame=sub, levels=TAIL_LEVELS).with_columns(
                    band=pl.lit(band), arm=pl.lit(arm)
                )
            )
        checks_all.append(
            rows.group_by("arm")
            .agg(
                mae_median=pl.col("abs_error_median").mean(),
                mae_mean=pl.col("abs_error_mean").mean(),
            )
            .with_columns(band=pl.lit(band))
        )
        # Fold-wise scores for the replicate and scored periods.
        for fold in common.FOLDS:
            by = (
                rows.filter(pl.col("fold") == fold)
                .group_by("arm", "time_series_id")
                .agg(score=pl.col("score").mean())
            )
            fold_all.append(
                by.group_by("arm")
                .agg(score=pl.col("score").mean())
                .with_columns(band=pl.lit(band), fold=pl.lit(fold))
            )
    out = common.DATA_DIR
    pl.DataFrame(scores_all).write_csv(out / "scores.csv")
    contrasts = pl.DataFrame(contrasts_all)
    contrasts.write_csv(out / "contrasts.csv")
    pl.concat(tails_all).write_csv(out / "tails.csv")
    pl.concat(farm_all).write_csv(out / "per_farm.csv")
    pl.concat(checks_all).write_csv(out / "checks.csv")
    pl.concat(sunny_all).write_csv(out / "sunny.csv")
    pl.concat(fold_all).write_csv(out / "fold_scores.csv")
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=70):
        print(pl.DataFrame(scores_all).sort("band", "score"))
        print(contrasts.filter(pl.col("farm") == "median"))
        print(pl.concat(sunny_all).sort("band", "fraction_fitted_median"))


if __name__ == "__main__":
    main()
