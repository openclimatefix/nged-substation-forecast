"""Score the arms of experiment 224 and write the tables the write-up quotes.

The quantile score is twice the mean pinball loss over the 11 levels, in points of capacity (x100),
over the daylight rows that every arm holds. Intervals resample whole calendar months, the same
months for every farm, with 20,000 draws. The headline averages farms with equal weight.

    uv run --no-sync python ml-experiments/00224-*/e00224_score.py
"""

from typing import Final

import e00224_common as common
import numpy as np
import polars as pl
from studies.tail_calibration import TAIL_LEVELS, level_column, tail_calibration_table

DRAWS: Final[int] = 20_000
SEED: Final[int] = 224
LEVELS: Final[np.ndarray] = np.array(common.LEVELS)
QUANTILE_COLUMNS: Final[list[str]] = [level_column(level=lv) for lv in common.LEVELS]
MASSES: Final[np.ndarray] = np.diff(np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]]))
REFERENCE: Final[str] = "a2_mean"
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C0 members minus mean": ("a4_members", REFERENCE),
    "C1 two-number minus mean": ("a3_two_number", REFERENCE),
    "C2 two-number minus members": ("a3_two_number", "a4_members"),
    "C3 two-number minus XGBoost": ("a3_two_number", "a5_xgboost"),
    "C4 members minus rescale placebo": ("a4_members", "a2b_rescale"),
    "C5 two-number minus rescale placebo": ("a3_two_number", "a2b_rescale"),
    "C6 two-number minus strong shuffle": ("a3_two_number", "p3_strong"),
    "C7 members minus strong shuffle": ("a4_members", "p4_strong"),
    "C8 two-number minus manual heuristic": ("a3_two_number", "floor_heuristic"),
    "C9 two-number minus climatology": ("a3_two_number", "floor_climatology"),
}


def common_rows(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the rows of each (fold, band) that every arm holds with finite quantiles."""
    arms = frame["arm"].unique().to_list()
    ok = frame.filter(pl.all_horizontal([pl.col(c).is_finite() for c in QUANTILE_COLUMNS]))
    counts = ok.group_by("fold", "band", "time_series_id", "valid_time").agg(
        n_arms=pl.col("arm").n_unique()
    )
    keep = counts.filter(pl.col("n_arms") == len(arms)).select(
        "fold", "band", "time_series_id", "valid_time"
    )
    return frame.join(keep, on=["fold", "band", "time_series_id", "valid_time"], how="inner")


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
        month=pl.col("valid_time").dt.strftime("%Y-%m"),
    )


def cell_sums(
    *, rows: pl.DataFrame, value: str
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return sums[arm, farm, month], counts[farm, month], and the arm and month labels."""
    arms = sorted(rows["arm"].unique().to_list())
    farms = common.FARMS
    months = sorted(rows["month"].unique().to_list())
    sums = np.zeros((len(arms), len(farms), len(months)))
    counts = np.zeros((len(farms), len(months)))
    grouped = rows.group_by("arm", "time_series_id", "month").agg(
        total=pl.col(value).sum(), n=pl.len()
    )
    for r in grouped.iter_rows(named=True):
        a, f, m = arms.index(r["arm"]), farms.index(r["time_series_id"]), months.index(r["month"])
        sums[a, f, m] = r["total"]
        counts[f, m] = r["n"]
    return sums, counts, arms, months


def boot_scores(*, sums: np.ndarray, counts: np.ndarray, draws: int) -> np.ndarray:
    """Return farm-averaged scores per arm and draw, shape (arms, draws)."""
    rng = np.random.default_rng(SEED)
    n_months = counts.shape[1]
    weights = rng.multinomial(n_months, np.ones(n_months) / n_months, size=draws).astype(float)
    denominator = weights @ counts.T
    out = np.empty((sums.shape[0], draws))
    with np.errstate(invalid="ignore", divide="ignore"):
        for a in range(sums.shape[0]):
            out[a] = np.nanmean(
                (weights @ sums[a].T) / np.where(denominator > 0, denominator, np.nan), axis=1
            )
    return out


def interval(values: np.ndarray) -> tuple[float, float]:
    """Return the 2.5th and 97.5th percentiles."""
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def main() -> None:
    """Score both folds and every lead band, then print and write the tables."""
    scores_all, contrasts_all, tails_all, checks_all, farm_all = [], [], [], [], []
    for fold in common.FOLDS:
        frame = pl.read_parquet(common.ARMS_DIR / f"{fold}.parquet")
        frame = common_rows(frame=frame)
        for band in sorted(frame["band"].unique().to_list()):
            rows = per_row(frame=frame.filter(pl.col("band") == band))
            arms = sorted(rows["arm"].unique().to_list())
            sums, counts, arm_labels, months = cell_sums(rows=rows, value="score")
            farm_n = counts.sum(axis=1)
            point = np.nanmean(
                sums.sum(axis=2) / np.where(farm_n > 0, farm_n, np.nan)[None, :], axis=1
            )
            boot = boot_scores(sums=sums, counts=counts, draws=DRAWS)
            index = {a: i for i, a in enumerate(arm_labels)}
            for arm in arms:
                i = index[arm]
                lo, hi = interval(boot[i])
                scores_all.append(
                    {
                        "fold": fold,
                        "band": band,
                        "arm": arm,
                        "score": point[i],
                        "lo": lo,
                        "hi": hi,
                        "n_rows": int(counts.sum()),
                        "n_months": len(months),
                    }
                )
            reference = point[index[REFERENCE]]
            for name, (arm, other) in CONTRASTS.items():
                delta = 100.0 * (boot[index[arm]] - boot[index[other]]) / boot[index[other]]
                base = 100.0 * (point[index[arm]] - point[index[other]]) / point[index[other]]
                lo, hi = interval(delta)
                contrasts_all.append(
                    {
                        "fold": fold,
                        "band": band,
                        "contrast": name,
                        "percent_change": base,
                        "lo": lo,
                        "hi": hi,
                    }
                )
            c0 = (
                100.0
                * (boot[index["a4_members"]] - boot[index[REFERENCE]])
                / boot[index[REFERENCE]]
            )
            c1 = (
                100.0
                * (boot[index["a3_two_number"]] - boot[index[REFERENCE]])
                / boot[index[REFERENCE]]
            )
            c0p = 100.0 * (point[index["a4_members"]] - reference) / reference
            c1p = 100.0 * (point[index["a3_two_number"]] - reference) / reference
            share = c1 / np.where(np.abs(c0) > 1e-9, c0, np.nan)
            lo, hi = interval(share)
            contrasts_all.append(
                {
                    "fold": fold,
                    "band": band,
                    "contrast": "retained share C1/C0",
                    "percent_change": c1p / c0p,
                    "lo": lo,
                    "hi": hi,
                }
            )
            # Per farm.
            for farm in common.FARMS:
                sub = rows.filter(pl.col("time_series_id") == farm)
                if sub.is_empty():
                    continue
                table = (
                    sub.group_by("arm")
                    .agg(
                        score=pl.col("score").mean(),
                        mae_median=pl.col("abs_error_median").mean(),
                        mae_mean=pl.col("abs_error_mean").mean(),
                        median_minus_mean=pl.col("median_minus_mean").mean(),
                        n=pl.len(),
                    )
                    .with_columns(fold=pl.lit(fold), band=pl.lit(band), farm=pl.lit(farm))
                )
                farm_all.append(table)
            # Tail calibration pooled over farms (points of capacity).
            for arm in arms:
                sub = rows.filter(pl.col("arm") == arm).with_columns(
                    **{c: pl.col(c) * 100.0 for c in QUANTILE_COLUMNS},
                    actual=pl.col("actual") * 100.0,
                )
                table = tail_calibration_table(frame=sub, levels=TAIL_LEVELS).with_columns(
                    fold=pl.lit(fold), band=pl.lit(band), arm=pl.lit(arm)
                )
                tails_all.append(table)
            checks_all.append(
                rows.group_by("arm")
                .agg(
                    mae_median=pl.col("abs_error_median").mean(),
                    mae_mean=pl.col("abs_error_mean").mean(),
                )
                .with_columns(fold=pl.lit(fold), band=pl.lit(band))
            )
    out = common.DATA_DIR
    scores = pl.DataFrame(scores_all)
    contrasts = pl.DataFrame(contrasts_all)
    tails = pl.concat(tails_all)
    farms = pl.concat(farm_all)
    checks = pl.concat(checks_all)
    scores.write_csv(out / "scores.csv")
    contrasts.write_csv(out / "contrasts.csv")
    tails.write_csv(out / "tails.csv")
    farms.write_csv(out / "per_farm.csv")
    checks.write_csv(out / "checks.csv")
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=60):
        print(scores.sort("fold", "band", "score"))
        print(contrasts)


if __name__ == "__main__":
    main()
