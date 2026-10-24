"""Score the pure solar arms of experiment 313 against experiment 310's stored arms.

Rows are experiment 310's scored-year runs that hold finite quantiles in every arm compared here.
`pure_mdn` averages the five seeds' quantiles, `pure_blend` averages `pure_xgb` and `pure_mdn`
level by level with equal weights, and the plant-based arms are read from experiment 310's stored
per-row losses. Intervals resample the 12 months of the run with farm-equal weights at 98.75%.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring \
        uv run --no-sync python ml-experiments/00313-*/e00313_solar_score.py
"""

from typing import Final

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_score as s310  # ty: ignore[unresolved-import]
import e00313_common as common
import numpy as np
import polars as pl

PURE: Final[tuple[str, ...]] = ("pure_xgb", "pure_mdn", "pure_blend")
PLANT: Final[tuple[str, ...]] = ("p2", "gam", "mdn_ens", "xgb", "blend")
FLOORS: Final[tuple[str, ...]] = ("floor_climatology", "floor_heuristic")
PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("pure_mdn", "mdn_ens"),
    ("pure_xgb", "xgb"),
    ("pure_blend", "blend"),
    ("pure_mdn", "pure_xgb"),
    ("pure_blend", "pure_mdn"),
    ("pure_blend", "pure_xgb"),
    ("pure_xgb", "floor_climatology"),
    ("pure_mdn", "floor_climatology"),
    ("pure_blend", "floor_climatology"),
    ("pure_mdn", "floor_heuristic"),
    ("pure_mdn", "blend"),
    ("mdn_ens", "floor_climatology"),
    ("blend", "floor_climatology"),
)
SLICE_METRICS: Final[tuple[str, ...]] = (
    "score",
    "pin01",
    "pin99",
    "ae_median",
    "ae_mean",
    "median_minus_mean",
)


def load_pure(*, band: str, setting: str) -> pl.DataFrame:
    """Return the pure arms' quantile rows: seed-averaged mixture, XGBoost, and their blend."""
    files = sorted((common.SOLAR_PRED_DIR / band / setting).glob("*.parquet"))
    frame = pl.concat(
        [
            pl.read_parquet(f)
            .select(
                *s310.KEYS, "lead_hours", "arm", "seed", "setting", "band", "actual", *s310.QCOLS
            )
            .with_columns(pl.col(s310.QCOLS).cast(pl.Float64), pl.col("actual").cast(pl.Float64))
            for f in files
        ]
    ).with_columns(pl.col("lead_hours").cast(pl.Float64))
    mean = (
        frame.group_by(*s310.KEYS, "arm")
        .agg(
            pl.col("lead_hours").first(),
            pl.col("setting").first(),
            pl.col("band").first(),
            pl.col("actual").first(),
            pl.col(s310.QCOLS).mean(),
            n=pl.len(),
        )
        .filter(
            ((pl.col("arm") == "pure_mdn") & (pl.col("n") == 5))
            | ((pl.col("arm") == "pure_xgb") & (pl.col("n") == 1))
        )
        .drop("n")
    )
    xgb_rows = mean.filter(pl.col("arm") == "pure_xgb")
    mdn_rows = mean.filter(pl.col("arm") == "pure_mdn")
    joined = xgb_rows.join(mdn_rows, on=s310.KEYS, suffix="_m")
    blend = joined.select(
        *s310.KEYS,
        "lead_hours",
        "setting",
        "band",
        "actual",
        arm=pl.lit("pure_blend"),
        **{q: (pl.col(q) + pl.col(f"{q}_m")) / 2.0 for q in s310.QCOLS},
    )
    return pl.concat([mean, blend.select(mean.columns)])


def band_rows(*, band: str, plant: pl.DataFrame) -> pl.DataFrame:
    """Return per-row losses of every compared arm on the rows all of them hold."""
    pieces = []
    for setting in c310.SETTINGS:
        pure = s310.row_losses(
            frame=load_pure(band=band, setting=setting).filter(
                pl.all_horizontal([pl.col(c).is_finite() for c in s310.QCOLS])
            )
        )
        theirs = plant.filter(
            (pl.col("band") == band)
            & (pl.col("setting") == setting)
            & pl.col("arm").is_in([*PLANT, *FLOORS])
        ).select(pure.columns)
        both = pl.concat([pure, theirs])
        keep = (
            both.group_by(s310.KEYS)
            .agg(n=pl.col("arm").n_unique())
            .filter(pl.col("n") == len(PURE) + len(PLANT) + len(FLOORS))
            .select(s310.KEYS)
        )
        pieces.append(both.join(keep, on=s310.KEYS, how="inner"))
    return pl.concat(pieces).with_columns(
        month=pl.col("init_time").dt.strftime("%Y-%m"),
        farm=pl.col("time_series_id"),
    )


def main() -> None:
    """Write the score and contrast tables."""
    plant = pl.read_parquet(c310.DATA_DIR / "row_losses.parquet")
    scores, contrasts, counts = [], [], []
    for band in c310.BAND_ORDER:
        rows = band_rows(band=band, plant=plant)
        for setting in c310.SETTINGS:
            sub = rows.filter(pl.col("setting") == setting)
            n_rows = sub.group_by("arm").agg(n=pl.len())["n"].unique().to_list()
            counts.append(
                {
                    "band": band,
                    "setting": setting,
                    "rows_per_arm": n_rows[0] if len(n_rows) == 1 else -1,
                }
            )
            for metric in (*SLICE_METRICS, "cover01", "cover99"):
                sums, cnt, arms = s310.farm_sums(rows=sub, value=metric)
                point, draws = s310.boot(sums=sums, counts=cnt)
                idx = {a: i for i, a in enumerate(arms)}
                for a in arms:
                    lo, hi = np.nanpercentile(draws[idx[a]], [s310.LO_PCT, s310.HI_PCT])
                    scores.append(
                        {
                            "band": band,
                            "setting": setting,
                            "metric": metric,
                            "arm": a,
                            "value": point[idx[a]],
                            "lo": lo,
                            "hi": hi,
                        }
                    )
                for arm, ref in PAIRS:
                    if metric.startswith("cover") or metric == "median_minus_mean":
                        continue
                    change = 100.0 * (draws[idx[arm]] - draws[idx[ref]]) / draws[idx[ref]]
                    lo, hi = np.nanpercentile(change, [s310.LO_PCT, s310.HI_PCT])
                    contrasts.append(
                        {
                            "band": band,
                            "setting": setting,
                            "metric": metric,
                            "arm": arm,
                            "reference": ref,
                            "percent_change": 100.0
                            * (point[idx[arm]] - point[idx[ref]])
                            / point[idx[ref]],
                            "lo": lo,
                            "hi": hi,
                        }
                    )
        print("scored", band, flush=True)
    pl.DataFrame(scores).write_csv(common.DATA_DIR / "solar_scores.csv")
    pl.DataFrame(contrasts).write_csv(common.DATA_DIR / "solar_contrasts.csv")
    pl.DataFrame(counts).write_csv(common.DATA_DIR / "solar_row_counts.csv")
    pl.Config.set_tbl_rows(100)
    head = pl.DataFrame(scores).filter(pl.col("metric") == "score")
    for setting in c310.SETTINGS:
        print(f"--- {setting}: quantile score by band")
        print(
            head.filter(pl.col("setting") == setting)
            .pivot(on="arm", index="band", values="value")
            .sort("band")
        )


if __name__ == "__main__":
    main()
