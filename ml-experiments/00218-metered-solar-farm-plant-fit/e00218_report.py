"""Score experiment 218: fraction of output fitted, contrasts, slices, and the flat-top test.

Reads `predictions.parquet` from `e00218_fit.py` and prints (and writes under `data/`) every number
the write-up quotes. Farms appear only as labels A to F.
"""

from typing import Final

import e00218_common as common
import e00218_fit as fit
import numpy as np
import polars as pl
from studies.pv_dataset import pv_sites

ARMS: Final[tuple[str, ...]] = (
    "arm1_fleet_shape",
    "arm2_per_farm",
    "arm3_learned_response",
    "arm4_heat_off",
    "arm2_pooled_geometry",
    "ablation_no_clip",
    "ablation_fleet_geometry",
)
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1 arm2 - arm1": ("arm2_per_farm", "arm1_fleet_shape"),
    "C2 arm3 - arm2": ("arm3_learned_response", "arm2_per_farm"),
    "C3 arm2 - arm4": ("arm2_per_farm", "arm4_heat_off"),
    "C4 arm2 - pooled geometry": ("arm2_per_farm", "arm2_pooled_geometry"),
}
HOT_QUANTILE: Final[float] = 0.90
MIN_FLAT_TOP_HALF_HOURS: Final[int] = 30
OUT_DIR = common.DATA_DIR


def month_table(*, predictions: pl.DataFrame) -> pl.DataFrame:
    """Return per (fold, farm, arm, month): squared error, squared output, absolute error, count."""
    scored = predictions.filter(pl.col("scored"))
    return pl.concat(
        [
            scored.group_by("fold", "site", "month_index")
            .agg(
                sse=((pl.col("output") - pl.col(arm)) ** 2).sum(),
                ssq=(pl.col("output") ** 2).sum(),
                sae=(pl.col("output") - pl.col(arm)).abs().sum(),
                n=pl.len(),
            )
            .with_columns(arm=pl.lit(arm))
            for arm in ARMS
        ]
    )


def _dense(*, table: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return months and the arrays (arm, farm, month) of squared error and squared output.

    A farm without a row for a month has zero in both arrays, so it adds nothing to that month.
    """
    months = np.sort(table["month_index"].unique().to_numpy())
    farms = sorted(table["site"].unique().to_list())
    sse = np.zeros((len(ARMS), len(farms), len(months)))
    ssq = np.zeros_like(sse)
    for row in table.iter_rows(named=True):
        i, j = ARMS.index(row["arm"]), farms.index(row["site"])
        k = int(np.searchsorted(months, row["month_index"]))
        sse[i, j, k] = row["sse"]
        ssq[i, j, k] = row["ssq"]
    return months, sse, ssq


def median_fractions(*, sse: np.ndarray, ssq: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Return, per draw and arm, the median over farms of 1 - SSE/SSQ with months weighted.

    Args:
        sse: Squared errors, shape (arm, farm, month).
        ssq: Squared outputs, the same shape.
        counts: How often each month is drawn, shape (draw, month).

    Returns:
        Shape (draw, arm).
    """
    error = np.einsum("afm,dm->daf", sse, counts)
    scale = np.einsum("afm,dm->daf", ssq, counts)
    fractions = 1.0 - error / np.where(scale > 0, scale, np.nan)
    return np.nanmedian(fractions, axis=2)


def bootstrap(*, table: pl.DataFrame) -> pl.DataFrame:
    """Resample the scored months and return point values and intervals for arms and contrasts."""
    rng = np.random.default_rng(common.BOOTSTRAP_SEED)
    months, sse, ssq = _dense(table=table)
    draws = rng.integers(0, len(months), size=(common.BOOTSTRAP_DRAWS, len(months)))
    counts = np.stack([np.bincount(d, minlength=len(months)) for d in draws]).astype(np.float64)
    resampled = median_fractions(sse=sse, ssq=ssq, counts=counts)
    point = median_fractions(sse=sse, ssq=ssq, counts=np.ones((1, len(months))))[0]
    low, high = (1 - common.BOOTSTRAP_LEVEL) / 2, 1 - (1 - common.BOOTSTRAP_LEVEL) / 2
    rows = [
        {
            "name": arm,
            "point": float(point[i]),
            "low": float(np.quantile(resampled[:, i], low)),
            "high": float(np.quantile(resampled[:, i], high)),
        }
        for i, arm in enumerate(ARMS)
    ]
    for name, (plus, minus) in CONTRASTS.items():
        i, j = ARMS.index(plus), ARMS.index(minus)
        delta = resampled[:, i] - resampled[:, j]
        rows.append(
            {
                "name": name,
                "point": float(point[i] - point[j]),
                "low": float(np.quantile(delta, low)),
                "high": float(np.quantile(delta, high)),
            }
        )
    return pl.DataFrame(rows)


def per_farm(*, predictions: pl.DataFrame, capacity: dict[str, float]) -> pl.DataFrame:
    """Return the fraction of output fitted, MAE in MW, and MAE in points of capacity per farm."""
    scored = predictions.filter(pl.col("scored"))
    rows = []
    for arm in ARMS:
        agg = scored.group_by("fold", "site").agg(
            fraction=1
            - ((pl.col("output") - pl.col(arm)) ** 2).sum() / (pl.col("output") ** 2).sum(),
            mae_capacity_points=(pl.col("output") - pl.col(arm)).abs().mean() * 100,
        )
        rows.append(agg.with_columns(arm=pl.lit(arm)))
    table = pl.concat(rows).with_columns(
        mae_mw=pl.col("mae_capacity_points") / 100 * pl.col("site").replace_strict(capacity)
    )
    return table.sort("fold", "arm", "site")


def slices(*, predictions: pl.DataFrame) -> pl.DataFrame:
    """Return the median-over-farms fraction on sunny, hot, and shoulder-month half-hours."""
    scored = predictions.filter(pl.col("scored") & (pl.col("fold") == 1))
    hot_cut = predictions.group_by("site").agg(
        pl.col("temperature_c").quantile(HOT_QUANTILE).alias("hot")
    )
    scored = scored.join(hot_cut, on="site")
    masks = {
        "sunny": pl.col("clear_sky") > common.FLAT_TOP_CLEAR_SKY_W_M2,
        "hot": pl.col("temperature_c") > pl.col("hot"),
        "shoulder": pl.col("month").is_in(common.SHOULDER_MONTHS),
        "all": pl.lit(value=True),
    }
    rows = []
    for name, mask in masks.items():
        for arm in ARMS:
            by_farm = (
                scored.filter(mask)
                .group_by("site")
                .agg(
                    fraction=1
                    - ((pl.col("output") - pl.col(arm)) ** 2).sum() / (pl.col("output") ** 2).sum()
                )
            )
            rows.append(
                {
                    "slice": name,
                    "arm": arm,
                    "median_fraction": float(np.median(by_farm["fraction"].to_numpy())),
                }
            )
    return pl.DataFrame(rows)


def flat_top(*, predictions: pl.DataFrame, scored_only: bool = False) -> pl.DataFrame:
    """Return the seasonal range of the 90th and 99th percentile flat level, metered, modelled."""
    frame = predictions.filter(
        (pl.col("fold") == 1)
        & (pl.col("clear_sky") > common.FLAT_TOP_CLEAR_SKY_W_M2)
        & pl.col("month").is_in(common.FLAT_TOP_MONTHS)
        & pl.col("site").is_in(common.FLAT_TOP_FARMS)
        & (pl.col("scored") if scored_only else pl.lit(value=True))
    )
    rows = []
    for quantile in (0.90, 0.99):
        for series in ("output", *ARMS):
            levels = (
                frame.group_by("site", "month")
                .agg(level=pl.col(series).quantile(quantile), n=pl.len())
                .filter(pl.col("n") >= MIN_FLAT_TOP_HALF_HOURS)
            )
            for site, part in levels.group_by("site"):
                values = part["level"].to_numpy()
                rows.append(
                    {
                        "quantile": quantile,
                        "series": series,
                        "site": site[0],
                        "range_over_mean": float((values.max() - values.min()) / values.mean()),
                        "months": part.height,
                    }
                )
    table = pl.DataFrame(rows)
    observed = table.filter(pl.col("series") == "output").select(
        "quantile", "site", observed=pl.col("range_over_mean")
    )
    return (
        table.join(observed, on=["quantile", "site"])
        .with_columns(gap=(pl.col("range_over_mean") - pl.col("observed")).abs())
        .sort("quantile", "series", "site")
    )


def main() -> None:
    """Print and write every table the write-up quotes."""
    predictions = pl.read_parquet(fit.PREDICTIONS_PATH)
    sites = pv_sites()
    capacity = dict(sites.select("site", "effective_capacity_mw").iter_rows())
    farms = per_farm(predictions=predictions, capacity=capacity)
    farms.write_csv(OUT_DIR / "per_farm_scores.csv")
    print(
        farms.filter(pl.col("fold") == 1)
        .pivot(on="arm", index="site", values="fraction")
        .sort("site")
    )
    print(
        farms.filter(pl.col("fold") == 1)
        .pivot(on="arm", index="site", values="mae_capacity_points")
        .sort("site")
    )
    table = month_table(predictions=predictions)
    intervals = []
    for fold in sorted(common.FOLD_SCORED_STARTS):
        result = bootstrap(table=table.filter(pl.col("fold") == fold)).with_columns(
            fold=pl.lit(fold)
        )
        intervals.append(result)
        print(f"fold {fold}")
        print(result.select("name", "point", "low", "high"))
    pl.concat(intervals).write_csv(OUT_DIR / "intervals.csv")
    wins = (
        table.filter(pl.col("fold") == 1)
        .pivot(on="arm", index=["site", "month_index"], values="sse")
        .with_columns(better=pl.col("arm2_per_farm") < pl.col("arm1_fleet_shape"))
    )
    print("farm-months where arm 2 beats arm 1 (fold 1):", wins["better"].sum(), "of", wins.height)
    slice_table = slices(predictions=predictions)
    slice_table.write_csv(OUT_DIR / "slices.csv")
    print(slice_table.pivot(on="arm", index="slice", values="median_fraction"))
    flat = flat_top(predictions=predictions)
    flat.write_csv(OUT_DIR / "flat_top.csv")
    print(
        flat.filter(pl.col("quantile") == 0.90).pivot(
            on="site", index="series", values="range_over_mean"
        )
    )
    print(
        flat.filter(pl.col("series") != "output")
        .group_by("quantile", "series")
        .agg(mean_gap=pl.col("gap").mean())
        .sort("quantile", "series")
    )
    scored_flat = flat_top(predictions=predictions, scored_only=True)
    scored_flat.write_csv(OUT_DIR / "flat_top_scored_year.csv")
    print("flat top, scored year only: months per farm and mean gap")
    print(
        scored_flat.filter(pl.col("series") != "output")
        .group_by("quantile", "series")
        .agg(mean_gap=pl.col("gap").mean(), months=pl.col("months").min())
        .sort("quantile", "series")
    )
    parameters = pl.read_parquet(fit.PARAMETERS_PATH)
    print(
        parameters.filter(pl.col("arm").is_in(["arm2_per_farm", "arm4_heat_off"])).sort(
            "fold", "arm", "site"
        )
    )


if __name__ == "__main__":
    main()
