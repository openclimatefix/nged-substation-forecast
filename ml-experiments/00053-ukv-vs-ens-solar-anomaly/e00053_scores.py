"""Score the four arms against the CAMS anomaly on experiment 00046's validation rows.

Rows are experiment 00046's scored rows: the five verdict series, daylight half-hours, local months
April to September of the validation year (valid times from 2025-07-01 to 2026-06-30), with a CAMS
`u` and a finite observed power. Per variant and lead band (lead from the forecast origin) it prints
and writes the Pearson correlation of each arm's calibrated anomaly with the CAMS anomaly, the
calibration slope (slope through the origin of the CAMS anomaly on the calibrated anomaly; ideal
1), and the mean squared error. Intervals are 95% and resample whole calendar months of the
forecast origin (50,000 resamples). Also writes the paired differences in correlation and the
hour-of-day profile check.

    uv run python ml-experiments/00053-*/e00053_scores.py
"""

from typing import Final

import e00053_common as common
import numpy as np
import polars as pl

ARMS: Final[tuple[str, ...]] = ("ens", "ukv", "blend", "ukv3h", "ukv3w")
N_RESAMPLES: Final[int] = 50_000
SEED: Final[int] = 20261016
COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    ("ukv", "ens"),
    ("blend", "ens"),
    ("ukv3h", "ukv"),
    ("ukv3w", "ukv"),
    ("ukv3w", "ens"),
)
"""(arm, reference) pairs whose difference in correlation is read."""
POOLS: Final[tuple[tuple[str, tuple[int, ...]], ...]] = (
    ("0-54 h", (0, 1, 2, 3, 4)),
    ("0-24 h", (0, 1, 2)),
)


def scored_flags() -> pl.DataFrame:
    """Return experiment 00046's scored keys: rows whose observed power is finite.

    Returns:
        Columns `site`, `power_fcst_init_time`, `valid_time`.
    """
    series = common.series_table().select("time_series_id", "site")
    arms_path = common.E46_WORK / "arms" / "d0.parquet"
    return (
        pl.read_parquet(
            arms_path, columns=["time_series_id", "power_fcst_init_time", "valid_time", "actual"]
        )
        .filter(pl.col("actual").is_finite())
        .unique(["time_series_id", "power_fcst_init_time", "valid_time"])
        .join(series, on="time_series_id")
        .select(*common.KEYS)
    )


def scored_rows() -> pl.DataFrame:
    """Return the validation anomalies restricted to experiment 00046's scored rows.

    Returns:
        The rows of `validation_anomalies.parquet` that are scored, with `origin_block`.
    """
    anomalies = pl.read_parquet(common.WORK_DIR / "validation_anomalies.parquet")
    return (
        anomalies.join(scored_flags(), on=list(common.KEYS), how="semi")
        .filter(
            pl.col("site").is_in(common.VERDICT_SITES)
            & pl.col("month").is_in(list(common.SCORED_MONTHS))
            & (pl.col("valid_time") >= common.VALIDATION_START)
            & (pl.col("valid_time") < common.HOLDOUT)
        )
        .with_columns(
            origin_block=(
                pl.col("power_fcst_init_time").dt.year() * 100
                + pl.col("power_fcst_init_time").dt.month()
            )
        )
    )


def block_statistics(*, rows: pl.DataFrame, arm: str) -> pl.DataFrame:
    """Return per-origin-month sums for correlation, slope, and squared error of one arm.

    Args:
        rows: Scored rows of one variant and band set.
        arm: The arm.

    Returns:
        One row per `origin_block`: `n`, `sx`, `sy`, `sxx`, `syy`, `sxy`, `sse`.
    """
    x, y = pl.col(f"c_{arm}"), pl.col("a_cams")
    return (
        rows.group_by("origin_block")
        .agg(
            n=pl.len(),
            sx=x.sum(),
            sy=y.sum(),
            sxx=(x * x).sum(),
            syy=(y * y).sum(),
            sxy=(x * y).sum(),
            sse=((y - x) ** 2).sum(),
        )
        .sort("origin_block")
    )


def metrics(*, totals: np.ndarray) -> np.ndarray:
    """Return correlation, calibration slope, and mean squared error from summed statistics.

    Args:
        totals: Array (..., 7) of `n, sx, sy, sxx, syy, sxy, sse` summed over the blocks.

    Returns:
        Array (..., 3) of correlation, slope (through the origin), mean squared error.
    """
    n, sx, sy, sxx, syy, sxy, sse = np.moveaxis(totals, -1, 0)
    covariance = sxy / n - sx * sy / n**2
    variance_x = sxx / n - (sx / n) ** 2
    variance_y = syy / n - (sy / n) ** 2
    return np.stack([covariance / np.sqrt(variance_x * variance_y), sxy / sxx, sse / n], axis=-1)


def score_group(*, rows: pl.DataFrame, rng: np.random.Generator) -> dict[str, dict]:
    """Return every arm's metrics with intervals, and the paired correlation differences.

    Args:
        rows: Scored rows of one variant and one set of bands.
        rng: The random generator.

    Returns:
        `{"arms": {arm: {...}}, "differences": {"a - b": {...}}}`.
    """
    stats = {
        arm: block_statistics(rows=rows, arm=arm).drop("origin_block").to_numpy() for arm in ARMS
    }
    blocks = next(iter(stats.values())).shape[0]
    picks = rng.integers(0, blocks, size=(N_RESAMPLES, blocks))
    counts = np.stack([np.bincount(row, minlength=blocks) for row in picks])
    result: dict[str, dict] = {"arms": {}, "differences": {}}
    resampled = {}
    for arm in ARMS:
        point = metrics(totals=stats[arm].sum(axis=0))
        boot = metrics(totals=counts @ stats[arm])
        resampled[arm] = boot
        low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        result["arms"][arm] = {
            "n": int(stats[arm][:, 0].sum()),
            "correlation": float(point[0]),
            "correlation_low": float(low[0]),
            "correlation_high": float(high[0]),
            "slope": float(point[1]),
            "slope_low": float(low[1]),
            "slope_high": float(high[1]),
            "mse": float(point[2]),
            "mse_low": float(low[2]),
            "mse_high": float(high[2]),
        }
    for arm, reference in COMPARISONS:
        point = result["arms"][arm]["correlation"] - result["arms"][reference]["correlation"]
        boot = resampled[arm][:, 0] - resampled[reference][:, 0]
        low, high = np.nanpercentile(boot, [2.5, 97.5])
        mse_point = result["arms"][arm]["mse"] / result["arms"][reference]["mse"] - 1.0
        mse_boot = resampled[arm][:, 2] / resampled[reference][:, 2] - 1.0
        mse_low, mse_high = np.nanpercentile(mse_boot, [2.5, 97.5])
        result["differences"][f"{arm} - {reference}"] = {
            "correlation_difference": float(point),
            "low": float(low),
            "high": float(high),
            "mse_relative": float(mse_point),
            "mse_relative_low": float(mse_low),
            "mse_relative_high": float(mse_high),
        }
    return result


def profile_check(*, validation: pl.DataFrame) -> pl.DataFrame:
    """Return the mean `u` by UTC hour for CAMS, ENS, UKV, and 3-hourly UKV at leads under 24 h.

    Args:
        validation: Scored rows of the `t54` variant.

    Returns:
        Columns `hour`, `n`, `cams`, `ens`, `ukv`, `ukv3h`, `ukv3w`.
    """
    return (
        validation.filter(pl.col("lead_hours") <= 24)
        .group_by(hour=pl.col("valid_time").dt.hour())
        .agg(
            n=pl.len(),
            cams=pl.col("u_cams").mean(),
            ens=pl.col("u_ens").mean(),
            ukv=pl.col("u_ukv").mean(),
            ukv3h=pl.col("u_ukv3h").mean(),
            ukv3w=pl.col("u_ukv3w").mean(),
        )
        .sort("hour")
    )


def main() -> None:
    """Write the scores and print them."""
    rng = np.random.default_rng(SEED)
    rows = scored_rows()
    flat = []
    for variant in ("t54", "t54_late", "t120"):
        own = rows.filter(pl.col("variant") == variant)
        groups = [(name, [band]) for band, name in enumerate(common.BAND_NAMES)]
        groups += [(name, list(bands)) for name, bands in POOLS]
        if variant == "t120":
            groups.append(("0-120 h", [0, 1, 2, 3, 4, 5]))
        for name, bands in groups:
            part = own.filter(pl.col("band").is_in(bands))
            if part.height < 200:
                continue
            scored = score_group(rows=part, rng=rng)
            for arm, values in scored["arms"].items():
                flat.append({"variant": variant, "band": name, "arm": arm, **values})
    table = pl.DataFrame(flat)
    common.write_atomically(frame=table, target=common.RESULTS_DIR / "scores.parquet")
    differences = []
    rng = np.random.default_rng(SEED + 1)
    for variant in ("t54", "t54_late", "t120"):
        own = rows.filter(pl.col("variant") == variant)
        groups = [(name, [band]) for band, name in enumerate(common.BAND_NAMES)]
        groups += [(name, list(bands)) for name, bands in POOLS]
        for name, bands in groups:
            part = own.filter(pl.col("band").is_in(bands))
            if part.height < 200:
                continue
            scored = score_group(rows=part, rng=rng)
            for pair, values in scored["differences"].items():
                differences.append({"variant": variant, "band": name, "pair": pair, **values})
    diff_table = pl.DataFrame(differences)
    common.write_atomically(frame=diff_table, target=common.RESULTS_DIR / "differences.parquet")
    profile = profile_check(validation=rows.filter(pl.col("variant") == "t54"))
    common.write_atomically(frame=profile, target=common.RESULTS_DIR / "profile.parquet")
    with pl.Config(tbl_rows=200, tbl_cols=20, float_precision=3, tbl_width_chars=200):
        print("Scored rows:", rows.group_by("variant").len().sort("variant"))
        print(
            table.select(
                "variant",
                "band",
                "arm",
                "n",
                "correlation",
                "correlation_low",
                "correlation_high",
                "slope",
                "mse",
            )
        )
        print(diff_table)
        print(profile)
    print("Origin months:", sorted(rows["origin_block"].unique().to_list()))


if __name__ == "__main__":
    main()
