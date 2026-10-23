# ruff: noqa: E501
"""Bias of the mean forecast on clear half-hours, by calendar month and for April to July (experiment 293).

For each own-history arm and farm: the sum of actual output over the sum of the mean forecast (the
mean of the 11 quantiles, weighted by the width of each level) minus one, on scored rows whose
clear-sky index (ensemble-mean irradiance over clear-sky irradiance) is 0.9 or more. The April to
July shrink of an arm is the median over farms of P2's absolute bias minus the arm's, in percentage
points; its interval resamples the four months.

    uv run --no-sync python ml-experiments/00293-*/e00293_bias.py
"""

from typing import Final

import e00293_common as common
import e00293_score as score
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

QCOLS: Final[list[str]] = [level_column(level=lv) for lv in common.LEVELS]
MEAN_WEIGHTS: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (common.LEVELS[:-1] + common.LEVELS[1:]) / 2, [1.0]])
)
CLEAR_INDEX: Final[float] = 0.9
SPRING_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7)
ARMS: Final[tuple[str, ...]] = (
    "p2",
    "a_tilt20",
    "b_soft_clip",
    "c_tilt20_soft_clip",
    "d_pooled_shape",
    "e_hard_cap_999",
    "f_tilt20_hard_cap_999",
)
DRAWS: Final[int] = 5_000


def clear_rows(*, setting: str) -> pl.DataFrame:
    """Return the clear-sky scored rows of every arm with actual output and the mean forecast."""
    parts = []
    keys, _ = score.reference_rows()
    for site in common.FARMS:
        csi = common.load_farm(site).frame.select("time", "csi")
        for arm in ARMS:
            path = common.PRED_DIR / setting / f"{arm}_s0_{site}.parquet"
            if not path.exists():
                if arm == "d_pooled_shape" and setting != "own_history":
                    continue
                msg = f"missing predictions: {path}"
                raise FileNotFoundError(msg)
            frame = (
                pl.read_parquet(path)
                .join(keys, on=["time_series_id", "time"], how="inner")
                .join(csi, on="time", how="left", validate="1:1")
            )
            if frame["csi"].null_count():
                msg = f"null clear-sky index for farm {site}"
                raise ValueError(msg)
            q = np.sort(frame.select(QCOLS).to_numpy(), axis=1)
            parts.append(
                frame.select("time", "time_series_id", "arm", "actual", "csi").with_columns(
                    mean_forecast=pl.Series(q @ MEAN_WEIGHTS),
                    month=pl.col("time").dt.month(),
                )
            )
    return pl.concat(parts).filter(pl.col("csi") >= CLEAR_INDEX)


def main() -> None:
    """Print and write the monthly ratios and the April to July shrink."""
    pl.Config.set_tbl_rows(100)
    rows = clear_rows(setting="own_history")
    monthly = (
        rows.group_by("arm", "time_series_id", "month")
        .agg(actual=pl.col("actual").sum(), forecast=pl.col("mean_forecast").sum(), n=pl.len())
        .with_columns(bias=pl.col("actual") / pl.col("forecast") - 1.0)
    )
    monthly.write_csv(common.DATA_DIR / "bias_by_month.csv")
    spring = rows.filter(pl.col("month").is_in(SPRING_MONTHS))
    per_farm = (
        spring.group_by("arm", "time_series_id")
        .agg(actual=pl.col("actual").sum(), forecast=pl.col("mean_forecast").sum(), n=pl.len())
        .with_columns(bias_percent=100.0 * (pl.col("actual") / pl.col("forecast") - 1.0))
    )
    per_farm.write_csv(common.DATA_DIR / "bias_april_july_per_farm.csv")
    print(
        per_farm.pivot(on="arm", index="time_series_id", values="bias_percent").sort(
            "time_series_id"
        )
    )
    print(per_farm.pivot(on="arm", index="time_series_id", values="n").sort("time_series_id"))
    by_month = spring.group_by("arm", "time_series_id", "month").agg(
        actual=pl.col("actual").sum(), forecast=pl.col("mean_forecast").sum()
    )
    rng = np.random.default_rng(293)
    draws = rng.multinomial(
        len(SPRING_MONTHS), np.ones(len(SPRING_MONTHS)) / len(SPRING_MONTHS), size=DRAWS
    )

    def median_abs_bias(arm: str) -> tuple[float, np.ndarray]:
        actual = np.zeros((len(common.FARMS), len(SPRING_MONTHS)))
        forecast = np.zeros_like(actual)
        for r in by_month.filter(pl.col("arm") == arm).iter_rows(named=True):
            i, j = common.FARMS.index(r["time_series_id"]), SPRING_MONTHS.index(r["month"])
            actual[i, j], forecast[i, j] = r["actual"], r["forecast"]
        point = np.median(np.abs(100.0 * (actual.sum(1) / forecast.sum(1) - 1.0)))
        resampled = np.median(
            np.abs(100.0 * ((actual @ draws.T) / (forecast @ draws.T) - 1.0)), axis=0
        )
        return float(point), resampled

    base, base_draws = median_abs_bias("p2")
    print(f"p2: median absolute April to July bias {base:.2f} points")
    rows_out = [{"arm": "p2", "median_abs_bias": base, "shrink": 0.0, "lo": 0.0, "hi": 0.0}]
    for arm in ARMS[1:]:
        point, arm_draws = median_abs_bias(arm)
        shrink = base_draws - arm_draws
        lo, hi = np.percentile(shrink, [0.625, 99.375])
        rows_out.append(
            {"arm": arm, "median_abs_bias": point, "shrink": base - point, "lo": lo, "hi": hi}
        )
    table = pl.DataFrame(rows_out)
    table.write_csv(common.DATA_DIR / "bias_april_july_shrink.csv")
    print(table)


if __name__ == "__main__":
    main()
