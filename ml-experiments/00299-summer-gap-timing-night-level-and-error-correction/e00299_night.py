"""Part 3 of experiment 299: is D10's night-time level error a step on a date or a smooth drift?

Reads arm D's stored forecasts at D10 (experiment 155's tuning, replicate, and scored periods) and
experiment 201's driving temperature, builds the nightly mean residual, and fits one change point
against a fit on temperature band and month.

    OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 systemd-run --user --scope -p MemoryMax=8G \
        -p OOMPolicy=continue uv run --no-sync python ml-experiments/00299-*/e00299_night.py
"""

from typing import Final

import e00299_common as common  # isort: skip
import numpy as np
import polars as pl

SITE: Final[str] = "D10"
NIGHT_UTC_HOURS: Final[tuple[int, ...]] = (21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7)
TEMP_EDGES: Final[tuple[float, ...]] = (5.0, 10.0, 15.0)
MIN_NIGHT_ROWS: Final[int] = 8
MIN_SEGMENT_NIGHTS: Final[int] = 28
SCORED_YEAR_START: Final = "2025-07-02"


def megawatts_per_point() -> float:
    """Return the megawatts that one point of effective capacity stands for at D10.

    The master frame of experiment 266 scores in points and the stored forecasts in megawatts, so
    the ratio of the summed per-row quantile scores on the shared scored rows is the scale.
    """
    scored = pl.read_parquet(common.E155_PREDS / f"d_scored_{SITE}.parquet")
    master = common.read_master().filter(pl.col("site") == SITE)
    joined = scored.join(
        master.select("power_fcst_init_time", "valid_time", "qs_d"),
        on=["power_fcst_init_time", "valid_time"],
        how="inner",
    )
    qs_mw = common.qs_per_row(
        actual=joined["actual"].to_numpy(), q=joined.select(common.QUANTILES).to_numpy()
    )
    return float(qs_mw.sum() / joined["qs_d"].sum())


def load_rows(*, scale: float) -> pl.DataFrame:
    """Return arm D's D10 night rows (residuals in points) from 2024-07 to 2026-06."""
    parts = [
        pl.read_parquet(common.E155_PREDS / f"d_{period}_{SITE}.parquet")
        for period in ("tuning", "replicate", "scored")
    ]
    preds = pl.concat(parts).filter(~pl.col("switching_mask"))
    temp = pl.read_parquet(
        common.E201_FEATURES / f"{SITE}.parquet",
        columns=["power_fcst_init_time", "valid_time", "drv_temp_c", "true_temp_c"],
    )
    rows = preds.join(temp, on=["power_fcst_init_time", "valid_time"], how="left")
    start = pl.col("valid_time") - pl.duration(minutes=30)  # valid_time labels the half-hour end
    return (
        rows.with_columns(
            utc_hour=start.dt.hour(),
            night_date=(start - pl.duration(hours=12)).dt.date(),
            resid_mean=(pl.col("actual") - pl.col("mean")) / scale,
            resid_median=(pl.col("actual") - pl.col("q50")) / scale,
            weekday=(start - pl.duration(hours=12)).dt.weekday(),
            month=pl.col("valid_time").dt.strftime("%Y-%m"),
        )
        .filter(pl.col("utc_hour").is_in(list(NIGHT_UTC_HOURS)))
        .filter(pl.col("lead_hours") >= 12)
        .filter(pl.col("lead_hours") < 36)
    )


def band(*, temperature: np.ndarray) -> np.ndarray:
    """Return the index of the temperature band, from 0 (under 5 C) to 3 (15 C and over)."""
    return np.searchsorted(np.array(TEMP_EDGES), temperature, side="right")


def best_step(*, y: np.ndarray) -> tuple[int, float]:
    """Return the split index and R squared of the best one-change-point fit of `y`."""
    n = len(y)
    total = ((y - y.mean()) ** 2).sum()
    csum, csq = np.cumsum(y), np.cumsum(y**2)
    best, best_k = np.inf, 1
    for k in range(MIN_SEGMENT_NIGHTS, n - MIN_SEGMENT_NIGHTS):
        left = csq[k - 1] - csum[k - 1] ** 2 / k
        right = (csq[-1] - csq[k - 1]) - (csum[-1] - csum[k - 1]) ** 2 / (n - k)
        if left + right < best:
            best, best_k = left + right, k
    return best_k, float(1.0 - best / total)


def adjusted_r_squared(*, r2: float, n: int, parameters: int) -> float:
    """Return R squared adjusted for the number of fitted parameters (excluding the intercept)."""
    return 1.0 - (1.0 - r2) * (n - 1) / (n - parameters - 1)


def r_squared_on_groups(*, y: np.ndarray, groups: list[np.ndarray]) -> float:
    """Return R squared of a fit of `y` on the dummies of every grouping in `groups` (OLS)."""
    columns = [np.ones(len(y))]
    for g in groups:
        columns.extend((g == level).astype(float) for level in np.unique(g)[1:])
    x = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(1.0 - ((y - x @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main() -> None:
    """Write `work/night.json` and print the headline numbers."""
    scale = megawatts_per_point()
    print("MW per point", scale)
    rows = load_rows(scale=scale)
    print("rows", rows.height, "temperature nulls", rows["drv_temp_c"].null_count())
    out: dict = {"rows": rows.height, "mw_per_point": scale}
    nightly = (
        rows.group_by("night_date")
        .agg(
            pl.col("resid_mean").mean().alias("resid_mean"),
            pl.col("resid_median").mean().alias("resid_median"),
            pl.col("drv_temp_c").mean().alias("temp"),
            pl.col("true_temp_c").mean().alias("true_temp"),
            pl.col("weekday").first().alias("weekday"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= MIN_NIGHT_ROWS)
        .sort("night_date")
        .with_columns(
            month=pl.col("night_date").dt.strftime("%Y-%m"),
            year_month_label=pl.col("night_date").dt.strftime("%Y-%m"),
        )
    )
    nightly = nightly.filter(pl.col("temp").is_not_null())
    nightly.write_parquet(common.WORK_DIR / "night_nightly.parquet")
    out["nights"] = nightly.height
    out["monthly"] = (
        nightly.group_by("month")
        .agg(pl.col("resid_mean").mean(), pl.col("temp").mean(), pl.len().alias("nights"))
        .sort("month")
        .to_dicts()
    )
    out["by_weekday"] = (
        nightly.group_by("weekday")
        .agg(pl.col("resid_mean").mean(), pl.len())
        .sort("weekday")
        .to_dicts()
    )
    tb = band(temperature=nightly["temp"].to_numpy())
    out["by_temperature_band"] = [
        {
            "band": int(b),
            "resid_mean": float(nightly["resid_mean"].to_numpy()[tb == b].mean()),
            "nights": int((tb == b).sum()),
        }
        for b in np.unique(tb)
    ]
    full = nightly
    for label, series, window in (
        ("mean_all", "resid_mean", "all"),
        ("mean_scored_year", "resid_mean", "scored"),
    ):
        nightly = (
            full
            if window == "all"
            else full.filter(pl.col("night_date").cast(pl.String) >= SCORED_YEAR_START)
        )
        tb = band(temperature=nightly["temp"].to_numpy())
        y = nightly[series].to_numpy()
        k, r2_step = best_step(y=y)
        r2_tm = r_squared_on_groups(y=y, groups=[tb, nightly["month"].to_numpy().astype(str)])
        r2_t = r_squared_on_groups(y=y, groups=[tb])
        dates = nightly["night_date"].to_numpy()
        # Block bootstrap over ISO weeks for the change-point date.
        weeks = np.array([str(d.astype("datetime64[W]")) for d in dates])
        uniq = np.unique(weeks)
        rng = np.random.default_rng(0)
        found = []
        for _ in range(common.N_RESAMPLES // 10):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(weeks == w) for w in pick])
            idx = idx[np.argsort(dates[idx], kind="stable")]
            kk, _ = best_step(y=y[idx])
            found.append(str(dates[idx][kk]))
        found_dates = np.array(found, dtype="datetime64[D]")
        lo, hi = np.percentile(found_dates.astype("int64"), [2.5, 97.5]).astype(int)
        wd = nightly["weekday"].to_numpy()
        mo = nightly["month"].to_numpy().astype(str)
        out[label] = {
            "nights": nightly.height,
            "r2_weekday": r_squared_on_groups(y=y, groups=[wd]),
            "r2_month": r_squared_on_groups(y=y, groups=[mo]),
            "adj_r2_month": adjusted_r_squared(
                r2=r_squared_on_groups(y=y, groups=[mo]),
                n=len(y),
                parameters=len(np.unique(mo)) - 1,
            ),
            "adj_r2_step": adjusted_r_squared(r2=r2_step, n=len(y), parameters=2),
            "month_levels": len(np.unique(mo)),
            "r2_month_and_weekday": r_squared_on_groups(y=y, groups=[mo, wd]),
            "r2_temperature_band_month_weekday": r_squared_on_groups(y=y, groups=[tb, mo, wd]),
            "step_date": str(dates[k]),
            "level_before": float(y[:k].mean()),
            "level_after": float(y[k:].mean()),
            "r2_step": r2_step,
            "r2_temperature_band": r2_t,
            "r2_temperature_band_and_month": r2_tm,
            "step_date_95_lo": str(np.datetime64(int(lo), "D")),
            "step_date_95_hi": str(np.datetime64(int(hi), "D")),
            "interval_width_days": int(hi - lo),
        }
        print(label, out[label])
    # Same cut by calendar year of the night, for the April-to-June comparison.
    out["amj_by_year"] = (
        full.filter(pl.col("night_date").dt.month().is_in([4, 5, 6]))
        .group_by(pl.col("night_date").dt.year().alias("year"))
        .agg(pl.col("resid_mean").mean(), pl.col("temp").mean(), pl.len().alias("nights"))
        .sort("year")
        .to_dicts()
    )
    print(out["amj_by_year"])
    common.write_json_atomic(path=common.WORK_DIR / "night.json", content=out)


if __name__ == "__main__":
    main()
