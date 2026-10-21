"""Falsifier F1 of experiment 152: is the sign of D12's fitted flow supported by the BSP above it?

On sunny-midday half-hours of 2024-07 to 2026-06, regress the bulk supply point's (BSP's) reading
on the sum of the MW Primaries beneath it, the sum of the metered generation, the sum of the other
MVA Primaries, D12's term, the CAMS PV shape (a free term for the BSP's unmetered solar), and a
constant. D12's term is its reading `|S|` once, and its signed flow `s * |S|` once, where `s` is
the sign of experiment 110's L0 fitted real power. The sign is supported when the signed version
lowers the residual standard deviation, and the month-block interval of the drop excludes zero.
The free solar term is partly collinear with D12's reverse flow, so a pass is weak evidence and a
clear fail is strong.

    uv run python ml-experiments/00152-*/e00152_f1.py
"""

import json

import numpy as np
import polars as pl
from e00152_common import (
    CLEAR_SKY_INDEX_MINIMUM,
    EXPERIMENT_110_DIR,
    MIDDAY_FIRST_HALF_HOUR_UTC,
    MIDDAY_LAST_HALF_HOUR_UTC,
    SUNNY_MONTHS,
    WORK_DIR,
    load_cache,
)
from scipy.optimize import lsq_linear

import e00152_bsp as c124  # isort: skip

N_BOOTSTRAP = 10_000
SEED = 152
SITE = "D12"
MIN_SERIES_COVERAGE = 0.9


def _design(*, columns: list[np.ndarray]) -> np.ndarray:
    return np.stack(columns, axis=1)


def _fit(*, x: np.ndarray, y: np.ndarray, lower: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = lsq_linear(x, y, bounds=(lower, np.full(len(lower), np.inf)))
    return result.x, y - x @ result.x


def main() -> None:  # noqa: PLR0915
    """Run F1 and write `work/f1.json`."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    bsps = c124.load_bsps(cache=cache)
    bsp = next(b for b in bsps if SITE in b.mva_labels)
    # D12's reading and the sign of L0's fitted flow, on this cache's grid.
    steps = np.full(len(cache.end_times), np.nan)
    sign = np.full(len(cache.end_times), np.nan)
    flag = np.zeros(len(cache.end_times), dtype=bool)

    series_id = cache.sites[SITE].table.time_series_id
    for cut in ("replicate", "scored"):
        frame = pl.read_parquet(
            EXPERIMENT_110_DIR / "predictions" / f"{cut}_l0" / "predictions.parquet"
        ).filter(pl.col("time_series_id") == series_id)
        stamps = frame["valid_time"].dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
        position = np.searchsorted(cache.end_times, stamps)
        ok = (position < len(steps)) & (
            cache.end_times[np.minimum(position, len(steps) - 1)] == stamps
        )
        position = position[ok]
        steps[position] = frame["actual"].to_numpy()[ok]
        sign[position] = np.where(frame["p_mw"].to_numpy()[ok] < 0.0, -1.0, 1.0)
        flag[position] = frame["switching_mask"].to_numpy()[ok]
    eff = cache.sites[SITE].table.effective_capacity_mw
    reading_mva = steps * eff / 100.0
    # Sunny midday on the 124 grid.
    ends = cache.end_times
    starts = ends - np.timedelta64(30, "m")
    month = starts.astype("datetime64[M]").astype(int) % 12 + 1
    hour = (starts - starts.astype("datetime64[D]")) / np.timedelta64(1, "h")
    midday = (
        np.isin(month, SUNNY_MONTHS)
        & (hour >= MIDDAY_FIRST_HALF_HOUR_UTC)
        & (hour <= MIDDAY_LAST_HALF_HOUR_UTC)
    )
    # CAMS clear-sky index from the shared cache.
    ghi = cache.sites[SITE].ghi_w_m2
    from studies.baselines import haurwitz_w_m2
    from studies.solar import zenith

    mid = pl.Series(ends - np.timedelta64(15, "m")).dt.replace_time_zone("UTC")
    clear = haurwitz_w_m2(
        apparent_zenith_deg=zenith(
            stamps=mid,
            latitude=cache.sites[SITE].table.latitude,
            longitude=cache.sites[SITE].table.longitude,
        )
    )
    index = np.where(clear > 20.0, ghi / np.maximum(clear, 1.0), np.nan)
    sunny = midday & (np.nan_to_num(index, nan=0.0) >= CLEAR_SKY_INDEX_MINIMUM)

    window = sunny & ~flag & np.isfinite(reading_mva) & np.isfinite(bsp.reading) & np.isfinite(sign)

    def covered(series: list[np.ndarray]) -> list[np.ndarray]:
        """Keep series that hold a reading on 90% of the window, as experiment 124 does."""
        return [s for s in series if np.isfinite(s[window]).mean() >= MIN_SERIES_COVERAGE]

    others = covered(
        [v for v, label in zip(bsp.mva_list, bsp.mva_labels, strict=True) if label != SITE]
    )
    n = len(ends)
    mw_sum = c124.sum_all(arrays=covered(bsp.mw_list), n=n)
    gen_sum = c124.sum_all(arrays=covered(bsp.generator_list), n=n)
    other_sum = c124.sum_all(arrays=others, n=n)
    out: dict[str, object] = {"n_mw": len(bsp.mw_list), "n_other_mva": len(others)}
    base = (
        sunny
        & ~flag
        & np.isfinite(reading_mva)
        & np.isfinite(bsp.reading)
        & np.isfinite(mw_sum)
        & np.isfinite(gen_sum)
        & np.isfinite(other_sum)
        & np.isfinite(sign)
        & np.isfinite(bsp.pv_shape)
    )
    rows = np.flatnonzero(base)
    out["n_rows"] = len(rows)
    y = bsp.reading[rows]
    d12_unsigned = reading_mva[rows]
    d12_signed = sign[rows] * reading_mva[rows]
    common = [mw_sum[rows], gen_sum[rows], other_sum[rows]]
    shape = -bsp.pv_shape[rows]
    ones = np.ones(len(rows))
    # Columns: MW sum (>=0), generation sum (free), other MVA (>=0), D12 (>=0), solar (>=0), level.
    lower = np.array([0.0, -np.inf, 0.0, 0.0, 0.0, -np.inf])
    results = {}
    for name, term in (("unsigned", d12_unsigned), ("signed", d12_signed)):
        x = _design(columns=[*common, term, shape, ones])
        theta, residual = _fit(x=x, y=y, lower=lower)
        results[name] = {"theta": theta.tolist(), "residual": residual}
        out[name] = {
            "coef": dict(
                zip(
                    ["mw", "gen", "other_mva", "d12", "solar", "level"], theta.tolist(), strict=True
                )
            ),
            "residual_sd_mw": float(np.std(residual)),
        }
    drop = results["unsigned"]["residual"] ** 2 - results["signed"]["residual"] ** 2
    out["variance_drop_mw2"] = float(drop.mean())
    months = ends[rows].astype("datetime64[M]").astype(int)
    unique = np.unique(months)
    rng = np.random.default_rng(SEED)
    block_sum = np.array([drop[months == m].sum() for m in unique])
    block_n = np.array([(months == m).sum() for m in unique])
    draws = rng.integers(0, len(unique), size=(N_BOOTSTRAP, len(unique)))
    boot = block_sum[draws].sum(axis=1) / block_n[draws].sum(axis=1)
    out["variance_drop_interval_95"] = [
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    ]
    out["n_months"] = len(unique)
    sd_u, sd_s = out["unsigned"]["residual_sd_mw"], out["signed"]["residual_sd_mw"]  # ty: ignore[not-subscriptable]
    out["residual_sd_drop_share"] = float(1.0 - sd_s / sd_u)
    # Summary of BSP size for scale.
    out["bsp_reading_mean_mw"] = float(np.mean(y))
    out["d12_term_mean_unsigned_mva"] = float(np.mean(d12_unsigned))
    out["d12_term_mean_signed_mw"] = float(np.mean(d12_signed))
    out["bsp_index"] = bsp.index
    (WORK_DIR / "f1.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
