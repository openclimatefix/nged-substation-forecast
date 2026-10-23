# ruff: noqa: E501
"""RULE, ORACLE, and the fixed forecasts on experiment 258's real record (observed power only).

Past-weather GAM of experiment 130 (ERA5 and CAMS) plus the Kalman filter, 15 Primary series,
6-hourly origins, 336 leads. K0 is the forecast with lags and GAM_ONLY the forecast without. RULE
uses K0 while the stored score of experiment 308 says the arrangement is abnormal at the origin and
GAM_ONLY otherwise. Every arm shares experiment 258's quantile offset table, so a difference is a
difference of location. The threshold-weighted CRPS of the injected-world script is replaced here by
the pinball loss at the 95% quantile (`pin95`), because the offset table gives 11 quantiles per
row. Shifts are the 35 labelled real load shifts over 1 day. Intervals resample whole shifts and
whole calendar months, 20,000 draws.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00309_real.py
"""

from __future__ import annotations

import json
from typing import Any, Final

import e00309_data as dat
import e00309_harness as h
import e00309_state as st
import numpy as np
import polars as pl

N_DRAWS: Final[int] = 20_000
METRICS: Final[tuple[str, ...]] = ("qs", "pin95", "abs_median")
GROUPS: Final[tuple[str, ...]] = ("ordinary", "during", "after", "near", "during_before_alarm")
AFTER_HALF_HOURS: Final[int] = 144
RULE_ARMS: Final[tuple[str, ...]] = tuple(
    f"RULE_{score}_{k}" for score, ks in st.THRESHOLDS.items() for k in ks
)
ARMS: Final[tuple[str, ...]] = (
    "GAM_ONLY", "K0", *RULE_ARMS, "ORACLE_ROW", "ORACLE_STATE",
)  # fmt: skip


def arm_metrics(*, data: h.SeriesData, correction: np.ndarray, offsets: np.ndarray, band: np.ndarray) -> dict[str, np.ndarray]:
    """Per (origin, lead) metrics of one location, NaN where the target has no reading."""
    _, _, targets = h.origin_grid()
    clipped = np.minimum(targets, h.N_GRID - 1)
    actual = data.actual[clipped].astype(np.float64)
    location = data.expected[clipped] + correction
    cell = offsets[band[clipped], np.broadcast_to(h.lead_band_of_lead()[None, :], targets.shape)]
    bad = (targets >= h.N_GRID) | ~np.isfinite(actual)
    error = actual[..., None] - (location[..., None] + cell)
    pinball = np.maximum(h.LEVELS * error, (h.LEVELS - 1.0) * error)
    out = {"qs": 2.0 * pinball.mean(axis=-1), "pin95": pinball[..., 9], "abs_median": np.abs(location + cell[..., 5] - actual)}
    return {k: np.where(bad, np.nan, v) for k, v in out.items()}


def group_masks(*, series: int, origins: np.ndarray, targets: np.ndarray, state: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Boolean masks (origins by leads) per row group, the shift label per row, and the in-shift flag per origin."""
    all_shifts = dat.labelled_loads().filter(pl.col("series").list.eval(pl.element().is_in(list(h.SERIES))).list.any())
    shifts = all_shifts.with_row_index("shift_index").filter(pl.col("series").list.contains(series))
    clipped = np.minimum(targets, h.N_GRID - 1)
    own = np.zeros(h.N_GRID + AFTER_HALF_HOURS, dtype=bool)
    wanted = {str(series)} | {str(s) for s in dat.ADJACENCY[series]}
    for r in dat.logged_closures().iter_rows(named=True):
        if r["side_a"] in wanted or r["side_b"] in wanted:
            own[max(int(r["first_stamp"]), 0) : min(int(r["last_stamp_exclusive"]) + AFTER_HALF_HOURS, h.N_GRID)] = True
    extra = np.zeros(h.N_GRID, dtype=bool)
    if series == st.D8:
        for r in pl.read_parquet(dat.D8_DROPS).iter_rows(named=True):
            extra[int(r["first_stamp"]) : int(r["last_stamp"]) + 1] = True
    for lo, hi in dat.block_day_ranges().get(series, []):
        extra[max(lo, 0) : hi + 1] = True
    during = np.zeros(targets.shape, dtype=bool)
    after = np.zeros(targets.shape, dtype=bool)
    label = np.full(targets.shape, -1, dtype=np.int64)
    in_shift_origin = np.zeros(len(origins), dtype=bool)
    for r in shifts.iter_rows(named=True):
        i = r["shift_index"]
        d = (origins[:, None] >= r["onset"]) & (origins[:, None] < r["end"])
        a = (origins[:, None] >= r["end"]) & (targets >= r["end"]) & (targets < r["end"] + AFTER_HALF_HOURS)
        during |= d
        after |= a
        label = np.where((d | a) & (label < 0), i, label)
        in_shift_origin |= (origins >= r["onset"]) & (origins < r["end"])
    near = during | after
    ordinary = ~own[: h.N_GRID][clipped] & ~extra[clipped] & ~near
    masks = {"ordinary": ordinary, "during": during, "after": after, "near": near}
    return masks, label, in_shift_origin


def series_table(*, series: int) -> pl.DataFrame:
    """Per-unit sums of every metric, by arm and row group, for one series."""
    band = h.band_grid()
    data = h.load_series(series=series)
    stored = np.load(h.HARNESS_DIR / f"series_{series}.npz")
    offsets = stored["offsets"]
    origins, _, targets = h.origin_grid()
    clipped = np.minimum(targets, h.N_GRID - 1)
    z = np.load(st.GAP_DIR / f"real_{series}.npz")
    assert np.array_equal(z["origin"], origins)
    state = {}
    for score, ks in st.THRESHOLDS.items():
        sigma = st.sigma_of(series=series, score=score)
        for k in ks:
            state[f"RULE_{score}_{k}"] = st.abnormal_state(gap=z[score], sigma=sigma, k=k)
    masks, label, in_shift_origin = group_masks(series=series, origins=origins, targets=targets, state=state)
    year_month = dat.stamp_year_month()[clipped]
    odd = np.isin(h.utc_month_grid()[origins], list(h.TUNING_MONTHS))[:, None] & np.ones(targets.shape, dtype=bool)
    base = {arm: arm_metrics(data=data, correction=stored[arm].astype(np.float64), offsets=offsets, band=band) for arm in ("GAM_ONLY", "K0")}
    metrics = dict(base)
    for name, s in state.items():
        use = np.broadcast_to(s[:, None], targets.shape)
        metrics[name] = {m: np.where(use, base["K0"][m], base["GAM_ONLY"][m]) for m in METRICS}
    metrics["ORACLE_ROW"] = {m: np.where(base["K0"]["qs"] < base["GAM_ONLY"]["qs"], base["K0"][m], base["GAM_ONLY"][m]) for m in METRICS}
    use = np.broadcast_to(in_shift_origin[:, None], targets.shape)
    metrics["ORACLE_STATE"] = {m: np.where(use, base["K0"][m], base["GAM_ONLY"][m]) for m in METRICS}
    rows = []
    for arm, met in metrics.items():
        for group in GROUPS:
            if group == "during_before_alarm":
                continue
            mask = masks[group] & np.isfinite(met["qs"])
            unit = year_month if group == "ordinary" else label
            for parity in (True, False):
                m2 = mask & (odd == parity)
                if not m2.any():
                    continue
                u = unit[m2]
                codes, inv = np.unique(u, return_inverse=True)
                cols = {"n": np.bincount(inv).astype(float)} | {k: np.bincount(inv, weights=met[k][m2]) for k in METRICS}
                rows.append(pl.DataFrame({"series": series, "arm": arm, "group": group, "odd": parity, "unit": codes.astype(np.int64), **cols}))
    return pl.concat(rows)


def main() -> None:
    """Score every series, then print pooled and bootstrapped results."""
    out_dir = dat.WORK_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pl.concat([series_table(series=s) for s in h.SERIES])
    table.write_parquet(out_dir / "real_units.parquet")
    print(table.height, "unit rows written")


if __name__ == "__main__":
    main()
