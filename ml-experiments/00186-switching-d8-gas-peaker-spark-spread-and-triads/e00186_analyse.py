# ruff: noqa: E501
"""Experiment 186: are D8's multi-day drops an embedded gas peaker running?

Reads experiment 159's stored half-hourly residual panel (`arrays.pkl`: metered power, the frozen
convex model's expected power, and ERA5 temperature per series on experiment 130's grid), the
public Elexon system prices and the National Gas daily system average price (SAP) that
`e00186_fetch.py` stored, and a short list of NESO's published Triad half-hours. Prints every
number the write-up quotes into `printed_report.txt` and writes private tables under `WORK_DIR`.

Run from the repository root with `OMP_NUM_THREADS=1` under `systemd-run --user --scope -p
MemoryMax=16G -p OOMPolicy=continue`, as `uv run --no-sync python <this file>` with this folder on
`PYTHONPATH`.
"""

from __future__ import annotations

import pickle
from datetime import UTC, date, datetime, timedelta
from typing import Final

import e00186_common as common
import numpy as np
import polars as pl

GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
WINDOW_START: Final[date] = date(2023, 7, 1)
WINDOW_END: Final[date] = date(2026, 6, 30)
"""Prices exist from 2023-06-01; drop 1 (2022) lies before the window and is not scored."""
SMOOTHING_WINDOW: Final[int] = 48
DEPTH_THRESHOLD_MW: Final[float] = 1.5
MERGE_GAP: Final[int] = 24
MIN_DURATION: Final[int] = 96
"""The drop definition of experiment 159: 24-hour rolling median under -1.5 MW, merged, >= 2 days."""
MAX_PLAUSIBLE_GAS_P_PER_KWH: Final[float] = 15.0
"""Two daily values (72.8 and 63.9 pence per kWh) are data errors; the rest lie between 1 and 11."""
EFFICIENCIES: Final[tuple[float, ...]] = (0.35, 0.40, 0.45)
CENTRAL_EFFICIENCY: Final[float] = 0.40
VARIABLE_COST: Final[float] = 5.0
"""Pounds per MWh of electricity."""
FLAT_LIMIT: Final[float] = 0.25
N_DRAWS: Final[int] = 1000
SEED: Final[int] = 0
EVENING: Final[tuple[int, int]] = (33, 38)
"""Settlement periods 33 to 38, 16:00 to 19:00 local."""
TRIADS: Final[tuple[tuple[date, int], ...]] = (
    (date(2023, 12, 5), 35), (date(2024, 1, 4), 35), (date(2024, 1, 17), 36),
    (date(2024, 11, 20), 37), (date(2025, 1, 9), 36), (date(2025, 2, 10), 36),
    (date(2025, 11, 20), 34), (date(2026, 1, 5), 35), (date(2026, 2, 3), 37),
)  # fmt: skip
"""NESO's published Triads (date, settlement period) for charging years 2023/24 to 2025/26."""
REPORT: list[str] = []


def say(text: str = "") -> None:
    """Print a line and keep it for `printed_report.txt`."""
    print(text, flush=True)
    REPORT.append(text)


def find_drops(*, residual: np.ndarray) -> list[tuple[int, int]]:
    """Runs of the 24-hour rolling median residual under -1.5 MW, merged, of at least 2 days."""
    smooth = (
        pl.Series(residual)
        .rolling_median(SMOOTHING_WINDOW, min_samples=SMOOTHING_WINDOW // 2, center=True)
        .to_numpy()
    )
    below = np.isfinite(smooth) & (smooth < -DEPTH_THRESHOLD_MW)
    edges = np.flatnonzero(np.diff(np.r_[0, below.astype(int), 0]) != 0).reshape(-1, 2)
    merged: list[list[int]] = []
    for first, last in edges:
        if merged and first - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = int(last)
        else:
            merged.append([int(first), int(last)])
    return [(a, b) for a, b in merged if b - a >= MIN_DURATION]


def local_calendar(*, n: int) -> pl.DataFrame:
    """Local date, hour, weekday (0 = Monday), month, and settlement period per grid stamp."""
    utc = pl.Series(
        [GRID_START + timedelta(minutes=30 * i) for i in range(n)],
        dtype=pl.Datetime(time_zone="UTC"),
    )
    local = utc.dt.convert_time_zone(common.LOCAL_TZ)
    frame = pl.DataFrame({"utc": utc}).with_columns(
        date=local.dt.date(),
        hour=local.dt.hour() + local.dt.minute() / 60,
        weekday=local.dt.weekday() - 1,
        month=local.dt.month(),
        doy=local.dt.ordinal_day(),
    )
    # Settlement period 1 starts at local midnight; the first period of a day is at hour 0.
    return frame.with_columns(
        sp=(pl.col("hour") * 2).round().cast(pl.Int64) + 1,
    )


def prices() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Half-hourly system price and daily gas price (pounds per MWh)."""
    files = [
        *sorted((common.PUBLIC_DIR / "system_prices").glob("*.parquet")),
        *sorted((common.E185_PUBLIC / "system_prices").glob("*.parquet")),
        *sorted((common.E177_PUBLIC / "system_prices").glob("*.parquet")),
    ]
    system = (
        pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
        .with_columns(
            date=pl.col("settlementDate").str.to_date(),
            sp=pl.col("settlementPeriod").cast(pl.Int64),
            price=pl.max_horizontal("systemSellPrice", "systemBuyPrice"),
        )
        .select("date", "sp", "price")
        .unique(subset=["date", "sp"])
    )
    gas = (
        pl.read_csv(common.PUBLIC_DIR / "gas_sap.csv")
        .with_columns(
            date=pl.col("Applicable For").str.to_date("%d/%m/%Y"),
            gas_pence_per_kwh=pl.col("Value").cast(pl.Float64, strict=False),
        )
        .select("date", "gas_pence_per_kwh")
        .unique(subset="date")
        .filter(pl.col("gas_pence_per_kwh").is_between(0.5, MAX_PLAUSIBLE_GAS_P_PER_KWH))
        .with_columns(gas_gbp_per_mwh_thermal=pl.col("gas_pence_per_kwh") * 10.0)
    )
    return system, gas


def share_text(*, k: int, n: int) -> str:
    """Format a count out of a total with the share."""
    return f"{k} of {n} ({k / n:.2f})"


def rolling_median(*, values: np.ndarray, width: int) -> np.ndarray:
    """Centred rolling median that tolerates NaN."""
    return (
        pl.Series(values)
        .rolling_median(width, min_samples=max(2, width // 2), center=True)
        .to_numpy()
    )


def drop_shape(*, depth: np.ndarray, first: int, last: int, hour: np.ndarray) -> dict[str, float]:
    """Shape statistics of one drop from its half-hourly depth (megawatts below expected)."""
    inside = depth[first:last]
    smooth = rolling_median(values=inside, width=6)
    median = float(np.nanmedian(inside))
    ok = np.isfinite(smooth)
    q25, q75 = np.nanpercentile(smooth[ok], [25, 75])
    rel_iqr = float((q75 - q25) / median) if median > 0 else float("nan")
    flat = float(np.mean(np.abs(smooth[ok] - median) <= FLAT_LIMIT * median))
    hours = hour[first:last][ok]
    hourly = np.array([np.nanmean(smooth[ok][(hours >= h) & (hours < h + 1)]) for h in range(24)])
    resid_var = np.nanvar(smooth[ok] - hourly[hours.astype(int)])
    total_var = np.nanvar(smooth[ok])
    before = depth[max(first - 24, 0) : max(first - 12, 1)]
    after = depth[first + 12 : first + 24]
    end_before = depth[last - 24 : last - 12]
    end_after = depth[last + 12 : last + 24]
    start_step = (np.nanmedian(after) - np.nanmedian(before)) / median
    end_step = (np.nanmedian(end_before) - np.nanmedian(end_after)) / median
    return {
        "median_depth": median,
        "rel_iqr": rel_iqr,
        "flat_share": flat,
        "hour_r2": float(1 - resid_var / total_var) if total_var > 0 else float("nan"),
        "start_step": float(start_step),
        "end_step": float(end_step),
    }


def day_table(*, cal: pl.DataFrame, series: dict, drops: list[tuple[int, int]]) -> pl.DataFrame:
    """One row per local date: D8 drop share, ERA5 temperature, and mean depth."""
    in_drop = np.zeros(len(cal), dtype=bool)
    for first, last in drops:
        in_drop[first:last] = True
    return (
        cal.with_columns(
            in_drop=pl.Series(in_drop),
            temp=pl.Series(series["temp"]),
            depth=pl.Series(-series["resid"]),
        )
        .group_by("date")
        .agg(
            n=pl.len(),
            drop_share=pl.col("in_drop").mean(),
            temp=pl.col("temp").mean(),
            depth=pl.col("depth").mean(),
            weekday=pl.col("weekday").first(),
            doy=pl.col("doy").first(),
            month=pl.col("month").first(),
        )
        .filter(pl.col("n") >= 46)
        .sort("date")
    )


def add_spread(*, days: pl.DataFrame, system: pl.DataFrame, gas: pl.DataFrame) -> pl.DataFrame:
    """Join daily mean and evening system price, gas price, and spark spreads per efficiency."""
    daily = system.group_by("date").agg(
        price_mean=pl.col("price").mean(),
        price_eve=pl.col("price").filter(pl.col("sp").is_between(*EVENING)).mean(),
        n_sp=pl.len(),
    )
    out = days.join(daily, on="date", how="inner").join(gas, on="date", how="inner")
    out = out.filter(pl.col("n_sp") >= 46).sort("date")
    for eff in EFFICIENCIES:
        tag = f"{round(eff * 100)}"
        out = out.with_columns(
            (pl.col("price_mean") - pl.col("gas_gbp_per_mwh_thermal") / eff - VARIABLE_COST).alias(
                f"spread_{tag}"
            ),
            (pl.col("price_eve") - pl.col("gas_gbp_per_mwh_thermal") / eff - VARIABLE_COST).alias(
                f"spread_eve_{tag}"
            ),
        )
    return out


def matched_pool(
    *, table: pl.DataFrame, drop_days: pl.DataFrame, in_any_drop_pad: set[date]
) -> tuple[list[np.ndarray], pl.DataFrame]:
    """For each drop day, the row indices of ordinary days matched on weekday, season, temperature."""
    ordinary = table.filter(~pl.col("date").is_in(list(in_any_drop_pad)))
    o_idx = np.arange(len(ordinary))
    o_wd = ordinary["weekday"].to_numpy()
    o_doy = ordinary["doy"].to_numpy()
    o_temp = ordinary["temp"].to_numpy()
    pools = []
    for row in drop_days.iter_rows(named=True):
        gap = np.abs(o_doy - row["doy"])
        gap = np.minimum(gap, 365 - gap)
        for tol in (2.0, 4.0, 6.0):
            ok = (o_wd == row["weekday"]) & (gap <= 30) & (np.abs(o_temp - row["temp"]) <= tol)
            if ok.sum() >= 3:
                break
        pools.append(o_idx[ok])
    return pools, ordinary


def interval(values: np.ndarray) -> str:
    """Mean and 95% percentile interval."""
    lo, hi = np.percentile(values, [2.5, 97.5])
    return f"{values.mean():+.3f} ({lo:+.3f} to {hi:+.3f})"


def matched_test(
    *,
    rng: np.random.Generator,
    stat: np.ndarray,
    ordinary_stat: np.ndarray,
    pools: list[np.ndarray],
    drop_of_day: np.ndarray,
    n_drops: int,
) -> tuple[float, float, float, np.ndarray]:
    """Mean statistic on drop days, on matched ordinary days, and their difference with its interval.

    Each draw picks one matched ordinary day per drop day, and resamples the drops with
    replacement, so that the interval respects the clustering of days within a drop.
    """
    diffs = np.empty(N_DRAWS)
    obs_all = np.empty(N_DRAWS)
    mat_all = np.empty(N_DRAWS)
    for i in range(N_DRAWS):
        picks = np.array([ordinary_stat[rng.choice(p)] for p in pools])
        chosen = rng.integers(0, n_drops, n_drops)
        mask = np.concatenate([np.flatnonzero(drop_of_day == d) for d in chosen])
        obs_all[i] = np.nanmean(stat[mask])
        mat_all[i] = np.nanmean(picks[mask])
        diffs[i] = obs_all[i] - mat_all[i]
    return float(np.nanmean(stat)), float(mat_all.mean()), float(diffs.mean()), diffs


def main() -> None:  # noqa: PLR0912, PLR0915
    """Run every step and print the report."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    arrays = pickle.loads(common.ARRAYS_PATH.read_bytes())
    n = len(arrays[common.D8_SERIES]["resid"])
    cal = local_calendar(n=n)
    system, gas = prices()
    rng = np.random.default_rng(SEED)
    window = (cal["date"] >= WINDOW_START) & (cal["date"] <= WINDOW_END)
    window_np = window.to_numpy()
    hour = cal["hour"].to_numpy()

    # Step 1: shape of the shortfall at D8 and at D3, D14, D5.
    say("== Step 1: drop shape ==")
    shapes: dict[int, pl.DataFrame] = {}
    drops_by_series: dict[int, list[tuple[int, int]]] = {}
    for series, label in common.SERIES_LABELS.items():
        resid = arrays[series]["resid"]
        depth = -resid
        drops = [d for d in find_drops(residual=resid) if window_np[d[0]]]
        drops_by_series[series] = drops
        rows = [drop_shape(depth=depth, first=a, last=b, hour=hour) for a, b in drops]
        frame = pl.DataFrame(rows) if rows else pl.DataFrame()
        shapes[series] = frame
        if not rows:
            say(f"{label}: no drops in the window")
            continue
        flat = [r["rel_iqr"] < FLAT_LIMIT for r in rows]
        say(
            f"{label}: {len(rows)} drops since {WINDOW_START}; median depth "
            f"{frame['median_depth'].median():.2f} MW; flat blocks (relative interquartile range of "
            f"the 3-hour median depth under {FLAT_LIMIT}): {share_text(k=sum(flat), n=len(rows))}; "
            f"median relative range {frame['rel_iqr'].median():.2f}; sensitivity under 0.35: "
            f"{sum(r['rel_iqr'] < 0.35 for r in rows)}, under 0.50: "
            f"{sum(r['rel_iqr'] < 0.5 for r in rows)}; median share of half-hours within +/-25% of "
            f"the drop median: {frame['flat_share'].median():.2f}; median share of variance "
            f"explained by hour of day: {frame['hour_r2'].median():.2f}; median start step / depth "
            f"{frame['start_step'].median():.2f}, end step / depth {frame['end_step'].median():.2f}"
        )
    shapes[common.D8_SERIES].write_parquet(common.WORK_DIR / "d8_shapes.parquet")

    d8 = arrays[common.D8_SERIES]
    d8_drops = drops_by_series[common.D8_SERIES]
    drops_frame = pl.DataFrame(
        {
            "drop": range(1, len(d8_drops) + 1),
            "first": [a for a, _ in d8_drops],
            "last": [b for _, b in d8_drops],
            "start_date": [cal["date"][a] for a, _ in d8_drops],
            "end_date": [cal["date"][b - 1] for _, b in d8_drops],
            "start_hour": [cal["hour"][a] for a, _ in d8_drops],
        }
    )
    drops_frame.write_parquet(common.WORK_DIR / "d8_drops_in_window.parquet")

    # Positive control for the flat-block test: add a constant 2.8 MW block to ordinary windows.
    say("\n== Control: flat-block test on injected constant blocks ==")
    d8_depth = -d8["resid"]
    mask_drop = np.zeros(n, dtype=bool)
    for a_, b_ in drops_by_series[common.D8_SERIES]:
        mask_drop[max(a_ - 96, 0) : b_ + 96] = True
    lengths = [b_ - a_ for a_, b_ in drops_by_series[common.D8_SERIES]]
    starts = np.flatnonzero(window_np & ~mask_drop)
    injected = []
    for length in lengths:
        for _ in range(20):
            first = int(rng.choice(starts))
            if first + length >= n or mask_drop[first : first + length].any():
                continue
            fake = d8_depth.copy()
            fake[first : first + length] += 2.8
            r = drop_shape(depth=fake, first=first, last=first + length, hour=hour)
            injected.append(r["rel_iqr"])
    injected_arr = np.array(injected)
    say(
        f"Injected constant 2.8 MW blocks in ordinary windows of the drops' lengths "
        f"({len(injected_arr)} windows): median relative range {np.nanmedian(injected_arr):.2f}; "
        f"flat (under {FLAT_LIMIT}) in {(injected_arr < FLAT_LIMIT).mean():.2f}; under 0.50 in "
        f"{(injected_arr < 0.5).mean():.2f}"
    )
    say(
        "Control percentiles of the relative range (10, 25, 50, 75, 90): "
        + ", ".join(f"{v:.2f}" for v in np.nanpercentile(injected_arr, [10, 25, 50, 75, 90]))
    )
    d8_rel = shapes[common.D8_SERIES]["rel_iqr"].to_numpy()
    say(
        "D8 drops' relative range, same percentiles: "
        + ", ".join(f"{v:.2f}" for v in np.nanpercentile(d8_rel, [10, 25, 50, 75, 90]))
        + f"; drops under the control's median ({np.nanmedian(injected_arr):.2f}): "
        f"{int((d8_rel < np.nanmedian(injected_arr)).sum())} of {len(d8_rel)}; under the control's "
        f"75th percentile: {int((d8_rel < np.nanpercentile(injected_arr, 75)).sum())}"
    )
    pl.DataFrame(
        {
            "kind": ["control"] * len(injected_arr) + ["drop"] * len(d8_rel),
            "rel_iqr": [*injected_arr.tolist(), *d8_rel.tolist()],
        }
    ).write_parquet(common.WORK_DIR / "flatness.parquet")
    # Comparator series: runs under each series' own 5th percentile of the 24-hour median residual.
    for series in (3, 16, 17):
        resid = arrays[series]["resid"]
        smooth = rolling_median(values=resid, width=SMOOTHING_WINDOW)
        cut = np.nanpercentile(smooth[window_np], 5)
        below = np.isfinite(smooth) & (smooth < cut) & window_np
        edges = np.flatnonzero(np.diff(np.r_[0, below.astype(int), 0]) != 0).reshape(-1, 2)
        runs = [(int(a_), int(b_)) for a_, b_ in edges if b_ - a_ >= MIN_DURATION]
        rows = [drop_shape(depth=-resid, first=a_, last=b_, hour=hour) for a_, b_ in runs]
        if rows:
            fr = pl.DataFrame(rows)
            say(
                f"{common.SERIES_LABELS[series]} comparator (runs of 2 days or more under the series' "
                f"own 5th percentile, {cut:.2f} MW): {len(rows)} runs; median depth "
                f"{fr['median_depth'].median():.2f} MW; median relative range {fr['rel_iqr'].median():.2f}; "
                f"flat {sum(r['rel_iqr'] < FLAT_LIMIT for r in rows)}; median hour-of-day R-squared "
                f"{fr['hour_r2'].median():.2f}"
            )

    # Step 2: spark spread table.
    say("\n== Step 2: spark spread ==")
    days = day_table(cal=cal, series=d8, drops=d8_drops)
    days = days.filter((pl.col("date") >= WINDOW_START) & (pl.col("date") <= WINDOW_END))
    days = days.filter(pl.col("depth").is_not_nan() & pl.col("temp").is_not_nan())
    table = add_spread(days=days, system=system, gas=gas)
    table = table.with_columns(
        const_gas_spread_40=pl.col("price_mean")
        - pl.col("gas_gbp_per_mwh_thermal").mean() / CENTRAL_EFFICIENCY
        - VARIABLE_COST
    )
    say(f"Days with price, gas price and D8 reading: {len(table)} of {len(days)} in the window")
    say(
        f"System price pounds/MWh: mean {table['price_mean'].mean():.1f}, median "
        f"{table['price_mean'].median():.1f}; gas SAP pounds/MWh thermal: mean "
        f"{table['gas_gbp_per_mwh_thermal'].mean():.1f}"
    )
    for eff in EFFICIENCIES:
        tag = f"{round(eff * 100)}"
        pos = (table[f"spread_{tag}"] > 0).mean()
        pos_eve = (table[f"spread_eve_{tag}"] > 0).mean()
        say(
            f"Efficiency {eff:.0%}: gas cost per MWh electric {table['gas_gbp_per_mwh_thermal'].mean() / eff:.1f} "  # ty: ignore[unsupported-operator]
            f"+ 5.0 variable cost; share of all days with positive daily-mean spread {pos:.2f}; "
            f"evening (16-19) {pos_eve:.2f}"
        )
    table.write_parquet(common.WORK_DIR / "d8_day_table.parquet")

    # Step 3: drop days against matched ordinary days.
    say("\n== Step 3: drop days against matched ordinary days ==")
    drop_day_rows = table.filter(pl.col("drop_share") >= 0.75)
    date_to_drop = {}
    for k, row in enumerate(drops_frame.iter_rows(named=True)):
        d = row["start_date"]
        while d <= row["end_date"]:
            date_to_drop[d] = k
            d += timedelta(days=1)
    drop_day_rows = drop_day_rows.filter(pl.col("date").is_in(list(date_to_drop)))
    drop_of_day = np.array([date_to_drop[d] for d in drop_day_rows["date"]])
    n_drops_used = len(set(drop_of_day.tolist()))
    near = set()
    for d in date_to_drop:
        for off in range(-2, 3):
            near.add(d + timedelta(days=off))
    say(
        f"Drop days (>= 75% of the day inside a drop): {len(drop_day_rows)} in {n_drops_used} drops"
    )
    pools, ordinary = matched_pool(table=table, drop_days=drop_day_rows, in_any_drop_pad=near)
    say(
        f"Ordinary pool: {len(ordinary)} days; matches per drop day: median "
        f"{int(np.median([len(p) for p in pools]))}, min {min(len(p) for p in pools)}"
    )
    remap = {k: i for i, k in enumerate(sorted(set(drop_of_day.tolist())))}
    drop_idx = np.array([remap[k] for k in drop_of_day])
    price_cut = float(table["price_mean"].quantile(2 / 3))  # ty: ignore[invalid-argument-type]
    stats_spec: dict[str, tuple[str, float | None]] = {}
    for eff in EFFICIENCIES:
        tag = f"{round(eff * 100)}"
        stats_spec[f"positive daily spread, efficiency {eff:.0%}"] = (f"spread_{tag}", 0.0)
        stats_spec[f"positive evening spread, efficiency {eff:.0%}"] = (f"spread_eve_{tag}", 0.0)
    stats_spec[
        "positive daily spread with the gas price held at its mean, efficiency 40% (price alone)"
    ] = (
        "const_gas_spread_40",
        0.0,
    )
    stats_spec[f"system price alone in its top third (> {price_cut:.0f} pounds/MWh)"] = (
        "price_mean",
        price_cut,
    )
    results = {}
    rows3: list[dict] = []
    for name, (col, cut) in stats_spec.items():
        stat = (drop_day_rows[col] > cut).cast(pl.Float64).to_numpy()
        ordinary_stat = (ordinary[col] > cut).cast(pl.Float64).to_numpy()
        obs, mat, diff, diffs = matched_test(
            rng=rng, stat=stat, ordinary_stat=ordinary_stat, pools=pools,
            drop_of_day=drop_idx, n_drops=n_drops_used,
        )  # fmt: skip
        results[name] = diffs
        lo, hi = np.percentile(diffs * 100, [2.5, 97.5])
        rows3.append(
            {
                "name": name,
                "drop_days": obs,
                "matched": mat,
                "diff_points": diff * 100,
                "lo": lo,
                "hi": hi,
            }
        )
        say(
            f"{name}: drop days {obs:.2f}, matched ordinary days {mat:.2f}, difference "
            f"{interval(diffs * 100)} points"
        )
    for name, col in (
        ("system price (pounds/MWh)", "price_mean"),
        ("gas price (pounds/MWh thermal)", "gas_gbp_per_mwh_thermal"),
        ("daily spread at 40% efficiency (pounds/MWh)", "spread_40"),
        ("D8 day-mean ERA5 temperature (degrees C)", "temp"),
    ):
        stat = drop_day_rows[col].to_numpy()
        ordinary_stat = ordinary[col].to_numpy()
        obs, mat, diff, diffs = matched_test(
            rng=rng, stat=stat, ordinary_stat=ordinary_stat, pools=pools,
            drop_of_day=drop_idx, n_drops=n_drops_used,
        )  # fmt: skip
        say(f"{name}: drop days {obs:.2f}, matched {mat:.2f}, difference {interval(diffs)}")

    pl.DataFrame(rows3).write_parquet(common.WORK_DIR / "step3_results.parquet")

    # Half-hourly depth against half-hourly spread, inside the drops.
    say("\n== Step 3b: half-hourly depth inside drops, spread positive against not ==")
    cal_stamped = cal.with_columns(stamp=pl.int_range(pl.len()), depth=pl.Series(-d8["resid"]))
    half = (
        cal_stamped.select("stamp", "date", "sp", "depth")
        .join(system, on=["date", "sp"], how="inner")
        .join(gas.select("date", "gas_gbp_per_mwh_thermal"), on="date", how="inner")
    )
    for eff in EFFICIENCIES:
        diffs_by_drop = []
        for a, b in d8_drops:
            part = half.filter((pl.col("stamp") >= a) & (pl.col("stamp") < b)).with_columns(
                pos=(pl.col("price") - pl.col("gas_gbp_per_mwh_thermal") / eff - VARIABLE_COST) > 0
            )
            part = part.with_columns(
                hour_bin=((pl.col("sp") - 1) // 2),
            ).with_columns(depth_adj=pl.col("depth") - pl.col("depth").mean().over("hour_bin"))
            up = part.filter(pl.col("pos"))["depth_adj"]
            down = part.filter(~pl.col("pos"))["depth_adj"]
            if len(up) >= 12 and len(down) >= 12:
                diffs_by_drop.append(float(up.mean() - down.mean()))  # ty: ignore[unsupported-operator]
        arr = np.array(diffs_by_drop)
        boot = np.array([rng.choice(arr, len(arr)).mean() for _ in range(N_DRAWS)])
        say(
            f"Efficiency {eff:.0%}: depth (after removing the drop's own hour-of-day mean) in positive-spread half-hours minus other half-hours, "
            f"mean over {len(arr)} drops {arr.mean():+.2f} MW (95% {np.percentile(boot, 2.5):+.2f} to "
            f"{np.percentile(boot, 97.5):+.2f}); higher in {int((arr > 0).sum())} of {len(arr)} drops"
        )

    # Event study: spread in the 5 days before and after each drop start and end.
    say("\n== Step 3c: spread around drop starts and ends (daily spread, 40% efficiency) ==")
    spread = dict(zip(table["date"], table["spread_40"], strict=True))
    for label, key, sign in (("start", "start_date", 1), ("end", "end_date", -1)):
        diffs_ev = []
        for row in drops_frame.iter_rows(named=True):
            d0 = row[key]
            before = [spread.get(d0 - timedelta(days=k)) for k in range(1, 6)]
            after = [spread.get(d0 + timedelta(days=k)) for k in range(5)]
            before = [v for v in before if v is not None]
            after = [v for v in after if v is not None]
            if len(before) >= 3 and len(after) >= 3:
                diffs_ev.append(sign * (np.mean(after) - np.mean(before)))
        arr = np.array(diffs_ev)
        boot = np.array([rng.choice(arr, len(arr)).mean() for _ in range(N_DRAWS)])
        say(
            f"Change in the direction the hypothesis expects at the drop {label} "
            f"({'rise' if sign == 1 else 'fall'} of the mean spread over 5 days after versus 5 before): "
            f"mean {arr.mean():+.1f} pounds/MWh over {len(arr)} drops (95% "
            f"{np.percentile(boot, 2.5):+.1f} to {np.percentile(boot, 97.5):+.1f}); "
            f"in the predicted direction in {int((arr > 0).sum())} of {len(arr)}"
        )

    prof = []
    for label, key in (("start", "start_date"), ("end", "end_date")):
        for off in range(-10, 11):
            vals = [
                spread.get(row[key] + timedelta(days=off))
                for row in drops_frame.iter_rows(named=True)
            ]
            vals = [v for v in vals if v is not None]
            prof.append(
                {
                    "anchor": label,
                    "offset": off,
                    "mean_spread": float(np.mean(vals)),
                    "n": len(vals),
                }
            )
    pl.DataFrame(prof).write_parquet(common.WORK_DIR / "event_profile.parquet")

    # Step 4: triads.
    say("\n== Step 4: triads ==")
    stamp_of = {
        (r["date"], r["sp"]): r["stamp"]
        for r in cal_stamped.select("stamp", "date", "sp").iter_rows(named=True)
    }
    in_drop = np.zeros(n, dtype=bool)
    for a, b in d8_drops:
        in_drop[a:b] = True
    inside = []
    for d, sp in TRIADS:
        s = stamp_of[(d, sp)]
        inside.append(bool(in_drop[s]))
        day_stamps = [stamp_of[(d, p)] for p in range(1, 49)]
        say(
            f"Triad {d} period {sp}: inside a D8 drop {in_drop[s]}; D8 depth at the Triad "
            f"{-d8['resid'][s]:+.2f} MW against the day's median {np.nanmedian(-d8['resid'][day_stamps]):+.2f} MW"
        )
    winter_weekdays = table.filter(pl.col("month").is_in([11, 12, 1, 2]) & (pl.col("weekday") < 5))
    base_share = float((winter_weekdays["drop_share"] >= 0.5).mean())  # ty: ignore[invalid-argument-type]
    flags = (winter_weekdays["drop_share"] >= 0.5).to_numpy()
    k_obs = sum(inside)
    draws = np.array([rng.choice(flags, len(TRIADS), replace=False).sum() for _ in range(10000)])
    say(
        f"Triad half-hours inside a drop: {k_obs} of {len(TRIADS)}; share of November-to-February "
        f"weekdays in a drop {base_share:.2f}; chance of >= {k_obs} by random winter weekdays "
        f"{(draws >= k_obs).mean():.3f}"
    )
    # Evening-peak depth in winter drops.
    evening_rows = []
    for a, b in d8_drops:
        seg = cal_stamped.slice(a, b - a).filter(
            pl.col("month").is_in([11, 12, 1, 2]) & (pl.col("weekday") < 5)
        )
        if len(seg) < 48:
            continue
        peak = seg.filter((pl.col("hour") >= 16) & (pl.col("hour") < 19))["depth"].mean()
        rest = seg.filter(~((pl.col("hour") >= 16) & (pl.col("hour") < 19)))["depth"].mean()
        evening_rows.append(float(peak - rest))  # ty: ignore[unsupported-operator]
    arr = np.array(evening_rows)
    if len(arr):
        boot = np.array([rng.choice(arr, len(arr)).mean() for _ in range(N_DRAWS)])
        say(
            f"Winter weekday depth 16:00-19:00 minus other hours inside drops: mean {arr.mean():+.2f} MW "
            f"over {len(arr)} drops with winter weekdays (95% {np.percentile(boot, 2.5):+.2f} to "
            f"{np.percentile(boot, 97.5):+.2f}); deeper in {int((arr > 0).sum())}"
        )
    winter_out = cal_stamped.filter(
        pl.col("month").is_in([11, 12, 1, 2])
        & (pl.col("weekday") < 5)
        & ~pl.Series(in_drop)
        & (pl.col("date") >= WINDOW_START)
    )
    peak_out = winter_out.filter((pl.col("hour") >= 16) & (pl.col("hour") < 19))["depth"].mean()
    rest_out = winter_out.filter(~((pl.col("hour") >= 16) & (pl.col("hour") < 19)))["depth"].mean()
    say(f"Same contrast on winter weekdays outside drops: {peak_out - rest_out:+.2f} MW")  # ty: ignore[unsupported-operator]
    ww = cal_stamped.filter(
        pl.col("month").is_in([11, 12, 1, 2])
        & (pl.col("weekday") < 5)
        & (pl.col("date") >= WINDOW_START)
        & ~pl.Series(in_drop)
        & pl.col("depth").is_not_nan()
    )
    triad_excess = []
    for d, sp in TRIADS:
        ref = ww.filter((pl.col("sp") == sp) & (pl.col("date") != d))["depth"]
        triad_excess.append(float(-d8["resid"][stamp_of[(d, sp)]] - ref.mean()))
    arr = np.array(triad_excess)
    boot = np.array([rng.choice(arr, len(arr)).mean() for _ in range(N_DRAWS)])
    say(
        f"Depth at the Triad half-hour minus the mean depth at the same period on other winter "
        f"weekdays outside drops: mean {arr.mean():+.2f} MW over {len(arr)} Triads (95% "
        f"{np.percentile(boot, 2.5):+.2f} to {np.percentile(boot, 97.5):+.2f}); positive in "
        f"{int((arr > 0).sum())}"
    )
    # Drops that start within 3 days before a triad.
    for k, row in enumerate(drops_frame.iter_rows(named=True), start=1):
        for d, sp in TRIADS:
            if row["start_date"] - timedelta(days=3) <= d <= row["end_date"]:
                say(
                    f"Drop {k} ({row['start_date']} to {row['end_date']}) covers Triad {d} period {sp}"
                )

    # Daily association with controls.
    say("\n== Step 3d: all days, depth against spread, with temperature and calendar controls ==")
    import numpy.linalg as la

    def design(extra: list[str]) -> np.ndarray:
        cols = [np.ones(len(table))]
        doy = table["doy"].to_numpy()
        for k in (1, 2):
            cols.append(np.sin(2 * np.pi * k * doy / 365.25))
            cols.append(np.cos(2 * np.pi * k * doy / 365.25))
        cols.extend((table["weekday"].to_numpy() == w).astype(float) for w in range(1, 7))
        t = table["temp"].to_numpy()
        cols += [t, t**2]
        for c in extra:
            cols.append(table[c].to_numpy())
        return np.column_stack(cols)

    y = table["depth"].to_numpy()
    months = (table["date"].dt.year() * 12 + table["date"].dt.month()).to_numpy()
    uniq = np.unique(months)

    def oof_r2(extra: list[str]) -> float:
        x = design(extra)
        pred = np.empty(len(y))
        for m in uniq:
            test = months == m
            coef, *_ = la.lstsq(x[~test], y[~test], rcond=None)
            pred[test] = x[test] @ coef
        return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    base = oof_r2([])
    say(f"Out-of-fold R-squared (leave one month out): controls only {base:.3f}")
    for name, extra in (
        ("+ system price", ["price_mean"]),
        ("+ gas price", ["gas_gbp_per_mwh_thermal"]),
        ("+ spread 40%", ["spread_40"]),
        ("+ price and gas price", ["price_mean", "gas_gbp_per_mwh_thermal"]),
    ):
        say(f"  {name}: gain {oof_r2(extra) - base:+.4f}")

    (common.WORK_DIR / "printed_report.txt").write_text("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
