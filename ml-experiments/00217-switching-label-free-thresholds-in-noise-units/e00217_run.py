# ruff: noqa: E501, D103, PLR0915
"""Alarm rates of RO-NC and the pair alarm at fixed thresholds in noise units, and residual statistics.

No switching label, logged closure, or labelled transfer is read. Inputs are experiment 130's stored
frozen convex-model residuals (arm B; the model refitted on the 1st and on the 16th of each month)
and experiment 195's stored learned-score alarms. Writes tables under this experiment's private
folder and prints the numbers the write-up quotes.

    OMP_NUM_THREADS=2 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00217-*/e00217_run.py
"""

from __future__ import annotations

import pickle
from datetime import UTC, date, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
from studies.dp_calendar import bank_holiday_names
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00217-switching-label-free-thresholds-in-noise-units"
E130_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00130-ro-nc-detector-on-the-gam-k-residual"
E195_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00195-switching-neural-plus-detector-features"
SERIES_PATH: Final = (
    ML_EXPERIMENTS_DATA_DIR.parent / "studies/per_study/stage1_pilot/demand/inputs/series.parquet"
)
GRID_START: Final = datetime(2022, 1, 1, tzinfo=UTC)
GRID_LENGTH: Final = int((datetime(2026, 7, 1, tzinfo=UTC) - GRID_START).total_seconds() // 1800)
DAY: Final = 48
SERIES_IDS: Final = (3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 24, 25, 26)
"""Series with a convex-model residual (series 12 has none)."""
ADJACENCY: Final = {
    3: (15, 16, 26), 7: (24,), 8: (10, 14, 24), 9: (11, 24), 10: (8, 12, 13, 14, 24, 25),
    11: (9, 25), 12: (10, 13, 14, 16, 25), 13: (10, 12, 14, 16, 25), 14: (8, 10, 12, 13, 15, 16),
    15: (3, 14, 17, 18), 16: (3, 12, 13, 14, 26), 17: (15, 18, 26), 18: (15, 17),
    24: (7, 8, 9, 10, 25), 25: (10, 11, 12, 13, 24), 26: (3, 16, 17),
}  # fmt: skip
BALANCE_RANGE: Final = (0.4, 2.5)
MIN_PRESENT: Final = 0.75
RONC_WINDOW: Final = 48
RONC_Z: Final = 4.35
"""Experiment 130's stored threshold for the refit-offset step statistic."""
PAIR_WINDOW: Final = 8
PAIR_K: Final = 3.75
"""Experiment 156's matched threshold for the 4-hour window."""
WINDOWS: Final = (4, 8, 12, 48)
"""2, 4, 6, and 24 hours in half-hours."""
MISFIT_Z: Final = 3.0
MIN_MONTHS: Final = 12
BOOT: Final = 2000


def months_of_stamps() -> tuple[np.ndarray, np.ndarray]:
    """UTC calendar month (1-12) and a year-month label of each grid stamp."""
    index = pl.datetime_range(
        GRID_START, datetime(2026, 7, 1, tzinfo=UTC), interval="30m", closed="left",
        time_zone="UTC", eager=True,
    )  # fmt: skip
    return index.dt.month().to_numpy(), (index.dt.year() * 100 + index.dt.month()).to_numpy()


MONTH, YEAR_MONTH = months_of_stamps()
ODD = np.isin(MONTH, [1, 3, 5, 7, 9, 11])
WINTER = np.isin(MONTH, [12, 1, 2])


def level_shift(*, values: np.ndarray, window: int) -> np.ndarray:
    """Mean of the next `window` stamps minus mean of the previous `window`; NaN if under 75% present."""
    total = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values))])
    count = np.concatenate([[0], np.cumsum(np.isfinite(values))])
    t = np.arange(GRID_LENGTH)
    lo, hi = np.clip(t - window, 0, GRID_LENGTH), np.clip(t + window, 0, GRID_LENGTH)
    n_before, n_after = count[t] - count[lo], count[hi] - count[t]
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = (total[hi] - total[t]) / n_after - (total[t] - total[lo]) / n_before
    shift[(n_before < MIN_PRESENT * window) | (n_after < MIN_PRESENT * window)] = np.nan
    return shift


def refit_zone() -> np.ndarray:
    """True within 24 hours of 00:00 UTC on the 1st, where the first model's statistic is unusable."""
    zone = np.zeros(GRID_LENGTH, dtype=bool)
    first_of_month = np.flatnonzero(np.diff(YEAR_MONTH, prepend=0) != 0)
    for stamp in first_of_month:
        zone[max(stamp - DAY + 1, 0) : stamp + DAY] = True
    return zone


def local_peaks(*, z: np.ndarray, threshold: float, window: int) -> np.ndarray:
    """Stamps where |z| exceeds `threshold` and is the largest within +/- `window`."""
    magnitude = np.nan_to_num(np.abs(z))
    peaks = []
    for t in np.flatnonzero(magnitude > threshold):
        low, high = max(t - window, 0), min(t + window + 1, GRID_LENGTH)
        if magnitude[t] == magnitude[low:high].max():
            peaks.append(t)
    return np.array(peaks, dtype=int)


def mad_scale(*, x: np.ndarray, mask: np.ndarray | None = None) -> float:
    """1.4826 times the median absolute deviation of the finite values of `x` (in `mask`)."""
    use = np.isfinite(x) if mask is None else np.isfinite(x) & mask
    v = x[use]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def q95_scale(*, x: np.ndarray) -> float:
    """95th percentile of |x| divided by 1.96, so heavy tails raise it and a Gaussian leaves it."""
    return float(np.nanquantile(np.abs(x), 0.95) / 1.96)


def excess_kurtosis(*, x: np.ndarray) -> float:
    v = x[np.isfinite(x)]
    v = v - v.mean()
    return float((v**4).mean() / (v**2).mean() ** 2 - 3.0)


def ronc_alarms(*, shift: dict[int, np.ndarray], scale: dict[int, float], k: float) -> pl.DataFrame:
    """Peaks of the 24-hour step statistic above `k` noise units."""
    rows = [
        {"series": s, "stamp": int(t)}
        for s, x in shift.items()
        for t in local_peaks(z=x / scale[s], threshold=k, window=RONC_WINDOW)
    ]
    return pl.DataFrame(rows, schema={"series": pl.Int64, "stamp": pl.Int64})


def pair_alarms(*, shift: dict[int, np.ndarray], scale: dict[int, float], k: float) -> pl.DataFrame:
    """Experiment 156's pair alarm: a peak plus an opposite, balancing neighbour step, few same-signs."""
    half = PAIR_WINDOW // 2
    rows = []
    for s, x in shift.items():
        neighbours = [n for n in ADJACENCY.get(s, ()) if n in shift]
        for t in local_peaks(z=x / scale[s], threshold=k, window=half):
            sign = np.sign(x[t])
            partners, same = [], 0
            for n in neighbours:
                seg = shift[n][max(t - 1, 0) : t + 2] / scale[n]
                if not np.isfinite(seg).any():
                    continue
                if (-sign * seg).max() >= k / 2:
                    partners.append(shift[n][max(t - 1, 0) + int(np.nanargmax(-sign * seg))])
                elif (sign * seg).max() >= k / 2:
                    same += 1
            if not partners or same >= min(2, len(neighbours)):
                continue
            if BALANCE_RANGE[0] <= -sum(partners) / x[t] <= BALANCE_RANGE[1]:
                rows.append({"series": s, "stamp": int(t)})
    return pl.DataFrame(rows, schema={"series": pl.Int64, "stamp": pl.Int64})


def load_residuals() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    gam = E130_DIR / "gam"
    first = pickle.loads((gam / "panel_1.pkl").read_bytes())["frozen"]["residual"]
    second = pickle.loads((gam / "panel_16.pkl").read_bytes())["frozen"]["residual"]
    return {s: first[s] for s in SERIES_IDS}, {s: second[s] for s in SERIES_IDS}


def months_with_data(*, x: np.ndarray) -> int:
    """Calendar months in which at least 75% of the half-hours hold a finite value."""
    ok = 0
    for ym in np.unique(YEAR_MONTH):
        sel = ym == YEAR_MONTH
        ok += int(np.isfinite(x[sel]).mean() >= 0.75)
    return ok


def holiday_days() -> set[date]:
    names = bank_holiday_names()
    days = set(names)
    days |= {d + timedelta(days=i) for d, n in names.items() if n == "Good Friday" for i in (1, 2)}
    return days


def misfit_days(*, residual: np.ndarray, unit: float) -> tuple[np.ndarray, np.ndarray]:
    """Day index array and a flag: |day-mean residual| above MISFIT_Z noise units (>=75% present)."""
    n_days = GRID_LENGTH // DAY
    r = residual[: n_days * DAY].reshape(n_days, DAY)
    present = np.isfinite(r).mean(axis=1) >= MIN_PRESENT
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(r, axis=1)
    return present, present & (np.abs(mean) > MISFIT_Z * unit)


def block_counts(
    *, stamps: np.ndarray, exposure: np.ndarray, groups: dict[str, np.ndarray]
) -> dict:
    """Per (group, year-month): alarm count and exposure in half-hours."""
    out = {}
    months = np.unique(YEAR_MONTH)
    for g, gmask in groups.items():
        a = np.zeros(len(months))
        e = np.zeros(len(months))
        for i, ym in enumerate(months):
            sel = (ym == YEAR_MONTH) & gmask
            e[i] = exposure[sel].sum()
            if len(stamps):
                a[i] = ((YEAR_MONTH[stamps] == ym) & gmask[stamps]).sum()
        out[g] = (a, e)
    return out


YEAR_HALF_HOURS: Final = 365.25 * DAY


def rates_table(
    *, alarms: pl.DataFrame, exposure: dict[int, np.ndarray], label: str
) -> pl.DataFrame:
    """Alarms per Primary-year (all, winter, other) with whole-month bootstrap intervals."""
    rng = np.random.default_rng(0)
    groups = {"all": np.ones(GRID_LENGTH, bool), "winter": WINTER, "other": ~WINTER}
    months = np.unique(YEAR_MONTH)
    draws = rng.integers(0, len(months), size=(BOOT, len(months)))
    rows = []
    for s, series_exposure in exposure.items():
        st = alarms.filter(pl.col("series") == s)["stamp"].to_numpy()
        blocks = block_counts(stamps=st, exposure=series_exposure, groups=groups)
        for g, (a, e) in blocks.items():
            point = a.sum() / e.sum() * YEAR_HALF_HOURS if e.sum() else np.nan
            ea, ee = a[draws].sum(axis=1), e[draws].sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = ea / ee * YEAR_HALF_HOURS
            rows.append({
                "detector": label, "series": s, "group": g, "alarms": int(a.sum()),
                "years": float(e.sum() / YEAR_HALF_HOURS), "rate": float(point),
                "lo": float(np.nanquantile(r, 0.025)), "hi": float(np.nanquantile(r, 0.975)),
            })  # fmt: skip
    return pl.DataFrame(rows)


def main() -> None:
    sites = {
        int(r["time_series_id"]): r["site"]
        for r in pl.read_parquet(SERIES_PATH).iter_rows(named=True)
    }
    first, second = load_residuals()
    zone = refit_zone()
    shifts = {
        w: {
            s: np.where(
                zone,
                level_shift(values=second[s], window=w),
                level_shift(values=first[s], window=w),
            )
            for s in SERIES_IDS
        }
        for w in WINDOWS
    }
    keep = [s for s in SERIES_IDS if months_with_data(x=shifts[RONC_WINDOW][s]) >= MIN_MONTHS]
    print("series kept (>= 12 months with 75% residual):", [sites[s] for s in keep])
    print("dropped:", [sites[s] for s in SERIES_IDS if s not in keep])
    shifts = {w: {s: d[s] for s in keep} for w, d in shifts.items()}
    exposure = {s: np.isfinite(shifts[RONC_WINDOW][s]).astype(float) for s in keep}

    # Noise estimates per series and window.
    cur = {w: {s: mad_scale(x=shifts[w][s], mask=ODD) for s in keep} for w in WINDOWS}
    r1 = {w: {s: mad_scale(x=shifts[w][s]) for s in keep} for w in WINDOWS}
    r2 = {w: {s: q95_scale(x=shifts[w][s]) for s in keep} for w in WINDOWS}
    estimates = {"current": cur, "R1": r1, "R2": r2}

    tables, alarm_tables = [], {}
    for est, scales in estimates.items():
        ro = ronc_alarms(shift=shifts[RONC_WINDOW], scale=scales[RONC_WINDOW], k=RONC_Z)
        pa = pair_alarms(shift=shifts[PAIR_WINDOW], scale=scales[PAIR_WINDOW], k=PAIR_K)
        alarm_tables[("RO-NC", est)], alarm_tables[("pair", est)] = ro, pa
        tables += [
            rates_table(alarms=ro, exposure=exposure, label=f"RO-NC/{est}"),
            rates_table(alarms=pa, exposure=exposure, label=f"pair/{est}"),
        ]
    # Learned score: stored alarms of experiment 195, mean over the three folds.
    for arm in ("R re-ranker", "F logistic"):
        parts = []
        for fold in range(3):
            a = pl.read_parquet(E195_DIR / f"alarms_f{fold}.parquet").filter(pl.col("arm") == arm)
            parts.append(a.unique(["series", "stamp", "source"]).select("series", "stamp"))
        merged = pl.concat([p.with_columns(f=pl.lit(i)) for i, p in enumerate(parts)])
        per_fold = [
            rates_table(alarms=p.filter(pl.col("series").is_in(keep)), exposure=exposure, label=arm)
            for p in parts
        ]
        mean = (
            pl.concat(per_fold).group_by(["detector", "series", "group"])
            .agg(alarms=pl.col("alarms").mean(), years=pl.col("years").first(), rate=pl.col("rate").mean(),
                 lo=pl.col("lo").mean(), hi=pl.col("hi").mean())
        )  # fmt: skip
        tables.append(mean.select(per_fold[0].columns))
        alarm_tables[("learned " + arm, "stored")] = merged
    stored_xl = pl.read_parquet(E130_DIR / "B" / "alarms.parquet").filter(
        (pl.col("setting") == "XL") & pl.col("series").is_in(keep)
    )
    tables.append(rates_table(alarms=stored_xl, exposure=exposure, label="RO-NC XL stored"))
    rates = pl.concat(
        [t.with_columns(pl.col("alarms").cast(pl.Float64)) for t in tables]
    ).with_columns(site=pl.col("series").replace_strict(sites, return_dtype=pl.String))
    rates.write_parquet(WORK_DIR / "rates.parquet")

    # Residual statistics.
    holidays_set = holiday_days()
    day0 = GRID_START.date()
    stat_rows = []
    for s in keep:
        row = {"series": s, "site": sites[s]}
        for w in WINDOWS:
            row[f"cur_{w}"], row[f"mad_all_{w}"], row[f"q95_{w}"] = cur[w][s], r1[w][s], r2[w][s]
        row["kurt_48"] = excess_kurtosis(x=shifts[48][s])
        row["kurt_8"] = excess_kurtosis(x=shifts[8][s])
        present, misfit = misfit_days(residual=first[s], unit=cur[48][s])
        days = np.array([day0 + timedelta(days=i) for i in range(len(present))])
        is_hol = np.array([d in holidays_set for d in days])
        row["misfit_share"] = float(misfit.sum() / present.sum())
        row["holiday_misfit_share"] = float(
            misfit[is_hol & present].sum() / max((is_hol & present).sum(), 1)
        )
        row["resid_mad"] = mad_scale(x=first[s])
        row["resid_sd"] = float(np.nanstd(first[s]))
        row["years"] = float(exposure[s].sum() / YEAR_HALF_HOURS)
        stat_rows.append(row)
    stats = pl.DataFrame(stat_rows)
    stats.write_parquet(WORK_DIR / "stats.parquet")
    for k_, v_ in alarm_tables.items():
        v_.write_parquet(WORK_DIR / f"alarms_{k_[0].replace(' ', '_')}_{k_[1]}.parquet")
    pl.Config.set_tbl_rows(100).set_tbl_cols(30).set_fmt_float("full").set_tbl_width_chars(220)
    show = rates.filter(pl.col("group") == "all").with_columns(
        pl.col("rate").round(2),
        pl.col("lo").round(2),
        pl.col("hi").round(2),
        pl.col("years").round(2),
    )
    for det in show["detector"].unique(maintain_order=True):
        print(det)
        print(
            show.filter(pl.col("detector") == det).select(
                "site", "alarms", "years", "rate", "lo", "hi"
            )
        )
    print(
        stats.select(
            "site",
            "years",
            "cur_48",
            "mad_all_48",
            "q95_48",
            "cur_8",
            "q95_8",
            "kurt_48",
            "kurt_8",
            "misfit_share",
            "holiday_misfit_share",
        ).with_columns(pl.all().exclude("site").round(3))
    )


if __name__ == "__main__":
    main()
