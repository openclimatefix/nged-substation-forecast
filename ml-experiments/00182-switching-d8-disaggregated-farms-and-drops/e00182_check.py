# ruff: noqa: E501
"""Compare D8's multi-day drops (experiment 159) with the readings of its two disaggregated farms.

Farms appear only as `farm 1` and `farm 2`. Reads the drops table of experiment 159, the stored D8
power and expected-power panel, and the raw (uncleaned) power table. Writes private tables under
`WORK_DIR` and prints every number the write-up quotes into `printed_report.txt`.

Run as `OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue
uv run --no-sync python e00182_check.py`.
"""

import pickle
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pvlib
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

SLUG: Final[str] = "00182-switching-d8-disaggregated-farms-and-drops"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
DROPS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00159-switching-d8-multi-day-drops-recurrence-and-report"
    / "drops.parquet"
)
ARRAYS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "briefs" / "sw_false_alarms" / "work" / "arrays.pkl"
)
RAW_POWER: Final[str] = str(REPO_DATA_DIR / "NGED" / "power_time_series.delta")
GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
D8: Final[int] = 26
FARMS: Final[dict[int, str]] = {32: "farm 1", 33: "farm 2"}
D8_LAT, D8_LON = 52.9616, -0.0136
DAY: Final[int] = 48
DAYLIGHT_W_M2: Final[float] = 50.0
"""Clear-sky irradiance above which a half-hour counts as daylight."""
STUCK_RUN: Final[int] = 6
MIN_PRESENT: Final[float] = 0.5
N_DRAWS: Final[int] = 1000
RANDOM_FROM: Final[int] = 546 * DAY
"""2023-07-01 as a stamp: the first month after which D8 has nearly all its drops."""
REPORT: list[str] = []


def say(text: str = "") -> None:
    """Print a line and keep it for `printed_report.txt`."""
    print(text, flush=True)
    REPORT.append(text)


def wilson(*, k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval for a share."""
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def classify(*, power: np.ndarray, daylight: np.ndarray, a: int, b: int) -> str:
    """Status of one farm over stamps [a, b): missing, zero, stuck, or normal (daylight only)."""
    sel = daylight[a:b]
    window = power[a:b][sel]
    if window.size == 0 or np.isfinite(window).mean() < MIN_PRESENT:
        return "missing"
    present = window[np.isfinite(window)]
    if present.max() <= 0:
        return "zero"
    run = 1
    for i in range(1, present.size):
        run = run + 1 if present[i] == present[i - 1] and present[i] > 0 else 1
        if run >= STUCK_RUN:
            return "stuck"
    return "normal"


def main() -> None:  # noqa: D103, PLR0915
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    arrays = pickle.loads(ARRAYS_PATH.read_bytes())
    d8 = arrays[D8]
    n = len(d8["power"])
    stamps = pl.datetime_range(
        GRID_START,
        GRID_START + timedelta(minutes=30 * (n - 1)),
        interval="30m",
        time_zone="UTC",
        eager=True,
    )
    import pandas as pd  # noqa: TID251  (pvlib takes a pandas index)

    mid = pd.DatetimeIndex((stamps - timedelta(minutes=15)).to_numpy(), tz="UTC")
    ghi = (
        pvlib.location.Location(D8_LAT, D8_LON)
        .get_clearsky(mid, model="ineichen")["ghi"]
        .to_numpy()
    )
    daylight = ghi > DAYLIGHT_W_M2
    local_hour = stamps.dt.convert_time_zone("Europe/London").dt.hour().to_numpy()

    raw = (
        pl.scan_delta(RAW_POWER)
        .filter(pl.col("time_series_id").is_in(list(FARMS)))
        .collect()
        .with_columns(stamp=((pl.col("time") - GRID_START) / timedelta(minutes=30)).cast(pl.Int64))
    )
    farm_power: dict[int, np.ndarray] = {}
    for tsid, label in FARMS.items():
        arr = np.full(n, np.nan)
        sub = raw.filter(pl.col("time_series_id") == tsid, pl.col("stamp").is_between(0, n - 1))
        arr[sub["stamp"].to_numpy()] = sub["power"].to_numpy()
        farm_power[tsid] = arr
        say(
            f"{label}: {sub.height} raw readings on the grid, {sub.filter(pl.col('power') > 0).height} above zero, "
            f"first {sub['time'].min()}, last {sub['time'].max()}"
        )
    avail = pl.DataFrame(
        {"time": stamps, **{f"f{t}": np.isfinite(a) for t, a in farm_power.items()}}
    )
    monthly = avail.group_by_dynamic("time", every="1mo").agg(
        [pl.col(f"f{t}").sum().alias(FARMS[t].replace(" ", "_")) for t in FARMS]
    )
    monthly.write_parquet(WORK_DIR / "farm_availability_monthly.parquet")
    pos = {t: np.where(np.isfinite(a) & (a > 0), a, np.nan) for t, a in farm_power.items()}
    for t in FARMS:
        say(
            f"{FARMS[t]}: positive readings {int(np.isfinite(pos[t]).sum())}, last positive "
            f"{stamps[int(np.where(np.isfinite(pos[t]))[0].max())] if np.isfinite(pos[t]).any() else 'none'}"
        )

    drops = pl.read_parquet(DROPS_PATH).sort("drop")
    rows = []
    for r in drops.iter_rows(named=True):
        a, b = r["first_stamp"], r["last_stamp"]
        pre_a = a - 7 * DAY
        status = {t: classify(power=farm_power[t], daylight=daylight, a=a, b=b) for t in FARMS}
        pre = {t: classify(power=farm_power[t], daylight=daylight, a=pre_a, b=a) for t in FARMS}
        # Combined farm output at the same half-hours of the day over the 7 days before, daylight only.
        idx = np.arange(pre_a, a)
        idx = idx[daylight[idx]]
        comb = np.nansum(np.vstack([farm_power[t][idx] for t in FARMS]), axis=0)
        anyp = np.isfinite(np.vstack([farm_power[t][idx] for t in FARMS])).any(axis=0)
        pre_median = float(np.median(comb[anyp])) if anyp.any() else float("nan")
        resid = (d8["expected"] - d8["power"])[a:b]
        rows.append(
            {
                "drop": r["drop"],
                "start_date": r["start_date"],
                "depth_mw": r["depth_mw"],
                "duration_days": r["duration_days"],
                "farm1_in_drop": status[32],
                "farm2_in_drop": status[33],
                "farm1_before": pre[32],
                "farm2_before": pre[33],
                "farms_pre7d_daylight_median_mw": pre_median,
                "pre7d_readings_present": int(anyp.sum()),
                "any_anomaly": any(s != "normal" for s in status.values()),
                "anomaly_not_before": any(
                    status[t] != "normal" and pre[t] == "normal" for t in FARMS
                ),
                "d8_midday_depth_mw": float(
                    np.nanmedian(resid[(local_hour[a:b] >= 10) & (local_hour[a:b] < 15)])
                ),
                "d8_night_depth_mw": float(
                    np.nanmedian(resid[(local_hour[a:b] >= 22) | (local_hour[a:b] < 4)])
                ),
            }
        )
    tab = pl.DataFrame(rows)
    tab.write_parquet(WORK_DIR / "drops_vs_farms.parquet")
    k = int(tab["any_anomaly"].sum())
    low, high = wilson(k=k, n=tab.height)
    say(
        f"\nStep 1: drops with a farm anomaly (missing, zero, or stuck in daylight): {k} of {tab.height} "
        f"({k / tab.height:.2f}; {low:.2f} to {high:.2f})"
    )
    for col in ("farm1_in_drop", "farm2_in_drop", "farm1_before", "farm2_before"):
        say(f"  {col}: {dict(sorted(tab[col].value_counts().iter_rows()))}")
    say(
        f"  anomaly in the drop but the farm normal in the 7 days before: {int(tab['anomaly_not_before'].sum())} of {tab.height}"
    )
    say(
        f"  drops with any farm reading in the 7 daylight days before: {int((tab['pre7d_readings_present'] > 0).sum())}; "
        f"median of farms' combined daylight output where present: "
        f"{tab.filter(pl.col('pre7d_readings_present') > 0)['farms_pre7d_daylight_median_mw'].median()}"
    )
    say(f"  drop depth: median {tab['depth_mw'].median():.2f} MW")

    # Step 2: random windows of the same lengths and time of day.
    lengths = (drops["last_stamp"] - drops["first_stamp"]).to_numpy()
    tod = (drops["first_stamp"] % DAY).to_numpy()
    last_start = n - 1 - lengths.max()
    counts, changed = np.zeros(N_DRAWS, int), np.zeros(N_DRAWS, int)
    for d in range(N_DRAWS):
        for ln, t in zip(lengths, tod, strict=True):
            day0 = rng.integers(RANDOM_FROM // DAY, last_start // DAY)
            a = int(day0 * DAY + t)
            st = {
                f: classify(power=farm_power[f], daylight=daylight, a=a, b=a + int(ln))
                for f in FARMS
            }
            pr = {
                f: classify(power=farm_power[f], daylight=daylight, a=a - 7 * DAY, b=a)
                for f in FARMS
            }
            counts[d] += any(s != "normal" for s in st.values())
            changed[d] += any(st[f] != "normal" and pr[f] == "normal" for f in FARMS)
    say(
        f"\nStep 2: random windows ({N_DRAWS} draws of {tab.height}): anomaly count mean {counts.mean():.1f} "
        f"(5th to 95th percentile {np.percentile(counts, 5):.0f} to {np.percentile(counts, 95):.0f}); "
        f"share of draws with count >= observed {k}: {(counts >= k).mean():.3f}; "
        f"anomaly-not-before count mean {changed.mean():.1f}, observed {int(tab['anomaly_not_before'].sum())}, "
        f"share of draws >= observed: {(changed >= tab['anomaly_not_before'].sum()).mean():.3f}"
    )
    pl.DataFrame({"anomaly_count": counts, "changed_count": changed}).write_parquet(
        WORK_DIR / "random_counts.parquet"
    )

    # Step 3: depth by local hour of day inside drops against outside, and against clear-sky irradiance.
    in_drop = np.zeros(n, bool)
    for a, b in zip(drops["first_stamp"], drops["last_stamp"], strict=True):
        in_drop[a:b] = True
    depth = d8["expected"] - d8["power"]
    after = np.arange(n) >= RANDOM_FROM
    hours = np.arange(24)
    prof = pl.DataFrame(
        {
            "hour": hours,
            "depth_in_drops_mw": [
                float(np.nanmedian(depth[in_drop & (local_hour == h)])) for h in hours
            ],
            "depth_outside_mw": [
                float(np.nanmedian(depth[~in_drop & after & (local_hour == h)])) for h in hours
            ],
            "clear_sky_ghi": [float(np.mean(ghi[in_drop & (local_hour == h)])) for h in hours],
        }
    ).with_columns(excess=pl.col("depth_in_drops_mw") - pl.col("depth_outside_mw"))
    prof.write_parquet(WORK_DIR / "hour_profile.parquet")
    from scipy import stats

    rho_p, p_p = stats.spearmanr(prof["excess"], prof["clear_sky_ghi"])
    say(
        f"\nStep 3: hour-of-day profile of expected-minus-metered MW in drops minus outside drops; "
        f"Spearman with mean clear-sky irradiance over 24 hours: {rho_p:.2f} (p = {p_p:.3f})"
    )
    say(
        "  hour: "
        + " ".join(f"{h:02d}:{v:+.1f}" for h, v in zip(prof["hour"], prof["excess"], strict=True))
    )
    diff = (tab["d8_midday_depth_mw"] - tab["d8_night_depth_mw"]).drop_nulls().to_numpy()
    boot = [np.median(rng.choice(diff, diff.size)) for _ in range(5000)]
    say(
        f"  per drop, midday (10-15) minus night (22-04) depth: median {np.median(diff):+.2f} MW "
        f"(bootstrap 95% {np.percentile(boot, 2.5):+.2f} to {np.percentile(boot, 97.5):+.2f}); "
        f"midday deeper in {int((diff > 0).sum())} of {diff.size} drops"
    )
    rhos = []
    for a, b in zip(drops["first_stamp"], drops["last_stamp"], strict=True):
        h = np.array([np.nanmedian(depth[a:b][local_hour[a:b] == hh]) for hh in hours])
        rhos.append(stats.spearmanr(h, prof["clear_sky_ghi"])[0])
    say(
        f"  per-drop Spearman of hourly depth with clear-sky irradiance: median {np.nanmedian(rhos):+.2f}, "
        f"positive in {int((np.array(rhos) > 0).sum())} of {len(rhos)}"
    )
    out_rho = stats.spearmanr(prof["depth_outside_mw"], prof["clear_sky_ghi"])[0]
    say(f"  same correlation outside drops (control): {out_rho:+.2f}")
    (WORK_DIR / "printed_report.txt").write_text("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
