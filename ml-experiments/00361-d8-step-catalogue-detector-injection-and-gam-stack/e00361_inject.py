# ruff: noqa: E501
"""Step 2: inject known on/off components into Primaries with no engine and measure recovery.

Hosts are D2, D3, and D4 (experiment 354's placebo Primaries), over 2024-07-01 to 2025-06-30. Each
draw subtracts `k` independent components of 2, 3, or 4 MW. Each component alternates off-gaps and
on-spells resampled from D8's detected catalogue. A spell counts as recovered when detected
segments (steps or drops) cover at least half of its half-hours and the covering segment's level
is within 30% of the median injected total over the spell. The same windows on the uninjected host
give the chance-recovery rate. Prints every number the write-up quotes.

    PYTHONPATH=<the folders of experiments 252, 264, 354, and 361> \
        OMP_NUM_THREADS=2 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python -W ignore ml-experiments/00361-*/e00361_inject.py
"""

import json
from typing import Final

import e00361_catalogue as cat
import e00361_detect as detect
import numpy as np
import polars as pl

HOSTS: Final[tuple[str, ...]] = ("D2", "D3", "D4")
WINDOW: Final[tuple[str, str]] = ("2024-07-01", "2025-06-30")
SIZES_MW: Final[tuple[float, ...]] = (2.0, 3.0, 4.0)
OVERLAPS: Final[tuple[int, ...]] = (1, 2, 3)
N_DRAWS: Final[int] = 20
LEVEL_TOLERANCE: Final[float] = 0.3
MIN_COVER: Final[float] = 0.5
SEED: Final[int] = 0


def wilson(*, k: int, n: int) -> tuple[float, float]:
    """Return the 95% Wilson interval of a share."""
    if n == 0:
        return float("nan"), float("nan")
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def spells_for(*, rng: np.random.Generator, n: int, dwells: np.ndarray, gaps: np.ndarray) -> list:
    """Return one component's on-spells as (start, end) half-hour indices on a window of `n`."""
    out, t = [], int(rng.choice(gaps) * 48 * rng.random())
    while t < n:
        length = int(rng.choice(dwells) * 48)
        out.append((t, min(t + length, n)))
        t += length + max(int(rng.choice(gaps) * 48), detect.MERGE_GAP + 1)
    return [(s, e) for s, e in out if e - s >= detect.MIN_STEP]


def union_flag(*, n: int, power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a per-index detected flag and detected level (NaN where none) for steps and drops."""
    flag = np.zeros(n, dtype=bool)
    level = np.full(n, np.nan)
    for seg in (detect.detect(power_mw=power), detect.detect_drops(power_mw=power)):
        for s, e, lv in zip(seg.start, seg.end, seg.level_mw, strict=True):
            flag[s:e] = True
            level[s:e] = np.where(np.isfinite(level[s:e]), level[s:e], lv)
    return flag, level


def score_spells(*, spells: list, truth: np.ndarray, flag: np.ndarray, level: np.ndarray) -> list:
    """Return, per spell, whether it is recovered, and the start and level errors."""
    rows = []
    for s, e in spells:
        cover = float(flag[s:e].mean())
        found = np.flatnonzero(flag[max(s - 24, 0) : e + 24]) + max(s - 24, 0)
        detected_level = (
            float(np.nanmedian(level[s:e])) if np.isfinite(level[s:e]).any() else np.nan
        )
        injected = float(np.median(truth[s:e]))
        ok_level = np.isfinite(detected_level) and abs(detected_level + injected) <= (
            LEVEL_TOLERANCE * injected
        )
        rows.append(
            {
                "covered": cover,
                "recovered": cover >= MIN_COVER and bool(ok_level),
                "level_error": (-detected_level - injected) / injected,
                "start_error_h": (found[0] - s) / 2 if len(found) else np.nan,
            }
        )
    return rows


def main() -> None:
    """Run the injection test."""
    rng = np.random.default_rng(SEED)
    catalogue = (
        pl.concat(
            [
                cat.catalogue(site="D8", threshold_mw=detect.THRESHOLD_MW),
                cat.catalogue(site="D8", threshold_mw=detect.THRESHOLD_MW, drops=True),
            ]
        )
        .unique()
        .sort("start")
    )
    dwells = catalogue["dwell_days"].to_numpy()
    starts, ends = catalogue["start"].to_numpy(), catalogue["end"].to_numpy()
    gaps = (starts[1:] - ends[:-1]) / np.timedelta64(1, "D")
    gaps = gaps[gaps > 0.25]
    results, placebo, false_rate = [], [], {}
    for host in HOSTS:
        times, power = detect.load_power_mw(site=host)
        lo = int(np.searchsorted(times, np.datetime64(WINDOW[0])))
        hi = int(np.searchsorted(times, np.datetime64(WINDOW[1])))
        base = power[lo:hi]
        n = len(base)
        flag0, level0 = union_flag(n=n, power=base)
        false_rate[host] = float(detect.detect(power_mw=base).start.size)
        for k in OVERLAPS:
            for draw in range(N_DRAWS):
                injected = np.zeros(n)
                spells = []
                for _ in range(k):
                    size = float(rng.choice(SIZES_MW))
                    comp = spells_for(rng=rng, n=n, dwells=dwells, gaps=gaps)
                    for s, e in comp:
                        injected[s:e] += size
                        spells.append((s, e, size))
                flag, level = union_flag(n=n, power=base - injected)
                rows = score_spells(
                    spells=[(s, e) for s, e, _ in spells], truth=injected, flag=flag, level=level
                )
                pl_rows = score_spells(
                    spells=[(s, e) for s, e, _ in spells], truth=injected, flag=flag0, level=level0
                )
                for (s, e, size), r, p in zip(spells, rows, pl_rows, strict=True):
                    results.append(
                        {
                            "host": host,
                            "k": k,
                            "size": size,
                            "dwell_days": (e - s) / 48,
                            "draw": draw,
                            **r,
                        }
                    )
                    placebo.append(
                        {"host": host, "k": k, "size": size, "recovered": p["recovered"]}
                    )
    frame, plc = pl.DataFrame(results), pl.DataFrame(placebo)
    frame.write_parquet(cat.WORK / "injection_spells.parquet")
    out: dict = {"uninjected_steps_per_year": false_rate}
    for k in OVERLAPS:
        for size in (None, *SIZES_MW):
            sub = frame.filter(pl.col("k") == k)
            psub = plc.filter(pl.col("k") == k)
            if size is not None:
                sub, psub = sub.filter(pl.col("size") == size), psub.filter(pl.col("size") == size)
            kk, n = int(sub["recovered"].sum()), sub.height
            lo_, hi_ = wilson(k=kk, n=n)
            rec = sub.filter(pl.col("recovered"))
            out[f"k{k}_size{size}"] = {
                "spells": n,
                "recovered_share": kk / n,
                "wilson": [lo_, hi_],
                "chance_share": float(psub["recovered"].mean()),  # ty: ignore[invalid-argument-type]
                "median_level_error": float(rec["level_error"].median()) if rec.height else None,  # ty: ignore[invalid-argument-type]
                "median_abs_start_error_h": float(rec["start_error_h"].abs().median())  # ty: ignore[invalid-argument-type]
                if rec.height
                else None,
            }
    print(json.dumps(out, indent=1))
    (cat.WORK / "injection_report.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
