# ruff: noqa: E501, PLR0912, PLR0915
"""Part 2 report: detection, end rule, and mask forecast on the five injection arms.

```bash
uv run --no-sync python e00162_report2.py | tee <private folder>/printed_part2.txt
```
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Final

import numpy as np
import polars as pl
from e00162_h_shaped import terms
from e00162_s_d_common import WORK_DIR
from e00162_s_d_inject import stamp_of_month

ARMS: Final[tuple[str, ...]] = ("SHARE", "CONST", "DEMAND", "SOLAR", "MIX", "PVOUT")
DRAWS: Final[int] = 20_000
LEVEL: Final[float] = 0.05 / 9
"""Bonferroni level of the nine planned contrasts."""
MASK_ARMS: Final[tuple[str, ...]] = (
    "K0",
    "CLEAN",
    "GAM_ONLY",
    "MASK_RONC",
    "MASK_NEW",
    "MASK_UNION",
)
ONSET_TOLERANCE: Final[int] = 4
SEED: Final[int] = 162
SUMMARY: Final = WORK_DIR / "part2_summary.json"


def interval_of(draws: np.ndarray) -> tuple[float, float, float, float]:
    """95% and Bonferroni-level intervals."""
    lo95, hi95 = np.nanpercentile(draws, [2.5, 97.5])
    lo, hi = np.nanpercentile(draws, [100 * LEVEL / 2, 100 * (1 - LEVEL / 2)])
    return float(lo95), float(hi95), float(lo), float(hi)


def fmt(point: float, r: tuple[float, float, float, float]) -> str:
    """A point with its 95% and Bonferroni intervals."""
    return f"{point:+.3f} (95% {r[0]:+.3f} to {r[1]:+.3f}; {100 * (1 - LEVEL):.1f}% {r[2]:+.3f} to {r[3]:+.3f})"


def boot_groups(
    *, table: pl.DataFrame, key: str, stat: Callable[[pl.DataFrame], float]
) -> tuple[float, tuple[float, float, float, float]]:
    """Point and intervals from resampling whole calendar months (`key`)."""
    keys = table[key].unique().sort().to_list()
    groups = [table.filter(pl.col(key) == k) for k in keys]
    rng = np.random.default_rng(SEED)
    draws = np.empty(DRAWS)
    for i in range(DRAWS):
        picks = rng.integers(0, len(groups), len(groups))
        draws[i] = stat(pl.concat([groups[j] for j in picks]))
    return stat(table), interval_of(draws)


def boot_sums(
    *, months: np.ndarray, num: np.ndarray, den: np.ndarray | None = None
) -> tuple[float, tuple[float, float, float, float]]:
    """Point and intervals of sum(num)/sum(den) (or the mean) resampling months, by bincount."""
    labels, index = np.unique(months, return_inverse=True)
    sn = np.bincount(index, weights=num, minlength=len(labels))
    sd = (
        np.bincount(index, minlength=len(labels)).astype(float)
        if den is None
        else np.bincount(index, weights=den, minlength=len(labels))
    )
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(labels), size=(DRAWS, len(labels)))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = sn[picks].sum(axis=1) / sd[picks].sum(axis=1)
    return float(sn.sum() / sd.sum()), interval_of(draws)


def load_events(arm: str) -> pl.DataFrame:
    """Every event's outcome under every alarm family for one arm."""
    frame = pl.concat(
        [
            pl.read_parquet(p)
            for p in sorted((WORK_DIR / "detect" / arm).glob("world_*_events.parquet"))
        ]
    )
    ym = np.array([stamp_of_month(int(s)) for s in frame["onset"]])
    return frame.with_columns(ym=pl.Series(ym), month=pl.Series(ym % 100), arm=pl.lit(arm))


def end_coverage(arm: str) -> pl.DataFrame:
    """Per event: share of the true interval the causal live (R2_M48) interval covers, and its end error."""
    rows = []
    for path in sorted((WORK_DIR / "worlds" / arm).glob("world_*.parquet")):
        world = int(path.stem.split("_")[1])
        iv = pl.read_parquet(path).filter(
            (pl.col("setting") == "R2_M48") & (pl.col("role") == "alarm")
        )
        with np.load(path.with_suffix(".npz")) as stored:
            events = stored["events"]
        for event_id, donor, onset, end, _ in events:
            hit = iv.filter(
                (pl.col("series") == int(donor))
                & ((pl.col("onset") - int(onset)).abs() <= ONSET_TOLERANCE)
            )
            if not hit.height:
                rows.append(
                    {
                        "world": world,
                        "event_id": int(event_id),
                        "found_end": False,
                        "covered": None,
                        "end_error_h": None,
                        "early": None,
                    }
                )
                continue
            det_end = int(hit["end"].max())  # ty: ignore[invalid-argument-type]
            overlap = max(0, min(det_end, int(end)) - int(onset))
            true_len = int(end) - int(onset)
            rows.append(
                {
                    "world": world,
                    "event_id": int(event_id),
                    "found_end": True,
                    "covered": overlap / true_len,
                    "end_error_h": (det_end - int(end)) / 2,
                    "early": det_end < int(onset) + 0.8 * true_len,
                }
            )
    return pl.DataFrame(
        rows,
        schema_overrides={"covered": pl.Float64, "end_error_h": pl.Float64, "early": pl.Boolean},
    ).with_columns(arm=pl.lit(arm))


def daylight_flags() -> pl.DataFrame:
    """Per event: whether the donor has convex-model terms and the event holds daylight (solar mean of at least 0.05 MW)."""
    rows = []
    for path in sorted((WORK_DIR / "worlds" / "CONST").glob("world_*.npz")):
        world = int(path.stem.split("_")[1])
        with np.load(path) as stored:
            events = stored["events"]
        for event_id, donor, onset, end, _ in events:
            t = terms(series=int(donor))
            mean = (
                float(np.nanmean(t["solar"][int(onset) : int(end)]))
                if t is not None and np.isfinite(t["solar"][int(onset) : int(end)]).any()
                else float("nan")
            )
            rows.append(
                {
                    "world": world,
                    "event_id": int(event_id),
                    "daylight": bool(np.isfinite(mean) and mean >= 0.05),
                }
            )
    return pl.DataFrame(rows)


def donor_found(arm: str, family: str, *, anchored: bool) -> pl.DataFrame:
    """Per event: whether an interval on the donor starts near the onset (or near first light for solar-shaped arms)."""
    rows = []
    for path in sorted((WORK_DIR / "worlds" / arm).glob("world_*.npz")):
        world = int(path.stem.split("_")[1])
        with np.load(path) as stored:
            events = stored["events"]
        if family == "RONC":
            iv = pl.read_parquet(WORK_DIR / "worlds" / arm / f"world_{world}.parquet").filter(
                pl.col("setting") == "R0_M48"
            )
        else:
            iv = pl.read_parquet(WORK_DIR / "detect" / arm / f"world_{world}_W8_pair.parquet")
        onsets = {}
        for r in iv.iter_rows(named=True):
            onsets.setdefault(r["series"], []).append(r["onset"])
        for event_id, donor, onset, end, _ in events:
            anchor = int(onset)
            t = terms(series=int(donor))
            if anchored and arm in ("SOLAR", "PVOUT") and t is not None:
                sun = t["solar"][int(onset) : int(end)]
                if np.isfinite(sun).any() and np.nanmax(sun) > 0:
                    anchor = int(onset) + int(np.argmax(np.nan_to_num(sun) > 0.1 * np.nanmax(sun)))
            found = any(abs(o - anchor) <= ONSET_TOLERANCE for o in onsets.get(int(donor), []))
            rows.append({"world": world, "event_id": int(event_id), "found": found})
    return pl.DataFrame(rows)


def paired(*, frame: pl.DataFrame, family: str, arm: str, base: str = "CONST") -> pl.DataFrame:
    """Events of two arms side by side (arm 'a', base 'b')."""
    sel = frame.filter(pl.col("family") == family)
    a = sel.filter(pl.col("arm") == arm).select(
        "world", "event_id", "ym", "month", "moved_mw", found_a="found", lag_a="lag_hours"
    )
    b = sel.filter(pl.col("arm") == base).select(
        "world", "event_id", found_b="found", lag_b="lag_hours", mw_b="moved_mw"
    )
    return a.join(b, on=["world", "event_id"])


def recall(t: pl.DataFrame, which: str) -> float:
    """MW-weighted share found, weighted by the CONST arm's moved MW."""
    w = t["mw_b"].to_numpy()
    return float((t[which].cast(pl.Float64).to_numpy() * w).sum() / w.sum())


def main() -> None:
    """Print part 2."""
    events = pl.concat([load_events(a) for a in ARMS])
    even = events.filter(pl.col("month") % 2 == 0)
    print(
        f"events per arm {events.filter((pl.col('arm') == 'CONST') & (pl.col('family') == 'RONC')).height}; even months {even.filter((pl.col('arm') == 'CONST') & (pl.col('family') == 'RONC')).height}"
    )
    fb = {
        a: int(
            sum(int(np.load(p)["fallbacks"]) for p in (WORK_DIR / "worlds" / a).glob("world_*.npz"))
        )
        for a in ARMS
        if a != "SHARE"
    }
    print("events using the net-power fallback shape (donor without terms):", fb)
    mm = events.filter(pl.col("family") == "RONC").group_by("arm").agg(pl.col("moved_mw").mean())
    print(
        "mean moved MW per event by arm (positive deltas summed over series; not comparable for PVOUT, whose positive delta is at the donor; the donor mean is matched by construction):",
        dict(zip(mm["arm"], [round(x, 4) for x in mm["moved_mw"]], strict=True)),
    )
    print(
        "\n== detection, even-month events: MW-weighted recall (weights: CONST arm's moved MW) =="
    )
    for family, label in (("NEW8", "pair alarm 4 h"), ("RONC", "RO-NC"), ("UNION8", "union")):
        line = []
        for arm in ARMS:
            t = paired(frame=even, family=family, arm=arm)
            line.append(f"{arm} {recall(t, 'found_a'):.3f}")
        print(f"  {label}: " + "; ".join(line))
    print(
        "  median lag in hours (pair alarm 4 h, events found):",
        {
            a: round(
                float(
                    even.filter(  # ty: ignore[invalid-argument-type]
                        (pl.col("arm") == a) & (pl.col("family") == "NEW8") & pl.col("found")
                    )["lag_hours"].median()
                ),
                2,
            )
            for a in ARMS
        },
    )
    contrasts = {}
    for arm in ("SHARE", "DEMAND", "SOLAR", "MIX", "PVOUT"):
        print(f"\n-- {arm} minus CONST --")
        for family, name in (
            ("NEW8", "P4 pair-alarm recall"),
            ("RONC", "P5 RO-NC recall"),
            ("UNION8", "union recall"),
        ):
            t = paired(frame=even, family=family, arm=arm)
            r = boot_groups(
                table=t, key="ym", stat=lambda x: recall(x, "found_a") - recall(x, "found_b")
            )
            print(f"  {name}: {fmt(*r)}")
            contrasts[f"{arm}_{name}"] = r
        t = paired(frame=even, family="NEW8", arm=arm).filter(pl.col("found_a") & pl.col("found_b"))
        r = boot_groups(table=t, key="ym", stat=lambda x: float((x["lag_a"] - x["lag_b"]).median()))  # ty: ignore[invalid-argument-type]
        print(f"  P6 pair-alarm lag (h), {t.height} events both find: {fmt(*r)}")
        contrasts[f"{arm}_P6"] = r
    print(
        "\n== post hoc: recall by matching an interval on the donor only, a solar-shaped transfer scored from first light =="
    )
    key = even.filter((pl.col("arm") == "CONST") & (pl.col("family") == "RONC")).select(
        "world", "event_id", "ym", "moved_mw", "covered"
    )
    base_arm = {}
    for family, label in (("NEW8", "pair alarm"), ("RONC", "RO-NC")):
        base_arm[family] = donor_found("CONST", family, anchored=False).join(
            key, on=["world", "event_id"]
        )
        w = base_arm[family]["moved_mw"].to_numpy()
        print(
            f"  {label}: CONST (donor match, drawn onset) {float((base_arm[family]['found'].cast(pl.Float64).to_numpy() * w).sum() / w.sum()):.3f}; on events with a metered receiver {float((base_arm[family].filter(pl.col('covered'))['found'].cast(pl.Float64) * base_arm[family].filter(pl.col('covered'))['moved_mw']).sum() / base_arm[family].filter(pl.col('covered'))['moved_mw'].sum()):.3f}"  # ty: ignore[unsupported-operator]
        )
        for arm in ("SOLAR", "PVOUT", "MIX", "DEMAND", "SHARE"):
            for anchored in (False, True):
                if anchored and arm not in ("SOLAR", "PVOUT"):
                    continue
                t = (
                    donor_found(arm, family, anchored=anchored)
                    .join(key, on=["world", "event_id"])
                    .join(
                        base_arm[family].select("world", "event_id", found_b="found"),
                        on=["world", "event_id"],
                    )
                )
                r = boot_groups(
                    table=t,
                    key="ym",
                    stat=lambda x: float(
                        (  # ty: ignore[unsupported-operator]
                            (x["found"].cast(pl.Float64) - x["found_b"].cast(pl.Float64))
                            * x["moved_mw"]
                        ).sum()
                        / x["moved_mw"].sum()
                    ),
                )
                absolute = float(
                    (t["found"].cast(pl.Float64) * t["moved_mw"]).sum() / t["moved_mw"].sum()  # ty: ignore[unsupported-operator]
                )
                print(
                    f"    {arm} {'from first light' if anchored else 'drawn onset'}: recall {absolute:.3f}; minus CONST {fmt(*r)}"
                )
                contrasts[f"{arm}_{family}_donor_{'first_light' if anchored else 'onset'}"] = r
    print("\n== causal live end rule (R2_M48) on the donor, even-month events with an alarm ==")
    ends = pl.concat([end_coverage(a) for a in ARMS])
    key = events.filter(pl.col("family") == "RONC").select(
        "arm", "world", "event_id", "ym", "month"
    )
    ends = ends.join(key, on=["arm", "world", "event_id"]).filter(pl.col("month") % 2 == 0)
    for arm in ARMS:
        sub = ends.filter((pl.col("arm") == arm) & pl.col("found_end"))
        print(
            f"  {arm}: alarmed {sub.height}; mean share of true interval covered {sub['covered'].mean():.3f}; ends early (before 80% of true length) {sub['early'].mean():.3f}; median end error {sub['end_error_h'].median():.1f} h"
        )
    ends = ends.with_columns(cov0=pl.col("covered").fill_null(0.0))
    for arm in ARMS:
        sub = ends.filter(pl.col("arm") == arm)
        print(
            f"  {arm}: all {sub.height} even-month events, missed events count as 0: mean share of true interval covered {sub['cov0'].mean():.3f}"
        )
    flags = daylight_flags()
    ends = ends.join(flags, on=["world", "event_id"])
    for arm in ("SHARE", "DEMAND", "SOLAR", "MIX", "PVOUT"):
        a = ends.filter(pl.col("arm") == arm).select(
            "world", "event_id", "ym", "daylight", cov_a="cov0"
        )
        b = ends.filter(pl.col("arm") == "CONST").select("world", "event_id", cov_b="cov0")
        t = a.join(b, on=["world", "event_id"])
        r = boot_groups(table=t, key="ym", stat=lambda x: float((x["cov_a"] - x["cov_b"]).mean()))  # ty: ignore[invalid-argument-type]
        print(
            f"  P7 {arm} minus CONST, all {t.height} events (missed = 0): covered share {fmt(*r)}"
        )
        contrasts[f"{arm}_P7"] = r
        td = t.filter(pl.col("daylight"))
        rd = boot_groups(table=td, key="ym", stat=lambda x: float((x["cov_a"] - x["cov_b"]).mean()))  # ty: ignore[invalid-argument-type]
        print(f"      daylight events only ({td.height}): {fmt(*rd)}")
    print(
        f"  events with daylight and convex-model terms: {int(flags['daylight'].sum())} of {flags.height} (all worlds)"
    )
    print("\n== sensitivity: detection contrasts on daylight events only ==")
    flagged = even.join(flags, on=["world", "event_id"]).filter(pl.col("daylight"))
    for arm in ("DEMAND", "SOLAR", "MIX", "PVOUT"):
        for family, name in (
            ("NEW8", "pair-alarm recall"),
            ("RONC", "RO-NC recall"),
            ("UNION8", "union recall"),
        ):
            t = paired(frame=flagged, family=family, arm=arm)
            r = boot_groups(
                table=t, key="ym", stat=lambda x: recall(x, "found_a") - recall(x, "found_b")
            )
            print(f"  {arm} minus CONST, {name}, {t.height} events: {fmt(*r)}")
    print("\n== mask forecast, rows with a forecast origin after an onset ==")
    scores = {}
    for arm in ARMS:
        frame = pl.concat(
            [pl.read_parquet(p) for p in sorted((WORK_DIR / "scores" / arm).glob("*.parquet"))]
        )
        frame = frame.with_columns(
            month=(
                pl.lit(datetime(2024, 4, 1, 9))  # noqa: DTZ001
                + pl.duration(days=pl.col("run"))
                + pl.duration(minutes=(pl.col("lead_hours") * 60).cast(pl.Int64))
            ).dt.strftime("%Y-%m")
        )
        keys = ["site", "world", "period", "row"]
        wide = frame.pivot(on="arm", index=[*keys, "month"], values="qs")
        scores[arm] = wide
        print(
            f"  {arm}: rows {wide.height}; "
            + "; ".join(f"{a} {wide[a].mean():.3f}" for a in MASK_ARMS)
        )
        gap = (wide["K0"] - wide["CLEAN"]).to_numpy()
        closed = {
            a: float((wide["K0"] - wide[a]).sum() / gap.sum())
            for a in ("MASK_RONC", "MASK_NEW", "MASK_UNION", "GAM_ONLY")
        }
        print(
            "     share of the K0-to-CLEAN gap closed:", {a: round(v, 3) for a, v in closed.items()}
        )
    keys = ["site", "world", "period", "row"]
    base = scores["CONST"]
    for arm in ("SHARE", "DEMAND", "SOLAR", "MIX", "PVOUT"):
        t = scores[arm].join(base.select(*keys, **{f"{a}_c": a for a in MASK_ARMS}), on=keys)
        print(f"  {arm} vs CONST: rows {t.height} of {scores[arm].height} / {base.height}")
        months = t["month"].to_numpy()
        for a, name in (
            ("MASK_UNION", "P8/P9 MASK_UNION"),
            ("MASK_RONC", "MASK_RONC"),
            ("GAM_ONLY", "GAM_ONLY"),
            ("K0", "K0"),
        ):
            r = boot_sums(months=months, num=(t[a] - t[f"{a}_c"]).to_numpy())
            print(f"    {name}: {fmt(*r)}")
            contrasts[f"{arm}_{name}"] = r
        # share of gap closed by MASK_UNION, arm minus CONST
        gap_a = (t["K0"] - t["CLEAN"]).to_numpy()
        gap_c = (t["K0_c"] - t["CLEAN_c"]).to_numpy()
        print(
            f"    share of gap closed by MASK_UNION: {arm} {(t['K0'] - t['MASK_UNION']).sum() / gap_a.sum():.3f}; CONST {(t['K0_c'] - t['MASK_UNION_c']).sum() / gap_c.sum():.3f}"
        )
        print(
            f"    score of MASK_UNION minus GAM_ONLY: {arm} {(t['MASK_UNION'] - t['GAM_ONLY']).mean():+.3f}; CONST {(t['MASK_UNION_c'] - t['GAM_ONLY_c']).mean():+.3f}"
        )
    print(
        "\n== checks: error of the mean and median forecast, MASK_UNION, arm minus CONST (points of capacity) =="
    )
    raw = {}
    for arm in ("CONST", "SHARE", "DEMAND", "SOLAR", "MIX", "PVOUT"):
        f = pl.concat(
            [pl.read_parquet(p) for p in sorted((WORK_DIR / "scores" / arm).glob("*.parquet"))]
        ).filter(pl.col("arm") == "MASK_UNION")
        raw[arm] = f.with_columns(
            month=(
                pl.lit(datetime(2024, 4, 1, 9))  # noqa: DTZ001
                + pl.duration(days=pl.col("run"))
                + pl.duration(minutes=(pl.col("lead_hours") * 60).cast(pl.Int64))
            ).dt.strftime("%Y-%m")
        )
    for arm in ("SHARE", "DEMAND", "SOLAR", "MIX", "PVOUT"):
        t = raw[arm].join(
            raw["CONST"].select(*keys, mean_c="err_mean", med_c="err_median"), on=keys
        )
        for column, other, name in (
            ("err_mean", "mean_c", "mean forecast"),
            ("err_median", "med_c", "median forecast"),
        ):
            r = boot_sums(
                months=t["month"].to_numpy(), num=(t[column].abs() - t[other].abs()).to_numpy()
            )
            print(f"  {arm} minus CONST, absolute error of the {name}: {fmt(*r)}")
    k0 = pl.concat(
        [pl.read_parquet(p) for p in sorted((WORK_DIR / "scores" / "CONST").glob("*.parquet"))]
    ).filter(pl.col("arm") == "K0")
    by_site = (
        k0.group_by("site")
        .agg(diff=(pl.col("err_median") - pl.col("err_mean")).mean())
        .sort("site")
    )
    print(
        "  median minus mean forecast by site:",
        dict(zip(by_site["site"], [round(x, 3) for x in by_site["diff"]], strict=True)),
    )
    SUMMARY.write_text(json.dumps({k: [v[0], *v[1]] for k, v in contrasts.items()}))
    print("\nrow sets equal across arms:", all(scores[a].height == base.height for a in ARMS))


if __name__ == "__main__":
    main()
