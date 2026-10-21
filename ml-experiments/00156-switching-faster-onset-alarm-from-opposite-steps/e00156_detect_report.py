# ruff: noqa: E501, PLR0915
"""Stage 1 report: detection contrasts C1 to C3, coverage, false alarms, and the real transfers.

```bash
uv run --no-sync python e00156_detect_report.py | tee <private folder>/printed_detect.txt
```
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Final

import e00156_alarm as alarm
import e00156_real as real
import numpy as np
import polars as pl
from e00156_d_common import GRID_START, WORK_DIR
from e00156_d_inject import stamp_of_month

OUT = WORK_DIR / "detect"
DRAWS: Final[int] = 20_000
SERIES_LABEL: Final[dict[int, str]] = {
    7: "D9",
    8: "D11",
    26: "D8",
    17: "D5",
    10: "D10",
    16: "D14",
    18: "D16",
}
FALSE_ALARMS = WORK_DIR.parent / "briefs" / "sw_false_alarms" / "work" / "alarms_flagged.parquet"


def load_events() -> pl.DataFrame:
    """Every event's outcome under every alarm family, with the month of its onset."""
    frame = pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("world_*_events.parquet"))])
    months = np.array([stamp_of_month(int(s)) for s in frame["onset"]])
    return frame.with_columns(
        ym=pl.Series(months), month=pl.Series(months % 100), lag_hours=pl.col("lag_hours")
    )


def wide(*, frame: pl.DataFrame, family: str) -> pl.DataFrame:
    """Events with RONC and `family` found flags and lags side by side."""
    base = frame.filter(pl.col("family") == "RONC").select(
        "world",
        "event_id",
        "ym",
        "month",
        "moved_mw",
        "covered",
        found_r="found",
        lag_r="lag_hours",
    )
    other = frame.filter(pl.col("family") == family).select(
        "world", "event_id", found_n="found", lag_n="lag_hours"
    )
    return base.join(other, on=["world", "event_id"])


def boot(
    *, table: pl.DataFrame, stat: Callable[[pl.DataFrame], float], seed: int = 156
) -> tuple[float, float, float, float, float]:
    """Point value and 95% and 99% intervals from resampling whole calendar months of onset."""
    keys = table["ym"].unique().sort().to_list()
    groups = [table.filter(pl.col("ym") == k) for k in keys]
    point = stat(table)
    rng = np.random.default_rng(seed)
    draws = np.empty(DRAWS)
    for i in range(DRAWS):
        picks = rng.integers(0, len(groups), len(groups))
        draws[i] = stat(pl.concat([groups[j] for j in picks]))
    lo95, hi95 = np.nanpercentile(draws, [2.5, 97.5])
    lo99, hi99 = np.nanpercentile(draws, [0.5, 99.5])
    return point, float(lo95), float(hi95), float(lo99), float(hi99)


def recall_gap(t: pl.DataFrame) -> float:
    """MW-weighted share found, new minus RO-NC."""
    w = t["moved_mw"].to_numpy()
    return float(
        ((t["found_n"].cast(pl.Float64) - t["found_r"].cast(pl.Float64)).to_numpy() * w).sum()
        / w.sum()
    )


def lag_gap(t: pl.DataFrame) -> float:
    """Median over transfers both alarms find of new lag minus RO-NC lag, in hours."""
    both = t.filter(pl.col("found_n") & pl.col("found_r"))
    return float((both["lag_n"] - both["lag_r"]).median()) if both.height else float("nan")  # ty: ignore[invalid-argument-type]


def fmt(r: tuple) -> str:
    """Point with 95% and 99% intervals."""
    return f"{r[0]:+.3f} (95% {r[1]:+.3f} to {r[2]:+.3f}; 99% {r[3]:+.3f} to {r[4]:+.3f})"


def main() -> None:
    """Print the stage 1 numbers."""
    frame = load_events()
    chosen = json.loads((OUT / "chosen_k.json").read_text())
    print("matched k by window (half-hours):", chosen)
    events = frame.filter(pl.col("family") == "RONC")
    print(
        f"events {events.height}; covered (a metered receiver) {events['covered'].mean():.3f}; "
        f"MW-weighted coverage {(events['covered'].cast(pl.Float64) * events['moved_mw']).sum() / events['moved_mw'].sum():.3f}"  # ty: ignore[unsupported-operator]
    )
    odd = frame.filter(pl.col("month") % 2 == 1)
    even = frame.filter(pl.col("month") % 2 == 0)
    print(
        f"odd-month events {odd.filter(pl.col('family') == 'RONC').height}; even {even.filter(pl.col('family') == 'RONC').height}"
    )
    recall_odd = {}
    for w in alarm.WINDOWS:
        t = wide(frame=odd, family=f"NEW{w}")
        mw = t["moved_mw"].to_numpy()
        recall_odd[w] = float((t["found_n"].cast(pl.Float64).to_numpy() * mw).sum() / mw.sum())
        print(f"odd months W={w}: NEW MW-weighted recall {recall_odd[w]:.3f}")
    headline = max(recall_odd, key=recall_odd.get)  # ty: ignore[no-matching-overload]
    print(
        f"headline window: {headline} half-hours ({headline / 2:.0f} h), k {chosen[str(headline)]}"
    )
    (OUT / "headline.json").write_text(json.dumps({"window": headline}))
    for w in alarm.WINDOWS:
        tag = " (headline)" if w == headline else ""
        t = wide(frame=even, family=f"NEW{w}")
        u = wide(frame=even, family=f"UNION{w}")
        both = t.filter(pl.col("found_n") & pl.col("found_r"))
        mw = t["moved_mw"].to_numpy()
        rn = float((t["found_n"].cast(pl.Float64).to_numpy() * mw).sum() / mw.sum())
        rr = float((t["found_r"].cast(pl.Float64).to_numpy() * mw).sum() / mw.sum())
        ru = float((u["found_n"].cast(pl.Float64).to_numpy() * mw).sum() / mw.sum())
        cov = t.filter(pl.col("covered"))
        mwc = cov["moved_mw"].to_numpy()
        print(f"\n== even months, W={w}{tag} ==")
        print(
            f"  recall NEW {rn:.3f}, RO-NC {rr:.3f}, UNION {ru:.3f}; share found NEW {t['found_n'].mean():.3f}, RO-NC {t['found_r'].mean():.3f}"
        )
        print(
            f"  on covered transfers only: NEW {float((cov['found_n'].cast(pl.Float64).to_numpy() * mwc).sum() / mwc.sum()):.3f}, RO-NC {float((cov['found_r'].cast(pl.Float64).to_numpy() * mwc).sum() / mwc.sum()):.3f}"
        )
        print(
            f"  median lag (h): NEW {t.filter(pl.col('found_n'))['lag_n'].median():.2f}, RO-NC {t.filter(pl.col('found_r'))['lag_r'].median():.2f}; transfers both find {both.height}"
        )
        print("  C1 lag NEW minus RO-NC (h):", fmt(boot(table=t, stat=lag_gap)))
        print("  C2 recall NEW minus RO-NC:", fmt(boot(table=t, stat=recall_gap)))
        print("  C3 recall UNION minus RO-NC:", fmt(boot(table=u, stat=recall_gap)))
        overlap = t.filter(pl.col("found_n"))
        print(
            f"  of the {overlap.height} transfers NEW finds, RO-NC also finds "
            f"{int(overlap['found_r'].sum())}; RO-NC finds {int(t['found_r'].sum())} in all"
        )
        print(f"  UNION median lag (h): {u.filter(pl.col('found_n'))['lag_n'].median():.2f}")
    # false alarms by series at each window's matched k
    ro = pl.read_parquet(FALSE_ALARMS).filter(pl.col("unexplained"))
    ro_by = ro.group_by("series").len().sort("series")
    print(
        "\nRO-NC unexplained alarms:",
        ro.height,
        "by series:",
        dict(zip(ro_by["series"], ro_by["len"], strict=True)),
    )
    logged = real.__dict__.get("LOGGED", None)
    from e00156_d_common import TUNING_MONTHS, near_log_masks, stamp_months

    logged = near_log_masks(pad_stamps=alarm.EXPLAIN_PAD)
    for w in alarm.WINDOWS:
        t = pl.read_parquet(OUT / f"real_W{w}_alarms.parquet")
        stamps = t["stamp"].to_numpy()
        expl = np.array([bool(logged[s][x]) for s, x in zip(t["series"], stamps, strict=True)])
        odd_m = np.isin(stamp_months()[stamps], list(TUNING_MONTHS))
        unexpl = t.filter(pl.Series(~expl))
        by = unexpl.group_by("series").len().sort("series")
        print(
            f"W={w}: alarms {t.height}, unexplained {unexpl.height}, unexplained odd-month {int((~expl & odd_m).sum())}; by series {dict(zip(by['series'], by['len'], strict=True))}; D8 {int((unexpl['series'] == 26).sum())}, D9 {int((unexpl['series'] == 7).sum())}"
        )
    # real transfers over one day
    transfers = real.logged_transfers().filter((pl.col("end") - pl.col("onset")) > 48)
    ronc = pl.read_parquet(real.INTERVALS).filter(pl.col("setting") == "R0_M48")
    print(f"\nreal transfers over 1 day: {transfers.height}")
    for w in alarm.WINDOWS:
        pair = pl.read_parquet(OUT / f"real_W{w}_pair.parquet")
        found_p = found_r = 0
        lags = []
        for row in transfers.iter_rows(named=True):
            hit = pair.filter(
                pl.col("series").is_in(row["series"])
                & ((pl.col("onset") - row["onset"]).abs() <= 4)
            )
            hit_r = ronc.filter(
                pl.col("series").is_in(row["series"])
                & ((pl.col("onset") - row["onset"]).abs() <= 4)
            )
            found_p += hit.height > 0
            found_r += hit_r.height > 0
            if hit.height:
                lags.append((hit["onset_known"].min() - row["onset"]) / 2)
        print(
            f"  W={w}: NEW finds {found_p}, RO-NC finds {found_r}; NEW median lag {np.median(lags) if lags else float('nan'):.1f} h"
        )
    _ = GRID_START


if __name__ == "__main__":
    main()
