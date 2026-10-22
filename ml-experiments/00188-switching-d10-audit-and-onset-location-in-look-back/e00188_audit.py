"""Part 1: confirm the gated pair alarms that no logged transfer explains, then re-score the mask.

An unlogged gated alarm is a confirmed unlogged event when a measure independent of the alarm's own
statistic finds the same transfer: a persistent step (the median residual over the 12 hours after
the alarm's step stamp minus the median over the 12 hours before) at the alarm's series, and an
opposite step of similar size summed over the partner series the alarm names. The thresholds are
fixed in the plan (`result.md`). The rows of confirmed events (the step to 72 hours after the end)
are removed from experiment 171's no-harm rows, and the pooled contrast is scored again.

    uv run --no-sync python e00188_audit.py | tee printed_audit.txt
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Final

import e00188_events as ev
import e00188_fed as fed
import numpy as np
import polars as pl
from e00188_common import E156_DIR, E171_DIR, GRID_START, WORK_DIR, load_panel

SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
HALF_DAY: Final[int] = 24
"""Half-hours in 12 hours."""
GAP: Final[int] = 2
"""Half-hours skipped on each side of the step stamp."""
MIN_FINITE: Final[int] = 8
MIN_OWN_MW: Final[float] = 0.3
RATIO_RANGE: Final[tuple[float, float]] = (0.5, 2.0)
PLACEBO_SHIFTS: Final[tuple[int, ...]] = tuple(
    sign * 336 * weeks for weeks in (1, 2, 3) for sign in (-1, 1)
)
"""Whole-week shifts of +/-7, 14, and 21 days, in half-hours."""
FIRST_RUN: Final = GRID_START.replace(year=2024, month=4, day=10)
N_DRAWS: Final[int] = 20_000
LIMIT_ALL: Final[float] = 0.03
LIMIT_WINTER: Final[float] = 0.05
REPORT: dict[str, object] = {}


def step_of(*, residual: np.ndarray, stamp: int) -> float:
    """Median residual after the stamp minus median before it; NaN when readings are too few."""
    after = residual[stamp + GAP : stamp + GAP + HALF_DAY]
    before = residual[max(stamp - GAP - HALF_DAY, 0) : max(stamp - GAP, 0)]
    if np.isfinite(after).sum() < MIN_FINITE or np.isfinite(before).sum() < MIN_FINITE:
        return float("nan")
    return float(np.nanmedian(after) - np.nanmedian(before))


def confirmed(
    *, panel: dict[int, np.ndarray], series: int, partners: list[int], stamp: int
) -> tuple[bool, float, float]:
    """Apply the plan's rule at `stamp`: returns (confirmed, own step, partner sum)."""
    own = step_of(residual=panel[series], stamp=stamp)
    others = [step_of(residual=panel[p], stamp=stamp) for p in partners]
    total = float(np.sum(others)) if others and np.all(np.isfinite(others)) else float("nan")
    if not (np.isfinite(own) and np.isfinite(total)) or abs(own) < MIN_OWN_MW or own * total >= 0:
        return False, own, total
    ratio = abs(total) / abs(own)
    return RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1], own, total


def candidates() -> pl.DataFrame:
    """Every gated pair alarm from 2024-04-10, with the series it is on and its partners."""
    window = int(json.loads((E156_DIR / "detect" / "headline.json").read_text())["window"])
    alarms = pl.read_parquet(E156_DIR / "detect" / f"real_W{window}_alarms.parquet").select(
        alarm_series="series", onset="stamp", partners="partners"
    )
    gated = ev.alarm_tables()["GATED"].filter(pl.col("role") == "alarm")
    out = gated.join(alarms, on=["alarm_series", "onset"], how="left")
    first = int((FIRST_RUN - GRID_START).total_seconds() // 1800)
    return out.filter(pl.col("onset") >= first)


def partner_list(text: str | None) -> list[int]:
    """Partner series from the `n:share;n:share` text of the alarm table."""
    return [] if not text else [int(p.split(":")[0]) for p in text.split(";")]


def audit_table() -> pl.DataFrame:
    """One row per gated alarm: logged or not, own and partner steps, confirmed, and placebo."""
    panel = load_panel()["residual"]
    logged = ev.logged_transfers()
    rows = []
    for r in candidates().iter_rows(named=True):
        series = int(r["alarm_series"])
        partners = partner_list(r["partners"])
        on_logged = (
            logged.filter(
                pl.col("series").list.set_intersection([series, *partners]).list.len() > 0
            )
            .filter(
                (pl.col("onset") - ev.MATCH_BEFORE <= r["onset"]) & (r["onset"] < pl.col("end"))
            )
            .height
            > 0
        )
        ok, own, total = confirmed(panel=panel, series=series, partners=partners, stamp=r["onset"])
        placebo = [
            confirmed(panel=panel, series=series, partners=partners, stamp=r["onset"] + d)[0]
            for d in PLACEBO_SHIFTS
        ]
        rows.append(
            {
                "alarm_series": series,
                "onset": r["onset"],
                "end": r["end"],
                "balance_score": r["balance_score"],
                "logged": on_logged,
                "own_mw": own,
                "partner_sum_mw": total,
                "confirmed": ok,
                "placebo_confirmed": sum(placebo),
                "placebo_n": len(placebo),
            }
        )
    return pl.DataFrame(rows)


def when(stamp: int):  # noqa: ANN201
    """The datetime of the end of the half-hour with this stamp."""
    return GRID_START.replace(tzinfo=None) + timedelta(minutes=30 * (stamp + 1))


def site_flags(*, table: pl.DataFrame) -> pl.DataFrame:
    """Per (site, row): inside a confirmed unlogged event, and inside any unconfirmed gated alarm.

    Each gated alarm covers the intervals on its alarm series and on its partners (the interval
    table has a copy of the alarm on each), so a site is hit when its series is in either role.
    """
    gated = ev.alarm_tables()["GATED"].filter(
        pl.col("onset") >= int((FIRST_RUN - GRID_START).total_seconds() // 1800)
    )
    keyed = gated.join(
        table.select("alarm_series", "onset", "logged", "confirmed"),
        on=["alarm_series", "onset"],
        how="left",
    )
    out = []
    for site in SITES:
        series = fed.SERIES_OF_SITE[site]
        rows = pl.read_parquet(E171_DIR / f"rows_{site}.parquet").filter(pl.col("arm") == "K0")
        mine = keyed.filter(pl.col("series") == series, ~pl.col("logged"))
        times = rows["valid_time"].to_numpy()
        in_confirmed = np.zeros(rows.height, dtype=bool)
        in_unconfirmed = np.zeros(rows.height, dtype=bool)
        for r in mine.iter_rows(named=True):
            lo = np.datetime64(when(r["onset"]), "us")
            hi = np.datetime64(when(r["end"] + ev.AFTER_STAMPS), "us")
            inside = (times >= lo) & (times <= hi)
            if r["confirmed"]:
                in_confirmed |= inside
            else:
                in_unconfirmed |= inside
        out.append(
            rows.select("site", "row").with_columns(
                in_confirmed=pl.Series(in_confirmed), in_unconfirmed=pl.Series(in_unconfirmed)
            )
        )
    return pl.concat(out)


def boot(*, values: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    """Mean with a 95% interval from resampling whole calendar months, as in experiment 171."""
    labels, inverse = np.unique(groups, return_inverse=True)
    total = np.bincount(inverse, weights=values, minlength=len(labels))
    count = np.bincount(inverse, minlength=len(labels)).astype(float)
    picks = np.random.default_rng(188).integers(0, len(labels), size=(N_DRAWS, len(labels)))
    draws = total[picks].sum(axis=1) / count[picks].sum(axis=1)
    return (
        float(total.sum() / count.sum()),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def fmt(t: tuple[float, float, float]) -> str:
    """Format a point estimate with its interval."""
    return f"{t[0]:+.3f} ({t[1]:+.3f} to {t[2]:+.3f})"


def show(*, name: str, frame: pl.DataFrame) -> tuple[float, float, float]:
    """Print and store MASK_GATED minus K0 on `frame`."""
    if frame.is_empty():
        print(f"{name}: no rows")
        return (float("nan"), float("nan"), float("nan"))
    t = boot(
        values=(frame["MASK_GATED"] - frame["K0"]).to_numpy(), groups=frame["month"].to_numpy()
    )
    print(f"{name}: MASK_GATED minus K0 {fmt(t)}  rows {frame.height}")
    REPORT[name] = list(t)
    return t


def main() -> None:
    """Print every number of part 1."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    table = audit_table()
    table.write_parquet(WORK_DIR / "audit_table.parquet")
    print("gated alarms from 2024-04-10:", table.height)
    unlogged = table.filter(~pl.col("logged"))
    print("not matched to a logged transfer:", unlogged.height)
    print("of those, confirmed:", int(unlogged["confirmed"].sum()))
    placebo_share = int(unlogged["placebo_confirmed"].sum()) / int(unlogged["placebo_n"].sum())
    print(f"placebo share confirmed (shifted by whole weeks): {placebo_share:.3f}")
    REPORT["counts"] = {
        "gated": table.height,
        "unlogged": unlogged.height,
        "confirmed": int(unlogged["confirmed"].sum()),
        "placebo_share": float(placebo_share),
    }
    flags = site_flags(table=table)
    print("\n== alarms per site, by interval series (a transfer appears at both ends) ==")
    wide_parts = []
    for site in SITES:
        rows = pl.read_parquet(E171_DIR / f"rows_{site}.parquet")
        wide = rows.pivot(on="arm", index=["site", "row", "valid_time"], values="qs").select(
            "site", "row", "valid_time", "K0", "MASK_GATED"
        )
        wide_parts.append(wide)
    wide = pl.concat(wide_parts).join(flags, on=["site", "row"])
    in_logged = pl.concat(
        [
            pl.read_parquet(E171_DIR / f"flags_{s}.parquet").select("site", "row", "in_logged")
            for s in SITES
        ]
    )
    active = pl.concat(
        [
            pl.read_parquet(E171_DIR / f"flags_{s}.parquet").select(
                "site", "row", "active_MASK_GATED"
            )
            for s in SITES
        ]
    )
    wide = (
        wide.join(in_logged, on=["site", "row"])
        .join(active, on=["site", "row"])
        .filter(~pl.col("in_logged"))
        .with_columns(
            month=pl.col("valid_time").dt.strftime("%Y-%m"), mo=pl.col("valid_time").dt.month()
        )
        .with_columns(winter=pl.col("mo").is_in([11, 12, 1, 2, 3]))
    )
    print("\n== experiment 171's rows (reproduced) ==")
    show(name="171 pooled", frame=wide)
    after = wide.filter(~pl.col("in_confirmed"))
    print("\n== G1' and G2' ==")
    pooled = show(name="pooled, confirmed events removed", frame=after)
    winter = show(name="winter, confirmed events removed", frame=after.filter(pl.col("winter")))
    g1 = pooled[1] >= -LIMIT_ALL and pooled[2] <= LIMIT_ALL
    g2 = winter[1] >= -LIMIT_WINTER and winter[2] <= LIMIT_WINTER
    verdict = (
        "refuted"
        if pooled[0] > LIMIT_ALL
        else "supported"
        if g1 and g2
        else "partly supported"
        if g1
        else "inconclusive"
    )
    print(f"G1' {g1} (half-width {(pooled[2] - pooled[1]) / 2:.3f}); G2' {g2}; verdict {verdict}")
    REPORT["verdict"] = verdict
    print("\n== beside the contrast ==")
    show(name="rows of confirmed events alone", frame=wide.filter(pl.col("in_confirmed")))
    show(name="rows of unconfirmed gated alarms alone", frame=wide.filter(pl.col("in_unconfirmed")))
    for site in SITES:
        sub = wide.filter(pl.col("site") == site)
        show(name=f"only {site}, all rows", frame=sub)
        show(
            name=f"only {site}, confirmed events removed", frame=sub.filter(~pl.col("in_confirmed"))
        )
    show(
        name="gated alarm active, confirmed removed",
        frame=after.filter(pl.col("active_MASK_GATED")),
    )
    print("\n== per site: gated alarm intervals (unlogged) confirmed / total ==")
    # Alarm intervals per site count both ends of a transfer (alarm and partner copies).
    gated = ev.alarm_tables()["GATED"].join(
        table.select("alarm_series", "onset", "logged", "confirmed"),
        on=["alarm_series", "onset"],
        how="inner",
    )
    for site in SITES:
        mine = gated.filter(pl.col("series") == fed.SERIES_OF_SITE[site])
        un = mine.filter(~pl.col("logged"))
        done = int(un["confirmed"].sum())
        line = f"{site}: gated {mine.height}, unlogged {un.height}, confirmed {done}"
        print(line)
        REPORT[f"alarms|{site}"] = [mine.height, un.height, int(un["confirmed"].sum())]
    (WORK_DIR / "report_audit.json").write_text(json.dumps(REPORT, indent=1, default=str))


if __name__ == "__main__":
    main()
