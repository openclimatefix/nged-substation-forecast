# ruff: noqa: E501, PLR0915
"""Pool the three folds, compute the headline tables and intervals, and print every number the page quotes.

    uv run --no-sync python ml-experiments/00249-*/e00249_report.py > ml-experiments/00249-*/printed_report.txt

Intervals come from resampling whole Primary pairs with replacement, 10,000 draws.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from e00249_common import WORK_DIR, real_transfers

N_RESAMPLES: Final[int] = 10_000
ARMS: Final[tuple[str, ...]] = (
    "RO-NC",
    "Pair alarm",
    "F1",
    "R1",
    "F2 w0.25",
    "R2 w0.25",
    "F2 w0.5",
    "R2 w0.5",
    "F2 w0.25 even months",
    "R2 w0.25 even months",
    "F2 w0.25 series hold-out",
    "R2 w0.25 series hold-out",
    "F2 w0.25 strict tier",
    "R2 w0.25 strict tier",
    "F2 heavy",
    "R2 heavy",
    "F3 w0.25",
    "R3 w0.25",
)
PAIRS_OF_ROUND: Final[dict[str, str]] = {"R2 w0.25": "R1", "F2 w0.25": "F1"}


def load() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Pooled per-transfer hits, per-injected-event hits, alarms, and the pseudo-positives used."""
    folds = [0, 1, 2]
    real = pl.concat([pl.read_parquet(WORK_DIR / f"real_f{f}.parquet") for f in folds])
    injected = pl.concat([pl.read_parquet(WORK_DIR / f"injected_f{f}.parquet") for f in folds])
    alarms = pl.concat([pl.read_parquet(WORK_DIR / f"alarms_f{f}.parquet") for f in folds])
    used = pl.concat([pl.read_parquet(WORK_DIR / f"used_f{f}.parquet") for f in folds])
    transfers = real_transfers().select(
        "card_id", "hours", "moved_mw_p50", "pair", "onset", "end", "series"
    )
    return real.join(transfers, on="card_id"), injected, alarms, used


def pair_bootstrap(
    *, table: pl.DataFrame, arm: str, base: str | None, weighted: bool, seed: int = 249
) -> tuple[float, float, float]:
    """Point estimate and 95% interval of recall (base None) or of the recall difference arm - base."""
    ta = table.filter(pl.col("arm") == arm).sort("card_id")
    w = ta["moved_mw_p50"].fill_null(0.0).to_numpy() if weighted else np.ones(ta.height)
    pr = ta["pair"].to_numpy()
    pairs = sorted(set(pr.tolist()))
    fa = ta["found"].to_numpy().astype(float)
    fb = (
        table.filter(pl.col("arm") == base).sort("card_id")["found"].to_numpy().astype(float)
        if base
        else np.zeros(ta.height)
    )
    num = np.array([(w * (fa - fb))[pr == p].sum() for p in pairs])
    den = np.array([w[pr == p].sum() for p in pairs])
    picks = np.random.default_rng(seed).integers(0, len(pairs), size=(N_RESAMPLES, len(pairs)))
    draws = num[picks].sum(axis=1) / np.maximum(den[picks].sum(axis=1), 1e-9)
    return (
        float(num.sum() / den.sum()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def main() -> None:
    """Print the tables."""
    real, injected, alarms, used = load()
    long = real.filter(pl.col("hours") > 24)
    n = long.filter(pl.col("arm") == "R1").height
    print(f"== {n} labelled transfers over 1 day; arms at 80 unexplained odd-month alarms ==")
    print(
        "arm | found/35 | MW-weighted recall [95%] | count minus R1 (or F1 for F arms) [95%] | gained | lost | found/66"
    )
    for arm in ARMS:
        sub = long.filter(pl.col("arm") == arm).sort("card_id")
        found = sub["found"].to_numpy()
        base = "F1" if arm.startswith("F") else "R1"
        r1 = long.filter(pl.col("arm") == base).sort("card_id")["found"].to_numpy()
        point, lo, hi = pair_bootstrap(table=long, arm=arm, base=None, weighted=True)
        d = pair_bootstrap(table=long, arm=arm, base=base, weighted=False)
        print(
            f"{arm} | {int(found.sum())} | {point:.3f} [{lo:.3f}, {hi:.3f}] | {d[0] * n:+.1f} [{d[1] * n:+.1f}, {d[2] * n:+.1f}] | "
            f"{int((found & ~r1).sum())} | {int((~found & r1).sum())} | {int(real.filter((pl.col('arm') == arm) & pl.col('found')).height)}"
        )
    print("\n== MW-weighted recall difference minus round 1 [95%] ==")
    for arm in ARMS[4:]:
        base = "F1" if arm.startswith("F") else "R1"
        d = pair_bootstrap(table=long, arm=arm, base=base, weighted=True)
        print(f"{arm}: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
    pair_found = set(
        long.filter((pl.col("arm") == "Pair alarm") & pl.col("found"))["card_id"].to_list()
    )
    ronc_found = set(long.filter((pl.col("arm") == "RO-NC") & pl.col("found"))["card_id"].to_list())
    print("\n== Events found by a round 2 or 3 arm and missed by round 1 ==")
    for arm in ARMS[4:]:
        base = "F1" if arm.startswith("F") else "R1"
        base_found = set(
            long.filter((pl.col("arm") == base) & pl.col("found"))["card_id"].to_list()
        )
        got = long.filter(
            (pl.col("arm") == arm) & pl.col("found") & ~pl.col("card_id").is_in(list(base_found))
        )
        lost = base_found - set(
            long.filter((pl.col("arm") == arm) & pl.col("found"))["card_id"].to_list()
        )
        text = "; ".join(
            f"{r['hours']:.0f} h {r['moved_mw_p50']:.1f} MW pair-alarm={'yes' if r['card_id'] in pair_found else 'no'} RO-NC={'yes' if r['card_id'] in ronc_found else 'no'}"
            for r in got.iter_rows(named=True)
        )
        print(f"{arm}: gains {got.height} [{text}]; loses {len(lost)}")
    print("\n== Found per pair (35 transfers) ==")
    pairs = sorted(long["pair"].unique().to_list())
    letter = {p: chr(ord("A") + i) for i, p in enumerate(pairs)}
    for arm in ARMS:
        per = {
            letter[p]: int(g["found"].sum())
            for (p,), g in long.filter(pl.col("arm") == arm).group_by("pair")
        }
        print(f"{arm}: " + " ".join(f"{letter[p]}={per.get(letter[p], 0)}" for p in pairs))
    print("\n== Injected transfers (odd months): recall of start and end, by arm ==")
    for kind in ("eval_held", "eval_fitted"):
        for arm in ARMS:
            sub = injected.filter((pl.col("arm") == arm) & (pl.col("kind") == kind))
            long_i = sub.filter(pl.col("hours") > 24)
            print(
                f"{kind} | {arm} | n={sub.height} | found {sub['found'].mean():.3f} | over 1 day {long_i['found'].mean():.3f}"
            )
    print("\n== Pseudo-positives used (headline variant '2 w0.25'), pooled over folds ==")
    head = used.filter(pl.col("variant") == "2 w0.25")
    print(
        head.group_by("fold")
        .agg(
            n=pl.len(),
            odd=pl.col("odd").sum(),
            source_x=(pl.col("source") == "x").sum(),
            median_mw=pl.col("step_mw24").median(),
        )
        .sort("fold")
    )
    print("by series:", dict(sorted(head.group_by("series").agg(n=pl.len()).iter_rows())))
    bands = head.with_columns(
        band=pl.when(pl.col("step_mw24") < 1)
        .then(pl.lit("<1 MW"))
        .when(pl.col("step_mw24") < 3)
        .then(pl.lit("1-3 MW"))
        .otherwise(pl.lit(">=3 MW"))
    )
    print("by size:", dict(bands.group_by("band").agg(n=pl.len()).iter_rows()))
    unique = head.unique(["series", "stamp"])
    transfers = real_transfers()
    match = 0
    for row in unique.iter_rows(named=True):
        hit = transfers.filter(
            pl.col("series").list.contains(row["series"])
            & ((pl.col("onset") - row["stamp"]).abs() <= 4)
        )
        match += hit.height > 0
    print(
        f"distinct pseudo-positives across folds: {unique.height}; within 2 hours of a labelled transfer's onset on its series (post hoc, after scoring): {match}"
    )
    print("\n== Alarm accounting (mean over folds) ==")
    for arm in ARMS:
        sub = alarms.filter(pl.col("arm") == arm)
        unx = sub.filter(~pl.col("explained") & pl.col("odd"))
        print(
            f"{arm} | alarms (all months) {sub.height / 3:.0f} | unexplained odd {unx.height / 3:.0f}"
        )


if __name__ == "__main__":
    main()
