# ruff: noqa: E501
"""Print every number the write-up quotes: matches, alarms, rates, injected recall, contamination, harm.

OMP_NUM_THREADS=1 uv run --no-sync python e00302_report.py | tee printed_report.txt
"""

from __future__ import annotations

import json
from typing import Final

import e00302_rule as rule
import numpy as np
import polars as pl
from e00302_common import E195_DIR, WORK_DIR, stamp_months
from e00302_data import block_day_mask
from e00302_score import shifts_table

ARMS: Final[tuple[str, ...]] = ("L0", "L1", "L2", "L3", "L3_5", "L3_2")
BOOT: Final[int] = 4000
BONFERRONI: Final[tuple[float, float]] = (0.85, 99.15)
"""Percentiles of a 98.3% interval (three planned contrasts)."""
SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
RNG = np.random.default_rng(302)


def pct(x: np.ndarray, *, wide: bool = False) -> tuple[float, float]:
    """95% (or Bonferroni 98.3%) percentile interval."""
    lo, hi = BONFERRONI if wide else (2.5, 97.5)
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def matches() -> None:
    """Matches, alarms per match, recall, onset error by arm, stratum, and variant."""
    hits = pl.read_parquet(WORK_DIR / "shift_hits.parquet")
    counts = pl.read_parquet(WORK_DIR / "alarm_counts.parquet")
    print("\n== Matches of the logged load shifts (found / total), alarms per match ==")
    for variant in ("all", "confirmed"):
        for stratum, expr in (
            ("all 35", pl.lit(True)),
            ("long and deep (>=2 d, >=10% of load)", pl.col("long_deep")),
            ("long (>=2 d), depth unknown included", pl.col("long")),
        ):
            print(f"-- variant {variant}, stratum {stratum}")
            for arm in ARMS:
                h = hits.filter((pl.col("arm") == arm) & (pl.col("variant") == variant) & expr)
                alarms = float(
                    counts.filter((pl.col("arm") == arm) & (pl.col("variant") == variant))
                    .group_by("fold")
                    .agg(a=pl.col("alarms").sum())["a"]
                    .to_numpy()
                    .mean()
                )
                found = int(h["found"].sum())
                w = h["moved_mw_p50"].to_numpy()
                recall = float((w * h["found"].to_numpy()).sum() / w.sum())
                onset = h.filter(pl.col("found"))["onset_error_h"].median()
                print(
                    f"{arm:5s} found {found:2d}/{h.height}  start-only {int(h['onset_found'].sum()):2d}  "
                    f"alarms (mean over folds) {alarms:6.1f}  alarms/match {alarms / max(found, 1):5.1f}  "
                    f"MW-weighted recall {recall:.3f}  median onset error {onset} h"
                )


def pair_bootstrap(
    *, hits: pl.DataFrame, a: str, b: str, stratum: pl.Expr, variant: str = "all"
) -> dict:
    """Difference in matches (arm a minus arm b), scaled to the number of shifts, resampling whole pairs."""
    sel = hits.filter((pl.col("variant") == variant) & stratum)
    wide = sel.pivot(on="arm", index=["card_id", "pair"], values="found").sort("card_id")
    pairs = wide["pair"].unique().to_list()
    by_pair = {p: wide.filter(pl.col("pair") == p) for p in pairs}
    da = {p: int(by_pair[p][a].sum()) for p in pairs}
    db = {p: int(by_pair[p][b].sum()) for p in pairs}
    n = {p: by_pair[p].height for p in pairs}
    point = sum(da.values()) - sum(db.values())
    draws = []
    for _ in range(BOOT):
        pick = RNG.choice(len(pairs), size=len(pairs))
        chosen = [pairs[i] for i in pick]
        total = sum(n[p] for p in chosen)
        diff = sum(da[p] - db[p] for p in chosen)
        draws.append(diff * wide.height / max(total, 1))
    gained = int(((wide[a] == True) & (wide[b] == False)).sum())  # noqa: E712
    lost = int(((wide[a] == False) & (wide[b] == True)).sum())  # noqa: E712
    return {
        "point": point,
        "ci95": pct(np.array(draws)),
        "ci98": pct(np.array(draws), wide=True),
        "gained": gained,
        "lost": lost,
        "n": wide.height,
    }


def contrasts() -> None:
    """Planned contrasts in matched count, paired by load shift, resampling whole Primary pairs."""
    hits = pl.read_parquet(WORK_DIR / "shift_hits.parquet")
    print(
        "\n== Contrasts in matches (pairs resampled; difference scaled to the stratum's shifts) =="
    )
    for name, expr in (("all 35", pl.lit(True)), ("long and deep", pl.col("long_deep"))):
        for a, b, label in (
            ("L3", "L1", "C1 L3 minus L1 (decisive)"),
            ("L2", "L1", "C2 L2 minus L1 (negatives)"),
            ("L3", "L2", "C3 L3 minus L2 (threshold)"),
            ("L1", "L0", "reference L1 minus L0"),
            ("L3_5", "L1", "L3 at 5 per Primary-year minus L1"),
        ):
            r = pair_bootstrap(hits=hits, a=a, b=b, stratum=expr)
            print(
                f"{name:14s} {label:40s} {r['point']:+d} [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}] "
                f"(98.3%: [{r['ci98'][0]:+.1f}, {r['ci98'][1]:+.1f}]) gained {r['gained']} lost {r['lost']} of {r['n']}"
            )


def rates() -> None:
    """Alarms per Primary-year by Primary and arm (mean over folds)."""
    counts = pl.read_parquet(WORK_DIR / "alarm_counts.parquet").filter(pl.col("variant") == "all")
    per = (
        counts.group_by("arm", "series", "site", "years")
        .agg(alarms=pl.col("alarms").mean())
        .with_columns(rate=pl.col("alarms") / pl.col("years"))
    )
    print("\n== Alarms per Primary-year, mean over folds ==")
    wide = per.pivot(on="arm", index=["series", "site"], values="rate").sort("series")
    print(
        wide.select("site", *[a for a in ARMS if a in wide.columns]).with_columns(
            pl.col(pl.Float64).round(2)
        )
    )
    for arm in ARMS:
        r = per.filter(pl.col("arm") == arm)
        total = float(r["alarms"].to_numpy().sum() / r["years"].to_numpy().sum())
        med = float(np.median(r["rate"].to_numpy()))
        within2 = int(((r["rate"] >= med / 2) & (r["rate"] <= med * 2)).sum())
        print(
            f"{arm:5s} overall {total:.2f}, median {med:.2f}, within a factor 2 of median {within2} of {r.height}"
        )
    primaries = rule.primary_table()
    pred = rule.predicted_rates(primaries=primaries)
    print(
        "Leave-one-Primary-out predicted RO-NC rate (alarms per Primary-year):",
        {r["site"]: round(pred[int(r["series"])], 2) for r in primaries.iter_rows(named=True)},
    )


def injected() -> None:
    """Recall on injected load shifts."""
    inj = pl.read_parquet(WORK_DIR / "injected_hits.parquet")
    print("\n== Injected load shifts: recall (start and end found) and start found ==")
    print(
        inj.group_by("kind", "arm")
        .agg(recall=pl.col("found").mean(), onset=pl.col("onset_found").mean(), n=pl.len())
        .sort("kind", "arm")
        .with_columns(pl.col(pl.Float64).round(3))
    )
    print(
        "Reference: experiment 195's stored re-ranker (L0 model) at budget 80, odd-month injections:"
    )
    stored = pl.concat([pl.read_parquet(E195_DIR / f"injected_f{f}.parquet") for f in range(3)])
    print(
        stored.filter(pl.col("arm") == "R re-ranker")
        .group_by("kind")
        .agg(recall=pl.col("found").mean(), n=pl.len())
    )


def contamination() -> None:
    """How many real training negatives of L2 lie inside logged spans, D8 drops, and block days."""
    cands = pl.read_parquet(E195_DIR / "real_candidates.parquet").filter(
        (pl.col("source") == "x") & pl.col("odd")
    )
    blocks = block_day_mask()
    blocked = np.array(
        [
            bool(blocks.get(int(s), np.zeros(1, bool)).size > 1 and blocks[int(s)][t])
            for s, t in zip(cands["series"], cands["stamp"], strict=True)
        ]
    )
    c = cands.with_columns(blocked=pl.Series(blocked))
    any_label = c["explained"] | c["d8_drop"] | c["blocked"]
    print(
        "\n== Label contamination of L2's real training negatives (odd-month RO-NC candidates) =="
    )
    print(
        f"candidates {c.height}; inside logged or labelled spans {int(c['explained'].sum())}; in D8 drops {int(c['d8_drop'].sum())}; on block days {int(c['blocked'].sum())}; any of the three {int(any_label.sum())} ({any_label.mean():.1%})"
    )
    shifts = shifts_table()
    months = stamp_months()
    in_train = [
        (int(months[min(r["onset"], len(months) - 1)]) in rule_odd())
        for r in shifts.iter_rows(named=True)
    ]
    print(
        f"of the 35 logged load shifts, {sum(in_train)} start in an odd month, the months L1 to L3 train on"
    )
    near = 0
    for r, odd in zip(shifts.iter_rows(named=True), in_train, strict=True):
        if not odd:
            continue
        m = c.filter(
            pl.col("series").is_in(r["series"]) & ((pl.col("stamp") - r["onset"]).abs() <= 4)
        )
        near += int(m.height > 0)
    print(
        f"of those, {near} have a real candidate at the onset that L2 trains as a negative (L1 drops it)"
    )


def rule_odd() -> frozenset[int]:
    """Odd calendar months."""
    return frozenset({1, 3, 5, 7, 9, 11})


def harm() -> None:
    """Ordinary-day harm by arm on the months the arm did not train on, with whole-month intervals."""
    frames = []
    for site in SITES:
        new = pl.read_parquet(WORK_DIR / f"h_scores_{site}.parquet")
        meta = pl.read_parquet(WORK_DIR / f"h_meta_{site}.parquet")
        old = pl.read_parquet(
            WORK_DIR.parent
            / "00227-lsc-fewer-false-alarms-and-softer-use-part-2"
            / f"scores_{site}.parquet"
        ).filter(pl.col("arm") == "LR_b20")
        old = old.with_columns(arm=pl.lit("L0"))
        both = pl.concat([new.filter(pl.col("arm") != "L0"), old.select(new.columns)])
        k0 = new.filter(pl.col("arm") == "K0").select(
            "row", k0=pl.col("qs"), k0_mean=pl.col("mean_forecast")
        )
        per = (
            both.filter(pl.col("arm") != "K0")
            .group_by("arm", "row")
            .agg(qs=pl.col("qs").mean(), mean_forecast=pl.col("mean_forecast").mean())
            .join(k0, on="row")
            .join(meta.select("row", "valid_time"), on="row")
            .with_columns(site=pl.lit(site))
        )
        frames.append(per)
    d = pl.concat(frames).with_columns(
        harm=pl.col("qs") - pl.col("k0"),
        month=pl.col("valid_time").dt.year() * 100 + pl.col("valid_time").dt.month(),
        odd=(pl.col("valid_time").dt.month() % 2 == 1),
    )
    print(
        f"\n== Ordinary-day harm (arm minus K0 mean quantile score, points of effective capacity); {d['site'].n_unique()} sites =="
    )
    results: dict[str, dict] = {}
    for arm in ARMS:
        a = d.filter(pl.col("arm") == arm)
        if a.is_empty():
            continue
        for label, mask in (
            ("held-out months", a["odd"] if arm == "L0" else ~a["odd"]),
            ("in-sample months", ~a["odd"] if arm == "L0" else a["odd"]),
        ):
            sub = a.filter(mask)
            by_month = sub.group_by("month").agg(s=pl.col("harm").sum(), n=pl.len()).sort("month")
            s, n = by_month["s"].to_numpy(), by_month["n"].to_numpy()
            point = s.sum() / n.sum()
            draws = []
            for _ in range(BOOT):
                pick = RNG.choice(len(s), size=len(s))
                draws.append(s[pick].sum() / n[pick].sum())
            lo, hi = pct(np.array(draws))
            wlo, whi = pct(np.array(draws), wide=True)
            dm = float((sub["mean_forecast"] - 0).abs().mean()) if False else float("nan")
            print(
                f"{arm:5s} {label:16s} harm {point:+.3f} [{lo:+.3f}, {hi:+.3f}] (98.3%: [{wlo:+.3f}, {whi:+.3f}]) rows {sub.height}, months {len(s)}"
            )
            if label == "held-out months":
                results[arm] = {
                    "point": point,
                    "lo": lo,
                    "hi": hi,
                    "months": by_month["month"].to_list(),
                    "s": s.tolist(),
                    "n": n.tolist(),
                }
            del dm
    print(
        "\n-- paired differences on the same held-out months (even months: L1, L2, L3 variants) --"
    )
    for a, b in (("L3", "L1"), ("L2", "L1"), ("L3", "L2")):
        if a in results and b in results:
            ma = dict(
                zip(
                    results[a]["months"],
                    zip(results[a]["s"], results[a]["n"], strict=True),
                    strict=True,
                )
            )
            mb = dict(
                zip(
                    results[b]["months"],
                    zip(results[b]["s"], results[b]["n"], strict=True),
                    strict=True,
                )
            )
            months = sorted(set(ma) & set(mb))
            sa = np.array([ma[m][0] / ma[m][1] for m in months])
            sb = np.array([mb[m][0] / mb[m][1] for m in months])
            nn = np.array([ma[m][1] for m in months])
            point = float(((sa - sb) * nn).sum() / nn.sum())
            draws = []
            for _ in range(BOOT):
                pick = RNG.choice(len(months), size=len(months))
                draws.append(float(((sa - sb)[pick] * nn[pick]).sum() / nn[pick].sum()))
            lo, hi = pct(np.array(draws))
            wlo, whi = pct(np.array(draws), wide=True)
            print(
                f"{a} minus {b}: {point:+.3f} [{lo:+.3f}, {hi:+.3f}] (98.3%: [{wlo:+.3f}, {whi:+.3f}])"
            )
    json.dump(
        {
            k: {kk: vv for kk, vv in v.items() if kk in ("point", "lo", "hi")}
            for k, v in results.items()
        },
        (WORK_DIR / "harm_summary.json").open("w"),
        indent=1,
    )


def main() -> None:
    """Print every section."""
    contamination()
    matches()
    contrasts()
    rates()
    injected()
    harm()


if __name__ == "__main__":
    main()
