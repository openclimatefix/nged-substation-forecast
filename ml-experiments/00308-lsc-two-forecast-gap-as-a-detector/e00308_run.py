# ruff: noqa: E501
"""Calibrate the gap detectors on odd months, score every detector, and store the tables.

Run as `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue` around
`env OMP_NUM_THREADS=1 uv run --no-sync python e00308_run.py`.
"""

from __future__ import annotations

import json
from typing import Final

import e00308_data as dat
import e00308_detect as det
import e00308_harness as h
import numpy as np
import polars as pl

TARGETS: Final[tuple[str, ...]] = ("LR_b20", "RO_b80", "PG_g70")
SEED: Final[int] = 308


def calibrate(*, ctx: det.Context, score: str, variant: str) -> np.ndarray:
    """Odd-month false alarm rate per Primary-year at every threshold in the grid."""
    rates = []
    for k in det.K_GRID:
        n, years = det.fa_counts(
            ctx=ctx,
            opens=det.opens_all(ctx=ctx, score=score, variant=variant, k=float(k)),
            split="odd",
        )
        rates.append(n / years)
    return np.array(rates)


def pick_k(*, rates: np.ndarray, target: float) -> float:
    """Smallest threshold whose odd-month false alarm rate does not exceed the target."""
    ok = np.flatnonzero(rates <= target)
    return float(det.K_GRID[ok[0]]) if len(ok) else float(det.K_GRID[-1])


def classify_false_alarms(*, ctx: det.Context, opens: dict[int, np.ndarray]) -> dict[str, int]:
    """Where the false alarm episodes of all months fall."""
    cal = dat.calendar_flags()
    out = {"D8": 0, "D9": 0, "calendar day": 0, "November to March": 0, "other": 0}
    for s, v in ctx.real.items():
        o = opens.get(s, np.array([], dtype=np.int64))
        o = o[(o >= 0) & (o < h.N_GRID)]
        for stamp in o[v.ordinary_stamp[o]]:
            m = int(ctx.month[stamp])
            if s == det.D8:
                out["D8"] += 1
            elif s == det.D9:
                out["D9"] += 1
            elif cal[stamp]:
                out["calendar day"] += 1
            elif m in (11, 12, 1, 2, 3):
                out["November to March"] += 1
            else:
                out["other"] += 1
    return out


def fa_split(*, ctx: det.Context, opens: dict[int, np.ndarray]) -> dict[str, float]:
    """False alarms per Primary-year, all months: every series, without D8 and D9, and D8 and D9 alone."""
    out = {}
    for name, keep in (
        ("all", set(ctx.real)),
        ("without_d8_d9", set(ctx.real) - {det.D8, det.D9}),
        ("d8", {det.D8}),
        ("d9", {det.D9}),
    ):
        sub = det.Context(
            {s: v for s, v in ctx.real.items() if s in keep},
            ctx.month,
            ctx.year_month,
            ctx.odd_years,
            ctx.shifts,
            ctx.confirm,
        )
        n, years = det.fa_counts(
            ctx=sub, opens={s: o for s, o in opens.items() if s in keep}, split="all"
        )
        out[name] = n / years
    return out


def evaluate(
    *, ctx: det.Context, opens: dict[int, np.ndarray], weights: np.ndarray, rng: np.random.Generator
) -> dict:
    """The scores of one set of alarm episodes on the real record."""
    matched, lag = det.matches(ctx=ctx, opens=opens)
    late, _ = det.matches(ctx=ctx, opens=opens, late=det.LATE)
    res = det.bootstrap_matches(matched=matched, weights=weights, rng=rng)
    res["matched_late"] = int(late.sum())
    odd_n, odd_y = det.fa_counts(ctx=ctx, opens=opens, split="odd")
    all_n, _ = det.fa_counts(ctx=ctx, opens=opens, split="all")
    even = det.fa_by_month_interval(ctx=ctx, opens=opens, rng=rng, split="even")
    even_onset = np.isin(
        ctx.month[ctx.shifts["onset"].to_numpy()], list(h.TUNING_MONTHS), invert=True
    )
    res |= {
        "fa_odd": odd_n / odd_y,
        "fa_even": even[0],
        "fa_even_lo": even[1],
        "fa_even_hi": even[2],
        "alarms_per_match": (all_n + res["matched"]) / max(res["matched"], 1),
        "lag_median_hours": float(np.nanmedian(lag)) if np.isfinite(lag).any() else None,
        "matched_even_onset": int(matched[even_onset].sum()),
        "n_even_onset": int(even_onset.sum()),
        "fa_split": fa_split(ctx=ctx, opens=opens),
    }
    return res | {"_matched": matched.tolist(), "_late": late.tolist()}


def drops_and_blocks(*, opens: dict[int, np.ndarray]) -> dict:
    """Recall on D8's multi-day drops and the 17 block days."""
    drops = pl.read_parquet(dat.D8_DROPS)
    hit_d = [
        det.interval_hits(
            opens=opens, series=det.D8, lo=int(r["first_stamp"]), hi=int(r["last_stamp"])
        )
        for r in drops.iter_rows(named=True)
    ]
    blocks = [(s, lo, hi) for s, ranges in dat.block_day_ranges().items() for lo, hi in ranges]
    hit_b = [det.interval_hits(opens=opens, series=s, lo=lo, hi=hi) for s, lo, hi in blocks]
    return {
        "drops": int(sum(hit_d)),
        "n_drops": len(hit_d),
        "blocks": int(sum(hit_b)),
        "n_blocks": len(hit_b),
    }


def injected(*, ctx: det.Context, score: str, variant: str, k: float) -> dict:
    """Recall and false alarms on the 44 injected worlds at one threshold (variant b does not apply)."""
    if variant == "b":
        return {}
    sites = {"D5": 17, "D10": 10, "D11": 8, "D14": 16, "D16": 18, "D9": 7}
    rows = []
    fa_count, fa_rows = 0, 0.0
    for seed in range(9200, 9244):
        events = np.load(
            dat.DATA / "00141-switching-detector-fed-into-gam-k" / "worlds" / f"world_{seed}.npz"
        )
        for site, series in sites.items():
            path = dat.WORK_DIR / "gap" / f"world_{seed}_{site}.npz"
            if not path.exists():
                continue
            z = np.load(path)
            real = ctx.real[series]
            s = det.Series(
                series,
                real.cap,
                z["origin"],
                {c: z[c] for c in det.SCORES},
                real.sigma,
                real.ordinary,
                real.odd,
                real.ordinary_stamp,
            )
            opens = det.opens_of(s=s, score=score, variant=variant, k=k, confirm=None)
            delta = np.nan_to_num(events[f"delta_{series}"])
            blocked = np.zeros(h.N_GRID, dtype=bool)
            for e in events["events"]:
                lo, hi = int(e[2]), int(e[3])
                if not np.any(delta[lo:hi]):
                    continue
                hit = opens[(opens >= lo - det.MATCH_EARLY) & (opens <= hi)]
                rows.append(
                    {
                        "seed": seed,
                        "site": site,
                        "event": int(e[0]),
                        "fraction": float(e[4]),
                        "hours": (hi - lo) / 2.0,
                        "mw": float(np.abs(delta[lo:hi]).mean()),
                        "matched": bool(len(hit)),
                        "lag_hours": float(max(hit.min() - lo, 0) / 2.0) if len(hit) else None,
                        "month": int(ctx.month[lo]),
                    }
                )
                blocked[max(lo - 144, 0) : min(hi + 144, h.N_GRID)] = True
            clean = real.ordinary_stamp & ~blocked
            fa_count += int(clean[opens].sum())
            fa_rows += float((clean[z["origin"]] & z["has_data"]).sum())
    table = pl.DataFrame(rows)
    return {
        "events": table.height,
        "matched": int(table["matched"].sum()),
        "mw_share": float(
            (table["mw"].to_numpy() * table["matched"].to_numpy()).sum()
            / table["mw"].to_numpy().sum()
        ),
        "lag_median_hours": float(
            np.median(table.filter(pl.col("matched"))["lag_hours"].to_numpy())
        ),
        "fa_per_year": fa_count / (fa_rows * det.ORIGIN_YEARS),
        "table": table,
    }


def main() -> None:
    """Calibrate, score, and store."""
    det.RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ctx = det.build_context()
    weights = np.array([det.shift_weight(ctx=ctx, row=r) for r in ctx.shifts.iter_rows(named=True)])
    out: dict = {
        "n_shifts": ctx.shifts.height,
        "weights": weights.tolist(),
        "carried": {},
        "gap": {},
        "curves": {},
    }
    carried_matched: dict[str, np.ndarray] = {}
    carried_late: dict[str, np.ndarray] = {}
    carried_rate: dict[str, float] = {}
    for src in TARGETS:
        opens, steps = det.carried_opens(ctx=ctx, source=src)
        ev = evaluate(ctx=ctx, opens=opens, weights=weights, rng=rng)
        ev["step_rule_matched"] = int(det.step_matches(ctx=ctx, steps=steps).sum())
        ev["drops_blocks"] = drops_and_blocks(opens=opens)
        ev["fa_classes"] = classify_false_alarms(ctx=ctx, opens=opens)
        ev["n_alarms"] = int(sum(len(v) for v in opens.values()))
        out["carried"][src] = ev
        carried_matched[src] = np.array(ev["_matched"])
        carried_late[src] = np.array(ev["_late"])
        carried_rate[src] = ev["fa_odd"]
        print(src, {k: v for k, v in ev.items() if k not in ("_matched", "_late")}, flush=True)
    for score in det.SCORES:
        for variant in det.VARIANTS:
            rates = calibrate(ctx=ctx, score=score, variant=variant)
            out["curves"][f"{score}_{variant}"] = {
                "k": det.K_GRID.tolist(),
                "fa_odd": rates.tolist(),
            }
            for src in TARGETS:
                k = pick_k(rates=rates, target=carried_rate[src])
                opens = det.opens_all(ctx=ctx, score=score, variant=variant, k=k)
                ev = evaluate(ctx=ctx, opens=opens, weights=weights, rng=rng)
                ev["k"] = k
                ev["drops_blocks"] = drops_and_blocks(opens=opens)
                ev["vs_carried"] = det.bootstrap_difference(
                    a=np.array(ev["_matched"]), b=carried_matched[src], rng=rng
                )
                ev["vs_carried_late"] = det.bootstrap_difference(
                    a=np.array(ev["_late"]), b=carried_late[src], rng=rng
                )
                ev["injected"] = {
                    kk: vv
                    for kk, vv in injected(ctx=ctx, score=score, variant=variant, k=k).items()
                    if kk != "table"
                }
                if score == "d" and variant == "a":
                    ev["fa_classes"] = classify_false_alarms(ctx=ctx, opens=opens)
                    inj = injected(ctx=ctx, score=score, variant=variant, k=k)
                    if inj:
                        inj["table"].write_parquet(
                            dat.WORK_DIR / "results" / f"injected_{src}.parquet"
                        )
                out["gap"][f"{score}_{variant}_{src}"] = ev
                print(
                    f"{score}_{variant} @ {src}: k={k} matched {ev['matched']}/{ctx.shifts.height} fa_even {ev['fa_even']:.2f} apm {ev['alarms_per_match']:.1f} vs carried {ev['vs_carried']}",
                    flush=True,
                )
    (dat.WORK_DIR / "results" / "report.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
