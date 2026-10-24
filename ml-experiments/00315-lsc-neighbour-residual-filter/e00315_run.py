# ruff: noqa: E501, D107, PLR0912, PLR0915
"""Score the neighbour filter on the real record: curves, operating points, groups, intervals.

The filter rule and the arms are fixed in `result.md` before this script ran. It reads the stored
scores of experiments 308 and 309 through experiment 309's folder, the neighbour list written by
`e00315_ceiling.py`, and writes `results.json` and `curves.parquet` to the private folder.

    PYTHONPATH=ml-experiments/00309-lsc-two-forecast-gap-choosing-without-truth \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00315-lsc-neighbour-residual-filter/e00315_run.py
"""

from __future__ import annotations

import json
from typing import Final

import e00309_detect as det
import e00309_harness as h
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final = ML_EXPERIMENTS_DATA_DIR / "00315-lsc-neighbour-residual-filter"
THETA: Final[float] = 0.25
N_DRAWS: Final[int] = 20_000
SEED: Final[int] = 315
D8, D9 = 26, 7
PLAIN_K: Final[dict[str, dict[str, float]]] = {
    "r24": {"LR_b20": 3.5, "RO_b80": 2.6},
    "pit24": {"LR_b20": 2.8, "RO_b80": 2.4},
}
"""Experiments 308 and 309's thresholds, set on odd months to the false alarm rates of the learned score and RO-NC."""
RECALL_TARGETS: Final[dict[str, int]] = {"recall >= 0.5": 18, "recall >= 0.33": 12}


def keep_mask(
    *,
    ctx: det.Context,
    neighbours: dict[str, list[int]],
    series: int,
    score: str,
    theta: float,
    max_contradict: int,
) -> np.ndarray:
    """True at origins where the neighbour filter keeps a flagged origin of one series."""
    me = ctx.real[series]
    scale = (lambda s: s.cap / 100.0) if score == "r24" else (lambda s: 1.0)
    a = me.score[score] * scale(me)
    confirm = np.zeros(len(a), dtype=bool)
    contradict = np.zeros(len(a), dtype=int)
    finite = np.zeros(len(a), dtype=int)
    for n in neighbours[str(series)]:
        other = ctx.real[n]
        b = other.score[score] * scale(other)
        ok = np.isfinite(a) & np.isfinite(b)
        big = ok & (np.abs(b) >= theta * np.abs(a))
        confirm |= big & (a * b < 0)
        contradict += (big & (a * b > 0)).astype(int)
        finite += ok.astype(int)
    return confirm | (finite == 0) | (contradict < max_contradict)


def opens_for(
    *, ctx: det.Context, score: str, k: float, keep: dict[int, np.ndarray] | None
) -> dict[int, np.ndarray]:
    """Alarm opening stamps per series; `keep` removes flagged origins before episodes form."""
    out = {}
    for s, v in ctx.real.items():
        z = np.abs(v.score[score]) / v.sigma[score]
        flag = np.isfinite(z) & (z > k)
        if keep is not None:
            flag &= keep[s]
        out[s] = v.origin[det.episode_starts(flag=flag)]
    return out


class Scorer:
    """Matched shifts and false alarm counts per calendar month for any set of alarm openings."""

    def __init__(self, ctx: det.Context, groups: pl.DataFrame) -> None:
        self.ctx = ctx
        self.months = np.unique(ctx.year_month)
        self.month_index = {int(m): i for i, m in enumerate(self.months)}
        self.rows = np.zeros(len(self.months))
        for v in ctx.real.values():
            ok = v.ordinary
            for m, c in zip(
                *np.unique(ctx.year_month[v.origin][ok], return_counts=True), strict=True
            ):
                self.rows[self.month_index[int(m)]] += c
        self.month_years = self.rows * det.ORIGIN_YEARS
        self.closure = {s: det.closure_blocks(series=s) for s in ctx.real}
        self.extra = {s: det.extra_blocks(series=s) for s in ctx.real}
        self.even_month = ~np.isin(
            np.array([int(str(m)[4:]) for m in self.months]), list(h.TUNING_MONTHS)
        )
        g = {r["card_id"]: r for r in groups.iter_rows(named=True)}
        rows = [g[r["card_id"]] for r in ctx.shifts.iter_rows(named=True)]
        self.stratum = np.array([r["stratum"] for r in rows])
        self.counterpart = np.array([r["counterpart"] for r in rows])
        self.weights = np.array(
            [det.shift_weight(ctx=ctx, row=r) for r in ctx.shifts.iter_rows(named=True)]
        )

    def score(self, opens: dict[int, np.ndarray]) -> dict:
        """Matched array, false alarm counts per month, and side counts."""
        matched, _ = det.matches(ctx=self.ctx, opens=opens)
        fa = np.zeros(len(self.months))
        d8 = d9 = unlogged = 0
        for s, v in self.ctx.real.items():
            o = opens.get(s, np.array([], dtype=np.int64))
            o = o[(o >= 0) & (o < h.N_GRID)]
            clean = o[v.ordinary_stamp[o]]
            for m, c in zip(
                *np.unique(self.ctx.year_month[clean], return_counts=True), strict=True
            ):
                fa[self.month_index[int(m)]] += c
            d8 += len(clean) if s == D8 else 0
            d9 += len(clean) if s == D9 else 0
            unlogged += int((self.extra[s][o] & ~self.closure[s][o]).sum())
        return {"matched": matched, "fa": fa, "d8": d8, "d9": d9, "unlogged": unlogged}

    def apm(self, r: dict, group: np.ndarray | None = None) -> float:
        """(false alarms + matched shifts) divided by matched shifts in the group."""
        total = r["fa"].sum() + r["matched"].sum()
        m = r["matched"] if group is None else r["matched"] & group
        return float(total / m.sum()) if m.sum() else float("inf")

    def interval(
        self, *, arms: list[dict], group: np.ndarray | None, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """Paired draws (whole shifts, whole months) of alarms per match for each arm."""
        n, nm = len(arms[0]["matched"]), len(self.months)
        si = rng.integers(0, n, size=(N_DRAWS, n))
        mi = rng.integers(0, nm, size=(N_DRAWS, nm))
        out = []
        for r in arms:
            total = r["fa"][mi].sum(axis=1) + r["matched"][si].sum(axis=1)
            m = r["matched"] if group is None else r["matched"] & group
            hits = m[si].sum(axis=1)
            out.append(np.where(hits > 0, total / np.maximum(hits, 1), np.nan))
        return {"draws": np.array(out)}


def ci(x: np.ndarray) -> list[float]:
    """Median-free 95% interval of finite draws."""
    x = x[np.isfinite(x)]
    return (
        [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [np.nan, np.nan]
    )


def main() -> None:
    """Run every arm and print the report."""
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ctx = det.build_context()
    neighbours = json.loads((OUT / "neighbours.json").read_text())
    groups = pl.read_parquet(OUT / "ceiling.parquet")
    sc = Scorer(ctx, groups)
    print(
        f"shifts {ctx.shifts.height}; stratum {int(sc.stratum.sum())}; counterpart {int(sc.counterpart.sum())}; "
        f"stratum with counterpart {int((sc.stratum & sc.counterpart).sum())}"
    )

    def keeps(score: str, theta: float, max_contradict: int) -> dict[int, np.ndarray]:
        return {
            s: keep_mask(
                ctx=ctx,
                neighbours=neighbours,
                series=s,
                score=score,
                theta=theta,
                max_contradict=max_contradict,
            )
            for s in ctx.real
        }

    variants = {"plain": None}
    for score in ("r24", "pit24"):
        variants[f"{score}:filtered"] = keeps(score, THETA, 2)
    sens = {
        "r24:theta0.10": keeps("r24", 0.10, 2),
        "r24:theta0.50": keeps("r24", 0.50, 2),
        "r24:drop_on_one": keeps("r24", THETA, 1),
    }
    rows = []
    curves: dict[tuple[str, str], list[dict]] = {}
    for score in ("r24", "pit24"):
        for name, kp in (
            ("plain", None),
            ("filtered", variants[f"{score}:filtered"]),
            *([(n.split(":")[1], k) for n, k in sens.items()] if score == "r24" else []),
        ):
            pts = []
            for k in det.K_GRID:
                r = sc.score(opens_for(ctx=ctx, score=score, k=float(k), keep=kp))
                pts.append({"k": float(k), "r": r})
                rows.append(
                    {
                        "signal": score,
                        "variant": name,
                        "k": float(k),
                        "matched": int(r["matched"].sum()),
                        "fa": float(r["fa"].sum()),
                        "apm": sc.apm(r),
                    }
                )
            curves[(score, name)] = pts
    pl.DataFrame(rows).write_parquet(OUT / "curves.parquet")

    carried = {}
    for src in ("LR_b20", "RO_b80"):
        opens, _ = det.carried_opens(ctx=ctx, source=src)
        carried[src] = sc.score(opens)

    def at_k(score: str, name: str, k: float) -> dict:
        return next(p["r"] for p in curves[(score, name)] if abs(p["k"] - k) < 1e-9)

    def match_recall(
        score: str, name: str, target_matched: int, group: np.ndarray | None
    ) -> tuple[float, dict]:
        """Largest k at which the arm matches at least `target_matched` shifts in the group."""
        best = None
        for p in curves[(score, name)]:
            m = p["r"]["matched"] if group is None else p["r"]["matched"] & group
            if m.sum() >= target_matched:
                best = p
        if best is None:
            best = curves[(score, name)][0]
        return best["k"], best["r"]

    def line(label: str, r: dict, k: float | None = None) -> str:
        m = r["matched"]
        mw = float((m * sc.weights).sum() / sc.weights.sum())
        parts = [
            f"{label:34s} k={k if k is not None else float('nan'):4.1f} matched {int(m.sum()):2d}/35 MW share {mw:.2f} FA {r['fa'].sum():4.0f} apm {sc.apm(r):5.1f}"
        ]
        for gname, g in (
            ("stratum", sc.stratum),
            ("rest", ~sc.stratum),
            ("counterpart", sc.counterpart),
            ("no counterpart", ~sc.counterpart),
            ("stratum+cp", sc.stratum & sc.counterpart),
        ):
            hits = int((m & g).sum())
            gw = sc.weights[g].sum()
            parts.append(
                f"{gname}: {hits}/{int(g.sum())} apm {sc.apm(r, g):5.1f} mw {float((m * sc.weights)[g].sum() / gw):.2f}"
            )
        parts.append(f"D8 FA {r['d8']} D9 FA {r['d9']} unlogged {r['unlogged']}")
        ev = sc.even_month
        parts.append(f"FA/yr even {r['fa'][ev].sum() / sc.month_years[ev].sum():.2f}")
        return "  ".join(parts)

    results: dict = {}
    for src_name, rate_key in (("LR_b20", "LR_b20"), ("RO_b80", "RO_b80")):
        print(
            f"\n=== Operating point of {src_name}'s false alarm rate (plain thresholds of experiments 308 and 309) ==="
        )
        print(line(f"carried {src_name}", carried[src_name]))
        for score in ("r24", "pit24"):
            kp = PLAIN_K[score][rate_key]
            plain = at_k(score, "plain", kp)
            print(line(f"{score} plain", plain, kp))
            for gname, g in (("all", None), ("stratum+cp", sc.stratum & sc.counterpart)):
                target = int(plain["matched"].sum() if g is None else (plain["matched"] & g).sum())
                for fname in ("filtered", "theta0.10", "theta0.50", "drop_on_one"):
                    if (score, fname) not in curves:
                        continue
                    k, r = match_recall(score, fname, target, g)
                    print(line(f"{score} {fname} (equal {gname})", r, k))
                    if fname == "filtered":
                        draws = sc.interval(
                            arms=[plain, r], group=sc.stratum & sc.counterpart, rng=rng
                        )["draws"]
                        ratio = draws[1] / draws[0]
                        print(
                            f"    stratum+cp apm: plain {sc.apm(plain, sc.stratum & sc.counterpart):.1f} {ci(draws[0])}, filtered {sc.apm(r, sc.stratum & sc.counterpart):.1f} {ci(draws[1])}; ratio {sc.apm(r, sc.stratum & sc.counterpart) / sc.apm(plain, sc.stratum & sc.counterpart):.2f} {ci(ratio)}"
                        )
                        pooled = sc.interval(arms=[plain, r], group=None, rng=rng)["draws"]
                        print(
                            f"    pooled apm: plain {sc.apm(plain):.1f} {ci(pooled[0])}, filtered {sc.apm(r):.1f} {ci(pooled[1])}; ratio {sc.apm(r) / sc.apm(plain):.2f} {ci(pooled[1] / pooled[0])}"
                        )
                        results[f"{src_name}:{score}:{gname}"] = {
                            "ratio_stratum_cp": ci(ratio),
                            "k_filtered": k,
                        }
    print("\n=== Alarms per match at fixed recall (pooled; largest k reaching the target) ===")
    for tname, target in RECALL_TARGETS.items():
        for score in ("r24", "pit24"):
            for fname in ("plain", "filtered"):
                k, r = match_recall(score, fname, target, None)
                reached = int(r["matched"].sum()) >= target
                print(
                    f"{tname}: {score} {fname}: k={k:.1f} matched {int(r['matched'].sum())} apm {sc.apm(r):.1f} {'' if reached else '(target not reached)'}"
                )
        for src_name in ("LR_b20", "RO_b80"):
            r = carried[src_name]
            print(
                f"{tname}: carried {src_name}: matched {int(r['matched'].sum())} apm {sc.apm(r):.1f}"
            )
    # How often the filter fires on the matched shifts: contradicting neighbours at the flagged origin.
    print("\n=== Filter firing on alarms (r24 at k=3.5) ===")
    p, f = at_k("r24", "plain", 3.5), at_k("r24", "filtered", 3.5)
    print(
        f"false alarms plain {p['fa'].sum():.0f} -> filtered {f['fa'].sum():.0f}; matched {int(p['matched'].sum())} -> {int(f['matched'].sum())}"
    )
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
