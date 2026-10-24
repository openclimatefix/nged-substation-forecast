# ruff: noqa: E501
"""Same-threshold comparison and the filter's branch counts (added after the plan review).

Run from the repository root with `PYTHONPATH` set to the folders of experiments 309 and 315:

    PYTHONPATH=ml-experiments/00309-lsc-two-forecast-gap-choosing-without-truth:\
    ml-experiments/00315-lsc-neighbour-residual-filter \
    uv run --no-sync python ml-experiments/00315-lsc-neighbour-residual-filter/e00315_extra.py
"""

from __future__ import annotations

import json

import e00309_detect as det
import e00315_run as run
import numpy as np
import polars as pl


def main() -> None:
    """Print filtered-versus-plain at the plain threshold, and how often each filter branch fires."""
    rng = np.random.default_rng(run.SEED + 1)
    ctx = det.build_context()
    neighbours = json.loads((run.OUT / "neighbours.json").read_text())
    sc = run.Scorer(ctx, pl.read_parquet(run.OUT / "ceiling.parquet"))
    grp = sc.stratum & sc.counterpart
    for score in ("r24", "pit24"):
        keep = {
            s: run.keep_mask(
                ctx=ctx,
                neighbours=neighbours,
                series=s,
                score=score,
                theta=run.THETA,
                max_contradict=2,
            )
            for s in ctx.real
        }
        for src, k in run.PLAIN_K[score].items():
            plain = sc.score(run.opens_for(ctx=ctx, score=score, k=k, keep=None))
            filt = sc.score(run.opens_for(ctx=ctx, score=score, k=k, keep=keep))
            d = sc.interval(arms=[plain, filt], group=grp, rng=rng)["draws"]
            p = sc.interval(arms=[plain, filt], group=None, rng=rng)["draws"]
            print(
                f"{score} at {src}'s plain k={k}: matched {int(plain['matched'].sum())} -> {int(filt['matched'].sum())} (stratum+cp {int((plain['matched'] & grp).sum())} -> {int((filt['matched'] & grp).sum())}); FA {plain['fa'].sum():.0f} -> {filt['fa'].sum():.0f}; D8 FA {plain['d8']}->{filt['d8']}, D9 FA {plain['d9']}->{filt['d9']}"
            )
            print(
                f"    stratum+cp apm {sc.apm(plain, grp):.1f} -> {sc.apm(filt, grp):.1f}, ratio {sc.apm(filt, grp) / sc.apm(plain, grp):.2f} {run.ci(d[1] / d[0])}; pooled ratio {sc.apm(filt) / sc.apm(plain):.2f} {run.ci(p[1] / p[0])}"
            )
    # Branch counts at every flagged origin of the plain r24 arm at k=2.6 (the looser threshold).
    for score, k in (("r24", 2.6), ("pit24", 2.4)):
        tot = {
            "flagged": 0,
            "no finite neighbour": 0,
            "confirmed": 0,
            "contradicted by 2+": 0,
            "neither (kept)": 0,
        }
        for s, v in ctx.real.items():
            z = np.abs(v.score[score]) / v.sigma[score]
            flag = np.isfinite(z) & (z > k)
            a = v.score[score] * (v.cap / 100.0 if score == "r24" else 1.0)
            conf = np.zeros(len(a), bool)
            contra = np.zeros(len(a), int)
            fin = np.zeros(len(a), int)
            for n in neighbours[str(s)]:
                o = ctx.real[n]
                b = o.score[score] * (o.cap / 100.0 if score == "r24" else 1.0)
                ok = np.isfinite(a) & np.isfinite(b)
                big = ok & (np.abs(b) >= run.THETA * np.abs(a))
                conf |= big & (a * b < 0)
                contra += big & (a * b > 0)
                fin += ok
            tot["flagged"] += int(flag.sum())
            tot["no finite neighbour"] += int((flag & (fin == 0)).sum())
            tot["confirmed"] += int((flag & (fin > 0) & conf).sum())
            tot["contradicted by 2+"] += int((flag & (fin > 0) & ~conf & (contra >= 2)).sum())
            tot["neither (kept)"] += int((flag & (fin > 0) & ~conf & (contra < 2)).sum())
        print(f"branches of flagged origins, {score} at k={k}: {tot}")


if __name__ == "__main__":
    main()
