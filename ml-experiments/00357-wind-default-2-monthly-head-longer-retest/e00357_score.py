# ruff: noqa: E501, PLR0915
"""Step 2 of experiment 00357: score u3 (monthly retrained head) against default 2 on many more runs.

Rows: every kept case of the daily runs from 2025-07-01 (440 runs to 2026-09-13), from experiment
00356's case table. Scopes (all at true days 1 to 10, lead 24 to 240 hours, unless stated):

- `tue_year`: the 53 Tuesday runs of `fold_a_tuesdays_v1` (the brief's primary scope; month blocks).
- `all_year`: every run of 2025-07 to 2026-06 (month blocks).
- `rep_all`: every run of 2026-07-01 to 2026-09-13 (the replicate; run blocks, 95%).
- `rep_tue`: the Tuesday runs of 2026-07 onwards (experiment 00356's replicate; run blocks, 95%).
- `all_days`: every run, 2025-07 to 2026-09-13 (month blocks).

`u3` takes each month's rows from the head fitted on that month's first day. `u3p` (placebo) takes
them from a head fitted at a random earlier month, drawn from the months at least 3 months before
(the first 3 months use the first head). Contrasts are `arm - default 2`, so negative is better.
Writes `score.json` and `report.txt` to the data folder.

    PYTHONPATH=$PYTHONPATH_FOLDERS uv run --no-sync python e00357_score.py
"""

import json
import os
from datetime import UTC, datetime
from typing import Any, Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00356_heads as h356  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00356_score as s356  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00357_common as common  # noqa: E402
import e00357_heads as heads  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402

FIRST_RUN: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
REPLICATE_START: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
LEVEL: Final[float] = 0.99
ARMS: Final[tuple[str, ...]] = ("d1", "d2", "u3", "u3p")


def build_arms(*, frame: pl.DataFrame, rows: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Return the quantiles (5 seeds, rows, 11) of each arm on the scored rows, and the placebo draw."""
    idx = np.flatnonzero(rows)
    pos = np.full(frame.height, -1)
    pos[idx] = np.arange(len(idx))
    labels = heads.scored_months()
    month = frame["month"].to_numpy()
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    draw = {
        label: (
            labels[int(rng.integers(0, i - 2))] if i >= 3 else labels[0]
        )  # head index in 0..i-3
        for i, label in enumerate(labels)
    }
    d1_all = np.load(common.SOURCE_DIR / "d1.npy")
    out = {name: np.zeros((len(common.SEEDS), len(idx), 11), dtype=np.float32) for name in ARMS}
    for s in common.SEEDS:
        out["d2"][s] = np.load(common.SOURCE_DIR / "parts" / f"d2_h{s}.npy")[idx]
        out["d1"][s] = d1_all[s % 3][idx]
    for label in labels:
        in_month = (month == label) & rows
        for arm, head in (("u3", label), ("u3p", draw[label])):
            part_mask = heads.case_mask(frame=frame, first_month=head)
            part_rows = np.flatnonzero(part_mask)
            take = (month[part_rows] == label) & rows[part_rows]
            if int(take.sum()) != int(in_month.sum()):
                msg = f"{arm} {label}: {int(take.sum())} rows against {int(in_month.sum())}"
                raise ValueError(msg)
            target = pos[part_rows[take]]
            for s in common.SEEDS:
                part = np.load(common.PARTS_DIR / f"u3_s{s}_{head}.npy")
                out[arm][s][target] = part[take]
    return out, draw


def main() -> None:  # noqa: PLR0912
    """Score the arms and write the report."""
    frame, _ = h356.load()
    rows = (frame["kept"] & (frame["power_fcst_init_time"] >= FIRST_RUN)).to_numpy()
    arms, draw = build_arms(frame=frame, rows=rows)
    sub = frame.filter(pl.Series(rows))
    actual = sub["actual_norm"].to_numpy().astype(np.float64)
    hours = sub["lead_hours"].to_numpy().astype(np.float64)
    init = sub["power_fcst_init_time"]
    tue = sub["tuesday"].to_numpy()
    scored = sub["scored"].to_numpy()
    site = sub["site"].to_numpy()
    month = sub["month"].to_numpy()
    in_year = (init < REPLICATE_START).to_numpy()
    rep = ~in_year
    lead = {
        "headline": (hours >= 24) & (hours <= 240),
        "days_1_2": (hours >= 24) & (hours < 72),
        "days_3_10": (hours >= 72) & (hours <= 240),
        "all_leads": np.ones_like(tue),
    }
    scope_rows = {
        "tue_year": scored,
        "all_year": in_year,
        "rep_all": rep,
        "rep_tue": rep & tue,
        "all_days": np.ones_like(tue),
        "nontue_year": in_year & ~tue,
    }
    metrics = {name: s356.row_scores(q_all=arms[name], actual=actual) for name in ARMS}
    seed_qs = {
        name: np.stack(
            [
                s356.c303.quantile_score(quantiles=q.astype(np.float64), actual=actual)
                * s356.c303.POINTS_PER_FRACTION
                for q in arms[name]
            ]
        )
        for name in ("d2", "u3")
    }
    runs = init.dt.strftime("%Y-%m-%d").to_numpy()
    blocks_month = s356.Blocks(labels=month)
    blocks_run = s356.Blocks(labels=runs)
    lines: list[str] = []
    result: dict[str, Any] = {"placebo_draw": draw}

    def say(text: str = "") -> None:
        lines.append(text)
        print(text, flush=True)

    def keep(*, scope: str, lead_scope: str = "headline") -> np.ndarray:
        return scope_rows[scope] & lead[lead_scope]

    def interval(*, values: np.ndarray, scope: str, lead_scope: str = "headline") -> tuple:
        k = keep(scope=scope, lead_scope=lead_scope)
        if scope.startswith("rep"):
            return blocks_run.interval(values=values, keep=k, level=0.95)
        return blocks_month.interval(values=values, keep=k, level=LEVEL)

    say("== run counts (kept runs with at least one headline-lead row)")
    for scope in scope_rows:
        k = keep(scope=scope)
        say(f"{scope}: {len(np.unique(runs[k]))} runs, {int(k.sum()):,} rows")
        result[f"runs_{scope}"] = len(np.unique(runs[k]))

    say()
    say("== absolute scores, headline (points of capacity)")
    say("arm | " + " | ".join(scope_rows))
    for name in ARMS:
        vals = [metrics[name]["qs"][keep(scope=s)].mean() for s in scope_rows]
        say(f"{name} | " + " | ".join(f"{v:.3f}" for v in vals))
        result[f"abs_{name}"] = dict(zip(scope_rows, map(float, vals), strict=True))
    say("seed sd of the headline (tue_year, all_days):")
    for name, q in seed_qs.items():
        say(
            f"{name}: {q[:, keep(scope='tue_year')].mean(axis=1).std():.3f}, "
            f"{q[:, keep(scope='all_days')].mean(axis=1).std():.3f}"
        )

    say()
    say("== gate (tue_year, headline): d2 10.518, d1 10.752, u3 10.480 (experiment 356)")
    say(
        f"d2 {result['abs_d2']['tue_year']:.3f}, d1 {result['abs_d1']['tue_year']:.3f}, "
        f"u3 {result['abs_u3']['tue_year']:.3f}"
    )

    say()
    say("== gain over default 2 (arm minus default 2; month-block 99% or run-block 95% for rep_*)")
    say("arm | scope | lead scope | point | low | high")
    contrasts: dict[str, dict[str, list[float]]] = {}
    for name in ("u3", "u3p"):
        diff = metrics[name]["qs"] - metrics["d2"]["qs"]
        contrasts[name] = {}
        for scope in scope_rows:
            for lead_scope in ("headline", "days_1_2", "days_3_10", "all_leads"):
                if lead_scope != "headline" and scope not in ("tue_year", "all_year", "all_days"):
                    continue
                p, lo, hi = interval(values=diff, scope=scope, lead_scope=lead_scope)
                contrasts[name][f"{scope}/{lead_scope}"] = [p, lo, hi]
                say(f"{name} | {scope} | {lead_scope} | {p:+.3f} | {lo:+.3f} | {hi:+.3f}")
    diff = metrics["u3"]["qs"] - metrics["u3p"]["qs"]
    for scope in ("tue_year", "all_year", "all_days"):
        p, lo, hi = interval(values=diff, scope=scope)
        contrasts.setdefault("u3_minus_u3p", {})[scope] = [p, lo, hi]
        say(f"u3 minus u3p | {scope} | headline | {p:+.3f} | {lo:+.3f} | {hi:+.3f}")
    result["contrasts"] = contrasts

    say()
    say("== monthly paired differences, u3 minus d2 (headline) and a t interval over months")
    diff = metrics["u3"]["qs"] - metrics["d2"]["qs"]
    labels = heads.scored_months()
    monthly = []
    for label in labels:
        k = keep(scope="all_days") & (month == label)
        if k.any():
            monthly.append((label, float(diff[k].mean()), len(np.unique(runs[k]))))
            say(f"{label} | {monthly[-1][1]:+.3f} | {monthly[-1][2]} runs")
    arr = np.array([m[1] for m in monthly])
    half = stats.t.ppf(0.995, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr))
    say(
        f"mean over {len(arr)} months {arr.mean():+.3f}, 99% t interval {arr.mean() - half:+.3f} to {arr.mean() + half:+.3f}"
    )
    result["monthly"] = monthly
    result["t_interval_all_days"] = [
        float(arr.mean()),
        float(arr.mean() - half),
        float(arr.mean() + half),
    ]
    say(f"months with u3 better than d2: {int((arr < 0).sum())} of {len(arr)}")

    say()
    say("== tails (headline scope): share beyond each quantile (%), pinball losses")
    say(
        "arm | scope | below q01 | below q05 | above q95 | above q99 | p01 | p05 | p95 | p99 | 5-95 coverage"
    )
    tails: dict[str, list[float]] = {}
    for scope in ("tue_year", "all_days", "rep_all"):
        k = keep(scope=scope)
        for name in ARMS:
            m = metrics[name]
            vals = [
                100 * m["below_q01"][k].mean(),
                100 * m["below_q05"][k].mean(),
                100 * m["above_q95"][k].mean(),
                100 * m["above_q99"][k].mean(),
                m["p01"][k].mean(),
                m["p05"][k].mean(),
                m["p95"][k].mean(),
                m["p99"][k].mean(),
                100 * m["covered"][k].mean(),
            ]
            tails[f"{scope}/{name}"] = vals
            say(f"{name} | {scope} | " + " | ".join(f"{v:.3f}" for v in vals))
    result["tails"] = tails

    say()
    say("== harm: one-sided 95% upper bound of u3 minus default 2 (headline, p01, p99)")
    harm: dict[str, dict[str, float]] = {}
    for scope in ("tue_year", "all_year"):
        k = keep(scope=scope)
        out = {
            metric: blocks_month.upper95(
                values=metrics["u3"][metric] - metrics["d2"][metric], keep=k
            )
            for metric in ("qs", "p01", "p99")
        }
        harm[scope] = out
        say(f"{scope}: headline {out['qs']:+.3f}, p01 {out['p01']:+.3f}, p99 {out['p99']:+.3f}")
    result["harm"] = harm

    say()
    say("== by farm and leave-one-farm-out (u3 minus d2, headline)")
    for scope in ("tue_year", "all_year", "rep_all", "all_days"):
        k = keep(scope=scope)
        parts, loo = [], []
        for farm in common.c356.FARMS:
            kf = k & (site == farm)
            parts.append(f"{farm} {diff[kf].mean():+.3f}")
            loo.append(f"without {farm} {diff[k & (site != farm)].mean():+.3f}")
        say(f"{scope} | " + " | ".join(parts) + " || " + " | ".join(loo))

    say()
    say(
        "== checks: median minus mean forecast per farm; mean-forecast and median absolute error (all_days, headline)"
    )
    k = keep(scope="all_days")
    for name in ("d1", "d2", "u3", "u3p"):
        m = metrics[name]
        parts = [
            f"{farm} {m['med_minus_mean'][k & (site == farm)].mean():+.2f}"
            for farm in common.c356.FARMS
        ]
        say(
            f"{name} | "
            + " | ".join(parts)
            + f" || mean-fcst MAE {m['mae_mean'][k].mean():.3f}, median MAE {m['mae_median'][k].mean():.3f}"
        )
    say("mean-forecast absolute error contrast, u3 minus d2:")
    for scope in ("tue_year", "all_year", "all_days"):
        dm = metrics["u3"]["mae_mean"] - metrics["d2"]["mae_mean"]
        say(f"{scope}: {dm[keep(scope=scope)].mean():+.3f}")

    (common.DATA_DIR / "score.json").write_text(json.dumps(result, indent=1, default=float))
    (common.DATA_DIR / "report.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
