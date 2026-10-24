# ruff: noqa: E501, RUF059, PLR0912, PLR0915, D208
r"""Stage 3: cross-fitted selection, planned contrasts, controls and the printed report.

    PYTHONPATH=ml-experiments/00295-*:ml-experiments/00328-*:ml-experiments/00331-* \
        uv run --no-sync python ml-experiments/00331-*/e00331_report.py
"""

from __future__ import annotations

import os

import e00295_common as c295  # ty: ignore[unresolved-import]
import e00331_common as c
import numpy as np
import polars as pl

NOFILTER = os.environ.get("NOFILTER") == "1"
SCORE_DIR = c.WORK_DIR / "scores_nofilter" if NOFILTER else c.SCORE_DIR
FIXED_GAIN = 0.0 if NOFILTER else 0.10
N_BOOT = 4000
SEED = 331
D8_AFTER = ("2023-09-01", "2023-11-01")
NOMINAL_01 = 0.01
HALF_LIVES = c.HALF_LIVES
H_ARMS = [f"H{h}" for h in HALF_LIVES]
S_ARMS = [f"SH{h}" for h in HALF_LIVES]


def load_real() -> pl.DataFrame:
    """Return the real-record score rows."""
    files = sorted(SCORE_DIR.glob("real_*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files]).filter(pl.col("group") == "all")


def per_site_month(*, frame: pl.DataFrame, arm: str, ridge: float, gain: float) -> pl.DataFrame:
    """Return the per-site, per-month metrics of one configuration."""
    f = frame.filter((pl.col("arm") == arm) & (pl.col("ridge") == ridge) & (pl.col("gain") == gain))
    return f.select(
        "site", "month", "n",
        score=pl.col("pinball_sum") / pl.col("n"), mae=pl.col("abs_sum") / pl.col("n"),
        p95=pl.col("p95_sum") / pl.col("n"), p99=pl.col("p99_sum") / pl.col("n"),
        below01=pl.col("below01") / pl.col("n"), above99=pl.col("above99") / pl.col("n"),
    ).sort("site", "month")  # fmt: skip


def month_matrix(
    *,
    frame: pl.DataFrame,
    arm: str,
    ridge: float,
    gain: float,
    metric: str,
    sites: list[str],
    months: list[str],
) -> np.ndarray:
    """Return a (months, sites) matrix of one metric (NaN where missing)."""
    f = per_site_month(frame=frame, arm=arm, ridge=ridge, gain=gain)
    out = np.full((len(months), len(sites)), np.nan)
    mi = {m: i for i, m in enumerate(months)}
    si = {s: i for i, s in enumerate(sites)}
    for site, month, value in f.select("site", "month", metric).iter_rows():
        out[mi[month], si[site]] = value
    return out


def parity(month: str) -> int:
    """Return 0 for an even calendar month, 1 for an odd one."""
    return int(month[5:7]) % 2


def cross_fit(*, frame: pl.DataFrame, sites: list[str], months: list[str]) -> dict:
    """Select ridge, gain and half-life on one parity of months; return the other parity's matrices.

    Returns a dict of metric -> arm -> (months, sites) matrices, where each month uses the
    configuration chosen on the other parity, plus the choices.
    """
    metrics = ("score", "mae", "p95", "p99", "below01", "above99")
    par = np.array([parity(m) for m in months])
    cache: dict[tuple[str, float, float, str], np.ndarray] = {}

    def mat(arm: str, ridge: float, gain: float, metric: str) -> np.ndarray:
        k = (arm, ridge, gain, metric)
        if k not in cache:
            cache[k] = month_matrix(
                frame=frame,
                arm=arm,
                ridge=ridge,
                gain=gain,
                metric=metric,
                sites=sites,
                months=months,
            )
        return cache[k]

    arms = ["R0", "W24", *H_ARMS, *S_ARMS]
    out: dict = {m: {} for m in metrics}
    choices: list[dict] = []
    for sel in (0, 1):
        score_mask = par != sel
        sel_mask = par == sel
        best_cfg: dict[str, tuple[float, float, float]] = {}
        for arm in arms:
            vals = {
                (r, g): np.nanmean(mat(arm, r, g, "score")[sel_mask])
                for r in c.RIDGES
                for g in sorted(frame["gain"].unique().to_list())
            }
            (r, g), v = min(vals.items(), key=lambda kv: kv[1])
            best_cfg[arm] = (r, g, v)
            choices.append(
                {
                    "selected_on": "odd" if sel == 1 else "even",
                    "arm": arm,
                    "ridge": r,
                    "gain": g,
                    "selection_score": v,
                }
            )
        best_h = min(H_ARMS, key=lambda a: best_cfg[a][2])
        best_s = f"S{best_h}"
        choices.append(
            {
                "selected_on": "odd" if sel == 1 else "even",
                "arm": "bestH",
                "ridge": best_cfg[best_h][0],
                "gain": best_cfg[best_h][1],
                "selection_score": best_cfg[best_h][2],
            }
        )
        for metric in metrics:
            for arm in arms:
                r, g, _ = best_cfg[arm]
                m = mat(arm, r, g, metric)
                out[metric].setdefault(arm, np.full((len(months), len(sites)), np.nan))[
                    score_mask
                ] = m[score_mask]
            for label, source in (("bestH", best_h), ("bestS", best_s)):
                r, g, _ = best_cfg[source]
                m = mat(source, r, g, metric)
                out[metric].setdefault(label, np.full((len(months), len(sites)), np.nan))[
                    score_mask
                ] = m[score_mask]
    out["choices"] = choices  # type: ignore[assignment]
    return out


def pooled(matrix: np.ndarray, sites_mask: np.ndarray | None = None) -> np.ndarray:
    """Return the per-month mean over (selected) sites."""
    m = matrix if sites_mask is None else matrix[:, sites_mask]
    return np.nanmean(m, axis=1)


def month_bootstrap(
    *, diff: np.ndarray, level: float = 0.95, seed: int = SEED
) -> tuple[float, float, float, float]:
    """Return mean, lower, upper (two-sided at `level`) and the one-sided upper 95% bound.

    Resamples whole calendar months (rows of `diff`).
    """
    rng = np.random.default_rng(seed)
    d = diff[np.isfinite(diff)]
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(d.mean()), float(lo), float(hi), float(np.quantile(means, 0.95))


def moving_block_bootstrap(
    *, diff: np.ndarray, block: int = 6, level: float = 0.95
) -> tuple[float, float, float]:
    """Return mean and two-sided interval, resampling overlapping blocks of `block` adjacent months."""
    rng = np.random.default_rng(SEED)
    d = diff[np.isfinite(diff)]
    n = len(d)
    starts = rng.integers(0, n - block + 1, size=(N_BOOT, -(-n // block)))
    idx = (starts[:, :, None] + np.arange(block)).reshape(N_BOOT, -1)[:, :n]
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(d.mean()), float(lo), float(hi)


def effective_winters(*, sites: list[str]) -> dict[str, float]:
    """Return the median over sites of each weighted arm's effective number of winters at the last cut."""
    cache = c295.load_cache()
    cut = c.LAST_CUT
    arms = tuple(c.WEIGHTED_ARMS)
    rows: dict[str, list[float]] = {a: [] for a in ("R0", "W24", *arms)}
    for site in sites:
        index, ends = c.training_end_times(cache=cache, site=site, cut=cut)
        months = (ends - np.timedelta64(1, "s")).astype("datetime64[M]").astype(int) % 12 + 1
        winter = np.isin(months, [12, 1, 2])
        w = c.row_weights(end_times=ends, cut=cut, arms=arms)
        rows["R0"].append(float(winter.sum() / 4320.0))
        for k, a in enumerate(arms):
            ww = w[k][winter]
            rows[a].append(float(ww.sum() ** 2 / (ww**2).sum() / 4320.0))
        w24 = ends > ends.max() - np.timedelta64(730, "D")
        rows["W24"].append(float((winter & w24).sum() / 4320.0))
    return {a: float(np.median(v)) for a, v in rows.items()}


def main() -> None:
    """Print the report."""
    frame = load_real()
    sites = sorted(frame["site"].unique().to_list())
    months = sorted(frame["month"].unique().to_list())
    no_d8 = np.array([s != "D8" for s in sites])
    print(
        f"Primaries: {len(sites)} {sites}; scored cuts {months[0]} to {months[-1]} ({len(months)})"
    )

    # Controls: reproduction gate and placebo at ridge 0.1, gain 0.10.
    if NOFILTER:
        print("NO-FILTER variant (Kalman gain 0): reproduction gate skipped")
    print(
        "\n== Reproduction gate (points; 328 arms A and B, per Primary mean over scored months) =="
    )
    worst = {"R0": 0.0, "W24": 0.0}
    for site in sites:
        old = pl.read_parquet(c.EXP328_DIR / "work" / "model" / f"real_{site}.parquet").filter(
            pl.col("month").is_in(months)
        )
        for mine, theirs in (("R0", "A"), ("W24", "B")):
            o = old.filter(pl.col("arm") == theirs)
            ov = float(o["pinball_sum"].sum()) / float(o["n"].sum())
            f = frame.filter(
                (pl.col("arm") == mine)
                & (pl.col("ridge") == 0.1)
                & (pl.col("gain") == FIXED_GAIN)
                & (pl.col("site") == site)
            )
            nv = float(f["pinball_sum"].sum()) / float(f["n"].sum())
            worst[mine] = max(worst[mine], abs(nv - ov))
    print(
        f"max |difference| over Primaries: R0 vs A {worst['R0']:.4f}, W24 vs B {worst['W24']:.4f} (gate 0.005)"
    )
    placebo_gap = 0.0
    pa = per_site_month(frame=frame, arm="P", ridge=0.1, gain=FIXED_GAIN)
    ra = per_site_month(frame=frame, arm="R0", ridge=0.1, gain=FIXED_GAIN)
    placebo_gap = float((pa["score"] - ra["score"]).abs().max())  # ty: ignore[invalid-argument-type]
    print(
        f"placebo (half-life 1000 years) max |score difference| from R0: {placebo_gap:.5f} (gate 0.001)"
    )

    cf = cross_fit(frame=frame, sites=sites, months=months)
    print("\n== Selected settings (ridge, gain, selection-half score) ==")
    for ch in cf["choices"]:
        print(ch)
    arms = ["R0", "W24", *H_ARMS, "bestH", *S_ARMS, "bestS"]
    print("\n== Arms (cross-fitted; mean over all Primaries; points) ==")
    print(
        "arm | score | MAE median | q95 | q99 | cov q01 below | cov q99 above | winter score | score ex-D8"
    )
    winter_mask = np.array([int(m[5:7]) in (12, 1, 2) for m in months])
    for a in arms:
        s = pooled(cf["score"][a])
        print(
            f"{a} | {np.nanmean(s):.3f} | {np.nanmean(pooled(cf['mae'][a])):.3f} | {np.nanmean(pooled(cf['p95'][a])):.4f} | {np.nanmean(pooled(cf['p99'][a])):.4f} | {np.nanmean(pooled(cf['below01'][a])):.4f} | {np.nanmean(pooled(cf['above99'][a])):.4f} | {np.nanmean(s[winter_mask]):.3f} | {np.nanmean(pooled(cf['score'][a], no_d8)):.3f}"
        )

    def contrast(
        name: str,
        a: str,
        b: str,
        metric: str = "score",
        mask: np.ndarray | None = None,
        month_mask: np.ndarray | None = None,
    ) -> None:
        d = pooled(cf[metric][a], mask) - pooled(cf[metric][b], mask)
        if month_mask is not None:
            d = d[month_mask]
        mean, lo, hi, up = month_bootstrap(diff=d)
        print(
            f"{name}: {a} minus {b} on {metric}: {mean:+.4f} (95% {lo:+.4f} to {hi:+.4f}; one-sided upper 95% {up:+.4f})"
        )

    print("\n== Planned contrasts (pooled over all Primaries, month-block bootstrap) ==")
    contrast("C1", "bestH", "R0")
    contrast("C2", "bestH", "W24")
    contrast("  descriptive", "W24", "R0")
    contrast("C3", "bestS", "bestH")
    for h in H_ARMS:
        contrast("  each", h, "R0")
    contrast("  ex-D8", "bestH", "R0", mask=no_d8)
    contrast("  winter", "bestH", "R0", month_mask=winter_mask)
    for metric in ("mae", "p95", "p99"):
        contrast("  check", "bestH", "R0", metric=metric)
    print(
        "\n== Post hoc: moving 6-month-block bootstrap (monthly differences are autocorrelated) =="
    )
    for name, a, b, mask in (
        ("C1", "bestH", "R0", None),
        ("C1 ex-D8", "bestH", "R0", no_d8),
        ("C2", "bestH", "W24", None),
        ("C3", "bestS", "bestH", None),
    ):
        d = pooled(cf["score"][a], mask) - pooled(cf["score"][b], mask)
        mean, lo, hi = moving_block_bootstrap(diff=d)
        lag1 = float(np.corrcoef(d[:-1], d[1:])[0, 1])
        print(
            f"{name}: {a} minus {b}: {mean:+.4f} (95% {lo:+.4f} to {hi:+.4f}); lag-1 autocorrelation {lag1:.2f}"
        )
    print(
        "\n== Post hoc: bestH minus R0 by calendar year of the forecast origin (pooled, all Primaries) =="
    )
    d = pooled(cf["score"]["bestH"]) - pooled(cf["score"]["R0"])
    for year in sorted({m[:4] for m in months}):
        sel = np.array([m[:4] == year for m in months])
        print(f"{year}: {np.nanmean(d[sel]):+.4f} ({sel.sum()} origins)")
    print(
        "\n== Non-inferiority of bestH against R0 per Primary (margin 0.01, one-sided 95% upper bound) =="
    )
    for i, site in enumerate(sites):
        d = cf["score"]["bestH"][:, i] - cf["score"]["R0"][:, i]
        mean, _, _, up = month_bootstrap(diff=d)
        flag = "excluded (D8)" if site == "D8" else ("PASS" if up <= 0.01 else "LOOK")
        print(f"{site}: diff {mean:+.4f}, upper {up:+.4f} {flag}")
    d_pool = pooled(cf["score"]["bestH"], no_d8) - pooled(cf["score"]["R0"], no_d8)
    print("pooled ex-D8 upper 95% bound:", f"{month_bootstrap(diff=d_pool)[3]:+.4f}")

    print("\n== D8 on its own ==")
    i8 = sites.index("D8")
    after = np.array([D8_AFTER[0] <= m <= D8_AFTER[1] for m in months])
    for a in ["R0", "W24", *H_ARMS, "bestH"]:
        sc = cf["score"][a][:, i8]
        print(
            f"{a}: all months {np.nanmean(sc):.3f}; after break {np.nanmean(sc[after]):.3f}; q99 above rate after {np.nanmean(cf['above99'][a][:, i8][after]):.4f}; q01 below rate after {np.nanmean(cf['below01'][a][:, i8][after]):.4f}"
        )

    print("\n== Effective winters at the last origin (median over Primaries) ==")
    print(effective_winters(sites=sites))

    print("\n== Injected-step worlds (known answer; ridge 0.1, gain per arm fixed at 0.10) ==")
    inj = pl.concat([pl.read_parquet(f) for f in sorted(SCORE_DIR.glob("inj_*.parquet"))]).filter(
        pl.col("gain") == FIXED_GAIN
    )
    if inj.height:
        base = inj.filter(pl.col("arm") == "R0")
        for g in ("d0_30", "d30_90", "d90p"):
            for a in ["W24", *H_ARMS, "P"]:
                f = inj.filter((pl.col("arm") == a) & (pl.col("group") == g)).join(
                    base.filter(pl.col("group") == g).select(
                        "world", "site", "month", n0="n", pb0="pinball_sum"
                    ),
                    on=["world", "site", "month"],
                )
                per_world = f.group_by("world", "site").agg(
                    d=(pl.col("pinball_sum").sum() / pl.col("n").sum())
                    - (pl.col("pb0").sum() / pl.col("n0").sum())
                )
                onset = np.array([w.split("_")[2] for w in per_world["world"]])
                d = per_world["d"].to_numpy()
                blocks = np.unique(onset)
                block_means = np.array([d[onset == b].mean() for b in blocks])
                mean, lo, hi, _ = month_bootstrap(diff=block_means)
                print(
                    f"{g} {a} minus R0: {mean:+.4f} (95% {lo:+.4f} to {hi:+.4f}; {len(d)} worlds, {len(blocks)} onsets)"
                )
        for a in ["W24", *H_ARMS]:
            f = (
                inj.filter((pl.col("arm") == a) & pl.col("group").is_in(["d0_30", "d30_90"]))
                .group_by("world", "site", "month")
                .agg(pb=pl.col("pinball_sum").sum(), n=pl.col("n").sum())
            )
            g = (
                inj.filter((pl.col("arm") == "R0") & pl.col("group").is_in(["d0_30", "d30_90"]))
                .group_by("world", "site", "month")
                .agg(pb0=pl.col("pinball_sum").sum(), n0=pl.col("n").sum())
            )
            j = (
                f.join(g, on=["world", "site", "month"])
                .group_by("world", "site")
                .agg(
                    d=pl.col("pb").sum() / pl.col("n").sum()
                    - pl.col("pb0").sum() / pl.col("n0").sum()
                )
            )
            onset = np.array([w.split("_")[2] for w in j["world"]])
            dv = j["d"].to_numpy()
            bm = np.array([dv[onset == b].mean() for b in np.unique(onset)])
            mean, lo, hi, _ = month_bootstrap(diff=bm)
            print(
                f"KNOWN ANSWER first 90 days pooled {a} minus R0: {mean:+.4f} (95% {lo:+.4f} to {hi:+.4f}; {len(dv)} worlds)"
            )
        h90 = inj.filter(pl.col("group").is_in(["d0_30", "d30_90"]))
        for a in ["R0", "W24", *H_ARMS]:
            f = h90.filter(pl.col("arm") == a)
            print(
                f"first 90 days {a}: score {float(f['pinball_sum'].sum()) / float(f['n'].sum()):.3f}"
            )


if __name__ == "__main__":
    main()
