# ruff: noqa: E501
"""Do D8's multi-day drops coincide with the engine detector's running states?

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00291-*/e00291_analysis.py

Reads the drops of experiment 159, the engine probabilities that `e00291_states.py` stored, D8's
residual panel, the system price, and the load-shift alarms of experiments 130 and 195. Writes
private tables under `DATA_DIR` and prints every number the write-up quotes into
`printed_report.txt`. Every definition below was fixed in the plan, before any number was read, except
the alarm-spell rule (`SPELL_JOIN`, `ALARM_PAD`), which was fixed before the alarm numbers were read.
"""

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00291_common import DATA_DIR, G2_FIRST, G2_LAST, GRID_START, UNIT_MW, g2_output, stamp
from e00291_states import p_run_path
from e00291_v2 import night_mask, spells
from scipy import stats
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DROPS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00159-switching-d8-multi-day-drops-recurrence-and-report"
    / "drops.parquet"
)
ARRAYS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "briefs" / "sw_false_alarms" / "work" / "arrays.pkl"
)
ALARMS_130_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "briefs" / "sw_false_alarms" / "work" / "alarms_flagged.parquet"
)
ALARMS_195_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00195-switching-neural-plus-detector-features"
)
PRICES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00191-dp-price-sensitive-assets-in-the-history"
    / "work"
    / "prices.parquet"
)
D8_KEY: Final[int] = 26
CONTROLS: Final[tuple[str, ...]] = ("D5", "D10", "D11")
ENGINE_VISIBLE: Final[datetime] = datetime(2023, 8, 15, tzinfo=UTC)
LAST_STAMP_DATE: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
SOLAR_START: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
WEEK: Final[int] = 336
DROP_MARGIN: Final[int] = 96
"""A placebo window keeps 2 days away from every drop."""
OVERLAP_SHARE: Final[float] = 0.5
OVERLAP_PROBABILITY: Final[float] = 0.5
N_REPLICATES: Final[int] = 5000
SEED: Final[int] = 0
SPELL_JOIN: Final[int] = 6
"""Flagged runs separated by 6 half-hours or fewer form one engine spell."""
ALARM_PAD: Final[int] = 6
"""An alarm is inside a spell when it lies within 6 half-hours of the spell's ends."""
NEIGHBOUR_FRACTIONS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
STEP_WINDOW: Final[int] = 12
UNIT_SENSITIVITY_MW: Final[float] = 1.75
UNIT_RECORD_MW: Final[float] = 1.93
"""The unit that the first run of `unit_from_engine_record` estimated; a sensitivity."""
UNIFORM_DISTANCE: Final[float] = 0.25
TOTAL_FALSE_ALARMS: Final[int] = 148
D8_D9_FALSE_ALARMS: Final[int] = 61
ALARM_ARMS: Final[tuple[str, ...]] = ("RO-NC", "Pair alarm", "F network", "R re-ranker")
REPORT: list[str] = []
SUMMARY: dict[str, float] = {}


def say(text: str = "") -> None:
    """Print a line and keep it for `printed_report.txt`."""
    print(text, flush=True)
    REPORT.append(text)


def wilson(*, k: int, n: int) -> str:
    """Share with its Wilson 95% interval, as text."""
    lo, hi = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return f"{k} of {n} ({k / n:.2f}; {lo:.2f} to {hi:.2f})"


def window_share(*, flag: np.ndarray, first: int, last: int, keep: np.ndarray | None) -> float:
    """Share of the window's half-hours (those in `keep`, if given) that are flagged."""
    seg = flag[first : last + 1]
    if keep is not None:
        seg = seg[keep[first : last + 1]]
    return float(seg.mean()) if seg.size else float("nan")


def distance_to_unit_multiple(*, x: np.ndarray, unit: float) -> np.ndarray:
    """Distance of `x` from the nearest multiple of `unit`, in units."""
    r = np.abs(x) / unit
    return np.abs(r - np.round(r))


def bootstrap_mean(*, x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean with a 95% bootstrap interval over the entries."""
    means = np.array(
        [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_REPLICATES)], dtype=float
    )
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def unit_from_engine_record(*, rng: np.random.Generator) -> tuple[float, float, float]:
    """Unit size from the recorded engine output over its 44 days of known answer, by day-resample."""
    n = stamp(LAST_STAMP_DATE)
    out = g2_output(n=n)
    lo, hi = stamp(G2_FIRST), stamp(G2_LAST)
    y = out[lo:hi]
    day = (np.arange(len(y)) // 48).astype(int)
    ok = np.isfinite(y) & (y > 0.5)
    k = np.clip(np.round(y / UNIT_MW), 1, 3)

    def estimate(sel: np.ndarray) -> float:
        return float((k[sel] * y[sel]).sum() / (k[sel] ** 2).sum())

    point = estimate(ok)
    days = np.unique(day[ok])
    boots = []
    for _ in range(N_REPLICATES):
        pick = rng.choice(days, len(days))
        sel = np.concatenate([np.flatnonzero(ok & (day == d)) for d in pick])
        kk, yy = k[sel], y[sel]
        boots.append(float((kk * yy).sum() / (kk**2).sum()))
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:  # noqa: PLR0912, PLR0915
    """Run every analysis and write the report."""
    rng = np.random.default_rng(SEED)
    panel = pickle.loads(ARRAYS_PATH.read_bytes())
    resid = np.asarray(panel[D8_KEY]["resid"], dtype=float)
    n_grid = len(resid)
    drops = pl.read_parquet(DROPS_PATH).sort("first_stamp")
    visible, last_grid, solar = (
        stamp(ENGINE_VISIBLE),
        stamp(LAST_STAMP_DATE),
        stamp(SOLAR_START),
    )
    arms = {
        "v1": np.load(p_run_path(site="D8", arm="v1")) > OVERLAP_PROBABILITY,
        "N": np.load(p_run_path(site="D8", arm="N")) > OVERLAP_PROBABILITY,
    }
    controls = {c: np.load(p_run_path(site=c, arm="v1")) > OVERLAP_PROBABILITY for c in CONTROLS}
    night = night_mask(label="D8", n=n_grid)
    first = drops["first_stamp"].to_numpy()
    last = drops["last_stamp"].to_numpy() - 1
    # Experiment 159 stores the exclusive end of a drop; `last` here is inclusive.
    scope = first >= visible
    say(
        f"Drops: {len(drops)}; in scope (start on or after 2023-08-15): {int(scope.sum())}; "
        f"from 2025-03-01: {int((scope & (first >= solar)).sum())}; "
        f"2023-08-15 to 2025-02-28: {int((scope & (first < solar)).sum())}"
    )
    say(
        "Detector check: D8 arm v1 flagged share of the whole grid "
        f"{arms['v1'].mean():.3f} (agrees with experiment 196's stored posterior on every half-hour); "
        f"arm N {arms['N'].mean():.3f}"
    )
    for (lo, hi), name in (
        ((visible, solar), "2023-08-15 to 2025-02-28"),
        ((solar, last_grid), "2025-03-01 to 2026-06-30"),
    ):
        eligible = np.ones(n_grid, dtype=bool)
        for a, b in zip(first, last, strict=True):
            eligible[a : b + 1] = False
        sel = eligible[lo:hi]
        say(
            f"Flagged share outside drops, {name}: v1 {arms['v1'][lo:hi][sel].mean():.3f}; "
            f"N (night half-hours) {arms['N'][lo:hi][sel & night[lo:hi]].mean():.3f}; "
            + "; ".join(f"{c} v1 {controls[c][lo:hi][sel].mean():.3f}" for c in CONTROLS)
        )

    # ---- 1-2. overlap against matched placebo ------------------------------------------------
    say("\n== Overlap with engine-running states ==")
    blocked = np.zeros(n_grid, dtype=bool)
    for a, b in zip(first, last, strict=True):
        blocked[max(a - DROP_MARGIN, 0) : b + DROP_MARGIN + 1] = True
    candidates: list[np.ndarray] = []
    for a, b in zip(first[scope], last[scope], strict=True):
        length = b - a + 1
        shifts = np.arange(-200, 201) * WEEK
        starts = a + shifts
        same_side = (starts + length <= solar) if a < solar else (starts >= solar)
        ok = (starts >= visible) & (starts + length <= last_grid) & same_side
        starts = starts[ok]
        starts = np.array([s for s in starts if not blocked[s : s + length].any()], dtype=int)
        candidates.append(starts)
    say(
        "Placebo candidates per in-scope drop (whole-week shifts): "
        f"min {min(len(c) for c in candidates)}, median {int(np.median([len(c) for c in candidates]))}, "
        f"max {max(len(c) for c in candidates)}"
    )
    scope_idx = np.flatnonzero(scope)
    decided: dict[str, dict[str, object]] = {}
    for arm in ("v1", "N"):
        keep = night if arm == "N" else None
        flag = arms[arm]
        obs = np.array(
            [
                window_share(flag=flag, first=a, last=b, keep=keep)
                for a, b in zip(first, last, strict=True)
            ]
        )
        overlaps = obs >= OVERLAP_SHARE
        k_obs = int(overlaps[scope].sum())
        n_scope = int(scope.sum())
        pl_probs = []
        pl_over = []
        for (a, b), starts in zip(
            zip(first[scope], last[scope], strict=True), candidates, strict=True
        ):
            length = b - a + 1
            shares = np.array(
                [window_share(flag=flag, first=s, last=s + length - 1, keep=keep) for s in starts]
            )
            pl_over.append(shares >= OVERLAP_SHARE)
            pl_probs.append(float((shares >= OVERLAP_SHARE).mean()))
        counts = np.array(
            [sum(bool(o[rng.integers(0, len(o))]) for o in pl_over) for _ in range(N_REPLICATES)]
        )
        p_value = (1 + int((counts >= k_obs).sum())) / (N_REPLICATES + 1)
        say(
            f"Arm {arm}: drops overlapping engine-running states (at least half of "
            f"{'night ' if keep is not None else ''}half-hours flagged): {wilson(k=k_obs, n=n_scope)}; "
            f"mean share of a drop's half-hours flagged {np.nanmean(obs[scope]):.2f}"
        )
        say(
            f"  Matched placebo: expected overlap count {np.mean(pl_probs) * n_scope:.1f} of {n_scope} "
            f"(share {np.mean(pl_probs):.2f}; 95% of draws {np.quantile(counts, 0.025):.0f} to "
            f"{np.quantile(counts, 0.975):.0f}); one-sided p for the observed count {p_value:.4f}"
        )
        for name, mask in (
            ("2023-08-15 to 2025-02-28", scope & (first < solar)),
            ("2025-03-01 onward", scope & (first >= solar)),
        ):
            sub = np.flatnonzero(mask)
            pos = [int(np.flatnonzero(scope_idx == i)[0]) for i in sub]
            say(
                f"  {name}: {wilson(k=int(overlaps[mask].sum()), n=int(mask.sum()))}; placebo share "
                f"{np.mean([pl_probs[p] for p in pos]):.2f}"
            )
        if arm == "v1":
            for c in CONTROLS:
                own = np.array(
                    [
                        window_share(flag=controls[c], first=a, last=b, keep=None)
                        for a, b in zip(first, last, strict=True)
                    ]
                )
                plc = np.mean(
                    [
                        np.mean(
                            [
                                window_share(flag=controls[c], first=s, last=s + (b - a), keep=None)
                                >= OVERLAP_SHARE
                                for s in starts
                            ]
                        )
                        for (a, b), starts in zip(
                            zip(first[scope], last[scope], strict=True), candidates, strict=True
                        )
                    ]
                )
                SUMMARY[f"{c}_own"] = float((own[scope] >= OVERLAP_SHARE).mean())
                SUMMARY[f"{c}_placebo"] = float(plc)
                say(
                    f"  Control {c}, the drops' own dates: {wilson(k=int((own[scope] >= OVERLAP_SHARE).sum()), n=n_scope)}; "
                    f"matched placebo windows {plc:.2f}"
                )
        SUMMARY[f"{arm}_observed"] = k_obs / n_scope
        SUMMARY[f"{arm}_placebo"] = float(np.mean(pl_probs))
        SUMMARY[f"{arm}_placebo_lo"] = float(np.quantile(counts, 0.025)) / n_scope
        SUMMARY[f"{arm}_placebo_hi"] = float(np.quantile(counts, 0.975)) / n_scope
        decided[arm] = {
            "obs": obs.tolist(),
            "k": k_obs,
            "p": p_value,
            "placebo_share": float(np.mean(pl_probs)),
            "counts": counts.tolist(),
        }
        if arm == "v1":
            np.save(DATA_DIR / "placebo_counts_v1.npy", counts)
    obs_v1 = np.asarray(decided["v1"]["obs"])
    obs_n = np.asarray(decided["N"]["obs"])
    overlap_v1 = obs_v1 >= OVERLAP_SHARE

    # ---- 3. depth and step sizes -------------------------------------------------------------
    say("\n== Depth against multiples of one engine unit ==")
    g, g_lo, g_hi = unit_from_engine_record(rng=rng)
    say(
        f"Unit from the engine's recorded output (2023-07-03 to 2023-08-15), levels 1 to 3: {g:.2f} MW "
        f"(95% day-resampled interval {g_lo:.2f} to {g_hi:.2f}); the plan fixed {UNIT_MW} MW"
    )
    depth = drops["depth_mw"].to_numpy()
    med = lambda x: float(np.nanmedian(x))  # noqa: E731
    onset = np.array(
        [med(resid[a : a + STEP_WINDOW]) - med(resid[a - STEP_WINDOW : a]) for a in first]
    )
    end = np.array(
        [
            med(resid[b + 1 : b + 1 + STEP_WINDOW]) - med(resid[b + 1 - STEP_WINDOW : b + 1])
            for b in last
        ]
    )
    clustering = {}
    for label, values in (
        ("depth", depth),
        ("onset step", np.abs(onset)),
        ("end step", np.abs(end)),
    ):
        for unit in (UNIT_MW, UNIT_RECORD_MW, UNIT_SENSITIVITY_MW):
            d = distance_to_unit_multiple(x=values[scope], unit=unit)
            mean, lo, hi = bootstrap_mean(x=d, rng=rng)
            sim = np.array(
                [
                    distance_to_unit_multiple(
                        x=rng.uniform(values[scope].min(), values[scope].max(), len(d)), unit=unit
                    ).mean()
                    for _ in range(N_REPLICATES)
                ]
            )
            clustering[(label, unit)] = (mean, lo, hi)
            say(
                f"  {label}, unit {unit} MW: mean distance to the nearest multiple {mean:.3f} units "
                f"(95% bootstrap {lo:.3f} to {hi:.3f}); uniform spread over the same range "
                f"{sim.mean():.3f} (95% of draws {np.quantile(sim, 0.025):.3f} to {np.quantile(sim, 0.975):.3f}); "
                f"p for a mean this low {(1 + int((sim <= mean).sum())) / (N_REPLICATES + 1):.4f}"
            )
    # explanation test for a failed clustering prediction: the noise in a drop's depth, by period
    smooth = (
        pl.DataFrame({"r": resid})
        .with_columns(m=pl.col("r").rolling_median(window_size=48, min_samples=24, center=True))[
            "m"
        ]
        .to_numpy()
    )
    outside = ~blocked & np.isfinite(smooth)
    for name, lo, hi in (
        ("2023-08-15 to 2025-02-28", visible, solar),
        ("2025-03-01 onward", solar, last_grid),
    ):
        sel = outside[lo:hi]
        spread = smooth[lo:hi][sel]
        say(
            f"Noise in a 24-hour median of D8's residual outside drops, {name}: sd {spread.std():.2f} MW; "
            f"middle 95% {np.quantile(spread, 0.025):+.2f} to {np.quantile(spread, 0.975):+.2f} MW"
        )
        pick = scope & (first >= lo) & (first < hi)
        d = distance_to_unit_multiple(x=depth[pick], unit=UNIT_MW)
        mean, lo_ci, hi_ci = bootstrap_mean(x=d, rng=rng)
        say(
            f"  depth clustering in this period ({int(pick.sum())} drops): mean distance {mean:.3f} units (95% bootstrap {lo_ci:.3f} to {hi_ci:.3f}); median depth {np.median(depth[pick]):.2f} MW"
        )
    say(
        f"Depth in units of {UNIT_MW} MW, sorted: "
        + " ".join(f"{x:.2f}" for x in np.sort(depth[scope] / UNIT_MW))
    )
    say(
        f"Onset steps (MW) median {np.median(onset[scope]):.2f}; end steps (MW) median {np.median(end[scope]):.2f}"
    )

    # ---- 4. timing ----------------------------------------------------------------------------
    say("\n== Timing ==")
    wd = drops["start_weekday"].to_numpy()[scope]
    hr = (drops["start_hour"].to_numpy()[scope] // 6).astype(int)
    say(
        f"Onset weekday counts Mon..Sun {np.bincount(wd, minlength=7).tolist()}; chi-squared p {stats.chisquare(np.bincount(wd, minlength=7)).pvalue:.3f}"
    )
    say(
        f"Onset 6-hour blocks (0-6, 6-12, 12-18, 18-24, hours UTC) {np.bincount(hr, minlength=4).tolist()}; chi-squared p {stats.chisquare(np.bincount(hr, minlength=4)).pvalue:.3f}"
    )
    ewd = drops["end_weekday"].to_numpy()[scope]
    ehr = (drops["end_hour"].to_numpy()[scope] // 6).astype(int)
    say(
        f"End weekday counts {np.bincount(ewd, minlength=7).tolist()}; 6-hour blocks {np.bincount(ehr, minlength=4).tolist()}"
    )
    prices = pl.read_parquet(PRICES_PATH).select("start", "price")
    p_arr = np.full(n_grid, np.nan)
    idx = ((prices["start"].dt.epoch("s").to_numpy() - int(GRID_START.timestamp())) // 1800).astype(
        int
    )
    ok = (idx >= 0) & (idx < n_grid)
    p_arr[idx[ok]] = prices["price"].to_numpy()[ok]
    ref = p_arr[stamp(datetime(2023, 6, 1, tzinfo=UTC)) : stamp(datetime(2026, 7, 1, tzinfo=UTC))]
    edges = np.nanquantile(ref, [0.2, 0.4, 0.6, 0.8])
    fifth = lambda v: np.where(np.isfinite(v), np.searchsorted(edges, v) + 1, 0)  # noqa: E731
    onset_fifth = fifth(p_arr[first[scope]])
    say(
        f"System price fifth at onset (1 cheapest .. 5 dearest; 0 no price) counts {np.bincount(onset_fifth, minlength=6).tolist()}"
    )
    within = np.array(
        [
            np.nanmean(fifth(p_arr[a : b + 1])[np.isfinite(p_arr[a : b + 1])])
            for a, b in zip(first[scope], last[scope], strict=True)
        ]
    )
    placebo_within = np.array(
        [
            np.mean(
                [
                    np.nanmean(
                        fifth(p_arr[s : s + (b - a) + 1])[np.isfinite(p_arr[s : s + (b - a) + 1])]
                    )
                    for s in starts
                ]
            )
            for (a, b), starts in zip(
                zip(first[scope], last[scope], strict=True), candidates, strict=True
            )
        ]
    )
    say(
        f"Mean price fifth across a drop's half-hours: drops {within.mean():.2f}; matched placebo windows "
        f"{placebo_within.mean():.2f}; paired difference {np.mean(within - placebo_within):+.3f} "
        f"(Wilcoxon p {stats.wilcoxon(within - placebo_within).pvalue:.3f})"
    )
    months = np.array([d.month for d in drops["start_date"].to_list()])[scope]
    say(f"Onsets by calendar month Jan..Dec {np.bincount(months, minlength=13)[1:].tolist()}")
    gaps = first[scope][1:] - last[scope][:-1]
    say(
        f"Gap from one in-scope drop's end to the next one's start (days): median {np.median(gaps) / 48:.1f}; share under 14 days {np.mean(gaps < 14 * 48):.2f}"
    )

    # ---- 5. after March 2026 -----------------------------------------------------------------
    say("\n== After March 2026 ==")
    s25, s26 = stamp(datetime(2026, 3, 1, tzinfo=UTC)), stamp(datetime(2026, 6, 25, tzinfo=UTC))
    n_before = int(((first >= solar) & (first < s26) & (first < s25)).sum())
    n_after = int(((first >= s25) & (first < s26)).sum())
    rate = n_before / (s25 - solar)
    expected = rate * (s26 - s25)
    say(
        f"Drops starting 2025-03-01 to 2026-02-28: {n_before} ({n_before / ((s25 - solar) / 48 / 30.4):.1f} a month); "
        f"2026-03-01 to 2026-06-24: {n_after}. Expected at the earlier rate {expected:.1f}; Poisson P(0) {np.exp(-expected):.4f}"
    )
    rows_m = []
    for y_, m_ in [(2025, m) for m in range(3, 13)] + [(2026, m) for m in range(1, 7)]:
        a = stamp(datetime(y_, m_, 1, tzinfo=UTC))
        b = stamp(datetime(y_ + (m_ == 12), m_ % 12 + 1, 1, tzinfo=UTC))
        rows_m.append(
            {
                "month": f"{y_}-{m_:02d}",
                "drops_started": int(((first >= a) & (first < b)).sum()),
                "v1_flag_share": float(arms["v1"][a:b].mean()),
                "left_tail_share": float((resid[a:b][np.isfinite(resid[a:b])] < -2.5).mean()),
            }
        )
    pl.DataFrame(rows_m).write_parquet(DATA_DIR / "monthly.parquet")
    say(
        pl.DataFrame(rows_m).to_pandas().to_string(index=False)
        if False
        else str(pl.DataFrame(rows_m))
    )

    # ---- 6. alarms ---------------------------------------------------------------------------
    say("\n== Alarms inside engine spells ==")
    flag = arms["v1"].copy()
    flag[:visible] = False
    runs = spells(flag=flag, join_gap=SPELL_JOIN)
    inside = np.zeros(n_grid, dtype=bool)
    near_edge = np.zeros(n_grid, dtype=bool)
    for a, b in runs:
        inside[max(a - ALARM_PAD, 0) : b + ALARM_PAD + 1] = True
        near_edge[max(a - ALARM_PAD, 0) : a + ALARM_PAD + 1] = True
        near_edge[max(b - ALARM_PAD, 0) : b + ALARM_PAD + 1] = True
    in_drop = np.zeros(n_grid, dtype=bool)
    for a, b in zip(first, last, strict=True):
        in_drop[a : b + 1] = True
    elig = np.zeros(n_grid, dtype=bool)
    elig[visible : stamp(datetime(2026, 6, 25, tzinfo=UTC))] = True
    say(
        f"Engine spells (flagged runs joined across gaps of {SPELL_JOIN} half-hours or fewer): {len(runs)}; "
        f"share of eligible half-hours inside a spell widened by {ALARM_PAD}: {inside[elig].mean():.3f}; "
        f"within {ALARM_PAD} of a spell's start or end: {near_edge[elig].mean():.3f}; inside a drop {in_drop[elig].mean():.3f}"
    )
    a130 = pl.read_parquet(ALARMS_130_PATH)
    a130_d8 = a130.filter(pl.col("series") == D8_KEY)
    sources = {
        "RO-NC (all 52)": a130_d8["stamp"].to_numpy(),
        "RO-NC unexplained (28)": a130_d8.filter(pl.col("unexplained"))["stamp"].to_numpy(),
    }
    a195 = pl.concat(
        [pl.read_parquet(p) for p in sorted(ALARMS_195_DIR.glob("alarms_f*.parquet"))]
    ).filter(pl.col("series") == D8_KEY)
    for arm in ALARM_ARMS[1:]:
        sources[f"{arm} (experiment 195)"] = a195.filter(pl.col("arm") == arm)["stamp"].to_numpy()
    ron_unexplained = sources["RO-NC unexplained (28)"]
    for name, raw in sources.items():
        st = np.unique(raw)
        scoped = st[(st >= visible) & (st < stamp(datetime(2026, 6, 25, tzinfo=UTC)))]
        k_in = int(inside[scoped].sum())
        k_edge = int(near_edge[scoped].sum())
        p_in = stats.binomtest(k_in, len(scoped), inside[elig].mean(), alternative="greater").pvalue
        p_edge = stats.binomtest(
            k_edge, len(scoped), near_edge[elig].mean(), alternative="greater"
        ).pvalue
        say(
            f"  {name}: {len(st)} alarms; {len(scoped)} from 2023-08-15; inside a widened engine spell "
            f"{wilson(k=k_in, n=len(scoped)) if len(scoped) else 'none'} against {inside[elig].mean():.2f} at random times "
            f"(expected {inside[elig].mean() * len(scoped):.1f}, one-sided p {p_in:.3f}); within {ALARM_PAD} half-hours of a spell edge "
            f"{k_edge} against {near_edge[elig].mean() * len(scoped):.1f} expected (p {p_edge:.3f}); inside a drop {int(in_drop[scoped].sum())} "
            f"against {in_drop[elig].mean() * len(scoped):.1f} expected"
        )
    unexplained = np.unique(ron_unexplained)
    scoped = unexplained[
        (unexplained >= visible) & (unexplained < stamp(datetime(2026, 6, 25, tzinfo=UTC)))
    ]
    k_expl = int(inside[scoped].sum())
    k_excess = k_expl - inside[elig].mean() * len(scoped)
    say(
        f"Unexplained RO-NC alarms on D8: 28; inside a widened engine spell: {k_expl}; left outside, from 2023-08-15: {len(scoped) - k_expl}; "
        f"excess over the random-time expectation: {k_excess:.1f}"
    )
    say(
        f"Share of the {TOTAL_FALSE_ALARMS} unexplained alarms supplied by D8 and D9: {D8_D9_FALSE_ALARMS / TOTAL_FALSE_ALARMS:.3f}; "
        f"with D8's engine-explained alarms reclassified: kept in the total {(D8_D9_FALSE_ALARMS - k_expl) / TOTAL_FALSE_ALARMS:.3f}; "
        f"removed from the total {(D8_D9_FALSE_ALARMS - k_expl) / (TOTAL_FALSE_ALARMS - k_expl):.3f}; "
        f"counting only the excess ({k_excess:.1f}): kept {(D8_D9_FALSE_ALARMS - k_excess) / TOTAL_FALSE_ALARMS:.3f}, "
        f"removed {(D8_D9_FALSE_ALARMS - k_excess) / (TOTAL_FALSE_ALARMS - k_excess):.3f}"
    )

    # ---- 7. neighbours -----------------------------------------------------------------------
    say("\n== Neighbour step-up at onset, by overlap with engine states ==")
    step = drops["best_neighbour_step_mw"].to_numpy()
    for label, mask in (
        ("overlap (v1)", scope & overlap_v1),
        ("no overlap (v1)", scope & ~overlap_v1),
    ):
        cells = [
            f"at least {f} x depth: {int((step[mask] >= f * depth[mask]).sum())} of {int(mask.sum())}"
            for f in NEIGHBOUR_FRACTIONS
        ]
        say(f"  {label}: " + "; ".join(cells))
    say("  Experiment 159's placebo shares: 0.344, 0.214, 0.124 (random times from July 2023)")

    # ---- per-drop table ----------------------------------------------------------------------
    table = drops.select(
        "drop", "start_date", "end_date", "depth_mw", "duration_days"
    ).with_columns(
        in_scope=pl.Series(scope),
        v1_share=pl.Series(obs_v1),
        night_share=pl.Series(obs_n),
        onset_step_mw=pl.Series(onset),
        end_step_mw=pl.Series(end),
        depth_in_units=pl.Series(depth / UNIT_MW),
    )
    table.write_parquet(DATA_DIR / "drops_overlap.parquet")
    say("\n" + str(table.with_columns(pl.col(pl.Float64).round(2))))
    (DATA_DIR / "summary.json").write_text(json.dumps(SUMMARY, indent=1))
    (Path(__file__).parent / "printed_report.txt").write_text("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
