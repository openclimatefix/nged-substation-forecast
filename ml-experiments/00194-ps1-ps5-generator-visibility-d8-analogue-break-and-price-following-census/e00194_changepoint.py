# ruff: noqa: E501
"""Experiment 194, PS1 part B: the left-tail share of D8's residual and its change points.

The left-tail share is the share of half-hours in which the metered power minus the expected power
is below -2.5 MW. This script prints it by month and by quarter for D8, compares springs, checks
the months of experiment 191's scoring window, and fits a change point to the weekly share of D8
and of the controls D3, D5 and D14. It writes the weekly shares for the charts.

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python e00194_changepoint.py
"""

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import e00194_common as common
import numpy as np
import polars as pl

START: Final[datetime] = datetime(2023, 5, 1, tzinfo=UTC)
END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
PLATEAU_END: Final[datetime] = datetime(2026, 1, 1, tzinfo=UTC)
"""The upward change is fitted to weeks before this date, so the 2026 fall does not pull it."""
MIN_WEEK_HALF_HOURS: Final[int] = 168
"""A week needs at least half of its 336 half-hours to count."""
BLOCK_WEEKS: Final[int] = 4
N_BOOT: Final[int] = 1000
STEP_THRESHOLD: Final[float] = 0.10
REPORT: list[str] = []


def say(text: str = "") -> None:
    """Print a line and keep it for the printed report."""
    print(text, flush=True)
    REPORT.append(text)


def tail_frame(*, frame: pl.DataFrame, threshold_mw: float) -> pl.DataFrame:
    """Return the residual rows in the study window with a left-tail flag."""
    return frame.filter(
        pl.col("time").is_between(START, END, closed="left"),
        pl.col("resid").is_not_null(),
        pl.col("resid").is_not_nan(),
    ).with_columns(tail=(pl.col("resid") < threshold_mw).cast(pl.Int64))


def share_by(*, frame: pl.DataFrame, every: str) -> pl.DataFrame:
    """Return k, n and the share per period."""
    return (
        frame.sort("time")
        .group_by_dynamic("time", every=every)
        .agg(k=pl.col("tail").sum(), n=pl.len())
        .with_columns(share=pl.col("k") / pl.col("n"))
    )


def binom_ll(*, k: np.ndarray, n: np.ndarray) -> float:
    """Return the binomial log-likelihood of a constant share (without the combinatorial term)."""
    p = k.sum() / n.sum()
    if p <= 0 or p >= 1:
        return 0.0
    return float(k.sum() * np.log(p) + (n.sum() - k.sum()) * np.log(1 - p))


def best_step(*, k: np.ndarray, n: np.ndarray, min_side: int = 8) -> tuple[int, np.ndarray]:
    """Return the index of the first week of the second regime and the log-likelihood of every split."""
    ll = np.full(k.size, -np.inf)
    for t in range(min_side, k.size - min_side + 1):
        ll[t] = binom_ll(k=k[:t], n=n[:t]) + binom_ll(k=k[t:], n=n[t:])
    return int(np.argmax(ll)), ll


def block_bootstrap_cp(
    *, k: np.ndarray, n: np.ndarray, t_hat: int, rng: np.random.Generator
) -> np.ndarray:
    """Return bootstrap change points, resampling weeks in blocks within each regime."""
    out = np.empty(N_BOOT, dtype=int)
    share = k / n
    for b in range(N_BOOT):
        pieces = []
        for lo, hi in ((0, t_hat), (t_hat, k.size)):
            idx = np.arange(lo, hi)
            m = hi - lo
            starts = rng.integers(0, max(m - BLOCK_WEEKS + 1, 1), size=m // BLOCK_WEEKS + 1)
            drawn = np.concatenate([idx[s : s + BLOCK_WEEKS] for s in starts])[:m]
            pieces.append(drawn)
        order = np.concatenate(pieces)
        kb = np.rint(share[order] * n[order]).astype(float)
        out[b], _ = best_step(k=kb, n=n[order].astype(float))
    return out


def fit_one(*, label: str, weekly: pl.DataFrame, rng: np.random.Generator) -> dict:
    """Fit one step to the weekly shares before the plateau end and print the result."""
    w = weekly.filter(pl.col("n") >= MIN_WEEK_HALF_HOURS, pl.col("time") < PLATEAU_END).sort("time")
    k, n = w["k"].to_numpy().astype(float), w["n"].to_numpy().astype(float)
    t, ll = best_step(k=k, n=n)
    dates = w["time"].to_list()
    pre, post = k[:t].sum() / n[:t].sum(), k[t:].sum() / n[t:].sum()
    # Likelihood-ratio set: splits within 1.92 of the maximum.
    inside = np.where(ll >= ll[t] - 1.92)[0]
    boots = block_bootstrap_cp(k=k, n=n, t_hat=t, rng=rng)
    lo_b, hi_b = np.quantile(boots, [0.025, 0.975]).astype(int)
    result = {
        "label": label,
        "change": dates[t].date(),
        "pre": pre,
        "post": post,
        "step": post - pre,
        "lr_low": dates[int(inside.min())].date(),
        "lr_high": dates[int(inside.max())].date(),
        "boot_low": dates[lo_b].date(),
        "boot_high": dates[hi_b].date(),
        "weeks": int(k.size),
    }
    say(
        f"  {label}: change point (first week of new regime) {result['change']}; share {pre:.3f} before, {post:.3f} after (step {post - pre:+.3f}); "
        f"likelihood-ratio set {result['lr_low']} to {result['lr_high']}; block-bootstrap 95% {result['boot_low']} to {result['boot_high']}; {k.size} weeks"
    )
    return result


def main() -> None:  # noqa: PLR0915
    """Print the left-tail tables and the change-point fits."""
    panel = common.load_panel()
    rng = np.random.default_rng(0)
    d8 = common.residual_frame(panel=panel, label="D8")
    d8t = tail_frame(frame=d8, threshold_mw=common.LEFT_TAIL_MW)

    say("D8 left-tail share (residual below -2.5 MW) by month; n = half-hours with a residual")
    monthly = share_by(frame=d8t, every="1mo")
    d8_rows = " ".join(f"{r['time']:%Y-%m}:{r['share']:.3f}" for r in monthly.iter_rows(named=True))
    say("  " + d8_rows)
    say(
        "  design: May 2023 0.000; Jun 0.035; Jul 0.000; Aug 0.112; Sep 0.353; Oct 0.353; Nov 2025 0.147; Jan 2026 0.292; Feb 0.174; Mar 0.070; Apr 0.008; May 0.022; Jun 0.039"
    )
    low_n = monthly.filter(pl.col("n") < 0.8 * 48 * 28)
    say(
        f"  months with under 80% of half-hours present: {[f'{r["time"]:%Y-%m}:{r["n"]}' for r in low_n.iter_rows(named=True)]}"
    )
    monthly.write_parquet(common.DATA_DIR / "d8_monthly.parquet")

    # Wilson intervals for the months the design quotes.
    for label, y, m in (
        ("July 2023", 2023, 7),
        ("Sep 2023", 2023, 9),
        ("Jan 2026", 2026, 1),
        ("Apr 2026", 2026, 4),
    ):
        r = monthly.filter(pl.col("time").dt.year() == y, pl.col("time").dt.month() == m).row(
            0, named=True
        )
        lo, hi = common.wilson(k=r["k"], n=r["n"])
        say(f"  {label}: {r['k']}/{r['n']} = {r['share']:.3f} (Wilson {lo:.3f} to {hi:.3f})")

    say(
        "\nD8 left-tail share by quarter (design: 0.197 to 0.343 for every quarter 2023Q3 to 2025Q4)"
    )
    quarterly = share_by(frame=d8t, every="3mo")
    for r in quarterly.iter_rows(named=True):
        say(f"  {r['time']:%Y-%m}: {r['share']:.3f} (n={r['n']})")
    say("  calendar-quarter note: 3-month bins start in May, so also by calendar quarter below")
    cal = (
        d8t.with_columns(
            q=pl.col("time").dt.year().cast(pl.String)
            + "Q"
            + pl.col("time").dt.quarter().cast(pl.String)
        )
        .group_by("q")
        .agg(k=pl.col("tail").sum(), n=pl.len())
        .with_columns(share=pl.col("k") / pl.col("n"))
        .sort("q")
    )
    say(
        "  calendar quarters: "
        + " ".join(f"{r['q']}:{r['share']:.3f}" for r in cal.iter_rows(named=True))
    )
    cal_in = cal.filter(pl.col("q") >= "2023Q3", pl.col("q") <= "2025Q4")
    say(f"  2023Q3 to 2025Q4: min {cal_in['share'].min():.3f}, max {cal_in['share'].max():.3f}")
    months_in = monthly.filter(
        pl.col("time") >= datetime(2023, 9, 1, tzinfo=UTC),
        pl.col("time") < datetime(2026, 2, 1, tzinfo=UTC),
    )
    say(
        f"  every month Sep 2023 to Jan 2026 at least 0.10? min {months_in['share'].min():.3f} (month {months_in.sort('share').row(0, named=True)['time']:%Y-%m}); months below 0.10: {[f'{r["time"]:%Y-%m}:{r["share"]:.3f}' for r in months_in.filter(pl.col('share') < 0.10).iter_rows(named=True)]}"
    )
    jul = monthly.filter(pl.col("time").dt.year() == 2023, pl.col("time").dt.month() == 7).row(
        0, named=True
    )
    say(f"  July 2023 at most 0.02? {jul['share']:.3f}")

    say("\nSprings (March to May) and April, D8")
    for y in (2024, 2025, 2026):
        sub = d8t.filter(pl.col("time").dt.year() == y, pl.col("time").dt.month().is_in([3, 4, 5]))
        apr = d8t.filter(pl.col("time").dt.year() == y, pl.col("time").dt.month() == 4)
        say(
            f"  {y}: March-May {sub['tail'].mean():.3f} (n={sub.height}); April {apr['tail'].mean():.3f}"
        )

    say(
        "\nExperiment 191's scoring window (D8 data from 2025-04-01; scored runs 2025-07-01 to 2026-06-30)"
    )
    win = monthly.filter(pl.col("time") >= datetime(2025, 4, 1, tzinfo=UTC))
    say(
        "  D8 monthly left-tail share: "
        + " ".join(f"{r['time']:%Y-%m}:{r['share']:.3f}" for r in win.iter_rows(named=True))
    )
    scored = d8t.filter(pl.col("time") >= datetime(2025, 7, 1, tzinfo=UTC))
    before_feb = scored.filter(pl.col("time") < datetime(2026, 2, 1, tzinfo=UTC))
    after_feb = scored.filter(pl.col("time") >= datetime(2026, 2, 1, tzinfo=UTC))
    for tag, part in (
        ("scored year 2025-07 to 2026-06", scored),
        ("2025-07 to 2026-01", before_feb),
        ("2026-02 to 2026-06", after_feb),
    ):
        say(
            f"  {tag}: share {part['tail'].mean():.3f} over {part.height} half-hours ({part.height / scored.height:.2f} of the scored year)"
        )

    say(
        "\nWeekly step fits (weeks with at least 168 half-hours, before 2026-01-01; threshold -2.5 MW)"
    )
    results = []
    weekly_all = {}
    for label in ("D8", *common.CONTROL_LABELS):
        frame = d8 if label == "D8" else common.residual_frame(panel=panel, label=label)
        wk = share_by(frame=tail_frame(frame=frame, threshold_mw=common.LEFT_TAIL_MW), every="1w")
        weekly_all[label] = wk
        results.append(fit_one(label=label, weekly=wk, rng=rng))
        wk.write_parquet(common.DATA_DIR / f"weekly_{label}.parquet")
    d8_cp = results[0]["change"]
    for r in results[1:]:
        gap = abs((r["change"] - d8_cp).days)
        say(
            f"  control {r['label']}: change point {gap} days from D8's; step {r['step']:+.3f}; a change point of at least {STEP_THRESHOLD} in share: {r['step'] >= STEP_THRESHOLD}"
        )

    say(
        "\nSensitivity: capacity-scaled threshold (0.241 x the site's capacity, i.e. -2.5 MW at D8)"
    )
    for label in ("D8", *common.CONTROL_LABELS):
        cap = panel[common.key_of(panel=panel, label=label)]["cap"]
        thr = -0.241 * cap
        frame = d8 if label == "D8" else common.residual_frame(panel=panel, label=label)
        wk = share_by(frame=tail_frame(frame=frame, threshold_mw=thr), every="1w")
        say(
            f"  {label}: capacity {cap:.2f} MW, threshold {thr:.2f} MW, overall share {float(wk['k'].sum()) / float(wk['n'].sum()):.3f}"
        )
        fit_one(label=f"{label} (scaled)", weekly=wk, rng=rng)

    say(
        "\nTwo-step fit of D8 over the whole window (up then down), weeks with at least 168 half-hours"
    )
    w = weekly_all["D8"].filter(pl.col("n") >= MIN_WEEK_HALF_HOURS).sort("time")
    k, n = w["k"].to_numpy().astype(float), w["n"].to_numpy().astype(float)
    dates = w["time"].to_list()
    best = (-np.inf, 0, 0)
    for t1 in range(8, k.size - 16):
        ll1 = binom_ll(k=k[:t1], n=n[:t1])
        for t2 in range(t1 + 8, k.size - 8):
            ll = ll1 + binom_ll(k=k[t1:t2], n=n[t1:t2]) + binom_ll(k=k[t2:], n=n[t2:])
            if ll > best[0]:
                best = (ll, t1, t2)
    _, t1, t2 = best
    say(
        f"  up at {dates[t1].date()} (share {k[:t1].sum() / n[:t1].sum():.3f} -> {k[t1:t2].sum() / n[t1:t2].sum():.3f}); down at {dates[t2].date()} (-> {k[t2:].sum() / n[t2:].sum():.3f})"
    )

    Path(__file__).with_name("printed_changepoint.txt").write_text("\n".join(REPORT) + "\n")
    _ = date


if __name__ == "__main__":
    main()
