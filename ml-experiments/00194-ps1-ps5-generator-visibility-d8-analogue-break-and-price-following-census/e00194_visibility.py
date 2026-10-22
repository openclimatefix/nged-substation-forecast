# ruff: noqa: E501
"""Experiment 194, PS1 part C: does D8 see G2 while G2's readings arrive, and not after?

Three tests of the account that D8's reading is gross (G2's output added back) while G2's readings
arrive and net afterwards. (1) G2's reading availability by month. (2) D8's left-tail share by day
against G2's reading status (reading present, reading missing) around August 2023 and in 2026. (3)
D8's residual histogram before and after, for peaks at multiples of one engine unit. Prints to
`printed_visibility.txt`; writes the daily table for the charts.

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python e00194_visibility.py
"""

from datetime import UTC, datetime
from pathlib import Path

import e00194_common as common
import numpy as np
import polars as pl

REPORT: list[str] = []


def say(text: str = "") -> None:
    """Print a line and keep it for the printed report."""
    print(text, flush=True)
    REPORT.append(text)


def main() -> None:
    """Run the three tests."""
    panel = common.load_panel()
    d8 = common.residual_frame(panel=panel, label="D8").filter(
        pl.col("resid").is_not_null(), pl.col("resid").is_not_nan()
    )
    g2 = common.g2_readings()
    both = d8.join(g2.rename({"power": "g2"}), on="time", how="left").with_columns(
        tail=(pl.col("resid") < common.LEFT_TAIL_MW).cast(pl.Int64),
        g2_present=pl.col("g2").is_not_null(),
    )

    say(
        "(1) G2 reading availability: share of D8's half-hours with a G2 reading, and share of those above zero"
    )
    monthly = both.group_by_dynamic("time", every="1mo").agg(
        present=pl.col("g2_present").mean(),
        n_present=pl.col("g2_present").sum(),
        positive=(pl.col("g2") > 0).sum(),
        tail=pl.col("tail").mean(),
    )
    for r in monthly.filter(pl.col("time") >= datetime(2023, 4, 1, tzinfo=UTC)).iter_rows(
        named=True
    ):
        say(
            f"  {r['time']:%Y-%m}: reading present {r['present']:.3f} ({r['n_present']}), above zero {r['positive']}, D8 left-tail {r['tail']:.3f}"
        )

    say("\n(2) D8 left-tail share by G2 status")
    periods = {
        "2023-07-03 to 2023-08-15 (G2 output arrives)": (
            datetime(2023, 7, 3, tzinfo=UTC),
            datetime(2023, 8, 16, tzinfo=UTC),
        ),
        "2023-08-16 to 2023-08-24 (readings arrive, all zero)": (
            datetime(2023, 8, 16, tzinfo=UTC),
            datetime(2023, 8, 25, tzinfo=UTC),
        ),
        "2023-08-25 to 2023-12-31 (no readings)": (
            datetime(2023, 8, 25, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
        ),
        "2024-01-01 to 2026-01-26 (sparse readings)": (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 27, tzinfo=UTC),
        ),
        "2026-01-27 to 2026-06-30 (no readings)": (
            datetime(2026, 1, 27, tzinfo=UTC),
            datetime(2026, 7, 1, tzinfo=UTC),
        ),
    }
    for name, (a, b) in periods.items():
        sub = both.filter(pl.col("time").is_between(a, b, closed="left"))
        pres, absent = sub.filter(pl.col("g2_present")), sub.filter(~pl.col("g2_present"))
        text = f"  {name}: all {sub['tail'].mean():.3f} (n={sub.height})"
        if pres.height:
            lo, hi = common.wilson(k=int(pres["tail"].sum()), n=pres.height)
            text += f"; reading present {pres['tail'].mean():.3f} (n={pres.height}; {lo:.3f} to {hi:.3f})"
        if absent.height:
            lo, hi = common.wilson(k=int(absent["tail"].sum()), n=absent.height)
            text += f"; reading missing {absent['tail'].mean():.3f} (n={absent.height}; {lo:.3f} to {hi:.3f})"
        say(text)
    # Sparse-reading period: tail share when a reading is present, split by reading above zero.
    sparse = both.filter(
        pl.col("time").is_between(
            datetime(2023, 8, 25, tzinfo=UTC), datetime(2026, 1, 27, tzinfo=UTC), closed="left"
        ),
        pl.col("g2_present"),
    )
    say(
        f"  after 2023-08-24 with a reading present: {sparse.height} half-hours, readings above zero {int((sparse['g2'] > 0).sum())}, max {sparse['g2'].max()}"
    )
    days = (
        sparse.with_columns(day=pl.col("time").dt.date())
        .group_by("day")
        .agg(n=pl.len(), tail=pl.col("tail").mean())
        .sort("day")
    )
    say(
        f"  days with a reading after 2023-08-24: {days.height}; their median left-tail share {days['tail'].median():.3f}; readings per day median {days['n'].median():.0f}"
    )

    say(
        "\nDaily D8 left-tail share around the break (2023-08-01 to 2023-09-10) with G2 reading status"
    )
    daily = (
        both.filter(
            pl.col("time").is_between(
                datetime(2023, 8, 1, tzinfo=UTC), datetime(2023, 9, 11, tzinfo=UTC), closed="left"
            )
        )
        .with_columns(day=pl.col("time").dt.date())
        .group_by("day")
        .agg(tail=pl.col("tail").mean(), g2_n=pl.col("g2_present").sum(), g2_max=pl.col("g2").max())
        .sort("day")
    )
    say(
        "  "
        + " ".join(
            f"{r['day']:%m-%d}:{r['tail']:.2f}/{r['g2_n']}r/{(r['g2_max'] if r['g2_max'] is not None else float('nan')):.1f}"
            for r in daily.iter_rows(named=True)
        )
    )
    say("  (day: left-tail share / number of G2 readings that day / max G2 reading MW)")

    say(
        "\nCounterfactual: if D8 had been net during G2's output weeks, how often would the residual fall below -2.5 MW?"
    )
    win = both.filter(
        pl.col("time").is_between(common.G2_FIRST, common.G2_LAST, closed="left"),
        pl.col("g2").is_not_null(),
    )
    net = win["resid"] - win["g2"]
    say(
        f"  observed share {(win['resid'] < common.LEFT_TAIL_MW).mean():.3f}; share if G2's output were subtracted {(net < common.LEFT_TAIL_MW).mean():.3f} (n={win.height})"
    )
    slope = np.polyfit(win["g2"].to_numpy(), win["resid"].to_numpy(), 1)[0]
    say(
        f"  slope of D8's residual on G2's output in those weeks: {slope:+.3f} MW per MW (net would be near -1; gross near 0)"
    )

    say("\n(3) D8 residual histogram, 1 MW bins from -8 to +4 MW, share of half-hours")
    bins = np.arange(-8, 4.01, 1.0)
    for name, a, b in (
        (
            "before the engine, 2022-01-01 to 2023-06-12",
            datetime(2022, 1, 1, tzinfo=UTC),
            datetime(2023, 6, 13, tzinfo=UTC),
        ),
        ("G2 output weeks", common.G2_FIRST, common.G2_LAST),
        (
            "2023-09-01 to 2025-12-31",
            datetime(2023, 9, 1, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
        ),
        (
            "2026-04-01 to 2026-06-30",
            datetime(2026, 4, 1, tzinfo=UTC),
            datetime(2026, 7, 1, tzinfo=UTC),
        ),
    ):
        arr = d8.filter(pl.col("time").is_between(a, b, closed="left"))["resid"].to_numpy()
        hist, _ = np.histogram(arr, bins=bins)
        say(
            f"  {name} (n={arr.size}): "
            + " ".join(f"{bins[i]:+.0f}:{hist[i] / arr.size:.3f}" for i in range(hist.size))
        )
    post = d8.filter(
        pl.col("time").is_between(
            datetime(2023, 9, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC), closed="left"
        )
    )["resid"].to_numpy()
    hist, edges = np.histogram(post, bins=np.arange(-8, 2.01, 0.25))
    smooth = np.convolve(hist, np.ones(3) / 3, mode="same")
    peaks = [
        float(edges[i] + 0.125)
        for i in range(1, hist.size - 1)
        if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > 0.01 * post.size
    ]
    say(f"  local peaks of the 2023-09 to 2025-12 histogram (0.25 MW bins, 3-bin smooth): {peaks}")
    d8.write_parquet(common.DATA_DIR / "d8_resid.parquet")
    both.select("time", "resid", "g2", "tail").write_parquet(common.DATA_DIR / "d8_vs_g2.parquet")
    Path(__file__).with_name("printed_visibility.txt").write_text("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
