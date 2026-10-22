# ruff: noqa: E501
"""Experiment 194, PS1 part A: G2's recorded dispatch and D8's residual while G2's readings arrived.

Reproduces the design-time numbers on G2: how many half-hours join to the system price, the units
running, the running share by price fifth, hour of day and weekday, the spells of running, and D8's
residual while G2 ran. Prints every number the write-up quotes to `printed_g2.txt`. G2 is never
named.

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python e00194_g2.py
"""

from pathlib import Path

import e00194_common as common
import numpy as np
import polars as pl

REPORT: list[str] = []


def say(text: str = "") -> None:
    """Print a line and keep it for the printed report."""
    print(text, flush=True)
    REPORT.append(text)


def spells(*, running: np.ndarray) -> list[int]:
    """Return the lengths, in half-hours, of the runs of consecutive running half-hours."""
    out, run = [], 0
    for r in running:
        if r:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def main() -> None:  # noqa: PLR0915
    """Print G2's dispatch and D8's residual by units running."""
    g2 = common.g2_readings()
    say(
        f"G2 readings in the power table: {g2.height}, first {g2['time'].min()}, last {g2['time'].max()}"
    )
    positive = g2.filter(pl.col("power") > 0)
    say(
        f"  above zero: {positive.height}; first {positive['time'].min()}, last {positive['time'].max()}"
    )
    by_month = positive.group_by_dynamic("time", every="1mo").agg(
        n=pl.len(), max_mw=pl.col("power").max()
    )
    for r in by_month.iter_rows(named=True):
        say(f"  {r['time']:%Y-%m}: {r['n']} positive readings, max {r['max_mw']:.2f} MW")
    levels = (
        positive.with_columns(level=pl.col("power").round(2))
        .group_by("level")
        .len()
        .sort("len", descending=True)
    )
    say(f"  most common positive levels (MW, count): {levels.head(8).rows()}")

    window = g2.filter(pl.col("time").is_between(common.G2_FIRST, common.G2_LAST, closed="left"))
    prices = common.system_prices()
    joined = window.join(prices, left_on="time", right_on="start", how="inner")
    say(
        f"\nWindow {common.G2_FIRST:%Y-%m-%d} to {common.G2_LAST:%Y-%m-%d}: readings {window.height}"
    )
    say(f"  half-hours joined to the system price: {joined.height} (design: 2,054)")
    joined = joined.with_columns(units=(pl.col("power") / common.G2_UNIT_MW).round().cast(pl.Int64))
    counts = dict(sorted(joined["units"].value_counts().iter_rows()))
    say(f"  units running (design 0:1704, 1:137, 2:133, 3:80): {counts}")
    running = joined["units"] >= 1
    k, n = int(running.sum()), joined.height
    low, high = common.wilson(k=k, n=n)
    say(f"  running share: {k / n:.3f} (95% Wilson {low:.3f} to {high:.3f}); design 0.17")
    # Fifths of the system price, cheapest to dearest.
    edges = np.quantile(joined["price"].to_numpy(), [0.2, 0.4, 0.6, 0.8])
    fifth = np.searchsorted(edges, joined["price"].to_numpy(), side="right")
    shares = []
    for f in range(5):
        sel = fifth == f
        kk, nn = int(running.to_numpy()[sel].sum()), int(sel.sum())
        lo, hi = common.wilson(k=kk, n=nn)
        shares.append(kk / nn)
        say(f"  price fifth {f + 1}: running {kk}/{nn} = {kk / nn:.3f} ({lo:.3f} to {hi:.3f})")
    say(f"  design: 0.090, 0.151, 0.152, 0.191, 0.271; ours: {[round(s, 3) for s in shares]}")
    say(f"  price fifth edges (GBP/MWh): {[round(float(e), 1) for e in edges]}")
    # Spearman between price and running.
    ranks_p = pl.Series(joined["price"]).rank().to_numpy()
    ranks_r = pl.Series(running.cast(pl.Float64)).rank().to_numpy()
    say(f"  Spearman of price with running: {np.corrcoef(ranks_p, ranks_r)[0, 1]:.3f}")

    hours = joined["time"].dt.hour().to_numpy()
    hour_share = {h: float(running.to_numpy()[hours == h].mean()) for h in range(24)}
    say(
        "  running share by UTC hour: "
        + " ".join(f"{h:02d}:{hour_share[h]:.2f}" for h in range(24))
    )
    low_hours = [hour_share[h] for h in range(6)]
    high_hours = [hour_share[h] for h in (19, 20, 21)]
    say(
        f"  00-05 UTC range {min(low_hours):.2f} to {max(low_hours):.2f}; 19-21 UTC range {min(high_hours):.2f} to {max(high_hours):.2f} (design 0.02-0.10 and 0.25-0.28)"
    )
    weekday = joined["time"].dt.weekday().to_numpy()
    say(
        "  running share by weekday (Mon=1): "
        + " ".join(f"{d}:{float(running.to_numpy()[weekday == d].mean()):.2f}" for d in range(1, 8))
    )

    full = (
        pl.DataFrame(
            {
                "time": pl.datetime_range(
                    common.G2_FIRST,
                    common.G2_LAST - common.HALF_HOUR,
                    interval="30m",
                    time_zone="UTC",
                    eager=True,
                )
            }
        )
        .join(window, on="time", how="left")
        .sort("time")
    )
    present = full["power"].is_not_null().to_numpy()
    run_flag = full["power"].fill_null(0.0).to_numpy() >= 0.5 * common.G2_UNIT_MW
    lens = spells(running=run_flag)
    say(
        f"  half-hours on the grid in the window {full.height}, with a reading {int(present.sum())}"
    )
    say(
        f"  spells of running: {len(lens)} (design 55); length min {min(lens)}, max {max(lens)}, median {np.median(lens):.0f} half-hours (design 1 to 29, median 6)"
    )
    # Spells counting only consecutive readings (missing readings break a spell either way).

    panel = common.load_panel()
    d8 = common.residual_frame(panel=panel, label="D8")
    july = d8.filter(pl.col("time").dt.year() == 2023, pl.col("time").dt.month() == 7).join(
        joined.select("time", "units"), on="time", how="inner"
    )
    say(
        "\nD8 residual (metered minus expected) in July 2023 by G2 units running (design +0.49, +0.53, +0.23 at 1, 2, 3):"
    )
    for u in (0, 1, 2, 3):
        sub = july.filter(pl.col("units") == u)["resid"].drop_nulls()
        if sub.len():
            rng = np.random.default_rng(0)
            arr = sub.to_numpy()
            boots = [np.median(rng.choice(arr, arr.size)) for _ in range(500)]
            say(
                f"  {u} units: n={arr.size}, median {np.median(arr):+.2f} MW (bootstrap 95% {np.quantile(boots, 0.025):+.2f} to {np.quantile(boots, 0.975):+.2f}), mean {arr.mean():+.2f}"
            )
    both = d8.filter(
        pl.col("time").is_between(common.G2_FIRST, common.G2_LAST, closed="left")
    ).join(joined.select("time", "units"), on="time", how="inner")
    say("D8 residual over the whole window (2023-07-03 to 2023-08-15) by units:")
    for u in (0, 1, 2, 3):
        sub = both.filter(pl.col("units") == u)["resid"].drop_nulls().to_numpy()
        say(
            f"  {u} units: n={sub.size}, median {np.median(sub):+.2f} MW, share below {common.LEFT_TAIL_MW} MW {float((sub < common.LEFT_TAIL_MW).mean()):.3f}"
        )
    Path(__file__).with_name("printed_g2.txt").write_text("\n".join(REPORT) + "\n")
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    joined.write_parquet(common.DATA_DIR / "g2_window.parquet")


if __name__ == "__main__":
    main()
