# ruff: noqa: E501
"""Stage 2 report: the mask forecast driven by each alarm, scored on the injected worlds.

```bash
uv run --no-sync python e00156_consumer_report.py | tee <private folder>/printed_consumer.txt
```
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

import numpy as np
import polars as pl
from e00156_common import WORK_DIR

SCORES = WORK_DIR / "scores"
DRAWS: Final[int] = 20_000
ARMS: Final[tuple[str, ...]] = ("K0", "CLEAN", "GAM_ONLY", "MASK_RONC", "MASK_NEW", "MASK_UNION")
QUANTILES: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}


def load() -> pl.DataFrame:
    """Long score table with the calendar month of each row's target time."""
    frame = pl.concat([pl.read_parquet(p) for p in sorted(SCORES.glob("*.parquet"))])
    return frame.with_columns(
        month=(
            pl.lit(datetime(2024, 4, 1, 9))  # noqa: DTZ001
            + pl.duration(days=pl.col("run"))
            + pl.duration(minutes=(pl.col("lead_hours") * 60).cast(pl.Int64))
        ).dt.strftime("%Y-%m")
    )


def wide(frame: pl.DataFrame, *, value: str = "qs") -> pl.DataFrame:
    """One row per forecast row, one column per arm."""
    keys = ["site", "world", "period", "row"]
    meta = frame.filter(pl.col("arm") == "K0").select(
        *keys,
        "month",
        "event",
        "inside",
        "known_ronc",
        "known_new",
        "known_union",
        "cutoff_minus_onset",
        "event_hours",
    )
    arms = frame.pivot(on="arm", index=keys, values=value)
    return meta.join(arms, on=keys)


def interval(
    *, months: np.ndarray, num: np.ndarray, den: np.ndarray | None, seed: int = 156
) -> tuple[float, float, float, float, float]:
    """Point and 95% and 99% intervals of sum(num)/sum(den) (or the mean), resampling months."""
    labels, index = np.unique(months, return_inverse=True)
    sn = np.bincount(index, weights=num, minlength=len(labels))
    sd = (
        np.bincount(index, minlength=len(labels)).astype(float)
        if den is None
        else np.bincount(index, weights=den, minlength=len(labels))
    )
    point = sn.sum() / sd.sum()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(labels), size=(DRAWS, len(labels)))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = sn[picks].sum(axis=1) / sd[picks].sum(axis=1)
    lo95, hi95 = np.nanpercentile(draws, [2.5, 97.5])
    lo99, hi99 = np.nanpercentile(draws, [0.5, 99.5])
    return point, lo95, hi95, lo99, hi99


def show(label: str, r: tuple, *, percent: bool = False) -> None:
    """Print a point and its intervals."""
    scale = 100.0 if percent else 1.0
    unit = "%" if percent else ""
    print(
        f"  {label}: {r[0] * scale:+.3f}{unit} (95% {r[1] * scale:+.3f} to {r[2] * scale:+.3f}; "
        f"99% {r[3] * scale:+.3f} to {r[4] * scale:+.3f})"
    )


def contrasts(table: pl.DataFrame, *, name: str) -> None:
    """Print mean scores, differences, and shares of the gap for one row set."""
    print(f"\n== {name}: {table.height} rows, {table['month'].n_unique()} months ==")
    print("  mean quantile score:", {a: round(float(table[a].mean()), 3) for a in ARMS})  # ty: ignore[invalid-argument-type]
    months = table["month"].to_numpy()
    for a, b in (
        ("MASK_UNION", "MASK_RONC"),
        ("MASK_NEW", "MASK_RONC"),
        ("MASK_NEW", "GAM_ONLY"),
        ("MASK_UNION", "GAM_ONLY"),
        ("MASK_RONC", "GAM_ONLY"),
        ("MASK_UNION", "K0"),
        ("MASK_RONC", "K0"),
        ("MASK_NEW", "K0"),
    ):
        show(
            f"{a} minus {b}",
            interval(months=months, num=(table[a] - table[b]).to_numpy(), den=None),
        )
    gap = (table["K0"] - table["CLEAN"]).to_numpy()
    for a in ("MASK_RONC", "MASK_NEW", "MASK_UNION", "GAM_ONLY"):
        show(
            f"share of K0-to-CLEAN gap closed by {a}",
            interval(months=months, num=(table["K0"] - table[a]).to_numpy(), den=gap),
            percent=True,
        )


def per_transfer(table: pl.DataFrame) -> pl.DataFrame:
    """Mean score per transfer (a world, site, and event), dated by its first row's month."""
    return table.group_by(["site", "world", "event"]).agg(
        *[pl.col(a).mean() for a in ARMS], month=pl.col("month").min()
    )


def tails(frame: pl.DataFrame, *, rows: pl.DataFrame) -> None:
    """Coverage and pinball loss at the four tail quantiles for every arm."""
    keys = rows.select("site", "world", "period", "row")
    sub = frame.join(keys, on=["site", "world", "period", "row"])
    for arm in ARMS:
        part = sub.filter(pl.col("arm") == arm)
        out = []
        for column, tau in QUANTILES.items():
            err = (part["actual"] - part[column]).to_numpy()
            loss = np.maximum(tau * err, (tau - 1) * err).mean()
            out.append(f"{column} {float((err <= 0).mean()):.3f} ({loss:.3f})")
        print(f"  {arm:10s}", "; ".join(out))


def posthoc(table: pl.DataFrame) -> None:
    """Post hoc arms: the pair alarm with RO-NC filling in, and the pair alarm at the other windows."""
    extra = pl.concat(
        [pl.read_parquet(p) for p in sorted((WORK_DIR / "scores_posthoc").glob("*.parquet"))]
    )
    keys = ["site", "world", "period", "row"]
    both = table.join(
        extra.pivot(on="arm", index=keys, values="qs").select(
            *keys,
            "MASK_NEW_ALARM",
            "MASK_UNION2",
            "MASK_UNION2_MATCHED",
            "MASK_NEW_W4",
            "MASK_NEW_W12",
        ),
        on=keys,
    )
    months = both["month"].to_numpy()
    print(f"\n== post hoc arms, all rows after the onset: {both.height} rows ==")
    print(
        "  mean quantile score:",
        {
            a: round(float(both[a].mean()), 3)  # ty: ignore[invalid-argument-type]
            for a in (
                "MASK_NEW_ALARM",
                "MASK_UNION2",
                "MASK_UNION2_MATCHED",
                "MASK_NEW_W4",
                "MASK_NEW_W12",
            )
        },
    )
    gap = (both["K0"] - both["CLEAN"]).to_numpy()
    for a in (
        "MASK_NEW_ALARM",
        "MASK_UNION2",
        "MASK_UNION2_MATCHED",
        "MASK_NEW_W4",
        "MASK_NEW_W12",
    ):
        show(
            f"{a} minus MASK_RONC",
            interval(months=months, num=(both[a] - both["MASK_RONC"]).to_numpy(), den=None),
        )
        show(
            f"{a} minus K0",
            interval(months=months, num=(both[a] - both["K0"]).to_numpy(), den=None),
        )
        show(
            f"share of gap closed by {a}",
            interval(months=months, num=(both["K0"] - both[a]).to_numpy(), den=gap),
            percent=True,
        )


def main() -> None:
    """Print stage 2."""
    frame = load()
    table = wide(frame)
    err_mean = wide(frame, value="err_mean")
    err_median = wide(frame, value="err_median")
    print(
        f"rows {table.height}; worlds {table['world'].n_unique()}; sites {table['site'].n_unique()}"
    )
    contrasts(table, name="C4: all rows with origin after the onset")
    contrasts(
        table.filter(pl.col("known_ronc") > 0),
        name="origin before the RO-NC alarm is known (incl. missed)",
    )
    contrasts(table.filter(pl.col("known_ronc") <= 0), name="origin after the RO-NC alarm is known")
    contrasts(
        table.filter(pl.col("known_new") > 0),
        name="origin before the pair alarm is known (incl. missed)",
    )
    contrasts(table.filter(pl.col("known_new") <= 0), name="origin after the pair alarm is known")
    contrasts(
        table.filter(pl.col("known_new").is_infinite()), name="transfers the pair alarm misses"
    )
    contrasts(table.filter(pl.col("known_new").is_finite()), name="transfers the pair alarm finds")
    contrasts(table.filter(pl.col("inside")), name="targets inside the transfer")
    contrasts(table.filter(~pl.col("inside")), name="targets in the 72 h after the transfer")
    contrasts(table.filter(pl.col("site") != "D11"), name="all rows without D11")
    for site in sorted(table["site"].unique()):
        sub = table.filter(pl.col("site") == site)
        print(
            f"  {site}: rows {sub.height}; MASK_UNION minus MASK_RONC {float((sub['MASK_UNION'] - sub['MASK_RONC']).mean()):+.3f}"  # ty: ignore[invalid-argument-type]
        )
    print("\n== leave one site out: MASK_UNION or MASK_NEW minus MASK_RONC, all rows ==")
    for site in sorted(table["site"].unique()):
        sub = table.filter(pl.col("site") != site)
        m = sub["month"].to_numpy()
        for a in ("MASK_UNION", "MASK_NEW"):
            show(
                f"without {site}: {a} minus MASK_RONC",
                interval(months=m, num=(sub[a] - sub["MASK_RONC"]).to_numpy(), den=None),
            )
    neither = table.filter(pl.col("known_ronc").is_infinite() & pl.col("known_new").is_infinite())
    print(f"\nrows of transfers that neither alarm matches: {neither.height} of {table.height}")
    print("  mean quantile score:", {a: round(float(neither[a].mean()), 3) for a in ARMS})  # ty: ignore[invalid-argument-type]
    transfers = per_transfer(table)
    print(f"\n== each transfer weighted equally: {transfers.height} transfers ==")
    contrasts(transfers, name="per-transfer means")
    contrasts(
        per_transfer(table.filter(pl.col("site") != "D11")), name="per-transfer means without D11"
    )
    print("\n== error of the median and the mean forecast (contrast MASK_UNION minus MASK_RONC) ==")
    for name, w in (("err_median", err_median), ("err_mean", err_mean)):
        diff = (w["MASK_UNION"].abs() - w["MASK_RONC"].abs()).to_numpy()
        show(f"|{name}| difference", interval(months=w["month"].to_numpy(), num=diff, den=None))
    gap = (err_median["K0"].abs() - err_median["CLEAN"].abs()).to_numpy()
    print("  median minus mean forecast by site:")
    both = (
        frame.filter(pl.col("arm") == "K0")
        .group_by("site")
        .agg(diff=(pl.col("err_median") - pl.col("err_mean")).mean())
        .sort("site")
    )
    print("   ", dict(zip(both["site"], [round(x, 3) for x in both["diff"]], strict=True)))
    _ = gap
    posthoc(table)
    print(
        "\n== tail calibration on all rows after onset: share at or below the quantile (pinball) =="
    )
    tails(frame, rows=table)


if __name__ == "__main__":
    main()
