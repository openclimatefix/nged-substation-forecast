# ruff: noqa: E501
"""Re-blend experiment 87's stored Primary member forecasts with one member dropped, and score D8.

Reads the stored refit predictions of experiment 87 (no fits), rebuilds BG2's five members for all
16 Primaries with experiment 87's own code (vintage offsets, spread factor 1.15, 14-day correction
on the stack member only), then blends every subset with experiment 87's rolling monthly weights
(re-chosen on the grid for each subset, over all 16 Primaries). Scores the sites D8 (the 96,986 rows
of experiment 355) and D2, D3, D4 (placebos; all stored rows up to 168 h with a finite actual).

    PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00360-*/e00360_drop_one.py
"""

import dataclasses
import json
import sys
from pathlib import Path
from typing import Final

import e00087_assemble as asm  # ty: ignore[unresolved-import]
import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

OUT: Final[Path] = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/00360-bg2-member-drop-one-at-d8/work"
)
D8_ROWS: Final[Path] = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/00355-d8-weekly-refit-and-median-band-swap/work/scores_wide.parquet"
)
SITES: Final[tuple[str, ...]] = ("D8", "D2", "D3", "D4")
MEMBERS: Final[tuple[str, ...]] = asm.BLEND_FIVE
LEVELS_HEAD: Final[tuple[str, ...]] = common.LEVEL_COLUMNS
N_RESAMPLES: Final[int] = 20_000
SEED: Final[int] = 0
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
"""The 11 levels, scored as an unweighted mean as experiment 252 and 354 do (not 87's atom masses)."""


def score_rows(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the quantile score per row in points of capacity."""
    quantiles = frame.select(LEVELS_HEAD).to_numpy().astype(np.float64)
    error = frame["actual"].to_numpy().astype(np.float64)[:, None] - quantiles
    return 100.0 * 2.0 * np.mean(np.maximum(LEVELS * error, (LEVELS - 1.0) * error), axis=1)


def month_bootstrap(*, values: np.ndarray, months: np.ndarray) -> tuple[float, float, float]:
    """Return the mean and its 95% interval, resampling whole valid months."""
    labels, index = np.unique(months, return_inverse=True)
    sums = np.bincount(index, weights=values, minlength=labels.size)
    counts = np.bincount(index, minlength=labels.size).astype(np.float64)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, labels.size, size=(N_RESAMPLES, labels.size))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(sums.sum() / counts.sum()), float(low), float(high)


def score_site(*, site: str, frame: pl.DataFrame, capacity: float, names: list[str]) -> dict:
    """Score every arm at one site and return the table written to `results.json`."""
    months = (
        frame["valid_time"].dt.convert_time_zone("Europe/London").dt.strftime("%Y-%m").to_numpy()
    )
    per_row = {
        n: score_rows(frame=frame.select("actual", **{c: pl.col(f"{n}__{c}") for c in LEVELS_HEAD}))
        for n in names
    }
    actual_points = frame["actual"].to_numpy().astype(np.float64)
    out: dict = {"rows": frame.height, "capacity_mw": capacity, "arms": {}}
    if site == "D8":
        out["gate"] = {
            "stored_bg2_score": float(frame["s_bg2"].mean()),  # ty: ignore[invalid-argument-type]
            "reblended_full_score": float(per_row["full"].mean()),
            "median_gap_points": float(
                np.abs(frame["full__q50"].to_numpy() * 100 - frame["bg2_q50"].to_numpy()).max()
            ),
        }
    for n in names:
        mean, low, high = month_bootstrap(values=per_row[n], months=months)
        q50 = frame[f"{n}__q50"].to_numpy().astype(np.float64)
        q05 = frame[f"{n}__q05"].to_numpy().astype(np.float64)
        q95 = frame[f"{n}__q95"].to_numpy().astype(np.float64)
        entry = {
            "score": mean,
            "ci": [low, high],
            "median_error_mw": float(np.abs(actual_points - q50).mean() * capacity),
            "coverage_q05_q95": float(((actual_points >= q05) & (actual_points <= q95)).mean()),
        }
        if n != "full":
            d_mean, d_low, d_high = month_bootstrap(
                values=per_row[n] - per_row["full"], months=months
            )
            entry["minus_full"] = {"mean": d_mean, "ci": [d_low, d_high]}
        out["arms"][n] = entry
    return out


def rescore() -> None:
    """Score the saved site frames again (no blending)."""
    old = json.loads((OUT / "results.json").read_text())
    results = {}
    for site in SITES:
        frame = pl.read_parquet(OUT / f"frame_{site}.parquet")
        names = sorted(
            {c.split("__")[0] for c in frame.columns if "__q50" in c},
            key=lambda n: list(old[site]["arms"]).index(n),
        )
        results[site] = score_site(
            site=site, frame=frame, capacity=old[site]["capacity_mw"], names=names
        )
        print(
            site,
            json.dumps({n: round(v["score"], 3) for n, v in results[site]["arms"].items()}),
            flush=True,
        )
    (OUT / "results.json").write_text(json.dumps(results, indent=1))


def main() -> None:
    """Build every arm, score it, and write `work/results.json`."""
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = dataclasses.replace(
        asm.make_context(smoke=False, store=False), results_dir=OUT / "weights"
    )
    primary = ctx.sites_of(groups=("Primary",))
    tables: dict[str, pl.DataFrame] = {}
    for member in MEMBERS:
        raw = asm.raw_member(ctx=ctx, member=member, offset="vintage", sites=primary)
        tables[member] = asm.finish_member(ctx=ctx, raw=raw, correct=member == "ST", spread=True)
        print(f"member {member}: {tables[member].height:,} rows", flush=True)
    keys, by_member = asm.align(tables=tables, strict=True)
    del tables
    actual = asm.actual_array(frame=keys, power_of=ctx.power_of)
    arms: dict[str, np.ndarray] = {}
    arms["full"], weights = asm.rolling_blend(
        keys=keys, actual=actual, members=[by_member[m] for m in MEMBERS], names=list(MEMBERS)
    )
    weights.write_parquet(OUT / "weights_full.parquet")
    for dropped in MEMBERS:
        kept = [m for m in MEMBERS if m != dropped]
        arms[f"drop_{dropped}"], weights = asm.rolling_blend(
            keys=keys, actual=actual, members=[by_member[m] for m in kept], names=kept
        )
        weights.write_parquet(OUT / f"weights_drop_{dropped}.parquet")
    for member in MEMBERS:
        arms[f"only_{member}"] = by_member[member]
    names = list(arms)
    ids = keys["time_series_id"].to_numpy()
    site_of = {i: ctx.labels[int(i)][0] for i in np.unique(ids)}
    site_column = np.array([site_of[i] for i in ids])
    results: dict = {}
    for site in SITES:
        mask = site_column == site
        base = keys.filter(pl.Series(mask)).with_columns(
            actual=pl.Series(actual[mask]),
            lead_hours=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
            ),
        )
        capacity = float(ctx.capacity[int(ids[mask][0])])
        frame = base.hstack(
            pl.DataFrame(
                {f"{n}__{c}": arms[n][mask][:, k] for n in names for k, c in enumerate(LEVELS_HEAD)}
            )
        )
        if site == "D8":
            rows = pl.read_parquet(D8_ROWS).select(
                "power_fcst_init_time", "valid_time", "bg2_q50", "s_bg2", actual_355="actual"
            )
            frame = frame.join(rows, on=["power_fcst_init_time", "valid_time"], how="inner")
            # Experiment 355's actual (MVA-fold series, in points of capacity) replaces the raw series.
            frame = frame.with_columns(actual=pl.col("actual_355") / 100.0)
        else:
            frame = frame.filter(
                pl.col("actual").is_finite()
                & (pl.col("lead_hours") <= 168)
                & (pl.col("valid_time") < common.HOLDOUT)
            )
        frame.write_parquet(OUT / f"frame_{site}.parquet")
        out = score_site(site=site, frame=frame, capacity=capacity, names=names)
        results[site] = out
        print(
            site, json.dumps({n: round(v["score"], 3) for n, v in out["arms"].items()}), flush=True
        )
    (OUT / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    if "--rescore" in sys.argv:
        rescore()
    else:
        main()
