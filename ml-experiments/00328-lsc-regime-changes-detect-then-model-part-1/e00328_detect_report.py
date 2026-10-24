# ruff: noqa: E501, PLR0912, PLR0915, PTH207
"""Stage 1 report: choose each detector's threshold on the tune Primaries, score on the held-out ones.

The rule was fixed before any recall was read: the threshold is the smallest one in the grid whose
false changes per Primary-year on the real records of the tune Primaries are at most 1. Recall is
then read on the held-out Primaries at even-month onsets. Prints everything the write-up quotes.

    PYTHONPATH=ml-experiments/00295-*:ml-experiments/00328-* uv run --no-sync python \
        ml-experiments/00328-*/e00328_detect_report.py
"""

from __future__ import annotations

import glob
import json

import e00295_common as c295  # ty: ignore[unresolved-import]
import e00328_common as c
import numpy as np
import polars as pl

D8_FADE = np.datetime64("2026-03-01")
MATCH_BEFORE_DAYS, MATCH_AFTER_DAYS = 30, 120
ECR_ASSETS = (
    c.ML_EXPERIMENTS_DATA_DIR
    / "00191-dp-price-sensitive-assets-in-the-history"
    / "work"
    / "ecr_assets.parquet"
)


def false_rate(*, alarms: pl.DataFrame, sites: list[str], years: dict[str, float]) -> pl.DataFrame:
    """Return alarms per Primary-year at each detector and threshold over the given Primaries."""
    counts = (
        alarms.filter(pl.col("site").is_in(sites))
        .group_by("detector", "param")
        .agg(pl.len().alias("n"))
    )
    total = sum(years[s] for s in sites)
    return counts.with_columns(rate=pl.col("n") / total).sort("detector", "param")


def bootstrap_mean(*, groups: dict[str, np.ndarray], n: int = 4000) -> tuple[float, float, float]:
    """Return the mean and 95% interval of a mean over groups, resampling groups."""
    rng = np.random.default_rng(0)
    keys = list(groups)
    values = np.array([groups[k].mean() for k in keys])
    draws = [values[rng.integers(0, len(keys), len(keys))].mean() for _ in range(n)]
    return float(values.mean()), *map(float, np.percentile(draws, [2.5, 97.5]))  # ty: ignore[invalid-return-type]


def main() -> None:
    """Print the detection report and write the regime log."""
    chosen_295 = json.loads((c295.EXPERIMENT_DIR / "chosen_config.json").read_text())
    tune, hold = chosen_295["tune_sites"], chosen_295["hold_sites"]
    base = pl.concat(
        [pl.read_parquet(p) for p in glob.glob(str(c.WORK_DIR / "base" / "*_alarms.parquet"))]
    )
    spans = {}
    for path in glob.glob(str(c.WORK_DIR / "base" / "*.npz")):
        site = path.split("/")[-1].removesuffix(".npz")
        t = np.load(path)["times"]
        spans[site] = float((t[-1] - t[0]) / np.timedelta64(1, "D")) / 365.25
    non_d8 = base.filter(pl.col("site") != "D8")
    print("False changes per Primary-year on the real records (load shifts count as false)")
    print("tune Primaries:\n", false_rate(alarms=non_d8, sites=tune, years=spans))
    print("held-out Primaries:\n", false_rate(alarms=non_d8, sites=hold, years=spans))
    tune_rates = false_rate(alarms=non_d8, sites=tune, years=spans)
    chosen: dict[str, float] = {}
    for det in ("rolling_gam", "histogram"):
        sub = tune_rates.filter(pl.col("detector") == det)
        grid = c.CUSUM_H_GRID if det == "rolling_gam" else c.HISTOGRAM_BETA_GRID
        rates = dict(zip(sub["param"].to_list(), sub["rate"].to_list(), strict=True))
        for p in grid:  # a parameter with no alarms at all is absent from the table
            if rates.get(p, 0.0) <= 1.0:
                chosen[det] = p
                break
    print("\nChosen threshold (smallest with <= 1 false change per Primary-year on tune):", chosen)
    (c.WORK_DIR / "chosen_detectors.json").write_text(json.dumps(chosen))
    injected = pl.concat(
        [pl.read_parquet(p) for p in glob.glob(str(c.WORK_DIR / "inject" / "*.parquet"))]
    )
    for det, p in chosen.items():
        sel = injected.filter((pl.col("detector") == det) & (pl.col("param") == p))
        for label, sites, parity in (
            ("held-out, even onsets (REPORT)", hold, "report"),
            ("tune, odd onsets", tune, "tune"),
        ):
            part = sel.filter(pl.col("site").is_in(sites) & (pl.col("parity") == parity))
            print(f"\n{det} at {p}: {label}")
            for (shape, size), g in part.group_by("shape", "size", maintain_order=True):
                per_site = {
                    s: x["detected"].cast(pl.Float64).to_numpy() for (s,), x in g.group_by("site")
                }
                m, lo, hi = bootstrap_mean(groups=per_site)
                d = g.filter(pl.col("detected"))
                med = d["delay_days"].median() if d.height else float("nan")
                err = d["date_error_days"].median() if d.height else float("nan")
                print(
                    f"  {shape:5s} {size:.2f}: recall within 60 days {m:.2f} ({lo:.2f} to {hi:.2f}), n={g.height}, median delay {med} d, median date error {err} d"
                )
    print("\nAll thresholds, held-out Primaries, 10% steps (recall within 60 days):")
    print(
        injected.filter(
            pl.col("site").is_in(hold)
            & (pl.col("parity") == "report")
            & (pl.col("shape") == "step")
            & (pl.col("size") == 0.10)
        )
        .group_by("detector", "param")
        .agg(recall=pl.col("detected").mean())
        .sort("detector", "param")
    )
    print("\nD8 (real record): alarms near the 2023-08-15 break, and in 2026")
    d8 = base.filter(pl.col("site") == "D8").with_columns(
        pl.col("alarm_day").str.to_date(), pl.col("change_day").str.to_date()
    )
    for det, p in chosen.items():
        rows = d8.filter((pl.col("detector") == det) & (pl.col("param") == p)).sort("alarm_day")
        print(det, p, rows.select("alarm_day", "change_day").rows())
    print("\nECR connection dates (>= 1 MW, since 2020-06-01) against the chosen detectors' alarms")
    if ECR_ASSETS.exists():
        ecr = pl.read_parquet(ECR_ASSETS).filter(
            (pl.col("mw") >= 1.0) & (pl.col("connected_date") >= pl.date(2020, 6, 1))
        )
        for row in ecr.sort("site", "connected_date").iter_rows(named=True):
            hits = []
            for det, p in chosen.items():
                a = base.filter(
                    (pl.col("site") == row["site"])
                    & (pl.col("detector") == det)
                    & (pl.col("param") == p)
                )
                for ch in a["change_day"].str.to_date().to_list():
                    gap = (ch - row["connected_date"]).days
                    if -MATCH_BEFORE_DAYS <= gap <= MATCH_AFTER_DAYS:
                        hits.append(f"{det} {gap:+d} d")
            print(
                f"  {row['site']} {row['connected_date']} {row['mw']:.1f} MW {row['asset_class'] or row['source']}: {', '.join(hits) or 'no alarm'}"
            )
        n_assets = ecr.height
        print(f"  ({n_assets} assets of 1 MW or more connected since 2020-06-01)")
    # the regime log used by stage 2
    log = base.join(
        pl.DataFrame({"detector": list(chosen), "param": list(chosen.values())}),
        on=["detector", "param"],
    )
    log.write_parquet(c.WORK_DIR / "regime_log.parquet")


if __name__ == "__main__":
    main()
