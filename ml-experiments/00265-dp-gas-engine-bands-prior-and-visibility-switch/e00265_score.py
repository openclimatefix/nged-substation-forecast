# ruff: noqa: E501
"""Score every arm of experiment 265 and print each number the write-up quotes.

Quantile score is twice the mean pinball loss over the 11 quantiles, in points of effective
capacity. Intervals resample the 12 calendar months of the scored year (20,000 draws).

    uv run --no-sync python ml-experiments/00265-*/e00265_score.py
"""

import e00265_common as common
import e00265_widen as widen
import numpy as np
import polars as pl

N_RESAMPLES = 20_000
QUARTERS: dict[str, tuple[int, int]] = {
    "2025Q3": (7, 9),
    "2025Q4": (10, 12),
    "2026Q1": (1, 3),
    "2026Q2": (4, 6),
}
TAILS: tuple[tuple[str, float], ...] = (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99))
HEAVY_SHARE = 0.15
COVERAGE_BAND = (0.85, 0.95)


def load_arm(*, arm: str, site: str) -> pl.DataFrame:
    """Return one arm's stored forecast frame."""
    return pl.read_parquet(common.PREDS_DIR / arm / f"{site}.parquet")


def row_losses(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return per-row arrays: quantile score, mean absolute error, coverage, and tail losses."""
    q = frame.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    error = actual[:, None] - q
    pinball = np.maximum(common.LEVELS[None, :] * error, (common.LEVELS[None, :] - 1.0) * error)
    out = {
        "qs": 2.0 * pinball.mean(axis=1),
        "ae": np.abs(actual - frame["mean"].to_numpy().astype(np.float64)),
        "in90": ((actual >= q[:, 1]) & (actual <= q[:, 9])).astype(np.float64),
        "median_minus_mean": q[:, 5] - frame["mean"].to_numpy().astype(np.float64),
    }
    for name, level in TAILS:
        j = common.QUANTILES.index(name)
        e = actual - q[:, j]
        out[f"pin_{name}"] = np.maximum(level * e, (level - 1.0) * e)
        out[f"below_{name}"] = (actual < q[:, j]).astype(np.float64)
    return out


def month_of(*, frame: pl.DataFrame) -> np.ndarray:
    """Return year*100 + month of each row's valid time."""
    return (frame["valid_time"].dt.year() * 100 + frame["valid_time"].dt.month()).to_numpy()


def month_block_interval(
    *, difference: np.ndarray, months: np.ndarray, level: float
) -> tuple[float, float, float]:
    """Return the mean of `difference` and its month-resampled interval at `level` (two-sided)."""
    keys = np.unique(months)
    sums = np.array([difference[months == m].sum() for m in keys])
    counts = np.array([(months == m).sum() for m in keys], dtype=np.float64)
    draws = np.random.default_rng(265).integers(0, len(keys), size=(N_RESAMPLES, len(keys)))
    stats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(stats, [tail, 1.0 - tail])
    return float(sums.sum() / counts.sum()), float(low), float(high)


def quarter_mask(*, months: np.ndarray, quarter: str) -> np.ndarray:
    """Return a mask of the rows in the quarter."""
    m0, m1 = QUARTERS[quarter]
    return np.array([(m0 <= m % 100 <= m1) for m in months])


def census(*, site: str) -> dict:
    """Return, by quarter, the reference's shares below q05, above q95, and residual below -25 points."""
    frame = widen_free_reference(site=site)
    months = month_of(frame=frame)
    q05, q95, q50 = (frame[n].to_numpy().astype(np.float64) for n in ("q05", "q95", "q50"))
    actual = frame["actual"].to_numpy().astype(np.float64)
    out = {}
    for quarter in QUARTERS:
        in_q = quarter_mask(months=months, quarter=quarter)
        out[quarter] = {
            "below_q05": float((actual[in_q] < q05[in_q]).mean()),
            "above_q95": float((actual[in_q] > q95[in_q]).mean()),
            "residual_below_minus_25": float(((actual - q50)[in_q] < common.DROP_POINTS).mean()),
        }
    out["heavy"] = sum(out[q]["below_q05"] >= HEAVY_SHARE for q in QUARTERS) >= 2
    return out


def widen_free_reference(*, site: str) -> pl.DataFrame:
    """Return the unwidened reference rows of a site."""
    return load_arm(arm="arm1", site=site)


def main() -> None:
    """Print the absolute scores, contrasts, and checks."""
    arms = common.ARMS
    report: dict = {"census": {}, "sites": {}, "flips": {}}
    print("=== census (reference, by quarter): below q05 / above q95 / residual < -25 points")
    for site in common.SITES:
        c = census(site=site)
        report["census"][site] = c
        print(
            f"{site} heavy={c['heavy']}: "
            + "; ".join(
                f"{q} {c[q]['below_q05']:.2f}/{c[q]['above_q95']:.2f}/{c[q]['residual_below_minus_25']:.3f}"
                for q in QUARTERS
            )
        )
    for site in common.SITES:
        frames = {arm: load_arm(arm=arm, site=site) for arm in arms}
        keys = frames["arm1"].select(common.KEYS)
        for arm, frame in frames.items():
            assert frame.select(common.KEYS).equals(keys), f"{site} {arm} row keys differ"
        base = frames["arm1"]
        months = month_of(frame=base)
        losses = {arm: row_losses(frame=frame) for arm, frame in frames.items()}
        rep = report["sites"][site] = {}
        print(f"\n=== {site}: {base.height:,} rows (heavy={report['census'][site]['heavy']})")
        for arm in arms:
            row = {
                "qs": float(losses[arm]["qs"].mean()),
                "ae": float(losses[arm]["ae"].mean()),
                "cov90": float(losses[arm]["in90"].mean()),
                "median_minus_mean": float(losses[arm]["median_minus_mean"].mean()),
            }
            for quarter in QUARTERS:
                in_q = quarter_mask(months=months, quarter=quarter)
                row[f"cov90_{quarter}"] = float(losses[arm]["in90"][in_q].mean())
                row[f"qs_{quarter}"] = float(losses[arm]["qs"][in_q].mean())
            for name, _ in TAILS:
                row[f"below_{name}"] = float(losses[arm][f"below_{name}"].mean())
                row[f"pin_{name}"] = float(losses[arm][f"pin_{name}"].mean())
            row["quarters_in_band"] = sum(
                COVERAGE_BAND[0] <= row[f"cov90_{q}"] <= COVERAGE_BAND[1] for q in QUARTERS
            )
            rep[arm] = row
            print(
                f"{arm:10s} qs={row['qs']:.4f} mae={row['ae']:.3f} cov90={row['cov90']:.3f} quarters "
                + " ".join(f"{q}={row[f'cov90_{q}']:.3f}" for q in QUARTERS)
                + f" in-band {row['quarters_in_band']}/4 med-mean={row['median_minus_mean']:+.3f}"
            )
            print(
                "           tails "
                + " ".join(
                    f"{n}: below={row[f'below_{n}']:.3f} pin={row[f'pin_{n}']:.3f}"
                    for n, _ in TAILS
                )
            )
        pairs = [
            (a, "control") for a in ("arm1", "arm2", "arm2c", "arm3", "arm3_sens", "arm2p", "arm3p")
        ] + [
            ("arm2", "arm1"),
            ("arm3", "arm1"),
            ("arm3", "arm2"),
            ("arm2c", "arm2"),
            ("arm2p", "arm1"),
            ("arm3p", "arm2p"),
            ("arm3p", "arm1"),
        ]
        for arm, ref in pairs:
            diff = losses[arm]["qs"] - losses[ref]["qs"]
            mean, low, high = month_block_interval(difference=diff, months=months, level=0.95)
            ref_mean = float(losses[ref]["qs"].mean())
            loo = [float(diff[months != m].mean()) for m in np.unique(months)]
            ae_diff = float((losses[arm]["ae"] - losses[ref]["ae"]).mean())
            rep[f"{arm}_minus_{ref}"] = {
                "qs": mean,
                "low95": low,
                "high95": high,
                "pct": mean / ref_mean * 100,
                "pct_low95": low / ref_mean * 100,
                "pct_high95": high / ref_mean * 100,
                "loo_signs": [int(np.sign(v)) for v in loo],
                "ae_diff": ae_diff,
            }
            print(
                f"  {arm}-{ref} qs {mean:+.4f} [{low:+.4f}, {high:+.4f}] = {mean / ref_mean * 100:+.2f}% ({low / ref_mean * 100:+.2f}%, {high / ref_mean * 100:+.2f}%); loo signs {[int(np.sign(v)) for v in loo]}; MAE diff {ae_diff:+.4f}"
            )
        ref_frame = common.load_reference(site=site)
        report["flips"][site] = {
            f"{on:.2f}_{off:.2f}": widen.flip_dates(reference=ref_frame, on=on, off=off)
            for on, off in ((0.07, 0.04), (0.05, 0.03))
        }
        print("  switch flips:", report["flips"][site])
    print(
        "\n=== pass rule at each site (a: all four quarters in [0.85, 0.95]; b: D8 qs <= control +0.5%; c: placebo qs vs control <= +0.1% point and <= +0.5% upper)"
    )
    for arm in ("arm2", "arm2c", "arm3", "arm3_sens", "arm2p", "arm3p"):
        for site in common.SITES:
            r = report["sites"][site]
            d = r[f"{arm}_minus_control"]
            heavy = report["census"][site]["heavy"]
            a = r[arm]["quarters_in_band"] == 4
            b = d["pct"] <= 0.5
            c = d["pct"] <= 0.1 and d["pct_high95"] <= 0.5
            print(
                f"{arm:9s} {site:3s} heavy={heavy!s:5s} (a) in-band quarters {r[arm]['quarters_in_band']}/4 -> {a}; (b) qs vs control {d['pct']:+.2f}% -> {b}; (c) upper {d['pct_high95']:+.2f}% -> {c}"
            )
    common.write_json_atomic(path=common.WORK_DIR / "score.json", content=report)


if __name__ == "__main__":
    main()
