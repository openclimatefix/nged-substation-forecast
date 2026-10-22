# ruff: noqa: D103,E501,PTH207
"""Experiment 200: paired contrasts between the head, the blind control, and arm D (stage 2).

Reads the row files written by `e00200_score.py` and prints each contrast with a 99%
month-resampled interval, for the five decision substations and for all six.
"""

import e00200_data as data_module
import e00200_score as score_module
import numpy as np
import polars as pl

DECISION = ["D5", "D10", "D11", "D14", "D16"]


def main() -> None:
    head = pl.read_parquet(data_module.WORK_DIR / "stage2_head_rows.parquet")
    blind = pl.read_parquet(data_module.WORK_DIR / "stage2_blind_rows.parquet")
    rng = np.random.default_rng(0)
    h = head.filter(pl.col("arm") == "head")
    b = blind.filter(pl.col("arm") == "head")
    d = head.filter(pl.col("arm") == "arm_d")
    for scope, sites in (("decision", DECISION), ("all6", None)):
        sel = pl.col("site").is_in(sites) if sites else pl.lit(True)
        hs, bs, ds = (x.filter(sel) for x in (h, b, d))
        months = ds["month"].to_numpy()
        for name, left, right in (
            ("head - arm D", hs, ds),
            ("blind - arm D", bs, ds),
            ("head - blind", hs, bs),
        ):
            diff = left["loss"].to_numpy() - right["loss"].to_numpy()
            m, lo, hi = score_module.interval(diff=diff, months=months, level=0.99, rng=rng)
            print(
                f"{scope:8s} {name:14s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                f"arms: {left['loss'].mean():.4f} vs {right['loss'].mean():.4f}"
            )
        for name, x in (("arm D", ds), ("blind", bs), ("head", hs)):
            print(
                f"{scope:8s} {name:6s} cover95 {x['cover95'].mean():.3f} cover99 {x['cover99'].mean():.3f} "
                f"cover05 {x['cover05'].mean():.3f} cover01 {x['cover01'].mean():.3f} "
                f"pin01 {x['pin01'].mean():.3f} pin99 {x['pin99'].mean():.3f} mae {x['abs_err'].mean():.3f}"
            )
    # Leave-one-site-out for head - arm D
    for site in sorted(set(d["site"])):
        keep = pl.col("site") != site
        diff = h.filter(keep)["loss"].to_numpy() - d.filter(keep)["loss"].to_numpy()
        print(f"without {site}: head - arm D {diff.mean():+.4f}")


if __name__ == "__main__":
    main()


def spread_a_check() -> None:
    """Score experiment 169's calibrated-spread arm D (`d_spread_a`) on the same rows."""
    import glob

    q = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
    levels = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
    frames = [
        pl.read_parquet(p)
        for p in sorted(glob.glob(str(data_module.SRC_DIR / "d_spread_a_scored_D*.parquet")))
    ]
    df = pl.concat(frames).filter(
        (pl.col("seed") == 0)
        & pl.col("lead_hours").is_between(data_module.SCORE_LEAD_MIN, data_module.SCORE_LEAD_MAX)
        & ~pl.col("switching_mask")
        & (pl.col("power_fcst_init_time") >= pl.datetime(2025, 7, 2, time_zone="UTC"))
    )
    labels = score_module.site_labels()
    df = df.filter(pl.col("time_series_id").is_in([k for k, v in labels.items() if v in DECISION]))
    y = df["actual"].to_numpy().astype(np.float64)
    diff = y[:, None] - df.select(q).to_numpy().astype(np.float64)
    loss = 2 * np.maximum(levels * diff, (levels - 1) * diff).mean(axis=1)
    print(f"d_spread_a on decision rows: {loss.mean():.4f} ({len(loss)} rows)")


if __name__ == "__main__":
    spread_a_check()
