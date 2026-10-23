"""Print Part 2's tables: the share of a transfer learned as normal, by cleaning arm.

Share learned = (injected-trained forecast minus clean-trained forecast) over the 48 hours after the
transfer's end, divided by the mean injected transfer, so a positive share means the transfer was
learned as normal. Intervals are percentile bootstraps (2,000 resamples) over (site, onset) worlds.

    uv run --no-sync python e00298_report_cleaning.py > printed_cleaning.txt
"""

from __future__ import annotations

import e00298_common as common
import numpy as np
import polars as pl

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(200)
RNG = np.random.default_rng(298)


def boot(values: np.ndarray, *, n: int = 2000) -> tuple[float, float, float]:
    """Return (mean, 2.5th, 97.5th percentile) of the mean over resamples of `values`."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    means = RNG.choice(values, size=(n, len(values))).mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(v: tuple[float, float, float], scale: float = 1.0, digits: int = 1) -> str:
    """Format (mean, low, high) as `mean (low to high)`."""
    return f"{v[0] * scale:.{digits}f} ({v[1] * scale:.{digits}f} to {v[2] * scale:.{digits}f})"


def load() -> pl.DataFrame:
    """All per-window records of the clean stage."""
    files = [f for f in (common.WORLDS_DIR / "clean").glob("*.parquet") if not f.name.startswith("streams_")]
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal")


def world_table(rows: pl.DataFrame) -> pl.DataFrame:
    """One row per (world, mode) with the share learned in each after-window and the group scores."""
    key = ["site", "onset", "tag", "mode", "size", "shape", "duration", "metered"]
    during = rows.filter(pl.col("label") == "DURING").group_by(key).agg(
        transfer=(pl.col("transfer") * pl.col("n_rows")).sum() / pl.col("n_rows").sum()
    )
    out = during
    for label in ("AFTER_0_2", "AFTER_2_10", "AFTER_2_33"):
        part = rows.filter(pl.col("label") == label).group_by(key).agg(
            **{f"bias_{label}": (pl.col("bias") * pl.col("n_rows")).sum() / pl.col("n_rows").sum()}
        )
        out = out.join(part, on=key, how="left")
    for b in (3, 6, 12):
        part = rows.filter(pl.col("label") == f"STEP_B{b}").group_by(key).agg(
            **{f"bias_step{b}": (pl.col("bias") * pl.col("n_rows")).sum() / pl.col("n_rows").sum()}
        )
        out = out.join(part, on=key, how="left")
    near = rows.filter(pl.col("label").is_in(["DURING", "AFTER_0_2", "AFTER_2_10"])).group_by(key).agg(
        crps=(pl.col("crps_vs_nra") * pl.col("n_rows")).sum() / pl.col("n_rows").sum(),
        twcrps=(pl.col("twcrps_vs_nra") * pl.col("n_rows")).sum() / pl.col("n_rows").sum(),
        mae=(pl.col("mae_vs_nra") * pl.col("n_rows")).sum() / pl.col("n_rows").sum(),
    )
    flags = rows.group_by(key).agg(
        flagged_true=pl.col("flagged_days_true").max(), flagged_outside=pl.col("flagged_days_outside").max(),
        recheck=pl.col("recheck_overlap").mean(),
    )
    return out.join(near, on=key, how="left").join(flags, on=key, how="left").with_columns(
        share_0_2=pl.col("bias_AFTER_0_2") / pl.col("transfer"),
        share_2_10=pl.col("bias_AFTER_2_10") / pl.col("transfer"),
        share_2_33=pl.col("bias_AFTER_2_33") / pl.col("transfer"),
    )


def main() -> None:
    """Print the cleaning tables."""
    rows = load()
    worlds = world_table(rows)
    modes = [m for m in ("none", "oracle", "C", "E3", "F-C") if m in worlds["mode"].unique().to_list()]
    print(f"worlds x modes: {worlds.height}; modes {modes}")
    print("== Share learned as normal in the 48 h after the end (%), mean over sites and onsets ==")
    for size in (0.10, 0.25, 0.50):
        print(f"-- size {int(size * 100)}% of load, unmetered, share shape")
        for duration in (1, 3, 6, 7, 14, 30, 90):
            cells = []
            for m in modes:
                sel = worlds.filter((pl.col("mode") == m) & (pl.col("size") == size) & (pl.col("duration") == duration) & ~pl.col("metered") & (pl.col("shape") == "share"))
                cells.append(f"{m}: {fmt(boot(sel['share_0_2'].to_numpy()), 100, 1)}")
            print(f"{duration:3d} d  " + "   ".join(cells))
    print("== Headline cell: 30 days, 25% of load; share learned (%) by window after the end ==")
    for m in modes:
        sel = worlds.filter((pl.col("mode") == m) & (pl.col("size") == 0.25) & (pl.col("duration") == 30) & ~pl.col("metered") & (pl.col("shape") == "share"))
        print(f"{m:7s} 0-2 d {fmt(boot(sel['share_0_2'].to_numpy()), 100)}  2-10 d {fmt(boot(sel['share_2_10'].to_numpy()), 100)}  2-33 d {fmt(boot(sel['share_2_33'].to_numpy()), 100)}  n={sel.height}")
    print("== Same cell, paired difference from no cleaning (percentage points) ==")
    base = worlds.filter((pl.col("mode") == "none") & (pl.col("size") == 0.25) & (pl.col("duration") == 30) & ~pl.col("metered") & (pl.col("shape") == "share")).select("site", "onset", base=pl.col("share_0_2"))
    for m in modes[1:]:
        sel = worlds.filter((pl.col("mode") == m) & (pl.col("size") == 0.25) & (pl.col("duration") == 30) & ~pl.col("metered") & (pl.col("shape") == "share")).join(base, on=["site", "onset"])
        print(f"{m:7s} {fmt(boot((sel['share_0_2'] - sel['base']).to_numpy()), 100)}")
    print("== Metered receiver worlds (25%, share): share learned in 0-2 d (%) ==")
    for duration in (7, 14, 30, 90):
        cells = []
        for m in modes:
            sel = worlds.filter((pl.col("mode") == m) & (pl.col("duration") == duration) & pl.col("metered") & (pl.col("size") == 0.25))
            cells.append(f"{m}: {fmt(boot(sel['share_0_2'].to_numpy()), 100)}")
        print(f"{duration:3d} d  " + "   ".join(cells))
    print("== Fixed-block shape (25%): share learned in 0-2 d (%) ==")
    for duration in (7, 30, 90):
        cells = []
        for m in modes:
            sel = worlds.filter((pl.col("mode") == m) & (pl.col("duration") == duration) & ~pl.col("metered") & (pl.col("size") == 0.25) & (pl.col("shape") == "fixed"))
            cells.append(f"{m}: {fmt(boot(sel['share_0_2'].to_numpy()), 100)}")
        print(f"{duration:3d} d  " + "   ".join(cells))
    print("== Permanent steps (unmetered): share learned (%) after 3, 6 and 12 months ==")
    for size in (0.10, 0.25):
        for m in modes:
            sel = worlds.filter((pl.col("mode") == m) & (pl.col("duration") == -1) & ~pl.col("metered") & (pl.col("size") == size))
            print(f"{int(size * 100)}% {m:7s} " + "  ".join(f"{b} mo {fmt(boot((sel[f'bias_step{b}'] / sel['transfer']).to_numpy()), 100)}" for b in (3, 6, 12)))
    print("== Flagged days in the whole run, true shift and elsewhere (mean per world) ==")
    for m in modes[2:]:
        sel = worlds.filter((pl.col("mode") == m) & ~pl.col("metered") & (pl.col("shape") == "share"))
        for dur_label, cond in (("steps", pl.col("duration") == -1), ("30 d", pl.col("duration") == 30), ("1 to 7 d", pl.col("duration").is_between(1, 7))):
            part = sel.filter(cond)
            print(f"{m:5s} {dur_label:9s} true {part['flagged_true'].mean():.1f}  outside {part['flagged_outside'].mean():.1f}  recheck overlap {part['recheck'].mean():.2f}")
    print("== Near-a-shift rows (during + 10 days after), 30 d, 25%: CRPS, twCRPS, MAE against the NRA series (points) ==")
    for m in modes:
        sel = worlds.filter((pl.col("mode") == m) & (pl.col("size") == 0.25) & (pl.col("duration") == 30) & ~pl.col("metered") & (pl.col("shape") == "share"))
        print(f"{m:7s} CRPS {sel['crps'].mean():.3f}  twCRPS {sel['twcrps'].mean():.3f}  MAE {sel['mae'].mean():.3f}")
    print("== Clean-row control (no shift): MAE against observed (points), cost against no cleaning ==")
    ctrl_files = list((common.WORLDS_DIR / "control").glob("*.parquet"))
    ctrl_files = [f for f in ctrl_files if not f.name.startswith("streams_")]
    if ctrl_files:
        c = pl.concat([pl.read_parquet(f) for f in ctrl_files], how="diagonal").filter(pl.col("label") == "ALL")
        per = c.group_by("site", "mode").agg(mae=(pl.col("mae_vs_nra") * pl.col("n_rows")).sum() / pl.col("n_rows").sum(), flagged=pl.col("n_flagged").max())
        base_c = per.filter(pl.col("mode") == "none").select("site", base=pl.col("mae"))
        for m in [x for x in modes if x not in ("none", "oracle")]:
            part = per.filter(pl.col("mode") == m).join(base_c, on="site")
            diff = (part["mae"] - part["base"]).to_numpy()
            print(f"{m:5s} cost {fmt(boot(diff), 1, 4)}  flagged days (max per site) mean {part['flagged'].mean():.0f}")
            print(part.with_columns(cost=pl.col("mae") - pl.col("base")).sort("site"))


if __name__ == "__main__":
    main()
