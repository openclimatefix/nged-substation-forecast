"""Print every number the write-up quotes, from the stored segmentation and gain tables."""

from __future__ import annotations

import json
import pickle

import e00295_common as c
import numpy as np
import polars as pl
from e00295_tune import YEARS, load

LEVEL = 0.9875
DRAWS = 20000


def cluster_interval(
    *, values: np.ndarray, clusters: np.ndarray, scale: float = 1.0, seed: int = 0
) -> tuple[float, float, float]:
    """Return the mean of `values` times `scale`, with a cluster-bootstrap interval at LEVEL."""
    names = np.unique(clusters)
    sums = np.array([values[clusters == n].sum() for n in names])
    counts = np.array([(clusters == n).sum() for n in names])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(names), size=(DRAWS, len(names)))
    draws = sums[pick].sum(axis=1) / counts[pick].sum(axis=1) * scale
    tail = (1 - LEVEL) / 2 * 100
    return (
        float(sums.sum() / counts.sum() * scale),
        float(np.percentile(draws, tail)),
        float(np.percentile(draws, 100 - tail)),
    )


def fmt(triple: tuple[float, float, float]) -> str:
    """Format a mean with its interval."""
    return f"{triple[0]:.3f} [{triple[1]:.3f}, {triple[2]:.3f}]"


def main() -> None:
    """Print the report."""
    chosen = json.loads((c.EXPERIMENT_DIR / "chosen_config.json").read_text())
    beta, length = chosen["beta"], chosen["min_length"]
    tune, hold = chosen["tune_sites"], chosen["hold_sites"]
    print(f"chosen config: beta={beta}, min_length={length}, constraint met={chosen['constraint_met']}")
    print(f"tuning Primaries: {tune}\nheld-out Primaries: {hold}")
    clean, injections = load("clean_break"), load("injection")
    at = (pl.col("beta") == beta) & (pl.col("min_length") == length)

    states = pickle.loads((c.EXPERIMENT_DIR / "clean_states.pkl").read_bytes())  # noqa: S301
    print("\n== D8 known break 2023-08-15: nearest boundary, by configuration ==")
    for (b, m), bounds in sorted(states["D8"]["bounds"].items()):
        if not bounds:
            continue
        nearest = min(bounds, key=lambda x: abs(c.days_between(c.D8_BREAK, x.date)))
        mark = " <- chosen" if (b, m) == (beta, length) else ""
        print(f"beta={b:>5} m={m:>2}: {nearest.date} error {c.days_between(c.D8_BREAK, nearest.date):+.0f} days, shift {nearest.shift[0]:+.2f} points, n_breaks={len(bounds)}{mark}")

    print("\n== False breaks per Primary-year at the unmodified series (all months) ==")
    for name, sites in (("held-out", hold), ("tuning", tune)):
        rows = clean.filter(at & pl.col("site").is_in(sites))
        per_site = rows.group_by("site").agg(n=pl.len())
        counts = np.array([per_site.filter(pl.col("site") == s)["n"].sum() for s in sites], dtype=float)
        draws = np.random.default_rng(1).integers(0, len(sites), size=(DRAWS, len(sites)))
        rate = counts[draws].sum(axis=1) / (len(sites) * YEARS)
        tail = (1 - LEVEL) / 2 * 100
        print(f"{name}: {counts.sum() / (len(sites) * YEARS):.2f} per Primary-year, 98.75% interval [{np.percentile(rate, tail):.2f}, {np.percentile(rate, 100 - tail):.2f}]; per site {dict(zip(sites, counts.astype(int).tolist(), strict=True))}")
    rows = clean.filter(at & pl.col("site").is_in(hold)).with_columns(
        night=pl.col("shift_level") + pl.col("shift_shape") / 2,
        day=pl.col("shift_level") - pl.col("shift_shape") / 2,
    )
    pv_like = rows.filter(pl.col("day").abs() >= 2 * pl.col("night").abs())
    print(f"PV-like (day shift at least twice night shift, approximate): {pv_like.height} of {rows.height}; April to August: {rows.filter(pl.col('date').str.slice(5, 2).cast(pl.Int32).is_between(4, 8)).height} of {rows.height}")
    print(f"labels of held-out clean breaks: {rows.group_by('label').agg(n=pl.len()).sort('label').rows()}")

    print("\n== Injection recall at the chosen config (held-out Primaries, even-month onsets) ==")
    inj = injections.filter(at & pl.col("site").is_in(hold) & (pl.col("parity") == "even"))
    for absorption in ("fit", "residual"):
        for family in ("persistent", "temporary"):
            rows = inj.filter((pl.col("absorption") == absorption) & (pl.col("family") == family))
            hit = (rows["label"] == rows["expected"]).to_numpy().astype(float)
            found = rows["found"].to_numpy().astype(float)
            print(f"{absorption:>8} {family:>10}: recall {fmt(cluster_interval(values=hit, clusters=rows['site'].to_numpy()))}; found {found.mean():.3f}; n={rows.height}")
    rows = inj.filter((pl.col("absorption") == "fit") & (pl.col("family") == "persistent"))
    print("persistent recall by size and shape (fitted absorption):")
    print(rows.group_by("size", "shape").agg(recall=(pl.col("label") == "persistent").mean(), found=pl.col("found").mean(), survive=pl.col("survive").median(), n=pl.len()).sort("size", "shape"))
    rows = inj.filter((pl.col("absorption") == "fit") & (pl.col("family") == "temporary"))
    print("temporary recall by duration:")
    print(rows.group_by("duration_days").agg(recall=(pl.col("label") == "temporary").mean(), found=pl.col("found").mean(), survive=pl.col("survive").median(), n=pl.len()).sort("duration_days"))
    print("confusion (fitted absorption): expected x label")
    print(inj.filter((pl.col("absorption") == "fit") & pl.col("family").is_in(["persistent", "temporary", "transfer"])).group_by("family", "label").agg(n=pl.len()).sort("family", "label"))
    print("survival of the injected step in the residual, median over found steps:")
    print(inj.filter(pl.col("found") & pl.col("family").is_in(["persistent", "temporary", "transfer"])).group_by("family", "absorption").agg(survive=pl.col("survive").median(), n=pl.len()).sort("family", "absorption"))
    print("dating error of found injections (days): median absolute", float(inj.filter(pl.col("found"))["dating_error_days"].abs().median()))

    gain_files = sorted((c.EXPERIMENT_DIR / "gain").glob("*.parquet"))
    if gain_files:
        gain = pl.concat([pl.read_parquet(f) for f in gain_files], how="diagonal_relaxed")
        print("\n== Forecast gain: mean absolute error over the month after each cut, points of effective capacity ==")
        real = gain.filter(pl.col("scenario") == "real_d8")
        print("D8 (real break):")
        print(real.select("cut", "mae_none", "mae_segment", "mae_trailing", "bias_none", "offset_segment", "offset_trailing", "latest_boundary"))
        injected = gain.filter(pl.col("scenario") != "real_d8")
        key = injected["site"] + injected["scenario"] + injected["onset"]
        for name, frame, clusters in (("injected persistent steps", injected, key.to_numpy()), ("D8", real, real["cut"].to_numpy())):
            weights = frame["n_rows"].to_numpy().astype(float)
            for arm_a, arm_b in (("mae_segment", "mae_none"), ("mae_segment", "mae_trailing"), ("mae_trailing", "mae_none")):
                diff = (frame[arm_a] - frame[arm_b]).to_numpy()
                print(f"{name}: {arm_a} - {arm_b}: {fmt(cluster_interval(values=diff, clusters=clusters))}")
            print(f"{name}: mean absolute error none {frame['mae_none'].mean():.3f}, segment {frame['mae_segment'].mean():.3f}, trailing {frame['mae_trailing'].mean():.3f}")
            del weights
        print("by month after the onset (injected):")
        print(injected.group_by("month_after").agg(none=pl.col("mae_none").mean(), segment=pl.col("mae_segment").mean(), trailing=pl.col("mae_trailing").mean(), n=pl.len()).sort("month_after"))
        print("injected, by size and shape:")
        print(injected.group_by("scenario").agg(none=pl.col("mae_none").mean(), segment=pl.col("mae_segment").mean(), trailing=pl.col("mae_trailing").mean(), n=pl.len()).sort("scenario"))
    print("\n== Checks ==")
    print("median forecast minus mean forecast: zero by construction (GAM_ONLY is a point forecast).")


if __name__ == "__main__":
    main()
