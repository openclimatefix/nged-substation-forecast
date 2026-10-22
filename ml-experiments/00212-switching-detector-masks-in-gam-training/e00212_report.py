# ruff: noqa: E501, PLR0912
"""Print every number experiment 212's write-up quotes, from the stored per-row scores.

uv run --no-sync python e00212_report.py | tee printed_report.txt
"""

from __future__ import annotations

import json
import pickle

import e00212_g_common as common
import numpy as np
import polars as pl
from e00212_common import WORK_DIR

SITES = common.DECISION_SITES
RESULTS: list[dict] = []
DRAWS = 20_000
SEED = 212


def scalar(value: object) -> float:
    """Return a Polars scalar as a float."""
    return float(value)  # ty: ignore[invalid-argument-type]


def load() -> pl.DataFrame:
    """One row per (site, row) with every arm's score; C is the mean over the three folds."""
    frames = []
    for site in SITES:
        scores = pl.read_parquet(WORK_DIR / f"scores_{site}.parquet")
        wide = scores.pivot(
            on="arm", index="row", values=["qs", "mean_forecast", "q50", "q01", "q05", "q95", "q99"]
        )
        wide = wide.with_columns(
            qs_C=(pl.col("qs_C0") + pl.col("qs_C1") + pl.col("qs_C2")) / 3,
            mean_forecast_C=(
                pl.col("mean_forecast_C0") + pl.col("mean_forecast_C1") + pl.col("mean_forecast_C2")
            )
            / 3,
        )
        meta = pl.read_parquet(WORK_DIR / f"meta_{site}.parquet")
        frames.append(meta.join(wide, on="row"))
    frame = pl.concat(frames)
    return frame.with_columns(
        month=pl.col("valid_time").dt.year() * 100 + pl.col("valid_time").dt.month(),
        clean=pl.col("run_period") == "scored",
    )


def bootstrap(*, diff: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    """Mean difference and a 95% interval resampling whole groups."""
    keys, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=diff)
    counts = np.bincount(inverse).astype(float)
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(keys), size=(DRAWS, len(keys)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return scalar(diff.mean()), scalar(low), scalar(high)


def contrast(
    frame: pl.DataFrame, *, first: str, second: str, key: str, label: str, column: str = "qs"
) -> None:
    """Print `first - second` (positive means `first` is worse) with a resampled interval."""
    if len(frame) == 0:
        print(f"{label}: no rows")
        return
    diff = (frame[f"{column}_{first}"] - frame[f"{column}_{second}"]).to_numpy().astype(float)
    groups = frame[key].to_numpy()
    mean, lo, hi = bootstrap(diff=diff, groups=groups)
    width = "NOT RESOLVED" if (hi - lo) / 2 > 0.05 else ""
    RESULTS.append(
        {
            "set": label,
            "contrast": f"{first} - {second}",
            "mean": mean,
            "low": lo,
            "high": hi,
            "n": len(frame),
        }
    )
    print(
        f"{label}: {first}-{second} {mean:+.4f} ({lo:+.4f} to {hi:+.4f}) n={len(frame)} groups={len(np.unique(groups))} {width}"
    )


def positive_control() -> None:
    """Print arm B's scores against experiment 203's K0 scores on the same rows."""
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    print("== Positive control: arm B against experiment 203's K0 scores ==")
    for site in SITES:
        a = pl.read_parquet(WORK_DIR / f"scores_{site}.parquet").filter(pl.col("arm") == "B")
        b = pl.read_parquet(
            ML_EXPERIMENTS_DATA_DIR
            / "00203-switching-replay-learned-score-in-masked-forecast"
            / f"scores_{site}.parquet"
        ).filter(pl.col("arm") == "K0")
        j = a.join(b, on="row", suffix="_203")
        diff = scalar((j["qs"] - j["qs_203"]).abs().max())
        print(
            f"{site}: {len(j)} of {len(a)} rows matched, max |score difference| {diff:.3e}",
            "PASS" if diff == 0 and len(j) == len(a) else "FAIL",
        )


def main() -> None:
    """Print the report."""
    positive_control()
    frame = load()
    print(f"rows {len(frame)}; sites {SITES}")
    clean = frame.filter(pl.col("clean"))
    old = frame.filter(pl.col("post_old")).with_columns(cl=pl.col("cluster"))
    new = frame.filter(pl.col("post_new")).with_columns(
        cl=pl.col("cluster").fill_null("new"), key=pl.col("month")
    )
    print("\n== POST counts ==")
    print(
        "POST-OLD transfers (clusters):",
        old["cluster"].n_unique(),
        "rows",
        len(old),
        "months",
        old["month"].n_unique(),
    )
    print("POST-NEW rows", len(new), "months", new["month"].n_unique())
    print("\n== Mean score per arm (points) ==")
    for name, sub in (
        ("CLEAN", clean),
        ("POST-OLD", old),
        ("POST-NEW", new),
        ("ALL ordinary", frame),
    ):
        print(
            name,
            {
                arm: round(scalar(sub[f"qs_{arm}"].mean()), 4)
                for arm in ("A", "B", "C", "C0", "C1", "C2")
            },
        )
    print("\n== Headline contrasts ==")
    for first, second in (("C", "B"), ("A", "B"), ("A", "C")):
        contrast(clean, first=first, second=second, key="month", label="CLEAN")
    for first, second in (("A", "C"), ("A", "B"), ("C", "B")):
        contrast(old, first=first, second=second, key="cluster", label="POST-OLD")
    for first, second in (("A", "C"), ("A", "B"), ("C", "B")):
        contrast(new, first=first, second=second, key="month", label="POST-NEW")
    print("\n== Winter (Nov to Mar) CLEAN ==")
    winter = clean.filter(pl.col("month").mod(100).is_in([11, 12, 1, 2, 3]))
    for first, second in (("C", "B"), ("A", "B"), ("A", "C")):
        contrast(winter, first=first, second=second, key="month", label="CLEAN winter")
    print("\n== Each fold separately, CLEAN ==")
    for fold in ("C0", "C1", "C2"):
        contrast(clean, first=fold, second="B", key="month", label="CLEAN")
    print("\n== Odd and even months of the learned score's training, CLEAN, C-B ==")
    for name, parity in (("odd", 1), ("even", 0)):
        contrast(
            clean.filter(pl.col("month").mod(2) == parity),
            first="C",
            second="B",
            key="month",
            label=f"CLEAN {name}",
        )
    print(
        "\n== Mean-forecast absolute error, CLEAN (points of capacity; positive = first worse) =="
    )
    err = clean.with_columns(
        **{f"ae_{a}": (pl.col(f"mean_forecast_{a}") - pl.col("actual")).abs() for a in ("A", "B")},
        ae_C=sum((pl.col(f"mean_forecast_C{k}") - pl.col("actual")).abs() for k in range(3)) / 3,
    )
    for first, second in (("C", "B"), ("A", "B"), ("A", "C")):
        contrast(err, first=first, second=second, key="month", label="CLEAN MAE", column="ae")
    print("\n== Leave one substation out, CLEAN, C-B and A-B ==")
    for site in SITES:
        sub = clean.filter(pl.col("site") != site)
        contrast(sub, first="C", second="B", key="month", label=f"without {site}")
        contrast(sub, first="A", second="B", key="month", label=f"without {site}")
    print("\n== Leave one substation out, POST-OLD, A-C ==")
    for site in SITES:
        contrast(
            old.filter(pl.col("site") != site),
            first="A",
            second="C",
            key="cluster",
            label=f"without {site}",
        )
    print(
        "\n== Tail coverage (share of CLEAN readings below each quantile; nominal 0.01, 0.05, 0.95, 0.99) and median-mean =="
    )
    for arm in ("A", "B", "C0"):
        cov = {
            "q01": scalar((clean["actual"] < clean[f"q01_{arm}"]).mean()),
            "q05": scalar((clean["actual"] < clean[f"q05_{arm}"]).mean()),
            "q95": scalar((clean["actual"] < clean[f"q95_{arm}"]).mean()),
            "q99": scalar((clean["actual"] < clean[f"q99_{arm}"]).mean()),
        }
        print(arm, {k: round(v, 4) for k, v in cov.items()})
    for site in SITES:
        sub = clean.filter(pl.col("site") == site)
        print(
            site,
            "median-mean B",
            round(scalar((sub["q50_B"] - sub["mean_forecast_B"]).mean()), 4),
            "A",
            round(scalar((sub["q50_A"] - sub["mean_forecast_A"]).mean()), 4),
        )
    print("\n== Training rows per arm (fits at the last cut) ==")
    for site in SITES:
        counts = {
            a: pickle.loads((WORK_DIR / "fits" / a / f"{site}.pkl").read_bytes())[-1].n_rows
            for a in ("A", "B", "C0", "C1", "C2")
        }
        print(site, counts)


def worlds() -> None:
    """Print the injected-world contrasts: arm differences on CLEAN rows, net of the real record."""
    paths = sorted((WORK_DIR / "worlds").glob("scores_f*_w*_*.parquet"))
    if not paths:
        print("no worlds")
        return
    frame = pl.concat([pl.read_parquet(p) for p in paths])
    wide = frame.pivot(on="arm", index=["fold", "world", "site", "row"], values="qs").rename(
        {"A": "qs_A", "B": "qs_B", "C": "qs_C"}
    )
    real = (
        load()
        .filter(pl.col("clean"))
        .select(
            "site",
            "row",
            "month",
            real_A="qs_A",
            real_B="qs_B",
            real_C0="qs_C0",
            real_C1="qs_C1",
            real_C2="qs_C2",
        )
    )
    joined = wide.join(real, on=["site", "row"]).with_columns(
        real_C=pl.when(pl.col("fold") == 0)
        .then(pl.col("real_C0"))
        .when(pl.col("fold") == 1)
        .then(pl.col("real_C1"))
        .otherwise(pl.col("real_C2"))
    )
    joined = joined.with_columns(
        group=pl.col("fold").cast(pl.String) + "_" + pl.col("world").cast(pl.String),
    )
    info = frame.select("fold", "world", "site", "n_events", "n_found").unique()
    print("\n== Injected worlds ==")
    print(
        "world-site cells:",
        info.height,
        "events:",
        int(info["n_events"].sum()),
        "found by the re-ranker:",
        int(info["n_found"].sum()),
    )
    for name, expr in (
        (
            "A-B net of real",
            (pl.col("qs_A") - pl.col("qs_B")) - (pl.col("real_A") - pl.col("real_B")),
        ),
        (
            "A-C net of real",
            (pl.col("qs_A") - pl.col("qs_C")) - (pl.col("real_A") - pl.col("real_C")),
        ),
        (
            "C-B net of real",
            (pl.col("qs_C") - pl.col("qs_B")) - (pl.col("real_C") - pl.col("real_B")),
        ),
        ("A-B raw", pl.col("qs_A") - pl.col("qs_B")),
        ("A-C raw", pl.col("qs_A") - pl.col("qs_C")),
        ("C-B raw", pl.col("qs_C") - pl.col("qs_B")),
    ):
        diff = joined.select(d=expr)["d"].to_numpy().astype(float)
        mean, lo, hi = bootstrap(diff=diff, groups=joined["group"].to_numpy())
        RESULTS.append(
            {
                "set": "worlds",
                "contrast": name,
                "mean": mean,
                "low": lo,
                "high": hi,
                "n": len(joined),
            }
        )
        print(
            f"worlds {name}: {mean:+.4f} ({lo:+.5f} to {hi:+.5f}) cells-groups={joined['group'].n_unique()} n={len(joined)}"
        )
    print("per world cell (A-B net of real, C-B net of real):")
    cell = (
        joined.group_by("fold", "world", "site")
        .agg(
            ab=((pl.col("qs_A") - pl.col("qs_B")) - (pl.col("real_A") - pl.col("real_B"))).mean(),
            cb=((pl.col("qs_C") - pl.col("qs_B")) - (pl.col("real_C") - pl.col("real_B"))).mean(),
        )
        .join(info, on=["fold", "world", "site"])
        .sort("fold", "world", "site")
    )
    for r in cell.iter_rows(named=True):
        print(
            f"  f{r['fold']} w{r['world']} {r['site']}: events {r['n_events']} found {r['n_found']} A-B {r['ab']:+.4f} C-B {r['cb']:+.4f}"
        )


def coverage() -> None:
    """Print how much of the hand-labelled mask the detector's alarms cover at the last cut."""
    import e00212_fits as fits
    import e00212_g_members as members
    from studies.dp_features import naive_utc

    cache = common.load_cache()
    cut = members.load_fits(site=SITES[0])[-1].cut
    position = int(np.searchsorted(cache.end_times, naive_utc(moment=cut), side="right"))
    print(
        "\n== Detector alarms against the hand mask at the last cut (2026-06-01), half-hours in the 24-month window =="
    )
    window = np.zeros(len(cache.end_times), dtype=bool)
    window[max(position - 48 * 730, 0) : position] = True
    for site in SITES:
        features = cache.sites[site]
        hand = features.switching_mask & features.in_pool & ~features.lockdown & window
        shares = []
        for fold in range(3):
            spans = fits.detector_spans(cache=cache, site=site, fold=fold)
            det = (
                fits.detector_mask(spans=spans, n=len(window), cut_position=position)
                & window
                & features.in_pool
                & ~features.lockdown
            )
            shares.append(
                (
                    int(det.sum()),
                    scalar((det & hand).sum() / max(hand.sum(), 1)),
                    scalar((det & ~hand).sum() / max(det.sum(), 1)),
                )
            )
        print(
            site,
            "hand-mask half-hours",
            int(hand.sum()),
            "| per fold (detector half-hours, share of hand mask covered, share of detector outside hand mask):",
            [(d, round(a, 3), round(b, 3)) for d, a, b in shares],
        )


if __name__ == "__main__":
    main()
    worlds()
    coverage()
    (WORK_DIR / "report.json").write_text(json.dumps(RESULTS, indent=1))
