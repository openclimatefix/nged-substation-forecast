# ruff: noqa: E501, PLR0915, E731
"""Print every number the write-up quotes, from the stored segmentation and gain tables."""

from __future__ import annotations

import json
import pickle

import e00295_common as c
import numpy as np
import polars as pl
from e00295_segment import neighbours_of
from e00295_tune import load, site_years

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


ENGINE_FLAGS = (
    c.ML_EXPERIMENTS_DATA_DIR
    / "00199-ps2b-engine-detector-v2-ramp-night-price"
    / "flags_D8_clean_v1_2.npy"
)
GRID_START = np.datetime64("2022-01-01T00:00", "m")
SPELL_DAYS = 3
SPELL_SHARE = 0.3


def engine_spells() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the days, the daily share of flagged half-hours, and the spell-edge days."""
    flags = np.load(ENGINE_FLAGS).astype(float)
    days = (GRID_START + np.arange(len(flags)) * 30).astype("datetime64[D]")
    unique, inverse = np.unique(days, return_inverse=True)
    share = np.bincount(inverse, weights=flags) / np.bincount(inverse)
    on = share >= SPELL_SHARE
    edges = []
    start = 0
    for k in range(1, len(on) + 1):
        if k == len(on) or on[k] != on[start]:
            if k - start >= SPELL_DAYS and start > 0:
                edges.append(unique[start])
            start = k
    return unique, share, np.array(edges, dtype="datetime64[D]")


def d8_section(*, config: tuple[float, int]) -> None:
    """Print D8's boundaries, the known break, the engine spell edges, and the 2026 fade."""
    states = pickle.loads((c.EXPERIMENT_DIR / "clean_states.pkl").read_bytes())
    neighbours = neighbours_of(states=states)
    print("\n== D8 known break 2023-08-15: nearest boundary, by configuration ==")
    labelled_by_config = {}
    for (b, m), bounds in sorted(states["D8"]["bounds"].items()):
        labelled = c.label_boundaries(
            boundaries=bounds,
            neighbours=[states[n]["bounds"][(b, m)] for n in neighbours["D8"]],
            sigma=states["D8"]["sigma"],
            last_day=states["D8"]["series"].days[-1],
        )
        labelled_by_config[(b, m)] = labelled
        if not labelled:
            continue
        nearest = min(labelled, key=lambda x: abs(c.days_between(c.D8_BREAK, x.date)))
        mark = " <- chosen" if (b, m) == config else ""
        print(
            f"beta={b:>5} m={m:>2}: {nearest.date} ({nearest.label}) error {c.days_between(c.D8_BREAK, nearest.date):+.0f} days, centre shift {nearest.centre_shift:+.2f} points, n_boundaries={len(labelled)}{mark}"
        )
    days, share, edges = engine_spells()
    after = edges[edges > np.datetime64("2023-08-15")]
    print(
        f"\n== D8 engine spell edges after 2023-08-15 (runs of at least {SPELL_DAYS} days with share >= {SPELL_SHARE}): {len(after)} =="
    )
    series_days = states["D8"]["series"].days
    window = series_days[series_days > np.datetime64("2023-08-15")]
    rng = np.random.default_rng(3)
    for (b, m), labelled in labelled_by_config.items():
        if (b, m) != config and m != config[1]:
            continue
        for name, subset in (
            ("all", labelled),
            ("regime changes", [x for x in labelled if x.label != "temporary"]),
        ):
            dates = np.array(
                [x.date for x in subset if x.date > np.datetime64("2023-08-15")],
                dtype="datetime64[D]",
            )
            if not len(dates) or not len(after):
                print(f"beta={b} m={m} {name}: no boundaries after the break")
                continue
            near = lambda ds: float(
                np.mean(
                    [
                        np.min(np.abs((ds - e) / np.timedelta64(1, "D"))) <= c.MIRROR_DAYS
                        for e in after
                    ]
                )
            )
            random_share = float(
                np.mean(
                    [near(rng.choice(window, size=len(dates), replace=False)) for _ in range(500)]
                )
            )
            print(
                f"beta={b} m={m} {name}: {len(dates)} boundaries, share of edges with one within {c.MIRROR_DAYS} days {near(dates):.2f}; at random dates {random_share:.2f}"
            )
    print("\n== D8's fade in 2026 ==")
    fade = np.datetime64("2026-03-01")
    for (b, m), labelled in labelled_by_config.items():
        if m != config[1]:
            continue
        near = [x for x in labelled if abs(c.days_between(fade, x.date)) <= 60]
        print(
            f"beta={b} m={m}: "
            + "; ".join(
                f"{x.date} {x.label} centre {x.centre_shift:+.1f} dip rate {x.shift[0]:+.3f}"
                for x in near
            )
        )
    before = share[(days >= np.datetime64("2025-10-01")) & (days < np.datetime64("2026-03-01"))]
    later = share[(days >= np.datetime64("2026-03-15"))]
    print(
        f"engine detector flagged share of half-hours: 2025-10 to 2026-02 {before.mean():.3f}; from 2026-03-15 {later.mean():.3f}"
    )


def sweep_section(*, hold: list[str]) -> None:
    """Print the held-out recall and the regime changes per Primary-year at every penalty."""
    clean_all, injections = load("clean_break"), load("injection")
    years = sum(site_years()[s] for s in hold)
    print(
        "\n== Held-out sweep (7-day minimum segment): penalty, regime changes per Primary-year, recall =="
    )
    inj = injections.filter(
        pl.col("site").is_in(hold)
        & (pl.col("parity") == "even")
        & (pl.col("absorption") == "fit")
        & (pl.col("min_length") == 7)
    )
    clean = clean_all.filter(
        pl.col("site").is_in(hold) & (pl.col("min_length") == 7) & (pl.col("label") != "temporary")
    )
    for beta in c.BETAS:
        rate = clean.filter(pl.col("beta") == beta).height / years
        g = inj.filter(pl.col("beta") == beta)

        def recall(
            family: str, expected: str, size: float | None = None, g: pl.DataFrame = g
        ) -> float:
            rows = g.filter(pl.col("family") == family)
            if size is not None:
                rows = rows.filter(pl.col("size") == size)
            return float((rows["label"] == expected).to_numpy().mean())

        print(
            f"penalty {beta:>5}: {rate:.2f} per Primary-year; persistent 25% {recall('persistent', 'persistent', 0.25):.2f}, "
            f"10% {recall('persistent', 'persistent', 0.10):.2f}, temporary {recall('temporary', 'temporary'):.2f}"
        )


def main() -> None:
    """Print the report."""
    chosen = json.loads((c.EXPERIMENT_DIR / "chosen_config.json").read_text())
    beta, length = chosen["beta"], chosen["min_length"]
    tune, hold = chosen["tune_sites"], chosen["hold_sites"]
    print(
        f"chosen config: beta={beta}, min_length={length}, constraint met={chosen['constraint_met']}"
    )
    print(f"tuning Primaries: {tune}\nheld-out Primaries: {hold}")
    clean_all, injections = load("clean_break"), load("injection")
    clean = clean_all.filter(pl.col("label") != "temporary")
    print(
        f"all boundaries at held-out Primaries, chosen config: {clean_all.filter((pl.col('beta') == beta) & (pl.col('min_length') == length) & pl.col('site').is_in(hold)).height}; regime changes among them: {clean.filter((pl.col('beta') == beta) & (pl.col('min_length') == length) & pl.col('site').is_in(hold)).height}"
    )
    at = (pl.col("beta") == beta) & (pl.col("min_length") == length)

    d8_section(config=(beta, length))

    print(
        "\n== Regime changes (boundaries not labelled temporary) per Primary-year at the unmodified series =="
    )
    for name, sites in (("held-out", hold), ("tuning", tune)):
        rows = clean.filter(at & pl.col("site").is_in(sites))
        per_site = rows.group_by("site").agg(n=pl.len())
        counts = np.array(
            [per_site.filter(pl.col("site") == s)["n"].sum() for s in sites], dtype=float
        )
        years = np.array([site_years()[s] for s in sites])
        draws = np.random.default_rng(1).integers(0, len(sites), size=(DRAWS, len(sites)))
        rate = counts[draws].sum(axis=1) / years[draws].sum(axis=1)
        tail = (1 - LEVEL) / 2 * 100
        print(
            f"{name}: {counts.sum() / years.sum():.2f} per Primary-year, 98.75% interval [{np.percentile(rate, tail):.2f}, {np.percentile(rate, 100 - tail):.2f}]; per site {dict(zip(sites, counts.astype(int).tolist(), strict=True))}"
        )
    rows = clean.filter(at & pl.col("site").is_in(hold)).with_columns(
        night=pl.col("shift_centre") + pl.col("shift_shape") / 2,
        day=pl.col("shift_centre") - pl.col("shift_shape") / 2,
    )
    pv_like = rows.filter(pl.col("day").abs() >= 2 * pl.col("night").abs())
    print(
        f"PV-like (day shift at least twice night shift, approximate): {pv_like.height} of {rows.height}; April to August: {rows.filter(pl.col('date').str.slice(5, 2).cast(pl.Int32).is_between(4, 8)).height} of {rows.height}"
    )
    print(
        f"labels of held-out clean breaks: {rows.group_by('label').agg(n=pl.len()).sort('label').rows()}"
    )

    sweep_section(hold=hold)
    print("\n== Injection recall at the chosen config (held-out Primaries, even-month onsets) ==")
    inj = injections.filter(at & pl.col("site").is_in(hold) & (pl.col("parity") == "even"))
    for absorption in ("fit", "residual"):
        for family in ("persistent", "temporary"):
            rows = inj.filter((pl.col("absorption") == absorption) & (pl.col("family") == family))
            hit = (rows["label"] == rows["expected"]).to_numpy().astype(float)
            found = rows["found"].to_numpy().astype(float)
            print(
                f"{absorption:>8} {family:>10}: recall {fmt(cluster_interval(values=hit, clusters=rows['site'].to_numpy()))}; found {found.mean():.3f}; n={rows.height}"
            )
    rows = inj.filter((pl.col("absorption") == "fit") & (pl.col("family") == "persistent"))
    print("persistent recall by size and shape (fitted absorption):")
    print(
        rows.group_by("size", "shape")
        .agg(
            recall=(pl.col("label") == "persistent").mean(),
            found=pl.col("found").mean(),
            survive=pl.col("survive").median(),
            n=pl.len(),
        )
        .sort("size", "shape")
    )
    rows = inj.filter((pl.col("absorption") == "fit") & (pl.col("family") == "temporary"))
    print("temporary recall by duration:")
    print(
        rows.group_by("duration_days")
        .agg(
            recall=(pl.col("label") == "temporary").mean(),
            found=pl.col("found").mean(),
            survive=pl.col("survive").median(),
            n=pl.len(),
        )
        .sort("duration_days")
    )
    print("confusion (fitted absorption): expected x label")
    print(
        inj.filter(
            (pl.col("absorption") == "fit")
            & pl.col("family").is_in(["persistent", "temporary", "transfer"])
        )
        .group_by("family", "label")
        .agg(n=pl.len())
        .sort("family", "label")
    )
    print("survival of the injected step in the residual, median over found steps:")
    print(
        inj.filter(
            pl.col("found") & pl.col("family").is_in(["persistent", "temporary", "transfer"])
        )
        .group_by("family", "absorption")
        .agg(survive=pl.col("survive").median(), n=pl.len())
        .sort("family", "absorption")
    )
    print(
        "dating error of found injections (days; the stored error is relative to the day after the onset date, so 1 is added): median absolute",
        float(np.median(np.abs(inj.filter(pl.col("found"))["dating_error_days"].to_numpy() + 1))),
    )

    gain_files = sorted((c.EXPERIMENT_DIR / "gain").glob("*.parquet"))
    if gain_files:
        gain = pl.concat([pl.read_parquet(f) for f in gain_files], how="diagonal_relaxed")
        print(
            "\n== Forecast gain: mean absolute error over the month after each cut, points of effective capacity =="
        )
        real = gain.filter(pl.col("scenario") == "real_d8")
        print("D8 (real break):")
        print(
            real.select(
                "cut",
                "mae_none",
                "mae_segment",
                "mae_regime",
                "mae_trailing",
                "bias_none",
                "offset_segment",
                "offset_regime",
                "offset_trailing",
                "latest_boundary",
            )
        )
        injected = gain.filter(pl.col("scenario") != "real_d8")
        key = injected["site"]
        for name, frame, clusters in (
            ("injected persistent steps", injected, key.to_numpy()),
            ("D8", real, real["cut"].to_numpy()),
        ):
            weights = frame["n_rows"].to_numpy().astype(float)
            for arm_a, arm_b in (
                ("mae_segment", "mae_none"),
                ("mae_segment", "mae_trailing"),
                ("mae_regime", "mae_none"),
                ("mae_regime", "mae_trailing"),
                ("mae_trailing", "mae_none"),
            ):
                diff = (frame[arm_a] - frame[arm_b]).to_numpy()
                print(
                    f"{name}: {arm_a} - {arm_b}: {fmt(cluster_interval(values=diff, clusters=clusters))}"
                )
            print(
                f"{name}: mean absolute error none {frame['mae_none'].mean():.3f}, latest boundary {frame['mae_segment'].mean():.3f}, latest regime {frame['mae_regime'].mean():.3f}, trailing {frame['mae_trailing'].mean():.3f}"
            )
            del weights
        print("by month after the onset (injected):")
        print(
            injected.group_by("month_after")
            .agg(
                none=pl.col("mae_none").mean(),
                segment=pl.col("mae_segment").mean(),
                regime=pl.col("mae_regime").mean(),
                trailing=pl.col("mae_trailing").mean(),
                n=pl.len(),
            )
            .sort("month_after")
        )
        print("injected, by size and shape:")
        print(
            injected.group_by("scenario")
            .agg(
                none=pl.col("mae_none").mean(),
                segment=pl.col("mae_segment").mean(),
                regime=pl.col("mae_regime").mean(),
                trailing=pl.col("mae_trailing").mean(),
                n=pl.len(),
            )
            .sort("scenario")
        )
    print("\n== Checks ==")
    print(
        "median forecast minus mean forecast: zero by construction (GAM_ONLY is a point forecast)."
    )


if __name__ == "__main__":
    main()
