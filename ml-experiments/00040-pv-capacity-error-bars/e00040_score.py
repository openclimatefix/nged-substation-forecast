"""Score the misfit test: calibration (V1), power (V2), the bound B, real flags, and the verdict.

Every number the write-up quotes is printed to `work/report.txt`. The unit scores and the real
Primary-year scores are also written as parquet for the charts.

    POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        uv run python ml-experiments/00040-pv-capacity-error-bars/e00040_score.py
"""

import e00040_common as common
import e00040_detector as detector
import numpy as np
import polars as pl

M1D = common.ALL_FITS.index(common.ARM_M1D)
N_ARMS = len(common.ARMS)


def load_detector() -> detector.Detector:
    """Build the `Detector` from the unit, real, and bootstrap tables."""
    table = pl.read_parquet(common.UNIT_VECTORS_PATH)
    pv = table.filter(pl.col("kind") == "pv")
    units = []
    for row in pv.iter_rows(named=True):
        key = {"site": row["site"], "farm": row["farm"], "year": row["year"]}
        shapes = table.filter(
            (pl.col("site") == key["site"])
            & (pl.col("farm") == key["farm"])
            & (pl.col("year") == key["year"])
            & (pl.col("kind") != "pv")
        )
        units.append(
            detector.Unit(
                **key,
                pv=np.array([row[a] for a in common.ALL_FITS]),
                shapes={
                    r["kind"]: np.array([r[a] for a in common.ALL_FITS])
                    for r in shapes.iter_rows(named=True)
                },
            )
        )
    real_table = pl.read_parquet(common.REAL_VECTORS_PATH)
    real = {
        (r["site"], r["year"]): np.array([r[a] for a in common.ALL_FITS])
        for r in real_table.filter(pl.col("kind") == "real").iter_rows(named=True)
    }
    boot = pl.read_parquet(common.BOOTSTRAP_PATH)
    noise = {}
    for (site, year), draws in boot.group_by("site", "year", maintain_order=True):
        values = draws.sort("draw").select(common.ALL_FITS).to_numpy()
        noise[site, year] = values - values.mean(axis=0)
    return detector.Detector(units=units, real=real, noise=noise)


def weighted_median(*, values: np.ndarray, weights: np.ndarray) -> float:
    """Return the median of `values` where each value counts `weights` times."""
    keep = ~np.isnan(values) & (weights > 0)
    values, weights = values[keep], weights[keep]
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def cluster_interval(
    *, sites: np.ndarray, farms: np.ndarray, values: np.ndarray, statistic: str
) -> tuple[float, float]:
    """Return a 95% two-way cluster-bootstrap interval of a median or a mean over units.

    Primaries and farms are resampled independently, each with replacement, and a unit counts as
    often as its Primary's count times its farm's count.
    """
    rng = np.random.default_rng(common.CLUSTER_BOOTSTRAP_SEED)
    primary_index = np.array([common.DIRECTIONAL_PRIMARIES.index(site) for site in sites])
    farm_index = np.array([common.FARMS.index(farm) for farm in farms])
    out = []
    for _ in range(common.CLUSTER_BOOTSTRAP_DRAWS):
        p = np.bincount(rng.integers(0, 8, 8), minlength=8)
        f = np.bincount(rng.integers(0, 6, 6), minlength=6)
        weights = p[primary_index] * f[farm_index]
        if statistic == "median":
            out.append(weighted_median(values=values, weights=weights))
        else:
            out.append(float(np.average(values, weights=weights)))
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def scenario(
    *,
    scale: float,
    pv: np.ndarray,
    lam: np.ndarray | None,
    amplitude: float,
    noise: np.ndarray,
    ratios: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the flags and the `C / truth` of the noise draws for one injected scenario.

    The truth is `scale`. The confound adds `amplitude * scale` of M1 leakage, in the direction
    of the shape's M1 leakage.
    """
    vector = pv[:N_ARMS].copy()
    if lam is not None:
        assert lam[0] != 0.0
        vector = vector + amplitude * lam[:N_ARMS] / abs(lam[0])
    capacity, misfit = detector.capacity_and_misfit(
        vectors=scale * vector[None, :] + noise[:, :N_ARMS], ratios=ratios
    )
    return misfit > tau, capacity / scale


def unit_scores(*, det: detector.Detector) -> pl.DataFrame:
    """Return V1 and V2 rows: one per unit, shape (or `pure`), and amplitude."""
    rows = []
    for unit in det.units:
        excluded = frozenset({unit.farm})
        ratios = det.ratios(site=unit.site, excluded=excluded)
        if ratios is None:
            continue
        scale = det.capacity(site=unit.site, year=unit.year, excluded=excluded)
        tau = det.tau(site=unit.site, year=unit.year, excluded=excluded)
        noise = det.noise[unit.site, unit.year]
        cases = [("pure", 0.0, None)] + [
            (shape, a, unit.shapes[shape]) for shape in common.SHAPES for a in common.AMPLITUDES
        ]
        for shape, amplitude, lam in cases:
            flagged, ratio = scenario(
                scale=scale,
                pv=unit.pv,
                lam=lam,
                amplitude=amplitude,
                noise=noise,
                ratios=ratios,
                tau=tau,
            )
            kept = ratio[~flagged]
            rows.append(
                {
                    "site": unit.site,
                    "farm": unit.farm,
                    "year": unit.year,
                    "shape": shape,
                    "amplitude": amplitude,
                    "power": float(flagged.mean()),
                    "median_ratio": float(np.median(ratio)),
                    "n_undetected": int(kept.size),
                    "undetected_bias": float(np.median(kept) - 1.0) if kept.size else float("nan"),
                    "undetected_abs_error": float(np.median(np.abs(kept - 1.0)))
                    if kept.size
                    else float("nan"),
                }
            )
    return pl.DataFrame(rows)


def real_scores(
    *, det: detector.Detector, real_shapes: dict[tuple[str, int, str], np.ndarray]
) -> pl.DataFrame:
    """Return one row per real Primary-year: flag, capacities, bound B, and the shape range."""
    boot = pl.read_parquet(common.BOOTSTRAP_PATH)
    out = []
    for (site, year), vector in sorted(det.real.items()):
        ratios = det.ratios(site=site, excluded=frozenset())
        assert ratios is not None
        scale = det.capacity(site=site, year=year, excluded=frozenset())
        t_real = float(detector.capacity_and_misfit(vectors=vector[None, :], ratios=ratios)[1][0])
        tau = det.tau(site=site, year=year, excluded=frozenset())
        headline = vector[M1D] / ratios[M1D]
        draws = (
            boot.filter((pl.col("site") == site) & (pl.col("year") == year))[
                common.ARM_M1D
            ].to_numpy()
            / ratios[M1D]
        )
        noise = det.noise[site, year]
        units = [u for u in det.units if u.site == site]
        bound = {}
        power = {}
        for shape in common.OUT_OF_ARM_SHAPES:
            lam = real_shapes[site, year, shape]
            by_amplitude = {}
            for a in common.AMPLITUDES:
                flags, errors = [], []
                for unit in units:
                    flagged, ratio = scenario(
                        scale=scale,
                        pv=unit.pv,
                        lam=lam,
                        amplitude=a,
                        noise=noise,
                        ratios=ratios,
                        tau=tau,
                    )
                    flags.append(flagged)
                    errors.append(np.abs(ratio[~flagged] - 1.0))
                kept = np.concatenate(errors)
                by_amplitude[a] = (
                    float(np.mean(np.concatenate(flags))),
                    float(np.median(kept)) if kept.size else float("nan"),
                )
            power[shape] = by_amplitude[common.POWER_AMPLITUDE][0]
            weak = [a for a in common.AMPLITUDES if by_amplitude[a][0] < common.MIN_POWER]
            bound[shape] = max(by_amplitude[weak[-1]][1], 0.0) if weak else 0.0
        library = {}
        for shape in common.SHAPES:
            lam = real_shapes[site, year, shape][:N_ARMS]
            design = np.column_stack([ratios[:N_ARMS], lam])
            beta, *_ = np.linalg.lstsq(design, vector[:N_ARMS], rcond=None)
            library[shape] = float(beta[0])
        scaled = {s: v / scale * headline for s, v in library.items() if s != "sunload"}
        out.append(
            {
                "site": site,
                "year": year,
                "T": t_real,
                "tau": tau,
                "flagged": t_real > tau,
                "C": scale,
                "headline": float(headline),
                "headline_low": float(np.quantile(draws, 0.025)),
                "headline_high": float(np.quantile(draws, 0.975)),
                "B": float(np.nanmax(list(bound.values()))),
                **{f"B_{s}": b for s, b in bound.items()},
                **{f"power_{s}": p for s, p in power.items()},
                "range_low": min(scaled.values()),
                "range_high": max(scaled.values()),
            }
        )
    return pl.DataFrame(out)


def real_shape_vectors() -> dict[tuple[str, int, str], np.ndarray]:
    """Return the arm vector of each confound shape on each real Primary-year's own rows."""
    table = pl.read_parquet(common.REAL_VECTORS_PATH).filter(pl.col("kind") != "real")
    return {
        (r["site"], r["year"], r["kind"]): np.array([r[a] for a in common.ALL_FITS])
        for r in table.iter_rows(named=True)
    }


def verdict(
    *, v1: dict[str, float], power_by_shape: dict[str, float], clean: pl.DataFrame
) -> dict[str, object]:
    """Apply the pre-registered decision rule."""
    low, high = common.CALIBRATION_BAND
    calibration = (
        low <= v1["median_ratio"] <= high
        and v1["false_positive_rate"] <= common.MAX_FALSE_POSITIVE_RATE
    )
    power = all(p >= common.MIN_POWER for p in power_by_shape.values())
    n_clean = clean.height
    bounded = n_clean >= common.MIN_CLEAN_PRIMARIES and bool((clean["B"] <= common.MAX_BOUND).all())
    if not calibration:
        name = "miscalibrated"
    elif not power:
        name = "underpowered"
    elif bounded:
        name = "bounded"
    else:
        name = "bounded-few-clean"
    return {"verdict": name, "calibration": calibration, "power": power, "n_clean": n_clean}


def continuity() -> list[str]:
    """Return the report lines for K1 to K4 of experiment 00039 and the alternative K1."""
    checks = pl.read_parquet(common.REPORT_CHECKS_PATH)
    latest = checks.sort("year").group_by("site").last()
    alternative = checks.with_columns(
        k1_alt=(pl.col("c") > pl.col("placebo_p95"))
        & (pl.col("c") / pl.col("se") > common.K1_ALTERNATIVE_CSE)
    )
    alt_latest = alternative.sort("year").group_by("site").last()
    alt_all = alt_latest["k1_alt"] & alt_latest["k2"] & alt_latest["k3"] & alt_latest["k4"]
    lines = ["\nContinuity with experiment 00039 (latest season-year per Primary):"]
    lines.append(
        f"K1..K4 pass at {int(latest['k1'].sum())}, {int(latest['k2'].sum())}, "
        f"{int(latest['k3'].sum())}, {int(latest['k4'].sum())} of {latest.height}; "
        f"all four at {int(latest['all_pass'].sum())}."
    )
    lines.append(
        f"K1 with c / SE > {common.K1_ALTERNATIVE_CSE:g} instead of its second clause passes at "
        f"{int(alt_latest['k1_alt'].sum())} of {alt_latest.height}; all of K1(alt), K2, K3, K4 at "
        f"{int(alt_all.sum())}."
    )
    return lines


def main() -> None:
    """Score everything, write the tables, and print the report."""
    det = load_detector()
    units = unit_scores(det=det)
    units.write_parquet(common.UNIT_SCORES_PATH)
    real = real_scores(det=det, real_shapes=real_shape_vectors())
    real.write_parquet(common.REAL_SCORES_PATH)
    lines: list[str] = []

    pure = units.filter(pl.col("shape") == "pure")
    sites, farms = pure["site"].to_numpy(), pure["farm"].to_numpy()
    ratio = pure["median_ratio"].to_numpy()
    v1 = {
        "median_ratio": float(np.median(ratio)),
        "false_positive_rate": float(np.mean(pure["power"].to_numpy())),
    }
    r_low, r_high = cluster_interval(sites=sites, farms=farms, values=ratio, statistic="median")
    f_low, f_high = cluster_interval(
        sites=sites, farms=farms, values=pure["power"].to_numpy(), statistic="mean"
    )
    lines.append(
        f"V1 calibration over {pure.height} pure-PV units x {common.BOOTSTRAP_DRAWS} draws:"
    )
    lines.append(
        f"  median C / truth {v1['median_ratio']:.3f} (95% {r_low:.3f} to {r_high:.3f}); "
        "10th to 90th "
        f"percentile of units {np.quantile(ratio, 0.1):.3f} to {np.quantile(ratio, 0.9):.3f}"
    )
    lines.append(
        f"  false-positive rate {v1['false_positive_rate']:.3f} (95% {f_low:.3f} to {f_high:.3f})"
    )
    for farm, group in pure.group_by("farm", maintain_order=False):
        lines.append(
            f"  Solar farm {farm[0]}: median C / truth {group['median_ratio'].median():.3f}, "
            f"false-positive rate {group['power'].mean():.3f}"
        )

    lines.append("\nV2 power (median over units, then over the 8 Primaries' latest season-years):")
    latest = units.group_by("site").agg(pl.col("year").max().alias("latest"))
    latest_units = units.join(latest, on="site").filter(pl.col("year") == pl.col("latest"))
    per_primary = (
        latest_units.filter(pl.col("shape") != "pure")
        .group_by("site", "shape", "amplitude")
        .agg(pl.col("power").mean())
    )
    median_power = (
        per_primary.group_by("shape", "amplitude")
        .agg(
            pl.col("power").median(),
            pl.col("power").min().alias("min"),
            pl.len().alias("n_primaries"),
        )
        .sort("shape", "amplitude")
    )
    with pl.Config(tbl_rows=60, tbl_width_chars=160, float_precision=3):
        lines.append(str(median_power))
    power_by_shape = {
        s: float(
            median_power.filter(
                (pl.col("shape") == s) & (pl.col("amplitude") == common.POWER_AMPLITUDE)
            )["power"].item()
        )
        for s in common.OUT_OF_ARM_SHAPES
    }
    lines.append(
        f"power at a = {common.POWER_AMPLITUDE}: "
        + ", ".join(f"{s} {p:.2f}" for s, p in power_by_shape.items())
    )
    limit = (
        median_power.filter(pl.col("power") >= common.MIN_POWER)
        .group_by("shape")
        .agg(pl.col("amplitude").min().alias("smallest_amplitude_with_power_0_8"))
        .sort("shape")
    )
    lines.append("Post hoc: smallest grid amplitude with median power >= 0.8:\n" + str(limit))
    with pl.Config(tbl_rows=80, tbl_width_chars=160, float_precision=2):
        lines.append(
            "Power at a = 0.25 per Primary (latest season-year):\n"
            + str(
                per_primary.filter(pl.col("amplitude") == common.POWER_AMPLITUDE)
                .pivot(on="shape", index="site", values="power")
                .sort("site")
            )
        )
    blind = (
        units.filter(pl.col("shape") == "sunload")
        .group_by("amplitude")
        .agg(pl.col("power").median(), pl.col("undetected_bias").median())
        .sort("amplitude")
    )
    lines.append("Blind spot (sunload), median over units:\n" + str(blind))

    lines.append("\nReal Primary-years:")
    with pl.Config(tbl_rows=60, tbl_width_chars=220, float_precision=3, tbl_cols=30):
        lines.append(
            str(
                real.select(
                    "site",
                    "year",
                    "T",
                    "tau",
                    "flagged",
                    "C",
                    "headline",
                    "headline_low",
                    "headline_high",
                    "B",
                    "range_low",
                    "range_high",
                )
            )
        )
    latest_real = real.sort("year").group_by("site").last().sort("site")
    clean = latest_real.filter(~pl.col("flagged"))
    lines.append(
        f"\n2025/26: {clean.height} of {latest_real.height} Primaries clean: "
        f"{sorted(clean['site'].to_list())}"
    )
    lines.append(
        "Flagged Primary-years per Primary: "
        + str(
            real.filter(pl.col("flagged")).group_by("site").agg(pl.col("year").sort()).sort("site")
        )
    )
    lines.append(
        "B at the clean Primaries: "
        + ", ".join(f"{r['site']} {r['B']:.3f}" for r in clean.iter_rows(named=True))
    )
    lines.extend(continuity())
    result = verdict(v1=v1, power_by_shape=power_by_shape, clean=clean)
    lines.append(f"\nVerdict: {result}")
    text = "\n".join(lines)
    common.REPORT_PATH.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
