"""Bootstrap intervals, the four contrasts, the real checks K1 to K4, and the verdict.

Prints every number the write-up quotes and saves the tables that the charts read to
`work/report_*.parquet` and `work/report.txt`.

    uv run python ml-experiments/00039-pv-disaggregation-cams-injection/e00039_report.py
"""

import io
from contextlib import redirect_stdout

import e00039_common as common
import numpy as np
import polars as pl

CONFIGS: dict[str, tuple[str, str]] = {
    "M1_sub": ("sub", "M1"),
    "M1d_sub": ("sub", "M1d"),
    "M1L_sub": ("sub", "M1L"),
    "M1_own": ("own", "M1"),
    "M1d_own": ("own", "M1d"),
}


def weighted_median(*, values: np.ndarray, weights: np.ndarray) -> float:
    """Return the median of `values` where each value counts `weights` times."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2.0)])


class UnitTable:
    """Headline-size scores as arrays over (Primary, farm, year) units, one column per config."""

    def __init__(self, *, injection: pl.DataFrame, k: float) -> None:
        """Pivot the scores of size `k` to one row per unit.

        Args:
            injection: `injection.parquet`.
            k: The injected size to read.
        """
        scores = injection.filter(pl.col("k") == k).with_columns(
            config=pl.col("arm") + "_" + pl.col("which")
        )
        wide = scores.pivot(
            on="config", index=["primary", "farm", "year"], values=["R", "nmae"]
        ).sort("primary", "farm", "year")
        # M1L at the farm point does not exist, so rows may be null in that column only.
        self.wide = wide
        self.primary_index = np.array(
            [common.DIRECTIONAL_PRIMARIES.index(p) for p in wide["primary"]]
        )
        self.farm_index = np.array([common.FARMS.index(f) for f in wide["farm"]])

    def column(self, *, name: str) -> np.ndarray:
        """Return one pivoted column as an array."""
        return self.wide[name].to_numpy()


def bootstrap_medians(*, table: UnitTable, columns: list[str]) -> dict[str, np.ndarray]:
    """Draw two-way cluster-bootstrap medians: Primaries and farms resampled independently.

    A resampled (Primary, farm) pair brings all of its years with it. Each draw takes 8 Primaries
    and 6 farms with replacement; a unit's weight is the product of its Primary's and farm's
    counts in the draw.

    Args:
        table: The unit table.
        columns: The pivoted column names to take medians of.

    Returns:
        For each column, the median of every draw.
    """
    rng = np.random.default_rng(common.BOOTSTRAP_SEED)
    n_primaries, n_farms = len(common.DIRECTIONAL_PRIMARIES), len(common.FARMS)
    values = {c: table.column(name=c) for c in columns}
    out = {c: np.empty(common.BOOTSTRAP_DRAWS) for c in columns}
    for draw in range(common.BOOTSTRAP_DRAWS):
        primary_counts = np.bincount(
            rng.integers(0, n_primaries, n_primaries), minlength=n_primaries
        )
        farm_counts = np.bincount(rng.integers(0, n_farms, n_farms), minlength=n_farms)
        weights = (primary_counts[table.primary_index] * farm_counts[table.farm_index]).astype(
            float
        )
        for c in columns:
            ok = (weights > 0) & ~np.isnan(values[c])
            out[c][draw] = weighted_median(values=values[c][ok], weights=weights[ok])
    return out


def interval(*, draws: np.ndarray) -> tuple[float, float]:
    """Return the 2.5th and 97.5th percentiles of the draws."""
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def contrasts(*, table: UnitTable) -> dict[str, dict[str, float]]:
    """Compute C1 to C4 and the medians of every configuration, with 95% intervals."""
    columns = [f"R_{c}" for c in CONFIGS] + ["nmae_M1_sub", "nmae_M1d_sub", "nmae_M1_own"]
    draws = bootstrap_medians(table=table, columns=columns)
    point = {c: float(np.nanmedian(table.column(name=c))) for c in columns}
    rows: dict[str, dict[str, float]] = {}
    for config in CONFIGS:
        low, high = interval(draws=draws[f"R_{config}"])
        r10, r90 = np.nanpercentile(table.column(name=f"R_{config}"), [10, 90])
        rows[config] = {
            "median": point[f"R_{config}"],
            "low": low,
            "high": high,
            "p10": float(r10),
            "p90": float(r90),
        }
    pairs = {
        "C2": ("M1d_sub", "M1_sub"),
        "C3": ("M1_own", "M1_sub"),
        "C4": ("M1L_sub", "M1_sub"),
    }
    for name, (a, b) in pairs.items():
        difference = draws[f"R_{a}"] - draws[f"R_{b}"]
        low, high = interval(draws=difference)
        rows[name] = {"median": point[f"R_{a}"] - point[f"R_{b}"], "low": low, "high": high}
    for name in ("nmae_M1_sub", "nmae_M1d_sub", "nmae_M1_own"):
        low, high = interval(draws=draws[name])
        rows[name] = {"median": point[name], "low": low, "high": high}
    return rows


def real_checks(
    *, real: pl.DataFrame, placebo: pl.DataFrame, injection: pl.DataFrame
) -> pl.DataFrame:
    """Return K1 to K4 for every directional Primary-year, with the injected expectations.

    The expectations come from the injected units (size 0.20, CAMS at the substation), so they
    are fixed by the injection run, not chosen after looking at the real result.
    """
    head = injection.filter((pl.col("k") == common.HEADLINE_SIZE) & (pl.col("which") == "sub"))
    m1 = head.filter(pl.col("arm") == "M1")
    m1l = head.filter(pl.col("arm") == "M1L")
    expected_k2 = float(
        m1.select(((pl.col("R_winter") - pl.col("R_summer")) / pl.col("R")).median()).item()
    )
    joined = m1.join(m1l, on=["primary", "farm", "year"], suffix="_l")
    expected_k3 = float(joined.select((pl.col("R_l") / pl.col("R")).median()).item())
    placebo_stats = placebo.group_by("site", "year").agg(
        pl.col("c").quantile(0.95).alias("placebo_p95"),
        pl.col("c").abs().median().alias("placebo_median_abs"),
    )
    return (
        real.filter(pl.col("role") == "directional")
        .join(placebo_stats, on=["site", "year"], how="left")
        .with_columns(
            k2_statistic=(pl.col("c_winter") - pl.col("c_summer")) / pl.col("c"),
            k3_statistic=pl.col("c_cloud") / pl.col("c"),
            k4_statistic=(pl.col("c_weekend") - pl.col("c_weekday")) / pl.col("c"),
            expected_k2=expected_k2,
            expected_k3=expected_k3,
        )
        .with_columns(
            k1=(pl.col("c") > pl.col("placebo_p95"))
            & (pl.col("placebo_median_abs") < 0.1 * pl.col("c")),
            k2=(pl.col("k2_statistic") - expected_k2).abs() <= common.K2_TOLERANCE,
            k3=(pl.col("k3_statistic") - expected_k3).abs() <= common.K3_TOLERANCE,
            k4=pl.col("k4_statistic").abs() <= common.K4_TOLERANCE,
        )
        .with_columns(all_pass=pl.col("k1") & pl.col("k2") & pl.col("k3") & pl.col("k4"))
    )


def verdict(*, rows: dict[str, dict[str, float]], checks: pl.DataFrame) -> dict[str, object]:
    """Apply the pre-registered decision rule."""
    c2 = rows["C2"]
    prefer_daily = c2["median"] >= 0.05 and c2["low"] > 0.0
    best = "M1d" if prefer_daily else "M1"
    headline = rows[f"{best}_sub"]
    nmae = rows["nmae_M1d_sub" if prefer_daily else "nmae_M1_sub"]["median"]
    lo, hi = common.RECOVERY_BAND
    ilo, ihi = common.RECOVERY_INTERVAL_LIMITS
    recovery = (
        lo <= headline["median"] <= hi
        and ilo <= headline["low"]
        and headline["high"] <= ihi
        and nmae <= common.MAX_NMAE
    )
    latest = checks.sort("year").group_by("site").last()
    n_pass = int(latest["all_pass"].sum())
    k1_all = bool(latest["k1"].all())
    real_pass = n_pass >= 6 and k1_all
    own = rows["M1d_own" if prefer_daily else "M1_own"]
    own_nmae = rows["nmae_M1_own"]["median"]
    if recovery and real_pass:
        name = "separates"
    elif recovery:
        name = "recovers-but-confounded"
    elif (headline["median"] < 0.5 and own["median"] < 0.5) or (nmae > 0.25 and own_nmae > 0.25):
        name = "fails"
    elif (0.5 <= headline["median"] < lo or hi < headline["median"] <= 1.5) and (
        headline["p90"] - headline["p10"] < 0.20
    ):
        name = "biased-correctable"
    else:
        name = "none of the four (recovery rule failed without meeting another rule)"
    return {
        "verdict": name,
        "best": best,
        "prefer_daily": prefer_daily,
        "recovery_passes": recovery,
        "real_checks_pass": real_pass,
        "n_pass_latest": n_pass,
        "k1_all": k1_all,
        "headline": headline,
        "nmae": nmae,
    }


def main() -> None:
    """Print the report and save the checks table."""
    injection = pl.read_parquet(common.INJECTION_PATH)
    real = pl.read_parquet(common.REAL_PATH)
    placebo = pl.read_parquet(common.PLACEBO_PATH)
    buffer = io.StringIO()
    with redirect_stdout(buffer), pl.Config(tbl_rows=200, tbl_width_chars=240, float_precision=3):
        table = UnitTable(injection=injection, k=common.HEADLINE_SIZE)
        print(f"units at k={common.HEADLINE_SIZE}: {table.wide.height}")
        rows = contrasts(table=table)
        print("\nMedian R over units with 95% two-way bootstrap interval and 10th/90th percentiles")
        for name, r in rows.items():
            print(name, {key: round(v, 3) for key, v in r.items()})
        print("\nMedian R by size k (all units), config")
        print(
            injection.group_by("k", "which", "arm")
            .agg(
                pl.col("R").median(),
                pl.col("nmae").median(),
                pl.col("corr").median(),
                pl.col("energy_ratio").median(),
                pl.col("se_ratio").median(),
                pl.len(),
            )
            .sort("which", "arm", "k")
        )
        print(
            "\nTruth P99 over unit rows relative to k*cap (median):",
            injection["truth_p99_ratio"].median(),
        )
        checks = real_checks(real=real, placebo=placebo, injection=injection)
        checks.write_parquet(common.WORK_DIR / "report_checks.parquet")
        print(
            "\nExpected K2 / K3 statistics from the injected units:",
            checks["expected_k2"][0],
            checks["expected_k3"][0],
        )
        print(
            checks.select(
                "site",
                "year",
                "c",
                "se",
                "placebo_p95",
                "placebo_median_abs",
                "k2_statistic",
                "k3_statistic",
                "k4_statistic",
                "k1",
                "k2",
                "k3",
                "k4",
            ).sort("site", "year")
        )
        latest = checks.sort("year").group_by("site").last().sort("site")
        print("\nLatest season-year per directional Primary")
        print(latest.select("site", "year", "c", "k1", "k2", "k3", "k4", "all_pass"))
        result = verdict(rows=rows, checks=checks)
        print("\nVERDICT:", result)
        print("\nInjection by farm (M1, sub, k=0.2), median R:")
        print(
            injection.filter((pl.col("k") == common.HEADLINE_SIZE) & (pl.col("arm") == "M1"))
            .group_by("which", "farm")
            .agg(pl.col("R").median(), pl.col("km").mean(), pl.col("nmae").median())
            .sort("which", "farm")
        )
        print("\nSummer and winter R, injected (M1, k=0.2):")
        print(
            injection.filter((pl.col("k") == common.HEADLINE_SIZE) & (pl.col("arm") == "M1"))
            .group_by("which")
            .agg(pl.col("R_summer").median(), pl.col("R_winter").median())
        )
        print("\nDetection limit: median SE of c over capacity by k (M1, sub):")
        print(
            injection.filter((pl.col("arm") == "M1") & (pl.col("which") == "sub"))
            .group_by("k")
            .agg(pl.col("se_ratio").median())
            .sort("k")
        )
        print("\nYear-on-year falls of c larger than 2 combined SE, directional Primaries:")
        yoy = (
            real.filter(pl.col("role") == "directional")
            .sort("site", "year")
            .with_columns(
                dc=pl.col("c") - pl.col("c").shift(1).over("site"),
                dse=(pl.col("se") ** 2 + pl.col("se").shift(1).over("site") ** 2).sqrt(),
                gap=pl.col("year") - pl.col("year").shift(1).over("site"),
            )
            .filter(pl.col("gap") == 1)
        )
        falls = yoy.filter(pl.col("dc") < -2 * pl.col("dse")).select("site", "year", "dc", "dse")
        print(f"{falls.height} of {yoy.height} year pairs")
        print(falls)
        print("\nMVA Primaries (descriptive): c, c_summer, c_winter")
        print(
            real.filter(pl.col("role") == "mva")
            .select("site", "year", "c", "se", "c_summer", "c_winter")
            .sort("site", "year")
        )
        print("\nBSP identity: c(BSP) vs sum of directional Primaries below it + farms below it")
        parts = {
            "D17": [
                p for p in common.DIRECTIONAL_PRIMARIES if p in ("D5", "D14", "D15", "D16", "D18")
            ],
            "D13": [p for p in common.DIRECTIONAL_PRIMARIES if p in ("D9", "D10", "D11")],
        }
        for bsp, members in parts.items():
            below = (
                real.filter(pl.col("site").is_in(members))
                .group_by("year")
                .agg(pl.col("c").sum().alias("primaries_c"), pl.len().alias("n_primaries"))
            )
            farms = (
                real.filter(pl.col("site").is_in(common.BSP_FARMS[bsp]))
                .group_by("year")
                .agg(pl.col("c").sum().alias("farms_c"))
            )
            own = real.filter(pl.col("site") == bsp).select("year", "c", "se")
            print(bsp, members)
            print(
                own.join(below, on="year", how="left")
                .join(farms, on="year", how="left")
                .sort("year")
            )
    text = buffer.getvalue()
    print(text)
    common.REPORT_PATH.write_text(text)


if __name__ == "__main__":
    main()
