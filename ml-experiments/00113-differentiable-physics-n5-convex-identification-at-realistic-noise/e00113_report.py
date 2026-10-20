"""Print every number that experiment 113's page quotes into `report.txt` next to the page.

    uv run python ml-experiments/00113-*/e00113_report.py

Reads `work/summary.parquet`, `by_site.parquet`, `gate.json`, `noise_stats.parquet`,
`analytic.parquet`, `real_intervals.parquet`, `controls.json`, and the raw cell files.
"""

import json
from pathlib import Path
from typing import Final

import e00113_common as common
import e00113_summary as summary_module
import numpy as np
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
REPORT_PATH: Final[Path] = DOCS_DIR / "report.txt"


def _table(frame: pl.DataFrame, *, title: str) -> str:
    with pl.Config(tbl_rows=200, tbl_cols=40, tbl_width_chars=260, fmt_str_lengths=60):
        return f"== {title}\n{frame.with_columns(pl.col(pl.Float64).round(4))}\n"


def leave_one_site_out(*, long: pl.DataFrame) -> pl.DataFrame:
    """Return I1's gate coverages with each site left out, with site-clustered intervals.

    The label is the gate label from the point coverages. `crosses` is True where a clustered
    95% interval of either cell's coverage straddles 70% or 90% (audit trigger 4 asks whether the
    label moves, so the full-data row is the reference).
    """
    rows = []
    base = long.filter(
        (pl.col("window") == "all")
        & (pl.col("truth") == "hd")
        & (pl.col("arm") == "g_full")
        & (pl.col("method") == "i1")
    )
    for left_out in (None, *common.DECISION_SITES):
        kept = base if left_out is None else base.filter(pl.col("site") != left_out)
        for family in summary_module.FAMILIES:
            coverage = {}
            interval = {}
            for noise in ("real", "iid5"):
                part = kept.filter((pl.col("noise") == noise) & (pl.col("family") == family))
                per_draw = part.group_by(["site", "draw"]).agg(pl.col("covers_truth").mean())
                coverage[noise] = common.as_float(per_draw["covers_truth"].mean())
                interval[noise] = summary_module.clustered_interval(
                    rows=per_draw, column="covers_truth"
                )
            rows.append(
                {
                    "left_out": left_out or "none",
                    "family": family,
                    "coverage_real": coverage["real"],
                    "real_low": interval["real"][0],
                    "real_high": interval["real"][1],
                    "coverage_iid5": coverage["iid5"],
                    "iid5_low": interval["iid5"][0],
                    "iid5_high": interval["iid5"][1],
                    "label": summary_module.label(
                        coverage_decision=coverage["real"], coverage_iid=coverage["iid5"]
                    ),
                }
            )
    return pl.DataFrame(rows)


def width_ratios(*, summary: pl.DataFrame, analytic: pl.DataFrame) -> pl.DataFrame:
    """Compare G-0 over G-full widths on T-0, for I1 and I2, with sqrt(r).

    I2's width ratio is sqrt(r) by construction (both come from the same matrices), so I1's is the
    independent check.
    """
    wide = (
        summary.filter(
            (pl.col("window") == "all")
            & (pl.col("truth") == "t0")
            & pl.col("method").is_in(["i1", "i2"])
            & pl.col("arm").is_in(["g_full", "g_0"])
        )
        .select("noise", "method", "arm", "family", "median_width")
        .pivot(on="arm", index=["noise", "method", "family"], values="median_width")
    )
    r = (
        analytic.filter(
            (pl.col("window") == "all")
            & (pl.col("drift_step") == common.DRIFT_STEP_PART_2)
            & (pl.col("removed") == "g_0")
        )
        .with_columns(
            family=pl.when(pl.col("functional") == "capacity")
            .then(pl.lit("capacity"))
            .otherwise(pl.lit("band"))
        )
        .group_by("family")
        .agg(pl.col("r").median())
    )
    return (
        wide.join(r, on="family")
        .with_columns(mc_ratio=pl.col("g_0") / pl.col("g_full"), sqrt_r=pl.col("r").sqrt())
        .with_columns(difference=pl.col("mc_ratio") - pl.col("sqrt_r"))
        .sort("method", "noise", "family")
    )


def main() -> None:
    """Write `report.txt`."""
    work = common.WORK_DIR
    frame = summary_module.with_intervals(frame=summary_module.read_cells(tag="main"))
    long = summary_module.long_form(frame=frame)
    summary = pl.read_parquet(work / "summary.parquet")
    analytic = pl.read_parquet(work / "analytic.parquet")
    parts = []
    gate = json.loads((work / "gate.json").read_text())
    parts.append("== Gate (decision rule)\n" + json.dumps(gate, indent=2) + "\n")
    parts.append(
        _table(
            summary.filter(pl.col("method") == "i1")
            .select(
                "window",
                "truth",
                "noise",
                "arm",
                "family",
                "coverage_truth",
                "coverage_truth_low",
                "coverage_truth_high",
                "coverage_noise_free",
                "median_signed_relative_error",
                "rms_relative_error",
                "median_width",
                "median_relative_width",
                "median_abs_bias",
                "bias_over_width",
                "tau",
                "median_active_bound_share",
            )
            .sort("window", "truth", "noise", "arm", "family"),
            title="I1 (decision method) per cell and family",
        )
    )
    parts.append(
        _table(
            summary.filter(
                (pl.col("window") == "all")
                & (pl.col("truth") == "hd")
                & (pl.col("arm") == "g_full")
            )
            .select(
                "noise",
                "family",
                "method",
                "coverage_truth",
                "coverage_truth_low",
                "coverage_truth_high",
                "coverage_noise_free",
                "median_width",
                "median_relative_width",
            )
            .sort("noise", "family", "method"),
            title="Interval methods in the decision cell and the iid5 cell (G-full on T-HD)",
        )
    )
    parts.append(
        _table(
            pl.read_parquet(work / "by_site.parquet").filter(
                (pl.col("window") == "all")
                & (pl.col("truth") == "hd")
                & (pl.col("arm") == "g_full")
            ),
            title="I1 coverage per real site (G-full on T-HD)",
        )
    )
    parts.append(_table(leave_one_site_out(long=long), title="Leave-one-site-out gate"))
    parts.append(
        _table(
            analytic.filter(pl.col("r").is_not_nan() & (pl.col("window") == "all"))
            .group_by("drift_step", "removed", "functional")
            .agg(pl.col("r").median(), pl.col("rho").median())
            .sort("drift_step", "removed", "functional"),
            title="Absorption ratio r (median over sites) and collinearity share rho, all years",
        )
    )
    parts.append(
        _table(
            analytic.filter(pl.col("removed").str.starts_with("spectral"))
            .group_by("window", "removed", "functional")
            .agg(
                pl.col("short_period_share").median(),
                pl.col("partialled_variance_share").median(),
            )
            .sort("window", "removed", "functional"),
            title=(
                "Spectral check: share of the partialled column's variance at periods under 10 days"
            ),
        )
    )
    parts.append(
        _table(
            width_ratios(summary=summary, analytic=analytic),
            title="Monte Carlo width ratio (G-0 over G-full, T-0) against sqrt(r)",
        )
    )
    rho = (
        analytic.filter(
            (pl.col("window") == "all")
            & (pl.col("drift_step") == common.DRIFT_STEP_PART_2)
            & (pl.col("removed") == "g_0")
        )
        .with_columns(
            family=pl.when(pl.col("functional") == "capacity")
            .then(pl.lit("capacity"))
            .otherwise(pl.lit("band"))
        )
        .group_by("family")
        .agg(pl.col("rho").median())
    )
    tau = summary.filter(
        (pl.col("window") == "all")
        & (pl.col("truth") == "hd")
        & (pl.col("arm") == "g_full")
        & (pl.col("method") == "i1")
    ).select("noise", "family", "tau")
    parts.append(
        _table(
            tau.join(rho, on="family").with_columns(
                n_eff_share_of_rows=pl.col("rho") / pl.col("tau")
            ),
            title="Effective sample size as a share of the rows: n_eff / n = rho / tau",
        )
    )
    noise = pl.read_parquet(work / "noise_stats.parquet")
    parts.append(
        _table(
            noise.select(pl.all().exclude("site", "draw").mean()),
            title="Noise statistics (mean over 100 draws)",
        )
    )
    zero = summary.filter((pl.col("truth") == "zero") & (pl.col("method") == "i1")).select(
        "family", "coverage_truth", "median_width"
    )
    extra = summary_module.with_intervals(frame=summary_module.read_cells(tag="extra"))
    noise_only = (
        pl.concat(
            [
                frame.filter(pl.col("truth") == "zero").with_columns(fit=pl.lit("bounded")),
                extra.filter(pl.col("truth") == "zero").with_columns(fit=pl.lit("unbounded")),
            ]
        )
        .group_by("fit", "functional")
        .agg(
            pl.col("point").mean().alias("mean_estimate_from_noise_alone"),
            pl.col("point").abs().median().alias("median_abs_estimate_from_noise_alone"),
            pl.col("active_bound_point").mean().alias("share_fits_on_a_bound"),
        )
    )
    truth_scale = (
        frame.filter(
            (pl.col("truth") == "hd")
            & (pl.col("noise") == "real")
            & (pl.col("arm") == "g_full")
            & (pl.col("window") == "all")
        )
        .group_by("functional")
        .agg(pl.col("truth_value").mean().alias("mean_truth"))
    )
    parts.append(
        _table(
            noise_only.join(truth_scale, on="functional")
            .with_columns(
                bias_share_of_truth=pl.col("mean_estimate_from_noise_alone") / pl.col("mean_truth"),
                median_abs_share_of_truth=pl.col("median_abs_estimate_from_noise_alone")
                / pl.col("mean_truth"),
            )
            .sort("fit", "functional"),
            title=(
                "Noise-only bias: the full model fitted to the real noise alone, bounded at zero "
                "(as in the grid) and unbounded"
            ),
        )
    )
    decomposition = pl.read_parquet(work / "bias_decomposition.parquet").with_columns(
        family=pl.when(pl.col("functional") == "capacity")
        .then(pl.lit("capacity"))
        .otherwise(pl.lit("band"))
    )
    parts.append(
        _table(
            decomposition.group_by("truth", "drift_step", "family")
            .agg(
                pl.col("bias").median().alias("median_bias"),
                pl.col("bias").abs().median().alias("median_abs_bias"),
                pl.col("bias_from_drift_penalty").median().alias("median_bias_from_drift_penalty"),
                pl.col("bias_from_other_penalties").median().alias("median_bias_from_other"),
                (pl.col("bias") > 0).mean().alias("share_positive"),
            )
            .sort("truth", "drift_step", "family"),
            title="Analytic penalty bias of the full fit on the grid's truths, by penalty block",
        )
    )
    parts.append(_table(zero, title="Noise-only cell: I1 coverage of zero"))
    support = (
        frame.filter(
            (pl.col("window") == "all")
            & (pl.col("truth") == "hd")
            & (pl.col("noise") == "real")
            & (pl.col("arm") == "g_full")
        )
        .group_by("functional")
        .agg(
            pl.col("n_support_months").median(),
            pl.col("share_below_half_support").mean(),
            pl.col("i1_inflate").mean(),
        )
    )
    parts.append(
        _table(support, title="Supporting months and resamples below half of them (decision cell)")
    )
    real = pl.read_parquet(work / "real_intervals.parquet")
    parts.append(
        _table(
            real.filter((pl.col("cut") == "scored") & (pl.col("arm") == "g_full")).select(
                "site",
                "functional",
                "point",
                "i1_lo",
                "i1_hi",
                "i1u_lo",
                "i1u_hi",
                "i2_lo",
                "i2_hi",
                "vintage_season_mean",
                "tau",
                "n_eff_rows",
                "r_without_harmonics_and_drift",
            ),
            title="Real data, 2025-07-01 cut, G-full (D1 drift)",
        )
    )
    parts.append(
        _table(
            real.filter((pl.col("cut") == "scored") & (pl.col("arm") == "g_h")).select(
                "site", "functional", "point", "i1_lo", "i1_hi", "i2_lo", "i2_hi", "tau"
            ),
            title="Real data, 2025-07-01 cut, without D1 (harmonics kept)",
        )
    )
    parts.append(
        _table(
            real.filter((pl.col("cut") == "replicate") & (pl.col("arm") == "g_full")).select(
                "site", "functional", "point", "i1_lo", "i1_hi", "vintage_season_mean"
            ),
            title="Real data, replicate 2024-07-01 cut, G-full",
        )
    )
    latest = (
        real.filter(
            (pl.col("arm") == "g_full")
            & pl.col("functional").is_in(["capacity_2023", "capacity_2024"])
        )
        .with_columns(
            latest_season=pl.when(pl.col("cut") == "scored")
            .then(pl.lit("capacity_2024"))
            .otherwise(pl.lit("capacity_2023"))
        )
        .filter(pl.col("functional") == pl.col("latest_season"))
        .with_columns(
            vs_vintage=pl.col("point") / pl.col("vintage_season_mean") - 1.0,
            i1_overlaps_vintage_pm20=(pl.col("i1_hi") >= 0.8 * pl.col("vintage_season_mean"))
            & (pl.col("i1_lo") <= 1.2 * pl.col("vintage_season_mean")),
            i1_width=pl.col("i1_hi") - pl.col("i1_lo"),
        )
        .select(
            "cut",
            "site",
            "functional",
            "point",
            "vintage_season_mean",
            "vs_vintage",
            "i1_lo",
            "i1_hi",
            "i1_width",
            "i1_overlaps_vintage_pm20",
        )
        .sort("cut", "site")
    )
    parts.append(_table(latest, title="Real latest-season-year capacity against the vintage mean"))
    ratios = (
        real.filter((pl.col("cut") == "scored") & (pl.col("arm") == "g_full"))
        .with_columns(
            family=pl.when(pl.col("functional").str.starts_with("capacity"))
            .then(pl.lit("capacity"))
            .otherwise(pl.lit("band")),
            i1_over_i2_width=(pl.col("i1_hi") - pl.col("i1_lo"))
            / (pl.col("i2_hi") - pl.col("i2_lo")),
        )
        .group_by("family")
        .agg(
            pl.col("i1_over_i2_width").median().alias("median"),
            pl.col("i1_over_i2_width").min().alias("min"),
            pl.col("i1_over_i2_width").max().alias("max"),
        )
    )
    parts.append(_table(ratios, title="Real data: I1 width over I2 width (scored cut, with drift)"))
    controls = json.loads((work / "controls.json").read_text())
    control_1 = controls["control_1"]
    parts.append(
        "== Controls\n"
        f"control 1: largest relative difference {max(control_1.values()):.2e}\n"
        f"control 2: largest absolute error {controls['control_2']['max_abs_error']:.2e} "
        f"over {controls['control_2']['n_checked']} checks\n"
        f"control 3 (I2, 500 draws): bands "
        f"{controls['control_3_i2']['bands_i2_coverage_of_noise_free_value']:.3f}, "
        f"capacity {controls['control_3_i2']['capacity_i2_coverage_of_noise_free_value']:.3f}\n"
        f"control 4: {controls['control_4']}\n"
        f"control 5: all true = {all(controls['control_5'].values())}\n"
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(parts))
    print("wrote", REPORT_PATH, np.round(0, 0))


if __name__ == "__main__":
    main()
